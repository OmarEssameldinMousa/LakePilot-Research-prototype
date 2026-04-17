"""
LakeGym v5 — Online Specialist Training Utilities

Shared helpers for notebook-based online training of the compaction and
partition specialists against the real LakeSimulator environment.

Goals:
  • Use the exact same simulator, action mappings, feature extraction, and
    model architectures as the application.
  • Keep notebook code short and reproducible.
  • Support on-policy PPO-style updates for attentive_ppo / mlp_ppo.
  • Support off-policy DQN-style updates for ddqn.
"""

from __future__ import annotations

import json
import random
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

import numpy as np
import tensorflow as tf

from agents.compaction_agent import CompactionAgent
from agents.model_registry import build_model
from agents.partition_agent import PartitionAgent
from policies.base import (
    Action,
    COMPACT_IDX_TO_ACTION,
    PARTITION_IDX_TO_ACTION,
    Observation,
)
from simulation import LakeSimulator


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    actor_lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    rollout_steps: int = 256
    update_epochs: int = 4
    minibatch_size: int = 64
    grad_clip_norm: float = 0.5
    seed: int = 42


@dataclass
class DQNConfig:
    gamma: float = 0.99
    learning_rate: float = 1e-4
    buffer_size: int = 10000
    batch_size: int = 64
    target_update_freq: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000
    seed: int = 42


@dataclass
class EpisodeStats:
    episode: int
    reward_sum: float
    avg_reward: float
    avg_latency_ms: float
    avg_file_count: float
    avg_pruning: float
    action_counts: Dict[str, int]


class FeatureWindow:
    """Maintains a right-aligned zero-padded feature window."""

    def __init__(self, window_size: int, num_features: int):
        self.window_size = window_size
        self.num_features = num_features
        self._buffer: deque[np.ndarray] = deque(maxlen=window_size)

    def reset(self) -> None:
        self._buffer.clear()

    def push(self, features: np.ndarray) -> np.ndarray:
        self._buffer.append(features.astype(np.float32))
        window = np.zeros((self.window_size, self.num_features), dtype=np.float32)
        seq = list(self._buffer)
        window[-len(seq):] = np.stack(seq, axis=0)
        return window


class RolloutBuffer:
    def __init__(self):
        self.windows: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[float] = []

    def add(
        self,
        window: np.ndarray,
        action_idx: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        self.windows.append(window)
        self.actions.append(action_idx)
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        self.dones.append(float(done))

    def clear(self) -> None:
        self.windows.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.windows)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_plan(plan_path: str | Path) -> Dict:
    with open(plan_path, "r", encoding="utf-8") as f:
        return json.load(f)


class ReplayBuffer:
    """Simple circular replay buffer for DQN training."""

    def __init__(self, max_size: int = 10000):
        self.buffer: deque = deque(maxlen=max_size)

    def add(self, window, action_idx, reward, next_window, done):
        self.buffer.append((window, action_idx, reward, next_window, done))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        batch = [self.buffer[i] for i in indices]
        windows, actions, rewards, next_windows, dones = zip(*batch)
        return (
            np.array(windows, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_windows, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class OnlineSpecialistTrainer:
    """Online trainer for a single specialist. Supports PPO and DQN update paths."""

    def __init__(
        self,
        specialist: str,
        config: Optional[PPOConfig] = None,
        dqn_config: Optional[DQNConfig] = None,
        weights_path: Optional[str] = None,
        model_type: str = "attentive_ppo",
    ):
        if specialist not in {"compaction", "partition"}:
            raise ValueError("specialist must be 'compaction' or 'partition'")

        self.specialist = specialist
        self.model_type = model_type
        self.config = config or PPOConfig()
        self.dqn_config = dqn_config or DQNConfig()
        set_global_seeds(self.config.seed)

        # Store gamma as plain float for @tf.function compatibility
        self._gamma = float(self.config.gamma)
        self._dqn_gamma = float(self.dqn_config.gamma)

        if specialist == "compaction":
            self.agent = CompactionAgent(model_type=model_type)
            self.num_actions = self.agent.NUM_ACTIONS
            self.num_features = self.agent.NUM_FEATURES
            self.window_size = self.agent.WINDOW_SIZE
            self.reward_attr = "compact_reward"
            self.action_map = COMPACT_IDX_TO_ACTION
        else:
            self.agent = PartitionAgent(model_type=model_type)
            self.num_actions = self.agent.NUM_ACTIONS
            self.num_features = self.agent.NUM_FEATURES
            self.window_size = self.agent.WINDOW_SIZE
            self.reward_attr = "partition_reward"
            self.action_map = PARTITION_IDX_TO_ACTION

        self.model = build_model(
            model_type=model_type,
            num_actions=self.num_actions,
            num_features=self.num_features,
            window_size=self.window_size,
            weights_path=weights_path,
        )

        # Optimizer
        lr = self.dqn_config.learning_rate if model_type == "ddqn" else self.config.actor_lr
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.window = FeatureWindow(self.window_size, self.num_features)
        self.history: List[EpisodeStats] = []

        # DQN-specific: target network and replay buffer
        if model_type == "ddqn":
            self.target_model = build_model(
                model_type=model_type,
                num_actions=self.num_actions,
                num_features=self.num_features,
                window_size=self.window_size,
            )
            self.target_model.set_weights(self.model.get_weights())
            self.replay_buffer = ReplayBuffer(self.dqn_config.buffer_size)
            self._dqn_step_count = 0
            self._epsilon = self.dqn_config.epsilon_start

    def reset_episode_state(self) -> None:
        self.window.reset()
        self.agent.reset()

    def _obs_to_window(self, obs: Observation) -> np.ndarray:
        features = self.agent.extract_features(obs)
        return self.window.push(features)

    def select_action(self, window: np.ndarray, greedy: bool = False):
        probs, value = self.model(tf.constant(window[None, ...]), training=False)
        probs = probs.numpy()[0]
        value = float(value.numpy().reshape(-1)[0])

        if greedy:
            action_idx = int(np.argmax(probs))
        elif self.model_type == "ddqn" and not greedy:
            # Epsilon-greedy exploration for DDQN
            if np.random.random() < self._epsilon:
                action_idx = int(np.random.randint(self.num_actions))
            else:
                action_idx = int(np.argmax(probs))
        else:
            action_idx = int(np.random.choice(len(probs), p=probs))

        selected_prob = float(np.clip(probs[action_idx], 1e-8, 1.0))
        log_prob = float(np.log(selected_prob))
        return action_idx, self.action_map[action_idx], probs, value, log_prob

    def rollout_episode(
        self,
        sim: LakeSimulator,
        max_steps: int,
        greedy: bool = False,
    ) -> RolloutBuffer:
        buffer = RolloutBuffer()
        self.reset_episode_state()

        latencies: List[float] = []
        files: List[int] = []
        pruning: List[float] = []
        action_names: List[str] = []

        obs_dict = sim.get_observation()
        for step in range(max_steps):
            obs = Observation.from_dict(obs_dict)
            window = self._obs_to_window(obs)
            action_idx, action, _, value, log_prob = self.select_action(window, greedy=greedy)
            result = sim.run_step(action)
            done = (step == max_steps - 1)
            reward = float(getattr(result, self.reward_attr))
            buffer.add(window, action_idx, reward, value, log_prob, done)

            latencies.append(float(result.latency_ms))
            files.append(int(result.file_count))
            pruning.append(float(result.partition_pruning_ratio))
            action_names.append(result.action_taken)
            obs_dict = sim.get_observation()

        reward_sum = float(np.sum(buffer.rewards))
        stats = EpisodeStats(
            episode=len(self.history) + 1,
            reward_sum=reward_sum,
            avg_reward=reward_sum / max(len(buffer), 1),
            avg_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
            avg_file_count=float(np.mean(files)) if files else 0.0,
            avg_pruning=float(np.mean(pruning)) if pruning else 0.0,
            action_counts=dict(Counter(action_names)),
        )
        self.history.append(stats)
        return buffer

    def rollout_step(
        self,
        sim: LakeSimulator,
        obs_dict: Dict,
        greedy: bool = False,
    ) -> tuple:
        """
        Execute a single environment step. Returns
        (action_idx, reward, result, next_obs_dict, done_placeholder).
        Caller manages the RolloutBuffer and episode boundaries.
        """
        obs = Observation.from_dict(obs_dict)
        window = self._obs_to_window(obs)
        action_idx, action, probs, value, log_prob = self.select_action(window, greedy=greedy)
        result = sim.run_step(action)
        reward = float(getattr(result, self.reward_attr))
        next_obs_dict = sim.get_observation()
        return window, action_idx, reward, value, log_prob, result, next_obs_dict

    def _compute_advantages(self, rewards, values, dones):
        rewards = np.asarray(rewards, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        next_value = 0.0

        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.config.gamma * next_value * non_terminal - values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]

        advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-8)
        return advantages.astype(np.float32), returns.astype(np.float32)

    @tf.function
    def _ppo_step(self, X, a, old_logp, adv, returns):
        with tf.GradientTape() as tape:
            probs, values = self.model(X, training=True)
            values = tf.squeeze(values, axis=-1)

            a_onehot = tf.one_hot(a, depth=self.num_actions)
            chosen_probs = tf.reduce_sum(probs * a_onehot, axis=-1)
            new_logp = tf.math.log(chosen_probs + 1e-8)
            ratio = tf.exp(new_logp - old_logp)

            unclipped = ratio * adv
            clipped = tf.clip_by_value(
                ratio,
                1.0 - self.config.clip_ratio,
                1.0 + self.config.clip_ratio,
            ) * adv
            actor_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))

            critic_loss = tf.reduce_mean(tf.square(returns - values))
            entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=-1))
            loss = actor_loss + self.config.value_coef * critic_loss - self.config.entropy_coef * entropy

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.config.grad_clip_norm)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, actor_loss, critic_loss, entropy

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Dispatch to PPO or DQN update depending on model_type."""
        if self.model_type == "ddqn":
            return self._update_dqn(buffer)
        return self._update_ppo(buffer)

    def _update_ppo(self, buffer: RolloutBuffer) -> Dict[str, float]:
        X = np.asarray(buffer.windows, dtype=np.float32)
        a = np.asarray(buffer.actions, dtype=np.int32)
        old_logp = np.asarray(buffer.log_probs, dtype=np.float32)
        advantages, returns = self._compute_advantages(buffer.rewards, buffer.values, buffer.dones)

        N = len(X)
        metrics = {"loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
        batches = 0

        for _ in range(self.config.update_epochs):
            indices = np.random.permutation(N)
            for start in range(0, N, self.config.minibatch_size):
                end = min(start + self.config.minibatch_size, N)
                bi = indices[start:end]
                out = self._ppo_step(
                    tf.constant(X[bi]),
                    tf.constant(a[bi]),
                    tf.constant(old_logp[bi]),
                    tf.constant(advantages[bi]),
                    tf.constant(returns[bi]),
                )
                metrics["loss"] += float(out[0])
                metrics["actor_loss"] += float(out[1])
                metrics["critic_loss"] += float(out[2])
                metrics["entropy"] += float(out[3])
                batches += 1

        if batches > 0:
            for key in metrics:
                metrics[key] /= batches
        return metrics

    def _update_dqn(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """DQN-style update: store transitions in replay buffer, sample and train."""
        # Add transitions to replay buffer
        windows = buffer.windows
        for i in range(len(windows) - 1):
            self.replay_buffer.add(
                windows[i], buffer.actions[i], buffer.rewards[i],
                windows[i + 1], buffer.dones[i],
            )
        # Last transition (terminal)
        if len(windows) > 0:
            self.replay_buffer.add(
                windows[-1], buffer.actions[-1], buffer.rewards[-1],
                np.zeros_like(windows[-1]), 1.0,
            )

        if len(self.replay_buffer) < self.dqn_config.batch_size:
            return {"loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        total_loss = 0.0
        num_updates = max(1, len(buffer) // self.dqn_config.batch_size)

        for _ in range(num_updates):
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.dqn_config.batch_size)
            loss = self._dqn_step(
                tf.constant(states),
                tf.constant(actions),
                tf.constant(rewards),
                tf.constant(next_states),
                tf.constant(dones),
                tf.constant(self._dqn_gamma),
            )
            total_loss += float(loss)
            self._dqn_step_count += 1

            # Periodic target network sync
            if self._dqn_step_count % self.dqn_config.target_update_freq == 0:
                self.target_model.set_weights(self.model.get_weights())

        # Decay epsilon
        self._epsilon = max(
            self.dqn_config.epsilon_end,
            self._epsilon - (self.dqn_config.epsilon_start - self.dqn_config.epsilon_end) / self.dqn_config.epsilon_decay_steps,
        )

        avg_loss = total_loss / num_updates
        return {"loss": avg_loss, "actor_loss": 0.0, "critic_loss": avg_loss, "entropy": 0.0}

    @tf.function
    def _dqn_step(self, states, actions, rewards, next_states, dones, gamma):
        """Double DQN update step."""
        with tf.GradientTape() as tape:
            # Online network Q-values for current states
            q_probs, q_vals = self.model(states, training=True)

            # Get online Q-values for selected actions
            a_onehot = tf.one_hot(actions, depth=self.num_actions)
            q_selected = tf.reduce_sum(q_probs * a_onehot, axis=-1)

            # Double DQN: online net picks action, target net evaluates
            next_probs, _ = self.model(next_states, training=False)
            next_actions = tf.argmax(next_probs, axis=-1)
            target_probs, target_vals = self.target_model(next_states, training=False)
            next_a_onehot = tf.one_hot(next_actions, depth=self.num_actions)
            target_q_next = tf.reduce_sum(target_probs * next_a_onehot, axis=-1)

            # TD target: r + gamma * Q_target(s', argmax_a Q_online(s', a))
            targets = rewards + gamma * target_q_next * (1.0 - dones)
            loss = tf.reduce_mean(tf.square(targets - q_selected))

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def train(
        self,
        sim: LakeSimulator,
        num_episodes: int,
        steps_per_episode: int,
        workload_plan: Optional[Dict] = None,
        greedy_eval: bool = False,
    ) -> List[Dict[str, float]]:
        if workload_plan:
            sim.scenario_manager.load_plan(workload_plan)

        summaries: List[Dict[str, float]] = []
        for episode in range(num_episodes):
            sim.reset()
            buffer = self.rollout_episode(sim, max_steps=steps_per_episode, greedy=False)
            metrics = self.update(buffer)
            episode_stats = self.history[-1]
            summaries.append({
                "episode": episode + 1,
                "reward_sum": episode_stats.reward_sum,
                "avg_reward": episode_stats.avg_reward,
                "avg_latency_ms": episode_stats.avg_latency_ms,
                "avg_file_count": episode_stats.avg_file_count,
                "avg_pruning": episode_stats.avg_pruning,
                "loss": metrics["loss"],
                "actor_loss": metrics["actor_loss"],
                "critic_loss": metrics["critic_loss"],
                "entropy": metrics["entropy"],
            })

            if greedy_eval:
                sim.reset()
                self.rollout_episode(sim, max_steps=steps_per_episode, greedy=True)

        sim.scenario_manager.clear()
        return summaries

    def save_weights(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(out))


class OnlineCompactionTrainer(OnlineSpecialistTrainer):
    def __init__(self, config: Optional[PPOConfig] = None, weights_path: Optional[str] = None, model_type: str = "attentive_ppo"):
        super().__init__("compaction", config=config, weights_path=weights_path, model_type=model_type)


class OnlinePartitionTrainer(OnlineSpecialistTrainer):
    def __init__(self, config: Optional[PPOConfig] = None, weights_path: Optional[str] = None, model_type: str = "attentive_ppo"):
        super().__init__("partition", config=config, weights_path=weights_path, model_type=model_type)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Multi-Agent Online Trainer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MultiAgentOnlineTrainer:
    """
    Online trainer that trains BOTH compaction and partition specialists
    simultaneously through the real hierarchical meta-controller loop.

    At each step:
      1. MetaController decides delegation (compaction / partition / noop)
      2. Delegated specialist acts and gets its reward
      3. Transition is recorded in the specialist's rollout buffer

    After each episode:
      - Both specialists' buffers are used for PPO/DQN updates (if adaptive)
      - Episode stats are recorded for comparison plots
    """

    def __init__(
        self,
        model_type: str = "attentive_ppo",
        config: Optional[PPOConfig] = None,
        dqn_config: Optional[DQNConfig] = None,
        compact_weights: Optional[str] = None,
        partition_weights: Optional[str] = None,
    ):
        self.model_type = model_type
        self.config = config or PPOConfig()
        self.dqn_config = dqn_config or DQNConfig()
        set_global_seeds(self.config.seed)

        # Build specialist trainers (they handle model creation internally)
        if model_type == "ddqn":
            self.compact_trainer = OnlineSpecialistTrainer(
                "compaction", dqn_config=self.dqn_config,
                weights_path=compact_weights, model_type=model_type,
            )
            self.partition_trainer = OnlineSpecialistTrainer(
                "partition", dqn_config=self.dqn_config,
                weights_path=partition_weights, model_type=model_type,
            )
        else:
            self.compact_trainer = OnlineSpecialistTrainer(
                "compaction", config=self.config,
                weights_path=compact_weights, model_type=model_type,
            )
            self.partition_trainer = OnlineSpecialistTrainer(
                "partition", config=self.config,
                weights_path=partition_weights, model_type=model_type,
            )

        # Meta-controller (rule-based, not trained)
        from agents.meta_controller import MetaController, Delegation
        self._meta = MetaController()
        self._Delegation = Delegation

        # Episode history
        self.history: List[Dict[str, Any]] = []

    def reset_episode(self) -> None:
        """Reset all agents and meta-controller for a new episode."""
        self._meta.reset()
        self.compact_trainer.reset_episode_state()
        self.partition_trainer.reset_episode_state()

    def step(
        self, sim: LakeSimulator, obs_dict: Dict,
    ) -> Dict[str, Any]:
        """
        Execute one step in the multi-agent loop.

        Returns dict with keys:
            delegation, action_idx, reward, result, next_obs_dict,
            window, value, log_prob, specialist
        """
        obs = Observation.from_dict(obs_dict)
        decision = self._meta.decide(obs)

        if decision.delegation == self._Delegation.COMPACTION:
            trainer = self.compact_trainer
            specialist = "compaction"
        elif decision.delegation == self._Delegation.PARTITION:
            trainer = self.partition_trainer
            specialist = "partition"
        else:
            # NOOP — no specialist acts, just run a noop step
            result = sim.run_step(Action.NOOP)
            next_obs_dict = sim.get_observation()
            self._meta.notify_action(Action.NOOP)
            return {
                "delegation": "noop",
                "specialist": None,
                "action_idx": 0,
                "reward": 0.0,
                "compact_reward": float(result.compact_reward),
                "partition_reward": float(result.partition_reward),
                "global_reward": float(result.global_reward),
                "result": result,
                "next_obs_dict": next_obs_dict,
                "window": None,
                "value": 0.0,
                "log_prob": 0.0,
            }

        # Specialist acts
        window = trainer._obs_to_window(obs)
        action_idx, action, probs, value, log_prob = trainer.select_action(window)
        result = sim.run_step(action)
        reward = float(getattr(result, trainer.reward_attr))
        next_obs_dict = sim.get_observation()
        self._meta.notify_action(action)

        return {
            "delegation": decision.delegation,
            "specialist": specialist,
            "action_idx": action_idx,
            "reward": reward,
            "compact_reward": float(result.compact_reward),
            "partition_reward": float(result.partition_reward),
            "global_reward": float(result.global_reward),
            "result": result,
            "next_obs_dict": next_obs_dict,
            "window": window,
            "value": value,
            "log_prob": log_prob,
        }

    def update_models(
        self, compact_buffer: RolloutBuffer, partition_buffer: RolloutBuffer,
    ) -> Dict[str, Dict[str, float]]:
        """Update both specialists' models. Returns metrics for each."""
        compact_metrics = {"loss": 0.0, "entropy": 0.0}
        partition_metrics = {"loss": 0.0, "entropy": 0.0}

        if len(compact_buffer) > 0:
            compact_metrics = self.compact_trainer.update(compact_buffer)
        if len(partition_buffer) > 0:
            partition_metrics = self.partition_trainer.update(partition_buffer)

        return {
            "compaction": compact_metrics,
            "partition": partition_metrics,
        }

    def record_episode(self, stats: Dict[str, Any]) -> None:
        """Append episode stats to history."""
        self.history.append(stats)

    def save_weights(self, compact_path: str, partition_path: str) -> None:
        """Save both specialists' weights."""
        self.compact_trainer.save_weights(compact_path)
        self.partition_trainer.save_weights(partition_path)
