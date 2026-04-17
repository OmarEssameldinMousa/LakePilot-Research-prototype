"""
LakeGym v5 — Partition Agent Trainer (IQL)

Offline training using Implicit Q-Learning.
Can be run as CLI or imported for notebook use.

IQL avoids querying OOD actions by:
    1. Learning V(s) via expectile regression on Q(s,a)
    2. Learning Q(s,a) via standard Bellman backup using V(s')
    3. Extracting policy via advantage-weighted regression on Q - V
"""

from __future__ import annotations

import os
import argparse
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from agents.model import build_and_init
from training.data_utils import (
    load_transitions,
    extract_partition_features,
    build_windows,
    train_val_split,
)


# ═══════════════════════════════════════════════════════════════
# Q-NETWORK (simple MLP on flattened window)
# ═══════════════════════════════════════════════════════════════

class QNetwork(tf.keras.Model):
    """Simple Q-network: (state_window, action_onehot) → Q-value."""

    def __init__(self, state_dim: int, num_actions: int = 5, hidden: int = 128):
        super().__init__()
        self.num_actions = num_actions
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden, activation='relu'),
            tf.keras.layers.Dense(hidden, activation='relu'),
            tf.keras.layers.Dense(num_actions),  # Q(s, a) for each action
        ])

    def call(self, state_flat):
        return self.net(state_flat)  # (batch, num_actions)


class ValueNetwork(tf.keras.Model):
    """V-network: state → scalar value."""

    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden, activation='relu'),
            tf.keras.layers.Dense(hidden, activation='relu'),
            tf.keras.layers.Dense(1),
        ])

    def call(self, state_flat):
        return tf.squeeze(self.net(state_flat), axis=-1)  # (batch,)


# ═══════════════════════════════════════════════════════════════
# IQL TRAINER
# ═══════════════════════════════════════════════════════════════

class IQLTrainer:
    """
    Implicit Q-Learning trainer for the partition agent.

    Two-phase training:
        Phase 1: Train Q and V networks via IQL objective
        Phase 2: Extract policy into AttentivePPO via AWR on (Q - V)
    """

    def __init__(
        self,
        gamma: float = 0.99,
        expectile: float = 0.7,
        beta: float = 3.0,
        lr_qv: float = 3e-4,
        lr_policy: float = 1e-4,
        epochs_qv: int = 30,
        epochs_policy: int = 30,
        batch_size: int = 64,
    ):
        self.gamma = gamma
        self.expectile = expectile
        self.beta = beta
        self.lr_qv = lr_qv
        self.lr_policy = lr_policy
        self.epochs_qv = epochs_qv
        self.epochs_policy = epochs_policy
        self.batch_size = batch_size

        self.window_size = 20
        self.num_features = 14
        self.num_actions = 5

    def prepare_data(self, csv_paths: list[str]):
        """Load CSVs and prepare windowed features."""
        df = load_transitions(csv_paths)

        # Map global actions → partition local index
        # NOOP=0→0, PARTITION_HOUR=4→1, PARTITION_REGION=5→2, PARTITION_EVENT_TYPE=6→3, REMOVE_PARTITION=7→4
        action_map = {0: 0, 4: 1, 5: 2, 6: 3, 7: 4}
        mask = df['action'].isin(action_map.keys())
        df_part = df[mask].reset_index(drop=True)
        print(f"  Partition-relevant transitions: {len(df_part)}")

        # Features (current obs)
        features = extract_partition_features(df_part, prefix="")
        self.windows = build_windows(features, window_size=self.window_size)

        # Next-obs features
        next_features = extract_partition_features(df_part, prefix="next_")
        self.next_windows = build_windows(next_features, window_size=self.window_size)

        # Actions (local index)
        self.actions = np.array([action_map[a] for a in df_part['action'].values], dtype=np.int32)

        # Rewards
        self.rewards = df_part['partition_reward'].values.astype(np.float32)

        # Done flags
        self.dones = df_part['done'].values.astype(np.float32)

        # Flatten windows for Q/V networks
        self.states_flat = self.windows.reshape(len(self.windows), -1).astype(np.float32)
        self.next_states_flat = self.next_windows.reshape(len(self.next_windows), -1).astype(np.float32)

        print(f"  State dim (flat): {self.states_flat.shape[1]}")

    def train_qv(self) -> dict:
        """Phase 1: Train Q and V networks."""
        state_dim = self.states_flat.shape[1]
        self.q_net = QNetwork(state_dim, self.num_actions)
        self.v_net = ValueNetwork(state_dim)

        # Build
        dummy = tf.zeros((1, state_dim))
        self.q_net(dummy)
        self.v_net(dummy)

        q_opt = tf.keras.optimizers.Adam(self.lr_qv)
        v_opt = tf.keras.optimizers.Adam(self.lr_qv)

        N = len(self.states_flat)
        history = {'q_loss': [], 'v_loss': []}

        for epoch in range(self.epochs_qv):
            idx = np.random.permutation(N)
            q_epoch_loss, v_epoch_loss = 0.0, 0.0
            n_batches = 0

            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                bi = idx[start:end]

                s = tf.constant(self.states_flat[bi])
                a = tf.constant(self.actions[bi])
                r = tf.constant(self.rewards[bi])
                ns = tf.constant(self.next_states_flat[bi])
                d = tf.constant(self.dones[bi])

                # ── Q update ──
                with tf.GradientTape() as tape:
                    # Target: r + γ V(s')
                    next_v = tf.stop_gradient(self.v_net(ns))
                    target = r + self.gamma * (1 - d) * next_v

                    q_all = self.q_net(s)  # (batch, num_actions)
                    a_onehot = tf.one_hot(a, self.num_actions)
                    q_sa = tf.reduce_sum(q_all * a_onehot, axis=-1)
                    q_loss = tf.reduce_mean(tf.square(q_sa - target))

                q_grads = tape.gradient(q_loss, self.q_net.trainable_variables)
                q_opt.apply_gradients(zip(q_grads, self.q_net.trainable_variables))

                # ── V update (expectile regression) ──
                with tf.GradientTape() as tape:
                    v = self.v_net(s)
                    q_detached = tf.stop_gradient(q_sa)
                    diff = q_detached - v
                    weight = tf.where(diff > 0, self.expectile, 1 - self.expectile)
                    v_loss = tf.reduce_mean(weight * tf.square(diff))

                v_grads = tape.gradient(v_loss, self.v_net.trainable_variables)
                v_opt.apply_gradients(zip(v_grads, self.v_net.trainable_variables))

                q_epoch_loss += float(q_loss)
                v_epoch_loss += float(v_loss)
                n_batches += 1

            history['q_loss'].append(q_epoch_loss / max(n_batches, 1))
            history['v_loss'].append(v_epoch_loss / max(n_batches, 1))

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  QV Epoch {epoch+1:3d}/{self.epochs_qv} — "
                      f"q_loss: {history['q_loss'][-1]:.4f}, v_loss: {history['v_loss'][-1]:.4f}")

        return history

    def train_policy(self) -> dict:
        """Phase 2: Extract policy into AttentivePPO via AWR on (Q - V)."""
        self.policy_model = build_and_init(
            num_actions=self.num_actions,
            num_features=self.num_features,
            window_size=self.window_size,
        )
        policy_opt = tf.keras.optimizers.Adam(self.lr_policy)

        # Compute advantages: Q(s,a) - V(s) for all data
        q_all = self.q_net(tf.constant(self.states_flat)).numpy()
        v_all = self.v_net(tf.constant(self.states_flat)).numpy()
        a_onehot = np.eye(self.num_actions)[self.actions]
        q_sa = np.sum(q_all * a_onehot, axis=-1)
        advantages = q_sa - v_all

        N = len(self.windows)
        history = {'policy_loss': []}

        for epoch in range(self.epochs_policy):
            idx = np.random.permutation(N)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                bi = idx[start:end]

                w_b = tf.constant(self.windows[bi])
                a_b = tf.constant(self.actions[bi])
                adv_b = tf.constant(advantages[bi])

                with tf.GradientTape() as tape:
                    probs, values = self.policy_model(w_b, training=True)

                    # AWR
                    weights = tf.exp(adv_b / self.beta)
                    weights = tf.minimum(weights, 20.0)

                    a_oh = tf.one_hot(a_b, self.num_actions)
                    log_probs = tf.math.log(probs + 1e-8)
                    selected_lp = tf.reduce_sum(log_probs * a_oh, axis=-1)
                    actor_loss = -tf.reduce_mean(weights * selected_lp)

                    # Also train critic on returns
                    returns = tf.constant(advantages[bi] + v_all[bi])  # G ≈ A + V
                    critic_loss = tf.reduce_mean(tf.square(tf.squeeze(values, -1) - returns))

                    loss = actor_loss + 0.5 * critic_loss

                grads = tape.gradient(loss, self.policy_model.trainable_variables)
                policy_opt.apply_gradients(zip(grads, self.policy_model.trainable_variables))

                epoch_loss += float(loss)
                n_batches += 1

            history['policy_loss'].append(epoch_loss / max(n_batches, 1))
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Policy Epoch {epoch+1:3d}/{self.epochs_policy} — "
                      f"loss: {history['policy_loss'][-1]:.4f}")

        return history

    def save_weights(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.policy_model.save_weights(path)
        print(f"✅ Saved partition policy weights → {path}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train partition agent (IQL)")
    parser.add_argument("--data", nargs="+", required=True, help="Transition CSV files")
    parser.add_argument("--output", default="trained_models/partition_agent.weights.h5")
    parser.add_argument("--epochs-qv", type=int, default=30)
    parser.add_argument("--epochs-policy", type=int, default=30)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--lr-qv", type=float, default=3e-4)
    parser.add_argument("--lr-policy", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    trainer = IQLTrainer(
        beta=args.beta,
        expectile=args.expectile,
        lr_qv=args.lr_qv,
        lr_policy=args.lr_policy,
        epochs_qv=args.epochs_qv,
        epochs_policy=args.epochs_policy,
        batch_size=args.batch_size,
    )
    trainer.prepare_data(args.data)
    print("\n── Phase 1: Training Q/V networks ──")
    trainer.train_qv()
    print("\n── Phase 2: Extracting policy ──")
    trainer.train_policy()
    trainer.save_weights(args.output)


if __name__ == "__main__":
    main()
