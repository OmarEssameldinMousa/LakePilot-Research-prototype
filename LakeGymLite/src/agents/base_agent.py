"""
LakeGym v5 — Base Agent

Manages the observation window, feature normalization, and model inference.
CompactionAgent and PartitionAgent inherit from this.

Supports pluggable model backends via the model registry:
  'attentive_ppo' (default), 'ddqn', 'mlp_ppo'
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Tuple, Optional, Dict, Any

from agents.model_registry import BaseModel, build_model
from policies.base import Observation


class BaseAgent:
    """
    Abstract agent that wraps a model backend with:
      • A rolling observation window (deque)
      • Feature extraction & normalization from Observation
      • Action selection (greedy or stochastic)

    The model backend is selected via model_type:
      'attentive_ppo' — attention-based actor-critic (default)
      'ddqn'          — Dueling Double DQN
      'mlp_ppo'       — MLP actor-critic
    """

    def __init__(
        self,
        name: str,
        num_actions: int,
        num_features: int,
        window_size: int = 10,
        embed_dim: int = 64,
        num_heads: int = 4,
        weights_path: Optional[str] = None,
        model_type: str = "attentive_ppo",
    ):
        self.name = name
        self.num_actions = num_actions
        self.num_features = num_features
        self.window_size = window_size
        self.model_type = model_type
        self._model: Optional[BaseModel] = None
        self._window: deque = deque(maxlen=window_size)

        # Build model via registry
        self._model = build_model(
            model_type=model_type,
            num_actions=num_actions,
            num_features=num_features,
            window_size=window_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            weights_path=weights_path,
        )

    # ── Feature extraction (override in subclass) ──

    def extract_features(self, obs: Observation) -> np.ndarray:
        """
        Extract and normalize a feature vector from the full Observation.
        Must return shape (num_features,).
        Override in subclass.
        """
        raise NotImplementedError

    # ── Observation window ──

    def push_observation(self, obs: Observation) -> None:
        """Extract features and push to the rolling window."""
        features = self.extract_features(obs)
        self._window.append(features)

    def get_window(self) -> np.ndarray:
        """
        Get the observation window as (1, window_size, num_features) array.
        Zero-pads if not enough history.
        """
        if len(self._window) == 0:
            return np.zeros((1, self.window_size, self.num_features), dtype=np.float32)

        arr = np.array(list(self._window), dtype=np.float32)
        # Pad to window_size
        if arr.shape[0] < self.window_size:
            pad = np.zeros((self.window_size - arr.shape[0], self.num_features), dtype=np.float32)
            arr = np.concatenate([pad, arr], axis=0)

        return arr[np.newaxis, :, :]  # (1, window_size, num_features)

    # ── Inference ──

    def predict(self, obs: Observation) -> Tuple[int, np.ndarray, float]:
        """
        Push observation, run model, return (action_idx, probabilities, value).
        """
        import tensorflow as tf

        self.push_observation(obs)
        window = self.get_window()
        window_tf = tf.constant(window, dtype=tf.float32)
        probs, value = self._model(window_tf, training=False)
        probs_np = probs.numpy()[0]
        value_np = float(value.numpy()[0, 0])
        action_idx = int(np.argmax(probs_np))
        return action_idx, probs_np, value_np

    def predict_stochastic(self, obs: Observation) -> Tuple[int, np.ndarray, float]:
        """Same as predict() but samples from the distribution."""
        import tensorflow as tf

        self.push_observation(obs)
        window = self.get_window()
        window_tf = tf.constant(window, dtype=tf.float32)
        probs, value = self._model(window_tf, training=False)
        probs_np = probs.numpy()[0]
        value_np = float(value.numpy()[0, 0])
        action_idx = int(np.random.choice(len(probs_np), p=probs_np))
        return action_idx, probs_np, value_np

    # ── Lifecycle ──

    def reset(self) -> None:
        """Clear the observation window."""
        self._window.clear()

    def load_weights(self, path: str) -> bool:
        """Load model weights from file."""
        try:
            self._model.load_weights(path)
            return True
        except Exception as e:
            print(f"⚠️ {self.name}: failed to load weights from {path}: {e}")
            return False

    def save_weights(self, path: str) -> None:
        """Save model weights to file."""
        self._model.save_weights(path)

    def get_info(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'num_actions': self.num_actions,
            'num_features': self.num_features,
            'window_size': self.window_size,
            'window_fill': len(self._window),
        }
