"""
LakeGym v5 — Model Registry (Strategy Pattern)

Provides a unified interface for all model backends:
  • AttentivePPOModel  — original attention-based actor-critic
  • DuelingDDQNModel   — Dueling Double DQN with softmax action distribution
  • MlpPPOModel        — MLP-based actor-critic

All models accept (batch, window_size, num_features) input and return
(action_probs, value) for a uniform API.

Usage:
    model = build_model("ddqn", num_actions=4, num_features=8, window_size=10)
    probs, value = model(tf.zeros((1, 10, 8)))
"""

from __future__ import annotations

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from abc import ABC, abstractmethod
from typing import Tuple, Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Abstract base model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BaseModel(tf.keras.Model, ABC):
    """
    Contract for all model backends.

    Every model must accept input of shape (batch, window_size, num_features)
    and return (action_probs, value):
        action_probs: (batch, num_actions) — probability distribution over actions
        value:        (batch, 1)           — scalar state-value estimate
    """

    @abstractmethod
    def call(
        self, x: tf.Tensor, training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        ...


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AttentivePPO — wrapper around existing model.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AttentivePPOModel(BaseModel):
    """
    Attention-based PPO actor-critic (from agents/model.py).

    Architecture:
        Input → Dense embedding + positional encoding
              → Multi-head self-attention + residual + LayerNorm
              → FFN + residual + LayerNorm
              → Global average pooling
              → Actor (softmax) + Critic (linear)
    """

    def __init__(
        self,
        num_actions: int,
        num_features: int,
        window_size: int = 10,
        embed_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_features = num_features
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.embedding = tf.keras.layers.Dense(
            embed_dim, activation='relu', name='embedding'
        )
        self.pos_encoding = self.add_weight(
            name='pos_encoding',
            shape=(1, window_size, embed_dim),
            initializer='glorot_uniform',
            trainable=True,
        )
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1,
            name='self_attention',
        )
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='norm1')
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='norm2')
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(embed_dim * 2, activation='relu', name='ffn_dense1'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(embed_dim, name='ffn_dense2'),
        ], name='ffn')
        self.gap = tf.keras.layers.GlobalAveragePooling1D(name='gap')
        self.actor = tf.keras.layers.Dense(num_actions, activation='softmax', name='actor')
        self.critic = tf.keras.layers.Dense(1, name='critic')

    def call(
        self, x: tf.Tensor, training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        embedded = self.embedding(x) + self.pos_encoding
        attn_output = self.attention(embedded, embedded, training=training)
        x1 = self.norm1(embedded + attn_output)
        ffn_output = self.ffn(x1, training=training)
        x2 = self.norm2(x1 + ffn_output)
        context = self.gap(x2)
        return self.actor(context), self.critic(context)

    def get_config(self):
        return {
            'num_actions': self.num_actions,
            'num_features': self.num_features,
            'window_size': self.window_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dueling DDQN — built from scratch for the environment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DuelingDDQNModel(BaseModel):
    """
    Dueling Double DQN model for data lake optimization.

    Architecture:
        Input (batch, window_size, num_features)
          → Flatten
          → Shared hidden layers (256 → 128)
          → Value stream  → scalar V(s)
          → Advantage stream → A(s,a) for each action
          → Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
          → softmax(Q) → "action probabilities" (for uniform API)
          → max(Q) → "value" (for uniform API)

    Q-values are converted to softmax probabilities so the API
    matches AttentivePPO: returns (probs, value).
    """

    def __init__(
        self,
        num_actions: int,
        num_features: int,
        window_size: int = 10,
        embed_dim: int = 64,   # unused, kept for uniform factory signature
        num_heads: int = 4,    # unused, kept for uniform factory signature
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_features = num_features
        self.window_size = window_size

        input_dim = window_size * num_features

        # Shared feature extractor
        self.flatten = tf.keras.layers.Flatten()
        self.shared_fc1 = tf.keras.layers.Dense(256, activation='relu', name='shared_fc1')
        self.shared_bn1 = tf.keras.layers.BatchNormalization(name='shared_bn1')
        self.shared_fc2 = tf.keras.layers.Dense(128, activation='relu', name='shared_fc2')
        self.shared_bn2 = tf.keras.layers.BatchNormalization(name='shared_bn2')

        # Value stream
        self.value_fc = tf.keras.layers.Dense(64, activation='relu', name='value_fc')
        self.value_out = tf.keras.layers.Dense(1, name='value_out')

        # Advantage stream
        self.advantage_fc = tf.keras.layers.Dense(64, activation='relu', name='advantage_fc')
        self.advantage_out = tf.keras.layers.Dense(num_actions, name='advantage_out')

    def call(
        self, x: tf.Tensor, training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # Shared feature extraction
        flat = self.flatten(x)
        h = self.shared_fc1(flat)
        h = self.shared_bn1(h, training=training)
        h = self.shared_fc2(h)
        h = self.shared_bn2(h, training=training)

        # Value stream
        v = self.value_fc(h)
        v = self.value_out(v)  # (batch, 1)

        # Advantage stream
        a = self.advantage_fc(h)
        a = self.advantage_out(a)  # (batch, num_actions)

        # Dueling combination: Q = V + (A - mean(A))
        q_values = v + (a - tf.reduce_mean(a, axis=-1, keepdims=True))

        # Convert Q-values to probabilities via softmax (for uniform API)
        action_probs = tf.nn.softmax(q_values, axis=-1)

        # Value estimate = max Q
        value = tf.reduce_max(q_values, axis=-1, keepdims=True)

        return action_probs, value

    def get_config(self):
        return {
            'num_actions': self.num_actions,
            'num_features': self.num_features,
            'window_size': self.window_size,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MLP PPO — MLP actor-critic built from scratch
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MlpPPOModel(BaseModel):
    """
    MLP-based PPO actor-critic for data lake optimization.

    Architecture:
        Input (batch, window_size, num_features)
          → Flatten
          → Shared hidden layers (256 → 128)
          → Actor head (softmax → action probabilities)
          → Critic head (linear → scalar value estimate)

    Non-attentive baseline — processes the entire window as a flat vector.
    """

    def __init__(
        self,
        num_actions: int,
        num_features: int,
        window_size: int = 10,
        embed_dim: int = 64,   # unused, kept for uniform factory signature
        num_heads: int = 4,    # unused, kept for uniform factory signature
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_features = num_features
        self.window_size = window_size

        # Shared trunk
        self.flatten = tf.keras.layers.Flatten()
        self.shared_fc1 = tf.keras.layers.Dense(256, activation='relu', name='shared_fc1')
        self.shared_fc2 = tf.keras.layers.Dense(128, activation='relu', name='shared_fc2')
        self.shared_dropout = tf.keras.layers.Dropout(0.1)

        # Actor head
        self.actor_fc = tf.keras.layers.Dense(64, activation='relu', name='actor_fc')
        self.actor_out = tf.keras.layers.Dense(num_actions, activation='softmax', name='actor_out')

        # Critic head
        self.critic_fc = tf.keras.layers.Dense(64, activation='relu', name='critic_fc')
        self.critic_out = tf.keras.layers.Dense(1, name='critic_out')

    def call(
        self, x: tf.Tensor, training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        flat = self.flatten(x)
        h = self.shared_fc1(flat)
        h = self.shared_fc2(h)
        h = self.shared_dropout(h, training=training)

        # Actor
        a = self.actor_fc(h)
        action_probs = self.actor_out(a)

        # Critic
        c = self.critic_fc(h)
        value = self.critic_out(c)

        return action_probs, value

    def get_config(self):
        return {
            'num_actions': self.num_actions,
            'num_features': self.num_features,
            'window_size': self.window_size,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Registry & factory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODEL_REGISTRY = {
    'attentive_ppo': AttentivePPOModel,
    'ddqn': DuelingDDQNModel,
    'mlp_ppo': MlpPPOModel,
}


def build_model(
    model_type: str,
    num_actions: int,
    num_features: int,
    window_size: int = 10,
    embed_dim: int = 64,
    num_heads: int = 4,
    weights_path: Optional[str] = None,
) -> BaseModel:
    """
    Factory: build a model from the registry, initialize weights,
    and optionally load saved weights.

    Args:
        model_type:    Key in MODEL_REGISTRY ('attentive_ppo', 'ddqn', 'mlp_ppo').
        num_actions:   Action space size.
        num_features:  Features per timestep.
        window_size:   Observation window length.
        embed_dim:     Embedding dimension (AttentivePPO-specific).
        num_heads:     Attention heads (AttentivePPO-specific).
        weights_path:  Path to .weights.h5 file (optional).

    Returns:
        Initialized model instance.

    Raises:
        ValueError: If model_type is not in the registry.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_type]
    model = model_class(
        num_actions=num_actions,
        num_features=num_features,
        window_size=window_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
    )

    # Dummy forward pass to build all layers
    dummy = tf.zeros((1, window_size, num_features))
    model(dummy, training=False)

    # Load weights if provided
    if weights_path and os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"✅ [{model_type}] Loaded weights from {weights_path}")
    elif weights_path:
        print(f"⚠️  [{model_type}] Weights file not found: {weights_path}")

    return model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Quick self-test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("🧪 Testing Model Registry ...")

    configs = [
        # (model_type, num_actions, num_features, window_size)
        ('attentive_ppo', 4, 8, 10),    # Compaction
        ('attentive_ppo', 5, 14, 20),   # Partition
        ('ddqn', 4, 8, 10),
        ('ddqn', 5, 14, 20),
        ('mlp_ppo', 4, 8, 10),
        ('mlp_ppo', 5, 14, 20),
    ]

    for model_type, na, nf, ws in configs:
        m = build_model(model_type, num_actions=na, num_features=nf, window_size=ws)
        probs, val = m(tf.random.normal((2, ws, nf)))
        assert probs.shape == (2, na), f"probs shape mismatch: {probs.shape}"
        assert val.shape == (2, 1), f"value shape mismatch: {val.shape}"
        print(f"  ✅ {model_type:15s} ({na} actions, {nf} features, win={ws}) — OK")

    # Test invalid model type
    try:
        build_model("invalid_model", 4, 8, 10)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  ✅ Invalid model type raises ValueError — OK")

    print("✅ All model registry tests passed.")
