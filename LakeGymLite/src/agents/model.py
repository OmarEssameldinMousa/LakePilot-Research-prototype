"""
LakeGym v5 — Shared Model Architecture

Single source of truth for the AttentivePPO model used by all agents.
Parameterized by num_actions, num_features, window_size, embed_dim, num_heads.

This file is the ONLY place the model architecture is defined.
Agents, trainers, and notebooks must use or replicate this exact architecture.
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

from typing import Tuple


class AttentivePPO(tf.keras.Model):
    """
    Attention-based PPO model for data lake optimization.

    Architecture:
        Input (batch, window_size, num_features)
          → Dense embedding (num_features → embed_dim) + learnable positional encoding
          → Multi-head self-attention with dropout
          → Residual connection + LayerNorm
          → Feed-forward network with dropout
          → Residual connection + LayerNorm
          → Global average pooling (temporal aggregation)
          → Actor head (softmax → action probabilities)
          → Critic head (linear → scalar value estimate)

    Args:
        num_actions:  Size of discrete action space.
        num_features: Number of input features per timestep.
        window_size:  Number of timesteps in the observation window.
        embed_dim:    Internal embedding dimension.
        num_heads:    Number of attention heads.
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

        # --- Embedding ---
        self.embedding = tf.keras.layers.Dense(embed_dim, activation='relu', name='embedding')

        # --- Learnable positional encoding ---
        self.pos_encoding = self.add_weight(
            name="pos_encoding",
            shape=(1, window_size, embed_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        # --- Multi-head self-attention ---
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=0.1,
            name='self_attention',
        )

        # --- Layer normalization ---
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='norm1')
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='norm2')

        # --- Feed-forward network ---
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(embed_dim * 2, activation='relu', name='ffn_dense1'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(embed_dim, name='ffn_dense2'),
        ], name='ffn')

        # --- Global average pooling ---
        self.gap = tf.keras.layers.GlobalAveragePooling1D(name='gap')

        # --- Actor head (policy) ---
        self.actor = tf.keras.layers.Dense(num_actions, activation='softmax', name='actor')

        # --- Critic head (value) ---
        self.critic = tf.keras.layers.Dense(1, name='critic')

    def call(
        self, x: tf.Tensor, training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, window_size, num_features).
            training: Whether in training mode (affects dropout).

        Returns:
            action_probs: (batch, num_actions)  — probability over actions.
            value:         (batch, 1)           — scalar state-value estimate.
        """
        # Embed raw features + add positional encoding
        embedded = self.embedding(x) + self.pos_encoding

        # Self-attention + residual + norm
        attn_output = self.attention(embedded, embedded, training=training)
        x1 = self.norm1(embedded + attn_output)

        # FFN + residual + norm
        ffn_output = self.ffn(x1, training=training)
        x2 = self.norm2(x1 + ffn_output)

        # Temporal aggregation
        context = self.gap(x2)

        # Heads
        action_probs = self.actor(context)
        value = self.critic(context)

        return action_probs, value

    def get_config(self):
        return {
            'num_actions': self.num_actions,
            'num_features': self.num_features,
            'window_size': self.window_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
        }


def build_and_init(
    num_actions: int,
    num_features: int,
    window_size: int = 10,
    embed_dim: int = 64,
    num_heads: int = 4,
    weights_path: str | None = None,
) -> AttentivePPO:
    """
    Factory: build model, run a dummy forward pass to initialize weights,
    and optionally load saved weights.

    Args:
        num_actions:   Action space size.
        num_features:  Features per timestep.
        window_size:   Observation window length.
        embed_dim:     Embedding dimension.
        num_heads:     Attention heads.
        weights_path:  Path to .weights.h5 file (optional).

    Returns:
        Initialized AttentivePPO model.
    """
    model = AttentivePPO(
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
        print(f"✅ Loaded weights from {weights_path}")
    elif weights_path:
        print(f"⚠️  Weights file not found: {weights_path}")

    return model


# ─────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🧪 Testing AttentivePPO model ...")

    # Compaction agent config
    m1 = build_and_init(num_actions=4, num_features=8, window_size=10)
    probs, val = m1(tf.random.normal((2, 10, 8)))
    print(f"  Compaction model  — probs: {probs.shape}, value: {val.shape}")
    assert probs.shape == (2, 4) and val.shape == (2, 1)

    # Partition agent config
    m2 = build_and_init(num_actions=5, num_features=14, window_size=20)
    probs, val = m2(tf.random.normal((2, 20, 14)))
    print(f"  Partition model   — probs: {probs.shape}, value: {val.shape}")
    assert probs.shape == (2, 5) and val.shape == (2, 1)

    print("✅ All tests passed.")
