"""
LakeGym v5 — Compaction Agent Trainer (AWR)

Offline training using Advantage-Weighted Regression.
Can be run as CLI or imported for notebook use.

AWR loss:
    L_actor  = -𝔼[ exp(A / β) · log π(a|s) ]   (weighted BC)
    L_critic = 𝔼[ (V(s) - G_t)² ]               (MSE on returns)
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
    extract_compact_features,
    build_windows,
    compute_returns,
    train_val_split,
)


# ═══════════════════════════════════════════════════════════════
# AWR TRAINER
# ═══════════════════════════════════════════════════════════════

class AWRTrainer:
    """
    Advantage-Weighted Regression trainer for the compaction agent.

    Hyperparameters:
        beta:       Temperature for advantage weighting (lower → more selective)
        gamma:      Discount factor for returns
        lr:         Learning rate
        epochs:     Number of training epochs
        batch_size: Mini-batch size
    """

    def __init__(
        self,
        beta: float = 1.0,
        gamma: float = 0.99,
        lr: float = 3e-4,
        epochs: int = 50,
        batch_size: int = 64,
    ):
        self.beta = beta
        self.gamma = gamma
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        # Model
        self.model = build_and_init(
            num_actions=4,
            num_features=8,
            window_size=10,
        )
        self.optimizer = tf.keras.optimizers.Adam(lr)

    def prepare_data(self, csv_paths: list[str]):
        """Load CSVs and prepare windowed features + returns."""
        df = load_transitions(csv_paths)

        # Filter to compaction-relevant actions (NOOP + C32/C64/C128 = actions 0-3)
        mask = df['action'].isin([0, 1, 2, 3])
        df_compact = df[mask].reset_index(drop=True)
        print(f"  Compaction transitions: {len(df_compact)}")

        # Extract features
        features = extract_compact_features(df_compact, prefix="")
        windows = build_windows(features, window_size=10)

        # Actions (local index 0-3)
        actions = df_compact['action'].values.astype(np.int32)

        # Returns
        rewards = df_compact['compact_reward'].values.astype(np.float32)
        returns = compute_returns(rewards, gamma=self.gamma)

        # Train/val split
        splits = train_val_split(windows, actions, returns, val_fraction=0.15)
        self.X_train, self.X_val = splits[0]
        self.a_train, self.a_val = splits[1]
        self.G_train, self.G_val = splits[2]

        print(f"  Train: {len(self.X_train)}, Val: {len(self.X_val)}")

    @tf.function
    def _train_step(self, X_batch, a_batch, G_batch):
        with tf.GradientTape() as tape:
            probs, values = self.model(X_batch, training=True)
            values = tf.squeeze(values, axis=-1)

            # Critic loss
            critic_loss = tf.reduce_mean(tf.square(values - G_batch))

            # Advantages
            advantages = G_batch - tf.stop_gradient(values)

            # AWR weights
            weights = tf.exp(advantages / self.beta)
            weights = tf.minimum(weights, 20.0)  # clip for stability

            # Actor loss (weighted negative log-likelihood)
            a_onehot = tf.one_hot(a_batch, depth=4)
            log_probs = tf.math.log(probs + 1e-8)
            selected_log_probs = tf.reduce_sum(log_probs * a_onehot, axis=-1)
            actor_loss = -tf.reduce_mean(weights * selected_log_probs)

            loss = actor_loss + 0.5 * critic_loss

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, actor_loss, critic_loss

    def train(self) -> dict:
        """Run the training loop. Returns history dict."""
        N = len(self.X_train)
        history = {'loss': [], 'val_loss': []}

        for epoch in range(self.epochs):
            # Shuffle
            idx = np.random.permutation(N)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                bi = idx[start:end]
                X_b = tf.constant(self.X_train[bi])
                a_b = tf.constant(self.a_train[bi])
                G_b = tf.constant(self.G_train[bi])

                loss, _, _ = self._train_step(X_b, a_b, G_b)
                epoch_loss += float(loss)
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            # Validation
            val_probs, val_values = self.model(tf.constant(self.X_val), training=False)
            val_values = tf.squeeze(val_values, axis=-1)
            val_critic = float(tf.reduce_mean(tf.square(val_values - self.G_val)))

            history['loss'].append(avg_loss)
            history['val_loss'].append(val_critic)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{self.epochs} — loss: {avg_loss:.4f}, val_critic: {val_critic:.4f}")

        return history

    def save_weights(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.model.save_weights(path)
        print(f"✅ Saved compaction weights → {path}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train compaction agent (AWR)")
    parser.add_argument("--data", nargs="+", required=True, help="Transition CSV files")
    parser.add_argument("--output", default="trained_models/compaction_agent.weights.h5")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    trainer = AWRTrainer(
        beta=args.beta,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    trainer.prepare_data(args.data)
    trainer.train()
    trainer.save_weights(args.output)


if __name__ == "__main__":
    main()
