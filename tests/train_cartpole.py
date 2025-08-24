#!/usr/bin/env python3
"""
Training script for FSMN CartPole PPO agent.
"""

import os

import lightning as L
import torch
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import TensorBoardLogger

from sisyphus.tasks import FSMNCartPolePPOModule


def main():
    """Main training function."""

    # Set random seeds for reproducibility
    L.seed_everything(42)

    # Create the Lightning module
    model = FSMNCartPolePPOModule(
        # Model parameters
        obs_dim=4,
        hidden_dim=128,
        memory_order=8,
        num_blocks=2,
        action_dim=2,
        # Environment parameters
        num_envs=4,
        max_episode_steps=500,
        # Training parameters
        frames_per_batch=2000,
        rollout_length=200,
        num_epochs=5,  # Reduced for stability
        mini_batch_size=128,  # Increased for better gradients
        # PPO parameters
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        gae_lambda=0.95,
        gamma=0.99,
        # Optimizer parameters
        lr=3e-4,
        weight_decay=1e-5,
        # Logging
        log_interval=5,
    )

    # Create directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="outputs/",
        filename="fsmn-cartpole-{epoch:02d}-{train/episode_reward_mean:.2f}",
        monitor="train/episode_reward_mean",
        mode="max",
        save_top_k=3,
        save_last=True,
        every_n_epochs=10,  # Save less frequently
    )

    early_stopping = EarlyStopping(
        monitor="train/episode_reward_mean",
        mode="max",
        patience=100,  # Increased patience
        min_delta=1.0,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Setup logger
    logger = TensorBoardLogger(
        save_dir="logs/",
        name="fsmn_cartpole_ppo",
        version=None,  # Auto-increment version
    )

    # Create trainer
    trainer = L.Trainer(
        max_epochs=1000,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        log_every_n_steps=1,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=0.5,  # Global gradient clipping
        gradient_clip_algorithm="norm",
        # Remove check_val_every_n_epoch since we don't have validation
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )

    # Print model summary
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # Start training
    print("Starting training...")
    try:
        trainer.fit(model)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise

    print("Training completed!")

    # Save final model
    final_model_path = "outputs/final_model.ckpt"
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
