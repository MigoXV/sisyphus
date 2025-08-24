#!/usr/bin/env python3
"""
Simple test script to verify the FSMN CartPole PPO implementation.
"""

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from sisyphus.models.actor_critic import ActorCriticWrapper
from sisyphus.criterions.ppo import PPOLoss


def test_actor_critic():
    """Test the ActorCritic wrapper."""
    print("Testing ActorCritic wrapper...")
    
    # Create model
    model = ActorCriticWrapper(
        obs_dim=4,
        hidden_dim=64,
        memory_order=4,
        num_blocks=2,
        action_dim=2,
    )
    
    # Test forward pass
    batch_size = 8
    seq_len = 10
    obs = torch.randn(batch_size, seq_len, 4)
    
    output = model(obs)
    print(f"Output keys: {output.keys()}")
    print(f"Logits shape: {output['logits'].shape}")
    print(f"State value shape: {output['state_value'].shape}")
    
    # Test single step
    obs_single = torch.randn(batch_size, 4)
    logits, values, caches = model.get_action_value(obs_single)
    print(f"Single step logits shape: {logits.shape}")
    print(f"Single step values shape: {values.shape}")
    
    # Test with caches
    device = obs_single.device
    caches = model.allocate_caches(batch_size, device)
    logits, values, new_caches = model.get_action_value(obs_single, caches)
    print(f"With caches logits shape: {logits.shape}")
    print(f"Number of caches: {len(new_caches)}")
    
    print("ActorCritic test passed!\n")


def test_ppo_loss():
    """Test the PPO loss function."""
    print("Testing PPO loss...")
    
    # Create loss function
    ppo_loss = PPOLoss(
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
    )
    
    # Create dummy data
    batch_size = 32
    seq_len = 10
    action_dim = 2
    
    logits = torch.randn(batch_size, seq_len, action_dim)
    old_logits = torch.randn(batch_size, seq_len, action_dim)
    actions = torch.randint(0, action_dim, (batch_size, seq_len))
    values = torch.randn(batch_size, seq_len)
    returns = torch.randn(batch_size, seq_len)
    advantages = torch.randn(batch_size, seq_len)
    
    # Compute loss
    loss_dict = ppo_loss(
        logits=logits,
        old_logits=old_logits,
        actions=actions,
        values=values,
        returns=returns,
        advantages=advantages,
    )
    
    print(f"Loss components: {list(loss_dict.keys())}")
    print(f"Total loss: {loss_dict['loss'].item():.4f}")
    print(f"Policy loss: {loss_dict['policy_loss'].item():.4f}")
    print(f"Value loss: {loss_dict['value_loss'].item():.4f}")
    print(f"Entropy: {loss_dict['entropy'].item():.4f}")
    print(f"Approx KL: {loss_dict['approx_kl'].item():.4f}")
    print(f"Clip fraction: {loss_dict['clipfrac'].item():.4f}")
    
    print("PPO loss test passed!\n")


def test_integration():
    """Test integration between components."""
    print("Testing integration...")
    
    # Create components
    model = ActorCriticWrapper(obs_dim=4, hidden_dim=32, action_dim=2)
    ppo_loss = PPOLoss()
    
    # Simulate training step
    batch_size = 16
    seq_len = 5
    obs = torch.randn(batch_size, seq_len, 4)
    
    # Forward pass
    with torch.no_grad():
        output = model(obs)
        old_logits = output['logits'].clone()
        old_values = output['state_value'].clone()
    
    # Current forward pass
    output = model(obs)
    current_logits = output['logits']
    current_values = output['state_value']
    
    # Sample actions from old policy
    old_dist = torch.distributions.Categorical(logits=old_logits)
    actions = old_dist.sample()
    
    # Create dummy returns and advantages
    returns = torch.randn(batch_size, seq_len)
    advantages = torch.randn(batch_size, seq_len)
    
    # Compute loss
    loss_dict = ppo_loss(
        logits=current_logits,
        old_logits=old_logits,
        actions=actions,
        values=current_values,
        returns=returns,
        advantages=advantages,
    )
    
    # Backward pass
    loss = loss_dict['loss']
    loss.backward()
    
    print(f"Integration test completed. Loss: {loss.item():.4f}")
    print("Integration test passed!\n")


if __name__ == "__main__":
    print("Running FSMN CartPole PPO tests...\n")
    
    test_actor_critic()
    test_ppo_loss()
    test_integration()
    
    print("All tests passed! 🎉")
