#!/usr/bin/env python3
"""
Evaluation script for trained FSMN CartPole PPO agent.
"""

import argparse
import torch
import numpy as np
import gymnasium as gym
from sisyphus.tasks import FSMNCartPolePPOModule


def evaluate_model(checkpoint_path: str, num_episodes: int = 100, render: bool = False):
    """
    Evaluate a trained model.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        num_episodes: Number of episodes to run
        render: Whether to render the environment
    """
    # Load the model
    print(f"Loading model from {checkpoint_path}")
    model = FSMNCartPolePPOModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Create environment
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    
    # Initialize caches for the model
    device = next(model.parameters()).device
    caches = model.actor_critic.allocate_caches(1, device)
    
    episode_rewards = []
    episode_lengths = []
    
    print(f"Running {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Reset caches for new episode
        for cache in caches:
            cache.zero_()
        
        while not done:
            # Convert observation to tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Get action from model
            with torch.no_grad():
                logits, values, caches = model.actor_critic.get_action_value(obs_tensor, caches)
                
                # Sample action
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                action = action.cpu().numpy()[0]
            
            # Take step in environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if (episode + 1) % 10 == 0:
            mean_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{num_episodes}, Last 10 avg reward: {mean_reward:.2f}")
    
    env.close()
    
    # Print statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    
    print(f"\nEvaluation Results:")
    print(f"Episodes: {num_episodes}")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean length: {mean_length:.2f} ± {std_length:.2f}")
    print(f"Max reward: {max(episode_rewards):.2f}")
    print(f"Min reward: {min(episode_rewards):.2f}")
    print(f"Success rate (reward >= 475): {sum(r >= 475 for r in episode_rewards) / num_episodes * 100:.1f}%")
    
    return episode_rewards, episode_lengths


def main():
    parser = argparse.ArgumentParser(description="Evaluate FSMN CartPole PPO agent")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to run")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    
    args = parser.parse_args()
    
    try:
        evaluate_model(args.checkpoint, args.episodes, args.render)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
