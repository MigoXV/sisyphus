from sisyphus.trainer.train import main

if __name__ == "__main__":

    from dataclasses import dataclass

    @dataclass
    class PPOConfig:
        # Experiment
        exp_name: str = "default"

        # PyTorch
        seed: int = 42
        cuda: bool = False
        torch_deterministic: bool = False

        # Distributed
        num_envs: int = 8
        share_data: bool = False
        per_rank_batch_size: int = 64

        # Environment
        env_id: str = "CartPole-v1"
        num_steps: int = 64
        capture_video: bool = True

        # PPO
        total_timesteps: int = 2**20
        learning_rate: float = 1e-3
        anneal_lr: bool = False
        gamma: float = 0.99
        gae_lambda: float = 0.95
        update_epochs: int = 10
        normalize_advantages: bool = False
        clip_coef: float = 0.2
        clip_vloss: bool = False
        ent_coef: float = 0.0
        vf_coef: float = 1.0
        max_grad_norm: float = 0.5

    args = PPOConfig()
    main(args)
