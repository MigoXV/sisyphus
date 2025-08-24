# FSMN CartPole PPO Implementation

这个项目实现了使用FSMN (Feedforward Sequential Memory Network) 作为策略网络的PPO (Proximal Policy Optimization) 算法来训练CartPole-v1环境的强化学习代理。

## 特性

- **FSMN架构**: 使用因果FSMN块构建的策略网络，具有序列记忆能力
- **Actor-Critic设计**: 使用组合方式封装FSMN，支持actor和critic功能
- **PPO算法**: 自定义实现的PPO损失函数，支持完整的PPO训练流程
- **TorchRL集成**: 使用TorchRL的collector替代传统dataloader进行数据收集
- **Lightning框架**: 基于PyTorch Lightning的训练框架，支持自动日志记录和检查点

## 项目结构

```
src/sisyphus/
├── models/
│   ├── fsmn.py           # FSMN模型实现
│   └── actor_critic.py   # Actor-Critic封装
├── criterions/
│   └── ppo.py           # PPO损失函数
└── tasks/
    └── fsmn.py          # Lightning训练模块
```

## 核心组件

### 1. FSMN Policy (`models/fsmn.py`)
- `CausalFSMNBlock`: 因果FSMN块，支持序列和单步推理
- `FSMNPolicy`: 完整的FSMN策略网络，输出action logits和state values
- `fsmn_allocate_caches`: 缓存管理功能，支持有状态推理

### 2. Actor-Critic Wrapper (`models/actor_critic.py`)
- `ActorCriticWrapper`: 使用组合方式封装FSMN策略
- 支持独立的actor和critic功能
- 提供缓存管理和分布式采样接口

### 3. PPO Loss (`criterions/ppo.py`)
- `PPOLoss`: 完整的PPO损失实现
- 支持policy loss、value loss和entropy bonus
- 包含KL散度和clip fraction监控

### 4. Lightning Module (`tasks/fsmn.py`)
- `FSMNCartPolePPOModule`: 完整的训练流程
- TorchRL collector集成
- GAE (Generalized Advantage Estimation) 实现
- 自动优化和梯度裁剪

## 安装依赖

```bash
# 安装基础依赖
pip install torch torchrl lightning gymnasium tensordict

# 或者使用poetry (推荐)
poetry install
```

## 使用方法

### 训练模型

```bash
python train_cartpole.py
```

训练参数可以在`train_cartpole.py`中修改：

```python
model = FSMNCartPolePPOModule(
    # 模型参数
    obs_dim=4,           # 观察空间维度
    hidden_dim=128,      # 隐藏层维度
    memory_order=8,      # FSMN记忆阶数
    num_blocks=2,        # FSMN块数量
    action_dim=2,        # 动作空间维度
    
    # 环境参数
    num_envs=4,          # 并行环境数量
    max_episode_steps=500,
    
    # 训练参数
    frames_per_batch=2000,
    rollout_length=200,
    num_epochs=5,
    mini_batch_size=128,
    
    # PPO参数
    clip_epsilon=0.2,
    value_loss_coef=0.5,
    entropy_coef=0.01,
    gae_lambda=0.95,
    gamma=0.99,
    
    # 优化器参数
    lr=3e-4,
    weight_decay=1e-5,
)
```

### 评估模型

```bash
python evaluate_cartpole.py checkpoints/final_model.ckpt --episodes 100 --render
```

### 测试实现

```bash
python test_implementation.py
```

## 监控训练

训练过程中会自动记录以下指标：

- `train/loss`: 总损失
- `train/policy_loss`: 策略损失
- `train/value_loss`: 价值损失
- `train/entropy`: 策略熵
- `train/episode_reward_mean`: 平均episode奖励
- `train/approx_kl`: 近似KL散度
- `train/clipfrac`: 裁剪比例

使用TensorBoard查看训练进度：

```bash
tensorboard --logdir logs/
```

## 关键设计决策

### 1. FSMN序列记忆
FSMN提供了一种轻量级的序列建模能力，相比LSTM/GRU更加高效，特别适合实时推理场景。

### 2. 组合式Actor-Critic
使用组合而非继承的方式封装FSMN，保持了原始模型的完整性，便于复用和测试。

### 3. TorchRL Collector
使用TorchRL的collector替代传统dataloader，提供了更好的环境交互抽象和数据收集效率。

### 4. 手动优化
在Lightning中使用手动优化模式，以便更好地控制PPO的多epoch训练流程。

## 性能预期

在CartPole-v1环境中，该实现通常能够：
- 在50-200个epoch内收敛
- 达到475+的平均奖励（接近理论最优500）
- 保持稳定的性能，很少出现性能倒退

## 扩展建议

1. **其他环境**: 修改`obs_dim`和`action_dim`适配其他Gym环境
2. **连续控制**: 将Categorical分布替换为Gaussian分布支持连续动作
3. **多智能体**: 扩展为支持多智能体训练
4. **模型压缩**: 利用FSMN的结构特点进行模型量化和压缩

## 故障排除

1. **训练不稳定**: 降低学习率或减少num_epochs
2. **内存不足**: 减少num_envs或frames_per_batch
3. **收敛缓慢**: 增加hidden_dim或num_blocks
4. **梯度爆炸**: 降低gradient_clip_val
