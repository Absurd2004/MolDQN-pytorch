import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import logging
from torch.utils.tensorboard import SummaryWriter

from mol_dqn.chemgraph.dqn.double_dqn_network import QNetwork
from mol_dqn.utils.utils import get_fingerprint


class DoubleDQNAgent(nn.Module):
    """Double DQN Agent for molecule optimization."""
    
    def __init__(self, input_dim, hparams):
        """
        Args:
            input_dim: int. Input dimension (fingerprint_length + 1)
            hparams: Hyperparameter object containing all configurations
        """
        super(DoubleDQNAgent, self).__init__()
        
        self.hparams = hparams
        self.input_dim = input_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建主网络和目标网络
        self.main_network = QNetwork(input_dim, hparams)
        self.target_network = QNetwork(input_dim, hparams)
        
        # 初始化目标网络参数与主网络相同
        self.update_target_network()
        
        # 优化器
        if hparams.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.main_network.parameters(), 
                lr=hparams.learning_rate
            )
        elif hparams.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(
                self.main_network.parameters(), 
                lr=hparams.learning_rate
            )
        else:
            raise ValueError(f"Unsupported optimizer: {hparams.optimizer}")
        
        # 学习率调度器
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, 
            gamma=hparams.learning_rate_decay_rate
        )
        
        # 其他参数
        self.gamma = hparams.gamma
        self.epsilon = 1.0  # 初始探索率
        self.double_q = hparams.double_q
        self.grad_clipping = hparams.grad_clipping
        self.num_bootstrap_heads = hparams.num_bootstrap_heads
        
        # 训练步数计数器
        self.training_steps = 0

    def get_action(self, observations, head=0, update_epsilon=None, stochastic=True):
        """
            选择动作（带 epsilon-greedy 探索）
            
        Args:
            observations: np.array, shape [num_actions, fingerprint_length + 1]
            head: int, bootstrap head index
            update_epsilon: float, 更新 epsilon 值
            stochastic: bool, 是否使用随机策略
                
        Returns:
            int, 选择的动作索引
        """
        if update_epsilon is not None:
            self.epsilon = update_epsilon
            
        # Epsilon-greedy 探索
        if stochastic and np.random.uniform() < self.epsilon:
            return np.random.randint(0, observations.shape[0])
        else:
            return self._run_action_op(observations, head)
        
    def _run_action_op(self, observations, head):
        """
        运行前向传播选择最优动作
            
        Args:
            observations: np.array, shape [num_actions, fingerprint_length + 1]
            head: int, bootstrap head index
                
        Returns:
            int, 最优动作索引
        """
        self.main_network.eval()
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observations).to(self.device)
            q_values = self.main_network(obs_tensor)
            
            if self.num_bootstrap_heads:
                # 如果使用 bootstrap，选择对应的 head
                action = torch.argmax(q_values[:, head]).item()
            else:
                # 标准 DQN，选择最大 Q 值对应的动作
                action = torch.argmax(q_values.squeeze()).item()
                
        self.main_network.train()
        return action

    def train(self, states, rewards, next_states, done, weight):
        """训练网络"""
        self.main_network.train()
        self.training_steps += 1


        current_q_values = self.main_network(states)

        if self.num_bootstrap_heads:
            # Bootstrap DQN: 随机选择一个head进行训练
            head_mask = torch.randint(0, 2, (states.shape[0], self.num_bootstrap_heads), 
                                    dtype=torch.float32, device=self.device)
            current_q_values = (current_q_values * head_mask).sum(dim=1, keepdim=True)
        else:
            # 标准 DQN
            current_q_values = current_q_values.squeeze(-1).unsqueeze(-1)
        
        with torch.no_grad():
            if self.double_q:
                next_q_main = self.main_network(next_states)

                next_actions = torch.argmax(next_q_main, dim=-1 if not self.num_bootstrap_heads else -2)

                next_q_target = self.target_network(next_states)

                if self.num_bootstrap_heads:
                    # 处理 bootstrap heads
                    batch_size = next_q_target.shape[0]
                    head_idx = torch.randint(0, self.num_bootstrap_heads, (batch_size,), device=self.device)
                    next_q_values = next_q_target[torch.arange(batch_size), next_actions, head_idx].unsqueeze(-1)
                
                else:
                    next_q_values = next_q_target.gather(1, next_actions.unsqueeze(-1))

            else:
                # 标准 DQN
                next_q_target = self.target_network(next_states)
                if self.num_bootstrap_heads:
                    next_q_values = torch.max(next_q_target, dim=-1)[0].max(dim=-1, keepdim=True)[0]
                else:
                    next_q_values = torch.max(next_q_target, dim=-1, keepdim=True)[0]
            
            target_q_values = rewards + (self.gamma * next_q_values * (1 - done))

        
        td_error = current_q_values - target_q_values


        huber_loss = torch.where(
            torch.abs(td_error) < 1.0,
            0.5 * td_error.pow(2),
            torch.abs(td_error) - 0.5
        )

        weighted_loss = (weight * huber_loss).mean()


        self.optimizer.zero_grad()
        weighted_loss.backward()


        if self.grad_clipping:
            torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.grad_clipping)
        

        self.optimizer.step()
        
        # 学习率衰减
        if self.training_steps % self.hparams.learning_rate_decay_steps == 0:
            self.lr_scheduler.step()
        

        return weighted_loss.item(), td_error.detach()
    

    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.main_network.state_dict())
    

    def save_checkpoint(self, filepath):
        """保存检查点"""
        torch.save({
            'main_network': self.main_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'epsilon': self.epsilon
        }, filepath)

    def load_checkpoint(self, filepath):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.main_network.load_state_dict(checkpoint['main_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_steps = checkpoint['training_steps']
        self.epsilon = checkpoint['epsilon']
            


        