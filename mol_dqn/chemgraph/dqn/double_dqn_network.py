import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QNetwork(nn.Module):
    """Deep Q Network for molecule optimization.
    
    This network implements the multi-layer model used in the original MolDQN paper.
    It supports both standard DQN and Bootstrap DQN variants.
    """
    
    def __init__(self, input_dim, hparams):
        """
        Args:
            input_dim: int. Input dimension (fingerprint_length + 1)
            hparams: Hyperparameter object containing network configurations
        """
        super(QNetwork, self).__init__()
        
        self.hparams = hparams
        self.input_dim = input_dim
        self.dense_layers = hparams.dense_layers
        self.activation = hparams.activation
        self.batch_norm = hparams.batch_norm
        self.num_bootstrap_heads = hparams.num_bootstrap_heads
        
        # 构建网络层
        self._build_network()
        
        # 初始化权重
        self._init_weights()
    
    def _build_network(self):
        """构建网络结构"""
        layers = []
        prev_dim = self.input_dim
        
        # 构建隐藏层
        for i, units in enumerate(self.dense_layers):
            # 全连接层
            layers.append(nn.Linear(prev_dim, units))
            
            # 批量归一化（如果启用）
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(units))
            
            # 激活函数
            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'tanh':
                layers.append(nn.Tanh())
            elif self.activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Unsupported activation: {self.activation}")
            
            prev_dim = units
        
        # 主干网络
        self.backbone = nn.Sequential(*layers)
        
        # 输出层
        if self.num_bootstrap_heads:
            # Bootstrap DQN: 多个头
            self.output_layer = nn.Linear(prev_dim, self.num_bootstrap_heads)
        else:
            # 标准 DQN: 单个输出
            self.output_layer = nn.Linear(prev_dim, 1)
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用 Xavier 初始化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: torch.Tensor, shape [batch_size, input_dim] or [num_actions, input_dim]
            
        Returns:
            torch.Tensor: Q values
            - 如果使用 Bootstrap: shape [batch_size, num_bootstrap_heads]
            - 如果标准 DQN: shape [batch_size, 1]
        """
        # 通过主干网络
        features = self.backbone(x)
        
        # 输出层
        q_values = self.output_layer(features)
        
        return q_values


class MultiObjectiveQNetwork(nn.Module):
    """Multi-objective Q Network for multiple reward objectives.
    
    This network is used when optimizing multiple objectives simultaneously,
    such as similarity and QED in the multi-objective DQN setting.
    """
    
    def __init__(self, input_dim, hparams, num_objectives):
        """
        Args:
            input_dim: int. Input dimension
            hparams: Hyperparameter object
            num_objectives: int. Number of objectives to optimize
        """
        super(MultiObjectiveQNetwork, self).__init__()
        
        self.num_objectives = num_objectives
        self.hparams = hparams
        
        # 为每个目标创建独立的 Q 网络
        self.q_networks = nn.ModuleList([
            QNetwork(input_dim, hparams) for _ in range(num_objectives)
        ])
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: torch.Tensor, shape [batch_size, input_dim]
            
        Returns:
            list of torch.Tensor: 每个目标的 Q 值列表
        """
        q_values_list = []
        for q_net in self.q_networks:
            q_values = q_net(x)
            q_values_list.append(q_values)
        
        return q_values_list


# 用于测试网络的辅助函数
def test_qnetwork():
    """测试 QNetwork 的基本功能"""
    
    # 创建假的超参数
    class TestHParams:
        def __init__(self):
            self.dense_layers = [512, 256, 128]
            self.activation = 'relu'
            self.batch_norm = False
            self.num_bootstrap_heads = 12
    
    hparams = TestHParams()
    
    # 测试 Bootstrap DQN
    print("测试 Bootstrap DQN...")
    input_dim = 2049  # fingerprint_length + 1
    batch_size = 32
    
    network = QNetwork(input_dim, hparams)
    
    # 创建测试输入
    x = torch.randn(batch_size, input_dim)
    
    # 前向传播
    output = network(x)
    print(f"Bootstrap DQN 输出形状: {output.shape}")  # 应该是 [32, 12]
    assert output.shape == (batch_size, hparams.num_bootstrap_heads)
    
    # 测试标准 DQN
    print("测试标准 DQN...")
    hparams.num_bootstrap_heads = 0
    
    network_standard = QNetwork(input_dim, hparams)
    output_standard = network_standard(x)
    print(f"标准 DQN 输出形状: {output_standard.shape}")  # 应该是 [32, 1]
    assert output_standard.shape == (batch_size, 1)
    
    print("所有测试通过！")


if __name__ == "__main__":
    test_qnetwork()