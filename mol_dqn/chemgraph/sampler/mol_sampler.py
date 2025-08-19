import torch
import copy
from mol_dqn.chemgraph.dqn.agent import DoubleDQNAgent
import os

class MolecularSampler:
    """分子采样器，包装 DQN Agent 用于推理，参考 CompetEvo 的 DevSampler"""
    
    def __init__(self, agent_template: DoubleDQNAgent, device):
        # 深拷贝模板 agent 避免影响原始训练
        self.agent = copy.deepcopy(agent_template)
        self.device = device
        self.agent.to(device)
        self.agent.eval()  # 设为评估模式
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载对手的历史 checkpoint"""
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.agent.load_checkpoint(checkpoint_path)
        self.agent.eval()
    
    @torch.no_grad()
    def get_action(self, observations, head=0, update_epsilon = 0,stochastic=True):
        """
        获取动作（纯推理，不更新参数）
        
        Args:
            observations: np.array [num_actions, feature_dim]
            head: bootstrap head index
            stochastic: 是否使用随机策略（通常对手用贪心）
            
        Returns:
            int: 动作索引
        """
        return self.agent.get_action(
            observations, 
            head=head, 
            update_epsilon=update_epsilon,
            stochastic=stochastic
        )