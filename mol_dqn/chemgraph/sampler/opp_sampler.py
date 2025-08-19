import os
import math
import random
import pickle
from typing import List, Tuple, Optional

class OpponentHistory:
    """管理对手历史 checkpoint，参考 CompetEvo 设计"""
    
    def __init__(self):
        self._checkpoints: List[Tuple[int, str]] = []  # (epoch, checkpoint_path)
    
    def register(self, epoch: int, checkpoint_path: str):
        """注册新的 checkpoint"""
        self._checkpoints.append((epoch, checkpoint_path))
    
    def get_window(self, start_epoch: int, end_epoch: int) -> List[Tuple[int, str]]:
        """获取指定 epoch 窗口内的 checkpoints"""
        return [(e, p) for (e, p) in self._checkpoints 
                if start_epoch <= e <= end_epoch]
    
    def latest(self) -> Optional[str]:
        """获取最新的 checkpoint"""
        return self._checkpoints[-1][1] if self._checkpoints else None


class OpponentSampler:
    """对手采样器，参考 CompetEvo 的 delta 采样策略"""
    
    def __init__(self, history: OpponentHistory, delta: float):
        self.history = history
        self.delta = delta  # 采样窗口比例参数
    
    def sample(self, current_epoch: int) -> Optional[str]:
        """
        根据 delta 参数采样历史对手
        
        Args:
            current_epoch: 当前训练轮次
            
        Returns:
            采样到的 checkpoint 路径，如果没有历史则返回 None
        """
        if current_epoch == 0:
            # epoch=0时没有历史，返回None
            return None
        
        if not self.history._checkpoints:
            # 没有任何历史ckpt，返回None
            return None

        
        # 计算采样窗口：[start, current_epoch-1]
        start = max(1, math.floor(current_epoch * self.delta))
        end = current_epoch - 1

        if end < start:
            # 窗口无效，返回最新的（如果有的话）
            return self.history.latest()
        
        candidates = self.history.get_window(start, end)
        if not candidates:
            return self.history.latest()
        
        # 随机选择一个
        return random.choice(candidates)[1]