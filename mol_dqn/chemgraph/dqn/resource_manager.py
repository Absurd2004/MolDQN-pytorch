
class ResourceManager:
    """管理分子生成过程中的资源限制"""
    
    def __init__(self, atom_types, max_atom_uses, max_operation_uses,min_atom_uses=None, min_operation_uses=None, curriculum_learning=False):
        """
        Args:
            atom_types: list, 原子类型列表 ['C', 'O', 'N']
            max_atom_uses: dict, 每种原子的最大使用次数 {'C': 20, 'O': 10, 'N': 5}
            max_operation_uses: dict, 每种操作的最大使用次数 
            min_atom_uses: dict, 每种原子的最小使用次数（课程学习的下界）
            min_operation_uses: dict, 每种操作的最小使用次数（课程学习的下界）
            curriculum_learning: bool, 是否启用课程学习
        """
        self.atom_types = atom_types
        self.max_atom_uses = max_atom_uses.copy()
        self.max_operation_uses = max_operation_uses.copy()

        self.curriculum_learning = curriculum_learning
        if curriculum_learning:
            self.min_atom_uses = min_atom_uses or {atom: max(1, max_atom_uses[atom] // 4) for atom in atom_types}
            self.min_operation_uses = min_operation_uses or {op: max(1, val // 4) for op, val in max_operation_uses.items()}
        else:
            self.min_atom_uses = max_atom_uses.copy()
            self.min_operation_uses = max_operation_uses.copy()
        
        # 当前剩余使用次数
        self.remaining_atom_uses = max_atom_uses.copy()
        self.remaining_operation_uses = max_operation_uses.copy()
    
    def reset(self, new_limits=None, episode=None, total_episodes=None):
        """重置资源限制，支持课程学习"""
        if new_limits:
            self.max_atom_uses.update(new_limits.get('atoms', {}))
            self.max_operation_uses.update(new_limits.get('operations', {}))
        
        # === 新增：课程学习逻辑 ===
        if self.curriculum_learning and episode is not None and total_episodes is not None:
            # 计算当前episode的进度比例 (0.0 到 1.0)
            progress = min(episode / total_episodes, 1.0)
            
            # 线性插值：从最大值逐渐减少到最小值
            current_atom_uses = {}
            for atom in self.atom_types:
                max_val = self.max_atom_uses[atom]
                min_val = self.min_atom_uses[atom]
                current_val = int(max_val - progress * (max_val - min_val))
                current_atom_uses[atom] = max(current_val, min_val)
            
            current_operation_uses = {}
            for op, max_val in self.max_operation_uses.items():
                min_val = self.min_operation_uses[op]
                current_val = int(max_val - progress * (max_val - min_val))
                current_operation_uses[op] = max(current_val, min_val)
            
            self.remaining_atom_uses = current_atom_uses
            self.remaining_operation_uses = current_operation_uses
        else:
            # 普通重置
            self.remaining_atom_uses = self.max_atom_uses.copy()
            self.remaining_operation_uses = self.max_operation_uses.copy()
    
    def can_use_atom(self, atom_type):
        """检查是否还能使用指定原子"""
        return self.remaining_atom_uses.get(atom_type, 0) > 0
    
    def can_use_operation(self, operation_type):
        """检查是否还能使用指定操作"""
        return self.remaining_operation_uses.get(operation_type, 0) > 0
    
    def use_atom(self, atom_type):
        """使用一个原子，减少剩余次数"""
        if self.can_use_atom(atom_type):
            self.remaining_atom_uses[atom_type] -= 1
            return True
        return False
    
    def use_operation(self, operation_type):
        """使用一个操作，减少剩余次数"""
        if self.can_use_operation(operation_type):
            self.remaining_operation_uses[operation_type] -= 1
            return True
        return False
    
    def get_resource_vector(self):
        """获取资源状态向量，用于observation"""
        atom_vector = [self.remaining_atom_uses.get(atom, 0) for atom in self.atom_types]
        operation_vector = [
            self.remaining_operation_uses.get('bond_addition', 0),
            self.remaining_operation_uses.get('bond_removal', 0), 
            self.remaining_operation_uses.get('no_modification', 0)
        ]
        return atom_vector + operation_vector
    
    def get_resource_info(self):
        """获取详细的资源信息（用于调试）"""
        return {
            'atoms': self.remaining_atom_uses,
            'operations': self.remaining_operation_uses
        }

    def get_current_limits_info(self):
        """获取当前限制信息（用于调试和日志）"""
        return {
            'current_atom_limits': self.remaining_atom_uses.copy(),
            'current_operation_limits': self.remaining_operation_uses.copy(),
            'max_atom_limits': self.max_atom_uses.copy(),
            'min_atom_limits': self.min_atom_uses.copy() if self.curriculum_learning else None
        }