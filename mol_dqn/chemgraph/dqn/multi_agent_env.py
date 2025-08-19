import numpy as np
from mol_dqn.utils.utils import get_fingerprint

class DualMoleculeEnv:
    """
    双分子对抗环境包装器
    管理主智能体和对手智能体的独立分子环境
    """
    def __init__(self, env_class, env_kwargs, hparams):
        """
        Args:
            env_class: 分子环境类 (如 QEDRewardMolecule)
            env_kwargs: 环境初始化参数
            hparams: 超参数，包含 fingerprint_length 等
        """
        self.env_main = env_class(**env_kwargs)  # 主智能体环境
        self.env_opp = env_class(**env_kwargs)   # 对手智能体环境

        self.hparams = hparams
        self.max_steps = env_kwargs.get('max_steps', 40)

        self.current_step = 0
        self.terminated = False

    def reset(self):
        """重置双环境到初始状态"""
        self.env_main.initialize()
        self.env_opp.initialize()
        self.current_step = 0
        self.terminated = False

        return self._get_current_states()

    def _get_current_states(self):
        """获取当前双方状态信息"""
        return {
            'main_state': self.env_main.state,
            'main_mol': self.env_main.state_mol,
            'opp_state': self.env_opp.state,
            'opp_mol': self.env_opp.state_mol,
            'main_steps_left': self.max_steps - self.env_main.num_steps_taken,
            'opp_steps_left': self.max_steps - self.env_opp.num_steps_taken
        }
    
    def _build_molecular_feature(self, mol_state, steps_left):
        """
        构建分子特征向量
        
        Args:
            mol_state: 分子的 Mol 对象或 None
            steps_left: 剩余步数
            
        Returns:
            np.array: [fingerprint_length + 1] 的特征向量
        """
        fp_len = self.hparams.fingerprint_length

        
        if mol_state is not None:
            # 使用分子指纹
            fingerprint = get_fingerprint(mol_state, self.hparams)
        else:
            # 空分子用零向量
            fingerprint = np.zeros(fp_len, dtype=np.float32)
        
        return np.append(fingerprint, steps_left).astype(np.float32)
    
    def get_main_observations(self):
        """
        为主智能体构造观测（拼接对手状态）
        
        Returns:
            np.array: [num_main_actions, main_feature_dim + opp_feature_dim]
        """

        main_valid_actions = list(self.env_main.get_valid_actions())
        
        main_steps_left = self.max_steps - self.env_main.num_steps_taken
        #print(f"Main steps left: {main_steps_left}")
        main_action_features = np.vstack([
            np.append(get_fingerprint(action, self.hparams), main_steps_left)
            for action in main_valid_actions
        ])

        #print(f"Main valid actions: {len(main_valid_actions)}")
        #print(f"Main action features shape: {main_action_features.shape}")


        opp_steps_left = self.max_steps - self.env_opp.num_steps_taken
        #print(f"Opponent steps left: {opp_steps_left}")
        opp_state_feature = self._build_molecular_feature(
            self.env_opp.state, opp_steps_left
        )

        #print(f"Opponent state feature shape: {opp_state_feature.shape}")


        num_actions = main_action_features.shape[0]
        opp_features_tiled = np.repeat(
            opp_state_feature[None, :], num_actions, axis=0
        )

        combined_features = np.concatenate([
            main_action_features, opp_features_tiled
        ], axis=1)

        #print(f"Combined features shape: {combined_features.shape}")

        return combined_features, main_valid_actions
    
    def get_opp_observations(self):
        """
        为对手智能体构造观测（拼接主智能体状态）
        """
        # 获取对手的候选动作
        opp_valid_actions = list(self.env_opp.get_valid_actions())
        if not opp_valid_actions:
            return np.array([]).reshape(0, -1), []
        
        # 构造对手候选动作特征
        opp_steps_left = self.max_steps - self.env_opp.num_steps_taken
        #print(f"Opponent steps left: {opp_steps_left}")
        opp_action_features = np.vstack([
            np.append(get_fingerprint(action, self.hparams), opp_steps_left)
            for action in opp_valid_actions
        ])
        #print(f"Opponent valid actions: {len(opp_valid_actions)}")
        #print(f"Opponent action features shape: {opp_action_features.shape}")
        
        # 构造主智能体当前状态特征
        main_steps_left = self.max_steps - self.env_main.num_steps_taken
        #print(f"Main steps left: {main_steps_left}")
        main_state_feature = self._build_molecular_feature(
            self.env_main.state, main_steps_left
        )
        #print(f"Main state feature shape: {main_state_feature.shape}")
        
        
        # 为每个候选动作拼接主智能体特征
        num_actions = opp_action_features.shape[0]
        main_features_tiled = np.repeat(
            main_state_feature[None, :], num_actions, axis=0
        )
        
        # 拼接：[对手动作特征 + 主状态特征]
        combined_features = np.concatenate([
            opp_action_features, main_features_tiled
        ], axis=1)
        #print(f"Combined features shape: {combined_features.shape}")
        
        return combined_features, opp_valid_actions
    def step_both(self, main_action, opp_action):
        """
        同步执行双方动作
        
        Args:
            main_action: 主智能体选择的动作 (SMILES)
            opp_action: 对手智能体选择的动作 (SMILES)
            
        Returns:
            dict: 包含执行结果的信息
        """
        # 检查是否已终止
        if self.terminated:
            raise ValueError("Episode has already terminated!")
        
        # 执行主智能体动作
        main_result = self.env_main.step(main_action)

        opp_result = self.env_opp.step(opp_action)

        self.current_step += 1

        main_terminated = main_result.terminated
        opp_terminated = opp_result.terminated

        step_terminated = (self.current_step >= self.max_steps)

        self.terminated = main_terminated or opp_terminated or step_terminated


        return {
            'main_result': main_result,
            'opp_result': opp_result,
            'terminated': self.terminated,
            'step': self.current_step,
            'states': self._get_current_states()
        }
    
    def is_terminated(self):
        """检查是否已终止"""
        return self.terminated
    
    def get_input_dim(self):
        """
        计算拼接后的输入维度
        
        Returns:
            int: 输入特征维度
        """
        fp_len = self.hparams.fingerprint_length
        # 自己的特征 (fingerprint + steps_left) + 对手的特征 (fingerprint + steps_left)  
        return (fp_len + 1) + (fp_len + 1)



