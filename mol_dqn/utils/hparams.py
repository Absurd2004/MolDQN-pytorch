def get_hparams(**kwargs):
    """获取模型超参数
    
    Args:
        **kwargs: 参数覆盖字典
        
    Returns:
        包含所有超参数的对象
    """
    
    class HParams:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    
    default_params = {
        # 分子相关参数
        'atom_types': ['C', 'O', 'N'],
        'max_steps_per_episode': 40,
        'allow_removal': True,
        'allow_no_modification': True,
        'allow_bonds_between_rings': False,
        'allowed_ring_sizes': [3, 4, 5, 6],
        
        # 训练参数
        'replay_buffer_size': 1000000,
        'learning_rate': 1e-4,
        'learning_rate_decay_steps': 10000,
        'learning_rate_decay_rate': 0.8,
        'num_epoch': 1000,  # 统一使用 num_epoch
        'batch_size': 64,
        'learning_frequency': 4,
        'update_frequency': 20,
        'grad_clipping': 10.0,
        'gamma': 0.9,
        
        # DQN 参数
        'double_q': True,
        'num_bootstrap_heads': 12,
        'prioritized': False,
        'prioritized_alpha': 0.6,
        'prioritized_beta': 0.4,
        'prioritized_epsilon': 1e-6,
        
        # 网络参数
        'fingerprint_radius': 3,
        'fingerprint_length': 2048,
        'dense_layers': [1024, 512, 128, 32],
        'activation': 'relu',
        'optimizer': 'Adam',
        'batch_norm': False,
        
        # 保存参数
        'save_frequency': 1000,
        'max_num_checkpoints': 100,
        'discount_factor': 0.7,
        
        # 对抗训练参数
        'adversarial_enabled': False,
        'adversarial_delta': 0.5,
        'episodes_per_epoch': 10,
        
        # 稀疏奖励参数
        'use_sparse_reward': True,
        'sparse_win_reward': 100.0,
        'sparse_lose_reward': -100.0,
        'sparse_draw_reward': 0.0,
        'sparse_reward_annealing': True,
        'sparse_reward_termination_epoch': 800,
        
        # 评估参数
        'eval_frequency': 50,
        'eval_episodes': 20,
    }

    if kwargs.get('adversarial_enabled', default_params.get('adversarial_enabled')):
        fp_len = kwargs.get('fingerprint_length', default_params['fingerprint_length'])
        original_dim = fp_len + 1
        default_params['input_dim'] = original_dim * 2
    else:
        fp_len = kwargs.get('fingerprint_length', default_params['fingerprint_length'])
        default_params['input_dim'] = fp_len + 1
    
    # 用传入的参数覆盖默认参数
    default_params.update(kwargs)
    
    return HParams(**default_params)

# 在 mol_dqn/utils/hparams.py 中添加缺失的配置
def extend_hparams_for_adversarial(hparams):
    """为现有 hparams 添加对抗训练相关配置"""
    # 对抗训练开关
    hparams.adversarial_enabled = getattr(hparams, 'adversarial_enabled', False)
    
    # 对手采样参数
    hparams.adversarial_delta = getattr(hparams, 'adversarial_delta', 0.5)
    
    # 输入维度（如果启用对抗则自动调整）
    if hparams.adversarial_enabled:
        original_dim = hparams.fingerprint_length + 1
        hparams.input_dim = original_dim * 2
        print(f"Adversarial mode enabled. Input dim: {original_dim} -> {hparams.input_dim}")
    else:
        hparams.input_dim = hparams.fingerprint_length + 1
        
    # 每个 epoch 的 episodes 数量
    hparams.episodes_per_epoch = getattr(hparams, 'episodes_per_epoch', 10)
    
    # 奖励模式
    hparams.adversarial_reward_mode = getattr(hparams, 'adversarial_reward_mode', 'self')

    # 稀疏奖励配置
    hparams.use_sparse_reward = getattr(hparams, 'use_sparse_reward', True)
    hparams.sparse_win_reward = getattr(hparams, 'sparse_win_reward', 100.0)
    hparams.sparse_lose_reward = getattr(hparams, 'sparse_lose_reward', -100.0)
    hparams.sparse_draw_reward = getattr(hparams, 'sparse_draw_reward', 0.0)
    hparams.sparse_reward_annealing = getattr(hparams, 'sparse_reward_annealing', True)
    hparams.sparse_reward_termination_epoch = getattr(hparams, 'sparse_reward_termination_epoch', 800)
    
    return hparams