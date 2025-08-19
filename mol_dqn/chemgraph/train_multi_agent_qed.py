import argparse
import logging
import os
import wandb
from mol_dqn.utils.hparams import get_hparams
from datetime import datetime
from mol_dqn.chemgraph.dqn.agent import DoubleDQNAgent
from mol_dqn.chemgraph.runner.multi_agent_runner import AdversarialTrainer
from mol_dqn.chemgraph.optimize_qed import QEDRewardMolecule

def main():
    parser = argparse.ArgumentParser(description='Adversarial Molecular DQN Training')
    parser.add_argument('--config', type=str, default="./mol_dqn/chemgraph/configs/multi_agent_dqn.json",help='Path to JSON config file')
    parser.add_argument('--start_molecule', type=str, default=None,
                     help='Starting molecule SMILES string')
    parser.add_argument('--model_dir', type=str, default="./models", help='Directory to save models')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='mol-dqn-compete',
                     help='Weights & Biases project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                     help='Weights & Biases run name')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_model_dir = os.path.join(args.model_dir, timestamp)
    
    # 创建模型目录
    os.makedirs(timestamped_model_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Model directory: {timestamped_model_dir}")

    if args.use_wandb:
        run_name = args.wandb_run_name if args.wandb_run_name else f"compete-molecule-{timestamp}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args)
        )
    
    # 获取超参数
    if args.config:
        # 从JSON文件加载
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
        # 移除注释项
        config = {k: v for k, v in config.items() if not k.startswith('_comment')}
        hparams = get_hparams(**config)
    else:
        # 使用默认参数
        hparams = get_hparams(adversarial_enabled=True)
    
    # 创建环境
    env_kwargs = {
        'atom_types': hparams.atom_types,
        'init_mol': args.start_molecule,
        'allow_removal': hparams.allow_removal,
        'allow_no_modification': hparams.allow_no_modification,
        'allow_bonds_between_rings': hparams.allow_bonds_between_rings,
        'allowed_ring_sizes': hparams.allowed_ring_sizes,
        'max_steps': hparams.max_steps_per_episode,
        'discount_factor': hparams.discount_factor
    }
    
    # 创建两个智能体
    agent_A = DoubleDQNAgent(input_dim=hparams.input_dim, hparams=hparams)
    agent_B = DoubleDQNAgent(input_dim=hparams.input_dim, hparams=hparams)
    
    # 创建训练器
    trainer = AdversarialTrainer(
        hparams=hparams,
        env_class=QEDRewardMolecule,
        env_kwargs=env_kwargs,
        agent_A=agent_A,
        agent_B=agent_B,
        model_dir=timestamped_model_dir
    )
    
    # 开始训练
    trainer.run_training(use_wandb=args.use_wandb)

    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()