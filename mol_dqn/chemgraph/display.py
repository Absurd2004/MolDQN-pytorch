import argparse
import json
import logging
import os

from rdkit import Chem
from rdkit.Chem import QED

from mol_dqn.chemgraph.dqn import environment as molecules_mdp
from mol_dqn.chemgraph.dqn import run_dqn
from mol_dqn.chemgraph.dqn.agent import DoubleDQNAgent
from mol_dqn.utils.hparams import get_hparams


class QEDRewardMolecule(molecules_mdp.Molecule):
    """The molecule whose reward is the QED."""

    def __init__(self, discount_factor, **kwargs):
        super(QEDRewardMolecule, self).__init__(**kwargs)
        self.discount_factor = discount_factor

    def _reward(self):
        """Reward of a state."""
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        qed = QED.qed(molecule)
        return qed * self.discount_factor ** (self.max_steps - self.num_steps_taken)


def main():
    parser = argparse.ArgumentParser(description='Display trained MolDQN model')
    parser.add_argument('--hparams', type=str, required=True,
                       help='Path to hyperparameters JSON file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory to save display results')
    parser.add_argument('--start_molecule', type=str, default=None,
                       help='Starting molecule SMILES string')
    parser.add_argument('--num_episodes', type=int, default=5,
                       help='Number of episodes to display')
    parser.add_argument('--no_video', action='store_true',
                       help='Disable video generation')

    args = parser.parse_args()


    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.model_dir, 'display.log'))
        ]
    )

    with open(args.hparams, 'r') as f:
        hparams = get_hparams(**json.load(f))
    

    # 创建环境
    environment = QEDRewardMolecule(
        discount_factor=hparams.discount_factor,
        atom_types=set(hparams.atom_types),
        init_mol=args.start_molecule,
        allow_removal=hparams.allow_removal,
        allow_no_modification=hparams.allow_no_modification,
        allow_bonds_between_rings=hparams.allow_bonds_between_rings,
        allowed_ring_sizes=set(hparams.allowed_ring_sizes),
        max_steps=hparams.max_steps_per_episode
    )
    
    # 创建DQN智能体
    dqn = DoubleDQNAgent(
        input_dim=hparams.fingerprint_length + 1,
        hparams=hparams
    )
    
    # 运行显示
    run_dqn.run_display(
        hparams=hparams,
        environment=environment,
        dqn=dqn,
        model_dir=args.model_dir,
        checkpoint_path=args.checkpoint,
        num_episodes=args.num_episodes,
        save_video=not args.no_video
    )
    
    logging.info("Display completed!")


if __name__ == '__main__':
    main()


