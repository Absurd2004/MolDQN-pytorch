import argparse


from rdkit import Chem

from rdkit.Chem import QED
import json
import logging
import wandb


from mol_dqn.chemgraph.dqn import environment as molecules_mdp
from mol_dqn.chemgraph.dqn import run_dqn
from mol_dqn.chemgraph.dqn.params import core
from mol_dqn.chemgraph.dqn.agent import DoubleDQNAgent
from mol_dqn.utils.hparams import get_hparams
import os

class QEDRewardMolecule(molecules_mdp.Molecule):
    """The molecule whose reward is the QED."""

    def __init__(self, discount_factor, **kwargs):
        """Initializes the class.

        Args:
        discount_factor: Float. The discount factor. We only
            care about the molecule at the end of modification.
            In order to prevent a myopic decision, we discount
            the reward at each step by a factor of
            discount_factor ** num_steps_left,
            this encourages exploration with emphasis on long term rewards.
        **kwargs: The keyword arguments passed to the base class.
        """
        super(QEDRewardMolecule, self).__init__(**kwargs)
        self.discount_factor = discount_factor

    def _reward(self):
        """Reward of a state.

        Returns:
        Float. QED of the current state.
        """
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        qed = QED.qed(molecule)
        return qed * self.discount_factor ** (self.max_steps - self.num_steps_taken)


def main():
    parser = argparse.ArgumentParser(description='Optimize QED of a molecule with DQN')
    parser.add_argument('--hparams', type=str, default=None,
                     help='Path to hyperparameters JSON file')
    parser.add_argument('--start_molecule', type=str, default=None,
                     help='Starting molecule SMILES string')
    parser.add_argument('--model_dir', type=str, required=True,
                     help='Directory to save model and logs')

    parser.add_argument('--wandb_project', type=str, default='mol-dqn',
                     help='Weights & Biases project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                     help='Weights & Biases run name')
    parser.add_argument('--use_wandb', type=bool, default=False,
                     help='Disable Weights & Biases logging')
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )



    if args.hparams is not None:
        with open(args.hparams, 'r') as f:
            hparams = get_hparams(**json.load(f))
    else:
        hparams = get_hparams()
    

    if args.use_wandb:
        wandb.config.update(hparams.__dict__)
    

    environment = QEDRewardMolecule(
      discount_factor=hparams.discount_factor,
      atom_types=set(hparams.atom_types),
      init_mol=args.start_molecule,
      allow_removal=hparams.allow_removal,
      allow_no_modification=hparams.allow_no_modification,
      allow_bonds_between_rings=hparams.allow_bonds_between_rings,
      allowed_ring_sizes=set(hparams.allowed_ring_sizes),
      max_steps=hparams.max_steps_per_episode)

    dqn = DoubleDQNAgent(
        input_dim=hparams.fingerprint_length + 1,
        hparams=hparams,
        )

    #print(f"初始化成功")
    #assert False, "检查点"
    
    run_dqn.run_training(
      hparams=hparams,
      environment=environment,
      dqn=dqn,
      model_dir=args.model_dir,
      use_wandb=args.use_wandb)

    if args.use_wandb:
        wandb.finish()
    
    core.write_hparams(hparams, os.path.join(args.model_dir, 'config.json'))


if __name__ == '__main__':
    main()  # 移除 app.run(main)，直接调用 main()