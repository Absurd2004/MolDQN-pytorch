import argparse


from rdkit import Chem

from rdkit.Chem import QED
import json
import logging
import wandb


from mol_dqn.chemgraph.dqn import environment_constrain as molecules_mdp
from mol_dqn.chemgraph.dqn import run_dqn_constrain
from mol_dqn.chemgraph.dqn.params import core
from mol_dqn.chemgraph.dqn.agent import DoubleDQNAgent
from mol_dqn.chemgraph.dqn.py import molecules
from mol_dqn.utils.hparams import get_hparams
import os

class LogPRewardMolecule(molecules_mdp.Molecule):
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
        super(LogPRewardMolecule, self).__init__(**kwargs)
        self.discount_factor = discount_factor

    def _reward(self):
        """Reward of a state.

        Returns:
        Float. QED of the current state.
        """
        #molecule = Chem.MolFromSmiles(self._state)
        molecule = self.state_mol
        if molecule is None:
            return 0.0
        LogP = molecules.penalized_logp(molecule)
        return LogP * self.discount_factor ** (self.max_steps - self.num_steps_taken)


def main():
    parser = argparse.ArgumentParser(description='Optimize QED of a molecule with DQN')
    parser.add_argument('--hparams', type=str, default="./mol_dqn/chemgraph/configs/naive_dqn_constrain.json",
                     help='Path to hyperparameters JSON file')
    parser.add_argument('--start_molecule', type=str, default=None,
                     help='Starting molecule SMILES string')
    parser.add_argument('--model_dir', type=str, default="./models/one-agent-logP",
                     help='Directory to save model and logs')

    parser.add_argument('--wandb_project', type=str, default='mol-dqn',
                     help='Weights & Biases project name')
    parser.add_argument('--wandb_run_name', type=str, default="one-agent-logP constrain",
                     help='Weights & Biases run name')
    parser.add_argument('--use_wandb', type=bool, default=False,
                     help='Disable Weights & Biases logging')
    
    args = parser.parse_args()

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.model_dir = os.path.join(args.model_dir, timestamp)
    os.makedirs(args.model_dir, exist_ok=True)

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
    

    environment = LogPRewardMolecule(
      discount_factor=hparams.discount_factor,
      atom_types=set(hparams.atom_types),
      init_mol=args.start_molecule,
      allow_removal=hparams.allow_removal,
      allow_no_modification=hparams.allow_no_modification,
      allow_bonds_between_rings=hparams.allow_bonds_between_rings,
      allowed_ring_sizes=set(hparams.allowed_ring_sizes),
      max_steps=hparams.max_steps_per_episode,
      use_resource_limits=True,
      curriculum_learning=True,  # 启用课程学习
      max_atom_uses={'C': 40, 'O': 40, 'N': 40},      # 起始上界
      min_atom_uses={'C': 15, 'O': 15, 'N': 15}, 
      max_operation_uses={
        'bond_addition': 40,
        'bond_removal': 40, 
        'no_modification': 40
      },
      min_operation_uses={
        'bond_addition': 10,
        'bond_removal': 5,
        'no_modification': 5
      }
    )      # 最终下界)

    resource_dim = len(hparams.atom_types) + 3
    input_dim = hparams.fingerprint_length + 1 + resource_dim

    dqn = DoubleDQNAgent(
        input_dim=input_dim,
        hparams=hparams,
        )




    
    
    run_dqn_constrain.run_training(
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