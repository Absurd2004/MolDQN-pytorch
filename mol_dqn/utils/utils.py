from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import numpy as np
from rdkit.Chem import rdFingerprintGenerator



def get_fingerprint(smiles, hparams):
  """Get Morgan Fingerprint of a specific SMILES string.

  Args:
    smiles: String. The SMILES string of the molecule.
    hparams: tf.contrib.training.HParams. Hyper parameters.

  Returns:
    np.array. shape = [hparams.fingerprint_length]. The Morgan fingerprint.
  """
  if smiles is None:
    return np.zeros((hparams.fingerprint_length,))
  molecule = Chem.MolFromSmiles(smiles)
  if molecule is None:
    return np.zeros((hparams.fingerprint_length,))
  
  

  
  # 使用教程中推荐的方法
  mfpgen = rdFingerprintGenerator.GetMorganGenerator(
      radius=hparams.fingerprint_radius,
      fpSize=hparams.fingerprint_length,
      countSimulation=False,
  )

  fingerprint = mfpgen.GetFingerprint(molecule)

  arr = np.zeros((1,))

  DataStructs.ConvertToNumpyArray(fingerprint, arr)

  return arr