# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the Markov decision process of generating a molecule.

The problem of molecule generation as a Markov decision process, the
state space, action space, and reward function are defined.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import itertools

from rdkit import Chem
from rdkit.Chem import Draw
from six.moves import range
from six.moves import zip

from mol_dqn.chemgraph.dqn.py import molecules


class Result(
        collections.namedtuple('Result', ['state', 'reward', 'terminated'])):
    """A namedtuple defines the result of a step for the molecule class.

        The namedtuple contains the following fields:
            state: Chem.RWMol. The molecule reached after taking the action.
            reward: Float. The reward get after taking the action.
            terminated: Boolean. Whether this episode is terminated.
    """

def protect_initial_molecule(mol):
        """
        保护初始分子的所有原子和键
        
        Args:
                mol: RDKit Mol对象
        
        Returns:
                修改后的分子，所有原子和键都被标记为受保护
        """
        
        # 标记所有原子为受保护
        for atom in mol.GetAtoms():
                atom.SetBoolProp('_protected', True)
        
        # 标记所有键为受保护
        for bond in mol.GetBonds():
                bond.SetBoolProp('_protected', True)
        
        return mol

def is_atom_protected(mol, atom_idx):
        """检查原子是否受保护"""
        atom = mol.GetAtomWithIdx(atom_idx)
        return atom.HasProp('_protected') and atom.GetBoolProp('_protected')

def is_bond_protected(mol, atom1_idx, atom2_idx):
        """检查键是否受保护"""
        bond = mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
        if bond is None:
                return False
        return bond.HasProp('_protected') and bond.GetBoolProp('_protected')

def _would_form_ring_with_protected_atoms(mol, atom1_idx, atom2_idx):
        """
        在 mol 上模拟添加 atom1-atom2 键并检测新环是否包含任何受保护原子
        """

        if mol.GetBondBetweenAtoms(atom1_idx, atom2_idx) is not None:
                return False
        
        # 1. 拷贝并添加键
        new_mol = Chem.RWMol(mol)
        new_mol.AddBond(atom1_idx, atom2_idx, Chem.BondType.SINGLE)
        Chem.SanitizeMol(new_mol, catchErrors=True)



        # 2. 枚举所有环（SymmSSSR）
        rings = Chem.GetSymmSSSR(new_mol)

        # 3. 针对包含新增键两端原子的环检查受保护原子
        for ring in rings:
                if atom1_idx in ring and atom2_idx in ring:  # 仅新形成的环
                        for idx in ring:
                                atom = new_mol.GetAtomWithIdx(idx)
                                if atom.HasProp('_protected') and atom.GetBoolProp('_protected'):
                                        return True  # 有保护原子 -> 非法
        return False  # 未检测到违规

def _get_valid_actions_with_resources(state_mol, atom_types, allow_removal, allow_no_modification,
                                     allowed_ring_sizes, allow_bonds_between_rings, protect_initial, 
                                     resource_manager):
    """获取考虑资源限制的有效动作，同时返回资源消耗信息"""
    # 首先获取所有可能的动作（带分类信息）
    all_actions_with_resources = _generate_actions_with_resource_info(
        state_mol, atom_types, allow_removal, allow_no_modification,
        allowed_ring_sizes, allow_bonds_between_rings, protect_initial
    )
    
    # 过滤资源不足的动作
    valid_smiles = []
    valid_mols = []
    valid_resource_costs = []  # 记录每个动作的资源消耗
    
    for action_info in all_actions_with_resources:
        smiles = action_info['smiles']
        mol = action_info['mol']
        resource_cost = action_info['resource_cost']
        
        # 检查资源是否足够
        if _can_afford_resource_cost(resource_cost, resource_manager):
            valid_smiles.append(smiles)
            valid_mols.append(mol)
            valid_resource_costs.append(resource_cost)
    
    return valid_smiles, valid_mols, valid_resource_costs	
def _generate_actions_with_resource_info(state_mol, atom_types, allow_removal, allow_no_modification,
                                        allowed_ring_sizes, allow_bonds_between_rings, protect_initial):
    """生成所有可能的动作，并附带资源消耗信息"""
    if state_mol is None:
        # 空状态，返回初始原子类型
        actions_with_resources = []
        for atom_type in atom_types:
            actions_with_resources.append({
                'smiles': atom_type,
                'mol': Chem.MolFromSmiles(atom_type),
                'resource_cost': {'atom_type': atom_type, 'operation_type': None}
            })
        return actions_with_resources
    
    atom_valences = {
        atom_type: molecules.atom_valences([atom_type])[0]
        for atom_type in atom_types
    }

    atoms_with_free_valence = {}
    for i in range(1, max(atom_valences.values())):
        atoms_with_free_valence[i] = [
            atom.GetIdx() for atom in state_mol.GetAtoms() 
            if atom.GetNumImplicitHs() >= i
        ]
    
    actions_with_resources = []

    # 原子添加动作
    actions_with_resources.extend(
        _atom_addition_with_resource_info(
            state_mol, atom_types, atom_valences, atoms_with_free_valence, protect_initial))
    
    # 键添加动作
    actions_with_resources.extend(
        _bond_addition_with_resource_info(
            state_mol, atoms_with_free_valence, allowed_ring_sizes, 
            allow_bonds_between_rings, protect_initial))
    
    # 键删除动作
    if allow_removal:
        actions_with_resources.extend(
            _bond_removal_with_resource_info(state_mol, protect_initial))
    
    # 不修改动作
    if allow_no_modification:
        actions_with_resources.append({
            'smiles': Chem.MolToSmiles(state_mol),
            'mol': Chem.Mol(state_mol),
            'resource_cost': {'atom_type': None, 'operation_type': 'no_modification'}
        })

    return actions_with_resources

def _atom_addition_with_resource_info(state_mol, atom_types, atom_valences, 
                                     atoms_with_free_valence, protect_initial):
    """生成原子添加动作及其资源消耗信息"""
    bond_order = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }
    
    actions_with_resources = []
    
    for i in bond_order:
        for atom_idx in atoms_with_free_valence[i]:
                    
            for element in atom_types:
                if atom_valences[element] >= i:
                    new_mol = Chem.RWMol(state_mol)
                    idx = new_mol.AddAtom(Chem.Atom(element))
                    new_mol.AddBond(atom_idx, idx, bond_order[i])
                    
                    sanitization_result = Chem.SanitizeMol(new_mol, catchErrors=True)
                    if sanitization_result:
                        continue
                    
                    actions_with_resources.append({
                        'smiles': Chem.MolToSmiles(new_mol),
                        'mol': Chem.Mol(new_mol),
                        'resource_cost': {'atom_type': element, 'operation_type': None}
                    })
    
    return actions_with_resources

        



def _bond_addition_with_resource_info(state_mol, atoms_with_free_valence, allowed_ring_sizes,
                                     allow_bonds_between_rings, protect_initial):
    """生成键添加动作及其资源消耗信息"""
    bond_orders = [None, Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]
    actions_with_resources = []

    for valence, atoms in atoms_with_free_valence.items():
        for atom1, atom2 in itertools.combinations(atoms, 2):
            # 保护检查
            if protect_initial:
                if (is_atom_protected(state_mol, atom1) or 
                    is_atom_protected(state_mol, atom2)):
                    continue
                
                if is_bond_protected(state_mol, atom1, atom2):
                    continue
                
                if _would_form_ring_with_protected_atoms(state_mol, atom1, atom2):
                    continue
            
            bond = Chem.Mol(state_mol).GetBondBetweenAtoms(atom1, atom2)
            new_mol = Chem.RWMol(state_mol)
            Chem.Kekulize(new_mol, clearAromaticFlags=True)

            if bond is not None:
                if bond.GetBondType() not in bond_orders:
                    continue
                idx = bond.GetIdx()
                bond_order = bond_orders.index(bond.GetBondType())
                bond_order += valence
                if bond_order < len(bond_orders):
                    idx = bond.GetIdx()
                    bond.SetBondType(bond_orders[bond_order])
                    new_mol.ReplaceBond(idx, bond)
                else:
                    continue
            
            elif (not allow_bonds_between_rings and
                  (state_mol.GetAtomWithIdx(atom1).IsInRing() and
                   state_mol.GetAtomWithIdx(atom2).IsInRing())):
                continue
                
            elif (allowed_ring_sizes is not None and
                  len(Chem.rdmolops.GetShortestPath(state_mol, atom1, atom2)) not in allowed_ring_sizes):
                continue
            
            else:
                new_mol.AddBond(atom1, atom2, bond_orders[valence])
            
            sanitization_result = Chem.SanitizeMol(new_mol, catchErrors=True)
            if sanitization_result:
                continue
            
            actions_with_resources.append({
                'smiles': Chem.MolToSmiles(new_mol),
                'mol': Chem.Mol(new_mol),
                'resource_cost': {'atom_type': None, 'operation_type': 'bond_addition'}
            })

    return actions_with_resources


def _bond_removal_with_resource_info(state_mol, protect_initial):
    """生成键删除动作及其资源消耗信息"""
    bond_orders = [None, Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]
    actions_with_resources = []

    for valence in [1, 2, 3]:
        for bond in state_mol.GetBonds():
            bond = Chem.Mol(state_mol).GetBondBetweenAtoms(
                bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

            if protect_initial and bond.HasProp('_protected') and bond.GetBoolProp('_protected'):
                continue
            
            if bond.GetBondType() not in bond_orders:
                continue
                
            new_mol = Chem.RWMol(state_mol)
            Chem.Kekulize(new_mol, clearAromaticFlags=True)

            bond_order = bond_orders.index(bond.GetBondType())
            bond_order -= valence

            if bond_order > 0:
                idx = bond.GetIdx()
                bond.SetBondType(bond_orders[bond_order])
                new_mol.ReplaceBond(idx, bond)

                sanitization_result = Chem.SanitizeMol(new_mol, catchErrors=True)
                if sanitization_result:
                    continue
                
                actions_with_resources.append({
                    'smiles': Chem.MolToSmiles(new_mol),
                    'mol': Chem.Mol(new_mol),
                    'resource_cost': {'atom_type': None, 'operation_type': 'bond_removal'}
                })

            elif bond_order == 0:
                atom1 = bond.GetBeginAtom().GetIdx()
                atom2 = bond.GetEndAtom().GetIdx()
                new_mol.RemoveBond(atom1, atom2)

                sanitization_result = Chem.SanitizeMol(new_mol, catchErrors=True)
                if sanitization_result:
                    continue
                    
                smiles = Chem.MolToSmiles(new_mol)
                parts = sorted(smiles.split('.'), key=len)
                if len(parts) == 1 or len(parts[0]) == 1:
                    # 移除孤立原子
                    isolated = [atom.GetIdx() for atom in new_mol.GetAtoms() 
                              if atom.GetDegree() == 0]
                
                    for idx in sorted(isolated, reverse=True):
                        new_mol.RemoveAtom(idx)
                    
                    sanitization_result = Chem.SanitizeMol(new_mol, catchErrors=True)
                    if sanitization_result:
                        continue

                    if new_mol.GetNumAtoms() > 0:
                        actions_with_resources.append({
                            'smiles': Chem.MolToSmiles(new_mol),
                            'mol': Chem.Mol(new_mol),
                            'resource_cost': {'atom_type': None, 'operation_type': 'bond_removal'}
                        })

    return actions_with_resources

def _can_afford_resource_cost(resource_cost, resource_manager):
    """检查是否能承担指定的资源消耗"""
    atom_type = resource_cost.get('atom_type')
    operation_type = resource_cost.get('operation_type')
    
    if atom_type and not resource_manager.can_use_atom(atom_type):
        return False
    
    if operation_type and not resource_manager.can_use_operation(operation_type):
        return False
    
    return True

def get_protection_info(mol):
        """获取分子的保护信息用于调试"""
        if mol is None:
                return "Molecule is None"
        
        protected_atoms = []
        protected_bonds = []
        
        for atom in mol.GetAtoms():
                if atom.HasProp('_protected') and atom.GetBoolProp('_protected'):
                        protected_atoms.append(atom.GetIdx())
        
        for bond in mol.GetBonds():
                if bond.HasProp('_protected') and bond.GetBoolProp('_protected'):
                        protected_bonds.append(bond.GetIdx())
        
        return {
                'protected_atoms': protected_atoms,
                'protected_bonds': protected_bonds,
                'total_atoms': mol.GetNumAtoms(),
                'total_bonds': mol.GetNumBonds()
        }
class Molecule(object):
    """Defines the Markov decision process of generating a molecule."""

    def __init__(self,
                atom_types,
                init_mol=None,
                allow_removal=True,
                allow_no_modification=True,
                allow_bonds_between_rings=True,
                allowed_ring_sizes=None,
                max_steps=10,
                target_fn=None,
                record_path=False,
                protect_initial=True,
                # === 新增：资源限制参数 ===
                use_resource_limits=False,
                max_atom_uses=None,
                max_operation_uses=None,
                curriculum_learning=False,
                min_atom_uses=None,
                min_operation_uses=None):
        """Initializes the parameters for the MDP.

        Internal state will be stored as SMILES strings.

        Args:
            atom_types: The set of elements the molecule may contain.
            init_mol: String, Chem.Mol, or Chem.RWMol. If string is provided, it is
                considered as the SMILES string. The molecule to be set as the initial
                state. If None, an empty molecule will be created.
            allow_removal: Boolean. Whether to allow removal of a bond.
            allow_no_modification: Boolean. If true, the valid action set will
                include doing nothing to the current molecule, i.e., the current
                molecule itself will be added to the action set.
            allow_bonds_between_rings: Boolean. If False, new bonds connecting two
                atoms which are both in rings are not allowed.
                DANGER Set this to False will disable some of the transformations eg.
                c2ccc(Cc1ccccc1)cc2 -> c1ccc3c(c1)Cc2ccccc23
                But it will make the molecules generated make more sense chemically.
            allowed_ring_sizes: Set of integers or None. The size of the ring which
                is allowed to form. If None, all sizes will be allowed. If a set is
                provided, only sizes in the set is allowed.
            max_steps: Integer. The maximum number of steps to run.
            target_fn: A function or None. The function should have Args of a
                String, which is a SMILES string (the state), and Returns as
                a Boolean which indicates whether the input satisfies a criterion.
                If None, it will not be used as a criterion.
            record_path: Boolean. Whether to record the steps internally.
        """
        if isinstance(init_mol, str):
            init_mol = Chem.MolFromSmiles(init_mol)
        elif isinstance(init_mol, Chem.Mol):
            self.init_mol = init_mol
        self.init_mol = init_mol
        self.protect_initial = protect_initial
        self.atom_types = atom_types
        self.allow_removal = allow_removal
        self.allow_no_modification = allow_no_modification
        self.allow_bonds_between_rings = allow_bonds_between_rings
        self.allowed_ring_sizes = allowed_ring_sizes
        #self.max_steps = max_steps
        self.max_steps = max_steps

        self._state_mol = None  # 保存Mol对象
        self._state_smiles = None  # 保存SMILES用

        #self._valid_actions = []
        self._valid_actions_smiles = []  # 有效动作的SMILES列表
        self._valid_actions_mols = []    # 有效动
        # The status should be 'terminated' if initialize() is not called.
        self._counter = self.max_steps
        self._target_fn = target_fn
        self.record_path = record_path
        self._path = []
        self._max_bonds = 4
        atom_types = list(self.atom_types)
        self._max_new_bonds = dict(
                list(zip(atom_types, molecules.atom_valences(atom_types))))

        self.use_resource_limits = use_resource_limits
        self.resource_manager = None

        if self.use_resource_limits:
            from .resource_manager import ResourceManager
            
            # 设置默认值
            if max_atom_uses is None:
                max_atom_uses = {atom: 20 for atom in atom_types}
            if max_operation_uses is None:
                max_operation_uses = {
                    'bond_addition': 15,
                    'bond_removal': 10, 
                    'no_modification': 5
                }
            
            self.resource_manager = ResourceManager(
                atom_types=list(atom_types),
                max_atom_uses=max_atom_uses,
                max_operation_uses=max_operation_uses,
                # === 新增：课程学习参数 ===
                min_atom_uses=min_atom_uses,
                min_operation_uses=min_operation_uses,
                curriculum_learning=curriculum_learning
            )
        
        

    @property
    def state(self):
        return self._state_smiles
    
    @property
    def state_mol(self):
        """返回Mol对象状态"""
        return self._state_mol
    @property
    def num_steps_taken(self):
        return self._counter

    def get_path(self):
        return self._path

    def initialize(self, episode=None, total_episodes=None):
        """Resets the MDP to its initial state."""
        if self.init_mol is not None:
                # 创建初始分子的副本
                #self._state_mol = Chem.RWMol(self.init_mol)
                self._state_mol = Chem.Mol(self.init_mol)
                if self.protect_initial:
                        self._state_mol = protect_initial_molecule(self._state_mol)
                    
                self._state_smiles = Chem.MolToSmiles(self._state_mol)
        #self._state = self.init_mol
        else:
                self._state_mol = None
                self._state_smiles = None


        if self.record_path:
            self._path = [self._state_smiles]
        self._valid_actions = self.get_valid_actions(force_rebuild=True)
        #self._update_valid_actions()
        self._counter = 0

        if self.resource_manager:
            self.resource_manager.reset(episode=episode, total_episodes=total_episodes)
    
    def _update_valid_actions(self):
        """更新有效动作列表"""
        if self.use_resource_limits and self.resource_manager:
            # 使用资源限制的动作生成
            valid_smiles, valid_mols, valid_resource_costs = _get_valid_actions_with_resources(
                self._state_mol,
                atom_types=self.atom_types,
                allow_removal=self.allow_removal,
                allow_no_modification=self.allow_no_modification,
                allowed_ring_sizes=self.allowed_ring_sizes,
                allow_bonds_between_rings=self.allow_bonds_between_rings,
                protect_initial=self.protect_initial,
                resource_manager=self.resource_manager
            )
            # 将资源消耗信息存储为实例变量，供step函数使用
            self._valid_actions_resource_costs = valid_resource_costs
            self._valid_actions_smiles = valid_smiles
            self._valid_actions_mols = valid_mols
        else:
            # 原来的逻辑保持不变（需要实现原有的get_valid_actions_with_mols函数）
            raise NotImplementedError("Non-resource mode not implemented in this file")


    
    def get_valid_actions(self, state=None, force_rebuild=False):
                """获取有效动,(SMILES格式,保持兼容性)"""
                if force_rebuild or not self._valid_actions_smiles:
                        self._update_valid_actions()
                return copy.deepcopy(self._valid_actions_smiles)
        
            

    def _reward(self):
        """Gets the reward for the state.

        A child class can redefine the reward function if reward other than
        zero is desired.

        Returns:
            Float. The reward for the current state.
        """
        return 0.0

    def _goal_reached(self):
        """Sets the termination criterion for molecule Generation.

        A child class can define this function to terminate the MDP before
        max_steps is reached.

        Returns:
            Boolean, whether the goal is reached or not. If the goal is reached,
                the MDP is terminated.
        """
        if self._target_fn is None:
            return False
        return self._target_fn(self._state)

    def step(self, action):
        """Takes a step forward according to the action.

        Args:
            action: Chem.RWMol. The action is actually the target of the modification.

        Returns:
            results: Namedtuple containing the following fields:
                * state: The molecule reached after taking the action.
                * reward: The reward get after taking the action.
                * terminated: Whether this episode is terminated.

        Raises:
            ValueError: If the number of steps taken exceeds the preset max_steps, or
                the action is not in the set of valid_actions.

        """
        if self._counter >= self.max_steps or self._goal_reached():
            raise ValueError('This episode is terminated.')
        #if action not in self._valid_actions:
            #raise ValueError('Invalid action.')
        action_idx = self._valid_actions_smiles.index(action)
        selected_mol = self._valid_actions_mols[action_idx]
        #print(f"action: {action}, action_idx: {action_idx}, selected_mol: {selected_mol}")
        #self._state = action
        if self.use_resource_limits and self.resource_manager:
            resource_cost = self._valid_actions_resource_costs[action_idx]
            atom_type = resource_cost.get('atom_type')
            operation_type = resource_cost.get('operation_type')

            if atom_type:
                self.resource_manager.use_atom(atom_type)
            
            if operation_type:
                self.resource_manager.use_operation(operation_type)
            

        #self._state_mol = Chem.RWMol(selected_mol)
        self._state_mol = Chem.Mol(selected_mol)
        self._state_smiles = action

        if self.protect_initial:
                protection_info = get_protection_info(self._state_mol)
                #print(f" Step {self._counter + 1} 保护信息: {protection_info}")

        if self.record_path:
            self._path.append(self._state)
        #self._valid_actions = self.get_valid_actions(force_rebuild=True)
        self._update_valid_actions()
        self._counter += 1

        #valid_smiles = set(self._valid_actions_smiles)
        #print(f"Step {self._counter} 有效动作数量: {len(valid_smiles)}")
        #print(f"不重复的有效动作SMILES: {valid_smiles}")
        #print(f"Step {self._counter} 有效动作SMILES: {len(self._valid_actions_smiles)}")
        #print(f"包含重复的有效动作SMILES: {self._valid_actions_smiles}")
        #print(f"self._valid_actions_mols: {len(self._valid_actions_mols)}")
        #assert len(valid_smiles) == len(self._valid_actions_mols), \
                #'The valid actions SMILES and Mol lists should have the same length.'

        

        result = Result(
                #state=self._state,
                state=self._state_smiles,
                reward=self._reward(),
                terminated=(self._counter >= self.max_steps) or self._goal_reached())
        return result

    def visualize_state(self, state=None, **kwargs):
        """Draws the molecule of the state.

        Args:
            state: String, Chem.Mol, or Chem.RWMol. If string is prov ided, it is
                considered as the SMILES string. The state to query. If None, the
                current state will be considered.
            **kwargs: The keyword arguments passed to Draw.MolToImage.

        Returns:
            A PIL image containing a drawing of the molecule.
        """
        if state is None:
            state = self._state
        if isinstance(state, str):
            state = Chem.MolFromSmiles(state)
        return Draw.MolToImage(state, **kwargs)
    
    def get_resource_observation(self):
        """获取资源状态观察"""
        if self.use_resource_limits and self.resource_manager:
            return self.resource_manager.get_resource_vector()
        else:
            # 如果没有启用资源限制，返回空向量
            return [0] * (len(self.atom_types) + 3)