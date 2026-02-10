# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from rdkit import Chem
from functools import lru_cache
from unicore.data import BaseWrapperDataset


class BondTypeDataset(BaseWrapperDataset):
    """
    Dataset that extracts bond type information from SMILES and generates
    bond type matrix with the same shape as distance_target

    Bond type encoding:
    - 0: No bond
    - 1: Single bond
    - 2: Double bond
    - 3: Triple bond
    - 4: Aromatic bond
    """

    def __init__(self, raw_dataset, dataset, max_atoms=256, remove_hydrogen=True):
        """
        Args:
            raw_dataset: Raw LMDB dataset (containing SMILES)
            dataset: Pre-processed dataset
            max_atoms: Maximum number of atoms
            remove_hydrogen: Whether to remove hydrogen atoms
        """
        super().__init__(dataset)
        self.raw_dataset = raw_dataset
        self.dataset = dataset
        self.max_atoms = max_atoms
        self.remove_hydrogen = remove_hydrogen

    def smiles_to_bond_matrix(self, smiles):
        """
        Generate bond type matrix from SMILES string

        Args:
            smiles (str): SMILES string

        Returns:
            np.ndarray: [max_atoms, max_atoms] bond type matrix
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros((self.max_atoms, self.max_atoms), dtype=np.int32)

            # Handle hydrogen atoms
            if not self.remove_hydrogen:
                mol = Chem.AddHs(mol)

            num_atoms = mol.GetNumAtoms()
            if num_atoms > self.max_atoms:
                num_atoms = self.max_atoms

            # Initialize bond type matrix
            bond_matrix = np.zeros((self.max_atoms, self.max_atoms), dtype=np.int32)

            # Extract bond information
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

                # Check index range
                if i >= self.max_atoms or j >= self.max_atoms:
                    continue

                # Convert bond type
                bond_type = bond.GetBondType()

                if bond_type == Chem.rdchem.BondType.SINGLE:
                    bond_value = 1
                elif bond_type == Chem.rdchem.BondType.DOUBLE:
                    bond_value = 2
                elif bond_type == Chem.rdchem.BondType.TRIPLE:
                    bond_value = 3
                elif bond_type == Chem.rdchem.BondType.AROMATIC:
                    bond_value = 4
                else:
                    bond_value = 1  # Treat other bonds as single bond

                # Set symmetric matrix
                bond_matrix[i, j] = bond_value
                bond_matrix[j, i] = bond_value

            return bond_matrix

        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return np.zeros((self.max_atoms, self.max_atoms), dtype=np.int32)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        """
        Get data item (with bond type information added)

        Returns:
            dict: Existing data + 'bond_types' key
        """
        # Get pre-processed data
        item = self.dataset[idx].copy()

        # Get SMILES from raw data
        try:
            raw_item = self.raw_dataset[idx]
            smiles = raw_item.get('smi', '')

            # Generate bond type matrix
            bond_matrix = self.smiles_to_bond_matrix(smiles)
            item['bond_types'] = bond_matrix.astype(np.int32)

        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            # Generate empty matrix on error
            item['bond_types'] = np.zeros((self.max_atoms, self.max_atoms), dtype=np.int32)

        return item

    def get_bond_type_stats(self, num_samples=1000):
        """
        Return bond type statistics (for debugging)

        Args:
            num_samples (int): Number of samples to analyze

        Returns:
            dict: Count per bond type
        """
        bond_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

        max_samples = min(num_samples, len(self.dataset))

        for i in range(max_samples):
            try:
                item = self[i]
                bond_matrix = item['bond_types']

                unique, counts = np.unique(bond_matrix, return_counts=True)
                for bond_type, count in zip(unique, counts):
                    if bond_type in bond_counts:
                        bond_counts[bond_type] += count

            except Exception as e:
                print(f"Error in sample {i}: {e}")
                continue

        return bond_counts


class BondTypeLoss:
    """
    Loss function for bond type prediction (example)
    """

    def __init__(self, bond_type_weights=None):
        """
        Args:
            bond_type_weights (dict): Weights for each bond type
        """
        self.bond_type_weights = bond_type_weights or {0: 0.1, 1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5}

    def compute_loss(self, pred_bond_types, target_bond_types, mask=None):
        """
        Compute bond type prediction loss

        Args:
            pred_bond_types: [batch_size, max_atoms, max_atoms, num_bond_types] predictions
            target_bond_types: [batch_size, max_atoms, max_atoms] target bond types
            mask: [batch_size, max_atoms, max_atoms] mask (valid atom pairs)

        Returns:
            torch.Tensor: Loss value
        """
        import torch.nn.functional as F

        batch_size, max_atoms, _, num_classes = pred_bond_types.shape

        # Reshape for cross entropy
        pred_flat = pred_bond_types.view(-1, num_classes)
        target_flat = target_bond_types.view(-1)

        # Apply mask
        if mask is not None:
            mask_flat = mask.view(-1)
            pred_flat = pred_flat[mask_flat]
            target_flat = target_flat[mask_flat]

        # Cross entropy loss
        loss = F.cross_entropy(pred_flat, target_flat, reduction='mean')

        return loss