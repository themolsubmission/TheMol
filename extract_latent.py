#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for extracting latent space from molecules in SDF files

Usage:
    python extract_latent.py --sdf_path /path/to/molecules.sdf --output_path /path/to/output.pkl

    # Single molecule
    python extract_latent.py --sdf_path molecule.sdf --output_path latent.pkl

    # All SDFs in directory
    python extract_latent.py --sdf_dir /path/to/sdf_dir --output_path latents.pkl
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem

# Unicore
from unicore.data import Dictionary

# Add project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unimol  # register models, tasks, losses


class MoleculeProcessor:
    """Read molecules from SDF files and convert to model input format"""

    def __init__(self, dictionary: Dictionary, max_atoms: int = 126, remove_hydrogen: bool = True):
        """
        Args:
            dictionary: Token dictionary
            max_atoms: Maximum number of atoms (max_seq_len=128 during training, 126 excluding BOS/EOS)
            remove_hydrogen: Whether to remove hydrogens (only_polar=0 during training -> True)
        """
        self.dictionary = dictionary
        self.max_atoms = max_atoms
        self.remove_hydrogen = remove_hydrogen

        # Special tokens
        self.pad_idx = self.dictionary.pad()
        self.bos_idx = self.dictionary.bos()
        self.eos_idx = self.dictionary.eos()

    def read_sdf(self, sdf_path: str) -> List[Chem.Mol]:
        """Read molecules from SDF file"""
        mols = []
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
        for mol in suppl:
            if mol is not None:
                mols.append(mol)
        return mols

    def mol_to_atoms_coords(self, mol: Chem.Mol) -> Tuple[List[str], np.ndarray]:
        """Extract atom types and coordinates from RDKit Mol object"""
        # Generate 3D coordinates if not present
        if mol.GetNumConformers() == 0:
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            if result == -1:  # Embedding failed
                # Fallback: try alternative coordinate generation
                result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv2())
            if result == -1:
                raise ValueError(f"Failed to generate 3D coordinates for molecule")
            AllChem.MMFFOptimizeMolecule(mol)

        conf = mol.GetConformer()
        atoms = []
        coords = []

        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()

            # Hydrogen removal option
            if self.remove_hydrogen and symbol == 'H':
                continue

            atoms.append(symbol)
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append([pos.x, pos.y, pos.z])

        return atoms, np.array(coords, dtype=np.float32)

    def normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        """Normalize coordinates (move center to origin)"""
        center = coords.mean(axis=0)
        return coords - center

    def tokenize(self, atoms: List[str]) -> np.ndarray:
        """Convert atom symbols to token indices"""
        tokens = []
        for atom in atoms:
            if atom in self.dictionary.indices:
                tokens.append(self.dictionary.index(atom))
            else:
                tokens.append(self.dictionary.unk())
        return np.array(tokens, dtype=np.int64)

    def compute_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """Compute distance matrix (using scipy as in training code)"""
        from scipy.spatial import distance_matrix
        return distance_matrix(coords, coords).astype(np.float32)

    def compute_edge_type(self, tokens: np.ndarray, vocab_size: int) -> np.ndarray:
        """Compute edge type (atom pair indices)"""
        n = len(tokens)
        edge_type = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            for j in range(n):
                edge_type[i, j] = tokens[i] * vocab_size + tokens[j]
        return edge_type

    def prepend_append(self, arr: np.ndarray, pre_val, app_val) -> np.ndarray:
        """Add values at the beginning and end of array (for BOS, EOS tokens)"""
        if arr.ndim == 1:
            return np.concatenate([[pre_val], arr, [app_val]])
        elif arr.ndim == 2:
            # For coordinates
            pre = np.array([[pre_val, pre_val, pre_val]], dtype=arr.dtype)
            app = np.array([[app_val, app_val, app_val]], dtype=arr.dtype)
            return np.concatenate([pre, arr, app], axis=0)
        return arr

    def pad_1d(self, arr: np.ndarray, max_len: int, pad_val) -> np.ndarray:
        """Pad 1D array"""
        if len(arr) >= max_len:
            return arr[:max_len]
        padded = np.full(max_len, pad_val, dtype=arr.dtype)
        padded[:len(arr)] = arr
        return padded

    def pad_2d(self, arr: np.ndarray, max_len: int, pad_val) -> np.ndarray:
        """Pad 2D array"""
        n = arr.shape[0]
        if n >= max_len:
            return arr[:max_len, :max_len]
        padded = np.full((max_len, max_len), pad_val, dtype=arr.dtype)
        padded[:n, :n] = arr
        return padded

    def pad_coords(self, coords: np.ndarray, max_len: int) -> np.ndarray:
        """Pad coordinates"""
        n = coords.shape[0]
        if n >= max_len:
            return coords[:max_len]
        padded = np.zeros((max_len, 3), dtype=coords.dtype)
        padded[:n] = coords
        return padded

    def process_molecule(self, mol: Chem.Mol, apply_padding: bool = True, max_seq_len: int = 128) -> Optional[Dict[str, torch.Tensor]]:
        """
        Convert a single molecule to model input format

        Args:
            mol: RDKit Mol object
            apply_padding: If True, pad to max_seq_len (for single inference),
                          If False, return actual length only (for batching)
            max_seq_len: Maximum sequence length for padding

        Returns:
            Model input dictionary
        """
        try:
            # Extract atoms and coordinates
            atoms, coords = self.mol_to_atoms_coords(mol)

            if len(atoms) == 0:
                return None

            # Limit maximum number of atoms
            if len(atoms) > self.max_atoms:
                atoms = atoms[:self.max_atoms]
                coords = coords[:self.max_atoms]

            # Normalize coordinates
            coords = self.normalize_coords(coords)

            # Tokenize
            tokens = self.tokenize(atoms)

            # Distance matrix
            distance = self.compute_distance_matrix(coords)

            # Edge type
            edge_type = self.compute_edge_type(tokens, len(self.dictionary))

            # Add BOS/EOS
            tokens_with_special = self.prepend_append(tokens, self.bos_idx, self.eos_idx)
            coords_with_special = self.prepend_append(coords, 0.0, 0.0)

            # Recompute distance matrix (including special tokens)
            n = len(coords_with_special)
            distance_with_special = np.zeros((n, n), dtype=np.float32)
            distance_with_special[1:-1, 1:-1] = distance

            # Recompute edge type
            edge_type_with_special = np.zeros((n, n), dtype=np.int64)
            edge_type_with_special[1:-1, 1:-1] = edge_type

            # Actual sequence length (BOS + atoms + EOS)
            seq_len = len(tokens_with_special)

            # Bond type (dummy)
            bond_type_with_special = np.zeros((n, n), dtype=np.int64)

            if apply_padding:
                # For single inference: pad to fixed length
                tokens_out = self.pad_1d(tokens_with_special, max_seq_len, self.pad_idx)
                coords_out = self.pad_coords(coords_with_special, max_seq_len)
                distance_out = self.pad_2d(distance_with_special, max_seq_len, 0)
                edge_type_out = self.pad_2d(edge_type_with_special, max_seq_len, 0)
                bond_type_out = np.zeros((max_seq_len, max_seq_len), dtype=np.int64)
            else:
                # For batching: return actual length only
                tokens_out = tokens_with_special
                coords_out = coords_with_special
                distance_out = distance_with_special
                edge_type_out = edge_type_with_special
                bond_type_out = bond_type_with_special

            return {
                'src_tokens': torch.from_numpy(tokens_out).long(),
                'src_coord': torch.from_numpy(coords_out).float(),
                'src_distance': torch.from_numpy(distance_out).float(),
                'src_edge_type': torch.from_numpy(edge_type_out).long(),
                'src_bond_type': torch.from_numpy(bond_type_out).long(),
                'seq_len': seq_len,
                'smiles': Chem.MolToSmiles(mol) if mol else None,
            }

        except Exception as e:
            print(f"Error processing molecule: {e}")
            return None


def collate_batch(batch_data: List[Dict[str, torch.Tensor]], pad_idx: int) -> Dict[str, torch.Tensor]:
    """
    Combine multiple molecule data into a batch (dynamic padding)

    Args:
        batch_data: List of results from process_molecule(apply_padding=False)
                   Each tensor has size equal to actual sequence length
        pad_idx: Padding index

    Returns:
        Batched tensor dictionary
        - src_tokens: [B, max_len]
        - src_coord: [B, max_len, 3]
        - src_distance: [B, max_len, max_len]
        - src_edge_type: [B, max_len, max_len]
        - src_bond_type: [B, max_len, max_len]
    """
    if not batch_data:
        return None

    # Find maximum sequence length in batch (based on actual tensor size)
    max_len = max(d['src_tokens'].shape[0] for d in batch_data)
    # Pad to multiple of 8 (GPU efficiency)
    if max_len % 8 != 0:
        max_len = ((max_len // 8) + 1) * 8

    batch_size = len(batch_data)

    # Initialize batch tensors
    src_tokens = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)
    src_coord = torch.zeros((batch_size, max_len, 3), dtype=torch.float)
    src_distance = torch.zeros((batch_size, max_len, max_len), dtype=torch.float)
    src_edge_type = torch.zeros((batch_size, max_len, max_len), dtype=torch.long)
    src_bond_type = torch.zeros((batch_size, max_len, max_len), dtype=torch.long)

    seq_lens = []
    smiles_list = []

    for i, data in enumerate(batch_data):
        seq_len = data['src_tokens'].shape[0]  # Use actual tensor size
        seq_lens.append(seq_len)
        smiles_list.append(data.get('smiles'))

        # Copy actual data
        src_tokens[i, :seq_len] = data['src_tokens']
        src_coord[i, :seq_len] = data['src_coord']
        src_distance[i, :seq_len, :seq_len] = data['src_distance']
        src_edge_type[i, :seq_len, :seq_len] = data['src_edge_type']
        src_bond_type[i, :seq_len, :seq_len] = data['src_bond_type']

    return {
        'src_tokens': src_tokens,
        'src_coord': src_coord,
        'src_distance': src_distance,
        'src_edge_type': src_edge_type,
        'src_bond_type': src_bond_type,
        'seq_lens': seq_lens,
        'smiles_list': smiles_list,
    }


class LatentExtractor:
    """Extract latent space from trained model"""

    def __init__(
        self,
        checkpoint_path: str,
        dictionary_path: str,
        device: str = 'cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dictionary = Dictionary.load(dictionary_path)

        # Add [MASK] token (added during training)
        self.mask_idx = self.dictionary.add_symbol("[MASK]", is_special=True)

        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        # Processor (using dictionary with MASK token added)
        self.processor = MoleculeProcessor(self.dictionary)

    def _load_model(self, checkpoint_path: str):
        """Load model from checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Restore args
        args = checkpoint['args']

        # Select model based on checkpoint path
        if "Dual" in checkpoint_path:
            from unimol.models.unimol_MAE_padding import UniMolOptimalPaddingModelDual
            model = UniMolOptimalPaddingModelDual(args, self.dictionary)
            print("Using UniMolOptimalPaddingModelDual")
        else:
            from unimol.models.unimol_MAE_padding import UniMolOptimalPaddingModel2
            model = UniMolOptimalPaddingModel2(args, self.dictionary)
            print("Using UniMolOptimalPaddingModel2")

        # Load weights
        missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
        if missing:
            print(f"Warning: {len(missing)} missing keys in state dict")
        if unexpected:
            print(f"Warning: {len(unexpected)} unexpected keys in state dict")

        model = model.to(self.device)

        print(f"Model loaded successfully. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model

    @torch.no_grad()
    def extract_latent(
        self,
        src_tokens: torch.Tensor,
        src_distance: torch.Tensor,
        src_coord: torch.Tensor,
        src_edge_type: torch.Tensor,
        src_bond_type: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Extract latent from model"""

        # Add batch dimension
        if src_tokens.dim() == 1:
            src_tokens = src_tokens.unsqueeze(0)
            src_distance = src_distance.unsqueeze(0)
            src_coord = src_coord.unsqueeze(0)
            src_edge_type = src_edge_type.unsqueeze(0)
            src_bond_type = src_bond_type.unsqueeze(0)

        # Move to device
        src_tokens = src_tokens.to(self.device)
        src_distance = src_distance.to(self.device)
        src_coord = src_coord.to(self.device)
        src_edge_type = src_edge_type.to(self.device)
        src_bond_type = src_bond_type.to(self.device)

        # Convert BOS(1) and EOS(2) to PAD(0) (same as during training)
        src_tokens = src_tokens.clone()
        src_tokens[src_tokens == 1] = 0  # BOS -> PAD
        src_tokens[src_tokens == 2] = 0  # EOS -> PAD

        # Rotation normalization
        rotated_coord, rot_mat = self.model.rot(
            src_tokens=src_tokens,
            src_distance=src_distance,
            src_coord=src_coord,
            src_edge_type=src_edge_type,
            src_bond_type=src_bond_type,
        )

        # Encoding
        latent_mean, latent_logvar, encoder_x_norm, delta_pair_norm = self.model.enc(
            src_tokens=src_tokens,
            src_distance=src_distance,
            src_coord=rotated_coord,
            src_edge_type=src_edge_type,
            src_bond_type=src_bond_type,
        )

        # Compute standard deviation
        latent_logvar = torch.clamp(latent_logvar, -15, 15)
        latent_std = torch.exp(0.5 * latent_logvar)

        # Sampling (reparameterization trick)
        z = latent_mean + latent_std * torch.randn_like(latent_std)

        # Padding mask
        padding_mask = src_tokens.eq(self.dictionary.pad())

        return {
            'latent_mean': latent_mean.cpu(),      # [B, N, 8] - latent mean
            'latent_std': latent_std.cpu(),        # [B, N, 8] - latent standard deviation
            'latent_z': z.cpu(),                   # [B, N, 8] - sampled latent vector
            'padding_mask': padding_mask.cpu(),    # [B, N] - padding mask
            'rot_mat': rot_mat.cpu(),              # [B, 3, 3] - rotation matrix
        }

    def extract_from_sdf(
        self,
        sdf_path: str,
        aggregate: str = 'mean'  # 'mean', 'sum', 'cls', 'all'
    ) -> List[Dict]:
        """Extract latent from all molecules in SDF file"""

        mols = self.processor.read_sdf(sdf_path)
        print(f"Loaded {len(mols)} molecules from {sdf_path}")

        results = []
        for i, mol in enumerate(tqdm(mols, desc="Extracting latents")):
            processed = self.processor.process_molecule(mol)
            if processed is None:
                results.append(None)
                continue

            latent_dict = self.extract_latent(
                src_tokens=processed['src_tokens'],
                src_distance=processed['src_distance'],
                src_coord=processed['src_coord'],
                src_edge_type=processed['src_edge_type'],
                src_bond_type=processed['src_bond_type'],
            )

            # Select only actual atoms excluding padding, BOS, EOS (same as during training)
            # In training code, BOS(1), EOS(2) are converted to PAD(0) before loss calculation
            src_tokens = processed['src_tokens']
            atom_mask = (src_tokens != 0) & (src_tokens != 1) & (src_tokens != 2)  # Actual atoms only

            latent_mean = latent_dict['latent_mean'][0]  # [N, 8]
            latent_z = latent_dict['latent_z'][0]  # [N, 8]
            latent_std = latent_dict['latent_std'][0]  # [N, 8]

            if aggregate == 'mean':
                # Mean of actual atoms
                aggregated_mean = latent_mean[atom_mask].mean(dim=0)  # [8]
                aggregated_z = latent_z[atom_mask].mean(dim=0)  # [8]
            elif aggregate == 'sum':
                aggregated_mean = latent_mean[atom_mask].sum(dim=0)
                aggregated_z = latent_z[atom_mask].sum(dim=0)
            elif aggregate == 'cls':
                # Use first token (BOS) - Note: BOS does not contribute to loss during training
                aggregated_mean = latent_mean[0]
                aggregated_z = latent_z[0]
            else:  # 'all'
                aggregated_mean = latent_mean[atom_mask]
                aggregated_z = latent_z[atom_mask]

            results.append({
                'smiles': processed['smiles'],
                'latent_mean': aggregated_mean.numpy(),
                'latent_z': aggregated_z.numpy(),
                'latent_mean_full': latent_mean[atom_mask].numpy(),  # Latent of actual atoms
                'latent_std_full': latent_std[atom_mask].numpy(),
                'num_atoms': atom_mask.sum().item(),  # Number of actual atoms
            })

        return results

    def extract_from_mol(
        self,
        mol: Chem.Mol,
        aggregate: str = 'mean'
    ) -> Optional[Dict]:
        """Extract latent from a single RDKit Mol object"""
        processed = self.processor.process_molecule(mol)
        if processed is None:
            return None

        latent_dict = self.extract_latent(
            src_tokens=processed['src_tokens'],
            src_distance=processed['src_distance'],
            src_coord=processed['src_coord'],
            src_edge_type=processed['src_edge_type'],
            src_bond_type=processed['src_bond_type'],
        )

        # Select only actual atoms excluding padding, BOS, EOS (same as during training)
        src_tokens = processed['src_tokens']
        atom_mask = (src_tokens != 0) & (src_tokens != 1) & (src_tokens != 2)

        latent_mean = latent_dict['latent_mean'][0]
        latent_z = latent_dict['latent_z'][0]
        latent_std = latent_dict['latent_std'][0]

        if aggregate == 'mean':
            aggregated_mean = latent_mean[atom_mask].mean(dim=0)
            aggregated_z = latent_z[atom_mask].mean(dim=0)
        elif aggregate == 'sum':
            aggregated_mean = latent_mean[atom_mask].sum(dim=0)
            aggregated_z = latent_z[atom_mask].sum(dim=0)
        elif aggregate == 'cls':
            # Note: BOS does not contribute to loss during training
            aggregated_mean = latent_mean[0]
            aggregated_z = latent_z[0]
        else:  # 'all'
            aggregated_mean = latent_mean[atom_mask]
            aggregated_z = latent_z[atom_mask]

        return {
            'smiles': processed['smiles'],
            'latent_mean': aggregated_mean.numpy(),
            'latent_z': aggregated_z.numpy(),
            'latent_mean_full': latent_mean[atom_mask].numpy(),
            'latent_std_full': latent_std[atom_mask].numpy(),
            'num_atoms': atom_mask.sum().item(),
        }

    def extract_from_smiles(
        self,
        smiles: str,
        aggregate: str = 'mean'
    ) -> Optional[Dict]:
        """Extract latent from SMILES string"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Failed to parse SMILES: {smiles}")
            return None

        # Generate 3D coordinates (without hydrogens)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)  # Remove hydrogens (same as during training)

        return self.extract_from_mol(mol, aggregate)

    @torch.no_grad()
    def extract_latent_batch(
        self,
        src_tokens: torch.Tensor,
        src_distance: torch.Tensor,
        src_coord: torch.Tensor,
        src_edge_type: torch.Tensor,
        src_bond_type: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Extract latent from model in batch"""

        # Move to device
        src_tokens = src_tokens.to(self.device)
        src_distance = src_distance.to(self.device)
        src_coord = src_coord.to(self.device)
        src_edge_type = src_edge_type.to(self.device)
        src_bond_type = src_bond_type.to(self.device)

        # Convert BOS(1) and EOS(2) to PAD(0) (same as during training)
        src_tokens = src_tokens.clone()
        src_tokens[src_tokens == 1] = 0  # BOS -> PAD
        src_tokens[src_tokens == 2] = 0  # EOS -> PAD

        # Rotation normalization
        rotated_coord, rot_mat = self.model.rot(
            src_tokens=src_tokens,
            src_distance=src_distance,
            src_coord=src_coord,
            src_edge_type=src_edge_type,
            src_bond_type=src_bond_type,
        )

        # Encoding
        latent_mean, latent_logvar, encoder_x_norm, delta_pair_norm = self.model.enc(
            src_tokens=src_tokens,
            src_distance=src_distance,
            src_coord=rotated_coord,
            src_edge_type=src_edge_type,
            src_bond_type=src_bond_type,
        )

        # Compute standard deviation
        latent_logvar = torch.clamp(latent_logvar, -15, 15)
        latent_std = torch.exp(0.5 * latent_logvar)

        # Sampling (reparameterization trick)
        z = latent_mean + latent_std * torch.randn_like(latent_std)

        # Padding mask
        padding_mask = src_tokens.eq(self.dictionary.pad())

        return {
            'latent_mean': latent_mean.cpu(),      # [B, N, 8]
            'latent_std': latent_std.cpu(),        # [B, N, 8]
            'latent_z': z.cpu(),                   # [B, N, 8]
            'padding_mask': padding_mask.cpu(),    # [B, N]
            'rot_mat': rot_mat.cpu(),              # [B, 3, 3]
            'src_tokens': src_tokens.cpu(),        # [B, N] - for atom mask
        }

    def extract_from_sdf_list_batch(
        self,
        sdf_paths: List[str],
        aggregate: str = 'mean',
        batch_size: int = 32
    ) -> List[Optional[Dict]]:
        """
        Extract latent from multiple SDF files in batch (true GPU batch processing)

        Args:
            sdf_paths: List of SDF file paths
            aggregate: Aggregation method ('mean', 'sum', 'cls', 'all')
            batch_size: Batch size

        Returns:
            List of result dictionaries for each SDF
        """
        # Preprocess all molecules
        all_processed = []
        valid_indices = []

        for i, sdf_path in enumerate(sdf_paths):
            try:
                mols = self.processor.read_sdf(sdf_path)
                if mols and len(mols) > 0:
                    # apply_padding=False: return actual length only for dynamic padding in batch
                    processed = self.processor.process_molecule(mols[0], apply_padding=False)
                    if processed is not None:
                        all_processed.append(processed)
                        valid_indices.append(i)
                    else:
                        all_processed.append(None)
                else:
                    all_processed.append(None)
            except Exception:
                all_processed.append(None)

        # Extract valid molecules only
        valid_processed = [p for p in all_processed if p is not None]

        if not valid_processed:
            return [None] * len(sdf_paths)

        # Batch processing
        all_latent_results = []

        for batch_start in range(0, len(valid_processed), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_processed))
            batch_data = valid_processed[batch_start:batch_end]

            # Create batch
            collated = collate_batch(batch_data, self.dictionary.pad())

            # Batch processing on GPU
            batch_results = self.extract_latent_batch(
                src_tokens=collated['src_tokens'],
                src_distance=collated['src_distance'],
                src_coord=collated['src_coord'],
                src_edge_type=collated['src_edge_type'],
                src_bond_type=collated['src_bond_type'],
            )

            # Split batch results into individual results
            for j in range(len(batch_data)):
                src_tokens = batch_results['src_tokens'][j]
                latent_mean = batch_results['latent_mean'][j]
                latent_z = batch_results['latent_z'][j]
                latent_std = batch_results['latent_std'][j]

                # Select only actual atoms excluding padding, BOS, EOS
                atom_mask = (src_tokens != 0) & (src_tokens != 1) & (src_tokens != 2)

                if aggregate == 'mean':
                    aggregated_mean = latent_mean[atom_mask].mean(dim=0)
                    aggregated_z = latent_z[atom_mask].mean(dim=0)
                elif aggregate == 'sum':
                    aggregated_mean = latent_mean[atom_mask].sum(dim=0)
                    aggregated_z = latent_z[atom_mask].sum(dim=0)
                elif aggregate == 'cls':
                    aggregated_mean = latent_mean[0]
                    aggregated_z = latent_z[0]
                else:  # 'all'
                    aggregated_mean = latent_mean[atom_mask]
                    aggregated_z = latent_z[atom_mask]

                all_latent_results.append({
                    'smiles': collated['smiles_list'][j],
                    'latent_mean': aggregated_mean.numpy(),
                    'latent_z': aggregated_z.numpy(),
                    'latent_mean_full': latent_mean[atom_mask].numpy(),
                    'latent_std_full': latent_std[atom_mask].numpy(),
                    'num_atoms': atom_mask.sum().item(),
                })

        # Reconstruct results in original order
        results = [None] * len(sdf_paths)
        for i, idx in enumerate(valid_indices):
            if i < len(all_latent_results):
                results[idx] = all_latent_results[i]

        return results


def main():
    parser = argparse.ArgumentParser(description='Extract latent space from SDF files')
    parser.add_argument('--sdf_path', type=str, help='Path to SDF file')
    parser.add_argument('--sdf_dir', type=str, help='Directory containing SDF files')
    parser.add_argument('--sdf_list', type=str, help='Text file containing list of SDF file paths (one per line)')
    parser.add_argument('--smiles', type=str, help='SMILES string')
    parser.add_argument('--output_path', type=str, required=True, help='Output pickle path')
    parser.add_argument('--checkpoint_path', type=str,
                        required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dict_path', type=str,
                        default='./data/dict.txt',
                        help='Path to dictionary file')
    parser.add_argument('--aggregate', type=str, default='mean',
                        choices=['mean', 'sum', 'cls', 'all'],
                        help='Aggregation method for atom-level latents')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Input validation
    if not any([args.sdf_path, args.sdf_dir, args.sdf_list, args.smiles]):
        parser.error('At least one of --sdf_path, --sdf_dir, --sdf_list, or --smiles is required')

    # Initialize extractor
    extractor = LatentExtractor(
        checkpoint_path=args.checkpoint_path,
        dictionary_path=args.dict_path,
        device=args.device,
    )

    results = {}

    # Process SDF file
    if args.sdf_path:
        print(f"\nProcessing SDF file: {args.sdf_path}")
        results['molecules'] = extractor.extract_from_sdf(args.sdf_path, args.aggregate)

    # Process SDF directory
    if args.sdf_dir:
        print(f"\nProcessing SDF directory: {args.sdf_dir}")
        sdf_files = list(Path(args.sdf_dir).glob('*.sdf'))
        results['files'] = {}
        for sdf_file in tqdm(sdf_files, desc="Processing files"):
            results['files'][str(sdf_file)] = extractor.extract_from_sdf(
                str(sdf_file), args.aggregate
            )

    # Process SDF list file (batch mode)
    if args.sdf_list:
        print(f"\nProcessing SDF list file: {args.sdf_list}")
        with open(args.sdf_list, 'r') as f:
            sdf_files = [line.strip() for line in f if line.strip()]

        print(f"Found {len(sdf_files)} SDF files in list")
        results['files'] = {}

        for sdf_file in tqdm(sdf_files, desc="Processing files"):
            if not os.path.exists(sdf_file):
                print(f"Warning: File not found: {sdf_file}")
                results['files'][sdf_file] = None
                continue

            extracted = extractor.extract_from_sdf(sdf_file, args.aggregate)
            # extract_from_sdf returns a list, use first element for single file
            if extracted and len(extracted) > 0:
                results['files'][sdf_file] = extracted[0]
            else:
                results['files'][sdf_file] = None

    # Process SMILES
    if args.smiles:
        print(f"\nProcessing SMILES: {args.smiles}")
        results['smiles_result'] = extractor.extract_from_smiles(args.smiles, args.aggregate)

    # Save results
    print(f"\nSaving results to {args.output_path}")
    with open(args.output_path, 'wb') as f:
        pickle.dump(results, f)

    print("Done!")

    # Brief result summary
    if 'molecules' in results:
        valid_count = sum(1 for r in results['molecules'] if r is not None)
        print(f"Successfully processed {valid_count}/{len(results['molecules'])} molecules")
        if valid_count > 0:
            first_valid = next(r for r in results['molecules'] if r is not None)
            print(f"Latent dimension: {first_valid['latent_mean'].shape}")

    if 'files' in results:
        valid_count = sum(1 for r in results['files'].values() if r is not None)
        print(f"Successfully processed {valid_count}/{len(results['files'])} files")


if __name__ == '__main__':
    main()
