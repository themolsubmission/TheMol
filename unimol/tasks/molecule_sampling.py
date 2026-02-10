"""
Molecular Sampling Module for Flow-based Generative Model
Author: Claude
Date: 2025-01-20
Description: Improved sampling_data function for molecular generation during validation

Note: This module uses the same molecule reconstruction method as the original code
      (save_batch_to_sdf_reconstruct from unicore_task.py)
"""

import os
import sys
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from rdkit import Chem
from rdkit.Chem import AllChem

# Import original reconstruction function
from unicore.tasks.unicore_task import save_batch_to_sdf_reconstruct


class MoleculeSampler:
    """
    Sampler class for molecular generation

    Samples molecules using Flow-based model and saves to SDF files.
    Uses save_batch_to_sdf_reconstruct function from original unicore_task.py.
    """

    def __init__(
        self,
        vocab_dict: Optional[Dict[int, str]] = None,
        output_dir: str = "./generated_molecules",
        device: str = "cuda"
    ):
        """
        Args:
            vocab_dict: Atom type mapping dictionary (int -> atom_symbol)
            output_dir: Directory to save generated molecules
            device: Computation device ('cuda' or 'cpu')
        """
        self.vocab_dict = vocab_dict or self._create_default_vocab_dict()
        self.output_dir = output_dir
        self.device = device
        os.makedirs(output_dir, exist_ok=True)

    def _create_default_vocab_dict(self) -> Dict[int, str]:
        """Create default vocab dictionary"""
        # Common atom types
        vocab = {
            0: 'PAD',   # padding
            1: 'MASK',  # mask token
            2: 'UNK',   # unknown
            3: 'H',
            4: 'C',
            5: 'N',
            6: 'O',
            7: 'F',
            8: 'S',
            9: 'Cl',
            10: 'Br',
            11: 'I',
            12: 'P',
            13: 'Si',
            14: 'B',
            15: 'Na',
            16: 'K',
            17: 'Ca',
            18: 'Mg',
            19: 'Li',
        }
        return vocab

    def sample_molecules(
        self,
        model: torch.nn.Module,
        latent: torch.Tensor,
        num_atom: torch.Tensor,
        batch_size: int = 50,
        num_tokens: int = 56,
        emb_dim: int = 8,
        min_atoms: int = 20,
        max_atoms: int = 40,
        use_dopri5: bool = True,
        temperature: float = 1.0,
        optimize: bool = False,
    ) -> Tuple[List, Dict[str, float]]:
        """
        Sample molecules using Flow model

        Args:
            model: Trained model
            batch_size: Number of molecules to sample
            num_tokens: Maximum token (atom) count
            emb_dim: Embedding dimension
            min_atoms: Minimum atom count
            max_atoms: Maximum atom count
            use_dopri5: Whether to use DOPRI5 ODE solver (True: accurate, False: fast)
            temperature: Sampling temperature (higher increases diversity)

        Returns:
            mol_list: List of generated RDKit molecules [(mol, filepath), ...]
            stats: Generation statistics
        """
        model.eval()

        with torch.no_grad():
            # 1. Randomly sample atom count
            # num_atom = torch.randint(
            #     min_atoms,
            #     max_atoms + 1,
            #     (batch_size,),
            #     device=self.device
            # )

            # 2. Generate padding mask
            indices = torch.arange(num_tokens, device=self.device)
            num_atom_col = num_atom.unsqueeze(1)
            padding_mask = indices >= num_atom_col
            padding_mask[:, 0] = True  # First token is always padding (CLS token)
            # use_dopri5 = False
            # # 3. Latent sampling via flow matching
            # if use_dopri5:
            #     # DOPRI5: More accurate but slower
            #     sampling_dict = model.interpolant.sample_with_dopri5(
            #         batch_size,
            #         num_tokens,
            #         emb_dim,
            #         model.backbone,
            #         token_mask=~padding_mask
            #     )
            # else:
            #     # Euler method: Fast but less accurate
            #     sampling_dict = model.interpolant.sample(
            #         batch_size,
            #         num_tokens,
            #         emb_dim,
            #         model.backbone,
            #         token_mask=~padding_mask
            #     )
            if not optimize:
                sampling_dict = model.interpolant.sample(
                    batch_size,
                    num_tokens,
                    emb_dim,
                    model.backbone,
                    x_0=latent,
                    token_mask=~padding_mask
                )

                # 4. Extract latent from last timestep
                ode_latent = sampling_dict['tokens_traj'][-1]

            else:
                ode_latent = latent
            #print(ode_latent.device)
            # 5. Generate molecule through decoder
            decoder_output = model.dec(ode_latent, padding_mask)

            # Decoder output unpacking (may vary by model version)
            if len(decoder_output) >= 8:
                (
                    logits,
                    encoder_distance,
                    encoder_coord_,
                    encoder_x_norm,
                    decoder_x_norm,
                    delta_encoder_pair_rep_norm,
                    delta_decoder_pair_rep_norm,
                    _
                ) = decoder_output
            else:
                # Simple version
                logits = decoder_output[0]
                encoder_coord_ = decoder_output[2]

            # 6. Coordinate processing
            if isinstance(encoder_coord_, tuple) and len(encoder_coord_) == 2:
                encoder_coord, pred_bond = encoder_coord_
            else:
                encoder_coord = encoder_coord_
                pred_bond = None

            # 7. Extract only non-padding positions
            logits_flat = logits[~padding_mask]
            final_coords = encoder_coord[~padding_mask]

            # 8. Atom type prediction (temperature scaling)
            if temperature != 1.0:
                logits_flat = logits_flat / temperature
            final_type = [self.vocab_dict[i] for i in logits_flat.argmax(1).tolist()]
            final_atom_num = num_atom_col - 1  # Exclude CLS token

        # 9. Molecule reconstruction and saving (using original method)
        if hasattr(model, 'get_num_updates'):
            num_updates = model.get_num_updates()
            iteration = (num_updates + 1) if num_updates is not None else 0
        else:
            iteration = 0

        # Use original save_batch_to_sdf_reconstruct function
        try:
            mol_list = save_batch_to_sdf_reconstruct(
                final_type,
                final_atom_num,
                final_coords,
                iteration,
                output_dir=self.output_dir
            )

            # Calculate statistics
            batch_size = final_atom_num.shape[0]
            success_count = len([m for m in mol_list if m[0] is not None])

            stats = {
                'total': batch_size,
                'success': success_count,
                'success_rate': success_count / batch_size if batch_size > 0 else 0.0,
            }

            print(f"\n{'='*60}")
            print(f"Generation Summary (Iteration {iteration})")
            print(f"{'='*60}")
            print(f"Total molecules: {stats['total']}")
            print(f"Successful: {stats['success']} ({stats['success_rate']*100:.1f}%)")
            print(f"Failed: {stats['total'] - stats['success']}")
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"Error in molecule reconstruction: {e}")
            import traceback
            traceback.print_exc()
            mol_list = []
            stats = {
                'total': final_atom_num.shape[0] if hasattr(final_atom_num, 'shape') else 0,
                'success': 0,
                'success_rate': 0.0,
            }

        return mol_list, stats


# ============================================================================
# Utility Functions (Standalone)
# ============================================================================

def sample_molecules_from_model(
    model: torch.nn.Module,
    sample: Dict,
    vocab_dict: Optional[Dict] = None,
    output_dir: str = "./generated_molecules",
    batch_size: int = 50,
    num_tokens: int = 56,
    use_dopri5: bool = True
) -> List:
    """
    Sample molecules from model (simple functional interface)

    Args:
        model: Trained model
        sample: Input sample (needed for device info)
        vocab_dict: Atom type dictionary
        output_dir: Output directory
        batch_size: Number of molecules to sample
        num_tokens: Maximum token count
        use_dopri5: Whether to use DOPRI5 solver

    Returns:
        mol_list: List of generated molecules
    """
    device = sample['net_input']['src_tokens'].device

    sampler = MoleculeSampler(
        vocab_dict=vocab_dict,
        output_dir=output_dir,
        device=device
    )

    mol_list, stats = sampler.sample_molecules(
        model=model,
        batch_size=batch_size,
        num_tokens=num_tokens,
        use_dopri5=use_dopri5
    )

    return mol_list
