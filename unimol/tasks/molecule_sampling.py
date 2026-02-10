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
sys.path.insert(0, '/home/csy/anaconda3/envs/lf_cfm_cma/lib/python3.9/site-packages')
from unicore.tasks.unicore_task import save_batch_to_sdf_reconstruct


class MoleculeSampler:
    """
    분자 생성을 위한 샘플러 클래스

    Flow-based 모델을 사용하여 분자를 샘플링하고 SDF 파일로 저장합니다.
    원본 unicore_task.py의 save_batch_to_sdf_reconstruct 함수를 사용합니다.
    """

    def __init__(
        self,
        vocab_dict: Optional[Dict[int, str]] = None,
        output_dir: str = "./generated_molecules",
        device: str = "cuda"
    ):
        """
        Args:
            vocab_dict: 원자 타입 매핑 딕셔너리 (int -> atom_symbol)
            output_dir: 생성된 분자를 저장할 디렉토리
            device: 연산 디바이스 ('cuda' or 'cpu')
        """
        self.vocab_dict = vocab_dict or self._create_default_vocab_dict()
        self.output_dir = output_dir
        self.device = device
        os.makedirs(output_dir, exist_ok=True)

    def _create_default_vocab_dict(self) -> Dict[int, str]:
        """기본 vocab dictionary 생성"""
        # 일반적인 원자 타입들
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
        Flow 모델을 사용하여 분자 샘플링

        Args:
            model: 학습된 모델
            batch_size: 샘플링할 분자 개수
            num_tokens: 최대 토큰(원자) 개수
            emb_dim: Embedding 차원
            min_atoms: 최소 원자 개수
            max_atoms: 최대 원자 개수
            use_dopri5: DOPRI5 ODE solver 사용 여부 (True: 정확, False: 빠름)
            temperature: 샘플링 온도 (높을수록 다양성 증가)

        Returns:
            mol_list: 생성된 RDKit 분자 리스트 [(mol, filepath), ...]
            stats: 생성 통계
        """
        model.eval()

        with torch.no_grad():
            # 1. 랜덤하게 원자 개수 샘플링
            # num_atom = torch.randint(
            #     min_atoms,
            #     max_atoms + 1,
            #     (batch_size,),
            #     device=self.device
            # )

            # 2. Padding mask 생성
            indices = torch.arange(num_tokens, device=self.device)
            num_atom_col = num_atom.unsqueeze(1)
            padding_mask = indices >= num_atom_col
            padding_mask[:, 0] = True  # 첫 번째 토큰은 항상 패딩 (CLS token)
            # use_dopri5 = False
            # # 3. Flow matching을 통한 latent 샘플링
            # if use_dopri5:
            #     # DOPRI5: 더 정확하지만 느림
            #     sampling_dict = model.interpolant.sample_with_dopri5(
            #         batch_size,
            #         num_tokens,
            #         emb_dim,
            #         model.backbone,
            #         token_mask=~padding_mask
            #     )
            # else:
            #     # Euler method: 빠르지만 덜 정확
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

                # 4. 마지막 타임스텝의 latent 추출
                ode_latent = sampling_dict['tokens_traj'][-1]

            else:
                ode_latent = latent
            #print(ode_latent.device)
            # 5. Decoder를 통해 분자 생성
            decoder_output = model.dec(ode_latent, padding_mask)

            # Decoder output unpacking (모델 버전에 따라 다를 수 있음)
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
                # 간단한 버전
                logits = decoder_output[0]
                encoder_coord_ = decoder_output[2]

            # 6. Coordinate 처리
            if isinstance(encoder_coord_, tuple) and len(encoder_coord_) == 2:
                encoder_coord, pred_bond = encoder_coord_
            else:
                encoder_coord = encoder_coord_
                pred_bond = None

            # 7. Non-padding 위치의 데이터만 추출
            logits_flat = logits[~padding_mask]
            final_coords = encoder_coord[~padding_mask]

            # 8. 원자 타입 예측 (temperature scaling)
            if temperature != 1.0:
                logits_flat = logits_flat / temperature
            final_type = [self.vocab_dict[i] for i in logits_flat.argmax(1).tolist()]
            final_atom_num = num_atom_col - 1  # CLS token 제외

        # 9. 분자 reconstruction 및 저장 (원본 방식 사용)
        if hasattr(model, 'get_num_updates'):
            num_updates = model.get_num_updates()
            iteration = (num_updates + 1) if num_updates is not None else 0
        else:
            iteration = 0

        # 원본 save_batch_to_sdf_reconstruct 함수 사용
        try:
            mol_list = save_batch_to_sdf_reconstruct(
                final_type,
                final_atom_num,
                final_coords,
                iteration,
                output_dir=self.output_dir
            )

            # 통계 계산
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
    모델로부터 분자 샘플링 (간단한 함수형 인터페이스)

    Args:
        model: 학습된 모델
        sample: 입력 샘플 (device 정보를 위해 필요)
        vocab_dict: 원자 타입 딕셔너리
        output_dir: 출력 디렉토리
        batch_size: 샘플링할 분자 개수
        num_tokens: 최대 토큰 개수
        use_dopri5: DOPRI5 solver 사용 여부

    Returns:
        mol_list: 생성된 분자 리스트
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
