#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SDF 파일에서 분자의 Latent Space를 추출하는 스크립트

Usage:
    python extract_latent.py --sdf_path /path/to/molecules.sdf --output_path /path/to/output.pkl

    # 단일 분자
    python extract_latent.py --sdf_path molecule.sdf --output_path latent.pkl

    # 디렉토리 내 모든 SDF
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

# 프로젝트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unimol  # register models, tasks, losses


class MoleculeProcessor:
    """SDF 파일에서 분자를 읽고 모델 입력 형태로 변환"""

    def __init__(self, dictionary: Dictionary, max_atoms: int = 126, remove_hydrogen: bool = True):
        """
        Args:
            dictionary: 토큰 딕셔너리
            max_atoms: 최대 원자 수 (학습 시 max_seq_len=128, BOS/EOS 제외하면 126)
            remove_hydrogen: 수소 제거 여부 (학습 시 only_polar=0 → True)
        """
        self.dictionary = dictionary
        self.max_atoms = max_atoms
        self.remove_hydrogen = remove_hydrogen

        # 특수 토큰
        self.pad_idx = self.dictionary.pad()
        self.bos_idx = self.dictionary.bos()
        self.eos_idx = self.dictionary.eos()

    def read_sdf(self, sdf_path: str) -> List[Chem.Mol]:
        """SDF 파일에서 분자들을 읽음"""
        mols = []
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
        for mol in suppl:
            if mol is not None:
                mols.append(mol)
        return mols

    def mol_to_atoms_coords(self, mol: Chem.Mol) -> Tuple[List[str], np.ndarray]:
        """RDKit Mol 객체에서 원자 타입과 좌표 추출"""
        # 3D 좌표가 없으면 생성
        if mol.GetNumConformers() == 0:
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            if result == -1:  # 임베딩 실패
                # 대안: 랜덤 좌표 생성 시도
                result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv2())
            if result == -1:
                raise ValueError(f"Failed to generate 3D coordinates for molecule")
            AllChem.MMFFOptimizeMolecule(mol)

        conf = mol.GetConformer()
        atoms = []
        coords = []

        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()

            # 수소 제거 옵션
            if self.remove_hydrogen and symbol == 'H':
                continue

            atoms.append(symbol)
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append([pos.x, pos.y, pos.z])

        return atoms, np.array(coords, dtype=np.float32)

    def normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        """좌표 정규화 (중심을 원점으로 이동)"""
        center = coords.mean(axis=0)
        return coords - center

    def tokenize(self, atoms: List[str]) -> np.ndarray:
        """원자 심볼을 토큰 인덱스로 변환"""
        tokens = []
        for atom in atoms:
            if atom in self.dictionary.indices:
                tokens.append(self.dictionary.index(atom))
            else:
                tokens.append(self.dictionary.unk())
        return np.array(tokens, dtype=np.int64)

    def compute_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """거리 행렬 계산 (학습 코드와 동일하게 scipy 사용)"""
        from scipy.spatial import distance_matrix
        return distance_matrix(coords, coords).astype(np.float32)

    def compute_edge_type(self, tokens: np.ndarray, vocab_size: int) -> np.ndarray:
        """엣지 타입 계산 (원자쌍 인덱스)"""
        n = len(tokens)
        edge_type = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            for j in range(n):
                edge_type[i, j] = tokens[i] * vocab_size + tokens[j]
        return edge_type

    def prepend_append(self, arr: np.ndarray, pre_val, app_val) -> np.ndarray:
        """배열 앞뒤에 값 추가 (BOS, EOS 토큰용)"""
        if arr.ndim == 1:
            return np.concatenate([[pre_val], arr, [app_val]])
        elif arr.ndim == 2:
            # 좌표의 경우
            pre = np.array([[pre_val, pre_val, pre_val]], dtype=arr.dtype)
            app = np.array([[app_val, app_val, app_val]], dtype=arr.dtype)
            return np.concatenate([pre, arr, app], axis=0)
        return arr

    def pad_1d(self, arr: np.ndarray, max_len: int, pad_val) -> np.ndarray:
        """1D 배열 패딩"""
        if len(arr) >= max_len:
            return arr[:max_len]
        padded = np.full(max_len, pad_val, dtype=arr.dtype)
        padded[:len(arr)] = arr
        return padded

    def pad_2d(self, arr: np.ndarray, max_len: int, pad_val) -> np.ndarray:
        """2D 배열 패딩"""
        n = arr.shape[0]
        if n >= max_len:
            return arr[:max_len, :max_len]
        padded = np.full((max_len, max_len), pad_val, dtype=arr.dtype)
        padded[:n, :n] = arr
        return padded

    def pad_coords(self, coords: np.ndarray, max_len: int) -> np.ndarray:
        """좌표 패딩"""
        n = coords.shape[0]
        if n >= max_len:
            return coords[:max_len]
        padded = np.zeros((max_len, 3), dtype=coords.dtype)
        padded[:n] = coords
        return padded

    def process_molecule(self, mol: Chem.Mol, apply_padding: bool = True, max_seq_len: int = 128) -> Optional[Dict[str, torch.Tensor]]:
        """
        단일 분자를 모델 입력 형태로 변환

        Args:
            mol: RDKit Mol 객체
            apply_padding: True면 max_seq_len으로 패딩 (단일 추론용),
                          False면 패딩 없이 실제 길이만 반환 (배치용)
            max_seq_len: 패딩 시 최대 시퀀스 길이

        Returns:
            모델 입력 딕셔너리
        """
        try:
            # 원자와 좌표 추출
            atoms, coords = self.mol_to_atoms_coords(mol)

            if len(atoms) == 0:
                return None

            # 최대 원자 수 제한
            if len(atoms) > self.max_atoms:
                atoms = atoms[:self.max_atoms]
                coords = coords[:self.max_atoms]

            # 좌표 정규화
            coords = self.normalize_coords(coords)

            # 토큰화
            tokens = self.tokenize(atoms)

            # 거리 행렬
            distance = self.compute_distance_matrix(coords)

            # 엣지 타입
            edge_type = self.compute_edge_type(tokens, len(self.dictionary))

            # BOS/EOS 추가
            tokens_with_special = self.prepend_append(tokens, self.bos_idx, self.eos_idx)
            coords_with_special = self.prepend_append(coords, 0.0, 0.0)

            # 거리 행렬 재계산 (special 토큰 포함)
            n = len(coords_with_special)
            distance_with_special = np.zeros((n, n), dtype=np.float32)
            distance_with_special[1:-1, 1:-1] = distance

            # 엣지 타입 재계산
            edge_type_with_special = np.zeros((n, n), dtype=np.int64)
            edge_type_with_special[1:-1, 1:-1] = edge_type

            # 실제 시퀀스 길이 (BOS + atoms + EOS)
            seq_len = len(tokens_with_special)

            # 결합 타입 (더미)
            bond_type_with_special = np.zeros((n, n), dtype=np.int64)

            if apply_padding:
                # 단일 추론용: 고정 길이로 패딩
                tokens_out = self.pad_1d(tokens_with_special, max_seq_len, self.pad_idx)
                coords_out = self.pad_coords(coords_with_special, max_seq_len)
                distance_out = self.pad_2d(distance_with_special, max_seq_len, 0)
                edge_type_out = self.pad_2d(edge_type_with_special, max_seq_len, 0)
                bond_type_out = np.zeros((max_seq_len, max_seq_len), dtype=np.int64)
            else:
                # 배치용: 패딩 없이 실제 길이만
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
    여러 분자 데이터를 배치로 결합 (동적 패딩)

    Args:
        batch_data: process_molecule(apply_padding=False)의 결과 리스트
                   각 텐서는 실제 시퀀스 길이만큼의 크기를 가짐
        pad_idx: 패딩 인덱스

    Returns:
        배치화된 텐서 딕셔너리
        - src_tokens: [B, max_len]
        - src_coord: [B, max_len, 3]
        - src_distance: [B, max_len, max_len]
        - src_edge_type: [B, max_len, max_len]
        - src_bond_type: [B, max_len, max_len]
    """
    if not batch_data:
        return None

    # 배치 내 최대 시퀀스 길이 찾기 (실제 텐서 크기 기준)
    max_len = max(d['src_tokens'].shape[0] for d in batch_data)
    # 8의 배수로 패딩 (GPU 효율성)
    if max_len % 8 != 0:
        max_len = ((max_len // 8) + 1) * 8

    batch_size = len(batch_data)

    # 배치 텐서 초기화
    src_tokens = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)
    src_coord = torch.zeros((batch_size, max_len, 3), dtype=torch.float)
    src_distance = torch.zeros((batch_size, max_len, max_len), dtype=torch.float)
    src_edge_type = torch.zeros((batch_size, max_len, max_len), dtype=torch.long)
    src_bond_type = torch.zeros((batch_size, max_len, max_len), dtype=torch.long)

    seq_lens = []
    smiles_list = []

    for i, data in enumerate(batch_data):
        seq_len = data['src_tokens'].shape[0]  # 실제 텐서 크기 사용
        seq_lens.append(seq_len)
        smiles_list.append(data.get('smiles'))

        # 실제 데이터 복사
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
    """학습된 모델에서 Latent Space 추출"""

    def __init__(
        self,
        checkpoint_path: str,
        dictionary_path: str,
        device: str = 'cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dictionary = Dictionary.load(dictionary_path)

        # [MASK] 토큰 추가 (학습 시 추가됨)
        self.mask_idx = self.dictionary.add_symbol("[MASK]", is_special=True)

        # 모델 로드
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        # 프로세서 (MASK 토큰이 추가된 딕셔너리 사용)
        self.processor = MoleculeProcessor(self.dictionary)

    def _load_model(self, checkpoint_path: str):
        """체크포인트에서 모델 로드"""
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Args 복원
        args = checkpoint['args']

        # 체크포인트 경로에 따라 모델 선택
        if "Dual" in checkpoint_path:
            from unimol.models.unimol_MAE_padding import UniMolOptimalPaddingModelDual
            model = UniMolOptimalPaddingModelDual(args, self.dictionary)
            print("Using UniMolOptimalPaddingModelDual")
        else:
            from unimol.models.unimol_MAE_padding import UniMolOptimalPaddingModel2
            model = UniMolOptimalPaddingModel2(args, self.dictionary)
            print("Using UniMolOptimalPaddingModel2")

        # 가중치 로드
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
        """모델에서 latent 추출"""

        # 배치 차원 추가
        if src_tokens.dim() == 1:
            src_tokens = src_tokens.unsqueeze(0)
            src_distance = src_distance.unsqueeze(0)
            src_coord = src_coord.unsqueeze(0)
            src_edge_type = src_edge_type.unsqueeze(0)
            src_bond_type = src_bond_type.unsqueeze(0)

        # 디바이스로 이동
        src_tokens = src_tokens.to(self.device)
        src_distance = src_distance.to(self.device)
        src_coord = src_coord.to(self.device)
        src_edge_type = src_edge_type.to(self.device)
        src_bond_type = src_bond_type.to(self.device)

        # BOS(1)와 EOS(2)를 PAD(0)로 변환 (학습 시와 동일하게)
        src_tokens = src_tokens.clone()
        src_tokens[src_tokens == 1] = 0  # BOS → PAD
        src_tokens[src_tokens == 2] = 0  # EOS → PAD

        # 회전 정규화
        rotated_coord, rot_mat = self.model.rot(
            src_tokens=src_tokens,
            src_distance=src_distance,
            src_coord=src_coord,
            src_edge_type=src_edge_type,
            src_bond_type=src_bond_type,
        )

        # 인코딩
        latent_mean, latent_logvar, encoder_x_norm, delta_pair_norm = self.model.enc(
            src_tokens=src_tokens,
            src_distance=src_distance,
            src_coord=rotated_coord,
            src_edge_type=src_edge_type,
            src_bond_type=src_bond_type,
        )

        # 표준편차 계산
        latent_logvar = torch.clamp(latent_logvar, -15, 15)
        latent_std = torch.exp(0.5 * latent_logvar)

        # 샘플링 (reparameterization trick)
        z = latent_mean + latent_std * torch.randn_like(latent_std)

        # Padding mask
        padding_mask = src_tokens.eq(self.dictionary.pad())

        return {
            'latent_mean': latent_mean.cpu(),      # [B, N, 8] - 잠재 평균
            'latent_std': latent_std.cpu(),        # [B, N, 8] - 잠재 표준편차
            'latent_z': z.cpu(),                   # [B, N, 8] - 샘플링된 잠재 벡터
            'padding_mask': padding_mask.cpu(),    # [B, N] - 패딩 마스크
            'rot_mat': rot_mat.cpu(),              # [B, 3, 3] - 회전 행렬
        }

    def extract_from_sdf(
        self,
        sdf_path: str,
        aggregate: str = 'mean'  # 'mean', 'sum', 'cls', 'all'
    ) -> List[Dict]:
        """SDF 파일에서 모든 분자의 latent 추출"""

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

            # 패딩, BOS, EOS 제외하고 실제 원자만 선택 (학습 시와 동일)
            # 학습 코드에서 BOS(1), EOS(2)를 PAD(0)로 변경 후 Loss 계산
            src_tokens = processed['src_tokens']
            atom_mask = (src_tokens != 0) & (src_tokens != 1) & (src_tokens != 2)  # 실제 원자만

            latent_mean = latent_dict['latent_mean'][0]  # [N, 8]
            latent_z = latent_dict['latent_z'][0]  # [N, 8]
            latent_std = latent_dict['latent_std'][0]  # [N, 8]

            if aggregate == 'mean':
                # 실제 원자들의 평균
                aggregated_mean = latent_mean[atom_mask].mean(dim=0)  # [8]
                aggregated_z = latent_z[atom_mask].mean(dim=0)  # [8]
            elif aggregate == 'sum':
                aggregated_mean = latent_mean[atom_mask].sum(dim=0)
                aggregated_z = latent_z[atom_mask].sum(dim=0)
            elif aggregate == 'cls':
                # 첫 번째 토큰 (BOS) 사용 - 주의: BOS는 학습 시 Loss에 기여하지 않음
                aggregated_mean = latent_mean[0]
                aggregated_z = latent_z[0]
            else:  # 'all'
                aggregated_mean = latent_mean[atom_mask]
                aggregated_z = latent_z[atom_mask]

            results.append({
                'smiles': processed['smiles'],
                'latent_mean': aggregated_mean.numpy(),
                'latent_z': aggregated_z.numpy(),
                'latent_mean_full': latent_mean[atom_mask].numpy(),  # 실제 원자의 latent
                'latent_std_full': latent_std[atom_mask].numpy(),
                'num_atoms': atom_mask.sum().item(),  # 실제 원자 수
            })

        return results

    def extract_from_mol(
        self,
        mol: Chem.Mol,
        aggregate: str = 'mean'
    ) -> Optional[Dict]:
        """단일 RDKit Mol 객체에서 latent 추출"""
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

        # 패딩, BOS, EOS 제외하고 실제 원자만 선택 (학습 시와 동일)
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
            # 주의: BOS는 학습 시 Loss에 기여하지 않음
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
        """SMILES 문자열에서 latent 추출"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Failed to parse SMILES: {smiles}")
            return None

        # 3D 좌표 생성 (수소 없이)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)  # 수소 제거 (학습 시와 동일하게)

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
        """배치 단위로 모델에서 latent 추출"""

        # 디바이스로 이동
        src_tokens = src_tokens.to(self.device)
        src_distance = src_distance.to(self.device)
        src_coord = src_coord.to(self.device)
        src_edge_type = src_edge_type.to(self.device)
        src_bond_type = src_bond_type.to(self.device)

        # BOS(1)와 EOS(2)를 PAD(0)로 변환 (학습 시와 동일하게)
        src_tokens = src_tokens.clone()
        src_tokens[src_tokens == 1] = 0  # BOS → PAD
        src_tokens[src_tokens == 2] = 0  # EOS → PAD

        # 회전 정규화
        rotated_coord, rot_mat = self.model.rot(
            src_tokens=src_tokens,
            src_distance=src_distance,
            src_coord=src_coord,
            src_edge_type=src_edge_type,
            src_bond_type=src_bond_type,
        )

        # 인코딩
        latent_mean, latent_logvar, encoder_x_norm, delta_pair_norm = self.model.enc(
            src_tokens=src_tokens,
            src_distance=src_distance,
            src_coord=rotated_coord,
            src_edge_type=src_edge_type,
            src_bond_type=src_bond_type,
        )

        # 표준편차 계산
        latent_logvar = torch.clamp(latent_logvar, -15, 15)
        latent_std = torch.exp(0.5 * latent_logvar)

        # 샘플링 (reparameterization trick)
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
        여러 SDF 파일에서 배치로 latent 추출 (진정한 GPU 배치 처리)

        Args:
            sdf_paths: SDF 파일 경로 리스트
            aggregate: 집계 방식 ('mean', 'sum', 'cls', 'all')
            batch_size: 배치 크기

        Returns:
            각 SDF에 대한 결과 딕셔너리 리스트
        """
        # 모든 분자 전처리
        all_processed = []
        valid_indices = []

        for i, sdf_path in enumerate(sdf_paths):
            try:
                mols = self.processor.read_sdf(sdf_path)
                if mols and len(mols) > 0:
                    # apply_padding=False: 배치에서 동적 패딩을 위해 실제 길이만 반환
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

        # 유효한 분자만 추출
        valid_processed = [p for p in all_processed if p is not None]

        if not valid_processed:
            return [None] * len(sdf_paths)

        # 배치 처리
        all_latent_results = []

        for batch_start in range(0, len(valid_processed), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_processed))
            batch_data = valid_processed[batch_start:batch_end]

            # 배치 생성
            collated = collate_batch(batch_data, self.dictionary.pad())

            # GPU에서 배치 처리
            batch_results = self.extract_latent_batch(
                src_tokens=collated['src_tokens'],
                src_distance=collated['src_distance'],
                src_coord=collated['src_coord'],
                src_edge_type=collated['src_edge_type'],
                src_bond_type=collated['src_bond_type'],
            )

            # 배치 결과를 개별 결과로 분리
            for j in range(len(batch_data)):
                src_tokens = batch_results['src_tokens'][j]
                latent_mean = batch_results['latent_mean'][j]
                latent_z = batch_results['latent_z'][j]
                latent_std = batch_results['latent_std'][j]

                # 패딩, BOS, EOS 제외하고 실제 원자만 선택
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

        # 결과를 원래 순서로 재구성
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

    # 입력 검증
    if not any([args.sdf_path, args.sdf_dir, args.sdf_list, args.smiles]):
        parser.error('At least one of --sdf_path, --sdf_dir, --sdf_list, or --smiles is required')

    # 추출기 초기화
    extractor = LatentExtractor(
        checkpoint_path=args.checkpoint_path,
        dictionary_path=args.dict_path,
        device=args.device,
    )

    results = {}

    # SDF 파일 처리
    if args.sdf_path:
        print(f"\nProcessing SDF file: {args.sdf_path}")
        results['molecules'] = extractor.extract_from_sdf(args.sdf_path, args.aggregate)

    # SDF 디렉토리 처리
    if args.sdf_dir:
        print(f"\nProcessing SDF directory: {args.sdf_dir}")
        sdf_files = list(Path(args.sdf_dir).glob('*.sdf'))
        results['files'] = {}
        for sdf_file in tqdm(sdf_files, desc="Processing files"):
            results['files'][str(sdf_file)] = extractor.extract_from_sdf(
                str(sdf_file), args.aggregate
            )

    # SDF 리스트 파일 처리 (배치 모드)
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
            # extract_from_sdf는 리스트를 반환, 단일 파일이면 첫 번째 요소 사용
            if extracted and len(extracted) > 0:
                results['files'][sdf_file] = extracted[0]
            else:
                results['files'][sdf_file] = None

    # SMILES 처리
    if args.smiles:
        print(f"\nProcessing SMILES: {args.smiles}")
        results['smiles_result'] = extractor.extract_from_smiles(args.smiles, args.aggregate)

    # 결과 저장
    print(f"\nSaving results to {args.output_path}")
    with open(args.output_path, 'wb') as f:
        pickle.dump(results, f)

    print("Done!")

    # 간단한 결과 요약
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
