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
    SMILES로부터 결합 타입 정보를 추출하여 distance_target과 동일한 shape의
    결합 타입 행렬을 생성하는 데이터셋

    결합 타입 인코딩:
    - 0: 결합 없음
    - 1: 단일 결합 (Single bond)
    - 2: 이중 결합 (Double bond)
    - 3: 삼중 결합 (Triple bond)
    - 4: 방향족 결합 (Aromatic bond)
    """

    def __init__(self, raw_dataset, dataset, max_atoms=256, remove_hydrogen=True):
        """
        Args:
            raw_dataset: 원시 LMDB 데이터셋 (SMILES 포함)
            dataset: 기존 처리된 데이터셋
            max_atoms: 최대 원자 수
            remove_hydrogen: 수소 원자 제거 여부
        """
        super().__init__(dataset)
        self.raw_dataset = raw_dataset
        self.dataset = dataset
        self.max_atoms = max_atoms
        self.remove_hydrogen = remove_hydrogen

    def smiles_to_bond_matrix(self, smiles):
        """
        SMILES 문자열로부터 결합 타입 행렬 생성

        Args:
            smiles (str): SMILES 문자열

        Returns:
            np.ndarray: [max_atoms, max_atoms] 결합 타입 행렬
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros((self.max_atoms, self.max_atoms), dtype=np.int32)

            # 수소 원자 처리
            if not self.remove_hydrogen:
                mol = Chem.AddHs(mol)

            num_atoms = mol.GetNumAtoms()
            if num_atoms > self.max_atoms:
                num_atoms = self.max_atoms

            # 결합 타입 행렬 초기화
            bond_matrix = np.zeros((self.max_atoms, self.max_atoms), dtype=np.int32)

            # 결합 정보 추출
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

                # 인덱스 범위 체크
                if i >= self.max_atoms or j >= self.max_atoms:
                    continue

                # 결합 타입 변환
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
                    bond_value = 1  # 기타 결합은 단일결합으로 처리

                # 대칭 행렬 설정
                bond_matrix[i, j] = bond_value
                bond_matrix[j, i] = bond_value

            return bond_matrix

        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return np.zeros((self.max_atoms, self.max_atoms), dtype=np.int32)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        """
        데이터 아이템 가져오기 (결합 타입 정보 추가)

        Returns:
            dict: 기존 데이터 + 'bond_types' 키
        """
        # 기존 처리된 데이터 가져오기
        item = self.dataset[idx].copy()

        # 원시 데이터에서 SMILES 가져오기
        try:
            raw_item = self.raw_dataset[idx]
            smiles = raw_item.get('smi', '')

            # 결합 타입 행렬 생성
            bond_matrix = self.smiles_to_bond_matrix(smiles)
            item['bond_types'] = bond_matrix.astype(np.int32)

        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            # 오류 시 빈 행렬 생성
            item['bond_types'] = np.zeros((self.max_atoms, self.max_atoms), dtype=np.int32)

        return item

    def get_bond_type_stats(self, num_samples=1000):
        """
        결합 타입 통계 정보 반환 (디버깅용)

        Args:
            num_samples (int): 분석할 샘플 수

        Returns:
            dict: 결합 타입별 개수
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
    결합 타입 예측을 위한 손실 함수 (예시)
    """

    def __init__(self, bond_type_weights=None):
        """
        Args:
            bond_type_weights (dict): 각 결합 타입에 대한 가중치
        """
        self.bond_type_weights = bond_type_weights or {0: 0.1, 1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5}

    def compute_loss(self, pred_bond_types, target_bond_types, mask=None):
        """
        결합 타입 예측 손실 계산

        Args:
            pred_bond_types: [batch_size, max_atoms, max_atoms, num_bond_types] 예측값
            target_bond_types: [batch_size, max_atoms, max_atoms] 타겟 결합 타입
            mask: [batch_size, max_atoms, max_atoms] 마스크 (유효한 원자 쌍)

        Returns:
            torch.Tensor: 손실값
        """
        import torch.nn.functional as F

        batch_size, max_atoms, _, num_classes = pred_bond_types.shape

        # Reshape for cross entropy
        pred_flat = pred_bond_types.view(-1, num_classes)
        target_flat = target_bond_types.view(-1)

        # 마스크 적용
        if mask is not None:
            mask_flat = mask.view(-1)
            pred_flat = pred_flat[mask_flat]
            target_flat = target_flat[mask_flat]

        # Cross entropy loss
        loss = F.cross_entropy(pred_flat, target_flat, reduction='mean')

        return loss