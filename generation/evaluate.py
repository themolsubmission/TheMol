#!/usr/bin/env python3
"""
TheMol 모델로 분자 생성 후 평가 지표 계산
train_finetuning_geom.sh로 학습한 모델 사용

평가 지표 (FlowMol 호환):
1. Validity: RDKit sanitization 성공률, connectivity
2. Stability (Bond-based): unicore/eval.py의 bond-based stability
3. Stability (Valence-based): MiDi valence table 기반 stability
4. REOS: Structural alerts (Glaxo, Dundee rules)
5. Ring Systems: OOD ring system 비율
6. Energy: MMFF energy JS divergence
7. PoseBusters: 3D 구조 품질 검증
8. Geometry: xTB 최적화 기반 energy gain, RMSD, MMFF drop
"""

import os
import sys
import torch
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from collections import defaultdict

# TheMol 프로젝트 경로 추가
sys.path.insert(0, '/home/csy/work1/3D/TheMol/unimol')

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle


# ============================================================
# MiDi Valence Table (FlowMol에서 사용하는 것과 동일)
# ============================================================
MIDI_VALENCE_TABLE = {
    'H': {0: 1, 1: 0, -1: 0},
    'C': {0: [3, 4], 1: 3, -1: 3},
    'N': {0: [2, 3], 1: [2, 3, 4], -1: 2},
    'O': {0: 2, 1: 3, -1: 1},
    'F': {0: 1, -1: 0},
    'B': 3, 'Al': 3, 'Si': 4,
    'P': {0: [3, 5], 1: 4},
    'S': {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
    'Cl': 1, 'As': 3,
    'Br': {0: 1, 1: 2}, 'I': 1, 'Hg': [1, 2], 'Bi': [3, 5], 'Se': [2, 4, 6]
}


# ============================================================
# REOS Structural Alerts (FlowMol 호환 - CSV 파일에서 로드)
# ============================================================
# CSV 파일 경로 (useful_rdkit_utils와 동일한 alert_collection.csv)
REOS_RULES_CSV_PATH = Path('/home/csy/work1/3D/TheMol/data/reos_alert_collection.csv')
REOS_RULES_CSV_URL = 'https://raw.githubusercontent.com/PatWalters/rd_filters/master/rd_filters/data/alert_collection.csv'


def load_reos_rules(active_rules=["Glaxo", "Dundee"]):
    """REOS 규칙을 CSV에서 로드 (FlowMol/useful_rdkit_utils와 동일)

    Returns:
        DataFrame with columns: rule_set_name, description, smarts, max, pat (compiled)
        flag_header: list of flag names in format "{rule_set_name}::{description}"
    """
    import subprocess as sp

    # CSV 파일이 없으면 다운로드
    if not REOS_RULES_CSV_PATH.exists():
        print(f"Downloading REOS rules from {REOS_RULES_CSV_URL}...")
        REOS_RULES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        result = sp.run(f"wget -O {REOS_RULES_CSV_PATH} {REOS_RULES_CSV_URL}",
                       shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download REOS rules: {result.stderr}")

    # CSV 로드
    df = pd.read_csv(REOS_RULES_CSV_PATH)

    # active_rules 필터링
    df = df[df['rule_set_name'].isin(active_rules)].copy()

    # SMARTS 컴파일
    pat_list = []
    valid_rows = []
    for idx, row in df.iterrows():
        pat = Chem.MolFromSmarts(row['smarts'])
        if pat is not None:
            pat_list.append(pat)
            valid_rows.append(idx)

    df = df.loc[valid_rows].copy()
    df['pat'] = pat_list

    # flag header 생성 (FlowMol 형식)
    df['flag_name'] = df['rule_set_name'] + '::' + df['description']

    # 정렬
    df = df.sort_values('flag_name').reset_index(drop=True)

    return df


# ============================================================
# Model Loading Functions
# ============================================================
def load_dictionary(dict_path):
    """dictionary 로드"""
    from unicore.data import Dictionary

    dictionary = Dictionary(
        pad="[PAD]",
        bos="[CLS]",
        eos="[SEP]",
        unk="[UNK]",
        extra_special_symbols=["[MASK]"]
    )

    dictionary.add_symbol("[PAD]")
    dictionary.add_symbol("[CLS]")
    dictionary.add_symbol("[SEP]")
    dictionary.add_symbol("[UNK]")
    dictionary.add_symbol("[MASK]", is_special=True)

    with open(dict_path, 'r') as f:
        for line in f:
            symbol = line.strip()
            if symbol and symbol not in dictionary:
                dictionary.add_symbol(symbol)

    return dictionary


def load_model(checkpoint_path, device='cuda', dict_path='/home/csy/work1/3D/TheMol/GEOM_dataset/dict.txt'):
    """학습된 모델 로드"""
    from unicore import checkpoint_utils
    from unicore.models import register_model
    from models.unimol_MAE_padding import UniMolOptimalPaddingModelDual

    dictionary = load_dictionary(dict_path)
    print(f"Loaded dictionary with {len(dictionary)} symbols")

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = state['args']

    model = UniMolOptimalPaddingModelDual(args, dictionary)
    model.load_state_dict(state['model'], strict=False)
    model = model.to(device)
    model.eval()

    if hasattr(model, 'interpolant'):
        model.interpolant.device = device
        print(f"Updated interpolant device to {device}")

    print(f"Loaded model from {checkpoint_path}")
    print(f"Training step: {state.get('num_updates', 'unknown')}")

    return model, args


def create_vocab_dict(dict_path='/home/csy/work1/3D/TheMol/GEOM_dataset/dict.txt'):
    """원자 타입 vocab 딕셔너리 생성"""
    vocab = []
    with open(dict_path, 'r') as f:
        for line in f:
            symbol = line.strip()
            if symbol:
                vocab.append(symbol)

    if '[MASK]' not in vocab:
        vocab.insert(4, '[MASK]')

    return {i: v for i, v in enumerate(vocab)}


# ============================================================
# Molecule Sampling
# ============================================================
def sample_molecules(model, num_samples=100, batch_size=50, device='cuda'):
    """Flow Matching 모델로 분자 샘플링"""
    from unicore.tasks.unicore_task import save_batch_to_sdf_reconstruct

    vocab_dict = create_vocab_dict()
    all_mols = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_samples - len(all_mols))
        print(f"\nBatch {batch_idx + 1}/{num_batches}: Generating {current_batch_size} molecules...")

        with torch.no_grad():
            num_tokens = 56
            emb_dim = 8

            num_atom = torch.randint(10, 40, (current_batch_size,), device=device)
            indices = torch.arange(num_tokens, device=device)
            num_atom_col = num_atom.unsqueeze(1)
            padding_mask = indices >= num_atom_col
            padding_mask[:, 0] = True

            sampling_dict = model.interpolant.sample(
                current_batch_size, num_tokens, emb_dim,
                model.backbone, token_mask=~padding_mask
            )
            ode_latent = sampling_dict['tokens_traj'][-1]

            (
                logits, encoder_distance, encoder_coord_,
                encoder_x_norm, decoder_x_norm,
                delta_encoder_pair_rep_norm, delta_decoder_pair_rep_norm, _
            ) = model.dec(ode_latent, padding_mask)

            if len(encoder_coord_) == 2:
                encoder_coord, pred_bond = encoder_coord_
            elif len(encoder_coord_) == 3:
                encoder_coord, pred_bond, pred_coord_ = encoder_coord_
                
            else:
                encoder_coord = encoder_coord_

            final_coords = encoder_coord[~padding_mask]
            logits_valid = logits[~padding_mask]
            predicted_types = [vocab_dict[i] for i in logits_valid.argmax(1).tolist()]

            special_tokens = {'[PAD]', '[CLS]', '[SEP]', '[UNK]', '[MASK]'}
            final_type = []
            for t in predicted_types:
                if t in special_tokens:
                    final_type.append('C')
                else:
                    final_type.append(t)

            final_atom_num = num_atom_col - 1

            try:
                mol_list = save_batch_to_sdf_reconstruct(
                    final_type, final_atom_num, final_coords,
                    iteration=batch_idx,
                    output_dir=f"/tmp/themol_samples"
                )
                for mol_info in mol_list:
                    if mol_info and mol_info[0] is not None:
                        try:
                            Chem.SanitizeMol(mol_info[0])
                            smi = Chem.MolToSmiles(mol_info[0])
                            if not '.' in smi:
                                all_mols.append(mol_info[0])
                        except:
                            continue
            except Exception as e:
                print(f"Error in molecule reconstruction: {e}")

    print(f"\nGenerated {len(all_mols)} valid molecules out of {num_samples} attempts")
    return all_mols


                    #     Chem.SanitizeMol(mol)
                    #     #Chem.RemoveHs(mol)
                    #     smi = Chem.MolToSmiles(mol)
                    #     #print(smi)
                    #     if not '.' in smi:
                    #         success += 1
                    #         final_mol.append((mol, path))
                    # except Exception as e:


# ============================================================
# Validity Metrics
# ============================================================
def compute_validity(mol_list):
    """RDKit 기반 validity 계산"""
    n_valid = 0
    n_connected = 0
    num_components = []
    frag_fracs = []
    error_counts = {'valid': 0, 'disconnected': 0, 'valence': 0, 'kekulization': 0, 'other': 0}

    for mol in mol_list:
        try:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            num_components.append(len(mol_frags))

            if len(mol_frags) > 1:
                error_counts['disconnected'] += 1
            else:
                n_connected += 1

            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            largest_frag_frac = largest_mol.GetNumAtoms() / mol.GetNumAtoms()
            frag_fracs.append(largest_frag_frac)

            Chem.SanitizeMol(largest_mol)
            smiles = Chem.MolToSmiles(largest_mol)
            n_valid += 1
            error_counts['valid'] += 1

        except Chem.rdchem.AtomValenceException:
            error_counts['valence'] += 1
        except Chem.rdchem.KekulizeException:
            error_counts['kekulization'] += 1
        except Exception:
            error_counts['other'] += 1

    results = {
        'frac_valid_mols': n_valid / len(mol_list) if mol_list else 0,
        'frac_connected': n_connected / len(mol_list) if mol_list else 0,
        'avg_frag_frac': sum(frag_fracs) / len(frag_fracs) if frag_fracs else 0,
        'avg_num_components': sum(num_components) / len(num_components) if num_components else 0,
        'error_counts': error_counts
    }

    return results


# ============================================================
# Bond-based Stability (unicore/eval.py)
# ============================================================
def compute_bond_stability(mol_list):
    """unicore/eval.py의 bond-based stability 계산"""
    from unicore.eval import analyze_stability_for_rdkit_mols

    validity_dict, details = analyze_stability_for_rdkit_mols(
        mol_list,
        dataset_info=None,
        debug=False,
        verbose=True
    )

    return {
        'mol_stable': validity_dict['mol_stable'],
        'atm_stable': validity_dict['atm_stable']
    }


# ============================================================
# Valence-based Stability (MiDi style)
# ============================================================
def get_atom_valency(mol, atom_idx):
    """원자의 valency (결합 수) 계산"""
    atom = mol.GetAtomWithIdx(atom_idx)
    valency = 0
    for bond in atom.GetBonds():
        bond_type = bond.GetBondType()
        if bond_type == Chem.rdchem.BondType.SINGLE:
            valency += 1
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            valency += 2
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            valency += 3
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            valency += 1.5
    return int(valency)


def check_valence_stability(mol, valence_table=MIDI_VALENCE_TABLE):
    """MiDi valence table 기반 stability 체크"""
    n_stable_atoms = 0
    n_atoms = mol.GetNumAtoms()

    for atom_idx in range(n_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        atom_symbol = atom.GetSymbol()
        charge = atom.GetFormalCharge()
        valency = get_atom_valency(mol, atom_idx)

        if atom_symbol not in valence_table:
            continue

        possible_bonds = valence_table[atom_symbol]

        if isinstance(possible_bonds, int):
            is_stable = (possible_bonds == valency)
        elif isinstance(possible_bonds, dict):
            if charge in possible_bonds:
                expected = possible_bonds[charge]
            else:
                expected = possible_bonds.get(0, possible_bonds[list(possible_bonds.keys())[0]])

            if isinstance(expected, int):
                is_stable = (expected == valency)
            else:
                is_stable = (valency in expected)
        elif isinstance(possible_bonds, list):
            is_stable = (valency in possible_bonds)
        else:
            is_stable = False

        if is_stable:
            n_stable_atoms += 1

    mol_stable = (n_stable_atoms == n_atoms)
    return n_stable_atoms, mol_stable, n_atoms


def compute_valence_stability(mol_list):
    """Valence-based stability 계산 (FlowMol 스타일)"""
    n_atoms_total = 0
    n_stable_atoms_total = 0
    n_stable_molecules = 0

    for mol in mol_list:
        try:
            n_stable, mol_stable, n_atoms = check_valence_stability(mol)
            n_atoms_total += n_atoms
            n_stable_atoms_total += n_stable
            n_stable_molecules += int(mol_stable)
        except Exception as e:
            continue

    results = {
        'frac_atoms_stable': n_stable_atoms_total / n_atoms_total if n_atoms_total > 0 else 0,
        'frac_mols_stable_valence': n_stable_molecules / len(mol_list) if mol_list else 0
    }

    return results


# ============================================================
# REOS Structural Alerts (FlowMol 호환)
# ============================================================

# FlowMol reference 데이터 경로
REOS_TRAIN_DATA_PATH = Path('/home/csy/work1/3D/TheMol/data/train_reos_ring_counts.pkl')
REOS_TRAIN_DATA_URL = 'https://bits.csb.pitt.edu/files/FlowMol/data/train_reos_ring_counts.pkl'


def download_reos_train_data(download_path: Path):
    """FlowMol 서버에서 reference REOS 데이터 다운로드"""
    import subprocess as sp
    if not download_path.parent.exists():
        download_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading REOS reference data from {REOS_TRAIN_DATA_URL}...")
    wget_cmd = f"wget -O {download_path} {REOS_TRAIN_DATA_URL}"
    result = sp.run(wget_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error downloading REOS reference data: {result.stderr}")
    print(f"Downloaded to {download_path}")


def build_reos_df(flag_arr, flag_names):
    """FlowMol과 동일한 REOS DataFrame 생성"""
    flag_rates = flag_arr.sum(0) / flag_arr.shape[0]
    df_flags = pd.DataFrame({
        'flag_name': flag_names,
        'flag_count': flag_arr.sum(0),
        'flag_rate': flag_rates,
        'n_mols': flag_arr.shape[0],
    })
    return df_flags


def get_train_reos_df():
    """훈련 데이터의 REOS reference DataFrame 로드"""
    if not REOS_TRAIN_DATA_PATH.exists():
        print(f"Training REOS data not found at {REOS_TRAIN_DATA_PATH}")
        download_reos_train_data(REOS_TRAIN_DATA_PATH)
        if not REOS_TRAIN_DATA_PATH.exists():
            raise FileNotFoundError(f"Could not download REOS reference data")

    with open(REOS_TRAIN_DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    flag_arr = data['reos_flag_arr']
    flag_names = data['reos_flag_header']

    df_reos = build_reos_df(flag_arr, flag_names)
    return df_reos


def compute_cumulative_reos_deviation(df_reos, df_reos_train):
    """FlowMol과 동일한 reos_cum_dev 계산"""
    if df_reos is None:
        return {'reos_cum_dev': -1}

    # flag_name으로 정렬하여 동일한 순서로 비교
    df_reos_indexed = df_reos.set_index('flag_name')
    df_reos_train_indexed = df_reos_train.set_index('flag_name')

    # 공통 flag만 사용
    common_flags = df_reos_indexed.index.intersection(df_reos_train_indexed.index)

    if len(common_flags) == 0:
        return {'reos_cum_dev': -1}

    cum_deviation = np.abs(
        df_reos_indexed.loc[common_flags, 'flag_rate'] -
        df_reos_train_indexed.loc[common_flags, 'flag_rate']
    ).sum()

    return {'reos_cum_dev': float(cum_deviation)}


def compute_reos_metrics(mol_list):
    """REOS structural alerts 계산 (FlowMol 호환 + reos_cum_dev)

    reos_cum_dev: FlowMol reference 데이터와 비교하여 각 REOS flag의
    빈도 차이의 절대값 합계를 계산합니다. 값이 낮을수록 reference 분포에 가깝습니다.

    FlowMol과 동일한 REOS 규칙(Glaxo + Dundee)을 CSV에서 로드하여 사용합니다.
    """
    # REOS 규칙 로드 (FlowMol/useful_rdkit_utils와 동일)
    try:
        reos_df = load_reos_rules(active_rules=["Glaxo", "Dundee"])
    except Exception as e:
        print(f"Warning: Could not load REOS rules: {e}")
        return {
            'flag_rate': -1,
            'reos_flags_per_mol': -1,
            'has_flags_rate': -1,
            'reos_cum_dev': -1
        }

    # Reference 데이터 로드
    try:
        df_reos_train = get_train_reos_df()
        has_reference = True
    except Exception as e:
        print(f"Warning: Could not load REOS reference data: {e}")
        has_reference = False

    # sanitize된 분자 수집
    sanitized_mols = []
    for mol in mol_list:
        try:
            mol_copy = Chem.RWMol(mol)
            Chem.SanitizeMol(mol_copy)
            sanitized_mols.append(mol_copy)
        except:
            continue

    if len(sanitized_mols) == 0:
        return {
            'flag_rate': -1,
            'reos_flags_per_mol': -1,
            'has_flags_rate': -1,
            'reos_cum_dev': -1
        }

    # 분자별 flag array 생성 (FlowMol과 동일한 방식)
    flag_names = reos_df['flag_name'].tolist()
    flag_arr = np.zeros((len(sanitized_mols), len(flag_names)), dtype=bool)

    for mol_idx, mol in enumerate(sanitized_mols):
        for flag_idx, row in reos_df.iterrows():
            pat = row['pat']
            max_val = row['max']
            # FlowMol과 동일: matches > max_val 이면 flag
            if len(mol.GetSubstructMatches(pat)) > max_val:
                flag_arr[mol_idx, flag_idx] = True

    # 기본 메트릭 계산
    n_mols = len(sanitized_mols)
    total_flags = flag_arr.sum()
    n_mols_with_flags = (flag_arr.sum(axis=1) > 0).sum()

    results = {
        'flag_rate': float(total_flags / n_mols),
        'reos_flags_per_mol': float(total_flags / n_mols),
        'has_flags_rate': float(n_mols_with_flags / n_mols),
        'n_sanitized_for_reos': n_mols
    }

    # reos_cum_dev 계산 (reference와 비교)
    if has_reference:
        try:
            df_reos_model = build_reos_df(flag_arr, flag_names)
            cum_dev_metrics = compute_cumulative_reos_deviation(df_reos_model, df_reos_train)
            results.update(cum_dev_metrics)
        except Exception as e:
            print(f"Warning: Could not compute reos_cum_dev: {e}")
            results['reos_cum_dev'] = -1
    else:
        results['reos_cum_dev'] = -1

    return results


# ============================================================
# Ring System Analysis (FlowMol 호환 - ChEMBL 기반 OOD 계산)
# ============================================================

# ChEMBL ring system 데이터 경로 (pickle 형식 - pyarrow 불필요)
CHEMBL_RING_SYSTEMS_PATH = Path('/home/csy/work1/3D/TheMol/data/chembl_ring_systems.pkl')


class RingSystemFinder:
    """FlowMol/useful_rdkit_utils와 동일한 Ring System 추출 클래스"""

    def __init__(self):
        # Ring에 붙은 carbonyl 등을 보존하기 위한 패턴
        self.ring_db_pat = Chem.MolFromSmarts("[#6R,#18R]=[OR0,SR0,CR0,NR0]")
        self.ring_atom_pat = Chem.MolFromSmarts("[R]")

    def tag_bonds_to_preserve(self, mol):
        """Ring carbonyl 등의 bond를 보존 표시"""
        for bnd in mol.GetBonds():
            bnd.SetBoolProp("protected", False)
        for match in mol.GetSubstructMatches(self.ring_db_pat):
            bgn, end = match
            bnd = mol.GetBondBetweenAtoms(bgn, end)
            bnd.SetBoolProp("protected", True)

    @staticmethod
    def cleave_linker_bonds(mol):
        """Ring이 아닌 single bond를 끊어서 ring system 분리"""
        frag_bond_list = []
        for bnd in mol.GetBonds():
            if not bnd.IsInRing() and not bnd.GetBoolProp("protected") and bnd.GetBondType() == Chem.BondType.SINGLE:
                frag_bond_list.append(bnd.GetIdx())

        if len(frag_bond_list):
            frag_mol = Chem.FragmentOnBonds(mol, frag_bond_list)
            Chem.SanitizeMol(frag_mol)
            return frag_mol
        else:
            return mol

    def cleanup_fragments(self, mol):
        """Fragment에서 ring system만 추출"""
        frag_list = Chem.GetMolFrags(mol, asMols=True)
        ring_system_list = []
        for frag in frag_list:
            if frag.HasSubstructMatch(self.ring_atom_pat):
                for atm in frag.GetAtoms():
                    if atm.GetAtomicNum() == 0:
                        atm.SetAtomicNum(1)
                    atm.SetIsotope(0)
                frag = Chem.RemoveAllHs(frag)
                frag = self.fix_bond_stereo(frag)
                ring_system_list.append(frag)
        return ring_system_list

    @staticmethod
    def fix_bond_stereo(mol):
        """Double bond stereo 수정"""
        for bnd in mol.GetBonds():
            if bnd.GetBondType() == Chem.BondType.DOUBLE:
                begin_atm = bnd.GetBeginAtom()
                end_atm = bnd.GetEndAtom()
                if begin_atm.GetDegree() == 1 or end_atm.GetDegree() == 1:
                    bnd.SetStereo(Chem.BondStereo.STEREONONE)
        return mol

    def find_ring_systems(self, mol, as_mols=False):
        """분자에서 ring system 추출

        Args:
            mol: RDKit molecule
            as_mols: True면 molecule 객체 반환, False면 SMILES 반환
        """
        self.tag_bonds_to_preserve(mol)
        frag_mol = self.cleave_linker_bonds(mol)
        output_list = self.cleanup_fragments(frag_mol)
        if not as_mols:
            output_list = [Chem.MolToSmiles(x) for x in output_list]
        return output_list


class RingSystemLookup:
    """ChEMBL ring system 데이터베이스를 사용한 조회 클래스"""

    def __init__(self, ring_dict):
        self.ring_dict = ring_dict
        self.ring_system_finder = RingSystemFinder()

    @classmethod
    def default(cls):
        """기본 ChEMBL ring system 데이터 로드 (pickle 형식)"""
        if not CHEMBL_RING_SYSTEMS_PATH.exists():
            raise FileNotFoundError(f"ChEMBL ring systems data not found at {CHEMBL_RING_SYSTEMS_PATH}")

        with open(CHEMBL_RING_SYSTEMS_PATH, 'rb') as f:
            ring_dict = pickle.load(f)
        return cls(ring_dict)

    def process_mol(self, mol):
        """분자에서 ring system 추출 및 ChEMBL 빈도 조회

        Returns:
            list of (ring_smiles, chembl_count) tuples
        """
        output_ring_list = []
        if mol:
            ring_system_list = self.ring_system_finder.find_ring_systems(mol, as_mols=True)
            for ring in ring_system_list:
                smiles = Chem.MolToSmiles(ring)
                inchi = Chem.MolToInchiKey(ring)
                count = self.ring_dict.get(inchi, 0)
                output_ring_list.append((smiles, count))
        return output_ring_list


class RingSystemCounter:
    """FlowMol과 동일한 Ring System 카운터"""

    def __init__(self):
        self.ring_system_lookup = RingSystemLookup.default()

    def count_ring_systems(self, rdmols):
        """분자 리스트에서 ring system 카운트

        Returns:
            sample_counts: 샘플에서의 ring system 빈도
            chembl_counts: ChEMBL에서의 ring system 빈도
            n_mols: 분자 수
        """
        sample_counts = defaultdict(int)
        chembl_counts = {}
        n_mols = len(rdmols)

        for mol in rdmols:
            try:
                mol_ring_systems = self.ring_system_lookup.process_mol(mol)
                for ring_system_smi, chembl_count in mol_ring_systems:
                    sample_counts[ring_system_smi] += 1
                    chembl_counts[ring_system_smi] = chembl_count
            except:
                continue

        return sample_counts, chembl_counts, n_mols


def ring_counts_to_df(sample_counts, chembl_counts, n_mols):
    """Ring count를 DataFrame으로 변환"""
    ring_smi = list(sample_counts.keys())
    ring_counts = np.array([sample_counts[smi] for smi in ring_smi])
    chembl_counts_arr = np.array([chembl_counts[smi] for smi in ring_smi])
    sample_rate = ring_counts / n_mols if n_mols > 0 else ring_counts

    df_rings = pd.DataFrame({
        'ring_smi': ring_smi,
        'sample_rate': sample_rate,
        'sample_count': ring_counts,
        'n_mols': n_mols,
        'chembl_count': chembl_counts_arr
    })
    return df_rings


def compute_ring_metrics(mol_list):
    """Ring system OOD 분석 (FlowMol 호환)

    ChEMBL ring system 데이터베이스를 reference로 사용하여
    생성된 분자의 ring system이 ChEMBL에 존재하는지 확인합니다.
    ChEMBL에 없는 ring system은 OOD(Out-of-Distribution)로 간주됩니다.

    ood_rate = (ChEMBL에 없는 ring system 개수) / (전체 ring system 개수)
    """
    # sanitize된 분자 수집
    sanitized_mols = []
    for mol in mol_list:
        try:
            mol_copy = Chem.RWMol(mol)
            Chem.SanitizeMol(mol_copy)
            sanitized_mols.append(mol_copy)
        except:
            continue

    if len(sanitized_mols) == 0:
        return {
            'ood_rate': -1,
            'avg_rings_per_mol': -1,
            'n_unique_ring_systems': 0
        }

    # Ring system 분석
    try:
        ring_counter = RingSystemCounter()
        sample_counts, chembl_counts, n_mols = ring_counter.count_ring_systems(sanitized_mols)

        if len(sample_counts) == 0:
            return {
                'ood_rate': 0.0,
                'avg_rings_per_mol': 0.0,
                'n_unique_ring_systems': 0
            }

        # DataFrame으로 변환
        df_ring = ring_counts_to_df(sample_counts, chembl_counts, n_mols)

        # OOD rate 계산: ChEMBL에 없는 ring의 개수 / 전체 ring 개수
        ood_ring_count = df_ring[df_ring['chembl_count'] == 0]['sample_count'].sum()
        total_ring_count = df_ring['sample_count'].sum()
        ood_rate = ood_ring_count / n_mols if n_mols > 0 else 0

        # 평균 ring 수
        avg_rings_per_mol = total_ring_count / n_mols if n_mols > 0 else 0

        results = {
            'ood_rate': float(ood_rate),
            'avg_rings_per_mol': float(avg_rings_per_mol),
            'n_unique_ring_systems': len(sample_counts),
            'n_ood_ring_systems': int((df_ring['chembl_count'] == 0).sum()),
            'total_ring_count': int(total_ring_count)
        }

        return results

    except Exception as e:
        print(f"Warning: Could not compute ring metrics: {e}")
        return {
            'ood_rate': -1,
            'avg_rings_per_mol': -1,
            'n_unique_ring_systems': 0
        }


# ============================================================
# Energy Metrics (MMFF Force Field)
# ============================================================
def compute_mmff_energy(mol):
    """MMFF force field 에너지 계산"""
    try:
        ff = AllChem.MMFFGetMoleculeForceField(
            mol,
            AllChem.MMFFGetMoleculeProperties(mol),
            ignoreInterfragInteractions=False
        )
        if ff:
            return ff.CalcEnergy()
    except Exception:
        pass
    return None


def compute_energy_metrics(mol_list, reference_energies=None):
    """에너지 분포 메트릭 계산"""
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import gaussian_kde

    energies = []
    for mol in mol_list:
        try:
            mol_copy = Chem.RWMol(mol)
            Chem.SanitizeMol(mol_copy)
            energy = compute_mmff_energy(mol_copy)
            if energy is not None and not np.isnan(energy) and not np.isinf(energy):
                # 에너지를 원자 수로 정규화
                energy_per_atom = energy / mol_copy.GetNumAtoms()
                energies.append(energy_per_atom)
        except Exception:
            continue

    if len(energies) < 10:
        return {'energy_mean': -1, 'energy_std': -1, 'energy_js_div': -1}

    energies = np.array(energies)

    # 기본 통계
    results = {
        'energy_mean': float(np.mean(energies)),
        'energy_std': float(np.std(energies)),
        'energy_median': float(np.median(energies)),
        'n_valid_energies': len(energies)
    }

    # Reference 분포가 있으면 JS divergence 계산
    if reference_energies is not None and len(reference_energies) > 0:
        try:
            # 히스토그램 기반 JS divergence
            bins = np.linspace(
                min(energies.min(), np.min(reference_energies)),
                max(energies.max(), np.max(reference_energies)),
                50
            )

            hist_sample, _ = np.histogram(energies, bins=bins, density=True)
            hist_ref, _ = np.histogram(reference_energies, bins=bins, density=True)

            # 0 방지
            hist_sample = hist_sample + 1e-10
            hist_ref = hist_ref + 1e-10

            # 정규화
            hist_sample = hist_sample / hist_sample.sum()
            hist_ref = hist_ref / hist_ref.sum()

            js_div = jensenshannon(hist_sample, hist_ref)
            results['energy_js_div'] = float(js_div)
        except Exception as e:
            results['energy_js_div'] = -1
    else:
        results['energy_js_div'] = -1  # reference 없음

    return results


# ============================================================
# PoseBusters Metrics
# ============================================================
def compute_posebusters_metrics(mol_list):
    """PoseBusters 메트릭 계산"""
    try:
        import posebusters as pb

        print("\nRunning PoseBusters analysis...")
        buster = pb.PoseBusters(config='mol', max_workers=0)
        df_pb = buster.bust(mol_list, None, None)

        pb_results = df_pb.mean().to_dict()
        pb_results = {f'pb_{key}': pb_results[key] for key in pb_results}

        # Exclude 'non-aromatic_ring_non-flatness' from pb_valid calculation
        exclude_cols = ['non-aromatic_ring_non-flatness']
        check_cols = [col for col in df_pb.columns if col not in exclude_cols]

        n_pb_valid = df_pb[df_pb['sanitization'] == True][check_cols].values.astype(bool).all(axis=1).sum()
        pb_results['pb_valid'] = n_pb_valid / df_pb.shape[0] if df_pb.shape[0] > 0 else 0

        return pb_results

    except Exception as e:
        print(f"PoseBusters error: {e}")
        return {}


# ============================================================
# Geometry Evaluation (xTB optimization, RMSD, MMFF drop)
# ============================================================
def sdf_to_xyz(mol, filename):
    """RDKit mol을 XYZ 파일로 변환"""
    with open(filename, 'w') as f:
        f.write(f"{mol.GetNumAtoms()}\n\n")
        for atom in mol.GetAtoms():
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            f.write(f"{atom.GetSymbol()} {pos.x} {pos.y} {pos.z}\n")


def get_molecule_charge(mol):
    """분자의 총 formal charge 계산"""
    total_charge = 0
    for atom in mol.GetAtoms():
        total_charge += atom.GetFormalCharge()
    return total_charge


def run_xtb_optimization(xyz_filename, output_prefix, charge, work_dir):
    """xTB 최적화 실행 및 출력 캡처"""
    output_filename = os.path.join(work_dir, f"{output_prefix}_xtb_output.out")

    command = f"cd {work_dir} && xtb {os.path.basename(xyz_filename)} --opt --charge {charge} --namespace {output_prefix} > {os.path.basename(output_filename)} 2>&1"

    try:
        subprocess.run(command, shell=True, timeout=120)
        with open(output_filename, 'r') as f:
            xtb_output = f.read()
        return xtb_output
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        return None


def parse_xtb_output(xtb_output):
    """xTB 출력에서 energy gain과 RMSD 파싱"""
    if xtb_output is None:
        return None, None

    total_energy_gain = None
    total_rmsd = None

    lines = xtb_output.splitlines()
    for line in lines:
        if "total energy gain" in line:
            try:
                total_energy_gain = float(line.split()[6])  # kcal/mol
            except (IndexError, ValueError):
                pass
        elif "total RMSD" in line:
            try:
                total_rmsd = float(line.split()[5])  # Angstroms
            except (IndexError, ValueError):
                pass

    return total_energy_gain, total_rmsd


def parse_xtbtopo_mol(xtbtopo_filename):
    """xtbtopo.mol 파일을 RDKit molecule로 파싱"""
    if not os.path.exists(xtbtopo_filename):
        return None

    with open(xtbtopo_filename, 'r') as f:
        mol_block = f.read()
    mol = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=False)
    return mol


def process_molecule_xtb(mol, mol_idx, temp_dir):
    """단일 분자에 대해 xTB 최적화 수행"""
    if mol is None:
        return None, None, None

    xyz_filename = os.path.join(temp_dir, f"mol_{mol_idx}.xyz")
    output_prefix = f"mol_{mol_idx}"
    xtb_topo_filename = os.path.join(temp_dir, f"{output_prefix}.xtbtopo.mol")

    try:
        sdf_to_xyz(mol, xyz_filename)
        charge = get_molecule_charge(mol)

        xtb_output = run_xtb_optimization(xyz_filename, output_prefix, charge, temp_dir)
        total_energy_gain, total_rmsd = parse_xtb_output(xtb_output)

        optimized_mol = None
        if os.path.exists(xtb_topo_filename):
            optimized_mol = parse_xtbtopo_mol(xtb_topo_filename)

        return optimized_mol, total_energy_gain, total_rmsd

    except Exception as e:
        return None, None, None


def compute_rmsd_geometry(init_mol, opt_mol, hydrogens=True):
    """초기 분자와 최적화된 분자 간 RMSD 계산"""
    try:
        init_mol_copy = Chem.RWMol(init_mol)
        init_mol_copy.AddConformer(opt_mol.GetConformer(), assignId=True)

        if not hydrogens:
            init_mol_copy = Chem.RemoveAllHs(Chem.Mol(init_mol_copy))

        rmsd = AllChem.AlignMol(init_mol_copy, init_mol_copy, prbCid=0, refCid=1)
        return rmsd
    except Exception as e:
        return None


def compute_mmff_energy_drop(mol, max_iters=1000):
    """MMFF 최적화 전후 에너지 차이 계산"""
    try:
        mol_copy = Chem.Mol(mol)

        # 초기 MMFF 에너지
        props = AllChem.MMFFGetMoleculeProperties(mol_copy, mmffVariant='MMFF94')
        if props is None:
            return None
        ff = AllChem.MMFFGetMoleculeForceField(mol_copy, props)
        if ff is None:
            return None
        e_before = ff.CalcEnergy()

        # MMFF 최적화 실행
        success = AllChem.MMFFOptimizeMolecule(mol_copy, maxIters=max_iters)
        if success != 0 and success != 1:  # 0=converged, 1=not converged but ok
            return None

        # 최적화 후 에너지
        ff_opt = AllChem.MMFFGetMoleculeForceField(mol_copy, props)
        if ff_opt is None:
            return None
        e_after = ff_opt.CalcEnergy()

        return e_before - e_after

    except Exception as e:
        return None


def is_valid_for_geometry(mol):
    """Geometry 분석을 위한 분자 유효성 검사"""
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
        if len(Chem.GetMolFrags(mol)) > 1:
            return False
        return True
    except:
        return False


def compute_geometry_metrics(mol_list, n_subsets=1):
    """
    Geometry 메트릭 계산 (FlowMol fm3_evals/geometry 호환)

    Returns:
        - avg_energy_gain: 평균 GFN2-xTB 에너지 gain (kcal/mol)
        - med_energy_gain: 중앙값 GFN2-xTB 에너지 gain
        - avg_rmsd: 평균 RMSD (Å)
        - med_rmsd: 중앙값 RMSD
        - avg_mmff_drop: 평균 MMFF 에너지 drop (kcal/mol)
        - med_mmff_drop: 중앙값 MMFF 에너지 drop
    """
    print("\nRunning Geometry evaluation (xTB optimization)...")
    print("Note: This may take a while. Requires 'xtb' to be installed.")

    # xTB 설치 확인
    try:
        result = subprocess.run(['which', 'xtb'], capture_output=True, text=True)
        if result.returncode != 0:
            print("WARNING: 'xtb' not found in PATH. Skipping xTB-based metrics.")
            print("Install xTB: conda install -c conda-forge xtb")
            return compute_geometry_metrics_mmff_only(mol_list, n_subsets)
    except:
        print("WARNING: Could not check for 'xtb'. Attempting anyway...")

    # 유효한 분자만 필터링
    valid_mols = [(i, mol) for i, mol in enumerate(mol_list) if is_valid_for_geometry(mol)]

    if len(valid_mols) == 0:
        return {
            'avg_energy_gain': -1, 'med_energy_gain': -1,
            'avg_rmsd': -1, 'med_rmsd': -1,
            'avg_mmff_drop': -1, 'med_mmff_drop': -1,
            'n_geometry_valid': 0
        }

    energy_gains = []
    rmsds = []
    mmff_drops = []

    # 임시 디렉토리에서 작업
    with tempfile.TemporaryDirectory() as temp_dir:
        for idx, mol in tqdm(valid_mols, desc="xTB optimization"):
            # xTB 최적화
            opt_mol, energy_gain, xtb_rmsd = process_molecule_xtb(mol, idx, temp_dir)

            if energy_gain is not None:
                energy_gains.append(-energy_gain)  # 음수를 양수로 (에너지 감소량)

            # RMSD 계산 (xTB에서 제공하거나 직접 계산)
            if opt_mol is not None and xtb_rmsd is not None:
                rmsds.append(xtb_rmsd)
            elif opt_mol is not None:
                rmsd = compute_rmsd_geometry(mol, opt_mol, hydrogens=True)
                if rmsd is not None:
                    rmsds.append(rmsd)

            # MMFF energy drop
            mmff_drop = compute_mmff_energy_drop(mol)
            if mmff_drop is not None:
                mmff_drops.append(mmff_drop)

    # 결과 계산
    results = {
        'n_geometry_valid': len(valid_mols),
        'n_xtb_success': len(energy_gains),
    }

    if energy_gains:
        results['avg_energy_gain'] = float(np.mean(energy_gains))
        results['med_energy_gain'] = float(np.median(energy_gains))
    else:
        results['avg_energy_gain'] = -1
        results['med_energy_gain'] = -1

    if rmsds:
        results['avg_rmsd'] = float(np.mean(rmsds))
        results['med_rmsd'] = float(np.median(rmsds))
    else:
        results['avg_rmsd'] = -1
        results['med_rmsd'] = -1

    if mmff_drops:
        results['avg_mmff_drop'] = float(np.mean(mmff_drops))
        results['med_mmff_drop'] = float(np.median(mmff_drops))
    else:
        results['avg_mmff_drop'] = -1
        results['med_mmff_drop'] = -1

    # Subset 기반 CI 계산 (n_subsets > 1인 경우)
    if n_subsets > 1 and len(energy_gains) >= n_subsets:
        results.update(compute_geometry_ci(energy_gains, rmsds, mmff_drops, n_subsets))

    return results


def compute_geometry_metrics_mmff_only(mol_list, n_subsets=1):
    """xTB 없이 MMFF만 사용하는 geometry 메트릭"""
    print("Computing MMFF-only geometry metrics...")

    valid_mols = [mol for mol in mol_list if is_valid_for_geometry(mol)]

    if len(valid_mols) == 0:
        return {
            'avg_mmff_drop': -1, 'med_mmff_drop': -1,
            'n_geometry_valid': 0
        }

    mmff_drops = []
    for mol in tqdm(valid_mols, desc="MMFF optimization"):
        mmff_drop = compute_mmff_energy_drop(mol)
        if mmff_drop is not None:
            mmff_drops.append(mmff_drop)

    results = {
        'n_geometry_valid': len(valid_mols),
        'n_mmff_success': len(mmff_drops),
    }

    if mmff_drops:
        results['avg_mmff_drop'] = float(np.mean(mmff_drops))
        results['med_mmff_drop'] = float(np.median(mmff_drops))
    else:
        results['avg_mmff_drop'] = -1
        results['med_mmff_drop'] = -1

    return results


def compute_geometry_ci(energy_gains, rmsds, mmff_drops, n_subsets):
    """Geometry 메트릭의 95% CI 계산"""
    import math

    results = {}

    def compute_subset_ci(values, name, n_subsets):
        if len(values) < n_subsets:
            return {}
        subset_size = len(values) // n_subsets
        subset_means = []
        for i in range(n_subsets):
            subset = values[i * subset_size:(i + 1) * subset_size]
            if subset:
                subset_means.append(np.mean(subset))

        if len(subset_means) > 1:
            std = np.std(subset_means)
            ci95 = 1.96 * std / math.sqrt(len(subset_means))
            return {f'{name}_ci95': float(ci95)}
        return {}

    if energy_gains:
        results.update(compute_subset_ci(energy_gains, 'avg_energy_gain', n_subsets))
    if rmsds:
        results.update(compute_subset_ci(rmsds, 'avg_rmsd', n_subsets))
    if mmff_drops:
        results.update(compute_subset_ci(mmff_drops, 'avg_mmff_drop', n_subsets))

    return results


# ============================================================
# Main Evaluation Function
# ============================================================
def evaluate_molecules(mol_list, compute_energy=True, compute_geometry=False, n_subsets=1):
    """모든 평가 지표 계산"""
    print("\n" + "="*60)
    print("Molecule Evaluation (FlowMol Compatible)")
    print("="*60)

    n_steps = 8 if compute_geometry else 7
    metrics = {}

    # 1. Validity
    print(f"\n[1/{n_steps}] Computing validity metrics...")
    validity_metrics = compute_validity(mol_list)
    metrics.update(validity_metrics)

    # 2. Bond-based Stability
    print(f"\n[2/{n_steps}] Computing bond-based stability metrics...")
    bond_stability = compute_bond_stability(mol_list)
    metrics.update(bond_stability)

    # 3. Valence-based Stability
    print(f"\n[3/{n_steps}] Computing valence-based stability metrics...")
    valence_stability = compute_valence_stability(mol_list)
    metrics.update(valence_stability)

    # 4. REOS Structural Alerts
    print(f"\n[4/{n_steps}] Computing REOS structural alerts...")
    reos_metrics = compute_reos_metrics(mol_list)
    metrics.update(reos_metrics)

    # 5. Ring System Analysis
    print(f"\n[5/{n_steps}] Computing ring system metrics...")
    ring_metrics = compute_ring_metrics(mol_list)
    metrics.update(ring_metrics)

    # 6. Energy Metrics
    if compute_energy:
        print(f"\n[6/{n_steps}] Computing energy metrics...")
        energy_metrics = compute_energy_metrics(mol_list)
        metrics.update(energy_metrics)
    else:
        print(f"\n[6/{n_steps}] Skipping energy metrics...")

    # 7. PoseBusters
    print(f"\n[7/{n_steps}] Computing PoseBusters metrics...")
    pb_metrics = compute_posebusters_metrics(mol_list)
    metrics.update(pb_metrics)

    # 8. Geometry (optional)
    if compute_geometry:
        print(f"\n[8/{n_steps}] Computing Geometry metrics (xTB/MMFF)...")
        geometry_metrics = compute_geometry_metrics(mol_list, n_subsets=n_subsets)
        metrics.update(geometry_metrics)

    return metrics


# ============================================================
# Results Printing
# ============================================================
def print_metrics(metrics):
    """평가 지표 출력"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    # Validity
    validity_keys = ['frac_valid_mols', 'frac_connected', 'avg_frag_frac', 'avg_num_components']
    print("\n[Validity Metrics]")
    for key in validity_keys:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")

    # Bond-based Stability
    bond_stability_keys = ['mol_stable', 'atm_stable']
    print("\n[Bond-based Stability]")
    for key in bond_stability_keys:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")

    # Valence-based Stability
    valence_keys = ['frac_atoms_stable', 'frac_mols_stable_valence']
    print("\n[Valence-based Stability]")
    for key in valence_keys:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")

    # REOS
    reos_keys = ['flag_rate', 'has_flags_rate', 'reos_flags_per_mol', 'reos_cum_dev']
    print("\n[REOS Structural Alerts]")
    for key in reos_keys:
        if key in metrics and metrics[key] != -1:
            print(f"  {key}: {metrics[key]:.4f}")

    # Ring Systems
    ring_keys = ['ood_rate', 'avg_rings_per_mol', 'n_unique_ring_systems']
    print("\n[Ring System Analysis]")
    for key in ring_keys:
        if key in metrics and metrics[key] != -1:
            print(f"  {key}: {metrics[key]:.4f}" if isinstance(metrics[key], float) else f"  {key}: {metrics[key]}")

    # Energy
    energy_keys = ['energy_mean', 'energy_std', 'energy_median', 'energy_js_div']
    print("\n[Energy Metrics]")
    for key in energy_keys:
        if key in metrics and metrics[key] != -1:
            print(f"  {key}: {metrics[key]:.4f}")

    # PoseBusters
    pb_keys = [k for k in metrics.keys() if k.startswith('pb_')]
    if pb_keys:
        print("\n[PoseBusters Metrics]")
        for key in sorted(pb_keys):
            if isinstance(metrics[key], (int, float)):
                print(f"  {key}: {metrics[key]:.4f}")

    # Geometry
    geometry_keys = ['avg_energy_gain', 'med_energy_gain', 'avg_rmsd', 'med_rmsd',
                     'avg_mmff_drop', 'med_mmff_drop', 'n_geometry_valid', 'n_xtb_success',
                     'avg_energy_gain_ci95', 'avg_rmsd_ci95', 'avg_mmff_drop_ci95']
    geometry_present = [k for k in geometry_keys if k in metrics]
    if geometry_present:
        print("\n[Geometry Metrics (xTB/MMFF)]")
        for key in geometry_present:
            val = metrics[key]
            if val != -1:
                if isinstance(val, float):
                    print(f"  {key}: {val:.4f}")
                else:
                    print(f"  {key}: {val}")

    # Error Breakdown
    if 'error_counts' in metrics:
        print("\n[Error Breakdown]")
        for err_type, count in metrics['error_counts'].items():
            print(f"  {err_type}: {count}")

    print("="*60)


# ============================================================
# Main Function
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Generate molecules and evaluate with FlowMol-compatible metrics')
    # parser.add_argument('--checkpoint', type=str,
    #                     default='/home/csy/work1/3D/TheMol/saveGEOM/checkpoint_last.pt',
    #                     help='Path to model checkpoint')
    parser.add_argument('--checkpoint', type=str,
                        #default='/home/csy/work1/3D/TheMol/saveMIDI_reFLOW/checkpoint_last.pt',
                        default='/home/csy/work1/3D/TheMol/saveMIDI_Dual/checkpoint_last.pt',
                        #default='/home/csy/work1/3D/TheMol/saveOptimalDual/checkpoint_last.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of molecules to generate')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for generation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='/home/csy/work1/3D/TheMol/flowmol_eval',
                        help='Output directory for generated molecules')
    parser.add_argument('--skip_energy', action='store_true',
                        help='Skip energy metric calculation')
    parser.add_argument('--geometry', action='store_true',
                        help='Compute geometry metrics (xTB optimization, RMSD, MMFF drop)')
    parser.add_argument('--n_subsets', type=int, default=1,
                        help='Number of subsets for CI calculation (default: 1, no CI)')
    args = parser.parse_args()

    # GPU 설정
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # 1. 모델 로드
    print("\n[Step 1] Loading model...")
    model, model_args = load_model(args.checkpoint, device)

    # 2. 분자 생성
    print("\n[Step 2] Generating molecules...")
    mol_list = sample_molecules(
        model,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=device
    )

    if len(mol_list) == 0:
        print("No valid molecules generated. Exiting.")
        return

    # 3. 평가
    print("\n[Step 3] Evaluating molecules...")
    metrics = evaluate_molecules(
        mol_list,
        compute_energy=not args.skip_energy,
        compute_geometry=args.geometry,
        n_subsets=args.n_subsets
    )

    # 4. 결과 출력
    print_metrics(metrics)

    # 5. 결과 저장
    import json
    output_path = os.path.join(args.output_dir, 'evaluation_results.json')
    os.makedirs(args.output_dir, exist_ok=True)

    serializable_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            serializable_metrics[k] = float(v)
        elif isinstance(v, np.ndarray):
            serializable_metrics[k] = v.tolist()
        elif isinstance(v, dict):
            serializable_metrics[k] = v
        else:
            serializable_metrics[k] = v

    with open(output_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    print(f"\nResults saved to {output_path}")



if __name__ == '__main__':
    main()
