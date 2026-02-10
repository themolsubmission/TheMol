"""
Comprehensive Experiment for Molecule Generation Optimization

실험 목적:
1. No optimization (baseline) - Mean Docking, Mean SA
2. Docking only optimization - Mean Docking, Mean SA
3. SA only optimization - Mean Docking, Mean SA
4. Dual optimization (다양한 sa_weight 값) - Mean Docking, Mean SA

Usage:
    python run_comprehensive_experiment.py
"""

import sys
import os
import torch
import argparse
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from cmaes import CMA
import numpy as np
import random
import math
import gzip
import pickle
from collections import defaultdict
import warnings
import json
from datetime import datetime

# Suppress RDKit deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='rdkit')

# Add paths
sys.path.insert(0, '/home/csy/work1/3D/TheMol')
sys.path.insert(0, '/home/csy/anaconda3/envs/lf_cfm_cma/lib/python3.9/site-packages')

# Import unicore modules
from unicore import checkpoint_utils, options, tasks, utils
from unimol.tasks.molecule_sampling import MoleculeSampler
from unidock_zmq_client import UniDockClient
import glob

# ============================================================================
# SA Score 관련 함수들
# ============================================================================
_fscores = None

def _read_fragment_scores():
    """Fragment scores 로드 (fpscores.pkl.gz)"""
    global _fscores
    import os.path as op
    name = op.join('/home/csy/work1/previous/targetdiff_PINN/utils/evaluation', 'fpscores')
    with gzip.open('%s.pkl.gz' % name, 'rb') as f:
        data = pickle.load(f)
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict

def _num_bridgeheads_and_spiro(mol):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro

def _calculate_sa_score(m):
    """SA Score 계산 (1-10 범위, 낮을수록 합성 쉬움)"""
    global _fscores
    if _fscores is None:
        _read_fragment_scores()

    fp = rdMolDescriptors.GetMorganFingerprint(m, 2)
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = _num_bridgeheads_and_spiro(m)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms ** 1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    min_score = -4.0
    max_score = 2.5
    sascore = 11. - (sascore - min_score + 1) / (max_score - min_score) * 9.
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore

def compute_sa_score(rdmol):
    """정규화된 SA Score 계산 (0-1 범위, 높을수록 합성 쉬움)"""
    try:
        rdmol = Chem.MolFromSmiles(Chem.MolToSmiles(rdmol))
        if rdmol is None:
            return 0.5
        sa = _calculate_sa_score(rdmol)
        sa_norm = round((10 - sa) / 9, 2)
        return sa_norm
    except:
        return 0.5

# ============================================================================
# Utility 함수들
# ============================================================================
def set_random_seed(seed=42):
    """재현성을 위한 랜덤 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def filter_single_fragment_molecules(mol_list):
    """단일 fragment 분자만 필터링"""
    filtered_list = []
    for rd_mol, ligand_path in mol_list:
        if rd_mol is not None:
            frags = Chem.GetMolFrags(rd_mol)
            if len(frags) == 1:
                filtered_list.append((rd_mol, ligand_path))
    return filtered_list

def calculate_sa_scores(mol_list):
    """분자 리스트의 SA Score 계산"""
    sa_scores = {}
    for rd_mol, ligand_path in mol_list:
        ligand_name = os.path.basename(ligand_path)
        try:
            sa_norm = compute_sa_score(rd_mol)
            sa_scores[ligand_name] = sa_norm
        except:
            sa_scores[ligand_name] = 0.5
    return sa_scores

def dock_with_retry(ligand_paths, receptor, center_x, center_y, center_z,
                    size_x, size_y, size_z, output_dir, scoring="vina",
                    search_mode="fast", num_modes=1, port=5556, min_batch_size=1):
    """배치 도킹 실패 시 이진 분할하여 재시도"""
    if len(ligand_paths) == 0:
        return {}

    try:
        client = UniDockClient(port=port)
        result = client.dock(
            receptor=receptor, ligand=ligand_paths,
            center_x=center_x, center_y=center_y, center_z=center_z,
            size_x=size_x, size_y=size_y, size_z=size_z,
            output_dir=output_dir, scoring=scoring,
            search_mode=search_mode, num_modes=num_modes
        )
        client.close()

        if result.get('status') == 'success' and 'affinities' in result:
            return result['affinities']

        if len(ligand_paths) <= min_batch_size:
            return {}

        mid = len(ligand_paths) // 2
        left_result = dock_with_retry(
            ligand_paths[:mid], receptor, center_x, center_y, center_z,
            size_x, size_y, size_z, output_dir, scoring, search_mode,
            num_modes, port, min_batch_size
        )
        right_result = dock_with_retry(
            ligand_paths[mid:], receptor, center_x, center_y, center_z,
            size_x, size_y, size_z, output_dir, scoring, search_mode,
            num_modes, port, min_batch_size
        )
        merged = {}
        merged.update(left_result)
        merged.update(right_result)
        return merged

    except Exception as e:
        if len(ligand_paths) <= min_batch_size:
            return {}
        mid = len(ligand_paths) // 2
        left_result = dock_with_retry(
            ligand_paths[:mid], receptor, center_x, center_y, center_z,
            size_x, size_y, size_z, output_dir, scoring, search_mode,
            num_modes, port, min_batch_size
        )
        right_result = dock_with_retry(
            ligand_paths[mid:], receptor, center_x, center_y, center_z,
            size_x, size_y, size_z, output_dir, scoring, search_mode,
            num_modes, port, min_batch_size
        )
        merged = {}
        merged.update(left_result)
        merged.update(right_result)
        return merged

def get_ligand_info(sdf_path):
    """Reference ligand SDF 파일에서 분자의 중심 좌표와 원자 개수 계산"""
    try:
        supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
        if supplier and len(supplier) > 0 and supplier[0]:
            mol = supplier[0]
            conf = mol.GetConformer()
            num_atoms = mol.GetNumAtoms()
            positions = []
            for i in range(num_atoms):
                pos = conf.GetAtomPosition(i)
                positions.append([pos.x, pos.y, pos.z])
            positions = np.array(positions)
            center = positions.mean(axis=0)
            return float(center[0]), float(center[1]), float(center[2]), num_atoms
    except:
        return None

def find_reference_ligand(receptor_path):
    """Receptor path에서 reference ligand SDF 파일 찾기"""
    target_dir = os.path.dirname(receptor_path)
    ligand_files = glob.glob(os.path.join(target_dir, "*.sdf"))
    if not ligand_files:
        return None
    min_files = [f for f in ligand_files if '_min_' in f]
    return min_files[0] if min_files else ligand_files[0]


def select_random_targets(test_set_dir, num_targets=5, seed=42):
    """test_set 디렉토리에서 임의로 타겟 단백질을 선택

    Args:
        test_set_dir: test_set 디렉토리 경로
        num_targets: 선택할 타겟 개수
        seed: 랜덤 시드

    Returns:
        list of dict: [{'name': target_name, 'receptor': receptor_path, 'ref_ligand': ligand_path}, ...]
    """
    random.seed(seed)

    # 디렉토리 목록 가져오기 (파일 제외)
    all_items = os.listdir(test_set_dir)
    target_dirs = [d for d in all_items if os.path.isdir(os.path.join(test_set_dir, d))]

    # 유효한 타겟만 필터링 (receptor pdbqt 파일이 있는 것)
    valid_targets = []
    for target_name in target_dirs:
        target_path = os.path.join(test_set_dir, target_name)
        pdbqt_files = glob.glob(os.path.join(target_path, "*_rec.pdbqt"))
        if pdbqt_files:
            receptor_path = pdbqt_files[0]
            ref_ligand = find_reference_ligand(receptor_path)
            if ref_ligand:
                valid_targets.append({
                    'name': target_name,
                    'receptor': receptor_path,
                    'ref_ligand': ref_ligand
                })

    # 임의로 num_targets개 선택
    if len(valid_targets) < num_targets:
        print(f"Warning: Only {len(valid_targets)} valid targets found, using all of them")
        selected = valid_targets
    else:
        selected = random.sample(valid_targets, num_targets)

    return selected

# ============================================================================
# Model Loading
# ============================================================================
def load_model_and_task():
    """모델과 task 로드"""
    checkpoint_path = '/home/csy/work1/3D/TheMol/saveOptimal2_Flow/checkpoint_last.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    args = checkpoint['args']
    args.cpu = False
    args.distributed_world_size = 1
    args.distributed_rank = 0
    args.device_id = 0

    import unimol
    task = tasks.setup_task(args)
    model = task.build_model(args)

    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)

    model.eval()
    if not args.cpu and torch.cuda.is_available():
        model = model.cuda()
        if hasattr(model, 'interpolant') and hasattr(model.interpolant, 'device'):
            model.interpolant.device = torch.device('cuda:0')

    return model, task

# ============================================================================
# Experiment Functions
# ============================================================================
class ExperimentRunner:
    def __init__(self, model, task, receptor, center_x, center_y, center_z, num_atoms,
                 population_size=512, num_generations=5, num_final_samples=300,
                 sampling_std=0.1, seed=42, base_output_dir="./experiment_output", target_name="target"):
        self.model = model
        self.task = task
        self.receptor = receptor
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.num = num_atoms
        self.population_size = population_size
        self.num_generations = num_generations
        self.num_final_samples = num_final_samples
        self.sampling_std = sampling_std
        self.seed = seed
        self.base_output_dir = base_output_dir
        self.target_name = target_name

        # Fixed parameters
        self.emb_dim = 8
        self.min_atoms = 10
        self.max_atoms = max(60, num_atoms + 10)
        self.num_tokens = max(64, num_atoms + 10)

        # Create output directory for this target
        os.makedirs(self.base_output_dir, exist_ok=True)

        # Create sampler
        vocab_dict = {}
        if hasattr(task, 'dictionary') and hasattr(task.dictionary, 'symbols'):
            for idx, symbol in enumerate(task.dictionary.symbols):
                vocab_dict[idx] = symbol

        self.sampler = MoleculeSampler(
            vocab_dict=vocab_dict,
            output_dir=self.base_output_dir,
            device='cuda'
        )

    def generate_molecules(self, latent_input, batch_size):
        """Generate molecules from latent input"""
        num_atom = torch.full((batch_size,), self.num, device=self.sampler.device)

        mol_list, stats = self.sampler.sample_molecules(
            model=self.model,
            latent=latent_input,
            num_atom=num_atom,
            batch_size=batch_size,
            num_tokens=self.num_tokens,
            emb_dim=self.emb_dim,
            min_atoms=self.min_atoms,
            max_atoms=self.max_atoms,
            use_dopri5=False,
            temperature=1.0,
            optimize=False
        )
        return mol_list

    def evaluate_molecules(self, mol_list, output_dir, search_mode="balance"):
        """Evaluate molecules: docking + SA scores"""
        if len(mol_list) == 0:
            return [], 0.0, 0.0, 0

        mol_list = filter_single_fragment_molecules(mol_list)
        if len(mol_list) == 0:
            return [], 0.0, 0.0, 0

        ligand_paths = [p for _, p in mol_list]

        # Docking
        affinities = dock_with_retry(
            ligand_paths=ligand_paths,
            receptor=self.receptor,
            center_x=self.center_x, center_y=self.center_y, center_z=self.center_z,
            size_x=25.0, size_y=25.0, size_z=25.0,
            output_dir=output_dir,
            scoring="vina", search_mode=search_mode, num_modes=1
        )

        # SA scores
        sa_scores = calculate_sa_scores(mol_list)

        # Collect results
        results = []
        for rd_mol, ligand_path in mol_list:
            ligand_name = os.path.basename(ligand_path)
            docking = affinities.get(ligand_name, 0.0)
            sa = sa_scores.get(ligand_name, 0.5)
            if docking != 0.0:  # Valid docking
                results.append({'docking': docking, 'sa': sa, 'name': ligand_name})

        if len(results) == 0:
            return [], 0.0, 0.0, 0

        mean_docking = sum(r['docking'] for r in results) / len(results)
        mean_sa = sum(r['sa'] for r in results) / len(results)

        return results, mean_docking, mean_sa, len(results)

    def run_no_optimization(self):
        """Experiment 1: No optimization (baseline)"""
        print("\n" + "="*70)
        print("Experiment 1: No Optimization (Baseline)")
        print("="*70)

        set_random_seed(self.seed)

        # Generate from random latent
        batch_size = self.num_final_samples
        random_latent = torch.randn(batch_size, self.num_tokens, self.emb_dim).to(self.sampler.device)

        print(f"Generating {batch_size} molecules from random latent...")
        mol_list = self.generate_molecules(random_latent, batch_size)

        print(f"Generated {len(mol_list)} molecules, evaluating...")
        results, mean_docking, mean_sa, num_valid = self.evaluate_molecules(
            mol_list, os.path.join(self.base_output_dir, "baseline")
        )

        print(f"Results: Mean Docking={mean_docking:.2f}, Mean SA={mean_sa:.2f}, Valid={num_valid}")
        return {'mean_docking': mean_docking, 'mean_sa': mean_sa, 'num_valid': num_valid}

    def run_docking_only_optimization(self):
        """Experiment 2: Docking only optimization"""
        print("\n" + "="*70)
        print("Experiment 2: Docking Only Optimization")
        print("="*70)

        set_random_seed(self.seed)

        optimizer = CMA(mean=np.zeros(self.num * self.emb_dim), sigma=1.0,
                       population_size=self.population_size, lr_adapt=True, seed=self.seed)

        for gen in range(self.num_generations):
            z_batch_list = []
            z_origin_list = []

            for _ in range(optimizer.population_size):
                z_flat = optimizer.ask()
                z_reshaped = z_flat.reshape(self.num, self.emb_dim)
                z_origin_list.append(z_flat)
                z_batch_list.append(z_reshaped)

            z_tensor = torch.tensor(np.stack(z_batch_list), dtype=torch.float32)
            input_tensor = torch.cat([z_tensor, torch.randn(self.population_size, self.num_tokens - self.num, self.emb_dim)], dim=1).to(self.sampler.device)

            mol_list = self.generate_molecules(input_tensor, self.population_size)
            mol_list = filter_single_fragment_molecules(mol_list)

            if len(mol_list) > 0:
                ligand_paths = [p for _, p in mol_list]
                affinities = dock_with_retry(
                    ligand_paths=ligand_paths,
                    receptor=self.receptor,
                    center_x=self.center_x, center_y=self.center_y, center_z=self.center_z,
                    size_x=25.0, size_y=25.0, size_z=25.0,
                    output_dir=os.path.join(self.base_output_dir, f"docking_only_gen{gen}"),
                    scoring="vina", search_mode="fast", num_modes=1
                )

                mol_index_to_score = {}
                for rd_mol, ligand_path in mol_list:
                    ligand_name = os.path.basename(ligand_path)
                    docking = affinities.get(ligand_name, 0.0)
                    try:
                        mol_idx = int(ligand_name.split('_')[1].split('.')[0])
                        mol_index_to_score[mol_idx] = docking if docking != 0.0 else 10.0
                    except:
                        pass

                solutions = []
                for idx in range(optimizer.population_size):
                    z_flat = z_origin_list[idx]
                    score = mol_index_to_score.get(idx + 1, 10.0)
                    solutions.append((z_flat, score))

                optimizer.tell(solutions)

                valid_scores = [s for s in mol_index_to_score.values() if s < 10.0]
                if valid_scores:
                    print(f"  Gen {gen}: Best={min(valid_scores):.2f}, Mean={sum(valid_scores)/len(valid_scores):.2f}")

        # Final sampling
        print("Final sampling from optimized latent...")
        mean_tensor = torch.tensor(optimizer._mean, dtype=torch.float32)
        sampled_latent = torch.stack([torch.normal(mean=mean_tensor, std=self.sampling_std) for _ in range(self.num_final_samples)])
        input_raw = sampled_latent.view(self.num_final_samples, self.num, self.emb_dim)
        input_tensor = torch.cat([input_raw, torch.randn(self.num_final_samples, self.num_tokens - self.num, self.emb_dim)], dim=1).to(self.sampler.device)

        mol_list = self.generate_molecules(input_tensor, self.num_final_samples)
        results, mean_docking, mean_sa, num_valid = self.evaluate_molecules(
            mol_list, os.path.join(self.base_output_dir, "docking_only_final")
        )

        print(f"Results: Mean Docking={mean_docking:.2f}, Mean SA={mean_sa:.2f}, Valid={num_valid}")
        return {'mean_docking': mean_docking, 'mean_sa': mean_sa, 'num_valid': num_valid}

    def run_sa_only_optimization(self):
        """Experiment 3: SA only optimization"""
        print("\n" + "="*70)
        print("Experiment 3: SA Only Optimization")
        print("="*70)

        set_random_seed(self.seed)

        optimizer = CMA(mean=np.zeros(self.num * self.emb_dim), sigma=1.0,
                       population_size=self.population_size, lr_adapt=True, seed=self.seed)

        for gen in range(self.num_generations):
            z_batch_list = []
            z_origin_list = []

            for _ in range(optimizer.population_size):
                z_flat = optimizer.ask()
                z_reshaped = z_flat.reshape(self.num, self.emb_dim)
                z_origin_list.append(z_flat)
                z_batch_list.append(z_reshaped)

            z_tensor = torch.tensor(np.stack(z_batch_list), dtype=torch.float32)
            input_tensor = torch.cat([z_tensor, torch.randn(self.population_size, self.num_tokens - self.num, self.emb_dim)], dim=1).to(self.sampler.device)

            mol_list = self.generate_molecules(input_tensor, self.population_size)
            mol_list = filter_single_fragment_molecules(mol_list)

            if len(mol_list) > 0:
                sa_scores = calculate_sa_scores(mol_list)

                mol_index_to_score = {}
                for rd_mol, ligand_path in mol_list:
                    ligand_name = os.path.basename(ligand_path)
                    sa_norm = sa_scores.get(ligand_name, 0.5)
                    try:
                        mol_idx = int(ligand_name.split('_')[1].split('.')[0])
                        # Lower is better for CMA-ES, so use negative SA
                        mol_index_to_score[mol_idx] = -sa_norm * 10
                    except:
                        pass

                solutions = []
                for idx in range(optimizer.population_size):
                    z_flat = z_origin_list[idx]
                    # 실패 패널티를 1.0으로 설정 (SA=0보다 약간 나쁨)
                    # 성공: -10 ~ 0, 실패: +1
                    score = mol_index_to_score.get(idx + 1, 10.0)
                    solutions.append((z_flat, score))

                optimizer.tell(solutions)

                valid_scores = [s for s in mol_index_to_score.values()]
                if valid_scores:
                    best_sa = -min(valid_scores) / 10
                    mean_sa = -sum(valid_scores) / len(valid_scores) / 10
                    print(f"  Gen {gen}: Best SA={best_sa:.2f}, Mean SA={mean_sa:.2f}")

        # Final sampling
        print("Final sampling from optimized latent...")
        mean_tensor = torch.tensor(optimizer._mean, dtype=torch.float32)
        sampled_latent = torch.stack([torch.normal(mean=mean_tensor, std=self.sampling_std) for _ in range(self.num_final_samples)])
        input_raw = sampled_latent.view(self.num_final_samples, self.num, self.emb_dim)
        input_tensor = torch.cat([input_raw, torch.randn(self.num_final_samples, self.num_tokens - self.num, self.emb_dim)], dim=1).to(self.sampler.device)

        mol_list = self.generate_molecules(input_tensor, self.num_final_samples)
        results, mean_docking, mean_sa, num_valid = self.evaluate_molecules(
            mol_list, os.path.join(self.base_output_dir, "sa_only_final")
        )

        print(f"Results: Mean Docking={mean_docking:.2f}, Mean SA={mean_sa:.2f}, Valid={num_valid}")
        return {'mean_docking': mean_docking, 'mean_sa': mean_sa, 'num_valid': num_valid}

    def run_dual_optimization(self, sa_weight=1.0):
        """Experiment 4: Dual optimization (Docking + SA)"""
        print("\n" + "="*70)
        print(f"Experiment 4: Dual Optimization (sa_weight={sa_weight})")
        print("="*70)

        set_random_seed(self.seed)

        optimizer = CMA(mean=np.zeros(self.num * self.emb_dim), sigma=1.0,
                       population_size=self.population_size, lr_adapt=True, seed=self.seed)

        PENALTY_SCORE = 10.0

        for gen in range(self.num_generations):
            z_batch_list = []
            z_origin_list = []

            for _ in range(optimizer.population_size):
                z_flat = optimizer.ask()
                z_reshaped = z_flat.reshape(self.num, self.emb_dim)
                z_origin_list.append(z_flat)
                z_batch_list.append(z_reshaped)

            z_tensor = torch.tensor(np.stack(z_batch_list), dtype=torch.float32)
            input_tensor = torch.cat([z_tensor, torch.randn(self.population_size, self.num_tokens - self.num, self.emb_dim)], dim=1).to(self.sampler.device)

            mol_list = self.generate_molecules(input_tensor, self.population_size)
            mol_list = filter_single_fragment_molecules(mol_list)

            if len(mol_list) > 0:
                ligand_paths = [p for _, p in mol_list]
                affinities = dock_with_retry(
                    ligand_paths=ligand_paths,
                    receptor=self.receptor,
                    center_x=self.center_x, center_y=self.center_y, center_z=self.center_z,
                    size_x=25.0, size_y=25.0, size_z=25.0,
                    output_dir=os.path.join(self.base_output_dir, f"dual_w{sa_weight}_gen{gen}"),
                    scoring="vina", search_mode="fast", num_modes=1
                )

                sa_scores = calculate_sa_scores(mol_list)

                mol_index_to_score = {}
                mol_index_to_details = {}
                for rd_mol, ligand_path in mol_list:
                    ligand_name = os.path.basename(ligand_path)
                    docking = affinities.get(ligand_name, 5.0)
                    sa_norm = sa_scores.get(ligand_name, 0.5)

                    if docking == 0.0:
                        combined = PENALTY_SCORE
                    else:
                        sa_term = -sa_norm * 10
                        combined = docking + sa_weight * sa_term

                    try:
                        mol_idx = int(ligand_name.split('_')[1].split('.')[0])
                        mol_index_to_score[mol_idx] = combined
                        mol_index_to_details[mol_idx] = {'docking': docking, 'sa': sa_norm, 'combined': combined}
                    except:
                        pass

                solutions = []
                for idx in range(optimizer.population_size):
                    z_flat = z_origin_list[idx]
                    score = mol_index_to_score.get(idx + 1, PENALTY_SCORE)
                    solutions.append((z_flat, score))

                optimizer.tell(solutions)

                if mol_index_to_details:
                    best_idx = min(mol_index_to_score, key=mol_index_to_score.get)
                    best = mol_index_to_details[best_idx]
                    print(f"  Gen {gen}: Best Combined={best['combined']:.2f} (Dock={best['docking']:.2f}, SA={best['sa']:.2f})")

        # Final sampling
        print("Final sampling from optimized latent...")
        mean_tensor = torch.tensor(optimizer._mean, dtype=torch.float32)
        sampled_latent = torch.stack([torch.normal(mean=mean_tensor, std=self.sampling_std) for _ in range(self.num_final_samples)])
        input_raw = sampled_latent.view(self.num_final_samples, self.num, self.emb_dim)
        input_tensor = torch.cat([input_raw, torch.randn(self.num_final_samples, self.num_tokens - self.num, self.emb_dim)], dim=1).to(self.sampler.device)

        mol_list = self.generate_molecules(input_tensor, self.num_final_samples)
        results, mean_docking, mean_sa, num_valid = self.evaluate_molecules(
            mol_list, os.path.join(self.base_output_dir, f"dual_w{sa_weight}_final")
        )

        print(f"Results: Mean Docking={mean_docking:.2f}, Mean SA={mean_sa:.2f}, Valid={num_valid}")
        return {'mean_docking': mean_docking, 'mean_sa': mean_sa, 'num_valid': num_valid, 'sa_weight': sa_weight}


def run_experiments_for_target(model, task, target_info, args, sa_weights, experiments):
    """단일 타겟에 대해 선택된 실험 실행

    Args:
        model: 로드된 모델
        task: 로드된 task
        target_info: {'name': str, 'receptor': str, 'ref_ligand': str}
        args: 명령줄 인자
        sa_weights: SA weight 리스트
        experiments: 실행할 실험 set {'docking', 'sa', 'dual'}

    Returns:
        dict: 실험 결과
    """
    target_name = target_info['name']
    receptor = target_info['receptor']
    ref_ligand_path = target_info['ref_ligand']

    print("\n" + "#"*70)
    print(f"  TARGET: {target_name}")
    print("#"*70)

    # Get ligand info for docking center
    ligand_info = get_ligand_info(ref_ligand_path)
    if ligand_info:
        center_x, center_y, center_z, ref_num_atoms = ligand_info
        if ref_num_atoms <= 20:
            num_atoms = random.randint(30, 40)
        else:
            num_atoms = ref_num_atoms
        print(f"Reference ligand: {os.path.basename(ref_ligand_path)} ({ref_num_atoms} atoms)")
        print(f"Docking center: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})")
        print(f"Using {num_atoms} atoms for generation")
    else:
        num_atoms, center_x, center_y, center_z = 34, 0.0, 0.0, 0.0
        print("Warning: Could not get ligand info, using defaults")

    # Create target-specific output directory
    base_output_dir = f"./experiment_output/{target_name}"
    os.makedirs(base_output_dir, exist_ok=True)

    # Create experiment runner for this target
    runner = ExperimentRunner(
        model=model, task=task,
        receptor=receptor,
        center_x=center_x, center_y=center_y, center_z=center_z,
        num_atoms=num_atoms,
        population_size=args.population_size,
        num_generations=args.num_generations,
        num_final_samples=args.num_final_samples,
        sampling_std=args.sampling_std,
        seed=args.seed,
        base_output_dir=base_output_dir,
        target_name=target_name
    )

    # Run experiments
    results = {}

    # 1. No optimization (baseline) - 항상 실행
    results['no_optimization'] = runner.run_no_optimization()

    # 2. Docking only optimization
    if 'docking' in experiments:
        results['docking_only'] = runner.run_docking_only_optimization()

    # 3. SA only optimization
    if 'sa' in experiments:
        results['sa_only'] = runner.run_sa_only_optimization()

    # 4. Dual optimization with various sa_weights
    if 'dual' in experiments:
        results['dual'] = {}
        for sa_weight in sa_weights:
            results['dual'][f'sa_weight_{sa_weight}'] = runner.run_dual_optimization(sa_weight=sa_weight)

    return results


def print_final_summary(all_results, sa_weights, experiments):
    """모든 타겟에 대한 최종 결과 요약 출력

    Args:
        all_results: {target_name: results_dict, ...}
        sa_weights: SA weight 리스트
        experiments: 실행된 실험 set {'docking', 'sa', 'dual'}
    """
    print("\n" + "="*100)
    print("  FINAL EXPERIMENT SUMMARY - ALL TARGETS")
    print("="*100)

    # 각 타겟별 결과 출력
    for target_name, results in all_results.items():
        print(f"\n{'='*100}")
        print(f"  Target: {target_name}")
        print(f"{'='*100}")
        print(f"{'Experiment':<30} {'Mean Docking':<15} {'Mean SA':<15} {'Valid':<10}")
        print("-"*70)

        # No optimization (baseline) - 항상 출력
        exp = results['no_optimization']
        print(f"{'No Optimization (Baseline)':<30} {exp['mean_docking']:<15.2f} {exp['mean_sa']:<15.2f} {exp['num_valid']:<10}")

        # Docking only
        if 'docking_only' in results:
            exp = results['docking_only']
            print(f"{'Docking Only':<30} {exp['mean_docking']:<15.2f} {exp['mean_sa']:<15.2f} {exp['num_valid']:<10}")

        # SA only
        if 'sa_only' in results:
            exp = results['sa_only']
            print(f"{'SA Only':<30} {exp['mean_docking']:<15.2f} {exp['mean_sa']:<15.2f} {exp['num_valid']:<10}")

        # Dual
        if 'dual' in results:
            for key, exp in results['dual'].items():
                print(f"{'Dual (' + key + ')':<30} {exp['mean_docking']:<15.2f} {exp['mean_sa']:<15.2f} {exp['num_valid']:<10}")

    # 평균 결과 계산 및 출력
    print("\n" + "="*100)
    print("  AVERAGE ACROSS ALL TARGETS")
    print("="*100)
    print(f"{'Experiment':<30} {'Mean Docking':<15} {'Mean SA':<15} {'Avg Valid':<10}")
    print("-"*70)

    target_names = list(all_results.keys())
    num_targets = len(target_names)

    # No optimization average - 항상 출력
    avg_docking = sum(all_results[t]['no_optimization']['mean_docking'] for t in target_names) / num_targets
    avg_sa = sum(all_results[t]['no_optimization']['mean_sa'] for t in target_names) / num_targets
    avg_valid = sum(all_results[t]['no_optimization']['num_valid'] for t in target_names) / num_targets
    print(f"{'No Optimization (Baseline)':<30} {avg_docking:<15.2f} {avg_sa:<15.2f} {avg_valid:<10.1f}")

    # Docking only average
    if 'docking' in experiments:
        avg_docking = sum(all_results[t]['docking_only']['mean_docking'] for t in target_names) / num_targets
        avg_sa = sum(all_results[t]['docking_only']['mean_sa'] for t in target_names) / num_targets
        avg_valid = sum(all_results[t]['docking_only']['num_valid'] for t in target_names) / num_targets
        print(f"{'Docking Only':<30} {avg_docking:<15.2f} {avg_sa:<15.2f} {avg_valid:<10.1f}")

    # SA only average
    if 'sa' in experiments:
        avg_docking = sum(all_results[t]['sa_only']['mean_docking'] for t in target_names) / num_targets
        avg_sa = sum(all_results[t]['sa_only']['mean_sa'] for t in target_names) / num_targets
        avg_valid = sum(all_results[t]['sa_only']['num_valid'] for t in target_names) / num_targets
        print(f"{'SA Only':<30} {avg_docking:<15.2f} {avg_sa:<15.2f} {avg_valid:<10.1f}")

    # Dual averages
    if 'dual' in experiments:
        for w in sa_weights:
            key = f'sa_weight_{w}'
            avg_docking = sum(all_results[t]['dual'][key]['mean_docking'] for t in target_names) / num_targets
            avg_sa = sum(all_results[t]['dual'][key]['mean_sa'] for t in target_names) / num_targets
            avg_valid = sum(all_results[t]['dual'][key]['num_valid'] for t in target_names) / num_targets
            print(f"{'Dual (' + key + ')':<30} {avg_docking:<15.2f} {avg_sa:<15.2f} {avg_valid:<10.1f}")

    print("="*100)


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Molecule Generation Experiment')
    parser.add_argument('--population-size', type=int, default=512)
    parser.add_argument('--num-generations', type=int, default=10)
    parser.add_argument('--num-final-samples', type=int, default=300)
    parser.add_argument('--sampling-std', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sa-weights', type=str, default='0.5,1.0,2.0',
                       help='Comma-separated list of sa_weight values for dual optimization')
    parser.add_argument('--num-targets', type=int, default=5,
                       help='Number of target proteins to test')
    parser.add_argument('--test-set-dir', type=str,
                       default='/home/csy/work1/previous/targetdiff_PINN/data/test_set',
                       help='Directory containing target protein folders')
    parser.add_argument('--experiments', type=str, default='all',
                       help='Experiments to run: all, none, docking, sa, dual (comma-separated for multiple, e.g., "docking,sa")')
    parser.add_argument('--target', type=str, default=None,
                       help='Specific target name to test (e.g., BTRN_BACCI_2_250_0). If not specified, random targets will be selected.')
    args = parser.parse_args()

    # Parse experiments to run
    exp_input = args.experiments.lower().strip()
    if exp_input == 'all':
        experiments = {'docking', 'sa', 'dual'}
    elif exp_input == 'none':
        experiments = set()
    else:
        experiments = set(e.strip() for e in exp_input.split(','))
        # Validate experiment names
        valid_experiments = {'docking', 'sa', 'dual'}
        invalid = experiments - valid_experiments
        if invalid:
            print(f"Warning: Invalid experiment names ignored: {invalid}")
            experiments = experiments & valid_experiments

    print("="*70)
    print("  Comprehensive Molecule Generation Experiment")
    print("  Multi-Target Version")
    print("="*70)
    print(f"  population_size={args.population_size}")
    print(f"  num_generations={args.num_generations}")
    print(f"  num_final_samples={args.num_final_samples}")
    print(f"  sampling_std={args.sampling_std}")
    print(f"  seed={args.seed}")
    print(f"  sa_weights={args.sa_weights}")
    print(f"  num_targets={args.num_targets}")
    print(f"  test_set_dir={args.test_set_dir}")
    print(f"  experiments={experiments if experiments else 'none (baseline only)'}")
    print("="*70)

    # Parse SA weights
    sa_weights = [float(w.strip()) for w in args.sa_weights.split(',')]

    # Select targets
    if args.target:
        # 특정 타겟 지정
        target_path = os.path.join(args.test_set_dir, args.target)
        pdbqt_files = glob.glob(os.path.join(target_path, "*_rec.pdbqt"))
        if pdbqt_files:
            receptor_path = pdbqt_files[0]
            ref_ligand = find_reference_ligand(receptor_path)
            targets = [{'name': args.target, 'receptor': receptor_path, 'ref_ligand': ref_ligand}]
            print(f"\nUsing specified target: {args.target}")
        else:
            print(f"Error: Target {args.target} not found or no receptor file")
            sys.exit(1)
    else:
        # 기존 랜덤 선택
        print(f"\nSelecting {args.num_targets} random targets...")
        targets = select_random_targets(args.test_set_dir, args.num_targets, args.seed)
    print(f"Selected targets:")
    for i, t in enumerate(targets, 1):
        print(f"  {i}. {t['name']}")

    # Load model (once, reuse for all targets)
    print("\nLoading model...")
    model, task = load_model_and_task()
    print("Model loaded successfully!")

    # Create main output directory
    os.makedirs("./experiment_output", exist_ok=True)

    # Run experiments for each target
    all_results = {}
    for i, target_info in enumerate(targets, 1):
        print(f"\n{'*'*70}")
        print(f"  Processing target {i}/{len(targets)}: {target_info['name']}")
        print(f"{'*'*70}")

        results = run_experiments_for_target(model, task, target_info, args, sa_weights, experiments)
        all_results[target_info['name']] = results

    # Print final summary
    print_final_summary(all_results, sa_weights, experiments)

    # Save all results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"./experiment_output/all_results_{timestamp}.json"
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to: {result_file}")


if __name__ == "__main__":
    main()
