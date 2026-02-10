"""
Multi-Target Ligand Generation with Dual Optimization (Docking + SA)
- Baseline (no optimization) vs Optimized comparison
- Saves molecules as .pt (RDKit mol objects) and pdbqt files
- Supports configurable optimization parameters
"""

import sys
import os
import torch
import glob
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from cmaes import CMA
import argparse
import subprocess
import time
import atexit
import pickle
import math
import gzip
import random
import json
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Loading modules...")

# Import unicore modules
from unicore import tasks

# Import our module
from unimol.tasks.molecule_sampling import MoleculeSampler

# Import Uni-Dock ZMQ client
from unidock_zmq_client import UniDockClient


# ============================================================================
# SA Score Functions
# ============================================================================
_fscores = None

def _read_fragment_scores():
    """Load fragment scores (fpscores.pkl.gz)"""
    global _fscores
    import os.path as op
    # Use data directory relative to this script
    name = op.join(op.dirname(op.dirname(op.abspath(__file__))), 'data', 'fpscores')
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
    """SA Score calculation (1-10 range, lower = easier to synthesize)"""
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
    """Normalized SA Score (0-1 range, higher = easier to synthesize)"""
    try:
        rdmol = Chem.MolFromSmiles(Chem.MolToSmiles(rdmol))
        if rdmol is None:
            return 0.5
        sa = _calculate_sa_score(rdmol)
        sa_norm = round((10 - sa) / 9, 2)
        return sa_norm
    except:
        return 0.5

def calculate_sa_scores(mol_list):
    """Calculate SA scores for molecule list"""
    sa_scores = {}
    for rd_mol, ligand_path in mol_list:
        ligand_name = os.path.basename(ligand_path)
        try:
            sa_norm = compute_sa_score(rd_mol)
            sa_scores[ligand_name] = sa_norm
        except:
            sa_scores[ligand_name] = 0.5
    return sa_scores


# ============================================================================
# Utility Functions
# ============================================================================
def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def filter_single_fragment_molecules(mol_list):
    """Filter to keep only single fragment molecules"""
    filtered_list = []
    excluded_count = 0

    for rd_mol, ligand_path in mol_list:
        if rd_mol is not None:
            frags = Chem.GetMolFrags(rd_mol)
            if len(frags) == 1:
                filtered_list.append((rd_mol, ligand_path))
            else:
                excluded_count += 1

    if excluded_count > 0:
        print(f"    Filtered out {excluded_count} molecules with multiple fragments")

    return filtered_list


def dock_with_retry(ligand_paths, receptor, center_x, center_y, center_z,
                    size_x, size_y, size_z, output_dir, scoring="vina",
                    search_mode="fast", num_modes=1, port=5556, min_batch_size=1):
    """Batch docking with binary split retry on failure"""
    if len(ligand_paths) == 0:
        return {}

    try:
        client = UniDockClient(port=port)
        result = client.dock(
            receptor=receptor,
            ligand=ligand_paths,
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


# ============================================================================
# Server Management
# ============================================================================
UNIDOCK_SERVER_PROCESS = None

def cleanup_unidock_server():
    """Clean up Uni-Dock server on exit"""
    global UNIDOCK_SERVER_PROCESS
    if UNIDOCK_SERVER_PROCESS is not None:
        print("\nCleaning up Uni-Dock server...")
        try:
            UNIDOCK_SERVER_PROCESS.terminate()
            UNIDOCK_SERVER_PROCESS.wait(timeout=5)
            print("Uni-Dock server terminated successfully")
        except subprocess.TimeoutExpired:
            print("Force killing Uni-Dock server...")
            UNIDOCK_SERVER_PROCESS.kill()
            UNIDOCK_SERVER_PROCESS.wait()
        except Exception as e:
            print(f"Error cleaning up Uni-Dock server: {e}")
        UNIDOCK_SERVER_PROCESS = None

atexit.register(cleanup_unidock_server)


def check_port_in_use(port):
    """Check if a port is already in use"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except OSError:
            return True


def start_unidock_server(gpu_id, port):
    """Start Uni-Dock ZMQ server on specified GPU and port"""
    global UNIDOCK_SERVER_PROCESS

    if check_port_in_use(port):
        print(f"Port {port} is already in use, assuming server is running")
        return None

    print(f"\nStarting Uni-Dock Server on GPU {gpu_id}, Port {port}")

    server_script = os.environ.get('UNIDOCK_SERVER_SCRIPT', './unidock_zmq_server.py')
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"unidock_server_gpu{gpu_id}_port{port}.log")

    conda_activate = os.environ.get('UNIDOCK_CONDA_ACTIVATE', 'source ~/anaconda3/bin/activate unidock_env')
    server_cmd = f"{conda_activate} && python -u {server_script} --port {port} --gpu {gpu_id} > {log_file} 2>&1"

    try:
        process = subprocess.Popen(server_cmd, shell=True, executable='/bin/bash')
        UNIDOCK_SERVER_PROCESS = process

        print("  Waiting for server to initialize...")
        max_wait = 10
        elapsed = 0
        while elapsed < max_wait:
            time.sleep(0.5)
            elapsed += 0.5
            if process.poll() is not None:
                UNIDOCK_SERVER_PROCESS = None
                return None
            if check_port_in_use(port):
                print(f"  Uni-Dock server started (PID: {process.pid})")
                return process

        process.terminate()
        UNIDOCK_SERVER_PROCESS = None
        return None
    except Exception as e:
        print(f"Error starting Uni-Dock server: {e}")
        UNIDOCK_SERVER_PROCESS = None
        return None


# ============================================================================
# File Utilities
# ============================================================================
def get_ligand_info(sdf_path):
    """Get center coordinates and atom count from reference ligand"""
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
    except Exception as e:
        print(f"  Error reading ligand info: {e}")
        return None


def find_target_files(target_dir):
    """Find receptor PDBQT and reference ligand SDF files"""
    receptor_files = glob.glob(os.path.join(target_dir, "*_rec.pdbqt"))
    if not receptor_files:
        return None, None
    receptor_path = receptor_files[0]

    ligand_files = glob.glob(os.path.join(target_dir, "*.sdf"))
    if not ligand_files:
        return None, None
    min_files = [f for f in ligand_files if '_min_' in f]
    ligand_path = min_files[0] if min_files else ligand_files[0]

    return receptor_path, ligand_path


def save_molecules_pt(mol_list, output_path):
    """Save RDKit mol objects as .pt file"""
    mol_data = []
    for rd_mol, ligand_path in mol_list:
        if rd_mol is not None:
            mol_data.append({
                'mol': rd_mol,
                'smiles': Chem.MolToSmiles(rd_mol) if rd_mol else None,
                'path': ligand_path
            })
    torch.save(mol_data, output_path)
    return len(mol_data)


# ============================================================================
# Model Loading
# ============================================================================
def load_model_and_task():
    """Load model and task"""
    print("\n" + "="*70)
    print("Loading Model...")
    print("="*70)

    checkpoint_path = './checkpoints/checkpoint_last.pt'  # Path to pretrained checkpoint
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

    print("  Model loaded successfully")
    return model, task


# ============================================================================
# Main Processing Functions
# ============================================================================
def generate_molecules_from_latent(model, sampler, latent_input, num_atom, batch_size,
                                    num_tokens, emb_dim, min_atoms, max_atoms):
    """Generate molecules from latent input"""
    mol_list, stats = sampler.sample_molecules(
        model=model,
        latent=latent_input,
        num_atom=num_atom,
        batch_size=batch_size,
        num_tokens=num_tokens,
        emb_dim=emb_dim,
        min_atoms=min_atoms,
        max_atoms=max_atoms,
        use_dopri5=False,
        temperature=1.0,
        optimize=False
    )
    return mol_list, stats


def evaluate_molecules(mol_list, receptor, center_x, center_y, center_z,
                       output_dir, unidock_port, search_mode="balance"):
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
        receptor=receptor,
        center_x=center_x, center_y=center_y, center_z=center_z,
        size_x=25.0, size_y=25.0, size_z=25.0,
        output_dir=output_dir,
        scoring="vina", search_mode=search_mode, num_modes=1,
        port=unidock_port
    )

    # SA scores
    sa_scores = calculate_sa_scores(mol_list)

    # Collect results
    results = []
    for rd_mol, ligand_path in mol_list:
        ligand_name = os.path.basename(ligand_path)
        docking = affinities.get(ligand_name, 0.0)
        sa = sa_scores.get(ligand_name, 0.5)
        if docking != 0.0:
            results.append({'docking': docking, 'sa': sa, 'name': ligand_name, 'mol': rd_mol, 'path': ligand_path})

    if len(results) == 0:
        return [], 0.0, 0.0, 0

    mean_docking = sum(r['docking'] for r in results) / len(results)
    mean_sa = sum(r['sa'] for r in results) / len(results)

    return results, mean_docking, mean_sa, len(results)


def process_single_target(model, task, target_name, target_dir, output_base_dir,
                          unidock_port, args):
    """Process a single target with baseline and dual optimization"""
    print("\n" + "="*70)
    print(f"Processing Target: {target_name}")
    print("="*70)

    # Find receptor and reference ligand
    receptor_path, ref_ligand_path = find_target_files(target_dir)
    if not receptor_path or not ref_ligand_path:
        print(f"  Required files not found in {target_dir}")
        return None

    print(f"  Receptor: {os.path.basename(receptor_path)}")
    print(f"  Reference ligand: {os.path.basename(ref_ligand_path)}")

    # Get ligand info
    ligand_info = get_ligand_info(ref_ligand_path)
    if ligand_info is None:
        print(f"  Failed to read reference ligand info")
        return None

    center_x, center_y, center_z, ref_num_atoms = ligand_info
    print(f"  Center: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})")
    print(f"  Reference atoms: {ref_num_atoms}")

    # Determine atom count
    if ref_num_atoms <= 20:
        num = random.randint(30, 40)
        print(f"  Using random atom count: {num}")
    else:
        num = ref_num_atoms
        print(f"  Using reference atom count: {num}")

    # Parameters
    min_atoms = 10
    max_atoms = max(60, num + 10)
    num_tokens = max(64, num + 10)
    emb_dim = 8

    # Create output directory structure
    target_output_dir = os.path.join(output_base_dir, target_name)
    baseline_dir = os.path.join(target_output_dir, "baseline")
    optimized_dir = os.path.join(target_output_dir, "optimized")
    cma_temp_dir = os.path.join(target_output_dir, "cma_temp")

    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(optimized_dir, exist_ok=True)
    os.makedirs(cma_temp_dir, exist_ok=True)

    # Create vocabulary dict
    vocab_dict = {}
    if hasattr(task, 'dictionary') and hasattr(task.dictionary, 'symbols'):
        for idx, symbol in enumerate(task.dictionary.symbols):
            vocab_dict[idx] = symbol

    results = {}

    # ========================================
    # BASELINE: No optimization
    # ========================================
    print(f"\n  --- Baseline (No Optimization) ---")
    set_random_seed(args.seed)

    # Create sampler for baseline (output to baseline_dir)
    baseline_sampler = MoleculeSampler(
        vocab_dict=vocab_dict,
        output_dir=baseline_dir,
        device='cuda'
    )

    baseline_latent = torch.randn(args.num_final_samples, num, emb_dim)
    baseline_input = torch.cat([
        baseline_latent,
        torch.randn(args.num_final_samples, num_tokens - num, emb_dim)
    ], dim=1).to(baseline_sampler.device)
    num_atom = torch.full((args.num_final_samples,), num, device=baseline_sampler.device)

    baseline_mol_list, _ = generate_molecules_from_latent(
        model, baseline_sampler, baseline_input, num_atom, args.num_final_samples,
        num_tokens, emb_dim, min_atoms, max_atoms
    )

    baseline_results, baseline_docking, baseline_sa, baseline_valid = evaluate_molecules(
        baseline_mol_list, receptor_path, center_x, center_y, center_z,
        baseline_dir, unidock_port
    )

    # Save baseline molecules
    if baseline_results:
        save_molecules_pt([(r['mol'], r['path']) for r in baseline_results],
                         os.path.join(baseline_dir, "molecules.pt"))
        # Copy pdbqt files - need to find the actual docked files
        # Note: pdbqt files are in baseline_dir after docking

    print(f"  Baseline: Docking={baseline_docking:.2f}, SA={baseline_sa:.4f}, Valid={baseline_valid}")

    results['baseline'] = {
        'mean_docking': baseline_docking,
        'mean_sa': baseline_sa,
        'num_valid': baseline_valid
    }

    # ========================================
    # DUAL OPTIMIZATION: Docking + SA
    # ========================================
    print(f"\n  --- Dual Optimization (Docking + SA) ---")
    print(f"  Population: {args.population_size}, Generations: {args.num_generations}")
    print(f"  SA Weight: {args.sa_weight}")

    set_random_seed(args.seed)

    # Create sampler for CMA-ES optimization (output to cma_temp_dir)
    cma_sampler = MoleculeSampler(
        vocab_dict=vocab_dict,
        output_dir=cma_temp_dir,
        device='cuda'
    )

    optimizer = CMA(mean=np.zeros(num * emb_dim), sigma=1.0,
                   population_size=args.population_size, lr_adapt=True, seed=args.seed)

    PENALTY_SCORE = 10.0

    for gen in range(args.num_generations):
        z_batch_list = []
        z_origin_list = []

        for _ in range(optimizer.population_size):
            z_flat = optimizer.ask()
            z_reshaped = z_flat.reshape(num, emb_dim)
            z_origin_list.append(z_flat)
            z_batch_list.append(z_reshaped)

        z_tensor = torch.tensor(np.stack(z_batch_list), dtype=torch.float32)
        input_tensor = torch.cat([
            z_tensor,
            torch.randn(args.population_size, num_tokens - num, emb_dim)
        ], dim=1).to(cma_sampler.device)

        num_atom_batch = torch.full((args.population_size,), num, device=cma_sampler.device)

        mol_list, _ = generate_molecules_from_latent(
            model, cma_sampler, input_tensor, num_atom_batch, args.population_size,
            num_tokens, emb_dim, min_atoms, max_atoms
        )
        mol_list = filter_single_fragment_molecules(mol_list)

        if len(mol_list) > 0:
            ligand_paths = [p for _, p in mol_list]
            affinities = dock_with_retry(
                ligand_paths=ligand_paths,
                receptor=receptor_path,
                center_x=center_x, center_y=center_y, center_z=center_z,
                size_x=25.0, size_y=25.0, size_z=25.0,
                output_dir=cma_temp_dir,
                scoring="vina", search_mode="fast", num_modes=1,
                port=unidock_port
            )

            sa_scores = calculate_sa_scores(mol_list)

            mol_index_to_score = {}
            for rd_mol, ligand_path in mol_list:
                ligand_name = os.path.basename(ligand_path)
                docking = affinities.get(ligand_name, 5.0)
                sa_norm = sa_scores.get(ligand_name, 0.5)

                if docking == 0.0:
                    combined = PENALTY_SCORE
                else:
                    sa_term = -sa_norm * 10
                    combined = docking + args.sa_weight * sa_term

                try:
                    mol_idx = int(ligand_name.split('_')[1].split('.')[0])
                    mol_index_to_score[mol_idx] = combined
                except:
                    pass

            solutions = []
            for idx in range(optimizer.population_size):
                z_flat = z_origin_list[idx]
                score = mol_index_to_score.get(idx + 1, PENALTY_SCORE)
                solutions.append((z_flat, score))

            optimizer.tell(solutions)

            if mol_index_to_score:
                best_score = min(mol_index_to_score.values())
                mean_score = sum(mol_index_to_score.values()) / len(mol_index_to_score)
                print(f"    Gen {gen}: Best={best_score:.2f}, Mean={mean_score:.2f}")
        else:
            solutions = [(z_origin_list[idx], PENALTY_SCORE) for idx in range(optimizer.population_size)]
            optimizer.tell(solutions)
            print(f"    Gen {gen}: No valid molecules")

    # Final sampling from optimized latent
    print(f"\n  Final sampling from optimized latent...")

    # Create sampler for optimized molecules (output to optimized_dir)
    optimized_sampler = MoleculeSampler(
        vocab_dict=vocab_dict,
        output_dir=optimized_dir,
        device='cuda'
    )

    mean_tensor = torch.tensor(optimizer._mean, dtype=torch.float32)
    sampled_latent = torch.stack([
        torch.normal(mean=mean_tensor, std=args.sampling_std)
        for _ in range(args.num_final_samples)
    ])
    input_raw = sampled_latent.view(args.num_final_samples, num, emb_dim)
    final_input = torch.cat([
        input_raw,
        torch.randn(args.num_final_samples, num_tokens - num, emb_dim)
    ], dim=1).to(optimized_sampler.device)

    num_atom_final = torch.full((args.num_final_samples,), num, device=optimized_sampler.device)

    optimized_mol_list, _ = generate_molecules_from_latent(
        model, optimized_sampler, final_input, num_atom_final, args.num_final_samples,
        num_tokens, emb_dim, min_atoms, max_atoms
    )

    optimized_results, optimized_docking, optimized_sa, optimized_valid = evaluate_molecules(
        optimized_mol_list, receptor_path, center_x, center_y, center_z,
        optimized_dir, unidock_port
    )

    # Save optimized molecules
    if optimized_results:
        save_molecules_pt([(r['mol'], r['path']) for r in optimized_results],
                         os.path.join(optimized_dir, "molecules.pt"))

    print(f"  Optimized: Docking={optimized_docking:.2f}, SA={optimized_sa:.4f}, Valid={optimized_valid}")

    results['optimized'] = {
        'mean_docking': optimized_docking,
        'mean_sa': optimized_sa,
        'num_valid': optimized_valid
    }

    # Print comparison
    print(f"\n  --- Comparison ---")
    print(f"  {'Metric':<15} {'Baseline':<15} {'Optimized':<15} {'Improvement':<15}")
    print(f"  {'-'*60}")

    dock_diff = optimized_docking - baseline_docking
    sa_diff = optimized_sa - baseline_sa

    print(f"  {'Docking':<15} {baseline_docking:<15.2f} {optimized_docking:<15.2f} {dock_diff:+.2f}")
    print(f"  {'SA':<15} {baseline_sa:<15.4f} {optimized_sa:<15.4f} {sa_diff:+.4f}")
    print(f"  {'Valid':<15} {baseline_valid:<15} {optimized_valid:<15}")

    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Multi-Target Ligand Generation with Dual Optimization')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--test-set-dir', type=str,
                       default="./data/test_set",
                       help='Test set directory')
    parser.add_argument('--output-dir', type=str, default="./multi_target_dual_results",
                       help='Output directory')
    parser.add_argument('--population-size', type=int, default=50,
                       help='CMA-ES population size (default: 50)')
    parser.add_argument('--num-generations', type=int, default=10,
                       help='CMA-ES generations (default: 10)')
    parser.add_argument('--num-final-samples', type=int, default=100,
                       help='Number of final samples per target (default: 100)')
    parser.add_argument('--sampling-std', type=float, default=0.1,
                       help='Sampling std from optimized mean (default: 0.1)')
    parser.add_argument('--sa-weight', type=float, default=1.0,
                       help='SA weight in dual optimization (default: 1.0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--target', type=str, default=None,
                       help='Specific target name (if not set, process all targets)')
    parser.add_argument('--num-targets', type=int, default=None,
                       help='Number of targets to process (if not set, process all)')
    args = parser.parse_args()

    gpu_id = args.gpu
    unidock_port = 5556 + gpu_id

    # Create output directory with parameters
    param_str = f"pop{args.population_size}_gen{args.num_generations}_saw{args.sa_weight}"
    output_base_dir = os.path.join(args.output_dir, param_str)

    print("\n" + "="*70)
    print("  Multi-Target Dual Optimization")
    print("="*70)
    print(f"  GPU: {gpu_id}")
    print(f"  Port: {unidock_port}")
    print(f"  Population Size: {args.population_size}")
    print(f"  Generations: {args.num_generations}")
    print(f"  SA Weight: {args.sa_weight}")
    print(f"  Final Samples: {args.num_final_samples}")
    print(f"  Output: {output_base_dir}")
    print("="*70)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.makedirs(output_base_dir, exist_ok=True)

    # Start Uni-Dock server
    server_process = start_unidock_server(gpu_id, unidock_port)
    if server_process is None and not check_port_in_use(unidock_port):
        print("Failed to start Uni-Dock server")
        sys.exit(1)

    # Load model
    model, task = load_model_and_task()
    if model is None or task is None:
        print("Failed to load model")
        cleanup_unidock_server()
        sys.exit(1)

    # Get target directories
    if args.target:
        target_dirs = [os.path.join(args.test_set_dir, args.target)]
        if not os.path.isdir(target_dirs[0]):
            print(f"Target not found: {args.target}")
            sys.exit(1)
    else:
        target_dirs = sorted(glob.glob(os.path.join(args.test_set_dir, "*")))
        target_dirs = [d for d in target_dirs if os.path.isdir(d)]
        if args.num_targets:
            random.seed(args.seed)
            target_dirs = random.sample(target_dirs, min(args.num_targets, len(target_dirs)))

    print(f"\nProcessing {len(target_dirs)} targets...")

    # Process targets
    all_results = {}
    for i, target_dir in enumerate(target_dirs, 1):
        target_name = os.path.basename(target_dir)
        print(f"\n[{i}/{len(target_dirs)}] {target_name}")

        result = process_single_target(
            model, task, target_name, target_dir, output_base_dir,
            unidock_port, args
        )

        if result:
            all_results[target_name] = result

    # ========================================
    # Final Summary with Total Averages
    # ========================================
    print("\n" + "="*100)
    print("  FINAL SUMMARY - ALL TARGETS")
    print("="*100)

    if all_results:
        # Per-target results
        print(f"\n{'Target':<30} {'Baseline Dock':<15} {'Opt Dock':<15} {'Baseline SA':<15} {'Opt SA':<15}")
        print("-"*90)

        total_baseline_dock = []
        total_baseline_sa = []
        total_opt_dock = []
        total_opt_sa = []

        for target_name, result in all_results.items():
            b = result.get('baseline', {})
            o = result.get('optimized', {})

            b_dock = b.get('mean_docking', 0)
            b_sa = b.get('mean_sa', 0)
            o_dock = o.get('mean_docking', 0)
            o_sa = o.get('mean_sa', 0)

            if b_dock != 0:
                total_baseline_dock.append(b_dock)
                total_baseline_sa.append(b_sa)
            if o_dock != 0:
                total_opt_dock.append(o_dock)
                total_opt_sa.append(o_sa)

            print(f"{target_name:<30} {b_dock:<15.2f} {o_dock:<15.2f} {b_sa:<15.4f} {o_sa:<15.4f}")

        # Total averages
        print("-"*90)
        if total_baseline_dock and total_opt_dock:
            avg_b_dock = np.mean(total_baseline_dock)
            avg_b_sa = np.mean(total_baseline_sa)
            avg_o_dock = np.mean(total_opt_dock)
            avg_o_sa = np.mean(total_opt_sa)

            print(f"{'TOTAL AVERAGE':<30} {avg_b_dock:<15.2f} {avg_o_dock:<15.2f} {avg_b_sa:<15.4f} {avg_o_sa:<15.4f}")
            print("="*90)

            dock_improve = avg_o_dock - avg_b_dock
            sa_improve = avg_o_sa - avg_b_sa
            sa_improve_pct = (sa_improve / avg_b_sa * 100) if avg_b_sa > 0 else 0

            print(f"\n  Docking Improvement: {dock_improve:+.2f} kcal/mol")
            print(f"  SA Improvement: {sa_improve:+.4f} ({sa_improve_pct:+.2f}%)")

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_base_dir, f"all_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    print("\n" + "="*70)
    print("  Processing Complete")
    print("="*70)


if __name__ == "__main__":
    main()
