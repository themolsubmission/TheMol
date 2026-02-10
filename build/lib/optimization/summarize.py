"""
Summarize Docking Scores from Multi-Target Generation
Collects docking scores from all targets and organizes into CSV
"""

import os
import glob
import pandas as pd
import numpy as np
import torch


def parse_docking_output(output_pdbqt):
    """
    Extract docking score from Uni-Dock output PDBQT file

    Args:
        output_pdbqt: Path to Uni-Dock output PDBQT file

    Returns:
        Docking score (float) or None
    """
    try:
        # Read file directly to parse Uni-Dock RESULT section
        with open(output_pdbqt, 'r') as f:
            content = f.read()

        # Method 1: Look for REMARK VINA RESULT line (standard Vina/Uni-Dock format)
        # Format: "REMARK VINA RESULT:    -8.5      0.000      0.000"
        for line in content.split('\n'):
            if 'REMARK VINA RESULT:' in line:
                parts = line.split()
                # parts: ['REMARK', 'VINA', 'RESULT:', '-8.5', '0.000', '0.000']
                if len(parts) >= 4:
                    return float(parts[3])

        # Method 2: Look for ENERGY= line (alternative Uni-Dock format)
        if 'ENERGY=' in content:
            for line in content.split('\n'):
                if 'ENERGY=' in line:
                    # Parse: "ENERGY=   -9.014  LOWER_BOUND=..."
                    parts = line.split()
                    if len(parts) >= 2:
                        # parts[0] is 'ENERGY=', parts[1] is the actual value
                        return float(parts[1])

    except Exception as e:
        pass
    return None


def collect_target_scores(target_dir, target_name):
    """
    Collect docking scores for all ligands from a single target directory

    Reads final optimized results from the final_optimized subdirectory

    Args:
        target_dir: Path to target results directory
        target_name: Name of the target

    Returns:
        List of (target_name, ligand_name, docking_score) tuples
    """
    results = []

    # Look for final_optimized subdirectory
    final_dir = os.path.join(target_dir, "final_optimized")
    if not os.path.exists(final_dir):
        # Fallback to target_dir if final_optimized doesn't exist
        final_dir = target_dir

    # Find all output PDBQT files (pattern: molecule_*_out.pdbqt)
    output_files = glob.glob(os.path.join(final_dir, "molecule_*_out.pdbqt"))

    for output_file in output_files:
        ligand_name = os.path.basename(output_file).replace('_out.pdbqt', '.sdf')

        # Parse docking score from PDBQT
        score = parse_docking_output(output_file)

        # If parsing failed, try to extract from filename or default to 0.0
        if score is None:
            score = 0.0

        results.append((target_name, ligand_name, score))

    return results


def load_reference_scores(ref_pt_path):
    """
    Load pocket-wise scores from reference docking score file

    Args:
        ref_pt_path: Path to crossdocked_test_vina_docked_pose_checked.pt file

    Returns:
        dict: {pocket_name: reference_score}
    """
    if not os.path.exists(ref_pt_path):
        print(f"⚠ Reference file not found: {ref_pt_path}")
        return {}

    try:
        data = torch.load(ref_pt_path, weights_only=False)
        pocket_ref_scores = {}

        for item in data:
            ligand_filename = item.get('ligand_filename', '')
            # Extract pocket name (part before first '/')
            pocket_name = ligand_filename.split('/')[0] if '/' in ligand_filename else ''

            # Extract vina dock affinity
            vina_data = item.get('vina', {})
            dock_data = vina_data.get('dock', [])
            if dock_data and len(dock_data) > 0:
                affinity = dock_data[0].get('affinity', None)
                if affinity is not None and pocket_name:
                    if pocket_name not in pocket_ref_scores:
                        pocket_ref_scores[pocket_name] = []
                    pocket_ref_scores[pocket_name].append(float(affinity))

        # Calculate mean per pocket
        pocket_mean_scores = {}
        for pocket, scores in pocket_ref_scores.items():
            pocket_mean_scores[pocket] = np.mean(scores)

        print(f"✓ Loaded reference scores for {len(pocket_mean_scores)} pockets")
        return pocket_mean_scores

    except Exception as e:
        print(f"⚠ Error loading reference file: {e}")
        return {}


def main():
    """
    Main function: Calculate mean docking energy per pocket and overall statistics
    """
    print("="*70)
    print("  Pocket-wise Mean Docking Energy Analysis")
    print("="*70)

    # Settings
    results_base_dir = "./multi_target_results"
    output_csv = "./pocket_mean_docking_energies.csv"
    ref_pt_path = "./data/crossdocked_test_vina_docked_pose_checked.pt"  # Path to reference scores file

    if not os.path.exists(results_base_dir):
        print(f"✗ Results directory not found: {results_base_dir}")
        return

    # Load reference scores
    print("\nLoading reference docking scores...")
    ref_scores = load_reference_scores(ref_pt_path)

    # Find all target directories (pockets)
    target_dirs = sorted(glob.glob(os.path.join(results_base_dir, "*")))
    target_dirs = [d for d in target_dirs if os.path.isdir(d)]

    print(f"\nFound {len(target_dirs)} pockets")
    print(f"Collecting docking energies from final_optimized directories...\n")

    # Collect pocket-wise mean energies
    pocket_means = []

    for i, target_dir in enumerate(target_dirs, 1):
        pocket_name = os.path.basename(target_dir)

        # Find all docked SDF files in final_optimized
        final_dir = os.path.join(target_dir, "final_optimized")

        if not os.path.exists(final_dir):
            print(f"[{i}/{len(target_dirs)}] {pocket_name}: ✗ No final_optimized directory")
            continue

        output_files = glob.glob(os.path.join(final_dir, "*_out.pdbqt"))

        if len(output_files) == 0:
            print(f"[{i}/{len(target_dirs)}] {pocket_name}: ✗ No output SDF files")
            continue

        # Parse all docking energies for this pocket
        energies = []
        for output_file in output_files:
            energy = parse_docking_output(output_file)
            if energy is not None and energy < 0:  # Only valid negative energies
                energies.append(energy)

        if len(energies) > 0:
            mean_energy = np.mean(energies)
            median_energy = np.median(energies)
            ref_score = ref_scores.get(pocket_name, None)

            pocket_data = {
                'Pocket': pocket_name,
                'Num_Ligands': len(energies),
                'Mean_Docking_Energy': mean_energy,
                'Median_Docking_Energy': median_energy,
                'Std_Docking_Energy': np.std(energies),
                'Best_Energy': np.min(energies),
                'Worst_Energy': np.max(energies),
                'Reference_Energy': ref_score if ref_score is not None else np.nan
            }
            pocket_means.append(pocket_data)

            # Include reference score in output
            if ref_score is not None:
                diff = mean_energy - ref_score
                diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
                print(f"[{i}/{len(target_dirs)}] {pocket_name}: {len(energies)} ligands, Mean = {mean_energy:.2f}, Ref = {ref_score:.2f} (Δ{diff_str})")
            else:
                print(f"[{i}/{len(target_dirs)}] {pocket_name}: {len(energies)} ligands, Mean = {mean_energy:.2f} kcal/mol (no ref)")
        else:
            print(f"[{i}/{len(target_dirs)}] {pocket_name}: ✗ No valid docking energies")

    # Check if we have any results
    if len(pocket_means) == 0:
        print("\n✗ No valid pocket results found")
        return

    # Create DataFrame
    df = pd.DataFrame(pocket_means)
    df = df.sort_values('Mean_Docking_Energy')  # Sort by mean energy (best first)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Pocket-wise results saved to: {output_csv}")

    # Calculate overall statistics (mean and std of pocket means)
    print("\n" + "="*70)
    print("  Overall Statistics Across All Pockets")
    print("="*70)

    pocket_mean_values = df['Mean_Docking_Energy'].values
    overall_mean_of_means = np.mean(pocket_mean_values)
    overall_std_of_means = np.std(pocket_mean_values)

    # Calculate overall average energy across ALL ligands (not just pocket means)
    all_energies = []
    for target_dir in target_dirs:
        final_dir = os.path.join(target_dir, "final_optimized")
        if os.path.exists(final_dir):
            output_files = glob.glob(os.path.join(final_dir, "*_out.pdbqt"))
            for output_file in output_files:
                energy = parse_docking_output(output_file)
                if energy is not None and energy < 0:
                    all_energies.append(energy)

    overall_avg_energy = np.mean(all_energies) if all_energies else 0.0
    overall_std_energy = np.std(all_energies) if all_energies else 0.0

    print(f"Total pockets analyzed: {len(pocket_means)}")
    print(f"Total ligands: {int(df['Num_Ligands'].sum())}")
    print(f"\nPocket-wise Mean Docking Energies:")
    print(f"  Mean of pocket means: {overall_mean_of_means:.3f} ± {overall_std_of_means:.3f} kcal/mol")
    print(f"  Best pocket mean: {pocket_mean_values.min():.3f} kcal/mol ({df.iloc[0]['Pocket']})")
    print(f"  Worst pocket mean: {pocket_mean_values.max():.3f} kcal/mol ({df.iloc[-1]['Pocket']})")
    print(f"\nOverall Average Across All Ligands:")
    print(f"  Mean: {overall_avg_energy:.3f} ± {overall_std_energy:.3f} kcal/mol")
    print(f"  (calculated from {len(all_energies)} ligands across all pockets)")

    # Reference score comparison statistics
    ref_df = df.dropna(subset=['Reference_Energy'])
    if len(ref_df) > 0:
        ref_mean_values = ref_df['Reference_Energy'].values
        overall_ref_mean = np.mean(ref_mean_values)
        overall_ref_std = np.std(ref_mean_values)

        # Calculate difference
        diff_values = ref_df['Mean_Docking_Energy'].values - ref_df['Reference_Energy'].values
        mean_diff = np.mean(diff_values)
        std_diff = np.std(diff_values)

        print(f"\nReference Score Comparison ({len(ref_df)} pockets with reference):")
        print(f"  Reference mean: {overall_ref_mean:.3f} ± {overall_ref_std:.3f} kcal/mol")
        print(f"  Mean difference (Generated - Reference): {mean_diff:.3f} ± {std_diff:.3f} kcal/mol")
        print(f"  Pockets better than reference: {(diff_values < 0).sum()}/{len(diff_values)}")
        print(f"  Pockets worse than reference: {(diff_values > 0).sum()}/{len(diff_values)}")

    # Display top 10 pockets by mean energy
    print("\n" + "="*70)
    print("  Top 10 Pockets by Mean Docking Energy")
    print("="*70)
    print(f"{'Rank':<6}{'Pocket':<30}{'Mean':<10}{'Ref':<10}{'Diff':<10}{'N':<6}")
    print("-"*70)

    for i, (idx, row) in enumerate(df.head(10).iterrows(), 1):
        ref_val = row['Reference_Energy']
        if pd.notna(ref_val):
            diff = row['Mean_Docking_Energy'] - ref_val
            diff_str = f"{diff:+.2f}"
            ref_str = f"{ref_val:.2f}"
        else:
            diff_str = "N/A"
            ref_str = "N/A"
        print(f"{i:<6}{row['Pocket']:<30}{row['Mean_Docking_Energy']:<10.2f}{ref_str:<10}{diff_str:<10}{int(row['Num_Ligands']):<6}")

    print("\n" + "="*70)
    print("  Analysis Complete")
    print("="*70)
    print(f"Results saved to: {output_csv}")
    print("="*70)


if __name__ == "__main__":
    main()
