import subprocess
import os
import sys

class GninaDocker:
    def __init__(self, gnina_path="gnina", device=3):  # Configure gnina path
        """
        Args:
            gnina_path (str): Absolute path to the gnina executable.
            device (int): GPU device ID to use. Default is 3.
        """
        self.gnina_path = gnina_path
        self.device = str(device)
        
        if not os.path.exists(self.gnina_path):
            print(f"Warning: gnina executable not found at {self.gnina_path}")

    def dock(self, receptor, ligand, autobox_ligand, output, cnn="crossdock_default2018", num_modes=1):
        """
        Args:
            receptor (str): Path to the receptor PDB file.
            ligand (str): Path to the ligand SDF file.
            autobox_ligand (str): Path to the reference ligand SDF for autobox generation.
            output (str): Path where the docked SDF should be saved.
            cnn (str): GNINA CNN model to use. Default is "crossdock_default2018".
            num_modes (int): Number of binding modes to generate. Default is 1.

        Returns:
            dict: Dictionary containing 'status', 'stdout', 'stderr', and 'returncode'.
        """
        # Ensure absolute paths
        receptor = os.path.abspath(receptor)
        ligand = os.path.abspath(ligand)
        autobox_ligand = os.path.abspath(autobox_ligand)
        output = os.path.abspath(output)

        cmd = [
            self.gnina_path,
            "-r", receptor,
            "-l", ligand,
            "--autobox_ligand", autobox_ligand,
            "-o", output,
            "--device", self.device,
            "--cnn", cnn,
            "--num_modes", str(num_modes)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

            affinity = None
            if result.returncode == 0:
                # Parse affinity from stdout
                # Expected format:
                # mode |  affinity  | ...
                # -----+------------+ ...
                #    1       -5.31    ...
                for line in result.stdout.splitlines():
                    if line.strip().startswith("1") and "kcal/mol" not in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                affinity = float(parts[1])
                                break
                            except ValueError:
                                pass

            return {
                'status': 'success' if result.returncode == 0 else 'error',
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'affinity': affinity
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'stderr': str(e),
                'stdout': ''
            }
