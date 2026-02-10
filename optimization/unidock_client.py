import zmq
import json
import os
import sys
import glob
import re

class UniDockClient:
    def __init__(self, port=5556, host="localhost"):
        """
        Initialize the Uni-Dock ZeroMQ Client.

        Args:
            port (int): The port the server is listening on. Default is 5556.
            host (str): The hostname of the server. Default is "localhost".
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.address = f"tcp://{host}:{port}"
        try:
            self.socket.connect(self.address)
            print(f"Connected to ZeroMQ Uni-Dock Server at {self.address}")
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            raise

    def dock(self, receptor, ligand, center_x, center_y, center_z,
             size_x, size_y, size_z, output_dir,
             scoring="vina", search_mode="balance", num_modes=1,
             exhaustiveness=None, max_step=None, seed=0, verbosity=1):
        """
        Request docking from the Uni-Dock server.

        Args:
            receptor (str): Path to the receptor PDBQT file.
            ligand (str or list): Path to the ligand PDBQT or SDF file, or list of ligand paths for batch docking.
            center_x (float): X coordinate of the center (Angstrom).
            center_y (float): Y coordinate of the center (Angstrom).
            center_z (float): Z coordinate of the center (Angstrom).
            size_x (float): Size in the X dimension (Angstrom).
            size_y (float): Size in the Y dimension (Angstrom).
            size_z (float): Size in the Z dimension (Angstrom).
            output_dir (str): Output directory for docked results.
            scoring (str): Scoring function (vina, vinardo, or ad4). Default is "vina".
            search_mode (str): Search mode (fast, balance, detail). Default is "balance".
            num_modes (int): Maximum number of binding modes to generate. Default is 1.
            exhaustiveness (int): Exhaustiveness of the global search. Optional.
            max_step (int): Maximum number of steps in each MC run. Optional.
            seed (int): Explicit random seed. Default is 0.
            verbosity (int): Verbosity level (0=no output, 1=normal, 2=verbose). Default is 1.

        Returns:
            dict: The response from the server containing status, stdout, stderr, etc.
                  For single ligand: Additional key 'affinity' contains the docking score.
                  For batch ligands: Additional key 'affinities' contains dict of {ligand_name: affinity}.
        """
        # Ensure absolute paths
        receptor = os.path.abspath(receptor)

        # Handle both single ligand and list of ligands
        if isinstance(ligand, list):
            ligand = [os.path.abspath(lig) for lig in ligand]
            is_batch = True
        else:
            ligand = os.path.abspath(ligand)
            is_batch = False

        output_dir = os.path.abspath(output_dir)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        request = {
            'receptor': receptor,
            'ligand': ligand,
            'center_x': center_x,
            'center_y': center_y,
            'center_z': center_z,
            'size_x': size_x,
            'size_y': size_y,
            'size_z': size_z,
            'output_dir': output_dir,
            'scoring': scoring,
            'search_mode': search_mode,
            'num_modes': num_modes,
            'seed': seed,
            'verbosity': verbosity
        }

        if exhaustiveness is not None:
            request['exhaustiveness'] = exhaustiveness
        if max_step is not None:
            request['max_step'] = max_step

        try:
            self.socket.send_json(request)
            response = self.socket.recv_json()

            # Parse affinity score from stdout if successful
            if response.get('status') == 'success' and 'stdout' in response:
                if is_batch:
                    # Parse multiple affinities for batch docking
                    affinities = self._parse_batch_affinities(response['stdout'], ligand)
                    if affinities:
                        response['affinities'] = affinities
                else:
                    # Parse single affinity
                    affinity = self._parse_affinity(response['stdout'])
                    if affinity is not None:
                        response['affinity'] = affinity

            return response
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Communication error: {str(e)}"
            }

    def _parse_affinity(self, stdout):
        """
        Parse the affinity score from Uni-Dock stdout.

        Uni-Dock output format typically includes lines like:
        "mode |   affinity | dist from best mode"
        "     | (kcal/mol) | rmsd l.b.| rmsd u.b."
        "-----+------------+----------+----------"
        "   1       -8.532          0          0"

        Args:
            stdout (str): The stdout from Uni-Dock.

        Returns:
            float: The best affinity score, or None if not found.
        """
        try:
            lines = stdout.split('\n')
            for i, line in enumerate(lines):
                # Look for the header line with "mode" and "affinity"
                if 'affinity' in line.lower() and 'mode' in line.lower():
                    # Skip the next 2 lines (units line and separator line)
                    # The actual data is on the 3rd line after the header
                    data_line_idx = i + 3
                    if data_line_idx < len(lines):
                        data_line = lines[data_line_idx].strip()
                        parts = data_line.split()
                        # Format: "   1       -8.532          0          0"
                        # parts[0] = mode number, parts[1] = affinity
                        if len(parts) >= 2:
                            try:
                                affinity = float(parts[1])
                                return affinity
                            except ValueError:
                                # If parts[1] is not a number, continue searching
                                continue
            return None
        except Exception as e:
            print(f"Warning: Failed to parse affinity from stdout: {e}")
            return None

    def _parse_batch_affinities(self, stdout, ligand_files):
        """
        Parse affinity scores for batch docking from Uni-Dock stdout.

        Uni-Dock batch output shows multiple affinity tables in the same order as input ligands.
        Since ligand filenames are not included in the output, we match by order.

        Args:
            stdout (str): The stdout from Uni-Dock.
            ligand_files (list): List of ligand file paths in the order they were submitted.

        Returns:
            dict: Dictionary mapping ligand filename to affinity score {filename: affinity}.
        """
        try:
            affinities = {}
            lines = stdout.split('\n')

            # Extract ligand basenames
            ligand_basenames = [os.path.basename(lig) for lig in ligand_files]

            # Find all affinity scores in order
            affinity_scores = []

            for i, line in enumerate(lines):
                # Look for affinity table header
                if 'affinity' in line.lower() and 'mode' in line.lower():
                    # Skip the next 2 lines (units line and separator line)
                    # The actual data is on the 3rd line after the header
                    data_line_idx = i + 3
                    if data_line_idx < len(lines):
                        data_line = lines[data_line_idx].strip()
                        parts = data_line.split()
                        # Format: "   1       -8.532          0          0"
                        # parts[0] = mode number, parts[1] = affinity
                        if len(parts) >= 2:
                            try:
                                affinity = float(parts[1])
                                affinity_scores.append(affinity)
                            except ValueError:
                                pass

            # Match affinity scores to ligands by order
            if len(affinity_scores) == len(ligand_basenames):
                for ligand_name, affinity in zip(ligand_basenames, affinity_scores):
                    affinities[ligand_name] = affinity
                return affinities
            else:
                print(f"Warning: Found {len(affinity_scores)} affinity scores but expected {len(ligand_basenames)}")
                # Still try to match what we have
                for i, affinity in enumerate(affinity_scores):
                    if i < len(ligand_basenames):
                        affinities[ligand_basenames[i]] = affinity
                return affinities if affinities else None

        except Exception as e:
            print(f"Warning: Failed to parse batch affinities from stdout: {e}")
            return None

    def dock_and_get_score(self, receptor, ligand, center_x, center_y, center_z,
                          size_x, size_y, size_z, output_dir, **kwargs):
        """
        Simplified method that returns only the docking score.

        Args:
            Same as dock() method.
            **kwargs: Additional arguments passed to dock().

        Returns:
            float: Docking affinity score (kcal/mol), or 0.0 if docking failed.
        """
        result = self.dock(receptor, ligand, center_x, center_y, center_z,
                          size_x, size_y, size_z, output_dir, **kwargs)

        if result.get('status') == 'success':
            return result.get('affinity', 0.0)
        else:
            print(f"Docking failed: {result.get('stderr') or result.get('message')}")
            return 0.0

    def close(self):
        """Close the ZeroMQ socket."""
        self.socket.close()
        self.context.term()
