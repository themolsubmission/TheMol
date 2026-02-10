# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from rdkit import Chem
from rdkit.Chem import AllChem


class Add2DConformerDataset(BaseWrapperDataset):
    def __init__(self, dataset, smi, atoms, coordinates, max_atoms_filter=100):
        self.dataset = dataset
        self.smi = smi
        self.atoms = atoms
        self.coordinates = coordinates
        self.set_epoch(None)
        self.max_atoms_filter = max_atoms_filter
    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array(self.dataset[index][self.atoms])
        assert len(atoms) > 0
        
        if self.max_atoms_filter is not None and len(atoms) > self.max_atoms_filter:
            # Find next valid sample
            for offset in range(1, min(100, len(self.dataset))):
                next_index = (index + offset) % len(self.dataset)
                next_atoms = np.array(self.dataset[next_index][self.atoms])
                if len(next_atoms) <= self.max_atoms_filter:
                    index = next_index
                    atoms = next_atoms
                    break
                          
        smi = self.dataset[index][self.smi]
        coordinates_2d = smi2_2Dcoords(smi)
        coordinates = self.dataset[index][self.coordinates]
        # Convert to list if numpy.ndarray (for reordered lmdb compatibility)
        if isinstance(coordinates, np.ndarray):
            coordinates = [coordinates]
        coordinates.append(coordinates_2d)
        return {"smi": smi, "atoms": atoms, "coordinates": coordinates}

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


def smi2_2Dcoords(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(
        coordinates
    ), "2D coordinates shape is not align with {}".format(smi)
    return coordinates
