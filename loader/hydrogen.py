import os
import ast
import torch
import numpy as np

from tqdm import tqdm
from rdkit import Chem, RDLogger
from torch_geometric.data import InMemoryDataset, Data

from loader.process import extract_hydrogen_shift, is_valid_molecule, mol_to_graph


# Disable rdkit warnings
RDLogger.DisableLog('rdApp.*')


class HydrogenDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(HydrogenDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def processed_dir(self):
        return os.path.join(self.root, 'hydrogen/processed')
    
    @property
    def raw_dir(self):
        return os.path.join(self.root, 'hydrogen/raw')
        
    @property
    def processed_file_names(self):
        return 'processed.pt'
    
    @property
    def raw_file_names(self):
        return 'hydrogen_dataset.sdf'
    
    def process(self):
        sdf_file = os.path.join(self.raw_dir, self.raw_file_names)
        suppl = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=True)

        data_list = []
        
        for mol in tqdm(suppl, desc='Processing data', total=len(suppl)):
            # check if the molecule is valid
            if not is_valid_molecule(mol):
                continue
            
            # extract carbon shifts
            hydrogen_shifts = extract_hydrogen_shift(mol)
            
            for i, atom in enumerate(mol.GetAtoms()):
                if i in hydrogen_shifts:
                    atom.SetProp('shift', str(hydrogen_shifts[i]))
                    atom.SetBoolProp('mask', True)
                else: 
                    atom.SetProp('shift', str([0]))
                    atom.SetBoolProp('mask', False)
            
            graph = mol_to_graph(mol)

            # create data object
            data = Data()
            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.long)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.long)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.long)
            
            data.z = torch.tensor(graph["z"]).to(torch.long)
            data.pos = torch.tensor(graph["position"]).to(torch.float)
            
            # add carbon shifts
            shift = np.array([ast.literal_eval(atom.GetProp('shift')) for atom in mol.GetAtoms()])
            mask = np.array([atom.GetBoolProp('mask') for atom in mol.GetAtoms()])
            data.mask = torch.from_numpy(mask).to(torch.bool)
            data.y = torch.from_numpy(shift).to(torch.float)
            
            data_list.append(data)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        torch.save(self.collate(data_list), self.processed_paths[0])
        