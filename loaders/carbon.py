import os
import torch
import numpy as np

from tqdm import tqdm
from rdkit import Chem, RDLogger
from torch_geometric.data import InMemoryDataset, Data

from features.featurization import mol2graph
from features.geometry import generate_geometry
from utils.extractor import extract_carbonShift, genrate_mask


# Disable rdkit warnings
RDLogger.DisableLog('rdApp.*')


class CarbonDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(CarbonDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return 'carbon_dataset.sdf'
    
    @property
    def processed_file_names(self):
        return 'carbon.pt'
    
    @property
    def raw_dir(self):
        return os.path.join(self.root, 'carbon', 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'carbon', 'processed')
    
    def process(self):
        suppl = Chem.SDMolSupplier(
            os.path.join(self.raw_dir, self.raw_file_names), 
            removeHs=False,
            sanitize=True
        )

        data_list = []
        
        for mol in tqdm(suppl, desc='Processing data', total=len(suppl)):
            if mol is None:
                continue
            
            atom_shifts = extract_carbonShift(mol)

            if len(atom_shifts) == None:
                continue
            
            # generate labels
            labels, others_mask, inferences_mask = genrate_mask(mol, atom_shifts)
            
            # generate graph
            graph = mol2graph(mol)
            
            # create data object
            data = Data()
            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.long)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.long)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.long)
            data.y = torch.Tensor(labels[:, 1]).to(torch.float)
            
            data.others_mask = torch.Tensor(others_mask).to(torch.bool)
            data.inferences_mask = torch.Tensor(inferences_mask).to(torch.bool)
            
            # generate conformers
            data.pos = torch.Tensor(graph['pos']).to(torch.float)
            data.z = torch.Tensor(graph['z']).to(torch.long)

            data_list.append(data)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        torch.save(self.collate(data_list), self.processed_paths[0])
