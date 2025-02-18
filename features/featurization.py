r"""
Node and Edge features for datasets NMRDataset

Node features:
    - Atomic number (one-hot)
    - Chirality
    - Hybridization
    - Degree
    - Formal charge
    - Number of radical electrons
    - Number of hydrogen bonds
    - Implicit valence
    - Is aromatic
    - Is in ring

Bond features:
    - Bond type (one-hot)
    - Conjugated
    - Stereo
"""

import numpy as np

from typing import List, Dict
from rdkit import Chem


# Atom Features and Bond Features
ATOM_FEATURES = {
    'atomic_number': list(range(1, 119)),
    'chirality': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
    'degree': list(range(1, 6)),
    'formal_charge': list(range(-3, 4)),
    'num_radical_electrons': list(range(0, 4)),
    'num_hydrogen_bonds': list(range(0, 5)),
    'implicit_valence': list(range(0, 5)),
    'is_aromatic': [0, 1],
    'is_in_ring': [0, 1]
}

BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'is_conjugated': [0, 1],
    'stereo': [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS
    ]
}


# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))


def safe_index(lst, item):
    r"""
    Safely get the index of an item in a list.
    
    Parameters
    ----------
    lst : List
        The list to search in.
    item : Any
        The item to search for.

    Returns
    -------
    int
        The index of the item in the list, or -1 if it is not found.
    """
    try:
        return lst.index(item)
    except ValueError:
        return len(lst) - 1
    


def get_atom_fdim(atom_features: Dict[str, List[int]] = ATOM_FEATURES) -> int:
    r"""
    Computes the dimension of the atom features.

    Parameters
    ----------
    atom_features : Dict[str, List[int]], optional
        The atom features to consider.

    Returns
    -------
    int
        The dimension of the atom features.
    """
    return list(
        map(len, [
            atom_features['atomic_number'],
            atom_features['chirality'],
            atom_features['hybridization'],
            atom_features['degree'],    
            atom_features['formal_charge'],
            atom_features['num_radical_electrons'],
            atom_features['num_hydrogen_bonds'],
            atom_features['implicit_valence'],
            atom_features['is_aromatic'],
            atom_features['is_in_ring']
        ])
    )


def get_bond_fdim(bond_features: Dict[str, List[int]] = BOND_FEATURES) -> int:
    r"""
    Computes the dimension of the bond features.

    Parameters
    ----------
    bond_features : Dict[str, List[int]], optional
        The bond features to consider.

    Returns
    -------
    int
        The dimension of the bond features.
    """
    return list(
        map(len, [
            bond_features['bond_type'],
            bond_features['is_conjugated'],
            bond_features['stereo']
        ])
    )
    

def get_atom_features(atom: Chem.Atom, atom_features: Dict[str, List[int]] = ATOM_FEATURES) -> List[int]:
    r"""
    Computes the atom features for a given atom.

    Parameters
    ----------
    atom : Chem.Atom
        The atom for which to compute the features.
    atom_features : Dict[str, List[int]], optional
        The atom features to consider.

    Returns
    -------
    List[int]
        The computed atom features.
    """
    features = [
        safe_index(atom_features['atomic_number'], atom.GetAtomicNum()),
        safe_index(atom_features['chirality'], atom.GetChiralTag()),
        safe_index(atom_features['hybridization'], atom.GetHybridization()),
        safe_index(atom_features['degree'], atom.GetDegree()),
        safe_index(atom_features['formal_charge'], atom.GetFormalCharge()),
        safe_index(atom_features['num_radical_electrons'], atom.GetNumRadicalElectrons()),
        safe_index(atom_features['num_hydrogen_bonds'], atom.GetTotalNumHs()),
        safe_index(atom_features['implicit_valence'], atom.GetImplicitValence()),
        atom_features['is_aromatic'].index(int(atom.GetIsAromatic())),
        atom_features['is_in_ring'].index(int(atom.IsInRing()))
    ]
                               
    return features


def get_bond_features(bond: Chem.Bond, bond_features: Dict[str, List[int]] = BOND_FEATURES) -> List[int]:
    r"""
    Computes the bond features for a given bond.

    Parameters
    ----------
    bond : Chem.Bond
        The bond for which to compute the features.
    bond_features : Dict[str, List[int]], optional
        The bond features to consider.

    Returns
    -------
    List[int]
        The computed bond features.
    """
    features = [
        safe_index(bond_features['bond_type'], bond.GetBondType()),
        bond_features['is_conjugated'].index(int(bond.GetIsConjugated())),
        safe_index(bond_features['stereo'], bond.GetStereo())
    ]
    
    return features


def mol2graph(mol: Chem.Mol) -> Dict:
    r"""
    Converts a molecule to a graph representation.
    
    Parameters
    ----------
    mol : Chem.Mol
        The molecule to convert to a graph.

    Returns
    -------
    Dict
        The graph representation of the molecule.
    """
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(get_atom_features(atom))
    
    x = np.array(atom_features_list, dtype=np.int64)
    
    coords = mol.GetConformer().GetPositions()
    z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:
        edge_indices = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            
            edge_indices.append((i, j))
            edge_features_list.append(get_bond_features(bond))
            edge_indices.append((j, i))
            edge_features_list.append(get_bond_features(bond))
        
        edge_index = np.array(edge_indices, dtype=np.int64).T
        edge_attr = np.array(edge_features_list, dtype=np.int64)
    
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, num_bond_features), dtype=np.int64)
        
    graph = dict()
    
    graph['num_nodes'] = len(x)
    graph['node_feat'] = x
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['pos'] = coords
    graph['z'] = z
    
    return graph


if __name__ == '__main__':
    atom_fdim = get_atom_fdim()
    bond_fdim = get_bond_fdim()
    print(f"Atom feature dimension: {atom_fdim}")
    print(f"Bond feature dimension: {bond_fdim}")