import numpy as np

from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolfiles, rdmolops


allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOATROPCW',
        'STEREOATROPCCW'
    ], 
    'possible_is_conjugated_list': [False, True],
}



# Set the RDKit Logger to only display errors and warnings
RDLogger.DisableLog('rdApp.*')


def extract_carbon_shift(mol: Chem.rdchem.Mol) -> dict:
    r"""
    This function takes a molecule as input and extracts the carbon shift value from its property.
    
    Parameters
    ----------
    mol : Chem.rdchem.Mol
        The input molecule.
    
    Returns
    -------
    dict{int: float}
        A dictionary containing the carbon shift values for each atom in the molecule.
    """
    # get the shift value from the molecule's property
    prop = mol.GetPropsAsDict()

    atom_shift = {}
    for key in prop.keys():
        if key.startswith('Spectrum 13C'):
            for shift in prop[key].split('|')[:-1]:
                [shift_val, _, shift_idx] = shift.split(';')
                shift_val, shift_idx = float(shift_val), int(shift_idx)

                if shift_idx not in atom_shift:
                    atom_shift[shift_idx] = []
                
                atom_shift[shift_idx].append(shift_val)
    
    for i in range(mol.GetNumAtoms()):
        if i in atom_shift:
            atom_shift[i] = np.median(atom_shift[i])
    
    return atom_shift


def extract_hydrogen_shift(mol: Chem.rdchem.Mol) -> dict:
    r"""
    This function takes a molecule as input and extracts the hydrogen shift value from its property.
    
    Parameters
    ----------
    mol : Chem.rdchem.Mol
        The input molecule.
    
    Returns
    -------
    dict{int: float}
        A dictionary containing the hydrogen shift values for each atom in the molecule.
    """
    # get the shift value from the molecule's property
    prop = mol.GetPropsAsDict()

    atom_shifts = {}
    for key in prop.keys():
        if key.startswith('Spectrum 1H'):
            tmp_dict = {}
            for shift in prop[key].split('|')[:-1]:
                [shift_val, _, shift_idx] = shift.split(';')
                shift_val, shift_idx = float(shift_val), int(shift_idx)
                
                if shift_idx not in atom_shifts: 
                    atom_shifts[shift_idx] = []

                if shift_idx not in tmp_dict:
                    tmp_dict[shift_idx] = []
                    
                tmp_dict[shift_idx].append(shift_val)
            
            for shift_idx in tmp_dict.keys():
                atom_shifts[shift_idx].append(tmp_dict[shift_idx])
                
    for shift_idx in atom_shifts.keys():
        max_len = np.max([len(shifts) for shifts in atom_shifts[shift_idx]])
        
        for i in range(len(atom_shifts[shift_idx])):
            if len(atom_shifts[shift_idx][i]) < max_len:
                if len(atom_shifts[shift_idx][i]) == 1:
                    atom_shifts[shift_idx][i] = [atom_shifts[shift_idx][i][0] for _ in range(max_len)]

                elif len(atom_shifts[shift_idx][i]) > 1:
                    while len(atom_shifts[shift_idx][i]) < max_len:
                        atom_shifts[shift_idx][i].append(np.mean(atom_shifts[shift_idx][i]))

            atom_shifts[shift_idx][i] = sorted(atom_shifts[shift_idx][i])

        atom_shifts[shift_idx] = np.median(atom_shifts[shift_idx], 0).tolist()
    
    return atom_shifts


def is_valid_molecule(mol: Chem.rdchem.Mol) -> bool:
    r"""
    This function takes a molecule as input and checks if it is valid or not.
    
    Parameters
    ----------
    mol : Chem.rdchem.Mol
        The input molecule.
    
    Returns
    -------
    bool
        True if the molecule is valid, False otherwise.
    """
    # check if the molecule is valid
    if mol is None:
        return False
    
    # check if the molecule has a valid number of atoms
    if mol.GetNumAtoms() < 1:
        return False
    
    # check if the molecule has a valid number of bonds
    if mol.GetNumBonds() < 1:
        return False
    
    return True


def safe_index(element, target_list: list) -> int:
    r"""
    This function takes an element and a list and returns the index of the element in the list.
    If the element is not in the list, it returns the index of the last element in the list.

    Parameters
    ----------
    element : Any
        The input element.
    target_list : list
        The target list.
    
    Returns
    -------
    int
        The index of the element in the list.
    """
    try:
        return target_list.index(element)
    except ValueError:
        return len(target_list) - 1


def atom_to_feature(atom: Chem.rdchem.Atom) -> list:
    r"""
    This function takes an atom as input and converts it to a feature vector.
    
    Parameters
    ----------
    atom : Chem.rdchem.Atom
        The input atom.
    
    Returns
    -------
    list
        A list containing the feature vector of the atom.
    """
    atom_feature = [
        safe_index(atom.GetAtomicNum(), allowable_features['possible_atomic_num_list']),
        safe_index(str(atom.GetChiralTag()), allowable_features['possible_chirality_list']),
        safe_index(atom.GetTotalDegree(), allowable_features['possible_degree_list']),
        safe_index(atom.GetFormalCharge(), allowable_features['possible_formal_charge_list']),
        safe_index(atom.GetTotalNumHs(), allowable_features['possible_numH_list']),
        safe_index(atom.GetNumRadicalElectrons(), allowable_features['possible_number_radical_e_list']),
        safe_index(str(atom.GetHybridization()), allowable_features['possible_hybridization_list']),
        allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
        allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
    ]
    
    return atom_feature

def bond_to_feature(bond: Chem.rdchem.Bond) -> list:
    r"""
    This function takes a bond as input and converts it to a feature vector.
    
    Parameters
    ----------
    bond : Chem.rdchem.Bond
        The input bond.
    
    Returns
    -------
    list
        A list containing the feature vector of the bond.
    """
    bond_feature = [
        safe_index(str(bond.GetBondType()), allowable_features['possible_bond_type_list']),
        allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
        allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
    ]
    
    return bond_feature


def mol_to_graph(
    mol: Chem.rdchem.Mol, 
    canonical_atom_order: bool = True, 
    explicit_hydrogens: bool = False
) -> dict:
    r"""
    This function takes a molecule as input and converts it to a graph representation.
    
    Parameters
    ----------
    mol : Chem.rdchem.Mol
        The input molecule.
    canonical_atom_order : bool, default=True
        Whether to use a canonical atom order. If False, the order of atoms in the input molecule is preserved. 
    explicit_hydrogens : bool, default=False
        Whether to have hydrogen atoms as explicit nodes. If False, they are implicit.
    
    
    Returns
    -------
    dict{int: list}
        A dictionary containing the graph representation of the molecule.
    """
    # Whether to have hydrogen atoms as explicit nodes
    if explicit_hydrogens:
        mol = Chem.AddHs(mol)
        
    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)

    # atom features
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    coords = mol.GetConformer().GetPositions()
    z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    
    # bond features
    num_bond_features = 3
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
            edges_feature = bond_to_feature(bond)
            
            edges_list.append((i, j))
            edge_features_list.append(edges_feature)
            edges_list.append((j, i))
            edge_features_list.append(edges_feature)
        
        edge_index = np.array(edges_list, dtype=np.int64).T
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = dict()
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['position'] = coords
    graph['z'] = z
    
    return graph