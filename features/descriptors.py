import numpy as np

from rdkit import Chem

from typing import Union

from descriptastorus.descriptors import rdNormalizedDescriptors, rdDescriptors


def generate_features(mol: Union[str, Chem.Mol]) -> np.ndarray:
    r"""
    Generates RDKit features for a molecule.
    
    Parameters
    ----------
    mol : Union[str, Chem.Mol]
        A molecule (i.e., either a SMILES or an RDKit molecule).
    
    Returns
    -------
    np.ndarray
        The generated features.
    """
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
    
    generator = rdDescriptors.RDKit2D()
    features = np.array(generator.process(smiles)[1:])

    return features


def generate_normailized_features(mol: Union[str, Chem.Mol]) -> np.ndarray:
    r"""
    Generates RDKit normailized features for a molecule.
    
    Parameters
    ----------
    mol : Union[str, Chem.Mol]
        A molecule (i.e., either a SMILES or an RDKit molecule).
    
    Returns
    -------
    np.ndarray
        The generated features.
    """
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
    
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    features = np.array(generator.process(smiles)[1:])
    
    return features


