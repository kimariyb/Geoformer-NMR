import numpy as np

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from typing import Union, Tuple


# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')


def generate_geometry(mol: Union[str, Chem.Mol]) -> Chem.Mol:
    r"""
    Generate 3D coordinates for a molecule using RDKit.

    Parameters
    ----------
    mol : Union[str, Chem.Mol]
        A SMILES or RDKit molecule.

    Returns
    -------
    Chem.Mol
        A RDKit molecule with 3D coordinates.
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)

    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42, clearConfs=False)
        AllChem.MMFFOptimizeMolecule(mol)
        
        return mol
    
    except:
        return None
        

