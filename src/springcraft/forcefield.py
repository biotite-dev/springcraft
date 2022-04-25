"""
This module contains functionality for computing interaction matrices,
i.e. Kirchhoff and Hessian matrices.
"""

__name__ = "springcraft"
__author__ = "Patrick Kunzmann"
__all__ = ["ForceField", "InvariantForceField", "TypeSpecificForceField"]

import numbers
import abc
from os.path import join, dirname, realpath
import numpy as np
import biotite.structure as struc
import biotite.sequence as seq


DATA_DIR = join(dirname(realpath(__file__)), "data")


N_AMINO_ACIDS = 20
AA_LIST = [
    seq.ProteinSequence.convert_letter_1to3(letter)
    # Omit ambiguous amino acids and stop signal
    for letter in seq.ProteinSequence.alphabet.get_symbols()[:N_AMINO_ACIDS]
]
AA_TO_INDEX = {aa : i for i, aa in enumerate(AA_LIST)}


class ForceField(metaclass=abc.ABCMeta):
    """
    Subclasses of this abstract base class define the force constants of
    the modeled springs between atoms in a *Elastic network model*.

    Attributes
    ----------
    natoms : int or None
        The number of atoms in the model.
        If a :class:`ForceField` does not depend on the respective
        atoms, i.e. `atom_i` and `atom_j` is unused in
        :meth:`force_constant()`, this attribute is ``None`` instead.
    """


    @abc.abstractmethod
    def force_constant(self, atom_i, atom_j, sq_distance):
        """
        Get the force constant for the interaction of the given atoms.

        ABSTRACT: Override when inheriting.

        Parameters
        ----------
        atom_i, atom_j : ndarray, shape=(n,), dtype=int
            The indices to the first and second atoms in each 
            interacting atom pair.
        sq_distance : ndarray, shape=(n,), dtype=float
            The distance between the atoms indicated by `atom_i` and
            `atom_j`.
        """
        pass

    @property
    def natoms(self):
        return None


class InvariantForceField(ForceField):
    """
    This force field treats every interaction with the same force
    constant.
    """

    def force_constant(self, atom_i, atom_j, sq_distance):
        return np.ones(len(atom_i))


class TypeSpecificForceField(ForceField):
    """
    This force field is able to treat amino acid and connection specific 
    interactions.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,)
        The atoms in the model.
        Must contain only `CA` atoms and only canonic amino acids.
        `CA` atoms with the same chain ID and adjacent residue IDs
        are treated as bonded
    bonded, intra_chain, inter_chain : float or ndarray, shape=(20, 20), dtype=float
        The force constants for interactions between each combination of
        amino acid type.
        The order of amino acids is alphabetically, i.e.
        ``'ALA'``, ``'CYS'``, ``'ASP'``, ``'GLU'``, ``'PHE'``, 
        ``'GLY'``, ``'HIS'``, ``'ILE'``, ``'LYS'``, ``'LEU'``,
        ``'MET'``, ``'ASN'``, ``'PRO'``, ``'GLN'``, ``'ARG'``,
        ``'SER'``, ``'THR'``, ``'VAL'``, ``'TRP'``, ``'TYR'``.
        `bonded` gives values for bonded amino acids,
        `intra_chain` gives values for non-bonded interactions within
        the same peptide chain and 
        `inter_chain` gives values for non-bonded interactions for amino
        acids in different chains.
        If the force constant is independent of the amino acid type,
        a float can be given instead.
    
    Attributes
    ----------
    natoms : int or None
        The number of atoms in the model.
        If a :class:`ForceField` does not depend on the respective
        atoms, i.e. `atom_i` and `atom_j` is unused in
        :meth:`force_constant()`, this attribute is ``None`` instead.
    interaction_matrix : ndarray, shape=(n, n), dtype=float
        Force constants between the atoms in `atoms`.
    """
    def __init__(self, atoms, bonded, intra_chain, inter_chain):
        if not isinstance(atoms, struc.AtomArray):
            raise TypeError(
                f"Expected 'AtomArray', not {type(atoms).__name__}"
            )
        if not np.all((atoms.atom_name == "CA") & (atoms.element == "C")):
            raise struc.BadStructureError(
                f"AtomArray does not contain exclusively CA atoms"
            )
        
        self._natoms = atoms.array_length()
        
        self._bonded = _convert_to_matrix(bonded)
        self._intra_chain = _convert_to_matrix(intra_chain)
        self._inter_chain = _convert_to_matrix(inter_chain)

        # Maps pos-specific indices to type-specific_indices
        matrix_indices = np.array([AA_TO_INDEX[aa] for aa in atoms.res_name])

        # Find peptide bonds
        continuous_res_id = np.diff(atoms.res_id) == 1
        continuous_chain_id = atoms.chain_id[:-1] == atoms.chain_id[1:]
        peptide_bond_i = np.where(continuous_res_id & continuous_chain_id)[0]


        ### Fill interaction matrix
        ## Handle non-bonded interactions
        # Cartesian product of pos-specific indices
        # for first and second amino acid
        pos_indices = (
            np.repeat(np.arange(self._natoms), self._natoms),
            np.tile(np.arange(self._natoms), self._natoms),
        )
        # Convert indices to type-specific_indices
        type_indices = (
            matrix_indices[pos_indices[0]],
            matrix_indices[pos_indices[1]]
        )
        intra_interactions = self._intra_chain[type_indices[0], type_indices[1]]
        inter_interactions = self._inter_chain[type_indices[0], type_indices[1]]
        interactions = np.where(
            atoms.chain_id[pos_indices[0]] == atoms.chain_id[pos_indices[1]],
            intra_interactions, inter_interactions)
        # Initialize pos-specific interaction matrix
        # For simplicity bonded interactions are also handled as
        # non-bonded interactions at this point,
        # as they are overwritten later
        self._interaction_matrix = interactions.reshape(
            (self._natoms, self._natoms)
        )

        ## Handle bonded interactions
        # Convert pos-specific indices to type-specific indices
        indices = (
            matrix_indices[peptide_bond_i],
            matrix_indices[peptide_bond_i+1]
        )
        constants = self._bonded[indices]
        # Overwrite previous values
        self._interaction_matrix[peptide_bond_i, peptide_bond_i+1] = constants
        self._interaction_matrix[peptide_bond_i+1, peptide_bond_i] = constants

        ## Handle interaction of atoms with itself
        np.fill_diagonal(self._interaction_matrix, 0)

    def force_constant(self, atom_i, atom_j, sq_distance):
        return self._interaction_matrix[atom_i, atom_j]

    @property
    def natoms(self):
        return self._natoms

    @property
    def interaction_matrix(self):
        return self._interaction_matrix
    
    @staticmethod
    def sd_enm(atoms):
        bonded = _load_matrix("sdenm_bonded.csv")
        intra  = _load_matrix("sdenm_intra.csv")
        inter  = _load_matrix("sdenm_inter.csv")
        return TypeSpecificForceField(atoms, bonded, intra, inter)

    @staticmethod
    def e_anm(atoms):
        bonded = _load_matrix("eanm_bonded.csv")
        intra  = _load_matrix("eanm_intra.csv")
        inter  = _load_matrix("eanm_inter.csv")
        return TypeSpecificForceField(atoms, bonded, intra, inter)


def _convert_to_matrix(value):
    if isinstance(value, numbers.Number):
        return np.full((N_AMINO_ACIDS, N_AMINO_ACIDS), value, dtype=np.float32)
    else:
        matrix = np.asarray(value, dtype=np.float32)
        if matrix.shape != (N_AMINO_ACIDS, N_AMINO_ACIDS):
            raise IndexError(
                f"Expected array of shape {(N_AMINO_ACIDS, N_AMINO_ACIDS)}, "
                f"got {matrix.shape}"
            )
        if not np.allclose(matrix, matrix.T):
            raise ValueError(
                "Input matrix is not symmetric"
            )
        return matrix


matrices = {}
def _load_matrix(fname):
    if fname in matrices:
        # Matrix was already loaded
        return matrices[fname]
    
    matrix = np.loadtxt(join(DATA_DIR, fname), delimiter=",")
    matrices[fname] = matrix
    return matrix
