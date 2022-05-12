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
    This force field is able to treat interactions differently based
    on interacting amino acids and distances.

    The distances are separated into bins.
    A `value` is within `bin[i]`, if `value <= distance_edges[i]`.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,)
        The atoms in the model.
        Must contain only `CA` atoms and only canonic amino acids.
        `CA` atoms with the same chain ID and adjacent residue IDs
        are treated as bonded
    bonded, intra_chain, inter_chain : float or ndarray, shape=(k,) or shape=(20, 20) or shape=(k, 20, 20), dtype=float
        The force constants for interactions between each combination of
        amino acid type and for each distance bin.
        The order of amino acids is alphabetically with respect to the
        one-letter code, i.e.
        ``'ALA'``, ``'CYS'``, ``'ASP'``, ``'GLU'``, ``'PHE'``, 
        ``'GLY'``, ``'HIS'``, ``'ILE'``, ``'LYS'``, ``'LEU'``,
        ``'MET'``, ``'ASN'``, ``'PRO'``, ``'GLN'``, ``'ARG'``,
        ``'SER'``, ``'THR'``, ``'VAL'``, ``'TRP'`` and ``'TYR'``.
        `bonded` gives values for bonded amino acids,
        `intra_chain` gives values for non-bonded interactions within
        the same peptide chain and 
        `inter_chain` gives values for non-bonded interactions for amino
        acids in different chains.
        The possible shapes are:

            - Scalar value:
              Same value for all amino acid types and distances.
            - 1-dim array:
              Individual value for each distance bin.
              Note that the array is one element larger than the
              `distance_edges`
            - 2-dim array:
              Individual value for each pair of amino acid types.
              Note the alphabetical order shown above.
            - 3-dim array:
              Individual value for each distance bin and pair of amino
              acid types.

    distance_edges : ndarray, shape=(k-1,), dtype=float, optional
        If set, interactions are handled based on the distance of the
        two atoms.
    
    Attributes
    ----------
    natoms : int or None
        The number of atoms in the model.
    interaction_matrix : ndarray, shape=(n, n, k), dtype=float
        Force constants between the atoms in `atoms`.
        If `distance_edges` is set, *k* is the number of distance bins.
        Otherwise, *k = 1*.
        This is not a copy, modifications on this array affect the force
        field.
    """
    def __init__(self, atoms, bonded, intra_chain, inter_chain,
                 distance_edges=None):
        if not isinstance(atoms, struc.AtomArray):
            raise TypeError(
                f"Expected 'AtomArray', not {type(atoms).__name__}"
            )
        if not np.all((atoms.atom_name == "CA") & (atoms.element == "C")):
            raise struc.BadStructureError(
                f"AtomArray does not contain exclusively CA atoms"
            )
        
        self._natoms = atoms.array_length()
        
        if distance_edges is not None:
            self._edges = np.asarray(distance_edges)
            if not np.all(np.diff(self._edges) >= 0):
                raise ValueError(
                    "Distance bin edges are not sorted in increasing order"
                )
        else:
            self._edges = None
        
        # Always create 3D matrices, even if no bins are given,
        # to generalize the code
        n_bins = 1 if self._edges is None else len(self._edges) + 1
        self._bonded = _convert_to_matrix(bonded, n_bins)
        self._intra_chain = _convert_to_matrix(intra_chain, n_bins)
        self._inter_chain = _convert_to_matrix(inter_chain, n_bins)

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
        # Distinguish between intra- and inter-chain interactions
        interactions = np.where(
            atoms.chain_id[pos_indices[0]] == atoms.chain_id[pos_indices[1]],
            intra_interactions.T,
            inter_interactions.T
        ).T
        # Initialize pos-specific interaction matrix
        # For simplicity bonded interactions are also handled as
        # non-bonded interactions at this point,
        # as they are overwritten later
        self._interaction_matrix = interactions.reshape(
            (self._natoms, self._natoms, n_bins)
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
        diag_i, diag_j = np.diag_indices(len(self._interaction_matrix))
        self._interaction_matrix[diag_i, diag_j, :] = 0

    def force_constant(self, atom_i, atom_j, sq_distance):
        if self._edges is None:
            return self._interaction_matrix[atom_i, atom_j, 0]
        else:
            bin_indices = np.searchsorted(self._edges, sq_distance)
            return self._interaction_matrix[atom_i, atom_j, bin_indices]
    
    @property
    def natoms(self):
        return self._natoms

    @property
    def interaction_matrix(self):
        return self._interaction_matrix
    

    @staticmethod
    def keskin(atoms):
        fc = _load_matrix("keskin.csv")
        return TypeSpecificForceField(atoms, fc, fc, fc)
    
    @staticmethod
    def miyazawa(atoms):
        fc = _load_matrix("miyazawa.csv")
        return TypeSpecificForceField(atoms, fc, fc, fc)
    
    @staticmethod
    def s_enm_10(atoms):
        fc = _load_matrix("s_enm_10.csv")
        return TypeSpecificForceField(atoms, 10, fc, fc)
    
    @staticmethod
    def s_enm_13(atoms):
        fc = _load_matrix("s_enm_13.csv")
        return TypeSpecificForceField(atoms, 10, fc, fc)
    
    @staticmethod
    def d_enm(atoms):
        fc = _load_matrix("d_enm.csv")
        bin_edges = _load_matrix("d_enm_edges.csv")
        return TypeSpecificForceField(atoms, 46.83, fc, fc, bin_edges)
    
    @staticmethod
    def sd_enm(atoms):
        fc = _load_matrix("sd_enm.csv").reshape(-1, 20, 20).T
        bin_edges = _load_matrix("d_enm_edges.csv")
        return TypeSpecificForceField(atoms, 43.52, fc, fc, bin_edges)
    
    @staticmethod
    def e_anm(atoms):
        intra = _load_matrix("miyazawa.csv")
        inter = _load_matrix("keskin.csv")
        # TODO: Correct bonded interaction force constant
        return TypeSpecificForceField(atoms, np.nan, intra, inter)


def _convert_to_matrix(value, n_bins):
    """
    Perform checks on input interactions matrices and return consistent
    3D matrix.
    """
    if np.isnan(value).any():
        raise IndexError(
            f"Array contains NaN elements"
        )

    if isinstance(value, numbers.Number):
        # One value for all distances and types
        return np.full(
            (N_AMINO_ACIDS, N_AMINO_ACIDS, n_bins), value, dtype=np.float32
        )
    else:
        array = np.asarray(value, dtype=np.float32)
        
        if array.ndim == 1:
            # Individual value for distances
            if len(array) != n_bins:
                raise IndexError(
                    f"Array contains {len(array)} elements "
                    f"for {n_bins} distance bins"
                )
            # Reapeat bin-wise values into both residue type dimensions
            for _ in range(2):
                array = np.repeat(
                    array[np.newaxis, ...], N_AMINO_ACIDS, axis=0
                )
            return array
        
        elif array.ndim == 2:
            # Individual value for types
            _check_matrix(array)
            return np.repeat(array[..., np.newaxis], n_bins, axis=-1)
        
        elif array.ndim == 3:
            # Individual value for distances and types
            _check_matrix(array)
            if array.shape[-1] != n_bins:
                raise IndexError(
                    f"Array contains {len(array)} elements "
                    f"for {n_bins} distance bins"
                )
            return array
        
        else:
            raise IndexError(
                f"Expected array with at most 3 dimensions, {array.ndim} given"
            )


def _check_matrix(matrix):
    """
    Check matrix on number of elements and symmetry.
    """
    if matrix.shape[:2] != (N_AMINO_ACIDS, N_AMINO_ACIDS):
        raise IndexError(
            f"Expected matrix of shape {(N_AMINO_ACIDS, N_AMINO_ACIDS)}, "
            f"got {matrix.shape[:2]}"
        )
    tranpose_axes = (1, 0, 2) if matrix.ndim == 3 else (1, 0)
    if not np.allclose(matrix, np.transpose(matrix, tranpose_axes)):
        raise ValueError(
            "Input matrix is not symmetric"
        )

matrices = {}
def _load_matrix(fname):
    if fname in matrices:
        # Matrix was already loaded
        return matrices[fname]
    
    matrix = np.loadtxt(join(DATA_DIR, fname), delimiter=",")
    matrices[fname] = matrix
    return matrix
