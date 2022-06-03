"""
This module contains functionality for computing interaction matrices,
i.e. Kirchhoff and Hessian matrices.
"""

__name__ = "springcraft"
__author__ = "Patrick Kunzmann"
__all__ = ["ForceField", "PatchedForceField", "InvariantForceField",
           "HinsenForceField", "ParameterFreeForceField",
           "TabulatedForceField"]

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
    cutoff_distance : float or None
        The interaction of two atoms is only considered, if the distance
        between them is smaller or equal to this value.
        If ``None``, the interaction between all atoms is considered.
    natoms : int or None
        The number of atoms in the model.
        If a :class:`ForceField` does not depend on the respective
        atoms, i.e. `atom_i` and `atom_j` is unused in
        :meth:`force_constant()`, this attribute is ``None`` instead.
    contact_shutdown : ndarray, shape=(n,), dtype=float, optional
        Indices that point to atoms, whose contacts to all other atoms
        are artificially switched off.
    contact_pair_off : ndarray, shape=(n,2), dtype=int, optional
        Indices that point to pairs of atoms, whose contacts
        are artificially switched off.
        If ``None``, no contacts are switched off.
    contact_pair_on : ndarray, shape=(n,2), dtype=int, optional
        Indices that point to pairs of atoms, whose contacts
        are are established in any case.
        If ``None``, no contacts are artificially switched on.
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
        
        Notes
        -----
        Implementations of this method do not need
        to check whether two atoms are within the cutoff distance of the
        :class:`ForceField`:
        The given pairs of atoms are limited to pairs within cutoff
        distance of each other.
        However, if `cutoff_distance` is ``None``, the atom indices
        contain the Cartesian product of all atom indices, i.e. each
        possible combination.
        """
        pass

    @property
    def cutoff_distance(self):
        return None
    
    @property
    def contact_shutdown(self):
        return None

    @property
    def contact_pair_off(self):
        return None
    
    @property
    def contact_pair_on(self):
        return None

    @property
    def natoms(self):
        return None


class PatchedForceField(ForceField):
    
    def __init__(self, force_field,
                 contact_shutdown=None, contact_pair_off=None,
                 contact_pair_on=None, force_constants=None):
        # Support other array-like objects
        self._force_field = force_field
        self._contact_shutdown = (
            np.asarray(contact_shutdown)
            if contact_shutdown is not None else None
        )
        self._contact_pair_off = (
            np.asarray(contact_pair_off)
            if contact_pair_off is not None else None
        )
        self._contact_pair_on = (
            np.asarray(contact_pair_on)
            if contact_pair_on is not None else None
        )
        self._force_constants = (
            np.asarray(force_constants)
            if force_constants is not None else None
        )

        # Input argument checks
        _check_indices(self._contact_shutdown, force_field.natoms)
        _check_indices(self._contact_pair_off, force_field.natoms)
        _check_indices(self._contact_pair_on, force_field.natoms)
        if self._contact_pair_on is not None:
            if self._force_constants is None:
                raise TypeError(
                    "Individual force constants must be given, "
                    "if contacts are turned on"
                )
            if len(self._force_constants) != len(self._contact_pair_on):
                raise IndexError(
                    f"{len(self._force_constants)} force constants were "
                    f"given for "
                    f"{len(self._contact_pair_on)} switched on contact_pairs"
                )

    def force_constant(self, atom_i, atom_j, sq_distance):
        force_constants = self._force_field.force_constant(
            atom_i, atom_j, sq_distance
        )
        if self._contact_pair_on is not None:
            patch_atom_i, patch_atom_j = self._contact_pair_on.T
            # The minimum required size of the patch matrix is the
            # maximum of the indices + 1 
            required_size = np.max([
                np.max(patch_atom_i), np.max(patch_atom_j),
                np.max(atom_i), np.max(atom_j)
            ]) + 1
            # Fill matrix containing patched force constants
            # -1 for atom pairs with no patched force constants
            patch_matrix = np.full((required_size, required_size), -1, dtype=float)
            patch_matrix[patch_atom_i, patch_atom_j] = self._force_constants
            patch_matrix[patch_atom_j, patch_atom_i] = self._force_constants
            
            patched_force_constants = patch_matrix[atom_i, atom_j]

            # Return regular force constants for pairs where no patch exists
            return np.where(
                patched_force_constants == -1,
                force_constants,
                patched_force_constants
            )
        else:
            # No pairs are switched on -> no patching necessary
            return force_constants
    
    @property
    def cutoff_distance(self):
        return self._force_field.cutoff_distance
    
    @property
    def contact_shutdown(self):
        if self._force_field.contact_shutdown is None:
            return self._contact_shutdown
        else:
            return np.concatenate(
                [self._contact_shutdown, self._force_field.contact_shutdown]
            ) 

    @property
    def contact_pair_off(self):
        if self._force_field.contact_pair_off is None:
            return self._contact_pair_off
        else:
            return np.concatenate(
                [self._contact_pair_off, self._force_field.contact_pair_off]
            )
    
    @property
    def contact_pair_on(self):
        if self._force_field.contact_pair_on is None:
            return self._contact_pair_on
        else:
            return np.concatenate(
                [self._contact_pair_on, self._force_field.contact_pair_on]
            )

    @property
    def natoms(self):
        return self._force_field.natoms


class InvariantForceField(ForceField):
    """
    This force field treats every interaction with the same force
    constant.

    Parameters
    ----------
    cutoff_distance : float
        The interaction of two atoms is only considered, if the distance
        between them is smaller or equal to this value.
    """
    def __init__(self, cutoff_distance):
        if cutoff_distance is None:
            # A value of 'None' would give a fully connected network
            # with equal force constants for each connection,
            # which is unreasonable
            raise ValueError("Cutoff distance must be a float")
        self._cutoff_distance = cutoff_distance

    def force_constant(self, atom_i, atom_j, sq_distance):
        return np.ones(len(atom_i))
    
    @property
    def cutoff_distance(self):
        return self._cutoff_distance
    

class HinsenForceField(ForceField):
    """
    The Hinsen force field was parametrized using the *Amber94* force
    field for a local energy minimum, with crambin as template. 
    In a strict distance-dependent manner, contacts are subdivided
    into nearest-neighbour pairs along the backbone (r < 4 Å) and 
    mid-/far-range pair interactions (r >= 4 Å).
    Force constants for these interactions are computed with two
    distinct formulas. 
    2.9 Å is the lowest accepted distance between ``CA`` atoms.
    Values below that threshold are set to 2.9 Å.

    Parameters
    ----------
    cutoff_distance : float, optional
        The interaction of two atoms is only considered, if the distance
        between them is smaller or equal to this value.
        By default all interactions are included.
    
    References
    ----------
    .. [1] K Hinsen et al.,
        "Harmonicity in small proteins." 
        Chemical Physics 261(1-2): 25-37 (2000). 
    """
    def __init__(self, cutoff_distance=None):
        self._cutoff_distance = cutoff_distance

    def force_constant(self, atom_i, atom_j, sq_distance):
        distance = np.sqrt(sq_distance)
        distance = np.clip(distance, a_min=2.9, a_max=None)
        return np.where(
            distance < 4.0,
            distance * 8.6e2 - 2.39e3,
            distance**6 * 128e4
        )
    
    @property
    def cutoff_distance(self):
        return self._cutoff_distance
    

class ParameterFreeForceField(ForceField):
    """
    The "parameter free ANM" (pfENM) method is an extension of the
    original implementation of the original ANM forcefield with
    homogenous parametrization from the Bahar lab.
    Unlike in other ANMs, neither distance cutoffs nor
    distance-dependent spring constants are used.
    Instead, the residue-pair superelement of the Hessian matrix is
    weighted by the inverse of the squared distance between residue
    pairs.

    Parameters
    ----------
    cutoff_distance : float, optional
        The interaction of two atoms is only considered, if the distance
        between them is smaller or equal to this value.
        By default all interactions are included.
    
    References
    ----------
    .. [1] L Yanga, G Songa, and R L Jernigan,
        "Protein elastic network models and the ranges of cooperativity."
        PNAS.  106, 30, 12347-12352 (2009).
    """
    def __init__(self, cutoff_distance=None):
        self._cutoff_distance = cutoff_distance

    def force_constant(self, atom_i, atom_j, sq_distance):
        return 1 / sq_distance
    
    @property
    def cutoff_distance(self):
        return self._cutoff_distance


class TabulatedForceField(ForceField):
    """
    This force field uses tabulated force constants for interactions
    between atoms, based on amino acid type and distance between the
    atoms.

    The distances are separated into bins.
    A `value` is within `bin[i]`, if `value <= cutoff_distance[i]`.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,)
        The atoms in the model.
        Must contain only ``CA`` atoms and only canonic amino acids.
        ``CA`` atoms with the same chain ID and adjacent residue IDs
        are treated as bonded.
    bonded, intra_chain, inter_chain : float or ndarray, shape=(k,) or 
        shape=(20, 20) or shape=(20, 20, k), dtype=float
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
            - 2-dim array:
              Individual value for each pair of amino acid types.
              Note the alphabetical order shown above.
            - 3-dim array:
              Individual value for each distance bin and pair of amino
              acid types.

    cutoff_distance : float or None or ndarray, shape=(k), dtype=float
        If no distance dependent values are given for `bonded`,
        `intra_chain` and `inter_chain`, this parameter accepts a float,
        that represents the general cutoff distance, below which
        interactions between atoms are considered
        (or None for no cutoff distance).
        Otherwise, an array of monotonically increasing distance bin
        edges must be given.
        The edges represent the right edge of each bin.
        All interactions at distances above the last edge are not
        considered.

    Attributes
    ----------
    natoms : int or None
        The number of atoms in the model.
    interaction_matrix : ndarray, shape=(n, n, k), dtype=float
        Force constants between the atoms in `atoms`.
        If the tabulated force constants are distance dependent,
        *k* is the number of distance bins. Otherwise, *k = 1*.
        This is not a copy, modifications on this array affect the force
        field.
    """

    def __init__(self, atoms,
                 bonded, intra_chain, inter_chain, cutoff_distance):

        if not isinstance(atoms, struc.AtomArray):
            raise TypeError(
                f"Expected 'AtomArray', not {type(atoms).__name__}"
            )
        if not np.all((atoms.atom_name == "CA") & (atoms.element == "C")):
            raise struc.BadStructureError(
                f"AtomArray does not contain exclusively CA atoms"
            )
        
        self._natoms = atoms.array_length()
        

        if cutoff_distance is None:
            self._edges = None
            n_bins = 1
        elif isinstance(cutoff_distance, numbers.Real):
            self._edges = np.array([cutoff_distance])
            n_bins = 1
        else:
            self._edges = np.asarray(cutoff_distance)
            if not np.all(np.diff(self._edges) >= 0):
                raise ValueError(
                    "Distance bin edges are not sorted in increasing order"
                )
            n_bins = len(self._edges)
            
        # Always create 3D matrices, even if no bins are given,
        # to generalize the code
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
        # Convert pos-specific indices to type-specific indices -> general case
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
        if self._edges is None or len(self._edges) == 1:
            # Only a single distance bin -> No distance dependency
            return self._interaction_matrix[atom_i, atom_j, 0]
        else:
            # Distance dependence
            # Edges represent distances
            # -> square to obtain squared distances
            bin_indices = np.searchsorted(self._edges**2, sq_distance)
            try:
                return self._interaction_matrix[atom_i, atom_j, bin_indices]
            except IndexError:
                if (bin_indices >= len(self._edges)).any():
                    raise ValueError(
                        "Atom interactions above cutoff distance are not "
                        "allowed in TabulatedForceField"
                    )
                else:
                    raise
    
    @property
    def cutoff_distance(self):
        return None if self._edges is None else self._edges[-1]

    @property
    def natoms(self):
        return self._natoms

    @property
    def interaction_matrix(self):
        return self._interaction_matrix

    
    @staticmethod
    def s_enm_10(atoms):
        """
        The sENM10 forcefield by Dehouck and Mikhailov was parametrized
        by statisctical analysis of a NMR conformational ensemble dataset.
        Non-bonded interactions between amino acid species are parametrized in
        an amino acid type-specific manner, with a cutoff distance of 1 nm  
        Bonded interactions are evaluated with  10 R*T/(Ang**2), 
        corresponding to the tenfold mean of all amino acid species interactions
        at a distance 0f 3.5 nm.

        Parameters
        ----------
        atoms : AtomArray, shape=(n,)
            The atoms in the model.
            Must contain only ``CA`` atoms and only canonic amino acids.
            ``CA`` atoms with the same chain ID and adjacent residue IDs
            are treated as bonded

        Returns
        -------
        force_field : TabulatedForceField
            Force field tailored to the sENM10 parameter set.
        
        References
        ----------
        .. [1] Y Dehouck, A S Mikhailov,
           "Effective Harmonic Potentials: Insights into the Internal 
           Cooperativity and Sequence-Specificity of Protein Dynamics." 
           PLOS Computational Biology 9(8): e1003209 (2013). 
        """
        fc = _load_matrix("s_enm_10.csv")
        return TabulatedForceField(atoms, 10.0, fc, fc, 10.0)
    
    @staticmethod
    def s_enm_13(atoms):
        """
        The sENM13 forcefield by Dehouck and Mikhailov was parametrized
        by statisctical analysis of a NMR conformational ensemble dataset.
        Non-bonded interactions between amino acid species are parametrized in
        an amino acid type-specific manner, with a cutoff distance of 1.3 nm.  
        Bonded interactions are evaluated with  10 R*T/(Ang**2), 
        corresponding to the tenfold mean of all amino acid species interactions
        at a distance 0f 3.5 nm.

        Parameters
        ----------
        atoms : AtomArray, shape=(n,)
            The atoms in the model.
            Must contain only ``CA`` atoms and only canonic amino acids.
            ``CA`` atoms with the same chain ID and adjacent residue IDs
            are treated as bonded.

        Returns
        -------
        force_field : TabulatedForceField
            Force field tailored to the sENM13 parameter set.
        
        References
        ----------
        .. [1] Y Dehouck, A S Mikhailov,
           "Effective Harmonic Potentials: Insights into the Internal 
           Cooperativity and Sequence-Specificity of Protein Dynamics." 
           PLOS Computational Biology 9(8): e1003209 (2013). 
        """
        fc = _load_matrix("s_enm_13.csv")
        return TabulatedForceField(atoms, 10.0, fc, fc, 13.0)
    
    @staticmethod
    def d_enm(atoms):
        """
        The dENM forcefield by Dehouck and Mikhailov was parametrized
        by statisctical analysis of a NMR conformational ensemble dataset.
        Non-bonded amino acid interactions are solely assigned depending on the
        spatial pair distance, ignorant towards interacting amino acid species. 
        Spatial distances are divided into 27 bins.
        Bonded interactions are evaluated with  46.83 R*T/(Ang**2), 
        corresponding to the tenfold mean of all amino acid species interactions
        at a distance 0f 3.5 nm.

        Parameters
        ----------
        atoms : AtomArray, shape=(n,)
            The atoms in the model.
            Must contain only ``CA`` atoms and only canonic amino acids.
            ``CA`` atoms with the same chain ID and adjacent residue IDs
            are treated as bonded.

        Returns
        -------
        force_field : TabulatedForceField
            Force field tailored to the dENM parameter set.
        
        References
        ----------
        .. [1] Y Dehouck, A S Mikhailov,
           "Effective Harmonic Potentials: Insights into the Internal 
           Cooperativity and Sequence-Specificity of Protein Dynamics." 
           PLOS Computational Biology 9(8): e1003209 (2013). 
        """
        fc = _load_matrix("d_enm.csv")
        bin_edges = _load_matrix("d_enm_edges.csv")
        return TabulatedForceField(atoms, 46.83, fc, fc, bin_edges)
    
    @staticmethod
    def sd_enm(atoms):
        """
        The sdENM forcefield by Dehouck and Mikhailov was parametrized
        by statistical analysis of a NMR conformational ensemble
        dataset.
        Effective harmonic potentials for non-bonded interactions
        between amino acid pairs are evaluated according to interacting
        amino acid species as well as the spatial distance between them.
        Spatial distances are divided into 27 bins, with amino acid
        specific interaction tables for each distance bin.
        Bonded interactions are evaluated with  43.52 R*T/(Ang**2), 
        corresponding to the tenfold mean of all amino acid species
        interactions at a distance 0f 3.5 nm.

        Parameters
        ----------
        atoms : AtomArray, shape=(n,)
            The atoms in the model.
            Must contain only ``CA`` atoms and only canonic amino acids.
            ``CA`` atoms with the same chain ID and adjacent residue IDs
            are treated as bonded.

        Returns
        -------
        force_field : TabulatedForceField
            Force field tailored to the sdENM parameter set.
        
        References
        ----------
        .. [1] Y Dehouck, A S Mikhailov,
           "Effective Harmonic Potentials: Insights into the Internal 
           Cooperativity and Sequence-Specificity of Protein Dynamics." 
           PLOS Computational Biology 9(8): e1003209 (2013). 
        """
        fc = _load_matrix("sd_enm.csv").reshape(-1, 20, 20).T
        # TODO According to bio3d: sdENM in AU
        # -> * R * T to scale to kJ/(mol*A**2) -> verify; seems dubious.
        #fc = fc*0.0083144621*300*10
        bin_edges = _load_matrix("d_enm_edges.csv")
        return TabulatedForceField(atoms, 43.52, fc, fc, bin_edges)
    
    @staticmethod
    def e_anm(atoms, nonbonded="standard", nonbonded_mean=False):
        """
        The "extended ANM" (eANM) method discriminates between
        non-bonded interactions of amino acids within a single
        polypeptide chain (intrachain) and those present in different
        chains (interchain) in a residue-specific manner:
        the former are described by Miyazawa-Jernigan parameters, the 
        latter by Keskin parameters, which are both derived by
        mean-force statistical analysis of protein structures resolved
        by X-ray crystallography.
        Bonded interactions are evaluated with  82 R*T/(Ang**2).

        As variants, only Miyazawa-Jernigan- or Keskin- parameters are
        considered for non-bonded interactions.
        By averaging over all non-bonded residue-specific parameters,
        an eANM variant with homogenous parametrization of non-bonded
        interactions can be derived.  

        Parameters
        ----------
        atoms : AtomArray, shape=(n,)
            The atoms in the model.
            Must contain only ``CA`` atoms and only canonic amino acids.
            ``CA`` atoms with the same chain ID and adjacent residue IDs
            are treated as bonded.
        nonbonded  : String (optional)
            Keyword to specify the treatment of non-bonded interactions.
            For "standard", both Miyazawa-Jernigan parameters and Keskin
            parameters are used for the treatment of non-bonded intra-
            and intermolecular interactions, respectively.
            To use either of both parameter sets for both intra- and 
            interchain non-bonded interactions, use "miyazawa-jernigan" 
            (abbreviation: "mj") or "keskin" (abbrev.: "k")
            respectively.
        nonbonded_mean  :  Booleam  (optional)
            If True, the average of nonbonded interaction tables is
            computed and used for nonbonded interactions, which yields 
            an homogenous, amino acid-species ignorant parametrization
            of non-bonded contacts.

        Returns
        -------
        force_field : TabulatedForceField
            Force field tailored to the eANM method.
        
        References
        ----------
        .. [1] K Hamacher, J A McCammon,
           "Computing the Amino Acid Specificity of Fluctuations 
           in Biomolecular Systems."
           J Chem. Theory Comput.  2, 3, 873–878 (2006).

        .. [2] S Miyazawa, R L Jernigan,
           "Residue – Residue Potentials with a Favorable Contact Pair Term and 
           an Unfavorable High Packing Density Term, for Simulation 
           and Threading."  
           J Mol Biol., 256(3) 623-44 (1996).
        
        .. [3] O Keskin, I Bahar, R L Jernigan, A Y Badretdinov, O B Ptitsyn, 
           "Empirical solvent-mediated potentials hold for both intra-molecular 
           and inter-molecular inter-residue interactions."
           Protein Science, 7 2578-2586 (1998)
        """
        if nonbonded == "standard":
            intra_key = "miyazawa"
            inter_key = "keskin"
        elif nonbonded in ["miyazawa-jernigan", "mj"]:
            intra_key = "miyazawa"
            inter_key = "miyazawa"
        elif nonbonded in ["keskin", "k"]:
            intra_key = "keskin"
            inter_key = "keskin"
        else:
            raise ValueError(
                f"Unknown keyword '{nonbonded}' for "
                f"treatment of nonbonded interactions"
            )

        intra = _load_matrix(f"{intra_key}.csv")
        inter = _load_matrix(f"{inter_key}.csv")
        
        if nonbonded_mean:
            intra = np.average(intra) * np.ones(shape=(20, 20))
            inter = np.average(inter) * np.ones(shape=(20, 20))

        return TabulatedForceField(atoms, 82.0, intra, inter, 13.0)


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


def _check_indices(length, indices):
    if indices is None or length is None:
        return
    flat_indices = indices.flatten()
    out_of_bounds_i = np.where(flat_indices >= length)[0]
    if len(out_of_bounds_i) > 0:
        raise IndexError(
            f"Index {flat_indices[out_of_bounds_i[0]]} is out of bounds "
            f"for a structure of length {length}"
        )