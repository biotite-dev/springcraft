"""
This module contains the :class:`ENM` class. An abstract base class for molecular dynamics calculations Models.
"""

__name__ = "springcraft"
__author__ = "Raphael Sutter"
__all__ = ["ENM"]

from abc import ABC, abstractmethod

import biotite.structure as struc
import biotite.structure.info as strucinfo
import numpy as np

from . import nma
from .forcefield import ForceField

K_B = 1.380649e-23
N_A = 6.02214076e23


class ENM(ABC):
    """
    Abstract base class for an *Elastic Network Model*.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,) or ndarray, shape=(n,3), dtype=float
        The atoms or their coordinates that are part of the model.
        It usually contains only CA atoms.
    force_field : ForceField, natoms=n
        The :class:`ForceField` that defines the force constants between
        the given `atoms`.
    masses : bool or ndarray, shape=(n,), dtype=float, optional
        If an array is given, the interaction matrix is weighted with the
        inverse square root of the given masses.
        If set to true, these masses are automatically inferred from the
        ``res_name`` annotation of `atoms`, instead.
        This requires `atoms` to be an :class:`AtomArray`.
        By default no mass-weighting is applied.
    use_cell_list : bool, optional
        If true, a *cell list* is used to find atoms within cutoff
        distance instead of checking all pairwise atom distances.
        This significantly increases the performance for large number of
        atoms, but is slower for very small systems.
        If the `force_field` does not provide a cutoff, no cell list is
        used regardless.

    Attributes
    ----------
    covariance : ndarray, shape=(n,n), dtype=float
        The covariance matrix for this model, i.e. the inverted
        interaction matrix. The returned covariance matrix is not scaled
        correctly and does not have the correct unit. To obtain the true
        covariance matrix, you can calculate

        .. math::

            \\text{Cov}_\\text{true} = k_B T \\text{Cov}

        with Boltzman constant :math:`k_B` and absolut temperature
        :math:`[T] = K` in Kelvin.

        This is not a copy: Create a copy before modifying this matrix.
    masses : None or ndarray, shape=(n,), dtype=float
        The mass for each atom, `None` if no mass weighting is applied.
    """

    _adjacency: np.ndarray | None
    _coord: np.ndarray
    _covariance: np.ndarray | None
    _eigen_values: np.ndarray | None
    _eigen_values_zero: int
    _eigen_vectors: np.ndarray | None
    _ff: ForceField
    _free_energy_contrib: float | None
    _masses: np.ndarray | None
    _masses_weight_matrix: np.ndarray | None
    _natoms: int
    _use_cell_list: bool

    def __init__(
        self,
        atoms: struc.AtomArray | np.ndarray,
        force_field: ForceField,
        masses=None,
        use_cell_list=True,
    ):
        self._coord = np.asarray(struc.coord(atoms))
        self._natoms = len(self._coord)
        self._ff = force_field
        self._use_cell_list = use_cell_list

        if masses is None or masses is False:
            self._masses = None
        elif masses is True:
            if not isinstance(atoms, struc.AtomArray):
                raise TypeError(
                    "An AtomArray is required to automatically infer masses"
                )
            self._masses = np.array(
                [
                    strucinfo.mass(res_name, is_residue=True)
                    for res_name in atoms.res_name  # pyright: ignore[reportOptionalIterable]
                ]
            )
        else:
            if len(masses) != self._natoms:
                raise IndexError(f"{len(masses)} masses for {self._natoms} atoms given")
            if np.any(masses == 0):
                raise ValueError("Masses must not be 0")
            self._masses = np.array(masses, dtype=float)

        if self._masses is not None:
            self._mass_weight_matrix = self._calc_mass_weight_matrix(self._masses)
        else:
            self._mass_weight_matrix = None

        self._adjacency = None
        self._covariance = None
        self._eigen_values = None
        self._eigen_values_zero = 0
        self._eigen_vectors = None
        self._free_energy_contrib = None

    @property
    def masses(self) -> np.ndarray | None:
        return self._masses

    @property
    def covariance(self) -> np.ndarray:
        if self._covariance is None:
            # same algorithm as linalg.pinv
            # but we want to store calculates eigenvalues in the process
            s, vt, k = self.eigen()
            s[k:] = 1 / s[k:]  # zero eigenvalues stay zero
            u = vt.T

            self._covariance = u @ np.multiply(s[..., np.newaxis], vt)

        return self._covariance

    @covariance.setter
    def covariance(self, value: np.ndarray):
        length = self._natoms * self.dof_per_node
        if value.shape != (length, length):
            raise IndexError(f"Expected shape {(length, length)}, got {value.shape}")
        self._covariance = value
        self._eigen_values = None
        self._eigen_values_zero = 0
        self._eigen_vectors = None
        self._free_energy_contrib = None  # unscaled

    @property
    @abstractmethod
    def dof_per_node(self) -> int:
        pass

    def modify_contacts(
        self,
        atom_i: np.typing.ArrayLike,
        atom_j: np.typing.ArrayLike,
        delta: bool | float | np.typing.ArrayLike,
        skip_checks=False,
    ):
        """
        Modifies the force constant in the `interaction` matrix between a
        pair of `atoms`. The interaction between the atoms can either be
        - turned off (the force constant is set to 0),
        - changed by an arbitrary amount `delta` or
        - turned on (reset to the value defined by the `force_field`).
        Turning on only works, if the contact is within cutoff distance.
        Otherwise nothing happens.

        If the `covariance` matrix exists, a low complexity algorithm is
        used to update the covariance matrix according to the small
        pertubation introduced to `interaction` matrix.

        The `interaction` matrix needs to be calculated beforehand.

        Parameters
        ----------
        atom_i : array_like of int, shape=(k,)
            First atom index
        atom_j : array_like of int, shape=(k,)
            Second atom index with ``atom_i[idx] != atom_j[idx]``
        delta : bool or float or array_like of bool or float, shape (1,) or (k,)
            A bool gets interpreted as a turn on/off signal.
            The amount by which the interaction strength between
            atom i and j gets changed in the `kirchhoff` matrix.
            Must not be 0.
        skip_checks : bool, optional
            Whether to skip argument checks, by default False

        Raises (if checks are enabled)
        ------
        ValueError
            If the `interaction` matrix does not exist or
            If the index arrays cannot be converted to indices or
            If index arrays do not have the same size or
            If the `delta` value(s) are neither float nor bool.
        IndexError
            If any indices are out of bounds for the initialized structure or
            If the contact of an atom with itself shall be modified.
        TypeError
            If `delta` is neither a bool nor a float.
        """
        atom_i = np.atleast_1d(np.asarray(atom_i, dtype=int))
        atom_j = np.atleast_1d(np.asarray(atom_j, dtype=int))
        delta = np.asarray(delta)  # becomes 1d later

        if not skip_checks:
            if self._interactions is None:
                raise ValueError("Kirchhoff matrix must exist.")
            if atom_i.size != atom_j.size:
                raise ValueError(
                    f"Expected atom index arrays to have the same size "
                    f"but got {atom_i.size} and {atom_j.size}."
                )
            if (
                np.any(atom_i < 0)
                or np.any(atom_i >= self._natoms)
                or np.any(atom_j < 0)
                or np.any(atom_j >= self._natoms)
            ):
                raise IndexError(
                    f"Index out of bounds for a structure of length {self._natoms}"
                )
            mask = atom_i == atom_j
            if np.any(mask):
                raise IndexError(
                    f"Expected array indices to be different "
                    f"but was {np.stack((atom_i, atom_j), axis=1)[mask, :]} "
                    f"at {np.nonzero(mask)[0]}"
                )
            if delta.size != 1 and delta.size != atom_i.size:
                raise ValueError(
                    f"There must be either 1 delta for all updates "
                    f"or as many as updates. "
                    f"Expected {atom_i.size} or 1 "
                    f"but was {delta.size}."
                )
            if (
                not np.issubdtype(delta.dtype, np.integer)
                and not np.issubdtype(delta.dtype, np.floating)
                and not np.issubdtype(delta.dtype, np.bool)
            ):
                raise TypeError(
                    f"Expected delta to be float or bool but was {delta.dtype}"
                )

        if delta.size == 1:
            delta = np.repeat(delta, atom_i.size)

        if np.issubdtype(delta.dtype, np.bool):
            sq_dist = self._adjacency[atom_i, atom_j]

            mask_on = delta & (sq_dist != 0)
            mask_off = ~delta

            force_constants = np.zeros(delta.size)
            force_constants[mask_on] = self._ff.force_constant(
                atom_i[mask_on], atom_j[mask_on], sq_dist[mask_on]
            )

            delta = np.select(
                [mask_on, mask_off],
                [
                    -(force_constants + self._interactions[atom_i, atom_j]),
                    -self._interactions[atom_i, atom_j],
                ],
                default=0,
            )

        non_zero_mask = abs(delta) > 1e-8
        self._modify_contact_pair(
            atom_i[non_zero_mask], atom_j[non_zero_mask], delta[non_zero_mask]
        )

    def modify_atom(self, atom_i: int, new_atom: bool | struc.Atom, skip_checks=False):
        """
        Modifies the force constants in the `interaction` matrix between the
        `atom_i` and all its adjacent atoms. An atom is defined as adjacent
        if it is within cutoff distance. An atom can be either be
        - turned off (interactions to all atoms are turned off),
        - turned on (interactions to all adjacent atoms are turned on) or
        - modified in a way, that the interaction strengths to its
        adjacent atoms change. This results in a recalculation of all
        interactions of `atom_i`.

        If the `covariance` matrix exists, a low complexity algorithm is
        used to update the covariance matrix according to the small
        pertubation introduced to `kirchhoff` matrix.

        The `interaction` matrix needs to be calculated beforehand.

        Parameters
        ----------
        atom_i : int
            The index of the atom to change.
        new_atom : bool or Atom
            A bool gets interpreted as a turn on/off signal.
            An Atom may result in a change to the `ForceField`.
        skip_checks : bool, optional
            Whether to skip argument checks, by default False

        Raises (if checks are enabled)
        ------
        ValueError
            If the `interaction` matrix does not exist.
        IndexError
            If any indices are out of bounds for the initialized structure.
        TypeError
            If `delta` is neither a bool nor an Atom.
        """
        if not skip_checks:
            if self._interactions is None:
                raise ValueError("Interactions matrix must exist.")
            if atom_i < 0 or atom_i >= self._natoms:
                raise IndexError(
                    f"Index out of bounds for a structure of length {self._natoms}"
                )

        if isinstance(new_atom, bool):
            delta = new_atom
        else:  # new_atom is Atom
            if not self._ff.update(atom_i, new_atom, skip_checks=True):
                return  # ForceField did not change
            delta = True

        length = self._natoms
        atom_j = np.arange(length - 1)
        atom_j[atom_i:] = np.arange(atom_i + 1, length)
        self.modify_contacts(
            np.repeat(atom_i, length - 1), atom_j, delta, skip_checks=True
        )

    def eigen(self) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Compute or fetch the Eigenvalues and Eigenvectors of the
        *interaction* matrix.

        Returns
        -------
        eig_values : ndarray, shape=(k,), dtype=float
            Eigenvalues of the matrix in ascending order.

            This is not a copy: Create a copy before modifying this matrix.
        eig_vectors : ndarray, shape=(k,n), dtype=float
            Eigenvectors of the matrix.
            ``eig_values[i]`` corresponds to ``eigenvectors[i]``.

            This is not a copy: Create a copy before modifying this matrix.
        k : int
            The number of zero Eigenvalues.
        """
        if self._eigen_values is None or self._eigen_vectors is None:
            if self._interactions is None:
                raise ValueError("Initialize interactions matrix first.")

            self._eigen_values, self._eigen_vectors = np.linalg.eigh(self._interactions)

            self._eigen_values_zero = len(self._eigen_values)
            i = 0
            while i < self._eigen_values_zero:
                if self._eigen_values[i] > 1e-10:
                    self._eigen_values_zero = i
                    break
                i = i + 1

        return self._eigen_values, self._eigen_vectors.T, self._eigen_values_zero

    def frequencies(self) -> np.ndarray:
        """
        Compute the oscillation frequencies of the model.

        The first mode corresponds to rigid-body translations/rotations
        and is omitted in the return value.
        The returned units are arbitrary and should only be compared
        relative to each other.

        Returns
        -------
        frequencies : ndarray, shape=(n,), dtype=float
            Oscillation frequencies of the model in in ascending order.
            *NaN* values mark frequencies corresponding to translations
            or rotations.
        """
        return nma.frequencies(self)

    def mean_square_fluctuation(
        self,
        mode_subset: np.ndarray | None = None,
        tem: float | None = None,
        tem_factors: float = K_B,
    ) -> np.ndarray:
        """
        Compute the *mean square fluctuation* for the atoms according to
        the GNM.
        This is equal to the sum of the diagonal of of the
        GNM covariance matrix, if all k-1 non-trivial
        modes are considered.

        Parameters
        ----------
        mode_subset : ndarray, shape=(n,), dtype=int, optional
            Specifies the subset of modes considered in the MSF
            computation.
            Only non-trivial modes can be selected.
            The first mode is counted as 0 in accordance with
            Python conventions.
            If mode_subset is None, all modes except the first
            trivial mode (0) are included.
        tem : int, float, None, optional
            Temperature in Kelvin to compute the temperature scaling
            factor by multiplying with the Boltzmann constant.
            If tem is None, no temperature scaling is conducted.
        tem_factors : int, float, optional
            Factors included in temperature weighting
            (with K_B as preset).

        Returns
        -------
        msqf : ndarray, shape=(n,), dtype=float
            The mean square fluctuations for each atom in the model.
        """
        return nma.mean_square_fluctuation(self, mode_subset, tem, tem_factors)

    def bfactor(
        self,
        mode_subset: np.ndarray | None = None,
        tem: float | None = None,
        tem_factors: float = K_B,
    ) -> np.ndarray:
        """
        Computes the isotropic B-factors/temperature factors/
        Deby-Waller factors for atoms/coarse-grained nodes using
        the mean-square fluctuation.

        These can be used to relate results obtained from ENMs
        to experimental results.

        Parameters
        ----------
        mode_subset : ndarray, shape=(n,), dtype=int, optional
            Specifies the subset of modes considered in the MSF
            computation.
            Only non-trivial modes can be selected.
            The first mode is counted as 0 in accordance with
            Python conventions.
            If mode_subset is None, all modes except the first
            trivial mode (0) are included.
        tem : int, float, None, optional
            Temperature in Kelvin to compute the temperature scaling
            factor by multiplying with the Boltzmann constant.
            If tem is None, no temperature scaling is conducted.
        tem_factors : int, float, optional
            Factors included in temperature weighting
            (with K_B as preset).
        Returns
        -------
        bfac_values : ndarray, shape=(n,), dtype=float
            B-factors of C-alpha atoms.
        """
        return nma.bfactor(self, mode_subset, tem, tem_factors)

    def dcc(
        self,
        mode_subset: np.ndarray | None = None,
        norm: bool = True,
        tem: float | None = None,
        tem_factors: float = K_B,
    ) -> np.ndarray:
        r"""
        Computes the normalized *dynamic cross-correlation* between
        nodes of the GNM.

        The DCC is a measure for the correlation in fluctuations
        exhibited by a given pair of nodes. If normalized, pairs with
        correlated fluctuations (same phase and period),
        anticorrelated fluctuations (opposite phase, same period)
        and non-correlated fluctuations are assigned (normalized)
        DCC values of 1, -1 and 0 respectively.
        For results consistent with MSFs, temperature-weighted
        absolute values can be computed (only relevant if results
        are not normalized).

        Parameters
        ----------
        mode_subset : ndarray, shape=(n,), dtype=int, optional
            Specifies the subset of modes considered in the MSF
            computation.
            Only non-trivial modes can be selected.
            The first mode is counted as 0 in accordance with
            Python conventions.
            If mode_subset is None, all modes except the first
            trivial mode (0) are included.
        norm : bool, optional
            Normalize the DCC using the MSFs of interacting nodes.
        tem : int, float, None, optional
            Temperature in Kelvin to compute the temperature scaling
            factor by multiplying with the Boltzmann constant.
            If tem is None, no temperature scaling is conducted.
        tem_factors : int, float, optional
            Factors included in temperature weighting
            (with :math:`k_B` as preset).

        Returns
        -------
        dcc : ndarray, shape=(n, n), dtype=float
            DCC values for ENM nodes.

        Notes
        -----
        The DCC for a nodepair :math:`ij` is computed as:

        .. math::

            DCC_{ij} = \frac{3 k_B T}{\gamma} \sum_k^L \left[ \frac{\vec{u}_k \cdot \vec{u}_k^T}{\lambda_k} \right]_{ij}

        with :math:`\lambda` and :math:`\vec{u}` as
        Eigenvalues and Eigenvectors corresponding to mode :math:`k` of
        the modeset :math:`L`.

        DCCs can be normalized to MSFs exhibited by two compared nodes
        following:

        .. math::

            nDCC_{ij} = \frac{DCC_{ij}}{[DCC_{ii} DCC_{jj}]^{1/2}}
        """
        return nma.dcc(self, mode_subset, norm, tem, tem_factors)

    def free_energy_contribution(self, tem=None, tem_factors=K_B):
        """
        Calculates the contribution of the protein configuration to the
        Helmholtz free energy.

        According to
        Hamacher K. Free energy of contact formation in proteins: efficient computation in the elastic network approximation. Phys Rev E Stat Nonlin Soft Matter Phys. 2011 Jul;84(1 Pt 2):016703. doi: 10.1103/PhysRevE.84.016703. Epub 2011 Jul 7. PMID: 21867339.
        """
        if self._free_energy_contrib is None:
            self._free_energy_contrib = nma.free_energy_contribution(self)

        tem_scaling = 1
        offset = 0
        if tem is not None:
            tem_scaling = tem * tem_factors
            offset = (
                self._natoms - self._eigen_values_zero * self.dof_per_node
            ) * np.log(tem_scaling)
        return tem_scaling * (self._free_energy_contrib - offset)

    @property
    @abstractmethod
    def _interactions(self) -> np.ndarray | None:
        """
        Returns the characteristic interaction matrix for this ENM e.g.
        - the *Kirchhoff* matrix in case of the GNM or
        - the *Hessian* matrix in case of the ANM.

        Primary goal is to create a private accessor for the interaction
        matrix as the characteristic name can differ but the same operations
        are performed on both matrices (like Eigenvalue calculation).

        This is not a copy: be careful when modifying the matrix or create
        a copy.

        Returns
        -------
        interactions : ndarray, dtype=float, optional
            The characteristic interactions matrix.
        """
        pass

    @staticmethod
    @abstractmethod
    def _calc_mass_weight_matrix(masses: np.ndarray) -> np.ndarray:
        pass

    def _calc_adjacency(self):
        """
        Calculates the adjacency matrix and returns the values as lists.

        The adjacency matrix is a weigthed graph that spanns the
        elastic network. Atoms are considered as being in contact
        when their distance is below the `ForceField`s threshold.
        The contacts may be overridden by using a `PatchedForceField`.

        When `_use_cell_list` is set uses an optimized strategy
        working on point lists instead off matrizes which reduces
        base operations for sparse cases. Otherwise uses a
        computationally expensive brute-force approach.

        Sets the class attribute `_adjacency` and returns the indices,
        displacements and squared distances as list for faster
        processing in sparse matrices.

        Returns
        -------
        atom_i_list: ndarray, shape=(k,), dtype=list
            list of the first atom indices
        atom_j_list: ndarray, shape=(k,), dtype=list
            list of the sceond atom indices
        disp_list: ndarray, shape=(k,3), dtype=float
            displacement between atom_i and atom_j
        sq_dist_list: ndarray, shape=(k,), dtype=float
            squared distance between atom_i and atom_j
        """
        # Convert into higher precision to avert numerical issues in pseudoinverse calculation
        coord = self._coord.astype(np.float64, copy=False)
        # Find interacting atoms within cutoff distance
        cutoff_dist = self._ff.cutoff_distance

        # contacts to shut off
        turn_off = np.zeros((self._natoms, self._natoms), dtype=np.bool)
        if self._ff.contact_shutdown is not None:
            turn_off[self._ff.contact_shutdown, :] = True
            turn_off[:, self._ff.contact_shutdown] = True
        if self._ff.contact_pair_off is not None:
            contact_off = np.sort(self._ff.contact_pair_off, axis=1).T
            turn_off[contact_off[0, :], contact_off[1, :]] = True
            turn_off[contact_off[1, :], contact_off[0, :]] = True

        if cutoff_dist is not None and self._use_cell_list:
            # use optimized strategy
            sq_dist_matrix = np.zeros((self._natoms, self._natoms))
            cell_list = struc.CellList(coord, cutoff_dist)  # pyright: ignore[reportAttributeAccessIssue]
            adj_indices = cell_list.get_atoms(coord, cutoff_dist)

            # allocate return values
            init_len = adj_indices.size
            if self._ff.contact_pair_on is not None:
                init_len += np.size(self._ff.contact_pair_on, axis=0)
            atom_i_list = np.empty(init_len, dtype=np.int32)
            atom_j_list = np.empty(init_len, dtype=np.int32)
            disp_list = np.empty((init_len, 3))
            sq_dist_list = np.empty(init_len)
            idx = 0

            # iterate over atoms
            i_iter, j_iter = np.nested_iters(adj_indices, [[0], [1]], flags=["c_index"])
            for _ in i_iter:
                atom_i = i_iter.index
                atom_i_coord = coord[atom_i]

                # iterate over contacts
                for atom_j in j_iter:
                    if atom_j < 0:  # pyright: ignore[reportOperatorIssue]
                        # handled every contact for atom
                        break
                    if atom_j >= atom_i:  # pyright: ignore[reportOperatorIssue]
                        # only calc sq_distance for lower triangle
                        continue
                    if turn_off[atom_i, atom_j]:
                        continue

                    disp = struc.displacement(atom_i_coord, coord[atom_j])
                    sq_dist = np.dot(disp, disp)
                    sq_dist_matrix[atom_i, atom_j] = sq_dist
                    sq_dist_matrix[atom_j, atom_i] = sq_dist

                    atom_i_list[idx] = atom_i
                    atom_j_list[idx] = atom_j
                    disp_list[idx, :] = disp
                    sq_dist_list[idx] = sq_dist
                    idx += 1
                    atom_i_list[idx] = atom_j
                    atom_j_list[idx] = atom_i
                    disp_list[idx, :] = disp
                    sq_dist_list[idx] = sq_dist
                    idx += 1

            # manually switched on contacts
            if self._ff.contact_pair_on is not None:
                # TODO move input validation to ForceField
                contacts = np.sort(self._ff.contact_pair_on, axis=1)
                for atom_i, atom_j in np.nditer((contacts[:, 0], contacts[:, 1])):
                    if sq_dist_matrix[atom_i, atom_j] != 0:
                        # atom already on
                        continue

                    disp = struc.displacement(coord[atom_i], coord[atom_j])
                    sq_dist = np.dot(disp, disp)
                    sq_dist_matrix[atom_i, atom_j] = sq_dist
                    sq_dist_matrix[atom_j, atom_i] = sq_dist

                    atom_i_list[idx] = atom_i
                    atom_j_list[idx] = atom_j
                    disp_list[idx, :] = disp
                    sq_dist_list[idx] = sq_dist
                    idx += 1
                    atom_i_list[idx] = atom_j
                    atom_j_list[idx] = atom_i
                    disp_list[idx, :] = disp
                    sq_dist_list[idx] = sq_dist
                    idx += 1

            # trim output arrays
            atom_i_list = np.resize(atom_i_list, idx)
            atom_j_list = np.resize(atom_j_list, idx)
            disp_list = np.resize(disp_list, (idx, 3))
            sq_dist_list = np.resize(sq_dist_list, idx)

        else:
            # brute force
            disp_matrix = struc.displacement(
                coord[np.newaxis, :, :], coord[:, np.newaxis, :]
            )
            sq_dist_matrix = np.sum(disp_matrix * disp_matrix, axis=-1)

            # map which contacts to set zero
            if cutoff_dist is not None:
                map = sq_dist_matrix > cutoff_dist**2
            else:
                map = np.zeros(sq_dist_matrix.shape, dtype=np.bool)
            map |= turn_off
            if self._ff.contact_pair_on is not None:
                contacts_on = np.sort(self._ff.contact_pair_on, axis=1).T
                map[contacts_on[0, :], contacts_on[1, :]] = False
                map[contacts_on[1, :], contacts_on[0, :]] = False
            sq_dist_matrix[map] = 0

            # retrieve lists
            atom_i_list, atom_j_list = np.where(sq_dist_matrix)
            disp_list = disp_matrix[atom_i_list, atom_j_list, :]
            sq_dist_list = sq_dist_matrix[atom_i_list, atom_j_list]

        self._adjacency = sq_dist_matrix

        return atom_i_list, atom_j_list, disp_list, sq_dist_list

    def _modify_contact_pair(
        self, atom_i: np.ndarray, atom_j: np.ndarray, deltas: np.ndarray
    ):
        """
        Modifies the interaction strengths between the atoms i and j
        in the `kirchhoff` matrix. Requires for the `kirchhoff` matrix
        to exist and for the change delta to be not null. The interaction
        strength of an atom with itself shall not be changed.

        As this is a private method, the input arguments are not
        validated. Input argument validation happens in the user facing
        functions which will always supply semantically correct
        arguments.

        If the `covariance` matrix exists, this method performs a fast
        permutation to the `covariance` matrix based on the given
        permutation to the `kirchhoff` matrix. This speeds up
        calculations as the `covariance` matrix does not need to be
        calculated by SVD again.

        TODO
        - formulas
        - speedup

        Parameters
        ----------
        atom_i : ndarray, shape=(k,), dtype=int
            First atom index
        atom_j : ndarray, shape=(k,), dtype=int
            Second atom index with ``atom_i[idx] != atom_j[idx]``
        delta : ndarray, shape=(k,), dtype=float
            The amount by which the interaction strength between
            atom i and j gets changed in the `kirchhoff` matrix.
            Must not be 0.
        """
        for i, j, delta in np.nditer([atom_i, atom_j, deltas]):
            if self._covariance is not None:
                x = self._covariance[i, :] - self._covariance[j, :]
                beta = float(1 + delta * (x[j] - x[i]))

                if np.abs(beta) < 1e-10:  # TODO use relative instead of absolute diff?
                    self._modify_contact_pair_rank_decrease(x)
                    self._free_energy_contrib = None
                elif np.abs(self._covariance[i, j]) < 1e-10:  # TODO mathematical proof
                    self._modify_contact_pair_rank_increase(i, j, delta, x, beta)
                    self._free_energy_contrib = None
                else:
                    # default case
                    self._covariance += np.outer(x * delta / beta, x)

                    if self._free_energy_contrib is not None:
                        self._free_energy_contrib += beta  # TODO *3 for GNM

            self._interactions[i, j] += delta
            self._interactions[j, i] += delta
            # self._kirchhoff[j, i] = self._kirchhoff[i, j] # TODO why does this not work
            self._interactions[i, i] -= delta
            self._interactions[j, j] -= delta

            # TODO can we make it faster, if we only compute the upper triangle?

        self._eig_values = None
        self._eig_vectors = None

    def _modify_contact_pair_rank_decrease(self, x):
        cov_mul_diff = np.matvec(self._covariance, x)
        x_norm_sq = np.inner(x, x)

        dd_mul_cov = np.outer(x / -x_norm_sq, cov_mul_diff)
        alpha = np.inner(x, cov_mul_diff) / (x_norm_sq * x_norm_sq)
        k_cov_h_mul_kh = np.outer(alpha * x, x)

        self._covariance += dd_mul_cov + dd_mul_cov.T + k_cov_h_mul_kh

    def _modify_contact_pair_rank_increase(self, i, j, delta, x, beta):
        y = -np.matvec(self._interactions, x)
        y[i] += 1
        y[j] -= 1

        y_norm_sq = np.inner(y, y)
        x_y = np.outer(x / -y_norm_sq, y)
        beta_y_y = np.outer(y * beta / (-delta * y_norm_sq * y_norm_sq), y)

        self._covariance += x_y + x_y.T + beta_y_y
