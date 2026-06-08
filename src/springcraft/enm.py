"""
This module contains the :class:`ENM` class. An abstract base class for Elastic Network
Models.
"""

__name__ = "springcraft"
__author__ = "Raphael Sutter"
__all__ = ["ENM"]

from abc import ABC, abstractmethod
from typing import Literal, Union, overload

import biotite.structure as struc
import biotite.structure.info as strucinfo
import numpy as np

from springcraft import nma
from springcraft.forcefield import ForceField

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
        The :class:`ForceField` that defines the cutoff distance and pairwise
        interaction strengths between the given `atoms`.
    masses : bool or ndarray, shape=(n,), dtype=float, optional
        If an array is given, the interaction matrix is weighted with the inverse square
        root of the given masses.
        If set to true, these masses are automatically inferred from the `res_name`
        annotation of `atoms`, instead. This requires `atoms` to be an
        :class:`AtomArray`.
        By default no mass-weighting is applied.
    use_cell_list : bool, optional
        If true, a *cell list* is used to find atoms within cutoff distance instead of
        checking all pairwise atom distances. This significantly increases the
        performance for large number of atoms, but is slower for very small systems.
        If the `force_field` does not provide a cutoff, no cell list is used regardless.

    Attributes
    ----------
    masses : None or ndarray, shape=(n,), dtype=float
        The mass for each atom, `None` if no mass weighting is applied.
    covariance : ndarray, shape=(n,n), dtype=float
        The covariance matrix for this model, i.e. the inverted interaction matrix.
    has_covariance : bool
        Whether the covariance matrix is already calculated.
        If not the matrix gets calculated when the property is accessed.
    has_eigen : bool
        Whether the eigenvalues and -vectors of the interaction matrix are already
        calculated. If not the values get calculated when the property is accessed.
    dof : int
        Degrees of freedom considered per atom.

    Warnings
    --------
    The `covariance` attribute does not return a copy. Modification to the matrix may
    result in faulty behaviour of the ENM. Consider creating a copy before modification.
    """

    # euclidean coordinates of each atom
    _coord: np.ndarray
    # pseudo-inverse of the _interaction matrix
    _covariance: np.ndarray | None
    # eigenvalues/-vectors of the _interaction matrix
    _eig_values: np.ndarray | None
    _eig_vectors: np.ndarray | None
    # ForceField defining the atom interactions
    _ff: ForceField
    # atom masses
    _masses: np.ndarray | None
    # the mass weight matrix is used to weigh the _interactions matrix
    _masses_weight_matrix: np.ndarray | None
    # the number of atoms
    _natoms: int
    # whether to use an optimized algorithm to calculate the _interaction matrix
    _use_cell_list: bool

    def __init__(
        self,
        atoms: struc.AtomArray | np.ndarray,
        force_field: ForceField,
        masses=None,
        use_cell_list=True,
    ):
        self._coord = np.asarray(struc.coord(atoms)).astype(np.float64, copy=False)
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

        self._covariance = None
        self._eig_values = None
        self._eig_vectors = None

    @property
    def masses(self) -> np.ndarray | None:
        return self._masses

    @property
    def covariance(self) -> np.ndarray:
        if self._covariance is None:
            # same algorithm as linalg.pinv
            # but we want to store calculates eigenvalues in the process
            s, u, n_zero = self.eigen(n_zero=True, copy=False)

            si = np.zeros_like(s)
            si[n_zero:] = 1 / s[n_zero:]

            self._covariance = u.T @ np.multiply(si[..., np.newaxis], u)

        return self._covariance

    @covariance.setter
    def covariance(self, value: np.ndarray):
        length = self._natoms * self.dof
        if value.shape != (length, length):
            raise IndexError(f"Expected shape {(length, length)}, got {value.shape}")
        self._covariance = value

        # invalidate dependant values
        self._eig_values = None
        self._eig_vectors = None

        self._on_covariance_set()

    @property
    def has_covariance(self) -> bool:
        return self._covariance is not None

    @property
    @abstractmethod
    def dof(self) -> int:
        pass  # pragma: no cover

    @overload
    def eigen(
        self, n_zero: Literal[False] = False, copy: bool = True
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @overload
    def eigen(
        self, n_zero: Literal[True], copy: bool = True
    ) -> tuple[np.ndarray, np.ndarray, int]: ...

    def eigen(
        self, n_zero=False, copy=True
    ) -> Union[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, int]]:
        """
        Compute or fetch the eigenvalues and eigenvectors of the interaction matrix.

        The Laplacian interaction matrix is guaranteed to be rank-deficient. That means
        some eigenvalues are guaranteed to be zero. These are guaranteed to be the first
        ``k`` eigenvalues returned. The remaining eigenvalues are sorted in strictly
        ascending order. The eigenvectors have the same order as there corresponding
        eigenvalues.

        Numerical inconsistencies can occur during eigenvalue calculation resulting in
        negative eigenvalues. For ease of calculation these negative eigenvalues are
        swapped with the zero eigenvalues resulting in the order described above. This
        has the effect that updates to the eigenvalues can not be calculated.

        Parameters
        ----------
        n_zero : bool, optional
            Whether to return the number of (the first) zero eigenvalues.
            The default is ``False``.
        copy : bool, optional
            Whether to return the eigenvalues and eigenvectors as copies. If you choose
            not to return copies a modification to these values can result in incorrect
            behaviour of the class.
            The default is ``True``.

        Returns
        -------
        eig_values : ndarray, shape=(k,), dtype=float
            Eigenvalues of the matrix in ascending order.
        eig_vectors : ndarray, shape=(k, n), dtype=float
            Eigenvectors of the matrix, one per row. ``eig_values[i]`` corresponds to
            ``eig_vectors[i]``.
        eigen_n_zero : int, optional
            The number of the (first) zero eigenvalues. Only returned for `n_zero`.

        Warns
        -----
        RuntimeWarning
            When numerical inconsistencies result in negative eigenvalues.
        """
        if self._eig_values is None or self._eig_vectors is None:
            assert self._interactions is not None  # should never happen

            self._eig_values, self._eig_vectors = np.linalg.eigh(self._interactions)

            threshold = self._eig_values[-1] * 1e-6  # max(eig_values) * 10^-6
            i = 0
            while self._eig_values[i] < -threshold:
                i = i + 1
            n_neg = i
            while self._eig_values[i] <= threshold:
                i = i + 1
            n_triv = i + n_neg

            if n_neg:
                # numerical error with some eigenvalues below 0
                v, V, n, m = self._eig_values, self._eig_vectors, n_neg, n_triv
                v[:m], v[m : m + n] = v[n : n + m].copy(), v[:n].copy()
                V[:, :m], V[:, m : m + n] = V[:, n : n + m].copy(), V[:, :n].copy()

                raise RuntimeWarning(
                    "Numerical error during EigenValue calculation. "
                    "Some analysis might fail."
                )

            self._eigen_n_zero = n_triv

        val = self._eig_values
        vec = self._eig_vectors.T
        if copy:
            val = val.copy()
            vec = vec.copy()

        if n_zero:
            return val, vec, self._eigen_n_zero

        return val, vec

    def frequencies(self) -> np.ndarray:
        """
        Compute the oscillation frequencies of the model.

        Returns
        -------
        frequencies : ndarray, shape=(n,), dtype=float
            Oscillation frequencies of the model.

        See Also
        --------
        springcraft.nma.frequencies : The frequency calculation.
        """
        return nma.frequencies(self)

    def mean_square_fluctuation(
        self,
        mode_subset: np.ndarray | None = None,
        tem: float | int | None = None,
        tem_factors: float | int = K_B,
    ) -> np.ndarray:
        """
        Compute the *mean square fluctuation* for the atoms of the model.

        Parameters
        ----------
        mode_subset : ndarray, shape=(k,), dtype=int or None, optional
            Specifies the subset of modes considered in the computation.
            The default is ``None``.
        tem : float or int or None, optional
            Temperature in Kelvin. If ``tem`` is ``None``, no temp scaling is conducted.
            The default is ``None``.
        tem_factors : float or int, optional
            Factors included in temperature weighting.
            The default is ``K_B``.

        Returns
        -------
        msqf : ndarray, shape=(n,), dtype=float
            The mean square fluctuations for each atom in the model.

        See Also
        --------
        springcraft.nma.mean_square_fluctuation : Mean square fluctuation calculation.
        """
        return nma.mean_square_fluctuation(self, mode_subset, tem, tem_factors)

    def bfactor(
        self,
        mode_subset: np.ndarray | None = None,
        tem: float | int | None = None,
        tem_factors: float | int = K_B,
    ) -> np.ndarray:
        """
        Compute the *isotropic B-factors/temperature factors/Debye-Waller factors* for
        the atoms of the model.

        Parameters
        ----------
        mode_subset : ndarray, shape=(k,), dtype=int or None, optional
            Specifies the subset of modes considered in the computation.
            The default is ``None``.
        tem : float or int or None, optional
            Temperature in Kelvin. If ``tem`` is ``None``, no temp scaling is conducted.
            The default is ``None``.
        tem_factors : float or int, optional
            Factors included in temperature weighting.
            The default is ``K_B``.

        Returns
        -------
        b_factors : ndarray, shape=(n,), dtype=float
            B-factors of C-alpha atoms.

        See Also
        --------
        springcraft.nma.bfactor : The B-factor calculation.
        """
        return nma.bfactor(self, mode_subset, tem, tem_factors)

    def dcc(
        self,
        mode_subset: np.ndarray | None = None,
        norm: bool = True,
        tem: float | int | None = None,
        tem_factors: float | int = K_B,
    ) -> np.ndarray:
        """
        Compute the *dynamic cross-correlation* between nodes of the model.

        Parameters
        ----------
        mode_subset : ndarray, shape=(k,), dtype=int or None, optional
            Specifies the subset of modes considered in the computation.
            The default is ``None``.
        norm : bool, optional
            Normalize the DCC using the msqf of interacting nodes.
            The default is ``True``.
        tem : float or int or None, optional
            Temperature in Kelvin. If ``tem`` is ``None``, no temp scaling is conducted.
            The default is ``None``.
        tem_factors : float or int, optional
            Factors included in temperature weighting.
            The default is ``K_B``.

        Returns
        -------
        dcc : ndarray, shape=(n, n), dtype=float
            DCC values for the model nodes.

        See Also
        --------
        springcraft.nma.dcc : The DCC calculation.
        """
        return nma.dcc(self, mode_subset, norm, tem, tem_factors)

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
        interactions : ndarray, dtype=float or None
            The characteristic interactions matrix.
        """
        pass  # pragma: no cover

    @staticmethod
    @abstractmethod
    def _calc_mass_weight_matrix(masses: np.ndarray) -> np.ndarray:
        pass  # pragma: no cover

    @abstractmethod
    def _on_covariance_set(self):
        pass  # pragma: no cover
