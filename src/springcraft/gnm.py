"""
This module contains the :class:`GNM` class for molecular dynamics
calculations using *Gaussian Network Models*.
"""

__name__ = "springcraft"
__author__ = "Patrick Kunzmann, Faisal Islam"
__all__ = ["GNM"]

import numpy as np
import biotite.structure as struc
from .interaction import compute_kirchhoff

K_B = 1 #TODO

class GNM:
    """
    This class represents a *Gaussian Network Model*.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,) or ndarray, shape=(n,3), dtype=float
        The atoms or their coordinates that are part of the model.
        It usually contains only CA atoms.
    force_field : ForceField, natoms=n
        The :class:`ForceField` that defines the force constants between
        the given `atoms`.
    masses : bool or ndarray, shape=(n,), dtype=float, optional
        If an array is given, the Kirchhoff matrix is weighted with the
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
    kirchhoff : ndarray, shape=(n,n), dtype=float
        The *Kirchhoff* matrix for this model.
        This is not a copy: Create a copy before modifying this matrix.
    covariance : ndarray, shape=(n*3,n*3), dtype=float
        The covariance matrix for this model, i.e. the inverted
        *Kirchhofff* matrix.
        This is not a copy: Create a copy before modifying this matrix.
    """

    def __init__(self, atoms, force_field, masses=None, use_cell_list=True):
        self._coord = struc.coord(atoms)
        self._ff = force_field
        self._use_cell_list = use_cell_list

        if masses is None or masses is False:
            self._masses = None
        elif masses is True:
            if not isinstance(atoms, struc.AtomArray):
                raise TypeError(
                    "An AtomArray is required to automatically infer masses"
                )
            self._masses = np.array([
                struc.info.mass(res_name, is_residue=True)
                for res_name in atoms.res_name
            ])
        else:
            if len(masses) != atoms.array_length():
                raise IndexError(
                    f"{len(masses)} masses for "
                    f"{atoms.array_length()} atoms given"
                )
            if np.any(masses == 0):
                raise ValueError("Masses must not be 0")
            self._masses = np.array(masses, dtype=float)
        
        if self._masses is not None:
            mass_weights = 1 / np.sqrt(self._masses)
            self._mass_weight_matrix = np.outer(mass_weights, mass_weights)
        else:
            self._mass_weight_matrix = None

        self._kirchhoff = None
        self._covariance = None
    
    @property
    def masses(self):
        return self._masses

    @property
    def kirchhoff(self):
        if self._kirchhoff is None:
            if self._covariance is None:
                self._kirchhoff, _ = compute_kirchhoff(
                    self._coord, self._ff, self._use_cell_list
                )
                if self._mass_weight_matrix is not None:
                    self._kirchhoff *= self._mass_weight_matrix
            else:
                self._kirchhoff = np.linalg.pinv(
                    self._covariance, hermitian=True, rcond=1e-6
                )
        return self._kirchhoff
    
    @kirchhoff.setter
    def kirchhoff(self, value):
        if value.shape != (len(self._coord), len(self._coord)):
            raise ValueError(
                f"Expected shape "
                f"{(len(self._coord), len(self._coord))}, "
                f"got {value.shape}"
            )
        self._kirchhoff = value
        # Invalidate dependent values
        self._covariance = None
    
    @property
    def covariance(self):
        if self._covariance is None:
            self._covariance = np.linalg.pinv(
                self.kirchhoff, hermitian=True, rcond=1e-6
            )
        return self._covariance
    
    @covariance.setter
    def covariance(self, value):
        if value.shape != (len(self._coord), len(self._coord)):
            raise IndexError(
                f"Expected shape "
                f"{(len(self._coord), len(self._coord))}, "
                f"got {value.shape}"
            )
        self._covariance = value
        # Invalidate dependent values
        self._kirchhoff = None
    
    def correlation_matrix(self, temperature):
        """
        Compute the correlation matrix for the atoms in the model.

        Returns
        -------
        correlation_matrix : ndarray, shape=(n,n), dtype=float
            The correlation matrix.
        """
        return K_B * temperature * self.covariance
    
    def mean_square_fluctuation(self):
        """
        Compute the *mean square fluctuation* for the atoms in the
        model.

        This is equal to the main diagonal of
        :meth:`correlation_matrix()`.

        Returns
        -------
        correlation_matrix : ndarray, shape=(n,), dtype=float
            The mean square fluctuations for each atom in the model.
        """
        return np.diag(self.correlation_matrix())
    
    def eigen(self):
        """
        Compute the eigenvalues and eigenvectors of the
        *Kirchhoff* matrix.

        Returns
        -------
        eig_values : ndarray, shape=(k,), dtype=float
            Eigenvalues of the *Kirchhoff* matrix in ascending order.
        eig_vectors : ndarray, shape=(k,n), dtype=float
            Eigenvectors of the *Kirchhoff* matrix.
            ``eig_values[i]`` corresponds to ``eigenvectors[i]``.
        """
        # 'np.eigh' can be used since the Kirchhoff matrix is symmetric 
        eig_values, eig_vectors = np.linalg.eigh(self.kirchhoff)
        return eig_values, eig_vectors.T
    
    def frequencies(self):
        """
        Compute the oscillation frequencies of the model.

        The returned units are arbitrary and should only be compared
        relative to each other.

        Returns
        -------
        frequencies : ndarray, shape=(n,), dtype=float
            Oscillation frequencies of the model in in ascending order.
            *NaN* values mark frequencies corresponding to translations
            or rotations.
        """
        eig_values, _ = self.eigen()
        eig_values[np.isclose(eig_values, 0)] = np.nan
        return np.sqrt(eig_values)