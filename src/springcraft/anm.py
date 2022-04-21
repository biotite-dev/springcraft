"""
This module contains the :class:`GNM` class for molecular dynamics
calculations using *Gaussian Network Models*.
"""

__name__ = "springcraft"
__author__ = "Patrick Kunzmann"
__all__ = ["ANM"]

import numpy as np
import biotite.structure as struc
from .interaction import compute_hessian


K_B = 1 # TODO


class GNM:
    """
    This class represents a *Anisotropic Network Model*.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,) or ndarray, shape=(n,3), dtype=float
        The atoms or their coordinates that are part of the model.
        It usually contains only CA atoms.
    force_field : ForceField, natoms=n
        The :class:`ForceField` that defines the force constants between
        the given `atoms`.
    cutoff_distance : float
        The interaction of two atoms is only considered, if the distance
        between them is smaller or equal to this value.
    use_cell_list : bool, optional
        If true, a *cell list* is used to find atoms within cutoff
        distance instead of a brute-force approach.
        This significantly increases the performance for large number of
        atoms, but is slower for very small systems.

    Attributes
    ----------
    hessian : ndarray, shape=(n,n), dtype=float
        The *Hessian* matrix for this model.
    covariance : ndarray, shape=(n,n), dtype=float
        The covariance matrix for this model, i.e. the inverted *Hessian*.
    """

    def __init__(self, atoms, force_field, cutoff_distance,
                 use_cell_list=True):
        self._coord = struc.coord(atoms)
        self._ff = force_field
        self._cutoff = cutoff_distance
        self._use_cell_list = use_cell_list
        self._hessian = None
        self._covariance = None

    @property
    def hessian(self):
        if self._hessian is None:
            self._hessian, _ = compute_hessian(
                self._coord, self._ff, self._cutoff , self._use_cell_list
            )
        return self._hessian
    
    @hessian.setter
    def hessian(self, value):
        if value.shape != (self._coord, self._coord, 3, 3):
            raise ValueError(
                f"Expected shape {(self._coord, self._coord, 3, 3)}, "
                f"got {value.shape}"
            )
        self._hessian = value
        # Invalidate downstream values
        self._covariance = None
    
    @property
    def covariance(self, temperature):
        if self._covariance is None:
            # TODO: Add covariance calculation
            # self._covariance = ...
            raise NotImplementedError()
        return self._hessian
    
    @covariance.setter
    def covariance(self, value):
        if value.shape != (self._coord, self._coord, 3, 3):
            raise ValueError(
                f"Expected shape {(self._coord, self._coord, 3, 3)}, "
                f"got {value.shape}"
            )
        self._hessian = value
        # Invalidate downstream values
        self._covariance = None
    
    def eigen(self):
        """
        Compute the eigenvalues and eigenvectors of the
        *Kirchhoff* matrix.

        Returns
        -------
        eig_values : ndarray, shape=(n,), dtype=float
            Eigenvalues of the *Kirchhoff* matrix in ascending order.
        eig_vectors : ndarray, shape=(n,), dtype=float
            Eigenvectors of the Kirchhoff matrix.
            ``eig_values[i]`` corresponds to ``eigenvectors[i]``.
        """
        # 'np.eigh' can be used since the Kirchhoff matrix is symmetric 
        eig_values, eig_vectors = np.linalg.eigh(self.kirchhoff)
        return eig_values, eig_vectors.T