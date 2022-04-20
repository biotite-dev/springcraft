"""
This module contains the :class:`GNM` class for molecular dynamics
calculations using *Gaussian Network Models*.
"""

__name__ = "springcraft"
__author__ = "Patrick Kunzmann"
__all__ = ["GNM"]

import numpy as np
import biotite.structure as struc
from .interaction import compute_kirchhoff


K_B = 1 # TODO


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
    kirchhoff : ndarray, shape=(n,n), dtype=float
        The *Kirchhoff* matrix for this model.
    """

    def __init__(self, atoms, force_field, cutoff_distance,
                 use_cell_list=True):
        self._coord = struc.coord(atoms)
        self._ff = force_field
        self._cutoff = cutoff_distance
        self._use_cell_list = use_cell_list
        self._kirchhoff = None
        self._inv_kirchhoff = None

    @property
    def kirchhoff(self):
        if self._kirchhoff is None:
            self._kirchhoff, _ = compute_kirchhoff(
                self._coord, self._ff, self._cutoff , self._use_cell_list
            )
        return self._kirchhoff
    
    @kirchhoff.setter
    def kirchhoff(self, value):
        if value.shape != (self._coord, self._coord):
            raise ValueError(
                f"Expected shape {(self._coord, self._coord)}, "
                f"got {value.shape}"
            )
        self._kirchhoff = value
        # Invalidate downstream values
        self._inv_kirchhoff = None
    
    def correlation_matrix(self, temperature):
        """
        Compute the correlation matrix for the atoms in the model.

        Returns
        -------
        correlation_matrix : ndarray, shape=(n,n), dtype=float
            The correlation matrix.
        """
        if self._inv_kirchhoff is None:
            self._inv_kirchhoff = np.linalg.pinv(self._kirchhoff)
        return K_B * temperature * self._inv_kirchhoff