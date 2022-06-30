"""
This module contains the :class:`ANM` class for molecular dynamics
calculations using *Anisotropic Network Models*.
"""

__name__ = "springcraft"
__author__ = "Patrick Kunzmann"
__all__ = ["ANM"]

import numpy as np
import biotite.structure as struc
from .interaction import compute_hessian


K_B = 1 # TODO


class ANM:
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
    use_cell_list : bool, optional
        If true, a *cell list* is used to find atoms within cutoff
        distance instead of checking all pairwise atom distances.
        This significantly increases the performance for large number of
        atoms, but is slower for very small systems.
        If the `force_field` does not provide a cutoff, no cell list is
        used regardless.

    Attributes
    ----------
    hessian : ndarray, shape=(n*3,n*3), dtype=float
        The *Hessian* matrix for this model.
        Each dimension is partitioned in the form
        ``[x1, y1, z1, ... xn, yn, zn]``.
        This is not a copy: Create a copy before modifying this matrix.
    covariance : ndarray, shape=(n*3,n*3), dtype=float
        The covariance matrix for this model, i.e. the inverted
        *Hessian*.
        This is not a copy: Create a copy before modifying this matrix.
    """

    def __init__(self, atoms, force_field, use_cell_list=True):
        self._coord = struc.coord(atoms)
        self._ff = force_field
        self._use_cell_list = use_cell_list
        self._hessian = None
        self._covariance = None

    @property
    def hessian(self):
        if self._hessian is None:
            self._hessian, _ = compute_hessian(
                self._coord, self._ff, self._use_cell_list
            )
        return self._hessian
    
    @hessian.setter
    def hessian(self, value):
        if value.shape != (len(self._coord) * 3, len(self._coord) * 3):
            raise IndexError(
                f"Expected shape "
                f"{(len(self._coord) * 3, len(self._coord) * 3)}, "
                f"got {value.shape}"
            )
        self._hessian = value
        # Invalidate downstream values
        self._covariance = None
    
    @property
    def covariance(self):
        if self._covariance is None:
            self._covariance = np.linalg.pinv(
                self.hessian, hermitian=True, rcond=1e-6
            )
        return self._covariance
    
    @covariance.setter
    def covariance(self, value):
        if value.shape != (len(self._coord) * 3, len(self._coord) * 3):
            raise IndexError(
                f"Expected shape "
                f"{(len(self._coord) * 3, len(self._coord) * 3)}, "
                f"got {value.shape}"
            )
        self._covariance = value
    
    def eigen(self):
        """
        Compute the eigenvalues and eigenvectors of the
        *Hessian* matrix.

        Returns
        -------
        eig_values : ndarray, shape=(n,), dtype=float
            Eigenvalues of the *Hessian* matrix in ascending order.
        eig_vectors : ndarray, shape=(n,), dtype=float
            Eigenvectors of the *Hessian* matrix.
            ``eig_values[i]`` corresponds to ``eigenvectors[i]``.
        """
        # 'np.eigh' can be used since the Kirchhoff matrix is symmetric 
        eig_values, eig_vectors = np.linalg.eigh(self.hessian)
        return eig_values, eig_vectors.T
    
    def normal_mode(self, index, amplitude, frames, movement="sine"):
        """
        Create displacements for a trajectory depicting the given normal
        mode.

        This is especially useful for molecular animations of the chosen
        oscillation mode.

        Parameters
        ----------
        index : int
            The index of the oscillation.
            The index refers to the eigenvalues obtained from
            :meth:`eigen()`:
            Increasing indices refer to oscillations with increasing
            frequency.
            The first 6 oscillations represent oscillations and
            translations.
        amplitude : int
            The oscillation amplitude is scaled so that the maximum
            value for an atom is the given value.
        frames : int
            The number of frames (models) per oscillation.
        movement : {'sinusoidal', 'triangle'}
            Defines how to depict the oscillation.
            If set to ``'sine'`` the atom movement is sinusoidal.
            If set to ``'triangle'`` the atom movement is linear with
            *sharp* amplitude.
        
        Returns
        -------
        displacement : ndarray, shape=(m,n,3), dtype=float
            Atom displacements that depict a single oscillation.
            *m* is the number of frames.
        """
        _, eigenvectors = self.eigen()
        # Extract vectors for given mode and reshape to (n,3) array
        mode_vectors = eigenvectors[index].reshape((-1, 3))
        # Rescale, so that the largest vector has the length 'amplitude'
        vector_lenghts = np.sqrt(np.sum(mode_vectors**2, axis=-1))
        scale = amplitude / np.max(vector_lenghts)
        mode_vectors *= scale

        time = np.linspace(0, 1, frames, endpoint=False)
        if movement == "sine":
            normed_disp = np.sin(time * 2*np.pi)
        elif movement == "triangle":
            normed_disp = 2 * np.abs(2 * (time - np.floor(time + 0.5))) - 1
        else:
            raise ValueError(
                f"Movement '{movement}' is unknown"
            )
        disp = normed_disp[:, np.newaxis, np.newaxis] * mode_vectors
        return disp
    
    
    def linear_response(self, force):
        """
        Compute the atom displacement induced by the given force using
        *Linear Response Theory*.

        Parameters
        ----------
        force : ndarray, shape=(n,3) or shape=(n*3,), dtype=float
            The force that is applied to the atoms of the model.
            The first dimension gives the atom the force is applied on,
            the second dimension gives the three spatial dimensions.
            Alternatively, a flattened array in the form
            ``[x1, y1, z1, ... xn, yn, zn]`` can be given.

        Returns
        -------
        displacement : ndarray, shape=(n,3), dtype=float
            The vector of displacement induced by the given force.
            The first dimension represents the atom index,
            the second dimension represents spatial dimension.

        """
        if force.ndim == 2:
            if force.shape != (len(self._coord), 3):
                raise ValueError(
                    f"Expected force with shape {(len(self._coord), 3)}, "
                    f"got {force.shape}"
                )
            force = force.flatten()
        elif force.ndim == 1:
            if len(force) != len(self._coord) * 3:
                raise ValueError(
                    f"Expected force with length {len(self._coord) * 3}, "
                    f"got {len(force)}"
                )
        else:
            raise ValueError(
                f"Expected 1D or 2D array, got {force.ndim} dimensions"
            ) 

        return np.dot(self.covariance, force).reshape(len(self._coord), 3)
    

    def bfactor(self):
        """
        Compute the b-factors of each C-alpha atom by summing up the diagonal 
        of the *covariance* matrix.

        Returns
        -------
        bfac_values : ndarray, shape=(n,), dtype=float
            B-factors of C-alpha atoms.
        """
        diag = self.covariance.diagonal()
        reshape_diag = np.reshape(diag, (len(self.coord)/3,-1))
        diagsum = np.sum(reshape_diag, axis=2)
    
        return diagsum


# TODO Test 6 last eigenmodes for constant displacement