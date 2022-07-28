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


K_B = 1.380649e-23
N_A = 6.02214076e23

class ANM:
    """
    This class represents an *Anisotropic Network Model*.

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
        self._atoms = atoms
        self._coord = struc.coord(atoms)
        self._ff = force_field
        self._use_cell_list = use_cell_list

        self._masses = None
        self._hessian = None
        self._hessian_mw = None
        self._covariance = None
        self._covariance_mw = None

    @property
    def masses(self):
        if self._masses is None:
            # Assign masses to AS residues
            masslist=[]
            for a in self._atoms.res_name:
                masslist += [struc.info.mass(a, is_residue=True)] * 3
            self._masses = np.array(masslist)
        return self._masses
    
    @masses.setter
    def masses(self, value): 
        if value.shape != (len(self._coord) * 3, 1):
            if value.shape == (len(self._coord), 1):
                extender = []
                for v in value:
                    extender += [v]*3
                value = np.array(extender)
            else:
                raise IndexError(
                    f"Expected shape "
                    f"{(len(self._coord) * 3, 1)}, "
                    f"would tolerate {(len(self._coord), 1)}, "
                    f"got {value.shape}."
                )
        self._masses = value
        # Invalidate dependent values
        # TODO

    @property
    def hessian(self):
        if self._hessian is None:
            if self._covariance is None:
                self._hessian, _ = compute_hessian(
                    self._coord, self._ff, self._use_cell_list
                )
            else:
                self._hessian = np.linalg.pinv(
                    self._covariance, hermitian=True, rcond=1e-6
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
        # Invalidate dependent values
        self._covariance = None
    
    @property
    def hessian_mw(self):
        if self._hessian_mw is None:
            if self._covariance_mw is None:
                masses_3n = self.masses

                mass_matrix = []
                for m in masses_3n:
                    row = 1/(np.sqrt(m)*np.sqrt(masses_3n))
                    mass_matrix.append(row.tolist())
                mass_matrix = np.array(mass_matrix)
                # Multiply elementwise
                self._hessian_mw = mass_matrix * self.hessian
            else:
                self._hessian_mw = np.linalg.pinv(
                    self._covariance_mw, hermitian=True, rcond=1e-6
                )
        return self._hessian_mw

    @hessian_mw.setter
    def hessian_mw(self, value):
        if value.shape != (len(self._coord) * 3, len(self._coord) * 3):
            raise IndexError(
                f"Expected shape "
                f"{(len(self._coord) * 3, len(self._coord) * 3)}, "
                f"got {value.shape}"
            )
        self._hessian_mw = value
        # Invalidate dependent values
        self._covariance_mw = None
    
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
        # Invalidate dependent values
        self._hessian = None
    
    @property
    def covariance_mw(self):
        if self._covariance_mw is None:
            self._covariance_mw = np.linalg.pinv(
                self.hessian_mw, hermitian=True, rcond=1e-6
            )
        return self._covariance_mw

    @covariance_mw.setter
    def covariance_mw(self, value):
        if value.shape != (len(self._coord) * 3, len(self._coord) * 3):
            raise IndexError(
                f"Expected shape "
                f"{(len(self._coord) * 3, len(self._coord) * 3)}, "
                f"got {value.shape}"
            )
        self._covariance_mw = value
        # Invalidate dependent values
        self._hessian_mw = None
        
    def eigen(self, start_mode=6, mass_weighting=False):
        """
        Compute the eigenvalues and eigenvectors of the
        *Hessian* matrix.
        In the standard case, the first six trivial modes are omitted. 

        Parameters
        ----------
        start_mode: int, optional
            Specifies the starting index number of the returned eigenvalues
            and eigenvectors, in ascending order with regards to the eigenvalues
            starting from 0.
            Usually, the first 6 trivial modes resulting from
            rigid body movements are omitted.
        mass_weighting : bool, optional
            If True, a mass-weighted Hessian of the ANM is used to compute
            Eigenvalues. 
        
        Returns
        -------
        eig_values : ndarray, shape=(k,), dtype=float
            Eigenvalues of the *Hessian* matrix in ascending order.
        eig_vectors : ndarray, shape=(k,n), dtype=float
            Eigenvectors of the *Hessian* matrix.
            ``eig_values[i]`` corresponds to ``eigenvectors[i]``.
        """
        if mass_weighting:
            hess = self.hessian_mw
        else:
            hess = self.hessian

        # 'np.eigh' can be used since the Kirchhoff matrix is symmetric 
        eig_values, eig_vectors = np.linalg.eigh(hess)
        return eig_values[start_mode:], eig_vectors.T[start_mode:]
    
    def normal_mode(self, index, amplitude, frames, movement="sine", start_mode=6, mass_weighting=False):
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
        start_mode: int, optional
            Specifies the starting eigenvector included in the computation of
            mode vectors by index.
            Eigenvectors are enumerated starting with 0 in ascending order
            of their associated eigenvalues. 
            Usually, the first 6 trivial modes resulting from
            rigid body movements are omitted.
        mass_weighting : bool, optional
            If True, a mass-weighted Hessian of the ANM is used to compute
            Eigenvalues.
        Returns
        -------
        displacement : ndarray, shape=(m,n,3), dtype=float
            Atom displacements that depict a single oscillation.
            *m* is the number of frames.
        """
        _, eigenvectors = self.eigen(start_mode=start_mode, mass_weighting=mass_weighting)
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
        *Linear Response Theory*. [1]_

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
        
        References
        ----------
        .. [1] M Ikeguchi, J Ueno, M Sato, A Kidera,
            "Protein Structural Change Upon Ligand Binding:
            Linear Response Theory."
            Phys Rev Lett. 94, 7, 078102 (2005).

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
    
    def mean_square_fluctuation(self, T=None, mass_weighting=False):
        """
        Compute the *mean square fluctuation* for the atoms according
        to the ANM.
        This is equal to the sum of the diagonal of each 3x3 superelement of
        the covariance matrix.
        
        Parameters
        ----------
        T : int, float, None, optional
            Temperature in Kelvin to compute the temperature scaling factor.
            If T is None, the temperature scaling factor is set to 1. 
        mass_weighting : bool, optional
            If True, a mass-weighted Hessian of the ANM is used to compute
            Eigenvalues.

        Returns
        -------
        msqf : ndarray, shape=(n,), dtype=float
            The mean square fluctuations for each atom in the model.
        """
        if mass_weighting:
            cov = self.covariance_mw
        else:
            cov = self.covariance

        diag = cov.diagonal()
        reshape_diag = np.reshape(diag, (len(self._coord),-1))
        
        # Temperature scaling factor
        if T is not None:
            temp_scaling = 3*K_B*T
        else:
            temp_scaling = 1
        
        msqf = np.sum(reshape_diag, axis=1)*temp_scaling
        return msqf

    def frequencies(self, start_mode=6, mass_weighting=False):
        """
        Computes the frequency associated with each mode.

        Parameters
        ----------
        start_mode: int, optional
            Specifies the starting eigenvalue included in the computation of
            mode vectors by index.
            Eigenvalues are enumerated starting with 0 in ascending order. 
            Usually, the first 6 trivial eigenvalues resulting from
            rigid body movements are omitted.
        mass_weighting : bool, optional
            If True, a mass-weighted Hessian of the ANM is used to compute
            Eigenvalues.

        Returns
        -------
        freq : ndarray, shape=(n,), dtype=float
            The frequency in ascending order of the associated modes'
            eigenvalues.
        """
        eigenval, _ = self.eigen(start_mode=start_mode, mass_weighting=mass_weighting)
        eigenval[np.isclose(eigenval, 0)] = np.nan
        #freq = np.sqrt(eigenval)
        freq = 1/(2*np.pi)*np.sqrt(eigenval)

        return freq
    
    def bfactor(self, T=None, mass_weighting=False):
        """
        Computes the isotropic B-factors/temperature factors/Deby-Waller factors using 
        the mean-square fluctuation.

        Parameters
        ----------
        T : int, float, None, optional
            Temperature in Kelvin to compute the temperature scaling factor.
            If T is None, the temperature scaling factor is set to 1. 
        mass_weighting : bool, optional
            If True, a mass-weighted Hessian of the ANM is used to compute
            Eigenvalues.

        Returns
        -------
        bfac_values : ndarray, shape=(n,), dtype=float
            B-factors of C-alpha atoms.
        """
        msqf = self.mean_square_fluctuation(T=T, mass_weighting=mass_weighting)

        b_factors = ((8*np.pi**2)*msqf**2)/3

        return b_factors