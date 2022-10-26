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
    masses : bool or ndarray, shape=(n,), dtype=float, optional
        If an array is given, the Hessian is weighted with the inverse
        square root of the given masses.
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
    hessian : ndarray, shape=(n*3,n*3), dtype=float
        The *Hessian* matrix for this model.
        Each dimension is partitioned in the form
        ``[x1, y1, z1, ... xn, yn, zn]``.
        This is not a copy: Create a copy before modifying this matrix.
    covariance : ndarray, shape=(n*3,n*3), dtype=float
        The covariance matrix for this model, i.e. the inverted
        *Hessian*.
        This is not a copy: Create a copy before modifying this matrix.
    masses : None or ndarray, shape=(n,), dtype=float
        The mass for each atom, `None` if no mass weighting is applied
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
            # 3 repetitions,
            # as the Hessian has 3 entries (x, y, z) for each atom
            mass_weights = np.repeat(mass_weights, 3)
            self._mass_weight_matrix = np.outer(mass_weights, mass_weights)
        else:
            self._mass_weight_matrix = None

        self._hessian = None
        self._covariance = None

    @property
    def masses(self):
        return self._masses

    @property
    def hessian(self):
        if self._hessian is None:
            if self._covariance is None:
                self._hessian, _ = compute_hessian(
                    self._coord, self._ff, self._use_cell_list
                )
                if self._mass_weight_matrix is not None:
                    self._hessian *= self._mass_weight_matrix
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
        
    def eigen(self,):
        """
        Compute the eig_values and eig_vectors of the
        *Hessian* matrix.

        The first six eig_values/eig_vectors correspond to trivial modes 
        (translations/rotations) and are usually omitted 
        in normal mode analysis. 
        
        Returns
        -------
        eig_values : ndarray, shape=(k,), dtype=float
            eig_values of the *Hessian* matrix in ascending order.
        eig_vectors : ndarray, shape=(k,n), dtype=float
            eig_vectors of the *Hessian* matrix.
            ``eig_values[i]`` corresponds to ``eig_vectors[i]``.
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

        Note, that the first six modes correspond to rigid-body translations/
        rotations and are usually omitted in normal mode analysis.

        Parameters
        ----------
        index : int
            The index of the oscillation.
            The index refers to the eig_values obtained from
            :meth:`eigen()`:
            Increasing indices refer to oscillations with increasing
            frequency.
            The first 6 modes represent tigid body movements
            (rotations and translations). 
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
        _, eig_vectors = self.eigen()
        # Extract vectors for given mode and reshape to (n,3) array
        mode_vectors = eig_vectors[index].reshape((-1, 3))
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

    def frequencies(self):
        """
        Computes the frequency associated with each mode.

        The first six modes correspond to rigid-body translations/
        rotations and are usually omitted in normal mode analysis.

        The returned units are arbitrary and should only be compared
        relative to each other.

        Returns
        -------
        freq : ndarray, shape=(n,), dtype=float
            The frequency in ascending order of the associated modes'
            eig_values.
        """
        eig_values, _ = self.eigen()
        
        # The first six eig_values are usually close to 0; 
        # but can have a negative sign. 
        eig_values[0:6] = np.abs(eig_values[0:6])
        
        freq = 1/(2*np.pi)*np.sqrt(eig_values)
        return freq

    def mean_square_fluctuation(self, mode_subset=None, 
                                tem=None, tem_factors=K_B):
        """
        Compute the *mean square fluctuation* for the atoms according
        to the ANM.
        This is equal to the sum of the diagonal of each 3x3 
        superelement of the ANM covariance matrix, if all k-6 non-trivial 
        modes are considered.

        Parameters
        ----------
        mode_subset : ndarray, shape=(3n,), dtype=int, optional
            Specifies the subset of modes considered in the MSF
            computation.
            Only non-trivial modes can be selected.
            The first mode is counted as 0 in accordance with
            Python conventions.
            If mode_subset is None, all modes except the first six
            trivial modes (0-5) are included.
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
        eig_values, eig_vectors_3n = self.eigen()
        # Eigenvectors: 3N -> N
        cols_n = np.arange(0, len(eig_vectors_3n[0]), 3)
        eig_vectors_n = np.add.reduceat(np.square(eig_vectors_3n), cols_n, 
                                        axis=1)
        
        # Choose modes included in computation; raise error, if trivial 
        # modes are included
        if mode_subset is None:
            mode_subset = np.arange(6, len(eig_values))
        elif any(mode_subset <= 5):
            raise ValueError(
                "Trivial modes are included in the current selection."
                " Please check your input."
                )
        
        eig_values = eig_values[mode_subset]
        eig_vectors_n = eig_vectors_n[mode_subset]

        # Adjust shape of eig_values (N,) -> (N, 1)
        eig_values = eig_values.reshape(eig_values.shape[0], 1)
        # Eigenvecs in distinct rows; divide by associated 
        # squared Eigenvalues
        sq_div_eig_vectors = np.sum(eig_vectors_n/eig_values, axis=0)

        # Temperature weighting
        if tem is None:
            tem_scaling = 1
        else:
            tem_scaling = tem * tem_factors

        msqf = sq_div_eig_vectors * tem_scaling

        return msqf

    def bfactor(self, mode_subset=None, tem=None, 
                tem_factors=K_B):
        """
        Computes the isotropic B-factors/temperature factors/
        Deby-Waller factors for atoms/coarse-grained beads using 
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
            If mode_subset is None, all modes except the first six
            trivial modes (0-5) are included.
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
        msqf = self.mean_square_fluctuation(mode_subset, tem, 
                                            tem_factors)

        b_factors = ((8*np.pi**2)*msqf)/3

        return b_factors

    def dcc(self, mode_subset=None, norm=True, tem=None, tem_factors=K_B):
        """
        Computes the normalized *dynamic cross-correlation* between 
        nodes of the ANM.
        The DCC is a measure for the correlation in fluctuations
        exhibited by a given pair of nodes. If normalized to
        to MSFs exhibited by compared nodes, pairs with 
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
            If mode_subset is None, all modes except the first six
            trivial modes (0-5) are included.
        norm : bool, optional
            Normalize the DCC using the MSFs of interacting nodes.
        tem : int, float, None, optional
            Temperature in Kelvin to compute the temperature scaling 
            factor by multiplying with the Boltzmann constant.
            If tem is None, no temperature scaling is conducted. 
        tem_factors : int, float, optional
            Factors included in temperature weighting 
            (with K_B as preset).

        Returns
        -------
        dcc : ndarray, shape=(n, n), dtype=float
            DCC values for ENM nodes.
        """

        eig_values, eig_vectors = self.eigen()
        # Choose modes included in computation; raise error, if trivial 
        # modes are included
        if mode_subset is None:
            mode_subset = np.arange(6, len(eig_values))
        elif any(mode_subset <= 5):
            raise ValueError(
                "Trivial modes are included in the current selection."
                " Please check your input."
                )

        eig_values = eig_values[mode_subset]
        eig_vectors = eig_vectors[mode_subset]
        
        # Reshape array of eigenvectors (k,3n) -> (k,n,3)
        modes_reshaped = np.reshape(eig_vectors, 
                                    (eig_vectors.shape[0],\
                                    int(eig_vectors.shape[1]/3), 3)
                                    )
        # 3N -> N: Crate residue modes matrix
        modes_mat_3n = modes_reshaped[:,:,np.newaxis,:] *\
                        modes_reshaped[:,np.newaxis,:,:]

        modes_mat_n = np.sum(modes_mat_3n, axis=-1)
        modes_mat_n = modes_mat_n/eig_values[:,np.newaxis,np.newaxis]
        dcc = np.sum(modes_mat_n, axis=0)

        # Compute the normalized DCC
        if norm:
            dcc_ii = np.diagonal(dcc)
            dcc_ii = np.diagonal(dcc)
            dcc_ii = np.reshape(dcc_ii, (1,len(dcc_ii)))
            dcc_ii = np.repeat(dcc_ii, repeats=len(dcc_ii), axis=0)
            dcc = dcc/np.sqrt(dcc_ii*dcc_ii.T)
        # Temperature weighting
        elif tem is not None:
            dcc = dcc * tem * tem_factors
        return dcc