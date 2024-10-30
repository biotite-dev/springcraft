"""
This module contains the :class:`ANM` class for molecular dynamics
calculations using *Anisotropic Network Models*.
"""

__name__ = "springcraft"
__author__ = "Patrick Kunzmann"
__all__ = ["ANM"]

import biotite.structure as struc
import numpy as np

from . import nma
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
        The mass for each atom, `None` if no mass weighting is applied.
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
            self._masses = np.array(
                [
                    struc.info.mass(res_name, is_residue=True)
                    for res_name in atoms.res_name
                ]
            )
        else:
            if len(masses) != atoms.array_length():
                raise IndexError(
                    f"{len(masses)} masses for " f"{atoms.array_length()} atoms given"
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
            self._covariance = np.linalg.pinv(self.hessian, hermitian=True, rcond=1e-6)
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

    def eigen(self):
        """
        Compute the Eigenvalues and Eigenvectors of the
        *Hessian* matrix.

        The first six Eigenvalues/Eigenvectors correspond to
        trivial modes (translations/rotations) and are usually omitted
        in normal mode analysis.

        Returns
        -------
        eig_values : ndarray, shape=(k,), dtype=float
            Eigenvalues of the *Hessian* matrix in ascending order.
        eig_vectors : ndarray, shape=(k,n), dtype=float
            Eigenvectors of the *Hessian* matrix.
            ``eig_values[i]`` corresponds to ``eig_vectors[i]``.
        """
        return nma.eigen(self)

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
            The index refers to the Eigenvalues obtained from
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
        return nma.normal_mode(self, index, amplitude, frames, movement)

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
        return nma.linear_response(self, force)

    def frequencies(self):
        """
        Computes the frequency associated with each mode.

        The first six modes correspond to rigid-body translations/
        rotations and are omitted in the return value.

        The returned units are arbitrary and should only be compared
        relative to each other.

        Returns
        -------
        freq : ndarray, shape=(n,), dtype=float
            The frequency in ascending order of the associated modes'
            Eigenvalues.
        """
        return nma.frequencies(self)

    def mean_square_fluctuation(self, mode_subset=None, tem=None, tem_factors=K_B):
        """
        Compute the *mean square fluctuation* for the atoms according
        to the ANM.
        This is equal to the sum of the diagonal of each 3x3
        superelement of the ANM covariance matrix, if all k-6 non-trivial
        modes are considered.

        Parameters
        ----------
        mode_subset : ndarray, dtype=int, optional
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
        return nma.mean_square_fluctuation(self, mode_subset, tem, tem_factors)

    def bfactor(self, mode_subset=None, tem=None, tem_factors=K_B):
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
        return nma.bfactor(self, mode_subset, tem, tem_factors)

    def dcc(self, mode_subset=None, norm=True, tem=None, tem_factors=K_B):
        r"""
        Computes the normalized *dynamic cross-correlation* between
        nodes of the ANM.

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

    def prs_effector_sensor(self, norm=True):
        """
        Compute the perturbation response scanning matrix following and
        the derived effector and sensor profiles after
        Atilgan et al. [1]_ and General et al. [2]_
        PRS matrices can be used to assess the relevance
        of amino acid residues/ANM nodes in transmitting allosteric
        mechanical information.

        The PRS matrix contains mechanical information of the response
        of every node in column index position j after perturbation of
        every amino acid with row index i.
        In the general case, these matrices are normalized by the diagonal
        values to compensate for the self perturbation-response of a
        given residue.

        The effector/sensor profiles are the row and column averages
        of a normalized PRS, respectively.
        These profiles allow an assessment, whether perturbations
        at a given residue position are effectively spread to
        the remaining residues (high effectivity) and how perturbations
        at other positions affect a residue (high sensitivity).

        Parameters
        ----------
        norm: bool, optional
            Normalize by the self perturbation-response of the perturbed
            ANM node.

        Returns
        -------
        prs_matrix : ndarray, shape=(n,n), dtype=float
            A 2D matrix with the perturbation response at each ENM node position.
            The row indices i correspond to the perturbed node with the same index,
            the responses of nodes j are stored at the respective columnar
            index positions.
            The whole matrix is normalized to the value of the self-perturbation
            response of node i stored in the diagonal i=j for 'norm=True'.
        effector_profile: ndarray, shape=(n), dtype=float
            Row averages of the non-diagonal row elements of the PRS.
            This profiles the effectiveness/influence of a given amino acid
            in relaying a mechanical signal to the whole structure
            after perturbation.
        sensor_profile: ndarray, shape=(n), dtype=float
            Column average of the non-diagonal row elements of the PRS.
            The resultant array is a measure for the sensitivity of
            the corresponding amino acid to perturbations in other positions.

        References
        ----------
        .. [1] C Atilgan, AR Atilgan
            "Perturbation-Response Scanning Reveals Ligand Entry-Exit
            Mechanisms of Ferric Binding Protein."
            PLoS Comput Biol 5(10) (2009).
        .. [2] IJ General, Y Liu, ME Blackburn, W Mao, LM Gierasch et al.
            "ATPase Subdomain IA Is a Mediator of Interdomain Allostery
            in Hsp70 Molecular Chaperones."
            PLOS Computational Biology 10(5) (2014).
        """
        prs_mat = nma.prs(self, norm)
        eff, sens = nma.effector_sensor(prs_mat)
        return prs_mat, eff, sens
