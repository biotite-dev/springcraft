"""
This module contains the :class:`ANM` class for molecular dynamics calculations using
*Anisotropic Network Models*.
"""

__name__ = "springcraft"
__author__ = "Patrick Kunzmann, Raphael Sutter"
__all__ = ["ANM"]

from typing import Literal, Union, overload

import biotite.structure as struc
import numpy as np
from typing_extensions import override

from springcraft import nma
from springcraft.enm import ENM
from springcraft.forcefield import ForceField
from springcraft.interaction import compute_hessian


class ANM(ENM):
    """
    This class represents an *Anisotropic Network Model*.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,) or ndarray, shape=(n,3), dtype=float
        The atoms or their coordinates that are part of the model.
        It usually contains only CA atoms.
    force_field : ForceField, natoms=n
        The :class:`ForceField` that defines the cutoff distance and pairwise
        interaction strengths between the given `atoms`.
    masses : bool or ndarray, shape=(n,), dtype=float, optional
        If an array is given, the `hessian` matrix is weighted with the inverse square
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
    hessian : ndarray, shape=(n,n), dtype=float
        The *hessian* matrix for this model. Adjacency matrix of `atoms` that are
        within cutoff distance of another. The weights of this adjacency matrix are the
        force constants of the abstract springs between the atoms in the ENM.
    covariance : ndarray, shape=(n,n), dtype=float
        The covariance matrix for this model, i.e. the inverted `hessian` matrix.
    has_covariance : bool
        Whether the covariance matrix is already calculated.
        If not the matrix gets calculated when the property is accessed.
    has_eigen : bool
        Whether the eigenvalues and -vectors of the `hessian` matrix are already
        calculated. If not the values get calculated when the property is accessed.
    dof : int
        Degrees of freedom considered per atom, ``dof = 1``.

    Warnings
    --------
    The `hessian` and `covariance` attributes do not return a copy. Modification to
    the matrix may result in faulty behaviour of the ENM. Consider creating a copy
    before modification.

    Notes
    -----
    The Laplacian `hessian` matrix is guaranteed to be rank-deficient. The `covariance`
    matrix :math:`\\zeta` is therefor the pseudoinverse of the `hessian` matrix
    :math:`\\Gamma` which satisfies the following conditions

    .. math::

        \\Gamma = \\Gamma \\cdot \\zeta \\cdot \\Gamma \\\\
        \\zeta = \\zeta \\cdot \\Gamma \\cdot \\zeta
    """

    _hessian: np.ndarray | None

    def __init__(
        self,
        atoms: struc.AtomArray | np.ndarray,
        force_field: ForceField,
        masses: bool | np.ndarray | None = None,
        use_cell_list: bool = True,
    ):
        super().__init__(atoms, force_field, masses, use_cell_list)

        self._hessian = None

    @property
    def hessian(self) -> np.ndarray:
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
    def hessian(self, value: np.ndarray):
        if value.shape != (self._natoms * self.dof, self._natoms * self.dof):
            raise IndexError(
                f"Expected shape "
                f"{(self._natoms * self.dof, self._natoms * self.dof)}, "
                f"got {value.shape}"
            )
        self._hessian = value

        # Invalidate dependent values
        self._covariance = None
        self._eig_values = None
        self._eig_vectors = None

    @property
    @override
    def dof(self) -> int:
        return 3

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
        self.hessian  # calc hessian if non-existant
        return super().eigen(n_zero, copy)

    def normal_mode(
        self,
        index: int,
        amplitude: int,
        frames: int,
        movement: Literal["sine", "triangle"] = "sine",
    ) -> np.ndarray:
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
            The first 6 modes represent rigid body movements
            (rotations and translations).
        amplitude : int
            The oscillation amplitude is scaled so that the maximum
            value for an atom is the given value.
        frames : int
            The number of frames (models) per oscillation.
        movement : {'sine', 'triangle'}
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

    def linear_response(self, force: np.ndarray) -> np.ndarray:
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

    def prs_effector_sensor(
        self, norm: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        norm : bool, optional
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

    @property
    @override
    def _interactions(self) -> np.ndarray | None:
        return self._hessian

    @staticmethod
    @override
    def _calc_mass_weight_matrix(masses: np.ndarray) -> np.ndarray:
        mass_weights = 1 / np.sqrt(masses)
        # 3 repetitions,
        # as the Hessian has 3 entries (x, y, z) for each atom
        mass_weights = np.repeat(mass_weights, 3)
        return np.outer(mass_weights, mass_weights)

    @override
    def _on_covariance_set(self):
        self._hessian = None
