"""
This module contains functionality for extended NMA as set of separate
functions.
"""

__name__ = "springcraft"
__author__ = "Patrick Kunzmann, Jan Krumbach, Faisal Islam"
__all__ = [
    "eigen",
    "frequencies",
    "mean_square_fluctuation",
    "bfactor",
    "dcc",
    "normal_mode",
    "linear_response",
    "prs",
    "effector_sensor",
]

from typing import Literal

import numpy as np
from typing_extensions import deprecated

# -> Import ANM/GNM in functions to prevent circular import error

K_B = 1.380649e-23
N_A = 6.02214076e23


## NMA functions for GNMs/ANMs
@deprecated("Use class method instead.")
def eigen(enm) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Eigenvalues and Eigenvectors of the
    *Kirchhoff*/*Hessian* matrix for GNMs and ANMs respectively.

    Parameters
    ----------
    enm : ANM or GNM
        Elastic network model; an instance of either a GNM or ANM
        object.

    Returns
    -------
    eig_values : ndarray, shape=(k,), dtype=float
        Eigenvalues of the *Kirchhoff*/*Hessian* matrix
        in ascending order.
    eig_vectors : ndarray, shape=(k,n), dtype=float
        Eigenvectors of the *Kirchhoff*/*Hessian* matrix.
        ``eig_values[i]`` corresponds to ``eig_vectors[i]``.
    """
    return enm.eigen()


def frequencies(enm) -> np.ndarray:
    """
    Computes the frequency associated with each mode.

    The modes corresponding to rigid-body translations/rotations are ``NaN`` in the
    return value. The returned units are arbitrary and should only be compared relative
    to each other.

    Parameters
    ----------
    enm : ENM
        Elastic network model.

    Returns
    -------
    freq : ndarray, shape=(n,), dtype=float
        Oscillation frequencies of the model in descending order. ``NaN`` values mark
        frequencies corresponding to translations or rotations. These are guaranteed to
        be the ``k`` elements.

    Raises
    ------
    ValueError
        If the supplied `enm` is not an :class:`ENM`.

    See Also
    --------
    springcraft.enm.ENM.eigen : Eigenvalue calculation.

    Notes
    -----
    Given a set of :math:`\\lambda` of eigenvalues the frequencies :math:`f` are
    calculated as

    .. math:: f_i = \\frac{1}{2 \\pi \\sqrt{\\lambda_i}}
    """
    from springcraft.enm import ENM

    if not isinstance(enm, ENM):
        raise ValueError("Instance of ENM class expected.")

    eig_values, _ = enm.eigen(copy=False)

    # The very first / first six Eigenvalue(s) is/are usually close to 0;
    # but can have a negative sign.
    eig_values = np.abs(eig_values)

    return 1 / (2 * np.pi) * np.sqrt(eig_values)


def mean_square_fluctuation(
    enm,
    mode_subset: np.ndarray | None = None,
    tem: float | int | None = None,
    tem_factors: float | int = K_B,
) -> np.ndarray:
    """
    Compute the *mean square fluctuation* (msqf) for the atoms according to the
    :class:`ENM`.

    If all non-trivial modes are considered the msqf can be directly retrieved from the
    covariance matrix:

    * In case of the :class:`GNM` the msqf are simply the main diagonal entries.
    * In case of the :class:`ANM` the msqf are the traces of the main diagonal 3x3
      superelements.

    Parameters
    ----------
    enm : ENM
        Elastic network model.
    mode_subset : ndarray, shape=(k,), dtype=int or None, optional
        Specifies the subset of modes considered in the computation. Only non-trivial
        modes can be selected. The first mode is counted as 0 in accordance with Python
        conventions.
        If `mode_subset` is ``None``, all non-trivial modes are included.
        The default is ``None``.
    tem : float or int or None, optional
        Temperature in Kelvin to compute the temperature scaling factor by multiplying
        with `tem_factor`. If ``tem`` is ``None``, no temp scaling is conducted.
        The default is ``None``.
    tem_factors : float or int, optional
        Factors included in temperature weighting.
        The default is ``K_B`` (the *Boltzmann* constant).

    Returns
    -------
    msqf : ndarray, shape=(n,), dtype=float
        The mean square fluctuations for each atom in the model.

    Raises
    ------
    ValueError
        * If the supplied `enm` is not an :class:`ENM` or
        * if trivial modes are included in the subset.

    See Also
    --------
    springcraft.enm.ENM : Regarding the `covariance` attribute.
    springcraft.enm.ENM.eigen : Eigenvalue calculation.
    """
    from springcraft.enm import ENM

    if not isinstance(enm, ENM):
        raise ValueError("Instance of ENM class expected.")

    if enm.has_covariance and mode_subset is None:
        msqf = np.diag(enm.covariance).reshape((-1, enm.dof)).sum(axis=1)
    else:
        eig_values, eig_vectors, n_triv = enm.eigen(n_zero=True)

        # Choose modes included in computation; raise error, if trivial
        # modes are included
        if mode_subset is None:
            mode_subset = slice(n_triv, len(eig_values))  # pyright: ignore[reportAssignmentType]
        elif np.any(mode_subset < n_triv):
            raise ValueError(
                "Trivial modes are included in the current selection. "
                "Please check your input."
            )

        msqf = (eig_vectors[mode_subset].T ** 2) @ (1 / eig_values[mode_subset])
        msqf = msqf.reshape(-1, enm.dof).sum(axis=1)

    # Temperature weighting
    if tem is not None:
        msqf *= tem * tem_factors

    return msqf


def bfactor(
    enm,
    mode_subset: np.ndarray | None = None,
    tem: int | float | None = None,
    tem_factors: int | float = K_B,
) -> np.ndarray:
    """
    Computes the isotropic *B-factors/temperature factors/Debye-Waller factors* for
    atoms/coarse-grained nodes using the mean-square fluctuation. These can be used to
    relate results obtained from ENMs to experimental results.

    Parameters
    ----------
    enm : ENM
        Elastic network model.
    mode_subset : ndarray, shape=(k,), dtype=int or None, optional
        Specifies the subset of modes considered in the computation. Only non-trivial
        modes can be selected. The first mode is counted as 0 in accordance with Python
        conventions.
        If `mode_subset` is ``None``, all non-trivial modes are included.
        The default is ``None``.
    tem : float or int or None, optional
        Temperature in Kelvin to compute the temperature scaling factor by multiplying
        with `tem_factor`. If ``tem`` is ``None``, no temp scaling is conducted.
        The default is ``None``.
    tem_factors : float or int, optional
        Factors included in temperature weighting.
        The default is ``K_B`` (the *Boltzmann* constant).

    Returns
    -------
    b_factors : ndarray, shape=(n,), dtype=float
        B-factors of C-alpha atoms.

    See Also
    --------
    mean_square_fluctuation : The msqf calculation.

    Notes
    -----
    Given a set of :math:`m` of mean square fluctuations the B-factors :math:`b` are
    calculated as

    .. math:: b_i = \\frac{8 \\pi^2}{3} m_i
    """
    msqf = mean_square_fluctuation(enm, mode_subset, tem, tem_factors)
    b_factors = ((8 * np.pi**2) * msqf) / 3

    return b_factors


def dcc(
    enm,
    mode_subset: np.ndarray | None = None,
    norm: bool = True,
    tem: int | float | None = None,
    tem_factors: int | float = K_B,
) -> np.ndarray:
    r"""
    Computes the normalized *dynamic cross-correlation* between
    nodes of the GNM/ANM.

    Parameters
    ----------
    enm : ANM or GNM
        Elastic network model.
    mode_subset : ndarray, shape=(k,), dtype=int or None, optional
        Specifies the subset of modes considered in the computation. Only non-trivial
        modes can be selected. The first mode is counted as 0 in accordance with Python
        conventions.
        If `mode_subset` is ``None``, all non-trivial modes are included.
        The default is ``None``.
    norm : bool, optional
        Normalize the DCC using the msqf of interacting nodes.
        The default is ``True``.
    tem : float or int or None, optional
        Temperature in Kelvin to compute the temperature scaling factor by multiplying
        with `tem_factor`. If ``tem`` is ``None``, no temp scaling is conducted.
        The default is ``None``.
    tem_factors : float or int, optional
        Factors included in temperature weighting.
        The default is ``K_B`` (the *Boltzmann* constant).

    Returns
    -------
    dcc : ndarray, shape=(n, n), dtype=float
        DCC values for ENM nodes as NxN matrix.

    Notes
    -----

    The DCC for a nodepair :math:`ij` is computed as:

    .. math::

        DCC_{ij} = \frac{3 k_B T}{\gamma} \sum_k^L \left[
                     \frac{\vec{u}_k \cdot \vec{u}_k^T}{\lambda_k}
                   \right]_{ij}

    with :math:`\lambda` and :math:`\vec{u}` as
    Eigenvalues and Eigenvectors corresponding to mode :math:`k` of
    the modeset :math:`L`.

    DCCs can be normalized to MSFs exhibited by two compared nodes
    following:

    .. math::

        nDCC_{ij} = \frac{DCC_{ij}}{[DCC_{ii} DCC_{jj}]^{1/2}}

    When all modes are considered, the DCC is equal to the covariance matrix
    of GNMs or to the trace of all supermatrices (3x3) of the
    covariance matrix (3Nx3N) in the case of ANMs.
    Consequently, these are returned if standard parameters
    for ``mode_subset`` and ``memory_efficient`` are passed to the function.
    """
    from springcraft.enm import ENM

    if not isinstance(enm, ENM):
        raise ValueError("Instance of ENM class expected.")

    if mode_subset is None:
        dcc = (
            enm.covariance.reshape(enm._natoms, enm.dof, enm._natoms, enm.dof)
            .swapaxes(1, 2)
            .trace(axis1=2, axis2=3)
        )
    else:
        eig_values, eig_vectors, n_triv = enm.eigen(n_zero=True)

        # raise error, if trivialmodes are included
        if np.any(mode_subset < n_triv):
            raise ValueError(
                "Trivial modes are included in the current selection. "
                "Please check your input."
            )

        eig_values = eig_values[mode_subset]
        eig_vectors = eig_vectors[mode_subset]

        # Reshape array of eigenvectors
        # (k,3n) -> (k,n,3) for ANMs; (k,n) -> (k,n,1) for GNMs
        eig_vectors = np.reshape(eig_vectors, (len(mode_subset), -1, enm.dof))
        eig_vectors_scal = eig_vectors / eig_values[:, None, None]
        dcc = np.einsum("knd,kmd->nm", eig_vectors, eig_vectors_scal)

    # Compute the normalized DCC
    if norm:
        dcc_ii = np.sqrt(np.diagonal(dcc))
        dcc /= np.outer(dcc_ii, dcc_ii)

    # Temperature weighting
    if tem is not None:
        dcc = dcc * tem * tem_factors

    return dcc


## ANM specific functions
def normal_mode(
    anm,
    index: int,
    amplitude: int,
    frames: int,
    movement: Literal["sine", "triangle"] = "sine",
) -> np.ndarray:
    """
    Create displacements for a trajectory depicting the given normal
    mode for ANMs.

    Parameters
    ----------
    anm : ANM
        Instance of ANM object.
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
    from springcraft.anm import ANM

    if not isinstance(anm, ANM):
        raise ValueError("Instance of ANM class expected.")
    else:
        _, eig_vectors = anm.eigen()
        # Extract vectors for given mode and reshape to (n,3) array
        mode_vectors = eig_vectors[index].reshape((-1, 3))
        # Rescale, so that the largest vector has the length 'amplitude'
        vector_lenghts = np.sqrt(np.sum(mode_vectors**2, axis=-1))
        scale = amplitude / np.max(vector_lenghts)
        mode_vectors *= scale

        time = np.linspace(0, 1, frames, endpoint=False)
        if movement == "sine":
            normed_disp = np.sin(time * 2 * np.pi)
        elif movement == "triangle":
            normed_disp = 2 * np.abs(2 * (time - np.floor(time + 0.5))) - 1
        else:
            raise ValueError(f"Movement '{movement}' is unknown")
        disp = normed_disp[:, np.newaxis, np.newaxis] * mode_vectors

        return disp


def linear_response(anm, force: np.ndarray) -> np.ndarray:
    """
    Compute the atom displacement induced by the given force using
    *Linear Response Theory*. [1]_

    Parameters
    ----------
    anm : ANM
        Instance of ANM object.
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
    from springcraft.anm import ANM

    if not isinstance(anm, ANM):
        raise ValueError("Instance of ANM class expected.")
    else:
        if force.ndim == 2:
            if force.shape != (len(anm._coord), 3):
                raise ValueError(
                    f"Expected force with shape {(len(anm._coord), 3)}, "
                    f"got {force.shape}"
                )
            force = force.flatten()
        elif force.ndim == 1:
            if len(force) != len(anm._coord) * 3:
                raise ValueError(
                    f"Expected force with length {len(anm._coord) * 3}, "
                    f"got {len(force)}"
                )
        else:
            raise ValueError(f"Expected 1D or 2D array, got {force.ndim} dimensions")

        return np.dot(anm.covariance, force).reshape(len(anm._coord), 3)


def prs(anm, norm: bool = True) -> np.ndarray:
    """
    Compute the perturbation response scanning matrix following
    Atilgan et al. [1]_

    Parameters
    ----------
    anm : ANM
        Instance of ANM object.
    norm : bool, optional
        Normalize by the self perturbation-response of the perturbed ANM node.
        The default is ``True``.

    Returns
    -------
    prs_matrix : ndarray, shape=(n,n), dtype=float
        A 2D matrix with the perturbation response at each ENM node position.
        The row indices i correspond to the perturbed node with the same index,
        the responses of nodes j are stored at the respective columnar
        index positions.
        The whole matrix is normalized to the value of the self-perturbation
        response of node i stored in the diagonal i=j for 'norm=True'.

    References
    ----------
    .. [1] C Atilgan, AR Atilgan
        "Perturbation-Response Scanning Reveals Ligand Entry-Exit
        Mechanisms of Ferric Binding Protein."
        PLoS Comput Biol 5(10) (2009).
    """
    from springcraft.anm import ANM

    if not isinstance(anm, ANM):
        raise ValueError("Instance of ANM class expected.")

    cov = anm.covariance
    dim_3n = cov.shape[0]
    dim_n = anm._coord.shape[0]

    # 3Nx3N -> Nx3N -> NxN
    reduce_at_inds = np.arange(0, dim_3n, 3)
    sq_cov_summedrow = np.add.reduceat(cov**2, reduce_at_inds, axis=0)
    prs_matrix = np.add.reduceat(sq_cov_summedrow, reduce_at_inds, axis=1)

    if norm:
        prs_matrix_ii = np.diagonal(prs_matrix)
        prs_matrix_ii = np.repeat(np.reshape(prs_matrix_ii, (dim_n, 1)), dim_n, axis=1)
        prs_matrix = prs_matrix / prs_matrix_ii
    return prs_matrix


def effector_sensor(prs_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute effector/sensor residues according to the PRS-Matrix
    as described in General et al. [1]_
    Note, that the PRS matrix should be normalized (standard case).

    Parameters
    ----------
    prs_matrix : ndarray, shape=(n,n), dtype=float
        A 2D matrix with the perturbation response at each ENM node position.
        The row indices i correspond to the perturbed node with the same index,
        the responses of node j are stored at the respective columnar
        index positions.
        The whole matrix is normalized to the value of the self-perturbation
        response of node i stored in the diagonal i=j.

    Returns
    -------
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
    .. [1] IJ General, Y Liu, ME Blackburn, W Mao, LM Gierasch et al.
        "ATPase Subdomain IA Is a Mediator of Interdomain Allostery
        in Hsp70 Molecular Chaperones."
        PLOS Computational Biology 10(5) (2014).
    """
    # Weights for averaging -> 0 for off-diagonal elements
    prs_row_num = len(prs_matrix)
    av_weights = 1 - np.eye(prs_row_num)

    # Average over rows/columns -> eff/sens
    effector_profile = np.average(prs_matrix, weights=av_weights, axis=1)
    sensor_profile = np.average(prs_matrix, weights=av_weights, axis=0)
    return effector_profile, sensor_profile
