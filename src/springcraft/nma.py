"""
This module contains functionality for extended NMA as set of separate
functions.
"""

__name__ = "springcraft"
__author__ = "Patrick Kunzmann, Jan Krumbach, Faisal Islam"
__all__ = ["eigen", "frequencies", 
           "mean_square_fluctuation", "bfactor", "dcc",
           "normal_mode", "linear_response"]

import numpy as np
# -> Import ANM/GNM in functions to prevent circular import error

K_B = 1.380649e-23
N_A = 6.02214076e23

## NMA functions for GNMs/ANMs
def eigen(enm):
    """
    Compute the Eigenvalues and Eigenvectors of the
    *Kirchhoff*/*Hessian* matrix for GNMs and ANMs respectively.

    Parameters
    ----------
    enm : ANM or GNM
        Elastic network model; an instance of either an GNM or ANM
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
    from .gnm import GNM
    from .anm import ANM
    
    # Assign Kirchhoff/Hessian
    if isinstance(enm, GNM):
        mech_matrix = enm.kirchhoff
    elif isinstance(enm, ANM):
        mech_matrix = enm.hessian
    else:
        raise ValueError(
            "Instance of GNM/ANM class expected."
            )

    # 'np.eigh' can be used since the Hessian/Kirchhoff matrix is symmetric 
    eig_values, eig_vectors = np.linalg.eigh(mech_matrix)
    
    return eig_values, eig_vectors.T

def frequencies(enm):
    """
    Computes the frequency associated with each mode.

    The modes corresponding to rigid-body translations/rotations are
    omitted in the return value.
    The returned units are arbitrary and should only be compared
    relative to each other.
    
    Parameters
    ----------
    enm : ANM or GNM
        Elastic network model; an instance of either an GNM or ANM
        object.

    Returns
    -------
    freq : ndarray, shape=(n,), dtype=float
        The frequency in ascending order of the associated modes'
        Eigenvalues.
    """
    from .gnm import GNM
    from .anm import ANM
    
    if isinstance(enm, GNM):
        ntriv_modes = 1
    elif isinstance(enm, ANM):
        ntriv_modes = 6
    else:
        raise ValueError(
            "Instance of GNM/ANM class expected."
            )

    eig_values, _ = eigen(enm)
    
    # The very first / first six Eigenvalue(s) is/are usually close to 0; 
    # but can have a negative sign. 
    eig_values[0:ntriv_modes] = np.abs(eig_values[0:ntriv_modes])
    
    freq = 1/(2*np.pi)*np.sqrt(eig_values)
    
    return freq

def mean_square_fluctuation(enm, mode_subset=None, 
                            tem=None, tem_factors=K_B):
    """
    Compute the *mean square fluctuation* for the atoms according
    to the ANM/GNM.

    Parameters
    ----------
    enm : ANM or GNM
        Elastic network model; an instance of either an GNM or ANM
        object.
    mode_subset : ndarray, shape=(n,) or (3n,), dtype=int, optional
        Specifies the subset of modes considered in the MSF
        computation.
        Only non-trivial modes can be selected.
        The first mode is counted as 0 in accordance with
        Python conventions.
        If mode_subset is None, all modes except the first/first six
        trivial mode(s) (0, 0-5 respectively) are included.
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
    from .gnm import GNM
    from .anm import ANM

    if not isinstance(enm, (GNM, ANM)):
        raise ValueError(
            "Instance of GNM/ANM class expected."
        )
    
    eig_values, eig_vectors = eigen(enm)

    if isinstance(enm, ANM):
        # Eigenvectors: 3N -> N
        cols_n = np.arange(0, len(eig_vectors[0]), 3)
        eig_vectors = np.add.reduceat(
            np.square(eig_vectors), cols_n, axis=1
        )
        ntriv_modes = 6
    # -> GNMs
    else:
        eig_vectors = np.square(eig_vectors)
        ntriv_modes = 1
    
    # Choose modes included in computation; raise error, if trivial 
    # modes are included
    if mode_subset is None:
        mode_subset = np.arange(ntriv_modes, len(eig_values))
    elif any(mode_subset <= (ntriv_modes-1)):
        raise ValueError(
            "Trivial modes are included in the current selection."
            " Please check your input."
            )
    
    eig_values = eig_values[mode_subset]
    eig_vectors = eig_vectors[mode_subset]
    
    # Adjust shape of eig_values (N,) -> (N, 1)
    eig_values = eig_values.reshape(eig_values.shape[0], 1)
    # Eigenvecs in distinct rows; divide by associated 
    # squared Eigenvalues
    sq_div_eig_vectors = np.sum(eig_vectors/eig_values, axis=0)
    
    # Temperature weighting
    if tem is None:
        tem_scaling = 1
    else:
        tem_scaling = tem * tem_factors
    
    msqf = sq_div_eig_vectors * tem_scaling
    
    return msqf

def bfactor(enm, mode_subset=None, tem=None, 
            tem_factors=K_B):
    """
    Computes the isotropic B-factors/temperature factors/
    Deby-Waller factors for atoms/coarse-grained beads using 
    the mean-square fluctuation.
    These can be used to relate results obtained from ENMs 
    to experimental results.

    Parameters
    ----------
    enm : ANM or GNM
        Elastic network model; an instance of either an GNM or ANM
        object.
    mode_subset : ndarray, shape=(n,), dtype=int, optional
        Specifies the subset of modes considered in the MSF
        computation.
        Only non-trivial modes can be selected.
        The first mode is counted as 0 in accordance with
        Python conventions.
        If mode_subset is None, all modes except the first/first six
        trivial mode(s) (0, 0-5 respectively) are included.
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
    from .gnm import GNM
    from .anm import ANM

    if not isinstance(enm, (GNM, ANM)):
        raise ValueError(
            "Instance of GNM/ANM class expected."
            )

    msqf = mean_square_fluctuation(enm, mode_subset, tem, tem_factors)
    b_factors = ((8*np.pi**2)*msqf)/3
    
    return b_factors

def dcc(enm, mode_subset=None, norm=True, tem=None, tem_factors=K_B):
    r"""
    Computes the normalized *dynamic cross-correlation* between 
    nodes of the GNM/ANM.

    Parameters
    ----------
    enm : ANM or GNM
        Elastic network model; an instance of either an GNM or ANM
        object.
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
        DCC values for ENM nodes as NxN matrix.
    
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
    from .gnm import GNM
    from .anm import ANM
    
    if isinstance(enm, ANM):
        ntriv_modes = 6
    elif isinstance(enm, GNM):
        ntriv_modes = 1
    else:
        raise ValueError(
            "Instance of GNM/ANM class expected."
        )
    
    eig_values, eig_vectors = eigen(enm)
    
    # Choose modes included in computation; raise error, if trivial 
    # modes are included
    if mode_subset is None:
        mode_subset = np.arange(ntriv_modes, len(eig_values))
    elif any(mode_subset <= (ntriv_modes-1)):
        raise ValueError(
            "Trivial modes are included in the current selection."
            " Please check your input."
            )

    eig_values = eig_values[mode_subset]
    eig_vectors = eig_vectors[mode_subset]
    
    if isinstance(enm, GNM):
        # Reshape array of eigenvectors (k,n) -> (k,n,1)
        modes_reshaped = np.reshape(
            eig_vectors, (eig_vectors.shape[0], -1, 1)
        )
    # -> ANM
    else:
        # Reshape array of eigenvectors (k,3n) -> (k,n,3)
        modes_reshaped = np.reshape(
            eig_vectors, (eig_vectors.shape[0], -1, 3)
        )
    # Create residue modes matrix (3N -> N for ANMs)
    modes_mat_n = modes_reshaped[:, :, np.newaxis, :] *\
                  modes_reshaped[:, np.newaxis, :, :]
    modes_mat_n = np.sum(modes_mat_n, axis=-1)

    modes_mat_n = modes_mat_n / eig_values[:, np.newaxis, np.newaxis]
    dcc = np.sum(modes_mat_n, axis=0)

    # Compute the normalized DCC
    if norm:
        dcc_ii = np.diagonal(dcc)
        dcc_ii = np.reshape(dcc_ii, (1, len(dcc_ii)))
        dcc_ii = np.repeat(dcc_ii, repeats=len(dcc_ii), axis=0)
        
        dcc = dcc / np.sqrt(dcc_ii * dcc_ii.T)
    # Temperature weighting
    elif tem is not None:
        dcc = dcc * tem * tem_factors
    
    return dcc

## ANM specific functions
def normal_mode(anm, index, amplitude, frames, movement="sine"):
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
    from .anm import ANM
    
    if not isinstance(anm, ANM):
        raise ValueError(
            "Instance of ANM class expected."
        )
    else:
        _, eig_vectors = eigen(anm)
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

def linear_response(anm, force):
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
    from .anm import ANM
    
    if not isinstance(anm, ANM):
        raise ValueError(
            "Instance of ANM class expected."
        )
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
            raise ValueError(
                f"Expected 1D or 2D array, got {force.ndim} dimensions"
            ) 

        return np.dot(anm.covariance, force).reshape(len(anm._coord), 3)