"""
This module contains functionality for computing interaction matrices,
i.e. Kirchhoff and Hessian matrices.
"""

__name__ = "springcraft"
__author__ = "Patrick Kunzmann"
__all__ = ["compute_kirchhoff", "compute_hessian"]

import numpy as np
import biotite.structure as struc


def compute_kirchhoff(coord, force_field, cutoff_distance, use_cell_list=True):
    """
    Compute the *Kirchhoff* matrix for atoms with given coordinates and
    the chosen force field.

    Parameters
    ----------
    coord : ndarray, shape=(n,3), dtype=float
        The coordinates.
    force_field : ForceField, natoms=n
        The :class:`ForceField` that defines the force constants.
    cutoff_distance : float
        The interaction of two atoms is only considered, if the distance
        between them is smaller or equal to this value.
    use_cell_list : bool, optional
        If true, a *cell list* is used to find atoms within cutoff
        distance instead of a brute-force approach.
        This significantly increases the performance for large number of
        atoms, but is slower for very small systems.

    Returns
    -------
    kirchhoff : ndarray, shape=(n,n), dtype=float
        The *Kirchhoff* matrix for this model.
    """
    pairs, _, sq_dist = _prepare_values_for_interaction_matrix(
        coord, force_field, cutoff_distance, use_cell_list
    )

    kirchhoff = np.zeros((len(coord), len(coord)))
    force_constants = force_field.force_constant(
        pairs[:,0], pairs[:,1], sq_dist
    )
    kirchhoff[pairs[:,0], pairs[:,1]] = -force_constants
    # Set values for main diagonal
    np.fill_diagonal(kirchhoff, -np.sum(kirchhoff, axis=0))
    
    return kirchhoff, pairs


def compute_hessian(coord, force_field, cutoff_distance, use_cell_list=True):
    """
    Compute the *Hessian* matrix for atoms with given coordinates and
    the chosen force field.

    Parameters
    ----------
    coord : ndarray, shape=(n,3), dtype=float
        The coordinates.
    force_field : ForceField, natoms=n
        The :class:`ForceField` that defines the force constants.
    cutoff_distance : float
        The interaction of two atoms is only considered, if the distance
        between them is smaller or equal to this value.
    use_cell_list : bool, optional
        If true, a *cell list* is used to find atoms within cutoff
        distance instead of a brute-force approach.
        This significantly increases the performance for large number of
        atoms, but is slower for very small systems.
    
    Returns
    -------
    hessian : ndarray, shape=(n,n,3,3), dtype=float
        The *Hessian* matrix for this model.
        The super elements are represented by the last 2 dimensions.
    """
    pairs, disp, sq_dist = _prepare_values_for_interaction_matrix(
        coord, force_field, cutoff_distance, use_cell_list
    )

    # Hessian matrix has 3x3 matrices as superelements
    hessian = np.zeros((len(coord), len(coord), 3, 3))
    force_constants = force_field.force_constant(
        pairs[:,0], pairs[:,1], sq_dist
    )
    hessian[pairs[:,0], pairs[:,1]] = (
        -force_constants[:, np.newaxis, np.newaxis]
        / sq_dist[:, np.newaxis, np.newaxis]
        * disp[:, :, np.newaxis] * disp[:, np.newaxis, :]
    )
    # Set values for main diagonal
    indices = np.arange(len(coord))
    hessian[indices, indices] = -np.sum(hessian, axis=0)
    
    return hessian, pairs


def _prepare_values_for_interaction_matrix(coord, force_field, cutoff_distance,
                                           use_cell_list=True):
    """
    Check input values and calculate common intermediate values for
    :func:`compute_kirchhoff()` and :func:`compute_hessian()`.

    Parameters
    ----------
    coord : ndarray, shape=(n,3), dtype=float
        The coordinates.
    force_field : ForceField, natoms=n
        The :class:`ForceField` that defines the force constants.
    cutoff_distance : float
        The interaction of two atoms is only considered, if the distance
        between them is smaller or equal to this value.
    use_cell_list : bool, optional
        If true, a *cell list* is used to find atoms within cutoff
        distance instead of a brute-force approach.
        This significantly increases the performance for large number of
        atoms, but is slower for very small systems.
    
    Returns
    -------
    pairs : ndarray, shape=(k,2), dtype=int
        Indices for interacting atoms, i.e. atoms within
        `cutoff_distance`.
    disp : ndarray, shape=(k,3), dtype=float
        The displacement vector for the atom `pairs`.
    sq_dist : ndarray, shape=(k,3), dtype=float
        The squared distance for the atom `pairs`.
    """
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError(
            f"Expected coordinates with shape (n,3), got {coord.shape}"
        )
    if force_field.natoms is not None and len(coord) != force_field.natoms:
        raise ValueError(
            f"Got coordinates for {len(coord)} atoms, "
            f"but forcefield was built for {force_field.natoms} atoms"
        )
    
    # Find interacting atoms within cutoff distance
    if use_cell_list:
        cell_list = struc.CellList(coord, cutoff_distance)
        adj_matrix = cell_list.create_adjacency_matrix(cutoff_distance)
        atom_i, atom_j = np.where(adj_matrix)
    else:
        # Brute force: Calculate all pairwise squared distances
        disp_matrix = struc.displacement(
            coord[np.newaxis, :, :], coord[:, np.newaxis, :]
        )
        sq_dist_matrix = np.sum(disp_matrix * disp_matrix, axis=-1)
        atom_i, atom_j = np.where(sq_dist_matrix <= cutoff_distance**2)
    pairs = np.array((atom_i, atom_j)).T
    
    # Remove interactions of atoms with itself
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]

    # Get displacement vector for ANMs
    # and squared distances for distance-dependent force fields
    if use_cell_list:
        disp = struc.index_displacement(coord, pairs)
        sq_dist = np.sum(disp*disp, axis=-1)
    else:
        # Displacements and squared distances were already calculated
        disp = disp_matrix[pairs[:,0], pairs[:,1]]
        sq_dist = sq_dist_matrix[pairs[:,0], pairs[:,1]]

    return pairs, disp, sq_dist