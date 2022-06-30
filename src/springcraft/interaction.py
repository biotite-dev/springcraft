"""
This module contains functionality for computing interaction matrices,
i.e. Kirchhoff and Hessian matrices.
"""

__name__ = "springcraft"
__author__ = "Patrick Kunzmann"
__all__ = ["compute_kirchhoff", "compute_hessian"]

import numpy as np
import biotite.structure as struc


def compute_kirchhoff(coord, force_field, use_cell_list=True):
    """
    Compute the *Kirchhoff* matrix for atoms with given coordinates and
    the chosen force field.

    Parameters
    ----------
    coord : ndarray, shape=(n,3), dtype=float
        The coordinates.
    force_field : ForceField, natoms=n
        The :class:`ForceField` that defines the force constants.
    use_cell_list : bool, optional
        If true, a *cell list* is used to find atoms within cutoff
        distance instead of checking all pairwise atom distances.
        This significantly increases the performance for large number of
        atoms, but is slower for very small systems.
        If the `force_field` does not provide a cutoff, no cell list is
        used regardless.

    Returns
    -------
    kirchhoff : ndarray, shape=(n,n), dtype=float
        The computed *Kirchhoff* matrix.
    """
    # Convert into higher precision to avert numerical issues in
    # pseudoinverse calculation
    coord = coord.astype(np.float64, copy=False)
    pairs, _, sq_dist = _prepare_values_for_interaction_matrix(
        coord, force_field, use_cell_list
    )

    kirchhoff = np.zeros((len(coord), len(coord)))
    force_constants = force_field.force_constant(
        pairs[:,0], pairs[:,1], sq_dist
    )
    kirchhoff[pairs[:,0], pairs[:,1]] = -force_constants
    # Set values for main diagonal
    np.fill_diagonal(kirchhoff, -np.sum(kirchhoff, axis=0))
    
    return kirchhoff, pairs


def compute_hessian(coord, force_field, use_cell_list=True):
    """
    Compute the *Hessian* matrix for atoms with given coordinates and
    the chosen force field.

    Parameters
    ----------
    coord : ndarray, shape=(n,3), dtype=float
        The coordinates.
    force_field : ForceField, natoms=n
        The :class:`ForceField` that defines the force constants.
    use_cell_list : bool, optional
        If true, a *cell list* is used to find atoms within cutoff
        distance instead of checking all pairwise atom distances.
        This significantly increases the performance for large number of
        atoms, but is slower for very small systems.
        If the `force_field` does not provide a cutoff, no cell list is
        used regardless.

    Returns
    -------
    hessian : ndarray, shape=(n*3,n*3), dtype=float
        The computed *Hessian* matrix.
        Each dimension is partitioned in the form
        ``[x1, y1, z1, ... xn, yn, zn]``.
    """
    # Convert into higher precision to avert numerical issues in
    # pseudoinverse calculation
    coord = coord.astype(np.float64, copy=False)
    pairs, disp, sq_dist = _prepare_values_for_interaction_matrix(
        coord, force_field, use_cell_list
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
    
    # Reshape to (20*3, 20*3) matrix
    hessian = np.transpose(hessian, (0, 2, 1, 3)) \
              .reshape(len(coord)*3, len(coord)*3)

    return hessian, pairs
    

def _prepare_values_for_interaction_matrix(coord, force_field, use_cell_list):
    """
    Check input values and calculate common intermediate values for
    :func:`compute_kirchhoff()` and :func:`compute_hessian()`.

    Parameters
    ----------
    coord : ndarray, shape=(n,3), dtype=float
        The coordinates.
    force_field : ForceField, natoms=n
        The :class:`ForceField` that defines the force constants.
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
    cutoff_distance = force_field.cutoff_distance
    if cutoff_distance is None:
        # Include all possible interactions, except an atom with itself
        adj_matrix = np.ones((len(coord), len(coord)), dtype=bool)
    elif use_cell_list:
        cell_list = struc.CellList(coord, cutoff_distance,)
        adj_matrix = cell_list.create_adjacency_matrix(cutoff_distance)
    else:
        # Brute force: Calculate all pairwise squared distances
        disp_matrix = struc.displacement(
            coord[np.newaxis, :, :], coord[:, np.newaxis, :]
        )
        sq_dist_matrix = np.sum(disp_matrix * disp_matrix, axis=-1)
        adj_matrix = (sq_dist_matrix <= cutoff_distance**2)
    # Remove interactions of atoms with themselves
    np.fill_diagonal(adj_matrix, False)
    _patch_adjacency_matrix(
        adj_matrix, force_field.contact_shutdown,
        force_field.contact_pair_off, force_field.contact_pair_on
    )
    
    # Convert matrix to indices where interaction exists
    atom_i, atom_j = np.where(adj_matrix)
    pairs = np.array((atom_i, atom_j)).T

    # Get displacement vector for ANMs
    # and squared distances for distance-dependent force fields
    if cutoff_distance is None or use_cell_list:
        disp = struc.index_displacement(coord, pairs)
        sq_dist = np.sum(disp*disp, axis=-1)
    else:
        # Displacements and squared distances were already calculated
        disp = disp_matrix[pairs[:,0], pairs[:,1]]
        sq_dist = sq_dist_matrix[pairs[:,0], pairs[:,1]]

    return pairs, disp, sq_dist


def _patch_adjacency_matrix(matrix, contact_shutdown,
                            contact_pair_off, contact_pair_on):
    """
    Apply contacts that are artificially switched off/on to an
    adjacency matrix.
    The matrix is modified in-place.
    """
    if contact_shutdown is not None:
        matrix[:, contact_shutdown] = False
        matrix[contact_shutdown, :] = False
    if contact_pair_off is not None:
        atom_i, atom_j = contact_pair_off.T
        matrix[atom_i, atom_j] = False
        matrix[atom_j, atom_i] = False
    if contact_pair_on is not None:
        atom_i, atom_j = contact_pair_on.T
        if (atom_i == atom_j).any():
            raise ValueError(
                "Cannot turn on interaction of an atom with itself"
            )
        matrix[atom_i, atom_j] = True
        matrix[atom_j, atom_i] = True