"""
This module contains functionality for computing interaction matrices,
i.e. Kirchhoff and Hessian matrices.
"""

__name__ = "springcraft"
__author__ = "Patrick Kunzmann"
__all__ = ["compute_kirchhoff"]

import numpy as np
import biotite.structure as struc


def compute_kirchhoff(coord, force_field, cutoff_distance, use_cell_list=True):
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError(
            f"Expected coordinates with shape (n,3), got {coord.shape}"
        )
    if force_field.n_atoms is not None and len(coord) != force_field.n_atoms:
        raise ValueError(
            f"Got coordinates for {len(coord)} atoms, "
            f"but forcefield was built for {force_field.n_atoms} atoms"
        )
    
    # Find interacting atoms within cutoff distance
    if use_cell_list:
        cell_list = struc.CellList(coord, cutoff_distance)
        adj_matrix = cell_list.create_adjacency_matrix(cutoff_distance)
        atom_i, atom_j = np.where(adj_matrix)
    else:
        # Brute force: Calculate all pairwise squared distances
        disp = struc.displacement(
            coord[np.newaxis, :, :], coord[:, np.newaxis, :]
        )
        sq_dist_matrix = np.sum(disp*disp, axis=-1)
        atom_i, atom_j = np.where(sq_dist_matrix <= cutoff_distance**2)
    pairs = np.array((atom_i, atom_j)).T
    
    # Remove interactions of atoms with itself
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]

    # Get squared distances for distance-dependent force fields
    if use_cell_list:
        disp = struc.index_displacement(coord, pairs)
        sq_dist = np.sum(disp*disp, axis=-1)
    else:
        # Squared distances were already calculated
        sq_dist = sq_dist_matrix[pairs[:,0], pairs[:,1]]
    
    # Fill Kirchhoff matrix
    kirchhoff = np.zeros((len(coord), len(coord)))
    force_constants = force_field.force_constant(
        pairs[:,0], pairs[:,1], sq_dist
    )
    kirchhoff[pairs[:,0], pairs[:,1]] = -force_constants
    # Set values for main diagonal
    np.fill_diagonal(kirchhoff, -np.sum(kirchhoff, axis=0), wrap=False)

    return kirchhoff, pairs
