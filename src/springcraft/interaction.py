"""
This module contains functionality for computing interaction matrices,
i.e. Kirchhoff and Hessian matrices.
"""

__name__ = "springcraft"
__author__ = "Patrick Kunzmann"
__all__ = ["compute_kirchhoff", "compute_hessian"]

import numpy as np
import biotite.structure as struc


def compute_kirchhoff(coord, force_field, cutoff_distance=None, use_cell_list=True,
                      contact_shutdown=None, contact_pair_off=None, 
                      contact_pair_on=None):
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
        The computed *Kirchhoff* matrix.
    """
    pairs, _, sq_dist = _prepare_values_for_interaction_matrix(
        coord, force_field, cutoff_distance, use_cell_list, contact_shutdown, 
        contact_pair_off, contact_pair_on
    )

    kirchhoff = np.zeros((len(coord), len(coord)))
    force_constants = force_field.force_constant(
        pairs[:,0], pairs[:,1], sq_dist
    )
    kirchhoff[pairs[:,0], pairs[:,1]] = -force_constants
    # Set values for main diagonal
    np.fill_diagonal(kirchhoff, -np.sum(kirchhoff, axis=0))
    
    return kirchhoff, pairs

# TODO
# Messy solution for now: for cutoff_distance == None -> interaction between all atoms
# Better solution: Static FF classes should dictate cutoffs
def compute_hessian(coord, force_field, cutoff_distance=None, rmin = None, use_cell_list=True,
                    contact_shutdown=None, contact_pair_off=None,
                    contact_pair_on=None):
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
    contact_shutdown : int, array-like (elements -> int; shape=(n,)), optional
        Undirected shutdown of contacts involving the atom with the specified
        ID. 
    contact_pair_off : tuple (elements -> int; shape=(2,)), array-like
                       (elements -> tuple [elements -> int; shape=(2,)]; 
                       shape=(n,))
        Contacts between individual atom pairs are switched off.
    contact_pair_on  : tuple (elements -> int; shape=(2,)), array-like
                       (elements -> tuple [elements -> int; shape=(2,)]; 
                       shape=(n,))
        Contacts between individual atom pairs are established.

    Returns
    -------
    hessian : ndarray, shape=(n*3,n*3), dtype=float
        The computed *Hessian* matrix.
        Each dimension is partitioned in the form
        ``[x1, y1, z1, ... xn, yn, zn]``.
    """

    # Determine cut-off, if not specified in input.
    ff_specific_cutoff = force_field.ff_type_cutoff

    # Overwrite cut-off with FF-specific value, if input is None
    if cutoff_distance is None:
        cutoff_distance = ff_specific_cutoff
    elif ff_specific_cutoff is not None:
        if cutoff_distance is None or cutoff_distance != ff_specific_cutoff:
            print("Warning: Input cutoff and standard-cutoff for the chosen forcefield do not match.")
            print(f"With {cutoff_distance} Angstrom and {ff_specific_cutoff} Angstrom respectively")
            print("Standard FF distances will be overwritten.")
    else:
        if cutoff_distance is not None or cutoff_distance:
            print("Warning: Input cutoff and standard-cutoff for the chosen forcefield do not match.")
            print(f"With {cutoff_distance} Angstrom and {ff_specific_cutoff} Angstrom respectively")
            print("Standard FF distances will be overwritten.")

    # Determine rmin, if not specified in input
    ff_specific_rmin = force_field.ff_rmin
    
    if rmin is None:
        rmin = ff_specific_rmin

    pairs, disp, sq_dist = _prepare_values_for_interaction_matrix(
        coord, force_field, cutoff_distance, rmin, use_cell_list, contact_shutdown, 
        contact_pair_off, contact_pair_on
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
    
def _prepare_values_for_interaction_matrix(coord, force_field, cutoff_distance, rmin,
                                           use_cell_list=True,
                                           contact_shutdown=None, 
                                           contact_pair_off=None, 
                                           contact_pair_on=None):
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
        contact_shutdown : int, array-like (elements -> int; shape=(n,)), optional
        Undirected shutdown of contacts involving the atom with the specified
        ID. 
    contact_pair_off : tuple (elements -> int; shape=(2,)), array-like
                       (elements -> tuple [elements -> int; shape=(2,)]; 
                       shape=(n,))
        Contacts between individual atom pairs are switched off.
    contact_pair_on  : tuple (elements -> int; shape=(2,)), array-like
                       (elements -> tuple [elements -> int; shape=(2,)]; 
                       shape=(n,))
        Contacts between individual atom pairs are established.

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
        
    # Modification of adjacency matrix with switch on/off matrices
    # Translate contact_pair_off array into boolean array (n, n)
    if contact_pair_off is not None:   
        # Test, whether input is Array-Like
        if not isinstance(contact_pair_off, (list, np.ndarray)):
            contact_pair_off = [contact_pair_off]
        if not all(isinstance(test, tuple) for test in contact_pair_off):
            raise ValueError(
            "Expected list or array with tuples."
            )
        
        switchoff_matrix = np.full((len(coord), len(coord)), True)
        # Iterate over indices
        for switch in contact_pair_off:
            # For regular and inverted tuple: Set contact modification
            # matrix to False
            switchoff_matrix[switch[0]][switch[1]] = False
            switchoff_matrix[switch[1]][switch[0]] = False

    # Translate contact_pair_on array into boolean array (n, n)
    if contact_pair_on is not None:   
        # Test, whether input is Array-Like
        if not isinstance(contact_pair_on, (list, np.ndarray)):
            contact_pair_on = [contact_pair_on]
        if not all(isinstance(test, tuple) for test in contact_pair_on):
            raise ValueError(
            "Expected list or array with tuples."
            )

        switchon_matrix = np.full((len(coord), len(coord)),\
                                   False
                                    )
        # Iterate over indices
        for switch in contact_pair_on:
            # For regular and inverted tuple: Set contact modification
            # matrix to True
            switchon_matrix[switch[0]][switch[1]] = True
            switchon_matrix[switch[1]][switch[0]] = True

    # TODO Messy solution for cutoff_distance 
    if cutoff_distance is None:
        cutoff_distance = 10000
    
    # Find interacting atoms within cutoff distance
    if use_cell_list:
        # Contact switch-off
        # Translate contact_shutdown list into boolean array (n, )
        if contact_shutdown is not None:   
            # Test, whether input is Array-Like
            if not isinstance(contact_shutdown, (list, np.ndarray)):
                contact_shutdown = [contact_shutdown]
            shutdown_array = np.full((len(coord)), True)
            # Iterate over indices
            for shutdown_atom in contact_shutdown:
                shutdown_array[shutdown_atom] = False
        else:
            shutdown_array = None

        # Cell list with shutdown_array
        cell_list = struc.CellList(coord, cutoff_distance, 
                                   selection=shutdown_array
                                   )

        adj_matrix = cell_list.create_adjacency_matrix(cutoff_distance)

        if contact_pair_off is not None:
            
            # Transfer False entries from switchoff_matrix to adj_matrix
            adj_matrix = np.where(switchoff_matrix == False, switchoff_matrix,
                                  adj_matrix)

        if contact_pair_on is not None:   
            
            # Transfer True entries from switchon_matrix to adj_matrix
            adj_matrix = np.where(switchon_matrix == True, switchon_matrix,
                                  adj_matrix)

        atom_i, atom_j = np.where(adj_matrix)
    else:
        if contact_shutdown is not None:
            coord = np.delete(coord)
        # Brute force: Calculate all pairwise squared distances
        disp_matrix = struc.displacement(
            coord[np.newaxis, :, :], coord[:, np.newaxis, :]
        )
        sq_dist_matrix = np.sum(disp_matrix * disp_matrix, axis=-1)
        brute_adj_matrix = (sq_dist_matrix <= cutoff_distance**2)

        if contact_pair_off is not None:
            brute_adj_matrix = np.where(contact_pair_off == False, 
                                        contact_pair_off,
                                        brute_adj_matrix)
        if contact_pair_on is not None:
            brute_adj_matrix = np.where(contact_pair_on == True, 
                                        contact_pair_on,
                                        brute_adj_matrix)

        atom_i, atom_j = np.where(brute_adj_matrix)
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

    # Set disp and sq_dist to lower distance threshold
    if rmin is not None:
        disp[(disp < rmin)] = rmin
        sq_dist[(sq_dist < rmin**2)] = rmin**2

    return pairs, disp, sq_dist