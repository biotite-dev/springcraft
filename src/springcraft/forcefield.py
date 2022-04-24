"""
This module contains functionality for computing interaction matrices,
i.e. Kirchhoff and Hessian matrices.
"""

__name__ = "springcraft"
__author__ = "Patrick Kunzmann"
__all__ = ["ForceField", "InvariantForceField", "TypeSpecificForceField"]

import abc
import numpy as np
import biotite.structure as struc


class ForceField(metaclass=abc.ABCMeta):
    """
    Subclasses of this abstract base class define the force constants of
    the modeled springs between atoms in a *Elastic network model*.

    Attributes
    ----------
    natoms : int or None
        The number of atoms in the model.
        If a :class:`ForceField` does not depend on the respective
        atoms, i.e. `atom_i` and `atom_j` is unused in
        :meth:`force_constant()`, this attribute is ``None`` instead.
    """

    @abc.abstractmethod
    def force_constant(self, atom_i, atom_j, sq_distance):
        """
        Get the force constant for the interaction of the given atoms.

        ABSTRACT: Override when inheriting.

        Parameters
        ----------
        atom_i, atom_j : ndarray, shape=(n,), dtype=int
            The indices to the first and second atoms in each 
            interacting atom pair.
        sq_distance : ndarray, shape=(n,), dtype=float
            The distance between the atoms indicated by `atom_i` and
            `atom_j`.
        """
        pass

    @property
    def natoms(self):
        return None


class InvariantForceField(ForceField):
    """
    This force field treats every interaction with the same force
    constant.
    """

    def force_constant(self, atom_i, atom_j, sq_distance, adj_matrix):
        return np.ones(len(atom_i))


class TypeSpecificForceField(ForceField):
    """
    This force field treats every interaction with different force
    constants.
    """
    #def __init__(self, atoms):
    #    if not isinstance(atoms, struc.AtomArray):
    #        raise TypeError(
    #            f"Expected 'AtomArray', not {type(atoms).__name__}"
    #        )
    #    self._natoms = atoms.array_length()

    def force_constant(self, atom_i, atom_j, sq_distance, adj_matrix, res_list=None, noncov_inter_1=None, alpha=82, T=300, noncov_inter_2 = None, sq_dist = None, 
                        contact_switchoff = None, kappa=None, custom_nonbonded_matrix_1=None, custom_nonbonded_matrix_2=None):
        """
        Get the force constant for the interaction of the given atoms.

        Parameters
        ----------
        adj_matrix : ndarray, shape = (n, n), dtype=bool
            Contains interatomic contacts for delta_r_atoms <= cutoff 
        res_list : ndarray, shape = n, dtype=str
            AA sequence of the protein of interest
        noncov_inter_1|2 : str, None (optional)
            Keyword selection for matrices with AA specific non-bonded
            intrachain (noncov_inter_1) or interchain (noncov_inter_2)
            interactions.
            If set to None, nonbonded interactions are evaluated with a
            AA-indifferent factor kappa
        alpha : int, float
            Force constant factor for covalent bonds.
            Standard value: 82 * R * T * AA**2
        T : int, float
            Temperature
        sq_distance : ndarray, shape=(n,), dtype=float
            The distance between the atoms indicated by `atom_i` and
            `atom_j`.
        contact_switchoff: ndarray, shape = n, dtype=int or None
            Specifies the res_id for non-covalent switchoff
        kappa : int, float
            Scaling factor for non-covalent interactions, if AA-indifferent
            parametrization is chosen
            Standard value is the mean of Miyazawa-Jernigan: 3.166*RT/AA**2
        custom_nonbonded_matrix_1|2
        
        """

        # Homogenous parametrization for Kappa: 
        # Standard; for kappa != 1 -> custom input
        if noncov_inter_1 is None and noncov_inter_2 is None:
            # Non-covalent bonds: Homogenous -> kappa = average of Miyazawa-Jernigan AA interactions (R*T/Ang.**2)
            # W/o R*T?
            #kappa = 3.166*8.314*T
            kappa = 3.166
        # Access preset matrices with keywods?
        elif noncov_inter_1 == "mj1":
            raise NotImplementedError()
        
        # Insert kappa with adjacency matrix
        interaction_matrix = np.zeros((len(adj_matrix[:, 0]), len(adj_matrix[0, :])))
        interaction_matrix = np.where(adj_matrix, kappa, 0)

        # Get covalently bonded atoms in adjacency matrix; omit self-interactions
        # (Inelegant solution for now)
        cov_bond_lister = []
        for en, a in enumerate(adj_matrix):
            covbond = len(a)*[False]
            if en == 0:
                covbond[1] = True
            elif en == len(adj_matrix[:, 0])-1:
                covbond[-2] = True
            else:
                covbond[en-1] = True
                covbond[en+1] = True
            cov_bond_lister.append(covbond)
        covalent_matrix = np.array(cov_bond_lister)
        
        # W/o R*T?
        #alpha_rt = alpha * 8.314 * T
        alpha_rt = alpha
        # Replace entries in adjacency matrix corresponding to cov. bonds with alpha_rt        
        interaction_matrix = np.where(covalent_matrix, alpha_rt, interaction_matrix)
        # Set non-covalent self-interactions to 0, omit all 0-entries
        np.fill_diagonal(interaction_matrix, 0)
        
        force_constant = interaction_matrix[interaction_matrix != 0]

        return force_constant

    # Delete?
    #@property
    #def natoms(self):
    #    return self._natoms