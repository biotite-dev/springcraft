"""
This module contains functionality for computing interaction matrices,
i.e. Kirchhoff and Hessian matrices.
"""

__name__ = "springcraft"
__author__ = "Patrick Kunzmann"
__all__ = ["ForceField", "InvariantForceField"]

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

    def force_constant(self, atom_i, atom_j, sq_distance):
        return np.ones(len(atom_i))


class TypeSpecificForceField(ForceField):

    def __init__(self, atoms):
        if not isinstance(atoms, struc.AtomArray):
            raise TypeError(
                f"Expected 'AtomArray', not {type(atoms).__name__}"
            )
        self._natoms = atoms.array_length()

    def force_constant(self, atom_i, atom_j, sq_distance):
        raise NotImplementedError()

    @property
    def natoms(self):
        return self._natoms