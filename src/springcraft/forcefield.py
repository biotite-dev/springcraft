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

    @abc.abstractmethod
    def force_constant(self, atom_i, atom_j, sq_distance):
        pass

    @property
    def n_atoms(self):
        return None


class InvariantForceField(ForceField):

    def force_constant(self, atom_i, atom_j, sq_distance):
        return 1


class TypeSpecificForceField(ForceField):

    def __init__(self, atoms):
        if not isinstance(atoms, struc.AtomArray):
            raise TypeError(
                f"Expected 'AtomArray', not {type(atoms).__name__}"
            )
        self._n_atoms = atoms.array_length()

    def force_constant(self, atom_i, atom_j, sq_distance):
        raise NotImplementedError()

    @property
    def n_atoms(self):
        return self._n_atoms