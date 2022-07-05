Advanced usage
==============

.. currentmodule:: springcraft

Adding or removing contacts between atoms
-----------------------------------------
Altering the contacts between two atom can be achieved using the
:class:`PatchedForceField`.
It wraps another :class:`ForceField` and applied custom changes to it.
As example the contact between the first and second residue should be removed.

.. code-block:: python

    ff = springcraft.PatchedForceField(
        springcraft.InvariantForceField(cutoff_distance=13.0),
        contact_pair_off=[(0,1)]
    )
    anm = springcraft.ANM(atoms, ff)



Defining a custom force field
-----------------------------
To implement a custom force field, the :class:`ForceField` base
class needs to be inherited.
In the most basic case, only the method :meth:`ForceField.force_constant()`
must be overridden.
For atom pairs within cutoff distance this function must return the spring
force constant.
Furthermore, some properties can be overridden, to further customize the
force field:

    - `cutoff_distance` defines at which maximum distance atom pairs are
      considered.
      By default, no cutoff is applied.
    - `contact_shutdown`, `contact_pair_off` and `contact_pair_on` are
      used to add or remove atom pairs.
      For these atoms the cutoff distance is ignored.
      These are mainly used by the :class:`PatchedForceField`.
    - `natoms` gives the number of atoms in the model, if the
      :class:`ForceField` requires an :class:`AtomArray`.
      This is simply used to check whether the model used for the force field
      fits the coordinates given for `Kirchhoff` and `Hessian` calculation.

For the purpose of an example a chimeric force field is created


.. code-block:: python

    class ChimericForceField(springcraft.ForceField):
        """
        Uses a mixture of amino acid type and distance dependency.
        """

        def __init__(self, atoms):
            self._type_ff = springcraft.TabulatedForceField.s_enm_13(atoms)
            self._dist_ff = springcraft.ParameterFreeForceField()

        def force_constant(self, atom_i, atom_j, sq_distance):
            return (
                self._type_ff(atom_i, atom_j, sq_distance) *
                self._dist_ff(atom_i, atom_j, sq_distance)
            )
        
        @property
        def natoms(self):
            return self._type_ff.natoms


    ff = ChimericForceField(atoms)