import springcraft


class ChimericForceField(springcraft.ForceField):

    def __init__(self):
        self._type_ff = springcraft.TabulatedForceField.s_enm_13()
        self._dist_ff = springcraft.ParameterFreeForceField()

    def force_constant(self, atom_i, atom_j, sq_distance):
        return (
            self._type_ff(atom_i, atom_j, sq_distance) *
            self._dist_ff(atom_i, atom_j, sq_distance)
        )
            