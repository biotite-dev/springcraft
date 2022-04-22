import numpy as np
import biotite.structure.io.mmtf as mmtf
import biotite.database.rcsb as rcsb
import ammolite
import springcraft


PDB_ID = "1MUG"


from pymol import cmd, cgo, CmdException
def draw_arrow(name, start, end, radius=0.1, head_radius=0.20, head_length=0.5,
               color=(0.5, 0.5, 0.5)):
        CYLINDER = 9.0
        CONE = 27.0

        normal = (end - start) / np.linalg.norm(end - start)
        middle = end - normal * head_length

        ammolite.cmd.load_cgo(
            [
                CYLINDER, *start, *middle, radius, *color, *color,
                CONE, *middle, *end, head_radius, 0.0, *color, *color,
                1.0, 0.0
            ],
            name
        )


# Load structure
mmtf_file = mmtf.MMTFFile.read(rcsb.fetch(PDB_ID, "mmtf"))
atoms = mmtf.get_structure(mmtf_file, model=1)
ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]

ff = springcraft.InvariantForceField()
anm = springcraft.ANM(ca, ff, 13.0)
AMPLITUDE = 100
force = np.zeros((len(ca), 3))
force[40] = [AMPLITUDE, 0, 0]
displacement = anm.linear_response(force)


ammolite.launch_interactive_pymol()
pymol_object = ammolite.PyMOLObject.from_structure(ca)

start_coord = ca.coord
end_coord = ca.coord + displacement
for i in range(len(displacement)):
    draw_arrow(f"vector{i}", start_coord[i], end_coord[i])
pymol_object.orient()