import numpy as np
import biotite.structure.io.mmtf as mmtf
import biotite.database.rcsb as rcsb
import ammolite
import springcraft


PDB_ID = "1MUG"


# Load structure
mmtf_file = mmtf.MMTFFile.read(rcsb.fetch(PDB_ID, "mmtf"))
atoms = mmtf.get_structure(mmtf_file, model=1)
ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]

ff = springcraft.InvariantForceField(13.0)
anm = springcraft.ANM(ca, ff)
AMPLITUDE = 100
force = np.zeros((len(ca), 3))
force[40] = [AMPLITUDE, 0, 0]
displacement = anm.linear_response(force)


ammolite.launch_interactive_pymol()
pymol_object = ammolite.PyMOLObject.from_structure(ca)

start_coord = ca.coord
end_coord = ca.coord + displacement * 10
ammolite.draw_arrows(start_coord, end_coord)
pymol_object.orient()