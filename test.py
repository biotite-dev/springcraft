import biotite.structure.io.mmtf as mmtf
import springcraft
import numpy as np


mmtf_file = mmtf.MMTFFile.read("./tests/data/1l2y.mmtf")
atoms = mmtf.get_structure(mmtf_file, model=1)
ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]
ff = springcraft.InvariantForceField()
hessian, pairs = springcraft.compute_hessian(ca.coord, ff, 7.0)

np.set_printoptions(linewidth=100)
print(hessian)

ff = springcraft.TypeSpecificForceField()
hessian, pairs = springcraft.compute_hessian(ca.coord, ff, 13)

np.set_printoptions(linewidth=100)
print(hessian)