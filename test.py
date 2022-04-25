import biotite.structure.io.mmtf as mmtf
import springcraft
import numpy as np
import os
 
mmtf_file = mmtf.MMTFFile.read("./tests/data/1l2y.mmtf")
atoms = mmtf.get_structure(mmtf_file, model=1)
ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]
ff = springcraft.TypeSpecificForceField(atoms=ca)
hessian, pairs = springcraft.compute_hessian(ca.coord, ff, 13.0)
   
np.set_printoptions(linewidth=100)
print(hessian)
