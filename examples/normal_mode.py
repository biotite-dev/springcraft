# Code source: Patrick Kunzmann
# License: BSD 3 clause

import numpy as np
import biotite.structure as struc
import biotite.structure.io.mmtf as mmtf
import biotite.database.rcsb as rcsb
import ammolite
import springcraft


PDB_ID = "1MUG"
# The normal mode to be visualized
# '0' is the first (and most significant) one
MODE = 0
# The number of frames (models) per oscillation
FRAMES = 60
# The maximum oscillation amplitude for an atom
# (The length of the ANM's eigenvectors make only sense when compared
# relative to each other, the absolute values have no significance)
AMPLITUDE = 5


# Load structure
mmtf_file = mmtf.MMTFFile.read(rcsb.fetch(PDB_ID, "mmtf"))
structure = mmtf.get_structure(mmtf_file, model=1, include_bonds=True)

# Filter first peptide chain
protein_chain = structure[
    struc.filter_amino_acids(structure)
    & (structure.chain_id == structure.chain_id[0])
]
# Filter CA atoms
ca = protein_chain[protein_chain.atom_name == "CA"]

ff = springcraft.InvariantForceField()
anm = springcraft.ANM(ca, ff, 13.0)
disp = anm.normal_mode(MODE, AMPLITUDE, FRAMES)

# Apply oscillation of CA atom to all atoms in the corresponding residue
oscillation = np.zeros((FRAMES, len(protein_chain), 3))
residue_starts = struc.get_residue_starts(
    protein_chain,
    # The last array element will be the length of the atom array,
    # i.e. no valid index
    add_exclusive_stop=True
)
for i in range(len(residue_starts) -1):
    res_start = residue_starts[i]
    res_stop = residue_starts[i+1]
    oscillation[:, res_start:res_stop, :] \
        = protein_chain.coord[res_start:res_stop, :] + disp[:, i:i+1, :]

# An atom array stack containing all frames
oscillating_structure = struc.from_template(protein_chain, oscillation)


ammolite.launch_interactive_pymol()

pymol_object = ammolite.PyMOLObject.from_structure(oscillating_structure)