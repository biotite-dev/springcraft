import itertools
from os.path import join
import numpy as np
import pytest
import biotite.structure.io.mmtf as mmtf
import springcraft
from .util import data_dir


def test_type_specific_forcefield_sanity():
    BONDED = 1
    INTRA = 2
    INTER = 3

    mmtf_file = mmtf.MMTFFile.read(join(data_dir(), "1l2y.mmtf"))
    atoms = mmtf.get_structure(mmtf_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]
    #!#
    ca = ca[:5]
    #!#
    
    ca_new_chain = ca.copy()
    # Ensure different chain IDs for both chains
    ca.chain_id[:] = "A"
    ca_new_chain.chain_id[:] = "B"
    # Simply merge both chains into new structure
    # The fact that both chains perfectly overlap
    # does not influence TypeSpecificForceField
    merged = ca + ca_new_chain

    ff = springcraft.TypeSpecificForceField(merged, BONDED, INTRA, INTER)

    # Matrix should be symmetric
    assert np.allclose(ff.interaction_matrix, ff.interaction_matrix.T)
    for i in range(len(merged)):
        for j in range(i, len(merged)):
            force_constant = ff.interaction_matrix[i, j]
            try:
                if i == j:
                    assert force_constant == 0
                elif j == i+1 and merged.chain_id[i] == merged.chain_id[j]:
                    assert force_constant == BONDED
                elif merged.chain_id[i] == merged.chain_id[j]:
                    assert force_constant == INTRA
                else:
                    assert force_constant == INTER
            except AssertionError:
                print(f"Indices are {i} and {j}")
                print("Interaction matrix is:")
                print(ff.interaction_matrix)
                print()
                raise