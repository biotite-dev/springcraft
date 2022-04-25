import itertools
from os.path import join
import numpy as np
import pytest
import biotite.structure.io.mmtf as mmtf
import biotite.sequence as seq
import springcraft
from .util import data_dir


@pytest.fixture
def atoms():
    """
    Create a simple protein structure with two chains
    """
    mmtf_file = mmtf.MMTFFile.read(join(data_dir(), "1l2y.mmtf"))
    atoms = mmtf.get_structure(mmtf_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]

    ca_new_chain = ca.copy()
    # Ensure different chain IDs for both chains
    ca.chain_id[:] = "A"
    ca_new_chain.chain_id[:] = "B"
    # Simply merge both chains into new structure
    # The fact that both chains perfectly overlap
    # does not influence TypeSpecificForceField
    return ca + ca_new_chain

@pytest.fixture
def atoms_singlechain(atoms):
    ca = atoms[0:20]
    return ca
    
def test_type_specific_forcefield_homogeneous(atoms):
    BONDED = 1
    INTRA = 2
    INTER = 3

    ff = springcraft.TypeSpecificForceField(atoms, BONDED, INTRA, INTER)

    # Matrix should be symmetric
    assert np.allclose(ff.interaction_matrix, ff.interaction_matrix.T)
    for i in range(len(atoms)):
        for j in range(i, len(atoms)):
            force_constant = ff.interaction_matrix[i, j]
            try:
                if i == j:
                    assert force_constant == 0
                elif j == i+1 and atoms.chain_id[i] == atoms.chain_id[j]:
                    assert force_constant == BONDED
                elif atoms.chain_id[i] == atoms.chain_id[j]:
                    assert force_constant == INTRA
                else:
                    assert force_constant == INTER
            except AssertionError:
                print(f"Indices are {i} and {j}")
                print("Interaction matrix is:")
                print(ff.interaction_matrix)
                print()
                raise


def test_type_specific_forcefield_inhomogeneous(atoms):
    aa_list = [
        seq.ProteinSequence.convert_letter_1to3(letter)
        # Omit ambiguous amino acids and stop signal
        for letter in seq.ProteinSequence.alphabet.get_symbols()[:20]
    ]
    aa_to_index = {aa : i for i, aa in enumerate(aa_list)}
    # Maps pos-specific indices to type-specific_indices
    mapping = np.array([aa_to_index[aa] for aa in atoms.res_name])

    # Create symmetric random type-specific interaction matrices
    np.random.seed(0)
    triu = np.triu(np.random.rand(3, 20, 20))
    bonded, intra, inter = triu + np.transpose(triu, (0, 2, 1))

    ff = springcraft.TypeSpecificForceField(atoms, bonded, intra, inter)

    # Matrix should be symmetric
    assert np.allclose(ff.interaction_matrix, ff.interaction_matrix.T)
    for i in range(len(atoms)):
        for j in range(i, len(atoms)):
            force_constant = ff.interaction_matrix[i, j]
            try:
                if i == j:
                    assert force_constant == 0
                elif j == i+1 and atoms.chain_id[i] == atoms.chain_id[j]:
                    assert force_constant == pytest.approx(
                        bonded[mapping[i], mapping[j]]
                    )
                elif atoms.chain_id[i] == atoms.chain_id[j]:
                    assert force_constant == pytest.approx(
                        intra[mapping[i], mapping[j]]
                    )
                else:
                    assert force_constant == pytest.approx(
                        inter[mapping[i], mapping[j]]
                    )
            except AssertionError:
                print(f"Indices are {i} and {j}")
                print("Interaction matrix is:")
                print(ff.interaction_matrix)
                print()
                raise

def test_compare_with_biophysconnector(atoms_singlechain):
    bpc = np.loadtxt("./data/interaction_biophysconnector.txt", usecols=range(20))
    ff = springcraft.TypeSpecificForceField(atoms=atoms_singlechain, bonded=82, intra_chain=3.166, inter_chain=3.166)
    hessian, pairs = springcraft.compute_hessian(atoms_singlechain.coord, ff, 13.0)
    interaction = ff.interaction_matrix
    print(bpc.shape)
    print(interaction.shape)
    print(bpc)
    print(interaction)
    np.set_printoptions(precision=4)
    assert np.allclose(bpc, interaction, atol = 0.1)
    