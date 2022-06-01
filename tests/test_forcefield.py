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
    Create a simple protein structure with two chains.
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
    """
    Check contents of position-specifc interaction matrix, where the
    interactions are independent of the amino acid.
    Each matrix element is checked in a non-vectorized manner.
    """
    BONDED = 1
    INTRA = 2
    INTER = 3

    ff = springcraft.TypeSpecificForceField(atoms, BONDED, INTRA, INTER)

    # Expect only a single distance bin in interaction matrix
    assert ff.interaction_matrix.shape[2] == 1
    interaction_matrix = ff.interaction_matrix[:, :, 0]
    # Matrix should be symmetric
    assert np.allclose(interaction_matrix, interaction_matrix.T)
    for i in range(len(atoms)):
        for j in range(i, len(atoms)):
            force_constant = interaction_matrix[i, j]
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
                print(interaction_matrix)
                print()
                raise


def test_type_specific_forcefield_inhomogeneous(atoms):
    """
    Check contents of position-specifc interaction matrix, where the
    interactions for each type of amino acid is chosen randomly.
    Each matrix element is checked in a non-vectorized manner.
    """
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

    # Expect only a single distance bin in interaction matrix
    assert ff.interaction_matrix.shape[2] == 1
    interaction_matrix = ff.interaction_matrix[:, :, 0]
    # Matrix should be symmetric
    assert np.allclose(interaction_matrix, interaction_matrix.T)
    for i in range(len(atoms)):
        for j in range(i, len(atoms)):
            force_constant = interaction_matrix[i, j]
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
                print(interaction_matrix)
                print()
                raise


def test_type_specific_forcefield_distance(atoms):
    """
    Check whether the calculated force constants are correct for a force
    field with distance depdendance, but no amino acid or connection
    dependence.
    """
    N_EDGES = 100
    MAX_DISTANCE = 30
    N_SAMPLE_INTERACTIONS = 500

    np.random.seed(0)
    distance_edges = np.sort(np.random.rand(N_EDGES) * MAX_DISTANCE)
    # Bin edges must be unique
    assert np.all(np.unique(distance_edges) == distance_edges)
    # All edges should be lower than MAX_DISTANCE to include the bin
    # after the final edge in 'ff.force_constant()' (see below)
    assert np.all(distance_edges < MAX_DISTANCE)

    # Only distance dependent force constants 
    fc = np.arange(N_EDGES + 1)
    ff = springcraft.TypeSpecificForceField(atoms, fc, fc, fc, distance_edges)

    ## Check correctness of interaction matrix
    assert ff.interaction_matrix.shape == (len(atoms), len(atoms), N_EDGES+1)
    for i in range(ff.interaction_matrix.shape[0]):
        for j in range(ff.interaction_matrix.shape[0]):
            if i == j:
                # No interaction of an atom with itself
                assert np.all(ff.interaction_matrix[i, j] == 0)
            else:
                assert np.all(ff.interaction_matrix[i, j] == fc)
    
    ## Check if 'force_constant()' gives the correct values
    atom_i = np.random.randint(len(atoms), size=N_SAMPLE_INTERACTIONS)
    atom_j = np.random.randint(len(atoms), size=N_SAMPLE_INTERACTIONS)
    sample_bin_indices = np.random.randint(N_EDGES+1, size=N_SAMPLE_INTERACTIONS)
    sample_dist = np.append(distance_edges, [MAX_DISTANCE])[sample_bin_indices]
    assert MAX_DISTANCE in sample_dist
    test_force_constants = ff.force_constant(atom_i, atom_j, sample_dist)
    # Force constants (i.e. 'fc') are designed such that the force is
    # the index of the bin
    ref_force_constans = np.where(
        atom_i != atom_j,
        sample_bin_indices,
        0
    )
    assert np.allclose(test_force_constants, ref_force_constans)


N_RES = 20
@pytest.mark.parametrize("shape, n_edges, is_valid", [
    [(),           None, True ],
    [(),           1,    True ],
    [(),           10,   True ],
    [(10,),        None, False],
    [(10,),        1,    False],
    [(10,),        10,   False],
    [(11,),        10,   True ],
    [( 1,),        None, True ],
    [(20,  1),     None, False],
    [(20, 30),     None, False],
    [( 1, 20),     None, False],
    [(30, 20),     None, False],
    [(20, 20),     None, True ],
    [(20, 20),     1,    True ],
    [(20, 20),     10,   True ],
    [(20,  1, 11), 10,   False],
    [(20, 30, 11), 10,   False],
    [( 1, 20, 11), 10,   False],
    [(30, 20, 11), 10,   False],
    [(20, 20, 11), 10,   True ],
    [(20, 20,  1), None, True ],
    [(20, 20, 10), 10,   False],
])
def test_type_specific_forcefield_input_shapes(atoms, shape, n_edges, is_valid):
    """
    Test whether all supported input shapes are handled properly.
    This is chaked based on the calculated interaction matrix.
    Unsupported cases should raise an exception.
    """
    np.random.seed(0)
    fc = np.ones(shape) if shape != () else 1
    edges = np.arange(n_edges) if n_edges is not None else None

    if is_valid:
        ff = springcraft.TypeSpecificForceField(atoms, fc, fc, fc, edges)
        n_bins = n_edges+1 if n_edges is not None else 1
        assert ff.interaction_matrix.shape == (40, 40, n_bins)
    else:
        with pytest.raises(IndexError):
            ff = springcraft.TypeSpecificForceField(atoms, fc, fc, fc, edges)


@pytest.mark.parametrize(
    "name",
    ["hinsen_calpha", "s_enm_10", "s_enm_13", "d_enm", "sd_enm", "e_anm", "pf_enm"]
)
def test_type_specific_forcefield_predefined(atoms, name):
    """
    Test the instantiation of predefined force fields.
    These are implemented as static methods that merely require the
    structure as input.
    """
    meth = getattr(springcraft.TypeSpecificForceField, name)
    ff = meth(atoms)

def test_compare_with_biophysconnector(atoms_singlechain):
    """
    Comparisons between Hessians computed for eANMs using springcraft
    and BioPhysConnectoR.
    Cut-off: 1.3 nm.
    """
    # Load model 1 from 1l2y.mmtf
    mmtf_file = mmtf.MMTFFile.read(join(data_dir(), "1l2y.mmtf"))
    atoms = mmtf.get_structure(mmtf_file, model=1)
    ca = atoms[atoms.atom_name == "CA"]

    ff = springcraft.TypeSpecificForceField.e_anm(atoms=ca)
    # Load static class: eANM
    test_hess, _ = springcraft.compute_hessian(ca.coord, ff, 13.0)
    
    # Load .csv file data from BiophysConnectoR
    ref_hess = np.genfromtxt("./data/hessian_eANM_BioPhysConnectoR.csv", skip_header=1, delimiter=",")
    
    assert np.allclose(test_hess, ref_hess)

#def test_compare_with_bio3d(atoms_singlechain):
#    """
#    Comparisons between Hessians computed for ANMS using springcraft
#    and Bio3D.
#    The following ENM forcefields are compared:
#    Hinsen-Calpha, sdENM and pfENM.
#    Cut-off: 1.3 nm.
#    """
#    # Load model 1 from 1l2y.mmtf
#    mmtf_file = mmtf.MMTFFile.read(join(data_dir(), "1l2y.mmtf"))
#    atoms = mmtf.get_structure(mmtf_file, model=1)
#    ca = atoms[atoms.atom_name == "CA"]
#
#    
#    hinsen = springcraft.TypeSpecificForceField.hinsen_calpha(atoms=ca)
#    sd = springcraft.TypeSpecificForceField.sd_enm(atoms=ca)
#    pf = springcraft.TypeSpecificForceField.pf_enm(ca)
#
#    test_hess = []
#
#    test_hess.append(springcraft.compute_hessian(ca.coord, ff, 13.0) for ff in [hinsen, sd, pf])
#    
#    for ff in [hinsen, sd, pf]:
#        # Load static class: eANM
#        test_hess, _ = springcraft.compute_hessian(ca.coord, ff, 13.0)
#
#        # Load .csv file data from BiophysConnectoR
#        ref_hess = np.genfromtxt("./data/hessian_eANM_BioPhysConnectoR.csv", skip_header=1, delimiter=",")
#    
#    assert np.allclose(test_hess, ref_hess)