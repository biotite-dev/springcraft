import itertools
from os.path import join
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io.mmtf as mmtf
import biotite.sequence as seq
import springcraft
from springcraft.forcefield import InvariantForceField
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
    # does not influence TabulatedForceField
    return ca + ca_new_chain

@pytest.fixture
def atoms_singlechain(atoms):
    ca = atoms[0:20]
    return ca


def test_patched_force_field_shutdown(atoms):
    N_CONTACTS = 5
    
    np.random.seed(0)
    shutdown_indices = np.random.choice(
        np.arange(len(atoms)), size=N_CONTACTS, replace=False
    )
    
    ref_ff = InvariantForceField(7.0)
    ref_kirchhoff, _ = springcraft.compute_kirchhoff(atoms.coord, ref_ff)
    # Manual shutdown of contacts after Kirchhoff calculation
    ref_kirchhoff[shutdown_indices, :] = 0
    ref_kirchhoff[:, shutdown_indices] = 0
    
    test_ff = springcraft.PatchedForceField(
        ref_ff,
        contact_shutdown=shutdown_indices
    )
    test_kirchhoff, _ = springcraft.compute_kirchhoff(atoms.coord, test_ff)
    
    # Main diagonal is not easily adjusted
    # -> simply set main diagonal of ref and test matrix to 0
    np.fill_diagonal(test_kirchhoff, 0)
    np.fill_diagonal(ref_kirchhoff, 0)
    assert np.all(test_kirchhoff == ref_kirchhoff)


def test_patched_force_field_pairs_off(atoms):
    N_CONTACTS = 5
    
    np.random.seed(0)
    off_indices = np.random.choice(
        np.arange(len(atoms)), size=(N_CONTACTS, 2), replace=False
    )
    
    ref_ff = InvariantForceField(7.0)
    ref_kirchhoff, _ = springcraft.compute_kirchhoff(atoms.coord, ref_ff)
    # Manual shutdown of contacts after Kirchhoff calculation
    atom_i, atom_j = off_indices.T
    ref_kirchhoff[atom_i, atom_j] = 0
    ref_kirchhoff[atom_j, atom_i] = 0
    
    test_ff = springcraft.PatchedForceField(
        ref_ff,
        contact_pair_off=off_indices
    )
    test_kirchhoff, _ = springcraft.compute_kirchhoff(atoms.coord, test_ff)
    
    # Main diagonal is not easily adjusted
    # -> simply set main diagonal of ref and test matrix to 0
    np.fill_diagonal(test_kirchhoff, 0)
    np.fill_diagonal(ref_kirchhoff, 0)
    assert np.all(test_kirchhoff == ref_kirchhoff)


def test_patched_force_field_pairs_on(atoms):
    N_CONTACTS = 5
    
    np.random.seed(0)
    on_indices = np.random.choice(
        np.arange(len(atoms)), size=(N_CONTACTS, 2), replace=False
    )
    force_constants = np.random.rand(N_CONTACTS)
    
    ref_ff = InvariantForceField(7.0)
    ref_kirchhoff, _ = springcraft.compute_kirchhoff(atoms.coord, ref_ff)
    # Manual shutdown of contacts after Kirchhoff calculation
    atom_i, atom_j = on_indices.T
    ref_kirchhoff[atom_i, atom_j] = -force_constants
    ref_kirchhoff[atom_j, atom_i] = -force_constants
    
    test_ff = springcraft.PatchedForceField(
        ref_ff,
        contact_pair_on=on_indices,
        force_constants=force_constants
    )
    test_kirchhoff, _ = springcraft.compute_kirchhoff(atoms.coord, test_ff)
    
    # Main diagonal is not easily adjusted
    # -> simply set main diagonal of ref and test matrix to 0
    np.fill_diagonal(test_kirchhoff, 0)
    np.fill_diagonal(ref_kirchhoff, 0)
    np.set_printoptions(threshold=10000, linewidth=1000)
    assert np.all(test_kirchhoff == ref_kirchhoff)

    
def test_tabulated_forcefield_homogeneous(atoms):
    """
    Check contents of position-specifc interaction matrix, where the
    interactions are independent of the amino acid.
    Each matrix element is checked in a non-vectorized manner.
    """
    BONDED = 1
    INTRA = 2
    INTER = 3

    ff = springcraft.TabulatedForceField(atoms, BONDED, INTRA, INTER, None)

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


def test_tabulated_forcefield_inhomogeneous(atoms):
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

    ff = springcraft.TabulatedForceField(atoms, bonded, intra, inter, None)

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


def test_tabulated_forcefield_distance(atoms):
    """
    Check whether the calculated force constants are correct for a force
    field with distance dependence, but no amino acid or connection
    dependence.
    """
    N_BINS = 100
    MAX_DISTANCE = 30
    N_SAMPLE_INTERACTIONS = 500

    np.random.seed(0)
    distance_edges = np.sort(np.random.rand(N_BINS) * MAX_DISTANCE)
    # Bin edges must be unique
    assert np.all(np.unique(distance_edges) == distance_edges)
    # All edges should be lower than MAX_DISTANCE to include the bin
    # after the final edge in 'ff.force_constant()' (see below)
    assert np.all(distance_edges < MAX_DISTANCE)

    # Only distance dependent force constants 
    fc = np.arange(N_BINS)
    ff = springcraft.TabulatedForceField(atoms, fc, fc, fc, distance_edges)

    ## Check correctness of interaction matrix
    assert ff.interaction_matrix.shape == (len(atoms), len(atoms), N_BINS)
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
    sample_bin_indices = np.random.randint(N_BINS, size=N_SAMPLE_INTERACTIONS)
    sample_dist = distance_edges[sample_bin_indices]
    test_force_constants = ff.force_constant(atom_i, atom_j, sample_dist**2)
    # Force constants (i.e. 'fc') are designed such that the force is
    # the index of the bin
    ref_force_constans = np.where(
        atom_i != atom_j,
        sample_bin_indices,
        0
    )
    assert np.allclose(test_force_constants, ref_force_constans)


@pytest.mark.parametrize("cutoff_distance", [None, 7])
def test_tabulated_forcefield_cutoff(atoms, cutoff_distance):
    """
    Check whether a :class:`TabulatedForceField` with equal force
    constant for each pair of atoms result in a kirchhoff matrix,
    that simply represents adjacency.
    """
    ff = springcraft.TabulatedForceField(atoms, 1, 1, 1, cutoff_distance)
    kirchhoff, _ = springcraft.compute_kirchhoff(atoms.coord, ff)
    ref_adj_matrix = -kirchhoff
    np.fill_diagonal(ref_adj_matrix, 0)
    assert np.isin(ref_adj_matrix.flatten(), [0,1]).all()
    ref_adj_matrix = ref_adj_matrix.astype(bool)

    if cutoff_distance is None:
        # All atoms are connected except atoms with themselves
        test_adj_matrix = ~np.identity(atoms.array_length(), dtype=bool)
    else:
        # Search pairs of atoms within cutoff distance
        cell_list = struc.CellList(atoms, cutoff_distance)
        test_adj_matrix = cell_list.create_adjacency_matrix(cutoff_distance)
        np.fill_diagonal(test_adj_matrix, False)
    
    assert np.all(test_adj_matrix == ref_adj_matrix)


@pytest.mark.parametrize("shape, n_edges, is_valid", [
    [(),           None, True ],
    [(),           1,    True ],
    [(),           10,   True ],
    [(10,),        None, False],
    [(10,),        1,    False],
    [( 9,),        10,   False],
    [(10,),        10,   True ],
    [( 1,),        None, True ],
    [(20,  1),     1,    False],
    [(20, 30),     1,    False],
    [( 1, 20),     1,    False],
    [(30, 20),     1,    False],
    [(20, 20),     1,    True ],
    [(20, 20),     None, True ],
    [(20, 20),     10,   True ],
    [(20,  1, 10), 10,   False],
    [(20, 30, 10), 10,   False],
    [( 1, 20, 10), 10,   False],
    [(30, 20, 10), 10,   False],
    [(20, 20, 10), 10,   True ],
    [(20, 20,  1), 1,    True ],
    [(20, 20,  1), None, True ],
    [(20, 20, 10), 9,    False],
])
def test_tabulated_forcefield_input_shapes(atoms, shape, n_edges, is_valid):
    """
    Test whether all supported input shapes are handled properly.
    This is chaked based on the calculated interaction matrix.
    Unsupported cases should raise an exception.
    """
    np.random.seed(0)
    fc = np.ones(shape) if shape != () else 1
    edges = np.arange(n_edges) if n_edges is not None else None

    if is_valid:
        ff = springcraft.TabulatedForceField(atoms, fc, fc, fc, edges)
        n_bins = n_edges if n_edges is not None else 1
        assert ff.interaction_matrix.shape == (40, 40, n_bins)
    else:
        with pytest.raises(IndexError):
            ff = springcraft.TabulatedForceField(atoms, fc, fc, fc, edges)


@pytest.mark.parametrize(
    "name",
    ["s_enm_10", "s_enm_13", "d_enm", "sd_enm", "e_anm"]
)
def test_tabulated_forcefield_predefined(atoms, name):
    """
    Test the instantiation of predefined force fields.
    These are implemented as static methods that merely require the
    structure as input.
    """
    meth = getattr(springcraft.TabulatedForceField, name)
    ff = meth(atoms)


def test_parameterfree_forcefield():
    """
    Test whether all entries in the kirchhoff matrix are
    -1 / distance^2.
    """
    N_ATOMS = 5

    np.random.seed(0)
    coord = np.random.rand(N_ATOMS, 3)
    
    dist_matrix = struc.distance(
        coord[np.newaxis, :], coord[:, np.newaxis]
    )
    # Note that the main diagonal is not correct
    ref_kirchhoff = -1 / dist_matrix**2

    ff = springcraft.ParameterFreeForceField()
    test_kirchhoff, _ = springcraft.compute_kirchhoff(coord, ff)
    
    # Ignore main diagonal -> Set main diagonal of both matrices to 0
    np.fill_diagonal(ref_kirchhoff, 0)
    np.fill_diagonal(test_kirchhoff, 0)
    assert np.allclose(test_kirchhoff,  ref_kirchhoff)


@pytest.mark.parametrize("ff_name", ["e_anm", "e_anm_mj", "e_anm_ke"])
def test_compare_with_biophysconnector_heterogenous(atoms_singlechain, ff_name):
    """
    Comparisons between Hessians computed for eANM variants using springcraft
    and BioPhysConnectoR.
    Cut-off: 1.3 nm.
    """

    if ff_name == "e_anm":
        ff = springcraft.TabulatedForceField.e_anm(atoms_singlechain)
        ref_file = "hessian_eANM_BioPhysConnectoR.csv"
    if ff_name == "e_anm_mj":
        ff = springcraft.TabulatedForceField.e_anm_mj(atoms_singlechain)
        ref_file = "hessian_eANM_mj_BioPhysConnectoR.csv"
    if ff_name == "e_anm_ke":
        ff = springcraft.TabulatedForceField.e_anm_ke(atoms_singlechain)
        ref_file = "hessian_eANM_ke_BioPhysConnectoR.csv"
    
    test_hessian, _ = springcraft.compute_hessian(atoms_singlechain.coord, ff)
    
    # Load .csv file data from BiophysConnectoR
    ref_hessian = np.genfromtxt(
        join(data_dir(), ref_file),
        skip_header=1, delimiter=","
    )

    # Higher deviation for eANM_Ke-FF
    if ff_name == "e_anm_ke":
        assert np.allclose(test_hessian, ref_hessian, atol=1e-04)
    else:
        assert np.allclose(test_hessian, ref_hessian)

@pytest.mark.parametrize("ff_name", ["Hinsen", "sdENM", "pfENM"])
def test_compare_with_bio3d(atoms_singlechain, ff_name):
    """
    Comparisons between Hessians computed for ANMs using springcraft
    and Bio3D on different force fields.
    The following ENM forcefields are compared:
    Hinsen-Calpha, sdENM and pfENM.
    """

    if ff_name == "Hinsen":
        ff = springcraft.HinsenForceField()
        ref_file = "hessian_calpha_bio3d.csv"
    if ff_name == "sdENM":
        ff = springcraft.TabulatedForceField.sd_enm(atoms_singlechain)
        ref_file = "hessian_sdenm_bio3d.csv"
    if ff_name == "pfENM":
        ref_file = "hessian_pfenm_bio3d.csv"
        ff = springcraft.ParameterFreeForceField()

    test_hessian, _ = springcraft.compute_hessian(atoms_singlechain.coord, ff)

    ref_hessian = np.genfromtxt(
        join(data_dir(), ref_file),
        skip_header=1, delimiter=","
    )

    # Higher deviation for Hinsen-FF
    if ff_name == "Hinsen":
        assert np.allclose(test_hessian, ref_hessian, atol=1e-04)
    else:
        assert np.allclose(test_hessian, ref_hessian)