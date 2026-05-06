import itertools
from os.path import basename, join

import biotite.structure.io.pdb as pdb
import numpy as np
import pytest

import springcraft

from .util import data_dir


def prepare_gnm(file_path, cutoff):
    pdb_file = pdb.PDBFile.read(file_path)
    atoms = pdb.get_structure(pdb_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]

    ff = springcraft.InvariantForceField(cutoff)
    test_gnm = springcraft.GNM(ca, ff)
    test_gnm.kirchhoff

    return test_gnm


@pytest.mark.parametrize(
    "file_path, cutoff",
    itertools.product([join(data_dir(), "1l2y.pdb")], [4, 7, 13]),
)
def test_kirchhoff(file_path, cutoff):
    """
    Compare computed Kirchhoff matrix with output from *ProDy* with
    test files.
    """
    print(file_path)
    test_gnm = prepare_gnm(file_path, cutoff)
    pdb_name = basename(file_path).split(".")[0]
    ref_kirchhoff = np.genfromtxt(
        join(data_dir(), f"prody_gnm_{cutoff}_ang_cutoff_kirchhoff_{pdb_name}.csv.gz"),
        delimiter=",",
    )

    print(test_gnm.kirchhoff)
    print(ref_kirchhoff)
    assert test_gnm.kirchhoff.flatten().tolist() == pytest.approx(
        ref_kirchhoff.flatten().tolist()
    )


@pytest.mark.parametrize(
    "file_path, cutoff",
    itertools.product(
        [join(data_dir(), "1l2y.pdb")],
        # Cutoff must not be too large,
        # otherwise degenerate eigenvalues appear
        [4, 7],
    ),
)
def test_eigen(file_path, cutoff):
    """
    Compare computed eigenvalues and -vectors with output from *ProDy*
    with test files.
    """
    test_gnm = prepare_gnm(file_path, cutoff)

    test_eig_values, test_eig_vectors, _ = test_gnm.eigen()

    pdb_name = basename(file_path).split(".")[0]

    ref_eig_values = np.genfromtxt(
        join(data_dir(), f"prody_gnm_{cutoff}_ang_cutoff_evals_{pdb_name}.csv.gz"),
        delimiter=",",
    )
    ref_eig_vectors = np.genfromtxt(
        join(data_dir(), f"prody_gnm_{cutoff}_ang_cutoff_evecs_{pdb_name}.csv.gz"),
        delimiter=",",
    )

    # Adapt sign of eigenvectors # TODO Is this correct?
    test_eig_vectors *= np.sign(test_eig_vectors[:, 0])[:, np.newaxis]
    ref_eig_vectors *= np.sign(ref_eig_vectors[:, 0])[:, np.newaxis]

    assert np.allclose(test_eig_values[1:], ref_eig_values[1:])
    assert test_eig_values[1:].tolist() == pytest.approx(ref_eig_values[1:].tolist())
    assert test_eig_vectors[1:].flatten().tolist() == pytest.approx(
        ref_eig_vectors[1:].flatten().tolist()
    )


def test_mass_weights_simple():
    """
    Expect that mass weighting with unit masses does not have any
    influence on an GNM, but different weights do.
    """
    pdb_file = pdb.PDBFile.read(join(data_dir(), "1l2y.pdb"))
    atoms = pdb.get_structure(pdb_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]
    ff = springcraft.InvariantForceField(7.9)

    ref_gnm = springcraft.GNM(ca, ff)
    identical_gnm = springcraft.GNM(ca, ff, masses=np.ones(ca.array_length()))
    different_gnm = springcraft.GNM(
        ca, ff, masses=np.arange(1, ca.array_length() + 1, dtype=float)
    )

    assert np.allclose(identical_gnm.kirchhoff, ref_gnm.kirchhoff)
    assert not np.allclose(different_gnm.kirchhoff, ref_gnm.kirchhoff)


@pytest.mark.parametrize(
    "file_path, cutoff", itertools.product([join(data_dir(), "1l2y.pdb")], [4, 7])
)
def test_fluctuation_dcc(file_path, cutoff):
    """
    Comparison of mean-square fluctuations and
    dynamic cross-correlations computed with Springcraft and Prody.
    """
    test_gnm = prepare_gnm(file_path, cutoff)
    test_fluc = test_gnm.mean_square_fluctuation()
    test_dcc = test_gnm.dcc()
    test_dcc_absolute = test_gnm.dcc(norm=False)
    test_dcc_subset = test_gnm.dcc(mode_subset=np.arange(1, 17))

    pdb_name = basename(file_path).split(".")[0]

    reference_fluc = np.genfromtxt(
        join(
            data_dir(), f"prody_gnm_{cutoff}_ang_cutoff_fluctuations_{pdb_name}.csv.gz"
        ),
        delimiter=",",
    )
    reference_dcc = np.genfromtxt(
        join(data_dir(), f"prody_gnm_{cutoff}_ang_cutoff_dcc_norm_{pdb_name}.csv.gz"),
        delimiter=",",
    )
    reference_dcc_norm_subset = np.genfromtxt(
        join(
            data_dir(),
            f"prody_gnm_{cutoff}_ang_cutoff_dcc_norm_subset_{pdb_name}.csv.gz",
        ),
        delimiter=",",
    )
    reference_dcc_absolute = np.genfromtxt(
        join(
            data_dir(), f"prody_gnm_{cutoff}_ang_cutoff_dcc_absolute_{pdb_name}.csv.gz"
        ),
        delimiter=",",
    )

    print(test_dcc_subset.shape)
    print(reference_dcc_norm_subset.shape)
    assert np.allclose(test_fluc, reference_fluc)
    assert np.allclose(test_dcc, reference_dcc)
    assert np.allclose(test_dcc_subset, reference_dcc_norm_subset)
    assert np.allclose(test_dcc_absolute, reference_dcc_absolute)


@pytest.mark.parametrize("nb_of_changes", [1, 3, 20, 27])
def test_modify_contact_pair(nb_of_changes):
    """
    Tests whether permutations to the `kirchhoff` matrix are
    performed correctly and the resulting permutations to the
    `covariance` matrix are correct.

    TODO integrate numerical tests here?

    Parameters
    ----------
    nb_of_changes : int
        number of permutations to perform
    """
    pdb_file = pdb.PDBFile.read(join(data_dir(), "1l2y.pdb"))
    atoms = pdb.get_structure(pdb_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]
    ff = springcraft.InvariantForceField(7.9)

    gnm = springcraft.GNM(ca, ff)
    ref_kirchhoff = gnm.kirchhoff.copy()
    gnm.covariance

    rng = np.random.default_rng(1)
    atom_i, atom_j = np.zeros((2, nb_of_changes), dtype=int)
    delta = (rng.random(nb_of_changes) + 0.5) * rng.choice((-1, 1), nb_of_changes)
    for k in range(nb_of_changes):
        atom_i[k], atom_j[k] = rng.choice(len(ca), size=2, replace=False)

        ref_kirchhoff[atom_i[k], atom_j[k]] += delta[k]
        ref_kirchhoff[atom_j[k], atom_i[k]] += delta[k]
        ref_kirchhoff[atom_i[k], atom_i[k]] -= delta[k]
        ref_kirchhoff[atom_j[k], atom_j[k]] -= delta[k]

    ref_gnm = springcraft.GNM(ca, ff)
    ref_gnm.kirchhoff = ref_kirchhoff
    ref_covariance = ref_gnm.covariance

    gnm._modify_contact_pair(atom_i, atom_j, delta)
    mod_kirchhoff = gnm.kirchhoff
    mod_covariance = gnm.covariance

    assert np.allclose(mod_kirchhoff, ref_kirchhoff)
    print(np.max(np.abs(mod_covariance - ref_covariance)))
    assert np.allclose(mod_covariance, ref_covariance)


@pytest.mark.parametrize(
    "atom_i, atom_j, delta, expected_kirchhoff_change",
    [
        (1, 4, 0.3, 0.3),
        ([1, 8, 11], [4, 6, 12], [0.3, -1.1, 1e-9], [0.3, -1.1, 0]),
        ([1, 11], [4, 12], 0.3, [0.3, 0.3]),
        ([1, 3, 8], [4, 7, 11], [False, True, True], [6.616000175476074, 2.4, 0]),
        ([1, 11], [4, 12], False, [6.616000175476074, 46.83000183105469]),
        ([3, 15], [7, 18], True, [2.4, -11.7]),
    ],
)
def test_modify_contacts(atom_i, atom_j, delta, expected_kirchhoff_change):
    """
    Tests the wrapper function for correct argument conversion.
    """
    pdb_file = pdb.PDBFile.read(join(data_dir(), "1l2y.pdb"))
    atoms = pdb.get_structure(pdb_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]
    ff = springcraft.TabulatedForceField.d_enm(ca)

    gnm = springcraft.GNM(ca, ff)
    ref_kirchhoff = gnm.kirchhoff
    ref_kirchhoff[3, 7] -= 2.4
    ref_kirchhoff[15, 18] += 11.7
    ref_kirchhoff[8, 11] += 1e-9
    ref_kirchhoff = ref_kirchhoff.copy()
    gnm.covariance

    if type(atom_i) is list:
        for i, j, d in zip(atom_i, atom_j, expected_kirchhoff_change):
            ref_kirchhoff[i, j] += d
            ref_kirchhoff[j, i] += d
            ref_kirchhoff[i, i] -= d
            ref_kirchhoff[j, j] -= d
    else:
        ref_kirchhoff[atom_i, atom_j] += expected_kirchhoff_change
        ref_kirchhoff[atom_j, atom_i] += expected_kirchhoff_change
        ref_kirchhoff[atom_i, atom_i] -= expected_kirchhoff_change
        ref_kirchhoff[atom_j, atom_j] -= expected_kirchhoff_change

    ref_gnm = springcraft.GNM(ca, ff)
    ref_gnm.kirchhoff = ref_kirchhoff
    ref_covariance = ref_gnm.covariance

    gnm.modify_contacts(atom_i, atom_j, delta)
    mod_kirchhoff = gnm.kirchhoff
    mod_covariance = gnm.covariance

    assert np.allclose(mod_kirchhoff, ref_kirchhoff)
    assert np.allclose(mod_covariance, ref_covariance)


def test_modify_contacts_checks():
    """
    Tests the wrapper function checks the input arguments correctly.
    """
    pdb_file = pdb.PDBFile.read(join(data_dir(), "1l2y.pdb"))
    atoms = pdb.get_structure(pdb_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]
    ff = springcraft.InvariantForceField(7.0)

    gnm = springcraft.GNM(ca, ff)
    with pytest.raises(ValueError, match="Kirchhoff matrix must exist"):
        gnm.modify_contacts([1, 8, 11], [4, 6, 12], [0.3, -1.1, 1e-9])

    gnm.kirchhoff
    gnm.covariance
    with pytest.raises(ValueError, match="atom index arrays to have the same size"):
        gnm.modify_contacts([1, 8], [4, 6, 12], [0.3, -1.1, 1e-9])
    with pytest.raises(IndexError, match="Expected array indices to be different"):
        gnm.modify_contacts([1, 8, 11], [4, 8, 11], [0.3, -1.1, 1e-9])
    with pytest.raises(IndexError, match="Index out of bounds"):
        gnm.modify_contacts([-1, 8], [4, 6], [0.3, -1.1])
    with pytest.raises(IndexError, match="Index out of bounds"):
        gnm.modify_contacts([1, 8], [20, 6], [0.3, -1.1])
    with pytest.raises(IndexError, match="Index out of bounds"):
        gnm.modify_contacts([1, -1], [4, 6], [0.3, -1.1])
    with pytest.raises(IndexError, match="Index out of bounds"):
        gnm.modify_contacts([1, 8], [4, 20], [0.3, -1.1])
    with pytest.raises(ValueError, match=r"1 delta .* or as many as updates"):
        gnm.modify_contacts([1, 8, 11], [4, 6, 12], [0.3, -1.1])
    with pytest.raises(ValueError, match="invalid literal for int"):
        gnm.modify_contacts(["a", 8, 11], [4, 6, 12], [0.3, -1.1, 1e-9])
    with pytest.raises(ValueError, match="invalid literal for int"):
        gnm.modify_contacts([1, 8, 11], ["e", 6, 12], [0.3, -1.1, 1e-9])
    with pytest.raises(TypeError, match="Expected delta to be float or bool"):
        gnm.modify_contacts([1, 8, 11], [4, 6, 12], ["a", -1.1, 1e-9])


def test_modify_atom():
    """
    Tests the wrapper function for correct argument conversion.
    """
    pdb_file = pdb.PDBFile.read(join(data_dir(), "1l2y.pdb"))
    atoms = pdb.get_structure(pdb_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]
    ff = springcraft.TabulatedForceField.e_anm(ca)
    # ff = springcraft.InvariantForceField(7.0)

    gnm = springcraft.GNM(ca, ff)
    orig_kirchhoff = gnm.kirchhoff.copy()
    orig_covariance = gnm.covariance.copy()

    # turn off contact
    gnm.modify_atom(7, False)
    assert np.allclose(gnm.kirchhoff[7, :], np.zeros(len(ca)))

    ref_gnm = springcraft.GNM(ca, ff)
    ref_gnm.kirchhoff = gnm.kirchhoff.copy()
    ref_covariance = ref_gnm.covariance
    print(np.max(np.abs(gnm.covariance - ref_covariance)))
    assert np.allclose(gnm.covariance, ref_covariance)

    # turn on contact
    gnm.modify_atom(7, True)
    assert np.allclose(gnm.kirchhoff, orig_kirchhoff)
    assert np.allclose(gnm.covariance, orig_covariance)

    # change amino acid type
    ca.res_name[7] = "GLN"
    gnm.modify_atom(7, ca[7])
    ff = springcraft.TabulatedForceField.e_anm(ca)
    ref_gnm = springcraft.GNM(ca, ff)
    ref_gnm.kirchhoff = gnm.kirchhoff.copy()
    ref_covariance = ref_gnm.covariance
    assert np.allclose(gnm.covariance, ref_covariance)
