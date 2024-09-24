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

    test_eig_values, test_eig_vectors = test_gnm.eigen()

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
