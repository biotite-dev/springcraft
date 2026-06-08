import itertools
from os.path import join
from unittest.mock import patch

import biotite.structure.info as strucinfo
import numpy as np
import pytest

import springcraft
from tests.util import data_dir, load_protein_structure, prepare_gnm


@pytest.mark.parametrize(
    "pdb_id, cutoff",
    itertools.product(["1l2y"], [4, 7, 13]),
)
def test_kirchhoff(pdb_id, cutoff):
    """
    Compare computed Kirchhoff matrix with output from *ProDy* with
    test files.
    """
    test_gnm = prepare_gnm(pdb_id, cutoff)
    ref_kirchhoff = np.genfromtxt(
        join(data_dir(), f"prody_gnm_{cutoff}_ang_cutoff_kirchhoff_{pdb_id}.csv.gz"),
        delimiter=",",
    )

    assert test_gnm.kirchhoff.flatten().tolist() == pytest.approx(
        ref_kirchhoff.flatten().tolist()
    )


def test_mass_weights_simple():
    """
    Expect that mass weighting with unit masses does not have any
    influence on an GNM, but different weights do.
    Expect that supplying TRUE infers residue weights.
    """
    ca = load_protein_structure("1l2y")
    ff = springcraft.InvariantForceField(7.9)

    # unit masses
    ref_gnm = springcraft.GNM(ca, ff)
    assert ref_gnm.masses is None
    identical_gnm = springcraft.GNM(ca, ff, masses=np.ones(ca.array_length()))
    assert identical_gnm.masses is not None
    assert np.allclose(identical_gnm.masses, np.ones(ca.array_length()))
    assert np.allclose(identical_gnm.kirchhoff, ref_gnm.kirchhoff)

    # arbitrary masses
    residue_weights = np.array(
        [strucinfo.mass(res_name, is_residue=True) for res_name in ca.res_name]
    )
    with pytest.raises(IndexError, match="5 masses for 20 atoms given"):
        springcraft.GNM(ca, ff, masses=residue_weights[:5])
    with pytest.raises(ValueError, match="Masses must not be 0"):
        springcraft.GNM(ca, ff, masses=np.zeros_like(residue_weights))
    different_gnm = springcraft.GNM(ca, ff, masses=residue_weights)
    assert not np.allclose(different_gnm.kirchhoff, ref_gnm.kirchhoff)

    # infer residue weights
    with pytest.raises(
        TypeError, match="An AtomArray is required to automatically infer masses"
    ):
        springcraft.GNM(ca.coord, ff, masses=True)
    residue_weight_gnm = springcraft.GNM(ca, ff, masses=True)
    assert np.allclose(different_gnm.kirchhoff, residue_weight_gnm.kirchhoff)


def test_kirchhoff_covariance_setter():
    """
    Tests that the setter methods check for the correct matrix size and
    that dependend attributes are invalidated
    """
    test_gnm = prepare_gnm("1l2y", 7)
    test_kirchhoff1 = test_gnm.kirchhoff
    test_covariance1 = test_gnm.covariance
    test_eig_values1, _ = test_gnm.eigen()
    assert test_gnm._interactions is not None
    assert test_gnm._kirchhoff is not None
    assert test_gnm._covariance is not None
    assert test_gnm._eig_values is not None
    assert test_gnm._eig_vectors is not None

    with pytest.raises(ValueError, match="Expected shape \\(20, 20\\), got \\(5, 5\\)"):
        test_gnm.kirchhoff = np.ones((5, 5))
    test_gnm.kirchhoff = test_kirchhoff1
    assert test_gnm._interactions is not None
    assert test_gnm._kirchhoff is not None
    assert test_gnm._covariance is None
    assert test_gnm._eig_values is None
    assert test_gnm._eig_vectors is None

    test_kirchhoff2 = test_gnm.kirchhoff
    test_covariance2 = test_gnm.covariance
    test_eig_values2, _ = test_gnm.eigen()
    assert np.allclose(test_kirchhoff1, test_kirchhoff2)
    assert np.allclose(test_covariance1, test_covariance2)
    assert np.allclose(test_eig_values1, test_eig_values2)

    with pytest.raises(IndexError, match="Expected shape \\(20, 20\\), got \\(5, 5\\)"):
        test_gnm.covariance = np.ones((5, 5))
    test_gnm.covariance = test_covariance2
    assert test_gnm._interactions is None
    assert test_gnm._kirchhoff is None
    assert test_gnm._covariance is not None
    assert test_gnm._eig_values is None
    assert test_gnm._eig_vectors is None

    test_kirchhoff3 = test_gnm.kirchhoff
    test_covariance3 = test_gnm.covariance
    test_eig_values3, _ = test_gnm.eigen()
    assert np.allclose(test_kirchhoff2, test_kirchhoff3)
    assert np.allclose(test_covariance2, test_covariance3)
    assert np.allclose(test_eig_values2, test_eig_values3)


@pytest.mark.parametrize(
    "pdb_id, cutoff",
    itertools.product(
        ["1l2y", "104l", "10nm"],
        [4, 7, 13],
    ),
)
def test_covariance(pdb_id, cutoff):
    """
    Tests whether the covariance is the pseudo-inverse of the kirchhoff matrix.
    """
    test_anm = prepare_gnm(pdb_id, cutoff)
    assert np.allclose(
        test_anm.kirchhoff,
        test_anm.kirchhoff @ test_anm.covariance @ test_anm.kirchhoff,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "pdb_id, cutoff",
    itertools.product(
        ["1l2y"],
        # Cutoff must not be too large,
        # otherwise degenerate eigenvalues appear
        [4, 7],
    ),
)
def test_eigen(pdb_id, cutoff):
    """
    Compare computed eigenvalues and -vectors with output from *ProDy*
    with test files.
    """
    test_gnm = prepare_gnm(pdb_id, cutoff)

    test_eig_values, test_eig_vectors = test_gnm.eigen()

    ref_eig_values = np.genfromtxt(
        join(data_dir(), f"prody_gnm_{cutoff}_ang_cutoff_evals_{pdb_id}.csv.gz"),
        delimiter=",",
    )
    ref_eig_vectors = np.genfromtxt(
        join(data_dir(), f"prody_gnm_{cutoff}_ang_cutoff_evecs_{pdb_id}.csv.gz"),
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


def test_eigen_parameters():
    """
    Tests copies and the number of zero eigenvalues get returned
    depending on the input parameters.
    """
    pdb_id = "1l2y"
    cutoff = 7
    test_gnm = prepare_gnm(pdb_id, cutoff)

    eig_values1, eig_vectors1 = test_gnm.eigen(copy=False, n_zero=False)
    eig_values1[1] = 3
    eig_vectors1[1, 1] = 3
    eig_values2, eig_vectors2 = test_gnm.eigen(copy=False, n_zero=False)
    assert np.array_equal(eig_values1, eig_values2)
    assert np.array_equal(eig_vectors1, eig_vectors2)

    test_gnm = prepare_gnm(pdb_id, cutoff)

    eig_values1, eig_vectors1 = test_gnm.eigen(copy=True, n_zero=False)
    eig_values1[1] = 3
    eig_vectors1[1, 1] = 3
    eig_values2, eig_vectors2 = test_gnm.eigen(copy=True, n_zero=False)
    assert not np.array_equal(eig_values1, eig_values2)
    assert not np.array_equal(eig_vectors1, eig_vectors2)

    test_gnm = prepare_gnm(pdb_id, cutoff)

    eig_values1, eig_vectors1, eig_n_zero1 = test_gnm.eigen(copy=False, n_zero=True)
    eig_values1[1] = 3
    eig_vectors1[1, 1] = 3
    eig_values2, eig_vectors2, eig_n_zero2 = test_gnm.eigen(copy=False, n_zero=True)
    assert np.array_equal(eig_values1, eig_values2)
    assert np.array_equal(eig_vectors1, eig_vectors2)
    assert eig_n_zero1 == eig_n_zero2

    test_gnm = prepare_gnm(pdb_id, cutoff)

    eig_values1, eig_vectors1, eig_n_zero1 = test_gnm.eigen(copy=True, n_zero=True)
    eig_values1[1] = 3
    eig_vectors1[1, 1] = 3
    eig_values2, eig_vectors2, eig_n_zero2 = test_gnm.eigen(copy=True, n_zero=True)
    assert not np.array_equal(eig_values1, eig_values2)
    assert not np.array_equal(eig_vectors1, eig_vectors2)
    assert eig_n_zero1 == eig_n_zero2


@pytest.mark.parametrize(
    "pdb_id, cutoff",
    itertools.product(
        ["1l2y", "104l", "10nm"],
        [4, 7, 13],
    ),
)
def test_eigen_before_covariance(pdb_id, cutoff):
    """
    Tests that the `Kirchhoff` gets calculated if not present and no
    error is produced.
    Tests that covariance matrix calculation uses stored eigenvalues/-vector
    without calculating them all over again.
    """
    test_gnm = prepare_gnm(pdb_id, cutoff)

    eig_values, eig_vectors = test_gnm.eigen()
    # eigen() should calc the kirchhoff if not present
    ref_kirchhoff = test_gnm.kirchhoff.copy()
    for eig_value, eig_vector in zip(eig_values, eig_vectors):
        assert np.allclose(np.matvec(ref_kirchhoff, eig_vector), eig_value * eig_vector)

    with patch("numpy.linalg.eigh") as mock_eigh:
        test_covariance = test_gnm.covariance
        mock_eigh.assert_not_called()
    assert np.allclose(ref_kirchhoff, ref_kirchhoff @ test_covariance @ ref_kirchhoff)
    assert np.allclose(
        test_covariance, test_covariance @ ref_kirchhoff @ test_covariance
    )

    assert np.allclose(ref_kirchhoff, test_gnm.kirchhoff)


@pytest.mark.parametrize(
    "pdb_id, cutoff",
    itertools.product(
        ["1l2y", "104l", "10nm"],
        [4, 7, 13],
    ),
)
def test_eigen_after_covariance(pdb_id, cutoff):
    """
    Tests that calculating the covariance matrix works correctly
    and that in the process the eigenvalues/-vectors are stored
    so that they do not have to be recalculated again when accessing
    them afterwards.
    """
    test_gnm = prepare_gnm(pdb_id, cutoff)
    ref_kirchhoff = test_gnm.kirchhoff.copy()

    test_covariance = test_gnm.covariance
    assert np.allclose(ref_kirchhoff, ref_kirchhoff @ test_covariance @ ref_kirchhoff)
    assert np.allclose(
        test_covariance, test_covariance @ ref_kirchhoff @ test_covariance
    )

    with patch("numpy.linalg.eigh") as mock_eigh:
        eig_values, eig_vectors = test_gnm.eigen()
        mock_eigh.assert_not_called()
    for eig_value, eig_vector in zip(eig_values, eig_vectors):
        assert np.allclose(np.matvec(ref_kirchhoff, eig_vector), eig_value * eig_vector)

    assert np.allclose(ref_kirchhoff, test_gnm.kirchhoff)


def test_mean_square_fluctuation():
    """
    Tests whether the mean square fluctuations calculations
    work correctly.
    """
    pdb_id = "1l2y"
    cutoff = 7.0

    # test full set
    test_gnm = prepare_gnm(pdb_id, cutoff)
    test_gnm.kirchhoff
    assert test_gnm._covariance is None
    # calc with eigvecs
    msqf_eig_full = test_gnm.mean_square_fluctuation()
    test_gnm.covariance
    assert test_gnm._covariance is not None
    # read covariance
    msqf_cov_full = test_gnm.mean_square_fluctuation()
    assert np.allclose(msqf_eig_full, msqf_cov_full)

    # test small subset
    test_gnm = prepare_gnm(pdb_id, cutoff)
    test_gnm.kirchhoff
    with pytest.raises(ValueError, match="Trivial"):
        test_gnm.mean_square_fluctuation(mode_subset=np.array([0, 13]))
    test_gnm.mean_square_fluctuation(mode_subset=np.array([1, 19]))

    # test temp scaling
    test_gnm = prepare_gnm(pdb_id, cutoff)
    test_gnm.kirchhoff
    assert test_gnm._covariance is None
    # calc with eigvecs
    msqf_eig_temp = test_gnm.mean_square_fluctuation(tem=300)
    assert np.allclose(msqf_eig_temp, 300 * 1.380649e-23 * msqf_eig_full)
    test_gnm.covariance
    assert test_gnm._covariance is not None
    # read covariance
    msqf_cov_temp = test_gnm.mean_square_fluctuation(tem=300)
    assert np.allclose(msqf_eig_temp, msqf_cov_temp)


def test_bfactor():
    """
    Tests whether the bfactor calculations work correctly.
    """
    pdb_id = "1l2y"
    cutoff = 7.0

    # test full set
    test_gnm = prepare_gnm(pdb_id, cutoff)
    test_gnm.kirchhoff
    assert test_gnm._covariance is None
    # calc with eigvecs
    bfactor_eig_full = test_gnm.bfactor()
    test_gnm.covariance
    assert test_gnm._covariance is not None
    # read covariance
    bfactor_cov_full = test_gnm.bfactor()
    assert np.allclose(bfactor_eig_full, bfactor_cov_full)

    # test small subset
    test_gnm = prepare_gnm(pdb_id, cutoff)
    test_gnm.kirchhoff
    with pytest.raises(ValueError, match="Trivial"):
        test_gnm.bfactor(mode_subset=np.array([0, 13]))
    test_gnm.bfactor(mode_subset=np.array([1, 19]))

    # test temp scaling
    test_gnm = prepare_gnm(pdb_id, cutoff)
    test_gnm.kirchhoff
    assert test_gnm._covariance is None
    # calc with eigvecs
    bfactor_eig_temp = test_gnm.bfactor(tem=300)
    assert np.allclose(bfactor_eig_temp, 300 * 1.380649e-23 * bfactor_eig_full)
    test_gnm.covariance
    assert test_gnm._covariance is not None
    # read covariance
    bfactor_cov_temp = test_gnm.bfactor(tem=300)
    assert np.allclose(bfactor_eig_temp, bfactor_cov_temp)


@pytest.mark.parametrize("pdb_id, cutoff", itertools.product(["1l2y"], [4, 7]))
def test_fluctuation_dcc(pdb_id, cutoff):
    """
    Comparison of mean-square fluctuations and
    dynamic cross-correlations computed with Springcraft and Prody.
    """
    test_gnm = prepare_gnm(pdb_id, cutoff)
    test_fluc = test_gnm.mean_square_fluctuation()
    test_dcc = test_gnm.dcc()
    test_dcc_absolute = test_gnm.dcc(norm=False)
    test_dcc_subset = test_gnm.dcc(mode_subset=np.arange(1, 17))

    reference_fluc = np.genfromtxt(
        join(data_dir(), f"prody_gnm_{cutoff}_ang_cutoff_fluctuations_{pdb_id}.csv.gz"),
        delimiter=",",
    )
    reference_dcc = np.genfromtxt(
        join(data_dir(), f"prody_gnm_{cutoff}_ang_cutoff_dcc_norm_{pdb_id}.csv.gz"),
        delimiter=",",
    )
    reference_dcc_norm_subset = np.genfromtxt(
        join(
            data_dir(),
            f"prody_gnm_{cutoff}_ang_cutoff_dcc_norm_subset_{pdb_id}.csv.gz",
        ),
        delimiter=",",
    )
    reference_dcc_absolute = np.genfromtxt(
        join(data_dir(), f"prody_gnm_{cutoff}_ang_cutoff_dcc_absolute_{pdb_id}.csv.gz"),
        delimiter=",",
    )

    assert np.allclose(test_fluc, reference_fluc)
    assert np.allclose(test_dcc, reference_dcc)
    assert np.allclose(test_dcc_subset, reference_dcc_norm_subset)
    assert np.allclose(test_dcc_absolute, reference_dcc_absolute)
