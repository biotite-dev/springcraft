import itertools
from os.path import join
from unittest.mock import patch

import biotite.structure as struc
import biotite.structure.info as strucinfo
import numpy as np
import pytest

import springcraft
from tests.util import data_dir, load_protein_structure, prepare_anm


def test_mass_weights_simple():
    """
    Expect that mass weighting with unit masses does not have any
    influence on an ANM, but different weights do.
    """
    ca = load_protein_structure("1l2y")
    ff = springcraft.InvariantForceField(7.9)

    ref_anm = springcraft.ANM(ca, ff)
    identical_anm = springcraft.ANM(ca, ff, masses=np.ones(ca.array_length()))
    different_anm = springcraft.ANM(
        ca, ff, masses=np.arange(1, ca.array_length() + 1, dtype=float)
    )

    assert np.allclose(identical_anm.hessian, ref_anm.hessian)
    assert not np.allclose(different_anm.hessian, ref_anm.hessian)

    # unit masses
    ref_anm = springcraft.ANM(ca, ff)
    assert ref_anm.masses is None
    identical_anm = springcraft.ANM(ca, ff, masses=np.ones(ca.array_length()))
    assert identical_anm.masses is not None
    assert np.allclose(identical_anm.masses, np.ones(ca.array_length()))
    assert np.allclose(identical_anm.hessian, ref_anm.hessian)

    # arbitrary masses
    residue_weights = np.array(
        [
            strucinfo.mass(res_name, is_residue=True)
            for res_name in ca.res_name  # pyright: ignore[reportOptionalIterable]
        ]
    )
    with pytest.raises(IndexError, match="5 masses for 20 atoms given"):
        springcraft.ANM(ca, ff, masses=residue_weights[:5])
    with pytest.raises(ValueError, match="Masses must not be 0"):
        springcraft.ANM(ca, ff, masses=np.zeros_like(residue_weights))
    different_anm = springcraft.ANM(ca, ff, masses=residue_weights)
    assert not np.allclose(different_anm.hessian, ref_anm.hessian)

    # infer residue weights
    with pytest.raises(
        TypeError, match="An AtomArray is required to automatically infer masses"
    ):
        springcraft.ANM(ca.coord, ff, masses=True)
    residue_weight_anm = springcraft.ANM(ca, ff, masses=True)
    assert np.allclose(different_anm.hessian, residue_weight_anm.hessian)


def test_hessian_covariance_setter():
    """
    Tests that the setter methods check for the correct matrix size and
    that dependend attributes are invalidated
    """
    test_anm = prepare_anm("1l2y", 7)
    test_hessian1 = test_anm.hessian
    test_covariance1 = test_anm.covariance
    test_eig_values1, _ = test_anm.eigen()
    assert test_anm._interactions is not None
    assert test_anm._hessian is not None
    assert test_anm._covariance is not None
    assert test_anm._eig_values is not None
    assert test_anm._eig_vectors is not None

    with pytest.raises(IndexError, match="Expected shape \\(60, 60\\), got \\(5, 5\\)"):
        test_anm.hessian = np.ones((5, 5))
    test_anm.hessian = test_hessian1
    assert test_anm._interactions is not None
    assert test_anm._hessian is not None
    assert test_anm._covariance is None
    assert test_anm._eig_values is None
    assert test_anm._eig_vectors is None

    test_hessian2 = test_anm.hessian
    test_covariance2 = test_anm.covariance
    test_eig_values2, _ = test_anm.eigen()
    assert np.allclose(test_hessian1, test_hessian2)
    assert np.allclose(test_covariance1, test_covariance2)
    assert np.allclose(test_eig_values1, test_eig_values2)

    with pytest.raises(IndexError, match="Expected shape \\(60, 60\\), got \\(5, 5\\)"):
        test_anm.covariance = np.ones((5, 5))
    test_anm.covariance = test_covariance2
    assert test_anm._interactions is None
    assert test_anm._hessian is None
    assert test_anm._covariance is not None
    assert test_anm._eig_values is None
    assert test_anm._eig_vectors is None

    test_hessian3 = test_anm.hessian
    test_covariance3 = test_anm.covariance
    test_eig_values3, _ = test_anm.eigen()
    assert np.allclose(test_hessian2, test_hessian3)
    assert np.allclose(test_covariance2, test_covariance3)
    assert np.allclose(test_eig_values2, test_eig_values3)


@pytest.mark.parametrize(
    "pdb_id, cutoff",
    itertools.product(
        ["1l2y", "10nm"],
        [4, 7, 13],
    ),
)
def test_covariance(pdb_id, cutoff):
    """
    Tests whether the covariance is the pseudo-inverse of the hessian matrix.
    """
    test_anm = prepare_anm(pdb_id, cutoff)
    assert np.allclose(
        test_anm.hessian, test_anm.hessian @ test_anm.covariance @ test_anm.hessian
    )


@pytest.mark.parametrize("pdb_id", ["1l2y", "7cal"])
def test_compare_eigenvals_BiophysConnectoR(pdb_id):
    """
    Compare non-mass-weighted eigenvalues with those computed with
    BiophysConnectoR for eANMs.
    """
    ca = load_protein_structure(pdb_id)

    ff = springcraft.TabulatedForceField.e_anm(ca)
    eanm = springcraft.ANM(ca, ff)

    ref_file = f"biophysconnector_anm_eanm_evals_{pdb_id}.csv"

    test_eigenval, _ = eanm.eigen()

    # Load .csv.gz file data from BiophysConnectoR
    ref_eigenval = np.genfromtxt(
        join(data_dir(), ref_file), skip_header=1, delimiter=","
    )

    # Omit trivial modes
    assert np.allclose(test_eigenval[6:], ref_eigenval[6:])


@pytest.mark.parametrize(
    "pdb_id, ff_name",
    itertools.product(["1l2y", "7cal"], ["Hinsen", "sdENM", "pfENM"]),
)
def test_mass_weights_eigenvals(pdb_id, ff_name):
    """
    Compare mass-weighted eigenvalues with reference values obtained
    with bio3d to test the correctness of the mass-weighting procedure
    and the validity of results obtained with SVD.
    To this end, bio3d-assigned masses are used.
    """
    ca = load_protein_structure(pdb_id)

    if ff_name == "Hinsen":
        ff = springcraft.HinsenForceField()
        ff_bio3d_str = "calpha"
    elif ff_name == "sdENM":
        ff = springcraft.TabulatedForceField.sd_enm(ca)
        ff_bio3d_str = "sdenm"

        # NOTE: Different chains are not correctly identified in bio3d
        # -> Connect single chains with modified covalent contacts
        #    in springcraft
        if struc.get_chain_count(ca) > 1:
            after_chainbreak = struc.check_res_id_continuity(ca)
            prior_chainbreak = after_chainbreak - 1
            contact_mod_pairs = np.array([prior_chainbreak, after_chainbreak]).T
            bonded_force_constant = 43.52 * 0.0083144621 * 300 * 10
            ff = springcraft.PatchedForceField(
                ff,
                contact_pair_off=contact_mod_pairs,
                contact_pair_on=contact_mod_pairs,
                force_constants=np.full(len(contact_mod_pairs), bonded_force_constant),
            )
    elif ff_name == "pfENM":
        ff = springcraft.ParameterFreeForceField()
        ff_bio3d_str = "pfanm"
    else:
        raise ValueError("Unkown ForceField.")

    # ENM-NMA -> Reference
    bio3d_masses_file = f"bio3d_mass_{pdb_id}.csv.gz"
    bio3d_eigvals_file = f"bio3d_anm_{ff_bio3d_str}_ff_evals_mw_{pdb_id}.csv.gz"
    reference_masses = np.genfromtxt(join(data_dir(), bio3d_masses_file), delimiter=",")
    reference_eigenval = np.genfromtxt(
        join(data_dir(), bio3d_eigvals_file), delimiter=","
    )

    test = springcraft.ANM(ca, ff, masses=reference_masses)
    test_eigenval, _ = test.eigen()
    assert np.allclose(
        test_eigenval[6:], reference_eigenval[6:], rtol=5e-03, atol=2e-03
    )


def test_eigen_parameters():
    """
    Tests copies and the number of zero eigenvalues get returned
    depending on the input parameters.
    """
    pdb_id = "1l2y"
    cutoff = 7
    test_anm = prepare_anm(pdb_id, cutoff)

    eig_values1, eig_vectors1 = test_anm.eigen(copy=False, n_zero=False)
    eig_values1[1] = 3
    eig_vectors1[1, 1] = 3
    eig_values2, eig_vectors2 = test_anm.eigen(copy=False, n_zero=False)
    assert np.array_equal(eig_values1, eig_values2)
    assert np.array_equal(eig_vectors1, eig_vectors2)

    test_anm = prepare_anm(pdb_id, cutoff)

    eig_values1, eig_vectors1 = test_anm.eigen(copy=True, n_zero=False)
    eig_values1[1] = 3
    eig_vectors1[1, 1] = 3
    eig_values2, eig_vectors2 = test_anm.eigen(copy=True, n_zero=False)
    assert not np.array_equal(eig_values1, eig_values2)
    assert not np.array_equal(eig_vectors1, eig_vectors2)

    test_anm = prepare_anm(pdb_id, cutoff)

    eig_values1, eig_vectors1, eig_n_zero1 = test_anm.eigen(copy=False, n_zero=True)
    eig_values1[1] = 3
    eig_vectors1[1, 1] = 3
    eig_values2, eig_vectors2, eig_n_zero2 = test_anm.eigen(copy=False, n_zero=True)
    assert np.array_equal(eig_values1, eig_values2)
    assert np.array_equal(eig_vectors1, eig_vectors2)

    test_anm = prepare_anm(pdb_id, cutoff)

    eig_values1, eig_vectors1, eig_n_zero1 = test_anm.eigen(copy=True, n_zero=True)
    eig_values1[1] = 3
    eig_vectors1[1, 1] = 3
    eig_values2, eig_vectors2, eig_n_zero2 = test_anm.eigen(copy=True, n_zero=True)
    assert not np.array_equal(eig_values1, eig_values2)
    assert not np.array_equal(eig_vectors1, eig_vectors2)


@pytest.mark.parametrize(
    "pdb_id, cutoff",
    itertools.product(
        ["1l2y", "104l"],
        [4, 7, 13],
    ),
)
def test_eigen_before_covariance(pdb_id, cutoff):
    """
    Tests that the `Hessian` gets calculated if not present and no
    error is produced.
    Tests that covariance matrix calculation uses stored eigenvalues/-vector
    without calculating them all over again.
    """
    test_anm = prepare_anm(pdb_id, cutoff)

    eig_values, eig_vectors = test_anm.eigen()
    # eigen() should calc the hessian if not present
    ref_hessian = test_anm.hessian.copy()
    for eig_value, eig_vector in zip(eig_values, eig_vectors):
        assert np.allclose(np.matvec(ref_hessian, eig_vector), eig_value * eig_vector)

    with patch("numpy.linalg.eigh") as mock_eigh:
        test_covariance = test_anm.covariance
        mock_eigh.assert_not_called()
    assert np.allclose(ref_hessian, ref_hessian @ test_covariance @ ref_hessian)
    assert np.allclose(test_covariance, test_covariance @ ref_hessian @ test_covariance)

    assert np.allclose(ref_hessian, test_anm.hessian)


@pytest.mark.parametrize(
    "pdb_id, cutoff",
    itertools.product(
        ["1l2y", "104l"],
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
    test_anm = prepare_anm(pdb_id, cutoff)
    ref_hessian = test_anm.hessian.copy()

    test_covariance = test_anm.covariance
    assert np.allclose(ref_hessian, ref_hessian @ test_covariance @ ref_hessian)
    assert np.allclose(test_covariance, test_covariance @ ref_hessian @ test_covariance)

    with patch("numpy.linalg.eigh") as mock_eigh:
        eig_values, eig_vectors = test_anm.eigen()
        mock_eigh.assert_not_called()
    for eig_value, eig_vector in zip(eig_values, eig_vectors):
        assert np.allclose(np.matvec(ref_hessian, eig_vector), eig_value * eig_vector)

    assert np.allclose(ref_hessian, test_anm.hessian)


def test_mean_square_fluctuation():
    """
    Tests whether the mean square fluctuations calculations
    work correctly.
    """
    pdb_id = "1l2y"
    cutoff = 7.0

    # test full set
    test_anm = prepare_anm(pdb_id, cutoff)
    test_anm.hessian
    assert test_anm._covariance is None
    # calc with eigvecs
    msqf_eig_full = test_anm.mean_square_fluctuation()
    test_anm.covariance
    assert test_anm._covariance is not None
    # read covariance
    msqf_cov_full = test_anm.mean_square_fluctuation()
    assert np.allclose(msqf_eig_full, msqf_cov_full)

    # test small subset
    test_anm = prepare_anm(pdb_id, cutoff)
    test_anm.hessian
    with pytest.raises(ValueError, match="Trivial"):
        test_anm.mean_square_fluctuation(mode_subset=np.array([6, 13]))
    test_anm.mean_square_fluctuation(mode_subset=np.array([7, 59]))

    # test temp scaling
    test_anm = prepare_anm(pdb_id, cutoff)
    test_anm.hessian
    assert test_anm._covariance is None
    # calc with eigvecs
    msqf_eig_temp = test_anm.mean_square_fluctuation(tem=300)
    assert np.allclose(msqf_eig_temp, 300 * 1.380649e-23 * msqf_eig_full)
    test_anm.covariance
    assert test_anm._covariance is not None
    # read covariance
    msqf_cov_temp = test_anm.mean_square_fluctuation(tem=300)
    assert np.allclose(msqf_eig_temp, msqf_cov_temp)


def test_bfactor():
    """
    Tests whether the bfactor calculations work correctly.
    """
    pdb_id = "1l2y"
    cutoff = 7.0

    # test full set
    test_anm = prepare_anm(pdb_id, cutoff)
    test_anm.hessian
    assert test_anm._covariance is None
    # calc with eigvecs
    bfactor_eig_full = test_anm.bfactor()
    test_anm.covariance
    assert test_anm._covariance is not None
    # read covariance
    bfactor_cov_full = test_anm.bfactor()
    assert np.allclose(bfactor_eig_full, bfactor_cov_full)

    # test small subset
    test_anm = prepare_anm(pdb_id, cutoff)
    test_anm.hessian
    with pytest.raises(ValueError, match="Trivial"):
        test_anm.bfactor(mode_subset=np.array([6, 13]))
    test_anm.bfactor(mode_subset=np.array([7, 59]))

    # test temp scaling
    test_anm = prepare_anm(pdb_id, cutoff)
    test_anm.hessian
    assert test_anm._covariance is None
    # calc with eigvecs
    bfactor_eig_temp = test_anm.bfactor(tem=300)
    assert np.allclose(bfactor_eig_temp, 300 * 1.380649e-23 * bfactor_eig_full)
    test_anm.covariance
    assert test_anm._covariance is not None
    # read covariance
    bfactor_cov_temp = test_anm.bfactor(tem=300)
    assert np.allclose(bfactor_eig_temp, bfactor_cov_temp)


@pytest.mark.parametrize(
    "ff_name", ["ANM_standard", "Hinsen", "eANM", "sdENM", "pfENM"]
)
def test_frequency_fluctuation_dcc(ff_name):
    """
    Compare quantities commonly computed as part of NMA.
    Prody/BioPhysConnectoR/Bio3d are used as references.
    """
    K_B = 1.380649e-23
    N_A = 6.02214076e23
    tem = 300

    ca = load_protein_structure("1l2y")

    # Prody
    if ff_name == "ANM_standard":
        # No mass or temperature weighting for comparison with Prody
        ff = springcraft.InvariantForceField(13)
        test_anm = springcraft.ANM(ca, ff)
        test_freq_no_mw = test_anm.frequencies()
        test_fluc_nomw = test_anm.mean_square_fluctuation(tem=None)

        test_dcc = test_anm.dcc()
        test_dcc_absolute = test_anm.dcc(norm=False)
        test_dcc_subset = test_anm.dcc(mode_subset=np.arange(6, 36))

        ## Read in Prody reference csv files
        # Evals and fluctuations
        prody_ff_cutoff_name = "anm_13_ang_cutoff"
        prody_evals = np.genfromtxt(
            join(data_dir(), f"prody_{prody_ff_cutoff_name}_evals_1l2y.csv.gz"),
            delimiter=",",
        )
        reference_freq = 1 / (2 * np.pi) * np.sqrt(prody_evals)
        reference_fluc = np.genfromtxt(
            join(data_dir(), f"prody_{prody_ff_cutoff_name}_fluctuations_1l2y.csv.gz"),
            delimiter=",",
        )

        # DCC
        ref_dcc = np.genfromtxt(
            join(data_dir(), f"prody_{prody_ff_cutoff_name}_dcc_norm_1l2y.csv.gz"),
            delimiter=",",
        )

        # Subset: First 30 non-triv. modes
        ref_dcc_norm_subset = np.genfromtxt(
            join(
                data_dir(), f"prody_{prody_ff_cutoff_name}_dcc_norm_subset_1l2y.csv.gz"
            ),
            delimiter=",",
        )
        # Absolute values
        ref_dcc_absolute = np.genfromtxt(
            join(data_dir(), f"prody_{prody_ff_cutoff_name}_dcc_absolute_1l2y.csv.gz"),
            delimiter=",",
        )

        assert np.allclose(test_freq_no_mw[6:], reference_freq[6:])
        assert np.allclose(test_fluc_nomw, reference_fluc)
        assert np.allclose(test_dcc, ref_dcc)
        assert np.allclose(test_dcc_absolute, ref_dcc_absolute)
        assert np.allclose(test_dcc_subset, ref_dcc_norm_subset)

    # References computed with R packages -> read .csv.gz files in "data"
    else:
        # BioPhysConnectoR -> no internal computation of DCCs available;
        #                     no mass-/temperature weighting
        if ff_name == "eANM":
            ff = springcraft.TabulatedForceField.e_anm(ca)
            test_nomw = springcraft.ANM(ca, ff)
            ref_fluc = "biophysconnector_anm_eanm_bfacs_1l2y.csv"
            test_nomw = springcraft.ANM(ca, ff)
            test_fluc_nomw = test_nomw.mean_square_fluctuation()
            # -> For alternative MSF computation method;
            # no temperature weighting
            tem_scaling = 1
            tem = 1

            ## Read in reference file
            reference_fluc = np.genfromtxt(
                join(data_dir(), ref_fluc), skip_header=1, delimiter=","
            )

        # Bio3d -> Mass- and temperature weighting
        else:
            if ff_name == "Hinsen":
                ff = springcraft.HinsenForceField()
                ff_bio3d_str = "calpha"
            elif ff_name == "sdENM":
                ff = springcraft.TabulatedForceField.sd_enm(ca)
                ff_bio3d_str = "sdenm"
            elif ff_name == "pfENM":
                ff = springcraft.ParameterFreeForceField()
                ff_bio3d_str = "pfanm"

            ## Read in reference files
            # Frequencies and fluctuations
            bio3d_masses_file = "bio3d_mass_1l2y.csv.gz"
            reference_masses = np.genfromtxt(
                join(data_dir(), bio3d_masses_file), delimiter=","
            )
            reference_freq = np.genfromtxt(
                join(
                    data_dir(), f"bio3d_anm_{ff_bio3d_str}_ff_frequencies_mw_1l2y.csv"
                ),
                delimiter=",",
            )
            reference_fluc = np.genfromtxt(
                join(
                    data_dir(),
                    f"bio3d_anm_{ff_bio3d_str}_ff_fluctuations_non_mw_1l2y.csv",
                ),
                delimiter=",",
            )
            reference_fluc_subset = np.genfromtxt(
                join(
                    data_dir(),
                    f"bio3d_anm_{ff_bio3d_str}_ff_fluctuations_subset_mw_1l2y.csv",
                ),
                delimiter=",",
            )

            # DCC and DCC subset (first 30 nontriv. modes)
            reference_dcc = np.genfromtxt(
                join(data_dir(), f"bio3d_anm_{ff_bio3d_str}_ff_dcc_mw_1l2y.csv"),
                delimiter=",",
            )
            reference_dcc_subset = np.genfromtxt(
                join(data_dir(), f"bio3d_anm_{ff_bio3d_str}_ff_dcc_subset_mw_1l2y.csv"),
                delimiter=",",
            )

            tem_scaling = K_B * N_A
            test_nomw = springcraft.ANM(ca, ff)
            test_fluc_nomw = test_nomw.mean_square_fluctuation(
                tem=tem, tem_factors=tem_scaling
            )

            test = springcraft.ANM(ca, ff, masses=reference_masses)
            test_freq = test.frequencies()

            ## Scale for consistency with bio3d; T=300 K; no mass weighting
            # Start with mass_weighted eigenvals
            test_fluc = test.mean_square_fluctuation(
                tem=tem, tem_factors=tem_scaling
            ) / (1000 * reference_masses)

            # Select a subset of modes: 12-33
            test_fluc_subset = test.mean_square_fluctuation(
                tem=tem, tem_factors=tem_scaling, mode_subset=np.arange(11, 33)
            )
            test_fluc_subset /= 1000 * reference_masses

            # DCCs
            test_dcc = test.dcc()
            # Only consider the first 30 non-triv. modes
            # Mode 6-36 (conventional enumeration)
            test_dcc_subset = test.dcc(mode_subset=np.arange(6, 36))

        ## No mass-weighting
        # Alternative Method for MSF computation considering all modes
        diag = test_nomw.covariance.diagonal()
        reshape_diag = np.reshape(diag, (len(test_nomw._coord), -1))

        # Compute MSF directly from covariance matrix
        msqf_alternative = np.sum(reshape_diag, axis=1) * tem_scaling * tem

        if ff_name == "eANM":
            assert np.allclose(test_fluc_nomw, reference_fluc)
        # Bio3d-FFs
        else:
            assert np.allclose(
                test_freq[6:], reference_freq[6:], rtol=5e-03, atol=2e-03
            )
            assert np.allclose(test_fluc, reference_fluc, rtol=5e-03, atol=2e-03)
            assert np.allclose(
                test_fluc_subset, reference_fluc_subset, rtol=5e-03, atol=2e-03
            )
            print(test_dcc)
            print(reference_dcc)
            print(np.max(np.abs(test_dcc - reference_dcc)))
            assert np.allclose(test_dcc, reference_dcc, rtol=5e-03, atol=2e-03)
            assert np.allclose(
                test_dcc_subset, reference_dcc_subset, rtol=5e-03, atol=2e-03
            )

        # Compare with alternative method of MSF computation
        assert np.allclose(test_fluc_nomw, msqf_alternative)


@pytest.mark.parametrize(
    "pdb_id",
    ["1l2y", "7cal"],
)
def test_prs(pdb_id):
    """
    Compare perturbation response scanning (PRS)
    results with those obtained with ProDy.
    """
    test_anm = prepare_anm(pdb_id, cutoff=13)

    test_prs, test_eff, test_sens = test_anm.prs_effector_sensor()
    ref_prs, ref_eff, ref_sens = [
        np.genfromtxt(
            join(data_dir(), f"prody_anm_13_ang_cutoff_{prs_type}_{pdb_id}.csv.gz"),
            delimiter=",",
        )
        for prs_type in ["prs_mat", "prs_eff", "prs_sens"]
    ]

    assert np.allclose(test_prs, ref_prs)
    assert np.allclose(test_eff, ref_eff)
    assert np.allclose(test_sens, ref_sens)
