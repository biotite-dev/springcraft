import itertools
from multiprocessing.spawn import prepare
from os.path import basename, join
import glob
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import springcraft
from .util import data_dir


def prepare_springcraft_anm(file_path, cutoff):
    pdb_file = pdb.PDBFile.read(file_path)

    atoms = pdb.get_structure(pdb_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]

    ff = springcraft.InvariantForceField(cutoff)
    test_anm = springcraft.ANM(ca, ff)

    return test_anm


@pytest.mark.parametrize("file_path", glob.glob(join(data_dir(), "*.pdb")))
def test_covariance(file_path):
    test_anm = prepare_springcraft_anm(file_path, cutoff=13)
    test_hessian = test_anm.hessian
    test_covariance = test_anm.covariance

    assert np.allclose(
        test_hessian, np.dot(test_hessian, np.dot(test_covariance, test_hessian))
    )
    assert np.allclose(
        test_covariance, np.dot(test_covariance, np.dot(test_hessian, test_covariance))
    )


def test_mass_weights_simple():
    """
    Expect that mass weighting with unit masses does not have any
    influence on an ANM, but different weights do.
    """
    pdb_file = pdb.PDBFile.read(join(data_dir(), "1l2y.pdb"))
    atoms = pdb.get_structure(pdb_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]
    ff = springcraft.InvariantForceField(7.9)

    ref_anm = springcraft.ANM(ca, ff)
    identical_anm = springcraft.ANM(ca, ff, masses=np.ones(ca.array_length()))
    different_anm = springcraft.ANM(
        ca, ff, masses=np.arange(1, ca.array_length() + 1, dtype=float)
    )

    assert np.allclose(identical_anm.hessian, ref_anm.hessian)
    assert not np.allclose(different_anm.hessian, ref_anm.hessian)


@pytest.mark.parametrize("file_path", glob.glob(join(data_dir(), "*.pdb")))
def test_compare_eigenvals_BiophysConnectoR(file_path):
    """
    Compare non-mass-weighted eigenvalues with those computed with
    BiophysConnectoR for eANMs.
    """
    pdb_file = pdb.PDBFile.read(file_path)
    atoms = pdb.get_structure(pdb_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]

    ff = springcraft.TabulatedForceField.e_anm(ca)
    eanm = springcraft.ANM(ca, ff)

    ref_name = basename(file_path).split(".")[0]
    ref_file = f"biophysconnector_anm_eanm_evals_{ref_name}.csv"

    test_eigenval, _ = eanm.eigen()

    # Load .csv.gz file data from BiophysConnectoR
    ref_eigenval = np.genfromtxt(
        join(data_dir(), ref_file), skip_header=1, delimiter=","
    )

    # Omit trivial modes
    assert np.allclose(test_eigenval[6:], ref_eigenval[6:])


@pytest.mark.parametrize(
    "file_path, ff_name",
    itertools.product(
        glob.glob(join(data_dir(), "*.pdb")), ["Hinsen", "sdENM", "pfENM"]
    ),
)
def test_mass_weights_eigenvals(file_path, ff_name):
    """
    Compare mass-weighted eigenvalues with reference values obtained
    with bio3d to test the correctness of the mass-weighting procedure
    and the validity of results obtained with SVD.
    To this end, bio3d-assigned masses are used.
    """
    pdb_file = pdb.PDBFile.read(file_path)
    pdb_name = basename(file_path).split(".")[0]
    atoms = pdb.get_structure(pdb_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]

    if ff_name == "Hinsen":
        ff = springcraft.HinsenForceField()
        ff_bio3d_str = "calpha"
    if ff_name == "sdENM":
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
    if ff_name == "pfENM":
        ff = springcraft.ParameterFreeForceField()
        ff_bio3d_str = "pfanm"

    # ENM-NMA -> Reference
    bio3d_masses_file = f"bio3d_mass_{pdb_name}.csv.gz"
    bio3d_eigvals_file = f"bio3d_anm_{ff_bio3d_str}_ff_evals_mw_{pdb_name}.csv.gz"
    reference_masses = np.genfromtxt(join(data_dir(), bio3d_masses_file), delimiter=",")
    reference_eigenval = np.genfromtxt(
        join(data_dir(), bio3d_eigvals_file), delimiter=","
    )

    test = springcraft.ANM(ca, ff, masses=reference_masses)
    test_eigenval, _ = test.eigen()
    assert np.allclose(
        test_eigenval[6:], reference_eigenval[6:], rtol=5e-03, atol=2e-03
    )


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

    pdb_file = pdb.PDBFile.read(join(data_dir(), "1l2y.pdb"))
    atoms = pdb.get_structure(pdb_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]

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
            print(test_freq[6:])
            print(reference_freq[6:])
            assert np.allclose(
                test_freq[6:], reference_freq[6:], rtol=5e-03, atol=2e-03
            )
            assert np.allclose(test_fluc, reference_fluc, rtol=5e-03, atol=2e-03)
            assert np.allclose(
                test_fluc_subset, reference_fluc_subset, rtol=5e-03, atol=2e-03
            )
            assert np.allclose(test_dcc, reference_dcc, rtol=5e-03, atol=2e-03)
            assert np.allclose(
                test_dcc_subset, reference_dcc_subset, rtol=5e-03, atol=2e-03
            )

        # Compare with alternative method of MSF computation
        assert np.allclose(test_fluc_nomw, msqf_alternative)
