import itertools
from multiprocessing.spawn import prepare
from os.path import join
import glob
import numpy as np
import pytest
import prody
import biotite.structure.io.mmtf as mmtf
import springcraft
from .util import data_dir


def prepare_anms(file_path, cutoff):
    mmtf_file = mmtf.MMTFFile.read(file_path)

    atoms = mmtf.get_structure(mmtf_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]

    ff = springcraft.InvariantForceField(cutoff)
    test_anm = springcraft.ANM(ca, ff)
    
    ref_anm = prody.ANM()
    ref_anm.buildHessian(ca.coord, gamma=1.0, cutoff=13)

    return test_anm, ref_anm

@pytest.mark.parametrize("file_path",
        glob.glob(join(data_dir(), "*.mmtf"))
)
def test_covariance(file_path):
    test_anm, _ = prepare_anms(file_path, cutoff=13)
    test_hessian = test_anm.hessian
    test_covariance = test_anm.covariance

    assert np.allclose(
        test_hessian,
        np.dot(test_hessian, np.dot(test_covariance, test_hessian))
    )
    assert np.allclose(
        test_covariance,
        np.dot(test_covariance, np.dot(test_hessian, test_covariance))
    )

def test_mass_weights_simple():
    """
    Expect that mass weighting with unit masses does not have any
    influence on an ANM, but different weights do.
    """
    mmtf_file = mmtf.MMTFFile.read(join(data_dir(), "1l2y.mmtf"))
    atoms = mmtf.get_structure(mmtf_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]
    ff = springcraft.InvariantForceField(7.9)

    ref_anm = springcraft.ANM(atoms, ff)
    identical_anm = springcraft.ANM(
        atoms, ff, masses=np.ones(atoms.array_length())
    )
    different_anm = springcraft.ANM(
        atoms, ff, masses=np.arange(1, atoms.array_length() + 1, dtype=float)
    )

    assert np.allclose(identical_anm.hessian, ref_anm.hessian)
    assert not np.allclose(different_anm.hessian, ref_anm.hessian)

def test_compare_eigenvals_BiophysConnectoR():
    """
    Compare non-mass-weighted eigenvalues with those computed with 
    BiophysConnectoR for eANMs.
    """
    mmtf_file = mmtf.MMTFFile.read(join(data_dir(), "1l2y.mmtf"))
    atoms = mmtf.get_structure(mmtf_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]

    ff = springcraft.TabulatedForceField.e_anm(ca)
    eanm = springcraft.ANM(ca, ff)

    ref_file = "eigenval_eANM_BioPhysConnectoR.csv"

    test_eigenval, _ = eanm.eigen()

    # Load .csv file data from BiophysConnectoR
    ref_eigenval = np.genfromtxt(
        join(data_dir(), ref_file),
        skip_header=1, delimiter=","
    )

    # Omit trivial modes
    assert np.allclose(test_eigenval[6:], ref_eigenval[6:])

@pytest.mark.parametrize("ff_name", ["Hinsen", "sdENM", "pfENM"])
def test_mass_weights_eigenvals(ff_name):
    """
    Compare mass-weighted eigenvalues with reference values obtained
    with bio3d to test the correctness of the mass-weighting procedure
    and the validity of results obtained with SVD.
    To this end, bio3d-assigned masses are used.
    """
    reference_masses = np.genfromtxt(
        join(data_dir(), "1l2y_bio3d_masses.csv"),
        skip_header=1, delimiter=","
    )
    
    mmtf_file = mmtf.MMTFFile.read(join(data_dir(), "1l2y.mmtf"))
    atoms = mmtf.get_structure(mmtf_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]

    if ff_name == "Hinsen":
        ff = springcraft.HinsenForceField()
        ref_file = "mw_eigenvalues_calpha_bio3d.csv"
    if ff_name == "sdENM":
        ff = springcraft.TabulatedForceField.sd_enm(ca)
        ref_file = "mw_eigenvalues_sdenm_bio3d.csv"
    if ff_name == "pfENM":
        ff = springcraft.ParameterFreeForceField()
        ref_file = "mw_eigenvalues_pfenm_bio3d.csv"
    
    test = springcraft.ANM(ca, ff, masses=reference_masses)
    test_eigenval, _ = test.eigen()

    reference_eigenval = np.genfromtxt(
        join(data_dir(), ref_file),
        skip_header=1, delimiter=","
    )

    assert np.allclose(test_eigenval[6:], reference_eigenval[6:], atol=1e-06)

@pytest.mark.parametrize("ff_name", ["ANM_standard", "Hinsen", "eANM", "sdENM",
                                        "pfENM"]
                                        )
def test_frequency_fluctuation_dcc(ff_name):
    """
    Compare quantities commonly computed as part of NMA.
    Prody/BioPhysConnectoR/Bio3d are used as references.
    """
    # Load bio3d reference
    reference_masses = np.genfromtxt(
        join(data_dir(), "1l2y_bio3d_masses.csv"),
        skip_header=1, delimiter=","
    )

    mmtf_file = mmtf.MMTFFile.read(join(data_dir(), "1l2y.mmtf"))
    atoms = mmtf.get_structure(mmtf_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]

    K_B = 1.380649e-23
    N_A = 6.02214076e23
    tem = 300
    
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

        ref_anm = prody.ANM()
        ref_anm.buildHessian(ca.coord, gamma=1.0, cutoff=13)
        ref_anm.calcModes(n_modes="all")

        reference_freq = 1 / (2 * np.pi) * np.sqrt(ref_anm.getEigvals())
        reference_fluc = prody.calcSqFlucts(ref_anm[0:])
        ref_dcc = prody.calcCrossCorr(ref_anm[0:], norm=True)
        
        # Subset: First 30 non-triv. modes
        ref_dcc_norm_subset = prody.calcCrossCorr(ref_anm[0:30], norm=True)
        ref_dcc_absolute = prody.calcCrossCorr(ref_anm[0:], norm=False)
        
        assert np.allclose(test_freq_no_mw[6:], reference_freq) 
        assert np.allclose(test_fluc_nomw, reference_fluc)
        assert np.allclose(test_dcc, ref_dcc)
        assert np.allclose(test_dcc_absolute, ref_dcc_absolute)
        assert np.allclose(test_dcc_subset, ref_dcc_norm_subset)
    
    # References computed with R packages -> read .csv files in "data"
    else:    
        # BioPhysConnectoR -> no internal computation of DCCs, available;
        #                     no mass-/temperature weighting
        if ff_name =="eANM":
            ff = springcraft.TabulatedForceField.e_anm(ca)
            test_nomw = springcraft.ANM(ca, ff)
            ref_fluc = "bfacs_eANM_mj_BioPhysConnectoR.csv"
            test_nomw = springcraft.ANM(ca, ff)
            test_fluc_nomw = test_nomw.mean_square_fluctuation()
            # -> For alternative MSF computation method; 
            # no temperature weighting
            tem_scaling = 1
            tem = 1
        # Bio3d -> Mass- and temperature weighting
        else:
            if ff_name == "Hinsen":
                ff = springcraft.HinsenForceField()
                ref_freq = "mw_frequencies_calpha_bio3d.csv"
                ref_fluc = "mw_fluctuations_calpha_bio3d.csv"
                ref_fluc_subset = "mw_fluctuations_calpha_subset_bio3d.csv"
                ref_dcc = "dccm_calpha_bio3d.csv"
                ref_dcc_subset = "dccm_calpha_subset_bio3d.csv"
            elif ff_name == "sdENM":
                ff = springcraft.TabulatedForceField.sd_enm(ca)
                ref_freq = "mw_frequencies_sdenm_bio3d.csv"
                ref_fluc = "mw_fluctuations_sdenm_bio3d.csv"
                ref_fluc_subset = "mw_fluctuations_sdenm_subset_bio3d.csv"
                ref_dcc = "dccm_sdenm_bio3d.csv"
                ref_dcc_subset = "dccm_sdenm_subset_bio3d.csv"
            elif ff_name == "pfENM":
                ff = springcraft.ParameterFreeForceField()
                ref_freq = "mw_frequencies_pfenm_bio3d.csv"
                ref_fluc = "mw_fluctuations_pfenm_bio3d.csv"
                ref_fluc_subset = "mw_fluctuations_pfenm_subset_bio3d.csv"
                ref_dcc = "dccm_pfenm_bio3d.csv"
                ref_dcc_subset = "dccm_pfenm_subset_bio3d.csv"
            
            tem_scaling = K_B*N_A
            test_nomw = springcraft.ANM(ca, ff)
            test_fluc_nomw = test_nomw.mean_square_fluctuation(
                                    tem=tem, tem_factors=tem_scaling
                                )

            test = springcraft.ANM(ca, ff, masses=reference_masses)
            test_freq = test.frequencies()

            reference_freq = np.genfromtxt(
                join(data_dir(), ref_freq),
                skip_header=1, delimiter=","
                )

            ## Scale for consistency with bio3d; T=300 K; no mass weighting
            # Start with mass_weighted eigenvals
            test_fluc = test.mean_square_fluctuation(
                            tem=tem, tem_factors=tem_scaling
                            ) / (1000*reference_masses)
                            

            # Select a subset of modes: 12-33
            test_fluc_subset = test.mean_square_fluctuation(
                tem=tem, tem_factors=tem_scaling, mode_subset=np.arange(11, 33)
                )
            test_fluc_subset /= (1000 * reference_masses)

            reference_fluc_subset = np.genfromtxt(
                join(data_dir(), ref_fluc_subset),
                skip_header=1, delimiter=","
                )
            
            # DCCs
            test_dcc = test.dcc()
            # Only consider the first 30 non-triv. modes
            # Mode 6-36 (conventional enumeration)
            test_dcc_subset = test.dcc(mode_subset=np.arange(6, 36))

            reference_dcc = np.genfromtxt(
                join(data_dir(), ref_dcc),
                skip_header=1, delimiter=","
                )
            reference_dcc_subset = np.genfromtxt(
                join(data_dir(), ref_dcc_subset),
                skip_header=1, delimiter=","
                )       

            
        reference_fluc = np.genfromtxt(
            join(data_dir(), ref_fluc),
            skip_header=1, delimiter=","
            )

        ## No mass-weighting
        # Alternative Method for MSF computation considering all modes
        diag = test_nomw.covariance.diagonal()
        reshape_diag = np.reshape(diag, (len(test_nomw._coord),-1))

        # Compute MSF directly from covariance matrix
        msqf_alternative = np.sum(reshape_diag, axis=1)*tem_scaling*tem

        if ff_name == "eANM":
            assert np.allclose(test_fluc_nomw, reference_fluc)
        elif ff_name == "pfENM":
            # Lower values for pfENM compared to other FFs (10e-01)
            # -> Multiply with 1/1000    
            for i in [test_freq[6:], reference_freq[6:], test_fluc, 
                        reference_fluc, test_fluc_subset, 
                        reference_fluc_subset, test_dcc, reference_dcc,
                        test_dcc_subset, reference_dcc_subset]:
                i /= 1000
            
            assert np.allclose(test_freq[6:], reference_freq[6:], atol=1e-04)
            assert np.allclose(test_fluc, reference_fluc, atol=1e-03)
            assert np.allclose(test_fluc_subset, reference_fluc_subset, 
                                atol=1e-04)
            assert np.allclose(test_dcc, reference_dcc, atol=1e-04)
            assert np.allclose(test_dcc_subset, reference_dcc_subset, 
                                atol=1e-04)
        else:
            assert np.allclose(test_freq[6:], reference_freq[6:], atol=1e-06) 
            assert np.allclose(test_fluc, reference_fluc, atol=1e-03)
            assert np.allclose(test_fluc_subset, reference_fluc_subset, 
                                atol=1e-03) 
            assert np.allclose(test_dcc, reference_dcc, atol=1e-04)
            assert np.allclose(test_dcc_subset, reference_dcc_subset, 
                                atol=1e-04) 
   

        # Compare with alternative method of MSF computation
        assert np.allclose(test_fluc_nomw, msqf_alternative)

