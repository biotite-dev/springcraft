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

@pytest.mark.parametrize("ff_name", ["Hinsen", "sdENM"])
def test_mass_weights_eigenvals(ff_name):
    """
    Compare mass-weighted eigenvalues with reference values obtained
    with bio3d.
    To this end, bio3d-assigned masses are used.
    """
    reference_masses = np.genfromtxt(
        join(data_dir(), "1l2y_bio3d_masses.csv"),
        skip_header=1, delimiter=","
    )
    
    mmtf_file = mmtf.MMTFFile.read(join(data_dir(), "1l2y.mmtf"))
    atoms = mmtf.get_structure(mmtf_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]
    #ff = springcraft.HinsenForceField()

    if ff_name == "Hinsen":
        ff = springcraft.HinsenForceField()
        ref_file = "mw_eigenvalues_calpha_bio3d.csv"
    if ff_name == "sdENM":
        ff = springcraft.TabulatedForceField.sd_enm(ca)
        ref_file = "mw_eigenvalues_sdenm_bio3d.csv"
    
    test = springcraft.ANM(ca, ff, masses=reference_masses)
    test_eigenval, _ = test.eigen()

    reference_eigenval = np.genfromtxt(
        join(data_dir(), ref_file),
        skip_header=1, delimiter=","
    )

    assert np.allclose(test_eigenval[6:], reference_eigenval[6:], atol=1e-06)

@pytest.mark.parametrize("ff_name", ["Hinsen", "eANM", "sdENM", "pfENM"])
def test_analysis(ff_name):
    """
    Compare quantities commonly computed as part of NMA with bio3d.
    """
    reference_masses = np.genfromtxt(
        join(data_dir(), "1l2y_bio3d_masses.csv"),
        skip_header=1, delimiter=","
    )

    mmtf_file = mmtf.MMTFFile.read(join(data_dir(), "1l2y.mmtf"))
    atoms = mmtf.get_structure(mmtf_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]

    if ff_name =="eANM":
        ff = springcraft.TabulatedForceField.e_anm(ca)
        test_nomw = springcraft.ANM(ca, ff)
        test_fluc_nomw = test_nomw.mean_square_fluctuation()
        ref_fluc = "bfacs_eANM_mj_BioPhysConnectoR.csv"
    else:
        if ff_name == "Hinsen":
            ff = springcraft.HinsenForceField()
            ref_freq = "mw_frequencies_calpha_bio3d.csv"
            ref_fluc = "mw_fluctuations_calpha_bio3d.csv"
        if ff_name == "sdENM":
            ff = springcraft.TabulatedForceField.sd_enm(ca)
            ref_freq = "mw_frequencies_sdenm_bio3d.csv"
            ref_fluc = "mw_fluctuations_sdenm_bio3d.csv"
        if ff_name == "pfENM":
            ff = springcraft.ParameterFreeForceField()
            ref_freq = "mw_frequencies_pfenm_bio3d.csv"
            ref_fluc = "mw_fluctuations_pfenm_bio3d.csv"

        test_nomw = springcraft.ANM(ca, ff)
        test = springcraft.ANM(ca, ff, masses=reference_masses)
        test_freq = test.frequencies()

        ## Scale for consistency with bio3d; T=300 K; no mass weighting
        # -> deviates from bio3d implementation (less intermediary steps)
        test_fluc_nomw = test_nomw.mean_square_fluctuation(tem=300)/1000
        # start from mass_weighted eigenvals
        test_fluc = test.mean_square_fluctuation(tem=300)/(1000*reference_masses)
        
        reference_freq = np.genfromtxt(
            join(data_dir(), ref_freq),
            skip_header=1, delimiter=","
            )

    reference_fluc = np.genfromtxt(
        join(data_dir(), ref_fluc),
        skip_header=1, delimiter=","
        )

    if ff_name == "eANM":
        assert np.allclose(test_fluc_nomw, reference_fluc)
    elif ff_name == "pfENM":
        assert np.allclose(test_freq[6:], reference_freq[6:], atol=1e-05) 
        #assert np.allclose(np.round(test_fluc_nomw), np.round(reference_fluc))
    else:
        assert np.allclose(test_freq[6:], reference_freq[6:], atol=1e-03) 
        assert np.allclose(test_fluc, reference_fluc, atol=1e-03)