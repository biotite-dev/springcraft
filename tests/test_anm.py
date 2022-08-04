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

## Will be merged with prepare_anms
# Compare msqf with BioPhysConnectoR "B-factors"
@pytest.mark.parametrize("file_path",
        glob.glob(join(data_dir(), "*.mmtf"))
)
def test_mean_square_fluctuation(file_path):
    mmtf_file = mmtf.MMTFFile.read(file_path)

    atoms = mmtf.get_structure(mmtf_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]

    ff_eanm = springcraft.TabulatedForceField.e_anm(ca)
    test_eanm = springcraft.ANM(ca, ff_eanm)
    test_msqf = test_eanm.mean_square_fluctuation()

    # Load .csv file data from BiophysConnectoR
    ref_file = "bfacs_eANM_mj_BioPhysConnectoR.csv"
    ref_msqf = np.genfromtxt(
        join(data_dir(), ref_file),
        skip_header=1, delimiter=","
    )

    assert np.allclose(test_msqf, ref_msqf)


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
    ff = springcraft.HinsenForceField()

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