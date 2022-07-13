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
    print(file_path)
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

    assert np.allclose(test_hessian, np.dot(test_hessian, np.dot(test_covariance, test_hessian)))
    assert np.allclose(test_covariance, np.dot(test_covariance, np.dot(test_hessian, test_covariance)))

## Will be merged with prepare_anms
# Compare msqf with BioPhysConnectoR B-factors
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
