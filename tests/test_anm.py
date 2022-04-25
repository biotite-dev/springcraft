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

    ff = springcraft.InvariantForceField()
    test_anm = springcraft.ANM(ca, ff, cutoff)
    
    ref_anm = prody.ANM()
    ref_anm.buildHessian(ca.coord, gamma=1.0, cutoff=13)

    return test_anm, ref_anm

@pytest.mark.parametrize("file_path",
        glob.glob(join(data_dir(), "*.mmtf"))
)
def test_covariance(file_path):
    test_anm, ref_anm = prepare_anms(file_path, cutoff= 13)
    test_hessian = test_anm.hessian
    test_covariance = test_anm.covariance

    assert np.allclose(test_hessian, np.dot(test_hessian, np.dot(test_covariance, test_hessian)))
    assert np.allclose(test_covariance, np.dot(test_covariance, np.dot(test_hessian, test_covariance)))

