import itertools
from os.path import join
import glob
from statistics import covariance
import numpy as np
import pytest
import prody
import biotite.structure.io.mmtf as mmtf
import springcraft
from springcraft.anm import ANM
#from .util import data_dir

@pytest.mark.parametrize("seed, n_residue", itertools.product(
    np.arange(20),
    np.arange(100),
))
def test_covariance(seed, n_residue):
    np.random.seed(seed)
    hessian = np.random.rand(n_residue*3, n_residue*3)

    covariance = ANM.covariance(hessian)

    assert np.allclose(np.allclose(hessian, np.dot(hessian, np.dot(covariance, hessian))))
    assert np.allclose(np.allclose(covariance, np.dot(covariance, np.dot(hessian, covariance))))

