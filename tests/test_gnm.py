import itertools
from os.path import join
import glob
import numpy as np
import pytest
import prody
import biotite.structure.io.mmtf as mmtf
import springcraft
from .util import data_dir


def prepare_gnms(file_path, cutoff):
    mmtf_file = mmtf.MMTFFile.read(file_path)
    atoms = mmtf.get_structure(mmtf_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]

    ff = springcraft.InvariantForceField()
    test_gnm = springcraft.GNM(ca, ff, cutoff)
    
    ref_gnm = prody.GNM()
    ref_gnm.buildKirchhoff(ca.coord, gamma=1.0, cutoff=cutoff)

    return test_gnm, ref_gnm


@pytest.mark.parametrize("file_path, cutoff", itertools.product(
        glob.glob(join(data_dir(), "*.mmtf")),
        [7, 13]
))
def test_kirchhoff(file_path, cutoff):
    """
    Compare computed Kirchhoff matrix with output from *ProDy* with
    test files.
    """
    test_gnm, ref_gnm = prepare_gnms(file_path, cutoff)

    assert test_gnm.kirchhoff.flatten().tolist() \
        == pytest.approx(ref_gnm.getKirchhoff().flatten().tolist())


@pytest.mark.parametrize("file_path, cutoff", itertools.product(
        glob.glob(join(data_dir(), "*.mmtf")),
        [7, 13]
))
def test_eigen(file_path, cutoff):
    """
    Compare computed eigenvalues and -vectors with output from *ProDy*
    with test files.
    """
    test_gnm, ref_gnm = prepare_gnms(file_path, cutoff)

    test_eig_values, test_eig_vectors = test_gnm.eigen()
    # Remove trivial eigenvalues
    #test_eig_vectors = test_eig_vectors[test_eig_values != 0]
    #test_eig_values = test_eig_values[test_eig_values != 0]

    ref_gnm.calcModes("all", zeros=True)
    ref_eig_values = ref_gnm.getEigvals()
    ref_eig_vectors = ref_gnm.getEigvecs()
    ref_eig_vectors = ref_eig_vectors.T

    # Adapt sign of eigenvectors # TODO Is this correct?
    test_eig_vectors *= np.sign(test_eig_vectors[:,0])[:,np.newaxis]
    ref_eig_vectors *= np.sign(ref_eig_vectors[:,0])[:,np.newaxis]



    assert test_eig_values.tolist() == pytest.approx(ref_eig_values.tolist())
    # TODO Investigate issue with eigenvectors
    #assert test_eig_vectors.flatten().tolist() \
    #    == pytest.approx(ref_eig_vectors.flatten().tolist())