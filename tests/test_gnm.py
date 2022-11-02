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

    ff = springcraft.InvariantForceField(cutoff)
    test_gnm = springcraft.GNM(ca, ff)
    
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
        # Cutoff must not be too large,
        # otherwise degenerate eigenvalues appear
        [4, 7]
))
def test_eigen(file_path, cutoff):
    """
    Compare computed eigenvalues and -vectors with output from *ProDy*
    with test files.
    """
    test_gnm, ref_gnm = prepare_gnms(file_path, cutoff)

    test_eig_values, test_eig_vectors = test_gnm.eigen()

    ref_gnm.calcModes("all", zeros=True)
    ref_eig_values = ref_gnm.getEigvals()
    ref_eig_vectors = ref_gnm.getEigvecs().T

    # Adapt sign of eigenvectors # TODO Is this correct?
    test_eig_vectors *= np.sign(test_eig_vectors[:,0])[:,np.newaxis]
    ref_eig_vectors *= np.sign(ref_eig_vectors[:,0])[:,np.newaxis]

    assert test_eig_values.tolist() == pytest.approx(ref_eig_values.tolist())
    assert test_eig_vectors.flatten().tolist() \
        == pytest.approx(ref_eig_vectors.flatten().tolist())


def test_mass_weights_simple():
    """
    Expect that mass weighting with unit masses does not have any
    influence on an GNM, but different weights do.
    """
    mmtf_file = mmtf.MMTFFile.read(join(data_dir(), "1l2y.mmtf"))
    atoms = mmtf.get_structure(mmtf_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]
    ff = springcraft.InvariantForceField(7.9)

    ref_gnm = springcraft.GNM(atoms, ff)
    identical_gnm = springcraft.GNM(
        atoms, ff, masses=np.ones(atoms.array_length())
    )
    different_gnm = springcraft.GNM(
        atoms, ff, masses=np.arange(1, atoms.array_length() + 1, dtype=float)
    )

    assert np.allclose(identical_gnm.kirchhoff, ref_gnm.kirchhoff)
    assert not np.allclose(different_gnm.kirchhoff, ref_gnm.kirchhoff)

@pytest.mark.parametrize("file_path, cutoff", itertools.product(
        glob.glob(join(data_dir(), "*.mmtf")),
        [4, 7]
))
def test_fluctuation_dcc(file_path, cutoff):
    """
    Comparison of mean-square fluctuations and 
    dynamic cross-correlations computed with Springcraft and Prody. 
    """
    test_gnm, ref_gnm = prepare_gnms(file_path, cutoff)
    test_fluc = test_gnm.mean_square_fluctuation()
    test_dcc = test_gnm.dcc()
    test_dcc_absolute = test_gnm.dcc(norm=False)
    test_dcc_subset = test_gnm.dcc(mode_subset=np.arange(1, 17))

    ref_gnm.calcModes(n_modes="all")
    reference_fluc = prody.calcSqFlucts(ref_gnm[0:])
    reference_dcc = prody.calcCrossCorr(ref_gnm[0:])
    ref_dcc_norm_subset = prody.calcCrossCorr(ref_gnm[0:16], norm=True)
    reference_dcc_absolute = prody.calcCrossCorr(ref_gnm[0:], norm=False)


    assert np.allclose(test_fluc, reference_fluc)
    assert np.allclose(test_dcc, reference_dcc)
    assert np.allclose(test_dcc_subset, ref_dcc_norm_subset)
    assert np.allclose(test_dcc_absolute, reference_dcc_absolute)