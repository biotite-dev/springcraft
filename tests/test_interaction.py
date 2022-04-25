import itertools
import numpy as np
import pytest
import prody
import biotite.structure.io.pdb as pdb
import springcraft


@pytest.mark.parametrize("seed, cutoff, use_cell_list", itertools.product(
    np.arange(20),
    [5, 10, 15],
    [False, True],
))
def test_kirchhoff(seed, cutoff, use_cell_list):
    """
    Compare computed Kirchhoff matrix with output from *ProDy* with
    randomly generated coordinates.
    """
    N_ATOMS = 1000
    BOX_SIZE = 50

    np.random.seed(seed)
    coord = np.random.rand(N_ATOMS, 3) * BOX_SIZE

    ff = springcraft.InvariantForceField()
    test_kirchhoff, _ = springcraft.compute_kirchhoff(
        coord, ff, cutoff, use_cell_list
    )

    ref_gnm = prody.GNM()
    ref_gnm.buildKirchhoff(coord, gamma=1.0, cutoff=cutoff)
    ref_kirchhoff = ref_gnm.getKirchhoff()

    assert np.allclose(test_kirchhoff, ref_kirchhoff)


@pytest.mark.parametrize("seed, cutoff, use_cell_list", itertools.product(
    np.arange(20),
    [5, 10, 15],
    [False, True]
))
def test_hessian(seed, cutoff, use_cell_list):
    """
    Compare computed Hessian matrix with output from *ProDy* with
    randomly generated coordinates.
    """
    # Relatively small atoms number to increase performance
    N_ATOMS = 200
    BOX_SIZE = 20

    np.random.seed(seed)
    coord = np.random.rand(N_ATOMS, 3) * BOX_SIZE

    ff = springcraft.InvariantForceField()
    test_hessian, _ = springcraft.compute_hessian(
        coord, ff, cutoff, use_cell_list
    )

    ref_gnm = prody.ANM()
    ref_gnm.buildHessian(coord, gamma=1.0, cutoff=cutoff)
    ref_hessian = ref_gnm.getHessian()

    assert np.allclose(test_hessian, ref_hessian, atol=1e-6, rtol=1e-3)
    

@pytest.mark.parametrize("seed, cutoff, use_cell_list", itertools.product(
    np.arange(20),
    [5, 10, 15],
    [False, True],
))
def test_hessian_symmetric(seed, cutoff, use_cell_list):
    N_ATOMS = 1000
    BOX_SIZE = 50

    np.random.seed(seed)
    coord = np.random.rand(N_ATOMS, 3) * BOX_SIZE

    ff = springcraft.InvariantForceField()
    hessian, _ = springcraft.compute_hessian(
        coord, ff, cutoff, use_cell_list
    )

    assert np.allclose(hessian, hessian.T)