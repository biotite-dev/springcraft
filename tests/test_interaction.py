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

    assert test_kirchhoff.flatten().tolist() \
        == pytest.approx(ref_kirchhoff.flatten().tolist())