import itertools
from os.path import join

import numpy as np
import pytest
import springcraft

from .util import data_dir


@pytest.mark.parametrize(
    "seed, cutoff, use_cell_list",
    itertools.product(
        [1, 323, 777, 999],
        [5, 10, 15],
        [False, True],
    ),
)
def test_kirchhoff(seed, cutoff, use_cell_list):
    """
    Compare computed Kirchhoff matrix with output from *ProDy* with
    randomly generated coordinates.
    """
    # Load randomized coordinates from csv
    coord_rng = np.genfromtxt(
        join(data_dir(), f"random_coord_seed_{seed}.csv.gz"), delimiter=","
    )

    ff = springcraft.InvariantForceField(cutoff)
    test_kirchhoff, _ = springcraft.compute_kirchhoff(coord_rng, ff, use_cell_list)

    ref_kirchhoff = np.genfromtxt(
        join(
            data_dir(),
            f"prody_gnm_{cutoff}_ang_cutoff_kirchhoff_random_coords_seed_{seed}.csv.gz",
        ),
        delimiter=",",
    )

    assert np.allclose(test_kirchhoff, ref_kirchhoff)


@pytest.mark.parametrize(
    "seed, cutoff, use_cell_list",
    itertools.product([1, 323, 777, 999], [10, 15], [False, True]),
)
def test_hessian(seed, cutoff, use_cell_list):
    """
    Compare computed Hessian matrix with output from *ProDy* with
    randomly generated coordinates.
    """
    # Load randomized coordinates from csv
    coord_rng = np.genfromtxt(
        join(data_dir(), f"random_coord_seed_{seed}.csv.gz"), delimiter=","
    )

    ff = springcraft.InvariantForceField(cutoff)
    test_hessian, _ = springcraft.compute_hessian(coord_rng, ff, use_cell_list)

    ref_hessian = np.genfromtxt(
        join(
            data_dir(),
            f"prody_anm_{cutoff}_ang_cutoff_hessian_random_coords_seed_{seed}.csv.gz",
        ),
        delimiter=",",
    )

    assert np.allclose(test_hessian, ref_hessian, atol=1e-6, rtol=1e-3)


@pytest.mark.parametrize(
    "seed, cutoff, use_cell_list",
    itertools.product(
        np.arange(20),
        [5, 10, 15],
        [False, True],
    ),
)
def test_hessian_symmetric(seed, cutoff, use_cell_list):
    N_ATOMS = 1000
    BOX_SIZE = 50

    np.random.seed(seed)
    coord = np.random.rand(N_ATOMS, 3) * BOX_SIZE

    ff = springcraft.InvariantForceField(cutoff)
    hessian, _ = springcraft.compute_hessian(coord, ff, use_cell_list)

    assert np.allclose(hessian, hessian.T)


@pytest.mark.parametrize("use_cell_list", [False, True])
def test_cartesian_index_product(use_cell_list):
    """
    Check if all combinations of atoms are considered in the
    Kirchhoff/Hessian matrix, if no cutoff is given.
    """

    class AllConnectedForceField(springcraft.ForceField):
        def force_constant(self, atom_i, atom_j, sq_distance):
            return np.ones(len(atom_i))

    N_ATOMS = 10
    BOX_SIZE = 50

    np.random.seed(0)
    coord = np.random.rand(N_ATOMS, 3) * BOX_SIZE

    ff = AllConnectedForceField()
    _, pairs = springcraft.compute_hessian(coord, ff, use_cell_list)

    interaction_matrix = np.zeros((N_ATOMS, N_ATOMS), dtype=bool)
    interaction_matrix[tuple(pairs.T)] = True
    # Every possible pair of atoms should interact,
    # except an atom with itself
    assert (interaction_matrix == ~np.identity(N_ATOMS).astype(bool)).all()
