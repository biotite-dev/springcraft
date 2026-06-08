import itertools
from os.path import join

import biotite.structure as struc
import numpy as np
import pytest

import springcraft
from tests.util import data_dir, load_protein_structure


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


def test_patched_forcefield():
    """
    Tests that PatchedForceFields are correctly handled. Activating
    a contact takes precedence over deactivation.
    """
    ca = load_protein_structure("1l2y")
    coord = np.asarray(struc.coord(ca)).astype(np.float64, copy=False)
    ff = springcraft.InvariantForceField(7.0)
    ff = springcraft.PatchedForceField(
        ff,
        contact_shutdown=[2],
        contact_pair_off=[[3, 2], [3, 4], [3, 5]],
        contact_pair_on=[[2, 4], [3, 4], [3, 14]],
        force_constants=[2, 2, 2],
    )

    test_kirchhoff_cell_list, _ = springcraft.compute_kirchhoff(
        coord, ff, use_cell_list=True
    )
    test_kirchhoff_brute_force, _ = springcraft.compute_kirchhoff(
        coord, ff, use_cell_list=True
    )

    assert np.allclose(test_kirchhoff_cell_list, test_kirchhoff_brute_force)

    # contacts turned on
    kirchhoff = test_kirchhoff_cell_list
    assert kirchhoff[2, 4] == -2
    assert kirchhoff[4, 2] == -2
    assert kirchhoff[3, 4] == -2
    assert kirchhoff[4, 3] == -2
    assert kirchhoff[3, 14] == -2
    assert kirchhoff[14, 3] == -2

    # contacts turned off
    third = np.zeros(len(ca))
    third[2] = 2
    third[4] = -2
    assert np.array_equal(kirchhoff[2, :], third)
    assert np.array_equal(kirchhoff[:, 2], third)
    assert kirchhoff[3, 5] == 0
    assert kirchhoff[5, 3] == 0
