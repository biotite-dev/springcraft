import numpy as np
import pytest
import biotite.structure.io.pdb as pdb
import biotite.structure.io.mmtf as mmtf

import springcraft

def test_cov_noncov():
    # TODO Comment from Patrick:
    # TODO I think this test function is better organized into the
    # TODO existing 'test_forcefield.py'
    return # TODO: Fix test function
    mmtf_file = mmtf.MMTFFile.read("./data/1l2y.mmtf")    
    atoms = mmtf.get_structure(mmtf_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]

    ff = springcraft.TypeSpecificForceField(ca)
    test_hessian, pairs = springcraft.compute_hessian(ca.coord, ff, 13.0)

    biophysconnector_hessian = np.loadtxt("./data/hesse_1l2y_biophysconnector.txt", usecols=range(60))

    assert test_hessian.flatten().tolist() == pytest.approx(biophysconnector_hessian.flatten().tolist())