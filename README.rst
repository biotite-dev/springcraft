Springcraft
===========

The project in the CBS Hackathon to compute *Elastic Network Models*
with *Biotite*.

Installation
------------

All packages required for the development (including tests) are installed via

.. code-block:: console

   $ conda env create -f environment.yml

if *Conda* installed.
The package is installed for development via

.. code-block:: console

   $ pip install -e .

This command requires a recent *pip* version.

Example
=======

.. code-block:: python

   import biotite.structure.io.mmtf as mmtf
   import springcraft
   import numpy as np
   
   
   mmtf_file = mmtf.MMTFFile.read("path/to/1l2y.mmtf")
   atoms = mmtf.get_structure(mmtf_file, model=1)
   ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]
   ff = springcraft.InvariantForceField()
   hessian, pairs = springcraft.compute_hessian(ca.coord, ff, 7.0)
   
   np.set_printoptions(linewidth=100)
   print(hessian)