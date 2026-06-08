Springcraft
===========

*Springcraft* is a *Biotite* extension package, that allows the analysis
of `AtomArray` objects via *Elastic Network Models* (ENMs).
An ENM can be thought of as a system that connects residues via springs:
Interaction of nearby residues is governed by a harmonic potential, with the
native (input) conformation representing the energy minimum.
Normal mode analysis allows the researcher to investigate global
functional movements of a protein in a fast coarse-grained manner.

.. note::

  *Springcraft* is still in alpha stage.
  Although most implemented functionalities should already work as
  expected, some features are not well tested, yet.

Installation
------------

*Springcraft* can be installed via

.. code-block:: console

   $ pip install springcraft

or

.. code-block:: console

   $ conda install -c conda-forge springcraft

You can also install *Springcraft* from source.
The package uses `Poetry <https://python-poetry.org/>`_ for building
distributions.
Via :pep:`517` it is possible to install the package from local source code
via *pip*:

.. code-block:: console

   $ git clone https://github.com/biotite-dev/springcraft.git
   $ pip install ./springcraft

Development
-----------

Conda Environment
~~~~~~~~~~~~~~~~~

You can use conda's optimized libraries for development by creating the provided
conda enviroment in `environment.yml` and than installing the project in editable
mode. This creates symlinks from conda environment to your project files so that
changes are synced directly.

.. code-block:: console

   $ mamba env create -f environment.yml
   $ mamba activate springcraft-dev
   $ pip install -e .

Linting is done by `ruff` and tests are resolved by `pytest`. After installing the
dependencies into the conda env you can run them by

.. code-block:: console

   $ pip install -e .[lint,test]
   $ ruff check .  # run linter
   $ pytest  # run test
   $ pytest --cov=springcraft --cov-report=html --cov-branch  # run test with coverage

You can lint the numpy docstrings using `numpydoc` and built the documentation with
`sphinx-build` like this

.. code-block:: console

   $ pip install -e .[docs]
   $ python -m numpydoc lint src/springcraft/*.py
   $ sphinx-build doc build/doc

Hatch Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~

You can also Hatch's virtual environment system. After installing `hatch` all further
dependency management will be done by `hatch` itself. Dependencies will be installed
into a virtual environment. Just run the following:

.. code-block:: console

   $ hatch run lint:run  # run linter
   $ hatch run test:run  # run tests
   $ hatch run docs:lint  # run linter on docstrings

Building the documentation does not work using hatch. Use the conda env instead.

Generate test files
~~~~~~~~~~~~~~~~~~~

Scripts to generate reference files for tests are stored in tests/data;
a separate environment to rerun these locally can be found in `environment-test-data.yml`.
BioPhysConnectoR has to be installed separately.

Example
=======

.. code-block:: python

   import numpy as np
   import biotite.structure.io.pdbx as pdbx
   import springcraft


   pdbx_file = pdbx.PDBxFile.read("path/to/1l2y.cif")
   atoms = pdbx.get_structure(pdbx_file, model=1)
   ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]
   ff = springcraft.InvariantForceField(cutoff_distance=7.0)
   gnm = springcraft.GNM(ca, ff)
   kirchhoff = gnm.kirchhoff

   np.set_printoptions(linewidth=100)
   print(kirchhoff)

Output:

.. code-block:: none

   [[ 4. -1. -1. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    [-1.  6. -1. -1. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. -1.  0.]
    [-1. -1.  7. -1. -1. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. -1.  0.]
    [-1. -1. -1.  7. -1. -1. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    [-1. -1. -1. -1.  8. -1. -1. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    [ 0. -1. -1. -1. -1.  9. -1. -1. -1.  0. -1.  0.  0.  0.  0.  0.  0. -1.  0.  0.]
    [ 0.  0. -1. -1. -1. -1.  8. -1. -1. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    [ 0.  0.  0. -1. -1. -1. -1.  7. -1. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    [ 0.  0.  0.  0. -1. -1. -1. -1.  7. -1. -1.  0.  0. -1.  0.  0.  0.  0.  0.  0.]
    [ 0.  0.  0.  0.  0.  0. -1. -1. -1.  7. -1. -1. -1. -1.  0.  0.  0.  0.  0.  0.]
    [ 0.  0.  0.  0.  0. -1. -1. -1. -1. -1.  8. -1. -1. -1.  0.  0.  0.  0.  0.  0.]
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0. -1. -1.  7. -1. -1. -1. -1. -1.  0.  0.  0.]
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0. -1. -1. -1.  5. -1. -1.  0.  0.  0.  0.  0.]
    [ 0.  0.  0.  0.  0.  0.  0.  0. -1. -1. -1. -1. -1.  7. -1. -1.  0.  0.  0.  0.]
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. -1. -1. -1.  4. -1.  0.  0.  0.  0.]
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. -1.  0. -1. -1.  5. -1. -1.  0.  0.]
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. -1.  0.  0.  0. -1.  4. -1. -1.  0.]
    [ 0.  0.  0.  0.  0. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0. -1. -1.  5. -1. -1.]
    [ 0. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. -1. -1.  5. -1.]
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. -1. -1.  2.]]
