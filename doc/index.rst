Springcraft - Investigate molecular dynamics by elastic network models
======================================================================

.. currentmodule:: springcraft

*Springcraft* is a *Biotite* extension package, that allows the analysis
of `AtomArray` objects via *Elastic Network Models* (ENMs).
An ENM can be thought of as a system that connects residues via springs:
Interaction of nearby residues is governed by a harmonic potential, with the
native (input) conformation representing the energy minimum.
Normal mode analysis allows the researcher to investigate global
functional movements of a protein in a fast coarse-grained manner.
For a deeper dive into the theory of ENMs please refer to literature,
such as a `method review from 2010 <https://doi.org/10.1021/cr900095e>`_.


Installation
------------

*Springcraft* can be installed via

.. code-block:: console

   $ pip install springcraft

or 

.. code-block:: console

   $ conda install -c conda-forge springcraft

You can also install *Springcraft* from source on
`GitHub <https://github.com/biotite-dev/springcraft>`_.
The package uses `Poetry <https://python-poetry.org/>`_ for building
distributions.
Via :pep:`517` it is possible to install the package from local source code
via *pip*:

.. code-block:: console

   $ git clone https://github.com/biotite-dev/springcraft.git
   $ pip install ./springcraft


Usage
-----

To compute an ENM a :class:`ForceField` is required, that defines the spring
force constant for each pair of atoms via :meth:`force_constant()`.
A variety of :class:`ForceField` subclasses is available, each bringing
its own concept for parameterization.

.. code-block:: python

   import biotite.structure.io as strucio
   import springcraft

   # All pairs of atoms within a cutoff distance obtain the same force constant
   ff = springcraft.InvariantForceField(cutoff_distance=13.0)

   # The force constant is inversely proportional to the squared distance
   ff = springcraft.ParameterFreeForceField()

   # The force constant is read from tabulated values
   # based on residue types and distance
   # For the residue type the underlying CA-trace is required
   atoms = strucio.load_structure("path/to/structure.pdb")
   atoms = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]
   ff = springcraft.TabulatedForceField.sd_enm(atoms)

|

The functions :func:`compute_kirchhoff()` and :func:`compute_hessian()` use
atom coordinates at the presumed minimum energy conformation, which is usually
simply the experimentally determined conformation, and the :class:`ForceField`
to compute the *Kirchhoff* and *Hessian* matrix of the molecular system,
respectively.
As byproduct it also returns the pairs of atoms that were found within
cutoff distance given by the :class:`ForceField`.

.. code-block:: python

   ff = springcraft.InvariantForceField(cutoff_distance=13.0)
   hessian, pairs = springcraft.compute_hessian(atoms.coord, ff)

|

One type of ENMs are *Gaussian Network models* (GNM).
They do not include directional information, but can be used to
investigate whether atoms move collectively and which atoms are involved
in global movements.
A :class:`GNM` is created using an :class:`AtomArray` representing the
structure model and a :class:`ForceField`.
Internally, both are given to :func:`compute_kirchhoff()` to obtain the
*Kirchhoff* matrix, which can be accessed with the `kirchhoff` attribute.
Actually, :class:`GNM` is only a thin wrapper around
:func:`compute_kirchhoff()`, that conveniently provides common operations
performed on the *Kirchhoff* matrix.
This includes the calculation of eigenvectors and eigenvalues among other
things.
A useful quantity is the correlation between the displacements of two atoms.
This value is contained in the covariance matrix (the `covariance` attribute),
which is the pseudo-inverse of the *Kirchhoff* matrix.

.. note::

   Note that *Springcraft* uses arbitrary units, i.e. factors like the
   *Boltzmann* constant or the temperature are generally not applied,
   if not stated otherwise.

The `kirchhoff` and `covariance` matrix can also be modified to alter the
network model.
Changing one attribute also updates the other attribute.

.. code-block:: python

   gnm = springcraft.GNM(atoms, ff)
   eigen_values, eigen_vectors = gnm.eigen()

|

Analogous to :class:`GNM`, the class for *Anisotropic Network models*
:class:`ANM` is a wrapper around the *Hessian* matrix, which in contrast to the
*Kirchhoff* matrix also includes the directionality of oscillation modes.
Hence, its shape is :math:`3n \times 3n` since it comprises the three
spatial dimensions for each atom :math:`(x_1, y_1, z_1, x_2, ...)`.
Similar to :class:`GNM`, the *Hessian* as well as its pseudo-inverse, the
covariance matrix, are accessible via the `hessian` and `covariance`
attributes.
The added spatial information allows an :class:`ANM` the depiction of
atom oscillations in normal modes or the application of
*Linear Response Theory* to investigate conformational changes upon ligand
binding.

.. code-block:: python

   import numpy as np

   anm = springcraft.ANM(atoms, ff)
   force_vector = np.zeros((atoms.array_length(), 3))
   force_vector[42, 0] = 10
   displacements = anm.linear_response(force_vector)



.. toctree::
   :maxdepth: 1
   :hidden:
   
   examples/gallery/index
   advanced
   apidoc
   
