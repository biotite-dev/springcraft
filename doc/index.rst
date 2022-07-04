Springcraft - Investigate molecular dynamics with elastic network models 
========================================================================

*Springcraft* is a *Biotite* extension package, that allows the analysis
of `AtomArray` objects via *Elastic Network Models* (ENMs).
An ENM can be thought of a system that connects residues via springs:
Interaction of nearby residues is governed by an harmonic potential, with the
native (input) conformation representing the energy minimum.
Using normal mode analysis it allows the researcher to investigate global
functional movements of a protein in a fast coarse-grained manner.
For deeper dives into the theory of ENMs we like to refer to literature,
such as a `method review from 2010 <https://doi.org/10.1021/cr900095e>`_.

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

You can also install *Springcraft* from source on
`GitHub <https://github.com/biotite-dev/springcraft>`_.
The package uses `Poetry <https://python-poetry.org/>`_ for building
distributions.
Via :pep:`517` it is possible to install the local source code via *pip*:

.. code-block:: console

   $ git clone https://github.com/biotite-dev/springcraft.git
   $ pip install ./springcraft

Usage
-----

.. note::

  Note that *Springcraft* uses arbitrary units, i.e. factors like the
  *Boltzmann* constant or the temperature are not applied.


.. toctree::
   :maxdepth: 1
   :hidden:
   
   apidoc
   examples/gallery/index
