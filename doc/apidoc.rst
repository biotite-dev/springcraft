API Reference
=============

.. currentmodule:: springcraft


.. contents::
   :depth: 3


Network Models
--------------

.. autoclass:: GNM

   Normal Mode Analysis
   ~~~~~~~~~~~~~~~~~~~~
   .. automethod:: frequencies
   .. automethod:: mean_square_fluctuation
   .. automethod:: bfactor
   .. automethod:: dcc

|

.. autoclass:: ANM

   Normal Mode Analysis
   ~~~~~~~~~~~~~~~~~~~~
   .. automethod:: frequencies
   .. automethod:: mean_square_fluctuation
   .. automethod:: bfactor
   .. automethod:: dcc

|

.. autoclass:: springcraft.enm.ENM

   .. automethod:: eigen


Normal Mode Analysis
--------------------

.. autofunction:: springcraft.nma.frequencies

.. autofunction:: springcraft.nma.mean_square_fluctuation

.. autofunction:: springcraft.nma.bfactor

.. autofunction:: springcraft.nma.dcc


Force Fields
------------

.. autoclass:: ForceField
   :members:

|

.. autoclass:: PatchedForceField

|

.. autoclass:: InvariantForceField

|

.. autoclass:: HinsenForceField

|

.. autoclass:: ParameterFreeForceField

|

.. autoclass:: TabulatedForceField
   :members: s_enm_10, s_enm_13, d_enm, sd_enm, e_anm, e_anm_mj, e_anm_ke

|

Miscellaneous
-------------

.. autofunction:: compute_kirchhoff

.. autofunction:: compute_hessian
