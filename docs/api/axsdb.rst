axsdb
=====

.. py:module:: axsdb

.. rubric:: Submodules:

.. toctree::
    :maxdepth: 2
    :hidden:

    axsdb.interpolation
    axsdb.math
    axsdb.units

.. autosummary::

    interpolation
    math
    units

.. rubric:: Database classes

.. autosummary::

    AbsorptionDatabase
    MonoAbsorptionDatabase
    CKDAbsorptionDatabase

.. rubric:: Database factory

.. autosummary::

    AbsorptionDatabaseFactory

.. rubric:: Error handling

.. autosummary::

    get_error_handling_config
    set_error_handling_config
    ErrorHandlingConfiguration
    ErrorHandlingPolicy
    ErrorHandlingAction
    BoundsPolicy
    BoundsMode

--------------------------------------------------------------------------------

Database classes
----------------

.. autoclass:: AbsorptionDatabase
.. autoclass:: MonoAbsorptionDatabase
.. autoclass:: CKDAbsorptionDatabase

Database factory

.. autoclass:: AbsorptionDatabaseFactory

Error handling
--------------

.. autofunction:: get_error_handling_config
.. autofunction:: set_error_handling_config
.. autoclass:: ErrorHandlingConfiguration
.. autoclass:: ErrorHandlingPolicy
.. autoclass:: ErrorHandlingAction
    :no-autosummary:
.. autoclass:: BoundsPolicy
.. autoclass:: BoundsMode
    :no-autosummary:
