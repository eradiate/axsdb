Benchmarking
============

We monitor the performance of this package using benchmarks written with the
`pytest-benchmark <https://github.com/ionelmc/pytest-benchmark>`__ plugin.
Benchmarks are located in the ``benchmarks`` subdirectory and written using
regular pytest features.

Running the benchmarking suite
------------------------------

A local configuration file changes the prefixes used by pytest to discover
files, classes and functions. Just run

.. code-block:: shell

    pytest benchmarks

You can use all pytest features to select tests, increase verbosity, etc.
