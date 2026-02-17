Benchmarking
============

We monitor the performance of this package using benchmarks written with the
`pytest-benchmark <https://github.com/ionelmc/pytest-benchmark>`__ plugin.
Benchmarks are located in the ``benchmarks`` subdirectory and written using
regular pytest features.

Running the benchmarking suite
------------------------------

The pytest-benchmark plugin is loaded only when running the benchmarking suite. A local configuration file changes the prefixes used by pytest to
discover files (``_benchmark``), classes (``Benchmark``) and functions
(``_benchmark``). Just run

.. code-block:: shell

    # with the dedicated taskipy task
    uv run task benchmarks
    # or directly calling pytest
    pytest benchmarks

You can use all pytest features to select tests, increase verbosity, etc.
