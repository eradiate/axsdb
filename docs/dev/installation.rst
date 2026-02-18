Development installation
========================

Clone the repository and install in development mode:

.. code:: shell

    git clone https://github.com/eradiate/axsdb.git
    cd axsdb
    uv sync --dev --all-extras

Verify your installation:

.. code:: shell

    python -c "import axsdb; print(axsdb.__version__)"

macOS x86_64: numba compatibility
---------------------------------

Numba versions 0.63 and later do not support macOS x86_64. If you are
developing on an Intel Mac, you need to constrain numba to an older version.

Create a ``constraints-local.txt`` file at the project root (this file is
gitignored):

.. code:: shell

    echo "numba<0.63" > constraints-local.txt

Then set the ``UV_CONSTRAINT`` environment variable in your shell profile
(*e.g.* ``~/.zshrc``):

.. code:: shell

    export UV_CONSTRAINT=/path/to/axsdb/constraints-local.txt

After this, ``uv sync`` will automatically respect the constraint and install a
compatible numba version.
