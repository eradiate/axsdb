Interpolation
=============

A large of AxsDB's job consists in performing multivariate lookups and
interpolations in a sparse database. We spent time optimizing them to deliver
the best performance we can.

What makes AxsDB's workflow hard to optimize
--------------------------------------------

The data used to describe atmospheric absorption is 'sparse', in the sense that
the data does not map to a regular grid and can therefore not be indexed easily
with the usual xarray data model.

To be more concrete: a typical database will contain absorption cross-section
values for a set of spectral bins, indexed by pressure, temperature, and an
undefined number of species concentrations. Only the species that contribute
significantly to absorption in that specific spectral bin will appear. This is
a deliberate choice, as the number of dimensions of the data would otherwise
make the database size so large that it could not be reasonably stored on
affordable hardware.

Lookup
------

We start with the spectral information: we know in which spectral bin we need to
look. With this information, we can query the database for the species that are
available, the others being implicitly inactive (in the sense of radiative
absorption). This step is relatively simple to optimize and xarray will do most
of the work for us.

From that point, we can build the set of dimensions that will be used for
interpolation. Typically, for a CKD computation, the dimensions will be:

* g-point ``g``;
* temperature ``t``;
* pressure ``p``;
* species concentrations ``x_*``.

Interpolation
-------------

Interpolation is the harder part of the AxsDB workflow. Once dimensions are
identified, the interpolation of a dense data array has to be performed with a
set of constraints that complicate optimization. There are two ways we can group
dimensions:

* **Out-of-bounds (OOB) handling**: each dimension has its specific policy when
  values out of the database boundaries are encountered. For example, one might
  allow clamping for pressure and temperature (with a warning), and raise for
  species concentrations.
* **Destination dimension**: each dimension maps to another specific destination
  dimension, possibly itself or another that is shared with other dimensions. In
  the CKD case, ``g`` maps to itself, while ``t``, ``p`` and the ``x_*`` all map
  to the altitude ``z``.

Let's first look at the easy options we have to perform multilinear
interpolation on this dense array:

* **Use xarray's built-in multivariate interpolation.** The ``interp()`` method
  supports  multilinear interpolation with careful performance optimizations but
  lacks the control we want for OOB handling: all interpolated dimensions must
  share the  same OOB handling policy. We cannot use this solution.
* **Cascade univariate interpolations with xarray.** This used to be our method,
  but it got severely broken by important changes made to xarray with the
  release of v2025.1. At the time of writing, this issue remains unaddressed and
  entirely rules out xarray as a valid interpolation interface for our datasets.
  This is however the logic we will follow from now on.

We must therefore turn to a lower-level solution and implement our own dataset
interpolation logic. Ideally, we want to use stable basic linear interpolation
components from Numpy or Scipy, so let's have a look at them:

* **Numpy's interp().** This function most probably delivers best-class linear
  interpolation performance but lack one critical feature: it only accepts 1D
  arrays. This rules it out for our use case.
* **Scipy's interp1d().** This function essentially does what ``numpy.interp()``
  does, but also broadcasts on the other dimensions. It is however deprecated,
  meaning that its fate is undecided. Having suffered from trusting too much
  "deprecated but not really, we'll keep it around" promises, we rule out this
  function for our use case.
* **Scipy's interpn().** This function can do both univariate and multivariate
  linear interpolation. As an univariate interpolator, it performs reasonably,
  but we managed to do better; and as a multivariate interpolator, it does not
  support different OOB, which makes it impossible to apply straightforwardly to
  our use case.

Knowing that, we decided to go with a hybrid approach that uses a home-grown
high-performance 1D interpolation routine that support broadcasting on spectator
dimensions, and, when relevant, Scipy's ``interpn()``. As we will see later,
allows us to implement the features we need with excellent performance.

High-performance linear interpolation: Numba core
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At the core of our linear interpolation machinery lies the
:func:`axsdb.math.interp1d` function. Its interface borrows from Scipy's
``interp1d()`` and its core logic is implemented as a Numba generalized
universal function (*gufunc*). Numba's :func:`numba.guvectorize` decorator turns
out to be an easy way to do this, so this is the solution we went with: it
automates broadcasting on spectator dimensions, similar to :func:`scipy.interpolation.interp1d`.

To avoid a bottleneck due to an excessive amount of redundant binary searches
(we assume irregular grids), we added two functions to pre-compute
(:func:`lerp_indices`) and use (:func:`lerp_precomputed`)
element lookup indexes and linear interpolation weights. A Numba gufunc also
powers this path.

It should be noted that both implementations achieve different trade-offs in
terms of performance and interface. The :func:`interp1d` function provides
control on OOB handling, while the lerp functions provide this only during the
index and weight computation step.

High-performance linear interpolation: xarray wrapper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A high-level interface, :func:`axsdb.interpolation.interp_dataarray`, provides a
fast alternative to chained ``interp()`` calls on xarray DataArrays. It
essentially performs the same task, but with different logic optimized for the
use case of AxsDB. For that purpose, it chains univariate interpolations with
the following optimizations:

* **Dimension reordering**: we process shrinking dimensions (query size ≤ grid
  size) first, then expand dimensions in order of decreasing grid size. This
  minimizes intermediate array sizes and reduces the amount of computation.

* **Precomputed indexes**: binary search is done once when query points are
  uniform across batches, avoiding redundant searches.

* **Shared-dimension path**: when interpolating coordinates share a dimension
  with the working data, indexes are precomputed once and applied pointwise.
  This avoids redundant binary searches across the shared dimension.

Although Scipy's ``interpn()`` remains faster according to our benchmarks, chained univariate interpolations are the only way that allows for the
implementation of per-dimension OOB handling. That said, it is often possible to
group dimensions that share identical OOB handling policies and offload their
interpolation to ``interpn()``: this is the last optimization which is
implemented. Some limitations of ``interpn()`` (*e.g.* no asymmetric fill
values) required additional processing logic, but it still allows us to achieve
the best performance in all the scenarios we tested.

Benchmarks
^^^^^^^^^^

*TBD*
