Error handling
==============

AxsDB allows for fine-grained control on error handling. Each
:class:`.AbsorptionDatabase` instance can be assigned an error handling policy.
When unspecified, a default configuration is used (retrieved with
:func:`.get_error_handling_config`, set with :func:`.set_error_handling_config`).

Handled errors
--------------

The following errors can be controlled:

MISSING
    The coordinate is missing from the database.
SCALAR
    The coordinate is present, but it is scalar.
BOUNDS
    One or several values are out of coordinate bounds when interpolating.

The simplest level of control consists in specifying the *action* that will be
triggered when the error is encountered:

RAISE
    Raise an exception.
WARN
    Emit a warning.
IGNORE
    Ignore the error silently.

**BOUNDS** errors can be applied more fine-grained control, as we are about to see.

Out-of-bounds errors
--------------------

Out-of-bounds (OOB) values are checked early when interpolating the data. Each
bound (lower and upper) can be assigned a different behaviour.

For both the **WARN** and **IGNORE** actions, two modes are implemented:

FILL
    OOB points are assigned a specific.
CLAMP
    The coordinate is clamped.

By default, the fill value is 0.0, and it can be customized. If ``None`` is
used, :class:`np.nan` will be applied.

Configuration examples
----------------------

While error handling configuration is implemented by the
:class:`.ErrorHandlingConfiguration` and associated classes, the recommended
interface uses plain dictionaries. Let's see a few examples.

Basic setup
^^^^^^^^^^^

.. code-block:: pycon
    :emphasize-lines: 12-16

    >>> config = {
    ...     "t": {
    ...         "missing": "raise",
    ...         "scalar": "raise",
    ...         "bounds": "ignore",
    ...     },
    ...     "p": {
    ...         "missing": "raise",
    ...         "scalar": "raise",
    ...         "bounds": "ignore",
    ...     },
    ...     "x": {
    ...         "missing": "ignore",
    ...         "scalar": "ignore",
    ...         "bounds": "raise"
    ...     },
    ... }

All top-level dict entries are required, since we must define a policy of
all coordinates in the database. The
:meth:`.ErrorHandlingConfiguration.convert` constructor automatically
converts simple configuration keywords to appropriate values, in particular
for OOB handling entries:

.. code-block:: pycon
    :emphasize-lines: 3-10

    >>> pprint(ErrorHandlingConfiguration.convert(config))
    ErrorHandlingConfiguration(
        x=ErrorHandlingPolicy(
            missing=<IGNORE>,
            scalar=<IGNORE>,
            bounds=(
                BoundsPolicy(action=<RAISE>, mode=<FILL>, fill_value=0.0),
                BoundsPolicy(action=<RAISE>, mode=<FILL>, fill_value=0.0)
            )
        ),
        p=ErrorHandlingPolicy(
            missing=<RAISE>,
            scalar=<RAISE>,
            bounds=(
                BoundsPolicy(action=<IGNORE>, mode=<FILL>, fill_value=0.0),
                BoundsPolicy(action=<IGNORE>, mode=<FILL>, fill_value=0.0)
            )
        ),
        t=ErrorHandlingPolicy(
            missing=<RAISE>,
            scalar=<RAISE>,
            bounds=(
                BoundsPolicy(action=<IGNORE>, mode=<FILL>, fill_value=0.0),
                BoundsPolicy(action=<IGNORE>, mode=<FILL>, fill_value=0.0)
            )
        )
    )

Full bound specification
^^^^^^^^^^^^^^^^^^^^^^^^

The **MISSING** and **SCALAR** errors only expect an action specifier
(:class:`.ErrorHandlingAction`), which is passed using the corresponding
string. Out-of-bounds errors, on the other hand, are more complex and expect
more settings:

.. code-block:: pycon
    :emphasize-lines: 2-10, 25-32

    >>> config = {
    ...     "t": {
    ...         "missing": "raise",
    ...         "scalar": "raise",
    ...         "bounds": {
    ...             "action": "warn",
    ...             "mode": "fill",
    ...             "fill_value": None,  # will use np.nan
    ...         },
    ...     },
    ...     "p": {
    ...         "missing": "raise",
    ...         "scalar": "raise",
    ...         "bounds": "ignore",
    ...     },
    ...     "x": {
    ...         "missing": "ignore",
    ...         "scalar": "ignore",
    ...         "bounds": "raise"
    ...     },
    ... }
    >>> pprint(ErrorHandlingConfiguration.convert(config))
    ErrorHandlingConfiguration(
        ...,
        t=ErrorHandlingPolicy(
            missing=<RAISE>,
            scalar=<RAISE>,
            bounds=(
                BoundsPolicy(action=<WARN>, mode=<FILL>, fill_value=None),
                BoundsPolicy(action=<WARN>, mode=<FILL>, fill_value=None)
            )
        )
    )

Note that if the ``bounds`` entry receives a single value, symmetric
handling policies are assumed.

Per-bound control
^^^^^^^^^^^^^^^^^

Different out-of-bounds error handling policies can be passed for the lower
and higher bounds. For that purpose, pass a 2-tuple to the ``bounds`` entry:

.. code-block:: pycon
    :emphasize-lines: 5,27-28

    >>> config = {
    ...     "t": {
    ...         "missing": "raise",
    ...         "scalar": "raise",
    ...         "bounds": ("warn", "raise"),
    ...         # When unambiguous, strings are interpreted as actions or
    ...         # OOB modes; omitted entries are assigned the default value
    ...     },
    ...     "p": {
    ...         "missing": "raise",
    ...         "scalar": "raise",
    ...         "bounds": "ignore",
    ...     },
    ...     "x": {
    ...         "missing": "ignore",
    ...         "scalar": "ignore",
    ...         "bounds": "raise"
    ...     },
    ... }
    >>> pprint(ErrorHandlingConfiguration.convert(config))
    ErrorHandlingConfiguration(
        ...,
        t=ErrorHandlingPolicy(
            missing=<RAISE>,
            scalar=<RAISE>,
            bounds=(
                BoundsPolicy(action=<WARN>, mode=<FILL>, fill_value=0.0),
                BoundsPolicy(action=<RAISE>, mode=<FILL>, fill_value=0.0)
            )
        )
    )

This also works with dictionaries:

.. code-block:: pycon
    :emphasize-lines: 5-8,27-30

    >>> config = {
    ...     "t": {
    ...         "missing": "raise",
    ...         "scalar": "raise",
    ...         "bounds": (
    ...             {"action": "warn", "mode": "clamp"},
    ...             "raise",
    ...         ),
    ...     },
    ...     "p": {
    ...         "missing": "raise",
    ...         "scalar": "raise",
    ...         "bounds": "ignore",
    ...     },
    ...     "x": {
    ...         "missing": "ignore",
    ...         "scalar": "ignore",
    ...         "bounds": "raise"
    ...     },
    ... }
    >>> pprint(ErrorHandlingConfiguration.convert(config))
    ErrorHandlingConfiguration(
        ...,
        t=ErrorHandlingPolicy(
            missing=<RAISE>,
            scalar=<RAISE>,
            bounds=(
                BoundsPolicy(action=<WARN>, mode=<CLAMP>, fill_value=0.0),
                BoundsPolicy(action=<RAISE>, mode=<FILL>, fill_value=0.0)
            )
        )
    )
