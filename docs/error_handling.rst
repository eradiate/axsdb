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

The simplest level of control consists in specifying the action that will be
triggered when the error is encountered:

RAISE
    Raise an exception.

WARN
    Emit a warning.

IGNORE
    Ignore the error silently.

BOUNDS errors can be applied more fine-grained control, as we are about to see.

Out-of-bounds errors
--------------------

Out-of-bounds (OOB) values are checked early when interpolating the data. Each
bound (lower and upper) can be assigned a different behaviour.

For both the WARN and IGNORE actions, two modes are implemented:

FILL
    OOB points are assigned a specific.
CLAMP
    The coordinate is clamped.

By default, the fill value is 0.0, and it can be customized.

Configuration examples
----------------------

While error handling configuration is implemented by the :class:`.ErrorHandlingConfiguration` and associated classes, the configuration is
primarily done using plain dictionaries.

* Examples
