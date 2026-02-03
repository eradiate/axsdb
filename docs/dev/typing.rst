Type-hint conventions
=====================

AxsDB targets Python 3.9+.  The rules below keep annotations consistent across
the codebase and avoid deprecated ``typing`` shims.

``from __future__ import annotations``
    Every module that carries type annotations must open with::

        from __future__ import annotations

    This makes all annotations strings at runtime, which is what allows the
    remaining rules to work on Python 3.9 without runtime errors.

Built-in generics
    Use lowercase built-in types as generics directly, *e.g.*::

        dict[str, int]
        list[float]
        tuple[np.ndarray, np.ndarray]
        set[str]

    Avoid importing ``typing``\ 's ``Dict``, ``List``, ``Tuple``, or ``Set``.

Union and ``None``
    Use the pipe operator for all unions and for optional types::

        float | tuple[float, float]      # not Union[float, Tuple[float, float]]
        ErrorHandlingConfiguration | None  # not Optional[ErrorHandlingConfiguration]

Abstract container types
    Import abstract base classes from ``collections.abc``, not ``typing``::

        from collections.abc import Callable, Hashable, Mapping, Sequence

Class-object annotations
    Use lowercase ``type[X]`` instead of ``typing.Type[X]``::

        AbsorptionDatabaseT = type[AbsorptionDatabase]   # inside TYPE_CHECKING block

Permitted ``typing`` imports
    The following names have no built-in or ``collections.abc`` equivalent and are
    the only items that should appear in a ``from typing import …`` line:

    * ``Any``
    * ``Annotated``
    * ``Literal``
    * ``TypeAlias``
    * ``TypeVar``
    * ``TYPE_CHECKING``

Tests
    Test functions and methods do not carry type annotations. Helpers might,
    but this is not required.
