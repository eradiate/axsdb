from __future__ import annotations
from pathlib import Path

import attrs
from collections.abc import Callable
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from axsdb import AbsorptionDatabase

    AbsorptionDatabaseT = type[AbsorptionDatabase]


@attrs.define
class RegistryEntry:
    name: str = attrs.field()
    cls: AbsorptionDatabaseT = attrs.field(repr=False)
    _path: Path | Callable = attrs.field(repr=False)
    kwargs: dict[str, Any] = attrs.field(repr=False, factory=dict)

    def path(self):
        return self._path() if callable(self._path) else self._path


@attrs.define
class AbsorptionDatabaseFactory:
    """
    This factory instantiates :class:`.AbsorptionDatabase` subclasses given a
    name. Internally, a registry maps registered names to a matching database
    type, a path where its data is located, and default loading options.

    Examples
    --------
    Initialize a factory instance:

    >>> factory = AbsorptionDatabaseFactory()

    Register a new database:

    >>> factory.register(
    ...     name="nanockd", cls=CKDAbsorptionDatabase, path="~/Downloads/nanockd"
    ... )

    Instantiate a database:

    >>> factory.create("nanockd")  # doctest: +SKIP
    """

    _registry: dict[str, RegistryEntry] = attrs.field(factory=dict, init=False)

    def register(
        self,
        name: str,
        cls: AbsorptionDatabaseT,
        path: Path | Callable,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Register a new database to the factory.

        Parameters
        ----------
        name : str
            A unique identifier for this entry.

        cls : type
            The matching database type.

        path : path-like or callable
            The path to the directory where database data are located. If the
            name is dynamic (*e.g.* resolved by Eradiate's file resolver), a
            callable with signature ``f() -> Path | str`` can be passed.

        kwargs : dict, optional
            Default loading options passed to the constructor when instantiating
            this database.

        Examples
        --------
        Simplest registration pattern:

        >>> factory = AbsorptionDatabaseFactory()
        >>> factory.register(
        ...     name="nanomono", cls=MonoAbsorptionDatabase, path="~/Data/nanomono"
        ... )

        Get path dynamically from a callable:

        >>> def path():
        ...     return "~/Data/nanomono"
        >>> factory.register(
        ...     name="nanomono",
        ...     cls=MonoAbsorptionDatabase,
        ...     path=path,
        ... )

        Add default keyword arguments:

        >>> factory.register(
        ...     name="nanomono",
        ...     cls=MonoAbsorptionDatabase,
        ...     path="~/Data/nanomono",
        ...     kwargs={"lazy": True},
        ... )

        """
        if kwargs is None:
            kwargs = {}
        self._registry[name] = RegistryEntry(
            name=name, cls=cls, path=path, kwargs=kwargs
        )

    def create(self, name: str, **kwargs) -> AbsorptionDatabase:
        """
        Instantiate a database given its name using the
        :meth:`~.AbsorptionDatabase.from_directory` constructor.

        Parameters
        ----------
        name : str
            Name of a registered databased.

        kwargs
            Optional keyword arguments passed to the database constructor. These
            settings will override registered defaults.

        Returns
        -------
        AbsorptionDatabase
            Created database instance.
        """
        try:
            entry = self._registry[name]
        except KeyError:
            raise ValueError(f"no database registered under the name '{name}'")

        cls = entry.cls
        path = entry.path()
        kwargs = {**entry.kwargs, **kwargs}

        return cls.from_directory(path, **kwargs)
