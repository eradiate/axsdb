from __future__ import annotations
from pathlib import Path

import attrs
from typing import Callable, Type, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from axsdb import AbsorptionDatabase

    AbsorptionDatabaseT = Type[AbsorptionDatabase]


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
    _registry: dict[str, RegistryEntry] = attrs.field(factory=dict)

    def register(
        self,
        name: str,
        cls: AbsorptionDatabaseT,
        path: Path | Callable,
        kwargs: dict[str, Any] | None = None,
    ):
        if kwargs is None:
            kwargs = {}
        self._registry[name] = RegistryEntry(
            name=name, cls=cls, path=path, kwargs=kwargs
        )

    def create(self, name: str, **kwargs) -> AbsorptionDatabase:
        entry = self._registry[name]
        cls = entry.cls
        path = entry.path()
        kwargs = {**entry.kwargs, **kwargs}

        return cls.from_directory(path, **kwargs)
