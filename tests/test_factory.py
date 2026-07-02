import pytest

from axsdb import (
    AbsorptionDatabaseFactory,
    CKDAbsorptionDatabase,
    MonoAbsorptionDatabase,
)


def test_factory(shared_datadir):
    factory = AbsorptionDatabaseFactory()

    factory.register(
        name="nanockd", cls=CKDAbsorptionDatabase, path=shared_datadir / "nanockd_v1"
    )
    factory.register(
        name="nanomono",
        cls=MonoAbsorptionDatabase,
        path=lambda: shared_datadir / "nanomono_v1",
    )

    assert isinstance(factory.create("nanockd"), CKDAbsorptionDatabase)
    assert isinstance(factory.create("nanomono"), MonoAbsorptionDatabase)

    with pytest.raises(ValueError, match="no database registered"):
        factory.create("doesnt_exist")
