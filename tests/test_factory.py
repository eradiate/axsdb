import pytest

from axsdb import (
    AbsorptionDatabaseFactory,
    CKDAbsorptionDatabase,
    MonoAbsorptionDatabase,
)


class TestFactory:
    @pytest.fixture
    def path_mono(self, shared_datadir):
        return shared_datadir / "nanomono_v1"

    @pytest.fixture
    def path_ckd(self, shared_datadir):
        return shared_datadir / "nanockd_v1"

    def test_core(self, path_ckd, path_mono):
        factory = AbsorptionDatabaseFactory()

        # Static path registration form
        factory.register(
            name="nanockd",
            cls=CKDAbsorptionDatabase,
            path=path_ckd,
        )

        # Dynamic path registration form
        factory.register(
            name="nanomono",
            cls=MonoAbsorptionDatabase,
            path=lambda: path_mono,
        )

        # Registered databases are registered and that instantiation succeeds
        assert isinstance(factory.create("nanockd"), CKDAbsorptionDatabase)
        assert isinstance(factory.create("nanomono"), MonoAbsorptionDatabase)

        # An unregistered ID raises a comprehensive error
        with pytest.raises(ValueError, match="no database registered"):
            factory.create("doesnt_exist")

    def test_kwargs(self, path_mono):
        factory = AbsorptionDatabaseFactory()
        state = {"lazy": False}
        factory.register(
            name="nanomono",
            cls=MonoAbsorptionDatabase,
            path=path_mono,
            kwargs=lambda: dict(state),
        )

        # Callables are evaluated at each database instantiation
        assert factory.create("nanomono").lazy is False
        state["lazy"] = True
        assert factory.create("nanomono").lazy is True

        # Kwargs passed to create() take precedence
        assert factory.create("nanomono", lazy=False).lazy is False

        # Non-mappings raise an error
        factory.register(
            name="nanomono_nonmapping",
            cls=MonoAbsorptionDatabase,
            path=path_mono,
            kwargs=lambda: [("lazy", True)],
        )
        with pytest.raises(TypeError, match="nanomono_nonmapping"):
            factory.create("nanomono_nonmapping")
