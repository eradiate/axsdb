import pytest
import numpy
import xarray
import axsdb
import axsdb.interpolation
from rich.console import Console
from rich.pretty import Pretty


def pprint(obj):
    """Pretty print using rich without box drawing characters."""
    console = Console(legacy_windows=True, force_terminal=False, no_color=True)
    console.print(Pretty(obj, indent_guides=False))


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace["np"] = numpy
    doctest_namespace["xr"] = xarray
    doctest_namespace["axsdb"] = axsdb
    doctest_namespace["AbsorptionDatabaseFactory"] = axsdb.AbsorptionDatabaseFactory
    doctest_namespace["CKDAbsorptionDatabase"] = axsdb.CKDAbsorptionDatabase
    doctest_namespace["MonoAbsorptionDatabase"] = axsdb.MonoAbsorptionDatabase
    doctest_namespace["interp_dataarray"] = axsdb.interpolation.interp_dataarray
    doctest_namespace["ErrorHandlingConfiguration"] = axsdb.ErrorHandlingConfiguration
    doctest_namespace["pprint"] = pprint
