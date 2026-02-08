import pytest
import numpy
import xarray
import axsdb
import axsdb.interpolation


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace["np"] = numpy
    doctest_namespace["xr"] = xarray
    doctest_namespace["axsdb"] = axsdb
    doctest_namespace["AbsorptionDatabaseFactory"] = axsdb.AbsorptionDatabaseFactory
    doctest_namespace["CKDAbsorptionDatabase"] = axsdb.CKDAbsorptionDatabase
    doctest_namespace["MonoAbsorptionDatabase"] = axsdb.MonoAbsorptionDatabase
    doctest_namespace["interp_dataarray"] = axsdb.interpolation.interp_dataarray
