"""
Sanity checks for test data.
"""

from pathlib import Path
import pytest
import xarray as xr
from axsdb.schemas import AcMonov1Dataset, AcCKDv1Dataset
import pandera.xarray as pa


@pytest.fixture
def nanomono_v1_datasets(shared_datadir):
    return Path(shared_datadir / "nanomono_v1").glob("*.nc")


@pytest.fixture
def nanockd_v1_datasets(shared_datadir):
    return Path(shared_datadir / "nanockd_v1").glob("*.nc")


def test_sanity_nanomono_v1(nanomono_v1_datasets):
    """Verify if the nanomono_v1 test dataset is sane."""
    errors = {}

    for path in nanomono_v1_datasets:
        fname = path.parts[-1]
        with xr.open_dataset(path) as ds:
            try:
                AcMonov1Dataset.validate(ds, lazy=True)
            except pa.errors.SchemaErrors as exc:
                errors[fname] = exc.message

    assert not errors, f"Errors when checking {list(errors.keys())}"


def test_sanity_nanockd_v1(nanockd_v1_datasets):
    """Verify if the nanockd_v1 test dataset is sane."""
    errors = {}

    for path in nanockd_v1_datasets:
        fname = path.parts[-1]
        with xr.open_dataset(path) as ds:
            try:
                AcCKDv1Dataset.validate(ds, lazy=True)
            except pa.errors.SchemaErrors as exc:
                errors[fname] = exc.message

    assert not errors, f"Errors when checking {list(errors.keys())}"
