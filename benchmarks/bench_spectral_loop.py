import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from scipy.special import roots_sh_legendre

import axsdb
from axsdb import (
    CKDAbsorptionDatabase,
    ErrorHandlingConfiguration,
    MonoAbsorptionDatabase,
)

ROOT_DIR = Path(__file__).parent.parent

ureg = axsdb.units.get_unit_registry()
thermoprops_us_standard = xr.load_dataset(
    ROOT_DIR / "tests/data/afgl_1986-us_standard.nc"
)


def make_db(mode: str, lazy: bool):
    if mode == "mono":
        return MonoAbsorptionDatabase.from_directory(
            ROOT_DIR / "benchmarks/data/nanomono_v1", lazy=lazy
        )

    if mode == "ckd":
        return CKDAbsorptionDatabase.from_directory(
            ROOT_DIR / "benchmarks/data/nanockd_v1", lazy=lazy
        )

    raise ValueError(f"unknown mode {mode!r}")


class BenchSpectralLoopCKD:
    def main(self, db, thermoprops, ws, gs):
        for w in ws:
            for g in gs:
                db.eval_sigma_a_ckd(w, g, thermoprops=thermoprops)

    @pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
    def bench_spectral_loop(self, benchmark, lazy):
        db = make_db("ckd", lazy)
        ws = (
            db.spectral_coverage.index.get_level_values("wavelength [nm]").to_numpy()
        ) * ureg.nm
        gs, _ = roots_sh_legendre(16)

        benchmark(self.main, db, thermoprops_us_standard, ws, gs)


class BenchSpectralLoopMono:
    def main(self, db, thermoprops, ws):
        for w in ws:
            db.eval_sigma_a_mono(w, thermoprops=thermoprops)

    @pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
    def bench_spectral_loop_mono(self, benchmark, lazy):
        db = make_db("mono", lazy)
        ws = np.linspace(346.0, 354.0, 200) * ureg.nm
        benchmark(self.main, db, thermoprops_us_standard, ws)


class BenchSpectralLoopEradiate:
    """
    Full CKD spectral sweep over Eradiate's CKD databases, in Eradiate's default
    configuration (eager, Eradiate error handling).
    Point ``AXSDB_BENCH_MONOTROPA`` to the database directory to enable it.
    """

    def main(self, db, thermoprops, ws, gs):
        for w in ws:
            for g in gs:
                db.eval_sigma_a_ckd(w, g, thermoprops=thermoprops)

    @pytest.mark.parametrize("db_id", ["monotropa"])
    def bench_spectral_loop_eradiate(self, benchmark, db_id):
        envvar = f"AXSDB_BENCH_{db_id.upper()}"
        path = os.environ.get(envvar)
        if path is None:
            pytest.skip(f"{envvar} is not set")

        db = CKDAbsorptionDatabase.from_directory(
            path,
            lazy=False,
            error_handling_config=ErrorHandlingConfiguration.convert(
                {  # Eradiate default
                    "p": {"missing": "raise", "scalar": "raise", "bounds": "ignore"},
                    "t": {"missing": "raise", "scalar": "raise", "bounds": "ignore"},
                    "x": {"missing": "ignore", "scalar": "ignore", "bounds": "raise"},
                }
            ),
        )
        ws = (
            db.spectral_coverage.index.get_level_values("wavelength [nm]").to_numpy()
        ) * ureg.nm
        gs, _ = roots_sh_legendre(16)
        # Some monotropa files cap the x_CO grid at 1e-6 (float32), below the
        # AFGL profile's maximum of 5e-5: clip to avoid bound errors
        thermoprops = thermoprops_us_standard.copy()
        thermoprops["x_CO"] = thermoprops["x_CO"].clip(max=float(np.float32(1e-6)))

        # a few seconds per sweep: keep round count low
        benchmark.pedantic(
            self.main,
            args=(db, thermoprops, ws, gs),
            rounds=5,
            warmup_rounds=1,
        )


if __name__ == "__main__":
    # Use this for profiling
    db = make_db("ckd", False)
    ws = (
        db.spectral_coverage.index.get_level_values("wavelength [nm]").to_numpy()
    ) * ureg.nm
    gs, _ = roots_sh_legendre(16)
    BenchSpectralLoopCKD().main(db, thermoprops_us_standard, ws, gs)
