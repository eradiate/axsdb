from pathlib import Path

import numpy as np
import xarray as xr
from scipy.special import roots_sh_legendre

import axsdb
from axsdb import CKDAbsorptionDatabase, MonoAbsorptionDatabase

ROOT_DIR = Path(__file__).parent.parent

ureg = axsdb.units.get_unit_registry()
db = CKDAbsorptionDatabase.from_directory(ROOT_DIR / "benchmarks/data/nanockd_v1")

ws = db.spectral_coverage.index.get_level_values("wavelength [nm]").to_numpy() * ureg.nm
gs, _ = roots_sh_legendre(16)
thermoprops_us_standard = xr.load_dataset(
    ROOT_DIR / "tests/data/afgl_1986-us_standard.nc"
)

db_mono = MonoAbsorptionDatabase.from_directory(
    ROOT_DIR / "benchmarks/data/nanomono_v1"
)
ws_mono = np.linspace(346.0, 354.0, 200) * ureg.nm


class BenchSpectralLoop:
    def main(self):
        for w in ws:
            for g in gs:
                db.eval_sigma_a_ckd(w, g, thermoprops=thermoprops_us_standard)

    def bench_spectral_loop(self, benchmark):
        benchmark(self.main)


class BenchSpectralLoopMono:
    def main(self):
        for w in ws_mono:
            db_mono.eval_sigma_a_mono(w, thermoprops=thermoprops_us_standard)

    def bench_spectral_loop_mono(self, benchmark):
        benchmark(self.main)


if __name__ == "__main__":
    # Use this for profiling
    BenchSpectralLoop().main()
