from pathlib import Path

import xarray as xr
from scipy.special import roots_sh_legendre

import axsdb
from axsdb import CKDAbsorptionDatabase

ROOT_DIR = Path(__file__).parent.parent

ureg = axsdb.units.get_unit_registry()
db = CKDAbsorptionDatabase.from_directory(ROOT_DIR / "benchmarks/data/nanockd_v1")

ws = db.spectral_coverage.index.get_level_values("wavelength [nm]").to_numpy() * ureg.nm
gs, _ = roots_sh_legendre(16)
thermoprops_us_standard = xr.load_dataset(
    ROOT_DIR / "tests/data/afgl_1986-us_standard.nc"
)


class BenchSpectralLoop:
    def main(self):
        for w in ws:
            for g in gs:
                db.eval_sigma_a_ckd(w, g, thermoprops=thermoprops_us_standard)

    def bench_spectral_loop(self, benchmark):
        benchmark(self.main)


if __name__ == "__main__":
    # Use this for profiling
    BenchSpectralLoop().main()
