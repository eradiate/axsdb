"""
Practical usage benchmark.

This benchmark simulates usage of AxsDB in actual conditions and performs
evaluations for an atmospheric profile with various resolutions.
"""

from itertools import product

import numpy as np
import pytest

from axsdb import ErrorHandlingConfiguration
from axsdb.testing.fixtures import *  # noqa: F403
from axsdb.units import ureg

WGS = {
    "scalar_g": ([350.0] * ureg.nm, 0.5),
    # "vector_g": (
    #     [350.0] * ureg.nm,
    #     [
    #         0.00529953,
    #         0.02771249,
    #         0.0671844,
    #         0.1222978,
    #         0.19106188,
    #         0.27099161,
    #         0.35919822,
    #         0.45249375,
    #         0.54750625,
    #         0.64080178,
    #         0.72900839,
    #         0.80893812,
    #         0.8777022,
    #         0.9328156,
    #         0.97228751,
    #         0.99470047,
    #     ],
    # ),
}
Z_LEVELS = [121, 1201, 12001]


@pytest.fixture(
    params=list(product(WGS.values(), Z_LEVELS)),
    ids=[f"{wg}-{z_levels}" for wg, z_levels in product(WGS.keys(), Z_LEVELS)],
)
def setup_ckd(
    request,
    thermoprops_us_standard,
    absdb_ckd,
    absorption_database_error_handler_config,
):
    wg, z_levels = request.param

    # Resample thermoprops to requested number of z-levels
    z_new = np.linspace(
        thermoprops_us_standard["z"].values.min(),
        thermoprops_us_standard["z"].values.max(),
        z_levels,
    )
    thermoprops = thermoprops_us_standard.interp(z=z_new)

    yield {
        "wg": wg,
        "thermoprops": thermoprops,
        "absdb": absdb_ckd,
        "error_handler_config": ErrorHandlingConfiguration.convert(
            absorption_database_error_handler_config
        ),
    }


class BenchmarkEval:
    def eval_ckd(self, wg, absdb, thermoprops, error_handler_config):
        return absdb.eval_sigma_a_ckd(
            *wg, thermoprops=thermoprops, error_handling_config=error_handler_config
        )

    def benchmark_eval(self, setup_ckd, benchmark):
        wg = setup_ckd["wg"]
        absdb = setup_ckd["absdb"]
        thermoprops = setup_ckd["thermoprops"]
        error_handler_config = setup_ckd["error_handler_config"]

        benchmark(self.eval_ckd, wg, absdb, thermoprops, error_handler_config)
