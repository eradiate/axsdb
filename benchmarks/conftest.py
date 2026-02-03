import fnmatch

import pytest


def pytest_configure(config):
    config.addinivalue_line("python_classes", "Benchmark*")
    config.addinivalue_line("python_functions", "benchmark_*")


def pytest_collect_file(parent, file_path):
    if file_path.suffix == ".py" and fnmatch.fnmatch(file_path.name, "benchmark_*.py"):
        return pytest.Module.from_parent(parent, path=file_path)
