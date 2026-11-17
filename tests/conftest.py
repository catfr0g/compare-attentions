"""Explicit prerequisites for local engineering tests; never a science fallback."""

import os
import shutil
import sys

import pytest


def missing_windows_cpu_compiler(platform, cuda_available, compiler):
    """Mirror Inductor's Windows CXX/default-cl selection, without compiling."""
    return platform == "win32" and not cuda_available and shutil.which(compiler) is None


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_cpu_inductor: real compiled training; requires a host C++ compiler"
    )


def pytest_runtest_setup(item):
    if item.get_closest_marker("requires_cpu_inductor") is None:
        return
    import torch  # pylint: disable=import-outside-toplevel

    compiler = os.environ.get("CXX", "cl")
    if missing_windows_cpu_compiler(sys.platform, torch.cuda.is_available(), compiler):
        pytest.skip(
            "UNVERIFIED Windows CPU Inductor: selected CXX/default cl is absent from PATH; "
            "no compiler installation, backend fallback, or numerical-failure suppression"
        )
