import pytest


def pytest_addoption(parser):
    """Adds the command line flag."""
    parser.addoption(
        "--plots",
        action="store_true",
        default=False,
        help="Displays plots for fails during pytest execution.",
    )
    parser.addoption(
        "--backend",
        action="store",
        default="xp",
        help="Backend string. Can be np,torch,jax or cupy",
    )


@pytest.fixture
def plots_enabled(request):
    """Makes plots when a test fail
    Currently only working for the modal depth function
    """
    return request.config.getoption("--plots")


@pytest.fixture
def backend(request):
    """Request a specific backend to do computation
    Currently working:
    - array_api_strict # default
    - numpy
    - torch # super slow
    Would love to make it work (i.e. fell free to help us out)
    - jax
    - cupy
    - dask
    """
    backend_str = request.config.getoption("--backend")

    match backend_str:
        case "xp":
            import array_api_strict as xp

            return xp
        case "np" | "numpy":
            import numpy as np

            return np
        case "torch":
            import torch

            return torch
        case "dask":
            import dask.array as da

            return da
        case "jax":
            import jax
            import jax.numpy as jnp

            jax.config.update("jax_enable_x64", True)
            return jnp
        case "cupy":
            import cupy as cp

            return cp
        case _:
            raise ValueError(f"{backend_str} is not a known value")
