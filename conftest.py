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
        default="numpy",
        help="Backend string. Can be np,torch,jax or cupy",
    )


@pytest.fixture
def plots_enabled(request):
    """Fixture that returns True if --plots is passed."""
    return request.config.getoption("--plots")


@pytest.fixture
def backend(request):
    backend_str = request.config.getoption("--backend")

    match backend_str:
        case "np" | "numpy":
            import numpy as np

            backend = np
        case "torch":
            import torch

            backend = torch
        case "dask":
            import dask.array as da

            backend = da
        case "jax":
            import jax
            import jax.numpy as jnp

            jax.config.update("jax_enable_x64", True)
            backend = jnp
        case "cupy":
            import cupy as cp

            backend = cp
        case _:
            raise ValueError(f"{backend_str} is not a known value")
    return backend
