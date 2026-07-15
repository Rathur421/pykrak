from types import ModuleType

from array_api_compat import (
    is_array_api_strict_namespace,
    is_numpy_namespace,
    is_torch_namespace,
    numpy,
    torch,
)
from array_api_compat.common._typing import Array


def array_interp(x, xp, fp, namespace: ModuleType):
    if is_numpy_namespace(namespace):
        return namespace.interp(x, xp, fp)
    else:
        return _array_interp(x, xp, fp, namespace)


def finfo_precision(dtype, namespace: ModuleType):
    if is_array_api_strict_namespace(namespace):
        return numpy.finfo(numpy.float64).precision
    if is_torch_namespace(namespace):
        return -torch.log10(torch.asarray(torch.finfo(dtype).resolution))
    else:
        return namespace.finfo(dtype).precision


def array_append(arr: Array, values: float, namespace: ModuleType):
    if is_numpy_namespace(namespace):
        return namespace.append(arr, values)
    else:
        return _array_append(arr, values, namespace)


def _array_append(arr: Array, values: Array, namespace: ModuleType):
    return namespace.concat((arr, namespace.expand_dims(values, axis=0)))


def _array_interp(x: Array, xp: Array, fp: Array, namespace) -> Array:
    """
    Equivalent to numpy.interp(x, xp, fp)

    Parameters
    ----------
    x : torch.Tensor
        The x-coordinates at which to evaluate the interpolated values. (1D or scalar)
    xp : torch.Tensor
        The x-coordinates of the data points. Must be increasing. (1D)
    fp : torch.Tensor
        The y-coordinates of the data points, same length as xp. (1D)

    Returns
    -------
    torch.Tensor
        The interpolated values, same shape as x.
    """
    # Find the indices of xp that are just before x
    indices = namespace.searchsorted(xp, x)

    # Clip indices to valid range [1, len(xp) - 1]
    # np.interp uses the first/last point for extrapolation outside [xp[0], xp[-1]]
    right_index = namespace.clip(indices, 1, xp.shape[0] - 1)
    left_index = right_index - 1

    # Extract the bracketing points (xp_l, fp_l) and (xp_r, fp_r)
    xp_l = xp[left_index]
    fp_l = fp[left_index]
    xp_r = xp[right_index]
    fp_r = fp[right_index]

    # Handle the cases where x is exactly at the boundary (to avoid division by zero if xp_r == xp_l)
    # This shouldn't happen if xp is strictly increasing, but for robustness:
    delta_xp = xp_r - xp_l
    is_same = delta_xp == 0

    # Linear interpolation formula: y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    # Calculate slopes
    slope = namespace.where(
        is_same, namespace.zeros_like(fp_l), (fp_r - fp_l) / delta_xp
    )

    # Calculate interpolated values
    interp_values = fp_l + slope * (x - xp_l)

    # Handle extrapolation (np.interp behavior: use fp[0] if x < xp[0] and fp[-1] if x > xp[-1])
    interp_values = namespace.where(
        x < xp[0], fp[0] * namespace.ones_like(x), interp_values
    )
    interp_values = namespace.where(
        x > xp[-1], fp[-1] * namespace.ones_like(x), interp_values
    )

    return (
        xp.squeeze(interp_values) if x.shape[0] == 1 and x.ndim == 1 else interp_values
    )


def torch_interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    Equivalent to numpy.interp(x, xp, fp)

    Parameters
    ----------
    x : torch.Tensor
        The x-coordinates at which to evaluate the interpolated values. (1D or scalar)
    xp : torch.Tensor
        The x-coordinates of the data points. Must be increasing. (1D)
    fp : torch.Tensor
        The y-coordinates of the data points, same length as xp. (1D)

    Returns
    -------
    torch.Tensor
        The interpolated values, same shape as x.
    """
    if x.ndim == 0:
        x = x.unsqueeze(0)

    # Find the indices of xp that are just before x
    indices = torch.searchsorted(xp, x)

    # Clamp indices to valid range [1, len(xp) - 1]
    # np.interp uses the first/last point for extrapolation outside [xp[0], xp[-1]]
    right_index = torch.clamp(indices, 1, len(xp) - 1)
    left_index = right_index - 1

    # Extract the bracketing points (xp_l, fp_l) and (xp_r, fp_r)
    xp_l = xp[left_index]
    fp_l = fp[left_index]
    xp_r = xp[right_index]
    fp_r = fp[right_index]

    # Handle the cases where x is exactly at the boundary (to avoid division by zero if xp_r == xp_l)
    # This shouldn't happen if xp is strictly increasing, but for robustness:
    delta_xp = xp_r - xp_l
    is_same = delta_xp == 0

    # Linear interpolation formula: y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    # Calculate slopes
    slope = torch.where(is_same, torch.zeros_like(fp_l), (fp_r - fp_l) / delta_xp)

    # Calculate interpolated values
    interp_values = fp_l + slope * (x - xp_l)

    # Handle extrapolation (np.interp behavior: use fp[0] if x < xp[0] and fp[-1] if x > xp[-1])
    interp_values = torch.where(x < xp[0], fp[0] * torch.ones_like(x), interp_values)
    interp_values = torch.where(x > xp[-1], fp[-1] * torch.ones_like(x), interp_values)

    return interp_values.squeeze() if x.shape[0] == 1 and x.ndim == 1 else interp_values
