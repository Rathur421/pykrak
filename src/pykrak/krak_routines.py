"""
Description:

Date:

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

from math import inf, pi
from types import ModuleType
from typing import Literal

from array_api_compat import size
from array_api_compat.common._helpers import array_namespace
from array_api_compat.common._typing import Array
from matplotlib import pyplot as plt

from pykrak import attn_pert as ap
from pykrak.backend_compat import array_append, array_interp, finfo_precision


def initialize(
    h_arr: Array,
    ind_arr: Array,
    z_arr: Array,
    omega2: float,
    cp_arr: Array,
    cs_arr: Array,
    rho_arr: Array,
    cp_top: float,
    cs_top: float,
    rho_top: float,
    cp_bott: float,
    cs_bott: float,
    rho_bott: float,
    c_low: float,
    c_high: float,
):
    """
    Initializes arrays defining difference equations.

    Args:
    h_arr - mesh size for the different layers
    ind_arr - index of the start of each layer in z_arr
    z_arr - array of depths , contains doubled interface points
    omega2 - squared angular frequency
    cp_arr - compressional wave speed (complex)
    cs_arr - shear wave speed (complex)
    rho_arr - density

    cp_top - compressional wave speed in top hs
    cs_top - shear wave speed in top hs
    rho_top - density in top hs

    rho_top = 0 for Pressure Release, rho_top = 1e10 for Rigid
    Same for rho_bott

    cp_bott - compressional wave speed in bottom hs
    cs_bott - shear wave speed in bottom hs
    rho_bott - density in bottom hs

    c_low is min phase speed
    c_high is max phase speed

    There should be the same number of points

    """
    xp = array_namespace(h_arr, ind_arr, z_arr, cp_arr, cs_arr, rho_arr)
    elastic_flag = False  # set to true if any media are elastic
    c_min = inf
    Nmedia = size(h_arr)  # number of layers
    n_points = size(z_arr)  # z_arr contains the doubled interface depths
    first_acoustic = -1
    last_acoustic = 0
    # convert float to array object
    cp_top = xp.asarray(cp_top, dtype=xp.double)
    cs_top = xp.asarray(cs_top, dtype=xp.double)
    rho_top = xp.asarray(rho_top, dtype=xp.double)
    cp_bott = xp.asarray(cp_bott, dtype=xp.double)
    cs_bott = xp.asarray(cs_bott, dtype=xp.double)
    rho_bott = xp.asarray(rho_bott, dtype=xp.double)
    c_low = xp.asarray(c_low, dtype=xp.double)
    c_high = xp.asarray(c_high, dtype=xp.double)

    # Allocate arrays
    b1 = xp.zeros(n_points, dtype=xp.double)
    b1c = xp.zeros(n_points, dtype=xp.double)
    b2 = xp.zeros(n_points, dtype=xp.double)
    b3 = xp.zeros(n_points, dtype=xp.double)
    b4 = xp.zeros(n_points, dtype=xp.double)
    rho_arr = xp.asarray(rho_arr, copy=True, dtype=xp.double)  # why???

    # Process each medium
    for medium in range(Nmedia):
        ii = ind_arr[medium]
        if medium == Nmedia - 1:
            Nii = size(z_arr[ii:])
        else:
            Nii = size(z_arr[ii : ind_arr[medium + 1]])

        # Load diagonals
        if xp.real(cs_arr[ii]) == 0.0:  # Acoustic medium
            c_min = min(c_min, xp.min(xp.real(cp_arr[ii : ii + Nii])))
            if first_acoustic == -1:
                first_acoustic = medium
            last_acoustic = medium
            b1[ii : ii + Nii] = -2.0 + h_arr[medium] ** 2 * xp.real(
                omega2 / (cp_arr[ii : ii + Nii]) ** 2
            )
            b1c[ii : ii + Nii] = xp.imag(
                omega2 / (cp_arr[ii : ii + Nii]) ** 2
                + 0j  # HACK: ensure type is complex (not implemented in torch)
            )

        else:  # Elastic medium
            elastic_flag = True
            two_h = 2.0 * h_arr[medium]
            for j in range(ii, ii + Nii):
                c_min = min(xp.real(cs_arr[j]), c_min)
                cp2 = xp.real(cp_arr[j] ** 2)
                cs2 = xp.real(cs_arr[j] ** 2)
                b1[j] = two_h / (rho_arr[j] * cs2)
                b2[j] = two_h / (rho_arr[j] * cp2)
                b3[j] = 4.0 * two_h * rho_arr[j] * cs2 * (cp2 - cs2) / cp2
                b4[j] = two_h * (cp2 - 2.0 * cs2) / cp2
                rho_arr[j] *= two_h * omega2

    if (rho_top == 0.0) or (
        rho_top == 1e10
    ):  # pressure or rigid, no need to overwrite c_high
        pass
    else:
        if cs_top != 0.0:
            elastic_flag = True
            c_min = min(c_min, xp.real(cs_top))
            c_high = min(c_high, xp.real(cs_top))
        else:
            c_min = min(c_min, xp.real(cp_top))

    if (rho_bott == 0.0) or (
        rho_bott == 1e10
    ):  # pressure or rigid, no need to overwrite c_high
        pass
    else:
        if cs_bott != 0.0:
            elastic_flag = True
            c_min = min(c_min, xp.real(cs_bott))
            c_high = min(c_high, xp.real(cs_bott))
        else:
            c_min = min(c_min, xp.real(cp_bott))
            c_high = min(c_high, xp.real(cp_bott))

    if elastic_flag:  # for Scholte wave
        c_min *= 0.85
    c_low = max(c_low, c_min)

    return (
        b1,
        b1c,
        b2,
        b3,
        b4,
        rho_arr,
        c_low,
        c_high,
        elastic_flag,
        first_acoustic,
        last_acoustic,
    )


def get_f_g(
    cp: float,
    cs: float,
    rho: float,
    x: float,
    omega2: float,
    mode_count: int,
    complex_flag: bool,
    xp: ModuleType,
):
    if rho == 0.0:  # Vacuum
        f = xp.asarray(1.0, dtype=xp.double)
        g = xp.asarray(0.0, dtype=xp.double)
        yV = xp.asarray([f, g, 0.0, 0.0, 0.0], dtype=xp.double)
    elif rho == 1e10:  # Rigid
        f = xp.asarray(0.0, dtype=xp.double)
        g = xp.asarray(1.0, dtype=xp.double)
        yV = xp.asarray([f, g, 0.0, 0.0, 0.0], dtype=xp.double)
    else:  # Acousto-elastic halfspace
        if xp.real(xp.asarray(cs, dtype=xp.double)) > 0.0:
            gammaS2 = x - (omega2 / xp.real(cs) ** 2)
            gammaP2 = x - (omega2 / xp.real(cp) ** 2)
            gammaS = xp.real(xp.sqrt(gammaS2))
            gammaP = xp.real(xp.sqrt(gammaP2))
            mu = rho * xp.real(cs) ** 2

            yV = xp.zeros(5, dtype=xp.double)
            yV[0] = (gammaS * gammaP - x) / mu
            yV[1] = ((gammaS2 + x) ** 2 - 4.0 * gammaS * gammaP * x) * mu
            yV[2] = 2.0 * gammaS * gammaP - gammaS2 - x
            yV[3] = gammaP * (x - gammaS2)
            yV[4] = gammaS * (gammaS2 - x)

            f = omega2 * yV[3]
            g = yV[1]
            if g > 0.0:
                mode_count += 1

        else:
            gammap = xp.sqrt(
                x - omega2 / cp**2 + 0j  # HACK: ensure type is complex (error in torch)
            )
            # print(f"{gammap = }")
            f = gammap
            g = xp.asarray(rho, dtype=xp.double)
            if not complex_flag:
                f = xp.real(f)
                g = xp.real(g)
            yV = xp.asarray([1e10, 1e10, 1e10, 1e10, 1e10], dtype=xp.double)
    return f, g, yV


def elastic_up(
    x: float,
    yV: Array,
    iPower: int,
    h: float,
    b1: Array,
    b2: Array,
    b3: Array,
    b4: Array,
    rho_arr: Array,
    Floor: float,
    Roof: float,
    iPowerR: int,
    iPowerF: int,
    xp: ModuleType,
):
    """
    Propagates up through a single elastic layer using compound matrix formulation.

    Parameters:
    x : float
        Trial eigenvalue, k2.
    yV : numpy.ndarray
        Solution of differential equation (initial values passed as input).
    iPower : int
        Power scaling factor, modified during computation.
    h : float
        layer thickness for the medium
    b1, b2, b3, b4, rho : array of the discretized wave equation arrays freom initialize
    Floor, Roof : float
        Scaling thresholds.
    iPowerR, iPowerF : int
        Scaling adjustments for `iPower`.
    Returns:
    tuple
        Updated yV, iPower.
    """
    # Initialize variables
    two_x = 2.0 * x
    two_h = 2.0 * h
    four_h_x = 4.0 * h * x
    j = size(b1) - 1
    xb3 = x * b3[j] - rho_arr[j]

    zV = xp.zeros(5, dtype=xp.double)
    zV[0] = yV[0] - 0.5 * (b1[j] * yV[3] - b2[j] * yV[4])
    zV[1] = yV[1] - 0.5 * (-rho_arr[j] * yV[3] - xb3 * yV[4])
    zV[2] = yV[2] - 0.5 * (two_h * yV[3] + b4[j] * yV[4])
    zV[3] = yV[3] - 0.5 * (xb3 * yV[0] + b2[j] * yV[1] - two_x * b4[j] * yV[2])
    zV[4] = yV[4] - 0.5 * (rho_arr[j] * yV[0] - b1[j] * yV[1] - four_h_x * yV[2])

    # Modified midpoint method
    N = size(b1)
    for ii in range(N - 1):
        j -= 1
        # print('EUP, ii, j, Yv', ii, j, yV)
        # print('b1[j], b2[j], b3[j], b4[j], rho_arr[j]', b1[j], b2[j], b3[j], b4[j], rho_arr[j])

        xV = xp.asarray(yV, copy=True, dtype=xp.double)
        yV = xp.asarray(zV, copy=True, dtype=xp.double)

        xb3 = x * b3[j] - rho_arr[j]

        zV[0] = xV[0] - (b1[j] * yV[3] - b2[j] * yV[4])
        zV[1] = xV[1] - (-rho_arr[j] * yV[3] - xb3 * yV[4])
        zV[2] = xV[2] - (two_h * yV[3] + b4[j] * yV[4])
        zV[3] = xV[3] - (xb3 * yV[0] + b2[j] * yV[1] - two_x * b4[j] * yV[2])
        zV[4] = xV[4] - (rho_arr[j] * yV[0] - b1[j] * yV[1] - four_h_x * yV[2])

        # Scale if necessary
        if ii != N - 2:
            if abs(zV[1]) < Floor:
                zV *= Roof
                yV *= Roof
                iPower -= iPowerR
            elif abs(zV[1]) > Roof:
                zV *= Floor
                yV *= Floor
                iPower -= iPowerF

    # Apply the standard filter at the terminal point
    yV = (xV + 2.0 * yV + zV) / 4.0

    # print('Yv final', yV)

    return yV, iPower


def elastic_down(
    x: float,
    yV: Array,
    iPower: int,
    h: float,
    b1: Array,
    b2: Array,
    b3: Array,
    b4: Array,
    rho_arr: Array,
    Floor: float,
    Roof: float,
    iPowerR: int,
    iPowerF: int,
    xp: ModuleType,
):
    """
    Propagates down through a single elastic layer using compound matrix formulation.

    Parameters:
    x : float
        Trial eigenvalue, k2.
    yV : numpy.ndarray
        Solution of differential equation (initial values passed as input).
    iPower : int
        Power scaling factor, modified during computation.
    h : float
        layer thickness for the medium
    b1, b2, b3, b4, rho : array of the discretized wave equation arrays freom initialize
    Floor, Roof : float
        Scaling thresholds.
    iPowerR, iPowerF : int
        Scaling adjustments for `iPower`.
    Returns:
    tuple
        Updated yV, iPower.
    """
    # Initialize variables
    two_x = 2.0 * x
    two_h = 2.0 * h
    four_h_x = 4.0 * h * x
    j = 0
    xb3 = x * b3[j] - rho_arr[0]

    zV = xp.zeros(5, dtype=xp.double)
    # print(yV.dtype, b1.dtype, b2.dtype, b3.dtype, b4.dtype, rho_arr.dtype)
    zV[0] = yV[0] + 0.5 * (b1[j] * yV[3] - b2[j] * yV[4])
    zV[1] = yV[1] + 0.5 * (-rho_arr[j] * yV[3] - xb3 * yV[4])
    zV[2] = yV[2] + 0.5 * (two_h * yV[3] + b4[j] * yV[4])
    zV[3] = yV[3] + 0.5 * (xb3 * yV[0] + b2[j] * yV[1] - two_x * b4[j] * yV[2])
    zV[4] = yV[4] + 0.5 * (rho_arr[j] * yV[0] - b1[j] * yV[1] - four_h_x * yV[2])

    # Modified midpoint method
    N = size(b1)
    for ii in range(N - 1):
        j += 1
        # print('EDOwn, ii, j, Yv', ii, j, yV)

        xV = xp.asarray(yV, copy=True, dtype=xp.double)
        yV = xp.asarray(zV, copy=True, dtype=xp.double)

        xb3 = x * b3[j] - rho_arr[j]

        zV[0] = xV[0] + (b1[j] * yV[3] - b2[j] * yV[4])
        zV[1] = xV[1] + (-rho_arr[j] * yV[3] - xb3 * yV[4])
        zV[2] = xV[2] + (two_h * yV[3] + b4[j] * yV[4])
        zV[3] = xV[3] + (xb3 * yV[0] + b2[j] * yV[1] - two_x * b4[j] * yV[2])
        zV[4] = xV[4] + (rho_arr[j] * yV[0] - b1[j] * yV[1] - four_h_x * yV[2])

        # Scale if necessary
        if ii != N - 1:
            if abs(zV[1]) < Floor:
                zV *= Roof
                yV *= Roof
                iPower -= iPowerR
            elif abs(zV[1]) > Roof:
                zV *= Floor
                yV *= Floor
                iPower -= iPowerF

    # Apply the standard filter at the terminal point
    yV = (xV + 2.0 * yV + zV) / 4.0

    # print('Yv final', yV)

    return yV, iPower


def get_bc_impedance(
    x,
    omega2,
    top_flag,
    cp,
    cs,
    rho,
    h_arr: Array,
    ind_arr,
    z_arr: Array,
    b1: Array,
    b2: Array,
    b3: Array,
    b4: Array,
    rho_arr: Array,
    first_acoustic,
    last_acoustic,
    mode_count: int,
    complex_flag: bool,
    xp: ModuleType,
):
    """
    Compute the impedance functions for the top and bottom halfspaces
    top_flag - True if the top boundary
    """
    iPower = 0
    Floor = 1e-50
    Roof = 1e50
    iPowerR = 50
    iPowerF = -50

    f, g, Yv = get_f_g(cp, cs, rho, x, omega2, mode_count, complex_flag, xp=xp)
    if top_flag:
        g = -g

    # Shoot through elastic layers if necessary
    if top_flag:
        if first_acoustic != 0:  # there are elastic layers on the surface
            for medium in range(first_acoustic):
                if medium == size(ind_arr) - 1:
                    i0, i1 = (
                        ind_arr[medium],
                        ind_arr[medium] + size(z_arr[ind_arr[medium] :]),
                    )
                else:
                    i0, i1 = ind_arr[medium], ind_arr[medium + 1]
                Yv, iPower = elastic_down(
                    x,
                    Yv,
                    iPower,
                    h_arr[medium],
                    b1[i0:i1],
                    b2[i0:i1],
                    b3[i0:i1],
                    b4[i0:i1],
                    rho_arr[i0:i1],
                    Floor,
                    Roof,
                    iPowerR,
                    iPowerF,
                    xp=xp,
                )
            f = omega2 * Yv[3]
            g = Yv[1]
    else:
        if last_acoustic != size(h_arr) - 1:  # there are elastic layers below
            if xp.all(Yv == 1e10):
                raise ValueError(
                    "Yv is not initialized, need to use rigid halfspace when shooting up through elastic layers"
                )
            for medium in range(size(h_arr) - 1, last_acoustic, -1):
                # print('medium', medium)
                if medium == size(ind_arr) - 1:
                    i0, i1 = (
                        ind_arr[medium],
                        ind_arr[medium] + size(z_arr[ind_arr[medium] :]),
                    )
                else:
                    i0, i1 = ind_arr[medium], ind_arr[medium + 1]
                Yv, iPower = elastic_up(
                    x,
                    Yv,
                    iPower,
                    h_arr[medium],
                    b1[i0:i1],
                    b2[i0:i1],
                    b3[i0:i1],
                    b4[i0:i1],
                    rho_arr[i0:i1],
                    Floor,
                    Roof,
                    iPowerR,
                    iPowerF,
                    xp,
                )
            f = omega2 * Yv[3]
            g = Yv[1]

    if not complex_flag:
        f = xp.real(f)
        g = xp.real(g)
    return f, g, iPower, mode_count


def acoustic_layers(
    x: float,
    f: float,
    g: float,
    iPower: int,
    ind_arr: Array,
    h_arr: Array,
    z_arr: Array,
    b1: Array,
    rho_arr: Array,
    CountModes: int,
    mode_count: int,
    first_acoustic,
    last_acoustic,
):
    r"""
    Shoot through acoustic layers
    from the bottom, where there is a boundary condition
    f p + g \dv{p}{z} / rho(z-) = 0
    x - float 64, eigenvalue is x = kr^2
    f, g are impedance functions at the bottom (float 64)
    iPower - int, power of 10 for scaling the shooting solutions
    h_arr - mesh array

    """

    # Parameters
    iPowerF = -50
    Roof = 1.0e50
    Floor = 1.0e-50

    # Loop over successive acoustic media starting at the end and going up
    for Medium in range(last_acoustic, first_acoustic - 1, -1):
        hMedium = h_arr[Medium]
        # print('hMedium', hMedium)
        if Medium == size(ind_arr) - 1:
            z_layer = z_arr[ind_arr[Medium] :]
        else:
            z_layer = z_arr[ind_arr[Medium] : ind_arr[Medium + 1]]
        NMedium = size(z_layer)  # includes interface points
        h2k2 = hMedium**2 * x

        ii = ind_arr[Medium] + NMedium - 1
        rhoMedium = rho_arr[
            ind_arr[Medium]
        ]  # Density is homogeneous using value at the top of each medium

        p1 = -2.0 * g
        p2 = (b1[ii] - h2k2) * g - 2.0 * hMedium * f * rhoMedium

        # Shoot (towards surface) through a single medium
        for ii in range(ind_arr[Medium] + NMedium - 2, ind_arr[Medium] - 1, -1):
            p0 = p1
            p1 = p2
            p2 = (h2k2 - b1[ii]) * p1 - p0

            if CountModes:
                if p0 * p1 <= 0.0:
                    mode_count += 1

            if abs(p2) > Roof:  # Scale if necessary
                p0 *= Floor
                p1 *= Floor
                p2 *= Floor
                iPower -= iPowerF

        # Update f and g
        rhoMedium = rho_arr[ind_arr[Medium]]  # Density at the top of the layer
        f = -(p2 - p0) / (2.0 * hMedium * rhoMedium)
        g = -p1
    return f, g, iPower, mode_count


def funct(x, args, xp):
    """
    funct(x) = 0 is the dispersion relation
    (funct is the difference between the impedance for the solution obtained
    by shooting upwards from the bottom boundary
    with eigenvalue x and the impedance for the solution
    obtained by shooting downwards from the top boundary through
    any surface elastic layers to the first acoustic medium interface)

    ev_mat[i,j] is the jth mode for the ith mesh used in the richardson extrap.
    iset is the current index of the mesh to use in ev_mat

    """
    (
        omega2,
        ev_mat,
        iset,
        h_arr,
        ind_arr,
        z_arr,
        N_arr,
        cp_top,
        cs_top,
        rho_top,
        cp_bott,
        cs_bott,
        rho_bott,
        b1,
        b1c,
        b2,
        b3,
        b4,
        rho_arr,
        c_low,
        c_high,
        elastic_flag,
        first_acoustic,
        last_acoustic,
        mode,
        CountModes,
        mode_count,
    ) = args
    iPowerR = 50
    iPowerF = -50
    Roof = 1.0e50
    Floor = 1.0e-50

    mode_count = 0

    # shoot up from the bottom
    f_bott, g_bott, iPower, mode_count = get_bc_impedance(
        x,
        omega2,
        False,
        cp_bott,
        cs_bott,
        rho_bott,
        h_arr,
        ind_arr,
        z_arr,
        b1,
        b2,
        b3,
        b4,
        rho_arr,
        first_acoustic,
        last_acoustic,
        mode_count,
        False,
        xp,
    )
    # print('x, f_bott, g_bott', x, f_bott, g_bott)
    f, g, iPower, mode_count = acoustic_layers(
        x,
        f_bott.real,
        g_bott.real,
        iPower,
        ind_arr,
        h_arr,
        z_arr,
        b1,
        rho_arr,
        CountModes,
        mode_count,
        first_acoustic,
        last_acoustic,
    )
    # print('after al', f, g)
    # print('eig x, f, g, iPower after AcousticLayers = ', x, f, g, iPower)
    f_top, g_top, iPower_top, mode_count = get_bc_impedance(
        x,
        omega2,
        True,
        cp_top,
        cs_top,
        rho_top,
        h_arr,
        ind_arr,
        z_arr,
        b1,
        b2,
        b3,
        b4,
        rho_arr,
        first_acoustic,
        last_acoustic,
        mode_count,
        False,
        xp,
    )

    # print('at top', f_top, g_top)

    Delta = (f * g_top - g * f_top).real
    iPower = iPower + iPower_top

    if g.real * Delta > 0.0:
        mode_count = mode_count + 1

    # Deflate previous roots
    # NOTE: Modes are indexed from 1!
    # so the loop below deflates the previously found roots

    if (mode > 1) and (len(ind_arr) > last_acoustic - first_acoustic + 1):
        for j in range(mode - 1):
            Delta = Delta / (x - ev_mat[iset, j])

            # Scale if necessary
            while xp.abs(Delta) < Floor and xp.abs(Delta) > 0.0:
                Delta = Roof * Delta
                iPower = iPower - iPowerR

            while xp.abs(Delta) > Roof:
                Delta = Floor * Delta
                iPower = iPower - iPowerF
    return Delta, iPower, mode_count


def bisection(x_min, x_max, M, args, xp):
    """
    Returns isolating intervals (xL, xR) for each eigenvalue
    in the given interval [x_min, x_max].
    """
    max_bisections = 50

    # Initialize boundaries
    x_l = x_min * xp.ones(M, dtype=xp.double)
    x_r = x_max * xp.ones(M, dtype=xp.double)

    # Compute the initial number of modes at x_max
    count_modes = True
    mode_count = 0
    margs = args + (1, count_modes, 0)
    delta, i_power, mode_count = funct(x_max, margs, xp)
    n_zeros_initial = mode_count

    if M == 1:
        return x_l, x_r

    # Loop over eigenvalues to refine intervals
    for mode in range(1, M):
        mind = mode - 1
        if x_l[mind] == x_min:
            x2 = x_r[mind]
            x1 = max(xp.max(x_l[mind + 1 : M]), x_min)

            for _ in range(max_bisections):
                x = x1 + (x2 - x1) / 2
                margs = args + (mode, True, 0)
                delta, i_power, mcount = funct(x, margs, xp)
                n_zeros = mcount - n_zeros_initial

                if n_zeros < mode:  # new right bdry
                    x2 = x
                    x_r[mind] = x
                else:  # new left bdry
                    x1 = x
                    if x_r[n_zeros] >= x:
                        x_r[n_zeros] = x
                    if x_l[n_zeros - 1] <= x:
                        x_l[n_zeros - 1] = x

                if x_l[mode - 1] != x_min:
                    break

    return x_l, x_r


def solve1(args, h_v, xp):
    """
    Solve for eigenvalues using Sturm sequences and Brent's method.

    Parameters:
        omega2 : float
            Frequency squared.
        c_high : float
            Upper limit of phase speed.
        c_low : float
            Lower limit of phase speed.
        nfreq : int
            Number of frequencies.
        file_root : str
            Root name for output files.
        prt_file : file-like
            Output file for printing.
        title : str
            Title for output.
        i_set, ifreq, iprof : int
            Indices for current dataset.
        mode_count_func : callable
            Function to count modes.
        erro_out_func : callable
            Function to handle errors.
        bisection_func : callable
            Function to initialize bounds for eigenvalue refinement.
        zbrentx_func : callable
            Function to refine eigenvalues.
        evmat, extrap, k, vg : ndarray, optional
            Arrays to hold results. If provided, will be updated.
        n : ndarray
            Array of mode counts per acoustic layer.
        first_acoustic, last_acoustic : int
            Indices for the acoustic layers.
    """

    (
        omega2,
        ev_mat,
        iset,
        h_arr,
        ind_arr,
        z_arr,
        N_arr,
        cp_top,
        cs_top,
        rho_top,
        cp_bott,
        cs_bott,
        rho_bott,
        b1,
        b1c,
        b2,
        b3,
        b4,
        rho_arr,
        c_low,
        c_high,
        elastic_flag,
        first_acoustic,
        last_acoustic,
    ) = args

    # Initialization
    # Determine the number of modes
    x_min = 1.00001 * omega2 / c_high**2

    count_modes = True
    margs = args + (1, count_modes, 0)

    delta, i_power, mode_count = funct(x_min, margs, xp)
    m = mode_count

    # Check upper bound
    x_max = omega2 / c_low**2
    delta, i_power, mode_count = funct(x_max, margs, xp)
    m -= mode_count

    if m == 0:
        return ev_mat, m

    n_total = N_arr[first_acoustic : last_acoustic + 1].sum()
    if m > n_total / 5:
        print(f"Approximate number of modes = {m}")
        print(
            "Warning in KRAKEN - Solve1 : Mesh too coarse to sample the modes adequately"
        )

    # Initialize bounds for eigenvalue refinement
    x_l, x_r = bisection(x_min, x_max, m, args, xp)

    # Refine each eigenvalue
    count_modes = False
    # print('m', m)
    for mode in range(1, m + 1):
        x1 = x_l[mode - 1]
        x2 = x_r[mode - 1]
        eps = abs(x2) * 10.0 ** (2.0 - finfo_precision(x2.dtype, namespace=xp))

        margs = args + (mode, False, 0)
        x = zbrent(x1, x2, eps, margs, xp=xp)

        ev_mat[iset, mode - 1] = x
    # ev_mat = xp.asarray(ev_mat[:, :m],copy=True)
    #
    return ev_mat, m


def solve2(args, h_v, M, xp):
    """
    h_v is array of mesh sizes
    """

    max_iteration = 2000

    (
        omega2,
        ev_mat,
        iset,
        h_arr,
        ind_arr,
        z_arr,
        N_arr,
        cp_top,
        cs_top,
        rho_top,
        cp_bott,
        cs_bott,
        rho_bott,
        b1,
        b1c,
        b2,
        b3,
        b4,
        rho_arr,
        c_low,
        c_high,
        elastic_flag,
        first_acoustic,
        last_acoustic,
    ) = args
    CountModes = False
    mode_count = 0  # doesn't matter

    # inital guess
    x = omega2 / c_low**2

    for mode in range(1, M + 1):
        # Initial guess for x
        imode = mode - 1
        x *= 1.00001

        # use extrapolation to produce initial guess if possible
        if iset >= 1:
            p = xp.asarray(
                ev_mat[:iset, imode], copy=True, dtype=xp.double
            )  # load previous mesh estimates

            if iset >= 2:  # extrapolation
                for ii in range(iset - 1):
                    for j in range(iset - ii - 1):
                        x1 = h_v[j] ** 2
                        x2 = h_v[j + ii + 1] ** 2

                        p[j] = (
                            (h_v[iset - 1] ** 2 - x2) * p[j]
                            - (h_v[iset - 1] ** 2 - x1) * p[j + 1]
                        ) / (x1 - x2)
                x = p[0]

        # Calculate tolerance for root finder
        # tolerance = xp.abs(x) * b1.size * 10.0**(1.0 - 15) # 15 is precision for float 64
        tolerance = (
            xp.abs(x)
            * size(b1)
            * 10.0 ** (1.0 - finfo_precision(x.dtype, namespace=xp))
        )

        # Use secant method to refine eigenvalue
        margs = args + (mode, CountModes, mode_count)
        x, iteration, error_message = root_finder_secant_real(
            x, tolerance, max_iteration, funct, margs, xp
        )
        if error_message != "":
            print(f"Warning in Solve2 - RootFinderSecant: {error_message}")
            print(f"iset, mode = {iset}, {mode}")
            x = xp.finfo(x).tiny

        ev_mat[iset, imode] = x

        # Discard modes outside user-specified spectrum
        if omega2 / c_high**2 > x:
            # ev_mat = xp.asarray(ev_mat[:, :mode],copy=True)
            break
    return ev_mat, mode


def root_finder_secant_real(x2, tolerance, max_iterations, func, args, xp):
    """
    Secant method for finding roots of a real-valued function.
    Used for funct, so the function returns three things

    Parameters:
        x2 (float): Initial guess for the root (updated in-place).
        tolerance (float): Error bound for convergence.
        max_iterations (int): Maximum number of allowable iterations.
        func (callable): Function that computes the value of the function and an integer power.

    Returns:
        x2 (float): Estimated root.
        iteration (int): Number of iterations performed.
        error_message (str): Empty unless there was a failure to converge.
    """
    error_message = ""
    if tolerance <= 0.0:
        return x2, 0, "Non-positive tolerance specified"

    x1 = x2 + 10.0 * tolerance
    f1, i_power1, _ = func(x1, args, xp)

    for iteration in range(1, max_iterations + 1):
        x0, f0, i_power0 = x1, f1, i_power1
        x1 = x2
        f1, i_power1, _ = func(x1, args, xp)

        c_num = f1 * (x1 - x0)
        c_den = f1 - f0 * 10.0 ** (i_power0 - i_power1)

        if abs(c_num) >= abs(c_den * x1):
            shift = 0.1 * tolerance
        else:
            shift = c_num / c_den

        x2 = x1 - shift

        if abs(x2 - x1) + abs(x2 - x0) < tolerance:
            return x2, iteration, ""

    return x2, max_iterations, "Failure to converge in RootFinderSecant"


def root_finder_secant_complex(x2, tolerance, max_iterations, func, args):
    """
    Secant method for finding roots of a complex-valued function.

    Parameters:
        x2 (complex): Initial guess for the root (updated in-place).
        tolerance (float): Error bound for convergence.
        max_iterations (int): Maximum number of allowable iterations.
        func (callable): Function that computes the value of the function and an integer power.

    Returns:
        x2 (complex): Estimated root.
        iteration (int): Number of iterations performed.
        error_message (str): Empty unless there was a failure to converge.
    """
    error_message = ""

    if tolerance <= 0.0:
        return x2, 0, "Non-positive tolerance specified"

    x1 = x2 + 100.0 * tolerance
    f1, i_power1, _ = func(x1, args)

    for iteration in range(1, max_iterations + 1):
        x0, f0, i_power0 = x1, f1, i_power1
        x1 = x2
        f1, i_power1, _ = func(x1, args)

        c_num = f1 * (x1 - x0)
        c_den = f1 - f0 * 10.0 ** (i_power0 - i_power1)

        if abs(c_num) >= abs(c_den * x1):
            shift = 0.1 * tolerance
        else:
            shift = c_num / c_den

        x2 = x1 - shift

        if abs(x2 - x1) + abs(x2 - x0) < tolerance:
            return x2, iteration, ""

    return x2, max_iterations, "Failure to converge in RootFinderSecant"


def zbrent(a, b, t, args, xp):
    """
    Licensing:

      This code is distributed under the GNU LGPL license.

    Modified:

      08 April 2023

    Author:

      Original FORTRAN77 version by Richard Brent
      Python version by John Burkardt
      Numba-ized version specific for the layered S-L problem by Hunter Akins
    """
    machep = 1e-16

    sa = a
    sb = b
    fa, _, _ = funct(sa, args, xp)
    fb, _, _ = funct(sb, args, xp)

    c = sa
    fc = fa
    e = sb - sa
    d = e

    while True:
        if abs(fc) < abs(fb):
            sa = sb
            sb = c
            c = sa
            fa = fb
            fb = fc
            fc = fa

        tol = 2.0 * machep * abs(sb) + t
        m = 0.5 * (c - sb)

        if abs(m) <= tol or fb == 0.0:
            break

        if abs(e) < tol or abs(fa) <= abs(fb):
            e = m
            d = e

        else:
            s = fb / fa

            if sa == c:
                p = 2.0 * m * s
                q = 1.0 - s

            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (sb - sa) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)

            if 0.0 < p:
                q = -q

            else:
                p = -p

            s = e
            e = d

            if 2.0 * p < 3.0 * m * q - abs(tol * q) and p < abs(0.5 * s * q):
                d = p / q
            else:
                e = m
                d = e

        sa = sb
        fa = fb

        if tol < abs(d):
            sb = sb + d
        elif 0.0 < m:
            sb = sb + tol
        else:
            sb = sb - tol

        fb, _, _ = funct(sb, args, xp)

        if (0.0 < fb and 0.0 < fc) or (fb <= 0.0 and fc <= 0.0):
            c = sa
            fc = fa
            e = sb - sa
            d = e

    value = sb
    return value


def inverse_iter(d: Array, e: Array, xp: ModuleType, max_iteration=2000):
    """
    Perform inverse iteration to compute an eigenvector.

    Parameters:
    - d (numpy.ndarray): Diagonal elements of the matrix (1D array of size N).
    - e (numpy.ndarray): Off-diagonal elements (1D array of size N+1).
    - max_iteration (int): Maximum number of iterations (default: 100).

    Returns:
    - eigenvector (numpy.ndarray): Approximated eigenvector (1D array of size N).
    - i_error (int): error flag (0 if successful, -1 if convergence fails).
    """
    # Initialize variables
    i_error = 0
    N = size(d)

    # Compute the (infinity) norm of the matrix
    norm = xp.sum(xp.abs(d)) + xp.sum(xp.abs(e[1:N]))

    # Small thresholds
    eps3 = 100.0 * xp.finfo(xp.double).eps * norm
    uk = N
    eps4 = uk * eps3
    uk = eps4 / xp.sqrt(xp.asarray(uk, dtype=xp.double))
    # print('uk', uk)

    # Temporary arrays
    rv1 = xp.zeros(N, dtype=xp.double)
    rv2 = xp.zeros(N, dtype=xp.double)
    rv3 = xp.zeros(N, dtype=xp.double)
    rv4 = xp.zeros(N, dtype=xp.double)

    # elimination with interchanges
    xu = 1.0
    u = d[0]
    v = e[1]

    for i in range(1, N):
        if abs(e[i]) >= abs(u):
            xu = u / e[i]
            rv4[i] = xu
            rv1[i - 1] = e[i]
            rv2[i - 1] = d[i]
            rv3[i - 1] = e[i + 1]
            u = v - xu * rv2[i - 1]
            v = -xu * rv3[i - 1]
        else:
            xu = e[i] / u
            rv4[i] = xu
            rv1[i - 1] = u
            rv2[i - 1] = v
            rv3[i - 1] = 0.0
            u = d[i] - xu * v
            v = e[i + 1]
    if u == 0.0:
        u = eps3

    rv3[N - 2] = 0.0
    rv1[N - 1] = u
    rv2[N - 1] = 0.0
    rv3[N - 1] = 0.0

    # Initialize eigenvector
    eigenvector = uk * xp.ones(N, dtype=xp.double)

    # Main loop of inverse iteration
    for iteration in range(max_iteration):
        # Back substitution
        for i in range(N - 1, -1, -1):
            eigenvector[i] = (eigenvector[i] - u * rv2[i] - v * rv3[i]) / rv1[i]
            v = u
            u = eigenvector[i]

        # Compute norm of vector and test for convergence
        norm = xp.sum(xp.abs(eigenvector))
        if norm >= 1.0:
            return eigenvector, i_error  # Convergence achieved

        # Scale the vector down
        xu = eps4 / norm
        eigenvector = eigenvector * xu

        # Forward elimination
        for i in range(1, N):
            u = eigenvector[i]

            if rv1[i - 1] == e[i]:
                u = eigenvector[i - 1]
                eigenvector[i - 1] = eigenvector[i]

            eigenvector[i] = u - rv4[i] * eigenvector[i - 1]

    # If we fall through the loop, convergence failed
    i_error = -1
    return eigenvector, i_error


def normalize(phi, iTurningPoint, x, args, z, xp):
    """
    Normalize the eigenvector phi and compute perturbations from attenuation and group velocity.
    """
    # print('x', x)
    (
        omega2,
        ev_mat,
        iset,
        h_arr,
        ind_arr,
        z_arr,
        Ng_arr,
        cp_top,
        cs_top,
        rho_top,
        cp_bott,
        cs_bott,
        rho_bott,
        b1,
        b1c,
        b2,
        b3,
        b4,
        rho_arr,
        c_low,
        c_high,
        elastic_flag,
        first_acoustic,
        last_acoustic,
        M,
        sigma_arr,
    ) = args
    mode_count = 0
    count_modes = False

    # Initialization
    SqNorm = 0.0
    Perturbation_k = 0.0 + 0.0j
    sg = 0.0

    # Top half-space contribution
    if rho_top != 0.0 and rho_top != 1e10:
        Del = 1j * xp.imag(
            xp.sqrt(xp.asarray(x - omega2 / cp_top**2, dtype=xp.complex128))
        )
        Perturbation_k -= Del * phi[0] ** 2 / rho_top
        sg += (
            phi[0] ** 2
            / (2 * xp.sqrt(x - xp.real(omega2 / cp_top**2)))
            / xp.real(rho_top * cp_top**2)
        )

    # Volume contribution
    L = ind_arr[first_acoustic] - 1
    j = 0

    for Medium in range(first_acoustic, last_acoustic + 1):
        L += 1  # L is index of first value in layer
        rhoMedium = rho_arr[L]
        rho_omega_h2 = rhoMedium * omega2 * h_arr[Medium] ** 2
        # print('rho, h', rhoMedium, h_arr[Medium])

        # Top interface
        SqNorm += 0.5 * h_arr[Medium] * phi[j] ** 2 / rhoMedium
        sg += 0.5 * h_arr[Medium] * (b1[L] + 2.0) * phi[j] ** 2 / rho_omega_h2
        Perturbation_k += 0.5 * h_arr[Medium] * 1j * b1c[L] * phi[j] ** 2 / rhoMedium

        # Medium
        L1 = L + 1
        L += Ng_arr[Medium] - 1
        j1 = j + 1  #
        j += Ng_arr[Medium] - 1

        SqNorm += h_arr[Medium] * xp.sum(phi[j1:j] ** 2) / rhoMedium
        sg += h_arr[Medium] * xp.sum((b1[L1:L] + 2.0) * phi[j1:j] ** 2) / rho_omega_h2
        Perturbation_k += (
            h_arr[Medium] * 1j * xp.sum(b1c[L1:L] * phi[j1:j] ** 2) / rhoMedium
        )

        # Bottom interface
        SqNorm += 0.5 * h_arr[Medium] * phi[j] ** 2 / rhoMedium
        sg += 0.5 * h_arr[Medium] * (b1[L] + 2.0) * phi[j] ** 2 / rho_omega_h2
        Perturbation_k += 0.5 * h_arr[Medium] * 1j * b1c[L] * phi[j] ** 2 / rhoMedium

    # Bottom half-space contribution
    if rho_bott != 0 and rho_bott != 1e10:
        Del = 1j * xp.imag(
            xp.sqrt(xp.asarray(x - omega2 / cp_bott**2, dtype=xp.complex128))
        )
        Perturbation_k -= Del * phi[j] ** 2 / rho_bott
        sg += (
            phi[j] ** 2
            / (2 * xp.sqrt(x - xp.real(omega2 / cp_bott**2)))
            / (rho_bott * xp.real(cp_bott) ** 2)
        )

    # Deriv of top admittance
    x1 = 0.9999999 * x
    x2 = 1.0000001 * x

    f_top1, g_top1, iPower_top, mode_count = get_bc_impedance(
        x1,
        omega2,
        True,
        cp_top,
        cs_top,
        rho_top,
        h_arr,
        ind_arr,
        z_arr,
        b1,
        b2,
        b3,
        b4,
        rho_arr,
        first_acoustic,
        last_acoustic,
        mode_count,
        False,
        xp,
    )
    f_top2, g_top2, iPower_top, mode_count = get_bc_impedance(
        x2,
        omega2,
        True,
        cp_top,
        cs_top,
        rho_top,
        h_arr,
        ind_arr,
        z_arr,
        b1,
        b2,
        b3,
        b4,
        rho_arr,
        first_acoustic,
        last_acoustic,
        mode_count,
        False,
        xp,
    )
    drho_dx = 0.0
    if g_top1 != 0:
        drho_dx = xp.real((f_top2 / g_top2 - f_top1 / g_top1)) / (x2 - x1)

    # Bott
    f_bott1, g_bott1, iPower_bott, mode_count = get_bc_impedance(
        x1,
        omega2,
        False,
        cp_bott,
        cs_bott,
        rho_bott,
        h_arr,
        ind_arr,
        z_arr,
        b1,
        b2,
        b3,
        b4,
        rho_arr,
        first_acoustic,
        last_acoustic,
        mode_count,
        False,
        xp,
    )

    f_bott2, g_bott2, iPower_bott, mode_count = get_bc_impedance(
        x2,
        omega2,
        False,
        cp_bott,
        cs_bott,
        rho_bott,
        h_arr,
        ind_arr,
        z_arr,
        b1,
        b2,
        b3,
        b4,
        rho_arr,
        first_acoustic,
        last_acoustic,
        mode_count,
        False,
        xp,
    )
    deta_dx = 0.0
    if g_bott1 != 0:
        deta_dx = xp.real((f_bott2 / g_bott2 - f_bott1 / g_bott1)) / (x2 - x1)
    rn = SqNorm - drho_dx * phi[0] ** 2 + deta_dx * phi[j] ** 2
    if rn < 0.0:
        rn = -rn

    # print('Rn', rn)
    scale_factor = 1 / xp.sqrt(rn)
    if phi[iTurningPoint] < 0.0:
        scale_factor = -scale_factor
    w = phi * scale_factor
    sg = sg * scale_factor**2 * xp.sqrt(omega2) / xp.sqrt(x)
    # print('ug before', 1/(sg_before* scale_factor**2 * xp.sqrt(omega2) / xp.sqrt(x)))
    # print('ug afer', 1/sg)
    Perturbation_k = Perturbation_k * scale_factor**2
    ug = 1.0 / sg

    scattering_k = scatterloss(args, w, x, xp)

    Perturbation_k = Perturbation_k + scattering_k

    return w, Perturbation_k, sg, ug


def scatter_root(z, xp):
    z = xp.asarray(z)
    if xp.real(z) >= 0:
        return xp.sqrt(z)
    else:
        return -xp.sqrt(-z) * 1j


def kup_ing(sigma, eta1_sq, rho1, eta2_sq, rho2, P, U, xp):
    """
    Kuperman ingenito imaginary part of wavenumber for boundary roughnesss
    sigma - rms amplitude of the boundary roughness
    """
    ret = 0.0 + 0.0j
    if sigma == 0.0:
        return ret

    eta1 = scatter_root(eta1_sq, xp)
    eta2 = scatter_root(eta2_sq, xp)
    delta = rho1 * eta2 + rho2 * eta1
    if delta == 0.0:
        return ret
    else:
        a11 = (
            0.5 * (eta1_sq - eta2_sq)
            - (rho2 * eta1_sq - rho1 * eta2_sq) * (eta1 + eta2) / delta
        )
        a12 = 1j * (rho2 - rho1) ** 2 * eta1 * eta2 / delta
        a21 = -1j * (rho2 * eta1_sq - rho1 * eta2_sq) ** 2 / (rho1 * rho2 * delta)
        a22 = (
            0.5 * (eta1_sq - eta2_sq)
            + (rho2 - rho1) * eta1 * eta2 * (eta1 + eta2) / delta
        )

        ret = -(sigma**2) * (-a21 * P**2 + (a11 - a22) * P * U + a12 * U**2)
        return ret


def scatterloss(args, phi, x, xp):
    """
    perturbation_k - complex wavenumber, already includes volume attenuation from
    perturbation theory
    phi - mode shape
    x - eigenvalue
    """
    (
        omega2,
        ev_mat,
        iset,
        h_arr,
        ind_arr,
        z_arr,
        Ng_arr,
        cp_top,
        cs_top,
        rho_top,
        cp_bott,
        cs_bott,
        rho_bott,
        b1,
        b1c,
        b2,
        b3,
        b4,
        rho_arr,
        c_low,
        c_high,
        elastic_flag,
        first_acoustic,
        last_acoustic,
        M,
        sigma_arr,
    ) = args

    j = 0  # this index is for the phi depth grid which does not have doubled interface points

    scattering_perturbation_k = 0.0

    if size(b1) != xp.sum(Ng_arr):
        raise ValueError("b1.size != sum(Ng_arr), check the implementation")

    for i in range(first_acoustic, last_acoustic + 2):  # iterate over interfaces
        # print('Acoustic layer index', i)
        if i == first_acoustic:  # do top interface
            if rho_top == 0:  # Vacuum
                rho1 = 1e-9
                eta1_sq = 1.0
                rhoInside = rho_arr[ind_arr[i]]
                # print(ind_arr[i])
                # print('rhoInside', rhoInside)
                U = phi[1] / h_arr[i] / rhoInside
            elif rho_top == 1e10:  # Rigid
                rho1 = 1e9  # I used 1e10 to designate rigid, but to imitate Kraken I use 1e9 for the scatter loss
                eta1_sq = 1.0
                U = 0.0
            else:
                rho1 = rho_top
                eta1_sq = x - omega2 / cp_top**2
                U = xp.sqrt(eta1_sq) * phi[0] / rho1
        else:  # use rho1, etaSq, U layer
            h2 = h_arr[i - 1] ** 2
            j = (
                j + Ng_arr[i - 1] - 1
            )  # this is the index of the interface in the phi depth grid
            L = (
                ind_arr[i - 1] + Ng_arr[i - 1] - 1
            )  # point at interface from above in the numerical mesh
            rho1 = rho_arr[L]  # density at the top of the layer
            eta1_sq = (2.0 + b1[L]) / h2 - x
            U = (-phi[j - 1] - 0.5 * (b1[L] - h2 * x) * phi[j]) / (h_arr[i - 1] * rho1)

        # now get eta2sq, rho2,
        if i == last_acoustic + 1:  # bottom halfpsace
            if rho_bott == 0:  # Vacuum
                rho2 = 1e-9
                eta2_sq = 1.0
            elif rho_bott == 1e10:  # Rigid
                rho2 = 1e9  #
                eta2_sq = 1.0
            else:
                rho2 = rho_bott
                eta2_sq = omega2 / cp_bott**2 - x

        else:
            rho2 = rho_arr[ind_arr[i]]
            eta2_sq = (2.0 + b1[ind_arr[i]]) / h_arr[i] ** 2 - x

        phiC = phi[j]  # mode shape at the interface
        # print('rho1', rho1, 'rho2', rho2, 'sigma', sigma_arr[i], 'eta_sq', eta1_sq, eta2_sq, 'phiC', phiC, 'U', U)
        scattering_perturbation_k = scattering_perturbation_k + kup_ing(
            sigma_arr[i], eta1_sq, rho1, eta2_sq, rho2, phiC, U, xp
        )
    return scattering_perturbation_k


def get_phi(args, xp):
    (
        omega2,
        ev_mat,
        iset,
        h_arr,
        ind_arr,
        z_arr,
        Ng_arr,
        cp_top,
        cs_top,
        rho_top,
        cp_bott,
        cs_bott,
        rho_bott,
        b1,
        b1c,
        b2,
        b3,
        b4,
        rho_arr,
        c_low,
        c_high,
        elastic_flag,
        first_acoustic,
        last_acoustic,
        M,
        sigma_arr,
    ) = args

    CountModes = False
    mode_count = 0  # doesn't matter

    num_ac_layers = last_acoustic - first_acoustic + 1
    N_total1 = xp.sum(Ng_arr[first_acoustic : last_acoustic + 1]) - (num_ac_layers) + 1
    # print('N_total1', N_total1)

    for Medium in range(first_acoustic, last_acoustic + 1):
        h_rho = (
            h_arr[Medium] * rho_arr[ind_arr[Medium]]
        )  # density at the top of each layer
        if Medium == size(ind_arr) - 1:
            z_layer = z_arr[ind_arr[Medium] :]
        else:
            z_layer = z_arr[ind_arr[Medium] : ind_arr[Medium + 1]]
        # print('Nmedium', z_layer.size-1)
        if Medium == first_acoustic:
            e = 1.0 / h_rho * xp.ones(size(z_layer), dtype=xp.double)
            e[0] = 0.0
            z = z_layer
        else:
            e = xp.concatenate(
                (e, 1.0 / h_rho * xp.ones(size(z_layer) - 1, dtype=xp.double))
            )
            z = xp.concatenate((z, z_layer[1:]))  # get rid of the doubled points

    e = array_append(e, 1.0 / h_rho, namespace=xp)
    # Main loop: for each eigenvalue call InverseIteration to get eigenvector
    d = xp.zeros(size(z), dtype=xp.double)
    if size(z) != N_total1:
        raise Exception("z.size != N_total1, check the implementation")
    phi = xp.zeros((size(z), M), dtype=xp.double)
    pert_k_arr = xp.zeros(M, dtype=xp.complex128)
    sgs_arr = xp.zeros(M, dtype=xp.double)
    ugs_arr = xp.zeros(M, dtype=xp.double)

    for mode in range(1, M + 1):
        mind = mode - 1
        x = ev_mat[iset, mind]

        f_top, g_top, iPower_top, mode_count = get_bc_impedance(
            x,
            omega2,
            True,
            cp_top,
            cs_top,
            rho_top,
            h_arr,
            ind_arr,
            z_arr,
            b1,
            b2,
            b3,
            b4,
            rho_arr,
            first_acoustic,
            last_acoustic,
            mode_count,
            False,
            xp,
        )

        if g_top == 0.0:
            d[0] = 1.0
            e[1] = 0.0

        else:
            L = ind_arr[first_acoustic]
            xh2 = x * h_arr[first_acoustic] ** 2
            h_rho = h_arr[first_acoustic] * rho_arr[L]
            d[0] = (b1[L] - xh2) / h_rho / 2.0 + xp.real(f_top / g_top)

        iTurningPoint = size(z) - 1
        j = 0
        L = xp.asarray(ind_arr[first_acoustic], copy=True, dtype=xp.int32)
        # print('d[0], e[0]', d[0], e[0])
        for Medium in range(first_acoustic, last_acoustic + 1):
            xh2 = x * h_arr[Medium] ** 2
            h_rho = h_arr[Medium] * rho_arr[L + 1]
            if Medium >= first_acoustic + 1:
                L += 1  # advance to the top layer point in b1
                d[j] = (d[j] + (b1[L] - xh2) / h_rho) / 2.0
            for ii in range(Ng_arr[Medium] - 1):
                j += 1
                L += 1
                d[j] = (b1[L] - xh2) / h_rho
                # print('d[j], e[j]', j, d[j], e[j])
                if b1[L] - xh2 + 2.0 > 0.0:
                    iTurningPoint = min(j, iTurningPoint)

        f_bott, g_bott, iPower, mode_count = get_bc_impedance(
            x,
            omega2,
            False,
            cp_bott,
            cs_bott,
            rho_bott,
            h_arr,
            ind_arr,
            z_arr,
            b1,
            b2,
            b3,
            b4,
            rho_arr,
            first_acoustic,
            last_acoustic,
            mode_count,
            False,
            xp,
        )

        # print('f_bott', f_bott)
        # print('g_bott', g_bott)

        if g_bott == 0.0:
            d[N_total1 - 1] = 1.0
            e[N_total1 - 1] = 0.0
        else:
            d[N_total1 - 1] = d[j] / 2.0 - xp.real(f_bott / g_bott)

        # for i in range(N_total1):
        #    print('i, z[i], d[i], e[i]', i, z[i], d[i], e[i])
        w, i_error = inverse_iter(d, e, xp)

        w, pert_k, sg, ug = normalize(w, iTurningPoint, x, args, z, xp=xp)

        phi[:, mind] = w
        pert_k_arr[mind] = pert_k
        ugs_arr[mind] = ug
    return z, phi, pert_k_arr, ugs_arr


def mesh_list_inputs(
    z_list,
    cp_list,
    cs_list,
    rho_list,
    attnp_list,
    attns_list,
    Ng_arr,
    attn_units,
    omega,
    xp,
):
    num_layers = len(z_list)
    h_list = []
    for i in range(num_layers):
        z_arr_i = xp.linspace(z_list[i][0], z_list[i][-1], Ng_arr[i])
        cp_arr_i = array_interp(z_arr_i, z_list[i], cp_list[i], namespace=xp)
        cs_arr_i = array_interp(z_arr_i, z_list[i], cs_list[i], namespace=xp)
        rho_arr_i = array_interp(z_arr_i, z_list[i], rho_list[i], namespace=xp)
        attnp_arr_i = array_interp(z_arr_i, z_list[i], attnp_list[i], namespace=xp)
        attns_arr_i = array_interp(z_arr_i, z_list[i], attns_list[i], namespace=xp)
        h_list.append(z_arr_i[1] - z_arr_i[0])

        if i == 0:
            ind_list = [0]
            z_arr = z_arr_i
            cp_arr = cp_arr_i
            cs_arr = cs_arr_i
            rho_arr = rho_arr_i
            attnp_arr = attnp_arr_i
            attns_arr = attns_arr_i
        else:
            ind_list.append(size(z_arr))
            z_arr = xp.concatenate((z_arr, z_arr_i))
            cp_arr = xp.concatenate((cp_arr, cp_arr_i))
            cs_arr = xp.concatenate((cs_arr, cs_arr_i))
            rho_arr = xp.concatenate((rho_arr, rho_arr_i))
            attnp_arr = xp.concatenate((attnp_arr, attnp_arr_i))
            attns_arr = xp.concatenate((attns_arr, attns_arr_i))

    # Now convert speeds to complex
    if xp.any(attnp_arr > 0):
        cp_imag_arr = ap.get_c_imag(cp_arr, attnp_arr, attn_units, omega, xp=xp)
        cp_arr = cp_arr + 1j * cp_imag_arr
    if xp.any(attns_arr > 0):
        cs_imag_arr = ap.get_c_imag(cs_arr, attns_arr, attn_units, omega, xp=xp)
        cs_arr = cs_arr + 1j * cs_imag_arr

    ind_arr = xp.asarray(ind_list, dtype=xp.int32)
    h_arr = xp.asarray(h_list, dtype=xp.double)

    return h_arr, ind_arr, z_arr, cp_arr, cs_arr, rho_arr


def list_input_solve(
    freq: float,
    z_list: list[Array],
    cp_list: list[Array],
    cs_list: list[Array],
    rho_list: list[Array],
    attnp_list: list[Array],
    attns_list: list[Array],
    cp_top: float,
    cs_top: float,
    rho_top: float,
    attnp_top: float,
    attns_top: float,
    cp_bott: float,
    cs_bott: float,
    rho_bott: float,
    attnp_bott: float,
    attns_bott: float,
    attn_units: Literal["npm", "dbpm", "dbplam", "dbpkmhz", "q"],
    Ng_list: list[int],
    rmax: float,
    c_low: float,
    c_high: float,
    sigma_arr: Array,
    xp,
):
    """
    Take the environment specified as a list of arrays (each list item is a layer)
    and solve for the eigenvalues and eigenvectors eventually

    Currently only does c-linear interpolation

    Ng_list is number of mesh points as alist over layers
    """
    # initialize float has array of size (), to ensure compatibility with backend
    cp_top = xp.asarray(cp_top, dtype=xp.double)
    cs_top = xp.asarray(cs_top, dtype=xp.double)
    rho_top = xp.asarray(rho_top, dtype=xp.double)
    attnp_top = xp.asarray(attnp_top, dtype=xp.double)
    attns_top = xp.asarray(attns_top, dtype=xp.double)
    cp_bott = xp.asarray(cp_bott, dtype=xp.double)
    cs_bott = xp.asarray(cs_bott, dtype=xp.double)
    rho_bott = xp.asarray(rho_bott, dtype=xp.double)
    attnp_bott = xp.asarray(attnp_bott, dtype=xp.double)
    attns_bott = xp.asarray(attns_bott, dtype=xp.double)
    # First get a mesh
    omega = xp.asarray(2 * pi * freq, dtype=xp.double)
    omega2 = omega**2
    num_layers = len(z_list)
    if len(Ng_list) == 0:  # no mesh specified
        c = cp_list[-1][-1]  # arbitrary value, selected to agree with KRAKEN
        if cs_list[-1][-1] > c:
            c = cs_list[-1][-1]
        lam = c / freq
        dz_approx = lam / 20
        for i in range(num_layers):
            Nneeded = int((z_list[i][-1] - z_list[i][0]) / dz_approx)
            Nneeded = max(Nneeded, 10)
            Ng_list.append(Nneeded)

    Ng_arr0 = xp.asarray(Ng_list, dtype=xp.int32)
    # print('Ng_arr0', Ng_arr0)

    if attnp_top > 0:
        cp_top_imag = ap.get_c_imag(cp_top, attnp_top, attn_units, omega, xp=xp)
        cp_top = cp_top + 1j * cp_top_imag

    if attns_top > 0:
        cs_top_imag = ap.get_c_imag(cs_top, attns_top, attn_units, omega, xp=xp)
        cs_top = cs_top + 1j * cs_top_imag

    if attnp_bott > 0:
        cp_bott_imag = ap.get_c_imag(cp_bott, attnp_bott, attn_units, omega, xp=xp)
        cp_bott = cp_bott + 1j * cp_bott_imag

    if attns_bott > 0:
        cs_bott_imag = ap.get_c_imag(cs_bott, attns_bott, attn_units, omega, xp=xp)
        cs_bott = cs_bott + 1j * cs_bott_imag

    M_max = 5000
    M = M_max
    Nv = xp.asarray([1, 2, 4, 8, 16], dtype=xp.int32)  # mesh refinement factors
    Nset = len(Nv)
    ev_mat = xp.zeros((Nset, M_max), dtype=xp.double)  # real (for now)
    extrap = xp.zeros((Nset, M_max), dtype=xp.double)
    error = 1e10

    for iset in range(Nset):
        # Refine the mesh
        Ng_arr_i = (Ng_arr0 - 1) * Nv[iset] + 1
        h_arr, ind_arr, z_arr, cp_arr, cs_arr, rho_arr = mesh_list_inputs(
            z_list,
            cp_list,
            cs_list,
            rho_list,
            attnp_list,
            attns_list,
            Ng_arr_i,
            attn_units,
            omega,
            xp,
        )

        (
            b1,
            b1c,
            b2,
            b3,
            b4,
            rho_arr,
            c_low,
            c_high,
            elastic_flag,
            first_acoustic,
            last_acoustic,
        ) = initialize(
            h_arr,
            ind_arr,
            z_arr,
            omega2,
            cp_arr,
            cs_arr,
            rho_arr,
            cp_top,
            cs_top,
            rho_top,
            cp_bott,
            cs_bott,
            rho_bott,
            c_low,
            c_high,
        )

        # pack all this info into an asarray
        args = (
            omega2,
            ev_mat,
            iset,
            h_arr,
            ind_arr,
            z_arr,
            Ng_arr_i,
            cp_top,
            cs_top,
            rho_top,
            cp_bott,
            cs_bott,
            rho_bott,
            b1,
            b1c,
            b2,
            b3,
            b4,
            rho_arr,
            c_low,
            c_high,
            elastic_flag,
            first_acoustic,
            last_acoustic,
        )

        if iset == 0:
            h_v = xp.asarray([h_arr[0]], dtype=xp.double)
        else:
            if not xp.isclose(h_v[-1], h_arr[0] * Nv[iset] / Nv[iset - 1]):
                raise ValueError(
                    f"Mesh refinement factor mismatch: {h_v[-1]} != {h_arr[0] * Nv[iset]}"
                )
            h_v = array_append(h_v, h_arr[0], namespace=xp)

        if iset <= 1 and (last_acoustic - first_acoustic + 1 == num_layers):
            ev_mat, M = solve1(args, h_v, xp=xp)
            if M == 0:
                plt.figure()
                for i in range(num_layers):
                    plt.plot(cp_list[i], z_list[i])
                    plt.plot(cs_list[i], z_list[i])
                plt.show()
        else:  # solve2
            ev_mat, M = solve2(args, h_v, M, xp=xp)
            if omega2 / c_high**2 > ev_mat[iset, M - 1]:
                M -= 1

        # if iset == 0 get phi
        if iset == 0:
            pargs = args + (M, sigma_arr)
            z, phi, pert_k, ugs = get_phi(pargs, xp=xp)

        # print('iset', iset, 'M', M, ev_mat[iset, :M])

        extrap[iset, :M] = xp.asarray(ev_mat[iset, :M], copy=True, dtype=xp.double)

        error = 1e10

        KEY = int(2 * M / 3)  # index of element used to check convergence
        if iset > 0:
            T1 = extrap[0, KEY]
            for j in range(iset - 1, -1, -1):
                for m in range(M):
                    x1 = Nv[j] ** 2
                    x2 = Nv[iset] ** 2
                    F1 = extrap[j, m]
                    F2 = extrap[j + 1, m]
                    extrap[j, m] = F2 - (F1 - F2) / (x2 / x1 - 1.0)

            T2 = extrap[0, KEY]
            error = xp.abs(T2 - T1)
            if error * rmax < 1.0:
                break

        if error * rmax < 1.0:
            break

    M_final = min(
        size(pert_k), M
    )  # in case differing number of modes for different meshes
    krs = xp.sqrt(extrap[0, :M_final] + pert_k[:M_final])
    # krs = np.sqrt(extrap[0,:M])
    return krs, z, phi, ugs
