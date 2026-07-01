"""Pixel-index-to-physical-value transformations for WITec TDxxxTransformation tags.

Ported from wit_io's WITio.obj.wip.transform.m. The MATLAB code works with
1-based pixel indices (p = 1..N) and repeatedly computes `p - 1`; here we take
a 0-based numpy index array directly (i = 0..N-1, i.e. i == p-1), which lets
every formula drop its "-1".

All transforms return values in the *default* unit of their interpretation
kind (see units.py): 'um' for space, 'nm' for spectral. Use units.convert()
to map to a different display unit (e.g. Raman shift in 'rel. 1/cm').
"""
from __future__ import annotations

import numpy as np


def linear_transform(index, model_origin: float, world_origin: float, scale: float):
    """value = scale * (i - model_origin) + world_origin"""
    index = np.asarray(index, dtype=float)
    return scale * (index - model_origin) + world_origin


def lut_transform(index, lut: np.ndarray):
    """Piecewise-linear lookup, clamped at the ends (matches WITec Project 2.10.3 behavior)."""
    index = np.asarray(index, dtype=float)
    lut = np.asarray(lut, dtype=float)
    n = lut.size
    if n < 1:
        return np.full(index.shape, np.nan)
    if n == 1:
        return np.full(index.shape, lut[0])
    nodes = np.arange(n, dtype=float)
    value = np.interp(index, nodes, lut)
    value = np.where(index < 0, lut[0], value)
    value = np.where(index > n - 1, lut[-1], value)
    return value


def spectral_transform_polynomial(index, polynom):
    """SpectralTransformationType == 0: 2nd-order (at most) polynomial in pixel index."""
    index = np.asarray(index, dtype=float)
    polynom = np.asarray(polynom, dtype=float)
    n = min(polynom.size, 3)  # matches WITec Project 2.10.3, which ignores order >= 3
    value = np.zeros_like(index)
    for k in range(n):
        value = value + polynom[k] * index ** k
    return value


def spectral_transform_grating(index, *, nC, LambdaC, Gamma, Delta, m, d, x, f):
    """SpectralTransformationType == 1: grating-equation based wavelength calibration (nm).

    nC: pixel # (0-based here) at LambdaC. LambdaC: wavelength (nm) at array center.
    Gamma: included/deviation angle (rad). Delta: CCD inclination angle (rad).
    m: diffraction order. d: grating groove density (g/mm). x: pixel width (mm).
    f: instrument focal length (mm).
    """
    index = np.asarray(index, dtype=float)
    alpha = np.arcsin(LambdaC * m / d / (2 * np.cos(Gamma / 2))) - Gamma / 2
    l_h = f * np.cos(Delta)
    h_blambda_c = f * np.sin(Delta)
    h_blambda_n = x * (nC - index) - h_blambda_c
    beta_lambda_c = Gamma + alpha
    beta_h = beta_lambda_c - Delta
    beta_lambda_n = beta_h - np.arctan2(h_blambda_n, l_h)
    return d / m * (np.sin(alpha) + np.sin(beta_lambda_n))


def spectral_transform_free_polynomial(index, free_polynom, order, start_bin, stop_bin):
    """SpectralTransformationType == 2: constrained arbitrary-order polynomial, clamped outside
    [start_bin, stop_bin] (both given as 0-based pixel positions)."""
    index = np.asarray(index, dtype=float)
    free_polynom = np.asarray(free_polynom, dtype=float)
    order = min(int(order), free_polynom.size - 1)
    stop_bin = max(start_bin, stop_bin)  # matches WITec Project 2.10.3

    def poly(x):
        value = np.zeros_like(x)
        for k in range(order + 1):
            value = value + free_polynom[k] * x ** k
        return value

    value_start = poly(np.asarray(start_bin, dtype=float))
    value_stop = poly(np.asarray(stop_bin, dtype=float))

    value = np.empty_like(index)
    below = index <= start_bin
    above = index >= stop_bin
    inside = ~below & ~above
    value[below] = value_start
    value[above] = value_stop
    if np.any(inside):
        value[inside] = poly(index[inside])
    return value


def space_transform(pixel_xyz, model_origin, world_origin, scale, rotation):
    """Affine-transform 0-based (x, y, z) pixel indices to physical (X, Y, Z) coordinates.

    pixel_xyz: array shaped (..., 3). model_origin/world_origin: length-3 vectors.
    scale/rotation: 3x3 matrices. Returns an array shaped like pixel_xyz, in 'um'
    (or '1/um' for InverseSpace transformations).
    """
    pixel_xyz = np.asarray(pixel_xyz, dtype=float)
    model_origin = np.asarray(model_origin, dtype=float).reshape(3)
    world_origin = np.asarray(world_origin, dtype=float).reshape(3)
    scale = np.asarray(scale, dtype=float).reshape(3, 3)
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)

    shifted = pixel_xyz - model_origin
    transformed = shifted @ (rotation @ scale).T
    return transformed + world_origin
