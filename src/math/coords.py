"""
    src/math/coords.py
    ------------------
    Vectorized utilities for converting between Cartesian and Spherical
    (polar) coordinates and for combining/rotating spherical angles.
    This script includes functions for adding and subtracting angles.
"""
from __future__ import annotations

from typing import Final, Tuple, Union, Optional, Iterable
import numpy as np
import numpy.typing as npt


# Types
ArrayLike = Union[float, Iterable[float], npt.NDArray]
ArrayF = npt.NDArray[np.floating]

# A conservative machine eps baseline; we'll choose an eps based on array dtype later.
BASE_EPS: Final[float] = 1e-12


# ---------------========== Utility Helpers ==========--------------- #

def _as_1d_array(x: ArrayLike, dtype: Optional[np.dtype]=None) -> ArrayF:
    """
    Ensures that `x` us a 1-D array. This function does preserve the 
    data-type (dtype) unless `dtype` is given, if so conversion is 
    enforced. Scalars are being wrapped into the shape `(1,)`
    """
    array = np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)
    return array[None] if array.ndim == 0 else array


# Function for adjusting depending on precision 32-bit or 64-bit
def _eps_for_dtype(array: ArrayF) -> float:
    """
    Return a small epsilon appropriate for the array's float dtype.
    """
    if not np.issubdtype(array.dtype, np.floating):
        return BASE_EPS
    # Use machine epsilon scaled for safety; float32 -> larger eps
    info = np.finfo(array.dtype)
    return float(info.eps * 100.0)


# ---------------========== Coordinate Conversions ==========--------------- #

def cartesian_to_spherical(dvec: ArrayLike) -> Tuple[ArrayF, ArrayF, ArrayF]:
    """
    Convert `Cartesian coordinates` to corresponding `spherical coordinates`
    that is: (`x`,`y`.`z`) -> (`r`,`ф`,`θ`).

    - `r`:  Is the vector magnitude
    - `ф`:  Is the azimuth angles in degrees, measured from `+x` in the
            `xy` plane
    - `θ`:  Is the polar angle in degrees, measured from `+z` axis
    --------
        ф = np.atan2(y,x) ∈ [-180,180]
        θ = np.arccos(z/r) ∈ [0,180]

    Args:
        dvec:   Array-like with shape `(3,)` or `(N,3)` of spatial
                dimensions.
    Returns:
        Tuple of arrays `(r,ф,θ)`, each with shape `(N,)`.
    Raises:
        ValueError: If the dimensions of the distance vector `dvec` has
                    not 2 dimensions (samples and size), or the size is 
                    not exactly 3 dimensions, this raises an physical 
                    error problem.
    """
    array = np.asarray(dvec)
    if array.ndim == 1 and array.size == 3:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("dvec must have shape (3,) or (N,3)")

    x, y, z = array[:,0], array[:,1], array[:,2]

    # Radius extraction and tolerance epsilon matching the dtype
    radius = np.linalg.norm(array, axis=1)
    radius = np.clip(radius, _eps_for_dtype(radius), None)

    # `phi` and `theta` measured in degrees
    phi = np.degrees(np.arctan2(y, x))
    theta = np.degrees(np.arccos(np.clip(z / radius, -1.0, 1.0)))

    return radius, phi, theta


def spherical_to_cartesian(
    radius: ArrayLike, phi: ArrayLike, theta: ArrayLike
) -> ArrayF:
    """
    Convert spherical coordinates to Cartesian `(x, y, z)`.

    Angles are expected in degrees:
    - `phi` is the azimuth (rotation around `Z`).
    - `theta` is the polar angle from `+Z`.

    Args:
        radius: Radius or array of radii.
        phi: Azimuth angle(s) in degrees.
        theta: Polar angle(s) in degrees.    
    Returns:
        Cartesian coordinates with shape (N, 3). \\
    """
    r = _as_1d_array(radius)
    p = _as_1d_array(phi)
    t = _as_1d_array(theta)

    r, p, t = np.broadcast_arrays(r, p, t)

    # Convert angles to radians using numpy helper (preserves dtype)
    p_rad = np.deg2rad(p)
    t_rad = np.deg2rad(t)

    sin_t = np.sin(t_rad)
    x = r * np.cos(p_rad) * sin_t
    y = r * np.sin(p_rad) * sin_t
    z = r * np.cos(t_rad)

    return np.column_stack((x, y, z))


# ---------------========== Angle Combination / Rotation ==========--------------- #


def _angle_rotation_kernel(
    phi0: ArrayLike, theta0: ArrayLike,
    phi1: ArrayLike, theta1: ArrayLike,
    inverse: bool = False
) -> Tuple[ArrayF, ArrayF]:
    """
    Core rotation kernel that operates on radians arrays:
        rotate the direction given by (phi0, theta0) by the rotation
        defined by (phi1, theta1). All inputs are in radians here.

    Returns (phi, theta) in radians.
    """
    p0 = _as_1d_array(phi0)
    t0 = _as_1d_array(theta0)
    p1 = _as_1d_array(phi1)
    t1 = _as_1d_array(theta1)

    p0, t0, p1, t1 = np.broadcast_arrays(p0, t0, p1, t1)

    # precompute trig
    s_t0, c_t0 = np.sin(t0), np.cos(t0)
    s_p0, c_p0 = np.sin(p0), np.cos(p0)
    s_t1, c_t1 = np.sin(t1), np.cos(t1)
    s_p1, c_p1 = np.sin(p1), np.cos(p1)

    # original vector components (Cartesian) from spherical
    x0 = s_t0 * c_p0
    y0 = s_t0 * s_p0
    z0 = c_t0

    if not inverse:
        # R = Rz(phi1) @ Ry(theta1)   (rotation defined by phi1,theta1)
        m00 = c_p1 * c_t1; m01 = -s_p1;        m02 = c_p1 * s_t1
        m10 = s_p1 * c_t1; m11 =  c_p1;        m12 = s_p1 * s_t1
        m20 = -s_t1;       m21 =  0.0;         m22 = c_t1

        x = m00 * x0 + m01 * y0 + m02 * z0
        y = m10 * x0 + m11 * y0 + m12 * z0
        z = m20 * x0 + m21 * y0 + m22 * z0
    else:
        # inverse rotation: apply R.T
        m00 = c_p1 * c_t1; m10 = s_p1 * c_t1; m20 = -s_t1
        m01 = -s_p1;       m11 = c_p1;         m21 = 0.0
        m02 = c_p1 * s_t1; m12 = s_p1 * s_t1; m22 = c_t1

        x = m00 * x0 + m10 * y0 + m20 * z0
        y = m01 * x0 + m11 * y0 + m21 * z0
        z = m02 * x0 + m12 * y0 + m22 * z0

    # ensure numeric domain for acos: clip z to [-1,1]
    z = np.clip(z, -1.0, 1.0)
    return np.arctan2(y, x), np.arccos(z)


def _combine_angles(
    phi0: ArrayLike, theta0: ArrayLike, phi1: ArrayLike, theta1: ArrayLike,
    inverse: bool = False
) -> Tuple[ArrayF, ArrayF]:
    """
    Wrapper intake degrees which are converted to radians, calling a kernel 
    to decide the rotation.
    """
    # Convert from degrees to radians
    p0 = np.deg2rad(_as_1d_array(phi0))
    t0 = np.deg2rad(_as_1d_array(theta0))
    p1 = np.deg2rad(_as_1d_array(phi1))
    t1 = np.deg2rad(_as_1d_array(theta1))

    phi_r, theta_r = _angle_rotation_kernel(p0, t0, p1, t1, inverse=inverse)
    return np.rad2deg(phi_r), np.rad2deg(theta_r)


def add_angles(phi0: ArrayLike, theta0: ArrayLike,
               phi1: ArrayLike, theta1: ArrayLike) -> Tuple[ArrayF, ArrayF]:
    """Rotate (phi0,theta0) by (phi1,theta1) (degrees in/out)."""
    return _combine_angles(phi0, theta0, phi1, theta1, inverse=False)


def sub_angles(phi0: ArrayLike, theta0: ArrayLike,
               phi1: ArrayLike, theta1: ArrayLike) -> Tuple[ArrayF, ArrayF]:
    """Apply inverse rotation: subtract (phi1,theta1) from (phi0,theta0)."""
    return _combine_angles(phi0, theta0, phi1, theta1, inverse=True)
