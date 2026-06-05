# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
B-Spline Interpolation Kernels (Pure Warp)
==========================================

This module provides pure Warp kernels and launchers for B-spline interpolation
functions used in mesh-based calculations (e.g., Particle Mesh Ewald).

This module is framework-agnostic - it contains only Warp kernels and launchers.
For PyTorch bindings, use ``nvalchemiops.torch.spline`` instead.

SUPPORTED ORDERS
================

- Order 1: Constant (Nearest Grid Point)
- Order 2: Linear
- Order 3: Quadratic
- Order 4: Cubic (recommended for PME)
- Order 5: Quartic
- Order 6: Quintic

OPERATIONS
==========

1. SPREAD: Scatter atom values to mesh grid
   mesh[g] += value[atom] * weight(atom, g)

2. GATHER: Collect mesh values at atom positions
   value[atom] = Σ_g mesh[g] * weight(atom, g)

3. GATHER_VEC3: Collect 3D vector field values at atom positions
   vector[atom] = Σ_g mesh[g] * weight(atom, g)

4. GATHER_GRADIENT: Collect mesh values with weight gradients (forces)
   grad[atom] = sum_g mesh[g] * grad_weight(atom, g)

5. SPREAD_CHANNELS: Scatter multi-channel values (e.g., multipoles) to mesh
   mesh[c, g] += values[atom, c] * weight(atom, g)

6. GATHER_CHANNELS: Collect multi-channel values from mesh
   values[atom, c] = Σ_g mesh[c, g] * weight(atom, g)

REFERENCES
==========

- Essmann et al. (1995). J. Chem. Phys. 103, 8577 (PME B-splines)
"""

from __future__ import annotations

from typing import Any

import warp as wp

###########################################################################################
########################### B-Spline Weight Functions #####################################
###########################################################################################


@wp.func
def bspline_weight(u: Any, order: wp.int32) -> Any:
    """Compute B-spline basis function M_n(u).

    Parameters
    ----------
    u : float (Any)
        Parameter in [0, order). Type-generic (float32 or float64).
    order : wp.int32
        Spline order (1=constant, 2=linear, 3=quadratic, 4=cubic,
        5=quartic, 6=quintic).

    Returns
    -------
    float (Any)
        Weight value M_n(u). Same type as input.
    """
    # Type-generic constants
    zero = type(u)(0.0)
    one = type(u)(1.0)
    two = type(u)(2.0)
    three = type(u)(3.0)
    four = type(u)(4.0)
    five = type(u)(5.0)
    six = type(u)(6.0)

    if order == 6:
        # M_6 has 6 quintic pieces on [0, 6); peak M_6(3) = 11/20.
        # Derived as the 6-fold convolution of the unit boxcar:
        # M_6(u) = (1/120) Σ_{j=0}^{6} (-1)^j C(6,j) (u-j)_+^5.
        u2 = u * u
        u3 = u2 * u
        u4 = u3 * u
        u5 = u4 * u
        c120 = type(u)(120.0)
        if u >= zero and u < one:
            return u5 / c120
        elif u >= one and u < two:
            return (
                type(u)(-5.0) * u5
                + type(u)(30.0) * u4
                + type(u)(-60.0) * u3
                + type(u)(60.0) * u2
                + type(u)(-30.0) * u
                + six
            ) / c120
        elif u >= two and u < three:
            return (
                type(u)(10.0) * u5
                + type(u)(-120.0) * u4
                + type(u)(540.0) * u3
                + type(u)(-1140.0) * u2
                + type(u)(1170.0) * u
                + type(u)(-474.0)
            ) / c120
        elif u >= three and u < four:
            return (
                type(u)(-10.0) * u5
                + type(u)(180.0) * u4
                + type(u)(-1260.0) * u3
                + type(u)(4260.0) * u2
                + type(u)(-6930.0) * u
                + type(u)(4386.0)
            ) / c120
        elif u >= four and u < five:
            return (
                five * u5
                + type(u)(-120.0) * u4
                + type(u)(1140.0) * u3
                + type(u)(-5340.0) * u2
                + type(u)(12270.0) * u
                + type(u)(-10974.0)
            ) / c120
        elif u >= five and u < six:
            v = six - u
            return v * v * v * v * v / c120
        else:
            return zero
    elif order == 5:
        # M_5 has 5 quartic pieces on [0, 5); peak M_5(5/2) = 115/192.
        # Derived as the 5-fold convolution of the unit boxcar:
        # M_5(u) = (1/24) Σ_{j=0}^{5} (-1)^j C(5,j) (u-j)_+^4.
        u2 = u * u
        u3 = u2 * u
        u4 = u3 * u
        c24 = type(u)(24.0)
        if u >= zero and u < one:
            return u4 / c24
        elif u >= one and u < two:
            return (
                type(u)(-4.0) * u4
                + type(u)(20.0) * u3
                + type(u)(-30.0) * u2
                + type(u)(20.0) * u
                + type(u)(-5.0)
            ) / c24
        elif u >= two and u < three:
            return (
                type(u)(6.0) * u4
                + type(u)(-60.0) * u3
                + type(u)(210.0) * u2
                + type(u)(-300.0) * u
                + type(u)(155.0)
            ) / c24
        elif u >= three and u < four:
            return (
                type(u)(-4.0) * u4
                + type(u)(60.0) * u3
                + type(u)(-330.0) * u2
                + type(u)(780.0) * u
                + type(u)(-655.0)
            ) / c24
        elif u >= four and u < five:
            v = five - u
            return v * v * v * v / c24
        else:
            return zero
    elif order == 4:
        if u >= zero and u < one:
            return u * u * u / six
        elif u >= one and u < two:
            u2 = u * u
            u3 = u2 * u
            return (
                type(u)(-3.0) * u3 + type(u)(12.0) * u2 - type(u)(12.0) * u + four
            ) / six
        elif u >= two and u < three:
            u2 = u * u
            u3 = u2 * u
            return (
                three * u3 - type(u)(24.0) * u2 + type(u)(60.0) * u - type(u)(44.0)
            ) / six
        elif u >= three and u < four:
            v = four - u
            return v * v * v / six
        else:
            return zero
    elif order == 3:
        if u >= zero and u < one:
            return u * u / two
        elif u >= one and u < two:
            return type(u)(0.75) - (u - type(u)(1.5)) * (u - type(u)(1.5))
        elif u >= two and u < three:
            v = three - u
            return v * v / two
        else:
            return zero
    elif order == 2:
        if u >= zero and u < one:
            return u
        elif u >= one and u < two:
            return two - u
        else:
            return zero
    elif order == 1:
        if u >= zero and u < one:
            return one
        else:
            return zero
    else:
        return zero


@wp.func
def bspline_derivative(u: Any, order: wp.int32) -> Any:
    """Compute B-spline derivative dM_n(u)/du.

    Parameters
    ----------
    u : float (Any)
        Parameter in [0, order). Type-generic (float32 or float64).
    order : wp.int32
        Spline order.

    Returns
    -------
    float (Any)
        Derivative value. Same type as input.
    """
    # Type-generic constants
    zero = type(u)(0.0)
    one = type(u)(1.0)
    two = type(u)(2.0)
    three = type(u)(3.0)
    four = type(u)(4.0)
    five = type(u)(5.0)
    six = type(u)(6.0)

    if order == 6:
        # M_6'(u) — quartic on each of 6 unit pieces.
        u2 = u * u
        u3 = u2 * u
        u4 = u3 * u
        c120 = type(u)(120.0)
        if u >= zero and u < one:
            return five * u4 / c120
        elif u >= one and u < two:
            return (
                type(u)(-25.0) * u4
                + type(u)(120.0) * u3
                + type(u)(-180.0) * u2
                + type(u)(120.0) * u
                + type(u)(-30.0)
            ) / c120
        elif u >= two and u < three:
            return (
                type(u)(50.0) * u4
                + type(u)(-480.0) * u3
                + type(u)(1620.0) * u2
                + type(u)(-2280.0) * u
                + type(u)(1170.0)
            ) / c120
        elif u >= three and u < four:
            return (
                type(u)(-50.0) * u4
                + type(u)(720.0) * u3
                + type(u)(-3780.0) * u2
                + type(u)(8520.0) * u
                + type(u)(-6930.0)
            ) / c120
        elif u >= four and u < five:
            return (
                type(u)(25.0) * u4
                + type(u)(-480.0) * u3
                + type(u)(3420.0) * u2
                + type(u)(-10680.0) * u
                + type(u)(12270.0)
            ) / c120
        elif u >= five and u < six:
            v = six - u
            return -five * v * v * v * v / c120
        else:
            return zero
    elif order == 5:
        # M_5'(u) — cubic on each of 5 unit pieces.
        u2 = u * u
        u3 = u2 * u
        c24 = type(u)(24.0)
        if u >= zero and u < one:
            return four * u3 / c24
        elif u >= one and u < two:
            return (
                type(u)(-16.0) * u3
                + type(u)(60.0) * u2
                + type(u)(-60.0) * u
                + type(u)(20.0)
            ) / c24
        elif u >= two and u < three:
            return (
                type(u)(24.0) * u3
                + type(u)(-180.0) * u2
                + type(u)(420.0) * u
                + type(u)(-300.0)
            ) / c24
        elif u >= three and u < four:
            return (
                type(u)(-16.0) * u3
                + type(u)(180.0) * u2
                + type(u)(-660.0) * u
                + type(u)(780.0)
            ) / c24
        elif u >= four and u < five:
            v = five - u
            return -four * v * v * v / c24
        else:
            return zero
    elif order == 4:
        if u >= zero and u < one:
            return u * u / two
        elif u >= one and u < two:
            return (type(u)(-9.0) * u * u + type(u)(24.0) * u - type(u)(12.0)) / six
        elif u >= two and u < three:
            return (type(u)(9.0) * u * u - type(u)(48.0) * u + type(u)(60.0)) / six
        elif u >= three and u < four:
            v = four - u
            return -three * v * v / six
        else:
            return zero
    elif order == 3:
        if u >= zero and u < one:
            return u
        elif u >= one and u < two:
            return -two * (u - type(u)(1.5))
        elif u >= two and u < three:
            return -(three - u)
        else:
            return zero
    elif order == 2:
        if u >= zero and u < one:
            return one
        elif u >= one and u < two:
            return -one
        else:
            return zero
    else:
        return zero


###########################################################################################
########################### Grid Utility Functions ########################################
###########################################################################################


@wp.func
def compute_fractional_coords(
    position: Any,
    cell_inv_t: Any,
    mesh_dims: wp.vec3i,
) -> Any:
    """Convert Cartesian position to mesh coordinates.

    Parameters
    ----------
    position : vec3 (Any)
        Atomic position. Type-generic (vec3f or vec3d).
    cell_inv_t : mat33 (Any)
        Transpose of inverse cell. Type-generic (mat33f or mat33d).
    mesh_dims : wp.vec3i
        Mesh dimensions.

    Returns
    -------
    base_grid : wp.vec3i
        Base grid point (floor of mesh coords).
    theta : vec3 (Any)
        Fractional part [0, 1) in each dimension. Same type as position.

    Note: Returns (base_grid, theta) as a tuple via multiple return values.
    """
    # Convert to fractional coordinates
    frac = cell_inv_t * position
    p0 = position[0]
    # Scale to mesh coordinates
    mesh_x = frac[0] * type(p0)(mesh_dims[0])
    mesh_y = frac[1] * type(p0)(mesh_dims[1])
    mesh_z = frac[2] * type(p0)(mesh_dims[2])

    # Base grid point
    mx = wp.int32(wp.floor(mesh_x))
    my = wp.int32(wp.floor(mesh_y))
    mz = wp.int32(wp.floor(mesh_z))

    # Fractional part
    theta_x = mesh_x - type(p0)(mx)
    theta_y = mesh_y - type(p0)(my)
    theta_z = mesh_z - type(p0)(mz)

    return wp.vec3i(mx, my, mz), type(position)(theta_x, theta_y, theta_z)


@wp.func
def bspline_grid_offset(
    point_idx: wp.int32,
    order: wp.int32,
    theta: Any,
) -> wp.vec3i:
    """Compute grid offset for B-spline point index.

    For B-splines, points are indexed 0 to order^3-1 and arranged in a cube.
    The offset is computed such that the B-spline parameter u is always in [0, n).

    The offset_start for each dimension is floor(theta - (n-2)/2), which ensures
    that for any theta in [0, 1), all n grid points have valid u values.

    Parameters
    ----------
    point_idx : wp.int32
        Linear point index (0 to order^3-1).
    order : wp.int32
        Spline order.
    theta : vec3 (Any)
        Fractional position within the base grid cell [0, 1) in each dimension.
        Type-generic (vec3f or vec3d).

    Returns
    -------
    wp.vec3i
        Grid offset (relative to base grid point).
    """
    order2 = order * order
    i = point_idx // order2
    j = (point_idx % order2) // order
    k = point_idx % order

    t0 = theta[0]

    # Compute offset_start = floor(theta - (n-2)/2) for each dimension
    # This ensures u = n/2 + theta - offset is always in [0, n)
    half_n_minus_1 = type(t0)(order - 2) * type(t0)(0.5)
    offset_start_x = wp.int32(wp.floor(t0 - half_n_minus_1))
    offset_start_y = wp.int32(wp.floor(theta[1] - half_n_minus_1))
    offset_start_z = wp.int32(wp.floor(theta[2] - half_n_minus_1))

    return wp.vec3i(i + offset_start_x, j + offset_start_y, k + offset_start_z)


@wp.func
def bspline_weight_3d(
    theta: Any,
    offset: wp.vec3i,
    order: wp.int32,
) -> Any:
    """Compute 3D B-spline weight (separable product).

    The B-spline parameter u is computed as:

    .. math::

        u = \\text{order}/2 + \\theta - \\text{offset}

    When offset = i + offset_start (from bspline_grid_offset), this gives
    u values in [0, n) that sum to 1 and are centered at the atom position.

    Parameters
    ----------
    theta : vec3 (Any)
        Fractional position within the base grid cell [0, 1).
        Type-generic (vec3f or vec3d).
    offset : wp.vec3i
        Grid offset from base grid point (includes offset_start adjustment).
    order : wp.int32
        Spline order.

    Returns
    -------
    float (Any)
        Weight = M(u_x) * M(u_y) * M(u_z). Same scalar type as theta.
    """
    # Get scalar type from theta vector
    t0 = theta[0]
    half_order = type(t0)(order) * type(t0)(0.5)
    zero = type(t0)(0.0)
    order_f = type(t0)(order)

    # u = n/2 + theta - offset
    u_x = half_order + t0 - type(t0)(offset[0])
    u_y = half_order + theta[1] - type(t0)(offset[1])
    u_z = half_order + theta[2] - type(t0)(offset[2])

    if (
        u_x < zero
        or u_x >= order_f
        or u_y < zero
        or u_y >= order_f
        or u_z < zero
        or u_z >= order_f
    ):
        return zero

    return (
        bspline_weight(u_x, order)
        * bspline_weight(u_y, order)
        * bspline_weight(u_z, order)
    )


@wp.func
def bspline_weight_gradient_3d(
    theta: Any,
    offset: wp.vec3i,
    order: wp.int32,
    mesh_dims: wp.vec3i,
) -> Any:
    """Compute gradient of 3D B-spline weight.

    The B-spline parameter u is computed as:

    .. math::

        u = \\text{order}/2 + \\theta - \\text{offset}

    The gradient with respect to theta is:

    .. math::

        \\begin{aligned}
        \\frac{\\partial u}{\\partial \\theta} &= +1 \\\\
        \\frac{\\partial \\text{weight}}{\\partial \\theta} &= \\frac{\\partial M}{\\partial u} \\cdot \\frac{\\partial u}{\\partial \\theta} = \\frac{\\partial M}{\\partial u}
        \\end{aligned}

    Parameters
    ----------
    theta : vec3 (Any)
        Fractional position within the base grid cell [0, 1).
        Type-generic (vec3f or vec3d).
    offset : wp.vec3i
        Grid offset from base grid point (includes offset_start adjustment).
    order : wp.int32
        Spline order.
    mesh_dims : wp.vec3i
        Mesh dimensions (for scaling to Cartesian coordinates).

    Returns
    -------
    vec3 (Any)
        Gradient :math:`\\nabla` weight in fractional coordinates (scaled by mesh_dims).
        Same type as theta.
    """
    # Get scalar type from theta vector
    t0 = theta[0]
    half_order = type(t0)(order) * type(t0)(0.5)
    zero = type(t0)(0.0)
    order_f = type(t0)(order)

    # u = n/2 + theta - offset
    u_x = half_order + t0 - type(t0)(offset[0])
    u_y = half_order + theta[1] - type(t0)(offset[1])
    u_z = half_order + theta[2] - type(t0)(offset[2])

    if (
        u_x < zero
        or u_x >= order_f
        or u_y < zero
        or u_y >= order_f
        or u_z < zero
        or u_z >= order_f
    ):
        return type(theta)(zero, zero, zero)

    w_x = bspline_weight(u_x, order)
    w_y = bspline_weight(u_y, order)
    w_z = bspline_weight(u_z, order)

    # Positive sign because u = half_order + theta - offset, so ∂u/∂theta = +1
    dw_x = bspline_derivative(u_x, order) * type(t0)(mesh_dims[0])
    dw_y = bspline_derivative(u_y, order) * type(t0)(mesh_dims[1])
    dw_z = bspline_derivative(u_z, order) * type(t0)(mesh_dims[2])

    return type(theta)(dw_x * w_y * w_z, w_x * dw_y * w_z, w_x * w_y * dw_z)


@wp.func
def bspline_second_derivative(u: Any, order: wp.int32) -> Any:
    """Compute B-spline second derivative d²M_n(u)/du².

    Same piecewise structure as :func:`bspline_weight` /
    :func:`bspline_derivative`. Used by :func:`bspline_weight_hessian_3d`
    for the dipole-force gather kernel in Multipole PME (Phase 3).

    Parameters
    ----------
    u : float (Any)
        Parameter in [0, order). Type-generic (float32 or float64).
    order : wp.int32
        Spline order. Defined for orders 3, 4, 5, and 6 (orders 1, 2
        return 0 because M₁ and M₂ are piecewise constant / linear and
        have zero second derivative everywhere except on a measure-zero
        set; the Hessian gather samples at smooth interior points).

    Returns
    -------
    float (Any)
        Second-derivative value. Same type as input.
    """
    zero = type(u)(0.0)
    one = type(u)(1.0)
    two = type(u)(2.0)
    three = type(u)(3.0)
    four = type(u)(4.0)
    five = type(u)(5.0)
    six = type(u)(6.0)

    if order == 6:
        # M_6''(u) — cubic on each of 6 unit pieces.
        u2 = u * u
        u3 = u2 * u
        c120 = type(u)(120.0)
        if u >= zero and u < one:
            return type(u)(20.0) * u3 / c120
        elif u >= one and u < two:
            return (
                type(u)(-100.0) * u3
                + type(u)(360.0) * u2
                + type(u)(-360.0) * u
                + type(u)(120.0)
            ) / c120
        elif u >= two and u < three:
            return (
                type(u)(200.0) * u3
                + type(u)(-1440.0) * u2
                + type(u)(3240.0) * u
                + type(u)(-2280.0)
            ) / c120
        elif u >= three and u < four:
            return (
                type(u)(-200.0) * u3
                + type(u)(2160.0) * u2
                + type(u)(-7560.0) * u
                + type(u)(8520.0)
            ) / c120
        elif u >= four and u < five:
            return (
                type(u)(100.0) * u3
                + type(u)(-1440.0) * u2
                + type(u)(6840.0) * u
                + type(u)(-10680.0)
            ) / c120
        elif u >= five and u < six:
            v = six - u
            return type(u)(20.0) * v * v * v / c120
        else:
            return zero
    elif order == 5:
        # M_5''(u) — quadratic on each of 5 unit pieces.
        u2 = u * u
        c24 = type(u)(24.0)
        if u >= zero and u < one:
            # B(u) = u^4/24 → B''(u) = u^2/2 = 12 u^2 / 24.
            return type(u)(12.0) * u2 / c24
        elif u >= one and u < two:
            return (type(u)(-48.0) * u2 + type(u)(120.0) * u + type(u)(-60.0)) / c24
        elif u >= two and u < three:
            return (type(u)(72.0) * u2 + type(u)(-360.0) * u + type(u)(420.0)) / c24
        elif u >= three and u < four:
            return (type(u)(-48.0) * u2 + type(u)(360.0) * u + type(u)(-660.0)) / c24
        elif u >= four and u < five:
            v = five - u
            return type(u)(12.0) * v * v / c24
        else:
            return zero
    elif order == 4:
        if u >= zero and u < one:
            # B(u) = u³/6  →  B''(u) = u.
            return u
        elif u >= one and u < two:
            # B(u) = (-3u³ + 12u² − 12u + 4)/6  →  B''(u) = -3u + 4.
            return four - three * u
        elif u >= two and u < three:
            # B(u) = (3u³ − 24u² + 60u − 44)/6  →  B''(u) = 3u - 8.
            return three * u - type(u)(8.0)
        elif u >= three and u < four:
            # B(u) = (4 - u)³/6  →  B''(u) = 4 - u.
            return four - u
        else:
            return zero
    elif order == 3:
        if u >= zero and u < one:
            # B(u) = u²/2  →  B''(u) = 1.
            return one
        elif u >= one and u < two:
            # B(u) = 3/4 - (u - 3/2)²  →  B''(u) = -2.
            return -two
        elif u >= two and u < three:
            # B(u) = (3 - u)²/2  →  B''(u) = 1.
            return one
        else:
            return zero
    else:
        # Orders 1 and 2 are piecewise constant / linear; the second
        # derivative is zero on the open interior of each piece.
        return zero


@wp.func
def bspline_third_derivative(u: Any, order: wp.int32) -> Any:
    r"""Compute B-spline third derivative ``d³M_n(u)/du³``.

    Same piecewise structure as :func:`bspline_second_derivative`. Used
    by the multipole-PME ``l_max = 2`` (quadrupole) backward kernels —
    the position-gradient slot ``∂L/∂r_i`` of the Q channel needs
    ``∂³B`` (since ``E_recip^(Q) ∝ Q : ∇²φ`` and ``∂/∂r_i`` adds one
    more derivative).

    Defined for orders 4, 5, 6 (cubic and above). Lower orders return
    zero (their third derivative is a Dirac delta train, sampled at
    interior smooth points to be zero).

    Parameters
    ----------
    u : float (Any)
        Parameter in ``[0, order)``. Type-generic (float32 or float64).
    order : wp.int32
        Spline order.

    Returns
    -------
    float (Any)
        Third-derivative value. Same type as ``u``.
    """
    zero = type(u)(0.0)
    one = type(u)(1.0)
    two = type(u)(2.0)
    three = type(u)(3.0)
    four = type(u)(4.0)
    five = type(u)(5.0)
    six = type(u)(6.0)

    if order == 6:
        # M_6''' is quadratic on each of 6 unit pieces.
        u2 = u * u
        c120 = type(u)(120.0)
        if u >= zero and u < one:
            # M_6''(u) = 20 u³/120  →  M_6'''(u) = 60 u²/120 = u²/2.
            return type(u)(60.0) * u2 / c120
        elif u >= one and u < two:
            # M_6'' = (-100 u³ + 360 u² − 360 u + 120)/120
            # → M_6''' = (-300 u² + 720 u − 360)/120
            return (type(u)(-300.0) * u2 + type(u)(720.0) * u + type(u)(-360.0)) / c120
        elif u >= two and u < three:
            # M_6'' = (200 u³ − 1440 u² + 3240 u − 2280)/120
            # → M_6''' = (600 u² − 2880 u + 3240)/120
            return (type(u)(600.0) * u2 + type(u)(-2880.0) * u + type(u)(3240.0)) / c120
        elif u >= three and u < four:
            # M_6'' = (-200 u³ + 2160 u² − 7560 u + 8520)/120
            # → M_6''' = (-600 u² + 4320 u − 7560)/120
            return (
                type(u)(-600.0) * u2 + type(u)(4320.0) * u + type(u)(-7560.0)
            ) / c120
        elif u >= four and u < five:
            # M_6'' = (100 u³ − 1440 u² + 6840 u − 10680)/120
            # → M_6''' = (300 u² − 2880 u + 6840)/120
            return (type(u)(300.0) * u2 + type(u)(-2880.0) * u + type(u)(6840.0)) / c120
        elif u >= five and u < six:
            # M_6''(u) = 20 (6-u)³/120  →  ∂/∂u of that
            # = 20 · 3 (6-u)² · (-1) / 120 = -60 (6-u)²/120
            v = six - u
            return -type(u)(60.0) * v * v / c120
        else:
            return zero
    elif order == 5:
        # M_5''' is linear on each of 5 unit pieces.
        c24 = type(u)(24.0)
        if u >= zero and u < one:
            return type(u)(24.0) * u / c24  # ≡ u
        elif u >= one and u < two:
            return (type(u)(-96.0) * u + type(u)(120.0)) / c24
        elif u >= two and u < three:
            return (type(u)(144.0) * u + type(u)(-360.0)) / c24
        elif u >= three and u < four:
            return (type(u)(-96.0) * u + type(u)(360.0)) / c24
        elif u >= four and u < five:
            # M_5'' = 12 (5-u)²/24  →  d/du = -24(5-u)/24 = -(5-u)
            v = five - u
            return -v
        else:
            return zero
    elif order == 4:
        # M_4''' is piecewise constant on each of 4 unit pieces.
        if u >= zero and u < one:
            # M_4''(u) = u  →  M_4'''(u) = 1
            return one
        elif u >= one and u < two:
            # M_4''(u) = 4 - 3u  →  M_4'''(u) = -3
            return -three
        elif u >= two and u < three:
            # M_4''(u) = 3u - 8  →  M_4'''(u) = 3
            return three
        elif u >= three and u < four:
            # M_4''(u) = 4 - u  →  M_4'''(u) = -1
            return -one
        else:
            return zero
    else:
        # Orders 1, 2, 3 are piecewise constant/linear/quadratic; the
        # third derivative is zero on the open interior of each piece.
        return zero


@wp.func
def bspline_fourth_derivative(u: Any, order: wp.int32) -> Any:
    r"""Compute B-spline fourth derivative ``d⁴M_n(u)/du⁴``.

    Same piecewise structure as :func:`bspline_third_derivative`, one order
    higher. Used by the multipole-PME ``l_max = 2`` **double-backward** (Q5r-2):
    the position-position Hessian block ``∂(∂L/∂r_i)/∂r_j`` of the Q channel
    needs ``∂⁴B`` (the Q-channel forward spread already carries ``∂²B``, its
    first backward ``∂³B``, and create_graph adds one more).

    Defined for orders 5, 6. ``M_6'''`` is quadratic per piece → ``M_6''''``
    is **linear** per piece (C⁰-continuous, since ``M_6`` is C⁴). ``M_5'''`` is
    linear per piece → ``M_5''''`` is **piecewise-constant** ``(1, −4, 6, −4,
    1)`` (discontinuous at knots — ``M_5`` is only C³). Orders ≤ 4 return zero
    (their fourth derivative is a Dirac train, zero on the open interior).

    Parameters
    ----------
    u : float (Any)
        Parameter in ``[0, order)``. Type-generic (float32 or float64).
    order : wp.int32
        Spline order.

    Returns
    -------
    float (Any)
        Fourth-derivative value. Same type as ``u``.
    """
    zero = type(u)(0.0)
    one = type(u)(1.0)
    two = type(u)(2.0)
    three = type(u)(3.0)
    four = type(u)(4.0)
    five = type(u)(5.0)
    six = type(u)(6.0)

    if order == 6:
        # M_6'''' is linear on each of 6 unit pieces (∂/∂u of M_6''', /120).
        c120 = type(u)(120.0)
        if u >= zero and u < one:
            return type(u)(120.0) * u / c120
        elif u >= one and u < two:
            return (type(u)(-600.0) * u + type(u)(720.0)) / c120
        elif u >= two and u < three:
            return (type(u)(1200.0) * u + type(u)(-2880.0)) / c120
        elif u >= three and u < four:
            return (type(u)(-1200.0) * u + type(u)(4320.0)) / c120
        elif u >= four and u < five:
            return (type(u)(600.0) * u + type(u)(-2880.0)) / c120
        elif u >= five and u < six:
            # ∂/∂u of -60(6-u)²/120 = 120(6-u)/120 = (6-u).
            return six - u
        else:
            return zero
    elif order == 5:
        # M_5'''' is piecewise-constant (1, -4, 6, -4, 1) on the 5 unit pieces.
        if u >= zero and u < one:
            return one
        elif u >= one and u < two:
            return -four
        elif u >= two and u < three:
            return six
        elif u >= three and u < four:
            return -four
        elif u >= four and u < five:
            return one
        else:
            return zero
    else:
        # Orders <= 4: fourth derivative is zero on the open interior of each
        # piece (M_4''' is piecewise constant; lower orders even smoother here).
        return zero


@wp.func
def bspline_weight_hessian_3d(
    theta: Any,
    offset: wp.vec3i,
    order: wp.int32,
    mesh_dims: wp.vec3i,
):
    r"""Compute the Hessian of the 3D B-spline weight at a grid offset.

    The 3D weight is the product of three 1D weights:

    .. math::

        w_{3D}(\theta) = M_n(u_x) \, M_n(u_y) \, M_n(u_z)

    where :math:`u_\alpha = \mathrm{order}/2 + \theta_\alpha - \mathrm{offset}_\alpha`.
    The Hessian is the symmetric 3×3 matrix of second partials:

    * Diagonal :math:`\partial^2 w_{3D}/\partial \theta_\alpha^2 = M_n''(u_\alpha) \cdot \prod_{\beta \neq \alpha} M_n(u_\beta)`.
    * Off-diagonal :math:`\partial^2 w_{3D}/\partial \theta_\alpha \partial \theta_\beta = M_n'(u_\alpha) \, M_n'(u_\beta) \, M_n(u_\gamma)`
      with :math:`\gamma` the remaining axis.

    Each component is scaled by ``mesh_dims_α · mesh_dims_β`` to match
    the fractional-mesh convention used by :func:`bspline_weight_gradient_3d`
    (the chain rule from parametric ``u`` to ``r`` introduces the same
    Jacobian factors per derivative).

    Returns two ``vec3`` tiles:

    * ``diag = (Hxx, Hyy, Hzz)`` (3-vector of diagonal entries).
    * ``off  = (Hxy, Hxz, Hyz)`` (3-vector of unique off-diagonal entries).

    The full symmetric Hessian is::

        H = [[diag.x, off.x, off.y],
             [off.x, diag.y, off.z],
             [off.y, off.z, diag.z]]

    Two-vec3 return chosen over a ``mat33`` to avoid importing matrix
    types into the spline-primitive layer; callers (e.g., the
    Multipole PME Hessian-gather kernel) accumulate ``μ_i · H · μ_j``
    directly from the six unique entries without materializing a
    matrix.

    Parameters
    ----------
    theta : vec3 (Any)
        Fractional position within the base grid cell [0, 1).
        Type-generic (vec3f or vec3d).
    offset : wp.vec3i
        Grid offset from base grid point (includes offset_start adjustment).
    order : wp.int32
        Spline order.
    mesh_dims : wp.vec3i
        Mesh dimensions (for scaling to Cartesian coordinates).

    Returns
    -------
    diag : vec3 (Any)
        ``(Hxx, Hyy, Hzz)`` in fractional mesh-space.
    off : vec3 (Any)
        ``(Hxy, Hxz, Hyz)`` in fractional mesh-space.
    """
    t0 = theta[0]
    half_order = type(t0)(order) * type(t0)(0.5)
    zero = type(t0)(0.0)
    order_f = type(t0)(order)

    u_x = half_order + t0 - type(t0)(offset[0])
    u_y = half_order + theta[1] - type(t0)(offset[1])
    u_z = half_order + theta[2] - type(t0)(offset[2])

    if (
        u_x < zero
        or u_x >= order_f
        or u_y < zero
        or u_y >= order_f
        or u_z < zero
        or u_z >= order_f
    ):
        zvec = type(theta)(zero, zero, zero)
        return zvec, zvec

    w_x = bspline_weight(u_x, order)
    w_y = bspline_weight(u_y, order)
    w_z = bspline_weight(u_z, order)

    dw_x = bspline_derivative(u_x, order)
    dw_y = bspline_derivative(u_y, order)
    dw_z = bspline_derivative(u_z, order)

    ddw_x = bspline_second_derivative(u_x, order)
    ddw_y = bspline_second_derivative(u_y, order)
    ddw_z = bspline_second_derivative(u_z, order)

    md_x = type(t0)(mesh_dims[0])
    md_y = type(t0)(mesh_dims[1])
    md_z = type(t0)(mesh_dims[2])

    diag = type(theta)(
        ddw_x * w_y * w_z * md_x * md_x,
        w_x * ddw_y * w_z * md_y * md_y,
        w_x * w_y * ddw_z * md_z * md_z,
    )
    off = type(theta)(
        dw_x * dw_y * w_z * md_x * md_y,
        dw_x * w_y * dw_z * md_x * md_z,
        w_x * dw_y * dw_z * md_y * md_z,
    )
    return diag, off


@wp.func
def wrap_grid_index(idx: wp.int32, dim: wp.int32) -> wp.int32:
    """Wrap grid index for periodic boundaries."""
    return ((idx % dim) + dim) % dim


###########################################################################################
########################### Single-System Warp Kernels ####################################
###########################################################################################


@wp.kernel
def _bspline_weight_kernel(
    u: wp.array(dtype=Any),
    order: wp.int32,
    weights: wp.array(dtype=Any),
):
    """Compute B-spline weights for an array of inputs.

    Parameters
    ----------
    u : wp.array, shape (N,)
        Input values.
    order : wp.int32
        Spline order.
    weights : wp.array, shape (N,)
        Output weights.
    """
    i = wp.tid()
    weights[i] = bspline_weight(u[i], order)


@wp.kernel
def _bspline_spread_kernel(
    positions: wp.array(dtype=Any),
    values: wp.array(dtype=Any),
    cell_inv_t: wp.array(dtype=Any),
    order: wp.int32,
    mesh: wp.array3d(dtype=Any),
):
    """Spread (scatter) values from atoms to a 3D mesh using B-spline interpolation.

    For each atom, distributes its value to nearby grid points weighted by the
    B-spline basis function. This is the adjoint operation to gathering.

    Formula: mesh[g] += value[atom] * w(atom, g)

    where w(atom, g) is the product of 1D B-spline weights in each dimension.

    Launch Grid
    -----------
    dim = [num_atoms, order^3]

    Each thread handles one (atom, grid_point) pair within the atom's stencil.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates in Cartesian space.
    values : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Values to spread (e.g., charges).
    cell_inv_t : wp.array, shape (1, 3, 3), dtype=wp.mat33f or wp.mat33d
        Transpose of inverse cell matrix for fractional coordinate conversion.
    order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended for PME.
    mesh : wp.array3d, shape (nx, ny, nz), dtype=wp.float32 or wp.float64
        OUTPUT: 3D mesh to accumulate values into. Must be zero-initialized.

    Notes
    -----
    - Uses atomic adds for thread-safe accumulation to shared grid points.
    - Grid indices are wrapped using periodic boundary conditions.
    - Threads with 1e-8 weight skip the atomic add for efficiency.
    """
    atom_idx, point_idx = wp.tid()

    mesh_dims = wp.vec3i(mesh.shape[0], mesh.shape[1], mesh.shape[2])
    position = positions[atom_idx]
    value = values[atom_idx]

    base_grid, theta = compute_fractional_coords(position, cell_inv_t[0], mesh_dims)
    offset = bspline_grid_offset(point_idx, order, theta)
    weight = bspline_weight_3d(theta, offset, order)

    if weight > type(value)(0.0):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        wp.atomic_add(mesh, gx, gy, gz, value * weight)


@wp.kernel
def _bspline_gather_kernel(
    positions: wp.array(dtype=Any),
    cell_inv_t: wp.array(dtype=Any),
    order: wp.int32,
    mesh: wp.array3d(dtype=Any),
    output: wp.array(dtype=Any),
):
    """Gather (interpolate) values from a 3D mesh to atom positions using B-splines.

    For each atom, interpolates the mesh value at its position by summing nearby
    grid points weighted by the B-spline basis function.

    Formula: output[atom] = Σ_g mesh[g] * w(atom, g)

    where the sum is over the order^3 grid points in the atom's stencil.

    Launch Grid
    -----------
    dim = [num_atoms, order^3]

    Each thread handles one (atom, grid_point) pair within the atom's stencil.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates in Cartesian space.
    cell_inv_t : wp.array, shape (1, 3, 3), dtype=wp.mat33f or wp.mat33d
        Transpose of inverse cell matrix for fractional coordinate conversion.
    order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended for PME.
    mesh : wp.array3d, shape (nx, ny, nz), dtype=wp.float32 or wp.float64
        3D mesh containing values to interpolate (e.g., electrostatic potential).
    output : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Interpolated values per atom. Must be zero-initialized.

    Notes
    -----
    - Uses atomic adds since multiple threads contribute to each atom's output.
    - Grid indices are wrapped using periodic boundary conditions.
    - Threads with 1e-8 weight skip the atomic add for efficiency.
    """
    atom_idx, point_idx = wp.tid()

    mesh_dims = wp.vec3i(mesh.shape[0], mesh.shape[1], mesh.shape[2])
    position = positions[atom_idx]

    base_grid, theta = compute_fractional_coords(position, cell_inv_t[0], mesh_dims)
    offset = bspline_grid_offset(point_idx, order, theta)
    weight = bspline_weight_3d(theta, offset, order)

    mesh_val = mesh[0, 0, 0]  # Get type reference
    if weight > type(mesh_val)(1e-8):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        mesh_val = mesh[gx, gy, gz]
        wp.atomic_add(output, atom_idx, mesh_val * weight)


@wp.kernel
def _bspline_gather_vec3_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell_inv_t: wp.array(dtype=Any),
    order: wp.int32,
    mesh: wp.array3d(dtype=Any),
    output: wp.array(dtype=Any),
):
    """Gather charge-weighted 3D vector values from mesh to atoms using B-splines.

    Similar to _bspline_gather_kernel but multiplies by the atom's charge and
    outputs to a 3D vector array (for use with vector-valued mesh fields).

    Formula: output[atom] = q[atom] * Σ_g mesh[g] * w(atom, g)

    Launch Grid
    -----------
    dim = [num_atoms, order^3]

    Each thread handles one (atom, grid_point) pair within the atom's stencil.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates in Cartesian space.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges (or other scalar weights).
    cell_inv_t : wp.array, shape (1, 3, 3), dtype=wp.mat33f or wp.mat33d
        Transpose of inverse cell matrix for fractional coordinate conversion.
    order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended for PME.
    mesh : wp.array3d, shape (nx, ny, nz), dtype=wp.vec3f or wp.vec3d
        3D mesh containing vector values to interpolate.
    output : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Charge-weighted interpolated vectors per atom. Must be zero-initialized.

    Notes
    -----
    - Uses atomic adds since multiple threads contribute to each atom's output.
    - Grid indices are wrapped using periodic boundary conditions.
    - Threads with 1e-8 weight or less skip the atomic add for efficiency.
    """
    atom_idx, point_idx = wp.tid()

    mesh_dims = wp.vec3i(mesh.shape[0], mesh.shape[1], mesh.shape[2])
    position = positions[atom_idx]
    charge = charges[atom_idx]

    base_grid, theta = compute_fractional_coords(position, cell_inv_t[0], mesh_dims)
    offset = bspline_grid_offset(point_idx, order, theta)
    weight = bspline_weight_3d(theta, offset, order)

    if weight > type(charge)(1e-8):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        mesh_val = mesh[gx, gy, gz]
        wp.atomic_add(output, atom_idx, charge * mesh_val * weight)


@wp.kernel
def _bspline_gather_gradient_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell_inv_t: wp.array(dtype=Any),
    order: wp.int32,
    mesh: wp.array3d(dtype=Any),
    forces: wp.array(dtype=Any),
):
    """Compute forces by gathering mesh gradients using B-spline derivatives.

    Computes:

    .. math::

        F_i = -q_i \\sum_g \\phi(g) \\nabla w(r_i, g)

    The gradient ∇w is computed in fractional coordinates and then transformed
    to Cartesian coordinates via the cell matrix.

    Launch Grid
    -----------
    dim = [num_atoms, order^3]

    Each thread handles one (atom, grid_point) pair within the atom's stencil.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates in Cartesian space.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell_inv_t : wp.array, shape (1, 3, 3), dtype=wp.mat33f or wp.mat33d
        Transpose of inverse cell matrix for fractional coordinate conversion.
    order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended for PME.
    mesh : wp.array3d, shape (nx, ny, nz), dtype=wp.float32 or wp.float64
        3D mesh containing potential values (e.g., electrostatic potential φ).
    forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Forces per atom in Cartesian coordinates. Must be zero-initialized.

    Notes
    -----
    - Uses atomic adds since multiple threads contribute to each atom's force.
    - The gradient is computed in fractional coordinates, then transformed:
      F_cart = cell_inv_t^T * F_frac
    - Threads with zero gradient magnitude skip the atomic add for efficiency.
    - Grid indices are wrapped using periodic boundary conditions.
    """
    atom_idx, point_idx = wp.tid()

    mesh_dims = wp.vec3i(mesh.shape[0], mesh.shape[1], mesh.shape[2])
    position = positions[atom_idx]
    charge = charges[atom_idx]

    base_grid, theta = compute_fractional_coords(position, cell_inv_t[0], mesh_dims)
    offset = bspline_grid_offset(point_idx, order, theta)
    grad_frac = bspline_weight_gradient_3d(theta, offset, order, mesh_dims)

    grad_mag = wp.abs(grad_frac[0]) + wp.abs(grad_frac[1]) + wp.abs(grad_frac[2])

    if grad_mag > type(charge)(0.0):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        mesh_val = mesh[gx, gy, gz]

        force_frac = type(position)(
            -charge * mesh_val * grad_frac[0],
            -charge * mesh_val * grad_frac[1],
            -charge * mesh_val * grad_frac[2],
        )
        force = wp.transpose(cell_inv_t[0]) * force_frac

        wp.atomic_add(forces, atom_idx, force)


###########################################################################################
########################### Batch Warp Kernels #############################################
###########################################################################################


@wp.kernel
def _batch_bspline_spread_kernel(
    positions: wp.array(dtype=Any),
    values: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cell_inv_t: wp.array(dtype=Any),  # (B, 3, 3)
    order: wp.int32,
    mesh: wp.array(dtype=Any, ndim=4),  # (B, nx, ny, nz)
):
    """Spread values from atoms to a batched 4D mesh using B-splines.

    Batched version of _bspline_spread_kernel for multiple systems. Each atom
    is assigned to a system via batch_idx, and values are spread to that
    system's mesh slice.

    Formula: mesh[sys, g] += value[atom] * w(atom, g)

    Launch Grid
    -----------
    dim = [num_atoms_total, order^3]

    Each thread handles one (atom, grid_point) pair within the atom's stencil.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates for all systems concatenated.
    values : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Values to spread (e.g., charges) for all systems.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    cell_inv_t : wp.array, shape (B, 3, 3), dtype=wp.mat33f or wp.mat33d
        Per-system transpose of inverse cell matrix.
    order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended for PME.
    mesh : wp.array4d, shape (B, nx, ny, nz), dtype=wp.float32 or wp.float64
        OUTPUT: 4D mesh (batch × spatial) to accumulate values. Must be zero-initialized.

    Notes
    -----
    - Uses atomic adds for thread-safe accumulation to shared grid points.
    - Each system uses its own cell matrix for fractional coordinate conversion.
    - Grid indices are wrapped using periodic boundary conditions.
    - Threads with 1e-8 weight or less skip the atomic add for efficiency.
    """
    atom_idx, point_idx = wp.tid()

    sys_idx = batch_idx[atom_idx]
    mesh_dims = wp.vec3i(mesh.shape[1], mesh.shape[2], mesh.shape[3])
    position = positions[atom_idx]
    value = values[atom_idx]

    base_grid, theta = compute_fractional_coords(
        position, cell_inv_t[sys_idx], mesh_dims
    )
    offset = bspline_grid_offset(point_idx, order, theta)
    weight = bspline_weight_3d(theta, offset, order)

    if weight > type(value)(1e-8):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        wp.atomic_add(mesh, sys_idx, gx, gy, gz, value * weight)


@wp.kernel
def _batch_bspline_gather_kernel(
    positions: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cell_inv_t: wp.array(dtype=Any),  # (B, 3, 3)
    order: wp.int32,
    mesh: wp.array(dtype=Any, ndim=4),  # (B, nx, ny, nz)
    output: wp.array(dtype=Any),
):
    """Gather values from a batched 4D mesh to atom positions using B-splines.

    Batched version of _bspline_gather_kernel for multiple systems. Each atom
    reads from its assigned system's mesh slice via batch_idx.

    Formula: output[atom] = Σ_g mesh[sys, g] * w(atom, g)

    Launch Grid
    -----------
    dim = [num_atoms_total, order^3]

    Each thread handles one (atom, grid_point) pair within the atom's stencil.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates for all systems concatenated.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    cell_inv_t : wp.array, shape (B, 3, 3), dtype=wp.mat33f or wp.mat33d
        Per-system transpose of inverse cell matrix.
    order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended for PME.
    mesh : wp.array4d, shape (B, nx, ny, nz), dtype=wp.float32 or wp.float64
        4D mesh (batch × spatial) containing values to interpolate.
    output : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        OUTPUT: Interpolated values per atom. Must be zero-initialized.

    Notes
    -----
    - Uses atomic adds since multiple threads contribute to each atom's output.
    - Each system uses its own cell matrix for fractional coordinate conversion.
    - Grid indices are wrapped using periodic boundary conditions.
    - Threads with 1e-8 weight or less skip the atomic add for efficiency.
    """
    atom_idx, point_idx = wp.tid()

    sys_idx = batch_idx[atom_idx]
    mesh_dims = wp.vec3i(mesh.shape[1], mesh.shape[2], mesh.shape[3])
    position = positions[atom_idx]

    base_grid, theta = compute_fractional_coords(
        position, cell_inv_t[sys_idx], mesh_dims
    )
    offset = bspline_grid_offset(point_idx, order, theta)
    weight = bspline_weight_3d(theta, offset, order)

    mesh_val = mesh[0, 0, 0, 0]  # Get type reference
    if weight > type(mesh_val)(1e-8):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        mesh_val = mesh[sys_idx, gx, gy, gz]
        wp.atomic_add(output, atom_idx, mesh_val * weight)


@wp.kernel
def _batch_bspline_gather_vec3_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cell_inv_t: wp.array(dtype=Any),  # (B, 3, 3)
    order: wp.int32,
    mesh: wp.array(dtype=Any, ndim=4),  # (B, nx, ny, nz)
    output: wp.array(dtype=Any),
):
    """Gather charge-weighted 3D vector values from batched mesh using B-splines.

    Batched version of _bspline_gather_vec3_kernel for multiple systems.

    Formula: output[atom] = q[atom] * Σ_g mesh[sys, g] * w(atom, g)

    Launch Grid
    -----------
    dim = [num_atoms_total, order^3]

    Each thread handles one (atom, grid_point) pair within the atom's stencil.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates for all systems concatenated.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges (or other scalar weights).
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    cell_inv_t : wp.array, shape (B, 3, 3), dtype=wp.mat33f or wp.mat33d
        Per-system transpose of inverse cell matrix.
    order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended for PME.
    mesh : wp.array4d, shape (B, nx, ny, nz), dtype=wp.vec3f or wp.vec3d
        4D mesh (batch × spatial) containing vector values.
    output : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Charge-weighted interpolated vectors per atom. Must be zero-initialized.

    Notes
    -----
    - Uses atomic adds since multiple threads contribute to each atom's output.
    - Each system uses its own cell matrix for fractional coordinate conversion.
    - Grid indices are wrapped using periodic boundary conditions.
    - Threads with 1e-8 weight or less skip the atomic add for efficiency.
    """
    atom_idx, point_idx = wp.tid()

    sys_idx = batch_idx[atom_idx]
    mesh_dims = wp.vec3i(mesh.shape[1], mesh.shape[2], mesh.shape[3])
    position = positions[atom_idx]
    charge = charges[atom_idx]

    base_grid, theta = compute_fractional_coords(
        position, cell_inv_t[sys_idx], mesh_dims
    )
    offset = bspline_grid_offset(point_idx, order, theta)
    weight = bspline_weight_3d(theta, offset, order)

    if weight > type(charge)(1e-8):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        mesh_val = mesh[sys_idx, gx, gy, gz]
        wp.atomic_add(output, atom_idx, charge * mesh_val * weight)


@wp.kernel
def _batch_bspline_gather_gradient_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cell_inv_t: wp.array(dtype=Any),  # (B, 3, 3)
    order: wp.int32,
    mesh: wp.array(dtype=Any, ndim=4),  # (B, nx, ny, nz)
    forces: wp.array(dtype=Any),
):
    """Compute forces by gathering mesh gradients from batched mesh using B-spline derivatives.

    Computes:

    .. math::

        F_i = -q_i \\sum_g \\phi(g) \\nabla w(r_i, g)

    The gradient ∇w is computed in fractional coordinates and then transformed
    to Cartesian coordinates via each system's cell matrix.

    Launch Grid
    -----------
    dim = [num_atoms_total, order^3]

    Each thread handles one (atom, grid_point) pair within the atom's stencil.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates for all systems concatenated.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    cell_inv_t : wp.array, shape (B, 3, 3), dtype=wp.mat33f or wp.mat33d
        Per-system transpose of inverse cell matrix.
    order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended for PME.
    mesh : wp.array4d, shape (B, nx, ny, nz), dtype=wp.float32 or wp.float64
        4D mesh (batch × spatial) containing potential values.
    forces : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Forces per atom in Cartesian coordinates. Must be zero-initialized.

    Notes
    -----
    - Uses atomic adds since multiple threads contribute to each atom's force.
    - The gradient is computed in fractional coordinates, then transformed:
      F_cart = cell_inv_t[sys]^T * F_frac
    - Each system uses its own cell matrix for the transformation.
    - Threads with zero gradient magnitude skip the atomic add for efficiency.
    - Grid indices are wrapped using periodic boundary conditions.
    """
    atom_idx, point_idx = wp.tid()

    sys_idx = batch_idx[atom_idx]
    mesh_dims = wp.vec3i(mesh.shape[1], mesh.shape[2], mesh.shape[3])
    position = positions[atom_idx]
    charge = charges[atom_idx]

    base_grid, theta = compute_fractional_coords(
        position, cell_inv_t[sys_idx], mesh_dims
    )
    offset = bspline_grid_offset(point_idx, order, theta)
    grad_frac = bspline_weight_gradient_3d(theta, offset, order, mesh_dims)

    grad_mag = wp.abs(grad_frac[0]) + wp.abs(grad_frac[1]) + wp.abs(grad_frac[2])

    if grad_mag > type(charge)(0.0):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        mesh_val = mesh[sys_idx, gx, gy, gz]

        force_frac = type(position)(
            -charge * mesh_val * grad_frac[0],
            -charge * mesh_val * grad_frac[1],
            -charge * mesh_val * grad_frac[2],
        )
        force = wp.transpose(cell_inv_t[sys_idx]) * force_frac

        wp.atomic_add(forces, atom_idx, force)


###########################################################################################
########################### Multi-Channel Warp Kernels ####################################
###########################################################################################


@wp.kernel
def _bspline_spread_channels_kernel(
    positions: wp.array(dtype=Any),
    values: wp.array2d(dtype=Any),  # (N, C)
    cell_inv_t: wp.array(dtype=Any),
    order: wp.int32,
    mesh: wp.array(dtype=Any, ndim=4),  # (C, nx, ny, nz)
):
    """Spread multi-channel values from atoms to mesh using B-splines.

    Similar to _bspline_spread_kernel but handles multiple channels per atom,
    useful for multipole moments (e.g., monopole + dipole + quadrupole).

    Formula: mesh[c, g] += values[atom, c] * w(atom, g)

    for each channel c = 0, 1, ..., C-1.

    Launch Grid
    -----------
    dim = [num_atoms, order^3]

    Each thread handles one (atom, grid_point) pair and iterates over all channels.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates in Cartesian space.
    values : wp.array2d, shape (N, C), dtype=wp.float32 or wp.float64
        Multi-channel values to spread (e.g., multipole moments).
    cell_inv_t : wp.array, shape (1, 3, 3), dtype=wp.mat33f or wp.mat33d
        Transpose of inverse cell matrix for fractional coordinate conversion.
    order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended for PME.
    mesh : wp.array4d, shape (C, nx, ny, nz), dtype=wp.float32 or wp.float64
        OUTPUT: 4D mesh (channels × spatial) to accumulate values. Must be zero-initialized.

    Notes
    -----
    - Uses atomic adds for thread-safe accumulation to shared grid points.
    - Each channel is spread independently to its own mesh slice.
    - Grid indices are wrapped using periodic boundary conditions.
    - Threads with 1e-8 weight or less skip the atomic adds for efficiency.
    """
    atom_idx, point_idx = wp.tid()

    num_channels = values.shape[1]
    mesh_dims = wp.vec3i(mesh.shape[1], mesh.shape[2], mesh.shape[3])
    position = positions[atom_idx]

    base_grid, theta = compute_fractional_coords(position, cell_inv_t[0], mesh_dims)
    offset = bspline_grid_offset(point_idx, order, theta)
    weight = bspline_weight_3d(theta, offset, order)

    val = values[0, 0]  # Get type reference
    if weight > type(val)(1e-8):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        # Spread each channel
        for c in range(num_channels):
            val = values[atom_idx, c]
            wp.atomic_add(mesh, c, gx, gy, gz, val * weight)


@wp.kernel
def _bspline_gather_channels_kernel(
    positions: wp.array(dtype=Any),
    cell_inv_t: wp.array(dtype=Any),
    order: wp.int32,
    mesh: wp.array(dtype=Any, ndim=4),  # (C, nx, ny, nz)
    output: wp.array2d(dtype=Any),  # (N, C)
):
    """Gather multi-channel values from mesh to atoms using B-splines.

    Similar to _bspline_gather_kernel but handles multiple channels,
    useful for multipole-based methods.

    Formula: output[atom, c] = Σ_g mesh[c, g] * w(atom, g)

    for each channel c = 0, 1, ..., C-1.

    Launch Grid
    -----------
    dim = [num_atoms, order^3]

    Each thread handles one (atom, grid_point) pair and iterates over all channels.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates in Cartesian space.
    cell_inv_t : wp.array, shape (1, 3, 3), dtype=wp.mat33f or wp.mat33d
        Transpose of inverse cell matrix for fractional coordinate conversion.
    order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended for PME.
    mesh : wp.array4d, shape (C, nx, ny, nz), dtype=wp.float32 or wp.float64
        4D mesh (channels × spatial) containing values to interpolate.
    output : wp.array2d, shape (N, C), dtype=wp.float32 or wp.float64
        OUTPUT: Interpolated multi-channel values per atom. Must be zero-initialized.

    Notes
    -----
    - Uses atomic adds since multiple threads contribute to each atom's output.
    - Each channel is gathered independently from its own mesh slice.
    - Grid indices are wrapped using periodic boundary conditions.
    - Threads with 1e-8 weight or less skip the atomic adds for efficiency.
    """
    atom_idx, point_idx = wp.tid()

    num_channels = mesh.shape[0]
    mesh_dims = wp.vec3i(mesh.shape[1], mesh.shape[2], mesh.shape[3])
    position = positions[atom_idx]

    base_grid, theta = compute_fractional_coords(position, cell_inv_t[0], mesh_dims)
    offset = bspline_grid_offset(point_idx, order, theta)
    weight = bspline_weight_3d(theta, offset, order)

    mesh_val = mesh[0, 0, 0, 0]  # Get type reference
    if weight > type(mesh_val)(1e-8):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        # Gather each channel
        for c in range(num_channels):
            mesh_val = mesh[c, gx, gy, gz]
            wp.atomic_add(output, atom_idx, c, mesh_val * weight)


@wp.kernel
def _batch_bspline_spread_channels_kernel(
    positions: wp.array(dtype=Any),
    values: wp.array2d(dtype=Any),  # (N, C)
    batch_idx: wp.array(dtype=wp.int32),
    cell_inv_t: wp.array(dtype=Any),  # (B, 3, 3)
    order: wp.int32,
    num_channels: wp.int32,
    mesh: wp.array4d(dtype=Any),  # (B*C, nx, ny, nz) - flattened batch*channel
):
    """Spread multi-channel values from atoms to batched mesh using B-splines.

    Batched version of _bspline_spread_channels_kernel. Due to Warp's 4D array
    limit, the batch and channel dimensions are flattened into a single dimension.

    Formula: mesh[sys*C + c, g] += values[atom, c] * w(atom, g)

    Launch Grid
    -----------
    dim = [num_atoms_total, order^3]

    Each thread handles one (atom, grid_point) pair and iterates over all channels.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates for all systems concatenated.
    values : wp.array2d, shape (N_total, C), dtype=wp.float32 or wp.float64
        Multi-channel values to spread (e.g., multipole moments).
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    cell_inv_t : wp.array, shape (B, 3, 3), dtype=wp.mat33f or wp.mat33d
        Per-system transpose of inverse cell matrix.
    order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended for PME.
    num_channels : wp.int32
        Number of channels (C).
    mesh : wp.array4d, shape (B*C, nx, ny, nz), dtype=wp.float32 or wp.float64
        OUTPUT: Flattened 4D mesh to accumulate values. Must be zero-initialized.

    Notes
    -----
    - Mesh storage: (B*C, nx, ny, nz) with flat_idx = sys_idx * C + channel_idx.
    - Uses atomic adds for thread-safe accumulation to shared grid points.
    - Each system uses its own cell matrix for fractional coordinate conversion.
    - Grid indices are wrapped using periodic boundary conditions.
    - Threads with 1e-8 weight or less skip the atomic adds for efficiency.
    """
    atom_idx, point_idx = wp.tid()

    sys_idx = batch_idx[atom_idx]
    mesh_dims = wp.vec3i(mesh.shape[1], mesh.shape[2], mesh.shape[3])
    position = positions[atom_idx]

    base_grid, theta = compute_fractional_coords(
        position, cell_inv_t[sys_idx], mesh_dims
    )
    offset = bspline_grid_offset(point_idx, order, theta)
    weight = bspline_weight_3d(theta, offset, order)

    val = values[0, 0]  # Get type reference
    if weight > type(val)(1e-8):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        # Spread each channel using flattened batch*channel indexing
        for c in range(num_channels):
            flat_idx = sys_idx * num_channels + c
            val = values[atom_idx, c]
            wp.atomic_add(mesh, flat_idx, gx, gy, gz, val * weight)


@wp.kernel
def _batch_bspline_gather_channels_kernel(
    positions: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cell_inv_t: wp.array(dtype=Any),  # (B, 3, 3)
    order: wp.int32,
    num_channels: wp.int32,
    mesh: wp.array4d(dtype=Any),  # (B*C, nx, ny, nz) - flattened batch*channel
    output: wp.array2d(dtype=Any),  # (N, C)
):
    """Gather multi-channel values from batched mesh to atoms using B-splines.

    Batched version of _bspline_gather_channels_kernel. Due to Warp's 4D array
    limit, the batch and channel dimensions are flattened into a single dimension.

    Formula: output[atom, c] = Σ_g mesh[sys*C + c, g] * w(atom, g)

    Launch Grid
    -----------
    dim = [num_atoms_total, order^3]

    Each thread handles one (atom, grid_point) pair and iterates over all channels.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates for all systems concatenated.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    cell_inv_t : wp.array, shape (B, 3, 3), dtype=wp.mat33f or wp.mat33d
        Per-system transpose of inverse cell matrix.
    order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended for PME.
    num_channels : wp.int32
        Number of channels (C).
    mesh : wp.array4d, shape (B*C, nx, ny, nz), dtype=wp.float32 or wp.float64
        Flattened 4D mesh (batch*channels × spatial) containing values.
    output : wp.array2d, shape (N_total, C), dtype=wp.float32 or wp.float64
        OUTPUT: Interpolated multi-channel values per atom. Must be zero-initialized.

    Notes
    -----
    - Mesh storage: (B*C, nx, ny, nz) with flat_idx = sys_idx * C + channel_idx.
    - Uses atomic adds since multiple threads contribute to each atom's output.
    - Each system uses its own cell matrix for fractional coordinate conversion.
    - Grid indices are wrapped using periodic boundary conditions.
    - Threads with 1e-8 weight or less skip the atomic adds for efficiency.
    """
    atom_idx, point_idx = wp.tid()

    sys_idx = batch_idx[atom_idx]
    mesh_dims = wp.vec3i(mesh.shape[1], mesh.shape[2], mesh.shape[3])
    position = positions[atom_idx]

    base_grid, theta = compute_fractional_coords(
        position, cell_inv_t[sys_idx], mesh_dims
    )
    offset = bspline_grid_offset(point_idx, order, theta)
    weight = bspline_weight_3d(theta, offset, order)

    mesh_val = mesh[0, 0, 0, 0]  # Get type reference
    if weight > type(mesh_val)(1e-8):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        # Gather each channel using flattened batch*channel indexing
        for c in range(num_channels):
            flat_idx = sys_idx * num_channels + c
            mesh_val = mesh[flat_idx, gx, gy, gz]
            wp.atomic_add(output, atom_idx, c, mesh_val * weight)


###########################################################################################
########################### Kernel Overloads for Dtype Flexibility #########################
###########################################################################################

# Type lists for creating overloads
_T = [wp.float32, wp.float64]
_V = [wp.vec3f, wp.vec3d]
_M = [wp.mat33f, wp.mat33d]

# Single-system kernel overloads
_bspline_weight_kernel_overload = {}
_bspline_spread_kernel_overload = {}
_bspline_gather_kernel_overload = {}
_bspline_gather_vec3_kernel_overload = {}
_bspline_gather_gradient_kernel_overload = {}

# Batch kernel overloads
_batch_bspline_spread_kernel_overload = {}
_batch_bspline_gather_kernel_overload = {}
_batch_bspline_gather_vec3_kernel_overload = {}
_batch_bspline_gather_gradient_kernel_overload = {}

# Multi-channel kernel overloads
_bspline_spread_channels_kernel_overload = {}
_bspline_gather_channels_kernel_overload = {}
_batch_bspline_spread_channels_kernel_overload = {}
_batch_bspline_gather_channels_kernel_overload = {}

for t, v, m in zip(_T, _V, _M):
    # Single-system kernels
    _bspline_weight_kernel_overload[t] = wp.overload(
        _bspline_weight_kernel,
        [
            wp.array(dtype=t),  # u
            wp.int32,  # order
            wp.array(dtype=t),  # weights
        ],
    )
    _bspline_spread_kernel_overload[t] = wp.overload(
        _bspline_spread_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # values
            wp.array(dtype=m),  # cell_inv_t
            wp.int32,  # order
            wp.array3d(dtype=t),  # mesh
        ],
    )
    _bspline_gather_kernel_overload[t] = wp.overload(
        _bspline_gather_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=m),  # cell_inv_t
            wp.int32,  # order
            wp.array3d(dtype=t),  # mesh
            wp.array(dtype=t),  # output
        ],
    )
    _bspline_gather_vec3_kernel_overload[t] = wp.overload(
        _bspline_gather_vec3_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=m),  # cell_inv_t
            wp.int32,  # order
            wp.array3d(dtype=v),  # mesh
            wp.array(dtype=v),  # output
        ],
    )
    _bspline_gather_gradient_kernel_overload[t] = wp.overload(
        _bspline_gather_gradient_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=m),  # cell_inv_t
            wp.int32,  # order
            wp.array3d(dtype=t),  # mesh
            wp.array(dtype=v),  # forces
        ],
    )

    # Batch kernels
    _batch_bspline_spread_kernel_overload[t] = wp.overload(
        _batch_bspline_spread_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # values
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=m),  # cell_inv_t
            wp.int32,  # order
            wp.array(dtype=t, ndim=4),  # mesh
        ],
    )
    _batch_bspline_gather_kernel_overload[t] = wp.overload(
        _batch_bspline_gather_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=m),  # cell_inv_t
            wp.int32,  # order
            wp.array(dtype=t, ndim=4),  # mesh
            wp.array(dtype=t),  # output
        ],
    )
    _batch_bspline_gather_vec3_kernel_overload[t] = wp.overload(
        _batch_bspline_gather_vec3_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=m),  # cell_inv_t
            wp.int32,  # order
            wp.array(dtype=v, ndim=4),  # mesh
            wp.array(dtype=v),  # output
        ],
    )
    _batch_bspline_gather_gradient_kernel_overload[t] = wp.overload(
        _batch_bspline_gather_gradient_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=m),  # cell_inv_t
            wp.int32,  # order
            wp.array(dtype=t, ndim=4),  # mesh
            wp.array(dtype=v),  # forces
        ],
    )

    # Multi-channel kernels
    _bspline_spread_channels_kernel_overload[t] = wp.overload(
        _bspline_spread_channels_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array2d(dtype=t),  # values
            wp.array(dtype=m),  # cell_inv_t
            wp.int32,  # order
            wp.array(dtype=t, ndim=4),  # mesh
        ],
    )
    _bspline_gather_channels_kernel_overload[t] = wp.overload(
        _bspline_gather_channels_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=m),  # cell_inv_t
            wp.int32,  # order
            wp.array(dtype=t, ndim=4),  # mesh
            wp.array2d(dtype=t),  # output
        ],
    )
    _batch_bspline_spread_channels_kernel_overload[t] = wp.overload(
        _batch_bspline_spread_channels_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array2d(dtype=t),  # values
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=m),  # cell_inv_t
            wp.int32,  # order
            wp.int32,  # num_channels
            wp.array4d(dtype=t),  # mesh
        ],
    )
    _batch_bspline_gather_channels_kernel_overload[t] = wp.overload(
        _batch_bspline_gather_channels_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=m),  # cell_inv_t
            wp.int32,  # order
            wp.int32,  # num_channels
            wp.array4d(dtype=t),  # mesh
            wp.array2d(dtype=t),  # output
        ],
    )


###########################################################################################
########################### Warp Launcher Functions #######################################
###########################################################################################


def bspline_weight_launcher(
    u: wp.array,
    order: int,
    weights: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Compute B-spline weights for an array of inputs.

    Parameters
    ----------
    u : wp.array, shape (N,)
        Input values.
    order : int
        B-spline order.
    weights : wp.array, shape (N,)
        Output weights.
    wp_dtype : type
        Warp scalar dtype.
    device : str | None
        Warp device string.
    """
    num_points = u.shape[0]
    kernel = _bspline_weight_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=num_points,
        inputs=[u, wp.int32(order)],
        outputs=[weights],
        device=device,
    )


def spline_spread(
    positions: wp.array,
    values: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Spread values from atoms to mesh using B-spline interpolation.

    Framework-agnostic launcher for single-system spline spread.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    values : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Values to spread (e.g., charges).
    cell_inv_t : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
        Transpose of inverse cell matrix.
    order : int
        B-spline order (1-4).
    mesh : wp.array, shape (nx, ny, nz), dtype=wp.float32 or wp.float64
        OUTPUT: Mesh to accumulate values. Must be zero-initialized.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = positions.shape[0]
    num_points = order**3

    kernel = _bspline_spread_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(num_atoms, num_points),
        inputs=[positions, values, cell_inv_t, wp.int32(order)],
        outputs=[mesh],
        device=device,
    )


def spline_gather(
    positions: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    output: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Gather values from mesh to atoms using B-spline interpolation.

    Framework-agnostic launcher for single-system spline gather.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    cell_inv_t : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
        Transpose of inverse cell matrix.
    order : int
        B-spline order (1-4).
    mesh : wp.array, shape (nx, ny, nz), dtype=wp.float32 or wp.float64
        Mesh to interpolate from.
    output : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Interpolated values per atom. Must be zero-initialized.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = positions.shape[0]
    num_points = order**3

    kernel = _bspline_gather_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(num_atoms, num_points),
        inputs=[positions, cell_inv_t, wp.int32(order), mesh],
        outputs=[output],
        device=device,
    )


def spline_gather_vec3(
    positions: wp.array,
    charges: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    output: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Gather charge-weighted vector values from mesh using B-splines.

    Framework-agnostic launcher for single-system vec3 spline gather.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell_inv_t : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
        Transpose of inverse cell matrix.
    order : int
        B-spline order (1-4).
    mesh : wp.array, shape (nx, ny, nz), dtype=wp.vec3f or wp.vec3d
        Vector-valued mesh to interpolate from.
    output : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Charge-weighted interpolated vectors. Must be zero-initialized.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = positions.shape[0]
    num_points = order**3

    kernel = _bspline_gather_vec3_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(num_atoms, num_points),
        inputs=[positions, charges, cell_inv_t, wp.int32(order), mesh],
        outputs=[output],
        device=device,
    )


def spline_gather_gradient(
    positions: wp.array,
    charges: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    forces: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Compute forces using B-spline gradient interpolation.

    Framework-agnostic launcher for single-system spline gradient gather.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell_inv_t : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
        Transpose of inverse cell matrix.
    order : int
        B-spline order (1-4).
    mesh : wp.array, shape (nx, ny, nz), dtype=wp.float32 or wp.float64
        Potential mesh.
    forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Forces per atom. Must be zero-initialized.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = positions.shape[0]
    num_points = order**3

    kernel = _bspline_gather_gradient_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(num_atoms, num_points),
        inputs=[positions, charges, cell_inv_t, wp.int32(order), mesh],
        outputs=[forces],
        device=device,
    )


def batch_spline_spread(
    positions: wp.array,
    values: wp.array,
    batch_idx: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Spread values from atoms to batched mesh using B-spline interpolation.

    Framework-agnostic launcher for batched spline spread.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic positions for all systems.
    values : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Values to spread.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom.
    cell_inv_t : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Per-system transpose of inverse cell matrix.
    order : int
        B-spline order (1-4).
    mesh : wp.array, shape (B, nx, ny, nz), dtype=wp.float32 or wp.float64
        OUTPUT: Batched mesh to accumulate values. Must be zero-initialized.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = positions.shape[0]
    num_points = order**3

    kernel = _batch_bspline_spread_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(num_atoms, num_points),
        inputs=[positions, values, batch_idx, cell_inv_t, wp.int32(order)],
        outputs=[mesh],
        device=device,
    )


def batch_spline_gather(
    positions: wp.array,
    batch_idx: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    output: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Gather values from batched mesh to atoms using B-spline interpolation.

    Framework-agnostic launcher for batched spline gather.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic positions for all systems.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom.
    cell_inv_t : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Per-system transpose of inverse cell matrix.
    order : int
        B-spline order (1-4).
    mesh : wp.array, shape (B, nx, ny, nz), dtype=wp.float32 or wp.float64
        Batched mesh to interpolate from.
    output : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        OUTPUT: Interpolated values per atom. Must be zero-initialized.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = positions.shape[0]
    num_points = order**3

    kernel = _batch_bspline_gather_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(num_atoms, num_points),
        inputs=[positions, batch_idx, cell_inv_t, wp.int32(order), mesh],
        outputs=[output],
        device=device,
    )


def batch_spline_gather_vec3(
    positions: wp.array,
    charges: wp.array,
    batch_idx: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    output: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Gather charge-weighted vector values from batched mesh using B-splines.

    Framework-agnostic launcher for batched vec3 spline gather.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic positions for all systems.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom.
    cell_inv_t : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Per-system transpose of inverse cell matrix.
    order : int
        B-spline order (1-4).
    mesh : wp.array, shape (B, nx, ny, nz), dtype=wp.vec3f or wp.vec3d
        Batched vector mesh to interpolate from.
    output : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Charge-weighted interpolated vectors. Must be zero-initialized.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = positions.shape[0]
    num_points = order**3

    kernel = _batch_bspline_gather_vec3_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(num_atoms, num_points),
        inputs=[positions, charges, batch_idx, cell_inv_t, wp.int32(order), mesh],
        outputs=[output],
        device=device,
    )


def batch_spline_gather_gradient(
    positions: wp.array,
    charges: wp.array,
    batch_idx: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    forces: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Compute forces using B-spline gradient interpolation from batched mesh.

    Framework-agnostic launcher for batched spline gradient gather.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic positions for all systems.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom.
    cell_inv_t : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Per-system transpose of inverse cell matrix.
    order : int
        B-spline order (1-4).
    mesh : wp.array, shape (B, nx, ny, nz), dtype=wp.float32 or wp.float64
        Batched potential mesh.
    forces : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Forces per atom. Must be zero-initialized.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = positions.shape[0]
    num_points = order**3

    kernel = _batch_bspline_gather_gradient_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(num_atoms, num_points),
        inputs=[positions, charges, batch_idx, cell_inv_t, wp.int32(order), mesh],
        outputs=[forces],
        device=device,
    )


###########################################################################################
########################### Module Exports #################################################
###########################################################################################


__all__ = [
    # Warp functions (@wp.func)
    "bspline_weight",
    "bspline_derivative",
    "bspline_second_derivative",
    "bspline_third_derivative",
    "bspline_weight_3d",
    "bspline_weight_gradient_3d",
    "bspline_weight_hessian_3d",
    "compute_fractional_coords",
    "bspline_grid_offset",
    "wrap_grid_index",
    # Warp kernels (single-system, scalar)
    "_bspline_weight_kernel",
    "_bspline_spread_kernel",
    "_bspline_gather_kernel",
    "_bspline_gather_vec3_kernel",
    "_bspline_gather_gradient_kernel",
    # Warp kernels (batch, scalar)
    "_batch_bspline_spread_kernel",
    "_batch_bspline_gather_kernel",
    "_batch_bspline_gather_vec3_kernel",
    "_batch_bspline_gather_gradient_kernel",
    # Warp kernels (single-system, multi-channel)
    "_bspline_spread_channels_kernel",
    "_bspline_gather_channels_kernel",
    # Warp kernels (batch, multi-channel)
    "_batch_bspline_spread_channels_kernel",
    "_batch_bspline_gather_channels_kernel",
    # Kernel overloads
    "_bspline_weight_kernel_overload",
    "_bspline_spread_kernel_overload",
    "_bspline_gather_kernel_overload",
    "_bspline_gather_vec3_kernel_overload",
    "_bspline_gather_gradient_kernel_overload",
    "_batch_bspline_spread_kernel_overload",
    "_batch_bspline_gather_kernel_overload",
    "_batch_bspline_gather_vec3_kernel_overload",
    "_batch_bspline_gather_gradient_kernel_overload",
    "_bspline_spread_channels_kernel_overload",
    "_bspline_gather_channels_kernel_overload",
    "_batch_bspline_spread_channels_kernel_overload",
    "_batch_bspline_gather_channels_kernel_overload",
    # Warp launchers
    "bspline_weight_launcher",
    "spline_spread",
    "spline_gather",
    "spline_gather_vec3",
    "spline_gather_gradient",
    "batch_spline_spread",
    "batch_spline_gather",
    "batch_spline_gather_vec3",
    "batch_spline_gather_gradient",
]
