# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python reference multipole formulas (l_max ≤ 2) for GTO-Ewald.

Slow, correct, used only by tests as ground truth. NOT imported from
production code paths. Derived in ``tools/derive_multipole.py`` and
cross-validated there.

Convention (matches existing
``nvalchemiops/interactions/electrostatics/multipole_ewald_kernels.py``):

* Real-space pair kernel: ``T^(0)(r) = (erfc(b·r) - erfc(a·r))/r``
* ``a = 1/(2σ)``, ``b = 1/(2σ_c)``, ``σ_c = √(σ² + 1/(4α²))``
* Reduces to standard Ewald ``erfc(αr)/r`` at ``σ → 0``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import erfc


def _sigma_c(alpha: float, sigma: float) -> float:
    """GTO-Ewald combined width ``σ_c = sqrt(σ² + 1/(4α²))``."""
    return math.sqrt(sigma * sigma + 0.25 / (alpha * alpha))


def _ab(alpha: float, sigma: float):
    """Return ``(a, b)`` such that ``a = 1/(2σ)`` and ``b = 1/(2σ_c)``."""
    sc = _sigma_c(alpha, sigma)
    return 1.0 / (2.0 * sigma), 1.0 / (2.0 * sc)


# ---------------------------------------------------------------------------
# Lambdified radial derivatives T^(0..4)(r, alpha, sigma).
# Built once at module import; sympy call is the slow part.
# ---------------------------------------------------------------------------


def _build_radial_T_lambdas():
    """Lambdify ``T^(0..4)(r, α, σ)`` for vectorized numpy evaluation.

    Uses ``scipy`` + ``numpy`` modules so ``sympy.erfc`` resolves to
    ``scipy.special.erfc`` (vectorized) rather than falling back to
    ``math.erfc`` (scalar-only).
    """
    import sympy as sp

    r_sym, a_sym, s_sym = sp.symbols("r alpha sigma", positive=True)
    sigma_c = sp.sqrt(s_sym**2 + sp.Rational(1, 4) / a_sym**2)
    a_expr = 1 / (2 * s_sym)
    b_expr = 1 / (2 * sigma_c)
    T0_sym = (sp.erfc(b_expr * r_sym) - sp.erfc(a_expr * r_sym)) / r_sym
    derivs = [T0_sym]
    for _ in range(4):
        derivs.append(sp.diff(derivs[-1], r_sym))
    return [
        sp.lambdify((r_sym, a_sym, s_sym), e, modules=["scipy", "numpy"])
        for e in derivs
    ]


_RADIAL_T = _build_radial_T_lambdas()


# Public scalar/array T^(n) accessors.


def _T0_radial(r, alpha, sigma):
    return _RADIAL_T[0](r, alpha, sigma)


def _T1_radial(r, alpha, sigma):
    return _RADIAL_T[1](r, alpha, sigma)


def _T2_radial(r, alpha, sigma):
    return _RADIAL_T[2](r, alpha, sigma)


def _T3_radial(r, alpha, sigma):
    return _RADIAL_T[3](r, alpha, sigma)


def _T4_radial(r, alpha, sigma):
    return _RADIAL_T[4](r, alpha, sigma)


# ---------------------------------------------------------------------------
# Scalar Cartesian-tensor accessors (single-pair, slow path used by tests).
# ---------------------------------------------------------------------------


def T0(rx, ry, rz, alpha, sigma):
    """Scalar T^(0)(r) at one point; accepts scalars or numpy arrays."""
    r = np.sqrt(rx**2 + ry**2 + rz**2)
    return _T0_radial(r, alpha, sigma)


def T_alpha(r_vec, alpha, sigma):
    """Rank-1 ``T_α = ∂T^(0)/∂r_α``; returns shape (3,)."""
    r_vec = np.asarray(r_vec, dtype=float)
    r = float(np.linalg.norm(r_vec))
    T1 = float(_T1_radial(r, alpha, sigma))
    return (r_vec / r) * T1


def T_alphabeta(r_vec, alpha, sigma):
    """Symmetric ``T_{αβ}``; returns shape (3, 3)."""
    r_vec = np.asarray(r_vec, dtype=float)
    r = float(np.linalg.norm(r_vec))
    T1 = float(_T1_radial(r, alpha, sigma))
    T2 = float(_T2_radial(r, alpha, sigma))
    inv_r = 1.0 / r
    rhat = r_vec * inv_r
    delta = np.eye(3)
    rhat_outer = np.outer(rhat, rhat)
    return (delta - rhat_outer) * (T1 * inv_r) + rhat_outer * T2


def T_alphabetagamma(r_vec, alpha, sigma):
    """Symmetric rank-3 ``T_{αβγ}``; returns shape (3, 3, 3)."""
    r_vec = np.asarray(r_vec, dtype=float)
    r = float(np.linalg.norm(r_vec))
    T1 = float(_T1_radial(r, alpha, sigma))
    T2 = float(_T2_radial(r, alpha, sigma))
    T3 = float(_T3_radial(r, alpha, sigma))
    return _t_abg_from_radials(
        r_vec[None], np.array([r]), np.array([T1]), np.array([T2]), np.array([T3])
    )[0]


def T_alphabetagammadelta(r_vec, alpha, sigma):
    """Symmetric rank-4 ``T_{αβγδ}``; returns shape (3, 3, 3, 3)."""
    r_vec = np.asarray(r_vec, dtype=float)
    r = float(np.linalg.norm(r_vec))
    T1 = float(_T1_radial(r, alpha, sigma))
    T2 = float(_T2_radial(r, alpha, sigma))
    T3 = float(_T3_radial(r, alpha, sigma))
    T4 = float(_T4_radial(r, alpha, sigma))
    return _t_abgd_from_radials(
        r_vec[None],
        np.array([r]),
        np.array([T1]),
        np.array([T2]),
        np.array([T3]),
        np.array([T4]),
    )[0]


# ---------------------------------------------------------------------------
# Vectorized Cartesian-tensor builders (per-pair (M, 3, 3, 3) etc.).
# Derived from sympy at first call and cached.
# ---------------------------------------------------------------------------


def _t_abg_from_radials(r_vec, r_norm, T1, T2, T3):
    """Build ``T_{αβγ}`` per pair (M, 3, 3, 3) from radial derivatives.

    Closed form: differentiate ``T_{αβ} = A(r) δ_{αβ} + B(r) r̂_α r̂_β``
    where ``A(r) = T1/r`` and ``B(r) = T2 - T1/r``. Using

    .. math::

        \\partial_γ r̂_α = (δ_{αγ} - r̂_α r̂_γ) / r,

    we get

    .. math::

        T_{αβγ} = A'(r) δ_{αβ} r̂_γ + B'(r) r̂_α r̂_β r̂_γ
                + (B/r) (δ_{αγ} r̂_β + δ_{βγ} r̂_α - 2 r̂_α r̂_β r̂_γ),

    with ``A' = T2/r - T1/r²`` and ``B' = T3 - T2/r + T1/r²``.
    """
    rhat = r_vec / r_norm[:, None]  # (M, 3)
    inv_r = 1.0 / r_norm  # (M,)
    inv_r2 = inv_r * inv_r  # (M,)
    delta = np.eye(3)

    A_prime = T2 * inv_r - T1 * inv_r2
    B_val = T2 - T1 * inv_r
    B_prime = T3 - T2 * inv_r + T1 * inv_r2
    B_over_r = B_val * inv_r

    rhat_outer3 = np.einsum("ma,mb,mg->mabg", rhat, rhat, rhat)
    t1 = np.einsum("m,ab,mg->mabg", A_prime, delta, rhat)
    t2 = np.einsum("m,mabg->mabg", B_prime, rhat_outer3)
    t3 = (
        np.einsum("m,ag,mb->mabg", B_over_r, delta, rhat)
        + np.einsum("m,bg,ma->mabg", B_over_r, delta, rhat)
        - 2.0 * np.einsum("m,mabg->mabg", B_over_r, rhat_outer3)
    )
    return t1 + t2 + t3


def _t_abgd_from_radials(r_vec, r_norm, T1, T2, T3, T4):
    """Build ``T_{αβγδ}`` per pair (M, 3, 3, 3, 3) by direct sympy lambdify.

    The closed form is much messier than rank-3; rather than write it
    by hand, we lambdify the rank-4 expression at first call (cached)
    in terms of ``(rx, ry, rz, T1, T2, T3, T4)`` symbols.
    """
    fn = _t_abgd_lambda()
    rx, ry, rz = r_vec[:, 0], r_vec[:, 1], r_vec[:, 2]
    out = fn(rx, ry, rz, T1, T2, T3, T4)
    arr = np.asarray(out, dtype=float)
    if arr.ndim == 5:
        # sympy ImmutableDenseNDimArray lambdifies to shape (3,3,3,3,M).
        return np.moveaxis(arr, -1, 0)
    return arr


def _t_abgd_lambda():
    """Lambdify the rank-4 ``T_{αβγδ}`` in closed form.

    Strategy: differentiate ``T_{αβγ}`` (which we already have a clean
    closed form for) one more time w.r.t. ``r_δ`` using sympy's
    ``Function`` machinery indirectly. We use a free symbolic ``r``
    placeholder (not bound to ``√(rx²+ry²+rz²)``) and the radial
    derivatives ``T1..T4`` as input symbols; only the directional
    cosines depend on Cartesian indices.

    The closed form for ``T_{αβγ}`` (derived in
    ``_t_abg_from_radials``) is:

    .. math::

        T_{αβγ} = A'(r) δ_{αβ} \\hat r_γ + B'(r) \\hat r_α \\hat r_β \\hat r_γ
                + (B/r) (δ_{αγ} \\hat r_β + δ_{βγ} \\hat r_α - 2 \\hat r_α \\hat r_β \\hat r_γ)

    with ``A(r) = T1/r``, ``B(r) = T2 − T1/r``.

    Differentiating once more gives ``T_{αβγδ}``. We derive symbolically
    using ``∂_δ r̂_α = (δ_{αδ} − r̂_α r̂_δ)/r`` and a sympy expression
    in terms of placeholder radial scalars ``T1..T4``.
    """
    if hasattr(_t_abgd_lambda, "_cached"):
        return _t_abgd_lambda._cached

    import sympy as sp

    rx, ry, rz = sp.symbols("rx ry rz", real=True)
    T1s, T2s, T3s, T4s = sp.symbols("T1 T2 T3 T4", real=True)
    # Use a STANDALONE ``r`` symbol so sympy doesn't try to chain
    # through sqrt(). We substitute r → √(...) numerically inside
    # lambdified callable.
    r = sp.symbols("r", positive=True)
    rhat = [rx / r, ry / r, rz / r]

    def delta(a, b):
        return sp.Integer(1) if a == b else sp.Integer(0)

    # T1 = f'(r), T2 = f''(r), T3 = f'''(r), T4 = f''''(r).
    # A = T1/r,  B = T2 - T1/r
    # A'(r) = T2/r - T1/r²
    # B'(r) = T3 - T2/r + T1/r²
    # A''(r) = T3/r - 2 T2/r² + 2 T1/r³
    # B''(r) = T4 - T3/r + 2 T2/r² - 2 T1/r³
    Ap = T2s / r - T1s / r**2
    B = T2s - T1s / r
    Bp = T3s - T2s / r + T1s / r**2

    # T_{αβγ} = Ap·δ_{αβ}·r̂_γ + Bp·r̂_α·r̂_β·r̂_γ
    #        + (B/r)·(δ_{αγ}·r̂_β + δ_{βγ}·r̂_α − 2·r̂_α·r̂_β·r̂_γ)
    def T_abg(a, b, g):
        return (
            Ap * delta(a, b) * rhat[g]
            + Bp * rhat[a] * rhat[b] * rhat[g]
            + (B / r)
            * (
                delta(a, g) * rhat[b]
                + delta(b, g) * rhat[a]
                - 2 * rhat[a] * rhat[b] * rhat[g]
            )
        )

    # T_{αβγδ} = ∂_δ T_{αβγ}, treating r as r(rx, ry, rz) = √(rx²+ry²+rz²).
    # Use chain rule: ∂_δ f(r, rx, ry, rz) = (∂_δ r)·∂_r f + ∂_{rδ} f
    # where ∂_δ r = r̂_δ and ∂_{rδ} f means treating r̂ via
    # ``∂_δ r̂_α = (δ_{αδ} − r̂_α r̂_δ)/r``.
    #
    # Easier: define T_abg(a, b, g) in terms of (rx, ry, rz, r) where
    # r is treated as a free symbol and r̂ = (rx, ry, rz)/r. Differentiate
    # using sympy w.r.t. rx/ry/rz with substitution r = √(rx²+ry²+rz²)
    # AFTER differentiation.
    #
    # Implementation: differentiate symbolically with the chain rule.
    # Each rx component is also part of r = √(rx²+ry²+rz²). We make
    # r explicitly depend on (rx, ry, rz) for the differentiation.

    r_expr = sp.sqrt(rx**2 + ry**2 + rz**2)
    rvec = [rx, ry, rz]
    T = sp.MutableDenseNDimArray.zeros(3, 3, 3, 3)
    for a in range(3):
        for b in range(3):
            for g in range(3):
                # Substitute the free ``r`` symbol with sqrt(rx²+...).
                # We do this BEFORE differentiation so sympy can
                # propagate the chain rule.
                T_abg_concrete = T_abg(a, b, g).subs(r, r_expr)
                for d in range(3):
                    expr = sp.diff(T_abg_concrete, rvec[d])
                    # Substitute back to keep the expression in terms of
                    # (rx, ry, rz, T1..T4) — no need to put r back as
                    # the placeholder; lambdified function takes
                    # (rx, ry, rz, T1, T2, T3, T4) and computes r from
                    # those at call time IF needed (but sympy keeps
                    # rx, ry, rz inline).
                    T[a, b, g, d] = sp.simplify(expr)
    fn = sp.lambdify(
        (rx, ry, rz, T1s, T2s, T3s, T4s),
        sp.ImmutableDenseNDimArray(T),
        modules=["scipy", "numpy"],
    )
    _t_abgd_lambda._cached = fn
    return fn


# ---------------------------------------------------------------------------
# Pair-energy (single-pair, slow path)
# ---------------------------------------------------------------------------


def pair_energy(qi, mui, Qi, qj, muj, Qj, r_vec, alpha, sigma):
    """Real-space pair energy ``E_ij`` for two Cartesian multipoles.

    Returns the **raw** pair energy in atomic units (no ``F/(4π)``
    prefactor — the caller applies it). Multipole-expansion sign
    conventions:

    .. math::

        E_{ij} = q_i q_j T^{(0)}
               + q_j (\\boldsymbol\\mu_i \\cdot \\nabla T^{(0)})
               - q_i (\\boldsymbol\\mu_j \\cdot \\nabla T^{(0)})
               - \\mu_i^\\alpha \\mu_j^\\beta T_{\\alpha\\beta}
               + \\tfrac{1}{2}(q_j Q_i + q_i Q_j) : T_{\\alpha\\beta}
               + \\tfrac{1}{2}(Q_i^{\\beta\\gamma} \\mu_j^\\alpha
                              - \\mu_i^\\alpha Q_j^{\\beta\\gamma}) T_{\\alpha\\beta\\gamma}
               + \\tfrac{1}{4} Q_i^{\\alpha\\beta} Q_j^{\\gamma\\delta}
                 T_{\\alpha\\beta\\gamma\\delta}.
    """
    r_vec = np.asarray(r_vec, dtype=float)
    E = qi * qj * float(T0(r_vec[0], r_vec[1], r_vec[2], alpha, sigma))

    has_dipoles = mui is not None and muj is not None
    has_quad = Qi is not None and Qj is not None
    if not (has_dipoles or has_quad):
        return float(E)

    Ta = T_alpha(r_vec, alpha, sigma)
    Tab = T_alphabeta(r_vec, alpha, sigma)

    if has_dipoles:
        mui = np.asarray(mui, dtype=float)
        muj = np.asarray(muj, dtype=float)
        E += qj * float(np.dot(mui, Ta))
        E += -qi * float(np.dot(muj, Ta))
        E += -float(np.einsum("a,b,ab->", mui, muj, Tab))

    if has_quad:
        Qi = np.asarray(Qi, dtype=float)
        Qj = np.asarray(Qj, dtype=float)
        E += 0.5 * qj * float(np.einsum("ab,ab->", Qi, Tab))
        E += 0.5 * qi * float(np.einsum("ab,ab->", Qj, Tab))
        if has_dipoles:
            Tabg = T_alphabetagamma(r_vec, alpha, sigma)
            E += -0.5 * float(np.einsum("a,bg,abg->", mui, Qj, Tabg))
            E += +0.5 * float(np.einsum("bg,a,abg->", Qi, muj, Tabg))
        Tabgd = T_alphabetagammadelta(r_vec, alpha, sigma)
        E += 0.25 * float(np.einsum("ab,gd,abgd->", Qi, Qj, Tabgd))

    return float(E)


# ---------------------------------------------------------------------------
# Self-energy corrections
# ---------------------------------------------------------------------------


def self_energy(charges, dipoles=None, quadrupoles=None, *, alpha, sigma):
    r"""Per-atom self-energy corrections in RAW (pre-F/(4π)) units.

    Caller multiplies the full real + reciprocal − self sum by
    ``F/(4π)``. To match the Ewald path's F-baked self-energy formulas
    (``F·q²/(8π^(3/2)·σ_c)``, etc.) after the caller's F/(4π)
    multiplication, the raw self-energy must equal ``4π`` times those
    formulas:

    .. math::

        E_\text{self}^{(q)}_\text{raw}  &= \frac{q^2}{2 \sigma_c \sqrt\pi}, \\
        E_\text{self}^{(\mu)}_\text{raw} &= \frac{|\mu|^2}{12 \sigma_c^3 \sqrt\pi}, \\
        E_\text{self}^{(Q)}_\text{raw}  &= \frac{|Q|_F^2}{120 \sigma_c^5 \sqrt\pi}.

    These are the large-volume continuum limit of the reciprocal-sum
    self-image contribution (derived in
    ``tools/derive_multipole.py`` and cross-validated against the Ewald path's
    ``_multipole_ewald_self_energy_per_atom`` to 12 digits).

    Returns shape ``(N,)``.
    """
    charges = np.asarray(charges, dtype=float)
    sc = _sigma_c(alpha, sigma)
    sqpi = math.sqrt(math.pi)
    atom_self = charges**2 / (2.0 * sc * sqpi)
    if dipoles is not None:
        mu = np.asarray(dipoles, dtype=float)
        atom_self = atom_self + np.einsum("na,na->n", mu, mu) / (12.0 * sc**3 * sqpi)
    if quadrupoles is not None:
        Q = np.asarray(quadrupoles, dtype=float)
        atom_self = atom_self + np.einsum("nab,nab->n", Q, Q) / (120.0 * sc**5 * sqpi)
    return atom_self


# ---------------------------------------------------------------------------
# Direct Ewald (full real + reciprocal − self) reference
# ---------------------------------------------------------------------------


def direct_ewald_energy(
    positions,
    charges,
    *,
    dipoles=None,
    quadrupoles=None,
    cell,
    alpha,
    sigma,
    real_cutoff=None,
    kspace_cutoff=None,
):
    """Direct Ewald reference: real-space + reciprocal-space − self.

    Vectorized numpy implementation. Slow at large N but bit-correct.
    Used only by tests as ground truth.

    Returns total energy as a Python float in the same units as the
    production code (``F/(4π)`` Coulomb factor applied).
    """
    from nvalchemiops.torch.math import FIELD_CONSTANT

    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    N = positions.shape[0]
    charges = np.asarray(charges, dtype=float)
    mu = np.asarray(dipoles, dtype=float) if dipoles is not None else None
    Q = np.asarray(quadrupoles, dtype=float) if quadrupoles is not None else None

    if real_cutoff is None:
        real_cutoff = max(8.0, 4.0 / alpha)
    if kspace_cutoff is None:
        kspace_cutoff = max(6.0, 2.0 * alpha)

    # ---------------- Real-space pair sum (vectorized) ----------------
    box_lengths = np.linalg.norm(cell, axis=1)
    n_img = np.ceil(real_cutoff / box_lengths).astype(int) + 1
    nxs, nys, nzs = np.meshgrid(
        np.arange(-n_img[0], n_img[0] + 1),
        np.arange(-n_img[1], n_img[1] + 1),
        np.arange(-n_img[2], n_img[2] + 1),
        indexing="ij",
    )
    shifts_int = np.stack([nxs.ravel(), nys.ravel(), nzs.ravel()], axis=-1)
    shifts = shifts_int @ cell  # (N_img, 3)

    diff_ij = positions[None, :, :] - positions[:, None, :]  # (N, N, 3)
    r_vecs = shifts[:, None, None, :] + diff_ij[None, :, :, :]  # (N_img, N, N, 3)
    r_norms = np.linalg.norm(r_vecs, axis=-1)
    is_n0 = (
        (shifts_int[:, 0] == 0) & (shifts_int[:, 1] == 0) & (shifts_int[:, 2] == 0)
    )[:, None, None]
    i_idx, j_idx = np.indices((N, N))
    j_gt_i = (j_idx > i_idx)[None, :, :]
    keep = (r_norms > 1e-12) & (r_norms < real_cutoff) & (~is_n0 | j_gt_i)
    r_vecs_flat = r_vecs[keep]  # (M, 3)
    r_norms_flat = r_norms[keep]  # (M,)
    i_flat = np.broadcast_to(i_idx[None, :, :], r_norms.shape)[keep]
    j_flat = np.broadcast_to(j_idx[None, :, :], r_norms.shape)[keep]
    is_n0_flat = np.broadcast_to(is_n0, r_norms.shape)[keep]
    # On n=0 sheet: already enforced i<j → count once.
    # On n≠0 sheet: kept all (i, j) including self-images → /2 to avoid
    # double counting of (i,j,+n) vs (j,i,−n).
    pair_weight = np.where(is_n0_flat, 1.0, 0.5)  # (M,)

    a_val, b_val = _ab(alpha, sigma)
    inv_r = 1.0 / r_norms_flat
    erfc_a = erfc(a_val * r_norms_flat)
    erfc_b = erfc(b_val * r_norms_flat)
    T0_flat = (erfc_b - erfc_a) * inv_r
    qi = charges[i_flat]
    qj = charges[j_flat]
    E_real = float(np.sum(pair_weight * qi * qj * T0_flat))

    needs_grad_terms = (mu is not None) or (Q is not None)
    if needs_grad_terms:
        T1_flat = _T1_radial(r_norms_flat, alpha, sigma)
        T2_flat = _T2_radial(r_norms_flat, alpha, sigma)
        rhat = r_vecs_flat * inv_r[:, None]
        delta = np.eye(3)
        rhat_outer = np.einsum("ma,mb->mab", rhat, rhat)
        T_ab = (delta[None, :, :] - rhat_outer) * (T1_flat * inv_r)[
            :, None, None
        ] + rhat_outer * T2_flat[:, None, None]

        if mu is not None:
            mu_i = mu[i_flat]
            mu_j = mu[j_flat]
            T_a = rhat * T1_flat[:, None]
            E_real += float(np.sum(pair_weight * qj * np.einsum("ma,ma->m", mu_i, T_a)))
            E_real += float(
                np.sum(-pair_weight * qi * np.einsum("ma,ma->m", mu_j, T_a))
            )
            E_real += float(-np.einsum("m,ma,mb,mab->", pair_weight, mu_i, mu_j, T_ab))

        if Q is not None:
            Q_i = Q[i_flat]
            Q_j = Q[j_flat]
            E_real += float(
                0.5 * np.sum(pair_weight * qj * np.einsum("mab,mab->m", Q_i, T_ab))
            )
            E_real += float(
                0.5 * np.sum(pair_weight * qi * np.einsum("mab,mab->m", Q_j, T_ab))
            )
            T3_flat = _T3_radial(r_norms_flat, alpha, sigma)
            T4_flat = _T4_radial(r_norms_flat, alpha, sigma)
            T_abg = _t_abg_from_radials(
                r_vecs_flat, r_norms_flat, T1_flat, T2_flat, T3_flat
            )
            T_abgd = _t_abgd_from_radials(
                r_vecs_flat,
                r_norms_flat,
                T1_flat,
                T2_flat,
                T3_flat,
                T4_flat,
            )
            if mu is not None:
                E_real += float(
                    -0.5 * np.einsum("m,ma,mbg,mabg->", pair_weight, mu_i, Q_j, T_abg)
                )
                E_real += float(
                    +0.5 * np.einsum("m,mbg,ma,mabg->", pair_weight, Q_i, mu_j, T_abg)
                )
            E_real += float(
                0.25 * np.einsum("m,mab,mgd,mabgd->", pair_weight, Q_i, Q_j, T_abgd)
            )

    # ---------------- Reciprocal sum ----------------
    # ``kspace_cutoff`` is a PHYSICAL |k| bound in inverse length. We
    # iterate over the integer reciprocal-lattice indices needed to
    # cover that physical sphere (bounded per-axis by ⌈kspace_cutoff /
    # |G_axis|⌉ + 1).
    V = abs(np.linalg.det(cell))
    G = 2.0 * np.pi * np.linalg.inv(cell).T  # (3, 3); rows are reciprocal basis
    g_lengths = np.linalg.norm(G, axis=1)
    n_kmax = np.ceil(kspace_cutoff / g_lengths).astype(int) + 1
    E_recip = 0.0
    for kx in range(-n_kmax[0], n_kmax[0] + 1):
        for ky in range(-n_kmax[1], n_kmax[1] + 1):
            for kz in range(-n_kmax[2], n_kmax[2] + 1):
                if kx == 0 and ky == 0 and kz == 0:
                    continue
                k = kx * G[0] + ky * G[1] + kz * G[2]
                k_norm = float(np.linalg.norm(k))
                if k_norm > kspace_cutoff:
                    continue
                k2 = k_norm**2
                phase = positions @ k
                e_ikr = np.exp(-1j * phase)
                rho_k = (charges * e_ikr).sum(dtype=complex)
                if mu is not None:
                    rho_k -= 1j * ((mu @ k) * e_ikr).sum(dtype=complex)
                if Q is not None:
                    kQk = np.einsum("a,nab,b->n", k, Q, k)
                    rho_k -= 0.5 * (kQk * e_ikr).sum(dtype=complex)
                gauss = math.exp(-k2 * (0.25 / alpha**2 + sigma**2))
                E_recip += (
                    (4.0 * math.pi / k2) * gauss * (rho_k * rho_k.conjugate()).real
                )
    E_recip *= 0.5 / V

    # ---------------- Self-energy subtraction ----------------
    E_self = float(
        self_energy(
            charges,
            dipoles=mu,
            quadrupoles=Q,
            alpha=alpha,
            sigma=sigma,
        ).sum()
    )

    return (FIELD_CONSTANT / (4.0 * math.pi)) * (E_real + E_recip - E_self)


def direct_ewald_reciprocal_minus_self(
    positions,
    charges,
    *,
    dipoles=None,
    quadrupoles=None,
    cell,
    alpha,
    sigma,
    kspace_cutoff=None,
):
    """Reciprocal-only contribution + self-energy subtraction.

    Equivalent to ``multipole_pme_reciprocal_space``'s return value
    (which returns ``E_recip − E_self − E_bg``). Used as an exact
    reference for PME's reciprocal half at l_max ≤ 2 without needing
    an Ewald real-space implementation at l_max = 2.

    Returns total in F-units (``F/(4π)`` Coulomb factor applied).
    """
    from nvalchemiops.torch.math import FIELD_CONSTANT

    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    charges = np.asarray(charges, dtype=float)
    mu = np.asarray(dipoles, dtype=float) if dipoles is not None else None
    Q = np.asarray(quadrupoles, dtype=float) if quadrupoles is not None else None

    if kspace_cutoff is None:
        kspace_cutoff = max(6.0, 2.0 * alpha)

    V = abs(np.linalg.det(cell))
    G = 2.0 * np.pi * np.linalg.inv(cell).T
    g_lengths = np.linalg.norm(G, axis=1)
    n_kmax = np.ceil(kspace_cutoff / g_lengths).astype(int) + 1

    E_recip = 0.0
    for kx in range(-n_kmax[0], n_kmax[0] + 1):
        for ky in range(-n_kmax[1], n_kmax[1] + 1):
            for kz in range(-n_kmax[2], n_kmax[2] + 1):
                if kx == 0 and ky == 0 and kz == 0:
                    continue
                k = kx * G[0] + ky * G[1] + kz * G[2]
                k_norm = float(np.linalg.norm(k))
                if k_norm > kspace_cutoff:
                    continue
                k2 = k_norm**2
                phase = positions @ k
                e_ikr = np.exp(-1j * phase)
                rho_k = (charges * e_ikr).sum(dtype=complex)
                if mu is not None:
                    rho_k -= 1j * ((mu @ k) * e_ikr).sum(dtype=complex)
                if Q is not None:
                    kQk = np.einsum("a,nab,b->n", k, Q, k)
                    rho_k -= 0.5 * (kQk * e_ikr).sum(dtype=complex)
                gauss = math.exp(-k2 * (0.25 / alpha**2 + sigma**2))
                E_recip += (
                    (4.0 * math.pi / k2) * gauss * (rho_k * rho_k.conjugate()).real
                )
    E_recip *= 0.5 / V

    E_self = float(
        self_energy(
            charges,
            dipoles=mu,
            quadrupoles=Q,
            alpha=alpha,
            sigma=sigma,
        ).sum()
    )

    return (FIELD_CONSTANT / (4.0 * math.pi)) * (E_recip - E_self)


__all__ = [
    "T0",
    "T_alpha",
    "T_alphabeta",
    "T_alphabetagamma",
    "T_alphabetagammadelta",
    "pair_energy",
    "self_energy",
    "direct_ewald_energy",
    "direct_ewald_reciprocal_minus_self",
]
