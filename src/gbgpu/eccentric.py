"""Eccentric Galactic binary waveforms (multi-harmonic model of Moore et al. 2024).

This module implements the eccentric quasi-monochromatic binary model of
`Moore et al., MNRAS 531, 2817 (2024), arXiv:2310.06568
<https://arxiv.org/abs/2310.06568>`_ on top of the FastGB waveform of
:class:`gbgpu.gbgpu.GBGPUBase`.

Model (Moore et al. eqs. 1-4)::

    h_+ = -(1 + cos^2 i) sum_n A_n cos[2 pi f_n t + phi + (n - 2) Phi]
    h_x =   2 cos i      sum_n A_n sin[2 pi f_n t + phi + (n - 2) Phi]

    f_n = n/2 (f_GW + fdot_GW t / 2),   Phi = Phi0 + 2 pi f_PP t,
    A_n = A (S_n(e) + C_n(e)) / (2 (1 - e^2)) = A alpha_n(e).

Every harmonic is therefore a circular FastGB source with (eqs. 22-23)::

    amp_n   = A alpha_n(e)
    f0_n    = n/2 f_GW + (n - 2) f_PP
    fdot_n  = n/2 fdot_GW          (fddot_n = n/2 fddot_GW)
    phi0_n  = phi + (n - 2) Phi0

and the same inclination, polarization and sky position. The harmonics are
narrow-band and widely separated, so each one is evaluated as its own FastGB
row, which is exactly the multi-harmonic heterodyning scheme of the paper.

The harmonic amplitudes have the closed form
``alpha_n(e) = (n/4) [J_{n-2} - 2e J_{n-1} + (2/n) J_n + 2e J_{n+1} - J_{n+2}
+ sqrt(1-e^2) (J_{n-2} - 2 J_n + J_{n+2})]`` with ``J_k = J_k(n e)``
(dominant ``n l + 2 gamma`` branch of the Peters & Mathews decomposition).
By default they are evaluated, as in the paper, from their Taylor expansion
to tenth order in ``e`` (exact rational coefficients generated at import);
``ecc_order=None`` switches to an exact evaluation from the FFT of the
Keplerian orbit, valid up to ``e ~ 0.8``.

Design (shared-memory C++/CUDA backend)
---------------------------------------
* A batch of ``B`` eccentric binaries is mapped on the device to ``B * H``
  circular FastGB rows (``H`` harmonics) and handed to the existing
  ``SharedMemory*`` kernels (waveforms, likelihood, swap likelihood, global
  templates), so the heavy lifting stays in the GPU kernels and every
  backend option (TDI setup, TDI2, windows, multi-GPU splits, per-row ``N``)
  is inherited unchanged.
* The kernels divide by the amplitude, so every row is evaluated with unit
  amplitude and the outputs are rescaled by ``amp_n`` (``<d|h>`` by
  ``amp_n``, ``<h|h>`` by ``amp_n^2``, global-template ``factors`` by
  ``amp_n``). Harmonics with zero amplitude (``e = 0``) are then exact zeros.
* Each harmonic gets its own number of samples ``N`` (the kernels group rows
  by ``N``), so high, Doppler-broadened harmonics do not inflate ``N`` of the
  others.
* Inner products between different harmonics vanish unless their frequency
  bands overlap (``f_GW T / 2 <~ N``, i.e. very low frequencies). Those cross
  terms are detected from the band indices (after an ``O(num_binaries)``
  bound that rules out overlaps for almost all sources) and evaluated exactly
  with the swap kernel, whose ``add_remove`` output is ``<h_remove|h_add>``.
  A binary with overlapping harmonic bands uses its largest ``N`` for all of
  its harmonics, so cross terms, likelihood terms and global templates are
  built from the same waveforms.

Conventions
-----------
* The overall sign of eq. (1) is absorbed in ``phi`` so that ``e = 0``
  reproduces :class:`gbgpu.gbgpu.GBGPU` exactly. ``flip_ref_phase`` flips the
  phase of every harmonic, ``phi0_n -> -phi0_n``, as for circular sources.
* ``fPP`` enters as in eq. (1)/(22): harmonic ``n`` is shifted by
  ``+(n - 2) fPP``. For a prograde (GR-like) advance of the periastron the
  physical harmonic frequencies are ``n/2 f_GW - (n - 2) f_PP^GR``
  (the harmonic spacing is the radial frequency), i.e. such a binary is
  described by ``fPP = -f_PP^GR`` with ``f_PP^GR > 0`` from eq. (12)
  (:func:`get_fPP`). The waveform family is identical under ``fPP -> -fPP``;
  the sign only matters when mapping ``fPP`` to the total mass.
* Negative eccentricities are accepted as the analytic continuation,
  ``h(-e, Phi0) = h(e, Phi0 + pi)``, which keeps ``(e, Phi0)`` smooth at ``e = 0``.

Beyond the paper: the full Newtonian quadrupole
-----------------------------------------------
Eq. (1) keeps the ``n l + 2 omega`` branch of the quadrupole, i.e. the harmonic
expansion of the rotating part ``(x + i y)^2`` of the orbit. The full quadrupole
has two more terms, which ``GBGPUEccentric(breathing=True, counter_rotating=True)``
adds as extra circular rows with no new parameters
(:func:`eccentric_correction_amplitudes`):

* the **breathing** term ``h_+ = A sin^2(iota) sum_n J_n(ne) cos(n l)`` from the
  size ``x^2 + y^2 = r^2`` of the orbit: linearly polarized (a FastGB row at
  ``iota = pi/2``), absent face-on, at ``n/2 f + n fPP``. It is as large as
  ``alpha_1`` on ``n = 1`` and becomes relevant on all harmonics for ``e >~ 0.3``;
* the **counter-rotating** branch ``n l - 2 omega``, ``beta_n = O(e^{n+2})``, a row at
  ``iota -> pi - iota``, ``n/2 f + (n+2) fPP``.

Both are off by default. See ``gb_ecc/breathing_note`` for the derivation and the
comparison with the time-domain quadrupole of eGB-multi.

"""

import warnings
from fractions import Fraction
from functools import lru_cache
from math import factorial
from typing import Dict, Optional

import numpy as np

from lisatools.utils.constants import C_SI, G_SI, MSUN_SI, YRSID_SI

from .gbgpu import GBGPUBase
from .utils.utility import get_N


__all__ = [
    "GBGPUEccentric",
    "eccentric_amplitude_coefficients",
    "eccentric_correction_amplitudes",
    "eccentric_harmonic_amplitudes",
    "eccentric_to_circular_harmonics",
    "eccentricity_enhancement",
    "get_fdot_eccentric",
    "get_fPP",
    "get_masses_from_eccentric_parameters",
]

#: number of parameters of an eccentric binary (rows of ``params``)
NUM_ECC_PARAMS = 12

#: sample numbers supported by the shared-memory kernels
_KERNEL_N_MIN = 32
_KERNEL_N_MAX = 2048

_tdi_nchannels = {"AE": 2, "AET": 3, "XYZ": 3}


def _to_host(arr):
    """Return a NumPy view/copy of a NumPy or CuPy array."""
    try:
        return arr.get()
    except AttributeError:
        return np.asarray(arr)


def _get_xp(*arrays):
    """``cupy`` if any input lives on the GPU, ``numpy`` otherwise."""
    for arr in arrays:
        if hasattr(arr, "__cuda_array_interface__"):
            import cupy as cp

            return cp
    return np


@lru_cache(maxsize=None)
def _series_validity_limit(harmonics, order, tol=1e-2):
    """Largest ``e`` for which the order-``order`` series of every harmonic is
    within ``tol`` (absolute, in units of ``A``) of the exact amplitudes."""
    grid = np.linspace(0.0, 0.95, 191)
    exact = _alpha_kepler_fft(grid, np.asarray(harmonics), np, num_points=1024)
    coeffs = _coefficient_table(harmonics, order)
    series = np.polynomial.polynomial.polyval(grid, coeffs.T).T
    bad = np.flatnonzero(np.abs(series - exact).max(axis=1) > tol)
    return float(grid[bad[0] - 1]) if bad.size else float(grid[-1])


# =============================================================================
# Harmonic amplitudes alpha_n(e) = A_n / A
# =============================================================================


def _bessel_taylor(k, n, order):
    """Exact Taylor coefficients in ``e`` of ``J_k(n e)`` up to ``e**order``."""
    sign = 1
    if k < 0:
        k = -k
        sign = -1 if k % 2 else 1
    coeffs = [Fraction(0)] * (order + 1)
    half_n = Fraction(n, 2)
    m = 0
    while 2 * m + k <= order:
        coeffs[2 * m + k] = (
            sign * (-1) ** m * half_n ** (2 * m + k) / (factorial(m) * factorial(m + k))
        )
        m += 1
    return coeffs


def _taylor_mul(a, b, order):
    out = [Fraction(0)] * (order + 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j in range(order + 1 - i):
            if b[j] != 0:
                out[i + j] += ai * b[j]
    return out


@lru_cache(maxsize=None)
def _alpha_taylor_rational(n, order):
    """Rational Taylor coefficients of ``alpha_n(e)`` up to ``e**order``."""
    if n < 1:
        raise ValueError("Harmonic numbers must be positive integers.")
    e_series = [Fraction(0)] * (order + 1)
    if order >= 1:
        e_series[1] = Fraction(1)

    Jm2, Jm1, J0, Jp1, Jp2 = (_bessel_taylor(n + d, n, order) for d in (-2, -1, 0, 1, 2))

    # sqrt(1 - e^2) = sum_j binom(1/2, j) (-e^2)^j
    sqrt_1me2 = [Fraction(0)] * (order + 1)
    binom, j = Fraction(1), 0
    while 2 * j <= order:
        sqrt_1me2[2 * j] = binom * (-1) ** j
        binom = binom * (Fraction(1, 2) - j) / (j + 1)
        j += 1

    e_Jm1 = _taylor_mul(e_series, Jm1, order)
    e_Jp1 = _taylor_mul(e_series, Jp1, order)
    bracket_a = [
        Jm2[i] - 2 * e_Jm1[i] + Fraction(2, n) * J0[i] + 2 * e_Jp1[i] - Jp2[i]
        for i in range(order + 1)
    ]
    bracket_b = _taylor_mul(
        sqrt_1me2, [Jm2[i] - 2 * J0[i] + Jp2[i] for i in range(order + 1)], order
    )
    return tuple(Fraction(n, 4) * (bracket_a[i] + bracket_b[i]) for i in range(order + 1))


@lru_cache(maxsize=None)
def _coefficient_table(harmonics, order):
    table = np.array(
        [[float(c) for c in _alpha_taylor_rational(n, order)] for n in harmonics],
        dtype=np.float64,
    )
    table.setflags(write=False)
    return table


def eccentric_amplitude_coefficients(harmonics=12, order=10, exact=False):
    """Taylor coefficients of the harmonic amplitudes ``alpha_n(e) = A_n / A``.

    ``alpha_n(e) = sum_k C[i, k] e**k`` for ``n = harmonics[i]``. The
    coefficients are exact (computed with rational arithmetic from the Bessel
    series), e.g. ``alpha_1 = -3/4 e + 13/32 e^3 + ...``,
    ``alpha_2 = 1 - 5/2 e^2 + 23/16 e^4 + ...``, ``alpha_3 = 9/4 e - 171/32 e^3 + ...``.
    The expansion to ``order=10`` with ``n = 1..12`` is the one of Moore et al. (2024).

    Args:
        harmonics (int or sequence of int, optional): Harmonics ``n``. An integer
            means ``1..harmonics``. Default ``12``.
        order (int, optional): Expansion order in ``e``. Default ``10``.
        exact (bool, optional): If ``True``, return a list of lists of
            :class:`fractions.Fraction` instead of a float array.

    Returns:
        2D double np.ndarray: Shape ``(num_harmonics, order + 1)`` (or nested
        lists of Fractions if ``exact``).

    """
    harmonics = _parse_harmonics(harmonics)
    if exact:
        return [list(_alpha_taylor_rational(int(n), int(order))) for n in harmonics]
    return _coefficient_table(tuple(int(n) for n in harmonics), int(order)).copy()


def _alpha_kepler_fft(e, harmonics, xp, num_points=256, newton_iterations=8):
    """Exact ``alpha_n(e)`` from the FFT of the Keplerian orbit.

    ``alpha_n = n^2 F_n / 4`` where ``F_n`` are the Fourier coefficients in the
    mean anomaly of ``(r/a)^2 exp(2 i v) = (cos u - e + i sqrt(1-e^2) sin u)^2``.
    Accurate to ~1e-13 for ``e <= 0.8`` (``num_points=256``).
    """
    harmonics = np.asarray(harmonics)
    if harmonics.max() >= num_points // 2:
        raise ValueError("num_points must be larger than twice the maximum harmonic.")
    ecc = e[:, None]
    mean_anomaly = xp.arange(num_points, dtype=xp.float64) * (2.0 * np.pi / num_points)
    u = mean_anomaly[None, :] + 0.85 * ecc * xp.sign(xp.sin(mean_anomaly))[None, :]
    for _ in range(newton_iterations):
        u -= (u - ecc * xp.sin(u) - mean_anomaly[None, :]) / (1.0 - ecc * xp.cos(u))
    z = xp.cos(u) - ecc + 1j * xp.sqrt(1.0 - ecc * ecc) * xp.sin(u)
    coeffs = xp.fft.fft(z * z, axis=-1)[:, xp.asarray(harmonics)].real
    n2 = xp.asarray(harmonics.astype(np.float64) ** 2)
    return (0.25 / num_points) * n2[None, :] * coeffs


def _kepler_correction_fft(e, harmonics, xp, num_points=256, newton_iterations=8):
    """``J_n(n e)`` and ``beta_n(e)`` from the FFT of the Keplerian orbit.

    ``J_n(n e)`` is the Fourier coefficient of ``a / r = 1 / (1 - e cos u)``
    (``a/r = 1 + 2 sum_n J_n(ne) cos(n l)``); ``beta_n = n^2 F_{-n} / 4`` with
    ``F_k`` the coefficients of ``(r/a)^2 exp(2 i v)``, as ``alpha_n = n^2 F_n / 4``
    in :func:`_alpha_kepler_fft`. Negative ``e`` gives the analytic continuation.
    """
    harmonics = np.asarray(harmonics)
    if harmonics.max() >= num_points // 2:
        raise ValueError("num_points must be larger than twice the maximum harmonic.")
    ecc = e[:, None]
    mean_anomaly = xp.arange(num_points, dtype=xp.float64) * (2.0 * np.pi / num_points)
    u = mean_anomaly[None, :] + 0.85 * ecc * xp.sign(xp.sin(mean_anomaly))[None, :]
    for _ in range(newton_iterations):
        u -= (u - ecc * xp.sin(u) - mean_anomaly[None, :]) / (1.0 - ecc * xp.cos(u))
    n = xp.asarray(harmonics)
    a_over_r = 1.0 / (1.0 - ecc * xp.cos(u))
    bessel = xp.fft.fft(a_over_r, axis=-1)[:, n].real / num_points
    z = xp.cos(u) - ecc + 1j * xp.sqrt(1.0 - ecc * ecc) * xp.sin(u)
    n2 = xp.asarray(harmonics.astype(np.float64) ** 2)
    beta = (0.25 / num_points) * n2[None, :] * xp.fft.fft(z * z, axis=-1)[:, -n].real
    return bessel, beta


def eccentric_correction_amplitudes(e, harmonics=12, xp=None, num_points=256):
    """Amplitudes of the two terms of the Newtonian quadrupole the Moore et al. model leaves out.

    The full quadrupole of an eccentric orbit is (eGB-multi sign convention)::

        h_+ = -(1 + cos^2 i) A sum_n [alpha_n cos(n l + 2 w) + beta_n cos(n l - 2 w)]
              + A sin^2 i sum_n J_n(n e) cos(n l)
        h_x = -2 cos i A sum_n [alpha_n sin(n l + 2 w) - beta_n sin(n l - 2 w)]

    Moore et al. keep the ``alpha_n`` branch. The **breathing** term
    ``A sin^2 i J_n(ne) cos(n l)`` comes from the size ``r^2`` of the orbit
    (linearly polarized, absent face-on) and the **counter-rotating** branch
    ``beta_n = O(e^{n+2})`` from the negative-frequency coefficients of
    ``(x + i y)^2``. See ``gb_ecc/breathing_note``.

    Args:
        e (double or 1D double xp.ndarray): Eccentricities.
        harmonics (int or sequence of int, optional): Harmonics ``n``.
        xp (module, optional): ``numpy`` or ``cupy``. Default: inferred from ``e``.
        num_points (int, optional): Mean-anomaly samples of the Kepler FFT.

    Returns:
        tuple: ``(bessel, beta)``, each ``(num_bin, num_harmonics)``: ``J_n(n e)``
        (the breathing amplitude is ``A sin^2(iota) J_n(n e)``) and ``beta_n(e)``.

    """
    if xp is None:
        xp = _get_xp(e)
    e = xp.atleast_1d(xp.asarray(e, dtype=xp.float64))
    return _kepler_correction_fft(e, _parse_harmonics(harmonics), xp, num_points=num_points)


def _correction_rows(main_rows, params, harmonics, breathing, counter_rotating, xp, num_points=256):
    """Extra circular rows of the breathing and/or counter-rotating terms.

    Args:
        main_rows (3D xp.ndarray): ``(num_binaries, num_harmonics, 9)`` rows of
            the Moore harmonics (only columns 2, 3, 6, 7, 8 are reused).
        params (2D xp.ndarray): ``(num_binaries, 12)`` eccentric parameters.

    Returns:
        tuple: ``(rows, scale)`` of shapes ``(num_binaries, F, num_harmonics, 9)``
        and ``(num_binaries, F, num_harmonics)``, ``F`` the number of enabled
        families, in the order breathing, counter-rotating. Harmonic ``n`` is::

            breathing  A sin^2(iota) J_n(ne)  f = n/2 f0 + n fPP       phi = n Phi0 + pi        iota = pi/2
            counter    A beta_n(e)            f = n/2 f0 + (n+2) fPP   phi = -phi0 + (n+2) Phi0 iota = pi - iota

    """
    num_binaries, num_harm = main_rows.shape[:2]
    families = []
    if breathing:
        families.append("breathing")
    if counter_rotating:
        families.append("counter")
    rows = xp.repeat(main_rows[:, None], len(families), axis=1)
    scale = xp.empty((num_binaries, len(families), num_harm), dtype=xp.float64)
    if not families:
        return rows, scale
    n = xp.asarray(np.asarray(harmonics, dtype=np.float64))[None, :]
    f0, fpp, phi0, iota, Phi0 = (params[:, k:k + 1] for k in (1, 11, 4, 5, 10))
    bessel, beta = _kepler_correction_fft(params[:, 9], harmonics, xp, num_points=num_points)
    for k, family in enumerate(families):
        if family == "breathing":
            rows[:, k, :, 1] = 0.5 * n * f0 + n * fpp
            rows[:, k, :, 4] = n * Phi0 + np.pi
            rows[:, k, :, 5] = 0.5 * np.pi
            scale[:, k] = params[:, 0:1] * xp.sin(iota) ** 2 * bessel
        else:
            rows[:, k, :, 1] = 0.5 * n * f0 + (n + 2.0) * fpp
            rows[:, k, :, 4] = -phi0 + (n + 2.0) * Phi0
            rows[:, k, :, 5] = np.pi - iota
            scale[:, k] = params[:, 0:1] * beta
    rows[:, :, :, 0] = 1.0
    return rows, scale


def eccentric_harmonic_amplitudes(
    e, harmonics=12, order=10, xp=None, num_points=256, coefficients=None
):
    """Relative harmonic amplitudes ``alpha_n(e) = A_n / A`` (Moore et al. eq. 4).

    Args:
        e (double or 1D double xp.ndarray): Eccentricities ``0 <= e < 1``.
        harmonics (int or sequence of int, optional): Harmonics ``n``. An integer
            means ``1..harmonics``. Default ``12``.
        order (int or None, optional): Taylor order in ``e`` (paper: ``10``).
            If ``None``, the exact amplitudes are computed from the FFT of the
            Keplerian orbit. Default ``10``.
        xp (module, optional): ``numpy`` or ``cupy``. Default: inferred from ``e``.
        num_points (int, optional): Mean-anomaly samples for ``order=None``.
        coefficients (xp.ndarray, optional): Pre-computed (device) coefficient
            table from :func:`eccentric_amplitude_coefficients`. Internal use.

    Returns:
        2D double xp.ndarray: ``alpha`` with shape ``(num_bin, num_harmonics)``.

    """
    if xp is None:
        xp = _get_xp(e)
    e = xp.atleast_1d(xp.asarray(e, dtype=xp.float64))
    harmonics = _parse_harmonics(harmonics)

    if order is None:
        return _alpha_kepler_fft(e, harmonics, xp, num_points=num_points)

    if coefficients is None:
        coefficients = xp.asarray(eccentric_amplitude_coefficients(harmonics, order))

    # Horner evaluation, (num_bin, num_harm)
    alpha = xp.empty((e.shape[0], coefficients.shape[0]), dtype=xp.float64)
    alpha[:] = coefficients[None, :, -1]
    ecc = e[:, None]
    for k in range(coefficients.shape[1] - 2, -1, -1):
        alpha *= ecc
        alpha += coefficients[None, :, k]
    return alpha


def _parse_harmonics(harmonics):
    harmonics = _to_host(harmonics)
    if np.ndim(harmonics) == 0:
        n_max = int(harmonics)
        if n_max < 1:
            raise ValueError("Number of harmonics must be >= 1.")
        return np.arange(1, n_max + 1, dtype=np.int64)
    harmonics = np.unique(np.asarray(harmonics, dtype=np.int64))
    if harmonics.size == 0 or harmonics.min() < 1:
        raise ValueError("Harmonic numbers must be positive integers.")
    return harmonics


# =============================================================================
# Physical relations (Moore et al. Sec. 2)
# =============================================================================


def eccentricity_enhancement(e):
    """Peters & Mathews enhancement ``F(e)`` of the chirp rate (Moore et al. eq. 11)."""
    e2 = _get_xp(e).asarray(e) ** 2
    return (1.0 + 73.0 / 24.0 * e2 + 37.0 / 96.0 * e2 * e2) / (1.0 - e2) ** 3.5


def get_fdot_eccentric(f_GW, e, m1=None, m2=None, Mc=None):
    """GW-driven chirp rate of an eccentric binary (Moore et al. eq. 9).

    Args:
        f_GW (double or np.ndarray): Frequency of the ``n = 2`` harmonic in Hz.
        e (double or np.ndarray): Eccentricity.
        m1, m2 (double or np.ndarray, optional): Component masses in solar masses.
        Mc (double or np.ndarray, optional): Chirp mass in solar masses.

    Returns:
        double or np.ndarray: ``fdot_GW`` in Hz/s.

    """
    xp = _get_xp(f_GW, e, m1, m2, Mc)
    if Mc is None:
        if m1 is None or m2 is None:
            raise ValueError("Provide either m1 and m2 or Mc.")
        Mc = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
    Mc_s = G_SI * xp.asarray(Mc) * MSUN_SI / C_SI**3
    return (
        96.0
        / 5.0
        * np.pi ** (8.0 / 3.0)
        * Mc_s ** (5.0 / 3.0)
        * xp.asarray(f_GW) ** (11.0 / 3.0)
        * eccentricity_enhancement(xp.asarray(e))
    )


def get_fPP(f_GW, e, m1=None, m2=None, M=None):
    """GR periastron precession frequency (Moore et al. eq. 12), in Hz.

    Returns the positive (prograde) value of eq. (12). With the harmonic
    convention of eq. (1) used by :class:`GBGPUEccentric`, a binary whose
    periastron advances because of GR is described by ``fPP = -get_fPP(...)``
    (see module docstring).

    Args:
        f_GW (double or np.ndarray): Frequency of the ``n = 2`` harmonic in Hz.
        e (double or np.ndarray): Eccentricity.
        m1, m2 (double or np.ndarray, optional): Component masses in solar masses.
        M (double or np.ndarray, optional): Total mass in solar masses.

    """
    xp = _get_xp(f_GW, e, m1, m2, M)
    if M is None:
        if m1 is None or m2 is None:
            raise ValueError("Provide either m1 and m2 or M.")
        M = m1 + m2
    M_s = G_SI * xp.asarray(M) * MSUN_SI / C_SI**3
    return (
        3.0
        / (2.0 * np.pi)
        * M_s ** (2.0 / 3.0)
        * (np.pi * xp.asarray(f_GW)) ** (5.0 / 3.0)
        / (1.0 - xp.asarray(e) ** 2)
    )


def get_masses_from_eccentric_parameters(f_GW, fdot, fPP, e):
    """Invert eqs. (9) and (12): ``(f_GW, fdot, f_PP, e) -> (m1, m2, Mc, M)``.

    ``fPP`` is the GR value of eq. (12) (positive for a prograde advance; pass
    ``-fPP`` if it comes from the eq. (1) convention of :class:`GBGPUEccentric`).
    Non-physical combinations (``fdot <= 0``, ``fPP <= 0`` or symmetric mass
    ratio above 1/4) return ``nan`` component masses, as in the post-processing
    of Moore et al. Sec. 4.2.

    Returns:
        tuple: ``(m1, m2, Mc, M)`` in solar masses, ``m1 >= m2``.

    """
    xp = _get_xp(f_GW, fdot, fPP, e)
    f_GW = xp.asarray(f_GW, dtype=float)
    fdot = xp.asarray(fdot, dtype=float)
    fPP = xp.asarray(fPP, dtype=float)
    e = xp.asarray(e, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        Mc_s = (
            fdot
            / (96.0 / 5.0 * np.pi ** (8.0 / 3.0) * f_GW ** (11.0 / 3.0) * eccentricity_enhancement(e))
        ) ** (3.0 / 5.0)
        M_s = (fPP * (1.0 - e**2) * 2.0 * np.pi / (3.0 * (np.pi * f_GW) ** (5.0 / 3.0))) ** 1.5
        Mc = Mc_s * C_SI**3 / G_SI / MSUN_SI
        M = M_s * C_SI**3 / G_SI / MSUN_SI
        eta = (Mc / M) ** (5.0 / 3.0)
        root = xp.sqrt(1.0 - 4.0 * eta)
        m1 = 0.5 * M * (1.0 + root)
        m2 = 0.5 * M * (1.0 - root)
    return m1, m2, Mc, M


def eccentric_to_circular_harmonics(
    amp,
    f0,
    fdot,
    fddot,
    phi0,
    iota,
    psi,
    lam,
    beta,
    e,
    Phi0,
    fPP,
    harmonics=12,
    ecc_order=10,
    xp=None,
):
    """Circular FastGB parameters of every harmonic (Moore et al. eqs. 4, 22, 23).

    Useful to inject/recover individual harmonics with :class:`gbgpu.gbgpu.GBGPU`
    or to reclassify circular candidates as harmonics of an eccentric source.

    Returns:
        3D double xp.ndarray: Shape ``(9, num_bin, num_harmonics)`` with rows
        ``(amp_n, f0_n, fdot_n, fddot_n, phi0_n, iota, psi, lam, beta)``.

    """
    all_params = (amp, f0, fdot, fddot, phi0, iota, psi, lam, beta, e, Phi0, fPP)
    if xp is None:
        xp = _get_xp(*all_params)
    params = xp.broadcast_arrays(
        *(xp.atleast_1d(xp.asarray(p, dtype=xp.float64)) for p in all_params)
    )
    amp, f0, fdot, fddot, phi0, iota, psi, lam, beta, e, Phi0, fPP = params
    harmonics = _parse_harmonics(harmonics)
    n = xp.asarray(harmonics, dtype=xp.float64)[None, :]
    alpha = eccentric_harmonic_amplitudes(e, harmonics, order=ecc_order, xp=xp)
    shape = (amp.shape[0], harmonics.size)
    out = xp.empty((9,) + shape, dtype=xp.float64)
    out[0] = amp[:, None] * alpha
    out[1] = 0.5 * n * f0[:, None] + (n - 2.0) * fPP[:, None]
    out[2] = 0.5 * n * fdot[:, None]
    out[3] = 0.5 * n * fddot[:, None]
    out[4] = phi0[:, None] + (n - 2.0) * Phi0[:, None]
    for i, p in enumerate((iota, psi, lam, beta)):
        out[5 + i] = p[:, None]
    return out




# =============================================================================
# Waveform class
# =============================================================================


class GBGPUEccentric(GBGPUBase):
    """Eccentric Galactic binaries following Moore et al. (2024).

    An eccentric binary has 12 parameters, in this order::

        amp, f0, fdot, fddot, phi0, iota, psi, lam, beta, e, Phi0, fPP

    ``amp`` is the overall amplitude ``A`` (eq. 14); ``f0``, ``fdot`` and
    ``fddot`` describe the dominant ``n = 2`` harmonic (``f_GW`` of eq. 2);
    ``phi0`` is ``phi``; ``e`` the eccentricity; ``Phi0`` the initial argument
    of periastron and ``fPP`` the periastron precession frequency (eq. 3).
    ``fddot`` is not in the paper's model (set it to zero to reproduce it).

    The methods mirror :class:`gbgpu.gbgpu.GBGPUBase` with 12 instead of 9
    parameters per row: :meth:`run_wave`, :meth:`get_ll`,
    :meth:`swap_likelihood_difference`, :meth:`generate_global_template` and
    :meth:`information_matrix` (Python path). Waveform outputs follow the
    base layout on the *flattened* harmonics, ``num_bin = num_binaries *
    num_harmonics`` rows (row ``b * num_harmonics + h`` is harmonic
    ``harmonics[h]`` of binary ``b``); harmonic-resolved views are ``AETf_harm``,
    ``A_harm``, ``E_harm``, ``start_inds_harm`` and ``freqs_harm``.

    Every harmonic band must lie inside the data band passed to the
    likelihoods and global templates (``n = 1`` sits near ``f_GW/2`` and the
    highest harmonic near ``n_max f_GW / 2``): the kernels cannot read outside
    it, so a ``ValueError`` is raised beforehand.

    Args:
        orbits (Orbits, optional): LISA orbits. Default equal-armlength orbits.
        force_backend (str, optional): ``"cpu"``, ``"gpu"``, ``"cuda12x"``, ...
        t0 (double, optional): Absolute start time for the spacecraft orbits
            (see :class:`gbgpu.gbgpu.GBGPUBase`).
        flip_ref_phase (bool, optional): Flip the reference phase of every
            harmonic (``phi0_n -> -phi0_n``), as for circular sources.
        harmonics (int or sequence of int, optional): Harmonics included. An
            integer means ``1..harmonics``. Default ``12`` (as in the paper).
            The cost scales linearly with the number of harmonics; for
            ``e <~ 0.1`` harmonics above ``n ~ 5`` carry negligible power.
        ecc_order (int or None, optional): Taylor order of the amplitudes in
            ``e`` (paper: ``10``). ``None`` uses the exact amplitudes. With 12
            harmonics the order-10 series is accurate to ``1e-2 A`` only for
            ``|e| <~ 0.4`` (``series_validity_limit``); a warning is issued above.
        exact_num_points (int, optional): Mean-anomaly samples of the exact
            amplitude evaluation. Default ``256``.
        N_amp_threshold (double, optional): When ``N`` is not given to
            :meth:`run_wave` (a single ``N`` for all rows), only harmonics with
            ``|alpha_n| >= N_amp_threshold * max_n |alpha_n|`` set it.
        breathing (bool, optional): Add the **breathing** term of the Newtonian
            quadrupole, ``h_+ = A sin^2(iota) sum_n J_n(ne) cos(n l)``, which the
            Moore et al. model leaves out. It comes from the size ``r^2`` of the
            orbit, is linearly polarized and absent face-on, and is as large as
            ``alpha_1`` on ``n = 1``. Default ``False`` (the published model).
        counter_rotating (bool, optional): Add the **counter-rotating** branch
            ``n l - 2 omega`` of the quadrupole, ``beta_n = O(e^{n+2})``. Default ``False``.

    With ``breathing`` and/or ``counter_rotating`` every harmonic is evaluated as
    one circular FastGB row per enabled family (no new parameters)::

        main       A alpha_n               f0_n = n/2 f0 + (n-2) fPP   phi0_n = phi0 + (n-2) Phi0    iota
        breathing  A sin^2(iota) J_n(ne)   f0_n = n/2 f0 +  n    fPP   phi0_n = n Phi0 + pi          pi/2
        counter    A beta_n(e)             f0_n = n/2 f0 + (n+2) fPP   phi0_n = -phi0 + (n+2) Phi0   pi - iota

    The rows of a binary are ordered family by family (all main harmonics, then
    all breathing, then all counter-rotating): ``num_components = F *
    num_harmonics`` rows per binary, and the ``*_harm`` views and per-harmonic
    outputs (``d_h_harm``, ``h_h_harm``, ``harmonic_frequencies``) run over these
    components (``component_harmonics``, ``component_families``). The rows of the
    same harmonic are ``2 |fPP|`` apart, so their bands always overlap and their
    cross terms are always included. With both families the model reproduces the
    time-domain quadrupole of eGB-multi to a mismatch ``<~ 1e-6`` for ``e <= 0.6``.
    The swap kernel used for the cross terms supports ``N <= 1024``.

    """

    def __init__(
        self,
        orbits=None,
        force_backend=None,
        t0=None,
        flip_ref_phase=False,
        harmonics=12,
        ecc_order=10,
        exact_num_points=256,
        N_amp_threshold=1e-3,
        breathing=False,
        counter_rotating=False,
    ):
        super().__init__(
            orbits=orbits, force_backend=force_backend, t0=t0, flip_ref_phase=flip_ref_phase
        )
        self.exact_num_points = int(exact_num_points)
        self.N_amp_threshold = float(N_amp_threshold)
        self._ecc_order = None if ecc_order is None else int(ecc_order)
        self._breathing = bool(breathing)
        self._counter_rotating = bool(counter_rotating)
        self.num_binaries = 0
        self.harmonics = harmonics

    # ------------------------------------------------------------------ setup
    @property
    def harmonics(self):
        """Harmonic numbers included in the waveform (1D int np.ndarray)."""
        return self._harmonics

    @harmonics.setter
    def harmonics(self, harmonics):
        self._harmonics = _parse_harmonics(harmonics)
        self._update_coefficients()

    @property
    def ecc_order(self):
        """Taylor order of the harmonic amplitudes (``None``: exact)."""
        return self._ecc_order

    @ecc_order.setter
    def ecc_order(self, ecc_order):
        self._ecc_order = None if ecc_order is None else int(ecc_order)
        self._update_coefficients()

    @property
    def num_harmonics(self):
        """Number of harmonics per binary."""
        return len(self._harmonics)

    @property
    def breathing(self):
        """Whether the breathing term is included."""
        return self._breathing

    @breathing.setter
    def breathing(self, value):
        self._breathing = bool(value)
        self._update_coefficients()

    @property
    def counter_rotating(self):
        """Whether the counter-rotating branch is included."""
        return self._counter_rotating

    @counter_rotating.setter
    def counter_rotating(self, value):
        self._counter_rotating = bool(value)
        self._update_coefficients()

    @property
    def families(self):
        """Row families per harmonic, in row order (``"main"`` first)."""
        return ("main",) + (("breathing",) if self._breathing else ()) + (
            ("counter",) if self._counter_rotating else ())

    @property
    def num_components(self):
        """Circular rows per binary: ``len(families) * num_harmonics``."""
        return len(self.families) * len(self._harmonics)

    @property
    def component_harmonics(self):
        """Harmonic number of every row of a binary (1D int np.ndarray)."""
        return np.tile(self._harmonics, len(self.families))

    @property
    def component_families(self):
        """Family of every row of a binary (1D np.ndarray of str)."""
        return np.repeat(np.array(self.families), len(self._harmonics))

    def _update_coefficients(self):
        if not hasattr(self, "_harmonics"):
            return
        xp = self.xp
        self._n_dev = xp.asarray(self._harmonics, dtype=xp.float64)
        # row pairs of one binary: different harmonics (overlap only at very low
        # frequency, pre-filtered by a bound) and same harmonic in different
        # families (always overlapping, 2 |fPP| apart)
        comp_n = self.component_harmonics
        num_comp = comp_n.size
        iu, ju = np.triu_indices(num_comp, k=1)
        off_i, off_j = np.nonzero(~np.eye(num_comp, dtype=bool))
        same_u, same_o = comp_n[iu] == comp_n[ju], comp_n[off_i] == comp_n[off_j]
        self._pairs_upper = (xp.asarray(iu[~same_u]), xp.asarray(ju[~same_u]))
        self._pairs_offdiag = (xp.asarray(off_i[~same_o]), xp.asarray(off_j[~same_o]))
        self._pairs_upper_same = (xp.asarray(iu[same_u]), xp.asarray(ju[same_u]))
        self._pairs_offdiag_same = (xp.asarray(off_i[same_o]), xp.asarray(off_j[same_o]))
        if self._ecc_order is None:
            self._coeffs = None
            #: largest |e| for which the amplitude series is accurate to 1e-2 A
            self.series_validity_limit = 1.0
            return
        vanishing = self._harmonics[self._harmonics - 2 > self._ecc_order]
        if len(vanishing) > 0:
            warnings.warn(
                f"Harmonics {vanishing.tolist()} vanish identically at expansion order "
                f"{self._ecc_order} in e; increase ecc_order or drop them."
            )
        self._coeffs = xp.asarray(eccentric_amplitude_coefficients(self._harmonics, self._ecc_order))
        self.series_validity_limit = _series_validity_limit(
            tuple(int(n) for n in self._harmonics), self._ecc_order
        )

    def harmonic_amplitudes(self, e):
        """``alpha_n(e) = A_n / A`` for the configured harmonics, ``(num_bin, num_harm)``."""
        e = self.xp.atleast_1d(self.xp.asarray(e, dtype=self.xp.float64))
        return eccentric_harmonic_amplitudes(
            e,
            self._harmonics,
            order=self._ecc_order,
            xp=self.xp,
            num_points=self.exact_num_points,
            coefficients=self._coeffs,
        )

    def repeat_per_harmonic(self, arr):
        """Repeat a per-binary array (e.g. ``group_index``) for every row of the binary."""
        return self.xp.repeat(self.xp.asarray(arr), self.num_components)

    # -------------------------------------------------------- base-class hooks
    def prepare_additional_args(self, *args):
        """Not used: eccentric parameters are mapped to harmonic rows up front."""
        return args

    def shift_frequency(self, fi, xi, *args):
        """Circular FastGB frequency evolution (each harmonic row is circular)."""
        return fi

    def add_to_phasing(self, arg_phase, f0, fdot, fddot, xi, *args):
        """Circular FastGB phasing (each harmonic row is circular)."""
        return arg_phase

    def special_get_N(self, amp, f0, T, e, Phi0, fPP, oversample=1):
        """Number of samples needed by the significant harmonics of each binary.

        Applies :func:`gbgpu.utils.utility.get_N` to every harmonic with
        ``|alpha_n| >= N_amp_threshold * max_n |alpha_n|`` and returns the
        maximum over harmonics.

        Returns:
            1D int np.ndarray: ``N`` for each binary.

        """
        amp, f0, e, fPP = np.broadcast_arrays(
            *(np.atleast_1d(_to_host(p)).astype(np.float64) for p in (amp, f0, e, fPP))
        )
        n = self._harmonics.astype(np.float64)[None, :]
        alpha = np.abs(
            eccentric_harmonic_amplitudes(
                e, self._harmonics, order=self._ecc_order, xp=np, num_points=self.exact_num_points
            )
        )
        f0_n = np.abs(0.5 * n * f0[:, None] + (n - 2.0) * fPP[:, None])
        alpha_max = alpha.max(axis=1, keepdims=True)
        significant = (alpha >= self.N_amp_threshold * alpha_max) | (alpha_max == 0.0)
        N_all = np.zeros(alpha.shape, dtype=np.int64)
        amp_n = np.maximum(np.abs(amp[:, None]) * alpha, 1e-300)  # avoid log(0) in get_N
        N_all[significant] = get_N(
            amp_n[significant], f0_n[significant], T, oversample=oversample,
            armlength=self.orbits.armlength,
        )
        return N_all.max(axis=1)

    # ------------------------------------------------------ harmonic mapping
    def _as_param_rows(self, params, name="params"):
        xp = self.xp
        params = xp.asarray(params, dtype=xp.float64)
        if params.ndim == 1:
            params = params[None, :]
        if params.ndim != 2 or params.shape[1] != NUM_ECC_PARAMS:
            raise ValueError(
                f"{name} must have shape (num_binaries, {NUM_ECC_PARAMS}) with columns "
                "(amp, f0, fdot, fddot, phi0, iota, psi, lam, beta, e, Phi0, fPP)."
            )
        return params

    def _check_eccentricity(self, e):
        if e.size == 0:
            return
        e_abs_max = float(self.xp.abs(e).max())
        if not e_abs_max < 1.0:  # also catches nan
            raise ValueError("Eccentricities must satisfy |e| < 1.")
        if e_abs_max > self.series_validity_limit:
            warnings.warn(
                f"Eccentricities above {self.series_validity_limit:.2f}: the O(e^{self._ecc_order}) "
                "series of the harmonic amplitudes is off by more than 1e-2 A there. "
                "Use ecc_order=None (exact amplitudes) or a higher order."
            )

    def _harmonic_rows(self, params):
        """Circular FastGB rows of all harmonics (and correction families) with unit amplitude.

        Returns:
            tuple: ``(rows, scale, alpha)`` with ``rows`` of shape
            ``(num_binaries * num_components, 9)`` (amplitude column set to 1),
            ``scale`` the row amplitudes (``amp * alpha_n`` for the main family)
            flattened, and ``alpha`` of shape ``(num_binaries, num_harmonics)``.

        """
        xp = self.xp
        num_binaries, num_harm = params.shape[0], self.num_harmonics
        self._check_eccentricity(params[:, 9])
        alpha = self.harmonic_amplitudes(params[:, 9])
        n = self._n_dev[None, :]
        rows = xp.empty((num_binaries, num_harm, 9), dtype=xp.float64)
        rows[:, :, 0] = 1.0
        rows[:, :, 1] = 0.5 * n * params[:, 1:2] + (n - 2.0) * params[:, 11:12]
        rows[:, :, 2] = 0.5 * n * params[:, 2:3]
        rows[:, :, 3] = 0.5 * n * params[:, 3:4]
        rows[:, :, 4] = params[:, 4:5] + (n - 2.0) * params[:, 10:11]
        rows[:, :, 5:9] = params[:, None, 5:9]
        scale = params[:, 0:1] * alpha
        if self._breathing or self._counter_rotating:
            extra_rows, extra_scale = _correction_rows(
                rows, params, self._harmonics, self._breathing, self._counter_rotating, xp,
                num_points=max(self.exact_num_points, 4 * int(self._harmonics.max()) + 4),
            )
            rows = xp.concatenate([rows[:, None], extra_rows], axis=1)
            scale = xp.concatenate([scale[:, None], extra_scale], axis=1)
        num_comp = self.num_components
        return rows.reshape(num_binaries * num_comp, 9), scale.reshape(-1), alpha

    def _row_N(self, N, f0_rows, scale, T, oversample, num_binaries, clip_to_kernel=True,
               check_kernel=True):
        """Number of samples of every harmonic row (int32 xp array).

        ``N`` is ``None`` (:func:`gbgpu.utils.utility.get_N` per harmonic, from
        ``|amp alpha_n|`` and ``f_n``), an int, or one value per binary or per
        harmonic row. With ``check_kernel`` the values are validated for the
        shared-memory kernels (on the host unless ``N`` is a device array).
        """
        xp = self.xp
        num_rows = f0_rows.shape[0]
        if N is None:
            N_h = np.asarray(
                get_N(
                    np.maximum(np.abs(_to_host(scale)), 1e-300),
                    np.abs(_to_host(f0_rows)),
                    T,
                    oversample=oversample,
                    armlength=self.orbits.armlength,
                ),
                dtype=np.int64,
            )
            if clip_to_kernel:
                N_h = np.clip(N_h, _KERNEL_N_MIN, _KERNEL_N_MAX)
            if check_kernel:
                self._check_kernel_N(np.unique(N_h))
            return xp.asarray(N_h.astype(np.int32))
        if np.ndim(N) == 0:
            if check_kernel:
                self._check_kernel_N([int(N)])
            return xp.full(num_rows, int(N), dtype=xp.int32)
        N_arr = xp.asarray(N).astype(xp.int32).reshape(-1)
        if N_arr.size == num_binaries:
            N_arr = xp.repeat(N_arr, self.num_harmonics)
        elif N_arr.size != num_rows:
            raise ValueError("N must be an int, or have one entry per binary or per harmonic row.")
        if check_kernel:
            self._check_kernel_N(_to_host(xp.unique(N_arr)))
        return N_arr

    @staticmethod
    def _check_kernel_N(values):
        """Validate (host) sample numbers for the shared-memory kernels."""
        bad = sorted(
            {int(v) for v in np.atleast_1d(values)
             if v < _KERNEL_N_MIN or v > _KERNEL_N_MAX or (int(v) & (int(v) - 1))}
        )
        if bad:
            raise ValueError(
                f"The shared-memory kernels support N = 32, 64, ..., 2048 (powers of two); got {bad}."
            )

    @staticmethod
    def _data_length_of(arrays, data_length):
        first = arrays[0] if isinstance(arrays, (list, tuple)) else arrays
        if first.ndim == 1:
            if data_length is None:
                raise ValueError("data_length is required when the arrays are flattened.")
            return int(data_length)
        return int(first.shape[-1])

    @staticmethod
    def _plain_scalars(start_freq_ind, data_length, T, dt):
        """Python scalars where the base class expects ``int``/``float``."""
        if isinstance(start_freq_ind, np.integer):
            start_freq_ind = int(start_freq_ind)
        if data_length is not None:
            data_length = int(data_length)
        return start_freq_ind, data_length, float(T), float(dt)

    def _band_starts(self, f0_rows, N_rows, T):
        xp = self.xp
        return xp.rint(f0_rows * T).astype(xp.int64) - (N_rows.astype(xp.int64) // 2)

    def _check_bands(self, f0_rows, N_rows, index_rows, start_freq_ind, arrays, data_length, T,
                     where, row_ids=None):
        """Raise if a harmonic band falls outside the data/template band.

        The kernels read ``start_ind - start_freq_ind + i`` for ``i < N``; a
        band outside ``[0, data_length)`` throws on CPU and is skipped with a
        printout on GPU, so it is rejected up front (single-buffer layouts).
        """
        xp = self.xp
        if isinstance(start_freq_ind, (int, np.integer)):
            sfi = int(start_freq_ind)
        elif isinstance(start_freq_ind, xp.ndarray) and isinstance(arrays, xp.ndarray):
            sfi = start_freq_ind.astype(xp.int64)[xp.asarray(index_rows)]
        else:  # per-device buffers: left to the kernels
            return
        dl = self._data_length_of(arrays, data_length)
        offset = self._band_starts(f0_rows, N_rows, T) - sfi
        bad = (offset < 0) | (offset + N_rows.astype(xp.int64) > dl)
        if bool(xp.any(bad)):
            row = int(_to_host(xp.argmax(bad)))
            source_row = row if row_ids is None else int(_to_host(row_ids[row]))
            binary, h = divmod(source_row, self.num_components)
            lo = int(_to_host(offset[row]))
            n_max = int(self._harmonics.max())
            family = "" if self.component_families[h] == "main" else f" ({self.component_families[h]} row)"
            raise ValueError(
                f"{where}: harmonic n={int(self.component_harmonics[h])}{family} of binary {binary} spans bins "
                f"[{lo}, {lo + int(_to_host(N_rows[row]))}) of a band with {dl} bins. "
                "Eccentric sources need data/templates covering every harmonic (n=1 near "
                f"f_GW/2 up to n={n_max} near {n_max}/2 f_GW), or fewer harmonics. Binaries "
                "whose harmonic bands overlap use their largest N for every harmonic."
            )

    def _overlapping_pairs(self, specs, N_rows, T, max_candidates=2**22):
        """Row pairs of each binary whose bands overlap.

        Args:
            specs (sequence): ``(params_a, params_b, f_rows_a, f_rows_b, pairs, pairs_same)``
                tuples; ``pairs = (i, j)`` pairs row ``i`` of ``a`` with row ``j``
                of ``b`` within the same binary for different harmonics, and
                ``pairs_same`` for the same harmonic in different families
                (empty without correction families).
            N_rows (xp.ndarray): Samples per row; a band covers
                ``rint(f T) - N // 2 + [0, N)``.

        A row of harmonic ``n`` is centred on bin ``n u + d fPP T`` with
        ``u = (f0 / 2 + fPP) T`` and ``d = -2`` (main), ``0`` (breathing) or
        ``+2`` (counter-rotating), so for different harmonics the centres are at
        least ``g |u_a| - n_max |u_a - u_b| - 2 T |fPP_a - fPP_b|`` apart (``g``
        the smallest gap between harmonic numbers; ``2 T (|fPP_a| + |fPP_b|)``
        with correction families), while overlapping bands are less than the
        largest ``N`` apart. Only binaries failing this ``O(num_binaries)`` bound
        (very low frequencies, or large jumps between ``a`` and ``b``) are
        expanded into ``O(num_harmonics^2)`` candidates, in chunks of at most
        ``max_candidates`` pairs. The common case (no candidates) costs one
        synchronization. Rows of the same harmonic in different families are
        ``2 |fPP|`` apart and always candidates.

        Returns:
            list: ``(ia, ib)`` int64 row indices for every spec.

        """
        xp = self.xp
        num_comp = self.num_components
        empty = xp.zeros(0, dtype=xp.int64)
        # 5-tuples (no same-harmonic pairs) are accepted as before
        specs = [tuple(sp) if len(sp) == 6 else tuple(sp) + ((empty, empty),) for sp in specs]
        if num_comp == 1:
            return [(empty, empty) for _ in specs]
        num_binaries = N_rows.shape[0] // num_comp
        N_long = N_rows.astype(xp.int64)
        hits = [([], []) for _ in specs]

        def expand(binaries, pairs, f_a, f_b, out):
            step = max(1, int(max_candidates) // int(pairs[0].size))
            for start in range(0, int(binaries.size), step):
                base = (binaries[start:start + step] * num_comp)[:, None]
                ia = (base + pairs[0][None, :]).reshape(-1)
                ib = (base + pairs[1][None, :]).reshape(-1)
                sa = xp.rint(f_a[ia] * T).astype(xp.int64) - N_long[ia] // 2
                sb = xp.rint(f_b[ib] * T).astype(xp.int64) - N_long[ib] // 2
                keep = xp.nonzero((sb < sa + N_long[ia]) & (sa < sb + N_long[ib]))[0]
                out[0].append(ia[keep])
                out[1].append(ib[keep])

        # different harmonics: pre-filtered by the bound
        if self.num_harmonics > 1:
            gap = float(np.diff(self._harmonics).min())
            n_max = float(self._harmonics.max())
            N_max = N_long.reshape(num_binaries, num_comp).max(axis=1).astype(xp.float64)
            corrections = self._breathing or self._counter_rotating
            masks = []
            for params_a, params_b, _, _, _, _ in specs:
                u_a = (0.5 * params_a[:, 1] + params_a[:, 11]) * T
                u_b = (0.5 * params_b[:, 1] + params_b[:, 11]) * T
                if corrections:
                    fpp_term = xp.abs(params_a[:, 11]) + xp.abs(params_b[:, 11])
                else:
                    fpp_term = xp.abs(params_a[:, 11] - params_b[:, 11])
                bound = (
                    gap * xp.abs(u_a)
                    - n_max * xp.abs(u_a - u_b)
                    - 2.0 * T * fpp_term
                    - 2.0  # rint of both centres
                )
                masks.append(bound < N_max)
            if bool(xp.any(xp.stack(masks))):
                for (_, _, f_a, f_b, pairs, _), mask, out in zip(specs, masks, hits):
                    expand(xp.nonzero(mask)[0].astype(xp.int64), pairs, f_a, f_b, out)

        # same harmonic, different families: always candidates
        for (_, _, f_a, f_b, _, pairs_same), out in zip(specs, hits):
            if int(pairs_same[0].size) > 0:
                expand(xp.arange(num_binaries, dtype=xp.int64), pairs_same, f_a, f_b, out)

        return [(xp.concatenate(a), xp.concatenate(b)) if a else (empty, empty) for a, b in hits]

    def _harmonic_pairs(self, specs, N_rows, T):
        """Overlapping harmonic pairs with a common ``N`` per affected binary.

        The kernels evaluate both harmonics of a cross term with one ``N``.
        To keep ``<h|h>``, the swap terms and the global templates consistent
        with each other, every harmonic of a binary with overlapping harmonic
        bands is given the largest ``N`` of that binary (for a swap, of its
        added and removed harmonics), and the pairs are searched again.

        Returns:
            tuple: ``(N_rows, found)`` with ``found`` as in :meth:`_overlapping_pairs`.

        """
        xp = self.xp
        num_comp = self.num_components
        found = self._overlapping_pairs(specs, N_rows, T)
        hit_rows = [ia for ia, _ in found if ia.size > 0]
        if not hit_rows:
            return N_rows, found
        binaries = xp.unique(xp.concatenate(hit_rows) // num_comp)
        N_bin = N_rows.reshape(-1, num_comp)
        N_common = N_bin[binaries].max(axis=1, keepdims=True)
        if bool(xp.all(N_bin[binaries] == N_common)):
            return N_rows, found
        N_bin = N_bin.copy()
        N_bin[binaries] = N_common
        N_rows = N_bin.reshape(-1)
        return N_rows, self._overlapping_pairs(specs, N_rows, T)

    def _pair_products(self, add_rows, remove_rows, add_ids, remove_ids, N_pairs, data_index,
                       noise_index, data_minus_template, psd, start_freq_ind, data_length, T, common):
        """``<h_remove|h_add>`` for pairs of circular rows via the swap kernel.

        Both rows of a pair are evaluated with ``N_pairs`` samples; the swap
        kernel returns ``add_remove = sum conj(h_remove) h_add / S`` over the
        overlap of the two bands (zero if they do not overlap).
        """
        for rows, ids in ((add_rows, add_ids), (remove_rows, remove_ids)):
            self._check_bands(rows[:, 1], N_pairs, data_index, start_freq_ind, data_minus_template,
                              data_length, T, "harmonic cross terms", row_ids=ids)
        GBGPUBase.swap_likelihood_difference(
            self,
            remove_rows,
            add_rows,
            data_minus_template,
            psd,
            start_freq_ind=start_freq_ind,
            data_index=data_index,
            noise_index=noise_index,
            N=N_pairs,
            T=T,
            data_length=data_length,
            phase_maximize=False,
            **common,
        )
        return self.add_remove.copy()

    def _accumulate(self, values, binary_index, num_binaries):
        """Sum ``values`` per binary (unbuffered scatter-add, no synchronization)."""
        xp = self.xp
        if xp.iscomplexobj(values) and xp is not np:  # cupyx.scatter_add takes real arrays only
            return (self._accumulate(values.real, binary_index, num_binaries)
                    + 1j * self._accumulate(values.imag, binary_index, num_binaries))
        out = xp.zeros(num_binaries, dtype=values.dtype)
        if xp is np:
            np.add.at(out, binary_index, values)
        else:
            import cupyx

            cupyx.scatter_add(out, binary_index, values)
        return out

    # --------------------------------------------------------------- waveforms
    def run_wave(
        self,
        amp,
        f0,
        fdot,
        fddot,
        phi0,
        iota,
        psi,
        lam,
        beta,
        e,
        Phi0,
        fPP,
        N: Optional[int] = None,
        T: float = 4 * YRSID_SI,
        dt: float = 10.0,
        oversample: int = 1,
        tdi2: bool = False,
        tdi_channel_setup: str = "AE",
        use_c_implementation: bool = False,
        window: Optional[str] = None,
        window_alpha: float = 0.0,
    ):
        """Create eccentric waveforms for a batch of binaries.

        Arguments as :meth:`gbgpu.gbgpu.GBGPUBase.run_wave` plus ``e``, ``Phi0``
        and ``fPP``. All harmonic rows share one ``N``; if ``None`` it is set
        by the significant harmonics (:meth:`special_get_N`). The result is
        stored as ``AETf`` (or ``XYZf``) with ``num_binaries * num_components``
        rows (one per harmonic and family); ``use_c_implementation=True`` uses
        the shared-memory kernel.

        """
        xp = self.xp
        params = self._as_param_rows(
            xp.stack(
                xp.broadcast_arrays(
                    *(xp.atleast_1d(xp.asarray(p, dtype=xp.float64))
                      for p in (amp, f0, fdot, fddot, phi0, iota, psi, lam, beta, e, Phi0, fPP))
                ),
                axis=1,
            )
        )
        num_binaries = params.shape[0]
        rows, scale, alpha = self._harmonic_rows(params)

        if N is None:
            N = int(
                self.special_get_N(
                    params[:, 0], params[:, 1], T, params[:, 9], params[:, 10], params[:, 11],
                    oversample=oversample,
                ).max()
            )
            if use_c_implementation:
                N = int(np.clip(N, _KERNEL_N_MIN, _KERNEL_N_MAX))
        N = int(N)
        if use_c_implementation:
            self._check_kernel_N([N])

        GBGPUBase.run_wave(
            self,
            *rows.T,
            N=N,
            T=T,
            dt=dt,
            oversample=oversample,
            tdi2=tdi2,
            tdi_channel_setup=tdi_channel_setup,
            use_c_implementation=use_c_implementation,
            window=window,
            window_alpha=window_alpha,
        )
        response = self.XYZf if tdi_channel_setup == "XYZ" else self.AETf
        response *= scale[:, None, None]

        self.num_binaries = num_binaries
        self.num_bin = rows.shape[0]
        self.harmonic_amplitudes_last = alpha
        self.harmonic_frequencies = rows[:, 1].reshape(num_binaries, self.num_components)

    def _harm_view(self, arr, *trailing):
        return arr.reshape((self.num_binaries, self.num_components) + tuple(trailing))

    @property
    def AETf_harm(self):
        """AET (or AE) of every row, ``(num_binaries, num_components, nchannels, N)``."""
        return self._harm_view(self.AETf, self.AETf.shape[1], self.N)

    @property
    def XYZf_harm(self):
        """XYZ of every row, ``(num_binaries, num_components, 3, N)``."""
        return self._harm_view(self.XYZf, 3, self.N)

    @property
    def A_harm(self):
        """A channel, ``(num_binaries, num_components, N)``."""
        return self.AETf_harm[:, :, 0]

    @property
    def E_harm(self):
        """E channel, ``(num_binaries, num_components, N)``."""
        return self.AETf_harm[:, :, 1]

    @property
    def start_inds_harm(self):
        """Start frequency indices, ``(num_binaries, num_components)``."""
        return self._harm_view(self.start_inds)

    @property
    def freqs_harm(self):
        """Frequencies of each row band, ``(num_binaries, num_components, N)``."""
        return self._harm_view(self.freqs, self.N)

    # --------------------------------------------------------------- likelihood
    def get_ll(
        self,
        params,
        data_minus_template,
        psd,
        start_freq_ind=0,
        data_index=None,
        noise_index=None,
        adjust_inplace=False,
        use_c_implementation=True,
        N=None,
        T=4 * YRSID_SI,
        dt=10.0,
        data_length=None,
        data_splits=None,
        phase_maximize=False,
        tdi_channel_setup="AE",
        num_per_gpu=None,
        oversample=1,
        return_cupy=False,
        tdi2: bool = False,
        window: Optional[str] = None,
        window_alpha: float = 0.0,
        include_cross_terms: bool = True,
    ):
        """Batched log-likelihood of eccentric binaries.

        Same interface and data/PSD layouts as
        :meth:`gbgpu.gbgpu.GBGPUBase.get_ll`, with ``params`` of shape
        ``(num_binaries, 12)``. ``N`` may be ``None`` (one ``N`` per harmonic
        from :func:`gbgpu.utils.utility.get_N`), an int, or one value per
        binary or per harmonic row. The per-harmonic inner products are
        evaluated by the shared-memory kernel and summed (Moore et al. eq. 17);
        ``<h|h>`` includes the cross terms of harmonic bands that overlap
        (``include_cross_terms=False`` drops them), and such binaries use their
        largest ``N`` for every harmonic. ``phase_maximize`` maximizes over
        ``phi``, which rotates all harmonics together (the maximum is at
        ``phi0 - phase_angle``).

        After the call ``d_h_harm`` and ``h_h_harm`` hold the per-row inner
        products, ``(num_binaries, num_components)`` (one per harmonic without
        correction families): ``sqrt(h_h_harm.real)`` are the harmonic SNRs of
        eq. (20). With correction families they exclude the cross terms between
        the rows of the same harmonic.

        """
        if self.d_d is None:
            raise ValueError(
                "self.d_d attribute must be set before computing log-Likelihood. "
                "This attribute is the data with data inner product (<d|d>)."
            )
        self._require_shared_memory_backend(use_c_implementation, "get_ll")
        if phase_maximize and (self._breathing or self._counter_rotating):
            raise ValueError(
                "phase_maximize rotates every row by the same phase, which is only true for the main "
                "family: phi0 does not enter the breathing rows and enters the counter-rotating rows "
                "with the opposite sign."
            )
        xp = self.xp
        start_freq_ind, data_length, T, dt = self._plain_scalars(start_freq_ind, data_length, T, dt)

        params = self._as_param_rows(params)
        num_binaries, num_harm = params.shape[0], self.num_components
        rows, scale, _ = self._harmonic_rows(params)
        N_rows = self._row_N(N, rows[:, 1], scale, T, oversample, num_binaries)
        N_rows, ((ia, ib),) = self._harmonic_pairs(
            [(params, params, rows[:, 1], rows[:, 1], self._pairs_upper, self._pairs_upper_same)], N_rows, T
        )

        if data_index is None:
            data_index = xp.zeros(num_binaries, dtype=xp.int32)
        if noise_index is None:
            noise_index = xp.zeros(num_binaries, dtype=xp.int32)
        data_index_rows = self.repeat_per_harmonic(data_index).astype(xp.int32)
        noise_index_rows = self.repeat_per_harmonic(noise_index).astype(xp.int32)

        self._check_bands(rows[:, 1], N_rows, data_index_rows, start_freq_ind, data_minus_template,
                          data_length, T, "get_ll")

        common = dict(
            use_c_implementation=True, dt=dt, data_splits=data_splits,
            tdi_channel_setup=tdi_channel_setup, num_per_gpu=num_per_gpu, tdi2=tdi2,
            window=window, window_alpha=window_alpha,
        )

        d_d = self.d_d
        self.d_d = 0.0
        try:
            GBGPUBase.get_ll(
                self,
                rows,
                data_minus_template,
                psd,
                start_freq_ind=start_freq_ind,
                data_index=data_index_rows,
                noise_index=noise_index_rows,
                N=N_rows,
                T=T,
                data_length=data_length,
                phase_maximize=False,
                oversample=oversample,
                return_cupy=True,
                **common,
            )
        finally:
            self.d_d = d_d

        d_h_harm = (self.d_h * scale).reshape(num_binaries, num_harm)
        h_h_harm = (self.h_h * scale * scale).reshape(num_binaries, num_harm)
        d_h = d_h_harm.sum(axis=1)
        h_h = h_h_harm.sum(axis=1)

        if include_cross_terms and ia.size > 0:
            products = self._pair_products(
                rows[ia], rows[ib], ia, ib, xp.maximum(N_rows[ia], N_rows[ib]),
                data_index_rows[ia], noise_index_rows[ia],
                data_minus_template, psd, start_freq_ind, data_length, T, common,
            )
            h_h = h_h + self._accumulate(
                2.0 * (scale[ia] * scale[ib] * products).real, ia // num_harm, num_binaries
            )

        self.num_binaries = num_binaries
        self.num_bin = rows.shape[0]
        self.d_h_harm = d_h_harm
        self.h_h_harm = h_h_harm

        if phase_maximize:
            self.non_marg_d_h = _to_host(d_h.copy())
            self.phase_angle = xp.arctan2(d_h.imag, d_h.real)
            d_h = xp.abs(d_h)
        else:
            self.phase_angle = None

        self.h_h = h_h
        self.d_h = d_h

        like_out = -1.0 / 2.0 * (self.d_d + h_h - 2 * d_h).real
        return like_out if return_cupy else _to_host(like_out)

    def swap_likelihood_difference(
        self,
        params_remove,
        params_add,
        data_minus_template,
        psd,
        start_freq_ind=0,
        data_index=None,
        noise_index=None,
        adjust_inplace=False,
        use_c_implementation=True,
        N=None,
        T=4 * YRSID_SI,
        dt=10.0,
        oversample=1,
        data_length=None,
        data_splits=None,
        phase_maximize=False,
        tdi_channel_setup="AE",
        num_per_gpu=None,
        tdi2: bool = False,
        window: Optional[str] = None,
        window_alpha: float = 0.0,
        include_cross_terms: bool = True,
        **kwargs,
    ):
        """Log-likelihood change when replacing eccentric binaries.

        Same interface as :meth:`gbgpu.gbgpu.GBGPUBase.swap_likelihood_difference`
        with ``params_remove`` and ``params_add`` of shape ``(num_binaries, 12)``.
        Harmonic ``n`` of the added binary is paired with harmonic ``n`` of the
        removed one in the kernel (both evaluated with the larger of their
        ``N``); products between different harmonics are added wherever their
        bands overlap, so the five terms are exact also for unrelated sources.
        ``phase_maximize`` rotates the added binary (maximum at
        ``phi0_add - phase_angle``).

        """
        self._require_shared_memory_backend(use_c_implementation, "swap_likelihood_difference")
        if phase_maximize and (self._breathing or self._counter_rotating):
            raise ValueError(
                "phase_maximize rotates every row by the same phase, which is only true for the main "
                "family: phi0 does not enter the breathing rows and enters the counter-rotating rows "
                "with the opposite sign."
            )
        xp = self.xp
        start_freq_ind, data_length, T, dt = self._plain_scalars(start_freq_ind, data_length, T, dt)

        params_add = self._as_param_rows(params_add, "params_add")
        params_remove = self._as_param_rows(params_remove, "params_remove")
        if params_add.shape != params_remove.shape:
            raise ValueError("params_add and params_remove must have the same shape.")
        num_binaries, num_harm = params_add.shape[0], self.num_components

        rows_add, s_add, _ = self._harmonic_rows(params_add)
        rows_rem, s_rem, _ = self._harmonic_rows(params_remove)
        num_rows = rows_add.shape[0]
        if N is None:  # one host round trip for both sets of harmonics
            N_both = self._row_N(
                None, xp.concatenate([rows_add[:, 1], rows_rem[:, 1]]), xp.concatenate([s_add, s_rem]),
                T, oversample, 2 * num_binaries,
            )
            N_rows = xp.maximum(N_both[:num_rows], N_both[num_rows:])
        else:
            N_rows = self._row_N(N, rows_add[:, 1], s_add, T, oversample, num_binaries)
        N_rows, (aa, rr, ar) = self._harmonic_pairs(
            [
                (params_add, params_add, rows_add[:, 1], rows_add[:, 1], self._pairs_upper, self._pairs_upper_same),
                (params_remove, params_remove, rows_rem[:, 1], rows_rem[:, 1], self._pairs_upper, self._pairs_upper_same),
                (params_add, params_remove, rows_add[:, 1], rows_rem[:, 1], self._pairs_offdiag, self._pairs_offdiag_same),
            ],
            N_rows,
            T,
        )

        if data_index is None:
            data_index = xp.zeros(num_binaries, dtype=xp.int32)
        if noise_index is None:
            noise_index = xp.zeros(num_binaries, dtype=xp.int32)
        data_index_rows = self.repeat_per_harmonic(data_index).astype(xp.int32)
        noise_index_rows = self.repeat_per_harmonic(noise_index).astype(xp.int32)

        for rows in (rows_add, rows_rem):
            self._check_bands(rows[:, 1], N_rows, data_index_rows, start_freq_ind,
                              data_minus_template, data_length, T, "swap_likelihood_difference")

        common = dict(
            use_c_implementation=True, dt=dt, data_splits=data_splits,
            tdi_channel_setup=tdi_channel_setup, num_per_gpu=num_per_gpu, tdi2=tdi2,
            window=window, window_alpha=window_alpha,
        )

        GBGPUBase.swap_likelihood_difference(
            self,
            rows_rem,
            rows_add,
            data_minus_template,
            psd,
            start_freq_ind=start_freq_ind,
            data_index=data_index_rows,
            noise_index=noise_index_rows,
            N=N_rows,
            T=T,
            oversample=oversample,
            data_length=data_length,
            phase_maximize=False,
            **common,
        )

        shape = (num_binaries, num_harm)
        d_h_add = (self.d_h_add * s_add).reshape(shape).sum(axis=1)
        d_h_remove = (self.d_h_remove * s_rem).reshape(shape).sum(axis=1)
        add_add = (self.add_add * s_add * s_add).reshape(shape).sum(axis=1)
        remove_remove = (self.remove_remove * s_rem * s_rem).reshape(shape).sum(axis=1)
        add_remove = (self.add_remove * s_add * s_rem).reshape(shape).sum(axis=1)

        counts = [int(pair[0].size) for pair in (aa, rr, ar)]
        if include_cross_terms and sum(counts) > 0:
            add_pairs = xp.concatenate([rows_add[aa[0]], rows_rem[rr[0]], rows_add[ar[0]]])
            rem_pairs = xp.concatenate([rows_add[aa[1]], rows_rem[rr[1]], rows_rem[ar[1]]])
            first = xp.concatenate([aa[0], rr[0], ar[0]])
            second = xp.concatenate([aa[1], rr[1], ar[1]])
            products = self._pair_products(
                add_pairs, rem_pairs, first, second, xp.maximum(N_rows[first], N_rows[second]),
                data_index_rows[first], noise_index_rows[first],
                data_minus_template, psd, start_freq_ind, data_length, T, common,
            )
            c0, c1 = counts[0], counts[0] + counts[1]
            if counts[0]:
                add_add = add_add + self._accumulate(
                    2.0 * (s_add[aa[0]] * s_add[aa[1]] * products[:c0]).real,
                    aa[0] // num_harm, num_binaries,
                )
            if counts[1]:
                remove_remove = remove_remove + self._accumulate(
                    2.0 * (s_rem[rr[0]] * s_rem[rr[1]] * products[c0:c1]).real,
                    rr[0] // num_harm, num_binaries,
                )
            if counts[2]:
                add_remove = add_remove + self._accumulate(
                    s_add[ar[0]] * s_rem[ar[1]] * products[c1:], ar[0] // num_harm, num_binaries
                )

        self.num_binaries = num_binaries
        self.num_bin = rows_add.shape[0]

        if phase_maximize:
            self.phase_angle = xp.arctan2(d_h_add.imag + add_remove.imag, d_h_add.real + add_remove.real)
            self.non_marg_d_h_add = _to_host(d_h_add.copy())
            self.non_marg_add_remove = _to_host(add_remove.copy())
            d_h_add = d_h_add * xp.exp(-1j * self.phase_angle)
            add_remove = add_remove * xp.exp(-1j * self.phase_angle)

        self.d_h_remove = d_h_remove
        self.d_h_add = d_h_add
        self.add_remove = add_remove
        self.remove_remove = remove_remove
        self.add_add = add_add

        ll_diff = -1 / 2 * (-2 * d_h_add + 2 * d_h_remove - 2 * add_remove + add_add + remove_remove).real
        return _to_host(ll_diff)

    def get_fstat_ll(self, *args, **kwargs):
        """Not available: the F-statistic maximizes circular-source extrinsic
        parameters, which do not factor out of the eccentric multi-harmonic model."""
        raise NotImplementedError(
            "The F-statistic is not defined for the eccentric multi-harmonic model."
        )

    # -------------------------------------------------------- global templates
    def generate_global_template(
        self,
        params,
        group_index,
        templates,
        start_freq_ind=0,
        use_c_implementation=True,
        N=None,
        T=4 * YRSID_SI,
        dt=10.0,
        batch_size=None,
        oversample=1,
        data_length=None,
        tdi_channel_setup="AE",
        data_splits=None,
        num_per_gpu=None,
        factors=None,
        tdi2: bool = False,
        window: Optional[str] = None,
        window_alpha: float = 0.0,
        **kwargs,
    ):
        """Add eccentric binaries to global templates.

        Same interface as :meth:`gbgpu.gbgpu.GBGPUBase.generate_global_template`
        with ``params`` of shape ``(num_binaries, 12)`` and ``group_index`` /
        ``factors`` with one entry per binary: all harmonics of a binary are
        added (times its factor) to the template ``group_index``.

        """
        xp = self.xp
        params = self._as_param_rows(params)
        num_binaries = params.shape[0]
        start_freq_ind, data_length, T, dt = self._plain_scalars(start_freq_ind, data_length, T, dt)
        group_index = xp.asarray(group_index).astype(xp.int32)
        if group_index.ndim != 1 or group_index.shape[0] != num_binaries:
            raise ValueError(f"group_index must have one entry per binary ({num_binaries}).")
        factors = xp.ones(num_binaries) if factors is None else xp.asarray(factors, dtype=xp.float64)
        if factors.shape[0] != num_binaries:
            raise ValueError(f"factors must have one entry per binary ({num_binaries}).")

        rows, scale, _ = self._harmonic_rows(params)
        N_rows = self._row_N(N, rows[:, 1], scale, T, oversample, num_binaries,
                             clip_to_kernel=use_c_implementation, check_kernel=use_c_implementation)
        N_rows, _ = self._harmonic_pairs(
            [(params, params, rows[:, 1], rows[:, 1], self._pairs_upper, self._pairs_upper_same)], N_rows, T
        )
        group_rows = self.repeat_per_harmonic(group_index).astype(xp.int32)

        self._check_bands(rows[:, 1], N_rows, group_rows, start_freq_ind, templates, data_length, T,
                          "generate_global_template")

        if use_c_implementation:
            GBGPUBase.generate_global_template(
                self,
                rows,
                group_rows,
                templates,
                start_freq_ind=start_freq_ind,
                use_c_implementation=True,
                N=N_rows,
                T=T,
                dt=dt,
                batch_size=batch_size,
                oversample=oversample,
                data_length=data_length,
                tdi_channel_setup=tdi_channel_setup,
                data_splits=data_splits,
                num_per_gpu=num_per_gpu,
                factors=(self.repeat_per_harmonic(factors) * scale).astype(xp.float64),
                tdi2=tdi2,
                window=window,
                window_alpha=window_alpha,
            )
            return

        # Python path: circular FastGB of each (non-zero) harmonic, accumulated
        # with fill_global_template, grouped by N.
        if not isinstance(templates, xp.ndarray) or templates.ndim != 3:
            raise ValueError("The Python path needs one 3D (num_templates, 2, data_length) array.")
        if tdi_channel_setup != "AE":
            raise NotImplementedError("The Python path accumulates the A and E channels only.")
        amp_rows = self.repeat_per_harmonic(factors) * scale
        keep = xp.nonzero(amp_rows != 0.0)[0]
        phys_rows = rows[keep].copy()
        phys_rows[:, 0] = amp_rows[keep]
        N_keep = N_rows[keep]
        for N_here in _to_host(xp.unique(N_keep)):
            sel = xp.nonzero(N_keep == int(N_here))[0]
            step = int(sel.size) if batch_size is None else int(batch_size)
            for start in range(0, int(sel.size), step):
                chunk = sel[start:start + step]
                GBGPUBase.run_wave(
                    self, *phys_rows[chunk].T, N=int(N_here), T=T, dt=dt, tdi2=tdi2,
                    tdi_channel_setup="AE", use_c_implementation=False,
                    window=window, window_alpha=window_alpha,
                )
                group_chunk = group_rows[keep][chunk]
                start_inds = self.start_inds.astype(xp.int64)
                if isinstance(start_freq_ind, int):
                    start_inds = start_inds - start_freq_ind
                else:  # one start index per template
                    start_inds = start_inds - xp.asarray(start_freq_ind).astype(xp.int64)[group_chunk]
                self.fill_global_template(
                    group_chunk, templates, self.A_out, self.E_out,
                    start_inds.astype(xp.int32), self.N, start_freq_ind=0,
                )

    def inject_signal(self, params, data_length, start_freq_ind=0, tdi_channel_setup="AE", **kwargs):
        """Sum eccentric binaries into one frequency-domain data stream.

        Returns:
            2D complex xp.ndarray: ``(nchannels, data_length)`` on frequencies
            ``(start_freq_ind + arange(data_length)) / T``.

        """
        params = self._as_param_rows(params)
        nchannels = _tdi_nchannels[tdi_channel_setup]
        templates = self.xp.zeros((1, nchannels, data_length), dtype=self.xp.complex128)
        self.generate_global_template(
            params, self.xp.zeros(params.shape[0], dtype=self.xp.int32), templates,
            start_freq_ind=start_freq_ind, tdi_channel_setup=tdi_channel_setup,
            data_length=data_length, **kwargs,
        )
        return templates[0]

    # ------------------------------------------------------ information matrix
    def information_matrix(
        self,
        params,
        psd,
        inds=None,
        eps: float = 1e-9,
        parameter_transforms: Dict = {},
        easy_central_difference: bool = False,
        noise_index=None,
        use_c_implementation: bool = False,
        N=None,
        T: float = 4 * YRSID_SI,
        dt: float = 10.0,
        data_length=None,
        data_splits=None,
        tdi_channel_setup: str = "AE",
        num_per_gpu: Optional[int] = None,
        oversample: int = 1,
        return_cupy: bool = False,
        tdi2: bool = False,
        window: Optional[str] = None,
        window_alpha: float = 0.0,
        batch_size: int = 50000,
        workspace_budget_bytes: int = 4 * 1024**3,
        **kwargs,
    ):
        """Fisher information matrices of eccentric binaries (Python path).

        Same conventions as the Python path of
        :meth:`gbgpu.gbgpu.GBGPUBase.information_matrix`: ``params`` of shape
        ``(num_binaries, 12)``, absolute steps ``eps`` (scalar or one per entry
        of ``inds``) in the sampling basis mapped by ``parameter_transforms``,
        ``psd`` the flattened inverse-PSD array of length
        ``num_psd * nchannels (* nchannels) * data_length`` starting at
        frequency bin 0. The matrix is the sum of the harmonic contributions
        (bands of different harmonics do not overlap in practice). If a step
        moves the heterodyne bin of a harmonic, the perturbed band is
        re-aligned on the reference grid. With correction families, the rows of
        one harmonic are summed on the grid of its main row before the inner
        product (they are ``2 |fPP|`` apart; bins shifted out of that grid are
        dropped).

        Returns:
            3D double array: ``(num_binaries, num_derivs, num_derivs)``.

        """
        if use_c_implementation:
            raise NotImplementedError(
                "The shared-memory Fisher kernel works in the 9 circular parameters; "
                "use the Python path (use_c_implementation=False) for eccentric binaries."
            )
        if data_length is None:
            raise ValueError("data_length is required so the psd can be reshaped and sliced.")
        xp = self.xp

        params = self._as_param_rows(params)
        num_binaries, num_params = params.shape
        if inds is None:
            inds = np.arange(num_params)
        inds = np.atleast_1d(_to_host(inds)).astype(int)
        num_derivs = len(inds)
        steps = np.broadcast_to(np.asarray(_to_host(eps), dtype=np.float64), (num_derivs,))
        if noise_index is None:
            noise_index = xp.zeros(num_binaries, dtype=xp.int32)
        noise_index = xp.asarray(noise_index)

        nchannels = _tdi_nchannels[tdi_channel_setup]
        num_harm = self.num_harmonics
        num_fam = len(self.families)
        if tdi_channel_setup == "XYZ":
            inv_psd_all = psd.reshape(-1, 3, 3, data_length)
        else:
            inv_psd_all = psd.reshape(-1, nchannels, data_length)

        def transform(p):
            if not parameter_transforms:
                return p
            return self._apply_parameter_transforms(p.T.copy(), parameter_transforms).T

        phys = transform(params.copy())
        if N is None:
            N_val = int(
                self.special_get_N(phys[:, 0], phys[:, 1], T, phys[:, 9], phys[:, 10], phys[:, 11],
                                   oversample=oversample).max()
            )
        else:
            N_val = int(_to_host(xp.asarray(N)).max())
        self.df = 1.0 / T

        waveform_kwargs = dict(
            N=N_val, T=T, dt=dt, tdi2=tdi2, tdi_channel_setup=tdi_channel_setup,
            use_c_implementation=False, window=window, window_alpha=window_alpha,
        )

        bytes_per_binary = num_derivs * num_harm * num_fam * nchannels * N_val * 16 * (3 if easy_central_difference else 5)
        step_size = max(1, min(int(batch_size), int(workspace_budget_bytes) // max(bytes_per_binary, 1)))
        info_matrix = xp.zeros((num_binaries, num_derivs, num_derivs), dtype=xp.float64)
        bins = xp.arange(N_val, dtype=xp.int64)[None, :]

        def response():
            return self.XYZf if tdi_channel_setup == "XYZ" else self.AETf

        for start in range(0, num_binaries, step_size):
            end = min(start + step_size, num_binaries)
            batch = params[start:end]
            nb = end - start

            self.run_wave(*transform(batch.copy()).T, **waveform_kwargs)
            ref_start = self.start_inds.astype(xp.int64).copy()
            # inner products on the grid of the main row of every harmonic
            main_start = ref_start.reshape(nb, num_fam, num_harm)[:, 0].reshape(-1)

            f_idx = ref_start[:, None] + bins
            if int(f_idx.min()) < 0 or int(f_idx.max()) >= data_length:
                raise ValueError(
                    "A harmonic band falls outside the psd frequency range [0, data_length)."
                )
            f_idx = main_start[:, None] + bins
            w_idx = xp.repeat(noise_index[start:end], num_harm)
            if tdi_channel_setup == "XYZ":
                inv_psd = inv_psd_all[w_idx[:, None, None, None], xp.arange(3)[None, :, None, None],
                                      xp.arange(3)[None, None, :, None], f_idx[:, None, None, :]]
                inv_psd = inv_psd.reshape(nb, num_harm, 3, 3, N_val)
            else:
                inv_psd = inv_psd_all[w_idx[:, None, None], xp.arange(nchannels)[None, :, None],
                                      f_idx[:, None, :]].reshape(nb, num_harm, nchannels, N_val)

            def get_wave(p):
                self.run_wave(*transform(p).T, **waveform_kwargs)
                resp = response()
                offset = (ref_start - self.start_inds.astype(xp.int64))[:, None]
                if num_fam > 1:  # move every row onto the grid of its main row
                    offset = offset + (xp.repeat(main_start.reshape(nb, 1, num_harm), num_fam, axis=1).reshape(-1)
                                       - ref_start)[:, None]
                if bool(xp.any(offset != 0)):
                    idx = bins + offset
                    valid = (idx >= 0) & (idx < N_val)
                    idx = xp.broadcast_to(xp.clip(idx, 0, N_val - 1)[:, None, :], resp.shape)
                    resp = xp.where(valid[:, None, :], xp.take_along_axis(resp, idx, axis=-1), 0.0)
                resp = resp.reshape(nb, num_fam, num_harm, nchannels, N_val)
                return (resp[:, 0] if num_fam == 1 else resp.sum(axis=1)).copy()

            dh = xp.zeros((nb, num_derivs, num_harm, nchannels, N_val), dtype=xp.complex128)
            for i, (ind, step) in enumerate(zip(inds, steps)):
                def shifted(k):
                    p = batch.copy()
                    p[:, ind] += k * step
                    return get_wave(p)

                if easy_central_difference:
                    dh[:, i] = (shifted(1.0) - shifted(-1.0)) / (2.0 * step)
                else:
                    dh[:, i] = (
                        -shifted(2.0) + shifted(-2.0) + 8.0 * (shifted(1.0) - shifted(-1.0))
                    ) / (12.0 * step)

            if tdi_channel_setup == "XYZ":
                fim = xp.einsum("bihck,bhcdk,bjhdk->bij", dh.conj(), inv_psd, dh)
            else:
                fim = xp.einsum("bihck,bjhck,bhck->bij", dh.conj(), dh, inv_psd)
            info_matrix[start:end] = 4.0 * self.df * fim.real

        return info_matrix if return_cupy else _to_host(info_matrix)
