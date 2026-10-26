"""Tests of the eccentric Galactic-binary waveform (``gbgpu.eccentric``).

The eccentric model of Moore et al. (2024) is a sum of circular FastGB
harmonics evaluated by the shared-memory kernels, so the tests check it
against independent references: the circular ``GBGPU`` class (``e = 0`` and
each mapped harmonic), explicit inner products of the Python waveforms placed
in the data band (likelihood, swap, cross terms, all TDI setups), analytic
relations of the information matrix, and the Bessel closed form of the
harmonic amplitudes. Runs on the GPU when CuPy is available, else on the CPU.
"""

import unittest
import warnings
from fractions import Fraction

import numpy as np
from scipy.special import jv

try:
    import cupy as cp

    gpu_available = True
except (ImportError, ModuleNotFoundError):
    gpu_available = False

from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import A1TDISens
from lisatools.utils.constants import YRSID_SI

from gbgpu.gbgpu import GBGPU
from gbgpu.utils.utility import get_N
from gbgpu.eccentric import (
    GBGPUEccentric,
    eccentric_amplitude_coefficients,
    eccentric_correction_amplitudes,
    eccentric_harmonic_amplitudes,
    eccentric_to_circular_harmonics,
    eccentricity_enhancement,
    get_fdot_eccentric,
    get_fPP,
    get_masses_from_eccentric_parameters,
)

xp = cp if gpu_available else np
force_backend = "gpu" if gpu_available else "cpu"

dt = 15.0
# multiple of dt: run_wave rounds T to N_obs * dt, the kernels take T as given
T = float(int(YRSID_SI / dt) * dt)
df = 1.0 / T
N_WAVE = 256

#: amp, f0, fdot, fddot, phi0, iota, psi, lam, beta, e, Phi0, fPP
PARAMS = np.array(
    [
        [2.0e-22, 2.0e-3, 1e-17, 1e-28, 0.3, 0.8, 0.4, 1.2, 0.3, 0.25, 0.7, -2e-9],
        [1.5e-22, 2.3e-3, 3e-17, 0.0, 1.3, 0.5, 1.4, 4.2, -0.3, 0.1, 2.7, 3e-9],
    ]
)
HARMONICS = 6
# data band covering n = 1 ... 6 of both binaries
START_FREQ_IND = int(0.8e-3 * T)
DATA_LENGTH = int(7.2e-3 * T) - START_FREQ_IND


def to_np(arr):
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


def band_inv_psd(start_freq_ind, data_length):
    f = np.maximum((start_freq_ind + np.arange(data_length)) * df, df)
    return xp.asarray(1.0 / A1TDISens.get_Sn(f))


def inner(a, b, w):
    """4 df sum conj(a) b w (diagonal channels)."""
    return complex(4.0 * df * xp.sum(xp.conj(a) * b * w))


def inner_xyz(a, b, w):
    """4 df sum_{c,d} conj(a_c) w_cd b_d (full channel covariance)."""
    return complex(4.0 * df * xp.einsum("ck,cdk,dk->", xp.conj(a), w, b))


def place(wave, start_inds, start_freq_ind, data_length):
    """Sum harmonic waveforms ``(H, nchannels, N)`` into the data band."""
    wave = xp.asarray(wave)
    start_inds = to_np(start_inds)
    out = xp.zeros((wave.shape[1], data_length), dtype=complex)
    N = wave.shape[-1]
    for h in range(wave.shape[0]):
        s = int(start_inds[h]) - start_freq_ind
        out[:, s:s + N] += wave[h]
    return out


_ORBITS = []


def shared_orbits():
    """One configured orbits object for the whole module (configuring allocates ~1 GB)."""
    if not _ORBITS:
        _ORBITS.append(EqualArmlengthOrbits(force_backend=force_backend))
    return _ORBITS[0]


class EccentricTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orbits = shared_orbits()

    def make(self, **kwargs):
        kwargs.setdefault("harmonics", HARMONICS)
        return GBGPUEccentric(orbits=self.orbits, force_backend=force_backend, **kwargs)

    def assert_close(self, a, b, rtol=1e-10, atol=0.0):
        np.testing.assert_allclose(to_np(a), to_np(b), rtol=rtol, atol=atol)

    def harmonic_waves(self, gb, params, **kwargs):
        """Python-path harmonics of each binary placed in the data band."""
        setup = kwargs.get("tdi_channel_setup", "AE")
        gb.run_wave(*params.T, N=kwargs.pop("N", N_WAVE), T=T, dt=dt, use_c_implementation=False, **kwargs)
        wave = gb.XYZf_harm if setup == "XYZ" else gb.AETf_harm
        start = gb.start_inds_harm
        return [
            [place(wave[b, h:h + 1], start[b, h:h + 1], START_FREQ_IND, DATA_LENGTH)
             for h in range(gb.num_harmonics)]
            for b in range(params.shape[0])
        ]


class AmplitudeTest(EccentricTestBase):
    def test_rational_coefficients(self):
        coeffs = eccentric_amplitude_coefficients(3, order=4, exact=True)
        self.assertEqual(coeffs[0][:4], [0, Fraction(-3, 4), 0, Fraction(13, 32)])
        self.assertEqual(coeffs[1], [1, 0, Fraction(-5, 2), 0, Fraction(23, 16)])
        self.assertEqual(coeffs[2][:4], [0, Fraction(9, 4), 0, Fraction(-171, 32)])

    def test_bessel_closed_form(self):
        e = np.array([0.05, 0.3, 0.6, 0.8])
        n = np.arange(1, 13, dtype=float)[None, :]
        ne = n * e[:, None]
        ecc = e[:, None]
        ref = n / 4.0 * (
            jv(n - 2, ne) - 2 * ecc * jv(n - 1, ne) + 2.0 / n * jv(n, ne)
            + 2 * ecc * jv(n + 1, ne) - jv(n + 2, ne)
            + np.sqrt(1 - ecc**2) * (jv(n - 2, ne) - 2 * jv(n, ne) + jv(n + 2, ne))
        )
        exact = eccentric_harmonic_amplitudes(e, 12, order=None, xp=np)
        np.testing.assert_allclose(exact, ref, atol=1e-12)
        series = eccentric_harmonic_amplitudes(e[:1], 12, order=10, xp=np)
        np.testing.assert_allclose(series, ref[:1], atol=1e-11)
        # negative eccentricity: h(-e, Phi0) = h(e, Phi0 + pi)
        sign = (-1.0) ** np.arange(1, 13)
        for order in (10, None):
            np.testing.assert_allclose(
                eccentric_harmonic_amplitudes(-e[:3], 12, order=order, xp=np),
                sign * eccentric_harmonic_amplitudes(e[:3], 12, order=order, xp=np),
                atol=1e-13,
            )

    def test_series_validity_and_exact_mode(self):
        ecc = self.make(harmonics=12)
        self.assertAlmostEqual(ecc.series_validity_limit, 0.40, delta=0.011)
        p = PARAMS[0].copy()
        p[9] = 0.55
        with self.assertWarns(UserWarning):
            ecc.run_wave(*p, N=N_WAVE, T=T, dt=dt)
        exact = self.make(harmonics=12, ecc_order=None)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            exact.run_wave(*p, N=N_WAVE, T=T, dt=dt)
        self.assert_close(exact.harmonic_amplitudes_last,
                          eccentric_harmonic_amplitudes(np.array([0.55]), 12, order=None, xp=np),
                          rtol=0.0, atol=1e-12)


class WaveformTest(EccentricTestBase):
    def test_circular_limit(self):
        gb = GBGPU(orbits=self.orbits, force_backend=force_backend)
        ecc = self.make(harmonics=12)
        p = PARAMS[0].copy()
        p[9] = 0.0
        for use_c in (False, True):
            gb.run_wave(*p[:9], N=N_WAVE, T=T, dt=dt, use_c_implementation=use_c)
            ecc.run_wave(*p, N=N_WAVE, T=T, dt=dt, use_c_implementation=use_c)
            ref = to_np(gb.AETf[0])
            self.assert_close(ecc.AETf_harm[0, 1], ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())
            others = np.delete(to_np(ecc.AETf_harm[0]), 1, axis=0)
            self.assertEqual(np.abs(others).max(), 0.0)
            self.assertEqual(int(to_np(ecc.start_inds_harm)[0, 1]), int(to_np(gb.start_inds)[0]))

    def test_harmonics_are_circular_sources(self):
        gb = GBGPU(orbits=self.orbits, force_backend=force_backend)
        for harmonics in (HARMONICS, [2, 3, 5]):
            ecc = self.make(harmonics=harmonics)
            circ = eccentric_to_circular_harmonics(*PARAMS.T, harmonics=harmonics, xp=np)
            rows = circ.reshape(9, -1)
            outputs = []
            for use_c in (False, True):
                ecc.run_wave(*PARAMS.T, N=N_WAVE, T=T, dt=dt, use_c_implementation=use_c)
                gb.run_wave(*rows, N=N_WAVE, T=T, dt=dt, use_c_implementation=use_c)
                scale = np.abs(to_np(gb.AETf)).max()
                self.assert_close(ecc.AETf, gb.AETf, rtol=1e-10, atol=1e-12 * scale)
                np.testing.assert_array_equal(to_np(ecc.start_inds), to_np(gb.start_inds))
                outputs.append(to_np(ecc.AETf))
            self.assert_close(outputs[0], outputs[1], rtol=1e-10, atol=1e-12 * np.abs(outputs[0]).max())
            self.assertEqual(ecc.AETf_harm.shape, (2, len(ecc.harmonics), 2, N_WAVE))

    def test_flip_ref_phase(self):
        flipped = self.make(flip_ref_phase=True)
        plain = self.make()
        mirrored = PARAMS.copy()
        mirrored[:, [4, 10]] *= -1.0
        for use_c in (False, True):
            flipped.run_wave(*PARAMS.T, N=N_WAVE, T=T, dt=dt, use_c_implementation=use_c)
            plain.run_wave(*mirrored.T, N=N_WAVE, T=T, dt=dt, use_c_implementation=use_c)
            self.assert_close(flipped.AETf, plain.AETf, rtol=1e-12, atol=1e-12 * np.abs(to_np(plain.AETf)).max())

        # injection and likelihood agree with each other under the flip
        psd = band_inv_psd(START_FREQ_IND, DATA_LENGTH)[None] * xp.ones((1, 2, 1))
        templates = {}
        for use_c in (True, False):
            tmpl = xp.zeros((1, 2, DATA_LENGTH), dtype=complex)
            flipped.generate_global_template(
                PARAMS[:1], xp.zeros(1, dtype=xp.int32), tmpl, start_freq_ind=START_FREQ_IND,
                N=N_WAVE, T=T, dt=dt, use_c_implementation=use_c,
            )
            templates[use_c] = tmpl
        self.assert_close(templates[True], templates[False], rtol=1e-10,
                          atol=1e-12 * np.abs(to_np(templates[True])).max())
        flipped.d_d = inner(templates[True][0], templates[True][0], psd[0]).real
        ll = flipped.get_ll(PARAMS[:1], templates[True], psd, start_freq_ind=START_FREQ_IND, N=N_WAVE, T=T, dt=dt)
        self.assertLess(abs(ll[0]), 1e-9 * flipped.d_d)


class LikelihoodTest(EccentricTestBase):
    def setUp(self):
        self.ecc = self.make()
        self.waves = self.harmonic_waves(self.ecc, PARAMS)
        self.h = [sum(w) for w in self.waves]
        w = band_inv_psd(START_FREQ_IND, DATA_LENGTH)
        self.psd = xp.stack([xp.stack([w, 1.3 * w]), xp.stack([2.0 * w, 0.7 * w])])  # (2, 2, dl)
        # data 0: both binaries; data 1: a rescaled, shifted copy of binary 0
        self.data = xp.stack([self.h[0] + self.h[1], 0.8 * xp.roll(self.h[0], 3, axis=-1)])

    def explicit_ll(self, b, d, w):
        h = self.h[b]
        return -0.5 * (inner(d, d, w) + inner(h, h, w) - 2 * inner(d, h, w)).real

    def test_get_ll_matches_explicit_inner_products(self):
        ecc = self.ecc
        data_index = xp.array([1, 0], dtype=xp.int32)
        noise_index = xp.array([0, 1], dtype=xp.int32)
        d_d = [inner(self.data[1], self.data[1], self.psd[0]).real,
               inner(self.data[0], self.data[0], self.psd[1]).real]
        ecc.d_d = xp.asarray(d_d)
        layouts = [
            dict(data=self.data, psd=self.psd, start_freq_ind=START_FREQ_IND),
            dict(data=self.data.flatten(), psd=self.psd.flatten(), start_freq_ind=START_FREQ_IND,
                 data_length=DATA_LENGTH),
            dict(data=self.data, psd=self.psd,
                 start_freq_ind=xp.full(2, START_FREQ_IND, dtype=xp.int32)),
        ]
        for layout in layouts:
            ll = ecc.get_ll(PARAMS, layout.pop("data"), layout.pop("psd"), data_index=data_index,
                            noise_index=noise_index, N=N_WAVE, T=T, dt=dt, **layout)
            ref = [self.explicit_ll(0, self.data[1], self.psd[0]), self.explicit_ll(1, self.data[0], self.psd[1])]
            np.testing.assert_allclose(ll, ref, rtol=1e-10)
            for b, (d, w) in enumerate(((self.data[1], self.psd[0]), (self.data[0], self.psd[1]))):
                for h in range(HARMONICS):
                    wave = self.waves[b][h]
                    self.assertAlmostEqual(complex(ecc.d_h_harm[b, h]), inner(d, wave, w),
                                           delta=1e-10 * abs(inner(d, d, w)))
                    self.assertAlmostEqual(complex(ecc.h_h_harm[b, h]), inner(wave, wave, w),
                                           delta=1e-10 * abs(inner(d, d, w)))

    def test_get_ll_at_injection_and_phase_maximization(self):
        ecc = self.ecc
        data = self.h[0][None]
        w = self.psd[:1]
        ecc.d_d = inner(data[0], data[0], w[0]).real
        ll = ecc.get_ll(PARAMS[:1], data, w, start_freq_ind=START_FREQ_IND, N=N_WAVE, T=T, dt=dt)
        self.assertLess(abs(ll[0]), 1e-10 * ecc.d_d)

        # maximize over phi for a de-phased source: phi0 - phase_angle is the maximum
        p = PARAMS[:1].copy()
        p[0, 4] += 1.1
        ll_max = ecc.get_ll(p, data, w, start_freq_ind=START_FREQ_IND, N=N_WAVE, T=T, dt=dt, phase_maximize=True)
        p_rot = p.copy()
        p_rot[0, 4] -= float(to_np(ecc.phase_angle)[0])
        ll_rot = ecc.get_ll(p_rot, data, w, start_freq_ind=START_FREQ_IND, N=N_WAVE, T=T, dt=dt)
        self.assertAlmostEqual(ll_max[0], ll_rot[0], delta=1e-9 * ecc.d_d)
        self.assertLess(abs(ll_rot[0]), 1e-9 * ecc.d_d)

    def test_tdi_setups_windows_tdi2(self):
        ecc = self.make()
        w = band_inv_psd(START_FREQ_IND, DATA_LENGTH)
        cases = [
            dict(tdi_channel_setup="AET"),
            dict(tdi_channel_setup="XYZ", tdi2=True),
            dict(tdi_channel_setup="AE", window="tukey", window_alpha=0.2),
        ]
        for case in cases:
            setup = case["tdi_channel_setup"]
            waves = self.harmonic_waves(ecc, PARAMS, **case)
            h = [sum(wv) for wv in waves]
            data = (h[0] + 0.5 * h[1])[None]
            if setup == "XYZ":
                rho = -0.3
                cov = xp.full((3, 3), rho) + (1 - rho) * xp.eye(3)
                psd = (xp.linalg.inv(cov)[:, :, None] * w[None, None, :])[None]  # (1, 3, 3, dl)
                prod = lambda a, b: inner_xyz(a, b, psd[0])
            else:
                psd = xp.stack([w * (1.0 + 0.1 * c) for c in range(3 if setup == "AET" else 2)])[None]
                prod = lambda a, b: inner(a, b, psd[0])
            ecc.d_d = prod(data[0], data[0]).real
            ll = ecc.get_ll(PARAMS, data, psd, start_freq_ind=START_FREQ_IND, N=N_WAVE, T=T, dt=dt, **case)
            ref = [-0.5 * (ecc.d_d + prod(hb, hb) - 2 * prod(data[0], hb)).real for hb in h]
            np.testing.assert_allclose(ll, ref, rtol=1e-9, err_msg=str(case))

            templates = xp.zeros((1, data.shape[1], DATA_LENGTH), dtype=complex)
            ecc.generate_global_template(PARAMS, xp.zeros(2, dtype=xp.int32), templates,
                                         start_freq_ind=START_FREQ_IND, N=N_WAVE, T=T, dt=dt, **case)
            self.assert_close(templates[0], h[0] + h[1], rtol=1e-9, atol=1e-12 * float(xp.abs(h[0]).max()))

    def test_swap_likelihood_difference(self):
        ecc = self.ecc
        data = self.data[:1]
        w = self.psd[:1]
        # model currently holds both binaries; propose a nearby and an unrelated source
        p_add = PARAMS.copy()
        p_add[0, [1, 9, 10]] = [PARAMS[0, 1] * (1 + 3e-7), 0.3, 0.5]
        p_add[1] = [9e-23, 2.15e-3, 0.0, 0.0, 2.0, 1.1, 0.2, 5.0, 0.9, 0.35, -1.0, 1e-8]
        waves_add = self.harmonic_waves(ecc, p_add)
        resid = data[0] - self.h[0] - self.h[1]
        ref = []
        for b in range(2):
            h_add = sum(waves_add[b])
            new = resid + self.h[b] - h_add
            ref.append(-0.5 * (inner(new, new, w[0]) - inner(resid, resid, w[0])).real)
        ll_diff = ecc.swap_likelihood_difference(
            PARAMS, p_add, resid[None], w, start_freq_ind=START_FREQ_IND, N=N_WAVE, T=T, dt=dt
        )
        np.testing.assert_allclose(ll_diff, ref, rtol=1e-9)

        # phase maximization: phi0 - phase_angle realizes the maximized difference
        ll_max = ecc.swap_likelihood_difference(
            PARAMS, p_add, resid[None], w, start_freq_ind=START_FREQ_IND, N=N_WAVE, T=T, dt=dt,
            phase_maximize=True,
        )
        p_rot = p_add.copy()
        p_rot[:, 4] -= to_np(ecc.phase_angle)
        ll_rot = ecc.swap_likelihood_difference(
            PARAMS, p_rot, resid[None], w, start_freq_ind=START_FREQ_IND, N=N_WAVE, T=T, dt=dt
        )
        np.testing.assert_allclose(ll_max, ll_rot, rtol=1e-9)
        self.assertTrue(np.all(ll_max >= np.asarray(ref) - 1e-9 * np.abs(ref)))

    def test_global_template_paths(self):
        ecc = self.ecc
        group_index = xp.array([1, 0], dtype=xp.int32)
        factors = xp.array([0.5, -2.0])
        ref = xp.stack([-2.0 * self.h[1], 0.5 * self.h[0]])
        for use_c in (True, False):
            for sfi in (START_FREQ_IND, xp.full(2, START_FREQ_IND, dtype=xp.int32)):
                templates = xp.zeros((2, 2, DATA_LENGTH), dtype=complex)
                ecc.generate_global_template(PARAMS, group_index, templates, start_freq_ind=sfi, N=N_WAVE,
                                             T=T, dt=dt, factors=factors, use_c_implementation=use_c)
                self.assert_close(templates, ref, rtol=1e-9, atol=1e-12 * float(xp.abs(ref).max()))
        injected = ecc.inject_signal(PARAMS, DATA_LENGTH, start_freq_ind=START_FREQ_IND, N=N_WAVE, T=T, dt=dt)
        self.assert_close(injected, self.h[0] + self.h[1], rtol=1e-9, atol=1e-12 * float(xp.abs(ref).max()))

    def test_sample_numbers(self):
        ecc = self.ecc
        data, w = self.data[:1], self.psd[:1]
        ecc.d_d = inner(data[0], data[0], w[0]).real
        kwargs = dict(start_freq_ind=START_FREQ_IND, T=T, dt=dt)
        ll_int = ecc.get_ll(PARAMS, data, w, N=N_WAVE, **kwargs)
        ll_bin = ecc.get_ll(PARAMS, data, w, N=xp.full(2, N_WAVE, dtype=xp.int32), **kwargs)
        ll_row = ecc.get_ll(PARAMS, data, w, N=xp.full(2 * HARMONICS, N_WAVE, dtype=xp.int32), **kwargs)
        np.testing.assert_array_equal(ll_int, ll_bin)
        np.testing.assert_array_equal(ll_int, ll_row)

        # N=None: one N per harmonic from get_N on (amp alpha_n, f_n)
        circ = eccentric_to_circular_harmonics(*PARAMS.T, harmonics=HARMONICS, xp=np).reshape(9, -1)
        N_rows = np.clip(get_N(np.abs(circ[0]), circ[1], T, armlength=self.orbits.armlength), 32, 2048)
        ll_none = ecc.get_ll(PARAMS, data, w, N=None, **kwargs)
        ll_rows = ecc.get_ll(PARAMS, data, w, N=xp.asarray(N_rows.astype(np.int32)), **kwargs)
        np.testing.assert_array_equal(ll_none, ll_rows)

        with self.assertRaises(ValueError):
            ecc.get_ll(PARAMS, data, w, N=100, **kwargs)
        with self.assertRaises(ValueError):
            ecc.get_ll(PARAMS, data, w, N=xp.full(3, N_WAVE, dtype=xp.int32), **kwargs)


class CrossTermTest(EccentricTestBase):
    """Overlapping harmonic bands (very low frequency, long N)."""

    def test_cross_terms_are_exact(self):
        ecc = self.make(harmonics=5)
        N = 2048
        sfi, dl = 0, int(3.2e-4 * T)
        p = np.array([[2e-22, 1e-4, 0.0, 0.0, 0.3, 0.8, 0.4, 1.2, 0.3, 0.35, 0.7, -2e-11]])
        p_add = p.copy()
        p_add[0, [1, 4, 9, 10]] = [1.0003e-4, 2.0, 0.2, -1.0]
        w = band_inv_psd(sfi, dl) * xp.ones((1, 2, 1))

        def signal(params):
            ecc.run_wave(*params.T, N=N, T=T, dt=dt)
            starts = to_np(ecc.start_inds_harm)[0]
            self.assertTrue(np.any(np.diff(starts) < N))  # bands do overlap
            return place(ecc.AETf_harm[0], starts, sfi, dl)

        h, h_add = signal(p), signal(p_add)
        ecc.d_d = inner(h, h, w[0]).real
        kwargs = dict(start_freq_ind=sfi, N=N, T=T, dt=dt)
        ll = ecc.get_ll(p, h[None], w, **kwargs)
        self.assertLess(abs(ll[0]), 1e-10 * ecc.d_d)
        ll_diag = ecc.get_ll(p, h[None], w, include_cross_terms=False, **kwargs)
        self.assertGreater(abs(ll_diag[0]), 1e-6 * ecc.d_d)

        data = 1.3 * h
        resid = data - h
        new = data - h_add
        ref = -0.5 * (inner(new, new, w[0]) - inner(resid, resid, w[0])).real
        ll_diff = ecc.swap_likelihood_difference(p, p_add, resid[None], w, **kwargs)
        self.assertAlmostEqual(ll_diff[0], ref, delta=1e-10 * abs(ref))

        templates = xp.zeros((1, 2, dl), dtype=complex)
        ecc.generate_global_template(p, xp.zeros(1, dtype=xp.int32), templates, **kwargs)
        self.assert_close(templates[0], h, rtol=1e-9, atol=1e-12 * float(xp.abs(h).max()))

    def test_per_harmonic_N_with_overlapping_bands(self):
        # harmonic bands overlap and each harmonic asks for a different N: the
        # binary is evaluated with its largest N everywhere
        ecc = self.make(harmonics=4)
        sfi, dl = 0, int(2.4e-4 * T)
        p = np.array([[2e-22, 7e-5, 0.0, 0.0, 0.3, 0.8, 0.4, 1.2, 0.3, 0.3, 0.7, -2e-12]])
        p_add = p.copy()
        p_add[0, [1, 4, 9]] = [7.0002e-5, 1.0, 0.25]
        N_rows = xp.asarray([512, 2048, 512, 256], dtype=xp.int32)
        w = band_inv_psd(sfi, dl) * xp.ones((1, 2, 1))
        kwargs = dict(start_freq_ind=sfi, T=T, dt=dt)

        def signal(params):
            ecc.run_wave(*params.T, N=2048, T=T, dt=dt)
            return place(ecc.AETf_harm[0], ecc.start_inds_harm[0], sfi, dl)

        h, h_add = signal(p), signal(p_add)
        templates = xp.zeros((1, 2, dl), dtype=complex)
        ecc.generate_global_template(p, xp.zeros(1, dtype=xp.int32), templates, N=N_rows, **kwargs)
        self.assert_close(templates[0], h, rtol=1e-9, atol=1e-12 * float(xp.abs(h).max()))

        ecc.d_d = inner(h, h, w[0]).real
        ll = ecc.get_ll(p, templates, w, N=N_rows, **kwargs)
        self.assertLess(abs(ll[0]), 1e-10 * ecc.d_d)

        resid = 0.4 * h
        new = resid + h - h_add
        ref = -0.5 * (inner(new, new, w[0]) - inner(resid, resid, w[0])).real
        ll_diff = ecc.swap_likelihood_difference(p, p_add, resid[None], w, N=N_rows, **kwargs)
        self.assertAlmostEqual(ll_diff[0], ref, delta=1e-10 * abs(ref))

    def test_overlap_prefilter_matches_brute_force(self):
        rng = np.random.default_rng(42)
        ecc = self.make(harmonics=[1, 2, 3, 5, 8])
        H = ecc.num_harmonics
        B = 300
        p_a = np.zeros((B, 12))
        p_a[:, 0] = 1e-22
        p_a[:, 1] = 10 ** rng.uniform(-5.3, -2.5, B)
        p_a[:, 9] = 0.2
        p_a[:, 11] = rng.uniform(-0.2, 0.2, B) * p_a[:, 1]
        p_b = p_a.copy()
        jump = rng.random(B) < 0.3
        p_b[:, 1] *= np.where(jump, rng.uniform(0.3, 3.0, B), 1 + rng.normal(0, 1e-6, B))
        p_b[:, 11] += np.where(jump, rng.uniform(-0.3, 0.3, B) * p_b[:, 1], 0.0)
        N_rows = xp.asarray(2 ** rng.integers(5, 12, B * H), dtype=xp.int32)
        rows_a = ecc._harmonic_rows(xp.asarray(p_a))[0]
        rows_b = ecc._harmonic_rows(xp.asarray(p_b))[0]

        def brute(f_a, f_b, pairs):
            f_a, f_b, N = to_np(f_a), to_np(f_b), to_np(N_rows).astype(np.int64)
            sa = np.rint(f_a * T).astype(np.int64) - N // 2
            sb = np.rint(f_b * T).astype(np.int64) - N // 2
            found = set()
            for b in range(B):
                for i, j in zip(*pairs):
                    ia, ib = b * H + i, b * H + j
                    if sb[ib] < sa[ia] + N[ia] and sa[ia] < sb[ib] + N[ib]:
                        found.add((ia, ib))
            return found

        cases = (
            (p_a, p_a, rows_a, rows_a, np.triu_indices(H, k=1)),
            (p_a, p_b, rows_a, rows_b, np.nonzero(~np.eye(H, dtype=bool))),
        )
        specs = [(xp.asarray(pa), xp.asarray(pb), ra[:, 1], rb[:, 1], tuple(xp.asarray(x) for x in pairs))
                 for pa, pb, ra, rb, pairs in cases]
        for max_candidates in (2**22, 50):  # one chunk / many chunks
            found = ecc._overlapping_pairs(specs, N_rows, T, max_candidates=max_candidates)
            for (_, _, ra, rb, pairs), (ia, ib) in zip(cases, found):
                expected = brute(ra[:, 1], rb[:, 1], pairs)
                self.assertGreater(len(expected), 0)
                self.assertEqual(set(zip(to_np(ia).tolist(), to_np(ib).tolist())), expected)


class InformationMatrixTest(EccentricTestBase):
    def setUp(self):
        self.data_length = int(8e-3 * T)
        w = band_inv_psd(0, self.data_length)
        self.psd = xp.stack([w, w]).flatten()
        f0 = (int(2e-3 * T) + 0.1) / T  # keep the heterodyne bins away from rounding edges
        self.p = np.array([[2e-22, f0, 1e-17, 0.0, 0.3, 0.8, 0.4, 1.2, 0.3, 0.25, 0.7, -2e-9]])
        self.eps = np.array([1e-25, 1e-12, 1e-20, 1e-28, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-12])
        self.kwargs = dict(N=N_WAVE, T=T, dt=dt, data_length=self.data_length)

    def test_analytic_relations(self):
        ecc = self.make()
        F = ecc.information_matrix(self.p, self.psd, eps=self.eps, **self.kwargs)[0]
        ecc.d_d = 0.0
        ecc.get_ll(self.p, xp.zeros((1, 2, self.data_length), dtype=complex),
                   self.psd.reshape(1, 2, self.data_length), N=N_WAVE, T=T, dt=dt)
        hh = to_np(ecc.h_h_harm.real)[0]
        n = ecc.harmonics.astype(float)
        coeffs = eccentric_amplitude_coefficients(ecc.harmonics, 10)
        e = self.p[0, 9]
        dlog_alpha = (np.polynomial.polynomial.polyval(e, (coeffs[:, 1:] * np.arange(1, 11)).T)
                      / np.polynomial.polynomial.polyval(e, coeffs.T))
        amp = self.p[0, 0]
        np.testing.assert_allclose(F[0, 0], hh.sum() / amp**2, rtol=1e-8)
        np.testing.assert_allclose(F[4, 4], hh.sum(), rtol=1e-8)
        np.testing.assert_allclose(F[10, 10], ((n - 2) ** 2 * hh).sum(), rtol=1e-8)
        np.testing.assert_allclose(abs(F[4, 10]), abs(((n - 2) * hh).sum()), rtol=1e-8)
        np.testing.assert_allclose(F[9, 9], (dlog_alpha**2 * hh).sum(), rtol=1e-8)
        np.testing.assert_allclose(F[0, 9], (dlog_alpha * hh).sum() / amp, rtol=1e-8)
        self.assertLess(abs(F[0, 4]), 1e-8 * np.sqrt(F[0, 0] * F[4, 4]))
        np.testing.assert_allclose(F, F.T, rtol=1e-12, atol=0.0)

    def test_circular_limit_and_flip(self):
        ecc = self.make()
        gb = GBGPU(orbits=self.orbits, force_backend=force_backend)
        p0 = self.p.copy()
        p0[0, 9] = 0.0
        angles = np.arange(4, 9)
        F_ecc = ecc.information_matrix(p0, self.psd, inds=angles, eps=1e-6, **self.kwargs)
        F_gb = gb.information_matrix(p0[:, :9], self.psd, inds=angles, eps=1e-6, **self.kwargs)
        np.testing.assert_allclose(F_ecc, F_gb, rtol=1e-7, atol=1e-9 * np.abs(F_gb).max())

        # flip_ref_phase: F_flip(p) = J F(p') J with (phi, Phi0) -> -(phi, Phi0)
        flipped = self.make(flip_ref_phase=True)
        inds = np.array([1, 4, 5, 9, 10])
        F_flip = flipped.information_matrix(self.p, self.psd, inds=inds, eps=self.eps[inds], **self.kwargs)[0]
        mirrored = self.p.copy()
        mirrored[0, [4, 10]] *= -1.0
        F_mirr = ecc.information_matrix(mirrored, self.psd, inds=inds, eps=self.eps[inds], **self.kwargs)[0]
        J = np.diag([1.0, -1.0, 1.0, 1.0, -1.0])
        np.testing.assert_allclose(F_flip, J @ F_mirr @ J, rtol=1e-7)

    def test_channel_setups(self):
        ecc = self.make(harmonics=3)
        w = band_inv_psd(0, self.data_length)
        inds = np.array([4, 9, 10])
        F_ae = ecc.information_matrix(self.p, xp.stack([w, w]).flatten(), inds=inds, eps=1e-6, **self.kwargs)
        F_aet = ecc.information_matrix(self.p, xp.stack([w, w, 0.0 * w]).flatten(), inds=inds, eps=1e-6,
                                       tdi_channel_setup="AET", **self.kwargs)
        np.testing.assert_allclose(F_ae, F_aet, rtol=1e-10)
        xyz = (xp.eye(3)[:, :, None] * w[None, None, :]).flatten()
        F_xyz = ecc.information_matrix(self.p, xyz, inds=inds, eps=1e-6, tdi_channel_setup="XYZ", **self.kwargs)
        self.assertEqual(F_xyz.shape, (1, 3, 3))
        self.assertTrue(np.all(np.linalg.eigvalsh(F_xyz[0]) > 0))


class InterfaceTest(EccentricTestBase):
    def test_errors(self):
        ecc = self.make()
        w = band_inv_psd(START_FREQ_IND, DATA_LENGTH) * xp.ones((1, 2, 1))
        data = xp.zeros((1, 2, DATA_LENGTH), dtype=complex)
        kwargs = dict(start_freq_ind=START_FREQ_IND, N=N_WAVE, T=T, dt=dt)

        with self.assertRaises(ValueError):  # d_d not set
            ecc.get_ll(PARAMS, data, w, **kwargs)
        ecc.d_d = 0.0
        with self.assertRaisesRegex(ValueError, "harmonic n=1"):
            narrow = dict(kwargs, start_freq_ind=int(1.5e-3 * T))
            ecc.get_ll(PARAMS, data, w, **narrow)
        with self.assertRaisesRegex(ValueError, "generate_global_template: harmonic n="):
            ecc.generate_global_template(PARAMS, xp.zeros(2, dtype=xp.int32), data[:, :, : DATA_LENGTH // 2],
                                         **kwargs)
        bad_e = PARAMS.copy()
        bad_e[0, 9] = 1.0
        with self.assertRaises(ValueError):
            ecc.get_ll(bad_e, data, w, **kwargs)
        with self.assertRaises(ValueError):
            ecc.get_ll(PARAMS[:, :9], data, w, **kwargs)
        with self.assertRaises(NotImplementedError):
            ecc.get_ll(PARAMS, data, w, use_c_implementation=False, **kwargs)
        with self.assertRaises(NotImplementedError):
            ecc.get_fstat_ll(PARAMS, data, w)
        with self.assertRaises(NotImplementedError):
            ecc.information_matrix(PARAMS, w.flatten(), use_c_implementation=True, data_length=DATA_LENGTH)
        with self.assertRaises(ValueError):
            ecc.generate_global_template(PARAMS, xp.zeros(3, dtype=xp.int32), data, **kwargs)

    def test_physical_relations(self):
        self.assertEqual(float(eccentricity_enhancement(0.0)), 1.0)
        f, Mc = 3e-3, 0.8
        Mc_s = Mc * 1.98840987e30 * 6.6743e-11 / 299792458.0**3
        circ = 96 / 5 * np.pi ** (8 / 3) * Mc_s ** (5 / 3) * f ** (11 / 3)
        np.testing.assert_allclose(get_fdot_eccentric(f, 0.0, Mc=Mc), circ, rtol=1e-4)
        m1, m2, e = 1.4, 0.6, 0.3
        fdot = get_fdot_eccentric(f, e, m1=m1, m2=m2)
        fPP = get_fPP(f, e, m1=m1, m2=m2)
        self.assertGreater(fPP, 0.0)
        out = get_masses_from_eccentric_parameters(f, fdot, fPP, e)
        np.testing.assert_allclose(out[:2], (m1, m2), rtol=1e-8)
        self.assertTrue(np.isnan(get_masses_from_eccentric_parameters(f, fdot, -fPP, e)[0]))


def newtonian_quadrupole(amp, iota, phi0, e, ell):
    """Exact Newtonian quadrupole (h_+, h_x) over the mean anomaly ``ell``, GBGPU convention.

    Units ``G M = a = 1`` (mean motion 1), periastron at ``ell = 0`` (``Phi0 = 0``),
    argument of periastron ``omega = phi0 / 2``. GBGPU's sign of ``h_x`` corresponds to the
    eGB-multi expressions evaluated at ``pi - iota``.
    """
    u = ell.copy()
    for _ in range(60):
        u -= (u - e * np.sin(u) - ell) / (1.0 - e * np.cos(u))
    r = 1.0 - e * np.cos(u)
    rdot = e * np.sin(u) / r
    phidot = np.sqrt(1.0 - e * e) / r**2
    v = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(u / 2), np.sqrt(1.0 - e) * np.cos(u / 2))
    phi = 0.5 * phi0 + v
    a_term = 1.0 / r + r**2 * phidot**2 - rdot**2
    b_term = 2.0 * r * rdot * phidot
    ci, si2 = np.cos(np.pi - iota), np.sin(iota) ** 2
    s = 0.5 * amp  # A = 2 s G M / a
    hp = -s * ((1 + ci**2) * (a_term * np.cos(2 * phi) + b_term * np.sin(2 * phi))
               + si2 * (1.0 / r - r**2 * phidot**2 - rdot**2))
    hc = -2.0 * s * ci * (a_term * np.sin(2 * phi) - b_term * np.cos(2 * phi))
    return hp, hc


def rows_polarizations(rows, scale, ell):
    """Sum of circular rows h_+ = -(1 + cos^2 i) A cos(Psi), h_x = 2 cos i A sin(Psi), Psi = f ell + phi0.

    ``rows[:, 1]`` is the row frequency in units of the orbital frequency."""
    rows, scale = to_np(rows), to_np(scale)
    psi = rows[:, 1:2] * ell[None, :] + rows[:, 4:5]
    ci = np.cos(rows[:, 5:6])
    hp = (-(1 + ci**2) * scale[:, None] * np.cos(psi)).sum(0)
    hc = (2 * ci * scale[:, None] * np.sin(psi)).sum(0)
    return hp, hc


class CorrectionFamiliesTest(EccentricTestBase):
    """``breathing`` and ``counter_rotating``: the full Newtonian quadrupole as extra circular rows."""

    FLAGS = dict(breathing=True, counter_rotating=True)

    def placed(self, gb, params):
        """Python-path waveform of every binary (all rows) placed in the data band."""
        gb.run_wave(*params.T, N=N_WAVE, T=T, dt=dt, use_c_implementation=False)
        wave, start = gb.AETf_harm, gb.start_inds_harm
        return [place(wave[b], start[b], START_FREQ_IND, DATA_LENGTH) for b in range(params.shape[0])]

    def test_correction_amplitudes(self):
        n = np.arange(1, 9)
        for e in (0.05, 0.3, 0.7):
            bessel, beta = eccentric_correction_amplitudes(np.array([e]), n, xp=np, num_points=1024)
            np.testing.assert_allclose(bessel[0], jv(n, n * e), rtol=1e-10, atol=1e-14)
            bessel_m, beta_m = eccentric_correction_amplitudes(np.array([-e]), n, xp=np, num_points=1024)
            np.testing.assert_allclose(bessel_m[0], (-1.0) ** n * bessel[0], rtol=1e-10, atol=1e-14)
            np.testing.assert_allclose(beta_m[0], (-1.0) ** n * beta[0], rtol=1e-8, atol=1e-14)
        # beta_n = O(e^{n+2}): halving e divides it by ~2^{n+2}
        _, b1 = eccentric_correction_amplitudes(np.array([0.04]), n[:4], xp=np, num_points=1024)
        _, b2 = eccentric_correction_amplitudes(np.array([0.02]), n[:4], xp=np, num_points=1024)
        np.testing.assert_allclose(np.log2(b1[0] / b2[0]), n[:4] + 2.0, atol=0.02)

    def test_rows_reproduce_the_newtonian_quadrupole(self):
        ell = np.linspace(0.0, 2 * np.pi, 4096, endpoint=False)
        amp, iota, phi0 = 1.0, 1.1, 0.9
        for e in (0.1, 0.4, 0.6):
            p = np.array([[amp, 2.0, 0.0, 0.0, phi0, iota, 0.3, 1.0, 0.2, e, 0.0, 0.0]])
            hp, hc = newtonian_quadrupole(amp, iota, phi0, e, ell)
            norm = np.sqrt(np.sum(hp**2 + hc**2))
            res = {}
            for flags in (dict(), dict(breathing=True), self.FLAGS):
                ecc = self.make(harmonics=100, ecc_order=None, exact_num_points=1024, **flags)
                rows, scale, _ = ecc._harmonic_rows(xp.asarray(p))
                mp, mc = rows_polarizations(rows, scale, ell)
                res[len(flags)] = np.sqrt(np.sum((hp - mp) ** 2 + (hc - mc) ** 2)) / norm
            self.assertLess(res[2], 1e-11)         # full quadrupole, to round-off
            self.assertGreater(res[0], 1e3 * res[2])  # the Moore model misses the breathing term
            self.assertLess(res[1], res[0])

    def test_rows_layout_and_flags_off(self):
        p = PARAMS[:1]
        base = self.make()
        rows0, s0, a0 = base._harmonic_rows(xp.asarray(p))
        ecc = self.make(**self.FLAGS)
        rows, s, a = ecc._harmonic_rows(xp.asarray(p))
        H = HARMONICS
        self.assertEqual(ecc.num_components, 3 * H)
        self.assertEqual(list(ecc.families), ["main", "breathing", "counter"])
        self.assert_close(rows[:H], rows0, rtol=0)
        self.assert_close(s[:H], s0, rtol=0)
        n = np.arange(1, H + 1)
        amp, f0, fpp, phi, iota, Phi0, e = p[0, [0, 1, 11, 4, 5, 10, 9]]
        self.assert_close(rows[H:2 * H, 1], 0.5 * n * f0 + n * fpp, rtol=1e-14)
        self.assert_close(rows[2 * H:, 1], 0.5 * n * f0 + (n + 2) * fpp, rtol=1e-14)
        self.assert_close(rows[H:2 * H, 4], n * Phi0 + np.pi, rtol=1e-14)
        self.assert_close(rows[2 * H:, 4], -phi + (n + 2) * Phi0, rtol=1e-14)
        self.assert_close(rows[H:2 * H, 5], np.full(H, np.pi / 2), rtol=0)
        self.assert_close(rows[2 * H:, 5], np.full(H, np.pi - iota), rtol=1e-15)
        self.assert_close(s[H:2 * H], amp * np.sin(iota) ** 2 * jv(n, n * e), rtol=1e-10)
        # explicit False is the default model
        off = self.make(breathing=False, counter_rotating=False)
        off.run_wave(*p.T, N=N_WAVE, T=T, dt=dt)
        base.run_wave(*p.T, N=N_WAVE, T=T, dt=dt)
        self.assert_close(off.AETf, base.AETf, rtol=0)

    def test_likelihood_swap_and_templates(self):
        ecc = self.make(**self.FLAGS)
        h = self.placed(ecc, PARAMS)
        w = band_inv_psd(START_FREQ_IND, DATA_LENGTH) * xp.ones((1, 2, 1))
        data = 1.2 * (h[0] + h[1]) + 0.3 * xp.roll(h[0], 2, axis=-1)
        ecc.d_d = inner(data, data, w[0]).real
        ll = ecc.get_ll(PARAMS, data[None], w, start_freq_ind=START_FREQ_IND, N=N_WAVE, T=T, dt=dt)
        ref = [-0.5 * inner(data - hb, data - hb, w[0]).real for hb in h]
        np.testing.assert_allclose(ll, ref, rtol=1e-9)
        # without the cross terms between the rows of one harmonic the answer is wrong
        ll_diag = ecc.get_ll(PARAMS, data[None], w, start_freq_ind=START_FREQ_IND, N=N_WAVE, T=T, dt=dt,
                             include_cross_terms=False)
        self.assertGreater(np.max(np.abs(ll_diag - ref)), 1e-6 * ecc.d_d)

        p_add = PARAMS.copy()
        p_add[:, [1, 9, 10]] = [[PARAMS[0, 1] * (1 + 3e-7), 0.3, 0.5], [PARAMS[1, 1], 0.2, 1.0]]
        h_add = self.placed(ecc, p_add)
        resid = data - h[0] - h[1]
        ref_swap = [-0.5 * (inner(resid + h[b] - h_add[b], resid + h[b] - h_add[b], w[0])
                            - inner(resid, resid, w[0])).real for b in range(2)]
        ll_diff = ecc.swap_likelihood_difference(PARAMS, p_add, resid[None], w, start_freq_ind=START_FREQ_IND,
                                                 N=N_WAVE, T=T, dt=dt)
        np.testing.assert_allclose(ll_diff, ref_swap, rtol=1e-8)

        for use_c in (True, False):
            templates = xp.zeros((1, 2, DATA_LENGTH), dtype=complex)
            ecc.generate_global_template(PARAMS, xp.zeros(2, dtype=xp.int32), templates,
                                         start_freq_ind=START_FREQ_IND, N=N_WAVE, T=T, dt=dt,
                                         use_c_implementation=use_c)
            self.assert_close(templates[0], h[0] + h[1], rtol=1e-9, atol=1e-12 * float(xp.abs(h[0]).max()))

        with self.assertRaises(ValueError):
            ecc.get_ll(PARAMS, data[None], w, start_freq_ind=START_FREQ_IND, N=N_WAVE, T=T, dt=dt,
                       phase_maximize=True)

    def test_negative_eccentricity_continuation(self):
        ecc = self.make(**self.FLAGS)
        p = PARAMS[:1].copy()
        p_neg = p.copy()
        p_neg[0, 9] *= -1.0
        p_neg[0, 10] += np.pi
        a = ecc.inject_signal(p, DATA_LENGTH, start_freq_ind=START_FREQ_IND, N=N_WAVE, T=T, dt=dt)
        b = ecc.inject_signal(p_neg, DATA_LENGTH, start_freq_ind=START_FREQ_IND, N=N_WAVE, T=T, dt=dt)
        self.assert_close(a, b, rtol=1e-8, atol=1e-10 * float(xp.abs(a).max()))

    def test_multisource_likelihood(self):
        from gbgpu.eccentric_moves import MultiSourceLikelihood, gbgpu_to_eccentric

        ecc = self.make(**self.FLAGS)
        h = self.placed(ecc, PARAMS)
        w = band_inv_psd(START_FREQ_IND, DATA_LENGTH) * xp.ones((2, 1))
        data = 0.9 * (h[0] + h[1])
        ecc.d_d = inner(data, data, w).real
        ll = ecc.get_ll(PARAMS, data[None], w[None], start_freq_ind=START_FREQ_IND, N=N_WAVE, T=T, dt=dt)
        gb = GBGPU(orbits=self.orbits, force_backend=force_backend)
        like = MultiSourceLikelihood(gb, data, 1.0 / w, START_FREQ_IND, T, dt, N_WAVE, harmonics=HARMONICS,
                                     **self.FLAGS)
        ll_msl = like(gbgpu_to_eccentric(PARAMS, T), np.arange(2))
        np.testing.assert_allclose(ll_msl, ll, rtol=1e-8, atol=1e-8 * ecc.d_d)

    def test_information_matrix(self):
        ecc = self.make(harmonics=4, **self.FLAGS)
        data_length = int(8e-3 * T)
        w = band_inv_psd(0, data_length)
        f0 = (int(2e-3 * T) + 0.1) / T
        p = np.array([[2e-22, f0, 1e-17, 0.0, 0.3, 0.8, 0.4, 1.2, 0.3, 0.4, 0.7, -2e-9]])
        inds = np.array([5, 9, 10, 11])
        eps = np.array([1e-6, 1e-6, 1e-6, 1e-12])
        F = ecc.information_matrix(p, xp.stack([w, w]).flatten(), inds=inds, eps=eps, N=N_WAVE, T=T, dt=dt,
                                   data_length=data_length)[0]
        ww = xp.stack([w, w])

        def wave(q):
            return ecc.inject_signal(q, data_length, start_freq_ind=0, N=N_WAVE, T=T, dt=dt)

        d = []
        for i, step in zip(inds, eps):
            qp, qm = p.copy(), p.copy()
            qp[0, i] += step
            qm[0, i] -= step
            d.append((wave(qp) - wave(qm)) / (2 * step))
        F_ref = np.array([[inner(a, b, ww).real for b in d] for a in d])
        np.testing.assert_allclose(F, F_ref, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
