"""Galactic-binary waveform and its LISA TDI-A response (fastlisaresponse, GPU)."""
import numpy as np

from . import config

YRSID_SI = 31558149.763545603


class GBWave:
    """Plus and cross polarizations of a slowly chirping galactic binary, h+ + i hx."""

    def __init__(self, use_gpu=True):
        if use_gpu:
            import cupy
            self.xp = cupy
        else:
            self.xp = np

    def __call__(self, A, f, fdot, iota, phi0, psi, T=1.0, dt=config.DT):
        xp = self.xp
        t = xp.arange(0.0, T * YRSID_SI, dt)
        cos2psi, sin2psi, cosiota = xp.cos(2.0 * psi), xp.sin(2.0 * psi), xp.cos(iota)
        fddot = 11.0 / 3.0 * fdot ** 2 / f
        phase = 2 * np.pi * (f * t + 1.0 / 2.0 * fdot * t ** 2 + 1.0 / 6.0 * fddot * t ** 3) - phi0
        h_plus = -xp.cos(phase) * A * (1.0 + cosiota * cosiota)
        h_cross = -xp.sin(phase) * 2.0 * A * cosiota
        return (h_plus * cos2psi - h_cross * sin2psi) + 1j * (h_plus * sin2psi + h_cross * cos2psi)


def tdi_a_generator(use_gpu=True, dtype=np.float32):
    """Returns a function (log10 A, ln f0, ln fdot0) -> TDI-A time series of N_SAMPLES samples.

    float32 for the stored training waveforms.
    """
    from fastlisaresponse import ResponseWrapper
    from lisatools.detector import EqualArmlengthOrbits

    response = ResponseWrapper(
        GBWave(use_gpu=use_gpu), config.RESPONSE_YEARS, config.DT,
        6, 7,                                   # positions of lam and beta in the arguments
        t0=config.RESPONSE_T0, flip_hx=False, use_gpu=use_gpu,
        remove_sky_coords=True, is_ecliptic_latitude=True, remove_garbage=True,
        orbits=EqualArmlengthOrbits(use_gpu=use_gpu),
        order=config.RESPONSE_ORDER, tdi=config.TDI_GENERATION, tdi_chan="AET")

    def generate(parameters):
        log10_amplitude, log_f0, log_fdot0 = parameters
        channels = response(10.0 ** log10_amplitude, np.exp(log_f0), np.exp(log_fdot0),
                            config.IOTA, config.PHI0, config.PSI, config.LAM, config.BETA)
        channel_a = channels[0].get() if hasattr(channels[0], "get") else np.asarray(channels[0])
        return np.asarray(channel_a[:config.N_SAMPLES], dtype=dtype)

    return generate


def sample_prior(rng, n=None):
    """Draws from the uniform prior in (log10 A, ln f0, ln fdot0): shape (3,) or (n, 3)."""
    lower, upper = np.asarray(config.PRIOR_LOWER), np.asarray(config.PRIOR_UPPER)
    return lower + rng.uniform(size=3 if n is None else (n, 3)) * (upper - lower)
