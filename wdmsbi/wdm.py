"""Noise, gaps and the noise-scaled WDM image that the network sees.

Pipeline for one observation (all on the GPU during training):

    clean TDI-A waveform h  --(+ Gaussian noise n, x gap mask m)-->  d = m (h + n)
    d  --WDM transform, rows in [1, 5) mHz-->  w  (82 x 768)
    w  --divide each row by its noise level sigma_f-->  u = w / sigma_f
"""
import numpy as np

from . import config


def fourier_noise_std(psd, n_time=config.N_SAMPLES, dt=config.DT):
    """Standard deviation of the real and imaginary rfft noise coefficients.

    `psd` is the one-sided PSD at np.fft.rfftfreq(n_time, dt)[1:]. The square
    root is taken in float64 before casting to float32; the DC term is zero.
    """
    psd = np.asarray(psd, dtype=np.float64)
    if psd.shape != (n_time // 2,) or np.any(psd <= 0):
        raise ValueError("Expected a positive PSD at the n_time // 2 nonzero rfft frequencies.")
    return np.insert(np.sqrt(n_time * psd / (4.0 * dt)), 0, 0.0).astype(np.float32)


def lisa_noise_std(n_time=config.N_SAMPLES, dt=config.DT):
    """Fourier noise scales for the stationary TDI-A PSD of lisatools."""
    from lisatools.sensitivity import A1TDISens, get_sensitivity

    frequencies = np.fft.rfftfreq(n_time, dt)[1:]
    return fourier_noise_std(get_sensitivity(frequencies, sens_fn=A1TDISens, return_type="PSD"), n_time, dt)


def add_noise_and_gaps(clean, masks, noise_std, key):
    """d = m (h + n) for a batch, with n drawn in the Fourier domain."""
    import jax
    import jax.numpy as jnp

    clean = jnp.asarray(clean, dtype=jnp.float32)
    masks = jnp.asarray(masks, dtype=jnp.float32)
    noise_std = jnp.asarray(noise_std, dtype=jnp.float32)
    batch, n = clean.shape
    key_real, key_imag = jax.random.split(key)
    real = jax.random.normal(key_real, (batch, n // 2 + 1), dtype=jnp.float32)
    imag = jax.random.normal(key_imag, (batch, n // 2 + 1), dtype=jnp.float32)
    noise_f = ((real + 1j * imag) * noise_std).at[:, 0].set(0.0)
    return (clean + jnp.fft.irfft(noise_f, n=n)) * masks


def mask_from_indices(gap_indices, n_time=config.N_SAMPLES):
    """Dense 0/1 mask from the sparse indices of the missing samples."""
    mask = np.ones(n_time, dtype=np.float32)
    indices = np.asarray(gap_indices, dtype=np.int64)
    mask[indices[(indices >= 0) & (indices < n_time)]] = 0.0
    return mask


def band_rows(nf=config.NF, dt=config.DT, band=config.BAND_HZ):
    """First and one-past-last WDM row whose central frequency lies in the band."""
    centres = np.arange(nf) / (2.0 * nf * dt)
    return int(np.searchsorted(centres, band[0])), int(np.searchsorted(centres, band[1]))


def wdm_band(signals, dt=config.DT, nf=config.NF):
    """Real WDM coefficients of a batch (batch, time), band rows only: (batch, 82, NT)."""
    import jax
    import jax.numpy as jnp
    from pywavelet.transforms.jax.forward.from_time import transform_wavelet_time_helper
    from pywavelet.transforms.phi_computer import phi_vec

    signals = jnp.asarray(signals, dtype=jnp.float32)
    nt = signals.shape[1] // nf
    low, high = band_rows(nf, dt)
    phi = jnp.asarray(phi_vec(nf, d=4.0, q=32), dtype=jnp.float32)

    def one(signal):
        wave = transform_wavelet_time_helper(signal, Nf=nf, Nt=nt, phi=phi, mult=32)
        return (wave * jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float32))).real[low:high, :]

    return jax.lax.map(one, signals)


def network_input(signals, noise_scales):
    """Noise-scaled WDM images, shape (batch, 1, 82, NT)."""
    import jax.numpy as jnp

    scales = jnp.asarray(noise_scales, dtype=jnp.float32)
    return (wdm_band(signals) / scales[None, :, None])[:, None, :, :]


def calibrate_noise_scales(gap_library, noise_std, n_time=config.N_SAMPLES,
                           draws=config.NOISE_CALIBRATION_DRAWS,
                           seed=config.NOISE_CALIBRATION_SEED, batch_size=4):
    """Row-wise standard deviation of noise-only, gapped WDM coefficients.

    Uses independent noise realizations and masks drawn from the gap library,
    and no waveform. Moments are accumulated in float64. Computed once before
    training and then held fixed.
    """
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    total = np.zeros(config.N_ROWS)
    total_sq = np.zeros(config.N_ROWS)
    count = 0
    for start in range(0, draws, batch_size):
        size = min(batch_size, draws - start)
        masks = np.stack([mask_from_indices(gap_library[rng.integers(len(gap_library))], n_time)
                          for _ in range(size)])
        key, draw_key = jax.random.split(key)
        noise = add_noise_and_gaps(jnp.zeros((size, n_time), dtype=jnp.float32), masks, noise_std, draw_key)
        w = np.asarray(wdm_band(noise), dtype=np.float64)
        total += w.sum(axis=(0, 2))
        total_sq += np.square(w).sum(axis=(0, 2))
        count += w.shape[0] * w.shape[2]
    variance = total_sq / count - np.square(total / count)
    return np.sqrt(variance).astype(np.float32)
