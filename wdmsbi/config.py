"""Fixed settings of the paper: data, source, prior, WDM grid and network sizes."""
import math

# Data
DT = 5.0                          # sampling interval (s)
N_SAMPLES = 1_572_864             # NF * NT samples, about 91.02 days
NF, NT = 2048, 768                # WDM frequency rows and time bins
BAND_HZ = (1e-3, 5e-3)            # rows whose central frequency is in [1, 5) mHz
N_ROWS = 82                       # number of rows in the band

# Galactic binary: free parameters (log10 A, ln f0, ln fdot0), uniform prior
PARAMETER_NAMES = ("log10(A)", "ln(f0)", "ln(fdot0)")
PRIOR_CENTRE = (math.log10(1.5e-21), math.log(2e-3), math.log(1e-10))
PRIOR_HALF_WIDTH = (0.1, 1e-4, 1e-4)
PRIOR_LOWER = tuple(c - w for c, w in zip(PRIOR_CENTRE, PRIOR_HALF_WIDTH))
PRIOR_UPPER = tuple(c + w for c, w in zip(PRIOR_CENTRE, PRIOR_HALF_WIDTH))

# Fixed parameters (rad)
IOTA = 1.11820901                 # inclination
PHI0 = 4.91128699                 # initial phase
PSI = 2.3290324                   # polarization
BETA = 0.9805742971871619         # ecliptic latitude
LAM = 5.22979888                  # ecliptic longitude

# LISA response (fastlisaresponse)
RESPONSE_YEARS = 0.25
RESPONSE_T0 = 10_000.0
RESPONSE_ORDER = 25
TDI_GENERATION = "1st generation"

# Training data
N_WAVEFORMS = 30_000              # 24,000 training + 6,000 validation
TRAIN_FRACTION = 0.8
SPLIT_SEED = 42
VALIDATION_MASK_SEED = 11_042
VALIDATION_JAX_SEED = 21_042

# Noise scales of the WDM rows (Sec. II of the paper)
NOISE_CALIBRATION_DRAWS = 64
NOISE_CALIBRATION_SEED = 1042

# Networks
WIDTH = 512                       # token, context and velocity-network width
N_HEADS = 8
N_TRANSFORMER_BLOCKS = 4
N_RESIDUAL_BLOCKS = 4
SUMMARIZERS = ("transformer", "token_pooling", "pooled_cnn")

# Sampling
ODE_STEPS = 500                   # fixed-step RK4 from t = 0 to 1


def configure_jax():
    """Float32 networks with full-precision matrix products, as in training."""
    import jax
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", "highest")
