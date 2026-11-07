"""Simulate one gapped, noisy observation to try a trained network on.

    python scripts/simulate_observation.py --gaps data/gap_library.h5 --seed 7 --output observation.npz
    python scripts/simulate_observation.py --gaps data/gap_library.h5 --parameters -20.82 -6.2146 -23.0259 \
        --output observation.npz

The parameters (log10 A, ln f0, ln fdot0) are drawn from the prior unless given. The
mask is taken from the library at random, or by --mask; the noise is stationary LISA
noise. Saved: the gapped series `observed` (zeros in the gaps), the mask and the
parameters `truth`.
"""
import argparse

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gaps", required=True, help="HDF5 gap library from generate_gap_library.py")
    p.add_argument("--parameters", type=float, nargs=3, metavar=("LOG10_A", "LN_F0", "LN_FDOT0"))
    p.add_argument("--mask", type=int, help="index of the mask in the library (default: random)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    import jax
    import jax.numpy as jnp

    from wdmsbi import config
    from wdmsbi.data import load_gap_library
    from wdmsbi.waveform import sample_prior, tdi_a_generator
    from wdmsbi.wdm import add_noise_and_gaps, lisa_noise_std, mask_from_indices

    rng = np.random.default_rng(args.seed)
    truth = np.asarray(args.parameters) if args.parameters else sample_prior(rng)
    if np.any(truth < config.PRIOR_LOWER) or np.any(truth > config.PRIOR_UPPER):
        p.error(f"Parameters outside the training prior {config.PRIOR_LOWER} - {config.PRIOR_UPPER}.")
    library = load_gap_library(args.gaps)
    mask_index = args.mask if args.mask is not None else int(rng.integers(len(library)))
    mask = mask_from_indices(library[mask_index])
    clean = tdi_a_generator()(truth)
    noisy = add_noise_and_gaps(clean[None], np.ones((1, config.N_SAMPLES), np.float32), lisa_noise_std(),
                               jax.random.PRNGKey(args.seed))[0]
    observed = noisy * jnp.asarray(mask)
    np.savez(args.output, observed=np.asarray(observed, dtype=np.float64), mask=mask.astype(np.float64),
             truth=truth, mask_index=mask_index, seed=args.seed)
    print(f"truth {truth}; mask {mask_index}, {100 * np.mean(mask == 0):.1f}% missing; saved {args.output}")


if __name__ == "__main__":
    main()
