"""Draw parameters from the prior and store their clean TDI-A waveforms (GPU, fastlisaresponse).

Training set (24,000 training + 6,000 validation waveforms):
    python scripts/generate_waveforms.py --n 30000 --seed 1 --output data/waveforms.h5
"""
import argparse
import time

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    from wdmsbi import config
    from wdmsbi.waveform import sample_prior, tdi_a_generator

    parameters = sample_prior(np.random.default_rng(args.seed), args.n)
    generate = tdi_a_generator()
    with h5py.File(args.output, "x") as f:
        f.attrs.update(prior_lower=config.PRIOR_LOWER, prior_upper=config.PRIOR_UPPER, seed=args.seed,
                       parameter_names=list(config.PARAMETER_NAMES), dt_seconds=config.DT)
        f.create_dataset("parameters", data=parameters)
        signals = f.create_dataset("time_signals", shape=(args.n, config.N_SAMPLES), dtype="float32",
                                   chunks=(1, config.N_SAMPLES), compression="gzip", compression_opts=4)
        started = time.perf_counter()
        for i, theta in enumerate(parameters):
            signals[i] = generate(theta)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{args.n} waveforms, {(time.perf_counter() - started) / (i + 1):.2f} s each", flush=True)


if __name__ == "__main__":
    main()
