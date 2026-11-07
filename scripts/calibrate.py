"""Calibration on new observations: the credible level at which the truth enters the posterior.

    python scripts/calibrate.py --injections data/calibration.h5 --gaps data/calibration_gaps.h5 \
        --runs runs/transformer runs/token_pooling runs/pooled_cnn --seed 20261028 --output calibration.npz

For each stored waveform, a mask is drawn from --gaps and new noise is added; the
same observation and the same Gaussian draws are given to every network. The
levels are uniform for a calibrated posterior: printed are the fraction of
truths inside the 90% region and the Kolmogorov-Smirnov p-value of uniformity,
for each parameter alone, for (ln f0, ln fdot0) and for all three jointly.
"""
import argparse
from pathlib import Path

import numpy as np

SPACES = {"log10A": (0,), "ln f0": (1,), "ln fdot0": (2,), "(ln f0, ln fdot0)": (1, 2), "all three": (0, 1, 2)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--injections", required=True, help="HDF5 from generate_waveforms.py")
    p.add_argument("--gaps", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="training output folders")
    p.add_argument("--seed", type=int, default=20261028)
    p.add_argument("--draws", type=int, default=4000)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    import h5py
    import jax
    import jax.numpy as jnp
    from scipy import stats

    from wdmsbi import config, flow
    from wdmsbi.calibration import joint_level, marginal_levels
    from wdmsbi.data import load_gap_library
    from wdmsbi.networks import load
    from wdmsbi.wdm import add_noise_and_gaps, lisa_noise_std, mask_from_indices

    config.configure_jax()
    runs = [Path(r) for r in args.runs]
    models = [load(r / "best.eqx", summarizer_of(r)) for r in runs]
    scales = jnp.asarray(np.load(runs[0] / "noise_scales.npy"))
    normalization = np.load(runs[0] / "normalization.npz")
    mean, std = normalization["param_mean"], normalization["param_std"]
    library = load_gap_library(args.gaps)
    noise_std = lisa_noise_std()

    with h5py.File(args.injections, "r") as f:
        truths = f["parameters"][:]
        n = len(truths)
        mask_choice = np.random.default_rng(args.seed).integers(0, len(library), size=n)
        levels = np.full((len(models), n, len(SPACES)), np.nan)
        for i in range(n):
            mask = mask_from_indices(library[mask_choice[i]])
            observed = add_noise_and_gaps(f["time_signals"][i][None], mask[None], noise_std,
                                          jax.random.fold_in(jax.random.PRNGKey(args.seed), i))[0]
            source_key = jax.random.fold_in(jax.random.PRNGKey(args.seed + 1), i)   # same draws for every network
            truth = (truths[i] - mean) / std
            for m, model in enumerate(models):
                samples = flow.sample(model, observed, scales, source_key, n=args.draws)
                levels[m, i] = [marginal_levels(samples, truth)[d[0]] if len(d) == 1
                                else joint_level(samples, truth, d) for d in SPACES.values()]
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{n} observations", flush=True)

    np.savez(args.output, levels=levels, spaces=list(SPACES), runs=[str(r) for r in runs], truths=truths,
             mask_choice=mask_choice, seed=args.seed, draws=args.draws)
    print(f"\n{'run':28s}" + "".join(f"{s:>20s}" for s in SPACES) + "   (inside 90% / KS p)")
    for m, run in enumerate(runs):
        cells = [f"{100 * np.mean(levels[m, :, s] <= 0.9):.0f}% / {stats.kstest(levels[m, :, s], 'uniform').pvalue:.3f}"
                 for s in range(len(SPACES))]
        print(f"{run.name:28s}" + "".join(f"{c:>20s}" for c in cells))


def summarizer_of(run):
    return (run / "summarizer.txt").read_text().strip()


if __name__ == "__main__":
    main()
