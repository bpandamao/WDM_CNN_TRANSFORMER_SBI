"""Flow-matching posterior samples for one observation.

    python scripts/sample_posterior.py --run runs/transformer --observation observation.npz --output posterior.npz

The observation file holds the gapped series (`observed`, zeros in the gaps);
see simulate_observation.py. Samples are returned in (log10 A, ln f0, ln fdot0).
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="training output folder")
    p.add_argument("--observation", required=True)
    p.add_argument("--draws", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    import jax

    from wdmsbi import config, flow
    from wdmsbi.networks import load

    config.configure_jax()
    run = Path(args.run)
    model = load(run / "best.eqx", (run / "summarizer.txt").read_text().strip())
    normalization = np.load(run / "normalization.npz")
    obs = np.load(args.observation)
    theta = flow.sample(model, obs["observed"], np.load(run / "noise_scales.npy"), jax.random.PRNGKey(args.seed),
                        n=args.draws)
    samples = flow.to_physical(theta, normalization["param_mean"], normalization["param_std"])
    np.savez(args.output, samples=samples, seed=args.seed)
    print("posterior mean:", samples.mean(0), " sd:", samples.std(0))
    if "truth" in obs:
        print("truth:", obs["truth"])


if __name__ == "__main__":
    main()
