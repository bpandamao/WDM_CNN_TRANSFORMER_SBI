"""Generate a library of gap masks with lisaglitch and store the indices of the missing samples.

    python scripts/generate_gap_library.py --seed 42 --output data/gap_library.h5

Gap model (Sec. II of the paper): scheduled gaps of about 100 s three times a day,
of 3.3 h every two weeks and of 40 h six times a year; unscheduled gaps of 24 h,
60 h and 66 h, on average 30, 3 and 4 times a year. Each mask gets its own pair
of seeds (scheduled, unscheduled) drawn from --seed.
"""
import argparse
import json

import h5py
import numpy as np

# The settings of the training library (key names and order as used to generate it)
GAP_CONFIG = {
    "planned": {
        "Long Duration:": {"rate_per_year": 6, "duration_hr": 40},
        "Medium Duration": {"rate_per_year": 26, "duration_hr": 3.3},
        "Short Duration": {"rate_per_year": 1095, "duration_hr": 0.028},   # about 100 s
    },
    "unplanned": {
        "Platform interruptions": {"rate_per_year": 3, "duration_hr": 60},
        "Payload interruptions": {"rate_per_year": 4, "duration_hr": 66},
        "Environmental Interruptions": {"rate_per_year": 30, "duration_hr": 24},
    },
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patterns", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    from lisaglitch import GapMaskGenerator

    from wdmsbi import config

    times = np.arange(config.N_SAMPLES, dtype=np.float64) * config.DT
    seeds = np.random.default_rng(args.seed).integers(0, np.iinfo(np.int32).max, size=(args.patterns, 2),
                                                      dtype=np.int64)
    with h5py.File(args.output, "x") as f:
        library = f.create_dataset("sparse_gap_indices", shape=(args.patterns,), dtype=h5py.vlen_dtype(np.int64))
        missing = np.empty(args.patterns)
        for i, (planned, unplanned) in enumerate(seeds):
            mask = np.asarray(GapMaskGenerator(sim_t=times, gap_definitions=GAP_CONFIG, treat_as_nan=False,
                                               planseed=int(planned), unplanseed=int(unplanned)).generate_mask())
            gaps = np.flatnonzero((mask == 0) | ~np.isfinite(mask)).astype(np.int64)
            library[i] = gaps
            missing[i] = gaps.size / config.N_SAMPLES
        f.create_dataset("planned_seeds", data=seeds[:, 0])
        f.create_dataset("unplanned_seeds", data=seeds[:, 1])
        f.attrs.update(gap_config_json=json.dumps(GAP_CONFIG, sort_keys=True), seed=args.seed,
                       dt_seconds=config.DT, number_of_samples=config.N_SAMPLES)
    print(f"{args.patterns} masks; missing fraction {100 * missing.min():.1f}-{100 * missing.max():.1f}%, "
          f"median {100 * np.median(missing):.1f}%")


if __name__ == "__main__":
    main()
