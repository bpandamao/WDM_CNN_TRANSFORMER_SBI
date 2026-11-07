"""Train the summarizer and the flow-matching velocity network together.

    python scripts/train.py --waveforms data/waveforms.h5 --gaps data/gap_library.h5 \
        --summarizer transformer --output runs/transformer

The noise scales of the WDM rows are estimated once at the start and saved
(noise_scales.npy). The comparison summarizers reuse the Transformer run's
scales:  --summarizer token_pooling --noise-scales runs/transformer/noise_scales.npy

Every training step draws new noise, takes a random mask from the library for
each waveform, and computes the noise-scaled WDM images on the GPU. The
checkpoint with the lowest validation loss (fixed noise and masks) is kept.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--waveforms", required=True, help="HDF5 file from generate_waveforms.py")
    p.add_argument("--gaps", required=True, help="HDF5 gap library from generate_gap_library.py")
    p.add_argument("--summarizer", default="transformer", choices=["transformer", "token_pooling", "pooled_cnn"])
    p.add_argument("--output", required=True)
    p.add_argument("--noise-scales", help="reuse saved noise scales instead of estimating them")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--microbatches", type=int, default=8, help="gradient accumulation to fit GPU memory")
    p.add_argument("--lr-initial", type=float, default=1e-5)
    p.add_argument("--lr-peak", type=float, default=1e-4)
    p.add_argument("--lr-end", type=float, default=5e-7)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()
    if not 0 <= args.warmup_epochs < args.epochs or args.batch_size % args.microbatches:
        p.error("Need 0 <= warmup-epochs < epochs and a batch size divisible by microbatches.")

    import equinox as eqx
    import h5py
    import jax
    import jax.numpy as jnp
    import optax
    import torch
    from torch.utils.data import DataLoader, Subset

    from wdmsbi import config
    from wdmsbi.data import FixedMaskDataset, WaveformDataset, load_gap_library, train_validation_split
    from wdmsbi.networks import FlowMatcher, float32
    from wdmsbi.wdm import add_noise_and_gaps, calibrate_noise_scales, lisa_noise_std, network_input

    config.configure_jax()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    gap_library = load_gap_library(args.gaps)
    with h5py.File(args.waveforms, "r") as f:
        train_ids, val_ids = train_validation_split(len(f["parameters"]))
    dataset = WaveformDataset(args.waveforms, gap_library, train_ids)
    noise_std = jnp.asarray(lisa_noise_std())
    if args.noise_scales:
        noise_scales = np.load(args.noise_scales)
    else:
        noise_scales = calibrate_noise_scales(gap_library, noise_std)
    (output / "summarizer.txt").write_text(args.summarizer)
    np.save(output / "noise_scales.npy", noise_scales)
    np.savez(output / "normalization.npz", param_mean=dataset.param_mean, param_std=dataset.param_std)
    noise_scales = jnp.asarray(noise_scales)

    model = float32(FlowMatcher(args.summarizer, key=jax.random.PRNGKey(args.seed)))
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))
    print(f"{args.summarizer}: {n_params:,} trainable parameters", flush=True)

    updates_per_epoch = -(-len(train_ids) // args.batch_size)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=args.lr_initial, peak_value=args.lr_peak, warmup_steps=args.warmup_epochs * updates_per_epoch,
        decay_steps=args.epochs * updates_per_epoch, end_value=args.lr_end)
    optimizer = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(schedule))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    train_loader = DataLoader(Subset(dataset, train_ids.tolist()), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(FixedMaskDataset(dataset, val_ids), batch_size=8, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    def loss_per_parameter(m, theta1, clean, masks, key):
        """Conditional flow-matching loss on the linear path from N(0, I)."""
        noise_key, time_key, source_key = jax.random.split(key, 3)
        images = network_input(add_noise_and_gaps(clean, masks, noise_std, noise_key), noise_scales)
        theta0 = jax.random.normal(source_key, theta1.shape, dtype=jnp.float32)
        t = jax.random.uniform(time_key, (theta1.shape[0], 1), dtype=jnp.float32)
        theta_t = (1 - t) * theta0 + t * theta1
        return jnp.mean((jax.vmap(m)(theta_t, t, images) - (theta1 - theta0)) ** 2, axis=0)

    @eqx.filter_value_and_grad
    def objective(m, theta1, clean, masks, key):
        return jnp.mean(loss_per_parameter(m, theta1, clean, masks, key))

    @eqx.filter_jit
    def step(m, state, theta1, clean, masks, key):
        k = args.microbatches
        chunks = (theta1.reshape(k, -1, 3), clean.reshape(k, -1, clean.shape[-1]),
                  masks.reshape(k, -1, masks.shape[-1]), jax.random.split(key, k))
        zeros = jax.tree_util.tree_map(jnp.zeros_like, eqx.filter(m, eqx.is_inexact_array))

        def accumulate(carry, chunk):
            loss, grads = objective(m, *chunk)
            return (carry[0] + loss, jax.tree_util.tree_map(jnp.add, carry[1], grads)), None

        (loss, grads), _ = jax.lax.scan(accumulate, (jnp.float32(0.0), zeros), chunks)
        grads = jax.tree_util.tree_map(lambda g: g / k, grads)
        updates, state = optimizer.update(grads, state, eqx.filter(m, eqx.is_inexact_array))
        return eqx.apply_updates(m, updates), state, loss / k

    evaluate = eqx.filter_jit(loss_per_parameter)

    def validate(m):
        total, count = np.zeros(3), 0
        for i, (theta1, clean, masks) in enumerate(val_loader):
            key = jax.random.fold_in(jax.random.PRNGKey(config.VALIDATION_JAX_SEED), i)
            loss = np.asarray(evaluate(m, *(jnp.asarray(x.numpy()) for x in (theta1, clean, masks)), key))
            total += loss * len(theta1)
            count += len(theta1)
        return total / count

    best = float(validate(model).mean())
    best_epoch, history = 0, []
    eqx.tree_serialise_leaves(output / "best.eqx", model)
    key = jax.random.PRNGKey(args.seed)
    for epoch in range(1, args.epochs + 1):
        started, total, count = time.perf_counter(), 0.0, 0
        for theta1, clean, masks in train_loader:
            key, batch_key = jax.random.split(key)
            model, opt_state, loss = step(model, opt_state, *(jnp.asarray(x.numpy()) for x in (theta1, clean, masks)),
                                          batch_key)
            total += float(loss) * len(theta1)
            count += len(theta1)
        record = {"epoch": epoch, "train": total / count, "seconds": time.perf_counter() - started}
        if epoch % 5 == 0 or epoch >= 200 or epoch == args.epochs:     # sparse validation early on
            val = validate(model)
            record["validation"] = val.tolist()
            if val.mean() < best:
                best, best_epoch = float(val.mean()), epoch
                eqx.tree_serialise_leaves(output / "best.eqx", model)
        history.append(record)
        (output / "history.json").write_text(json.dumps(history, indent=1))
        print(f"epoch {epoch}: train {record['train']:.5f}, validation {record.get('validation', '-')}, "
              f"best {best:.5f} at {best_epoch}, {record['seconds']:.0f} s", flush=True)
    eqx.tree_serialise_leaves(output / "final.eqx", model)
    print(f"Best validation loss {best:.5f} at epoch {best_epoch}; weights in {output / 'best.eqx'}")


if __name__ == "__main__":
    main()
