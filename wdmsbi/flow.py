"""Posterior samples from a trained FlowMatcher.

Draw theta_0 ~ N(0, I) and integrate d theta_t / dt = v(theta_t, t, c) from
t = 0 to 1 with fixed-step RK4; the context c is computed once per observation.
Parameters are standardized inside the network; `to_physical` converts samples
to (log10 A, ln f0, ln fdot0).
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from . import config
from .wdm import network_input


def rk4(derivative, state, steps):
    """Integrate d state / dt = derivative(state, t) from t = 0 to 1."""
    dt = 1.0 / steps

    def step(i, s):
        t = jnp.asarray(i * dt, dtype=jnp.float32)
        k1 = derivative(s, t)
        k2 = derivative(jax.tree_util.tree_map(lambda a, b: a + 0.5 * dt * b, s, k1), t + 0.5 * dt)
        k3 = derivative(jax.tree_util.tree_map(lambda a, b: a + 0.5 * dt * b, s, k2), t + 0.5 * dt)
        k4 = derivative(jax.tree_util.tree_map(lambda a, b: a + dt * b, s, k3), t + dt)
        return jax.tree_util.tree_map(lambda a, b1, b2, b3, b4: a + dt / 6.0 * (b1 + 2 * b2 + 2 * b3 + b4),
                                      s, k1, k2, k3, k4)

    return jax.lax.fori_loop(0, steps, step, state)


@eqx.filter_jit
def context_of(model, signal, noise_scales):
    """Context of one gapped time series (N_SAMPLES,)."""
    return model.context(network_input(signal[None, :], noise_scales)[0])


@eqx.filter_jit
def transport(model, context, theta0, steps):
    """Push standard-normal draws (n, 3) to posterior draws (standardized)."""
    def derivative(states, t):
        return jax.vmap(lambda theta: model.velocity(theta, t, context))(states)
    return rk4(derivative, theta0, steps)


def batched(function, model, context, theta0, steps, batch_size):
    """Run in fixed-size batches (padding the last) so the ODE compiles once."""
    outputs = []
    for begin in range(0, theta0.shape[0], batch_size):
        rows = theta0[begin:begin + batch_size]
        count = rows.shape[0]
        rows = jnp.pad(rows, ((0, batch_size - count), (0, 0)))
        result = jax.block_until_ready(function(model, context, rows, steps))
        outputs.append(jax.tree_util.tree_map(lambda a: np.asarray(a[:count]), result))
    return jax.tree_util.tree_map(lambda *a: np.concatenate(a), *outputs)


def sample(model, signal, noise_scales, key, n=4000, steps=config.ODE_STEPS, batch_size=128):
    """n standardized posterior draws for one gapped observation."""
    context = context_of(model, jnp.asarray(signal, dtype=jnp.float32), noise_scales)
    theta0 = jax.random.normal(key, (n, 3), dtype=jnp.float32)
    return batched(transport, model, context, theta0, steps, min(batch_size, n))


def to_physical(theta, param_mean, param_std):
    """Undo the standardization of the parameters."""
    return np.asarray(theta, dtype=np.float64) * param_std + param_mean
