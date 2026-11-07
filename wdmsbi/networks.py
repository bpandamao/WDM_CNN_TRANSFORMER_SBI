"""Summarizers and the flow-matching velocity network (JAX / Equinox).

FlowMatcher(theta_t, t, image) returns the velocity v(theta_t, t, c) with the
context c = summarizer(image). Three summarizers are available:

  "transformer"    convolutional stem -> one token per time step -> class token
                   + positional embeddings -> 4 Transformer blocks; the context
                   combines the output class token with the mean of the tokens
                   (the model of the paper)
  "token_pooling"  the same stem and tokens, mean and max pooling instead of
                   the Transformer blocks
  "pooled_cnn"     five strided convolutions and global average pooling, no tokens

The velocity network is the same for all three: residual blocks conditioned
on (c, t) by adaptive layer normalization. Field order and initialization
follow the code used for the paper, so its checkpoints load directly.
"""
import math

import equinox as eqx
import jax
import jax.numpy as jnp

from . import config


class ConvStem(eqx.Module):
    """Three convolutions with kernels elongated along time; time is downsampled
    by 8 and every frequency row is kept. Each time step becomes one token."""
    conv1: eqx.nn.Conv2d
    gn1: eqx.nn.GroupNorm
    conv2: eqx.nn.Conv2d
    gn2: eqx.nn.GroupNorm
    conv3: eqx.nn.Conv2d
    gn3: eqx.nn.GroupNorm
    proj: eqx.nn.Linear
    ln: eqx.nn.LayerNorm

    def __init__(self, image_shape, width, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.conv1 = eqx.nn.Conv2d(1, 16, kernel_size=(7, 15), stride=(1, 2), padding=(3, 7), key=k1)
        self.gn1 = eqx.nn.GroupNorm(4, 16)
        self.conv2 = eqx.nn.Conv2d(16, 32, kernel_size=(7, 11), stride=(1, 2), padding=(3, 5), key=k2)
        self.gn2 = eqx.nn.GroupNorm(8, 32)
        self.conv3 = eqx.nn.Conv2d(32, 64, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3), key=k3)
        self.gn3 = eqx.nn.GroupNorm(16, 64)
        channels, rows, _ = self.features(jnp.zeros(image_shape)).shape
        self.proj = eqx.nn.Linear(channels * rows, width, key=k4)   # the whole band -> one token
        self.ln = eqx.nn.LayerNorm(width)

    def features(self, x):
        x = jax.nn.gelu(self.gn1(self.conv1(x)))
        x = jax.nn.gelu(self.gn2(self.conv2(x)))
        return jax.nn.gelu(self.gn3(self.conv3(x)))

    def __call__(self, x):
        x = self.features(x)                                  # (channels, rows, tokens)
        channels, rows, tokens = x.shape
        x = jnp.transpose(x.reshape((channels * rows, tokens)))
        return jax.vmap(self.ln)(jax.vmap(self.proj)(x))      # (tokens, width)


class TransformerBlock(eqx.Module):
    """Pre-norm self-attention and feed-forward block."""
    mha: eqx.nn.MultiheadAttention
    mlp: eqx.nn.MLP
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm

    def __init__(self, width, heads, key):
        k1, k2 = jax.random.split(key, 2)
        self.mha = eqx.nn.MultiheadAttention(heads, width, key=k1)
        self.mlp = eqx.nn.MLP(width, width, width * 4, 1, activation=jax.nn.gelu, key=k2)
        self.ln1 = eqx.nn.LayerNorm(width)
        self.ln2 = eqx.nn.LayerNorm(width)

    def __call__(self, x):
        y = jax.vmap(self.ln1)(x)
        x = x + self.mha(y, y, y)
        return x + jax.vmap(self.mlp)(jax.vmap(self.ln2)(x))


class TransformerSummarizer(eqx.Module):
    """Stem tokens + class token + positional embeddings -> Transformer blocks;
    context from the output class token and the mean of the other tokens."""
    patch_embed: ConvStem
    cls_token: jnp.ndarray
    pos_embed: jnp.ndarray
    blocks: list
    ln_final: eqx.nn.LayerNorm
    context_proj: eqx.nn.Linear

    def __init__(self, image_shape, width=config.WIDTH, heads=config.N_HEADS,
                 n_blocks=config.N_TRANSFORMER_BLOCKS, key=None):
        keys = jax.random.split(key, n_blocks + 4)
        self.patch_embed = ConvStem(image_shape, width, key=keys[0])
        n_tokens = self.patch_embed(jnp.zeros(image_shape)).shape[0]
        self.cls_token = jax.random.normal(keys[1], (1, width)) * 0.02
        self.pos_embed = jax.random.normal(keys[2], (n_tokens + 1, width)) * 0.02
        self.blocks = [TransformerBlock(width, heads, key=keys[i + 3]) for i in range(n_blocks)]
        self.ln_final = eqx.nn.LayerNorm(width)
        self.context_proj = eqx.nn.Linear(2 * width, width, key=keys[-1])

    def __call__(self, image):
        x = jnp.concatenate([self.cls_token, self.patch_embed(image)], axis=0) + self.pos_embed
        for block in self.blocks:
            x = eqx.filter_checkpoint(block)(x)               # recompute in the backward pass to save memory
        x = jax.vmap(self.ln_final)(x)
        pooled = jnp.concatenate([x[0], jnp.mean(x[1:], axis=0)], axis=-1)
        return jax.nn.gelu(self.context_proj(pooled))


class TokenPoolingSummarizer(eqx.Module):
    """The same stem tokens, mean and max pooled; no attention."""
    patch_embed: ConvStem
    ln_final: eqx.nn.LayerNorm
    context_proj: eqx.nn.Linear

    def __init__(self, image_shape, width=config.WIDTH, key=None):
        keys = jax.random.split(key, 8)                       # same keys as the Transformer summarizer
        self.patch_embed = ConvStem(image_shape, width, key=keys[0])
        self.ln_final = eqx.nn.LayerNorm(width)
        self.context_proj = eqx.nn.Linear(2 * width, width, key=keys[-1])

    def __call__(self, image):
        x = jax.vmap(self.ln_final)(self.patch_embed(image))
        pooled = jnp.concatenate([jnp.mean(x, axis=0), jnp.max(x, axis=0)], axis=-1)
        return jax.nn.gelu(self.context_proj(pooled))


class PooledCNNSummarizer(eqx.Module):
    """Five strided convolutions and global average pooling; no tokens."""
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    conv4: eqx.nn.Conv2d
    conv5: eqx.nn.Conv2d
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm
    norm3: eqx.nn.GroupNorm
    norm4: eqx.nn.GroupNorm
    norm5: eqx.nn.GroupNorm
    context_proj: eqx.nn.Linear

    def __init__(self, image_shape, width=config.WIDTH, key=None):
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        self.conv1 = eqx.nn.Conv2d(1, 16, (7, 15), stride=(2, 2), padding=(3, 7), key=k1)
        self.conv2 = eqx.nn.Conv2d(16, 32, (7, 11), stride=(2, 2), padding=(3, 5), key=k2)
        self.conv3 = eqx.nn.Conv2d(32, 64, (5, 7), stride=(2, 2), padding=(2, 3), key=k3)
        self.conv4 = eqx.nn.Conv2d(64, 128, (3, 5), stride=(2, 2), padding=(1, 2), key=k4)
        self.conv5 = eqx.nn.Conv2d(128, 256, (3, 3), stride=(2, 2), padding=(1, 1), key=k5)
        self.norm1 = eqx.nn.GroupNorm(4, 16)
        self.norm2 = eqx.nn.GroupNorm(8, 32)
        self.norm3 = eqx.nn.GroupNorm(16, 64)
        self.norm4 = eqx.nn.GroupNorm(32, 128)
        self.norm5 = eqx.nn.GroupNorm(64, 256)
        self.context_proj = eqx.nn.Linear(256, width, key=k6)

    def __call__(self, image):
        x = jax.nn.gelu(self.norm1(self.conv1(image)))
        x = jax.nn.gelu(self.norm2(self.conv2(x)))
        x = jax.nn.gelu(self.norm3(self.conv3(x)))
        x = jax.nn.gelu(self.norm4(self.conv4(x)))
        x = jax.nn.gelu(self.norm5(self.conv5(x)))
        return jax.nn.gelu(self.context_proj(jnp.mean(x, axis=(1, 2))))


class AdaLNResBlock(eqx.Module):
    """Residual block whose two layer norms are scaled and shifted by the condition."""
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    film: eqx.nn.Linear

    def __init__(self, width, cond_width, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.linear1 = eqx.nn.Linear(width, width, key=k1)
        self.linear2 = eqx.nn.Linear(width, width, key=k2)
        self.ln1 = eqx.nn.LayerNorm(width)
        self.ln2 = eqx.nn.LayerNorm(width)
        self.film = eqx.nn.Linear(cond_width, 4 * width, key=k3)

    def __call__(self, x, cond):
        gamma1, beta1, gamma2, beta2 = self.film(cond).reshape(4, -1)
        h = jax.nn.silu(self.linear1(self.ln1(x) * (1.0 + gamma1) + beta1))
        h = self.linear2(jax.nn.silu(self.ln2(h) * (1.0 + gamma2) + beta2))
        return x + h


def time_embedding(t, half=128):
    """Sinusoidal embedding of the flow time t (shape (1,)) -> (256,)."""
    freqs = jnp.exp(-math.log(10000.0) * jnp.arange(0, half, dtype=jnp.float32) / half)
    args = t * freqs
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)


SUMMARIZER_CLASSES = {"transformer": TransformerSummarizer,
                      "token_pooling": TokenPoolingSummarizer,
                      "pooled_cnn": PooledCNNSummarizer}


class FlowMatcher(eqx.Module):
    """Summarizer + velocity network v(theta_t, t, c), trained together."""
    encoder: eqx.Module
    time_mlp: eqx.nn.MLP
    param_in: eqx.nn.Linear
    cond_fuse: eqx.nn.Linear
    res_blocks: list
    vf_out: eqx.nn.Linear

    def __init__(self, summarizer="transformer", image_shape=(1, config.N_ROWS, config.NT),
                 n_params=3, width=config.WIDTH, n_blocks=config.N_RESIDUAL_BLOCKS, key=None):
        keys = jax.random.split(key, n_blocks + 5)
        self.encoder = SUMMARIZER_CLASSES[summarizer](image_shape, width=width, key=keys[0])
        self.time_mlp = eqx.nn.MLP(in_size=256, out_size=width, width_size=width, depth=2,
                                   activation=jax.nn.silu, key=keys[1])
        self.param_in = eqx.nn.Linear(n_params, width, key=keys[2])
        self.cond_fuse = eqx.nn.Linear(2 * width, width, key=keys[3])
        self.res_blocks = [AdaLNResBlock(width, width, key=keys[i + 4]) for i in range(n_blocks)]
        self.vf_out = eqx.nn.Linear(width, n_params, key=keys[-1])

    def context(self, image):
        """Summary of one image (1, 82, NT); computed once per observation."""
        return self.encoder(image)

    def velocity(self, theta_t, t, context):
        cond = jax.nn.silu(self.cond_fuse(jnp.concatenate([context, self.time_mlp(time_embedding(t))], axis=-1)))
        h = jax.nn.silu(self.param_in(theta_t))
        for block in self.res_blocks:
            h = block(h, cond)
        return self.vf_out(h)

    def __call__(self, theta_t, t, image):
        return self.velocity(theta_t, t, self.context(image))


def float32(model):
    return jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_inexact_array(x) else x, model)


def load(path, summarizer="transformer"):
    """Load trained weights saved with eqx.tree_serialise_leaves."""
    template = float32(FlowMatcher(summarizer, key=jax.random.PRNGKey(0)))
    return float32(eqx.tree_deserialise_leaves(path, template))
