import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(0)

# ==========================================
# 1. TOY SIMULATOR: NOISE-SCALED WDM IMAGE WITH GAPS
# ==========================================
# A small stand-in for the WDM image of the paper: NF frequency rows, NT time bins,
# unit noise in every pixel, a track whose frequency drifts upward, and gaps that
# remove whole time bins (their coefficients are zero). Two parameters in [-1, 1]:
# theta[0] sets the start frequency (like ln f0), theta[1] the drift (like ln fdot0).
NF, NT = 24, 64
AMP, WIDTH = 0.8, 1.0
ROWS = torch.arange(NF, dtype=torch.float32)
COLS = torch.arange(NT, dtype=torch.float32) / (NT - 1)


def track(theta):
    """Mean image (B, NF, NT) of the signal."""
    centre = 6 + 3 * theta[:, :1] + (6 + 4 * theta[:, 1:2]) * COLS.to(theta.device)   # (B, NT)
    return AMP * torch.exp(-0.5 * ((ROWS.to(theta.device)[None, :, None] - centre[:, None, :]) / WIDTH) ** 2)


def random_masks(n):
    """1 to 3 gaps of 2 to 5 time bins each, at random positions (new for every sample)."""
    mask = torch.ones(n, NT)
    for i in range(n):
        for _ in range(np.random.randint(1, 4)):
            length = np.random.randint(2, 6)
            start = np.random.randint(0, NT - length + 1)
            mask[i, start:start + length] = 0
    return mask


def simulate(theta, mask):
    """Gapped, noise-scaled image: d = m * (h + n)."""
    noise = torch.randn(len(theta), NF, NT, device=theta.device)
    return (track(theta) + noise) * mask[:, None, :].to(theta.device)


def sample_prior(n):
    return 2 * torch.rand(n, 2) - 1


THETA_SCALE = 1 / math.sqrt(3)     # standard deviation of U(-1, 1); the flow works on theta / THETA_SCALE

# ==========================================
# 2. SUMMARIZER: CNN STEM + TRANSFORMER
# ==========================================
class Summarizer(nn.Module):
    def __init__(self, dim=64, depth=2, heads=4):
        super().__init__()
        # kernels elongated along time; only time is downsampled, so every frequency row is kept
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, (3, 7), padding=(1, 3)), nn.GELU(),
            nn.Conv2d(16, 32, (3, 7), stride=(1, 2), padding=(1, 3)), nn.GELU(),
            nn.Conv2d(32, 32, (3, 7), stride=(1, 2), padding=(1, 3)), nn.GELU(),
        )
        n_tokens = NT // 4
        self.to_token = nn.Linear(32 * NF, dim)           # the whole band at one time step -> one token
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(0.02 * torch.randn(1, n_tokens + 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(2 * dim, dim)

    def forward(self, x):
        """x: (B, NF, NT) image -> context c: (B, dim)."""
        h = self.stem(x[:, None])                         # (B, 32, NF, NT/4)
        h = h.permute(0, 3, 1, 2).flatten(2)              # (B, NT/4, 32*NF)
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), self.to_token(h)], 1) + self.pos
        tokens = self.norm(self.encoder(tokens))          # every token attends to every other one
        # context: class token together with the mean of the other tokens
        return self.out(torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], -1))

# ==========================================
# 3. VECTOR FIELD NETWORK v(theta_t, t, c)
# ==========================================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """Maps a scalar t to a vector of size 'dim' using sinusoidal functions."""
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class AdaLNBlock(nn.Module):
    """Residual block conditioned on (c, t) through adaptive layer normalization."""
    def __init__(self, width, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.film = nn.Linear(cond_dim, 2 * width)
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, h, cond):
        scale, shift = self.film(cond).chunk(2, dim=-1)
        return h + self.mlp(self.norm(h) * (1 + scale) + shift)


class VectorFieldNet(nn.Module):
    def __init__(self, n_params=2, width=128, cond_dim=64, depth=3):
        super().__init__()
        self.time_mlp = nn.Sequential(TimeEmbedding(cond_dim), nn.Linear(cond_dim, cond_dim), nn.SiLU())
        self.inp = nn.Linear(n_params, width)
        self.blocks = nn.ModuleList([AdaLNBlock(width, cond_dim) for _ in range(depth)])
        self.out = nn.Linear(width, n_params)

    def forward(self, theta_t, t, c):
        cond = c + self.time_mlp(t)
        h = self.inp(theta_t)
        for block in self.blocks:
            h = block(h, cond)
        return self.out(h)


summarizer = Summarizer().to(device)
model = VectorFieldNet().to(device)

# ==========================================
# 4. TRAINING: SUMMARIZER AND FLOW TOGETHER, NEW NOISE AND GAPS AT EVERY STEP
# ==========================================
batch_size = 256
steps = 4000
params = list(summarizer.parameters()) + list(model.parameters())
optimizer = optim.Adam(params, lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

CHECKPOINT = "fm_wdm_toy.pt"
if os.path.exists(CHECKPOINT):
    state = torch.load(CHECKPOINT, map_location=device)
    summarizer.load_state_dict(state["summarizer"])
    model.load_state_dict(state["model"])
    steps = -1
    print(f"Loaded trained networks from {CHECKPOINT} (delete it to retrain)")
else:
    print("Training summarizer + flow matching...")
for step in range(steps + 1):
    optimizer.zero_grad()

    # 1. Simulate: parameters from the prior, new noise and gaps
    theta = sample_prior(batch_size).to(device)
    x = simulate(theta, random_masks(batch_size))

    # 2. Probability path (linear interpolation) from N(0, I) to the standardized parameters
    theta1 = theta / THETA_SCALE
    theta0 = torch.randn_like(theta1)
    t = torch.rand(batch_size, 1, device=device)
    theta_t = (1 - t) * theta0 + t * theta1

    # 3. Loss: regress the velocity theta1 - theta0, conditioned on the context
    loss = torch.mean((model(theta_t, t, summarizer(x)) - (theta1 - theta0)) ** 2)
    loss.backward()
    nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    scheduler.step()

    if step % 500 == 0:
        print(f"Step {step:4d} | Loss: {loss.item():.4f}")
if steps >= 0:
    torch.save({"summarizer": summarizer.state_dict(), "model": model.state_dict()}, CHECKPOINT)

# ==========================================
# 5. TWO TEST OBSERVATIONS AND THEIR POSTERIORS
# ==========================================
def gaps_to_mask(gaps):
    mask = torch.ones(1, NT)
    for start, length in gaps:
        mask[0, start:start + length] = 0
    return mask


OBS = [dict(theta=[-0.25, 0.30], gaps=[(22, 3)]),
       dict(theta=[0.35, -0.35], gaps=[(2, 4), (29, 4), (58, 4)])]
torch.manual_seed(1)
for o in OBS:
    o['mask'] = gaps_to_mask(o['gaps'])
    o['x'] = simulate(torch.tensor([o['theta']], device=device), o['mask'])

n_particles, n_steps = 2000, 50
dt = 1.0 / n_steps
summarizer.eval()
model.eval()
with torch.no_grad():
    for o in OBS:
        c = summarizer(o['x'])                            # computed once per observation
        o['context'] = c[0].cpu().numpy()
        x_traj = torch.randn(n_particles, 2, device=device)
        path = [x_traj.cpu().numpy()]
        for i in range(n_steps):                          # fourth-order Runge-Kutta
            t = torch.full((n_particles, 1), i * dt, device=device)
            v = lambda th, tt: model(th, tt, c.expand(n_particles, -1))
            k1 = v(x_traj, t)
            k2 = v(x_traj + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = v(x_traj + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = v(x_traj + dt * k3, t + dt)
            x_traj = x_traj + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            path.append(x_traj.cpu().numpy())
        o['path'] = np.array(path)

        # exact posterior on a grid, for comparison (Gaussian likelihood, uniform prior)
        g = torch.linspace(-1, 1, 241)
        G = torch.stack(torch.meshgrid(g, g, indexing='xy'), -1).reshape(-1, 2).to(device)
        resid = (o['x'] - track(G) * o['mask'][:, None, :].to(device)) ** 2
        logp = -0.5 * resid.sum((1, 2))
        p = torch.exp(logp - logp.max()).reshape(241, 241).cpu().numpy()
        level = np.sort(p.ravel())[::-1]
        o['p'], o['level90'] = p, level[np.searchsorted(np.cumsum(level) / level.sum(), 0.9)]
        o['grid'] = g.numpy() / THETA_SCALE

        samples = o['path'][-1] * THETA_SCALE
        w = p.ravel() / p.sum()
        mean = w @ G.cpu().numpy()
        sd = np.sqrt(w @ (G.cpu().numpy() - mean) ** 2)
        print(f"theta = {o['theta']}: flow mean {samples.mean(0).round(3)}, sd {samples.std(0).round(3)} | "
              f"exact mean {mean.round(3)}, sd {sd.round(3)}")

# ==========================================
# 6. ANIMATION
# ==========================================
PARTICLE, ARROW, GAP, INK = '#b2182b', '#2c5f8a', '#c8c8c8', '#333333'
LIM = 3.2
fig = plt.figure(figsize=(12, 5.4))
header = fig.text(0.04, 0.93, '', fontsize=13, weight='bold', color=INK)

ax_img = fig.add_axes([0.04, 0.38, 0.40, 0.44])
ax_img.set_title('network input: gapped, noise-scaled WDM image (toy)', loc='left', fontsize=11, color=INK)
cmap = plt.get_cmap('RdBu_r').copy()
cmap.set_bad(GAP)
img = ax_img.imshow(np.zeros((NF, NT)), origin='lower', aspect='auto', cmap=cmap, vmin=-3, vmax=3,
                    interpolation='nearest')
ax_img.set_xlabel('time', fontsize=10, color=INK)
ax_img.set_ylabel('frequency', fontsize=10, color=INK)
ax_img.set_xticks([])
ax_img.set_yticks([])
track_line, = ax_img.plot([], [], ls='--', lw=1.2, color='k', label='signal track (buried in the noise)')
ax_img.legend(loc='upper left', fontsize=9, framealpha=0.9)

ax_sum = fig.add_axes([0.04, 0.03, 0.40, 0.22])
ax_sum.set_xlim(0, 1)
ax_sum.set_ylim(0, 1)
ax_sum.axis('off')
ax_sum.add_patch(FancyBboxPatch((0.02, 0.22), 0.38, 0.50, boxstyle='round,pad=0.01,rounding_size=0.04',
                                fc='#eef2f7', ec='#7a8ca3', lw=1.2, clip_on=False))
ax_sum.text(0.21, 0.47, 'summarizer\nCNN stem + Transformer', ha='center', va='center', fontsize=10.5, color=INK)
n_cells = 16
cell_w = 0.42 / n_cells
cells = [ax_sum.add_patch(Rectangle((0.54 + j * cell_w, 0.32), cell_w * 0.9, 0.30, ec='white', lw=0.5))
         for j in range(n_cells)]
ax_sum.text(0.54, 0.70, r'context $c$', fontsize=11, color=INK)
ax_sum.add_patch(FancyArrowPatch((0.41, 0.47), (0.53, 0.47), arrowstyle='-|>', mutation_scale=14, color=INK, lw=1.2))
fig.add_artist(FancyArrowPatch((0.12, 0.30), (0.12, 0.19), transform=fig.transFigure, arrowstyle='-|>',
                               mutation_scale=14, color=INK, lw=1.2))
fig.add_artist(FancyArrowPatch((0.445, 0.135), (0.545, 0.23), transform=fig.transFigure,
                               connectionstyle='arc3,rad=0.25', arrowstyle='-|>', mutation_scale=16, color=INK, lw=1.4))
fig.text(0.48, 0.115, 'condition', fontsize=10.5, color=INK, style='italic')

ax_fm = fig.add_axes([0.55, 0.12, 0.40, 0.74])
ax_fm.set_xlim(-LIM, LIM)
ax_fm.set_ylim(-LIM, LIM)
ax_fm.set_aspect('equal', adjustable='box')
ax_fm.set_xlabel(r'start frequency $\theta_1$ (like $\ln f_0$)', fontsize=11, color=INK)
ax_fm.set_ylabel(r'drift $\theta_2$ (like $\ln \dot f_0$)', fontsize=11, color=INK)
ax_fm.set_title(r'flow matching:  $\mathrm{d}\theta_t/\mathrm{d}t = v_\varphi(\theta_t, t, c)$', fontsize=12, color=INK)

grid = np.linspace(-LIM + 0.2, LIM - 0.2, 16)
GX, GY = np.meshgrid(grid, grid)
grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
quiv = ax_fm.quiver(grid_pts[:, 0], grid_pts[:, 1], np.zeros(len(grid_pts)), np.zeros(len(grid_pts)),
                    color=ARROW, alpha=0.6, angles='xy', scale_units='xy', scale=7, width=0.0035, headwidth=3.5)
dots = ax_fm.scatter([], [], s=3, color=PARTICLE, alpha=0.45, lw=0)
truth, = ax_fm.plot([], [], marker='*', ms=13, color='k', mec='white', mew=0.8, ls='none')
trace_ids = np.random.default_rng(0).choice(n_particles, 10, replace=False)
traces = [ax_fm.plot([], [], lw=0.8, color='#555555', alpha=0.8)[0] for _ in trace_ids]
clock = ax_fm.text(0.03, 0.96, '', transform=ax_fm.transAxes, fontsize=12, va='top', color=INK,
                   bbox=dict(fc='white', ec='#bbbbbb', boxstyle='round,pad=0.3'))
ax_fm.legend(handles=[Line2D([], [], color=ARROW, lw=1.5, label=r'learned velocity $v_\varphi$'),
                      Line2D([], [], color=PARTICLE, marker='o', ms=4, ls='none', label='flow samples'),
                      Line2D([], [], color='k', ls='--', lw=1, label='exact posterior (90%)'),
                      Line2D([], [], color='k', marker='*', ms=10, ls='none', label='truth')],
             loc='upper right', fontsize=9, frameon=True)

# precompute the learned field on the grid at every step
with torch.no_grad():
    G_t = torch.tensor(grid_pts, dtype=torch.float32, device=device)
    for o in OBS:
        c = torch.tensor(o['context'], device=device)[None].expand(len(G_t), -1)
        o['field'] = [model(G_t, torch.full((len(G_t), 1), i * dt, device=device), c).cpu().numpy()
                      for i in range(n_steps + 1)]

contour = [None]
shown = [None]


def show_observation(k):
    o = OBS[k]
    missing = 100 * (1 - o['mask'].mean().item())
    header.set_text(f'Observation {k + 1}:  {missing:.0f}% of the data missing')
    image = o['x'][0].cpu().numpy().copy()
    image[:, o['mask'][0].numpy() == 0] = np.nan
    img.set_data(image)
    centre = 6 + 3 * o['theta'][0] + (6 + 4 * o['theta'][1]) * COLS.numpy()
    centre[o['mask'][0].numpy() == 0] = np.nan
    track_line.set_data(np.arange(NT), centre)
    ctx = o['context'][:n_cells]
    for cell, value in zip(cells, (ctx - ctx.min()) / (np.ptp(ctx) + 1e-9)):
        cell.set_facecolor(plt.get_cmap('Blues')(0.25 + 0.7 * value))
    truth.set_data([o['theta'][0] / THETA_SCALE], [o['theta'][1] / THETA_SCALE])


def show_exact(o, on):
    """Dashed 90% region of the exact posterior, drawn at t = 1 only."""
    if on and contour[0] is None:
        contour[0] = ax_fm.contour(o['grid'], o['grid'], o['p'], levels=[o['level90']], colors='k',
                                   linestyles='--', linewidths=1.4, zorder=5)
    elif not on and contour[0] is not None:
        try:
            contour[0].remove()
        except AttributeError:          # matplotlib < 3.8
            for line in contour[0].collections:
                line.remove()
        contour[0] = None


def update(frame):
    k, i = frame
    o = OBS[k]
    if shown[0] != k:
        show_observation(k)
        shown[0] = k
    pts = o['path'][i]
    dots.set_offsets(pts)
    # show the field only where there are samples
    near = np.min(((grid_pts[:, None, :] - pts[None, ::10, :]) ** 2).sum(-1), axis=1) < 0.3 ** 2
    v = o['field'][i]
    quiv.set_UVC(np.ma.masked_where(~near, v[:, 0]), np.ma.masked_where(~near, v[:, 1]))
    for line, j in zip(traces, trace_ids):
        line.set_data(o['path'][:i + 1, j, 0], o['path'][:i + 1, j, 1])
    show_exact(o, i == n_steps)
    clock.set_text(f't = {i * dt:.2f}')
    return []


frames = []
for k in range(len(OBS)):
    frames += [(k, 0)] * 8 + [(k, i) for i in range(n_steps + 1)] + [(k, n_steps)] * 20

print("Saving GIF...")
ani = FuncAnimation(fig, update, frames=frames, interval=80)
ani.save("fm_wdm_gaps.gif", writer=PillowWriter(fps=12), dpi=80)
plt.close()
print("Success! Saved as 'fm_wdm_gaps.gif'")
