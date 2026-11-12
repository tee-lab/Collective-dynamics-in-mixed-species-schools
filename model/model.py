"""
Mixed-species schooling model: readable version (numpy). model_optimized.py is the same model with a numba kernel.

Fish i (species s_i, parameters of its species) with neighbours j in J_i (n_i = |J_i| neighbours):

    dv_i/dt   = mu0 (v0 - v_i)
                + mu_sm / n_i  sum_j w^sm_ij (v_j cos(phi_j - phi_i) - v_i)          speed matching
                + mu_at / n_i  sum_j w^at_ij (r_ij - r0) rhat_ij . e_i               attraction (along heading)
                + sqrt(2 D_v) eta_v
    dphi_i/dt = 1 / (v_i + v_c) [ mu_al / n_i  sum_j w^al_ij v_j sin(phi_j - phi_i)    alignment
                + mu_at / n_i  sum_j w^at_ij (r_ij - r0) rhat_ij . e_i^perp          attraction (turning)
                + sqrt(2 D_phi) eta_phi ]
    dx_i/dt   = v_i e_i,  e_i = (cos phi_i, sin phi_i);  v_i >= 0 (reflecting)

r_ij: distance from i to j, rhat_ij: unit vector from i to j. Weights w^c_ij = 1 for fish of the same species; across
species w^c_AB (A focal, B neighbour) and w^c_BA (B focal, A neighbour) for c = sm, al, at. v_c is the smaller of the
two species' v_c. Neighbours: Voronoi (Delaunay) neighbours, recomputed every `voronoi_every` steps, or mean field
(all other fish). Euler-Maruyama with time step dt. With N_B = 0 this is the single-species model.

Units: BL (body lengths) and s. v0 BL/s; mu0, mu_sm, mu_al 1/s; mu_at 1/s^2; r0 BL; D_phi, D_v BL^2/s^3; v_c BL/s.

Usage
    import model
    pA, pB, w = model.load_pair("TB_RB")
    out = model.simulate(pA, pB, w, N_A=8, N_B=8, T=10**5, dt=0.001, save_every=40, seed=1)
    out = model.simulate(model.load_species("TB"), None, None, N_A=16, N_B=0, T=10**5, dt=0.001)   # single species
"""
import math
import tomllib
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.spatial import Delaunay, QhullError

HERE = Path(__file__).parent
KEYS = ("v0", "mu0", "mu_sm", "mu_al", "mu_at", "r0", "D_phi", "D_v", "v_c")
INTERACTIONS = ("sm", "al", "at")                 # speed matching, alignment, attraction


# ----------------------------------------------------------------------------- parameters
def load_species(sp, path=HERE / "singlesp-params.toml"):
    """Parameter dict of one species (v0, mu0, mu_sm, mu_al, mu_at, r0, D_phi, D_v, v_c)."""
    with open(path, "rb") as f:
        return tomllib.load(f)[sp]


def load_pair(pair, sp_path=HERE / "singlesp-params.toml", mix_path=HERE / "mixsp-params.toml"):
    """(pA, pB, w) for a species pair, e.g. "TB_RB"; w: dict of cross-species weights w_<c>_AB, w_<c>_BA."""
    with open(mix_path, "rb") as f:
        m = tomllib.load(f)[pair]
    w = {f"w_{c}_{ab}": m[f"w_{c}_{ab}"] for c in INTERACTIONS for ab in ("AB", "BA")}
    return load_species(m["A"], sp_path), load_species(m["B"], sp_path), w


# ----------------------------------------------------------------------------- initial conditions
def initial_headings(N, m0, rng):
    """N headings with polarisation exactly m0: evenly spread over a fraction s of the circle, shuffled, randomly rotated."""
    spread = lambda s: 2 * np.pi * s * (np.arange(N) / N - 0.5)
    s = 0.0 if m0 >= 1 else 1.0 if m0 <= 0 else brentq(lambda s: abs(np.mean(np.exp(1j * spread(s)))) - m0, 0.0, 1.0)
    return rng.permutation(spread(s)) + rng.uniform(-np.pi, np.pi)


def initial_state(P, v_init, m0, rng):
    """Positions uniform in a disc of radius 0.5 mean(r0) sqrt(N) (species mixed), polarisation m0,
    all speeds v_init (default: each fish's v0)."""
    N = len(P["v0"])
    R0 = 0.5 * P["r0"].mean() * math.sqrt(N)
    rad, ang = R0 * np.sqrt(rng.random(N)), 2 * np.pi * rng.random(N)
    x, y = rad * np.cos(ang), rad * np.sin(ang)
    phi = initial_headings(N, m0, rng)
    v = P["v0"].copy() if v_init is None else np.full(N, float(v_init))
    return x, y, v, phi


# ----------------------------------------------------------------------------- neighbours
def neighbours(x, y, neighbourhood):
    """N x N boolean matrix A, A[i, j] = True if j is a neighbour of i."""
    N = len(x)
    if neighbourhood == "mean-field" or N < 3:
        return ~np.eye(N, dtype=bool)                 # all other fish
    pts = np.column_stack((x, y))
    try:
        tri = Delaunay(pts)
    except QhullError:                                # e.g. all fish on a line
        tri = Delaunay(pts, qhull_options="QJ")
    indptr, indices = tri.vertex_neighbor_vertices    # Voronoi neighbours = Delaunay edges
    A = np.zeros((N, N), dtype=bool)
    for i in range(N):
        A[i, indices[indptr[i]:indptr[i + 1]]] = True
    return A


# ----------------------------------------------------------------------------- dynamics
def forces(x, y, v, phi, A, W, P):
    """Deterministic parts F_v (dv/dt) and F_phi (turning term in brackets) for all fish.
    W[c]: N x N weight matrix of interaction c (W[c][i, j] = weight fish i gives fish j)."""
    n = np.maximum(A.sum(1), 1)                       # number of neighbours (sums are 0 if there are none)
    dphi = phi[None, :] - phi[:, None]                # [i, j] = phi_j - phi_i
    dx, dy = x[None, :] - x[:, None], y[None, :] - y[:, None]
    r = np.hypot(dx, dy)
    r_safe = np.where(r > 0, r, 1.0)                  # coincident fish: no attraction
    along = (dx * np.cos(phi)[:, None] + dy * np.sin(phi)[:, None]) / r_safe     # rhat_ij . e_i
    across = (-dx * np.sin(phi)[:, None] + dy * np.cos(phi)[:, None]) / r_safe   # rhat_ij . e_i^perp
    pull = np.where(r > 0, r - P["r0"][:, None], 0.0)                            # r_ij - r0

    sm = np.sum(A * W["sm"] * (v[None, :] * np.cos(dphi) - v[:, None]), axis=1)
    al = np.sum(A * W["al"] * v[None, :] * np.sin(dphi), axis=1)
    at_along = np.sum(A * W["at"] * pull * along, axis=1)
    at_across = np.sum(A * W["at"] * pull * across, axis=1)

    F_v = P["mu0"] * (P["v0"] - v) + (P["mu_sm"] * sm + P["mu_at"] * at_along) / n
    F_phi = (P["mu_al"] * al + P["mu_at"] * at_across) / n
    return F_v, F_phi


def step(x, y, v, phi, F_v, F_phi, noise, P, v_c, dt):
    """One Euler-Maruyama step (in place), all updates from the old state. noise: (2, N) standard normals."""
    x += v * np.cos(phi) * dt
    y += v * np.sin(phi) * dt
    phi += (F_phi * dt + np.sqrt(2 * P["D_phi"] * dt) * noise[1]) / (v + v_c)
    phi[:] = (phi + np.pi) % (2 * np.pi) - np.pi      # wrap to [-pi, pi)
    v += F_v * dt + np.sqrt(2 * P["D_v"] * dt) * noise[0]
    v[:] = np.abs(v)                                  # reflecting boundary at v = 0


def advance(x, y, v, phi, A, W, P, v_c, dt, noise, n0, save_every, k, out):
    """len(noise) steps with a fixed neighbour matrix A; step n0 + s + 1 is saved as frame k if it is a multiple of
    save_every. Returns the next frame index."""
    for s in range(len(noise)):
        F_v, F_phi = forces(x, y, v, phi, A, W, P)
        step(x, y, v, phi, F_v, F_phi, noise[s], P, v_c, dt)
        if (n0 + s + 1) % save_every == 0:
            for key, val in zip(("x", "y", "v", "phi"), (x, y, v, phi)):
                out[key][k] = val
            k += 1
    return k


# ----------------------------------------------------------------------------- driver
def simulate(pA, pB, w, N_A, N_B, T, dt, v_init=None, m0=0.5, save_every=1, voronoi_every=10,
             neighbourhood="voronoi", seed=None):
    """T steps of length dt (s). Fish 0..N_A-1 are species A, N_A..N_A+N_B-1 species B (pB, w unused if N_B = 0).
    v_init: initial speed of all fish (default: each fish's v0); m0: initial polarisation.
    neighbourhood: "voronoi" (recomputed every voronoi_every steps) or "mean-field".
    Returns dict: t (frames,) in s; x, y (BL), v (BL/s), phi (rad) (frames, N); species (N,) 0 = A, 1 = B."""
    return _simulate(advance, pA, pB, w, N_A, N_B, T, dt, v_init, m0, save_every, voronoi_every, neighbourhood, seed)


def _simulate(advance, pA, pB, w, N_A, N_B, T, dt, v_init, m0, save_every, voronoi_every, neighbourhood, seed):
    """simulate() with a given time-stepping function (shared by model.py and model_optimized.py)."""
    if neighbourhood not in ("voronoi", "mean-field"):
        raise ValueError("neighbourhood must be 'voronoi' or 'mean-field'")
    if N_B == 0:
        pB, w = pA, {f"w_{c}_{ab}": 1.0 for c in INTERACTIONS for ab in ("AB", "BA")}
    N = N_A + N_B
    species = np.r_[np.zeros(N_A, int), np.ones(N_B, int)]
    P = {key: np.where(species == 0, float(pA[key]), float(pB[key])) for key in KEYS}    # per-fish parameters
    v_c = min(pA["v_c"], pB["v_c"])
    W = {c: np.array([[1.0, w[f"w_{c}_AB"]], [w[f"w_{c}_BA"], 1.0]])[species[:, None], species[None, :]]
         for c in INTERACTIONS}                                                           # N x N weight matrices

    rng = np.random.default_rng(seed)
    x, y, v, phi = initial_state(P, v_init, m0, rng)
    n_frames = T // save_every + 1
    out = {key: np.empty((n_frames, N)) for key in ("x", "y", "v", "phi")}
    for key, val in zip(("x", "y", "v", "phi"), (x, y, v, phi)):
        out[key][0] = val

    chunk = voronoi_every if neighbourhood == "voronoi" else 1000     # steps per neighbour update / noise draw
    A = neighbours(x, y, neighbourhood)
    k, n = 1, 0
    while n < T:
        m = min(chunk, T - n)
        if neighbourhood == "voronoi":
            A = neighbours(x, y, neighbourhood)
        noise = rng.standard_normal((m, 2, N))                         # [step, (speed, heading), fish]
        k = advance(x, y, v, phi, A, W, P, v_c, dt, noise, n, save_every, k, out)
        n += m
    return dict(t=np.arange(n_frames) * save_every * dt, species=species, **out)
