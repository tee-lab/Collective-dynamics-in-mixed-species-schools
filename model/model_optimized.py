"""
Mixed-species schooling model: numba version of model.py (same model, same arguments, same outputs to rounding error
for the same seed). Set-up, initial conditions, neighbours and noise come from model.py; only the time stepping is
compiled. See model.py for the equations.

Usage
    import model_optimized as mo
    pA, pB, w = mo.load_pair("TB_RB")
    out = mo.simulate(pA, pB, w, N_A=8, N_B=8, T=10**6, dt=0.001, save_every=40, seed=1)
"""
import math
import sys
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).parent))
import model
from model import load_pair, load_species  # noqa: F401  (re-exported)


@njit(cache=True)
def _kernel(x, y, v, phi, indptr, indices, W_sm, W_al, W_at, mu0, v0, mu_sm, mu_al, mu_at, r0, sq_v, sq_phi, v_c,
            dt, noise, n0, save_every, k, xs, ys, vs, phis):
    """Same steps as model.advance, with neighbours in CSR form (neighbours of i: indices[indptr[i]:indptr[i+1]])."""
    N = x.shape[0]
    F_v = np.empty(N); F_phi = np.empty(N)
    for s in range(noise.shape[0]):
        for i in range(N):                            # forces (model.forces)
            ci = math.cos(phi[i]); si = math.sin(phi[i])
            sm = 0.0; al = 0.0; at_along = 0.0; at_across = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                j = indices[p]
                dphi = phi[j] - phi[i]
                sm += W_sm[i, j] * (v[j] * math.cos(dphi) - v[i])
                al += W_al[i, j] * v[j] * math.sin(dphi)
                dx = x[j] - x[i]; dy = y[j] - y[i]; r = math.sqrt(dx * dx + dy * dy)
                if r > 0.0:
                    f = W_at[i, j] * (r - r0[i]) / r
                    at_along += f * (dx * ci + dy * si)
                    at_across += f * (-dx * si + dy * ci)
            n = max(indptr[i + 1] - indptr[i], 1)
            F_v[i] = mu0[i] * (v0[i] - v[i]) + (mu_sm[i] * sm + mu_at[i] * at_along) / n
            F_phi[i] = (mu_al[i] * al + mu_at[i] * at_across) / n
        for i in range(N):                            # Euler-Maruyama step (model.step)
            vi = v[i]
            x[i] += vi * math.cos(phi[i]) * dt
            y[i] += vi * math.sin(phi[i]) * dt
            ph = phi[i] + (F_phi[i] * dt + sq_phi[i] * noise[s, 1, i]) / (vi + v_c)
            phi[i] = (ph + math.pi) % (2.0 * math.pi) - math.pi
            v[i] = abs(vi + F_v[i] * dt + sq_v[i] * noise[s, 0, i])
        if (n0 + s + 1) % save_every == 0:
            xs[k] = x; ys[k] = y; vs[k] = v; phis[k] = phi
            k += 1
    return k


def advance(x, y, v, phi, A, W, P, v_c, dt, noise, n0, save_every, k, out):
    """Drop-in replacement for model.advance."""
    rows, cols = np.nonzero(A)                        # row-ordered neighbour pairs -> CSR
    indptr = np.r_[0, np.cumsum(A.sum(1))].astype(np.int64)
    return _kernel(x, y, v, phi, indptr, cols.astype(np.int64), W["sm"], W["al"], W["at"],
                   P["mu0"], P["v0"], P["mu_sm"], P["mu_al"], P["mu_at"], P["r0"],
                   np.sqrt(2 * P["D_v"] * dt), np.sqrt(2 * P["D_phi"] * dt), v_c,
                   dt, noise, n0, save_every, k, out["x"], out["y"], out["v"], out["phi"])


def simulate(pA, pB, w, N_A, N_B, T, dt, v_init=None, m0=0.5, save_every=1, voronoi_every=10,
             neighbourhood="voronoi", seed=None):
    """Same arguments and output as model.simulate."""
    return model._simulate(advance, pA, pB, w, N_A, N_B, T, dt, v_init, m0, save_every, voronoi_every,
                           neighbourhood, seed)
