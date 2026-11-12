"""Histograms and plot panels for the model figures (plot_pairs.py, plot_single.py)."""
import numpy as np

V_EDGES = np.arange(0, 60 + 1e-9, 0.01)       # speed (BL/s), fine bins; re-binned for display (rebin)
M_EDGES = np.linspace(0, 1, 51)               # polarisation |m|
D_EDGES = np.arange(0, 40 + 1e-9, 0.01)       # subgroup centroid distance (BL)
NND_EDGES = np.arange(0, 60 + 1e-9, 0.01)     # nearest-neighbour distance (BL)
COL = {"all": "black", "A": "#e3702d", "B": "#1a9e77", "null": "#7f7f7f", "obs": "#8b1010", "model": "#2a78d6"}
TDROP = 50.0                                  # transient dropped: min(TDROP / mu0, half the run)


def transient(mu0, T, dt):
    """Initial time (s) dropped from the analysis: 50 / mu0 (the slower species for a pair), at most half the run."""
    return min(TDROP / mu0, 0.5 * T * dt)


def polarisation(phi):
    """|m| per frame of headings phi (frames, n)."""
    return np.abs(np.exp(1j * phi).mean(1))


def centroid_distance(x, y, idx_A, idx_B):
    """Per-frame distance between the centroids of fish idx_A and idx_B ((frames, n) index arrays or (n,))."""
    take = lambda a, i: np.take_along_axis(a, i, 1) if i.ndim == 2 else a[:, i]
    return np.hypot(take(x, idx_A).mean(1) - take(x, idx_B).mean(1), take(y, idx_A).mean(1) - take(y, idx_B).mean(1))


def nnd(x, y):
    """Nearest-neighbour distance of every fish in every frame, (frames, n)."""
    d = np.hypot(x[:, :, None] - x[:, None], y[:, :, None] - y[:, None])
    d[:, np.arange(x.shape[1]), np.arange(x.shape[1])] = np.inf
    return d.min(-1)


def mixed_stats(runs, t_drop, seed=0):
    """Histogram counts pooled over runs (frames with t > t_drop): h_v[g], h_m[g] for g = all, A, B; h_d (distance
    between the subgroup centroids) and h_null (same for a random split into subgroups of the same sizes, per frame)."""
    rng = np.random.default_rng(seed)
    S = {"h_v": dict.fromkeys(("all", "A", "B"), 0), "h_m": dict.fromkeys(("all", "A", "B"), 0), "h_d": 0, "h_null": 0}
    for o in runs:
        k = o["t"] > t_drop; v, phi, x, y = o["v"][k], o["phi"][k], o["x"][k], o["y"][k]
        A, B = np.flatnonzero(o["species"] == 0), np.flatnonzero(o["species"] == 1)
        for g, idx in (("all", np.r_[A, B]), ("A", A), ("B", B)):
            S["h_v"][g] = S["h_v"][g] + np.histogram(v[:, idx], V_EDGES)[0]
            S["h_m"][g] = S["h_m"][g] + np.histogram(polarisation(phi[:, idx]), M_EDGES)[0]
        S["h_d"] = S["h_d"] + np.histogram(centroid_distance(x, y, A, B), D_EDGES)[0]
        perm = rng.permuted(np.tile(np.arange(len(A) + len(B)), (len(v), 1)), axis=1)
        S["h_null"] = S["h_null"] + np.histogram(centroid_distance(x, y, perm[:, :len(A)], perm[:, len(A):]), D_EDGES)[0]
    return S


def single_stats(runs, t_drop):
    """Histogram counts pooled over runs (frames with t > t_drop): h_m (polarisation), h_v (speed), h_nnd."""
    S = {"h_m": 0, "h_v": 0, "h_nnd": 0}
    for o in runs:
        k = o["t"] > t_drop
        S["h_m"] = S["h_m"] + np.histogram(polarisation(o["phi"][k]), M_EDGES)[0]
        S["h_v"] = S["h_v"] + np.histogram(o["v"][k], V_EDGES)[0]
        S["h_nnd"] = S["h_nnd"] + np.histogram(nnd(o["x"][k], o["y"][k]), NND_EDGES)[0]
    return S


def density(h, edges):
    h = np.asarray(h, float)
    return h / (h.sum() * np.diff(edges))


def x_upper(hists, edges, q=0.995, margin=1.2):
    """Axis upper limit: margin x the largest q-quantile over the histograms (counts on edges)."""
    return margin * max(np.interp(q, np.cumsum(h) / np.sum(h), edges[1:]) for h in hists)


def rebin(h, edges, xmax, n=100):
    """Fine histogram (counts on edges) -> n equal bins on [edges[0], xmax], by interpolating the cumulative counts.
    Returns (centres, density), normalised by all counts (mass beyond xmax is not renormalised away)."""
    h = np.asarray(h, float); e = np.linspace(edges[0], xmax, n + 1)
    C = np.interp(e, edges, np.r_[0.0, np.cumsum(h)])
    return 0.5 * (e[1:] + e[:-1]), np.diff(C) / (h.sum() * np.diff(e))


def plot_column(axs, S, names, vmax, dmax):
    """Speed, polarisation and centroid-distance panels of one species pair. names: labels of subgroups A and B;
    vmax, dmax: x ranges of the speed and centroid-distance panels (100 bins each)."""
    lab = {"all": "Mixed group", "A": f"Subgroup {names[0]}", "B": f"Subgroup {names[1]}"}
    mc = 0.5 * (M_EDGES[1:] + M_EDGES[:-1])
    for a, key, xmax in ((axs[0], "h_v", vmax), (axs[1], "h_m", 1.0)):
        for g in ("all", "A", "B"):
            c, p = rebin(S[key][g], V_EDGES, vmax) if key == "h_v" else (mc, density(S[key][g], M_EDGES))
            a.step(c, p, where="mid", color=COL[g], lw=2.2 if g == "all" else 1.5, ls=":" if g == "all" else "-", label=lab[g])
        a.set(xlim=(0, xmax), ylim=(0, None), ylabel="Density")
    axs[0].set_xlabel("Individual speed (BL/s)"); axs[1].set_xlabel("Polarisation |m|")
    for key, g, lb in (("h_null", "null", "Null"), ("h_d", "obs", "Observed")):
        c, p = rebin(S[key], D_EDGES, dmax)
        axs[2].plot(c, p, color=COL[g], lw=2.5, label=lb); axs[2].fill_between(c, p, color=COL[g], alpha=0.35, lw=0)
    axs[2].set(xlim=(0, dmax), ylim=(0, None), ylabel="Density", xlabel="Centroid distance (BL)")
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
        a.legend(frameon=False, loc="upper left" if a is axs[1] else "upper right")


MARKERS = {"sm": ("o", r"speed modulation, $\hat{\mu}_{\rm sm}$"), "al": ("s", r"alignment, $\hat{\mu}_{\rm al}$"),
           "at": ("^", r"attraction, $\hat{\mu}_{\rm at}$")}


def interaction_points(pA, pB, w):
    """Non-dimensional interaction strengths {(i, c): (conspecific, heterospecific)} for i = A, B and c = sm, al, at
    (mu / mu0_i; attraction sqrt(mu_at) / mu0_i), and c = "soc": net sociality
    S_ij = (mu_ij^al + mu_ij^sm + sqrt(mu_ij^at)) / (3 mu0_i), the mean of the three."""
    p = {"A": pA, "B": pB}; out = {}
    for i, j in (("A", "B"), ("B", "A")):
        m0 = p[i]["mu0"]
        own = {"sm": p[i]["mu_sm"] / m0, "al": p[i]["mu_al"] / m0, "at": np.sqrt(p[i]["mu_at"]) / m0}
        het = {"sm": w[f"w_sm_{i}{j}"] * own["sm"], "al": w[f"w_al_{i}{j}"] * own["al"], "at": np.sqrt(w[f"w_at_{i}{j}"]) * own["at"]}
        for c in own:
            out[i, c] = (own[c], het[c])
        out[i, "soc"] = (sum(own.values()) / 3, sum(het.values()) / 3)
    return out


def plot_interactions(a, P, names, lim):
    """Conspecific vs heterospecific panel of one pair (P from interaction_points); grey: heterospecific > conspecific."""
    import matplotlib.pyplot as plt
    a.fill_between([0, lim], [0, lim], lim, color="0.9", lw=0, zorder=0)
    a.plot([0, lim], [0, lim], color="black", lw=0.6, zorder=1)
    for i in "AB":
        a.plot(*P[i, "soc"], "D", color=COL[i], mec="black", mew=0.8, ms=9, ls="none", zorder=2, clip_on=False)
        for c, (m, _) in MARKERS.items():
            a.plot(*P[i, c], m, color=COL[i], mec="black", mew=0.5, ms=5, ls="none", alpha=0.6, zorder=3, clip_on=False)
    a.set(xlim=(0, lim), ylim=(0, lim), aspect="equal", xlabel=r"Conspecific ($\hat{\mu}_{AA}, \hat{\mu}_{BB}$)")
    a.spines[["top", "right"]].set_visible(False)
    a.legend(handles=[plt.Line2D([], [], marker="o", ls="none", color=COL[i], mec="black", mew=0.5, label=n)
                      for i, n in zip("AB", names)], frameon=False, loc="upper left", handletextpad=0.2, borderaxespad=0.1)


def interaction_legend(fig):
    """Marker legend of the interaction panels, above the figure."""
    import matplotlib.pyplot as plt
    fig.legend(handles=[plt.Line2D([], [], marker=m, ls="none", color="white", mec="black", ms=5, label=l) for m, l in MARKERS.values()]
                       + [plt.Line2D([], [], marker="D", ls="none", color="white", mec="black", mew=0.8, ms=7, label=r"net sociality $S$")],
               loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, columnspacing=1.2)
