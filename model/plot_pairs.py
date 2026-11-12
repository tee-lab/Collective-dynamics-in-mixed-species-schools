"""Figure of the mixed-species simulations saved by simulate_pairs.py (4 x 4, one column per pair): conspecific vs
heterospecific interaction strengths and net sociality; speed, polarisation and subgroup centroid distance (with a
null of random subgroups). Parameters for the top row are read from the HDF5 file. Writes <stem>.png and <stem>.pdf.

usage: python plot_pairs.py [output/mixsp_sims.h5] [--out output/mixsp_sims]"""
import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

import analysis as AN

HERE = Path(__file__).parent
KEYS = ("v0", "mu0", "mu_sm", "mu_al", "mu_at", "r0", "D_phi", "D_v", "v_c")
W_KEYS = [f"w_{c}_{ab}" for ab in ("AB", "BA") for c in ("sm", "al", "at")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5", nargs="?", default=HERE / "output" / "mixsp_sims.h5")
    ap.add_argument("--out", help="output stem (default: the HDF5 path without .h5)")
    a = ap.parse_args()
    stem = a.out or str(Path(a.h5).with_suffix(""))

    pairs = {}
    with h5py.File(a.h5) as f:
        n_A, n_B = int(f.attrs["n_fish_A"]), int(f.attrs["n_fish_B"])
        for name, g in f.items():
            runs = [{k: r[k][:].astype(float) if k != "species" else r[k][:] for k in ("t", "x", "y", "v", "phi", "species")}
                    for r in g.values()]
            pairs[name] = dict(A=g.attrs["A"], B=g.attrs["B"], pA={k: float(g.attrs[f"A_{k}"]) for k in KEYS},
                               pB={k: float(g.attrs[f"B_{k}"]) for k in KEYS}, w={k: float(g.attrs[k]) for k in W_KEYS},
                               S=AN.mixed_stats(runs, t_drop=g.attrs["transient_s"]))
    order = [p for p in ("TB_RB", "BPT_ST", "NT_ST", "NT_RET") if p in pairs] + [p for p in pairs if p not in ("TB_RB", "BPT_ST", "NT_ST", "NT_RET")]

    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 6.5, "lines.markersize": 5})
    PTS = {n: AN.interaction_points(pairs[n]["pA"], pairs[n]["pB"], pairs[n]["w"]) for n in order}
    lim = 1.08 * max(v for P in PTS.values() for xy in P.values() for v in xy)     # common limit of the top row
    fig, ax = plt.subplots(4, len(order), figsize=(17 / 2.54, 18 / 2.54), dpi=150, squeeze=False)
    for j, name in enumerate(order):
        q = pairs[name]; S = q["S"]
        AN.plot_interactions(ax[0, j], PTS[name], (q["A"], q["B"]), lim)
        ax[0, j].set_title(f"{q['A']} + {q['B']}", fontsize=8)
        if j == 0:
            ax[0, j].set_ylabel("Heterospecific\n" + r"($\hat{\mu}_{AB}, \hat{\mu}_{BA}$)", linespacing=1.6)
        dmax = min(AN.x_upper((S["h_d"], S["h_null"]), AN.D_EDGES), AN.D_EDGES[-1])   # capped at the histogram range
        AN.plot_column(ax[1:, j], S, names=(f"{q['A']}{n_A}", f"{q['B']}{n_B}"),
                       vmax=2 * max(q["pA"]["v0"], q["pB"]["v0"]), dmax=dmax)
        if j:
            for b in ax[2:, j]:
                b.get_legend().remove()
            for b in ax[1:, j]:
                b.set_ylabel("")
    AN.interaction_legend(fig)
    fig.tight_layout(pad=0.3, h_pad=0.8, w_pad=0.5, rect=(0, 0, 1, 0.965))
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}")
    print(f"saved {stem}.png, {stem}.pdf")


if __name__ == "__main__":
    main()
