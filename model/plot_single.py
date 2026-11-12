"""Figure of the single-species simulations saved by simulate_single.py (one column per species): polarisation,
speed and nearest-neighbour distance (NND) histograms. Writes <stem>.png and <stem>.pdf.

usage: python plot_single.py [output/singlesp_sims.h5] [--out output/singlesp_sims]"""
import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

import analysis as AN

HERE = Path(__file__).parent
ORDER = ["TB", "RB", "BPT", "ST", "NT", "RET"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5", nargs="?", default=HERE / "output" / "singlesp_sims.h5")
    ap.add_argument("--out", help="output stem (default: the HDF5 path without .h5)")
    a = ap.parse_args()
    stem = a.out or str(Path(a.h5).with_suffix(""))

    S = {}
    with h5py.File(a.h5) as f:
        for sp, g in f.items():
            runs = [{k: r[k][:].astype(float) for k in ("t", "x", "y", "v", "phi")} for r in g.values()]
            S[sp] = AN.single_stats(runs, t_drop=g.attrs["transient_s"])
    order = [s for s in ORDER if s in S] + [s for s in S if s not in ORDER]

    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7})
    fig, ax = plt.subplots(3, len(order), figsize=(17 / 2.54, 9.5 / 2.54), dpi=150, squeeze=False)
    mc = 0.5 * (AN.M_EDGES[1:] + AN.M_EDGES[:-1])
    rows = (("h_m", AN.M_EDGES, "Polarisation", "|m|"), ("h_v", AN.V_EDGES, "Speed", "speed (BL/s)"), ("h_nnd", AN.NND_EDGES, "NND", "NND (BL)"))
    for j, sp in enumerate(order):
        ax[0, j].set_title(sp, fontsize=8)
        for i, (key, E, ylab, xlab) in enumerate(rows):
            b = ax[i, j]; h = S[sp][key]
            if key == "h_m":
                c, p, xmax = mc, AN.density(h, E), 1.0
            else:
                xmax = AN.x_upper([h], E); c, p = AN.rebin(h, E, xmax, 60)
            b.step(c, p, where="mid", color=AN.COL["model"], lw=1.1)
            b.set(xlim=(0, xmax), ylim=(0, None)); b.spines[["top", "right"]].set_visible(False)
            b.set_xlabel(xlab, labelpad=1)
            if j == 0:
                b.set_ylabel(f"{ylab}\ndensity")
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}")
    print(f"saved {stem}.png, {stem}.pdf")


if __name__ == "__main__":
    main()
