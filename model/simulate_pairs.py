"""Simulate the four mixed-species pairs (8 + 8 fish) and save the trajectories to HDF5.
Species parameters: singlesp-params.toml; cross-species weights: mixsp-params.toml (or --weights), or all six
weights of every pair set to one value with --w-all (e.g. 0: no cross-species interactions; 1: species identical).
Per pair: --runs runs of --steps steps (dt = 0.001 s), frames every 40 steps (0.04 s), random initial conditions
(initial speed uniform in [0, 2 mean(v0)], polarisation uniform in [0, 1]).

usage: python simulate_pairs.py [--weights mixsp-params.toml | --w-all VALUE] [--out output/mixsp_sims.h5]
                                [--runs 5] [--steps 1000000] [--neighbourhood voronoi|mean-field]"""
import argparse
import datetime
import tomllib
from pathlib import Path

import h5py
import numpy as np
from joblib import Parallel, delayed

import analysis as AN
import model_optimized as model

HERE = Path(__file__).parent
PAIRS = ["TB_RB", "BPT_ST", "NT_ST", "NT_RET"]
N_A, N_B, DT, SAVE = 8, 8, 0.001, 40
W_KEYS = [f"w_{c}_{ab}" for ab in ("AB", "BA") for c in ("sm", "al", "at")]


def run(pA, pB, w, T, v_init, m0, seed, neighbourhood):
    o = model.simulate(pA, pB, w, N_A, N_B, T, DT, v_init=v_init, m0=m0, save_every=SAVE, seed=seed, neighbourhood=neighbourhood)
    return {k: o[k].astype(np.float32) for k in ("x", "y", "v", "phi")} | {"t": o["t"], "species": o["species"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=HERE / "mixsp-params.toml")
    ap.add_argument("--params", default=HERE / "singlesp-params.toml")
    ap.add_argument("--w-all", type=float, help="set all cross-species weights to this value (ignores --weights values)")
    ap.add_argument("--out", help="default: output/mixsp_sims.h5, or output/mixsp_sims_w<VALUE>.h5 with --w-all")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--steps", type=int, default=10**6)
    ap.add_argument("--neighbourhood", default="voronoi", choices=("voronoi", "mean-field"))
    a = ap.parse_args()
    if a.out is None:
        a.out = HERE / "output" / ("mixsp_sims.h5" if a.w_all is None else f"mixsp_sims_w{a.w_all:g}.h5")
    with open(a.weights, "rb") as f:
        MIX = tomllib.load(f)

    pairs, jobs = {}, []
    for name in PAIRS:
        pA, pB, w = model.load_pair(name, a.params, a.weights)
        if a.w_all is not None:
            w = dict.fromkeys(w, a.w_all)
        pairs[name] = dict(A=MIX[name]["A"], B=MIX[name]["B"], pA=pA, pB=pB, w=w)
        rng = np.random.default_rng(0)
        v_inits = rng.uniform(0, 2 * 0.5 * (pA["v0"] + pB["v0"]), a.runs); m0s = rng.uniform(0, 1, a.runs)
        jobs += [(name, i, v, m, 100 * i + 1) for i, (v, m) in enumerate(zip(v_inits, m0s))]
    out = Parallel(n_jobs=-1, verbose=5)(delayed(run)(pairs[n]["pA"], pairs[n]["pB"], pairs[n]["w"], a.steps, v, m, s, a.neighbourhood)
                                         for n, _, v, m, s in jobs)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.out, "w") as f:
        f.attrs.update(
            description=f"Mixed-species schooling model ({N_A} fish of species A + {N_B} of species B): trajectories of "
                        f"{a.runs} runs per species pair with random initial conditions. Equations: model.py.",
            created=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            source="simulate_pairs.py, simulator model_optimized.py",
            parameter_source=f"species: {Path(a.params).name}; cross-species weights: " + (Path(a.weights).name if a.w_all is None else f"all set to {a.w_all:g} (--w-all)"),
            n_fish_A=N_A, n_fish_B=N_B, n_steps=a.steps, dt_s=DT, save_every_steps=SAVE, frame_interval_s=SAVE * DT,
            neighbourhood="voronoi (recomputed every 10 steps)" if a.neighbourhood == "voronoi" else "mean-field",
            integrator="Euler-Maruyama, reflecting speed boundary at 0",
            layout="/<PAIR>/run_<i>/{t (frames,), x, y, v, phi (frames, 16), species (16,): 0 = A, 1 = B}; "
                   "parameters as attributes of /<PAIR> (A_<param>, B_<param>, weights, normalise, v_c)",
            units="t: s; x, y: body lengths (BL); v: speed, BL/s; phi: heading, rad in [-pi, pi); "
                  "v0 BL/s, mu0 mu_sm mu_al 1/s, mu_at 1/s^2, r0 BL, D_phi D_v BL^2/s^3, v_c BL/s",
            weights_note="w_<c>_AB: weight an A fish gives a B neighbour, w_<c>_BA: B's weight for an A neighbour; "
                         "c = sm (speed matching), al (alignment), at (attraction); conspecific weights 1",
            transient_note="Frames are saved from t = 0; the figure drops t <= transient_s (attribute of /<PAIR>).")
        for (name, i, v_init, m0, seed), o in zip(jobs, out):
            q = pairs[name]; g = f.require_group(name)
            if not g.attrs:
                g.attrs.update({f"A_{k}": float(v) for k, v in q["pA"].items()} | {f"B_{k}": float(v) for k, v in q["pB"].items()}
                               | {k: float(q["w"][k]) for k in W_KEYS},
                               A=q["A"], B=q["B"], normalise="count", v_c=min(q["pA"]["v_c"], q["pB"]["v_c"]),
                               transient_s=AN.transient(min(q["pA"]["mu0"], q["pB"]["mu0"]), a.steps, DT))
            r = g.create_group(f"run_{i}")
            r.attrs.update(seed=seed, v_init_BL_per_s=float(v_init), m0=float(m0),
                           initial_condition="all fish at speed v_init, headings with polarisation m0, positions uniform in a disc, species mixed")
            r.create_dataset("t", data=o["t"]); r.create_dataset("species", data=o["species"].astype(np.int8))
            for k in ("x", "y", "v", "phi"):
                r.create_dataset(k, data=o[k], compression="gzip", compression_opts=4, shuffle=True,
                                 chunks=(min(4096, len(o["t"])), N_A + N_B))
    print("saved", a.out)


if __name__ == "__main__":
    main()
