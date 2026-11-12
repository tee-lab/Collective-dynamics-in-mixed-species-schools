"""Simulate single-species groups (16 fish) of each species and save the trajectories to HDF5.
Parameters: singlesp-params.toml. Per species: --runs runs of --steps steps (dt = 0.001 s), frames every 40 steps
(0.04 s), random initial conditions (initial speed uniform in [0, 2 v0], polarisation uniform in [0, 1]).

usage: python simulate_single.py [--params singlesp-params.toml] [--out output/singlesp_sims.h5]
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
N, DT, SAVE = 16, 0.001, 40
UNITS = {"v0": "BL/s", "mu0": "1/s", "mu_sm": "1/s", "mu_al": "1/s", "mu_at": "1/s^2", "r0": "BL",
         "D_phi": "BL^2/s^3", "D_v": "BL^2/s^3", "v_c": "BL/s"}


def run(p, T, v_init, m0, seed, neighbourhood):
    o = model.simulate(p, None, None, N, 0, T, DT, v_init=v_init, m0=m0, save_every=SAVE, seed=seed, neighbourhood=neighbourhood)
    return {k: o[k].astype(np.float32) for k in ("x", "y", "v", "phi")} | {"t": o["t"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default=HERE / "singlesp-params.toml")
    ap.add_argument("--out", default=HERE / "output" / "singlesp_sims.h5")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--steps", type=int, default=10**6)
    ap.add_argument("--neighbourhood", default="voronoi", choices=("voronoi", "mean-field"))
    a = ap.parse_args()
    with open(a.params, "rb") as f:
        params = tomllib.load(f)

    jobs = []
    for sp, p in params.items():
        rng = np.random.default_rng(0)
        v_inits = rng.uniform(0, 2 * p["v0"], a.runs); m0s = rng.uniform(0, 1, a.runs)
        jobs += [(sp, i, v, m, 100 * i + 1) for i, (v, m) in enumerate(zip(v_inits, m0s))]
    out = Parallel(n_jobs=-1, verbose=5)(delayed(run)(params[sp], a.steps, v, m, s, a.neighbourhood) for sp, _, v, m, s in jobs)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.out, "w") as f:
        f.attrs.update(
            description=f"Single-species schooling model (N = {N} fish): trajectories of {a.runs} runs per species with "
                        "random initial conditions. Equations: model.py (N_B = 0).",
            created=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            source="simulate_single.py, simulator model_optimized.py",
            parameter_source=Path(a.params).name,
            n_fish=N, n_steps=a.steps, dt_s=DT, save_every_steps=SAVE, frame_interval_s=SAVE * DT,
            neighbourhood="voronoi (recomputed every 10 steps)" if a.neighbourhood == "voronoi" else "mean-field",
            integrator="Euler-Maruyama, reflecting speed boundary at 0",
            layout="/<SP>/run_<i>/{t (frames,), x, y, v, phi (frames, n_fish)}; parameters as attributes of /<SP>",
            units="t: s; x, y: body lengths (BL); v: speed, BL/s; phi: heading, rad in [-pi, pi)",
            transient_note="Frames are saved from t = 0. The figure drops t <= transient_s (attribute of /<SP>), "
                           "= min(50 / mu0, T*dt / 2).")
        for (sp, i, v_init, m0, seed), o in zip(jobs, out):
            p = params[sp]; g = f.require_group(sp)
            if not g.attrs:
                g.attrs.update({k: float(p[k]) for k in UNITS}, transient_s=AN.transient(p["mu0"], a.steps, DT), units=str(UNITS))
            r = g.create_group(f"run_{i}")
            r.attrs.update(seed=seed, v_init_BL_per_s=float(v_init), m0=float(m0),
                           initial_condition="all fish at speed v_init, headings with polarisation m0, positions uniform in a disc")
            r.create_dataset("t", data=o["t"])
            for k in ("x", "y", "v", "phi"):
                r.create_dataset(k, data=o[k], compression="gzip", compression_opts=4, shuffle=True,
                                 chunks=(min(4096, len(o["t"])), N))
    print("saved", a.out)


if __name__ == "__main__":
    main()
