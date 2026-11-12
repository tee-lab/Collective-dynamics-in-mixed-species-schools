# Mixed-species fish schools: group properties

This repository contains Jupyter Notebooks used for the analysis of single- and mixed-species fish schools (group size 16) in:

*Contrasting collective dynamics in mixed-species fish schools*.

---

# Notebooks

| Notebook | Description |
| -------- | ----------- |
| `single species group properties.ipynb` | Compares real data with the single-species model for all six species: individual speed, group polarization and nearest-neighbour distance |
| `mixed_species_group_properties.ipynb` | Computes group properties of the mixed-species schools: speed of the two species subgroups, polarization of the subgroups, the whole group and a random subgroup, and the distance between the two subgroup centroids against a label-shuffle null |
| `mixed_species_model_group_properties.ipynb` | Same group properties for the mixed-species model, with the non-dimensional interaction strengths of each pair; one figure each for the fitted weights and for all cross-species weights set to 0 and to 1 |
| `boundary_effect.ipynb` | Tests how much the distributions depend on data recorded near the arena wall by removing the outer 25 % of the arena area, with the Hellinger distance between the two distributions |

---

# Data

## Source

The datasets contain processed trajectories derived from raw tracking data; available at **([Zenodo](<link>))**

Four species pairs were tested, each with three treatments (species A alone, species B alone, 8 A + 8 B):

| Pair (A - B) | Single species | Mixed |
| ------------ | -------------- | ----- |
| TB - RB      | `TB16`, `RB16`   | `TB8RB8` |
| BPT - ST     | `BPT16`, `ST16`  | `BPT8ST8` |
| NT - ST      | `NT16`, `ST16`   | `NT8ST8` |
| NT - RET     | `NT16`, `RET16`  | `NT8RET8` | 

TB = tiger barbs, RB = rosy barbs, BPT = black phantom tetra, ST = serape tetra, NT = neon tetra and RET = red eye tetra.

---

## `<treatment>_<replicate>_traj_bl.npz`

*Input for all four notebooks* (single-species files: `TB16`, `RB16`, `BPT16`, `ST16`, `NT16`, `RET16`; mixed files: `BPT8ST8`, `NT8ST8`, `NT8RET8`)

| Key          | Shape                            | Description                                                        |
| ------------ | -------------------------------- | ------------------------------------------------------------------ |
| `positions`  | `(frames, individuals, 2)`       | x, y position in body lengths (BL); untracked fish are `NaN`       |
| `velocities` | `(frames, individuals, 2)`       | x, y velocity in body lengths per second (BL/s)                    |

In the mixed files the **first 8 individuals are species A** (BPT or NT).

---

## `TBRB16_traj_bl.csv`

*Input for `mixed_species_group_properties.ipynb` and `boundary_effect.ipynb` (TB8 + RB8 trials)*

| Column        | Description                                             |
| ------------- | ------------------------------------------------------- |
| `Replicate`   | Replicate (5 replicates, 200 frames each)               |
| `FrameID`     | Frame index                                             |
| `FishID`      | Fish ID                                                 |
| `SPECIES ID`  | Species identity (TB = tiger barbs, RB = rosy barbs)    |
| `body_length` | Body length                                             |
| `x`, `y`      | Position (BL)                                           |
| `vx`, `vy`    | Velocity (BL/s)                                         |

---

## `model_single_species.h5`

*Input for `single species group properties.ipynb`*

| Path                                  | Description                                                        |
| ------------------------------------- | ------------------------------------------------------------------ |
| `/<SP>/run_<i>/t`                     | Time (s), one frame every 0.04 s                                   |
| `/<SP>/run_<i>/x`, `y`                | Positions (BL), shape `(frames, individuals)`                      |
| `/<SP>/run_<i>/v`, `phi`              | Speed (BL/s) and heading (rad), shape `(frames, individuals)`      |
| attributes of `/<SP>`                 | Model parameters and `transient_s` (initial transient, dropped)    |

---

## `model_mixed_species.h5`, `model_mixed_species_w0.h5`, `model_mixed_species_w1.h5`

*Input for `mixed_species_model_group_properties.ipynb`* (fitted cross-species weights; all weights 0; all weights 1)

| Path                                  | Description                                                        |
| ------------------------------------- | ------------------------------------------------------------------ |
| `/<A>_<B>/run_<i>/t, x, y, v, phi`    | As in the single-species file                                      |
| `/<A>_<B>/run_<i>/species`            | Species of each individual (0 = A, 1 = B)                          |
| attributes of `/<A>_<B>`              | Parameters of both species (`A_mu0`, `A_mu_sm`, `A_mu_al`, `A_mu_at`, ...), cross-species weights (`w_sm_AB`, `w_al_BA`, ...) and `transient_s` |

---

## ⚙️ How to Run

### Requirements

```bash
pip install numpy pandas matplotlib h5py jupyter
```

All notebooks are in `Code/`, read the data from `../Data` and save the figures to `../Results`.

---

### Notebook 1 — Single species group properties
1. Place the single-species `*16_*_traj_bl.npz` files and `model_single_species.h5` in `Data/`.
2. Run:

   ```bash
   jupyter notebook "single species group properties.ipynb"
   ```
3. Execute all cells.

---

### Notebook 2 — Mixed species group properties
1. Place the mixed `*8*8_*_traj_bl.npz` files and `TBRB16_traj_bl.csv` in `Data/`.
2. Run:

   ```bash
   jupyter notebook mixed_species_group_properties.ipynb
   ```
3. Execute all cells (the 1000-shuffle null takes a few minutes).

---

### Notebook 3 — Mixed species model group properties
1. Place `model_mixed_species.h5`, `model_mixed_species_w0.h5` and `model_mixed_species_w1.h5` in `Data/`.
2. Run:

   ```bash
   jupyter notebook mixed_species_model_group_properties.ipynb
   ```
3. Execute all cells.

---

### Notebook 4 — Boundary effect
1. Place all `*_traj_bl.npz` files and `TBRB16_traj_bl.csv` in `Data/`.
2. Run:

   ```bash
   jupyter notebook boundary_effect.ipynb
   ```
3. Execute all cells.

---

## Outputs

All figures are saved as PNG and PDF (300 dpi, A4 width).

### Single species group properties notebook

| Figure | Description | File |
| ------ | ----------- | ---- |
| Fig. 1 | Speed, polarization and nearest-neighbour distance, data vs model, for the six species | `single_species_group_properties_gs16` |

---

### Mixed species group properties notebook

| Figure | Description | File |
| ------ | ----------- | ---- |
| Fig. 1 | Speed, polarization (with random subgroup) and subgroup centroid distance (with null) for the four pairs | `mixed_species_group_properties_gs16` |

---

### Mixed species model group properties notebook

| Figure | Description | File |
| ------ | ----------- | ---- |
| Fig. 1 | Interaction strengths, speed, polarization and centroid distance, fitted weights | `mixed_species_model_group_properties_gs16` |
| Fig. 2 | Same, all cross-species weights 0 | `mixed_species_model_w0_group_properties_gs16` |
| Fig. 3 | Same, all cross-species weights 1 | `mixed_species_model_w1_group_properties_gs16` |

---

### Boundary effect notebook

| Figure | Description | File |
| ------ | ----------- | ---- |
| Fig. 1 | Speed, polarization and NND, all data vs outer 25 % of the area removed, single species | `boundary_effect_single_species_gs16` |
| Fig. 2 | Speed, polarization, NND and centroid distance, all data vs outer 25 % removed, mixed species | `boundary_effect_mixed_species_gs16` |

---

## Notes

- A `treatment` is the group composition (e.g. `TB16`, `TB8RB8`) and a `replicate` is an independent experimental trial. RB and TB have 5 replicates per treatment, the other species 10 (RET16 and NT8RET8 have 12 files; RET16 replicates 1-10 are used in the single-species notebook).
- Species A is always the first species of a pair (orange in the figures), species B the second (blue).
- Polarization, nearest-neighbour distance and centroid distance use only frames with at least 75 % of the fish (or of each subgroup) tracked; files with less than 1 % tracked positions are skipped.
- Model frames before `transient_s` are discarded.
- The arena wall in the boundary analysis is the outermost tracked point $R$ of each trial; removing the outer 25 % of the area keeps data within $R\sqrt{0.75}$.

---

## Reproducibility

- The experimental null model shuffles species identities among the tracked fish 1000 times per frame; the model null splits the 16 fish at random into two groups of 8 in every frame.
- The random subgroup is 8 fish chosen at random in every frame, ignoring species.
- Random seeds are fixed in the notebooks (`numpy.random.default_rng(0)` and `(1)`), so the figures are reproducible.

---

## Contact

For questions or collaboration, please contact the author.
