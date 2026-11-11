# Mixed-species fish schools: group properties (group size 16)

Analysis code for the trajectories of fish schools of 16 individuals, swimming either as a single species (16 fish of one species) or as a mixed-species school (8 + 8 fish of two species).
For each school we compute individual speed, group polarization, nearest-neighbour distance and the distance between the two species' subgroups, and compare the experiments with a mixed-species schooling model.

Four species pairs were tested, each with three treatments (species A alone, species B alone, 8 A + 8 B):

| Pair (A - B) | Single species | Mixed |
|---|---|---|
| TB - RB | TB16, RB16 | TB8RB8 |
| BPT - ST | BPT16, ST16 | BPT8ST8 |
| NT - ST | NT16, ST16 | NT8ST8 |
| NT - RET | NT16, RET16 | NT8RET8 |

Species A is always the first species of the pair (orange in the figures), species B the second (blue).

## Repository layout

```
Code/       Jupyter notebooks (one per figure set)
Data/       trajectories and model output (see below)
Results/    figures written by the notebooks (PNG and PDF, 300 dpi)
```

Every notebook reads from `../Data` and writes to `../Results`, so run them from inside `Code/`.

## Data

All positions are in body lengths (BL), speeds in BL/s; videos were recorded at 25 frames per second.

| File(s) | Content |
|---|---|
| `<SP>16_<rep>_traj_bl.npz` | Single-species trials (TB, RB: 5 replicates; BPT, ST, NT: 10; RET: 12, of which 1-10 are used). Arrays `positions` and `velocities`, shape (frames, 16, 2); untracked fish are NaN. |
| `<A>8<B>8_<rep>_traj_bl.npz` | Mixed trials (BPT8ST8, NT8ST8: 10 replicates; NT8RET8: 12). Same arrays; the **first 8 fish are species A** (BPT or NT). |
| `TBRB16_traj_bl.csv` | Mixed TB8 + RB8 trials (5 replicates, 200 frames each, 1000 frames in total) with the species of every fish preserved. Columns `Replicate, FrameID, FishID, SPECIES ID, body_length, x, y, vx, vy` (16 rows per frame). Used for this pair instead of `TB8RB8_*_traj_bl.npz`, where the species of each fish is not preserved. |
| `MS_<rep>_RB8+TB8_c<clip>_<TB/RB>1.csv` | Single-fish trajectories from the TB8 + RB8 trials (5 replicates x 5 clips x one TB and one RB fish). Not used by the notebooks below. |
| `model_single_species.h5` | Single-species model runs: `/<SP>/run_<i>/{t, x, y, v, phi}`; species parameters and `transient_s` stored as attributes of `/<SP>`. |
| `model_mixed_species.h5` | Mixed-species model runs with the fitted cross-species weights: `/<A>_<B>/run_<i>/{t, x, y, v, phi, species}` (species 0 = A, 1 = B); parameters of both species (`A_mu0`, `A_mu_sm`, ...), cross-species weights (`w_sm_AB`, `w_al_BA`, ...) and `transient_s` as attributes of `/<A>_<B>`. |
| `model_mixed_species_w0.h5`, `model_mixed_species_w1.h5` | Same, with all cross-species weights set to 0 (species ignore each other) and to 1 (a heterospecific neighbour counts like a conspecific one). |
| `singlesp-params.toml`, `mixsp-params.toml` | Parameter files the model runs were generated from (the notebooks read the parameters from the h5 attributes). |

## Notebooks

### `single species group properties.ipynb`
Real data vs the single-species model for all six species (columns TB, RB, BPT, ST, NT, RET).
Rows: individual speed, group polarization and nearest-neighbour distance (NND), as probability density histograms (data solid, model dashed).
Velocity is computed from the positions by central difference; polarization $m = |\frac{1}{n}\sum_i \mathbf{v}_i/|\mathbf{v}_i||$ over the fish present (at least 8); NND in frames with at least 75 % of the fish tracked. Model frames before the transient are dropped.
Output: `single_species_group_properties_gs16`.

### `mixed_species_group_properties.ipynb`
Mixed-species experiments, one column per pair. Rows:
1. individual speed of the two species subgroups and the whole group, with box plots;
2. polarization of the two subgroups, the whole group and a random subgroup (8 fish chosen at random in every frame, ignoring species);
3. distance between the centroids of the two species' subgroups, observed vs a null in which the species labels are shuffled 1000 times per frame.

Frames need at least 75 % of the fish tracked. The 1000-shuffle null makes this the slowest notebook (a few minutes).
Output: `mixed_species_group_properties_gs16`.

### `mixed_species_model_group_properties.ipynb`
The same rows for the model, preceded by a row of non-dimensional interaction strengths: conspecific ($\hat\mu_{AA}$, $\hat\mu_{BB}$) vs heterospecific ($\hat\mu_{AB}$, $\hat\mu_{BA}$) for speed matching, alignment and attraction,
$\hat\mu^{c}_{ij} = w^{c}_{ij}\mu^{c}_i/\mu_{0,i}$ (speed matching, alignment) and $\hat\mu^{\rm at}_{ij} = \sqrt{w^{\rm at}_{ij}\mu^{\rm at}_i}/\mu_{0,i}$ (attraction), with $w_{ii} = 1$.
The centroid-distance null splits the 16 fish at random into two groups of 8 in every frame.
One figure per model file: `mixed_species_model_group_properties_gs16` (fitted weights), `mixed_species_model_w0_group_properties_gs16` and `mixed_species_model_w1_group_properties_gs16`.

### `boundary_effect.ipynb`
Checks how much the results depend on data recorded close to the arena wall. Every distribution is drawn twice: with all data and with the outer 25 % of the arena area removed (the wall of each trial is its outermost tracked point $R$; the cut keeps everything within $R\sqrt{0.75}$). Speed and NND are filtered by the fish's distance from the centre, polarization and centroid distance by the group centroid's distance. The Hellinger distance $H$ between the two distributions (0 = identical, 1 = no overlap) is shown above each panel.
Outputs: `boundary_effect_single_species_gs16` (speed, polarization, NND for the six species) and `boundary_effect_mixed_species_gs16` (speed, polarization, NND and centroid distance for the four pairs).

## Requirements

Python 3.10 or later with `numpy`, `pandas`, `matplotlib`, `h5py` and `jupyter`:

```bash
pip install numpy pandas matplotlib h5py jupyter
cd Code
jupyter nbconvert --to notebook --execute --inplace mixed_species_group_properties.ipynb   # or open the notebooks in Jupyter
```

Figures use Arial if it is installed (otherwise Liberation Sans or DejaVu Sans) and are sized for an A4 page width.
