# Mixed-species schooling model

This folder contains the code for simulating the individual-based, mixed-species fish-schooling model and generating the associated figures.

- `model.py` is a simplified, readable version of the model, while `model_optimized.py` is an optimised version of the code, with Numba optimisations, which is much faster for batch-simulations, parameter-search etc. 
- `singlesp-params.toml` contains the species-specific parameter values for each of the six species.
- `mixsp-params.toml` contains the heterospecific interaction parameters for the four species pairs. For species pair $A, B$ and interaction $k \in \{ \text{al, sm, at} \}$, the parameters are represented in terms of $\omega^k_{AB} = \mu^k_{AB}/\mu^k_{AA}$.

`plot_single.py` and `plot_pairs.py` will reproduce the respective single- and mixed-species model figures from the manuscript. The remaining scripts contain helper code.