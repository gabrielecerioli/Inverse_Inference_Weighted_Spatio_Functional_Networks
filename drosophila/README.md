# Drosophila (FlyWire FAFB v783) analysis

Estimates the spatial/latent mixing parameter λ and the physical scale r for each of the 13 major brain systems of the adult fly connectome. It uses the estimator of the paper (`../exact_inference.py`, Methods Eqs. 1–9), unchanged.

## Data

| File | Source |
|---|---|
| `flywire_synapses_783.feather` | FlyWire FAFB v783 synapse table, [Zenodo 10676866](https://zenodo.org/records/10676866) (md5 `f8f1b97c9d4b0ea9b4c8b287f6b99091`) |
| `drosophila_neuron_coordinates.csv` | One row per proofread neuron: `root_id`, `major_system` (13 systems), `x_um,y_um,z_um` (centroid of all its synapses), `primary_np_{x,y,z}_um` (centroid of its synapses in its primary neuropil) |

## Pipeline

```bash
pip install numpy pandas scipy pyarrow torch matplotlib

# 1. one weighted undirected network per major system
python prepare_flywire.py --synapses flywire_synapses_783.feather \
                          --neurons drosophila_neuron_coordinates.csv --out fly_npz

# 2. fits: M=50 induced subgraphs of n=600 neurons per system, d_s=3, profile-likelihood CI
python run_drosophila.py --npz fly_npz --coords primary_np   --device cuda:0
python run_drosophila.py --npz fly_npz --coords syn_centroid --device cuda:1   # robustness check

# 3. per-system summary and ranking plot
python analyze_drosophila.py results/fits_primary_np_ds3.csv results/fits_syn_centroid_ds3.csv
```

## What each step does

**Network.** $W_{ij}$ is the number of synapses $i\to j$ plus $j\to i$ between two distinct neurons of the same major system. Synapses that cross systems, and autapses, are dropped.

**Subgraphs.** For system $s$ and replica $k$, the seed is the $k$-th integer drawn from `numpy.random.default_rng(crc32(s))`. The subgraph is `default_rng(seed).choice(N, 600, replace=False)`, an induced subgraph sampled uniformly. Seeds don't depend on the coordinate choice, so both coordinate variants are fitted on identical subgraphs.

**Fit** (`infer_parameters_exact`, defaults):
- kernel $W^{\rm raw}_{ij}=(1-\lambda)e^{-D_{ij}/r}+\lambda e^{-\|s_i-s_j\|/\sigma_0}$, with $\sigma_0=1$ and $s_i\in\mathbb R^{d_s}$
- balanced propensity $\widetilde W_{ij}=\tfrac12 W^{\rm raw}_{ij}\big(1/\sum_k W^{\rm raw}_{ik}+1/\sum_k W^{\rm raw}_{jk}\big)$, recomputed at every step
- intensity $\Lambda_{ij}=\langle s\rangle_{\rm obs}\widetilde W_{ij}$, so the predicted total equals the observed total
- Poisson NLL plus $\tfrac{\gamma}{2}(\ln r-\ln r_0)^2$, with $\gamma=0.25$ and $r_0 = 2\times$ the mean nearest-neighbour distance
- spectral residual initialisation of $S$
- Adam (40 steps, lr 0.04), then L-BFGS (40 iterations, strong Wolfe)
- profile-likelihood curvature at $\hat\lambda\pm0.04$, giving $\sigma_\lambda$ and a 95% CI

**Output.** `results/fits_<coords>_ds<d_s>.csv` has one row per fit: λ̂, σ_λ, CI, r̂, r₀, loss, edges, synapses, runtime. The run can be resumed. `summary_by_system.csv` holds the mean, std (ddof=0), SE and median per system, plus the fraction of fits with a valid (positive-curvature) profile.

## Notes
- `sigma_lambda` is `NaN` when the profile curvature at λ̂ is not positive. On fly subgraphs this happens in most fits with the default 40 + 40 iterations, so check `frac_profile_valid` before quoting profile CIs.
- GPU and CPU runs with the same seed can differ slightly in λ̂ (floating-point order in L-BFGS).
- Timing: about 1 s per fit on a T4 GPU (13 × 50 fits ≈ 10 min per coordinate variant).
