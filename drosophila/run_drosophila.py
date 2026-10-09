"""Estimate lambda and r for each FlyWire major system with the estimator of the paper
(exact_inference.infer_parameters_exact, Methods Eqs. 1-9), on M random induced subgraphs of n neurons.

Per system and replica k = 1..M:
  seed_k  = k-th integer of default_rng(crc32(file stem))       (deterministic, identical across --coords)
  nodes   = default_rng(seed_k).choice(N, n, replace=False)     (uniform induced subgraph)
  fit     = infer_parameters_exact(X[nodes], W[nodes][:, nodes], soc_dim=d_s, seed=seed_k,
                                   compute_profile_ci=True)
Each fit is appended to <out>/fits_<coords>_ds<d_s>.csv as soon as it finishes; reruns skip finished fits.

Example
  python run_drosophila.py --npz fly_npz --coords primary_np --device cuda:0
  python run_drosophila.py --npz fly_npz --coords syn_centroid --device cuda:1   # robustness check"""
import argparse, os, sys, time, zlib, numpy as np, pandas as pd, scipy.sparse as sp, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from exact_inference import infer_parameters_exact

ap = argparse.ArgumentParser()
ap.add_argument('--npz', default='fly_npz', help='output folder of prepare_flywire.py')
ap.add_argument('--coords', default='primary_np', choices=['primary_np', 'syn_centroid'])
ap.add_argument('--systems', nargs='*', help='npz file stems (default: all)')
ap.add_argument('--M', type=int, default=50); ap.add_argument('--n', type=int, default=600)
ap.add_argument('--ds', type=int, default=3); ap.add_argument('--no-profile', action='store_true')
ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
ap.add_argument('--out', default='results')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

systems = a.systems or sorted(f[:-4] for f in os.listdir(a.npz) if f.endswith('.npz'))
out = f'{a.out}/fits_{a.coords}_ds{a.ds}.csv'
done = set(map(tuple, pd.read_csv(out)[['system', 'replica']].values)) if os.path.exists(out) else set()

data = {}
for s in systems:
    z = np.load(f'{a.npz}/{s}.npz', allow_pickle=True); X = z[f'X_{a.coords}'].astype(float); N = len(X)
    W = sp.coo_matrix((z['w'], (z['i'], z['j'])), shape=(N, N)).tocsr(); W = (W + W.T).tocsr()
    rng = np.random.default_rng(zlib.crc32(s.encode()))
    data[s] = (X, W, [int(rng.integers(0, 10**8)) for _ in range(a.M)])

for k in range(1, a.M + 1):                       # round-robin: partial output stays balanced
    for s in systems:
        if (s, k) in done: continue
        X, W, seeds = data[s]; seed = seeds[k - 1]; N = len(X)
        idx = np.random.default_rng(seed).choice(N, size=min(a.n, N), replace=False)
        Ws = W[idx][:, idx].toarray(); t = time.time()
        f = infer_parameters_exact(X[idx], Ws, soc_dim=a.ds, device=a.device, seed=seed,
                                   compute_profile_ci=not a.no_profile)
        lo, hi = f.get('ci_95_lambda', (np.nan, np.nan))
        row = dict(system=s, coords=a.coords, d_s=a.ds, replica=k, seed=seed, N_system=N, n=len(idx),
                   edges=int((Ws > 0).sum() // 2), synapses=float(Ws.sum() / 2),
                   lambda_hat=f['lambda'], sigma_lambda=f.get('sigma_lambda', np.nan), ci_low=lo, ci_high=hi,
                   r_hat=f['r'], r0=f['r0'], loss=f['loss'], runtime_s=time.time() - t)
        pd.DataFrame([row]).to_csv(out, mode='a', header=not os.path.exists(out), index=False)
        print(f"{s:40s} k={k:2d} lambda={row['lambda_hat']:.4f} r={row['r_hat']:.2f} um "
              f"sigma={row['sigma_lambda']:.3g} {row['runtime_s']:.1f}s", flush=True)
