"""Summarise the fits of run_drosophila.py per system and plot the lambda ranking.
  python analyze_drosophila.py results/fits_primary_np_ds3.csv [results/fits_syn_centroid_ds3.csv ...]
Writes summary_<input>.csv and lambda_by_system.{png,pdf} next to the first input."""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

d = pd.concat([pd.read_csv(f) for f in sys.argv[1:]]); outdir = os.path.dirname(os.path.abspath(sys.argv[1]))
summ = (d.groupby(['coords', 'd_s', 'system'])
         .agg(N=('N_system', 'first'), M=('lambda_hat', 'size'),
              lambda_mean=('lambda_hat', 'mean'), lambda_std=('lambda_hat', lambda x: x.std(ddof=0)),
              lambda_median=('lambda_hat', 'median'),
              r_mean=('r_hat', 'mean'), r_std=('r_hat', lambda x: x.std(ddof=0)), r_median=('r_hat', 'median'),
              sigma_lambda_median=('sigma_lambda', 'median'),
              frac_profile_valid=('sigma_lambda', lambda x: np.isfinite(x).mean()),
              edges_mean=('edges', 'mean'))
         .reset_index())
summ['lambda_se'] = summ.lambda_std / np.sqrt(summ.M)
summ.round(5).to_csv(f'{outdir}/summary_by_system.csv', index=False); print(summ.round(3).to_string(index=False))

order = summ.groupby('system').lambda_mean.mean().sort_values().index; y = np.arange(len(order))
fig, ax = plt.subplots(figsize=(7, 0.35 * len(order) + 1.2))
for j, ((c, ds), s) in enumerate(summ.groupby(['coords', 'd_s'])):
    s = s.set_index('system').loc[order]
    ax.errorbar(s.lambda_mean, y + 0.15 * j, xerr=s.lambda_std, fmt='o', ms=4, capsize=2, label=f'{c}, d_s={ds}')
ax.set_yticks(y); ax.set_yticklabels([o.replace('_', ' ') for o in order]); ax.set_xlim(0, 1.02)
ax.set_xlabel(r'$\hat\lambda$ (mean $\pm$ std over subgraphs)'); ax.legend(fontsize=8, frameon=False); ax.grid(axis='x', alpha=.3)
fig.tight_layout(); fig.savefig(f'{outdir}/lambda_by_system.png', dpi=200); fig.savefig(f'{outdir}/lambda_by_system.pdf')
