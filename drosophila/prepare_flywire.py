"""Build one weighted, undirected network per FlyWire major brain system.

Inputs
  --synapses   flywire_synapses_783.feather   (FlyWire FAFB v783, Zenodo record 10676866;
               one row per synapse, FlyWire's cleft_score >= 50 filter already applied)
  --neurons    drosophila_neuron_coordinates.csv  (one row per proofread neuron: root_id,
               major_system, x_um/y_um/z_um = centroid of all its synapses,
               primary_np_{x,y,z}_um = centroid of its synapses in its primary neuropil)
Output (--out)
  <System>.npz      nodes (root ids), X_primary_np, X_syn_centroid (um), upper-triangular edge list i<j, w
  prep_summary.csv  neurons, edges and synapses per system

Edge weight: W_ij = #synapses i->j + #synapses j->i between two distinct neurons of the SAME system.
Synapses between different systems and autapses are discarded."""
import argparse, os, numpy as np, pandas as pd, pyarrow as pa, pyarrow.ipc as ipc

ap = argparse.ArgumentParser()
ap.add_argument('--synapses', required=True)
ap.add_argument('--neurons', required=True)
ap.add_argument('--out', default='fly_npz')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

nd = pd.read_csv(a.neurons, usecols=['root_id', 'major_system', 'x_um', 'y_um', 'z_um',
                                     'primary_np_x_um', 'primary_np_y_um', 'primary_np_z_um'])
nd = nd.sort_values('root_id').reset_index(drop=True)
rid = nd.root_id.values.astype(np.int64); N = len(rid)

# 1. directed synapse counts between proofread neurons, streamed batch by batch (the table is ~9.5 GB)
reader = ipc.open_file(pa.memory_map(a.synapses))
keys, counts = [], []
def compact(keys, counts):
    k = np.concatenate(keys); c = np.concatenate(counts)
    u, inv = np.unique(k, return_inverse=True)
    return [u], [np.bincount(inv, weights=c).astype(np.int64)]
for b in range(reader.num_record_batches):
    t = reader.get_batch(b)
    pre = t.column('pre_pt_root_id').to_numpy(); post = t.column('post_pt_root_id').to_numpy()
    ip = np.minimum(np.searchsorted(rid, pre), N - 1); jp = np.minimum(np.searchsorted(rid, post), N - 1)
    ok = (rid[ip] == pre) & (rid[jp] == post) & (pre != post)
    k, c = np.unique(ip[ok].astype(np.int64) * N + jp[ok], return_counts=True)
    keys.append(k); counts.append(c)
    if len(keys) >= 20: keys, counts = compact(keys, counts)
keys, counts = compact(keys, counts)
K, C = keys[0], counts[0]; I, J = K // N, K % N

# 2. per-system undirected networks
rows = []
for s in sorted(nd.major_system.unique()):
    nodes = np.where(nd.major_system.values == s)[0]; n = len(nodes)
    loc = -np.ones(N, np.int64); loc[nodes] = np.arange(n)
    m = (loc[I] >= 0) & (loc[J] >= 0)
    i, j = loc[I[m]], loc[J[m]]
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    u, inv = np.unique(lo * n + hi, return_inverse=True); w = np.bincount(inv, weights=C[m])
    name = s.replace(' ', '_').replace('&', 'and').replace('(', '').replace(')', '').replace('/', '-')
    np.savez_compressed(f'{a.out}/{name}.npz', system=s, nodes=rid[nodes], i=u // n, j=u % n, w=w,
        X_primary_np=nd.loc[nodes, ['primary_np_x_um', 'primary_np_y_um', 'primary_np_z_um']].values,
        X_syn_centroid=nd.loc[nodes, ['x_um', 'y_um', 'z_um']].values)
    rows.append(dict(system=s, file=f'{name}.npz', neurons=n, edges=len(u), synapses=int(w.sum())))
summary = pd.DataFrame(rows); summary.to_csv(f'{a.out}/prep_summary.csv', index=False); print(summary.to_string(index=False))
