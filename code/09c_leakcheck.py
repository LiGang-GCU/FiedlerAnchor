"""Stage 9c: leakage control for the GNN branch.

A same-architecture GNN reaches 0.72 (chemical) and 0.81 (chemical+structural)
where the hand-crafted blocks reached 0.53 and 0.64.  A jump that large is
either a real representation effect or a leak, and the only way to tell is to
break the label-to-input association while keeping everything else identical.

The control permutes, within each molecule, the mapping from edge to label.
The marginal label rate, the graph, the features and the training schedule are
unchanged; only the correspondence between a bond and "was this bond cleaved"
is destroyed.  A clean pipeline must fall to the analytic random baseline.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from importlib import import_module
_g = import_module('09_gnn')


def shuffled_copy(graphs, seed):
    rng = np.random.default_rng(seed)
    out = []
    for g in graphs:
        y = g.y.clone()
        perm = torch.from_numpy(rng.permutation(len(y)).astype(np.int64))
        g2 = g.clone()
        g2.y = y[perm]
        out.append(g2)
    return out


def main(train_p, test_p, variant='both', epochs=12, seed=7):
    # Build once, then permute labels in BOTH splits.
    tr_all = _g.load_split(train_p, variant)
    te = _g.load_split(test_p, variant)
    tr_all = shuffled_copy(tr_all, seed)
    te = shuffled_copy(te, seed + 1)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(tr_all))
    k = int(0.15 * len(tr_all))
    va = [tr_all[i] for i in idx[:k]]
    tr = [tr_all[i] for i in idx[k:]]

    from torch_geometric.loader import DataLoader
    tr_loader = DataLoader(tr, batch_size=32, shuffle=True)
    va_loader = DataLoader(va, batch_size=32)
    te_loader = DataLoader(te, batch_size=32)

    in_dim = tr[0].x.shape[1]
    edge_dim = tr[0].edge_attr.shape[1]
    model = _g.Net(in_dim, edge_dim).to(_g.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    n_pos = sum(float(g.y.sum()) for g in tr)
    n_tot = sum(float(g.y.numel()) for g in tr)
    pos_weight = (n_tot - n_pos) / max(n_pos, 1.0)

    best, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        loss = _g.train_epoch(model, tr_loader, opt, _g.DEV, pos_weight)
        vacc = _g.per_molecule_hits(model, va_loader, _g.DEV).mean()
        print('  ep %2d loss %.4f valid %.4f' % (ep, loss, vacc))
        if vacc > best:
            best = vacc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    hits = _g.per_molecule_hits(model, te_loader, _g.DEV)

    base = np.mean([float(g.y.numel() and g.y.mean()) for g in te])
    print()
    print('SHUFFLED-LABEL CONTROL  variant=%s' % variant)
    print('  best validation      %.4f' % best)
    print('  test accuracy        %.4f' % hits.mean())
    print('  random baseline      %.4f' % base)
    ok = hits.mean() < base + 0.05
    print('  %s' % ('PASS: collapses to chance, so the reported accuracy '
                    'is not a leak' if ok else
                    'FAIL: still above chance with destroyed labels -- '
                    'there is a leak to find'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2],
                  sys.argv[3] if len(sys.argv) > 3 else 'both'))
