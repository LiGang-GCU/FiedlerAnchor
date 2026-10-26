"""Stage 4 (v2): feature-subset ablation, with validation-based tuning.

Reviewer-driven changes:

  * every configuration is selected on a validation split carved out of
    train, then evaluated once on test.  The previous version reported the
    maximum over test accuracies, which is test-set selection.
  * the structural and spectral sweeps receive the SAME budget, so the
    capacity comparison is symmetric.
  * conditional increments are reported in a form that does not reverse
    direction: adding topology TO chemistry+spectral, and adding spectral TO
    chemistry+topology, are both computed from the joint model.
  * the "ring removal" check now removes the bridge flag as well, because on
    a connected undirected graph "in a cycle" and "not a bridge" are the same
    predicate; leaving the bridge flag in would leave exact ring membership
    accessible.
"""
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import warnings
warnings.filterwarnings('ignore')

SPECTRAL = ['btw', 'btw_rank', 'df', 'df_rank', 'fsum_abs', 'fabs_sum',
            'cross', 'res', 'lam2']
RING_DIRECT = ['in_cycle', 'detour', 'bridge']
RING_INDIRECT = ['comp_edges', 'core_min', 'core_max', 'tri_min', 'tri_max']
TOPO = ['deg_min', 'deg_max', 'in_cycle', 'detour', 'comp_edges',
        'core_min', 'core_max', 'tri_min', 'tri_max', 'n', 'm', 'cyc_rank',
        'deg_std', 'bridge']
ALL = SPECTRAL + TOPO

# 03_models.py selected this configuration on the validation split for all
# three blocks (chem, struct, combined), so the ablation holds capacity fixed
# at the validation-chosen setting instead of re-running the grid per block.
GRID = [dict(max_iter=1500, learning_rate=0.02, max_leaf_nodes=127)]


def load(path):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh]


def split_tr_va(recs, frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(recs))
    k = int(len(recs) * frac)
    return [recs[i] for i in idx[k:]], [recs[i] for i in idx[:k]]


def build(recs, ck, sk):
    Xc, Xs, y = [], [], []
    for r in recs:
        for b in r['bonds']:
            Xc.append([b['chem'][k] for k in ck])
            Xs.append([b['struct'][k] for k in sk])
            y.append(1 if b['truth'] else 0)
    return np.asarray(Xc, float), np.asarray(Xs, float), np.asarray(y, int)


def fit(X, y, cfg):
    return HistGradientBoostingClassifier(
        min_samples_leaf=40, l2_regularization=1.0,
        class_weight='balanced', random_state=0, early_stopping=False,
        **cfg).fit(X, y)


def acc(recs, scores):
    out, pos = [], 0
    for r in recs:
        n = len(r['bonds'])
        s = scores[pos:pos + n]
        pos += n
        out.append(1.0 if r['bonds'][int(np.argmax(s))]['truth'] else 0.0)
    return np.array(out)


def tuned_fit_predict(Xtr, ytr, Xva, yva, va_recs, Xte, te):
    best, best_acc, best_pred = None, -1.0, None
    for cfg in GRID:
        m = fit(Xtr, ytr, cfg)
        a = float(acc(va_recs, m.predict_proba(Xva)[:, 1]).mean())
        if a > best_acc:
            best, best_acc = cfg, a
            best_pred = acc(te, m.predict_proba(Xte)[:, 1])
    return best_pred, best, best_acc


def main(train_p, test_p, out_p):
    all_tr, te = load(train_p), load(test_p)
    tr, va = split_tr_va(all_tr, frac=0.15, seed=0)
    ck_all = sorted(tr[0]['bonds'][0]['chem'].keys())
    ck_bond = [k for k in ck_all if k.startswith('b_')]
    ck_atom = [k for k in ck_all if k.startswith('a1_') or k.startswith('a2_')]

    def build_multi(recs, ck, sk):
        Xc, Xs, y = build(recs, ck, sk)
        return Xc, Xs, y

    res = {}
    cols = {}
    for name, keys in [
            ('S:spectral', SPECTRAL),
            ('S:topology', [k for k in TOPO if k not in RING_DIRECT]),
            ('S:struct_all', ALL),
            ('S:noring_direct', [k for k in ALL if k not in RING_DIRECT]),
            ('S:noring_all', [k for k in ALL
                              if k not in RING_DIRECT + RING_INDIRECT]),
    ]:
        # build() returns (Xc, Xs, y); the structural block is the SECOND
        # element.  An earlier version took the first here, so every S: row
        # was silently scoring the chemical matrix.
        _, Xtr, ytr = build(tr, ck_all, keys)
        _, Xva, yva = build(va, ck_all, keys)
        _, Xte, yte = build(te, ck_all, keys)
        pred, cfg, vacc = tuned_fit_predict(Xtr, ytr, Xva, yva, va, Xte, te)
        res[name] = pred
        cols[name] = keys
        print('%-22s %2d feats  valid %.4f  test %.4f  %s'
              % (name, len(keys), vacc, pred.mean(), cfg))

    if os.environ.get('ONLY_S'):
        # recompute only the structural rows and merge into the existing file;
        # the C: and D: rows were built from the correct matrices already.
        old = json.load(open(out_p))
        for k, v in res.items():
            old[k] = v.tolist()
        assert old['n'] == len(te)
        json.dump(old, open(out_p, 'w'))
        print('merged S: rows into', out_p)
        return

    for name, keys in [
            ('C:bond_order', ck_bond),
            ('C:atom_identity', ck_atom),
            ('C:chem_all', ck_all)]:
        Xtr, _, ytr = build(tr, keys, ALL)
        Xva, _, yva = build(va, keys, ALL)
        Xte, _, yte = build(te, keys, ALL)
        pred, cfg, vacc = tuned_fit_predict(Xtr, ytr, Xva, yva, va, Xte, te)
        res[name] = pred
        print('%-22s %2d feats  valid %.4f  test %.4f  %s'
              % (name, len(keys), vacc, pred.mean(), cfg))

    COMBOS = {
        'D:chem+spectral': (ck_all, SPECTRAL),
        'D:chem+topology': (ck_all, [k for k in TOPO if k not in RING_DIRECT]),
        'D:chem+struct': (ck_all, ALL),
    }
    for name, (cK, sK) in COMBOS.items():
        Xc_tr, Xs_tr, ytr = build(tr, cK, sK)
        Xc_va, Xs_va, yva = build(va, cK, sK)
        Xc_te, Xs_te, yte = build(te, cK, sK)
        Xtr = np.hstack([Xc_tr, Xs_tr])
        Xva = np.hstack([Xc_va, Xs_va])
        Xte = np.hstack([Xc_te, Xs_te])
        pred, cfg, vacc = tuned_fit_predict(Xtr, ytr, Xva, yva, va, Xte, te)
        res[name] = pred
        print('%-22s %2d feats  valid %.4f  test %.4f  %s'
              % (name, len(cK) + len(sK), vacc, pred.mean(), cfg))

    A = np.array([sum(1 for b in r['bonds'] if b['truth']) / len(r['bonds'])
                  for r in te])
    res['A:random'] = A

    out = dict(n=len(te), **{k: v.tolist() for k, v in res.items()})
    json.dump(out, open(out_p, 'w'))
    print('wrote', out_p)


if __name__ == '__main__':
    main(*sys.argv[1:4])
