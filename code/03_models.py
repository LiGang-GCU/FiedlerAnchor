"""Stage 3 (v2): block-restricted models, fitted on train, tuned on a held-out
validation split carved out of train, evaluated once on test.

Reviewer-driven changes over the first version:

  * hyperparameters are selected on a VALIDATION split, never on test.  The
    earlier version reported the maximum over test configurations, which is
    test-set selection.  Here every configuration is scored on validation and
    the winner is evaluated on test exactly once.
  * the BRICS selector is labelled for what it is: it ranks among BRICS
    candidate bonds using the Fiedler gradient, so it is not a pure BRICS
    ordering.
  * structural and spectral sweeps use the same budget.

Feature blocks:
  B  chemical   -- requires atom labels
  C  structural -- computable from the unlabelled adjacency matrix
  D  combined

The blocks are NOT information-disjoint: the chemical block contains labelled
neighbour counts whose sum is the degree on a hydrogen-suppressed graph.  The
manuscript states this and reports the decomposition as conditional
performance attribution, not as a partition of information.
"""
import json
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import warnings
warnings.filterwarnings('ignore')

STRUCT_KEYS = ['btw', 'btw_rank', 'df', 'df_rank', 'fsum_abs', 'fabs_sum',
               'cross', 'res', 'bridge', 'deg_min', 'deg_max', 'in_cycle',
               'detour', 'comp_edges', 'core_min', 'core_max', 'tri_min',
               'tri_max', 'n', 'm', 'lam2', 'cyc_rank', 'deg_std']

GRID = [dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15),
        dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31),
        dict(max_iter=800, learning_rate=0.03, max_leaf_nodes=63),
        dict(max_iter=1500, learning_rate=0.02, max_leaf_nodes=127)]


def load(path):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh]


def split_train_valid(recs, frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(recs))
    k = int(len(recs) * frac)
    return [recs[i] for i in idx[k:]], [recs[i] for i in idx[:k]]


def build(recs, ck):
    Xc, Xs, y = [], [], []
    for r in recs:
        for b in r['bonds']:
            Xc.append([b['chem'][k] for k in ck])
            Xs.append([b['struct'][k] for k in STRUCT_KEYS])
            y.append(1 if b['truth'] else 0)
    return np.asarray(Xc, float), np.asarray(Xs, float), np.asarray(y, int)


def fit(X, y, cfg):
    return HistGradientBoostingClassifier(
        min_samples_leaf=40, l2_regularization=1.0,
        class_weight='balanced', random_state=0, early_stopping=False,
        **cfg).fit(X, y)


def argmax_model(recs, scores):
    out, pos = [], 0
    for r in recs:
        n = len(r['bonds'])
        s = scores[pos:pos + n]
        pos += n
        b = r['bonds'][int(np.argmax(s))]
        out.append(dict(hit=1.0 if b['truth'] else 0.0, env=b['env'],
                        ring=b['ring'], abstain=0))
    return out


def pick_by(recs, fn):
    out = []
    for r in recs:
        j = fn(r['bonds'])
        if j is None:
            out.append(dict(hit=0.0, env=None, ring=None, abstain=1))
        else:
            b = r['bonds'][j]
            out.append(dict(hit=1.0 if b['truth'] else 0.0, env=b['env'],
                            ring=b['ring'], abstain=0))
    return out


def argmax_struct(key):
    return lambda bonds: int(np.argmax([b['struct'][key] for b in bonds]))


def sbe_rule(bonds):
    cand = [j for j, b in enumerate(bonds) if b['struct']['cross'] > 0.5]
    if not cand:
        return None
    return max(cand, key=lambda j: (bonds[j]['struct']['df'],
                                    bonds[j]['struct']['btw']))


def brics_rule(bonds):
    """Rank BRICS candidate bonds by Fiedler gradient, then betweenness.

    This is 'BRICS candidates + spectral ranking', not a pure BRICS ordering;
    the manuscript labels it accordingly.
    """
    cand = [j for j, b in enumerate(bonds) if b['brics']]
    if not cand:
        return None
    return max(cand, key=lambda j: (bonds[j]['struct']['df'],
                                    bonds[j]['struct']['btw']))


def tune(Xtr, ytr, Xva, yva, va_recs):
    """Pick the configuration with the best VALIDATION accuracy."""
    best, best_acc = None, -1.0
    log = []
    for cfg in GRID:
        m = fit(Xtr, ytr, cfg)
        a = float(np.mean([d['hit'] for d in
                           argmax_model(va_recs, m.predict_proba(Xva)[:, 1])]))
        log.append(dict(cfg=cfg, valid_acc=a))
        if a > best_acc:
            best, best_acc = cfg, a
    return best, best_acc, log


def main(train_p, test_p, out_p, log_p):
    all_tr = load(train_p)
    te = load(test_p)
    tr, va = split_train_valid(all_tr, frac=0.15, seed=0)
    ck = sorted(tr[0]['bonds'][0]['chem'].keys())

    Xc_tr, Xs_tr, y_tr = build(tr, ck)
    Xc_va, Xs_va, y_va = build(va, ck)
    Xc_te, Xs_te, y_te = build(te, ck)
    print('train %d mols | valid %d mols | test %d mols'
          % (len(tr), len(va), len(te)))

    log = {}
    cfgs = {}
    for name, Xtr, Xva in [('chem', Xc_tr, Xc_va),
                           ('struct', Xs_tr, Xs_va),
                           ('combined', np.hstack([Xc_tr, Xs_tr]),
                            np.hstack([Xc_va, Xs_va]))]:
        cfg, acc, trace = tune(Xtr, y_tr, Xva, y_va, va)
        cfgs[name] = cfg
        log[name] = dict(chosen=cfg, valid_acc=acc, trace=trace)
        print('  %-9s chose %s (valid %.4f)' % (name, cfg, acc))

    mB = fit(Xc_tr, y_tr, cfgs['chem'])
    mC = fit(Xs_tr, y_tr, cfgs['struct'])
    mD = fit(np.hstack([Xc_tr, Xs_tr]), y_tr, cfgs['combined'])

    sel = {}
    sel['random'] = [dict(hit=float(sum(1 for b in r['bonds'] if b['truth']) /
                                 len(r['bonds'])), env=None, ring=None,
                          abstain=0) for r in te]
    sel['betweenness'] = pick_by(te, argmax_struct('btw'))
    sel['resistance'] = pick_by(te, argmax_struct('res'))
    sel['fiedler_df'] = pick_by(te, argmax_struct('df'))
    sel['sbe'] = pick_by(te, sbe_rule)
    sel['brics'] = pick_by(te, brics_rule)
    sel['chem_model'] = argmax_model(te, mB.predict_proba(Xc_te)[:, 1])
    sel['struct_model'] = argmax_model(te, mC.predict_proba(Xs_te)[:, 1])
    sel['combined'] = argmax_model(
        te, mD.predict_proba(np.hstack([Xc_te, Xs_te]))[:, 1])

    print()
    for k, v in sel.items():
        print('  %-14s %.4f' % (k, np.mean([x['hit'] for x in v])))

    with open(out_p, 'w', encoding='utf-8') as fo:
        for i, r in enumerate(te):
            row = dict(rid=r['rid'], cls=r['cls'], n_atoms=r['n_atoms'],
                       n_bonds=len(r['bonds']),
                       n_truth=sum(1 for b in r['bonds'] if b['truth']),
                       n_truth_nonring=sum(1 for b in r['bonds']
                                           if b['truth'] and not b['ring']),
                       n_truth_ring=sum(1 for b in r['bonds']
                                        if b['truth'] and b['ring']),
                       lam2=r['lam2'], lam3=r['lam3'],
                       zero_vertex=r['zero_vertex'], n_sbe=r['n_sbe'],
                       has_nonring=any(not b['ring'] for b in r['bonds']),
                       has_hetero=any(b['env'] == 'hetero' for b in r['bonds']))
            for k, v in sel.items():
                row[k] = v[i]['hit']
                row[k + '_env'] = v[i].get('env')
                row[k + '_ring'] = v[i].get('ring')
                row[k + '_abstain'] = v[i].get('abstain', 0)
            fo.write(json.dumps(row) + '\n')
    json.dump(log, open(log_p, 'w'), indent=1)
    print('wrote', out_p, 'and', log_p)


if __name__ == '__main__':
    main(*sys.argv[1:5])
