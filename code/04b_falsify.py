"""Stage 4b: falsification checks for the two claims a reviewer will attack.

CHECK 1  Is the spectral model undertrained?  Sweep capacity and learning
         rate.  If the best spectral model is still far below the full
         structural model, the gap is a property of the descriptor set,
         not of the fit.

CHECK 2  Is the structural advantage just a ring prior?  Measure the
         incremental value of each structural feature and refit with the
         ring features removed.  If the model still beats the spectral
         model, the advantage is not only "avoid the ring".

CHECK 3  Does the spectral block help at all once non-spectral topology is
         present?  (Is the spectral contribution subsumed?)

CHECK 4  Random baseline inside the theorem's non-qualifying stratum, so
         the "below chance" statement is measured against the right pool.
"""
import json
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import warnings
warnings.filterwarnings('ignore')

SPECTRAL = ['btw', 'btw_rank', 'df', 'df_rank', 'fsum', 'fabs', 'cross',
            'res', 'bridge', 'lam2']
TOPO = ['deg_u', 'deg_v', 'deg_min', 'deg_max', 'in_cycle', 'detour',
        'comp_edges', 'core_min', 'core_max', 'tri_u', 'tri_v',
        'n', 'm', 'cyc_rank', 'deg_std']
RING_FEATS = ['in_cycle', 'detour', 'comp_edges', 'core_min', 'core_max',
              'tri_u', 'tri_v']
ALL = SPECTRAL + TOPO


def load(p):
    with open(p, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh]


def build(recs, ck, sk):
    Xc, Xs, y = [], [], []
    for r in recs:
        for b in r['bonds']:
            Xc.append([b['chem'][k] for k in ck])
            Xs.append([b['struct'][k] for k in sk])
            y.append(1 if b['truth'] else 0)
    return np.asarray(Xc, float), np.asarray(Xs, float), np.asarray(y, int)


def fit(X, y, it=400, lr=0.06, leaf=31):
    return HistGradientBoostingClassifier(
        max_iter=it, learning_rate=lr, max_leaf_nodes=leaf,
        min_samples_leaf=40, l2_regularization=1.0,
        class_weight='balanced', random_state=0,
        early_stopping=False).fit(X, y)


def acc(recs, scores):
    out, pos = [], 0
    for r in recs:
        n = len(r['bonds'])
        s = scores[pos:pos + n]
        pos += n
        out.append(1.0 if r['bonds'][int(np.argmax(s))]['truth'] else 0.0)
    return np.array(out)


def main(train_p, test_p, out_p):
    tr, te = load(train_p), load(test_p)
    ck = sorted(tr[0]['bonds'][0]['chem'].keys())
    Xc_tr, Xs_tr, y_tr = build(tr, ck, ALL)
    Xc_te, Xs_te, y_te = build(te, ck, ALL)
    ix = {k: i for i, k in enumerate(ALL)}
    ic = {k: i for i, k in enumerate(ck)}
    res = {}

    # ---- CHECK 1: capacity sweep on the spectral block ----------------
    cols = [ix[k] for k in SPECTRAL]
    sweep = []
    for it, lr, leaf in [(200, 0.05, 15), (400, 0.06, 31), (800, 0.03, 63),
                         (1500, 0.02, 127), (3000, 0.01, 255)]:
        m = fit(Xs_tr[:, cols], y_tr, it, lr, leaf)
        a = acc(te, m.predict_proba(Xs_te[:, cols])[:, 1]).mean()
        sweep.append(dict(it=it, lr=lr, leaf=leaf, acc=float(a)))
        print('CHECK1 spectral  it=%4d lr=%.2f leaf=%3d -> %.4f'
              % (it, lr, leaf, a))
    res['spectral_sweep'] = sweep
    res['spectral_best'] = max(s['acc'] for s in sweep)

    # ---- CHECK 1b: same sweep on the full structural block -----------
    cols_all = [ix[k] for k in ALL]
    sweep2 = []
    for it, lr, leaf in [(400, 0.06, 31), (1500, 0.02, 127)]:
        m = fit(Xs_tr[:, cols_all], y_tr, it, lr, leaf)
        a = acc(te, m.predict_proba(Xs_te[:, cols_all])[:, 1]).mean()
        sweep2.append(dict(it=it, lr=lr, leaf=leaf, acc=float(a)))
        print('CHECK1 struct    it=%4d lr=%.2f leaf=%3d -> %.4f'
              % (it, lr, leaf, a))
    res['struct_sweep'] = sweep2

    # ---- CHECK 2: structural block with ring features removed --------
    topo_noring = [k for k in TOPO if k not in RING_FEATS]
    for name, keys in [('struct_noring', topo_noring + SPECTRAL),
                       ('struct_noring_nospectral', topo_noring),
                       ('topo_without_cycle_detour',
                        [k for k in ALL if k not in ('in_cycle', 'detour')])]:
        cols = [ix[k] for k in keys]
        m = fit(Xs_tr[:, cols], y_tr)
        a = acc(te, m.predict_proba(Xs_te[:, cols])[:, 1]).mean()
        res[name] = dict(acc=float(a), n_feats=len(cols))
        print('CHECK2 %-26s %2d feats -> %.4f' % (name, len(cols), a))

    # ---- CHECK 3: full chemical + full structural, minus spectral ----
    combos = {
        'chem+struct': ck + ALL,
    }
    Xtr = np.hstack([Xc_tr, Xs_tr])
    Xte = np.hstack([Xc_te, Xs_te])
    m = fit(Xtr, y_tr)
    res['chem_plus_struct'] = float(
        acc(te, m.predict_proba(Xte)[:, 1]).mean())

    cs = list(range(len(ck))) + [len(ck) + ix[k] for k in TOPO]
    m = fit(Xtr[:, cs], y_tr)
    res['chem_plus_topo_nospectral'] = float(
        acc(te, m.predict_proba(Xte[:, cs])[:, 1]).mean())

    ct = list(range(len(ck))) + [len(ck) + ix[k] for k in SPECTRAL]
    m = fit(Xtr[:, ct], y_tr)
    res['chem_plus_spectral'] = float(
        acc(te, m.predict_proba(Xte[:, ct])[:, 1]).mean())
    print('CHECK3 chem+struct %.4f | chem+topo %.4f | chem+spectral %.4f'
          % (res['chem_plus_struct'], res['chem_plus_topo_nospectral'],
             res['chem_plus_spectral']))

    # ---- CHECK 4: random baseline inside the theorem strata ----------
    ppm = load('../data/mol_test.jsonl')

    def sbe_pick(bonds):
        cand = [j for j, b in enumerate(bonds) if b['struct']['cross'] > 0.5]
        if not cand:
            return None
        return max(cand, key=lambda j: (bonds[j]['struct']['df'],
                                        bonds[j]['struct']['btw']))

    # pair strictly by position: product ids ('rid') repeat across reactions,
    # so keying by them would silently merge distinct molecules
    assert len(ppm) == len(te), (len(ppm), len(te))
    strata = {'qualified': [], 'unqualified': []}
    for f, row in zip(te, ppm):
        assert f['rid'] == row['rid']
        j = sbe_pick(f['bonds'])
        if j is None:
            continue
        qualified = (f['n_sbe'] == 1 and not f['zero_vertex'])
        strata['qualified' if qualified else 'unqualified'].append(row)
    res['strata_random'] = {}
    for k, g in strata.items():
        if not g:
            continue
        res['strata_random'][k] = dict(
            n=len(g),
            random=float(np.mean([r['random'] for r in g])),
            sbe=float(np.mean([r['sbe'] for r in g])),
            combined=float(np.mean([r['combined'] for r in g])))
        print('CHECK4 %-12s n=%-5d random=%.4f sbe=%.4f combined=%.4f'
              % (k, len(g), res['strata_random'][k]['random'],
                 res['strata_random'][k]['sbe'],
                 res['strata_random'][k]['combined']))

    json.dump(res, open(out_p, 'w'), indent=1)
    print('wrote', out_p)


if __name__ == '__main__':
    main(*sys.argv[1:4])
