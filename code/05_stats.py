"""Stage 5: statistics and the single source of truth for every number.

Reads data/mol_test.jsonl (per-molecule hits from stage 3) and
data/ablation.json (per-molecule hits from stage 4) and writes
data/numbers.json.  LaTeX reads only numbers.json, via 05_emit_tex.py.

Nothing in the manuscript is typed by hand: if a number is not in
numbers.json it does not appear in the paper.

Statistics performed here:
  * bootstrap CI for every selector's accuracy (molecule-level resampling,
    paired across selectors so differences get paired intervals)
  * exact McNemar on every pair of interest
  * commonality decomposition of the accuracy lift over random into
    unique-chemical, unique-structural and shared parts
  * stratified versions of the main comparison
"""
import json
import sys
import collections

import numpy as np
from scipy.stats import binomtest

RNG = np.random.default_rng(20261022)
NBOOT = 4000

ORDER = ['random', 'resistance', 'sbe', 'betweenness', 'fiedler_df', 'brics',
         'struct_model', 'chem_model', 'combined']

NICE = {'random': 'Uniform random (analytic)',
        'resistance': 'Effective resistance, argmax',
        'sbe': 'Fiedler sign boundary (SBE)',
        'betweenness': 'Edge betweenness, argmax',
        'fiedler_df': 'Fiedler gradient $|\\Delta f|$, argmax',
        'brics': 'BRICS',
        'struct_model': 'Structural model (unlabelled graph)',
        'chem_model': 'Chemical model (atom labels)',
        'combined': 'Chemical + structural model'}


def load(p):
    with open(p, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh]


def boot_ci(x, rng, nboot=NBOOT):
    x = np.asarray(x, float)
    n = len(x)
    idx = rng.integers(0, n, size=(nboot, n))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    m = n01 + n10
    p = 1.0 if m == 0 else float(binomtest(n10, m, 0.5).pvalue)
    return dict(a_only=n10, b_only=n01, n_disc=m, p=p)


def paired_diff_ci(a, b, rng, nboot=NBOOT):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    idx = rng.integers(0, n, size=(nboot, n))
    m = d[idx].mean(axis=1)
    return (float(d.mean()), float(np.percentile(m, 2.5)),
            float(np.percentile(m, 97.5)))


def main(mol_p, abl_p, out_p):
    recs = load(mol_p)
    n = len(recs)
    abl = json.load(open(abl_p))

    R = {}
    R['n_test_molecules'] = n
    R['n_test_bonds'] = sum(r['n_bonds'] for r in recs)
    R['n_test_truth'] = sum(r['n_truth'] for r in recs)

    # ---- main selector table ----------------------------------------
    main_tab = []
    for k in ORDER:
        x = np.array([r[k] for r in recs], float)
        lo, hi = boot_ci(x, RNG)
        main_tab.append(dict(
            key=k, label=NICE[k], acc=float(x.mean()), lo=lo, hi=hi,
            abstain=int(sum(r.get(k + '_abstain', 0) for r in recs))))
    R['main'] = main_tab
    M = {d['key']: d for d in main_tab}

    # ---- pairwise tests ---------------------------------------------
    pairs = [('betweenness', 'fiedler_df'), ('sbe', 'betweenness'),
             ('struct_model', 'betweenness'), ('struct_model', 'chem_model'),
             ('combined', 'chem_model'), ('combined', 'struct_model'),
             ('chem_model', 'betweenness'), ('brics', 'fiedler_df')]
    R['pairs'] = {}
    for a, b in pairs:
        xa = [r[a] for r in recs]
        xb = [r[b] for r in recs]
        t = mcnemar(xa, xb)
        d, lo, hi = paired_diff_ci(xa, xb, RNG)
        t.update(diff=d, lo=lo, hi=hi)
        R['pairs']['%s__%s' % (a, b)] = t

    # ---- commonality decomposition ----------------------------------
    A = M['random']['acc']
    Bchem = M['chem_model']['acc']
    Cstr = M['struct_model']['acc']
    Dcmb = M['combined']['acc']
    total = Dcmb - A
    uniq_b = Dcmb - Cstr
    uniq_c = Dcmb - Bchem
    shared = Bchem + Cstr - Dcmb - A
    R['decomp'] = dict(A=float(A), B=float(Bchem), C=float(Cstr), D=float(Dcmb),
                       total=float(total),
                       unique_chem=float(uniq_b), unique_struct=float(uniq_c),
                       shared=float(shared),
                       sum=float(uniq_b + uniq_c + shared),
                       shared_frac=float(shared / total),
                       chem_frac=float(uniq_b / total),
                       struct_frac=float(uniq_c / total),
                       additive=float(Bchem + Cstr - A),
                       interaction=float(Dcmb - (Bchem + Cstr - A)))
    # bootstrap the decomposition jointly (molecule-level, paired)
    boots = []
    arrays = {k: np.array([r[k] for r in recs], float)
              for k in ('random', 'chem_model', 'struct_model', 'combined')}
    for _ in range(NBOOT):
        idx = RNG.integers(0, n, n)
        mm = {k: v[idx].mean() for k, v in arrays.items()}
        tot = mm['combined'] - mm['random']
        ub = mm['combined'] - mm['struct_model']
        uc = mm['combined'] - mm['chem_model']
        sh = mm['chem_model'] + mm['struct_model'] - mm['combined'] - mm['random']
        boots.append([tot, ub, uc, sh])
    boots = np.array(boots)
    R['decomp_ci'] = dict(
        total=[float(np.percentile(boots[:, 0], 2.5)),
               float(np.percentile(boots[:, 0], 97.5))],
        unique_chem=[float(np.percentile(boots[:, 1], 2.5)),
                     float(np.percentile(boots[:, 1], 97.5))],
        unique_struct=[float(np.percentile(boots[:, 2], 2.5)),
                       float(np.percentile(boots[:, 2], 97.5))],
        shared=[float(np.percentile(boots[:, 3], 2.5)),
                float(np.percentile(boots[:, 3], 97.5))])

    # ---- spectral vs topology vs full structural --------------------
    abl_tab = []
    AKEYS = [('A:random', 'Uniform random (analytic)'),
             ('S:spectral', 'Spectral descriptors only'),
             ('S:topology', 'Non-spectral topology only'),
             ('S:struct_all', 'All unlabelled-graph features'),
             ('S:noring_direct', 'Structural, ring features removed'),
             ('S:noring_all', 'Structural, ring features and proxies removed'),
             ('C:bond_order', 'Bond order block only'),
             ('C:atom_identity', 'Atom identity block only'),
             ('C:chem_all', 'All chemical features'),
             ('D:chem+spectral', 'Chemical + spectral'),
             ('D:chem+topology', 'Chemical + non-spectral topology'),
             ('D:chem+struct', 'Chemical + all structural')]
    for k, lab in AKEYS:
        x = np.array(abl[k], float)
        lo, hi = boot_ci(x, RNG)
        abl_tab.append(dict(key=k, label=lab, acc=float(x.mean()),
                            lo=lo, hi=hi))
    R['ablation'] = abl_tab
    AM = {d['key']: d['acc'] for d in abl_tab}
    for a, b in [('S:struct_all', 'S:spectral'),
                 ('S:struct_all', 'S:topology'),
                 ('D:chem+struct', 'D:chem+spectral'),
                 ('D:chem+struct', 'D:chem+topology')]:
        d, lo, hi = paired_diff_ci(abl[a], abl[b], RNG)
        R['pairs']['%s__%s' % (a, b)] = dict(
            diff=d, lo=lo, hi=hi, **mcnemar(abl[a], abl[b]))

    # ---- ring-truth stratification ----------------------------------
    groups = {
        'all_nonring_truth': [r for r in recs
                              if any(b for b in [r['n_truth_nonring']])
                              and r['n_truth_nonring'] == r['n_truth']],
        'has_ring_truth': [r for r in recs if r['n_truth_nonring'] < r['n_truth']],
        'ring_only_truth': [r for r in recs if r['n_truth_nonring'] == 0],
    }
    R['strata'] = {}
    for gname, g in groups.items():
        if not g:
            continue
        row = dict(n=len(g))
        for k in ['random', 'betweenness', 'fiedler_df', 'brics',
                  'struct_model', 'chem_model', 'combined']:
            row[k] = float(np.mean([r[k] for r in g]))
        R['strata'][gname] = row

    # ---- theorem scope analysis ---------------------------------------
    # The bridge theorem is stated for |SBE| = 1 with no zero-component
    # vertex.  We partition the test population on exactly that hypothesis
    # and measure what the SBE selector does on each side.
    # feat records are written in the same order as the per-molecule records;
    # pair by position because rid repeats across reactions of one patent
    feat_list = load(mol_p.replace('mol_test', 'feat_test'))
    assert len(feat_list) == len(recs), (len(feat_list), len(recs))
    for a, b in zip(feat_list, recs):
        assert a['rid'] == b['rid']

    def sbe_pick(bonds):
        cand = [j for j, b in enumerate(bonds) if b['struct']['cross'] > 0.5]
        if not cand:
            return None
        return max(cand, key=lambda j: (bonds[j]['struct']['df'],
                                        bonds[j]['struct']['btw']))

    # Three strata, not two: the rule abstains when no edge crosses the sign
    # boundary, and an earlier version silently dropped those molecules so
    # that the reported strata summed to less than the test set.
    qual, unq, abst = [], [], []
    for row, f in zip(recs, feat_list):
        j = sbe_pick(f['bonds'])
        if j is None:
            abst.append(dict(ring=False, truth=False, n_sbe=0,
                             zero=bool(row['zero_vertex']), row=row))
            continue
        rec = dict(ring=bool(f['bonds'][j]['ring']),
                   truth=bool(f['bonds'][j]['truth']),
                   n_sbe=len([b for b in f['bonds'] if b['struct']['cross'] > 0.5]),
                   zero=bool(row['zero_vertex']),
                   # conditional baselines inside the same stratum
                   df_hit=bool(f['bonds'][int(np.argmax(
                       [b['struct']['df'] for b in f['bonds']]))]['truth']),
                   btw_hit=bool(f['bonds'][int(np.argmax(
                       [b['struct']['btw'] for b in f['bonds']]))]['truth']),
                   row=row)
        qualified = (rec['n_sbe'] == 1 and not rec['zero'])
        (qual if qualified else unq).append(rec)

    def summarize(g, with_baselines=False):
        if not g:
            return dict(n=0)
        d = dict(n=len(g),
                 ring_pick=float(np.mean([x['ring'] for x in g])),
                 acc=float(np.mean([x['truth'] for x in g])))
        if with_baselines:
            d['fiedler_df'] = float(np.mean([x['df_hit'] for x in g]))
            d['betweenness'] = float(np.mean([x['btw_hit'] for x in g]))
            d['random'] = float(np.mean([x['row']['random'] for x in g]))
            d['chem_model'] = float(np.mean([x['row']['chem_model'] for x in g]))
            d['combined'] = float(np.mean([x['row']['combined'] for x in g]))
        return d

    Q, U = summarize(qual, True), summarize(unq, True)
    R['scope'] = dict(
        qualified=Q, unqualified=U,
        abstained=dict(n=len(abst)),
        violations=int(sum(1 for x in qual if x['ring'])),
        unqual_ring=int(sum(1 for x in unq if x['ring'])),
        n_selected=len(qual) + len(unq) + len(abst),
        n_total=len(recs))

    # ---- does the rule help when composed with a fallback? ------------
    # Operational question: use SBE when qualified, otherwise fall back to
    # the combined model.  Compared against the combined model everywhere.
    R['composed'] = dict(
        sbe_then_combined=float(np.mean(
            [x['truth'] for x in qual] +
            [x['row']['combined'] for x in unq + abst])),
        combined_everywhere=float(np.mean(
            [x['row']['combined'] for x in qual + unq + abst])),
        n_qual=len(qual))

    # what fraction of ground-truth bonds lie inside rings
    allb = [b for f in feat_list for b in f['bonds']]
    R['pool'] = dict(
        ring_bond_frac=float(np.mean([b['ring'] for b in allb])),
        truth_ring_frac=float(np.mean([b['ring'] for b in allb if b['truth']])),
        truth_rate_ring=float(np.mean([b['truth'] for b in allb if b['ring']])),
        truth_rate_nonring=float(np.mean([b['truth'] for b in allb
                                          if not b['ring']])))

    # ---- per class ---------------------------------------------------
    by = collections.defaultdict(list)
    for r in recs:
        by[str(r['cls'])].append(r)
    R['per_class'] = []
    for c in sorted(by, key=lambda x: int(x) if x.isdigit() else 99):
        g = by[c]
        R['per_class'].append(dict(
            cls=c, n=len(g),
            **{k: float(np.mean([r[k] for r in g]))
               for k in ['betweenness', 'fiedler_df', 'brics',
                          'struct_model', 'chem_model', 'combined']}))

    # ---- population flow ---------------------------------------------
    flow = {}
    for name in ('pool_test_flow', 'pool_train_flow'):
        f = json.load(open('../data/%s.json' % name))
        flow[name.replace('pool_', '').replace('_flow', '')] = f
    R['flow'] = flow

    R['bond_positive_rate'] = R['n_test_truth'] / R['n_test_bonds']

    json.dump(R, open(out_p, 'w'), indent=1)

    print('n =', n)
    for d in main_tab:
        print('  %-40s %.4f [%.4f, %.4f]' % (d['label'], d['acc'], d['lo'], d['hi']))
    print('decomposition: total %.4f = chem %.4f + struct %.4f + shared %.4f'
          % (total, uniq_b, uniq_c, shared))
    print('  shares: chem %.1f%% struct %.1f%% shared %.1f%%'
          % (100 * uniq_b / total, 100 * uniq_c / total, 100 * shared / total))
    print('wrote', out_p)


if __name__ == '__main__':
    main(*sys.argv[1:4])
