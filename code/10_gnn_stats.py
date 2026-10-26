"""Stage 10: fold the GNN results into the single-source number pipeline.

Reads the per-seed GNN run files written by 09d_multiseed.py and emits
data/gnn_numbers.json, which 06_emit_tex.py consumes alongside numbers.json.

Everything reported about the GNN branch originates here: means, standard
deviations, per-seed values, and paired seed-matched comparisons between
variants.  No GNN number is typed into the manuscript.
"""
import glob
import json
import os
import sys

import numpy as np

ORDER = ['spec', 'struct', 'chem', 'both']
LABEL = {'spec': 'Spectral descriptors',
         'struct': 'Structural topology (non-spectral)',
         'chem': 'Chemical (atom labels)',
         'both': 'Chemical + structural'}
# the hand-crafted, gradient-boosted counterparts, keyed to the same branch
GBDT_KEY = {'spec': 'S:spectral', 'struct': 'S:topology',
            'chem': 'C:chem_all', 'both': 'D:chem+struct'}


def load_runs(indir):
    """Collect every seed that actually finished.

    The seed range is discovered by glob rather than fixed.  A hard-coded
    (0, 1, 2) here silently discarded seeds 3 and 4 -- the runs were on disk
    but the summary averaged three of five, so the manuscript reported a
    five-seed protocol on three-seed statistics.  Any future change to the
    seed list in 09d_multiseed.py now propagates without touching this file.
    """
    runs = {}
    for v in ORDER:
        seeds = {}
        for p in sorted(glob.glob(os.path.join(indir, 'gnn_%s_s*.json' % v))):
            stem = os.path.basename(p)[len('gnn_%s_s' % v):-len('.json')]
            try:
                s = int(stem)
            except ValueError:
                continue
            seeds[s] = json.load(open(p))
        if seeds:
            runs[v] = seeds
    return runs


def main(indir, abl_p, out_p):
    runs = load_runs(indir)
    abl = json.load(open(abl_p))

    out = {'variants': {}, 'order': ORDER}
    for v in ORDER:
        if v not in runs:
            continue
        seeds = runs[v]
        accs = [seeds[s]['test_acc'] for s in sorted(seeds)]
        out['variants'][v] = dict(
            label=LABEL[v],
            accs=accs,
            mean=float(np.mean(accs)),
            sd=float(np.std(accs)),
            lo=float(np.min(accs)),
            hi=float(np.max(accs)),
            best_epochs=[seeds[s]['best_epoch'] for s in sorted(seeds)],
            gbdt=float(np.mean(abl[GBDT_KEY[v]])),
        )

    # ---- paired, seed-matched comparisons ---------------------------
    # Same seed means the same train/validation split and the same
    # initialisation, so the difference is attributable to the feature set.
    pairs = [('spec', 'struct'), ('chem', 'struct'),
             ('both', 'spec'), ('both', 'chem')]
    out['pairs'] = {}
    for a, b in pairs:
        if a not in runs or b not in runs:
            continue
        ds = []
        for s in sorted(set(runs[a]) & set(runs[b])):
            ds.append(runs[a][s]['test_acc'] - runs[b][s]['test_acc'])
        out['pairs']['%s__%s' % (a, b)] = dict(
            per_seed=ds, mean=float(np.mean(ds)), sd=float(np.std(ds)))

    # ---- architecture contrast: GNN minus gradient-boosted ----------
    out['arch_gain'] = {}
    for v in ORDER:
        if v not in out['variants']:
            continue
        d = out['variants'][v]
        out['arch_gain'][v] = d['mean'] - d['gbdt']

    json.dump(out, open(out_p, 'w'), indent=1)

    print('%-8s %-34s %6s %6s   %6s  %s'
          % ('variant', 'label', 'mean', 'sd', 'GBDT', 'gain'))
    for v in ORDER:
        if v not in out['variants']:
            continue
        d = out['variants'][v]
        print('%-8s %-34s %.4f %.4f   %.4f  %+.4f'
              % (v, d['label'], d['mean'], d['sd'], d['gbdt'],
                 out['arch_gain'][v]))
    print()
    for k, d in out['pairs'].items():
        print('  %-18s %+.4f (sd %.4f)  per seed %s'
              % (k, d['mean'], d['sd'],
                 ' '.join('%+.3f' % x for x in d['per_seed'])))
    print('wrote', out_p)


if __name__ == '__main__':
    main(*sys.argv[1:4])
