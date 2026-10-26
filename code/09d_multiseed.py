"""Stage 9d: multi-seed GNN runs.

A single run per variant is not evidence: seed-to-seed variation in a small
graph network is large enough to reorder close variants.  This driver runs
each variant under several seeds and reports mean, sd and the per-seed values.

One GPU, so runs are sequential.
"""
import json
import os
import subprocess
import sys

VARIANTS = ['spec', 'struct', 'chem', 'both']
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 15


def main(train_p, test_p, outdir):
    os.makedirs(outdir, exist_ok=True)
    summary = {}
    for v in VARIANTS:
        accs = []
        for s in SEEDS:
            out = os.path.join(outdir, 'gnn_%s_s%d.json' % (v, s))
            if os.path.exists(out):
                d = json.load(open(out))
            else:
                cmd = [sys.executable, '09_gnn.py', v, train_p, test_p,
                       out, '0', str(EPOCHS), str(s)]
                print('RUN', ' '.join(cmd), flush=True)
                r = subprocess.run(cmd, capture_output=True, text=True)
                if not os.path.exists(out):
                    print(r.stdout[-2000:])
                    print(r.stderr[-2000:])
                    raise RuntimeError('run failed: %s seed %d' % (v, s))
                d = json.load(open(out))
            accs.append(d['test_acc'])
            print('  %-7s seed %d -> %.4f' % (v, s, d['test_acc']), flush=True)
        summary[v] = dict(seeds=SEEDS, accs=accs,
                          mean=sum(accs) / len(accs))
        print('%-7s mean %.4f  sd %.4f'
              % (v, summary[v]['mean'],
                 (sum((a - summary[v]['mean']) ** 2 for a in accs)
                  / len(accs)) ** 0.5), flush=True)

    json.dump(summary, open(os.path.join(outdir, 'gnn_summary.json'), 'w'),
              indent=1)
    print('wrote', os.path.join(outdir, 'gnn_summary.json'))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
