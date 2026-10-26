"""Stage 9e: verify that the "direct ring" structural features are not
independent of the two-edge-connected component size.

Motivation.  The ablation row `S:noring_direct` drops in_cycle, detour and
bridge, and its accuracy came out identical to the full block.  An identical
accuracy is exactly what a silently-ineffective feature drop looks like -- a
bug of precisely the kind this pipeline already produced once -- so before the
row can be interpreted it has to be established which of the two it is.

On a connected undirected graph, "lies in a cycle", "is not a bridge" and
"detour length is zero" are the same predicate, and comp_edges is the size of
the bond's two-edge-connected component, which is zero exactly on the bridges.
So the three dropped features are deterministic functions of a column that is
kept, and dropping them cannot change any model's input distribution.  The
unchanged accuracy is therefore the correct outcome, not a defect -- but that
is a claim about the data, so it is checked here rather than asserted.

Writes data/ring_redundancy.json for 06_emit_tex.py.
"""
import json
import sys


def main(feat_p, out_p):
    n = 0
    mism = {'bridge': 0, 'in_cycle': 0, 'detour': 0}
    with open(feat_p, encoding='utf-8') as fh:
        for line in fh:
            for b in json.loads(line)['bonds']:
                s = b['struct']
                n += 1
                is_bridge = (s['comp_edges'] == 0.0)
                if s['bridge'] != is_bridge:
                    mism['bridge'] += 1
                if s['in_cycle'] != (not is_bridge):
                    mism['in_cycle'] += 1
                if (s['detour'] == 0.0) != is_bridge:
                    mism['detour'] += 1

    out = dict(n_bonds=n, mismatches=mism,
               total_mismatches=int(sum(mism.values())))
    json.dump(out, open(out_p, 'w'), indent=1)
    print('bonds checked: %d' % n)
    for k, v in mism.items():
        print('  %-10s disagrees with comp_edges==0 on %d bonds' % (k, v))
    print('wrote', out_p)


if __name__ == '__main__':
    main(*sys.argv[1:3])
