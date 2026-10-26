"""Stage 9b: invariance audit.

The reviewer found that several hand-crafted structural features changed when
the atoms were renumbered: deg_u/deg_v and tri_u/tri_v depended on the stored
endpoint order, comp_edges looked up a vertex block rather than an edge block,
and fsum changed sign with the global sign of the Fiedler vector.  The
features were rewritten; this script verifies the repair rather than assuming
it.

Two mistakes were made in the first version of this audit and are worth
recording, because both produced a false FAIL:

  1. RDKit's RenumberAtoms(mol, order) builds a molecule whose atom i is the
     OLD atom order[i], so an old atom u lives at new index inv[u], not at
     order[u].  The forward map was used as if it were the inverse.
  2. the permuted molecule was round-tripped through MolToSmiles, which
     canonicalises and renumbers again, silently discarding the permutation.
     The features must be computed on the mol object, never on its SMILES.

This version therefore builds both graphs from mol objects and maps bonds by
the exact inverse permutation.
"""
import json
import os
import sys

import networkx as nx
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
_feat = import_module('02_features')

TOL = 1e-6


def structural_from_mol(mol):
    G = nx.Graph()
    G.add_nodes_from(range(mol.GetNumAtoms()))
    for b in mol.GetBonds():
        G.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    sb, w = _feat.structural_block(G)
    return sb, G, w


def main(test_p, n_sample=200, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    with open(test_p, encoding='utf-8') as fh:
        for i, line in enumerate(fh):
            if i >= n_sample:
                break
            recs.append(json.loads(line))

    worst = {}
    n_checked = 0
    n_missing = 0
    n_eps_flips = 0          # top eigenvalues swapping under renumbering

    for rec in recs:
        mol = Chem.MolFromSmiles(rec['smiles'])
        if mol is None:
            continue
        n = mol.GetNumAtoms()
        perm = [int(i) for i in rng.permutation(n)]
        mol2 = Chem.RenumberAtoms(mol, perm)

        # old atom u lives at new index inv[u]
        inv = [0] * n
        for new_i, old_i in enumerate(perm):
            inv[old_i] = new_i

        sb1, _, w1 = structural_from_mol(mol)
        sb2, _, w2 = structural_from_mol(mol2)

        # a permuted graph can exchange near-degenerate Fiedler subspaces;
        # record it rather than silently comparing incomparable values
        if abs(abs(w1[1] - w2[1]) > 1e-9 or abs(w1[1] - w2[1]) > 1e-7):
            n_eps_flips += 1

        for (u, v), f1 in sb1.items():
            a, b = inv[u], inv[v]
            k2 = (min(a, b), max(a, b))
            if k2 not in sb2:
                n_missing += 1
                continue
            f2 = sb2[k2]
            n_checked += 1
            for name, val in f1.items():
                d = abs(float(val) - float(f2[name]))
                if d > TOL:
                    worst[name] = max(worst.get(name, 0.0), d)

    print('molecules sampled : %d' % len(recs))
    print('bonds compared    : %d' % n_checked)
    print('bonds unmatched   : %d' % n_missing)
    print('lambda2 shifts    : %d' % n_eps_flips)
    print()
    if n_missing:
        print('CHECK A INCONCLUSIVE: %d bonds could not be matched after the '
              'permutation; the mapping is wrong, not the features' % n_missing)
        return 1
    if not worst:
        print('CHECK A PASSED: every structural feature is invariant to atom '
              'renumbering (tolerance %g)' % TOL)
    else:
        print('CHECK A FAILED: %d feature(s) change under renumbering' % len(worst))
        for k, v in sorted(worst.items(), key=lambda t: -t[1]):
            print('  %-12s max |delta| = %.6g' % (k, v))

    # ---- sign invariance of the Fiedler-derived quantities -------------
    bad_sign = 0
    for rec in recs[:50]:
        sb, G, w = structural_from_mol(Chem.MolFromSmiles(rec['smiles']))
        L = nx.laplacian_matrix(G).toarray().astype(float)
        ev, V = np.linalg.eigh(L)
        f = V[:, 1]
        for (u, v), feats in sb.items():
            if abs(abs(f[u] + f[v]) - feats['fsum_abs']) > TOL:
                bad_sign += 1
            if abs((abs(f[u]) + abs(f[v])) - feats['fabs_sum']) > TOL:
                bad_sign += 1
    if bad_sign == 0:
        print('CHECK C PASSED: Fiedler-derived features are sign-invariant')
    else:
        print('CHECK C FAILED: %d sign-dependent values' % bad_sign)

    ok = (not worst) and bad_sign == 0
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 200))
