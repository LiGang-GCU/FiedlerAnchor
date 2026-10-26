"""Pick a representative molecule for the TOC graphic and dump its Fiedler signs."""
import json
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')

ZERO = 1e-9


def fiedler(mol):
    n = mol.GetNumAtoms()
    A = np.zeros((n, n))
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        w = 1.5 if b.GetIsAromatic() else 1.0
        A[i, j] = A[j, i] = w
    D = np.diag(A.sum(axis=1))
    L = D - A
    w, v = np.linalg.eigh(L)
    return w, v[:, 1], A


def main(src, n_want):
    best = []
    for line in open(src, encoding='utf-8'):
        r = json.loads(line)
        if r['n_sbe'] != 1 or r['zero_vertex'] or r['n_truth'] != 1:
            continue
        cross = [e for e in r['edges'] if e['cross']]
        if len(cross) != 1 or not cross[0]['truth']:
            continue
        n = r['n_atoms']
        if not (13 <= n <= 18):
            continue
        mol = Chem.MolFromSmiles(r['smiles'])
        if mol is None:
            continue
        try:
            w, f, A = fiedler(mol)
        except Exception:
            continue
        if w[1] < 1e-6:
            continue
        pos = int((f > ZERO).sum())
        neg = int((f < -ZERO).sum())
        if pos < 3 or neg < 3:
            continue
        # prefer a balanced split (visually clear two-colour division)
        balance = min(pos, neg) / max(pos, neg)
        # and prefer some ring content so it reads as a drug-like scaffold
        ring = sum(1 for e in r['edges'] if e['ring'])
        score = balance * 2.0 + ring / len(r['edges'])
        best.append((score, balance, pos, neg, ring, len(r['edges']),
                     r['rid'], r['smiles'], list(map(float, f))))
    best.sort(reverse=True)
    print('candidates:', len(best))
    for b in best[:6]:
        print('score %.2f  balance %.2f  +%d/-%d  ring %d/%d  %s'
              % (b[0], b[1], b[2], b[3], b[4], b[5], b[6]))
        print('   ', b[7][:90])
    if best:
        pick = best[0]
        out = {'rid': pick[6], 'smiles': pick[7], 'f': pick[8],
               'n_pos': pick[2], 'n_neg': pick[3]}
        json.dump(out, open('data/toc_molecule.json', 'w'), indent=1)
        print('\nwrote data/toc_molecule.json ->', pick[6])


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1)
