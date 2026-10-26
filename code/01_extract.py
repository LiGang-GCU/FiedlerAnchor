"""Stage 1: build the per-molecule pool file from atom-mapped reactions.

Writes  data/pool.jsonl  -- one JSON object per product molecule.
Every downstream number is derived from this file; nothing is typed by hand.

Population stages (applied identically to train and test):
  P0  rows read from the csv
  P1  product parses, carries atom maps, 5 <= heavy atoms <= 60
  P2  product graph is connected
  P3  at least one product bond has a mapped atom pair absent from the
      reactants  (a reaction-derived disconnection exists)
  P4  (TEST ONLY, the analysis population) at least one non-ring bond exists
      so that every selector has a candidate pool.
      P4 is defined WITHOUT reference to any spectral quantity.
"""
import csv, json, sys, os
import networkx as nx
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS
from scipy.linalg import eigh

RDLogger.DisableLog('rdApp.*')

ZERO = 1e-9       # |f_v| below this is a zero-component vertex
DEGEN = 1e-8      # lambda3 - lambda2 below this is a degenerate split


def mapped_edges(m):
    return {frozenset((b.GetBeginAtom().GetAtomMapNum(),
                       b.GetEndAtom().GetAtomMapNum())) for b in m.GetBonds()}


def env_class(sym1, sym2, ring, arom):
    """Coarse bond environment used for the descriptive joint table."""
    if arom:
        return 'aromatic'
    if ring:
        return 'ring'
    return 'hetero' if (sym1 != 'C' or sym2 != 'C') else 'CC'


def process(row, split):
    rxn = row.get('reactants>reagents>production')
    if not rxn:
        return None, 'unparsed'
    parts = rxn.split('>')
    if len(parts) != 3:
        return None, 'unparsed'
    prods = [Chem.MolFromSmiles(s) for s in parts[2].split('.')]
    prods = [m for m in prods if m is not None
             and any(a.GetAtomMapNum() for a in m.GetAtoms())]
    if not prods:
        return None, 'unparsed'
    prods = [(m, s) for m, s in zip(prods, parts[2].split('.'))
             if m is not None and any(a.GetAtomMapNum() for a in m.GetAtoms())]
    prod, prod_smi = max(prods, key=lambda t: t[0].GetNumAtoms())
    if not (5 <= prod.GetNumAtoms() <= 60):
        return None, 'size'

    G = nx.Graph()
    for a in prod.GetAtoms():
        G.add_node(a.GetIdx())
    for b in prod.GetBonds():
        G.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    if not nx.is_connected(G):
        return None, 'disconnected'

    # ---- reaction-derived ground truth -------------------------------
    react_edges = set()
    for m in [Chem.MolFromSmiles(s) for s in parts[0].split('.')]:
        if m is not None:
            react_edges |= mapped_edges(m)
    prod_edges = mapped_edges(prod)
    broken_pairs = prod_edges - react_edges

    mp = {a.GetIdx(): a.GetAtomMapNum() for a in prod.GetAtoms()}
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in prod.GetBonds()]
    truth = set()
    for u, v in bonds:
        if frozenset((mp[u], mp[v])) in broken_pairs:
            truth.add((min(u, v), max(u, v)))
    if not truth:
        return None, 'no_disconnection'

    # ---- graph quantities --------------------------------------------
    L = nx.laplacian_matrix(G).toarray().astype(float)
    w, V = eigh(L)
    degenerate = bool(abs(w[2] - w[1]) < DEGEN) if len(w) > 2 else True
    f = V[:, 1]
    zero_vertex = bool((np.abs(f) <= ZERO).any())
    btw_raw = nx.edge_betweenness_centrality(G, normalized=True)
    btw = {(min(a, b), max(a, b)): v for (a, b), v in btw_raw.items()}

    brics_bonds = {(min(a, b), max(a, b)) for (a, b), _ in BRICS.FindBRICSBonds(prod)}

    edges = []
    for b in prod.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        key = (min(u, v), max(u, v))
        s1, s2 = b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()
        ring, arom = b.IsInRing(), b.GetIsAromatic()
        df = float(abs(f[u] - f[v]))
        cross = bool((f[u] > ZERO and f[v] < -ZERO) or (f[v] > ZERO and f[u] < -ZERO))
        edges.append(dict(
            u=key[0], v=key[1],
            e1=s1, e2=s2,
            ring=bool(ring), arom=bool(arom),
            env=env_class(s1, s2, ring, arom),
            btw=float(btw[key]),
            df=df, cross=cross,
            zero=bool(abs(f[u]) <= ZERO or abs(f[v]) <= ZERO),
            truth=bool(key in truth),
            brics=bool(key in brics_bonds),
        ))

    return dict(
        split=split,
        rid=row.get('id', ''),
        cls=row.get('class', ''),
        smiles=prod_smi,
        n_atoms=prod.GetNumAtoms(),
        n_bonds=len(edges),
        degenerate=degenerate,
        zero_vertex=zero_vertex,
        lam2=float(w[1]), lam3=float(w[2]) if len(w) > 2 else float('nan'),
        n_sbe=sum(1 for e in edges if e['cross']),
        n_truth=len(truth),
        n_truth_nonring=sum(1 for e in edges if e['truth'] and not e['ring']),
        edges=edges,
    ), None


def main(src, split, out):
    counts = {}
    n = 0
    with open(src, newline='', encoding='utf-8') as fh, open(out, 'w', encoding='utf-8') as fo:
        for row in csv.DictReader(fh):
            n += 1
            rec, why = process(row, split)
            if rec is None:
                counts[why] = counts.get(why, 0) + 1
                continue
            if split == 'test' and not any(not e['ring'] for e in rec['edges']):
                counts['no_nonring_bond'] = counts.get('no_nonring_bond', 0) + 1
                continue
            fo.write(json.dumps(rec) + '\n')
            counts['kept'] = counts.get('kept', 0) + 1
    print(json.dumps(dict(split=split, rows=n, **counts), indent=1))
    json.dump(dict(split=split, rows=n, **counts),
              open(out.replace('.jsonl', '_flow.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
