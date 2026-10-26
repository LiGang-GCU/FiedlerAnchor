"""Stage 2 (v2): per-bond features with verified invariance.

Fixes over the first version, all found in review:

  * two-edge-connected components were computed with
    nx.biconnected_components, which returns VERTEX blocks; articulation
    vertices belong to several blocks and the assignment silently
    overwrote.  Now computed correctly by deleting bridges and taking
    connected components of what remains, and assigned per EDGE.
  * deg_u/deg_v and tri_u/tri_v depended on the stored endpoint order.
    Now reported as min/max.
  * fsum = f(u) + f(v) flips sign with the arbitrary global sign of the
    Fiedler vector.  Replaced by its absolute value.
  * every feature in the structural block is checked for invariance under
    atom renumbering by an explicit test in 02b_invariance.py.

CHEMICAL block: requires atom labels.
STRUCTURAL block: computable from the unlabelled graph.

Reviewer note acted on: the chemical block contains nbr_C and nbr_het, whose
sum is the degree on a hydrogen-suppressed graph, so the blocks are NOT
information-disjoint.  The manuscript now says so explicitly and reports the
decomposition as conditional performance attribution rather than as a
partition of information.
"""
import json
import sys

import networkx as nx
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdchem

RDLogger.DisableLog('rdApp.*')

ELEMENTS = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'Si', 'Se']
HYB = [rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2,
       rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D,
       rdchem.HybridizationType.SP3D2]
ZERO = 1e-9


# ----------------------------------------------------------------- chemical
def atom_block(a):
    """Requires element identity -> CHEMICAL."""
    f = {}
    for e in ELEMENTS:
        f['el_' + e] = float(a.GetSymbol() == e)
    f['el_other'] = float(a.GetSymbol() not in ELEMENTS)
    for h in HYB:
        f['hyb_%d' % int(h)] = float(a.GetHybridization() == h)
    f['nh'] = float(a.GetTotalNumHs())
    f['chg'] = float(a.GetFormalCharge())
    f['nbr_C'] = float(sum(1 for n in a.GetNeighbors() if n.GetSymbol() == 'C'))
    f['nbr_N'] = float(sum(1 for n in a.GetNeighbors() if n.GetSymbol() == 'N'))
    f['nbr_O'] = float(sum(1 for n in a.GetNeighbors() if n.GetSymbol() == 'O'))
    f['nbr_S'] = float(sum(1 for n in a.GetNeighbors() if n.GetSymbol() == 'S'))
    f['nbr_het'] = float(sum(1 for n in a.GetNeighbors()
                            if n.GetSymbol() not in ('C', 'H')))
    return f


def bond_block_chem(b):
    """Requires atom labels or annotated bond order -> CHEMICAL."""
    f = {}
    o = b.GetBondTypeAsDouble()
    f['order1'] = float(o == 1.0)
    f['order15'] = float(o == 1.5)
    f['order2'] = float(o == 2.0)
    f['order3'] = float(o == 3.0)
    f['aromatic'] = float(b.GetIsAromatic())
    f['conjugated'] = float(b.GetIsConjugated())
    a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
    f['both_C'] = float(a1.GetSymbol() == 'C' and a2.GetSymbol() == 'C')
    f['both_het'] = float(a1.GetSymbol() != 'C' and a2.GetSymbol() != 'C')
    f['one_het'] = 1.0 - f['both_C'] - f['both_het']
    f['has_N'] = float('N' in (a1.GetSymbol(), a2.GetSymbol()))
    f['has_O'] = float('O' in (a1.GetSymbol(), a2.GetSymbol()))
    f['has_S'] = float('S' in (a1.GetSymbol(), a2.GetSymbol()))
    f['has_hal'] = float(bool({a1.GetSymbol(), a2.GetSymbol()} &
                              {'F', 'Cl', 'Br', 'I'}))
    return f


# --------------------------------------------------------------- structural
def edge_bicomponent_sizes(G):
    """Size in edges of the two-edge-connected component of every edge.

    Two edges are in the same two-edge-connected component iff they lie on a
    common cycle.  Deleting all bridges and taking connected components of
    the remainder gives exactly that partition, at EDGE granularity -- unlike
    nx.biconnected_components, which is a vertex-block decomposition.
    """
    bridges = {frozenset((u, v)) for u, v in nx.bridges(G)}
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v in G.edges():
        if frozenset((u, v)) not in bridges:
            H.add_edge(u, v)
    size = {}
    for comp in nx.connected_components(H):
        ecount = H.subgraph(comp).number_of_edges()
        for v in comp:
            size[v] = ecount
    out = {}
    for u, v in G.edges():
        k = frozenset((u, v))
        out[(min(u, v), max(u, v))] = 0.0 if k in bridges else float(size[u])
    return out, bridges


def structural_block(G):
    """Every quantity here is computable from the UNLABELLED graph, and every
    per-bond quantity is invariant to atom renumbering and to the global sign
    of the Fiedler vector."""
    n = G.number_of_nodes()
    m = G.number_of_edges()

    btw_raw = nx.edge_betweenness_centrality(G, normalized=True)
    btw = {(min(a, b), max(a, b)): v for (a, b), v in btw_raw.items()}

    L = nx.laplacian_matrix(G).toarray().astype(float)
    w, V = np.linalg.eigh(L)
    fv = V[:, 1]
    Lp = np.linalg.pinv(L)

    comp_edges, bridges = edge_bicomponent_sizes(G)
    core = nx.core_number(G)
    triangles = nx.triangles(G)

    E = [(min(a, b), max(a, b)) for a, b in G.edges()]
    btw_vals = np.array([btw[k] for k in E])
    df_vals = np.array([abs(fv[a] - fv[b]) for a, b in G.edges()])

    def avg_rank(vals):
        """Within-molecule rank with ties resolved to their average rank.

        A raw (vals <= vals[i]).sum() count is not invariant under atom
        renumbering: in a symmetric molecule the equivalent bonds have values
        that differ only by floating-point noise, so which of them counts as
        "above" the other flips with the atom order.  Averaging over the tie
        group removes that dependence.

        The tolerance is RELATIVE to the scale of the values.  An absolute
        tolerance puts any third value that happens to sit at the boundary in
        or out of the tie group depending on rounding, which is exactly the
        non-invariance this is meant to remove.

        The tolerance must also be LOOSER than the reproducibility of the
        eigenvalues, not tighter.  With 1e-7 a pair that agrees to 1e-6 across
        two atom orderings is still split into different tie groups, so the
        rank changes under renumbering even though the underlying value does
        not.  1e-6 matches the tolerance at which the invariance audit
        compares, and is far below any chemically meaningful difference in
        betweenness or Fiedler gradient.
        """
        scale = float(np.max(np.abs(vals))) if len(vals) else 0.0
        tol = 1e-6 * max(scale, 1e-12)
        out = np.empty(len(vals), float)
        for i, v in enumerate(vals):
            tie = np.abs(vals - v) <= tol
            lo = int((vals < v - tol).sum())
            hi = lo + int(tie.sum())
            out[i] = 0.5 * (lo + hi) / max(1, len(vals))
        return out

    btw_rank = dict(zip(E, avg_rank(btw_vals)))
    df_rank = dict(zip(E, avg_rank(df_vals)))

    out = {}
    for a, b in G.edges():
        key = (min(a, b), max(a, b))
        H = G.copy()
        H.remove_edge(a, b)
        detour = float(nx.shortest_path_length(H, a, b)) if nx.has_path(H, a, b) else 0.0
        dau, dbv = G.degree(a), G.degree(b)
        out[key] = dict(
            btw=float(btw[key]),
            btw_rank=btw_rank[key],
            df=float(abs(fv[a] - fv[b])),
            df_rank=df_rank[key],
            fsum_abs=float(abs(fv[a] + fv[b])),
            fabs_sum=float(abs(fv[a]) + abs(fv[b])),
            cross=float((fv[a] > ZERO and fv[b] < -ZERO) or
                        (fv[b] > ZERO and fv[a] < -ZERO)),
            res=float(Lp[a, a] + Lp[b, b] - 2 * Lp[a, b]),
            bridge=float(frozenset(key) in bridges),
            deg_min=float(min(dau, dbv)), deg_max=float(max(dau, dbv)),
            in_cycle=float(detour > 0),
            detour=detour,
            comp_edges=comp_edges[key],
            core_min=float(min(core[a], core[b])),
            core_max=float(max(core[a], core[b])),
            tri_min=float(min(triangles[a], triangles[b])),
            tri_max=float(max(triangles[a], triangles[b])),
            n=float(n), m=float(m),
            lam2=float(w[1]),
            cyc_rank=float(m - n + 1),
            deg_std=float(np.std([d for _, d in G.degree()])),
        )
    return out, w


def main(src, out):
    n = 0
    with open(src, encoding='utf-8') as fh, open(out, 'w', encoding='utf-8') as fo:
        for line in fh:
            rec = json.loads(line)
            mol = Chem.MolFromSmiles(rec['smiles'])
            if mol is None:
                continue
            G = nx.Graph()
            G.add_nodes_from(range(mol.GetNumAtoms()))
            for b in mol.GetBonds():
                G.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
            sb, w = structural_block(G)

            meta = {}
            for e in rec['edges']:
                meta[(min(e['u'], e['v']), max(e['u'], e['v']))] = e

            rows = []
            for b in mol.GetBonds():
                u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                key = (min(u, v), max(u, v))
                if key not in meta:
                    continue
                chem = dict(('b_' + k, x) for k, x in bond_block_chem(b).items())
                a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
                t1 = (a1.GetSymbol(), a1.GetTotalNumHs(), a1.GetDegree())
                t2 = (a2.GetSymbol(), a2.GetTotalNumHs(), a2.GetDegree())
                if t1 <= t2:
                    x1, x2 = atom_block(a1), atom_block(a2)
                else:
                    x1, x2 = atom_block(a2), atom_block(a1)
                chem.update(('a1_' + k, x) for k, x in x1.items())
                chem.update(('a2_' + k, x) for k, x in x2.items())

                mm = meta[key]
                rows.append(dict(
                    u=key[0], v=key[1],
                    chem=chem, struct=sb[key],
                    truth=mm['truth'], brics=mm['brics'],
                    ring=mm['ring'], env=mm['env'],
                ))
            json.dump(dict(rid=rec['rid'], cls=rec['cls'], split=rec['split'],
                           smiles=rec['smiles'],
                           n_atoms=rec['n_atoms'], lam2=rec['lam2'],
                           lam3=rec['lam3'], zero_vertex=rec['zero_vertex'],
                           n_sbe=rec['n_sbe'], bonds=rows), fo)
            fo.write('\n')
            n += 1
    print('wrote', n, '->', out)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
