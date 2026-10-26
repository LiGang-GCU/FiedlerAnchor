"""Stage 9: graph-neural-network structural branch.

Purpose.  The hand-crafted structural block reaches about 0.38 on the test
split.  That is a bound on what THOSE FEATURES support, not on what the
unlabelled graph supports.  This script asks the second question with a model
whose inductive bias is graph structure itself.

Label blindness is enforced at the input, not promised in prose:

  * node features come from the adjacency matrix only -- degree histogram,
    k-core number, triangle count.  No element identity, no hybridisation,
    no charge, no hydrogen count.
  * edge features are structural only -- cycle membership and detour length
    (shortest endpoint-to-endpoint path avoiding the bond).  No bond order,
    no aromaticity.

Two variants:
  adj     -- constant node feature, no edge feature: the pure adjacency matrix.
  struct  -- the structural descriptors above.

Message passing is permutation-equivariant by construction, so the model is
invariant to atom renumbering.  The failure mode the reviewer found in the
hand-crafted features -- per-bond quantities that changed under atom
renumbering -- cannot occur here, and 09b_invariance.py checks it empirically.

Protocol matches the rest of the pipeline: fit on train, select the epoch on
a validation split carved out of train, touch test once.
"""
import json
import os
import sys
import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


# ------------------------------------------------------------------ featurise
def graph_from_pool(rec, variant):
    """Build one PyG Data object; edge i corresponds to rec['edges'][i].

    Variants share one architecture so that the chemical-versus-structural
    comparison does not confound information type with model class:

      adj     constant node feature, no edge feature: the pure adjacency matrix
      struct  unlabelled-graph descriptors (degree, k-core, triangles; ring,
              detour).  No atom identity of any kind.
      chem    atom labels only (element, hybridisation, charge, hydrogens;
              bond order, aromaticity).  No graph-global quantity.
      spec    the classical spectral toolkit only (betweenness, Fiedler
              gradient, effective resistance, bridge flag, algebraic
              connectivity), as edge and node scalars.  No atom identity and
              no non-spectral topology.
      both    chem + struct
    """
    n = rec['n_atoms']
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for e in rec['edges']:
        G.add_edge(e['u'], e['v'])

    want_struct = variant in ('struct', 'both')
    want_chem = variant in ('chem', 'both')
    want_spec = variant == 'spec'
    if variant == 'adj':
        want_struct = want_chem = False

    xblocks, eblocks = [], []

    if want_struct:
        deg = np.array([G.degree(i) for i in range(n)], float)
        core = nx.core_number(G)
        tri = nx.triangles(G)
        # Eccentricity is a GLOBAL per-vertex coordinate that is not spectral.
        # The spectral variant receives the Fiedler coordinate as a node
        # feature; without a comparable global coordinate here, the
        # spectral-versus-structural comparison would confound "global vs
        # local information" with "spectral vs non-spectral".
        if n > 1:
            ecc = np.array([float(nx.eccentricity(G, v)) for v in range(n)], float)
            ecmax = float(ecc.max()) if ecc.max() > 0 else 1.0
            ecc = ecc / ecmax
        else:
            ecc = np.zeros(n, float)
        doh = np.zeros((n, 8), np.float32)
        doh[np.arange(n), np.clip(deg, 0, 7).astype(int)] = 1.0
        xblocks.append(doh)
        xblocks.append((np.array([core[i] for i in range(n)], float) / 8.0)[:, None].astype(np.float32))
        xblocks.append((np.array([tri[i] for i in range(n)], float) / 8.0)[:, None].astype(np.float32))
        xblocks.append((ecc / ecmax)[:, None].astype(np.float32))

        srow = []
        for e in rec['edges']:
            a, b = e['u'], e['v']
            H = G.copy()
            H.remove_edge(a, b)
            detour = nx.shortest_path_length(H, a, b) if nx.has_path(H, a, b) else 0
            # Four edge features to match the spec variant's edge-feature count:
            # ring membership, detour, |core_a - core_b|, |tri_a - tri_b|.
            # Using absolute differences preserves invariance under endpoint swap.
            row = [float(e['ring']),
                   min(detour, 40) / 40.0,
                   abs(core[a] - core[b]) / 8.0,
                   abs(tri[a] - tri[b]) / 8.0]
            srow.append(row)
        eblocks.append(np.array(srow, np.float32) if srow
                       else np.zeros((0, 4), np.float32))

    if want_spec:
        # The classical spectral toolkit, with no atom identity and no
        # non-spectral topology.  Providing it in the same architecture that
        # serves the structural block is what makes the spectral-versus-
        # structural comparison a statement about information rather than
        # about model class.
        import scipy.linalg as sla
        L = nx.laplacian_matrix(G).toarray().astype(float)
        w, V = np.linalg.eigh(L)
        fv = V[:, 1]
        Lp = np.linalg.pinv(L)
        btw_raw = nx.edge_betweenness_centrality(G, normalized=True)
        btw = {(min(a, b), max(a, b)): v for (a, b), v in btw_raw.items()}
        bridges = {frozenset((u, v)) for u, v in nx.bridges(G)}

        # node: Fiedler coordinate and its rank, both sign-invariant
        fr = np.argsort(np.argsort(fv)).astype(float) / max(1, n - 1)
        xblocks.append(np.stack([fv, fr], axis=1).astype(np.float32))

        srow = []
        for e in rec['edges']:
            a, b = e['u'], e['v']
            key = (min(a, b), max(a, b))
            res = float(Lp[a, a] + Lp[b, b] - 2 * Lp[a, b])
            srow.append([float(btw[key]),
                         float(abs(fv[a] - fv[b])),
                         1.0 / (1.0 + res),
                         float(frozenset(key) in bridges)])
        eblocks.append(np.array(srow, np.float32) if srow
                       else np.zeros((0, 4), np.float32))

    if want_chem:
        # Parse the stored product SMILES rather than requiring new pool
        # fields: the pool stores the raw product SMILES and indexes edges
        # against the same parse, so atom order matches rec['edges'].
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        mol = Chem.MolFromSmiles(rec['smiles'])
        EL = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
        doh = np.zeros((n, len(EL) + 1), np.float32)
        for a in mol.GetAtoms():
            s = a.GetSymbol()
            doh[a.GetIdx(), EL.index(s) if s in EL else len(EL)] = 1.0
        xblocks.append(doh)

        crow = []
        for e in rec['edges']:
            bond = mol.GetBondBetweenAtoms(e['u'], e['v'])
            o = bond.GetBondTypeAsDouble()
            crow.append([float(o == 1.0), float(o == 1.5),
                         float(o == 2.0), float(o == 3.0),
                         float(bond.GetIsAromatic())])
        eblocks.append(np.array(crow, np.float32) if crow
                       else np.zeros((0, 5), np.float32))

    if not xblocks:
        x = np.ones((n, 1), np.float32)
        eattr = np.zeros((len(rec['edges']), 1), np.float32)
    else:
        x = np.concatenate(xblocks, axis=1).astype(np.float32)
        eattr = np.concatenate(eblocks, axis=1).astype(np.float32)

    src, dst = [], []
    for e in rec['edges']:
        u, v = e['u'], e['v']
        src += [u, v]
        dst += [v, u]
    lab = np.array([1.0 if e['truth'] else 0.0 for e in rec['edges']], np.float32)

    m = len(rec['edges'])
    # edge_index is interleaved: [u1,v1, u2,v2, ...], so each undirected
    # edge's two directed copies are at consecutive indices (2i, 2i+1).
    # The edge attributes must follow the same interleaving, which means
    # REPEAT each row, not concatenate two copies: np.repeat gives
    # [a1,a1, a2,a2, ...] while np.concatenate([eattr,eattr]) gives
    # [a1,a2,..., a1,a2,...] and misaligns every directed edge past the first.
    ea = np.repeat(eattr, 2, axis=0) if m else \
        np.zeros((0, eattr.shape[1]), np.float32)

    return Data(x=torch.from_numpy(x),
                edge_index=torch.tensor([src, dst], dtype=torch.long),
                edge_attr=torch.from_numpy(ea.astype(np.float32)),
                y=torch.from_numpy(lab),
                n_und=torch.tensor([m], dtype=torch.long))


def load_split(path, variant, limit=None):
    out = []
    with open(path, encoding='utf-8') as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            out.append(graph_from_pool(json.loads(line), variant))
    return out


# ---------------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden=96, layers=3, dropout=0.15):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden * 2, 1)

    def embed(self, data):
        h = self.inp(data.x)
        for conv, norm in zip(self.convs, self.norms):
            h = h + conv(h, data.edge_index, data.edge_attr)
            h = norm(h)
            h = F.relu(h)
            h = self.drop(h)
        return h

    def scores(self, data):
        """Score each UNDIRECTED edge.

        Edges are stored as (u->v, v->u) pairs, so directed index 2i and 2i+1
        are the same undirected bond.  The readout (h_u + h_v, |h_u - h_v|) is
        symmetric in the endpoints, so both directions get the same score.
        """
        h = self.embed(data)
        src, dst = data.edge_index
        hu, hv = h[src], h[dst]
        pair = torch.cat([hu + hv, (hu - hv).abs()], dim=-1)
        s = self.out(pair).squeeze(-1)
        return s[0::2]


def per_molecule_hits(model, loader, device):
    model.eval()
    hits = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            s = model.scores(batch)
            y = batch.y
            n_und = batch.n_und.tolist()
            off = 0
            for c in n_und:
                seg = s[off:off + c]
                yseg = y[off:off + c]
                hits.append(float(yseg[int(torch.argmax(seg))].item()))
                off += c
    return np.array(hits)


def train_epoch(model, loader, opt, device, pos_weight):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        s = model.scores(batch)
        # Only about 3.5% of bonds are cleaved, so an unweighted loss is
        # minimised by predicting "not cleaved" everywhere.  pos_weight is the
        # ratio of negatives to positives in the training split.
        loss = F.binary_cross_entropy_with_logits(
            s, batch.y, pos_weight=torch.tensor(pos_weight, device=device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total += float(loss) * batch.num_graphs
    return total / len(loader.dataset)


def run(variant, train_p, test_p, out_p, epochs=25, bs=32, seed=0, limit=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_all = load_split(train_p, variant, limit)
    te = load_split(test_p, variant, limit)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(tr_all))
    k = int(0.15 * len(tr_all))
    va = [tr_all[i] for i in idx[:k]]
    tr = [tr_all[i] for i in idx[k:]]

    in_dim = tr[0].x.shape[1]
    edge_dim = tr[0].edge_attr.shape[1]
    print('[%s] train %d valid %d test %d | node feat %d edge feat %d | %s'
          % (variant, len(tr), len(va), len(te), in_dim, edge_dim, DEV))

    tr_loader = DataLoader(tr, batch_size=bs, shuffle=True)
    va_loader = DataLoader(va, batch_size=bs)
    te_loader = DataLoader(te, batch_size=bs)

    model = Net(in_dim, edge_dim).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    n_pos = sum(float(g.y.sum()) for g in tr)
    n_tot = sum(float(g.y.numel()) for g in tr)
    pos_weight = (n_tot - n_pos) / max(n_pos, 1.0)
    print('  positives %.4f of bonds -> pos_weight %.1f'
          % (n_pos / n_tot, pos_weight))

    best_acc, best_state, best_ep = -1.0, None, -1
    hist = []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, tr_loader, opt, DEV, pos_weight)
        sched.step()
        vacc = per_molecule_hits(model, va_loader, DEV).mean()
        hist.append(dict(epoch=ep, loss=loss, valid_acc=float(vacc)))
        print('  ep %2d loss %.4f valid %.4f  (%.1fs)'
              % (ep, loss, vacc, time.time() - t0))
        if vacc > best_acc:
            best_acc = vacc
            best_ep = ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    hits = per_molecule_hits(model, te_loader, DEV)
    print('  selected epoch %d (valid %.4f) -> test %.4f'
          % (best_ep, best_acc, hits.mean()))

    json.dump(dict(variant=variant, n=len(te), best_epoch=best_ep,
                   valid_acc=float(best_acc), test_acc=float(hits.mean()),
                   per_molecule=hits.tolist(), history=hist),
              open(out_p, 'w'))
    print('  wrote', out_p)
    return hits.mean()


if __name__ == '__main__':
    variant = sys.argv[1]
    train_p, test_p, out_p = sys.argv[2], sys.argv[3], sys.argv[4]
    lim = None
    if len(sys.argv) > 5 and sys.argv[5].strip():
        v = int(sys.argv[5])
        lim = v if v > 0 else None       # 0 or negative means "no limit"
    ep = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6].strip() else 25
    sd = int(sys.argv[7]) if len(sys.argv) > 7 and sys.argv[7].strip() else 0
    run(variant, train_p, test_p, out_p, epochs=ep, limit=lim, seed=sd)
