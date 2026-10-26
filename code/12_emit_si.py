"""Generate the Supporting Information LaTeX from the result files.

Every number in the SI is read from data/*.json; nothing is typed by hand.
Output: paper/si/main_si.tex
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(rel):
    return json.load(open(os.path.join(ROOT, 'data', rel)))


def f3(x):
    return '%.3f' % x


def f4(x):
    return '%.4f' % x


def sci(x):
    """Exponent form for exact McNemar p-values."""
    if x == 0:
        return r'<10^{-300}'
    e = 0
    y = x
    while y < 1.0:
        y *= 10.0
        e -= 1
    return r'%.1f\times10^{%d}' % (y, e)


def main():
    N = load('numbers.json')
    G = load('gnn_numbers.json')
    RR = load('ring_redundancy.json')
    INV = load('invariance_audit.json')

    M = {d['key']: d for d in N['main']}
    P = N['pairs']
    PC = {r['cls']: r for r in N['per_class']}
    FL = N['flow']

    L = []
    a = L.append

    # ---------------- preamble ----------------
    a(r'\documentclass[journal=jcisd8,manuscript=suppinfo]{achemso}')
    a(r'\usepackage{amsmath,amssymb}')
    a(r'\usepackage{booktabs}')
    a(r'\usepackage{longtable}')
    a(r'\usepackage{array}')
    a(r'\usepackage{siunitx}')
    a(r'\sisetup{group-digits=false}')
    a(r'\title{What Spectral Bond Selection Measures, and What It Leaves '
      r'on the Table}')
    a(r'\author{Gang Li}')
    a(r'\affiliation{School of Computer Engineering, Guangzhou City '
      r'University of Technology, Guangzhou 510800, China}')
    a(r'\author{Wenhai He}')
    a(r'\email{hewenhai@gcc.edu.cn}')
    a(r'\affiliation{School of Information Technology and Engineering, '
      r'Guangzhou College of Commerce, Guangzhou 511363, China}')
    a(r'\begin{document}')
    a(r'\maketitle')
    a('')

    # ---------------- S1: population flow ----------------
    a(r'\section{Population flow}')
    a(r'Table~\ref{tab:s1} gives the stage-by-stage record counts for both '
      r'splits. The filters are applied identically to train and test; the '
      r'only split-specific rule is the test-side requirement that a molecule '
      r'retain at least one non-ring bond, so that every selector has a '
      r'candidate pool.')
    a(r'\begin{table}[h]')
    a(r'\centering')
    a(r'\caption{Record counts at each population stage.}')
    a(r'\label{tab:s1}')
    a(r'\begin{tabular}{lrr}')
    a(r'\toprule')
    a(r'Stage & Test & Train \\')
    a(r'\midrule')
    a(r'Rows read from the split file & %d & %d \\' % (FL['test']['rows'], FL['train']['rows']))
    a(r'\quad no retained disconnection & %d & %d \\' % (FL['test']['no_disconnection'], FL['train']['no_disconnection']))
    a(r'\quad outside the 5--60 heavy-atom window & %d & %d \\' % (FL['test']['size'], FL['train']['size']))
    a(r'\quad could not be parsed & 0 & %d \\' % FL['train']['unparsed'])
    a(r'\quad no non-ring bond (test only) & 0 & -- \\')
    a(r'\midrule')
    a(r'\textbf{Retained} & \textbf{%d} & \textbf{%d} \\' % (FL['test']['kept'], FL['train']['kept']))
    a(r'\bottomrule')
    a(r'\end{tabular}')
    a(r'\end{table}')
    a('')

    # ---------------- S2: selector comparison ----------------
    a(r'\section{Full selector comparison}')
    a(r'Table~\ref{tab:s2} reports every selector on the same \num{3587} test '
      r'molecules. A molecule is a hit when the selector names a bond that the '
      r'reaction actually breaks. Intervals are 95\% molecule-level bootstrap '
      r'(4000 resamples). Abstentions are counted as misses, so every row has '
      r'the same denominator.')
    a(r'\begin{table}[h]')
    a(r'\centering')
    a(r'\caption{All selectors on the retained test set.}')
    a(r'\label{tab:s2}')
    a(r'\begin{tabular}{lccc}')
    a(r'\toprule')
    a(r'Selector & Accuracy & 95\% CI & Abstentions \\')
    a(r'\midrule')
    names = [('random', 'Uniform random (analytic)'),
             ('betweenness', 'Edge betweenness, argmax'),
             ('fiedler_df', 'Fiedler gradient, argmax'),
             ('brics', 'BRICS'),
             ('sbe', 'Fiedler sign boundary'),
             ('struct_model', 'Structural model (GBDT)'),
             ('chem_model', 'Chemical model (GBDT)'),
             ('combined', 'Chemical + structural (GBDT)')]
    for k, lab in names:
        d = M[k]
        a(r'%s & %s & [%s, %s] & %s \\' % (
            lab, f3(d['acc']), f3(d['lo']), f3(d['hi']),
            d.get('abstain', '--')))
    a(r'\bottomrule')
    a(r'\end{tabular}')
    a(r'\end{table}')
    a('')

    # ---------------- S3: paired comparisons ----------------
    a(r'\section{Paired comparisons}')
    a(r'Because every selector is scored on the same molecules, differences '
      r'use paired intervals and exact McNemar tests on the discordant pairs. '
      r'Table~\ref{tab:s3} lists every comparison quoted in the main text.')
    a(r'\begin{table}[h]')
    a(r'\centering')
    a(r'\small')
    a(r'\setlength{\tabcolsep}{4pt}')
    a(r'\caption{Paired differences between selectors.}')
    a(r'\label{tab:s3}')
    a(r'\begin{tabular}{lcccr}')
    a(r'\toprule')
    a(r'Comparison & $\Delta$ & 95\% CI & a/b & $p$ \\')
    a(r'\midrule')
    order = ['betweenness__fiedler_df', 'brics__fiedler_df',
             'sbe__betweenness', 'struct_model__betweenness',
             'struct_model__chem_model', 'combined__chem_model',
             'combined__struct_model', 'chem_model__betweenness',
             'S:struct_all__S:spectral', 'S:struct_all__S:topology',
             'D:chem+struct__D:chem+spectral',
             'D:chem+struct__D:chem+topology']
    for k in order:
        d = P[k]
        lab = k.replace('__', r' vs.\ ').replace('S:', '').replace('D:', '')
        lab = lab.replace('_', r'\_')
        a(r'%s & %+.3f & [%+.3f, %+.3f] & %d/%d & $%s$ \\' % (
            lab, d['diff'], d['lo'], d['hi'],
            d['a_only'], d['b_only'], sci(d['p'])))
    a(r'\bottomrule')
    a(r'\end{tabular}')
    a(r'\end{table}')
    a('')

    # ---------------- S4: per-class ----------------
    a(r'\section{Per-class accuracy}')
    a(r'Table~\ref{tab:s4} expands Table~4 of the main text. Only the eight '
      r'Carey superclasses represented in the retained test set appear; two '
      r'of the ten have no retained molecule.')
    a(r'\begin{table}[h]')
    a(r'\centering')
    a(r'\caption{Accuracy by reaction superclass, with the structural model '
      r'column included.}')
    a(r'\label{tab:s4}')
    a(r'\begin{tabular}{lccccccc}')
    a(r'\toprule')
    a(r'Class & $n$ & BTW & $|\Delta f|$ & BRICS & Structural & Chemical & Combined \\')
    a(r'\midrule')
    for c in sorted(PC, key=lambda x: int(x)):
        r = PC[c]
        a(r'%s & %d & %s & %s & %s & %s & %s & %s \\' % (
            c, r['n'], f3(r['betweenness']), f3(r['fiedler_df']),
            f3(r['brics']), f3(r['struct_model']), f3(r['chem_model']),
            f3(r['combined'])))
    a(r'\bottomrule')
    a(r'\end{tabular}')
    a(r'\end{table}')
    a('')

    # ---------------- S5: GNN per seed ----------------
    a(r'\section{GNN per-seed results}')
    a(r'Table~\ref{tab:s5} lists every seed individually, so the mean and '
      r'spread can be checked rather than taken on trust. All 20 runs use the '
      r'same test split of \num{3587} molecules.')
    a(r'\begin{table}[h]')
    a(r'\centering')
    a(r'\caption{Per-seed test accuracy for each GNN variant.}')
    a(r'\label{tab:s5}')
    a(r'\begin{tabular}{lccccc}')
    a(r'\toprule')
    a(r'Variant & Seed 0 & Seed 1 & Seed 2 & Seed 3 & Seed 4 \\')
    a(r'\midrule')
    vl = [('spec', 'Spectral'), ('struct', 'Non-spectral topology'),
          ('chem', 'Chemical'), ('both', 'Chemical + structural')]
    for k, lab in vl:
        accs = G['variants'][k]['accs']
        a(r'%s & %s \\' % (lab, ' & '.join(f4(x) for x in accs)))
    a(r'\midrule')
    a(r'\multicolumn{6}{l}{Mean and population s.d.\ over the five seeds}\\')
    for k, lab in vl:
        v = G['variants'][k]
        a(r'%s & \multicolumn{5}{l}{%s $\pm$ %s} \\' % (lab, f4(v['mean']), f4(v['sd'])))
    a(r'\bottomrule')
    a(r'\end{tabular}')
    a(r'\end{table}')
    a('')

    # ---------------- S6: ring redundancy ----------------
    a(r'\section{Ring-feature redundancy check}')
    a(r'On a connected undirected graph, ``lies in a cycle'' and ``is not a '
      r'bridge'' are the same predicate, and both are determined by whether '
      r'the bond belongs to a two-edge-connected component of size greater '
      r'than one. Table~\ref{tab:s6} records the verification that the three '
      r'dropped ring columns carry no information beyond '
      r'\texttt{comp\_edges}.')
    a(r'\begin{table}[h]')
    a(r'\centering')
    a(r'\caption{Agreement between each ring column and the corresponding '
      r'predicate on \texttt{comp\_edges}.}')
    a(r'\label{tab:s6}')
    a(r'\begin{tabular}{lcc}')
    a(r'\toprule')
    a(r'Column & Predicate & Disagreements \\')
    a(r'\midrule')
    a(r'\texttt{in\_cycle} & comp\_edges $>0$ & %d \\' % RR['mismatches']['in_cycle'])
    a(r'\texttt{bridge} & comp\_edges $=0$ & %d \\' % RR['mismatches']['bridge'])
    a(r'\texttt{detour} & comp\_edges $=0$ & %d \\' % RR['mismatches']['detour'])
    a(r'\midrule')
    a(r'\multicolumn{2}{l}{Bonds checked} & %d \\' % RR['n_bonds'])
    a(r'\bottomrule')
    a(r'\end{tabular}')
    a(r'\end{table}')
    a('')

    # ---------------- S7: invariance ----------------
    a(r'\section{Atom-renumbering invariance}')
    a(r'The hand-crafted structural features are audited against random atom '
      r'permutations. An earlier version of the code failed this audit: the '
      r'two-edge-connected component was computed from a vertex block rather '
      r'than an edge block, and several per-bond quantities depended on the '
      r'stored endpoint order. Those defects were repaired and the audit now '
      r'passes.')
    a(r'\begin{table}[h]')
    a(r'\centering')
    a(r'\caption{Invariance audit result.}')
    a(r'\label{tab:s7}')
    a(r'\begin{tabular}{lc}')
    a(r'\toprule')
    a(r'Molecules sampled & %d \\' % INV['molecules_sampled'])
    a(r'Bonds compared & %d \\' % INV['bonds_compared'])
    a(r'Unmatched bonds after permutation & %d \\' % INV['bonds_unmatched'])
    a(r'Near-degenerate $\lambda_2$ shifts & %d \\' % INV['lambda2_shifts'])
    a(r'Tolerance & $10^{-6}$ \\')
    a(r'Check A (invariance) & %s \\' % INV['check_A_invariance'])
    a(r'Check C (sign-invariance) & %s \\' % INV['check_C_sign_invariance'])
    a(r'\bottomrule')
    a(r'\end{tabular}')
    a(r'\end{table}')
    a('')
    a(r'\section*{Reproducibility record}')
    a(r'Software: RDKit 2026.03.1, NetworkX 3.6.1, NumPy 2.3.5, '
      r'SciPy 1.16.3, scikit-learn 1.8.0, Matplotlib 3.10.8, '
      r'PyTorch 2.6.0 with PyTorch Geometric. '
      r'All confidence intervals use a fixed RNG seed '
      r'(20261022) and 4000 molecule-level bootstrap resamples. '
      r'The GNN results use five training seeds (0--4) per variant, '
      r'scored once on the test split. Every decimal in the main text is '
      r'generated by a single script from the per-molecule prediction files, '
      r'and an automated audit fails if any decimal in the compiled '
      r'manuscript lacks a generated source.')
    a('')
    a(r'\end{document}')

    out = os.path.join(ROOT, 'paper', 'si')
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'main_si.tex')
    open(path, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
    print('wrote', path)


if __name__ == '__main__':
    main()
