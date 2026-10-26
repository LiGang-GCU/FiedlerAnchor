"""Stage 8: figures.  Every plotted value is read from numbers.json, so the
figure cannot disagree with the tables or the text."""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The manuscript is not part of this repository; point PAPER_DIR at your own
# checkout of the LaTeX source if you want the figures written next to it.
PAPER = os.environ.get('PAPER_DIR', os.path.join(HERE, 'paper'))

# Colour-blind safe: orange / blue / purple / grey. Never red-green.
C_STRUCT = '#1f77b4'
C_CHEM = '#d95f02'
C_BOTH = '#7570b3'
C_GREY = '#999999'


def fig_ablation(R, outdir):
    AB = {d['key']: d for d in R['ablation']}
    keys = [('A:random', 'Random'),
            ('S:spectral', 'Spectral\nonly'),
            ('S:topology', 'Non-spectral\ntopology'),
            ('S:struct_all', 'All structural\nfeatures'),
            ('C:chem_all', 'Chemical\nonly'),
            ('D:chem+spectral', 'Chemical\n+ spectral'),
            ('D:chem+topology', 'Chemical\n+ topology'),
            ('D:chem+struct', 'Chemical\n+ structural')]
    lab, val, lo, hi, col = [], [], [], [], []
    for k, l in keys:
        d = AB[k]
        lab.append(l)
        val.append(d['acc'])
        lo.append(d['acc'] - d['lo'])
        hi.append(d['hi'] - d['acc'])
        if k.startswith('A'):
            col.append(C_GREY)
        elif k.startswith('S'):
            col.append(C_STRUCT)
        elif k.startswith('C'):
            col.append(C_CHEM)
        else:
            col.append(C_BOTH)

    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    x = np.arange(len(lab))
    ax.bar(x, val, yerr=[lo, hi], color=col, width=0.68,
           error_kw=dict(elinewidth=0.9, capsize=2.2, ecolor='#333333'))
    ax.axhline(R['decomp']['A'], ls=':', lw=1.0, color=C_GREY)
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=7.5)
    ax.set_ylabel('Accuracy', fontsize=9)
    ax.set_ylim(0, 0.82)
    ax.tick_params(axis='y', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_ablation.pdf'))
    fig.savefig(os.path.join(outdir, 'fig_ablation.png'), dpi=200)
    plt.close(fig)


def fig_scope(R, outdir):
    S = R['scope']
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))

    names = ['Inside\nhypothesis', 'Outside\nhypothesis']
    ring = [100 * S['qualified']['ring_pick'], 100 * S['unqualified']['ring_pick']]
    acc = [S['qualified']['acc'], S['unqualified']['acc']]
    bas = [R['decomp']['A'], R['decomp']['A']]

    ax = axes[0]
    x = np.arange(2)
    ax.bar(x, ring, color=[C_STRUCT, C_CHEM], width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('Ring-bond picks (%)', fontsize=8.5)
    ax.set_ylim(0, 112)
    ax.tick_params(labelsize=7.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, v in enumerate(ring):
        ax.text(i, v + 3, '%.1f' % v, ha='center', fontsize=8)

    ax = axes[1]
    ax.bar(x - 0.17, acc, width=0.34, color=[C_STRUCT, C_CHEM],
           label='SBE rule')
    ax.bar(x + 0.17, bas, width=0.34, color=C_GREY, label='Random')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('Accuracy', fontsize=8.5)
    ax.set_ylim(0, 0.34)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7, frameon=False, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, v in enumerate(acc):
        ax.text(i - 0.17, v + 0.008, '%.3f' % v, ha='center', fontsize=7.5)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_scope.pdf'))
    fig.savefig(os.path.join(outdir, 'fig_scope.png'), dpi=200)
    plt.close(fig)


def fig_decomp(R, outdir):
    D = R['decomp']
    C = R['decomp_ci']
    parts = [('Unique to\nchemistry', D['unique_chem'],
              C['unique_chem']),
             ('Unique to\nstructure', D['unique_struct'],
              C['unique_struct']),
             ('Shared', D['shared'], C['shared'])]
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    x = np.arange(3)
    vals = [p[1] for p in parts]
    err = [[p[1] - p[2][0] for p in parts], [p[2][1] - p[1] for p in parts]]
    ax.bar(x, vals, yerr=err, color=[C_CHEM, C_STRUCT, C_BOTH], width=0.55,
           error_kw=dict(elinewidth=0.9, capsize=2.2, ecolor='#333333'))
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in parts], fontsize=8)
    ax.set_ylabel('Share of lift over random', fontsize=8.5)
    ax.set_ylim(0, 0.30)
    ax.tick_params(labelsize=7.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.022, '%.3f' % v, ha='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_decomp.pdf'))
    fig.savefig(os.path.join(outdir, 'fig_decomp.png'), dpi=200)
    plt.close(fig)


def main(num_p, outdir):
    R = json.load(open(num_p))
    os.makedirs(outdir, exist_ok=True)
    fig_ablation(R, outdir)
    fig_scope(R, outdir)
    fig_decomp(R, outdir)
    print('figures written to', outdir)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
