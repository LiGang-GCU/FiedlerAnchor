"""TOC graphic for the JCIM submission.

ACS specifies 8.25 x 4.44 cm at print size.  The graphic must survive
shrinking to a thumbnail, so it carries one visual idea: the same bond pool
split two ways, and the theorem's regime partition on the right.

Colour-blind safe: blue / orange / purple / grey.  Never red-green.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.environ.get('PAPER_DIR', os.path.join(HERE, 'paper'))

C_CHEM = '#d95f02'
C_STRUCT = '#1f77b4'
C_BOTH = '#7570b3'
C_GREY = '#999999'


def main(num_p, outdir):
    R = json.load(open(num_p))
    dec = R['decomp']
    sc = R['scope']

    W, H = 8.25 / 2.54, 4.44 / 2.54      # inches, ACS print size
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # ---- left: the decomposition as a stacked bar ----------------
    tot = dec['total']
    parts = [(dec['unique_chem'] / tot, C_CHEM, 'chem'),
             (dec['unique_struct'] / tot, C_STRUCT, 'struct'),
             (dec['shared'] / tot, C_BOTH, 'shared')]
    y0, hgt = 0.60, 0.20
    x = 0.04
    wbar = 0.52
    for frac, col, _ in parts:
        w = frac * wbar
        ax.add_patch(Rectangle((x, y0), w, hgt, facecolor=col,
                               edgecolor='white', linewidth=1.0))
        x += w
    ax.text(0.04, 0.86, 'where the lift comes from',
            fontsize=5.6, fontweight='bold', color='#222222')
    # legend: two lines, so the three labels cannot collide at print size
    ax.text(0.04, y0 - 0.075, 'chemistry %.1f%%' % (100 * parts[0][0]),
            fontsize=4.9, color=C_CHEM, ha='left')
    ax.text(0.30, y0 - 0.075, 'structure %.1f%%' % (100 * parts[1][0]),
            fontsize=4.9, color=C_STRUCT, ha='left')
    ax.text(0.04, y0 - 0.145, 'shared %.1f%%' % (100 * parts[2][0]),
            fontsize=4.9, color=C_BOTH, ha='left')

    # ---- left bottom: uniform random reference ----------------
    ax.plot([0.04, 0.04 + wbar], [0.40, 0.40], color=C_GREY, lw=1.0, ls=':')
    ax.text(0.04, 0.33, 'random baseline %.3f' % dec['A'],
            fontsize=4.8, color=C_GREY)

    # ---- right: the theorem's regime split --------------------
    ax.add_patch(FancyBboxPatch((0.61, 0.62), 0.15, 0.18,
                                boxstyle='round,pad=0.01',
                                facecolor='#e8eef7', edgecolor=C_STRUCT,
                                linewidth=0.8))
    ax.text(0.685, 0.71, 'SBE = 1', fontsize=5.0, ha='center',
            color='#1a4a72', fontweight='bold')
    ax.text(0.685, 0.665, 'acc %.3f' % sc['qualified']['acc'],
            fontsize=4.8, ha='center', color='#1a4a72')

    ax.add_patch(FancyBboxPatch((0.61, 0.30), 0.15, 0.18,
                                boxstyle='round,pad=0.01',
                                facecolor='#fdeee2', edgecolor=C_CHEM,
                                linewidth=0.8))
    ax.text(0.685, 0.39, 'SBE > 1', fontsize=5.0, ha='center',
            color='#8a3d02', fontweight='bold')
    ax.text(0.685, 0.345, 'acc %.3f' % sc['unqualified']['acc'],
            fontsize=4.8, ha='center', color='#8a3d02')

    ax.text(0.61, 0.87, "the Fiedler guarantee", fontsize=5.6,
            fontweight='bold', color='#222222')
    ax.text(0.61, 0.19, 'good rule inside,\nworse than chance outside',
            fontsize=4.6, color='#555555', va='top')

    arrow = FancyArrowPatch((0.58, 0.50), (0.60, 0.50),
                            arrowstyle='-|>', mutation_scale=6,
                            color='#666666', linewidth=0.8)
    ax.add_patch(arrow)

    out = os.path.join(outdir, 'toc_graphic.pdf')
    fig.savefig(out, dpi=600, format='pdf')
    fig.savefig(out.replace('.pdf', '.png'), dpi=600, format='png')
    plt.close(fig)
    print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
