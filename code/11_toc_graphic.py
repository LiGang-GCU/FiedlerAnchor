"""TOC graphic for the JCIM submission.

ACS requires the graphic at final print size, no larger than
3.25 x 1.75 in = 8.25 x 4.45 cm.  Everything below is laid out in
CENTIMETRES against that box, so the axes, the text and the molecule all
share one unit and the result does not depend on the figure's aspect ratio.

Colour-blind safe throughout: blue / orange / purple / grey.  Never red-green.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')

W_CM, H_CM = 8.25, 4.44

C_CHEM = '#d95f02'      # orange -- chemistry, f < 0
C_STRUCT = '#1f77b4'    # blue   -- structure, f > 0
C_BOTH = '#6f5aa8'      # purple -- shared / the crossing bond
C_GREY = '#8f8f8f'
C_INK = '#1f1f1f'


def mol_xy(smiles):
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    xy = np.array([list(mol.GetConformer().GetAtomPosition(i))[:2]
                   for i in range(mol.GetNumAtoms())])
    return mol, xy


def draw_molecule(ax, smiles, f, box):
    """Molecule drawn with atoms coloured by the sign of the Fiedler vector."""
    mol, xy = mol_xy(smiles)
    x0, y0, w, h = box
    xy = xy - xy.mean(axis=0)
    span = xy.max(axis=0) - xy.min(axis=0)
    span[span == 0] = 1.0
    scale = min(w / span[0], h / span[1]) * 0.95
    xy = xy * scale
    xy += np.array([x0 + w / 2, y0 + h / 2])
    f = np.asarray(f)

    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if f[i] * f[j] < 0:
            ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                    color=C_BOTH, lw=2.0, solid_capstyle='round', zorder=3)
        else:
            ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                    color='#c2c2c2', lw=0.9, solid_capstyle='round', zorder=2)

    r_atom = 0.052
    for i in range(mol.GetNumAtoms()):
        col = C_STRUCT if f[i] > 0 else C_CHEM
        ax.add_patch(plt.Circle(xy[i], r_atom, facecolor=col,
                                edgecolor='white', linewidth=0.45, zorder=4))


def main(num_p, mol_p, outdir):
    R = json.load(open(num_p))
    M = json.load(open(mol_p))
    dec = R['decomp']
    sc = R['scope']

    fig = plt.figure(figsize=(W_CM / 2.54, H_CM / 2.54))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_CM)
    ax.set_ylim(0, H_CM)
    ax.set_aspect('equal')
    ax.axis('off')

    y_title = H_CM - 0.30
    FS_TITLE = 5.2
    FS_NUM = 5.0
    FS_SMALL = 4.2

    # ================= left panel: the molecule =================
    cx0, cw = 0.06, 2.34
    draw_molecule(ax, M['smiles'], M['f'], (cx0, 0.72, cw, 3.34))
    ax.text(cx0 + cw / 2, y_title, 'one molecule', ha='center', va='center',
            fontsize=FS_TITLE, fontweight='bold', color=C_INK)
    ax.text(cx0 + cw / 2, 0.56, 'atom colour = Fiedler sign',
            ha='center', va='center', fontsize=FS_SMALL, color='#6a6a6a')

    # ================= middle panel: decomposition ==============
    bx, bw = 2.78, 2.36
    y_title_mid = y_title
    ax.text(bx + bw / 2, y_title_mid, 'where the lift comes from',
            ha='center', va='center', fontsize=FS_TITLE,
            fontweight='bold', color=C_INK)

    tot = dec['total']
    fracs = [(dec['unique_chem'] / tot, C_CHEM),
             (dec['unique_struct'] / tot, C_STRUCT),
             (dec['shared'] / tot, C_BOTH)]

    bar_y, bar_h = 2.72, 0.50
    x = bx
    for frac, col in fracs:
        wseg = frac * bw
        ax.add_patch(Rectangle((x, bar_y), wseg, bar_h, facecolor=col,
                               edgecolor='white', linewidth=0.9, zorder=3))
        x += wseg

    # percentages: the 19.4% segment is too narrow for an inside label, so
    # chemistry and shared go inside and structure goes above with a tick
    ax.text(bx + fracs[0][0] * bw / 2, bar_y + bar_h / 2,
            '%.1f' % (100 * fracs[0][0]), ha='center', va='center',
            fontsize=FS_NUM + 0.6, fontweight='bold', color='white', zorder=5)
    ax.text(bx + bw - fracs[2][0] * bw / 2, bar_y + bar_h / 2,
            '%.1f' % (100 * fracs[2][0]), ha='center', va='center',
            fontsize=FS_NUM, fontweight='bold', color='white', zorder=5)
    x_struct = bx + fracs[0][0] * bw + fracs[1][0] * bw / 2
    ax.text(x_struct, bar_y + bar_h + 0.20, '%.1f' % (100 * fracs[1][0]),
            ha='center', va='bottom', fontsize=FS_NUM, fontweight='bold',
            color=C_STRUCT)
    ax.plot([x_struct, x_struct], [bar_y + bar_h + 0.02, bar_y + bar_h + 0.17],
            color=C_STRUCT, lw=0.5)

    # legend under the bar
    ly = bar_y - 0.34
    ax.text(bx, ly, 'chemistry', ha='left', va='center',
            fontsize=FS_SMALL, color=C_CHEM)
    ax.text(bx + bw / 2, ly, 'structure', ha='center', va='center',
            fontsize=FS_SMALL, color=C_STRUCT)
    ax.text(bx + bw, ly, 'shared', ha='right', va='center',
            fontsize=FS_SMALL, color=C_BOTH)

    ax.plot([bx, bx + bw], [ly - 0.36, ly - 0.36], color=C_GREY, lw=0.6, ls=':')
    ax.text(bx + bw / 2, ly - 0.62, 'random baseline %.3f' % dec['A'],
            ha='center', va='center', fontsize=FS_SMALL, color=C_GREY)

    # ================= right panel: the two regimes =============
    rx, rw = 5.52, 2.62
    ax.text(rx + rw / 2, y_title, 'the Fiedler guarantee', ha='center',
            va='center', fontsize=FS_TITLE, fontweight='bold', color=C_INK)

    top_y, top_h = 2.46, 1.06
    ax.add_patch(FancyBboxPatch((rx, top_y), rw, top_h,
                                boxstyle='round,pad=0.02',
                                facecolor='#e9f0f8', edgecolor=C_STRUCT,
                                linewidth=0.8, zorder=3))
    ax.text(rx + rw / 2, top_y + top_h - 0.26, 'SBE = 1', ha='center',
            va='center', fontsize=FS_NUM, fontweight='bold', color='#17496f')
    ax.text(rx + rw / 2, top_y + top_h - 0.62,
            '%.3f' % sc['qualified']['acc'], ha='center', va='center',
            fontsize=FS_NUM + 3.6, fontweight='bold', color='#17496f')
    ax.text(rx + rw / 2, top_y + 0.20, 'never a ring bond', ha='center',
            va='center', fontsize=FS_SMALL + 0.3, color='#3c6f9c')

    bot_y, bot_h = 0.94, 1.06
    ax.add_patch(FancyBboxPatch((rx, bot_y), rw, bot_h,
                                boxstyle='round,pad=0.02',
                                facecolor='#fdf0e4', edgecolor=C_CHEM,
                                linewidth=0.8, zorder=3))
    ax.text(rx + rw / 2, bot_y + bot_h - 0.26, 'SBE > 1', ha='center',
            va='center', fontsize=FS_NUM, fontweight='bold', color='#8a3d02')
    ax.text(rx + rw / 2, bot_y + bot_h - 0.62,
            '%.3f' % sc['unqualified']['acc'], ha='center', va='center',
            fontsize=FS_NUM + 3.6, fontweight='bold', color='#8a3d02')
    ax.text(rx + rw / 2, bot_y + 0.20, 'always a ring bond', ha='center',
            va='center', fontsize=FS_SMALL + 0.3, color='#a8632a')

    ax.annotate('', xy=(rx + rw / 2, bot_y + bot_h + 0.05),
                xytext=(rx + rw / 2, top_y - 0.05),
                arrowprops=dict(arrowstyle='-|>', color='#8a8a8a',
                                lw=0.7, mutation_scale=4.5))

    # ================= flow arrows between panels ===============
    for x0, x1 in ((2.46, 2.72), (5.20, 5.46)):
        ax.annotate('', xy=(x1, 2.99), xytext=(x0, 2.99),
                    arrowprops=dict(arrowstyle='-|>', color='#9a9a9a',
                                    lw=0.8, mutation_scale=5))

    # ================= one-line takeaway along the bottom =======
    ax.plot([0.06, W_CM - 0.06], [0.40, 0.40], color='#d8d8d8', lw=0.5)
    ax.text(W_CM / 2, 0.22,
            'spectral selection measures a real structural signal, '
            'but only inside a regime that can be named in advance',
            ha='center', va='center', fontsize=FS_SMALL, style='italic',
            color='#555555')

    out = os.path.join(outdir, 'toc_graphic.pdf')
    fig.savefig(out, dpi=600, format='pdf')
    fig.savefig(out.replace('.pdf', '.png'), dpi=600, format='png')
    plt.close(fig)
    print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2],
         sys.argv[3] if len(sys.argv) > 3 else '.')
