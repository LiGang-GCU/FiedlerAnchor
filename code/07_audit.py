"""Stage 7: audit the compiled manuscript against the generated manifest.

Every statistic in the manuscript is produced by 06_emit_tex.py, which records
each decimal string it emitted in paper/manifest.json.  This script checks
that no decimal in the manuscript body lacks such a source.

Figure files are excluded from the audit by compiling a copy of the paper with
\\includegraphics disabled.  Axis tick labels live inside the PDF figures and
are not statistics; including them would make the audit report false drift.
Everything that remains comes either from a generated macro or from prose --
so a hand-typed or stale number shows up as DRIFT.

Exit code 1 if any drift is found.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The manuscript is not part of this repository; point PAPER_DIR at your own
# checkout of the LaTeX source when running the audit.
PAPER = os.environ.get('PAPER_DIR', os.path.join(HERE, 'paper'))


def build_audit_copy(paper):
    """Compile a variant with graphics suppressed, in a scratch directory."""
    tmp = tempfile.mkdtemp(prefix='audit_')
    for name in ('main.tex', 'numbers.tex', 'references.bib'):
        shutil.copy(os.path.join(paper, name), tmp)
    for sub in ('sections', 'tables'):
        shutil.copytree(os.path.join(paper, sub), os.path.join(tmp, sub))
    os.makedirs(os.path.join(tmp, 'figures'), exist_ok=True)

    src = open(os.path.join(tmp, 'main.tex'), encoding='utf-8').read()
    # neutralise every \includegraphics so no vector text reaches the PDF
    src = src.replace('\\usepackage{graphicx}',
                      '\\usepackage{graphicx}\n'
                      '\\renewcommand{\\includegraphics}[2][]{}')
    open(os.path.join(tmp, 'main.tex'), 'w', encoding='utf-8').write(src)

    for _ in range(2):
        subprocess.run(['pdflatex', '-interaction=nonstopmode', 'main.tex'],
                       cwd=tmp, capture_output=True)
    pdf = os.path.join(tmp, 'main.pdf')
    if not os.path.exists(pdf):
        raise RuntimeError('audit copy failed to compile')
    return tmp, pdf


def pdf_text(pdf):
    out = pdf[:-4] + '.txt'
    subprocess.run(['pdftotext', '-layout', pdf, out], capture_output=True)
    return open(out, encoding='utf-8', errors='replace').read()


def norm(s):
    try:
        return round(float(s), 6)
    except ValueError:
        return None


def main():
    manifest = json.load(open(os.path.join(PAPER, 'manifest.json')))
    canonical = {norm(s) for s in manifest} - {None}

    tmp, pdf = build_audit_copy(PAPER)
    text = pdf_text(pdf)

    if 'References' in text:
        text = text.split('References')[0]
    # section headings and cross-references carry dotted numbers
    text = re.sub(r'(?m)^\s*\d+(\.\d+)*\s+\S.*$', ' ', text)
    text = re.sub(r'\b(Section|Table|Figure|Theorem|Proposition|Corollary|'
                  r'Remark|Eq\.?)\s*~?\d+(\.\d+)*', ' ', text)

    found = re.findall(r'(?<![\w.])(\d+\.\d+)(?![\w.])', text)
    values = {}
    for f in found:
        values.setdefault(norm(f), []).append(f)

    drift = [(v, raw) for v, raw in sorted(values.items())
             if v is not None and v not in canonical]

    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print('decimals in manuscript text : %d distinct' % len(values))
    print('generated manifest values   : %d distinct' % len(canonical))
    print()

    if drift:
        print('=== DRIFT: decimals with no generated source ===')
        for v, raw in drift:
            m = re.search(re.escape(raw[0]), text)
            ctx = ' '.join(text[max(0, m.start() - 70):m.end() + 70].split()) \
                if m else ''
            print('  %-10s n=%-3d  ...%s...' % (raw[0], len(raw), ctx))
        print()
        print('AUDIT FAILED: %d drifting decimal(s)' % len(drift))
        shutil.rmtree(tmp, ignore_errors=True)
        return 1

    unused = sorted(v for v in canonical if v not in values)
    print('=== generated values not referenced in the text (%d) ===' % len(unused))
    print('   (unused macros are harmless; listed for completeness)')
    print('   ', ', '.join('%.3f' % v for v in unused[:20]),
          '...' if len(unused) > 20 else '')
    print()
    print('AUDIT PASSED: every decimal in the manuscript traces to the generator')
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
