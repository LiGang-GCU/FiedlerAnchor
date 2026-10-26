# FiedlerAnchor v4 — measurement study: what spectral bond selection measures

Code, results and figures for a measurement study of bond-cleavage prediction
on USPTO-50K. The manuscript itself is **not** in this repository.

## What this repository contains

| Path | Contents |
|---|---|
| `code/` | The full pipeline, stages 01–10 |
| `data/*.json` | Result files: per-molecule predictions, ablation, statistics |
| `data/*.jsonl` | Per-molecule pools for the retained test split |
| `figures/` | Rendered figures (PDF + PNG) |
| `logs/` | Training logs from the ablation and GNN runs |

## The single-source-of-numbers rule

No statistic in the manuscript is typed by hand. The chain is:

```
data/pool_test.jsonl  ->  data/mol_test.jsonl  ->  data/numbers.json
                                                  |
                              code/06_emit_tex.py -+-> paper/numbers.tex
                                                       paper/tables/*.tex
                                                       paper/manifest.json
                                                  |
                              code/07_audit.py ----+-> checks every decimal
                                                       in the compiled text
```

`06_emit_tex.py` records every decimal string it emits in `manifest.json`;
`07_audit.py` then fails if the manuscript contains a decimal that has no
entry there. To run the audit you need a checkout of the LaTeX source and
must point at it:

```bash
PAPER_DIR=/path/to/paper python code/07_audit.py
```

## Pipeline

Run from the repository root. Each stage takes its paths as arguments.

| Stage | Script | Reads | Writes |
|---|---|---|---|
| 1 | `01_extract.py <test.csv> <train.csv> <outdir>` | atom-mapped reactions | `pool_*.jsonl`, `pool_*_flow.json` |
| 2 | `02_features.py <pool.jsonl> <feat.jsonl>` | bond pool | per-bond feature table (large, not tracked) |
| 3 | `03_models.py` | features | `mol_test.jsonl` |
| 4 | `04_ablation.py` | features | `ablation.json` |
| 4b | `04b_falsify.py` | features | `falsify.json` |
| 5 | `05_stats.py` | `mol_test.jsonl`, `ablation.json` | `numbers.json` |
| 6 | `06_emit_tex.py <numbers.json> <falsify.json> <paper_dir>` | result files | LaTeX macros + tables |
| 7 | `07_audit.py` | compiled manuscript + manifest | audit verdict |
| 8 | `08_figures.py` | `numbers.json` | `figures/*.pdf` |
| 9 | `09_gnn.py`, `09d_multiseed.py` | features | `gnn_runs/*.json` |
| 9b/c/e | invariance, leakage, ring-redundancy audits | features | audit JSONs |
| 10 | `10_gnn_stats.py` | `gnn_runs/*.json` | `gnn_numbers.json` |

## Data

The pipeline expects the USPTO-50K atom-mapped split files as two CSVs. They
are not redistributed here; obtain them from the original USPTO-50K release.
The retained analysis population is 3587 test molecules (from 5007 records)
and 17665 training molecules (from 24587 records); the filters are enumerated
in stage 1 and in the manuscript's data section.

`data/pool_train.jsonl` (108 MB) is excluded from version control. Regenerate
it with `01_extract.py`, or drop it and re-run from stage 2 onward if you only
need the test-side results — every reported number is computed on the test
split.

## Notable implementation points

- **Random baseline is analytic**, not sampled: `E[hit] = |T_i| / |E_i|` over
  each molecule's own candidate pool. Sampling noise at this pool size is
  large enough to reorder close selectors.
- **Confidence intervals** are molecule-level bootstrap (4000 resamples) for
  tree models, and Student-t over 5 paired seeds for the GNN. The GNN uses t
  rather than bootstrap because bootstrap undercovers at n = 5.
- **Seed discovery is by glob**, not a fixed range: `10_gnn_stats.py` reads
  every `gnn_<variant>_s*.json` present. A hard-coded seed range silently
  dropped two of the five runs in an earlier version.
- **Colour palette** is orange / blue / purple / grey — no red-green pairing.

## Requirements

```
python >= 3.10
rdkit 2026.03.1      networkx 3.6.1     numpy 2.3.5
scipy 1.16.3         scikit-learn 1.8.0 matplotlib 3.10.8
torch 2.6.0          torch_geometric
```

## License

MIT.
