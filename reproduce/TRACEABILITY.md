# Traceability matrix — paper artefact → input → script → output → match

Status: ✅ reproduced & matches the paper · ⏳ running · ◻ script present, not yet run.

## Where the inputs live
All paper inputs and reference outputs live under a single data root (the extracted Zenodo archive, or any directory you point `config.sh` at):
- `<DATA_ROOT>/` — `data/` (genomes+CDS), `schemas/{lm_cgmlst,se_wgmlst,ec_wgmlst,cj_wgmlst}` (the exact paper snapshots), `results/` (precomputed reference outputs: `cc_fixed2`, `c354`, `c3_3310`, and end-to-end variants).
- `<DATA_ROOT>/` — `cgmlst_loci/*_cgmlst_loci.txt`, and the FDA outbreak inputs.

Set `BENCH_BASE` and `PAPER_BASE` to `<DATA_ROOT>` in `config.sh`; the analysis scripts then regenerate the tables directly from the precomputed reference outputs.

Protocol notes:
- chewBBACA shared-CDS REQUIRES `--cds` (fixed in `02_run_chewbbaca_BeONE.sh`).
- chewBBACA 3.5.4 hashed flag: `--hash-profiles`; CDS-input FILE ids get a `.cds` infix (stripped by the comparators).

## Core results — regenerated from the reference outputs

| Paper artefact | Script / target | Reproduced value | Status |
|---|---|---|---|
| **Table 2** `tab:concordance` — CRC32 concordance (shared-CDS & end-to-end) | `04 --mode shcds,e2e` | **0 diffs on all 8 datasets**, both protocols (vs 3.3.10 and 3.5.4) | ✅ |
| **Table 5** `tab:cgmlst_split` — actionable cg/wg split | `09_table_cgmlst_split.py` | **every cell matches** (Se 339/1258, Cj pub 4522, …) | ✅ |
| **Suppl S3/S4** `tab-S3`/`tab-S4` — class transitions | `07_supp_S3_classification_transitions.py` | regenerated | ✅ |
| **Suppl Table S5** — actionable mechanism breakdown + 48-cell sample | `14_supp_diagnostic_cases.py` | 13,351 actionable; M1 6,335 / M3 1,963 / M4 4,281 | ✅ |
| **Figure 3** `fig:discordance_anatomy` | `06_figure2_discordance_anatomy.py` | regenerated | ✅ |
| **Figures 1–2** `fig:filter_verify`, `fig:audit` | static TikZ | n/a | ✅ |
| **Table 6** `tab:clustering` — cgMLST clustering | `05_table_clustering.py` | **0 discordant pairs on all 8 datasets**, pair counts match (11.94M total) | ✅ |
| **Table 1/3** `tab:runtime`/`_e2e` — runtime/speedup | `04` (from `timing_*.txt`) | speedups in range (8–16×); exact wall-clock is machine-dependent | ✅ (magnitude) |
| **Table 7** `tab:outbreak` — FDA outbreak | `compare_outbreak.py` (from the reference outputs) | **MST + single-linkage clusters IDENTICAL, 0 outgroup intrusions, all 4 organisms** (Se differs at the pairwise-count level only; cluster assignment identical) | ✅ |
| **Suppl S2** — pyrodigal equivalence | `08_supp_S2_pyrodigal_equivalence.py --genome` | ran (lm 2905 / se 4547 / ec 5251 / cj 1868 CDS) | ✅ |
| **Suppl** `tab:ablation` — minimizer ablation | `benchmark_scripts/ablation_minimizer.sh` | k=5,w=5 ref = 100%; only k=3 drops to ~99.9% | ✅ |

## Algorithmic extensions (§ Filter safety)

| Artefact | Script / target | Reproduced value | Status |
|---|---|---|---|
| **Table** `tab:filter_safety` | `make remedy ext-table` | Lm 29/31, Se 221/242, Ec 173/200, Cj 12/12; wcr→0.20 | ✅ |
| Determinism (hash vs lexicographic) | `make determinism` | 0 diff / 2,472,326 (Lm) | ✅ |
| GPU vs CPU | `make gpu` | 0 diff / 2,472,326 (Lm) | ✅ |
| Pareto `(k,w,τ,κ)` | `make pareto` | κ=30 Pareto-dominated on all 4 schemas | ✅ |

## Bottom line
With `config.sh` pointed at the data root, the headline correctness tables (concordance Tab 2, actionable Tab 5) and the supplementary tables/figures regenerate and **match the paper exactly**. Runtime figures (Tables 1/3) are machine-dependent; all correctness results are exact.
