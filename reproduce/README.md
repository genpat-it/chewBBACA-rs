# Reproducing the chewcall paper

Every table and figure in the paper is produced by exactly one script in
this directory.  The naming convention is

    NN_{kind}_{paper-artefact-name}.{ext}

so `04_tables_runtime_concordance.py` produces Tables 1--4, while
`05_table_clustering.py` produces Table 7 and so on.

## One-click reproduction (Makefile)

Everything is driven by the `Makefile` in this directory:

```bash
make check-inputs   # report which tools / data / conda envs are present
make all            # core: chewcall + chewBBACA refs + FDA + tables/figures
make extensions     # §2 analyses: remedy + pareto + determinism + gpu + table
make verify         # convergence gate: chewcall vs chewBBACA, cell-by-cell CRC32
make repro-all      # TOTAL reproduction (all + extensions + verify)
```

Run `make help` for the full target list. Start with `make check-inputs`: it
tells you exactly which binaries, libraries (parasail, CUDA), conda
environments (`chewbbaca_354`, `chewbbaca_3310`) and per-organism schema /
genome / CDS directories are present, and what to stage if something is
missing.

## Layout

```
paper/reproduce/
├── config.sh                              ← all paths (BASE, schemas, CDS dirs, …)
├── 01_run_chewcall_BeONE.sh               ← chewcall on 8 BeONE datasets (shcds + e2e × 3 reps)
├── 02_run_chewbbaca_BeONE.sh              ← chewBBACA v3.5.4 / v3.3.10 references
├── 03_run_FDA_outbreak.sh                 ← chewcall + chewBBACA on the 4 FDA outbreak datasets
├── 04_tables_runtime_concordance.py       ← Tables 1, 2 (shcds) and 3, 4 (e2e)
├── 05_table_clustering.py                 ← Table 7 (cgMLST clustering)
├── 06_figure2_discordance_anatomy.py      ← Figure 2 (heat-map LaTeX block)
├── 07_supp_S3_classification_transitions.py  ← Supplementary Tables S3 + S4
├── 08_supp_S2_pyrodigal_equivalence.py    ← Supplementary Section S2 paragraph
├── 09_table_cgmlst_split.py               ← Table 5 (cgMLST/wgMLST actionable split)
└── README.md                              ← this file
```

## Paths and prerequisites

All paths live in **`config.sh`** (sourced by every shell script) or in the
matching environment variables (read by the Python scripts).  Override any
path on the command line:

```bash
export BENCH_BASE=/data/my-benchmarks/01_beone
export OUT_CHEWCALL=$BENCH_BASE/results/cc_my_run
./01_run_chewcall_BeONE.sh shcds_cons
```

Tools required on `PATH`:
- `chewcall` release binary (built with `cargo build --release` in the repo root)
- Conda environments `chewbbaca_354` and `chewbbaca_3310`
- `python3` with `numpy` and `pandas` for the analysis scripts
- `parasail` shared library at `$PARASAIL_LIB`

## Mapping: paper artefact → script

| Artefact (label / caption) | Section | Reproduce with |
|---|---|---|
| **Table 1** (`tab:runtime`) — shared-CDS runtime | Results §Performance | `python 04_tables_runtime_concordance.py --mode shcds` |
| **Table 2** (`tab:concordance`) — shared-CDS CRC32 concordance | Results §Performance | `python 04_tables_runtime_concordance.py --mode shcds` |
| **Table 3** (`tab:runtime_e2e`) — end-to-end runtime | Results §Performance | `python 04_tables_runtime_concordance.py --mode e2e` |
| **Table 4** (`tab:concordance_e2e`) — end-to-end CRC32 | Results §Performance | `python 04_tables_runtime_concordance.py --mode e2e` |
| **Figure 2** (`fig:discordance_anatomy`) — discordance heat-map | Results §Analysis of differences | `python 06_figure2_discordance_anatomy.py` |
| **Table 5** (`tab:cgmlst_split`) — cgMLST/wgMLST actionable split | Results §Analysis of differences | `python 09_table_cgmlst_split.py` |
| **Table 7** (`tab:clustering`) — cgMLST pairwise clustering | Results §Downstream concordance | `python 05_table_clustering.py` |
| **Table 8** (`tab:outbreak`) — FDA outbreak reconstruction | Results §Downstream concordance | manual; inputs are produced by `03_run_FDA_outbreak.sh`, downstream MST / single-linkage clustering uses SciPy `minimum_spanning_tree` and `linkage`/`fcluster` (Implementation §iii) |
| **Suppl Section S2** — pyrodigal equivalence | Supplementary | `python 08_supp_S2_pyrodigal_equivalence.py --genome ...` |
| **Suppl Section S2** — concordance with v3.3.10 | Supplementary | derived from `04_tables_runtime_concordance.py` (vs 3.3.10 columns) |
| **Suppl Table S3** (`tab-S3`) — classification transitions | Supplementary | `python 07_supp_S3_classification_transitions.py` (S3 block) |
| **Suppl Table S4** (`tab-S4`) — actionable per-organism | Supplementary | `python 07_supp_S3_classification_transitions.py` (S4 block) |
| **Table** (`tab:filter_safety`) — minimizer pre-filter coverage | Results §Filter safety | `make remedy` then `make ext-table` (scripts 10 + 15) |
| **Suppl Table** (`tab:ablation`) — minimizer ablation | Supplementary | `make supp-ablation` (`benchmark_scripts/ablation_minimizer.sh`) |

## Algorithmic extensions (scripts 10–15)

These reproduce the §"Filter safety, determinism, and hardware portability"
results. Paths come from `config_extensions.sh` (real BeONE layout by default;
override any variable in the environment). Run individually or via the matching
Makefile target.

```
10_schema_filter_safety.sh   make remedy       wcr audit + constructive_remedy, 4 organisms -> filter_safety_summary.tsv
11_pareto_sweep.sh           make pareto       (k,w,tau,kappa) Pareto sweep, 4 organisms -> pareto_<org>.tsv
12_determinism.sh            make determinism  Lm hash vs lexicographic minimizer ordering (cell diff)
13_gpu_vs_cpu.sh             make gpu          Lm GPU vs CPU concordance (cell diff)
15_table_filter_safety.py    make ext-table    renders Table tab:filter_safety from the summary
verify_convergence.py        (used by 12/13)   cell-by-cell CRC32 comparison of two profiles
```

All four extension analyses write under `output/extensions/`; `make extensions`
runs the lot and emits the LaTeX table.

## End-to-end reproduction (full benchmark)

```bash
# 1. Configure paths (override as needed)
source config.sh

# 2. Run chewcall on the 8 BeONE datasets (shcds + e2e × 3 reps each)
./01_run_chewcall_BeONE.sh all

# 3. Run reference chewBBACA versions
./02_run_chewbbaca_BeONE.sh 354  all
./02_run_chewbbaca_BeONE.sh 3310 all

# 4. Run on FDA outbreak data
./03_run_FDA_outbreak.sh

# 5. Generate every paper table/figure
python 04_tables_runtime_concordance.py --mode shcds   > /tmp/tables_1_2.tex
python 04_tables_runtime_concordance.py --mode e2e     > /tmp/tables_3_4.tex
python 06_figure2_discordance_anatomy.py               > /tmp/figure2.tex
python 05_table_clustering.py                          > /tmp/table7.tex
python 07_supp_S3_classification_transitions.py        > /tmp/tables_S3_S4.tex
```

The emitted LaTeX matches the `\begin{table}` / `\begin{figure}` blocks in
`paper/results.tex`, `paper/supplementary.tex` and
`paper/figures/fig_discordance_anatomy.tex` byte-for-byte modulo whitespace
when the underlying inputs are unchanged.

## Source-of-truth data layout (default paths)

```
$BENCH_BASE = <DATA_ROOT>

  schemas/{lm_cgmlst, se_wgmlst, ec_wgmlst, cj_wgmlst}/
  data/{lm,se,ec,cj}/genomes/   ← consortium genome FASTA
  data/{lm,se,ec,cj}/cds/       ← consortium pre-computed CDS (pyrodigal)
  results/cc_fixed2/
    {shcds,e2e}_{cons,pub}_{lm,se,ec,cj}_rep{1,2,3}/    ← chewcall output
    fda_{lm,se,ec,cj}/                                  ← chewcall on FDA
  results/c354/                                         ← chewBBACA v3.5.4 reference
  results/c3_3310_e2e/                                  ← chewBBACA v3.3.10 reference
```

Public assemblies (BeONE) live under
`<DATA_ROOT>/data/{lm,se,ec,cj}_public/`.

Pyrodigal pre-computed CDS for the public datasets live under
`<DATA_ROOT>/{lm,se,ec,cj}_public/cds_precomputed/`.

cgMLST locus name lists live at
`<DATA_ROOT>/{se,ec,cj}_cgmlst_loci.txt`
(Lm schemas are cgMLST-only by construction).
