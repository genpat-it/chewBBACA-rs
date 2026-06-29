#!/bin/bash
# Central paths for the reproduction pipeline. All data paths default to
# locations RELATIVE to this repository / archive, so the pipeline runs without
# editing after a fresh checkout or after extracting the Zenodo archive.
#
# To point at data stored elsewhere (e.g. a local benchmark mount), either:
#   - export BENCH_BASE / PAPER_BASE (and any *_PUB path) in your shell, or
#   - create reproduce/config.local.sh with your overrides (auto-sourced, and
#     git-ignored).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"          # paper_gm/ (repo) or archive root

# Optional local overrides (not committed / not shipped)
[ -f "$HERE/config.local.sh" ] && source "$HERE/config.local.sh"

# ---- Tools --------------------------------------------------------
# Use the repo-built binary if present, otherwise expect chewcall on PATH.
_repo_bin="$HERE/../../target/release/chewcall"
: "${CHEWCALL_BIN:=$([ -x "$_repo_bin" ] && echo "$_repo_bin" || echo chewcall)}"
: "${PARASAIL_LIB:=/usr/local/lib}"
: "${CONDA_ENV_354:=chewbbaca_354}"
: "${CONDA_ENV_3310:=chewbbaca_3310}"

# ---- Benchmark data roots (relative to ROOT by default) ------------
: "${BENCH_BASE:=$ROOT}"
: "${PAPER_BASE:=$ROOT}"

# Schemas (pristine snapshots)
: "${SCHEMA_LM:=$BENCH_BASE/schemas/lm_cgmlst}"
: "${SCHEMA_SE:=$BENCH_BASE/schemas/se_wgmlst}"
: "${SCHEMA_EC:=$BENCH_BASE/schemas/ec_wgmlst}"
: "${SCHEMA_CJ:=$BENCH_BASE/schemas/cj_wgmlst}"

# Consortium genomes + pre-computed CDS
: "${GENOMES_LM_CONS:=$BENCH_BASE/data/lm/genomes}"
: "${GENOMES_SE_CONS:=$BENCH_BASE/data/se/genomes}"
: "${GENOMES_EC_CONS:=$BENCH_BASE/data/ec/genomes}"
: "${GENOMES_CJ_CONS:=$BENCH_BASE/data/cj/genomes}"
: "${CDS_LM_CONS:=$BENCH_BASE/data/lm/cds}"
: "${CDS_SE_CONS:=$BENCH_BASE/data/se/cds}"
: "${CDS_EC_CONS:=$BENCH_BASE/data/ec/cds}"
: "${CDS_CJ_CONS:=$BENCH_BASE/data/cj/cds}"

# Public assemblies (CDS not shipped; regenerate with pyrodigal — see README)
: "${GENOMES_LM_PUB:=$BENCH_BASE/data/lm_public}"
: "${GENOMES_SE_PUB:=$BENCH_BASE/data/se_public}"
: "${GENOMES_EC_PUB:=$BENCH_BASE/data/ec_public}"
: "${GENOMES_CJ_PUB:=$BENCH_BASE/data/cj_public}"
: "${CDS_LM_PUB:=$BENCH_BASE/data/lm_public/cds}"
: "${CDS_SE_PUB:=$BENCH_BASE/data/se_public/cds}"
: "${CDS_EC_PUB:=$BENCH_BASE/data/ec_public/cds}"
: "${CDS_CJ_PUB:=$BENCH_BASE/data/cj_public/cds}"

# cgMLST locus lists + FDA outbreak data
: "${CGMLST_LOCI_SE:=$PAPER_BASE/cgmlst_loci/se_cgmlst_loci.txt}"
: "${CGMLST_LOCI_EC:=$PAPER_BASE/cgmlst_loci/ec_cgmlst_loci.txt}"
: "${CGMLST_LOCI_CJ:=$PAPER_BASE/cgmlst_loci/cj_cgmlst_loci.txt}"
: "${FDA_BASE:=$PAPER_BASE/fda}"

# ---- Output -------------------------------------------------------
: "${OUT_ROOT:=$ROOT/results}"
: "${OUT_CHEWCALL:=$OUT_ROOT/cc}"
: "${OUT_C354:=$OUT_ROOT/c354}"
: "${OUT_C3310:=$OUT_ROOT/c3310}"

: "${CPU:=8}"
: "${REPS:=3}"

export CHEWCALL_BIN PARASAIL_LIB CONDA_ENV_354 CONDA_ENV_3310 \
       BENCH_BASE PAPER_BASE SCHEMA_LM SCHEMA_SE SCHEMA_EC SCHEMA_CJ \
       GENOMES_LM_CONS GENOMES_SE_CONS GENOMES_EC_CONS GENOMES_CJ_CONS \
       GENOMES_LM_PUB GENOMES_SE_PUB GENOMES_EC_PUB GENOMES_CJ_PUB \
       CDS_LM_CONS CDS_SE_CONS CDS_EC_CONS CDS_CJ_CONS \
       CDS_LM_PUB CDS_SE_PUB CDS_EC_PUB CDS_CJ_PUB \
       CGMLST_LOCI_SE CGMLST_LOCI_EC CGMLST_LOCI_CJ FDA_BASE \
       OUT_ROOT OUT_CHEWCALL OUT_C354 OUT_C3310 CPU REPS
