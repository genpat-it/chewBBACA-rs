#!/bin/bash
# Centralised paths used by every reproduction script.
# Override any variable in your environment to adapt to your setup.

# ---- Tools --------------------------------------------------------
: "${CHEWCALL_BIN:=$HOME/chewbbacca-rs/target/release/chewcall}"
: "${PARASAIL_LIB:=$HOME/parasail/build}"

# Conda environments providing chewBBACA at the two reference versions.
: "${CONDA_ENV_354:=chewbbaca_354}"
: "${CONDA_ENV_3310:=chewbbaca_3310}"

# ---- Benchmark data root ------------------------------------------
: "${BENCH_BASE:=/mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone}"

# Schemas (read-only, copied to a temp dir per run because chewBBACA mutates them)
: "${SCHEMA_LM:=$BENCH_BASE/schemas/lm_cgmlst}"
: "${SCHEMA_SE:=$BENCH_BASE/schemas/se_wgmlst}"
: "${SCHEMA_EC:=$BENCH_BASE/schemas/ec_wgmlst}"
: "${SCHEMA_CJ:=$BENCH_BASE/schemas/cj_wgmlst}"

# Consortium genome FASTA directories
: "${GENOMES_LM_CONS:=$BENCH_BASE/data/lm/genomes}"
: "${GENOMES_SE_CONS:=$BENCH_BASE/data/se/genomes}"
: "${GENOMES_EC_CONS:=$BENCH_BASE/data/ec/genomes}"
: "${GENOMES_CJ_CONS:=$BENCH_BASE/data/cj/genomes}"

# Public genome FASTA directories
: "${GENOMES_LM_PUB:=/mnt/disk2/a.deruvo/beone_benchmarks/data/lm_public/Lm_public/assemblies}"
: "${GENOMES_SE_PUB:=/mnt/disk2/a.deruvo/beone_benchmarks/data/se_public/Se_public/assemblies}"
: "${GENOMES_EC_PUB:=/mnt/disk2/a.deruvo/beone_benchmarks/data/ec_public/Ec_public/assemblies}"
: "${GENOMES_CJ_PUB:=/mnt/disk2/a.deruvo/beone_benchmarks/data/cj_public/Cj_public/assemblies}"

# Pre-computed CDS (pyrodigal) for shared-CDS comparisons
: "${CDS_LM_CONS:=$BENCH_BASE/data/lm/cds}"
: "${CDS_SE_CONS:=$BENCH_BASE/data/se/cds}"
: "${CDS_EC_CONS:=$BENCH_BASE/data/ec/cds}"
: "${CDS_CJ_CONS:=$BENCH_BASE/data/cj/cds}"
: "${CDS_LM_PUB:=/mnt/disk2/a.deruvo/chew_results/lm_public/cds_precomputed}"
: "${CDS_SE_PUB:=/mnt/disk2/a.deruvo/chew_results/se_public/cds_precomputed}"
: "${CDS_EC_PUB:=/mnt/disk2/a.deruvo/chew_results/ec_public/cds_precomputed}"
: "${CDS_CJ_PUB:=/mnt/disk2/a.deruvo/chew_results/cj_public/cds_precomputed}"

# FDA outbreak datasets (Table 7)
: "${FDA_BASE:=/mnt/disk2/a.deruvo/chewcall_paper/06_outbreak_validation}"

# cgMLST locus-name lists (for filtering wgMLST schemas to cgMLST distance)
: "${CGMLST_LOCI_SE:=/mnt/disk2/a.deruvo/chew_results/se_cgmlst_loci.txt}"
: "${CGMLST_LOCI_EC:=/mnt/disk2/a.deruvo/chew_results/ec_cgmlst_loci.txt}"
: "${CGMLST_LOCI_CJ:=/mnt/disk2/a.deruvo/chew_results/cj_cgmlst_loci.txt}"
# Lm schemas are cgMLST-only by construction.

# ---- Output root --------------------------------------------------
: "${OUT_ROOT:=$BENCH_BASE/results}"
: "${OUT_CHEWCALL:=$OUT_ROOT/cc_fixed2}"
: "${OUT_C354:=$OUT_ROOT/c354}"
: "${OUT_C3310:=$OUT_ROOT/c3_3310_e2e}"

# Number of CPU threads used by every tool
: "${CPU:=8}"
# Replicates for runtime medians
: "${REPS:=3}"

export CHEWCALL_BIN PARASAIL_LIB CONDA_ENV_354 CONDA_ENV_3310 \
       BENCH_BASE SCHEMA_LM SCHEMA_SE SCHEMA_EC SCHEMA_CJ \
       GENOMES_LM_CONS GENOMES_SE_CONS GENOMES_EC_CONS GENOMES_CJ_CONS \
       GENOMES_LM_PUB GENOMES_SE_PUB GENOMES_EC_PUB GENOMES_CJ_PUB \
       CDS_LM_CONS CDS_SE_CONS CDS_EC_CONS CDS_CJ_CONS \
       CDS_LM_PUB CDS_SE_PUB CDS_EC_PUB CDS_CJ_PUB \
       FDA_BASE CGMLST_LOCI_SE CGMLST_LOCI_EC CGMLST_LOCI_CJ \
       OUT_ROOT OUT_CHEWCALL OUT_C354 OUT_C3310 CPU REPS
