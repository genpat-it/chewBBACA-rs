#!/bin/bash
# Paths for the algorithmic-extension scripts (10-15). Defaults are relative to
# this repository / archive; override via the environment or reproduce/config.local.sh.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

[ -f "$HERE/config.local.sh" ] && source "$HERE/config.local.sh"

_repo_root="$(cd "$HERE/../.." && pwd)"   # chewbbacca-rs when run from the repo
: "${CHEWCALL_BIN:=$([ -x "$_repo_root/target/release/chewcall" ] && echo "$_repo_root/target/release/chewcall" || echo chewcall)}"
: "${CONSTRUCTIVE_REMEDY_BIN:=$([ -x "$_repo_root/target/release/constructive_remedy" ] && echo "$_repo_root/target/release/constructive_remedy" || echo constructive_remedy)}"
: "${SCHEMA_AUDIT_BIN:=$([ -x "$_repo_root/target/release/schema_audit" ] && echo "$_repo_root/target/release/schema_audit" || echo schema_audit)}"
: "${PARETO_BIN:=$([ -x "$_repo_root/target/release/schema_audit_pareto" ] && echo "$_repo_root/target/release/schema_audit_pareto" || echo schema_audit_pareto)}"
: "${PARASAIL_LIB:=/usr/local/lib}"
: "${CUDA_LIB:=/usr/local/cuda/lib64}"
: "${CUDA_HOME:=/usr/local/cuda}"

# Per-organism schemas (relative to ROOT)
: "${AUDIT_SCHEMA_lm:=$ROOT/schemas/lm_cgmlst}"
: "${AUDIT_SCHEMA_se:=$ROOT/schemas/se_wgmlst}"
: "${AUDIT_SCHEMA_ec:=$ROOT/schemas/ec_wgmlst}"
: "${AUDIT_SCHEMA_cj:=$ROOT/schemas/cj_wgmlst}"

: "${GENOMES_lm:=$ROOT/data/lm/genomes}"
: "${CDS_lm:=$ROOT/data/lm/cds}"

: "${AUDIT_K:=5}"
: "${AUDIT_W:=5}"
: "${AUDIT_TAU:=0.20}"
: "${PARETO_K_VALUES:=4,5,6}"
: "${PARETO_W_VALUES:=3,5,7}"
: "${PARETO_TAU_VALUES:=0.15,0.20,0.25}"
: "${PARETO_KAPPA_VALUES:=5,10,30,0}"
: "${CPU:=8}"
: "${EXT_OUT:=$ROOT/results/extensions}"
: "${ORGANISMS:=lm se ec cj}"

export CHEWCALL_BIN CONSTRUCTIVE_REMEDY_BIN SCHEMA_AUDIT_BIN PARETO_BIN \
       PARASAIL_LIB CUDA_LIB CUDA_HOME \
       AUDIT_SCHEMA_lm AUDIT_SCHEMA_se AUDIT_SCHEMA_ec AUDIT_SCHEMA_cj \
       GENOMES_lm CDS_lm AUDIT_K AUDIT_W AUDIT_TAU \
       PARETO_K_VALUES PARETO_W_VALUES PARETO_TAU_VALUES PARETO_KAPPA_VALUES \
       CPU EXT_OUT ORGANISMS

audit_schema_for() { local v="AUDIT_SCHEMA_$1"; echo "${!v}"; }
