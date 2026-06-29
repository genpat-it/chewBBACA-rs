#!/bin/bash
# GPU/CPU concordance: run chewcall on Lm on the CPU and on the GPU (CUDA), then
# compare the allelic profiles cell-by-cell. A converged result (0 differing
# cells) shows the GPU Smith-Waterman backend reproduces the CPU calls exactly.
# Requires a CUDA-capable GPU and the cudarc-linked binary.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config_extensions.sh"

OUT="$EXT_OUT/gpu"
mkdir -p "$OUT"
schema="$AUDIT_SCHEMA_lm"; genomes="$GENOMES_lm"; cds="$CDS_lm"
for d in "$schema/short" "$genomes" "$cds"; do
  [[ -e "$d" ]] || { echo "missing input: $d"; exit 2; }
done

oc="$OUT/lm_cpu"; og="$OUT/lm_gpu"; rm -rf "$oc" "$og"
echo "[cpu] chewcall (CPU) ..."
LD_LIBRARY_PATH="$PARASAIL_LIB" "$CHEWCALL_BIN" -i "$genomes" -g "$schema" -o "$oc" \
  --cpu "$CPU" --cds-input "$cds" > "$OUT/lm_cpu.log" 2>&1
echo "[gpu] chewcall --gpu ..."
CUDA_HOME="$CUDA_HOME" LD_LIBRARY_PATH="$PARASAIL_LIB:$CUDA_LIB" "$CHEWCALL_BIN" \
  -i "$genomes" -g "$schema" -o "$og" --cpu "$CPU" --cds-input "$cds" --gpu > "$OUT/lm_gpu.log" 2>&1

echo "=== GPU vs CPU ==="
python3 "$HERE/verify_convergence.py" \
  "$og/results_alleles_hashed.tsv" "$oc/results_alleles_hashed.tsv" --label lm-gpu-vs-cpu
