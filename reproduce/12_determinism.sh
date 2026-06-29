#!/bin/bash
# Minimizer-ordering determinism: run chewcall on Lm with FNV-1a hash ordering
# and with lexicographic (chewBBACA-style) ordering, then compare the allelic
# profiles cell-by-cell. A converged result (0 differing cells) shows the call
# set is insensitive to the minimizer ordering.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config_extensions.sh"

OUT="$EXT_OUT/determinism"
mkdir -p "$OUT"
schema="$AUDIT_SCHEMA_lm"; genomes="$GENOMES_lm"; cds="$CDS_lm"
for d in "$schema/short" "$genomes" "$cds"; do
  [[ -e "$d" ]] || { echo "missing input: $d"; exit 2; }
done

oh="$OUT/lm_hash"; ol="$OUT/lm_lex"; rm -rf "$oh" "$ol"
echo "[hash] chewcall --minimizer-order hash ..."
LD_LIBRARY_PATH="$PARASAIL_LIB" "$CHEWCALL_BIN" -i "$genomes" -g "$schema" -o "$oh" \
  --cpu "$CPU" --cds-input "$cds" --minimizer-order hash > "$OUT/lm_hash.log" 2>&1
echo "[lex]  chewcall --minimizer-order lexicographic ..."
LD_LIBRARY_PATH="$PARASAIL_LIB" "$CHEWCALL_BIN" -i "$genomes" -g "$schema" -o "$ol" \
  --cpu "$CPU" --cds-input "$cds" --minimizer-order lexicographic > "$OUT/lm_lex.log" 2>&1

echo "=== hash vs lexicographic ==="
python3 "$HERE/verify_convergence.py" \
  "$oh/results_alleles_hashed.tsv" "$ol/results_alleles_hashed.tsv" --label lm-hash-vs-lex
