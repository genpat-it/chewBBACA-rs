#!/bin/bash
# Four-parameter (k, w, tau, kappa) Pareto sweep of the minimizer pre-filter on
# all 4 BeONE organisms. Writes one TSV per organism and prints whether the
# chewcall default (k=5, w=5, tau=0.20, kappa=30) lies on the Pareto frontier.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config_extensions.sh"

OUT="$EXT_OUT/pareto"
mkdir -p "$OUT"

for org in $ORGANISMS; do
  schema="$(audit_schema_for "$org")"
  if [[ ! -d "$schema/short" ]]; then echo "[$org] SKIP (no schema at $schema)"; continue; fi
  echo "[$org] Pareto sweep (k=$PARETO_K_VALUES w=$PARETO_W_VALUES tau=$PARETO_TAU_VALUES kappa=$PARETO_KAPPA_VALUES) ..."
  LD_LIBRARY_PATH="$PARASAIL_LIB" "$PARETO_BIN" \
    -g "$schema" \
    --k-values "$PARETO_K_VALUES" --w-values "$PARETO_W_VALUES" \
    --tau-values "$PARETO_TAU_VALUES" --kappa-values "$PARETO_KAPPA_VALUES" \
    --cpu "$CPU" --exclude-inferred -o "$OUT/pareto_${org}.tsv" 2> "$OUT/pareto_${org}.log"
  grep -E "chewcall default|Pareto frontier|on Pareto" "$OUT/pareto_${org}.log" | sed "s/^/[$org] /" || true
done

echo "=== Pareto TSVs in $OUT/ ==="
ls -1 "$OUT"/pareto_*.tsv
