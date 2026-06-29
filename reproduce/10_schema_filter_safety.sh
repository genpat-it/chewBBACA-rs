#!/bin/bash
# Filter-safety audit + constructive remedy on all 4 BeONE organisms.
# Produces, per organism, the worst-case minimizer overlap (wcr) before/after
# promoting witness alleles to representatives, and an enriched schema that
# satisfies wcr >= tau by construction. Emits a combined summary TSV.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config_extensions.sh"

OUT="$EXT_OUT/filter_safety"
mkdir -p "$OUT"
SUMMARY="$OUT/filter_safety_summary.tsv"
printf "organism\tloci\tloci_flagged\talleles_promoted\twcr_before\twcr_after\n" > "$SUMMARY"

for org in $ORGANISMS; do
  schema="$(audit_schema_for "$org")"
  if [[ ! -d "$schema/short" ]]; then echo "[$org] SKIP (no schema at $schema)"; continue; fi
  rep_out="$OUT/remedy_$org"
  rm -rf "$rep_out"
  echo "[$org] constructive_remedy on $schema ..."
  LD_LIBRARY_PATH="$PARASAIL_LIB" "$CONSTRUCTIVE_REMEDY_BIN" \
    -g "$schema" -o "$rep_out" --k "$AUDIT_K" --w "$AUDIT_W" --threshold "$AUDIT_TAU" \
    --cpu "$CPU" --exclude-inferred 2> "$OUT/remedy_${org}.log"
  rep="$rep_out/constructive_remedy_report.tsv"
  awk -F'\t' -v org="$org" '
    NR>1 { tot++;
           if ($5>0) { flag++; prom+=$5 }
           if (NR==2 || $3<minb) minb=$3;
           if (NR==2 || $4<mina) mina=$4 }
    END  { printf "%s\t%d\t%d\t%d\t%.4f\t%.4f\n", org, tot, flag, prom, minb, mina }' \
    "$rep" >> "$SUMMARY"
done

echo "=== filter-safety summary (=> $SUMMARY) ==="
column -t "$SUMMARY"
