#!/bin/bash
# Ablation study: minimizer parameters k and w
# Usage: ./ablation_minimizer.sh <chewcall_binary> <genomes_dir> <schema_dir> <cds_dir> <output_dir> [threads]
#
# Example:
#   ./ablation_minimizer.sh ./target/release/chewcall \
#       data/genomes data/schema data/cds results/ablation 8

set -euo pipefail

if [ $# -lt 5 ]; then
    echo "Usage: $0 <chewcall> <genomes> <schema> <cds> <output_dir> [threads]"
    exit 1
fi

CHEWCALL="$1"
GENOMES="$2"
SCHEMA="$3"
CDS="$4"
OUTBASE="$5"
THREADS="${6:-8}"

mkdir -p "$OUTBASE"
echo "k,w,runtime_s,cells_total,cells_matching,concordance_pct" > "$OUTBASE/results.csv"

REF_OUTPUT="$OUTBASE/k5_w5"

# Reference run (k=5, w=5)
if [ ! -d "$REF_OUTPUT" ]; then
    echo "=== Running reference k=5 w=5 ==="
    START=$(date +%s%N)
    "$CHEWCALL" -i "$GENOMES" -g "$SCHEMA" -o "$REF_OUTPUT" \
        --cds-input "$CDS" --cpu "$THREADS" \
        --minimizer-k 5 --minimizer-w 5 2>&1 | tail -3
    END=$(date +%s%N)
    RUNTIME=$(echo "scale=1; ($END - $START) / 1000000000" | bc)
    echo "Reference done in ${RUNTIME}s"
fi

REF_PROFILE="$REF_OUTPUT/results_alleles.tsv"

for k in 3 4 5 6 7; do
    for w in 3 5 7; do
        OUTDIR="$OUTBASE/k${k}_w${w}"

        if [ -d "$OUTDIR" ] && [ "$k" -eq 5 ] && [ "$w" -eq 5 ]; then
            echo "=== Skipping k=$k w=$w (reference) ==="
            RUNTIME="ref"
            PROFILE="$OUTDIR/results_alleles.tsv"
        else
            echo "=== Running k=$k w=$w ==="
            rm -rf "$OUTDIR"
            mkdir -p "$OUTDIR"

            START=$(date +%s%N)
            "$CHEWCALL" -i "$GENOMES" -g "$SCHEMA" -o "$OUTDIR" \
                --cds-input "$CDS" --cpu "$THREADS" \
                --minimizer-k "$k" --minimizer-w "$w" 2>&1 | tail -3
            END=$(date +%s%N)
            RUNTIME=$(echo "scale=1; ($END - $START) / 1000000000" | bc)
            PROFILE="$OUTDIR/results_alleles.tsv"
            echo "  Done in ${RUNTIME}s"
        fi

        # Compare profiles against reference
        if [ -f "$PROFILE" ] && [ -f "$REF_PROFILE" ]; then
            python3 -c "
import csv, sys
with open(sys.argv[1]) as f1, open(sys.argv[2]) as f2:
    r1 = list(csv.reader(f1, delimiter='\t'))
    r2 = list(csv.reader(f2, delimiter='\t'))
    total = match = 0
    for row1, row2 in zip(r1[1:], r2[1:]):
        for c1, c2 in zip(row1[1:], row2[1:]):
            total += 1
            if c1 == c2: match += 1
    pct = 100.0 * match / total if total > 0 else 0
    print(f'{total},{match},{pct:.6f}')
" "$REF_PROFILE" "$PROFILE" > /tmp/ablation_cmp.txt
            STATS=$(cat /tmp/ablation_cmp.txt)
            TOTAL=$(echo "$STATS" | cut -d, -f1)
            MATCH=$(echo "$STATS" | cut -d, -f2)
            PCT=$(echo "$STATS" | cut -d, -f3)
            echo "  Concordance: $PCT% ($MATCH/$TOTAL)"
            echo "$k,$w,$RUNTIME,$TOTAL,$MATCH,$PCT" >> "$OUTBASE/results.csv"
        fi
    done
done

echo ""
echo "=== ABLATION RESULTS ==="
column -t -s, "$OUTBASE/results.csv"
