#!/bin/bash
# 03_run_FDA_outbreak.sh
#
# Runs both tools on the four FDA outbreak datasets used by Table 7
# (tab:outbreak): chewcall + chewBBACA v3.5.4 + chewBBACA v3.3.10.
# Each dataset is processed once (no replicates) since these runs are scored
# for clustering accuracy rather than runtime.
#
# Output: $OUT_CHEWCALL/fda_{lm,se,ec,cj}/    (chewcall)
#         $OUT_C354/fda_{lm,se,ec,cj}_rep1/   (chewBBACA v3.5.4)
#         $OUT_C3310/fda_{lm,se,ec,cj}_rep1/  (chewBBACA v3.3.10)

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"
export LD_LIBRARY_PATH="$PARASAIL_LIB:${LD_LIBRARY_PATH:-}"

declare -A SCHEMA=([lm]="$SCHEMA_LM" [se]="$SCHEMA_SE" [ec]="$SCHEMA_EC" [cj]="$SCHEMA_CJ")
declare -A FDA_CDS=(
    [lm]="$FDA_BASE/lm/cds"
    [se]="$FDA_BASE/se/cds"
    [ec]="$FDA_BASE/ec/cds"
    [cj]="$FDA_BASE/cj/cds"
)

mkdir -p "$OUT_CHEWCALL" "$OUT_C354" "$OUT_C3310"

for org in lm se ec cj; do
    cds="${FDA_CDS[$org]}"

    # chewcall (shared-CDS) ----------------------------------------
    out="$OUT_CHEWCALL/fda_${org}"
    if [ ! -f "$out/results_alleles_hashed.tsv" ]; then
        schema_tmp="/tmp/schema_fda_cc_${org}_$$"
        rm -rf "$schema_tmp"; cp -r "${SCHEMA[$org]}" "$schema_tmp"
        mkdir -p "$out"
        echo "=== chewcall fda_$org $(date) ==="
        "$CHEWCALL_BIN" -i "$cds" -g "$schema_tmp" -o "$out" \
            --cpu "$CPU" --cds-input "$cds" > "$out/run.log" 2>&1
        rm -rf "$schema_tmp"
    fi

    # chewBBACA 3.5.4 ----------------------------------------------
    for ver in 354 3310; do
        case "$ver" in
            354)  ENV="$CONDA_ENV_354"; OUT_BASE="$OUT_C354" ;;
            3310) ENV="$CONDA_ENV_3310"; OUT_BASE="$OUT_C3310" ;;
        esac
        out="$OUT_BASE/fda_${org}_rep1"
        if [ -f "$out/results_alleles_hashed.tsv" ] || \
           compgen -G "$out/results_*/results_alleles_hashed.tsv" > /dev/null; then
            continue
        fi
        schema_tmp="/tmp/schema_fda_cb${ver}_${org}_$$"
        rm -rf "$schema_tmp"; cp -r "${SCHEMA[$org]}" "$schema_tmp"
        mkdir -p "$out"
        echo "=== chewBBACA $ver fda_$org $(date) ==="
        conda run -n "$ENV" chewie AlleleCall \
            -i "$cds" -g "$schema_tmp" -o "$out" \
            --cpu "$CPU" --no-inferred --hash-profile crc32 \
            > "$out/run.log" 2>&1
        rm -rf "$schema_tmp"
    done
done
echo "FDA outbreak runs COMPLETE $(date)"
