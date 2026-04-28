#!/bin/bash
# 02_run_chewbbaca_BeONE.sh
#
# Reference runs of chewBBACA on the same 8 BeONE datasets, against which
# chewcall concordance is measured.  Two versions are supported:
#   v3.5.4  -> $OUT_C354
#   v3.3.10 -> $OUT_C3310
# Each runs in shared-CDS mode (Tables 1/2) and end-to-end mode (Tables 3/4).
# Always uses --no-inferred --hash-profile crc32 to match the paper protocol.
#
# Usage:
#   ./02_run_chewbbaca_BeONE.sh 354  shcds_cons
#   ./02_run_chewbbaca_BeONE.sh 3310 e2e_pub
#   ./02_run_chewbbaca_BeONE.sh 354  all
#
# Requires conda environments named in config.sh ($CONDA_ENV_354, $CONDA_ENV_3310).

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"

VERSION="${1:-354}"
JOB="${2:-all}"

case "$VERSION" in
    354)  CONDA_ENV="$CONDA_ENV_354";  OUT_BASE="$OUT_C354"  ;;
    3310) CONDA_ENV="$CONDA_ENV_3310"; OUT_BASE="$OUT_C3310" ;;
    *) echo "Unknown version: $VERSION (expected 354 or 3310)"; exit 1 ;;
esac
mkdir -p "$OUT_BASE"

declare -A SCHEMA=([lm]="$SCHEMA_LM" [se]="$SCHEMA_SE" [ec]="$SCHEMA_EC" [cj]="$SCHEMA_CJ")
declare -A G_CONS=([lm]="$GENOMES_LM_CONS" [se]="$GENOMES_SE_CONS" [ec]="$GENOMES_EC_CONS" [cj]="$GENOMES_CJ_CONS")
declare -A G_PUB=([lm]="$GENOMES_LM_PUB" [se]="$GENOMES_SE_PUB" [ec]="$GENOMES_EC_PUB" [cj]="$GENOMES_CJ_PUB")
declare -A C_CONS=([lm]="$CDS_LM_CONS" [se]="$CDS_SE_CONS" [ec]="$CDS_EC_CONS" [cj]="$CDS_CJ_CONS")
declare -A C_PUB=([lm]="$CDS_LM_PUB" [se]="$CDS_SE_PUB" [ec]="$CDS_EC_PUB" [cj]="$CDS_CJ_PUB")

run_one() {
    local label="$1" input="$2" schema_src="$3" outdir="$4"
    if [ -f "$outdir/results_alleles_hashed.tsv" ] || compgen -G "$outdir/results_*/results_alleles_hashed.tsv" > /dev/null; then
        echo "SKIP: $label (already done)"
        return
    fi
    echo "=== START: $label $(date) ==="
    local schema_tmp="/tmp/schema_cb${VERSION}_${label}_$$"
    rm -rf "$schema_tmp"
    cp -r "$schema_src" "$schema_tmp"
    mkdir -p "$outdir"
    local START=$(date +%s.%N)
    conda run -n "$CONDA_ENV" chewie AlleleCall \
        -i "$input" -g "$schema_tmp" -o "$outdir" \
        --cpu "$CPU" --no-inferred --hash-profile crc32 \
        > "$outdir/run.log" 2>&1
    local END=$(date +%s.%N)
    local ELAPSED=$(echo "$END - $START" | bc)
    echo "$label $ELAPSED" >> "$OUT_BASE/timing_${MODE_TAG}.txt"
    echo "=== DONE: $label ${ELAPSED}s ==="
    rm -rf "$schema_tmp"
}

run_block() {
    local mode="$1" ds="$2"
    MODE_TAG="${mode}_${ds}"
    declare -n GENOMES=$([ "$ds" = "cons" ] && echo G_CONS || echo G_PUB)
    declare -n CDSDIR=$([ "$ds" = "cons" ] && echo C_CONS || echo C_PUB)
    for org in lm se ec cj; do
        for rep in $(seq 1 "$REPS"); do
            local name="${mode}_${ds}_${org}_rep${rep}"
            # shared-CDS mode passes the CDS directory to AlleleCall as input
            # (chewBBACA accepts a directory of CDS FASTA files).
            local input="${GENOMES[$org]}"
            [ "$mode" = "shcds" ] && input="${CDSDIR[$org]}"
            run_one "$name" "$input" "${SCHEMA[$org]}" "$OUT_BASE/$name"
        done
    done
}

case "$JOB" in
    shcds_cons|shcds_pub|e2e_cons|e2e_pub)
        IFS=_ read -r mode ds <<< "$JOB"; run_block "$mode" "$ds" ;;
    all)
        run_block shcds cons; run_block shcds pub
        run_block e2e   cons; run_block e2e   pub
        ;;
    *)
        echo "Unknown job: $JOB"; exit 1 ;;
esac
echo "JOB $JOB on chewBBACA $VERSION COMPLETE $(date)"
