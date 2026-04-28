#!/bin/bash
# 01_run_chewcall_BeONE.sh
#
# Runs chewcall on the 8 BeONE benchmark datasets (4 organisms x 2 collections,
# consortium + public), in two configurations:
#   - shcds: shared pre-computed CDS (pyrodigal)            -> Tables 1, 2
#   - e2e:   end-to-end with built-in prodigal-rs           -> Tables 3, 4
# Three replicates per dataset/configuration, used for runtime medians.
#
# Output: $OUT_CHEWCALL/{shcds,e2e}_{cons,pub}_{lm,se,ec,cj}_rep{1,2,3}/
# Plus per-mode timing files: timing_{shcds,e2e}_{cons,pub}.txt
#
# Usage:
#   ./01_run_chewcall_BeONE.sh shcds_cons     # only shared-CDS, consortium
#   ./01_run_chewcall_BeONE.sh shcds_pub
#   ./01_run_chewcall_BeONE.sh e2e_cons
#   ./01_run_chewcall_BeONE.sh e2e_pub
#   ./01_run_chewcall_BeONE.sh all            # all four
#
# Override paths in config.sh or via environment variables.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"
export LD_LIBRARY_PATH="$PARASAIL_LIB:${LD_LIBRARY_PATH:-}"

JOB="${1:-all}"
mkdir -p "$OUT_CHEWCALL"

declare -A SCHEMA=([lm]="$SCHEMA_LM" [se]="$SCHEMA_SE" [ec]="$SCHEMA_EC" [cj]="$SCHEMA_CJ")
declare -A G_CONS=([lm]="$GENOMES_LM_CONS" [se]="$GENOMES_SE_CONS" [ec]="$GENOMES_EC_CONS" [cj]="$GENOMES_CJ_CONS")
declare -A G_PUB=([lm]="$GENOMES_LM_PUB" [se]="$GENOMES_SE_PUB" [ec]="$GENOMES_EC_PUB" [cj]="$GENOMES_CJ_PUB")
declare -A C_CONS=([lm]="$CDS_LM_CONS" [se]="$CDS_SE_CONS" [ec]="$CDS_EC_CONS" [cj]="$CDS_CJ_CONS")
declare -A C_PUB=([lm]="$CDS_LM_PUB" [se]="$CDS_SE_PUB" [ec]="$CDS_EC_PUB" [cj]="$CDS_CJ_PUB")

run_one() {
    local label="$1" genomes="$2" schema_src="$3" outdir="$4" cds="${5:-}"
    if [ -f "$outdir/results_alleles_hashed.tsv" ]; then
        echo "SKIP: $label (already done)"
        return
    fi
    echo "=== START: $label $(date) ==="
    local schema_tmp="/tmp/schema_repro_${label}_$$"
    rm -rf "$schema_tmp"
    cp -r "$schema_src" "$schema_tmp"
    mkdir -p "$outdir"
    local cds_flag=""
    [ -n "$cds" ] && cds_flag="--cds-input $cds"
    local START=$(date +%s.%N)
    "$CHEWCALL_BIN" -i "$genomes" -g "$schema_tmp" -o "$outdir" --cpu "$CPU" $cds_flag \
        > "$outdir/run.log" 2>&1
    local END=$(date +%s.%N)
    local ELAPSED=$(echo "$END - $START" | bc)
    echo "$label $ELAPSED" >> "$OUT_CHEWCALL/timing_${MODE_TAG}.txt"
    echo "=== DONE: $label ${ELAPSED}s ==="
    rm -rf "$schema_tmp"
}

run_block() {
    local mode="$1"  # shcds | e2e
    local ds="$2"    # cons | pub
    MODE_TAG="${mode}_${ds}"
    declare -n GENOMES=$([ "$ds" = "cons" ] && echo G_CONS || echo G_PUB)
    declare -n CDSDIR=$([ "$ds" = "cons" ] && echo C_CONS || echo C_PUB)
    for org in lm se ec cj; do
        for rep in $(seq 1 "$REPS"); do
            local name="${mode}_${ds}_${org}_rep${rep}"
            local cds=""
            [ "$mode" = "shcds" ] && cds="${CDSDIR[$org]}"
            run_one "$name" "${GENOMES[$org]}" "${SCHEMA[$org]}" \
                    "$OUT_CHEWCALL/$name" "$cds"
        done
    done
}

case "$JOB" in
    shcds_cons) run_block shcds cons ;;
    shcds_pub)  run_block shcds pub  ;;
    e2e_cons)   run_block e2e   cons ;;
    e2e_pub)    run_block e2e   pub  ;;
    all)
        run_block shcds cons
        run_block shcds pub
        run_block e2e   cons
        run_block e2e   pub
        ;;
    *)
        echo "Unknown job: $JOB. Use shcds_cons | shcds_pub | e2e_cons | e2e_pub | all"
        exit 1
        ;;
esac
echo "JOB $JOB COMPLETE $(date)"
