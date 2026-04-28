#!/bin/bash
set -euo pipefail

###############################################################################
# Benchmark 08: chewBBACA v2.8.5 (EFSA-recommended) only
#
# Results for chewcall, c3_original, c3_fixed already exist in bench_01.
# This script runs ONLY v2.8.5 on the same BeONE datasets.
#
# Key constraints of v2.8.5:
#   - No --cds flag (always runs prodigal internally)
#   - No --no-inferred (always modifies schema in-place, adds novel alleles)
#   → FRESH SCHEMA COPY before EVERY run (critical!)
#   - Does NOT produce results_alleles_hashed.tsv
#   → Post-process with hash_v2_profiles.py using the modified schema
#
# Usage:
#   ./bench_08_v2_only.sh              # all organisms, 3 reps
#   ./bench_08_v2_only.sh lm           # only Listeria
#   ./bench_08_v2_only.sh lm se        # Listeria + Salmonella
###############################################################################

BENCH_DIR="/mnt/disk2/a.deruvo/chewcall_benchmarks/08_efsa"
CONDA_V2="chewbbaca_v2"
CPU=8
REPS=1  # Reduced from 3 for Ec/Cj — v2.8.5 too slow to run 3 reps
TIMING_LOG="$BENCH_DIR/timing.tsv"
MASTER_LOG="$BENCH_DIR/benchmark.log"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# BeONE data
BEONE="/mnt/disk2/a.deruvo/beone_benchmarks/data"

declare -A SCHEMA_SRC GENOME_DIR ORG_NAME
SCHEMA_SRC[lm]="$BEONE/lm/schema"
SCHEMA_SRC[se]="$BEONE/se/schema/Salmonella_enterica_wgMLST"
SCHEMA_SRC[ec]="$BEONE/ec/schema/Escherichia_coli_wgMLST"
SCHEMA_SRC[cj]="$BEONE/cj/schema/Campylobacter_jejuni_wgMLST"

GENOME_DIR[lm]="$BEONE/lm/genomes"
GENOME_DIR[se]="$BEONE/se/genomes"
GENOME_DIR[ec]="$BEONE/ec/genomes"
GENOME_DIR[cj]="$BEONE/cj/genomes"

ORG_NAME[lm]="Listeria monocytogenes"
ORG_NAME[se]="Salmonella enterica"
ORG_NAME[ec]="Escherichia coli"
ORG_NAME[cj]="Campylobacter jejuni"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$MASTER_LOG"
}

has_results() {
    local dir="$1"
    [ -f "$dir/results_alleles.tsv" ] && return 0
    find "$dir" -maxdepth 2 -name "results_alleles.tsv" 2>/dev/null | grep -q . && return 0
    return 1
}

flatten_results() {
    local dir="$1"
    if [ ! -f "$dir/results_alleles.tsv" ]; then
        local found
        found=$(find "$dir" -maxdepth 2 -name "results_alleles.tsv" 2>/dev/null | head -1)
        if [ -n "$found" ]; then
            local subdir
            subdir=$(dirname "$found")
            for f in results_alleles.tsv results_statistics.tsv results_contigsInfo.tsv; do
                [ -f "$subdir/$f" ] && ln -sf "$subdir/$f" "$dir/$f"
            done
        fi
    fi
}

run_v2() {
    local org="$1" rep="$2"
    local outdir="$BENCH_DIR/results/v2/${org}_rep${rep}"

    if has_results "$outdir"; then
        log "SKIP v2/$org rep$rep — results exist"
        return
    fi

    # CRITICAL: fresh schema copy (v2.8.5 modifies schema in-place)
    local schema_tmp
    schema_tmp=$(mktemp -d /tmp/schema_v2_XXXX)
    cp -r "${SCHEMA_SRC[$org]}" "$schema_tmp/schema"
    local schema="$schema_tmp/schema"

    mkdir -p "$outdir"

    local n_genomes
    n_genomes=$(ls "${GENOME_DIR[$org]}"/*.fasta 2>/dev/null | wc -l)

    log "START v2/${org} rep${rep} — ${n_genomes} genomes, schema: fresh copy"
    local start end elapsed
    start=$(date +%s.%N)
    conda run -n $CONDA_V2 chewBBACA.py AlleleCall \
        -i "${GENOME_DIR[$org]}" \
        -g "$schema" \
        -o "$outdir" \
        --cpu $CPU > "$outdir/run.log" 2>&1 || true
    end=$(date +%s.%N)
    elapsed=$(echo "$end - $start" | bc)

    printf "08_efsa\tv2\t%s\t%d\tgenome\t%.1f\t%s\n" \
        "$org" "$rep" "$elapsed" "$(date -Iseconds)" >> "$TIMING_LOG"
    log "DONE  v2/${org} rep${rep} → ${elapsed}s"

    flatten_results "$outdir"

    # Generate CRC32-hashed profiles using the MODIFIED schema
    # (v2.8.5 appended novel alleles to the FASTA files during the run)
    local raw_tsv="$outdir/results_alleles.tsv"
    if [ -f "$raw_tsv" ]; then
        log "  Generating CRC32-hashed profiles..."
        python3 "$SCRIPT_DIR/hash_v2_profiles.py" \
            "$raw_tsv" "$schema" \
            "$outdir/results_alleles_hashed.tsv" >> "$outdir/run.log" 2>&1 || \
            log "  WARNING: CRC32 hashing failed"
    fi

    rm -rf "$schema_tmp"
}

###############################################################################
# Main
###############################################################################

mkdir -p "$BENCH_DIR"
if [ ! -f "$TIMING_LOG" ]; then
    printf "benchmark\ttool\torganism\trep\tmode\tseconds\ttimestamp\n" > "$TIMING_LOG"
fi

# Preflight
v2_ver=$(conda run -n $CONDA_V2 pip show chewbbaca 2>/dev/null | grep Version | awk '{print $2}')
[ "$v2_ver" = "2.8.5" ] || { echo "ERROR: expected v2.8.5, got $v2_ver" >&2; exit 1; }

log "================================================================"
log "  Benchmark 08: chewBBACA v2.8.5 ONLY"
log "  v2.8.5 ($v2_ver), BLAST $(conda run -n $CONDA_V2 blastp -version 2>&1 | head -1)"
log "  $CPU CPUs, $REPS reps"
log "  Existing results: bench_01 (chewcall, c3_original, c3_fixed)"
log "================================================================"

organisms=("${@:-lm se ec cj}")
if [ $# -eq 0 ]; then
    organisms=(lm se ec cj)
fi

for org in "${organisms[@]}"; do
    n_genomes=$(ls "${GENOME_DIR[$org]}"/*.fasta 2>/dev/null | wc -l)
    n_loci=$(ls "${SCHEMA_SRC[$org]}"/*.fasta 2>/dev/null | wc -l)
    log ""
    log "=== ${ORG_NAME[$org]} ($org): $n_genomes genomes, $n_loci loci ==="

    for rep in $(seq 1 $REPS); do
        run_v2 "$org" "$rep"
    done
done

log ""
log "===== v2.8.5 RUNS COMPLETE ====="
log "Timing: $TIMING_LOG"
log ""
log "Compare with existing bench_01 results:"
log "  chewcall:     /mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results/chewcall/"
log "  c3_original:  /mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results/c3_original/"
log "  c3_fixed:     /mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results/c3_fixed/"
log "  v2.8.5:       $BENCH_DIR/results/v2/"
