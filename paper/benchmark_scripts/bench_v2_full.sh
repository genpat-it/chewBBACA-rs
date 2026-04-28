#!/bin/bash
set -euo pipefail

###############################################################################
# Benchmark 08: EFSA regulatory comparison
# chewcall vs chewBBACA v2.8.5 (EFSA-recommended) vs v3.5.3 (original + fixed)
#
# Key design:
#   - v2.8.5 does NOT support --cds or --no-inferred
#   - v2.8.5 runs prodigal internally and modifies schema in-place (adds novel alleles)
#   - Therefore: FRESH SCHEMA COPY before EVERY run (critical!)
#   - Two sub-benchmarks:
#     (a) CDS-shared: chewcall + c3_original + c3_fixed use pre-computed CDS
#         (same as bench_01, isolates alignment engine)
#     (b) End-to-end: ALL tools use genome input + internal prodigal
#         (fair comparison since v2.8.5 cannot use --cds)
#   - Concordance is compared on CRC32-hashed profiles
#   - Clustering is compared at EFSA thresholds (7 AD Lm, 5 AD Se, 10 AD Ec, 7 AD Cj)
#
# 4 organisms (BeONE consortium): Lm, Se, Ec, Cj
# 4 tools × 3 reps × 4 organisms = 48 runs (end-to-end)
# 3 tools × 3 reps × 4 organisms = 36 runs (CDS-shared, no v2.8.5)
#
# Usage:
#   ./bench_08_efsa.sh              # run all organisms
#   ./bench_08_efsa.sh lm           # run only Listeria
#   ./bench_08_efsa.sh lm se        # run Listeria and Salmonella
###############################################################################

BENCH_DIR="/mnt/disk2/a.deruvo/chewcall_benchmarks/08_efsa"
CHEWCALL="/home/IZSNT/a.deruvo/chewbbacca-rs/target/release/chewcall"
PARASAIL_LIB="/home/IZSNT/a.deruvo/parasail/build"
CPU=8
REPS=3
TIMING_LOG="$BENCH_DIR/timing.tsv"
MASTER_LOG="$BENCH_DIR/benchmark.log"

# Conda environments
CONDA_V2="chewbbaca_v2"          # chewBBACA 2.8.5
CONDA_V3="chewbbaca_original"    # chewBBACA 3.5.3

# chewBBACA v3 allele_call.py (for toggling sort bug)
ALLELE_CALL="/home/IZSNT/a.deruvo/miniconda3/envs/${CONDA_V3}/lib/python3.11/site-packages/CHEWBBACA/AlleleCall/allele_call.py"

# === Data ===
BEONE="/mnt/disk2/a.deruvo/beone_benchmarks/data"

# === Schemas (pristine copies — NEVER point tools directly at these) ===
declare -A SCHEMA_SRC GENOME_DIR CDS_DIR EFSA_THRESHOLD
SCHEMA_SRC[lm]="$BEONE/lm/schema"
SCHEMA_SRC[se]="$BEONE/se/schema/Salmonella_enterica_wgMLST"
SCHEMA_SRC[ec]="$BEONE/ec/schema/Escherichia_coli_wgMLST"
SCHEMA_SRC[cj]="$BEONE/cj/schema/Campylobacter_jejuni_wgMLST"

GENOME_DIR[lm]="$BEONE/lm/genomes"
GENOME_DIR[se]="$BEONE/se/genomes"
GENOME_DIR[ec]="$BEONE/ec/genomes"
GENOME_DIR[cj]="$BEONE/cj/genomes"

CDS_DIR[lm]="$BEONE/lm/cds"
CDS_DIR[se]="$BEONE/se/cds"
CDS_DIR[ec]="$BEONE/ec/cds"
CDS_DIR[cj]="$BEONE/cj/cds"

# EFSA-recommended clustering thresholds (allelic differences)
EFSA_THRESHOLD[lm]=7
EFSA_THRESHOLD[se]=5
EFSA_THRESHOLD[ec]=10
EFSA_THRESHOLD[cj]=7

declare -A ORG_NAME
ORG_NAME[lm]="Listeria monocytogenes"
ORG_NAME[se]="Salmonella enterica"
ORG_NAME[ec]="Escherichia coli"
ORG_NAME[cj]="Campylobacter jejuni"

###############################################################################
# Utilities
###############################################################################

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$MASTER_LOG"
}

init() {
    mkdir -p "$BENCH_DIR"
    if [ ! -f "$TIMING_LOG" ]; then
        printf "benchmark\ttool\torganism\trep\tmode\tseconds\ttimestamp\n" > "$TIMING_LOG"
    fi
    [ -f "${ALLELE_CALL}.orig" ] || cp "$ALLELE_CALL" "${ALLELE_CALL}.orig"
}

# Toggle sort bug in v3: "original" = buggy (x[5]), "fixed" = correct (x[6])
set_sort_bug() {
    local mode="$1"
    if [ "$mode" = "original" ]; then
        sed -i 's/lambda x: int(x\[6\])/lambda x: int(x[5])/' "$ALLELE_CALL"
    else
        sed -i 's/lambda x: int(x\[5\])/lambda x: int(x[6])/' "$ALLELE_CALL"
    fi
}

has_results() {
    local dir="$1"
    [ -f "$dir/results_alleles.tsv" ] && return 0
    find "$dir" -maxdepth 2 -name "results_alleles.tsv" 2>/dev/null | grep -q . && return 0
    return 1
}

# Copy chewBBACA results from timestamped subdir to main dir
flatten_results() {
    local dir="$1"
    if [ ! -f "$dir/results_alleles.tsv" ]; then
        local found
        found=$(find "$dir" -maxdepth 2 -name "results_alleles.tsv" 2>/dev/null | head -1)
        if [ -n "$found" ]; then
            local subdir
            subdir=$(dirname "$found")
            for f in results_alleles.tsv results_alleles_hashed.tsv results_statistics.tsv; do
                [ -f "$subdir/$f" ] && ln -sf "$subdir/$f" "$dir/$f"
            done
        fi
    fi
}

# Create a fresh schema copy — returns path to the copy
# CRITICAL: every run must start from a pristine schema without novel alleles
fresh_schema() {
    local schema_src="$1"
    local tmpdir
    tmpdir=$(mktemp -d /tmp/schema_bench08_XXXX)
    cp -r "$schema_src" "$tmpdir/schema"
    echo "$tmpdir/schema"
}

###############################################################################
# Tool runners
###############################################################################

# --- chewcall ---
# mode: "cds" (pre-computed CDS) or "genome" (built-in prodigal-rs)
run_chewcall() {
    local org="$1" rep="$2" mode="$3"
    local outdir="$BENCH_DIR/results/${mode}/chewcall/${org}_rep${rep}"

    if has_results "$outdir"; then
        log "SKIP chewcall/$org rep$rep ($mode) — results exist"
        return
    fi

    local schema
    schema=$(fresh_schema "${SCHEMA_SRC[$org]}")
    mkdir -p "$outdir"

    local cmd=("$CHEWCALL"
        -i "${GENOME_DIR[$org]}"
        -g "$schema"
        -o "$outdir"
        --cpu $CPU)

    if [ "$mode" = "cds" ]; then
        cmd+=(--cds-input "${CDS_DIR[$org]}")
    fi

    log "START chewcall/$org rep$rep ($mode)"
    local start end elapsed
    start=$(date +%s.%N)
    LD_LIBRARY_PATH=$PARASAIL_LIB "${cmd[@]}" > "$outdir/run.log" 2>&1 || true
    end=$(date +%s.%N)
    elapsed=$(echo "$end - $start" | bc)

    printf "08_efsa\tchewcall\t%s\t%d\t%s\t%.1f\t%s\n" \
        "$org" "$rep" "$mode" "$elapsed" "$(date -Iseconds)" >> "$TIMING_LOG"
    log "DONE  chewcall/$org rep$rep ($mode) → ${elapsed}s"
    rm -rf "$(dirname "$schema")"
}

# --- chewBBACA v3.5.3 (original or fixed) ---
# mode: "cds" or "genome"
run_c3() {
    local tool="$1" org="$2" rep="$3" mode="$4"
    local outdir="$BENCH_DIR/results/${mode}/${tool}/${org}_rep${rep}"

    if has_results "$outdir"; then
        log "SKIP $tool/$org rep$rep ($mode) — results exist"
        return
    fi

    # Toggle sort bug
    if [ "$tool" = "c3_original" ]; then
        set_sort_bug "original"
    else
        set_sort_bug "fixed"
    fi

    local schema
    schema=$(fresh_schema "${SCHEMA_SRC[$org]}")
    mkdir -p "$outdir"

    local input_dir cds_flag=""
    if [ "$mode" = "cds" ]; then
        input_dir="${CDS_DIR[$org]}"
        cds_flag="--cds"
    else
        input_dir="${GENOME_DIR[$org]}"
    fi

    log "START $tool/$org rep$rep ($mode)"
    local start end elapsed
    start=$(date +%s.%N)
    conda run -n $CONDA_V3 chewie AlleleCall \
        -i "$input_dir" \
        $cds_flag \
        -g "$schema" \
        -o "$outdir" \
        --cpu $CPU \
        --no-inferred > "$outdir/run.log" 2>&1 || true
    end=$(date +%s.%N)
    elapsed=$(echo "$end - $start" | bc)

    printf "08_efsa\t%s\t%s\t%d\t%s\t%.1f\t%s\n" \
        "$tool" "$org" "$rep" "$mode" "$elapsed" "$(date -Iseconds)" >> "$TIMING_LOG"
    log "DONE  $tool/$org rep$rep ($mode) → ${elapsed}s"
    flatten_results "$outdir"
    rm -rf "$(dirname "$schema")"
}

# --- chewBBACA v2.8.5 ---
# ALWAYS end-to-end (no --cds support, no --no-inferred)
# Schema is modified in-place by v2.8.5 → fresh copy is CRITICAL
#
# v2.8.5 does NOT produce results_alleles_hashed.tsv, so we generate it
# by reading the modified schema (which now contains the novel alleles
# that v2 appended) and hashing allele DNA sequences with CRC32.
run_v2() {
    local org="$1" rep="$2"
    local outdir="$BENCH_DIR/results/genome/v2/${org}_rep${rep}"

    if has_results "$outdir"; then
        log "SKIP v2/$org rep$rep — results exist"
        return
    fi

    local schema
    schema=$(fresh_schema "${SCHEMA_SRC[$org]}")
    mkdir -p "$outdir"

    # v2.8.5 CLI: chewBBACA.py AlleleCall -i <dir> -g <schema> -o <dir> --cpu N
    # No --cds, no --no-inferred — it always runs prodigal and modifies schema
    log "START v2/$org rep$rep (genome, schema: fresh copy)"
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
    log "DONE  v2/$org rep$rep → ${elapsed}s"
    flatten_results "$outdir"

    # Generate CRC32-hashed profiles (v2 doesn't produce them)
    # Use the MODIFIED schema (v2 appended novel alleles to it)
    local raw_tsv="$outdir/results_alleles.tsv"
    if [ -f "$raw_tsv" ]; then
        log "  Generating CRC32-hashed profiles for v2/$org rep$rep..."
        python3 "$(dirname "$0")/hash_v2_profiles.py" \
            "$raw_tsv" "$schema" \
            "$outdir/results_alleles_hashed.tsv" >> "$outdir/run.log" 2>&1 || \
            log "  WARNING: CRC32 hashing failed for v2/$org rep$rep"
    fi

    rm -rf "$(dirname "$schema")"
}

###############################################################################
# Per-organism benchmark
###############################################################################

run_organism() {
    local org="$1"
    local n_genomes n_loci
    n_genomes=$(ls "${GENOME_DIR[$org]}" | grep -c -E '\.(fasta|fa|fna)$')
    n_loci=$(ls "${SCHEMA_SRC[$org]}" | grep -c '\.fasta$')

    log ""
    log "================================================================"
    log "  ${ORG_NAME[$org]} ($org)"
    log "  $n_genomes genomes, $n_loci loci, EFSA threshold: ${EFSA_THRESHOLD[$org]} AD"
    log "  Schema: ${SCHEMA_SRC[$org]}"
    log "================================================================"

    for rep in $(seq 1 $REPS); do
        # --- Sub-benchmark (a): CDS-shared (v2.8.5 excluded, cannot use --cds) ---
        run_chewcall  "$org" "$rep" "cds"
        run_c3 "c3_original" "$org" "$rep" "cds"
        run_c3 "c3_fixed"    "$org" "$rep" "cds"

        # --- Sub-benchmark (b): End-to-end (all tools, genome input) ---
        run_chewcall  "$org" "$rep" "genome"
        run_c3 "c3_original" "$org" "$rep" "genome"
        run_c3 "c3_fixed"    "$org" "$rep" "genome"
        run_v2        "$org" "$rep"
    done

    # Restore sort bug to fixed
    set_sort_bug "fixed"
}

###############################################################################
# Pre-flight checks
###############################################################################

preflight() {
    log "================================================================"
    log "  Benchmark 08: EFSA regulatory comparison"
    log "  chewcall vs v2.8.5 vs v3.5.3 (original + fixed)"
    log "  $CPU CPUs, $REPS repetitions per tool"
    log "================================================================"

    [ -x "$CHEWCALL" ] || { echo "ERROR: chewcall not found: $CHEWCALL" >&2; exit 1; }
    [ -f "$PARASAIL_LIB/libparasail.so" ] || { echo "ERROR: parasail not found" >&2; exit 1; }

    local v2_ver v3_ver
    v2_ver=$(conda run -n $CONDA_V2 pip show chewbbaca 2>/dev/null | grep Version | awk '{print $2}')
    v3_ver=$(conda run -n $CONDA_V3 pip show chewbbaca 2>/dev/null | grep Version | awk '{print $2}')
    [ "$v2_ver" = "2.8.5" ] || { echo "ERROR: expected chewBBACA 2.8.5 in $CONDA_V2, got $v2_ver" >&2; exit 1; }
    [ "$v3_ver" = "3.5.3" ] || echo "WARNING: expected chewBBACA 3.5.3 in $CONDA_V3, got $v3_ver" >&2

    log "chewcall:   $(cd /home/IZSNT/a.deruvo/chewbbacca-rs && git log -1 --oneline 2>/dev/null)"
    log "v2.8.5:     $v2_ver (env: $CONDA_V2)"
    log "  BLAST:    $(conda run -n $CONDA_V2 blastp -version 2>&1 | head -1)"
    log "  prodigal: $(conda run -n $CONDA_V2 prodigal -v 2>&1 | head -1)"
    log "v3.5.3:     $v3_ver (env: $CONDA_V3)"
    log "  BLAST:    $(conda run -n $CONDA_V3 blastp -version 2>&1 | head -1)"
    log "CPU:        $CPU / $(nproc) available"
    log "Host:       $(hostname)"
    log "Date:       $(date -Iseconds)"
    log ""

    # Verify schemas are pristine (no novel alleles from previous runs)
    for org in lm se ec cj; do
        local short_dir="${SCHEMA_SRC[$org]}/short"
        if [ -d "$short_dir" ]; then
            local n_short
            n_short=$(ls "$short_dir" | grep -c '_short\.fasta$' || true)
            log "Schema $org: $n_short loci in short/"
        fi
    done
}

###############################################################################
# Main
###############################################################################

main() {
    init
    preflight

    local organisms=("${@:-lm se ec cj}")
    if [ $# -eq 0 ]; then
        organisms=(lm se ec cj)
    fi

    for org in "${organisms[@]}"; do
        if [[ ! -v "SCHEMA_SRC[$org]" ]]; then
            echo "ERROR: unknown organism '$org' (use: lm, se, ec, cj)" >&2
            continue
        fi
        run_organism "$org"
    done

    log ""
    log "===== BENCHMARK 08 COMPLETE ====="
    log "Timing: $TIMING_LOG"
    log "Results: $BENCH_DIR/results/"
    log ""
    log "Next steps:"
    log "  1. Compare CRC32 hashed profiles (pairwise: cc↔v2, cc↔v3, v2↔v3)"
    log "  2. Compute cgMLST clustering at EFSA thresholds"
    log "  3. Run: python check_concordance_08.py $BENCH_DIR/results/"
}

main "$@"
