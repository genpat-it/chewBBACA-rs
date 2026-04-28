#!/bin/bash
set -euo pipefail

###############################################################################
# Master benchmark runner — chewcall paper
# 3-WAY: chewcall vs c3_original (with sort bug) vs c3_fixed
# 3 repetitions, report median, fresh schema copy before EVERY run, 8 CPUs
#
# Usage:
#   ./run_all.sh              # run all benchmarks
#   ./run_all.sh 01           # run only benchmark 01
#   ./run_all.sh 01 04 05     # run benchmarks 01, 04, 05
###############################################################################

BENCH_DIR="/mnt/disk2/a.deruvo/chewcall_benchmarks"
CHEWCALL="/home/IZSNT/a.deruvo/chewbbacca-rs/target/release/chewcall"
PARASAIL_LIB="/home/IZSNT/a.deruvo/parasail/build"
CONDA_ENV="chewbbaca_original"
CPU=8
REPS=3
TIMING_LOG="$BENCH_DIR/timing.tsv"
MASTER_LOG="$BENCH_DIR/benchmark.log"

# chewBBACA allele_call.py (for toggling sort bug)
ALLELE_CALL="/home/IZSNT/a.deruvo/miniconda3/envs/chewbbaca_original/lib/python3.11/site-packages/CHEWBBACA/AlleleCall/allele_call.py"

# === Data roots ===
BEONE="/mnt/disk2/a.deruvo/beone_benchmarks/data"
SYNTHETIC="/mnt/disk2/a.deruvo/chew_results"
OUTBREAK="/mnt/disk2/a.deruvo/chewcall_paper/06_outbreak_validation"

# === Schemas ===
SCHEMA_LM="$BEONE/lm/schema"
SCHEMA_SE="$BEONE/se/schema/Salmonella_enterica_wgMLST"
SCHEMA_EC="$BEONE/ec/schema/Escherichia_coli_wgMLST"
SCHEMA_CJ="$BEONE/cj/schema/Campylobacter_jejuni_wgMLST"

###############################################################################
# Utilities
###############################################################################

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$MASTER_LOG"
}

init_timing_log() {
    if [ ! -f "$TIMING_LOG" ]; then
        printf "benchmark\ttool\torganism\tdataset\trep\tseconds\ttimestamp\n" > "$TIMING_LOG"
    fi
}

# Toggle sort bug: "original" = buggy (x[5]), "fixed" = correct (x[6])
set_sort_bug() {
    local mode="$1"
    if [ "$mode" = "original" ]; then
        sed -i 's/lambda x: int(x\[6\])/lambda x: int(x[5])/' "$ALLELE_CALL"
    else
        sed -i 's/lambda x: int(x\[5\])/lambda x: int(x[6])/' "$ALLELE_CALL"
    fi
    log "  sort_bug → $mode ($(grep -o 'lambda x: int(x\[[56]\])' "$ALLELE_CALL"))"
}

# Check if benchmark run already completed
has_results() {
    local dir="$1"
    # chewcall: results directly in dir
    [ -f "$dir/results_alleles.tsv" ] && return 0
    # chewBBACA: results in timestamped subdir
    find "$dir" -maxdepth 2 -name "results_alleles.tsv" 2>/dev/null | grep -q . && return 0
    return 1
}

# Copy chewBBACA results from timestamped subdir to main dir for easy access
flatten_c3_results() {
    local dir="$1"
    # If results_alleles.tsv not in root, find it in subdir and symlink
    if [ ! -f "$dir/results_alleles.tsv" ]; then
        local found
        found=$(find "$dir" -maxdepth 2 -name "results_alleles.tsv" 2>/dev/null | head -1)
        if [ -n "$found" ]; then
            local subdir
            subdir=$(dirname "$found")
            ln -sf "$subdir/results_alleles.tsv" "$dir/results_alleles.tsv"
            [ -f "$subdir/results_statistics.tsv" ] && ln -sf "$subdir/results_statistics.tsv" "$dir/results_statistics.tsv"
        fi
    fi
}

###############################################################################
# Core runners
###############################################################################

# run_chewcall <benchmark> <org> <dataset> <rep> <genome_dir> <cds_dir_or_empty> <schema_src> <outdir>
run_chewcall() {
    local benchmark="$1" org="$2" dataset="$3" rep="$4"
    local genome_dir="$5" cds_dir="$6" schema_src="$7" outdir="$8"

    if has_results "$outdir"; then
        log "SKIP chewcall $benchmark/$dataset rep$rep (results exist)"
        return
    fi

    local schema_copy
    schema_copy=$(mktemp -d /tmp/schema_cc_XXXX)
    cp -r "$schema_src" "$schema_copy/schema"
    mkdir -p "$outdir"

    local cds_flag=""
    if [ -n "$cds_dir" ]; then
        cds_flag="--cds-input $cds_dir"
    fi

    log "START chewcall $benchmark/$dataset rep$rep"
    local start end elapsed
    start=$(date +%s.%N)
    LD_LIBRARY_PATH=$PARASAIL_LIB $CHEWCALL \
        -i "$genome_dir" \
        $cds_flag \
        -g "$schema_copy/schema" \
        -o "$outdir" \
        --cpu $CPU > "$outdir/run.log" 2>&1 || true
    end=$(date +%s.%N)
    elapsed=$(echo "$end - $start" | bc)

    printf "%s\tchewcall\t%s\t%s\t%d\t%.1f\t%s\n" \
        "$benchmark" "$org" "$dataset" "$rep" "$elapsed" "$(date -Iseconds)" >> "$TIMING_LOG"
    log "DONE  chewcall $benchmark/$dataset rep$rep → ${elapsed}s"
    rm -rf "$schema_copy"
}

# run_c3 <tool> <benchmark> <org> <dataset> <rep> <input_dir> <is_cds:0|1> <schema_src> <outdir>
run_c3() {
    local tool="$1" benchmark="$2" org="$3" dataset="$4" rep="$5"
    local input_dir="$6" is_cds="$7" schema_src="$8" outdir="$9"

    if has_results "$outdir"; then
        log "SKIP $tool $benchmark/$dataset rep$rep (results exist)"
        return
    fi

    # Toggle sort bug
    if [ "$tool" = "c3_original" ]; then
        set_sort_bug "original"
    else
        set_sort_bug "fixed"
    fi

    local schema_copy
    schema_copy=$(mktemp -d /tmp/schema_c3_XXXX)
    cp -r "$schema_src" "$schema_copy/schema"
    mkdir -p "$outdir"

    local cds_flag=""
    if [ "$is_cds" = "1" ]; then
        cds_flag="--cds"
    fi

    log "START $tool $benchmark/$dataset rep$rep"
    local start end elapsed
    start=$(date +%s.%N)
    conda run -n $CONDA_ENV chewie AlleleCall \
        -i "$input_dir" \
        $cds_flag \
        -g "$schema_copy/schema" \
        -o "$outdir" \
        --cpu $CPU \
        --no-inferred > "$outdir/run.log" 2>&1 || true
    end=$(date +%s.%N)
    elapsed=$(echo "$end - $start" | bc)

    printf "%s\t%s\t%s\t%s\t%d\t%.1f\t%s\n" \
        "$benchmark" "$tool" "$org" "$dataset" "$rep" "$elapsed" "$(date -Iseconds)" >> "$TIMING_LOG"
    log "DONE  $tool $benchmark/$dataset rep$rep → ${elapsed}s"
    flatten_c3_results "$outdir"
    rm -rf "$schema_copy"
}

# run_3way_cds: both tools use pre-computed CDS
# <benchmark> <org> <dataset> <genome_dir> <cds_dir> <schema_src> <result_base>
run_3way_cds() {
    local benchmark="$1" org="$2" dataset="$3"
    local genome_dir="$4" cds_dir="$5" schema_src="$6" result_base="$7"

    for rep in $(seq 1 $REPS); do
        run_chewcall "$benchmark" "$org" "$dataset" "$rep" \
            "$genome_dir" "$cds_dir" "$schema_src" \
            "$result_base/chewcall/${dataset}_rep${rep}"

        # For c3, -i points to CDS dir, --cds flag is set
        run_c3 "c3_original" "$benchmark" "$org" "$dataset" "$rep" \
            "$cds_dir" "1" "$schema_src" \
            "$result_base/c3_original/${dataset}_rep${rep}"

        run_c3 "c3_fixed" "$benchmark" "$org" "$dataset" "$rep" \
            "$cds_dir" "1" "$schema_src" \
            "$result_base/c3_fixed/${dataset}_rep${rep}"
    done
}

# run_3way_genome: both tools run prodigal internally
# <benchmark> <org> <dataset> <genome_dir> <schema_src> <result_base>
run_3way_genome() {
    local benchmark="$1" org="$2" dataset="$3"
    local genome_dir="$4" schema_src="$5" result_base="$6"

    for rep in $(seq 1 $REPS); do
        run_chewcall "$benchmark" "$org" "$dataset" "$rep" \
            "$genome_dir" "" "$schema_src" \
            "$result_base/chewcall/${dataset}_rep${rep}"

        run_c3 "c3_original" "$benchmark" "$org" "$dataset" "$rep" \
            "$genome_dir" "0" "$schema_src" \
            "$result_base/c3_original/${dataset}_rep${rep}"

        run_c3 "c3_fixed" "$benchmark" "$org" "$dataset" "$rep" \
            "$genome_dir" "0" "$schema_src" \
            "$result_base/c3_fixed/${dataset}_rep${rep}"
    done
}

###############################################################################
# Pre-flight checks
###############################################################################

preflight() {
    log "=========================================="
    log "  chewcall paper — benchmark suite"
    log "  3-way × 3 reps × 8 CPUs"
    log "=========================================="

    [ -x "$CHEWCALL" ] || { echo "ERROR: chewcall not found" >&2; exit 1; }
    [ -f "$PARASAIL_LIB/libparasail.so" ] || { echo "ERROR: parasail not found" >&2; exit 1; }

    local cb_ver
    cb_ver=$(conda run -n $CONDA_ENV pip show chewbbaca 2>/dev/null | grep Version | awk '{print $2}')
    [ "$cb_ver" = "3.5.3" ] || echo "WARNING: expected chewBBACA 3.5.3, got $cb_ver" >&2

    # Backup allele_call.py if not already backed up
    [ -f "${ALLELE_CALL}.orig" ] || cp "$ALLELE_CALL" "${ALLELE_CALL}.orig"

    log "chewcall git: $(cd /home/IZSNT/a.deruvo/chewbbacca-rs && git log -1 --oneline 2>/dev/null)"
    log "chewBBACA:    $cb_ver"
    log "BLAST:        $(conda run -n $CONDA_ENV blastp -version 2>&1 | head -1)"
    log "CPU:          $CPU / $(nproc) available"
    log "Host:         $(hostname)"
    log "Date:         $(date -Iseconds)"
    log ""

    init_timing_log
}

###############################################################################
# Benchmark 01: BeONE
# 4 organisms × 3 tools × 3 reps = 36 runs (CDS pre-computed)
###############################################################################

bench_01() {
    log "===== BENCHMARK 01: BeONE (3-way, CDS pre-computed) ====="

    declare -A GENOMES CDS SCHEMA
    GENOMES[lm]="$BEONE/lm/genomes"; CDS[lm]="$BEONE/lm/cds"; SCHEMA[lm]="$SCHEMA_LM"
    GENOMES[se]="$BEONE/se/genomes"; CDS[se]="$BEONE/se/cds"; SCHEMA[se]="$SCHEMA_SE"
    GENOMES[ec]="$BEONE/ec/genomes"; CDS[ec]="$BEONE/ec/cds"; SCHEMA[ec]="$SCHEMA_EC"
    GENOMES[cj]="$BEONE/cj/genomes"; CDS[cj]="$BEONE/cj/cds"; SCHEMA[cj]="$SCHEMA_CJ"

    for org in lm se ec cj; do
        run_3way_cds "01_beone" "$org" "$org" \
            "${GENOMES[$org]}" "${CDS[$org]}" "${SCHEMA[$org]}" \
            "$BENCH_DIR/01_beone/results"
    done
}

###############################################################################
# Benchmark 02: Circular genomes (perfect assemblies)
# 4 organisms × 3 tools × 3 reps = 36 runs (CDS pre-computed)
###############################################################################

bench_02() {
    log "===== BENCHMARK 02: Circular genomes (3-way, CDS pre-computed) ====="
    local base="$BENCH_DIR/02_circular/results"
    local circ="$BENCH_DIR/02_circular"

    declare -A GENOMES CDS SCHEMA
    GENOMES[lm]="$circ/genomes/lm"; CDS[lm]="$circ/cds/lm"; SCHEMA[lm]="$SCHEMA_LM"
    GENOMES[se]="$circ/genomes/se"; CDS[se]="$circ/cds/se"; SCHEMA[se]="$SCHEMA_SE"
    GENOMES[ec]="$circ/genomes/ec"; CDS[ec]="$circ/cds/ec"; SCHEMA[ec]="$SCHEMA_EC"
    GENOMES[cj]="$circ/genomes/cj"; CDS[cj]="$circ/cds/cj"; SCHEMA[cj]="$SCHEMA_CJ"

    for org in lm se ec cj; do
        run_3way_cds "02_circular" "$org" "$org" \
            "${GENOMES[$org]}" "${CDS[$org]}" "${SCHEMA[$org]}" "$base"
    done
}

###############################################################################
# Benchmark 03: Coverage degradation
# 4 organisms × ~30 coverages×reps × 3 tools × 3 tool-reps
###############################################################################

bench_03() {
    log "===== BENCHMARK 03: Degradation (3-way, CDS pre-computed) ====="
    local base_data="$BENCH_DIR/03_degradation"
    local base_results="$BENCH_DIR/03_degradation/results"

    declare -A SCHEMA_MAP
    SCHEMA_MAP[lm]="$SCHEMA_LM"
    SCHEMA_MAP[se]="$SCHEMA_SE"
    SCHEMA_MAP[ec]="$SCHEMA_EC"
    SCHEMA_MAP[cj]="$SCHEMA_CJ"

    for org in lm se ec cj; do
        # Find coverage dirs that have actual data
        for covdir in $(ls "$base_data/$org/" 2>/dev/null | grep "x_rep" | sort -V); do
            local gdir="$base_data/$org/$covdir/genomes"
            local cdir="$base_data/$org/$covdir/cds"
            local dataset="${org}_${covdir}"

            # Skip if no genome or no CDS
            [ -z "$(ls -A "$gdir" 2>/dev/null)" ] && continue
            [ -z "$(ls -A "$cdir" 2>/dev/null)" ] && continue

            # Single assembly per covdir, but 3 tool-reps for timing
            for rep in $(seq 1 $REPS); do
                run_chewcall "03_degradation" "$org" "$dataset" "$rep" \
                    "$gdir" "$cdir" "${SCHEMA_MAP[$org]}" \
                    "$base_results/chewcall/${dataset}_rep${rep}"

                run_c3 "c3_original" "03_degradation" "$org" "$dataset" "$rep" \
                    "$cdir" "1" "${SCHEMA_MAP[$org]}" \
                    "$base_results/c3_original/${dataset}_rep${rep}"

                run_c3 "c3_fixed" "03_degradation" "$org" "$dataset" "$rep" \
                    "$cdir" "1" "${SCHEMA_MAP[$org]}" \
                    "$base_results/c3_fixed/${dataset}_rep${rep}"
            done
        done
    done
}

###############################################################################
# Benchmark 04: Synthetic ground truth
# 4 organisms × 3 tools × 3 reps = 36 runs (CDS synthetic, no genomes)
###############################################################################

bench_04() {
    log "===== BENCHMARK 04: Synthetic (3-way, CDS synthetic) ====="

    declare -A CDS SCHEMA
    CDS[lm]="$SYNTHETIC/synthetic_cds_lm/cds"; SCHEMA[lm]="$SCHEMA_LM"
    CDS[se]="$SYNTHETIC/synthetic_cds_se/cds"; SCHEMA[se]="$SCHEMA_SE"
    CDS[ec]="$SYNTHETIC/synthetic_cds_ec/cds"; SCHEMA[ec]="$SCHEMA_EC"
    CDS[cj]="$SYNTHETIC/synthetic_cds_cj/cds"; SCHEMA[cj]="$SCHEMA_CJ"

    for org in lm se ec cj; do
        # For synthetic: no genome assemblies, pass CDS dir as genome_dir too
        # (chewcall reads .fasta files from -i for contig lengths, not critical for CDS-only)
        run_3way_cds "04_synthetic" "$org" "$org" \
            "${CDS[$org]}" "${CDS[$org]}" "${SCHEMA[$org]}" \
            "$BENCH_DIR/04_synthetic/results"
    done
}

###############################################################################
# Benchmark 05: Outbreak reconstruction
# 4 organisms × 3 tools × 3 reps = 36 runs (CDS pre-computed)
###############################################################################

bench_05() {
    log "===== BENCHMARK 05: Outbreak (3-way, CDS pre-computed) ====="

    declare -A GENOMES CDS SCHEMA
    GENOMES[lm]="$OUTBREAK/lm/assemblies"; CDS[lm]="$OUTBREAK/lm/cds"; SCHEMA[lm]="$SCHEMA_LM"
    GENOMES[se]="$OUTBREAK/se/assemblies"; CDS[se]="$OUTBREAK/se/cds"; SCHEMA[se]="$SCHEMA_SE"
    GENOMES[ec]="$OUTBREAK/ec/assemblies"; CDS[ec]="$OUTBREAK/ec/cds"; SCHEMA[ec]="$SCHEMA_EC"
    GENOMES[cj]="$OUTBREAK/cj/assemblies"; CDS[cj]="$OUTBREAK/cj/cds"; SCHEMA[cj]="$SCHEMA_CJ"

    for org in lm se ec cj; do
        run_3way_cds "05_outbreak" "$org" "$org" \
            "${GENOMES[$org]}" "${CDS[$org]}" "${SCHEMA[$org]}" \
            "$BENCH_DIR/05_outbreak/results"
    done
}

###############################################################################
# Benchmark 06: c3 scalability (Mamede et al. datasets)
# 3 organisms × variable sizes × 3 tools × 3 reps (genome input, prodigal)
###############################################################################

bench_06() {
    log "===== BENCHMARK 06: c3 scalability (3-way, with prodigal) ====="

    local schema_zip_dir="$BENCH_DIR/06_c3_scalability/schemas"
    local genomes_dir="$BENCH_DIR/06_c3_scalability/genomes"
    local base="$BENCH_DIR/06_c3_scalability/results"

    declare -A ZIP INNER SIZES
    ZIP[lm]="$schema_zip_dir/lm_cl_c3_schema.zip"; INNER[lm]="lm_cl_c3_schema"
    ZIP[se]="$schema_zip_dir/se_cl_c3_schema.zip"; INNER[se]="se_cl_c3_schema"
    ZIP[sp]="$schema_zip_dir/sp_cl_c3_schema.zip"; INNER[sp]="sp_cl_c3_schema"
    SIZES[lm]="128 256 512 1024 2048 4096"
    SIZES[se]="128 512 1024 2048 4096"
    SIZES[sp]="128 512 1024 2048 4096"

    for org in lm se sp; do
        for size in ${SIZES[$org]}; do
            local gdir="$genomes_dir/$org/$size"
            if [ ! -d "$gdir" ] || [ -z "$(ls -A "$gdir" 2>/dev/null)" ]; then
                log "SKIP 06_c3/$org/$size — no genomes"
                continue
            fi
            local ngenomes
            ngenomes=$(ls "$gdir" | wc -l)
            log "06_c3/$org/$size: $ngenomes genomes"

            for rep in $(seq 1 $REPS); do
                local dataset="${org}_${size}"

                # --- chewcall ---
                local outdir="$base/chewcall/${dataset}_rep${rep}"
                if ! has_results "$outdir"; then
                    local sc
                    sc=$(mktemp -d /tmp/schema_c3_XXXX)
                    unzip -q "${ZIP[$org]}" -d "$sc"
                    mkdir -p "$outdir"
                    log "START chewcall 06_c3/$dataset rep$rep"
                    local start end elapsed
                    start=$(date +%s.%N)
                    LD_LIBRARY_PATH=$PARASAIL_LIB $CHEWCALL \
                        -i "$gdir" -g "$sc/${INNER[$org]}" -o "$outdir" \
                        --cpu $CPU > "$outdir/run.log" 2>&1 || true
                    end=$(date +%s.%N)
                    elapsed=$(echo "$end - $start" | bc)
                    printf "06_c3\tchewcall\t%s\t%s\t%d\t%.1f\t%s\n" \
                        "$org" "$dataset" "$rep" "$elapsed" "$(date -Iseconds)" >> "$TIMING_LOG"
                    log "DONE  chewcall 06_c3/$dataset rep$rep → ${elapsed}s"
                    rm -rf "$sc"
                fi

                # --- c3_original ---
                outdir="$base/c3_original/${dataset}_rep${rep}"
                if ! has_results "$outdir"; then
                    set_sort_bug "original"
                    local sc
                    sc=$(mktemp -d /tmp/schema_c3_XXXX)
                    unzip -q "${ZIP[$org]}" -d "$sc"
                    mkdir -p "$outdir"
                    log "START c3_original 06_c3/$dataset rep$rep"
                    start=$(date +%s.%N)
                    conda run -n $CONDA_ENV chewie AlleleCall \
                        -i "$gdir" -g "$sc/${INNER[$org]}" -o "$outdir" \
                        --cpu $CPU --no-inferred > "$outdir/run.log" 2>&1 || true
                    end=$(date +%s.%N)
                    elapsed=$(echo "$end - $start" | bc)
                    printf "06_c3\tc3_original\t%s\t%s\t%d\t%.1f\t%s\n" \
                        "$org" "$dataset" "$rep" "$elapsed" "$(date -Iseconds)" >> "$TIMING_LOG"
                    log "DONE  c3_original 06_c3/$dataset rep$rep → ${elapsed}s"
                    flatten_c3_results "$outdir"
                    rm -rf "$sc"
                fi

                # --- c3_fixed ---
                outdir="$base/c3_fixed/${dataset}_rep${rep}"
                if ! has_results "$outdir"; then
                    set_sort_bug "fixed"
                    local sc
                    sc=$(mktemp -d /tmp/schema_c3_XXXX)
                    unzip -q "${ZIP[$org]}" -d "$sc"
                    mkdir -p "$outdir"
                    log "START c3_fixed 06_c3/$dataset rep$rep"
                    start=$(date +%s.%N)
                    conda run -n $CONDA_ENV chewie AlleleCall \
                        -i "$gdir" -g "$sc/${INNER[$org]}" -o "$outdir" \
                        --cpu $CPU --no-inferred > "$outdir/run.log" 2>&1 || true
                    end=$(date +%s.%N)
                    elapsed=$(echo "$end - $start" | bc)
                    printf "06_c3\tc3_fixed\t%s\t%s\t%d\t%.1f\t%s\n" \
                        "$org" "$dataset" "$rep" "$elapsed" "$(date -Iseconds)" >> "$TIMING_LOG"
                    log "DONE  c3_fixed 06_c3/$dataset rep$rep → ${elapsed}s"
                    flatten_c3_results "$outdir"
                    rm -rf "$sc"
                fi
            done
        done
    done

    # Restore to fixed
    set_sort_bug "fixed"
}

###############################################################################
# Main
###############################################################################

main() {
    preflight

    if [ $# -eq 0 ]; then
        bench_01
        bench_02
        bench_03
        bench_04
        bench_05
        bench_06
    else
        for arg in "$@"; do
            case "$arg" in
                01|beone)        bench_01 ;;
                02|circular)     bench_02 ;;
                03|degradation)  bench_03 ;;
                04|synthetic)    bench_04 ;;
                05|outbreak)     bench_05 ;;
                06|c3)           bench_06 ;;
                *)               echo "Unknown: $arg (use 01-06)" >&2 ;;
            esac
        done
    fi

    set_sort_bug "fixed"
    log ""
    log "===== ALL DONE ====="
    log "Timing: $TIMING_LOG"
    log "Runs:   $(tail -n +2 "$TIMING_LOG" | wc -l)"
}

main "$@"
