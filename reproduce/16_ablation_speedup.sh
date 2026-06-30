#!/usr/bin/env bash
# 16_ablation_speedup.sh — Supplementary Table S7 (speedup decomposition).
#
# Controlled runtime ablation on ONE representative dataset (L. monocytogenes
# consortium: 1426 genomes, 1748 loci), 8 threads, shared pre-computed CDS.
# Toggles one chewcall component at a time, holding input/schema/threads fixed:
#   - fast        : parasail SIMD Smith-Waterman + minimizer pre-filter (default)
#   - compatible  : BLASTp + minimizer pre-filter   (isolates the alignment engine)
#   - brute       : parasail, pre-filter DISABLED    (isolates the filter necessity;
#                   aligns every residual CDS against all representatives -> very slow,
#                   intended to be capped: it demonstrates intractability, not a number)
#
# The chewBBACA v3.5.4 runtime for the same dataset (Table 1) provides the
# architecture/subprocess factor (chewBBACA BLASTp pipeline vs chewcall compatible).
#
# Env (defaults assume the archive layout / config.sh):
#   CHEWCALL_BIN  (default: ./target/release/chewcall, or $CHEWCALL_BIN)
#   GENOMES_LM    (default: $BENCH_BASE/data/lm/genomes)
#   CDS_LM        (default: $BENCH_BASE/data/lm/cds)
#   SCHEMA_LM     (default: $BENCH_BASE/schemas/lm_cgmlst)
#   BLASTP        (path to blastp, required for the compatible run)
#   CPU           (default: 8)
#   BRUTE_CAP     (seconds; default 5400 = 90 min, after which the brute run is killed)
set -uo pipefail

CHEWCALL_BIN="${CHEWCALL_BIN:-./target/release/chewcall}"
BENCH_BASE="${BENCH_BASE:-.}"
GENOMES_LM="${GENOMES_LM:-$BENCH_BASE/data/lm/genomes}"
CDS_LM="${CDS_LM:-$BENCH_BASE/data/lm/cds}"
SCHEMA_LM="${SCHEMA_LM:-$BENCH_BASE/schemas/lm_cgmlst}"
BLASTP="${BLASTP:-blastp}"
CPU="${CPU:-8}"
BRUTE_CAP="${BRUTE_CAP:-5400}"
OUT="${OUT:-./abl_out}"
mkdir -p "$OUT"

run_timed() {   # $1=label  $2..=extra chewcall args
    local label="$1"; shift
    local schema_tmp="$OUT/schema_${label}"
    rm -rf "$schema_tmp"; cp -r "$SCHEMA_LM" "$schema_tmp"
    mkdir -p "$OUT/$label"
    local start end
    start=$(date +%s.%N)
    "$CHEWCALL_BIN" -i "$GENOMES_LM" -g "$schema_tmp" -o "$OUT/$label" \
        --cpu "$CPU" --cds-input "$CDS_LM" "$@" > "$OUT/$label.log" 2>&1
    end=$(date +%s.%N)
    echo "$label $(echo "$end - $start" | bc) s   (LNF: $(grep -oE 'LNF: [0-9]+' "$OUT/$label.log" | tail -1))"
    rm -rf "$schema_tmp"
}

echo "=== S7 speedup ablation — Lm consortium, $CPU threads ==="
run_timed fast
run_timed compatible --mode compatible --blastp-path "$BLASTP"

echo "--- brute-residual (capped at ${BRUTE_CAP}s; expected to be killed) ---"
schema_b="$OUT/schema_brute"; rm -rf "$schema_b"; cp -r "$SCHEMA_LM" "$schema_b"; mkdir -p "$OUT/brute"
start=$(date +%s.%N)
timeout "$BRUTE_CAP" "$CHEWCALL_BIN" -i "$GENOMES_LM" -g "$schema_b" -o "$OUT/brute" \
    --cpu "$CPU" --cds-input "$CDS_LM" --brute-residual > "$OUT/brute.log" 2>&1
rc=$?
end=$(date +%s.%N)
if [ $rc -eq 124 ]; then
    echo "brute >${BRUTE_CAP}s (capped; >= $(echo "$BRUTE_CAP / 15" | bc)x the fast run) — exact SW without the pre-filter is intractable at schema scale"
else
    echo "brute $(echo "$end - $start" | bc) s"
fi
rm -rf "$schema_b"
echo "=== done; chewBBACA v3.5.4 reference runtime for this dataset is in Table 1 ==="
