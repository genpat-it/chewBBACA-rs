#!/bin/bash
# Self-contained end-to-end smoke test: build a tiny synthetic chewBBACA-layout
# schema and a matching sample, run chewcall, and check it produces a valid
# allelic profile. Exercises CDS loading, hashing, exact matching and output
# writing with no external data. Run from the repo root in CI.
set -euo pipefail

BIN="${CHEWCALL_BIN:-./target/release/chewcall}"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/schema/short" "$T/genomes" "$T/cds"

# One locus, one allele: a clean CDS (starts ATG, ends TAA, length a multiple of
# 3, no internal stop codon).
ALLELE="ATGGCTGGTCTTATTGTTACTTCTCCTAAAGATGCTGGTCTTATTGTTACTTCTCCTAAAGATTAA"
printf '>locus1_1\n%s\n' "$ALLELE" > "$T/schema/locus1.fasta"
printf '>locus1_1\n%s\n' "$ALLELE" > "$T/schema/short/locus1_short.fasta"

# A sample whose single CDS is exactly that allele -> expect an exact (EXC) call.
printf '>contig1\n%s\n' "$ALLELE" > "$T/genomes/sample1.fasta"
printf '>sample1_1\n%s\n'  "$ALLELE" > "$T/cds/sample1.cds.fasta"

echo "Running chewcall on the synthetic dataset..."
"$BIN" -i "$T/genomes" -g "$T/schema" -o "$T/out" --cpu 1 --cds-input "$T/cds"

prof="$T/out/results_alleles.tsv"
[ -f "$prof" ] || { echo "FAIL: results_alleles.tsv not produced"; exit 1; }
echo "--- results_alleles.tsv ---"; cat "$prof"

# The sample row must exist and assign allele 1 (exact match) at locus1.
if grep -q "sample1" "$prof" && grep -Eq "(^|[[:space:]])1([[:space:]]|$)" "$prof"; then
  echo "SMOKE VALIDATION OK: exact allele call produced as expected"
else
  echo "FAIL: expected an exact allele call for the synthetic sample"; exit 1
fi
