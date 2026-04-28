#!/usr/bin/env python3
"""
Generate CRC32-hashed allelic profiles from chewBBACA v2.8.5 raw output.

v2.8.5 does not produce results_alleles_hashed.tsv. This script reads the
raw results_alleles.tsv and the schema FASTA files, maps allele IDs to their
DNA sequences, computes CRC32 hashes, and writes a hashed profile TSV
compatible with v3/chewcall output.

For non-callable classifications (LNF, NIPH, ASM, ALM, PLOT, etc.), the
original classification string is kept as-is (same as v3 behavior).

For callable alleles (bare number = EXC, INF-N = novel):
  - EXC (bare number): look up allele ID in schema FASTA, hash DNA sequence
  - INF-N: look up novel allele in schema FASTA (v2.8.5 appends them in-place)

This script requires the MODIFIED schema (after v2.8.5 ran), because v2.8.5
adds novel alleles to the FASTA files during the run.

Usage:
    python hash_v2_profiles.py <results_alleles.tsv> <schema_dir> [output.tsv]
"""

import csv
import os
import sys
import zlib


def parse_fasta_alleles(fasta_path):
    """Parse a locus FASTA file. Returns {allele_id: dna_sequence}."""
    alleles = {}
    current_id = None
    current_seq = []
    locus_name = os.path.basename(fasta_path).replace('.fasta', '')

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    alleles[current_id] = ''.join(current_seq).upper()
                header = line[1:].split()[0]
                # Extract allele ID: "locus_name_123" -> "123"
                if '_' in header:
                    parts = header.rsplit('_', 1)
                    try:
                        current_id = int(parts[-1])
                    except ValueError:
                        current_id = header
                else:
                    current_id = header
                current_seq = []
            else:
                current_seq.append(line)
    if current_id is not None:
        alleles[current_id] = ''.join(current_seq).upper()

    return alleles


def crc32_hash(dna_seq):
    """Compute CRC32 hash of a DNA sequence (uppercase)."""
    return zlib.crc32(dna_seq.upper().encode()) & 0xFFFFFFFF


def classify_cell(val):
    """Classify a cell value."""
    if not val or val == '-':
        return 'LNF', None
    if val.startswith('INF-'):
        try:
            return 'INF', int(val[4:].lstrip('*'))
        except ValueError:
            return 'INF', None
    try:
        allele_id = int(val.lstrip('*'))
        return 'EXC', allele_id
    except ValueError:
        return val, None


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    results_path = sys.argv[1]
    schema_dir = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else results_path.replace(
        'results_alleles.tsv', 'results_alleles_hashed.tsv')

    # Load raw profiles
    with open(results_path) as f:
        rows = list(csv.reader(f, delimiter='\t'))

    header = rows[0]
    loci = header[1:]

    # Normalize locus names to match FASTA filenames
    locus_names = []
    for l in loci:
        name = l.split('/')[-1].replace('.fasta', '')
        locus_names.append(name)

    # Load schema alleles for each locus
    print(f"Loading schema from {schema_dir}...")
    schema = {}  # locus_name -> {allele_id -> dna_seq}
    for locus_name in set(locus_names):
        fasta_path = os.path.join(schema_dir, f"{locus_name}.fasta")
        if os.path.exists(fasta_path):
            schema[locus_name] = parse_fasta_alleles(fasta_path)

    print(f"Loaded {len(schema)} loci from schema")

    # Hash profiles
    hashed_rows = [header]  # keep same header
    n_hashed = 0
    n_missing = 0
    n_noncallable = 0

    for row in rows[1:]:
        genome_id = row[0]
        hashed_values = [genome_id]

        for i, val in enumerate(row[1:]):
            locus_name = locus_names[i] if i < len(locus_names) else None
            cls, allele_id = classify_cell(val)

            if cls in ('EXC', 'INF') and allele_id is not None and locus_name in schema:
                alleles = schema[locus_name]
                if allele_id in alleles:
                    h = crc32_hash(alleles[allele_id])
                    hashed_values.append(str(h))
                    n_hashed += 1
                else:
                    # Allele ID not found in schema — keep original value
                    hashed_values.append(val)
                    n_missing += 1
            else:
                # Non-callable or unknown locus
                hashed_values.append(val)
                n_noncallable += 1

        hashed_rows.append(hashed_values)

    # Write output
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for row in hashed_rows:
            writer.writerow(row)

    total = n_hashed + n_missing + n_noncallable
    print(f"Output: {output_path}")
    print(f"Cells: {total} (hashed: {n_hashed}, missing allele: {n_missing}, "
          f"non-callable: {n_noncallable})")


if __name__ == '__main__':
    main()
