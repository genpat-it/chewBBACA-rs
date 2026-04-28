#!/usr/bin/env python3
"""
Generate synthetic CDS files with evolution-based ground truth.

Unlike generate_synthetic_cds.py which uses BSR-calibrated binary search
(creating circular bias toward parasail), this script uses a simple
evolutionary model: random point mutations at controlled rates.

Ground truth is defined by GENEALOGY (this sequence derives from that locus),
not by BSR. Both tools are evaluated fairly — neither defines the ground truth.

Usage:
    python generate_synthetic_evolution.py \
        --schema /path/to/schema \
        --output /path/to/output \
        --n-genomes 50 \
        --seed 42
"""

import argparse
import csv
import os
import random
import sys
from collections import Counter
from pathlib import Path

from Bio.Seq import Seq


# ── Constants ──────────────────────────────────────────────────────────────

SIZE_THRESHOLD_DEFAULT = 0.2
MIN_CDS_LEN = 201

STOP_CODONS = {'TAA', 'TAG', 'TGA'}
VALID_STARTS = {'ATG', 'GTG', 'TTG'}

# Class distribution (same as original for comparability)
CLASS_WEIGHTS = {
    'EXC': 0.45,
    'INF_LOW': 0.08,    # 1-3 SNPs (trivially detectable)
    'INF_MED': 0.08,    # 4-10 SNPs
    'INF_HIGH': 0.06,   # 11-20 SNPs
    'INF_MANY': 0.03,   # 20-40 SNPs (potentially below BSR threshold)
    'ASM': 0.05,
    'ALM': 0.05,
    'LNF': 0.10,
    'NIPH': 0.05,
    'NIPHEM': 0.05,
}


# ── FASTA I/O ─────────────────────────────────────────────────────────────

def read_fasta(path):
    entries = []
    header = None
    seq_parts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header is not None:
                    entries.append((header, ''.join(seq_parts)))
                header = line[1:].split()[0]
                seq_parts = []
            elif line:
                seq_parts.append(line.upper())
    if header is not None:
        entries.append((header, ''.join(seq_parts)))
    return entries


def write_fasta(path, entries):
    with open(path, 'w') as f:
        for header, seq in entries:
            f.write(f'>{header}\n')
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + '\n')


# ── Translation ────────────────────────────────────────────────────────────

def translate_dna(dna):
    if len(dna) < 3 or len(dna) % 3 != 0:
        return None
    try:
        protein = str(Seq(dna).translate(table=11))
        if protein.endswith('*'):
            protein = protein[:-1]
        if '*' in protein:
            return None
        if len(protein) > 0:
            protein = 'M' + protein[1:]
        return protein
    except Exception:
        return None


# ── Mutation ───────────────────────────────────────────────────────────────

def mutate_sequence(seq, n_snps, rng):
    """Introduce n_snps substitutions, preserving start/stop and avoiding internal stops."""
    bases = list('ACGT')
    seq = list(seq)
    seq_len = len(seq)
    mutable = list(range(3, seq_len - 3))
    if not mutable:
        return ''.join(seq)
    positions = rng.sample(mutable, min(n_snps, len(mutable)))
    for pos in positions:
        original = seq[pos]
        alternatives = [b for b in bases if b != original]
        rng.shuffle(alternatives)
        for alt in alternatives:
            seq[pos] = alt
            codon_start = (pos // 3) * 3
            codon = ''.join(seq[codon_start:codon_start+3])
            if codon not in STOP_CODONS:
                break
        else:
            seq[pos] = original
    return ''.join(seq)


def ensure_valid_cds(dna):
    if len(dna) < 9:
        return None
    dna = dna[:len(dna) - (len(dna) % 3)]
    if dna[:3] not in VALID_STARTS:
        return None
    if dna[-3:] not in STOP_CODONS:
        return None
    for i in range(3, len(dna) - 3, 3):
        if dna[i:i+3] in STOP_CODONS:
            return None
    return dna


def truncate_cds(dna, target_fraction, rng):
    target_len = max(9, int(len(dna) * target_fraction))
    target_len = (target_len // 3) * 3
    if target_len >= len(dna):
        return dna
    # Keep start, truncate, add stop
    body = dna[3:target_len-3]
    stop = rng.choice(list(STOP_CODONS))
    result = dna[:3] + body + stop
    return ensure_valid_cds(result)


def extend_cds(dna, target_fraction, rng):
    target_len = int(len(dna) * target_fraction)
    target_len = (target_len // 3) * 3
    if target_len <= len(dna):
        return dna
    extra_codons = (target_len - len(dna)) // 3
    bases = 'ACGT'
    # Insert random codons before the stop codon
    body = dna[:-3]
    for _ in range(extra_codons):
        while True:
            codon = ''.join(rng.choice(bases) for _ in range(3))
            if codon not in STOP_CODONS:
                break
        body += codon
    result = body + dna[-3:]
    return ensure_valid_cds(result)


# ── Schema loading ─────────────────────────────────────────────────────────

def load_schema(schema_dir):
    """Load schema: {locus_name: [(allele_id, dna_seq), ...]}"""
    schema = {}
    locus_files = sorted(Path(schema_dir).glob('*.fasta'))
    for fpath in locus_files:
        locus_name = fpath.stem
        if locus_name.endswith('_short') or locus_name.startswith('.'):
            continue
        entries = read_fasta(str(fpath))
        if entries:
            schema[locus_name] = entries
    return schema


def compute_mode_length(alleles):
    """Compute mode allele length in bp."""
    lengths = [len(seq) for _, seq in alleles]
    return max(set(lengths), key=lengths.count) if lengths else 0


# ── Main generation ────────────────────────────────────────────────────────

def generate_synthetic(schema_dir, output_dir, n_genomes, seed):
    rng = random.Random(seed)
    np_rng = rng  # use same for simplicity

    schema = load_schema(schema_dir)
    loci = sorted(schema.keys())
    print(f"Loaded {len(loci)} loci from {schema_dir}")

    os.makedirs(output_dir, exist_ok=True)
    cds_dir = os.path.join(output_dir, 'cds')
    os.makedirs(cds_dir, exist_ok=True)

    # Compute mode lengths
    mode_lengths = {l: compute_mode_length(schema[l]) for l in loci}

    # Class list for weighted sampling
    classes = list(CLASS_WEIGHTS.keys())
    weights = [CLASS_WEIGHTS[c] for c in classes]

    # Ground truth records
    detail_rows = []
    class_matrix = {}  # (genome, locus) -> class

    for gi in range(n_genomes):
        genome_id = f'synthetic_{gi:04d}'
        cds_entries = []

        for locus in loci:
            alleles = schema[locus]
            mode_len = mode_lengths[locus]

            # Pick class
            cls = rng.choices(classes, weights=weights, k=1)[0]

            # Pick a random allele as source
            source_header, source_dna = rng.choice(alleles)

            if len(source_dna) < MIN_CDS_LEN:
                # Too short, force EXC or LNF
                cls = rng.choice(['EXC', 'LNF'])

            detail = {
                'genome': genome_id,
                'locus': locus,
                'source_allele': source_header,
                'assigned_class': cls,
                'n_snps': 0,
                'dna_length': len(source_dna),
            }

            if cls == 'EXC':
                cds_entries.append((f'{locus}_{source_header.split("_")[-1]}', source_dna))
                detail['ground_truth'] = 'EXC'

            elif cls.startswith('INF'):
                # Determine SNP count by sub-class
                protein_len = (len(source_dna) - 6) // 3  # approximate
                if cls == 'INF_LOW':
                    n_snps = rng.randint(1, max(1, min(3, protein_len // 10)))
                elif cls == 'INF_MED':
                    n_snps = rng.randint(4, max(4, min(10, protein_len // 5)))
                elif cls == 'INF_HIGH':
                    n_snps = rng.randint(11, max(11, min(20, protein_len // 3)))
                else:  # INF_MANY
                    n_snps = rng.randint(20, max(20, min(40, protein_len // 2)))

                mutated = mutate_sequence(source_dna, n_snps, rng)
                valid = ensure_valid_cds(mutated)
                if valid and translate_dna(valid):
                    cds_entries.append((f'{locus}_novel_{gi}_{n_snps}snps', valid))
                    detail['n_snps'] = n_snps
                    detail['ground_truth'] = 'INF'
                    detail['dna_length'] = len(valid)
                else:
                    # Fallback to EXC
                    cds_entries.append((f'{locus}_{source_header.split("_")[-1]}', source_dna))
                    detail['ground_truth'] = 'EXC'
                    detail['assigned_class'] = 'EXC'

            elif cls == 'ASM':
                fraction = rng.uniform(0.50, 0.78)
                truncated = truncate_cds(source_dna, fraction, rng)
                if truncated and translate_dna(truncated):
                    cds_entries.append((f'{locus}_trunc_{gi}', truncated))
                    detail['ground_truth'] = 'ASM'
                    detail['dna_length'] = len(truncated)
                else:
                    detail['ground_truth'] = 'LNF'
                    detail['assigned_class'] = 'LNF'

            elif cls == 'ALM':
                fraction = rng.uniform(1.22, 1.50)
                extended = extend_cds(source_dna, fraction, rng)
                if extended and translate_dna(extended):
                    cds_entries.append((f'{locus}_ext_{gi}', extended))
                    detail['ground_truth'] = 'ALM'
                    detail['dna_length'] = len(extended)
                else:
                    detail['ground_truth'] = 'LNF'
                    detail['assigned_class'] = 'LNF'

            elif cls == 'LNF':
                # Don't add any CDS
                detail['ground_truth'] = 'LNF'

            elif cls == 'NIPH':
                # Two different alleles
                if len(alleles) >= 2:
                    a1, a2 = rng.sample(alleles, 2)
                    cds_entries.append((f'{locus}_{a1[0].split("_")[-1]}', a1[1]))
                    cds_entries.append((f'{locus}_{a2[0].split("_")[-1]}_dup', a2[1]))
                    detail['ground_truth'] = 'NIPH'
                else:
                    # Only 1 allele, make a mutated copy
                    mutated = mutate_sequence(source_dna, 5, rng)
                    valid = ensure_valid_cds(mutated)
                    if valid:
                        cds_entries.append((f'{locus}_{source_header.split("_")[-1]}', source_dna))
                        cds_entries.append((f'{locus}_para_{gi}', valid))
                        detail['ground_truth'] = 'NIPH'
                    else:
                        cds_entries.append((f'{locus}_{source_header.split("_")[-1]}', source_dna))
                        detail['ground_truth'] = 'EXC'
                        detail['assigned_class'] = 'EXC'

            elif cls == 'NIPHEM':
                # Two identical alleles
                cds_entries.append((f'{locus}_{source_header.split("_")[-1]}', source_dna))
                cds_entries.append((f'{locus}_{source_header.split("_")[-1]}_dup', source_dna))
                detail['ground_truth'] = 'NIPHEM'

            detail_rows.append(detail)
            class_matrix[(genome_id, locus)] = detail['ground_truth']

        # Write CDS FASTA
        write_fasta(os.path.join(cds_dir, f'{genome_id}.cds.fasta'), cds_entries)

        if (gi + 1) % 10 == 0:
            print(f"  Generated {gi+1}/{n_genomes} genomes...")

    # Write ground truth detail
    detail_path = os.path.join(output_dir, 'ground_truth_detail.tsv')
    with open(detail_path, 'w', newline='') as f:
        w = csv.DictWriter(f, delimiter='\t',
                           fieldnames=['genome', 'locus', 'ground_truth', 'assigned_class',
                                       'source_allele', 'n_snps', 'dna_length'])
        w.writeheader()
        w.writerows(detail_rows)

    # Write ground truth class matrix (genome x loci)
    class_path = os.path.join(output_dir, 'ground_truth_class.tsv')
    genomes = sorted(set(g for g, _ in class_matrix.keys()))
    with open(class_path, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['genome'] + loci)
        for g in genomes:
            row = [g] + [class_matrix.get((g, l), 'UNKNOWN') for l in loci]
            w.writerow(row)

    # Print stats
    gt_counts = Counter(d['ground_truth'] for d in detail_rows)
    total = len(detail_rows)
    print(f"\nGenerated {total} cells ({n_genomes} genomes x {len(loci)} loci)")
    print("Ground truth distribution:")
    for cls, count in sorted(gt_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:8s}: {count:>6d} ({100*count/total:.1f}%)")

    # SNP distribution for INF classes
    inf_snps = [d['n_snps'] for d in detail_rows if d['ground_truth'] == 'INF']
    if inf_snps:
        print(f"\nINF SNP distribution: min={min(inf_snps)}, median={sorted(inf_snps)[len(inf_snps)//2]}, max={max(inf_snps)}")


def main():
    parser = argparse.ArgumentParser(description='Generate evolution-based synthetic CDS')
    parser.add_argument('--schema', required=True, help='Schema directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--n-genomes', type=int, default=50, help='Number of synthetic genomes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    generate_synthetic(args.schema, args.output, args.n_genomes, args.seed)


if __name__ == '__main__':
    main()
