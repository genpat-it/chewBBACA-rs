#!/usr/bin/env python3
"""
Generate synthetic genomes with known ground truth for allele calling validation.

For each synthetic genome, assigns a known classification to each locus:
  - EXC: exact allele from schema (randomly picked)
  - INF: novel allele with controlled SNPs (known BSR range)
  - ASM: truncated allele (shorter than mode × (1 - size_threshold))
  - ALM: extended allele (longer than mode × (1 + size_threshold))
  - LNF: locus absent from genome
  - NIPH: two copies of the locus on different contigs

Output:
  - synthetic genomes (FASTA files)
  - ground_truth.tsv: genome × locus matrix with expected classification
  - ground_truth_alleles.tsv: genome × locus matrix with allele IDs (for EXC)

Usage:
    python generate_synthetic.py \
        --schema /path/to/schema \
        --output /path/to/output \
        --n-genomes 100 \
        --seed 42
"""

import argparse
import csv
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def read_fasta(path):
    """Read a FASTA file, return list of (header, sequence)."""
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
    """Write entries as FASTA. entries = [(header, seq), ...]"""
    with open(path, 'w') as f:
        for header, seq in entries:
            f.write(f'>{header}\n')
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + '\n')


def compute_mode(lengths):
    """Compute mode length (most frequent), matching chewBBACA behavior."""
    if not lengths:
        return 0
    c = Counter(lengths)
    return c.most_common(1)[0][0]


def mutate_sequence(seq, n_snps, rng):
    """Introduce n_snps random substitutions into a DNA sequence."""
    bases = list('ACGT')
    seq = list(seq)
    positions = rng.sample(range(len(seq)), min(n_snps, len(seq)))
    for pos in positions:
        original = seq[pos]
        alternatives = [b for b in bases if b != original]
        seq[pos] = rng.choice(alternatives)
    return ''.join(seq)


def truncate_sequence(seq, target_fraction, rng):
    """Truncate sequence to target_fraction of original length (keep reading frame)."""
    target_len = max(9, int(len(seq) * target_fraction))
    # Round down to multiple of 3
    target_len = (target_len // 3) * 3
    # Random start position (keep in frame)
    max_start = len(seq) - target_len
    if max_start <= 0:
        return seq[:target_len]
    start = rng.randrange(0, max_start, 3)
    return seq[start:start + target_len]


def extend_sequence(seq, target_fraction, rng):
    """Extend sequence by appending random codons to reach target_fraction of original."""
    target_len = int(len(seq) * target_fraction)
    target_len = (target_len // 3) * 3
    extra_len = target_len - len(seq)
    if extra_len <= 0:
        return seq
    # Generate random codons (avoiding stop codons in table 11)
    stop_codons = {'TAA', 'TAG', 'TGA'}
    bases = 'ACGT'
    extra = []
    while len(extra) < extra_len:
        codon = ''.join(rng.choice(bases) for _ in range(3))
        if codon not in stop_codons:
            extra.append(codon)
    # Insert before stop codon
    if seq[-3:] in stop_codons:
        return seq[:-3] + ''.join(extra)[:extra_len] + seq[-3:]
    return seq + ''.join(extra)[:extra_len]


def load_schema(schema_dir):
    """Load schema: locus names, alleles, mode lengths."""
    schema_dir = Path(schema_dir)

    # Find loci from FASTA files
    fasta_files = sorted(schema_dir.glob('*.fasta'))
    loci = {}
    for f in fasta_files:
        locus_name = f.stem
        alleles = read_fasta(str(f))
        if not alleles:
            continue
        lengths = [len(seq) for _, seq in alleles]
        mode_len = compute_mode(lengths)
        loci[locus_name] = {
            'alleles': alleles,
            'mode_length': mode_len,
            'path': str(f),
        }

    return loci


def generate_synthetic_genomes(schema_dir, output_dir, n_genomes, seed,
                                class_probs=None, flanking_len=200):
    """
    Generate n_genomes synthetic genomes with known ground truth.

    class_probs: dict with probabilities for each class.
    Default distribution designed to test borderline cases heavily.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    if class_probs is None:
        class_probs = {
            'EXC': 0.60,       # 60% exact matches
            'INF_easy': 0.05,  # 5% novel, BSR >> 0.6 (easy)
            'INF_border': 0.10, # 10% novel, BSR ~ 0.58-0.65 (borderline!)
            'ASM': 0.05,       # 5% truncated
            'ALM': 0.05,       # 5% extended
            'LNF': 0.10,       # 10% absent
            'NIPH': 0.05,      # 5% paralog (two copies)
        }

    # Normalize
    total = sum(class_probs.values())
    class_probs = {k: v / total for k, v in class_probs.items()}
    classes = list(class_probs.keys())
    probs = [class_probs[c] for c in classes]

    print(f"Loading schema from {schema_dir}...")
    loci = load_schema(schema_dir)
    locus_names = sorted(loci.keys())
    n_loci = len(locus_names)
    print(f"  {n_loci} loci loaded")

    # BSR threshold
    bsr_threshold = 0.6
    size_threshold = 0.2

    os.makedirs(output_dir, exist_ok=True)
    genomes_dir = os.path.join(output_dir, 'genomes')
    os.makedirs(genomes_dir, exist_ok=True)

    # Ground truth matrices
    gt_class = {}   # (genome, locus) -> class
    gt_allele = {}  # (genome, locus) -> allele_id or description

    # Track stats
    class_counts = Counter()

    for gi in range(n_genomes):
        genome_name = f'synthetic_{gi:04d}'
        contigs = []  # list of (contig_name, sequence)
        contig_idx = 0

        # Flanking sequence (random DNA around each locus)
        def random_dna(length):
            return ''.join(rng.choice('ACGT') for _ in range(length))

        # For each locus, decide class and generate sequence
        locus_seqs = []  # (locus_name, class, contig_contributions)

        for locus_name in locus_names:
            locus = loci[locus_name]
            alleles = locus['alleles']
            mode_len = locus['mode_length']

            # Pick class
            cls = np_rng.choice(classes, p=probs)
            class_counts[cls] += 1

            if cls == 'EXC':
                # Pick a random existing allele
                header, seq = rng.choice(alleles)
                allele_id = header.rsplit('_', 1)[-1]
                locus_seqs.append((locus_name, 'EXC', seq, allele_id))
                gt_class[(genome_name, locus_name)] = 'EXC'
                gt_allele[(genome_name, locus_name)] = f'EXC_{allele_id}'

            elif cls == 'INF_easy':
                # Novel allele with few mutations (BSR well above threshold)
                header, seq = rng.choice(alleles)
                # ~2-5% mutation rate -> BSR typically 0.7-0.9
                n_snps = max(1, int(len(seq) * rng.uniform(0.02, 0.05)))
                novel = mutate_sequence(seq, n_snps, rng)
                locus_seqs.append((locus_name, 'INF', novel, f'novel_{n_snps}snps'))
                gt_class[(genome_name, locus_name)] = 'INF'
                gt_allele[(genome_name, locus_name)] = f'INF_novel_{n_snps}snps'

            elif cls == 'INF_border':
                # Novel allele with mutations calibrated for BSR ~ 0.58-0.65
                # This is the critical zone. ~8-12% mutation rate at protein level.
                header, seq = rng.choice(alleles)
                # ~7-15% DNA mutation rate to get borderline protein BSR
                n_snps = max(3, int(len(seq) * rng.uniform(0.07, 0.15)))
                novel = mutate_sequence(seq, n_snps, rng)
                locus_seqs.append((locus_name, 'INF_BORDER', novel, f'border_{n_snps}snps'))
                # Ground truth: should be INF if BSR >= 0.6, LNF if BSR < 0.6
                # We don't know the exact BSR without aligning, so mark as BORDER
                gt_class[(genome_name, locus_name)] = 'INF_BORDER'
                gt_allele[(genome_name, locus_name)] = f'BORDER_{n_snps}snps'

            elif cls == 'ASM':
                # Truncated allele: shorter than mode × (1 - size_threshold)
                header, seq = rng.choice(alleles)
                # Target: 60-75% of mode length (below ASM threshold of 80%)
                fraction = rng.uniform(0.50, 1.0 - size_threshold - 0.02)
                truncated = truncate_sequence(seq, fraction, rng)
                locus_seqs.append((locus_name, 'ASM', truncated, f'trunc_{len(truncated)}bp'))
                gt_class[(genome_name, locus_name)] = 'ASM'
                gt_allele[(genome_name, locus_name)] = f'ASM_{len(truncated)}bp'

            elif cls == 'ALM':
                # Extended allele: longer than mode × (1 + size_threshold)
                header, seq = rng.choice(alleles)
                # Target: 125-150% of mode length (above ALM threshold of 120%)
                fraction = rng.uniform(1.0 + size_threshold + 0.02, 1.5)
                extended = extend_sequence(seq, fraction, rng)
                locus_seqs.append((locus_name, 'ALM', extended, f'ext_{len(extended)}bp'))
                gt_class[(genome_name, locus_name)] = 'ALM'
                gt_allele[(genome_name, locus_name)] = f'ALM_{len(extended)}bp'

            elif cls == 'LNF':
                # Locus not present
                locus_seqs.append((locus_name, 'LNF', None, 'absent'))
                gt_class[(genome_name, locus_name)] = 'LNF'
                gt_allele[(genome_name, locus_name)] = 'LNF'

            elif cls == 'NIPH':
                # Two copies of the locus (paralog)
                h1, seq1 = rng.choice(alleles)
                h2, seq2 = rng.choice(alleles)
                aid1 = h1.rsplit('_', 1)[-1]
                aid2 = h2.rsplit('_', 1)[-1]
                # Put on different contigs to simulate paralogs
                locus_seqs.append((locus_name, 'NIPH', (seq1, seq2), f'paralog_{aid1}_{aid2}'))
                gt_class[(genome_name, locus_name)] = 'NIPH'
                gt_allele[(genome_name, locus_name)] = f'NIPH_{aid1}_{aid2}'

        # Build contigs: group ~50-100 loci per contig with flanking regions
        loci_per_contig = 80
        for chunk_start in range(0, len(locus_seqs), loci_per_contig):
            chunk = locus_seqs[chunk_start:chunk_start + loci_per_contig]
            contig_idx += 1
            contig_name = f'contig{contig_idx:05d}'
            contig_parts = [random_dna(flanking_len)]

            for locus_name, cls, seq_data, desc in chunk:
                if cls == 'LNF':
                    # Skip — locus not present
                    continue
                elif cls == 'NIPH':
                    # First copy goes here
                    seq1, seq2 = seq_data
                    contig_parts.append(seq1)
                    contig_parts.append(random_dna(flanking_len))
                else:
                    contig_parts.append(seq_data)
                    contig_parts.append(random_dna(flanking_len))

            contigs.append((contig_name, ''.join(contig_parts)))

            # For NIPH: add second copy on a separate contig
            niph_seqs = [(ln, sd) for ln, c, sd, _ in chunk if c == 'NIPH']
            if niph_seqs:
                contig_idx += 1
                paralog_name = f'contig{contig_idx:05d}'
                parts = [random_dna(flanking_len)]
                for ln, (s1, s2) in niph_seqs:
                    parts.append(s2)
                    parts.append(random_dna(flanking_len))
                contigs.append((paralog_name, ''.join(parts)))

        # Write genome FASTA
        genome_path = os.path.join(genomes_dir, f'{genome_name}.fasta')
        write_fasta(genome_path, contigs)

        if (gi + 1) % 10 == 0 or gi == 0:
            print(f"  Generated {gi + 1}/{n_genomes} genomes")

    # Write ground truth
    gt_class_path = os.path.join(output_dir, 'ground_truth_class.tsv')
    gt_allele_path = os.path.join(output_dir, 'ground_truth_alleles.tsv')

    with open(gt_class_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['genome'] + locus_names)
        for gi in range(n_genomes):
            genome_name = f'synthetic_{gi:04d}'
            row = [genome_name] + [gt_class.get((genome_name, ln), '?') for ln in locus_names]
            writer.writerow(row)

    with open(gt_allele_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['genome'] + locus_names)
        for gi in range(n_genomes):
            genome_name = f'synthetic_{gi:04d}'
            row = [genome_name] + [gt_allele.get((genome_name, ln), '?') for ln in locus_names]
            writer.writerow(row)

    # Summary
    print(f"\n=== Summary ===")
    print(f"  Genomes: {n_genomes}")
    print(f"  Loci: {n_loci}")
    print(f"  Total cells: {n_genomes * n_loci}")
    print(f"  Class distribution:")
    for cls in sorted(class_counts.keys()):
        pct = class_counts[cls] / sum(class_counts.values()) * 100
        print(f"    {cls:15s}: {class_counts[cls]:>8d} ({pct:.1f}%)")
    print(f"\n  Output: {output_dir}")
    print(f"  Ground truth: {gt_class_path}")

    return gt_class_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic genomes with known ground truth for allele calling validation"
    )
    parser.add_argument("--schema", required=True, help="Path to schema directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n-genomes", type=int, default=100, help="Number of genomes (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--flanking", type=int, default=200, help="Flanking DNA length (default: 200)")
    args = parser.parse_args()

    generate_synthetic_genomes(
        schema_dir=args.schema,
        output_dir=args.output,
        n_genomes=args.n_genomes,
        seed=args.seed,
        flanking_len=args.flanking,
    )


if __name__ == '__main__':
    main()
