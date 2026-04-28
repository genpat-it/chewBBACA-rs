#!/usr/bin/env python3
"""
Generate synthetic CDS files with known ground truth for allele calling validation.

KEY DESIGN: generates CDS files directly (no genome assembly, no CDS prediction).
Both tools receive IDENTICAL CDS → any difference is purely alignment engine.

BSR is pre-computed using parasail (same BLOSUM62/gap11/ext1 as both tools).
Borderline BSR cases are calibrated via binary search to hit exact BSR windows.

Usage:
    python generate_synthetic_cds.py \
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

import numpy as np
import parasail
from Bio.Seq import Seq


# ── Constants ──────────────────────────────────────────────────────────────

BLOSUM62 = parasail.blosum62
GAP_OPEN = 11
GAP_EXTEND = 1
BSR_THRESHOLD = 0.6
SIZE_THRESHOLD_DEFAULT = 0.2
MIN_CDS_LEN = 201  # minimum CDS length in bp


def read_schema_config(schema_dir):
    """Read size_threshold from .schema_config pickle if it exists."""
    import pickle
    config_path = os.path.join(schema_dir, '.schema_config')
    if os.path.exists(config_path):
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        st = config.get('size_threshold', [SIZE_THRESHOLD_DEFAULT])
        return st[0] if isinstance(st, list) else st
    return SIZE_THRESHOLD_DEFAULT


# ── FASTA I/O ─────────────────────────────────────────────────────────────

def read_fasta(path):
    """Read FASTA file → [(header, sequence), ...]"""
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
    """Write [(header, seq), ...] as FASTA."""
    with open(path, 'w') as f:
        for header, seq in entries:
            f.write(f'>{header}\n')
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + '\n')


# ── Translation ────────────────────────────────────────────────────────────

def translate_dna(dna):
    """Translate DNA to protein using table 11. Returns None if invalid."""
    if len(dna) < 3 or len(dna) % 3 != 0:
        return None
    try:
        protein = str(Seq(dna).translate(table=11))
        # Remove trailing stop codon
        if protein.endswith('*'):
            protein = protein[:-1]
        # Check no internal stops
        if '*' in protein:
            return None
        # Force first aa to M (standard behavior for both tools)
        if len(protein) > 0:
            protein = 'M' + protein[1:]
        return protein
    except Exception:
        return None


# ── Smith-Waterman scoring ─────────────────────────────────────────────────

def sw_score(query_protein, subject_protein):
    """Compute SW alignment score (parasail, BLOSUM62, gap 11/1)."""
    result = parasail.sw_striped_sat(
        query_protein, subject_protein,
        GAP_OPEN, GAP_EXTEND, BLOSUM62
    )
    return result.score


def compute_self_score(protein):
    """Self-score = SW(protein, protein)."""
    return sw_score(protein, protein)


# ── Mutation ───────────────────────────────────────────────────────────────

STOP_CODONS = {'TAA', 'TAG', 'TGA'}
VALID_STARTS = {'ATG', 'GTG', 'TTG'}


def mutate_sequence(seq, n_snps, rng):
    """Introduce n_snps substitutions, preserving start/stop and avoiding internal stops."""
    bases = list('ACGT')
    seq = list(seq)
    seq_len = len(seq)

    # Only mutate positions 3..(len-3) to preserve start and stop codons
    mutable = list(range(3, seq_len - 3))
    if not mutable:
        return ''.join(seq)
    positions = rng.sample(mutable, min(n_snps, len(mutable)))

    for pos in positions:
        original = seq[pos]
        alternatives = [b for b in bases if b != original]
        rng.shuffle(alternatives)
        # Try each alternative, pick one that doesn't create internal stop
        for alt in alternatives:
            seq[pos] = alt
            codon_start = (pos // 3) * 3
            codon = ''.join(seq[codon_start:codon_start+3])
            if codon not in STOP_CODONS:
                break
        else:
            # All alternatives create stop; revert
            seq[pos] = original

    return ''.join(seq)


def ensure_valid_cds(dna):
    """Ensure CDS has valid start, stop, no internal stops, length % 3 == 0."""
    if len(dna) < 9:
        return None
    # Fix length to multiple of 3
    dna = dna[:len(dna) - (len(dna) % 3)]
    # Check start codon
    if dna[:3] not in VALID_STARTS:
        return None
    # Check stop codon
    if dna[-3:] not in STOP_CODONS:
        return None
    # Check no internal stop codons
    for i in range(3, len(dna) - 3, 3):
        if dna[i:i+3] in STOP_CODONS:
            return None
    return dna


def generate_calibrated_mutation(allele_dna, rep_protein, rep_self_score,
                                  target_bsr_low, target_bsr_high, rng,
                                  max_iter=100):
    """
    Binary search over mutation count to achieve target BSR window.
    Returns (mutated_dna, actual_bsr, n_snps) or None if impossible.
    """
    seq_len = len(allele_dna)
    low = 1
    high = min(seq_len // 2, seq_len - 6)  # don't mutate everything

    best = None
    best_dist = float('inf')

    for iteration in range(max_iter):
        mid = (low + high) // 2
        if mid < 1:
            mid = 1

        # Try several random mutations at this count
        for _ in range(5):
            mutated = mutate_sequence(allele_dna, mid, rng)
            valid = ensure_valid_cds(mutated)
            if valid is None:
                continue
            protein = translate_dna(valid)
            if protein is None or len(protein) < 3:
                continue
            score = sw_score(protein, rep_protein)
            bsr = score / rep_self_score

            dist = 0
            if bsr < target_bsr_low:
                dist = target_bsr_low - bsr
            elif bsr > target_bsr_high:
                dist = bsr - target_bsr_high

            if dist < best_dist:
                best_dist = dist
                best = (valid, bsr, mid)

            if target_bsr_low <= bsr <= target_bsr_high:
                return valid, bsr, mid

        # Adjust binary search
        if best and best[1] > target_bsr_high:
            low = mid + 1
        elif best and best[1] < target_bsr_low:
            high = mid - 1
        else:
            # Try random direction
            if rng.random() < 0.5:
                low = mid + 1
            else:
                high = mid - 1

        if low > high:
            break

    # Return best attempt even if not in window
    return best


def truncate_cds(dna, target_fraction, rng):
    """Truncate CDS to target_fraction, keeping frame and start/stop."""
    target_len = max(9, int(len(dna) * target_fraction))
    target_len = (target_len // 3) * 3

    # Keep start codon, truncate from the end, add stop codon
    if target_len >= len(dna):
        return dna

    # Take from start, replace last 3 with stop codon
    truncated = dna[:target_len - 3] + dna[-3:]  # keep original stop
    if len(truncated) % 3 != 0:
        truncated = truncated[:len(truncated) - (len(truncated) % 3)]
    return truncated


def extend_cds(dna, target_fraction, rng):
    """Extend CDS by inserting random codons before stop codon."""
    target_len = int(len(dna) * target_fraction)
    target_len = (target_len // 3) * 3
    extra_codons_needed = (target_len - len(dna)) // 3

    if extra_codons_needed <= 0:
        return dna

    bases = 'ACGT'
    extra = []
    for _ in range(extra_codons_needed):
        while True:
            codon = ''.join(rng.choice(bases) for _ in range(3))
            if codon not in STOP_CODONS:
                extra.append(codon)
                break

    # Insert before stop codon
    stop = dna[-3:]
    body = dna[:-3]
    return body + ''.join(extra) + stop


# ── Schema loading ─────────────────────────────────────────────────────────

def load_schema(schema_dir):
    """Load schema: loci, alleles, ALL representatives with self-scores."""
    schema_dir = Path(schema_dir)
    short_dir = schema_dir / 'short'

    fasta_files = sorted(schema_dir.glob('*.fasta'))
    loci = {}

    for f in fasta_files:
        locus_name = f.stem
        alleles = read_fasta(str(f))
        if not alleles:
            continue

        # Load ALL representatives from short/ directory
        short_file = short_dir / f'{locus_name}_short.fasta'
        reps = []
        if short_file.exists():
            rep_entries = read_fasta(str(short_file))
        else:
            rep_entries = [alleles[0]]

        for rh, rd in rep_entries:
            rp = translate_dna(rd)
            if rp is None or len(rp) < 3:
                continue
            rss = compute_self_score(rp)
            if rss <= 0:
                continue
            reps.append({'header': rh, 'dna': rd, 'protein': rp, 'self_score': rss})

        if not reps:
            continue

        lengths = [len(seq) for _, seq in alleles]
        mode_len = Counter(lengths).most_common(1)[0][0]

        loci[locus_name] = {
            'alleles': alleles,
            'reps': reps,
            'mode_length': mode_len,
        }

    return loci


def best_bsr(protein, reps):
    """Compute max BSR across all representatives for a locus."""
    max_bsr = 0
    for rep in reps:
        score = sw_score(protein, rep['protein'])
        bsr = score / rep['self_score']
        if bsr > max_bsr:
            max_bsr = bsr
    return max_bsr


# ── Main generator ─────────────────────────────────────────────────────────

def generate_synthetic_cds(schema_dir, output_dir, n_genomes, seed):
    """Generate synthetic CDS files with deterministic ground truth."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    # Read size_threshold from schema config
    SIZE_THRESHOLD = read_schema_config(schema_dir)
    print(f"Size threshold from schema: {SIZE_THRESHOLD}")

    # Class probabilities
    class_probs = {
        'EXC':               0.45,
        'INF_HIGH':          0.08,   # BSR 0.80-0.95
        'INF_MED':           0.06,   # BSR 0.65-0.80
        'INF_BORDER_ABOVE':  0.06,   # BSR 0.60-0.65
        'INF_BORDER_BELOW':  0.06,   # BSR 0.55-0.60
        'ASM':               0.05,
        'ALM':               0.05,
        'LNF':               0.09,
        'NIPH':              0.05,
        'NIPHEM':            0.05,
    }

    # Normalize
    total_p = sum(class_probs.values())
    class_probs = {k: v / total_p for k, v in class_probs.items()}
    classes = list(class_probs.keys())
    probs = [class_probs[c] for c in classes]

    print(f"Loading schema from {schema_dir}...")
    loci = load_schema(schema_dir)
    locus_names = sorted(loci.keys())
    n_loci = len(locus_names)
    print(f"  {n_loci} loci loaded with self-scores")

    os.makedirs(output_dir, exist_ok=True)
    cds_dir = os.path.join(output_dir, 'cds')
    os.makedirs(cds_dir, exist_ok=True)

    # Ground truth storage
    gt_rows = []
    class_counts = Counter()
    bsr_calibration_failures = 0

    for gi in range(n_genomes):
        genome_name = f'synthetic_{gi:04d}'
        cds_entries = []  # (header, dna_seq) for the CDS FASTA file
        cds_idx = 0

        for locus_name in locus_names:
            locus = loci[locus_name]
            alleles = locus['alleles']
            reps = locus['reps']
            # Use first rep for calibration (binary search), but best_bsr for ground truth
            rep_protein = reps[0]['protein']
            rep_self_score = reps[0]['self_score']
            mode_len = locus['mode_length']

            cls = np_rng.choice(classes, p=probs)
            class_counts[cls] += 1

            if cls == 'EXC':
                header, seq = rng.choice(alleles)
                allele_id = header.rsplit('_', 1)[-1]
                cds_idx += 1
                cds_entries.append((f'cds_{cds_idx:06d}', seq))
                protein = translate_dna(seq)
                bsr = best_bsr(protein, reps) if protein else 0
                gt_rows.append({
                    'genome': genome_name, 'locus': locus_name,
                    'expected_class': 'EXC', 'test_class': 'EXC',
                    'source_allele': allele_id, 'n_mutations': 0,
                    'precomputed_bsr': f'{bsr:.4f}',
                    'cds_length': len(seq), 'mode_length': mode_len,
                })

            elif cls in ('INF_HIGH', 'INF_MED', 'INF_BORDER_ABOVE', 'INF_BORDER_BELOW'):
                # BSR target windows
                bsr_windows = {
                    'INF_HIGH':          (0.80, 0.95),
                    'INF_MED':           (0.65, 0.80),
                    'INF_BORDER_ABOVE':  (0.60, 0.65),
                    'INF_BORDER_BELOW':  (0.55, 0.5999),
                }
                target_low, target_high = bsr_windows[cls]

                header, seq = rng.choice(alleles)
                result = generate_calibrated_mutation(
                    seq, rep_protein, rep_self_score,
                    target_low, target_high, rng)
                if result is None:
                    gt_rows.append({
                        'genome': genome_name, 'locus': locus_name,
                        'expected_class': 'LNF', 'test_class': f'{cls}_FAIL',
                        'source_allele': '', 'n_mutations': 0,
                        'precomputed_bsr': '0', 'cds_length': 0,
                        'mode_length': mode_len,
                    })
                    bsr_calibration_failures += 1
                    continue

                mutated, _, n_snps = result

                # Recompute BSR against ALL representatives (ground truth)
                protein = translate_dna(mutated)
                bsr = best_bsr(protein, reps) if protein else 0

                cds_idx += 1
                cds_entries.append((f'cds_{cds_idx:06d}', mutated))

                # Classify based on best BSR across all reps
                size_ratio = len(mutated) / mode_len
                if bsr >= BSR_THRESHOLD:
                    if size_ratio < (1 - SIZE_THRESHOLD):
                        expected = 'ASM'
                    elif size_ratio > (1 + SIZE_THRESHOLD):
                        expected = 'ALM'
                    else:
                        expected = 'INF'
                else:
                    expected = 'LNF'

                gt_rows.append({
                    'genome': genome_name, 'locus': locus_name,
                    'expected_class': expected, 'test_class': cls,
                    'source_allele': header.rsplit('_', 1)[-1],
                    'n_mutations': n_snps,
                    'precomputed_bsr': f'{bsr:.4f}',
                    'cds_length': len(mutated), 'mode_length': mode_len,
                })

            elif cls == 'ASM':
                header, seq = rng.choice(alleles)
                fraction = rng.uniform(0.50, 1.0 - SIZE_THRESHOLD - 0.02)
                truncated = truncate_cds(seq, fraction, rng)
                valid = ensure_valid_cds(truncated)
                if valid is None or len(valid) < MIN_CDS_LEN:
                    gt_rows.append({
                        'genome': genome_name, 'locus': locus_name,
                        'expected_class': 'LNF', 'test_class': 'ASM_FAIL',
                        'source_allele': '', 'n_mutations': 0,
                        'precomputed_bsr': '0', 'cds_length': 0,
                        'mode_length': mode_len,
                    })
                    continue
                protein = translate_dna(valid)
                if protein is None:
                    gt_rows.append({
                        'genome': genome_name, 'locus': locus_name,
                        'expected_class': 'LNF', 'test_class': 'ASM_FAIL',
                        'source_allele': '', 'n_mutations': 0,
                        'precomputed_bsr': '0', 'cds_length': 0,
                        'mode_length': mode_len,
                    })
                    continue
                bsr = best_bsr(protein, reps)
                cds_idx += 1
                cds_entries.append((f'cds_{cds_idx:06d}', valid))
                if bsr >= BSR_THRESHOLD:
                    expected = 'ASM'
                else:
                    expected = 'LNF'
                gt_rows.append({
                    'genome': genome_name, 'locus': locus_name,
                    'expected_class': expected, 'test_class': 'ASM',
                    'source_allele': header.rsplit('_', 1)[-1],
                    'n_mutations': 0,
                    'precomputed_bsr': f'{bsr:.4f}',
                    'cds_length': len(valid), 'mode_length': mode_len,
                })

            elif cls == 'ALM':
                header, seq = rng.choice(alleles)
                fraction = rng.uniform(1.0 + SIZE_THRESHOLD + 0.02, 1.5)
                extended = extend_cds(seq, fraction, rng)
                valid = ensure_valid_cds(extended)
                if valid is None:
                    gt_rows.append({
                        'genome': genome_name, 'locus': locus_name,
                        'expected_class': 'LNF', 'test_class': 'ALM_FAIL',
                        'source_allele': '', 'n_mutations': 0,
                        'precomputed_bsr': '0', 'cds_length': 0,
                        'mode_length': mode_len,
                    })
                    continue
                protein = translate_dna(valid)
                if protein is None:
                    gt_rows.append({
                        'genome': genome_name, 'locus': locus_name,
                        'expected_class': 'LNF', 'test_class': 'ALM_FAIL',
                        'source_allele': '', 'n_mutations': 0,
                        'precomputed_bsr': '0', 'cds_length': 0,
                        'mode_length': mode_len,
                    })
                    continue
                bsr = best_bsr(protein, reps)
                cds_idx += 1
                cds_entries.append((f'cds_{cds_idx:06d}', valid))
                if bsr >= BSR_THRESHOLD:
                    expected = 'ALM'
                else:
                    expected = 'LNF'
                gt_rows.append({
                    'genome': genome_name, 'locus': locus_name,
                    'expected_class': expected, 'test_class': 'ALM',
                    'source_allele': header.rsplit('_', 1)[-1],
                    'n_mutations': 0,
                    'precomputed_bsr': f'{bsr:.4f}',
                    'cds_length': len(valid), 'mode_length': mode_len,
                })

            elif cls == 'LNF':
                # No CDS for this locus
                gt_rows.append({
                    'genome': genome_name, 'locus': locus_name,
                    'expected_class': 'LNF', 'test_class': 'LNF',
                    'source_allele': '', 'n_mutations': 0,
                    'precomputed_bsr': '0', 'cds_length': 0,
                    'mode_length': mode_len,
                })

            elif cls == 'NIPH':
                # Two CDS matching the same locus with DIFFERENT proteins → NIPH
                # Two CDS matching the same locus with SAME protein → NIPHEM
                if len(alleles) < 2:
                    # Not enough alleles, fallback to EXC
                    header, seq = alleles[0]
                    allele_id = header.rsplit('_', 1)[-1]
                    cds_idx += 1
                    cds_entries.append((f'cds_{cds_idx:06d}', seq))
                    protein = translate_dna(seq)
                    bsr = sw_score(protein, rep_protein) / rep_self_score if protein else 0
                    gt_rows.append({
                        'genome': genome_name, 'locus': locus_name,
                        'expected_class': 'EXC', 'test_class': 'NIPH_FALLBACK_EXC',
                        'source_allele': allele_id, 'n_mutations': 0,
                        'precomputed_bsr': f'{bsr:.4f}',
                        'cds_length': len(seq), 'mode_length': mode_len,
                    })
                    continue

                # Pick two alleles with DIFFERENT proteins
                h1, s1 = rng.choice(alleles)
                p1 = translate_dna(s1)
                h2, s2 = rng.choice(alleles)
                p2 = translate_dna(s2)
                attempts = 0
                while (p1 is not None and p2 is not None and p1 == p2) and attempts < 50:
                    h2, s2 = rng.choice(alleles)
                    p2 = translate_dna(s2)
                    attempts += 1

                # Determine actual class based on protein identity
                if p1 is None or p2 is None or p1 == p2:
                    # Could not find two different proteins → this is NIPHEM
                    actual_class = 'NIPHEM'
                else:
                    actual_class = 'NIPH'

                a1 = h1.rsplit('_', 1)[-1]
                a2 = h2.rsplit('_', 1)[-1]
                cds_idx += 1
                cds_entries.append((f'cds_{cds_idx:06d}', s1))
                cds_idx += 1
                cds_entries.append((f'cds_{cds_idx:06d}', s2))
                gt_rows.append({
                    'genome': genome_name, 'locus': locus_name,
                    'expected_class': actual_class, 'test_class': 'NIPH',
                    'source_allele': f'{a1},{a2}', 'n_mutations': 0,
                    'precomputed_bsr': '1.0',
                    'cds_length': len(s1), 'mode_length': mode_len,
                })

            elif cls == 'NIPHEM':
                # Two copies of the SAME allele
                header, seq = rng.choice(alleles)
                allele_id = header.rsplit('_', 1)[-1]
                cds_idx += 1
                cds_entries.append((f'cds_{cds_idx:06d}', seq))
                cds_idx += 1
                cds_entries.append((f'cds_{cds_idx:06d}', seq))
                gt_rows.append({
                    'genome': genome_name, 'locus': locus_name,
                    'expected_class': 'NIPHEM', 'test_class': 'NIPHEM',
                    'source_allele': allele_id, 'n_mutations': 0,
                    'precomputed_bsr': '1.0',
                    'cds_length': len(seq), 'mode_length': mode_len,
                })

        # Write CDS FASTA for this genome
        cds_path = os.path.join(cds_dir, f'{genome_name}.fasta')
        write_fasta(cds_path, cds_entries)

        if (gi + 1) % 10 == 0 or gi == 0:
            print(f"  Generated {gi + 1}/{n_genomes} genomes ({cds_idx} CDS)")

    # ── Write ground truth (detailed) ──────────────────────────────────────

    gt_detail_path = os.path.join(output_dir, 'ground_truth_detail.tsv')
    with open(gt_detail_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=[
            'genome', 'locus', 'expected_class', 'test_class',
            'source_allele', 'n_mutations', 'precomputed_bsr',
            'cds_length', 'mode_length',
        ])
        writer.writeheader()
        writer.writerows(gt_rows)

    # ── Write ground truth (matrix format, compatible with evaluate script) ──

    gt_matrix_path = os.path.join(output_dir, 'ground_truth_class.tsv')
    # Build lookup
    gt_lookup = {(r['genome'], r['locus']): r['expected_class'] for r in gt_rows}
    with open(gt_matrix_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['genome'] + locus_names)
        for gi in range(n_genomes):
            genome_name = f'synthetic_{gi:04d}'
            row = [genome_name] + [gt_lookup.get((genome_name, ln), '?') for ln in locus_names]
            writer.writerow(row)

    # ── Summary ────────────────────────────────────────────────────────────

    print(f"\n=== Summary ===")
    print(f"  Genomes: {n_genomes}")
    print(f"  Loci: {n_loci}")
    total_cells = n_genomes * n_loci
    print(f"  Total cells: {total_cells}")
    print(f"  BSR calibration failures: {bsr_calibration_failures}")
    print(f"\n  Test class distribution:")
    for cls in sorted(class_counts.keys()):
        pct = class_counts[cls] / sum(class_counts.values()) * 100
        print(f"    {cls:25s}: {class_counts[cls]:>8d} ({pct:.1f}%)")

    # Expected class distribution
    exp_counts = Counter(r['expected_class'] for r in gt_rows)
    print(f"\n  Expected class distribution:")
    for cls in sorted(exp_counts.keys()):
        pct = exp_counts[cls] / len(gt_rows) * 100
        print(f"    {cls:15s}: {exp_counts[cls]:>8d} ({pct:.1f}%)")

    print(f"\n  Output: {output_dir}")
    print(f"  Ground truth (detail): {gt_detail_path}")
    print(f"  Ground truth (matrix): {gt_matrix_path}")
    print(f"  CDS files: {cds_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic CDS with BSR-calibrated ground truth"
    )
    parser.add_argument("--schema", required=True, help="Path to schema directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n-genomes", type=int, default=50, help="Number of genomes (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    generate_synthetic_cds(
        schema_dir=args.schema,
        output_dir=args.output,
        n_genomes=args.n_genomes,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
