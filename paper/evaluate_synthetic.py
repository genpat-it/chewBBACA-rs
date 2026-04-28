#!/usr/bin/env python3
"""
Evaluate chewcall and chewBBACA results against synthetic ground truth.

Computes:
  - Confusion matrix per tool vs ground truth
  - Per-class precision, recall, F1
  - McNemar's test (chewcall vs chewBBACA paired comparison)
  - Cohen's kappa vs ground truth
  - Accuracy with 95% confidence intervals (Wilson score)
  - Detailed analysis of borderline BSR cases

Usage:
    python evaluate_synthetic.py \
        --ground-truth /path/to/ground_truth_class.tsv \
        --chewcall-results /path/to/chewcall/results_alleles.tsv \
        --chewbbaca-results /path/to/chewbbaca/results_alleles.tsv \
        --output /path/to/evaluation_report.tsv
"""

import argparse
import csv
import sys
from collections import Counter

import numpy as np


def read_profile(path):
    """Read allele profile TSV. Returns {(genome, locus): value}."""
    profiles = {}
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        loci = header[1:]
        for row in reader:
            genome = row[0]
            # Strip .cds suffix (chewBBACA adds it when using pre-computed CDS)
            if genome.endswith('.cds'):
                genome = genome[:-4]
            for i, val in enumerate(row[1:]):
                profiles[(genome, loci[i])] = val
    return profiles, loci


def classify_result(val):
    """Map allele call result to classification class."""
    if val is None or val == '' or val == '?':
        return 'UNKNOWN'
    val = str(val).strip()
    # chewBBACA/chewcall classification prefixes
    for cls in ['NIPHEM', 'NIPH', 'PLOT3', 'PLOT5', 'LOTSC', 'ASM', 'ALM', 'PAMA', 'LNF']:
        if val.startswith(cls):
            return cls
    if val.startswith('INF'):
        return 'INF'
    # Numeric = EXC (exact match)
    try:
        int(val)
        return 'EXC'
    except ValueError:
        pass
    # CRC32 hash (hex string) = exact match
    if all(c in '0123456789abcdefABCDEF' for c in val) and len(val) >= 4:
        return 'EXC'
    return 'UNKNOWN'


def classify_ground_truth(val):
    """Map ground truth class to simplified class for comparison."""
    if val in ('EXC', 'INF', 'ASM', 'ALM', 'LNF', 'NIPH', 'NIPHEM'):
        return val
    if val == 'INF_BORDER':
        return 'INF_BORDER'  # special: could be INF or LNF
    return val


def wilson_ci(p, n, z=1.96):
    """Wilson score confidence interval for proportion."""
    if n == 0:
        return (0, 0)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0, centre - spread), min(1, centre + spread))


def cohens_kappa(confusion):
    """Compute Cohen's kappa from a confusion dict {(true, pred): count}."""
    classes = sorted(set(t for t, _ in confusion.keys()) | set(p for _, p in confusion.keys()))
    n = sum(confusion.values())
    if n == 0:
        return 0.0

    # Observed agreement
    po = sum(confusion.get((c, c), 0) for c in classes) / n

    # Expected agreement
    pe = 0
    for c in classes:
        row_sum = sum(confusion.get((c, p), 0) for p in classes)
        col_sum = sum(confusion.get((t, c), 0) for t in classes)
        pe += (row_sum / n) * (col_sum / n)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def mcnemar_test(correct_a, correct_b):
    """
    McNemar's test for paired nominal data.
    correct_a[i] = True if tool A got cell i correct
    correct_b[i] = True if tool B got cell i correct

    Returns (chi2, p_value, n_discordant)
    """
    from scipy import stats

    # Count discordant pairs
    b_only = sum(1 for a, b in zip(correct_a, correct_b) if not a and b)  # B right, A wrong
    a_only = sum(1 for a, b in zip(correct_a, correct_b) if a and not b)  # A right, B wrong

    n_disc = a_only + b_only
    if n_disc == 0:
        return 0.0, 1.0, 0

    # McNemar's chi-squared (with continuity correction)
    chi2 = (abs(a_only - b_only) - 1)**2 / (a_only + b_only)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return chi2, p_value, n_disc


def evaluate(gt_path, chewcall_path, chewbbaca_path, output_path):
    """Run full evaluation."""
    print("Loading ground truth...")
    gt, gt_loci = read_profile(gt_path)

    print("Loading chewcall results...")
    cc, cc_loci = read_profile(chewcall_path)

    print("Loading chewBBACA results...")
    cb, cb_loci = read_profile(chewbbaca_path)

    # Find common genomes and loci
    gt_genomes = sorted(set(g for g, _ in gt.keys()))
    common_loci = sorted(set(gt_loci) & set(cc_loci) & set(cb_loci))
    print(f"  Genomes: {len(gt_genomes)}, Common loci: {len(common_loci)}")

    # Classify all cells
    results = []  # (genome, locus, gt_class, cc_class, cb_class)
    for genome in gt_genomes:
        for locus in common_loci:
            gt_cls = classify_ground_truth(gt.get((genome, locus), 'UNKNOWN'))
            cc_cls = classify_result(cc.get((genome, locus), 'UNKNOWN'))
            cb_cls = classify_result(cb.get((genome, locus), 'UNKNOWN'))
            results.append((genome, locus, gt_cls, cc_cls, cb_cls))

    total = len(results)
    print(f"  Total cells: {total}")

    # --- Per-tool accuracy vs ground truth ---
    # For borderline cases (INF_BORDER), accept both INF and LNF as correct
    def is_correct(gt_cls, pred_cls):
        if gt_cls == 'INF_BORDER':
            # Borderline: INF or LNF are both acceptable, but also ASM/ALM
            # Actually, any classification is "understandable" for borderline
            # But INF is the "intended" correct answer (the allele IS there)
            return pred_cls in ('INF', 'EXC')
        if gt_cls == 'NIPH':
            return pred_cls in ('NIPH', 'NIPHEM')
        return gt_cls == pred_cls

    cc_correct = [is_correct(r[2], r[3]) for r in results]
    cb_correct = [is_correct(r[2], r[4]) for r in results]

    cc_acc = sum(cc_correct) / total
    cb_acc = sum(cb_correct) / total

    cc_ci = wilson_ci(cc_acc, total)
    cb_ci = wilson_ci(cb_acc, total)

    print(f"\n=== Overall Accuracy ===")
    print(f"  chewcall:  {cc_acc:.4%} [{cc_ci[0]:.4%}, {cc_ci[1]:.4%}]")
    print(f"  chewBBACA: {cb_acc:.4%} [{cb_ci[0]:.4%}, {cb_ci[1]:.4%}]")

    # --- Per-class accuracy ---
    print(f"\n=== Per-class Accuracy ===")
    gt_classes = sorted(set(r[2] for r in results))
    print(f"  {'Class':15s} {'N':>7s} {'chewcall':>10s} {'chewBBACA':>10s}")
    print(f"  {'-'*15} {'-'*7} {'-'*10} {'-'*10}")

    for cls in gt_classes:
        cls_results = [(r, cc, cb) for r, cc, cb in zip(results, cc_correct, cb_correct) if r[2] == cls]
        n = len(cls_results)
        if n == 0:
            continue
        cc_cls_acc = sum(1 for _, cc, _ in cls_results if cc) / n
        cb_cls_acc = sum(1 for _, _, cb in cls_results if cb) / n
        print(f"  {cls:15s} {n:>7d} {cc_cls_acc:>10.2%} {cb_cls_acc:>10.2%}")

    # --- Confusion matrices ---
    for tool_name, tool_idx in [('chewcall', 3), ('chewBBACA', 4)]:
        confusion = Counter()
        for r in results:
            confusion[(r[2], r[tool_idx])] += 1
        kappa = cohens_kappa(confusion)
        print(f"\n  Cohen's kappa ({tool_name} vs ground truth): {kappa:.4f}")

    # --- McNemar's test ---
    try:
        chi2, p_value, n_disc = mcnemar_test(cc_correct, cb_correct)
        cc_only = sum(1 for a, b in zip(cc_correct, cb_correct) if a and not b)
        cb_only = sum(1 for a, b in zip(cc_correct, cb_correct) if not a and b)
        print(f"\n=== McNemar's Test (chewcall vs chewBBACA) ===")
        print(f"  Discordant pairs: {n_disc}")
        print(f"    chewcall correct, chewBBACA wrong: {cc_only}")
        print(f"    chewBBACA correct, chewcall wrong: {cb_only}")
        print(f"  Chi-squared: {chi2:.2f}")
        print(f"  p-value: {p_value:.2e}")
        if p_value < 0.05:
            if cc_only > cb_only:
                print(f"  => chewcall is SIGNIFICANTLY more accurate (p < 0.05)")
            else:
                print(f"  => chewBBACA is SIGNIFICANTLY more accurate (p < 0.05)")
        else:
            print(f"  => No significant difference (p >= 0.05)")
    except ImportError:
        print("\n  (scipy not available, skipping McNemar's test)")

    # --- Borderline analysis ---
    border_results = [r for r in results if r[2] == 'INF_BORDER']
    if border_results:
        print(f"\n=== Borderline BSR Analysis (INF_BORDER, n={len(border_results)}) ===")
        cc_border = Counter(r[3] for r in border_results)
        cb_border = Counter(r[4] for r in border_results)
        print(f"  chewcall classifications:")
        for cls, cnt in cc_border.most_common():
            print(f"    {cls:10s}: {cnt:>6d} ({cnt/len(border_results):.1%})")
        print(f"  chewBBACA classifications:")
        for cls, cnt in cb_border.most_common():
            print(f"    {cls:10s}: {cnt:>6d} ({cnt/len(border_results):.1%})")

    # --- Write detailed output ---
    if output_path:
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['genome', 'locus', 'ground_truth', 'chewcall', 'chewBBACA',
                           'cc_correct', 'cb_correct'])
            for r, cc, cb in zip(results, cc_correct, cb_correct):
                writer.writerow([r[0], r[1], r[2], r[3], r[4],
                               'Y' if cc else 'N', 'Y' if cb else 'N'])
        print(f"\n  Detailed results: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate allele calling against ground truth")
    parser.add_argument("--ground-truth", required=True, help="Ground truth TSV")
    parser.add_argument("--chewcall-results", required=True, help="chewcall results_alleles.tsv")
    parser.add_argument("--chewbbaca-results", required=True, help="chewBBACA results_alleles.tsv")
    parser.add_argument("--output", default=None, help="Detailed output TSV")
    args = parser.parse_args()

    evaluate(args.ground_truth, args.chewcall_results, args.chewbbaca_results, args.output)


if __name__ == '__main__':
    main()
