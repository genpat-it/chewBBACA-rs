#!/usr/bin/env python3
"""
Analyze results from Benchmark 08 (EFSA regulatory comparison).

Compares chewcall vs chewBBACA v2.8.5 vs v3.5.3 (original + fixed):
  - CRC32 concordance (pairwise between all tool pairs)
  - Actionable differences (called vs not-called)
  - cgMLST clustering at EFSA thresholds
  - Source attribution (nearest non-human isolate)
  - Runtime comparison

Usage:
    python analyze_08_efsa.py /mnt/disk2/a.deruvo/chewcall_benchmarks/08_efsa
"""

import csv
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


# EFSA-recommended cgMLST thresholds (allelic differences)
EFSA_THRESHOLDS = {'lm': 7, 'se': 5, 'ec': 10, 'cj': 7}

# EFSA cgMLST loci counts (INNUENDO schemas)
EFSA_CGMLST_LOCI = {'lm': 1748, 'se': 3254, 'ec': 2360, 'cj': 678}

ORG_NAMES = {
    'lm': 'L. monocytogenes',
    'se': 'S. enterica',
    'ec': 'E. coli',
    'cj': 'C. jejuni',
}

TOOL_LABELS = {
    'chewcall': 'chewcall',
    'c3_original': 'c3 (original)',
    'c3_fixed': 'c3 (fixed)',
    'v2': 'v2.8.5',
}


def norm_id(gid):
    """Normalize genome ID."""
    gid = gid.split('/')[-1]
    for suffix in ['.cds.fasta', '.cds', '.fasta', '.fna', '.fa']:
        if gid.endswith(suffix):
            gid = gid[:-len(suffix)]
    return gid


def norm_locus(locus):
    return locus.split('/')[-1].replace('.fasta', '')


def classify(val):
    """Classify a cell value into one of the 11 classes."""
    if not val or val == '-':
        return 'LNF'
    if val.startswith('INF-'):
        return 'INF'
    try:
        int(val.lstrip('*'))
        return 'EXC'
    except ValueError:
        pass
    for tag in ['LNF', 'NIPHEM', 'NIPH', 'ASM', 'ALM', 'PLOT3', 'PLOT5', 'LOTSC', 'PAMA']:
        if val.startswith(tag):
            return tag
    return val


def is_callable(cls):
    return cls in ('EXC', 'INF')


def load_profiles(path):
    """Load allelic profiles from TSV. Returns (loci, {genome_id: [values]})."""
    with open(path) as f:
        rows = list(csv.reader(f, delimiter='\t'))
    loci = [norm_locus(l) for l in rows[0][1:]]
    data = {}
    for row in rows[1:]:
        gid = norm_id(row[0])
        data[gid] = row[1:]
    return loci, data


def load_hashed(path):
    """Load CRC32-hashed profiles."""
    return load_profiles(path)


def find_results_file(base_dir, tool, org, rep, filename):
    """Find results file, handling chewBBACA timestamped subdirs."""
    direct = os.path.join(base_dir, tool, f"{org}_rep{rep}", filename)
    if os.path.exists(direct):
        return direct
    # Search in subdirs (chewBBACA puts results in timestamped dirs)
    parent = os.path.join(base_dir, tool, f"{org}_rep{rep}")
    if os.path.isdir(parent):
        for d in os.listdir(parent):
            candidate = os.path.join(parent, d, filename)
            if os.path.exists(candidate):
                return candidate
    return None


def compare_profiles(loci_a, data_a, loci_b, data_b):
    """Compare two sets of profiles. Returns detailed comparison dict."""
    idx_a = {l: i for i, l in enumerate(loci_a)}
    idx_b = {l: i for i, l in enumerate(loci_b)}
    common_loci = sorted(set(loci_a) & set(loci_b))
    common_genomes = sorted(set(data_a) & set(data_b))

    result = {
        'n_genomes': len(common_genomes),
        'n_loci': len(common_loci),
        'total_cells': 0,
        'same_value': 0,
        'same_class': 0,
        'a_calls': 0,    # a calls allele, b doesn't
        'b_calls': 0,    # b calls allele, a doesn't
        'diff_allele': 0, # both call but different CRC32
        'transitions': Counter(),
    }

    for gid in common_genomes:
        for locus in common_loci:
            ia = idx_a.get(locus)
            ib = idx_b.get(locus)
            if ia is None or ib is None:
                continue

            va = data_a[gid][ia] if ia < len(data_a[gid]) else ''
            vb = data_b[gid][ib] if ib < len(data_b[gid]) else ''
            ca = classify(va)
            cb = classify(vb)

            result['total_cells'] += 1
            if va == vb:
                result['same_value'] += 1
            if ca == cb:
                result['same_class'] += 1
            else:
                result['transitions'][(ca, cb)] += 1

            a_call = is_callable(ca)
            b_call = is_callable(cb)

            if a_call and not b_call:
                result['a_calls'] += 1
            elif b_call and not a_call:
                result['b_calls'] += 1
            elif a_call and b_call and va != vb:
                # Both call but different value (check CRC32 for hashed profiles)
                result['diff_allele'] += 1

    return result


def compute_hamming_distances(loci, data, common_loci, idx, genomes):
    """Compute pairwise Hamming distances on callable loci."""
    profiles = {}
    for gid in genomes:
        prof = []
        for locus in common_loci:
            i = idx.get(locus)
            if i is None or i >= len(data[gid]):
                prof.append(None)
                continue
            v = data[gid][i]
            c = classify(v)
            if is_callable(c):
                prof.append(v)
            else:
                prof.append(None)
        profiles[gid] = prof

    distances = {}
    glist = sorted(genomes)
    for i, g1 in enumerate(glist):
        for j in range(i + 1, len(glist)):
            g2 = glist[j]
            d = 0
            shared = 0
            for a, b in zip(profiles[g1], profiles[g2]):
                if a is not None and b is not None:
                    shared += 1
                    if a != b:
                        d += 1
            distances[(g1, g2)] = (d, shared)
    return distances


def compare_clustering(dist_a, dist_b, threshold):
    """Compare clustering at a given threshold. Returns (same, diff, a_closer, b_closer)."""
    same = diff = a_closer = b_closer = 0
    for pair in set(dist_a) | set(dist_b):
        da, _ = dist_a.get(pair, (999999, 0))
        db, _ = dist_b.get(pair, (999999, 0))
        in_a = da <= threshold
        in_b = db <= threshold
        if in_a == in_b:
            same += 1
        else:
            diff += 1
            if in_a and not in_b:
                a_closer += 1
            else:
                b_closer += 1
    return same, diff, a_closer, b_closer


def load_timing(timing_file):
    """Load timing data. Returns {(tool, org, mode): [seconds]}."""
    timings = defaultdict(list)
    if not os.path.exists(timing_file):
        return timings
    with open(timing_file) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row.get('benchmark') != '08_efsa':
                continue
            key = (row['tool'], row['organism'], row.get('mode', 'genome'))
            timings[key].append(float(row['seconds']))
    return timings


def median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0
    return s[n // 2]


def analyze_organism(bench_dir, org, mode='genome'):
    """Analyze one organism for one mode (cds or genome)."""
    results_dir = os.path.join(bench_dir, 'results', mode)

    # Determine which tools to compare
    if mode == 'cds':
        tools = ['chewcall', 'c3_original', 'c3_fixed']
    else:
        tools = ['chewcall', 'c3_original', 'c3_fixed', 'v2']

    # Load profiles (use rep1 for concordance analysis)
    profiles = {}  # tool -> (loci, data)
    hashed = {}    # tool -> (loci, data)

    for tool in tools:
        # Try hashed first
        h_path = find_results_file(results_dir, tool, org, 1, 'results_alleles_hashed.tsv')
        r_path = find_results_file(results_dir, tool, org, 1, 'results_alleles.tsv')

        if r_path and os.path.exists(r_path):
            profiles[tool] = load_profiles(r_path)
        if h_path and os.path.exists(h_path):
            hashed[tool] = load_hashed(h_path)

    if len(profiles) < 2:
        print(f"\n  {ORG_NAMES.get(org, org)} ({mode}): insufficient results ({len(profiles)} tools)")
        return

    print(f"\n{'=' * 75}")
    print(f"  {ORG_NAMES.get(org, org)} ({org}) — {mode} mode")
    print(f"{'=' * 75}")

    # --- Pairwise CRC32 concordance ---
    print(f"\n  --- CRC32 concordance (hashed profiles) ---")
    source = hashed if hashed else profiles
    tool_list = [t for t in tools if t in source]

    for ta, tb in combinations(tool_list, 2):
        la, da = source[ta]
        lb, db = source[tb]
        comp = compare_profiles(la, da, lb, db)
        total = comp['total_cells']
        if total == 0:
            continue

        same_pct = comp['same_value'] / total * 100
        label_a = TOOL_LABELS.get(ta, ta)
        label_b = TOOL_LABELS.get(tb, tb)
        print(f"\n  {label_a} vs {label_b}:")
        print(f"    Cells:          {total:>12,}")
        print(f"    Same value:     {comp['same_value']:>12,} ({same_pct:.4f}%)")
        print(f"    {label_a} calls: {comp['a_calls']:>10,}")
        print(f"    {label_b} calls: {comp['b_calls']:>10,}")
        print(f"    Diff allele:    {comp['diff_allele']:>10,}")

        # Top transitions
        if comp['transitions']:
            print(f"    Top transitions ({label_a} → {label_b}):")
            for (ca, cb), count in comp['transitions'].most_common(5):
                print(f"      {ca:>8} → {cb:<8}  {count:>6,}")

    # --- Clustering at EFSA threshold ---
    threshold = EFSA_THRESHOLDS.get(org, 7)
    print(f"\n  --- Clustering at EFSA threshold ({threshold} AD) ---")

    if profiles:
        source_prof = profiles
        tool_list_prof = [t for t in tools if t in source_prof]

        # Compute Hamming distances for each tool
        distances = {}
        for tool in tool_list_prof:
            loci, data = source_prof[tool]
            idx = {l: i for i, l in enumerate(loci)}
            common_genomes = sorted(data.keys())
            if len(common_genomes) <= 500:  # skip if too many pairs
                distances[tool] = compute_hamming_distances(
                    loci, data, loci, idx, common_genomes)

        for ta, tb in combinations([t for t in tool_list_prof if t in distances], 2):
            same, diff, a_closer, b_closer = compare_clustering(
                distances[ta], distances[tb], threshold)
            label_a = TOOL_LABELS.get(ta, ta)
            label_b = TOOL_LABELS.get(tb, tb)
            total_pairs = same + diff
            print(f"  {label_a} vs {label_b}: "
                  f"{diff} different / {total_pairs} pairs "
                  f"({label_a} closer: {a_closer}, {label_b} closer: {b_closer})")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    bench_dir = sys.argv[1]
    org_filter = sys.argv[2] if len(sys.argv) > 2 else None

    organisms = [org_filter] if org_filter else ['lm', 'se', 'ec', 'cj']

    # --- Timing summary ---
    timing_file = os.path.join(bench_dir, 'timing.tsv')
    timings = load_timing(timing_file)

    if timings:
        print(f"\n{'=' * 75}")
        print(f"  RUNTIME SUMMARY (median of {3} reps, {8} CPUs)")
        print(f"{'=' * 75}")
        print(f"\n  {'Organism':<20} {'chewcall':>10} {'v2.8.5':>10} {'v3 (fixed)':>12} "
              f"{'cc/v2':>8} {'cc/v3':>8} {'v2/v3':>8}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")

        for org in organisms:
            cc_t = median(timings.get(('chewcall', org, 'genome'), [0]))
            v2_t = median(timings.get(('v2', org, 'genome'), [0]))
            v3_t = median(timings.get(('c3_fixed', org, 'genome'), [0]))

            cc_s = f"{cc_t:.0f}s" if cc_t else "N/A"
            v2_s = f"{v2_t:.0f}s" if v2_t else "N/A"
            v3_s = f"{v3_t:.0f}s" if v3_t else "N/A"

            sp_v2 = f"{v2_t/cc_t:.1f}x" if cc_t and v2_t else "N/A"
            sp_v3 = f"{v3_t/cc_t:.1f}x" if cc_t and v3_t else "N/A"
            sp_23 = f"{v2_t/v3_t:.1f}x" if v2_t and v3_t else "N/A"

            name = ORG_NAMES.get(org, org)
            print(f"  {name:<20} {cc_s:>10} {v2_s:>10} {v3_s:>12} "
                  f"{sp_v2:>8} {sp_v3:>8} {sp_23:>8}")

    # --- Per-organism analysis ---
    for org in organisms:
        # Genome mode (includes v2.8.5)
        analyze_organism(bench_dir, org, mode='genome')

        # CDS mode (excludes v2.8.5)
        results_cds = os.path.join(bench_dir, 'results', 'cds')
        if os.path.isdir(results_cds):
            analyze_organism(bench_dir, org, mode='cds')

    print(f"\n{'=' * 75}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'=' * 75}")


if __name__ == '__main__':
    main()
