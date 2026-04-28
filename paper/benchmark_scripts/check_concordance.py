#!/usr/bin/env python3
"""
Comprehensive concordance analysis between allele calling tools.
Compares chewcall vs c3_fixed vs c3_original on results_alleles.tsv files.

Robustness checks:
  1. EXC-EXC same allele (critical: both exact -> same allele ID)
  2. EXC <-> LNF transitions (critical: one finds match, other misses entirely)
  3. EXC <-> INF transitions (suspicious: exact vs novel disagreement)
  4. Callable agreement (EXC/INF concordance rate)
  5. Hamming distance preservation (pairwise distance comparison)

Usage:
  python3 check_concordance.py <benchmark_results_dir> [organism]
  python3 check_concordance.py /mnt/disk2/.../02_circular/results
  python3 check_concordance.py /mnt/disk2/.../01_beone/results lm
"""
import csv
import glob
import os
import sys
from collections import Counter
from pathlib import Path


def norm_id(gid):
    """Normalize genome ID: strip path and common suffixes."""
    gid = gid.split('/')[-1]
    for suffix in ['.cds.fasta', '.cds', '.fasta', '.fna', '.fa']:
        if gid.endswith(suffix):
            gid = gid[:-len(suffix)]
    return gid


def norm_locus(locus):
    return locus.split('/')[-1].replace('.fasta', '')


def load(path):
    with open(path) as f:
        r = list(csv.reader(f, delimiter='\t'))
    loci = [norm_locus(l) for l in r[0][1:]]
    data = {}
    for row in r[1:]:
        gid = norm_id(row[0])
        data[gid] = row[1:]
    return loci, data


def classify(val):
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


def find_results(base_dir, tool, org):
    patterns = [
        f'{base_dir}/{tool}/{org}_rep1/results_alleles.tsv',
        f'{base_dir}/{tool}/{org}_rep1/results_*/results_alleles.tsv',
    ]
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    return None


def pairwise_hamming(loci, data, common_loci, idx, common_genomes):
    """Compute pairwise Hamming distances on callable loci."""
    # Build profiles: only EXC/INF values, None for non-callable
    profiles = {}
    for gid in common_genomes:
        prof = []
        for locus in common_loci:
            i = idx[locus]
            v = data[gid][i] if i < len(data[gid]) else ''
            c = classify(v)
            if c == 'EXC':
                prof.append(v.lstrip('*'))
            elif c == 'INF':
                prof.append(v)
            else:
                prof.append(None)
        profiles[gid] = prof

    distances = {}
    glist = sorted(common_genomes)
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


def compare_pair(name_a, loci_a, data_a, name_b, loci_b, data_b):
    common_loci = sorted(set(loci_a) & set(loci_b))
    idx_a = {l: i for i, l in enumerate(loci_a)}
    idx_b = {l: i for i, l in enumerate(loci_b)}
    common_genomes = sorted(set(data_a) & set(data_b))

    if not common_genomes or not common_loci:
        print(f"  WARNING: no overlap! genomes={len(common_genomes)}, loci={len(common_loci)}")
        return

    total = same_value = same_class = 0
    transitions = Counter()
    exc_exc_diff_allele = []
    exc_lnf = []
    exc_inf = []
    callable_total = callable_same = 0

    for gid in common_genomes:
        for locus in common_loci:
            va = data_a[gid][idx_a[locus]] if idx_a[locus] < len(data_a[gid]) else ''
            vb = data_b[gid][idx_b[locus]] if idx_b[locus] < len(data_b[gid]) else ''
            ca = classify(va)
            cb = classify(vb)
            total += 1

            if va == vb:
                same_value += 1
            if ca == cb:
                same_class += 1
            else:
                transitions[(ca, cb)] += 1

            a_call = ca in ('EXC', 'INF')
            b_call = cb in ('EXC', 'INF')
            if a_call and b_call:
                callable_total += 1
                if ca == cb:
                    callable_same += 1

            # Rule 1: EXC-EXC different allele
            if ca == 'EXC' and cb == 'EXC':
                if va.lstrip('*') != vb.lstrip('*'):
                    exc_exc_diff_allele.append((gid, locus, va, vb))

            # Rule 2: EXC <-> LNF
            if (ca == 'EXC' and cb == 'LNF') or (ca == 'LNF' and cb == 'EXC'):
                exc_lnf.append((gid, locus, va, vb, ca, cb))

            # Rule 3: EXC <-> INF
            if (ca == 'EXC' and cb == 'INF') or (ca == 'INF' and cb == 'EXC'):
                exc_inf.append((gid, locus, va, vb, ca, cb))

    # Rule 5: Hamming distances
    do_hamming = len(common_genomes) >= 2 and len(common_genomes) <= 200
    if do_hamming:
        dist_a = pairwise_hamming(loci_a, data_a, common_loci, idx_a, common_genomes)
        dist_b = pairwise_hamming(loci_b, data_b, common_loci, idx_b, common_genomes)
        max_hdiff = 0
        hdiffs = []
        for pair in dist_a:
            da, _ = dist_a[pair]
            db, _ = dist_b.get(pair, (0, 0))
            diff = abs(da - db)
            if diff > 0:
                hdiffs.append((pair, da, db, diff))
            max_hdiff = max(max_hdiff, diff)

    # --- REPORT ---
    print(f"\n{'='*70}")
    print(f"  {name_a} vs {name_b}")
    print(f"  {len(common_genomes)} genomes x {len(common_loci)} loci = {total:,} cells")
    print(f"{'='*70}")

    print(f"\n  --- Overall ---")
    print(f"  Same value:       {same_value:>10,} / {total:,} ({100*same_value/total:.2f}%)")
    print(f"  Same class:       {same_class:>10,} / {total:,} ({100*same_class/total:.2f}%)")

    print(f"\n  --- Callable agreement (EXC+INF) ---")
    print(f"  Both callable:    {callable_total:>10,}")
    if callable_total > 0:
        print(f"  Same class:       {callable_same:>10,} ({100*callable_same/callable_total:.4f}%)")

    # Rule 1
    print(f"\n  [Rule 1] EXC-EXC same allele (CRITICAL)")
    if exc_exc_diff_allele:
        print(f"  *** FAIL: {len(exc_exc_diff_allele)} cells with DIFFERENT allele IDs ***")
        for gid, locus, va, vb in exc_exc_diff_allele[:5]:
            print(f"    {gid} @ {locus}: {name_a}={va}, {name_b}={vb}")
    else:
        print(f"  PASS: 0 violations")

    # Rule 2
    print(f"\n  [Rule 2] EXC <-> LNF (CRITICAL)")
    if exc_lnf:
        a_finds = sum(1 for _, _, _, _, ca, cb in exc_lnf if ca == 'EXC')
        b_finds = sum(1 for _, _, _, _, ca, cb in exc_lnf if cb == 'EXC')
        print(f"  {len(exc_lnf)} transitions: {name_a} finds {a_finds}, {name_b} finds {b_finds}")
        for gid, locus, va, vb, ca, cb in exc_lnf[:5]:
            print(f"    {gid} @ {locus}: {name_a}={va}({ca}), {name_b}={vb}({cb})")
    else:
        print(f"  PASS: 0 transitions")

    # Rule 3
    print(f"\n  [Rule 3] EXC <-> INF (suspicious)")
    if exc_inf:
        a_exc = sum(1 for _, _, _, _, ca, cb in exc_inf if ca == 'EXC')
        b_exc = sum(1 for _, _, _, _, ca, cb in exc_inf if cb == 'EXC')
        print(f"  {len(exc_inf)} transitions: {name_a} EXC={a_exc}, {name_b} EXC={b_exc}")
    else:
        print(f"  PASS: 0 transitions")

    # Rule 5: Hamming distances
    if do_hamming:
        print(f"\n  [Rule 5] Hamming distance preservation")
        n_pairs = len(dist_a)
        print(f"  Pairs:            {n_pairs}")
        print(f"  Pairs with diff:  {len(hdiffs)}")
        print(f"  Max Hamming diff: {max_hdiff}")
        if hdiffs:
            hdiffs.sort(key=lambda x: -x[3])
            for pair, da, db, diff in hdiffs[:3]:
                print(f"    {pair[0]} vs {pair[1]}: {name_a}={da}, {name_b}={db} (d={diff})")
    else:
        if len(common_genomes) < 2:
            print(f"\n  [Rule 5] Hamming: skipped (< 2 genomes)")
        else:
            print(f"\n  [Rule 5] Hamming: skipped ({len(common_genomes)} genomes, too many pairs)")

    # Transitions
    if transitions:
        print(f"\n  --- Transitions ({name_a} -> {name_b}) ---")
        for (ca, cb), count in transitions.most_common(15):
            print(f"    {ca:>8} -> {cb:<8}  {count:>6,}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    results_dir = sys.argv[1]
    org_filter = sys.argv[2] if len(sys.argv) > 2 else None

    tools = ['chewcall', 'c3_fixed', 'c3_original']
    orgs = set()
    for tool in tools:
        tool_dir = os.path.join(results_dir, tool)
        if os.path.isdir(tool_dir):
            for d in os.listdir(tool_dir):
                if '_rep' in d:
                    org = d.rsplit('_rep', 1)[0]
                    orgs.add(org)

    if org_filter:
        orgs = {org_filter} & orgs
    orgs = sorted(orgs)

    if not orgs:
        print(f"No organisms found in {results_dir}")
        sys.exit(1)

    for org in orgs:
        datasets = {}
        for tool in tools:
            path = find_results(results_dir, tool, org)
            if path:
                loci, data = load(path)
                datasets[tool] = (loci, data)

        if 'chewcall' not in datasets:
            print(f"\n  {org}: chewcall results not found, skipping")
            continue

        pairs = [
            ('chewcall', 'c3_fixed'),
            ('chewcall', 'c3_original'),
            ('c3_original', 'c3_fixed'),
        ]
        for a, b in pairs:
            if a in datasets and b in datasets:
                la, da = datasets[a]
                lb, db = datasets[b]
                compare_pair(f"{org}/{a}", la, da, f"{org}/{b}", lb, db)


if __name__ == '__main__':
    main()
