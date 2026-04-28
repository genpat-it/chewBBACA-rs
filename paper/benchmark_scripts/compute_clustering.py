#!/usr/bin/env python3
"""
Compute pairwise cgMLST clustering comparison between chewcall and chewBBACA.
Reproduces the clustering table from the paper.

Uses RAW allelic profiles (results_alleles.tsv), not hashed.
A cell is "callable" if it's a bare number or INF-NNN or *NNN.
Hamming distance = loci where BOTH tools call an allele AND the alleles differ.
Loci where either tool is not callable are excluded (missing data).

Usage:
    python compute_clustering.py <cc_alleles.tsv> <cb_alleles.tsv> --thresholds 5 7 10
"""
import argparse
import csv
import sys


def norm_id(gid):
    gid = gid.split('/')[-1]
    for s in ['.cds.fasta', '.cds', '.fasta', '.fna', '.fa']:
        if gid.endswith(s):
            gid = gid[:-len(s)]
    return gid


def norm_locus(l):
    return l.split('/')[-1].replace('.fasta', '')


def classify(val):
    """Classify a cell value. Returns (is_callable, normalized_value).

    Normalizes allele IDs so that INF-702, *702, and 702 all map to '702'.
    This is necessary because within a single tool's output:
    - The first genome to discover a novel allele reports INF-NNN
    - Subsequent genomes with the same CDS report bare NNN (or *NNN in c3)
    These all represent the same allele and must compare as equal.
    """
    if not val or val == '-':
        return False, val
    # Strip INF- prefix
    v = val
    if v.startswith('INF-'):
        v = v[4:]
    # Strip * prefix (c3 protein match marker)
    v = v.lstrip('*')
    # Check if it's a number (allele ID)
    try:
        int(v)
        return True, v
    except ValueError:
        return False, val


def load_profiles(path):
    with open(path) as f:
        rows = list(csv.reader(f, delimiter='\t'))
    loci = [norm_locus(l) for l in rows[0][1:]]
    data = {norm_id(r[0]): r[1:] for r in rows[1:]}
    return loci, data


def main():
    parser = argparse.ArgumentParser(description='Compute clustering comparison')
    parser.add_argument('file_a', help='chewcall allelic profiles TSV')
    parser.add_argument('file_b', help='chewBBACA allelic profiles TSV')
    parser.add_argument('--thresholds', nargs='+', type=int, default=[5, 7, 10])
    parser.add_argument('--cgmlst-loci', help='File with one cgMLST locus name per line (filter to these loci only)')
    args = parser.parse_args()

    print(f"Loading {args.file_a}...")
    loci_a, data_a = load_profiles(args.file_a)
    print(f"Loading {args.file_b}...")
    loci_b, data_b = load_profiles(args.file_b)

    common_loci = sorted(set(loci_a) & set(loci_b))
    if args.cgmlst_loci:
        with open(args.cgmlst_loci) as f:
            cgmlst_set = {line.strip() for line in f if line.strip()}
        before = len(common_loci)
        common_loci = sorted(l for l in common_loci if l in cgmlst_set)
        print(f"Filtered to cgMLST: {before} -> {len(common_loci)} loci ({len(cgmlst_set)} in list)")
    common_genomes = sorted(set(data_a) & set(data_b))
    idx_a = {l: i for i, l in enumerate(loci_a)}
    idx_b = {l: i for i, l in enumerate(loci_b)}

    n_pairs = len(common_genomes) * (len(common_genomes) - 1) // 2
    print(f"Common: {len(common_genomes)} genomes, {len(common_loci)} loci")
    print(f"Pairs: {n_pairs}")

    # Build per-genome callable profiles for each tool
    print("Building profiles...")
    profiles_a = {}
    profiles_b = {}
    for gid in common_genomes:
        pa = []
        pb = []
        for locus in common_loci:
            ia = idx_a[locus]
            ib = idx_b[locus]
            va = data_a[gid][ia] if ia < len(data_a[gid]) else ''
            vb = data_b[gid][ib] if ib < len(data_b[gid]) else ''
            ca, na = classify(va)
            cb, nb = classify(vb)
            pa.append((ca, na))
            pb.append((cb, nb))
        profiles_a[gid] = pa
        profiles_b[gid] = pb

    # Compute pairwise Hamming distances for each tool
    print("Computing pairwise distances...")
    glist = sorted(common_genomes)
    dist_a = {}
    dist_b = {}
    for i in range(len(glist)):
        if i % 100 == 0 and i > 0:
            print(f"  {i}/{len(glist)} genomes processed...")
        for j in range(i + 1, len(glist)):
            g1, g2 = glist[i], glist[j]
            # Tool A distances
            da = 0
            for k in range(len(common_loci)):
                ca1, na1 = profiles_a[g1][k]
                ca2, na2 = profiles_a[g2][k]
                if ca1 and ca2:  # both callable in tool A
                    if na1 != na2:
                        da += 1
            dist_a[(g1, g2)] = da

            # Tool B distances
            db = 0
            for k in range(len(common_loci)):
                cb1, nb1 = profiles_b[g1][k]
                cb2, nb2 = profiles_b[g2][k]
                if cb1 and cb2:  # both callable in tool B
                    if nb1 != nb2:
                        db += 1
            dist_b[(g1, g2)] = db

    # Compare clustering at each threshold
    print(f"\n{'Threshold':>10} {'Pairs':>10} {'Same':>10} {'Diff':>6} {'cc_closer':>10} {'cb_closer':>10}")
    print("-" * 60)
    for t in args.thresholds:
        same = diff = cc_closer = cb_closer = 0
        for pair in dist_a:
            da = dist_a[pair]
            db = dist_b[pair]
            in_a = da <= t
            in_b = db <= t
            if in_a == in_b:
                same += 1
            else:
                diff += 1
                if in_a and not in_b:
                    cc_closer += 1
                else:
                    cb_closer += 1
        total = same + diff
        print(f"{t:>10} {total:>10} {same:>10} {diff:>6} {cc_closer:>10} {cb_closer:>10}")


if __name__ == '__main__':
    main()
