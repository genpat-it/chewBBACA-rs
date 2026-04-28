#!/usr/bin/env python3
"""
Verify that two pyrodigal versions produce identical CDS predictions.

Runs pyrodigal from two conda environments on one genome per organism
and compares CDS count and MD5 hash of concatenated sequences.

Usage:
    python check_pyrodigal_equivalence.py \
        --env1 chewbbaca_354 --env2 chewbbaca_3310 \
        --genomes /path/to/lm.fasta /path/to/se.fasta ...
"""
import argparse
import hashlib
import subprocess
import sys

PYRODIGAL_SCRIPT = '''
import pyrodigal, hashlib, sys
from Bio import SeqIO
seqs = list(SeqIO.parse(sys.argv[1], 'fasta'))
p = pyrodigal.GeneFinder(meta=False, closed=True, mask=True)
p.train(bytes(seqs[0].seq), translation_table=11)
all_cds = []
for s in seqs:
    for g in p.find_genes(bytes(s.seq)):
        all_cds.append(g.sequence())
h = hashlib.md5('|'.join(all_cds).encode()).hexdigest()
print(f"{len(all_cds)}\\t{h}")
'''


def run_pyrodigal(env, genome):
    result = subprocess.run(
        ["conda", "run", "-n", env, "python3", "-c", PYRODIGAL_SCRIPT, genome],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"ERROR in {env}: {result.stderr}", file=sys.stderr)
        return None, None
    parts = result.stdout.strip().split('\t')
    return int(parts[0]), parts[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env1', required=True, help='First conda environment')
    parser.add_argument('--env2', required=True, help='Second conda environment')
    parser.add_argument('--genomes', nargs='+', required=True, help='Genome FASTA files')
    args = parser.parse_args()

    diffs = 0
    for genome in args.genomes:
        n1, h1 = run_pyrodigal(args.env1, genome)
        n2, h2 = run_pyrodigal(args.env2, genome)
        match = "IDENTICAL" if h1 == h2 else "DIFFERENT"
        if h1 != h2:
            diffs += 1
        print(f"{genome}: {args.env1}={n1} CDS (md5={h1}), {args.env2}={n2} CDS (md5={h2}) -> {match}")

    print(f"\nTotal: {len(args.genomes)} genomes, {diffs} differences")
    sys.exit(1 if diffs > 0 else 0)


if __name__ == '__main__':
    main()
