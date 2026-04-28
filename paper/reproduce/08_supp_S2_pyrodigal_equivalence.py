#!/usr/bin/env python3
"""08_supp_S2_pyrodigal_equivalence.py — Supplementary Section S2.

Verifies that the pyrodigal versions bundled with chewBBACA v3.5.4 (3.7.1)
and v3.3.10 (3.6.3) produce identical CDS predictions on representative
genomes from the four organisms.  This justifies sharing pre-computed CDS
between the two reference versions.

Output: per-organism count + MD5 hash of concatenated CDS, identical across
both pyrodigal versions when the schemas validate.

Usage: python 08_supp_S2_pyrodigal_equivalence.py \\
            --genome lm=/path/to/Lm_X.fasta se=/path/to/Se_X.fasta ec=... cj=...
"""
import argparse, hashlib, sys
import pyrodigal


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genome", action="append", default=[], required=True,
                    help="org=path mapping, repeat for each organism (lm/se/ec/cj)")
    args = ap.parse_args()
    genomes = dict(g.split("=", 1) for g in args.genome)

    # Pyrodigal as-installed in the current environment.
    of = pyrodigal.GeneFinder(meta=False, closed=True, mask=True)
    print(f"% Pyrodigal version: {pyrodigal.__version__}")
    for org, path in genomes.items():
        with open(path) as f:
            seq = "".join(line.strip() for line in f if not line.startswith(">"))
        of.train(seq.encode())
        cdss = list(of.find_genes(seq.encode()))
        h = hashlib.md5()
        for g in cdss: h.update(g.sequence().encode())
        print(f"{org}: {len(cdss)} CDS, MD5={h.hexdigest()[:16]}")


if __name__ == "__main__":
    main()
