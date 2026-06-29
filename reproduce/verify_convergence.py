#!/usr/bin/env python3
"""Convergence gate: chewcall vs chewBBACA allelic profiles must agree.

Compares two ``results_alleles_hashed.tsv`` matrices cell-by-cell on the set of
genome-locus cells that *both* tools call (i.e. assign a concrete allele to).
Convergence is defined exactly as in the paper:

  * For every jointly-callable cell, the two CRC32 allele hashes must be equal.
    Any disagreement is a hard failure (exit code 1) -- the results have NOT
    converged.
  * Cells where one tool calls and the other does not ("actionable
    differences") are reported but do not fail the gate; they are the expected,
    documented residual.

Usage:
    verify_convergence.py CHEWCALL_HASHED.tsv CHEWBBACA_HASHED.tsv [--max-actionable PCT]

Exit codes:
    0  converged (zero different-allele cells; actionable rate within bound)
    1  diverged  (>=1 different-allele cell, or actionable rate above bound)
    2  usage / input error
"""
import argparse
import sys

import pandas as pd

# Tokens that denote a non-call (no concrete allele assigned). Anything else in
# a hashed profile is a CRC32 allele hash = a callable cell.
NON_CALL = {
    "LNF", "PLOT3", "PLOT5", "PLOT", "LOTSC", "NIPH", "NIPHEM",
    "ALM", "ASM", "PAMA", "ASM&ALM", "0", "", "-", "N",
}


# Filename tokens to strip from sample (FILE) identifiers so that, e.g., a
# chewcall row "Lm_0001" (from Lm_0001.fasta) matches a chewBBACA row
# "Lm_0001.cds" (from the shared-CDS input Lm_0001.cds.fasta).
_ID_SUFFIXES = {"cds", "fasta", "fa", "fna", "fas", "fsa", "fastq", "fq"}


def normalize_id(s: str) -> str:
    s = str(s).strip()
    parts = s.split(".")
    while len(parts) > 1 and parts[-1].lower() in _ID_SUFFIXES:
        parts.pop()
    return ".".join(parts)


def load_matrix(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, index_col=0)
    df.index = [normalize_id(x) for x in df.index]
    df.columns = [normalize_id(c) for c in df.columns]
    df = df[~df.index.duplicated(keep="first")]
    return df


def is_callable(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip()
    # Strip chewBBACA's "INF-" prefix on inferred alleles before classifying.
    s = s.str.replace(r"^INF-", "", regex=True)
    return ~s.isin(NON_CALL)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chewcall", help="chewcall results_alleles_hashed.tsv")
    ap.add_argument("chewbbaca", help="chewBBACA results_alleles_hashed.tsv")
    ap.add_argument("--max-actionable", type=float, default=100.0,
                    help="fail if actionable-difference rate (%%) exceeds this "
                         "(default: 100, i.e. never fail on actionable alone)")
    ap.add_argument("--label", default="", help="dataset label for the report line")
    args = ap.parse_args()

    try:
        cc = load_matrix(args.chewcall)
        cb = load_matrix(args.chewbbaca)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR reading inputs: {e}", file=sys.stderr)
        return 2

    genomes = cc.index.intersection(cb.index)
    loci = cc.columns.intersection(cb.columns)
    if len(genomes) == 0 or len(loci) == 0:
        print("ERROR: no shared genomes/loci between the two matrices", file=sys.stderr)
        return 2

    cc = cc.loc[genomes, loci]
    cb = cb.loc[genomes, loci]

    cc_call = cc.apply(is_callable)
    cb_call = cb.apply(is_callable)

    both = cc_call & cb_call
    one_only = cc_call ^ cb_call

    # Normalise INF- prefix for the equality test on jointly-callable cells.
    cc_n = cc.apply(lambda s: s.fillna("").astype(str).str.strip().str.replace(r"^INF-", "", regex=True))
    cb_n = cb.apply(lambda s: s.fillna("").astype(str).str.strip().str.replace(r"^INF-", "", regex=True))

    differ = both & (cc_n != cb_n)

    n_cells = int(both.values.sum())
    n_differ = int(differ.values.sum())
    n_actionable = int(one_only.values.sum())
    total = len(genomes) * len(loci)

    agree_pct = 100.0 * (n_cells - n_differ) / n_cells if n_cells else float("nan")
    actionable_pct = 100.0 * n_actionable / total if total else 0.0

    label = f"[{args.label}] " if args.label else ""
    print(f"{label}genomes={len(genomes)} loci={len(loci)} "
          f"jointly-callable={n_cells:,} different-allele={n_differ} "
          f"agreement={agree_pct:.6f}% actionable={n_actionable:,} ({actionable_pct:.4f}%)")

    if n_differ > 0:
        ex = [(g, l) for g in genomes for l in loci if differ.at[g, l]][:10]
        print(f"  DIVERGED: {n_differ} cell(s) with different alleles. First few:", file=sys.stderr)
        for g, l in ex:
            print(f"    {g} / {l}: chewcall={cc.at[g, l]!r} chewbbaca={cb.at[g, l]!r}", file=sys.stderr)
        return 1
    if actionable_pct > args.max_actionable:
        print(f"  FAIL: actionable rate {actionable_pct:.4f}% > bound {args.max_actionable}%", file=sys.stderr)
        return 1

    print(f"  CONVERGED: 100% allele agreement on {n_cells:,} jointly-callable cells.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
