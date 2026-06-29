#!/usr/bin/env python3
"""Render the filter-safety table (Table: minimizer pre-filter coverage) from
the summary produced by 10_schema_filter_safety.sh.

Usage:
    15_table_filter_safety.py output/extensions/filter_safety/filter_safety_summary.tsv
"""
import sys

NAME = {
    "lm": (r"\textit{L.~monocytogenes}", "cgMLST"),
    "se": (r"\textit{S.~enterica}", "wgMLST"),
    "ec": (r"\textit{E.~coli}", "wgMLST"),
    "cj": (r"\textit{C.~jejuni}", "wgMLST"),
}


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    rows = []
    with open(sys.argv[1]) as fh:
        header = fh.readline()
        for line in fh:
            org, loci, flagged, prom, wb, wa = line.rstrip("\n").split("\t")
            rows.append((org, int(loci), int(flagged), int(prom), float(wb), float(wa)))

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Worst-case minimizer-overlap (wcr) audit of the chewcall "
          r"pre-filter on the four BeONE schemas (inferred alleles excluded). "
          r"\emph{Flagged} loci have wcr $<\tau=0.20$; \texttt{constructive\_remedy} "
          r"promotes witness alleles to representatives so every locus attains "
          r"wcr $\geq\tau$ by construction.}")
    print(r"\label{tab:filter_safety}")
    print(r"\begin{tabular}{llrrrrr}")
    print(r"\toprule")
    print(r"Organism & Schema & Loci & Flagged & \% & Promoted & wcr$_\text{min}$ \\")
    print(r"\midrule")
    for org, loci, flagged, prom, wb, wa in rows:
        name, kind = NAME.get(org, (org, "?"))
        pct = 100.0 * flagged / loci if loci else 0.0
        print(f"{name} & {kind} & {loci:,} & {flagged} & {pct:.1f} & {prom} & {wb:.3f} \\\\".replace(",", r"\,"))
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
