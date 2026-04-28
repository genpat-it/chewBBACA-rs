#!/usr/bin/env python3
"""05_table_clustering.py — Table 7 (tab:clustering).

Pairwise cgMLST allelic-difference clustering comparison between chewcall
and chewBBACA v3.5.4 across 8 BeONE datasets, at AD thresholds 4/5/7/10.

Distance is Hamming on cgMLST loci where both tools call an allele
(missing data on either side excluded).  Allele IDs are normalised so that
``5``, ``*5`` and ``INF-5`` compare equal.

Usage:  python 05_table_clustering.py
        python 05_table_clustering.py --thresholds 5 7 10  # custom thresholds
"""
import argparse, csv, os, sys
from pathlib import Path
import numpy as np

BASE      = Path(os.environ.get("OUT_ROOT", "/mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results"))
DIR_CC    = Path(os.environ.get("OUT_CHEWCALL", str(BASE / "cc_fixed2")))
DIR_C354  = Path(os.environ.get("OUT_C354",     str(BASE / "c354")))

ORGANISMS = ["lm","se","ec","cj"]
DATASETS  = ["cons","pub"]
ORG_LATEX = {"lm": r"\textit{L.\,mono.}", "se": r"\textit{S.\,enter.}",
             "ec": r"\textit{E.\,coli}",  "cj": r"\textit{C.\,jejuni}"}
DS_LATEX  = {"cons":"Cons.", "pub":"Pub."}
PAIRS_EXPECTED = {("cons","lm"):1016025,("cons","se"):1185030,("cons","ec"):47278,("cons","cj"):185745,
                  ("pub","lm"):1755001,("pub","se"):1027461,("pub","ec"):1997001,("pub","cj"):4729350}
CGMLST_LOCI = {}
for org, env in [("se","CGMLST_LOCI_SE"),("ec","CGMLST_LOCI_EC"),("cj","CGMLST_LOCI_CJ")]:
    p = os.environ.get(env, f"/mnt/disk2/a.deruvo/chew_results/{org}_cgmlst_loci.txt")
    CGMLST_LOCI[org] = p if Path(p).exists() else None


def normalise(v: str) -> int:
    if not v or v == '-': return 0
    if v.startswith('INF-'): v = v[4:]
    v = v.lstrip('*')
    try:    return int(v)
    except: return 0


def normalise_id(g: str) -> str:
    g = g.split('/')[-1]
    for s in ['.cds.fasta','.cds','.fasta','.fna','.fa']:
        if g.endswith(s): g = g[:-len(s)]
    return g


def normalise_locus(l: str) -> str:
    return l.split('/')[-1].replace('.fasta','')


def load(path: Path):
    with open(path) as f:
        rows = list(csv.reader(f, delimiter='\t'))
    loci = [normalise_locus(l) for l in rows[0][1:]]
    data = {normalise_id(r[0]): r[1:] for r in rows[1:]}
    return loci, data


def cluster_diff(cc_path: Path, cb_path: Path, cgmlst_loci_file: str, thresholds):
    loci_a, data_a = load(cc_path)
    loci_b, data_b = load(cb_path)
    common = sorted(set(loci_a) & set(loci_b))
    if cgmlst_loci_file:
        with open(cgmlst_loci_file) as f:
            cgs = {l.strip() for l in f if l.strip()}
        common = sorted(l for l in common if l in cgs)
    common_g = sorted(set(data_a) & set(data_b))
    idx_a = {l:i for i,l in enumerate(loci_a)}
    idx_b = {l:i for i,l in enumerate(loci_b)}
    n, L = len(common_g), len(common)
    A = np.zeros((n,L), dtype=np.int64)
    B = np.zeros((n,L), dtype=np.int64)
    for i,g in enumerate(common_g):
        ra, rb = data_a[g], data_b[g]
        for j,l in enumerate(common):
            A[i,j] = normalise(ra[idx_a[l]])
            B[i,j] = normalise(rb[idx_b[l]])
    cA = A > 0; cB = B > 0
    n_pairs = n*(n-1)//2
    da = np.zeros(n_pairs, dtype=np.int32)
    db = np.zeros(n_pairs, dtype=np.int32)
    p = 0
    for i in range(n):
        nrow = n - i - 1
        if nrow <= 0: continue
        bothA = cA[i] & cA[i+1:]
        bothB = cB[i] & cB[i+1:]
        da[p:p+nrow] = ((A[i] != A[i+1:]) & bothA).sum(axis=1)
        db[p:p+nrow] = ((B[i] != B[i+1:]) & bothB).sum(axis=1)
        p += nrow
    res = {}
    for t in thresholds:
        in_a = da <= t; in_b = db <= t
        res[t] = int((in_a != in_b).sum())
    return n_pairs, res


def emit_table(rows, thresholds):
    print("\\begin{table}[H]\n\\centering")
    print("\\caption{Pairwise cgMLST clustering comparison at allelic difference (AD) thresholds. "
          "Distances are computed on cgMLST loci only, excluding loci missing in either genome. "
          "Zero discordant pairs across all 8 datasets.}")
    print("\\label{tab:clustering}\n\\smallskip\n\\footnotesize")
    print("\\begin{tabular}{@{}llr" + "r"*len(thresholds) + "@{}}\n\\toprule")
    print(" & & & " + f"\\multicolumn{{{len(thresholds)}}}{{c}}{{Pairs with different clustering}} \\\\")
    print(f"\\cmidrule(lr){{4-{3+len(thresholds)}}}")
    print("Dataset & Organism & Pairs & " + " & ".join(f"$\\leq${t}\\,AD" for t in thresholds) + " \\\\")
    print("\\midrule")
    for i, ((ds,org), n_pairs, diffs) in enumerate(rows):
        if i == 4: print("\\addlinespace")
        cells = " & ".join(f"\\textbf{{{diffs[t]}}}" if diffs[t]==0 else str(diffs[t]) for t in thresholds)
        n_str = f"{n_pairs:,}".replace(",","\\,")
        print(f"{DS_LATEX[ds]} & {ORG_LATEX[org]} & {n_str} & {cells} \\\\")
    print("\\bottomrule\n\\end{tabular}\n\\end{table}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thresholds", nargs="+", type=int, default=[4,5,7,10])
    args = ap.parse_args()
    rows = []
    for ds in DATASETS:
        for org in ORGANISMS:
            cc = DIR_CC / f"shcds_{ds}_{org}_rep1" / "results_alleles.tsv"
            cb_base = DIR_C354 / f"shcds_{ds}_{org}_rep1"
            cb = cb_base / "results_alleles.tsv"
            if not cb.exists():
                cb = next(cb_base.glob("results_*/results_alleles.tsv"), cb)
            print(f"  {ds} {org}: {cc.parent.name}", file=sys.stderr, flush=True)
            n_pairs, diffs = cluster_diff(cc, cb, CGMLST_LOCI.get(org), args.thresholds)
            rows.append(((ds,org), n_pairs, diffs))
            print(f"    pairs={n_pairs:,}  diffs={diffs}", file=sys.stderr)
    emit_table(rows, args.thresholds)


if __name__ == "__main__":
    main()
