#!/usr/bin/env python3
"""06_figure2_discordance_anatomy.py — Figure 2 (fig:discordance_anatomy).

Computes the 6-category discordance breakdown per dataset x cgMLST/wgMLST,
and emits the LaTeX/colortbl tabular block used by Figure 2.

Categories (chewBBACA = cb, chewcall = cc):
  cb_real      : cb callable (EXC/INF), cc returns LNF/ASM/ALM/PLOT/LOTSC/PAMA
  cb_paralog   : cb callable, cc returns NIPH/NIPHEM
  cc_real      : cc callable, cb returns LNF/ASM/ALM/PLOT/LOTSC/PAMA
  cc_paralog   : cc callable, cb returns NIPH/NIPHEM
  paralog_both : neither callable, at least one is NIPH/NIPHEM
  other        : neither callable, neither NIPH/NIPHEM

Reads chewcall outputs from $OUT_CHEWCALL (shcds mode) and v3.5.4 reference
from $OUT_C354 (shcds mode), and the cgMLST locus lists from
$CGMLST_LOCI_{SE,EC,CJ}.

Usage:  python 06_figure2_discordance_anatomy.py
        python 06_figure2_discordance_anatomy.py --raw     # dump per-org dicts
"""
import argparse, csv, math, os, sys
from collections import defaultdict
from pathlib import Path

BASE      = Path(os.environ.get("OUT_ROOT", "/mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results"))
DIR_CC    = Path(os.environ.get("OUT_CHEWCALL", str(BASE / "cc_fixed2")))
DIR_C354  = Path(os.environ.get("OUT_C354",     str(BASE / "c354")))

ORGANISMS = ["lm","se","ec","cj"]
DATASETS  = ["cons","pub"]
CGMLST = {"lm": None}
for org, env in [("se","CGMLST_LOCI_SE"),("ec","CGMLST_LOCI_EC"),("cj","CGMLST_LOCI_CJ")]:
    p = os.environ.get(env, f"/mnt/disk2/a.deruvo/chew_results/{org}_cgmlst_loci.txt")
    if Path(p).exists():
        with open(p) as f: CGMLST[org] = {l.strip() for l in f if l.strip()}
    else: CGMLST[org] = set()

CATS = ["cb_real","cb_paralog","cc_real","cc_paralog","paralog_both","other"]
CATS_LATEX = ['cCBreal','cCBpar','cCCreal','cCCpar','cParBoth','cOther']


def classify(v: str) -> str:
    if not v: return "LNF"
    if v.startswith("INF-") or v.startswith("*"): return "INF"
    if v.isdigit(): return "EXC"
    if v.startswith("NIPHEM"): return "NIPHEM"
    if v.startswith("NIPH"):   return "NIPH"
    return v


def category(cb: str, cc: str) -> str:
    cb_call = cb in ("EXC","INF"); cc_call = cc in ("EXC","INF")
    cb_par  = cb in ("NIPH","NIPHEM"); cc_par = cc in ("NIPH","NIPHEM")
    if cb_call and not cc_call: return "cb_paralog" if cc_par else "cb_real"
    if cc_call and not cb_call: return "cc_paralog" if cb_par else "cc_real"
    if cb_par or cc_par:        return "paralog_both"
    return "other"


def load_alleles(path: Path):
    with open(path) as f:
        rows = list(csv.reader(f, delimiter='\t'))
    loci = rows[0][1:]
    data = {}
    for r in rows[1:]:
        g = r[0]
        for s in [".cds.fasta",".cds.fna",".cds",".fasta",".fa"]:
            if g.endswith(s): g = g[:-len(s)]; break
        data[g] = r[1:]
    return data, loci


def fmt(v: int) -> str:
    return "0" if v == 0 else f"{v:,}".replace(",", "\\,")


def per_dataset(ds: str, org: str):
    cc = DIR_CC / f"shcds_{ds}_{org}_rep1" / "results_alleles.tsv"
    cb_base = DIR_C354 / f"shcds_{ds}_{org}_rep1"
    cb = cb_base / "results_alleles.tsv"
    if not cb.exists():
        cb = next(cb_base.glob("results_*/results_alleles.tsv"), cb)
    cgs = CGMLST[org]
    cc_d, cc_l = load_alleles(cc); cb_d, cb_l = load_alleles(cb)
    common_l = [l for l in cc_l if l in set(cb_l)]
    cc_idx = {l:i for i,l in enumerate(cc_l)}
    cb_idx = {l:i for i,l in enumerate(cb_l)}
    cg = {c:0 for c in CATS}; wg = {c:0 for c in CATS}
    for g in sorted(set(cc_d) & set(cb_d)):
        for l in common_l:
            cc_c = classify(cc_d[g][cc_idx[l]])
            cb_c = classify(cb_d[g][cb_idx[l]])
            if cc_c == cb_c: continue
            bucket = category(cb_c, cc_c)
            tgt = cg if (cgs is None or l in cgs) else wg
            tgt[bucket] += 1
    return cg, wg


def emit_latex(per):
    rows = []
    for ds in ("cons","pub"):
        for org in ("lm","se","ec","cj"):
            cg, wg = per[(ds,org)]
            rows.append((f"{org.capitalize()} {ds}.", "cg", cg))
            if any(wg.values()):
                rows.append((f"{org.capitalize()} {ds}.", "wg", wg))
    mx = [max(r[2][c] for r in rows) or 1 for c in CATS]
    print("\\begin{figure}[H]\n\\centering")
    print("\\definecolor{cCBreal}{RGB}{192,57,43}")
    print("\\definecolor{cCBpar}{RGB}{245,176,65}")
    print("\\definecolor{cCCreal}{RGB}{39,174,96}")
    print("\\definecolor{cCCpar}{RGB}{125,206,160}")
    print("\\definecolor{cParBoth}{RGB}{155,89,182}")
    print("\\definecolor{cOther}{RGB}{189,195,199}")
    print("\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\renewcommand{\\arraystretch}{1.05}")
    print("\\begin{tabular}{@{}llcccccc|r@{}}\n\\toprule")
    print(" & & \\multicolumn{2}{c}{cb finds, cc does not} & \\multicolumn{2}{c}{cc finds, cb does not} & & & \\\\")
    print("\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}")
    print("Dataset & Locus & \\cellcolor{cCBreal!50}real & \\cellcolor{cCBpar!50}NIPH/NIPHEM & "
          "\\cellcolor{cCCreal!50}real & \\cellcolor{cCCpar!50}NIPH/NIPHEM & "
          "\\cellcolor{cParBoth!50}Both paralog & \\cellcolor{cOther!50}Other & Total \\\\")
    print("\\midrule")
    for i, (ds, lt, vals) in enumerate(rows):
        if i > 0 and ds == "Lm pub.":
            print("\\addlinespace")
        total = sum(vals[c] for c in CATS)
        cells = []
        for ci, c in enumerate(CATS):
            v = vals[c]
            if v == 0: cells.append("\\cellcolor{white}0")
            else:
                inten = max(5, min(70, int(round(70*math.sqrt(v/mx[ci])))))
                pct = 100*v/total if total else 0
                pct_str = f"{pct:.1f}\\%" if pct >= 0.05 else "<0.1\\%"
                cells.append(f"\\cellcolor{{{CATS_LATEX[ci]}!{inten}}}"
                             f"\\begin{{tabular}}[c]{{@{{}}r@{{}}}}{fmt(v)}\\\\[-1pt]"
                             f"{{\\tiny {pct_str}}}\\end{{tabular}}")
        print(f"{ds} & {lt} & " + " & ".join(cells) + f" & \\textbf{{{fmt(total)}}} \\\\")
    print("\\bottomrule\n\\end{tabular}")
    print("\\caption{Classification discordance anatomy across 8 BeONE datasets "
          "(chewcall vs chewBBACA~v3.5.4, shared pre-computed CDS), split by cgMLST (cg) and wgMLST-only (wg) loci. "
          "Each cell shows the absolute count and the row-relative percentage; "
          "cell colour intensity is proportional to the per-category square-root of the count "
          "(column-normalised). For \\textit{L.\\,monocytogenes}, all loci are cgMLST.}")
    print("\\label{fig:discordance_anatomy}\n\\end{figure}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", action="store_true", help="Dump per-org breakdown to stderr")
    args = ap.parse_args()
    per = {}
    for ds in DATASETS:
        for org in ORGANISMS:
            cg, wg = per_dataset(ds, org)
            per[(ds,org)] = (cg, wg)
            if args.raw:
                print(f"{ds}_{org}: cg={cg}, wg={wg}", file=sys.stderr)
    emit_latex(per)


if __name__ == "__main__":
    main()
