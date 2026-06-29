#!/usr/bin/env python3
"""09_table_cgmlst_split.py — Table 5 (tab:cgmlst_split).

Reports actionable differences (cells where one tool is callable and the other
is not) per organism × dataset × {cgMLST, wgMLST-only}, plus the per-cell rate.
This is the same data as Supplementary Table S4 but presented in the
main-text format with `Diffs` and `%` columns, partitioned by locus class.

Reads chewcall outputs from $OUT_CHEWCALL/shcds_{cons,pub}_{org}_rep1/
and chewBBACA v3.5.4 references from $OUT_C354/shcds_{cons,pub}_{org}_rep1/.

Usage: python 09_table_cgmlst_split.py
"""
import csv, os, sys
from pathlib import Path

BASE      = Path(os.environ.get("OUT_ROOT", "results"))
DIR_CC    = Path(os.environ.get("OUT_CHEWCALL", str(BASE / "cc_fixed2")))
DIR_C354  = Path(os.environ.get("OUT_C354",     str(BASE / "c354")))

ORGS = ["lm","se","ec","cj"]
DATASETS = ["cons","pub"]
DS_HEAD = {"cons":"Cons.", "pub":"Pub."}
ORG_TEX = {"lm": r"\textit{L.\,mono.}", "se": r"\textit{S.\,enter.}",
           "ec": r"\textit{E.\,coli}",  "cj": r"\textit{C.\,jejuni}"}
N_LOCI_CG = {"lm":1748, "se":3254, "ec":2360, "cj":678}
N_LOCI_WG = {"lm":None, "se":5304, "ec":5241, "cj":2116}

CGMLST = {"lm": None}
for org, env in [("se","CGMLST_LOCI_SE"),("ec","CGMLST_LOCI_EC"),("cj","CGMLST_LOCI_CJ")]:
    p = os.environ.get(env, f"cgmlst_loci/{org}_cgmlst_loci.txt")
    if Path(p).exists():
        with open(p) as f: CGMLST[org] = {l.strip() for l in f if l.strip()}
    else: CGMLST[org] = set()


def is_callable(v: str) -> bool:
    if not v: return False
    if v.startswith("INF-") or v.startswith("*"): return True
    return v.lstrip("-+").isdigit()


def load(p: Path):
    with open(p) as f:
        rows = list(csv.reader(f, delimiter='\t'))
    loci = rows[0][1:]
    data = {}
    for r in rows[1:]:
        g = r[0]
        for s in [".cds.fasta",".cds.fna",".cds",".fasta",".fa"]:
            if g.endswith(s): g = g[:-len(s)]; break
        data[g] = r[1:]
    return data, loci


def actionable(cc_path: Path, cb_path: Path, cgmlst_set):
    cc, cc_l = load(cc_path); cb, cb_l = load(cb_path)
    cc_idx = {l:i for i,l in enumerate(cc_l)}
    cb_idx = {l:i for i,l in enumerate(cb_l)}
    common_g = sorted(set(cc) & set(cb))
    common_l = [l for l in cc_l if l in cb_idx]
    cg_cells=cg_diffs=wg_cells=wg_diffs=0
    for g in common_g:
        for l in common_l:
            cc_v = cc[g][cc_idx[l]]; cb_v = cb[g][cb_idx[l]]
            cc_call = is_callable(cc_v); cb_call = is_callable(cb_v)
            is_cg = cgmlst_set is None or l in cgmlst_set
            if is_cg: cg_cells += 1
            else:     wg_cells += 1
            if cc_call != cb_call:
                if is_cg: cg_diffs += 1
                else:     wg_diffs += 1
    return cg_cells, cg_diffs, wg_cells, wg_diffs


def fmt(v): return "0" if v == 0 else f"{v:,}".replace(",","\\,")


def main():
    print("\\begin{table}[H]\n\\centering")
    print("\\caption{Actionable differences partitioned by cgMLST versus wgMLST-only loci. "
          "A cell is \\emph{actionable} when one tool assigns a callable allele (EXC or INF) "
          "and the other returns a non-callable label (LNF, ASM, ALM, PLOT3/PLOT5, LOTSC, PAMA, NIPH or NIPHEM); "
          "when both tools call, the underlying CRC32 hashes are identical (Table~\\ref{tab:concordance}). "
          "For \\textit{L.\\,monocytogenes}, all loci are cgMLST; the wgMLST-only columns are therefore empty.}")
    print("\\label{tab:cgmlst_split}\n\\smallskip\n\\footnotesize")
    print("\\resizebox{\\textwidth}{!}{%")
    print("\\begin{tabular}{@{}llrrrrrrrr@{}}\n\\toprule")
    print("& & \\multicolumn{4}{c}{cgMLST} & \\multicolumn{4}{c}{wgMLST-only} \\\\")
    print("\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}")
    print("Dataset & Organism & Loci & Cells & Diffs & \\% & Loci & Cells & Diffs & \\% \\\\")
    print("\\midrule")
    for i, (ds, org) in enumerate([(d,o) for d in DATASETS for o in ORGS]):
        if i == 4: print("\\addlinespace")
        cc = DIR_CC / f"shcds_{ds}_{org}_rep1" / "results_alleles.tsv"
        cb_base = DIR_C354 / f"shcds_{ds}_{org}_rep1"
        cb = cb_base / "results_alleles.tsv"
        if not cb.exists():
            cb = next(cb_base.glob("results_*/results_alleles.tsv"), cb)
        if not cc.exists() or not cb.exists():
            print(f"{DS_HEAD[ds]} & {ORG_TEX[org]} & \\multicolumn{{8}}{{c}}{{(inputs not staged)}} \\\\")
            continue
        cg_cells, cg_d, wg_cells, wg_d = actionable(cc, cb, CGMLST[org])
        cg_pct = 100*cg_d/cg_cells if cg_cells else 0
        cg_pct_str = f"{cg_pct:.3f}".rstrip("0").rstrip(".") if cg_pct < 0.001 and cg_pct > 0 else f"{cg_pct:.3f}"
        if cg_pct < 0.001 and cg_pct > 0: cg_pct_str = "$<$0.001"
        elif cg_pct == 0: cg_pct_str = "0.000"
        if wg_cells:
            wg_pct = 100*wg_d/wg_cells if wg_cells else 0
            wg_pct_str = f"{wg_pct:.3f}"
            wg_loci_str = fmt(N_LOCI_WG[org])
            wg_cells_str = fmt(wg_cells); wg_d_str = fmt(wg_d)
        else:
            wg_loci_str = wg_cells_str = wg_d_str = wg_pct_str = "---"
        print(f"{DS_HEAD[ds]} & {ORG_TEX[org]} & {fmt(N_LOCI_CG[org])} & {fmt(cg_cells)} & {fmt(cg_d)} & {cg_pct_str} & "
              f"{wg_loci_str} & {wg_cells_str} & {wg_d_str} & {wg_pct_str} \\\\")
    print("\\bottomrule\n\\end{tabular}}\n\\end{table}")


if __name__ == "__main__":
    main()
