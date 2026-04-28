#!/usr/bin/env python3
"""07_supp_S3_classification_transitions.py — Supplementary Table S3.

Per-organism classification transition matrix (chewBBACA v3.5.4 -> chewcall)
on the 8 BeONE datasets, shared pre-computed CDS.  Emits the unified
LaTeX block (Cons | Pub side-by-side) used by Supplementary Table S3.

Also computes Supplementary Table S4 (actionable cells per organism, split
by cgMLST/wgMLST), printed after Table S3.

Usage: python 07_supp_S3_classification_transitions.py
"""
import csv, os, sys
from collections import Counter
from pathlib import Path

BASE      = Path(os.environ.get("OUT_ROOT", "/mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results"))
DIR_CC    = Path(os.environ.get("OUT_CHEWCALL", str(BASE / "cc_fixed2")))
DIR_C354  = Path(os.environ.get("OUT_C354",     str(BASE / "c354")))

ORGS = ["lm","se","ec","cj"]
DATASETS = ["cons","pub"]
DS_HEAD = {"cons":"Cons.","pub":"Pub."}
ORG_TEX = {"lm": r"\textit{L.\,mono.}", "se": r"\textit{S.\,enter.}",
           "ec": r"\textit{E.\,coli}",  "cj": r"\textit{C.\,jejuni}"}

CGMLST = {"lm": None}
for org, env in [("se","CGMLST_LOCI_SE"),("ec","CGMLST_LOCI_EC"),("cj","CGMLST_LOCI_CJ")]:
    p = os.environ.get(env, f"/mnt/disk2/a.deruvo/chew_results/{org}_cgmlst_loci.txt")
    if Path(p).exists():
        with open(p) as f: CGMLST[org] = {l.strip() for l in f if l.strip()}
    else: CGMLST[org] = set()


def classify(v: str) -> str:
    v = v.strip()
    if not v: return "LNF"
    if v.startswith("INF-") or v.startswith("*"): return "INF"
    if v.startswith("NIPHEM"): return "NIPHEM"
    if v.startswith("NIPH"):   return "NIPH"
    if v in ("LNF","ASM","ALM","PLOT3","PLOT5","LOTSC","PAMA"): return v
    if v.lstrip("-+").isdigit(): return "EXC"
    return v


def is_callable(c: str) -> bool: return c in ("EXC","INF")


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


def transitions_actionable(cc_path, cb_path, cgmlst_set):
    cc, cc_l = load_alleles(cc_path); cb, cb_l = load_alleles(cb_path)
    cc_idx = {l:i for i,l in enumerate(cc_l)}
    cb_idx = {l:i for i,l in enumerate(cb_l)}
    common_g = sorted(set(cc) & set(cb))
    common_l = [l for l in cc_l if l in cb_idx]
    trans = Counter()
    cg_total=cg_cc=cg_cb=0; wg_total=wg_cc=wg_cb=0
    for g in common_g:
        for l in common_l:
            cv = classify(cc[g][cc_idx[l]]); bv = classify(cb[g][cb_idx[l]])
            is_cg = cgmlst_set is None or l in cgmlst_set
            if is_cg: cg_total += 1
            else:     wg_total += 1
            if cv == bv: continue
            trans[(bv,cv)] += 1
            if is_callable(cv) and not is_callable(bv):
                if is_cg: cg_cc += 1
                else:     wg_cc += 1
            elif is_callable(bv) and not is_callable(cv):
                if is_cg: cg_cb += 1
                else:     wg_cb += 1
    return trans, (cg_total,cg_cc,cg_cb,wg_total,wg_cc,wg_cb)


def fmt(v): return "0" if v == 0 else f"{v:,}".replace(",","\\,")


def emit_S3(all_trans):
    print("% Supplementary Table S3 — classification transitions")
    print("\\begin{table}[ht]\n\\centering")
    print("\\caption*{\\hypertarget{tab-S3}{Table~S3.} Classification transitions in discordant cells per organism (chewBBACA~v3.5.4$\\to$chewcall, shared pre-computed CDS). C = consortium; P = public.}")
    print("\\smallskip\n\\footnotesize")
    print("\\begin{tabular}{@{}llrrrrrrrr@{}}\n\\toprule")
    print(" & & \\multicolumn{4}{c}{Consortium} & \\multicolumn{4}{c}{Public} \\\\")
    print("\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}")
    print("chewBBACA & chewcall & " +
          " & ".join([r"\textit{L.m.}", r"\textit{S.e.}", r"\textit{E.c.}", r"\textit{C.j.}"]*2) + " \\\\")
    print("\\midrule")
    keys = [("Cons",o) for o in ORGS] + [("Pub",o) for o in ORGS]
    all_types = Counter()
    for k in keys:
        for tr,c in all_trans[k].items(): all_types[tr] += c
    top = [t for t,_ in all_types.most_common(15)]
    totals = {k: sum(all_trans[k].values()) for k in keys}
    for tr in top:
        cells = [fmt(all_trans[k].get(tr,0)) if all_trans[k].get(tr,0) else "--" for k in keys]
        print(f"{tr[0]} & {tr[1]} & " + " & ".join(cells) + " \\\\")
    others = [totals[k] - sum(all_trans[k].get(t,0) for t in top) for k in keys]
    if any(others):
        cells = [fmt(o) if o else "--" for o in others]
        print("Other & & " + " & ".join(cells) + " \\\\")
    print("\\addlinespace")
    print("\\textbf{Total} & & " + " & ".join(f"\\textbf{{{fmt(totals[k])}}}" for k in keys) + " \\\\")
    print("\\bottomrule\n\\end{tabular}\n\\end{table}")


def emit_S4(actionable):
    print("% Supplementary Table S4 — actionable differences per organism, cgMLST/wgMLST split")
    print("\\begin{table}[ht]\n\\centering")
    print("\\caption*{\\hypertarget{tab-S4}{Table~S4.} Actionable differences across all 8 BeONE datasets (chewcall vs chewBBACA~v3.5.4, shared pre-computed CDS), partitioned into cgMLST and wgMLST-only loci. ``cc'' = chewcall callable (EXC/INF) but chewBBACA not; ``cb'' = vice versa. For \\textit{L.\\,monocytogenes}, all loci are cgMLST.}")
    print("\\smallskip")
    print("\\begin{tabular}{@{}llrrrrrrrr@{}}\n\\toprule")
    print(" & & \\multicolumn{4}{c}{cgMLST} & \\multicolumn{4}{c}{wgMLST-only} \\\\")
    print("\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}")
    print("Dataset & Organism & Cells & cc & cb & \\% & Cells & cc & cb & \\% \\\\")
    print("\\midrule")
    for i, (ds, org) in enumerate([(d,o) for d in DATASETS for o in ORGS]):
        if i == 4: print("\\addlinespace")
        cg_total, cg_cc, cg_cb, wg_total, wg_cc, wg_cb = actionable[(ds,org)]
        cg_pct = 100*(cg_cc+cg_cb)/cg_total if cg_total else 0
        cg_pct_str = f"{cg_pct:.3f}".rstrip("0").rstrip(".") if cg_pct < 0.001 else f"{cg_pct:.3f}"
        if cg_pct < 0.001: cg_pct_str = "$<$0.001"
        if wg_total:
            wg_pct = 100*(wg_cc+wg_cb)/wg_total if wg_total else 0
            wg_cells = fmt(wg_total); wg_cc_s = fmt(wg_cc); wg_cb_s = fmt(wg_cb)
            wg_pct_str = f"{wg_pct:.3f}"
        else:
            wg_cells = wg_cc_s = wg_cb_s = wg_pct_str = "---"
        print(f"{DS_HEAD[ds]} & {ORG_TEX[org]} & {fmt(cg_total)} & {fmt(cg_cc)} & {fmt(cg_cb)} & {cg_pct_str} & "
              f"{wg_cells} & {wg_cc_s} & {wg_cb_s} & {wg_pct_str} \\\\")
    print("\\bottomrule\n\\end{tabular}\n\\end{table}")


def main():
    all_trans = {}; actionable = {}
    for ds in DATASETS:
        for org in ORGS:
            cc = DIR_CC / f"shcds_{ds}_{org}_rep1" / "results_alleles.tsv"
            cb_base = DIR_C354 / f"shcds_{ds}_{org}_rep1"
            cb = cb_base / "results_alleles.tsv"
            if not cb.exists():
                cb = next(cb_base.glob("results_*/results_alleles.tsv"), cb)
            print(f"  {ds:<5} {org}: ...", file=sys.stderr, flush=True)
            tr, ac = transitions_actionable(cc, cb, CGMLST[org])
            tag = "Cons" if ds == "cons" else "Pub"
            all_trans[(tag, org)] = tr
            actionable[(ds, org)] = ac
            print(f"    transitions={sum(tr.values()):,}  actionable_cg=({ac[1]},{ac[2]})  actionable_wg=({ac[4]},{ac[5]})",
                  file=sys.stderr)
    emit_S3(all_trans)
    print()
    emit_S4(actionable)


if __name__ == "__main__":
    main()
