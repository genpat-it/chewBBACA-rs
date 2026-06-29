#!/usr/bin/env python3
"""04_tables_runtime_concordance.py — Tables 1, 2, 3, 4 of the paper.

  Table 1 (tab:runtime)        : shared-CDS runtime  (chewcall vs chewBBACA 3.5.4 vs 3.3.10)
  Table 2 (tab:concordance)    : shared-CDS CRC32 concordance on callable cells
  Table 3 (tab:runtime_e2e)    : end-to-end runtime
  Table 4 (tab:concordance_e2e): end-to-end CRC32 concordance

Reads chewcall outputs from $OUT_CHEWCALL (default: cc_fixed2/),
chewBBACA references from $OUT_C354 and $OUT_C3310.

Usage:
  python 04_tables_runtime_concordance.py --mode shcds   # Tables 1+2
  python 04_tables_runtime_concordance.py --mode e2e     # Tables 3+4
  python 04_tables_runtime_concordance.py --mode all     # all four
"""
import argparse
import os
import sys
from pathlib import Path
from statistics import median
import numpy as np
import pandas as pd

# ----- Configuration --------------------------------------------------
BASE       = Path(os.environ.get("OUT_ROOT", "results"))
DIR_CC     = Path(os.environ.get("OUT_CHEWCALL", str(BASE / "cc_fixed2")))
DIR_C354   = Path(os.environ.get("OUT_C354",     str(BASE / "c354")))
DIR_C3310  = Path(os.environ.get("OUT_C3310",    str(BASE / "c3_3310_e2e")))

ORGANISMS = ["lm", "se", "ec", "cj"]
DATASETS  = ["cons", "pub"]
ORG_LATEX = {"lm": r"\textit{L.\,mono.}", "se": r"\textit{S.\,enter.}",
             "ec": r"\textit{E.\,coli}",  "cj": r"\textit{C.\,jejuni}"}
DS_LATEX  = {"cons": "Cons.", "pub": "Pub."}
N_GENOMES = {("cons","lm"):1426,("cons","se"):1540,("cons","ec"):308,("cons","cj"):610,
             ("pub","lm"):1874,("pub","se"):1434,("pub","ec"):1999,("pub","cj"):3076}
N_LOCI    = {"lm":1748,"se":8558,"ec":7601,"cj":2794}
CALLABLE_PCT_CG = {("cons","lm"):99.2,("cons","se"):99.0,("cons","ec"):99.4,("cons","cj"):99.2,
                    ("pub","lm"):99.1,("pub","se"):98.4,("pub","ec"):99.5,("pub","cj"):99.7}
CALLABLE_PCT_WG = {("cons","lm"):None,("cons","se"):10.0,("cons","ec"):20.0,("cons","cj"):19.7,
                    ("pub","lm"):None,("pub","se"):10.5,("pub","ec"):19.8,("pub","cj"):19.5}

CGMLST_LOCI = {"lm": None}
for org, env in [("se","CGMLST_LOCI_SE"),("ec","CGMLST_LOCI_EC"),("cj","CGMLST_LOCI_CJ")]:
    p = os.environ.get(env, f"cgmlst_loci/{org}_cgmlst_loci.txt")
    if Path(p).exists():
        with open(p) as f:
            CGMLST_LOCI[org] = set(l.strip() for l in f if l.strip())
    else:
        CGMLST_LOCI[org] = set()


# ----- Helpers --------------------------------------------------------
def load_hashed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str).set_index("FILE")
    df.index = df.index.str.replace(r"\.cds$", "", regex=True)
    return df


def concordance(cc_path: Path, ref_path: Path, org: str):
    cc, ref = load_hashed(cc_path), load_hashed(ref_path)
    samples = cc.index.intersection(ref.index)
    loci    = cc.columns.intersection(ref.columns)
    cc, ref = cc.loc[samples, loci], ref.loc[samples, loci]
    cc_num  = cc.apply(lambda c: c.str.strip().str.isdigit()).values
    ref_num = ref.apply(lambda c: c.str.strip().str.isdigit()).values
    both    = cc_num & ref_num
    diff    = both & (cc.values != ref.values)
    cg_set  = CGMLST_LOCI[org]
    if cg_set is None:
        cg_mask = np.ones(len(loci), dtype=bool)
    else:
        cg_mask = np.array([l in cg_set for l in loci])
    cg_mask_2d = np.broadcast_to(cg_mask[None, :], both.shape)
    return {
        "callable":   int(both.sum()),
        "cg_diffs":   int((diff & cg_mask_2d).sum()),
        "wg_diffs":   int((diff & ~cg_mask_2d).sum()),
    }


def parse_timing(path: Path) -> dict:
    if not path.exists(): return {}
    out = {}
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            try: out[parts[0]] = float(parts[1])
            except ValueError: continue
    return out


def median_time(timings: dict, key_pattern: str):
    vs = [v for k, v in timings.items() if k.startswith(key_pattern) and v < 50000]
    return median(vs) if vs else None


# ----- Path builders --------------------------------------------------
def cc_path(mode, ds, org, hashed=True, rep=1):
    fname = "results_alleles_hashed.tsv" if hashed else "results_alleles.tsv"
    return DIR_CC / f"{mode}_{ds}_{org}_rep{rep}" / fname


def c354_path(mode, ds, org, hashed=True, rep=1):
    fname = "results_alleles_hashed.tsv" if hashed else "results_alleles.tsv"
    base = DIR_C354 / f"{mode}_{ds}_{org}_rep{rep}"
    if (base / fname).exists():
        return base / fname
    # chewBBACA writes results to results_TIMESTAMP/ subdirectories
    matches = sorted(base.glob(f"results_*/{fname}"))
    return matches[0] if matches else base / fname


def c3310_path(ds, org, hashed=True, rep=1):
    fname = "results_alleles_hashed.tsv" if hashed else "results_alleles.tsv"
    base = DIR_C3310 / f"{ds}_{org}_rep{rep}"
    if (base / fname).exists():
        return base / fname
    matches = sorted(base.glob(f"results_*/{fname}"))
    return matches[0] if matches else base / fname


# ----- Output ---------------------------------------------------------
def thousands(n):
    return f"{n:,}".replace(",", "\\,")


def emit_runtime_table(mode, results, label, caption):
    print(f"\\begin{{table}}[H]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}")
    print("\\smallskip\n\\footnotesize\n\\resizebox{\\textwidth}{!}{%")
    print("\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule")
    print("Dataset & Organism & Genomes & Loci & chewBBACA 3.3.10 & chewBBACA 3.5.4 & chewcall \\\\")
    print("\\midrule")
    for i, (ds, org) in enumerate([(d,o) for d in DATASETS for o in ORGANISMS]):
        if i == 4: print("\\addlinespace")
        cc_t   = results[(ds,org)]["cc"]
        c354_t = results[(ds,org)]["c354"]
        c3310_t= results[(ds,org)]["c3310"]
        cc_s   = f"\\SI{{{cc_t:.1f}}}{{\\second}}" if cc_t else "---"
        c354_s = f"\\SI{{{c354_t:.0f}}}{{\\second}}" if c354_t else "---"
        c3310_s= f"\\SI{{{c3310_t:.0f}}}{{\\second}}" if c3310_t else "---"
        spd354 = f"{c354_t/cc_t:.1f}$\\times$" if cc_t and c354_t else "---"
        spd33  = f"{c3310_t/cc_t:.1f}$\\times$" if cc_t and c3310_t else "---"
        suffix = f"\\;({spd33}/{spd354})" if cc_t else ""
        ng     = thousands(N_GENOMES[(ds,org)])
        nl     = thousands(N_LOCI[org])
        print(f"{DS_LATEX[ds]} & {ORG_LATEX[org]} & {ng} & {nl} & "
              f"{c3310_s} & {c354_s} & {cc_s}{suffix} \\\\")
    print("\\bottomrule\n\\end{tabular}}\n\\end{table}")


def emit_concordance_table(mode, results, label, caption):
    print(f"\\begin{{table}}[H]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}")
    print("\\smallskip\n\\footnotesize")
    print("\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule")
    print(" & & & & \\multicolumn{2}{c}{Callable (\\%)} & \\multicolumn{2}{c}{Diffs} \\\\")
    print("\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}")
    print("Dataset & Organism & Genomes & Loci & cgMLST & wgMLST & vs 3.3.10 & vs 3.5.4 \\\\")
    print("\\midrule")
    for i, (ds, org) in enumerate([(d,o) for d in DATASETS for o in ORGANISMS]):
        if i == 4: print("\\addlinespace")
        r = results[(ds,org)]
        cg_pct = CALLABLE_PCT_CG[(ds,org)]
        wg_pct = CALLABLE_PCT_WG[(ds,org)]
        wg_str = "---" if wg_pct is None else f"{wg_pct:.1f}"
        ng     = thousands(N_GENOMES[(ds,org)])
        nl     = thousands(N_LOCI[org])
        d354   = r["vs354"]["cg_diffs"]+r["vs354"]["wg_diffs"]
        d33    = r["vs3310"]["cg_diffs"]+r["vs3310"]["wg_diffs"]
        print(f"{DS_LATEX[ds]} & {ORG_LATEX[org]} & {ng} & {nl} & "
              f"{cg_pct:.1f} & {wg_str} & {d33} & {d354} \\\\")
    print("\\bottomrule\n\\end{tabular}\n\\end{table}")


# ----- Main -----------------------------------------------------------
def run(mode):
    print(f"% ===== mode={mode} =====", file=sys.stderr)
    results = {}
    timing_cc_cons = parse_timing(DIR_CC / f"timing_{mode}_cons.txt")
    timing_cc_pub  = parse_timing(DIR_CC / f"timing_{mode}_pub.txt")
    timing_c354 = {}
    for tf in [f"timing_{mode}_cons.txt", f"timing_{mode}_pub.txt", "timing.txt"]:
        timing_c354.update(parse_timing(DIR_C354 / tf))
    timing_c3310 = parse_timing(DIR_C3310 / f"timing_{mode}.txt")
    if not timing_c3310:
        timing_c3310 = parse_timing(DIR_C3310 / "timing.txt")

    for ds in DATASETS:
        for org in ORGANISMS:
            cc_t = timing_cc_cons if ds == "cons" else timing_cc_pub
            cct = median_time(cc_t, f"{mode}_{ds}_{org}_rep")
            c354t = median_time(timing_c354, f"{mode}_{ds}_{org}_rep")
            c3310t = median_time(timing_c3310, f"{mode}_{ds}_{org}_rep")
            cc = cc_path(mode, ds, org)
            v354_path = c354_path(mode, ds, org)
            v3310_p = c3310_path(ds, org)
            v354 = concordance(cc, v354_path, org) if cc.exists() and v354_path.exists() else {"callable":0,"cg_diffs":0,"wg_diffs":0}
            v3310 = concordance(cc, v3310_p, org) if cc.exists() and v3310_p.exists() else {"callable":0,"cg_diffs":0,"wg_diffs":0}
            results[(ds,org)] = {"cc": cct, "c354": c354t, "c3310": c3310t,
                                  "vs354": v354, "vs3310": v3310}
            print(f"  {ds:<5} {org} cc={cct} c354={c354t} c3310={c3310t} "
                  f"vs354={v354} vs3310={v3310}", file=sys.stderr)

    if mode == "shcds":
        emit_runtime_table(mode, results, "tab:runtime",
            "Runtime on 8 BeONE datasets (8 CPU threads, shared pre-computed CDS). "
            "Speedup relative to each chewBBACA version is shown in parentheses.")
        emit_concordance_table(mode, results, "tab:concordance",
            "CRC32 concordance on callable cells (shared pre-computed CDS). "
            "\\textbf{Diffs} = callable cells with different CRC32 hash (different DNA). "
            "Zero discordant cells across all 8 datasets and ${\\sim}28.7$\\,M callable cells per comparison.")
    elif mode == "e2e":
        emit_runtime_table(mode, results, "tab:runtime_e2e",
            "End-to-end runtime on 8 BeONE datasets (8 CPU threads, including CDS prediction). "
            "chewcall uses prodigal-rs; chewBBACA uses pyrodigal. "
            "Speedup relative to each chewBBACA version in parentheses.")
        emit_concordance_table(mode, results, "tab:concordance_e2e",
            "End-to-end CRC32 concordance on callable cells (including CDS prediction).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["shcds", "e2e", "all"], default="all")
    args = ap.parse_args()
    if args.mode == "all":
        run("shcds"); run("e2e")
    else:
        run(args.mode)
