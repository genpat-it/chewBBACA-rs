#!/usr/bin/env python3
"""Compute CRC32 concordance metrics for chewcall vs chewBBACA (3.3.10 and 3.5.4)
across all 8 BeONE datasets, plus runtime medians. Uses vectorized pandas."""

import pandas as pd
import numpy as np
from pathlib import Path
from statistics import median

BASE = Path("/mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results")

ORGANISMS = ["lm", "se", "ec", "cj"]
ORG_NAMES = {"lm": r"L.\ mono.", "se": r"S.\ enterica", "ec": r"E.\ coli", "cj": r"C.\ jejuni"}
ORG_NAMES_PLAIN = {"lm": "L. mono.", "se": "S. enterica", "ec": "E. coli", "cj": "C. jejuni"}
DATASETS = ["cons", "pub"]
DS_NAMES = {"cons": "Consortium", "pub": "Public"}

# --- cgMLST loci ---
CGMLST_LOCI = {}
CGMLST_LOCI["lm"] = None  # all loci are cgMLST

for org, path in [("se", "/mnt/disk2/a.deruvo/chew_results/se_cgmlst_loci.txt"),
                  ("ec", "/mnt/disk2/a.deruvo/chew_results/ec_cgmlst_loci.txt"),
                  ("cj", "/mnt/disk2/a.deruvo/chew_results/cj_cgmlst_loci.txt")]:
    with open(path) as f:
        CGMLST_LOCI[org] = set(line.strip() for line in f if line.strip())


def load_hashed(path):
    df = pd.read_csv(path, sep="\t", dtype=str).set_index("FILE")
    df.index = df.index.str.replace(r"\.cds$", "", regex=True)
    return df


def compute_concordance(cc_path, ref_path, org):
    """Vectorized concordance computation."""
    cc = load_hashed(cc_path)
    ref = load_hashed(ref_path)

    common_samples = cc.index.intersection(ref.index)
    common_loci = cc.columns.intersection(ref.columns)

    cc = cc.loc[common_samples, common_loci]
    ref = ref.loc[common_samples, common_loci]

    # Vectorized: check if each cell is numeric (callable)
    cc_vals = cc.values
    ref_vals = ref.values

    # Build boolean masks
    def is_numeric_array(arr):
        """Vectorized check for purely numeric strings."""
        mask = np.ones(arr.shape, dtype=bool)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if pd.isna(v) or not str(v).strip().isdigit():
                    mask[i, j] = False
        return mask

    # Actually, let's use a faster approach with pandas
    cc_numeric = cc.apply(lambda col: col.str.strip().str.isdigit(), axis=0).values
    ref_numeric = ref.apply(lambda col: col.str.strip().str.isdigit(), axis=0).values

    # Handle NaN -> False
    cc_numeric = np.where(pd.isna(cc.values), False, cc_numeric)
    ref_numeric = np.where(pd.isna(ref.values), False, ref_numeric)

    both_callable = cc_numeric & ref_numeric

    # Where both callable, check if different
    cc_str = np.where(both_callable, cc.values, "")
    ref_str = np.where(both_callable, ref.values, "")
    different = both_callable & (cc_str != ref_str)

    # Split by cgMLST vs wgMLST
    loci = common_loci.tolist()
    if CGMLST_LOCI[org] is None:
        cg_mask = np.ones(len(loci), dtype=bool)
    else:
        cg_mask = np.array([l in CGMLST_LOCI[org] for l in loci])

    # Broadcast locus mask across samples
    cg_mask_2d = np.broadcast_to(cg_mask[np.newaxis, :], both_callable.shape)
    wg_mask_2d = ~cg_mask_2d

    total_callable = int(both_callable.sum())
    cg_callable = int((both_callable & cg_mask_2d).sum())
    wg_callable = int((both_callable & wg_mask_2d).sum())
    cg_diffs = int((different & cg_mask_2d).sum())
    wg_diffs = int((different & wg_mask_2d).sum())
    total_diffs = cg_diffs + wg_diffs
    agreement_pct = (total_callable - total_diffs) / total_callable * 100 if total_callable > 0 else 0.0

    return {
        "n_samples": len(common_samples),
        "n_loci": len(common_loci),
        "total_callable": total_callable,
        "cg_callable": cg_callable,
        "wg_callable": wg_callable,
        "cg_diffs": cg_diffs,
        "wg_diffs": wg_diffs,
        "total_diffs": total_diffs,
        "agreement_pct": agreement_pct,
    }


# --- File paths ---
import os
MODE = os.environ.get("CC_MODE", "shcds")  # "shcds" or "e2e"

def cc_path(org, ds):
    ds_short = "cons" if ds == "cons" else "pub"
    return BASE / "cc_fixed2" / f"{MODE}_{ds_short}_{org}_rep1" / "results_alleles_hashed.tsv"

def c354_path(org, ds):
    ds_short = "cons" if ds == "cons" else "pub"
    return BASE / "c354" / f"{MODE}_{ds_short}_{org}_rep1" / "results_alleles_hashed.tsv"

def c3310_path(org, ds):
    p = BASE / "c3_3310_e2e" / f"{ds}_{org}_rep1" / "results_alleles_hashed.tsv"
    if not p.exists():
        p = BASE / "c3310_e2e_noinferred" / f"{ds}_{org}_rep1" / "results_alleles_hashed.tsv"
    return p


# --- Compute ---
print("=" * 120)
print("CRC32 CONCORDANCE: chewcall vs chewBBACA")
print("=" * 120)

results = []

for ds in DATASETS:
    for org in ORGANISMS:
        cc_file = cc_path(org, ds)

        for ver, ref_fn in [("vs 3.5.4", c354_path), ("vs 3.3.10", c3310_path)]:
            ref_file = ref_fn(org, ds)
            if cc_file.exists() and ref_file.exists():
                print(f"  Computing {ORG_NAMES_PLAIN[org]} {DS_NAMES[ds]} {ver}...", flush=True)
                m = compute_concordance(cc_file, ref_file, org)
                m["org"] = org
                m["ds"] = ds
                m["comparison"] = ver
                results.append(m)
                print(f"    callable={m['total_callable']:>9,d}  "
                      f"cgDiffs={m['cg_diffs']:>5d}  wgDiffs={m['wg_diffs']:>5d}  "
                      f"agree={m['agreement_pct']:.4f}%")
            else:
                print(f"  MISSING: {ORG_NAMES_PLAIN[org]} {DS_NAMES[ds]} {ver} "
                      f"(cc={cc_file.exists()}, ref={ref_file.exists()})")
    print()


# --- Timing ---
print("\n" + "=" * 120)
print("RUNTIME MEDIANS (seconds)")
print("=" * 120)


def parse_timing(path):
    timings = {}
    if not path.exists():
        return timings
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                name, t = parts
                timings[name] = float(t)
    return timings


def get_median_time(timings, org, ds, prefix=""):
    pattern = f"{prefix}{ds}_{org}_rep"
    vals = [v for k, v in timings.items() if k.startswith(pattern) and v < 50000]
    return median(vals) if vals else None


cc_cons_timing = parse_timing(BASE / "cc_fixed2" / "timing_e2e_cons.txt")
cc_pub_timing = parse_timing(BASE / "cc_fixed2" / "timing_e2e_pub.txt")

c354_timing = {}
for tf in ["timing.txt", "timing_extra.txt", "timing_rep23.txt"]:
    c354_timing.update(parse_timing(BASE / "c354" / tf))

c3310_timing = parse_timing(BASE / "c3_3310_e2e" / "timing.txt")
ni_timing = parse_timing(BASE / "c3310_e2e_noinferred" / "timing.txt")
for k, v in ni_timing.items():
    c3310_timing[f"e2e_{k}"] = v

print(f"\n{'Organism':<15s} {'Dataset':<12s} {'chewcall':>10s} {'c3.5.4':>10s} {'c3.3.10':>10s} {'Spdup354':>10s} {'Spdup3310':>10s}")
print("-" * 80)

for ds in DATASETS:
    for org in ORGANISMS:
        cc_t = get_median_time(cc_cons_timing if ds == "cons" else cc_pub_timing, org, ds, prefix="e2e_")
        c354_t = get_median_time(c354_timing, org, ds, prefix="e2e_")
        c3310_t = get_median_time(c3310_timing, org, ds, prefix="e2e_")

        cc_s = f"{cc_t:.0f}" if cc_t else "N/A"
        c354_s = f"{c354_t:.0f}" if c354_t else "N/A"
        c3310_s = f"{c3310_t:.0f}" if c3310_t else "N/A"
        sp354 = f"{c354_t/cc_t:.1f}x" if cc_t and c354_t else "N/A"
        sp3310 = f"{c3310_t/cc_t:.1f}x" if cc_t and c3310_t else "N/A"

        print(f"{ORG_NAMES_PLAIN[org]:<15s} {DS_NAMES[ds]:<12s} {cc_s:>10s} {c354_s:>10s} {c3310_s:>10s} {sp354:>10s} {sp3310:>10s}")


# --- LaTeX: Concordance ---
print("\n\n" + "=" * 120)
print("LaTeX TABLE: Concordance")
print("=" * 120)

print(r"""
\begin{table}[htbp]
\centering
\caption{CRC32 allelic profile concordance between \texttt{chewcall} and chewBBACA versions across 8 BeONE datasets.
Callable = cells where both tools assign a numeric CRC32 hash. cgDiffs/wgDiffs = discordant callable cells in cgMLST / wgMLST-only loci.}
\label{tab:concordance}
\resizebox{\textwidth}{!}{%
\begin{tabular}{ll rrr rrr}
\toprule
& & \multicolumn{3}{c}{vs.\ chewBBACA 3.5.4} & \multicolumn{3}{c}{vs.\ chewBBACA 3.3.10} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-8}
Organism & Dataset & Callable & cgDiffs & wgDiffs & Callable & cgDiffs & wgDiffs \\
\midrule""")

for ds in DATASETS:
    for org in ORGANISMS:
        r354 = next((r for r in results if r["org"]==org and r["ds"]==ds and r["comparison"]=="vs 3.5.4"), None)
        r3310 = next((r for r in results if r["org"]==org and r["ds"]==ds and r["comparison"]=="vs 3.3.10"), None)

        def fmt(r):
            if r is None:
                return "--- & --- & ---"
            return f"{r['total_callable']:,d} & {r['cg_diffs']} & {r['wg_diffs']}"

        print(f"{ORG_NAMES[org]} & {DS_NAMES[ds]} & {fmt(r354)} & {fmt(r3310)} \\\\")
    if ds == "cons":
        print(r"\midrule")

print(r"""\bottomrule
\end{tabular}%
}
\end{table}""")


# --- LaTeX: Runtime ---
print("\n\n" + "=" * 120)
print("LaTeX TABLE: Runtime")
print("=" * 120)

print(r"""
\begin{table}[htbp]
\centering
\caption{End-to-end runtime comparison (median of 3 replicates, seconds). Speedup = chewBBACA time / \texttt{chewcall} time.}
\label{tab:runtime}
\begin{tabular}{ll rrr cc}
\toprule
Organism & Dataset & chewcall & c3.5.4 & c3.3.10 & Speedup (3.5.4) & Speedup (3.3.10) \\
\midrule""")

for ds in DATASETS:
    for org in ORGANISMS:
        cc_t = get_median_time(cc_cons_timing if ds == "cons" else cc_pub_timing, org, ds, prefix="e2e_")
        c354_t = get_median_time(c354_timing, org, ds, prefix="e2e_")
        c3310_t = get_median_time(c3310_timing, org, ds, prefix="e2e_")

        cc_s = f"{cc_t:.0f}" if cc_t else "---"
        c354_s = f"{c354_t:.0f}" if c354_t else "---"
        c3310_s = f"{c3310_t:.0f}" if c3310_t else "---"
        sp354 = f"{c354_t/cc_t:.1f}$\\times$" if cc_t and c354_t else "---"
        sp3310 = f"{c3310_t/cc_t:.1f}$\\times$" if cc_t and c3310_t else "---"

        print(f"{ORG_NAMES[org]} & {DS_NAMES[ds]} & {cc_s} & {c354_s} & {c3310_s} & {sp354} & {sp3310} \\\\")
    if ds == "cons":
        print(r"\midrule")

print(r"""\bottomrule
\end{tabular}
\end{table}""")


# --- Summary ---
print("\n\n" + "=" * 120)
print("SUMMARY")
print("=" * 120)

for comp in ["vs 3.5.4", "vs 3.3.10"]:
    subset = [r for r in results if r["comparison"] == comp]
    if not subset:
        continue
    total_callable = sum(r["total_callable"] for r in subset)
    total_cg = sum(r["cg_diffs"] for r in subset)
    total_wg = sum(r["wg_diffs"] for r in subset)
    total_diffs = total_cg + total_wg
    pct = (total_callable - total_diffs) / total_callable * 100

    print(f"\n{comp}:")
    print(f"  Total callable cells: {total_callable:,d}")
    print(f"  cgMLST diffs:         {total_cg:,d}")
    print(f"  wgMLST-only diffs:    {total_wg:,d}")
    print(f"  Total diffs:          {total_diffs:,d}")
    print(f"  Overall agreement:    {pct:.6f}%")

    for ds in DATASETS:
        for org in ORGANISMS:
            r = next((x for x in subset if x["org"]==org and x["ds"]==ds), None)
            if r:
                print(f"    {ORG_NAMES_PLAIN[org]:15s} {DS_NAMES[ds]:12s}: "
                      f"{r['agreement_pct']:.6f}% "
                      f"(cg={r['cg_diffs']}, wg={r['wg_diffs']}, callable={r['total_callable']:,d})")
