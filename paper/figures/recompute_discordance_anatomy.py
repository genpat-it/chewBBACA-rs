#!/usr/bin/env python3
"""Recompute discordance categories per dataset, split into cgMLST/wgMLST.
Emits pgfplots coordinates for fig_discordance_anatomy.tex.
"""
import csv
import os

BASE = "/mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results"
CGMLST = {
    "lm": None,
    "se": "/mnt/disk2/a.deruvo/chew_results/se_cgmlst_loci.txt",
    "ec": "/mnt/disk2/a.deruvo/chew_results/ec_cgmlst_loci.txt",
    "cj": "/mnt/disk2/a.deruvo/chew_results/cj_cgmlst_loci.txt",
}

cc_paths = {
    (org, ds): f"{BASE}/cc_fixed2/shcds_{ds}_{org}_rep1/results_alleles.tsv"
    for org in ("lm", "se", "ec", "cj") for ds in ("cons", "pub")
}
cb_paths = {
    (org, ds): f"{BASE}/c354/shcds_{ds}_{org}_rep1/results_alleles.tsv"
    for org in ("lm", "se", "ec", "cj") for ds in ("cons", "pub")
}

def load_cgmlst(org):
    path = CGMLST[org]
    if path is None:
        return None
    with open(path) as f:
        return set(l.strip() for l in f if l.strip())

def load_alleles(path):
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        loci = [c for c in reader.fieldnames if c != "FILE"]
        data = {}
        for row in reader:
            g = row["FILE"]
            for suf in (".cds.fasta", ".cds.fna", ".cds", ".fasta", ".fa"):
                if g.endswith(suf):
                    g = g[: -len(suf)]
                    break
            data[g] = [row[l] for l in loci]
    return data, loci

def classify(v):
    if not v: return "LNF"
    if v.startswith("INF-") or v.startswith("*"): return "INF"
    if v.isdigit(): return "EXC"
    if v.startswith("NIPHEM"): return "NIPHEM"
    if v.startswith("NIPH"): return "NIPH"
    return v

def category(cb, cc):
    cb_call = cb in ("EXC", "INF")
    cc_call = cc in ("EXC", "INF")
    cb_par  = cb in ("NIPH", "NIPHEM")
    cc_par  = cc in ("NIPH", "NIPHEM")
    # cb finds an allele
    if cb_call and not cc_call:
        return "cb_paralog" if cc_par else "cb_real"
    # cc finds an allele
    if cc_call and not cb_call:
        return "cc_paralog" if cb_par else "cc_real"
    # neither finds a callable allele
    if cb_par or cc_par:
        return "paralog_both"
    return "other"

CATS = ["cb_real", "cb_paralog", "cc_real", "cc_paralog", "paralog_both", "other"]
results = {}

for ds in ("cons", "pub"):
    for org in ("lm", "se", "ec", "cj"):
        cg_loci = load_cgmlst(org)
        cc_d, cc_l = load_alleles(cc_paths[(org, ds)])
        cb_d, cb_l = load_alleles(cb_paths[(org, ds)])
        common_loci = [l for l in cc_l if l in set(cb_l)]
        cb_idx = {l: i for i, l in enumerate(cb_l)}
        cc_idx = {l: i for i, l in enumerate(cc_l)}
        common_g = sorted(set(cc_d) & set(cb_d))

        cg_counts = {c: 0 for c in CATS}
        wg_counts = {c: 0 for c in CATS}

        for g in common_g:
            cc_row = cc_d[g]
            cb_row = cb_d[g]
            for l in common_loci:
                cc_v = cc_row[cc_idx[l]]
                cb_v = cb_row[cb_idx[l]]
                cc_c = classify(cc_v)
                cb_c = classify(cb_v)
                if cc_c == cb_c:
                    continue
                cat = category(cb_c, cc_c)
                is_cg = cg_loci is None or l in cg_loci
                if is_cg:
                    cg_counts[cat] += 1
                else:
                    wg_counts[cat] += 1

        key = f"{org}_{ds}"
        results[key] = (cg_counts, wg_counts)
        print(f"{key}: cg={cg_counts}, wg={wg_counts}")

# Emit pgfplots coordinates for each category
print("\n\n% PGFPLOTS COORDINATES")
LABELS = {
    "lm_cons": "Lm cons.\\ cg",  # Lm only cg
    "se_cons_cg": "Se cons.\\ cg", "se_cons_wg": "Se cons.\\ wg",
    "ec_cons_cg": "Ec cons.\\ cg", "ec_cons_wg": "Ec cons.\\ wg",
    "cj_cons_cg": "Cj cons.\\ cg", "cj_cons_wg": "Cj cons.\\ wg",
    "lm_pub": "Lm pub.\\ cg",
    "se_pub_cg": "Se pub.\\ cg", "se_pub_wg": "Se pub.\\ wg",
    "ec_pub_cg": "Ec pub.\\ cg", "ec_pub_wg": "Ec pub.\\ wg",
    "cj_pub_cg": "Cj pub.\\ cg", "cj_pub_wg": "Cj pub.\\ wg",
}

CAT_NAMES = {
    "INFEXC": "INF->EXC",
    "cb_only": "cb calls, cc does not",
    "cc_only": "cc calls, cb does not",
    "paralog": "Paralog differences",
    "other": "Other non-callable",
}

for cat in CATS:
    print(f"\n% {CAT_NAMES[cat]}")
    pieces = []
    for org, ds in [("lm", "cons"), ("se", "cons"), ("ec", "cons"), ("cj", "cons"),
                    ("lm", "pub"), ("se", "pub"), ("ec", "pub"), ("cj", "pub")]:
        key = f"{org}_{ds}"
        cg, wg = results[key]
        if org == "lm":
            label = f"Lm {ds}.\\ cg"
            pieces.append(f"({cg[cat]},{{{label}}})")
        else:
            for which, counts in (("cg", cg), ("wg", wg)):
                label = f"{org.capitalize()} {ds}.\\ {which}"
                pieces.append(f"({counts[cat]},{{{label}}})")
    print("    " + " ".join(pieces))
