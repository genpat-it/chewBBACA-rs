#!/usr/bin/env python3
"""10_supp_diagnostic_cases.py — Supplementary Table S5.

Mechanism breakdown of the actionable differences (cells where exactly one of
chewcall / chewBBACA v3.5.4 returns a callable allele) on the 8 BeONE shcds
datasets, plus a deterministic representative sample of individual cells.

This is a DIAGNOSTIC categorisation by classification transition, NOT a
biological validation: there is no cell-level ground truth, so each case is
labelled by the mechanism implied by its (chewBBACA -> chewcall) transition.

Usage: python 10_supp_diagnostic_cases.py
Env:   OUT_CHEWCALL (default results/cc_fixed2), OUT_C354 (default results/c354)
"""
import csv, os
from collections import Counter, defaultdict
from pathlib import Path

BASE     = Path(os.environ.get("OUT_ROOT", "results"))
DIR_CC   = Path(os.environ.get("OUT_CHEWCALL", str(BASE / "cc_fixed2")))
DIR_C354 = Path(os.environ.get("OUT_C354",     str(BASE / "c354")))
ORGS = ["lm", "se", "ec", "cj"]
DS   = ["cons", "pub"]
SAMPLE_PER_CLASS = 3          # deterministic cap per transition class

def classify(v):
    v = v.strip()
    if not v: return "LNF"
    if v.startswith("INF-") or v.startswith("*"): return "INF"
    if v.startswith("NIPHEM"): return "NIPHEM"
    if v.startswith("NIPH"): return "NIPH"
    if v in ("LNF","ASM","ALM","PLOT3","PLOT5","LOTSC","PAMA"): return v
    if v.lstrip("-+").isdigit(): return "EXC"
    return v

def callable_(c): return c in ("EXC","INF")

def norm(g):
    for s in (".cds",".fasta",".fa",".fna",".fastq"):
        if g.endswith(s): g = g[:-len(s)]
    return g

def load(path):
    with open(path) as f:
        rows = list(csv.reader(f, delimiter="\t"))
    loci = rows[0][1:]
    return {norm(r[0]): dict(zip(loci, r[1:])) for r in rows[1:]}

def mechanism(direction, cb, cc):
    """Map a transition to a coarse mechanism category."""
    if direction == "cc_only":
        if cb == "LNF":
            return "M1: chewcall recovers an allele chewBBACA reports as LNF"
        if cb in ("NIPH","NIPHEM"):
            return "M2: chewcall resolves a single allele where chewBBACA flags a paralog"
        return "M5: length/position edge (ASM/ALM)"
    else:  # cb_only
        if cc == "LNF":
            return "M3: minimizer pre-filter recall miss (chewcall LNF)"
        if cc in ("NIPH","NIPHEM","PAMA"):
            return "M4: chewcall is more conservative, flags a paralog/multi-match"
        return "M5: length/position edge (ASM/ALM)"

trans = Counter()
examples = defaultdict(list)
total = 0
per_org = Counter()

for org in ORGS:
    for ds in DS:
        pc = DIR_CC   / f"shcds_{ds}_{org}_rep1" / "results_alleles.tsv"
        pb = DIR_C354 / f"shcds_{ds}_{org}_rep1" / "results_alleles.tsv"
        if not (pc.exists() and pb.exists()):
            print(f"# MISSING {org} {ds}"); continue
        cc, cb = load(pc), load(pb)
        genomes = sorted(set(cc) & set(cb))
        loci = sorted(set(next(iter(cc.values()))) & set(next(iter(cb.values()))))
        for g in genomes:
            cg, bg = cc[g], cb[g]
            for L in loci:
                if L not in cg or L not in bg: continue
                ccl, cbl = classify(cg[L]), classify(bg[L])
                if callable_(ccl) == callable_(cbl): continue
                total += 1; per_org[org] += 1
                direction = "cc_only" if callable_(ccl) else "cb_only"
                key = (direction, cbl, ccl)
                trans[key] += 1
                examples[key].append((org, ds, g, L, cbl, ccl))

# ---- mechanism breakdown ----
mech_count = Counter()
for (d, cb, cc), n in trans.items():
    mech_count[mechanism(d, cb, cc)] += n

print(f"\n# Total actionable cells (8 shcds datasets): {total}\n")
print("# Mechanism breakdown")
for m, n in sorted(mech_count.items()):
    print(f"  {n:6d}  {100*n/total:5.1f}%  {m}")

print("\n# Transition distribution (direction | chewBBACA -> chewcall | count)")
for (d, cb, cc), n in trans.most_common():
    print(f"  {d:8s} {cb:7s} -> {cc:7s} : {n}")

# ---- deterministic representative sample ----
sample = []
for (d, cb, cc), n in trans.most_common():
    for ex in sorted(examples[(d, cb, cc)])[:SAMPLE_PER_CLASS]:
        sample.append((ex, mechanism(d, cb, cc)))
print(f"\n# Representative sample: {len(sample)} cells "
      f"(<= {SAMPLE_PER_CLASS} per transition class)")

# ---- LaTeX for Table S5.2 (sample) ----
ORG_TEX = {"lm":"L.m.", "se":"S.e.", "ec":"E.c.", "cj":"C.j."}
MECH_TAG = {"M1":"M1","M2":"M2","M3":"M3","M4":"M4","M5":"M5"}
print("\n% ---- LaTeX: representative sample (S5.2) ----")
for (org, ds, g, L, cb, cc), m in sample:
    tag = m.split(":")[0]
    print(f"{ORG_TEX[org]} & {ds} & \\texttt{{{g}}} & \\texttt{{{L}}} & {cb} & {cc} & {tag} \\\\")
