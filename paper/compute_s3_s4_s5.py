#!/usr/bin/env python3
"""Compute supplementary tables S3, S4, S5 for chewcall paper.

Compares chewcall vs chewBBACA v3.5.4 results_alleles.tsv (non-hashed)
on all 8 BeONE datasets (shared pre-computed CDS).
"""

import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# --- File paths ---

CHEWCALL_CONS = {
    "lm": "/mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results/cc_fixed2/shcds_cons_lm_rep1/results_alleles.tsv",
    "se": "/mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results/cc_fixed2/shcds_cons_se_rep1/results_alleles.tsv",
    "ec": "/mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results/cc_fixed2/shcds_cons_ec_rep1/results_alleles.tsv",
    "cj": "/mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results/cc_fixed2/shcds_cons_cj_rep1/results_alleles.tsv",
}

CHEWCALL_PUB = {
    "lm": "/mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results/cc_fixed2/shcds_pub_lm_rep1/results_alleles.tsv",
    "se": "/mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results/cc_fixed2/shcds_pub_se_rep1/results_alleles.tsv",
    "ec": "/mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results/cc_fixed2/shcds_pub_ec_rep1/results_alleles.tsv",
    "cj": "/mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results/cc_fixed2/shcds_pub_cj_rep1/results_alleles.tsv",
}

C354_CONS = {
    "lm": "/mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results/c354/shcds_cons_lm_rep1/results_alleles.tsv",
    "se": "/mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results/c354/shcds_cons_se_rep1/results_alleles.tsv",
    "ec": "/mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results/c354/shcds_cons_ec_rep1/results_alleles.tsv",
    "cj": "/mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results/c354/shcds_cons_cj_rep1/results_alleles.tsv",
}

C354_PUB = {
    "lm": "/mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results/c354/shcds_pub_lm_rep1/results_alleles.tsv",
    "se": "/mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results/c354/shcds_pub_se_rep1/results_alleles.tsv",
    "ec": "/mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results/c354/shcds_pub_ec_rep1/results_alleles.tsv",
    "cj": "/mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results/c354/shcds_pub_cj_rep1/results_alleles.tsv",
}

ORG_NAMES = {"lm": "L.m.", "se": "S.e.", "ec": "E.c.", "cj": "C.j."}
ORG_ORDER = ["lm", "se", "ec", "cj"]


def classify_cell(value: str, is_chewbbaca: bool = False) -> str:
    """Classify a cell value into a label."""
    v = value.strip()
    if v == "":
        return "LNF"  # shouldn't happen but just in case

    # Both tools may use * prefix for INF (novel alleles added to schema)
    # chewBBACA v3.5.4: *NNN, INF-*NNN
    # chewcall: INF-NNN or *NNN (when --update-schema was used)
    if v.startswith("INF-"):
        return "INF"
    if v.startswith("*"):
        return "INF"

    # Check label types (before numeric check)
    if v == "NIPHEM":
        return "NIPHEM"
    if v == "NIPH":
        return "NIPH"
    if v in ("LNF", "ASM", "ALM", "PLOT3", "PLOT5", "LOTSC", "PAMA"):
        return v

    # Pure number = EXC (or previously inferred allele now in schema)
    if re.match(r"^\d+$", v):
        return "EXC"

    # Fallback
    print(f"WARNING: unrecognized cell value: '{v}' (chewbbaca={is_chewbbaca})", file=sys.stderr)
    return v


def is_callable(label: str) -> bool:
    return label in ("EXC", "INF")


def load_and_align(cc_path: str, cb_path: str):
    """Load two results_alleles.tsv files and align by genome x locus.

    Returns aligned DataFrames with matching genome rows and locus columns.
    """
    cc = pd.read_csv(cc_path, sep="\t", dtype=str)
    cb = pd.read_csv(cb_path, sep="\t", dtype=str)

    # Normalize genome names: strip .cds suffix from chewBBACA
    cc = cc.set_index("FILE")
    cb = cb.set_index("FILE")
    cb.index = cb.index.str.replace(r"\.cds$", "", regex=True)

    # Find common genomes and loci
    common_genomes = sorted(set(cc.index) & set(cb.index))
    common_loci = sorted(set(cc.columns) & set(cb.columns))

    if len(common_genomes) == 0:
        print(f"WARNING: no common genomes between {cc_path} and {cb_path}", file=sys.stderr)
        print(f"  cc genomes sample: {list(cc.index[:5])}", file=sys.stderr)
        print(f"  cb genomes sample: {list(cb.index[:5])}", file=sys.stderr)
        return None, None

    cc_aligned = cc.loc[common_genomes, common_loci]
    cb_aligned = cb.loc[common_genomes, common_loci]

    n_genomes = len(common_genomes)
    n_loci = len(common_loci)
    print(f"  Aligned: {n_genomes} genomes x {n_loci} loci = {n_genomes * n_loci:,} cells")

    # Check for missing genomes/loci
    cc_only_genomes = set(cc.index) - set(cb.index)
    cb_only_genomes = set(cb.index) - set(cc.index)
    if cc_only_genomes:
        print(f"  WARNING: {len(cc_only_genomes)} genomes only in chewcall", file=sys.stderr)
    if cb_only_genomes:
        print(f"  WARNING: {len(cb_only_genomes)} genomes only in chewBBACA", file=sys.stderr)

    return cc_aligned, cb_aligned


def compute_transitions_and_actionable(cc_df, cb_df):
    """Compute classification transitions and actionable differences."""
    transitions = Counter()
    total_cells = 0
    same = 0
    cc_calls = 0  # chewcall calls, chewBBACA doesn't
    cb_calls = 0  # chewBBACA calls, chewcall doesn't

    for locus in cc_df.columns:
        for genome in cc_df.index:
            cc_val = cc_df.at[genome, locus]
            cb_val = cb_df.at[genome, locus]

            cc_label = classify_cell(cc_val, is_chewbbaca=False)
            cb_label = classify_cell(cb_val, is_chewbbaca=True)

            total_cells += 1

            cc_callable = is_callable(cc_label)
            cb_callable = is_callable(cb_label)

            if cc_label != cb_label:
                transitions[(cb_label, cc_label)] += 1

            # Actionable differences
            if cc_callable and cb_callable:
                # Both call - count as same (we don't check allele identity here
                # since the non-hashed values differ in format)
                same += 1
            elif not cc_callable and not cb_callable:
                if cc_label == cb_label:
                    same += 1
                else:
                    # Both non-callable but different labels - still "same" for
                    # actionable purposes (neither calls an allele)
                    same += 1
            elif cc_callable and not cb_callable:
                cc_calls += 1
            elif cb_callable and not cc_callable:
                cb_calls += 1

    return transitions, total_cells, same, cc_calls, cb_calls


def format_transitions_table(all_transitions, org_order, title, top_n=15):
    """Format transitions as LaTeX table rows."""
    # Collect all transition types across organisms
    all_types = Counter()
    for org in org_order:
        for (cb, cc), count in all_transitions[org].items():
            all_types[(cb, cc)] += count

    # Sort by total count
    sorted_types = sorted(all_types.items(), key=lambda x: -x[1])

    # Take top N, rest go to "Other"
    top_types = [t for t, _ in sorted_types[:top_n]]

    lines = []
    other_counts = {org: 0 for org in org_order}
    total_counts = {org: sum(all_transitions[org].values()) for org in org_order}

    for (cb_label, cc_label) in top_types:
        cols = []
        for org in org_order:
            count = all_transitions[org].get((cb_label, cc_label), 0)
            if count == 0:
                cols.append("--")
            else:
                cols.append(f"{count:,}".replace(",", "\\,"))
        lines.append(f"\t{cb_label} & {cc_label} & {' & '.join(cols)} \\\\")

    # Compute "Other" counts
    for org in org_order:
        top_sum = sum(all_transitions[org].get(t, 0) for t in top_types)
        other_counts[org] = total_counts[org] - top_sum

    # Add Other row if non-zero
    if any(v > 0 for v in other_counts.values()):
        cols = []
        for org in org_order:
            if other_counts[org] == 0:
                cols.append("--")
            else:
                cols.append(f"{other_counts[org]:,}".replace(",", "\\,"))
        lines.append(f"\tOther & & {' & '.join(cols)} \\\\")

    # Total row
    cols = []
    for org in org_order:
        cols.append("\\textbf{" + f"{total_counts[org]:,}".replace(",", "\\,") + "}")
    lines.append(f"\t\\addlinespace\n\t\\textbf{{Total}} & & {' & '.join(cols)} \\\\")

    return "\n".join(lines)


def format_transitions_unified(cons_trans, pub_trans, org_order, top_n=15):
    """Format unified 8-col transitions table (4 cons + 4 pub)."""
    keys = [("Cons", o) for o in org_order] + [("Pub", o) for o in org_order]
    data = {("Cons", o): cons_trans[o] for o in org_order}
    data.update({("Pub", o): pub_trans[o] for o in org_order})

    all_types = Counter()
    for k in keys:
        for t, c in data[k].items():
            all_types[t] += c

    sorted_types = sorted(all_types.items(), key=lambda x: -x[1])
    top_types = [t for t, _ in sorted_types[:top_n]]

    lines = []
    total_counts = {k: sum(data[k].values()) for k in keys}

    for (cb_label, cc_label) in top_types:
        cols = []
        for k in keys:
            count = data[k].get((cb_label, cc_label), 0)
            cols.append("--" if count == 0 else f"{count:,}".replace(",", "\\,"))
        lines.append(f"\t{cb_label} & {cc_label} & {' & '.join(cols)} \\\\")

    other_counts = {k: total_counts[k] - sum(data[k].get(t, 0) for t in top_types) for k in keys}
    if any(v > 0 for v in other_counts.values()):
        cols = ["--" if other_counts[k] == 0 else f"{other_counts[k]:,}".replace(",", "\\,") for k in keys]
        lines.append(f"\tOther & & {' & '.join(cols)} \\\\")

    cols = ["\\textbf{" + f"{total_counts[k]:,}".replace(",", "\\,") + "}" for k in keys]
    lines.append(f"\t\\addlinespace\n\t\\textbf{{Total}} & & {' & '.join(cols)} \\\\")

    return "\n".join(lines)


def main():
    datasets = [
        ("Consortium", CHEWCALL_CONS, C354_CONS),
        ("Public", CHEWCALL_PUB, C354_PUB),
    ]

    all_transitions = {}
    all_actionable = {}

    for ds_name, cc_paths, cb_paths in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        for org in ORG_ORDER:
            print(f"\n--- {ORG_NAMES[org]} ---")
            key = (ds_name, org)

            cc_aligned, cb_aligned = load_and_align(cc_paths[org], cb_paths[org])
            if cc_aligned is None:
                continue

            transitions, total, same, cc_calls, cb_calls = \
                compute_transitions_and_actionable(cc_aligned, cb_aligned)

            all_transitions[key] = transitions
            all_actionable[key] = (total, same, cc_calls, cb_calls)

            print(f"  Total cells: {total:,}")
            print(f"  Same: {same:,}")
            print(f"  cc calls: {cc_calls:,}")
            print(f"  cb calls: {cb_calls:,}")
            print(f"  Total transitions: {sum(transitions.values()):,}")

            # Print top transitions
            for (cb_l, cc_l), count in sorted(transitions.items(), key=lambda x: -x[1])[:10]:
                print(f"    {cb_l} -> {cc_l}: {count}")

    # --- Generate LaTeX ---
    print("\n\n" + "="*80)
    print("LATEX OUTPUT")
    print("="*80)

    # S3 unified: Consortium + Public transitions
    cons_trans = {org: all_transitions.get(("Consortium", org), Counter()) for org in ORG_ORDER}
    pub_trans = {org: all_transitions.get(("Public", org), Counter()) for org in ORG_ORDER}
    print("\n% Table S3 UNIFIED: Classification transitions (Cons | Pub)")
    print(format_transitions_unified(cons_trans, pub_trans, ORG_ORDER))

    # S5: Actionable differences
    print("\n% Table S5: Actionable differences")
    for ds_name, ds_short in [("Consortium", "Cons."), ("Public", "Pub.")]:
        for org in ORG_ORDER:
            key = (ds_name, org)
            if key in all_actionable:
                total, same, cc_calls, cb_calls = all_actionable[key]
                org_tex = {"lm": "\\textit{L.\\,mono.}", "se": "\\textit{S.\\,enter.}",
                           "ec": "\\textit{E.\\,coli}", "cj": "\\textit{C.\\,jejuni}"}[org]
                total_fmt = f"{total:,}".replace(",", "\\,")
                same_fmt = f"{same:,}".replace(",", "\\,")
                cc_fmt = f"{cc_calls:,}".replace(",", "\\,")
                cb_fmt = f"{cb_calls:,}".replace(",", "\\,")
                print(f"\t{ds_short} & {org_tex} & {total_fmt} & {same_fmt} & {cc_fmt} & {cb_fmt} \\\\")
        if ds_name == "Consortium":
            print("\t\\addlinespace")


if __name__ == "__main__":
    main()
