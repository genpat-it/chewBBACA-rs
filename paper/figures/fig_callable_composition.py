#!/usr/bin/env python3
"""
Generate stacked bar chart showing callable cell composition per dataset.
For each organism/dataset: both callable same CRC32, both callable diff CRC32,
callable only in chewcall, callable only in chewBBACA, both non-callable.
Two panels: cc vs 3.5.4 (left) and cc vs 3.3.10 (right).
"""
import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BASE = "/mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results"

def find_hashed(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    d = os.path.dirname(path)
    if not os.path.isdir(d):
        return path
    for f in sorted(os.listdir(d)):
        if f.startswith("results_2"):
            subpath = os.path.join(d, f, "results_alleles_hashed.tsv")
            if os.path.exists(subpath):
                return subpath
    return path

def load_hashed(path):
    path = find_hashed(path)
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        loci = [c for c in reader.fieldnames if c != 'FILE']
        data = {}
        for row in reader:
            g = row['FILE']
            for suf in ['.cds.fasta', '.cds.fna', '.cds', '.fasta', '.fa']:
                if g.endswith(suf):
                    g = g[:-len(suf)]
                    break
            data[g] = {l: row[l] for l in loci}
    return data, loci

def is_callable(val):
    return val.isdigit()

datasets = [
    ('Cons. Lm', 'lm', 'cons'),
    ('Cons. Se', 'se', 'cons'),
    ('Cons. Ec', 'ec', 'cons'),
    ('Cons. Cj', 'cj', 'cons'),
    ('Pub. Lm', 'lm', 'pub'),
    ('Pub. Se', 'se', 'pub'),
    ('Pub. Ec', 'ec', 'pub'),
    ('Pub. Cj', 'cj', 'pub'),
]

cc_paths = {
    ('lm', 'cons'): f"{BASE}/chewcall/lm_rep1/results_alleles_hashed.tsv",
    ('se', 'cons'): f"{BASE}/chewcall/se_rep1/results_alleles_hashed.tsv",
    ('ec', 'cons'): f"{BASE}/chewcall/ec_rep1/results_alleles_hashed.tsv",
    ('cj', 'cons'): f"{BASE}/chewcall/cj_rep1/results_alleles_hashed.tsv",
    ('lm', 'pub'): "/mnt/disk2/a.deruvo/chew_results/lm_public/rust_results/results_alleles_hashed.tsv",
    ('se', 'pub'): "/mnt/disk2/a.deruvo/chew_results/se_public/rust_results/results_alleles_hashed.tsv",
    ('ec', 'pub'): "/mnt/disk2/a.deruvo/chew_results/ec_public/rust_results/results_alleles_hashed.tsv",
    ('cj', 'pub'): "/mnt/disk2/a.deruvo/chew_results/cj_public/rust_results/results_alleles_hashed.tsv",
}

cb_paths = {
    '354': {
        ('lm', 'cons'): f"{BASE}/c354/shcds_cons_lm_rep1/results_alleles_hashed.tsv",
        ('se', 'cons'): f"{BASE}/c354/shcds_cons_se_rep1/results_alleles_hashed.tsv",
        ('ec', 'cons'): f"{BASE}/c354/shcds_cons_ec_rep1/results_alleles_hashed.tsv",
        ('cj', 'cons'): f"{BASE}/c354/shcds_cons_cj_rep1/results_alleles_hashed.tsv",
        ('lm', 'pub'): f"{BASE}/c354/shcds_pub_lm_rep1/results_alleles_hashed.tsv",
        ('se', 'pub'): f"{BASE}/c354/shcds_pub_se_rep1/results_alleles_hashed.tsv",
        ('ec', 'pub'): f"{BASE}/c354/shcds_pub_ec_rep1/results_alleles_hashed.tsv",
        ('cj', 'pub'): f"{BASE}/c354/shcds_pub_cj_rep1/results_alleles_hashed.tsv",
    },
    '3310': {
        ('lm', 'cons'): f"{BASE}/c3_3310/lm_rep1/results_alleles_hashed.tsv",
        ('se', 'cons'): f"{BASE}/c3_3310/se_rep1/results_alleles_hashed.tsv",
        ('ec', 'cons'): f"{BASE}/c3_3310/ec_rep1/results_alleles_hashed.tsv",
        ('cj', 'cons'): f"{BASE}/c3_3310/cj_rep1/results_alleles_hashed.tsv",
        ('lm', 'pub'): f"{BASE}/c3_3310/lm_public_rep1/results_alleles_hashed.tsv",
        ('se', 'pub'): f"{BASE}/c3_3310/se_public_rep1/results_alleles_hashed.tsv",
        ('ec', 'pub'): f"{BASE}/c3_3310/ec_public_rep1/results_alleles_hashed.tsv",
        ('cj', 'pub'): f"{BASE}/c3_3310/cj_public_rep1/results_alleles_hashed.tsv",
    },
}

def compute_composition(cc_path, cb_path):
    cc_data, cc_loci = load_hashed(cc_path)
    cb_data, cb_loci = load_hashed(cb_path)
    common_loci = sorted(set(cc_loci) & set(cb_loci))
    common_genomes = sorted(set(cc_data.keys()) & set(cb_data.keys()))

    both_same = 0
    both_diff = 0
    cc_only = 0
    cb_only = 0
    both_non = 0

    for g in common_genomes:
        for l in common_loci:
            cc_val = cc_data[g].get(l, '')
            cb_val = cb_data[g].get(l, '')
            cc_call = is_callable(cc_val)
            cb_call = is_callable(cb_val)

            if cc_call and cb_call:
                if cc_val == cb_val:
                    both_same += 1
                else:
                    both_diff += 1
            elif cc_call and not cb_call:
                cc_only += 1
            elif not cc_call and cb_call:
                cb_only += 1
            else:
                both_non += 1

    return both_same, both_diff, cc_only, cb_only, both_non

# Compute for both versions
results = {}
for ver in ['354', '3310']:
    results[ver] = []
    for label, org, ds in datasets:
        print(f"Computing {ver} {label}...")
        comp = compute_composition(cc_paths[(org, ds)], cb_paths[ver][(org, ds)])
        results[ver].append(comp)
        total = sum(comp)
        print(f"  same={comp[0]:,} diff={comp[1]} cc_only={comp[2]:,} cb_only={comp[3]:,} non={comp[4]:,} total={total:,}")

# Compute cgMLST vs wgMLST callable fractions
cgmlst_loci = {}
for org in ['se', 'ec', 'cj']:
    f = f"/mnt/disk2/a.deruvo/chew_results/{org}_cgmlst_loci.txt"
    with open(f) as fh:
        cgmlst_loci[org] = set(l.strip() for l in fh if l.strip())

def compute_cg_wg(cc_path, cb_path, org):
    cc_data, cc_l = load_hashed(cc_path)
    cb_data, cb_l = load_hashed(cb_path)
    common_loci = sorted(set(cc_l) & set(cb_l))
    common_genomes = sorted(set(cc_data.keys()) & set(cb_data.keys()))

    cg = {'callable': 0, 'total': 0, 'diff': 0, 'cc_only': 0, 'cb_only': 0}
    wg = {'callable': 0, 'total': 0, 'diff': 0, 'cc_only': 0, 'cb_only': 0}

    for g in common_genomes:
        for l in common_loci:
            cc_val = cc_data[g].get(l, '')
            cb_val = cb_data[g].get(l, '')
            cc_call = is_callable(cc_val)
            cb_call = is_callable(cb_val)

            is_cg = (org == 'lm') or (l in cgmlst_loci.get(org, set()))
            bucket = cg if is_cg else wg
            bucket['total'] += 1

            if cc_call and cb_call:
                bucket['callable'] += 1
                if cc_val != cb_val:
                    bucket['diff'] += 1
            elif cc_call:
                bucket['cc_only'] += 1
            elif cb_call:
                bucket['cb_only'] += 1

    return cg, wg

ver = '354'
cg_wg_data = []
for label, org, ds in datasets:
    print(f"Computing cgMLST/wgMLST split for {label}...")
    cg, wg = compute_cg_wg(cc_paths[(org, ds)], cb_paths[ver][(org, ds)], org)
    cg_wg_data.append((cg, wg))
    cg_pct = cg['callable'] / cg['total'] * 100 if cg['total'] > 0 else 0
    wg_pct = wg['callable'] / wg['total'] * 100 if wg['total'] > 0 else 0
    print(f"  cgMLST: {cg_pct:.1f}% ({cg['callable']:,}/{cg['total']:,})  wgMLST: {wg_pct:.1f}% ({wg['callable']:,}/{wg['total']:,})")

# Plot: grouped bars (cgMLST vs wgMLST) per dataset
# Skip Lm wgMLST (doesn't exist)
fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))

labels_short = ['Lm\ncons.', 'Se\ncons.', 'Ec\ncons.', 'Cj\ncons.',
                'Lm\npub.', 'Se\npub.', 'Ec\npub.', 'Cj\npub.']

x = np.arange(len(labels_short))
width = 0.35

color_cg = '#2980b9'  # blue
color_wg = '#e67e22'  # orange
color_cg_non = '#aed6f1'  # light blue
color_wg_non = '#fad7a0'  # light orange

cg_callable_pct = []
wg_callable_pct = []
cg_actionable_pct = []
wg_actionable_pct = []

for cg, wg in cg_wg_data:
    cg_pct = cg['callable'] / cg['total'] * 100 if cg['total'] > 0 else 0
    wg_pct = wg['callable'] / wg['total'] * 100 if wg['total'] > 0 else 0
    cg_act = (cg['cc_only'] + cg['cb_only']) / cg['total'] * 100 if cg['total'] > 0 else 0
    wg_act = (wg['cc_only'] + wg['cb_only']) / wg['total'] * 100 if wg['total'] > 0 else 0
    cg_callable_pct.append(cg_pct)
    wg_callable_pct.append(wg_pct if wg['total'] > 0 else -1)  # -1 = N/A
    cg_actionable_pct.append(cg_act)
    wg_actionable_pct.append(wg_act if wg['total'] > 0 else 0)

cg_callable_pct = np.array(cg_callable_pct)
wg_callable_pct = np.array(wg_callable_pct)

# cgMLST bars (solid = callable match, light = non-callable)
bars_cg = ax.bar(x - width/2, cg_callable_pct, width,
                 color=color_cg, label='cgMLST callable (100% CRC32 match)', edgecolor='white', linewidth=0.5)
bars_cg_top = ax.bar(x - width/2, 100 - cg_callable_pct, width, bottom=cg_callable_pct,
                     color=color_cg_non, label='cgMLST non-callable', edgecolor='white', linewidth=0.5)

# wgMLST bars (skip Lm which has no wgMLST)
wg_vals = np.where(wg_callable_pct >= 0, wg_callable_pct, 0)
wg_top = np.where(wg_callable_pct >= 0, 100 - wg_callable_pct, 0)
mask_wg = wg_callable_pct >= 0

bars_wg = ax.bar(x[mask_wg] + width/2, wg_vals[mask_wg], width,
                 color=color_wg, label='wgMLST-only callable (100% CRC32 match)', edgecolor='white', linewidth=0.5)
bars_wg_top = ax.bar(x[mask_wg] + width/2, wg_top[mask_wg], width, bottom=wg_vals[mask_wg],
                     color=color_wg_non, label='wgMLST-only non-callable', edgecolor='white', linewidth=0.5)

# Add percentage labels and cg/wg labels at base
for i in range(len(x)):
    ax.text(x[i] - width/2, cg_callable_pct[i] + 1.5, f'{cg_callable_pct[i]:.0f}%',
            ha='center', va='bottom', fontsize=7, fontweight='bold', color=color_cg)
    ax.text(x[i] - width/2, -5, 'cg', ha='center', va='top', fontsize=7,
            fontweight='bold', color=color_cg)
    if wg_callable_pct[i] >= 0:
        ax.text(x[i] + width/2, wg_vals[i] + 1.5, f'{wg_vals[i]:.0f}%',
                ha='center', va='bottom', fontsize=7, fontweight='bold', color=color_wg)
        ax.text(x[i] + width/2, -5, 'wg', ha='center', va='top', fontsize=7,
                fontweight='bold', color=color_wg)

ax.set_xticks(x)
ax.set_xticklabels(labels_short, fontsize=9)
ax.set_ylim(-8, 108)
ax.set_ylabel('Cells (%)', fontsize=11)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(fontsize=8, loc='center right', framealpha=0.9)

plt.tight_layout()

outpath = os.path.join(os.path.dirname(__file__), 'fig_callable_composition.pdf')
fig.savefig(outpath, bbox_inches='tight', dpi=300)
print(f"\nFigure saved to {outpath}")

# Also save as PNG for quick preview
outpath_png = outpath.replace('.pdf', '.png')
fig.savefig(outpath_png, bbox_inches='tight', dpi=150)
print(f"PNG preview saved to {outpath_png}")
