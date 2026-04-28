#!/usr/bin/env python3
"""
Discordance anatomy plot: for each organism, shows the breakdown of
discordant cells by transition category. Two panels:
- Left: absolute scale showing concordance vs discordance (log or linear)
- Right: zoom into discordant cells, broken down by category
"""
import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BASE = "/mnt/disk2/a.deruvo/chewcall_benchmarks/01_beone/results"

def find_tsv(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    d = os.path.dirname(path)
    if not os.path.isdir(d):
        return path
    for f in sorted(os.listdir(d)):
        if f.startswith("results_2"):
            subpath = os.path.join(d, f, os.path.basename(path))
            if os.path.exists(subpath):
                return subpath
    return path

def load_alleles(path):
    path = find_tsv(path)
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

def classify_cell(val):
    """Classify a cell value into a category."""
    if not val or val == '':
        return 'LNF'
    if val.startswith('INF-') or val.startswith('*'):
        return 'INF'
    if val.isdigit():
        return 'EXC'
    if val.startswith('NIPHEM'):
        return 'NIPHEM'
    if val.startswith('NIPH'):
        return 'NIPH'
    return val  # LNF, ASM, ALM, PLOT3, PLOT5, LOTSC, PAMA

def transition_category(cb_class, cc_class):
    """Categorize a transition into a high-level group."""
    cb_callable = cb_class in ('EXC', 'INF')
    cc_callable = cc_class in ('EXC', 'INF')

    if cb_class == 'INF' and cc_class == 'EXC':
        return 'INF\u2192EXC\n(same sequence)'
    if cb_callable and not cc_callable:
        return 'cb calls,\ncc does not'
    if not cb_callable and cc_callable:
        return 'cc calls,\ncb does not'
    if cb_class in ('EXC', 'INF') and cc_class in ('EXC', 'INF') and cb_class != cc_class:
        return 'EXC\u2194INF\n(label only)'
    if cb_class in ('NIPH', 'NIPHEM') or cc_class in ('NIPH', 'NIPHEM'):
        return 'Paralog\ndifferences'
    return 'Other\nnon-callable'

# Dataset definitions
datasets = [
    ('Lm cons.', 'lm', 'cons'),
    ('Se cons.', 'se', 'cons'),
    ('Ec cons.', 'ec', 'cons'),
    ('Cj cons.', 'cj', 'cons'),
    ('Lm pub.', 'lm', 'pub'),
    ('Se pub.', 'se', 'pub'),
    ('Ec pub.', 'ec', 'pub'),
    ('Cj pub.', 'cj', 'pub'),
]

cc_paths = {
    ('lm', 'cons'): f"{BASE}/chewcall/lm_rep1/results_alleles.tsv",
    ('se', 'cons'): f"{BASE}/chewcall/se_rep1/results_alleles.tsv",
    ('ec', 'cons'): f"{BASE}/chewcall/ec_rep1/results_alleles.tsv",
    ('cj', 'cons'): f"{BASE}/chewcall/cj_rep1/results_alleles.tsv",
    ('lm', 'pub'): "/mnt/disk2/a.deruvo/chew_results/lm_public/rust_results/results_alleles.tsv",
    ('se', 'pub'): "/mnt/disk2/a.deruvo/chew_results/se_public/rust_results/results_alleles.tsv",
    ('ec', 'pub'): "/mnt/disk2/a.deruvo/chew_results/ec_public/rust_results/results_alleles.tsv",
    ('cj', 'pub'): "/mnt/disk2/a.deruvo/chew_results/cj_public/rust_results/results_alleles.tsv",
}
cb_paths = {
    ('lm', 'cons'): f"{BASE}/c354/shcds_cons_lm_rep1/results_alleles.tsv",
    ('se', 'cons'): f"{BASE}/c354/shcds_cons_se_rep1/results_alleles.tsv",
    ('ec', 'cons'): f"{BASE}/c354/shcds_cons_ec_rep1/results_alleles.tsv",
    ('cj', 'cons'): f"{BASE}/c354/shcds_cons_cj_rep1/results_alleles.tsv",
    ('lm', 'pub'): f"{BASE}/c354/shcds_pub_lm_rep1/results_alleles.tsv",
    ('se', 'pub'): f"{BASE}/c354/shcds_pub_se_rep1/results_alleles.tsv",
    ('ec', 'pub'): f"{BASE}/c354/shcds_pub_ec_rep1/results_alleles.tsv",
    ('cj', 'pub'): f"{BASE}/c354/shcds_pub_cj_rep1/results_alleles.tsv",
}

# Compute transitions
all_results = []
categories_order = [
    'INF\u2192EXC\n(same sequence)',
    'cb calls,\ncc does not',
    'cc calls,\ncb does not',
    'Paralog\ndifferences',
    'EXC\u2194INF\n(label only)',
    'Other\nnon-callable',
]

for label, org, ds in datasets:
    print(f"Processing {label}...")
    cc_data, cc_loci = load_alleles(cc_paths[(org, ds)])
    cb_data, cb_loci = load_alleles(cb_paths[(org, ds)])
    common_loci = sorted(set(cc_loci) & set(cb_loci))
    common_genomes = sorted(set(cc_data.keys()) & set(cb_data.keys()))

    concordant = 0
    cat_counts = {c: 0 for c in categories_order}

    for g in common_genomes:
        for l in common_loci:
            cc_val = cc_data[g].get(l, '')
            cb_val = cb_data[g].get(l, '')
            cc_class = classify_cell(cc_val)
            cb_class = classify_cell(cb_val)

            if cc_class == cb_class:
                concordant += 1
            else:
                cat = transition_category(cb_class, cc_class)
                if cat in cat_counts:
                    cat_counts[cat] += 1
                else:
                    cat_counts['Other\nnon-callable'] += 1

    total_discord = sum(cat_counts.values())
    total = concordant + total_discord
    print(f"  Concordant: {concordant:,} ({concordant/total*100:.2f}%), Discordant: {total_discord:,} ({total_discord/total*100:.2f}%)")
    for cat, cnt in cat_counts.items():
        if cnt > 0:
            print(f"    {cat.replace(chr(10), ' ')}: {cnt:,}")

    all_results.append({
        'label': label,
        'concordant': concordant,
        'total': total,
        'cats': cat_counts,
    })

# Plot: single panel showing discordance breakdown
fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))

labels = [r['label'] for r in all_results]
y = np.arange(len(labels))
height = 0.6

colors = {
    'INF\u2192EXC\n(same sequence)': '#3498db',     # blue
    'cb calls,\ncc does not': '#e67e22',              # orange
    'cc calls,\ncb does not': '#2ecc71',              # green
    'Paralog\ndifferences': '#9b59b6',                # purple
    'EXC\u2194INF\n(label only)': '#1abc9c',          # teal
    'Other\nnon-callable': '#bdc3c7',                 # gray
}

left = np.zeros(len(labels))
handles = []
for cat in categories_order:
    vals = np.array([r['cats'][cat] for r in all_results])
    if vals.sum() > 0:
        bars = ax.barh(y, vals, height, left=left, color=colors[cat], edgecolor='white', linewidth=0.5)
        handles.append((bars, cat.replace('\n', ' ')))
        left += vals

# Add concordance % annotation at end of each bar
for i, r in enumerate(all_results):
    total_disc = sum(r['cats'].values())
    conc_pct = r['concordant'] / r['total'] * 100
    ax.text(left[i] + max(left) * 0.02, y[i], f'{conc_pct:.2f}% concordant',
            ha='left', va='center', fontsize=7.5, color='#555555', style='italic')

ax.set_xlabel('Discordant cells (count)', fontsize=10)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_title('Classification discordance anatomy\n(chewcall vs chewBBACA v3.5.4, shared pre-computed CDS)', fontsize=11, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

ax.legend([h[0] for h in handles], [h[1] for h in handles],
          fontsize=8, loc='lower right', framealpha=0.9)

plt.tight_layout()

outpath = os.path.join(os.path.dirname(__file__), 'fig_discordance_anatomy.pdf')
fig.savefig(outpath, bbox_inches='tight', dpi=300)
print(f"\nFigure saved to {outpath}")

outpath_png = outpath.replace('.pdf', '.png')
fig.savefig(outpath_png, bbox_inches='tight', dpi=150)
print(f"PNG preview saved to {outpath_png}")
