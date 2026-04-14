<p align="center">
  <img src="logo.svg" alt="chewcall" width="120">
</p>

# chewcall

> [!WARNING]
> **Exploratory project — NOT for production use.**
> This is a research prototype developed to explore performance optimization strategies for allele calling. It has been validated on BeONE datasets but has **not** undergone the extensive testing and validation required for use in clinical or public health surveillance. For production use, please use the original [chewBBACA](https://github.com/B-UMMI/chewBBACA).

A high-performance allele caller for cgMLST/wgMLST schemas, inspired by and compatible with [chewBBACA](https://github.com/B-UMMI/chewBBACA).

**chewcall** reimplements the AlleleCall algorithm from chewBBACA in Rust, replacing BLASTp with SIMD-accelerated exact Smith-Waterman protein alignment via [parasail](https://github.com/jeffdaily/parasail), achieving **6–10x faster** allele calling with **fully deterministic, near-identical** results on 8 BeONE datasets (up to 3,076 genomes, 8,558 loci).

### Key features

- **Compatible** with chewBBACA schemas (Chewie-NS, PrepExternalSchema, CreateSchema)
- **6–10x faster** than chewBBACA v3.5.3 on multi-core systems (8 threads), **1.8–3.4x faster** than chewBBACA v3
- **Fully deterministic** — bit-identical results on every run
- **No external dependencies at runtime** — built-in pure-Rust gene prediction (prodigal-rs) and parasail SIMD alignment
- **Higher accuracy on truncated alleles** — 99.7–100% ASM sensitivity vs. 92.9–96.5% for chewBBACA (synthetic ground truth)
- **Zero different-allele cases** — across 59 million cells on 8 datasets, when both tools call an allele, they always call the same one
- **Identical outbreak clustering** — source attribution and cluster assignments match chewBBACA 100% on FDA benchmark datasets
- **Parallel everything**: schema loading, CDS prediction, deduplication, clustering, and SW alignment via [rayon](https://github.com/rayon-rs/rayon)
- **Optional GPU acceleration** via CUDA for large-scale datasets
- **Minimizer-based pre-filtering**: top-K cluster selection reduces alignment pairs by ~8x without affecting results
- All 11 chewBBACA classification classes: EXC, INF, PLOT3, PLOT5, LOTSC, NIPH, NIPHEM, ALM, ASM, PAMA, LNF

## Installation

### Requirements

- Rust 1.75+ (install via [rustup](https://rustup.rs/))
- [parasail](https://github.com/jeffdaily/parasail) (SIMD-accelerated Smith-Waterman library)
- Optional: CUDA 12+ and NVIDIA GPU (for `--gpu` mode)

### Build

```bash
# Build parasail (one-time)
git clone https://github.com/jeffdaily/parasail.git
cd parasail && mkdir build && cd build
cmake .. && make -j$(nproc)
cd ../..

# Standard build
RUSTFLAGS="-C target-cpu=native" cargo build --release

# With GPU support (requires CUDA)
CUDA_HOME=/usr/local/cuda RUSTFLAGS="-C target-cpu=native" cargo build --release

# Run (parasail must be in LD_LIBRARY_PATH)
LD_LIBRARY_PATH=/path/to/parasail/build ./target/release/chewcall [OPTIONS]
```

The binary is at `target/release/chewcall`.

## Usage

### Quick start

```bash
# Run allele calling (built-in CDS prediction via prodigal-rs)
chewcall \
    -i /path/to/genomes \
    -g /path/to/schema \
    -o /path/to/output \
    --cpu 8

# Or with pre-computed CDS (pyrodigal, for exact comparison with chewBBACA)
python predict_cds.py \
    -i /path/to/genomes \
    -g /path/to/schema \
    -o /path/to/cds_output

chewcall \
    -i /path/to/genomes \
    -g /path/to/schema \
    -o /path/to/output \
    --cpu 8 \
    --cds-input /path/to/cds_output
```

### Full options

```
chewcall [OPTIONS] -i <INPUT> -g <SCHEMA> -o <OUTPUT>

Options:
  -i, --input <INPUT>           Input directory with genome FASTA files
  -g, --schema <SCHEMA>         Schema directory (chewBBACA format)
  -o, --output <OUTPUT>         Output directory
      --cpu <CPU>               Number of CPU threads [default: 1]
      --cds-input <CDS_INPUT>   Pre-computed CDS directory (skip built-in prediction)
      --mode <MODE>             Alignment mode: "fast" (parasail) or "compatible" (BLAST) [default: fast]
      --blastp-path <PATH>      Path to blastp binary (required for --mode compatible)
      --gpu                     Use GPU (CUDA) for Smith-Waterman alignment
      --update-schema           Append novel alleles (INF) to schema FASTA files in place
      --bsr <BSR>               BLAST Score Ratio threshold [default: 0.6]
      --size-threshold <SIZE>   Size threshold for ASM/ALM [default: 0.2]
      --min-length <MIN>        Minimum sequence length [default: 0]
  -t, --translation-table <TT>  Genetic code [default: 11]
      --prodigal-mode <MODE>    Prodigal mode: single or meta [default: single]
```

### CDS prediction modes

chewcall supports three CDS prediction modes:

1. **Built-in prodigal-rs** (default) — Pure Rust reimplementation of Prodigal 2.6.3 (single mode). No external dependencies. Uses the `.trn` training file from the schema directory.
2. **Pre-computed CDS** (`--cds-input`) — Reads CDS from a directory of FASTA files pre-computed with pyrodigal or prodigal. Recommended for exact comparison with chewBBACA.
3. **External prodigal** (`--prodigal-path`) — Spawns prodigal as a subprocess for each genome.

### Schema compatibility

chewcall works with any schema in the standard chewBBACA format:

```
schema/
├── locus1.fasta          # Full allele sequences
├── locus2.fasta
├── short/
│   ├── locus1_short.fasta  # Representative alleles
│   └── locus2_short.fasta
└── *.trn                 # Prodigal training file
```

Schemas can be downloaded from [Chewie-NS](https://chewbbaca.online/) or prepared with chewBBACA's `PrepExternalSchema` / `CreateSchema`.

```bash
chewBBACA.py DownloadSchema -sp <species_id> -sc <schema_id> -o schema_dir
```

### Output files

| File | Description |
|------|-------------|
| `results_alleles.tsv` | Allelic profiles (locus x genome matrix) |
| `results_alleles_hashed.tsv` | CRC32-hashed allelic profiles |
| `results_statistics.tsv` | Per-genome classification statistics |
| `loci_summary_stats.tsv` | Per-locus classification counts |
| `results_contigsInfo.tsv` | CDS coordinates on contigs |
| `novel_alleles.fasta` | Novel allele sequences (INF) |

## Validation

Validated on 8 [BeONE](https://onehealthejp.eu/projects/foodborne-zoonoses/jrp-beone) datasets (4 consortium + 4 public), comparing chewcall (fast mode, parasail) vs chewBBACA v3.5.3. Both tools use the same pre-computed CDS (pyrodigal) to ensure identical gene predictions. CRC32-hashed allelic profiles are compared cell-by-cell; CRC32 hashing maps each allele to the hash of its DNA sequence, making the comparison independent of allele ID numbering.

### wgMLST (all loci)

| Dataset | Organism | Genomes | Loci | Cells | Diffs | CRC32 match |
|---------|----------|---------|------|-------|-------|-------------|
| Consortium | *L. monocytogenes* | 1,426 | 1,748 (cgMLST) | 2,492,648 | 7 | **99.9997%** |
| Consortium | *S. enterica* | 1,540 | 8,558 (wgMLST) | 13,179,320 | 817 | 99.9938% |
| Consortium | *E. coli* | 308 | 7,601 (wgMLST) | 2,341,108 | 488 | 99.9792% |
| Consortium | *C. jejuni* | 610 | 2,794 (wgMLST) | 1,704,340 | 1,137 | 99.9333% |
| Public | *L. monocytogenes* | 1,874 | 1,748 (cgMLST) | 3,275,752 | 26 | **99.9992%** |
| Public | *S. enterica* | 1,434 | 8,558 (wgMLST) | 12,272,172 | 2,479 | 99.9798% |
| Public | *E. coli* | 1,999 | 7,601 (wgMLST) | 15,194,399 | 5,073 | 99.9666% |
| Public | *C. jejuni* | 3,076 | 2,794 (wgMLST) | 8,594,344 | 5,925 | 99.9311% |

### Actionable differences

A key finding is that **when both tools call an allele, they always call the same one**: across all 59 million cells in 8 datasets, there are zero cases where both tools produce a CRC32 hash but with different values. All differences are exclusively of the "called vs. not-called" type — one tool finds an allele while the other classifies the locus as missing.

| Dataset | Organism | Cells | cc calls | cb calls | Different allele |
|---------|----------|-------|----------|----------|-----------------|
| Cons. | *L. monocytogenes* | 2,492,648 | 1 | 6 | **0** |
| Cons. | *S. enterica* | 13,179,320 | 463 | 354 | **0** |
| Cons. | *E. coli* | 2,341,108 | 336 | 152 | **0** |
| Cons. | *C. jejuni* | 1,704,340 | 1,006 | 131 | **0** |
| Pub. | *L. monocytogenes* | 3,275,752 | 5 | 21 | **0** |
| Pub. | *S. enterica* | 12,272,172 | 1,910 | 569 | **0** |
| Pub. | *E. coli* | 15,194,399 | 3,266 | 1,807 | **0** |
| Pub. | *C. jejuni* | 8,594,344 | 5,228 | 697 | **0** |

### Outbreak clustering

On 4 FDA Gen-FS Gopher benchmark datasets (*L. monocytogenes* stone fruit recall, *Salmonella* Bareilly tuna scrape, *E. coli* O121:H19, *C. jejuni* raw milk), both tools produce **identical MST and single-linkage cluster assignments** on all four organisms, with **zero outgroup intrusions**.

Source attribution on all 4 BeONE consortium datasets (2,570 patient isolates, 1,020 non-human sources) produces **identical nearest-source assignments and cluster memberships** in 100% of cases.

### Synthetic ground truth

On 466,440 synthetic cells with BSR-calibrated mutations across 4 organisms, chewcall achieves significantly higher accuracy on *L. monocytogenes* (McNemar's *p* < 10⁻¹⁵) and *E. coli* (*p* = 0.031). BLAST's sensitivity loss is concentrated on truncated alleles (ASM class: 92.9–96.5% vs. 99.7–100% for chewcall).

### Why are there remaining wgMLST differences?

chewBBACA uses **BLASTp** for protein alignment, while chewcall uses **parasail Smith-Waterman** (BLOSUM62, gap_open=11, gap_extend=1). Both use the same scoring matrix and gap penalties, but BLAST employs **database-size-dependent heuristics** (E-value thresholds, word seeding) that parasail's exact Smith-Waterman does not.

The remaining differences arise from:

1. **Heuristic hit loss** (83% of differences) — BLAST's word-seeding heuristics occasionally miss valid alignments that exact Smith-Waterman finds, particularly on short or divergent query proteins (truncated alleles).

2. **Cascading novel allele effects** — When one tool discovers a novel allele (INF) that the other misses, subsequent genomes can match that novel allele. A single borderline difference in one genome can cascade into multiple discordant cells across other genomes for the same locus.

3. **Accessory loci are noisier** — Accessory loci (present in <95% of genomes) are inherently more variable and have weaker matches to schema representatives. All differences are confined to accessory loci and do not affect cgMLST-based epidemiological analysis.

## Performance

### vs. chewBBACA v3.5.3

Benchmarked on 8 [BeONE](https://onehealthejp.eu/projects/foodborne-zoonoses/jrp-beone) datasets (8 CPU threads). Both tools use the same pre-computed CDS ([pyrodigal](https://github.com/althonos/pyrodigal)).

| Dataset | Organism | Genomes | Loci | chewBBACA | chewcall | Speedup |
|---------|----------|---------|------|-----------|----------|---------|
| Consortium | *L. monocytogenes* | 1,426 | 1,748 | 148s | 14.4s | **10.3x** |
| Consortium | *S. enterica* | 1,540 | 8,558 | 599s | 66.6s | **9.0x** |
| Consortium | *E. coli* | 308 | 7,601 | 567s | 59.5s | **9.5x** |
| Consortium | *C. jejuni* | 610 | 2,794 | 214s | 22.1s | **9.7x** |
| Public | *L. monocytogenes* | 1,874 | 1,748 | 206s | 22.4s | **9.2x** |
| Public | *S. enterica* | 1,434 | 8,558 | 687s | 93.2s | **7.4x** |
| Public | *E. coli* | 1,999 | 7,601 | 1,586s | 259.2s | **6.1x** |
| Public | *C. jejuni* | 3,076 | 2,794 | 473s | 65.4s | **7.2x** |

#### Benchmark environment

| Component | Specification |
|-----------|---------------|
| CPU | 2x Intel Xeon Gold 6542Y (80 cores total) |
| RAM | 504 GB DDR5 |
| GPU | NVIDIA L4 (24 GB VRAM) |
| OS | AlmaLinux 10.1, kernel 6.12 |
| Rust | 1.85 (cargo build --release, target-cpu=native) |
| chewBBACA | v3.5.3 (Python 3.11, BLAST+ 2.16) |

### vs. chewBBACA v3

Compared on the scalability benchmark from Mamede et al. (2026, *Genome Medicine*), using their publicly available datasets and cgMLST schemas (8 CPU threads for chewcall, 6 for chewBBACA 3 as published).

| Organism | Loci | Genomes | chewcall | chewBBACA 3 | Speedup |
|----------|------|---------|----------|-------------|---------|
| *L. monocytogenes* | 2,449 | 128 | 29s | 99s | **3.4x** |
| | | 4,096 | 724s | 1,507s | **2.1x** |
| *S. enterica* | 3,192 | 128 | 63s | 230s | **3.6x** |
| | | 4,096 | 1,810s | 3,393s | **1.9x** |
| *S. pyogenes* | 1,345 | 128 | 19s | 54s | **2.8x** |
| | | 4,096 | 420s | 767s | **1.8x** |

On all three organisms, chewcall and chewBBACA 3 produce **identical pairwise cgMLST distances** (0 disagreements across 3,675 genome pairs).

## Algorithm

chewcall follows the same pipeline as chewBBACA AlleleCall:

1. **Schema loading** — Parallel FASTA parsing, SHA-256 hashing, CRC32 computation
2. **CDS prediction** — Built-in prodigal-rs (pure Rust, default), or pre-computed CDS, or external prodigal
3. **Deduplication** — SHA-256 dedup across all genomes
4. **Exact DNA matching** — Hash lookup against schema alleles
5. **Translation + exact protein matching** — Hash lookup of translated CDS
6. **Clustering + Smith-Waterman** — Minimizer-based pre-filter (k=5, w=5) + BLOSUM62 SW alignment + BSR scoring
7. **Representative determination** — Iterative expansion: borderline BSR candidates (0.6–0.7) become new representatives, re-index, re-align until convergence
8. **Classification** — INF, EXC, ASM, ALM, PLOT3, PLOT5, LOTSC, NIPH, NIPHEM, PAMA, LNF
9. **Output** — TSV profiles, CRC32-hashed profiles, statistics, novel alleles

### Differences from chewBBACA

- **SIMD Smith-Waterman** via [parasail](https://github.com/jeffdaily/parasail) (AVX2/SSE4.1) replaces BLASTp. Same BLOSUM62 matrix and affine gap penalties (open=11, extend=1). Exact optimal scores, no heuristic approximations.
- **Minimizer pre-filter** replaces BLASTp's internal word seeding. Top-5 candidates per query by shared minimizer count (k=5, w=5 validated as optimal across all organisms by ablation study).
- **Built-in CDS prediction** via prodigal-rs (pure Rust Prodigal 2.6.3). No Python/pyrodigal dependency required.
- **No BLAST dependency** — only requires parasail shared library (optional `--mode compatible` uses external BLAST).
- **Read-only schema by default** — novel alleles go to `novel_alleles.fasta`; use `--update-schema` to append them to the schema.
- **Full determinism** — bit-identical output across runs regardless of system load or thread scheduling.
- **Optional GPU mode** via CUDA for large-scale alignment batches.

## Limitations

- **AlleleCall only** — chewcall reimplements only the AlleleCall algorithm. Schema creation, evaluation, and other chewBBACA modules are not included.
- **Read-only schema by default** — novel alleles (INF) are written to `novel_alleles.fasta` in the output directory. Use `--update-schema` to also append them to the schema FASTA files in place (matching chewBBACA behavior).
- **GPU mode** — experimental CUDA support is included but not yet production-ready for very large batches.
- **Not a fork** — this is an independent reimplementation inspired by chewBBACA, not a fork of the original Python codebase.

## Acknowledgments

chewcall is inspired by the allele calling algorithm of **chewBBACA** by Silva et al. The classification logic, BSR-based scoring, representative determination, and output format are all derived from the original implementation. We are grateful to the chewBBACA team for their excellent tool and for making schemas publicly available via Chewie-NS.

Benchmark datasets are from the [BeONE](https://onehealthejp.eu/projects/foodborne-zoonoses/jrp-beone) project (One Health EJP).

## References

- Silva M, Machado MP, Silva DN, et al. (2018). **chewBBACA: A complete suite for gene-by-gene schema creation and strain identification.** *Microbial Genomics*, 4(3). DOI: [10.1099/mgen.0.000166](https://doi.org/10.1099/mgen.0.000166)
- Mamede R, Silva M, Machado MP, et al. (2026). **chewBBACA 3: lowering the barrier for scalable and detailed whole- and core-genome multilocus sequence typing.** *Genome Medicine*, 18:50. DOI: [10.1186/s13073-026-01625-x](https://doi.org/10.1186/s13073-026-01625-x)
- Silva M, Rossi M, Moran-Gilad J, et al. (2024). **Chewie Nomenclature Server (chewie-NS): a deployable nomenclature server for easy sharing of core and whole genome MLST schemas.** *Nucleic Acids Research*, 52(D1), D733–D738. DOI: [10.1093/nar/gkad957](https://doi.org/10.1093/nar/gkad957)
- Daily J. (2016). **Parasail: SIMD C library for global, semi-global, and local pairwise sequence alignments.** *BMC Bioinformatics*, 17:81. DOI: [10.1186/s12859-016-0930-z](https://doi.org/10.1186/s12859-016-0930-z)
- Larivière M, Allard MW, Nachman RE, et al. (2022). **BeONE: An integrated dataset of assembled genomes from foodborne pathogens.** *Zenodo*. DOI: [10.5281/zenodo.7802702](https://doi.org/10.5281/zenodo.7802702)

## License

GPL-3.0 — same as the original [chewBBACA](https://github.com/B-UMMI/chewBBACA).

## Authors

GenPat Team — [Istituto Zooprofilattico Sperimentale dell'Abruzzo e del Molise](https://www.izs.it/)
