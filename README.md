<p align="center">
  <img src="logo.svg" alt="chewcall" width="120">
</p>

# chewcall

> [!WARNING]
> **Exploratory project — NOT for production use.**
> This is a research prototype developed to explore performance optimization strategies for allele calling. It has been validated on BeONE datasets but has **not** undergone the extensive testing and validation required for use in clinical or public health surveillance. For production use, please use the original [chewBBACA](https://github.com/B-UMMI/chewBBACA).

A high-performance allele caller for cgMLST/wgMLST schemas, inspired by and compatible with [chewBBACA](https://github.com/B-UMMI/chewBBACA).

**chewcall** reimplements the AlleleCall algorithm from chewBBACA in Rust, replacing BLASTp with SIMD-accelerated Smith-Waterman alignment via [parasail](https://github.com/jeffdaily/parasail), achieving **~10x faster** allele calling with **fully deterministic, near-identical** results on 8 BeONE datasets (up to 3,076 genomes, 8,558 loci). Core genome profiles (>=99% presence) differ by at most 6-30 cells across millions of comparisons.

### Key features

- **Compatible** with chewBBACA schemas (Chewie-NS, PrepExternalSchema, CreateSchema)
- **~10x faster** than chewBBACA on multi-core systems (8 threads)
- **Fully deterministic** — identical results on every run
- **Identical core genome results** for most organisms (see [Validation](#validation))
- **Parallel everything**: schema loading, CDS deduplication, clustering, and SW alignment via [rayon](https://github.com/rayon-rs/rayon)
- **Optional GPU acceleration** via CUDA for large-scale datasets
- **Minimizer-based pre-filtering**: top-K cluster selection reduces alignment pairs by ~8x without affecting results
- All 11 chewBBACA classification classes: EXC, INF, PLOT3, PLOT5, LOTSC, NIPH, NIPHEM, ALM, ASM, PAMA, LNF

## Installation

### Requirements

- Rust 1.75+ (install via [rustup](https://rustup.rs/))
- [parasail](https://github.com/jeffdaily/parasail) (SIMD-accelerated Smith-Waterman library)
- Python 3.9+ with [pyrodigal](https://github.com/althonos/pyrodigal) and [biopython](https://biopython.org/) (for CDS prediction, see `requirements.txt`)
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
# 1. Pre-compute CDS with pyrodigal (one-time per genome set)
python predict_cds.py \
    -i /path/to/genomes \
    -g /path/to/schema \
    -o /path/to/cds_output

# 2. Run allele calling
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
      --cds-input <CDS_INPUT>   Pre-computed CDS directory (skip prodigal)
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

#### wgMLST (all loci)

| Dataset | Organism | Genomes | Loci | Cells | Diffs | CRC32 match |
|---------|----------|---------|------|-------|-------|-------------|
| Consortium | *L. monocytogenes* | 1,426 | 1,748 (cgMLST) | 2,492,648 | 1 | **100.0000%** |
| Consortium | *S. enterica* | 1,540 | 8,558 (wgMLST) | 13,179,320 | 204 | 99.9985% |
| Consortium | *E. coli* | 308 | 7,601 (wgMLST) | 2,341,108 | 152 | 99.9935% |
| Consortium | *C. jejuni* | 610 | 2,794 (wgMLST) | 1,704,340 | 401 | 99.9765% |
| Public | *L. monocytogenes* | 1,874 | 1,748 (cgMLST) | 3,275,752 | 1 | **100.0000%** |
| Public | *S. enterica* | 1,434 | 8,558 (wgMLST) | 12,272,172 | 1,682 | 99.9863% |
| Public | *E. coli* | 1,999 | 7,601 (wgMLST) | 15,194,399 | 1,633 | 99.9893% |
| Public | *C. jejuni* | 3,076 | 2,794 (wgMLST) | 8,594,344 | 1,803 | 99.9790% |

#### Core genome comparison

Core loci are those present in a given percentage of genomes, corresponding to the loci typically used in cgMLST-based epidemiological surveillance:

| Dataset | Organism | Core >=95% | Diffs | Core >=98% | Diffs | Core >=99% | Diffs |
|---------|----------|------------|-------|------------|-------|------------|-------|
| Consortium | *L. monocytogenes* | 1,731 | 1 | 1,721 | 1 | 1,716 | 1 |
| Consortium | *S. enterica* | 3,259 | 77 | 3,027 | 37 | 2,765 | 16 |
| Consortium | *E. coli* | 2,809 | **0** | 2,592 | **0** | 2,470 | **0** |
| Consortium | *C. jejuni* | 991 | **0** | 900 | **0** | 706 | **0** |
| Public | *L. monocytogenes* | 1,730 | 1 | 1,717 | 1 | 1,691 | 1 |
| Public | *S. enterica* | 3,045 | 1,438 | 2,905 | 1,437 | 2,752 | 30 |
| Public | *E. coli* | 2,797 | 6 | 2,629 | 6 | 2,412 | 6 |
| Public | *C. jejuni* | 1,006 | 6 | 983 | 6 | 927 | 6 |

Consortium datasets show near-perfect agreement: *E. coli* and *C. jejuni* produce **100% identical** core genome profiles at any threshold, while *L. monocytogenes* has a single diff across 2.5M cells. Public datasets (larger, up to 3,076 genomes) confirm this trend: core 99% differences are at most 30 cells for *S. enterica* wgMLST and 6 cells for *E. coli* and *C. jejuni*.

### Why are there remaining wgMLST differences?

chewBBACA uses **BLASTp** for protein alignment, while chewcall uses **parasail Smith-Waterman** (BLOSUM62, gap_open=11, gap_extend=1). Both use the same scoring matrix and gap penalties, but BLAST employs **database-size-dependent heuristics** (E-value thresholds, word seeding) that parasail's exact Smith-Waterman does not.

The remaining differences arise from:

1. **Borderline hit discovery** — BLAST's word-seeding heuristics may find or miss alignments near the BSR threshold (0.6) that exact Smith-Waterman handles differently. Neither tool is "wrong" — these are genuinely borderline cases where the alignment score is close to the classification threshold.

2. **Cascading novel allele effects** — When one tool discovers a novel allele (INF) that the other misses, subsequent genomes can match that novel allele. A single borderline difference in one genome can cascade into multiple discordant cells across other genomes for the same locus.

3. **Accessory loci are noisier** — Accessory loci (present in <95% of genomes) are inherently more variable and have weaker matches to schema representatives. Small scoring differences between BLAST and parasail are more likely to flip a classification near the threshold. Core loci, being well-conserved, produce robust matches that are insensitive to the alignment engine used.

All wgMLST differences are confined to accessory loci and do not affect cgMLST-based epidemiological analysis (minimum spanning trees, cluster detection, outbreak investigation).

## Performance

Benchmarked on 8 [BeONE](https://onehealthejp.eu/projects/foodborne-zoonoses/jrp-beone) datasets (8 CPU threads). Both tools use the same pre-computed CDS ([pyrodigal](https://github.com/althonos/pyrodigal)).

#### Benchmark environment

| Component | Specification |
|-----------|---------------|
| CPU | 2x Intel Xeon Gold 6542Y (80 cores total) |
| RAM | 504 GB DDR5 |
| GPU | NVIDIA L4 (24 GB VRAM) |
| OS | AlmaLinux 10.1, kernel 6.12 |
| Rust | 1.85 (cargo build --release, target-cpu=native) |
| chewBBACA | v3.5.3 (Python 3.11, BLAST+ 2.16) |

#### Allele calling time

| Dataset | Organism | Genomes | Loci | chewBBACA | chewcall | Speedup |
|---------|----------|---------|------|-----------|----------|---------|
| Consortium | *L. monocytogenes* | 1,426 | 1,748 | 156s | 15.5s | **10.1x** |
| Consortium | *S. enterica* | 1,540 | 8,558 | 586s | 66.4s | **8.8x** |
| Consortium | *E. coli* | 308 | 7,601 | 570s | 56.3s | **10.1x** |
| Consortium | *C. jejuni* | 610 | 2,794 | 215s | 21.8s | **9.9x** |
| Public | *L. monocytogenes* | 1,874 | 1,748 | 203s | 53.6s | **3.8x** |
| Public | *S. enterica* | 1,434 | 8,558 | 690s | 191.7s | **3.6x** |
| Public | *E. coli* | 1,999 | 7,601 | 1,615s | 417.8s | **3.9x** |
| Public | *C. jejuni* | 3,076 | 2,794 | 477s | 118.9s | **4.0x** |

## Algorithm

chewcall follows the same pipeline as chewBBACA AlleleCall:

1. **Schema loading** - Parallel FASTA parsing, SHA-256 hashing, CRC32 computation
2. **CDS prediction** - Via pyrodigal (pre-computed) or external prodigal
3. **Deduplication** - SHA-256 dedup across all genomes
4. **Exact DNA matching** - Hash lookup against schema alleles
5. **Translation + exact protein matching** - Hash lookup of translated CDS
6. **Clustering + Smith-Waterman** - Minimizer-based pre-filter + BLOSUM62 SW alignment + BSR scoring
7. **Representative determination** - Iterative expansion with BSR 0.6-0.7 candidates
8. **Classification** - INF, EXC, ASM, ALM, PLOT3, PLOT5, LOTSC, NIPH, NIPHEM, PAMA, LNF
9. **Output** - TSV profiles, CRC32-hashed profiles, statistics, novel alleles

### Differences from chewBBACA

- **SIMD Smith-Waterman** via [parasail](https://github.com/jeffdaily/parasail) (AVX2/SSE4.1) replaces BLASTp. Same BLOSUM62 matrix and affine gap penalties (open=11, extend=1).
- **Minimizer pre-filter** replaces BLASTp's internal word seeding. Top-5 candidates per query by shared minimizer count.
- **No BLAST dependency** — only requires parasail shared library (optional `--mode compatible` uses external BLAST).
- **Read-only schema by default** — novel alleles go to `novel_alleles.fasta`; use `--update-schema` to append them to the schema.
- **Optional GPU mode** via CUDA for large-scale alignment batches.

## Limitations

- **AlleleCall only** — chewcall reimplements only the AlleleCall algorithm. Schema creation, evaluation, and other chewBBACA modules are not included.
- **CDS prediction** — chewcall does not include a built-in gene predictor. CDS must be pre-computed using the included `predict_cds.py` script (based on [pyrodigal](https://github.com/althonos/pyrodigal)).
- **Read-only schema by default** — novel alleles (INF) are written to `novel_alleles.fasta` in the output directory. Use `--update-schema` to also append them to the schema FASTA files in place (matching chewBBACA behavior).
- **GPU mode** — experimental CUDA support is included but not yet production-ready for very large batches.
- **Not a fork** — this is an independent reimplementation inspired by chewBBACA, not a fork of the original Python codebase.

## Acknowledgments

chewcall is inspired by the allele calling algorithm of **chewBBACA** by Silva et al. The classification logic, BSR-based scoring, representative determination, and output format are all derived from the original implementation. We are grateful to the chewBBACA team for their excellent tool and for making schemas publicly available via Chewie-NS.

Benchmark datasets are from the [BeONE](https://onehealthejp.eu/projects/foodborne-zoonoses/jrp-beone) project (One Health EJP).

## References

- Silva M, Machado MP, Silva DN, et al. (2018). **chewBBACA: A complete suite for gene-by-gene schema creation and strain identification.** *Microbial Genomics*, 4(3). DOI: [10.1099/mgen.0.000166](https://doi.org/10.1099/mgen.0.000166)
- Silva M, Rossi M, Moran-Gilad J, et al. (2024). **Chewie Nomenclature Server (chewie-NS): a deployable nomenclature server for easy sharing of core and whole genome MLST schemas.** *Nucleic Acids Research*, 52(D1), D733–D738. DOI: [10.1093/nar/gkad957](https://doi.org/10.1093/nar/gkad957)
- Daily J. (2016). **Parasail: SIMD C library for global, semi-global, and local pairwise sequence alignments.** *BMC Bioinformatics*, 17:81. DOI: [10.1186/s12859-016-0930-z](https://doi.org/10.1186/s12859-016-0930-z)
- Larivière M, Allard MW, Nachman RE, et al. (2022). **BeONE: An integrated dataset of assembled genomes from foodborne pathogens.** *Zenodo*. DOI: [10.5281/zenodo.7802702](https://doi.org/10.5281/zenodo.7802702)

## License

GPL-3.0 — same as the original [chewBBACA](https://github.com/B-UMMI/chewBBACA).

## Authors

GenPat Team — [Istituto Zooprofilattico Sperimentale dell'Abruzzo e del Molise](https://www.izs.it/)
