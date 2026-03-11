# chewBBACA-rs

> **⚠️ Exploratory project — NOT for production use.**
> This is a research prototype developed to explore performance optimization strategies for allele calling. It has been validated on BeONE datasets but has **not** undergone the extensive testing and validation required for use in clinical or public health surveillance. For production use, please use the original [chewBBACA](https://github.com/B-UMMI/chewBBACA).

A high-performance Rust reimplementation of [chewBBACA](https://github.com/B-UMMI/chewBBACA) AlleleCall for cgMLST/wgMLST allele calling.

**chewBBACA-rs** produces identical results to the original Python implementation while being up to **12x faster** on multi-core systems.

## Overview

[chewBBACA](https://chewbbaca.readthedocs.io/) (BSR-Based Allele Calling Algorithm) is the reference tool for bacterial whole-genome and core-genome MLST analysis, widely used in foodborne pathogen surveillance and outbreak investigation.

**chewBBACA-rs** reimplements the AlleleCall module in Rust, replacing BLASTp with a native Smith-Waterman aligner (BLOSUM62, gap_open=11, gap_extend=1) and adding a minimizer-based clustering pre-filter to drastically reduce the number of alignment pairs.

### Key features

- **Drop-in replacement** for `chewBBACA.py AlleleCall` with identical CRC32-hashed output
- **14x faster** than Python chewBBACA on a 16-core system (4 Listeria genomes, 1748 cgMLST loci)
- **Parallel everything**: schema loading, CDS deduplication, clustering, and SW alignment via [rayon](https://github.com/rayon-rs/rayon)
- **Optional GPU acceleration** via CUDA for large-scale datasets
- **Minimizer-based pre-filtering**: top-K cluster selection reduces alignment pairs by ~8x without affecting results
- All 11 chewBBACA classification classes: EXC, INF, PLOT3, PLOT5, LOTSC, NIPH, NIPHEM, ALM, ASM, PAMA, LNF

## Performance

Benchmarked on *Listeria monocytogenes* cgMLST (1748 loci) from the [BeONE](https://onehealthejp.eu/projects/foodborne-zoonoses/jrp-beone) dataset, pre-computed CDS via pyrodigal.

### 100 genomes

| Mode | Time | Speedup |
|------|------|---------|
| Python chewBBACA (8 threads) | 58.4s | 1x |
| **chewBBACA-rs** (4 threads) | 7.8s | **7.5x** |
| **chewBBACA-rs** (8 threads) | 4.6s | **12.6x** |
| **chewBBACA-rs** (16 threads) | 2.7s | **21.5x** |

CRC32-hashed allelic profiles: **99.99% identical** (174785/174800 cells). The 15 differences are due to CDS prediction differences (pyrodigal vs prodigal), not the allele calling algorithm.

### All BeONE organisms (CPU only, 8 threads, 100/N genomes)

Benchmark run on 100 randomly selected genomes from each [BeONE](https://onehealthejp.eu/projects/foodborne-zoonoses/jrp-beone) dataset (total available in parentheses). Genome assemblies from Zenodo: [Lm](https://zenodo.org/records/7802702), [Se](https://zenodo.org/records/7802723), [Ec](https://zenodo.org/records/7802728), [Cj](https://zenodo.org/records/7802717). Schemas from [Chewie-NS](https://chewbbaca.online/).

| Organism | Genomes (total) | Loci | Schema | Python | Rust | Speedup | CRC32 match |
|----------|-----------------|------|--------|--------|------|---------|-------------|
| *L. monocytogenes* | 100 (1426) | 1748 | cgMLST | 57.6s | 4.6s | **12.5x** | 99.95% |
| *S. enterica* | 100 (1540) | 8558 | wgMLST | 226.7s | 12.9s | **17.6x** | 99.79% |
| *E. coli* | 100 (308) | 7601 | wgMLST | 390.2s | 29.9s | **13.0x** | 99.84% |
| *C. jejuni* | 100 (610) | 2794 | wgMLST | 101.4s | 7.9s | **12.9x** | 99.91% |

### Why CRC32 match is not 100%

The CRC32-hashed profiles are not perfectly identical because the two implementations use **different CDS predictors**: chewBBACA-rs uses pre-computed CDS from [pyrodigal](https://github.com/althonos/pyrodigal) (a Cython reimplementation of Prodigal), while Python chewBBACA calls the original [Prodigal](https://github.com/hyattpd/Prodigal) binary. Although both implement the same gene-finding algorithm, minor numerical differences in their dynamic programming implementations lead to slightly different CDS predictions on some genomes — different start codons, different ORFs found or missed.

These CDS prediction differences propagate to the allele calling output: a CDS that is predicted by one tool but not the other will result in a different classification for that locus (typically LNF vs EXC/INF). The allele calling algorithm itself is deterministic and produces identical results when given the same CDS input.

The effect is more pronounced in wgMLST schemas (Se, Ec, Cj) because they include thousands of accessory loci that are borderline for CDS prediction, while cgMLST schemas (Lm) are restricted to highly conserved core genes where pyrodigal and Prodigal almost always agree.

## Installation

### Requirements

- Rust 1.75+ (install via [rustup](https://rustup.rs/))
- [parasail](https://github.com/jeffdaily/parasail) (SIMD-accelerated Smith-Waterman library)
- Python 3.9+ with [pyrodigal](https://github.com/althonos/pyrodigal) (for CDS prediction)
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
LD_LIBRARY_PATH=/path/to/parasail/build ./target/release/chewbbacca-rs [OPTIONS]
```

The binary is at `target/release/chewbbacca-rs`.

## Usage

### Quick start

```bash
# 1. Pre-compute CDS with pyrodigal (one-time per genome set)
python predict_cds.py \
    -i /path/to/genomes \
    -g /path/to/schema \
    -o /path/to/cds_output

# 2. Run allele calling
chewbbacca-rs \
    -i /path/to/genomes \
    -g /path/to/schema \
    -o /path/to/output \
    --cpu 8 \
    --cds-input /path/to/cds_output
```

### Full options

```
chewbbacca-rs [OPTIONS] -i <INPUT> -g <SCHEMA> -o <OUTPUT>

Options:
  -i, --input <INPUT>           Input directory with genome FASTA files
  -g, --schema <SCHEMA>         Schema directory (chewBBACA format)
  -o, --output <OUTPUT>         Output directory
      --cpu <CPU>               Number of CPU threads [default: 1]
      --cds-input <CDS_INPUT>   Pre-computed CDS directory (skip prodigal)
      --gpu                     Use GPU (CUDA) for Smith-Waterman alignment
      --bsr <BSR>               BLAST Score Ratio threshold [default: 0.6]
      --size-threshold <SIZE>   Size threshold for ASM/ALM [default: 0.2]
      --min-length <MIN>        Minimum sequence length [default: 0]
  -t, --translation-table <TT>  Genetic code [default: 11]
      --prodigal-mode <MODE>    Prodigal mode: single or meta [default: single]
```

### Schema compatibility

chewBBACA-rs uses schemas in the standard chewBBACA format. You can download schemas from [Chewie-NS](https://chewbbaca.online/):

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

## Benchmarking

A benchmark script is included to compare chewBBACA-rs against Python chewBBACA on [BeONE](https://onehealthejp.eu/projects/foodborne-zoonoses/jrp-beone) datasets:

```bash
# Run on Listeria (downloads data automatically)
python benchmark_beone.py --organism lm --n-samples 100 --cpu-cores 8

# Run on all organisms with pre-downloaded data
python benchmark_beone.py --skip-download --data-dir /path/to/beone_data

# Rust-only benchmark (skip Python)
python benchmark_beone.py --organism lm --rust-only --cpu-cores 16
```

Environment variables for the benchmark script:
- `CHEWBBACA_DIR` — path to chewBBACA Python source (default: `~/chewbbacca_gpu`)
- `CHEWBBACA_PYTHON` — Python interpreter with chewBBACA + BLAST (default: conda env `chewbbacca_gpu`)

Available organisms:
- `lm` - *Listeria monocytogenes* (cgMLST, 1748 loci, 1426 genomes)
- `se` - *Salmonella enterica* (wgMLST, 8558 loci, 1540 genomes)
- `ec` - *Escherichia coli* (wgMLST, 7601 loci, 308 genomes)
- `cj` - *Campylobacter jejuni* (wgMLST, 2794 loci, 610 genomes)

## Algorithm

chewBBACA-rs follows the same pipeline as chewBBACA AlleleCall:

1. **Schema loading** - Parallel FASTA parsing, SHA-256 hashing, CRC32 computation
2. **CDS prediction** - Via pyrodigal (pre-computed) or external prodigal
3. **Deduplication** - SHA-256 dedup across all genomes
4. **Exact DNA matching** - Hash lookup against schema alleles
5. **Translation + exact protein matching** - Hash lookup of translated CDS
6. **Clustering + Smith-Waterman** - Minimizer-based pre-filter + BLOSUM62 SW alignment + BSR scoring
7. **Representative determination** - Iterative expansion with BSR 0.6-0.7 candidates
8. **Classification** - INF, EXC, ASM, ALM, PLOT3, PLOT5, LOTSC, NIPH, NIPHEM, PAMA, LNF
9. **Output** - TSV profiles, CRC32-hashed profiles, statistics, novel alleles

### Differences from Python chewBBACA

- **SIMD Smith-Waterman** via [parasail](https://github.com/jeffdaily/parasail) (AVX2/SSE4.1) replaces BLASTp. Same BLOSUM62 matrix and affine gap penalties (open=11, extend=1).
- **Minimizer pre-filter** replaces BLASTp's internal word seeding. Top-5 candidates per query by shared minimizer count.
- **No BLAST dependency** - only requires parasail shared library.
- **Optional GPU mode** via CUDA for large-scale alignment batches.

## Motivation

[chewBBACA](https://github.com/B-UMMI/chewBBACA) is the reference implementation for gene-by-gene allele calling in bacterial genomics, widely adopted in public health laboratories for foodborne pathogen surveillance (Salmonella, Listeria, E. coli, Campylobacter). Its correctness and schema ecosystem (via [Chewie-NS](https://chewbbaca.online/)) are well established.

However, in real-time surveillance scenarios — where hundreds or thousands of genomes must be typed daily — the Python/BLASTp pipeline can become a bottleneck. **chewBBACA-rs** was developed to address this by:

1. **Eliminating the BLAST dependency** — replacing BLASTp with SIMD-accelerated Smith-Waterman (via [parasail](https://github.com/jeffdaily/parasail)) using the same scoring parameters (BLOSUM62, gap_open=11, gap_extend=1)
2. **Parallelizing all pipeline stages** — schema loading, CDS deduplication, clustering, and alignment all run in parallel via [rayon](https://github.com/rayon-rs/rayon)
3. **Reducing alignment pairs** — a minimizer-based pre-filter selects the top-5 candidate loci per CDS, reducing the number of Smith-Waterman alignments by ~8x without affecting results

## Limitations

- **AlleleCall only** — chewBBACA-rs reimplements only the `AlleleCall` module. Schema creation (`CreateSchema`), schema evaluation, and other chewBBACA modules are not included. Use the original chewBBACA for these tasks.
- **CDS prediction** — chewBBACA-rs does not include a built-in gene predictor. CDS must be pre-computed using the included `predict_cds.py` script (based on [pyrodigal](https://github.com/althonos/pyrodigal)). Minor differences between pyrodigal and prodigal may cause a small number of classification differences (~0.01%).
- **Read-only schema** — unlike chewBBACA mode 4, chewBBACA-rs does **not** update the schema in place. Novel alleles (INF) are written to `novel_alleles.fasta` in the output directory but are not appended to the schema FASTA files. This means INF alleles discovered during a run are not available as references for subsequent genomes within the same run.
- **Schema format** — reads standard chewBBACA FASTA schemas but does not read/write pickle caches. Hash tables and mode lengths are recomputed from FASTA files at startup.
- **GPU mode** — experimental CUDA support is included but not yet production-ready for very large batches.
- **Not a fork** — this is an independent reimplementation, not a fork of the original Python codebase. Bugs or edge cases not covered by the BeONE validation datasets may exist.

## Acknowledgments

This project builds entirely on the algorithmic foundations of **chewBBACA** by Silva et al. The classification logic, BSR-based scoring, representative determination, and output format are all derived from the original implementation. We are grateful to the chewBBACA team for their excellent tool and for making schemas publicly available via Chewie-NS.

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
