<p align="center">
  <img src="logo.svg" alt="chewcall" width="120">
</p>

# chewcall

> **Exploratory project — NOT for production use.**
> This is a research prototype developed to explore performance optimization strategies for allele calling. It has been validated on BeONE datasets but has **not** undergone the extensive testing and validation required for use in clinical or public health surveillance. For production use, please use the original [chewBBACA](https://github.com/B-UMMI/chewBBACA).

A high-performance allele caller for cgMLST/wgMLST schemas, inspired by and compatible with [chewBBACA](https://github.com/B-UMMI/chewBBACA).

**chewcall** reimplements the AlleleCall algorithm from chewBBACA in Rust, replacing BLASTp with SIMD-accelerated Smith-Waterman alignment via [parasail](https://github.com/jeffdaily/parasail), achieving **5-12x faster** allele calling with **identical or near-identical** results (100% match on *L. monocytogenes* cgMLST).

## Overview

[chewBBACA](https://chewbbaca.readthedocs.io/) (BSR-Based Allele Calling Algorithm) is the reference tool for bacterial whole-genome and core-genome MLST analysis, widely used in foodborne pathogen surveillance and outbreak investigation.

**chewcall** reimplements the AlleleCall module in Rust, replacing BLASTp with a native Smith-Waterman aligner (BLOSUM62, gap_open=11, gap_extend=1) and adding a minimizer-based clustering pre-filter to drastically reduce the number of alignment pairs.

### Key features

- **Compatible** with chewBBACA schemas (Chewie-NS, PrepExternalSchema, CreateSchema)
- **Identical results** for *L. monocytogenes* cgMLST, **>99.98% match** across all tested organisms (see [Validation](#validation))
- **5-12x faster** than chewBBACA on multi-core systems
- **Parallel everything**: schema loading, CDS deduplication, clustering, and SW alignment via [rayon](https://github.com/rayon-rs/rayon)
- **Optional GPU acceleration** via CUDA for large-scale datasets
- **Minimizer-based pre-filtering**: top-K cluster selection reduces alignment pairs by ~8x without affecting results
- All 11 chewBBACA classification classes: EXC, INF, PLOT3, PLOT5, LOTSC, NIPH, NIPHEM, ALM, ASM, PAMA, LNF

## Validation

Validated on 4 organisms from the [BeONE](https://onehealthejp.eu/projects/foodborne-zoonoses/jrp-beone) project (100 genomes each), comparing chewcall vs chewBBACA v3.3.10 AlleleCall (mode 4). Schemas from [Chewie-NS](https://chewbbaca.online/). Both tools use the same pre-computed CDS (pyrodigal) to ensure identical gene predictions.

CRC32-hashed allelic profiles are compared cell-by-cell. CRC32 hashing maps each allele to the hash of its DNA sequence, making the comparison independent of allele ID numbering:

#### Full wgMLST comparison

| Organism | Loci | Cells | CRC32 match |
|----------|------|-------|-------------|
| *L. monocytogenes* | 1748 (cgMLST) | 174,800 | **100.0000% (IDENTICAL)** |
| *S. enterica* | 8558 (wgMLST) | 855,800 | 99.986% |
| *E. coli* | 7601 (wgMLST) | 760,100 | 99.995% |
| *C. jejuni* | 2794 (wgMLST) | 279,400 | 99.996% |

#### Core genome (cgMLST) comparison

To assess accuracy on the loci that matter most for epidemiological surveillance, we restrict the comparison to **core loci** — those present in >=95% of genomes in each dataset. These correspond to the loci typically included in cgMLST schemas:

| Organism | Core loci (>=95%) | Cells | CRC32 match |
|----------|-------------------|-------|-------------|
| *L. monocytogenes* | 1748 | 174,800 | **100.0000% (IDENTICAL)** |
| *S. enterica* | 3293 | 329,300 | **99.9982%** (6 diffs) |
| *E. coli* | 2805 | 280,500 | **100.0000% (IDENTICAL)** |
| *C. jejuni* | 994 | 99,400 | **100.0000% (IDENTICAL)** |

Three out of four organisms produce **100% identical** core genome profiles. The 6 remaining *S. enterica* core diffs are all cases where chewcall correctly identifies alleles (INF) that chewBBACA misses (LNF), concentrated on 2 borderline loci at the 95% presence threshold (wgMLST-00047202, present in 95/100 genomes, and wgMLST-00047811, 99/100). At a 98% presence threshold, only 1 diff remains across all organisms.

### Why are there remaining wgMLST differences?

chewBBACA uses **BLASTp** for protein alignment, while chewcall uses **parasail Smith-Waterman** (BLOSUM62, gap_open=11, gap_extend=1). Both use the same scoring matrix and gap penalties, but BLAST employs **database-size-dependent heuristics** (E-value thresholds, word seeding) that parasail's exact Smith-Waterman does not.

The ~0.01% remaining differences across wgMLST schemas arise from:

1. **Borderline hit discovery** — BLAST's word-seeding heuristics may find or miss alignments near the BSR threshold (0.6) that exact Smith-Waterman handles differently. Neither tool is "wrong" — these are genuinely borderline cases where the alignment score is close to the classification threshold.

2. **Cascading novel allele effects** — When one tool discovers a novel allele (INF) that the other misses, subsequent genomes can match that novel allele. A single borderline difference in one genome can cascade into multiple discordant cells across other genomes for the same locus. This explains why most differences cluster around a few specific loci (e.g., *wgMLST-00027883* in *E. coli*).

3. **Accessory loci are noisier** — Accessory loci (present in <95% of genomes) are inherently more variable and have weaker matches to schema representatives. Small scoring differences between BLAST and parasail are more likely to flip a classification near the threshold. Core loci, being well-conserved, produce robust matches that are insensitive to the alignment engine used.

All wgMLST differences are confined to accessory loci and do not affect cgMLST-based epidemiological analysis (minimum spanning trees, cluster detection, outbreak investigation).

## Performance

Benchmarked on [BeONE](https://onehealthejp.eu/projects/foodborne-zoonoses/jrp-beone) datasets (100 genomes each, 8 CPU threads). Schemas from [Chewie-NS](https://chewbbaca.online/). Both tools use pre-computed CDS from pyrodigal.

| Organism | Loci | Schema | chewBBACA | chewcall | Speedup | CRC32 match |
|----------|------|--------|-----------|----------|---------|-------------|
| *L. monocytogenes* | 1748 | cgMLST | 50s | 4.1s | **12.4x** | **IDENTICAL** |
| *S. enterica* | 8558 | wgMLST | 130s | 22.2s | **5.9x** | 99.986% |
| *E. coli* | 7601 | wgMLST | 388s | 53.6s | **7.2x** | 99.995% |
| *C. jejuni* | 2794 | wgMLST | 100s | 10.4s | **9.6x** | 99.996% |

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
      --gpu                     Use GPU (CUDA) for Smith-Waterman alignment
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
- **No BLAST dependency** — only requires parasail shared library.
- **Read-only schema** — novel alleles are written to `novel_alleles.fasta` but not appended to the schema.
- **Optional GPU mode** via CUDA for large-scale alignment batches.

## Limitations

- **AlleleCall only** — chewcall reimplements only the AlleleCall algorithm. Schema creation, evaluation, and other chewBBACA modules are not included.
- **CDS prediction** — chewcall does not include a built-in gene predictor. CDS must be pre-computed using the included `predict_cds.py` script (based on [pyrodigal](https://github.com/althonos/pyrodigal)).
- **Read-only schema** — unlike chewBBACA, chewcall does **not** update the schema in place. Novel alleles (INF) are written to `novel_alleles.fasta` in the output directory but are not appended to the schema FASTA files.
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
