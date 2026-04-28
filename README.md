<p align="center">
  <img src="logo.png" alt="chewcall" width="420">
</p>

# chewcall

A high-performance allele caller for cgMLST/wgMLST schemas, inspired by and compatible with [chewBBACA](https://github.com/B-UMMI/chewBBACA).

**chewcall** reimplements the AlleleCall algorithm from chewBBACA in Rust, replacing BLASTp with SIMD-accelerated exact Smith-Waterman protein alignment via [parasail](https://github.com/jeffdaily/parasail).

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
cargo build --release

# Optional: enable host-specific SIMD codegen for a small additional speedup
# (≤4% on the BeONE benchmarks; the published timings use the standard build above)
# RUSTFLAGS="-C target-cpu=native" cargo build --release

# With GPU support (requires CUDA)
CUDA_HOME=/usr/local/cuda cargo build --release

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

Required:
  -i, --input <INPUT>                 Input directory with genome FASTA files
  -g, --schema <SCHEMA>               Schema directory (chewBBACA format)
  -o, --output <OUTPUT>               Output directory

Calling parameters:
      --bsr <BSR>                     BLAST Score Ratio threshold [default: 0.6]
      --size-threshold <SIZE>         Size threshold for ASM/ALM classification [default: 0.2]
      --min-length <MIN>              Minimum sequence length in bp [default: 0]
  -t, --translation-table <TT>        Genetic code [default: 11]

Minimizer pre-filter (Phase 4 candidate selection):
      --minimizer-k <K>               Minimizer k-mer size [default: 5]
      --minimizer-w <W>               Minimizer window size [default: 5]
      --minimizer-threshold <FRAC>    Minimum shared-minimizer fraction [default: 0.2]
                                      Lower this on schemas with high allelic
                                      diversity if some loci appear under-represented
                                      in the short/ subdirectory.

Alignment backend:
      --mode <MODE>                   Alignment mode: "fast" (parasail SIMD) or
                                      "compatible" (BLASTp subprocess) [default: fast]
      --blastp-path <PATH>            Path to blastp binary (required for --mode compatible)
      --gpu                           Use GPU (CUDA) for Smith-Waterman alignment

CDS prediction:
      --cds-input <CDS_INPUT>         Directory with pre-computed CDS FASTA files
                                      (from predict_cds.py); skips built-in prodigal-rs
      --prodigal-path <PATH>          Path to prodigal binary (subprocess fallback)
      --prodigal-ffi                  Use the bundled libprodigal FFI instead of a
                                      subprocess (faster startup, requires training file)
      --prodigal-mode <MODE>          Prodigal mode: single | meta [default: single]

Runtime / output:
      --cpu <CPU>                     Number of CPU threads [default: 1]
      --update-schema                 Append novel alleles (INF) to the schema FASTA
                                      files in place. By default chewcall is read-only
                                      and writes novel alleles only to the output dir.
```

### Auditing a schema before a run (`tune_minimizer`)

A separate binary `tune_minimizer` audits a schema against the minimizer pre-filter
parameters and reports loci where the worst-case minimizer-Jaccard between an
allele and its best-matching `short/` representative falls below the chosen
threshold (Kchooser-style — see ksnp4). Useful for detecting loci with
under-sized representative sets that would silently turn into LNFs at allele
calling time.

```bash
tune_minimizer --schema /path/to/schema --threshold 0.20 --exclude-inferred \
    --out tune_report.tsv --cpu 8
```

The TSV lists every locus with `n_reps`, `n_alleles`, `worst_recall`, the
identifier of the worst-recalled allele, and a `flagged` column. The summary on
stderr suggests either lowering `--minimizer-threshold` for the next chewcall
run or — preferably — expanding the representative set via
`chewBBACA SchemaEvaluator` / `CreateSchema`. `--exclude-inferred` skips
`*N`-prefixed alleles, which are by construction outliers and rarely appear as
actual query CDS.

### CDS prediction modes

chewcall supports three CDS prediction modes:

1. **Built-in prodigal-rs** (default) — Pure Rust reimplementation of Prodigal 2.6.3 (single mode). No external dependencies. Uses the `.trn` training file from the schema directory.
2. **Pre-computed CDS** (`--cds-input`) — Reads CDS from a directory of FASTA files pre-computed with pyrodigal or prodigal.
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

## License

GPL-3.0 — same as the original [chewBBACA](https://github.com/B-UMMI/chewBBACA).

## Authors

GenPat Team — [Istituto Zooprofilattico Sperimentale dell'Abruzzo e del Molise](https://www.izs.it/)
