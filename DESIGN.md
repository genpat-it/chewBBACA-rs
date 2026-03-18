# chewcall: Design Document

Fast allele caller for cgMLST/wgMLST schemas, inspired by the AlleleCall algorithm of [chewBBACA](https://github.com/B-UMMI/chewBBACA).

## Architecture

chewcall is implemented in Rust for performance-critical paths, with Python (pyrodigal) for CDS prediction. Alignment is performed via [parasail](https://github.com/jeffdaily/parasail) (SIMD Smith-Waterman, BLOSUM62, gap_open=11, gap_extend=1), with optional CUDA GPU acceleration.

### Source layout

```
chewcall/
├── Cargo.toml
├── predict_cds.py        # CDS prediction via pyrodigal
├── src/
│   ├── main.rs           # CLI (clap) + genome discovery
│   ├── pipeline.rs       # Pipeline orchestration (7 phases)
│   ├── types.rs          # Shared types (Config, Cds, Classification, etc.)
│   ├── schema.rs         # Schema loading (FASTA parsing, hashing, config)
│   ├── cds.rs            # CDS prediction (prodigal subprocess) + loading
│   ├── translate.rs      # Codon translation (genetic code tables)
│   ├── dedup.rs          # SHA-256 deduplication across genomes
│   ├── cluster.rs        # Minimizer-based clustering + SW alignment
│   ├── classify.rs       # Classification logic (all 11 classes)
│   ├── repdet.rs         # Representative determination (iterative)
│   ├── blast.rs          # External BLAST wrapper (compatible mode)
│   ├── sw.rs             # Pure-Rust Smith-Waterman (BLOSUM62)
│   ├── parasail_ffi.rs   # FFI bindings to parasail C library
│   ├── gpu_sw.rs         # CUDA GPU Smith-Waterman (via cudarc)
│   ├── lib.rs            # Library root
│   └── output.rs         # TSV/FASTA output writers
```

### Dependencies

| Crate | Purpose |
|-------|---------|
| `clap` | CLI argument parsing |
| `rayon` | Data parallelism (all pipeline phases) |
| `needletail` | Fast FASTA parsing |
| `sha2` | SHA-256 hashing for deduplication |
| `rustc-hash` | FxHashMap (fast hash map) |
| `crc32fast` | CRC32 hashing for allele profiles |
| `csv` | TSV output |
| `cudarc` | CUDA runtime bindings (optional GPU) |

### External dependencies

| Library | Purpose |
|---------|---------|
| [parasail](https://github.com/jeffdaily/parasail) | SIMD Smith-Waterman (AVX2/SSE4.1) |
| [pyrodigal](https://github.com/althonos/pyrodigal) | CDS prediction (Python) |

## Pipeline (7 phases)

### Phase 0: Schema loading
- Discovers loci from `short/` directory (representative alleles)
- Parses all FASTA files in parallel (rayon)
- Builds SHA-256 hash tables for DNA and protein sequences
- Computes CRC32 hashes for allele profile output
- Reads `.schema_config` pickle for BSR/size thresholds
- Computes or loads cached self-scores for representatives (parasail self-alignment)

### Phase 1: CDS prediction
- Runs prodigal as subprocess for each genome (parallel via rayon)
- Or loads pre-computed CDS from `--cds-input` directory (recommended)
- Pre-computation via `predict_cds.py` uses pyrodigal with `closed=True, mask=True` to match chewBBACA parameters
- Collects contig lengths for PLOT3/PLOT5 classification

### Phase 2: Deduplication
- SHA-256 hash of each CDS DNA sequence (uppercased)
- Groups identical CDS across all genomes
- Processes only distinct sequences in subsequent phases
- Maintains `hash → [(genome_idx, cds_idx)]` mapping for result propagation

### Phase 3a: Exact DNA matching
- Hash lookup: CDS DNA hash against schema allele hashes
- Match → **EXC** classification
- Multiple EXC matches for same genome+locus → **NIPHEM**

### Phase 3b: Translation
- Translates unmatched CDS to protein (genetic code 11 default)
- Filters by minimum length

### Phase 3c: Exact protein matching
- Hash lookup: protein hash against schema protein hashes
- Match → **INF** (first genome to see this allele) or **EXC** (subsequent genomes)
- Novel alleles: assigns next allele ID, writes to `novel_alleles.fasta`
- CRC32 hash computed for hashed output

### Phase 4: Clustering + alignment
- **Minimizer index**: builds minimizer (k=5, w=5) index over all representative proteins
- **Clustering**: for each unmatched protein, finds top-5 representatives by shared minimizer count (min_shared=1)
- **Alignment**: Smith-Waterman (BLOSUM62, gap_open=11, gap_extend=1) via parasail SIMD or CUDA GPU
- **BSR**: `score / representative_self_score` (target self-score)
- BSR >= threshold → classify based on alignment positions and sequence lengths
- Builds all alignment pairs first, then batches to GPU or CPU

### Phase 5: Representative determination (iterative)
- Processes Phase 4 results: BSR in [threshold, threshold+0.1) → candidate new representatives
- Adds candidates to representative set
- Rebuilds minimizer index and re-aligns remaining unmatched proteins
- Repeats until no new representatives found (max 10 iterations)
- Uses GPU if available, falls back to CPU

### Phase 6: Classification
For each inexact match (BSR >= threshold):

```
1. Target alignment coverage:
   - target coverage < 100% AND contig too short → LOTSC
   - alignment doesn't reach target 5' end → PLOT5
   - alignment doesn't reach target 3' end → PLOT3

2. Size comparison vs mode allele length:
   - CDS length < mode × (1 - size_threshold) → ASM
   - CDS length > mode × (1 + size_threshold) → ALM

3. Multiple matches same locus:
   - Single EXC/INF + ASM/ALM → keeps EXC/INF
   - All EXC → NIPHEM
   - Otherwise → NIPH

4. Multiple matches different loci → PAMA

5. No match → LNF

6. Default → INF (novel inferred allele)
```

### Phase 7: Output
- `results_alleles.tsv` — allelic profile matrix (genome × locus)
- `results_alleles_hashed.tsv` — CRC32-hashed allelic profiles
- `results_statistics.tsv` — per-genome classification counts
- `loci_summary_stats.tsv` — per-locus classification counts
- `results_contigsInfo.tsv` — CDS coordinates for classified loci
- `novel_alleles.fasta` — novel allele sequences (INF)

## Key design decisions

### Two alignment modes
chewcall supports two alignment modes via `--mode`:

- **`fast`** (default): parasail SIMD Smith-Waterman (BLOSUM62, gap_open=11, gap_extend=1). Eliminates the BLAST dependency and enables direct library-level alignment calls without subprocess overhead. Uses minimizer-based clustering to select top-5 candidates per query. **4-6x faster** than chewBBACA with 99.97-100% CRC32 agreement.

- **`compatible`**: external BLASTp for alignment (requires `--blastp-path`). Uses all-vs-all BLAST queries instead of Python chewBBACA's per-cluster approach. Slower (1.1-2.3x speedup) but provides a BLAST-based comparison point. Note: the all-vs-all approach produces different E-values than Python's per-cluster BLAST, so results are not necessarily closer to chewBBACA than the fast mode.

The fast mode is recommended: it is both faster and produces equal or better concordance with chewBBACA on core genome loci.

### Target self-score BSR
chewcall uses `BSR = alignment_score / representative_self_score` (target self-score), while chewBBACA uses `score / query_self_score`. With parasail SIMD alignment, target self-score produces better concordance with chewBBACA results than query self-score (validated empirically).

### Minimizer pre-filter
Instead of BLAST's internal word seeding, chewcall uses a minimizer index (k=5, w=5) to select the top-5 candidate representatives per query protein. This reduces alignment pairs by ~8x without affecting classification results. FNV-1a hash is used for k-mer hashing.

### Read-only schema
Unlike chewBBACA (mode 4), chewcall does not modify schema files. Novel alleles are tracked in memory during the run (for deduplication across genomes) and written to `novel_alleles.fasta`, but never appended to schema FASTA files. This avoids schema contamination and makes runs reproducible.

### GPU acceleration (optional)
CUDA GPU support via `cudarc` for batched Smith-Waterman alignment. The GPU kernel processes all alignment pairs in a single batch. GPU is used only for Phase 4 and Phase 5 alignment; all other phases run on CPU. Falls back to CPU parasail if GPU initialization fails.

## Schema compatibility

chewcall reads schemas in the standard chewBBACA format:

```
schema/
├── locus1.fasta            # Full allele sequences per locus
├── locus2.fasta
├── short/
│   ├── locus1_short.fasta  # Representative allele(s) per locus
│   └── locus2_short.fasta
├── *.trn                   # Prodigal training file
└── .schema_config          # (optional) pickle with BSR/size thresholds
```

- Locus list is derived from `short/*_short.fasta` filenames
- Allele IDs are parsed from FASTA headers (e.g., `>locus_1_1`, `>locus_1_2`)
- Mode length (most frequent allele length) is computed from full FASTA files
- `.schema_config` pickle is read for BSR and size_threshold values (overrides CLI defaults)
- Self-scores are cached in `short/self_scores_rs.tsv` for fast re-runs

Compatible with schemas from:
- [Chewie-NS](https://chewbbaca.online/) (`DownloadSchema`)
- `chewBBACA.py PrepExternalSchema`
- `chewBBACA.py CreateSchema`

## Validation

Validated against chewBBACA v3.3.10 on full BeONE datasets (up to 1540 genomes per organism, 8 CPU threads). Both tools use the same pre-computed CDS (pyrodigal) to ensure identical gene predictions. CRC32 hashing maps each allele to the hash of its DNA sequence, making the comparison independent of allele ID numbering.

### wgMLST (all loci)

| Organism | Genomes | Loci | Cells | CRC32 match |
|----------|---------|------|-------|-------------|
| *L. monocytogenes* | 1,426 | 1,748 | 2,492,648 | 100.0000% (1 diff) |
| *S. enterica* | 1,540 | 8,558 | 13,179,320 | 99.9985% (204 diffs) |
| *E. coli* | 308 | 7,601 | 2,341,108 | 99.9935% (152 diffs) |
| *C. jejuni* | 610 | 2,794 | 1,704,340 | 99.9765% (401 diffs) |

### Core genome (cgMLST)

| Organism | Core >=95% (loci) | Diffs | Core >=98% (loci) | Diffs | Core >=99% (loci) | Diffs |
|----------|-------------------|-------|-------------------|-------|-------------------|-------|
| *L. monocytogenes* | 1,731 | 1 | 1,721 | 1 | 1,716 | 1 |
| *S. enterica* | 3,259 | 77 | 3,027 | 37 | 2,765 | 16 |
| *E. coli* | 2,809 | 0 | 2,592 | 0 | 2,470 | 0 |
| *C. jejuni* | 991 | 0 | 900 | 0 | 706 | 0 |

Remaining differences are confined to accessory loci with borderline BSR scores, where parasail exact SW and BLASTp heuristics disagree. These do not affect cgMLST-based epidemiological analysis.

## Performance

Benchmarked on full BeONE datasets (8 CPU threads, fast mode):

| Organism | Genomes | Loci | chewBBACA | chewcall (fast) | Speedup |
|----------|---------|------|-----------|-----------------|---------|
| *L. monocytogenes* | 1,426 | 1,748 | 156s | 38.5s | **4.1x** |
| *S. enterica* | 1,540 | 8,558 | 586s | 147s | **4.0x** |
| *E. coli* | 308 | 7,601 | 570s | 97s | **5.9x** |
| *C. jejuni* | 610 | 2,794 | 215s | 49.5s | **4.3x** |
