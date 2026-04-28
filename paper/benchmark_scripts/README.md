# Benchmark scripts — chewcall paper

All scripts needed to reproduce the benchmark results reported in the paper.

## Setup

1. Copy and edit the configuration:
   ```bash
   cp config.sh.template config.sh
   # Edit paths in config.sh for your environment
   ```

2. Install dependencies:
   ```bash
   # chewBBACA v3.5.3
   conda create -n chewbbaca_v3 python=3.11
   conda activate chewbbaca_v3
   pip install chewBBACA==3.5.3
   conda install -c bioconda blast prodigal

   # chewBBACA v2.8.5
   conda create -n chewbbaca_v2 python=3.9
   conda activate chewbbaca_v2
   pip install chewBBACA==2.8.5
   conda install -c bioconda blast prodigal

   # CDS pre-computation
   pip install pyrodigal biopython
   ```

3. Build chewcall:
   ```bash
   RUSTFLAGS="-C target-cpu=native" cargo build --release
   ```

4. Download data (see Data Availability in the paper).

## Scripts

### Benchmarks

| Script | Paper section | Description |
|--------|--------------|-------------|
| `run_all.sh 01` | Results 3.1-3.4 | BeONE shared CDS (cc vs v3.5.3) |
| `run_all.sh 04` | Results 3.7 | Synthetic ground truth |
| `run_all.sh 05` | Results 3.5 | FDA outbreak reconstruction |
| `run_all.sh 07` | Results 3.1 | End-to-end cc vs v3.5.3 |
| `bench_08_v2_only.sh` | Results 3.6 | End-to-end cc vs v2.8.5 |
| `run_all.sh 03` | Suppl. S1 | Coverage degradation |
| `run_all.sh 06` | Suppl. S3 | Mamede scalability |
| `ablation_minimizer.sh` | Methods Table 1 | Minimizer parameter sweep |

### Data generation

| Script | Description |
|--------|-------------|
| `predict_cds.py` | Pre-compute CDS with pyrodigal |
| `generate_synthetic.py` | Generate synthetic CDS with known ground truth |
| `generate_synthetic_cds.py` | Alternative synthetic CDS generator |

### Analysis

| Script | Description |
|--------|-------------|
| `check_concordance.py` | Pairwise concordance (raw profiles) |
| `analyze_08_efsa.py` | v2.8.5 concordance + runtime |
| `evaluate_synthetic.py` | Synthetic benchmark evaluation |
| `hash_v2_profiles.py` | Generate CRC32 hashes from v2.8.5 output |

### Utilities

| Script | Description |
|--------|-------------|
| `config.sh.template` | Configuration template (copy to config.sh) |
| `prepare_data.sh` | Download and prepare benchmark data |
| `toggle_sort_fix.sh` | Toggle chewBBACA v3 sort bug (original vs fixed) |
| `benchmark_beone.py` | Standalone BeONE benchmark runner |

## Usage

```bash
source config.sh
./run_all.sh          # all benchmarks
./run_all.sh 01       # single benchmark
./bench_08_v2_only.sh # v2.8.5 only
```

## Notes

- Each run uses a **pristine schema copy** (fresh copy before every execution).
- Runtimes are the **median of 3 runs** (configurable via `REPS` in config.sh).
- v2.8.5 does not support `--cds` or `--no-inferred`.
