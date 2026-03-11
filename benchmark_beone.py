#!/usr/bin/env python3
"""
Benchmark: chewbbacca-rs vs chewBBACA (Python) on BeONE datasets.

Downloads genome assemblies from Zenodo and cgMLST/wgMLST schemas from
Chewie-NS, then runs allele calling with both implementations, comparing
CRC32-hashed profiles for determinism.

BeONE project: https://onehealthejp.eu/projects/foodborne-zoonoses/jrp-beone
Zenodo datasets: https://zenodo.org/records/7802702 (Lm), 7802723 (Se),
                  7802728 (Ec), 7802717 (Cj)

Usage:
    python benchmark_beone.py [options]

    # Run all organisms
    python benchmark_beone.py

    # Run only Listeria with 100 genomes
    python benchmark_beone.py --organism lm --n-samples 100

    # Run only Rust (skip Python comparison)
    python benchmark_beone.py --rust-only

    # Use pre-downloaded data
    python benchmark_beone.py --skip-download --data-dir /mnt/disk2/a.deruvo/beone_benchmarks/data
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
import glob

# ── BeONE datasets ──────────────────────────────────────────────────────────

DATASETS = {
    'lm': {
        'name': 'Listeria monocytogenes',
        'short': 'Lm',
        'zenodo_url': 'https://zenodo.org/api/records/7802702/files/BeONE_Lm_assemblies.zip/content',
        'zenodo_zip': 'BeONE_Lm_assemblies.zip',
        'chewie_sp': '18',
        'chewie_sc': '1',
        'schema_type': 'cgMLST',
    },
    'se': {
        'name': 'Salmonella enterica',
        'short': 'Se',
        'zenodo_url': 'https://zenodo.org/api/records/7802723/files/BeONE_Se_assemblies.zip/content',
        'zenodo_zip': 'BeONE_Se_assemblies.zip',
        'chewie_sp': '14',
        'chewie_sc': '1',
        'schema_type': 'wgMLST',
    },
    'ec': {
        'name': 'Escherichia coli',
        'short': 'Ec',
        'zenodo_url': 'https://zenodo.org/api/records/7802728/files/BeONE_Ec_assemblies.zip/content',
        'zenodo_zip': 'BeONE_Ec_assemblies.zip',
        'chewie_sp': '10',
        'chewie_sc': '1',
        'schema_type': 'wgMLST',
    },
    'cj': {
        'name': 'Campylobacter jejuni',
        'short': 'Cj',
        'zenodo_url': 'https://zenodo.org/api/records/7802717/files/BeONE_Cj_assemblies.zip/content',
        'zenodo_zip': 'BeONE_Cj_assemblies.zip',
        'chewie_sp': '6',
        'chewie_sc': '1',
        'schema_type': 'wgMLST',
    },
}

# ── Configuration ───────────────────────────────────────────────────────────

# Path to chewbbacca-rs binary (relative to this script)
RUST_BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "target", "release", "chewbbacca-rs")

# Path to predict_cds.py (pre-computes CDS to avoid pyrodigal dependency in Rust)
PREDICT_CDS_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "predict_cds.py")

# Path to chewBBACA project (for Python baseline).
# Set CHEWBBACA_DIR env var to override, or install chewBBACA via pip.
CHEWBBACA_DIR = os.environ.get("CHEWBBACA_DIR",
                                os.path.expanduser("~/chewbbacca_gpu"))

# ── Utilities ────────────────────────────────────────────────────────────────

def download_file(url, dest_path):
    """Download a file. Tries wget, then curl."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"    Already downloaded: {os.path.basename(dest_path)}")
        return
    print(f"    Downloading {os.path.basename(dest_path)} ...")
    tmp_path = dest_path + ".part"
    try:
        subprocess.run(["wget", "-q", "-O", tmp_path, url], check=True)
    except FileNotFoundError:
        subprocess.run(["curl", "-L", "--progress-bar", "-o", tmp_path, url],
                        check=True)
    os.rename(tmp_path, dest_path)
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"    Done ({size_mb:.0f} MB)")


def _count_fastas(directory):
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory)
                if f.endswith(('.fasta', '.fa', '.fna'))])


def _find_schema_dir(base_dir):
    """Find the actual schema directory (may be nested after DownloadSchema)."""
    if not os.path.exists(base_dir):
        return base_dir
    if any(f.endswith('.fasta') for f in os.listdir(base_dir)):
        return base_dir
    for d in sorted(os.listdir(base_dir)):
        candidate = os.path.join(base_dir, d)
        if not os.path.isdir(candidate):
            continue
        if any(f.endswith('.fasta') for f in os.listdir(candidate)):
            return candidate
        for d2 in sorted(os.listdir(candidate)):
            candidate2 = os.path.join(candidate, d2)
            if os.path.isdir(candidate2):
                if any(f.endswith('.fasta') for f in os.listdir(candidate2)):
                    return candidate2
    return base_dir


# ── Download functions ───────────────────────────────────────────────────────

def download_genomes(dataset, data_dir):
    genomes_dir = os.path.join(data_dir, "genomes")
    if _count_fastas(genomes_dir) > 0:
        print(f"    Genomes already present: {_count_fastas(genomes_dir)} FASTA files")
        return genomes_dir

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, dataset['zenodo_zip'])
    download_file(dataset['zenodo_url'], zip_path)

    print("    Extracting assemblies...")
    extract_tmp = os.path.join(data_dir, "_extract_tmp")
    os.makedirs(extract_tmp, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_tmp)

    os.makedirs(genomes_dir, exist_ok=True)
    for ext in ('*.fasta', '*.fa', '*.fna'):
        for f in glob.glob(os.path.join(extract_tmp, "**", ext), recursive=True):
            shutil.move(f, os.path.join(genomes_dir, os.path.basename(f)))
    shutil.rmtree(extract_tmp, ignore_errors=True)

    n = _count_fastas(genomes_dir)
    if n == 0:
        print("    ERROR: No FASTA files found in the archive!")
        sys.exit(1)
    print(f"    Ready: {n} genome assemblies")
    return genomes_dir


def download_schema(dataset, data_dir):
    schema_base = os.path.join(data_dir, "schema")
    if os.path.exists(schema_base):
        schema_dir = _find_schema_dir(schema_base)
        n = len([f for f in os.listdir(schema_dir) if f.endswith('.fasta')])
        if n > 0:
            print(f"    Schema already present: {n} loci ({dataset['schema_type']})")
            return schema_dir

    sp, sc = dataset['chewie_sp'], dataset['chewie_sc']
    print(f"    Downloading {dataset['schema_type']} schema from Chewie-NS (sp={sp}, sc={sc})...")
    try:
        subprocess.run([
            sys.executable, "-m", "CHEWBBACA.chewBBACA",
            "DownloadSchema", "-sp", sp, "-sc", sc, "-o", schema_base,
        ], check=True)
    except subprocess.CalledProcessError:
        print(f"\n    ERROR: Failed to download schema.")
        print(f"    Try manually: chewBBACA.py DownloadSchema -sp {sp} -sc {sc} -o {schema_base}")
        sys.exit(1)

    schema_dir = _find_schema_dir(schema_base)
    n = len([f for f in os.listdir(schema_dir) if f.endswith('.fasta')])
    print(f"    Ready: {n} loci")
    return schema_dir


# ── CDS pre-computation ─────────────────────────────────────────────────────

def precompute_cds(genomes_dir, schema_dir, cds_output_dir, n_samples, cpu_cores):
    """Pre-compute CDS using predict_cds.py (pyrodigal) for Rust to consume."""
    if os.path.exists(cds_output_dir):
        n_cds = len([f for f in os.listdir(cds_output_dir) if f.endswith('.cds.fasta')])
        if n_cds > 0:
            print(f"    CDS already pre-computed: {n_cds} files")
            return cds_output_dir

    os.makedirs(cds_output_dir, exist_ok=True)

    # If n_samples, create a symlinked subset directory
    input_dir = genomes_dir
    if n_samples:
        all_fastas = sorted([
            f for f in os.listdir(genomes_dir)
            if f.endswith(('.fasta', '.fa', '.fna'))
        ])[:n_samples]
        input_dir = cds_output_dir + "_genomes_subset"
        os.makedirs(input_dir, exist_ok=True)
        for f in all_fastas:
            dst = os.path.join(input_dir, f)
            if not os.path.exists(dst):
                os.symlink(os.path.join(genomes_dir, f), dst)
        n_genomes = len(all_fastas)
    else:
        n_genomes = _count_fastas(genomes_dir)

    print(f"    Pre-computing CDS for {n_genomes} genomes...")
    cmd = [
        sys.executable, PREDICT_CDS_SCRIPT,
        "-i", input_dir,
        "-g", schema_dir,
        "-o", cds_output_dir,
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    predict_cds.py stderr: {result.stderr[-500:]}")
        raise RuntimeError(f"predict_cds.py failed (exit {result.returncode})")
    dt = time.time() - t0
    n_cds = len([f for f in os.listdir(cds_output_dir) if f.endswith('.cds.fasta')])
    print(f"    CDS prediction: {n_cds} files in {dt:.1f}s")
    return cds_output_dir


# ── Run implementations ─────────────────────────────────────────────────────

def run_python_chewbbaca(output_dir, schema_dir, genomes_dir, n_samples,
                         cpu_cores):
    """Run original Python chewBBACA (BLAST backend) via subprocess.

    Requires a Python environment with chewBBACA and BLAST installed.
    Set CHEWBBACA_PYTHON to the Python interpreter path, or it will use
    the conda env 'chewbbacca_gpu' by default.
    """
    conda_python = os.environ.get(
        "CHEWBBACA_PYTHON",
        os.path.expanduser("~/miniconda3/envs/chewbbacca_gpu/bin/python3")
    )
    if not os.path.exists(conda_python):
        print(f"    ERROR: Python interpreter not found: {conda_python}")
        print(f"    Set CHEWBBACA_PYTHON env var to your chewBBACA Python interpreter.")
        return False

    # Build genome list
    all_fastas = sorted([
        f for f in os.listdir(genomes_dir)
        if f.endswith(('.fasta', '.fa', '.fna'))
    ])
    if n_samples:
        all_fastas = all_fastas[:n_samples]
    genome_files = [os.path.join(genomes_dir, f) for f in all_fastas]

    # Write a runner script that imports chewBBACA and runs AlleleCall
    runner_script = os.path.join(output_dir, '_run_chewbbaca.py')
    with open(runner_script, 'w') as f:
        f.write(f"""#!/usr/bin/env python3
import sys, os, shutil, tempfile
sys.path.insert(0, {CHEWBBACA_DIR!r})
os.environ['PATH'] = os.path.expanduser('~/miniconda3/envs/chewbbacca_gpu/bin') + ':' + os.environ.get('PATH', '')

from CHEWBBACA.utils import constants as ct, blast_wrapper as bw, file_operations as fo
from CHEWBBACA.AlleleCall import allele_call

bw.disable_gpu()

schema_dir = {schema_dir!r}
output_dir = {output_dir!r}
genome_files = {genome_files!r}
cpu_cores = {cpu_cores!r}

# Copy schema (chewBBACA modifies it in-place)
py_schema = tempfile.mkdtemp(prefix='schema_py_', dir=output_dir)
import shutil
shutil.copytree(schema_dir, os.path.join(py_schema, 'schema'), dirs_exist_ok=True)
work_schema = os.path.join(py_schema, 'schema')

genome_list_file = os.path.join(output_dir, 'genomes.txt')
with open(genome_list_file, 'w') as gf:
    gf.write('\\n'.join(genome_files))

loci_list_file = os.path.join(output_dir, 'loci.txt')
schema_files = [os.path.join(work_schema, f) for f in os.listdir(work_schema) if f.endswith('.fasta')]
with open(loci_list_file, 'w') as lf:
    lf.write('\\n'.join(schema_files))

config_file = os.path.join(work_schema, ct.SCHEMA_CONFIG_BASENAME)
schema_params = fo.pickle_loader(config_file)

def unwrap(val, default):
    if isinstance(val, list):
        return val[0] if val else default
    return val if val is not None else default

config = {{
    'Minimum sequence length': unwrap(schema_params.get('minimum_locus_length'), ct.MINIMUM_LENGTH_DEFAULT),
    'Size threshold': unwrap(schema_params.get('size_threshold'), ct.SIZE_THRESHOLD_DEFAULT),
    'Translation table': unwrap(schema_params.get('translation_table'), ct.GENETIC_CODES_DEFAULT),
    'BLAST Score Ratio': unwrap(schema_params.get('bsr'), ct.DEFAULT_BSR),
    'Word size': ct.WORD_SIZE_DEFAULT,
    'Window size': ct.WINDOW_SIZE_DEFAULT,
    'Clustering similarity': ct.CLUSTERING_SIMILARITY_DEFAULT,
    'Prodigal training file': None,
    'CPU cores': cpu_cores,
    'BLAST path': '',
    'CDS input': False,
    'Prodigal mode': 'single',
    'Mode': 4,
}}

ptf_files = [os.path.join(work_schema, f) for f in os.listdir(work_schema) if f.endswith('.trn')]
if ptf_files:
    config['Prodigal training file'] = ptf_files[0]

allele_call.main(genome_list_file, loci_list_file, work_schema,
                 output_dir, False, False, False,
                 False, False, False, 'crc32', False, config)

shutil.rmtree(py_schema, ignore_errors=True)
""")

    # Run with conda env python (has BLAST in PATH)
    env = os.environ.copy()
    env['PATH'] = os.path.expanduser('~/miniconda3/envs/chewbbacca_gpu/bin') + ':' + env.get('PATH', '')

    result = subprocess.run(
        [conda_python, runner_script],
        capture_output=True, text=True, env=env
    )
    if result.returncode != 0:
        print(f"    Python chewBBACA failed (exit {result.returncode}):")
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
        return False
    return True


def run_rust_chewbbacca(output_dir, schema_dir, genomes_dir, cds_dir,
                        n_samples, cpu_cores, use_gpu=False):
    """Run chewbbacca-rs."""
    if not os.path.exists(RUST_BINARY):
        print(f"    ERROR: Rust binary not found at {RUST_BINARY}")
        print(f"    Build with: cd {os.path.dirname(RUST_BINARY)}/.. && "
              f"CUDA_HOME=/usr/local/cuda RUSTFLAGS='-C target-cpu=native' "
              f"cargo build --release")
        return False

    # Limit genomes if n_samples
    input_dir = genomes_dir
    if n_samples:
        all_fastas = sorted([
            f for f in os.listdir(genomes_dir)
            if f.endswith(('.fasta', '.fa', '.fna'))
        ])[:n_samples]
        input_dir = os.path.join(output_dir, "_genomes_subset")
        os.makedirs(input_dir, exist_ok=True)
        for f in all_fastas:
            src = os.path.join(genomes_dir, f)
            dst = os.path.join(input_dir, f)
            if not os.path.exists(dst):
                os.symlink(src, dst)

    cmd = [
        RUST_BINARY,
        "-i", input_dir,
        "-g", schema_dir,
        "-o", output_dir,
        "--cpu", str(cpu_cores),
    ]
    if cds_dir:
        cmd += ["--cds-input", cds_dir]
    if use_gpu:
        cmd += ["--gpu"]

    env = os.environ.copy()
    parasail_dir = os.path.expanduser("~/parasail/build")
    env['LD_LIBRARY_PATH'] = parasail_dir + ':' + env.get('LD_LIBRARY_PATH', '')

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"    Rust failed (exit {result.returncode}):")
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
        return False

    # Print timing info from stderr
    for line in result.stderr.split('\n'):
        if '[TIMING]' in line or 'Summary' in line or 'EXC:' in line or 'Done' in line:
            print(f"    {line.strip()}")

    return True


# ── Comparison ───────────────────────────────────────────────────────────────

def compare_hashed_profiles(dir_a, dir_b, label_a="A", label_b="B"):
    """Compare CRC32-hashed allelic profiles. Returns (total, matching, diff_details)."""
    def find_hashed(d):
        for f in os.listdir(d):
            if 'hashed' in f.lower() and f.endswith('.tsv'):
                return os.path.join(d, f)
        return None

    file_a = find_hashed(dir_a)
    file_b = find_hashed(dir_b)

    if not file_a or not file_b:
        return 0, 0, f"Missing hashed file: {label_a}={file_a is not None}, {label_b}={file_b is not None}"

    with open(file_a) as f:
        rows_a = list(csv.reader(f, delimiter='\t'))
    with open(file_b) as f:
        rows_b = list(csv.reader(f, delimiter='\t'))

    total = 0
    matching = 0

    for i in range(1, min(len(rows_a), len(rows_b))):
        for j in range(1, min(len(rows_a[i]), len(rows_b[i]))):
            total += 1
            if rows_a[i][j] == rows_b[i][j]:
                matching += 1

    if total == 0:
        return 0, 0, "No cells to compare"

    pct = matching / total * 100
    return total, matching, f"{matching}/{total} ({pct:.4f}%)"


# ── Per-organism benchmark ───────────────────────────────────────────────────

def run_organism_benchmark(key, dataset, args):
    name = dataset['name']
    data_dir = os.path.join(args.data_dir, key)
    output_dir = os.path.join(args.output_dir, key)
    os.makedirs(output_dir, exist_ok=True)

    n_label = args.n_samples or "all"
    print(f"\n{'=' * 70}")
    print(f"  {name} ({dataset['short']}) - {n_label} genomes, {args.cpu_cores} threads")
    print(f"{'=' * 70}")

    # Download
    if not args.skip_download:
        print(f"\n  [1/5] Downloading assemblies...")
        genomes_dir = download_genomes(dataset, data_dir)
        print(f"\n  [2/5] Downloading schema...")
        schema_dir = download_schema(dataset, data_dir)
    else:
        genomes_dir = os.path.join(data_dir, "genomes")
        schema_base = os.path.join(data_dir, "schema")
        if _count_fastas(genomes_dir) == 0:
            print(f"  ERROR: No genomes in {genomes_dir}. Run without --skip-download.")
            return None
        schema_dir = _find_schema_dir(schema_base)

    n_genomes = _count_fastas(genomes_dir)
    if args.n_samples:
        n_genomes = min(n_genomes, args.n_samples)
    n_loci = len([f for f in os.listdir(schema_dir) if f.endswith('.fasta')])

    result = {
        'name': name, 'short': dataset['short'], 'key': key,
        'n_genomes': n_genomes, 'n_loci': n_loci,
        'schema_type': dataset['schema_type'],
        'python_time': 0, 'rust_time': 0, 'rust_gpu_time': 0,
        'cds_time': 0,
        'total_cells': 0, 'matching_cells': 0, 'match_detail': 'N/A',
    }

    # Pre-compute CDS for Rust
    cds_dir = None
    if not args.python_only:
        print(f"\n  [3/5] Pre-computing CDS (pyrodigal)...")
        cds_output = os.path.join(output_dir, "cds_precomputed")
        t0 = time.time()
        try:
            cds_dir = precompute_cds(genomes_dir, schema_dir, cds_output,
                                     args.n_samples, args.cpu_cores)
            result['cds_time'] = time.time() - t0
        except Exception as e:
            print(f"    CDS prediction failed: {e}")
            import traceback; traceback.print_exc()

    python_dir = None
    rust_dir = None

    # Python (BLAST) run
    if not args.rust_only:
        print(f"\n  [4/5] Running Python chewBBACA (BLAST)...")
        python_dir = os.path.join(output_dir, "python_results")
        if os.path.exists(python_dir):
            shutil.rmtree(python_dir)
        os.makedirs(python_dir, exist_ok=True)
        t0 = time.time()
        ok = run_python_chewbbaca(python_dir, schema_dir, genomes_dir,
                                   args.n_samples, args.cpu_cores)
        if ok:
            result['python_time'] = time.time() - t0
            print(f"    Python time: {result['python_time']:.1f}s")
        else:
            result['python_time'] = 0
            python_dir = None

    # Rust run
    if not args.python_only:
        print(f"\n  [5/5] Running chewbbacca-rs (CPU, {args.cpu_cores} threads)...")
        rust_dir = os.path.join(output_dir, "rust_results")
        if os.path.exists(rust_dir):
            shutil.rmtree(rust_dir)
        os.makedirs(rust_dir, exist_ok=True)
        t0 = time.time()
        ok = run_rust_chewbbacca(rust_dir, schema_dir, genomes_dir,
                                  cds_dir, args.n_samples, args.cpu_cores)
        if ok:
            result['rust_time'] = time.time() - t0
            print(f"    Rust time: {result['rust_time']:.1f}s")
        else:
            result['rust_time'] = 0

        # GPU run (optional)
        if args.gpu:
            print(f"\n  [5b] Running chewbbacca-rs (GPU)...")
            rust_gpu_dir = os.path.join(output_dir, "rust_gpu_results")
            if os.path.exists(rust_gpu_dir):
                shutil.rmtree(rust_gpu_dir)
            os.makedirs(rust_gpu_dir, exist_ok=True)
            t0 = time.time()
            ok = run_rust_chewbbacca(rust_gpu_dir, schema_dir, genomes_dir,
                                      cds_dir, args.n_samples, args.cpu_cores,
                                      use_gpu=True)
            if ok:
                result['rust_gpu_time'] = time.time() - t0
                print(f"    Rust GPU time: {result['rust_gpu_time']:.1f}s")

    # Compare
    if python_dir and rust_dir:
        print(f"\n  Comparing CRC32 hashed profiles...")
        total, matching, detail = compare_hashed_profiles(
            python_dir, rust_dir, "Python", "Rust")
        result['total_cells'] = total
        result['matching_cells'] = matching
        result['match_detail'] = detail
        print(f"    {detail}")

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark chewbbacca-rs vs Python chewBBACA on BeONE datasets"
    )
    parser.add_argument("--data-dir",
                        default="/mnt/disk2/a.deruvo/beone_benchmarks/data",
                        help="Directory for downloaded data")
    parser.add_argument("--output-dir", default="./beone_benchmark",
                        help="Directory for benchmark outputs (default: ./beone_benchmark)")
    parser.add_argument("--organism", choices=list(DATASETS.keys()),
                        nargs='+', default=None,
                        help="Organisms to benchmark (default: all). "
                             "Options: lm (Listeria), se (Salmonella), "
                             "ec (E. coli), cj (Campylobacter)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of genomes per organism (default: all)")
    parser.add_argument("--cpu-cores", type=int, default=8,
                        help="CPU threads (default: 8)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download step (data must exist)")
    parser.add_argument("--rust-only", action="store_true",
                        help="Run only Rust (skip Python)")
    parser.add_argument("--python-only", action="store_true",
                        help="Run only Python (skip Rust)")
    parser.add_argument("--gpu", action="store_true",
                        help="Also run Rust with --gpu flag")
    args = parser.parse_args()

    args.data_dir = os.path.abspath(args.data_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    organisms = args.organism or list(DATASETS.keys())

    print("=" * 70)
    print("  BeONE BENCHMARK - chewbbacca-rs vs Python chewBBACA")
    print(f"  Organisms: {', '.join(DATASETS[k]['name'] for k in organisms)}")
    print(f"  Samples: {args.n_samples or 'all'} per organism")
    print(f"  CPU threads: {args.cpu_cores}")
    if args.gpu:
        print(f"  GPU: enabled")
    print("=" * 70)

    results = []
    for key in organisms:
        r = run_organism_benchmark(key, DATASETS[key], args)
        if r:
            results.append(r)

    # Summary table
    print(f"\n\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")

    header = f"  {'Organism':<25} {'Genomes':>7} {'Loci':>6} {'Python':>8} {'Rust':>8}"
    if args.gpu:
        header += f" {'GPU':>8}"
    header += f" {'Speedup':>8} {'CRC32':>12}"
    print(header)
    print(f"  {'-'*25} {'-'*7} {'-'*6} {'-'*8} {'-'*8}", end="")
    if args.gpu:
        print(f" {'-'*8}", end="")
    print(f" {'-'*8} {'-'*12}")

    for r in results:
        py_t = f"{r['python_time']:.0f}s" if r['python_time'] > 0 else "N/A"
        rs_t = f"{r['rust_time']:.1f}s" if r['rust_time'] > 0 else "N/A"
        if r['python_time'] > 0 and r['rust_time'] > 0:
            speedup = f"{r['python_time']/r['rust_time']:.1f}x"
        else:
            speedup = "N/A"
        if r['total_cells'] > 0:
            pct = r['matching_cells'] / r['total_cells'] * 100
            if pct == 100.0:
                crc = "IDENTICAL"
            else:
                crc = f"{pct:.4f}%"
        else:
            crc = "N/A"

        line = f"  {r['name']:<25} {r['n_genomes']:>7} {r['n_loci']:>6} {py_t:>8} {rs_t:>8}"
        if args.gpu:
            gpu_t = f"{r['rust_gpu_time']:.1f}s" if r['rust_gpu_time'] > 0 else "N/A"
            line += f" {gpu_t:>8}"
        line += f" {speedup:>8} {crc:>12}"
        print(line)

    print(f"\n  Output: {args.output_dir}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
