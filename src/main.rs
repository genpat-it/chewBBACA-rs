//! chewBBACA-rs: Rust reimplementation of chewBBACA AlleleCall.

use std::path::PathBuf;
use clap::Parser;

mod types;
mod translate;
mod schema;
mod cds;
mod dedup;
mod classify;
mod cluster;
mod sw;
mod parasail_ffi;
mod repdet;
mod output;
mod gpu_sw;
mod pipeline;

use types::Config;

#[derive(Parser, Debug)]
#[command(name = "chewbbacca-rs")]
#[command(about = "Fast allele calling for cgMLST/wgMLST (Rust reimplementation of chewBBACA)")]
#[command(version)]
struct Cli {
    /// Input directory with genome FASTA files
    #[arg(short = 'i', long)]
    input: PathBuf,

    /// Schema directory
    #[arg(short = 'g', long)]
    schema: PathBuf,

    /// Output directory
    #[arg(short = 'o', long)]
    output: PathBuf,

    /// BLAST Score Ratio threshold
    #[arg(long, default_value = "0.6")]
    bsr: f64,

    /// Size threshold for ASM/ALM classification
    #[arg(long, default_value = "0.2")]
    size_threshold: f64,

    /// Minimum sequence length (bp)
    #[arg(long, default_value = "0")]
    min_length: u32,

    /// Translation table (genetic code)
    #[arg(short = 't', long, default_value = "11")]
    translation_table: u8,

    /// Number of CPU cores
    #[arg(long, default_value = "1")]
    cpu: usize,

    /// Prodigal mode (single or meta)
    #[arg(long, default_value = "single")]
    prodigal_mode: String,

    /// Directory with pre-computed CDS FASTA files (from predict_cds.py).
    /// If provided, skips prodigal and reads CDS from these files.
    #[arg(long)]
    cds_input: Option<PathBuf>,

    /// Use GPU (CUDA) for Smith-Waterman alignment
    #[arg(long)]
    gpu: bool,
}

fn main() {
    let cli = Cli::parse();

    // Set rayon thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(cli.cpu)
        .build_global()
        .unwrap();

    let config = Config {
        bsr_threshold: cli.bsr,
        size_threshold: cli.size_threshold,
        min_sequence_length: cli.min_length,
        translation_table: cli.translation_table,
        cpu_cores: cli.cpu,
        prodigal_mode: cli.prodigal_mode,
        use_gpu: cli.gpu,
    };

    // Discover genome files
    let genome_paths = discover_genomes(&cli.input);
    if genome_paths.is_empty() {
        eprintln!("Error: no genome FASTA files found in {}", cli.input.display());
        std::process::exit(1);
    }
    eprintln!("Found {} genome files", genome_paths.len());

    // Run pipeline
    if let Err(e) = pipeline::run_allele_call(
        &genome_paths,
        &cli.schema,
        &cli.output,
        &config,
        cli.cds_input.as_deref(),
    ) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }

    eprintln!("Done.");
}

/// Find all FASTA files in input directory.
fn discover_genomes(input_dir: &PathBuf) -> Vec<String> {
    let mut paths = Vec::new();

    if input_dir.is_file() {
        paths.push(input_dir.to_string_lossy().to_string());
        return paths;
    }

    if let Ok(entries) = std::fs::read_dir(input_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                let ext = ext.to_string_lossy().to_lowercase();
                if matches!(ext.as_str(), "fasta" | "fa" | "fna" | "fas" | "fsa") {
                    paths.push(path.to_string_lossy().to_string());
                }
            }
        }
    }

    paths.sort();
    paths
}
