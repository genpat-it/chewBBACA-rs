//! schema_audit — worst-case minimizer-overlap audit for chewcall's pre-filter.
//!
//! Reference implementation of Algorithm 2 (SchemaAudit) of the paper. For each
//! locus in a chewBBACA schema, scans every allele in the full locus FASTA and
//! asks: at the given (k, w), what is the maximum minimizer-overlap (MO,
//! query-normalised) between this allele and any representative in `short/`?
//!
//! The locus-level statistic is then `min over alleles of max over reps`, i.e.
//! the worst-case recall the minimizer pre-filter offers on this locus when a
//! query happens to look exactly like one of the existing alleles. Loci whose
//! worst-case recall falls below the requested threshold are flagged.
//!
//! Output is a TSV plus a one-line recommendation suitable for use with
//! `chewcall --minimizer-threshold ...`.
//!
//! Example:
//!     chewcall-schema-audit --schema /path/to/schema --k 5 --w 5 --threshold 0.20

use clap::Parser;
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::path::PathBuf;

use chewcall::translate;

#[derive(Parser, Debug)]
#[command(name = "chewcall-schema-audit", version)]
#[command(about = "Schema-side audit of chewcall's minimizer-overlap pre-filter (computes per-locus wcr)")]
struct Cli {
    /// Schema directory (chewBBACA layout: locus FASTA files + short/ subdir)
    #[arg(short = 'g', long)]
    schema: PathBuf,

    /// Output TSV path (default: stdout)
    #[arg(short = 'o', long)]
    out: Option<PathBuf>,

    /// Minimizer k-mer size to evaluate
    #[arg(long, default_value = "5")]
    k: usize,

    /// Minimizer window size to evaluate
    #[arg(long, default_value = "5")]
    w: usize,

    /// Threshold to compare against; loci whose worst-case minimizer-overlap falls
    /// below this are flagged.
    #[arg(long, default_value = "0.20")]
    threshold: f64,

    /// Translation table (genetic code) for DNA-to-protein conversion
    #[arg(short = 't', long, default_value = "11")]
    translation_table: u8,

    /// Number of CPU threads (rayon)
    #[arg(long, default_value = "1")]
    cpu: usize,

    /// Print only the per-locus rows that fall below threshold (in addition to the summary)
    #[arg(long)]
    failing_only: bool,

    /// Exclude inferred alleles (FASTA headers ending in `*N`) from the audit.
    /// Inferred alleles are by definition outliers added by previous runs and
    /// are typically rare in subsequent query genomes; including them in the
    /// recall metric biases the worst-case downward. Recommended for production
    /// schemas. Default: false (keep all alleles for a strictly conservative view).
    #[arg(long)]
    exclude_inferred: bool,
}

#[derive(Debug, Clone)]
struct LocusReport {
    name: String,
    n_reps: usize,
    n_alleles: usize,
    prot_min: usize,
    prot_max: usize,
    /// min over alleles of (max over reps of Jaccard)
    worst_recall: f64,
    /// allele identifier exhibiting the worst recall (best-effort header parse)
    worst_allele: String,
}

fn fnv1a(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001B3;
    let mut hash = FNV_OFFSET;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Extract the minimizer set of a protein sequence using the same hash-based
/// algorithm as chewcall's Phase 4 pre-filter (FNV-1a + sliding window).
/// Returns a deduplicated, sorted set of u64 minimizer hashes.
fn minimizers(seq: &[u8], k: usize, w: usize) -> FxHashSet<u64> {
    let mut out = FxHashSet::default();
    if seq.len() < k {
        return out;
    }
    let n = seq.len() - k + 1;
    let kmer_hashes: Vec<u64> = (0..n).map(|i| fnv1a(&seq[i..i + k])).collect();
    if kmer_hashes.len() < w {
        if let Some(&m) = kmer_hashes.iter().min() {
            out.insert(m);
        }
        return out;
    }
    for window_start in 0..=(kmer_hashes.len() - w) {
        let m = *kmer_hashes[window_start..window_start + w].iter().min().unwrap();
        out.insert(m);
    }
    out
}

fn read_proteins(path: &std::path::Path, table: u8) -> Vec<(String, Vec<u8>)> {
    use needletail::parse_fastx_file;
    let mut out = Vec::new();
    let Ok(mut reader) = parse_fastx_file(path) else {
        return out;
    };
    while let Some(rec) = reader.next() {
        let Ok(rec) = rec else { continue };
        let id = std::str::from_utf8(rec.id())
            .unwrap_or("")
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_string();
        let upper: Vec<u8> = rec.seq().iter().map(|b| b.to_ascii_uppercase()).collect();
        if let Some(p) = translate::translate_cds(&upper, table, true) {
            if !p.is_empty() {
                out.push((id, p));
            }
        }
    }
    out
}

/// Query-normalised minimizer containment J_M(p, r) = |M(p) ∩ M(r)| / |M(p)|.
/// Asymmetric: |M(p)| is the query's minimizer set size, used to align with the
/// paper's `J_M` definition and with the runtime filter operator in cluster.rs.
/// When the query has no minimizers (short proteins), J_M is defined as 0 by
/// convention, so such queries are never retained by the filter.
fn j_m(a: &FxHashSet<u64>, b: &FxHashSet<u64>) -> f64 {
    if a.is_empty() {
        0.0
    } else {
        let inter = a.intersection(b).count();
        inter as f64 / a.len() as f64
    }
}

/// Return triples of (locus_name, full_locus_fasta, short_rep_fasta).
/// Walks the schema's `short/` directory to enumerate loci, then locates the
/// matching full FASTA in the schema root.
fn list_loci(schema_dir: &std::path::Path) -> Vec<(String, PathBuf, PathBuf)> {
    let short_dir = schema_dir.join("short");
    let read_dir = match std::fs::read_dir(&short_dir) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", short_dir.display(), e);
            std::process::exit(2);
        }
    };
    let mut out = Vec::new();
    for entry in read_dir.flatten() {
        let short_path = entry.path();
        let name = match short_path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        if !name.ends_with(".fasta") && !name.ends_with(".fna") && !name.ends_with(".fa") {
            continue;
        }
        let stem = name
            .trim_end_matches(".fasta")
            .trim_end_matches(".fna")
            .trim_end_matches(".fa")
            .trim_end_matches("_short")
            .to_string();
        let full = schema_dir.join(format!("{}.fasta", stem));
        if !full.exists() {
            continue;
        }
        out.push((stem, full, short_path));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn main() {
    let cli = Cli::parse();
    rayon::ThreadPoolBuilder::new()
        .num_threads(cli.cpu)
        .build_global()
        .ok();

    let loci = list_loci(&cli.schema);
    eprintln!(
        "Auditing {} loci from {} at (k={}, w={}, threshold={}) ...",
        loci.len(),
        cli.schema.display(),
        cli.k,
        cli.w,
        cli.threshold
    );

    let reports: Vec<LocusReport> = loci
        .par_iter()
        .filter_map(|(name, full_path, short_path)| {
            let mut alleles = read_proteins(full_path, cli.translation_table);
            let reps = read_proteins(short_path, cli.translation_table);
            if reps.is_empty() || alleles.is_empty() {
                return None;
            }
            if cli.exclude_inferred {
                alleles.retain(|(id, _)| {
                    id.rsplit('_').next().map_or(true, |suf| !suf.starts_with('*'))
                });
                if alleles.is_empty() {
                    return None;
                }
            }
            let rep_mins: Vec<FxHashSet<u64>> = reps
                .iter()
                .map(|(_, p)| minimizers(p, cli.k, cli.w))
                .collect();
            let prot_min = alleles.iter().map(|(_, p)| p.len()).min().unwrap_or(0);
            let prot_max = alleles.iter().map(|(_, p)| p.len()).max().unwrap_or(0);

            // For each allele, compute the best Jaccard against any rep; track
            // the worst (allele, rep) pair so we can name the offending allele.
            let mut worst = 1.0_f64;
            let mut worst_id = String::new();
            for (aid, prot) in &alleles {
                let amins = minimizers(prot, cli.k, cli.w);
                let best = rep_mins
                    .iter()
                    .map(|rm| j_m(&amins, rm))
                    .fold(0.0_f64, f64::max);
                if best < worst {
                    worst = best;
                    worst_id = aid.clone();
                }
            }
            Some(LocusReport {
                name: name.clone(),
                n_reps: reps.len(),
                n_alleles: alleles.len(),
                prot_min,
                prot_max,
                worst_recall: worst,
                worst_allele: worst_id,
            })
        })
        .collect();

    // Emit TSV
    let mut writer: Box<dyn std::io::Write> = match &cli.out {
        Some(p) => Box::new(std::io::BufWriter::new(
            std::fs::File::create(p).expect("cannot open out file"),
        )),
        None => Box::new(std::io::BufWriter::new(std::io::stdout().lock())),
    };
    use std::io::Write;
    writeln!(
        writer,
        "locus\tn_reps\tn_alleles\tprot_min\tprot_max\tworst_recall\tworst_allele\tflagged"
    )
    .ok();
    for r in &reports {
        let flagged = r.worst_recall < cli.threshold;
        if cli.failing_only && !flagged {
            continue;
        }
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{}",
            r.name,
            r.n_reps,
            r.n_alleles,
            r.prot_min,
            r.prot_max,
            r.worst_recall,
            r.worst_allele,
            if flagged { "yes" } else { "no" }
        )
        .ok();
    }
    drop(writer);

    // Summary on stderr
    let n_total = reports.len();
    let n_flagged = reports.iter().filter(|r| r.worst_recall < cli.threshold).count();
    let global_min = reports
        .iter()
        .map(|r| r.worst_recall)
        .fold(1.0_f64, f64::min);
    let recommended = if n_flagged == 0 {
        cli.threshold
    } else {
        ((global_min * 100.0).floor() / 100.0).max(0.05)
    };

    eprintln!();
    eprintln!("=== chewcall tune-minimizer summary ===");
    eprintln!("loci scanned                : {}", n_total);
    eprintln!(
        "flagged (worst_recall < {:.2}): {}",
        cli.threshold, n_flagged
    );
    eprintln!(
        "global worst recall (minimum allele-vs-best-rep Jaccard): {:.4}",
        global_min
    );
    if n_flagged == 0 {
        eprintln!(
            "✓ all loci satisfy the requested threshold; default --minimizer-threshold {:.2} is safe",
            cli.threshold
        );
    } else {
        eprintln!(
            "→ recommend running chewcall with: --minimizer-threshold {:.2}",
            recommended
        );
        eprintln!(
            "  (the smallest fraction reachable by any allele against the closest rep is {:.4};",
            global_min
        );
        eprintln!(
            "   raising the rep set via SchemaEvaluator/CreateSchema is the principled remedy.)"
        );
    }
}
