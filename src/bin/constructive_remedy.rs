//! constructive_remedy — make a chewBBACA schema's minimizer pre-filter safe by
//! construction.
//!
//! Companion to `schema_audit` (Algorithm 2). The audit *flags* loci whose
//! worst-case minimizer overlap (wcr) falls below the filter threshold tau;
//! this tool *fixes* them. For every flagged locus it greedily promotes
//! witness alleles from the full locus FASTA into the representative set
//! (`short/`) until every allele attains query-normalised minimizer overlap
//! >= tau against some representative. Because an allele has J_M = 1 against
//! itself, promoting it guarantees coverage of that allele; the greedy thus
//! always terminates with wcr >= tau on every locus -- a constructive
//! guarantee, not a heuristic.
//!
//! The result is written as a complete, self-contained enriched schema: every
//! input file is hard-linked (cheap) into the output directory, and only the
//! `short/` FASTAs of modified loci are rewritten with the promoted alleles
//! appended.
//!
//! Example:
//!     constructive_remedy -g schema/ -o schema_enriched/ --k 5 --w 5 --threshold 0.20

use clap::Parser;
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::path::{Path, PathBuf};

use chewcall::translate;

#[derive(Parser, Debug)]
#[command(name = "constructive_remedy", version)]
#[command(about = "Promote witness alleles so a schema satisfies wcr >= tau by construction")]
struct Cli {
    /// Input schema directory (chewBBACA layout: locus FASTA + short/ subdir)
    #[arg(short = 'g', long)]
    schema: PathBuf,

    /// Output directory for the enriched schema (must not already exist)
    #[arg(short = 'o', long)]
    out: PathBuf,

    /// TSV report of per-locus promotions (default: <out>/constructive_remedy_report.tsv)
    #[arg(long)]
    report: Option<PathBuf>,

    /// Minimizer k-mer size
    #[arg(long, default_value = "5")]
    k: usize,

    /// Minimizer window size
    #[arg(long, default_value = "5")]
    w: usize,

    /// Target worst-case minimizer overlap (the filter threshold tau)
    #[arg(long, default_value = "0.20")]
    threshold: f64,

    /// Translation table (genetic code)
    #[arg(short = 't', long, default_value = "11")]
    translation_table: u8,

    /// Number of CPU threads (rayon)
    #[arg(long, default_value = "1")]
    cpu: usize,

    /// Exclude inferred alleles (headers ending in `*N`) from the coverage target.
    #[arg(long)]
    exclude_inferred: bool,
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

/// Minimizer set of a protein, identical to chewcall's Phase 4 pre-filter.
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
        let m = *kmer_hashes[window_start..window_start + w]
            .iter()
            .min()
            .unwrap();
        out.insert(m);
    }
    out
}

/// Query-normalised minimizer overlap J_M(p, r) = |M(p) ∩ M(r)| / |M(p)|.
fn j_m(a: &FxHashSet<u64>, b: &FxHashSet<u64>) -> f64 {
    if a.is_empty() {
        0.0
    } else {
        a.intersection(b).count() as f64 / a.len() as f64
    }
}

/// An allele with its raw DNA record (header line + sequence) and protein.
struct Allele {
    id: String,
    header: String,
    dna: Vec<u8>,
    protein: Vec<u8>,
}

fn read_alleles(path: &Path, table: u8) -> Vec<Allele> {
    use needletail::parse_fastx_file;
    let mut out = Vec::new();
    let Ok(mut reader) = parse_fastx_file(path) else {
        return out;
    };
    while let Some(rec) = reader.next() {
        let Ok(rec) = rec else { continue };
        let header = std::str::from_utf8(rec.id()).unwrap_or("").to_string();
        let id = header.split_whitespace().next().unwrap_or("").to_string();
        let dna: Vec<u8> = rec.seq().iter().map(|b| b.to_ascii_uppercase()).collect();
        if let Some(protein) = translate::translate_cds(&dna, table, true) {
            if !protein.is_empty() {
                out.push(Allele {
                    id,
                    header,
                    dna,
                    protein,
                });
            }
        }
    }
    out
}

/// (locus_stem, full_fasta, short_fasta)
fn list_loci(schema_dir: &Path) -> Vec<(String, PathBuf, PathBuf)> {
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

/// Recursively hard-link every file under `src` into `dst` (falling back to a
/// byte copy when hard-linking fails, e.g. across filesystems).
fn hardlink_tree(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            hardlink_tree(&from, &to)?;
        } else if std::fs::hard_link(&from, &to).is_err() {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

#[derive(Debug)]
struct LocusOutcome {
    name: String,
    n_alleles: usize,
    wcr_before: f64,
    wcr_after: f64,
    /// IDs of alleles promoted to representatives (in promotion order)
    promoted: Vec<String>,
    short_file: String,
}

/// Greedily select witness alleles to promote so that every allele attains
/// J_M >= tau against the (growing) representative set. Returns the indices of
/// promoted alleles and the resulting worst-case recall.
fn greedy_promote(
    allele_mins: &[FxHashSet<u64>],
    rep_mins: &[FxHashSet<u64>],
    tau: f64,
) -> (Vec<usize>, f64, f64) {
    let n = allele_mins.len();
    // best[i] = max J_M(allele_i, rep) over the current representative set.
    let mut best: Vec<f64> = allele_mins
        .iter()
        .map(|am| {
            rep_mins
                .iter()
                .map(|rm| j_m(am, rm))
                .fold(0.0_f64, f64::max)
        })
        .collect();
    let wcr_before = best.iter().cloned().fold(1.0_f64, f64::min);

    let mut promoted = Vec::new();
    loop {
        // Find the allele with the lowest current coverage.
        let mut worst_i = 0usize;
        let mut worst_v = f64::INFINITY;
        for i in 0..n {
            if best[i] < worst_v {
                worst_v = best[i];
                worst_i = i;
            }
        }
        if worst_v >= tau {
            break;
        }
        // Promote it: it now covers itself (J_M = 1) and raises coverage of any
        // allele sharing minimizers with it.
        promoted.push(worst_i);
        let new_rep = &allele_mins[worst_i];
        for i in 0..n {
            if best[i] < tau {
                let s = j_m(&allele_mins[i], new_rep);
                if s > best[i] {
                    best[i] = s;
                }
            }
        }
    }
    let wcr_after = best.iter().cloned().fold(1.0_f64, f64::min);
    (promoted, wcr_before, wcr_after)
}

fn main() {
    let cli = Cli::parse();
    rayon::ThreadPoolBuilder::new()
        .num_threads(cli.cpu)
        .build_global()
        .ok();

    if cli.out.exists() {
        eprintln!(
            "error: output directory {} already exists; refusing to overwrite",
            cli.out.display()
        );
        std::process::exit(2);
    }

    let loci = list_loci(&cli.schema);
    eprintln!(
        "constructive_remedy: {} loci from {} at (k={}, w={}, tau={})",
        loci.len(),
        cli.schema.display(),
        cli.k,
        cli.w,
        cli.threshold
    );

    // Compute promotions per locus in parallel (pure analysis, no I/O yet).
    let outcomes: Vec<LocusOutcome> = loci
        .par_iter()
        .filter_map(|(name, full_path, short_path)| {
            let mut alleles = read_alleles(full_path, cli.translation_table);
            let reps = read_alleles(short_path, cli.translation_table);
            if alleles.is_empty() || reps.is_empty() {
                return None;
            }
            if cli.exclude_inferred {
                alleles.retain(|a| {
                    a.id.rsplit('_')
                        .next()
                        .map_or(true, |suf| !suf.starts_with('*'))
                });
                if alleles.is_empty() {
                    return None;
                }
            }
            let allele_mins: Vec<FxHashSet<u64>> = alleles
                .iter()
                .map(|a| minimizers(&a.protein, cli.k, cli.w))
                .collect();
            let rep_mins: Vec<FxHashSet<u64>> = reps
                .iter()
                .map(|r| minimizers(&r.protein, cli.k, cli.w))
                .collect();

            let (promoted_idx, wcr_before, wcr_after) =
                greedy_promote(&allele_mins, &rep_mins, cli.threshold);

            let short_file = short_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();

            // Promoted allele DNA is resolved later (writer loop) by re-reading
            // the full FASTA; here we only keep the promoted IDs.
            Some(LocusOutcome {
                name: name.clone(),
                n_alleles: alleles.len(),
                wcr_before,
                wcr_after,
                promoted: promoted_idx
                    .iter()
                    .map(|&i| alleles[i].id.clone())
                    .collect(),
                short_file,
            })
        })
        .collect();

    // Materialise the enriched schema: hard-link the whole tree, then rewrite
    // the short/ FASTAs of modified loci with promoted alleles appended.
    if let Err(e) = hardlink_tree(&cli.schema, &cli.out) {
        eprintln!(
            "error: failed to stage schema into {}: {}",
            cli.out.display(),
            e
        );
        std::process::exit(2);
    }

    let loci_by_name: std::collections::HashMap<&str, &(String, PathBuf, PathBuf)> =
        loci.iter().map(|t| (t.0.as_str(), t)).collect();

    let mut n_modified = 0usize;
    let mut n_promoted_total = 0usize;
    for o in &outcomes {
        if o.promoted.is_empty() {
            continue;
        }
        let Some((_, full_path, short_path)) = loci_by_name.get(o.name.as_str()).copied() else {
            continue;
        };
        // Resolve promoted DNA records from the full FASTA.
        let promoted_set: FxHashSet<&str> = o.promoted.iter().map(|s| s.as_str()).collect();
        let records = read_alleles(full_path, cli.translation_table);
        let mut appended = String::new();
        for rec in &records {
            if promoted_set.contains(rec.id.as_str()) {
                appended.push('>');
                appended.push_str(&rec.header);
                appended.push('\n');
                appended.push_str(std::str::from_utf8(&rec.dna).unwrap_or(""));
                appended.push('\n');
            }
        }
        // Read original short content, then write enriched copy (break the hardlink first).
        let original = std::fs::read_to_string(short_path).unwrap_or_default();
        let out_short = cli.out.join("short").join(&o.short_file);
        let _ = std::fs::remove_file(&out_short); // break hard link to the original
        let mut content = original;
        if !content.ends_with('\n') && !content.is_empty() {
            content.push('\n');
        }
        content.push_str(&appended);
        if let Err(e) = std::fs::write(&out_short, content) {
            eprintln!("warning: could not write {}: {}", out_short.display(), e);
            continue;
        }
        n_modified += 1;
        n_promoted_total += o.promoted.len();
    }

    // Report TSV.
    let report_path = cli
        .report
        .clone()
        .unwrap_or_else(|| cli.out.join("constructive_remedy_report.tsv"));
    if let Ok(mut f) = std::fs::File::create(&report_path) {
        use std::io::Write;
        writeln!(
            f,
            "locus\tn_alleles\twcr_before\twcr_after\tn_promoted\tpromoted_alleles"
        )
        .ok();
        for o in &outcomes {
            writeln!(
                f,
                "{}\t{}\t{:.4}\t{:.4}\t{}\t{}",
                o.name,
                o.n_alleles,
                o.wcr_before,
                o.wcr_after,
                o.promoted.len(),
                o.promoted.join(",")
            )
            .ok();
        }
    }

    let global_before = outcomes
        .iter()
        .map(|o| o.wcr_before)
        .fold(1.0_f64, f64::min);
    let global_after = outcomes.iter().map(|o| o.wcr_after).fold(1.0_f64, f64::min);
    eprintln!();
    eprintln!("=== constructive_remedy summary ===");
    eprintln!("loci processed        : {}", outcomes.len());
    eprintln!("loci modified         : {}", n_modified);
    eprintln!("alleles promoted      : {}", n_promoted_total);
    eprintln!("global wcr before     : {:.4}", global_before);
    eprintln!("global wcr after      : {:.4}", global_after);
    if global_after + 1e-9 >= cli.threshold {
        eprintln!(
            "✓ every locus now satisfies wcr >= {:.2} by construction",
            cli.threshold
        );
    } else {
        eprintln!(
            "✗ unexpected: global wcr {:.4} < tau {:.2} (please report)",
            global_after, cli.threshold
        );
    }
    eprintln!("enriched schema       : {}", cli.out.display());
    eprintln!("report                : {}", report_path.display());
}
