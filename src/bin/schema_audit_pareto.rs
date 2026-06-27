//! schema_audit_pareto — parameter-sweep audit for chewcall's minimizer-overlap pre-filter.
//!
//! Exploratory companion to `schema_audit`. Instead of evaluating one fixed
//! (k, w, τ) tuple, it sweeps a user-specified grid over four parameters
//! (k, w, τ, κ) and reports for each tuple:
//!   - n_flagged = |V_τ|  (loci with wcr < τ at this k, w)
//!   - mean / median / 10th-percentile / global-min of per-locus wcr
//!   - sw_work_proxy = worst-case number of exact-SW evaluations the cluster
//!     stage would trigger: per query allele, min(κ, #reps with MO ≥ τ),
//!     summed over the schema (κ = 0 means an unbounded candidate cap)
//!   - mean_M = mean number of minimizers per schema allele at this (k, w)
//! It then marks the Pareto-optimal tuples on the bi-objective
//! (n_flagged, sw_work_proxy) — fewer flagged loci AND less SW work, both
//! minimized.
//!
//! κ (the top-K candidate cap, `--max-targets` in chewcall's cluster stage)
//! does NOT change best-match recall under the per-locus audit model: the
//! highest-MO representative of a locus is always rank 1 and always survives a
//! top-κ truncation for κ ≥ 1. κ therefore only bounds SW work here; its
//! residual cross-locus recall effect (a same-locus rep displaced from the
//! global top-κ by higher-MO reps of OTHER loci) is outside the per-locus
//! model and is a documented scope limitation of the audit.
//!
//! This is a stand-alone exploratory tool and does NOT replace `schema_audit`;
//! the latter remains the canonical single-point audit used in the paper.
//!
//! Example:
//!     schema_audit_pareto \
//!         --schema /path/to/schema \
//!         --k-values 4,5,6,7 \
//!         --w-values 3,5,7,9 \
//!         --tau-values 0.10,0.15,0.20,0.25,0.30 \
//!         --kappa-values 5,10,20,30,0 \
//!         --out pareto.tsv \
//!         --cpu 8

use clap::Parser;
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::path::PathBuf;

use chewcall::translate;

#[derive(Parser, Debug)]
#[command(name = "chewcall-schema-audit-pareto", version)]
#[command(
    about = "Pareto-frontier (k, w, τ, κ) parameter sweep for chewcall's minimizer-overlap pre-filter"
)]
struct Cli {
    /// Schema directory (chewBBACA layout: locus FASTA files + short/ subdir)
    #[arg(short = 'g', long)]
    schema: PathBuf,

    /// Output TSV path (default: stdout)
    #[arg(short = 'o', long)]
    out: Option<PathBuf>,

    /// Comma-separated list of k-mer sizes to evaluate (e.g. "4,5,6,7")
    #[arg(long, default_value = "4,5,6,7")]
    k_values: String,

    /// Comma-separated list of window sizes to evaluate (e.g. "3,5,7,9")
    #[arg(long, default_value = "3,5,7,9")]
    w_values: String,

    /// Comma-separated list of thresholds to evaluate (e.g. "0.10,0.15,0.20,0.25,0.30")
    #[arg(long, default_value = "0.10,0.15,0.20,0.25,0.30")]
    tau_values: String,

    /// Comma-separated list of top-K candidate caps to evaluate (e.g. "5,10,20,30,0").
    /// 0 means an unbounded cap (every representative above τ is a candidate).
    #[arg(long, default_value = "5,10,20,30,0")]
    kappa_values: String,

    /// Translation table for DNA-to-protein conversion
    #[arg(short = 't', long, default_value = "11")]
    translation_table: u8,

    /// Number of CPU threads (rayon)
    #[arg(long, default_value = "1")]
    cpu: usize,

    /// Exclude inferred alleles (FASTA headers ending in `*N`) from the audit.
    #[arg(long)]
    exclude_inferred: bool,
}

fn parse_csv_usize(s: &str) -> Vec<usize> {
    s.split(',')
        .filter_map(|x| x.trim().parse::<usize>().ok())
        .collect()
}

fn parse_csv_f64(s: &str) -> Vec<f64> {
    s.split(',')
        .filter_map(|x| x.trim().parse::<f64>().ok())
        .collect()
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

fn j_m(a: &FxHashSet<u64>, b: &FxHashSet<u64>) -> f64 {
    if a.is_empty() {
        0.0
    } else {
        let inter = a.intersection(b).count();
        inter as f64 / a.len() as f64
    }
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

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p / 100.0).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[derive(Debug, Clone)]
struct TupleResult {
    k: usize,
    w: usize,
    tau: f64,
    kappa: usize,
    n_total: usize,
    n_flagged: usize,
    mean_wcr: f64,
    median_wcr: f64,
    p10_wcr: f64,
    min_wcr: f64,
    sw_work_proxy: usize,
    mean_n_minimizers: f64,
    pareto: bool,
}

/// Per-(k, w) precomputed structure.
struct KwAudit {
    /// per_locus_best[i][j] = max-over-reps MO for allele j of locus i
    per_locus_best: Vec<Vec<f64>>,
    /// per_allele_mos[a] = MOs of allele a against each same-locus rep
    /// (flattened across loci; used for the κ-capped sw_work proxy)
    per_allele_mos: Vec<Vec<f64>>,
    /// mean |M(p)| over all schema alleles
    mean_n_min: f64,
}

fn run_kw_audit(
    locus_alleles: &[Vec<Vec<u8>>],
    locus_reps: &[Vec<Vec<u8>>],
    k: usize,
    w: usize,
) -> KwAudit {
    let per_locus: Vec<(Vec<f64>, Vec<Vec<f64>>, usize, usize)> = locus_alleles
        .par_iter()
        .zip(locus_reps.par_iter())
        .map(|(alleles, reps)| {
            let rep_mins: Vec<FxHashSet<u64>> = reps.iter().map(|p| minimizers(p, k, w)).collect();
            let mut best_per_allele = Vec::with_capacity(alleles.len());
            let mut allele_mos = Vec::with_capacity(alleles.len());
            let mut total_min = 0usize;
            for prot in alleles {
                let amins = minimizers(prot, k, w);
                total_min += amins.len();
                let mut mos = Vec::with_capacity(rep_mins.len());
                let mut best = 0.0_f64;
                for rm in &rep_mins {
                    let mo = j_m(&amins, rm);
                    mos.push(mo);
                    if mo > best {
                        best = mo;
                    }
                }
                best_per_allele.push(best);
                allele_mos.push(mos);
            }
            (best_per_allele, allele_mos, total_min, alleles.len())
        })
        .collect();

    let mut per_locus_best = Vec::with_capacity(per_locus.len());
    let mut per_allele_mos = Vec::new();
    let mut total_min_global = 0usize;
    let mut total_alleles = 0usize;
    for (best, mos, tmin, na) in per_locus {
        per_locus_best.push(best);
        per_allele_mos.extend(mos);
        total_min_global += tmin;
        total_alleles += na;
    }
    let mean_n_min = if total_alleles > 0 {
        total_min_global as f64 / total_alleles as f64
    } else {
        0.0
    };
    KwAudit {
        per_locus_best,
        per_allele_mos,
        mean_n_min,
    }
}

fn aggregate(audit: &KwAudit, k: usize, w: usize, tau: f64, kappa: usize) -> TupleResult {
    let mut per_locus_wcr = Vec::with_capacity(audit.per_locus_best.len());
    for best in &audit.per_locus_best {
        if best.is_empty() {
            continue;
        }
        let wcr = best.iter().cloned().fold(1.0_f64, f64::min);
        per_locus_wcr.push(wcr);
    }
    per_locus_wcr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n_total = per_locus_wcr.len();
    let n_flagged = per_locus_wcr.iter().filter(|x| **x < tau).count();
    let sum: f64 = per_locus_wcr.iter().sum();
    let mean_wcr = if n_total > 0 {
        sum / n_total as f64
    } else {
        0.0
    };
    let median_wcr = percentile(&per_locus_wcr, 50.0);
    let p10_wcr = percentile(&per_locus_wcr, 10.0);
    let min_wcr = if n_total > 0 { per_locus_wcr[0] } else { 0.0 };

    // κ-capped SW work: per query allele, min(κ, #reps above τ), summed.
    let sw_work_proxy: usize = audit
        .per_allele_mos
        .iter()
        .map(|mos| {
            let c = mos.iter().filter(|x| **x >= tau).count();
            if kappa == 0 {
                c
            } else {
                c.min(kappa)
            }
        })
        .sum();

    TupleResult {
        k,
        w,
        tau,
        kappa,
        n_total,
        n_flagged,
        mean_wcr,
        median_wcr,
        p10_wcr,
        min_wcr,
        sw_work_proxy,
        mean_n_minimizers: audit.mean_n_min,
        pareto: false,
    }
}

fn pareto_frontier(results: &mut [TupleResult]) {
    // Both n_flagged and sw_work_proxy are minimized
    let n = results.len();
    for i in 0..n {
        let mut dominated = false;
        for j in 0..n {
            if i == j {
                continue;
            }
            let a = &results[j];
            let b = &results[i];
            let leq = a.n_flagged <= b.n_flagged && a.sw_work_proxy <= b.sw_work_proxy;
            let lt = a.n_flagged < b.n_flagged || a.sw_work_proxy < b.sw_work_proxy;
            if leq && lt {
                dominated = true;
                break;
            }
        }
        results[i].pareto = !dominated;
    }
}

fn kappa_label(kappa: usize) -> String {
    if kappa == 0 {
        "inf".to_string()
    } else {
        kappa.to_string()
    }
}

fn main() {
    let cli = Cli::parse();
    rayon::ThreadPoolBuilder::new()
        .num_threads(cli.cpu)
        .build_global()
        .ok();

    let loci = list_loci(&cli.schema);
    let ks = parse_csv_usize(&cli.k_values);
    let ws = parse_csv_usize(&cli.w_values);
    let taus = parse_csv_f64(&cli.tau_values);
    let kappas = parse_csv_usize(&cli.kappa_values);

    eprintln!(
        "Pareto sweep: schema={}, |loci|={}, |K|={}, |W|={}, |τ|={}, |κ|={}, total tuples={}",
        cli.schema.display(),
        loci.len(),
        ks.len(),
        ws.len(),
        taus.len(),
        kappas.len(),
        ks.len() * ws.len() * taus.len() * kappas.len()
    );
    eprintln!(
        "Sweeping K={:?}, W={:?}, τ={:?}, κ={:?} (0=unbounded)",
        ks, ws, taus, kappas
    );

    // Read all proteins once per locus (one pass over disk)
    eprintln!("Reading and translating proteins (one pass)...");
    let read_t0 = std::time::Instant::now();
    let per_locus: Vec<(String, Vec<Vec<u8>>, Vec<Vec<u8>>)> = loci
        .par_iter()
        .filter_map(|(name, full_path, short_path)| {
            let mut alleles = read_proteins(full_path, cli.translation_table);
            let reps = read_proteins(short_path, cli.translation_table);
            if reps.is_empty() || alleles.is_empty() {
                return None;
            }
            if cli.exclude_inferred {
                alleles.retain(|(id, _)| {
                    id.rsplit('_')
                        .next()
                        .map_or(true, |suf| !suf.starts_with('*'))
                });
                if alleles.is_empty() {
                    return None;
                }
            }
            let allele_seqs: Vec<Vec<u8>> = alleles.into_iter().map(|(_, p)| p).collect();
            let rep_seqs: Vec<Vec<u8>> = reps.into_iter().map(|(_, p)| p).collect();
            Some((name.clone(), allele_seqs, rep_seqs))
        })
        .collect();
    eprintln!(
        "  → {} loci loaded in {:.1}s",
        per_locus.len(),
        read_t0.elapsed().as_secs_f64()
    );

    let locus_alleles: Vec<Vec<Vec<u8>>> = per_locus.iter().map(|(_, a, _)| a.clone()).collect();
    let locus_reps: Vec<Vec<Vec<u8>>> = per_locus.iter().map(|(_, _, r)| r.clone()).collect();

    let mut all_results: Vec<TupleResult> = Vec::new();
    for &k in &ks {
        for &w in &ws {
            let t0 = std::time::Instant::now();
            let audit = run_kw_audit(&locus_alleles, &locus_reps, k, w);
            let kw_t = t0.elapsed().as_secs_f64();
            let n_pairs: usize = audit.per_allele_mos.iter().map(|m| m.len()).sum();
            eprintln!(
                "  (k={}, w={}): pair-MOs={}, mean |M|={:.2}, audit in {:.1}s",
                k, w, n_pairs, audit.mean_n_min, kw_t
            );
            for &tau in &taus {
                for &kappa in &kappas {
                    all_results.push(aggregate(&audit, k, w, tau, kappa));
                }
            }
        }
    }

    pareto_frontier(&mut all_results);

    // Output TSV
    let mut writer: Box<dyn std::io::Write> = match &cli.out {
        Some(p) => Box::new(std::io::BufWriter::new(
            std::fs::File::create(p).expect("cannot open out file"),
        )),
        None => Box::new(std::io::BufWriter::new(std::io::stdout().lock())),
    };
    use std::io::Write;
    writeln!(
        writer,
        "k\tw\ttau\tkappa\tn_loci\tn_flagged\tflagged_pct\tmean_wcr\tmedian_wcr\tp10_wcr\tmin_wcr\tsw_work_proxy\tmean_n_minimizers\tpareto"
    )
    .ok();
    all_results.sort_by(|a, b| {
        (a.k, a.w, (a.tau * 1e9) as i64, a.kappa).cmp(&(b.k, b.w, (b.tau * 1e9) as i64, b.kappa))
    });
    for r in &all_results {
        let pct = if r.n_total > 0 {
            100.0 * r.n_flagged as f64 / r.n_total as f64
        } else {
            0.0
        };
        writeln!(
            writer,
            "{}\t{}\t{:.2}\t{}\t{}\t{}\t{:.3}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{:.2}\t{}",
            r.k,
            r.w,
            r.tau,
            kappa_label(r.kappa),
            r.n_total,
            r.n_flagged,
            pct,
            r.mean_wcr,
            r.median_wcr,
            r.p10_wcr,
            r.min_wcr,
            r.sw_work_proxy,
            r.mean_n_minimizers,
            if r.pareto { "yes" } else { "no" }
        )
        .ok();
    }
    drop(writer);

    // Summary on stderr: list Pareto frontier
    eprintln!();
    eprintln!("=== Pareto frontier on (n_flagged ↓, sw_work_proxy ↓) ===");
    let frontier: Vec<&TupleResult> = all_results.iter().filter(|r| r.pareto).collect();
    let mut frontier_sorted = frontier.clone();
    frontier_sorted.sort_by(|a, b| {
        a.n_flagged
            .cmp(&b.n_flagged)
            .then(a.sw_work_proxy.cmp(&b.sw_work_proxy))
    });
    eprintln!(
        "{:>4}  {:>4}  {:>5}  {:>5}  {:>10}  {:>14}  {:>10}",
        "k", "w", "tau", "kappa", "n_flagged", "sw_work_proxy", "mean_wcr"
    );
    for r in &frontier_sorted {
        eprintln!(
            "{:>4}  {:>4}  {:>5.2}  {:>5}  {:>10}  {:>14}  {:>10.4}",
            r.k,
            r.w,
            r.tau,
            kappa_label(r.kappa),
            r.n_flagged,
            r.sw_work_proxy,
            r.mean_wcr
        );
    }
    eprintln!();
    eprintln!(
        "Total tuples evaluated: {} ({} on Pareto frontier)",
        all_results.len(),
        frontier_sorted.len()
    );
    // Highlight chewBBACA / chewcall default (k=5, w=5, τ=0.20, κ=30 cluster cap)
    if let Some(default) = all_results
        .iter()
        .find(|r| r.k == 5 && r.w == 5 && (r.tau - 0.20).abs() < 1e-6 && r.kappa == 30)
    {
        eprintln!(
            "chewcall default (k=5, w=5, τ=0.20, κ=30): n_flagged={}, sw_work_proxy={}, pareto={}",
            default.n_flagged,
            default.sw_work_proxy,
            if default.pareto {
                "yes"
            } else {
                "NO (dominated)"
            }
        );
    }
}
