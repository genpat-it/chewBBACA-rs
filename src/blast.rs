//! BLAST integration for chewBBACA-compatible allele calling.
//!
//! Two operational modes:
//! 1. **Selective validation** (fast mode): re-score borderline repdet hits with BLAST
//! 2. **Full compatible mode**: replicate chewBBACA's exact BLAST flow for 100% identical output

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::cluster::ClusterResult;
use crate::types::Representative;
use crate::sw;

static BLASTP_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();

fn detect_blastp() -> Option<&'static Path> {
    BLASTP_PATH
        .get_or_init(|| {
            if let Ok(path) = std::env::var("CHEWCALL_BLASTP") {
                let candidate = PathBuf::from(path);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }

            if let Some(path) = std::env::var_os("PATH").and_then(|path| {
                std::env::split_paths(&path)
                    .map(|dir| dir.join("blastp"))
                    .find(|candidate| candidate.is_file())
            }) {
                return Some(path);
            }

            let fallbacks = [
                "/home/IZSNT/a.deruvo/miniconda3/envs/chewbbacca_gpu/bin/blastp",
                "/home/IZSNT/a.deruvo/miniconda3/pkgs/blast-2.17.0-h66d330f_0/bin/blastp",
            ];
            fallbacks
                .iter()
                .map(PathBuf::from)
                .find(|candidate| candidate.is_file())
        })
        .as_deref()
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}_{nanos}"))
}

fn write_fasta(path: &Path, records: &[(String, Vec<u8>)]) -> std::io::Result<()> {
    let mut file = fs::File::create(path)?;
    for (id, seq) in records {
        writeln!(file, ">{id}")?;
        writeln!(file, "{}", String::from_utf8_lossy(seq))?;
    }
    Ok(())
}

#[derive(Clone)]
struct BlastRawHit {
    cds_idx: usize,
    rep_idx: usize,
    score: i32,
    qstart: u32,
    qend: u32,
    qlen: u32,
    slen: u32,
}

/// Parasail-based validation for RepDet borderline hits (deterministic, no BLAST).
///
/// For each borderline CDS, re-aligns against ALL representatives for the matched
/// locus using parasail SIMD Smith-Waterman. This ensures the best rep is found
/// (not just the top-5 from minimizer clustering) while being fully deterministic.
pub fn validate_repdet_hits_parasail(
    hits_by_locus: &FxHashMap<usize, Vec<ClusterResult>>,
    proteins_by_idx: &FxHashMap<usize, Vec<u8>>,
    representatives: &[Representative],
    bsr_threshold: f64,
) -> FxHashMap<usize, Vec<ClusterResult>> {
    use rayon::prelude::*;
    use crate::parasail_ffi;

    // Pre-build locus → rep indices mapping
    let mut locus_reps: FxHashMap<u32, Vec<usize>> = FxHashMap::default();
    for (ri, rep) in representatives.iter().enumerate() {
        locus_reps.entry(rep.locus_idx).or_default().push(ri);
    }

    let loci: Vec<usize> = hits_by_locus.keys()
        .filter(|&&locus| {
            hits_by_locus.get(&locus)
                .map(|hits| hits.iter().any(|h| h.best_bsr >= bsr_threshold))
                .unwrap_or(false)
        })
        .copied()
        .collect();

    loci.par_iter()
        .map(|&locus_idx| {
            let hits = &hits_by_locus[&locus_idx];
            let rep_indices = locus_reps.get(&(locus_idx as u32)).cloned().unwrap_or_default();

            let mut best_per_cds: FxHashMap<usize, ClusterResult> = FxHashMap::default();

            for hit in hits {
                if hit.best_bsr < bsr_threshold { continue; }
                let Some(protein) = proteins_by_idx.get(&hit.cds_idx) else { continue };

                for &rep_idx in &rep_indices {
                    let rep = &representatives[rep_idx];
                    if rep.self_score <= 0.0 { continue; }

                    let (score, _, _) = parasail_ffi::sw_simd(protein, &rep.protein_seq);
                    let bsr = score as f64 / rep.self_score;
                    if bsr < bsr_threshold { continue; }

                    let is_better = match best_per_cds.get(&hit.cds_idx) {
                        None => true,
                        Some(e) => score > e.score
                            || (score == e.score && rep_idx < e.representative_idx),
                    };
                    if is_better {
                        let (_, _, _, target_start, target_end) =
                            parasail_ffi::sw_simd_full(protein, &rep.protein_seq);
                        best_per_cds.insert(hit.cds_idx, ClusterResult {
                            cds_idx: hit.cds_idx,
                            representative_idx: rep_idx,
                            best_locus: locus_idx as u32,
                            best_bsr: bsr,
                            score,
                            rep_dna_len: rep.dna_length,
                            query_start: 0,
                            query_end: 0,
                            query_len: protein.len() as u32,
                            target_start,
                            target_end,
                            target_len: rep.protein_seq.len() as u32,
                        });
                    }
                }
            }

            let mut validated: Vec<_> = best_per_cds.into_values().collect();
            validated.sort_unstable_by(|a, b| b.score.cmp(&a.score).then(a.cds_idx.cmp(&b.cds_idx)));
            (locus_idx, validated)
        })
        .collect()
}

/// Re-score borderline hits for one locus with BLAST and keep only hits that
/// still satisfy the BSR threshold under BLAST's raw scores.
pub fn validate_locus_hits(
    hits: &[ClusterResult],
    proteins_by_idx: &FxHashMap<usize, Vec<u8>>,
    representatives: &[Representative],
    bsr_threshold: f64,
) -> Vec<ClusterResult> {
    let Some(blastp) = detect_blastp() else {
        return hits.to_vec();
    };

    if hits.is_empty() {
        return Vec::new();
    }

    let temp_dir = unique_temp_dir("chewcall_blast_validate");
    if fs::create_dir_all(&temp_dir).is_err() {
        return hits.to_vec();
    }

    let query_path = temp_dir.join("query.faa");
    let subject_path = temp_dir.join("subject.faa");
    let output_path = temp_dir.join("blast.tsv");

    let locus_indices: FxHashSet<_> = hits
        .iter()
        .map(|hit| representatives[hit.representative_idx].locus_idx)
        .collect();
    let mut rep_indices: Vec<usize> = representatives
        .iter()
        .enumerate()
        .filter_map(|(rep_idx, rep)| locus_indices.contains(&rep.locus_idx).then_some(rep_idx))
        .collect();
    rep_indices.sort_unstable();
    rep_indices.dedup();

    let mut query_records = Vec::with_capacity(rep_indices.len());
    for &rep_idx in &rep_indices {
        query_records.push((
            format!("rep:{rep_idx}"),
            representatives[rep_idx].protein_seq.clone(),
        ));
    }

    let mut subject_ids: FxHashSet<String> = FxHashSet::default();
    let mut subject_records = Vec::new();
    for hit in hits {
        if let Some(protein) = proteins_by_idx.get(&hit.cds_idx) {
            let id = format!("cds:{}", hit.cds_idx);
            if subject_ids.insert(id.clone()) {
                subject_records.push((id, protein.clone()));
            }
        }
    }
    for &rep_idx in &rep_indices {
        subject_records.push((
            format!("self:{rep_idx}"),
            representatives[rep_idx].protein_seq.clone(),
        ));
    }

    let status = write_fasta(&query_path, &query_records)
        .and_then(|_| write_fasta(&subject_path, &subject_records))
        .and_then(|_| {
            Command::new(blastp)
                .args([
                    "-query",
                    query_path.to_str().unwrap(),
                    "-subject",
                    subject_path.to_str().unwrap(),
                    "-out",
                    output_path.to_str().unwrap(),
                    "-outfmt",
                    "6 qseqid qstart qend qlen sseqid slen score",
                    "-max_hsps",
                    "1",
                    "-evalue",
                    "0.001",
                    "-comp_based_stats",
                    "0",
                ])
                .status()
                .map_err(std::io::Error::other)
        });

    if status.is_err() {
        let _ = fs::remove_dir_all(&temp_dir);
        return hits.to_vec();
    }

    let mut rep_self_scores: FxHashMap<usize, f64> = FxHashMap::default();
    let mut best_raw_per_cds: FxHashMap<usize, BlastRawHit> = FxHashMap::default();

    if let Ok(file) = fs::File::open(&output_path) {
        let reader = BufReader::new(file);
        for line in reader.lines().map_while(Result::ok) {
            let fields: Vec<_> = line.split('\t').collect();
            if fields.len() != 7 {
                continue;
            }

            let Some(rep_idx) = fields[0].strip_prefix("rep:").and_then(|s| s.parse::<usize>().ok()) else {
                continue;
            };
            let qstart = fields[1].parse::<u32>().ok();
            let qend = fields[2].parse::<u32>().ok();
            let qlen = fields[3].parse::<u32>().ok();
            let slen = fields[5].parse::<u32>().ok();
            let score = fields[6].parse::<i32>().ok();
            let (Some(qstart), Some(qend), Some(qlen), Some(slen), Some(score)) =
                (qstart, qend, qlen, slen, score)
            else {
                continue;
            };

            if let Some(self_idx) = fields[4]
                .strip_prefix("self:")
                .and_then(|s| s.parse::<usize>().ok())
            {
                if self_idx == rep_idx && score > 0 {
                    rep_self_scores.insert(rep_idx, score as f64);
                }
                continue;
            }

            let Some(cds_idx) = fields[4]
                .strip_prefix("cds:")
                .and_then(|s| s.parse::<usize>().ok())
            else {
                continue;
            };

            let candidate = BlastRawHit {
                cds_idx,
                rep_idx,
                score,
                qstart,
                qend,
                qlen,
                slen,
            };
            let dominated = match best_raw_per_cds.get(&cds_idx) {
                None => true,
                Some(e) => candidate.score > e.score
                    || (candidate.score == e.score && candidate.rep_idx < e.rep_idx),
            };
            if dominated {
                best_raw_per_cds.insert(cds_idx, candidate);
            }
        }
    }

    let mut validated = Vec::new();
    for raw_hit in best_raw_per_cds.into_values() {
        let cds_idx = raw_hit.cds_idx;
        let rep_idx = raw_hit.rep_idx;
        let score = raw_hit.score;
        let rep_self_score = *rep_self_scores
            .get(&rep_idx)
            .unwrap_or(&representatives[rep_idx].self_score);
        if rep_self_score <= 0.0 {
            continue;
        }
        let bsr = score as f64 / rep_self_score;
        if bsr < bsr_threshold {
            continue;
        }

        validated.push(ClusterResult {
            cds_idx,
            representative_idx: rep_idx,
            best_locus: representatives[rep_idx].locus_idx,
            best_bsr: bsr,
            score,
            rep_dna_len: representatives[rep_idx].dna_length,
            query_start: 0,
            query_end: 0,
            query_len: raw_hit.slen,
            target_start: raw_hit.qstart,
            target_end: raw_hit.qend,
            target_len: raw_hit.qlen,
        });
    }

    let _ = fs::remove_dir_all(&temp_dir);

    validated.sort_unstable_by(|a, b| b.score.cmp(&a.score).then(a.cds_idx.cmp(&b.cds_idx)));
    validated
}

/// Select representative candidates with chewBBACA's BLAST-based semantics.
///
/// Returns `None` if BLAST is unavailable or the temporary BLAST run fails,
/// allowing the caller to fall back to a local aligner implementation.
pub fn select_representative_candidates(
    cds_indices: &[usize],
    proteins_by_idx: &FxHashMap<usize, Vec<u8>>,
    bsr_threshold: f64,
) -> Option<Vec<usize>> {
    let blastp = detect_blastp()?;
    if cds_indices.is_empty() {
        return Some(Vec::new());
    }

    let mut ordered_indices: Vec<_> = cds_indices.to_vec();
    ordered_indices.sort_unstable();

    let records: Vec<_> = ordered_indices
        .iter()
        .filter_map(|idx| {
            proteins_by_idx
                .get(idx)
                .map(|protein| (format!("cds:{idx}"), protein.clone()))
        })
        .collect();
    if records.is_empty() {
        return Some(Vec::new());
    }

    let temp_dir = unique_temp_dir("chewcall_blast_select_reps");
    fs::create_dir_all(&temp_dir).ok()?;
    let fasta_path = temp_dir.join("candidates.faa");
    let output_path = temp_dir.join("blast.tsv");

    let status = write_fasta(&fasta_path, &records)
        .and_then(|_| {
            Command::new(blastp)
                .args([
                    "-query",
                    fasta_path.to_str().unwrap(),
                    "-subject",
                    fasta_path.to_str().unwrap(),
                    "-out",
                    output_path.to_str().unwrap(),
                    "-outfmt",
                    "6 qseqid qlen sseqid score",
                    "-max_hsps",
                    "1",
                    "-evalue",
                    "0.001",
                    "-comp_based_stats",
                    "0",
                ])
                .status()
                .map_err(std::io::Error::other)
        })
        .ok()?;
    if !status.success() {
        let _ = fs::remove_dir_all(&temp_dir);
        return None;
    }

    let mut self_scores: FxHashMap<usize, f64> = FxHashMap::default();
    let mut raw_hits = Vec::new();

    if let Ok(file) = fs::File::open(&output_path) {
        let reader = BufReader::new(file);
        for (line_no, line) in reader.lines().map_while(Result::ok).enumerate() {
            let fields: Vec<_> = line.split('\t').collect();
            if fields.len() != 4 {
                continue;
            }

            let Some(query_idx) = fields[0]
                .strip_prefix("cds:")
                .and_then(|s| s.parse::<usize>().ok())
            else {
                continue;
            };
            let Some(qlen) = fields[1].parse::<usize>().ok() else {
                continue;
            };
            let Some(subject_idx) = fields[2]
                .strip_prefix("cds:")
                .and_then(|s| s.parse::<usize>().ok())
            else {
                continue;
            };
            let Some(score) = fields[3].parse::<f64>().ok() else {
                continue;
            };

            if query_idx == subject_idx {
                self_scores.insert(query_idx, score);
            } else {
                raw_hits.push((qlen, line_no, query_idx, subject_idx, score));
            }
        }
    }

    let _ = fs::remove_dir_all(&temp_dir);

    raw_hits.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut excluded: FxHashSet<usize> = FxHashSet::default();
    for (_qlen, _line_no, query_idx, subject_idx, score) in raw_hits {
        let Some(self_score) = self_scores.get(&query_idx).copied() else {
            continue;
        };
        if self_score <= 0.0 || excluded.contains(&query_idx) {
            continue;
        }
        let bsr = score / self_score;
        if bsr >= bsr_threshold + 0.1 {
            excluded.insert(subject_idx);
        }
    }

    let mut selected: Vec<_> = ordered_indices
        .into_iter()
        .filter(|idx| !excluded.contains(idx))
        .collect();
    selected.sort_unstable();
    Some(selected)
}

// =============================================================================
// Compatible mode: full BLAST pipeline replicating chewBBACA's exact flow
// =============================================================================

fn run_blastp(
    blastp: &Path,
    query: &Path,
    subject: &Path,
    output: &Path,
    num_threads: usize,
    outfmt: &str,
    extra_args: &[&str],
) -> bool {
    let mut cmd = Command::new(blastp);
    cmd.args([
        "-query", query.to_str().unwrap(),
        "-subject", subject.to_str().unwrap(),
        "-out", output.to_str().unwrap(),
        "-outfmt", outfmt,
        "-max_hsps", "1",
        "-num_threads", &num_threads.to_string(),
        "-evalue", "0.001",
        "-comp_based_stats", "0",
    ]);
    for arg in extra_args {
        cmd.arg(arg);
    }
    cmd.stdout(std::process::Stdio::null())
       .stderr(std::process::Stdio::null())
       .status()
       .map(|s| s.success())
       .unwrap_or(false)
}

/// Create a BLAST database from a FASTA file using makeblastdb.
fn make_blast_db(blastp: &Path, fasta: &Path, db_path: &Path, db_type: &str) -> bool {
    // makeblastdb is in the same directory as blastp
    let makeblastdb = blastp.with_file_name("makeblastdb");
    Command::new(&makeblastdb)
        .args([
            "-in", fasta.to_str().unwrap(),
            "-out", db_path.to_str().unwrap(),
            "-dbtype", db_type,
            "-parse_seqids",  // Required for -seqidlist to work
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Run BLASTp using -db mode (matching Python chewBBACA's approach).
fn run_blastp_db(
    blastp: &Path,
    query: &Path,
    db: &Path,
    output: &Path,
    num_threads: usize,
    outfmt: &str,
) -> bool {
    run_blastp_db_ext(blastp, query, db, output, num_threads, outfmt, &[], "0.001")
}

fn run_blastp_db_ext(
    blastp: &Path,
    query: &Path,
    db: &Path,
    output: &Path,
    num_threads: usize,
    outfmt: &str,
    extra_args: &[&str],
    evalue: &str,
) -> bool {
    let mut cmd = Command::new(blastp);
    cmd.args([
        "-query", query.to_str().unwrap(),
        "-db", db.to_str().unwrap(),
        "-out", output.to_str().unwrap(),
        "-outfmt", outfmt,
        "-max_hsps", "1",
        "-num_threads", &num_threads.to_string(),
        "-evalue", evalue,
        "-comp_based_stats", "0",
    ]);
    cmd.args(extra_args);
    cmd.stdout(std::process::Stdio::null())
       .stderr(std::process::Stdio::null())
       .status()
       .map(|s| s.success())
       .unwrap_or(false)
}

/// Compute BLAST self-scores for representative proteins.
/// Replicates chewBBACA's determine_self_scores(): each sequence BLASTed against itself.
/// Returns: rep_index → BLAST raw self-score.
pub fn blast_self_scores(
    representatives: &[Representative],
    blastp_path: &str,
    num_threads: usize,
) -> FxHashMap<usize, f64> {
    let blastp = Path::new(blastp_path);
    let temp_dir = unique_temp_dir("chewcall_self_scores");
    if fs::create_dir_all(&temp_dir).is_err() {
        eprintln!("  WARNING: cannot create temp dir for BLAST self-scores, using parasail");
        return parasail_self_scores(representatives);
    }

    // Write all rep proteins to one FASTA
    let fasta_path = temp_dir.join("reps.faa");
    let output_path = temp_dir.join("self_blast.tsv");
    {
        let records: Vec<_> = representatives.iter().enumerate()
            .map(|(i, rep)| (format!("rep:{i}"), rep.protein_seq.clone()))
            .collect();
        if write_fasta(&fasta_path, &records).is_err() {
            let _ = fs::remove_dir_all(&temp_dir);
            return parasail_self_scores(representatives);
        }
    }

    // BLAST each rep against itself using -db mode (matching Python chewBBACA)
    let db_path = temp_dir.join("reps_db");
    let blast_ok = if make_blast_db(blastp, &fasta_path, &db_path, "prot") {
        run_blastp_db(blastp, &fasta_path, &db_path, &output_path, num_threads,
                      "6 qseqid qstart qend qlen sseqid slen score")
    } else {
        // Fallback to -subject mode
        run_blastp(blastp, &fasta_path, &fasta_path, &output_path, num_threads,
                   "6 qseqid qstart qend qlen sseqid slen score", &[])
    };
    if !blast_ok {
        let _ = fs::remove_dir_all(&temp_dir);
        eprintln!("  WARNING: BLAST self-score failed, using parasail");
        return parasail_self_scores(representatives);
    }

    let mut scores: FxHashMap<usize, f64> = FxHashMap::default();
    if let Ok(file) = fs::File::open(&output_path) {
        let reader = BufReader::new(file);
        for line in reader.lines().map_while(Result::ok) {
            let fields: Vec<_> = line.split('\t').collect();
            if fields.len() < 7 { continue; }
            // Self-hit: qseqid == sseqid
            if fields[0] != fields[4] { continue; }
            let Some(rep_idx) = fields[0].strip_prefix("rep:")
                .and_then(|s| s.parse::<usize>().ok()) else { continue };
            let Some(score) = fields[6].parse::<f64>().ok() else { continue };
            if score > 0.0 {
                scores.insert(rep_idx, score);
            }
        }
    }

    let _ = fs::remove_dir_all(&temp_dir);
    scores
}

/// Fallback: compute self-scores via parasail if BLAST is unavailable.
fn parasail_self_scores(representatives: &[Representative]) -> FxHashMap<usize, f64> {
    let mut scores = FxHashMap::default();
    for (i, rep) in representatives.iter().enumerate() {
        scores.insert(i, sw::self_score(&rep.protein_seq) as f64);
    }
    scores
}

/// Phase 4 BLAST: representatives (query) vs unclassified proteins (subject).
/// Replicates chewBBACA's blast_clusters() + select_highest_scores() + process_blast_results().
///
/// - Representatives are QUERY (like chewBBACA)
/// - Unclassified proteins are SUBJECT
/// - BSR = score / rep_self_score
/// - select_highest_scores: sort by score DESC, keep first hit per target (subject)
/// - Returns one ClusterResult per matched protein (best rep per protein)
pub fn blast_phase4(
    proteins: &[(usize, Vec<u8>)],  // (cds_idx, protein)
    representatives: &[Representative],
    self_scores: &FxHashMap<usize, f64>,
    bsr_threshold: f64,
    blastp_path: &str,
    num_threads: usize,
) -> Vec<ClusterResult> {
    blast_phase4_impl(proteins, representatives, self_scores, bsr_threshold, blastp_path, num_threads, true)
}

/// Like blast_phase4 but returns ALL hits with BSR computed (no dedup, no BSR filter).
/// Used when caller will apply cluster filter → dedup → BSR filter.
pub fn blast_phase4_raw(
    proteins: &[(usize, Vec<u8>)],
    representatives: &[Representative],
    self_scores: &FxHashMap<usize, f64>,
    _bsr_threshold: f64,
    blastp_path: &str,
    num_threads: usize,
) -> Vec<ClusterResult> {
    // No dedup, no BSR filter (threshold=0.0) — caller handles everything
    blast_phase4_impl(proteins, representatives, self_scores, 0.0, blastp_path, num_threads, false)
}

fn blast_phase4_impl(
    proteins: &[(usize, Vec<u8>)],
    representatives: &[Representative],
    self_scores: &FxHashMap<usize, f64>,
    bsr_threshold: f64,
    blastp_path: &str,
    num_threads: usize,
    dedup_per_locus: bool,
) -> Vec<ClusterResult> {
    if proteins.is_empty() || representatives.is_empty() {
        return Vec::new();
    }

    let blastp = Path::new(blastp_path);
    let temp_dir = unique_temp_dir("chewcall_phase4");
    if fs::create_dir_all(&temp_dir).is_err() {
        eprintln!("  WARNING: cannot create temp dir for Phase 4 BLAST");
        return Vec::new();
    }

    // Query: representatives (matching Python: rep is query)
    let query_path = temp_dir.join("query.faa");
    let query_records: Vec<_> = representatives.iter().enumerate()
        .map(|(i, rep)| (format!("rep:{i}"), rep.protein_seq.clone()))
        .collect();

    // Subject: unclassified proteins (will become a BLAST DB)
    let subject_path = temp_dir.join("subject.faa");
    let subject_records: Vec<_> = proteins.iter()
        .map(|(cds_idx, prot)| (format!("cds:{cds_idx}"), prot.clone()))
        .collect();

    let output_path = temp_dir.join("blast.tsv");
    let db_path = temp_dir.join("subject_db");

    if write_fasta(&query_path, &query_records).is_err()
        || write_fasta(&subject_path, &subject_records).is_err()
    {
        let _ = fs::remove_dir_all(&temp_dir);
        return Vec::new();
    }

    // Use -db mode (matching Python chewBBACA) for correct E-value computation.
    // Python creates a makeblastdb from CDS proteins and queries with representatives.
    let blast_ok = if make_blast_db(blastp, &subject_path, &db_path, "prot") {
        run_blastp_db(blastp, &query_path, &db_path, &output_path, num_threads,
                      "6 qseqid qstart qend qlen sseqid slen score")
    } else {
        eprintln!("  WARNING: makeblastdb failed, falling back to -subject mode");
        run_blastp(blastp, &query_path, &subject_path, &output_path, num_threads,
                   "6 qseqid qstart qend qlen sseqid slen score", &[])
    };

    if !blast_ok {
        let _ = fs::remove_dir_all(&temp_dir);
        eprintln!("  WARNING: Phase 4 BLAST failed");
        return Vec::new();
    }

    eprintln!("  BLAST Phase4: {} query reps, {} subject proteins", query_records.len(), subject_records.len());

    // Parse BLAST results
    let results = parse_blast_results(&output_path, representatives, self_scores, bsr_threshold, dedup_per_locus);

    // Keep temp dir if CHEWCALL_DEBUG_BLAST is set
    if std::env::var("CHEWCALL_DEBUG_BLAST").is_ok() {
        eprintln!("  DEBUG: BLAST files kept at {}", temp_dir.display());
    } else {
        let _ = fs::remove_dir_all(&temp_dir);
    }
    results
}

/// RepDet BLAST: current representatives (query) vs remaining unclassified (subject).
/// Same flow as blast_phase4 but may use different threshold.
pub fn blast_repdet(
    proteins: &[(usize, Vec<u8>)],
    representatives: &[Representative],
    self_scores: &FxHashMap<usize, f64>,
    bsr_threshold: f64,
    blastp_path: &str,
    num_threads: usize,
) -> Vec<ClusterResult> {
    // Same as Phase 4 but with different threshold
    blast_phase4(proteins, representatives, self_scores, bsr_threshold, blastp_path, num_threads)
}

/// Parse BLAST output and apply select_highest_scores logic PER LOCUS.
///
/// chewBBACA processes BLAST results per-locus: each locus gets its own
/// `select_highest_scores` call, so the same CDS can match multiple loci.
/// Within each locus, keep the highest-scoring hit per target (subject).
/// BSR = score / rep_self_score (query self-score).
fn parse_blast_results(
    output_path: &Path,
    representatives: &[Representative],
    self_scores: &FxHashMap<usize, f64>,
    bsr_threshold: f64,
    dedup_per_locus: bool,
) -> Vec<ClusterResult> {
    // Format: qseqid qstart qend qlen sseqid slen score
    let mut raw_hits: Vec<(i32, usize, usize, u32, u32, u32, u32)> = Vec::new();
    // (score, rep_idx, cds_idx, qstart, qend, qlen, slen)

    if let Ok(file) = fs::File::open(output_path) {
        let reader = BufReader::new(file);
        for line in reader.lines().map_while(Result::ok) {
            let fields: Vec<_> = line.split('\t').collect();
            if fields.len() < 7 { continue; }

            let Some(rep_idx) = fields[0].strip_prefix("rep:")
                .and_then(|s| s.parse::<usize>().ok()) else { continue };
            let Some(cds_idx) = fields[4].strip_prefix("cds:")
                .and_then(|s| s.parse::<usize>().ok()) else { continue };
            let Some(qstart) = fields[1].parse::<u32>().ok() else { continue };
            let Some(qend) = fields[2].parse::<u32>().ok() else { continue };
            let Some(qlen) = fields[3].parse::<u32>().ok() else { continue };
            let Some(slen) = fields[5].parse::<u32>().ok() else { continue };
            let Some(score) = fields[6].parse::<i32>().ok() else { continue };

            raw_hits.push((score, rep_idx, cds_idx, qstart, qend, qlen, slen));
        }
    }

    // Replicate Python chewBBACA's select_highest_scores sort behavior.
    // Python sorts by int(x[5]) = slen (subject length) DESCENDING, NOT by score.
    // This is a bug in Python (comment says "decreasing raw score" but lambda uses x[5]=slen
    // instead of x[6]=score). When slen ties (very common — same CDS hit by multiple reps),
    // Python's stable sort preserves file order (BLAST output order ≈ rep_idx order).
    raw_hits.sort_by(|a, b| {
        // tuple: (score, rep_idx, cds_idx, qstart, qend, qlen, slen)
        b.6.cmp(&a.6)           // slen descending (Python's int(x[5]) sort)
            .then_with(|| a.1.cmp(&b.1))  // rep_idx ascending (FASTA order)
    });

    // When dedup_per_locus is true: keep first hit per (locus, cds_idx) and apply BSR.
    // When false: return ALL hits that pass BSR (dedup done by caller after cluster filter).
    let mut seen_per_locus: FxHashMap<u32, FxHashSet<usize>> = FxHashMap::default();
    let mut results = Vec::new();

    for (score, rep_idx, cds_idx, qstart, qend, qlen, slen) in raw_hits {
        if dedup_per_locus {
            let locus_idx = representatives[rep_idx].locus_idx;
            let seen = seen_per_locus.entry(locus_idx).or_default();
            if !seen.insert(cds_idx) {
                continue; // Already have a hit for this target in this locus
            }
        }

        let rep_self = self_scores.get(&rep_idx)
            .copied()
            .unwrap_or(representatives[rep_idx].self_score);
        if rep_self <= 0.0 { continue; }
        let bsr = score as f64 / rep_self;
        if bsr < bsr_threshold { continue; }

        results.push(ClusterResult {
            cds_idx,
            representative_idx: rep_idx,
            best_locus: representatives[rep_idx].locus_idx,
            best_bsr: bsr,
            score,
            rep_dna_len: representatives[rep_idx].dna_length,
            query_start: 0,
            query_end: 0,
            query_len: slen, // subject length (the CDS protein)
            // target positions = query positions from BLAST (rep is query)
            target_start: qstart,
            target_end: qend,
            target_len: qlen, // query length (the representative protein)
        });
    }

    results
}

// =============================================================================
// Per-cluster BLAST: replicate Python chewBBACA's exact per-cluster BLAST flow
// =============================================================================

/// Convert a text seqidlist to binary format using blastdb_aliastool.
fn run_blastdb_aliastool(blastp: &Path, text_file: &Path, binary_file: &Path) -> bool {
    let tool = blastp.with_file_name("blastdb_aliastool");
    Command::new(&tool)
        .args([
            "-seqid_file_in", text_file.to_str().unwrap(),
            "-seqid_file_out", binary_file.to_str().unwrap(),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Per-cluster BLAST replicating Python chewBBACA's exact flow.
///
/// Python flow:
/// 1. Build BLAST DB from all CDS proteins
/// 2. For each cluster (rep + its CDS): write seqidlist, run blastp with -seqidlist
/// 3. Concatenate per-cluster results per locus
/// 4. select_highest_scores: sort by slen DESC, keep first per target CDS
/// 5. process_blast_results: BSR filter
pub fn blast_per_cluster(
    proteins: &[(usize, Vec<u8>)],
    representatives: &[Representative],
    self_scores: &FxHashMap<usize, f64>,
    clusters: &FxHashMap<usize, Vec<usize>>,  // rep_idx → [cds_idx, ...]
    bsr_threshold: f64,
    blastp_path: &str,
    num_threads: usize,
) -> Vec<ClusterResult> {
    if proteins.is_empty() || representatives.is_empty() || clusters.is_empty() {
        return Vec::new();
    }

    let blastp = Path::new(blastp_path);
    let temp_dir = unique_temp_dir("chewcall_percluster");
    if fs::create_dir_all(&temp_dir).is_err() {
        eprintln!("  WARNING: cannot create temp dir for per-cluster BLAST");
        return Vec::new();
    }

    // Build CDS ID set for fast lookup
    let cds_idx_set: FxHashSet<usize> = proteins.iter().map(|(idx, _)| *idx).collect();

    // Write all CDS proteins to FASTA and build BLAST DB
    let subject_path = temp_dir.join("subject.faa");
    let db_path = temp_dir.join("subject_db");
    {
        let subject_records: Vec<_> = proteins.iter()
            .map(|(cds_idx, prot)| (format!("cds:{cds_idx}"), prot.clone()))
            .collect();
        if write_fasta(&subject_path, &subject_records).is_err() {
            let _ = fs::remove_dir_all(&temp_dir);
            return Vec::new();
        }
    }
    if !make_blast_db(blastp, &subject_path, &db_path, "prot") {
        eprintln!("  WARNING: makeblastdb failed for per-cluster BLAST");
        let _ = fs::remove_dir_all(&temp_dir);
        return Vec::new();
    }

    // Prepare per-cluster BLAST tasks, sorted by rep_idx for deterministic ordering
    let mut cluster_list: Vec<(usize, &Vec<usize>)> = clusters.iter()
        .map(|(k, v)| (*k, v))
        .collect();
    cluster_list.sort_unstable_by_key(|(rep_idx, _)| *rep_idx);

    let cluster_dir = temp_dir.join("clusters");
    let _ = fs::create_dir_all(&cluster_dir);

    // Run per-cluster BLAST with parallelism matching Python's approach.
    // Python runs cpu_cores BLAST processes in parallel.
    let num_clusters = cluster_list.len();

    // Pre-create all per-cluster files, then run BLAST in parallel
    struct ClusterTask {
        rep_idx: usize,
        query_path: PathBuf,
        ids_bin_path: PathBuf,
        output_path: PathBuf,
    }

    let mut tasks: Vec<ClusterTask> = Vec::with_capacity(num_clusters);
    for (cluster_no, (rep_idx, cds_indices)) in cluster_list.iter().enumerate() {
        let valid_cds: Vec<usize> = cds_indices.iter()
            .filter(|idx| cds_idx_set.contains(idx))
            .copied()
            .collect();
        if valid_cds.is_empty() {
            continue;
        }

        let cdir = cluster_dir.join(format!("{cluster_no}"));
        if fs::create_dir_all(&cdir).is_err() { continue; }

        // Write single-rep query FASTA
        let query_path = cdir.join("query.faa");
        let rep = &representatives[*rep_idx];
        if write_fasta(&query_path, &[(format!("rep:{rep_idx}"), rep.protein_seq.clone())]).is_err() {
            continue;
        }

        // Write seqidlist (text)
        let ids_text_path = cdir.join("ids.txt");
        {
            let Ok(mut f) = fs::File::create(&ids_text_path) else { continue };
            for &cds_idx in &valid_cds {
                let _ = writeln!(f, "cds:{cds_idx}");
            }
        }

        // Convert to binary format
        let ids_bin_path = cdir.join("ids.txt.bin");
        if !run_blastdb_aliastool(blastp, &ids_text_path, &ids_bin_path) {
            continue;
        }

        let output_path = cdir.join("blastout.tsv");
        tasks.push(ClusterTask {
            rep_idx: *rep_idx,
            query_path,
            ids_bin_path,
            output_path,
        });
    }

    // Run BLAST tasks in parallel using thread pool
    use std::sync::atomic::{AtomicUsize, Ordering};
    let completed = AtomicUsize::new(0);
    let total = tasks.len();

    // Use rayon for parallel execution
    use rayon::prelude::*;
    tasks.par_iter().for_each(|task| {
        run_blastp_db_ext(
            blastp, &task.query_path, &db_path, &task.output_path, 1,
            "6 qseqid qstart qend qlen sseqid slen score",
            &["-seqidlist", task.ids_bin_path.to_str().unwrap()],
            "0.001",
        );
        let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
        if done % 1000 == 0 || done == total {
            eprint!("\r  Per-cluster BLAST: {done}/{total}  ");
        }
    });
    eprintln!();

    // Collect results: group by locus and concatenate (preserving rep_idx order)
    // This matches Python's merge_blast_results
    let mut locus_hits: FxHashMap<u32, Vec<(i32, usize, usize, u32, u32, u32, u32)>> =
        FxHashMap::default();

    for task in &tasks {
        let locus_idx = representatives[task.rep_idx].locus_idx;
        if let Ok(file) = fs::File::open(&task.output_path) {
            let reader = BufReader::new(file);
            for line in reader.lines().map_while(Result::ok) {
                let fields: Vec<_> = line.split('\t').collect();
                if fields.len() < 7 { continue; }

                let Some(r_idx) = fields[0].strip_prefix("rep:")
                    .and_then(|s| s.parse::<usize>().ok()) else { continue };
                let Some(cds_idx) = fields[4].strip_prefix("cds:")
                    .and_then(|s| s.parse::<usize>().ok()) else { continue };
                let Some(qstart) = fields[1].parse::<u32>().ok() else { continue };
                let Some(qend) = fields[2].parse::<u32>().ok() else { continue };
                let Some(qlen) = fields[3].parse::<u32>().ok() else { continue };
                let Some(slen) = fields[5].parse::<u32>().ok() else { continue };
                let Some(score) = fields[6].parse::<i32>().ok() else { continue };

                locus_hits.entry(locus_idx).or_default().push(
                    (score, r_idx, cds_idx, qstart, qend, qlen, slen)
                );
            }
        }
    }

    // Per-locus: select_highest_scores (sort by slen DESC, keep first per target CDS)
    let mut results = Vec::new();
    for (_locus_idx, hits) in &mut locus_hits {
        // Sort by slen DESC. For ties, rep_idx ASC (approximates Python's concatenation order
        // since we process clusters in rep_idx order).
        hits.sort_by(|a, b| {
            b.6.cmp(&a.6)
                .then(a.1.cmp(&b.1))
        });

        // Keep first hit per target CDS
        let mut seen_targets: FxHashSet<usize> = FxHashSet::default();
        for &(score, rep_idx, cds_idx, qstart, qend, qlen, slen) in hits.iter() {
            if !seen_targets.insert(cds_idx) {
                continue;
            }

            let rep_self = self_scores.get(&rep_idx)
                .copied()
                .unwrap_or(representatives[rep_idx].self_score);
            if rep_self <= 0.0 { continue; }
            let bsr = score as f64 / rep_self;
            if bsr < bsr_threshold { continue; }

            results.push(ClusterResult {
                cds_idx,
                representative_idx: rep_idx,
                best_locus: representatives[rep_idx].locus_idx,
                best_bsr: bsr,
                score,
                rep_dna_len: representatives[rep_idx].dna_length,
                query_start: 0,
                query_end: 0,
                query_len: slen,
                target_start: qstart,
                target_end: qend,
                target_len: qlen,
            });
        }
    }

    // Cleanup
    if std::env::var("CHEWCALL_DEBUG_BLAST").is_ok() {
        eprintln!("  DEBUG: per-cluster BLAST files kept at {}", temp_dir.display());
    } else {
        let _ = fs::remove_dir_all(&temp_dir);
    }

    eprintln!("  Per-cluster BLAST results: {} hits across {} loci",
        results.len(), locus_hits.len());

    results
}
