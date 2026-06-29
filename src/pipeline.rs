//! Pipeline orchestration: wires all phases together.

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::path::Path;

use crate::blast;
use crate::cds;
use crate::classify;
use crate::cluster;
use crate::dedup;
use crate::output;
use crate::prodigal_rs;
use crate::schema;
use crate::sw;
use crate::translate;
use crate::types::*;

#[derive(Clone, Copy)]
struct BorderlineCandidate {
    cds_idx: usize,
    bsr: f64,
}

fn merge_result(
    result: &mut LocusResult,
    incoming_class: Classification,
    incoming_allele_id: Option<AlleleId>,
    incoming_is_novel: bool,
    incoming_dna_hash: Option<SeqHash>,
) {
    let previous_class = result.class;
    let previous_allele_id = result.allele_id;
    let previous_is_novel = result.is_novel;
    let previous_dna_hash = result.dna_hash;

    let merged = match result.class {
        Classification::LNF => incoming_class,
        Classification::NIPH => Classification::NIPH,
        Classification::NIPHEM => {
            if incoming_class == Classification::EXC {
                Classification::NIPHEM
            } else {
                Classification::NIPH
            }
        }
        current => classify::resolve_multi_match(&[current, incoming_class]),
    };

    result.class = merged;
    match merged {
        Classification::EXC | Classification::INF => {
            if merged == previous_class && previous_allele_id.is_some() {
                result.allele_id = previous_allele_id;
                result.is_novel = previous_is_novel;
                result.dna_hash = previous_dna_hash;
            } else {
                result.allele_id = incoming_allele_id;
                result.is_novel = incoming_is_novel;
                result.dna_hash = incoming_dna_hash;
            }
        }
        _ => {
            result.allele_id = None;
            result.is_novel = false;
            result.dna_hash = None;
        }
    }
}

fn mark_paralogous_matches(all_results: &mut [Vec<LocusResult>]) {
    for genome_results in all_results {
        let mut loci_by_hash: FxHashMap<SeqHash, Vec<usize>> = FxHashMap::default();
        for (locus_idx, result) in genome_results.iter().enumerate() {
            if result.class.is_valid() {
                if let Some(hash) = result.dna_hash {
                    loci_by_hash.entry(hash).or_default().push(locus_idx);
                }
            }
        }

        for loci in loci_by_hash.values() {
            if loci.len() <= 1 {
                continue;
            }
            for &locus_idx in loci {
                let result = &mut genome_results[locus_idx];
                result.class = Classification::PAMA;
                result.allele_id = None;
                result.is_novel = false;
                result.dna_hash = None;
            }
        }
    }
}

fn register_novel_allele(
    schema: &mut schema::Schema,
    next_allele_id: &mut [AlleleId],
    novel_alleles: &mut Vec<(String, Vec<u8>)>,
    loci_list: &[String],
    locus_idx: usize,
    cds_item: &Cds,
    dna_hash: SeqHash,
) -> AlleleId {
    let inf_id = next_allele_id[locus_idx];
    next_allele_id[locus_idx] += 1;

    novel_alleles.push((
        format!("{}_{}", loci_list[locus_idx], cds_item.id),
        cds_item.dna_seq.clone(),
    ));

    let seq_str = String::from_utf8_lossy(&cds_item.dna_seq);
    schema.allele_crc32.insert(
        (locus_idx as u32, inf_id),
        crc32fast::hash(seq_str.as_bytes()),
    );
    schema
        .dna_hashes
        .entry(dna_hash)
        .or_default()
        .push((locus_idx as u32, inf_id));
    // Mark this runtime-inferred allele so the output writer prefixes it with '*'.
    // chewBBACA prefixes runtime-inferred allele IDs with '*' too; without this, EXC
    // matches against a runtime-inferred ID write a plain integer, which downstream
    // tools (and the discordance comparator) misclassify as a schema-original EXC
    // versus chewBBACA's INF '*N'. The classification stays EXC, so paralog
    // detection (multi-EXC → NIPHEM) is unaffected.
    schema
        .inferred_allele_ids
        .insert((locus_idx as u32, inf_id));

    inf_id
}

fn group_hits_by_locus(
    hits: Vec<cluster::ClusterResult>,
) -> FxHashMap<usize, Vec<cluster::ClusterResult>> {
    let mut by_locus: FxHashMap<usize, Vec<cluster::ClusterResult>> = FxHashMap::default();
    for hit in hits {
        by_locus
            .entry(hit.best_locus as usize)
            .or_default()
            .push(hit);
    }
    for locus_hits in by_locus.values_mut() {
        locus_hits.sort_unstable_by(|a, b| b.score.cmp(&a.score).then(a.cds_idx.cmp(&b.cds_idx)));
    }
    by_locus
}

fn process_locus_hits(
    locus_idx: usize,
    hits: &[cluster::ClusterResult],
    bsr_threshold: f64,
    candidate_upper_bsr: Option<f64>,
    only_fill_lnf: bool,
    unmatched_cds: &[&Cds],
    protein_hashes_by_idx: &FxHashMap<usize, SeqHash>,
    cds_indices_by_protein_hash: &FxHashMap<SeqHash, Vec<usize>>,
    hash_to_genomes: &FxHashMap<SeqHash, Vec<dedup::CdsGenomeEntry>>,
    all_results: &mut [Vec<LocusResult>],
    next_allele_id: &mut [AlleleId],
    novel_alleles: &mut Vec<(String, Vec<u8>)>,
    schema: &mut schema::Schema,
    loci_list: &[String],
    config: &Config,
) -> (FxHashSet<usize>, Vec<usize>) {
    let mut seen_dna: FxHashMap<SeqHash, AlleleId> = FxHashMap::default();
    let mut seen_prot: FxHashSet<SeqHash> = FxHashSet::default();
    let mut excluded_cds: FxHashSet<usize> = FxHashSet::default();
    let mut first_inf_by_genome: FxHashMap<usize, BorderlineCandidate> = FxHashMap::default();
    for hit in hits.iter() {
        if hit.best_bsr < bsr_threshold {
            continue;
        }

        let Some(&protein_hash) = protein_hashes_by_idx.get(&hit.cds_idx) else {
            continue;
        };
        let related_cds_indices = cds_indices_by_protein_hash
            .get(&protein_hash)
            .map(Vec::as_slice)
            .unwrap_or(std::slice::from_ref(&hit.cds_idx));

        for &related_cds_idx in related_cds_indices {
            let cds_item = unmatched_cds[related_cds_idx];
            let dna_upper: Vec<u8> = cds_item
                .dna_seq
                .iter()
                .map(|b| b.to_ascii_uppercase())
                .collect();
            let dna_hash = schema::sha256(&dna_upper);

            // Compute base class WITHOUT position classification —
            // PLOT3/PLOT5/LOTSC depend on per-genome contig coordinates
            // and will be checked per-genome below.
            let base_class = if seen_dna.contains_key(&dna_hash) {
                Classification::EXC
            } else if seen_prot.contains(&protein_hash) {
                Classification::INF
            } else {
                classify::classify_inexact(
                    hit.best_bsr,
                    config.bsr_threshold,
                    cds_item.dna_seq.len() as u32,
                    schema.loci[locus_idx].mode_length,
                    config.size_threshold,
                    None, // coord=None: skip position classification here
                    hit.rep_dna_len,
                    hit.target_start,
                    hit.target_end,
                    hit.target_len,
                    config.cds_input,
                )
            };

            if base_class == Classification::LNF {
                continue;
            }

            let Some(genomes) = hash_to_genomes.get(&dna_hash) else {
                continue;
            };
            let mut sorted_genomes: Vec<_> = genomes.iter().collect();
            sorted_genomes.sort_by_key(|(gi, _)| *gi);

            let mut allele_id = seen_dna.get(&dna_hash).copied();

            for (genome_pos, (genome_idx, genome_coord)) in sorted_genomes.iter().enumerate() {
                let genome_idx = *genome_idx;
                // Apply per-genome position classification (PLOT3/PLOT5/LOTSC)
                let class = if !config.cds_input
                    && matches!(
                        base_class,
                        Classification::INF | Classification::ASM | Classification::ALM
                    ) {
                    // Only INF/ASM/ALM can be reclassified to PLOT — EXC stays EXC
                    if let Some(coord) = genome_coord.as_ref() {
                        classify::position_classification_pub(
                            coord,
                            hit.rep_dna_len,
                            hit.target_start,
                            hit.target_end,
                            hit.target_len,
                        )
                        .unwrap_or(base_class)
                    } else {
                        base_class
                    }
                } else {
                    base_class
                };
                let gi = genome_idx as usize;
                if gi >= all_results.len() || locus_idx >= all_results[gi].len() {
                    continue;
                }
                let debug_li = std::env::var("DEBUG_LOCUS")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok());
                if debug_li == Some(locus_idx) && gi == 0 {
                    let prev = all_results[gi][locus_idx].class;
                    let cds_id = &cds_item.id;
                    let cds_len = cds_item.dna_seq.len();
                    eprintln!("  [DEBUG Phase4/5] genome={} locus={}: {} + {:?} (bsr={:.3}, cds_idx={}, related_cds_idx={}, cds_id={}, len={}, hash0={:02x}{:02x})",
                        gi, locus_idx, prev, class, hit.best_bsr, hit.cds_idx, related_cds_idx, cds_id, cds_len, dna_hash[0], dna_hash[1]);
                }
                if only_fill_lnf && all_results[gi][locus_idx].class != Classification::LNF {
                    continue;
                }
                if all_results[gi][locus_idx].class == Classification::EXC
                    && matches!(
                        class,
                        Classification::PLOT3 | Classification::PLOT5 | Classification::LOTSC
                    )
                {
                    continue;
                }
                if candidate_upper_bsr.is_some()
                    && all_results[gi][locus_idx].class == Classification::EXC
                    && class != Classification::EXC
                    && hit.rep_dna_len <= 300
                    && hit.best_bsr < config.bsr_threshold + 0.02
                {
                    continue;
                }
                match class {
                    Classification::INF => {
                        let aid = *allele_id.get_or_insert_with(|| {
                            register_novel_allele(
                                schema,
                                next_allele_id,
                                novel_alleles,
                                loci_list,
                                locus_idx,
                                cds_item,
                                dna_hash,
                            )
                        });
                        let incoming_class = if genome_pos == 0 && !seen_dna.contains_key(&dna_hash)
                        {
                            Classification::INF
                        } else {
                            Classification::EXC
                        };
                        merge_result(
                            &mut all_results[gi][locus_idx],
                            incoming_class,
                            Some(aid),
                            true,
                            Some(dna_hash),
                        );
                        if incoming_class == Classification::INF {
                            first_inf_by_genome
                                .entry(gi)
                                .or_insert(BorderlineCandidate {
                                    cds_idx: related_cds_idx,
                                    bsr: hit.best_bsr,
                                });
                        }
                    }
                    Classification::EXC => {
                        let Some(aid) = allele_id else {
                            continue;
                        };
                        merge_result(
                            &mut all_results[gi][locus_idx],
                            Classification::EXC,
                            Some(aid),
                            true,
                            Some(dna_hash),
                        );
                    }
                    other => {
                        merge_result(&mut all_results[gi][locus_idx], other, None, false, None);
                    }
                }
            }

            if base_class == Classification::INF {
                if let Some(aid) = allele_id {
                    seen_dna.insert(dna_hash, aid);
                }
                seen_prot.insert(protein_hash);
            }
            if base_class != Classification::EXC {
                excluded_cds.insert(related_cds_idx);
            } else if seen_dna.contains_key(&dna_hash) {
                excluded_cds.insert(related_cds_idx);
            }
        }
    }

    let mut candidates = Vec::new();
    if let Some(upper_bsr) = candidate_upper_bsr {
        let mut candidate_set: FxHashSet<usize> = FxHashSet::default();
        for (genome_idx, info) in first_inf_by_genome {
            if all_results[genome_idx][locus_idx].class == Classification::INF
                && info.bsr >= bsr_threshold
                && info.bsr < upper_bsr
                && candidate_set.insert(info.cds_idx)
            {
                candidates.push(info.cds_idx);
            }
        }
        candidates.sort_unstable();
    }

    (excluded_cds, candidates)
}

fn select_new_representatives(
    candidate_cds_by_locus: &FxHashMap<usize, Vec<usize>>,
    unmatched_cds: &[&Cds],
    proteins_by_idx: &FxHashMap<usize, Vec<u8>>,
    bsr_threshold: f64,
    use_blast: bool,
) -> Vec<Representative> {
    use crate::parasail_ffi;

    let mut selected = Vec::new();
    let mut loci: Vec<_> = candidate_cds_by_locus.keys().copied().collect();
    loci.sort_unstable();

    for locus_idx in loci {
        let mut cds_indices = candidate_cds_by_locus[&locus_idx].clone();
        cds_indices.sort_unstable();
        cds_indices.dedup();

        if use_blast {
            if let Some(selected_indices) = blast::select_representative_candidates(
                &cds_indices,
                proteins_by_idx,
                bsr_threshold,
            ) {
                for cds_idx in selected_indices {
                    let Some(protein_seq) = proteins_by_idx.get(&cds_idx) else {
                        continue;
                    };
                    let cds_item = unmatched_cds[cds_idx];
                    selected.push(Representative {
                        locus_idx: locus_idx as u32,
                        seq_id: cds_item.id.clone(),
                        protein_seq: protein_seq.clone(),
                        dna_length: cds_item.dna_seq.len() as u32,
                        self_score: sw::self_score(protein_seq) as f64,
                    });
                }
                continue;
            }
        } // use_blast

        let mut self_scores: FxHashMap<usize, f64> = FxHashMap::default();
        for &cds_idx in &cds_indices {
            if let Some(protein) = proteins_by_idx.get(&cds_idx) {
                self_scores.insert(cds_idx, sw::self_score(protein) as f64);
            }
        }

        let mut ordered = cds_indices;
        ordered.sort_unstable_by(|a, b| {
            let len_a = proteins_by_idx.get(a).map(|p| p.len()).unwrap_or(0);
            let len_b = proteins_by_idx.get(b).map(|p| p.len()).unwrap_or(0);
            len_a.cmp(&len_b).then(a.cmp(b))
        });

        let mut excluded: FxHashSet<usize> = FxHashSet::default();
        for &query_idx in &ordered {
            if excluded.contains(&query_idx) {
                continue;
            }
            let Some(query) = proteins_by_idx.get(&query_idx) else {
                continue;
            };
            let Some(&self_score) = self_scores.get(&query_idx) else {
                continue;
            };
            if self_score <= 0.0 {
                continue;
            }

            for &target_idx in &ordered {
                if query_idx == target_idx {
                    continue;
                }
                let Some(target) = proteins_by_idx.get(&target_idx) else {
                    continue;
                };
                let (score, _, _) = parasail_ffi::sw_simd(query, target);
                let bsr = score as f64 / self_score;
                if bsr >= bsr_threshold + 0.1 {
                    excluded.insert(target_idx);
                }
            }
        }

        for cds_idx in ordered {
            if excluded.contains(&cds_idx) {
                continue;
            }
            let Some(protein_seq) = proteins_by_idx.get(&cds_idx) else {
                continue;
            };
            let Some(&self_score) = self_scores.get(&cds_idx) else {
                continue;
            };
            let cds_item = unmatched_cds[cds_idx];
            selected.push(Representative {
                locus_idx: locus_idx as u32,
                seq_id: cds_item.id.clone(),
                protein_seq: protein_seq.clone(),
                dna_length: cds_item.dna_seq.len() as u32,
                self_score,
            });
        }
    }

    selected.sort_unstable_by(|a, b| a.locus_idx.cmp(&b.locus_idx).then(a.seq_id.cmp(&b.seq_id)));
    selected
}

/// Run the full AlleleCall pipeline.
pub fn run_allele_call(
    genome_paths: &[String],
    schema_dir: &Path,
    output_dir: &Path,
    config: &Config,
    cds_input_dir: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(output_dir)?;

    // Start GPU init ASAP in background (1.3s CUDA context creation)
    let use_gpu = config.use_gpu;
    let gpu_handle = if use_gpu {
        Some(std::thread::spawn(
            || -> Result<crate::gpu_sw::GpuAligner, String> {
                crate::gpu_sw::GpuAligner::new().map_err(|e| e.to_string())
            },
        ))
    } else {
        None
    };

    let t0 = std::time::Instant::now();

    // --- Phase 0: Load schema ---
    eprintln!("[Phase 0] Loading schema...");
    let loci_list = discover_loci(schema_dir)?;
    let num_loci = loci_list.len();
    eprintln!("  Found {} loci", num_loci);

    let mut schema = schema::load_schema(schema_dir, &loci_list, config.translation_table);

    // Read schema config (chewBBACA .schema_config pickle)
    // and apply schema-specific parameters (override CLI defaults)
    let schema_config = schema::read_schema_config(schema_dir);
    let mut effective_config = config.clone();
    if let Some(bsr) = schema_config.bsr {
        if (config.bsr_threshold - 0.6).abs() < 1e-9 {
            // User didn't override BSR, use schema value
            effective_config.bsr_threshold = bsr;
            eprintln!("  Schema BSR threshold: {}", bsr);
        }
    }
    if let Some(st) = schema_config.size_threshold {
        if (config.size_threshold - 0.2).abs() < 1e-9 {
            // User didn't override size_threshold, use schema value
            effective_config.size_threshold = st;
            eprintln!("  Schema size threshold: {}", st);
        }
    }
    let config = &effective_config;

    // Find prodigal training file in schema directory
    let training_file = find_training_file(schema_dir);
    if let Some(ref trn) = training_file {
        eprintln!("  Using training file: {}", trn.display());
    }

    // Load Rust prodigal training data (used by default when training file exists)
    // If PRODIGAL_SUBPROCESS env var is set, force C prodigal subprocess
    let force_subprocess = std::env::var("PRODIGAL_SUBPROCESS").is_ok();
    let rs_training: Option<prodigal_rs::Training> = if force_subprocess || config.use_prodigal_ffi
    {
        if force_subprocess {
            eprintln!("  PRODIGAL_SUBPROCESS set: using C prodigal subprocess");
        }
        None
    } else {
        training_file
            .as_ref()
            .and_then(|trn| match prodigal_rs::Training::from_file(trn) {
                Ok(t) => {
                    eprintln!("  Using built-in Rust prodigal (no subprocess)");
                    Some(t)
                }
                Err(e) => {
                    eprintln!(
                        "  Warning: failed to load training for Rust prodigal: {}",
                        e
                    );
                    None
                }
            })
    };

    // Compute (or load cached) self-scores for representatives
    let cache_path = schema_dir.join("short").join("self_scores_rs.tsv");
    let cached = load_self_scores_cache(&cache_path, &schema.representatives);

    // In compatible mode, compute self-scores via BLAST (not parasail)
    let use_blast_scores = config.align_mode == crate::types::AlignMode::Compatible;

    if let Some(scores) = cached.filter(|_| !use_blast_scores) {
        eprintln!("[Phase 0] Loading cached self-scores...");
        for (i, rep) in schema.representatives.iter_mut().enumerate() {
            rep.self_score = scores[i];
        }
    } else if use_blast_scores {
        // Try loading from Python pickle first (authoritative), then BLAST cache, then recompute
        let blast_cache = schema_dir.join("short").join("self_scores_blast.tsv");
        let python_pickle = schema_dir.join("short").join("self_scores");
        if let Some(pickle_scores) =
            load_python_self_scores_pickle(&python_pickle, &schema.representatives)
        {
            eprintln!("[Phase 0] Loading self-scores from Python pickle...");
            for (i, rep) in schema.representatives.iter_mut().enumerate() {
                rep.self_score = pickle_scores[i];
            }
            save_self_scores_cache(&blast_cache, &schema.representatives);
        } else if let Some(scores) = load_self_scores_cache(&blast_cache, &schema.representatives) {
            eprintln!("[Phase 0] Loading cached BLAST self-scores...");
            for (i, rep) in schema.representatives.iter_mut().enumerate() {
                rep.self_score = scores[i];
            }
        } else {
            eprintln!("[Phase 0] Computing BLAST self-scores (compatible mode)...");
            let blast_scores = blast::blast_self_scores(
                &schema.representatives,
                &config.blastp_path,
                config.cpu_cores,
            );
            for (i, rep) in schema.representatives.iter_mut().enumerate() {
                if let Some(&score) = blast_scores.get(&i) {
                    rep.self_score = score;
                } else {
                    rep.self_score = sw::self_score(&rep.protein_seq) as f64;
                }
            }
            save_self_scores_cache(&blast_cache, &schema.representatives);
        }
    } else {
        eprintln!("[Phase 0] Computing representative self-scores...");
        for rep in &mut schema.representatives {
            let score = sw::self_score(&rep.protein_seq);
            rep.self_score = score as f64;
        }
        save_self_scores_cache(&cache_path, &schema.representatives);
    }
    for locus in &mut schema.loci {
        locus.self_score = 0.0;
    }
    for rep in &schema.representatives {
        let locus = &mut schema.loci[rep.locus_idx as usize];
        if rep.self_score > locus.self_score {
            locus.self_score = rep.self_score;
        }
    }

    let num_genomes = genome_paths.len();
    eprintln!(
        "[Phase 0] Processing {} genomes against {} loci",
        num_genomes, num_loci
    );

    // Initialize results: all LNF by default
    let mut all_results: Vec<Vec<LocusResult>> = (0..num_genomes)
        .map(|_| {
            (0..num_loci)
                .map(|_| LocusResult {
                    class: Classification::LNF,
                    allele_id: None,
                    is_novel: false,
                    dna_hash: None,
                    matches: Vec::new(),
                })
                .collect()
        })
        .collect();

    let mut novel_alleles: Vec<(String, Vec<u8>)> = Vec::new();
    let mut contigs_info: Vec<output::ContigInfo> = Vec::new();

    // Track next allele ID per locus (for INF assignments)
    let mut next_allele_id: Vec<AlleleId> =
        schema.loci.iter().map(|l| l.max_allele_id + 1).collect();

    // --- Phase 1: CDS prediction (parallel per genome) ---
    let t1 = std::time::Instant::now();
    eprintln!(
        "  [TIMING] Phase 0: {:.1}s",
        t1.duration_since(t0).as_secs_f64()
    );
    eprintln!("[Phase 1] CDS prediction...");
    // Per-genome CDS loader (used by the chunked, memory-bounded load below).
    let load_one = |genome_idx: usize| -> (Vec<Cds>, u32) {
        let path = &genome_paths[genome_idx];
        let genome_path = Path::new(path);
        let genome_stem = genome_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();

        // Check for pre-computed CDS
        if let Some(cds_dir) = cds_input_dir {
            // Strip .cds suffix if present (e.g., "Se_0001.cds" → "Se_0001")
            let stem_str = genome_stem.to_string();
            let base_stem = stem_str.strip_suffix(".cds").unwrap_or(&stem_str);
            let cds_file = cds_dir.join(format!("{}.cds.fasta", base_stem));
            if cds_file.exists() {
                return cds::load_precomputed_cds(&cds_file, genome_idx as GenomeIdx);
            }
            // Fallback: try genome file itself as CDS (when -i and --cds-input point to same dir)
            if let Some(fname) = genome_path.file_name() {
                let alt = cds_dir.join(fname);
                if alt.exists() {
                    return cds::load_precomputed_cds(&alt, genome_idx as GenomeIdx);
                }
            }
        }

        // Otherwise, run prodigal (Rust built-in, FFI, or subprocess)
        let (mut cds_list, invalid) = if let Some(ref trn) = rs_training {
            // Fastest: pure Rust prodigal (no subprocess, no FFI)
            cds::predict_cds_rs(genome_path, genome_idx as GenomeIdx, trn)
        } else if config.use_prodigal_ffi {
            if let Some(ref trn) = training_file {
                cds::predict_cds_ffi(
                    genome_path,
                    genome_idx as GenomeIdx,
                    config.translation_table,
                    trn,
                )
            } else {
                panic!("--prodigal-ffi requires a training file (.trn) in the schema directory");
            }
        } else {
            cds::predict_cds(
                genome_path,
                genome_idx as GenomeIdx,
                config.translation_table,
                &config.prodigal_mode,
                training_file.as_deref(),
                &config.prodigal_path,
            )
        };

        // Fill contig lengths
        let contig_lengths = cds::get_contig_lengths(genome_path);
        let contig_map: FxHashMap<&str, u32> = contig_lengths
            .iter()
            .map(|(name, len)| (name.as_str(), *len))
            .collect();

        for cds_item in &mut cds_list {
            if let Some(ref mut coord) = cds_item.coord {
                if let Some(&len) = contig_map.get(coord.contig.as_str()) {
                    coord.contig_len = len;
                }
            }
        }

        (cds_list, invalid)
    };

    // Chunked load fused with streaming deduplication. We hold the full DNA of
    // at most one chunk of genomes at a time and retain dna_seq only for the
    // first (distinct) occurrence of each sequence; duplicate occurrences keep
    // metadata only (dna_seq emptied). Distinct selection follows genome order,
    // identical to the previous all-at-once deduplication, so results are
    // unchanged while peak memory drops from O(all CDS) to O(one chunk +
    // distinct CDS). The per-sequence length needed by the contig-info scan is
    // cached in `hash_to_len`.
    const GENOME_CHUNK: usize = 64;
    let mut distinct_cds: Vec<Cds> = Vec::new();
    let mut hash_to_genomes: FxHashMap<SeqHash, Vec<dedup::CdsGenomeEntry>> = FxHashMap::default();
    let mut hash_to_len: FxHashMap<SeqHash, u32> = FxHashMap::default();
    let mut seen: FxHashSet<SeqHash> = FxHashSet::default();
    let mut total_cds = 0usize;
    let mut gstart = 0usize;
    while gstart < genome_paths.len() {
        let gend = (gstart + GENOME_CHUNK).min(genome_paths.len());
        // Load this chunk's genomes in parallel, then flatten.
        let loaded: Vec<(Vec<Cds>, u32)> = (gstart..gend).into_par_iter().map(&load_one).collect();
        let mut chunk_cds: Vec<Cds> = Vec::new();
        for (cds_list, _invalid) in loaded {
            chunk_cds.extend(cds_list);
        }
        // Hash the chunk in parallel (the expensive step; matches the previous
        // parallel deduplicate_cds), then merge sequentially (cheap).
        let chunk_hashes: Vec<SeqHash> = chunk_cds
            .par_iter()
            .map(|cds| {
                let upper: Vec<u8> = cds.dna_seq.iter().map(|b| b.to_ascii_uppercase()).collect();
                schema::sha256(&upper)
            })
            .collect();
        for (cds, hash) in chunk_cds.into_iter().zip(chunk_hashes) {
            total_cds += 1;
            hash_to_genomes
                .entry(hash)
                .or_default()
                .push((cds.genome_idx, cds.coord.clone()));
            if seen.insert(hash) {
                // first occurrence: keep the sequence as the distinct representative
                hash_to_len.insert(hash, cds.dna_seq.len() as u32);
                distinct_cds.push(cds);
            }
            // duplicate occurrences are dropped; their metadata is in hash_to_genomes
        }
        gstart = gend;
    }
    eprintln!("  Total CDS predicted: {}", total_cds);

    // --- Phase 2: Deduplication (streaming; completed during load) ---
    let t2 = std::time::Instant::now();
    eprintln!(
        "  [TIMING] Phase 1: {:.1}s",
        t2.duration_since(t1).as_secs_f64()
    );
    eprintln!("[Phase 2] Deduplicating CDS...");
    eprintln!(
        "  Distinct CDS: {} (from {})",
        distinct_cds.len(),
        total_cds
    );

    // Track classification per CDS hash → locus
    // We'll track per-genome, per-locus results through hash_to_genomes mapping.
    let mut cds_classifications: FxHashMap<SeqHash, (LocusIdx, Classification, Option<AlleleId>)> =
        FxHashMap::default();

    // --- Phase 3a: Exact DNA matching ---
    let t3a = std::time::Instant::now();
    eprintln!(
        "  [TIMING] Phase 2: {:.1}s",
        t3a.duration_since(t2).as_secs_f64()
    );
    eprintln!("[Phase 3a] Exact DNA matching...");
    let mut unmatched_cds: Vec<&Cds> = Vec::new();
    let mut dna_exact_count = 0u32;

    for cds_item in &distinct_cds {
        let upper: Vec<u8> = cds_item
            .dna_seq
            .iter()
            .map(|b| b.to_ascii_uppercase())
            .collect();
        let hash = schema::sha256(&upper);

        if let Some(matches) = schema.dna_hashes.get(&hash) {
            // Exact DNA match → EXC. Whether the allele was inferred (*-prefixed in schema
            // or runtime-inferred during this run) is encoded at write time via the
            // inferred_allele_ids set, not via the classification, so paralog detection
            // (multi-EXC → NIPHEM) keeps working.
            dna_exact_count += 1;
            for &(locus_idx, allele_id) in matches {
                cds_classifications.insert(hash, (locus_idx, Classification::EXC, Some(allele_id)));
                // Apply to all genomes with this hash
                if let Some(genomes) = hash_to_genomes.get(&hash) {
                    for (genome_idx, _) in genomes {
                        let gi = *genome_idx as usize;
                        let li = locus_idx as usize;
                        if gi < all_results.len() && li < num_loci {
                            let debug_li = std::env::var("DEBUG_LOCUS")
                                .ok()
                                .and_then(|s| s.parse::<usize>().ok());
                            if debug_li == Some(li) && gi == 0 {
                                let prev = all_results[gi][li].class;
                                eprintln!("  [DEBUG Phase3a] genome={} locus={}: {} + EXC(allele={}) hash={:?}",
                                    gi, li, prev, allele_id, &hash[..4]);
                            }
                            merge_result(
                                &mut all_results[gi][li],
                                Classification::EXC,
                                Some(allele_id),
                                false,
                                Some(hash),
                            );
                        }
                    }
                }
            }
        } else {
            unmatched_cds.push(cds_item);
        }
    }
    eprintln!("  DNA exact matches: {}", dna_exact_count);

    // --- Phase 3b: Translation ---
    eprintln!("[Phase 3b] Translating unmatched CDS...");
    let mut translated: Vec<(usize, Vec<u8>, SeqHash)> = Vec::new(); // (idx_in_unmatched, protein, dna_hash)

    for (idx, cds_item) in unmatched_cds.iter().enumerate() {
        let upper: Vec<u8> = cds_item
            .dna_seq
            .iter()
            .map(|b| b.to_ascii_uppercase())
            .collect();
        if let Some(protein) = translate::translate_cds(&upper, config.translation_table, true) {
            if protein.len() >= config.min_sequence_length as usize / 3 {
                let dna_hash = schema::sha256(&upper);
                translated.push((idx, protein, dna_hash));
            }
        }
    }
    eprintln!("  Translated proteins: {}", translated.len());
    let all_translated = translated.clone();

    // --- Phase 3c: Exact protein matching ---
    eprintln!("[Phase 3c] Exact protein matching...");
    let mut unmatched_proteins: Vec<(usize, Vec<u8>, SeqHash)> = Vec::new();
    let mut prot_exact_count = 0u32;

    for (idx, protein, dna_hash) in translated {
        let prot_hash = schema::sha256(&protein);

        if let Some(matches) = schema.protein_hashes.get(&prot_hash) {
            prot_exact_count += 1;
            if let Some(&(locus_idx, _allele_id)) = matches.first() {
                let li = locus_idx as usize;
                if let Some(genomes) = hash_to_genomes.get(&dna_hash) {
                    // Sort genomes to process in order (first gets INF, rest get EXC)
                    let mut sorted_genomes: Vec<_> = genomes.iter().collect();
                    sorted_genomes.sort_by_key(|(gi, _)| *gi);

                    let mut inf_allele_id: Option<AlleleId> = None;
                    for (genome_idx, _) in sorted_genomes.iter() {
                        let gi = *genome_idx as usize;
                        if gi < all_results.len() && li < num_loci {
                            let debug_li = std::env::var("DEBUG_LOCUS")
                                .ok()
                                .and_then(|s| s.parse::<usize>().ok());
                            if debug_li == Some(li) && gi == 0 {
                                let prev = all_results[gi][li].class;
                                eprintln!("  [DEBUG Phase3c] genome={} locus={}: {} + protein_match (inf_allele={:?})",
                                    gi, li, prev, inf_allele_id);
                            }
                            let aid = if let Some(aid) = inf_allele_id {
                                merge_result(
                                    &mut all_results[gi][li],
                                    Classification::EXC,
                                    Some(aid),
                                    true,
                                    Some(dna_hash),
                                );
                                aid
                            } else {
                                let aid = register_novel_allele(
                                    &mut schema,
                                    &mut next_allele_id,
                                    &mut novel_alleles,
                                    &loci_list,
                                    li,
                                    unmatched_cds[idx],
                                    dna_hash,
                                );
                                inf_allele_id = Some(aid);
                                merge_result(
                                    &mut all_results[gi][li],
                                    Classification::INF,
                                    Some(aid),
                                    true,
                                    Some(dna_hash),
                                );
                                aid
                            };
                            if !schema
                                .dna_hashes
                                .get(&dna_hash)
                                .map(|ids| ids.contains(&(locus_idx, aid)))
                                .unwrap_or(false)
                            {
                                schema
                                    .dna_hashes
                                    .entry(dna_hash)
                                    .or_default()
                                    .push((locus_idx, aid));
                            }
                        }
                    }
                }
            }
        } else {
            unmatched_proteins.push((idx, protein, dna_hash));
        }
    }
    eprintln!("  Protein exact matches: {}", prot_exact_count);

    // --- Phase 4: Clustering + alignment ---
    let t4 = std::time::Instant::now();
    eprintln!(
        "  [TIMING] Phases 3a-c: {:.1}s",
        t4.duration_since(t3a).as_secs_f64()
    );
    let mode_label = if config.use_gpu { " (GPU)" } else { "" };
    eprintln!("[Phase 4] Clustering + alignment{}...", mode_label);
    let k = config.minimizer_k;
    let w = config.minimizer_w;
    let min_shared = 1;

    let proteins_by_idx: FxHashMap<usize, Vec<u8>> = unmatched_proteins
        .iter()
        .map(|(idx, protein, _)| (*idx, protein.clone()))
        .collect();
    let protein_hashes_by_idx: FxHashMap<usize, SeqHash> = unmatched_proteins
        .iter()
        .map(|(idx, protein, _)| (*idx, schema::sha256(protein)))
        .collect();
    let mut cds_indices_by_protein_hash: FxHashMap<SeqHash, Vec<usize>> = FxHashMap::default();
    for (idx, _, _) in &unmatched_proteins {
        if let Some(&protein_hash) = protein_hashes_by_idx.get(idx) {
            cds_indices_by_protein_hash
                .entry(protein_hash)
                .or_default()
                .push(*idx);
        }
    }
    for cds_indices in cds_indices_by_protein_hash.values_mut() {
        cds_indices.sort_unstable();
        cds_indices.dedup();
    }
    let mut cluster_input: Vec<(usize, Vec<u8>)> = cds_indices_by_protein_hash
        .values()
        .filter_map(|cds_indices| {
            let &cds_idx = cds_indices.first()?;
            let protein = proteins_by_idx.get(&cds_idx)?.clone();
            Some((cds_idx, protein))
        })
        .collect();
    cluster_input.sort_unstable_by_key(|(cds_idx, _)| *cds_idx);

    // Wait for GPU init if applicable
    let _gpu_aligner = if let Some(handle) = gpu_handle {
        match handle.join().expect("GPU init thread panicked") {
            Ok(a) => {
                eprintln!("  GPU aligner initialized");
                Some(a)
            }
            Err(e) => {
                eprintln!("  WARNING: GPU init failed ({}), falling back to CPU", e);
                None
            }
        }
    } else {
        None
    };

    let compat_mode = config.align_mode == crate::types::AlignMode::Compatible;
    // Lexicographic minimizer ordering (compat) is used in BLAST-compatible mode
    // and whenever the user explicitly requests it (--minimizer-order lexicographic).
    let lex_order = compat_mode || config.lexicographic_minimizers;
    let index = if lex_order {
        cluster::build_minimizer_index_compat(&schema.representatives, k, w)
    } else {
        cluster::build_minimizer_index(&schema.representatives, k, w)
    };
    let phase4_similarity = config.minimizer_threshold;
    let phase4_threshold = config.bsr_threshold + 0.1;

    // Build self_scores map for BLAST functions
    let blast_self_scores: FxHashMap<usize, f64> = schema
        .representatives
        .iter()
        .enumerate()
        .map(|(i, rep)| (i, rep.self_score))
        .collect();

    let cluster_results = if config.align_mode == crate::types::AlignMode::Compatible {
        eprintln!("  Using BLAST (compatible mode)");
        let raw_results = blast::blast_phase4_raw(
            &cluster_input,
            &schema.representatives,
            &blast_self_scores,
            phase4_threshold,
            &config.blastp_path,
            config.cpu_cores,
        );
        // Post-filter: replicate chewBBACA's behavior.
        // Python flow:
        //   1. Per-cluster BLAST (only CDS that cluster with a rep are BLASTed against it)
        //   2. Concatenate per-cluster results per locus
        //   3. select_highest_scores: sort by slen DESC, keep first per (locus, target)
        //   4. process_blast_results: BSR filter
        //
        // Rust equivalent with all-vs-all BLAST:
        //   1. Cluster filter (replaces per-cluster BLAST restriction)
        //   2. Per-locus dedup (replaces select_highest_scores)
        //   3. BSR filter (replaces process_blast_results)
        let proteins_map: FxHashMap<usize, &[u8]> = cluster_input
            .iter()
            .map(|(idx, prot)| (*idx, prot.as_slice()))
            .collect();
        let raw_count = raw_results.len();
        // Step 1: Cluster filter — only keep hits where CDS clusters with the representative.
        let seq_num_cluster = config.max_targets;
        let mut cluster_cache: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
        let cluster_filtered: Vec<_> = raw_results
            .into_iter()
            .filter(|hit| {
                let Some(protein) = proteins_map.get(&hit.cds_idx) else {
                    return false;
                };
                let clusters = cluster_cache.entry(hit.cds_idx).or_insert_with(|| {
                    cluster::find_clusters_similarity_compat(
                        protein,
                        &index,
                        k,
                        w,
                        phase4_similarity,
                        seq_num_cluster,
                    )
                });
                clusters.contains(&hit.representative_idx)
            })
            .collect();
        let cluster_filtered_count = cluster_filtered.len();
        // Step 2: Sort by slen DESC, then rep seq_id lexicographic ASC.
        // Python's select_highest_scores sorts by slen (x[5]) DESC. When slen ties,
        // Python's stable sort preserves concatenation order. We approximate with
        // rep_id lexicographic ASC, which works for most cases.
        let mut sorted_hits = cluster_filtered;
        sorted_hits.sort_by(|a, b| {
            b.query_len
                .cmp(&a.query_len) // slen DESC
                .then_with(|| {
                    let a_id = &schema.representatives[a.representative_idx].seq_id;
                    let b_id = &schema.representatives[b.representative_idx].seq_id;
                    a_id.cmp(b_id) // rep allele name lexicographic ASC
                })
        });
        // Step 3: Per-locus dedup — keep first hit per (locus, cds_idx)
        let mut seen_per_locus: FxHashMap<u32, FxHashSet<usize>> = FxHashMap::default();
        let deduped: Vec<_> = sorted_hits
            .into_iter()
            .filter(|hit| {
                seen_per_locus
                    .entry(hit.best_locus)
                    .or_default()
                    .insert(hit.cds_idx)
            })
            .collect();
        let deduped_count = deduped.len();
        // Step 4: BSR filter
        let filtered: Vec<_> = deduped
            .into_iter()
            .filter(|hit| hit.best_bsr >= phase4_threshold)
            .collect();
        // Diagnostic
        let clustered_count = cluster_cache.values().filter(|v| !v.is_empty()).count();
        let unclustered_count = cluster_cache.values().filter(|v| v.is_empty()).count();
        eprintln!(
            "  Clustering: {}/{} CDS clustered ({} unclustered)",
            clustered_count,
            cluster_cache.len(),
            unclustered_count
        );
        eprintln!(
            "  BLAST raw hits: {}, after cluster: {}, after dedup: {}, after BSR: {}",
            raw_count,
            cluster_filtered_count,
            deduped_count,
            filtered.len()
        );
        filtered
    } else if config.brute_residual {
        eprintln!(
            "  Brute-residual: aligning {} residual CDS against all {} representatives",
            cluster_input.len(),
            schema.representatives.len()
        );
        cluster::cluster_and_align_multi_brute(&cluster_input, &schema.representatives)
    } else if config.lexicographic_minimizers {
        cluster::cluster_and_align_multi_similarity_compat(
            &cluster_input,
            &schema.representatives,
            &index,
            k,
            w,
            phase4_similarity,
            30,
        )
    } else {
        cluster::cluster_and_align_multi_similarity(
            &cluster_input,
            &schema.representatives,
            &index,
            k,
            w,
            phase4_similarity,
            30,
        )
    };

    eprintln!("  Phase 4 candidate hits: {}", cluster_results.len());

    let phase4_hits_by_locus = group_hits_by_locus(cluster_results);
    let mut phase4_loci: Vec<_> = phase4_hits_by_locus.keys().copied().collect();
    phase4_loci.sort_unstable();
    let mut phase4_excluded: FxHashSet<usize> = FxHashSet::default();
    for locus_idx in phase4_loci {
        if let Some(hits) = phase4_hits_by_locus.get(&locus_idx) {
            let (excluded, _) = process_locus_hits(
                locus_idx,
                hits,
                phase4_threshold,
                None,
                false,
                &unmatched_cds,
                &protein_hashes_by_idx,
                &cds_indices_by_protein_hash,
                &hash_to_genomes,
                &mut all_results,
                &mut next_allele_id,
                &mut novel_alleles,
                &mut schema,
                &loci_list,
                config,
            );
            phase4_excluded.extend(&excluded);
        }
    }

    // --- Phase 5: Representative determination ---
    let t5 = std::time::Instant::now();
    eprintln!(
        "  [TIMING] Phase 4: {:.1}s",
        t5.duration_since(t4).as_secs_f64()
    );
    eprintln!("  Phase 4 excluded: {}", phase4_excluded.len());
    let mut still_unmatched: Vec<(usize, Vec<u8>)> = cluster_input
        .iter()
        .filter(|(idx, _)| !phase4_excluded.contains(idx))
        .cloned()
        .collect();
    eprintln!(
        "[Phase 5] Representative determination ({} unmatched)...",
        still_unmatched.len()
    );

    let mut repdet_match_count = 0usize;
    let mut current_reps = schema.representatives.clone();
    let max_repdet_iterations = 10;
    let repdet_threshold = config.bsr_threshold;
    let repdet_candidate_upper = repdet_threshold + 0.1;

    for iteration in 1..=max_repdet_iterations {
        if still_unmatched.is_empty() || current_reps.is_empty() {
            break;
        }
        let t5_iter_start = std::time::Instant::now();

        let repdet_hits = if config.align_mode == crate::types::AlignMode::Compatible {
            // Compatible mode: use BLAST for repdet
            let repdet_self_scores: FxHashMap<usize, f64> = current_reps
                .iter()
                .enumerate()
                .map(|(i, rep)| (i, rep.self_score))
                .collect();
            blast::blast_repdet(
                &still_unmatched,
                &current_reps,
                &repdet_self_scores,
                repdet_threshold,
                &config.blastp_path,
                config.cpu_cores,
            )
        } else if config.brute_residual {
            cluster::cluster_and_align_multi_brute_bsr(
                &still_unmatched,
                &current_reps,
                repdet_threshold,
            )
        } else {
            let rep_index = cluster::build_minimizer_index(&current_reps, k, w);
            cluster::cluster_and_align_multi_limited_bsr(
                &still_unmatched,
                &current_reps,
                &rep_index,
                k,
                w,
                min_shared,
                5,
                repdet_threshold,
            )
        };
        let t5_align_done = std::time::Instant::now();
        eprintln!(
            "    iter {}: align {:.1}s ({} unmatched, {} reps, {} hits)",
            iteration,
            t5_align_done.duration_since(t5_iter_start).as_secs_f64(),
            still_unmatched.len(),
            current_reps.len(),
            repdet_hits.len()
        );
        if repdet_hits.is_empty() {
            break;
        }
        repdet_match_count += repdet_hits.len();

        let hits_by_locus = group_hits_by_locus(repdet_hits);

        // Split hits into validated (BSR >= threshold+0.1) and borderline (threshold..threshold+0.1)
        // Only borderline hits need BLAST re-validation
        let borderline_by_locus: FxHashMap<usize, Vec<cluster::ClusterResult>> = hits_by_locus
            .iter()
            .filter_map(|(&locus, hits)| {
                let borderline: Vec<_> = hits
                    .iter()
                    .filter(|h| {
                        h.best_bsr >= repdet_threshold && h.best_bsr < repdet_candidate_upper
                    })
                    .cloned()
                    .collect();
                if borderline.is_empty() {
                    None
                } else {
                    Some((locus, borderline))
                }
            })
            .collect();

        // Validate borderline hits with parasail (deterministic, no BLAST subprocess)
        let parasail_validated: FxHashMap<usize, Vec<cluster::ClusterResult>> = if config.align_mode
            == crate::types::AlignMode::Compatible
            || borderline_by_locus.is_empty()
        {
            FxHashMap::default()
        } else {
            blast::validate_repdet_hits_parasail(
                &borderline_by_locus,
                &proteins_by_idx,
                &current_reps,
                repdet_threshold,
            )
        };

        // Merge: validated (high BSR) + BLAST-validated borderline
        let mut validated_by_locus: FxHashMap<usize, Vec<cluster::ClusterResult>> =
            FxHashMap::default();
        for (&locus, hits) in &hits_by_locus {
            let mut merged: Vec<cluster::ClusterResult> =
                if config.align_mode == crate::types::AlignMode::Compatible {
                    // Compatible mode: BLAST scores already correct, keep all above threshold
                    hits.iter()
                        .filter(|h| h.best_bsr >= repdet_threshold)
                        .cloned()
                        .collect()
                } else {
                    // Fast mode: keep high-BSR hits directly
                    hits.iter()
                        .filter(|h| h.best_bsr >= repdet_candidate_upper)
                        .cloned()
                        .collect()
                };
            // Add BLAST-validated borderline hits
            if let Some(validated) = parasail_validated.get(&locus) {
                merged.extend(validated.iter().cloned());
            }
            merged.sort_unstable_by(|a, b| b.score.cmp(&a.score).then(a.cds_idx.cmp(&b.cds_idx)));
            validated_by_locus.insert(locus, merged);
        }

        let mut loci: Vec<_> = validated_by_locus.keys().copied().collect();
        loci.sort_unstable();

        let mut newly_excluded: FxHashSet<usize> = FxHashSet::default();
        let mut candidate_cds_by_locus: FxHashMap<usize, Vec<usize>> = FxHashMap::default();

        for locus_idx in loci {
            if let Some(validated_hits) = validated_by_locus.get(&locus_idx) {
                let (excluded, candidates) = process_locus_hits(
                    locus_idx,
                    &validated_hits,
                    repdet_threshold,
                    Some(repdet_candidate_upper),
                    false,
                    &unmatched_cds,
                    &protein_hashes_by_idx,
                    &cds_indices_by_protein_hash,
                    &hash_to_genomes,
                    &mut all_results,
                    &mut next_allele_id,
                    &mut novel_alleles,
                    &mut schema,
                    &loci_list,
                    config,
                );
                newly_excluded.extend(excluded);
                if !candidates.is_empty() {
                    candidate_cds_by_locus.insert(locus_idx, candidates);
                }
            }
        }

        let t5_classify_done = std::time::Instant::now();
        eprintln!(
            "    iter {}: classify {:.1}s",
            iteration,
            t5_classify_done.duration_since(t5_align_done).as_secs_f64()
        );

        still_unmatched.retain(|(cds_idx, _)| !newly_excluded.contains(cds_idx));

        let selected_reps = select_new_representatives(
            &candidate_cds_by_locus,
            &unmatched_cds,
            &proteins_by_idx,
            repdet_threshold,
            config.align_mode == crate::types::AlignMode::Compatible,
        );
        eprintln!(
            "  RepDet iter {}: hits={}, classified={}, selected={}",
            iteration,
            repdet_match_count,
            newly_excluded.len(),
            selected_reps.len(),
        );

        if selected_reps.is_empty() {
            break;
        }
        current_reps = selected_reps;
    }

    eprintln!("  RepDet matches: {}", repdet_match_count);
    let t5_rescue = std::time::Instant::now();
    eprintln!(
        "  [TIMING] Phase 5 RepDet iters: {:.1}s",
        t5_rescue.duration_since(t5).as_secs_f64()
    );

    // Rescue phase: re-BLAST unassigned CDS against all representatives.
    // Python chewBBACA does NOT have this rescue phase, so skip in compatible mode.
    if config.align_mode != crate::types::AlignMode::Compatible {
        let assigned_hashes: FxHashSet<SeqHash> = all_results
            .iter()
            .flat_map(|genome_results| genome_results.iter())
            .filter_map(|result| {
                if result.class.is_valid() {
                    result.dna_hash
                } else {
                    None
                }
            })
            .collect();

        let rescue_input: Vec<(usize, Vec<u8>)> = all_translated
            .iter()
            .filter(|(_, _, dna_hash)| !assigned_hashes.contains(dna_hash))
            .map(|(idx, protein, _)| (*idx, protein.clone()))
            .collect();
        eprintln!(
            "  [TIMING] Rescue: {} proteins to re-align",
            rescue_input.len()
        );
        if !rescue_input.is_empty() {
            let rescue_hits = cluster::cluster_and_align_multi_similarity(
                &rescue_input,
                &schema.representatives,
                &index,
                k,
                w,
                phase4_similarity,
                30,
            );
            let rescue_hits_by_locus = group_hits_by_locus(rescue_hits);
            let mut rescue_loci: Vec<_> = rescue_hits_by_locus.keys().copied().collect();
            rescue_loci.sort_unstable();
            for locus_idx in rescue_loci {
                if let Some(hits) = rescue_hits_by_locus.get(&locus_idx) {
                    let _ = process_locus_hits(
                        locus_idx,
                        hits,
                        phase4_threshold,
                        None,
                        true,
                        &unmatched_cds,
                        &protein_hashes_by_idx,
                        &cds_indices_by_protein_hash,
                        &hash_to_genomes,
                        &mut all_results,
                        &mut next_allele_id,
                        &mut novel_alleles,
                        &mut schema,
                        &loci_list,
                        config,
                    );
                }
            }
        }
    }

    mark_paralogous_matches(&mut all_results);

    // --- Phase 6: Build contigs info ---
    let t6 = std::time::Instant::now();
    eprintln!(
        "  [TIMING] Phase 5: {:.1}s",
        t6.duration_since(t5).as_secs_f64()
    );
    eprintln!("[Phase 6] Building contigs info...");
    // O(all_cds) scan: for each CDS, check if it was classified to a locus
    // Uses pre-computed hashes from Phase 2 (avoids re-hashing 4M+ sequences)
    for (hash, occurrences) in &hash_to_genomes {
        let Some(&(locus_idx, _, _)) = cds_classifications.get(hash) else {
            continue;
        };
        let li = locus_idx as usize;
        let cds_length = hash_to_len.get(hash).copied().unwrap_or(0);
        for (genome_idx, coord) in occurrences {
            let gi = *genome_idx as usize;
            if gi < all_results.len() && li < num_loci && all_results[gi][li].class.is_valid() {
                if let Some(coord) = coord {
                    contigs_info.push(output::ContigInfo {
                        genome: genome_paths[gi].clone(),
                        contig: coord.contig.clone(),
                        locus: loci_list[li].clone(),
                        start: coord.start,
                        stop: coord.stop,
                        strand: coord.strand,
                        cds_length,
                        class: all_results[gi][li].class,
                    });
                }
            }
        }
    }

    // --- Phase 7: Write output files ---
    let t7 = std::time::Instant::now();
    eprintln!(
        "  [TIMING] Phase 6: {:.1}s",
        t7.duration_since(t6).as_secs_f64()
    );
    eprintln!("[Phase 7] Writing output...");
    let genome_names: Vec<String> = genome_paths
        .iter()
        .map(|p| {
            Path::new(p)
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string()
        })
        .collect();

    output::write_alleles_tsv(
        &output_dir.join("results_alleles.tsv"),
        &genome_names,
        &loci_list,
        &all_results,
        &schema.inferred_allele_ids,
    )?;

    output::write_alleles_hashed_tsv(
        &output_dir.join("results_alleles_hashed.tsv"),
        &genome_names,
        &loci_list,
        &all_results,
        &schema.allele_crc32,
    )?;

    output::write_statistics_tsv(
        &output_dir.join("results_statistics.tsv"),
        &genome_names,
        &loci_list,
        &all_results,
    )?;

    output::write_loci_summary(
        &output_dir.join("loci_summary_stats.tsv"),
        &loci_list,
        &all_results,
    )?;

    output::write_novel_alleles(&output_dir.join("novel_alleles.fasta"), &novel_alleles)?;

    // Update schema in place if requested
    if config.update_schema {
        output::update_schema_fasta(schema_dir, &novel_alleles, &loci_list)?;
        let n = novel_alleles.len();
        eprintln!("[Schema] Appended {n} novel alleles to schema FASTA files.");
    }

    output::write_contigs_info(&output_dir.join("results_contigsInfo.tsv"), &contigs_info)?;

    // Summary
    let mut class_counts: FxHashMap<Classification, usize> = FxHashMap::default();
    for genome_results in &all_results {
        for result in genome_results {
            *class_counts.entry(result.class).or_default() += 1;
        }
    }
    eprintln!("\n=== AlleleCall Summary ===");
    eprintln!("Genomes: {}", num_genomes);
    eprintln!("Loci: {}", num_loci);
    for cls in &[
        Classification::EXC,
        Classification::INF,
        Classification::PLOT3,
        Classification::PLOT5,
        Classification::LOTSC,
        Classification::NIPH,
        Classification::NIPHEM,
        Classification::ALM,
        Classification::ASM,
        Classification::PAMA,
        Classification::LNF,
    ] {
        eprintln!(
            "  {}: {}",
            cls.as_str(),
            class_counts.get(cls).unwrap_or(&0)
        );
    }

    Ok(())
}

/// Discover loci from schema directory (list all FASTA files).
fn discover_loci(schema_dir: &Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut loci = Vec::new();
    let short_dir = schema_dir.join("short");

    // Look for FASTA files in the short/ directory to determine locus list
    if short_dir.exists() {
        for entry in std::fs::read_dir(&short_dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with("_short.fasta") {
                let locus_name = name.trim_end_matches("_short.fasta").to_string();
                loci.push(locus_name);
            }
        }
    }

    // Fallback: look in main schema dir
    if loci.is_empty() {
        for entry in std::fs::read_dir(schema_dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".fasta") && !entry.path().is_dir() {
                let locus_name = name.trim_end_matches(".fasta").to_string();
                loci.push(locus_name);
            }
        }
    }

    loci.sort();
    Ok(loci)
}

/// Find prodigal training file (.trn) in the schema directory.
fn find_training_file(schema_dir: &Path) -> Option<std::path::PathBuf> {
    if let Ok(entries) = std::fs::read_dir(schema_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".trn") {
                return Some(entry.path());
            }
        }
    }
    None
}

fn load_self_scores_cache(path: &Path, reps: &[Representative]) -> Option<Vec<f64>> {
    use std::collections::HashMap;

    let content = std::fs::read_to_string(path).ok()?;
    let mut by_id: HashMap<&str, f64> = HashMap::new();
    for line in content.lines() {
        let mut parts = line.splitn(2, '\t');
        let id = parts.next()?;
        let score = parts.next()?.parse().ok()?;
        by_id.insert(id, score);
    }

    let mut scores = Vec::with_capacity(reps.len());
    for rep in reps {
        scores.push(*by_id.get(rep.seq_id.as_str())?);
    }
    Some(scores)
}

/// Load self-scores from Python chewBBACA pickle (short/self_scores).
/// Format: dict { rep_id (str) → (prot_len (int), blast_raw_score (float)) }
fn load_python_self_scores_pickle(path: &Path, reps: &[Representative]) -> Option<Vec<f64>> {
    use std::collections::HashMap;

    let data = std::fs::read(path).ok()?;
    // Quick heuristic parse: scan for rep_id strings and extract the float score that follows.
    // Python pickle stores string keys and tuple values.
    // We look for each rep's seq_id in the binary data and extract the associated score.
    //
    // More robust approach: find key strings and scan forward for the float64 value.
    // Pickle3: X + 4-byte len + string for keys, G + 8-byte double for floats, K/J/M for ints.

    let mut by_id: HashMap<String, f64> = HashMap::new();

    // Scan for each rep_id.
    // Must find an exact match (not a substring of a longer key).
    // In pickle protocol 4, string keys are preceded by a length prefix and followed
    // by a MEMOIZE (0x94) opcode. Check that the byte after the key is NOT alphanumeric
    // to avoid matching "key_1" inside "key_110".
    for rep in reps {
        let key_bytes = rep.seq_id.as_bytes();
        let mut search_from = 0;
        let found_pos = loop {
            if let Some(rel_pos) = data[search_from..]
                .windows(key_bytes.len())
                .position(|w| w == key_bytes)
            {
                let pos = search_from + rel_pos;
                let after = pos + key_bytes.len();
                // Accept if at end of data or next byte is not alphanumeric/underscore
                if after >= data.len()
                    || !matches!(data[after], b'0'..=b'9' | b'a'..=b'z' | b'A'..=b'Z' | b'_')
                {
                    break Some(pos);
                }
                // This was a substring match (e.g., "_1" matched inside "_110"); skip it
                search_from = pos + 1;
            } else {
                break None;
            }
        };
        if let Some(pos) = found_pos {
            // After the key string in pickle, the format is:
            //   \x94 (MEMOIZE)
            //   M + 2 bytes (BININT2 for prot_len) OR K + 1 byte (BINUINT1) OR J + 4 bytes (BININT)
            //   G + 8 bytes (BINFLOAT for self_score)
            //   \x86 (TUPLE2)
            //
            // We must properly skip the integer opcode to find the real BINFLOAT (G),
            // because 'G' (0x47) can appear inside integer data bytes.
            let search_start = pos + key_bytes.len();
            let search_end = (search_start + 30).min(data.len());
            let window = &data[search_start..search_end];

            let mut i = 0;
            while i < window.len() {
                match window[i] {
                    0x94 => {
                        i += 1;
                    } // MEMOIZE: skip
                    b'K' => {
                        i += 2;
                    } // BINUINT1: skip opcode + 1 byte
                    b'M' => {
                        i += 3;
                    } // BININT2: skip opcode + 2 bytes
                    b'J' => {
                        i += 5;
                    } // BININT: skip opcode + 4 bytes
                    b'G' if i + 9 <= window.len() => {
                        // BINFLOAT: 8-byte big-endian double
                        let bytes: [u8; 8] = window[i + 1..i + 9].try_into().ok()?;
                        let val = f64::from_be_bytes(bytes);
                        if val > 0.0 && val < 1e10 {
                            by_id.insert(rep.seq_id.clone(), val);
                        }
                        break;
                    }
                    0x86 => {
                        break;
                    } // TUPLE2: end of tuple, stop
                    _ => {
                        i += 1;
                    } // Unknown opcode, skip
                }
            }
        }
    }

    if by_id.len() < reps.len() / 2 {
        // Too few matches, pickle format might be wrong
        return None;
    }

    let mut scores = Vec::with_capacity(reps.len());
    for rep in reps {
        if let Some(&score) = by_id.get(&rep.seq_id) {
            scores.push(score);
        } else {
            // Fallback to parasail for missing entries
            scores.push(crate::sw::self_score(&rep.protein_seq) as f64);
        }
    }
    Some(scores)
}

fn save_self_scores_cache(path: &Path, reps: &[Representative]) {
    use std::io::Write;
    if let Ok(file) = std::fs::File::create(path) {
        let mut w = std::io::BufWriter::new(file);
        for rep in reps {
            let _ = writeln!(w, "{}\t{}", rep.seq_id, rep.self_score);
        }
    }
}
