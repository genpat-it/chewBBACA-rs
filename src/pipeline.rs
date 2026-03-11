//! Pipeline orchestration: wires all phases together.

use std::path::Path;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::types::*;
use crate::schema;
use crate::cds;
use crate::dedup;
use crate::translate;
use crate::sw;
use crate::cluster;
use crate::classify;
use crate::repdet;
use crate::output;

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
        Some(std::thread::spawn(|| -> Result<crate::gpu_sw::GpuAligner, String> {
            crate::gpu_sw::GpuAligner::new().map_err(|e| e.to_string())
        }))
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

    // Find prodigal training file in schema directory
    let training_file = find_training_file(schema_dir);
    if let Some(ref trn) = training_file {
        eprintln!("  Using training file: {}", trn.display());
    }

    // Compute (or load cached) self-scores for representatives
    let cache_path = schema_dir.join("short").join("self_scores_rs.tsv");
    let cached = load_self_scores_cache(&cache_path, num_loci);

    if let Some(scores) = cached {
        eprintln!("[Phase 0] Loading cached self-scores...");
        for (i, rep) in schema.representatives.iter_mut().enumerate() {
            rep.self_score = scores[i];
        }
    } else {
        eprintln!("[Phase 0] Computing representative self-scores...");
        for rep in &mut schema.representatives {
            let score = sw::self_score(&rep.protein_seq);
            rep.self_score = score as f64;
        }
        save_self_scores_cache(&cache_path, &schema.representatives, &loci_list);
    }
    for (i, rep) in schema.representatives.iter().enumerate() {
        schema.loci[i].self_score = rep.self_score;
    }

    let num_genomes = genome_paths.len();
    eprintln!("[Phase 0] Processing {} genomes against {} loci", num_genomes, num_loci);

    // Initialize results: all LNF by default
    let mut all_results: Vec<Vec<LocusResult>> = (0..num_genomes)
        .map(|_| {
            (0..num_loci)
                .map(|_| LocusResult {
                    class: Classification::LNF,
                    allele_id: None,
                    is_novel: false, matches: Vec::new(),
                })
                .collect()
        })
        .collect();

    let mut novel_alleles: Vec<(String, Vec<u8>)> = Vec::new();
    let mut contigs_info: Vec<output::ContigInfo> = Vec::new();

    // Track next allele ID per locus (for INF assignments)
    let mut next_allele_id: Vec<AlleleId> = schema.loci.iter()
        .map(|l| l.allele_count + 1)
        .collect();

    // --- Phase 1: CDS prediction (parallel per genome) ---
    let t1 = std::time::Instant::now(); eprintln!("  [TIMING] Phase 0: {:.1}s", t1.duration_since(t0).as_secs_f64()); eprintln!("[Phase 1] CDS prediction...");
    let genome_cds: Vec<(Vec<Cds>, u32)> = genome_paths
        .par_iter()
        .enumerate()
        .map(|(genome_idx, path)| {
            let genome_path = Path::new(path);
            let genome_stem = genome_path.file_stem().unwrap_or_default().to_string_lossy();

            // Check for pre-computed CDS
            if let Some(cds_dir) = cds_input_dir {
                let cds_file = cds_dir.join(format!("{}.cds.fasta", genome_stem));
                if cds_file.exists() {
                    return cds::load_precomputed_cds(&cds_file, genome_idx as GenomeIdx);
                }
            }

            // Otherwise, run prodigal
            let (mut cds_list, invalid) = cds::predict_cds(
                genome_path,
                genome_idx as GenomeIdx,
                config.translation_table,
                &config.prodigal_mode,
                training_file.as_deref(),
                &config.prodigal_path,
            );

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
        })
        .collect();

    let total_cds: usize = genome_cds.iter().map(|(cds, _)| cds.len()).sum();
    eprintln!("  Total CDS predicted: {}", total_cds);

    // Flatten all CDS
    let mut all_cds: Vec<Cds> = Vec::with_capacity(total_cds);
    for (cds_list, _) in genome_cds {
        all_cds.extend(cds_list);
    }

    // --- Phase 2: Deduplication ---
    let t2 = std::time::Instant::now(); eprintln!("  [TIMING] Phase 1: {:.1}s", t2.duration_since(t1).as_secs_f64()); eprintln!("[Phase 2] Deduplicating CDS...");
    let (distinct_cds, hash_to_genomes) = dedup::deduplicate_cds(&all_cds);
    eprintln!("  Distinct CDS: {} (from {})", distinct_cds.len(), all_cds.len());

    // Track classification per CDS hash → locus
    // We'll track per-genome, per-locus results through hash_to_genomes mapping.
    let mut cds_classifications: FxHashMap<SeqHash, (LocusIdx, Classification, Option<AlleleId>)> =
        FxHashMap::default();

    // --- Phase 3a: Exact DNA matching ---
    let t3a = std::time::Instant::now(); eprintln!("  [TIMING] Phase 2: {:.1}s", t3a.duration_since(t2).as_secs_f64()); eprintln!("[Phase 3a] Exact DNA matching...");
    let mut unmatched_cds: Vec<&Cds> = Vec::new();
    let mut dna_exact_count = 0u32;

    for cds_item in &distinct_cds {
        let upper: Vec<u8> = cds_item.dna_seq.iter().map(|b| b.to_ascii_uppercase()).collect();
        let hash = schema::sha256(&upper);

        if let Some(matches) = schema.dna_hashes.get(&hash) {
            // Exact DNA match → EXC (may match multiple loci)
            dna_exact_count += 1;
            for &(locus_idx, allele_id) in matches {
                cds_classifications.insert(hash, (locus_idx, Classification::EXC, Some(allele_id)));
                // Apply to all genomes with this hash
                if let Some(genomes) = hash_to_genomes.get(&hash) {
                    for &(genome_idx, _) in genomes {
                        let gi = genome_idx as usize;
                        let li = locus_idx as usize;
                        if gi < all_results.len() && li < num_loci {
                            if all_results[gi][li].class == Classification::LNF {
                                all_results[gi][li] = LocusResult {
                                    class: Classification::EXC,
                                    allele_id: Some(allele_id),
                                    is_novel: false, matches: Vec::new(),
                                };
                            } else if all_results[gi][li].class == Classification::EXC {
                                // Multiple EXC matches for same locus → NIPHEM
                                all_results[gi][li].class = Classification::NIPHEM;
                            }
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
        let upper: Vec<u8> = cds_item.dna_seq.iter().map(|b| b.to_ascii_uppercase()).collect();
        if let Some(protein) = translate::translate(&upper, config.translation_table) {
            if protein.len() >= config.min_sequence_length as usize / 3 {
                let dna_hash = schema::sha256(&upper);
                translated.push((idx, protein, dna_hash));
            }
        }
    }
    eprintln!("  Translated proteins: {}", translated.len());

    // --- Phase 3c: Exact protein matching ---
    eprintln!("[Phase 3c] Exact protein matching...");
    let mut unmatched_proteins: Vec<(usize, Vec<u8>, SeqHash)> = Vec::new();
    let mut prot_exact_count = 0u32;

    for (idx, protein, dna_hash) in translated {
        let prot_hash = schema::sha256(&protein);

        if let Some(matches) = schema.protein_hashes.get(&prot_hash) {
            prot_exact_count += 1;
            for &(locus_idx, _allele_id) in matches {
                let li = locus_idx as usize;
                if let Some(genomes) = hash_to_genomes.get(&dna_hash) {
                    // Sort genomes to process in order (first gets INF, rest get EXC)
                    let mut sorted_genomes: Vec<_> = genomes.iter().collect();
                    sorted_genomes.sort_by_key(|(gi, _)| *gi);

                    let mut inf_allele_id: Option<AlleleId> = None;
                    for &&(genome_idx, _) in &sorted_genomes {
                        let gi = genome_idx as usize;
                        if gi < all_results.len() && li < num_loci {
                            if all_results[gi][li].class == Classification::LNF {
                                if let Some(aid) = inf_allele_id {
                                    // Subsequent genomes: EXC with same allele
                                    all_results[gi][li] = LocusResult {
                                        class: Classification::EXC,
                                        allele_id: Some(aid),
                                        is_novel: true, matches: Vec::new(),
                                    };
                                    // Add to schema DNA hashes for future matches
                                    schema.dna_hashes.entry(dna_hash)
                                        .or_default()
                                        .push((locus_idx, aid));
                                } else {
                                    // First genome: INF
                                    let inf_id = next_allele_id[li];
                                    next_allele_id[li] += 1;
                                    inf_allele_id = Some(inf_id);
                                    all_results[gi][li] = LocusResult {
                                        class: Classification::INF,
                                        allele_id: Some(inf_id),
                                        is_novel: true, matches: Vec::new(),
                                    };
                                    let cds_item = unmatched_cds[idx];
                                    novel_alleles.push((
                                        format!("{}_{}", loci_list[li], cds_item.id),
                                        cds_item.dna_seq.clone(),
                                    ));
                                    // CRC32 for hashed output
                                    let seq_str = String::from_utf8_lossy(&cds_item.dna_seq);
                                    schema.allele_crc32.insert((locus_idx, inf_id), crc32fast::hash(seq_str.as_bytes()));
                                    // Add to schema DNA hashes
                                    schema.dna_hashes.entry(dna_hash)
                                        .or_default()
                                        .push((locus_idx, inf_id));
                                }
                            } else if all_results[gi][li].class == Classification::EXC {
                                all_results[gi][li].class = Classification::NIPHEM;
                            }
                        }
                    }
                }
                break;
            }
        } else {
            unmatched_proteins.push((idx, protein, dna_hash));
        }
    }
    eprintln!("  Protein exact matches: {}", prot_exact_count);

    // --- Phase 4: Clustering + alignment ---
    let t4 = std::time::Instant::now(); eprintln!("  [TIMING] Phases 3a-c: {:.1}s", t4.duration_since(t3a).as_secs_f64()); eprintln!("[Phase 4] Clustering + SW alignment{}...", if config.use_gpu { " (GPU)" } else { "" });
    let k = 5;
    let w = 5;
    let min_shared = 1;

    let index = cluster::build_minimizer_index(&schema.representatives, k, w);

    let cluster_input: Vec<(usize, Vec<u8>)> = unmatched_proteins
        .iter()
        .map(|(idx, protein, _)| (*idx, protein.clone()))
        .collect();

    // Build alignment pairs on CPU (no GPU needed yet)
    let (pair_protein_idx, pair_rep_idx) = cluster::build_alignment_pairs(
        &cluster_input, &index, k, w, min_shared,
    );
    eprintln!("  {} alignment pairs from {} proteins", pair_protein_idx.len(), cluster_input.len());

    // NOW wait for GPU init (started at top of function, overlaps with phases 0-3 + clustering)
    let gpu_aligner = if let Some(handle) = gpu_handle {
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

    let cluster_results = if let Some(ref aligner) = gpu_aligner {
        cluster::align_pairs_gpu(
            &cluster_input,
            &schema.representatives,
            &pair_protein_idx,
            &pair_rep_idx,
            aligner,
        )
    } else {
        cluster::cluster_and_align(
            &cluster_input,
            &schema.representatives,
            &index,
            k, w, min_shared,
        )
    };

    eprintln!("  Cluster alignment results: {}", cluster_results.len());

    // Process cluster results
    let mut still_unmatched: Vec<(usize, Vec<u8>)> = Vec::new();
    let mut matched_in_cluster: FxHashMap<usize, bool> = FxHashMap::default();

    for result in &cluster_results {
        if result.best_bsr >= config.bsr_threshold {
            matched_in_cluster.insert(result.cds_idx, true);

            let cds_item = unmatched_cds[result.cds_idx];
            let li = result.best_locus as usize;
            let dna_upper: Vec<u8> = cds_item.dna_seq.iter().map(|b| b.to_ascii_uppercase()).collect();
            let dna_hash = schema::sha256(&dna_upper);

            // Classify this inexact match using target (representative) alignment positions
            let class = classify::classify_inexact(
                result.best_bsr,
                config.bsr_threshold,
                cds_item.dna_seq.len() as u32,
                schema.loci[li].mode_length,
                config.size_threshold,
                cds_item.coord.as_ref(),
                schema.representatives[li].dna_length,
                result.target_start,
                result.target_end,
                result.target_len,
            );

            // Apply to all genomes with this CDS (first gets INF, rest EXC)
            if let Some(genomes) = hash_to_genomes.get(&dna_hash) {
                let mut sorted_genomes: Vec<_> = genomes.iter().collect();
                sorted_genomes.sort_by_key(|(gi, _)| *gi);

                let mut inf_allele_id: Option<AlleleId> = None;
                for &&(genome_idx, _) in &sorted_genomes {
                    let gi = genome_idx as usize;
                    if gi < all_results.len() && li < num_loci {
                        if all_results[gi][li].class == Classification::LNF {
                            if class == Classification::INF {
                                if let Some(aid) = inf_allele_id {
                                    // Subsequent genomes: EXC with same allele
                                    all_results[gi][li] = LocusResult {
                                        class: Classification::EXC,
                                        allele_id: Some(aid),
                                        is_novel: true, matches: Vec::new(),
                                    };
                                } else {
                                    // First genome: INF
                                    let inf_id = next_allele_id[li];
                                    next_allele_id[li] += 1;
                                    inf_allele_id = Some(inf_id);
                                    all_results[gi][li] = LocusResult {
                                        class: Classification::INF,
                                        allele_id: Some(inf_id),
                                        is_novel: true, matches: Vec::new(),
                                    };
                                    novel_alleles.push((
                                        format!("{}_{}", loci_list[li], cds_item.id),
                                        cds_item.dna_seq.clone(),
                                    ));
                                    // CRC32 for hashed output
                                    let seq_str = String::from_utf8_lossy(&cds_item.dna_seq);
                                    schema.allele_crc32.insert((result.best_locus, inf_id), crc32fast::hash(seq_str.as_bytes()));
                                    // Add to schema for future matches
                                    schema.dna_hashes.entry(dna_hash)
                                        .or_default()
                                        .push((result.best_locus, inf_id));
                                }
                            } else {
                                all_results[gi][li] = LocusResult {
                                    class,
                                    allele_id: None,
                                    is_novel: false, matches: Vec::new(),
                                };
                            }
                        } else if all_results[gi][li].class == Classification::EXC && class.is_valid() {
                            all_results[gi][li].class = Classification::NIPHEM;
                        } else if all_results[gi][li].class.is_valid() && class.is_valid() {
                            all_results[gi][li].class = Classification::NIPH;
                        }
                    }
                }
            }
        }
    }

    // Collect still-unmatched for repdet
    for (idx, protein, _) in &unmatched_proteins {
        if !matched_in_cluster.contains_key(idx) {
            still_unmatched.push((*idx, protein.clone()));
        }
    }

    // --- Phase 5: Representative determination ---
    let t5 = std::time::Instant::now(); eprintln!("  [TIMING] Phase 4: {:.1}s", t5.duration_since(t4).as_secs_f64()); eprintln!("[Phase 5] Representative determination ({} unmatched)...", still_unmatched.len());
    let mut reps = schema.representatives.clone();
    let repdet_results = repdet::iterative_repdet(
        &still_unmatched,
        &mut reps,
        config,
        k, w, min_shared,
        gpu_aligner.as_ref(),
        &cluster_results,
    );

    eprintln!("  RepDet matches: {}", repdet_results.len());

    for result in &repdet_results {
        let cds_item = unmatched_cds[result.cds_idx];
        let li = result.best_locus as usize;
        let dna_upper: Vec<u8> = cds_item.dna_seq.iter().map(|b| b.to_ascii_uppercase()).collect();
        let dna_hash = schema::sha256(&dna_upper);

        let class = classify::classify_inexact(
            result.best_bsr,
            config.bsr_threshold,
            cds_item.dna_seq.len() as u32,
            schema.loci[li].mode_length,
            config.size_threshold,
            cds_item.coord.as_ref(),
            schema.representatives[li].dna_length,
            result.target_start,
            result.target_end,
            result.target_len,
        );

        if let Some(genomes) = hash_to_genomes.get(&dna_hash) {
            let mut sorted_genomes: Vec<_> = genomes.iter().collect();
            sorted_genomes.sort_by_key(|(gi, _)| *gi);

            let mut inf_allele_id: Option<AlleleId> = None;
            for &&(genome_idx, _) in &sorted_genomes {
                let gi = genome_idx as usize;
                if gi < all_results.len() && li < num_loci {
                    if all_results[gi][li].class == Classification::LNF {
                        if class == Classification::INF {
                            if let Some(aid) = inf_allele_id {
                                all_results[gi][li] = LocusResult {
                                    class: Classification::EXC,
                                    allele_id: Some(aid),
                                    is_novel: true, matches: Vec::new(),
                                };
                            } else {
                                let inf_id = next_allele_id[li];
                                next_allele_id[li] += 1;
                                inf_allele_id = Some(inf_id);
                                all_results[gi][li] = LocusResult {
                                    class: Classification::INF,
                                    allele_id: Some(inf_id),
                                    is_novel: true, matches: Vec::new(),
                                };
                                novel_alleles.push((
                                    format!("{}_{}", loci_list[li], cds_item.id),
                                    cds_item.dna_seq.clone(),
                                ));
                                // CRC32 for hashed output
                                let seq_str = String::from_utf8_lossy(&cds_item.dna_seq);
                                schema.allele_crc32.insert((li as u32, inf_id), crc32fast::hash(seq_str.as_bytes()));
                            }
                        } else {
                            all_results[gi][li] = LocusResult {
                                class,
                                allele_id: None,
                                is_novel: false, matches: Vec::new(),
                            };
                        }
                    }
                }
            }
        }
    }

    // --- Phase 6: Build contigs info ---
    let t6 = std::time::Instant::now(); eprintln!("  [TIMING] Phase 5: {:.1}s", t6.duration_since(t5).as_secs_f64()); eprintln!("[Phase 6] Building contigs info...");
    // O(all_cds) scan: for each CDS, check if it was classified to a locus
    for cds_item in &all_cds {
        let upper: Vec<u8> = cds_item.dna_seq.iter().map(|b| b.to_ascii_uppercase()).collect();
        let hash = schema::sha256(&upper);
        if let Some(&(locus_idx, _, _)) = cds_classifications.get(&hash) {
            let gi = cds_item.genome_idx as usize;
            let li = locus_idx as usize;
            if gi < all_results.len() && li < num_loci && all_results[gi][li].class.is_valid() {
                if let Some(ref coord) = cds_item.coord {
                    contigs_info.push(output::ContigInfo {
                        genome: genome_paths[gi].clone(),
                        contig: coord.contig.clone(),
                        locus: loci_list[li].clone(),
                        start: coord.start,
                        stop: coord.stop,
                        strand: coord.strand,
                        cds_length: cds_item.dna_seq.len() as u32,
                        class: all_results[gi][li].class,
                    });
                }
            }
        }
    }

    // --- Phase 7: Write output files ---
    let t7 = std::time::Instant::now(); eprintln!("  [TIMING] Phase 6: {:.1}s", t7.duration_since(t6).as_secs_f64()); eprintln!("[Phase 7] Writing output...");
    let genome_names: Vec<String> = genome_paths
        .iter()
        .map(|p| Path::new(p).file_stem().unwrap_or_default().to_string_lossy().to_string())
        .collect();

    output::write_alleles_tsv(
        &output_dir.join("results_alleles.tsv"),
        &genome_names,
        &loci_list,
        &all_results,
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

    output::write_novel_alleles(
        &output_dir.join("novel_alleles.fasta"),
        &novel_alleles,
    )?;

    output::write_contigs_info(
        &output_dir.join("results_contigsInfo.tsv"),
        &contigs_info,
    )?;

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
        Classification::EXC, Classification::INF,
        Classification::PLOT3, Classification::PLOT5, Classification::LOTSC,
        Classification::NIPH, Classification::NIPHEM,
        Classification::ALM, Classification::ASM,
        Classification::PAMA, Classification::LNF,
    ] {
        eprintln!("  {}: {}", cls.as_str(), class_counts.get(cls).unwrap_or(&0));
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

fn load_self_scores_cache(path: &Path, expected_count: usize) -> Option<Vec<f64>> {
    let content = std::fs::read_to_string(path).ok()?;
    let scores: Vec<f64> = content.lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 2 { parts[1].parse().ok() } else { None }
        })
        .collect();
    if scores.len() == expected_count { Some(scores) } else { None }
}

fn save_self_scores_cache(path: &Path, reps: &[Representative], loci_names: &[String]) {
    use std::io::Write;
    if let Ok(file) = std::fs::File::create(path) {
        let mut w = std::io::BufWriter::new(file);
        for (i, rep) in reps.iter().enumerate() {
            let _ = writeln!(w, "{}\t{}", loci_names[i], rep.self_score);
        }
    }
}
