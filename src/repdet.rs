//! Representative determination: iterative loop to find more matches
//! by expanding the set of representative alleles.
//!
//! In chewBBACA, after initial clustering, CDS with BSR in [0.6, 0.7)
//! become candidate new representatives. These are aligned against the
//! remaining unclassified CDS in subsequent iterations until no new
//! representatives are found.

use rustc_hash::FxHashMap;

use crate::cluster;
use crate::gpu_sw::GpuAligner;
use crate::sw;
use crate::types::*;

/// Run representative determination iterations.
///
/// `initial_results` contains cluster results from Phase 4 (first pass),
/// which avoids a redundant first alignment pass.
pub fn iterative_repdet(
    unclassified: &[(usize, Vec<u8>)],
    representatives: &mut Vec<Representative>,
    config: &Config,
    k: usize,
    w: usize,
    min_shared: usize,
    gpu_aligner: Option<&GpuAligner>,
    initial_results: &[cluster::ClusterResult],
) -> Vec<cluster::ClusterResult> {
    let mut all_results: Vec<cluster::ClusterResult> = Vec::new();
    let mut remaining: Vec<(usize, Vec<u8>)> = unclassified.to_vec();

    let max_iterations = 10;

    // Process initial_results (from Phase 4) without re-aligning
    let mut new_reps = process_results(
        initial_results, &mut remaining, &mut all_results, config,
    );

    if new_reps.is_empty() {
        // No new rep candidates from Phase 4 → nothing more to find
        return all_results;
    }

    representatives.extend(new_reps);

    for _iter in 1..max_iterations {
        if remaining.is_empty() {
            break;
        }

        // Rebuild minimizer index with current representatives
        let index = cluster::build_minimizer_index(representatives, k, w);

        // Cluster and align
        let results = if let Some(aligner) = gpu_aligner {
            cluster::cluster_and_align_gpu(
                &remaining,
                representatives,
                &index,
                k, w, min_shared,
                aligner,
            )
        } else {
            cluster::cluster_and_align(
                &remaining,
                representatives,
                &index,
                k, w, min_shared,
            )
        };

        if results.is_empty() {
            break;
        }

        new_reps = process_results(
            &results, &mut remaining, &mut all_results, config,
        );

        if new_reps.is_empty() {
            break;
        }

        representatives.extend(new_reps);
    }

    all_results
}

/// Process alignment results: extract matches, find new rep candidates,
/// remove matched from remaining. Returns new representative candidates.
fn process_results(
    results: &[cluster::ClusterResult],
    remaining: &mut Vec<(usize, Vec<u8>)>,
    all_results: &mut Vec<cluster::ClusterResult>,
    config: &Config,
) -> Vec<Representative> {
    let mut matched_indices: FxHashMap<usize, bool> = FxHashMap::default();
    let mut new_reps: Vec<Representative> = Vec::new();

    for result in results {
        if result.best_bsr >= config.bsr_threshold {
            matched_indices.insert(result.cds_idx, true);

            // Candidates for new representatives: BSR in [threshold, threshold+0.1)
            if result.best_bsr < config.bsr_threshold + 0.1 {
                if let Some((_, protein)) = remaining.iter().find(|(idx, _)| *idx == result.cds_idx) {
                    let self_score = sw::self_score(protein) as f64;
                    new_reps.push(Representative {
                        locus_idx: result.best_locus,
                        seq_id: format!("repdet_{}", result.cds_idx),
                        protein_seq: protein.clone(),
                        dna_length: 0,
                        self_score,
                    });
                }
            }
        }
    }

    // Collect matched results (need to clone since we borrow results)
    for result in results {
        if result.best_bsr >= config.bsr_threshold {
            all_results.push(cluster::ClusterResult {
                cds_idx: result.cds_idx,
                best_locus: result.best_locus,
                best_bsr: result.best_bsr,
                score: result.score,
                query_start: result.query_start,
                query_end: result.query_end,
                query_len: result.query_len,
                target_start: result.target_start,
                target_end: result.target_end,
                target_len: result.target_len,
            });
        }
    }

    // Remove matched from remaining
    remaining.retain(|(idx, _)| !matched_indices.contains_key(idx));

    new_reps
}
