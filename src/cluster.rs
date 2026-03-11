//! Minimizer-based protein clustering for inexact allele matching.
//!
//! Groups unclassified CDS proteins with schema representatives based on
//! shared minimizer content. Within each cluster, Smith-Waterman alignment
//! is performed to compute BSR.

use rustc_hash::FxHashMap;

use crate::sw;
use crate::gpu_sw::GpuAligner;
use crate::types::*;

/// A minimizer: hash of a k-mer window.
type Minimizer = u64;

/// Build a minimizer index from representative proteins.
/// Returns: minimizer → list of representative indices.
pub fn build_minimizer_index(
    representatives: &[Representative],
    k: usize,
    w: usize,
) -> FxHashMap<Minimizer, Vec<usize>> {
    let mut index: FxHashMap<Minimizer, Vec<usize>> = FxHashMap::default();

    for (idx, rep) in representatives.iter().enumerate() {
        let minimizers = extract_minimizers(&rep.protein_seq, k, w);
        for m in minimizers {
            index.entry(m).or_default().push(idx);
        }
    }

    // Deduplicate representative lists
    for list in index.values_mut() {
        list.sort_unstable();
        list.dedup();
    }

    index
}

/// Find which representatives a query protein clusters with.
/// Returns representative indices that share enough minimizers.
/// If max_targets > 0, returns only the top max_targets by shared count.
pub fn find_clusters(
    protein: &[u8],
    index: &FxHashMap<Minimizer, Vec<usize>>,
    k: usize,
    w: usize,
    min_shared: usize,
    max_targets: usize,
) -> Vec<usize> {
    let query_minimizers = extract_minimizers(protein, k, w);
    if query_minimizers.is_empty() {
        return Vec::new();
    }

    // Count shared minimizers per representative
    let mut counts: FxHashMap<usize, usize> = FxHashMap::default();
    for m in &query_minimizers {
        if let Some(reps) = index.get(m) {
            for &rep_idx in reps {
                *counts.entry(rep_idx).or_default() += 1;
            }
        }
    }

    // Filter by minimum shared minimizers
    let mut result: Vec<(usize, usize)> = counts
        .into_iter()
        .filter(|&(_, count)| count >= min_shared)
        .collect();

    // Sort by count descending, then by index for stability
    result.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    // Limit to top-K if specified
    if max_targets > 0 && result.len() > max_targets {
        result.truncate(max_targets);
    }

    let mut indices: Vec<usize> = result.into_iter().map(|(idx, _)| idx).collect();
    indices.sort_unstable();
    indices
}

/// Extract minimizers from a protein sequence.
/// Uses canonical (minimum) hash within each window of w consecutive k-mers.
fn extract_minimizers(seq: &[u8], k: usize, w: usize) -> Vec<Minimizer> {
    if seq.len() < k {
        return Vec::new();
    }

    let num_kmers = seq.len() - k + 1;
    if num_kmers == 0 {
        return Vec::new();
    }

    // Compute k-mer hashes
    let kmer_hashes: Vec<u64> = (0..num_kmers)
        .map(|i| hash_kmer(&seq[i..i + k]))
        .collect();

    if kmer_hashes.len() < w {
        // Fewer k-mers than window size: take the minimum
        if let Some(&min_hash) = kmer_hashes.iter().min() {
            return vec![min_hash];
        }
        return Vec::new();
    }

    // Sliding window minimizers
    let mut minimizers = Vec::new();
    let mut prev_min = u64::MAX;

    for window_start in 0..=(kmer_hashes.len() - w) {
        let window = &kmer_hashes[window_start..window_start + w];
        let min_hash = *window.iter().min().unwrap();
        if min_hash != prev_min || window_start == 0 {
            minimizers.push(min_hash);
            prev_min = min_hash;
        }
    }

    minimizers.sort_unstable();
    minimizers.dedup();
    minimizers
}

/// Simple hash for a k-mer (FNV-1a style).
fn hash_kmer(kmer: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in kmer {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Result of clustering + alignment for one CDS.
#[derive(Debug)]
pub struct ClusterResult {
    pub cds_idx: usize,
    pub best_locus: LocusIdx,
    pub best_bsr: f64,
    pub score: i32,
    pub query_start: u32,
    pub query_end: u32,
    pub query_len: u32,
    pub target_start: u32,
    pub target_end: u32,
    pub target_len: u32,
}

/// Perform clustering and alignment for a batch of unclassified CDS proteins.
///
/// For each unclassified CDS:
/// 1. Find matching representatives via minimizer index
/// 2. Run SW alignment against matching representatives
/// 3. Compute BSR = score / self_score
/// 4. Return best match per CDS
pub fn cluster_and_align(
    proteins: &[(usize, Vec<u8>)],  // (cds_idx, protein_seq)
    representatives: &[Representative],
    index: &FxHashMap<Minimizer, Vec<usize>>,
    k: usize,
    w: usize,
    min_shared: usize,
) -> Vec<ClusterResult> {
    use rayon::prelude::*;
    use crate::parasail_ffi;

    proteins.par_iter().filter_map(|(cds_idx, protein)| {
        let clusters = find_clusters(protein, index, k, w, min_shared, 5);
        if clusters.is_empty() {
            return None;
        }

        // Fast score + end positions with parasail SIMD for all candidates
        let mut best_score = 0i32;
        let mut best_rep = 0usize;

        for &rep_idx in &clusters {
            let (score, _, _) = parasail_ffi::sw_simd(protein, &representatives[rep_idx].protein_seq);
            if score > best_score {
                best_score = score;
                best_rep = rep_idx;
            }
        }

        if best_score <= 0 {
            return None;
        }

        let self_score = representatives[best_rep].self_score;
        if self_score <= 0.0 {
            return None;
        }

        let bsr = best_score as f64 / self_score;

        // Get full positions (including target_start) for the best match only
        // This costs one extra SIMD reverse-alignment per protein
        let (_, _, _, target_start, target_end) = parasail_ffi::sw_simd_full(
            protein, &representatives[best_rep].protein_seq
        );

        Some(ClusterResult {
            cds_idx: *cds_idx,
            best_locus: representatives[best_rep].locus_idx,
            best_bsr: bsr,
            score: best_score,
            query_start: 0, // not needed for classification
            query_end: 0,
            query_len: protein.len() as u32,
            target_start,
            target_end,
            target_len: representatives[best_rep].protein_seq.len() as u32,
        })
    }).collect()
}

/// Build alignment pairs via minimizer clustering (CPU-only, no GPU needed).
/// Returns (pair_protein_indices, pair_rep_indices).
pub fn build_alignment_pairs(
    proteins: &[(usize, Vec<u8>)],
    index: &FxHashMap<Minimizer, Vec<usize>>,
    k: usize,
    w: usize,
    min_shared: usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut pair_protein_idx = Vec::new();
    let mut pair_rep_idx = Vec::new();

    for (i, (_cds_idx, protein)) in proteins.iter().enumerate() {
        let clusters = find_clusters(protein, index, k, w, min_shared, 5);
        for &rep_idx in &clusters {
            pair_protein_idx.push(i);
            pair_rep_idx.push(rep_idx);
        }
    }

    (pair_protein_idx, pair_rep_idx)
}

/// GPU-accelerated alignment from pre-built pairs.
pub fn align_pairs_gpu(
    proteins: &[(usize, Vec<u8>)],
    representatives: &[Representative],
    pair_protein_idx: &[usize],
    pair_rep_idx: &[usize],
    aligner: &GpuAligner,
) -> Vec<ClusterResult> {
    if pair_protein_idx.is_empty() {
        return Vec::new();
    }

    eprintln!("  GPU: {} alignment pairs from {} proteins", pair_protein_idx.len(), proteins.len());

    let query_slices: Vec<&[u8]> = proteins.iter().map(|(_, p)| p.as_slice()).collect();
    let target_slices: Vec<&[u8]> = representatives.iter().map(|r| r.protein_seq.as_slice()).collect();

    let gpu_results = aligner.align_indexed(
        &query_slices,
        &target_slices,
        pair_protein_idx,
        pair_rep_idx,
    ).expect("GPU alignment failed");

    // Find best per protein: (score, rep_i)
    let mut best_per_protein: FxHashMap<usize, (i32, usize)> = FxHashMap::default();

    for (pair_i, res) in gpu_results.iter().enumerate() {
        let prot_i = pair_protein_idx[pair_i];
        let rep_i = pair_rep_idx[pair_i];
        let score = res.score;

        let entry = best_per_protein.entry(prot_i).or_insert((0, 0));
        if score > entry.0 {
            *entry = (score, rep_i);
        }
    }

    // For each best match, get full positions via parasail reverse alignment
    let mut results = Vec::new();
    for (prot_i, (score, rep_i)) in best_per_protein {
        if score <= 0 { continue; }
        let self_score = representatives[rep_i].self_score;
        if self_score <= 0.0 { continue; }
        let bsr = score as f64 / self_score;
        let (cds_idx, protein) = &proteins[prot_i];

        // Get full positions (target_start/end) via parasail for best match only
        let (_, _, _, target_start, target_end) = crate::parasail_ffi::sw_simd_full(
            protein, &representatives[rep_i].protein_seq
        );

        results.push(ClusterResult {
            cds_idx: *cds_idx,
            best_locus: representatives[rep_i].locus_idx,
            best_bsr: bsr,
            score,
            query_start: 0,
            query_end: 0,
            query_len: protein.len() as u32,
            target_start,
            target_end,
            target_len: representatives[rep_i].protein_seq.len() as u32,
        });
    }

    results
}

/// GPU-accelerated version: cluster first, then batch all SW pairs to GPU.
/// GPU computes scores to find the best match, then parasail CPU computes
/// exact target positions for the best match only (for PLOT3/PLOT5 classification).
pub fn cluster_and_align_gpu(
    proteins: &[(usize, Vec<u8>)],
    representatives: &[Representative],
    index: &FxHashMap<Minimizer, Vec<usize>>,
    k: usize,
    w: usize,
    min_shared: usize,
    aligner: &GpuAligner,
) -> Vec<ClusterResult> {
    // Phase 1: Minimizer clustering (CPU, fast)
    let mut pair_protein_idx = Vec::new();
    let mut pair_rep_idx = Vec::new();

    for (i, (_cds_idx, protein)) in proteins.iter().enumerate() {
        let clusters = find_clusters(protein, index, k, w, min_shared, 5);
        for &rep_idx in &clusters {
            pair_protein_idx.push(i);
            pair_rep_idx.push(rep_idx);
        }
    }

    if pair_protein_idx.is_empty() {
        return Vec::new();
    }

    eprintln!("  GPU: {} alignment pairs from {} proteins", pair_protein_idx.len(), proteins.len());

    // Phase 2: Batch SW on GPU — scores only, to find best match per protein
    let query_slices: Vec<&[u8]> = proteins.iter().map(|(_, p)| p.as_slice()).collect();
    let target_slices: Vec<&[u8]> = representatives.iter().map(|r| r.protein_seq.as_slice()).collect();

    let gpu_results = aligner.align_indexed(
        &query_slices,
        &target_slices,
        &pair_protein_idx,
        &pair_rep_idx,
    ).expect("GPU alignment failed");

    // Phase 3: Find best per protein: (score, rep_i)
    let mut best_per_protein: FxHashMap<usize, (i32, usize)> = FxHashMap::default();

    for (pair_i, res) in gpu_results.iter().enumerate() {
        let prot_i = pair_protein_idx[pair_i];
        let rep_i = pair_rep_idx[pair_i];
        let score = res.score;

        let entry = best_per_protein.entry(prot_i).or_insert((0, 0));
        if score > entry.0 {
            *entry = (score, rep_i);
        }
    }

    // Phase 4: Get exact target positions via parasail (CPU) for best match only
    let mut results = Vec::new();
    for (prot_i, (score, rep_i)) in best_per_protein {
        if score <= 0 { continue; }
        let self_score = representatives[rep_i].self_score;
        if self_score <= 0.0 { continue; }
        let bsr = score as f64 / self_score;
        let (cds_idx, protein) = &proteins[prot_i];

        let (_, _, _, target_start, target_end) = crate::parasail_ffi::sw_simd_full(
            protein, &representatives[rep_i].protein_seq
        );

        results.push(ClusterResult {
            cds_idx: *cds_idx,
            best_locus: representatives[rep_i].locus_idx,
            best_bsr: bsr,
            score,
            query_start: 0,
            query_end: 0,
            query_len: protein.len() as u32,
            target_start,
            target_end,
            target_len: representatives[rep_i].protein_seq.len() as u32,
        });
    }

    results
}
