//! Minimizer-based protein clustering for inexact allele matching.
//!
//! Groups unclassified CDS proteins with schema representatives based on
//! shared minimizer content. Within each cluster, Smith-Waterman alignment
//! is performed to compute BSR.

use rustc_hash::{FxHashMap, FxHashSet};

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

/// Find representative candidates based on the shared-minimizer proportion.
pub fn find_clusters_similarity(
    protein: &[u8],
    index: &FxHashMap<Minimizer, Vec<usize>>,
    k: usize,
    w: usize,
    min_similarity: f64,
    max_targets: usize,
) -> Vec<usize> {
    let query_minimizers = extract_minimizers(protein, k, w);
    if query_minimizers.is_empty() {
        return Vec::new();
    }

    let mut counts: FxHashMap<usize, usize> = FxHashMap::default();
    for m in &query_minimizers {
        if let Some(reps) = index.get(m) {
            for &rep_idx in reps {
                *counts.entry(rep_idx).or_default() += 1;
            }
        }
    }

    let denom = query_minimizers.len() as f64;
    let mut result: Vec<(usize, f64)> = counts
        .into_iter()
        .map(|(idx, count)| (idx, count as f64 / denom))
        .filter(|&(_, sim)| sim >= min_similarity)
        .collect();

    result.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

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
    extract_minimizers_hash(seq, k, w)
}

/// Hash-based minimizer extraction (fast mode).
/// Selects the k-mer with minimum FNV-1a hash in each window.
fn extract_minimizers_hash(seq: &[u8], k: usize, w: usize) -> Vec<Minimizer> {
    if seq.len() < k {
        return Vec::new();
    }

    let num_kmers = seq.len() - k + 1;
    if num_kmers == 0 {
        return Vec::new();
    }

    // Compute k-mer hashes
    let kmer_hashes: Vec<u64> = (0..num_kmers).map(|i| hash_kmer(&seq[i..i + k])).collect();

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

/// Lexicographic minimizer extraction (compatible mode).
/// Replicates Python chewBBACA's `determine_minimizers` from iterables_manipulation.py,
/// including its window-skipping heuristic. When the minimizer is NOT at position 0
/// in the window, Python skips forward by `minimizer_idx` positions instead of sliding
/// by 1. It also has a recovery step that checks skipped k-mers.
fn extract_minimizers_compat(seq: &[u8], k: usize, w: usize) -> Vec<Minimizer> {
    if seq.len() < k {
        return Vec::new();
    }

    let num_kmers = seq.len() - k + 1;
    if num_kmers == 0 {
        return Vec::new();
    }

    if num_kmers < w {
        // Python's determine_minimizers returns empty list when there are
        // fewer k-mers than the window size (last_window = len(kmers) - w < 0,
        // so the while loop never executes).
        return Vec::new();
    }

    // Replicate Python's window-skipping heuristic exactly
    let last_window = num_kmers - w; // 0-based index of last window start
    let mut i = 0usize;
    let mut previous: Option<&[u8]> = None;
    let mut sell = false;
    let mut minimizers_raw: Vec<&[u8]> = Vec::new();

    while i <= last_window {
        // Get k-mers in current window [i .. i+w)
        // Find the lexicographically smallest k-mer and its position in the window
        let mut min_idx_in_window = 0usize;
        let mut min_kmer = &seq[i..i + k];
        for j in 1..w {
            let kmer = &seq[i + j..i + j + k];
            if kmer < min_kmer {
                min_kmer = kmer;
                min_idx_in_window = j;
            }
        }

        if previous.is_none() {
            // First window: simply store the minimizer
            minimizers_raw.push(min_kmer);
        } else {
            let prev = previous.unwrap();
            // Check if minimizer is different from previous
            if min_kmer != prev {
                // Recovery: check k-mers between positions 1..minimizer_idx
                // (the "skipped" portion of the window) for anything < previous
                let mut minimal = prev;
                for j in 1..min_idx_in_window {
                    let kmer = &seq[i + j..i + j + k];
                    if kmer < minimal {
                        minimizers_raw.push(kmer);
                        minimal = kmer;
                    }
                }
                // Always add the current window's minimizer
                minimizers_raw.push(min_kmer);
            }
            // If minimizer == previous, skip (don't store duplicate)
        }

        if min_idx_in_window == 0 {
            // Minimizer at position 0: slide by 1
            i += 1;
            previous = None;
        } else {
            // Skip forward by minimizer_idx
            i += min_idx_in_window;
            // Handle last-window boundary
            if i > last_window && !sell {
                i = last_window;
                sell = true;
            }
            previous = Some(min_kmer);
        }
    }

    // Deduplicate and hash (Python uses set(minimizers))
    let mut seen: Vec<&[u8]> = Vec::new();
    let mut minimizers = Vec::new();
    for kmer in minimizers_raw {
        if !seen.contains(&kmer) {
            seen.push(kmer);
            minimizers.push(hash_kmer(kmer));
        }
    }

    minimizers.sort_unstable();
    minimizers.dedup();
    minimizers
}

/// Build a minimizer index using lexicographic comparison (compatible mode).
pub fn build_minimizer_index_compat(
    representatives: &[Representative],
    k: usize,
    w: usize,
) -> FxHashMap<Minimizer, Vec<usize>> {
    let mut index: FxHashMap<Minimizer, Vec<usize>> = FxHashMap::default();

    for (idx, rep) in representatives.iter().enumerate() {
        let minimizers = extract_minimizers_compat(&rep.protein_seq, k, w);
        for m in minimizers {
            index.entry(m).or_default().push(idx);
        }
    }

    for list in index.values_mut() {
        list.sort_unstable();
        list.dedup();
    }

    index
}

/// Find representative candidates using lexicographic minimizers (compatible mode).
pub fn find_clusters_similarity_compat(
    protein: &[u8],
    index: &FxHashMap<Minimizer, Vec<usize>>,
    k: usize,
    w: usize,
    min_similarity: f64,
    max_targets: usize,
) -> Vec<usize> {
    let query_minimizers = extract_minimizers_compat(protein, k, w);
    if query_minimizers.is_empty() {
        return Vec::new();
    }

    // Python uses distinct minimizers (set) for both counting and denominator
    let distinct_minimizers: FxHashSet<Minimizer> = query_minimizers.iter().copied().collect();

    let mut counts: FxHashMap<usize, usize> = FxHashMap::default();
    for m in &distinct_minimizers {
        if let Some(reps) = index.get(m) {
            for &rep_idx in reps {
                *counts.entry(rep_idx).or_default() += 1;
            }
        }
    }

    let denom = distinct_minimizers.len() as f64;
    let mut result: Vec<(usize, f64)> = counts
        .into_iter()
        .map(|(idx, count)| (idx, count as f64 / denom))
        .filter(|&(_, sim)| sim >= min_similarity)
        .collect();

    result.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

    if max_targets > 0 && result.len() > max_targets {
        result.truncate(max_targets);
    }

    let mut indices: Vec<usize> = result.into_iter().map(|(idx, _)| idx).collect();
    indices.sort_unstable();
    indices
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
#[derive(Debug, Clone)]
pub struct ClusterResult {
    pub cds_idx: usize,
    pub representative_idx: usize,
    pub best_locus: LocusIdx,
    pub best_bsr: f64,
    pub score: i32,
    pub rep_dna_len: u32,
    pub query_start: u32,
    pub query_end: u32,
    pub query_len: u32,
    pub target_start: u32,
    pub target_end: u32,
    pub target_len: u32,
}

fn is_better_global_hit(
    candidate_bsr: f64,
    candidate_score: i32,
    candidate_rep: usize,
    best_bsr: f64,
    best_score: i32,
    best_rep: usize,
) -> bool {
    if candidate_bsr > best_bsr {
        return true;
    }
    if candidate_bsr < best_bsr {
        return false;
    }
    if candidate_score > best_score {
        return true;
    }
    if candidate_score < best_score {
        return false;
    }
    candidate_rep < best_rep
}

fn make_result(
    cds_idx: usize,
    representative_idx: usize,
    protein: &[u8],
    representative: &Representative,
    score: i32,
    bsr: f64,
    target_start: u32,
    target_end: u32,
) -> ClusterResult {
    ClusterResult {
        cds_idx,
        representative_idx,
        best_locus: representative.locus_idx,
        best_bsr: bsr,
        score,
        rep_dna_len: representative.dna_length,
        query_start: 0,
        query_end: 0,
        query_len: protein.len() as u32,
        target_start,
        target_end,
        target_len: representative.protein_seq.len() as u32,
    }
}

/// Perform clustering and alignment for a batch of unclassified CDS proteins.
///
/// For each unclassified CDS:
/// 1. Find matching representatives via minimizer index
/// 2. Run SW alignment against matching representatives
/// 3. Compute BSR = score / self_score
/// 4. Return best match per CDS
pub fn cluster_and_align(
    proteins: &[(usize, Vec<u8>)], // (cds_idx, protein_seq)
    representatives: &[Representative],
    index: &FxHashMap<Minimizer, Vec<usize>>,
    k: usize,
    w: usize,
    min_shared: usize,
) -> Vec<ClusterResult> {
    use crate::parasail_ffi;
    use rayon::prelude::*;

    proteins
        .par_iter()
        .filter_map(|(cds_idx, protein)| {
            let clusters = find_clusters(protein, index, k, w, min_shared, 10);
            if clusters.is_empty() {
                return None;
            }

            // Fast score + end positions with parasail SIMD for all candidates
            let mut best_score = 0i32;
            let mut best_bsr = 0.0f64;
            let mut best_rep = 0usize;

            for &rep_idx in &clusters {
                let (score, _, _) =
                    parasail_ffi::sw_simd(protein, &representatives[rep_idx].protein_seq);
                if score <= 0 {
                    continue;
                }
                let self_score = representatives[rep_idx].self_score;
                if self_score <= 0.0 {
                    continue;
                }
                let bsr = score as f64 / self_score;
                if is_better_global_hit(bsr, score, rep_idx, best_bsr, best_score, best_rep) {
                    best_score = score;
                    best_bsr = bsr;
                    best_rep = rep_idx;
                }
            }

            if best_score <= 0 {
                return None;
            }

            // Get full positions (including target_start) for the best match only
            // This costs one extra SIMD reverse-alignment per protein
            let (_, _, _, target_start, target_end) =
                parasail_ffi::sw_simd_full(protein, &representatives[best_rep].protein_seq);

            Some(make_result(
                *cds_idx,
                best_rep,
                protein,
                &representatives[best_rep],
                best_score,
                best_bsr,
                target_start,
                target_end,
            ))
        })
        .collect()
}

fn collect_best_hits_per_locus(
    protein: &[u8],
    representatives: &[Representative],
    clusters: &[usize],
) -> Vec<(LocusIdx, usize, i32, f64)> {
    use crate::parasail_ffi;

    // Try ALL representatives per locus and keep the one with highest BSR.
    // Previous code took the first rep per locus (by index), which could miss
    // a much better-scoring representative of the same locus.
    let mut best_per_locus: FxHashMap<LocusIdx, (usize, i32, f64)> = FxHashMap::default();

    for &rep_idx in clusters {
        let locus_idx = representatives[rep_idx].locus_idx;

        let (score, _, _) = parasail_ffi::sw_simd(protein, &representatives[rep_idx].protein_seq);
        if score <= 0 {
            continue;
        }
        let self_score = representatives[rep_idx].self_score;
        if self_score <= 0.0 {
            continue;
        }
        let bsr = score as f64 / self_score;

        match best_per_locus.entry(locus_idx) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                if bsr > e.get().2 {
                    e.insert((rep_idx, score, bsr));
                }
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert((rep_idx, score, bsr));
            }
        }
    }

    let mut per_locus: Vec<(LocusIdx, usize, i32, f64)> = best_per_locus
        .into_iter()
        .map(|(locus_idx, (rep_idx, score, bsr))| (locus_idx, rep_idx, score, bsr))
        .collect();
    per_locus.sort_unstable_by_key(|(locus_idx, _, _, _)| *locus_idx);
    per_locus
}

fn cluster_and_align_multi_inner<F>(
    proteins: &[(usize, Vec<u8>)],
    representatives: &[Representative],
    find_candidates: F,
) -> Vec<ClusterResult>
where
    F: Fn(&[u8]) -> Vec<usize> + Sync,
{
    cluster_and_align_multi_inner_bsr(proteins, representatives, find_candidates, 0.0)
}

fn cluster_and_align_multi_inner_bsr<F>(
    proteins: &[(usize, Vec<u8>)],
    representatives: &[Representative],
    find_candidates: F,
    min_bsr: f64,
) -> Vec<ClusterResult>
where
    F: Fn(&[u8]) -> Vec<usize> + Sync,
{
    use crate::parasail_ffi;
    use rayon::prelude::*;

    proteins
        .par_iter()
        .flat_map_iter(|(cds_idx, protein)| {
            let clusters = find_candidates(protein);
            if clusters.is_empty() {
                return Vec::new().into_iter();
            }

            let best_hits = collect_best_hits_per_locus(protein, representatives, &clusters);
            let mut hits = Vec::with_capacity(best_hits.len());

            for (_locus_idx, rep_idx, score, bsr) in best_hits {
                // Skip expensive sw_simd_full for hits below BSR threshold
                if bsr < min_bsr {
                    continue;
                }
                let (_, _, _, target_start, target_end) =
                    parasail_ffi::sw_simd_full(protein, &representatives[rep_idx].protein_seq);
                hits.push(make_result(
                    *cds_idx,
                    rep_idx,
                    protein,
                    &representatives[rep_idx],
                    score,
                    bsr,
                    target_start,
                    target_end,
                ));
            }

            hits.into_iter()
        })
        .collect()
}

/// Return the best hit for each candidate locus using a shared-count filter.
pub fn cluster_and_align_multi_limited(
    proteins: &[(usize, Vec<u8>)],
    representatives: &[Representative],
    index: &FxHashMap<Minimizer, Vec<usize>>,
    k: usize,
    w: usize,
    min_shared: usize,
    max_targets: usize,
) -> Vec<ClusterResult> {
    cluster_and_align_multi_inner(proteins, representatives, |protein| {
        find_clusters(protein, index, k, w, min_shared, max_targets)
    })
}

/// Like `cluster_and_align_multi_limited` but skips hits below `min_bsr`,
/// avoiding expensive full-position alignment for low-scoring hits.
pub fn cluster_and_align_multi_limited_bsr(
    proteins: &[(usize, Vec<u8>)],
    representatives: &[Representative],
    index: &FxHashMap<Minimizer, Vec<usize>>,
    k: usize,
    w: usize,
    min_shared: usize,
    max_targets: usize,
    min_bsr: f64,
) -> Vec<ClusterResult> {
    cluster_and_align_multi_inner_bsr(
        proteins,
        representatives,
        |protein| find_clusters(protein, index, k, w, min_shared, max_targets),
        min_bsr,
    )
}

/// Brute-force residual safety net: align every query against ALL representatives,
/// bypassing the minimizer pre-filter entirely. Per locus, the best-scoring
/// representative is kept (same as the filtered path). Eliminates filter-induced
/// misses at the cost of an exhaustive per-query scan.
pub fn cluster_and_align_multi_brute(
    proteins: &[(usize, Vec<u8>)],
    representatives: &[Representative],
) -> Vec<ClusterResult> {
    let all: Vec<usize> = (0..representatives.len()).collect();
    cluster_and_align_multi_inner(proteins, representatives, |_protein| all.clone())
}

/// Like `cluster_and_align_multi_brute` but skips hits below `min_bsr`
/// (used by the representative-determination stage).
pub fn cluster_and_align_multi_brute_bsr(
    proteins: &[(usize, Vec<u8>)],
    representatives: &[Representative],
    min_bsr: f64,
) -> Vec<ClusterResult> {
    let all: Vec<usize> = (0..representatives.len()).collect();
    cluster_and_align_multi_inner_bsr(proteins, representatives, |_protein| all.clone(), min_bsr)
}

/// Return the best hit for each candidate locus using a similarity filter.
pub fn cluster_and_align_multi_similarity(
    proteins: &[(usize, Vec<u8>)],
    representatives: &[Representative],
    index: &FxHashMap<Minimizer, Vec<usize>>,
    k: usize,
    w: usize,
    min_similarity: f64,
    max_targets: usize,
) -> Vec<ClusterResult> {
    cluster_and_align_multi_inner(proteins, representatives, |protein| {
        find_clusters_similarity(protein, index, k, w, min_similarity, max_targets)
    })
}

/// Like `cluster_and_align_multi_similarity` but uses lexicographic (compat)
/// minimizer ordering for candidate selection. The `index` MUST have been built
/// with `build_minimizer_index_compat` so extraction and lookup agree.
pub fn cluster_and_align_multi_similarity_compat(
    proteins: &[(usize, Vec<u8>)],
    representatives: &[Representative],
    index: &FxHashMap<Minimizer, Vec<usize>>,
    k: usize,
    w: usize,
    min_similarity: f64,
    max_targets: usize,
) -> Vec<ClusterResult> {
    cluster_and_align_multi_inner(proteins, representatives, |protein| {
        find_clusters_similarity_compat(protein, index, k, w, min_similarity, max_targets)
    })
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

    eprintln!(
        "  GPU: {} alignment pairs from {} proteins",
        pair_protein_idx.len(),
        proteins.len()
    );

    let query_slices: Vec<&[u8]> = proteins.iter().map(|(_, p)| p.as_slice()).collect();
    let target_slices: Vec<&[u8]> = representatives
        .iter()
        .map(|r| r.protein_seq.as_slice())
        .collect();

    let gpu_results = aligner
        .align_indexed(
            &query_slices,
            &target_slices,
            pair_protein_idx,
            pair_rep_idx,
        )
        .expect("GPU alignment failed");

    // Find best per protein: (bsr, score, rep_i)
    let mut best_per_protein: FxHashMap<usize, (f64, i32, usize)> = FxHashMap::default();

    for (pair_i, res) in gpu_results.iter().enumerate() {
        let prot_i = pair_protein_idx[pair_i];
        let rep_i = pair_rep_idx[pair_i];
        let score = res.score;
        if score <= 0 {
            continue;
        }
        let self_score = representatives[rep_i].self_score;
        if self_score <= 0.0 {
            continue;
        }
        let bsr = score as f64 / self_score;

        let entry = best_per_protein
            .entry(prot_i)
            .or_insert((0.0, 0, usize::MAX));
        if is_better_global_hit(bsr, score, rep_i, entry.0, entry.1, entry.2) {
            *entry = (bsr, score, rep_i);
        }
    }

    // For each best match, get full positions via parasail reverse alignment
    let mut results = Vec::new();
    for (prot_i, (bsr, score, rep_i)) in best_per_protein {
        if score <= 0 {
            continue;
        }
        let (cds_idx, protein) = &proteins[prot_i];

        // Get full positions (target_start/end) via parasail for best match only
        let (_, _, _, target_start, target_end) =
            crate::parasail_ffi::sw_simd_full(protein, &representatives[rep_i].protein_seq);

        results.push(make_result(
            *cds_idx,
            rep_i,
            protein,
            &representatives[rep_i],
            score,
            bsr,
            target_start,
            target_end,
        ));
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

    eprintln!(
        "  GPU: {} alignment pairs from {} proteins",
        pair_protein_idx.len(),
        proteins.len()
    );

    // Phase 2: Batch SW on GPU — scores only, to find best match per protein
    let query_slices: Vec<&[u8]> = proteins.iter().map(|(_, p)| p.as_slice()).collect();
    let target_slices: Vec<&[u8]> = representatives
        .iter()
        .map(|r| r.protein_seq.as_slice())
        .collect();

    let gpu_results = aligner
        .align_indexed(
            &query_slices,
            &target_slices,
            &pair_protein_idx,
            &pair_rep_idx,
        )
        .expect("GPU alignment failed");

    // Phase 3: Find best per protein: (bsr, score, rep_i)
    let mut best_per_protein: FxHashMap<usize, (f64, i32, usize)> = FxHashMap::default();

    for (pair_i, res) in gpu_results.iter().enumerate() {
        let prot_i = pair_protein_idx[pair_i];
        let rep_i = pair_rep_idx[pair_i];
        let score = res.score;
        if score <= 0 {
            continue;
        }
        let self_score = representatives[rep_i].self_score;
        if self_score <= 0.0 {
            continue;
        }
        let bsr = score as f64 / self_score;

        let entry = best_per_protein
            .entry(prot_i)
            .or_insert((0.0, 0, usize::MAX));
        if is_better_global_hit(bsr, score, rep_i, entry.0, entry.1, entry.2) {
            *entry = (bsr, score, rep_i);
        }
    }

    // Phase 4: Get exact target positions via parasail (CPU) for best match only
    let mut results = Vec::new();
    for (prot_i, (bsr, score, rep_i)) in best_per_protein {
        if score <= 0 {
            continue;
        }
        let (cds_idx, protein) = &proteins[prot_i];

        let (_, _, _, target_start, target_end) =
            crate::parasail_ffi::sw_simd_full(protein, &representatives[rep_i].protein_seq);

        results.push(make_result(
            *cds_idx,
            rep_i,
            protein,
            &representatives[rep_i],
            score,
            bsr,
            target_start,
            target_end,
        ));
    }

    results
}

#[cfg(test)]
mod tests {
    use super::is_better_global_hit;

    #[test]
    fn prefers_higher_bsr_over_higher_raw_score() {
        assert!(is_better_global_hit(0.61, 90, 2, 0.60, 100, 1));
        assert!(!is_better_global_hit(0.60, 101, 2, 0.61, 90, 1));
    }

    #[test]
    fn prefers_lower_rep_idx_on_exact_tie() {
        assert!(is_better_global_hit(0.60, 100, 2, 0.60, 100, 3));
        assert!(!is_better_global_hit(0.60, 100, 3, 0.60, 100, 2));
    }
}
