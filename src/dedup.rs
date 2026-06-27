//! CDS deduplication: group identical DNA sequences by SHA256 hash.

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::schema::sha256;
use crate::types::*;

/// Per-genome info stored alongside each CDS occurrence in hash_to_genomes.
/// The CDS id is intentionally not stored: it is never read downstream, and
/// dropping it saves one String allocation per CDS occurrence (millions).
pub type CdsGenomeEntry = (GenomeIdx, Option<CdsCoord>);

/// Deduplicate a list of CDSs by DNA sequence hash.
/// Returns:
/// - distinct_cds: one CDS per unique sequence (the representative)
/// - hash_to_genomes: mapping from DNA hash → list of (genome_idx, cds_id, coord)
/// - all_hashes: pre-computed SHA-256 hash for each CDS (parallel)
pub fn deduplicate_cds(
    all_cds: &[Cds],
) -> (
    Vec<&Cds>,
    FxHashMap<SeqHash, Vec<CdsGenomeEntry>>,
    Vec<SeqHash>,
) {
    // Step 1: Compute hashes in parallel
    let all_hashes: Vec<SeqHash> = all_cds
        .par_iter()
        .map(|cds| {
            let upper: Vec<u8> = cds.dna_seq.iter().map(|b| b.to_ascii_uppercase()).collect();
            sha256(&upper)
        })
        .collect();

    // Step 2: Build maps sequentially (HashMap is not thread-safe)
    let mut hash_to_genomes: FxHashMap<SeqHash, Vec<CdsGenomeEntry>> = FxHashMap::default();
    let mut seen: FxHashMap<SeqHash, usize> = FxHashMap::default();
    let mut distinct_cds: Vec<&Cds> = Vec::new();

    for (i, cds) in all_cds.iter().enumerate() {
        let hash = all_hashes[i];

        hash_to_genomes
            .entry(hash)
            .or_default()
            .push((cds.genome_idx, cds.coord.clone()));

        if !seen.contains_key(&hash) {
            seen.insert(hash, distinct_cds.len());
            distinct_cds.push(cds);
        }
    }

    (distinct_cds, hash_to_genomes, all_hashes)
}
