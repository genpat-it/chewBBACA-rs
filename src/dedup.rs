//! CDS deduplication: group identical DNA sequences by SHA256 hash.

use rustc_hash::FxHashMap;

use crate::types::*;
use crate::schema::sha256;

/// Deduplicate a list of CDSs by DNA sequence hash.
/// Returns:
/// - distinct_cds: one CDS per unique sequence (the representative)
/// - hash_to_genomes: mapping from DNA hash → list of (genome_idx, cds_id)
pub fn deduplicate_cds(
    all_cds: &[Cds],
) -> (Vec<&Cds>, FxHashMap<SeqHash, Vec<(GenomeIdx, String)>>) {
    let mut hash_to_genomes: FxHashMap<SeqHash, Vec<(GenomeIdx, String)>> = FxHashMap::default();
    let mut seen: FxHashMap<SeqHash, usize> = FxHashMap::default(); // hash -> index in distinct_cds
    let mut distinct_cds: Vec<&Cds> = Vec::new();

    for cds in all_cds {
        let upper: Vec<u8> = cds.dna_seq.iter().map(|b| b.to_ascii_uppercase()).collect();
        let hash = sha256(&upper);

        hash_to_genomes
            .entry(hash)
            .or_default()
            .push((cds.genome_idx, cds.id.clone()));

        if !seen.contains_key(&hash) {
            seen.insert(hash, distinct_cds.len());
            distinct_cds.push(cds);
        }
    }

    (distinct_cds, hash_to_genomes)
}
