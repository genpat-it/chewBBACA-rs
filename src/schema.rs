//! Schema reading: load locus FASTA files, compute allele hashes and modes.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use sha2::{Sha256, Digest};
use rustc_hash::FxHashMap;
use rayon::prelude::*;

use crate::translate;
use crate::types::*;

/// Per-locus data collected in parallel.
struct LocusData {
    locus: Locus,
    representative: Representative,
    dna_entries: Vec<(SeqHash, LocusIdx, AlleleId)>,
    protein_entries: Vec<(SeqHash, LocusIdx, AlleleId)>,
    crc32_entries: Vec<((LocusIdx, AlleleId), u32)>,
}

/// Load a schema from a directory (parallelized with rayon).
pub fn load_schema(
    schema_dir: &Path,
    loci_list: &[String],
    translation_table: u8,
) -> Schema {
    let short_dir = schema_dir.join("short");

    // Process each locus in parallel
    let locus_data: Vec<LocusData> = loci_list.par_iter().enumerate().map(|(locus_idx, locus_name)| {
        let locus_idx = locus_idx as LocusIdx;
        let fasta_path = find_locus_fasta(schema_dir, locus_name);
        let short_path = find_locus_short(&short_dir, locus_name);

        let mut allele_lengths: Vec<u32> = Vec::new();
        let mut allele_count = 0u32;
        let mut dna_entries = Vec::new();
        let mut protein_entries = Vec::new();
        let mut crc32_entries = Vec::new();

        if let Ok(mut reader) = needletail::parse_fastx_file(&fasta_path) {
            while let Some(Ok(record)) = reader.next() {
                allele_count += 1;
                let seq = record.seq();
                allele_lengths.push(seq.len() as u32);

                let header = std::str::from_utf8(record.id()).unwrap_or("");
                let allele_id = parse_allele_id(header);

                let dna_upper: Vec<u8> = seq.iter().map(|b| b.to_ascii_uppercase()).collect();
                let dna_hash = sha256(&dna_upper);
                dna_entries.push((dna_hash, locus_idx, allele_id));

                let seq_str = String::from_utf8_lossy(&seq);
                let crc = crc32fast::hash(seq_str.as_bytes());
                crc32_entries.push(((locus_idx, allele_id), crc));

                if let Some(protein) = translate::translate(&dna_upper, translation_table) {
                    let prot_hash = sha256(&protein);
                    protein_entries.push((prot_hash, locus_idx, allele_id));
                }
            }
        }

        let mode_length = compute_mode(&allele_lengths);

        // Read representative allele
        let mut rep_protein = Vec::new();
        let mut rep_dna_len = 0u32;
        let mut rep_id = String::new();

        if let Ok(mut reader) = needletail::parse_fastx_file(&short_path) {
            while let Some(Ok(record)) = reader.next() {
                let seq = record.seq();
                let dna_upper: Vec<u8> = seq.iter().map(|b| b.to_ascii_uppercase()).collect();
                rep_dna_len = dna_upper.len() as u32;
                rep_id = String::from_utf8_lossy(record.id()).to_string();
                if let Some(protein) = translate::translate(&dna_upper, translation_table) {
                    rep_protein = protein;
                }
                break;
            }
        }

        LocusData {
            locus: Locus {
                id: locus_name.clone(),
                fasta_path: fasta_path.to_string_lossy().to_string(),
                short_path: short_path.to_string_lossy().to_string(),
                allele_count,
                mode_length,
                self_score: 0.0,
            },
            representative: Representative {
                locus_idx,
                seq_id: rep_id,
                protein_seq: rep_protein,
                dna_length: rep_dna_len,
                self_score: 0.0,
            },
            dna_entries,
            protein_entries,
            crc32_entries,
        }
    }).collect();

    // Merge results (single-threaded, fast)
    let num_loci = loci_list.len();
    let mut loci = Vec::with_capacity(num_loci);
    let mut representatives = Vec::with_capacity(num_loci);
    let mut dna_hashes: FxHashMap<SeqHash, Vec<(LocusIdx, AlleleId)>> = FxHashMap::default();
    let mut protein_hashes: FxHashMap<SeqHash, Vec<(LocusIdx, AlleleId)>> = FxHashMap::default();
    let mut allele_crc32: FxHashMap<(LocusIdx, AlleleId), u32> = FxHashMap::default();

    for data in locus_data {
        loci.push(data.locus);
        representatives.push(data.representative);
        for (hash, li, ai) in data.dna_entries {
            dna_hashes.entry(hash).or_default().push((li, ai));
        }
        for (hash, li, ai) in data.protein_entries {
            protein_hashes.entry(hash).or_default().push((li, ai));
        }
        for (key, crc) in data.crc32_entries {
            allele_crc32.insert(key, crc);
        }
    }

    Schema {
        loci,
        dna_hashes,
        protein_hashes,
        representatives,
        allele_crc32,
    }
}

/// Schema with all pre-computed data.
pub struct Schema {
    pub loci: Vec<Locus>,
    pub dna_hashes: FxHashMap<SeqHash, Vec<(LocusIdx, AlleleId)>>,
    pub protein_hashes: FxHashMap<SeqHash, Vec<(LocusIdx, AlleleId)>>,
    pub representatives: Vec<Representative>,
    /// CRC32 of DNA sequence per (locus_idx, allele_id), for hashed profile output.
    pub allele_crc32: FxHashMap<(LocusIdx, AlleleId), u32>,
}

fn find_locus_fasta(schema_dir: &Path, locus_name: &str) -> PathBuf {
    let with_ext = schema_dir.join(format!("{}.fasta", locus_name));
    if with_ext.exists() {
        return with_ext;
    }
    let without = schema_dir.join(locus_name);
    if without.exists() {
        return without;
    }
    with_ext
}

fn find_locus_short(short_dir: &Path, locus_name: &str) -> PathBuf {
    let short_name = format!("{}_short.fasta", locus_name);
    let p = short_dir.join(&short_name);
    if p.exists() {
        return p;
    }
    find_locus_fasta(short_dir, locus_name)
}

fn parse_allele_id(header: &str) -> AlleleId {
    header
        .rsplit('_')
        .next()
        .and_then(|s| s.trim_end_matches('*').parse().ok())
        .unwrap_or(0)
}

pub fn sha256(data: &[u8]) -> SeqHash {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}

fn compute_mode(lengths: &[u32]) -> u32 {
    if lengths.is_empty() {
        return 0;
    }
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for &l in lengths {
        *counts.entry(l).or_default() += 1;
    }
    counts.into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(len, _)| len)
        .unwrap_or(0)
}
