//! Output generation: write results_alleles.tsv, results_statistics.tsv, etc.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::types::*;

/// Write the allelic profile matrix (results_alleles.tsv).
///
/// Format:
/// - Header: FILE\tlocus1\tlocus2\t...
/// - Rows: genome_path\tallele_or_class\t...
pub fn write_alleles_tsv(
    output_path: &Path,
    genome_names: &[String],
    locus_names: &[String],
    // results[genome_idx][locus_idx]
    results: &[Vec<LocusResult>],
    inferred: &rustc_hash::FxHashSet<(u32, u32)>,
) -> std::io::Result<()> {
    let file = File::create(output_path)?;
    let mut w = BufWriter::new(file);

    // Header
    write!(w, "FILE")?;
    for locus in locus_names {
        write!(w, "\t{}", locus)?;
    }
    writeln!(w)?;

    // Rows
    for (genome_idx, genome_name) in genome_names.iter().enumerate() {
        write!(w, "{}", genome_name)?;

        for locus_idx in 0..locus_names.len() {
            let result = &results[genome_idx][locus_idx];
            let cell = format_allele_cell(result, locus_idx as u32, inferred);
            write!(w, "\t{}", cell)?;
        }
        writeln!(w)?;
    }

    Ok(())
}

/// Format one cell of the allelic matrix.
/// `inferred` tracks alleles that were inferred (had `*` prefix in schema);
/// novel alleles created during this run are also marked with `*`.
fn format_allele_cell(
    result: &LocusResult,
    locus_idx: u32,
    inferred: &rustc_hash::FxHashSet<(u32, u32)>,
) -> String {
    match result.class {
        Classification::EXC => {
            if let Some(id) = result.allele_id {
                // Only mark with '*' if the allele was already inferred in the schema
                // (had '*' prefix in the schema FASTA header).
                // Novel alleles discovered during this run are NOT marked with '*'
                // (matching Python chewBBACA behavior for non-Chewie-NS schemas).
                let is_inferred = inferred.contains(&(locus_idx, id));
                if is_inferred {
                    format!("*{}", id)
                } else {
                    format!("{}", id)
                }
            } else {
                "EXC".to_string()
            }
        }
        Classification::INF => {
            if let Some(id) = result.allele_id {
                format!("INF-{}", id)
            } else {
                "INF".to_string()
            }
        }
        other => other.as_str().to_string(),
    }
}

/// Write results_statistics.tsv.
///
/// Counts of each classification per genome.
pub fn write_statistics_tsv(
    output_path: &Path,
    genome_names: &[String],
    _locus_names: &[String],
    results: &[Vec<LocusResult>],
) -> std::io::Result<()> {
    let file = File::create(output_path)?;
    let mut w = BufWriter::new(file);

    let classes = [
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
    ];

    // Header
    write!(w, "FILE")?;
    for cls in &classes {
        write!(w, "\t{}", cls.as_str())?;
    }
    writeln!(w)?;

    // Rows
    for (genome_idx, genome_name) in genome_names.iter().enumerate() {
        write!(w, "{}", genome_name)?;

        for cls in &classes {
            let count = results[genome_idx]
                .iter()
                .filter(|r| r.class == *cls)
                .count();
            write!(w, "\t{}", count)?;
        }
        writeln!(w)?;
    }

    Ok(())
}

/// Write loci_summary_stats.tsv.
///
/// Counts of each classification per locus.
pub fn write_loci_summary(
    output_path: &Path,
    locus_names: &[String],
    results: &[Vec<LocusResult>],
) -> std::io::Result<()> {
    let file = File::create(output_path)?;
    let mut w = BufWriter::new(file);

    let classes = [
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
    ];

    // Header
    write!(w, "Locus")?;
    for cls in &classes {
        write!(w, "\t{}", cls.as_str())?;
    }
    writeln!(w)?;

    let num_genomes = results.len();

    for (locus_idx, locus_name) in locus_names.iter().enumerate() {
        write!(w, "{}", locus_name)?;

        for cls in &classes {
            let count = (0..num_genomes)
                .filter(|&g| results[g][locus_idx].class == *cls)
                .count();
            write!(w, "\t{}", count)?;
        }
        writeln!(w)?;
    }

    Ok(())
}

/// Write novel alleles FASTA.
pub fn write_novel_alleles(
    output_path: &Path,
    novel_alleles: &[(String, Vec<u8>)], // (header, dna_seq)
) -> std::io::Result<()> {
    let file = File::create(output_path)?;
    let mut w = BufWriter::new(file);

    for (header, seq) in novel_alleles {
        writeln!(w, ">{}", header)?;
        // Write sequence in 80-char lines
        for chunk in seq.chunks(80) {
            w.write_all(chunk)?;
            writeln!(w)?;
        }
    }

    Ok(())
}

/// Write results_contigsInfo.tsv (CDS coordinates for valid matches).
pub fn write_contigs_info(output_path: &Path, contigs_info: &[ContigInfo]) -> std::io::Result<()> {
    let file = File::create(output_path)?;
    let mut w = BufWriter::new(file);

    writeln!(
        w,
        "Genome\tContig\tLocus\tStart\tStop\tStrand\tCDS_Length\tClassification"
    )?;

    for info in contigs_info {
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            info.genome,
            info.contig,
            info.locus,
            info.start,
            info.stop,
            info.strand,
            info.cds_length,
            info.class.as_str()
        )?;
    }

    Ok(())
}

/// Write CRC32-hashed allelic profile (results_alleles_hashed.tsv).
/// Each allele ID is replaced with CRC32 hash of its DNA sequence.
/// Non-allele classes (LNF, NIPH, etc.) are kept as-is.
pub fn write_alleles_hashed_tsv(
    output_path: &Path,
    genome_names: &[String],
    locus_names: &[String],
    results: &[Vec<LocusResult>],
    allele_crc32: &rustc_hash::FxHashMap<(u32, u32), u32>,
) -> std::io::Result<()> {
    let file = File::create(output_path)?;
    let mut w = BufWriter::new(file);

    // Header
    write!(w, "FILE")?;
    for locus in locus_names {
        write!(w, "\t{}", locus)?;
    }
    writeln!(w)?;

    // Rows
    for (genome_idx, genome_name) in genome_names.iter().enumerate() {
        write!(w, "{}", genome_name)?;

        for locus_idx in 0..locus_names.len() {
            let result = &results[genome_idx][locus_idx];
            let cell = format_hashed_cell(result, locus_idx as u32, allele_crc32);
            write!(w, "\t{}", cell)?;
        }
        writeln!(w)?;
    }

    Ok(())
}

/// Format a hashed cell: if allele has a CRC32 from DNA sequence, use it.
/// For class labels (LNF, NIPH, etc.), output "-" (matching Python chewBBACA).
fn format_hashed_cell(
    result: &LocusResult,
    locus_idx: u32,
    allele_crc32: &rustc_hash::FxHashMap<(u32, u32), u32>,
) -> String {
    match result.class {
        Classification::EXC | Classification::INF => {
            if let Some(id) = result.allele_id {
                if let Some(&crc) = allele_crc32.get(&(locus_idx, id)) {
                    format!("{}", crc)
                } else {
                    // Fallback: just output the ID
                    format!("{}", id)
                }
            } else {
                "-".to_string()
            }
        }
        _ => "-".to_string(),
    }
}

/// Append novel alleles to the schema FASTA files in place.
///
/// For each novel allele, finds the corresponding locus FASTA file and appends
/// the DNA sequence with a header `>locus_name_*N` (where N is the next allele ID,
/// `*` prefix indicates inferred allele, matching chewBBACA convention).
pub fn update_schema_fasta(
    schema_dir: &Path,
    novel_alleles: &[(String, Vec<u8>)], // (header, dna_seq) — header is "locus_cdsid"
    loci_list: &[String],
) -> std::io::Result<()> {
    use std::collections::HashMap;

    if novel_alleles.is_empty() {
        return Ok(());
    }

    // Group novel alleles by locus name
    let mut by_locus: HashMap<&str, Vec<&Vec<u8>>> = HashMap::new();
    for (header, seq) in novel_alleles {
        // Header format: "locus_name_cdsid" — find which locus matches
        for locus_name in loci_list {
            if header.starts_with(locus_name)
                && header.as_bytes().get(locus_name.len()) == Some(&b'_')
            {
                by_locus.entry(locus_name.as_str()).or_default().push(seq);
                break;
            }
        }
    }

    // For each locus, find the max allele ID and append
    for (locus_name, seqs) in &by_locus {
        let fasta_path = schema_dir.join(format!("{}.fasta", locus_name));
        if !fasta_path.exists() {
            eprintln!(
                "[Schema] Warning: {} not found, skipping",
                fasta_path.display()
            );
            continue;
        }

        // Find max allele ID in existing FASTA
        let mut max_id: u32 = 0;
        let content = std::fs::read_to_string(&fasta_path)?;
        for line in content.lines() {
            if let Some(header) = line.strip_prefix('>') {
                // Header format: "locus_name_N" or "locus_name_*N"
                let suffix = header
                    .strip_prefix(locus_name)
                    .and_then(|s| s.strip_prefix('_'));
                if let Some(id_str) = suffix {
                    let id_str = id_str.trim_start_matches('*');
                    if let Ok(id) = id_str.parse::<u32>() {
                        max_id = max_id.max(id);
                    }
                }
            }
        }

        // Append novel alleles
        let file = std::fs::OpenOptions::new().append(true).open(&fasta_path)?;
        let mut w = BufWriter::new(file);
        let mut next_id = max_id + 1;
        for seq in seqs {
            writeln!(w, ">{}_{}", locus_name, next_id)?;
            for chunk in seq.chunks(80) {
                w.write_all(chunk)?;
                writeln!(w)?;
            }
            next_id += 1;
        }
    }

    Ok(())
}

/// Info for one CDS-locus match (for contigsInfo output).
#[derive(Debug)]
pub struct ContigInfo {
    pub genome: String,
    pub contig: String,
    pub locus: String,
    pub start: u32,
    pub stop: u32,
    pub strand: i8,
    pub cds_length: u32,
    pub class: Classification,
}
