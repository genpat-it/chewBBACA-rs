//! Classification logic: assign classes to CDS matches.

use crate::types::*;

/// Classify an inexact match based on position, size, and BSR.
///
/// target_start/target_end are 1-based protein alignment positions on the representative.
/// These determine PLOT3/PLOT5 (how much of the representative was not covered by the alignment).
pub fn classify_inexact(
    bsr: f64,
    bsr_threshold: f64,
    cds_dna_len: u32,
    locus_mode_len: u32,
    size_threshold: f64,
    coord: Option<&CdsCoord>,
    rep_dna_len: u32,
    target_start: u32, // 1-based protein position on representative
    target_end: u32,   // 1-based protein position on representative
    target_len: u32,   // representative protein length
    cds_input: bool,   // true when --cds-input used: skip position classification
) -> Classification {
    if bsr < bsr_threshold {
        return Classification::LNF;
    }

    // 1. Position classification (only for genome assemblies with coordinates)
    // When cds_input is true, chewBBACA skips PLOT3/PLOT5/LOTSC checks entirely.
    if !cds_input {
        if let Some(coord) = coord {
            if let Some(pos_class) =
                position_classification(coord, rep_dna_len, target_start, target_end, target_len)
            {
                return pos_class;
            }
        }
    }

    // 2. Size classification
    if let Some(size_class) = size_classification(cds_dna_len, locus_mode_len, size_threshold) {
        return size_class;
    }

    // 3. If passes all checks → INF (new inferred allele)
    Classification::INF
}

/// Check if the CDS is near a contig boundary (partial match).
///
/// Uses the representative (target) alignment positions to determine how much
/// of the representative was not aligned, then checks if the contig has enough
/// room to accommodate those unaligned portions.
///
/// This matches Python chewBBACA's `contig_position_classification()`:
/// - representative_leftmost_pos → (target_start - 1) * 3 in DNA
/// - representative_rightmost_pos → target_end * 3 in DNA
pub fn position_classification_pub(
    coord: &CdsCoord,
    rep_dna_len: u32,
    target_start: u32,
    target_end: u32,
    target_len: u32,
) -> Option<Classification> {
    position_classification(coord, rep_dna_len, target_start, target_end, target_len)
}

fn position_classification(
    coord: &CdsCoord,
    rep_dna_len: u32,
    target_start: u32, // 1-based protein position on representative
    target_end: u32,   // 1-based protein position on representative
    target_len: u32,   // representative protein length
) -> Option<Classification> {
    if coord.contig_len == 0 {
        return None; // no contig info
    }

    // Check LOTSC: contig shorter than representative
    if coord.contig_len < rep_dna_len {
        return Some(Classification::LOTSC);
    }

    // Convert representative protein alignment positions to DNA positions
    // representative_leftmost_pos = how many DNA bases at the start of rep didn't align
    let rep_leftmost_rest = (target_start.saturating_sub(1)) * 3;
    // representative_rightmost_rest = how many DNA bases at the end of rep didn't align
    let rep_rightmost_rest = if target_len > target_end {
        (target_len - target_end) * 3
    } else {
        0
    };

    // If representative fully aligned, no PLOT possible
    if rep_leftmost_rest == 0 && rep_rightmost_rest == 0 {
        return None;
    }

    // Contig space around the CDS
    // contig_leftmost_rest = CDS start position (space before CDS)
    let contig_leftmost_rest = coord.start;
    // contig_rightmost_rest = space after CDS end
    let contig_rightmost_rest = coord.contig_len.saturating_sub(coord.stop);

    // Check if unaligned portions of representative exceed contig boundaries
    if coord.strand > 0 {
        // Forward strand
        if rep_leftmost_rest > 0 && contig_leftmost_rest < rep_leftmost_rest {
            return Some(Classification::PLOT5);
        }
        if rep_rightmost_rest > 0 && contig_rightmost_rest < rep_rightmost_rest {
            return Some(Classification::PLOT3);
        }
    } else {
        // Reverse strand: directions are swapped
        if rep_leftmost_rest > 0 && contig_rightmost_rest < rep_leftmost_rest {
            return Some(Classification::PLOT5);
        }
        if rep_rightmost_rest > 0 && contig_leftmost_rest < rep_rightmost_rest {
            return Some(Classification::PLOT3);
        }
    }

    None
}

/// Check if CDS length is within acceptable range of locus mode.
fn size_classification(
    cds_dna_len: u32,
    mode_len: u32,
    size_threshold: f64,
) -> Option<Classification> {
    if mode_len == 0 {
        return None;
    }

    let mode_len_f = mode_len as f64;
    let min_len = mode_len_f * (1.0 - size_threshold);
    let max_len = mode_len_f * (1.0 + size_threshold);
    let target_len = cds_dna_len as f64;

    if target_len < min_len {
        Some(Classification::ASM)
    } else if target_len > max_len {
        Some(Classification::ALM)
    } else {
        None
    }
}

/// Resolve multiple matches for the same genome+locus pair.
/// Returns the final classification when there are multiple hits.
/// Matches Python chewBBACA's logic from allele_call.py lines 290-318.
pub fn resolve_multi_match(classes: &[Classification]) -> Classification {
    if classes.len() <= 1 {
        return classes.first().copied().unwrap_or(Classification::LNF);
    }

    // Count each classification type
    let mut exc_count = 0u32;
    let mut inf_count = 0u32;
    let mut has_plot = false;
    let mut distinct_classes = Vec::new();

    for &c in classes {
        match c {
            Classification::EXC => exc_count += 1,
            Classification::INF => inf_count += 1,
            Classification::PLOT3 | Classification::PLOT5 | Classification::LOTSC => {
                has_plot = true
            }
            _ => {}
        }
        if !distinct_classes.contains(&c) {
            distinct_classes.push(c);
        }
    }

    if distinct_classes.len() == 1 {
        // Multiple matches, single class
        if classes[0] == Classification::EXC {
            Classification::NIPHEM
        } else {
            Classification::NIPH
        }
    } else {
        // Multiple matches, multiple classes
        if exc_count > 0 && inf_count > 0 {
            // Both EXC and INF → NIPH
            Classification::NIPH
        } else if has_plot {
            // Any class + PLOT3/PLOT5/LOTSC → NIPH
            Classification::NIPH
        } else if exc_count > 0 || inf_count > 0 {
            // EXC or INF with ASM/ALM
            let match_count = if exc_count > 0 { exc_count } else { inf_count };
            if match_count == 1 {
                // Single EXC/INF + ASM/ALM → keep the EXC/INF
                if exc_count > 0 {
                    Classification::EXC
                } else {
                    Classification::INF
                }
            } else {
                Classification::NIPH
            }
        } else {
            // Multiple ASM/ALM → NIPH
            Classification::NIPH
        }
    }
}

/// Record one match for a genome+locus pair and recompute the final class.
pub fn record_match(result: &mut LocusResult, match_result: MatchResult) {
    result.matches.push(match_result);

    let classes: Vec<_> = result.matches.iter().map(|m| m.class).collect();
    result.class = resolve_multi_match(&classes);

    if matches!(result.class, Classification::EXC | Classification::INF) {
        if let Some(primary) = result.matches.iter().find(|m| m.class == result.class) {
            result.allele_id = primary.allele_id;
            result.is_novel = result.class == Classification::INF;
            return;
        }
    }

    result.allele_id = None;
    result.is_novel = false;
}

#[cfg(test)]
mod tests {
    use super::{record_match, resolve_multi_match};
    use crate::types::Classification;
    use crate::types::{LocusResult, MatchResult};

    #[test]
    fn resolve_exact_plus_inferred_is_niph() {
        assert_eq!(
            resolve_multi_match(&[Classification::EXC, Classification::INF]),
            Classification::NIPH,
        );
    }

    #[test]
    fn record_match_recomputes_final_class() {
        let mut result = LocusResult {
            class: Classification::LNF,
            allele_id: None,
            is_novel: false,
            dna_hash: None,
            matches: Vec::new(),
        };

        record_match(
            &mut result,
            MatchResult {
                locus_idx: 0,
                allele_id: Some(10),
                class: Classification::EXC,
                bsr: 1.0,
                cds_hash: [0; 32],
                protein_hash: None,
                representative_id: None,
            },
        );
        assert_eq!(result.class, Classification::EXC);
        assert_eq!(result.allele_id, Some(10));

        record_match(
            &mut result,
            MatchResult {
                locus_idx: 0,
                allele_id: Some(11),
                class: Classification::INF,
                bsr: 0.9,
                cds_hash: [1; 32],
                protein_hash: None,
                representative_id: None,
            },
        );
        assert_eq!(result.class, Classification::NIPH);
        assert_eq!(result.allele_id, None);
    }
}
