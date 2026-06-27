//! Codon translation (genetic code tables).

/// Translate a DNA sequence to protein using the specified genetic code.
/// Returns None if the sequence length is not a multiple of 3,
/// contains ambiguous bases, or has internal stop codons.
///
/// When `cds` is true, behaves like BioPython's `Seq.translate(cds=True)`:
/// - Validates the first codon is a valid start codon for the table
/// - Translates the first codon to M (methionine) regardless of standard encoding
/// This matches bacterial convention where alternative start codons (TTG, GTG, etc.)
/// all encode fMet (formyl-methionine) at initiation.
pub fn translate(dna: &[u8], table: u8) -> Option<Vec<u8>> {
    translate_cds(dna, table, false)
}

/// Translate with CDS mode (alternative start codons → M).
/// In CDS mode (matching BioPython's `Seq.translate(cds=True)`):
/// - First codon must be a valid start codon → translated to M
/// - Last codon must be a stop codon
/// - No internal stop codons allowed
pub fn translate_cds(dna: &[u8], table: u8, cds: bool) -> Option<Vec<u8>> {
    if dna.len() < 3 || dna.len() % 3 != 0 {
        return None;
    }

    if cds {
        // Check first codon is a valid start
        let first_codon = &dna[0..3];
        if !is_start_codon(first_codon, table) {
            return None;
        }
        // Check last codon is a stop codon
        let last_codon = &dna[dna.len() - 3..];
        let last_aa = translate_codon(last_codon, table)?;
        if last_aa != b'*' {
            return None;
        }
    }

    let codons = dna.len() / 3;
    let mut protein = Vec::with_capacity(codons);

    for i in 0..codons {
        let codon = &dna[i * 3..i * 3 + 3];
        let aa = translate_codon(codon, table)?;

        // Stop codon at end is expected, skip it
        if aa == b'*' {
            if i == codons - 1 {
                break; // terminal stop — OK
            } else {
                return None; // internal stop — invalid
            }
        }

        // In CDS mode, replace first codon with M
        if i == 0 && cds {
            protein.push(b'M');
        } else {
            protein.push(aa);
        }
    }

    if protein.is_empty() {
        return None;
    }

    Some(protein)
}

/// Check if a codon is a valid start codon for the given genetic code table.
fn is_start_codon(codon: &[u8], table: u8) -> bool {
    let upper: [u8; 3] = [
        codon[0].to_ascii_uppercase(),
        codon[1].to_ascii_uppercase(),
        codon[2].to_ascii_uppercase(),
    ];
    match table {
        // Table 11 start codons: TTG, CTG, ATT, ATC, ATA, ATG, GTG
        11 => matches!(
            &upper,
            b"TTG" | b"CTG" | b"ATT" | b"ATC" | b"ATA" | b"ATG" | b"GTG"
        ),
        // Table 1 start codons: ATG only (standard)
        1 => &upper == b"ATG",
        // Table 4 start codons: TTA, TTG, CTG, ATT, ATC, ATA, ATG, GTG
        4 => matches!(
            &upper,
            b"TTA" | b"TTG" | b"CTG" | b"ATT" | b"ATC" | b"ATA" | b"ATG" | b"GTG"
        ),
        _ => matches!(
            &upper,
            b"TTG" | b"CTG" | b"ATT" | b"ATC" | b"ATA" | b"ATG" | b"GTG"
        ),
    }
}

/// Translate a single codon to amino acid.
/// Returns None for ambiguous bases.
fn translate_codon(codon: &[u8], table: u8) -> Option<u8> {
    let c0 = base_idx(codon[0])?;
    let c1 = base_idx(codon[1])?;
    let c2 = base_idx(codon[2])?;
    let idx = c0 * 16 + c1 * 4 + c2;

    let table = match table {
        11 => &GENETIC_CODE_11,
        1 => &GENETIC_CODE_1,
        4 => &GENETIC_CODE_4,
        _ => &GENETIC_CODE_11,
    };

    Some(table[idx])
}

fn base_idx(b: u8) -> Option<usize> {
    match b {
        b'T' | b't' => Some(0),
        b'C' | b'c' => Some(1),
        b'A' | b'a' => Some(2),
        b'G' | b'g' => Some(3),
        _ => None, // ambiguous
    }
}

// Genetic code 11 (Bacterial, Archaeal and Plant Plastid)
// Order: TTT, TTC, TTA, TTG, TCT, TCC, TCA, TCG, TAT, TAC, TAA, TAG,
//        TGT, TGC, TGA, TGG, CTT, CTC, CTA, CTG, CCT, CCC, CCA, CCG,
//        CAT, CAC, CAA, CAG, CGT, CGC, CGA, CGG, ATT, ATC, ATA, ATG,
//        ACT, ACC, ACA, ACG, AAT, AAC, AAA, AAG, AGT, AGC, AGA, AGG,
//        GTT, GTC, GTA, GTG, GCT, GCC, GCA, GCG, GAT, GAC, GAA, GAG,
//        GGT, GGC, GGA, GGG
static GENETIC_CODE_11: [u8; 64] = [
    b'F', b'F', b'L', b'L', b'S', b'S', b'S', b'S', b'Y', b'Y', b'*', b'*', b'C', b'C', b'*', b'W',
    b'L', b'L', b'L', b'L', b'P', b'P', b'P', b'P', b'H', b'H', b'Q', b'Q', b'R', b'R', b'R', b'R',
    b'I', b'I', b'I', b'M', b'T', b'T', b'T', b'T', b'N', b'N', b'K', b'K', b'S', b'S', b'R', b'R',
    b'V', b'V', b'V', b'V', b'A', b'A', b'A', b'A', b'D', b'D', b'E', b'E', b'G', b'G', b'G', b'G',
];

// Genetic code 1 (Standard)
static GENETIC_CODE_1: [u8; 64] = [
    b'F', b'F', b'L', b'L', b'S', b'S', b'S', b'S', b'Y', b'Y', b'*', b'*', b'C', b'C', b'*', b'W',
    b'L', b'L', b'L', b'L', b'P', b'P', b'P', b'P', b'H', b'H', b'Q', b'Q', b'R', b'R', b'R', b'R',
    b'I', b'I', b'I', b'M', b'T', b'T', b'T', b'T', b'N', b'N', b'K', b'K', b'S', b'S', b'R', b'R',
    b'V', b'V', b'V', b'V', b'A', b'A', b'A', b'A', b'D', b'D', b'E', b'E', b'G', b'G', b'G', b'G',
];

// Genetic code 4 (Mycoplasma/Spiroplasma)
static GENETIC_CODE_4: [u8; 64] = [
    b'F', b'F', b'L', b'L', b'S', b'S', b'S', b'S', b'Y', b'Y', b'*', b'*', b'C', b'C', b'W', b'W',
    b'L', b'L', b'L', b'L', b'P', b'P', b'P', b'P', b'H', b'H', b'Q', b'Q', b'R', b'R', b'R', b'R',
    b'I', b'I', b'I', b'M', b'T', b'T', b'T', b'T', b'N', b'N', b'K', b'K', b'S', b'S', b'R', b'R',
    b'V', b'V', b'V', b'V', b'A', b'A', b'A', b'A', b'D', b'D', b'E', b'E', b'G', b'G', b'G', b'G',
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_simple() {
        // ATG (M) + GCT (A) + TAA (*)
        let dna = b"ATGGCTTAA";
        let protein = translate(dna, 11).unwrap();
        assert_eq!(protein, b"MA");
    }

    #[test]
    fn test_internal_stop() {
        // ATG (M) + TAA (*) + GCT (A) + TAA (*)
        let dna = b"ATGTAAGCTTAA";
        assert!(translate(dna, 11).is_none());
    }

    #[test]
    fn test_ambiguous_base() {
        let dna = b"ATGNNN";
        assert!(translate(dna, 11).is_none());
    }

    #[test]
    fn test_translate_cds_ttg_start() {
        // TTG normally translates to L, but in CDS mode → M
        let dna = b"TTGGCTTAA";
        let protein = translate_cds(dna, 11, true).unwrap();
        assert_eq!(protein, b"MA");
        // Without CDS mode, TTG → L
        let protein_std = translate(dna, 11).unwrap();
        assert_eq!(protein_std, b"LA");
    }

    #[test]
    fn test_translate_cds_gtg_start() {
        // GTG normally translates to V, but in CDS mode → M
        let dna = b"GTGGCTTAA";
        let protein = translate_cds(dna, 11, true).unwrap();
        assert_eq!(protein, b"MA");
        let protein_std = translate(dna, 11).unwrap();
        assert_eq!(protein_std, b"VA");
    }

    #[test]
    fn test_translate_cds_atg_start() {
        // ATG → M in both modes
        let dna = b"ATGGCTTAA";
        let protein = translate_cds(dna, 11, true).unwrap();
        assert_eq!(protein, b"MA");
    }

    #[test]
    fn test_translate_cds_no_stop_rejected() {
        // CDS mode requires terminal stop codon
        let dna = b"ATGGCT"; // ATG (M) + GCT (A), no stop
        assert!(translate_cds(dna, 11, true).is_none());
        // Standard mode accepts it
        let protein = translate(dna, 11).unwrap();
        assert_eq!(protein, b"MA");
    }

    #[test]
    fn test_translate_cds_invalid_start_rejected() {
        // CDS mode requires valid start codon
        let dna = b"GCTGCTTAA"; // GCT is not a start codon
        assert!(translate_cds(dna, 11, true).is_none());
        // Standard mode accepts it
        let protein = translate(dna, 11).unwrap();
        assert_eq!(protein, b"AA");
    }
}
