//! Codon translation (genetic code tables).

/// Translate a DNA sequence to protein using the specified genetic code.
/// Returns None if the sequence length is not a multiple of 3,
/// contains ambiguous bases, or has internal stop codons.
pub fn translate(dna: &[u8], table: u8) -> Option<Vec<u8>> {
    if dna.len() < 3 || dna.len() % 3 != 0 {
        return None;
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

        protein.push(aa);
    }

    if protein.is_empty() {
        return None;
    }

    Some(protein)
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
    b'F', b'F', b'L', b'L', b'S', b'S', b'S', b'S', b'Y', b'Y', b'*', b'*',
    b'C', b'C', b'*', b'W', b'L', b'L', b'L', b'L', b'P', b'P', b'P', b'P',
    b'H', b'H', b'Q', b'Q', b'R', b'R', b'R', b'R', b'I', b'I', b'I', b'M',
    b'T', b'T', b'T', b'T', b'N', b'N', b'K', b'K', b'S', b'S', b'R', b'R',
    b'V', b'V', b'V', b'V', b'A', b'A', b'A', b'A', b'D', b'D', b'E', b'E',
    b'G', b'G', b'G', b'G',
];

// Genetic code 1 (Standard)
static GENETIC_CODE_1: [u8; 64] = [
    b'F', b'F', b'L', b'L', b'S', b'S', b'S', b'S', b'Y', b'Y', b'*', b'*',
    b'C', b'C', b'*', b'W', b'L', b'L', b'L', b'L', b'P', b'P', b'P', b'P',
    b'H', b'H', b'Q', b'Q', b'R', b'R', b'R', b'R', b'I', b'I', b'I', b'M',
    b'T', b'T', b'T', b'T', b'N', b'N', b'K', b'K', b'S', b'S', b'R', b'R',
    b'V', b'V', b'V', b'V', b'A', b'A', b'A', b'A', b'D', b'D', b'E', b'E',
    b'G', b'G', b'G', b'G',
];

// Genetic code 4 (Mycoplasma/Spiroplasma)
static GENETIC_CODE_4: [u8; 64] = [
    b'F', b'F', b'L', b'L', b'S', b'S', b'S', b'S', b'Y', b'Y', b'*', b'*',
    b'C', b'C', b'W', b'W', b'L', b'L', b'L', b'L', b'P', b'P', b'P', b'P',
    b'H', b'H', b'Q', b'Q', b'R', b'R', b'R', b'R', b'I', b'I', b'I', b'M',
    b'T', b'T', b'T', b'T', b'N', b'N', b'K', b'K', b'S', b'S', b'R', b'R',
    b'V', b'V', b'V', b'V', b'A', b'A', b'A', b'A', b'D', b'D', b'E', b'E',
    b'G', b'G', b'G', b'G',
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
}
