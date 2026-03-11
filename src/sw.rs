//! Smith-Waterman protein alignment with BLOSUM62.
//! Affine gap penalties: gap_open=11, gap_extend=1 (BLAST defaults).

use crate::types::SwResult;

// BLOSUM62 scoring matrix.
// Amino acid order: A R N D C Q E G H I L K M F P S T W Y V B Z X *

static BLOSUM62: [[i8; 24]; 24] = [
//   A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   B   Z   X   *
  [  4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4], // A
  [ -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4], // R
  [ -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4], // N
  [ -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4], // D
  [  0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4], // C
  [ -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4], // Q
  [ -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4], // E
  [  0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4], // G
  [ -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4], // H
  [ -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4], // I
  [ -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4], // L
  [ -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4], // K
  [ -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4], // M
  [ -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4], // F
  [ -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4], // P
  [  1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4], // S
  [  0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4], // T
  [ -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4], // W
  [ -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4], // Y
  [  0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4], // V
  [ -2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4], // B
  [ -1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4], // Z
  [  0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4], // X
  [ -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1], // *
];

/// Lookup table: ASCII byte -> BLOSUM62 index (0-23), 23 for unknown.
static AA_TO_IDX: [u8; 256] = {
    let mut table = [22u8; 256]; // default to X (index 22)
    let order = b"ARNDCQEGHILKMFPSTWYVBZX*";
    let mut i = 0;
    while i < 24 {
        table[order[i] as usize] = i as u8;
        // lowercase
        if order[i] >= b'A' && order[i] <= b'Z' {
            table[(order[i] + 32) as usize] = i as u8;
        }
        i += 1;
    }
    table
};

const GAP_OPEN: i32 = 11;
const GAP_EXTEND: i32 = 1;

/// Encode a protein sequence to BLOSUM62 indices.
pub fn encode_protein(seq: &[u8]) -> Vec<u8> {
    seq.iter().map(|&b| AA_TO_IDX[b as usize]).collect()
}

/// Smith-Waterman local alignment with affine gap penalties.
/// Returns (score, query_start_1based, query_end_1based).
pub fn smith_waterman(query: &[u8], target: &[u8]) -> (i32, u32, u32, u32, u32) {
    let m = query.len();
    let n = target.len();

    if m == 0 || n == 0 {
        return (0, 0, 0, 0, 0);
    }

    // Pre-encode sequences
    let q_enc = encode_protein(query);
    let t_enc = encode_protein(target);

    sw_encoded(&q_enc, &t_enc)
}

/// SW on pre-encoded sequences (BLOSUM62 indices).
/// Returns (score, query_start_1based, query_end_1based, target_start_1based, target_end_1based).
pub fn sw_encoded(query: &[u8], target: &[u8]) -> (i32, u32, u32, u32, u32) {
    let m = query.len();
    let n = target.len();
    let gap_oe = GAP_OPEN + GAP_EXTEND;

    // 1D arrays for column-by-column processing
    let mut h_row = vec![0i32; m + 1];
    let mut e_row = vec![0i32; m + 1];
    // Query start tracking
    let mut hs_row = vec![0u32; m + 1];
    for i in 0..=m {
        hs_row[i] = i as u32;
    }
    let mut es_row = hs_row.clone();
    // Target start tracking
    let mut hts_row = vec![0u32; m + 1]; // target start per cell
    let mut ets_row = vec![0u32; m + 1];

    let mut max_score = 0i32;
    let mut best_qi = 0u32;
    let mut best_qstart = 0u32;
    let mut best_tj = 0u32;
    let mut best_tstart = 0u32;

    for j in 0..n {
        let tj = target[j] as usize;
        let mut f_val = 0i32;
        let mut fs = 0u32;
        let mut fts = 0u32;
        let mut h_diag = 0i32;
        let mut h_diag_s = 0u32;
        let mut h_diag_ts = 0u32;

        for i in 1..=m {
            let qi = query[i - 1] as usize;
            let h_left = h_row[i];
            let h_left_s = hs_row[i];
            let h_left_ts = hts_row[i];

            let match_val = h_diag + BLOSUM62[qi][tj] as i32;
            let match_s = h_diag_s;
            let match_ts = h_diag_ts;

            // E: gap in target (horizontal)
            let e_open = h_left - gap_oe;
            let e_ext = e_row[i] - GAP_EXTEND;
            if e_open > e_ext {
                e_row[i] = e_open;
                es_row[i] = h_left_s;
                ets_row[i] = h_left_ts;
            } else {
                e_row[i] = e_ext;
            }

            // F: gap in query (vertical)
            let f_open = h_row[i - 1] - gap_oe;
            let f_ext = f_val - GAP_EXTEND;
            if f_open > f_ext {
                f_val = f_open;
                fs = hs_row[i - 1];
                fts = hts_row[i - 1];
            } else {
                f_val = f_ext;
            }

            // H: best of match, E, F, or 0
            let mut h = match_val;
            let mut hs = match_s;
            let mut hts = match_ts;
            if e_row[i] > h { h = e_row[i]; hs = es_row[i]; hts = ets_row[i]; }
            if f_val > h { h = f_val; hs = fs; hts = fts; }
            if h < 0 { h = 0; hs = i as u32; hts = j as u32; }

            h_diag = h_left;
            h_diag_s = h_left_s;
            h_diag_ts = h_left_ts;
            h_row[i] = h;
            hs_row[i] = hs;
            hts_row[i] = hts;

            if h > max_score {
                max_score = h;
                best_qi = i as u32;
                best_qstart = hs;
                best_tj = j as u32 + 1; // 1-based
                best_tstart = hts;
            }
        }
    }

    (max_score, best_qstart + 1, best_qi, best_tstart + 1, best_tj) // 1-based positions
}

/// Batch Smith-Waterman: align multiple query-target pairs.
/// Uses rayon for parallelism.
pub fn sw_batch(pairs: &[(usize, usize)], queries: &[Vec<u8>], targets: &[Vec<u8>]) -> Vec<SwResult> {
    use rayon::prelude::*;

    pairs.par_iter().filter_map(|&(qi, ti)| {
        let (score, qstart, qend, tstart, tend) = sw_encoded(&queries[qi], &targets[ti]);
        if score > 0 {
            Some(SwResult {
                query_id: qi,
                target_id: ti,
                score,
                query_start: qstart,
                query_end: qend,
                query_len: queries[qi].len() as u32,
                target_len: targets[ti].len() as u32,
                target_start: tstart,
                target_end: tend,
            })
        } else {
            None
        }
    }).collect()
}

/// Compute self-alignment score for a protein sequence.
pub fn self_score(protein: &[u8]) -> i32 {
    let enc = encode_protein(protein);
    sw_encoded(&enc, &enc).0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_alignment() {
        let seq = b"ACDEFGHIKLMNPQ";
        let score = self_score(seq);
        assert_eq!(score, 80); // known value from GPU kernel test
    }

    #[test]
    fn test_identical_seqs() {
        let (score, qstart, qend, tstart, tend) = smith_waterman(b"ACDEFGHIKLMNPQ", b"ACDEFGHIKLMNPQ");
        assert_eq!(score, 80);
        assert_eq!(qstart, 1);
        assert_eq!(qend, 14);
        assert_eq!(tstart, 1);
        assert_eq!(tend, 14);
    }

    #[test]
    fn test_empty() {
        let (score, _, _, _, _) = smith_waterman(b"", b"ACDEF");
        assert_eq!(score, 0);
    }
}
