//! FFI bindings to parasail for SIMD-accelerated Smith-Waterman alignment.
//!
//! parasail provides highly optimized SW implementations using SSE4.1, AVX2,
//! and AVX-512 striped algorithms that auto-detect the best instruction set.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

// --- C types ---

#[repr(C)]
pub struct ParasailResult {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ParasailMatrix {
    _private: [u8; 0],
}

// --- C function declarations ---

#[link(name = "parasail")]
extern "C" {
    fn parasail_matrix_lookup(name: *const c_char) -> *const ParasailMatrix;

    fn parasail_sw_striped_sat(
        s1: *const c_char, s1_len: c_int,
        s2: *const c_char, s2_len: c_int,
        open: c_int, gap: c_int,
        matrix: *const ParasailMatrix,
    ) -> *mut ParasailResult;

    fn parasail_result_get_score(result: *const ParasailResult) -> c_int;
    fn parasail_result_get_end_query(result: *const ParasailResult) -> c_int;
    fn parasail_result_get_end_ref(result: *const ParasailResult) -> c_int;
    fn parasail_result_free(result: *mut ParasailResult);
}

// --- Safe wrapper ---

/// Thread-safe reference to the BLOSUM62 matrix (parasail matrices are static).
static BLOSUM62_PTR: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

fn get_blosum62() -> *const ParasailMatrix {
    let ptr = *BLOSUM62_PTR.get_or_init(|| {
        let name = b"blosum62\0";
        let ptr = unsafe { parasail_matrix_lookup(name.as_ptr() as *const c_char) };
        assert!(!ptr.is_null(), "Failed to load BLOSUM62 matrix from parasail");
        ptr as usize
    });
    ptr as *const ParasailMatrix
}

/// SIMD-accelerated Smith-Waterman using parasail.
/// Returns (score, query_end_1based, target_end_1based).
/// Uses the striped algorithm with automatic SIMD width selection (AVX2/SSE4.1).
#[inline]
pub fn sw_simd(query: &[u8], target: &[u8]) -> (i32, u32, u32) {
    if query.is_empty() || target.is_empty() {
        return (0, 0, 0);
    }

    let matrix = get_blosum62();

    let result = unsafe {
        parasail_sw_striped_sat(
            query.as_ptr() as *const c_char, query.len() as c_int,
            target.as_ptr() as *const c_char, target.len() as c_int,
            11, 1, // gap_open, gap_extend (BLAST defaults)
            matrix,
        )
    };

    if result.is_null() {
        return (0, 0, 0);
    }

    let score = unsafe { parasail_result_get_score(result) };
    let end_query = unsafe { parasail_result_get_end_query(result) }; // 0-based
    let end_ref = unsafe { parasail_result_get_end_ref(result) }; // 0-based

    unsafe { parasail_result_free(result) };

    (score, (end_query + 1) as u32, (end_ref + 1) as u32) // convert to 1-based
}

/// SIMD SW with full position tracking (score, query_start/end, target_start/end).
///
/// Computes target_start by reverse-aligning from the end positions.
/// Only use this when you need start positions (e.g., PLOT3/PLOT5 detection).
pub fn sw_simd_full(query: &[u8], target: &[u8]) -> (i32, u32, u32, u32, u32) {
    let (score, query_end, target_end) = sw_simd(query, target);
    if score <= 0 {
        return (0, 0, 0, 0, 0);
    }

    // To find start positions: reverse both sequences up to the end positions,
    // re-align, and the end positions of the reverse alignment give us the starts.
    let qe = query_end as usize;
    let te = target_end as usize;

    let mut q_rev: Vec<u8> = query[..qe].to_vec();
    q_rev.reverse();
    let mut t_rev: Vec<u8> = target[..te].to_vec();
    t_rev.reverse();

    let (_, rev_qend, rev_tend) = sw_simd(&q_rev, &t_rev);

    // Start positions in original orientation
    let query_start = qe as u32 - rev_qend + 1;
    let target_start = te as u32 - rev_tend + 1;

    (score, query_start, query_end, target_start, target_end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical() {
        let seq = b"ACDEFGHIKLMNPQ";
        let (score, qend, tend) = sw_simd(seq, seq);
        assert!(score > 0);
        assert_eq!(qend, 14);
        assert_eq!(tend, 14);
    }

    #[test]
    fn test_self_score_matches_rust() {
        let seq = b"ACDEFGHIKLMNPQ";
        let (score, _, _) = sw_simd(seq, seq);
        let rust_score = crate::sw::self_score(seq);
        assert_eq!(score, rust_score, "parasail vs Rust self-score mismatch");
    }

    #[test]
    fn test_full_positions() {
        let query = b"ACDEFGHIKLMNPQ";
        let target = b"XXXXXACDEFGHIKLMNPQYYYYY";
        let (score, qs, qe, ts, te) = sw_simd_full(query, target);
        assert!(score > 0);
        assert_eq!(qs, 1);
        assert_eq!(qe, 14);
        assert_eq!(ts, 6); // target starts at position 6
        assert_eq!(te, 19); // target ends at position 19
    }
}
