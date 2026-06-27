//! Pure Rust reimplementation of Prodigal 2.6.3 gene prediction.
//! Bit-exact output matching the original C implementation.
//!
//! Only implements the "single" mode with pre-trained .trn file,
//! which is what chewBBACA uses (prodigal -p single -t file.trn).

use std::fs::File;
use std::io::{self, BufRead, Read as IoRead};
use std::path::Path;

// ============================================================================
// Constants (from sequence.h, node.h, dprog.h, gene.h)
// ============================================================================

const MAX_SEQ: usize = 32_000_000;
const MAX_LINE: usize = 10_000;
const WINDOW: usize = 120;
const MASK_SIZE: usize = 50;
const MAX_MASKS: usize = 5000;

const ATG: u8 = 0;
const GTG: u8 = 1;
const TTG: u8 = 2;
const STOP: u8 = 3;

const MIN_GENE: i32 = 90;
const MIN_EDGE_GENE: i32 = 60;
const MAX_SAM_OVLP: i32 = 60;
const OPER_DIST: i32 = 60;
const EDGE_BONUS: f64 = 0.74;
const EDGE_UPS: f64 = -1.00;
const META_PEN: f64 = 7.5;

const MAX_OPP_OVLP: i32 = 200;
const MAX_NODE_DIST: usize = 500;
const MAX_GENES: usize = 30000;

// ============================================================================
// Bitmap operations (inline, replacing the C bitmap.c)
// ============================================================================

#[inline(always)]
fn bm_test(bm: &[u8], ndx: usize) -> u8 {
    // Safety: callers ensure ndx/8 < bm.len() (sequence buffers are MAX_SEQ/4 bytes)
    unsafe {
        if (*bm.get_unchecked(ndx >> 3) & (1 << (ndx & 0x07))) != 0 {
            1
        } else {
            0
        }
    }
}

#[inline(always)]
fn bm_set(bm: &mut [u8], ndx: usize) {
    unsafe {
        *bm.get_unchecked_mut(ndx >> 3) |= 1 << (ndx & 0x07);
    }
}

#[inline(always)]
fn bm_clear(bm: &mut [u8], ndx: usize) {
    unsafe {
        *bm.get_unchecked_mut(ndx >> 3) &= !(1 << (ndx & 0x07));
    }
}

#[inline(always)]
fn bm_toggle(bm: &mut [u8], ndx: usize) {
    unsafe {
        *bm.get_unchecked_mut(ndx >> 3) ^= 1 << (ndx & 0x07);
    }
}

// ============================================================================
// Sequence operations (from sequence.c)
// ============================================================================

/// Read a 2-bit nucleotide value from the packed bitmap at position n.
/// Returns: A=0, G=1, C=2, T=3 (matching prodigal's bitmap encoding).
/// Safety: seq buffer is MAX_SEQ/4 bytes, position n < MAX_SEQ, so byte_idx+1 is always valid.
#[inline(always)]
fn nuc_val(seq: &[u8], n: usize) -> u8 {
    let bit_pos = n * 2;
    let byte_idx = bit_pos >> 3;
    let bit_offset = bit_pos & 7;
    unsafe {
        // Read 2 bytes to handle bit pairs that straddle byte boundaries (bit_offset=7)
        let word = (*seq.get_unchecked(byte_idx) as u16)
            | ((*seq.get_unchecked(byte_idx + 1) as u16) << 8);
        ((word >> bit_offset) & 3) as u8
    }
}

#[inline(always)]
fn is_a(seq: &[u8], n: usize) -> bool {
    nuc_val(seq, n) == 0
}

#[inline(always)]
fn is_c(seq: &[u8], n: usize) -> bool {
    nuc_val(seq, n) == 2
}

#[inline(always)]
fn is_g(seq: &[u8], n: usize) -> bool {
    nuc_val(seq, n) == 1
}

#[inline(always)]
fn is_t(seq: &[u8], n: usize) -> bool {
    nuc_val(seq, n) == 3
}

#[inline(always)]
fn is_n(useq: &[u8], n: usize) -> bool {
    bm_test(useq, n) != 0
}

#[inline(always)]
fn is_gc(seq: &[u8], n: usize) -> i32 {
    // G=01, C=10 → exactly one bit set → nuc_val is 1 or 2
    let v = nuc_val(seq, n);
    if v == 1 || v == 2 {
        1
    } else {
        0
    }
}

/// Read 6-bit codon value (3 nucleotides) from packed bitmap at position n.
/// Returns: bits[0:1]=nuc[n], bits[2:3]=nuc[n+1], bits[4:5]=nuc[n+2]
/// Encoding: A=0, G=1, C=2, T=3
#[inline(always)]
fn codon_val(seq: &[u8], n: usize) -> u8 {
    let bit_start = n * 2;
    let byte_start = bit_start >> 3;
    let bit_offset = bit_start & 7;
    unsafe {
        let raw = (*seq.get_unchecked(byte_start) as u16)
            | ((*seq.get_unchecked(byte_start + 1) as u16) << 8);
        ((raw >> bit_offset) & 0x3F) as u8
    }
}

// Codon classification bits
const CC_STOP: u8 = 1;
const CC_START: u8 = 2;
const CC_ATG: u8 = 4;
const CC_GTG: u8 = 8;
const CC_TTG: u8 = 16;

/// Build a 64-entry codon classification table for a given translation table.
/// Each entry is a bitmask: CC_STOP, CC_START, CC_ATG, CC_GTG, CC_TTG.
fn build_codon_table(tt: i32) -> [u8; 64] {
    let mut table = [0u8; 64];
    for val in 0u8..64 {
        let n0 = val & 3; // A=0, G=1, C=2, T=3
        let n1 = (val >> 2) & 3;
        let n2 = (val >> 4) & 3;

        // --- Stop codons ---
        let stop = match (n0, n1, n2) {
            (3, 0, 1) => !(tt == 6 || tt == 15 || tt == 16 || tt == 22), // TAG
            (3, 1, 0) => {
                !((tt >= 2 && tt <= 5)
                    || tt == 9
                    || tt == 10
                    || tt == 13
                    || tt == 14
                    || tt == 21
                    || tt == 25)
            } // TGA
            (3, 0, 0) => !(tt == 6 || tt == 14),                         // TAA
            (0, 1, 0) => tt == 2,                                        // AGA (code 2)
            (0, 1, 1) => tt == 2,                                        // AGG (code 2)
            (3, 2, 0) => tt == 22,                                       // TCA (code 22)
            (3, 3, 0) => tt == 23,                                       // TTA (code 23)
            _ => false,
        };
        if stop {
            table[val as usize] |= CC_STOP;
            continue;
        }

        // --- Start codons ---
        // ATG: universal start
        if n0 == 0 && n1 == 3 && n2 == 1 {
            // ATG
            table[val as usize] |= CC_START | CC_ATG;
            continue;
        }
        if tt == 6 || tt == 10 || tt == 14 || tt == 15 || tt == 16 || tt == 22 {
            continue; // only ATG for these tables
        }
        // GTG
        if n0 == 1 && n1 == 3 && n2 == 1 {
            // GTG
            if !(tt == 1 || tt == 3 || tt == 12 || tt == 22) {
                table[val as usize] |= CC_START | CC_GTG;
            }
            continue;
        }
        // TTG
        if n0 == 3 && n1 == 3 && n2 == 1 {
            // TTG
            if !(tt < 4 || tt == 9 || (tt >= 21 && tt < 25)) {
                table[val as usize] |= CC_START | CC_TTG;
            }
        }
    }
    table
}

fn is_stop(seq: &[u8], n: usize, tt: i32) -> bool {
    // TAG
    if is_t(seq, n) && is_a(seq, n + 1) && is_g(seq, n + 2) {
        if tt == 6 || tt == 15 || tt == 16 || tt == 22 {
            return false;
        }
        return true;
    }
    // TGA
    if is_t(seq, n) && is_g(seq, n + 1) && is_a(seq, n + 2) {
        if (tt >= 2 && tt <= 5)
            || tt == 9
            || tt == 10
            || tt == 13
            || tt == 14
            || tt == 21
            || tt == 25
        {
            return false;
        }
        return true;
    }
    // TAA
    if is_t(seq, n) && is_a(seq, n + 1) && is_a(seq, n + 2) {
        if tt == 6 || tt == 14 {
            return false;
        }
        return true;
    }
    // Code 2
    if tt == 2 && is_a(seq, n) && is_g(seq, n + 1) && is_a(seq, n + 2) {
        return true;
    }
    if tt == 2 && is_a(seq, n) && is_g(seq, n + 1) && is_g(seq, n + 2) {
        return true;
    }
    // Code 22
    if tt == 22 && is_t(seq, n) && is_c(seq, n + 1) && is_a(seq, n + 2) {
        return true;
    }
    // Code 23
    if tt == 23 && is_t(seq, n) && is_t(seq, n + 1) && is_a(seq, n + 2) {
        return true;
    }
    false
}

fn is_start(seq: &[u8], n: usize, tt: i32) -> bool {
    // ATG
    if is_a(seq, n) && is_t(seq, n + 1) && is_g(seq, n + 2) {
        return true;
    }
    if tt == 6 || tt == 10 || tt == 14 || tt == 15 || tt == 16 || tt == 22 {
        return false;
    }
    // GTG
    if is_g(seq, n) && is_t(seq, n + 1) && is_g(seq, n + 2) {
        if tt == 1 || tt == 3 || tt == 12 || tt == 22 {
            return false;
        }
        return true;
    }
    // TTG
    if is_t(seq, n) && is_t(seq, n + 1) && is_g(seq, n + 2) {
        if tt < 4 || tt == 9 || (tt >= 21 && tt < 25) {
            return false;
        }
        return true;
    }
    false
}

#[inline(always)]
fn is_atg(seq: &[u8], n: usize) -> bool {
    is_a(seq, n) && is_t(seq, n + 1) && is_g(seq, n + 2)
}

#[inline(always)]
fn is_gtg(seq: &[u8], n: usize) -> bool {
    is_g(seq, n) && is_t(seq, n + 1) && is_g(seq, n + 2)
}

#[inline(always)]
fn is_ttg(seq: &[u8], n: usize) -> bool {
    is_t(seq, n) && is_t(seq, n + 1) && is_g(seq, n + 2)
}

fn rcom_seq(seq: &[u8], rseq: &mut [u8], useq: &[u8], len: usize) {
    // Fast byte-level reverse complement: process 4 nucleotides per iteration
    let full_bytes = len / 4;
    let remainder = len % 4;

    for b in 0..full_bytes {
        // rseq byte b: nucleotides at positions 4b..4b+3
        // These map to seq positions: len-1-4b, len-2-4b, len-3-4b, len-4-4b
        let p3 = len - 1 - 4 * b;
        let n0 = nuc_val(seq, p3) ^ 3; // complement
        let n1 = nuc_val(seq, p3 - 1) ^ 3;
        let n2 = nuc_val(seq, p3 - 2) ^ 3;
        let n3 = nuc_val(seq, p3 - 3) ^ 3;
        unsafe {
            *rseq.get_unchecked_mut(b) = n0 | (n1 << 2) | (n2 << 4) | (n3 << 6);
        }
    }

    // Handle remainder nucleotides
    if remainder > 0 {
        let mut byte = 0u8;
        for r in 0..remainder {
            let pos = 4 * full_bytes + r;
            let src = len - 1 - pos;
            let nv = nuc_val(seq, src) ^ 3;
            byte |= nv << (r * 2);
        }
        unsafe {
            *rseq.get_unchecked_mut(full_bytes) = byte;
        }
    }

    // Fix N bases: toggle both bits of the reverse complement position
    let slen = len * 2;
    for i in 0..len {
        if bm_test(useq, i) == 1 {
            bm_toggle(rseq, slen - 1 - i * 2);
            bm_toggle(rseq, slen - 2 - i * 2);
        }
    }
}

/// Extract a 2*len bit mer index from the packed bitmap at position pos.
/// For len=6 (hexamer): reads 12 bits. For len=3: 6 bits. For len=1: 2 bits.
/// Uses bulk byte reads instead of individual bitmap tests (12x fewer memory ops for hexamers).
#[inline(always)]
fn mer_ndx(len: usize, seq: &[u8], pos: usize) -> usize {
    let bit_start = pos * 2;
    let byte_start = bit_start >> 3;
    let bit_offset = bit_start & 7;
    let bits_needed = 2 * len; // max 12 for len=6
                               // Read 3 bytes (24 bits) which covers max 12+7=19 bits needed
    unsafe {
        let raw = (*seq.get_unchecked(byte_start) as u32)
            | ((*seq.get_unchecked(byte_start + 1) as u32) << 8)
            | ((*seq.get_unchecked(byte_start + 2) as u32) << 16);
        ((raw >> bit_offset) & ((1u32 << bits_needed) - 1)) as usize
    }
}

fn max_fr(n1: i32, n2: i32, n3: i32) -> usize {
    if n1 > n2 {
        if n1 > n3 {
            0
        } else {
            2
        }
    } else {
        if n2 > n3 {
            1
        } else {
            2
        }
    }
}

fn calc_most_gc_frame(seq: &[u8], slen: usize) -> Vec<i32> {
    let mut gp = vec![-1i32; slen];
    let mut fwd = vec![0i32; slen];
    let mut bwd = vec![0i32; slen];
    let mut tot = vec![0i32; slen];

    for i in 0..3usize {
        let mut j = i;
        while j < slen {
            if j < 3 {
                fwd[j] = is_gc(seq, j);
            } else {
                fwd[j] = fwd[j - 3] + is_gc(seq, j);
            }
            let rj = slen - j - 1;
            if j < 3 {
                bwd[rj] = is_gc(seq, rj);
            } else {
                bwd[rj] = bwd[rj + 3] + is_gc(seq, rj);
            }
            j += 3;
        }
    }
    for i in 0..slen {
        tot[i] = fwd[i] + bwd[i] - is_gc(seq, i);
        if i >= WINDOW / 2 {
            tot[i] -= fwd[i - WINDOW / 2];
        }
        if i + WINDOW / 2 < slen {
            tot[i] -= bwd[i + WINDOW / 2];
        }
    }
    let mut i = 0;
    while i + 2 < slen {
        let win = max_fr(tot[i], tot[i + 1], tot[i + 2]);
        for j in 0..3 {
            if i + j < slen {
                gp[i + j] = win as i32;
            }
        }
        i += 3;
    }
    gp
}

// ============================================================================
// Training data (from training.h/training.c)
// ============================================================================

#[repr(C)]
#[derive(Clone)]
pub struct Training {
    pub gc: f64,
    pub trans_table: i32,
    pub st_wt: f64,
    pub bias: [f64; 3],
    pub type_wt: [f64; 3],
    pub uses_sd: i32,
    pub rbs_wt: [f64; 28],
    pub ups_comp: [[f64; 4]; 32],
    pub mot_wt: [[[f64; 4096]; 4]; 4],
    pub no_mot: f64,
    pub gene_dc: [f64; 4096],
}

impl Training {
    pub fn from_file(path: &Path) -> io::Result<Self> {
        let mut f = File::open(path)?;
        let size = std::mem::size_of::<Training>();
        let mut buf = vec![0u8; size];
        f.read_exact(&mut buf)?;
        // Safety: Training is repr(C) with no pointers, just f64/i32 arrays
        let tinf = unsafe { std::ptr::read(buf.as_ptr() as *const Training) };
        Ok(tinf)
    }
}

// ============================================================================
// Mask
// ============================================================================

#[derive(Clone, Copy, Default)]
struct Mask {
    begin: i32,
    end: i32,
}

// ============================================================================
// Motif
// ============================================================================

#[derive(Clone, Copy, Default)]
struct Motif {
    ndx: i32,
    len: i32,
    spacer: i32,
    spacendx: i32,
    score: f64,
}

// ============================================================================
// Node
// ============================================================================

#[derive(Clone)]
struct Node {
    ntype: u8, // 0=ATG, 1=GTG, 2=TTG, 3=STOP
    edge: i32,
    ndx: i32,
    strand: i32, // 1 or -1
    stop_val: i32,
    star_ptr: [i32; 3],
    gc_bias: i32,
    gc_score: [f64; 3],
    cscore: f64,
    gc_cont: f64,
    rbs: [i32; 2],
    mot: Motif,
    uscore: f64,
    tscore: f64,
    rscore: f64,
    sscore: f64,
    traceb: i32,
    tracef: i32,
    ov_mark: i32,
    score: f64,
    elim: i32,
}

impl Default for Node {
    fn default() -> Self {
        Node {
            ntype: 0,
            edge: 0,
            ndx: 0,
            strand: 0,
            stop_val: 0,
            star_ptr: [0; 3],
            gc_bias: 0,
            gc_score: [0.0; 3],
            cscore: 0.0,
            gc_cont: 0.0,
            rbs: [0; 2],
            mot: Motif::default(),
            uscore: 0.0,
            tscore: 0.0,
            rscore: 0.0,
            sscore: 0.0,
            traceb: -1,
            tracef: -1,
            ov_mark: -1,
            score: 0.0,
            elim: 0,
        }
    }
}

// ============================================================================
// Gene
// ============================================================================

#[derive(Clone, Default)]
struct Gene {
    begin: i32,
    end: i32,
    start_ndx: usize,
    stop_ndx: usize,
}

// ============================================================================
// Sequence reading (multi-contig FASTA)
// ============================================================================

struct SeqReader {
    lines: Vec<String>,
    pos: usize,
    sctr: i32,
    /// Saved header from previous call's "next header" detection (like C's new_hdr -> cur_hdr).
    next_header: String,
}

impl SeqReader {
    fn new(path: &Path) -> io::Result<Self> {
        let f = File::open(path)?;
        let reader = io::BufReader::new(f);
        let lines: Vec<String> = reader.lines().map(|l| l.unwrap_or_default()).collect();
        Ok(SeqReader {
            lines,
            pos: 0,
            sctr: 0,
            next_header: String::new(),
        })
    }

    /// Read next sequence, returns (slen, gc, cur_header, masks)
    /// Returns None when done.
    fn next_seq(
        &mut self,
        seq: &mut [u8],
        useq: &mut [u8],
        do_mask: bool,
    ) -> Option<(usize, f64, String, Vec<Mask>)> {
        let mut reading_seq = false;
        let mut genbank_end = false;
        let mut bctr: usize = 0;
        let mut len: usize = 0;
        let mut gc_cont: usize = 0;
        let mut mask_beg: i32 = -1;
        let mut masks = Vec::new();

        // C prodigal copies new_hdr -> cur_hdr between calls
        let mut cur_header = if self.sctr > 0 && !self.next_header.is_empty() {
            self.next_header.clone()
        } else {
            format!("Prodigal_Seq_{}", self.sctr + 1)
        };
        let mut new_header = format!("Prodigal_Seq_{}", self.sctr + 2);

        if self.sctr > 0 {
            reading_seq = true;
        }

        while self.pos < self.lines.len() {
            let line = &self.lines[self.pos];
            self.pos += 1;

            if line.starts_with('>')
                || (line.starts_with("SQ"))
                || (line.len() > 6 && &line[..6] == "ORIGIN")
            {
                if reading_seq || genbank_end || self.sctr > 0 {
                    if line.starts_with('>') {
                        new_header = line[1..].trim_end().to_string();
                    }
                    break;
                }
                if line.starts_with('>') {
                    cur_header = line[1..].trim_end().to_string();
                }
                reading_seq = true;
            } else if reading_seq && line.starts_with("//") {
                reading_seq = false;
                genbank_end = true;
            } else if reading_seq {
                for &ch in line.as_bytes() {
                    if ch < b'A' || ch > b'z' {
                        continue;
                    }
                    if do_mask && mask_beg != -1 && ch != b'N' && ch != b'n' {
                        if (len as i32) - mask_beg >= MASK_SIZE as i32 {
                            if masks.len() < MAX_MASKS {
                                masks.push(Mask {
                                    begin: mask_beg,
                                    end: len as i32 - 1,
                                });
                            }
                        }
                        mask_beg = -1;
                    }
                    if do_mask && mask_beg == -1 && (ch == b'N' || ch == b'n') {
                        mask_beg = len as i32;
                    }
                    match ch {
                        b'g' | b'G' => {
                            bm_set(seq, bctr);
                            gc_cont += 1;
                        }
                        b't' | b'T' => {
                            bm_set(seq, bctr);
                            bm_set(seq, bctr + 1);
                        }
                        b'c' | b'C' => {
                            bm_set(seq, bctr + 1);
                            gc_cont += 1;
                        }
                        b'a' | b'A' => { /* A = 00, already cleared */ }
                        _ => {
                            bm_set(seq, bctr + 1);
                            bm_set(useq, len);
                        } // N -> C encoding + useq mark
                    }
                    bctr += 2;
                    len += 1;
                }
            }
            if len + MAX_LINE >= MAX_SEQ {
                break;
            }
        }

        if len == 0 {
            return None;
        }
        let gc = gc_cont as f64 / len as f64;
        self.sctr += 1;
        self.next_header = new_header;
        Some((len, gc, cur_header, masks))
    }
}

// ============================================================================
// calc_short_header
// ============================================================================

fn calc_short_header(header: &str, sctr: i32) -> String {
    for (i, ch) in header.chars().enumerate() {
        if ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n' {
            let s = &header[..i];
            if s.is_empty() {
                return format!("Prodigal_Seq_{}", sctr);
            }
            return s.to_string();
        }
    }
    if header.is_empty() {
        format!("Prodigal_Seq_{}", sctr)
    } else {
        header.to_string()
    }
}

// ============================================================================
// Shine-Dalgarno scoring (from sequence.c)
// ============================================================================

fn imin(x: i32, y: i32) -> i32 {
    if x < y {
        x
    } else {
        y
    }
}
fn dmin(x: f64, y: f64) -> f64 {
    if x < y {
        x
    } else {
        y
    }
}
fn dmax(x: f64, y: f64) -> f64 {
    if x > y {
        x
    } else {
        y
    }
}

fn shine_dalgarno_exact(seq: &[u8], pos: i32, start: i32, rwt: &[f64; 28]) -> i32 {
    let limit = imin(6, start - 4 - pos) as usize;
    let mut match_arr = [-10.0f64; 6];

    for i in 0..limit {
        let pi = pos + i as i32;
        if pi >= 0 {
            let p = pi as usize;
            if i % 3 == 0 && is_a(seq, p) {
                match_arr[i] = 2.0;
            } else if i % 3 != 0 && is_g(seq, p) {
                match_arr[i] = 3.0;
            }
        }
    }

    let mut max_val: i32 = 0;
    let mut i = limit;
    while i >= 3 {
        for j in 0..=(limit - i) {
            let mut cur_ctr = -2.0f64;
            let mut mism = 0;
            for k in j..(j + i) {
                cur_ctr += match_arr[k];
                if match_arr[k] < 0.0 {
                    mism += 1;
                }
            }
            if mism > 0 {
                continue;
            }
            let rdis = start - (pos + (j + i) as i32);
            let dis_flag: i32;
            if rdis < 5 && (i as i32) < 5 {
                dis_flag = 2;
            } else if rdis < 5 && (i as i32) >= 5 {
                dis_flag = 1;
            } else if rdis > 10 && rdis <= 12 && (i as i32) < 5 {
                dis_flag = 1;
            } else if rdis > 10 && rdis <= 12 && (i as i32) >= 5 {
                dis_flag = 2;
            } else if rdis >= 13 {
                dis_flag = 3;
            } else {
                dis_flag = 0;
            }
            if rdis > 15 || cur_ctr < 6.0 {
                continue;
            }

            let cur_val;
            if cur_ctr < 6.0 {
                cur_val = 0;
            } else if cur_ctr == 6.0 && dis_flag == 2 {
                cur_val = 1;
            } else if cur_ctr == 6.0 && dis_flag == 3 {
                cur_val = 2;
            } else if cur_ctr == 8.0 && dis_flag == 3 {
                cur_val = 3;
            } else if cur_ctr == 9.0 && dis_flag == 3 {
                cur_val = 3;
            } else if cur_ctr == 6.0 && dis_flag == 1 {
                cur_val = 6;
            } else if cur_ctr == 11.0 && dis_flag == 3 {
                cur_val = 10;
            } else if cur_ctr == 12.0 && dis_flag == 3 {
                cur_val = 10;
            } else if cur_ctr == 14.0 && dis_flag == 3 {
                cur_val = 10;
            } else if cur_ctr == 8.0 && dis_flag == 2 {
                cur_val = 11;
            } else if cur_ctr == 9.0 && dis_flag == 2 {
                cur_val = 11;
            } else if cur_ctr == 8.0 && dis_flag == 1 {
                cur_val = 12;
            } else if cur_ctr == 9.0 && dis_flag == 1 {
                cur_val = 12;
            } else if cur_ctr == 6.0 && dis_flag == 0 {
                cur_val = 13;
            } else if cur_ctr == 8.0 && dis_flag == 0 {
                cur_val = 15;
            } else if cur_ctr == 9.0 && dis_flag == 0 {
                cur_val = 16;
            } else if cur_ctr == 11.0 && dis_flag == 2 {
                cur_val = 20;
            } else if cur_ctr == 11.0 && dis_flag == 1 {
                cur_val = 21;
            } else if cur_ctr == 11.0 && dis_flag == 0 {
                cur_val = 22;
            } else if cur_ctr == 12.0 && dis_flag == 2 {
                cur_val = 20;
            } else if cur_ctr == 12.0 && dis_flag == 1 {
                cur_val = 23;
            } else if cur_ctr == 12.0 && dis_flag == 0 {
                cur_val = 24;
            } else if cur_ctr == 14.0 && dis_flag == 2 {
                cur_val = 25;
            } else if cur_ctr == 14.0 && dis_flag == 1 {
                cur_val = 26;
            } else if cur_ctr == 14.0 && dis_flag == 0 {
                cur_val = 27;
            } else {
                cur_val = 0;
            }

            if rwt[cur_val as usize] < rwt[max_val as usize] {
                continue;
            }
            if rwt[cur_val as usize] == rwt[max_val as usize] && cur_val < max_val {
                continue;
            }
            max_val = cur_val;
        }
        if i == 0 {
            break;
        }
        i -= 1;
    }
    max_val
}

fn shine_dalgarno_mm(seq: &[u8], pos: i32, start: i32, rwt: &[f64; 28]) -> i32 {
    let limit = imin(6, start - 4 - pos) as usize;
    let mut match_arr = [-10.0f64; 6];

    for i in 0..limit {
        let pi = pos + i as i32;
        if pi >= 0 {
            let p = pi as usize;
            if i % 3 == 0 {
                if is_a(seq, p) {
                    match_arr[i] = 2.0;
                } else {
                    match_arr[i] = -3.0;
                }
            } else {
                if is_g(seq, p) {
                    match_arr[i] = 3.0;
                } else {
                    match_arr[i] = -2.0;
                }
            }
        }
    }

    let mut max_val: i32 = 0;
    let mut i = limit;
    while i >= 5 {
        for j in 0..=(limit - i) {
            let mut cur_ctr = -2.0f64;
            let mut mism = 0;
            for k in j..(j + i) {
                cur_ctr += match_arr[k];
                if match_arr[k] < 0.0 {
                    mism += 1;
                    if k <= j + 1 || k >= j + i - 2 {
                        cur_ctr -= 10.0;
                    }
                }
            }
            if mism != 1 {
                continue;
            }
            let rdis = start - (pos + (j + i) as i32);
            let dis_flag: i32;
            if rdis < 5 {
                dis_flag = 1;
            } else if rdis > 10 && rdis <= 12 {
                dis_flag = 2;
            } else if rdis >= 13 {
                dis_flag = 3;
            } else {
                dis_flag = 0;
            }
            if rdis > 15 || cur_ctr < 6.0 {
                continue;
            }

            let cur_val;
            if cur_ctr < 6.0 {
                cur_val = 0;
            } else if cur_ctr == 6.0 && dis_flag == 3 {
                cur_val = 2;
            } else if cur_ctr == 7.0 && dis_flag == 3 {
                cur_val = 2;
            } else if cur_ctr == 9.0 && dis_flag == 3 {
                cur_val = 3;
            } else if cur_ctr == 6.0 && dis_flag == 2 {
                cur_val = 4;
            } else if cur_ctr == 6.0 && dis_flag == 1 {
                cur_val = 5;
            } else if cur_ctr == 6.0 && dis_flag == 0 {
                cur_val = 9;
            } else if cur_ctr == 7.0 && dis_flag == 2 {
                cur_val = 7;
            } else if cur_ctr == 7.0 && dis_flag == 1 {
                cur_val = 8;
            } else if cur_ctr == 7.0 && dis_flag == 0 {
                cur_val = 14;
            } else if cur_ctr == 9.0 && dis_flag == 2 {
                cur_val = 17;
            } else if cur_ctr == 9.0 && dis_flag == 1 {
                cur_val = 18;
            } else if cur_ctr == 9.0 && dis_flag == 0 {
                cur_val = 19;
            } else {
                cur_val = 0;
            }

            if rwt[cur_val as usize] < rwt[max_val as usize] {
                continue;
            }
            if rwt[cur_val as usize] == rwt[max_val as usize] && cur_val < max_val {
                continue;
            }
            max_val = cur_val;
        }
        if i == 0 {
            break;
        }
        i -= 1;
    }
    max_val
}

// ============================================================================
// Mask crossing check
// ============================================================================

fn cross_mask(x: i32, y: i32, mlist: &[Mask]) -> bool {
    for m in mlist {
        if y < m.begin || x > m.end {
            continue;
        }
        return true;
    }
    false
}

// ============================================================================
// Node operations (from node.c)
// ============================================================================

fn add_nodes(
    seq: &[u8],
    rseq: &[u8],
    slen: usize,
    nodes: &mut Vec<Node>,
    closed: bool,
    mlist: &[Mask],
    tt: i32,
) -> usize {
    nodes.clear();
    let slen_i = slen as i32;
    let slmod = (slen % 3) as i32;

    // Build codon classification lookup table for this translation table
    let ct = build_codon_table(tt);

    // Forward strand
    let mut last = [0i32; 3];
    let mut saw_start = [false; 3];
    let mut min_dist = [MIN_EDGE_GENE; 3];

    for i in 0..3i32 {
        last[((i + slmod) % 3) as usize] = slen_i + i;
        saw_start[(i % 3) as usize] = false;
        min_dist[(i % 3) as usize] = MIN_EDGE_GENE;
        if !closed {
            while last[((i + slmod) % 3) as usize] + 2 > slen_i - 1 {
                last[((i + slmod) % 3) as usize] -= 3;
            }
        }
    }

    let mut i = slen_i - 3;
    while i >= 0 {
        let fr = (i % 3) as usize;
        let cv = codon_val(seq, i as usize);
        let cc = ct[cv as usize];
        if cc & CC_STOP != 0 {
            if saw_start[fr] {
                let mut n = Node::default();
                if ct[codon_val(seq, last[fr] as usize) as usize] & CC_STOP == 0 {
                    n.edge = 1;
                }
                n.ndx = last[fr];
                n.ntype = STOP;
                n.strand = 1;
                n.stop_val = i;
                nodes.push(n);
            }
            min_dist[fr] = MIN_GENE;
            last[fr] = i;
            saw_start[fr] = false;
            i -= 1;
            continue;
        }
        if last[fr] >= slen_i {
            i -= 1;
            continue;
        }

        let gene_len = last[fr] - i + 3;
        if cc & CC_ATG != 0 && gene_len >= min_dist[fr] && !cross_mask(i, last[fr], mlist) {
            let mut n = Node::default();
            n.ndx = i;
            n.ntype = ATG;
            saw_start[fr] = true;
            n.stop_val = last[fr];
            n.strand = 1;
            nodes.push(n);
        } else if cc & CC_GTG != 0 && gene_len >= min_dist[fr] && !cross_mask(i, last[fr], mlist) {
            let mut n = Node::default();
            n.ndx = i;
            n.ntype = GTG;
            saw_start[fr] = true;
            n.stop_val = last[fr];
            n.strand = 1;
            nodes.push(n);
        } else if cc & CC_TTG != 0 && gene_len >= min_dist[fr] && !cross_mask(i, last[fr], mlist) {
            let mut n = Node::default();
            n.ndx = i;
            n.ntype = TTG;
            saw_start[fr] = true;
            n.stop_val = last[fr];
            n.strand = 1;
            nodes.push(n);
        } else if i <= 2
            && !closed
            && (last[fr] - i) > MIN_EDGE_GENE
            && !cross_mask(i, last[fr], mlist)
        {
            let mut n = Node::default();
            n.ndx = i;
            n.ntype = ATG;
            saw_start[fr] = true;
            n.edge = 1;
            n.stop_val = last[fr];
            n.strand = 1;
            nodes.push(n);
        }
        i -= 1;
    }
    for i in 0..3 {
        if saw_start[i] {
            let mut n = Node::default();
            if ct[codon_val(seq, last[i] as usize) as usize] & CC_STOP == 0 {
                n.edge = 1;
            }
            n.ndx = last[i];
            n.ntype = STOP;
            n.strand = 1;
            n.stop_val = i as i32 - 6;
            nodes.push(n);
        }
    }

    // Reverse strand
    for i in 0..3i32 {
        last[((i + slmod) % 3) as usize] = slen_i + i;
        saw_start[(i % 3) as usize] = false;
        min_dist[(i % 3) as usize] = MIN_EDGE_GENE;
        if !closed {
            while last[((i + slmod) % 3) as usize] + 2 > slen_i - 1 {
                last[((i + slmod) % 3) as usize] -= 3;
            }
        }
    }

    let mut i = slen_i - 3;
    while i >= 0 {
        let fr = (i % 3) as usize;
        let cv = codon_val(rseq, i as usize);
        let cc = ct[cv as usize];
        if cc & CC_STOP != 0 {
            if saw_start[fr] {
                let mut n = Node::default();
                if ct[codon_val(rseq, last[fr] as usize) as usize] & CC_STOP == 0 {
                    n.edge = 1;
                }
                n.ndx = slen_i - last[fr] - 1;
                n.ntype = STOP;
                n.strand = -1;
                n.stop_val = slen_i - i - 1;
                nodes.push(n);
            }
            min_dist[fr] = MIN_GENE;
            last[fr] = i;
            saw_start[fr] = false;
            i -= 1;
            continue;
        }
        if last[fr] >= slen_i {
            i -= 1;
            continue;
        }

        let gene_len = last[fr] - i + 3;
        if cc & CC_ATG != 0
            && gene_len >= min_dist[fr]
            && !cross_mask(slen_i - last[fr] - 1, slen_i - i - 1, mlist)
        {
            let mut n = Node::default();
            n.ndx = slen_i - i - 1;
            n.ntype = ATG;
            saw_start[fr] = true;
            n.stop_val = slen_i - last[fr] - 1;
            n.strand = -1;
            nodes.push(n);
        } else if cc & CC_GTG != 0
            && gene_len >= min_dist[fr]
            && !cross_mask(slen_i - last[fr] - 1, slen_i - i - 1, mlist)
        {
            let mut n = Node::default();
            n.ndx = slen_i - i - 1;
            n.ntype = GTG;
            saw_start[fr] = true;
            n.stop_val = slen_i - last[fr] - 1;
            n.strand = -1;
            nodes.push(n);
        } else if cc & CC_TTG != 0
            && gene_len >= min_dist[fr]
            && !cross_mask(slen_i - last[fr] - 1, slen_i - i - 1, mlist)
        {
            let mut n = Node::default();
            n.ndx = slen_i - i - 1;
            n.ntype = TTG;
            saw_start[fr] = true;
            n.stop_val = slen_i - last[fr] - 1;
            n.strand = -1;
            nodes.push(n);
        } else if i <= 2
            && !closed
            && (last[fr] - i) > MIN_EDGE_GENE
            && !cross_mask(slen_i - last[fr] - 1, slen_i - i - 1, mlist)
        {
            let mut n = Node::default();
            n.ndx = slen_i - i - 1;
            n.ntype = ATG;
            saw_start[fr] = true;
            n.edge = 1;
            n.stop_val = slen_i - last[fr] - 1;
            n.strand = -1;
            nodes.push(n);
        }
        i -= 1;
    }
    for i in 0..3 {
        if saw_start[i] {
            let mut n = Node::default();
            if !is_stop(rseq, last[i] as usize, tt) {
                n.edge = 1;
            }
            n.ndx = slen_i - last[i] - 1;
            n.ntype = STOP;
            n.strand = -1;
            n.stop_val = slen_i - i as i32 + 5;
            nodes.push(n);
        }
    }

    // Sort nodes by (ndx asc, strand desc)
    nodes.sort_by(|a, b| a.ndx.cmp(&b.ndx).then_with(|| b.strand.cmp(&a.strand)));

    nodes.len()
}

fn reset_node_scores(nodes: &mut [Node]) {
    for n in nodes.iter_mut() {
        n.star_ptr = [0; 3];
        n.gc_score = [0.0; 3];
        n.rbs = [0; 2];
        n.score = 0.0;
        n.cscore = 0.0;
        n.sscore = 0.0;
        n.rscore = 0.0;
        n.tscore = 0.0;
        n.uscore = 0.0;
        n.traceb = -1;
        n.tracef = -1;
        n.ov_mark = -1;
        n.elim = 0;
        n.gc_bias = 0;
        n.mot = Motif::default();
    }
}

// ============================================================================
// record_overlapping_starts
// ============================================================================

fn record_overlapping_starts(nodes: &mut [Node], tinf: &Training, flag: i32) {
    let nn = nodes.len();
    for i in 0..nn {
        nodes[i].star_ptr = [-1, -1, -1];
        if nodes[i].ntype != STOP || nodes[i].edge == 1 {
            continue;
        }
        if nodes[i].strand == 1 {
            let mut max_sc = -100.0f64;
            let start_j = if i + 3 < nn { i + 3 } else { nn - 1 };
            let mut j = start_j as i32;
            while j >= 0 {
                let ju = j as usize;
                if ju >= nn || nodes[ju].ndx > nodes[i].ndx + 2 {
                    j -= 1;
                    continue;
                }
                if nodes[ju].ndx + MAX_SAM_OVLP < nodes[i].ndx {
                    break;
                }
                if nodes[ju].strand == 1 && nodes[ju].ntype != STOP {
                    if nodes[ju].stop_val <= nodes[i].ndx {
                        j -= 1;
                        continue;
                    }
                    let fr = (nodes[ju].ndx % 3) as usize;
                    if flag == 0 && nodes[i].star_ptr[fr] == -1 {
                        nodes[i].star_ptr[fr] = j;
                    } else if flag == 1 {
                        let sc = nodes[ju].cscore
                            + nodes[ju].sscore
                            + intergenic_mod_nodes(nodes, i, ju, tinf);
                        if sc > max_sc {
                            nodes[i].star_ptr[fr] = j;
                            max_sc = sc;
                        }
                    }
                }
                j -= 1;
            }
        } else {
            let mut max_sc = -100.0f64;
            let start_j = if i >= 3 { i - 3 } else { 0 };
            let mut j = start_j;
            while j < nn {
                if nodes[j].ndx < nodes[i].ndx - 2 {
                    j += 1;
                    continue;
                }
                if nodes[j].ndx - MAX_SAM_OVLP > nodes[i].ndx {
                    break;
                }
                if nodes[j].strand == -1 && nodes[j].ntype != STOP {
                    if nodes[j].stop_val >= nodes[i].ndx {
                        j += 1;
                        continue;
                    }
                    let fr = (nodes[j].ndx % 3) as usize;
                    if flag == 0 && nodes[i].star_ptr[fr] == -1 {
                        nodes[i].star_ptr[fr] = j as i32;
                    } else if flag == 1 {
                        let sc = nodes[j].cscore
                            + nodes[j].sscore
                            + intergenic_mod_nodes(nodes, j, i, tinf);
                        if sc > max_sc {
                            nodes[i].star_ptr[fr] = j as i32;
                            max_sc = sc;
                        }
                    }
                }
                j += 1;
            }
        }
    }
}

fn intergenic_mod_nodes(nodes: &[Node], i1: usize, i2: usize, tinf: &Training) -> f64 {
    intergenic_mod(&nodes[i1], &nodes[i2], tinf)
}

fn intergenic_mod(n1: &Node, n2: &Node, tinf: &Training) -> f64 {
    let mut rval = 0.0;
    let mut ovlp = 0.0;
    if (n1.strand == 1 && n2.strand == 1 && (n1.ndx + 2 == n2.ndx || n1.ndx - 1 == n2.ndx))
        || (n1.strand == -1 && n2.strand == -1 && (n1.ndx + 2 == n2.ndx || n1.ndx - 1 == n2.ndx))
    {
        if n1.strand == 1 && n2.rscore < 0.0 {
            rval -= n2.rscore;
        }
        if n1.strand == -1 && n1.rscore < 0.0 {
            rval -= n1.rscore;
        }
        if n1.strand == 1 && n2.uscore < 0.0 {
            rval -= n2.uscore;
        }
        if n1.strand == -1 && n1.uscore < 0.0 {
            rval -= n1.uscore;
        }
    }
    let dist = (n1.ndx - n2.ndx).abs();
    if n1.strand == 1 && n2.strand == 1 && n1.ndx + 2 >= n2.ndx {
        ovlp = 1.0;
    } else if n1.strand == -1 && n2.strand == -1 && n1.ndx >= n2.ndx + 2 {
        ovlp = 1.0;
    }
    if dist > 3 * OPER_DIST || n1.strand != n2.strand {
        rval -= 0.15 * tinf.st_wt;
    } else if (dist <= OPER_DIST && ovlp == 0.0) || dist < OPER_DIST / 4 {
        rval += (2.0 - (dist as f64) / OPER_DIST as f64) * 0.15 * tinf.st_wt;
    }
    rval
}

// ============================================================================
// GC frame bias
// ============================================================================

fn record_gc_bias(gc: &[i32], nodes: &mut [Node], tinf: &mut Training) {
    let nn = nodes.len();
    if nn == 0 {
        return;
    }
    let mut ctr = [[0i32; 3]; 3];
    let mut last = [0i32; 3];

    // Forward pass
    for i in (0..nn).rev() {
        let fr = (nodes[i].ndx % 3) as usize;
        let frmod = (3 - fr as i32) as usize;
        if nodes[i].strand == 1 && nodes[i].ntype == STOP {
            ctr[fr] = [0; 3];
            last[fr] = nodes[i].ndx;
            ctr[fr][(gc[nodes[i].ndx as usize] as usize + frmod) % 3] = 1;
        } else if nodes[i].strand == 1 {
            let mut j = last[fr] - 3;
            while j >= nodes[i].ndx {
                ctr[fr][(gc[j as usize] as usize + frmod) % 3] += 1;
                j -= 3;
            }
            let mfr = max_fr(ctr[fr][0], ctr[fr][1], ctr[fr][2]);
            nodes[i].gc_bias = mfr as i32;
            let denom = (nodes[i].stop_val - nodes[i].ndx + 3) as f64;
            for k in 0..3 {
                nodes[i].gc_score[k] = 3.0 * ctr[fr][k] as f64 / denom;
            }
            last[fr] = nodes[i].ndx;
        }
    }

    // Reverse pass
    ctr = [[0; 3]; 3];
    for i in 0..nn {
        let fr = (nodes[i].ndx % 3) as usize;
        let frmod = fr;
        if nodes[i].strand == -1 && nodes[i].ntype == STOP {
            ctr[fr] = [0; 3];
            last[fr] = nodes[i].ndx;
            ctr[fr][((3 - gc[nodes[i].ndx as usize]) as usize + frmod) % 3] = 1;
        } else if nodes[i].strand == -1 {
            let mut j = last[fr] + 3;
            while j <= nodes[i].ndx {
                ctr[fr][((3 - gc[j as usize]) as usize + frmod) % 3] += 1;
                j += 3;
            }
            let mfr = max_fr(ctr[fr][0], ctr[fr][1], ctr[fr][2]);
            nodes[i].gc_bias = mfr as i32;
            let denom = (nodes[i].ndx - nodes[i].stop_val + 3) as f64;
            for k in 0..3 {
                nodes[i].gc_score[k] = 3.0 * ctr[fr][k] as f64 / denom;
            }
            last[fr] = nodes[i].ndx;
        }
    }

    // Compute bias
    for i in 0..3 {
        tinf.bias[i] = 0.0;
    }
    for i in 0..nn {
        if nodes[i].ntype != STOP {
            let len = (nodes[i].stop_val - nodes[i].ndx).abs() + 1;
            tinf.bias[nodes[i].gc_bias as usize] +=
                nodes[i].gc_score[nodes[i].gc_bias as usize] * len as f64 / 1000.0;
        }
    }
    let tot: f64 = tinf.bias.iter().sum();
    if tot > 0.0 {
        for i in 0..3 {
            tinf.bias[i] *= 3.0 / tot;
        }
    }
}

// ============================================================================
// Coding score
// ============================================================================

fn calc_orf_gc(seq: &[u8], _rseq: &[u8], _slen: usize, nodes: &mut [Node], _tinf: &Training) {
    let nn = nodes.len();
    let mut gc = [0.0f64; 3];
    let mut last = [0i32; 3];

    // Forward
    for i in (0..nn).rev() {
        let fr = (nodes[i].ndx % 3) as usize;
        if nodes[i].strand == 1 && nodes[i].ntype == STOP {
            last[fr] = nodes[i].ndx;
            gc[fr] = (is_gc(seq, nodes[i].ndx as usize)
                + is_gc(seq, (nodes[i].ndx + 1) as usize)
                + is_gc(seq, (nodes[i].ndx + 2) as usize)) as f64;
        } else if nodes[i].strand == 1 {
            let mut j = last[fr] - 3;
            while j >= nodes[i].ndx {
                gc[fr] += (is_gc(seq, j as usize)
                    + is_gc(seq, (j + 1) as usize)
                    + is_gc(seq, (j + 2) as usize)) as f64;
                j -= 3;
            }
            let gsize = ((nodes[i].stop_val - nodes[i].ndx).abs() + 3) as f64;
            nodes[i].gc_cont = gc[fr] / gsize;
            last[fr] = nodes[i].ndx;
        }
    }

    // Reverse
    gc = [0.0; 3];
    for i in 0..nn {
        let fr = (nodes[i].ndx % 3) as usize;
        if nodes[i].strand == -1 && nodes[i].ntype == STOP {
            last[fr] = nodes[i].ndx;
            gc[fr] = (is_gc(seq, nodes[i].ndx as usize)
                + is_gc(seq, (nodes[i].ndx - 1) as usize)
                + is_gc(seq, (nodes[i].ndx - 2) as usize)) as f64;
        } else if nodes[i].strand == -1 {
            let mut j = last[fr] + 3;
            while j <= nodes[i].ndx {
                gc[fr] += (is_gc(seq, j as usize)
                    + is_gc(seq, (j + 1) as usize)
                    + is_gc(seq, (j + 2) as usize)) as f64;
                j += 3;
            }
            let gsize = ((nodes[i].stop_val - nodes[i].ndx).abs() + 3) as f64;
            nodes[i].gc_cont = gc[fr] / gsize;
            last[fr] = nodes[i].ndx;
        }
    }
}

fn raw_coding_score(seq: &[u8], rseq: &[u8], slen: usize, nodes: &mut [Node], tinf: &Training) {
    let nn = nodes.len();
    let no_stop: f64;
    if tinf.trans_table != 11 {
        let mut ns = ((1.0 - tinf.gc) * (1.0 - tinf.gc) * tinf.gc) / 8.0;
        ns += ((1.0 - tinf.gc) * (1.0 - tinf.gc) * (1.0 - tinf.gc)) / 8.0;
        no_stop = 1.0 - ns;
    } else {
        let mut ns = ((1.0 - tinf.gc) * (1.0 - tinf.gc) * tinf.gc) / 4.0;
        ns += ((1.0 - tinf.gc) * (1.0 - tinf.gc) * (1.0 - tinf.gc)) / 8.0;
        no_stop = 1.0 - ns;
    }

    // Initial pass: coding potential
    let mut score = [0.0f64; 3];
    let mut last = [0i32; 3];
    for i in (0..nn).rev() {
        let fr = (nodes[i].ndx % 3) as usize;
        if nodes[i].strand == 1 && nodes[i].ntype == STOP {
            last[fr] = nodes[i].ndx;
            score[fr] = 0.0;
        } else if nodes[i].strand == 1 {
            let mut j = last[fr] - 3;
            while j >= nodes[i].ndx {
                score[fr] += tinf.gene_dc[mer_ndx(6, seq, j as usize)];
                j -= 3;
            }
            nodes[i].cscore = score[fr];
            last[fr] = nodes[i].ndx;
        }
    }
    score = [0.0; 3];
    for i in 0..nn {
        let fr = (nodes[i].ndx % 3) as usize;
        if nodes[i].strand == -1 && nodes[i].ntype == STOP {
            last[fr] = nodes[i].ndx;
            score[fr] = 0.0;
        } else if nodes[i].strand == -1 {
            let mut j = last[fr] + 3;
            while j <= nodes[i].ndx {
                score[fr] += tinf.gene_dc[mer_ndx(6, rseq, (slen as i32 - j - 1) as usize)];
                j += 3;
            }
            nodes[i].cscore = score[fr];
            last[fr] = nodes[i].ndx;
        }
    }

    // Second pass: penalize ascending coding
    score = [-10000.0; 3];
    for i in 0..nn {
        let fr = (nodes[i].ndx % 3) as usize;
        if nodes[i].strand == 1 && nodes[i].ntype == STOP {
            score[fr] = -10000.0;
        } else if nodes[i].strand == 1 {
            if nodes[i].cscore > score[fr] {
                score[fr] = nodes[i].cscore;
            } else {
                nodes[i].cscore -= score[fr] - nodes[i].cscore;
            }
        }
    }
    score = [-10000.0; 3];
    for i in (0..nn).rev() {
        let fr = (nodes[i].ndx % 3) as usize;
        if nodes[i].strand == -1 && nodes[i].ntype == STOP {
            score[fr] = -10000.0;
        } else if nodes[i].strand == -1 {
            if nodes[i].cscore > score[fr] {
                score[fr] = nodes[i].cscore;
            } else {
                nodes[i].cscore -= score[fr] - nodes[i].cscore;
            }
        }
    }

    // Third pass: length factor (precompute expensive powf values)
    let ns80 = no_stop.powf(80.0);
    let ns1000 = no_stop.powf(1000.0);
    let l80 = ((1.0 - ns80) / ns80).ln();
    let l1000 = ((1.0 - ns1000) / ns1000).ln();
    let l1000_minus_l80 = l1000 - l80;

    for i in 0..nn {
        let fr = (nodes[i].ndx % 3) as usize;
        if nodes[i].strand == 1 && nodes[i].ntype == STOP {
            score[fr] = -10000.0;
        } else if nodes[i].strand == 1 {
            let gsize = ((nodes[i].stop_val - nodes[i].ndx).abs() + 3) as f64 / 3.0;
            let lfac;
            if gsize > 1000.0 {
                lfac = l1000_minus_l80 * (gsize - 80.0) / 920.0;
            } else {
                let nsg = no_stop.powf(gsize);
                lfac = ((1.0 - nsg) / nsg).ln() - l80;
            }
            let mut lf = lfac;
            if lf > score[fr] {
                score[fr] = lf;
            } else {
                lf -= dmax(dmin(score[fr] - lf, lf), 0.0);
            }
            if lf > 3.0 && nodes[i].cscore < 0.5 * lf {
                nodes[i].cscore = 0.5 * lf;
            }
            nodes[i].cscore += lf;
        }
    }
    for i in (0..nn).rev() {
        let fr = (nodes[i].ndx % 3) as usize;
        if nodes[i].strand == -1 && nodes[i].ntype == STOP {
            score[fr] = -10000.0;
        } else if nodes[i].strand == -1 {
            let gsize = ((nodes[i].stop_val - nodes[i].ndx).abs() + 3) as f64 / 3.0;
            let lfac;
            if gsize > 1000.0 {
                lfac = l1000_minus_l80 * (gsize - 80.0) / 920.0;
            } else {
                let nsg = no_stop.powf(gsize);
                lfac = ((1.0 - nsg) / nsg).ln() - l80;
            }
            let mut lf = lfac;
            if lf > score[fr] {
                score[fr] = lf;
            } else {
                lf -= dmax(dmin(score[fr] - lf, lf), 0.0);
            }
            if lf > 3.0 && nodes[i].cscore < 0.5 * lf {
                nodes[i].cscore = 0.5 * lf;
            }
            nodes[i].cscore += lf;
        }
    }
}

// ============================================================================
// RBS scoring
// ============================================================================

fn rbs_score(seq: &[u8], rseq: &[u8], slen: usize, nodes: &mut [Node], tinf: &Training) {
    let nn = nodes.len();
    for i in 0..nn {
        if nodes[i].ntype == STOP || nodes[i].edge == 1 {
            continue;
        }
        nodes[i].rbs = [0, 0];
        if nodes[i].strand == 1 {
            for j in (nodes[i].ndx - 20)..=(nodes[i].ndx - 6) {
                if j < 0 {
                    continue;
                }
                let e = shine_dalgarno_exact(seq, j, nodes[i].ndx, &tinf.rbs_wt);
                let m = shine_dalgarno_mm(seq, j, nodes[i].ndx, &tinf.rbs_wt);
                if e > nodes[i].rbs[0] {
                    nodes[i].rbs[0] = e;
                }
                if m > nodes[i].rbs[1] {
                    nodes[i].rbs[1] = m;
                }
            }
        } else {
            let slen_i = slen as i32;
            for j in (slen_i - nodes[i].ndx - 21)..=(slen_i - nodes[i].ndx - 7) {
                if j > slen_i - 1 {
                    continue;
                }
                let e = shine_dalgarno_exact(rseq, j, slen_i - 1 - nodes[i].ndx, &tinf.rbs_wt);
                let m = shine_dalgarno_mm(rseq, j, slen_i - 1 - nodes[i].ndx, &tinf.rbs_wt);
                if e > nodes[i].rbs[0] {
                    nodes[i].rbs[0] = e;
                }
                if m > nodes[i].rbs[1] {
                    nodes[i].rbs[1] = m;
                }
            }
        }
    }
}

// ============================================================================
// Upstream composition scoring
// ============================================================================

fn score_upstream_composition(seq: &[u8], slen: usize, nod: &mut Node, tinf: &Training) {
    let start = if nod.strand == 1 {
        nod.ndx
    } else {
        slen as i32 - 1 - nod.ndx
    };
    nod.uscore = 0.0;
    let mut count = 0usize;
    for i in 1..45i32 {
        if i > 2 && i < 15 {
            continue;
        }
        if start - i >= 0 {
            nod.uscore +=
                0.4 * tinf.st_wt * tinf.ups_comp[count][mer_ndx(1, seq, (start - i) as usize)];
        }
        count += 1;
    }
}

// ============================================================================
// Upstream motif finding (for non-SD organisms)
// ============================================================================

fn find_best_upstream_motif(
    tinf: &Training,
    seq: &[u8],
    rseq: &[u8],
    slen: usize,
    nod: &mut Node,
    stage: i32,
) {
    if nod.ntype == STOP || nod.edge == 1 {
        return;
    }
    let wseq = if nod.strand == 1 { seq } else { rseq };
    let start = if nod.strand == 1 {
        nod.ndx
    } else {
        slen as i32 - 1 - nod.ndx
    };

    let mut max_sc = -100.0f64;
    let mut max_spacendx = 0i32;
    let mut max_spacer = 0i32;
    let mut max_ndx = 0i32;
    let mut max_len = 0i32;

    for i in (0..=3i32).rev() {
        for j in (start - 18 - i)..=(start - 6 - i) {
            if j < 0 {
                continue;
            }
            let spacer = start - j - i - 3;
            let spacendx;
            if j <= start - 16 - i {
                spacendx = 3;
            } else if j <= start - 14 - i {
                spacendx = 2;
            } else if j >= start - 7 - i {
                spacendx = 1;
            } else {
                spacendx = 0;
            }
            let index = mer_ndx((i + 3) as usize, wseq, j as usize);
            let sc = tinf.mot_wt[i as usize][spacendx as usize][index];
            if sc > max_sc {
                max_sc = sc;
                max_spacendx = spacendx;
                max_spacer = spacer;
                max_ndx = index as i32;
                max_len = i + 3;
            }
        }
    }

    if stage == 2 && (max_sc == -4.0 || max_sc < tinf.no_mot + 0.69) {
        nod.mot = Motif {
            ndx: 0,
            len: 0,
            spacendx: 0,
            spacer: 0,
            score: tinf.no_mot,
        };
    } else {
        nod.mot = Motif {
            ndx: max_ndx,
            len: max_len,
            spacendx: max_spacendx,
            spacer: max_spacer,
            score: max_sc,
        };
    }
}

// ============================================================================
// score_nodes - master scoring function
// ============================================================================

fn score_nodes(
    seq: &[u8],
    rseq: &[u8],
    slen: usize,
    nodes: &mut [Node],
    tinf: &Training,
    closed: bool,
    is_meta: bool,
) {
    let nn = nodes.len();
    let slen_i = slen as i32;

    // Step 1: coding potential
    calc_orf_gc(seq, rseq, slen, nodes, tinf);
    raw_coding_score(seq, rseq, slen, nodes, tinf);

    // Step 2: RBS
    if tinf.uses_sd == 1 {
        rbs_score(seq, rseq, slen, nodes, tinf);
    } else {
        for i in 0..nn {
            if nodes[i].ntype == STOP || nodes[i].edge == 1 {
                continue;
            }
            find_best_upstream_motif(tinf, seq, rseq, slen, &mut nodes[i], 2);
        }
    }

    // Step 3: Score start nodes
    for i in 0..nn {
        if nodes[i].ntype == STOP {
            continue;
        }

        let mut edge_gene = 0.0f64;
        if nodes[i].edge == 1 {
            edge_gene += 1.0;
        }
        if (nodes[i].strand == 1 && !is_stop(seq, nodes[i].stop_val as usize, tinf.trans_table))
            || (nodes[i].strand == -1
                && !is_stop(
                    rseq,
                    (slen_i - 1 - nodes[i].stop_val) as usize,
                    tinf.trans_table,
                ))
        {
            edge_gene += 1.0;
        }

        if nodes[i].edge == 1 {
            nodes[i].tscore = EDGE_BONUS * tinf.st_wt / edge_gene;
            nodes[i].uscore = 0.0;
            nodes[i].rscore = 0.0;
        } else {
            // Type score
            nodes[i].tscore = tinf.type_wt[nodes[i].ntype as usize] * tinf.st_wt;

            // RBS score
            let rbs1 = tinf.rbs_wt[nodes[i].rbs[0] as usize];
            let rbs2 = tinf.rbs_wt[nodes[i].rbs[1] as usize];
            let sd_score = dmax(rbs1, rbs2) * tinf.st_wt;
            if tinf.uses_sd == 1 {
                nodes[i].rscore = sd_score;
            } else {
                nodes[i].rscore = tinf.st_wt * nodes[i].mot.score;
                if nodes[i].rscore < sd_score && tinf.no_mot > -0.5 {
                    nodes[i].rscore = sd_score;
                }
            }

            // Upstream score
            if nodes[i].strand == 1 {
                score_upstream_composition(seq, slen, &mut nodes[i], tinf);
            } else {
                score_upstream_composition(rseq, slen, &mut nodes[i], tinf);
            }

            // Edge penalties
            if !closed && nodes[i].ndx <= 2 && nodes[i].strand == 1 {
                nodes[i].uscore += EDGE_UPS * tinf.st_wt;
            } else if !closed && nodes[i].ndx >= slen_i - 3 && nodes[i].strand == -1 {
                nodes[i].uscore += EDGE_UPS * tinf.st_wt;
            } else if i < 500 && nodes[i].strand == 1 {
                for j in (0..i).rev() {
                    if nodes[j].edge == 1 && nodes[i].stop_val == nodes[j].stop_val {
                        nodes[i].uscore += EDGE_UPS * tinf.st_wt;
                        break;
                    }
                }
            } else if nn >= 500 && i >= nn - 500 && nodes[i].strand == -1 {
                for j in (i + 1)..nn {
                    if nodes[j].edge == 1 && nodes[i].stop_val == nodes[j].stop_val {
                        nodes[i].uscore += EDGE_UPS * tinf.st_wt;
                        break;
                    }
                }
            }
        }

        // Convert edge starts
        if ((nodes[i].ndx <= 2 && nodes[i].strand == 1)
            || (nodes[i].ndx >= slen_i - 3 && nodes[i].strand == -1))
            && nodes[i].edge == 0
            && !closed
        {
            edge_gene += 1.0;
            nodes[i].edge = 1;
            nodes[i].tscore = 0.0;
            nodes[i].uscore = EDGE_BONUS * tinf.st_wt / edge_gene;
            nodes[i].rscore = 0.0;
        }

        // Penalize starts with no stop codon
        if nodes[i].edge == 0 && edge_gene == 1.0 {
            nodes[i].uscore -= 0.5 * EDGE_BONUS * tinf.st_wt;
        }

        // Penalize short genes
        let gene_len = (nodes[i].ndx - nodes[i].stop_val).abs();
        if edge_gene == 0.0 && gene_len < 250 {
            let negf = 250.0 / gene_len as f64;
            let posf = gene_len as f64 / 250.0;
            if nodes[i].rscore < 0.0 {
                nodes[i].rscore *= negf;
            }
            if nodes[i].uscore < 0.0 {
                nodes[i].uscore *= negf;
            }
            if nodes[i].tscore < 0.0 {
                nodes[i].tscore *= negf;
            }
            if nodes[i].rscore > 0.0 {
                nodes[i].rscore *= posf;
            }
            if nodes[i].uscore > 0.0 {
                nodes[i].uscore *= posf;
            }
            if nodes[i].tscore > 0.0 {
                nodes[i].tscore *= posf;
            }
        }

        // Meta penalization (not used in single mode but included for correctness)
        if is_meta && slen < 3000 && edge_gene == 0.0 && (nodes[i].cscore < 5.0 || gene_len < 120) {
            nodes[i].cscore -= META_PEN * dmax(0.0, (3000 - slen) as f64 / 2700.0);
        }

        // Base start score
        nodes[i].sscore = nodes[i].tscore + nodes[i].rscore + nodes[i].uscore;

        // Penalize starts if coding is negative
        if nodes[i].cscore < 0.0 {
            if edge_gene > 0.0 && nodes[i].edge == 0 {
                if !is_meta || slen > 1500 {
                    nodes[i].sscore -= tinf.st_wt;
                } else {
                    nodes[i].sscore -= 10.31 - 0.004 * slen as f64;
                }
            } else if is_meta && slen < 3000 && nodes[i].edge == 1 {
                let min_meta_len = (slen as f64).sqrt() * 5.0;
                if gene_len as f64 >= min_meta_len {
                    if nodes[i].cscore >= 0.0 {
                        nodes[i].cscore = -1.0;
                    }
                    nodes[i].sscore = 0.0;
                    nodes[i].uscore = 0.0;
                }
            } else {
                nodes[i].sscore -= 0.5;
            }
        } else if nodes[i].cscore < 5.0 && is_meta && gene_len < 120 && nodes[i].sscore < 0.0 {
            nodes[i].sscore -= tinf.st_wt;
        }
    }
}

// ============================================================================
// Dynamic programming (from dprog.c)
// ============================================================================

fn dprog(nodes: &mut [Node], tinf: &Training, flag: i32) -> i32 {
    let nn = nodes.len();
    if nn == 0 {
        return -1;
    }

    // Pre-classify nodes: 0=fwd_start, 1=fwd_stop, 2=rev_stop, 3=rev_start
    let cats: Vec<u8> = nodes
        .iter()
        .map(|n| match (n.strand, n.ntype) {
            (1, STOP) => 1,
            (1, _) => 0,
            (-1, STOP) => 2,
            _ => 3,
        })
        .collect();

    // Validity lookup: VALID[cat_j][cat_i] = can j connect to i?
    // Derived from score_connection's early return rules:
    //   rule1: both non-STOP + same strand → invalid
    //   rule2: fwd start → rev anything → invalid
    //   rule3: rev stop → fwd anything → invalid
    //   rule4: rev start → fwd stop → invalid
    const VALID: [[bool; 4]; 4] = [
        // n2=fwd_start, fwd_stop, rev_stop, rev_start
        [false, true, false, false], // n1=fwd_start: only gene (→fwd_stop) valid
        [true, true, true, true],    // n1=fwd_stop: all can connect (traceb check dynamic)
        [false, false, true, true],  // n1=rev_stop: only rev_stop, rev_start valid
        [true, false, true, false],  // n1=rev_start: fwd_start, rev_stop valid
    ];

    // `alive` tracks which nodes are eligible as source (p1) in score_connection.
    // Categories 0 (fwd_start) and 2 (rev_stop) are always eligible.
    // Categories 1 (fwd_stop) and 3 (rev_start) require traceb != -1.
    let mut alive = vec![false; nn];
    for i in 0..nn {
        let cat = unsafe { *cats.get_unchecked(i) };
        if cat == 0 || cat == 2 {
            alive[i] = true;
        }
    }

    for n in nodes.iter_mut() {
        n.score = 0.0;
        n.traceb = -1;
        n.tracef = -1;
    }

    for i in 0..nn {
        let mut min = if i < MAX_NODE_DIST {
            0
        } else {
            i - MAX_NODE_DIST
        };
        if nodes[i].strand == -1 && nodes[i].ntype != STOP && nodes[min].ndx >= nodes[i].stop_val {
            while min > 0 && nodes[min].ndx >= nodes[i].stop_val {
                min -= 1;
            }
        }
        if nodes[i].strand == 1 && nodes[i].ntype == STOP && nodes[min].ndx >= nodes[i].stop_val {
            while min > 0 && nodes[min].ndx >= nodes[i].stop_val {
                min -= 1;
            }
        }
        if min < MAX_NODE_DIST {
            min = 0;
        } else {
            min -= MAX_NODE_DIST;
        }

        let cat_i = unsafe { *cats.get_unchecked(i) };
        for j in min..i {
            let cat_j = unsafe { *cats.get_unchecked(j) };
            // Fast category + alive check on compact arrays (no Node struct access)
            if !VALID[cat_j as usize][cat_i as usize] {
                continue;
            }
            if unsafe { !*alive.get_unchecked(j) } {
                continue;
            }
            score_connection(nodes, j, i, tinf, flag);
        }
        // Update alive status for node i after it may have received a connection
        let cat = unsafe { *cats.get_unchecked(i) };
        if cat == 1 || cat == 3 {
            unsafe {
                *alive.get_unchecked_mut(i) = nodes.get_unchecked(i).traceb != -1;
            }
        }
    }

    let mut max_sc = -1.0f64;
    let mut max_ndx: i32 = -1;
    for i in (0..nn).rev() {
        if nodes[i].strand == 1 && nodes[i].ntype != STOP {
            continue;
        }
        if nodes[i].strand == -1 && nodes[i].ntype == STOP {
            continue;
        }
        if nodes[i].score > max_sc {
            max_sc = nodes[i].score;
            max_ndx = i as i32;
        }
    }

    if max_ndx < 0 {
        return -1;
    }

    // First pass: untangle triple overlaps
    let mut path = max_ndx;
    while nodes[path as usize].traceb != -1 {
        let nxt = nodes[path as usize].traceb;
        if nodes[path as usize].strand == -1
            && nodes[path as usize].ntype == STOP
            && nodes[nxt as usize].strand == 1
            && nodes[nxt as usize].ntype == STOP
            && nodes[path as usize].ov_mark != -1
            && nodes[path as usize].ndx > nodes[nxt as usize].ndx
        {
            let tmp = nodes[path as usize].star_ptr[nodes[path as usize].ov_mark as usize];
            let mut ii = tmp;
            while nodes[ii as usize].ndx != nodes[tmp as usize].stop_val {
                ii -= 1;
            }
            nodes[path as usize].traceb = tmp;
            nodes[tmp as usize].traceb = ii;
            nodes[ii as usize].ov_mark = -1;
            nodes[ii as usize].traceb = nxt;
        }
        path = nodes[path as usize].traceb;
    }

    // Second pass: untangle simple overlaps
    path = max_ndx;
    while nodes[path as usize].traceb != -1 {
        let nxt = nodes[path as usize].traceb;
        if nodes[path as usize].strand == -1
            && nodes[path as usize].ntype != STOP
            && nodes[nxt as usize].strand == 1
            && nodes[nxt as usize].ntype == STOP
        {
            let mut ii = path;
            while nodes[ii as usize].ndx != nodes[path as usize].stop_val {
                ii -= 1;
            }
            nodes[path as usize].traceb = ii;
            nodes[ii as usize].traceb = nxt;
        }
        if nodes[path as usize].strand == 1
            && nodes[path as usize].ntype == STOP
            && nodes[nxt as usize].strand == 1
            && nodes[nxt as usize].ntype == STOP
        {
            let sp = nodes[nxt as usize].star_ptr[(nodes[path as usize].ndx % 3) as usize];
            nodes[path as usize].traceb = sp;
            nodes[sp as usize].traceb = nxt;
        }
        if nodes[path as usize].strand == -1
            && nodes[path as usize].ntype == STOP
            && nodes[nxt as usize].strand == -1
            && nodes[nxt as usize].ntype == STOP
        {
            let sp = nodes[path as usize].star_ptr[(nodes[nxt as usize].ndx % 3) as usize];
            nodes[path as usize].traceb = sp;
            nodes[sp as usize].traceb = nxt;
        }
        path = nodes[path as usize].traceb;
    }

    // Mark forward pointers
    path = max_ndx;
    while nodes[path as usize].traceb != -1 {
        let tb = nodes[path as usize].traceb;
        nodes[tb as usize].tracef = path;
        path = tb;
    }

    if nodes[max_ndx as usize].traceb == -1 {
        -1
    } else {
        max_ndx
    }
}

#[inline(always)]
fn score_connection(nodes: &mut [Node], p1: usize, p2: usize, tinf: &Training, flag: i32) {
    // NOTE: category-based filtering + traceb checks in dprog() guarantee these are valid pairs.
    let (n1_strand, n1_type, n1_ndx, n1_stop, n1_score, n1_traceb) = unsafe {
        let n = nodes.get_unchecked(p1);
        (n.strand, n.ntype, n.ndx, n.stop_val, n.score, n.traceb)
    };
    let (n2_strand, n2_type, n2_ndx, n2_stop) = unsafe {
        let n = nodes.get_unchecked(p2);
        (n.strand, n.ntype, n.ndx, n.stop_val)
    };

    let mut left = n1_ndx;
    let mut right = n2_ndx;
    let mut score = 0.0f64;
    let mut scr_mod = 0.0f64;
    let mut ovlp = 0i32;
    let mut maxfr: i32 = -1;

    // 5'fwd->3'fwd (gene)
    if n1_strand == n2_strand && n1_strand == 1 && n1_type != STOP && n2_type == STOP {
        if n2_stop >= n1_ndx {
            return;
        }
        if n1_ndx % 3 != n2_ndx % 3 {
            return;
        }
        right += 2;
        if flag == 0 {
            scr_mod = tinf.bias[0] * nodes[p1].gc_score[0]
                + tinf.bias[1] * nodes[p1].gc_score[1]
                + tinf.bias[2] * nodes[p1].gc_score[2];
        } else {
            score = nodes[p1].cscore + nodes[p1].sscore;
        }
    }
    // 3'rev->5'rev
    else if n1_strand == n2_strand && n1_strand == -1 && n1_type == STOP && n2_type != STOP {
        if n1_stop <= n2_ndx {
            return;
        }
        if n1_ndx % 3 != n2_ndx % 3 {
            return;
        }
        left -= 2;
        if flag == 0 {
            scr_mod = tinf.bias[0] * nodes[p2].gc_score[0]
                + tinf.bias[1] * nodes[p2].gc_score[1]
                + tinf.bias[2] * nodes[p2].gc_score[2];
        } else {
            score = nodes[p2].cscore + nodes[p2].sscore;
        }
    }
    // 3'fwd->5'fwd
    else if n1_strand == 1 && n1_type == STOP && n2_strand == 1 && n2_type != STOP {
        left += 2;
        if left >= right {
            return;
        }
        if flag == 1 {
            score = intergenic_mod(&nodes[p1], &nodes[p2], tinf);
        }
    }
    // 3'fwd->3'rev
    else if n1_strand == 1 && n1_type == STOP && n2_strand == -1 && n2_type == STOP {
        left += 2;
        right -= 2;
        if left >= right {
            return;
        }
        // Triple overlap check
        let mut maxval = 0.0f64;
        for fr in 0..3 {
            let sp = nodes[p2].star_ptr[fr];
            if sp == -1 {
                continue;
            }
            let sp_u = sp as usize;
            let o = left - nodes[sp_u].stop_val + 3;
            if o <= 0 || o >= MAX_OPP_OVLP {
                continue;
            }
            if o >= nodes[sp_u].ndx - left {
                continue;
            }
            if n1_traceb == -1 {
                continue;
            }
            if o >= nodes[sp_u].stop_val - nodes[n1_traceb as usize].ndx - 2 {
                continue;
            }
            if flag == 1 {
                let s = nodes[sp_u].cscore
                    + nodes[sp_u].sscore
                    + intergenic_mod(&nodes[sp_u], &nodes[p2], tinf);
                if s > maxval {
                    maxfr = fr as i32;
                    maxval = s;
                }
            } else {
                let s = tinf.bias[0] * nodes[sp_u].gc_score[0]
                    + tinf.bias[1] * nodes[sp_u].gc_score[1]
                    + tinf.bias[2] * nodes[sp_u].gc_score[2];
                if s > maxval {
                    maxfr = fr as i32;
                    maxval = s;
                }
            }
        }
        if maxfr != -1 {
            let sp = nodes[p2].star_ptr[maxfr as usize] as usize;
            if flag == 0 {
                scr_mod = tinf.bias[0] * nodes[sp].gc_score[0]
                    + tinf.bias[1] * nodes[sp].gc_score[1]
                    + tinf.bias[2] * nodes[sp].gc_score[2];
            } else {
                score = nodes[sp].cscore
                    + nodes[sp].sscore
                    + intergenic_mod(&nodes[sp], &nodes[p2], tinf);
            }
        } else if flag == 1 {
            score = intergenic_mod(&nodes[p1], &nodes[p2], tinf);
        }
    }
    // 5'rev->3'rev
    else if n1_strand == -1 && n1_type != STOP && n2_strand == -1 && n2_type == STOP {
        right -= 2;
        if left >= right {
            return;
        }
        if flag == 1 {
            score = intergenic_mod(&nodes[p1], &nodes[p2], tinf);
        }
    }
    // 5'rev->5'fwd
    else if n1_strand == -1 && n1_type != STOP && n2_strand == 1 && n2_type != STOP {
        if left >= right {
            return;
        }
        if flag == 1 {
            score = intergenic_mod(&nodes[p1], &nodes[p2], tinf);
        }
    }
    // 3'fwd->3'fwd (operon)
    else if n1_strand == 1 && n2_strand == 1 && n1_type == STOP && n2_type == STOP {
        if n2_stop >= n1_ndx {
            return;
        }
        let sp_idx = (n2_ndx % 3) as usize;
        if nodes[p1].star_ptr[sp_idx] == -1 {
            return;
        }
        let sp = nodes[p1].star_ptr[sp_idx] as usize;
        left = nodes[sp].ndx;
        right += 2;
        if flag == 0 {
            scr_mod = tinf.bias[0] * nodes[sp].gc_score[0]
                + tinf.bias[1] * nodes[sp].gc_score[1]
                + tinf.bias[2] * nodes[sp].gc_score[2];
        } else {
            score =
                nodes[sp].cscore + nodes[sp].sscore + intergenic_mod(&nodes[p1], &nodes[sp], tinf);
        }
    }
    // 3'rev->3'rev (operon)
    else if n1_strand == -1 && n1_type == STOP && n2_strand == -1 && n2_type == STOP {
        if n1_stop <= n2_ndx {
            return;
        }
        let sp_idx = (n1_ndx % 3) as usize;
        if nodes[p2].star_ptr[sp_idx] == -1 {
            return;
        }
        let sp = nodes[p2].star_ptr[sp_idx] as usize;
        left -= 2;
        right = nodes[sp].ndx;
        if flag == 0 {
            scr_mod = tinf.bias[0] * nodes[sp].gc_score[0]
                + tinf.bias[1] * nodes[sp].gc_score[1]
                + tinf.bias[2] * nodes[sp].gc_score[2];
        } else {
            score =
                nodes[sp].cscore + nodes[sp].sscore + intergenic_mod(&nodes[sp], &nodes[p2], tinf);
        }
    }
    // 3'fwd->5'rev (opposite strand overlap)
    else if n1_strand == 1 && n1_type == STOP && n2_strand == -1 && n2_type != STOP {
        if n2_stop - 2 >= n1_ndx + 2 {
            return;
        }
        ovlp = (n1_ndx + 2) - (n2_stop - 2) + 1;
        if ovlp >= MAX_OPP_OVLP {
            return;
        }
        if (n1_ndx + 2 - n2_stop - 2 + 1) >= (n2_ndx - n1_ndx + 3 + 1) {
            return;
        }
        let bnd = if n1_traceb == -1 {
            0
        } else {
            nodes[n1_traceb as usize].ndx
        };
        if (n1_ndx + 2 - n2_stop - 2 + 1) >= (n2_stop - 3 - bnd + 1) {
            return;
        }
        left = n2_stop - 2;
        if flag == 0 {
            scr_mod = tinf.bias[0] * nodes[p2].gc_score[0]
                + tinf.bias[1] * nodes[p2].gc_score[1]
                + tinf.bias[2] * nodes[p2].gc_score[2];
        } else {
            score = nodes[p2].cscore + nodes[p2].sscore - 0.15 * tinf.st_wt;
        }
    } else {
        return;
    }

    if flag == 0 {
        score = (right - left + 1 - ovlp * 2) as f64 * scr_mod;
    }

    if n1_score + score >= unsafe { nodes.get_unchecked(p2).score } {
        unsafe {
            let n2 = nodes.get_unchecked_mut(p2);
            n2.score = n1_score + score;
            n2.traceb = p1 as i32;
            n2.ov_mark = maxfr;
        }
    }
}

fn eliminate_bad_genes(nodes: &mut [Node], dbeg: i32, tinf: &Training) {
    if dbeg == -1 {
        return;
    }

    // Find start of path
    let mut path = dbeg;
    while nodes[path as usize].traceb != -1 {
        path = nodes[path as usize].traceb;
    }

    // First sweep: add intergenic_mod to sscore
    while nodes[path as usize].tracef != -1 {
        let tf = nodes[path as usize].tracef;
        if nodes[path as usize].strand == 1 && nodes[path as usize].ntype == STOP {
            let igm = intergenic_mod(&nodes[path as usize], &nodes[tf as usize], tinf);
            nodes[tf as usize].sscore += igm;
        }
        if nodes[path as usize].strand == -1 && nodes[path as usize].ntype != STOP {
            let igm = intergenic_mod(&nodes[path as usize], &nodes[tf as usize], tinf);
            nodes[path as usize].sscore += igm;
        }
        path = tf;
    }

    // Second sweep: eliminate bad genes
    path = dbeg;
    while nodes[path as usize].traceb != -1 {
        path = nodes[path as usize].traceb;
    }
    while nodes[path as usize].tracef != -1 {
        let tf = nodes[path as usize].tracef;
        if nodes[path as usize].strand == 1
            && nodes[path as usize].ntype != STOP
            && nodes[path as usize].cscore + nodes[path as usize].sscore < 0.0
        {
            nodes[path as usize].elim = 1;
            nodes[tf as usize].elim = 1;
        }
        if nodes[path as usize].strand == -1
            && nodes[path as usize].ntype == STOP
            && nodes[tf as usize].cscore + nodes[tf as usize].sscore < 0.0
        {
            nodes[path as usize].elim = 1;
            nodes[tf as usize].elim = 1;
        }
        path = tf;
    }
}

// ============================================================================
// Gene extraction
// ============================================================================

fn add_genes(genes: &mut Vec<Gene>, nodes: &[Node], dbeg: i32) -> usize {
    genes.clear();
    if dbeg == -1 {
        return 0;
    }

    let mut path = dbeg;
    while nodes[path as usize].traceb != -1 {
        path = nodes[path as usize].traceb;
    }

    let mut ctr = 0usize;
    while path != -1 {
        let pu = path as usize;
        if nodes[pu].elim == 1 {
            path = nodes[pu].tracef;
            continue;
        }
        if nodes[pu].strand == 1 && nodes[pu].ntype != STOP {
            if ctr >= genes.len() {
                genes.push(Gene::default());
            }
            genes[ctr].begin = nodes[pu].ndx + 1;
            genes[ctr].start_ndx = pu;
        }
        if nodes[pu].strand == -1 && nodes[pu].ntype == STOP {
            if ctr >= genes.len() {
                genes.push(Gene::default());
            }
            genes[ctr].begin = nodes[pu].ndx - 1;
            genes[ctr].stop_ndx = pu;
        }
        if nodes[pu].strand == 1 && nodes[pu].ntype == STOP {
            if ctr >= genes.len() {
                genes.push(Gene::default());
            }
            genes[ctr].end = nodes[pu].ndx + 3;
            genes[ctr].stop_ndx = pu;
            ctr += 1;
        }
        if nodes[pu].strand == -1 && nodes[pu].ntype != STOP {
            if ctr >= genes.len() {
                genes.push(Gene::default());
            }
            genes[ctr].end = nodes[pu].ndx + 1;
            genes[ctr].start_ndx = pu;
            ctr += 1;
        }
        path = nodes[pu].tracef;
        if ctr >= MAX_GENES {
            break;
        }
    }
    genes.truncate(ctr);
    ctr
}

fn tweak_final_starts(genes: &mut [Gene], nodes: &mut [Node], tinf: &Training) {
    let ng = genes.len();
    let nn = nodes.len();

    for i in 0..ng {
        let ndx = genes[i].start_ndx;
        let sc = nodes[ndx].sscore + nodes[ndx].cscore;
        let mut igm = 0.0f64;

        if i > 0 && nodes[ndx].strand == 1 && nodes[genes[i - 1].start_ndx].strand == 1 {
            igm = intergenic_mod(&nodes[genes[i - 1].stop_ndx], &nodes[ndx], tinf);
        }
        if i > 0 && nodes[ndx].strand == 1 && nodes[genes[i - 1].start_ndx].strand == -1 {
            igm = intergenic_mod(&nodes[genes[i - 1].start_ndx], &nodes[ndx], tinf);
        }
        if i < ng - 1 && nodes[ndx].strand == -1 && nodes[genes[i + 1].start_ndx].strand == 1 {
            igm = intergenic_mod(&nodes[ndx], &nodes[genes[i + 1].start_ndx], tinf);
        }
        if i < ng - 1 && nodes[ndx].strand == -1 && nodes[genes[i + 1].start_ndx].strand == -1 {
            igm = intergenic_mod(&nodes[ndx], &nodes[genes[i + 1].stop_ndx], tinf);
        }

        let mut maxndx = [-1i32; 2];
        let mut maxsc = [0.0f64; 2];
        let mut maxigm = [0.0f64; 2];

        let lo = if ndx >= 100 { ndx - 100 } else { 0 };
        let hi = if ndx + 100 < nn { ndx + 100 } else { nn };
        for j in lo..hi {
            if j == ndx {
                continue;
            }
            if nodes[j].ntype == STOP || nodes[j].stop_val != nodes[ndx].stop_val {
                continue;
            }

            let mut tigm = 0.0f64;
            if i > 0 && nodes[j].strand == 1 && nodes[genes[i - 1].start_ndx].strand == 1 {
                if nodes[genes[i - 1].stop_ndx].ndx - nodes[j].ndx > MAX_SAM_OVLP {
                    continue;
                }
                tigm = intergenic_mod(&nodes[genes[i - 1].stop_ndx], &nodes[j], tinf);
            }
            if i > 0 && nodes[j].strand == 1 && nodes[genes[i - 1].start_ndx].strand == -1 {
                if nodes[genes[i - 1].start_ndx].ndx - nodes[j].ndx >= 0 {
                    continue;
                }
                tigm = intergenic_mod(&nodes[genes[i - 1].start_ndx], &nodes[j], tinf);
            }
            if i < ng - 1 && nodes[j].strand == -1 && nodes[genes[i + 1].start_ndx].strand == 1 {
                if nodes[j].ndx - nodes[genes[i + 1].start_ndx].ndx >= 0 {
                    continue;
                }
                tigm = intergenic_mod(&nodes[j], &nodes[genes[i + 1].start_ndx], tinf);
            }
            if i < ng - 1 && nodes[j].strand == -1 && nodes[genes[i + 1].start_ndx].strand == -1 {
                if nodes[j].ndx - nodes[genes[i + 1].stop_ndx].ndx > MAX_SAM_OVLP {
                    continue;
                }
                tigm = intergenic_mod(&nodes[j], &nodes[genes[i + 1].stop_ndx], tinf);
            }

            let jsc = nodes[j].cscore + nodes[j].sscore;
            if maxndx[0] == -1 {
                maxndx[0] = j as i32;
                maxsc[0] = jsc;
                maxigm[0] = tigm;
            } else if jsc + tigm > maxsc[0] {
                maxndx[1] = maxndx[0];
                maxsc[1] = maxsc[0];
                maxigm[1] = maxigm[0];
                maxndx[0] = j as i32;
                maxsc[0] = jsc;
                maxigm[0] = tigm;
            } else if maxndx[1] == -1 || jsc + tigm > maxsc[1] {
                maxndx[1] = j as i32;
                maxsc[1] = jsc;
                maxigm[1] = tigm;
            }
        }

        // Decide whether to change start
        for j in 0..2 {
            let mndx_i = maxndx[j];
            if mndx_i == -1 {
                continue;
            }
            let mu = mndx_i as usize;

            // Less common type with better everything
            if nodes[mu].tscore < nodes[ndx].tscore
                && maxsc[j] - nodes[mu].tscore >= sc - nodes[ndx].tscore + tinf.st_wt
                && nodes[mu].rscore > nodes[ndx].rscore
                && nodes[mu].uscore > nodes[ndx].uscore
                && nodes[mu].cscore > nodes[ndx].cscore
                && (nodes[mu].ndx - nodes[ndx].ndx).abs() > 15
            {
                maxsc[j] += nodes[ndx].tscore - nodes[mu].tscore;
            }
            // Close starts
            else if (nodes[mu].ndx - nodes[ndx].ndx).abs() <= 15
                && nodes[mu].rscore + nodes[mu].tscore > nodes[ndx].rscore + nodes[ndx].tscore
                && nodes[ndx].edge == 0
                && nodes[mu].edge == 0
            {
                if nodes[ndx].cscore > nodes[mu].cscore {
                    maxsc[j] += nodes[ndx].cscore - nodes[mu].cscore;
                }
                if nodes[ndx].uscore > nodes[mu].uscore {
                    maxsc[j] += nodes[ndx].uscore - nodes[mu].uscore;
                }
                if igm > maxigm[j] {
                    maxsc[j] += igm - maxigm[j];
                }
            } else {
                maxsc[j] = -1000.0;
            }
        }

        let mut best = -1i32;
        for j in 0..2 {
            if maxndx[j] == -1 {
                continue;
            }
            if best == -1 && maxsc[j] + maxigm[j] > sc + igm {
                best = j as i32;
            } else if best >= 0
                && maxsc[j] + maxigm[j] > maxsc[best as usize] + maxigm[best as usize]
            {
                best = j as i32;
            }
        }

        if best != -1 {
            let mu = maxndx[best as usize] as usize;
            if nodes[mu].strand == 1 {
                genes[i].start_ndx = mu;
                genes[i].begin = nodes[mu].ndx + 1;
            } else {
                genes[i].start_ndx = mu;
                genes[i].end = nodes[mu].ndx + 1;
            }
        }
    }
}

// ============================================================================
// Write nucleotide sequences (the output we need for chewBBACA)
// ============================================================================

/// SD motif names (indexed by rbs[0..1] values 0..27)
const SD_STRING: [&str; 28] = [
    "None",
    "GGA/GAG/AGG",
    "3Base/5BMM",
    "4Base/6BMM",
    "AGxAG",
    "AGxAG",
    "GGA/GAG/AGG",
    "GGxGG",
    "GGxGG",
    "AGxAG",
    "AGGAG(G)/GGAGG",
    "AGGA/GGAG/GAGG",
    "AGGA/GGAG/GAGG",
    "GGA/GAG/AGG",
    "GGxGG",
    "AGGA",
    "GGAG/GAGG",
    "AGxAGG/AGGxGG",
    "AGxAGG/AGGxGG",
    "AGxAGG/AGGxGG",
    "AGGAG/GGAGG",
    "AGGAG",
    "AGGAG",
    "GGAGG",
    "GGAGG",
    "AGGAGG",
    "AGGAGG",
    "AGGAGG",
];

/// SD spacer strings (indexed by rbs[0..1] values 0..27)
const SD_SPACER: [&str; 28] = [
    "None", "3-4bp", "13-15bp", "13-15bp", "11-12bp", "3-4bp", "11-12bp", "11-12bp", "3-4bp",
    "5-10bp", "13-15bp", "3-4bp", "11-12bp", "5-10bp", "5-10bp", "5-10bp", "5-10bp", "11-12bp",
    "3-4bp", "5-10bp", "11-12bp", "3-4bp", "5-10bp", "3-4bp", "5-10bp", "11-12bp", "3-4bp",
    "5-10bp",
];

/// Convert a motif index to a text string (like mer_text in C).
fn mer_text(len: i32, ndx: i32) -> String {
    if len == 0 {
        return "None".to_string();
    }
    let letters = [b'A', b'G', b'C', b'T'];
    let mut qt = Vec::with_capacity(len as usize);
    for i in 0..len {
        let val = ((ndx & (1 << (2 * i))) + (ndx & (1 << (2 * i + 1)))) >> (i * 2);
        qt.push(letters[val as usize]);
    }
    String::from_utf8(qt).unwrap()
}

/// Build gene_data string exactly matching C prodigal's record_gene_data().
fn build_gene_data(
    gene: &Gene,
    nodes: &[Node],
    tinf: &Training,
    gene_num: usize,
    num_seq: i32,
) -> String {
    let ndx = gene.start_ndx;
    let sndx = gene.stop_ndx;

    // partial flags
    let partial_left = if (nodes[ndx].edge == 1 && nodes[ndx].strand == 1)
        || (nodes[sndx].edge == 1 && nodes[ndx].strand == -1)
    {
        1
    } else {
        0
    };
    let partial_right = if (nodes[sndx].edge == 1 && nodes[ndx].strand == 1)
        || (nodes[ndx].edge == 1 && nodes[ndx].strand == -1)
    {
        1
    } else {
        0
    };

    // start type
    let st_type = if nodes[ndx].edge == 1 {
        3usize
    } else {
        nodes[ndx].ntype as usize
    };
    let type_strings = ["ATG", "GTG", "TTG", "Edge"];

    let mut data = format!(
        "ID={}_{};partial={}{};start_type={};",
        num_seq, gene_num, partial_left, partial_right, type_strings[st_type]
    );

    // RBS data
    let rbs1 = tinf.rbs_wt[nodes[ndx].rbs[0] as usize] * tinf.st_wt;
    let rbs2 = tinf.rbs_wt[nodes[ndx].rbs[1] as usize] * tinf.st_wt;

    if tinf.uses_sd == 1 {
        if rbs1 > rbs2 {
            data.push_str(&format!(
                "rbs_motif={};rbs_spacer={}",
                SD_STRING[nodes[ndx].rbs[0] as usize], SD_SPACER[nodes[ndx].rbs[0] as usize]
            ));
        } else {
            data.push_str(&format!(
                "rbs_motif={};rbs_spacer={}",
                SD_STRING[nodes[ndx].rbs[1] as usize], SD_SPACER[nodes[ndx].rbs[1] as usize]
            ));
        }
    } else {
        let qt = mer_text(nodes[ndx].mot.len, nodes[ndx].mot.ndx);
        if tinf.no_mot > -0.5 && rbs1 > rbs2 && rbs1 > nodes[ndx].mot.score * tinf.st_wt {
            data.push_str(&format!(
                "rbs_motif={};rbs_spacer={}",
                SD_STRING[nodes[ndx].rbs[0] as usize], SD_SPACER[nodes[ndx].rbs[0] as usize]
            ));
        } else if tinf.no_mot > -0.5 && rbs2 >= rbs1 && rbs2 > nodes[ndx].mot.score * tinf.st_wt {
            data.push_str(&format!(
                "rbs_motif={};rbs_spacer={}",
                SD_STRING[nodes[ndx].rbs[1] as usize], SD_SPACER[nodes[ndx].rbs[1] as usize]
            ));
        } else if nodes[ndx].mot.len == 0 {
            data.push_str("rbs_motif=None;rbs_spacer=None");
        } else {
            data.push_str(&format!(
                "rbs_motif={};rbs_spacer={}bp",
                qt, nodes[ndx].mot.spacer
            ));
        }
    }

    data.push_str(&format!(";gc_cont={:.3}", nodes[ndx].gc_cont));
    data
}

fn write_nucleotide_seqs(
    out: &mut Vec<u8>,
    genes: &[Gene],
    nodes: &[Node],
    seq: &[u8],
    rseq: &[u8],
    useq: &[u8],
    slen: usize,
    tinf: &Training,
    num_seq: i32,
    short_header: &str,
) {
    for (i, gene) in genes.iter().enumerate() {
        let gene_data = build_gene_data(gene, nodes, tinf, i + 1, num_seq);
        let ndx = gene.start_ndx;
        let strand_str = if nodes[ndx].strand == 1 { "1" } else { "-1" };

        let hdr = format!(
            ">{}_{} # {} # {} # {} # {}\n",
            short_header,
            i + 1,
            gene.begin,
            gene.end,
            strand_str,
            gene_data,
        );
        out.extend_from_slice(hdr.as_bytes());

        if nodes[ndx].strand == 1 {
            let mut col = 0;
            for j in (gene.begin - 1)..gene.end {
                let ju = j as usize;
                if is_a(seq, ju) {
                    out.push(b'A');
                } else if is_t(seq, ju) {
                    out.push(b'T');
                } else if is_g(seq, ju) {
                    out.push(b'G');
                } else if is_c(seq, ju) && !is_n(useq, ju) {
                    out.push(b'C');
                } else {
                    out.push(b'N');
                }
                col += 1;
                if col % 70 == 0 {
                    out.push(b'\n');
                }
            }
            if col % 70 != 0 {
                out.push(b'\n');
            }
        } else {
            let slen_i = slen as i32;
            let mut col = 0;
            for j in (slen_i - gene.end)..(slen_i + 1 - gene.begin) {
                let ju = j as usize;
                if is_a(rseq, ju) {
                    out.push(b'A');
                } else if is_t(rseq, ju) {
                    out.push(b'T');
                } else if is_g(rseq, ju) {
                    out.push(b'G');
                } else if is_c(rseq, ju) && !is_n(useq, slen - 1 - ju) {
                    out.push(b'C');
                } else {
                    out.push(b'N');
                }
                col += 1;
                if col % 70 == 0 {
                    out.push(b'\n');
                }
            }
            if col % 70 != 0 {
                out.push(b'\n');
            }
        }
    }
}

// ============================================================================
// Public API - run prodigal on a genome file
// ============================================================================

/// Run Prodigal gene prediction on a FASTA file using pre-trained data.
/// Returns the nucleotide CDS sequences as FASTA bytes (identical to `prodigal -d /dev/stdout`).
/// Pre-allocated buffers for prodigal, reused across genomes via thread-local storage.
struct ProdigalBuffers {
    seq: Vec<u8>,
    rseq: Vec<u8>,
    useq: Vec<u8>,
    nodes: Vec<Node>,
    genes: Vec<Gene>,
    output: Vec<u8>,
}

impl ProdigalBuffers {
    fn new() -> Self {
        let seq_bytes = MAX_SEQ / 4;
        let useq_bytes = MAX_SEQ / 8;
        ProdigalBuffers {
            seq: vec![0u8; seq_bytes],
            rseq: vec![0u8; seq_bytes],
            useq: vec![0u8; useq_bytes],
            nodes: Vec::with_capacity(100_000),
            genes: Vec::with_capacity(MAX_GENES),
            output: Vec::with_capacity(256 * 1024),
        }
    }
}

thread_local! {
    static PRODIGAL_BUFS: std::cell::RefCell<ProdigalBuffers> =
        std::cell::RefCell::new(ProdigalBuffers::new());
}

pub fn run_prodigal(
    fasta_path: &Path,
    training: &Training,
    trans_table: i32,
    closed: bool,
) -> io::Result<Vec<u8>> {
    PRODIGAL_BUFS.with(|cell| {
        let mut bufs = cell.borrow_mut();
        run_prodigal_with_bufs(fasta_path, training, trans_table, closed, &mut bufs)
    })
}

fn run_prodigal_with_bufs(
    fasta_path: &Path,
    training: &Training,
    trans_table: i32,
    closed: bool,
    bufs: &mut ProdigalBuffers,
) -> io::Result<Vec<u8>> {
    bufs.output.clear();

    let mut reader = SeqReader::new(fasta_path)?;

    let mut prev_slen: usize = 0;
    loop {
        // Only clear the portion used by the previous sequence
        let clear_bytes = ((prev_slen + 3) / 4) + 1;
        let clear_ubytes = ((prev_slen + 7) / 8) + 1;
        unsafe {
            std::ptr::write_bytes(bufs.seq.as_mut_ptr(), 0, clear_bytes.min(bufs.seq.len()));
            std::ptr::write_bytes(bufs.rseq.as_mut_ptr(), 0, clear_bytes.min(bufs.rseq.len()));
            std::ptr::write_bytes(bufs.useq.as_mut_ptr(), 0, clear_ubytes.min(bufs.useq.len()));
        }

        let result = reader.next_seq(&mut bufs.seq, &mut bufs.useq, false);
        if result.is_none() {
            break;
        }
        let (slen, _gc, cur_header, masks) = result.unwrap();
        if slen == 0 {
            continue;
        }
        prev_slen = slen;

        let num_seq = reader.sctr;
        let short_header = calc_short_header(&cur_header, num_seq);

        rcom_seq(&bufs.seq, &mut bufs.rseq, &bufs.useq, slen);

        let nn = add_nodes(
            &bufs.seq,
            &bufs.rseq,
            slen,
            &mut bufs.nodes,
            closed,
            &masks,
            trans_table,
        );
        if nn == 0 {
            continue;
        }

        reset_node_scores(&mut bufs.nodes);
        score_nodes(
            &bufs.seq,
            &bufs.rseq,
            slen,
            &mut bufs.nodes,
            training,
            closed,
            false,
        );
        record_overlapping_starts(&mut bufs.nodes, training, 1);

        let ipath = dprog(&mut bufs.nodes, training, 1);
        eliminate_bad_genes(&mut bufs.nodes, ipath, training);

        let ng = add_genes(&mut bufs.genes, &bufs.nodes, ipath);
        if ng == 0 {
            continue;
        }

        tweak_final_starts(&mut bufs.genes, &mut bufs.nodes, training);
        write_nucleotide_seqs(
            &mut bufs.output,
            &bufs.genes,
            &bufs.nodes,
            &bufs.seq,
            &bufs.rseq,
            &bufs.useq,
            slen,
            training,
            num_seq,
            &short_header,
        );
    }

    Ok(std::mem::take(&mut bufs.output))
}

/// Thread-safe wrapper: each call creates its own buffers.
/// Designed for parallel genome processing with rayon.
pub fn predict_genome(fasta_path: &Path, training: &Training) -> io::Result<Vec<u8>> {
    run_prodigal(fasta_path, training, training.trans_table, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn test_vs_c_prodigal() {
        // Set PRODIGAL_TEST_GENOME, PRODIGAL_TEST_TRN, and PRODIGAL_BIN to run this test.
        // Example: PRODIGAL_TEST_GENOME=/path/to/genome.fasta PRODIGAL_TEST_TRN=/path/to/file.trn PRODIGAL_BIN=prodigal cargo test
        let genome_str = match std::env::var("PRODIGAL_TEST_GENOME") {
            Ok(v) => v,
            Err(_) => {
                eprintln!("PRODIGAL_TEST_GENOME not set, skipping test");
                return;
            }
        };
        let trn_str = match std::env::var("PRODIGAL_TEST_TRN") {
            Ok(v) => v,
            Err(_) => {
                eprintln!("PRODIGAL_TEST_TRN not set, skipping test");
                return;
            }
        };
        let prodigal_bin = std::env::var("PRODIGAL_BIN").unwrap_or_else(|_| "prodigal".to_string());

        let genome = Path::new(&genome_str);
        let trn_path = Path::new(&trn_str);

        if !genome.exists() || !trn_path.exists() {
            eprintln!("Test data not found, skipping");
            return;
        }

        // Run C prodigal: -d to file, SCO to /dev/null
        let nuc_file = "/tmp/prodigal_rs_test_c.fasta";
        let c_result = Command::new(&prodigal_bin)
            .args(&[
                "-i",
                genome.to_str().unwrap(),
                "-t",
                trn_path.to_str().unwrap(),
                "-g",
                "11",
                "-d",
                nuc_file,
                "-f",
                "sco",
                "-q",
            ])
            .output()
            .expect("Failed to run C prodigal");
        assert!(c_result.status.success(), "C prodigal failed");
        let c_stdout = std::fs::read(nuc_file).expect("Failed to read C prodigal output");
        // Wrap in a struct for convenience
        struct COutput {
            stdout: Vec<u8>,
        }
        let c_output = COutput { stdout: c_stdout };

        // Run Rust prodigal
        let training = Training::from_file(trn_path).expect("Failed to load training file");
        let rs_output = run_prodigal(genome, &training, 11, false).expect("Rust prodigal failed");

        // Compare headers first
        let c_headers: Vec<&str> = std::str::from_utf8(&c_output.stdout)
            .unwrap()
            .lines()
            .filter(|l| l.starts_with('>'))
            .collect();
        let rs_headers: Vec<&str> = std::str::from_utf8(&rs_output)
            .unwrap()
            .lines()
            .filter(|l| l.starts_with('>'))
            .collect();

        eprintln!(
            "C prodigal: {} genes, {} bytes",
            c_headers.len(),
            c_output.stdout.len()
        );
        eprintln!(
            "Rust prodigal: {} genes, {} bytes",
            rs_headers.len(),
            rs_output.len()
        );

        // Show first few differences
        let max_show = 10;
        let mut diff_count = 0;
        for i in 0..c_headers.len().max(rs_headers.len()) {
            let c_h = c_headers.get(i).unwrap_or(&"<missing>");
            let rs_h = rs_headers.get(i).unwrap_or(&"<missing>");
            if c_h != rs_h {
                if diff_count < max_show {
                    eprintln!("DIFF gene {}:", i + 1);
                    eprintln!("  C:  {}", c_h);
                    eprintln!("  RS: {}", rs_h);
                }
                diff_count += 1;
            }
        }
        eprintln!("Total header diffs: {} / {}", diff_count, c_headers.len());

        // Compare full output
        if c_output.stdout != rs_output {
            // Find first byte difference
            let min_len = c_output.stdout.len().min(rs_output.len());
            for i in 0..min_len {
                if c_output.stdout[i] != rs_output[i] {
                    let start = if i > 50 { i - 50 } else { 0 };
                    let end = (i + 50).min(min_len);
                    eprintln!("First byte diff at offset {}:", i);
                    eprintln!(
                        "  C:  {:?}",
                        String::from_utf8_lossy(&c_output.stdout[start..end])
                    );
                    eprintln!(
                        "  RS: {:?}",
                        String::from_utf8_lossy(&rs_output[start..end])
                    );
                    break;
                }
            }
            if c_output.stdout.len() != rs_output.len() {
                eprintln!(
                    "Output length differs: C={}, RS={}",
                    c_output.stdout.len(),
                    rs_output.len()
                );
            }
        }

        assert_eq!(
            c_output.stdout, rs_output,
            "Output does not match C prodigal"
        );
    }

    #[test]
    fn test_vs_c_prodigal_multi_genomes() {
        // Set PRODIGAL_TEST_GENOMES_DIR, PRODIGAL_TEST_TRN, and PRODIGAL_BIN to run this test.
        let genomes_str = match std::env::var("PRODIGAL_TEST_GENOMES_DIR") {
            Ok(v) => v,
            Err(_) => {
                eprintln!("PRODIGAL_TEST_GENOMES_DIR not set, skipping test");
                return;
            }
        };
        let trn_str = match std::env::var("PRODIGAL_TEST_TRN") {
            Ok(v) => v,
            Err(_) => {
                eprintln!("PRODIGAL_TEST_TRN not set, skipping test");
                return;
            }
        };
        let prodigal = std::env::var("PRODIGAL_BIN").unwrap_or_else(|_| "prodigal".to_string());

        let genomes_dir = Path::new(&genomes_str);
        let trn_path = Path::new(&trn_str);

        if !genomes_dir.exists() || !trn_path.exists() {
            eprintln!("Test data not found, skipping");
            return;
        }

        let training = Training::from_file(trn_path).expect("Failed to load training file");

        // Test first 10 genomes
        let mut entries: Vec<_> = std::fs::read_dir(genomes_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "fasta"))
            .collect();
        entries.sort_by_key(|e| e.file_name());
        entries.truncate(10);

        let mut total_match = 0;
        let mut total_fail = 0;
        for entry in &entries {
            let genome = entry.path();
            let nuc_file = format!(
                "/tmp/prodigal_rs_test_{}.fasta",
                genome.file_stem().unwrap().to_str().unwrap()
            );

            let c_result = Command::new(prodigal)
                .args(&[
                    "-i",
                    genome.to_str().unwrap(),
                    "-t",
                    trn_path.to_str().unwrap(),
                    "-g",
                    "11",
                    "-d",
                    &nuc_file,
                    "-f",
                    "sco",
                    "-q",
                ])
                .output()
                .expect("Failed to run C prodigal");
            assert!(c_result.status.success());
            let c_output = std::fs::read(&nuc_file).unwrap();

            let rs_output = run_prodigal(&genome, &training, 11, false).unwrap();

            if c_output == rs_output {
                total_match += 1;
            } else {
                total_fail += 1;
                eprintln!(
                    "MISMATCH: {} (C={} bytes, RS={} bytes)",
                    genome.display(),
                    c_output.len(),
                    rs_output.len()
                );
                // Find first byte difference
                let c_str = String::from_utf8_lossy(&c_output);
                let rs_str = String::from_utf8_lossy(&rs_output);
                let c_hdrs: Vec<&str> = c_str.lines().filter(|l| l.starts_with('>')).collect();
                let rs_hdrs: Vec<&str> = rs_str.lines().filter(|l| l.starts_with('>')).collect();
                eprintln!("  C genes: {}, RS genes: {}", c_hdrs.len(), rs_hdrs.len());
                let min_len = c_output.len().min(rs_output.len());
                for bi in 0..min_len {
                    if c_output[bi] != rs_output[bi] {
                        let start = if bi > 100 { bi - 100 } else { 0 };
                        let end = (bi + 100).min(min_len);
                        eprintln!("  First diff at byte {}:", bi);
                        eprintln!(
                            "    C:  {:?}",
                            String::from_utf8_lossy(&c_output[start..end])
                        );
                        eprintln!(
                            "    RS: {:?}",
                            String::from_utf8_lossy(&rs_output[start..end])
                        );
                        break;
                    }
                }
            }
            std::fs::remove_file(&nuc_file).ok();
        }
        eprintln!(
            "{}/{} genomes matched perfectly",
            total_match,
            total_match + total_fail
        );
        assert_eq!(total_fail, 0, "Some genomes didn't match");
    }
}
