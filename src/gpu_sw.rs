//! GPU-accelerated Smith-Waterman via CUDA (ported from chewbbacca_gpu).
//!
//! Uses the exact same CUDA kernel as gpu_sw.py: BLOSUM62, gap_open=11, gap_extend=1.

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

const AA_ORDER: &[u8] = b"ARNDCQEGHILKMFPSTWYVBZX*";

const GAP_OPEN: i32 = 11;
const GAP_EXTEND: i32 = 1;

/// BLOSUM62 scoring matrix (24x24), same order as Python: ARNDCQEGHILKMFPSTWYVBZX*
#[rustfmt::skip]
const BLOSUM62: [i32; 576] = [
     4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4,
    -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4,
    -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4,
    -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4,
     0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4,
    -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4,
    -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4,
     0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4,
    -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4,
    -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4,
    -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4,
    -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4,
    -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4,
    -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4,
    -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4,
     1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4,
     0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4,
    -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4,
    -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4,
     0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4,
    -2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4,
    -1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4,
     0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4,
    -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1,
];

/// Exact same CUDA kernel as gpu_sw.py
const SW_KERNEL_CODE: &str = r#"
extern "C" __global__
void smith_waterman_batch(
    const int* __restrict__ queries,
    const int* __restrict__ targets,
    const int* __restrict__ query_offsets,
    const int* __restrict__ target_offsets,
    const int* __restrict__ query_lengths,
    const int* __restrict__ target_lengths,
    const int* __restrict__ blosum62,
    const int gap_open,
    const int gap_extend,
    int* __restrict__ results,
    const int num_pairs,
    const int max_query_len,
    const int max_target_len
) {
    int pair_idx = blockIdx.x;
    if (pair_idx >= num_pairs) return;
    if (threadIdx.x != 0) return;

    int qlen = query_lengths[pair_idx];
    int tlen = target_lengths[pair_idx];
    int qoff = query_offsets[pair_idx];
    int toff = target_offsets[pair_idx];

    int stride = max_query_len + 1;
    extern __shared__ int shared_mem[];
    // Layout: H[stride], E[stride], blosum[576]
    int* H = shared_mem;
    int* E = H + stride;
    int* blosum_s = E + stride;

    for (int k = 0; k < 576; k++) {
        blosum_s[k] = blosum62[k];
    }

    for (int i = 0; i <= qlen; i++) {
        H[i] = 0;
        E[i] = 0;
    }

    int max_score = 0;
    int best_qi = 0, best_qj = 0;
    int gap_oe = gap_open + gap_extend;

    for (int j = 1; j <= tlen; j++) {
        int tj = targets[toff + j - 1];
        const int* brow = blosum_s + tj;
        int Fval = 0;
        int h_diag = 0;

        for (int i = 1; i <= qlen; i++) {
            int qi = queries[qoff + i - 1];
            int h_left = H[i];
            int match_val = h_diag + brow[qi * 24];

            int e_open = h_left - gap_oe;
            int e_ext  = E[i] - gap_extend;
            if (e_open > e_ext) { E[i] = e_open; }
            else { E[i] = e_ext; }

            int f_open = H[i-1] - gap_oe;
            int f_ext  = Fval - gap_extend;
            if (f_open > f_ext) { Fval = f_open; }
            else { Fval = f_ext; }

            int h = match_val;
            if (E[i] > h) { h = E[i]; }
            if (Fval > h) { h = Fval; }
            if (h < 0) { h = 0; }

            h_diag = h_left;
            H[i] = h;

            if (h > max_score) {
                max_score = h;
                best_qi = i;
                best_qj = j;
            }
        }
    }

    // Output: score, query_end (1-based), target_end (1-based), qlen, tlen
    // Start positions will be computed on CPU via reverse alignment (parasail)
    int out_idx = pair_idx * 5;
    results[out_idx + 0] = max_score;
    results[out_idx + 1] = best_qi;       // query end, 1-based
    results[out_idx + 2] = best_qj;       // target end, 1-based
    results[out_idx + 3] = qlen;
    results[out_idx + 4] = tlen;
}
"#;


/// Build the amino acid lookup table (byte -> BLOSUM62 index).
fn build_aa_lookup() -> [i32; 256] {
    let mut lookup = [22i32; 256]; // default to 'X' index
    for (i, &aa) in AA_ORDER.iter().enumerate() {
        lookup[aa as usize] = i as i32;
        lookup[aa.to_ascii_lowercase() as usize] = i as i32;
    }
    lookup
}

/// Encode a protein sequence to BLOSUM62 indices.
fn encode_protein(seq: &[u8], lookup: &[i32; 256]) -> Vec<i32> {
    seq.iter().map(|&b| lookup[b as usize]).collect()
}

/// Result of a single GPU SW alignment.
/// Only score and end positions come from the GPU kernel.
/// Start positions are computed on CPU via parasail reverse alignment.
#[derive(Debug, Clone)]
pub struct GpuSwResult {
    pub pair_idx: usize,
    pub score: i32,
    pub query_end: u32,    // 1-based
    pub target_end: u32,   // 1-based
    pub query_len: u32,
    pub target_len: u32,
}

/// GPU Smith-Waterman aligner using cudarc 0.19.
pub struct GpuAligner {
    stream: Arc<CudaStream>,
    blosum_gpu: CudaSlice<i32>,
    func: CudaFunction,
}

impl GpuAligner {
    /// Create a new GPU aligner on device 0.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let t0 = std::time::Instant::now();
        let ctx = CudaContext::new(0)?;
        let t_ctx = t0.elapsed();
        let stream = ctx.default_stream();

        // Compile kernel
        let ptx = compile_ptx(SW_KERNEL_CODE)?;
        let t_ptx = t0.elapsed();
        let module = ctx.load_module(ptx)?;
        let func = module.load_function("smith_waterman_batch")?;
        let t_load = t0.elapsed();

        // Upload BLOSUM62
        let blosum_gpu = stream.clone_htod(&BLOSUM62)?;

        eprintln!("  GPU init: ctx={:.1}ms, ptx_compile={:.1}ms, load_module={:.1}ms, total={:.1}ms",
            t_ctx.as_secs_f64() * 1000.0,
            (t_ptx - t_ctx).as_secs_f64() * 1000.0,
            (t_load - t_ptx).as_secs_f64() * 1000.0,
            t0.elapsed().as_secs_f64() * 1000.0,
        );

        Ok(Self { stream, blosum_gpu, func })
    }

    /// Align batched pairs of protein sequences on GPU.
    pub fn align_pairs(
        &self,
        queries: &[Vec<u8>],
        targets: &[Vec<u8>],
    ) -> Result<Vec<GpuSwResult>, Box<dyn std::error::Error>> {
        let num_pairs = queries.len();
        if num_pairs == 0 {
            return Ok(Vec::new());
        }

        let lookup = build_aa_lookup();

        // Encode sequences
        let encoded_queries: Vec<Vec<i32>> = queries.iter().map(|q| encode_protein(q, &lookup)).collect();
        let encoded_targets: Vec<Vec<i32>> = targets.iter().map(|t| encode_protein(t, &lookup)).collect();

        // Compute offsets and lengths
        let mut q_lengths = Vec::with_capacity(num_pairs);
        let mut t_lengths = Vec::with_capacity(num_pairs);
        let mut q_offsets = Vec::with_capacity(num_pairs);
        let mut t_offsets = Vec::with_capacity(num_pairs);

        let mut q_off = 0i32;
        let mut t_off = 0i32;
        for i in 0..num_pairs {
            q_offsets.push(q_off);
            t_offsets.push(t_off);
            q_lengths.push(encoded_queries[i].len() as i32);
            t_lengths.push(encoded_targets[i].len() as i32);
            q_off += encoded_queries[i].len() as i32;
            t_off += encoded_targets[i].len() as i32;
        }

        let max_qlen = *q_lengths.iter().max().unwrap_or(&0);
        let max_tlen = *t_lengths.iter().max().unwrap_or(&0);

        // Flatten sequences
        let all_queries: Vec<i32> = encoded_queries.into_iter().flatten().collect();
        let all_targets: Vec<i32> = encoded_targets.into_iter().flatten().collect();

        // Upload to GPU
        let d_queries = self.stream.clone_htod(&all_queries)?;
        let d_targets = self.stream.clone_htod(&all_targets)?;
        let d_q_offsets = self.stream.clone_htod(&q_offsets)?;
        let d_t_offsets = self.stream.clone_htod(&t_offsets)?;
        let d_q_lengths = self.stream.clone_htod(&q_lengths)?;
        let d_t_lengths = self.stream.clone_htod(&t_lengths)?;
        let mut d_results = self.stream.alloc_zeros::<i32>(num_pairs * 5)?;

        // Shared memory: 2 arrays (H, E) of stride ints + BLOSUM62
        let shared_mem_bytes = (2 * (max_qlen as usize + 1) * 4 + 576 * 4) as u32;

        let cfg = LaunchConfig {
            grid_dim: (num_pairs as u32, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes,
        };

        let mut builder = self.stream.launch_builder(&self.func);
        builder.arg(&d_queries);
        builder.arg(&d_targets);
        builder.arg(&d_q_offsets);
        builder.arg(&d_t_offsets);
        builder.arg(&d_q_lengths);
        builder.arg(&d_t_lengths);
        builder.arg(&self.blosum_gpu);
        builder.arg(&GAP_OPEN);
        builder.arg(&GAP_EXTEND);
        builder.arg(&mut d_results);
        let num_pairs_i32 = num_pairs as i32;
        builder.arg(&num_pairs_i32);
        builder.arg(&max_qlen);
        builder.arg(&max_tlen);

        unsafe {
            builder.launch(cfg)?;
        }

        // Download results
        let results_host = self.stream.clone_dtoh(&d_results)?;

        let mut results = Vec::with_capacity(num_pairs);
        for i in 0..num_pairs {
            let base = i * 5;
            results.push(GpuSwResult {
                pair_idx: i,
                score: results_host[base],
                query_end: results_host[base + 1] as u32,
                target_end: results_host[base + 2] as u32,
                query_len: results_host[base + 3] as u32,
                target_len: results_host[base + 4] as u32,
            });
        }

        Ok(results)
    }

    /// Align pairs by indices into pre-existing sequence arrays.
    /// Uses query-length bucketing for better GPU occupancy and chunking
    /// to avoid CUDA_ERROR_INVALID_VALUE on large batches.
    pub fn align_indexed(
        &self,
        all_queries: &[&[u8]],
        all_targets: &[&[u8]],
        query_indices: &[usize],
        target_indices: &[usize],
    ) -> Result<Vec<GpuSwResult>, Box<dyn std::error::Error>> {
        const GPU_CHUNK_SIZE: usize = 50_000;

        let num_pairs = query_indices.len();
        if num_pairs == 0 {
            return Ok(Vec::new());
        }

        let t0 = std::time::Instant::now();
        let lookup = build_aa_lookup();

        // Encode unique sequences only (dedup by index)
        let mut q_encoded_cache: Vec<Option<Vec<i32>>> = vec![None; all_queries.len()];
        let mut t_encoded_cache: Vec<Option<Vec<i32>>> = vec![None; all_targets.len()];

        for &qi in query_indices {
            if q_encoded_cache[qi].is_none() {
                q_encoded_cache[qi] = Some(encode_protein(all_queries[qi], &lookup));
            }
        }
        for &ti in target_indices {
            if t_encoded_cache[ti].is_none() {
                t_encoded_cache[ti] = Some(encode_protein(all_targets[ti], &lookup));
            }
        }

        // Bucket pairs by query length for shared memory optimization.
        // Max query length for GPU: 5000 aa (shared mem = 2*5001*4+2304 = 42312 < 48KB).
        // Longer sequences fall back to CPU parasail SIMD.
        const MAX_GPU_QLEN: usize = 5000;
        const BUCKET_LIMITS: &[usize] = &[64, 128, 256, 512, 1024, 2048, 5000];
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); BUCKET_LIMITS.len()];
        let mut cpu_fallback: Vec<usize> = Vec::new();

        for i in 0..num_pairs {
            let qlen = q_encoded_cache[query_indices[i]].as_ref().unwrap().len();
            if qlen > MAX_GPU_QLEN {
                cpu_fallback.push(i);
                continue;
            }
            for (b, &limit) in BUCKET_LIMITS.iter().enumerate() {
                if qlen <= limit {
                    buckets[b].push(i);
                    break;
                }
            }
        }

        // Build flattened arrays per bucket and launch kernels
        let mut results = vec![GpuSwResult {
            pair_idx: 0, score: 0, query_end: 0,
            target_end: 0, query_len: 0, target_len: 0,
        }; num_pairs];

        for bucket in buckets.iter() {
            if bucket.is_empty() { continue; }

            // Process bucket in chunks to avoid CUDA errors on large batches
            for chunk_start in (0..bucket.len()).step_by(GPU_CHUNK_SIZE) {
                let chunk_end = (chunk_start + GPU_CHUNK_SIZE).min(bucket.len());
                let chunk = &bucket[chunk_start..chunk_end];
                let chunk_size = chunk.len();

                let mut flat_queries = Vec::new();
                let mut flat_targets = Vec::new();
                let mut q_offsets = Vec::with_capacity(chunk_size);
                let mut t_offsets = Vec::with_capacity(chunk_size);
                let mut q_lengths = Vec::with_capacity(chunk_size);
                let mut t_lengths = Vec::with_capacity(chunk_size);
                let mut max_qlen = 0i32;
                let mut max_tlen = 0i32;

                for &pair_i in chunk {
                    let qe = q_encoded_cache[query_indices[pair_i]].as_ref().unwrap();
                    let te = t_encoded_cache[target_indices[pair_i]].as_ref().unwrap();
                    q_offsets.push(flat_queries.len() as i32);
                    t_offsets.push(flat_targets.len() as i32);
                    let ql = qe.len() as i32;
                    let tl = te.len() as i32;
                    q_lengths.push(ql);
                    t_lengths.push(tl);
                    if ql > max_qlen { max_qlen = ql; }
                    if tl > max_tlen { max_tlen = tl; }
                    flat_queries.extend_from_slice(qe);
                    flat_targets.extend_from_slice(te);
                }

                // Upload to GPU
                let d_queries = self.stream.clone_htod(&flat_queries)?;
                let d_targets = self.stream.clone_htod(&flat_targets)?;
                let d_q_offsets = self.stream.clone_htod(&q_offsets)?;
                let d_t_offsets = self.stream.clone_htod(&t_offsets)?;
                let d_q_lengths = self.stream.clone_htod(&q_lengths)?;
                let d_t_lengths = self.stream.clone_htod(&t_lengths)?;
                let mut d_results = self.stream.alloc_zeros::<i32>(chunk_size * 5)?;

                // Shared memory: 2 arrays (H, E) + BLOSUM62
                let shared_mem_bytes = (2 * (max_qlen as usize + 1) * 4 + 576 * 4) as u32;

                let cfg = LaunchConfig {
                    grid_dim: (chunk_size as u32, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes,
                };

                let mut builder = self.stream.launch_builder(&self.func);
                builder.arg(&d_queries);
                builder.arg(&d_targets);
                builder.arg(&d_q_offsets);
                builder.arg(&d_t_offsets);
                builder.arg(&d_q_lengths);
                builder.arg(&d_t_lengths);
                builder.arg(&self.blosum_gpu);
                builder.arg(&GAP_OPEN);
                builder.arg(&GAP_EXTEND);
                builder.arg(&mut d_results);
                let chunk_size_i32 = chunk_size as i32;
                builder.arg(&chunk_size_i32);
                builder.arg(&max_qlen);
                builder.arg(&max_tlen);

                unsafe {
                    builder.launch(cfg)?;
                }

                let results_host = self.stream.clone_dtoh(&d_results)?;

                // Map results back to original pair indices
                for (local_i, &pair_i) in chunk.iter().enumerate() {
                    let base = local_i * 5;
                    results[pair_i] = GpuSwResult {
                        pair_idx: pair_i,
                        score: results_host[base],
                        query_end: results_host[base + 1] as u32,
                        target_end: results_host[base + 2] as u32,
                        query_len: results_host[base + 3] as u32,
                        target_len: results_host[base + 4] as u32,
                    };
                }
            }
        }

        // CPU fallback for long sequences (> MAX_GPU_QLEN)
        if !cpu_fallback.is_empty() {
            eprintln!("  GPU: {} pairs falling back to CPU (query > {} aa)",
                      cpu_fallback.len(), MAX_GPU_QLEN);
            for &pair_i in &cpu_fallback {
                let q = all_queries[query_indices[pair_i]];
                let t = all_targets[target_indices[pair_i]];
                let (score, _, qe) = crate::parasail_ffi::sw_simd(q, t);
                results[pair_i] = GpuSwResult {
                    pair_idx: pair_i,
                    score,
                    query_end: qe,
                    target_end: 0, // will be resolved in cluster.rs
                    query_len: q.len() as u32,
                    target_len: t.len() as u32,
                };
            }
        }

        let t_done = t0.elapsed();
        eprintln!("  GPU: {} pairs aligned in {:.1}ms ({} GPU + {} CPU fallback)",
                  num_pairs, t_done.as_secs_f64() * 1000.0,
                  num_pairs - cpu_fallback.len(), cpu_fallback.len());

        Ok(results)
    }
}
