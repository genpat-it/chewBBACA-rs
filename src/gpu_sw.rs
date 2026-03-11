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
    float* __restrict__ results,
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
    extern __shared__ float shared_mem[];
    float* H  = shared_mem;
    float* E  = shared_mem + stride;
    int*   Hs = (int*)(shared_mem + 2 * stride);
    int*   Es = Hs + stride;
    int*   blosum_s = Es + stride;

    for (int k = 0; k < 576; k++) {
        blosum_s[k] = blosum62[k];
    }

    for (int i = 0; i <= qlen; i++) {
        H[i] = 0.0f;
        E[i] = 0.0f;
        Hs[i] = i;
        Es[i] = i;
    }

    float max_score = 0.0f;
    int best_qi = 0, best_qj = 0, best_qstart = 0;
    int gap_oe = gap_open + gap_extend;

    for (int j = 1; j <= tlen; j++) {
        int tj = targets[toff + j - 1];
        const int* brow = blosum_s + tj;
        float Fval = 0.0f;
        int Fs = 0;
        float h_diag = 0.0f;
        int h_diag_s = 0;

        for (int i = 1; i <= qlen; i++) {
            int qi = queries[qoff + i - 1];
            float h_left = H[i];
            int h_left_s = Hs[i];
            float match_val = h_diag + (float)brow[qi * 24];
            int match_s = h_diag_s;

            float e_open = h_left - gap_oe;
            float e_ext  = E[i] - gap_extend;
            if (e_open > e_ext) {
                E[i] = e_open;
                Es[i] = h_left_s;
            } else {
                E[i] = e_ext;
            }

            float f_open = H[i-1] - gap_oe;
            float f_ext  = Fval - gap_extend;
            if (f_open > f_ext) {
                Fval = f_open;
                Fs = Hs[i-1];
            } else {
                Fval = f_ext;
            }

            float h = match_val;
            int hs = match_s;
            if (E[i] > h) { h = E[i]; hs = Es[i]; }
            if (Fval > h) { h = Fval; hs = Fs; }
            if (h < 0.0f) { h = 0.0f; hs = i; }

            h_diag = h_left;
            h_diag_s = h_left_s;
            H[i] = h;
            Hs[i] = hs;

            if (h > max_score) {
                max_score = h;
                best_qi = i;
                best_qj = j;
                best_qstart = hs;
            }
        }
    }

    int out_idx = pair_idx * 5;
    results[out_idx + 0] = max_score;
    results[out_idx + 1] = (float)(best_qstart + 1);
    results[out_idx + 2] = (float)best_qi;
    results[out_idx + 3] = (float)qlen;
    results[out_idx + 4] = (float)tlen;
}
"#;

/// V2 kernel: uses global memory for work arrays, int instead of float.
/// Higher occupancy since only 2304 bytes shared memory (BLOSUM62 only).
const SW_KERNEL_V2: &str = r#"
extern "C" __global__
void sw_batch_v2(
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
    int* __restrict__ work_buf,
    const int work_stride
) {
    __shared__ int blosum_s[576];
    for (int k = threadIdx.x; k < 576; k += blockDim.x) {
        blosum_s[k] = blosum62[k];
    }
    __syncthreads();

    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= num_pairs) return;

    int qlen = query_lengths[pair_idx];
    int tlen = target_lengths[pair_idx];
    int qoff = query_offsets[pair_idx];
    int toff = target_offsets[pair_idx];

    long long base = (long long)pair_idx * work_stride * 4;
    int* H  = work_buf + base;
    int* E  = H + work_stride;
    int* Hs = E + work_stride;
    int* Es = Hs + work_stride;

    for (int i = 0; i <= qlen; i++) {
        H[i] = 0; E[i] = 0; Hs[i] = i; Es[i] = i;
    }

    int max_score = 0;
    int best_qi = 0, best_qstart = 0;
    int gap_oe = gap_open + gap_extend;

    for (int j = 1; j <= tlen; j++) {
        int tj = targets[toff + j - 1];
        int Fval = 0, Fs = 0;
        int h_diag = 0, h_diag_s = 0;

        for (int i = 1; i <= qlen; i++) {
            int qi = queries[qoff + i - 1];
            int h_left = H[i];
            int h_left_s = Hs[i];
            int match_val = h_diag + blosum_s[qi * 24 + tj];
            int match_s = h_diag_s;

            int e_open = h_left - gap_oe;
            int e_ext  = E[i] - gap_extend;
            if (e_open > e_ext) { E[i] = e_open; Es[i] = h_left_s; }
            else { E[i] = e_ext; }

            int f_open = H[i-1] - gap_oe;
            int f_ext  = Fval - gap_extend;
            if (f_open > f_ext) { Fval = f_open; Fs = Hs[i-1]; }
            else { Fval = f_ext; }

            int h = match_val;
            int hs = match_s;
            if (E[i] > h) { h = E[i]; hs = Es[i]; }
            if (Fval > h) { h = Fval; hs = Fs; }
            if (h < 0) { h = 0; hs = i; }

            h_diag = h_left;
            h_diag_s = h_left_s;
            H[i] = h;
            Hs[i] = hs;

            if (h > max_score) { max_score = h; best_qi = i; best_qstart = hs; }
        }
    }

    int out_idx = pair_idx * 5;
    results[out_idx + 0] = max_score;
    results[out_idx + 1] = best_qstart + 1;
    results[out_idx + 2] = best_qi;
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
#[derive(Debug, Clone)]
pub struct GpuSwResult {
    pub pair_idx: usize,
    pub score: f32,
    pub query_start: u32, // 1-based
    pub query_end: u32,   // 1-based
    pub query_len: u32,
    pub target_len: u32,
}

/// GPU Smith-Waterman aligner using cudarc 0.19.
pub struct GpuAligner {
    stream: Arc<CudaStream>,
    blosum_gpu: CudaSlice<i32>,
    func: CudaFunction,
    func_v2: CudaFunction,
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
        let func_v2 = func.clone(); // placeholder
        let t_load = t0.elapsed();

        // Upload BLOSUM62
        let blosum_gpu = stream.clone_htod(&BLOSUM62)?;

        eprintln!("  GPU init: ctx={:.1}ms, ptx_compile={:.1}ms, load_module={:.1}ms, total={:.1}ms",
            t_ctx.as_secs_f64() * 1000.0,
            (t_ptx - t_ctx).as_secs_f64() * 1000.0,
            (t_load - t_ptx).as_secs_f64() * 1000.0,
            t0.elapsed().as_secs_f64() * 1000.0,
        );

        Ok(Self { stream, blosum_gpu, func, func_v2 })
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
        let mut d_results = self.stream.alloc_zeros::<f32>(num_pairs * 5)?;

        // Shared memory: 4 arrays (H, E, Hs, Es) of stride floats/ints + BLOSUM62
        let shared_mem_bytes = (4 * (max_qlen as usize + 1) * 4 + 576 * 4) as u32;

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
                query_start: results_host[base + 1] as u32,
                query_end: results_host[base + 2] as u32,
                query_len: results_host[base + 3] as u32,
                target_len: results_host[base + 4] as u32,
            });
        }

        Ok(results)
    }

    /// Align pairs by indices into pre-existing sequence arrays.
    /// Uses query-length bucketing for better GPU occupancy.
    pub fn align_indexed(
        &self,
        all_queries: &[&[u8]],
        all_targets: &[&[u8]],
        query_indices: &[usize],
        target_indices: &[usize],
    ) -> Result<Vec<GpuSwResult>, Box<dyn std::error::Error>> {
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

        let t_encode = t0.elapsed();

        // Bucket pairs by query length for shared memory optimization
        const BUCKET_LIMITS: &[usize] = &[64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, usize::MAX];
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); BUCKET_LIMITS.len()];

        for i in 0..num_pairs {
            let qlen = q_encoded_cache[query_indices[i]].as_ref().unwrap().len();
            for (b, &limit) in BUCKET_LIMITS.iter().enumerate() {
                if qlen <= limit {
                    buckets[b].push(i);
                    break;
                }
            }
        }

        // Build flattened arrays per bucket and launch kernels
        let mut results = vec![GpuSwResult {
            pair_idx: 0, score: 0.0, query_start: 0, query_end: 0, query_len: 0, target_len: 0,
        }; num_pairs];

        let t_bucket = t0.elapsed();

        for (b_idx, bucket) in buckets.iter().enumerate() {
            if bucket.is_empty() { continue; }
            let t_bucket_start = std::time::Instant::now();

            let bucket_size = bucket.len();
            let mut flat_queries = Vec::new();
            let mut flat_targets = Vec::new();
            let mut q_offsets = Vec::with_capacity(bucket_size);
            let mut t_offsets = Vec::with_capacity(bucket_size);
            let mut q_lengths = Vec::with_capacity(bucket_size);
            let mut t_lengths = Vec::with_capacity(bucket_size);
            let mut max_qlen = 0i32;
            let mut max_tlen = 0i32;

            for &pair_i in bucket {
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
            let mut d_results = self.stream.alloc_zeros::<f32>(bucket_size * 5)?;

            // V1 kernel: shared memory for H/E/Hs/Es + BLOSUM62
            let shared_mem_bytes = (4 * (max_qlen as usize + 1) * 4 + 576 * 4) as u32;

            let cfg = LaunchConfig {
                grid_dim: (bucket_size as u32, 1, 1),
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
            let bucket_size_i32 = bucket_size as i32;
            builder.arg(&bucket_size_i32);
            builder.arg(&max_qlen);
            builder.arg(&max_tlen);

            unsafe {
                builder.launch(cfg)?;
            }

            let results_host = self.stream.clone_dtoh(&d_results)?;

            // Map results back to original pair indices
            for (local_i, &pair_i) in bucket.iter().enumerate() {
                let base = local_i * 5;
                results[pair_i] = GpuSwResult {
                    pair_idx: pair_i,
                    score: results_host[base],
                    query_start: results_host[base + 1] as u32,
                    query_end: results_host[base + 2] as u32,
                    query_len: results_host[base + 3] as u32,
                    target_len: results_host[base + 4] as u32,
                };
            }

            let _bucket_ms = t_bucket_start.elapsed().as_secs_f64() * 1000.0;
        }

        let t_done = t0.elapsed();
        eprintln!("  GPU: {} pairs aligned in {:.1}ms", num_pairs, t_done.as_secs_f64() * 1000.0);

        Ok(results)
    }
}
