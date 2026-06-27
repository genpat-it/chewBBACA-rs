//! FFI bindings to libprodigal for in-process CDS prediction.
//!
//! Each rayon worker thread gets its own `ProdigalCtx` via `thread_local!`,
//! avoiding both subprocess overhead and thread-safety issues.

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::path::Path;

/// Opaque handle to a prodigal context (C side).
#[repr(C)]
pub struct prodigal_ctx {
    _opaque: [u8; 0],
}

#[link(name = "prodigal", kind = "static")]
#[link(name = "m")]
extern "C" {
    fn prodigal_create() -> *mut prodigal_ctx;
    fn prodigal_destroy(ctx: *mut prodigal_ctx);
    fn prodigal_load_training(ctx: *mut prodigal_ctx, train_file: *const c_char) -> c_int;
    fn prodigal_set_trans_table(ctx: *mut prodigal_ctx, table: c_int);
    fn prodigal_set_closed(ctx: *mut prodigal_ctx, closed: c_int);
    fn prodigal_set_mask(ctx: *mut prodigal_ctx, mask: c_int);
    fn prodigal_run_file_with_seqs(
        ctx: *mut prodigal_ctx,
        fasta_path: *const c_char,
        out_buf: *mut *mut c_char,
        out_len: *mut usize,
    ) -> c_int;
}

/// Safe wrapper around the C prodigal context.
/// NOT Send/Sync -- only used within a single thread.
pub struct ProdigalCtx {
    ptr: *mut prodigal_ctx,
}

impl ProdigalCtx {
    /// Create a new prodigal context. Returns None on allocation failure.
    pub fn new() -> Option<Self> {
        let ptr = unsafe { prodigal_create() };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Load a training file (.trn). Returns Ok(()) on success.
    pub fn load_training(&mut self, path: &Path) -> Result<(), String> {
        let c_path = CString::new(path.to_str().ok_or("invalid UTF-8 in path")?)
            .map_err(|e| format!("CString error: {}", e))?;
        let rv = unsafe { prodigal_load_training(self.ptr, c_path.as_ptr()) };
        match rv {
            0 => Ok(()),
            -1 => Err(format!("failed to read training file: {}", path.display())),
            -2 => Err(format!("training file not found: {}", path.display())),
            _ => Err(format!("unknown error loading training file: {}", rv)),
        }
    }

    /// Set the translation table (genetic code).
    pub fn set_trans_table(&mut self, table: u8) {
        unsafe { prodigal_set_trans_table(self.ptr, table as c_int) };
    }

    /// Set closed ends mode.
    #[allow(dead_code)]
    pub fn set_closed(&mut self, closed: bool) {
        unsafe { prodigal_set_closed(self.ptr, closed as c_int) };
    }

    /// Set mask mode.
    #[allow(dead_code)]
    pub fn set_mask(&mut self, mask: bool) {
        unsafe { prodigal_set_mask(self.ptr, mask as c_int) };
    }

    /// Run gene prediction on a FASTA file and return CDS nucleotide FASTA data.
    /// Returns (fasta_bytes, num_genes) or an error string.
    pub fn run_file_with_seqs(&mut self, fasta_path: &Path) -> Result<(Vec<u8>, i32), String> {
        let c_path = CString::new(fasta_path.to_str().ok_or("invalid UTF-8 in path")?)
            .map_err(|e| format!("CString error: {}", e))?;

        let mut buf_ptr: *mut c_char = std::ptr::null_mut();
        let mut buf_len: usize = 0;

        let num_genes = unsafe {
            prodigal_run_file_with_seqs(self.ptr, c_path.as_ptr(), &mut buf_ptr, &mut buf_len)
        };

        if num_genes < 0 {
            // Clean up if buffer was partially allocated
            if !buf_ptr.is_null() {
                unsafe { libc::free(buf_ptr as *mut libc::c_void) };
            }
            return Err(format!("prodigal failed on {}", fasta_path.display()));
        }

        let data = if buf_len > 0 && !buf_ptr.is_null() {
            let slice = unsafe { std::slice::from_raw_parts(buf_ptr as *const u8, buf_len) };
            let v = slice.to_vec();
            unsafe { libc::free(buf_ptr as *mut libc::c_void) };
            v
        } else {
            if !buf_ptr.is_null() {
                unsafe { libc::free(buf_ptr as *mut libc::c_void) };
            }
            Vec::new()
        };

        Ok((data, num_genes))
    }
}

impl Drop for ProdigalCtx {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { prodigal_destroy(self.ptr) };
        }
    }
}

// ProdigalCtx is NOT Send/Sync -- it holds internal mutable C state.
// We use it only inside thread_local! storage.

// Thread-local prodigal context storage.
// Each rayon worker thread lazily initializes its own context.
thread_local! {
    static TL_CTX: RefCell<Option<ProdigalCtx>> = const { RefCell::new(None) };
}

/// Initialize the thread-local prodigal context with a training file and translation table.
/// Must be called on each worker thread before `run_ffi`.
fn ensure_ctx(training_file: &Path, translation_table: u8) -> Result<(), String> {
    TL_CTX.with(|cell| {
        let mut opt = cell.borrow_mut();
        if opt.is_none() {
            let mut ctx = ProdigalCtx::new().ok_or("failed to create prodigal context")?;
            ctx.load_training(training_file)?;
            ctx.set_trans_table(translation_table);
            *opt = Some(ctx);
        }
        Ok(())
    })
}

/// Run prodigal FFI on a single genome file. Returns raw FASTA bytes of CDS.
/// Automatically initializes the thread-local context if needed.
pub fn run_ffi(
    genome_path: &Path,
    training_file: &Path,
    translation_table: u8,
) -> Result<Vec<u8>, String> {
    ensure_ctx(training_file, translation_table)?;
    TL_CTX.with(|cell| {
        let mut opt = cell.borrow_mut();
        let ctx = opt.as_mut().unwrap();
        let (data, _num_genes) = ctx.run_file_with_seqs(genome_path)?;
        Ok(data)
    })
}
