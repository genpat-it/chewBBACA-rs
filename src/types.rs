//! Shared types for the AlleleCall pipeline.


/// SHA256 hash of a DNA or protein sequence (32 bytes).
pub type SeqHash = [u8; 32];

/// Genome index (0-based).
pub type GenomeIdx = u32;

/// Locus index (0-based).
pub type LocusIdx = u32;

/// Allele identifier within a locus.
pub type AlleleId = u32;

/// CDS coordinate information.
#[derive(Debug, Clone)]
pub struct CdsCoord {
    pub genome_idx: GenomeIdx,
    pub contig: String,
    pub start: u32,
    pub stop: u32,
    pub strand: i8, // +1 or -1
    pub contig_len: u32,
}

/// A predicted CDS with its metadata.
#[derive(Debug, Clone)]
pub struct Cds {
    pub id: String,         // unique ID: "genome_idx-proteinN"
    pub dna_seq: Vec<u8>,   // DNA sequence
    pub genome_idx: GenomeIdx,
    pub coord: Option<CdsCoord>,
}

/// Classification of a CDS match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Classification {
    EXC,    // Exact DNA match
    INF,    // Inferred new allele
    PLOT3,  // Partial loss of target 3'
    PLOT5,  // Partial loss of target 5'
    LOTSC,  // Loss of target sequence coverage
    NIPH,   // Non-inferred partial hit (multiple inexact matches)
    NIPHEM, // NIPH exact match homolog
    ALM,    // Allele length mismatch (too long)
    ASM,    // Allele size mismatch (too short)
    PAMA,   // Paralogous match (matches multiple loci)
    LNF,    // Locus not found
}

impl Classification {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::EXC => "EXC",
            Self::INF => "INF",
            Self::PLOT3 => "PLOT3",
            Self::PLOT5 => "PLOT5",
            Self::LOTSC => "LOTSC",
            Self::NIPH => "NIPH",
            Self::NIPHEM => "NIPHEM",
            Self::ALM => "ALM",
            Self::ASM => "ASM",
            Self::PAMA => "PAMA",
            Self::LNF => "LNF",
        }
    }

    /// Whether this is a "valid" classification (allele found).
    pub fn is_valid(&self) -> bool {
        matches!(self, Self::EXC | Self::INF)
    }
}

impl std::fmt::Display for Classification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Result of classifying a single CDS against a locus.
#[derive(Debug, Clone)]
pub struct MatchResult {
    pub locus_idx: LocusIdx,
    pub allele_id: Option<AlleleId>,
    pub class: Classification,
    pub bsr: f64,
    pub cds_hash: SeqHash,
    pub protein_hash: Option<SeqHash>,
    pub representative_id: Option<String>,
}

/// Per-locus classification for a genome.
#[derive(Debug, Clone)]
pub struct LocusResult {
    pub class: Classification,
    pub allele_id: Option<AlleleId>,
    pub is_novel: bool, // true if allele was inferred (novel), shown with * prefix
    pub dna_hash: Option<SeqHash>,
    pub matches: Vec<MatchResult>,
}

/// Schema locus information.
#[derive(Debug, Clone)]
pub struct Locus {
    pub id: String,           // locus identifier (e.g., "locus_1")
    pub fasta_path: String,   // path to locus FASTA file
    pub short_path: String,   // path to representative alleles FASTA
    pub allele_count: u32,
    pub max_allele_id: u32,   // highest allele identifier observed in schema FASTA
    pub mode_length: u32,     // most frequent allele DNA length
    pub self_score: f64,      // representative self-alignment score
}

/// Representative allele for a locus.
#[derive(Debug, Clone)]
pub struct Representative {
    pub locus_idx: LocusIdx,
    pub seq_id: String,
    pub protein_seq: Vec<u8>,
    pub dna_length: u32,
    pub self_score: f64,
}

/// Alignment result from Smith-Waterman.
#[derive(Debug, Clone)]
pub struct SwResult {
    pub query_id: usize,   // index into query array
    pub target_id: usize,  // index into target array
    pub score: i32,
    pub query_start: u32,  // 1-based
    pub query_end: u32,    // 1-based
    pub query_len: u32,
    pub target_len: u32,
    pub target_start: u32, // 1-based
    pub target_end: u32,   // 1-based
}

/// Alignment mode for Phase 4 / RepDet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignMode {
    /// Fast mode: parasail SIMD (12-17x faster, 99.997% match)
    Fast,
    /// Compatible mode: BLAST (100% identical to chewBBACA)
    Compatible,
}

/// Configuration for AlleleCall.
#[derive(Debug, Clone)]
pub struct Config {
    pub bsr_threshold: f64,        // default 0.6
    pub size_threshold: f64,       // default 0.2
    pub min_sequence_length: u32,  // default 0 (auto)
    pub translation_table: u8,     // default 11
    pub cpu_cores: usize,
    pub prodigal_mode: String,     // "single" or "meta"
    pub use_gpu: bool,             // use CUDA GPU for SW alignment
    pub prodigal_path: String,     // path to prodigal binary
    pub align_mode: AlignMode,     // fast (parasail) or compatible (BLAST)
    pub blastp_path: String,       // path to blastp binary
    pub cds_input: bool,           // true if --cds-input was used (skip PLOT classification)
    pub update_schema: bool,       // append novel alleles to schema FASTA files
    pub minimizer_k: usize,        // minimizer k-mer size (default 5)
    pub minimizer_w: usize,        // minimizer window size (default 5)
    pub use_prodigal_ffi: bool,    // use libprodigal FFI instead of subprocess
}

impl Default for Config {
    fn default() -> Self {
        Self {
            bsr_threshold: 0.6,
            size_threshold: 0.2,
            min_sequence_length: 0,
            translation_table: 11,
            cpu_cores: 1,
            prodigal_mode: "single".to_string(),
            use_gpu: false,
            prodigal_path: "prodigal".to_string(),
            align_mode: AlignMode::Fast,
            blastp_path: "blastp".to_string(),
            cds_input: false,
            update_schema: false,
            minimizer_k: 5,
            minimizer_w: 5,
            use_prodigal_ffi: false,
        }
    }
}
