# Changelog

All notable changes to chewcall are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.3.0] - 2026-06-29

### Added
- Docker image (multi-stage build of parasail + chewcall, CPU/SIMD) and a
  GitHub Actions workflow publishing it to `ghcr.io/genpat-it/chewcall` on tags.
- The prodigal FFI is now optional at build time (`cfg(have_prodigal)`, set by
  build.rs only when `libprodigal.a` is present); chewcall builds and runs
  without libprodigal using the default pure-Rust prodigal-rs predictor.
- `constructive_remedy` binary: greedily promotes witness alleles to
  representatives so that every locus attains worst-case minimizer overlap
  `wcr >= tau` by construction (schema-side safety net for the pre-filter).
- `schema_audit_pareto`: extended to a four-parameter `(k, w, tau, kappa)` sweep
  reporting Pareto-optimal operating points (filter recall vs. scoring work).
- `--brute-residual`: optional run-time safety net that bypasses the minimizer
  pre-filter and scores residual CDS against all representatives.
- `--minimizer-order {hash,lexicographic}`: select FNV-1a hash (default) or
  chewBBACA-style lexicographic minimizer ordering (for determinism studies).
- `--max-targets <N>`: configurable top-K candidate cap per query (default 30,
  matching chewBBACA; 0 = unbounded). Previously hardcoded to 30; the default is
  unchanged, so existing results are bit-identical. Pairs with the offline
  `schema_audit_pareto --kappa-values` sweep.

### Changed
- `schema_audit`: corrected terminology — the per-locus statistic is the
  query-normalised **minimizer containment / overlap (MO)**, not "Jaccard"
  (the computed value was already containment; only the label was wrong).

### Performance
- Peak resident memory reduced ~2.7–3.7× (e.g. *L. monocytogenes* consortium
  6.4 → 1.7 GB; *S. enterica* 12.2 → 4.5 GB) by loading genomes in chunks with
  streaming deduplication and retaining DNA only for **distinct** CDS, with the
  contig-info scan driven from the hash→genomes map. Allele calls are
  byte-identical (verified: 0 different-allele cells on all four BeONE
  consortium datasets); runtime is unchanged (parallel per-chunk hashing).

## [0.2.0]

### Added
- `schema_audit` (renamed from `tune_minimizer`): per-locus worst-case
  minimizer-overlap audit of the pre-filter; README reorganised around the
  schema-audit framework.
- `schema_audit_pareto` binary (initial `(k, w, tau)` sweep).

## [0.1.0]

- Initial release: Rust reimplementation of the chewBBACA AlleleCall pipeline
  with SIMD Smith-Waterman (parasail) scoring; the version benchmarked in the
  chewcall manuscript.

[0.3.0]: https://github.com/genpat-it/chewcall/releases/tag/v0.3.0
[0.2.0]: https://github.com/genpat-it/chewcall/releases/tag/v0.2.0
[0.1.0]: https://github.com/genpat-it/chewcall/releases/tag/v0.1.0
