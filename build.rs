fn main() {
    // Declare the custom cfg so newer rustc does not warn about it.
    println!("cargo:rustc-check-cfg=cfg(have_prodigal)");
    // Re-run this script (and re-evaluate have_prodigal / parasail linking) when
    // the library locations change.
    println!("cargo:rerun-if-env-changed=PRODIGAL_DIR");
    println!("cargo:rerun-if-env-changed=PARASAIL_DIR");
    println!("cargo:rerun-if-changed=build.rs");

    // Link parasail library for SIMD-accelerated Smith-Waterman
    let parasail_dir = std::env::var("PARASAIL_DIR").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/usr/local".into());
        format!("{}/parasail/build", home)
    });

    if std::path::Path::new(&parasail_dir)
        .join("libparasail.so")
        .exists()
    {
        println!("cargo:rustc-link-search=native={}", parasail_dir);
        println!("cargo:rustc-link-lib=dylib=parasail");
        println!("cargo:rustc-env=LD_LIBRARY_PATH={}", parasail_dir);
    } else {
        println!(
            "cargo:warning=parasail library not found in {}, SIMD SW will not be available",
            parasail_dir
        );
    }

    // Link libprodigal for FFI-based CDS prediction
    let prodigal_dir = std::env::var("PRODIGAL_DIR").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/usr/local".into());
        format!("{}/prodigal-src/build", home)
    });

    if std::path::Path::new(&prodigal_dir)
        .join("libprodigal.a")
        .exists()
    {
        println!("cargo:rustc-link-search=native={}", prodigal_dir);
        println!("cargo:rustc-link-lib=static=prodigal");
        println!("cargo:rustc-link-lib=dylib=m"); // libm for math functions (exp, etc.)
        println!("cargo:rustc-cfg=have_prodigal"); // enable the FFI module
    } else {
        println!(
            "cargo:warning=libprodigal.a not found in {}, --prodigal-ffi will not be available",
            prodigal_dir
        );
    }
}
