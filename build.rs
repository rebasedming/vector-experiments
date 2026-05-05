fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_os == "linux" && target_arch == "aarch64" {
        // Link against BLIS for the cblas_sgemm symbol on Graviton-class
        // ARM. The pthread variant scales best across cores; install via
        // `sudo apt install libblis-pthread-dev`.
        println!("cargo:rustc-link-lib=dylib=blis-pthread");
    }
}
