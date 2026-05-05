fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_os == "linux" && target_arch == "aarch64" {
        // Link against BLIS for the cblas_sgemm symbol on Graviton-class
        // ARM. Install via `sudo apt install libblis-pthread-dev`. The
        // Ubuntu package installs the pthread-flavored libblis.so.4 under
        // /usr/lib/aarch64-linux-gnu/blis-pthread/, so we add that to
        // the link search path; the soname linked is plain `blis`.
        println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu/blis-pthread");
        println!("cargo:rustc-link-lib=dylib=blis");
    }
}
