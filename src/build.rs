fn main() {
    let target = std::env::var("TARGET").unwrap();

    // The Ryzen 5 3500U is a modern Zen+ architecture (which supports AVX2).
    // The "znver1" target profile is optimized for the first-generation Zen architecture (Ryzen 1000 series),
    // but often provides excellent results for Zen+, explicitly enabling AVX2/FMA/etc.
    // However, specifying target-cpu=native is often safer when the engine will run on the same CPU it was compiled on.
    
    // For cross-compilation simplicity while ensuring AVX2/FMA3 is active:
    let features = String::from("+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+avx,+avx2,+fma");
    
    if target.contains("x86_64") {
        println!("cargo:rustc-env=RUSTFLAGS=-C target-feature={}", features);
        println!("cargo:warning=Aether compilation targeting AVX2/FMA3 optimization.");
    }
}