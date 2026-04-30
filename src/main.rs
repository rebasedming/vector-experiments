#![allow(dead_code, unused_imports)]

mod dataset;
mod quantization;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Result};
use clap::{Parser, ValueEnum};

use crate::dataset::{default_cohere_dir, load_cohere, metrics_for};
use crate::quantization::naivesq::NaiveSqQuantizer;
use crate::quantization::rabitq::bench::RabitqBench;
use crate::quantization::turboquant::bench::TurboQuantBench;
use crate::quantization::{compression_ratio, VectorQuantizer};

#[derive(Debug, Clone, ValueEnum)]
enum QuantizerKind {
    Turboquant,
    Rabitq,
    Naivesq,
    All,
}

#[derive(Debug, Parser)]
#[command(about = "Recall vs compression experiments for vector quantizers")]
struct Args {
    #[arg(long, value_enum, default_value = "all")]
    quantizer: QuantizerKind,

    #[arg(long, value_delimiter = ',', default_value = "4,5,6,8")]
    bits: Vec<u8>,

    #[arg(long, value_delimiter = ',', default_value = "10,50,100")]
    k: Vec<usize>,

    #[arg(long, default_value_t = 1_000_000)]
    n: usize,

    #[arg(long, default_value_t = 100)]
    queries: usize,

    #[arg(long, default_value_t = 768)]
    dims: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = true)]
    normalize: bool,

    #[arg(long, default_value_t = false)]
    fetch: bool,

    #[arg(long)]
    dataset_dir: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let max_k = *args
        .k
        .iter()
        .max()
        .ok_or_else(|| anyhow::anyhow!("--k must not be empty"))?;
    let dataset_dir = args.dataset_dir.unwrap_or_else(default_cohere_dir);

    eprintln!(
        "loading Cohere dataset: dir={} n={} queries={} dims={} max_k={}",
        dataset_dir.display(),
        args.n,
        args.queries,
        args.dims,
        max_k
    );
    let data = load_cohere(
        &dataset_dir,
        args.n,
        args.dims,
        args.queries,
        max_k,
        args.normalize,
        args.fetch,
    )?;

    println!(
        "method,bits,bytes_per_vector,compression_x,k,recall,ndcg,encode_seconds,query_seconds,qps"
    );

    for kind in selected_quantizers(&args.quantizer) {
        let bits = bits_for_kind(kind, &args.bits);
        for bit_width in bits {
            let mut quantizer = build_quantizer(kind, args.dims, bit_width, args.seed)?;
            let encode_start = Instant::now();
            quantizer.encode(&data.docs)?;
            let encode_seconds = encode_start.elapsed().as_secs_f64();

            let query_start = Instant::now();
            let ranked_by_query: Vec<Vec<(u64, f32)>> = data
                .queries
                .iter()
                .take(args.queries)
                .map(|query| quantizer.top_k(&data.doc_ids, query, max_k))
                .collect();
            let query_seconds = query_start.elapsed().as_secs_f64();
            let qps = args.queries as f64 / query_seconds.max(1e-9);

            let bytes = quantizer.bytes_per_vector();
            let compression = compression_ratio(args.dims, bytes);
            for &k in &args.k {
                let metrics: Vec<_> = ranked_by_query
                    .iter()
                    .enumerate()
                    .map(|(qi, got)| metrics_for(&data.ground_truth[qi], got, k))
                    .collect();
                let recall = metrics.iter().map(|m| m.0).sum::<f32>() / metrics.len() as f32;
                let ndcg = metrics.iter().map(|m| m.1).sum::<f32>() / metrics.len() as f32;
                println!(
                    "{},{},{},{:.2},{},{:.4},{:.4},{:.3},{:.3},{:.2}",
                    quantizer.name(),
                    quantizer.bits(),
                    bytes,
                    compression,
                    k,
                    recall,
                    ndcg,
                    encode_seconds,
                    query_seconds,
                    qps
                );
            }
        }
    }

    Ok(())
}

fn selected_quantizers(kind: &QuantizerKind) -> Vec<&'static QuantizerKind> {
    static TURBO: QuantizerKind = QuantizerKind::Turboquant;
    static RABITQ: QuantizerKind = QuantizerKind::Rabitq;
    static NAIVESQ: QuantizerKind = QuantizerKind::Naivesq;
    match kind {
        QuantizerKind::Turboquant => vec![&TURBO],
        QuantizerKind::Rabitq => vec![&RABITQ],
        QuantizerKind::Naivesq => vec![&NAIVESQ],
        QuantizerKind::All => vec![&TURBO, &RABITQ, &NAIVESQ],
    }
}

fn bits_for_kind(kind: &QuantizerKind, requested: &[u8]) -> Vec<u8> {
    match kind {
        QuantizerKind::Rabitq => requested.to_vec(),
        QuantizerKind::Turboquant | QuantizerKind::Naivesq | QuantizerKind::All => requested
            .iter()
            .copied()
            .filter(|bits| *bits >= 1 && *bits <= 8)
            .collect(),
    }
}

fn build_quantizer(
    kind: &QuantizerKind,
    dims: usize,
    bits: u8,
    seed: u64,
) -> Result<Box<dyn VectorQuantizer>> {
    match kind {
        QuantizerKind::Turboquant => Ok(Box::new(TurboQuantBench::new(dims, bits, seed))),
        QuantizerKind::Rabitq => Ok(Box::new(RabitqBench::new(dims, bits, seed))),
        QuantizerKind::Naivesq => Ok(Box::new(NaiveSqQuantizer::new(dims, bits))),
        QuantizerKind::All => bail!("internal error: build_quantizer called with all"),
    }
}
