use std::time::Instant;

use anyhow::{anyhow, Result};
use clap::Args;

use crate::dataset::Dataset;
use crate::experiment::{Experiment, ExperimentOutput};
use crate::metrics::{average_metrics, metrics_for};
use crate::quantization::compression_ratio;
use crate::quantization::factory::{
    build_quantizer, selected_quantizers, QuantizerKind, QuantizerVariant,
};

#[derive(Debug, Args)]
pub struct QuantizationRecallExperiment {
    /// Quantizer family to evaluate. `all` runs every implemented quantizer.
    #[arg(long, value_enum, default_value = "all")]
    quantizer: QuantizerKind,

    /// Quantizer implementation variant to run. `default` uses the recommended variant per quantizer.
    #[arg(long, value_enum, default_value = "default")]
    variant: QuantizerVariant,

    /// Recall/NDCG cutoffs to report from the same brute-force ranking.
    #[arg(long, value_delimiter = ',', default_value = "10,50,100")]
    k: Vec<usize>,

    /// Number of Cohere test queries to evaluate for this brute-force recall experiment.
    #[arg(long, default_value_t = 10)]
    queries: usize,

    /// Use transposed/batched scoring for quantizers that support it.
    #[arg(long, default_value_t = false)]
    transposed: bool,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Chunk size for PDX sequential clusters (`--quantizer pdx` only; ignored elsewhere).
    #[arg(long, default_value_t = 8192)]
    pdx_chunk_size: usize,
}

impl Experiment for QuantizationRecallExperiment {
    fn name(&self) -> &'static str {
        "quantization-recall"
    }

    fn run(&self, data: &Dataset) -> Result<ExperimentOutput> {
        let max_k = self
            .k
            .iter()
            .copied()
            .max()
            .ok_or_else(|| anyhow!("k_values must not be empty"))?;
        let query_count = self.queries.min(data.queries.len()).min(data.ground_truth.len());
        if query_count == 0 {
            return Err(anyhow!("queries must be greater than zero"));
        }

        let mut output = ExperimentOutput::new([
            "method",
            "variant",
            "layout",
            "bits",
            "bytes_per_vector",
            "total_bytes",
            "compression_x",
            "k",
            "recall",
            "ndcg",
            "encode_seconds",
            "query_seconds",
            "qps",
        ]);

        for spec in selected_quantizers(self.quantizer, self.variant)? {
            eprintln!(
                "encoding quantizer={:?} variant={:?} docs={}",
                spec.kind,
                spec.variant,
                data.docs.len()
            );
            let mut quantizer =
                build_quantizer(spec, data.dims, self.seed, self.pdx_chunk_size)?;
            let transposed_enabled = quantizer.set_transposed(self.transposed);

            let encode_start = Instant::now();
            quantizer.encode(&data.docs)?;
            let encode_seconds = encode_start.elapsed().as_secs_f64();

            eprintln!(
                "querying quantizer={} variant={} layout={} bits={} queries={} max_k={max_k}",
                quantizer.name(),
                quantizer.variant(),
                quantizer.scoring_layout(),
                quantizer.bits(),
                query_count
            );
            let query_start = Instant::now();
            let ranked_by_query: Vec<Vec<(u64, f32)>> = data
                .queries
                .iter()
                .take(query_count)
                .map(|query| quantizer.top_k(&data.doc_ids, query, max_k))
                .collect();
            let query_seconds = query_start.elapsed().as_secs_f64();
            let qps = query_count as f64 / query_seconds.max(1e-9);
            eprintln!(
                "finished quantizer={} variant={} layout={} bits={} encode_seconds={encode_seconds:.3} query_seconds={query_seconds:.3}",
                quantizer.name(),
                quantizer.variant(),
                quantizer.scoring_layout(),
                quantizer.bits()
            );

            let bytes = quantizer.bytes_per_vector();
            let total_bytes = match quantizer.total_bytes() {
                0 => bytes * data.docs.len(),
                total_bytes => total_bytes,
            };
            let compression = compression_ratio(data.dims, bytes);
            let layout = if self.transposed && !transposed_enabled {
                "doc-major-unsupported"
            } else {
                quantizer.scoring_layout()
            };
            for &k in &self.k {
                let per_query: Vec<_> = ranked_by_query
                    .iter()
                    .enumerate()
                    .map(|(qi, got)| metrics_for(&data.ground_truth[qi], got, k))
                    .collect();
                let avg = average_metrics(&per_query);
                output.push_row([
                    quantizer.name().to_string(),
                    quantizer.variant().to_string(),
                    layout.to_string(),
                    quantizer.bits().to_string(),
                    bytes.to_string(),
                    total_bytes.to_string(),
                    format!("{compression:.2}"),
                    k.to_string(),
                    format!("{:.4}", avg.recall),
                    format!("{:.4}", avg.ndcg),
                    format!("{encode_seconds:.3}"),
                    format!("{query_seconds:.3}"),
                    format!("{qps:.2}"),
                ]);
            }
        }

        Ok(output)
    }
}
