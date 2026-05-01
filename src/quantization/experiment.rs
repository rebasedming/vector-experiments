use std::time::Instant;

use anyhow::{anyhow, Result};
use clap::Args;

use crate::dataset::Dataset;
use crate::experiment::{Experiment, ExperimentOutput};
use crate::metrics::{average_metrics, metrics_for};
use crate::quantization::compression_ratio;
use crate::quantization::factory::{
    bits_for_kind, build_quantizer, selected_quantizers, QuantizerKind, QuantizerVariant,
};

#[derive(Debug, Args)]
pub struct QuantizationRecallExperiment {
    /// Quantizer family to evaluate. `all` runs every implemented quantizer.
    #[arg(long, value_enum, default_value = "all")]
    quantizer: QuantizerKind,

    /// Quantizer implementation variant to run. `default` uses the recommended variant per quantizer.
    #[arg(long, value_enum, default_value = "default")]
    variant: QuantizerVariant,

    /// Quantizer bit widths to sweep. Unsupported widths are filtered per quantizer.
    #[arg(long, value_delimiter = ',', default_value = "4,5,6,8")]
    bits: Vec<u8>,

    /// Recall/NDCG cutoffs to report from the same brute-force ranking.
    #[arg(long, value_delimiter = ',', default_value = "10,50,100")]
    k: Vec<usize>,

    /// Number of Cohere test queries to evaluate for this brute-force recall experiment.
    #[arg(long, default_value_t = 10)]
    queries: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(skip = 768usize)]
    dims: usize,
}

impl QuantizationRecallExperiment {
    pub fn set_dims(&mut self, dims: usize) {
        self.dims = dims;
    }
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
        let query_count = self
            .queries
            .min(data.queries.len())
            .min(data.ground_truth.len());
        if query_count == 0 {
            return Err(anyhow!("queries must be greater than zero"));
        }

        let mut output = ExperimentOutput::new([
            "method",
            "variant",
            "bits",
            "bytes_per_vector",
            "compression_x",
            "k",
            "recall",
            "ndcg",
            "encode_seconds",
            "query_seconds",
            "qps",
        ]);

        for spec in selected_quantizers(self.quantizer, self.variant)? {
            let bits = bits_for_kind(spec, &self.bits);
            for bit_width in bits {
                eprintln!(
                    "encoding quantizer={:?} variant={:?} bits={bit_width} docs={}",
                    spec.kind,
                    spec.variant,
                    data.docs.len()
                );
                let mut quantizer = build_quantizer(spec, self.dims, bit_width, self.seed)?;

                let encode_start = Instant::now();
                quantizer.encode(&data.docs)?;
                let encode_seconds = encode_start.elapsed().as_secs_f64();

                eprintln!(
                    "querying quantizer={} variant={} bits={} queries={} max_k={max_k}",
                    quantizer.name(),
                    quantizer.variant(),
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
                    "finished quantizer={} variant={} bits={} encode_seconds={encode_seconds:.3} query_seconds={query_seconds:.3}",
                    quantizer.name(),
                    quantizer.variant(),
                    quantizer.bits()
                );

                let bytes = quantizer.bytes_per_vector();
                let compression = compression_ratio(self.dims, bytes);
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
                        quantizer.bits().to_string(),
                        bytes.to_string(),
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
        }

        Ok(output)
    }
}
