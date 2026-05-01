#![allow(dead_code, unused_imports)]

mod dataset;
mod experiment;
mod metrics;
mod quantization;

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use crate::dataset::{default_cohere_dir, load_cohere, COHERE_DIMS};
use crate::experiment::ExperimentKind;

#[derive(Debug, Parser)]
#[command(about = "Experiments over the VectorDBBench Cohere 1M dataset")]
struct Args {
    /// Optional smoke-test cap on corpus vectors. Leave unset for the full Cohere 1M corpus.
    #[arg(long)]
    limit: Option<usize>,

    /// Download missing Cohere parquet files into dataset_dir before loading.
    #[arg(long)]
    fetch: bool,

    /// Directory containing shuffle_train.parquet, test.parquet, and neighbors.parquet.
    #[arg(long)]
    dataset_dir: Option<PathBuf>,

    /// Experiment to run. Experiment-specific flags come after the subcommand.
    #[command(subcommand)]
    experiment: ExperimentKind,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dataset_dir = args.dataset_dir.unwrap_or_else(default_cohere_dir);
    let experiment = args.experiment.into_experiment(COHERE_DIMS);

    eprintln!(
        "loading Cohere dataset: dir={} doc_limit={} dims={}",
        dataset_dir.display(),
        args.limit
            .map(|limit| limit.to_string())
            .unwrap_or_else(|| "full".to_string()),
        COHERE_DIMS,
    );
    let data = load_cohere(&dataset_dir, args.limit, args.fetch)?;

    eprintln!("running experiment: {}", experiment.name());
    experiment.run(&data)?.print_csv();

    Ok(())
}
