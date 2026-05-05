#![allow(dead_code, unused_imports)]

#[cfg(target_os = "macos")]
extern crate accelerate_src;
// Linux/aarch64: BLIS is linked via build.rs.

mod dataset;
mod experiment;
mod kmeans;
mod metrics;
mod quantization;

use std::path::PathBuf;

use anyhow::Result;
use clap::{ArgAction, Parser};

use crate::dataset::{default_dataset_root, load_dataset, DatasetArg};
use crate::experiment::ExperimentKind;

#[derive(Debug, Parser)]
#[command(about = "Experiments over default VectorDBBench embedding datasets")]
struct Args {
    /// Optional smoke-test cap on corpus vectors per dataset. Leave unset for full corpora.
    #[arg(long)]
    limit: Option<usize>,

    /// Dataset to run. Defaults to all built-in benchmark datasets.
    #[arg(long, value_enum, default_value = "all")]
    dataset: DatasetArg,

    /// Do not download missing parquet files before loading.
    #[arg(long = "no-fetch", action = ArgAction::SetFalse, default_value_t = true)]
    fetch: bool,

    /// Root containing VectorDBBench dataset folders, grouped by dataset source.
    #[arg(long)]
    dataset_root: Option<PathBuf>,

    /// Experiment to run. Experiment-specific flags come after the subcommand.
    #[command(subcommand)]
    experiment: ExperimentKind,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dataset_root = args.dataset_root.unwrap_or_else(default_dataset_root);
    let experiment = args.experiment.into_experiment();

    for spec in args.dataset.specs() {
        let dataset_dir = spec.data_dir(&dataset_root);
        eprintln!(
            "loading {} dataset: dir={} doc_limit={} dims={}",
            spec.name,
            dataset_dir.display(),
            args.limit
                .map(|limit| limit.to_string())
                .unwrap_or_else(|| "full".to_string()),
            spec.dims,
        );
        let data = load_dataset(spec, &dataset_dir, args.limit, args.fetch)?;

        eprintln!(
            "running experiment: {} dataset={}",
            experiment.name(),
            data.name
        );
        println!("\n## {}", data.name);
        experiment.run(&data)?.print_table();
    }

    Ok(())
}
