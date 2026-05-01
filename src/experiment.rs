use anyhow::Result;
use clap::Subcommand;

use crate::dataset::Dataset;
use crate::quantization::experiment::QuantizationRecallExperiment;

pub trait Experiment {
    fn name(&self) -> &'static str;
    fn run(&self, data: &Dataset) -> Result<ExperimentOutput>;
}

pub struct ExperimentOutput {
    header: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl ExperimentOutput {
    pub fn new(header: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            header: header.into_iter().map(Into::into).collect(),
            rows: Vec::new(),
        }
    }

    pub fn push_row(&mut self, row: impl IntoIterator<Item = impl Into<String>>) {
        self.rows.push(row.into_iter().map(Into::into).collect());
    }

    pub fn print_csv(&self) {
        println!("{}", self.header.join(","));
        for row in &self.rows {
            println!("{}", row.join(","));
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum ExperimentKind {
    QuantizationRecall(QuantizationRecallExperiment),
}

impl ExperimentKind {
    pub fn into_experiment(self, dims: usize) -> Box<dyn Experiment> {
        match self {
            ExperimentKind::QuantizationRecall(mut experiment) => {
                experiment.set_dims(dims);
                Box::new(experiment)
            }
        }
    }
}
