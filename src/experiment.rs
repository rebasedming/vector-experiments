use anyhow::Result;
use clap::Subcommand;

use crate::dataset::Dataset;
use crate::kmeans::experiment::KMeansExperiment;
use crate::quantization::experiment::QuantizationRecallExperiment;

pub trait Experiment {
    fn name(&self) -> &'static str;
    fn run(&self, data: &Dataset) -> Result<ExperimentOutput>;
}

pub struct ExperimentOutput {
    header: Vec<String>,
    rows: Vec<Vec<String>>,
    /// Extra pre-formatted sections appended after the main table (e.g. a
    /// recall sub-table with a different column structure).
    extras: Vec<String>,
}

impl ExperimentOutput {
    pub fn new(header: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            header: header.into_iter().map(Into::into).collect(),
            rows: Vec::new(),
            extras: Vec::new(),
        }
    }

    pub fn push_row(&mut self, row: impl IntoIterator<Item = impl Into<String>>) {
        self.rows.push(row.into_iter().map(Into::into).collect());
    }

    pub fn push_extra(&mut self, section: impl Into<String>) {
        self.extras.push(section.into());
    }

    pub fn print_csv(&self) {
        println!("{}", self.header.join(","));
        for row in &self.rows {
            println!("{}", row.join(","));
        }
    }

    pub fn print_table(&self) {
        let mut widths: Vec<usize> = self.header.iter().map(|cell| cell.len()).collect();
        for row in &self.rows {
            for (idx, cell) in row.iter().enumerate() {
                widths[idx] = widths[idx].max(cell.len());
            }
        }

        let numeric = (0..self.header.len())
            .map(|idx| self.rows.iter().all(|row| row[idx].parse::<f64>().is_ok()))
            .collect::<Vec<_>>();

        print_row(&self.header, &widths, &numeric);
        print_separator(&widths);
        for row in &self.rows {
            print_row(row, &widths, &numeric);
        }
        for extra in &self.extras {
            println!();
            print!("{extra}");
        }
    }
}

fn print_row(row: &[String], widths: &[usize], numeric: &[bool]) {
    print!("|");
    for (idx, cell) in row.iter().enumerate() {
        if numeric[idx] {
            print!(" {:>width$} |", cell, width = widths[idx]);
        } else {
            print!(" {:<width$} |", cell, width = widths[idx]);
        }
    }
    println!();
}

fn print_separator(widths: &[usize]) {
    print!("|");
    for width in widths {
        print!("{}|", "-".repeat(*width + 2));
    }
    println!();
}

#[derive(Debug, Subcommand)]
pub enum ExperimentKind {
    QuantizationRecall(QuantizationRecallExperiment),
    /// Train SuperKMeans on each dataset's corpus and report per-iteration stats.
    #[command(name = "kmeans")]
    KMeans(KMeansExperiment),
}

impl ExperimentKind {
    pub fn into_experiment(self) -> Box<dyn Experiment> {
        match self {
            ExperimentKind::QuantizationRecall(experiment) => Box::new(experiment),
            ExperimentKind::KMeans(experiment) => Box::new(experiment),
        }
    }
}
