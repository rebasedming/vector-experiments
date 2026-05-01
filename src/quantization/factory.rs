use anyhow::{bail, Result};
use clap::ValueEnum;

use crate::quantization::naivesq::NaiveSqQuantizer;
use crate::quantization::rabitq::bench::{RabitqBench, RabitqVariant};
use crate::quantization::turboquant::bench::TurboQuantBench;
use crate::quantization::VectorQuantizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum QuantizerKind {
    Turboquant,
    Rabitq,
    Naivesq,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum QuantizerVariant {
    /// Use each quantizer's recommended/default variant.
    Default,
    /// Use fixed/precomputed quantization parameters where supported.
    Fixed,
    /// Use per-vector optimal quantization parameters where supported.
    Optimal,
    /// Run every variant supported by the selected quantizer.
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizerSpec {
    pub kind: QuantizerKind,
    pub variant: QuantizerVariant,
}

pub fn selected_quantizers(
    kind: QuantizerKind,
    variant: QuantizerVariant,
) -> Result<Vec<QuantizerSpec>> {
    match kind {
        QuantizerKind::Turboquant => default_only(kind, variant),
        QuantizerKind::Rabitq => rabitq_specs(variant),
        QuantizerKind::Naivesq => default_only(kind, variant),
        QuantizerKind::All => {
            let mut specs = Vec::new();
            specs.extend(default_only(
                QuantizerKind::Turboquant,
                QuantizerVariant::Default,
            )?);
            specs.extend(rabitq_specs(variant)?);
            specs.extend(default_only(
                QuantizerKind::Naivesq,
                QuantizerVariant::Default,
            )?);
            Ok(specs)
        }
    }
}

fn default_only(kind: QuantizerKind, variant: QuantizerVariant) -> Result<Vec<QuantizerSpec>> {
    match variant {
        QuantizerVariant::Default | QuantizerVariant::All => Ok(vec![QuantizerSpec {
            kind,
            variant: QuantizerVariant::Default,
        }]),
        QuantizerVariant::Fixed | QuantizerVariant::Optimal => {
            bail!("{kind:?} does not support the {variant:?} variant")
        }
    }
}

fn rabitq_specs(variant: QuantizerVariant) -> Result<Vec<QuantizerSpec>> {
    let specs = match variant {
        QuantizerVariant::Default | QuantizerVariant::Fixed => vec![QuantizerSpec {
            kind: QuantizerKind::Rabitq,
            variant: QuantizerVariant::Fixed,
        }],
        QuantizerVariant::Optimal => vec![QuantizerSpec {
            kind: QuantizerKind::Rabitq,
            variant: QuantizerVariant::Optimal,
        }],
        QuantizerVariant::All => vec![
            QuantizerSpec {
                kind: QuantizerKind::Rabitq,
                variant: QuantizerVariant::Fixed,
            },
            QuantizerSpec {
                kind: QuantizerKind::Rabitq,
                variant: QuantizerVariant::Optimal,
            },
        ],
    };
    Ok(specs)
}

pub fn bits_for_kind(spec: QuantizerSpec, requested: &[u8]) -> Vec<u8> {
    match spec.kind {
        QuantizerKind::Rabitq => requested.to_vec(),
        QuantizerKind::Turboquant | QuantizerKind::Naivesq | QuantizerKind::All => requested
            .iter()
            .copied()
            .filter(|bits| *bits >= 1 && *bits <= 8)
            .collect(),
    }
}

pub fn build_quantizer(
    spec: QuantizerSpec,
    dims: usize,
    bits: u8,
    seed: u64,
) -> Result<Box<dyn VectorQuantizer>> {
    match spec.kind {
        QuantizerKind::Turboquant => Ok(Box::new(TurboQuantBench::new(dims, bits, seed))),
        QuantizerKind::Rabitq => {
            let variant = match spec.variant {
                QuantizerVariant::Default | QuantizerVariant::Fixed => RabitqVariant::Fixed,
                QuantizerVariant::Optimal => RabitqVariant::Optimal,
                QuantizerVariant::All => bail!("internal error: unresolved rabitq variant"),
            };
            Ok(Box::new(RabitqBench::new(dims, bits, seed, variant)))
        }
        QuantizerKind::Naivesq => Ok(Box::new(NaiveSqQuantizer::new(dims, bits))),
        QuantizerKind::All => bail!("internal error: build_quantizer called with all"),
    }
}
