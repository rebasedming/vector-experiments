use anyhow::{bail, Result};
use clap::ValueEnum;

use crate::quantization::naivesq::NaiveSqQuantizer;
use crate::quantization::rabitq::bench::{RabitqBench, RabitqVariant};
use crate::quantization::turboquant::bench::{TurboQuantBench, TurboQuantVariant};
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
    /// Use the dense Gaussian QJL projection from the TurboQuant paper where supported.
    GaussianQjl,
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
        QuantizerKind::Turboquant => turboquant_specs(variant),
        QuantizerKind::Rabitq => rabitq_specs(variant),
        QuantizerKind::Naivesq => default_only(kind, variant),
        QuantizerKind::All => {
            let mut specs = Vec::new();
            specs.extend(turboquant_specs(match variant {
                QuantizerVariant::All => QuantizerVariant::All,
                QuantizerVariant::GaussianQjl => QuantizerVariant::GaussianQjl,
                _ => QuantizerVariant::Default,
            })?);
            specs.extend(rabitq_specs(match variant {
                QuantizerVariant::All => QuantizerVariant::All,
                QuantizerVariant::GaussianQjl => QuantizerVariant::Default,
                other => other,
            })?);
            specs.extend(default_only(QuantizerKind::Naivesq, QuantizerVariant::Default)?);
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
        QuantizerVariant::Fixed | QuantizerVariant::Optimal | QuantizerVariant::GaussianQjl => {
            bail!("{kind:?} does not support the {variant:?} variant")
        }
    }
}

fn turboquant_specs(variant: QuantizerVariant) -> Result<Vec<QuantizerSpec>> {
    let specs = match variant {
        QuantizerVariant::Default => vec![QuantizerSpec {
            kind: QuantizerKind::Turboquant,
            variant: QuantizerVariant::Default,
        }],
        QuantizerVariant::GaussianQjl => vec![QuantizerSpec {
            kind: QuantizerKind::Turboquant,
            variant: QuantizerVariant::GaussianQjl,
        }],
        QuantizerVariant::All => vec![
            QuantizerSpec {
                kind: QuantizerKind::Turboquant,
                variant: QuantizerVariant::Default,
            },
            QuantizerSpec {
                kind: QuantizerKind::Turboquant,
                variant: QuantizerVariant::GaussianQjl,
            },
        ],
        QuantizerVariant::Fixed | QuantizerVariant::Optimal => {
            bail!("Turboquant does not support the {variant:?} variant")
        }
    };
    Ok(specs)
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
        QuantizerVariant::GaussianQjl => bail!("Rabitq does not support the GaussianQjl variant"),
    };
    Ok(specs)
}

/// Turboquant / RaBitQ / NaiveSQ experiments run at **5 bits** (optimized SIMD paths).
pub const RECALL_QUANT_BITS: u8 = 5;

pub fn build_quantizer(
    spec: QuantizerSpec,
    dims: usize,
    seed: u64,
    _pdx_chunk_size: usize,
) -> Result<Box<dyn VectorQuantizer>> {
    match spec.kind {
        QuantizerKind::Turboquant => {
            let variant = match spec.variant {
                QuantizerVariant::Default => TurboQuantVariant::Srht,
                QuantizerVariant::GaussianQjl => TurboQuantVariant::GaussianQjl,
                QuantizerVariant::Fixed | QuantizerVariant::Optimal | QuantizerVariant::All => {
                    bail!("internal error: unresolved turboquant variant")
                }
            };
            Ok(Box::new(TurboQuantBench::new(dims, seed, variant)))
        }
        QuantizerKind::Rabitq => {
            let variant = match spec.variant {
                QuantizerVariant::Default | QuantizerVariant::Fixed => RabitqVariant::Fixed,
                QuantizerVariant::Optimal => RabitqVariant::Optimal,
                QuantizerVariant::GaussianQjl | QuantizerVariant::All => {
                    bail!("internal error: unresolved rabitq variant")
                }
            };
            Ok(Box::new(RabitqBench::new(dims, seed, variant)))
        }
        QuantizerKind::Naivesq => Ok(Box::new(NaiveSqQuantizer::new(dims))),
        QuantizerKind::All => bail!("internal error: build_quantizer called with all"),
    }
}
