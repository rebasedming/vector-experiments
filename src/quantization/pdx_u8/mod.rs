//! PDX-style global **U8** scalar quantization + **PDX cluster layout** (cwida/PDX).

pub mod adsampling;
pub mod bench;
pub mod distance_u8;
pub mod layout;
pub mod search_pipeline;

pub use bench::PdxU8Bench;
pub use search_pipeline::{FlatPdxU8SearchIndex, SearchScratch};
