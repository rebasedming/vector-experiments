# vector-experiments

Standalone Rust experiments over **VectorDBBench** embedding benchmarks shipped with this repo’s CLI.

The harness loads one or more datasets (your choice), runs the selected experiment on each corpus in turn, and prints a Markdown section header (`## <dataset>`) plus tables on stdout.

Each experiment owns its CLI arguments, work loop, measurements, and output format.

## Experiments

Experiment modules live under `src/`. `src/main.rs` parses shared dataset flags, loads embeddings and ground-truth neighbors, then delegates to the experiment implementation.

To add a new experiment:

1. Create a module folder under `src/`, for example `src/my_experiment/`.
2. Define a clap args struct for the experiment, usually with `#[derive(Debug, Args)]`.
3. Implement the `Experiment` trait from `src/experiment.rs`:

```rust
pub trait Experiment {
    fn name(&self) -> &'static str;
    fn run(&self, data: &Dataset) -> Result<ExperimentOutput>;
}
```

4. Return results through `ExperimentOutput`, which owns the CSV header and rows.
5. Add a variant to `ExperimentKind` in `src/experiment.rs`, then dispatch it in
   `ExperimentKind::into_experiment`.

Experiment-specific implementation details and CLI flags should stay inside the
experiment module. The top-level CLI should remain limited to dataset loading
and experiment selection.

Current experiment docs:

- [`src/quantization/README.md`](src/quantization/README.md): quantization recall/compression benchmarks.

## Datasets

Built-in presets mirror **VectorDBBench “medium”** corpora (see `DatasetSpec` in `src/dataset.rs`). Paths are rooted at **`--dataset-root`**, which defaults to:

```text
/tmp/vectordb_bench/dataset
```

Each corpus lives under `<dataset_root>/<source>/<corpus_dir>/` with the same three parquet files:

```text
shuffle_train.parquet
test.parquet
neighbors.parquet
```

| CLI `--dataset` value | Corpus directory | Rough scale | Embedding dims | Notes |
|----------------------|------------------|-------------|----------------|-------|
| `cohere-medium` | `cohere/cohere_medium_1m` | ~1M docs | 768 | Wikipedia text, **Cohere V2** embeddings. |
| `openai-medium` | `openai/openai_medium_500k` | ~500K docs | 1536 | C4 web crawl, **OpenAI** embeddings. |
| `bioasq-medium` | `bioasq/bioasq_medium_1m` | ~1M docs | 1024 | BioASQ biomedical text, **Cohere V3** embeddings. |

Use **`--dataset all`** (default) to run the experiment sequentially on **all three** presets above.

The loader normalizes corpus and query vectors so inner-product scoring matches cosine-like retrieval used for these benchmarks.

If parquet files are missing, downloads are attempted automatically from VectorDBBench’s public asset URLs (unless you pass **`--no-fetch`**).

Example:

```sh
# Single corpus under the default root
cargo run --release -- --dataset cohere-medium quantization-recall --quantizer turboquant

# Custom root; smoke-test with a doc cap
cargo run --release -- --dataset-root /data/vectordb_bench/dataset --dataset openai-medium --limit 10000 quantization-recall

# Run every built-in dataset (default --dataset all)
cargo run --release -- quantization-recall --quantizer all
```

Status and progress messages go to stderr; experiment tables go to stdout.
