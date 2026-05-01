# vector-experiments

Standalone Rust experiments over the VectorDBBench Cohere 1M dataset.

The repo is structured around a generic experiment runner. Each experiment owns
its CLI arguments, work loop, measurements, and output format.

## Experiments

Experiments live as folders/modules under `src/`. The generic harness in
`src/main.rs` loads the Cohere dataset once, selects the requested experiment
from the CLI, and then delegates the actual work to that experiment.

To add a new experiment:

1. Create a new module folder under `src/`, for example `src/my_experiment/`.
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

## Dataset

By default the harness looks for Cohere 1M at:

```text
/tmp/vectordb_bench/dataset/cohere/cohere_medium_1m
```

Expected files:

```text
shuffle_train.parquet
test.parquet
neighbors.parquet
```

The loader uses Cohere's fixed 768 dimensions and normalizes vectors for
inner-product/cosine evaluation.

If the files are missing, pass `--fetch` to download them from VectorDBBench's
public asset path.

You can also point at an existing directory:

```sh
cargo run --release -- --dataset-dir /path/to/cohere_medium_1m <experiment>
```

Status/progress messages are printed to stderr. Experiment results are printed
to stdout.
