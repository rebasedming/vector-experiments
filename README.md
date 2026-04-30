# vector-experiments

Standalone Rust experiments for comparing vector quantization recall and compression.

The current benchmark targets the VectorDBBench Cohere 1M dataset and compares:

- `turboquant`
- `rabitq`
- `naivesq` (per-dimension min/max scalar quantization)

All three quantizers implement the shared `VectorQuantizer` trait in `src/quantization/mod.rs`.

## Layout

```text
src/
  dataset.rs
  main.rs
  quantization/
    naivesq/
    rabitq/
    turboquant/
```

`rabitq` and `turboquant` are lifted from the Tantivy vector experiments. `naivesq`
is the simple calibrated scalar baseline.

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

If the files are missing, pass `--fetch` to download them from VectorDBBench's
public asset path:

```sh
cargo run --release -- --fetch --n 1000000 --queries 100
```

You can also point at an existing directory:

```sh
cargo run --release -- --dataset-dir /path/to/cohere_medium_1m
```

## Examples

Run all quantizers at the default bit widths:

```sh
cargo run --release -- --quantizer all --bits 4,5,6,8 --k 10,50,100 --n 1000000 --queries 100
```

Run the zero-centroid RaBitQ sweep:

```sh
cargo run --release -- --quantizer rabitq --bits 1,2,3,4,5,6,8 --k 10,50,100 --n 1000000 --queries 100
```

Output is CSV on stdout:

```text
method,bits,bytes_per_vector,compression_x,k,recall,ndcg,encode_seconds,query_seconds,qps
```

Status/progress messages are printed to stderr.
