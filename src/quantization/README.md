# Quantization Experiments

The `quantization-recall` experiment compares quantized brute-force search
against full-precision Cohere 1M ground truth. It reports recall, NDCG,
compression, and encode time for the selected quantizers.

Implemented quantizers:

- `turboquant`: [TurboQuant](https://arxiv.org/abs/2504.19874).
- `rabitq`: [RaBitQ](https://arxiv.org/abs/2405.12497).
- `naivesq`: A simple scalar quantizer that learns per-dimension min/max
  ranges from the corpus, uniformly quantizes each coordinate, and bit-packs
  the resulting codes.

All three quantizers implement the shared `VectorQuantizer` trait in
`src/quantization/mod.rs`.

## Structure

- `src/quantization/experiment.rs`: recall/compression experiment loop.
- `src/quantization/factory.rs`: quantizer and variant selection.
- `src/quantization/naivesq/`: naive scalar quantization baseline.
- `src/quantization/rabitq/`: RaBitQ implementation and benchmark adapter.
- `src/quantization/turboquant/`: TurboQuant implementation and benchmark adapter.

## Usage

Run all quantizers at the default bit widths:

```sh
cargo run --release -- quantization-recall --quantizer all --variant default --k 10,50,100 --queries 10
```

Use `--limit` before the subcommand for quick smoke tests. Full benchmark runs
should leave it unset:

```sh
cargo run --release -- --limit 100 quantization-recall --quantizer all --variant default --bits 1 --k 10 --queries 10
```

Run the zero-centroid RaBitQ variants:

```sh
cargo run --release -- quantization-recall --quantizer rabitq --variant all --bits 5 --k 10 --queries 10
```

Run the zero-centroid RaBitQ bit sweep:

```sh
cargo run --release -- quantization-recall --quantizer rabitq --variant all --bits 1,2,3,4,5,6,8 --k 10,50,100 --queries 10
```

Output is a table on stdout with these columns:

```text
method,variant,bits,bytes_per_vector,total_bytes,compression_x,k,recall,ndcg,encode_seconds,query_seconds,qps
```

## RaBitQ Variants

`rabitq` supports two encode variants:

- `fixed`: uses a precomputed scaling factor shared by all vectors. This avoids
  the expensive per-vector scale search and is the recommended default for fast
  encoding.
- `optimal`: computes the scaling factor independently for every vector. This
  can improve recall slightly, but is much slower to encode because it runs a
  per-vector optimization.

Both variants produce the same record size for a given bit width, so compression
is unchanged. Query time is also expected to be similar because the packed record
shape is the same.

## Latest 5-Bit Cohere 1M Result

This run used 1M corpus vectors, 10 test queries, `k=10`, and normalized Cohere
vectors. Query-time stats are omitted here because this table focuses on
recall/compression and encode cost.

| Method | Variant | Bits | Bytes / Vector | Compression | Recall@10 | NDCG@10 | Encode Time |
|---|---|---:|---:|---:|---:|---:|---:|
| Naive SQ | default | 5 | 480 | 6.40x | 0.9100 | 0.9398 | 1.267s |
| TurboQuant | default | 5 | 484 | 6.35x | 0.8500 | 0.9006 | 17.138s |
| RaBitQ | fixed | 5 | 512 | 6.00x | 0.9300 | 0.9538 | 15.872s |
| RaBitQ | optimal | 5 | 512 | 6.00x | 0.9500 | 0.9659 | 78.150s |
