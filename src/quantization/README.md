# Quantization Experiments

This experiment measures the recall and encoding times of different quantization strategies.
Note that "recall" is order agnostic, it simply asks the question "what percentage of the
top K ground truth vectors are found in the top K quantized vectors?"

Implemented quantizers:

- `turboquant`: https://arxiv.org/abs/2504.19874
- `rabitq`: https://arxiv.org/abs/2405.12497
- `naivesq`: A simple scalar quantizer that ...

## Results

### 5/30/26

| Method | Variant | Bits | Bytes / Vector | Compression | Recall@10 | NDCG@10 | Encode Time |
|---|---|---:|---:|---:|---:|---:|---:|
| Naive SQ | default | 5 | 480 | 6.40x | 0.9100 | 0.9398 | 1.267s |
| TurboQuant | default | 5 | 484 | 6.35x | 0.8500 | 0.9006 | 17.138s |
| RaBitQ | fixed | 5 | 512 | 6.00x | 0.9300 | 0.9538 | 15.872s |
| RaBitQ | optimal | 5 | 512 | 6.00x | 0.9500 | 0.9659 | 78.150s |

On `rabitq`, note that it is encoded against a zero centroid to line up with Turboquant's data-obliviousness. There are two variants of `rabitq`:

- `fixed`: uses a precomputed scaling factor shared by all vectors. This avoids
  the expensive per-vector scale search and is the recommended default for fast
  encoding.
- `optimal`: computes the scaling factor independently for every vector. This
  can improve recall slightly, but is much slower to encode because it runs a
  per-vector optimization.

Surprisingly, and contrary to the Turboquant paper, we have a variant of Rabitq which beats Turboquant on both recall and encode time.

## Usage

Run all quantizers at the default bit widths:

```sh
cargo run --release -- quantization-recall --quantizer all --variant default --bits 4,5,6,8 --k 10,50,100 --queries 10
```
