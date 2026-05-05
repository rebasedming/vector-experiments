# K-Means Clustering Experiments

Benchmarking k-means variants for IVF index construction on the Cohere-medium dataset (1M vectors, d=768).

## Usage

```bash
# Flat Lloyd's k-means, k=1000 (centroid_ratio × n)
cargo run --release -- --dataset cohere-medium kmeans --centroid-ratio 0.001 --iters 5

# Hierarchical balanced bisection (required for large k)
cargo run --release -- --dataset cohere-medium kmeans --centroid-ratio 0.001 --hierarchical

# With boundary duplication for higher recall
cargo run --release -- --dataset cohere-medium kmeans --centroid-ratio 0.001 \
  --hierarchical --duplicate-max 4 --duplicate-ratio 5.0

# Sweep multiple centroid ratios, custom recall evaluation
cargo run --release -- --dataset cohere-medium kmeans \
  --centroid-ratio 0.001,0.005,0.01 --recall-k 1,10,100 --recall-n 0.001,0.01,0.05
```

## How it works

### Flat k-means (vanilla)

Standard Lloyd's algorithm. Pick k initial centroids, then iterate: assign every vector to its nearest centroid, recompute centroids as cluster means.

### Hierarchical balanced bisection (`--hierarchical`)

Instead of running Lloyd's with all k centroids at once, recursively split the dataset in half using k=2 Lloyd's until there are k leaf clusters. Each split sorts vectors by signed distance to the bisecting hyperplane and cuts at the median, guaranteeing balanced cluster sizes. The result is a binary tree where leaves map to clusters.

Training is O(n × d × log k) instead of O(n × k × d × iters), making it feasible at large k (100K+) where flat Lloyd's is too slow.

### Boundary duplication (`--duplicate-max`, `--duplicate-ratio`)

[SPANN](https://arxiv.org/abs/2111.08566) idea -- after primary assignment, vectors near cluster boundaries are copied into neighboring clusters. A vector gets duplicated into a runner-up cluster if its distance to that centroid is within `duplicate_ratio` of its primary centroid's distance. Each vector appears in at most `duplicate_max + 1` clusters total.

### Recall evaluation

For each query, find the top-N nearest centroids (by fanout fraction), collect all vectors assigned to those clusters, and check what fraction of the true k-nearest neighbors are covered. This simulates IVF search without building the full index.

## Knobs

| Flag | Effect |
|------|--------|
| `--centroid-ratio` | k = n × ratio. Higher = better recall, more memory, slower flat training |
| `--hierarchical` | Use balanced bisection tree instead of flat Lloyd's. Required for k > ~10K |
| `--duplicate-max` | Max additional clusters per vector (0 = disabled). More = better recall, more disk |
| `--duplicate-ratio` | Distance ratio threshold for duplication. Lower = more aggressive duplication |
| `--recall-n` | Fanout fractions to evaluate (fraction of clusters probed per query) |
| `--iters` | Lloyd's iterations (flat only, ignored with `--hierarchical`) |

## Key findings

### Centroid ratio drives the recall/speed tradeoff

More centroids = better recall at the same fanout, but slower training and more memory for centroids.

| ratio | k | train time | r@10 at 1% fanout | centroid memory | cluster balance (min/med/max) |
|-------|---|------------|-------------------|-----------------|-------------------------------|
| 0.001 | 1K | 13s | 0.828 | 2.9 MB | 1 / 996 / 3613 |
| 0.005 | 5K | 82s | 0.928 | 14.6 MB | 1 / 178 / 1361 |
| 0.100 | 100K | 43s* | 0.974 | 293 MB | 10 / 37 / 226 |

### Duplication dramatically improves recall

Adding `--duplicate-max 4 --duplicate-ratio 5.0` at k=5K:

| fanout | r@10 (no dup) | r@10 (dup 4×) | disk cost |
|--------|--------------|---------------|-----------|
| 0.1% (n=5) | 0.713 | 0.877 | 2.9 → 11.4 GB |
| 1% (n=50) | 0.928 | 0.984 | 2.9 → 11.4 GB |
| 5% (n=250) | 0.986 | 0.999 | 2.9 → 11.4 GB |

Duplication costs ~4× disk (each vector stored in up to 5 clusters) but boosts r@10 by 5-16 percentage points at low fanout. The duplicate step itself takes ~13s at k=5K.

### Hierarchical is required at large k

Hierarchical bisection trains in O(n × d × log k) and finishes in ~43s at k=100K where flat takes 30+ minutes.

The tradeoff: hierarchical recall is slightly lower than flat at the same k because each binary split is locally optimal, not globally.

### Cluster balance

Flat Lloyd's produces highly imbalanced clusters (min=1, max=3613 at k=1K). Hierarchical bisection enforces near-exact balance (min=10, max=226 at k=100K), which bounds worst-case query latency when scanning posting lists.

