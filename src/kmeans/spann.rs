//! SPANN-style cluster building.
//!
//! Implements the two ideas from
//! [SPANN (Chen et al., NeurIPS 2021)](https://arxiv.org/abs/2111.08566)
//! that target k-means imbalance for IVF:
//!
//! 1. **Hierarchical balanced clustering** — recursive bisection where each
//!    `k=2` step is forced to produce two halves whose sizes are proportional
//!    to the target cluster counts of their subtrees. This is a tight version
//!    of SPANN's algorithm: we use exact size splits via signed-distance
//!    sorting rather than the paper's tolerance-based approach. The result is
//!    a tree where every leaf has approximately `n / k` points.
//!
//! 2. **Boundary-point duplication** — after primary assignment, every point
//!    that's nearly equidistant to multiple centroids gets duplicated into
//!    the runner-up clusters. Controlled by a fan-out cap and a distance
//!    ratio.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rayon::prelude::*;

use super::distance::{find_k_nearest_neighbors, squared_l2, X_BATCH_SIZE, Y_BATCH_SIZE};

/// Reified bisection tree. Each internal node carries the routing centroids
/// for its two children (the means produced by the parent's `balanced_bisect`).
/// At query time we descend by picking the closer child; for top-N we use a
/// best-first PQ traversal — same idea as SPTAG-BKT.
pub enum BisectTree {
    Internal {
        left_centroid: Vec<f32>,
        left: Box<BisectTree>,
        right_centroid: Vec<f32>,
        right: Box<BisectTree>,
    },
    Leaf {
        cluster_id: u32,
    },
}

/// Hierarchical balanced bisection.
///
/// Returns `(centroids, primary_assignments)` where `centroids` is row-major
/// `k × d` and `primary_assignments[i]` is the leaf cluster index for point i.
///
/// `n_iters_per_bisect` controls how many balance-constrained Lloyd iterations
/// each k=2 step runs. 5–10 is a reasonable range; the splits stabilize fast.
pub fn hierarchical_balanced_kmeans(
    data: &[f32],
    n: usize,
    dims: usize,
    k: usize,
    n_iters_per_bisect: usize,
    seed: u64,
) -> (Vec<f32>, Vec<u32>, BisectTree) {
    assert!(k > 0);
    assert!(n >= k, "n ({n}) must be ≥ k ({k}) for hierarchical clustering");
    let mut centroids = Vec::with_capacity(k * dims);
    let mut assignments = vec![0u32; n];
    let indices: Vec<u32> = (0..n as u32).collect();
    let max_depth = (k as f32).log2().ceil() as usize + 1;
    let mut level_ms = vec![0.0f64; max_depth];
    let tree = bisect_recurse(
        data,
        dims,
        &indices,
        k,
        0,
        n_iters_per_bisect,
        &mut centroids,
        &mut assignments,
        seed,
        0,
        &mut level_ms,
    );
    debug_assert_eq!(centroids.len(), k * dims);
    let total: f64 = level_ms.iter().sum();
    let mut summary = String::from("    hierarchical bisect by level: ");
    for (d, ms) in level_ms.iter().enumerate() {
        if *ms > 0.0 {
            summary.push_str(&format!("L{}={:.1}s ", d, ms / 1000.0));
        }
    }
    summary.push_str(&format!("(sum={:.1}s)", total / 1000.0));
    eprintln!("{}", summary);
    (centroids, assignments, tree)
}

#[allow(clippy::too_many_arguments)]
fn bisect_recurse(
    data: &[f32],
    dims: usize,
    indices: &[u32],
    k: usize,
    cluster_offset: u32,
    n_iters_per_bisect: usize,
    centroids: &mut Vec<f32>,
    assignments: &mut [u32],
    seed: u64,
    depth: usize,
    level_ms: &mut [f64],
) -> BisectTree {
    if k == 1 {
        // Leaf: centroid is the mean of assigned points.
        let mut centroid = vec![0.0f32; dims];
        for &i in indices {
            let row = &data[i as usize * dims..(i as usize + 1) * dims];
            for d in 0..dims {
                centroid[d] += row[d];
            }
        }
        if !indices.is_empty() {
            let inv = 1.0 / indices.len() as f32;
            for v in centroid.iter_mut() {
                *v *= inv;
            }
        }
        centroids.extend_from_slice(&centroid);
        for &i in indices {
            assignments[i as usize] = cluster_offset;
        }
        return BisectTree::Leaf { cluster_id: cluster_offset };
    }

    let k_left = k / 2;
    let k_right = k - k_left;
    let n_subset = indices.len();
    let target_left = (n_subset * k_left) / k;
    // Ensure each side has at least k_side points so leaves are non-empty.
    let target_left = target_left.max(k_left).min(n_subset.saturating_sub(k_right));
    let target_right = n_subset - target_left;
    debug_assert!(target_right >= k_right);

    let bisect_t = std::time::Instant::now();
    let (left_indices, right_indices, centroid_a, centroid_b) = balanced_bisect(
        data,
        dims,
        indices,
        target_left,
        target_right,
        n_iters_per_bisect,
        seed,
    );
    if depth < level_ms.len() {
        level_ms[depth] += bisect_t.elapsed().as_secs_f64() * 1000.0;
    }

    let left = bisect_recurse(
        data,
        dims,
        &left_indices,
        k_left,
        cluster_offset,
        n_iters_per_bisect,
        centroids,
        assignments,
        seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(1),
        depth + 1,
        level_ms,
    );
    let right = bisect_recurse(
        data,
        dims,
        &right_indices,
        k_right,
        cluster_offset + k_left as u32,
        n_iters_per_bisect,
        centroids,
        assignments,
        seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(2),
        depth + 1,
        level_ms,
    );

    BisectTree::Internal {
        left_centroid: centroid_a,
        left: Box::new(left),
        right_centroid: centroid_b,
        right: Box::new(right),
    }
}

/// Split a subset of points into a "left" group of exactly `target_left` points
/// and a "right" group of `target_right` points using balance-constrained
/// Lloyd's iteration on `k=2`.
///
/// Each iteration: compute signed distance `dist(p, A) − dist(p, B)` for every
/// point, sort ascending (negative ⇒ closer to A), and take the first
/// `target_left` indices as A, the rest as B. Recompute centroids as means.
/// Repeat. The split is *exact*; the iteration only refines the centroid
/// positions within the constrained partitioning.
fn balanced_bisect(
    data: &[f32],
    dims: usize,
    indices: &[u32],
    target_left: usize,
    target_right: usize,
    n_iters: usize,
    seed: u64,
) -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>) {
    debug_assert_eq!(target_left + target_right, indices.len());

    // Initialise centroids with two reasonably far-apart points: pick one at
    // random, then pick the farthest point from it. (Cheap k-means++ for k=2.)
    let mut rng = TinyRng::new(seed);
    let n_subset = indices.len();
    let i0 = indices[(rng.next() as usize) % n_subset] as usize;
    let row0 = &data[i0 * dims..(i0 + 1) * dims];
    let i1 = indices
        .par_iter()
        .map(|&i| {
            let row = &data[i as usize * dims..(i as usize + 1) * dims];
            (squared_l2(row0, row), i)
        })
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, i)| i)
        .unwrap_or(indices[0]) as usize;

    let mut centroid_a: Vec<f32> = data[i0 * dims..(i0 + 1) * dims].to_vec();
    let mut centroid_b: Vec<f32> = data[i1 * dims..(i1 + 1) * dims].to_vec();
    let mut sorted: Vec<u32> = indices.to_vec();

    for _ in 0..n_iters {
        // Score each point by signed distance.
        let mut scored: Vec<(f32, u32)> = indices
            .par_iter()
            .map(|&i| {
                let row = &data[i as usize * dims..(i as usize + 1) * dims];
                let da = squared_l2(row, &centroid_a);
                let db = squared_l2(row, &centroid_b);
                (da - db, i)
            })
            .collect();
        scored.par_sort_unstable_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted = scored.into_iter().map(|(_, i)| i).collect();

        // Recompute centroids from constrained partition.
        let (left, right) = sorted.split_at(target_left);
        recompute_centroid(data, dims, left, &mut centroid_a);
        recompute_centroid(data, dims, right, &mut centroid_b);
    }

    let left = sorted[..target_left].to_vec();
    let right = sorted[target_left..].to_vec();
    (left, right, centroid_a, centroid_b)
}

fn recompute_centroid(data: &[f32], dims: usize, indices: &[u32], centroid: &mut [f32]) {
    centroid.iter_mut().for_each(|v| *v = 0.0);
    for &i in indices {
        let row = &data[i as usize * dims..(i as usize + 1) * dims];
        for d in 0..dims {
            centroid[d] += row[d];
        }
    }
    if !indices.is_empty() {
        let inv = 1.0 / indices.len() as f32;
        for v in centroid.iter_mut() {
            *v *= inv;
        }
    }
}

/// Best-first PQ traversal returning approximate top-`n` nearest leaves.
/// Visits at most `budget` leaves; pruning stops early when the frontier's
/// best lower bound exceeds the current Nth-best leaf distance. Distances
/// are squared L2.
pub fn tree_search_top_n(
    root: &BisectTree,
    query: &[f32],
    n: usize,
    budget: usize,
) -> Vec<(f32, u32)> {
    if n == 0 {
        return Vec::new();
    }
    let mut frontier: BinaryHeap<FrontierEntry> = BinaryHeap::new();
    let mut best: BinaryHeap<LeafEntry> = BinaryHeap::with_capacity(n + 1);

    frontier.push(FrontierEntry {
        neg_dist: 0.0,
        node: root,
    });

    let mut leaves_visited = 0usize;
    while let Some(entry) = frontier.pop() {
        let dist = -entry.neg_dist;
        if best.len() >= n {
            let worst = best.peek().map(|l| l.dist).unwrap_or(f32::INFINITY);
            if dist >= worst {
                break;
            }
        }
        if leaves_visited >= budget {
            break;
        }
        match entry.node {
            BisectTree::Leaf { cluster_id } => {
                leaves_visited += 1;
                if best.len() < n {
                    best.push(LeafEntry {
                        dist,
                        cluster: *cluster_id,
                    });
                } else if dist < best.peek().unwrap().dist {
                    best.pop();
                    best.push(LeafEntry {
                        dist,
                        cluster: *cluster_id,
                    });
                }
            }
            BisectTree::Internal {
                left_centroid,
                left,
                right_centroid,
                right,
            } => {
                let dl = squared_l2(query, left_centroid);
                let dr = squared_l2(query, right_centroid);
                frontier.push(FrontierEntry {
                    neg_dist: -dl,
                    node: left.as_ref(),
                });
                frontier.push(FrontierEntry {
                    neg_dist: -dr,
                    node: right.as_ref(),
                });
            }
        }
    }

    let mut result: Vec<(f32, u32)> = best
        .into_iter()
        .map(|l| (l.dist, l.cluster))
        .collect();
    result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    result
}

// BinaryHeap is a max-heap; we negate distance so popping gives min-distance.
struct FrontierEntry<'a> {
    neg_dist: f32,
    node: &'a BisectTree,
}
impl<'a> PartialEq for FrontierEntry<'a> {
    fn eq(&self, o: &Self) -> bool {
        self.neg_dist == o.neg_dist
    }
}
impl<'a> Eq for FrontierEntry<'a> {}
impl<'a> Ord for FrontierEntry<'a> {
    fn cmp(&self, o: &Self) -> Ordering {
        self.neg_dist.partial_cmp(&o.neg_dist).unwrap_or(Ordering::Equal)
    }
}
impl<'a> PartialOrd for FrontierEntry<'a> {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

// Max-heap by dist so peek is the worst (largest) — easy to evict.
struct LeafEntry {
    dist: f32,
    cluster: u32,
}
impl PartialEq for LeafEntry {
    fn eq(&self, o: &Self) -> bool {
        self.dist == o.dist
    }
}
impl Eq for LeafEntry {}
impl Ord for LeafEntry {
    fn cmp(&self, o: &Self) -> Ordering {
        self.dist.partial_cmp(&o.dist).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for LeafEntry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// Tree-routed duplication: for each doc, use best-first traversal of the
/// bisect tree to approximate top-`max_copies` nearest centroids in O(log k)
/// per query instead of O(k). Falls back to brute-force semantics on the
/// returned candidate set (same ratio + max-replicas filter as
/// `duplicate_assignments`).
#[allow(clippy::too_many_arguments)]
pub fn duplicate_assignments_via_tree(
    data: &[f32],
    n: usize,
    dims: usize,
    centroids: &[f32],
    n_clusters: usize,
    primary: &[u32],
    max_copies: usize,
    ratio_threshold: f32,
    tree: &BisectTree,
    budget_factor: usize,
) -> Vec<Vec<u32>> {
    if max_copies <= 1 || ratio_threshold <= 1.0 {
        return primary.iter().map(|&c| vec![c]).collect();
    }
    let max_copies = max_copies.min(n_clusters);
    let request = (max_copies + 1).min(n_clusters);
    let budget = (max_copies * budget_factor.max(1)).max(request);
    let ratio_sq = ratio_threshold * ratio_threshold;

    (0..n)
        .into_par_iter()
        .map(|i| {
            let query = &data[i * dims..(i + 1) * dims];
            let neighbors = tree_search_top_n(tree, query, request, budget);
            let primary_c = primary[i];
            let mut clusters = Vec::with_capacity(max_copies);
            clusters.push(primary_c);

            // Primary distance for the ratio cap. If primary isn't in the
            // tree's top-N (can happen when balance forced the doc into a
            // far leaf), compute it directly.
            let primary_dist = neighbors
                .iter()
                .find(|(_, c)| *c == primary_c)
                .map(|(d, _)| *d)
                .unwrap_or_else(|| {
                    let c = &centroids
                        [primary_c as usize * dims..(primary_c as usize + 1) * dims];
                    squared_l2(query, c)
                });
            let cap = primary_dist * ratio_sq;

            for (d, c) in &neighbors {
                if *c == primary_c {
                    continue;
                }
                if *d <= cap {
                    clusters.push(*c);
                    if clusters.len() >= max_copies {
                        break;
                    }
                } else {
                    break;
                }
            }
            clusters
        })
        .collect()
}

/// SPANN-style boundary-point duplication.
///
/// For each point, find the closest `max_copies` centroids. Keep the primary
/// (already determined elsewhere) plus any other centroid whose squared
/// distance is within `ratio² × primary_distance²` of the primary. The squared
/// ratio is what `ratio_threshold` represents (i.e. apply it directly to
/// squared L2 distances).
///
/// Returns `Vec<Vec<u32>>` where the first cluster of each entry is the
/// primary assignment. If `max_copies <= 1`, returns a wrapped view of the
/// primary assignments unchanged.
pub fn duplicate_assignments(
    data: &[f32],
    n: usize,
    dims: usize,
    centroids: &[f32],
    n_clusters: usize,
    primary: &[u32],
    max_copies: usize,
    ratio_threshold: f32,
) -> Vec<Vec<u32>> {
    if max_copies <= 1 || ratio_threshold <= 1.0 {
        return primary.iter().map(|&c| vec![c]).collect();
    }
    let max_copies = max_copies.min(n_clusters);

    // Compute the top-`max_copies` nearest centroids for every point.
    let mut q_norms = vec![0.0f32; n];
    q_norms.par_iter_mut().enumerate().for_each(|(i, slot)| {
        let row = &data[i * dims..(i + 1) * dims];
        *slot = row.iter().map(|v| v * v).sum();
    });
    let mut c_norms = vec![0.0f32; n_clusters];
    c_norms.par_iter_mut().enumerate().for_each(|(i, slot)| {
        let row = &centroids[i * dims..(i + 1) * dims];
        *slot = row.iter().map(|v| v * v).sum();
    });

    let mut top = vec![0u32; n * max_copies];
    let mut top_dists = vec![0.0f32; n * max_copies];
    let mut tmp = vec![0.0f32; X_BATCH_SIZE * Y_BATCH_SIZE];
    find_k_nearest_neighbors(
        data,
        centroids,
        n,
        n_clusters,
        dims,
        &q_norms,
        &c_norms,
        max_copies,
        &mut top,
        &mut top_dists,
        &mut tmp,
    );

    let ratio_sq = ratio_threshold * ratio_threshold;
    let mut result: Vec<Vec<u32>> = Vec::with_capacity(n);
    for i in 0..n {
        // Anchor on the primary assignment so the first entry is always the
        // user-facing "primary" cluster, even if find_k_nearest_neighbors
        // returns a slightly different ordering due to ties.
        let primary_c = primary[i];
        let mut clusters = vec![primary_c];
        let primary_dist = top_dists[i * max_copies];
        let cap = primary_dist * ratio_sq;
        for j in 0..max_copies {
            let c = top[i * max_copies + j];
            if c == primary_c {
                continue;
            }
            if top_dists[i * max_copies + j] <= cap {
                clusters.push(c);
            } else {
                break;
            }
        }
        result.push(clusters);
    }
    result
}

/// Per-cluster cumulative size accounting for duplicated assignments.
pub fn duplicated_cluster_sizes(assignments: &[Vec<u32>], n_clusters: usize) -> Vec<u32> {
    let mut sizes = vec![0u32; n_clusters];
    for clusters in assignments {
        for &c in clusters {
            sizes[c as usize] += 1;
        }
    }
    sizes
}

/// Cluster balance summary derived from a flat sizes vector.
pub fn balance_from_sizes(sizes: &[u32]) -> (f32, f32, u32, u32, f32) {
    let n_clusters = sizes.len();
    let total: u64 = sizes.iter().map(|&s| s as u64).sum();
    let mean = total as f32 / n_clusters.max(1) as f32;
    let mut log_sum = 0.0f32;
    let mut nz = 0usize;
    for &s in sizes {
        if s > 0 {
            log_sum += (s as f32).ln();
            nz += 1;
        }
    }
    let geometric_mean = if nz > 0 {
        (log_sum / nz as f32).exp()
    } else {
        0.0
    };
    let sq: u128 = sizes.iter().map(|&s| (s as u128) * (s as u128)).sum();
    let variance = (sq as f32 / n_clusters.max(1) as f32) - mean * mean;
    let stdev = variance.max(0.0).sqrt();
    let cv = if mean > 0.0 { stdev / mean } else { 0.0 };
    let min = *sizes.iter().min().unwrap_or(&0);
    let max = *sizes.iter().max().unwrap_or(&0);
    let _ = geometric_mean;
    (mean, stdev, min, max, cv)
}

/// Tiny xorshift64 RNG; we don't pull in StdRng here to keep recursion light.
struct TinyRng(u64);
impl TinyRng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn hierarchical_produces_balanced_clusters() {
        // Synthetic: 1000 points, 8 clusters of 125 each — exact division.
        let n = 1000usize;
        let dims = 16usize;
        let mut rng = StdRng::seed_from_u64(0);
        let mut data = Vec::with_capacity(n * dims);
        for _ in 0..n {
            for _ in 0..dims {
                data.push(rng.gen_range(-1.0..1.0_f32));
            }
        }
        let (centroids, assignments) =
            hierarchical_balanced_kmeans(&data, n, dims, 8, 5, 42);
        assert_eq!(centroids.len(), 8 * dims);
        let mut sizes = vec![0usize; 8];
        for &a in &assignments {
            sizes[a as usize] += 1;
        }
        // Exact partitioning: each cluster has exactly 125 points.
        for s in sizes {
            assert_eq!(s, 125);
        }
    }

    #[test]
    fn duplication_respects_ratio() {
        // 1 point, 4 centroids, two of which are within ratio of nearest.
        let dims = 4usize;
        let data = vec![0.0f32, 0.0, 0.0, 0.0];
        // centroid 0 at distance² = 1 (nearest).
        // centroid 1 at distance² = 1.5 (within 1.5× ratio).
        // centroid 2 at distance² = 4 (way outside).
        // centroid 3 at distance² = 1.21 (within 1.5× ratio).
        let centroids = vec![
            1.0, 0.0, 0.0, 0.0, // dist² = 1
            f32::sqrt(1.5), 0.0, 0.0, 0.0, // dist² = 1.5
            2.0, 0.0, 0.0, 0.0, // dist² = 4
            1.1, 0.0, 0.0, 0.0, // dist² = 1.21
        ];
        let primary = vec![0u32];
        // ratio_threshold so that ratio² = 1.5 → ratio = sqrt(1.5) ≈ 1.225
        let r = (1.5f32).sqrt();
        let result = duplicate_assignments(&data, 1, dims, &centroids, 4, &primary, 4, r);
        let clusters = &result[0];
        assert_eq!(clusters[0], 0, "primary always first");
        // Expect clusters 1 and 3 included, 2 excluded.
        assert!(clusters.contains(&1), "centroid 1 should duplicate");
        assert!(clusters.contains(&3), "centroid 3 should duplicate");
        assert!(!clusters.contains(&2), "centroid 2 too far");
    }
}
