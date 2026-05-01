use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq)]
struct ScoredDoc {
    score: f32,
    id: u64,
}

impl Eq for ScoredDoc {}

impl PartialOrd for ScoredDoc {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredDoc {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RetrievalMetrics {
    pub recall: f32,
    pub ndcg: f32,
}

pub fn top_k_by_score<I>(scores: I, k: usize) -> Vec<(u64, f32)>
where
    I: IntoIterator<Item = (u64, f32)>,
{
    let mut heap: BinaryHeap<Reverse<ScoredDoc>> = BinaryHeap::with_capacity(k + 1);
    for (id, score) in scores {
        let candidate = Reverse(ScoredDoc { score, id });
        if heap.len() < k {
            heap.push(candidate);
        } else if let Some(worst) = heap.peek() {
            if candidate.0 > worst.0 {
                heap.pop();
                heap.push(candidate);
            }
        }
    }
    let mut top: Vec<_> = heap
        .into_iter()
        .map(|Reverse(scored)| (scored.id, scored.score))
        .collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    top
}

pub fn metrics_for(
    ground_truth_ids: &[u64],
    got_ranked: &[(u64, f32)],
    k: usize,
) -> RetrievalMetrics {
    let gt: Vec<u64> = ground_truth_ids.iter().copied().take(k).collect();
    let gt_set: HashSet<u64> = gt.iter().copied().collect();
    let hits = got_ranked
        .iter()
        .take(k)
        .filter(|(id, _)| gt_set.contains(id))
        .count();
    let recall = hits as f32 / k as f32;
    let dcg = got_ranked
        .iter()
        .take(k)
        .enumerate()
        .filter_map(|(rank, (id, _))| {
            gt_set
                .contains(id)
                .then_some(1.0f32 / ((rank + 2) as f32).log2())
        })
        .sum::<f32>();
    let ideal = (0..gt.len())
        .map(|rank| 1.0f32 / ((rank + 2) as f32).log2())
        .sum::<f32>();
    let ndcg = if ideal > 0.0 { dcg / ideal } else { 0.0 };
    RetrievalMetrics { recall, ndcg }
}

pub fn average_metrics(metrics: &[RetrievalMetrics]) -> RetrievalMetrics {
    let denom = metrics.len().max(1) as f32;
    RetrievalMetrics {
        recall: metrics.iter().map(|m| m.recall).sum::<f32>() / denom,
        ndcg: metrics.iter().map(|m| m.ndcg).sum::<f32>() / denom,
    }
}
