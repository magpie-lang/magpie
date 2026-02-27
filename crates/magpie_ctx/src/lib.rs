//! magpie_ctx

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    pub chunk_id: String,
    pub kind: String,
    pub subject_id: String,
    pub body: String,
    pub token_cost: u32,
    pub score: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetPolicy {
    Balanced,
    DiagnosticsFirst,
    SlicesFirst,
    Minimal,
}

impl Default for BudgetPolicy {
    fn default() -> Self {
        Self::Balanced
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextPack {
    pub chunks: Vec<Chunk>,
    pub budget: u32,
    pub used_budget: u32,
    pub remaining_budget: u32,
    pub policy: BudgetPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bucket {
    Structural,
    Problem,
    Capsule,
    Retrieved,
    Other,
}

#[derive(Debug, Clone)]
struct Variant {
    body: String,
    token_cost: u32,
}

#[derive(Debug, Clone)]
struct Candidate {
    chunk: Chunk,
    bucket: Bucket,
    rank_score: f64,
    variants: [Variant; 4], // v3 -> v0
}

pub fn build_context_pack(chunks: Vec<Chunk>, budget: u32, policy: BudgetPolicy) -> ContextPack {
    if budget == 0 || chunks.is_empty() {
        return ContextPack {
            chunks: Vec::new(),
            budget,
            used_budget: 0,
            remaining_budget: budget,
            policy,
        };
    }

    let candidates: Vec<Candidate> = chunks
        .into_iter()
        .map(|chunk| {
            let rank_score = compute_rank_score(&chunk);
            let variants = build_variants(&chunk);
            let bucket = classify_bucket(&chunk.kind);
            Candidate {
                chunk,
                bucket,
                rank_score,
                variants,
            }
        })
        .collect();

    let mut structural_pool = Vec::new();
    let mut problem_pool = Vec::new();
    let mut capsule_pool = Vec::new();
    let mut retrieved_pool = Vec::new();
    let mut other_pool = Vec::new();

    for (idx, candidate) in candidates.iter().enumerate() {
        match candidate.bucket {
            Bucket::Structural => structural_pool.push(idx),
            Bucket::Problem => problem_pool.push(idx),
            Bucket::Capsule => capsule_pool.push(idx),
            Bucket::Retrieved => retrieved_pool.push(idx),
            Bucket::Other => other_pool.push(idx),
        }
    }

    let mut problem_with_capsules = Vec::new();
    problem_with_capsules.extend(problem_pool.iter().copied());
    problem_with_capsules.extend(capsule_pool.iter().copied());
    problem_with_capsules.extend(other_pool.iter().copied());

    let mut problem_without_capsules = Vec::new();
    problem_without_capsules.extend(problem_pool.iter().copied());
    problem_without_capsules.extend(other_pool.iter().copied());

    let mut selected = vec![false; candidates.len()];
    let mut structural_out = Vec::new();
    let mut problem_out = Vec::new();
    let mut retrieved_out = Vec::new();

    let mut used_total = 0u32;

    match policy {
        BudgetPolicy::Balanced => {
            let [b_struct, b_problem, b_retrieved] = split_budget(budget, [25, 45, 30]);
            used_total += select_from_pool(
                &candidates,
                &structural_pool,
                b_struct,
                &mut selected,
                &mut structural_out,
            );
            used_total += select_from_pool(
                &candidates,
                &problem_with_capsules,
                b_problem,
                &mut selected,
                &mut problem_out,
            );
            used_total += select_from_pool(
                &candidates,
                &retrieved_pool,
                b_retrieved,
                &mut selected,
                &mut retrieved_out,
            );
        }
        BudgetPolicy::DiagnosticsFirst => {
            let [b_struct, b_problem, b_retrieved] = split_budget(budget, [30, 60, 10]);
            used_total += select_from_pool(
                &candidates,
                &structural_pool,
                b_struct,
                &mut selected,
                &mut structural_out,
            );
            used_total += select_from_pool(
                &candidates,
                &problem_with_capsules,
                b_problem,
                &mut selected,
                &mut problem_out,
            );
            used_total += select_from_pool(
                &candidates,
                &retrieved_pool,
                b_retrieved,
                &mut selected,
                &mut retrieved_out,
            );
        }
        BudgetPolicy::SlicesFirst => {
            let [b_struct, b_slices, b_retrieved] = split_budget(budget, [35, 55, 10]);
            used_total += select_from_pool(
                &candidates,
                &structural_pool,
                b_struct,
                &mut selected,
                &mut structural_out,
            );
            let used_slices = select_from_pool(
                &candidates,
                &capsule_pool,
                b_slices,
                &mut selected,
                &mut problem_out,
            );
            used_total += used_slices;
            let slices_left = b_slices.saturating_sub(used_slices);
            if slices_left > 0 {
                used_total += select_from_pool(
                    &candidates,
                    &problem_without_capsules,
                    slices_left,
                    &mut selected,
                    &mut problem_out,
                );
            }
            used_total += select_from_pool(
                &candidates,
                &retrieved_pool,
                b_retrieved,
                &mut selected,
                &mut retrieved_out,
            );
        }
        BudgetPolicy::Minimal => {
            let [b_struct, b_problem, _b_retrieved] = split_budget(budget, [60, 40, 0]);
            used_total += select_from_pool(
                &candidates,
                &structural_pool,
                b_struct,
                &mut selected,
                &mut structural_out,
            );
            used_total += select_from_pool(
                &candidates,
                &problem_without_capsules,
                b_problem,
                &mut selected,
                &mut problem_out,
            );
        }
    }

    // Appendix G default spillover order: structural -> problem -> retrieved.
    let mut remaining = budget.saturating_sub(used_total);
    if remaining > 0 {
        let used = select_from_pool(
            &candidates,
            &structural_pool,
            remaining,
            &mut selected,
            &mut structural_out,
        );
        used_total += used;
        remaining = budget.saturating_sub(used_total);
    }
    if remaining > 0 {
        let spill_problem_pool = match policy {
            BudgetPolicy::Minimal => &problem_without_capsules,
            _ => &problem_with_capsules,
        };
        let used = select_from_pool(
            &candidates,
            spill_problem_pool,
            remaining,
            &mut selected,
            &mut problem_out,
        );
        used_total += used;
        remaining = budget.saturating_sub(used_total);
    }
    if remaining > 0 {
        select_from_pool(
            &candidates,
            &retrieved_pool,
            remaining,
            &mut selected,
            &mut retrieved_out,
        );
    }

    let mut out =
        Vec::with_capacity(structural_out.len() + problem_out.len() + retrieved_out.len());
    out.extend(structural_out);
    out.extend(problem_out);
    out.extend(retrieved_out);

    let used_budget = out.iter().map(|c| c.token_cost).sum::<u32>().min(budget);
    ContextPack {
        chunks: out,
        budget,
        used_budget,
        remaining_budget: budget.saturating_sub(used_budget),
        policy,
    }
}

fn select_from_pool(
    candidates: &[Candidate],
    pool: &[usize],
    budget: u32,
    selected: &mut [bool],
    out: &mut Vec<Chunk>,
) -> u32 {
    if budget == 0 || pool.is_empty() {
        return 0;
    }

    let mut sorted = pool.to_vec();
    sorted.sort_by(|a, b| compare_candidates(&candidates[*a], &candidates[*b]));

    let mut used = 0u32;
    for idx in sorted {
        if selected[idx] {
            continue;
        }
        let remaining = budget.saturating_sub(used);
        if remaining == 0 {
            break;
        }

        let candidate = &candidates[idx];
        let Some(variant) = pick_variant(candidate, remaining) else {
            continue;
        };
        used = used.saturating_add(variant.token_cost);
        selected[idx] = true;

        let mut selected_chunk = candidate.chunk.clone();
        selected_chunk.body = variant.body.clone();
        selected_chunk.token_cost = variant.token_cost;
        selected_chunk.score = candidate.rank_score;
        out.push(selected_chunk);
    }

    used
}

fn pick_variant<'a>(candidate: &'a Candidate, remaining_budget: u32) -> Option<&'a Variant> {
    // v3 -> v2 -> v1 -> v0: pick highest fidelity variant that fits.
    candidate
        .variants
        .iter()
        .find(|variant| variant.token_cost <= remaining_budget)
}

fn compare_candidates(a: &Candidate, b: &Candidate) -> Ordering {
    b.rank_score
        .total_cmp(&a.rank_score)
        .then_with(|| {
            normalize_token_cost(a.chunk.token_cost).cmp(&normalize_token_cost(b.chunk.token_cost))
        })
        .then_with(|| a.chunk.chunk_id.cmp(&b.chunk.chunk_id))
}

fn compute_rank_score(chunk: &Chunk) -> f64 {
    let base = base_priority(&chunk.kind) as f64;
    let size_penalty = (normalize_token_cost(chunk.token_cost) / 200) as f64;

    if is_retrieved_kind(&chunk.kind) {
        // For retrieved chunks, `score` is treated as raw MMS score.
        let retrieval_score = sanitize_score(chunk.score).floor().clamp(0.0, 25.0);
        base + retrieval_score - size_penalty
    } else {
        // For non-retrieved chunks, `score` carries relevance + proximity adjustments.
        base + sanitize_score(chunk.score) - size_penalty
    }
}

fn sanitize_score(score: f64) -> f64 {
    if score.is_finite() {
        score
    } else {
        0.0
    }
}

fn base_priority(kind: &str) -> i32 {
    match kind {
        "module_header" => 100,
        "mpd_public_api" => 90,
        "symgraph_summary" => 85,
        "diagnostics" => 80,
        "ownership_trace" => 78,
        "cfg_summary" => 72,
        "symbol_capsule" => 70,
        "snippet" => 60,
        "rag_item" => 55,
        "deps_summary" => 50,
        _ => 50,
    }
}

fn classify_bucket(kind: &str) -> Bucket {
    match kind {
        "module_header" | "mpd_public_api" | "symgraph_summary" | "deps_summary" => {
            Bucket::Structural
        }
        "diagnostics" | "ownership_trace" | "cfg_summary" => Bucket::Problem,
        "symbol_capsule" | "snippet" => Bucket::Capsule,
        "rag_item" => Bucket::Retrieved,
        _ => Bucket::Other,
    }
}

fn is_retrieved_kind(kind: &str) -> bool {
    kind == "rag_item"
}

fn split_budget(total: u32, percents: [u32; 3]) -> [u32; 3] {
    let mut buckets = [0u32; 3];
    let mut allocated = 0u32;
    for i in 0..3 {
        buckets[i] = total.saturating_mul(percents[i]) / 100;
        allocated = allocated.saturating_add(buckets[i]);
    }

    // Deterministic remainder assignment.
    let mut rem = total.saturating_sub(allocated);
    let mut i = 0usize;
    while rem > 0 {
        buckets[i] = buckets[i].saturating_add(1);
        rem -= 1;
        i = (i + 1) % 3;
    }

    buckets
}

fn build_variants(chunk: &Chunk) -> [Variant; 4] {
    let v3_body = normalize_body(&chunk.body);
    let v2_body = compress_v2(&v3_body);
    let v1_body = compress_v1(&v2_body);
    let v0_body = compress_v0(chunk, &v3_body);

    let v3_cost = normalize_token_cost(chunk.token_cost);
    let v2_cost = estimate_token_cost(&v2_body).min(v3_cost).max(1);
    let v1_cost = estimate_token_cost(&v1_body).min(v2_cost).max(1);
    let v0_cost = estimate_token_cost(&v0_body).min(v1_cost).max(1);

    [
        Variant {
            body: v3_body,
            token_cost: v3_cost,
        },
        Variant {
            body: v2_body,
            token_cost: v2_cost,
        },
        Variant {
            body: v1_body,
            token_cost: v1_cost,
        },
        Variant {
            body: v0_body,
            token_cost: v0_cost,
        },
    ]
}

fn normalize_body(body: &str) -> String {
    body.lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn compress_v2(body: &str) -> String {
    let lines = collect_non_empty_lines(body);
    if lines.is_empty() {
        return String::new();
    }

    let mut out = Vec::new();
    push_unique(&mut out, &lines[0]);
    for line in &lines {
        if is_signature_or_key_line(line) {
            push_unique(&mut out, line);
            if out.len() >= 32 {
                break;
            }
        }
    }

    if out.len() <= 1 {
        for line in lines.iter().take(24) {
            push_unique(&mut out, line);
        }
    } else if out.len() < 8 {
        for line in lines.iter().take(24) {
            push_unique(&mut out, line);
            if out.len() >= 24 {
                break;
            }
        }
    }

    out.join("\n")
}

fn compress_v1(body: &str) -> String {
    let lines = collect_non_empty_lines(body);
    if lines.is_empty() {
        return String::new();
    }

    lines
        .into_iter()
        .take(20)
        .map(|line| format!("- {line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn compress_v0(chunk: &Chunk, body: &str) -> String {
    let name = extract_name(body).unwrap_or_else(|| chunk.subject_id.clone());
    format!("SID={} name={} type={}", chunk.subject_id, name, chunk.kind)
}

fn collect_non_empty_lines(body: &str) -> Vec<String> {
    body.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn is_signature_or_key_line(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    lower.starts_with("fn ")
        || lower.starts_with("type ")
        || lower.starts_with("struct ")
        || lower.starts_with("enum ")
        || lower.starts_with("trait ")
        || lower.starts_with("module ")
        || lower.starts_with("imports ")
        || lower.starts_with("exports ")
        || lower.starts_with("digest ")
        || lower.contains(" error")
        || lower.contains(" warning")
        || lower.contains("->")
        || lower.contains("::")
        || line.contains('@')
        || line.contains('%')
}

fn push_unique(out: &mut Vec<String>, line: &str) {
    if out.iter().any(|existing| existing == line) {
        return;
    }
    out.push(line.to_string());
}

fn extract_name(body: &str) -> Option<String> {
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        for prefix in ["fn ", "type ", "struct ", "enum ", "trait ", "module "] {
            if let Some(rest) = line.strip_prefix(prefix) {
                if let Some(name) = take_identifier(rest) {
                    return Some(name);
                }
            }
        }

        if let Some(name) = take_identifier(line) {
            return Some(name);
        }
    }
    None
}

fn take_identifier(input: &str) -> Option<String> {
    let mut out = String::new();
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | ':' | '@' | '%') {
            out.push(ch);
        } else {
            break;
        }
    }

    while matches!(out.chars().last(), Some('.' | ':' | '@' | '%')) {
        out.pop();
    }

    (!out.is_empty()).then_some(out)
}

fn normalize_token_cost(cost: u32) -> u32 {
    cost.max(1)
}

fn estimate_token_cost(text: &str) -> u32 {
    let chars = text.chars().count() as u32;
    chars.saturating_add(3).saturating_div(4).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(id: &str, kind: &str, body: &str, token_cost: u32, score: f64) -> Chunk {
        Chunk {
            chunk_id: id.to_string(),
            kind: kind.to_string(),
            subject_id: format!("S:{id}"),
            body: body.to_string(),
            token_cost,
            score,
        }
    }

    #[test]
    fn build_context_pack_orders_buckets_structural_problem_retrieved() {
        let chunks = vec![
            chunk("retrieved-1", "rag_item", "retrieved detail", 8, 9.0),
            chunk("problem-1", "diagnostics", "diagnostic detail", 8, 5.0),
            chunk("struct-1", "module_header", "module demo", 8, 1.0),
        ];

        let pack = build_context_pack(chunks, 120, BudgetPolicy::Balanced);
        let ids = pack
            .chunks
            .iter()
            .map(|c| c.chunk_id.as_str())
            .collect::<Vec<_>>();

        assert_eq!(ids, vec!["struct-1", "problem-1", "retrieved-1"]);
        assert_eq!(pack.policy, BudgetPolicy::Balanced);
        assert_eq!(pack.used_budget + pack.remaining_budget, pack.budget);
    }

    #[test]
    fn build_context_pack_selects_compressed_variant_when_budget_is_tight() {
        let original_body = "module demo\nfn heavy_compute() -> i64\nline one\nline two\nline three";
        let pack = build_context_pack(
            vec![chunk("struct-compact", "module_header", original_body, 80, 0.0)],
            25,
            BudgetPolicy::Balanced,
        );

        assert_eq!(pack.chunks.len(), 1);
        assert!(pack.chunks[0].token_cost <= 25);
        assert!(
            pack.chunks[0].token_cost < 80,
            "a lower-cost variant should be selected under budget pressure"
        );
        assert_eq!(pack.used_budget + pack.remaining_budget, pack.budget);
    }
}
