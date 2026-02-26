//! magpie_memory

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;
const MAX_TERM_CHARS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MmsTermKind {
    Word,
    Symbol,
    Number,
    Code,
    Diag,
    Path,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MmsTerm {
    pub term_text: String,
    pub term_kind: MmsTermKind,
    pub position: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MmsItem {
    pub item_id: String,
    pub kind: String,
    pub sid: String,
    pub fqn: String,
    pub module_sid: String,
    pub source_digest: String,
    pub body_digest: String,
    pub text: String,
    pub tags: Vec<String>,
    pub priority: u32,
    pub token_cost: BTreeMap<String, u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MmsPosting {
    pub doc_idx: usize,
    pub tf: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MmsIndex {
    pub items: Vec<MmsItem>,
    pub inverted_index: HashMap<String, Vec<MmsPosting>>,
    pub doc_lens: Vec<usize>,
    pub n_docs: usize,
    pub avgdl: f64,
    pub k1: f64,
    pub b: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MmsResult {
    pub item_id: String,
    pub score: f64,
    pub token_cost: u32,
    pub item: MmsItem,
}

pub fn tokenize_mms(text: &str) -> Vec<MmsTerm> {
    tokenize_impl(text, TokenizeMode::Document)
}

pub fn build_index(items: &[MmsItem]) -> MmsIndex {
    let mut inverted_index: HashMap<String, Vec<MmsPosting>> = HashMap::new();
    let mut doc_lens = Vec::with_capacity(items.len());

    for (doc_idx, item) in items.iter().enumerate() {
        let terms = tokenize_mms(&item.text);
        let mut tf_map: HashMap<String, u32> = HashMap::new();
        for term in terms {
            *tf_map.entry(term.term_text).or_insert(0) += 1;
        }

        let doc_len: usize = tf_map.values().map(|v| *v as usize).sum();
        doc_lens.push(doc_len);

        for (term, tf) in tf_map {
            inverted_index
                .entry(term)
                .or_default()
                .push(MmsPosting { doc_idx, tf });
        }
    }

    for postings in inverted_index.values_mut() {
        postings.sort_by_key(|p| p.doc_idx);
    }

    let n_docs = items.len();
    let total_len: usize = doc_lens.iter().sum();
    let avgdl = if n_docs == 0 {
        0.0
    } else {
        total_len as f64 / n_docs as f64
    };

    MmsIndex {
        items: items.to_vec(),
        inverted_index,
        doc_lens,
        n_docs,
        avgdl,
        k1: BM25_K1,
        b: BM25_B,
    }
}

pub fn query_bm25(index: &MmsIndex, query: &str, k: usize) -> Vec<MmsResult> {
    if k == 0 || index.n_docs == 0 {
        return Vec::new();
    }

    let query_terms = tokenize_impl(query, TokenizeMode::Query);
    if query_terms.is_empty() {
        return Vec::new();
    }

    let avgdl = if index.avgdl > 0.0 { index.avgdl } else { 1.0 };
    let mut base_scores: HashMap<usize, f64> = HashMap::new();

    for qterm in &query_terms {
        let Some(postings) = index.inverted_index.get(&qterm.term_text) else {
            continue;
        };

        let df = postings.len() as f64;
        let n = index.n_docs as f64;
        let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();

        for posting in postings {
            let tf = posting.tf as f64;
            let dl = index.doc_lens[posting.doc_idx] as f64;
            let denom = tf + index.k1 * (1.0 - index.b + index.b * dl / avgdl);
            if denom == 0.0 {
                continue;
            }
            let term_score = idf * ((tf * (index.k1 + 1.0)) / denom);
            *base_scores.entry(posting.doc_idx).or_insert(0.0) += term_score;
        }
    }

    if base_scores.is_empty() {
        return Vec::new();
    }

    let query_diag_codes: BTreeSet<String> = query_terms
        .iter()
        .filter(|t| t.term_kind == MmsTermKind::Diag)
        .map(|t| t.term_text.clone())
        .collect();

    let query_module_terms: BTreeSet<String> = query_terms
        .iter()
        .filter_map(|t| match t.term_kind {
            MmsTermKind::Word | MmsTermKind::Code | MmsTermKind::Symbol | MmsTermKind::Path => {
                Some(t.term_text.clone())
            }
            _ => None,
        })
        .collect();

    let mut results = Vec::with_capacity(base_scores.len());
    for (doc_idx, base_score) in base_scores {
        let item = &index.items[doc_idx];
        let score = apply_field_boosts(base_score, item, &query_diag_codes, &query_module_terms);
        results.push(MmsResult {
            item_id: item.item_id.clone(),
            score,
            token_cost: item_token_cost(item),
            item: item.clone(),
        });
    }

    results.sort_by(compare_results);
    results.truncate(results.len().min(k));
    results
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenizeMode {
    Document,
    Query,
}

fn tokenize_impl(text: &str, mode: TokenizeMode) -> Vec<MmsTerm> {
    let normalized = normalize_input(text);
    if normalized.is_empty() {
        return Vec::new();
    }

    let with_stopwords = scan_terms(&normalized, true);
    if mode == TokenizeMode::Query && with_stopwords.is_empty() {
        // Appendix E allows preserving stopwords when a query is only stopwords.
        return scan_terms(&normalized, false);
    }

    with_stopwords
}

fn normalize_input(text: &str) -> String {
    // NOTE: Full NFKC + full case folding requires an external Unicode crate.
    // This implementation is deterministic and applies newline + whitespace normalization,
    // with per-term lowercase folding during token emission.
    let replaced = text.replace("\r\n", "\n").replace('\r', "\n");
    collapse_whitespace(&replaced)
}

fn collapse_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev_ws = false;

    for ch in text.chars() {
        if ch.is_whitespace() {
            if !prev_ws {
                out.push(' ');
                prev_ws = true;
            }
        } else {
            out.push(ch);
            prev_ws = false;
        }
    }

    out.trim().to_string()
}

fn scan_terms(text: &str, apply_stopwords: bool) -> Vec<MmsTerm> {
    let mut out = Vec::new();
    let mut position = 0usize;
    let mut i = 0usize;

    while i < text.len() {
        let ch = text[i..]
            .chars()
            .next()
            .expect("scan position must be on a char boundary");

        let mut best: Option<(usize, MmsTermKind, usize)> = None;

        consider_candidate(&mut best, i, 0, match_diag(text, i), MmsTermKind::Diag);
        consider_candidate(
            &mut best,
            i,
            1,
            match_symbol(text, i),
            MmsTermKind::Symbol,
        );
        consider_candidate(&mut best, i, 2, match_path(text, i), MmsTermKind::Path);
        consider_candidate(
            &mut best,
            i,
            3,
            match_number(text, i),
            MmsTermKind::Number,
        );
        consider_candidate(&mut best, i, 4, match_word(text, i), MmsTermKind::Word);

        if let Some((end, kind, _priority_rank)) = best {
            let raw_token = &text[i..end];
            let term_text = fold_case(raw_token);
            maybe_push_term(
                &mut out,
                &mut position,
                term_text,
                kind,
                apply_stopwords,
            );

            if matches!(kind, MmsTermKind::Symbol | MmsTermKind::Word) {
                let mut seen_codes = BTreeSet::new();
                for part in decompose_identifier(raw_token) {
                    let code = fold_case(&part);
                    if seen_codes.insert(code.clone()) {
                        maybe_push_term(
                            &mut out,
                            &mut position,
                            code,
                            MmsTermKind::Code,
                            apply_stopwords,
                        );
                    }
                }
            }

            i = end;
        } else {
            i += ch.len_utf8();
        }
    }

    out
}

fn consider_candidate(
    best: &mut Option<(usize, MmsTermKind, usize)>,
    start: usize,
    rank: usize,
    end: Option<usize>,
    kind: MmsTermKind,
) {
    let Some(end) = end else {
        return;
    };
    let len = end.saturating_sub(start);
    if len == 0 {
        return;
    }

    match best {
        None => *best = Some((end, kind, rank)),
        Some((best_end, _best_kind, best_rank)) => {
            let best_len = best_end.saturating_sub(start);
            if len > best_len || (len == best_len && rank < *best_rank) {
                *best = Some((end, kind, rank));
            }
        }
    }
}

fn match_diag(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if start + 7 > bytes.len() {
        return None;
    }

    let seg = &bytes[start..start + 7];
    if !seg.iter().all(u8::is_ascii) {
        return None;
    }

    if seg[0].to_ascii_lowercase() == b'm'
        && seg[1].to_ascii_lowercase() == b'p'
        && seg[2].is_ascii_alphabetic()
        && seg[3].is_ascii_digit()
        && seg[4].is_ascii_digit()
        && seg[5].is_ascii_digit()
        && seg[6].is_ascii_digit()
    {
        Some(start + 7)
    } else {
        None
    }
}

fn match_symbol(text: &str, start: usize) -> Option<usize> {
    let end = consume_while(text, start, is_symbol_char);
    if end <= start {
        return None;
    }

    let token = &text[start..end];
    let has_symbol_marker = token.contains('@') || token.contains('%');
    let marker_shape = token.starts_with('@')
        || token.starts_with('%')
        || token.contains(".@")
        || token.contains(".%");

    if has_symbol_marker && marker_shape {
        Some(end)
    } else {
        None
    }
}

fn match_path(text: &str, start: usize) -> Option<usize> {
    let end = consume_while(text, start, is_path_char);
    if end <= start {
        return None;
    }

    let token = &text[start..end];
    if token.contains('/') {
        Some(end)
    } else {
        None
    }
}

fn match_number(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if start >= bytes.len() || !bytes[start].is_ascii() {
        return None;
    }

    if start + 2 <= bytes.len()
        && bytes[start] == b'0'
        && (bytes[start + 1] == b'x' || bytes[start + 1] == b'X')
    {
        let mut i = start + 2;
        while i < bytes.len() && bytes[i].is_ascii_hexdigit() {
            i += 1;
        }
        if i > start + 2 {
            return Some(i);
        }
        return None;
    }

    if !bytes[start].is_ascii_digit() {
        return None;
    }

    let mut i = start + 1;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    Some(i)
}

fn match_word(text: &str, start: usize) -> Option<usize> {
    let end = consume_while(text, start, is_word_char);
    (end > start).then_some(end)
}

fn consume_while<F>(text: &str, start: usize, mut pred: F) -> usize
where
    F: FnMut(char) -> bool,
{
    let mut end = start;
    for (off, ch) in text[start..].char_indices() {
        if !pred(ch) {
            break;
        }
        end = start + off + ch.len_utf8();
    }
    end
}

fn is_word_char(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '_'
}

fn is_symbol_char(ch: char) -> bool {
    ch.is_alphanumeric() || matches!(ch, '@' | '%' | '.' | ':' | '/' | '_' | '-')
}

fn is_path_char(ch: char) -> bool {
    ch.is_alphanumeric() || matches!(ch, '.' | ':' | '/' | '_' | '-')
}

fn fold_case(text: &str) -> String {
    text.chars().flat_map(char::to_lowercase).collect()
}

fn maybe_push_term(
    out: &mut Vec<MmsTerm>,
    position: &mut usize,
    term_text: String,
    term_kind: MmsTermKind,
    apply_stopwords: bool,
) {
    let mut text = truncate_chars(&term_text, MAX_TERM_CHARS);

    if term_kind != MmsTermKind::Diag && text.chars().count() < 2 {
        return;
    }

    if apply_stopwords
        && matches!(term_kind, MmsTermKind::Word | MmsTermKind::Code)
        && is_stopword(&text)
    {
        return;
    }

    if term_kind == MmsTermKind::Diag {
        // Canonicalize diagnostics for robust matching against tags/queries.
        text = text.to_ascii_uppercase();
    }

    out.push(MmsTerm {
        term_text: text,
        term_kind,
        position: *position,
    });
    *position += 1;
}

fn truncate_chars(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    text.chars().take(limit).collect()
}

fn is_stopword(term: &str) -> bool {
    matches!(
        term,
        "a"
            | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "be"
            | "by"
            | "for"
            | "from"
            | "has"
            | "have"
            | "if"
            | "in"
            | "is"
            | "it"
            | "its"
            | "of"
            | "on"
            | "or"
            | "that"
            | "the"
            | "to"
            | "was"
            | "were"
            | "with"
            | "true"
            | "false"
            | "unit"
            | "ret"
            | "br"
            | "cbr"
            | "switch"
            | "bb"
            | "fn"
            | "module"
            | "imports"
            | "exports"
            | "digest"
            | "const"
            | "call"
            | "call_void"
            | "new"
    )
}

fn decompose_identifier(raw: &str) -> Vec<String> {
    let mut out = Vec::new();

    for segment in raw
        .split(|ch: char| !(ch.is_alphanumeric() || ch == '_' || ch == '-'))
        .filter(|s| !s.is_empty())
    {
        for piece in segment.split(['_', '-']).filter(|s| !s.is_empty()) {
            for part in split_camel_and_digit_boundaries(piece) {
                if part.chars().count() >= 2 {
                    out.push(part);
                }
            }
        }

        let compressed: String = segment
            .chars()
            .filter(|ch| *ch != '_' && *ch != '-')
            .collect();
        if compressed.chars().count() >= 2 {
            out.push(compressed);
        }
    }

    out
}

fn split_camel_and_digit_boundaries(piece: &str) -> Vec<String> {
    let chars: Vec<char> = piece.chars().collect();
    if chars.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut start = 0usize;

    for i in 1..chars.len() {
        let prev = chars[i - 1];
        let curr = chars[i];
        let next = chars.get(i + 1).copied();

        let lower_to_upper = prev.is_lowercase() && curr.is_uppercase();
        let acronym_to_word =
            prev.is_uppercase() && curr.is_uppercase() && next.map(char::is_lowercase).unwrap_or(false);
        let alpha_digit = prev.is_alphabetic() && curr.is_ascii_digit();
        let digit_alpha = prev.is_ascii_digit() && curr.is_alphabetic();

        if lower_to_upper || acronym_to_word || alpha_digit || digit_alpha {
            out.push(chars[start..i].iter().collect());
            start = i;
        }
    }

    out.push(chars[start..].iter().collect());
    out
}

fn apply_field_boosts(
    base_score: f64,
    item: &MmsItem,
    query_diag_codes: &BTreeSet<String>,
    query_module_terms: &BTreeSet<String>,
) -> f64 {
    let mut boosted = base_score * kind_boost(&item.kind);

    if !query_diag_codes.is_empty() && doc_has_diag_tag(item, query_diag_codes) {
        boosted *= 1.30;
    }

    if !query_module_terms.is_empty() && doc_has_module_overlap(item, query_module_terms) {
        boosted *= 1.15;
    }

    boosted * priority_boost(item.priority)
}

fn kind_boost(kind: &str) -> f64 {
    match kind {
        "diag_template" => 1.40,
        "spec_excerpt" => 1.25,
        "mpd_signature" => 1.20,
        "symbol_capsule" => 1.15,
        "test_case" => 1.10,
        "repair_episode" => 1.05,
        _ => 1.0,
    }
}

fn priority_boost(priority: u32) -> f64 {
    (0.5 + priority as f64 / 100.0).clamp(0.75, 1.50)
}

fn doc_has_diag_tag(item: &MmsItem, query_diag_codes: &BTreeSet<String>) -> bool {
    item.tags
        .iter()
        .map(|tag| fold_case(tag).to_ascii_uppercase())
        .any(|tag| query_diag_codes.contains(&tag))
}

fn doc_has_module_overlap(item: &MmsItem, query_module_terms: &BTreeSet<String>) -> bool {
    let mut doc_terms = BTreeSet::new();

    for t in tokenize_mms(&item.fqn) {
        if matches!(
            t.term_kind,
            MmsTermKind::Word | MmsTermKind::Code | MmsTermKind::Symbol | MmsTermKind::Path
        ) {
            doc_terms.insert(t.term_text);
        }
    }

    for t in tokenize_mms(&item.module_sid) {
        if matches!(
            t.term_kind,
            MmsTermKind::Word | MmsTermKind::Code | MmsTermKind::Symbol | MmsTermKind::Path
        ) {
            doc_terms.insert(t.term_text);
        }
    }

    query_module_terms
        .iter()
        .any(|term| doc_terms.contains(term))
}

fn item_token_cost(item: &MmsItem) -> u32 {
    item
        .token_cost
        .get("approx:utf8_4chars")
        .copied()
        .or_else(|| item.token_cost.values().min().copied())
        .unwrap_or(u32::MAX)
}

fn compare_results(a: &MmsResult, b: &MmsResult) -> Ordering {
    b.score
        .total_cmp(&a.score)
        .then_with(|| a.token_cost.cmp(&b.token_cost))
        .then_with(|| a.item_id.cmp(&b.item_id))
}
