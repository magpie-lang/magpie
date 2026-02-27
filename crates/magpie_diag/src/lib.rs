//! Magpie diagnostics engine (§26).

use magpie_ast::Span;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Diagnostic {
    pub code: String,
    pub severity: Severity,
    pub title: String,
    pub primary_span: Option<Span>,
    pub secondary_spans: Vec<Span>,
    pub message: String,
    pub explanation_md: Option<String>,
    pub why: Option<WhyTrace>,
    pub suggested_fixes: Vec<SuggestedFix>,
    #[serde(default)]
    pub rag_bundle: Vec<serde_json::Value>,
    #[serde(default)]
    pub related_docs: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Severity {
    #[serde(rename = "error")]
    Error,
    #[serde(rename = "warning")]
    Warning,
    #[serde(rename = "info")]
    Info,
    #[serde(rename = "hint")]
    Hint,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WhyTrace {
    pub kind: String,
    pub trace: Vec<WhyEvent>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WhyEvent {
    pub description: String,
    pub span: Option<Span>,
}

impl WhyTrace {
    pub fn new(kind: impl Into<String>, trace: Vec<WhyEvent>) -> Self {
        Self {
            kind: kind.into(),
            trace,
        }
    }

    pub fn ownership(trace: Vec<WhyEvent>) -> Self {
        Self::new("ownership", trace)
    }
}

impl WhyEvent {
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            span: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuggestedFix {
    pub title: String,
    pub patch_format: String,
    pub patch: String,
    pub confidence: f64,
    #[serde(default)]
    pub requires_fmt: bool,
    #[serde(default)]
    pub applies_to: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    pub produces: std::collections::BTreeMap<String, String>,
}

#[derive(Clone, Debug, Default)]
pub struct DiagnosticBag {
    pub diagnostics: Vec<Diagnostic>,
    pub max_errors: usize,
}

impl DiagnosticBag {
    pub fn new(max_errors: usize) -> Self {
        Self {
            diagnostics: Vec::new(),
            max_errors,
        }
    }

    pub fn emit(&mut self, diag: Diagnostic) {
        if self.diagnostics.len() < self.max_errors {
            self.diagnostics.push(diag);
        }
    }

    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| matches!(d.severity, Severity::Error))
    }

    pub fn error_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| matches!(d.severity, Severity::Error))
            .count()
    }
}

/// JSON output envelope (§26.1)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutputEnvelope {
    pub magpie_version: String,
    pub command: String,
    pub target: Option<String>,
    pub success: bool,
    pub artifacts: Vec<String>,
    pub diagnostics: Vec<Diagnostic>,
    #[serde(default = "default_graphs")]
    pub graphs: serde_json::Value,
    pub timing_ms: serde_json::Value,
    pub llm_budget: Option<serde_json::Value>,
}

fn default_graphs() -> serde_json::Value {
    serde_json::json!({})
}

/// Diagnostic code namespaces (§26.3)
pub mod codes {
    // Parse/lex
    pub const MPP_PREFIX: &str = "MPP";
    // Types
    pub const MPT_PREFIX: &str = "MPT";
    pub const MPT1005: &str = "MPT1005"; // VALUE_TYPE_CONTAINS_HEAP
    pub const MPT1010: &str = "MPT1010"; // RECURSIVE_VALUE_TYPE
    pub const MPT1020: &str = "MPT1020"; // VALUE_ENUM_DEFERRED
    pub const MPT1021: &str = "MPT1021"; // AGGREGATE_TYPE_DEFERRED
    pub const MPT1022: &str = "MPT1022"; // COLLECTION_DUPLICATION_REQUIRES_DUPABLE
    pub const MPT1023: &str = "MPT1023"; // MISSING_REQUIRED_TRAIT_IMPL
    pub const MPT1030: &str = "MPT1030"; // TCALLABLE_SUSPEND_FORBIDDEN
    pub const MPT1200: &str = "MPT1200"; // ORPHAN_IMPL
                                         // Ownership
    pub const MPO_PREFIX: &str = "MPO";
    pub const MPO0003: &str = "MPO0003"; // BORROW_ESCAPES_SCOPE
    pub const MPO0004: &str = "MPO0004"; // SHARED_MUTATION
    pub const MPO0011: &str = "MPO0011"; // MOVE_WHILE_BORROWED
    pub const MPO0101: &str = "MPO0101"; // BORROW_CROSSES_BLOCK
    pub const MPO0102: &str = "MPO0102"; // BORROW_IN_PHI
    pub const MPO0103: &str = "MPO0103"; // MAP_GET_REQUIRES_DUPABLE_V
                                         // ARC
    pub const MPA_PREFIX: &str = "MPA";
    // SSA verification
    pub const MPS_PREFIX: &str = "MPS";
    pub const MPS0024: &str = "MPS0024"; // UNSAFE_PTR_OP_OUTSIDE_UNSAFE_CONTEXT
    pub const MPS0025: &str = "MPS0025"; // UNSAFE_FN_CALL_OUTSIDE_UNSAFE_CONTEXT
                                         // Async
    pub const MPAS_PREFIX: &str = "MPAS";
    pub const MPAS0001: &str = "MPAS0001"; // SUSPEND_IN_NON_ASYNC
                                           // FFI
    pub const MPF0001: &str = "MPF0001"; // FFI_RETURN_OWNERSHIP_REQUIRED
                                         // GPU
    pub const MPG_PREFIX: &str = "MPG";
    // Web
    pub const MPW_PREFIX: &str = "MPW";
    pub const MPW1001: &str = "MPW1001"; // DUPLICATE_ROUTE
                                         // Package
    pub const MPK_PREFIX: &str = "MPK";
    // Lint / LLM
    pub const MPL_PREFIX: &str = "MPL";
    pub const MPL0801: &str = "MPL0801"; // TOKEN_BUDGET_TOO_SMALL
    pub const MPL0802: &str = "MPL0802"; // TOKENIZER_FALLBACK_USED
    pub const MPL2001: &str = "MPL2001"; // UNUSED_VARIABLE
    pub const MPL2002: &str = "MPL2002"; // UNUSED_FUNCTION
    pub const MPL2003: &str = "MPL2003"; // UNNECESSARY_BORROW
    pub const MPL2005: &str = "MPL2005"; // EMPTY_BLOCK
    pub const MPL2007: &str = "MPL2007"; // UNREACHABLE_CODE
    pub const MPL2020: &str = "MPL2020"; // EXCESSIVE_MONO
    pub const MPL2021: &str = "MPL2021"; // MIXED_GENERICS_MODE
}

/// Token budget configuration (§3.1/§3.2/§3.4).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenBudget {
    pub budget: u32,
    pub tokenizer: String,
    pub policy: String,
}

/// Patch JSON envelope (§27.2).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PatchEnvelope {
    pub title: String,
    pub patch_format: String,
    pub patch: String,
    pub applies_to: std::collections::BTreeMap<String, String>,
    pub produces: std::collections::BTreeMap<String, String>,
    pub confidence: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BudgetDrop {
    pub field: String,
    pub reason: String,
}

/// LLM budget report (§3.4).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BudgetReport {
    pub token_budget: u32,
    pub tokenizer: String,
    pub estimated_tokens: u32,
    pub policy: String,
    pub dropped: Vec<BudgetDrop>,
}

fn estimated_envelope_tokens(envelope: &OutputEnvelope, tokenizer: &str) -> u32 {
    let payload = canonical_json_encode(envelope).unwrap_or_default();
    estimate_tokens(&payload, tokenizer)
}

pub fn canonical_json_encode<T: serde::Serialize>(value: &T) -> Result<String, serde_json::Error> {
    let value = serde_json::to_value(value)?;
    Ok(canonical_json_string(&value))
}

pub fn canonical_json_string(value: &Value) -> String {
    let mut out = String::new();
    write_canonical_json(value, &mut out);
    out
}

fn write_canonical_json(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(v) => {
            if *v {
                out.push_str("true");
            } else {
                out.push_str("false");
            }
        }
        Value::Number(v) => out.push_str(&v.to_string()),
        Value::String(v) => {
            let escaped = serde_json::to_string(v).expect("string JSON encoding cannot fail");
            out.push_str(&escaped);
        }
        Value::Array(items) => {
            out.push('[');
            for (idx, item) in items.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                write_canonical_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            let mut keys = map.keys().cloned().collect::<Vec<_>>();
            keys.sort_unstable();
            for (idx, key) in keys.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                let encoded_key =
                    serde_json::to_string(key).expect("object key JSON encoding cannot fail");
                out.push_str(&encoded_key);
                out.push(':');
                if let Some(item) = map.get(key) {
                    write_canonical_json(item, out);
                }
            }
            out.push('}');
        }
    }
}

fn truncate_chars(input: &str, max_chars: usize) -> String {
    input.chars().take(max_chars).collect()
}

/// Estimate token count for a text payload.
///
/// `approx:utf8_4chars` uses zero-dependency approximation:
/// `ceil(utf8_bytes / 4)`.
pub fn estimate_tokens(text: &str, _tokenizer: &str) -> u32 {
    (text.len() as u32).div_ceil(4)
}

/// Enforce token budget with deterministic drop order (§3.3): Tier 4 -> 3 -> 2 -> 1.
pub fn enforce_budget(envelope: &mut OutputEnvelope, budget: &TokenBudget) {
    let mut dropped: Vec<BudgetDrop> = Vec::new();
    let tokenizer = if budget.tokenizer.trim().is_empty() {
        "approx:utf8_4chars".to_string()
    } else {
        budget.tokenizer.clone()
    };

    // Tier 0 cap: keep only short messages (<= 200 chars).
    for (i, diag) in envelope.diagnostics.iter_mut().enumerate() {
        if diag.message.chars().count() > 200 {
            diag.message = truncate_chars(&diag.message, 200);
            dropped.push(BudgetDrop {
                field: format!("diagnostics[{i}].message"),
                reason: "tier0_truncate".to_string(),
            });
        }
    }

    if estimated_envelope_tokens(envelope, &tokenizer) > budget.budget {
        // Tier 4: drop long explanations; minimize verbose traces.
        for (i, diag) in envelope.diagnostics.iter_mut().enumerate() {
            if diag.explanation_md.take().is_some() {
                dropped.push(BudgetDrop {
                    field: format!("diagnostics[{i}].explanation_md"),
                    reason: "budget_tier4".to_string(),
                });
            }
            if let Some(why) = &mut diag.why {
                if why.trace.len() > 1 {
                    why.trace.truncate(1);
                    dropped.push(BudgetDrop {
                        field: format!("diagnostics[{i}].why.trace"),
                        reason: "budget_tier4".to_string(),
                    });
                }
            }
            if !diag.rag_bundle.is_empty() {
                diag.rag_bundle.clear();
                dropped.push(BudgetDrop {
                    field: format!("diagnostics[{i}].rag_bundle"),
                    reason: "budget_tier4".to_string(),
                });
            }
            if !diag.related_docs.is_empty() {
                diag.related_docs.clear();
                dropped.push(BudgetDrop {
                    field: format!("diagnostics[{i}].related_docs"),
                    reason: "budget_tier4".to_string(),
                });
            }
        }
    }

    if estimated_envelope_tokens(envelope, &tokenizer) > budget.budget {
        // Tier 3: remove extra patches and truncate very large patch bodies.
        for (i, diag) in envelope.diagnostics.iter_mut().enumerate() {
            if diag.suggested_fixes.len() > 1 {
                diag.suggested_fixes.truncate(1);
                dropped.push(BudgetDrop {
                    field: format!("diagnostics[{i}].suggested_fixes[1..]"),
                    reason: "budget_tier3".to_string(),
                });
            }
            if let Some(fix) = diag.suggested_fixes.first_mut() {
                if fix.patch.len() > 4096 {
                    fix.patch.truncate(4096);
                    dropped.push(BudgetDrop {
                        field: format!("diagnostics[{i}].suggested_fixes[0].patch"),
                        reason: "budget_tier3_truncate".to_string(),
                    });
                }
            }
        }
    }

    if estimated_envelope_tokens(envelope, &tokenizer) > budget.budget {
        // Tier 2: drop secondary spans.
        for (i, diag) in envelope.diagnostics.iter_mut().enumerate() {
            if !diag.secondary_spans.is_empty() {
                diag.secondary_spans.clear();
                dropped.push(BudgetDrop {
                    field: format!("diagnostics[{i}].secondary_spans"),
                    reason: "budget_tier2".to_string(),
                });
            }
        }
    }

    if estimated_envelope_tokens(envelope, &tokenizer) > budget.budget {
        // Tier 1: drop suggested fix and why trace only as last resort.
        for (i, diag) in envelope.diagnostics.iter_mut().enumerate() {
            if !diag.suggested_fixes.is_empty() {
                diag.suggested_fixes.clear();
                dropped.push(BudgetDrop {
                    field: format!("diagnostics[{i}].suggested_fixes"),
                    reason: "budget_tier1".to_string(),
                });
            }
            if diag.why.take().is_some() {
                dropped.push(BudgetDrop {
                    field: format!("diagnostics[{i}].why"),
                    reason: "budget_tier1".to_string(),
                });
            }
        }
    }

    if estimated_envelope_tokens(envelope, &tokenizer) > budget.budget {
        // Hard trim: remove artifacts and extra diagnostics before final fallback.
        if envelope.graphs != default_graphs() {
            envelope.graphs = default_graphs();
            dropped.push(BudgetDrop {
                field: "graphs".to_string(),
                reason: "budget_hard_trim".to_string(),
            });
        }
        if !envelope.artifacts.is_empty() {
            envelope.artifacts.clear();
            dropped.push(BudgetDrop {
                field: "artifacts".to_string(),
                reason: "budget_hard_trim".to_string(),
            });
        }
        while estimated_envelope_tokens(envelope, &tokenizer) > budget.budget
            && envelope.diagnostics.len() > 1
        {
            envelope.diagnostics.pop();
            dropped.push(BudgetDrop {
                field: "diagnostics[last]".to_string(),
                reason: "budget_hard_trim".to_string(),
            });
        }
        while estimated_envelope_tokens(envelope, &tokenizer) > budget.budget {
            let Some(diag) = envelope.diagnostics.first_mut() else {
                break;
            };
            let current_len = diag.message.chars().count();
            if current_len <= 64 {
                break;
            }
            let next_len = (current_len / 2).max(64);
            diag.message = truncate_chars(&diag.message, next_len);
            dropped.push(BudgetDrop {
                field: "diagnostics[0].message".to_string(),
                reason: "budget_hard_trim".to_string(),
            });
        }
    }

    if estimated_envelope_tokens(envelope, &tokenizer) > budget.budget {
        // Tier 0 alone still exceeds budget: return MPL0801 minimal envelope.
        let mut tier0_probe = envelope.clone();
        tier0_probe.graphs = default_graphs();
        tier0_probe.artifacts.clear();
        for diag in &mut tier0_probe.diagnostics {
            diag.secondary_spans.clear();
            diag.explanation_md = None;
            diag.why = None;
            diag.suggested_fixes.clear();
            diag.rag_bundle.clear();
            diag.related_docs.clear();
            if diag.message.chars().count() > 200 {
                diag.message = truncate_chars(&diag.message, 200);
            }
        }
        let recommended_budget = estimated_envelope_tokens(&tier0_probe, &tokenizer)
            .saturating_mul(2)
            .max(1);

        envelope.success = false;
        envelope.graphs = default_graphs();
        envelope.artifacts.clear();
        envelope.diagnostics = vec![Diagnostic {
            code: codes::MPL0801.to_string(),
            severity: Severity::Error,
            title: "token budget too small".to_string(),
            primary_span: None,
            secondary_spans: Vec::new(),
            message: format!(
                "Configured budget {} is too small; recommended minimum is {}.",
                budget.budget, recommended_budget
            ),
            explanation_md: None,
            why: None,
            suggested_fixes: Vec::new(),
            rag_bundle: Vec::new(),
            related_docs: Vec::new(),
        }];
        dropped.push(BudgetDrop {
            field: "diagnostics".to_string(),
            reason: "tier0_only_fallback".to_string(),
        });
    }

    let mut report = BudgetReport {
        token_budget: budget.budget,
        tokenizer: tokenizer.clone(),
        estimated_tokens: 0,
        policy: budget.policy.clone(),
        dropped,
    };
    envelope.llm_budget = serde_json::to_value(&report).ok();
    report.estimated_tokens = estimated_envelope_tokens(envelope, &tokenizer);
    if report.estimated_tokens > budget.budget {
        let dropped_count = report.dropped.len();
        report.dropped = vec![BudgetDrop {
            field: "dropped".to_string(),
            reason: format!("{} entries omitted", dropped_count),
        }];
        envelope.llm_budget = serde_json::to_value(&report).ok();
        report.estimated_tokens = estimated_envelope_tokens(envelope, &tokenizer);
        if report.estimated_tokens > budget.budget {
            envelope.llm_budget = None;
            return;
        }
    }
    envelope.llm_budget = serde_json::to_value(&report).ok();
}

/// Return a compact remediation template for major diagnostic namespaces.
pub fn explain_code(code: &str) -> Option<String> {
    let normalized = code.trim().to_ascii_uppercase();
    let template = match normalized.as_str() {
        "MPP0001" => "Source I/O failed. Example: file read/write path missing. Fix template: verify path, create directories, and ensure read/write permissions.",
        "MPP0002" => "Syntax/tokenization error. Example: invalid token or malformed grammar production. Fix template: run `magpie fmt`, then correct the highlighted token sequence.",
        "MPP0003" => "Artifact emission failed. Example: `.ll`/`.mpir`/graph write error. Fix template: verify target directory permissions and free disk space, then rebuild.",
        "MPM0001" => "MPIR lowering produced no modules. Example: HIR lowered to empty module set. Fix template: confirm entry module parses and lowering pass receives at least one resolved module.",
        "MPHIR01" => "HIR invariant violated: `getfield` object is not borrow/mutborrow. Example: direct field read from owned value. Fix template: borrow the object first, then perform `getfield`.",
        "MPHIR02" => "HIR invariant violated: `setfield` requires mutborrow. Example: writing through shared/owned handle. Fix template: obtain `mutborrow` and perform mutation through that handle.",
        "MPHIR03" => "HIR invariant violated: borrow value escapes via return. Example: function returns `borrow T`. Fix template: return owned/shared value or redesign API to keep borrow local.",
        "MPO0003" => "Borrow escapes scope. Example: storing borrow in global/collection. Fix template: store owned/shared values instead, or shorten borrow to local use only.",
        "MPO0004" => "Mutation through shared/invalid ownership mode. Example: mutating intrinsic called on shared handle. Fix template: require unique/mutborrow receiver or clone into mutable owner.",
        "MPO0007" => "Use-after-move. Example: reading `%x` after `move %x`. Fix template: avoid subsequent uses, clone before move if sharing is required.",
        "MPO0011" => "Move while borrowed. Example: `move %x` while borrow of `%x` is alive. Fix template: end borrow scope before move, or duplicate value when legal.",
        "MPO0101" => "Borrow crosses basic block boundary. Example: borrow defined in one block, used in another. Fix template: recreate borrow in each block or pass owned/shared value instead.",
        "MPO0102" => "Borrow in phi is forbidden. Example: phi incoming includes borrow handle. Fix template: phi owned/shared values only; recreate borrows after join.",
        "MPO0103" => "Map `get` requires Dupable value type. Example: map value is move-only. Fix template: use `map.get_ref`, change value type to Dupable, or redesign ownership flow.",
        "MPS0000" => "Module resolution failed without specific diagnostics. Fix template: validate module headers/import graph and rerun with full diagnostics.",
        "MPS0001" => "Duplicate definition / SSA single-definition violation. Example: module path or `%local` defined twice. Fix template: keep one declaration per symbol/local.",
        "MPS0002" => "Unresolved reference/import target. Example: imported module/symbol missing. Fix template: add missing module, correct import path, or declare value before use.",
        "MPS0003" => "Symbol resolution failure. Example: import item or SSA local cannot be resolved. Fix template: fix name/path typo and ensure declaration exists in scope.",
        "MPS0004" => "Namespace/SID consistency violation. Example: import conflicts or SID kind mismatch. Fix template: remove conflicting symbol names and ensure SID kind matches entity.",
        "MPS0005" => "Namespace/SID validity violation. Example: type import conflict or malformed SID. Fix template: use unique type names and valid canonical SID format.",
        "MPS0006" => "Ambiguous import. Example: same short name resolves to multiple FQNs. Fix template: import with unambiguous names or reference by fully-qualified name.",
        "MPS0008" => "Invalid control-flow edge. Example: branch/switch targets missing block. Fix template: ensure every terminator target exists in function block list.",
        "MPS0009" => "MPIR entry block invariant failed. Example: expected `bb0` missing. Fix template: preserve canonical entry block and reachable CFG structure.",
        "MPS0010" => "Type/invariant check failed. Example: unknown type in signature or invalid phi/value type. Fix template: use declared types and keep phi/value invariants valid.",
        "MPS0011" => "Instruction typing/evaluation mismatch. Example: op argument type incompatible with expected type. Fix template: insert explicit cast or correct operand types.",
        "MPS0012" => "Call arity mismatch. Example: passed argument count differs from callee signature. Fix template: align call args with signature exactly.",
        "MPS0013" => "Return type mismatch. Example: returned value type differs from function return type. Fix template: return the declared type or adjust function signature.",
        "MPS0014" => "ARC stage violation. Example: `arc.*` op appears before ARC insertion stage. Fix template: run ARC insertion before verification or remove pre-ARC `arc.*` ops.",
        "MPS0015" => "Call argument contract mismatch. Example: argument type/ownership differs from parameter contract. Fix template: pass values with matching type and ownership mode.",
        "MPS0016" => "Unknown or invalid callee reference. Example: call target SID/name not found. Fix template: ensure callee exists and reference is fully resolved.",
        "MPS0017" => "Conditional branch predicate type invalid. Example: non-boolean condition in `cbr`. Fix template: produce `bool` condition before branching.",
        "MPS0020" => "No-overloads rule violated in function/global namespace. Fix template: rename duplicate `@` symbols within module.",
        "MPS0021" => "No-overloads rule violated in type namespace. Fix template: rename duplicate `T` symbols within module.",
        "MPS0022" => "No-overloads rule violated for globals/functions. Fix template: keep each `@` symbol name unique per module.",
        "MPS0023" => "No-overloads rule violated in signature namespace. Fix template: rename duplicate `sig` declarations.",
        "MPS0024" => "Unsafe raw-pointer opcode used outside unsafe context. Fix template: move `ptr.*` into `unsafe {}` or mark function `unsafe fn`.",
        "MPS0025" => "Unsafe function call outside unsafe context. Fix template: wrap call in `unsafe {}` or perform call from `unsafe fn`.",
        "MPL0001" => "Unknown emit kind. Example: unsupported `--emit` value. Fix template: use supported emits (e.g. `exe`, `llvm-ir`, `mpir`, `symgraph`).",
        "MPL0002" => "Requested artifact missing. Example: build succeeded but expected emit output file is absent. Fix template: inspect linker/codegen diagnostics and ensure all requested emits are producible.",
        "MPL0801" => "Token budget too small for required output. Fix template: increase `--llm-token-budget` or use `minimal` budget policy.",
        "MPL0802" => "Requested tokenizer unavailable; fallback used. Fix template: install/configure requested tokenizer or explicitly use `approx:utf8_4chars`.",
        "MPL2001" => "Function too large lint. Fix template: split function into smaller helpers and keep CFG/local scope compact.",
        "MPL2002" => "Unused function lint. Fix template: remove dead function, reference it from call sites, or mark as intentionally exported.",
        "MPL2003" => "Unnecessary borrow lint. Fix template: pass owned/shared value directly when no borrow semantics are required.",
        "MPL2005" => "Empty block lint. Fix template: remove empty block or add meaningful instructions/terminator flow.",
        "MPL2007" => "Unreachable code lint. Fix template: delete dead instructions or restructure control flow to make intent explicit.",
        "MPL2020" => "Monomorphization budget exceeded. Fix template: reduce generic instance explosion or enable shared-generics mode.",
        "MPL2021" => "Mixed generics mode conflict. Fix template: use a single generics strategy consistently for the build target/profile.",
        "MPLINK01" => "Primary native link path failed; fallback started. Fix template: install/configure `llc` + system linker, or rely on clang IR fallback.",
        "MPLINK02" => "Native linking unavailable after fallback. Fix template: install linker toolchain for target triple; use LLVM IR artifacts until toolchain is fixed.",
        "MPT2032" => "Impl binding target missing. Example: `impl trait for Type = @fn` references a function that cannot be resolved. Fix template: define/import the function and ensure the fn_ref matches exactly.",
        _ => {
            if normalized.starts_with(codes::MPO_PREFIX) {
                "Ownership remediation: avoid using values after move, shorten borrow lifetimes, and clone only when sharing is required."
            } else if normalized.starts_with(codes::MPT_PREFIX) {
                "Type remediation: align declared and inferred types, satisfy required trait bounds, and remove recursive/value-layout mismatches."
            } else if normalized.starts_with(codes::MPS_PREFIX) {
                "SSA remediation: ensure each value is defined before use, phi inputs match predecessor blocks, and control-flow edges stay structurally valid."
            } else if normalized.starts_with(codes::MPL_PREFIX) {
                "Lint/LLM remediation: adjust build policy, token budget, or code structure to satisfy deterministic output constraints."
            } else {
                return None;
            }
        }
    };

    Some(template.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_json_encode_sorts_object_keys_deterministically() {
        let value = serde_json::json!({
            "zeta": 1,
            "alpha": {
                "beta": 2,
                "aardvark": 3
            },
            "items": [
                {"d": 4, "c": 3},
                {"b": 2, "a": 1}
            ]
        });

        let encoded = canonical_json_encode(&value).expect("canonical encoding should succeed");
        assert_eq!(
            encoded,
            r#"{"alpha":{"aardvark":3,"beta":2},"items":[{"c":3,"d":4},{"a":1,"b":2}],"zeta":1}"#
        );
    }

    #[test]
    fn enforce_budget_truncates_to_fit_budget() {
        let long_text = "x".repeat(8_000);
        let mut envelope = OutputEnvelope {
            magpie_version: "0.1.0".to_string(),
            command: "build".to_string(),
            target: Some("x86_64-unknown-linux".to_string()),
            success: false,
            artifacts: vec!["very-long-artifact-name".repeat(40)],
            diagnostics: vec![
                Diagnostic {
                    code: "MPL2001".to_string(),
                    severity: Severity::Error,
                    title: "too large".to_string(),
                    primary_span: None,
                    secondary_spans: Vec::new(),
                    message: long_text.clone(),
                    explanation_md: Some(long_text.clone()),
                    why: Some(WhyTrace {
                        kind: "trace".to_string(),
                        trace: vec![
                            WhyEvent {
                                description: long_text.clone(),
                                span: None,
                            },
                            WhyEvent {
                                description: long_text.clone(),
                                span: None,
                            },
                        ],
                    }),
                    suggested_fixes: vec![
                        SuggestedFix {
                            title: "fix".to_string(),
                            patch_format: "unified".to_string(),
                            patch: long_text.clone(),
                            confidence: 0.5,
                            requires_fmt: false,
                            applies_to: std::collections::BTreeMap::new(),
                            produces: std::collections::BTreeMap::new(),
                        },
                        SuggestedFix {
                            title: "fix2".to_string(),
                            patch_format: "unified".to_string(),
                            patch: long_text.clone(),
                            confidence: 0.4,
                            requires_fmt: false,
                            applies_to: std::collections::BTreeMap::new(),
                            produces: std::collections::BTreeMap::new(),
                        },
                    ],
                    rag_bundle: vec![serde_json::json!({ "chunk": long_text.clone() })],
                    related_docs: vec!["doc://example".to_string()],
                },
                Diagnostic {
                    code: "MPL2002".to_string(),
                    severity: Severity::Error,
                    title: "also large".to_string(),
                    primary_span: None,
                    secondary_spans: Vec::new(),
                    message: long_text.clone(),
                    explanation_md: Some(long_text),
                    why: None,
                    suggested_fixes: Vec::new(),
                    rag_bundle: Vec::new(),
                    related_docs: Vec::new(),
                },
            ],
            graphs: serde_json::json!({
                "cfg": {
                    "nodes": [1, 2, 3],
                    "edges": [[1, 2], [2, 3]]
                }
            }),
            timing_ms: serde_json::json!({}),
            llm_budget: None,
        };

        let budget = TokenBudget {
            budget: 500,
            tokenizer: "approx:utf8_4chars".to_string(),
            policy: "balanced".to_string(),
        };

        enforce_budget(&mut envelope, &budget);

        if let Some(report) = envelope
            .llm_budget
            .as_ref()
            .and_then(|value| serde_json::from_value::<BudgetReport>(value.clone()).ok())
        {
            assert!(
                !report.dropped.is_empty(),
                "expected budget enforcement to drop some fields, report={report:?}"
            );
        } else {
            assert!(
                estimated_envelope_tokens(&envelope, "approx:utf8_4chars") <= budget.budget,
                "payload should fit budget when budget metadata is omitted"
            );
        }
    }
}
