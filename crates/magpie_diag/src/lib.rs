//! Magpie diagnostics engine (§26).

use magpie_ast::Span;
use serde::{Deserialize, Serialize};

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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuggestedFix {
    pub title: String,
    pub patch_format: String,
    pub patch: String,
    pub confidence: f64,
}

#[derive(Clone, Debug, Default)]
pub struct DiagnosticBag {
    pub diagnostics: Vec<Diagnostic>,
    pub max_errors: usize,
}

impl DiagnosticBag {
    pub fn new(max_errors: usize) -> Self {
        Self { diagnostics: Vec::new(), max_errors }
    }

    pub fn emit(&mut self, diag: Diagnostic) {
        if self.diagnostics.len() < self.max_errors {
            self.diagnostics.push(diag);
        }
    }

    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(|d| matches!(d.severity, Severity::Error))
    }

    pub fn error_count(&self) -> usize {
        self.diagnostics.iter().filter(|d| matches!(d.severity, Severity::Error)).count()
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
    pub timing_ms: serde_json::Value,
    pub llm_budget: Option<serde_json::Value>,
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
    pub const MPL2001: &str = "MPL2001"; // FN_TOO_LARGE
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
    let payload = serde_json::to_string(envelope).unwrap_or_default();
    estimate_tokens(&payload, tokenizer)
}

fn truncate_chars(input: &str, max_chars: usize) -> String {
    input.chars().take(max_chars).collect()
}

/// Estimate token count for a text payload.
///
/// `approx:utf8_4chars` uses zero-dependency approximation:
/// `ceil(utf8_bytes / 4)`.
pub fn estimate_tokens(text: &str, tokenizer: &str) -> u32 {
    match tokenizer {
        "approx:utf8_4chars" => (text.len() as u32 + 3) / 4,
        _ => (text.len() as u32 + 3) / 4,
    }
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
        // Tier 0 alone still exceeds budget: return MPL0801 minimal envelope.
        let mut tier0_probe = envelope.clone();
        tier0_probe.artifacts.clear();
        for diag in &mut tier0_probe.diagnostics {
            diag.secondary_spans.clear();
            diag.explanation_md = None;
            diag.why = None;
            diag.suggested_fixes.clear();
            if diag.message.chars().count() > 200 {
                diag.message = truncate_chars(&diag.message, 200);
            }
        }
        let recommended_budget = estimated_envelope_tokens(&tier0_probe, &tokenizer).saturating_mul(2).max(1);

        envelope.success = false;
        envelope.artifacts.clear();
        envelope.diagnostics = vec![Diagnostic {
            code: codes::MPL0801.to_string(),
            severity: Severity::Error,
            title: "token budget too small".to_string(),
            primary_span: None,
            secondary_spans: Vec::new(),
            message: format!("Configured budget {} is too small; recommended minimum is {}.", budget.budget, recommended_budget),
            explanation_md: None,
            why: None,
            suggested_fixes: Vec::new(),
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
    envelope.llm_budget = serde_json::to_value(&report).ok();
}

/// Return a compact remediation template for major diagnostic namespaces.
pub fn explain_code(code: &str) -> Option<String> {
    if code.starts_with(codes::MPO_PREFIX) {
        return Some(
            "Ownership remediation: avoid using values after move, shorten borrow lifetimes, and clone only when sharing is required.".to_string(),
        );
    }
    if code.starts_with(codes::MPT_PREFIX) {
        return Some(
            "Type remediation: align declared and inferred types, satisfy required trait bounds, and remove recursive/value-layout mismatches.".to_string(),
        );
    }
    if code.starts_with(codes::MPS_PREFIX) {
        return Some(
            "SSA remediation: ensure each value is defined before use, phi inputs match predecessor blocks, and control-flow edges stay structurally valid.".to_string(),
        );
    }
    None
}
