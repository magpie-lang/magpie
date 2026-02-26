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
