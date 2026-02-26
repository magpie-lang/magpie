//! Magpie compiler driver (§5.2, §22, §26.1).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use magpie_ast::{AstFile, FileId};
use magpie_csnf::{format_csnf, update_digest};
use magpie_diag::{Diagnostic, DiagnosticBag, OutputEnvelope, Severity};
use magpie_hir::{verify_hir, HirModule};
use magpie_lex::lex;
use magpie_parse::parse_file;
use magpie_sema::{lower_to_hir, resolve_modules};
use magpie_types::TypeCtx;
use serde::{Deserialize, Serialize};
use serde_json::json;

const DEFAULT_MAX_ERRORS: usize = 20;

const STAGE_1: &str = "stage1_read_lex_parse";
const STAGE_2: &str = "stage2_resolve";
const STAGE_3: &str = "stage3_typecheck";
const STAGE_4: &str = "stage4_verify_hir";
const STAGE_5: &str = "stage5_ownership_check";
const STAGE_6: &str = "stage6_lower_mpir";
const STAGE_7: &str = "stage7_verify_mpir";
const STAGE_8: &str = "stage8_arc_insertion";
const STAGE_9: &str = "stage9_arc_optimization";
const STAGE_10: &str = "stage10_codegen";
const STAGE_11: &str = "stage11_link";

const PIPELINE_STAGES: [&str; 11] = [
    STAGE_1, STAGE_2, STAGE_3, STAGE_4, STAGE_5, STAGE_6, STAGE_7, STAGE_8, STAGE_9, STAGE_10,
    STAGE_11,
];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum BuildProfile {
    Dev,
    Release,
}

impl BuildProfile {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Dev => "dev",
            Self::Release => "release",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DriverConfig {
    pub entry_path: String,
    pub profile: BuildProfile,
    pub target_triple: String,
    pub emit: Vec<String>,
    pub max_errors: usize,
    pub llm_mode: bool,
    pub token_budget: Option<u32>,
    pub shared_generics: bool,
    pub features: Vec<String>,
}

impl Default for DriverConfig {
    fn default() -> Self {
        Self {
            entry_path: "src/main.mp".to_string(),
            profile: BuildProfile::Dev,
            target_triple: default_target_triple(),
            emit: vec!["exe".to_string()],
            max_errors: DEFAULT_MAX_ERRORS,
            llm_mode: false,
            token_budget: None,
            shared_generics: false,
            features: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BuildResult {
    pub success: bool,
    pub diagnostics: Vec<Diagnostic>,
    pub artifacts: Vec<String>,
    pub timing_ms: HashMap<String, u64>,
}

/// Build JSON output envelope per §26.1.
pub fn json_output_envelope(
    command: &str,
    config: &DriverConfig,
    result: &BuildResult,
) -> OutputEnvelope {
    OutputEnvelope {
        magpie_version: env!("CARGO_PKG_VERSION").to_string(),
        command: command.to_string(),
        target: Some(config.target_triple.clone()),
        success: result.success,
        artifacts: result.artifacts.clone(),
        diagnostics: result.diagnostics.clone(),
        timing_ms: serde_json::to_value(&result.timing_ms).unwrap_or_else(|_| json!({})),
        llm_budget: llm_budget_value(config),
    }
}

/// Full compilation pipeline (§22.1).
pub fn build(config: &DriverConfig) -> BuildResult {
    let max_errors = config.max_errors.max(1);
    let mut result = BuildResult::default();

    // Stage 1: read + lex + parse + in-memory CSNF canonicalization.
    let mut ast_files: Vec<AstFile> = Vec::new();
    {
        let start = Instant::now();
        let mut diag = DiagnosticBag::new(max_errors);
        let source = match fs::read_to_string(&config.entry_path) {
            Ok(source) => Some(source),
            Err(err) => {
                emit_driver_diag(
                    &mut diag,
                    "MPP0001",
                    Severity::Error,
                    "failed to read source file",
                    format!("Could not read '{}': {}", config.entry_path, err),
                );
                None
            }
        };

        if let Some(source) = source {
            let file_id = FileId(0);
            let tokens = lex(file_id, &source, &mut diag);
            if let Ok(ast) = parse_file(&tokens, file_id, &mut diag) {
                let canonical = format_csnf(&ast);
                let _canonical_with_digest = update_digest(&canonical);
                ast_files.push(ast);
            }
        }

        result.timing_ms.insert(STAGE_1.to_string(), elapsed_ms(start));
        let stage_failed = append_stage_diagnostics(&mut result, diag);
        if stage_failed {
            mark_skipped_from(&mut result.timing_ms, 1);
            return finalize_build_result(result, config);
        }
    }

    // Stage 2: module resolution.
    let resolved_modules = {
        let start = Instant::now();
        let mut diag = DiagnosticBag::new(max_errors);
        let resolved = match resolve_modules(&ast_files, &mut diag) {
            Ok(resolved) => Some(resolved),
            Err(()) => {
                if !diag.has_errors() {
                    emit_driver_diag(
                        &mut diag,
                        "MPS0000",
                        Severity::Error,
                        "resolve failed",
                        "Module resolution failed without diagnostics.",
                    );
                }
                None
            }
        };

        result.timing_ms.insert(STAGE_2.to_string(), elapsed_ms(start));
        let stage_failed = append_stage_diagnostics(&mut result, diag);
        if stage_failed {
            mark_skipped_from(&mut result.timing_ms, 2);
            return finalize_build_result(result, config);
        }
        resolved.unwrap_or_default()
    };

    // Stage 3: typecheck placeholder (use sema lowering if available).
    let mut type_ctx = TypeCtx::new();
    let mut hir_modules: Vec<HirModule> = Vec::new();
    {
        let start = Instant::now();
        let mut diag = DiagnosticBag::new(max_errors);
        for module in &resolved_modules {
            if let Ok(hir) = lower_to_hir(module, &mut type_ctx, &mut diag) {
                hir_modules.push(hir);
            }
        }
        result.timing_ms.insert(STAGE_3.to_string(), elapsed_ms(start));
        let stage_failed = append_stage_diagnostics(&mut result, diag);
        if stage_failed {
            mark_skipped_from(&mut result.timing_ms, 3);
            return finalize_build_result(result, config);
        }
    }

    // Stage 4: verify HIR.
    {
        let start = Instant::now();
        let mut diag = DiagnosticBag::new(max_errors);
        for module in &hir_modules {
            let _ = verify_hir(module, &type_ctx, &mut diag);
        }
        result.timing_ms.insert(STAGE_4.to_string(), elapsed_ms(start));
        let stage_failed = append_stage_diagnostics(&mut result, diag);
        if stage_failed {
            mark_skipped_from(&mut result.timing_ms, 4);
            return finalize_build_result(result, config);
        }
    }

    // Stages 5-11: placeholders.
    run_placeholder_stage(&mut result.timing_ms, STAGE_5);
    run_placeholder_stage(&mut result.timing_ms, STAGE_6);
    run_placeholder_stage(&mut result.timing_ms, STAGE_7);
    run_placeholder_stage(&mut result.timing_ms, STAGE_8);
    run_placeholder_stage(&mut result.timing_ms, STAGE_9);
    run_placeholder_stage(&mut result.timing_ms, STAGE_10);
    run_placeholder_stage(&mut result.timing_ms, STAGE_11);

    finalize_build_result(result, config)
}

/// `magpie fmt` entry-point: lex + parse + CSNF format + write back.
pub fn format_files(paths: &[String], fix_meta: bool) -> BuildResult {
    let mut result = BuildResult::default();
    let mut stage_read_lex_parse = 0_u64;
    let mut stage_csnf_format = 0_u64;
    let mut stage_write_back = 0_u64;

    for path in paths {
        let mut diag = DiagnosticBag::new(DEFAULT_MAX_ERRORS);

        let stage1_start = Instant::now();
        let source = match fs::read_to_string(path) {
            Ok(source) => Some(source),
            Err(err) => {
                emit_driver_diag(
                    &mut diag,
                    "MPP0001",
                    Severity::Error,
                    "failed to read source file",
                    format!("Could not read '{}': {}", path, err),
                );
                None
            }
        };

        let mut parsed: Option<AstFile> = None;
        if let Some(source) = source {
            let file_id = FileId(0);
            let tokens = lex(file_id, &source, &mut diag);
            if let Ok(ast) = parse_file(&tokens, file_id, &mut diag) {
                parsed = Some(ast);
            }
        }
        stage_read_lex_parse += elapsed_ms(stage1_start);

        let stage_failed = append_stage_diagnostics(&mut result, diag);
        if stage_failed {
            continue;
        }

        let Some(ast) = parsed else {
            continue;
        };

        let stage2_start = Instant::now();
        let mut formatted = format_csnf(&ast);
        formatted = update_digest(&formatted);
        if fix_meta {
            // Placeholder: meta synthesis is not implemented yet.
        }
        stage_csnf_format += elapsed_ms(stage2_start);

        let stage3_start = Instant::now();
        match fs::write(path, formatted) {
            Ok(()) => result.artifacts.push(path.clone()),
            Err(err) => {
                result.diagnostics.push(simple_diag(
                    "MPP0001",
                    Severity::Error,
                    "failed to write source file",
                    format!("Could not write '{}': {}", path, err),
                ));
            }
        }
        stage_write_back += elapsed_ms(stage3_start);
    }

    result
        .timing_ms
        .insert(STAGE_1.to_string(), stage_read_lex_parse);
    result
        .timing_ms
        .insert("stage2_csnf_format".to_string(), stage_csnf_format);
    result
        .timing_ms
        .insert("stage3_write_back".to_string(), stage_write_back);
    result.success = !has_errors(&result.diagnostics);
    result
}

/// `magpie new <name>` scaffolding per §5.2.1.
pub fn create_project(name: &str) -> Result<(), String> {
    if name.trim().is_empty() {
        return Err("Project name must not be empty.".to_string());
    }

    let base = Path::new(name);
    fs::create_dir_all(base.join("src"))
        .map_err(|e| format!("Failed to create '{}': {}", base.join("src").display(), e))?;
    fs::create_dir_all(base.join("tests"))
        .map_err(|e| format!("Failed to create '{}': {}", base.join("tests").display(), e))?;
    fs::create_dir_all(base.join(".magpie"))
        .map_err(|e| format!("Failed to create '{}': {}", base.join(".magpie").display(), e))?;

    let manifest = format!(
        r#"[package]
name = "{name}"
version = "0.1.0"
edition = "2026"

[build]
entry = "src/main.mp"
profile_default = "dev"

[dependencies]
std = {{ version = "^0.1" }}

[llm]
mode_default = true
token_budget = 12000
tokenizer = "approx:utf8_4chars"
budget_policy = "balanced"
"#
    );
    fs::write(base.join("Magpie.toml"), manifest)
        .map_err(|e| format!("Failed to write Magpie.toml: {}", e))?;

    let main_source = format!(
        r#"module {name}.main
exports {{ @main }}
imports {{ }}
digest ""

fn @main() -> i32 {{
bb0:
  ret const.i32 0
}}
"#
    );
    let main_source = update_digest(&main_source);
    fs::write(base.join("src/main.mp"), main_source)
        .map_err(|e| format!("Failed to write src/main.mp: {}", e))?;

    fs::write(base.join("Magpie.lock"), "")
        .map_err(|e| format!("Failed to write Magpie.lock: {}", e))?;

    Ok(())
}

fn finalize_build_result(mut result: BuildResult, config: &DriverConfig) -> BuildResult {
    result.success = !has_errors(&result.diagnostics);
    if result.success {
        result.artifacts = planned_artifacts(config, &mut result.diagnostics);
    }
    result
}

fn append_stage_diagnostics(result: &mut BuildResult, bag: DiagnosticBag) -> bool {
    let failed = bag.has_errors();
    result.diagnostics.extend(bag.diagnostics);
    failed
}

fn mark_skipped_from(timing: &mut HashMap<String, u64>, from_stage_idx: usize) {
    for stage in PIPELINE_STAGES.iter().skip(from_stage_idx) {
        timing.entry((*stage).to_string()).or_insert(0);
    }
}

fn run_placeholder_stage(timing: &mut HashMap<String, u64>, stage: &str) {
    let start = Instant::now();
    timing.insert(stage.to_string(), elapsed_ms(start));
}

fn elapsed_ms(start: Instant) -> u64 {
    start.elapsed().as_millis().try_into().unwrap_or(u64::MAX)
}

fn has_errors(diags: &[Diagnostic]) -> bool {
    diags.iter().any(|d| matches!(d.severity, Severity::Error))
}

fn simple_diag(code: &str, severity: Severity, title: impl Into<String>, message: impl Into<String>) -> Diagnostic {
    Diagnostic {
        code: code.to_string(),
        severity,
        title: title.into(),
        primary_span: None,
        secondary_spans: Vec::new(),
        message: message.into(),
        explanation_md: None,
        why: None,
        suggested_fixes: Vec::new(),
    }
}

fn emit_driver_diag(
    bag: &mut DiagnosticBag,
    code: &str,
    severity: Severity,
    title: impl Into<String>,
    message: impl Into<String>,
) {
    bag.emit(simple_diag(code, severity, title, message));
}

fn planned_artifacts(config: &DriverConfig, diagnostics: &mut Vec<Diagnostic>) -> Vec<String> {
    let mut out = Vec::new();
    let stem = Path::new(&config.entry_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("main");
    let base = PathBuf::from("target")
        .join(&config.target_triple)
        .join(config.profile.as_str());
    let is_windows = config.target_triple.contains("windows");
    let is_darwin = config.target_triple.contains("apple-darwin");

    for emit in &config.emit {
        let path = match emit.as_str() {
            "llvm-ir" => Some(base.join(format!("{stem}.ll"))),
            "llvm-bc" => Some(base.join(format!("{stem}.bc"))),
            "object" => Some(base.join(format!("{stem}{}", if is_windows { ".obj" } else { ".o" }))),
            "asm" => Some(base.join(format!("{stem}.s"))),
            "spv" => Some(base.join(format!("{stem}.spv"))),
            "exe" => Some(base.join(format!("{stem}{}", if is_windows { ".exe" } else { "" }))),
            "shared-lib" => {
                let ext = if is_windows {
                    ".dll"
                } else if is_darwin {
                    ".dylib"
                } else {
                    ".so"
                };
                Some(base.join(format!("lib{stem}{ext}")))
            }
            "mpir" => Some(base.join(format!("{stem}.mpir"))),
            "mpd" => Some(base.join(format!("{stem}.mpd"))),
            "symgraph" => Some(base.join(format!("{stem}.symgraph.json"))),
            _ => {
                diagnostics.push(simple_diag(
                    "MPL0001",
                    Severity::Warning,
                    "unknown emit kind",
                    format!("Unknown emit kind '{}'; skipping.", emit),
                ));
                None
            }
        };

        if let Some(path) = path {
            let path = path.to_string_lossy().to_string();
            if !out.contains(&path) {
                out.push(path);
            }
        }
    }

    out
}

fn llm_budget_value(config: &DriverConfig) -> Option<serde_json::Value> {
    if !config.llm_mode && config.token_budget.is_none() {
        return None;
    }

    Some(json!({
        "token_budget": config.token_budget.unwrap_or(12000),
        "tokenizer": "approx:utf8_4chars",
        "estimated_tokens": serde_json::Value::Null,
        "policy": "balanced",
        "dropped": [],
    }))
}

fn default_target_triple() -> String {
    format!(
        "{}-unknown-{}",
        std::env::consts::ARCH,
        std::env::consts::OS
    )
}
