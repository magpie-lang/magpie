//! Magpie compiler driver (§5.2, §22, §26.1).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use magpie_arc::{insert_arc_ops, optimize_arc};
use magpie_ast::{
    AstBaseType, AstBuiltinType, AstDecl, AstFile, AstFnDecl, AstFnMeta, AstInstr, AstOp,
    AstOpVoid, AstType, ExportItem, FileId,
};
use magpie_csnf::{format_csnf, update_digest};
use magpie_diag::{codes, Diagnostic, DiagnosticBag, OutputEnvelope, Severity, SuggestedFix};
use magpie_hir::{
    verify_hir, BlockId, HirBlock, HirConst, HirConstLit, HirFunction, HirInstr, HirModule, HirOp,
    HirOpVoid, HirTerminator, HirValue, LocalId,
};
use magpie_lex::lex;
use magpie_memory::{build_index, MmsItem};
use magpie_mpir::{
    print_mpir, verify_mpir, MpirBlock, MpirFn, MpirInstr, MpirLocalDecl, MpirModule, MpirOp,
    MpirOpVoid, MpirTerminator, MpirTypeTable, MpirValue,
};
use magpie_own::check_ownership;
use magpie_parse::parse_file;
use magpie_sema::{generate_sid, lower_to_hir, resolve_modules};
use magpie_types::{fixed_type_ids, Sid, TypeCtx};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[path = "../../magpie_codegen_llvm/src/lib.rs"]
mod magpie_codegen_llvm;

const DEFAULT_MAX_ERRORS: usize = 20;

const STAGE_1: &str = "stage1_read_lex_parse";
const STAGE_2: &str = "stage2_resolve";
const STAGE_3: &str = "stage3_typecheck";
const STAGE_35: &str = "stage3_5_async_lowering";
const STAGE_4: &str = "stage4_verify_hir";
const STAGE_5: &str = "stage5_ownership_check";
const STAGE_6: &str = "stage6_lower_mpir";
const STAGE_7: &str = "stage7_verify_mpir";
const STAGE_8: &str = "stage8_arc_insertion";
const STAGE_9: &str = "stage9_arc_optimization";
const STAGE_10: &str = "stage10_codegen";
const STAGE_11: &str = "stage11_link";
const STAGE_12: &str = "stage12_mms_update";

const PIPELINE_STAGES: [&str; 13] = [
    STAGE_1, STAGE_2, STAGE_3, STAGE_35, STAGE_4, STAGE_5, STAGE_6, STAGE_7, STAGE_8, STAGE_9,
    STAGE_10, STAGE_11, STAGE_12,
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

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TestResult {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub test_names: Vec<(String, bool)>,
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

/// Explain a diagnostic code using shared templates.
pub fn explain_code(code: &str) -> Option<String> {
    magpie_diag::explain_code(code)
}

/// Import a C header and generate a minimal `extern "C"` Magpie module.
pub fn import_c_header(header_path: &str, out_path: &str) -> BuildResult {
    let mut result = BuildResult::default();
    let source = match fs::read_to_string(header_path) {
        Ok(source) => source,
        Err(err) => {
            result.diagnostics.push(simple_diag(
                "MPF1000",
                Severity::Error,
                "failed to read C header",
                format!("Could not read '{}': {}", header_path, err),
            ));
            return result;
        }
    };

    let extern_items = parse_c_header_functions(&source);
    if extern_items.is_empty() {
        result.diagnostics.push(simple_diag(
            "MPF1001",
            Severity::Warning,
            "no C functions detected",
            "No supported function declarations were found in the header.",
        ));
    }

    let payload = render_extern_module("ffi_import", &extern_items);
    let out_path = PathBuf::from(out_path);
    match write_text_artifact(&out_path, &payload) {
        Ok(()) => {
            result.success = true;
            result.artifacts.push(out_path.to_string_lossy().to_string());
            result.diagnostics.push(simple_diag(
                "MPF1002",
                Severity::Info,
                "ffi import generated",
                format!(
                    "Generated {} extern declarations from '{}'.",
                    extern_items.len(),
                    header_path
                ),
            ));
        }
        Err(err) => {
            result.diagnostics.push(simple_diag(
                "MPF1003",
                Severity::Error,
                "failed to write ffi output",
                err,
            ));
        }
    }

    result
}

/// Emit one of the compiler graph payloads (`symbols`, `deps`, `ownership`, `cfg`) as JSON.
pub fn emit_graph(kind: &str, modules: &[HirModule], type_ctx: &TypeCtx) -> String {
    let normalized = kind.trim().to_ascii_lowercase();
    let payload = match normalized.as_str() {
        "symbols" | "symgraph" => emit_symbols_graph(modules),
        "deps" | "depsgraph" => emit_deps_graph(modules),
        "ownership" | "ownershipgraph" => emit_ownership_graph(modules, type_ctx),
        "cfg" | "cfggraph" => emit_cfg_graph(modules),
        _ => json!({
            "graph": normalized,
            "error": "unknown_graph_kind",
            "supported": ["symbols", "deps", "ownership", "cfg"],
        }),
    };
    serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "{}".to_string())
}

/// Full compilation pipeline (§22.1).
pub fn build(config: &DriverConfig) -> BuildResult {
    let max_errors = config.max_errors.max(1);
    let mut result = BuildResult::default();

    if config.shared_generics {
        let mut diag = DiagnosticBag::new(max_errors);
        emit_driver_diag(
            &mut diag,
            codes::MPL2021,
            Severity::Error,
            "shared generics unsupported",
            "shared generics (vtable mode) is not yet implemented in v0.1",
        );
        mark_skipped_from(&mut result.timing_ms, 0);
        let _ = append_stage_diagnostics(&mut result, diag);
        return finalize_build_result(result, config);
    }

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

        result
            .timing_ms
            .insert(STAGE_1.to_string(), elapsed_ms(start));
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

        result
            .timing_ms
            .insert(STAGE_2.to_string(), elapsed_ms(start));
        let stage_failed = append_stage_diagnostics(&mut result, diag);
        if stage_failed {
            mark_skipped_from(&mut result.timing_ms, 2);
            return finalize_build_result(result, config);
        }
        resolved.unwrap_or_default()
    };

    // Stage 3: typecheck.
    // Per §22.1 stage 3, type checking is performed during AST -> HIR lowering in sema.
    let mut type_ctx = TypeCtx::new();
    let mut hir_modules: Vec<HirModule> = Vec::new();
    {
        let start = Instant::now();
        let mut diag = DiagnosticBag::new(max_errors);
        for module in &resolved_modules {
            match lower_to_hir(module, &mut type_ctx, &mut diag) {
                Ok(hir) => hir_modules.push(hir),
                Err(()) => {
                    if !diag.has_errors() {
                        emit_driver_diag(
                            &mut diag,
                            "MPT0000",
                            Severity::Error,
                            "typecheck failed",
                            format!(
                                "Type checking failed for module '{}' without diagnostics.",
                                module.path
                            ),
                        );
                    }
                }
            }
        }
        result
            .timing_ms
            .insert(STAGE_3.to_string(), elapsed_ms(start));
        let stage_failed = append_stage_diagnostics(&mut result, diag);
        if stage_failed {
            mark_skipped_from(&mut result.timing_ms, 3);
            return finalize_build_result(result, config);
        }
    }

    // Stage 3.5: async lowering.
    {
        let start = Instant::now();
        lower_async_functions(&mut hir_modules, &mut type_ctx);
        result
            .timing_ms
            .insert(STAGE_35.to_string(), elapsed_ms(start));
    }

    // Stage 4: verify HIR.
    {
        let start = Instant::now();
        let mut diag = DiagnosticBag::new(max_errors);
        for module in &hir_modules {
            let _ = verify_hir(module, &type_ctx, &mut diag);
        }
        result
            .timing_ms
            .insert(STAGE_4.to_string(), elapsed_ms(start));
        let stage_failed = append_stage_diagnostics(&mut result, diag);
        if stage_failed {
            mark_skipped_from(&mut result.timing_ms, 5);
            return finalize_build_result(result, config);
        }
    }

    // Stage 5: ownership checking.
    {
        let start = Instant::now();
        let mut diag = DiagnosticBag::new(max_errors);
        for module in &hir_modules {
            let _ = check_ownership(module, &type_ctx, &mut diag);
        }
        result
            .timing_ms
            .insert(STAGE_5.to_string(), elapsed_ms(start));
        let stage_failed = append_stage_diagnostics(&mut result, diag);
        if stage_failed {
            mark_skipped_from(&mut result.timing_ms, 6);
            return finalize_build_result(result, config);
        }
    }

    // Optional graph emission from verified HIR.
    {
        let mut diag = DiagnosticBag::new(max_errors);
        for (emit_kind, graph_kind, file_suffix) in [
            ("symgraph", "symbols", "symgraph"),
            ("depsgraph", "deps", "depsgraph"),
            ("ownershipgraph", "ownership", "ownershipgraph"),
            ("cfggraph", "cfg", "cfggraph"),
        ] {
            if !emit_contains(&config.emit, emit_kind) {
                continue;
            }
            let graph_path = stage_graph_output_path(config, file_suffix);
            let payload = emit_graph(graph_kind, &hir_modules, &type_ctx);
            match write_text_artifact(&graph_path, &payload) {
                Ok(()) => {
                    let graph_path = graph_path.to_string_lossy().to_string();
                    if !result.artifacts.contains(&graph_path) {
                        result.artifacts.push(graph_path);
                    }
                }
                Err(err) => {
                    emit_driver_diag(
                        &mut diag,
                        "MPP0003",
                        Severity::Error,
                        "failed to write graph artifact",
                        err,
                    );
                }
            }
        }
        let stage_failed = append_stage_diagnostics(&mut result, diag);
        if stage_failed {
            mark_skipped_from(&mut result.timing_ms, 6);
            return finalize_build_result(result, config);
        }
    }

    // Optional `.mpdbg` debug sidecar (G60 stub).
    {
        let mut diag = DiagnosticBag::new(max_errors);
        if emit_contains(&config.emit, "mpdbg") {
            // TODO(G60): replace placeholder payload with real per-pass debug trace schema.
            let mpdbg_path = stage_mpdbg_output_path(config);
            let payload = serde_json::to_string_pretty(&json!({
                "format": "mpdbg.v0.stub",
                "entry_path": config.entry_path,
                "note": "TODO(G60): wire full debug-event emission.",
            }))
            .unwrap_or_else(|_| "{}".to_string());
            match write_text_artifact(&mpdbg_path, &payload) {
                Ok(()) => {
                    let mpdbg_path = mpdbg_path.to_string_lossy().to_string();
                    if !result.artifacts.contains(&mpdbg_path) {
                        result.artifacts.push(mpdbg_path);
                    }
                }
                Err(err) => emit_driver_diag(
                    &mut diag,
                    "MPP0003",
                    Severity::Error,
                    "failed to write mpdbg artifact",
                    err,
                ),
            }
        }
        let stage_failed = append_stage_diagnostics(&mut result, diag);
        if stage_failed {
            mark_skipped_from(&mut result.timing_ms, 6);
            return finalize_build_result(result, config);
        }
    }

    // Stage 6: lower HIR to MPIR.
    let mut mpir_modules: Vec<MpirModule> = Vec::new();
    {
        let start = Instant::now();
        let mut diag = DiagnosticBag::new(max_errors);
        for module in &hir_modules {
            mpir_modules.push(lower_hir_module_to_mpir(module, &type_ctx));
        }
        if mpir_modules.is_empty() {
            emit_driver_diag(
                &mut diag,
                "MPM0001",
                Severity::Error,
                "mpir lowering produced no modules",
                "Expected at least one lowered MPIR module.",
            );
        }
        result
            .timing_ms
            .insert(STAGE_6.to_string(), elapsed_ms(start));
        let stage_failed = append_stage_diagnostics(&mut result, diag);
        if stage_failed {
            mark_skipped_from(&mut result.timing_ms, 6);
            return finalize_build_result(result, config);
        }
    }

    // Stage 7: verify MPIR and optionally emit textual MPIR.
    {
        let start = Instant::now();
        let mut diag = DiagnosticBag::new(max_errors);
        let module_count = mpir_modules.len();
        for (idx, module) in mpir_modules.iter().enumerate() {
            let _ = verify_mpir(module, &type_ctx, &mut diag);
            if emit_contains(&config.emit, "mpir") {
                let mpir_path = stage_module_output_path(config, idx, module_count, "mpir");
                if let Err(err) = write_text_artifact(&mpir_path, &print_mpir(module, &type_ctx)) {
                    emit_driver_diag(
                        &mut diag,
                        "MPP0003",
                        Severity::Error,
                        "failed to write mpir artifact",
                        err,
                    );
                }
            }
        }
        result
            .timing_ms
            .insert(STAGE_7.to_string(), elapsed_ms(start));
        let stage_failed = append_stage_diagnostics(&mut result, diag);
        if stage_failed {
            mark_skipped_from(&mut result.timing_ms, 7);
            return finalize_build_result(result, config);
        }
    }

    // Stage 8: ARC insertion.
    {
        let start = Instant::now();
        let mut diag = DiagnosticBag::new(max_errors);
        for module in &mut mpir_modules {
            let _ = insert_arc_ops(module, &type_ctx, &mut diag);
        }
        result
            .timing_ms
            .insert(STAGE_8.to_string(), elapsed_ms(start));
        let stage_failed = append_stage_diagnostics(&mut result, diag);
        if stage_failed {
            mark_skipped_from(&mut result.timing_ms, 8);
            return finalize_build_result(result, config);
        }
    }

    // Stage 9: ARC peephole optimization.
    {
        let start = Instant::now();
        for module in &mut mpir_modules {
            optimize_arc(module, &type_ctx);
        }
        result
            .timing_ms
            .insert(STAGE_9.to_string(), elapsed_ms(start));
    }

    // Stage 10: LLVM codegen + write .ll.
    let mut llvm_ir_paths: Vec<PathBuf> = Vec::new();
    {
        let start = Instant::now();
        let mut diag = DiagnosticBag::new(max_errors);
        let module_count = mpir_modules.len();
        for (idx, module) in mpir_modules.iter().enumerate() {
            match magpie_codegen_llvm::codegen_module(module, &type_ctx) {
                Ok(llvm_ir) => {
                    let llvm_path = stage_module_output_path(config, idx, module_count, "ll");
                    if let Err(err) = write_text_artifact(&llvm_path, &llvm_ir) {
                        emit_driver_diag(
                            &mut diag,
                            "MPP0003",
                            Severity::Error,
                            "failed to write llvm ir artifact",
                            err,
                        );
                    } else {
                        llvm_ir_paths.push(llvm_path);
                    }
                }
                Err(err) => {
                    emit_driver_diag(
                        &mut diag,
                        "MPG0001",
                        Severity::Error,
                        "llvm codegen failed",
                        err,
                    );
                }
            }
        }
        result
            .timing_ms
            .insert(STAGE_10.to_string(), elapsed_ms(start));
        let stage_failed = append_stage_diagnostics(&mut result, diag);
        if stage_failed {
            mark_skipped_from(&mut result.timing_ms, 10);
            return finalize_build_result(result, config);
        }
    }

    // Stage 11: native linking.
    {
        let start = Instant::now();
        let mut diag = DiagnosticBag::new(max_errors);
        if emit_contains_any(&config.emit, &["exe", "shared-lib", "object", "asm"]) {
            let output_path = stage_link_output_path(config);
            let link_shared =
                emit_contains(&config.emit, "shared-lib") && !emit_contains(&config.emit, "exe");
            match link_via_llc_and_linker(config, &llvm_ir_paths, &output_path, link_shared) {
                Ok(object_paths) => {
                    if emit_contains(&config.emit, "object") {
                        for object in object_paths {
                            let object = object.to_string_lossy().to_string();
                            if !result.artifacts.contains(&object) {
                                result.artifacts.push(object);
                            }
                        }
                    }
                    let output = output_path.to_string_lossy().to_string();
                    if !result.artifacts.contains(&output) {
                        result.artifacts.push(output);
                    }
                }
                Err(primary_err) => {
                    emit_driver_diag(
                        &mut diag,
                        "MPLINK01",
                        Severity::Warning,
                        "native link fallback",
                        format!(
                            "llc + cc/clang link failed; trying clang -x ir fallback. Reason: {primary_err}"
                        ),
                    );
                    match link_via_clang_ir(config, &llvm_ir_paths, &output_path, link_shared) {
                        Ok(()) => {
                            let output = output_path.to_string_lossy().to_string();
                            if !result.artifacts.contains(&output) {
                                result.artifacts.push(output);
                            }
                        }
                        Err(fallback_err) => {
                            let outputs = llvm_ir_paths
                                .iter()
                                .map(|p| p.to_string_lossy().to_string())
                                .collect::<Vec<_>>();
                            emit_driver_diag(
                                &mut diag,
                                "MPLINK02",
                                Severity::Warning,
                                "native linking unavailable",
                                format!(
                                    "Could not produce native output; keeping LLVM IR artifacts [{}]. llc/cc failure: {}. clang -x ir failure: {}.",
                                    outputs.join(", "),
                                    primary_err,
                                    fallback_err
                                ),
                            );
                            for path in outputs {
                                if !result.artifacts.contains(&path) {
                                    result.artifacts.push(path);
                                }
                            }
                        }
                    }
                }
            }
        }
        result
            .timing_ms
            .insert(STAGE_11.to_string(), elapsed_ms(start));
        let stage_failed = append_stage_diagnostics(&mut result, diag);
        if stage_failed {
            mark_skipped_from(&mut result.timing_ms, 11);
            return finalize_build_result(result, config);
        }
    }

    // Stage 12: MMS index update from build artifacts.
    {
        let start = Instant::now();
        let mut diag = DiagnosticBag::new(max_errors);
        let mms_artifacts =
            collect_mms_artifact_paths(config, &result.artifacts, &llvm_ir_paths, &mpir_modules);
        if let Some(index_path) =
            update_mms_index(config, &mms_artifacts, &hir_modules, &type_ctx, &mut diag)
        {
            let index_path = index_path.to_string_lossy().to_string();
            if !result.artifacts.contains(&index_path) {
                result.artifacts.push(index_path);
            }
        }
        result
            .timing_ms
            .insert(STAGE_12.to_string(), elapsed_ms(start));
        let _ = append_stage_diagnostics(&mut result, diag);
    }

    finalize_build_result(result, config)
}

/// Lint entrypoint (`magpie lint`).
pub fn lint(config: &DriverConfig) -> BuildResult {
    let max_errors = config.max_errors.max(1);
    let mut result = BuildResult::default();

    let mut ast_files: Vec<AstFile> = Vec::new();
    {
        let start = Instant::now();
        let mut diag = DiagnosticBag::new(max_errors);
        let source_paths = collect_lint_source_paths(&config.entry_path);
        for (idx, path) in source_paths.iter().enumerate() {
            let source = match fs::read_to_string(path) {
                Ok(source) => source,
                Err(err) => {
                    emit_driver_diag(
                        &mut diag,
                        "MPP0001",
                        Severity::Error,
                        "failed to read source file",
                        format!("Could not read '{}': {}", path.display(), err),
                    );
                    continue;
                }
            };
            let file_id = FileId(idx as u32);
            let tokens = lex(file_id, &source, &mut diag);
            if let Ok(ast) = parse_file(&tokens, file_id, &mut diag) {
                ast_files.push(ast);
            }
        }
        result
            .timing_ms
            .insert("lint_stage1_read_lex_parse".to_string(), elapsed_ms(start));
        let stage_failed = append_stage_diagnostics(&mut result, diag);
        if stage_failed {
            result.success = false;
            return result;
        }
    }

    if ast_files.is_empty() {
        result.diagnostics.push(simple_diag(
            "MPP0001",
            Severity::Error,
            "no source files found",
            "No .mp source files were found to lint.",
        ));
        result.success = false;
        return result;
    }

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
        result
            .timing_ms
            .insert("lint_stage2_resolve".to_string(), elapsed_ms(start));
        let stage_failed = append_stage_diagnostics(&mut result, diag);
        if stage_failed {
            result.success = false;
            return result;
        }
        resolved.unwrap_or_default()
    };

    let mut type_ctx = TypeCtx::new();
    let mut hir_modules: Vec<HirModule> = Vec::new();
    {
        let start = Instant::now();
        let mut diag = DiagnosticBag::new(max_errors);
        for module in &resolved_modules {
            match lower_to_hir(module, &mut type_ctx, &mut diag) {
                Ok(hir) => hir_modules.push(hir),
                Err(()) => {
                    if !diag.has_errors() {
                        emit_driver_diag(
                            &mut diag,
                            "MPT0000",
                            Severity::Error,
                            "typecheck failed",
                            format!(
                                "Type checking failed for module '{}' without diagnostics.",
                                module.path
                            ),
                        );
                    }
                }
            }
        }
        result
            .timing_ms
            .insert("lint_stage3_typecheck".to_string(), elapsed_ms(start));
        let stage_failed = append_stage_diagnostics(&mut result, diag);
        if stage_failed {
            result.success = false;
            return result;
        }
    }

    {
        let start = Instant::now();
        result.diagnostics.extend(run_lints(&hir_modules, &type_ctx));
        result
            .timing_ms
            .insert("lint_stage4_lints".to_string(), elapsed_ms(start));
    }

    result.success = !has_errors(&result.diagnostics);
    result
}

/// Run lint checks over lowered HIR modules.
pub fn run_lints(modules: &[HirModule], type_ctx: &TypeCtx) -> Vec<Diagnostic> {
    let _ = type_ctx;
    let mut diagnostics = Vec::new();

    let mut called_sids: HashSet<String> = HashSet::new();
    for module in modules {
        for func in &module.functions {
            for block in &func.blocks {
                for instr in &block.instrs {
                    match &instr.op {
                        HirOp::Call { callee_sid, .. } | HirOp::SuspendCall { callee_sid, .. } => {
                            called_sids.insert(callee_sid.0.clone());
                        }
                        _ => {}
                    }
                }
                for vop in &block.void_ops {
                    if let HirOpVoid::CallVoid { callee_sid, .. } = vop {
                        called_sids.insert(callee_sid.0.clone());
                    }
                }
            }
        }
    }

    for module in modules {
        let source_path = module_source_path(&module.path);
        for func in &module.functions {
            if !called_sids.contains(&func.sid.0) && !is_lint_entry_function(&func.name) {
                diagnostics.push(lint_diag(
                    codes::MPL2002,
                    "unused function",
                    format!(
                        "Function '{}' in module '{}' is never called.",
                        func.name, module.path
                    ),
                    Vec::new(),
                ));
            }

            let mut defined_locals: HashSet<u32> = HashSet::new();
            let mut used_locals: HashSet<u32> = HashSet::new();

            for (local, _) in &func.params {
                defined_locals.insert(local.0);
            }
            for block in &func.blocks {
                for instr in &block.instrs {
                    defined_locals.insert(instr.dst.0);
                    for value in hir_op_values(&instr.op) {
                        if let HirValue::Local(local) = value {
                            used_locals.insert(local.0);
                        }
                    }
                }
                for vop in &block.void_ops {
                    for value in hir_op_void_values(vop) {
                        if let HirValue::Local(local) = value {
                            used_locals.insert(local.0);
                        }
                    }
                }
                for value in hir_terminator_values(&block.terminator) {
                    if let HirValue::Local(local) = value {
                        used_locals.insert(local.0);
                    }
                }
            }

            let mut unused_locals = defined_locals
                .difference(&used_locals)
                .copied()
                .collect::<Vec<_>>();
            unused_locals.sort_unstable();
            for local_id in unused_locals {
                diagnostics.push(lint_diag(
                    codes::MPL2001,
                    "unused variable",
                    format!(
                        "Local '{}' in function '{}' is defined but never used.",
                        local_name(local_id),
                        func.name
                    ),
                    vec![suggested_fix_unused_local(&source_path, local_id)],
                ));
            }

            for block in &func.blocks {
                for instr in &block.instrs {
                    let borrow_source = match &instr.op {
                        HirOp::BorrowShared { v } | HirOp::BorrowMut { v } => Some(v),
                        _ => None,
                    };
                    let Some(borrow_source) = borrow_source else {
                        continue;
                    };

                    let borrow_local = instr.dst.0;
                    let mut use_count = 0usize;
                    let mut deref_only = true;
                    for other_block in &func.blocks {
                        for other_instr in &other_block.instrs {
                            if op_uses_local(&other_instr.op, borrow_local) {
                                use_count += 1;
                                let is_getfield = matches!(
                                    other_instr.op,
                                    HirOp::GetField {
                                        obj: HirValue::Local(id),
                                        ..
                                    } if id.0 == borrow_local
                                );
                                if !is_getfield {
                                    deref_only = false;
                                }
                            }
                        }
                        for vop in &other_block.void_ops {
                            if op_void_uses_local(vop, borrow_local) {
                                use_count += 1;
                                let is_setfield = matches!(
                                    vop,
                                    HirOpVoid::SetField {
                                        obj: HirValue::Local(id),
                                        ..
                                    } if id.0 == borrow_local
                                );
                                if !is_setfield {
                                    deref_only = false;
                                }
                            }
                        }
                        if terminator_uses_local(&other_block.terminator, borrow_local) {
                            use_count += 1;
                            deref_only = false;
                        }
                    }

                    if use_count == 1 && deref_only {
                        diagnostics.push(lint_diag(
                            codes::MPL2003,
                            "unnecessary borrow",
                            format!(
                                "Borrow '{}' in function '{}' is immediately dereferenced and can be removed.",
                                local_name(borrow_local),
                                func.name
                            ),
                            vec![suggested_fix_unnecessary_borrow(
                                &source_path,
                                borrow_local,
                                &hir_value_display(borrow_source),
                            )],
                        ));
                    }
                }
            }

            for block in &func.blocks {
                if block.instrs.is_empty() && block.void_ops.is_empty() {
                    diagnostics.push(lint_diag(
                        codes::MPL2005,
                        "empty block",
                        format!(
                            "Block 'bb{}' in function '{}' contains no instructions.",
                            block.id.0, func.name
                        ),
                        vec![suggested_fix_empty_block(&source_path, block.id.0)],
                    ));
                }
            }

            let reachable = reachable_block_ids(func);
            for block in &func.blocks {
                if !reachable.contains(&block.id.0) {
                    diagnostics.push(lint_diag(
                        codes::MPL2007,
                        "unreachable code",
                        format!(
                            "Block 'bb{}' in function '{}' is unreachable (code after return/panic).",
                            block.id.0, func.name
                        ),
                        Vec::new(),
                    ));
                }

                if let Some(panic_idx) = block
                    .instrs
                    .iter()
                    .position(|instr| matches!(instr.op, HirOp::Panic { .. }))
                {
                    if panic_idx + 1 < block.instrs.len() {
                        diagnostics.push(lint_diag(
                            codes::MPL2007,
                            "unreachable code",
                            format!(
                                "Instructions after panic in block 'bb{}' of function '{}' are unreachable.",
                                block.id.0, func.name
                            ),
                            Vec::new(),
                        ));
                    }
                }
                if let Some(panic_idx) = block
                    .void_ops
                    .iter()
                    .position(|op| matches!(op, HirOpVoid::Panic { .. }))
                {
                    if panic_idx + 1 < block.void_ops.len() {
                        diagnostics.push(lint_diag(
                            codes::MPL2007,
                            "unreachable code",
                            format!(
                                "Void operations after panic in block 'bb{}' of function '{}' are unreachable.",
                                block.id.0, func.name
                            ),
                            Vec::new(),
                        ));
                    }
                }
            }
        }
    }

    diagnostics
}

fn collect_lint_source_paths(entry_path: &str) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    push_mp_path(&mut paths, Path::new(entry_path));
    collect_mp_files(Path::new("src"), &mut paths);
    paths.sort();
    paths.dedup();
    paths
}

fn lint_diag(
    code: &str,
    title: impl Into<String>,
    message: impl Into<String>,
    suggested_fixes: Vec<SuggestedFix>,
) -> Diagnostic {
    Diagnostic {
        code: code.to_string(),
        severity: Severity::Warning,
        title: title.into(),
        primary_span: None,
        secondary_spans: Vec::new(),
        message: message.into(),
        explanation_md: None,
        why: None,
        suggested_fixes,
    }
}

fn is_lint_entry_function(name: &str) -> bool {
    name == "@main" || name.starts_with("@test_")
}

fn local_name(local_id: u32) -> String {
    format!("%v{}", local_id)
}

fn module_source_path(module_path: &str) -> String {
    let parts = module_path.split('.').collect::<Vec<_>>();
    if parts.len() <= 1 {
        return format!("src/{}.mp", parts.first().copied().unwrap_or("main"));
    }
    format!("src/{}.mp", parts[1..].join("/"))
}

fn suggested_fix_unused_local(source_path: &str, local_id: u32) -> SuggestedFix {
    let old_local = local_name(local_id);
    let new_local = format!("%_v{}", local_id);
    let patch = format!(
        "diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@\n-  {old_local}: <type> = <expr>\n+  {new_local}: <type> = <expr>\n",
        path = source_path,
    );
    SuggestedFix {
        title: format!("Prefix '{}' with '_' to mark intentionally unused", old_local),
        patch_format: "unified-diff".to_string(),
        patch,
        confidence: 0.72,
    }
}

fn suggested_fix_empty_block(source_path: &str, block_id: u32) -> SuggestedFix {
    let patch = format!(
        "diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@\n-bb{block_id}:\n-  ; empty block\n+; removed empty block bb{block_id}\n",
        path = source_path,
    );
    SuggestedFix {
        title: format!("Remove empty block bb{}", block_id),
        patch_format: "unified-diff".to_string(),
        patch,
        confidence: 0.62,
    }
}

fn suggested_fix_unnecessary_borrow(
    source_path: &str,
    borrow_local: u32,
    borrow_source: &str,
) -> SuggestedFix {
    let local = local_name(borrow_local);
    let patch = format!(
        "diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@\n-  {local}: <borrow_ty> = borrow.shared {borrow_source}\n-  %v_next: <type> = getfield {local}, <field>\n+  %v_next: <type> = getfield {borrow_source}, <field>\n",
        path = source_path,
    );
    SuggestedFix {
        title: format!("Remove unnecessary borrow '{}'", local),
        patch_format: "unified-diff".to_string(),
        patch,
        confidence: 0.68,
    }
}

fn reachable_block_ids(func: &HirFunction) -> HashSet<u32> {
    let mut block_map = HashMap::new();
    for block in &func.blocks {
        block_map.insert(block.id.0, block);
    }

    let mut reachable = HashSet::new();
    let Some(entry) = func.blocks.first() else {
        return reachable;
    };
    let mut worklist = vec![entry.id.0];

    while let Some(block_id) = worklist.pop() {
        if !reachable.insert(block_id) {
            continue;
        }
        let Some(block) = block_map.get(&block_id) else {
            continue;
        };
        worklist.extend(lint_block_successors(&block.terminator));
    }

    reachable
}

fn lint_block_successors(term: &HirTerminator) -> Vec<u32> {
    match term {
        HirTerminator::Ret(_) | HirTerminator::Unreachable => Vec::new(),
        HirTerminator::Br(block_id) => vec![block_id.0],
        HirTerminator::Cbr {
            then_bb, else_bb, ..
        } => vec![then_bb.0, else_bb.0],
        HirTerminator::Switch { arms, default, .. } => {
            let mut out = arms.iter().map(|(_, block)| block.0).collect::<Vec<_>>();
            out.push(default.0);
            out
        }
    }
}

fn op_uses_local(op: &HirOp, local_id: u32) -> bool {
    hir_op_values(op)
        .into_iter()
        .any(|value| matches!(value, HirValue::Local(local) if local.0 == local_id))
}

fn op_void_uses_local(op: &HirOpVoid, local_id: u32) -> bool {
    hir_op_void_values(op)
        .into_iter()
        .any(|value| matches!(value, HirValue::Local(local) if local.0 == local_id))
}

fn terminator_uses_local(term: &HirTerminator, local_id: u32) -> bool {
    hir_terminator_values(term)
        .into_iter()
        .any(|value| matches!(value, HirValue::Local(local) if local.0 == local_id))
}

fn hir_value_display(value: &HirValue) -> String {
    match value {
        HirValue::Local(local) => local_name(local.0),
        HirValue::Const(_) => "<const>".to_string(),
    }
}

fn hir_op_values(op: &HirOp) -> Vec<HirValue> {
    match op {
        HirOp::Const(_) => vec![],
        HirOp::Move { v } => vec![v.clone()],
        HirOp::BorrowShared { v } => vec![v.clone()],
        HirOp::BorrowMut { v } => vec![v.clone()],
        HirOp::New { fields, .. } => fields.iter().map(|(_, v)| v.clone()).collect(),
        HirOp::GetField { obj, .. } => vec![obj.clone()],
        HirOp::IAdd { lhs, rhs }
        | HirOp::ISub { lhs, rhs }
        | HirOp::IMul { lhs, rhs }
        | HirOp::ISDiv { lhs, rhs }
        | HirOp::IUDiv { lhs, rhs }
        | HirOp::ISRem { lhs, rhs }
        | HirOp::IURem { lhs, rhs }
        | HirOp::IAddWrap { lhs, rhs }
        | HirOp::ISubWrap { lhs, rhs }
        | HirOp::IMulWrap { lhs, rhs }
        | HirOp::IAddChecked { lhs, rhs }
        | HirOp::ISubChecked { lhs, rhs }
        | HirOp::IMulChecked { lhs, rhs }
        | HirOp::IAnd { lhs, rhs }
        | HirOp::IOr { lhs, rhs }
        | HirOp::IXor { lhs, rhs }
        | HirOp::IShl { lhs, rhs }
        | HirOp::ILshr { lhs, rhs }
        | HirOp::IAshr { lhs, rhs }
        | HirOp::ICmp { lhs, rhs, .. }
        | HirOp::FCmp { lhs, rhs, .. }
        | HirOp::FAdd { lhs, rhs }
        | HirOp::FSub { lhs, rhs }
        | HirOp::FMul { lhs, rhs }
        | HirOp::FDiv { lhs, rhs }
        | HirOp::FRem { lhs, rhs }
        | HirOp::FAddFast { lhs, rhs }
        | HirOp::FSubFast { lhs, rhs }
        | HirOp::FMulFast { lhs, rhs }
        | HirOp::FDivFast { lhs, rhs } => {
            vec![lhs.clone(), rhs.clone()]
        }
        HirOp::Cast { v, .. } => vec![v.clone()],
        HirOp::PtrNull { .. } => vec![],
        HirOp::PtrAddr { p } => vec![p.clone()],
        HirOp::PtrFromAddr { addr, .. } => vec![addr.clone()],
        HirOp::PtrAdd { p, count } => vec![p.clone(), count.clone()],
        HirOp::PtrLoad { p, .. } => vec![p.clone()],
        HirOp::PtrStore { p, v, .. } => vec![p.clone(), v.clone()],
        HirOp::Call { args, .. } => args.clone(),
        HirOp::CallIndirect { callee, args } | HirOp::CallVoidIndirect { callee, args } => {
            let mut vs = vec![callee.clone()];
            vs.extend(args.iter().cloned());
            vs
        }
        HirOp::SuspendCall { args, .. } => args.clone(),
        HirOp::SuspendAwait { fut } => vec![fut.clone()],
        HirOp::Phi { incomings, .. } => incomings.iter().map(|(_, v)| v.clone()).collect(),
        HirOp::Share { v }
        | HirOp::CloneShared { v }
        | HirOp::CloneWeak { v }
        | HirOp::WeakDowngrade { v }
        | HirOp::WeakUpgrade { v } => vec![v.clone()],
        HirOp::EnumNew { args, .. } => args.iter().map(|(_, v)| v.clone()).collect(),
        HirOp::EnumTag { v } | HirOp::EnumPayload { v, .. } | HirOp::EnumIs { v, .. } => {
            vec![v.clone()]
        }
        HirOp::CallableCapture { captures, .. } => {
            captures.iter().map(|(_, v)| v.clone()).collect()
        }
        HirOp::ArrNew { cap, .. } => vec![cap.clone()],
        HirOp::ArrLen { arr } => vec![arr.clone()],
        HirOp::ArrGet { arr, idx } => vec![arr.clone(), idx.clone()],
        HirOp::ArrSet { arr, idx, val } => vec![arr.clone(), idx.clone(), val.clone()],
        HirOp::ArrPush { arr, val } => vec![arr.clone(), val.clone()],
        HirOp::ArrPop { arr } => vec![arr.clone()],
        HirOp::ArrSlice { arr, start, end } => vec![arr.clone(), start.clone(), end.clone()],
        HirOp::ArrContains { arr, val } => vec![arr.clone(), val.clone()],
        HirOp::ArrSort { arr } => vec![arr.clone()],
        HirOp::ArrMap { arr, func }
        | HirOp::ArrFilter { arr, func }
        | HirOp::ArrForeach { arr, func } => {
            vec![arr.clone(), func.clone()]
        }
        HirOp::ArrReduce { arr, init, func } => vec![arr.clone(), init.clone(), func.clone()],
        HirOp::MapNew { .. } => vec![],
        HirOp::MapLen { map } => vec![map.clone()],
        HirOp::MapGet { map, key }
        | HirOp::MapGetRef { map, key }
        | HirOp::MapDelete { map, key }
        | HirOp::MapContainsKey { map, key }
        | HirOp::MapDeleteVoid { map, key } => {
            vec![map.clone(), key.clone()]
        }
        HirOp::MapKeys { map } | HirOp::MapValues { map } => vec![map.clone()],
        HirOp::MapSet { map, key, val } => vec![map.clone(), key.clone(), val.clone()],
        HirOp::StrConcat { a, b } | HirOp::StrEq { a, b } => vec![a.clone(), b.clone()],
        HirOp::StrLen { s } | HirOp::StrBytes { s } => vec![s.clone()],
        HirOp::StrSlice { s, start, end } => vec![s.clone(), start.clone(), end.clone()],
        HirOp::StrBuilderNew => vec![],
        HirOp::StrBuilderAppendStr { b, s } => vec![b.clone(), s.clone()],
        HirOp::StrBuilderAppendI64 { b, v }
        | HirOp::StrBuilderAppendI32 { b, v }
        | HirOp::StrBuilderAppendF64 { b, v }
        | HirOp::StrBuilderAppendBool { b, v } => {
            vec![b.clone(), v.clone()]
        }
        HirOp::StrBuilderBuild { b } => vec![b.clone()],
        HirOp::StrParseI64 { s }
        | HirOp::StrParseU64 { s }
        | HirOp::StrParseF64 { s }
        | HirOp::StrParseBool { s } => vec![s.clone()],
        HirOp::JsonEncode { v, .. } => vec![v.clone()],
        HirOp::JsonDecode { s, .. } => vec![s.clone()],
        HirOp::GpuThreadId
        | HirOp::GpuWorkgroupId
        | HirOp::GpuWorkgroupSize
        | HirOp::GpuGlobalId => vec![],
        HirOp::GpuBufferLoad { buf, idx } => vec![buf.clone(), idx.clone()],
        HirOp::GpuBufferLen { buf } => vec![buf.clone()],
        HirOp::GpuShared { size, .. } => vec![size.clone()],
        HirOp::GpuLaunch {
            device,
            groups,
            threads,
            args,
            ..
        }
        | HirOp::GpuLaunchAsync {
            device,
            groups,
            threads,
            args,
            ..
        } => {
            let mut vs = vec![device.clone(), groups.clone(), threads.clone()];
            vs.extend(args.iter().cloned());
            vs
        }
        HirOp::Panic { msg } => vec![msg.clone()],
    }
}

fn hir_op_void_values(op: &HirOpVoid) -> Vec<HirValue> {
    match op {
        HirOpVoid::CallVoid { args, .. } => args.clone(),
        HirOpVoid::CallVoidIndirect { callee, args } => {
            let mut vs = vec![callee.clone()];
            vs.extend(args.iter().cloned());
            vs
        }
        HirOpVoid::SetField { obj, value, .. } => vec![obj.clone(), value.clone()],
        HirOpVoid::ArrSet { arr, idx, val } => vec![arr.clone(), idx.clone(), val.clone()],
        HirOpVoid::ArrPush { arr, val } => vec![arr.clone(), val.clone()],
        HirOpVoid::ArrSort { arr } => vec![arr.clone()],
        HirOpVoid::ArrForeach { arr, func } => vec![arr.clone(), func.clone()],
        HirOpVoid::MapSet { map, key, val } => vec![map.clone(), key.clone(), val.clone()],
        HirOpVoid::MapDeleteVoid { map, key } => vec![map.clone(), key.clone()],
        HirOpVoid::StrBuilderAppendStr { b, s } => vec![b.clone(), s.clone()],
        HirOpVoid::StrBuilderAppendI64 { b, v }
        | HirOpVoid::StrBuilderAppendI32 { b, v }
        | HirOpVoid::StrBuilderAppendF64 { b, v }
        | HirOpVoid::StrBuilderAppendBool { b, v } => {
            vec![b.clone(), v.clone()]
        }
        HirOpVoid::PtrStore { p, v, .. } => vec![p.clone(), v.clone()],
        HirOpVoid::Panic { msg } => vec![msg.clone()],
        HirOpVoid::GpuBarrier => vec![],
        HirOpVoid::GpuBufferStore { buf, idx, val } => vec![buf.clone(), idx.clone(), val.clone()],
    }
}

fn hir_terminator_values(term: &HirTerminator) -> Vec<HirValue> {
    match term {
        HirTerminator::Ret(Some(v)) => vec![v.clone()],
        HirTerminator::Ret(None) => vec![],
        HirTerminator::Br(_) => vec![],
        HirTerminator::Cbr { cond, .. } => vec![cond.clone()],
        HirTerminator::Switch { val, .. } => vec![val.clone()],
        HirTerminator::Unreachable => vec![],
    }
}

/// Test discovery + execution entrypoint (§5.2.7, §33.1).
pub fn run_tests(config: &DriverConfig, filter: Option<&str>) -> TestResult {
    let build_result = build(config);
    let mut discovered = discover_test_functions_from_hir(config);

    if let Some(pattern) = filter {
        discovered.retain(|name| name.contains(pattern));
    }

    let executable = if emit_contains(&config.emit, "exe") && build_result.success {
        find_executable_artifact(&config.target_triple, &build_result.artifacts)
    } else {
        None
    };

    let mut test_names = Vec::with_capacity(discovered.len());
    match executable {
        Some(path) => {
            for test_name in discovered {
                let passed = run_single_test_binary(&path, &test_name);
                test_names.push((test_name, passed));
            }
        }
        None => {
            let passed = build_result.success;
            for test_name in discovered {
                test_names.push((test_name, passed));
            }
        }
    }

    let passed = test_names.iter().filter(|(_, passed)| *passed).count();
    let failed = test_names.len().saturating_sub(passed);

    TestResult {
        total: test_names.len(),
        passed,
        failed,
        test_names,
    }
}

fn discover_test_functions_from_hir(config: &DriverConfig) -> Vec<String> {
    let mut ast_files = Vec::new();
    let mut discovered = Vec::new();
    let max_errors = config.max_errors.max(1);
    let mut diag = DiagnosticBag::new(max_errors);

    let source_paths = collect_test_source_paths(&config.entry_path);
    for (idx, path) in source_paths.iter().enumerate() {
        let source = match fs::read_to_string(path) {
            Ok(source) => source,
            Err(_) => continue,
        };
        let file_id = FileId(idx as u32);
        let tokens = lex(file_id, &source, &mut diag);
        if let Ok(ast) = parse_file(&tokens, file_id, &mut diag) {
            ast_files.push(ast);
        }
    }

    if ast_files.is_empty() {
        return discovered;
    }

    let resolved = match resolve_modules(&ast_files, &mut diag) {
        Ok(modules) => modules,
        Err(()) => return discovered,
    };

    let mut type_ctx = TypeCtx::new();
    for module in &resolved {
        let Ok(hir) = lower_to_hir(module, &mut type_ctx, &mut diag) else {
            continue;
        };
        for function in hir.functions {
            if function.name.starts_with("@test_") {
                discovered.push(function.name);
            }
        }
    }

    discovered.sort();
    discovered.dedup();
    discovered
}

fn collect_test_source_paths(entry_path: &str) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    push_mp_path(&mut paths, Path::new(entry_path));
    collect_mp_files(Path::new("src"), &mut paths);
    collect_mp_files(Path::new("tests"), &mut paths);
    paths.sort();
    paths.dedup();
    paths
}

fn collect_mp_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };

    let mut paths: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            collect_mp_files(&path, out);
            continue;
        }
        push_mp_path(out, &path);
    }
}

fn push_mp_path(out: &mut Vec<PathBuf>, path: &Path) {
    if path.is_file()
        && path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext == "mp")
    {
        out.push(path.to_path_buf());
    }
}

fn find_executable_artifact(target_triple: &str, artifacts: &[String]) -> Option<String> {
    let is_windows = is_windows_target(target_triple);
    artifacts.iter().find_map(|artifact| {
        let path = Path::new(artifact);
        let is_executable = if is_windows {
            path.extension().and_then(|ext| ext.to_str()) == Some("exe")
        } else {
            path.extension().is_none()
        };
        (is_executable && path.exists()).then(|| artifact.clone())
    })
}

fn run_single_test_binary(path: &str, test_name: &str) -> bool {
    Command::new(path)
        .env("MAGPIE_TEST_NAME", test_name)
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn emit_symbols_graph(modules: &[HirModule]) -> serde_json::Value {
    let mut ordered_modules = modules.iter().collect::<Vec<_>>();
    ordered_modules.sort_by(|a, b| a.path.cmp(&b.path).then(a.sid.0.cmp(&b.sid.0)));

    let modules_json = ordered_modules
        .into_iter()
        .map(|module| {
            let mut functions = module
                .functions
                .iter()
                .map(|func| json!({ "name": func.name, "sid": func.sid.0 }))
                .collect::<Vec<_>>();
            functions.sort_by(|lhs, rhs| lhs["name"].as_str().cmp(&rhs["name"].as_str()));

            let mut types = module
                .type_decls
                .iter()
                .map(|decl| match decl {
                    magpie_hir::HirTypeDecl::Struct { sid, name, .. } => {
                        json!({ "name": name, "sid": sid.0, "kind": "struct" })
                    }
                    magpie_hir::HirTypeDecl::Enum { sid, name, .. } => {
                        json!({ "name": name, "sid": sid.0, "kind": "enum" })
                    }
                })
                .collect::<Vec<_>>();
            types.sort_by(|lhs, rhs| lhs["name"].as_str().cmp(&rhs["name"].as_str()));

            let mut globals = module
                .globals
                .iter()
                .map(|global| {
                    let fqn = format!("{}.{}", module.path, global.name);
                    let sid = generate_sid('G', &fqn);
                    json!({ "name": global.name, "sid": sid.0 })
                })
                .collect::<Vec<_>>();
            globals.sort_by(|lhs, rhs| lhs["name"].as_str().cmp(&rhs["name"].as_str()));

            json!({
                "module_path": module.path,
                "module_sid": module.sid.0,
                "functions": functions,
                "types": types,
                "globals": globals,
            })
        })
        .collect::<Vec<_>>();

    json!({
        "graph": "symbols",
        "modules": modules_json,
    })
}

fn emit_deps_graph(modules: &[HirModule]) -> serde_json::Value {
    let mut fn_owner: HashMap<String, (String, String, String)> = HashMap::new();
    for module in modules {
        for func in &module.functions {
            fn_owner.insert(
                func.sid.0.clone(),
                (module.path.clone(), module.sid.0.clone(), func.name.clone()),
            );
        }
    }

    let mut edges: BTreeSet<(String, String, String, String, String, String, String)> =
        BTreeSet::new();
    for module in modules {
        for func in &module.functions {
            for block in &func.blocks {
                for instr in &block.instrs {
                    match &instr.op {
                        HirOp::Call { callee_sid, .. } => {
                            record_dep_edge(
                                &mut edges,
                                &fn_owner,
                                module,
                                func,
                                callee_sid,
                                "call",
                            );
                        }
                        HirOp::SuspendCall { callee_sid, .. } => {
                            record_dep_edge(
                                &mut edges,
                                &fn_owner,
                                module,
                                func,
                                callee_sid,
                                "suspend.call",
                            );
                        }
                        _ => {}
                    }
                }
                for void_op in &block.void_ops {
                    if let HirOpVoid::CallVoid { callee_sid, .. } = void_op {
                        record_dep_edge(
                            &mut edges,
                            &fn_owner,
                            module,
                            func,
                            callee_sid,
                            "call_void",
                        );
                    }
                }
            }
        }
    }

    let mut modules_json = modules
        .iter()
        .map(|module| json!({ "module_path": module.path, "module_sid": module.sid.0 }))
        .collect::<Vec<_>>();
    modules_json.sort_by(|lhs, rhs| lhs["module_path"].as_str().cmp(&rhs["module_path"].as_str()));

    let edges_json = edges
        .into_iter()
        .map(
            |(from_path, from_sid, to_path, to_sid, caller_fn_sid, callee_fn_sid, via)| {
                json!({
                    "from_module_path": from_path,
                    "from_module_sid": from_sid,
                    "to_module_path": to_path,
                    "to_module_sid": to_sid,
                    "caller_fn_sid": caller_fn_sid,
                    "callee_fn_sid": callee_fn_sid,
                    "via": via,
                })
            },
        )
        .collect::<Vec<_>>();

    json!({
        "graph": "deps",
        "modules": modules_json,
        "edges": edges_json,
    })
}

fn record_dep_edge(
    edges: &mut BTreeSet<(String, String, String, String, String, String, String)>,
    fn_owner: &HashMap<String, (String, String, String)>,
    caller_module: &HirModule,
    caller_fn: &HirFunction,
    callee_sid: &magpie_types::Sid,
    via: &str,
) {
    let Some((to_path, to_sid, _to_fn_name)) = fn_owner.get(&callee_sid.0) else {
        return;
    };
    if to_sid == &caller_module.sid.0 {
        return;
    }
    edges.insert((
        caller_module.path.clone(),
        caller_module.sid.0.clone(),
        to_path.clone(),
        to_sid.clone(),
        caller_fn.sid.0.clone(),
        callee_sid.0.clone(),
        via.to_string(),
    ));
}

fn emit_cfg_graph(modules: &[HirModule]) -> serde_json::Value {
    let mut functions = Vec::new();
    for module in modules {
        for func in &module.functions {
            let mut blocks = func
                .blocks
                .iter()
                .map(|block| json!({ "id": block.id.0 }))
                .collect::<Vec<_>>();
            blocks.sort_by_key(|b| b["id"].as_u64().unwrap_or(0));

            let mut edges = Vec::new();
            for block in &func.blocks {
                for (to, kind) in cfg_successors(&block.terminator) {
                    edges.push((block.id.0, to, kind));
                }
            }
            edges.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0).then(lhs.1.cmp(&rhs.1)).then(lhs.2.cmp(&rhs.2)));

            let edges = edges
                .into_iter()
                .map(|(from, to, kind)| json!({ "from": from, "to": to, "kind": kind }))
                .collect::<Vec<_>>();

            functions.push(json!({
                "module_path": module.path,
                "module_sid": module.sid.0,
                "fn_name": func.name,
                "fn_sid": func.sid.0,
                "blocks": blocks,
                "edges": edges,
            }));
        }
    }

    functions.sort_by(|lhs, rhs| {
        lhs["module_path"]
            .as_str()
            .cmp(&rhs["module_path"].as_str())
            .then(lhs["fn_name"].as_str().cmp(&rhs["fn_name"].as_str()))
    });

    json!({
        "graph": "cfg",
        "functions": functions,
    })
}

fn cfg_successors(term: &HirTerminator) -> Vec<(u32, String)> {
    match term {
        HirTerminator::Ret(_) | HirTerminator::Unreachable => Vec::new(),
        HirTerminator::Br(target) => vec![(target.0, "br".to_string())],
        HirTerminator::Cbr { then_bb, else_bb, .. } => vec![
            (then_bb.0, "cbr_true".to_string()),
            (else_bb.0, "cbr_false".to_string()),
        ],
        HirTerminator::Switch { arms, default, .. } => {
            let mut out = arms
                .iter()
                .map(|(_, block)| (block.0, "switch_arm".to_string()))
                .collect::<Vec<_>>();
            out.push((default.0, "switch_default".to_string()));
            out
        }
    }
}

fn emit_ownership_graph(modules: &[HirModule], type_ctx: &TypeCtx) -> serde_json::Value {
    let mut functions = Vec::new();
    for module in modules {
        for func in &module.functions {
            let mut chains = Vec::new();
            for block in &func.blocks {
                for instr in &block.instrs {
                    if let Some((kind, src)) = ownership_chain_step(&instr.op) {
                        chains.push(json!({
                            "block": block.id.0,
                            "op": kind,
                            "from": src,
                            "to": format!("%{}", instr.dst.0),
                            "ty": type_ctx.type_str(instr.ty),
                        }));
                    }
                }
            }

            functions.push(json!({
                "module_path": module.path,
                "module_sid": module.sid.0,
                "fn_name": func.name,
                "fn_sid": func.sid.0,
                "chains": chains,
            }));
        }
    }

    functions.sort_by(|lhs, rhs| {
        lhs["module_path"]
            .as_str()
            .cmp(&rhs["module_path"].as_str())
            .then(lhs["fn_name"].as_str().cmp(&rhs["fn_name"].as_str()))
    });

    json!({
        "graph": "ownership",
        "functions": functions,
    })
}

fn ownership_chain_step(op: &HirOp) -> Option<(&'static str, String)> {
    let (kind, value) = match op {
        HirOp::Move { v } => ("move", v),
        HirOp::BorrowShared { v } => ("borrow_shared", v),
        HirOp::BorrowMut { v } => ("borrow_mut", v),
        HirOp::Share { v } => ("share", v),
        HirOp::CloneShared { v } => ("clone_shared", v),
        HirOp::CloneWeak { v } => ("clone_weak", v),
        HirOp::WeakDowngrade { v } => ("weak_downgrade", v),
        HirOp::WeakUpgrade { v } => ("weak_upgrade", v),
        _ => return None,
    };
    Some((kind, ownership_value_str(value)))
}

fn ownership_value_str(value: &HirValue) -> String {
    match value {
        HirValue::Local(local) => format!("%{}", local.0),
        HirValue::Const(_) => "const".to_string(),
    }
}

fn collect_mms_artifact_paths(
    config: &DriverConfig,
    artifact_paths: &[String],
    llvm_ir_paths: &[PathBuf],
    mpir_modules: &[MpirModule],
) -> Vec<PathBuf> {
    let mut paths = artifact_paths.iter().map(PathBuf::from).collect::<Vec<_>>();
    paths.extend(llvm_ir_paths.iter().cloned());

    if emit_contains(&config.emit, "mpir") {
        let module_count = mpir_modules.len();
        for idx in 0..module_count {
            paths.push(stage_module_output_path(config, idx, module_count, "mpir"));
        }
    }

    for (emit_kind, suffix) in [
        ("symgraph", "symgraph"),
        ("depsgraph", "depsgraph"),
        ("ownershipgraph", "ownershipgraph"),
        ("cfggraph", "cfggraph"),
    ] {
        if emit_contains(&config.emit, emit_kind) {
            paths.push(stage_graph_output_path(config, suffix));
        }
    }

    let mut deduped: BTreeSet<PathBuf> = BTreeSet::new();
    for path in paths {
        if path.is_file() {
            deduped.insert(path);
        }
    }
    deduped.into_iter().collect()
}

fn update_mms_index(
    config: &DriverConfig,
    artifacts: &[PathBuf],
    hir_modules: &[HirModule],
    _type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) -> Option<PathBuf> {
    let items = build_mms_items(artifacts, hir_modules);
    let index = build_index(&items);
    let encoded = match serde_json::to_string_pretty(&index) {
        Ok(encoded) => encoded,
        Err(err) => {
            emit_driver_diag(
                diag,
                "MPP0003",
                Severity::Warning,
                "failed to encode mms index",
                format!("Could not serialize MMS index: {}", err),
            );
            return None;
        }
    };

    let index_path = stage_mms_index_output_path(config);
    if let Err(err) = write_text_artifact(&index_path, &encoded) {
        emit_driver_diag(
            diag,
            "MPP0003",
            Severity::Warning,
            "failed to write mms index",
            err,
        );
        return None;
    }

    Some(index_path)
}

fn build_mms_items(artifacts: &[PathBuf], hir_modules: &[HirModule]) -> Vec<MmsItem> {
    let default_module_sid = hir_modules
        .first()
        .map(|module| module.sid.0.clone())
        .unwrap_or_else(|| "M:0000000000".to_string());

    artifacts
        .iter()
        .enumerate()
        .map(|(idx, path)| {
            let path_text = path.to_string_lossy().to_string();
            let text = fs::read_to_string(path)
                .unwrap_or_else(|_| format!("artifact path: {}", path.display()));
            let ext = path
                .extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            let kind = if ext == "json" {
                "spec_excerpt"
            } else {
                "symbol_capsule"
            };
            let mut token_cost = BTreeMap::new();
            token_cost.insert(
                "approx:utf8_4chars".to_string(),
                ((text.len() as u32).saturating_add(3)) / 4,
            );

            MmsItem {
                item_id: format!("I:{:016X}", stable_hash_u64(path_text.as_bytes())),
                kind: kind.to_string(),
                sid: default_module_sid.clone(),
                fqn: path_text.clone(),
                module_sid: default_module_sid.clone(),
                source_digest: format!("{:016x}", stable_hash_u64(path_text.as_bytes())),
                body_digest: format!("{:016x}", stable_hash_u64(text.as_bytes())),
                text,
                tags: vec![
                    "artifact".to_string(),
                    ext,
                    format!("order:{}", idx),
                ],
                priority: 50,
                token_cost,
            }
        })
        .collect()
}

fn stable_hash_u64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001b3;
    let mut hash = FNV_OFFSET;
    for b in bytes {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn lower_async_functions(modules: &mut [HirModule], type_ctx: &mut TypeCtx) {
    let _ = type_ctx;
    let mut lowered_async_sids: HashSet<Sid> = HashSet::new();

    for module in modules.iter_mut() {
        for func in &mut module.functions {
            if !func.is_async {
                continue;
            }
            if lower_async_function(func) {
                lowered_async_sids.insert(func.sid.clone());
            }
        }
    }

    if lowered_async_sids.is_empty() {
        return;
    }

    for module in modules.iter_mut() {
        for func in &mut module.functions {
            for block in &mut func.blocks {
                for instr in &mut block.instrs {
                    match &mut instr.op {
                        HirOp::Call {
                            callee_sid, args, ..
                        }
                        | HirOp::SuspendCall {
                            callee_sid, args, ..
                        } if lowered_async_sids.contains(callee_sid) => {
                            args.insert(0, hir_i32_value(0));
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

fn lower_async_function(func: &mut HirFunction) -> bool {
    let suspend_count = func
        .blocks
        .iter()
        .map(|block| {
            block
                .instrs
                .iter()
                .filter(|instr| is_suspend_op(&instr.op))
                .count()
        })
        .sum::<usize>();
    if suspend_count == 0 {
        func.is_async = false;
        return false;
    }

    let state_param = LocalId(next_local_id(func));
    func.params.insert(0, (state_param, fixed_type_ids::I32));

    let mut next_block = next_block_id(func);
    let mut resume_blocks = Vec::new();
    let mut blk_idx = 0usize;
    while blk_idx < func.blocks.len() {
        let split_at = func.blocks[blk_idx]
            .instrs
            .iter()
            .position(|instr| is_suspend_op(&instr.op));

        let Some(split_at) = split_at else {
            blk_idx += 1;
            continue;
        };

        let resume_id = BlockId(next_block);
        next_block = next_block.saturating_add(1);

        let tail_instrs = func.blocks[blk_idx].instrs.split_off(split_at + 1);
        let tail_void_ops = std::mem::take(&mut func.blocks[blk_idx].void_ops);
        let tail_term =
            std::mem::replace(&mut func.blocks[blk_idx].terminator, HirTerminator::Br(resume_id));
        func.blocks.insert(
            blk_idx + 1,
            HirBlock {
                id: resume_id,
                instrs: tail_instrs,
                void_ops: tail_void_ops,
                terminator: tail_term,
            },
        );
        resume_blocks.push(resume_id);
        blk_idx += 1;
    }

    let entry_id = func.blocks[0].id;
    let dispatch_id = BlockId(next_block);
    next_block = next_block.saturating_add(1);
    let invalid_state_id = BlockId(next_block);

    let mut arms = Vec::with_capacity(resume_blocks.len() + 1);
    arms.push((hir_i32_const(0), entry_id));
    for (idx, resume_id) in resume_blocks.iter().enumerate() {
        arms.push((hir_i32_const((idx + 1) as i32), *resume_id));
    }

    func.blocks.insert(
        0,
        HirBlock {
            id: dispatch_id,
            instrs: Vec::new(),
            void_ops: Vec::new(),
            terminator: HirTerminator::Switch {
                val: HirValue::Local(state_param),
                arms,
                default: invalid_state_id,
            },
        },
    );
    func.blocks.push(HirBlock {
        id: invalid_state_id,
        instrs: Vec::new(),
        void_ops: Vec::new(),
        terminator: HirTerminator::Unreachable,
    });

    func.is_async = false;
    true
}

fn is_suspend_op(op: &HirOp) -> bool {
    matches!(op, HirOp::SuspendCall { .. } | HirOp::SuspendAwait { .. })
}

fn next_local_id(func: &HirFunction) -> u32 {
    let from_params = func.params.iter().map(|(id, _)| id.0).max().unwrap_or(0);
    let from_instrs = func
        .blocks
        .iter()
        .flat_map(|block| block.instrs.iter().map(|instr| instr.dst.0))
        .max()
        .unwrap_or(0);
    from_params.max(from_instrs).saturating_add(1)
}

fn next_block_id(func: &HirFunction) -> u32 {
    func.blocks
        .iter()
        .map(|block| block.id.0)
        .max()
        .unwrap_or(0)
        .saturating_add(1)
}

fn hir_i32_const(value: i32) -> HirConst {
    HirConst {
        ty: fixed_type_ids::I32,
        lit: HirConstLit::IntLit(i128::from(value)),
    }
}

fn hir_i32_value(value: i32) -> HirValue {
    HirValue::Const(hir_i32_const(value))
}

pub fn lower_hir_module_to_mpir(module: &HirModule, type_ctx: &TypeCtx) -> MpirModule {
    let mut functions = Vec::with_capacity(module.functions.len());
    for func in &module.functions {
        functions.push(lower_hir_function_to_mpir(func));
    }

    MpirModule {
        sid: module.sid.clone(),
        path: module.path.clone(),
        type_table: MpirTypeTable {
            types: type_ctx.types.clone(),
        },
        functions,
        globals: module
            .globals
            .iter()
            .map(|g| (g.id, g.ty, g.init.clone()))
            .collect(),
    }
}

fn lower_hir_function_to_mpir(func: &HirFunction) -> MpirFn {
    let mut blocks = Vec::with_capacity(func.blocks.len());
    let mut locals = Vec::new();

    for block in &func.blocks {
        for instr in &block.instrs {
            locals.push(MpirLocalDecl {
                id: instr.dst,
                ty: instr.ty,
                name: format!("v{}", instr.dst.0),
            });
        }
        blocks.push(lower_hir_block_to_mpir(block));
    }

    locals.sort_by_key(|l| l.id.0);
    locals.dedup_by_key(|l| l.id.0);

    MpirFn {
        sid: func.sid.clone(),
        name: func.name.clone(),
        params: func.params.clone(),
        ret_ty: func.ret_ty,
        blocks,
        locals,
        is_async: func.is_async,
    }
}

fn lower_hir_block_to_mpir(block: &HirBlock) -> MpirBlock {
    MpirBlock {
        id: block.id,
        instrs: block
            .instrs
            .iter()
            .map(lower_hir_instr_to_mpir)
            .collect::<Vec<_>>(),
        void_ops: block
            .void_ops
            .iter()
            .map(lower_hir_void_op_to_mpir)
            .collect::<Vec<_>>(),
        terminator: lower_hir_terminator_to_mpir(&block.terminator),
    }
}

fn lower_hir_instr_to_mpir(instr: &HirInstr) -> MpirInstr {
    MpirInstr {
        dst: instr.dst,
        ty: instr.ty,
        op: lower_hir_op_to_mpir(&instr.op),
    }
}

fn lower_hir_op_to_mpir(op: &HirOp) -> MpirOp {
    match op {
        HirOp::Const(v) => MpirOp::Const(v.clone()),
        HirOp::Move { v } => MpirOp::Move {
            v: lower_hir_value_to_mpir(v),
        },
        HirOp::BorrowShared { v } => MpirOp::BorrowShared {
            v: lower_hir_value_to_mpir(v),
        },
        HirOp::BorrowMut { v } => MpirOp::BorrowMut {
            v: lower_hir_value_to_mpir(v),
        },
        HirOp::New { ty, fields } => MpirOp::New {
            ty: *ty,
            fields: fields
                .iter()
                .map(|(name, value)| (name.clone(), lower_hir_value_to_mpir(value)))
                .collect(),
        },
        HirOp::GetField { obj, field } => MpirOp::GetField {
            obj: lower_hir_value_to_mpir(obj),
            field: field.clone(),
        },
        HirOp::IAdd { lhs, rhs } => MpirOp::IAdd {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::ISub { lhs, rhs } => MpirOp::ISub {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::IMul { lhs, rhs } => MpirOp::IMul {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::ISDiv { lhs, rhs } => MpirOp::ISDiv {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::IUDiv { lhs, rhs } => MpirOp::IUDiv {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::ISRem { lhs, rhs } => MpirOp::ISRem {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::IURem { lhs, rhs } => MpirOp::IURem {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::IAddWrap { lhs, rhs } => MpirOp::IAddWrap {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::ISubWrap { lhs, rhs } => MpirOp::ISubWrap {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::IMulWrap { lhs, rhs } => MpirOp::IMulWrap {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::IAddChecked { lhs, rhs } => MpirOp::IAddChecked {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::ISubChecked { lhs, rhs } => MpirOp::ISubChecked {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::IMulChecked { lhs, rhs } => MpirOp::IMulChecked {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::IAnd { lhs, rhs } => MpirOp::IAnd {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::IOr { lhs, rhs } => MpirOp::IOr {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::IXor { lhs, rhs } => MpirOp::IXor {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::IShl { lhs, rhs } => MpirOp::IShl {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::ILshr { lhs, rhs } => MpirOp::ILshr {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::IAshr { lhs, rhs } => MpirOp::IAshr {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::ICmp { pred, lhs, rhs } => MpirOp::ICmp {
            pred: pred.clone(),
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::FCmp { pred, lhs, rhs } => MpirOp::FCmp {
            pred: pred.clone(),
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::FAdd { lhs, rhs } => MpirOp::FAdd {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::FSub { lhs, rhs } => MpirOp::FSub {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::FMul { lhs, rhs } => MpirOp::FMul {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::FDiv { lhs, rhs } => MpirOp::FDiv {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::FRem { lhs, rhs } => MpirOp::FRem {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::FAddFast { lhs, rhs } => MpirOp::FAddFast {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::FSubFast { lhs, rhs } => MpirOp::FSubFast {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::FMulFast { lhs, rhs } => MpirOp::FMulFast {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::FDivFast { lhs, rhs } => MpirOp::FDivFast {
            lhs: lower_hir_value_to_mpir(lhs),
            rhs: lower_hir_value_to_mpir(rhs),
        },
        HirOp::Cast { to, v } => MpirOp::Cast {
            to: *to,
            v: lower_hir_value_to_mpir(v),
        },
        HirOp::PtrNull { to } => MpirOp::PtrNull { to: *to },
        HirOp::PtrAddr { p } => MpirOp::PtrAddr {
            p: lower_hir_value_to_mpir(p),
        },
        HirOp::PtrFromAddr { to, addr } => MpirOp::PtrFromAddr {
            to: *to,
            addr: lower_hir_value_to_mpir(addr),
        },
        HirOp::PtrAdd { p, count } => MpirOp::PtrAdd {
            p: lower_hir_value_to_mpir(p),
            count: lower_hir_value_to_mpir(count),
        },
        HirOp::PtrLoad { to, p } => MpirOp::PtrLoad {
            to: *to,
            p: lower_hir_value_to_mpir(p),
        },
        HirOp::PtrStore { to, p, v } => MpirOp::PtrStore {
            to: *to,
            p: lower_hir_value_to_mpir(p),
            v: lower_hir_value_to_mpir(v),
        },
        HirOp::Call {
            callee_sid,
            inst,
            args,
        } => MpirOp::Call {
            callee_sid: callee_sid.clone(),
            inst: inst.clone(),
            args: args.iter().map(lower_hir_value_to_mpir).collect(),
        },
        HirOp::CallIndirect { callee, args } => MpirOp::CallIndirect {
            callee: lower_hir_value_to_mpir(callee),
            args: args.iter().map(lower_hir_value_to_mpir).collect(),
        },
        HirOp::CallVoidIndirect { callee, args } => MpirOp::CallVoidIndirect {
            callee: lower_hir_value_to_mpir(callee),
            args: args.iter().map(lower_hir_value_to_mpir).collect(),
        },
        HirOp::SuspendCall {
            callee_sid,
            inst,
            args,
        } => MpirOp::SuspendCall {
            callee_sid: callee_sid.clone(),
            inst: inst.clone(),
            args: args.iter().map(lower_hir_value_to_mpir).collect(),
        },
        HirOp::SuspendAwait { fut } => MpirOp::SuspendAwait {
            fut: lower_hir_value_to_mpir(fut),
        },
        HirOp::Phi { ty, incomings } => MpirOp::Phi {
            ty: *ty,
            incomings: incomings
                .iter()
                .map(|(block, value)| (*block, lower_hir_value_to_mpir(value)))
                .collect(),
        },
        HirOp::Share { v } => MpirOp::Share {
            v: lower_hir_value_to_mpir(v),
        },
        HirOp::CloneShared { v } => MpirOp::CloneShared {
            v: lower_hir_value_to_mpir(v),
        },
        HirOp::CloneWeak { v } => MpirOp::CloneWeak {
            v: lower_hir_value_to_mpir(v),
        },
        HirOp::WeakDowngrade { v } => MpirOp::WeakDowngrade {
            v: lower_hir_value_to_mpir(v),
        },
        HirOp::WeakUpgrade { v } => MpirOp::WeakUpgrade {
            v: lower_hir_value_to_mpir(v),
        },
        HirOp::EnumNew { variant, args } => MpirOp::EnumNew {
            variant: variant.clone(),
            args: args
                .iter()
                .map(|(name, value)| (name.clone(), lower_hir_value_to_mpir(value)))
                .collect(),
        },
        HirOp::EnumTag { v } => MpirOp::EnumTag {
            v: lower_hir_value_to_mpir(v),
        },
        HirOp::EnumPayload { variant, v } => MpirOp::EnumPayload {
            variant: variant.clone(),
            v: lower_hir_value_to_mpir(v),
        },
        HirOp::EnumIs { variant, v } => MpirOp::EnumIs {
            variant: variant.clone(),
            v: lower_hir_value_to_mpir(v),
        },
        HirOp::CallableCapture { fn_ref, captures } => MpirOp::CallableCapture {
            fn_ref: fn_ref.clone(),
            captures: captures
                .iter()
                .map(|(name, value)| (name.clone(), lower_hir_value_to_mpir(value)))
                .collect(),
        },
        HirOp::ArrNew { elem_ty, cap } => MpirOp::ArrNew {
            elem_ty: *elem_ty,
            cap: lower_hir_value_to_mpir(cap),
        },
        HirOp::ArrLen { arr } => MpirOp::ArrLen {
            arr: lower_hir_value_to_mpir(arr),
        },
        HirOp::ArrGet { arr, idx } => MpirOp::ArrGet {
            arr: lower_hir_value_to_mpir(arr),
            idx: lower_hir_value_to_mpir(idx),
        },
        HirOp::ArrSet { arr, idx, val } => MpirOp::ArrSet {
            arr: lower_hir_value_to_mpir(arr),
            idx: lower_hir_value_to_mpir(idx),
            val: lower_hir_value_to_mpir(val),
        },
        HirOp::ArrPush { arr, val } => MpirOp::ArrPush {
            arr: lower_hir_value_to_mpir(arr),
            val: lower_hir_value_to_mpir(val),
        },
        HirOp::ArrPop { arr } => MpirOp::ArrPop {
            arr: lower_hir_value_to_mpir(arr),
        },
        HirOp::ArrSlice { arr, start, end } => MpirOp::ArrSlice {
            arr: lower_hir_value_to_mpir(arr),
            start: lower_hir_value_to_mpir(start),
            end: lower_hir_value_to_mpir(end),
        },
        HirOp::ArrContains { arr, val } => MpirOp::ArrContains {
            arr: lower_hir_value_to_mpir(arr),
            val: lower_hir_value_to_mpir(val),
        },
        HirOp::ArrSort { arr } => MpirOp::ArrSort {
            arr: lower_hir_value_to_mpir(arr),
        },
        HirOp::ArrMap { arr, func } => MpirOp::ArrMap {
            arr: lower_hir_value_to_mpir(arr),
            func: lower_hir_value_to_mpir(func),
        },
        HirOp::ArrFilter { arr, func } => MpirOp::ArrFilter {
            arr: lower_hir_value_to_mpir(arr),
            func: lower_hir_value_to_mpir(func),
        },
        HirOp::ArrReduce { arr, init, func } => MpirOp::ArrReduce {
            arr: lower_hir_value_to_mpir(arr),
            init: lower_hir_value_to_mpir(init),
            func: lower_hir_value_to_mpir(func),
        },
        HirOp::ArrForeach { arr, func } => MpirOp::ArrForeach {
            arr: lower_hir_value_to_mpir(arr),
            func: lower_hir_value_to_mpir(func),
        },
        HirOp::MapNew { key_ty, val_ty } => MpirOp::MapNew {
            key_ty: *key_ty,
            val_ty: *val_ty,
        },
        HirOp::MapLen { map } => MpirOp::MapLen {
            map: lower_hir_value_to_mpir(map),
        },
        HirOp::MapGet { map, key } => MpirOp::MapGet {
            map: lower_hir_value_to_mpir(map),
            key: lower_hir_value_to_mpir(key),
        },
        HirOp::MapGetRef { map, key } => MpirOp::MapGetRef {
            map: lower_hir_value_to_mpir(map),
            key: lower_hir_value_to_mpir(key),
        },
        HirOp::MapSet { map, key, val } => MpirOp::MapSet {
            map: lower_hir_value_to_mpir(map),
            key: lower_hir_value_to_mpir(key),
            val: lower_hir_value_to_mpir(val),
        },
        HirOp::MapDelete { map, key } => MpirOp::MapDelete {
            map: lower_hir_value_to_mpir(map),
            key: lower_hir_value_to_mpir(key),
        },
        HirOp::MapContainsKey { map, key } => MpirOp::MapContainsKey {
            map: lower_hir_value_to_mpir(map),
            key: lower_hir_value_to_mpir(key),
        },
        HirOp::MapDeleteVoid { map, key } => MpirOp::MapDeleteVoid {
            map: lower_hir_value_to_mpir(map),
            key: lower_hir_value_to_mpir(key),
        },
        HirOp::MapKeys { map } => MpirOp::MapKeys {
            map: lower_hir_value_to_mpir(map),
        },
        HirOp::MapValues { map } => MpirOp::MapValues {
            map: lower_hir_value_to_mpir(map),
        },
        HirOp::StrConcat { a, b } => MpirOp::StrConcat {
            a: lower_hir_value_to_mpir(a),
            b: lower_hir_value_to_mpir(b),
        },
        HirOp::StrLen { s } => MpirOp::StrLen {
            s: lower_hir_value_to_mpir(s),
        },
        HirOp::StrEq { a, b } => MpirOp::StrEq {
            a: lower_hir_value_to_mpir(a),
            b: lower_hir_value_to_mpir(b),
        },
        HirOp::StrSlice { s, start, end } => MpirOp::StrSlice {
            s: lower_hir_value_to_mpir(s),
            start: lower_hir_value_to_mpir(start),
            end: lower_hir_value_to_mpir(end),
        },
        HirOp::StrBytes { s } => MpirOp::StrBytes {
            s: lower_hir_value_to_mpir(s),
        },
        HirOp::StrBuilderNew => MpirOp::StrBuilderNew,
        HirOp::StrBuilderAppendStr { b, s } => MpirOp::StrBuilderAppendStr {
            b: lower_hir_value_to_mpir(b),
            s: lower_hir_value_to_mpir(s),
        },
        HirOp::StrBuilderAppendI64 { b, v } => MpirOp::StrBuilderAppendI64 {
            b: lower_hir_value_to_mpir(b),
            v: lower_hir_value_to_mpir(v),
        },
        HirOp::StrBuilderAppendI32 { b, v } => MpirOp::StrBuilderAppendI32 {
            b: lower_hir_value_to_mpir(b),
            v: lower_hir_value_to_mpir(v),
        },
        HirOp::StrBuilderAppendF64 { b, v } => MpirOp::StrBuilderAppendF64 {
            b: lower_hir_value_to_mpir(b),
            v: lower_hir_value_to_mpir(v),
        },
        HirOp::StrBuilderAppendBool { b, v } => MpirOp::StrBuilderAppendBool {
            b: lower_hir_value_to_mpir(b),
            v: lower_hir_value_to_mpir(v),
        },
        HirOp::StrBuilderBuild { b } => MpirOp::StrBuilderBuild {
            b: lower_hir_value_to_mpir(b),
        },
        HirOp::StrParseI64 { s } => MpirOp::StrParseI64 {
            s: lower_hir_value_to_mpir(s),
        },
        HirOp::StrParseU64 { s } => MpirOp::StrParseU64 {
            s: lower_hir_value_to_mpir(s),
        },
        HirOp::StrParseF64 { s } => MpirOp::StrParseF64 {
            s: lower_hir_value_to_mpir(s),
        },
        HirOp::StrParseBool { s } => MpirOp::StrParseBool {
            s: lower_hir_value_to_mpir(s),
        },
        HirOp::JsonEncode { ty, v } => MpirOp::JsonEncode {
            ty: *ty,
            v: lower_hir_value_to_mpir(v),
        },
        HirOp::JsonDecode { ty, s } => MpirOp::JsonDecode {
            ty: *ty,
            s: lower_hir_value_to_mpir(s),
        },
        HirOp::GpuThreadId => MpirOp::GpuThreadId,
        HirOp::GpuWorkgroupId => MpirOp::GpuWorkgroupId,
        HirOp::GpuWorkgroupSize => MpirOp::GpuWorkgroupSize,
        HirOp::GpuGlobalId => MpirOp::GpuGlobalId,
        HirOp::GpuBufferLoad { buf, idx } => MpirOp::GpuBufferLoad {
            buf: lower_hir_value_to_mpir(buf),
            idx: lower_hir_value_to_mpir(idx),
        },
        HirOp::GpuBufferLen { buf } => MpirOp::GpuBufferLen {
            buf: lower_hir_value_to_mpir(buf),
        },
        HirOp::GpuShared { ty, size } => MpirOp::GpuShared {
            ty: *ty,
            size: lower_hir_value_to_mpir(size),
        },
        HirOp::GpuLaunch {
            device,
            kernel,
            groups,
            threads,
            args,
        } => MpirOp::GpuLaunch {
            device: lower_hir_value_to_mpir(device),
            kernel: kernel.clone(),
            groups: lower_hir_value_to_mpir(groups),
            threads: lower_hir_value_to_mpir(threads),
            args: args.iter().map(lower_hir_value_to_mpir).collect(),
        },
        HirOp::GpuLaunchAsync {
            device,
            kernel,
            groups,
            threads,
            args,
        } => MpirOp::GpuLaunchAsync {
            device: lower_hir_value_to_mpir(device),
            kernel: kernel.clone(),
            groups: lower_hir_value_to_mpir(groups),
            threads: lower_hir_value_to_mpir(threads),
            args: args.iter().map(lower_hir_value_to_mpir).collect(),
        },
        HirOp::Panic { msg } => MpirOp::Panic {
            msg: lower_hir_value_to_mpir(msg),
        },
    }
}

fn lower_hir_void_op_to_mpir(op: &HirOpVoid) -> MpirOpVoid {
    match op {
        HirOpVoid::CallVoid {
            callee_sid,
            inst,
            args,
        } => MpirOpVoid::CallVoid {
            callee_sid: callee_sid.clone(),
            inst: inst.clone(),
            args: args.iter().map(lower_hir_value_to_mpir).collect(),
        },
        HirOpVoid::CallVoidIndirect { callee, args } => MpirOpVoid::CallVoidIndirect {
            callee: lower_hir_value_to_mpir(callee),
            args: args.iter().map(lower_hir_value_to_mpir).collect(),
        },
        HirOpVoid::SetField { obj, field, value } => MpirOpVoid::SetField {
            obj: lower_hir_value_to_mpir(obj),
            field: field.clone(),
            value: lower_hir_value_to_mpir(value),
        },
        HirOpVoid::ArrSet { arr, idx, val } => MpirOpVoid::ArrSet {
            arr: lower_hir_value_to_mpir(arr),
            idx: lower_hir_value_to_mpir(idx),
            val: lower_hir_value_to_mpir(val),
        },
        HirOpVoid::ArrPush { arr, val } => MpirOpVoid::ArrPush {
            arr: lower_hir_value_to_mpir(arr),
            val: lower_hir_value_to_mpir(val),
        },
        HirOpVoid::ArrSort { arr } => MpirOpVoid::ArrSort {
            arr: lower_hir_value_to_mpir(arr),
        },
        HirOpVoid::ArrForeach { arr, func } => MpirOpVoid::ArrForeach {
            arr: lower_hir_value_to_mpir(arr),
            func: lower_hir_value_to_mpir(func),
        },
        HirOpVoid::MapSet { map, key, val } => MpirOpVoid::MapSet {
            map: lower_hir_value_to_mpir(map),
            key: lower_hir_value_to_mpir(key),
            val: lower_hir_value_to_mpir(val),
        },
        HirOpVoid::MapDeleteVoid { map, key } => MpirOpVoid::MapDeleteVoid {
            map: lower_hir_value_to_mpir(map),
            key: lower_hir_value_to_mpir(key),
        },
        HirOpVoid::StrBuilderAppendStr { b, s } => MpirOpVoid::StrBuilderAppendStr {
            b: lower_hir_value_to_mpir(b),
            s: lower_hir_value_to_mpir(s),
        },
        HirOpVoid::StrBuilderAppendI64 { b, v } => MpirOpVoid::StrBuilderAppendI64 {
            b: lower_hir_value_to_mpir(b),
            v: lower_hir_value_to_mpir(v),
        },
        HirOpVoid::StrBuilderAppendI32 { b, v } => MpirOpVoid::StrBuilderAppendI32 {
            b: lower_hir_value_to_mpir(b),
            v: lower_hir_value_to_mpir(v),
        },
        HirOpVoid::StrBuilderAppendF64 { b, v } => MpirOpVoid::StrBuilderAppendF64 {
            b: lower_hir_value_to_mpir(b),
            v: lower_hir_value_to_mpir(v),
        },
        HirOpVoid::StrBuilderAppendBool { b, v } => MpirOpVoid::StrBuilderAppendBool {
            b: lower_hir_value_to_mpir(b),
            v: lower_hir_value_to_mpir(v),
        },
        HirOpVoid::PtrStore { to, p, v } => MpirOpVoid::PtrStore {
            to: *to,
            p: lower_hir_value_to_mpir(p),
            v: lower_hir_value_to_mpir(v),
        },
        HirOpVoid::Panic { msg } => MpirOpVoid::Panic {
            msg: lower_hir_value_to_mpir(msg),
        },
        HirOpVoid::GpuBarrier => MpirOpVoid::GpuBarrier,
        HirOpVoid::GpuBufferStore { buf, idx, val } => MpirOpVoid::GpuBufferStore {
            buf: lower_hir_value_to_mpir(buf),
            idx: lower_hir_value_to_mpir(idx),
            val: lower_hir_value_to_mpir(val),
        },
    }
}

fn lower_hir_terminator_to_mpir(term: &HirTerminator) -> MpirTerminator {
    match term {
        HirTerminator::Ret(value) => {
            MpirTerminator::Ret(value.as_ref().map(lower_hir_value_to_mpir))
        }
        HirTerminator::Br(block_id) => MpirTerminator::Br(*block_id),
        HirTerminator::Cbr {
            cond,
            then_bb,
            else_bb,
        } => MpirTerminator::Cbr {
            cond: lower_hir_value_to_mpir(cond),
            then_bb: *then_bb,
            else_bb: *else_bb,
        },
        HirTerminator::Switch { val, arms, default } => MpirTerminator::Switch {
            val: lower_hir_value_to_mpir(val),
            arms: arms.clone(),
            default: *default,
        },
        HirTerminator::Unreachable => MpirTerminator::Unreachable,
    }
}

fn lower_hir_value_to_mpir(value: &HirValue) -> MpirValue {
    match value {
        HirValue::Local(local) => MpirValue::Local(*local),
        HirValue::Const(c) => MpirValue::Const(c.clone()),
    }
}

fn emit_contains(emit: &[String], needle: &str) -> bool {
    emit.iter().any(|kind| kind == needle)
}

fn emit_contains_any(emit: &[String], needles: &[&str]) -> bool {
    emit.iter()
        .any(|kind| needles.iter().any(|needle| kind == needle))
}

fn stage_module_output_path(
    config: &DriverConfig,
    module_idx: usize,
    module_count: usize,
    extension: &str,
) -> PathBuf {
    let stem = Path::new(&config.entry_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("main");
    let module_stem = if module_count <= 1 {
        stem.to_string()
    } else {
        format!("{stem}.{module_idx}")
    };
    PathBuf::from("target")
        .join(&config.target_triple)
        .join(config.profile.as_str())
        .join(format!("{module_stem}.{extension}"))
}

fn stage_graph_output_path(config: &DriverConfig, suffix: &str) -> PathBuf {
    let stem = Path::new(&config.entry_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("main");
    PathBuf::from("target")
        .join(&config.target_triple)
        .join(config.profile.as_str())
        .join(format!("{stem}.{suffix}.json"))
}

fn stage_mpdbg_output_path(config: &DriverConfig) -> PathBuf {
    let stem = Path::new(&config.entry_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("main");
    PathBuf::from("target")
        .join(&config.target_triple)
        .join(config.profile.as_str())
        .join(format!("{stem}.mpdbg"))
}

fn stage_mms_index_output_path(config: &DriverConfig) -> PathBuf {
    let stem = Path::new(&config.entry_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("main");
    PathBuf::from("target")
        .join(&config.target_triple)
        .join(config.profile.as_str())
        .join(format!("{stem}.mms_index.json"))
}

fn write_text_artifact(path: &Path, text: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("Could not create '{}': {}", parent.display(), err))?;
    }
    fs::write(path, text).map_err(|err| format!("Could not write '{}': {}", path.display(), err))
}

#[derive(Clone, Debug)]
struct CExternItem {
    name: String,
    params: Vec<(String, String)>,
    ret_ty: String,
}

fn parse_c_header_functions(header: &str) -> Vec<CExternItem> {
    let block_comments = Regex::new(r"(?s)/\*.*?\*/").expect("valid regex");
    let line_comments = Regex::new(r"(?m)//.*$").expect("valid regex");
    let cleaned = block_comments.replace_all(header, "");
    let cleaned = line_comments.replace_all(&cleaned, "");

    let declaration =
        Regex::new(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_\s\*\t]*?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^;{}]*)\)\s*;")
            .expect("valid regex");
    let mut out = Vec::new();
    for captures in declaration.captures_iter(&cleaned) {
        let ret_raw = captures.get(1).map(|m| m.as_str()).unwrap_or("");
        let name = captures.get(2).map(|m| m.as_str()).unwrap_or("").trim();
        let params_raw = captures.get(3).map(|m| m.as_str()).unwrap_or("");
        if name.is_empty() {
            continue;
        }
        out.push(CExternItem {
            name: name.to_string(),
            params: parse_c_params(params_raw),
            ret_ty: map_c_type_to_magpie(ret_raw),
        });
    }
    out
}

fn parse_c_params(params: &str) -> Vec<(String, String)> {
    let params = params.trim();
    if params.is_empty() || params == "void" {
        return Vec::new();
    }

    params
        .split(',')
        .enumerate()
        .filter_map(|(idx, raw)| {
            let raw = raw.trim();
            if raw.is_empty() || raw == "..." {
                return None;
            }
            let compact = raw.split_whitespace().collect::<Vec<_>>().join(" ");
            let (ty_raw, name_raw) = split_c_param_type_and_name(&compact);
            let ty = map_c_type_to_magpie(ty_raw);
            let name = sanitize_c_ident(name_raw.unwrap_or(""), idx);
            Some((name, ty))
        })
        .collect()
}

fn split_c_param_type_and_name(input: &str) -> (&str, Option<&str>) {
    let input = input.trim();
    let mut last_ident_start = None;
    for (idx, ch) in input.char_indices().rev() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            last_ident_start = Some(idx);
        } else if last_ident_start.is_some() {
            break;
        }
    }

    let Some(start) = last_ident_start else {
        return (input, None);
    };
    let ident = &input[start..];
    if ident.is_empty() {
        return (input, None);
    }
    let type_part = input[..start].trim_end();
    if type_part.is_empty() {
        (input, None)
    } else {
        (type_part, Some(ident))
    }
}

fn sanitize_c_ident(name: &str, idx: usize) -> String {
    let mut out = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        return format!("arg{idx}");
    }
    let starts_valid = out
        .chars()
        .next()
        .map(|ch| ch.is_ascii_alphabetic() || ch == '_')
        .unwrap_or(false);
    if starts_valid {
        out
    } else {
        format!("arg{idx}_{out}")
    }
}

fn map_c_type_to_magpie(raw_ty: &str) -> String {
    let normalized = raw_ty
        .replace('\t', " ")
        .replace("const ", "")
        .replace("volatile ", "")
        .replace("struct ", "")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_ascii_lowercase();
    let pointer_count = normalized.matches('*').count();
    let base = normalized.replace('*', "").trim().to_string();

    if pointer_count > 0 {
        if base == "char" || base == "signed char" || base == "unsigned char" {
            return "Str".to_string();
        }
        return "rawptr<u8>".to_string();
    }

    match base.as_str() {
        "void" => "unit".to_string(),
        "bool" | "_bool" => "bool".to_string(),
        "char" | "signed char" => "i8".to_string(),
        "unsigned char" => "u8".to_string(),
        "short" | "short int" | "signed short" | "signed short int" => "i16".to_string(),
        "unsigned short" | "unsigned short int" => "u16".to_string(),
        "int" | "signed" | "signed int" => "i32".to_string(),
        "unsigned" | "unsigned int" => "u32".to_string(),
        "long" | "long int" | "signed long" | "signed long int" => "i64".to_string(),
        "unsigned long" | "unsigned long int" => "u64".to_string(),
        "long long" | "long long int" | "signed long long" | "signed long long int" => {
            "i64".to_string()
        }
        "unsigned long long" | "unsigned long long int" => "u64".to_string(),
        "float" => "f32".to_string(),
        "double" | "long double" => "f64".to_string(),
        _ => "i32".to_string(),
    }
}

fn render_extern_module(module_name: &str, items: &[CExternItem]) -> String {
    let mut out = String::new();
    out.push_str("module ffi.imported\n");
    out.push_str("exports { }\n");
    out.push_str("imports { }\n");
    out.push_str("digest \"0000000000000000\"\n\n");
    out.push_str(&format!("extern \"C\" module {module_name} {{\n"));
    for item in items {
        let params = item
            .params
            .iter()
            .map(|(name, ty)| format!("%{name}: {ty}"))
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!(
            "  fn @{}({}) -> {} attrs {{ link_name=\"{}\" }}\n",
            item.name, params, item.ret_ty, item.name
        ));
    }
    out.push_str("}\n");
    out
}

fn stage_link_output_path(config: &DriverConfig) -> PathBuf {
    let stem = Path::new(&config.entry_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("main");
    let base = PathBuf::from("target")
        .join(&config.target_triple)
        .join(config.profile.as_str());
    let is_windows = is_windows_target(&config.target_triple);
    if emit_contains(&config.emit, "shared-lib") && !emit_contains(&config.emit, "exe") {
        let ext = if is_windows {
            ".dll"
        } else if is_darwin_target(&config.target_triple) {
            ".dylib"
        } else {
            ".so"
        };
        base.join(format!("lib{stem}{ext}"))
    } else {
        base.join(format!("{stem}{}", if is_windows { ".exe" } else { "" }))
    }
}

fn link_via_llc_and_linker(
    config: &DriverConfig,
    llvm_ir_paths: &[PathBuf],
    output_path: &Path,
    link_shared: bool,
) -> Result<Vec<PathBuf>, String> {
    if llvm_ir_paths.is_empty() {
        return Err("no LLVM IR inputs were generated".to_string());
    }
    if !command_available("llc") {
        return Err("llc is not available in PATH".to_string());
    }
    let linker = if command_available("cc") {
        "cc"
    } else if command_available("clang") {
        "clang"
    } else {
        return Err("neither cc nor clang is available in PATH".to_string());
    };
    ensure_parent_dir(output_path)?;

    let obj_ext = if is_windows_target(&config.target_triple) {
        "obj"
    } else {
        "o"
    };
    let mut objects = Vec::with_capacity(llvm_ir_paths.len());
    for llvm_ir in llvm_ir_paths {
        let obj_path = llvm_ir.with_extension(obj_ext);
        ensure_parent_dir(&obj_path)?;
        run_command(
            "llc",
            &[
                format!("-mtriple={}", config.target_triple),
                "-filetype=obj".to_string(),
                "-o".to_string(),
                obj_path.to_string_lossy().to_string(),
                llvm_ir.to_string_lossy().to_string(),
            ],
        )?;
        objects.push(obj_path);
    }

    let mut args = Vec::new();
    if linker == "clang" {
        args.push(format!("--target={}", config.target_triple));
    }
    if link_shared {
        args.push("-shared".to_string());
    }
    args.push("-o".to_string());
    args.push(output_path.to_string_lossy().to_string());
    args.extend(
        objects
            .iter()
            .map(|path| path.to_string_lossy().to_string())
            .collect::<Vec<_>>(),
    );
    args.extend(runtime_linker_args(config));
    run_command(linker, &args)?;
    Ok(objects)
}

fn link_via_clang_ir(
    config: &DriverConfig,
    llvm_ir_paths: &[PathBuf],
    output_path: &Path,
    link_shared: bool,
) -> Result<(), String> {
    if llvm_ir_paths.is_empty() {
        return Err("no LLVM IR inputs were generated".to_string());
    }
    if !command_available("clang") {
        return Err("clang is not available in PATH".to_string());
    }
    ensure_parent_dir(output_path)?;

    let mut args = Vec::new();
    args.push(format!("--target={}", config.target_triple));
    if link_shared {
        args.push("-shared".to_string());
    }
    args.push("-x".to_string());
    args.push("ir".to_string());
    args.push("-o".to_string());
    args.push(output_path.to_string_lossy().to_string());
    args.extend(
        llvm_ir_paths
            .iter()
            .map(|path| path.to_string_lossy().to_string())
            .collect::<Vec<_>>(),
    );
    args.extend(runtime_linker_args(config));
    run_command("clang", &args)
}

fn runtime_linker_args(config: &DriverConfig) -> Vec<String> {
    let runtime_dir = find_runtime_lib_dir(config).unwrap_or_else(|| {
        PathBuf::from("target")
            .join(&config.target_triple)
            .join(config.profile.as_str())
    });
    let mut args = vec![
        format!("-L{}", runtime_dir.to_string_lossy()),
        "-lmagpie_rt".to_string(),
    ];
    if !is_windows_target(&config.target_triple) {
        args.push("-lpthread".to_string());
        if is_darwin_target(&config.target_triple) {
            args.push("-lSystem".to_string());
        } else {
            args.push("-ldl".to_string());
        }
        args.push("-lm".to_string());
    }
    args
}

fn find_runtime_lib_dir(config: &DriverConfig) -> Option<PathBuf> {
    let names = if is_windows_target(&config.target_triple) {
        vec!["magpie_rt.lib", "libmagpie_rt.lib"]
    } else {
        vec!["libmagpie_rt.a", "libmagpie_rt.dylib", "libmagpie_rt.so"]
    };
    runtime_library_search_paths(config)
        .into_iter()
        .find(|dir| names.iter().any(|name| dir.join(name).is_file()))
}

fn runtime_library_search_paths(config: &DriverConfig) -> Vec<PathBuf> {
    let target_root = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target"));
    let mut out = Vec::new();
    let mut push_unique = |path: PathBuf| {
        if !out.contains(&path) {
            out.push(path);
        }
    };
    push_unique(
        target_root
            .join(&config.target_triple)
            .join(config.profile.as_str()),
    );
    push_unique(
        target_root
            .join(&config.target_triple)
            .join(config.profile.as_str())
            .join("deps"),
    );
    push_unique(target_root.join(config.profile.as_str()));
    push_unique(target_root.join(config.profile.as_str()).join("deps"));
    out
}

fn ensure_parent_dir(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("Could not create '{}': {}", parent.display(), err))?;
    }
    Ok(())
}

fn command_available(bin: &str) -> bool {
    Command::new(bin)
        .arg("--version")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

fn run_command(program: &str, args: &[String]) -> Result<(), String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|err| format!("Failed to run '{}': {}", format_command(program, args), err))?;
    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let mut details = Vec::new();
    if !stderr.is_empty() {
        details.push(format!("stderr: {}", stderr));
    }
    if !stdout.is_empty() {
        details.push(format!("stdout: {}", stdout));
    }
    let detail = if details.is_empty() {
        "no process output".to_string()
    } else {
        details.join(" | ")
    };
    Err(format!(
        "Command '{}' failed with status {} ({})",
        format_command(program, args),
        output.status,
        detail
    ))
}

fn format_command(program: &str, args: &[String]) -> String {
    if args.is_empty() {
        program.to_string()
    } else {
        format!("{} {}", program, args.join(" "))
    }
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

        let Some(mut ast) = parsed else {
            continue;
        };

        if fix_meta {
            synthesize_meta_blocks(&mut ast);
        }

        let stage2_start = Instant::now();
        let mut formatted = format_csnf(&ast);
        formatted = update_digest(&formatted);
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

/// `magpie doc` entry-point: parse source files + emit one `.mpd` per module.
pub fn generate_docs(paths: &[String]) -> BuildResult {
    let mut result = BuildResult::default();
    let mut stage_read_lex_parse = 0_u64;
    let mut stage_mpd_generate = 0_u64;
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
        let mpd = render_mpd(&ast);
        stage_mpd_generate += elapsed_ms(stage2_start);

        let stage3_start = Instant::now();
        let out_path = Path::new(path).with_extension("mpd");
        if let Err(err) = ensure_parent_dir(&out_path) {
            result.diagnostics.push(simple_diag(
                "MPP0001",
                Severity::Error,
                "failed to prepare output path",
                err,
            ));
            stage_write_back += elapsed_ms(stage3_start);
            continue;
        }
        match fs::write(&out_path, mpd) {
            Ok(()) => result
                .artifacts
                .push(out_path.to_string_lossy().to_string()),
            Err(err) => {
                result.diagnostics.push(simple_diag(
                    "MPP0001",
                    Severity::Error,
                    "failed to write .mpd file",
                    format!("Could not write '{}': {}", out_path.display(), err),
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
        .insert("stage2_mpd_generate".to_string(), stage_mpd_generate);
    result
        .timing_ms
        .insert("stage3_write_back".to_string(), stage_write_back);
    result.success = !has_errors(&result.diagnostics);
    result
}

fn synthesize_meta_blocks(ast: &mut AstFile) {
    let mut calls_by_decl: HashMap<usize, Vec<String>> = HashMap::new();
    for (idx, decl) in ast.decls.iter().enumerate() {
        let Some(func) = ast_fn_decl(&decl.node) else {
            continue;
        };
        let mut calls = collect_direct_calls(func);
        calls.remove(&func.name);
        calls_by_decl.insert(idx, calls.into_iter().collect());
    }

    for (idx, decl) in ast.decls.iter_mut().enumerate() {
        let Some(func) = ast_fn_decl_mut(&mut decl.node) else {
            continue;
        };
        let uses = calls_by_decl.remove(&idx).unwrap_or_default();
        let (effects, cost) = func
            .meta
            .as_ref()
            .map(|meta| (meta.effects.clone(), meta.cost.clone()))
            .unwrap_or_else(|| (Vec::new(), Vec::new()));
        func.meta = Some(AstFnMeta {
            uses,
            effects,
            cost,
        });
    }
}

fn collect_direct_calls(func: &AstFnDecl) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for block in &func.blocks {
        for instr in &block.node.instrs {
            collect_calls_from_instr(&instr.node, &mut out);
        }
    }
    out
}

fn collect_calls_from_instr(instr: &AstInstr, out: &mut BTreeSet<String>) {
    match instr {
        AstInstr::Assign { op, .. } => collect_calls_from_op(op, out),
        AstInstr::Void(op) => collect_calls_from_void_op(op, out),
        AstInstr::UnsafeBlock(instrs) => {
            for instr in instrs {
                collect_calls_from_instr(&instr.node, out);
            }
        }
    }
}

fn collect_calls_from_op(op: &AstOp, out: &mut BTreeSet<String>) {
    match op {
        AstOp::Call { callee, .. } | AstOp::Try { callee, .. } | AstOp::SuspendCall { callee, .. } => {
            if !callee.is_empty() {
                out.insert(callee.clone());
            }
        }
        AstOp::CallableCapture { fn_ref, .. } => {
            if !fn_ref.is_empty() {
                out.insert(fn_ref.clone());
            }
        }
        _ => {}
    }
}

fn collect_calls_from_void_op(op: &AstOpVoid, out: &mut BTreeSet<String>) {
    if let AstOpVoid::CallVoid { callee, .. } = op {
        if !callee.is_empty() {
            out.insert(callee.clone());
        }
    }
}

fn ast_fn_decl(decl: &AstDecl) -> Option<&AstFnDecl> {
    match decl {
        AstDecl::Fn(func) | AstDecl::AsyncFn(func) | AstDecl::UnsafeFn(func) => Some(func),
        AstDecl::GpuFn(gpu) => Some(&gpu.inner),
        _ => None,
    }
}

fn ast_fn_decl_mut(decl: &mut AstDecl) -> Option<&mut AstFnDecl> {
    match decl {
        AstDecl::Fn(func) | AstDecl::AsyncFn(func) | AstDecl::UnsafeFn(func) => Some(func),
        AstDecl::GpuFn(gpu) => Some(&mut gpu.inner),
        _ => None,
    }
}

fn render_mpd(ast: &AstFile) -> String {
    let mut out = String::new();
    out.push_str("module ");
    out.push_str(&ast.header.node.module_path.node.to_string());
    out.push('\n');

    out.push_str("exports { ");
    let exports = ast
        .header
        .node
        .exports
        .iter()
        .map(|item| match &item.node {
            ExportItem::Fn(name) | ExportItem::Type(name) => name.clone(),
        })
        .collect::<Vec<_>>()
        .join(", ");
    out.push_str(&exports);
    out.push_str(" }\n\n");

    out.push_str("types:\n");
    let mut wrote_type = false;
    for decl in &ast.decls {
        match &decl.node {
            AstDecl::HeapStruct(s) => {
                wrote_type = true;
                out.push_str("  heap struct ");
                out.push_str(&s.name);
                out.push('\n');
                out.push_str("    signature: ");
                out.push_str(&render_struct_signature("heap struct", s));
                out.push('\n');
                push_doc_lines(&mut out, "    ", s.doc.as_deref());
            }
            AstDecl::ValueStruct(s) => {
                wrote_type = true;
                out.push_str("  value struct ");
                out.push_str(&s.name);
                out.push('\n');
                out.push_str("    signature: ");
                out.push_str(&render_struct_signature("value struct", s));
                out.push('\n');
                push_doc_lines(&mut out, "    ", s.doc.as_deref());
            }
            AstDecl::HeapEnum(e) => {
                wrote_type = true;
                out.push_str("  heap enum ");
                out.push_str(&e.name);
                out.push('\n');
                out.push_str("    signature: ");
                out.push_str(&render_enum_signature("heap enum", e));
                out.push('\n');
                push_doc_lines(&mut out, "    ", e.doc.as_deref());
            }
            AstDecl::ValueEnum(e) => {
                wrote_type = true;
                out.push_str("  value enum ");
                out.push_str(&e.name);
                out.push('\n');
                out.push_str("    signature: ");
                out.push_str(&render_enum_signature("value enum", e));
                out.push('\n');
                push_doc_lines(&mut out, "    ", e.doc.as_deref());
            }
            _ => {}
        }
    }
    if !wrote_type {
        out.push_str("  (none)\n");
    }

    out.push('\n');
    out.push_str("functions:\n");
    let mut wrote_fn = false;
    for decl in &ast.decls {
        match &decl.node {
            AstDecl::Fn(f) => {
                wrote_fn = true;
                out.push_str("  ");
                out.push_str(&render_fn_signature("fn", f, None));
                out.push('\n');
                push_doc_lines(&mut out, "    ", f.doc.as_deref());
            }
            AstDecl::AsyncFn(f) => {
                wrote_fn = true;
                out.push_str("  ");
                out.push_str(&render_fn_signature("async fn", f, None));
                out.push('\n');
                push_doc_lines(&mut out, "    ", f.doc.as_deref());
            }
            AstDecl::UnsafeFn(f) => {
                wrote_fn = true;
                out.push_str("  ");
                out.push_str(&render_fn_signature("unsafe fn", f, None));
                out.push('\n');
                push_doc_lines(&mut out, "    ", f.doc.as_deref());
            }
            AstDecl::GpuFn(gpu) => {
                wrote_fn = true;
                out.push_str("  ");
                out.push_str(&render_fn_signature("gpu fn", &gpu.inner, Some(&gpu.target)));
                out.push('\n');
                push_doc_lines(&mut out, "    ", gpu.inner.doc.as_deref());
            }
            _ => {}
        }
    }
    if !wrote_fn {
        out.push_str("  (none)\n");
    }

    out
}

fn render_fn_signature(prefix: &str, func: &AstFnDecl, target: Option<&str>) -> String {
    let mut out = String::new();
    out.push_str(prefix);
    out.push(' ');
    out.push_str(&func.name);
    out.push('(');
    out.push_str(
        &func
            .params
            .iter()
            .map(|param| format!("%{}: {}", param.name, render_ast_type(&param.ty.node)))
            .collect::<Vec<_>>()
            .join(", "),
    );
    out.push(')');
    out.push_str(" -> ");
    out.push_str(&render_ast_type(&func.ret_ty.node));
    if let Some(target) = target {
        out.push_str(" target(");
        out.push_str(target);
        out.push(')');
    }
    out
}

fn render_struct_signature(prefix: &str, decl: &magpie_ast::AstStructDecl) -> String {
    let fields = decl
        .fields
        .iter()
        .map(|field| format!("{}: {}", field.name, render_ast_type(&field.ty.node)))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{prefix} {} {{ {} }}", decl.name, fields)
}

fn render_enum_signature(prefix: &str, decl: &magpie_ast::AstEnumDecl) -> String {
    let variants = decl
        .variants
        .iter()
        .map(|variant| {
            if variant.fields.is_empty() {
                variant.name.clone()
            } else {
                let args = variant
                    .fields
                    .iter()
                    .map(|field| format!("{}: {}", field.name, render_ast_type(&field.ty.node)))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{}({})", variant.name, args)
            }
        })
        .collect::<Vec<_>>()
        .join(" | ");
    format!("{prefix} {} {{ {} }}", decl.name, variants)
}

fn render_ast_type(ty: &AstType) -> String {
    let base = render_ast_base_type(&ty.base);
    match ty.ownership {
        Some(magpie_ast::OwnershipMod::Shared) => format!("shared {}", base),
        Some(magpie_ast::OwnershipMod::Borrow) => format!("borrow {}", base),
        Some(magpie_ast::OwnershipMod::MutBorrow) => format!("mutborrow {}", base),
        Some(magpie_ast::OwnershipMod::Weak) => format!("weak {}", base),
        None => base,
    }
}

fn render_ast_base_type(base: &AstBaseType) -> String {
    match base {
        AstBaseType::Prim(name) => name.clone(),
        AstBaseType::Named { path, name, targs } => {
            let mut out = if let Some(path) = path {
                if path.segments.is_empty() {
                    name.clone()
                } else {
                    format!("{}.{}", path.to_string(), name)
                }
            } else {
                name.clone()
            };
            if !targs.is_empty() {
                out.push('<');
                out.push_str(
                    &targs
                        .iter()
                        .map(render_ast_type)
                        .collect::<Vec<_>>()
                        .join(", "),
                );
                out.push('>');
            }
            out
        }
        AstBaseType::Builtin(b) => render_builtin_type(b),
        AstBaseType::Callable { sig_ref } => format!("TCallable<{}>", sig_ref),
        AstBaseType::RawPtr(inner) => format!("rawptr<{}>", render_ast_type(inner)),
    }
}

fn render_builtin_type(builtin: &AstBuiltinType) -> String {
    match builtin {
        AstBuiltinType::Str => "str".to_string(),
        AstBuiltinType::Array(inner) => format!("Array<{}>", render_ast_type(inner)),
        AstBuiltinType::Map(key, val) => {
            format!("Map<{}, {}>", render_ast_type(key), render_ast_type(val))
        }
        AstBuiltinType::TOption(inner) => format!("TOption<{}>", render_ast_type(inner)),
        AstBuiltinType::TResult(ok, err) => {
            format!("TResult<{}, {}>", render_ast_type(ok), render_ast_type(err))
        }
        AstBuiltinType::TStrBuilder => "TStrBuilder".to_string(),
        AstBuiltinType::TMutex(inner) => format!("TMutex<{}>", render_ast_type(inner)),
        AstBuiltinType::TRwLock(inner) => format!("TRwLock<{}>", render_ast_type(inner)),
        AstBuiltinType::TCell(inner) => format!("TCell<{}>", render_ast_type(inner)),
        AstBuiltinType::TFuture(inner) => format!("TFuture<{}>", render_ast_type(inner)),
        AstBuiltinType::TChannelSend(inner) => format!("TChannelSend<{}>", render_ast_type(inner)),
        AstBuiltinType::TChannelRecv(inner) => format!("TChannelRecv<{}>", render_ast_type(inner)),
    }
}

fn push_doc_lines(out: &mut String, indent: &str, doc: Option<&str>) {
    out.push_str(indent);
    out.push_str("doc:");
    match doc {
        Some(doc) if !doc.trim().is_empty() => {
            out.push('\n');
            for line in doc.lines() {
                out.push_str(indent);
                out.push_str("  ");
                out.push_str(line.trim_end());
                out.push('\n');
            }
        }
        _ => {
            out.push_str(" (none)\n");
        }
    }
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
    fs::create_dir_all(base.join(".magpie")).map_err(|e| {
        format!(
            "Failed to create '{}': {}",
            base.join(".magpie").display(),
            e
        )
    })?;

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
        let planned = planned_artifacts(config, &mut result.diagnostics);
        if result.artifacts.is_empty() {
            result.artifacts = planned;
        } else {
            for artifact in planned {
                if Path::new(&artifact).exists() && !result.artifacts.contains(&artifact) {
                    result.artifacts.push(artifact);
                }
            }
        }
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

fn elapsed_ms(start: Instant) -> u64 {
    start.elapsed().as_millis().try_into().unwrap_or(u64::MAX)
}

fn has_errors(diags: &[Diagnostic]) -> bool {
    diags.iter().any(|d| matches!(d.severity, Severity::Error))
}

fn simple_diag(
    code: &str,
    severity: Severity,
    title: impl Into<String>,
    message: impl Into<String>,
) -> Diagnostic {
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
    let is_windows = is_windows_target(&config.target_triple);
    let is_darwin = is_darwin_target(&config.target_triple);

    for emit in &config.emit {
        let path = match emit.as_str() {
            "llvm-ir" => Some(base.join(format!("{stem}.ll"))),
            "llvm-bc" => Some(base.join(format!("{stem}.bc"))),
            "object" => {
                Some(base.join(format!("{stem}{}", if is_windows { ".obj" } else { ".o" })))
            }
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
            "mpdbg" => Some(base.join(format!("{stem}.mpdbg"))),
            "symgraph" => Some(base.join(format!("{stem}.symgraph.json"))),
            "depsgraph" => Some(base.join(format!("{stem}.depsgraph.json"))),
            "ownershipgraph" => Some(base.join(format!("{stem}.ownershipgraph.json"))),
            "cfggraph" => Some(base.join(format!("{stem}.cfggraph.json"))),
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

fn is_windows_target(target_triple: &str) -> bool {
    target_triple.contains("windows")
}

fn is_darwin_target(target_triple: &str) -> bool {
    target_triple.contains("darwin")
        || target_triple.contains("apple")
        || target_triple.contains("macos")
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn test_create_project() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        let project_dir = std::env::temp_dir().join(format!(
            "magpie_driver_create_project_{}_{}",
            std::process::id(),
            nonce
        ));
        let project_path = project_dir.to_string_lossy().into_owned();

        if project_dir.exists() {
            std::fs::remove_dir_all(&project_dir).expect("failed to clear pre-existing test dir");
        }

        create_project(&project_path).expect("create_project should succeed");

        assert!(project_dir.join("Magpie.toml").is_file());
        assert!(project_dir.join("Magpie.lock").is_file());
        assert!(project_dir.join("src/main.mp").is_file());
        assert!(project_dir.join("tests").is_dir());
        assert!(project_dir.join(".magpie").is_dir());

        std::fs::remove_dir_all(&project_dir).expect("failed to clean up test dir");
    }

    #[test]
    fn test_parse_c_header_functions_and_render_extern_module() {
        let header = r#"
            int add(int a, int b);
            const char* version(void);
            void write_bytes(unsigned char* data, unsigned long len);
        "#;

        let funcs = parse_c_header_functions(header);
        assert_eq!(funcs.len(), 3);
        assert_eq!(funcs[0].name, "add");
        assert_eq!(funcs[0].ret_ty, "i32");
        assert_eq!(funcs[0].params, vec![("a".to_string(), "i32".to_string()), ("b".to_string(), "i32".to_string())]);
        assert_eq!(funcs[1].ret_ty, "Str");
        assert_eq!(funcs[2].params[0].1, "Str");

        let rendered = render_extern_module("ffi_import", &funcs);
        assert!(rendered.contains("extern \"C\" module ffi_import"));
        assert!(rendered.contains("fn @add(%a: i32, %b: i32) -> i32"));
        assert!(rendered.contains("fn @version() -> Str"));
    }
}
