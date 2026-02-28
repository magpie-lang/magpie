//! Magpie compiler driver (§5.2, §22, §26.1).
#![allow(clippy::field_reassign_with_default)]

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
use magpie_diag::{
    canonical_json_encode, codes, enforce_budget, Diagnostic, DiagnosticBag, OutputEnvelope,
    Severity, SuggestedFix, TokenBudget,
};
use magpie_gpu::{compute_kernel_layout, generate_kernel_registry_ir, generate_spirv_with_layout};
use magpie_hir::{
    verify_hir, BlockId, HirBlock, HirConst, HirConstLit, HirFunction, HirInstr, HirModule, HirOp,
    HirOpVoid, HirTerminator, HirTypeDecl, HirValue, LocalId,
};
use magpie_lex::lex;
use magpie_memory::{build_index_with_sources, MmsItem, MmsSourceFingerprint};
use magpie_mpir::{
    print_mpir, verify_mpir, MpirBlock, MpirFn, MpirInstr, MpirLocalDecl, MpirModule, MpirOp,
    MpirOpVoid, MpirTerminator, MpirTypeTable, MpirValue,
};
use magpie_own::check_ownership;
use magpie_parse::parse_file;
use magpie_sema::{
    check_trait_impls, check_v01_restrictions, generate_sid, lower_to_hir, resolve_modules,
    typecheck_module, ResolvedModule,
};
use magpie_types::{fixed_type_ids, Sid, TypeCtx, TypeId};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[allow(dead_code)]
#[path = "../../magpie_codegen_llvm/src/lib.rs"]
mod magpie_codegen_llvm;

const DEFAULT_MAX_ERRORS: usize = 20;
pub const DEFAULT_LLM_TOKEN_BUDGET: u32 = 12_000;
pub const DEFAULT_LLM_TOKENIZER: &str = "approx:utf8_4chars";
pub const DEFAULT_LLM_BUDGET_POLICY: &str = "balanced";

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
    pub cache_dir: Option<String>,
    pub jobs: Option<u32>,
    pub offline: bool,
    pub no_default_features: bool,
    pub max_errors: usize,
    pub llm_mode: bool,
    pub token_budget: Option<u32>,
    pub llm_tokenizer: Option<String>,
    pub llm_budget_policy: Option<String>,
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
            cache_dir: None,
            jobs: None,
            offline: false,
            no_default_features: false,
            max_errors: DEFAULT_MAX_ERRORS,
            llm_mode: false,
            token_budget: None,
            llm_tokenizer: None,
            llm_budget_policy: None,
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
    pub timing_ms: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TestResult {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub test_names: Vec<(String, bool)>,
}

#[derive(Clone, Debug)]
struct GpuKernelDecl {
    sid: Sid,
    target: String,
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
        graphs: output_envelope_graphs(&result.artifacts),
        timing_ms: serde_json::to_value(&result.timing_ms).unwrap_or_else(|_| json!({})),
        llm_budget: llm_budget_value(config),
    }
}

fn default_output_envelope_graphs() -> serde_json::Value {
    json!({
        "symbols": {},
        "deps": {},
        "ownership": {},
        "cfg": {},
    })
}

fn output_graph_slot_for_artifact(artifact: &str) -> Option<&'static str> {
    if artifact.ends_with(".symgraph.json") {
        Some("symbols")
    } else if artifact.ends_with(".depsgraph.json") {
        Some("deps")
    } else if artifact.ends_with(".ownershipgraph.json") {
        Some("ownership")
    } else if artifact.ends_with(".cfggraph.json") {
        Some("cfg")
    } else {
        None
    }
}

fn output_envelope_graphs(artifacts: &[String]) -> serde_json::Value {
    let mut graphs = default_output_envelope_graphs();
    let Some(graph_map) = graphs.as_object_mut() else {
        return default_output_envelope_graphs();
    };

    for artifact in artifacts {
        let Some(slot) = output_graph_slot_for_artifact(artifact) else {
            continue;
        };
        let Ok(raw) = fs::read_to_string(artifact) else {
            continue;
        };
        let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&raw) else {
            continue;
        };
        graph_map.insert(slot.to_string(), parsed);
    }

    graphs
}

pub fn apply_llm_budget(config: &DriverConfig, envelope: &mut OutputEnvelope) {
    let Some(mut budget) = effective_budget_config(config) else {
        return;
    };

    if budget.tokenizer != DEFAULT_LLM_TOKENIZER {
        envelope.diagnostics.push(Diagnostic {
            code: codes::MPL0802.to_string(),
            severity: Severity::Warning,
            title: "tokenizer fallback used".to_string(),
            primary_span: None,
            secondary_spans: Vec::new(),
            message: format!(
                "Tokenizer '{}' is unavailable; falling back to '{}'.",
                budget.tokenizer, DEFAULT_LLM_TOKENIZER
            ),
            explanation_md: None,
            why: None,
            suggested_fixes: Vec::new(),
            rag_bundle: Vec::new(),
            related_docs: Vec::new(),
        });
        budget.tokenizer = DEFAULT_LLM_TOKENIZER.to_string();
    }

    enforce_budget(envelope, &budget);
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
            result
                .artifacts
                .push(out_path.to_string_lossy().to_string());
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

fn resolve_dependencies_for_build(
    entry_path: &str,
    offline: bool,
) -> Result<Option<PathBuf>, String> {
    let Some(manifest_path) = discover_manifest_path(entry_path) else {
        return Ok(None);
    };
    let manifest = magpie_pkg::parse_manifest(&manifest_path)?;
    let lock = magpie_pkg::resolve_deps(&manifest, offline)?;
    let lock_path = manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("Magpie.lock");
    magpie_pkg::write_lockfile(&lock, &lock_path)?;
    Ok(Some(lock_path))
}

fn is_default_entry_path(entry_path: &str) -> bool {
    let path = Path::new(entry_path);
    let file_name = path.file_name().and_then(|value| value.to_str());
    let parent_name = path
        .parent()
        .and_then(|value| value.file_name())
        .and_then(|value| value.to_str());
    file_name == Some("main.mp") && parent_name == Some("src")
}

fn discover_manifest_from_cwd() -> Option<PathBuf> {
    let mut current = std::env::current_dir().ok()?;
    loop {
        let candidate = current.join("Magpie.toml");
        if candidate.is_file() {
            return Some(candidate);
        }
        if !current.pop() {
            break;
        }
    }
    None
}

fn resolve_manifest_build_entry(entry_path: &str) -> Option<String> {
    let manifest_path = discover_manifest_path(entry_path).or_else(discover_manifest_from_cwd)?;
    let manifest = magpie_pkg::parse_manifest(&manifest_path).ok()?;
    let raw_entry = manifest.build.entry.trim();
    if raw_entry.is_empty() {
        return None;
    }

    let raw_path = Path::new(raw_entry);
    let resolved = if raw_path.is_absolute() {
        raw_path.to_path_buf()
    } else {
        manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(raw_path)
    };
    Some(resolved.to_string_lossy().to_string())
}

fn discover_manifest_path(entry_path: &str) -> Option<PathBuf> {
    let path = PathBuf::from(entry_path);
    let mut current = if path.is_dir() {
        path
    } else {
        path.parent()?.to_path_buf()
    };

    loop {
        let candidate = current.join("Magpie.toml");
        if candidate.is_file() {
            return Some(candidate);
        }
        if !current.pop() {
            break;
        }
    }
    None
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
    canonical_json_encode(&payload).unwrap_or_else(|_| "{}".to_string())
}

/// Full compilation pipeline (§22.1).
pub fn build(config: &DriverConfig) -> BuildResult {
    let mut effective_config = config.clone();
    if is_default_entry_path(&effective_config.entry_path) {
        if let Some(manifest_entry) = resolve_manifest_build_entry(&effective_config.entry_path) {
            effective_config.entry_path = manifest_entry;
        }
    }
    let config = &effective_config;

    let max_errors = config.max_errors.max(1);
    let mut result = BuildResult::default();

    {
        let start = Instant::now();
        match resolve_dependencies_for_build(&config.entry_path, config.offline) {
            Ok(Some(lock_path)) => {
                result
                    .artifacts
                    .push(lock_path.to_string_lossy().to_string());
            }
            Ok(None) => {}
            Err(err) => {
                result
                    .timing_ms
                    .insert("stage0_pkg_resolve".to_string(), elapsed_ms(start));
                result.diagnostics.push(simple_diag(
                    "MPK0010",
                    Severity::Error,
                    "dependency resolution failed",
                    err,
                ));
                mark_skipped_from(&mut result.timing_ms, 0);
                return finalize_build_result(result, config);
            }
        }
        result
            .timing_ms
            .insert("stage0_pkg_resolve".to_string(), elapsed_ms(start));
    }

    // Stage 1: read + lex + parse + in-memory CSNF canonicalization.
    let ast_files: Vec<AstFile>;
    {
        let start = Instant::now();
        let mut diag = DiagnosticBag::new(max_errors);
        ast_files = load_stage1_ast_files(&config.entry_path, &mut diag);

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
    let gpu_kernel_decls = collect_gpu_kernel_decls(&resolved_modules);

    // Stage 3: typecheck.
    // Per §22.1 stage 3, type checking is performed during AST -> HIR lowering in sema.
    let mut type_ctx = TypeCtx::new();
    let mut hir_modules: Vec<HirModule> = Vec::new();
    {
        let start = Instant::now();
        let mut diag = DiagnosticBag::new(max_errors);
        for module in &resolved_modules {
            match lower_to_hir(module, &mut type_ctx, &mut diag) {
                Ok(hir) => {
                    let _ = typecheck_module(&hir, &type_ctx, &module.symbol_table, &mut diag);
                    let impl_decls = module
                        .ast
                        .decls
                        .iter()
                        .filter_map(|decl| match &decl.node {
                            AstDecl::Impl(impl_decl) => Some(impl_decl.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    let _ = check_trait_impls(
                        &hir,
                        &type_ctx,
                        &module.symbol_table,
                        &impl_decls,
                        &module.resolved_imports,
                        &mut diag,
                    );
                    let _ = check_v01_restrictions(&hir, &type_ctx, &mut diag);
                    hir_modules.push(hir);
                }
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

    let type_id_remap = type_ctx.finalize_type_ids_with_remap();
    if !type_id_remap.is_empty() {
        remap_hir_modules_type_ids(&mut hir_modules, &type_id_remap);
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

    // Stage 10: LLVM codegen + GPU SPIR-V/registry generation.
    let mut llvm_ir_paths: Vec<PathBuf> = Vec::new();
    {
        let start = Instant::now();
        let mut diag = DiagnosticBag::new(max_errors);
        let module_count = mpir_modules.len();
        let emit_llvm_bc = emit_contains(&config.emit, "llvm-bc");
        let emit_spv = emit_contains(&config.emit, "spv");
        for (idx, module) in mpir_modules.iter().enumerate() {
            match magpie_codegen_llvm::codegen_module_with_options(
                module,
                &type_ctx,
                magpie_codegen_llvm::CodegenOptions {
                    shared_generics: config.shared_generics,
                },
            ) {
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
                        if emit_llvm_bc {
                            let bc_path = stage_module_output_path(config, idx, module_count, "bc");
                            if let Err(err) = compile_llvm_ir_to_bitcode(
                                llvm_ir_paths.last().expect("llvm path pushed"),
                                &bc_path,
                            ) {
                                emit_driver_diag(
                                    &mut diag,
                                    "MPLLVM02",
                                    Severity::Warning,
                                    "failed to emit llvm bitcode",
                                    err,
                                );
                            }
                        }
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

        let kernel_by_sid: HashMap<String, &GpuKernelDecl> = gpu_kernel_decls
            .iter()
            .map(|decl| (decl.sid.0.clone(), decl))
            .collect();
        let mut kernel_entries = Vec::new();
        let mut seen_kernel_sids: HashSet<String> = HashSet::new();

        for module in &mpir_modules {
            for func in &module.functions {
                let Some(kernel_decl) = kernel_by_sid.get(&func.sid.0) else {
                    continue;
                };
                seen_kernel_sids.insert(func.sid.0.clone());

                if !kernel_decl.target.eq_ignore_ascii_case("spv") {
                    emit_driver_diag(
                        &mut diag,
                        "MPG1200",
                        Severity::Error,
                        "unsupported gpu target",
                        format!(
                            "gpu kernel '{}' declares unsupported target '{}'; only target(spv) is supported in v0.1.",
                            func.name, kernel_decl.target
                        ),
                    );
                    continue;
                }

                let _ = magpie_gpu::validate_kernel(func, &type_ctx, &mut diag);
                let layout = compute_kernel_layout(func, &type_ctx);
                match generate_spirv_with_layout(func, &layout, &type_ctx) {
                    Ok(spirv) => {
                        if emit_spv {
                            let spv_path = stage_gpu_kernel_spv_output_path(config, &func.sid);
                            match write_binary_artifact(&spv_path, &spirv) {
                                Ok(()) => {
                                    let artifact = spv_path.to_string_lossy().to_string();
                                    if !result.artifacts.contains(&artifact) {
                                        result.artifacts.push(artifact);
                                    }
                                }
                                Err(err) => emit_driver_diag(
                                    &mut diag,
                                    "MPP0003",
                                    Severity::Error,
                                    "failed to write spir-v artifact",
                                    err,
                                ),
                            }
                        }
                        kernel_entries.push((func.sid.0.clone(), layout, spirv));
                    }
                    Err(err) => emit_driver_diag(
                        &mut diag,
                        "MPG1202",
                        Severity::Error,
                        "gpu kernel spir-v lowering failed",
                        err,
                    ),
                }
            }
        }

        for decl in &gpu_kernel_decls {
            if !seen_kernel_sids.contains(&decl.sid.0) {
                emit_driver_diag(
                    &mut diag,
                    "MPG1203",
                    Severity::Warning,
                    "gpu kernel missing in lowered mpir",
                    format!(
                        "gpu kernel sid '{}' was declared but no lowered MPIR function was found; it will not be registered.",
                        decl.sid.0
                    ),
                );
            }
        }

        let registry_ir = generate_kernel_registry_ir(&kernel_entries);
        let registry_path = stage_gpu_registry_output_path(config);
        if let Err(err) = write_text_artifact(&registry_path, &registry_ir) {
            emit_driver_diag(
                &mut diag,
                "MPP0003",
                Severity::Error,
                "failed to write gpu kernel registry artifact",
                err,
            );
        } else {
            llvm_ir_paths.push(registry_path);
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
            if let Err(err) = verify_generics_mode_markers(&llvm_ir_paths, config.shared_generics) {
                emit_driver_diag(
                    &mut diag,
                    codes::MPL2021,
                    Severity::Error,
                    "mixed generics mode",
                    err,
                );
            } else {
                let output_path = stage_link_output_path(config);
                let link_shared = emit_contains(&config.emit, "shared-lib")
                    && !emit_contains(&config.emit, "exe");
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
        for index_path in
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

/// `magpie parse` entry-point: read + lex + parse and emit debug AST artifact.
pub fn parse_entry(config: &DriverConfig) -> BuildResult {
    let max_errors = config.max_errors.max(1);
    let mut result = BuildResult::default();

    let stage_start = Instant::now();
    let mut diag = DiagnosticBag::new(max_errors);

    let mut parsed: Option<AstFile> = None;
    match fs::read_to_string(&config.entry_path) {
        Ok(source) => {
            let file_id = FileId(0);
            let tokens = lex(file_id, &source, &mut diag);
            if let Ok(ast) = parse_file(&tokens, file_id, &mut diag) {
                parsed = Some(ast);
            }
        }
        Err(err) => emit_driver_diag(
            &mut diag,
            "MPP0001",
            Severity::Error,
            "failed to read source file",
            format!("Could not read '{}': {}", config.entry_path, err),
        ),
    }

    result.timing_ms.insert(
        "parse_stage1_read_lex_parse".to_string(),
        elapsed_ms(stage_start),
    );

    let stage_failed = append_stage_diagnostics(&mut result, diag);
    if !stage_failed {
        if let Some(ast) = parsed {
            let ast_path = stage_parse_ast_output_path(config);
            let ast_text = format!("{:#?}\n", ast);
            match write_text_artifact(&ast_path, &ast_text) {
                Ok(()) => result
                    .artifacts
                    .push(ast_path.to_string_lossy().to_string()),
                Err(err) => result.diagnostics.push(simple_diag(
                    "MPP0003",
                    Severity::Error,
                    "failed to write ast artifact",
                    err,
                )),
            }
        } else {
            result.diagnostics.push(simple_diag(
                "MPP0002",
                Severity::Error,
                "parse failed",
                format!(
                    "Could not parse '{}' but no parser diagnostics were emitted.",
                    config.entry_path
                ),
            ));
        }
    }

    result.success = !has_errors(&result.diagnostics);
    result
}

fn load_stage1_ast_files(entry_path: &str, diag: &mut DiagnosticBag) -> Vec<AstFile> {
    let source = match fs::read_to_string(entry_path) {
        Ok(source) => source,
        Err(err) => {
            emit_driver_diag(
                diag,
                "MPP0001",
                Severity::Error,
                "failed to read source file",
                format!("Could not read '{}': {}", entry_path, err),
            );
            return Vec::new();
        }
    };

    let mut next_file_id = 0_u32;
    let entry_file_id = FileId(next_file_id);
    next_file_id = next_file_id.saturating_add(1);
    let Some(entry_ast) = parse_stage1_ast(&source, entry_file_id, diag) else {
        return Vec::new();
    };

    let mut ast_files = vec![entry_ast];
    let mut discovered_module_paths: BTreeSet<String> = BTreeSet::new();
    let mut parsed_module_paths: BTreeSet<String> = BTreeSet::new();
    let mut pending_module_paths: BTreeSet<String> = BTreeSet::new();
    let project_root = stage1_project_root(entry_path);

    let entry_module_path = ast_files[0].header.node.module_path.node.to_string();
    discovered_module_paths.insert(entry_module_path.clone());
    parsed_module_paths.insert(entry_module_path);
    queue_stage1_imports(
        &ast_files[0],
        &mut discovered_module_paths,
        &mut pending_module_paths,
    );

    while let Some(module_path) = pop_first_module_path(&mut pending_module_paths) {
        let Some(module_file_path) = resolve_stage1_module_file_path(&module_path, &project_root)
        else {
            emit_driver_diag(
                diag,
                "MPS0003",
                Severity::Warning,
                "imported module not found",
                format!(
                    "Could not resolve imported module '{}'; searched from '{}'.",
                    module_path,
                    project_root.display()
                ),
            );
            continue;
        };
        let source = match fs::read_to_string(&module_file_path) {
            Ok(source) => source,
            Err(err) => {
                emit_driver_diag(
                    diag,
                    "MPS0004",
                    Severity::Warning,
                    "could not read imported module",
                    format!("Could not read '{}': {}", module_file_path.display(), err),
                );
                continue;
            }
        };

        let file_id = FileId(next_file_id);
        next_file_id = next_file_id.saturating_add(1);
        let Some(ast) = parse_stage1_ast(&source, file_id, diag) else {
            continue;
        };

        let declared_module_path = ast.header.node.module_path.node.to_string();
        discovered_module_paths.insert(declared_module_path.clone());
        if !parsed_module_paths.insert(declared_module_path) {
            continue;
        }

        queue_stage1_imports(
            &ast,
            &mut discovered_module_paths,
            &mut pending_module_paths,
        );
        ast_files.push(ast);
    }

    ast_files
}

fn parse_stage1_ast(source: &str, file_id: FileId, diag: &mut DiagnosticBag) -> Option<AstFile> {
    let tokens = lex(file_id, source, diag);
    let ast = parse_file(&tokens, file_id, diag).ok()?;
    let canonical = format_csnf(&ast);
    let _canonical_with_digest = update_digest(&canonical);
    Some(ast)
}

fn queue_stage1_imports(
    ast: &AstFile,
    discovered_module_paths: &mut BTreeSet<String>,
    pending_module_paths: &mut BTreeSet<String>,
) {
    let mut imports = ast
        .header
        .node
        .imports
        .iter()
        .map(|group| group.node.module_path.to_string())
        .collect::<Vec<_>>();
    imports.sort();
    imports.dedup();
    for module_path in imports {
        if discovered_module_paths.insert(module_path.clone()) {
            pending_module_paths.insert(module_path);
        }
    }
}

fn pop_first_module_path(module_paths: &mut BTreeSet<String>) -> Option<String> {
    let module_path = module_paths.iter().next()?.clone();
    module_paths.remove(&module_path);
    Some(module_path)
}

fn module_path_to_stage1_file_path(module_path: &str) -> Option<PathBuf> {
    let segments = module_path
        .split('.')
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>();
    if segments.is_empty() {
        return None;
    }

    if segments[0] == "std" {
        if segments.len() != 2 {
            return None;
        }
        let std_name = segments[1];
        return Some(
            PathBuf::from("std")
                .join(format!("std.{std_name}"))
                .join(format!("{std_name}.mp")),
        );
    }

    let rel_segments = if segments.len() == 1 {
        &segments[..]
    } else {
        &segments[1..]
    };
    if rel_segments.is_empty() {
        return None;
    }

    let mut path = PathBuf::from("src");
    for segment in rel_segments {
        path.push(segment);
    }
    path.set_extension("mp");
    Some(path)
}

fn stage1_project_root(entry_path: &str) -> PathBuf {
    let entry = PathBuf::from(entry_path);
    let entry_abs = if entry.is_absolute() {
        entry
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(entry)
    };
    let parent = entry_abs
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    if parent.file_name().and_then(|name| name.to_str()) == Some("src") {
        parent.parent().unwrap_or(&parent).to_path_buf()
    } else {
        parent
    }
}

fn resolve_stage1_module_file_path(module_path: &str, project_root: &Path) -> Option<PathBuf> {
    let rel = module_path_to_stage1_file_path(module_path)?;
    if rel.is_absolute() {
        return rel.is_file().then_some(rel);
    }

    let primary = project_root.join(&rel);
    if primary.is_file() {
        return Some(primary);
    }

    if rel.starts_with("std") {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        for ancestor in cwd.ancestors() {
            let candidate = ancestor.join(&rel);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    Path::new(&rel).is_file().then_some(rel)
}

fn collect_gpu_kernel_decls(resolved_modules: &[ResolvedModule]) -> Vec<GpuKernelDecl> {
    let mut out = Vec::new();
    for module in resolved_modules {
        for decl in &module.ast.decls {
            let AstDecl::GpuFn(gpu_fn) = &decl.node else {
                continue;
            };
            if let Some(sym) = module.symbol_table.functions.get(&gpu_fn.inner.name) {
                out.push(GpuKernelDecl {
                    sid: sym.sid.clone(),
                    target: gpu_fn.target.clone(),
                });
            }
        }
    }
    out.sort_by(|a, b| a.sid.0.cmp(&b.sid.0));
    out.dedup_by(|a, b| a.sid.0 == b.sid.0);
    out
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
        result
            .diagnostics
            .extend(run_lints(&hir_modules, &type_ctx));
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
    let code_str = code.to_string();
    Diagnostic {
        code: code_str.clone(),
        severity: Severity::Warning,
        title: title.into(),
        primary_span: None,
        secondary_spans: Vec::new(),
        message: message.into(),
        explanation_md: magpie_diag::explain_code(&code_str),
        why: None,
        suggested_fixes,
        rag_bundle: Vec::new(),
        related_docs: Vec::new(),
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

fn normalize_fix_path(path: &str) -> String {
    let raw = Path::new(path);
    if raw.is_absolute() {
        if let Ok(cwd) = std::env::current_dir() {
            if let Ok(rel) = raw.strip_prefix(&cwd) {
                return rel.to_string_lossy().replace('\\', "/");
            }
        }
    }
    path.trim_start_matches("./").replace('\\', "/")
}

fn fix_digest_maps(
    path: &str,
    patch: &str,
) -> (BTreeMap<String, String>, BTreeMap<String, String>) {
    let mut applies_to = BTreeMap::new();
    let mut produces = BTreeMap::new();
    let norm = normalize_fix_path(path);
    let pre_bytes = fs::read(path).unwrap_or_default();
    let pre_digest = blake3::hash(&pre_bytes).to_hex().to_string();
    let post_digest = blake3::hash(format!("{pre_digest}\n{patch}").as_bytes())
        .to_hex()
        .to_string();
    applies_to.insert(norm.clone(), pre_digest);
    produces.insert(norm, post_digest);
    (applies_to, produces)
}

fn suggested_fix_with_digests(
    title: String,
    patch_format: &str,
    patch: String,
    confidence: f64,
    requires_fmt: bool,
    source_path: &str,
) -> SuggestedFix {
    let (applies_to, produces) = fix_digest_maps(source_path, &patch);
    SuggestedFix {
        title,
        patch_format: patch_format.to_string(),
        patch,
        confidence,
        requires_fmt,
        applies_to,
        produces,
    }
}

fn suggested_fix_unused_local(source_path: &str, local_id: u32) -> SuggestedFix {
    let old_local = local_name(local_id);
    let new_local = format!("%_v{}", local_id);
    let patch = format!(
        "diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@\n-  {old_local}: <type> = <expr>\n+  {new_local}: <type> = <expr>\n",
        path = source_path,
    );
    suggested_fix_with_digests(
        format!(
            "Prefix '{}' with '_' to mark intentionally unused",
            old_local
        ),
        "unified-diff",
        patch,
        0.72,
        false,
        source_path,
    )
}

fn suggested_fix_empty_block(source_path: &str, block_id: u32) -> SuggestedFix {
    let patch = format!(
        "diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@\n-bb{block_id}:\n-  ; empty block\n+; removed empty block bb{block_id}\n",
        path = source_path,
    );
    suggested_fix_with_digests(
        format!("Remove empty block bb{}", block_id),
        "unified-diff",
        patch,
        0.62,
        false,
        source_path,
    )
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
    suggested_fix_with_digests(
        format!("Remove unnecessary borrow '{}'", local),
        "unified-diff",
        patch,
        0.68,
        false,
        source_path,
    )
}

fn extract_quoted_segment(message: &str) -> Option<String> {
    let start = message.find('\'')?;
    let rest = &message[start + 1..];
    let end = rest.find('\'')?;
    Some(rest[..end].to_string())
}

fn suggested_fix_add_missing_import(source_path: &str, symbol_hint: &str) -> SuggestedFix {
    let symbol_hint = if symbol_hint.trim().is_empty() {
        "MissingSymbol"
    } else {
        symbol_hint
    };
    let patch = format!(
        "diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@\n imports {{ ... }}\n+imports {{ std::{symbol_hint} }}\n",
        path = source_path
    );
    suggested_fix_with_digests(
        format!("Add import for '{}'", symbol_hint),
        "unified-diff",
        patch,
        0.58,
        true,
        source_path,
    )
}

fn suggested_fix_map_get_to_contains_get_ref(source_path: &str) -> SuggestedFix {
    let patch = format!(
        "diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@\n-  %v: <V> = map.get {{ map=%m, key=%k }}\n+  %has: bool = map.contains_key {{ map=%m, key=%k }}\n+  cbr %has, bb_has, bb_missing\n+bb_has:\n+  %v_ref: ptr <V> = map.get_ref {{ map=%m, key=%k }}\n+  %v: <V> = load %v_ref\n+bb_missing:\n+  panic \"missing key\"\n",
        path = source_path
    );
    suggested_fix_with_digests(
        "Replace map.get with contains_key + get_ref pattern".to_string(),
        "unified-diff",
        patch,
        0.7,
        true,
        source_path,
    )
}

fn suggested_fix_insert_share_clone(source_path: &str) -> SuggestedFix {
    let patch = format!(
        "diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@\n-  %owned: <T> = <expr>\n+  %owned: <T> = <expr>\n+  %shared: shared <T> = share %owned\n+  %shared2: shared <T> = clone.shared %shared\n",
        path = source_path
    );
    suggested_fix_with_digests(
        "Insert share/clone.shared before shared use".to_string(),
        "unified-diff",
        patch,
        0.64,
        true,
        source_path,
    )
}

fn suggested_fix_split_borrow_scope(source_path: &str) -> SuggestedFix {
    let patch = format!(
        "diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@\n-  %b: borrow <T> = borrow.shared %x\n-  br bb_next\n+  br bb_next\n+\n+bb_next:\n+  %b: borrow <T> = borrow.shared %x\n",
        path = source_path
    );
    suggested_fix_with_digests(
        "Split borrow so it is recreated per basic block".to_string(),
        "unified-diff",
        patch,
        0.66,
        true,
        source_path,
    )
}

fn suggested_fix_add_trait_impl_stub(
    source_path: &str,
    trait_name: &str,
    ty_name: &str,
) -> SuggestedFix {
    let trait_name = if trait_name.trim().is_empty() {
        "eq"
    } else {
        trait_name
    };
    let ty_name = if ty_name.trim().is_empty() {
        "MyType"
    } else {
        ty_name
    };
    let fn_name = format!("@{}_{}", trait_name, ty_name.replace(['.', ':'], "_"));
    let patch = format!(
        "diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@\n+fn {fn_name}(%a: borrow {ty_name}, %b: borrow {ty_name}) -> i32 {{\n+bb0:\n+  ret 0\n+}}\n+\n+impl {trait_name} for {ty_name} = {fn_name}\n",
        path = source_path
    );
    suggested_fix_with_digests(
        format!("Add missing `impl {trait_name} for {ty_name}` stub"),
        "unified-diff",
        patch,
        0.61,
        true,
        source_path,
    )
}

fn apply_core_fixers(diag: &mut Diagnostic) {
    if !diag.suggested_fixes.is_empty() {
        return;
    }

    let default_path = "src/main.mp";
    let fixes = match diag.code.as_str() {
        "MPS0002" | "MPS0003" | "MPS0006" => {
            let symbol_hint = if let Some(seg) = extract_quoted_segment(&diag.message) {
                seg.split("::")
                    .last()
                    .unwrap_or("MissingSymbol")
                    .to_string()
            } else {
                "MissingSymbol".to_string()
            };
            vec![suggested_fix_add_missing_import(default_path, &symbol_hint)]
        }
        "MPO0103" => vec![suggested_fix_map_get_to_contains_get_ref(default_path)],
        "MPO0004" | "MPO0011" => vec![suggested_fix_insert_share_clone(default_path)],
        "MPO0101" | "MPO0102" => vec![suggested_fix_split_borrow_scope(default_path)],
        "MPT1023" => {
            let re = Regex::new(r"`impl\s+([A-Za-z0-9_]+)\s+for\s+([^`]+)`").ok();
            let (trait_name, ty_name) = if let Some(re) = re {
                if let Some(caps) = re.captures(&diag.message) {
                    (
                        caps.get(1).map(|m| m.as_str()).unwrap_or("eq"),
                        caps.get(2).map(|m| m.as_str()).unwrap_or("MyType"),
                    )
                } else {
                    ("eq", "MyType")
                }
            } else {
                ("eq", "MyType")
            };
            vec![suggested_fix_add_trait_impl_stub(
                default_path,
                trait_name,
                ty_name,
            )]
        }
        "MPT2032" => {
            let trait_name = Regex::new(r"impl '([^']+)'")
                .ok()
                .and_then(|re| re.captures(&diag.message))
                .and_then(|c| c.get(1).map(|m| m.as_str().to_string()))
                .unwrap_or_else(|| "eq".to_string());
            let ty_name = Regex::new(r"for '([^']+)'")
                .ok()
                .and_then(|re| re.captures(&diag.message))
                .and_then(|c| c.get(1).map(|m| m.as_str().to_string()))
                .unwrap_or_else(|| "MyType".to_string());
            vec![suggested_fix_add_trait_impl_stub(
                default_path,
                &trait_name,
                &ty_name,
            )]
        }
        _ => Vec::new(),
    };

    if !fixes.is_empty() {
        diag.suggested_fixes = fixes;
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
    run_tests_with(
        config,
        filter,
        build,
        discover_test_functions_from_hir,
        run_single_test_binary,
    )
}

fn run_tests_with<BuildFn, DiscoverFn, RunTestFn>(
    config: &DriverConfig,
    filter: Option<&str>,
    mut build_fn: BuildFn,
    mut discover_fn: DiscoverFn,
    mut run_test_fn: RunTestFn,
) -> TestResult
where
    BuildFn: FnMut(&DriverConfig) -> BuildResult,
    DiscoverFn: FnMut(&DriverConfig) -> Vec<String>,
    RunTestFn: FnMut(&str, &str) -> bool,
{
    let initial_build = build_fn(config);
    let mut discovered = discover_fn(config);

    if let Some(pattern) = filter {
        discovered.retain(|name| name.contains(pattern));
    }

    let mut executable = if initial_build.success {
        find_executable_artifact(&config.target_triple, &initial_build.artifacts)
    } else {
        None
    };

    if executable.is_none() && initial_build.success {
        let fallback_config = fallback_test_driver_config(config);
        let fallback_build = build_fn(&fallback_config);
        if fallback_build.success {
            executable =
                find_executable_artifact(&fallback_config.target_triple, &fallback_build.artifacts);
        }
    }

    let mut test_names = Vec::with_capacity(discovered.len());
    match executable {
        Some(path) => {
            for test_name in discovered {
                let passed = run_test_fn(&path, &test_name);
                test_names.push((test_name, passed));
            }
        }
        None if discovered.is_empty() => {}
        None => {
            for test_name in discovered {
                test_names.push((test_name, false));
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

fn fallback_test_driver_config(config: &DriverConfig) -> DriverConfig {
    let mut fallback = config.clone();
    if !emit_contains(&fallback.emit, "exe") {
        fallback.emit.push("exe".to_string());
    }
    fallback
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
                                &mut edges, &fn_owner, module, func, callee_sid, "call",
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
    modules_json.sort_by(|lhs, rhs| {
        lhs["module_path"]
            .as_str()
            .cmp(&rhs["module_path"].as_str())
    });

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
            edges.sort_by(|lhs, rhs| {
                lhs.0
                    .cmp(&rhs.0)
                    .then(lhs.1.cmp(&rhs.1))
                    .then(lhs.2.cmp(&rhs.2))
            });

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
        HirTerminator::Cbr {
            then_bb, else_bb, ..
        } => vec![
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
) -> Vec<PathBuf> {
    let items = build_mms_items(artifacts, hir_modules);
    let source_fingerprints = build_mms_source_fingerprints(config, hir_modules);
    let index = build_index_with_sources(&items, &source_fingerprints);
    let encoded = match canonical_json_encode(&index) {
        Ok(encoded) => encoded,
        Err(err) => {
            emit_driver_diag(
                diag,
                "MPP0003",
                Severity::Warning,
                "failed to encode mms index",
                format!("Could not serialize MMS index: {}", err),
            );
            return Vec::new();
        }
    };

    let mut written_paths = Vec::new();

    let index_path = stage_mms_index_output_path(config);
    if let Err(err) = write_text_artifact(&index_path, &encoded) {
        emit_driver_diag(
            diag,
            "MPP0003",
            Severity::Warning,
            "failed to write mms index",
            err,
        );
        return Vec::new();
    }
    written_paths.push(index_path);

    let mirror_index_path = stage_mms_memory_index_output_path(config);
    match write_text_artifact(&mirror_index_path, &encoded) {
        Ok(()) => written_paths.push(mirror_index_path),
        Err(err) => emit_driver_diag(
            diag,
            "MPP0003",
            Severity::Warning,
            "failed to write mirrored mms index",
            err,
        ),
    }

    written_paths
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
                tags: vec!["artifact".to_string(), ext, format!("order:{}", idx)],
                priority: 50,
                token_cost,
            }
        })
        .collect()
}

fn build_mms_source_fingerprints(
    config: &DriverConfig,
    hir_modules: &[HirModule],
) -> Vec<MmsSourceFingerprint> {
    let mut digest_by_path: BTreeMap<String, String> = BTreeMap::new();

    let project_root = project_root_from_entry_path(&config.entry_path);

    for module in hir_modules {
        let source_rel = module_source_path(&module.path);
        let source_path = project_root.join(&source_rel);
        let source_bytes = match fs::read(&source_path) {
            Ok(bytes) => bytes,
            Err(_) => continue,
        };

        digest_by_path.insert(
            source_path.to_string_lossy().to_string(),
            format!("{:016x}", stable_hash_u64(&source_bytes)),
        );
    }

    digest_by_path
        .into_iter()
        .map(|(path, digest)| MmsSourceFingerprint { path, digest })
        .collect()
}

fn project_root_from_entry_path(entry_path: &str) -> PathBuf {
    let entry = PathBuf::from(entry_path);
    if entry.is_absolute() {
        if let Some(src_dir) = entry.parent() {
            if src_dir
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name == "src")
            {
                if let Some(parent) = src_dir.parent() {
                    return parent.to_path_buf();
                }
            }
            return src_dir.to_path_buf();
        }
    }

    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
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
        let tail_term = std::mem::replace(
            &mut func.blocks[blk_idx].terminator,
            HirTerminator::Br(resume_id),
        );
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

    // Keep is_async = true so that post-lowering verifiers can skip
    // SSA domination checks for async coroutine state machines.
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

fn remap_type_id(id: TypeId, remap: &HashMap<TypeId, TypeId>) -> TypeId {
    remap.get(&id).copied().unwrap_or(id)
}

fn remap_hir_modules_type_ids(modules: &mut [HirModule], remap: &HashMap<TypeId, TypeId>) {
    for module in modules {
        remap_hir_module_type_ids(module, remap);
    }
}

fn remap_hir_module_type_ids(module: &mut HirModule, remap: &HashMap<TypeId, TypeId>) {
    for func in &mut module.functions {
        remap_hir_function_type_ids(func, remap);
    }

    for global in &mut module.globals {
        global.ty = remap_type_id(global.ty, remap);
        remap_hir_const_type_ids(&mut global.init, remap);
    }

    for decl in &mut module.type_decls {
        remap_hir_type_decl_type_ids(decl, remap);
    }
}

fn remap_hir_type_decl_type_ids(decl: &mut HirTypeDecl, remap: &HashMap<TypeId, TypeId>) {
    match decl {
        HirTypeDecl::Struct { fields, .. } => {
            for (_, ty) in fields {
                *ty = remap_type_id(*ty, remap);
            }
        }
        HirTypeDecl::Enum { variants, .. } => {
            for variant in variants {
                for (_, ty) in &mut variant.fields {
                    *ty = remap_type_id(*ty, remap);
                }
            }
        }
    }
}

fn remap_hir_function_type_ids(func: &mut HirFunction, remap: &HashMap<TypeId, TypeId>) {
    for (_, ty) in &mut func.params {
        *ty = remap_type_id(*ty, remap);
    }
    func.ret_ty = remap_type_id(func.ret_ty, remap);

    for block in &mut func.blocks {
        remap_hir_block_type_ids(block, remap);
    }
}

fn remap_hir_block_type_ids(block: &mut HirBlock, remap: &HashMap<TypeId, TypeId>) {
    for instr in &mut block.instrs {
        remap_hir_instr_type_ids(instr, remap);
    }
    for op in &mut block.void_ops {
        remap_hir_void_op_type_ids(op, remap);
    }
    remap_hir_terminator_type_ids(&mut block.terminator, remap);
}

fn remap_hir_instr_type_ids(instr: &mut HirInstr, remap: &HashMap<TypeId, TypeId>) {
    instr.ty = remap_type_id(instr.ty, remap);
    remap_hir_op_type_ids(&mut instr.op, remap);
}

fn remap_hir_op_type_ids(op: &mut HirOp, remap: &HashMap<TypeId, TypeId>) {
    match op {
        HirOp::Const(v) => remap_hir_const_type_ids(v, remap),
        HirOp::Move { v }
        | HirOp::BorrowShared { v }
        | HirOp::BorrowMut { v }
        | HirOp::SuspendAwait { fut: v }
        | HirOp::Share { v }
        | HirOp::CloneShared { v }
        | HirOp::CloneWeak { v }
        | HirOp::WeakDowngrade { v }
        | HirOp::WeakUpgrade { v }
        | HirOp::EnumTag { v }
        | HirOp::EnumPayload { v, .. }
        | HirOp::EnumIs { v, .. }
        | HirOp::ArrLen { arr: v }
        | HirOp::ArrPop { arr: v }
        | HirOp::ArrSort { arr: v }
        | HirOp::MapLen { map: v }
        | HirOp::MapKeys { map: v }
        | HirOp::MapValues { map: v }
        | HirOp::StrLen { s: v }
        | HirOp::StrBytes { s: v }
        | HirOp::StrBuilderBuild { b: v }
        | HirOp::StrParseI64 { s: v }
        | HirOp::StrParseU64 { s: v }
        | HirOp::StrParseF64 { s: v }
        | HirOp::StrParseBool { s: v }
        | HirOp::GpuBufferLen { buf: v }
        | HirOp::Panic { msg: v } => remap_hir_value_type_ids(v, remap),
        HirOp::New { ty, fields } => {
            *ty = remap_type_id(*ty, remap);
            for (_, value) in fields {
                remap_hir_value_type_ids(value, remap);
            }
        }
        HirOp::GetField { obj, .. } => remap_hir_value_type_ids(obj, remap),
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
        | HirOp::FDivFast { lhs, rhs }
        | HirOp::StrConcat { a: lhs, b: rhs }
        | HirOp::StrEq { a: lhs, b: rhs } => {
            remap_hir_value_type_ids(lhs, remap);
            remap_hir_value_type_ids(rhs, remap);
        }
        HirOp::Cast { to, v } => {
            *to = remap_type_id(*to, remap);
            remap_hir_value_type_ids(v, remap);
        }
        HirOp::PtrNull { to } => *to = remap_type_id(*to, remap),
        HirOp::PtrAddr { p } => remap_hir_value_type_ids(p, remap),
        HirOp::PtrFromAddr { to, addr } => {
            *to = remap_type_id(*to, remap);
            remap_hir_value_type_ids(addr, remap);
        }
        HirOp::PtrAdd { p, count } => {
            remap_hir_value_type_ids(p, remap);
            remap_hir_value_type_ids(count, remap);
        }
        HirOp::PtrLoad { to, p } => {
            *to = remap_type_id(*to, remap);
            remap_hir_value_type_ids(p, remap);
        }
        HirOp::PtrStore { to, p, v } => {
            *to = remap_type_id(*to, remap);
            remap_hir_value_type_ids(p, remap);
            remap_hir_value_type_ids(v, remap);
        }
        HirOp::Call { inst, args, .. } | HirOp::SuspendCall { inst, args, .. } => {
            for ty in inst {
                *ty = remap_type_id(*ty, remap);
            }
            for value in args {
                remap_hir_value_type_ids(value, remap);
            }
        }
        HirOp::CallIndirect { callee, args } | HirOp::CallVoidIndirect { callee, args } => {
            remap_hir_value_type_ids(callee, remap);
            for value in args {
                remap_hir_value_type_ids(value, remap);
            }
        }
        HirOp::Phi { ty, incomings } => {
            *ty = remap_type_id(*ty, remap);
            for (_, value) in incomings {
                remap_hir_value_type_ids(value, remap);
            }
        }
        HirOp::EnumNew { args, .. } => {
            for (_, value) in args {
                remap_hir_value_type_ids(value, remap);
            }
        }
        HirOp::CallableCapture { captures, .. } => {
            for (_, value) in captures {
                remap_hir_value_type_ids(value, remap);
            }
        }
        HirOp::ArrNew { elem_ty, cap } => {
            *elem_ty = remap_type_id(*elem_ty, remap);
            remap_hir_value_type_ids(cap, remap);
        }
        HirOp::ArrGet { arr, idx }
        | HirOp::MapGet { map: arr, key: idx }
        | HirOp::MapGetRef { map: arr, key: idx }
        | HirOp::MapDelete { map: arr, key: idx }
        | HirOp::MapContainsKey { map: arr, key: idx }
        | HirOp::MapDeleteVoid { map: arr, key: idx }
        | HirOp::GpuBufferLoad { buf: arr, idx } => {
            remap_hir_value_type_ids(arr, remap);
            remap_hir_value_type_ids(idx, remap);
        }
        HirOp::ArrSet { arr, idx, val }
        | HirOp::MapSet {
            map: arr,
            key: idx,
            val,
        } => {
            remap_hir_value_type_ids(arr, remap);
            remap_hir_value_type_ids(idx, remap);
            remap_hir_value_type_ids(val, remap);
        }
        HirOp::ArrPush { arr, val } | HirOp::ArrContains { arr, val } => {
            remap_hir_value_type_ids(arr, remap);
            remap_hir_value_type_ids(val, remap);
        }
        HirOp::ArrSlice { arr, start, end } | HirOp::StrSlice { s: arr, start, end } => {
            remap_hir_value_type_ids(arr, remap);
            remap_hir_value_type_ids(start, remap);
            remap_hir_value_type_ids(end, remap);
        }
        HirOp::ArrMap { arr, func }
        | HirOp::ArrFilter { arr, func }
        | HirOp::ArrForeach { arr, func } => {
            remap_hir_value_type_ids(arr, remap);
            remap_hir_value_type_ids(func, remap);
        }
        HirOp::ArrReduce { arr, init, func } => {
            remap_hir_value_type_ids(arr, remap);
            remap_hir_value_type_ids(init, remap);
            remap_hir_value_type_ids(func, remap);
        }
        HirOp::MapNew { key_ty, val_ty } => {
            *key_ty = remap_type_id(*key_ty, remap);
            *val_ty = remap_type_id(*val_ty, remap);
        }
        HirOp::StrBuilderNew
        | HirOp::GpuThreadId
        | HirOp::GpuWorkgroupId
        | HirOp::GpuWorkgroupSize
        | HirOp::GpuGlobalId => {}
        HirOp::StrBuilderAppendStr { b, s } => {
            remap_hir_value_type_ids(b, remap);
            remap_hir_value_type_ids(s, remap);
        }
        HirOp::StrBuilderAppendI64 { b, v }
        | HirOp::StrBuilderAppendI32 { b, v }
        | HirOp::StrBuilderAppendF64 { b, v }
        | HirOp::StrBuilderAppendBool { b, v } => {
            remap_hir_value_type_ids(b, remap);
            remap_hir_value_type_ids(v, remap);
        }
        HirOp::JsonEncode { ty, v } => {
            *ty = remap_type_id(*ty, remap);
            remap_hir_value_type_ids(v, remap);
        }
        HirOp::JsonDecode { ty, s } => {
            *ty = remap_type_id(*ty, remap);
            remap_hir_value_type_ids(s, remap);
        }
        HirOp::GpuShared { ty, size } => {
            *ty = remap_type_id(*ty, remap);
            remap_hir_value_type_ids(size, remap);
        }
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
            remap_hir_value_type_ids(device, remap);
            remap_hir_value_type_ids(groups, remap);
            remap_hir_value_type_ids(threads, remap);
            for value in args {
                remap_hir_value_type_ids(value, remap);
            }
        }
    }
}

fn remap_hir_void_op_type_ids(op: &mut HirOpVoid, remap: &HashMap<TypeId, TypeId>) {
    match op {
        HirOpVoid::CallVoid { inst, args, .. } => {
            for ty in inst {
                *ty = remap_type_id(*ty, remap);
            }
            for value in args {
                remap_hir_value_type_ids(value, remap);
            }
        }
        HirOpVoid::CallVoidIndirect { callee, args } => {
            remap_hir_value_type_ids(callee, remap);
            for value in args {
                remap_hir_value_type_ids(value, remap);
            }
        }
        HirOpVoid::SetField { obj, value, .. } => {
            remap_hir_value_type_ids(obj, remap);
            remap_hir_value_type_ids(value, remap);
        }
        HirOpVoid::ArrSet { arr, idx, val }
        | HirOpVoid::MapSet {
            map: arr,
            key: idx,
            val,
        } => {
            remap_hir_value_type_ids(arr, remap);
            remap_hir_value_type_ids(idx, remap);
            remap_hir_value_type_ids(val, remap);
        }
        HirOpVoid::ArrPush { arr, val } => {
            remap_hir_value_type_ids(arr, remap);
            remap_hir_value_type_ids(val, remap);
        }
        HirOpVoid::ArrSort { arr } => remap_hir_value_type_ids(arr, remap),
        HirOpVoid::ArrForeach { arr, func } => {
            remap_hir_value_type_ids(arr, remap);
            remap_hir_value_type_ids(func, remap);
        }
        HirOpVoid::MapDeleteVoid { map, key } => {
            remap_hir_value_type_ids(map, remap);
            remap_hir_value_type_ids(key, remap);
        }
        HirOpVoid::StrBuilderAppendStr { b, s } => {
            remap_hir_value_type_ids(b, remap);
            remap_hir_value_type_ids(s, remap);
        }
        HirOpVoid::StrBuilderAppendI64 { b, v }
        | HirOpVoid::StrBuilderAppendI32 { b, v }
        | HirOpVoid::StrBuilderAppendF64 { b, v }
        | HirOpVoid::StrBuilderAppendBool { b, v } => {
            remap_hir_value_type_ids(b, remap);
            remap_hir_value_type_ids(v, remap);
        }
        HirOpVoid::PtrStore { to, p, v } => {
            *to = remap_type_id(*to, remap);
            remap_hir_value_type_ids(p, remap);
            remap_hir_value_type_ids(v, remap);
        }
        HirOpVoid::Panic { msg } => remap_hir_value_type_ids(msg, remap),
        HirOpVoid::GpuBarrier => {}
        HirOpVoid::GpuBufferStore { buf, idx, val } => {
            remap_hir_value_type_ids(buf, remap);
            remap_hir_value_type_ids(idx, remap);
            remap_hir_value_type_ids(val, remap);
        }
    }
}

fn remap_hir_terminator_type_ids(term: &mut HirTerminator, remap: &HashMap<TypeId, TypeId>) {
    match term {
        HirTerminator::Ret(Some(value)) => remap_hir_value_type_ids(value, remap),
        HirTerminator::Ret(None) | HirTerminator::Br(_) | HirTerminator::Unreachable => {}
        HirTerminator::Cbr { cond, .. } => remap_hir_value_type_ids(cond, remap),
        HirTerminator::Switch { val, arms, .. } => {
            remap_hir_value_type_ids(val, remap);
            for (const_val, _) in arms {
                remap_hir_const_type_ids(const_val, remap);
            }
        }
    }
}

fn remap_hir_value_type_ids(value: &mut HirValue, remap: &HashMap<TypeId, TypeId>) {
    if let HirValue::Const(c) = value {
        remap_hir_const_type_ids(c, remap);
    }
}

fn remap_hir_const_type_ids(value: &mut HirConst, remap: &HashMap<TypeId, TypeId>) {
    value.ty = remap_type_id(value.ty, remap);
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

fn build_output_root(config: &DriverConfig) -> PathBuf {
    config
        .cache_dir
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target"))
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
    build_output_root(config)
        .join(&config.target_triple)
        .join(config.profile.as_str())
        .join(format!("{module_stem}.{extension}"))
}

fn stage_gpu_registry_output_path(config: &DriverConfig) -> PathBuf {
    let stem = Path::new(&config.entry_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("main");
    build_output_root(config)
        .join(&config.target_triple)
        .join(config.profile.as_str())
        .join(format!("{stem}.gpu_registry.ll"))
}

fn stage_gpu_kernel_spv_output_path(config: &DriverConfig, sid: &Sid) -> PathBuf {
    build_output_root(config)
        .join(&config.target_triple)
        .join(config.profile.as_str())
        .join("gpu")
        .join(format!("{}.spv", sid_artifact_component(&sid.0)))
}

fn sid_artifact_component(raw: &str) -> String {
    let out = raw
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    if out.is_empty() {
        "kernel".to_string()
    } else {
        out
    }
}

fn stage_parse_ast_output_path(config: &DriverConfig) -> PathBuf {
    let stem = Path::new(&config.entry_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("main");
    build_output_root(config)
        .join(&config.target_triple)
        .join(config.profile.as_str())
        .join(format!("{stem}.ast.txt"))
}

fn stage_graph_output_path(config: &DriverConfig, suffix: &str) -> PathBuf {
    let stem = Path::new(&config.entry_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("main");
    build_output_root(config)
        .join(&config.target_triple)
        .join(config.profile.as_str())
        .join(format!("{stem}.{suffix}.json"))
}

fn stage_mpdbg_output_path(config: &DriverConfig) -> PathBuf {
    let stem = Path::new(&config.entry_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("main");
    build_output_root(config)
        .join(&config.target_triple)
        .join(config.profile.as_str())
        .join(format!("{stem}.mpdbg"))
}

fn stage_mms_index_output_path(config: &DriverConfig) -> PathBuf {
    let stem = Path::new(&config.entry_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("main");
    build_output_root(config)
        .join(&config.target_triple)
        .join(config.profile.as_str())
        .join(format!("{stem}.mms_index.json"))
}

fn stage_mms_memory_index_output_path(config: &DriverConfig) -> PathBuf {
    let stem = Path::new(&config.entry_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("main");
    PathBuf::from(".magpie")
        .join("memory")
        .join(format!("{stem}.mms_index.json"))
}

fn write_text_artifact(path: &Path, text: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("Could not create '{}': {}", parent.display(), err))?;
    }
    fs::write(path, text).map_err(|err| format!("Could not write '{}': {}", path.display(), err))
}

fn write_binary_artifact(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("Could not create '{}': {}", parent.display(), err))?;
    }
    fs::write(path, bytes).map_err(|err| format!("Could not write '{}': {}", path.display(), err))
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

    let declaration = Regex::new(
        r"(?m)^\s*([A-Za-z_][A-Za-z0-9_\s\*\t]*?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^;{}]*)\)\s*;",
    )
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
        _ => "rawptr<u8>".to_string(),
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
        let has_rawptr_param = item.params.iter().any(|(_, ty)| ty.starts_with("rawptr<"));
        let returns_rawptr = item.ret_ty.starts_with("rawptr<");
        let mut attrs = vec![format!("link_name=\"{}\"", item.name)];
        if returns_rawptr {
            attrs.push("returns=\"borrowed\"".to_string());
        }
        if has_rawptr_param {
            attrs.push("params=\"borrowed\"".to_string());
        }
        let attrs_text = attrs.join(" ");
        out.push_str(&format!(
            "  fn @{}({}) -> {} attrs {{ {} }}\n",
            item.name, params, item.ret_ty, attrs_text
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
    let base = build_output_root(config)
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

fn compile_llvm_ir_to_bitcode(llvm_ir_path: &Path, bitcode_path: &Path) -> Result<(), String> {
    ensure_parent_dir(bitcode_path)?;

    if command_available("llvm-as") {
        return run_command(
            "llvm-as",
            &[
                llvm_ir_path.to_string_lossy().to_string(),
                "-o".to_string(),
                bitcode_path.to_string_lossy().to_string(),
            ],
        );
    }

    if command_available("clang") {
        return run_command(
            "clang",
            &[
                "-x".to_string(),
                "ir".to_string(),
                "-emit-llvm".to_string(),
                "-c".to_string(),
                "-o".to_string(),
                bitcode_path.to_string_lossy().to_string(),
                llvm_ir_path.to_string_lossy().to_string(),
            ],
        );
    }

    Err("neither llvm-as nor clang is available in PATH".to_string())
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
    let mut linkers: Vec<(&str, bool)> = Vec::new();
    if command_available("clang") {
        // Prefer lld via clang first for deterministic, fast linking.
        linkers.push(("clang", true));
        linkers.push(("clang", false));
    }
    if command_available("cc") {
        // Fallback to cc; try lld first when supported.
        linkers.push(("cc", true));
        linkers.push(("cc", false));
    }
    if linkers.is_empty() {
        return Err("neither cc nor clang is available in PATH".to_string());
    }
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

    let mut last_err = String::new();
    for (linker, use_lld) in linkers {
        let mut args = Vec::new();
        if linker == "clang" {
            args.push(format!("--target={}", config.target_triple));
        }
        if use_lld {
            args.push("-fuse-ld=lld".to_string());
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
        match run_command(linker, &args) {
            Ok(()) => return Ok(objects),
            Err(err) => last_err = err,
        }
    }

    Err(last_err)
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

    let mut last_err = String::new();
    for use_lld in [true, false] {
        let mut args = Vec::new();
        args.push(format!("--target={}", config.target_triple));
        if use_lld {
            args.push("-fuse-ld=lld".to_string());
        }
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
        match run_command("clang", &args) {
            Ok(()) => return Ok(()),
            Err(err) => last_err = err,
        }
    }
    Err(last_err)
}

fn verify_generics_mode_markers(
    llvm_ir_paths: &[PathBuf],
    shared_generics: bool,
) -> Result<(), String> {
    let expected = if shared_generics { 1_u8 } else { 0_u8 };
    let mut found = Vec::new();
    for path in llvm_ir_paths {
        let text = fs::read_to_string(path).map_err(|err| {
            format!(
                "Could not read LLVM IR artifact '{}' while validating generics mode: {}",
                path.display(),
                err
            )
        })?;
        if let Some(mode) = parse_generics_mode_marker(&text) {
            found.push((path.clone(), mode));
        }
    }

    if found.is_empty() {
        return Err(
            "No `mp$0$ABI$generics_mode` marker found in generated LLVM IR artifacts.".to_string(),
        );
    }

    let bad = found
        .iter()
        .filter(|(_, mode)| *mode != expected)
        .map(|(path, mode)| format!("{} => {}", path.display(), mode))
        .collect::<Vec<_>>();
    if !bad.is_empty() {
        return Err(format!(
            "MIXED_GENERICS_MODE: expected mode {} but found mismatched markers [{}].",
            expected,
            bad.join(", ")
        ));
    }

    Ok(())
}

fn parse_generics_mode_marker(llvm_ir: &str) -> Option<u8> {
    for line in llvm_ir.lines() {
        if !line.contains("mp$0$ABI$generics_mode") {
            continue;
        }
        let Some(idx) = line.find("constant i8") else {
            continue;
        };
        let tail = &line[idx + "constant i8".len()..];
        let digits = tail
            .trim_start()
            .chars()
            .take_while(|ch| ch.is_ascii_digit())
            .collect::<String>();
        if let Ok(value) = digits.parse::<u8>() {
            return Some(value);
        }
    }
    None
}

fn runtime_linker_args(config: &DriverConfig) -> Vec<String> {
    let runtime_dir = find_runtime_lib_dir(config).unwrap_or_else(|| {
        build_output_root(config)
            .join(&config.target_triple)
            .join(config.profile.as_str())
    });
    let mut args = Vec::new();

    // Reset language mode so archive/object files are not treated as LLVM IR
    args.push("-x".to_string());
    args.push("none".to_string());

    if let Some(static_lib) = find_static_runtime_library(config) {
        args.push(static_lib.to_string_lossy().to_string());
    } else {
        args.push(format!("-L{}", runtime_dir.to_string_lossy()));
        args.push("-lmagpie_rt".to_string());
    }

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

fn find_static_runtime_library(config: &DriverConfig) -> Option<PathBuf> {
    let names = if is_windows_target(&config.target_triple) {
        vec!["magpie_rt.lib", "libmagpie_rt.lib"]
    } else {
        vec!["libmagpie_rt.a"]
    };

    runtime_library_search_paths(config)
        .into_iter()
        .find_map(|dir| {
            names
                .iter()
                .map(|name| dir.join(name))
                .find(|path| path.is_file())
        })
}

fn runtime_library_search_paths(config: &DriverConfig) -> Vec<PathBuf> {
    let target_root = config
        .cache_dir
        .as_deref()
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from))
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
    // Cargo uses "debug" directory for the "dev" profile
    if config.profile.as_str() == "dev" {
        push_unique(target_root.join(&config.target_triple).join("debug"));
        push_unique(target_root.join("debug"));
        push_unique(target_root.join("debug").join("deps"));
    }
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
        AstOp::Call { callee, .. }
        | AstOp::Try { callee, .. }
        | AstOp::SuspendCall { callee, .. } => {
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
    let module_path = ast.header.node.module_path.node.to_string();
    let module_sid = generate_sid('M', &module_path);
    let source_digest = ast.header.node.digest.node.as_str();

    let mut out = String::new();
    out.push_str("module ");
    out.push_str(&module_path);
    out.push('\n');
    out.push_str("module_path: ");
    out.push_str(&module_path);
    out.push('\n');
    out.push_str("module_sid: ");
    out.push_str(&module_sid.0);
    out.push('\n');
    out.push_str("source_digest: ");
    out.push_str(source_digest);
    out.push('\n');
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
                let fqn = symbol_fqn(&module_path, &s.name);
                out.push_str("    sid: ");
                out.push_str(&generate_sid('T', &fqn).0);
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
                let fqn = symbol_fqn(&module_path, &s.name);
                out.push_str("    sid: ");
                out.push_str(&generate_sid('T', &fqn).0);
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
                let fqn = symbol_fqn(&module_path, &e.name);
                out.push_str("    sid: ");
                out.push_str(&generate_sid('T', &fqn).0);
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
                let fqn = symbol_fqn(&module_path, &e.name);
                out.push_str("    sid: ");
                out.push_str(&generate_sid('T', &fqn).0);
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
                let fqn = symbol_fqn(&module_path, &f.name);
                out.push_str("    sid: ");
                out.push_str(&generate_sid('F', &fqn).0);
                out.push('\n');
                push_doc_lines(&mut out, "    ", f.doc.as_deref());
            }
            AstDecl::AsyncFn(f) => {
                wrote_fn = true;
                out.push_str("  ");
                out.push_str(&render_fn_signature("async fn", f, None));
                out.push('\n');
                let fqn = symbol_fqn(&module_path, &f.name);
                out.push_str("    sid: ");
                out.push_str(&generate_sid('F', &fqn).0);
                out.push('\n');
                push_doc_lines(&mut out, "    ", f.doc.as_deref());
            }
            AstDecl::UnsafeFn(f) => {
                wrote_fn = true;
                out.push_str("  ");
                out.push_str(&render_fn_signature("unsafe fn", f, None));
                out.push('\n');
                let fqn = symbol_fqn(&module_path, &f.name);
                out.push_str("    sid: ");
                out.push_str(&generate_sid('F', &fqn).0);
                out.push('\n');
                push_doc_lines(&mut out, "    ", f.doc.as_deref());
            }
            AstDecl::GpuFn(gpu) => {
                wrote_fn = true;
                out.push_str("  ");
                out.push_str(&render_fn_signature(
                    "gpu fn",
                    &gpu.inner,
                    Some(&gpu.target),
                ));
                out.push('\n');
                let fqn = symbol_fqn(&module_path, &gpu.inner.name);
                out.push_str("    sid: ");
                out.push_str(&generate_sid('F', &fqn).0);
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

fn symbol_fqn(module_path: &str, local_name: &str) -> String {
    if module_path.is_empty() {
        local_name.to_string()
    } else {
        format!("{}.{}", module_path, local_name)
    }
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
                    format!("{}.{}", path, name)
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
    let mut planned = Vec::new();
    result.success = !has_errors(&result.diagnostics);
    if result.success {
        planned = planned_artifacts(config, &mut result.diagnostics);
        for artifact in &planned {
            if Path::new(artifact).exists() && !result.artifacts.contains(artifact) {
                result.artifacts.push(artifact.clone());
            } else if !Path::new(artifact).exists() {
                // Check for multi-module indexed variants (e.g. hello.0.mpir, hello.1.mpir)
                let p = Path::new(artifact);
                if let (Some(parent), Some(stem), Some(ext)) = (
                    p.parent(),
                    p.file_stem().and_then(|s| s.to_str()),
                    p.extension().and_then(|e| e.to_str()),
                ) {
                    let mut idx = 0;
                    loop {
                        let indexed = parent.join(format!("{stem}.{idx}.{ext}"));
                        if indexed.exists() {
                            let s = indexed.to_string_lossy().to_string();
                            if !result.artifacts.contains(&s) {
                                result.artifacts.push(s);
                            }
                            idx += 1;
                        } else {
                            break;
                        }
                    }
                    // If we found at least one indexed artifact, add the planned name
                    // to signal it was satisfied
                    if idx > 0 && !planned.contains(artifact) {
                        // Artifact requirement met via indexed files
                    }
                }
            }
        }
    }

    if emit_contains(&config.emit, "mpdbg") {
        let mpdbg_path = stage_mpdbg_output_path(config);
        let payload = canonical_json_encode(&build_mpdbg_payload(config, &result))
            .unwrap_or_else(|_| "{}".to_string());
        match write_text_artifact(&mpdbg_path, &payload) {
            Ok(()) => {
                let artifact = mpdbg_path.to_string_lossy().to_string();
                if !result.artifacts.contains(&artifact) {
                    result.artifacts.push(artifact);
                }
            }
            Err(err) => result.diagnostics.push(simple_diag(
                "MPP0003",
                Severity::Error,
                "failed to write mpdbg artifact",
                err,
            )),
        }
    }

    if !has_errors(&result.diagnostics) {
        validate_requested_artifacts(&mut result, config, &planned);
    }

    result.success = !has_errors(&result.diagnostics);
    result
}

fn validate_requested_artifacts(
    result: &mut BuildResult,
    config: &DriverConfig,
    planned: &[String],
) {
    let mut missing = planned
        .iter()
        .filter(|path| {
            if Path::new(path.as_str()).exists() {
                return false;
            }
            // Check for multi-module indexed variants (e.g. hello.0.mpir)
            let p = Path::new(path.as_str());
            if let (Some(parent), Some(stem), Some(ext)) = (
                p.parent(),
                p.file_stem().and_then(|s| s.to_str()),
                p.extension().and_then(|e| e.to_str()),
            ) {
                let indexed = parent.join(format!("{stem}.0.{ext}"));
                if indexed.exists() {
                    return false; // satisfied by indexed files
                }
            }
            true
        })
        .cloned()
        .collect::<Vec<_>>();

    if emit_contains(&config.emit, "object") {
        let has_object = result.artifacts.iter().any(|artifact| {
            (artifact.ends_with(".o") || artifact.ends_with(".obj")) && Path::new(artifact).exists()
        });
        if !has_object {
            missing.push("<object>".to_string());
        }
    }

    if missing.is_empty() {
        return;
    }

    missing.sort();
    missing.dedup();
    result.diagnostics.push(simple_diag(
        "MPL0002",
        Severity::Error,
        "requested artifact(s) missing",
        format!(
            "Requested emit artifact(s) were not produced: {}.",
            missing.join(", ")
        ),
    ));
}

fn build_mpdbg_payload(config: &DriverConfig, result: &BuildResult) -> serde_json::Value {
    let mut by_severity = BTreeMap::from([
        ("error".to_string(), 0_u64),
        ("warning".to_string(), 0_u64),
        ("info".to_string(), 0_u64),
        ("hint".to_string(), 0_u64),
    ]);
    let mut by_code: BTreeMap<String, u64> = BTreeMap::new();
    for diagnostic in &result.diagnostics {
        let severity_key = match diagnostic.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Info => "info",
            Severity::Hint => "hint",
        };
        *by_severity.entry(severity_key.to_string()).or_insert(0) += 1;
        *by_code.entry(diagnostic.code.clone()).or_insert(0) += 1;
    }

    let mut stage_timing = Vec::new();
    for stage in PIPELINE_STAGES {
        stage_timing.push(json!({
            "stage": stage,
            "ms": result.timing_ms.get(stage).copied().unwrap_or(0),
        }));
    }

    json!({
        "format": "mpdbg.v0",
        "entry_path": config.entry_path,
        "target_triple": config.target_triple,
        "profile": config.profile.as_str(),
        "emit": config.emit,
        "success": result.success,
        "diagnostics": {
            "count": result.diagnostics.len(),
            "by_severity": by_severity,
            "by_code": by_code,
        },
        "timing_ms": {
            "stages": stage_timing,
            "raw": result.timing_ms,
        },
        "artifacts": result.artifacts,
    })
}

fn append_stage_diagnostics(result: &mut BuildResult, mut bag: DiagnosticBag) -> bool {
    let failed = bag.has_errors();
    for diag in &mut bag.diagnostics {
        if diag.explanation_md.is_none() {
            diag.explanation_md = magpie_diag::explain_code(&diag.code);
        }
        apply_core_fixers(diag);
    }
    result.diagnostics.extend(bag.diagnostics);
    failed
}

fn mark_skipped_from(timing: &mut BTreeMap<String, u64>, from_stage_idx: usize) {
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
    let code_str = code.to_string();
    Diagnostic {
        code: code_str.clone(),
        severity,
        title: title.into(),
        primary_span: None,
        secondary_spans: Vec::new(),
        message: message.into(),
        explanation_md: magpie_diag::explain_code(&code_str),
        why: None,
        suggested_fixes: Vec::new(),
        rag_bundle: Vec::new(),
        related_docs: Vec::new(),
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
    let base = build_output_root(config)
        .join(&config.target_triple)
        .join(config.profile.as_str());
    let is_windows = is_windows_target(&config.target_triple);
    let is_darwin = is_darwin_target(&config.target_triple);

    for emit in &config.emit {
        let path = match emit.as_str() {
            "llvm-ir" => Some(base.join(format!("{stem}.ll"))),
            "llvm-bc" => Some(base.join(format!("{stem}.bc"))),
            // Object outputs are emitted per-LLVM-input unit and discovered dynamically.
            "object" => None,
            "asm" => Some(base.join(format!("{stem}.s"))),
            // SPIR-V outputs are emitted per-kernel under `<target>/<triple>/<profile>/gpu/`.
            // They are discovered dynamically during stage10, so no single planned file applies.
            "spv" => None,
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

    let token_budget = config.token_budget.unwrap_or(DEFAULT_LLM_TOKEN_BUDGET);
    let tokenizer = config
        .llm_tokenizer
        .clone()
        .unwrap_or_else(|| DEFAULT_LLM_TOKENIZER.to_string());
    let policy = config
        .llm_budget_policy
        .clone()
        .unwrap_or_else(|| DEFAULT_LLM_BUDGET_POLICY.to_string());

    Some(json!({
        "token_budget": token_budget,
        "tokenizer": tokenizer,
        "estimated_tokens": serde_json::Value::Null,
        "policy": policy,
        "dropped": [],
    }))
}

fn effective_budget_config(config: &DriverConfig) -> Option<TokenBudget> {
    if !config.llm_mode && config.token_budget.is_none() {
        return None;
    }

    Some(TokenBudget {
        budget: config.token_budget.unwrap_or(DEFAULT_LLM_TOKEN_BUDGET),
        tokenizer: config
            .llm_tokenizer
            .clone()
            .unwrap_or_else(|| DEFAULT_LLM_TOKENIZER.to_string()),
        policy: config
            .llm_budget_policy
            .clone()
            .unwrap_or_else(|| DEFAULT_LLM_BUDGET_POLICY.to_string()),
    })
}

fn default_target_triple() -> String {
    let arch = std::env::consts::ARCH;
    let os = std::env::consts::OS;
    match os {
        "macos" => format!("{}-apple-macos", arch),
        "windows" => format!("{}-pc-windows-msvc", arch),
        _ => format!("{}-unknown-{}", arch, os),
    }
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
    fn resolve_dependencies_for_build_writes_lockfile_for_version_dep() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "magpie_driver_pkg_preflight_{}_{}",
            std::process::id(),
            nonce
        ));
        std::fs::create_dir_all(root.join("src")).expect("failed to create temp source dir");
        std::fs::write(
            root.join("Magpie.toml"),
            r#"[package]
name = "demo"
version = "0.1.0"
edition = "2026"

[build]
entry = "src/main.mp"
profile_default = "dev"

[dependencies]
std = { version = "^0.1" }
"#,
        )
        .expect("failed to write temp manifest");
        std::fs::write(
            root.join("src/main.mp"),
            r#"module demo.main
exports { @main }
imports { }
digest "0000000000000000"

fn @main() -> i32 {
bb0:
  ret const.i32 0
}
"#,
        )
        .expect("failed to write temp entry");

        let entry = root.join("src/main.mp");
        let lock_path = resolve_dependencies_for_build(entry.to_string_lossy().as_ref(), false)
            .expect("dependency resolution should succeed")
            .expect("manifest should be discovered");
        assert!(lock_path.is_file(), "lockfile should be written");

        let lock = magpie_pkg::read_lockfile(&lock_path).expect("lockfile should parse");
        assert!(lock.packages.iter().any(|pkg| pkg.name == "std"));

        std::fs::remove_dir_all(&root).expect("failed to cleanup temp project");
    }

    #[test]
    fn build_uses_manifest_build_entry_when_default_entry_path_is_requested() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "magpie_driver_manifest_entry_{}_{}",
            std::process::id(),
            nonce
        ));
        std::fs::create_dir_all(root.join("src")).expect("failed to create temp source dir");
        std::fs::write(
            root.join("Magpie.toml"),
            r#"[package]
name = "demo"
version = "0.1.0"
edition = "2026"

[build]
entry = "src/app.mp"
profile_default = "dev"
"#,
        )
        .expect("failed to write temp manifest");
        std::fs::write(
            root.join("src/app.mp"),
            r#"module demo.app
exports { @main }
imports { }
digest "0000000000000000"

fn @main() -> i32 {
bb0:
  ret const.i32 0
}
"#,
        )
        .expect("failed to write temp entry");

        let mut config = DriverConfig::default();
        config.entry_path = root.join("src/main.mp").to_string_lossy().to_string();
        config.target_triple = format!("manifest-entry-test-{}-{}", std::process::id(), nonce);
        config.emit = vec!["mpir".to_string()];

        let result = build(&config);
        assert!(
            result.success,
            "build should succeed with manifest entry override: {:?}",
            result.diagnostics
        );
        assert!(
            result
                .artifacts
                .iter()
                .any(|artifact| artifact.ends_with("app.mpir")),
            "expected app.mpir artifact, got {:?}",
            result.artifacts
        );

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(PathBuf::from("target").join(&config.target_triple));
    }

    #[test]
    fn test_parse_c_header_functions_and_render_extern_module() {
        let header = r#"
            int add(int a, int b);
            const char* version(void);
            void write_bytes(unsigned char* data, unsigned long len);
            MyType open_handle(void);
            void mutate_handle(MyType* handle);
        "#;

        let funcs = parse_c_header_functions(header);
        assert_eq!(funcs.len(), 5);
        assert_eq!(funcs[0].name, "add");
        assert_eq!(funcs[0].ret_ty, "i32");
        assert_eq!(
            funcs[0].params,
            vec![
                ("a".to_string(), "i32".to_string()),
                ("b".to_string(), "i32".to_string())
            ]
        );
        assert_eq!(funcs[1].ret_ty, "Str");
        assert_eq!(funcs[2].params[0].1, "Str");
        assert_eq!(funcs[3].ret_ty, "rawptr<u8>");
        assert_eq!(funcs[4].params[0].1, "rawptr<u8>");

        let rendered = render_extern_module("ffi_import", &funcs);
        assert!(rendered.contains("extern \"C\" module ffi_import"));
        assert!(rendered.contains("fn @add(%a: i32, %b: i32) -> i32"));
        assert!(rendered.contains("fn @version() -> Str"));
        assert!(rendered.contains("returns=\"borrowed\""));
        assert!(rendered.contains("params=\"borrowed\""));
    }

    #[test]
    fn test_output_envelope_graphs_default_shape() {
        assert_eq!(
            output_envelope_graphs(&[]),
            serde_json::json!({
                "symbols": {},
                "deps": {},
                "ownership": {},
                "cfg": {},
            })
        );
    }

    #[test]
    fn test_output_envelope_graphs_pickup_from_artifacts() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        let temp_dir = std::env::temp_dir().join(format!(
            "magpie_driver_graph_pickup_{}_{}",
            std::process::id(),
            nonce
        ));
        std::fs::create_dir_all(&temp_dir).expect("failed to create graph pickup temp dir");

        let sym_path = temp_dir.join("main.symgraph.json");
        let deps_path = temp_dir.join("main.depsgraph.json");
        let ownership_path = temp_dir.join("main.ownershipgraph.json");
        let cfg_path = temp_dir.join("main.cfggraph.json");

        std::fs::write(&sym_path, r#"{"symbols":["S"]}"#).expect("write symgraph");
        std::fs::write(&deps_path, r#"{"deps":["#).expect("write malformed depsgraph");
        std::fs::write(&ownership_path, r#"{"owners":["O"]}"#).expect("write ownershipgraph");
        std::fs::write(&cfg_path, r#"{"blocks":["B0"]}"#).expect("write cfggraph");

        let graphs = output_envelope_graphs(&[
            sym_path.to_string_lossy().to_string(),
            deps_path.to_string_lossy().to_string(),
            ownership_path.to_string_lossy().to_string(),
            cfg_path.to_string_lossy().to_string(),
        ]);

        assert_eq!(graphs["symbols"], serde_json::json!({"symbols": ["S"]}));
        assert_eq!(graphs["deps"], serde_json::json!({}));
        assert_eq!(graphs["ownership"], serde_json::json!({"owners": ["O"]}));
        assert_eq!(graphs["cfg"], serde_json::json!({"blocks": ["B0"]}));

        std::fs::remove_dir_all(&temp_dir).expect("failed to clean up graph pickup temp dir");
    }

    #[test]
    fn test_module_path_to_stage1_file_path_conventions() {
        assert_eq!(
            module_path_to_stage1_file_path("demo.main"),
            Some(PathBuf::from("src/main.mp"))
        );
        assert_eq!(
            module_path_to_stage1_file_path("demo.net.http"),
            Some(PathBuf::from("src/net/http.mp"))
        );
        assert_eq!(
            module_path_to_stage1_file_path("std.io"),
            Some(PathBuf::from("std/std.io/io.mp"))
        );
        assert_eq!(module_path_to_stage1_file_path("std.io.extra"), None);
    }

    #[test]
    fn test_pop_first_module_path_is_lexicographic() {
        let mut pending = BTreeSet::from([
            "pkg.z".to_string(),
            "pkg.a".to_string(),
            "pkg.m".to_string(),
        ]);

        assert_eq!(
            pop_first_module_path(&mut pending),
            Some("pkg.a".to_string())
        );
        assert_eq!(
            pop_first_module_path(&mut pending),
            Some("pkg.m".to_string())
        );
        assert_eq!(
            pop_first_module_path(&mut pending),
            Some("pkg.z".to_string())
        );
        assert_eq!(pop_first_module_path(&mut pending), None);
    }

    #[test]
    fn test_stage_parse_ast_output_path_is_deterministic() {
        let config = DriverConfig {
            entry_path: "src/hello.mp".to_string(),
            profile: BuildProfile::Release,
            target_triple: "x86_64-unknown-linux-gnu".to_string(),
            ..DriverConfig::default()
        };

        let expected = PathBuf::from("target")
            .join("x86_64-unknown-linux-gnu")
            .join("release")
            .join("hello.ast.txt");

        assert_eq!(stage_parse_ast_output_path(&config), expected);
        assert_eq!(stage_parse_ast_output_path(&config), expected);
    }

    #[test]
    fn test_stage_mms_memory_index_output_path_is_deterministic() {
        let config = DriverConfig {
            entry_path: "src/hello.mp".to_string(),
            profile: BuildProfile::Release,
            target_triple: "x86_64-unknown-linux-gnu".to_string(),
            ..DriverConfig::default()
        };

        let expected = PathBuf::from(".magpie")
            .join("memory")
            .join("hello.mms_index.json");

        assert_eq!(stage_mms_memory_index_output_path(&config), expected);
        assert_eq!(stage_mms_memory_index_output_path(&config), expected);
    }

    #[test]
    fn test_parse_generics_mode_marker_extracts_value() {
        let ir0 = "@\"mp$0$ABI$generics_mode\" = weak_odr constant i8 0";
        let ir1 = "@\"mp$0$ABI$generics_mode\" = weak_odr constant i8 1";
        assert_eq!(parse_generics_mode_marker(ir0), Some(0));
        assert_eq!(parse_generics_mode_marker(ir1), Some(1));
        assert_eq!(parse_generics_mode_marker("; no marker"), None);
    }

    #[test]
    fn test_verify_generics_mode_markers_detects_mismatch() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "magpie_driver_generics_mode_check_{}_{}",
            std::process::id(),
            nonce
        ));
        std::fs::create_dir_all(&root).expect("failed to create temp dir");

        let a = root.join("a.ll");
        let b = root.join("b.ll");
        std::fs::write(&a, "@\"mp$0$ABI$generics_mode\" = weak_odr constant i8 0\n")
            .expect("write a.ll");
        std::fs::write(&b, "@\"mp$0$ABI$generics_mode\" = weak_odr constant i8 1\n")
            .expect("write b.ll");

        let err =
            verify_generics_mode_markers(&[a, b], false).expect_err("mismatch should be rejected");
        assert!(err.contains("MIXED_GENERICS_MODE"));

        std::fs::remove_dir_all(root).expect("cleanup temp dir");
    }

    #[test]
    fn test_finalize_build_result_writes_structured_mpdbg_sidecar() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        let target_triple = format!("mpdbg-test-{}-{nonce}", std::process::id());
        let config = DriverConfig {
            entry_path: "src/hello.mp".to_string(),
            target_triple: target_triple.clone(),
            emit: vec!["mpdbg".to_string()],
            ..DriverConfig::default()
        };
        let mut result = BuildResult::default();
        result.timing_ms.insert(STAGE_1.to_string(), 7);
        result.diagnostics.push(simple_diag(
            "MPL2001",
            Severity::Warning,
            "lint",
            "example warning",
        ));

        let finalized = finalize_build_result(result, &config);
        let mpdbg_path = stage_mpdbg_output_path(&config);
        assert!(
            mpdbg_path.is_file(),
            "expected mpdbg sidecar at {}",
            mpdbg_path.display()
        );
        assert!(
            finalized
                .artifacts
                .contains(&mpdbg_path.to_string_lossy().to_string()),
            "finalized artifacts should include mpdbg path"
        );

        let payload = std::fs::read_to_string(&mpdbg_path).expect("read mpdbg payload");
        assert!(!payload.contains("mpdbg.v0.stub"));
        let value: serde_json::Value =
            serde_json::from_str(&payload).expect("mpdbg payload should be valid json");
        assert_eq!(value["format"], "mpdbg.v0");
        assert_eq!(value["diagnostics"]["by_severity"]["warning"], 1);
        assert_eq!(value["timing_ms"]["stages"][0]["stage"], STAGE_1);

        let _ = std::fs::remove_dir_all(PathBuf::from("target").join(target_triple));
    }

    #[test]
    fn test_finalize_build_result_reports_missing_requested_artifacts() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        let target_triple = format!("missing-artifact-test-{}-{nonce}", std::process::id());
        let config = DriverConfig {
            entry_path: "src/hello.mp".to_string(),
            target_triple: target_triple.clone(),
            emit: vec!["mpd".to_string()],
            ..DriverConfig::default()
        };

        let result = BuildResult {
            success: true,
            diagnostics: Vec::new(),
            artifacts: Vec::new(),
            timing_ms: BTreeMap::new(),
        };
        let finalized = finalize_build_result(result, &config);
        assert!(
            !finalized.success,
            "missing requested artifacts must fail finalization"
        );
        assert!(
            finalized
                .diagnostics
                .iter()
                .any(|diag| diag.code == "MPL0002"),
            "expected MPL0002 diagnostic, got {:?}",
            finalized
                .diagnostics
                .iter()
                .map(|diag| diag.code.clone())
                .collect::<Vec<_>>()
        );

        let _ = std::fs::remove_dir_all(PathBuf::from("target").join(target_triple));
    }

    #[test]
    fn test_append_stage_diagnostics_injects_core_fix_and_patch_digests() {
        let mut result = BuildResult::default();
        let mut bag = DiagnosticBag::new(8);
        bag.emit(simple_diag(
            "MPT1023",
            Severity::Error,
            "missing trait impl",
            "missing required trait impl: `impl eq for demo.Key`.",
        ));

        let failed = append_stage_diagnostics(&mut result, bag);
        assert!(failed, "error diagnostics should mark stage as failed");
        assert_eq!(result.diagnostics.len(), 1);

        let diag = &result.diagnostics[0];
        assert!(
            diag.explanation_md.is_some(),
            "driver should hydrate explanation templates"
        );
        assert_eq!(diag.suggested_fixes.len(), 1);

        let fix = &diag.suggested_fixes[0];
        assert_eq!(fix.patch_format, "unified-diff");
        assert!(fix.patch.contains("impl eq for demo.Key"));
        assert!(
            !fix.applies_to.is_empty(),
            "applies_to digest map is required"
        );
        assert!(!fix.produces.is_empty(), "produces digest map is required");
    }

    #[test]
    fn test_lint_fixer_includes_patch_digest_envelopes() {
        let fix = suggested_fix_unused_local("src/main.mp", 7);
        assert_eq!(fix.patch_format, "unified-diff");
        assert!(!fix.applies_to.is_empty());
        assert!(!fix.produces.is_empty());
        assert!(fix.applies_to.contains_key("src/main.mp"));
        assert!(fix.produces.contains_key("src/main.mp"));
    }

    #[test]
    fn test_core_fixers_cover_required_templates() {
        let cases = vec![
            (
                "MPS0002",
                "Cannot resolve import 'dep.math::sum'.",
                "imports { std::sum }",
            ),
            (
                "MPO0103",
                "map.get requires Dupable value type",
                "map.contains_key",
            ),
            ("MPO0004", "shared mutation forbidden", "clone.shared"),
            ("MPO0101", "borrow crosses block boundary", "borrow.shared"),
            (
                "MPT2032",
                "impl 'eq' for 'demo.Key' references unknown local function '@eq_key'.",
                "impl eq for demo.Key",
            ),
        ];

        for (code, message, patch_snippet) in cases {
            let mut result = BuildResult::default();
            let mut bag = DiagnosticBag::new(8);
            bag.emit(simple_diag(code, Severity::Error, "case", message));
            let _ = append_stage_diagnostics(&mut result, bag);

            let diag = result
                .diagnostics
                .first()
                .expect("diagnostic should be present");
            assert!(
                !diag.suggested_fixes.is_empty(),
                "expected fixer for code {code}"
            );
            let fix = &diag.suggested_fixes[0];
            assert_eq!(fix.patch_format, "unified-diff");
            assert!(
                fix.patch.contains(patch_snippet),
                "expected patch for {code} to contain '{patch_snippet}', got: {}",
                fix.patch
            );
            assert!(
                !fix.applies_to.is_empty() && !fix.produces.is_empty(),
                "expected digest maps for {code}"
            );
        }
    }

    #[test]
    fn test_parse_entry_success_writes_ast_artifact() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        let temp_dir = std::env::temp_dir().join(format!(
            "magpie_driver_parse_entry_{}_{}",
            std::process::id(),
            nonce
        ));
        std::fs::create_dir_all(&temp_dir).expect("failed to create parse test temp dir");

        let entry_path = temp_dir.join("parse_success.mp");
        std::fs::write(
            &entry_path,
            r#"module parse_success.main
exports { @main }
imports { }
digest "0000000000000000"

fn @main() -> i32 {
bb0:
  ret const.i32 0
}
"#,
        )
        .expect("failed to write parse test source");

        let mut config = DriverConfig::default();
        config.entry_path = entry_path.to_string_lossy().to_string();
        config.target_triple = format!("parse-entry-test-{}-{}", std::process::id(), nonce);

        let ast_path = stage_parse_ast_output_path(&config);
        if ast_path.exists() {
            std::fs::remove_file(&ast_path).expect("failed to remove stale ast artifact");
        }

        let result = parse_entry(&config);
        assert!(
            result.success,
            "parse_entry should succeed, diagnostics: {:?}",
            result.diagnostics
        );
        assert!(
            result
                .artifacts
                .contains(&ast_path.to_string_lossy().to_string()),
            "artifact list should include {}",
            ast_path.display()
        );
        let ast_dump =
            std::fs::read_to_string(&ast_path).expect("ast artifact should be written on success");
        assert!(ast_dump.contains("AstFile"));
        assert!(result
            .diagnostics
            .iter()
            .all(|diag| !matches!(diag.severity, Severity::Error)));

        std::fs::remove_dir_all(&temp_dir).expect("failed to cleanup parse test temp dir");
        let _ = std::fs::remove_dir_all(PathBuf::from("target").join(&config.target_triple));
    }

    #[test]
    fn test_build_emits_gpu_spv_and_kernel_registry_module() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        let temp_dir = std::env::temp_dir().join(format!(
            "magpie_driver_gpu_emit_{}_{}",
            std::process::id(),
            nonce
        ));
        std::fs::create_dir_all(&temp_dir).expect("failed to create gpu temp dir");

        let entry_path = temp_dir.join("gpu_emit.mp");
        std::fs::write(
            &entry_path,
            r#"module gpumod.emit
exports { @main }
imports { }
digest "0000000000000000"

gpu fn @k() -> unit target(spv) {
bb0:
  ret
}

fn @main() -> i32 {
bb0:
  ret const.i32 0
}
"#,
        )
        .expect("failed to write gpu test source");

        let mut config = DriverConfig::default();
        config.entry_path = entry_path.to_string_lossy().to_string();
        config.target_triple = format!("gpu-emit-test-{}-{}", std::process::id(), nonce);
        config.emit = vec!["spv".to_string()];

        let kernel_sid = generate_sid('F', "gpumod.emit.@k");
        let expected_spv_path = stage_gpu_kernel_spv_output_path(&config, &kernel_sid);
        let registry_path = stage_gpu_registry_output_path(&config);

        let result = build(&config);
        assert!(
            result.success,
            "gpu build should succeed, diagnostics: {:?}",
            result.diagnostics
        );
        assert!(
            expected_spv_path.is_file(),
            "expected spir-v artifact at {}",
            expected_spv_path.display()
        );
        assert!(
            result
                .artifacts
                .contains(&expected_spv_path.to_string_lossy().to_string()),
            "result artifacts should include spir-v output"
        );
        assert!(
            registry_path.is_file(),
            "expected gpu kernel registry ll at {}",
            registry_path.display()
        );
        let registry_ir = std::fs::read_to_string(&registry_path).expect("read registry ir");
        assert!(registry_ir.contains("@mp_gpu_register_all_kernels"));
        assert!(registry_ir.contains("@mp_rt_gpu_register_kernels"));

        std::fs::remove_dir_all(&temp_dir).expect("failed to cleanup gpu temp dir");
        let _ = std::fs::remove_dir_all(PathBuf::from("target").join(&config.target_triple));
    }

    #[test]
    fn test_build_with_shared_generics_flag_emits_mode_marker() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        let temp_dir = std::env::temp_dir().join(format!(
            "magpie_driver_shared_generics_{}_{}",
            std::process::id(),
            nonce
        ));
        std::fs::create_dir_all(&temp_dir).expect("failed to create temp dir");

        let entry_path = temp_dir.join("shared_generics.mp");
        std::fs::write(
            &entry_path,
            r#"module sgmod.generics
exports { @main }
imports { }
digest "0000000000000000"

fn @main() -> i32 {
bb0:
  ret const.i32 0
}
"#,
        )
        .expect("failed to write test source");

        let mut config = DriverConfig::default();
        config.entry_path = entry_path.to_string_lossy().to_string();
        config.target_triple = format!("shared-generics-test-{}-{}", std::process::id(), nonce);
        config.emit = vec!["llvm-ir".to_string()];
        config.shared_generics = true;

        let result = build(&config);
        assert!(
            result.success,
            "shared-generics build should succeed, diagnostics: {:?}",
            result.diagnostics
        );

        let llvm_path = stage_module_output_path(&config, 0, 1, "ll");
        let llvm_ir = std::fs::read_to_string(&llvm_path).expect("read generated llvm ir");
        assert!(llvm_ir.contains("\"mp$0$ABI$generics_mode\""));
        assert!(llvm_ir.contains("constant i8 1"));

        std::fs::remove_dir_all(&temp_dir).expect("failed to cleanup temp dir");
        let _ = std::fs::remove_dir_all(PathBuf::from("target").join(&config.target_triple));
    }

    #[test]
    fn test_render_mpd_includes_sid_and_digest_metadata() {
        let source = r#"module docs.docmeta
exports { TDocType, @main }
imports { }
digest "deadbeefcafebabe"

heap struct TDocType {
  field v: i32
}

fn @main() -> i32 {
bb0:
  ret const.i32 0
}
"#;

        let mut diag = DiagnosticBag::new(DEFAULT_MAX_ERRORS);
        let file_id = FileId(0);
        let tokens = lex(file_id, source, &mut diag);
        let ast = parse_file(&tokens, file_id, &mut diag).expect("source should parse");
        assert!(
            diag.diagnostics
                .iter()
                .all(|d| !matches!(d.severity, Severity::Error)),
            "unexpected parse diagnostics: {:?}",
            diag.diagnostics
        );

        let mpd = render_mpd(&ast);
        assert!(mpd.contains("module docs.docmeta"));
        assert!(mpd.contains("module_path: docs.docmeta"));
        assert!(mpd.contains("source_digest: deadbeefcafebabe"));
        assert!(mpd.contains(&format!(
            "module_sid: {}",
            generate_sid('M', "docs.docmeta").0
        )));
        assert!(mpd.contains(&format!(
            "    sid: {}",
            generate_sid('T', "docs.docmeta.TDocType").0
        )));
        assert!(mpd.contains(&format!(
            "    sid: {}",
            generate_sid('F', "docs.docmeta.@main").0
        )));
    }

    #[test]
    fn test_remap_hir_modules_type_ids_updates_type_payloads() {
        let old = |n: u32| TypeId(1000 + n);
        let new = |n: u32| TypeId(2000 + n);
        let mut remap = HashMap::new();
        for n in 0..=41 {
            remap.insert(old(n), new(n));
        }

        let mut module = HirModule {
            module_id: magpie_types::ModuleId(0),
            sid: Sid("M:TESTMODULE0".to_string()),
            path: "test.mod".to_string(),
            functions: vec![HirFunction {
                fn_id: magpie_types::FnId(0),
                sid: Sid("F:TESTFUNC00".to_string()),
                name: "f".to_string(),
                params: vec![(LocalId(0), old(0))],
                ret_ty: old(1),
                blocks: vec![HirBlock {
                    id: BlockId(0),
                    instrs: vec![
                        HirInstr {
                            dst: LocalId(1),
                            ty: old(2),
                            op: HirOp::Const(HirConst {
                                ty: old(3),
                                lit: HirConstLit::IntLit(1),
                            }),
                        },
                        HirInstr {
                            dst: LocalId(2),
                            ty: old(4),
                            op: HirOp::New {
                                ty: old(5),
                                fields: vec![(
                                    "x".to_string(),
                                    HirValue::Const(HirConst {
                                        ty: old(6),
                                        lit: HirConstLit::IntLit(2),
                                    }),
                                )],
                            },
                        },
                        HirInstr {
                            dst: LocalId(3),
                            ty: old(7),
                            op: HirOp::Cast {
                                to: old(8),
                                v: HirValue::Const(HirConst {
                                    ty: old(9),
                                    lit: HirConstLit::IntLit(3),
                                }),
                            },
                        },
                        HirInstr {
                            dst: LocalId(4),
                            ty: old(10),
                            op: HirOp::PtrLoad {
                                to: old(11),
                                p: HirValue::Const(HirConst {
                                    ty: old(12),
                                    lit: HirConstLit::IntLit(4),
                                }),
                            },
                        },
                        HirInstr {
                            dst: LocalId(5),
                            ty: old(13),
                            op: HirOp::Call {
                                callee_sid: Sid("F:CALLEE0001".to_string()),
                                inst: vec![old(14)],
                                args: vec![HirValue::Const(HirConst {
                                    ty: old(15),
                                    lit: HirConstLit::IntLit(5),
                                })],
                            },
                        },
                        HirInstr {
                            dst: LocalId(6),
                            ty: old(16),
                            op: HirOp::Phi {
                                ty: old(17),
                                incomings: vec![(
                                    BlockId(0),
                                    HirValue::Const(HirConst {
                                        ty: old(18),
                                        lit: HirConstLit::IntLit(6),
                                    }),
                                )],
                            },
                        },
                        HirInstr {
                            dst: LocalId(7),
                            ty: old(19),
                            op: HirOp::ArrNew {
                                elem_ty: old(20),
                                cap: HirValue::Const(HirConst {
                                    ty: old(21),
                                    lit: HirConstLit::IntLit(7),
                                }),
                            },
                        },
                        HirInstr {
                            dst: LocalId(8),
                            ty: old(22),
                            op: HirOp::MapNew {
                                key_ty: old(23),
                                val_ty: old(24),
                            },
                        },
                        HirInstr {
                            dst: LocalId(9),
                            ty: old(25),
                            op: HirOp::JsonEncode {
                                ty: old(26),
                                v: HirValue::Const(HirConst {
                                    ty: old(27),
                                    lit: HirConstLit::IntLit(8),
                                }),
                            },
                        },
                        HirInstr {
                            dst: LocalId(10),
                            ty: old(28),
                            op: HirOp::GpuShared {
                                ty: old(29),
                                size: HirValue::Const(HirConst {
                                    ty: old(30),
                                    lit: HirConstLit::IntLit(9),
                                }),
                            },
                        },
                    ],
                    void_ops: vec![
                        HirOpVoid::CallVoid {
                            callee_sid: Sid("F:VOIDCAL001".to_string()),
                            inst: vec![old(31)],
                            args: vec![HirValue::Const(HirConst {
                                ty: old(32),
                                lit: HirConstLit::IntLit(10),
                            })],
                        },
                        HirOpVoid::PtrStore {
                            to: old(33),
                            p: HirValue::Const(HirConst {
                                ty: old(34),
                                lit: HirConstLit::IntLit(11),
                            }),
                            v: HirValue::Const(HirConst {
                                ty: old(35),
                                lit: HirConstLit::IntLit(12),
                            }),
                        },
                    ],
                    terminator: HirTerminator::Switch {
                        val: HirValue::Const(HirConst {
                            ty: old(36),
                            lit: HirConstLit::IntLit(0),
                        }),
                        arms: vec![(
                            HirConst {
                                ty: old(37),
                                lit: HirConstLit::IntLit(1),
                            },
                            BlockId(1),
                        )],
                        default: BlockId(2),
                    },
                }],
                is_async: false,
                is_unsafe: false,
            }],
            globals: vec![magpie_hir::HirGlobal {
                id: magpie_types::GlobalId(0),
                name: "g".to_string(),
                ty: old(38),
                init: HirConst {
                    ty: old(39),
                    lit: HirConstLit::IntLit(13),
                },
            }],
            type_decls: vec![
                HirTypeDecl::Struct {
                    sid: Sid("T:STRUCTTEST".to_string()),
                    name: "S".to_string(),
                    fields: vec![("f".to_string(), old(40))],
                },
                HirTypeDecl::Enum {
                    sid: Sid("T:ENUMTEST00".to_string()),
                    name: "E".to_string(),
                    variants: vec![magpie_hir::HirEnumVariant {
                        name: "V".to_string(),
                        tag: 0,
                        fields: vec![("e".to_string(), old(41))],
                    }],
                },
            ],
        };

        remap_hir_modules_type_ids(std::slice::from_mut(&mut module), &remap);

        let func = &module.functions[0];
        assert_eq!(func.params[0].1, new(0));
        assert_eq!(func.ret_ty, new(1));

        assert_eq!(module.globals[0].ty, new(38));
        assert_eq!(module.globals[0].init.ty, new(39));
        match &module.type_decls[0] {
            HirTypeDecl::Struct { fields, .. } => assert_eq!(fields[0].1, new(40)),
            _ => panic!("expected struct decl"),
        }
        match &module.type_decls[1] {
            HirTypeDecl::Enum { variants, .. } => assert_eq!(variants[0].fields[0].1, new(41)),
            _ => panic!("expected enum decl"),
        }

        let block = &func.blocks[0];
        assert_eq!(block.instrs[0].ty, new(2));
        match &block.instrs[0].op {
            HirOp::Const(value) => assert_eq!(value.ty, new(3)),
            _ => panic!("expected const op"),
        }
        match &block.instrs[1].op {
            HirOp::New { ty, fields } => {
                assert_eq!(*ty, new(5));
                match &fields[0].1 {
                    HirValue::Const(value) => assert_eq!(value.ty, new(6)),
                    _ => panic!("expected const field"),
                }
            }
            _ => panic!("expected new op"),
        }
        match &block.instrs[2].op {
            HirOp::Cast { to, v } => {
                assert_eq!(*to, new(8));
                match v {
                    HirValue::Const(value) => assert_eq!(value.ty, new(9)),
                    _ => panic!("expected const cast arg"),
                }
            }
            _ => panic!("expected cast op"),
        }
        match &block.instrs[3].op {
            HirOp::PtrLoad { to, .. } => assert_eq!(*to, new(11)),
            _ => panic!("expected ptrload op"),
        }
        match &block.instrs[4].op {
            HirOp::Call { inst, args, .. } => {
                assert_eq!(inst[0], new(14));
                match &args[0] {
                    HirValue::Const(value) => assert_eq!(value.ty, new(15)),
                    _ => panic!("expected const call arg"),
                }
            }
            _ => panic!("expected call op"),
        }
        match &block.instrs[5].op {
            HirOp::Phi { ty, incomings } => {
                assert_eq!(*ty, new(17));
                match &incomings[0].1 {
                    HirValue::Const(value) => assert_eq!(value.ty, new(18)),
                    _ => panic!("expected const phi incoming"),
                }
            }
            _ => panic!("expected phi op"),
        }
        match &block.instrs[6].op {
            HirOp::ArrNew { elem_ty, .. } => assert_eq!(*elem_ty, new(20)),
            _ => panic!("expected arrnew op"),
        }
        match &block.instrs[7].op {
            HirOp::MapNew { key_ty, val_ty } => {
                assert_eq!(*key_ty, new(23));
                assert_eq!(*val_ty, new(24));
            }
            _ => panic!("expected mapnew op"),
        }
        match &block.instrs[8].op {
            HirOp::JsonEncode { ty, .. } => assert_eq!(*ty, new(26)),
            _ => panic!("expected json encode op"),
        }
        match &block.instrs[9].op {
            HirOp::GpuShared { ty, .. } => assert_eq!(*ty, new(29)),
            _ => panic!("expected gpu shared op"),
        }

        match &block.void_ops[0] {
            HirOpVoid::CallVoid { inst, args, .. } => {
                assert_eq!(inst[0], new(31));
                match &args[0] {
                    HirValue::Const(value) => assert_eq!(value.ty, new(32)),
                    _ => panic!("expected const callvoid arg"),
                }
            }
            _ => panic!("expected callvoid op"),
        }
        match &block.void_ops[1] {
            HirOpVoid::PtrStore { to, .. } => assert_eq!(*to, new(33)),
            _ => panic!("expected ptrstore op"),
        }
        match &block.terminator {
            HirTerminator::Switch { val, arms, .. } => {
                match val {
                    HirValue::Const(value) => assert_eq!(value.ty, new(36)),
                    _ => panic!("expected const switch value"),
                }
                assert_eq!(arms[0].0.ty, new(37));
            }
            _ => panic!("expected switch terminator"),
        }
    }

    #[test]
    fn run_tests_uses_fallback_executable_build() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        let temp_dir = std::env::temp_dir().join(format!(
            "magpie_driver_run_tests_fallback_exec_{}_{}",
            std::process::id(),
            nonce
        ));
        std::fs::create_dir_all(&temp_dir).expect("fallback temp dir should be created");

        let mut config = DriverConfig::default();
        config.target_triple = default_target_triple();
        config.emit = vec!["mpir".to_string()];

        let fallback_exe = if is_windows_target(&config.target_triple) {
            temp_dir.join("fallback_test.exe")
        } else {
            temp_dir.join("fallback_test")
        };
        std::fs::write(&fallback_exe, b"test-binary").expect("fallback executable marker");

        let mut build_calls = 0usize;
        let mut seen_emits = Vec::new();
        let mut seen_runs = Vec::new();
        let fallback_exe_str = fallback_exe.to_string_lossy().to_string();
        let result = run_tests_with(
            &config,
            None,
            |build_config| {
                build_calls += 1;
                seen_emits.push(build_config.emit.clone());
                if build_calls == 1 {
                    BuildResult {
                        success: true,
                        artifacts: vec!["target/fallback_test.ll".to_string()],
                        ..BuildResult::default()
                    }
                } else {
                    BuildResult {
                        success: true,
                        artifacts: vec![fallback_exe_str.clone()],
                        ..BuildResult::default()
                    }
                }
            },
            |_| vec!["@test_alpha".to_string(), "@test_beta".to_string()],
            |path, test_name| {
                seen_runs.push((path.to_string(), test_name.to_string()));
                test_name == "@test_alpha"
            },
        );

        assert_eq!(build_calls, 2, "fallback build should run once");
        assert_eq!(seen_emits[0], vec!["mpir".to_string()]);
        assert!(
            seen_emits[1].iter().any(|emit| emit == "exe"),
            "fallback build should request executable output"
        );
        assert_eq!(result.total, 2);
        assert_eq!(result.passed, 1);
        assert_eq!(result.failed, 1);
        assert_eq!(
            seen_runs,
            vec![
                (fallback_exe_str.clone(), "@test_alpha".to_string()),
                (fallback_exe_str.clone(), "@test_beta".to_string()),
            ]
        );

        std::fs::remove_dir_all(&temp_dir).expect("fallback temp dir should be removed");
    }

    #[test]
    fn run_tests_marks_discovered_tests_failed_without_executable() {
        let mut config = DriverConfig::default();
        config.emit = vec!["mpir".to_string()];

        let mut build_calls = 0usize;
        let result = run_tests_with(
            &config,
            None,
            |build_config| {
                build_calls += 1;
                if build_calls == 1 {
                    assert_eq!(build_config.emit, vec!["mpir".to_string()]);
                } else {
                    assert!(build_config.emit.iter().any(|emit| emit == "exe"));
                }
                BuildResult {
                    success: true,
                    artifacts: vec!["target/tests.mpir".to_string()],
                    ..BuildResult::default()
                }
            },
            |_| vec!["@test_missing_exec".to_string()],
            |_, _| panic!("test binaries should not run without executable"),
        );

        assert_eq!(build_calls, 2, "fallback build should run once");
        assert_eq!(result.total, 1);
        assert_eq!(result.passed, 0);
        assert_eq!(result.failed, 1);
        assert_eq!(
            result.test_names,
            vec![("@test_missing_exec".to_string(), false)]
        );
    }

    #[test]
    fn run_tests_keeps_zero_tests_as_pass_without_executable() {
        let mut config = DriverConfig::default();
        config.emit = vec!["mpir".to_string()];

        let mut build_calls = 0usize;
        let result = run_tests_with(
            &config,
            None,
            |_| {
                build_calls += 1;
                BuildResult {
                    success: true,
                    artifacts: vec!["target/no-tests.mpir".to_string()],
                    ..BuildResult::default()
                }
            },
            |_| Vec::new(),
            |_, _| panic!("no tests should execute"),
        );

        assert_eq!(build_calls, 2, "fallback build should run once");
        assert_eq!(result.total, 0);
        assert_eq!(result.passed, 0);
        assert_eq!(result.failed, 0);
        assert!(result.test_names.is_empty());
    }
}
