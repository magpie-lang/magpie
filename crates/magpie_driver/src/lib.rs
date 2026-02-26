//! Magpie compiler driver (§5.2, §22, §26.1).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use magpie_arc::{insert_arc_ops, optimize_arc};
use magpie_ast::{AstFile, FileId};
use magpie_csnf::{format_csnf, update_digest};
use magpie_diag::{Diagnostic, DiagnosticBag, OutputEnvelope, Severity};
use magpie_hir::{
    verify_hir, HirBlock, HirFunction, HirInstr, HirModule, HirOp, HirOpVoid, HirTerminator,
    HirValue,
};
use magpie_lex::lex;
use magpie_mpir::{
    print_mpir, verify_mpir, MpirBlock, MpirFn, MpirInstr, MpirLocalDecl, MpirModule, MpirOp,
    MpirOpVoid, MpirTerminator, MpirTypeTable, MpirValue,
};
use magpie_own::check_ownership;
use magpie_parse::parse_file;
use magpie_sema::{lower_to_hir, resolve_modules};
use magpie_types::TypeCtx;
use serde::{Deserialize, Serialize};
use serde_json::json;

extern crate self as regex;

#[derive(Clone, Debug, Default)]
pub struct Regex;

#[derive(Clone, Debug, Default)]
pub struct RegexError;

impl std::fmt::Display for RegexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "regex backend unavailable")
    }
}

impl std::error::Error for RegexError {}

impl Regex {
    pub fn new(pattern: &str) -> Result<Self, RegexError> {
        if pattern.is_empty() {
            Err(RegexError)
        } else {
            Ok(Self)
        }
    }

    pub fn replace_all<'a>(
        &self,
        haystack: &'a str,
        _replacement: &str,
    ) -> std::borrow::Cow<'a, str> {
        std::borrow::Cow::Borrowed(haystack)
    }

    pub fn captures<'a>(&self, _haystack: &'a str) -> Option<Captures<'a>> {
        None
    }

    pub fn captures_iter<'a>(&self, _haystack: &'a str) -> CapturesIter<'a> {
        CapturesIter {
            _marker: std::marker::PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Captures<'a> {
    _marker: std::marker::PhantomData<&'a str>,
}

impl<'a> Captures<'a> {
    pub fn get(&self, _idx: usize) -> Option<Match<'a>> {
        None
    }
}

#[derive(Clone, Debug)]
pub struct Match<'a> {
    text: &'a str,
}

impl<'a> Match<'a> {
    pub fn as_str(&self) -> &'a str {
        self.text
    }
}

pub struct CapturesIter<'a> {
    _marker: std::marker::PhantomData<&'a str>,
}

impl<'a> Iterator for CapturesIter<'a> {
    type Item = Captures<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

#[path = "../../magpie_codegen_llvm/src/lib.rs"]
mod magpie_codegen_llvm;

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
const STAGE_12: &str = "stage12_mms_update";

const PIPELINE_STAGES: [&str; 12] = [
    STAGE_1, STAGE_2, STAGE_3, STAGE_4, STAGE_5, STAGE_6, STAGE_7, STAGE_8, STAGE_9, STAGE_10,
    STAGE_11, STAGE_12,
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
        result
            .timing_ms
            .insert(STAGE_3.to_string(), elapsed_ms(start));
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
        result
            .timing_ms
            .insert(STAGE_4.to_string(), elapsed_ms(start));
        let stage_failed = append_stage_diagnostics(&mut result, diag);
        if stage_failed {
            mark_skipped_from(&mut result.timing_ms, 4);
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
            mark_skipped_from(&mut result.timing_ms, 5);
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

    // Stage 11: link placeholder.
    {
        let start = Instant::now();
        let mut diag = DiagnosticBag::new(max_errors);
        if emit_contains_any(&config.emit, &["exe", "shared-lib", "object", "asm"]) {
            let outputs = llvm_ir_paths
                .iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            emit_driver_diag(
                &mut diag,
                "MPLINK00",
                Severity::Warning,
                "link stage is currently a placeholder",
                format!(
                    "LLVM IR emitted at [{}]. Native linking via llc/clang is not wired in this driver yet.",
                    outputs
                ),
            );
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

    // Stage 12: MMS update placeholder.
    {
        let start = Instant::now();
        result
            .timing_ms
            .insert(STAGE_12.to_string(), elapsed_ms(start));
    }

    finalize_build_result(result, config)
}

fn lower_hir_module_to_mpir(module: &HirModule, type_ctx: &TypeCtx) -> MpirModule {
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

fn write_text_artifact(path: &Path, text: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("Could not create '{}': {}", parent.display(), err))?;
    }
    fs::write(path, text).map_err(|err| format!("Could not write '{}': {}", path.display(), err))
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
    let is_windows = config.target_triple.contains("windows");
    let is_darwin = config.target_triple.contains("apple-darwin");

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
}
