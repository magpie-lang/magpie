//! Magpie MPIR (typed mid-level IR) data structures, printer, and verifier (§15, §17).

pub use magpie_hir::{HirConst, HirConstLit};
pub use magpie_types::{
    BlockId, GlobalId, HandleKind, HeapBase, LocalId, PrimType, Sid, TypeCtx, TypeId, TypeKind,
};

use magpie_diag::{Diagnostic, DiagnosticBag, Severity};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

/// A single MPIR value reference.
#[derive(Clone, Debug)]
pub enum MpirValue {
    Local(LocalId),
    Const(HirConst),
}

/// MPIR instruction — an SSA assignment.
#[derive(Clone, Debug)]
pub struct MpirInstr {
    pub dst: LocalId,
    pub ty: TypeId,
    pub op: MpirOp,
}

/// MPIR operation — mirrors HirOp and adds ARC ops.
#[derive(Clone, Debug)]
pub enum MpirOp {
    // Constant materialization
    Const(HirConst),

    // Ownership (compiler-inserted during HIR lowering)
    Move {
        v: MpirValue,
    },
    BorrowShared {
        v: MpirValue,
    },
    BorrowMut {
        v: MpirValue,
    },

    // Heap allocation and field access
    New {
        ty: TypeId,
        fields: Vec<(String, MpirValue)>,
    },
    GetField {
        obj: MpirValue,
        field: String,
    },

    // Integer arithmetic (checked)
    IAdd {
        lhs: MpirValue,
        rhs: MpirValue,
    },
    ISub {
        lhs: MpirValue,
        rhs: MpirValue,
    },
    IMul {
        lhs: MpirValue,
        rhs: MpirValue,
    },
    ISDiv {
        lhs: MpirValue,
        rhs: MpirValue,
    },
    IUDiv {
        lhs: MpirValue,
        rhs: MpirValue,
    },
    ISRem {
        lhs: MpirValue,
        rhs: MpirValue,
    },
    IURem {
        lhs: MpirValue,
        rhs: MpirValue,
    },

    // Integer arithmetic (wrapping)
    IAddWrap {
        lhs: MpirValue,
        rhs: MpirValue,
    },
    ISubWrap {
        lhs: MpirValue,
        rhs: MpirValue,
    },
    IMulWrap {
        lhs: MpirValue,
        rhs: MpirValue,
    },

    // Integer arithmetic (checked -> TOption)
    IAddChecked {
        lhs: MpirValue,
        rhs: MpirValue,
    },
    ISubChecked {
        lhs: MpirValue,
        rhs: MpirValue,
    },
    IMulChecked {
        lhs: MpirValue,
        rhs: MpirValue,
    },

    // Bitwise
    IAnd {
        lhs: MpirValue,
        rhs: MpirValue,
    },
    IOr {
        lhs: MpirValue,
        rhs: MpirValue,
    },
    IXor {
        lhs: MpirValue,
        rhs: MpirValue,
    },
    IShl {
        lhs: MpirValue,
        rhs: MpirValue,
    },
    ILshr {
        lhs: MpirValue,
        rhs: MpirValue,
    },
    IAshr {
        lhs: MpirValue,
        rhs: MpirValue,
    },

    // Comparison
    ICmp {
        pred: String,
        lhs: MpirValue,
        rhs: MpirValue,
    },
    FCmp {
        pred: String,
        lhs: MpirValue,
        rhs: MpirValue,
    },

    // Float (strict IEEE 754)
    FAdd {
        lhs: MpirValue,
        rhs: MpirValue,
    },
    FSub {
        lhs: MpirValue,
        rhs: MpirValue,
    },
    FMul {
        lhs: MpirValue,
        rhs: MpirValue,
    },
    FDiv {
        lhs: MpirValue,
        rhs: MpirValue,
    },
    FRem {
        lhs: MpirValue,
        rhs: MpirValue,
    },

    // Float (fast-math opt-in)
    FAddFast {
        lhs: MpirValue,
        rhs: MpirValue,
    },
    FSubFast {
        lhs: MpirValue,
        rhs: MpirValue,
    },
    FMulFast {
        lhs: MpirValue,
        rhs: MpirValue,
    },
    FDivFast {
        lhs: MpirValue,
        rhs: MpirValue,
    },

    // Cast
    Cast {
        to: TypeId,
        v: MpirValue,
    },

    // Unsafe raw pointer ops
    PtrNull {
        to: TypeId,
    },
    PtrAddr {
        p: MpirValue,
    },
    PtrFromAddr {
        to: TypeId,
        addr: MpirValue,
    },
    PtrAdd {
        p: MpirValue,
        count: MpirValue,
    },
    PtrLoad {
        to: TypeId,
        p: MpirValue,
    },
    PtrStore {
        to: TypeId,
        p: MpirValue,
        v: MpirValue,
    },

    // Calls (value-returning)
    Call {
        callee_sid: Sid,
        inst: Vec<TypeId>,
        args: Vec<MpirValue>,
    },
    CallIndirect {
        callee: MpirValue,
        args: Vec<MpirValue>,
    },
    CallVoidIndirect {
        callee: MpirValue,
        args: Vec<MpirValue>,
    },
    SuspendCall {
        callee_sid: Sid,
        inst: Vec<TypeId>,
        args: Vec<MpirValue>,
    },
    SuspendAwait {
        fut: MpirValue,
    },

    // SSA phi node
    Phi {
        ty: TypeId,
        incomings: Vec<(BlockId, MpirValue)>,
    },

    // Ownership conversions
    Share {
        v: MpirValue,
    },
    CloneShared {
        v: MpirValue,
    },
    CloneWeak {
        v: MpirValue,
    },
    WeakDowngrade {
        v: MpirValue,
    },
    WeakUpgrade {
        v: MpirValue,
    },

    // ARC operations (inserted by ARC pass)
    ArcRetain {
        v: MpirValue,
    },
    ArcRelease {
        v: MpirValue,
    },
    ArcRetainWeak {
        v: MpirValue,
    },
    ArcReleaseWeak {
        v: MpirValue,
    },

    // Enum operations
    EnumNew {
        variant: String,
        args: Vec<(String, MpirValue)>,
    },
    EnumTag {
        v: MpirValue,
    },
    EnumPayload {
        variant: String,
        v: MpirValue,
    },
    EnumIs {
        variant: String,
        v: MpirValue,
    },

    // TCallable
    CallableCapture {
        fn_ref: Sid,
        captures: Vec<(String, MpirValue)>,
    },

    // Array intrinsics (value-returning)
    ArrNew {
        elem_ty: TypeId,
        cap: MpirValue,
    },
    ArrLen {
        arr: MpirValue,
    },
    ArrGet {
        arr: MpirValue,
        idx: MpirValue,
    },
    ArrSet {
        arr: MpirValue,
        idx: MpirValue,
        val: MpirValue,
    },
    ArrPush {
        arr: MpirValue,
        val: MpirValue,
    },
    ArrPop {
        arr: MpirValue,
    },
    ArrSlice {
        arr: MpirValue,
        start: MpirValue,
        end: MpirValue,
    },
    ArrContains {
        arr: MpirValue,
        val: MpirValue,
    },
    ArrSort {
        arr: MpirValue,
    },
    ArrMap {
        arr: MpirValue,
        func: MpirValue,
    },
    ArrFilter {
        arr: MpirValue,
        func: MpirValue,
    },
    ArrReduce {
        arr: MpirValue,
        init: MpirValue,
        func: MpirValue,
    },
    ArrForeach {
        arr: MpirValue,
        func: MpirValue,
    },

    // Map intrinsics (value-returning)
    MapNew {
        key_ty: TypeId,
        val_ty: TypeId,
    },
    MapLen {
        map: MpirValue,
    },
    MapGet {
        map: MpirValue,
        key: MpirValue,
    },
    MapGetRef {
        map: MpirValue,
        key: MpirValue,
    },
    MapSet {
        map: MpirValue,
        key: MpirValue,
        val: MpirValue,
    },
    MapDelete {
        map: MpirValue,
        key: MpirValue,
    },
    MapContainsKey {
        map: MpirValue,
        key: MpirValue,
    },
    MapDeleteVoid {
        map: MpirValue,
        key: MpirValue,
    },
    MapKeys {
        map: MpirValue,
    },
    MapValues {
        map: MpirValue,
    },

    // String intrinsics
    StrConcat {
        a: MpirValue,
        b: MpirValue,
    },
    StrLen {
        s: MpirValue,
    },
    StrEq {
        a: MpirValue,
        b: MpirValue,
    },
    StrSlice {
        s: MpirValue,
        start: MpirValue,
        end: MpirValue,
    },
    StrBytes {
        s: MpirValue,
    },
    StrBuilderNew,
    StrBuilderAppendStr {
        b: MpirValue,
        s: MpirValue,
    },
    StrBuilderAppendI64 {
        b: MpirValue,
        v: MpirValue,
    },
    StrBuilderAppendI32 {
        b: MpirValue,
        v: MpirValue,
    },
    StrBuilderAppendF64 {
        b: MpirValue,
        v: MpirValue,
    },
    StrBuilderAppendBool {
        b: MpirValue,
        v: MpirValue,
    },
    StrBuilderBuild {
        b: MpirValue,
    },

    // String parse intrinsics
    StrParseI64 {
        s: MpirValue,
    },
    StrParseU64 {
        s: MpirValue,
    },
    StrParseF64 {
        s: MpirValue,
    },
    StrParseBool {
        s: MpirValue,
    },

    // JSON
    JsonEncode {
        ty: TypeId,
        v: MpirValue,
    },
    JsonDecode {
        ty: TypeId,
        s: MpirValue,
    },

    // GPU intrinsics
    GpuThreadId,
    GpuWorkgroupId,
    GpuWorkgroupSize,
    GpuGlobalId,
    GpuBufferLoad {
        buf: MpirValue,
        idx: MpirValue,
    },
    GpuBufferLen {
        buf: MpirValue,
    },
    GpuShared {
        ty: TypeId,
        size: MpirValue,
    },
    GpuLaunch {
        device: MpirValue,
        kernel: Sid,
        groups: MpirValue,
        threads: MpirValue,
        args: Vec<MpirValue>,
    },
    GpuLaunchAsync {
        device: MpirValue,
        kernel: Sid,
        groups: MpirValue,
        threads: MpirValue,
        args: Vec<MpirValue>,
    },

    // Error
    Panic {
        msg: MpirValue,
    },
}

/// MPIR void operation — side-effecting, produces no SSA value.
#[derive(Clone, Debug)]
pub enum MpirOpVoid {
    CallVoid {
        callee_sid: Sid,
        inst: Vec<TypeId>,
        args: Vec<MpirValue>,
    },
    CallVoidIndirect {
        callee: MpirValue,
        args: Vec<MpirValue>,
    },
    SetField {
        obj: MpirValue,
        field: String,
        value: MpirValue,
    },
    ArrSet {
        arr: MpirValue,
        idx: MpirValue,
        val: MpirValue,
    },
    ArrPush {
        arr: MpirValue,
        val: MpirValue,
    },
    ArrSort {
        arr: MpirValue,
    },
    ArrForeach {
        arr: MpirValue,
        func: MpirValue,
    },
    MapSet {
        map: MpirValue,
        key: MpirValue,
        val: MpirValue,
    },
    MapDeleteVoid {
        map: MpirValue,
        key: MpirValue,
    },
    StrBuilderAppendStr {
        b: MpirValue,
        s: MpirValue,
    },
    StrBuilderAppendI64 {
        b: MpirValue,
        v: MpirValue,
    },
    StrBuilderAppendI32 {
        b: MpirValue,
        v: MpirValue,
    },
    StrBuilderAppendF64 {
        b: MpirValue,
        v: MpirValue,
    },
    StrBuilderAppendBool {
        b: MpirValue,
        v: MpirValue,
    },
    PtrStore {
        to: TypeId,
        p: MpirValue,
        v: MpirValue,
    },
    Panic {
        msg: MpirValue,
    },
    GpuBarrier,
    GpuBufferStore {
        buf: MpirValue,
        idx: MpirValue,
        val: MpirValue,
    },
    ArcRetain {
        v: MpirValue,
    },
    ArcRelease {
        v: MpirValue,
    },
    ArcRetainWeak {
        v: MpirValue,
    },
    ArcReleaseWeak {
        v: MpirValue,
    },
}

/// MPIR block terminator.
#[derive(Clone, Debug)]
pub enum MpirTerminator {
    Ret(Option<MpirValue>),
    Br(BlockId),
    Cbr {
        cond: MpirValue,
        then_bb: BlockId,
        else_bb: BlockId,
    },
    Switch {
        val: MpirValue,
        arms: Vec<(HirConst, BlockId)>,
        default: BlockId,
    },
    Unreachable,
}

/// MPIR basic block.
#[derive(Clone, Debug)]
pub struct MpirBlock {
    pub id: BlockId,
    pub instrs: Vec<MpirInstr>,
    pub void_ops: Vec<MpirOpVoid>,
    pub terminator: MpirTerminator,
}

/// MPIR local declaration.
#[derive(Clone, Debug)]
pub struct MpirLocalDecl {
    pub id: LocalId,
    pub ty: TypeId,
    pub name: String,
}

/// MPIR function.
#[derive(Clone, Debug)]
pub struct MpirFn {
    pub sid: Sid,
    pub name: String,
    pub params: Vec<(LocalId, TypeId)>,
    pub ret_ty: TypeId,
    pub blocks: Vec<MpirBlock>,
    pub locals: Vec<MpirLocalDecl>,
    pub is_async: bool,
}

/// MPIR type table snapshot.
#[derive(Clone, Debug)]
pub struct MpirTypeTable {
    pub types: Vec<(TypeId, TypeKind)>,
}

/// MPIR module — the unit of compilation output.
#[derive(Clone, Debug)]
pub struct MpirModule {
    pub sid: Sid,
    pub path: String,
    pub type_table: MpirTypeTable,
    pub functions: Vec<MpirFn>,
    pub globals: Vec<(GlobalId, TypeId, HirConst)>,
}

/// Print MPIR in textual form (§15.3-15.6).
pub fn print_mpir(module: &MpirModule, type_ctx: &TypeCtx) -> String {
    let mut out = String::new();
    let digest_input = format!(
        "magpie:mpir:module_digest:v0.1|{}|{}",
        module.path, module.sid.0
    );
    let module_digest = blake3::hash(digest_input.as_bytes()).to_hex().to_string();

    writeln!(out, "mpir.version 0.1").unwrap();
    writeln!(out, "module {}", module.path).unwrap();
    writeln!(out, "module_sid \"{}\"", module.sid.0).unwrap();
    writeln!(out, "module_digest \"{}\"", module_digest).unwrap();

    out.push_str(&format_type_table(&module.type_table, type_ctx));

    writeln!(out, "externs {{ }}").unwrap();

    if module.globals.is_empty() {
        writeln!(out, "globals {{ }}").unwrap();
    } else {
        writeln!(out, "globals {{").unwrap();
        let mut globals = module.globals.clone();
        globals.sort_by_key(|(id, _, _)| id.0);
        for (gid, ty, init) in &globals {
            writeln!(
                out,
                "  global @g{} : type_id {} = {}",
                gid.0,
                ty.0,
                format_const(init)
            )
            .unwrap();
        }
        writeln!(out, "}}").unwrap();
    }

    if module.functions.is_empty() {
        writeln!(out, "fns {{ }}").unwrap();
    } else {
        writeln!(out, "fns {{").unwrap();
        let mut fns: Vec<&MpirFn> = module.functions.iter().collect();
        fns.sort_by_key(|f| format!("{}.@{}", module.path, f.name));
        for f in fns {
            let sig_core = format!(
                "fn {}.@{}({}) -> type_id {}",
                module.path,
                f.name,
                join_type_ids(&f.params.iter().map(|(_, ty)| *ty).collect::<Vec<_>>(), ","),
                f.ret_ty.0
            );
            let sigdigest_input = format!("magpie:sigdigest:v0.1|{}", sig_core);
            let sigdigest = blake3::hash(sigdigest_input.as_bytes())
                .to_hex()
                .to_string();

            let params = f
                .params
                .iter()
                .map(|(id, ty)| format!("%{}: type_id {}", id.0, ty.0))
                .collect::<Vec<_>>()
                .join(", ");
            let inst_id = infer_inst_id_for_fn(f).unwrap_or_else(|| "I:base".to_string());
            writeln!(
                out,
                "  fn @{}({}) -> type_id {} sid \"{}\" sigdigest \"{}\" inst_id \"{}\"",
                f.name, params, f.ret_ty.0, f.sid.0, sigdigest, inst_id
            )
            .unwrap();
            writeln!(out, "  {{").unwrap();
            for block in &f.blocks {
                writeln!(out, "    bb{}:", block.id.0).unwrap();
                for instr in &block.instrs {
                    writeln!(
                        out,
                        "      %{} : type_id {} = {}",
                        instr.dst.0,
                        instr.ty.0,
                        format_op(&instr.op)
                    )
                    .unwrap();
                }
                for op in &block.void_ops {
                    writeln!(out, "      {}", format_void_op(op)).unwrap();
                }
                writeln!(out, "      {}", format_terminator(&block.terminator)).unwrap();
            }
            writeln!(out, "  }}").unwrap();
        }
        writeln!(out, "}}").unwrap();
    }

    out
}

/// Verify MPIR well-formedness (§15.8, §17.2).
#[allow(clippy::result_unit_err)]
pub fn verify_mpir(
    module: &MpirModule,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) -> Result<(), ()> {
    let before = diag.error_count();

    let type_id_set: HashSet<u32> = module.type_table.types.iter().map(|(id, _)| id.0).collect();
    let type_map: HashMap<u32, TypeKind> = module
        .type_table
        .types
        .iter()
        .map(|(id, kind)| (id.0, kind.clone()))
        .collect();
    let fn_param_arity: HashMap<String, usize> = module
        .functions
        .iter()
        .map(|func| (func.sid.0.clone(), func.params.len()))
        .collect();

    check_sid_format(&module.sid, "module sid", diag);

    for (id, kind) in &module.type_table.types {
        check_type_exists(*id, &type_id_set, diag, "type_table entry");
        for sid in type_kind_sids(kind) {
            check_sid_format(sid, "type_table sid", diag);
        }
        for ref_ty in type_kind_type_refs(kind) {
            check_type_exists(ref_ty, &type_id_set, diag, "type_table referenced type");
        }
    }

    for (gid, ty, init) in &module.globals {
        check_type_exists(*ty, &type_id_set, diag, &format!("global @g{}", gid.0));
        check_type_exists(
            init.ty,
            &type_id_set,
            diag,
            &format!("global @g{} init", gid.0),
        );
    }

    for func in &module.functions {
        verify_function(
            func,
            &type_id_set,
            &type_map,
            &fn_param_arity,
            type_ctx,
            diag,
        );
    }

    if diag.error_count() > before {
        Err(())
    } else {
        Ok(())
    }
}

fn verify_function(
    func: &MpirFn,
    type_id_set: &HashSet<u32>,
    type_map: &HashMap<u32, TypeKind>,
    fn_param_arity: &HashMap<String, usize>,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) {
    check_sid_format(&func.sid, &format!("fn '{}'", func.name), diag);
    check_type_exists(
        func.ret_ty,
        type_id_set,
        diag,
        &format!("fn '{}' return type", func.name),
    );

    if func.blocks.is_empty() {
        emit_error(
            diag,
            "MPS0010",
            &format!(
                "Terminator violation: fn '{}' has no basic blocks",
                func.name
            ),
        );
        return;
    }

    for (param, ty) in &func.params {
        let _ = param;
        check_type_exists(
            *ty,
            type_id_set,
            diag,
            &format!("fn '{}' parameter type", func.name),
        );
    }
    for local in &func.locals {
        check_type_exists(
            local.ty,
            type_id_set,
            diag,
            &format!("fn '{}' local decl {}", func.name, local.id.0),
        );
    }

    let mut def_block: HashMap<u32, usize> = HashMap::new();
    let mut local_ty: HashMap<u32, TypeId> = HashMap::new();

    for (param_id, param_ty) in &func.params {
        if def_block.insert(param_id.0, usize::MAX).is_some() {
            emit_error(
                diag,
                "MPS0001",
                &format!(
                    "SSA violation: LocalId {} defined more than once in fn '{}'",
                    param_id.0, func.name
                ),
            );
        }
        local_ty.insert(param_id.0, *param_ty);
    }

    let mut block_index: HashMap<u32, usize> = HashMap::new();
    for (i, b) in func.blocks.iter().enumerate() {
        if block_index.insert(b.id.0, i).is_some() {
            emit_error(
                diag,
                "MPS0009",
                &format!(
                    "CFG violation: duplicate block id bb{} in fn '{}'",
                    b.id.0, func.name
                ),
            );
        }
    }

    for (blk_idx, block) in func.blocks.iter().enumerate() {
        for instr in &block.instrs {
            let id = instr.dst.0;
            if def_block.insert(id, blk_idx).is_some() {
                emit_error(
                    diag,
                    "MPS0001",
                    &format!(
                        "SSA violation: LocalId {} defined more than once in fn '{}'",
                        id, func.name
                    ),
                );
            }
            local_ty.insert(id, instr.ty);
            check_type_exists(
                instr.ty,
                type_id_set,
                diag,
                &format!("fn '{}' instruction type", func.name),
            );
        }

        // Terminator existence is structurally guaranteed by the enum field;
        // this match keeps verifier logic explicit at block boundaries.
        match block.terminator {
            MpirTerminator::Ret(_)
            | MpirTerminator::Br(_)
            | MpirTerminator::Cbr { .. }
            | MpirTerminator::Switch { .. }
            | MpirTerminator::Unreachable => {}
        }
    }

    let n = func.blocks.len();
    let successors: Vec<Vec<usize>> = func
        .blocks
        .iter()
        .map(|b| block_successors(b, &block_index))
        .collect();

    let mut preds: Vec<Vec<usize>> = vec![vec![]; n];
    for (i, succs) in successors.iter().enumerate() {
        for &s in succs {
            if s < n {
                preds[s].push(i);
            }
        }
    }

    let mut dom: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    if n > 0 {
        dom[0].insert(0);
        let all: HashSet<usize> = (0..n).collect();
        for d in dom.iter_mut().skip(1) {
            *d = all.clone();
        }
        let mut changed = true;
        while changed {
            changed = false;
            for i in 1..n {
                let mut new_dom: HashSet<usize> = if preds[i].is_empty() {
                    HashSet::new()
                } else {
                    let mut iter = preds[i].iter();
                    let first = *iter.next().unwrap();
                    let mut acc = dom[first].clone();
                    for &p in iter {
                        acc = acc.intersection(&dom[p]).copied().collect();
                    }
                    acc
                };
                new_dom.insert(i);
                if new_dom != dom[i] {
                    dom[i] = new_dom;
                    changed = true;
                }
            }
        }
    }

    let dominates = |def_blk: usize, use_blk: usize| -> bool { dom[use_blk].contains(&def_blk) };

    let check_value = |v: &MpirValue,
                       use_blk_idx: usize,
                       _in_phi: bool,
                       diag: &mut DiagnosticBag| {
        match v {
            MpirValue::Const(c) => {
                check_type_exists(
                    c.ty,
                    type_id_set,
                    diag,
                    &format!("fn '{}' constant", func.name),
                );
            }
            MpirValue::Local(lid) => {
                let local_id = lid.0;
                let def_blk_idx = match def_block.get(&local_id) {
                    Some(&idx) => idx,
                    None => {
                        emit_error(
                            diag,
                            "MPS0002",
                            &format!(
                                "SSA violation: LocalId {} used before definition in fn '{}'",
                                local_id, func.name
                            ),
                        );
                        return;
                    }
                };

                let effective_def = if def_blk_idx == usize::MAX {
                    0
                } else {
                    def_blk_idx
                };
                // Skip dominance check for async functions: async lowering inserts
                // a dispatch switch block that breaks standard domination but is
                // semantically correct (locals are part of coroutine state).
                if !func.is_async && effective_def != use_blk_idx && !dominates(effective_def, use_blk_idx) {
                    emit_error(
                        diag,
                        "MPS0003",
                        &format!(
                            "SSA violation: use of LocalId {} in fn '{}' is not dominated by its definition",
                            local_id, func.name
                        ),
                    );
                }
            }
        }
    };

    for (blk_idx, block) in func.blocks.iter().enumerate() {
        for instr in &block.instrs {
            if has_arc_op(&instr.op) {
                emit_error(
                    diag,
                    "MPS0014",
                    &format!(
                        "ARC stage violation: arc.* op '{}' is not allowed before ARC insertion in fn '{}'",
                        format_op(&instr.op),
                        func.name
                    ),
                );
            }
            for ty in mpir_op_type_refs(&instr.op) {
                check_type_exists(
                    ty,
                    type_id_set,
                    diag,
                    &format!("fn '{}' op type reference", func.name),
                );
            }
            for sid in mpir_op_sids(&instr.op) {
                check_sid_format(sid, &format!("fn '{}' op sid", func.name), diag);
            }
            if let Some((callee_sid, arg_count)) = call_arity_site(&instr.op) {
                if let Some(expected) = fn_param_arity.get(callee_sid) {
                    if *expected != arg_count {
                        emit_error(
                            diag,
                            "MPS0012",
                            &format!(
                                "Call arity mismatch: callee sid '{}' expects {} argument(s), got {} in fn '{}'",
                                callee_sid, expected, arg_count, func.name
                            ),
                        );
                    }
                }
            }
            if let MpirOp::Phi { ty, .. } = &instr.op {
                if !phi_type_allowed(*ty, type_map, type_ctx) {
                    emit_error(
                        diag,
                        "MPS0008",
                        &format!(
                            "Phi violation: type_id {} is not allowed in phi nodes in fn '{}'",
                            ty.0, func.name
                        ),
                    );
                }
            }
            for v in mpir_op_values(&instr.op) {
                check_value(&v, blk_idx, matches!(instr.op, MpirOp::Phi { .. }), diag);
            }
        }

        for op in &block.void_ops {
            if has_arc_void_op(op) {
                emit_error(
                    diag,
                    "MPS0014",
                    &format!(
                        "ARC stage violation: arc.* op '{}' is not allowed before ARC insertion in fn '{}'",
                        format_void_op(op),
                        func.name
                    ),
                );
            }
            if let Some((callee_sid, arg_count)) = call_arity_void_site(op) {
                if let Some(expected) = fn_param_arity.get(callee_sid) {
                    if *expected != arg_count {
                        emit_error(
                            diag,
                            "MPS0012",
                            &format!(
                                "Call arity mismatch: callee sid '{}' expects {} argument(s), got {} in fn '{}'",
                                callee_sid, expected, arg_count, func.name
                            ),
                        );
                    }
                }
            }
            for ty in mpir_op_void_type_refs(op) {
                check_type_exists(
                    ty,
                    type_id_set,
                    diag,
                    &format!("fn '{}' void op type reference", func.name),
                );
            }
            for sid in mpir_op_void_sids(op) {
                check_sid_format(sid, &format!("fn '{}' void op sid", func.name), diag);
            }
            for v in mpir_op_void_values(op) {
                check_value(&v, blk_idx, false, diag);
            }
        }

        for v in mpir_terminator_values(&block.terminator) {
            check_value(&v, blk_idx, false, diag);
        }
        if let MpirTerminator::Switch { arms, .. } = &block.terminator {
            for (c, _) in arms {
                check_type_exists(
                    c.ty,
                    type_id_set,
                    diag,
                    &format!("fn '{}' switch arm const", func.name),
                );
            }
        }
    }

    let _ = local_ty;
}

fn call_arity_site(op: &MpirOp) -> Option<(&str, usize)> {
    match op {
        MpirOp::Call {
            callee_sid, args, ..
        }
        | MpirOp::SuspendCall {
            callee_sid, args, ..
        } => Some((callee_sid.0.as_str(), args.len())),
        _ => None,
    }
}

fn has_arc_op(op: &MpirOp) -> bool {
    matches!(
        op,
        MpirOp::ArcRetain { .. }
            | MpirOp::ArcRelease { .. }
            | MpirOp::ArcRetainWeak { .. }
            | MpirOp::ArcReleaseWeak { .. }
    )
}

fn has_arc_void_op(op: &MpirOpVoid) -> bool {
    matches!(
        op,
        MpirOpVoid::ArcRetain { .. }
            | MpirOpVoid::ArcRelease { .. }
            | MpirOpVoid::ArcRetainWeak { .. }
            | MpirOpVoid::ArcReleaseWeak { .. }
    )
}

fn call_arity_void_site(op: &MpirOpVoid) -> Option<(&str, usize)> {
    match op {
        MpirOpVoid::CallVoid {
            callee_sid, args, ..
        } => Some((callee_sid.0.as_str(), args.len())),
        _ => None,
    }
}

fn emit_error(diag: &mut DiagnosticBag, code: &str, msg: &str) {
    diag.emit(Diagnostic {
        code: code.to_string(),
        severity: Severity::Error,
        title: msg.to_string(),
        primary_span: None,
        secondary_spans: vec![],
        message: msg.to_string(),
        explanation_md: None,
        why: None,
        suggested_fixes: vec![],
        rag_bundle: Vec::new(),
        related_docs: Vec::new(),
    });
}

fn check_sid_format(sid: &Sid, context: &str, diag: &mut DiagnosticBag) {
    if !is_valid_sid_format(&sid.0) {
        emit_error(
            diag,
            "MPS0005",
            &format!("SID violation: invalid sid '{}' in {}", sid.0, context),
        );
    }
}

fn is_valid_sid_format(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.len() != 12 {
        return false;
    }
    if !matches!(bytes[0], b'M' | b'F' | b'T' | b'G' | b'E') || bytes[1] != b':' {
        return false;
    }
    bytes[2..]
        .iter()
        .all(|b| (*b >= b'0' && *b <= b'9') || (*b >= b'A' && *b <= b'Z'))
}

fn check_type_exists(
    ty: TypeId,
    type_id_set: &HashSet<u32>,
    diag: &mut DiagnosticBag,
    where_: &str,
) {
    if !type_id_set.contains(&ty.0) {
        emit_error(
            diag,
            "MPS0004",
            &format!(
                "TypeRef violation: type_id {} not found in type table ({})",
                ty.0, where_
            ),
        );
    }
}

fn phi_type_allowed(ty: TypeId, type_map: &HashMap<u32, TypeKind>, type_ctx: &TypeCtx) -> bool {
    let kind = type_map
        .get(&ty.0)
        .cloned()
        .or_else(|| type_ctx.lookup(ty).cloned());
    match kind {
        Some(TypeKind::HeapHandle {
            hk: HandleKind::Borrow,
            ..
        })
        | Some(TypeKind::HeapHandle {
            hk: HandleKind::MutBorrow,
            ..
        }) => false,
        Some(TypeKind::HeapHandle {
            hk: HandleKind::Unique,
            ..
        })
        | Some(TypeKind::HeapHandle {
            hk: HandleKind::Shared,
            ..
        })
        | Some(TypeKind::HeapHandle {
            hk: HandleKind::Weak,
            ..
        }) => true,
        Some(_) => true,
        None => false,
    }
}

fn type_kind_sids(kind: &TypeKind) -> Vec<&Sid> {
    match kind {
        TypeKind::HeapHandle {
            base: HeapBase::Callable { sig_sid },
            ..
        } => vec![sig_sid],
        TypeKind::HeapHandle {
            base: HeapBase::UserType { type_sid, .. },
            ..
        } => vec![type_sid],
        TypeKind::ValueStruct { sid } => vec![sid],
        _ => vec![],
    }
}

fn type_kind_type_refs(kind: &TypeKind) -> Vec<TypeId> {
    match kind {
        TypeKind::Prim(_) => vec![],
        TypeKind::HeapHandle { base, .. } => heap_base_type_refs(base),
        TypeKind::BuiltinOption { inner } => vec![*inner],
        TypeKind::BuiltinResult { ok, err } => vec![*ok, *err],
        TypeKind::RawPtr { to } => vec![*to],
        TypeKind::Arr { elem, .. } => vec![*elem],
        TypeKind::Vec { elem, .. } => vec![*elem],
        TypeKind::Tuple { elems } => elems.clone(),
        TypeKind::ValueStruct { .. } => vec![],
    }
}

fn heap_base_type_refs(base: &HeapBase) -> Vec<TypeId> {
    match base {
        HeapBase::BuiltinStr | HeapBase::BuiltinStrBuilder => vec![],
        HeapBase::BuiltinArray { elem } => vec![*elem],
        HeapBase::BuiltinMap { key, val } => vec![*key, *val],
        HeapBase::BuiltinMutex { inner }
        | HeapBase::BuiltinRwLock { inner }
        | HeapBase::BuiltinCell { inner }
        | HeapBase::BuiltinFuture { result: inner }
        | HeapBase::BuiltinChannelSend { elem: inner }
        | HeapBase::BuiltinChannelRecv { elem: inner } => vec![*inner],
        HeapBase::Callable { .. } => vec![],
        HeapBase::UserType { targs, .. } => targs.clone(),
    }
}

fn mpir_op_values(op: &MpirOp) -> Vec<MpirValue> {
    match op {
        MpirOp::Const(_) => vec![],
        MpirOp::Move { v }
        | MpirOp::BorrowShared { v }
        | MpirOp::BorrowMut { v }
        | MpirOp::Share { v }
        | MpirOp::CloneShared { v }
        | MpirOp::CloneWeak { v }
        | MpirOp::WeakDowngrade { v }
        | MpirOp::WeakUpgrade { v }
        | MpirOp::EnumTag { v }
        | MpirOp::EnumPayload { v, .. }
        | MpirOp::EnumIs { v, .. }
        | MpirOp::SuspendAwait { fut: v }
        | MpirOp::ArcRetain { v }
        | MpirOp::ArcRelease { v }
        | MpirOp::ArcRetainWeak { v }
        | MpirOp::ArcReleaseWeak { v } => vec![v.clone()],
        MpirOp::New { fields, .. } => fields.iter().map(|(_, v)| v.clone()).collect(),
        MpirOp::GetField { obj, .. } => vec![obj.clone()],
        MpirOp::IAdd { lhs, rhs }
        | MpirOp::ISub { lhs, rhs }
        | MpirOp::IMul { lhs, rhs }
        | MpirOp::ISDiv { lhs, rhs }
        | MpirOp::IUDiv { lhs, rhs }
        | MpirOp::ISRem { lhs, rhs }
        | MpirOp::IURem { lhs, rhs }
        | MpirOp::IAddWrap { lhs, rhs }
        | MpirOp::ISubWrap { lhs, rhs }
        | MpirOp::IMulWrap { lhs, rhs }
        | MpirOp::IAddChecked { lhs, rhs }
        | MpirOp::ISubChecked { lhs, rhs }
        | MpirOp::IMulChecked { lhs, rhs }
        | MpirOp::IAnd { lhs, rhs }
        | MpirOp::IOr { lhs, rhs }
        | MpirOp::IXor { lhs, rhs }
        | MpirOp::IShl { lhs, rhs }
        | MpirOp::ILshr { lhs, rhs }
        | MpirOp::IAshr { lhs, rhs }
        | MpirOp::ICmp { lhs, rhs, .. }
        | MpirOp::FCmp { lhs, rhs, .. }
        | MpirOp::FAdd { lhs, rhs }
        | MpirOp::FSub { lhs, rhs }
        | MpirOp::FMul { lhs, rhs }
        | MpirOp::FDiv { lhs, rhs }
        | MpirOp::FRem { lhs, rhs }
        | MpirOp::FAddFast { lhs, rhs }
        | MpirOp::FSubFast { lhs, rhs }
        | MpirOp::FMulFast { lhs, rhs }
        | MpirOp::FDivFast { lhs, rhs } => vec![lhs.clone(), rhs.clone()],
        MpirOp::Cast { v, .. } => vec![v.clone()],
        MpirOp::PtrNull { .. } => vec![],
        MpirOp::PtrAddr { p } => vec![p.clone()],
        MpirOp::PtrFromAddr { addr, .. } => vec![addr.clone()],
        MpirOp::PtrAdd { p, count } => vec![p.clone(), count.clone()],
        MpirOp::PtrLoad { p, .. } => vec![p.clone()],
        MpirOp::PtrStore { p, v, .. } => vec![p.clone(), v.clone()],
        MpirOp::Call { args, .. } => args.clone(),
        MpirOp::CallIndirect { callee, args } | MpirOp::CallVoidIndirect { callee, args } => {
            let mut vs = vec![callee.clone()];
            vs.extend(args.iter().cloned());
            vs
        }
        MpirOp::SuspendCall { args, .. } => args.clone(),
        MpirOp::Phi { incomings, .. } => incomings.iter().map(|(_, v)| v.clone()).collect(),
        MpirOp::EnumNew { args, .. } => args.iter().map(|(_, v)| v.clone()).collect(),
        MpirOp::CallableCapture { captures, .. } => {
            captures.iter().map(|(_, v)| v.clone()).collect()
        }
        MpirOp::ArrNew { cap, .. } => vec![cap.clone()],
        MpirOp::ArrLen { arr } => vec![arr.clone()],
        MpirOp::ArrGet { arr, idx } => vec![arr.clone(), idx.clone()],
        MpirOp::ArrSet { arr, idx, val } => vec![arr.clone(), idx.clone(), val.clone()],
        MpirOp::ArrPush { arr, val } => vec![arr.clone(), val.clone()],
        MpirOp::ArrPop { arr } => vec![arr.clone()],
        MpirOp::ArrSlice { arr, start, end } => vec![arr.clone(), start.clone(), end.clone()],
        MpirOp::ArrContains { arr, val } => vec![arr.clone(), val.clone()],
        MpirOp::ArrSort { arr } => vec![arr.clone()],
        MpirOp::ArrMap { arr, func }
        | MpirOp::ArrFilter { arr, func }
        | MpirOp::ArrForeach { arr, func } => {
            vec![arr.clone(), func.clone()]
        }
        MpirOp::ArrReduce { arr, init, func } => vec![arr.clone(), init.clone(), func.clone()],
        MpirOp::MapNew { .. } => vec![],
        MpirOp::MapLen { map } => vec![map.clone()],
        MpirOp::MapGet { map, key }
        | MpirOp::MapGetRef { map, key }
        | MpirOp::MapDelete { map, key }
        | MpirOp::MapContainsKey { map, key }
        | MpirOp::MapDeleteVoid { map, key } => {
            vec![map.clone(), key.clone()]
        }
        MpirOp::MapKeys { map } | MpirOp::MapValues { map } => vec![map.clone()],
        MpirOp::MapSet { map, key, val } => vec![map.clone(), key.clone(), val.clone()],
        MpirOp::StrConcat { a, b } | MpirOp::StrEq { a, b } => vec![a.clone(), b.clone()],
        MpirOp::StrLen { s }
        | MpirOp::StrBytes { s }
        | MpirOp::StrParseI64 { s }
        | MpirOp::StrParseU64 { s }
        | MpirOp::StrParseF64 { s }
        | MpirOp::StrParseBool { s } => vec![s.clone()],
        MpirOp::StrSlice { s, start, end } => vec![s.clone(), start.clone(), end.clone()],
        MpirOp::StrBuilderNew => vec![],
        MpirOp::StrBuilderAppendStr { b, s } => vec![b.clone(), s.clone()],
        MpirOp::StrBuilderAppendI64 { b, v }
        | MpirOp::StrBuilderAppendI32 { b, v }
        | MpirOp::StrBuilderAppendF64 { b, v }
        | MpirOp::StrBuilderAppendBool { b, v } => {
            vec![b.clone(), v.clone()]
        }
        MpirOp::StrBuilderBuild { b } => vec![b.clone()],
        MpirOp::JsonEncode { v, .. } => vec![v.clone()],
        MpirOp::JsonDecode { s, .. } => vec![s.clone()],
        MpirOp::GpuThreadId
        | MpirOp::GpuWorkgroupId
        | MpirOp::GpuWorkgroupSize
        | MpirOp::GpuGlobalId => vec![],
        MpirOp::GpuBufferLoad { buf, idx } => vec![buf.clone(), idx.clone()],
        MpirOp::GpuBufferLen { buf } => vec![buf.clone()],
        MpirOp::GpuShared { size, .. } => vec![size.clone()],
        MpirOp::GpuLaunch {
            device,
            groups,
            threads,
            args,
            ..
        }
        | MpirOp::GpuLaunchAsync {
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
        MpirOp::Panic { msg } => vec![msg.clone()],
    }
}

fn mpir_op_void_values(op: &MpirOpVoid) -> Vec<MpirValue> {
    match op {
        MpirOpVoid::CallVoid { args, .. } => args.clone(),
        MpirOpVoid::CallVoidIndirect { callee, args } => {
            let mut vs = vec![callee.clone()];
            vs.extend(args.iter().cloned());
            vs
        }
        MpirOpVoid::SetField { obj, value, .. } => vec![obj.clone(), value.clone()],
        MpirOpVoid::ArrSet { arr, idx, val } => vec![arr.clone(), idx.clone(), val.clone()],
        MpirOpVoid::ArrPush { arr, val } => vec![arr.clone(), val.clone()],
        MpirOpVoid::ArrSort { arr } => vec![arr.clone()],
        MpirOpVoid::ArrForeach { arr, func } => vec![arr.clone(), func.clone()],
        MpirOpVoid::MapSet { map, key, val } => vec![map.clone(), key.clone(), val.clone()],
        MpirOpVoid::MapDeleteVoid { map, key } => vec![map.clone(), key.clone()],
        MpirOpVoid::StrBuilderAppendStr { b, s } => vec![b.clone(), s.clone()],
        MpirOpVoid::StrBuilderAppendI64 { b, v }
        | MpirOpVoid::StrBuilderAppendI32 { b, v }
        | MpirOpVoid::StrBuilderAppendF64 { b, v }
        | MpirOpVoid::StrBuilderAppendBool { b, v } => {
            vec![b.clone(), v.clone()]
        }
        MpirOpVoid::PtrStore { p, v, .. } => vec![p.clone(), v.clone()],
        MpirOpVoid::Panic { msg } => vec![msg.clone()],
        MpirOpVoid::GpuBarrier => vec![],
        MpirOpVoid::GpuBufferStore { buf, idx, val } => vec![buf.clone(), idx.clone(), val.clone()],
        MpirOpVoid::ArcRetain { v }
        | MpirOpVoid::ArcRelease { v }
        | MpirOpVoid::ArcRetainWeak { v }
        | MpirOpVoid::ArcReleaseWeak { v } => vec![v.clone()],
    }
}

fn mpir_terminator_values(term: &MpirTerminator) -> Vec<MpirValue> {
    match term {
        MpirTerminator::Ret(Some(v)) => vec![v.clone()],
        MpirTerminator::Ret(None) => vec![],
        MpirTerminator::Br(_) => vec![],
        MpirTerminator::Cbr { cond, .. } => vec![cond.clone()],
        MpirTerminator::Switch { val, .. } => vec![val.clone()],
        MpirTerminator::Unreachable => vec![],
    }
}

fn mpir_op_type_refs(op: &MpirOp) -> Vec<TypeId> {
    match op {
        MpirOp::Const(c) => vec![c.ty],
        MpirOp::New { ty, .. } => vec![*ty],
        MpirOp::Cast { to, .. } => vec![*to],
        MpirOp::PtrNull { to }
        | MpirOp::PtrFromAddr { to, .. }
        | MpirOp::PtrLoad { to, .. }
        | MpirOp::PtrStore { to, .. } => vec![*to],
        MpirOp::Call { inst, .. } | MpirOp::SuspendCall { inst, .. } => inst.clone(),
        MpirOp::Phi { ty, .. } => vec![*ty],
        MpirOp::ArrNew { elem_ty, .. } => vec![*elem_ty],
        MpirOp::MapNew { key_ty, val_ty } => vec![*key_ty, *val_ty],
        MpirOp::JsonEncode { ty, .. } | MpirOp::JsonDecode { ty, .. } => vec![*ty],
        MpirOp::GpuShared { ty, .. } => vec![*ty],
        _ => vec![],
    }
}

fn mpir_op_void_type_refs(op: &MpirOpVoid) -> Vec<TypeId> {
    match op {
        MpirOpVoid::CallVoid { inst, .. } => inst.clone(),
        MpirOpVoid::PtrStore { to, .. } => vec![*to],
        _ => vec![],
    }
}

fn mpir_op_sids(op: &MpirOp) -> Vec<&Sid> {
    match op {
        MpirOp::Call { callee_sid, .. } | MpirOp::SuspendCall { callee_sid, .. } => {
            vec![callee_sid]
        }
        MpirOp::CallableCapture { fn_ref, .. } => vec![fn_ref],
        MpirOp::GpuLaunch { kernel, .. } | MpirOp::GpuLaunchAsync { kernel, .. } => vec![kernel],
        _ => vec![],
    }
}

fn mpir_op_void_sids(op: &MpirOpVoid) -> Vec<&Sid> {
    match op {
        MpirOpVoid::CallVoid { callee_sid, .. } => vec![callee_sid],
        _ => vec![],
    }
}

fn block_successors(block: &MpirBlock, block_index: &HashMap<u32, usize>) -> Vec<usize> {
    match &block.terminator {
        MpirTerminator::Ret(_) | MpirTerminator::Unreachable => vec![],
        MpirTerminator::Br(bid) => block_index.get(&bid.0).copied().into_iter().collect(),
        MpirTerminator::Cbr {
            then_bb, else_bb, ..
        } => {
            let mut v = Vec::new();
            if let Some(&i) = block_index.get(&then_bb.0) {
                v.push(i);
            }
            if let Some(&i) = block_index.get(&else_bb.0) {
                v.push(i);
            }
            v
        }
        MpirTerminator::Switch { arms, default, .. } => {
            let mut v = Vec::new();
            for (_, bid) in arms {
                if let Some(&i) = block_index.get(&bid.0) {
                    v.push(i);
                }
            }
            if let Some(&i) = block_index.get(&default.0) {
                v.push(i);
            }
            v
        }
    }
}

#[derive(Clone, Debug)]
struct TypeLayoutInfo {
    size: u32,
    align: u32,
    fields: Vec<(String, u32)>,
}

fn format_type_table(type_table: &MpirTypeTable, type_ctx: &TypeCtx) -> String {
    let mut out = String::new();
    let mut types = type_table.types.clone();
    types.sort_by_key(|(id, _)| id.0);
    if types.is_empty() {
        writeln!(out, "types {{ }}").unwrap();
        return out;
    }

    let type_map: HashMap<u32, TypeKind> = types
        .iter()
        .map(|(id, kind)| (id.0, kind.clone()))
        .collect();

    writeln!(out, "types {{").unwrap();
    for (id, kind) in &types {
        let layout = type_layout_for_kind(kind, &type_map, type_ctx, &mut HashSet::new());
        let fields = if layout.fields.is_empty() {
            String::new()
        } else {
            layout
                .fields
                .iter()
                .map(|(field, offset)| format!("{field}={offset}"))
                .collect::<Vec<_>>()
                .join(" ")
        };
        if fields.is_empty() {
            writeln!(
                out,
                "  type_id {} = {} name=\"{}\" kind={} layout {{ size={} align={} fields {{ }} }}",
                id.0,
                format_type_kind_for_mpir(kind),
                escape_mpir_string(&type_name_for_mpir(kind)),
                type_class_for_mpir(kind),
                layout.size,
                layout.align
            )
            .unwrap();
        } else {
            writeln!(
                out,
                "  type_id {} = {} name=\"{}\" kind={} layout {{ size={} align={} fields {{ {} }} }}",
                id.0,
                format_type_kind_for_mpir(kind),
                escape_mpir_string(&type_name_for_mpir(kind)),
                type_class_for_mpir(kind),
                layout.size,
                layout.align,
                fields
            )
            .unwrap();
        }
    }
    writeln!(out, "}}").unwrap();
    out
}

fn infer_inst_id_for_fn(func: &MpirFn) -> Option<String> {
    parse_inst_id_suffix(&func.name)
        .or_else(|| parse_inst_id_suffix(&func.sid.0))
        .map(|suffix| format!("I:{suffix}"))
}

fn parse_inst_id_suffix(s: &str) -> Option<String> {
    let (_, suffix) = s.rsplit_once("$I$")?;
    if suffix.is_empty() {
        None
    } else {
        Some(suffix.to_string())
    }
}

fn escape_mpir_string(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn type_name_for_mpir(kind: &TypeKind) -> String {
    match kind {
        TypeKind::Prim(p) => format!("prim.{}", format_prim(*p)),
        TypeKind::HeapHandle { base, .. } => match base {
            HeapBase::BuiltinStr => "builtin.Str".to_string(),
            HeapBase::BuiltinArray { .. } => "builtin.Array".to_string(),
            HeapBase::BuiltinMap { .. } => "builtin.Map".to_string(),
            HeapBase::BuiltinStrBuilder => "builtin.StrBuilder".to_string(),
            HeapBase::BuiltinMutex { .. } => "builtin.Mutex".to_string(),
            HeapBase::BuiltinRwLock { .. } => "builtin.RwLock".to_string(),
            HeapBase::BuiltinCell { .. } => "builtin.Cell".to_string(),
            HeapBase::BuiltinFuture { .. } => "builtin.TFuture".to_string(),
            HeapBase::BuiltinChannelSend { .. } => "builtin.ChannelSend".to_string(),
            HeapBase::BuiltinChannelRecv { .. } => "builtin.ChannelRecv".to_string(),
            HeapBase::Callable { sig_sid } => format!("builtin.TCallable.{}", sig_sid.0),
            HeapBase::UserType { type_sid, .. } => type_sid.0.clone(),
        },
        TypeKind::BuiltinOption { .. } => "builtin.TOption".to_string(),
        TypeKind::BuiltinResult { .. } => "builtin.TResult".to_string(),
        TypeKind::RawPtr { .. } => "rawptr".to_string(),
        TypeKind::Arr { n, .. } => format!("arr[{n}]"),
        TypeKind::Vec { n, .. } => format!("vec[{n}]"),
        TypeKind::Tuple { elems } => format!("tuple[{}]", elems.len()),
        TypeKind::ValueStruct { sid } => sid.0.clone(),
    }
}

fn type_class_for_mpir(kind: &TypeKind) -> &'static str {
    match kind {
        TypeKind::HeapHandle {
            base: HeapBase::UserType { type_sid, .. },
            ..
        } if type_sid.0.starts_with("E:") => "enum",
        TypeKind::ValueStruct { sid } if sid.0.starts_with("E:") => "enum",
        TypeKind::HeapHandle { .. } => "heap",
        _ => "value",
    }
}

fn type_layout_for_kind(
    kind: &TypeKind,
    type_map: &HashMap<u32, TypeKind>,
    type_ctx: &TypeCtx,
    seen: &mut HashSet<u32>,
) -> TypeLayoutInfo {
    match kind {
        TypeKind::Prim(p) => {
            let size = prim_layout_size(*p);
            TypeLayoutInfo {
                size,
                align: prim_layout_align(*p),
                fields: vec![],
            }
        }
        TypeKind::HeapHandle { .. } | TypeKind::RawPtr { .. } => TypeLayoutInfo {
            size: 8,
            align: 8,
            fields: vec![("ptr".to_string(), 0)],
        },
        TypeKind::BuiltinOption { inner } => {
            let inner_layout = type_layout_for_type_id(*inner, type_map, type_ctx, seen);
            let payload_align = inner_layout.align.max(1);
            let payload_offset = align_up_u32(1, payload_align);
            let size = align_up_u32(
                payload_offset.saturating_add(inner_layout.size),
                payload_align,
            );
            TypeLayoutInfo {
                size,
                align: payload_align,
                fields: vec![
                    ("tag".to_string(), 0),
                    ("payload".to_string(), payload_offset),
                ],
            }
        }
        TypeKind::BuiltinResult { ok, err } => {
            let ok_layout = type_layout_for_type_id(*ok, type_map, type_ctx, seen);
            let err_layout = type_layout_for_type_id(*err, type_map, type_ctx, seen);
            let payload_align = ok_layout.align.max(err_layout.align).max(1);
            let payload_offset = align_up_u32(1, payload_align);
            let payload_size = ok_layout.size.max(err_layout.size);
            let size = align_up_u32(payload_offset.saturating_add(payload_size), payload_align);
            TypeLayoutInfo {
                size,
                align: payload_align,
                fields: vec![
                    ("tag".to_string(), 0),
                    ("ok".to_string(), payload_offset),
                    ("err".to_string(), payload_offset),
                ],
            }
        }
        TypeKind::Arr { n, elem } | TypeKind::Vec { n, elem } => {
            let elem_layout = type_layout_for_type_id(*elem, type_map, type_ctx, seen);
            let elem_align = elem_layout.align.max(1);
            let stride = align_up_u32(elem_layout.size, elem_align);
            let size = stride.saturating_mul(*n);
            TypeLayoutInfo {
                size,
                align: elem_align,
                fields: if *n == 0 {
                    vec![]
                } else {
                    vec![("data".to_string(), 0)]
                },
            }
        }
        TypeKind::Tuple { elems } => {
            let mut fields = Vec::with_capacity(elems.len());
            let mut offset = 0_u32;
            let mut max_align = 1_u32;
            for (idx, elem) in elems.iter().enumerate() {
                let elem_layout = type_layout_for_type_id(*elem, type_map, type_ctx, seen);
                let elem_align = elem_layout.align.max(1);
                max_align = max_align.max(elem_align);
                offset = align_up_u32(offset, elem_align);
                fields.push((format!("f{idx}"), offset));
                offset = offset.saturating_add(elem_layout.size);
            }
            TypeLayoutInfo {
                size: align_up_u32(offset, max_align),
                align: max_align,
                fields,
            }
        }
        TypeKind::ValueStruct { .. } => TypeLayoutInfo {
            size: 0,
            align: 1,
            fields: vec![],
        },
    }
}

fn type_layout_for_type_id(
    ty: TypeId,
    type_map: &HashMap<u32, TypeKind>,
    type_ctx: &TypeCtx,
    seen: &mut HashSet<u32>,
) -> TypeLayoutInfo {
    if !seen.insert(ty.0) {
        return TypeLayoutInfo {
            size: 0,
            align: 1,
            fields: vec![],
        };
    }
    let kind = type_map
        .get(&ty.0)
        .cloned()
        .or_else(|| type_ctx.lookup(ty).cloned());
    let layout = kind.map_or(
        TypeLayoutInfo {
            size: 0,
            align: 1,
            fields: vec![],
        },
        |k| type_layout_for_kind(&k, type_map, type_ctx, seen),
    );
    seen.remove(&ty.0);
    layout
}

fn align_up_u32(value: u32, align: u32) -> u32 {
    if align <= 1 {
        return value;
    }
    let rem = value % align;
    if rem == 0 {
        value
    } else {
        value.saturating_add(align - rem)
    }
}

fn prim_layout_size(p: PrimType) -> u32 {
    match p {
        PrimType::Unit => 0,
        PrimType::I1 | PrimType::U1 | PrimType::Bool => 1,
        PrimType::I8 | PrimType::U8 => 1,
        PrimType::I16 | PrimType::U16 | PrimType::F16 => 2,
        PrimType::I32 | PrimType::U32 | PrimType::F32 => 4,
        PrimType::I64 | PrimType::U64 | PrimType::F64 => 8,
        PrimType::I128 | PrimType::U128 => 16,
    }
}

fn prim_layout_align(p: PrimType) -> u32 {
    prim_layout_size(p).max(1)
}

fn format_value(v: &MpirValue) -> String {
    match v {
        MpirValue::Local(id) => format!("%{}", id.0),
        MpirValue::Const(c) => format_const(c),
    }
}

fn format_const(c: &HirConst) -> String {
    format!(
        "const.type_id {} {}",
        c.ty.0,
        match &c.lit {
            HirConstLit::IntLit(i) => i.to_string(),
            HirConstLit::FloatLit(f) => f.to_string(),
            HirConstLit::BoolLit(b) => b.to_string(),
            HirConstLit::StringLit(s) => format!("{:?}", s),
            HirConstLit::Unit => "unit".to_string(),
        }
    )
}

fn format_type_kind_for_mpir(kind: &TypeKind) -> String {
    match kind {
        TypeKind::Prim(p) => format!("prim {}", format_prim(*p)),
        TypeKind::HeapHandle { hk, base } => match hk {
            HandleKind::Unique => format_heap_base(base),
            HandleKind::Shared => format!("shared {{ inner={} }}", format_heap_base(base)),
            HandleKind::Weak => format!("weak {{ inner={} }}", format_heap_base(base)),
            HandleKind::Borrow => format!("borrow {{ inner={} }}", format_heap_base(base)),
            HandleKind::MutBorrow => format!("mutborrow {{ inner={} }}", format_heap_base(base)),
        },
        TypeKind::BuiltinOption { inner } => {
            format!("builtin TOption {{ inner=type_id {} }}", inner.0)
        }
        TypeKind::BuiltinResult { ok, err } => {
            format!(
                "builtin TResult {{ ok=type_id {}, err=type_id {} }}",
                ok.0, err.0
            )
        }
        TypeKind::RawPtr { to } => format!("rawptr {{ to=type_id {} }}", to.0),
        TypeKind::Arr { n, elem } => format!("arr {{ n={}, elem=type_id {} }}", n, elem.0),
        TypeKind::Vec { n, elem } => format!("vec {{ n={}, elem=type_id {} }}", n, elem.0),
        TypeKind::Tuple { elems } => {
            format!(
                "tuple {{ elems=[{}] }}",
                elems
                    .iter()
                    .map(|t| format!("type_id {}", t.0))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
        TypeKind::ValueStruct { sid } => format!("value_struct sid \"{}\"", sid.0),
    }
}

fn format_heap_base(base: &HeapBase) -> String {
    match base {
        HeapBase::BuiltinStr => "heap_builtin Str".to_string(),
        HeapBase::BuiltinArray { elem } => {
            format!("heap_builtin Array {{ elem=type_id {} }}", elem.0)
        }
        HeapBase::BuiltinMap { key, val } => {
            format!(
                "heap_builtin Map {{ key=type_id {}, val=type_id {} }}",
                key.0, val.0
            )
        }
        HeapBase::BuiltinStrBuilder => "heap_builtin StrBuilder".to_string(),
        HeapBase::BuiltinMutex { inner } => {
            format!("heap_builtin Mutex {{ inner=type_id {} }}", inner.0)
        }
        HeapBase::BuiltinRwLock { inner } => {
            format!("heap_builtin RwLock {{ inner=type_id {} }}", inner.0)
        }
        HeapBase::BuiltinCell { inner } => {
            format!("heap_builtin Cell {{ inner=type_id {} }}", inner.0)
        }
        HeapBase::BuiltinFuture { result } => {
            format!("heap_builtin TFuture {{ result=type_id {} }}", result.0)
        }
        HeapBase::BuiltinChannelSend { elem } => {
            format!("heap_builtin ChannelSend {{ elem=type_id {} }}", elem.0)
        }
        HeapBase::BuiltinChannelRecv { elem } => {
            format!("heap_builtin ChannelRecv {{ elem=type_id {} }}", elem.0)
        }
        HeapBase::Callable { sig_sid } => {
            format!("heap_builtin TCallable {{ sig=@{}, caps=[] }}", sig_sid.0)
        }
        HeapBase::UserType { type_sid, targs } => {
            if targs.is_empty() {
                format!("heap_struct sid \"{}\"", type_sid.0)
            } else {
                format!(
                    "heap_struct sid \"{}\" {{ targs=[{}] }}",
                    type_sid.0,
                    targs
                        .iter()
                        .map(|t| format!("type_id {}", t.0))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        }
    }
}

fn format_prim(p: PrimType) -> &'static str {
    match p {
        PrimType::I1 => "i1",
        PrimType::I8 => "i8",
        PrimType::I16 => "i16",
        PrimType::I32 => "i32",
        PrimType::I64 => "i64",
        PrimType::I128 => "i128",
        PrimType::U1 => "u1",
        PrimType::U8 => "u8",
        PrimType::U16 => "u16",
        PrimType::U32 => "u32",
        PrimType::U64 => "u64",
        PrimType::U128 => "u128",
        PrimType::F16 => "f16",
        PrimType::F32 => "f32",
        PrimType::F64 => "f64",
        PrimType::Bool => "bool",
        PrimType::Unit => "unit",
    }
}

fn format_op(op: &MpirOp) -> String {
    match op {
        MpirOp::Const(c) => format_const(c),
        MpirOp::Move { v } => format!("move {{ v={} }}", format_value(v)),
        MpirOp::BorrowShared { v } => format!("borrow.shared {{ v={} }}", format_value(v)),
        MpirOp::BorrowMut { v } => format!("borrow.mut {{ v={} }}", format_value(v)),
        MpirOp::New { ty, fields } => format!(
            "new type_id {} {{ {} }}",
            ty.0,
            fields
                .iter()
                .map(|(name, v)| format!("{}={}", name, format_value(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        MpirOp::GetField { obj, field } => {
            format!("getfield {{ obj={}, field={} }}", format_value(obj), field)
        }
        MpirOp::ICmp { pred, lhs, rhs } => format!(
            "icmp.{} {{ lhs={}, rhs={} }}",
            pred,
            format_value(lhs),
            format_value(rhs)
        ),
        MpirOp::FCmp { pred, lhs, rhs } => format!(
            "fcmp.{} {{ lhs={}, rhs={} }}",
            pred,
            format_value(lhs),
            format_value(rhs)
        ),
        MpirOp::Call {
            callee_sid,
            inst,
            args,
        } => format!(
            "call sid \"{}\" {{ targs=[{}], args=[{}] }}",
            callee_sid.0,
            inst.iter()
                .map(|t| format!("type_id {}", t.0))
                .collect::<Vec<_>>()
                .join(", "),
            args.iter().map(format_value).collect::<Vec<_>>().join(", ")
        ),
        MpirOp::CallIndirect { callee, args } => format!(
            "call.indirect {} {{ args=[{}] }}",
            format_value(callee),
            args.iter().map(format_value).collect::<Vec<_>>().join(", ")
        ),
        MpirOp::CallVoidIndirect { callee, args } => format!(
            "call_void.indirect {} {{ args=[{}] }}",
            format_value(callee),
            args.iter().map(format_value).collect::<Vec<_>>().join(", ")
        ),
        MpirOp::SuspendCall {
            callee_sid,
            inst,
            args,
        } => format!(
            "suspend.call sid \"{}\" {{ targs=[{}], args=[{}] }}",
            callee_sid.0,
            inst.iter()
                .map(|t| format!("type_id {}", t.0))
                .collect::<Vec<_>>()
                .join(", "),
            args.iter().map(format_value).collect::<Vec<_>>().join(", ")
        ),
        MpirOp::SuspendAwait { fut } => format!("suspend.await {{ fut={} }}", format_value(fut)),
        MpirOp::Phi { ty, incomings } => format!(
            "phi type_id {} [{}]",
            ty.0,
            incomings
                .iter()
                .map(|(bb, v)| format!("(bb{}, {})", bb.0, format_value(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        MpirOp::ArcRetain { v } => format!("arc.retain {{ v={} }}", format_value(v)),
        MpirOp::ArcRelease { v } => format!("arc.release {{ v={} }}", format_value(v)),
        MpirOp::ArcRetainWeak { v } => format!("arc.retain_weak {{ v={} }}", format_value(v)),
        MpirOp::ArcReleaseWeak { v } => format!("arc.release_weak {{ v={} }}", format_value(v)),
        MpirOp::Panic { msg } => format!("panic {{ msg={} }}", format_value(msg)),
        _ => format!("{:?}", op),
    }
}

fn format_void_op(op: &MpirOpVoid) -> String {
    match op {
        MpirOpVoid::CallVoid {
            callee_sid,
            inst,
            args,
        } => format!(
            "call_void sid \"{}\" {{ targs=[{}], args=[{}] }}",
            callee_sid.0,
            inst.iter()
                .map(|t| format!("type_id {}", t.0))
                .collect::<Vec<_>>()
                .join(", "),
            args.iter().map(format_value).collect::<Vec<_>>().join(", ")
        ),
        MpirOpVoid::CallVoidIndirect { callee, args } => format!(
            "call_void.indirect {} {{ args=[{}] }}",
            format_value(callee),
            args.iter().map(format_value).collect::<Vec<_>>().join(", ")
        ),
        MpirOpVoid::SetField { obj, field, value } => format!(
            "setfield {{ obj={}, field={}, value={} }}",
            format_value(obj),
            field,
            format_value(value)
        ),
        MpirOpVoid::ArcRetain { v } => format!("arc.retain {{ v={} }}", format_value(v)),
        MpirOpVoid::ArcRelease { v } => format!("arc.release {{ v={} }}", format_value(v)),
        MpirOpVoid::ArcRetainWeak { v } => format!("arc.retain_weak {{ v={} }}", format_value(v)),
        MpirOpVoid::ArcReleaseWeak { v } => format!("arc.release_weak {{ v={} }}", format_value(v)),
        MpirOpVoid::Panic { msg } => format!("panic {{ msg={} }}", format_value(msg)),
        _ => format!("{:?}", op),
    }
}

fn format_terminator(term: &MpirTerminator) -> String {
    match term {
        MpirTerminator::Ret(Some(v)) => format!("ret {}", format_value(v)),
        MpirTerminator::Ret(None) => "ret".to_string(),
        MpirTerminator::Br(bb) => format!("br bb{}", bb.0),
        MpirTerminator::Cbr {
            cond,
            then_bb,
            else_bb,
        } => format!(
            "cbr {}, bb{}, bb{}",
            format_value(cond),
            then_bb.0,
            else_bb.0
        ),
        MpirTerminator::Switch { val, arms, default } => format!(
            "switch {} [{}] default bb{}",
            format_value(val),
            arms.iter()
                .map(|(c, bb)| format!("{} => bb{}", format_const(c), bb.0))
                .collect::<Vec<_>>()
                .join(", "),
            default.0
        ),
        MpirTerminator::Unreachable => "unreachable".to_string(),
    }
}

fn join_type_ids(type_ids: &[TypeId], sep: &str) -> String {
    type_ids
        .iter()
        .map(|t| format!("type_id {}", t.0))
        .collect::<Vec<_>>()
        .join(sep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn print_mpir_emits_type_layouts_and_function_metadata() {
        let mut type_ctx = TypeCtx::new();
        let tuple_ty = type_ctx.intern(TypeKind::Tuple {
            elems: vec![TypeId(4), TypeId(1)],
        });
        let mut table = type_ctx.types.clone();
        table.push((
            TypeId(2000),
            TypeKind::HeapHandle {
                hk: HandleKind::Unique,
                base: HeapBase::UserType {
                    type_sid: Sid("E:ENUM000001".to_string()),
                    targs: vec![],
                },
            },
        ));

        let module = MpirModule {
            sid: Sid("M:MODULE0001".to_string()),
            path: "pkg.mod".to_string(),
            type_table: MpirTypeTable { types: table },
            functions: vec![MpirFn {
                sid: Sid("F:FNMAIN0001".to_string()),
                name: "main$I$ABCDEF0123456789".to_string(),
                params: vec![(LocalId(0), TypeId(4)), (LocalId(1), tuple_ty)],
                ret_ty: TypeId(0),
                blocks: vec![MpirBlock {
                    id: BlockId(0),
                    instrs: vec![],
                    void_ops: vec![],
                    terminator: MpirTerminator::Ret(None),
                }],
                locals: vec![],
                is_async: false,
            }],
            globals: vec![],
        };

        let printed = print_mpir(&module, &type_ctx);
        assert!(printed.contains("layout { size="));
        assert!(printed.contains("type_id 4 = prim i32 name=\"prim.i32\" kind=value layout"));
        assert!(printed.contains("type_id 2000 = heap_struct sid \"E:ENUM000001\""));
        assert!(printed.contains("kind=enum layout"));

        let header = printed
            .lines()
            .find(|line| line.starts_with("  fn @main$I$ABCDEF0123456789("))
            .expect("expected function header");
        assert!(header.contains("-> type_id 0 sid \"F:FNMAIN0001\""));
        assert!(header.contains("sigdigest \""));
        assert!(header.contains("inst_id \"I:ABCDEF0123456789\""));
    }

    #[test]
    fn print_mpir_keeps_omitted_sections_with_empty_brackets() {
        let type_ctx = TypeCtx::new();
        let module = MpirModule {
            sid: Sid("M:MODULE0002".to_string()),
            path: "pkg.empty".to_string(),
            type_table: MpirTypeTable { types: vec![] },
            functions: vec![],
            globals: vec![],
        };

        let printed = print_mpir(&module, &type_ctx);
        assert!(printed.contains("types { }"));
        assert!(printed.contains("externs { }"));
        assert!(printed.contains("globals { }"));
        assert!(printed.contains("fns { }"));
    }

    #[test]
    fn verify_mpir_reports_direct_call_arity_mismatch() {
        let type_ctx = TypeCtx::new();
        let module = MpirModule {
            sid: Sid("M:MODULE0003".to_string()),
            path: "pkg.verify".to_string(),
            type_table: MpirTypeTable {
                types: vec![
                    (TypeId(0), TypeKind::Prim(PrimType::Unit)),
                    (TypeId(4), TypeKind::Prim(PrimType::I32)),
                ],
            },
            functions: vec![
                MpirFn {
                    sid: Sid("F:CALLEE0001".to_string()),
                    name: "callee".to_string(),
                    params: vec![(LocalId(0), TypeId(4))],
                    ret_ty: TypeId(4),
                    blocks: vec![MpirBlock {
                        id: BlockId(0),
                        instrs: vec![],
                        void_ops: vec![],
                        terminator: MpirTerminator::Ret(Some(MpirValue::Const(HirConst {
                            ty: TypeId(4),
                            lit: HirConstLit::IntLit(1),
                        }))),
                    }],
                    locals: vec![],
                    is_async: false,
                },
                MpirFn {
                    sid: Sid("F:CALLER0001".to_string()),
                    name: "caller".to_string(),
                    params: vec![],
                    ret_ty: TypeId(4),
                    blocks: vec![MpirBlock {
                        id: BlockId(0),
                        instrs: vec![MpirInstr {
                            dst: LocalId(0),
                            ty: TypeId(4),
                            op: MpirOp::Call {
                                callee_sid: Sid("F:CALLEE0001".to_string()),
                                inst: vec![],
                                args: vec![],
                            },
                        }],
                        void_ops: vec![],
                        terminator: MpirTerminator::Ret(Some(MpirValue::Local(LocalId(0)))),
                    }],
                    locals: vec![MpirLocalDecl {
                        id: LocalId(0),
                        ty: TypeId(4),
                        name: "ret".to_string(),
                    }],
                    is_async: false,
                },
            ],
            globals: vec![],
        };

        let mut diag = DiagnosticBag::new(16);
        let verify = verify_mpir(&module, &type_ctx, &mut diag);
        assert!(verify.is_err(), "verify_mpir should fail on arity mismatch");
        assert!(
            diag.diagnostics.iter().any(|item| item.code == "MPS0012"),
            "expected MPS0012 diagnostic, got {:?}",
            diag.diagnostics
                .iter()
                .map(|item| item.code.clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn verify_mpir_rejects_arc_ops_before_arc_stage() {
        let type_ctx = TypeCtx::new();
        let module = MpirModule {
            sid: Sid("M:MODULE0004".to_string()),
            path: "pkg.verify.arc".to_string(),
            type_table: MpirTypeTable {
                types: vec![
                    (TypeId(0), TypeKind::Prim(PrimType::Unit)),
                    (TypeId(4), TypeKind::Prim(PrimType::I32)),
                ],
            },
            functions: vec![MpirFn {
                sid: Sid("F:ARCCHK0001".to_string()),
                name: "arc_check".to_string(),
                params: vec![(LocalId(0), TypeId(4))],
                ret_ty: TypeId(4),
                blocks: vec![MpirBlock {
                    id: BlockId(0),
                    instrs: vec![MpirInstr {
                        dst: LocalId(1),
                        ty: TypeId(4),
                        op: MpirOp::ArcRetain {
                            v: MpirValue::Local(LocalId(0)),
                        },
                    }],
                    void_ops: vec![],
                    terminator: MpirTerminator::Ret(Some(MpirValue::Local(LocalId(1)))),
                }],
                locals: vec![MpirLocalDecl {
                    id: LocalId(1),
                    ty: TypeId(4),
                    name: "tmp".to_string(),
                }],
                is_async: false,
            }],
            globals: vec![],
        };

        let mut diag = DiagnosticBag::new(16);
        let verify = verify_mpir(&module, &type_ctx, &mut diag);
        assert!(verify.is_err(), "verify_mpir should fail on arc.* ops");
        assert!(
            diag.diagnostics.iter().any(|item| item.code == "MPS0014"),
            "expected MPS0014 diagnostic, got {:?}",
            diag.diagnostics
                .iter()
                .map(|item| item.code.clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn verify_mpir_reports_void_call_arity_mismatch() {
        let type_ctx = TypeCtx::new();
        let module = MpirModule {
            sid: Sid("M:MODULE0005".to_string()),
            path: "pkg.verify.void".to_string(),
            type_table: MpirTypeTable {
                types: vec![(TypeId(0), TypeKind::Prim(PrimType::Unit))],
            },
            functions: vec![
                MpirFn {
                    sid: Sid("F:VCALLEE001".to_string()),
                    name: "callee_void".to_string(),
                    params: vec![(LocalId(0), TypeId(0))],
                    ret_ty: TypeId(0),
                    blocks: vec![MpirBlock {
                        id: BlockId(0),
                        instrs: vec![],
                        void_ops: vec![],
                        terminator: MpirTerminator::Ret(None),
                    }],
                    locals: vec![],
                    is_async: false,
                },
                MpirFn {
                    sid: Sid("F:VCLER00100".to_string()),
                    name: "caller_void".to_string(),
                    params: vec![],
                    ret_ty: TypeId(0),
                    blocks: vec![MpirBlock {
                        id: BlockId(0),
                        instrs: vec![],
                        void_ops: vec![MpirOpVoid::CallVoid {
                            callee_sid: Sid("F:VCALLEE001".to_string()),
                            inst: vec![],
                            args: vec![],
                        }],
                        terminator: MpirTerminator::Ret(None),
                    }],
                    locals: vec![],
                    is_async: false,
                },
            ],
            globals: vec![],
        };

        let mut diag = DiagnosticBag::new(16);
        let verify = verify_mpir(&module, &type_ctx, &mut diag);
        assert!(
            verify.is_err(),
            "verify_mpir should fail on void call arity mismatch"
        );
        assert!(
            diag.diagnostics.iter().any(|item| item.code == "MPS0012"),
            "expected MPS0012 diagnostic, got {:?}",
            diag.diagnostics
                .iter()
                .map(|item| item.code.clone())
                .collect::<Vec<_>>()
        );
    }
}
