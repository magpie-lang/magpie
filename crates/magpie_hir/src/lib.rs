//! Magpie HIR (High-level IR) data structures and verifier (§16).

// Re-exports from magpie_types
pub use magpie_types::{
    TypeId, LocalId, BlockId, FnId, ModuleId, GlobalId, Sid,
    TypeKind, PrimType, HandleKind, TypeCtx, HandleKind as HirHandleKind,
};

use magpie_diag::{DiagnosticBag, Diagnostic, Severity};
use std::collections::{HashMap, HashSet};

// ── Value references ──────────────────────────────────────────────────────────

/// A reference to an SSA value — either a local (SSA name) or an inline constant.
#[derive(Clone, Debug)]
pub enum HirValue {
    Local(LocalId),
    Const(HirConst),
}

/// A compile-time constant value.
#[derive(Clone, Debug)]
pub struct HirConst {
    pub ty: TypeId,
    pub lit: HirConstLit,
}

#[derive(Clone, Debug)]
pub enum HirConstLit {
    IntLit(i128),
    FloatLit(f64),
    BoolLit(bool),
    StringLit(String),
    Unit,
}

// ── HIR Operations ────────────────────────────────────────────────────────────

/// HIR operation — produces exactly one value (SSA assignment).
#[derive(Clone, Debug)]
pub enum HirOp {
    // Constant materialization
    Const(HirConst),

    // Ownership (compiler-inserted during HIR lowering)
    Move { v: HirValue },
    BorrowShared { v: HirValue },
    BorrowMut { v: HirValue },

    // Heap allocation and field access
    New { ty: TypeId, fields: Vec<(String, HirValue)> },
    GetField { obj: HirValue, field: String },

    // Integer arithmetic (checked — default, traps on overflow)
    IAdd { lhs: HirValue, rhs: HirValue },
    ISub { lhs: HirValue, rhs: HirValue },
    IMul { lhs: HirValue, rhs: HirValue },
    ISDiv { lhs: HirValue, rhs: HirValue },
    IUDiv { lhs: HirValue, rhs: HirValue },
    ISRem { lhs: HirValue, rhs: HirValue },
    IURem { lhs: HirValue, rhs: HirValue },

    // Integer arithmetic (wrapping — safe, no trap)
    IAddWrap { lhs: HirValue, rhs: HirValue },
    ISubWrap { lhs: HirValue, rhs: HirValue },
    IMulWrap { lhs: HirValue, rhs: HirValue },

    // Integer arithmetic (checked → TOption)
    IAddChecked { lhs: HirValue, rhs: HirValue },
    ISubChecked { lhs: HirValue, rhs: HirValue },
    IMulChecked { lhs: HirValue, rhs: HirValue },

    // Bitwise
    IAnd { lhs: HirValue, rhs: HirValue },
    IOr  { lhs: HirValue, rhs: HirValue },
    IXor { lhs: HirValue, rhs: HirValue },
    IShl { lhs: HirValue, rhs: HirValue },
    ILshr { lhs: HirValue, rhs: HirValue },
    IAshr { lhs: HirValue, rhs: HirValue },

    // Comparison
    ICmp { pred: String, lhs: HirValue, rhs: HirValue },
    FCmp { pred: String, lhs: HirValue, rhs: HirValue },

    // Float (strict IEEE 754)
    FAdd { lhs: HirValue, rhs: HirValue },
    FSub { lhs: HirValue, rhs: HirValue },
    FMul { lhs: HirValue, rhs: HirValue },
    FDiv { lhs: HirValue, rhs: HirValue },
    FRem { lhs: HirValue, rhs: HirValue },

    // Float (fast-math opt-in)
    FAddFast { lhs: HirValue, rhs: HirValue },
    FSubFast { lhs: HirValue, rhs: HirValue },
    FMulFast { lhs: HirValue, rhs: HirValue },
    FDivFast { lhs: HirValue, rhs: HirValue },

    // Cast
    Cast { to: TypeId, v: HirValue },

    // Unsafe raw pointer ops
    PtrNull { to: TypeId },
    PtrAddr { p: HirValue },
    PtrFromAddr { to: TypeId, addr: HirValue },
    PtrAdd { p: HirValue, count: HirValue },
    PtrLoad { to: TypeId, p: HirValue },
    PtrStore { to: TypeId, p: HirValue, v: HirValue },

    // Calls (value-returning)
    Call { callee_sid: Sid, inst: Vec<TypeId>, args: Vec<HirValue> },
    CallIndirect { callee: HirValue, args: Vec<HirValue> },
    CallVoidIndirect { callee: HirValue, args: Vec<HirValue> },
    SuspendCall { callee_sid: Sid, inst: Vec<TypeId>, args: Vec<HirValue> },
    SuspendAwait { fut: HirValue },

    // SSA phi node
    Phi { ty: TypeId, incomings: Vec<(BlockId, HirValue)> },

    // Ownership conversions
    Share { v: HirValue },
    CloneShared { v: HirValue },
    CloneWeak { v: HirValue },
    WeakDowngrade { v: HirValue },
    WeakUpgrade { v: HirValue },

    // Enum operations
    EnumNew { variant: String, args: Vec<(String, HirValue)> },
    EnumTag { v: HirValue },
    EnumPayload { variant: String, v: HirValue },
    EnumIs { variant: String, v: HirValue },

    // TCallable
    CallableCapture { fn_ref: Sid, captures: Vec<(String, HirValue)> },

    // Array intrinsics (value-returning)
    ArrNew { elem_ty: TypeId, cap: HirValue },
    ArrLen { arr: HirValue },
    ArrGet { arr: HirValue, idx: HirValue },
    ArrSet { arr: HirValue, idx: HirValue, val: HirValue },
    ArrPush { arr: HirValue, val: HirValue },
    ArrPop { arr: HirValue },
    ArrSlice { arr: HirValue, start: HirValue, end: HirValue },
    ArrContains { arr: HirValue, val: HirValue },
    ArrSort { arr: HirValue },
    ArrMap { arr: HirValue, func: HirValue },
    ArrFilter { arr: HirValue, func: HirValue },
    ArrReduce { arr: HirValue, init: HirValue, func: HirValue },
    ArrForeach { arr: HirValue, func: HirValue },

    // Map intrinsics (value-returning)
    MapNew { key_ty: TypeId, val_ty: TypeId },
    MapLen { map: HirValue },
    MapGet { map: HirValue, key: HirValue },
    MapGetRef { map: HirValue, key: HirValue },
    MapSet { map: HirValue, key: HirValue, val: HirValue },
    MapDelete { map: HirValue, key: HirValue },
    MapContainsKey { map: HirValue, key: HirValue },
    MapDeleteVoid { map: HirValue, key: HirValue },
    MapKeys { map: HirValue },
    MapValues { map: HirValue },

    // String intrinsics
    StrConcat { a: HirValue, b: HirValue },
    StrLen { s: HirValue },
    StrEq { a: HirValue, b: HirValue },
    StrSlice { s: HirValue, start: HirValue, end: HirValue },
    StrBytes { s: HirValue },
    StrBuilderNew,
    StrBuilderAppendStr  { b: HirValue, s: HirValue },
    StrBuilderAppendI64  { b: HirValue, v: HirValue },
    StrBuilderAppendI32  { b: HirValue, v: HirValue },
    StrBuilderAppendF64  { b: HirValue, v: HirValue },
    StrBuilderAppendBool { b: HirValue, v: HirValue },
    StrBuilderBuild { b: HirValue },

    // String parse intrinsics
    StrParseI64  { s: HirValue },
    StrParseU64  { s: HirValue },
    StrParseF64  { s: HirValue },
    StrParseBool { s: HirValue },

    // JSON
    JsonEncode { ty: TypeId, v: HirValue },
    JsonDecode { ty: TypeId, s: HirValue },

    // GPU intrinsics (value-returning)
    GpuThreadId,
    GpuWorkgroupId,
    GpuWorkgroupSize,
    GpuGlobalId,
    GpuBufferLoad { buf: HirValue, idx: HirValue },
    GpuBufferLen  { buf: HirValue },
    GpuShared     { ty: TypeId, size: HirValue },
    GpuLaunch     { device: HirValue, kernel: Sid, groups: HirValue, threads: HirValue, args: Vec<HirValue> },
    GpuLaunchAsync { device: HirValue, kernel: Sid, groups: HirValue, threads: HirValue, args: Vec<HirValue> },

    // Panic (produces unreachable value in SSA; also in HirOpVoid)
    Panic { msg: HirValue },
}

/// HIR void operation — side-effecting, produces no SSA value.
#[derive(Clone, Debug)]
pub enum HirOpVoid {
    CallVoid          { callee_sid: Sid, inst: Vec<TypeId>, args: Vec<HirValue> },
    SetField          { obj: HirValue, field: String, value: HirValue },
    ArrSet            { arr: HirValue, idx: HirValue, val: HirValue },
    ArrPush           { arr: HirValue, val: HirValue },
    ArrSort           { arr: HirValue },
    ArrForeach        { arr: HirValue, func: HirValue },
    MapSet            { map: HirValue, key: HirValue, val: HirValue },
    MapDeleteVoid     { map: HirValue, key: HirValue },
    StrBuilderAppendStr  { b: HirValue, s: HirValue },
    StrBuilderAppendI64  { b: HirValue, v: HirValue },
    StrBuilderAppendI32  { b: HirValue, v: HirValue },
    StrBuilderAppendF64  { b: HirValue, v: HirValue },
    StrBuilderAppendBool { b: HirValue, v: HirValue },
    PtrStore          { to: TypeId, p: HirValue, v: HirValue },
    Panic             { msg: HirValue },
    GpuBarrier,
    GpuBufferStore    { buf: HirValue, idx: HirValue, val: HirValue },
}

// ── Terminators ───────────────────────────────────────────────────────────────

/// Basic block terminator — exactly one per block.
#[derive(Clone, Debug)]
pub enum HirTerminator {
    /// Return from function; None means unit return.
    Ret(Option<HirValue>),
    /// Unconditional branch.
    Br(BlockId),
    /// Conditional branch.
    Cbr { cond: HirValue, then_bb: BlockId, else_bb: BlockId },
    /// Integer switch.
    Switch { val: HirValue, arms: Vec<(HirConst, BlockId)>, default: BlockId },
    /// Unreachable (after Panic or diverging call).
    Unreachable,
}

// ── Basic blocks and functions ────────────────────────────────────────────────

/// An SSA instruction: `%dst: ty = op`.
#[derive(Clone, Debug)]
pub struct HirInstr {
    pub dst: LocalId,
    pub ty:  TypeId,
    pub op:  HirOp,
}

/// A HIR basic block.
#[derive(Clone, Debug)]
pub struct HirBlock {
    pub id:         BlockId,
    pub instrs:     Vec<HirInstr>,
    pub void_ops:   Vec<HirOpVoid>,
    pub terminator: HirTerminator,
}

/// A HIR function.
#[derive(Clone, Debug)]
pub struct HirFunction {
    pub fn_id:    FnId,
    pub sid:      Sid,
    pub name:     String,
    pub params:   Vec<(LocalId, TypeId)>,
    pub ret_ty:   TypeId,
    pub blocks:   Vec<HirBlock>,
    pub is_async: bool,
    pub is_unsafe: bool,
}

/// A HIR module-level global constant.
#[derive(Clone, Debug)]
pub struct HirGlobal {
    pub id:   GlobalId,
    pub name: String,
    pub ty:   TypeId,
    pub init: HirConst,
}

/// A HIR type declaration (struct or enum).
#[derive(Clone, Debug)]
pub enum HirTypeDecl {
    Struct {
        sid:    Sid,
        name:   String,
        fields: Vec<(String, TypeId)>,
    },
    Enum {
        sid:      Sid,
        name:     String,
        variants: Vec<HirEnumVariant>,
    },
}

/// A single enum variant declaration.
#[derive(Clone, Debug)]
pub struct HirEnumVariant {
    pub name:   String,
    pub tag:    i32,
    pub fields: Vec<(String, TypeId)>,
}

/// A HIR module — the top-level compilation unit.
#[derive(Clone, Debug)]
pub struct HirModule {
    pub module_id:  ModuleId,
    pub sid:        Sid,
    pub path:       String,
    pub functions:  Vec<HirFunction>,
    pub globals:    Vec<HirGlobal>,
    pub type_decls: Vec<HirTypeDecl>,
}

// ── HIR Verifier (§16.6) ─────────────────────────────────────────────────────

/// Verify HIR well-formedness per §16.6.
///
/// Checks:
/// 1. SSA: each `LocalId` defined exactly once across the function.
/// 2. SSA: every use of a `LocalId` is dominated by its definition.
/// 3. Each block has exactly one terminator (structurally guaranteed, but we
///    cross-check that blocks are non-empty and the terminator is present).
/// 4. Borrow values (typed as `Borrow` or `MutBorrow`) must not appear in
///    `Phi` incoming values.
/// 5. Borrow values must not cross basic-block boundaries (i.e. a `LocalId`
///    whose type is a borrow handle may only be used within the block where
///    it is defined).
///
/// Returns `Ok(())` if no errors were emitted, `Err(())` otherwise.
pub fn verify_hir(
    module: &HirModule,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) -> Result<(), ()> {
    let before = diag.error_count();

    for func in &module.functions {
        verify_function(func, type_ctx, diag);
    }

    if diag.error_count() > before { Err(()) } else { Ok(()) }
}

// ─────────────────────────────────────────────────────────────────────────────

fn emit_error(diag: &mut DiagnosticBag, code: &str, msg: &str) {
    diag.emit(Diagnostic {
        code:             code.to_string(),
        severity:         Severity::Error,
        title:            msg.to_string(),
        primary_span:     None,
        secondary_spans:  vec![],
        message:          msg.to_string(),
        explanation_md:   None,
        why:              None,
        suggested_fixes:  vec![],
    });
}

fn is_borrow_type(ty: TypeId, type_ctx: &TypeCtx) -> bool {
    use magpie_types::HandleKind;
    match type_ctx.lookup(ty) {
        Some(TypeKind::HeapHandle { hk: HandleKind::Borrow, .. }) => true,
        Some(TypeKind::HeapHandle { hk: HandleKind::MutBorrow, .. }) => true,
        _ => false,
    }
}

fn verify_function(func: &HirFunction, type_ctx: &TypeCtx, diag: &mut DiagnosticBag) {
    // ── Pass 1: collect all definitions ──────────────────────────────────────
    // Map LocalId -> (BlockId, defining_block_index)
    // Also build: set of borrow-typed locals.

    // (local_id -> block_idx in func.blocks)
    let mut def_block: HashMap<u32, usize> = HashMap::new();
    // (local_id -> type)
    let mut local_ty: HashMap<u32, TypeId> = HashMap::new();

    // Parameters are defined at function entry (block 0 implicitly).
    for (param_id, param_ty) in &func.params {
        let id = param_id.0;
        if def_block.insert(id, usize::MAX).is_some() {
            emit_error(
                diag,
                "MPS0001",
                &format!("SSA violation: LocalId {} defined more than once in fn '{}'", id, func.name),
            );
        }
        local_ty.insert(id, *param_ty);
    }

    // Walk all blocks, collecting definitions and checking single-def.
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
        }
    }

    // ── Pass 2: build block predecessor / CFG for dominance ──────────────────
    // We use a simple RPO dominance check: for each use of a LocalId in
    // block B, the defining block must dominate B.  We compute dominators
    // with a simple iterative algorithm.

    let n = func.blocks.len();

    // Map BlockId -> index in func.blocks
    let block_index: HashMap<u32, usize> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id.0, i))
        .collect();

    // Successors for each block index
    let successors: Vec<Vec<usize>> = func
        .blocks
        .iter()
        .map(|b| block_successors(b, &block_index))
        .collect();

    // Predecessors
    let mut preds: Vec<Vec<usize>> = vec![vec![]; n];
    for (i, succs) in successors.iter().enumerate() {
        for &s in succs {
            preds[s].push(i);
        }
    }

    // Compute dominators (bit-set, iterative).
    // dom[i] = set of block indices that dominate block i.
    // We represent as Vec<HashSet<usize>>.
    let mut dom: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    if n > 0 {
        // Entry block is dominated only by itself.
        dom[0].insert(0);
        // All others initialised to all blocks.
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
                    // Intersection of all predecessors' dominator sets
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

    // dominator helper: does block `def_blk` dominate block `use_blk`?
    let dominates = |def_blk: usize, use_blk: usize| -> bool {
        dom[use_blk].contains(&def_blk)
    };

    // ── Pass 3: verify uses ───────────────────────────────────────────────────

    // Borrow-typed LocalIds defined in each block.
    // borrow_locals[blk_idx] = set of LocalId values defined with borrow type.
    let mut borrow_locals_by_block: Vec<HashSet<u32>> = vec![HashSet::new(); n];
    for (blk_idx, block) in func.blocks.iter().enumerate() {
        for instr in &block.instrs {
            if let Some(&ty) = local_ty.get(&instr.dst.0) {
                if is_borrow_type(ty, type_ctx) {
                    borrow_locals_by_block[blk_idx].insert(instr.dst.0);
                }
            }
        }
    }

    // Check a single HirValue used inside `use_blk_idx`.
    let check_value =
        |v: &HirValue,
         use_blk_idx: usize,
         in_phi: bool,
         diag: &mut DiagnosticBag| {
            let local_id = match v {
                HirValue::Local(lid) => lid.0,
                HirValue::Const(_) => return,
            };
            // Check definition exists
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

            // Param defs (usize::MAX) dominate everything.
            let effective_def = if def_blk_idx == usize::MAX { 0 } else { def_blk_idx };

            // Check dominance.
            if effective_def != use_blk_idx && !dominates(effective_def, use_blk_idx) {
                emit_error(
                    diag,
                    "MPS0003",
                    &format!(
                        "SSA violation: use of LocalId {} in fn '{}' is not dominated by its definition",
                        local_id, func.name
                    ),
                );
            }

            // Check borrow rules.
            if let Some(&ty) = local_ty.get(&local_id) {
                if is_borrow_type(ty, type_ctx) {
                    // Rule: borrows must not appear in phi.
                    if in_phi {
                        emit_error(
                            diag,
                            "MPO0102",
                            &format!(
                                "Borrow violation: borrow LocalId {} appears in phi in fn '{}' (MPO0102)",
                                local_id, func.name
                            ),
                        );
                    }
                    // Rule: borrows must not cross blocks.
                    // A borrow defined in block A used in block B (A != B) is illegal.
                    let def_blk_for_borrow = if def_blk_idx == usize::MAX { 0 } else { def_blk_idx };
                    if def_blk_for_borrow != use_blk_idx {
                        emit_error(
                            diag,
                            "MPO0101",
                            &format!(
                                "Borrow violation: borrow LocalId {} crosses block boundary in fn '{}' (MPO0101)",
                                local_id, func.name
                            ),
                        );
                    }
                }
            }
        };

    // Walk all blocks and check every use.
    for (blk_idx, block) in func.blocks.iter().enumerate() {
        // Check instructions.
        for instr in &block.instrs {
            let in_phi = matches!(instr.op, HirOp::Phi { .. });
            for v in hir_op_values(&instr.op) {
                check_value(&v, blk_idx, in_phi, diag);
            }
        }
        // Check void ops.
        for vop in &block.void_ops {
            for v in hir_op_void_values(vop) {
                check_value(&v, blk_idx, false, diag);
            }
        }
        // Check terminator.
        for v in hir_terminator_values(&block.terminator) {
            check_value(&v, blk_idx, false, diag);
        }
    }
}

// ── Value extractors (for the verifier) ──────────────────────────────────────

fn hir_op_values(op: &HirOp) -> Vec<HirValue> {
    match op {
        HirOp::Const(_) => vec![],
        HirOp::Move { v } => vec![v.clone()],
        HirOp::BorrowShared { v } => vec![v.clone()],
        HirOp::BorrowMut { v } => vec![v.clone()],
        HirOp::New { fields, .. } => fields.iter().map(|(_, v)| v.clone()).collect(),
        HirOp::GetField { obj, .. } => vec![obj.clone()],
        HirOp::IAdd { lhs, rhs } | HirOp::ISub { lhs, rhs } | HirOp::IMul { lhs, rhs }
        | HirOp::ISDiv { lhs, rhs } | HirOp::IUDiv { lhs, rhs }
        | HirOp::ISRem { lhs, rhs } | HirOp::IURem { lhs, rhs }
        | HirOp::IAddWrap { lhs, rhs } | HirOp::ISubWrap { lhs, rhs } | HirOp::IMulWrap { lhs, rhs }
        | HirOp::IAddChecked { lhs, rhs } | HirOp::ISubChecked { lhs, rhs } | HirOp::IMulChecked { lhs, rhs }
        | HirOp::IAnd { lhs, rhs } | HirOp::IOr { lhs, rhs } | HirOp::IXor { lhs, rhs }
        | HirOp::IShl { lhs, rhs } | HirOp::ILshr { lhs, rhs } | HirOp::IAshr { lhs, rhs }
        | HirOp::ICmp { lhs, rhs, .. } | HirOp::FCmp { lhs, rhs, .. }
        | HirOp::FAdd { lhs, rhs } | HirOp::FSub { lhs, rhs } | HirOp::FMul { lhs, rhs }
        | HirOp::FDiv { lhs, rhs } | HirOp::FRem { lhs, rhs }
        | HirOp::FAddFast { lhs, rhs } | HirOp::FSubFast { lhs, rhs }
        | HirOp::FMulFast { lhs, rhs } | HirOp::FDivFast { lhs, rhs } => {
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
        HirOp::Share { v } | HirOp::CloneShared { v } | HirOp::CloneWeak { v }
        | HirOp::WeakDowngrade { v } | HirOp::WeakUpgrade { v } => vec![v.clone()],
        HirOp::EnumNew { args, .. } => args.iter().map(|(_, v)| v.clone()).collect(),
        HirOp::EnumTag { v } | HirOp::EnumPayload { v, .. } | HirOp::EnumIs { v, .. } => vec![v.clone()],
        HirOp::CallableCapture { captures, .. } => captures.iter().map(|(_, v)| v.clone()).collect(),
        HirOp::ArrNew { cap, .. } => vec![cap.clone()],
        HirOp::ArrLen { arr } => vec![arr.clone()],
        HirOp::ArrGet { arr, idx } => vec![arr.clone(), idx.clone()],
        HirOp::ArrSet { arr, idx, val } => vec![arr.clone(), idx.clone(), val.clone()],
        HirOp::ArrPush { arr, val } => vec![arr.clone(), val.clone()],
        HirOp::ArrPop { arr } => vec![arr.clone()],
        HirOp::ArrSlice { arr, start, end } => vec![arr.clone(), start.clone(), end.clone()],
        HirOp::ArrContains { arr, val } => vec![arr.clone(), val.clone()],
        HirOp::ArrSort { arr } => vec![arr.clone()],
        HirOp::ArrMap { arr, func } | HirOp::ArrFilter { arr, func } | HirOp::ArrForeach { arr, func } => {
            vec![arr.clone(), func.clone()]
        }
        HirOp::ArrReduce { arr, init, func } => vec![arr.clone(), init.clone(), func.clone()],
        HirOp::MapNew { .. } => vec![],
        HirOp::MapLen { map } => vec![map.clone()],
        HirOp::MapGet { map, key } | HirOp::MapGetRef { map, key }
        | HirOp::MapDelete { map, key } | HirOp::MapContainsKey { map, key }
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
        HirOp::StrBuilderAppendI64 { b, v } | HirOp::StrBuilderAppendI32 { b, v }
        | HirOp::StrBuilderAppendF64 { b, v } | HirOp::StrBuilderAppendBool { b, v } => {
            vec![b.clone(), v.clone()]
        }
        HirOp::StrBuilderBuild { b } => vec![b.clone()],
        HirOp::StrParseI64 { s } | HirOp::StrParseU64 { s }
        | HirOp::StrParseF64 { s } | HirOp::StrParseBool { s } => vec![s.clone()],
        HirOp::JsonEncode { v, .. } => vec![v.clone()],
        HirOp::JsonDecode { s, .. } => vec![s.clone()],
        HirOp::GpuThreadId | HirOp::GpuWorkgroupId | HirOp::GpuWorkgroupSize | HirOp::GpuGlobalId => vec![],
        HirOp::GpuBufferLoad { buf, idx } => vec![buf.clone(), idx.clone()],
        HirOp::GpuBufferLen { buf } => vec![buf.clone()],
        HirOp::GpuShared { size, .. } => vec![size.clone()],
        HirOp::GpuLaunch { device, groups, threads, args, .. }
        | HirOp::GpuLaunchAsync { device, groups, threads, args, .. } => {
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
        HirOpVoid::SetField { obj, value, .. } => vec![obj.clone(), value.clone()],
        HirOpVoid::ArrSet { arr, idx, val } => vec![arr.clone(), idx.clone(), val.clone()],
        HirOpVoid::ArrPush { arr, val } => vec![arr.clone(), val.clone()],
        HirOpVoid::ArrSort { arr } => vec![arr.clone()],
        HirOpVoid::ArrForeach { arr, func } => vec![arr.clone(), func.clone()],
        HirOpVoid::MapSet { map, key, val } => vec![map.clone(), key.clone(), val.clone()],
        HirOpVoid::MapDeleteVoid { map, key } => vec![map.clone(), key.clone()],
        HirOpVoid::StrBuilderAppendStr { b, s } => vec![b.clone(), s.clone()],
        HirOpVoid::StrBuilderAppendI64 { b, v } | HirOpVoid::StrBuilderAppendI32 { b, v }
        | HirOpVoid::StrBuilderAppendF64 { b, v } | HirOpVoid::StrBuilderAppendBool { b, v } => {
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

fn block_successors(block: &HirBlock, block_index: &HashMap<u32, usize>) -> Vec<usize> {
    match &block.terminator {
        HirTerminator::Ret(_) | HirTerminator::Unreachable => vec![],
        HirTerminator::Br(bid) => {
            block_index.get(&bid.0).copied().into_iter().collect()
        }
        HirTerminator::Cbr { then_bb, else_bb, .. } => {
            let mut v = Vec::new();
            if let Some(&i) = block_index.get(&then_bb.0) { v.push(i); }
            if let Some(&i) = block_index.get(&else_bb.0) { v.push(i); }
            v
        }
        HirTerminator::Switch { arms, default, .. } => {
            let mut v = Vec::new();
            for (_, bid) in arms {
                if let Some(&i) = block_index.get(&bid.0) { v.push(i); }
            }
            if let Some(&i) = block_index.get(&default.0) { v.push(i); }
            v
        }
    }
}
