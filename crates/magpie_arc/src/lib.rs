//! ARC insertion and local ARC peephole optimization for MPIR.

use magpie_diag::{Diagnostic, DiagnosticBag, Severity};
use magpie_mpir::{
    HandleKind, HeapBase, LocalId, MpirBlock, MpirFn, MpirInstr, MpirLocalDecl, MpirModule,
    MpirOp, MpirOpVoid, MpirTerminator, MpirValue, TypeCtx, TypeId, TypeKind,
};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Default)]
struct MovedAnalysis {
    moved_in: Vec<HashSet<LocalId>>,
    moved_out: Vec<HashSet<LocalId>>,
    edge_phi_consumes: HashMap<(usize, usize), HashSet<LocalId>>,
    phi_consumed_out: Vec<HashSet<LocalId>>,
}

#[derive(Debug, Default)]
struct LivenessAnalysis {
    live_in: Vec<HashSet<LocalId>>,
    live_out: Vec<HashSet<LocalId>>,
    block_defs: Vec<HashSet<LocalId>>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum DropKind {
    Strong,
    Weak,
}

pub fn insert_arc_ops(
    module: &mut MpirModule,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) -> Result<(), ()> {
    let before = diag.error_count();
    let module_types = &mut module.type_table.types;

    for func in &mut module.functions {
        insert_arc_ops_for_fn(func, module_types, type_ctx, diag);
    }

    if diag.error_count() > before {
        Err(())
    } else {
        Ok(())
    }
}

pub fn optimize_arc(module: &mut MpirModule, type_ctx: &TypeCtx) {
    let _ = type_ctx;

    for func in &mut module.functions {
        for block in &mut func.blocks {
            optimize_arc_void_ops(&mut block.void_ops);
        }
    }
}

fn insert_arc_ops_for_fn(
    func: &mut MpirFn,
    module_types: &mut Vec<(TypeId, TypeKind)>,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) {
    let mut local_types = collect_local_types(func);
    let mut pre_void_arc_ops: Vec<Vec<MpirOpVoid>> = vec![Vec::new(); func.blocks.len()];
    let mut share_retyped: Vec<(LocalId, TypeId)> = Vec::new();

    for (blk_idx, block) in func.blocks.iter_mut().enumerate() {
        for instr in &mut block.instrs {
            match &instr.op {
                MpirOp::Share { v } => {
                    let Some(src_ty) = value_type(v, &local_types) else {
                        emit_error(
                            diag,
                            "MPA0001",
                            &format!(
                                "ARC insertion: cannot resolve source type for share dst %{} in fn '{}'",
                                instr.dst.0, func.name
                            ),
                        );
                        continue;
                    };

                    let src_kind = lookup_type_kind(src_ty, type_ctx, module_types).cloned();
                    let Some(src_kind) = src_kind else {
                        emit_error(
                            diag,
                            "MPA0002",
                            &format!(
                                "ARC insertion: unknown type_id {} for share source in fn '{}'",
                                src_ty.0, func.name
                            ),
                        );
                        continue;
                    };

                    if let TypeKind::HeapHandle {
                        hk: HandleKind::Unique,
                        base,
                    } = src_kind
                    {
                        let Some(shared_ty) = ensure_shared_type_id(&base, module_types, type_ctx)
                        else {
                            emit_error(
                                diag,
                                "MPA0003",
                                &format!(
                                    "ARC insertion: missing shared type for base {:?} in fn '{}'",
                                    base, func.name
                                ),
                            );
                            continue;
                        };
                        if instr.ty != shared_ty {
                            instr.ty = shared_ty;
                            share_retyped.push((instr.dst, shared_ty));
                        }
                    }
                }
                MpirOp::CloneShared { v } => {
                    pre_void_arc_ops[blk_idx].push(MpirOpVoid::ArcRetain { v: v.clone() });
                }
                MpirOp::CloneWeak { v } | MpirOp::WeakDowngrade { v } => {
                    pre_void_arc_ops[blk_idx].push(MpirOpVoid::ArcRetainWeak { v: v.clone() });
                }
                _ => {}
            }
            local_types.insert(instr.dst, instr.ty);
        }
    }

    for (local, ty) in share_retyped {
        if let Some(decl) = func.locals.iter_mut().find(|d| d.id == local) {
            decl.ty = ty;
        }
        local_types.insert(local, ty);
    }

    let block_index = build_block_index(func);
    let successors: Vec<Vec<usize>> = func
        .blocks
        .iter()
        .map(|b| block_successors(b, &block_index))
        .collect();
    let preds = build_predecessors(successors.len(), &successors);

    let track_move_locals: HashSet<LocalId> = local_types
        .iter()
        .filter_map(|(l, ty)| {
            if drop_kind_for_type(*ty, type_ctx, module_types).is_some() {
                Some(*l)
            } else {
                None
            }
        })
        .collect();

    let moved = analyze_moved_sets(func, &track_move_locals, &block_index, &preds);
    let liveness = analyze_liveness(func, &block_index, &successors);
    let entry_params: HashSet<LocalId> = func.params.iter().map(|(l, _)| *l).collect();

    let mut next_local = next_local_id(func);

    for blk_idx in 0..func.blocks.len() {
        let block = &mut func.blocks[blk_idx];
        let mut new_void_ops = std::mem::take(&mut pre_void_arc_ops[blk_idx]);
        let old_void_ops = std::mem::take(&mut block.void_ops);
        let mut generated_instrs: Vec<MpirInstr> = Vec::new();

        for op in old_void_ops {
            match op {
                MpirOpVoid::SetField { obj, field, value } => {
                    if let Some(field_ty) = value_type(&value, &local_types) {
                        if drop_kind_for_type(field_ty, type_ctx, module_types).is_some() {
                            let old_local = LocalId(next_local);
                            next_local += 1;

                            func.locals.push(MpirLocalDecl {
                                id: old_local,
                                ty: field_ty,
                                name: format!("__arc_old_{}", old_local.0),
                            });
                            local_types.insert(old_local, field_ty);

                            generated_instrs.push(MpirInstr {
                                dst: old_local,
                                ty: field_ty,
                                op: MpirOp::GetField {
                                    obj: obj.clone(),
                                    field: field.clone(),
                                },
                            });

                            emit_drop_for_value(
                                MpirValue::Local(old_local),
                                field_ty,
                                type_ctx,
                                module_types,
                                &mut new_void_ops,
                            );
                        }
                    } else {
                        emit_error(
                            diag,
                            "MPA0004",
                            &format!(
                                "ARC insertion: cannot resolve setfield value type in fn '{}', bb{}",
                                func.name, block.id.0
                            ),
                        );
                    }

                    new_void_ops.push(MpirOpVoid::SetField { obj, field, value });
                }
                _ => new_void_ops.push(op),
            }
        }

        block.instrs.extend(generated_instrs);

        let mut release_locals: HashSet<LocalId> = HashSet::new();
        release_locals.extend(liveness.block_defs[blk_idx].iter().copied());
        release_locals.extend(liveness.live_in[blk_idx].iter().copied());
        if blk_idx == 0 {
            release_locals.extend(entry_params.iter().copied());
        }

        let mut release_locals_vec: Vec<LocalId> = release_locals.into_iter().collect();
        release_locals_vec.sort_by_key(|l| l.0);
        release_locals_vec.dedup();

        for local in release_locals_vec {
            if liveness.live_out[blk_idx].contains(&local) {
                continue;
            }
            if moved.moved_out[blk_idx].contains(&local) {
                continue;
            }
            if moved.phi_consumed_out[blk_idx].contains(&local) {
                continue;
            }
            let Some(ty) = local_types.get(&local).copied() else {
                continue;
            };
            emit_drop_for_value(
                MpirValue::Local(local),
                ty,
                type_ctx,
                module_types,
                &mut new_void_ops,
            );
        }

        block.void_ops = new_void_ops;
    }
}

fn optimize_arc_void_ops(void_ops: &mut Vec<MpirOpVoid>) {
    let old = std::mem::take(void_ops);
    let mut out = Vec::with_capacity(old.len());

    for op in old {
        let cancel = match (&op, out.last()) {
            (MpirOpVoid::ArcRelease { v: vr }, Some(MpirOpVoid::ArcRetain { v: vt })) => {
                same_value(vr, vt)
            }
            (
                MpirOpVoid::ArcReleaseWeak { v: vr },
                Some(MpirOpVoid::ArcRetainWeak { v: vt }),
            ) => same_value(vr, vt),
            _ => false,
        };

        if cancel {
            out.pop();
        } else {
            out.push(op);
        }
    }

    *void_ops = out;
}

fn analyze_moved_sets(
    func: &MpirFn,
    track_locals: &HashSet<LocalId>,
    block_index: &HashMap<u32, usize>,
    preds: &[Vec<usize>],
) -> MovedAnalysis {
    let n = func.blocks.len();
    let mut analysis = MovedAnalysis {
        moved_in: vec![HashSet::new(); n],
        moved_out: vec![HashSet::new(); n],
        edge_phi_consumes: HashMap::new(),
        phi_consumed_out: vec![HashSet::new(); n],
    };

    for (succ_idx, block) in func.blocks.iter().enumerate() {
        for instr in &block.instrs {
            let MpirOp::Phi { incomings, .. } = &instr.op else {
                continue;
            };
            for (pred_bid, v) in incomings {
                let Some(local) = as_local(v) else {
                    continue;
                };
                if !track_locals.contains(&local) {
                    continue;
                }
                let Some(&pred_idx) = block_index.get(&pred_bid.0) else {
                    continue;
                };
                analysis
                    .edge_phi_consumes
                    .entry((pred_idx, succ_idx))
                    .or_default()
                    .insert(local);
                analysis.phi_consumed_out[pred_idx].insert(local);
            }
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for (idx, block) in func.blocks.iter().enumerate() {
            let mut new_in = HashSet::new();
            for &p in &preds[idx] {
                new_in.extend(analysis.moved_out[p].iter().copied());
                if let Some(edge) = analysis.edge_phi_consumes.get(&(p, idx)) {
                    new_in.extend(edge.iter().copied());
                }
            }

            let mut cur = new_in.clone();
            for instr in &block.instrs {
                for local in op_consumed_locals(&instr.op) {
                    if track_locals.contains(&local) {
                        cur.insert(local);
                    }
                }
            }
            for vop in &block.void_ops {
                for local in op_void_consumed_locals(vop) {
                    if track_locals.contains(&local) {
                        cur.insert(local);
                    }
                }
            }
            for local in terminator_consumed_locals(&block.terminator) {
                if track_locals.contains(&local) {
                    cur.insert(local);
                }
            }

            if new_in != analysis.moved_in[idx] || cur != analysis.moved_out[idx] {
                analysis.moved_in[idx] = new_in;
                analysis.moved_out[idx] = cur;
                changed = true;
            }
        }
    }

    analysis
}

fn analyze_liveness(
    func: &MpirFn,
    block_index: &HashMap<u32, usize>,
    successors: &[Vec<usize>],
) -> LivenessAnalysis {
    let n = func.blocks.len();
    let mut uses = vec![HashSet::new(); n];
    let mut defs = vec![HashSet::new(); n];
    let mut phi_defs = vec![HashSet::new(); n];
    let mut edge_phi_uses: HashMap<(usize, usize), HashSet<LocalId>> = HashMap::new();

    for (idx, block) in func.blocks.iter().enumerate() {
        for instr in &block.instrs {
            if let MpirOp::Phi { incomings, .. } = &instr.op {
                defs[idx].insert(instr.dst);
                phi_defs[idx].insert(instr.dst);

                for (pred_bid, v) in incomings {
                    let Some(local) = as_local(v) else {
                        continue;
                    };
                    let Some(&pred_idx) = block_index.get(&pred_bid.0) else {
                        continue;
                    };
                    edge_phi_uses
                        .entry((pred_idx, idx))
                        .or_default()
                        .insert(local);
                }
                continue;
            }

            for local in op_used_locals(&instr.op) {
                if !defs[idx].contains(&local) {
                    uses[idx].insert(local);
                }
            }
            defs[idx].insert(instr.dst);
        }

        for vop in &block.void_ops {
            for local in op_void_used_locals(vop) {
                if !defs[idx].contains(&local) {
                    uses[idx].insert(local);
                }
            }
        }

        for local in terminator_used_locals(&block.terminator) {
            if !defs[idx].contains(&local) {
                uses[idx].insert(local);
            }
        }
    }

    let mut live_in = vec![HashSet::new(); n];
    let mut live_out = vec![HashSet::new(); n];
    let mut changed = true;

    while changed {
        changed = false;
        for idx in (0..n).rev() {
            let mut new_out = HashSet::new();
            for &succ in &successors[idx] {
                let mut carry = live_in[succ].clone();
                for phi_def in &phi_defs[succ] {
                    carry.remove(phi_def);
                }
                new_out.extend(carry.into_iter());

                if let Some(edge_uses) = edge_phi_uses.get(&(idx, succ)) {
                    new_out.extend(edge_uses.iter().copied());
                }
            }

            let mut new_in = uses[idx].clone();
            let mut out_minus_def = new_out.clone();
            for def in &defs[idx] {
                out_minus_def.remove(def);
            }
            new_in.extend(out_minus_def.into_iter());

            if new_in != live_in[idx] || new_out != live_out[idx] {
                live_in[idx] = new_in;
                live_out[idx] = new_out;
                changed = true;
            }
        }
    }

    LivenessAnalysis {
        live_in,
        live_out,
        block_defs: defs,
    }
}

fn collect_local_types(func: &MpirFn) -> HashMap<LocalId, TypeId> {
    let mut out = HashMap::new();
    for (local, ty) in &func.params {
        out.insert(*local, *ty);
    }
    for local in &func.locals {
        out.insert(local.id, local.ty);
    }
    for block in &func.blocks {
        for instr in &block.instrs {
            out.insert(instr.dst, instr.ty);
        }
    }
    out
}

fn value_type(v: &MpirValue, local_types: &HashMap<LocalId, TypeId>) -> Option<TypeId> {
    match v {
        MpirValue::Local(l) => local_types.get(l).copied(),
        MpirValue::Const(c) => Some(c.ty),
    }
}

fn as_local(v: &MpirValue) -> Option<LocalId> {
    match v {
        MpirValue::Local(l) => Some(*l),
        MpirValue::Const(_) => None,
    }
}

fn same_value(a: &MpirValue, b: &MpirValue) -> bool {
    match (a, b) {
        (MpirValue::Local(x), MpirValue::Local(y)) => x == y,
        _ => false,
    }
}

fn lookup_type_kind<'a>(
    ty: TypeId,
    type_ctx: &'a TypeCtx,
    module_types: &'a [(TypeId, TypeKind)],
) -> Option<&'a TypeKind> {
    type_ctx
        .lookup(ty)
        .or_else(|| module_types.iter().find(|(id, _)| *id == ty).map(|(_, kind)| kind))
}

fn ensure_shared_type_id(
    base: &HeapBase,
    module_types: &mut Vec<(TypeId, TypeKind)>,
    type_ctx: &TypeCtx,
) -> Option<TypeId> {
    let expected = TypeKind::HeapHandle {
        hk: HandleKind::Shared,
        base: base.clone(),
    };

    if let Some((id, _)) = module_types.iter().find(|(_, k)| *k == expected) {
        return Some(*id);
    }

    if let Some((id, kind)) = type_ctx.types.iter().find(|(_, k)| *k == expected) {
        if !module_types.iter().any(|(mid, _)| *mid == *id) {
            module_types.push((*id, kind.clone()));
        }
        return Some(*id);
    }

    None
}

fn drop_kind_for_type(
    ty: TypeId,
    type_ctx: &TypeCtx,
    module_types: &[(TypeId, TypeKind)],
) -> Option<DropKind> {
    let kind = lookup_type_kind(ty, type_ctx, module_types)?;
    match kind {
        TypeKind::HeapHandle {
            hk: HandleKind::Unique | HandleKind::Shared,
            ..
        } => Some(DropKind::Strong),
        TypeKind::HeapHandle {
            hk: HandleKind::Weak,
            ..
        } => Some(DropKind::Weak),
        TypeKind::BuiltinOption { inner } => drop_kind_for_type(*inner, type_ctx, module_types),
        _ => None,
    }
}

fn emit_drop_for_value(
    v: MpirValue,
    ty: TypeId,
    type_ctx: &TypeCtx,
    module_types: &[(TypeId, TypeKind)],
    out: &mut Vec<MpirOpVoid>,
) {
    match drop_kind_for_type(ty, type_ctx, module_types) {
        Some(DropKind::Strong) => out.push(MpirOpVoid::ArcRelease { v }),
        Some(DropKind::Weak) => out.push(MpirOpVoid::ArcReleaseWeak { v }),
        None => {}
    }
}

fn next_local_id(func: &MpirFn) -> u32 {
    let mut max_id = 0u32;
    for (id, _) in &func.params {
        max_id = max_id.max(id.0);
    }
    for decl in &func.locals {
        max_id = max_id.max(decl.id.0);
    }
    for block in &func.blocks {
        for instr in &block.instrs {
            max_id = max_id.max(instr.dst.0);
        }
    }
    max_id.saturating_add(1)
}

fn build_block_index(func: &MpirFn) -> HashMap<u32, usize> {
    func.blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id.0, i))
        .collect()
}

fn build_predecessors(n: usize, succs: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut preds = vec![Vec::new(); n];
    for (i, vs) in succs.iter().enumerate() {
        for &s in vs {
            preds[s].push(i);
        }
    }
    preds
}

fn block_successors(block: &MpirBlock, block_index: &HashMap<u32, usize>) -> Vec<usize> {
    match &block.terminator {
        MpirTerminator::Ret(_) | MpirTerminator::Unreachable => vec![],
        MpirTerminator::Br(bid) => block_index.get(&bid.0).copied().into_iter().collect(),
        MpirTerminator::Cbr {
            then_bb, else_bb, ..
        } => {
            let mut out = Vec::new();
            if let Some(&idx) = block_index.get(&then_bb.0) {
                out.push(idx);
            }
            if let Some(&idx) = block_index.get(&else_bb.0) {
                out.push(idx);
            }
            out
        }
        MpirTerminator::Switch { arms, default, .. } => {
            let mut out = Vec::new();
            for (_, bid) in arms {
                if let Some(&idx) = block_index.get(&bid.0) {
                    out.push(idx);
                }
            }
            if let Some(&idx) = block_index.get(&default.0) {
                out.push(idx);
            }
            out
        }
    }
}

fn op_used_locals(op: &MpirOp) -> Vec<LocalId> {
    let mut out = Vec::new();
    for_each_value_in_op(op, |v| {
        if let Some(l) = as_local(v) {
            out.push(l);
        }
    });
    out
}

fn op_void_used_locals(op: &MpirOpVoid) -> Vec<LocalId> {
    let mut out = Vec::new();
    for_each_value_in_void_op(op, |v| {
        if let Some(l) = as_local(v) {
            out.push(l);
        }
    });
    out
}

fn terminator_used_locals(term: &MpirTerminator) -> Vec<LocalId> {
    let mut out = Vec::new();
    for_each_value_in_terminator(term, |v| {
        if let Some(l) = as_local(v) {
            out.push(l);
        }
    });
    out
}

fn op_consumed_locals(op: &MpirOp) -> Vec<LocalId> {
    let mut out = Vec::new();
    let push = |v: &MpirValue, out: &mut Vec<LocalId>| {
        if let Some(l) = as_local(v) {
            out.push(l);
        }
    };

    match op {
        MpirOp::Const(_) => {}
        MpirOp::Move { v } => push(v, &mut out),
        MpirOp::BorrowShared { .. } | MpirOp::BorrowMut { .. } => {}

        MpirOp::New { fields, .. } => {
            for (_, v) in fields {
                push(v, &mut out);
            }
        }

        MpirOp::GetField { .. }
        | MpirOp::IAdd { .. }
        | MpirOp::ISub { .. }
        | MpirOp::IMul { .. }
        | MpirOp::ISDiv { .. }
        | MpirOp::IUDiv { .. }
        | MpirOp::ISRem { .. }
        | MpirOp::IURem { .. }
        | MpirOp::IAddWrap { .. }
        | MpirOp::ISubWrap { .. }
        | MpirOp::IMulWrap { .. }
        | MpirOp::IAddChecked { .. }
        | MpirOp::ISubChecked { .. }
        | MpirOp::IMulChecked { .. }
        | MpirOp::IAnd { .. }
        | MpirOp::IOr { .. }
        | MpirOp::IXor { .. }
        | MpirOp::IShl { .. }
        | MpirOp::ILshr { .. }
        | MpirOp::IAshr { .. }
        | MpirOp::ICmp { .. }
        | MpirOp::FCmp { .. }
        | MpirOp::FAdd { .. }
        | MpirOp::FSub { .. }
        | MpirOp::FMul { .. }
        | MpirOp::FDiv { .. }
        | MpirOp::FRem { .. }
        | MpirOp::FAddFast { .. }
        | MpirOp::FSubFast { .. }
        | MpirOp::FMulFast { .. }
        | MpirOp::FDivFast { .. }
        | MpirOp::Cast { .. }
        | MpirOp::PtrNull { .. }
        | MpirOp::PtrAddr { .. }
        | MpirOp::PtrFromAddr { .. }
        | MpirOp::PtrAdd { .. }
        | MpirOp::PtrLoad { .. }
        | MpirOp::SuspendAwait { .. }
        | MpirOp::CloneShared { .. }
        | MpirOp::CloneWeak { .. }
        | MpirOp::WeakDowngrade { .. }
        | MpirOp::WeakUpgrade { .. }
        | MpirOp::ArcRetain { .. }
        | MpirOp::ArcRelease { .. }
        | MpirOp::ArcRetainWeak { .. }
        | MpirOp::ArcReleaseWeak { .. }
        | MpirOp::EnumTag { .. }
        | MpirOp::EnumPayload { .. }
        | MpirOp::EnumIs { .. }
        | MpirOp::ArrNew { .. }
        | MpirOp::ArrLen { .. }
        | MpirOp::ArrGet { .. }
        | MpirOp::ArrPop { .. }
        | MpirOp::ArrSlice { .. }
        | MpirOp::ArrContains { .. }
        | MpirOp::ArrSort { .. }
        | MpirOp::ArrMap { .. }
        | MpirOp::ArrFilter { .. }
        | MpirOp::ArrForeach { .. }
        | MpirOp::MapNew { .. }
        | MpirOp::MapLen { .. }
        | MpirOp::MapGet { .. }
        | MpirOp::MapGetRef { .. }
        | MpirOp::MapDelete { .. }
        | MpirOp::MapContainsKey { .. }
        | MpirOp::MapDeleteVoid { .. }
        | MpirOp::MapKeys { .. }
        | MpirOp::MapValues { .. }
        | MpirOp::StrLen { .. }
        | MpirOp::StrEq { .. }
        | MpirOp::StrSlice { .. }
        | MpirOp::StrBytes { .. }
        | MpirOp::StrBuilderNew
        | MpirOp::StrBuilderAppendStr { .. }
        | MpirOp::StrBuilderAppendI64 { .. }
        | MpirOp::StrBuilderAppendI32 { .. }
        | MpirOp::StrBuilderAppendF64 { .. }
        | MpirOp::StrBuilderAppendBool { .. }
        | MpirOp::StrParseI64 { .. }
        | MpirOp::StrParseU64 { .. }
        | MpirOp::StrParseF64 { .. }
        | MpirOp::StrParseBool { .. }
        | MpirOp::JsonEncode { .. }
        | MpirOp::JsonDecode { .. }
        | MpirOp::GpuThreadId
        | MpirOp::GpuWorkgroupId
        | MpirOp::GpuWorkgroupSize
        | MpirOp::GpuGlobalId
        | MpirOp::GpuBufferLoad { .. }
        | MpirOp::GpuBufferLen { .. }
        | MpirOp::GpuShared { .. }
        | MpirOp::Panic { .. }
        | MpirOp::Phi { .. } => {}

        MpirOp::PtrStore { v, .. } => push(v, &mut out),
        MpirOp::Call { args, .. } | MpirOp::SuspendCall { args, .. } => {
            for v in args {
                push(v, &mut out);
            }
        }
        MpirOp::CallIndirect { args, .. } | MpirOp::CallVoidIndirect { args, .. } => {
            for v in args {
                push(v, &mut out);
            }
        }
        MpirOp::Share { v } => push(v, &mut out),
        MpirOp::EnumNew { args, .. } => {
            for (_, v) in args {
                push(v, &mut out);
            }
        }
        MpirOp::CallableCapture { captures, .. } => {
            for (_, v) in captures {
                push(v, &mut out);
            }
        }
        MpirOp::ArrSet { val, .. } | MpirOp::ArrPush { val, .. } => push(val, &mut out),
        MpirOp::ArrReduce { init, .. } => push(init, &mut out),
        MpirOp::MapSet { key, val, .. } => {
            push(key, &mut out);
            push(val, &mut out);
        }
        MpirOp::StrConcat { a, b } => {
            push(a, &mut out);
            push(b, &mut out);
        }
        MpirOp::StrBuilderBuild { b } => push(b, &mut out),
        MpirOp::GpuLaunch { args, .. } | MpirOp::GpuLaunchAsync { args, .. } => {
            for v in args {
                push(v, &mut out);
            }
        }
    }

    out
}

fn op_void_consumed_locals(op: &MpirOpVoid) -> Vec<LocalId> {
    let mut out = Vec::new();
    let push = |v: &MpirValue, out: &mut Vec<LocalId>| {
        if let Some(l) = as_local(v) {
            out.push(l);
        }
    };

    match op {
        MpirOpVoid::CallVoid { args, .. } => {
            for v in args {
                push(v, &mut out);
            }
        }
        MpirOpVoid::SetField { value, .. } => push(value, &mut out),
        MpirOpVoid::ArrSet { val, .. } | MpirOpVoid::ArrPush { val, .. } => push(val, &mut out),
        MpirOpVoid::MapSet { key, val, .. } => {
            push(key, &mut out);
            push(val, &mut out);
        }
        MpirOpVoid::PtrStore { v, .. } => push(v, &mut out),
        MpirOpVoid::GpuBufferStore { val, .. } => push(val, &mut out),

        MpirOpVoid::ArrSort { .. }
        | MpirOpVoid::ArrForeach { .. }
        | MpirOpVoid::MapDeleteVoid { .. }
        | MpirOpVoid::StrBuilderAppendStr { .. }
        | MpirOpVoid::StrBuilderAppendI64 { .. }
        | MpirOpVoid::StrBuilderAppendI32 { .. }
        | MpirOpVoid::StrBuilderAppendF64 { .. }
        | MpirOpVoid::StrBuilderAppendBool { .. }
        | MpirOpVoid::Panic { .. }
        | MpirOpVoid::GpuBarrier
        | MpirOpVoid::ArcRetain { .. }
        | MpirOpVoid::ArcRelease { .. }
        | MpirOpVoid::ArcRetainWeak { .. }
        | MpirOpVoid::ArcReleaseWeak { .. } => {}
    }

    out
}

fn terminator_consumed_locals(term: &MpirTerminator) -> Vec<LocalId> {
    match term {
        MpirTerminator::Ret(Some(v)) => as_local(v).into_iter().collect(),
        MpirTerminator::Ret(None)
        | MpirTerminator::Br(_)
        | MpirTerminator::Cbr { .. }
        | MpirTerminator::Switch { .. }
        | MpirTerminator::Unreachable => vec![],
    }
}

fn for_each_value_in_op(op: &MpirOp, mut f: impl FnMut(&MpirValue)) {
    match op {
        MpirOp::Const(_) => {}
        MpirOp::Move { v }
        | MpirOp::BorrowShared { v }
        | MpirOp::BorrowMut { v }
        | MpirOp::Cast { v, .. }
        | MpirOp::PtrAddr { p: v }
        | MpirOp::PtrFromAddr { addr: v, .. }
        | MpirOp::PtrLoad { p: v, .. }
        | MpirOp::Share { v }
        | MpirOp::CloneShared { v }
        | MpirOp::CloneWeak { v }
        | MpirOp::WeakDowngrade { v }
        | MpirOp::WeakUpgrade { v }
        | MpirOp::ArcRetain { v }
        | MpirOp::ArcRelease { v }
        | MpirOp::ArcRetainWeak { v }
        | MpirOp::ArcReleaseWeak { v }
        | MpirOp::EnumTag { v }
        | MpirOp::EnumPayload { v, .. }
        | MpirOp::EnumIs { v, .. }
        | MpirOp::ArrNew { cap: v, .. }
        | MpirOp::ArrLen { arr: v }
        | MpirOp::ArrPop { arr: v }
        | MpirOp::ArrSort { arr: v }
        | MpirOp::MapLen { map: v }
        | MpirOp::MapKeys { map: v }
        | MpirOp::MapValues { map: v }
        | MpirOp::StrLen { s: v }
        | MpirOp::StrBytes { s: v }
        | MpirOp::StrBuilderBuild { b: v }
        | MpirOp::StrParseI64 { s: v }
        | MpirOp::StrParseU64 { s: v }
        | MpirOp::StrParseF64 { s: v }
        | MpirOp::StrParseBool { s: v }
        | MpirOp::SuspendAwait { fut: v }
        | MpirOp::JsonEncode { v, .. }
        | MpirOp::JsonDecode { s: v, .. }
        | MpirOp::GpuBufferLen { buf: v }
        | MpirOp::GpuShared { size: v, .. }
        | MpirOp::Panic { msg: v }
        | MpirOp::GetField { obj: v, .. } => f(v),

        MpirOp::New { fields, .. } => {
            for (_, v) in fields {
                f(v);
            }
        }
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
        | MpirOp::FDivFast { lhs, rhs }
        | MpirOp::StrConcat { a: lhs, b: rhs }
        | MpirOp::StrEq { a: lhs, b: rhs } => {
            f(lhs);
            f(rhs);
        }
        MpirOp::PtrNull { .. }
        | MpirOp::MapNew { .. }
        | MpirOp::StrBuilderNew
        | MpirOp::GpuThreadId
        | MpirOp::GpuWorkgroupId
        | MpirOp::GpuWorkgroupSize
        | MpirOp::GpuGlobalId => {}

        MpirOp::PtrAdd { p, count }
        | MpirOp::ArrGet { arr: p, idx: count }
        | MpirOp::ArrContains { arr: p, val: count }
        | MpirOp::MapGet { map: p, key: count }
        | MpirOp::MapGetRef { map: p, key: count }
        | MpirOp::MapDelete { map: p, key: count }
        | MpirOp::MapContainsKey { map: p, key: count }
        | MpirOp::MapDeleteVoid { map: p, key: count }
        | MpirOp::StrBuilderAppendStr { b: p, s: count }
        | MpirOp::StrBuilderAppendI64 { b: p, v: count }
        | MpirOp::StrBuilderAppendI32 { b: p, v: count }
        | MpirOp::StrBuilderAppendF64 { b: p, v: count }
        | MpirOp::StrBuilderAppendBool { b: p, v: count }
        | MpirOp::GpuBufferLoad { buf: p, idx: count }
        | MpirOp::PtrStore { p, v: count, .. } => {
            f(p);
            f(count);
        }

        MpirOp::Call { args, .. } | MpirOp::SuspendCall { args, .. } => {
            for arg in args {
                f(arg);
            }
        }

        MpirOp::CallIndirect { callee, args } | MpirOp::CallVoidIndirect { callee, args } => {
            f(callee);
            for arg in args {
                f(arg);
            }
        }

        MpirOp::Phi { incomings, .. } => {
            for (_, v) in incomings {
                f(v);
            }
        }

        MpirOp::EnumNew { args, .. } | MpirOp::CallableCapture { captures: args, .. } => {
            for (_, v) in args {
                f(v);
            }
        }

        MpirOp::ArrSet { arr, idx, val } => {
            f(arr);
            f(idx);
            f(val);
        }
        MpirOp::ArrPush { arr, val } => {
            f(arr);
            f(val);
        }
        MpirOp::ArrSlice { arr, start, end } => {
            f(arr);
            f(start);
            f(end);
        }
        MpirOp::ArrMap { arr, func }
        | MpirOp::ArrFilter { arr, func }
        | MpirOp::ArrForeach { arr, func } => {
            f(arr);
            f(func);
        }
        MpirOp::ArrReduce { arr, init, func } => {
            f(arr);
            f(init);
            f(func);
        }

        MpirOp::MapSet { map, key, val } => {
            f(map);
            f(key);
            f(val);
        }

        MpirOp::StrSlice { s, start, end } => {
            f(s);
            f(start);
            f(end);
        }

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
            f(device);
            f(groups);
            f(threads);
            for arg in args {
                f(arg);
            }
        }
    }
}

fn for_each_value_in_void_op(op: &MpirOpVoid, mut f: impl FnMut(&MpirValue)) {
    match op {
        MpirOpVoid::CallVoid { args, .. } => {
            for arg in args {
                f(arg);
            }
        }
        MpirOpVoid::SetField { obj, value, .. } => {
            f(obj);
            f(value);
        }
        MpirOpVoid::ArrSet { arr, idx, val } => {
            f(arr);
            f(idx);
            f(val);
        }
        MpirOpVoid::ArrPush { arr, val } => {
            f(arr);
            f(val);
        }
        MpirOpVoid::ArrSort { arr } => f(arr),
        MpirOpVoid::ArrForeach { arr, func } => {
            f(arr);
            f(func);
        }
        MpirOpVoid::MapSet { map, key, val } => {
            f(map);
            f(key);
            f(val);
        }
        MpirOpVoid::MapDeleteVoid { map, key } => {
            f(map);
            f(key);
        }
        MpirOpVoid::StrBuilderAppendStr { b, s } => {
            f(b);
            f(s);
        }
        MpirOpVoid::StrBuilderAppendI64 { b, v }
        | MpirOpVoid::StrBuilderAppendI32 { b, v }
        | MpirOpVoid::StrBuilderAppendF64 { b, v }
        | MpirOpVoid::StrBuilderAppendBool { b, v } => {
            f(b);
            f(v);
        }
        MpirOpVoid::PtrStore { p, v, .. } => {
            f(p);
            f(v);
        }
        MpirOpVoid::Panic { msg } => f(msg),
        MpirOpVoid::GpuBarrier => {}
        MpirOpVoid::GpuBufferStore { buf, idx, val } => {
            f(buf);
            f(idx);
            f(val);
        }
        MpirOpVoid::ArcRetain { v }
        | MpirOpVoid::ArcRelease { v }
        | MpirOpVoid::ArcRetainWeak { v }
        | MpirOpVoid::ArcReleaseWeak { v } => {
            f(v);
        }
    }
}

fn for_each_value_in_terminator(term: &MpirTerminator, mut f: impl FnMut(&MpirValue)) {
    match term {
        MpirTerminator::Ret(Some(v)) => f(v),
        MpirTerminator::Ret(None) | MpirTerminator::Br(_) | MpirTerminator::Unreachable => {}
        MpirTerminator::Cbr { cond, .. } => f(cond),
        MpirTerminator::Switch { val, .. } => f(val),
    }
}

fn emit_error(diag: &mut DiagnosticBag, code: &str, message: &str) {
    diag.emit(Diagnostic {
        code: code.to_string(),
        severity: Severity::Error,
        title: message.to_string(),
        primary_span: None,
        secondary_spans: vec![],
        message: message.to_string(),
        explanation_md: None,
        why: None,
        suggested_fixes: vec![],
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use magpie_types::{BlockId, Sid};

    #[test]
    fn test_arc_insertion_on_clone_shared() {
        let mut type_ctx = TypeCtx::new();
        let shared_ty = type_ctx.intern(TypeKind::HeapHandle {
            hk: HandleKind::Shared,
            base: HeapBase::BuiltinStr,
        });

        let mut module = MpirModule {
            sid: Sid("M:ARCINSERT00".to_string()),
            path: "test.arc".to_string(),
            type_table: magpie_mpir::MpirTypeTable {
                types: vec![(
                    shared_ty,
                    TypeKind::HeapHandle {
                        hk: HandleKind::Shared,
                        base: HeapBase::BuiltinStr,
                    },
                )],
            },
            functions: vec![MpirFn {
                sid: Sid("F:ARCINSERT00".to_string()),
                name: "clone_shared".to_string(),
                params: vec![(LocalId(0), shared_ty)],
                ret_ty: shared_ty,
                blocks: vec![MpirBlock {
                    id: BlockId(0),
                    instrs: vec![MpirInstr {
                        dst: LocalId(1),
                        ty: shared_ty,
                        op: MpirOp::CloneShared {
                            v: MpirValue::Local(LocalId(0)),
                        },
                    }],
                    void_ops: vec![],
                    terminator: MpirTerminator::Ret(Some(MpirValue::Local(LocalId(1)))),
                }],
                locals: vec![MpirLocalDecl {
                    id: LocalId(1),
                    ty: shared_ty,
                    name: "tmp".to_string(),
                }],
                is_async: false,
            }],
            globals: vec![],
        };

        let mut diag = DiagnosticBag::new(16);
        let _ = insert_arc_ops(&mut module, &type_ctx, &mut diag);

        let void_ops = &module.functions[0].blocks[0].void_ops;
        assert!(
            void_ops.iter().any(|op| matches!(
                op,
                MpirOpVoid::ArcRetain {
                    v: MpirValue::Local(LocalId(0))
                }
            )),
            "expected ArcRetain(Local(0)), got: {:?}",
            void_ops
        );
    }

    #[test]
    fn test_arc_optimize_pair_elimination() {
        let type_ctx = TypeCtx::new();
        let mut module = MpirModule {
            sid: Sid("M:ARCOPTIM000".to_string()),
            path: "test.arc".to_string(),
            type_table: magpie_mpir::MpirTypeTable { types: vec![] },
            functions: vec![MpirFn {
                sid: Sid("F:ARCOPTIM000".to_string()),
                name: "opt".to_string(),
                params: vec![],
                ret_ty: TypeId(0),
                blocks: vec![MpirBlock {
                    id: BlockId(0),
                    instrs: vec![],
                    void_ops: vec![
                        MpirOpVoid::ArcRetain {
                            v: MpirValue::Local(LocalId(0)),
                        },
                        MpirOpVoid::ArcRelease {
                            v: MpirValue::Local(LocalId(0)),
                        },
                    ],
                    terminator: MpirTerminator::Ret(None),
                }],
                locals: vec![],
                is_async: false,
            }],
            globals: vec![],
        };

        optimize_arc(&mut module, &type_ctx);
        assert!(
            module.functions[0].blocks[0].void_ops.is_empty(),
            "expected retain/release pair to be removed"
        );
    }
}
