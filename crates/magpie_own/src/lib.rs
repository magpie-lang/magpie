//! Ownership and borrow checking for HIR (§10).

use magpie_diag::{codes, Diagnostic, DiagnosticBag, Severity};
use magpie_hir::{
    HirBlock, HirFunction, HirInstr, HirModule, HirOp, HirOpVoid, HirTerminator, HirValue,
};
use magpie_types::{BlockId, HandleKind, LocalId, TypeCtx, TypeId, TypeKind};
use std::collections::{HashMap, HashSet};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum BorrowFlavor {
    Shared,
    Mut,
}

#[derive(Copy, Clone, Debug)]
struct BorrowTrack {
    owner: LocalId,
    flavor: BorrowFlavor,
    release_at: usize,
}

#[derive(Debug, Default)]
struct MovedAnalysis {
    moved_in: Vec<HashSet<LocalId>>,
    moved_out: Vec<HashSet<LocalId>>,
    edge_phi_consumes: HashMap<(usize, usize), HashSet<LocalId>>,
}

pub fn check_ownership(
    module: &HirModule,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) -> Result<(), ()> {
    let before = diag.error_count();

    for global in &module.globals {
        if is_borrow_type(global.ty, type_ctx) || is_borrow_type(global.init.ty, type_ctx) {
            emit_error(
                diag,
                codes::MPO0003,
                &format!(
                    "global '{}' stores/declares a borrow, which escapes scope (MPO0003)",
                    global.name
                ),
            );
        }
    }

    for func in &module.functions {
        check_function(func, type_ctx, diag);
    }

    if diag.error_count() > before {
        Err(())
    } else {
        Ok(())
    }
}

pub fn is_move_only(type_id: TypeId, type_ctx: &TypeCtx) -> bool {
    fn go(ty: TypeId, type_ctx: &TypeCtx, visiting: &mut HashSet<TypeId>) -> bool {
        if !visiting.insert(ty) {
            return false;
        }

        let out = match type_ctx.lookup(ty) {
            Some(TypeKind::Prim(_)) => false,
            Some(TypeKind::RawPtr { .. }) => false,
            Some(TypeKind::HeapHandle { .. }) => true,
            Some(TypeKind::BuiltinOption { inner }) => go(*inner, type_ctx, visiting),
            Some(TypeKind::BuiltinResult { ok, err }) => {
                go(*ok, type_ctx, visiting) || go(*err, type_ctx, visiting)
            }
            Some(TypeKind::Arr { elem, .. }) | Some(TypeKind::Vec { elem, .. }) => {
                go(*elem, type_ctx, visiting)
            }
            Some(TypeKind::Tuple { elems }) => elems.iter().any(|e| go(*e, type_ctx, visiting)),
            // TypeCtx currently stores only SID for value structs; in absence of a field table,
            // treat as move-only conservatively.
            Some(TypeKind::ValueStruct { .. }) => true,
            None => true,
        };

        visiting.remove(&ty);
        out
    }

    go(type_id, type_ctx, &mut HashSet::new())
}

fn check_function(func: &HirFunction, type_ctx: &TypeCtx, diag: &mut DiagnosticBag) {
    let local_types = collect_local_types(func);
    let move_only_locals: HashSet<LocalId> = local_types
        .iter()
        .filter_map(|(l, ty)| if is_move_only(*ty, type_ctx) { Some(*l) } else { None })
        .collect();

    let block_index = build_block_index(func);
    let successors: Vec<Vec<usize>> = func
        .blocks
        .iter()
        .map(|b| block_successors(b, &block_index))
        .collect();
    let preds = build_predecessors(successors.len(), &successors);

    let moved = analyze_moved_sets(func, &move_only_locals, &block_index, &preds);
    let def_block = collect_def_blocks(func, &block_index);

    for (blk_idx, block) in func.blocks.iter().enumerate() {
        for instr in &block.instrs {
            if let HirOp::Phi { ty, incomings } = &instr.op {
                if is_borrow_type(*ty, type_ctx) {
                    emit_error(
                        diag,
                        codes::MPO0102,
                        &format!(
                            "fn '{}' block {}: phi result is a borrow type (MPO0102)",
                            func.name, block.id.0
                        ),
                    );
                }

                for (pred_bid, v) in incomings {
                    if is_borrow_value(v, &local_types, type_ctx) {
                        emit_error(
                            diag,
                            codes::MPO0102,
                            &format!(
                                "fn '{}' block {}: borrow value appears in phi incoming from block {} (MPO0102)",
                                func.name, block.id.0, pred_bid.0
                            ),
                        );
                    }

                    let Some(local) = as_local(v) else {
                        continue;
                    };

                    if !move_only_locals.contains(&local) {
                        continue;
                    }

                    if let Some(&pred_idx) = block_index.get(&pred_bid.0) {
                        if moved.moved_out[pred_idx].contains(&local) {
                            emit_error(
                                diag,
                                "MPO0007",
                                &format!(
                                    "fn '{}' block {}: use of moved value %{} in phi incoming from block {}",
                                    func.name, block.id.0, local.0, pred_bid.0
                                ),
                            );
                        }
                    }
                }
            }
        }

        let last_use = block_last_use(block);
        let mut active_borrows: HashMap<LocalId, BorrowTrack> = HashMap::new();
        let mut shared_count: HashMap<LocalId, u32> = HashMap::new();
        let mut mut_active: HashSet<LocalId> = HashSet::new();
        let mut moved_now = moved.moved_in[blk_idx].clone();

        let mut index = 0usize;
        for instr in &block.instrs {
            if matches!(instr.op, HirOp::Phi { .. }) {
                continue;
            }

            check_cross_block_borrow_uses(
                func,
                blk_idx,
                op_used_locals(&instr.op),
                &local_types,
                &def_block,
                type_ctx,
                diag,
            );

            check_use_after_move(
                func,
                block,
                op_used_locals(&instr.op),
                &move_only_locals,
                &moved_now,
                diag,
            );

            check_store_constraints_instr(instr, &local_types, type_ctx, diag);
            check_collection_constraints_instr(instr, &local_types, type_ctx, diag);

            if let Some((owner, flavor)) = borrow_creation(&instr.op) {
                if move_only_locals.contains(&owner) {
                    let shared = *shared_count.get(&owner).unwrap_or(&0);
                    let has_mut = mut_active.contains(&owner);

                    let ok = match flavor {
                        BorrowFlavor::Shared => !has_mut,
                        BorrowFlavor::Mut => shared == 0 && !has_mut,
                    };

                    if !ok {
                        emit_error(
                            diag,
                            codes::MPO0011,
                            &format!(
                                "fn '{}' block {}: illegal borrow state for %{} while creating {:?}",
                                func.name, block.id.0, owner.0, flavor
                            ),
                        );
                    } else {
                        match flavor {
                            BorrowFlavor::Shared => {
                                *shared_count.entry(owner).or_insert(0) += 1;
                            }
                            BorrowFlavor::Mut => {
                                mut_active.insert(owner);
                            }
                        }

                        let release_at = last_use.get(&instr.dst).copied().unwrap_or(index);
                        active_borrows.insert(
                            instr.dst,
                            BorrowTrack {
                                owner,
                                flavor,
                                release_at,
                            },
                        );
                    }
                }
            }

            consume_locals(
                func,
                block,
                op_consumed_locals(&instr.op),
                &move_only_locals,
                &mut shared_count,
                &mut mut_active,
                &mut moved_now,
                diag,
            );

            release_finished_borrows(
                index,
                &mut active_borrows,
                &mut shared_count,
                &mut mut_active,
            );
            index += 1;
        }

        for vop in &block.void_ops {
            check_cross_block_borrow_uses(
                func,
                blk_idx,
                op_void_used_locals(vop),
                &local_types,
                &def_block,
                type_ctx,
                diag,
            );

            check_use_after_move(
                func,
                block,
                op_void_used_locals(vop),
                &move_only_locals,
                &moved_now,
                diag,
            );

            check_store_constraints_void(vop, &local_types, type_ctx, diag);
            check_collection_constraints_void(vop, &local_types, type_ctx, diag);

            consume_locals(
                func,
                block,
                op_void_consumed_locals(vop),
                &move_only_locals,
                &mut shared_count,
                &mut mut_active,
                &mut moved_now,
                diag,
            );

            release_finished_borrows(
                index,
                &mut active_borrows,
                &mut shared_count,
                &mut mut_active,
            );
            index += 1;
        }

        check_cross_block_borrow_uses(
            func,
            blk_idx,
            terminator_used_locals(&block.terminator),
            &local_types,
            &def_block,
            type_ctx,
            diag,
        );

        check_use_after_move(
            func,
            block,
            terminator_used_locals(&block.terminator),
            &move_only_locals,
            &moved_now,
            diag,
        );

        consume_locals(
            func,
            block,
            terminator_consumed_locals(&block.terminator),
            &move_only_locals,
            &mut shared_count,
            &mut mut_active,
            &mut moved_now,
            diag,
        );

        release_finished_borrows(
            index,
            &mut active_borrows,
            &mut shared_count,
            &mut mut_active,
        );
    }
}

fn check_cross_block_borrow_uses(
    func: &HirFunction,
    use_block: usize,
    used: Vec<LocalId>,
    local_types: &HashMap<LocalId, TypeId>,
    def_block: &HashMap<LocalId, usize>,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) {
    for local in dedup_locals(used) {
        let Some(ty) = local_types.get(&local).copied() else {
            continue;
        };
        if !is_borrow_type(ty, type_ctx) {
            continue;
        }
        let Some(def) = def_block.get(&local).copied() else {
            continue;
        };
        if def != use_block {
            emit_error(
                diag,
                codes::MPO0101,
                &format!(
                    "fn '{}': borrow %{} crosses block boundary (def block index {}, use block index {}) (MPO0101)",
                    func.name, local.0, def, use_block
                ),
            );
        }
    }
}

fn check_use_after_move(
    func: &HirFunction,
    block: &HirBlock,
    used: Vec<LocalId>,
    move_only_locals: &HashSet<LocalId>,
    moved_now: &HashSet<LocalId>,
    diag: &mut DiagnosticBag,
) {
    for local in dedup_locals(used) {
        if move_only_locals.contains(&local) && moved_now.contains(&local) {
            emit_error(
                diag,
                "MPO0007",
                &format!(
                    "fn '{}' block {}: use of moved value %{}",
                    func.name, block.id.0, local.0
                ),
            );
        }
    }
}

fn consume_locals(
    func: &HirFunction,
    block: &HirBlock,
    consumed: Vec<LocalId>,
    move_only_locals: &HashSet<LocalId>,
    shared_count: &mut HashMap<LocalId, u32>,
    mut_active: &mut HashSet<LocalId>,
    moved_now: &mut HashSet<LocalId>,
    diag: &mut DiagnosticBag,
) {
    for local in dedup_locals(consumed) {
        if !move_only_locals.contains(&local) {
            continue;
        }

        let shared = *shared_count.get(&local).unwrap_or(&0);
        if shared > 0 || mut_active.contains(&local) {
            emit_error(
                diag,
                codes::MPO0011,
                &format!(
                    "fn '{}' block {}: move of %{} while borrowed (MPO0011)",
                    func.name, block.id.0, local.0
                ),
            );
        }

        moved_now.insert(local);
    }
}

fn check_store_constraints_instr(
    instr: &HirInstr,
    local_types: &HashMap<LocalId, TypeId>,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) {
    match &instr.op {
        HirOp::New { fields, .. } => {
            for (_, v) in fields {
                check_stored_value(v, "new field initializer", local_types, type_ctx, diag);
            }
        }
        HirOp::ArrNew { elem_ty, .. } => {
            if is_borrow_type(*elem_ty, type_ctx) {
                emit_error(
                    diag,
                    codes::MPO0003,
                    "arr.new uses borrow element type; borrows cannot be stored in arrays (MPO0003)",
                );
            }
        }
        HirOp::ArrSet { val, .. } | HirOp::ArrPush { val, .. } => {
            check_stored_value(val, "array write", local_types, type_ctx, diag);
        }
        HirOp::MapNew { key_ty, val_ty } => {
            if is_borrow_type(*key_ty, type_ctx) || is_borrow_type(*val_ty, type_ctx) {
                emit_error(
                    diag,
                    codes::MPO0003,
                    "map.new uses borrow key/value type; borrows cannot be stored in maps (MPO0003)",
                );
            }
        }
        HirOp::MapSet { key, val, .. } => {
            check_stored_value(key, "map key write", local_types, type_ctx, diag);
            check_stored_value(val, "map value write", local_types, type_ctx, diag);
        }
        _ => {}
    }
}

fn check_store_constraints_void(
    op: &HirOpVoid,
    local_types: &HashMap<LocalId, TypeId>,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) {
    match op {
        HirOpVoid::SetField { value, .. } => {
            check_stored_value(value, "setfield", local_types, type_ctx, diag);
        }
        HirOpVoid::ArrSet { val, .. } | HirOpVoid::ArrPush { val, .. } => {
            check_stored_value(val, "array write", local_types, type_ctx, diag);
        }
        HirOpVoid::MapSet { key, val, .. } => {
            check_stored_value(key, "map key write", local_types, type_ctx, diag);
            check_stored_value(val, "map value write", local_types, type_ctx, diag);
        }
        _ => {}
    }
}

fn check_stored_value(
    v: &HirValue,
    context: &str,
    local_types: &HashMap<LocalId, TypeId>,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) {
    if let Some(ty) = value_type(v, local_types) {
        if is_borrow_type(ty, type_ctx) {
            emit_error(
                diag,
                codes::MPO0003,
                &format!(
                    "{} stores a borrow value; borrows cannot escape scope (MPO0003)",
                    context
                ),
            );
        }
    }
}

fn check_collection_constraints_instr(
    instr: &HirInstr,
    local_types: &HashMap<LocalId, TypeId>,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) {
    match &instr.op {
        HirOp::ArrSet { arr, .. }
        | HirOp::ArrPush { arr, .. }
        | HirOp::ArrPop { arr }
        | HirOp::ArrSort { arr }
        | HirOp::MapSet { map: arr, .. }
        | HirOp::MapDelete { map: arr, .. }
        | HirOp::MapDeleteVoid { map: arr, .. }
        | HirOp::StrBuilderAppendStr { b: arr, .. }
        | HirOp::StrBuilderAppendI64 { b: arr, .. }
        | HirOp::StrBuilderAppendI32 { b: arr, .. }
        | HirOp::StrBuilderAppendF64 { b: arr, .. }
        | HirOp::StrBuilderAppendBool { b: arr, .. } => {
            check_mutating_target(arr, local_types, type_ctx, diag);
        }

        HirOp::ArrLen { arr }
        | HirOp::ArrGet { arr, .. }
        | HirOp::ArrSlice { arr, .. }
        | HirOp::ArrContains { arr, .. }
        | HirOp::ArrMap { arr, .. }
        | HirOp::ArrFilter { arr, .. }
        | HirOp::ArrReduce { arr, .. }
        | HirOp::ArrForeach { arr, .. }
        | HirOp::MapLen { map: arr }
        | HirOp::MapGet { map: arr, .. }
        | HirOp::MapGetRef { map: arr, .. }
        | HirOp::MapContainsKey { map: arr, .. }
        | HirOp::MapKeys { map: arr }
        | HirOp::MapValues { map: arr }
        | HirOp::StrLen { s: arr }
        | HirOp::StrSlice { s: arr, .. }
        | HirOp::StrBytes { s: arr } => {
            check_read_target(arr, local_types, type_ctx, diag);
        }
        HirOp::StrEq { a, b } => {
            check_read_target(a, local_types, type_ctx, diag);
            check_read_target(b, local_types, type_ctx, diag);
        }
        HirOp::StrBuilderBuild { b } => {
            check_mutating_target(b, local_types, type_ctx, diag);
        }
        _ => {}
    }
}

fn check_collection_constraints_void(
    op: &HirOpVoid,
    local_types: &HashMap<LocalId, TypeId>,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) {
    match op {
        HirOpVoid::SetField { obj, .. }
        | HirOpVoid::ArrSet { arr: obj, .. }
        | HirOpVoid::ArrPush { arr: obj, .. }
        | HirOpVoid::ArrSort { arr: obj }
        | HirOpVoid::MapSet { map: obj, .. }
        | HirOpVoid::MapDeleteVoid { map: obj, .. }
        | HirOpVoid::StrBuilderAppendStr { b: obj, .. }
        | HirOpVoid::StrBuilderAppendI64 { b: obj, .. }
        | HirOpVoid::StrBuilderAppendI32 { b: obj, .. }
        | HirOpVoid::StrBuilderAppendF64 { b: obj, .. }
        | HirOpVoid::StrBuilderAppendBool { b: obj, .. } => {
            check_mutating_target(obj, local_types, type_ctx, diag);
        }
        HirOpVoid::ArrForeach { arr, .. } => {
            check_read_target(arr, local_types, type_ctx, diag);
        }
        _ => {}
    }
}

fn check_mutating_target(
    target: &HirValue,
    local_types: &HashMap<LocalId, TypeId>,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) {
    let Some(ty) = value_type(target, local_types) else {
        return;
    };

    match handle_kind(ty, type_ctx) {
        Some(HandleKind::Unique) | Some(HandleKind::MutBorrow) => {}
        Some(HandleKind::Shared) => emit_error(
            diag,
            codes::MPO0004,
            "mutating intrinsic on shared reference is forbidden (MPO0004)",
        ),
        _ => emit_error(
            diag,
            codes::MPO0004,
            "mutating intrinsic requires unique or mutborrow ownership",
        ),
    }
}

fn check_read_target(
    target: &HirValue,
    local_types: &HashMap<LocalId, TypeId>,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) {
    let Some(ty) = value_type(target, local_types) else {
        return;
    };

    match handle_kind(ty, type_ctx) {
        Some(HandleKind::Borrow) | Some(HandleKind::MutBorrow) => {}
        _ => emit_error(
            diag,
            codes::MPO0004,
            "read intrinsic requires borrow/mutborrow operand",
        ),
    }
}

fn analyze_moved_sets(
    func: &HirFunction,
    move_only_locals: &HashSet<LocalId>,
    block_index: &HashMap<u32, usize>,
    preds: &[Vec<usize>],
) -> MovedAnalysis {
    let n = func.blocks.len();
    let mut analysis = MovedAnalysis {
        moved_in: vec![HashSet::new(); n],
        moved_out: vec![HashSet::new(); n],
        edge_phi_consumes: HashMap::new(),
    };

    for (succ_idx, block) in func.blocks.iter().enumerate() {
        for instr in &block.instrs {
            let HirOp::Phi { incomings, .. } = &instr.op else {
                continue;
            };
            for (pred_bid, v) in incomings {
                let Some(local) = as_local(v) else {
                    continue;
                };
                if !move_only_locals.contains(&local) {
                    continue;
                }
                if let Some(&pred_idx) = block_index.get(&pred_bid.0) {
                    analysis
                        .edge_phi_consumes
                        .entry((pred_idx, succ_idx))
                        .or_default()
                        .insert(local);
                }
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
                    if move_only_locals.contains(&local) {
                        cur.insert(local);
                    }
                }
            }
            for vop in &block.void_ops {
                for local in op_void_consumed_locals(vop) {
                    if move_only_locals.contains(&local) {
                        cur.insert(local);
                    }
                }
            }
            for local in terminator_consumed_locals(&block.terminator) {
                if move_only_locals.contains(&local) {
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

fn release_finished_borrows(
    index: usize,
    active_borrows: &mut HashMap<LocalId, BorrowTrack>,
    shared_count: &mut HashMap<LocalId, u32>,
    mut_active: &mut HashSet<LocalId>,
) {
    let to_release: Vec<LocalId> = active_borrows
        .iter()
        .filter_map(|(borrow_local, track)| {
            if track.release_at == index {
                Some(*borrow_local)
            } else {
                None
            }
        })
        .collect();

    for borrow_local in to_release {
        let Some(track) = active_borrows.remove(&borrow_local) else {
            continue;
        };

        match track.flavor {
            BorrowFlavor::Shared => {
                let entry = shared_count.entry(track.owner).or_insert(0);
                if *entry > 0 {
                    *entry -= 1;
                }
            }
            BorrowFlavor::Mut => {
                mut_active.remove(&track.owner);
            }
        }
    }
}

fn borrow_creation(op: &HirOp) -> Option<(LocalId, BorrowFlavor)> {
    match op {
        HirOp::BorrowShared { v } => as_local(v).map(|l| (l, BorrowFlavor::Shared)),
        HirOp::BorrowMut { v } => as_local(v).map(|l| (l, BorrowFlavor::Mut)),
        _ => None,
    }
}

fn block_last_use(block: &HirBlock) -> HashMap<LocalId, usize> {
    let mut out = HashMap::new();
    let mut idx = 0usize;

    for instr in &block.instrs {
        if matches!(instr.op, HirOp::Phi { .. }) {
            continue;
        }
        for local in op_used_locals(&instr.op) {
            out.insert(local, idx);
        }
        idx += 1;
    }

    for vop in &block.void_ops {
        for local in op_void_used_locals(vop) {
            out.insert(local, idx);
        }
        idx += 1;
    }

    for local in terminator_used_locals(&block.terminator) {
        out.insert(local, idx);
    }

    out
}

fn collect_local_types(func: &HirFunction) -> HashMap<LocalId, TypeId> {
    let mut out = HashMap::new();
    for (local, ty) in &func.params {
        out.insert(*local, *ty);
    }
    for block in &func.blocks {
        for instr in &block.instrs {
            out.insert(instr.dst, instr.ty);
        }
    }
    out
}

fn collect_def_blocks(func: &HirFunction, block_index: &HashMap<u32, usize>) -> HashMap<LocalId, usize> {
    let mut out = HashMap::new();
    for (local, _) in &func.params {
        out.insert(*local, 0);
    }
    for block in &func.blocks {
        if let Some(&idx) = block_index.get(&block.id.0) {
            for instr in &block.instrs {
                out.insert(instr.dst, idx);
            }
        }
    }
    out
}

fn build_block_index(func: &HirFunction) -> HashMap<u32, usize> {
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

fn block_successors(block: &HirBlock, block_index: &HashMap<u32, usize>) -> Vec<usize> {
    match &block.terminator {
        HirTerminator::Ret(_) | HirTerminator::Unreachable => vec![],
        HirTerminator::Br(bid) => block_index.get(&bid.0).copied().into_iter().collect(),
        HirTerminator::Cbr {
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
        HirTerminator::Switch { arms, default, .. } => {
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

fn value_type(v: &HirValue, local_types: &HashMap<LocalId, TypeId>) -> Option<TypeId> {
    match v {
        HirValue::Local(l) => local_types.get(l).copied(),
        HirValue::Const(c) => Some(c.ty),
    }
}

fn as_local(v: &HirValue) -> Option<LocalId> {
    match v {
        HirValue::Local(l) => Some(*l),
        HirValue::Const(_) => None,
    }
}

fn is_borrow_value(v: &HirValue, local_types: &HashMap<LocalId, TypeId>, type_ctx: &TypeCtx) -> bool {
    value_type(v, local_types)
        .map(|ty| is_borrow_type(ty, type_ctx))
        .unwrap_or(false)
}

fn is_borrow_type(ty: TypeId, type_ctx: &TypeCtx) -> bool {
    matches!(
        type_ctx.lookup(ty),
        Some(TypeKind::HeapHandle {
            hk: HandleKind::Borrow | HandleKind::MutBorrow,
            ..
        })
    )
}

fn handle_kind(ty: TypeId, type_ctx: &TypeCtx) -> Option<HandleKind> {
    match type_ctx.lookup(ty) {
        Some(TypeKind::HeapHandle { hk, .. }) => Some(*hk),
        _ => None,
    }
}

fn dedup_locals(locals: Vec<LocalId>) -> Vec<LocalId> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for l in locals {
        if seen.insert(l) {
            out.push(l);
        }
    }
    out
}

fn op_used_locals(op: &HirOp) -> Vec<LocalId> {
    let mut out = Vec::new();
    for_each_value_in_op(op, |v| {
        if let Some(l) = as_local(v) {
            out.push(l);
        }
    });
    out
}

fn op_void_used_locals(op: &HirOpVoid) -> Vec<LocalId> {
    let mut out = Vec::new();
    for_each_value_in_void_op(op, |v| {
        if let Some(l) = as_local(v) {
            out.push(l);
        }
    });
    out
}

fn terminator_used_locals(term: &HirTerminator) -> Vec<LocalId> {
    let mut out = Vec::new();
    for_each_value_in_terminator(term, |v| {
        if let Some(l) = as_local(v) {
            out.push(l);
        }
    });
    out
}

fn op_consumed_locals(op: &HirOp) -> Vec<LocalId> {
    let mut out = Vec::new();
    let push = |v: &HirValue, out: &mut Vec<LocalId>| {
        if let Some(l) = as_local(v) {
            out.push(l);
        }
    };

    match op {
        HirOp::Const(_) => {}
        HirOp::Move { v } => push(v, &mut out),
        HirOp::BorrowShared { .. } | HirOp::BorrowMut { .. } => {}
        HirOp::New { fields, .. } => {
            for (_, v) in fields {
                push(v, &mut out);
            }
        }
        HirOp::GetField { .. }
        | HirOp::IAdd { .. }
        | HirOp::ISub { .. }
        | HirOp::IMul { .. }
        | HirOp::ISDiv { .. }
        | HirOp::IUDiv { .. }
        | HirOp::ISRem { .. }
        | HirOp::IURem { .. }
        | HirOp::IAddWrap { .. }
        | HirOp::ISubWrap { .. }
        | HirOp::IMulWrap { .. }
        | HirOp::IAddChecked { .. }
        | HirOp::ISubChecked { .. }
        | HirOp::IMulChecked { .. }
        | HirOp::IAnd { .. }
        | HirOp::IOr { .. }
        | HirOp::IXor { .. }
        | HirOp::IShl { .. }
        | HirOp::ILshr { .. }
        | HirOp::IAshr { .. }
        | HirOp::ICmp { .. }
        | HirOp::FCmp { .. }
        | HirOp::FAdd { .. }
        | HirOp::FSub { .. }
        | HirOp::FMul { .. }
        | HirOp::FDiv { .. }
        | HirOp::FRem { .. }
        | HirOp::FAddFast { .. }
        | HirOp::FSubFast { .. }
        | HirOp::FMulFast { .. }
        | HirOp::FDivFast { .. }
        | HirOp::Cast { .. }
        | HirOp::PtrNull { .. }
        | HirOp::PtrAddr { .. }
        | HirOp::PtrFromAddr { .. }
        | HirOp::PtrAdd { .. }
        | HirOp::PtrLoad { .. }
        | HirOp::SuspendAwait { .. }
        | HirOp::CloneShared { .. }
        | HirOp::CloneWeak { .. }
        | HirOp::WeakDowngrade { .. }
        | HirOp::WeakUpgrade { .. }
        | HirOp::EnumTag { .. }
        | HirOp::EnumPayload { .. }
        | HirOp::EnumIs { .. }
        | HirOp::ArrNew { .. }
        | HirOp::ArrLen { .. }
        | HirOp::ArrGet { .. }
        | HirOp::ArrPop { .. }
        | HirOp::ArrSlice { .. }
        | HirOp::ArrContains { .. }
        | HirOp::ArrSort { .. }
        | HirOp::ArrMap { .. }
        | HirOp::ArrFilter { .. }
        | HirOp::ArrForeach { .. }
        | HirOp::MapNew { .. }
        | HirOp::MapLen { .. }
        | HirOp::MapGet { .. }
        | HirOp::MapGetRef { .. }
        | HirOp::MapDelete { .. }
        | HirOp::MapContainsKey { .. }
        | HirOp::MapDeleteVoid { .. }
        | HirOp::MapKeys { .. }
        | HirOp::MapValues { .. }
        | HirOp::StrLen { .. }
        | HirOp::StrEq { .. }
        | HirOp::StrSlice { .. }
        | HirOp::StrBytes { .. }
        | HirOp::StrBuilderNew
        | HirOp::StrBuilderAppendStr { .. }
        | HirOp::StrBuilderAppendI64 { .. }
        | HirOp::StrBuilderAppendI32 { .. }
        | HirOp::StrBuilderAppendF64 { .. }
        | HirOp::StrBuilderAppendBool { .. }
        | HirOp::StrParseI64 { .. }
        | HirOp::StrParseU64 { .. }
        | HirOp::StrParseF64 { .. }
        | HirOp::StrParseBool { .. }
        | HirOp::JsonEncode { .. }
        | HirOp::JsonDecode { .. }
        | HirOp::GpuThreadId
        | HirOp::GpuWorkgroupId
        | HirOp::GpuWorkgroupSize
        | HirOp::GpuGlobalId
        | HirOp::GpuBufferLoad { .. }
        | HirOp::GpuBufferLen { .. }
        | HirOp::GpuShared { .. }
        | HirOp::Panic { .. }
        | HirOp::Phi { .. } => {}

        HirOp::PtrStore { v, .. } => push(v, &mut out),
        HirOp::Call { args, .. } | HirOp::SuspendCall { args, .. } => {
            for v in args {
                push(v, &mut out);
            }
        }
        HirOp::CallIndirect { args, .. } | HirOp::CallVoidIndirect { args, .. } => {
            for v in args {
                push(v, &mut out);
            }
        }
        HirOp::Share { v } => push(v, &mut out),
        HirOp::EnumNew { args, .. } => {
            for (_, v) in args {
                push(v, &mut out);
            }
        }
        HirOp::CallableCapture { captures, .. } => {
            for (_, v) in captures {
                push(v, &mut out);
            }
        }
        HirOp::ArrSet { val, .. } | HirOp::ArrPush { val, .. } => push(val, &mut out),
        HirOp::ArrReduce { init, .. } => push(init, &mut out),
        HirOp::MapSet { key, val, .. } => {
            push(key, &mut out);
            push(val, &mut out);
        }
        HirOp::StrConcat { a, b } => {
            push(a, &mut out);
            push(b, &mut out);
        }
        HirOp::StrBuilderBuild { b } => push(b, &mut out),
        HirOp::GpuLaunch { args, .. } | HirOp::GpuLaunchAsync { args, .. } => {
            for v in args {
                push(v, &mut out);
            }
        }
    }

    out
}

fn op_void_consumed_locals(op: &HirOpVoid) -> Vec<LocalId> {
    let mut out = Vec::new();
    let push = |v: &HirValue, out: &mut Vec<LocalId>| {
        if let Some(l) = as_local(v) {
            out.push(l);
        }
    };

    match op {
        HirOpVoid::CallVoid { args, .. } => {
            for v in args {
                push(v, &mut out);
            }
        }
        HirOpVoid::SetField { value, .. } => push(value, &mut out),
        HirOpVoid::ArrSet { val, .. } | HirOpVoid::ArrPush { val, .. } => push(val, &mut out),
        HirOpVoid::MapSet { key, val, .. } => {
            push(key, &mut out);
            push(val, &mut out);
        }
        HirOpVoid::PtrStore { v, .. } => push(v, &mut out),
        HirOpVoid::GpuBufferStore { val, .. } => push(val, &mut out),

        HirOpVoid::ArrSort { .. }
        | HirOpVoid::ArrForeach { .. }
        | HirOpVoid::MapDeleteVoid { .. }
        | HirOpVoid::StrBuilderAppendStr { .. }
        | HirOpVoid::StrBuilderAppendI64 { .. }
        | HirOpVoid::StrBuilderAppendI32 { .. }
        | HirOpVoid::StrBuilderAppendF64 { .. }
        | HirOpVoid::StrBuilderAppendBool { .. }
        | HirOpVoid::Panic { .. }
        | HirOpVoid::GpuBarrier => {}
    }

    out
}

fn terminator_consumed_locals(term: &HirTerminator) -> Vec<LocalId> {
    match term {
        HirTerminator::Ret(Some(v)) => as_local(v).into_iter().collect(),
        HirTerminator::Ret(None)
        | HirTerminator::Br(_)
        | HirTerminator::Cbr { .. }
        | HirTerminator::Switch { .. }
        | HirTerminator::Unreachable => vec![],
    }
}

fn for_each_value_in_op(op: &HirOp, mut f: impl FnMut(&HirValue)) {
    match op {
        HirOp::Const(_) => {}
        HirOp::Move { v }
        | HirOp::BorrowShared { v }
        | HirOp::BorrowMut { v }
        | HirOp::Cast { v, .. }
        | HirOp::PtrAddr { p: v }
        | HirOp::PtrFromAddr { addr: v, .. }
        | HirOp::PtrLoad { p: v, .. }
        | HirOp::Share { v }
        | HirOp::CloneShared { v }
        | HirOp::CloneWeak { v }
        | HirOp::WeakDowngrade { v }
        | HirOp::WeakUpgrade { v }
        | HirOp::EnumTag { v }
        | HirOp::EnumPayload { v, .. }
        | HirOp::EnumIs { v, .. }
        | HirOp::ArrNew { cap: v, .. }
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
        | HirOp::SuspendAwait { fut: v }
        | HirOp::JsonEncode { v, .. }
        | HirOp::JsonDecode { s: v, .. }
        | HirOp::GpuBufferLen { buf: v }
        | HirOp::GpuShared { size: v, .. }
        | HirOp::Panic { msg: v }
        | HirOp::GetField { obj: v, .. } => f(v),

        HirOp::New { fields, .. } => {
            for (_, v) in fields {
                f(v);
            }
        }
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
            f(lhs);
            f(rhs);
        }
        HirOp::PtrNull { .. }
        | HirOp::MapNew { .. }
        | HirOp::StrBuilderNew
        | HirOp::GpuThreadId
        | HirOp::GpuWorkgroupId
        | HirOp::GpuWorkgroupSize
        | HirOp::GpuGlobalId => {}

        HirOp::PtrAdd { p, count }
        | HirOp::ArrGet { arr: p, idx: count }
        | HirOp::ArrContains { arr: p, val: count }
        | HirOp::MapGet { map: p, key: count }
        | HirOp::MapGetRef { map: p, key: count }
        | HirOp::MapDelete { map: p, key: count }
        | HirOp::MapContainsKey { map: p, key: count }
        | HirOp::MapDeleteVoid { map: p, key: count }
        | HirOp::StrBuilderAppendStr { b: p, s: count }
        | HirOp::StrBuilderAppendI64 { b: p, v: count }
        | HirOp::StrBuilderAppendI32 { b: p, v: count }
        | HirOp::StrBuilderAppendF64 { b: p, v: count }
        | HirOp::StrBuilderAppendBool { b: p, v: count }
        | HirOp::GpuBufferLoad { buf: p, idx: count }
        | HirOp::PtrStore { p, v: count, .. } => {
            f(p);
            f(count);
        }

        HirOp::Call { args, .. } | HirOp::SuspendCall { args, .. } => {
            for arg in args {
                f(arg);
            }
        }

        HirOp::CallIndirect { callee, args } | HirOp::CallVoidIndirect { callee, args } => {
            f(callee);
            for arg in args {
                f(arg);
            }
        }

        HirOp::Phi { incomings, .. } => {
            for (_, v) in incomings {
                f(v);
            }
        }

        HirOp::EnumNew { args, .. } | HirOp::CallableCapture { captures: args, .. } => {
            for (_, v) in args {
                f(v);
            }
        }

        HirOp::ArrSet { arr, idx, val } => {
            f(arr);
            f(idx);
            f(val);
        }
        HirOp::ArrPush { arr, val } => {
            f(arr);
            f(val);
        }
        HirOp::ArrSlice { arr, start, end } => {
            f(arr);
            f(start);
            f(end);
        }
        HirOp::ArrMap { arr, func }
        | HirOp::ArrFilter { arr, func }
        | HirOp::ArrForeach { arr, func } => {
            f(arr);
            f(func);
        }
        HirOp::ArrReduce { arr, init, func } => {
            f(arr);
            f(init);
            f(func);
        }

        HirOp::MapSet { map, key, val } => {
            f(map);
            f(key);
            f(val);
        }

        HirOp::StrSlice { s, start, end } => {
            f(s);
            f(start);
            f(end);
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
            f(device);
            f(groups);
            f(threads);
            for arg in args {
                f(arg);
            }
        }
    }
}

fn for_each_value_in_void_op(op: &HirOpVoid, mut f: impl FnMut(&HirValue)) {
    match op {
        HirOpVoid::CallVoid { args, .. } => {
            for arg in args {
                f(arg);
            }
        }
        HirOpVoid::SetField { obj, value, .. } => {
            f(obj);
            f(value);
        }
        HirOpVoid::ArrSet { arr, idx, val } => {
            f(arr);
            f(idx);
            f(val);
        }
        HirOpVoid::ArrPush { arr, val } => {
            f(arr);
            f(val);
        }
        HirOpVoid::ArrSort { arr } => f(arr),
        HirOpVoid::ArrForeach { arr, func } => {
            f(arr);
            f(func);
        }
        HirOpVoid::MapSet { map, key, val } => {
            f(map);
            f(key);
            f(val);
        }
        HirOpVoid::MapDeleteVoid { map, key } => {
            f(map);
            f(key);
        }
        HirOpVoid::StrBuilderAppendStr { b, s } => {
            f(b);
            f(s);
        }
        HirOpVoid::StrBuilderAppendI64 { b, v }
        | HirOpVoid::StrBuilderAppendI32 { b, v }
        | HirOpVoid::StrBuilderAppendF64 { b, v }
        | HirOpVoid::StrBuilderAppendBool { b, v } => {
            f(b);
            f(v);
        }
        HirOpVoid::PtrStore { p, v, .. } => {
            f(p);
            f(v);
        }
        HirOpVoid::Panic { msg } => f(msg),
        HirOpVoid::GpuBarrier => {}
        HirOpVoid::GpuBufferStore { buf, idx, val } => {
            f(buf);
            f(idx);
            f(val);
        }
    }
}

fn for_each_value_in_terminator(term: &HirTerminator, mut f: impl FnMut(&HirValue)) {
    match term {
        HirTerminator::Ret(Some(v)) => f(v),
        HirTerminator::Ret(None)
        | HirTerminator::Br(_)
        | HirTerminator::Unreachable => {}
        HirTerminator::Cbr { cond, .. } => f(cond),
        HirTerminator::Switch { val, .. } => f(val),
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

#[allow(dead_code)]
fn _block_id_to_usize(b: BlockId) -> usize {
    b.0 as usize
}
