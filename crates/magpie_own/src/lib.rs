//! Ownership and borrow checking for HIR (§10).

use magpie_diag::{codes, Diagnostic, DiagnosticBag, Severity, WhyEvent, WhyTrace};
use magpie_hir::{
    HirBlock, HirFunction, HirInstr, HirModule, HirOp, HirOpVoid, HirTerminator, HirTypeDecl,
    HirValue,
};
use magpie_types::{BlockId, HandleKind, HeapBase, LocalId, TypeCtx, TypeId, TypeKind};
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
    let fn_param_types = collect_fn_param_types(module);
    let struct_fields = collect_struct_fields(module);

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
        check_function(func, type_ctx, &fn_param_types, &struct_fields, diag);
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

fn check_function(
    func: &HirFunction,
    type_ctx: &TypeCtx,
    fn_param_types: &HashMap<String, Vec<TypeId>>,
    struct_fields: &HashMap<String, HashMap<String, TypeId>>,
    diag: &mut DiagnosticBag,
) {
    let local_types = collect_local_types(func);
    let callable_captures = collect_callable_capture_types(func, &local_types);
    let move_only_locals: HashSet<LocalId> = local_types
        .iter()
        .filter_map(|(l, ty)| {
            if is_move_only(*ty, type_ctx) {
                Some(*l)
            } else {
                None
            }
        })
        .collect();

    let block_index = build_block_index(func);
    let successors: Vec<Vec<usize>> = func
        .blocks
        .iter()
        .map(|b| block_successors(b, &block_index))
        .collect();
    let preds = build_predecessors(successors.len(), &successors);

    let moved = analyze_moved_sets(
        func,
        &move_only_locals,
        &block_index,
        &preds,
        &local_types,
        type_ctx,
        fn_param_types,
    );
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
            check_projection_constraints_instr(
                func,
                instr,
                &local_types,
                type_ctx,
                struct_fields,
                diag,
            );
            check_call_argument_modes_instr(
                func,
                &instr.op,
                &local_types,
                type_ctx,
                fn_param_types,
                diag,
            );
            check_spawn_send_constraints_instr(
                func,
                &instr.op,
                &local_types,
                &callable_captures,
                type_ctx,
                diag,
            );

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
                op_consumed_locals(&instr.op, &local_types, type_ctx, fn_param_types),
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
            check_call_argument_modes_void(
                func,
                vop,
                &local_types,
                type_ctx,
                fn_param_types,
                diag,
            );
            check_spawn_send_constraints_void(
                func,
                vop,
                &local_types,
                &callable_captures,
                type_ctx,
                diag,
            );

            consume_locals(
                func,
                block,
                op_void_consumed_locals(vop, &local_types, type_ctx, fn_param_types),
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

fn collect_fn_param_types(module: &HirModule) -> HashMap<String, Vec<TypeId>> {
    module
        .functions
        .iter()
        .map(|f| {
            (
                f.sid.0.clone(),
                f.params.iter().map(|(_, ty)| *ty).collect::<Vec<_>>(),
            )
        })
        .collect()
}

fn collect_struct_fields(module: &HirModule) -> HashMap<String, HashMap<String, TypeId>> {
    let mut out = HashMap::new();
    for decl in &module.type_decls {
        if let HirTypeDecl::Struct { sid, fields, .. } = decl {
            out.insert(sid.0.clone(), fields.iter().cloned().collect());
        }
    }
    out
}

fn collect_callable_capture_types(
    func: &HirFunction,
    local_types: &HashMap<LocalId, TypeId>,
) -> HashMap<LocalId, Vec<TypeId>> {
    let mut captures = HashMap::new();
    for block in &func.blocks {
        for instr in &block.instrs {
            if let HirOp::CallableCapture {
                captures: cap_values, ..
            } = &instr.op
            {
                let captured_types = cap_values
                    .iter()
                    .filter_map(|(_, value)| value_type(value, local_types))
                    .collect::<Vec<_>>();
                captures.insert(instr.dst, captured_types);
            }
        }
    }
    captures
}

fn check_spawn_send_constraints_instr(
    func: &HirFunction,
    op: &HirOp,
    local_types: &HashMap<LocalId, TypeId>,
    callable_captures: &HashMap<LocalId, Vec<TypeId>>,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) {
    match op {
        HirOp::Call {
            callee_sid, args, ..
        }
        | HirOp::SuspendCall {
            callee_sid, args, ..
        } => check_spawn_send_constraints(
            func,
            &callee_sid.0,
            args,
            local_types,
            callable_captures,
            type_ctx,
            diag,
        ),
        _ => {}
    }
}

fn check_spawn_send_constraints_void(
    func: &HirFunction,
    op: &HirOpVoid,
    local_types: &HashMap<LocalId, TypeId>,
    callable_captures: &HashMap<LocalId, Vec<TypeId>>,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) {
    if let HirOpVoid::CallVoid {
        callee_sid, args, ..
    } = op
    {
        check_spawn_send_constraints(
            func,
            &callee_sid.0,
            args,
            local_types,
            callable_captures,
            type_ctx,
            diag,
        );
    }
}

fn check_spawn_send_constraints(
    func: &HirFunction,
    callee_sid: &str,
    args: &[HirValue],
    local_types: &HashMap<LocalId, TypeId>,
    callable_captures: &HashMap<LocalId, Vec<TypeId>>,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) {
    if !is_spawn_callee(callee_sid) {
        return;
    }
    let Some(HirValue::Local(callable_local)) = args.first() else {
        emit_error(
            diag,
            "MPO0201",
            &format!(
                "fn '{}': spawn-like call '{}' requires first argument TCallable",
                func.name, callee_sid
            ),
        );
        return;
    };
    let Some(callable_ty) = local_types.get(callable_local).copied() else {
        return;
    };
    if !is_callable_type(callable_ty, type_ctx) {
        emit_error(
            diag,
            "MPO0201",
            &format!(
                "fn '{}': spawn-like call '{}' requires first argument TCallable, got {}",
                func.name,
                callee_sid,
                type_ctx.type_str(callable_ty)
            ),
        );
        return;
    }
    let Some(captured_types) = callable_captures.get(callable_local) else {
        return;
    };
    for captured_ty in captured_types {
        if !is_send_type(*captured_ty, type_ctx, &mut HashSet::new()) {
            emit_error(
                diag,
                "MPO0201",
                &format!(
                    "fn '{}': spawn-like callable captures non-send value of type {}",
                    func.name,
                    type_ctx.type_str(*captured_ty)
                ),
            );
        }
    }
}

fn is_spawn_callee(callee_sid: &str) -> bool {
    let lower = callee_sid.to_ascii_lowercase();
    lower.contains("spawn")
}

fn is_callable_type(type_id: TypeId, type_ctx: &TypeCtx) -> bool {
    matches!(
        type_ctx.lookup(type_id),
        Some(TypeKind::HeapHandle {
            base: HeapBase::Callable { .. },
            ..
        })
    )
}

fn is_send_type(type_id: TypeId, type_ctx: &TypeCtx, visiting: &mut HashSet<TypeId>) -> bool {
    if !visiting.insert(type_id) {
        return true;
    }

    let send = match type_ctx.lookup(type_id) {
        Some(TypeKind::Prim(_)) => true,
        Some(TypeKind::RawPtr { .. }) => false,
        Some(TypeKind::HeapHandle { hk, base }) => match hk {
            HandleKind::Borrow | HandleKind::MutBorrow | HandleKind::Weak => false,
            HandleKind::Unique | HandleKind::Shared => match base {
                HeapBase::BuiltinStr => true,
                HeapBase::BuiltinArray { elem } => is_send_type(*elem, type_ctx, visiting),
                HeapBase::BuiltinMap { key, val } => {
                    is_send_type(*key, type_ctx, visiting) && is_send_type(*val, type_ctx, visiting)
                }
                HeapBase::BuiltinFuture { result } => is_send_type(*result, type_ctx, visiting),
                HeapBase::BuiltinChannelSend { elem } | HeapBase::BuiltinChannelRecv { elem } => {
                    is_send_type(*elem, type_ctx, visiting)
                }
                HeapBase::Callable { .. } => true,
                HeapBase::BuiltinMutex { inner } | HeapBase::BuiltinRwLock { inner } => {
                    is_send_type(*inner, type_ctx, visiting)
                }
                HeapBase::BuiltinStrBuilder | HeapBase::BuiltinCell { .. } => false,
                HeapBase::UserType { .. } => false,
            },
        },
        Some(TypeKind::BuiltinOption { inner }) => is_send_type(*inner, type_ctx, visiting),
        Some(TypeKind::BuiltinResult { ok, err }) => {
            is_send_type(*ok, type_ctx, visiting) && is_send_type(*err, type_ctx, visiting)
        }
        Some(TypeKind::Arr { elem, .. }) | Some(TypeKind::Vec { elem, .. }) => {
            is_send_type(*elem, type_ctx, visiting)
        }
        Some(TypeKind::Tuple { elems }) => elems
            .iter()
            .all(|elem| is_send_type(*elem, type_ctx, visiting)),
        Some(TypeKind::ValueStruct { .. }) => false,
        None => false,
    };

    visiting.remove(&type_id);
    send
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum ProjectionOwnerMode {
    Shared,
    Mut,
}

fn check_projection_constraints_instr(
    func: &HirFunction,
    instr: &HirInstr,
    local_types: &HashMap<LocalId, TypeId>,
    type_ctx: &TypeCtx,
    struct_fields: &HashMap<String, HashMap<String, TypeId>>,
    diag: &mut DiagnosticBag,
) {
    match &instr.op {
        HirOp::GetField { obj, field } => {
            let Some(obj_ty) = value_type(obj, local_types) else {
                return;
            };
            let Some((field_ty, owner_mode)) =
                projected_field_type(obj_ty, field, type_ctx, struct_fields)
            else {
                emit_error(
                    diag,
                    codes::MPO0004,
                    &format!(
                        "fn '{}': getfield requires borrow/mutborrow struct operand with known field '{}'",
                        func.name, field
                    ),
                );
                return;
            };
            check_projection_result_type(
                func,
                "getfield",
                instr.ty,
                field_ty,
                owner_mode,
                type_ctx,
                diag,
            );
        }
        HirOp::ArrGet { arr, .. } => {
            let Some(arr_ty) = value_type(arr, local_types) else {
                return;
            };
            let Some((elem_ty, owner_mode)) = projected_array_elem_type(arr_ty, type_ctx) else {
                emit_error(
                    diag,
                    codes::MPO0004,
                    &format!(
                        "fn '{}': arr.get requires borrow/mutborrow Array<T> operand",
                        func.name
                    ),
                );
                return;
            };
            check_projection_result_type(
                func,
                "arr.get",
                instr.ty,
                elem_ty,
                owner_mode,
                type_ctx,
                diag,
            );
        }
        HirOp::MapGetRef { map, .. } => {
            let Some(map_ty) = value_type(map, local_types) else {
                return;
            };
            let Some((val_ty, owner_mode)) = projected_map_val_type(map_ty, type_ctx) else {
                emit_error(
                    diag,
                    codes::MPO0004,
                    &format!(
                        "fn '{}': map.get_ref requires borrow/mutborrow Map<K,V> operand",
                        func.name
                    ),
                );
                return;
            };
            check_projection_result_type(
                func,
                "map.get_ref",
                instr.ty,
                val_ty,
                owner_mode,
                type_ctx,
                diag,
            );
        }
        HirOp::MapGet { map, .. } => {
            let Some(map_ty) = value_type(map, local_types) else {
                return;
            };
            let Some((val_ty, _)) = projected_map_val_type(map_ty, type_ctx) else {
                emit_error(
                    diag,
                    codes::MPO0004,
                    &format!(
                        "fn '{}': map.get requires borrow/mutborrow Map<K,V> operand",
                        func.name
                    ),
                );
                return;
            };

            if !is_dupable_type(val_ty, type_ctx) {
                emit_error(
                    diag,
                    codes::MPO0103,
                    &format!(
                        "fn '{}': map.get requires Dupable map value type; got {} (MPO0103)",
                        func.name,
                        type_ctx.type_str(val_ty)
                    ),
                );
                return;
            }

            if !is_option_of(instr.ty, val_ty, type_ctx) {
                emit_error(
                    diag,
                    codes::MPO0004,
                    &format!(
                        "fn '{}': map.get result type must be TOption<{}>",
                        func.name,
                        type_ctx.type_str(val_ty)
                    ),
                );
            }
        }
        _ => {}
    }
}

fn projected_field_type(
    obj_ty: TypeId,
    field: &str,
    type_ctx: &TypeCtx,
    struct_fields: &HashMap<String, HashMap<String, TypeId>>,
) -> Option<(TypeId, ProjectionOwnerMode)> {
    match type_ctx.lookup(obj_ty) {
        Some(TypeKind::HeapHandle {
            hk: HandleKind::Borrow,
            base: HeapBase::UserType { type_sid, .. },
        }) => struct_fields
            .get(&type_sid.0)
            .and_then(|fields| fields.get(field).copied())
            .map(|ty| (ty, ProjectionOwnerMode::Shared)),
        Some(TypeKind::HeapHandle {
            hk: HandleKind::MutBorrow,
            base: HeapBase::UserType { type_sid, .. },
        }) => struct_fields
            .get(&type_sid.0)
            .and_then(|fields| fields.get(field).copied())
            .map(|ty| (ty, ProjectionOwnerMode::Mut)),
        _ => None,
    }
}

fn projected_array_elem_type(
    arr_ty: TypeId,
    type_ctx: &TypeCtx,
) -> Option<(TypeId, ProjectionOwnerMode)> {
    match type_ctx.lookup(arr_ty) {
        Some(TypeKind::HeapHandle {
            hk: HandleKind::Borrow,
            base: HeapBase::BuiltinArray { elem },
        }) => Some((*elem, ProjectionOwnerMode::Shared)),
        Some(TypeKind::HeapHandle {
            hk: HandleKind::MutBorrow,
            base: HeapBase::BuiltinArray { elem },
        }) => Some((*elem, ProjectionOwnerMode::Mut)),
        _ => None,
    }
}

fn projected_map_val_type(
    map_ty: TypeId,
    type_ctx: &TypeCtx,
) -> Option<(TypeId, ProjectionOwnerMode)> {
    match type_ctx.lookup(map_ty) {
        Some(TypeKind::HeapHandle {
            hk: HandleKind::Borrow,
            base: HeapBase::BuiltinMap { val, .. },
        }) => Some((*val, ProjectionOwnerMode::Shared)),
        Some(TypeKind::HeapHandle {
            hk: HandleKind::MutBorrow,
            base: HeapBase::BuiltinMap { val, .. },
        }) => Some((*val, ProjectionOwnerMode::Mut)),
        _ => None,
    }
}

fn check_projection_result_type(
    func: &HirFunction,
    op_name: &str,
    result_ty: TypeId,
    stored_ty: TypeId,
    owner_mode: ProjectionOwnerMode,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) {
    if !is_move_only(stored_ty, type_ctx) {
        if result_ty != stored_ty {
            emit_error(
                diag,
                codes::MPO0004,
                &format!(
                    "fn '{}': {} on Copy type must return by-value {}",
                    func.name,
                    op_name,
                    type_ctx.type_str(stored_ty)
                ),
            );
        }
        return;
    }

    match type_ctx.lookup(stored_ty) {
        Some(TypeKind::HeapHandle {
            hk: HandleKind::Weak,
            base,
        }) => {
            if !matches_handle_type(result_ty, HandleKind::Weak, base, type_ctx) {
                emit_error(
                    diag,
                    codes::MPO0004,
                    &format!(
                        "fn '{}': {} on weak field/element must return weak clone {}",
                        func.name,
                        op_name,
                        type_ctx.type_str(stored_ty)
                    ),
                );
            }
        }
        Some(TypeKind::HeapHandle {
            hk: HandleKind::Unique,
            base,
        }) => {
            let expected_hk = if owner_mode == ProjectionOwnerMode::Mut {
                HandleKind::MutBorrow
            } else {
                HandleKind::Borrow
            };
            if !matches_handle_type(result_ty, expected_hk, base, type_ctx) {
                emit_error(
                    diag,
                    codes::MPO0004,
                    &format!(
                        "fn '{}': {} on strong handle must return {} {}",
                        func.name,
                        op_name,
                        if expected_hk == HandleKind::MutBorrow {
                            "mutborrow"
                        } else {
                            "borrow"
                        },
                        heap_base_name(base, type_ctx)
                    ),
                );
            }
        }
        Some(TypeKind::HeapHandle {
            hk: HandleKind::Shared,
            base,
        }) => {
            if !matches_handle_type(result_ty, HandleKind::Borrow, base, type_ctx) {
                emit_error(
                    diag,
                    codes::MPO0004,
                    &format!(
                        "fn '{}': {} on shared handle must return borrow {}",
                        func.name,
                        op_name,
                        heap_base_name(base, type_ctx)
                    ),
                );
            }
        }
        _ => {
            let expected = if owner_mode == ProjectionOwnerMode::Mut {
                HandleKind::MutBorrow
            } else {
                HandleKind::Borrow
            };
            if !matches!(handle_kind(result_ty, type_ctx), Some(hk) if hk == expected) {
                emit_error(
                    diag,
                    codes::MPO0004,
                    &format!(
                        "fn '{}': {} on move-only value type must return {} projection",
                        func.name,
                        op_name,
                        if expected == HandleKind::MutBorrow {
                            "mutborrow"
                        } else {
                            "borrow"
                        }
                    ),
                );
            }
        }
    }
}

fn matches_handle_type(
    ty: TypeId,
    expected_hk: HandleKind,
    expected_base: &HeapBase,
    type_ctx: &TypeCtx,
) -> bool {
    matches!(
        type_ctx.lookup(ty),
        Some(TypeKind::HeapHandle { hk, base }) if *hk == expected_hk && base == expected_base
    )
}

fn heap_base_name(base: &HeapBase, type_ctx: &TypeCtx) -> String {
    type_ctx
        .types
        .iter()
        .find_map(|(id, kind)| match kind {
            TypeKind::HeapHandle {
                hk: HandleKind::Unique,
                base: candidate,
            } if candidate == base => Some(type_ctx.type_str(*id)),
            _ => None,
        })
        .unwrap_or_else(|| format!("{:?}", base))
}

fn is_option_of(option_ty: TypeId, inner_ty: TypeId, type_ctx: &TypeCtx) -> bool {
    matches!(
        type_ctx.lookup(option_ty),
        Some(TypeKind::BuiltinOption { inner }) if *inner == inner_ty
    )
}

fn is_dupable_type(ty: TypeId, type_ctx: &TypeCtx) -> bool {
    if !is_move_only(ty, type_ctx) {
        return true;
    }
    matches!(
        type_ctx.lookup(ty),
        Some(TypeKind::HeapHandle {
            hk: HandleKind::Shared | HandleKind::Weak,
            ..
        })
    )
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum ParamMode {
    ByValueCopy,
    ByValueMove,
    Borrow,
    MutBorrow,
}

fn param_mode(ty: TypeId, type_ctx: &TypeCtx) -> ParamMode {
    match handle_kind(ty, type_ctx) {
        Some(HandleKind::Borrow) => ParamMode::Borrow,
        Some(HandleKind::MutBorrow) => ParamMode::MutBorrow,
        _ => {
            if is_move_only(ty, type_ctx) {
                ParamMode::ByValueMove
            } else {
                ParamMode::ByValueCopy
            }
        }
    }
}

fn check_call_argument_modes_instr(
    func: &HirFunction,
    op: &HirOp,
    local_types: &HashMap<LocalId, TypeId>,
    type_ctx: &TypeCtx,
    fn_param_types: &HashMap<String, Vec<TypeId>>,
    diag: &mut DiagnosticBag,
) {
    match op {
        HirOp::Call {
            callee_sid, args, ..
        }
        | HirOp::SuspendCall {
            callee_sid, args, ..
        } => check_call_argument_modes(
            func,
            &callee_sid.0,
            args,
            fn_param_types.get(&callee_sid.0),
            local_types,
            type_ctx,
            diag,
        ),
        _ => {}
    }
}

fn check_call_argument_modes_void(
    func: &HirFunction,
    op: &HirOpVoid,
    local_types: &HashMap<LocalId, TypeId>,
    type_ctx: &TypeCtx,
    fn_param_types: &HashMap<String, Vec<TypeId>>,
    diag: &mut DiagnosticBag,
) {
    if let HirOpVoid::CallVoid {
        callee_sid, args, ..
    } = op
    {
        check_call_argument_modes(
            func,
            &callee_sid.0,
            args,
            fn_param_types.get(&callee_sid.0),
            local_types,
            type_ctx,
            diag,
        );
    }
}

fn check_call_argument_modes(
    func: &HirFunction,
    callee_sid: &str,
    args: &[HirValue],
    param_types: Option<&Vec<TypeId>>,
    local_types: &HashMap<LocalId, TypeId>,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) {
    let Some(param_types) = param_types else {
        return;
    };

    for (idx, arg) in args.iter().enumerate() {
        let Some(param_ty) = param_types.get(idx).copied() else {
            continue;
        };
        let Some(arg_ty) = value_type(arg, local_types) else {
            continue;
        };

        let ok = match param_mode(param_ty, type_ctx) {
            ParamMode::Borrow => matches!(
                handle_kind(arg_ty, type_ctx),
                Some(HandleKind::Borrow | HandleKind::MutBorrow)
            ),
            ParamMode::MutBorrow => matches!(handle_kind(arg_ty, type_ctx), Some(HandleKind::MutBorrow)),
            ParamMode::ByValueCopy | ParamMode::ByValueMove => {
                !matches!(
                    handle_kind(arg_ty, type_ctx),
                    Some(HandleKind::Borrow | HandleKind::MutBorrow)
                )
            }
        };

        if !ok {
            emit_error(
                diag,
                codes::MPO0004,
                &format!(
                    "fn '{}': call argument {} does not match parameter ownership mode for callee '{}'",
                    func.name, idx, callee_sid
                ),
            );
        }
    }
}

fn consumed_call_args(
    args: &[HirValue],
    param_types: Option<&Vec<TypeId>>,
    local_types: &HashMap<LocalId, TypeId>,
    type_ctx: &TypeCtx,
) -> Vec<LocalId> {
    let mut out = Vec::new();
    for (idx, arg) in args.iter().enumerate() {
        let Some(local) = as_local(arg) else {
            continue;
        };

        let expected = param_types.and_then(|ps| ps.get(idx)).copied();
        let consume = match expected {
            Some(param_ty) => matches!(param_mode(param_ty, type_ctx), ParamMode::ByValueMove),
            None => value_type(arg, local_types)
                .map(|ty| is_move_only(ty, type_ctx) && !is_borrow_type(ty, type_ctx))
                .unwrap_or(false),
        };

        if consume {
            out.push(local);
        }
    }
    out
}

fn analyze_moved_sets(
    func: &HirFunction,
    move_only_locals: &HashSet<LocalId>,
    block_index: &HashMap<u32, usize>,
    preds: &[Vec<usize>],
    local_types: &HashMap<LocalId, TypeId>,
    type_ctx: &TypeCtx,
    fn_param_types: &HashMap<String, Vec<TypeId>>,
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
                for local in op_consumed_locals(&instr.op, local_types, type_ctx, fn_param_types)
                {
                    if move_only_locals.contains(&local) {
                        cur.insert(local);
                    }
                }
            }
            for vop in &block.void_ops {
                for local in op_void_consumed_locals(vop, local_types, type_ctx, fn_param_types) {
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

fn collect_def_blocks(
    func: &HirFunction,
    block_index: &HashMap<u32, usize>,
) -> HashMap<LocalId, usize> {
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

fn is_borrow_value(
    v: &HirValue,
    local_types: &HashMap<LocalId, TypeId>,
    type_ctx: &TypeCtx,
) -> bool {
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

fn op_consumed_locals(
    op: &HirOp,
    local_types: &HashMap<LocalId, TypeId>,
    type_ctx: &TypeCtx,
    fn_param_types: &HashMap<String, Vec<TypeId>>,
) -> Vec<LocalId> {
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
        HirOp::Call {
            callee_sid, args, ..
        }
        | HirOp::SuspendCall {
            callee_sid, args, ..
        } => out.extend(consumed_call_args(
            args,
            fn_param_types.get(&callee_sid.0),
            local_types,
            type_ctx,
        )),
        HirOp::CallIndirect { args, .. } | HirOp::CallVoidIndirect { args, .. } => out.extend(
            consumed_call_args(args, None, local_types, type_ctx),
        ),
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

fn op_void_consumed_locals(
    op: &HirOpVoid,
    local_types: &HashMap<LocalId, TypeId>,
    type_ctx: &TypeCtx,
    fn_param_types: &HashMap<String, Vec<TypeId>>,
) -> Vec<LocalId> {
    let mut out = Vec::new();
    match op {
        HirOpVoid::CallVoid {
            callee_sid, args, ..
        } => out.extend(consumed_call_args(
            args,
            fn_param_types.get(&callee_sid.0),
            local_types,
            type_ctx,
        )),
        HirOpVoid::CallVoidIndirect { args, .. } => out.extend(consumed_call_args(
            args,
            None,
            local_types,
            type_ctx,
        )),
        _ => out.extend(op_void_non_call_consumed_locals(op)),
    }

    out
}

fn op_void_non_call_consumed_locals(op: &HirOpVoid) -> Vec<LocalId> {
    let mut out = Vec::new();
    let push = |v: &HirValue, out: &mut Vec<LocalId>| {
        if let Some(l) = as_local(v) {
            out.push(l);
        }
    };

    match op {
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
        | HirOpVoid::GpuBarrier
        | HirOpVoid::CallVoid { .. }
        | HirOpVoid::CallVoidIndirect { .. } => {}
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
        HirOpVoid::CallVoidIndirect { callee, args } => {
            f(callee);
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
        HirTerminator::Ret(None) | HirTerminator::Br(_) | HirTerminator::Unreachable => {}
        HirTerminator::Cbr { cond, .. } => f(cond),
        HirTerminator::Switch { val, .. } => f(val),
    }
}

fn ownership_why_trace(code: &str, message: &str) -> Option<WhyTrace> {
    match code {
        codes::MPO0003 => Some(WhyTrace::ownership(vec![
            WhyEvent::new("Definition site: borrow value is introduced (borrow op or borrowed parameter)."),
            WhyEvent::new(format!("Conflicting use site: {message}")),
        ])),
        codes::MPO0004 => Some(WhyTrace::ownership(vec![
            WhyEvent::new("Definition site: reference ownership mode is established for the receiver value."),
            WhyEvent::new(format!("Conflicting use site: {message}")),
        ])),
        codes::MPO0011 => Some(WhyTrace::ownership(vec![
            WhyEvent::new("Definition site: move-only value is defined and later borrowed."),
            WhyEvent::new("Borrow chain: an active shared/mut borrow is still live when move is attempted."),
            WhyEvent::new(format!("Conflicting use site: {message}")),
        ])),
        codes::MPO0101 => Some(WhyTrace::ownership(vec![
            WhyEvent::new("Definition site: borrow is created in a predecessor/source block."),
            WhyEvent::new("Control-flow conflict: borrow is consumed in a different block."),
            WhyEvent::new(format!("Conflicting use site: {message}")),
        ])),
        codes::MPO0102 => Some(WhyTrace::ownership(vec![
            WhyEvent::new("Definition site: borrow appears in phi result or phi incoming value."),
            WhyEvent::new("Merge-point conflict: borrows are not allowed to flow through phi nodes."),
            WhyEvent::new(format!("Conflicting use site: {message}")),
        ])),
        _ => None,
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
        why: ownership_why_trace(code, message),
        suggested_fixes: vec![],
    });
}

#[allow(dead_code)]
fn _block_id_to_usize(b: BlockId) -> usize {
    b.0 as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use magpie_types::{fixed_type_ids, FnId, HeapBase, ModuleId, Sid};

    #[test]
    fn test_move_only_detection() {
        let mut type_ctx = TypeCtx::new();
        let heap_ty = type_ctx.intern(TypeKind::HeapHandle {
            hk: HandleKind::Unique,
            base: HeapBase::BuiltinStr,
        });

        assert!(is_move_only(heap_ty, &type_ctx));
        assert!(!is_move_only(fixed_type_ids::I32, &type_ctx));
    }

    #[test]
    fn test_borrow_in_phi_rejected() {
        let mut type_ctx = TypeCtx::new();
        let borrow_ty = type_ctx.intern(TypeKind::HeapHandle {
            hk: HandleKind::Borrow,
            base: HeapBase::BuiltinStr,
        });

        let module = HirModule {
            module_id: ModuleId(0),
            sid: Sid("M:OWNTEST0000".to_string()),
            path: "test.own".to_string(),
            functions: vec![HirFunction {
                fn_id: FnId(0),
                sid: Sid("F:OWNTEST0000".to_string()),
                name: "phi_borrow".to_string(),
                params: vec![(LocalId(0), borrow_ty)],
                ret_ty: fixed_type_ids::UNIT,
                blocks: vec![HirBlock {
                    id: BlockId(0),
                    instrs: vec![HirInstr {
                        dst: LocalId(1),
                        ty: borrow_ty,
                        op: HirOp::Phi {
                            ty: borrow_ty,
                            incomings: vec![(BlockId(0), HirValue::Local(LocalId(0)))],
                        },
                    }],
                    void_ops: vec![],
                    terminator: HirTerminator::Ret(None),
                }],
                is_async: false,
                is_unsafe: false,
            }],
            globals: vec![],
            type_decls: vec![],
        };

        let mut diag = DiagnosticBag::new(16);
        let _ = check_ownership(&module, &type_ctx, &mut diag);

        assert!(
            diag.diagnostics.iter().any(|d| d.code == codes::MPO0102),
            "expected MPO0102 diagnostics, got: {:?}",
            diag.diagnostics
                .iter()
                .map(|d| d.code.as_str())
                .collect::<Vec<_>>()
        );
        let phi_diag = diag
            .diagnostics
            .iter()
            .find(|d| d.code == codes::MPO0102)
            .expect("missing MPO0102 diagnostic");
        assert!(
            phi_diag
                .why
                .as_ref()
                .map(|why| why.trace.len() >= 2)
                .unwrap_or(false),
            "expected MPO0102 diagnostic to include why.trace with definition/conflict events"
        );
    }

    #[test]
    fn test_projection_rules_and_map_get_dupable_enforced() {
        let mut type_ctx = TypeCtx::new();
        let user_sid = Sid("T:OWNPROJ0000".to_string());
        let weak_str = type_ctx.intern(TypeKind::HeapHandle {
            hk: HandleKind::Weak,
            base: HeapBase::BuiltinStr,
        });
        let borrow_str = type_ctx.intern(TypeKind::HeapHandle {
            hk: HandleKind::Borrow,
            base: HeapBase::BuiltinStr,
        });
        let obj_borrow = type_ctx.intern(TypeKind::HeapHandle {
            hk: HandleKind::Borrow,
            base: HeapBase::UserType {
                type_sid: user_sid.clone(),
                targs: vec![],
            },
        });
        let arr_borrow_unique = type_ctx.intern(TypeKind::HeapHandle {
            hk: HandleKind::Borrow,
            base: HeapBase::BuiltinArray {
                elem: fixed_type_ids::STR,
            },
        });
        let map_borrow_weak = type_ctx.intern(TypeKind::HeapHandle {
            hk: HandleKind::Borrow,
            base: HeapBase::BuiltinMap {
                key: fixed_type_ids::I32,
                val: weak_str,
            },
        });
        let map_borrow_unique = type_ctx.intern(TypeKind::HeapHandle {
            hk: HandleKind::Borrow,
            base: HeapBase::BuiltinMap {
                key: fixed_type_ids::I32,
                val: fixed_type_ids::STR,
            },
        });
        let option_str = type_ctx.intern(TypeKind::BuiltinOption {
            inner: fixed_type_ids::STR,
        });

        let zero = HirValue::Const(magpie_hir::HirConst {
            ty: fixed_type_ids::I32,
            lit: magpie_hir::HirConstLit::IntLit(0),
        });

        let module = HirModule {
            module_id: ModuleId(0),
            sid: Sid("M:OWNPROJ0000".to_string()),
            path: "test.own".to_string(),
            functions: vec![HirFunction {
                fn_id: FnId(0),
                sid: Sid("F:OWNPROJ0000".to_string()),
                name: "projection_bad".to_string(),
                params: vec![
                    (LocalId(0), obj_borrow),
                    (LocalId(1), arr_borrow_unique),
                    (LocalId(2), map_borrow_weak),
                    (LocalId(3), map_borrow_unique),
                ],
                ret_ty: fixed_type_ids::UNIT,
                blocks: vec![HirBlock {
                    id: BlockId(0),
                    instrs: vec![
                        HirInstr {
                            dst: LocalId(10),
                            ty: borrow_str,
                            op: HirOp::GetField {
                                obj: HirValue::Local(LocalId(0)),
                                field: "copy_i32".to_string(),
                            },
                        },
                        HirInstr {
                            dst: LocalId(11),
                            ty: fixed_type_ids::STR,
                            op: HirOp::ArrGet {
                                arr: HirValue::Local(LocalId(1)),
                                idx: zero.clone(),
                            },
                        },
                        HirInstr {
                            dst: LocalId(12),
                            ty: borrow_str,
                            op: HirOp::MapGetRef {
                                map: HirValue::Local(LocalId(2)),
                                key: zero.clone(),
                            },
                        },
                        HirInstr {
                            dst: LocalId(13),
                            ty: option_str,
                            op: HirOp::MapGet {
                                map: HirValue::Local(LocalId(3)),
                                key: zero,
                            },
                        },
                    ],
                    void_ops: vec![],
                    terminator: HirTerminator::Ret(None),
                }],
                is_async: false,
                is_unsafe: false,
            }],
            globals: vec![],
            type_decls: vec![magpie_hir::HirTypeDecl::Struct {
                sid: user_sid,
                name: "TProj".to_string(),
                fields: vec![
                    ("copy_i32".to_string(), fixed_type_ids::I32),
                    ("strong_str".to_string(), fixed_type_ids::STR),
                    ("weak_str".to_string(), weak_str),
                ],
            }],
        };

        let mut diag = DiagnosticBag::new(32);
        let _ = check_ownership(&module, &type_ctx, &mut diag);

        assert!(
            diag.diagnostics.iter().any(|d| d.code == codes::MPO0004),
            "expected MPO0004 projection diagnostics, got: {:?}",
            diag.diagnostics
                .iter()
                .map(|d| d.code.as_str())
                .collect::<Vec<_>>()
        );
        assert!(
            diag.diagnostics.iter().any(|d| d.code == codes::MPO0103),
            "expected MPO0103 map.get Dupable diagnostics, got: {:?}",
            diag.diagnostics
                .iter()
                .map(|d| d.code.as_str())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_call_borrow_param_not_consumed() {
        let mut type_ctx = TypeCtx::new();
        let borrow_str = type_ctx.intern(TypeKind::HeapHandle {
            hk: HandleKind::Borrow,
            base: HeapBase::BuiltinStr,
        });

        let callee_sid = Sid("F:TAKEBORROW0".to_string());
        let caller_sid = Sid("F:CALLER00000".to_string());
        let module = HirModule {
            module_id: ModuleId(0),
            sid: Sid("M:OWNCALL0000".to_string()),
            path: "test.own".to_string(),
            functions: vec![
                HirFunction {
                    fn_id: FnId(0),
                    sid: callee_sid.clone(),
                    name: "take_borrow".to_string(),
                    params: vec![(LocalId(0), borrow_str)],
                    ret_ty: fixed_type_ids::UNIT,
                    blocks: vec![HirBlock {
                        id: BlockId(0),
                        instrs: vec![],
                        void_ops: vec![],
                        terminator: HirTerminator::Ret(None),
                    }],
                    is_async: false,
                    is_unsafe: false,
                },
                HirFunction {
                    fn_id: FnId(1),
                    sid: caller_sid,
                    name: "caller".to_string(),
                    params: vec![(LocalId(0), borrow_str)],
                    ret_ty: fixed_type_ids::UNIT,
                    blocks: vec![HirBlock {
                        id: BlockId(0),
                        instrs: vec![
                            HirInstr {
                                dst: LocalId(1),
                                ty: fixed_type_ids::UNIT,
                                op: HirOp::Call {
                                    callee_sid: callee_sid.clone(),
                                    inst: vec![],
                                    args: vec![HirValue::Local(LocalId(0))],
                                },
                            },
                            HirInstr {
                                dst: LocalId(2),
                                ty: fixed_type_ids::UNIT,
                                op: HirOp::Call {
                                    callee_sid,
                                    inst: vec![],
                                    args: vec![HirValue::Local(LocalId(0))],
                                },
                            },
                        ],
                        void_ops: vec![],
                        terminator: HirTerminator::Ret(None),
                    }],
                    is_async: false,
                    is_unsafe: false,
                },
            ],
            globals: vec![],
            type_decls: vec![],
        };

        let mut diag = DiagnosticBag::new(32);
        let _ = check_ownership(&module, &type_ctx, &mut diag);
        assert!(
            !diag.diagnostics.iter().any(|d| d.code == "MPO0007"),
            "borrow call args should not be consumed; got diagnostics: {:?}",
            diag.diagnostics
                .iter()
                .map(|d| d.code.as_str())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_spawn_callable_requires_send_captures() {
        let mut type_ctx = TypeCtx::new();
        let borrow_str = type_ctx.intern(TypeKind::HeapHandle {
            hk: HandleKind::Borrow,
            base: HeapBase::BuiltinStr,
        });
        let callable_ty = type_ctx.intern(TypeKind::HeapHandle {
            hk: HandleKind::Unique,
            base: HeapBase::Callable {
                sig_sid: Sid("E:SPAWNTEST0".to_string()),
            },
        });

        let module = HirModule {
            module_id: ModuleId(0),
            sid: Sid("M:SPAWNTEST0".to_string()),
            path: "test.own".to_string(),
            functions: vec![HirFunction {
                fn_id: FnId(0),
                sid: Sid("F:SPAWNTEST0".to_string()),
                name: "spawn_non_send_capture".to_string(),
                params: vec![(LocalId(0), borrow_str)],
                ret_ty: fixed_type_ids::UNIT,
                blocks: vec![HirBlock {
                    id: BlockId(0),
                    instrs: vec![
                        HirInstr {
                            dst: LocalId(1),
                            ty: callable_ty,
                            op: HirOp::CallableCapture {
                                fn_ref: Sid("F:WORKITEM00".to_string()),
                                captures: vec![("msg".to_string(), HirValue::Local(LocalId(0)))],
                            },
                        },
                        HirInstr {
                            dst: LocalId(2),
                            ty: fixed_type_ids::UNIT,
                            op: HirOp::Call {
                                callee_sid: Sid("F:std.thread.spawn".to_string()),
                                inst: vec![],
                                args: vec![HirValue::Local(LocalId(1))],
                            },
                        },
                    ],
                    void_ops: vec![],
                    terminator: HirTerminator::Ret(None),
                }],
                is_async: false,
                is_unsafe: false,
            }],
            globals: vec![],
            type_decls: vec![],
        };

        let mut diag = DiagnosticBag::new(32);
        let _ = check_ownership(&module, &type_ctx, &mut diag);

        assert!(
            diag.diagnostics.iter().any(|d| d.code == "MPO0201"),
            "expected MPO0201 diagnostics, got: {:?}",
            diag.diagnostics
                .iter()
                .map(|d| d.code.as_str())
                .collect::<Vec<_>>()
        );
    }
}
