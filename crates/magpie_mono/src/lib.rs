//! Magpie monomorphization pass (§8.4, §8.6, §18.7).

use magpie_diag::{codes, Diagnostic, DiagnosticBag, Severity};
use magpie_hir::{
    FnId, HirBlock, HirConst, HirFunction, HirInstr, HirModule, HirOp, HirOpVoid, HirTerminator,
    HirValue, Sid, TypeCtx, TypeId, TypeKind,
};
use magpie_types::HeapBase;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct InstanceKey {
    callee_sid: Sid,
    type_args: Vec<TypeId>,
}

pub fn monomorphize(
    modules: &[HirModule],
    type_ctx: &mut TypeCtx,
    max_instances: u32,
    diag: &mut DiagnosticBag,
) -> Result<Vec<HirModule>, ()> {
    let before_errors = diag.error_count();

    let mut out_modules = modules.to_vec();
    let mut template_by_sid: HashMap<String, (usize, HirFunction)> = HashMap::new();
    for (module_idx, module) in modules.iter().enumerate() {
        for func in &module.functions {
            template_by_sid.insert(func.sid.0.clone(), (module_idx, func.clone()));
        }
    }

    let mut instance_sid_map: HashMap<InstanceKey, Sid> = HashMap::new();
    let mut instance_count: u32 = 0;

    loop {
        let mut requested_instances: HashSet<InstanceKey> = HashSet::new();
        for module in &out_modules {
            for func in &module.functions {
                collect_generic_call_keys(func, &mut requested_instances);
            }
        }

        let mut new_keys = requested_instances
            .into_iter()
            .filter(|key| !instance_sid_map.contains_key(key))
            .collect::<Vec<_>>();
        new_keys.sort_by(|a, b| {
            a.callee_sid.0.cmp(&b.callee_sid.0).then_with(|| {
                a.type_args
                    .iter()
                    .map(|t| t.0)
                    .cmp(b.type_args.iter().map(|t| t.0))
            })
        });

        let mut created_any = false;
        for key in new_keys {
            let Some((owner_module_idx, template)) =
                template_by_sid.get(&key.callee_sid.0).cloned()
            else {
                // External/unknown callee: leave call as-is.
                continue;
            };

            if instance_count >= max_instances {
                emit_excessive_mono(diag, max_instances);
                return Err(());
            }

            let mut specialized = specialize_function(&template, &key.type_args, type_ctx);
            specialized.fn_id = next_fn_id(&out_modules[owner_module_idx]);

            let new_sid = specialized.sid.clone();
            out_modules[owner_module_idx]
                .functions
                .push(specialized.clone());
            template_by_sid.insert(new_sid.0.clone(), (owner_module_idx, specialized));
            instance_sid_map.insert(key, new_sid);

            instance_count = instance_count.saturating_add(1);
            created_any = true;
        }

        let mut rewrote_any = false;
        for module in &mut out_modules {
            for func in &mut module.functions {
                if rewrite_generic_calls(func, &instance_sid_map) {
                    rewrote_any = true;
                }
            }
        }

        if !created_any && !rewrote_any {
            break;
        }
    }

    if diag.error_count() > before_errors {
        Err(())
    } else {
        Ok(out_modules)
    }
}

fn emit_excessive_mono(diag: &mut DiagnosticBag, max_instances: u32) {
    let msg = format!(
        "Monomorphization budget exceeded: max_mono_instances={} (MPL2020 EXCESSIVE_MONO).",
        max_instances
    );
    diag.emit(Diagnostic {
        code: codes::MPL2020.to_string(),
        severity: Severity::Error,
        title: "EXCESSIVE_MONO".to_string(),
        primary_span: None,
        secondary_spans: vec![],
        message: msg,
        explanation_md: None,
        why: None,
        suggested_fixes: vec![],
    });
}

fn next_fn_id(module: &HirModule) -> FnId {
    let next = module
        .functions
        .iter()
        .map(|f| f.fn_id.0)
        .max()
        .map_or(0, |m| m.saturating_add(1));
    FnId(next)
}

fn collect_generic_call_keys(func: &HirFunction, out: &mut HashSet<InstanceKey>) {
    for block in &func.blocks {
        for instr in &block.instrs {
            match &instr.op {
                HirOp::Call {
                    callee_sid, inst, ..
                }
                | HirOp::SuspendCall {
                    callee_sid, inst, ..
                } if !inst.is_empty() => {
                    out.insert(InstanceKey {
                        callee_sid: callee_sid.clone(),
                        type_args: inst.clone(),
                    });
                }
                _ => {}
            }
        }

        for void_op in &block.void_ops {
            if let HirOpVoid::CallVoid {
                callee_sid, inst, ..
            } = void_op
            {
                if !inst.is_empty() {
                    out.insert(InstanceKey {
                        callee_sid: callee_sid.clone(),
                        type_args: inst.clone(),
                    });
                }
            }
        }
    }
}

fn rewrite_generic_calls(
    func: &mut HirFunction,
    instance_sid_map: &HashMap<InstanceKey, Sid>,
) -> bool {
    let mut changed = false;

    for block in &mut func.blocks {
        for instr in &mut block.instrs {
            match &mut instr.op {
                HirOp::Call {
                    callee_sid, inst, ..
                }
                | HirOp::SuspendCall {
                    callee_sid, inst, ..
                } => {
                    if inst.is_empty() {
                        continue;
                    }
                    let key = InstanceKey {
                        callee_sid: callee_sid.clone(),
                        type_args: inst.clone(),
                    };
                    if let Some(new_sid) = instance_sid_map.get(&key) {
                        *callee_sid = new_sid.clone();
                        inst.clear();
                        changed = true;
                    }
                }
                _ => {}
            }
        }

        for void_op in &mut block.void_ops {
            if let HirOpVoid::CallVoid {
                callee_sid, inst, ..
            } = void_op
            {
                if inst.is_empty() {
                    continue;
                }
                let key = InstanceKey {
                    callee_sid: callee_sid.clone(),
                    type_args: inst.clone(),
                };
                if let Some(new_sid) = instance_sid_map.get(&key) {
                    *callee_sid = new_sid.clone();
                    inst.clear();
                    changed = true;
                }
            }
        }
    }

    changed
}

fn type_param_map_for_specialization(
    func: &HirFunction,
    type_args: &[TypeId],
    type_ctx: &TypeCtx,
) -> HashMap<TypeId, TypeId> {
    let mut all_types = Vec::new();
    let mut seen = HashSet::new();
    for (_, ty) in &func.params {
        collect_type_ids_preorder(*ty, type_ctx, &mut seen, &mut all_types);
    }
    collect_type_ids_preorder(func.ret_ty, type_ctx, &mut seen, &mut all_types);

    let mut candidates = Vec::new();
    let mut candidate_seen = HashSet::new();
    for ty in &all_types {
        if is_type_param_candidate(*ty, type_ctx) && candidate_seen.insert(*ty) {
            candidates.push(*ty);
        }
    }

    if candidates.len() < type_args.len() {
        for ty in &all_types {
            if candidate_seen.insert(*ty) {
                candidates.push(*ty);
            }
            if candidates.len() >= type_args.len() {
                break;
            }
        }
    }

    let mut map = HashMap::new();
    for (param_ty, concrete_ty) in candidates.into_iter().zip(type_args.iter().copied()) {
        map.insert(param_ty, concrete_ty);
    }
    map
}

fn is_type_param_candidate(ty: TypeId, type_ctx: &TypeCtx) -> bool {
    match type_ctx.lookup(ty) {
        None => true,
        Some(TypeKind::ValueStruct { .. }) => true,
        Some(TypeKind::HeapHandle {
            base: HeapBase::UserType { targs, .. },
            ..
        }) => targs.is_empty(),
        _ => false,
    }
}

fn collect_type_ids_preorder(
    ty: TypeId,
    type_ctx: &TypeCtx,
    seen: &mut HashSet<TypeId>,
    out: &mut Vec<TypeId>,
) {
    if !seen.insert(ty) {
        return;
    }

    out.push(ty);
    let Some(kind) = type_ctx.lookup(ty) else {
        return;
    };

    match kind {
        TypeKind::Prim(_) | TypeKind::ValueStruct { .. } => {}
        TypeKind::HeapHandle { base, .. } => match base {
            HeapBase::BuiltinStr | HeapBase::BuiltinStrBuilder | HeapBase::Callable { .. } => {}
            HeapBase::BuiltinArray { elem }
            | HeapBase::BuiltinMutex { inner: elem }
            | HeapBase::BuiltinRwLock { inner: elem }
            | HeapBase::BuiltinCell { inner: elem }
            | HeapBase::BuiltinFuture { result: elem }
            | HeapBase::BuiltinChannelSend { elem }
            | HeapBase::BuiltinChannelRecv { elem } => {
                collect_type_ids_preorder(*elem, type_ctx, seen, out);
            }
            HeapBase::BuiltinMap { key, val } => {
                collect_type_ids_preorder(*key, type_ctx, seen, out);
                collect_type_ids_preorder(*val, type_ctx, seen, out);
            }
            HeapBase::UserType { targs, .. } => {
                for targ in targs {
                    collect_type_ids_preorder(*targ, type_ctx, seen, out);
                }
            }
        },
        TypeKind::BuiltinOption { inner }
        | TypeKind::RawPtr { to: inner }
        | TypeKind::Arr { elem: inner, .. }
        | TypeKind::Vec { elem: inner, .. } => {
            collect_type_ids_preorder(*inner, type_ctx, seen, out);
        }
        TypeKind::BuiltinResult { ok, err } => {
            collect_type_ids_preorder(*ok, type_ctx, seen, out);
            collect_type_ids_preorder(*err, type_ctx, seen, out);
        }
        TypeKind::Tuple { elems } => {
            for elem in elems {
                collect_type_ids_preorder(*elem, type_ctx, seen, out);
            }
        }
    }
}

fn instance_id(callee_sid: &Sid, type_args: &[TypeId], type_ctx: &TypeCtx) -> String {
    let mut payload = format!("magpie:inst:v0.1|{}", callee_sid.0);
    for ty in type_args {
        payload.push('|');
        payload.push_str(&canonical_type_str(*ty, type_ctx));
    }
    let digest_hex = blake3_hex_best_effort(&payload);
    let short = digest_hex.chars().take(16).collect::<String>();
    format!("I:{}", short)
}

fn specialized_sid(base_sid: &Sid, inst_id: &str) -> Sid {
    let suffix = inst_id.strip_prefix("I:").unwrap_or(inst_id);
    Sid(format!("{}$I${}", base_sid.0, suffix))
}

fn specialized_name(base_name: &str, inst_id: &str) -> String {
    let suffix = inst_id.strip_prefix("I:").unwrap_or(inst_id);
    format!("{}$I${}", base_name, suffix)
}

fn canonical_type_str(ty: TypeId, type_ctx: &TypeCtx) -> String {
    let Some(kind) = type_ctx.lookup(ty) else {
        return format!("type#{}", ty.0);
    };

    match kind {
        TypeKind::Prim(p) => match p {
            magpie_hir::PrimType::I1 => "i1".to_string(),
            magpie_hir::PrimType::I8 => "i8".to_string(),
            magpie_hir::PrimType::I16 => "i16".to_string(),
            magpie_hir::PrimType::I32 => "i32".to_string(),
            magpie_hir::PrimType::I64 => "i64".to_string(),
            magpie_hir::PrimType::I128 => "i128".to_string(),
            magpie_hir::PrimType::U1 => "u1".to_string(),
            magpie_hir::PrimType::U8 => "u8".to_string(),
            magpie_hir::PrimType::U16 => "u16".to_string(),
            magpie_hir::PrimType::U32 => "u32".to_string(),
            magpie_hir::PrimType::U64 => "u64".to_string(),
            magpie_hir::PrimType::U128 => "u128".to_string(),
            magpie_hir::PrimType::F16 => "f16".to_string(),
            magpie_hir::PrimType::F32 => "f32".to_string(),
            magpie_hir::PrimType::F64 => "f64".to_string(),
            magpie_hir::PrimType::Bool => "bool".to_string(),
            magpie_hir::PrimType::Unit => "unit".to_string(),
        },
        TypeKind::HeapHandle { hk, base } => {
            let base_s = canonical_heap_base_str(base, type_ctx);
            match hk {
                magpie_hir::HandleKind::Unique => base_s,
                magpie_hir::HandleKind::Shared => format!("shared {}", base_s),
                magpie_hir::HandleKind::Borrow => format!("borrow {}", base_s),
                magpie_hir::HandleKind::MutBorrow => format!("mutborrow {}", base_s),
                magpie_hir::HandleKind::Weak => format!("weak {}", base_s),
            }
        }
        TypeKind::BuiltinOption { inner } => {
            format!("TOption<{}>", canonical_type_str(*inner, type_ctx))
        }
        TypeKind::BuiltinResult { ok, err } => {
            format!(
                "TResult<{},{}>",
                canonical_type_str(*ok, type_ctx),
                canonical_type_str(*err, type_ctx)
            )
        }
        TypeKind::RawPtr { to } => format!("rawptr<{}>", canonical_type_str(*to, type_ctx)),
        TypeKind::Arr { n, elem } => format!("arr<{},{}>", n, canonical_type_str(*elem, type_ctx)),
        TypeKind::Vec { n, elem } => format!("vec<{},{}>", n, canonical_type_str(*elem, type_ctx)),
        TypeKind::Tuple { elems } => {
            let joined = elems
                .iter()
                .map(|t| canonical_type_str(*t, type_ctx))
                .collect::<Vec<_>>()
                .join(",");
            format!("tuple<{}>", joined)
        }
        TypeKind::ValueStruct { sid } => sid.0.clone(),
    }
}

fn canonical_heap_base_str(base: &HeapBase, type_ctx: &TypeCtx) -> String {
    match base {
        HeapBase::BuiltinStr => "Str".to_string(),
        HeapBase::BuiltinArray { elem } => {
            format!("Array<{}>", canonical_type_str(*elem, type_ctx))
        }
        HeapBase::BuiltinMap { key, val } => format!(
            "Map<{},{}>",
            canonical_type_str(*key, type_ctx),
            canonical_type_str(*val, type_ctx)
        ),
        HeapBase::BuiltinStrBuilder => "TStrBuilder".to_string(),
        HeapBase::BuiltinMutex { inner } => {
            format!("TMutex<{}>", canonical_type_str(*inner, type_ctx))
        }
        HeapBase::BuiltinRwLock { inner } => {
            format!("TRwLock<{}>", canonical_type_str(*inner, type_ctx))
        }
        HeapBase::BuiltinCell { inner } => {
            format!("TCell<{}>", canonical_type_str(*inner, type_ctx))
        }
        HeapBase::BuiltinFuture { result } => {
            format!("TFuture<{}>", canonical_type_str(*result, type_ctx))
        }
        HeapBase::BuiltinChannelSend { elem } => {
            format!("TChannelSend<{}>", canonical_type_str(*elem, type_ctx))
        }
        HeapBase::BuiltinChannelRecv { elem } => {
            format!("TChannelRecv<{}>", canonical_type_str(*elem, type_ctx))
        }
        HeapBase::Callable { sig_sid } => format!("TCallable<{}>", sig_sid.0),
        HeapBase::UserType { type_sid, targs } => {
            if targs.is_empty() {
                type_sid.0.clone()
            } else {
                let joined = targs
                    .iter()
                    .map(|t| canonical_type_str(*t, type_ctx))
                    .collect::<Vec<_>>()
                    .join(",");
                format!("{}<{}>", type_sid.0, joined)
            }
        }
    }
}

fn blake3_hex_best_effort(payload: &str) -> String {
    let hash = blake3::hash(payload.as_bytes());
    hash.to_hex().to_string()
}

fn substitute_function_types(
    func: &mut HirFunction,
    param_map: &HashMap<TypeId, TypeId>,
    type_ctx: &mut TypeCtx,
) {
    for (_, ty) in &mut func.params {
        *ty = substitute_type(*ty, param_map, type_ctx);
    }
    func.ret_ty = substitute_type(func.ret_ty, param_map, type_ctx);

    for block in &mut func.blocks {
        substitute_block_types(block, param_map, type_ctx);
    }
}

fn substitute_block_types(
    block: &mut HirBlock,
    param_map: &HashMap<TypeId, TypeId>,
    type_ctx: &mut TypeCtx,
) {
    for instr in &mut block.instrs {
        substitute_instr_types(instr, param_map, type_ctx);
    }
    for void_op in &mut block.void_ops {
        substitute_void_op_types(void_op, param_map, type_ctx);
    }
    substitute_terminator_types(&mut block.terminator, param_map, type_ctx);
}

fn substitute_instr_types(
    instr: &mut HirInstr,
    param_map: &HashMap<TypeId, TypeId>,
    type_ctx: &mut TypeCtx,
) {
    instr.ty = substitute_type(instr.ty, param_map, type_ctx);
    substitute_op_types(&mut instr.op, param_map, type_ctx);
}

fn substitute_value_types(
    v: &mut HirValue,
    param_map: &HashMap<TypeId, TypeId>,
    type_ctx: &mut TypeCtx,
) {
    if let HirValue::Const(c) = v {
        substitute_const_types(c, param_map, type_ctx);
    }
}

fn substitute_const_types(
    c: &mut HirConst,
    param_map: &HashMap<TypeId, TypeId>,
    type_ctx: &mut TypeCtx,
) {
    c.ty = substitute_type(c.ty, param_map, type_ctx);
}

fn substitute_op_types(
    op: &mut HirOp,
    param_map: &HashMap<TypeId, TypeId>,
    type_ctx: &mut TypeCtx,
) {
    match op {
        HirOp::Const(c) => substitute_const_types(c, param_map, type_ctx),
        HirOp::Move { v }
        | HirOp::BorrowShared { v }
        | HirOp::BorrowMut { v }
        | HirOp::PtrAddr { p: v }
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
        | HirOp::Panic { msg: v } => {
            substitute_value_types(v, param_map, type_ctx);
        }
        HirOp::New { ty, fields } => {
            *ty = substitute_type(*ty, param_map, type_ctx);
            for (_, v) in fields {
                substitute_value_types(v, param_map, type_ctx);
            }
        }
        HirOp::GetField { obj, .. } => substitute_value_types(obj, param_map, type_ctx),
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
            substitute_value_types(lhs, param_map, type_ctx);
            substitute_value_types(rhs, param_map, type_ctx);
        }
        HirOp::Cast { to, v } => {
            *to = substitute_type(*to, param_map, type_ctx);
            substitute_value_types(v, param_map, type_ctx);
        }
        HirOp::PtrNull { to } => {
            *to = substitute_type(*to, param_map, type_ctx);
        }
        HirOp::PtrFromAddr { to, addr } => {
            *to = substitute_type(*to, param_map, type_ctx);
            substitute_value_types(addr, param_map, type_ctx);
        }
        HirOp::PtrAdd { p, count } => {
            substitute_value_types(p, param_map, type_ctx);
            substitute_value_types(count, param_map, type_ctx);
        }
        HirOp::PtrLoad { to, p } => {
            *to = substitute_type(*to, param_map, type_ctx);
            substitute_value_types(p, param_map, type_ctx);
        }
        HirOp::PtrStore { to, p, v } => {
            *to = substitute_type(*to, param_map, type_ctx);
            substitute_value_types(p, param_map, type_ctx);
            substitute_value_types(v, param_map, type_ctx);
        }
        HirOp::Call { inst, args, .. } | HirOp::SuspendCall { inst, args, .. } => {
            for ty in inst {
                *ty = substitute_type(*ty, param_map, type_ctx);
            }
            for arg in args {
                substitute_value_types(arg, param_map, type_ctx);
            }
        }
        HirOp::CallIndirect { callee, args } | HirOp::CallVoidIndirect { callee, args } => {
            substitute_value_types(callee, param_map, type_ctx);
            for arg in args {
                substitute_value_types(arg, param_map, type_ctx);
            }
        }
        HirOp::Phi { ty, incomings } => {
            *ty = substitute_type(*ty, param_map, type_ctx);
            for (_, v) in incomings {
                substitute_value_types(v, param_map, type_ctx);
            }
        }
        HirOp::EnumNew { args, .. } => {
            for (_, v) in args {
                substitute_value_types(v, param_map, type_ctx);
            }
        }
        HirOp::CallableCapture { captures, .. } => {
            for (_, v) in captures {
                substitute_value_types(v, param_map, type_ctx);
            }
        }
        HirOp::ArrNew { elem_ty, cap } => {
            *elem_ty = substitute_type(*elem_ty, param_map, type_ctx);
            substitute_value_types(cap, param_map, type_ctx);
        }
        HirOp::ArrGet { arr, idx } => {
            substitute_value_types(arr, param_map, type_ctx);
            substitute_value_types(idx, param_map, type_ctx);
        }
        HirOp::ArrSet { arr, idx, val } => {
            substitute_value_types(arr, param_map, type_ctx);
            substitute_value_types(idx, param_map, type_ctx);
            substitute_value_types(val, param_map, type_ctx);
        }
        HirOp::ArrPush { arr, val } | HirOp::ArrContains { arr, val } => {
            substitute_value_types(arr, param_map, type_ctx);
            substitute_value_types(val, param_map, type_ctx);
        }
        HirOp::ArrSlice { arr, start, end } => {
            substitute_value_types(arr, param_map, type_ctx);
            substitute_value_types(start, param_map, type_ctx);
            substitute_value_types(end, param_map, type_ctx);
        }
        HirOp::ArrMap { arr, func }
        | HirOp::ArrFilter { arr, func }
        | HirOp::ArrForeach { arr, func } => {
            substitute_value_types(arr, param_map, type_ctx);
            substitute_value_types(func, param_map, type_ctx);
        }
        HirOp::ArrReduce { arr, init, func } => {
            substitute_value_types(arr, param_map, type_ctx);
            substitute_value_types(init, param_map, type_ctx);
            substitute_value_types(func, param_map, type_ctx);
        }
        HirOp::MapNew { key_ty, val_ty } => {
            *key_ty = substitute_type(*key_ty, param_map, type_ctx);
            *val_ty = substitute_type(*val_ty, param_map, type_ctx);
        }
        HirOp::MapGet { map, key }
        | HirOp::MapGetRef { map, key }
        | HirOp::MapDelete { map, key }
        | HirOp::MapContainsKey { map, key }
        | HirOp::MapDeleteVoid { map, key } => {
            substitute_value_types(map, param_map, type_ctx);
            substitute_value_types(key, param_map, type_ctx);
        }
        HirOp::MapSet { map, key, val } => {
            substitute_value_types(map, param_map, type_ctx);
            substitute_value_types(key, param_map, type_ctx);
            substitute_value_types(val, param_map, type_ctx);
        }
        HirOp::StrConcat { a, b } | HirOp::StrEq { a, b } => {
            substitute_value_types(a, param_map, type_ctx);
            substitute_value_types(b, param_map, type_ctx);
        }
        HirOp::StrSlice { s, start, end } => {
            substitute_value_types(s, param_map, type_ctx);
            substitute_value_types(start, param_map, type_ctx);
            substitute_value_types(end, param_map, type_ctx);
        }
        HirOp::StrBuilderNew => {}
        HirOp::StrBuilderAppendStr { b, s } => {
            substitute_value_types(b, param_map, type_ctx);
            substitute_value_types(s, param_map, type_ctx);
        }
        HirOp::StrBuilderAppendI64 { b, v }
        | HirOp::StrBuilderAppendI32 { b, v }
        | HirOp::StrBuilderAppendF64 { b, v }
        | HirOp::StrBuilderAppendBool { b, v } => {
            substitute_value_types(b, param_map, type_ctx);
            substitute_value_types(v, param_map, type_ctx);
        }
        HirOp::JsonEncode { ty, v } => {
            *ty = substitute_type(*ty, param_map, type_ctx);
            substitute_value_types(v, param_map, type_ctx);
        }
        HirOp::JsonDecode { ty, s } => {
            *ty = substitute_type(*ty, param_map, type_ctx);
            substitute_value_types(s, param_map, type_ctx);
        }
        HirOp::GpuThreadId
        | HirOp::GpuWorkgroupId
        | HirOp::GpuWorkgroupSize
        | HirOp::GpuGlobalId => {}
        HirOp::GpuBufferLoad { buf, idx } => {
            substitute_value_types(buf, param_map, type_ctx);
            substitute_value_types(idx, param_map, type_ctx);
        }
        HirOp::GpuBufferLen { buf } => {
            substitute_value_types(buf, param_map, type_ctx);
        }
        HirOp::GpuShared { ty, size } => {
            *ty = substitute_type(*ty, param_map, type_ctx);
            substitute_value_types(size, param_map, type_ctx);
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
            substitute_value_types(device, param_map, type_ctx);
            substitute_value_types(groups, param_map, type_ctx);
            substitute_value_types(threads, param_map, type_ctx);
            for arg in args {
                substitute_value_types(arg, param_map, type_ctx);
            }
        }
    }
}

fn substitute_void_op_types(
    op: &mut HirOpVoid,
    param_map: &HashMap<TypeId, TypeId>,
    type_ctx: &mut TypeCtx,
) {
    match op {
        HirOpVoid::CallVoid { inst, args, .. } => {
            for ty in inst {
                *ty = substitute_type(*ty, param_map, type_ctx);
            }
            for arg in args {
                substitute_value_types(arg, param_map, type_ctx);
            }
        }
        HirOpVoid::CallVoidIndirect { callee, args } => {
            substitute_value_types(callee, param_map, type_ctx);
            for arg in args {
                substitute_value_types(arg, param_map, type_ctx);
            }
        }
        HirOpVoid::SetField { obj, value, .. } => {
            substitute_value_types(obj, param_map, type_ctx);
            substitute_value_types(value, param_map, type_ctx);
        }
        HirOpVoid::ArrSet { arr, idx, val } => {
            substitute_value_types(arr, param_map, type_ctx);
            substitute_value_types(idx, param_map, type_ctx);
            substitute_value_types(val, param_map, type_ctx);
        }
        HirOpVoid::ArrPush { arr, val } => {
            substitute_value_types(arr, param_map, type_ctx);
            substitute_value_types(val, param_map, type_ctx);
        }
        HirOpVoid::ArrSort { arr } => substitute_value_types(arr, param_map, type_ctx),
        HirOpVoid::ArrForeach { arr, func } => {
            substitute_value_types(arr, param_map, type_ctx);
            substitute_value_types(func, param_map, type_ctx);
        }
        HirOpVoid::MapSet { map, key, val } => {
            substitute_value_types(map, param_map, type_ctx);
            substitute_value_types(key, param_map, type_ctx);
            substitute_value_types(val, param_map, type_ctx);
        }
        HirOpVoid::MapDeleteVoid { map, key } => {
            substitute_value_types(map, param_map, type_ctx);
            substitute_value_types(key, param_map, type_ctx);
        }
        HirOpVoid::StrBuilderAppendStr { b, s } => {
            substitute_value_types(b, param_map, type_ctx);
            substitute_value_types(s, param_map, type_ctx);
        }
        HirOpVoid::StrBuilderAppendI64 { b, v }
        | HirOpVoid::StrBuilderAppendI32 { b, v }
        | HirOpVoid::StrBuilderAppendF64 { b, v }
        | HirOpVoid::StrBuilderAppendBool { b, v } => {
            substitute_value_types(b, param_map, type_ctx);
            substitute_value_types(v, param_map, type_ctx);
        }
        HirOpVoid::PtrStore { to, p, v } => {
            *to = substitute_type(*to, param_map, type_ctx);
            substitute_value_types(p, param_map, type_ctx);
            substitute_value_types(v, param_map, type_ctx);
        }
        HirOpVoid::Panic { msg } => substitute_value_types(msg, param_map, type_ctx),
        HirOpVoid::GpuBarrier => {}
        HirOpVoid::GpuBufferStore { buf, idx, val } => {
            substitute_value_types(buf, param_map, type_ctx);
            substitute_value_types(idx, param_map, type_ctx);
            substitute_value_types(val, param_map, type_ctx);
        }
    }
}

fn substitute_terminator_types(
    term: &mut HirTerminator,
    param_map: &HashMap<TypeId, TypeId>,
    type_ctx: &mut TypeCtx,
) {
    match term {
        HirTerminator::Ret(Some(v)) => substitute_value_types(v, param_map, type_ctx),
        HirTerminator::Ret(None) | HirTerminator::Br(_) | HirTerminator::Unreachable => {}
        HirTerminator::Cbr { cond, .. } => substitute_value_types(cond, param_map, type_ctx),
        HirTerminator::Switch { val, arms, .. } => {
            substitute_value_types(val, param_map, type_ctx);
            for (c, _) in arms {
                substitute_const_types(c, param_map, type_ctx);
            }
        }
    }
}

pub fn specialize_function(
    func: &HirFunction,
    type_args: &[TypeId],
    type_ctx: &mut TypeCtx,
) -> HirFunction {
    let mut specialized = func.clone();
    let param_map = type_param_map_for_specialization(func, type_args, type_ctx);
    substitute_function_types(&mut specialized, &param_map, type_ctx);

    let inst = instance_id(&func.sid, type_args, type_ctx);
    specialized.sid = specialized_sid(&func.sid, &inst);
    specialized.name = specialized_name(&func.name, &inst);
    specialized
}

pub fn substitute_type(
    ty: TypeId,
    param_map: &HashMap<TypeId, TypeId>,
    type_ctx: &mut TypeCtx,
) -> TypeId {
    if let Some(mapped) = param_map.get(&ty) {
        return *mapped;
    }

    let Some(kind) = type_ctx.lookup(ty).cloned() else {
        return ty;
    };

    let new_kind = match kind.clone() {
        TypeKind::Prim(_) | TypeKind::ValueStruct { .. } => {
            return ty;
        }
        TypeKind::HeapHandle { hk, base } => TypeKind::HeapHandle {
            hk,
            base: substitute_heap_base(base, param_map, type_ctx),
        },
        TypeKind::BuiltinOption { inner } => TypeKind::BuiltinOption {
            inner: substitute_type(inner, param_map, type_ctx),
        },
        TypeKind::BuiltinResult { ok, err } => TypeKind::BuiltinResult {
            ok: substitute_type(ok, param_map, type_ctx),
            err: substitute_type(err, param_map, type_ctx),
        },
        TypeKind::RawPtr { to } => TypeKind::RawPtr {
            to: substitute_type(to, param_map, type_ctx),
        },
        TypeKind::Arr { n, elem } => TypeKind::Arr {
            n,
            elem: substitute_type(elem, param_map, type_ctx),
        },
        TypeKind::Vec { n, elem } => TypeKind::Vec {
            n,
            elem: substitute_type(elem, param_map, type_ctx),
        },
        TypeKind::Tuple { elems } => TypeKind::Tuple {
            elems: elems
                .into_iter()
                .map(|e| substitute_type(e, param_map, type_ctx))
                .collect(),
        },
    };

    if new_kind == kind {
        ty
    } else {
        type_ctx.intern(new_kind)
    }
}

fn substitute_heap_base(
    base: HeapBase,
    param_map: &HashMap<TypeId, TypeId>,
    type_ctx: &mut TypeCtx,
) -> HeapBase {
    match base {
        HeapBase::BuiltinStr => HeapBase::BuiltinStr,
        HeapBase::BuiltinArray { elem } => HeapBase::BuiltinArray {
            elem: substitute_type(elem, param_map, type_ctx),
        },
        HeapBase::BuiltinMap { key, val } => HeapBase::BuiltinMap {
            key: substitute_type(key, param_map, type_ctx),
            val: substitute_type(val, param_map, type_ctx),
        },
        HeapBase::BuiltinStrBuilder => HeapBase::BuiltinStrBuilder,
        HeapBase::BuiltinMutex { inner } => HeapBase::BuiltinMutex {
            inner: substitute_type(inner, param_map, type_ctx),
        },
        HeapBase::BuiltinRwLock { inner } => HeapBase::BuiltinRwLock {
            inner: substitute_type(inner, param_map, type_ctx),
        },
        HeapBase::BuiltinCell { inner } => HeapBase::BuiltinCell {
            inner: substitute_type(inner, param_map, type_ctx),
        },
        HeapBase::BuiltinFuture { result } => HeapBase::BuiltinFuture {
            result: substitute_type(result, param_map, type_ctx),
        },
        HeapBase::BuiltinChannelSend { elem } => HeapBase::BuiltinChannelSend {
            elem: substitute_type(elem, param_map, type_ctx),
        },
        HeapBase::BuiltinChannelRecv { elem } => HeapBase::BuiltinChannelRecv {
            elem: substitute_type(elem, param_map, type_ctx),
        },
        HeapBase::Callable { sig_sid } => HeapBase::Callable { sig_sid },
        HeapBase::UserType { type_sid, targs } => HeapBase::UserType {
            type_sid,
            targs: targs
                .into_iter()
                .map(|t| substitute_type(t, param_map, type_ctx))
                .collect(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magpie_types::fixed_type_ids;

    #[test]
    fn collect_generic_call_keys_captures_all_generic_call_sites() {
        let generic_call_key = InstanceKey {
            callee_sid: Sid("F:callee".to_string()),
            type_args: vec![fixed_type_ids::I32],
        };
        let suspend_key = InstanceKey {
            callee_sid: Sid("F:async_callee".to_string()),
            type_args: vec![fixed_type_ids::I64, fixed_type_ids::U64],
        };
        let void_key = InstanceKey {
            callee_sid: Sid("F:void_callee".to_string()),
            type_args: vec![fixed_type_ids::BOOL],
        };

        let func = HirFunction {
            fn_id: FnId(0),
            sid: Sid("F:test".to_string()),
            name: "test".to_string(),
            params: Vec::new(),
            ret_ty: fixed_type_ids::UNIT,
            blocks: vec![HirBlock {
                id: magpie_hir::BlockId(0),
                instrs: vec![
                    HirInstr {
                        dst: magpie_hir::LocalId(1),
                        ty: fixed_type_ids::I32,
                        op: HirOp::Call {
                            callee_sid: generic_call_key.callee_sid.clone(),
                            inst: generic_call_key.type_args.clone(),
                            args: Vec::new(),
                        },
                    },
                    HirInstr {
                        dst: magpie_hir::LocalId(2),
                        ty: fixed_type_ids::I32,
                        op: HirOp::Call {
                            callee_sid: generic_call_key.callee_sid.clone(),
                            inst: generic_call_key.type_args.clone(),
                            args: Vec::new(),
                        },
                    },
                    HirInstr {
                        dst: magpie_hir::LocalId(3),
                        ty: fixed_type_ids::UNIT,
                        op: HirOp::SuspendCall {
                            callee_sid: suspend_key.callee_sid.clone(),
                            inst: suspend_key.type_args.clone(),
                            args: Vec::new(),
                        },
                    },
                    HirInstr {
                        dst: magpie_hir::LocalId(4),
                        ty: fixed_type_ids::I32,
                        op: HirOp::Call {
                            callee_sid: Sid("F:non_generic".to_string()),
                            inst: Vec::new(),
                            args: Vec::new(),
                        },
                    },
                ],
                void_ops: vec![HirOpVoid::CallVoid {
                    callee_sid: void_key.callee_sid.clone(),
                    inst: void_key.type_args.clone(),
                    args: Vec::new(),
                }],
                terminator: HirTerminator::Ret(None),
            }],
            is_async: false,
            is_unsafe: false,
        };

        let mut out = HashSet::new();
        collect_generic_call_keys(&func, &mut out);

        assert_eq!(out.len(), 3, "duplicate call keys should collapse");
        assert!(out.contains(&generic_call_key));
        assert!(out.contains(&suspend_key));
        assert!(out.contains(&void_key));
    }

    #[test]
    fn instance_id_and_specialized_ids_are_deterministic() {
        let type_ctx = TypeCtx::new();
        let sid = Sid("F:generic.add".to_string());

        let inst_a = instance_id(&sid, &[fixed_type_ids::I32], &type_ctx);
        let inst_a_again = instance_id(&sid, &[fixed_type_ids::I32], &type_ctx);
        let inst_b = instance_id(&sid, &[fixed_type_ids::I64], &type_ctx);

        assert_eq!(inst_a, inst_a_again);
        assert_ne!(inst_a, inst_b);
        assert!(inst_a.starts_with("I:"));
        assert_eq!(inst_a.len(), 18);

        let specialized_sid_value = specialized_sid(&sid, &inst_a);
        let specialized_name_value = specialized_name("generic.add", &inst_a);
        assert_eq!(
            specialized_sid_value.0,
            format!("{}$I${}", sid.0, inst_a.trim_start_matches("I:"))
        );
        assert_eq!(
            specialized_name_value,
            format!("generic.add$I${}", inst_a.trim_start_matches("I:"))
        );
    }
}
