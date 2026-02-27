//! magpie_codegen_wasm

use magpie_mpir::{HirConstLit, MpirFn, MpirModule, MpirOp, MpirOpVoid, MpirTerminator, MpirValue};
use magpie_types::{fixed_type_ids, TypeId};
use std::collections::{HashMap, HashSet};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MpRtHeader32 {
    pub strong: u32,
    pub weak: u32,
    pub type_id: u32,
    pub flags: u32,
}

const _: () = {
    assert!(std::mem::size_of::<MpRtHeader32>() == 16);
};

/// Rewrites pointer-address carrier integers from 64-bit to 32-bit for wasm32.
pub fn adjust_module_for_wasm32(module: &mut MpirModule) {
    for func in &mut module.functions {
        let mut pointer_addr_locals = HashSet::new();

        for block in &func.blocks {
            for instr in &block.instrs {
                if matches!(instr.op, MpirOp::PtrAddr { .. }) {
                    pointer_addr_locals.insert(instr.dst);
                }

                if let MpirOp::PtrFromAddr { addr, .. } = &instr.op {
                    if let MpirValue::Local(local) = addr {
                        pointer_addr_locals.insert(*local);
                    }
                }
            }
        }

        if pointer_addr_locals.is_empty() {
            continue;
        }

        for (param_local, param_ty) in &mut func.params {
            if pointer_addr_locals.contains(param_local) {
                if let Some(rewritten) = wasm32_pointer_int_ty(*param_ty) {
                    *param_ty = rewritten;
                }
            }
        }

        for local in &mut func.locals {
            if pointer_addr_locals.contains(&local.id) {
                if let Some(rewritten) = wasm32_pointer_int_ty(local.ty) {
                    local.ty = rewritten;
                }
            }
        }

        for block in &mut func.blocks {
            for instr in &mut block.instrs {
                if pointer_addr_locals.contains(&instr.dst) {
                    if let Some(rewritten) = wasm32_pointer_int_ty(instr.ty) {
                        instr.ty = rewritten;
                    }

                    if let MpirOp::Const(c) = &mut instr.op {
                        if let Some(rewritten) = wasm32_pointer_int_ty(c.ty) {
                            c.ty = rewritten;
                        }
                    }
                }

                match &mut instr.op {
                    MpirOp::PtrFromAddr { addr, .. } => rewrite_pointer_value(addr),
                    MpirOp::PtrAdd { count, .. } => rewrite_pointer_value(count),
                    _ => {}
                }
            }
        }
    }
}

pub fn generate_wasm_runtime_imports() -> String {
    [
        r#"(import \"magpie_rt\" \"mp_rt_init\" (func $mp_rt_init))"#,
        r#"(import \"magpie_rt\" \"mp_rt_retain_strong\" (func $mp_rt_retain_strong (param i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_release_strong\" (func $mp_rt_release_strong (param i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_retain_weak\" (func $mp_rt_retain_weak (param i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_release_weak\" (func $mp_rt_release_weak (param i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_weak_upgrade\" (func $mp_rt_weak_upgrade (param i32) (result i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_panic\" (func $mp_rt_panic (param i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_arr_new\" (func $mp_rt_arr_new (param i32 i64 i64) (result i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_arr_len\" (func $mp_rt_arr_len (param i32) (result i64)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_arr_get\" (func $mp_rt_arr_get (param i32 i64) (result i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_arr_set\" (func $mp_rt_arr_set (param i32 i64 i32 i64)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_arr_push\" (func $mp_rt_arr_push (param i32 i32 i64)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_arr_pop\" (func $mp_rt_arr_pop (param i32 i32 i64) (result i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_arr_slice\" (func $mp_rt_arr_slice (param i32 i64 i64) (result i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_arr_contains\" (func $mp_rt_arr_contains (param i32 i32 i64 i32) (result i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_arr_sort\" (func $mp_rt_arr_sort (param i32 i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_map_new\" (func $mp_rt_map_new (param i32 i32 i64 i64 i64 i32 i32) (result i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_map_len\" (func $mp_rt_map_len (param i32) (result i64)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_map_get\" (func $mp_rt_map_get (param i32 i32 i64) (result i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_map_set\" (func $mp_rt_map_set (param i32 i32 i64 i32 i64)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_map_take\" (func $mp_rt_map_take (param i32 i32 i64 i32 i64) (result i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_map_delete\" (func $mp_rt_map_delete (param i32 i32 i64) (result i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_map_contains_key\" (func $mp_rt_map_contains_key (param i32 i32 i64) (result i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_map_keys\" (func $mp_rt_map_keys (param i32) (result i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_map_values\" (func $mp_rt_map_values (param i32) (result i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_str_concat\" (func $mp_rt_str_concat (param i32 i32) (result i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_str_len\" (func $mp_rt_str_len (param i32) (result i64)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_str_eq\" (func $mp_rt_str_eq (param i32 i32) (result i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_str_slice\" (func $mp_rt_str_slice (param i32 i64 i64) (result i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_str_bytes\" (func $mp_rt_str_bytes (param i32 i32) (result i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_strbuilder_new\" (func $mp_rt_strbuilder_new (result i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_strbuilder_append_str\" (func $mp_rt_strbuilder_append_str (param i32 i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_strbuilder_append_i64\" (func $mp_rt_strbuilder_append_i64 (param i32 i64)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_strbuilder_append_i32\" (func $mp_rt_strbuilder_append_i32 (param i32 i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_strbuilder_append_f64\" (func $mp_rt_strbuilder_append_f64 (param i32 f64)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_strbuilder_append_bool\" (func $mp_rt_strbuilder_append_bool (param i32 i32)))"#,
        r#"(import \"magpie_rt\" \"mp_rt_strbuilder_build\" (func $mp_rt_strbuilder_build (param i32) (result i32)))"#,
    ]
    .join("\n")
}

/// Emit a minimal WAT module from MPIR. The output can be assembled by `wat2wasm`.
pub fn generate_wat(module: &MpirModule) -> String {
    emit_wat_module(module)
}

/// Emit a minimal WAT module from MPIR. The output can be assembled by `wat2wasm`.
pub fn emit_wat_module(module: &MpirModule) -> String {
    let mut out = String::from("(module\n");
    let fn_symbols = function_symbol_map(module);

    for (idx, func) in module.functions.iter().enumerate() {
        emit_wat_function(func, idx, &fn_symbols, &mut out);
    }

    out.push_str(")\n");
    out
}

fn emit_wat_function(
    func: &MpirFn,
    idx: usize,
    fn_symbols: &HashMap<String, String>,
    out: &mut String,
) {
    let fn_symbol = fn_symbols
        .get(&func.sid.0)
        .cloned()
        .unwrap_or_else(|| fallback_function_symbol(func, idx));
    out.push_str(&format!("  (func ${fn_symbol}"));

    let mut local_types: HashMap<u32, TypeId> = HashMap::new();
    for (local, ty) in &func.params {
        local_types.insert(local.0, *ty);
        out.push_str(&format!(" (param {} {})", wat_local(*local), wasm_ty(*ty)));
    }

    if func.ret_ty != fixed_type_ids::UNIT {
        out.push_str(&format!(" (result {})", wasm_ty(func.ret_ty)));
    }

    let mut declared = HashSet::new();
    for (local, _) in &func.params {
        declared.insert(local.0);
    }

    for local in &func.locals {
        local_types.insert(local.id.0, local.ty);
        if declared.insert(local.id.0) {
            out.push_str(&format!(" (local {} {})", wat_local(local.id), wasm_ty(local.ty)));
        }
    }

    for block in &func.blocks {
        for instr in &block.instrs {
            local_types.insert(instr.dst.0, instr.ty);
            if declared.insert(instr.dst.0) {
                out.push_str(&format!(
                    " (local {} {})",
                    wat_local(instr.dst),
                    wasm_ty(instr.ty)
                ));
            }
        }
    }

    out.push('\n');

    for block in &func.blocks {
        for instr in &block.instrs {
            emit_wat_instr(instr, fn_symbols, &local_types, out);
        }
        for op in &block.void_ops {
            emit_wat_void_op(op, fn_symbols, &local_types, out);
        }
        emit_wat_terminator(&block.terminator, func.ret_ty, &local_types, out);
    }

    out.push_str("  )\n");
}

fn emit_wat_instr(
    instr: &magpie_mpir::MpirInstr,
    fn_symbols: &HashMap<String, String>,
    local_types: &HashMap<u32, TypeId>,
    out: &mut String,
) {
    match &instr.op {
        MpirOp::Const(c) => {
            emit_wat_const(&c.lit, instr.ty, out);
            out.push_str(&format!("    local.set {}\n", wat_local(instr.dst)));
        }
        MpirOp::Move { v }
        | MpirOp::BorrowShared { v }
        | MpirOp::BorrowMut { v }
        | MpirOp::Share { v }
        | MpirOp::CloneShared { v }
        | MpirOp::CloneWeak { v }
        | MpirOp::WeakDowngrade { v }
        | MpirOp::WeakUpgrade { v } => {
            emit_wat_value(v, local_types, out);
            out.push_str(&format!("    local.set {}\n", wat_local(instr.dst)));
        }
        MpirOp::IAdd { lhs, rhs } | MpirOp::IAddWrap { lhs, rhs } => {
            emit_wat_value(lhs, local_types, out);
            emit_wat_value(rhs, local_types, out);
            out.push_str("    i32.add\n");
            out.push_str(&format!("    local.set {}\n", wat_local(instr.dst)));
        }
        MpirOp::ISub { lhs, rhs } | MpirOp::ISubWrap { lhs, rhs } => {
            emit_wat_value(lhs, local_types, out);
            emit_wat_value(rhs, local_types, out);
            out.push_str("    i32.sub\n");
            out.push_str(&format!("    local.set {}\n", wat_local(instr.dst)));
        }
        MpirOp::IMul { lhs, rhs } | MpirOp::IMulWrap { lhs, rhs } => {
            emit_wat_value(lhs, local_types, out);
            emit_wat_value(rhs, local_types, out);
            out.push_str("    i32.mul\n");
            out.push_str(&format!("    local.set {}\n", wat_local(instr.dst)));
        }
        MpirOp::Call {
            callee_sid, args, ..
        }
        | MpirOp::SuspendCall {
            callee_sid, args, ..
        } => {
            for arg in args {
                emit_wat_value(arg, local_types, out);
            }
            let callee = fn_symbols
                .get(&callee_sid.0)
                .cloned()
                .unwrap_or_else(|| sanitize_wat_ident(&callee_sid.0));
            out.push_str(&format!("    call ${callee}\n"));
            if instr.ty != fixed_type_ids::UNIT {
                out.push_str(&format!("    local.set {}\n", wat_local(instr.dst)));
            } else {
                out.push_str("    i32.const 0\n");
                out.push_str(&format!("    local.set {}\n", wat_local(instr.dst)));
            }
        }
        _ => {
            emit_wat_default(instr.ty, out);
            out.push_str(&format!("    local.set {}\n", wat_local(instr.dst)));
        }
    }
}

fn emit_wat_void_op(
    op: &MpirOpVoid,
    fn_symbols: &HashMap<String, String>,
    local_types: &HashMap<u32, TypeId>,
    out: &mut String,
) {
    if let MpirOpVoid::CallVoid {
        callee_sid, args, ..
    } = op
    {
        for arg in args {
            emit_wat_value(arg, local_types, out);
        }
        let callee = fn_symbols
            .get(&callee_sid.0)
            .cloned()
            .unwrap_or_else(|| sanitize_wat_ident(&callee_sid.0));
        out.push_str(&format!("    call ${callee}\n"));
    }
}

fn emit_wat_terminator(
    term: &MpirTerminator,
    ret_ty: TypeId,
    local_types: &HashMap<u32, TypeId>,
    out: &mut String,
) {
    match term {
        MpirTerminator::Ret(Some(v)) => {
            emit_wat_value(v, local_types, out);
            out.push_str("    return\n");
        }
        MpirTerminator::Ret(None) => {
            out.push_str("    return\n");
        }
        MpirTerminator::Br(_) | MpirTerminator::Cbr { .. } | MpirTerminator::Switch { .. } => {
            if ret_ty != fixed_type_ids::UNIT {
                emit_wat_default(ret_ty, out);
            }
            out.push_str("    return\n");
        }
        MpirTerminator::Unreachable => {
            if ret_ty != fixed_type_ids::UNIT {
                emit_wat_default(ret_ty, out);
            }
            out.push_str("    return\n");
        }
    }
}

fn emit_wat_value(value: &MpirValue, _local_types: &HashMap<u32, TypeId>, out: &mut String) {
    match value {
        MpirValue::Local(local) => {
            out.push_str(&format!("    local.get {}\n", wat_local(*local)));
        }
        MpirValue::Const(c) => {
            emit_wat_const(&c.lit, c.ty, out);
        }
    }
}

fn emit_wat_const(lit: &HirConstLit, ty: TypeId, out: &mut String) {
    match wasm_ty(ty) {
        "i64" => {
            let n = match lit {
                HirConstLit::IntLit(v) => *v as i64,
                HirConstLit::BoolLit(v) => {
                    if *v {
                        1
                    } else {
                        0
                    }
                }
                HirConstLit::FloatLit(v) => *v as i64,
                HirConstLit::StringLit(_) | HirConstLit::Unit => 0,
            };
            out.push_str(&format!("    i64.const {n}\n"));
        }
        "f32" => {
            let n = match lit {
                HirConstLit::FloatLit(v) => *v as f32,
                HirConstLit::IntLit(v) => *v as f32,
                HirConstLit::BoolLit(v) => {
                    if *v {
                        1.0
                    } else {
                        0.0
                    }
                }
                HirConstLit::StringLit(_) | HirConstLit::Unit => 0.0,
            };
            out.push_str(&format!("    f32.const {n}\n"));
        }
        "f64" => {
            let n = match lit {
                HirConstLit::FloatLit(v) => *v,
                HirConstLit::IntLit(v) => *v as f64,
                HirConstLit::BoolLit(v) => {
                    if *v {
                        1.0
                    } else {
                        0.0
                    }
                }
                HirConstLit::StringLit(_) | HirConstLit::Unit => 0.0,
            };
            out.push_str(&format!("    f64.const {n}\n"));
        }
        _ => {
            let n = match lit {
                HirConstLit::IntLit(v) => *v as i32,
                HirConstLit::BoolLit(v) => {
                    if *v {
                        1
                    } else {
                        0
                    }
                }
                HirConstLit::FloatLit(v) => *v as i32,
                HirConstLit::StringLit(_) | HirConstLit::Unit => 0,
            };
            out.push_str(&format!("    i32.const {n}\n"));
        }
    }
}

fn emit_wat_default(ty: TypeId, out: &mut String) {
    match wasm_ty(ty) {
        "i64" => out.push_str("    i64.const 0\n"),
        "f32" => out.push_str("    f32.const 0\n"),
        "f64" => out.push_str("    f64.const 0\n"),
        _ => out.push_str("    i32.const 0\n"),
    }
}

fn wasm_ty(ty: TypeId) -> &'static str {
    match ty {
        fixed_type_ids::I64 | fixed_type_ids::U64 => "i64",
        fixed_type_ids::F16 | fixed_type_ids::F32 => "f32",
        fixed_type_ids::F64 => "f64",
        _ => "i32",
    }
}

fn wat_local(local: magpie_mpir::LocalId) -> String {
    format!("$l{}", local.0)
}

fn function_symbol_map(module: &MpirModule) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for (idx, func) in module.functions.iter().enumerate() {
        out.insert(func.sid.0.clone(), fallback_function_symbol(func, idx));
    }
    out
}

fn fallback_function_symbol(func: &MpirFn, idx: usize) -> String {
    format!("f{idx}_{}", sanitize_wat_ident(&func.name))
}

fn sanitize_wat_ident(name: &str) -> String {
    let mut out = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push('f');
    }
    out
}

pub fn is_wasm_target(triple: &str) -> bool {
    let arch = triple
        .split('-')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    arch.starts_with("wasm")
}

fn wasm32_pointer_int_ty(ty: TypeId) -> Option<TypeId> {
    match ty {
        fixed_type_ids::I64 => Some(fixed_type_ids::I32),
        fixed_type_ids::U64 => Some(fixed_type_ids::U32),
        _ => None,
    }
}

fn rewrite_pointer_value(value: &mut MpirValue) {
    if let MpirValue::Const(c) = value {
        if let Some(rewritten) = wasm32_pointer_int_ty(c.ty) {
            c.ty = rewritten;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magpie_mpir::{MpirBlock, MpirFn, MpirInstr, MpirTerminator, MpirTypeTable};
    use magpie_types::{BlockId, LocalId, Sid};

    #[test]
    fn emit_simple_wat_output() {
        let module = MpirModule {
            sid: Sid("M:WATTEST0000".to_string()),
            path: "test.mag".to_string(),
            type_table: MpirTypeTable { types: vec![] },
            functions: vec![MpirFn {
                sid: Sid("F:WATADD00000".to_string()),
                name: "add".to_string(),
                params: vec![
                    (LocalId(0), fixed_type_ids::I32),
                    (LocalId(1), fixed_type_ids::I32),
                ],
                ret_ty: fixed_type_ids::I32,
                blocks: vec![MpirBlock {
                    id: BlockId(0),
                    instrs: vec![MpirInstr {
                        dst: LocalId(2),
                        ty: fixed_type_ids::I32,
                        op: MpirOp::IAdd {
                            lhs: MpirValue::Local(LocalId(0)),
                            rhs: MpirValue::Local(LocalId(1)),
                        },
                    }],
                    void_ops: vec![],
                    terminator: MpirTerminator::Ret(Some(MpirValue::Local(LocalId(2)))),
                }],
                locals: vec![],
                is_async: false,
            }],
            globals: vec![],
        };

        let wat = emit_wat_module(&module);
        assert!(wat.contains("(module"));
        assert!(wat.contains("(func $f0_add"));
        assert!(wat.contains("(param $l0 i32)"));
        assert!(wat.contains("(param $l1 i32)"));
        assert!(wat.contains("(result i32)"));
        assert!(wat.contains("local.get $l0"));
        assert!(wat.contains("local.get $l1"));
        assert!(wat.contains("i32.add"));
        assert!(wat.contains("return"));
    }
}
