//! magpie_codegen_wasm

use magpie_mpir::{MpirModule, MpirOp, MpirValue};
use magpie_types::{fixed_type_ids, TypeId};
use std::collections::HashSet;

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
