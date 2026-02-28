//! magpie_codegen_llvm
#![allow(clippy::too_many_arguments, clippy::write_with_newline)]

use magpie_mpir::{
    HirConst, HirConstLit, MpirBlock, MpirFn, MpirInstr, MpirModule, MpirOp, MpirOpVoid,
    MpirTerminator, MpirValue,
};
use magpie_types::{fixed_type_ids, HeapBase, PrimType, Sid, TypeCtx, TypeId, TypeKind};
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;

const MP_RT_HEADER_SIZE: u64 = 32;
const MP_RT_FLAG_HEAP: u32 = 0x1;

#[derive(Copy, Clone, Debug, Default)]
pub struct CodegenOptions {
    pub shared_generics: bool,
}

#[derive(Clone, Debug)]
struct RuntimeTypeRegistryInfo {
    symbol: String,
    count: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct CallbackUsage {
    hash: bool,
    eq: bool,
    cmp: bool,
}

pub fn codegen_module(mpir: &MpirModule, type_ctx: &TypeCtx) -> Result<String, String> {
    codegen_module_with_options(mpir, type_ctx, CodegenOptions::default())
}

pub fn codegen_module_with_options(
    mpir: &MpirModule,
    type_ctx: &TypeCtx,
    options: CodegenOptions,
) -> Result<String, String> {
    let mut cg = LlvmTextCodegen::new(mpir, type_ctx, options);
    cg.codegen_module()
}

struct LlvmTextCodegen<'a> {
    mpir: &'a MpirModule,
    type_ctx: &'a TypeCtx,
    type_map: BTreeMap<u32, TypeKind>,
    options: CodegenOptions,
}

impl<'a> LlvmTextCodegen<'a> {
    fn new(mpir: &'a MpirModule, type_ctx: &'a TypeCtx, options: CodegenOptions) -> Self {
        let mut type_map = BTreeMap::new();
        for (tid, kind) in &mpir.type_table.types {
            type_map.insert(tid.0, kind.clone());
        }
        for (tid, kind) in &type_ctx.types {
            type_map.entry(tid.0).or_insert_with(|| kind.clone());
        }
        Self {
            mpir,
            type_ctx,
            type_map,
            options,
        }
    }

    fn codegen_module(&mut self) -> Result<String, String> {
        let mut out = String::new();
        let module_init = mangle_init_types(&self.mpir.sid);
        let main_fn = self
            .mpir
            .functions
            .iter()
            .find(|f| f.name == "main" || f.name == "@main");

        writeln!(out, "; ModuleID = '{}'", llvm_quote(&self.mpir.path))
            .map_err(|e| e.to_string())?;
        writeln!(out, "source_filename = \"{}\"", llvm_quote(&self.mpir.path))
            .map_err(|e| e.to_string())?;
        writeln!(out).map_err(|e| e.to_string())?;

        let value_struct_ids = self
            .type_map
            .iter()
            .filter_map(|(id, kind)| matches!(kind, TypeKind::ValueStruct { .. }).then_some(*id))
            .collect::<Vec<_>>();
        for id in value_struct_ids {
            let layout = self.type_ctx.compute_layout(TypeId(id));
            let size = layout.size.max(1);
            writeln!(out, "%mp_t{} = type [{} x i8]", id, size).map_err(|e| e.to_string())?;
        }
        let generics_mode = if self.options.shared_generics {
            1_u8
        } else {
            0_u8
        };
        writeln!(
            out,
            "@\"mp$0$ABI$generics_mode\" = weak_odr constant i8 {generics_mode}"
        )
        .map_err(|e| e.to_string())?;
        if !out.ends_with('\n') {
            writeln!(out).map_err(|e| e.to_string())?;
        }

        self.emit_declarations(&mut out)?;
        writeln!(out).map_err(|e| e.to_string())?;
        let runtime_registry = self.emit_runtime_type_registry_globals(&mut out)?;
        if runtime_registry.is_some() {
            writeln!(out).map_err(|e| e.to_string())?;
        }
        self.emit_trait_callback_wrappers(&mut out)?;
        writeln!(out).map_err(|e| e.to_string())?;

        writeln!(out, "define internal void @{}() {{", module_init).map_err(|e| e.to_string())?;
        writeln!(out, "entry:").map_err(|e| e.to_string())?;
        if let Some(reg) = runtime_registry {
            writeln!(
                out,
                "  call void @mp_rt_register_types(ptr getelementptr inbounds ([{} x %MpRtTypeInfo], ptr @{}, i64 0, i64 0), i32 {})",
                reg.count, reg.symbol, reg.count
            )
            .map_err(|e| e.to_string())?;
        }
        writeln!(out, "  ret void").map_err(|e| e.to_string())?;
        writeln!(out, "}}").map_err(|e| e.to_string())?;
        writeln!(out).map_err(|e| e.to_string())?;

        for f in &self.mpir.functions {
            let body = self.codegen_fn(f)?;
            out.push_str(&body);
            out.push('\n');
        }

        if main_fn.is_some() {
            self.emit_c_main(&mut out, module_init.as_str(), main_fn)?;
        }
        Ok(out)
    }

    fn emit_declarations(&self, out: &mut String) -> Result<(), String> {
        let decls = [
            "declare void @mp_rt_init()",
            "declare void @mp_gpu_register_all_kernels()",
            "declare ptr @mp_rt_alloc(i32, i64, i64, i32)",
            "declare void @mp_rt_register_types(ptr, i32)",
            "declare void @mp_rt_retain_strong(ptr)",
            "declare void @mp_rt_release_strong(ptr)",
            "declare void @mp_rt_retain_weak(ptr)",
            "declare void @mp_rt_release_weak(ptr)",
            "declare ptr @mp_rt_weak_upgrade(ptr)",
            "declare void @mp_rt_panic(ptr) noreturn",
            "declare ptr @mp_rt_arr_new(i32, i64, i64)",
            "declare i64 @mp_rt_arr_len(ptr)",
            "declare ptr @mp_rt_arr_get(ptr, i64)",
            "declare void @mp_rt_arr_set(ptr, i64, ptr, i64)",
            "declare void @mp_rt_arr_push(ptr, ptr, i64)",
            "declare i32 @mp_rt_arr_pop(ptr, ptr, i64)",
            "declare ptr @mp_rt_arr_slice(ptr, i64, i64)",
            "declare i32 @mp_rt_arr_contains(ptr, ptr, i64, ptr)",
            "declare void @mp_rt_arr_sort(ptr, ptr)",
            "declare void @mp_rt_arr_foreach(ptr, ptr)",
            "declare ptr @mp_rt_arr_map(ptr, ptr, i32, i64)",
            "declare ptr @mp_rt_arr_filter(ptr, ptr)",
            "declare void @mp_rt_arr_reduce(ptr, ptr, i64, ptr)",
            "declare ptr @mp_rt_callable_new(ptr, ptr)",
            "declare ptr @mp_rt_callable_fn_ptr(ptr)",
            "declare ptr @mp_rt_callable_data_ptr(ptr)",
            "declare i64 @mp_rt_callable_capture_size(ptr)",
            "declare ptr @mp_rt_map_new(i32, i32, i64, i64, i64, ptr, ptr)",
            "declare i64 @mp_rt_map_len(ptr)",
            "declare ptr @mp_rt_map_get(ptr, ptr, i64)",
            "declare void @mp_rt_map_set(ptr, ptr, i64, ptr, i64)",
            "declare i32 @mp_rt_map_take(ptr, ptr, i64, ptr, i64)",
            "declare i32 @mp_rt_map_delete(ptr, ptr, i64)",
            "declare i32 @mp_rt_map_contains_key(ptr, ptr, i64)",
            "declare ptr @mp_rt_map_keys(ptr)",
            "declare ptr @mp_rt_map_values(ptr)",
            "declare ptr @mp_rt_str_concat(ptr, ptr)",
            "declare i64 @mp_rt_str_len(ptr)",
            "declare i32 @mp_rt_str_eq(ptr, ptr)",
            "declare i32 @mp_rt_str_cmp(ptr, ptr)",
            "declare ptr @mp_rt_str_slice(ptr, i64, i64)",
            "declare ptr @mp_rt_str_bytes(ptr, ptr)",
            "declare ptr @mp_rt_str_from_utf8(ptr, i64)",
            "declare i64 @mp_std_hash_str(ptr)",
            "declare i64 @mp_rt_bytes_hash(ptr, i64)",
            "declare i32 @mp_rt_bytes_eq(ptr, ptr, i64)",
            "declare i32 @mp_rt_bytes_cmp(ptr, ptr, i64)",
            "declare ptr @mp_rt_json_encode(ptr, i32)",
            "declare ptr @mp_rt_json_decode(ptr, i32)",
            "declare i64 @mp_rt_str_parse_i64(ptr)",
            "declare i64 @mp_rt_str_parse_u64(ptr)",
            "declare double @mp_rt_str_parse_f64(ptr)",
            "declare i32 @mp_rt_str_parse_bool(ptr)",
            "declare ptr @mp_rt_strbuilder_new()",
            "declare void @mp_rt_strbuilder_append_str(ptr, ptr)",
            "declare void @mp_rt_strbuilder_append_i64(ptr, i64)",
            "declare void @mp_rt_strbuilder_append_i32(ptr, i32)",
            "declare void @mp_rt_strbuilder_append_f64(ptr, double)",
            "declare void @mp_rt_strbuilder_append_bool(ptr, i32)",
            "declare ptr @mp_rt_strbuilder_build(ptr)",
            "declare i32 @mp_rt_future_poll(ptr)",
            "declare void @mp_rt_future_take(ptr, ptr)",
            "declare i64 @mp_rt_gpu_buffer_len(ptr)",
            "declare i32 @mp_rt_gpu_buffer_read(ptr, i64, ptr, i64)",
            "declare i32 @mp_rt_gpu_buffer_write(ptr, i64, ptr, i64)",
            "declare i32 @mp_rt_gpu_launch_sync(ptr, i64, i32, i32, i32, i32, i32, i32, ptr, i64, ptr)",
            "declare i32 @mp_rt_gpu_launch_async(ptr, i64, i32, i32, i32, i32, i32, i32, ptr, i64, ptr, ptr)",
            "declare { i8, i1 } @llvm.sadd.with.overflow.i8(i8, i8)",
            "declare { i8, i1 } @llvm.ssub.with.overflow.i8(i8, i8)",
            "declare { i8, i1 } @llvm.smul.with.overflow.i8(i8, i8)",
            "declare { i16, i1 } @llvm.sadd.with.overflow.i16(i16, i16)",
            "declare { i16, i1 } @llvm.ssub.with.overflow.i16(i16, i16)",
            "declare { i16, i1 } @llvm.smul.with.overflow.i16(i16, i16)",
            "declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32)",
            "declare { i32, i1 } @llvm.ssub.with.overflow.i32(i32, i32)",
            "declare { i32, i1 } @llvm.smul.with.overflow.i32(i32, i32)",
            "declare { i64, i1 } @llvm.sadd.with.overflow.i64(i64, i64)",
            "declare { i64, i1 } @llvm.ssub.with.overflow.i64(i64, i64)",
            "declare { i64, i1 } @llvm.smul.with.overflow.i64(i64, i64)",
            "declare { i128, i1 } @llvm.sadd.with.overflow.i128(i128, i128)",
            "declare { i128, i1 } @llvm.ssub.with.overflow.i128(i128, i128)",
            "declare { i128, i1 } @llvm.smul.with.overflow.i128(i128, i128)",
        ];
        for d in decls {
            writeln!(out, "{d}").map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn emit_runtime_type_registry_globals(
        &self,
        out: &mut String,
    ) -> Result<Option<RuntimeTypeRegistryInfo>, String> {
        #[derive(Clone, Debug)]
        struct Entry {
            type_id: u32,
            payload_size: u64,
            payload_align: u64,
            debug_fqn: String,
        }

        let mut entries = Vec::new();
        for (type_id, kind) in &self.type_map {
            let Some((type_sid, _)) = (match kind {
                TypeKind::HeapHandle {
                    base: HeapBase::UserType { type_sid, targs },
                    ..
                } => Some((type_sid.clone(), targs)),
                _ => None,
            }) else {
                continue;
            };

            let layout = self
                .type_ctx
                .user_enum_layout(&type_sid)
                .or_else(|| self.type_ctx.user_struct_layout(&type_sid))
                .unwrap_or(magpie_types::TypeLayout {
                    size: 0,
                    align: 1,
                    fields: Vec::new(),
                });

            entries.push(Entry {
                type_id: *type_id,
                payload_size: layout.size.max(1),
                payload_align: layout.align.max(1),
                debug_fqn: self.type_ctx.type_str(TypeId(*type_id)),
            });
        }

        if entries.is_empty() {
            return Ok(None);
        }

        writeln!(
            out,
            "%MpRtTypeInfo = type {{ i32, i32, i64, i64, ptr, ptr }}"
        )
        .map_err(|e| e.to_string())?;

        let mut name_symbols = Vec::with_capacity(entries.len());
        for (idx, entry) in entries.iter().enumerate() {
            let (bytes_len, literal) = llvm_c_string_literal(&entry.debug_fqn);
            let sym = format!("mp_type_debug_name_{}", idx);
            writeln!(
                out,
                "@{sym} = private constant [{bytes_len} x i8] c\"{literal}\""
            )
            .map_err(|e| e.to_string())?;
            name_symbols.push((sym, bytes_len));
        }

        let type_entries = entries
            .iter()
            .enumerate()
            .map(|(idx, entry)| {
                let (name_sym, name_len) = &name_symbols[idx];
                format!(
                    "%MpRtTypeInfo {{ i32 {}, i32 {}, i64 {}, i64 {}, ptr null, ptr getelementptr inbounds ([{} x i8], ptr @{}, i64 0, i64 0) }}",
                    entry.type_id,
                    MP_RT_FLAG_HEAP,
                    entry.payload_size,
                    entry.payload_align,
                    name_len,
                    name_sym
                )
            })
            .collect::<Vec<_>>()
            .join(", ");

        let symbol = format!("mp_type_registry_{}", sid_suffix(&self.mpir.sid));
        writeln!(
            out,
            "@{symbol} = private constant [{} x %MpRtTypeInfo] [{}]",
            entries.len(),
            type_entries
        )
        .map_err(|e| e.to_string())?;

        Ok(Some(RuntimeTypeRegistryInfo {
            symbol,
            count: entries.len(),
        }))
    }

    fn emit_trait_callback_wrappers(&self, out: &mut String) -> Result<(), String> {
        let usage = self.collect_trait_callback_usage();
        if usage.is_empty() {
            return Ok(());
        }

        for (tid, flags) in usage {
            let ty = TypeId(tid);
            if flags.hash {
                self.emit_hash_wrapper(out, ty)?;
                writeln!(out).map_err(|e| e.to_string())?;
            }
            if flags.eq {
                self.emit_eq_wrapper(out, ty)?;
                writeln!(out).map_err(|e| e.to_string())?;
            }
            if flags.cmp {
                self.emit_cmp_wrapper(out, ty)?;
                writeln!(out).map_err(|e| e.to_string())?;
            }
        }

        Ok(())
    }

    fn emit_hash_wrapper(&self, out: &mut String, ty: TypeId) -> Result<(), String> {
        let sym = callback_hash_symbol(ty);
        writeln!(out, "define weak_odr i64 @{sym}(ptr %x_bytes) {{").map_err(|e| e.to_string())?;
        writeln!(out, "entry:").map_err(|e| e.to_string())?;
        if self.is_str_handle_ty(ty) {
            writeln!(out, "  %x = load ptr, ptr %x_bytes").map_err(|e| e.to_string())?;
            writeln!(out, "  %h = call i64 @mp_std_hash_str(ptr %x)").map_err(|e| e.to_string())?;
            writeln!(out, "  ret i64 %h").map_err(|e| e.to_string())?;
        } else {
            writeln!(
                out,
                "  %h = call i64 @mp_rt_bytes_hash(ptr %x_bytes, i64 {})",
                self.size_of_ty(ty)
            )
            .map_err(|e| e.to_string())?;
            writeln!(out, "  ret i64 %h").map_err(|e| e.to_string())?;
        }
        writeln!(out, "}}").map_err(|e| e.to_string())
    }

    fn emit_eq_wrapper(&self, out: &mut String, ty: TypeId) -> Result<(), String> {
        let sym = callback_eq_symbol(ty);
        writeln!(
            out,
            "define weak_odr i32 @{sym}(ptr %a_bytes, ptr %b_bytes) {{"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "entry:").map_err(|e| e.to_string())?;
        if self.is_str_handle_ty(ty) {
            writeln!(out, "  %a = load ptr, ptr %a_bytes").map_err(|e| e.to_string())?;
            writeln!(out, "  %b = load ptr, ptr %b_bytes").map_err(|e| e.to_string())?;
            writeln!(out, "  %eq = call i32 @mp_rt_str_eq(ptr %a, ptr %b)")
                .map_err(|e| e.to_string())?;
            writeln!(out, "  ret i32 %eq").map_err(|e| e.to_string())?;
        } else {
            writeln!(
                out,
                "  %eq = call i32 @mp_rt_bytes_eq(ptr %a_bytes, ptr %b_bytes, i64 {})",
                self.size_of_ty(ty)
            )
            .map_err(|e| e.to_string())?;
            writeln!(out, "  ret i32 %eq").map_err(|e| e.to_string())?;
        }
        writeln!(out, "}}").map_err(|e| e.to_string())
    }

    fn emit_cmp_wrapper(&self, out: &mut String, ty: TypeId) -> Result<(), String> {
        let sym = callback_cmp_symbol(ty);
        writeln!(
            out,
            "define weak_odr i32 @{sym}(ptr %a_bytes, ptr %b_bytes) {{"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "entry:").map_err(|e| e.to_string())?;
        if self.is_str_handle_ty(ty) {
            writeln!(out, "  %a = load ptr, ptr %a_bytes").map_err(|e| e.to_string())?;
            writeln!(out, "  %b = load ptr, ptr %b_bytes").map_err(|e| e.to_string())?;
            writeln!(out, "  %ord = call i32 @mp_rt_str_cmp(ptr %a, ptr %b)")
                .map_err(|e| e.to_string())?;
            writeln!(out, "  ret i32 %ord").map_err(|e| e.to_string())?;
        } else {
            writeln!(
                out,
                "  %ord = call i32 @mp_rt_bytes_cmp(ptr %a_bytes, ptr %b_bytes, i64 {})",
                self.size_of_ty(ty)
            )
            .map_err(|e| e.to_string())?;
            writeln!(out, "  ret i32 %ord").map_err(|e| e.to_string())?;
        }
        writeln!(out, "}}").map_err(|e| e.to_string())
    }

    fn collect_trait_callback_usage(&self) -> BTreeMap<u32, CallbackUsage> {
        fn mark(
            usage: &mut BTreeMap<u32, CallbackUsage>,
            ty: TypeId,
            hash: bool,
            eq: bool,
            cmp: bool,
        ) {
            let entry = usage.entry(ty.0).or_default();
            entry.hash |= hash;
            entry.eq |= eq;
            entry.cmp |= cmp;
        }

        fn value_type(local_tys: &HashMap<u32, TypeId>, v: &MpirValue) -> Option<TypeId> {
            match v {
                MpirValue::Local(id) => local_tys.get(&id.0).copied(),
                MpirValue::Const(c) => Some(c.ty),
            }
        }

        let mut usage = BTreeMap::<u32, CallbackUsage>::new();

        for f in &self.mpir.functions {
            let mut local_tys = HashMap::<u32, TypeId>::new();
            for (pid, pty) in &f.params {
                local_tys.insert(pid.0, *pty);
            }
            for l in &f.locals {
                local_tys.insert(l.id.0, l.ty);
            }
            for b in &f.blocks {
                for i in &b.instrs {
                    local_tys.insert(i.dst.0, i.ty);
                }
            }

            for b in &f.blocks {
                for i in &b.instrs {
                    match &i.op {
                        MpirOp::MapNew { key_ty, .. } => {
                            mark(&mut usage, *key_ty, true, true, false);
                        }
                        MpirOp::ArrContains { val, .. } => {
                            if let Some(elem_ty) = value_type(&local_tys, val) {
                                mark(&mut usage, elem_ty, false, true, false);
                            }
                        }
                        MpirOp::ArrSort { arr } => {
                            if let Some(elem_ty) = self.array_elem_type_for_value(arr, &local_tys) {
                                mark(&mut usage, elem_ty, false, false, true);
                            }
                        }
                        _ => {}
                    }
                }
                for vop in &b.void_ops {
                    if let MpirOpVoid::ArrSort { arr } = vop {
                        if let Some(elem_ty) = self.array_elem_type_for_value(arr, &local_tys) {
                            mark(&mut usage, elem_ty, false, false, true);
                        }
                    }
                }
            }
        }

        usage
    }

    fn array_elem_type_for_value(
        &self,
        value: &MpirValue,
        local_tys: &HashMap<u32, TypeId>,
    ) -> Option<TypeId> {
        let arr_ty = match value {
            MpirValue::Local(id) => local_tys.get(&id.0).copied(),
            MpirValue::Const(c) => Some(c.ty),
        }?;
        match self.kind_of(arr_ty) {
            Some(TypeKind::HeapHandle {
                base: HeapBase::BuiltinArray { elem },
                ..
            }) => Some(*elem),
            _ => None,
        }
    }

    fn is_str_handle_ty(&self, ty: TypeId) -> bool {
        matches!(
            self.kind_of(ty),
            Some(TypeKind::HeapHandle {
                base: HeapBase::BuiltinStr,
                ..
            })
        )
    }

    fn codegen_fn(&self, f: &MpirFn) -> Result<String, String> {
        let mut fb = FnBuilder::new(self, f)?;
        fb.codegen()?;
        Ok(fb.out)
    }

    fn emit_c_main(
        &self,
        out: &mut String,
        module_init: &str,
        main_fn: Option<&MpirFn>,
    ) -> Result<(), String> {
        writeln!(out, "define i32 @main(i32 %argc, ptr %argv) {{").map_err(|e| e.to_string())?;
        writeln!(out, "entry:").map_err(|e| e.to_string())?;
        writeln!(out, "  call void @mp_rt_init()").map_err(|e| e.to_string())?;
        writeln!(out, "  call void @{}()", module_init).map_err(|e| e.to_string())?;
        writeln!(out, "  call void @mp_gpu_register_all_kernels()").map_err(|e| e.to_string())?;
        if let Some(magpie_main) = main_fn {
            let fn_name = mangle_fn(&magpie_main.sid);
            let ret_ty = self.llvm_ty(magpie_main.ret_ty);
            if ret_ty == "void" {
                writeln!(out, "  call void @{}()", fn_name).map_err(|e| e.to_string())?;
                writeln!(out, "  ret i32 0").map_err(|e| e.to_string())?;
            } else if ret_ty == "i32" {
                writeln!(out, "  %ret = call i32 @{}()", fn_name).map_err(|e| e.to_string())?;
                writeln!(out, "  ret i32 %ret").map_err(|e| e.to_string())?;
            } else if ret_ty.starts_with('i') {
                writeln!(out, "  %ret_main = call {} @{}()", ret_ty, fn_name)
                    .map_err(|e| e.to_string())?;
                writeln!(out, "  %ret_i32 = trunc {} %ret_main to i32", ret_ty)
                    .map_err(|e| e.to_string())?;
                writeln!(out, "  ret i32 %ret_i32").map_err(|e| e.to_string())?;
            } else {
                writeln!(out, "  call {} @{}()", ret_ty, fn_name).map_err(|e| e.to_string())?;
                writeln!(out, "  ret i32 0").map_err(|e| e.to_string())?;
            }
        } else {
            writeln!(out, "  ret i32 0").map_err(|e| e.to_string())?;
        }
        writeln!(out, "}}").map_err(|e| e.to_string())?;
        Ok(())
    }

    fn kind_of(&self, ty: TypeId) -> Option<&TypeKind> {
        self.type_map
            .get(&ty.0)
            .or_else(|| self.type_ctx.lookup(ty))
    }

    fn llvm_ty(&self, ty: TypeId) -> String {
        match self.kind_of(ty) {
            Some(TypeKind::Prim(prim)) => match prim {
                PrimType::I1 | PrimType::U1 | PrimType::Bool => "i1".to_string(),
                PrimType::I8 | PrimType::U8 => "i8".to_string(),
                PrimType::I16 | PrimType::U16 => "i16".to_string(),
                PrimType::I32 | PrimType::U32 => "i32".to_string(),
                PrimType::I64 | PrimType::U64 => "i64".to_string(),
                PrimType::I128 | PrimType::U128 => "i128".to_string(),
                PrimType::F16 => "half".to_string(),
                PrimType::F32 => "float".to_string(),
                PrimType::F64 => "double".to_string(),
                PrimType::Unit => "void".to_string(),
            },
            Some(TypeKind::HeapHandle { .. }) => "ptr".to_string(),
            Some(TypeKind::RawPtr { .. }) => "ptr".to_string(),
            Some(TypeKind::BuiltinOption { inner }) => {
                format!("{{ {}, i1 }}", self.llvm_storage_ty(*inner))
            }
            Some(TypeKind::BuiltinResult { ok, err }) => {
                format!(
                    "{{ i1, {}, {} }}",
                    self.llvm_storage_ty(*ok),
                    self.llvm_storage_ty(*err)
                )
            }
            Some(TypeKind::Arr { n, elem }) => format!("[{} x {}]", n, self.llvm_ty(*elem)),
            Some(TypeKind::Vec { n, elem }) => format!("<{} x {}>", n, self.llvm_ty(*elem)),
            Some(TypeKind::Tuple { elems }) => {
                if elems.is_empty() {
                    "{}".to_string()
                } else {
                    let members = elems
                        .iter()
                        .map(|e| self.llvm_ty(*e))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("{{ {} }}", members)
                }
            }
            Some(TypeKind::ValueStruct { .. }) => format!("%mp_t{}", ty.0),
            None => "ptr".to_string(),
        }
    }

    fn llvm_storage_ty(&self, ty: TypeId) -> String {
        let ty = self.llvm_ty(ty);
        if ty == "void" {
            "i8".to_string()
        } else {
            ty
        }
    }

    fn size_of_ty(&self, ty: TypeId) -> u64 {
        match self.kind_of(ty) {
            Some(TypeKind::Prim(prim)) => match prim {
                PrimType::Unit => 0,
                PrimType::I1 | PrimType::U1 | PrimType::Bool => 1,
                PrimType::I8 | PrimType::U8 => 1,
                PrimType::I16 | PrimType::U16 | PrimType::F16 => 2,
                PrimType::I32 | PrimType::U32 | PrimType::F32 => 4,
                PrimType::I64 | PrimType::U64 | PrimType::F64 => 8,
                PrimType::I128 | PrimType::U128 => 16,
            },
            Some(TypeKind::HeapHandle { .. }) | Some(TypeKind::RawPtr { .. }) => 8,
            Some(TypeKind::BuiltinOption { inner }) => self.size_of_ty(*inner).max(1) + 1,
            Some(TypeKind::BuiltinResult { ok, err }) => {
                1 + self.size_of_ty(*ok) + self.size_of_ty(*err)
            }
            Some(TypeKind::Arr { n, elem }) => (*n as u64).saturating_mul(self.size_of_ty(*elem)),
            Some(TypeKind::Vec { n, elem }) => (*n as u64).saturating_mul(self.size_of_ty(*elem)),
            Some(TypeKind::Tuple { elems }) => elems.iter().map(|e| self.size_of_ty(*e)).sum(),
            Some(TypeKind::ValueStruct { .. }) => 0,
            None => 8,
        }
    }

    fn is_gpu_buffer_param_ty(&self, ty: TypeId) -> bool {
        if ty == fixed_type_ids::GPU_BUFFER_BASE {
            return true;
        }
        match self.kind_of(ty) {
            Some(TypeKind::HeapHandle { base, .. }) => !matches!(
                base,
                HeapBase::BuiltinStr
                    | HeapBase::BuiltinArray { .. }
                    | HeapBase::BuiltinMap { .. }
                    | HeapBase::BuiltinStrBuilder
                    | HeapBase::BuiltinMutex { .. }
                    | HeapBase::BuiltinRwLock { .. }
                    | HeapBase::BuiltinCell { .. }
                    | HeapBase::BuiltinFuture { .. }
                    | HeapBase::BuiltinChannelSend { .. }
                    | HeapBase::BuiltinChannelRecv { .. }
                    | HeapBase::Callable { .. }
            ),
            _ => false,
        }
    }

    fn is_signed_int(&self, ty: TypeId) -> bool {
        matches!(
            self.kind_of(ty),
            Some(TypeKind::Prim(
                PrimType::I1
                    | PrimType::I8
                    | PrimType::I16
                    | PrimType::I32
                    | PrimType::I64
                    | PrimType::I128
            ))
        )
    }
}

#[derive(Clone)]
struct Operand {
    ty: String,
    ty_id: TypeId,
    repr: String,
}

struct FnBuilder<'a> {
    cg: &'a LlvmTextCodegen<'a>,
    f: &'a MpirFn,
    out: String,
    tmp_idx: u32,
    locals: HashMap<u32, Operand>,
    local_tys: HashMap<u32, TypeId>,
}

impl<'a> FnBuilder<'a> {
    fn new(cg: &'a LlvmTextCodegen<'a>, f: &'a MpirFn) -> Result<Self, String> {
        let mut local_tys = HashMap::new();
        for (pid, pty) in &f.params {
            local_tys.insert(pid.0, *pty);
        }
        for l in &f.locals {
            local_tys.insert(l.id.0, l.ty);
        }
        for b in &f.blocks {
            for i in &b.instrs {
                local_tys.insert(i.dst.0, i.ty);
            }
        }

        let ret_ty = cg.llvm_ty(f.ret_ty);
        let params = f
            .params
            .iter()
            .map(|(id, ty)| format!("{} %arg{}", cg.llvm_ty(*ty), id.0))
            .collect::<Vec<_>>()
            .join(", ");
        let mut out = String::new();
        writeln!(
            out,
            "define {} @{}({}) {{",
            ret_ty,
            mangle_fn(&f.sid),
            params
        )
        .map_err(|e| e.to_string())?;

        let mut locals = HashMap::new();
        for (id, ty) in &f.params {
            locals.insert(
                id.0,
                Operand {
                    ty: cg.llvm_ty(*ty),
                    ty_id: *ty,
                    repr: format!("%arg{}", id.0),
                },
            );
        }

        Ok(Self {
            cg,
            f,
            out,
            tmp_idx: 0,
            locals,
            local_tys,
        })
    }

    fn codegen(&mut self) -> Result<(), String> {
        if self.f.blocks.is_empty() {
            writeln!(self.out, "entry:").map_err(|e| e.to_string())?;
            let ret_ty = self.cg.llvm_ty(self.f.ret_ty);
            if ret_ty == "void" {
                writeln!(self.out, "  ret void").map_err(|e| e.to_string())?;
            } else {
                writeln!(self.out, "  ret {} {}", ret_ty, self.zero_lit(&ret_ty))
                    .map_err(|e| e.to_string())?;
            }
            writeln!(self.out, "}}").map_err(|e| e.to_string())?;
            return Ok(());
        }

        for b in &self.f.blocks {
            self.codegen_block(b)?;
        }
        writeln!(self.out, "}}").map_err(|e| e.to_string())?;
        Ok(())
    }

    fn codegen_block(&mut self, b: &MpirBlock) -> Result<(), String> {
        writeln!(self.out, "bb{}:", b.id.0).map_err(|e| e.to_string())?;
        for i in &b.instrs {
            self.codegen_instr(i)?;
        }
        for op in &b.void_ops {
            self.codegen_void_op(op)?;
        }
        self.codegen_term(&b.terminator)
    }

    fn codegen_instr(&mut self, i: &MpirInstr) -> Result<(), String> {
        let dst_ty = self.cg.llvm_ty(i.ty);
        let dst = format!("%l{}", i.dst.0);
        match &i.op {
            MpirOp::Const(c) => {
                let lit = match &c.lit {
                    HirConstLit::StringLit(s) => self.emit_string_literal_runtime(s)?,
                    _ => self.const_lit(c)?,
                };
                self.locals.insert(
                    i.dst.0,
                    Operand {
                        ty: dst_ty,
                        ty_id: i.ty,
                        repr: lit,
                    },
                );
            }
            MpirOp::Move { v }
            | MpirOp::BorrowShared { v }
            | MpirOp::BorrowMut { v }
            | MpirOp::Share { v }
            | MpirOp::CloneShared { v }
            | MpirOp::CloneWeak { v }
            | MpirOp::WeakDowngrade { v }
            | MpirOp::WeakUpgrade { v } => {
                let op = self.value(v)?;
                self.assign_or_copy(i.dst, i.ty, op)?;
            }
            MpirOp::New { ty, fields } => {
                self.emit_new_struct(i.dst, i.ty, *ty, fields)?;
            }
            MpirOp::GetField { obj, field } => {
                self.emit_get_field(i.dst, i.ty, obj, field)?;
            }
            MpirOp::IAdd { lhs, rhs } | MpirOp::IAddWrap { lhs, rhs } => {
                self.emit_bin(i.dst, i.ty, "add", lhs, rhs)?
            }
            MpirOp::ISub { lhs, rhs } | MpirOp::ISubWrap { lhs, rhs } => {
                self.emit_bin(i.dst, i.ty, "sub", lhs, rhs)?
            }
            MpirOp::IMul { lhs, rhs } | MpirOp::IMulWrap { lhs, rhs } => {
                self.emit_bin(i.dst, i.ty, "mul", lhs, rhs)?
            }
            MpirOp::ISDiv { lhs, rhs } => self.emit_bin(i.dst, i.ty, "sdiv", lhs, rhs)?,
            MpirOp::IUDiv { lhs, rhs } => self.emit_bin(i.dst, i.ty, "udiv", lhs, rhs)?,
            MpirOp::ISRem { lhs, rhs } => self.emit_bin(i.dst, i.ty, "srem", lhs, rhs)?,
            MpirOp::IURem { lhs, rhs } => self.emit_bin(i.dst, i.ty, "urem", lhs, rhs)?,
            MpirOp::IAnd { lhs, rhs } => self.emit_bin(i.dst, i.ty, "and", lhs, rhs)?,
            MpirOp::IOr { lhs, rhs } => self.emit_bin(i.dst, i.ty, "or", lhs, rhs)?,
            MpirOp::IXor { lhs, rhs } => self.emit_bin(i.dst, i.ty, "xor", lhs, rhs)?,
            MpirOp::IShl { lhs, rhs } => self.emit_bin(i.dst, i.ty, "shl", lhs, rhs)?,
            MpirOp::ILshr { lhs, rhs } => self.emit_bin(i.dst, i.ty, "lshr", lhs, rhs)?,
            MpirOp::IAshr { lhs, rhs } => self.emit_bin(i.dst, i.ty, "ashr", lhs, rhs)?,
            MpirOp::FAdd { lhs, rhs } | MpirOp::FAddFast { lhs, rhs } => {
                self.emit_bin(i.dst, i.ty, "fadd", lhs, rhs)?
            }
            MpirOp::FSub { lhs, rhs } | MpirOp::FSubFast { lhs, rhs } => {
                self.emit_bin(i.dst, i.ty, "fsub", lhs, rhs)?
            }
            MpirOp::FMul { lhs, rhs } | MpirOp::FMulFast { lhs, rhs } => {
                self.emit_bin(i.dst, i.ty, "fmul", lhs, rhs)?
            }
            MpirOp::FDiv { lhs, rhs } | MpirOp::FDivFast { lhs, rhs } => {
                self.emit_bin(i.dst, i.ty, "fdiv", lhs, rhs)?
            }
            MpirOp::FRem { lhs, rhs } => self.emit_bin(i.dst, i.ty, "frem", lhs, rhs)?,
            MpirOp::IAddChecked { lhs, rhs } => self.emit_checked(i.dst, i.ty, lhs, rhs, "sadd")?,
            MpirOp::ISubChecked { lhs, rhs } => self.emit_checked(i.dst, i.ty, lhs, rhs, "ssub")?,
            MpirOp::IMulChecked { lhs, rhs } => self.emit_checked(i.dst, i.ty, lhs, rhs, "smul")?,
            MpirOp::ICmp { pred, lhs, rhs } => {
                let l = self.value(lhs)?;
                let r = self.value(rhs)?;
                let cmp_ty = l.ty.clone();
                let icmp = normalize_icmp_pred(pred);
                writeln!(
                    self.out,
                    "  {dst} = icmp {icmp} {cmp_ty} {}, {}",
                    l.repr, r.repr
                )
                .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::FCmp { pred, lhs, rhs } => {
                let l = self.value(lhs)?;
                let r = self.value(rhs)?;
                let cmp_ty = l.ty.clone();
                let fcmp = normalize_fcmp_pred(pred);
                writeln!(
                    self.out,
                    "  {dst} = fcmp {fcmp} {cmp_ty} {}, {}",
                    l.repr, r.repr
                )
                .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::Cast { to, v } => {
                self.emit_cast(i.dst, i.ty, *to, v)?;
            }
            MpirOp::PtrNull { .. } => {
                self.locals.insert(
                    i.dst.0,
                    Operand {
                        ty: dst_ty,
                        ty_id: i.ty,
                        repr: "null".to_string(),
                    },
                );
            }
            MpirOp::PtrAddr { p } => {
                let p = self.value(p)?;
                writeln!(
                    self.out,
                    "  {dst} = ptrtoint {} {} to {}",
                    p.ty, p.repr, dst_ty
                )
                .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::PtrFromAddr { .. } => {
                let addr = match &i.op {
                    MpirOp::PtrFromAddr { addr, .. } => self.value(addr)?,
                    _ => unreachable!(),
                };
                writeln!(
                    self.out,
                    "  {dst} = inttoptr {} {} to {}",
                    addr.ty, addr.repr, dst_ty
                )
                .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::PtrAdd { p, count } => {
                let p = self.ensure_ptr_value(p)?;
                let count = self.cast_i64_value(count)?;
                writeln!(self.out, "  {dst} = getelementptr i8, ptr {p}, i64 {count}")
                    .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::PtrLoad { to, p } => {
                let p = self.ensure_ptr_value(p)?;
                let load_ty = self.cg.llvm_ty(*to);
                writeln!(self.out, "  {dst} = load {load_ty}, ptr {p}")
                    .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::PtrStore { to, p, v } => {
                let p = self.ensure_ptr_value(p)?;
                let v = self.value(v)?;
                writeln!(
                    self.out,
                    "  store {} {}, ptr {}",
                    self.cg.llvm_ty(*to),
                    v.repr,
                    p
                )
                .map_err(|e| e.to_string())?;
                self.set_default(i.dst, i.ty)?;
            }
            MpirOp::Call {
                callee_sid, args, ..
            } => {
                let args = self.call_args(args)?;
                let name = mangle_fn(callee_sid);
                if dst_ty == "void" {
                    writeln!(self.out, "  call void @{}({})", name, args)
                        .map_err(|e| e.to_string())?;
                    self.set_default(i.dst, i.ty)?;
                } else {
                    writeln!(self.out, "  {dst} = call {dst_ty} @{}({})", name, args)
                        .map_err(|e| e.to_string())?;
                    self.set_local(i.dst, i.ty, dst_ty, dst);
                }
            }
            MpirOp::CallIndirect { callee, args } => {
                self.emit_call_indirect(i.dst, i.ty, callee, args, false)?;
            }
            MpirOp::CallVoidIndirect { callee, args } => {
                self.emit_call_indirect(i.dst, i.ty, callee, args, true)?;
            }
            MpirOp::SuspendCall {
                callee_sid, args, ..
            } => {
                // v0.1: async lowering should run before codegen; emit as a regular call.
                let args = self.call_args(args)?;
                let name = mangle_fn(callee_sid);
                if dst_ty == "void" {
                    writeln!(self.out, "  call void @{}({})", name, args)
                        .map_err(|e| e.to_string())?;
                    self.set_default(i.dst, i.ty)?;
                } else {
                    writeln!(self.out, "  {dst} = call {dst_ty} @{}({})", name, args)
                        .map_err(|e| e.to_string())?;
                    self.set_local(i.dst, i.ty, dst_ty, dst);
                }
            }
            MpirOp::SuspendAwait { fut } => {
                // v0.1: poll in a busy loop until Ready, then take the value.
                let fut = self.ensure_ptr_value(fut)?;
                let poll_label = self.label("await_poll");
                let ready_label = self.label("await_ready");
                let poll_state = self.tmp();
                let is_ready = self.tmp();
                writeln!(self.out, "  br label %{poll_label}").map_err(|e| e.to_string())?;
                writeln!(self.out, "{poll_label}:").map_err(|e| e.to_string())?;
                writeln!(
                    self.out,
                    "  {poll_state} = call i32 @mp_rt_future_poll(ptr {fut})"
                )
                .map_err(|e| e.to_string())?;
                writeln!(self.out, "  {is_ready} = icmp ne i32 {poll_state}, 0")
                    .map_err(|e| e.to_string())?;
                writeln!(
                    self.out,
                    "  br i1 {is_ready}, label %{ready_label}, label %{poll_label}"
                )
                .map_err(|e| e.to_string())?;
                writeln!(self.out, "{ready_label}:").map_err(|e| e.to_string())?;
                if dst_ty == "void" {
                    writeln!(
                        self.out,
                        "  call void @mp_rt_future_take(ptr {fut}, ptr null)"
                    )
                    .map_err(|e| e.to_string())?;
                    self.set_default(i.dst, i.ty)?;
                } else {
                    let slot = self.tmp();
                    writeln!(self.out, "  {slot} = alloca {dst_ty}").map_err(|e| e.to_string())?;
                    writeln!(
                        self.out,
                        "  call void @mp_rt_future_take(ptr {fut}, ptr {slot})"
                    )
                    .map_err(|e| e.to_string())?;
                    writeln!(self.out, "  {dst} = load {dst_ty}, ptr {slot}")
                        .map_err(|e| e.to_string())?;
                    self.set_local(i.dst, i.ty, dst_ty, dst);
                }
            }
            MpirOp::Phi { incomings, .. } => {
                let mut parts = Vec::with_capacity(incomings.len());
                for (bb, v) in incomings {
                    let op = self.value(v)?;
                    parts.push(format!("[ {}, %bb{} ]", op.repr, bb.0));
                }
                writeln!(self.out, "  {dst} = phi {dst_ty} {}", parts.join(", "))
                    .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::EnumNew { variant, args } => {
                self.emit_enum_new(i.dst, i.ty, variant, args)?;
            }
            MpirOp::EnumTag { v } => {
                self.emit_enum_tag(i.dst, i.ty, v)?;
            }
            MpirOp::EnumPayload { variant, v } => {
                self.emit_enum_payload(i.dst, i.ty, variant, v)?;
            }
            MpirOp::EnumIs { variant, v } => {
                self.emit_enum_is(i.dst, i.ty, variant, v)?;
            }
            MpirOp::ArcRetain { v } => {
                let op = self.value(v)?;
                if op.ty.starts_with('{') {
                    self.emit_arc_retain_composite(&op)?;
                } else {
                    let p = self.ensure_ptr(op)?;
                    writeln!(self.out, "  call void @mp_rt_retain_strong(ptr {p})")
                        .map_err(|e| e.to_string())?;
                }
                self.assign_or_copy_value(i.dst, i.ty, v)?;
            }
            MpirOp::ArcRelease { v } => {
                let op = self.value(v)?;
                if op.ty.starts_with('{') {
                    self.emit_arc_release_composite(&op)?;
                } else {
                    let p = self.ensure_ptr(op)?;
                    writeln!(self.out, "  call void @mp_rt_release_strong(ptr {p})")
                        .map_err(|e| e.to_string())?;
                }
                self.set_default(i.dst, i.ty)?;
            }
            MpirOp::ArcRetainWeak { v } => {
                let op = self.value(v)?;
                if op.ty.starts_with('{') {
                    self.emit_arc_retain_composite(&op)?;
                } else {
                    let p = self.ensure_ptr(op)?;
                    writeln!(self.out, "  call void @mp_rt_retain_weak(ptr {p})")
                        .map_err(|e| e.to_string())?;
                }
                self.assign_or_copy_value(i.dst, i.ty, v)?;
            }
            MpirOp::ArcReleaseWeak { v } => {
                let op = self.value(v)?;
                if op.ty.starts_with('{') {
                    self.emit_arc_release_composite(&op)?;
                } else {
                    let p = self.ensure_ptr(op)?;
                    writeln!(self.out, "  call void @mp_rt_release_weak(ptr {p})")
                        .map_err(|e| e.to_string())?;
                }
                self.set_default(i.dst, i.ty)?;
            }
            MpirOp::ArrNew { elem_ty, cap } => {
                let cap = self.cast_i64_value(cap)?;
                let elem_size = self.cg.size_of_ty(*elem_ty);
                writeln!(
                    self.out,
                    "  {dst} = call ptr @mp_rt_arr_new(i32 {}, i64 {}, i64 {})",
                    elem_ty.0, elem_size, cap
                )
                .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::ArrLen { arr } => {
                let arr = self.ensure_ptr_value(arr)?;
                let len = self.tmp();
                writeln!(self.out, "  {len} = call i64 @mp_rt_arr_len(ptr {arr})")
                    .map_err(|e| e.to_string())?;
                self.assign_cast_int(i.dst, i.ty, len, "i64")?;
            }
            MpirOp::ArrGet { arr, idx } => {
                let arr = self.ensure_ptr_value(arr)?;
                let idx = self.cast_i64_value(idx)?;
                let p = self.tmp();
                writeln!(
                    self.out,
                    "  {p} = call ptr @mp_rt_arr_get(ptr {arr}, i64 {idx})"
                )
                .map_err(|e| e.to_string())?;
                writeln!(self.out, "  {dst} = load {dst_ty}, ptr {p}")
                    .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::ArrSet { arr, idx, val } => {
                self.emit_arr_set(arr, idx, val)?;
                self.set_default(i.dst, i.ty)?;
            }
            MpirOp::ArrPush { arr, val } => {
                self.emit_arr_push(arr, val)?;
                self.set_default(i.dst, i.ty)?;
            }
            MpirOp::ArrPop { arr } => {
                self.emit_arr_pop(i.dst, i.ty, &dst_ty, arr)?;
            }
            MpirOp::ArrSlice { arr, start, end } => {
                let arr = self.ensure_ptr_value(arr)?;
                let start = self.cast_i64_value(start)?;
                let end = self.cast_i64_value(end)?;
                writeln!(
                    self.out,
                    "  {dst} = call ptr @mp_rt_arr_slice(ptr {arr}, i64 {start}, i64 {end})"
                )
                .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::ArrContains { arr, val } => {
                let arr = self.ensure_ptr_value(arr)?;
                let val = self.value(val)?;
                let slot = self.stack_slot(&val)?;
                let stat = self.tmp();
                let eq_cb = callback_eq_symbol(val.ty_id);
                writeln!(
                    self.out,
                    "  {stat} = call i32 @mp_rt_arr_contains(ptr {arr}, ptr {slot}, i64 {}, ptr @{eq_cb})",
                    self.cg.size_of_ty(val.ty_id),
                )
                .map_err(|e| e.to_string())?;
                self.assign_cast_int(i.dst, i.ty, stat, "i32")?;
            }
            MpirOp::ArrSort { arr } => {
                let arr = self.ensure_ptr_value(arr)?;
                let cmp_cb = self
                    .array_elem_ty_for_value(&arr)
                    .map(callback_cmp_symbol)
                    .unwrap_or_else(|| "0".to_string());
                let cmp_ref = if cmp_cb == "0" {
                    "null".to_string()
                } else {
                    format!("@{cmp_cb}")
                };
                writeln!(
                    self.out,
                    "  call void @mp_rt_arr_sort(ptr {arr}, ptr {cmp_ref})"
                )
                .map_err(|e| e.to_string())?;
                self.set_default(i.dst, i.ty)?;
            }
            MpirOp::ArrMap { arr, func } => {
                let arr = self.ensure_ptr_value(arr)?;
                let func = self.ensure_ptr_value(func)?;
                let (elem_tid, elem_size) = self.array_result_elem(i.ty);
                writeln!(
                    self.out,
                    "  {dst} = call ptr @mp_rt_arr_map(ptr {arr}, ptr {func}, i32 {}, i64 {})",
                    elem_tid.0, elem_size
                )
                .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::ArrFilter { arr, func } => {
                let arr = self.ensure_ptr_value(arr)?;
                let func = self.ensure_ptr_value(func)?;
                writeln!(
                    self.out,
                    "  {dst} = call ptr @mp_rt_arr_filter(ptr {arr}, ptr {func})"
                )
                .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::ArrReduce { arr, init, func } => {
                let arr = self.ensure_ptr_value(arr)?;
                let init = self.value(init)?;
                let slot = self.stack_slot(&init)?;
                let func = self.ensure_ptr_value(func)?;
                writeln!(
                    self.out,
                    "  call void @mp_rt_arr_reduce(ptr {arr}, ptr {slot}, i64 {}, ptr {func})",
                    self.cg.size_of_ty(init.ty_id)
                )
                .map_err(|e| e.to_string())?;
                writeln!(self.out, "  {dst} = load {dst_ty}, ptr {slot}")
                    .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::ArrForeach { arr, func } => {
                let arr = self.ensure_ptr_value(arr)?;
                let func = self.ensure_ptr_value(func)?;
                writeln!(
                    self.out,
                    "  call void @mp_rt_arr_foreach(ptr {arr}, ptr {func})"
                )
                .map_err(|e| e.to_string())?;
                self.set_default(i.dst, i.ty)?;
            }
            MpirOp::CallableCapture { fn_ref, captures } => {
                let fn_name = mangle_fn(fn_ref);
                let captures_ptr = if captures.is_empty() {
                    "null".to_string()
                } else {
                    let captured = captures
                        .iter()
                        .map(|(_, v)| self.value(v))
                        .collect::<Result<Vec<_>, _>>()?;
                    let env_size = captured
                        .iter()
                        .map(|op| self.cg.size_of_ty(op.ty_id))
                        .sum::<u64>();
                    let packed_size = env_size + 8;
                    let packed_slot = self.tmp();
                    writeln!(self.out, "  {packed_slot} = alloca [{packed_size} x i8]")
                        .map_err(|e| e.to_string())?;

                    let size_ptr = self.tmp();
                    writeln!(
                        self.out,
                        "  {size_ptr} = getelementptr i8, ptr {packed_slot}, i64 0"
                    )
                    .map_err(|e| e.to_string())?;
                    writeln!(self.out, "  store i64 {env_size}, ptr {size_ptr}")
                        .map_err(|e| e.to_string())?;

                    let mut offset = 0u64;
                    for op in &captured {
                        let field_ptr = self.tmp();
                        writeln!(
                            self.out,
                            "  {field_ptr} = getelementptr i8, ptr {packed_slot}, i64 {}",
                            offset + 8
                        )
                        .map_err(|e| e.to_string())?;
                        writeln!(self.out, "  store {} {}, ptr {field_ptr}", op.ty, op.repr)
                            .map_err(|e| e.to_string())?;
                        offset += self.cg.size_of_ty(op.ty_id);
                    }
                    packed_slot
                };
                writeln!(
                    self.out,
                    "  {dst} = call ptr @mp_rt_callable_new(ptr @{}, ptr {captures_ptr})",
                    fn_name
                )
                .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::MapNew { key_ty, val_ty } => {
                let key_size = self.cg.size_of_ty(*key_ty);
                let val_size = self.cg.size_of_ty(*val_ty);
                let hash_cb = callback_hash_symbol(*key_ty);
                let eq_cb = callback_eq_symbol(*key_ty);
                writeln!(
                    self.out,
                    "  {dst} = call ptr @mp_rt_map_new(i32 {}, i32 {}, i64 {}, i64 {}, i64 0, ptr @{hash_cb}, ptr @{eq_cb})",
                    key_ty.0, val_ty.0, key_size, val_size,
                )
                .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::MapLen { map } => {
                let map = self.ensure_ptr_value(map)?;
                let len = self.tmp();
                writeln!(self.out, "  {len} = call i64 @mp_rt_map_len(ptr {map})")
                    .map_err(|e| e.to_string())?;
                self.assign_cast_int(i.dst, i.ty, len, "i64")?;
            }
            MpirOp::MapGet { map, key } => {
                let map = self.ensure_ptr_value(map)?;
                let key = self.value(key)?;
                let keyp = self.stack_slot(&key)?;
                let p = self.tmp();
                writeln!(
                    self.out,
                    "  {p} = call ptr @mp_rt_map_get(ptr {map}, ptr {keyp}, i64 {})",
                    self.cg.size_of_ty(key.ty_id)
                )
                .map_err(|e| e.to_string())?;
                writeln!(self.out, "  {dst} = load {dst_ty}, ptr {p}")
                    .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::MapGetRef { map, key } => {
                let map = self.ensure_ptr_value(map)?;
                let key = self.value(key)?;
                let keyp = self.stack_slot(&key)?;
                writeln!(
                    self.out,
                    "  {dst} = call ptr @mp_rt_map_get(ptr {map}, ptr {keyp}, i64 {})",
                    self.cg.size_of_ty(key.ty_id)
                )
                .map_err(|e| e.to_string())?;
                let is_missing = self.tmp();
                let panic_label = self.label("map_get_ref_panic");
                let ok_label = self.label("map_get_ref_ok");
                writeln!(self.out, "  {is_missing} = icmp eq ptr {dst}, null")
                    .map_err(|e| e.to_string())?;
                writeln!(
                    self.out,
                    "  br i1 {is_missing}, label %{panic_label}, label %{ok_label}"
                )
                .map_err(|e| e.to_string())?;
                writeln!(self.out, "{panic_label}:").map_err(|e| e.to_string())?;
                writeln!(self.out, "  call void @mp_rt_panic(ptr null)")
                    .map_err(|e| e.to_string())?;
                writeln!(self.out, "  unreachable").map_err(|e| e.to_string())?;
                writeln!(self.out, "{ok_label}:").map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::MapSet { map, key, val } => {
                self.emit_map_set(map, key, val)?;
                self.set_default(i.dst, i.ty)?;
            }
            MpirOp::MapDelete { map, key } | MpirOp::MapDeleteVoid { map, key } => {
                let map = self.ensure_ptr_value(map)?;
                let key = self.value(key)?;
                let keyp = self.stack_slot(&key)?;
                let stat = self.tmp();
                writeln!(
                    self.out,
                    "  {stat} = call i32 @mp_rt_map_delete(ptr {map}, ptr {keyp}, i64 {})",
                    self.cg.size_of_ty(key.ty_id)
                )
                .map_err(|e| e.to_string())?;
                self.assign_cast_int(i.dst, i.ty, stat, "i32")?;
            }
            MpirOp::MapContainsKey { map, key } => {
                let map = self.ensure_ptr_value(map)?;
                let key = self.value(key)?;
                let keyp = self.stack_slot(&key)?;
                let stat = self.tmp();
                writeln!(
                    self.out,
                    "  {stat} = call i32 @mp_rt_map_contains_key(ptr {map}, ptr {keyp}, i64 {})",
                    self.cg.size_of_ty(key.ty_id)
                )
                .map_err(|e| e.to_string())?;
                self.assign_cast_int(i.dst, i.ty, stat, "i32")?;
            }
            MpirOp::MapKeys { map } => {
                let map = self.ensure_ptr_value(map)?;
                writeln!(self.out, "  {dst} = call ptr @mp_rt_map_keys(ptr {map})")
                    .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::MapValues { map } => {
                let map = self.ensure_ptr_value(map)?;
                writeln!(self.out, "  {dst} = call ptr @mp_rt_map_values(ptr {map})")
                    .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::StrConcat { a, b } => {
                let a = self.ensure_ptr_value(a)?;
                let b = self.ensure_ptr_value(b)?;
                writeln!(
                    self.out,
                    "  {dst} = call ptr @mp_rt_str_concat(ptr {a}, ptr {b})"
                )
                .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::StrLen { s } => {
                let s = self.ensure_ptr_value(s)?;
                let len = self.tmp();
                writeln!(self.out, "  {len} = call i64 @mp_rt_str_len(ptr {s})")
                    .map_err(|e| e.to_string())?;
                self.assign_cast_int(i.dst, i.ty, len, "i64")?;
            }
            MpirOp::StrEq { a, b } => {
                let a = self.ensure_ptr_value(a)?;
                let b = self.ensure_ptr_value(b)?;
                let eq = self.tmp();
                writeln!(
                    self.out,
                    "  {eq} = call i32 @mp_rt_str_eq(ptr {a}, ptr {b})"
                )
                .map_err(|e| e.to_string())?;
                self.assign_cast_int(i.dst, i.ty, eq, "i32")?;
            }
            MpirOp::StrSlice { s, start, end } => {
                let s = self.ensure_ptr_value(s)?;
                let start = self.cast_i64_value(start)?;
                let end = self.cast_i64_value(end)?;
                writeln!(
                    self.out,
                    "  {dst} = call ptr @mp_rt_str_slice(ptr {s}, i64 {start}, i64 {end})"
                )
                .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::StrBytes { s } => {
                let s = self.ensure_ptr_value(s)?;
                let len_slot = self.tmp();
                writeln!(self.out, "  {len_slot} = alloca i64").map_err(|e| e.to_string())?;
                writeln!(
                    self.out,
                    "  {dst} = call ptr @mp_rt_str_bytes(ptr {s}, ptr {len_slot})"
                )
                .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::StrParseI64 { s } => {
                let s = self.ensure_ptr_value(s)?;
                let parsed = self.tmp();
                writeln!(
                    self.out,
                    "  {parsed} = call i64 @mp_rt_str_parse_i64(ptr {s})"
                )
                .map_err(|e| e.to_string())?;
                self.assign_cast_int(i.dst, i.ty, parsed, "i64")?;
            }
            MpirOp::StrParseU64 { s } => {
                let s = self.ensure_ptr_value(s)?;
                let parsed = self.tmp();
                writeln!(
                    self.out,
                    "  {parsed} = call i64 @mp_rt_str_parse_u64(ptr {s})"
                )
                .map_err(|e| e.to_string())?;
                self.assign_cast_int(i.dst, i.ty, parsed, "i64")?;
            }
            MpirOp::StrParseF64 { s } => {
                let s = self.ensure_ptr_value(s)?;
                writeln!(
                    self.out,
                    "  {dst} = call double @mp_rt_str_parse_f64(ptr {s})"
                )
                .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::StrParseBool { s } => {
                let s = self.ensure_ptr_value(s)?;
                let parsed = self.tmp();
                writeln!(
                    self.out,
                    "  {parsed} = call i32 @mp_rt_str_parse_bool(ptr {s})"
                )
                .map_err(|e| e.to_string())?;
                self.assign_cast_int(i.dst, i.ty, parsed, "i32")?;
            }
            MpirOp::JsonEncode { ty, v } => {
                let v = self.ensure_ptr_value(v)?;
                writeln!(
                    self.out,
                    "  {dst} = call ptr @mp_rt_json_encode(ptr {v}, i32 {})",
                    ty.0
                )
                .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::JsonDecode { ty, s } => {
                let s = self.ensure_ptr_value(s)?;
                writeln!(
                    self.out,
                    "  {dst} = call ptr @mp_rt_json_decode(ptr {s}, i32 {})",
                    ty.0
                )
                .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::GpuThreadId
            | MpirOp::GpuWorkgroupId
            | MpirOp::GpuWorkgroupSize
            | MpirOp::GpuGlobalId => {
                self.assign_cast_int(i.dst, i.ty, "0".to_string(), "i32")?;
            }
            MpirOp::GpuBufferLen { buf } => {
                let buf = self.ensure_ptr_value(buf)?;
                let len = self.tmp();
                writeln!(
                    self.out,
                    "  {len} = call i64 @mp_rt_gpu_buffer_len(ptr {buf})"
                )
                .map_err(|e| e.to_string())?;
                self.assign_cast_int(i.dst, i.ty, len, "i64")?;
            }
            MpirOp::GpuBufferLoad { buf, idx } => {
                self.emit_gpu_buffer_load(i.dst, i.ty, buf, idx)?;
            }
            MpirOp::GpuShared { ty, size } => {
                self.emit_gpu_shared(i.dst, i.ty, *ty, size)?;
            }
            MpirOp::GpuLaunch {
                device,
                kernel,
                groups,
                threads,
                args,
            } => {
                self.emit_gpu_launch(i.dst, i.ty, device, kernel, groups, threads, args, false)?;
            }
            MpirOp::GpuLaunchAsync {
                device,
                kernel,
                groups,
                threads,
                args,
            } => {
                self.emit_gpu_launch(i.dst, i.ty, device, kernel, groups, threads, args, true)?;
            }
            MpirOp::StrBuilderNew => {
                writeln!(self.out, "  {dst} = call ptr @mp_rt_strbuilder_new()")
                    .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::StrBuilderAppendStr { b, s } => {
                let b = self.ensure_ptr_value(b)?;
                let s = self.ensure_ptr_value(s)?;
                writeln!(
                    self.out,
                    "  call void @mp_rt_strbuilder_append_str(ptr {b}, ptr {s})"
                )
                .map_err(|e| e.to_string())?;
                self.set_default(i.dst, i.ty)?;
            }
            MpirOp::StrBuilderAppendI64 { b, v } => {
                let b = self.ensure_ptr_value(b)?;
                let v = self.cast_i64_value(v)?;
                writeln!(
                    self.out,
                    "  call void @mp_rt_strbuilder_append_i64(ptr {b}, i64 {v})"
                )
                .map_err(|e| e.to_string())?;
                self.set_default(i.dst, i.ty)?;
            }
            MpirOp::StrBuilderAppendI32 { b, v } => {
                let b = self.ensure_ptr_value(b)?;
                let v = self.cast_i32_value(v)?;
                writeln!(
                    self.out,
                    "  call void @mp_rt_strbuilder_append_i32(ptr {b}, i32 {v})"
                )
                .map_err(|e| e.to_string())?;
                self.set_default(i.dst, i.ty)?;
            }
            MpirOp::StrBuilderAppendF64 { b, v } => {
                let b = self.ensure_ptr_value(b)?;
                let v = self.cast_f64_value(v)?;
                writeln!(
                    self.out,
                    "  call void @mp_rt_strbuilder_append_f64(ptr {b}, double {v})"
                )
                .map_err(|e| e.to_string())?;
                self.set_default(i.dst, i.ty)?;
            }
            MpirOp::StrBuilderAppendBool { b, v } => {
                let b = self.ensure_ptr_value(b)?;
                let v = self.cast_i32_value(v)?;
                writeln!(
                    self.out,
                    "  call void @mp_rt_strbuilder_append_bool(ptr {b}, i32 {v})"
                )
                .map_err(|e| e.to_string())?;
                self.set_default(i.dst, i.ty)?;
            }
            MpirOp::StrBuilderBuild { b } => {
                let b = self.ensure_ptr_value(b)?;
                writeln!(
                    self.out,
                    "  {dst} = call ptr @mp_rt_strbuilder_build(ptr {b})"
                )
                .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::Panic { msg } => {
                let msg = self.ensure_ptr_value(msg)?;
                writeln!(self.out, "  call void @mp_rt_panic(ptr {msg})")
                    .map_err(|e| e.to_string())?;
                self.set_default(i.dst, i.ty)?;
            }
        }
        Ok(())
    }

    fn codegen_void_op(&mut self, op: &MpirOpVoid) -> Result<(), String> {
        match op {
            MpirOpVoid::CallVoid {
                callee_sid, args, ..
            } => {
                let args = self.call_args(args)?;
                writeln!(self.out, "  call void @{}({})", mangle_fn(callee_sid), args)
                    .map_err(|e| e.to_string())?;
            }
            MpirOpVoid::CallVoidIndirect { callee, args } => {
                self.emit_call_indirect_void(callee, args)?;
            }
            MpirOpVoid::SetField { obj, field, value } => {
                self.emit_set_field(obj, field, value)?;
            }
            MpirOpVoid::ArrSet { arr, idx, val } => self.emit_arr_set(arr, idx, val)?,
            MpirOpVoid::ArrPush { arr, val } => self.emit_arr_push(arr, val)?,
            MpirOpVoid::ArrSort { arr } => {
                let arr = self.ensure_ptr_value(arr)?;
                let cmp_cb = self
                    .array_elem_ty_for_value(&arr)
                    .map(callback_cmp_symbol)
                    .unwrap_or_else(|| "0".to_string());
                let cmp_ref = if cmp_cb == "0" {
                    "null".to_string()
                } else {
                    format!("@{cmp_cb}")
                };
                writeln!(
                    self.out,
                    "  call void @mp_rt_arr_sort(ptr {arr}, ptr {cmp_ref})"
                )
                .map_err(|e| e.to_string())?;
            }
            MpirOpVoid::ArrForeach { arr, func } => {
                let arr = self.ensure_ptr_value(arr)?;
                let func = self.ensure_ptr_value(func)?;
                writeln!(
                    self.out,
                    "  call void @mp_rt_arr_foreach(ptr {arr}, ptr {func})"
                )
                .map_err(|e| e.to_string())?;
            }
            MpirOpVoid::MapSet { map, key, val } => self.emit_map_set(map, key, val)?,
            MpirOpVoid::MapDeleteVoid { map, key } => {
                let map = self.ensure_ptr_value(map)?;
                let key = self.value(key)?;
                let keyp = self.stack_slot(&key)?;
                writeln!(
                    self.out,
                    "  call i32 @mp_rt_map_delete(ptr {map}, ptr {keyp}, i64 {})",
                    self.cg.size_of_ty(key.ty_id)
                )
                .map_err(|e| e.to_string())?;
            }
            MpirOpVoid::StrBuilderAppendStr { b, s } => {
                let b = self.ensure_ptr_value(b)?;
                let s = self.ensure_ptr_value(s)?;
                writeln!(
                    self.out,
                    "  call void @mp_rt_strbuilder_append_str(ptr {b}, ptr {s})"
                )
                .map_err(|e| e.to_string())?;
            }
            MpirOpVoid::StrBuilderAppendI64 { b, v } => {
                let b = self.ensure_ptr_value(b)?;
                let v = self.cast_i64_value(v)?;
                writeln!(
                    self.out,
                    "  call void @mp_rt_strbuilder_append_i64(ptr {b}, i64 {v})"
                )
                .map_err(|e| e.to_string())?;
            }
            MpirOpVoid::StrBuilderAppendI32 { b, v } => {
                let b = self.ensure_ptr_value(b)?;
                let v = self.cast_i32_value(v)?;
                writeln!(
                    self.out,
                    "  call void @mp_rt_strbuilder_append_i32(ptr {b}, i32 {v})"
                )
                .map_err(|e| e.to_string())?;
            }
            MpirOpVoid::StrBuilderAppendF64 { b, v } => {
                let b = self.ensure_ptr_value(b)?;
                let v = self.cast_f64_value(v)?;
                writeln!(
                    self.out,
                    "  call void @mp_rt_strbuilder_append_f64(ptr {b}, double {v})"
                )
                .map_err(|e| e.to_string())?;
            }
            MpirOpVoid::StrBuilderAppendBool { b, v } => {
                let b = self.ensure_ptr_value(b)?;
                let v = self.cast_i32_value(v)?;
                writeln!(
                    self.out,
                    "  call void @mp_rt_strbuilder_append_bool(ptr {b}, i32 {v})"
                )
                .map_err(|e| e.to_string())?;
            }
            MpirOpVoid::PtrStore { to, p, v } => {
                let p = self.ensure_ptr_value(p)?;
                let v = self.value(v)?;
                writeln!(
                    self.out,
                    "  store {} {}, ptr {}",
                    self.cg.llvm_ty(*to),
                    v.repr,
                    p
                )
                .map_err(|e| e.to_string())?;
            }
            MpirOpVoid::Panic { msg } => {
                let msg = self.ensure_ptr_value(msg)?;
                writeln!(self.out, "  call void @mp_rt_panic(ptr {msg})")
                    .map_err(|e| e.to_string())?;
            }
            MpirOpVoid::GpuBarrier => {
                // CPU fallback runtime has no explicit barrier primitive.
            }
            MpirOpVoid::GpuBufferStore { buf, idx, val } => {
                self.emit_gpu_buffer_store(buf, idx, val)?;
            }
            MpirOpVoid::ArcRetain { v } => {
                let op = self.value(v)?;
                if op.ty.starts_with('{') {
                    self.emit_arc_retain_composite(&op)?;
                } else {
                    let p = self.ensure_ptr(op)?;
                    writeln!(self.out, "  call void @mp_rt_retain_strong(ptr {p})")
                        .map_err(|e| e.to_string())?;
                }
            }
            MpirOpVoid::ArcRelease { v } => {
                let op = self.value(v)?;
                if op.ty.starts_with('{') {
                    self.emit_arc_release_composite(&op)?;
                } else {
                    let p = self.ensure_ptr(op)?;
                    writeln!(self.out, "  call void @mp_rt_release_strong(ptr {p})")
                        .map_err(|e| e.to_string())?;
                }
            }
            MpirOpVoid::ArcRetainWeak { v } => {
                let op = self.value(v)?;
                if op.ty.starts_with('{') {
                    self.emit_arc_retain_composite(&op)?;
                } else {
                    let p = self.ensure_ptr(op)?;
                    writeln!(self.out, "  call void @mp_rt_retain_weak(ptr {p})")
                        .map_err(|e| e.to_string())?;
                }
            }
            MpirOpVoid::ArcReleaseWeak { v } => {
                let op = self.value(v)?;
                if op.ty.starts_with('{') {
                    self.emit_arc_release_composite(&op)?;
                } else {
                    let p = self.ensure_ptr(op)?;
                    writeln!(self.out, "  call void @mp_rt_release_weak(ptr {p})")
                        .map_err(|e| e.to_string())?;
                }
            }
        }
        Ok(())
    }

    fn codegen_term(&mut self, t: &MpirTerminator) -> Result<(), String> {
        match t {
            MpirTerminator::Ret(Some(v)) => {
                let rv = self.value(v)?;
                let ret_ty = self.cg.llvm_ty(self.f.ret_ty);
                if ret_ty == "void" {
                    writeln!(self.out, "  ret void").map_err(|e| e.to_string())?;
                } else {
                    writeln!(self.out, "  ret {} {}", rv.ty, rv.repr).map_err(|e| e.to_string())?;
                }
            }
            MpirTerminator::Ret(None) => {
                let ret_ty = self.cg.llvm_ty(self.f.ret_ty);
                if ret_ty == "void" {
                    writeln!(self.out, "  ret void").map_err(|e| e.to_string())?;
                } else {
                    writeln!(self.out, "  ret {} {}", ret_ty, self.zero_lit(&ret_ty))
                        .map_err(|e| e.to_string())?;
                }
            }
            MpirTerminator::Br(bb) => {
                writeln!(self.out, "  br label %bb{}", bb.0).map_err(|e| e.to_string())?;
            }
            MpirTerminator::Cbr {
                cond,
                then_bb,
                else_bb,
            } => {
                let cond = self.cond_i1(cond)?;
                writeln!(
                    self.out,
                    "  br i1 {cond}, label %bb{}, label %bb{}",
                    then_bb.0, else_bb.0
                )
                .map_err(|e| e.to_string())?;
            }
            MpirTerminator::Switch { val, arms, default } => {
                let val = self.value(val)?;
                let mut arm_text = String::new();
                for (c, bb) in arms {
                    let lit = self.const_lit(c)?;
                    write!(arm_text, "    {} {}, label %bb{}\n", val.ty, lit, bb.0)
                        .map_err(|e| e.to_string())?;
                }
                writeln!(
                    self.out,
                    "  switch {} {}, label %bb{} [",
                    val.ty, val.repr, default.0
                )
                .map_err(|e| e.to_string())?;
                write!(self.out, "{arm_text}").map_err(|e| e.to_string())?;
                writeln!(self.out, "  ]").map_err(|e| e.to_string())?;
            }
            MpirTerminator::Unreachable => {
                writeln!(self.out, "  unreachable").map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }

    fn emit_bin(
        &mut self,
        dst_id: magpie_types::LocalId,
        dst_ty_id: TypeId,
        op: &str,
        lhs: &MpirValue,
        rhs: &MpirValue,
    ) -> Result<(), String> {
        let lhs = self.value(lhs)?;
        let rhs = self.value(rhs)?;
        let dst_ty = self.cg.llvm_ty(dst_ty_id);
        let dst = format!("%l{}", dst_id.0);
        writeln!(
            self.out,
            "  {dst} = {op} {dst_ty} {}, {}",
            lhs.repr, rhs.repr
        )
        .map_err(|e| e.to_string())?;
        self.set_local(dst_id, dst_ty_id, dst_ty, dst);
        Ok(())
    }

    fn emit_checked(
        &mut self,
        dst: magpie_types::LocalId,
        dst_ty: TypeId,
        lhs: &MpirValue,
        rhs: &MpirValue,
        kind: &str,
    ) -> Result<(), String> {
        let l = self.value(lhs)?;
        let r = self.value(rhs)?;
        let int_ty = l.ty.clone();
        let bits = int_bits(&int_ty).unwrap_or(32);
        let intr_name = format!("@llvm.{kind}.with.overflow.i{bits}");
        let pair_ty = format!("{{ {}, i1 }}", int_ty);
        let call_tmp = self.tmp();
        writeln!(
            self.out,
            "  {call_tmp} = call {pair_ty} {intr_name}({int_ty} {}, {int_ty} {})",
            l.repr, r.repr
        )
        .map_err(|e| e.to_string())?;
        let dst_name = format!("%l{}", dst.0);
        let expect = self.cg.llvm_ty(dst_ty);
        if expect == pair_ty {
            self.set_local(dst, dst_ty, expect, call_tmp);
            return Ok(());
        }
        let val_tmp = self.tmp();
        writeln!(
            self.out,
            "  {val_tmp} = extractvalue {pair_ty} {call_tmp}, 0"
        )
        .map_err(|e| e.to_string())?;
        let ov_tmp = self.tmp();
        writeln!(
            self.out,
            "  {ov_tmp} = extractvalue {pair_ty} {call_tmp}, 1"
        )
        .map_err(|e| e.to_string())?;
        let none_tag = self.tmp();
        writeln!(self.out, "  {none_tag} = xor i1 {ov_tmp}, true").map_err(|e| e.to_string())?;
        let agg0 = self.tmp();
        writeln!(
            self.out,
            "  {agg0} = insertvalue {expect} undef, {int_ty} {val_tmp}, 0"
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            self.out,
            "  {dst_name} = insertvalue {expect} {agg0}, i1 {none_tag}, 1"
        )
        .map_err(|e| e.to_string())?;
        self.set_local(dst, dst_ty, expect, dst_name);
        Ok(())
    }

    fn emit_cast(
        &mut self,
        dst_id: magpie_types::LocalId,
        dst_ty_id: TypeId,
        to_ty: TypeId,
        v: &MpirValue,
    ) -> Result<(), String> {
        let src = self.value(v)?;
        let src_ty = src.ty.clone();
        let dst_ty = self.cg.llvm_ty(to_ty);
        let dst = format!("%l{}", dst_id.0);

        if src_ty == dst_ty {
            self.assign_or_copy(dst_id, dst_ty_id, src)?;
            return Ok(());
        }

        if src_ty == "ptr" && dst_ty == "ptr" {
            writeln!(self.out, "  {dst} = bitcast ptr {} to ptr", src.repr)
                .map_err(|e| e.to_string())?;
        } else if src_ty == "ptr" && is_int_ty(&dst_ty) {
            writeln!(
                self.out,
                "  {dst} = ptrtoint ptr {} to {}",
                src.repr, dst_ty
            )
            .map_err(|e| e.to_string())?;
        } else if is_int_ty(&src_ty) && dst_ty == "ptr" {
            writeln!(
                self.out,
                "  {dst} = inttoptr {} {} to ptr",
                src_ty, src.repr
            )
            .map_err(|e| e.to_string())?;
        } else if is_int_ty(&src_ty) && is_int_ty(&dst_ty) {
            let src_bits = int_bits(&src_ty).unwrap_or(64);
            let dst_bits = int_bits(&dst_ty).unwrap_or(64);
            let signed = self.cg.is_signed_int(src.ty_id);
            let op = if src_bits == dst_bits {
                "add"
            } else if src_bits < dst_bits {
                if signed {
                    "sext"
                } else {
                    "zext"
                }
            } else {
                "trunc"
            };
            if op == "add" {
                writeln!(self.out, "  {dst} = add {} {}, 0", dst_ty, src.repr)
                    .map_err(|e| e.to_string())?;
            } else {
                writeln!(
                    self.out,
                    "  {dst} = {op} {} {} to {}",
                    src_ty, src.repr, dst_ty
                )
                .map_err(|e| e.to_string())?;
            }
        } else if is_float_ty(&src_ty) && is_float_ty(&dst_ty) {
            let src_bits = float_bits(&src_ty).unwrap_or(64);
            let dst_bits = float_bits(&dst_ty).unwrap_or(64);
            let op = if src_bits < dst_bits {
                "fpext"
            } else {
                "fptrunc"
            };
            writeln!(
                self.out,
                "  {dst} = {op} {} {} to {}",
                src_ty, src.repr, dst_ty
            )
            .map_err(|e| e.to_string())?;
        } else if is_int_ty(&src_ty) && is_float_ty(&dst_ty) {
            let op = if self.cg.is_signed_int(src.ty_id) {
                "sitofp"
            } else {
                "uitofp"
            };
            writeln!(
                self.out,
                "  {dst} = {op} {} {} to {}",
                src_ty, src.repr, dst_ty
            )
            .map_err(|e| e.to_string())?;
        } else if is_float_ty(&src_ty) && is_int_ty(&dst_ty) {
            let op = if self.cg.is_signed_int(dst_ty_id) {
                "fptosi"
            } else {
                "fptoui"
            };
            writeln!(
                self.out,
                "  {dst} = {op} {} {} to {}",
                src_ty, src.repr, dst_ty
            )
            .map_err(|e| e.to_string())?;
        } else {
            writeln!(
                self.out,
                "  {dst} = bitcast {} {} to {}",
                src_ty, src.repr, dst_ty
            )
            .map_err(|e| e.to_string())?;
        }
        self.set_local(dst_id, dst_ty_id, dst_ty, dst);
        Ok(())
    }

    fn emit_arr_set(
        &mut self,
        arr: &MpirValue,
        idx: &MpirValue,
        val: &MpirValue,
    ) -> Result<(), String> {
        let arr = self.ensure_ptr_value(arr)?;
        let idx = self.cast_i64_value(idx)?;
        let val = self.value(val)?;
        let slot = self.stack_slot(&val)?;
        writeln!(
            self.out,
            "  call void @mp_rt_arr_set(ptr {arr}, i64 {idx}, ptr {slot}, i64 {})",
            self.cg.size_of_ty(val.ty_id)
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    fn emit_arr_push(&mut self, arr: &MpirValue, val: &MpirValue) -> Result<(), String> {
        let arr = self.ensure_ptr_value(arr)?;
        let val = self.value(val)?;
        let slot = self.stack_slot(&val)?;
        writeln!(
            self.out,
            "  call void @mp_rt_arr_push(ptr {arr}, ptr {slot}, i64 {})",
            self.cg.size_of_ty(val.ty_id)
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    fn emit_arr_pop(
        &mut self,
        dst_id: magpie_types::LocalId,
        dst_ty_id: TypeId,
        dst_ty: &str,
        arr: &MpirValue,
    ) -> Result<(), String> {
        let arr = self.ensure_ptr_value(arr)?;
        match self.cg.kind_of(dst_ty_id) {
            Some(TypeKind::BuiltinOption { inner }) => {
                let inner_ty = self.cg.llvm_storage_ty(*inner);
                let out = self.tmp();
                writeln!(self.out, "  {out} = alloca {inner_ty}").map_err(|e| e.to_string())?;
                let stat = self.tmp();
                writeln!(
                    self.out,
                    "  {stat} = call i32 @mp_rt_arr_pop(ptr {arr}, ptr {out}, i64 {})",
                    self.cg.size_of_ty(*inner)
                )
                .map_err(|e| e.to_string())?;
                let loaded = self.tmp();
                writeln!(self.out, "  {loaded} = load {inner_ty}, ptr {out}")
                    .map_err(|e| e.to_string())?;
                let ok = self.tmp();
                writeln!(self.out, "  {ok} = icmp eq i32 {stat}, 1").map_err(|e| e.to_string())?;
                let agg0 = self.tmp();
                writeln!(
                    self.out,
                    "  {agg0} = insertvalue {dst_ty} undef, {inner_ty} {loaded}, 0"
                )
                .map_err(|e| e.to_string())?;
                let dst = format!("%l{}", dst_id.0);
                writeln!(
                    self.out,
                    "  {dst} = insertvalue {dst_ty} {agg0}, i1 {ok}, 1"
                )
                .map_err(|e| e.to_string())?;
                self.set_local(dst_id, dst_ty_id, dst_ty.to_string(), dst);
            }
            _ => {
                let out = self.tmp();
                writeln!(self.out, "  {out} = alloca {dst_ty}").map_err(|e| e.to_string())?;
                let stat = self.tmp();
                writeln!(
                    self.out,
                    "  {stat} = call i32 @mp_rt_arr_pop(ptr {arr}, ptr {out}, i64 {})",
                    self.cg.size_of_ty(dst_ty_id)
                )
                .map_err(|e| e.to_string())?;
                let _ = stat;
                let dst = format!("%l{}", dst_id.0);
                writeln!(self.out, "  {dst} = load {dst_ty}, ptr {out}")
                    .map_err(|e| e.to_string())?;
                self.set_local(dst_id, dst_ty_id, dst_ty.to_string(), dst);
            }
        }
        Ok(())
    }

    fn emit_map_set(
        &mut self,
        map: &MpirValue,
        key: &MpirValue,
        val: &MpirValue,
    ) -> Result<(), String> {
        let map = self.ensure_ptr_value(map)?;
        let key = self.value(key)?;
        let val = self.value(val)?;
        let keyp = self.stack_slot(&key)?;
        let valp = self.stack_slot(&val)?;
        writeln!(
            self.out,
            "  call void @mp_rt_map_set(ptr {map}, ptr {keyp}, i64 {}, ptr {valp}, i64 {})",
            self.cg.size_of_ty(key.ty_id),
            self.cg.size_of_ty(val.ty_id)
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    fn emit_new_struct(
        &mut self,
        dst_id: magpie_types::LocalId,
        dst_ty_id: TypeId,
        new_ty: TypeId,
        fields: &[(String, MpirValue)],
    ) -> Result<(), String> {
        match self.cg.kind_of(new_ty) {
            Some(TypeKind::ValueStruct { sid }) => {
                self.emit_new_value_struct(dst_id, dst_ty_id, sid, fields)
            }
            Some(TypeKind::HeapHandle {
                base: HeapBase::UserType { type_sid, .. },
                ..
            }) => self.emit_new_heap_struct(dst_id, dst_ty_id, new_ty, type_sid, fields),
            _ => self.set_default(dst_id, dst_ty_id),
        }
    }

    fn emit_new_value_struct(
        &mut self,
        dst_id: magpie_types::LocalId,
        dst_ty_id: TypeId,
        sid: &Sid,
        fields: &[(String, MpirValue)],
    ) -> Result<(), String> {
        let storage_ty = self.cg.llvm_storage_ty(dst_ty_id);
        let slot = self.tmp();
        writeln!(self.out, "  {slot} = alloca {storage_ty}").map_err(|e| e.to_string())?;
        writeln!(self.out, "  store {storage_ty} zeroinitializer, ptr {slot}")
            .map_err(|e| e.to_string())?;

        for (name, value) in fields {
            if let Some((field_ty, offset)) = self.cg.type_ctx.user_struct_field(sid, name) {
                self.store_value_at_offset(&slot, offset, field_ty, value)?;
            }
        }

        let loaded = format!("%l{}", dst_id.0);
        writeln!(self.out, "  {loaded} = load {storage_ty}, ptr {slot}")
            .map_err(|e| e.to_string())?;
        self.set_local(dst_id, dst_ty_id, self.cg.llvm_ty(dst_ty_id), loaded);
        Ok(())
    }

    fn emit_new_heap_struct(
        &mut self,
        dst_id: magpie_types::LocalId,
        dst_ty_id: TypeId,
        new_ty: TypeId,
        sid: &Sid,
        fields: &[(String, MpirValue)],
    ) -> Result<(), String> {
        let layout = self
            .cg
            .type_ctx
            .user_struct_layout(sid)
            .unwrap_or(magpie_types::TypeLayout {
                size: 0,
                align: 1,
                fields: Vec::new(),
            });
        let payload_size = layout.size.max(1);
        let payload_align = layout.align.max(1);

        let obj = format!("%l{}", dst_id.0);
        writeln!(
            self.out,
            "  {obj} = call ptr @mp_rt_alloc(i32 {}, i64 {payload_size}, i64 {payload_align}, i32 1)",
            new_ty.0
        )
        .map_err(|e| e.to_string())?;

        let payload_base = self.tmp();
        writeln!(
            self.out,
            "  {payload_base} = getelementptr i8, ptr {obj}, i64 {}",
            MP_RT_HEADER_SIZE
        )
        .map_err(|e| e.to_string())?;

        for (name, value) in fields {
            if let Some((field_ty, offset)) = self.cg.type_ctx.user_struct_field(sid, name) {
                self.store_value_at_offset(&payload_base, offset, field_ty, value)?;
            }
        }

        self.set_local(dst_id, dst_ty_id, self.cg.llvm_ty(dst_ty_id), obj);
        Ok(())
    }

    fn emit_get_field(
        &mut self,
        dst_id: magpie_types::LocalId,
        dst_ty_id: TypeId,
        obj: &MpirValue,
        field: &str,
    ) -> Result<(), String> {
        let obj_op = self.value(obj)?;
        let Some((field_ty, field_offset, object_payload_offset)) =
            self.resolve_field_access(obj_op.ty_id, field)
        else {
            return self.set_default(dst_id, dst_ty_id);
        };

        if self.cg.llvm_storage_ty(field_ty) == "i8" && self.cg.llvm_ty(dst_ty_id) == "void" {
            return self.set_default(dst_id, dst_ty_id);
        }

        let obj_ptr = self.ensure_ptr(obj_op)?;
        let base = if object_payload_offset == 0 {
            obj_ptr
        } else {
            let tmp = self.tmp();
            writeln!(
                self.out,
                "  {tmp} = getelementptr i8, ptr {obj_ptr}, i64 {object_payload_offset}"
            )
            .map_err(|e| e.to_string())?;
            tmp
        };

        let ptr = self.tmp();
        writeln!(
            self.out,
            "  {ptr} = getelementptr i8, ptr {base}, i64 {field_offset}"
        )
        .map_err(|e| e.to_string())?;

        let storage_ty = self.cg.llvm_storage_ty(field_ty);
        let loaded = self.tmp();
        writeln!(self.out, "  {loaded} = load {storage_ty}, ptr {ptr}")
            .map_err(|e| e.to_string())?;
        self.assign_or_copy(
            dst_id,
            dst_ty_id,
            Operand {
                ty: storage_ty,
                ty_id: field_ty,
                repr: loaded,
            },
        )
    }

    fn emit_set_field(
        &mut self,
        obj: &MpirValue,
        field: &str,
        value: &MpirValue,
    ) -> Result<(), String> {
        let obj_op = self.value(obj)?;
        let Some((field_ty, field_offset, object_payload_offset)) =
            self.resolve_field_access(obj_op.ty_id, field)
        else {
            return Ok(());
        };

        let obj_ptr = self.ensure_ptr(obj_op)?;
        let base = if object_payload_offset == 0 {
            obj_ptr
        } else {
            let tmp = self.tmp();
            writeln!(
                self.out,
                "  {tmp} = getelementptr i8, ptr {obj_ptr}, i64 {object_payload_offset}"
            )
            .map_err(|e| e.to_string())?;
            tmp
        };

        self.store_value_at_offset(&base, field_offset, field_ty, value)
    }

    fn resolve_field_access(&self, ty: TypeId, field: &str) -> Option<(TypeId, u64, u64)> {
        match self.cg.kind_of(ty) {
            Some(TypeKind::ValueStruct { sid }) => self
                .cg
                .type_ctx
                .user_struct_field(sid, field)
                .map(|(field_ty, field_offset)| (field_ty, field_offset, 0)),
            Some(TypeKind::HeapHandle {
                base: HeapBase::UserType { type_sid, .. },
                ..
            }) => self
                .cg
                .type_ctx
                .user_struct_field(type_sid, field)
                .map(|(field_ty, field_offset)| (field_ty, field_offset, MP_RT_HEADER_SIZE)),
            _ => None,
        }
    }

    fn store_value_at_offset(
        &mut self,
        base_ptr: &str,
        byte_offset: u64,
        field_ty: TypeId,
        value: &MpirValue,
    ) -> Result<(), String> {
        let ptr = self.tmp();
        writeln!(
            self.out,
            "  {ptr} = getelementptr i8, ptr {base_ptr}, i64 {byte_offset}"
        )
        .map_err(|e| e.to_string())?;

        let field_storage_ty = self.cg.llvm_storage_ty(field_ty);
        let value_op = self.value(value)?;

        if field_storage_ty == "i8" && value_op.ty == "void" {
            writeln!(self.out, "  store i8 0, ptr {ptr}").map_err(|e| e.to_string())?;
            return Ok(());
        }

        if value_op.ty == field_storage_ty {
            writeln!(
                self.out,
                "  store {field_storage_ty} {}, ptr {ptr}",
                value_op.repr
            )
            .map_err(|e| e.to_string())?;
            return Ok(());
        }

        writeln!(
            self.out,
            "  store {field_storage_ty} {}, ptr {ptr}",
            self.zero_lit(&field_storage_ty)
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    fn emit_call_indirect(
        &mut self,
        dst_id: magpie_types::LocalId,
        dst_ty_id: TypeId,
        callee: &MpirValue,
        args: &[MpirValue],
        force_void: bool,
    ) -> Result<(), String> {
        let callee = self.ensure_ptr_value(callee)?;
        let fn_ptr = self.tmp();
        writeln!(
            self.out,
            "  {fn_ptr} = call ptr @mp_rt_callable_fn_ptr(ptr {callee})"
        )
        .map_err(|e| e.to_string())?;
        let data_ptr = self.tmp();
        writeln!(
            self.out,
            "  {data_ptr} = call ptr @mp_rt_callable_data_ptr(ptr {callee})"
        )
        .map_err(|e| e.to_string())?;

        let args = if args.is_empty() {
            format!("ptr {data_ptr}")
        } else {
            format!("ptr {data_ptr}, {}", self.call_args(args)?)
        };
        let dst_ty = self.cg.llvm_ty(dst_ty_id);
        if force_void || dst_ty == "void" {
            writeln!(self.out, "  call void {fn_ptr}({args})").map_err(|e| e.to_string())?;
            return self.set_default(dst_id, dst_ty_id);
        }

        let dst = format!("%l{}", dst_id.0);
        writeln!(self.out, "  {dst} = call {dst_ty} {fn_ptr}({args})")
            .map_err(|e| e.to_string())?;
        self.set_local(dst_id, dst_ty_id, dst_ty, dst);
        Ok(())
    }

    fn emit_call_indirect_void(
        &mut self,
        callee: &MpirValue,
        args: &[MpirValue],
    ) -> Result<(), String> {
        let callee = self.ensure_ptr_value(callee)?;
        let fn_ptr = self.tmp();
        writeln!(
            self.out,
            "  {fn_ptr} = call ptr @mp_rt_callable_fn_ptr(ptr {callee})"
        )
        .map_err(|e| e.to_string())?;
        let data_ptr = self.tmp();
        writeln!(
            self.out,
            "  {data_ptr} = call ptr @mp_rt_callable_data_ptr(ptr {callee})"
        )
        .map_err(|e| e.to_string())?;
        let args = if args.is_empty() {
            format!("ptr {data_ptr}")
        } else {
            format!("ptr {data_ptr}, {}", self.call_args(args)?)
        };
        writeln!(self.out, "  call void {fn_ptr}({args})").map_err(|e| e.to_string())
    }

    fn emit_enum_new(
        &mut self,
        dst_id: magpie_types::LocalId,
        dst_ty_id: TypeId,
        variant: &str,
        args: &[(String, MpirValue)],
    ) -> Result<(), String> {
        match self.cg.kind_of(dst_ty_id) {
            Some(TypeKind::BuiltinOption { inner }) => {
                let agg_ty = self.cg.llvm_ty(dst_ty_id);
                let payload_ty = self.cg.llvm_storage_ty(*inner);
                let tag_value = if variant == "Some" { "1" } else { "0" };
                let payload = args
                    .iter()
                    .find(|(name, _)| name == "v")
                    .and_then(|(_, value)| self.value(value).ok())
                    .filter(|op| op.ty == payload_ty)
                    .map(|op| op.repr)
                    .unwrap_or_else(|| self.zero_lit(&payload_ty));

                let tmp0 = self.tmp();
                writeln!(
                    self.out,
                    "  {tmp0} = insertvalue {agg_ty} undef, {payload_ty} {payload}, 0"
                )
                .map_err(|e| e.to_string())?;
                let dst = format!("%l{}", dst_id.0);
                writeln!(
                    self.out,
                    "  {dst} = insertvalue {agg_ty} {tmp0}, i1 {tag_value}, 1"
                )
                .map_err(|e| e.to_string())?;
                self.set_local(dst_id, dst_ty_id, agg_ty, dst);
                Ok(())
            }
            Some(TypeKind::BuiltinResult { ok, err }) => {
                let agg_ty = self.cg.llvm_ty(dst_ty_id);
                let ok_ty = self.cg.llvm_storage_ty(*ok);
                let err_ty = self.cg.llvm_storage_ty(*err);
                let is_err = variant == "Err";
                let ok_payload = args
                    .iter()
                    .find(|(name, _)| name == "v")
                    .and_then(|(_, value)| self.value(value).ok())
                    .filter(|op| op.ty == ok_ty)
                    .map(|op| op.repr)
                    .unwrap_or_else(|| self.zero_lit(&ok_ty));
                let err_payload = args
                    .iter()
                    .find(|(name, _)| name == "e")
                    .and_then(|(_, value)| self.value(value).ok())
                    .filter(|op| op.ty == err_ty)
                    .map(|op| op.repr)
                    .unwrap_or_else(|| self.zero_lit(&err_ty));

                let tmp0 = self.tmp();
                writeln!(
                    self.out,
                    "  {tmp0} = insertvalue {agg_ty} undef, i1 {}, 0",
                    if is_err { 1 } else { 0 }
                )
                .map_err(|e| e.to_string())?;
                let tmp1 = self.tmp();
                writeln!(
                    self.out,
                    "  {tmp1} = insertvalue {agg_ty} {tmp0}, {ok_ty} {}, 1",
                    if is_err {
                        self.zero_lit(&ok_ty)
                    } else {
                        ok_payload
                    }
                )
                .map_err(|e| e.to_string())?;
                let dst = format!("%l{}", dst_id.0);
                writeln!(
                    self.out,
                    "  {dst} = insertvalue {agg_ty} {tmp1}, {err_ty} {}, 2",
                    if is_err {
                        err_payload
                    } else {
                        self.zero_lit(&err_ty)
                    }
                )
                .map_err(|e| e.to_string())?;
                self.set_local(dst_id, dst_ty_id, agg_ty, dst);
                Ok(())
            }
            Some(TypeKind::ValueStruct { sid })
                if self.cg.type_ctx.user_enum_variants(sid).is_some() =>
            {
                self.emit_user_enum_new_value(dst_id, dst_ty_id, sid, variant, args)
            }
            Some(TypeKind::HeapHandle {
                base: HeapBase::UserType { type_sid, .. },
                ..
            }) if self.cg.type_ctx.user_enum_variants(type_sid).is_some() => {
                self.emit_user_enum_new_heap(dst_id, dst_ty_id, type_sid, variant, args)
            }
            _ => self.set_default(dst_id, dst_ty_id),
        }
    }

    fn emit_enum_tag(
        &mut self,
        dst_id: magpie_types::LocalId,
        dst_ty_id: TypeId,
        v: &MpirValue,
    ) -> Result<(), String> {
        let op = self.value(v)?;
        let tag_i32 = match self.cg.kind_of(op.ty_id) {
            Some(TypeKind::BuiltinOption { .. }) => {
                let tag = self.tmp();
                writeln!(self.out, "  {tag} = extractvalue {} {}, 1", op.ty, op.repr)
                    .map_err(|e| e.to_string())?;
                let z = self.tmp();
                writeln!(self.out, "  {z} = zext i1 {tag} to i32").map_err(|e| e.to_string())?;
                z
            }
            Some(TypeKind::BuiltinResult { .. }) => {
                let tag = self.tmp();
                writeln!(self.out, "  {tag} = extractvalue {} {}, 0", op.ty, op.repr)
                    .map_err(|e| e.to_string())?;
                let z = self.tmp();
                writeln!(self.out, "  {z} = zext i1 {tag} to i32").map_err(|e| e.to_string())?;
                z
            }
            Some(TypeKind::ValueStruct { sid })
                if self.cg.type_ctx.user_enum_variants(sid).is_some() =>
            {
                self.load_user_enum_tag_from_value(&op)?
            }
            Some(TypeKind::HeapHandle {
                base: HeapBase::UserType { type_sid, .. },
                ..
            }) if self.cg.type_ctx.user_enum_variants(type_sid).is_some() => {
                self.load_user_enum_tag_from_heap(&op)?
            }
            _ => "0".to_string(),
        };
        self.assign_cast_int(dst_id, dst_ty_id, tag_i32, "i32")
    }

    fn emit_enum_payload(
        &mut self,
        dst_id: magpie_types::LocalId,
        dst_ty_id: TypeId,
        variant: &str,
        v: &MpirValue,
    ) -> Result<(), String> {
        let op = self.value(v)?;
        match self.cg.kind_of(op.ty_id) {
            Some(TypeKind::BuiltinOption { inner }) => {
                if variant != "Some" {
                    return self.set_default(dst_id, dst_ty_id);
                }
                let payload_ty = self.cg.llvm_storage_ty(*inner);
                let payload = self.tmp();
                writeln!(
                    self.out,
                    "  {payload} = extractvalue {} {}, 0",
                    op.ty, op.repr
                )
                .map_err(|e| e.to_string())?;
                self.assign_or_copy(
                    dst_id,
                    dst_ty_id,
                    Operand {
                        ty: payload_ty,
                        ty_id: *inner,
                        repr: payload,
                    },
                )
            }
            Some(TypeKind::BuiltinResult { ok, err }) => {
                let (idx, payload_ty_id) = if variant == "Err" {
                    (2, *err)
                } else {
                    (1, *ok)
                };
                let payload_ty = self.cg.llvm_storage_ty(payload_ty_id);
                let payload = self.tmp();
                writeln!(
                    self.out,
                    "  {payload} = extractvalue {} {}, {}",
                    op.ty, op.repr, idx
                )
                .map_err(|e| e.to_string())?;
                self.assign_or_copy(
                    dst_id,
                    dst_ty_id,
                    Operand {
                        ty: payload_ty,
                        ty_id: payload_ty_id,
                        repr: payload,
                    },
                )
            }
            Some(TypeKind::ValueStruct { sid })
                if self.cg.type_ctx.user_enum_variants(sid).is_some() =>
            {
                self.emit_user_enum_payload_value(dst_id, dst_ty_id, sid, variant, &op)
            }
            Some(TypeKind::HeapHandle {
                base: HeapBase::UserType { type_sid, .. },
                ..
            }) if self.cg.type_ctx.user_enum_variants(type_sid).is_some() => {
                self.emit_user_enum_payload_heap(dst_id, dst_ty_id, type_sid, variant, &op)
            }
            _ => self.set_default(dst_id, dst_ty_id),
        }
    }

    fn emit_enum_is(
        &mut self,
        dst_id: magpie_types::LocalId,
        dst_ty_id: TypeId,
        variant: &str,
        v: &MpirValue,
    ) -> Result<(), String> {
        let op = self.value(v)?;
        let (tag_idx, expected) = match self.cg.kind_of(op.ty_id) {
            Some(TypeKind::BuiltinOption { .. }) => (1, if variant == "Some" { 1 } else { 0 }),
            Some(TypeKind::BuiltinResult { .. }) => (0, if variant == "Err" { 1 } else { 0 }),
            Some(TypeKind::ValueStruct { sid })
                if self.cg.type_ctx.user_enum_variants(sid).is_some() =>
            {
                let tag_i32 = self.load_user_enum_tag_from_value(&op)?;
                return self.emit_enum_is_from_tag(
                    dst_id,
                    dst_ty_id,
                    tag_i32,
                    self.cg
                        .type_ctx
                        .user_enum_variant_tag(sid, variant)
                        .unwrap_or(0),
                );
            }
            Some(TypeKind::HeapHandle {
                base: HeapBase::UserType { type_sid, .. },
                ..
            }) if self.cg.type_ctx.user_enum_variants(type_sid).is_some() => {
                let tag_i32 = self.load_user_enum_tag_from_heap(&op)?;
                return self.emit_enum_is_from_tag(
                    dst_id,
                    dst_ty_id,
                    tag_i32,
                    self.cg
                        .type_ctx
                        .user_enum_variant_tag(type_sid, variant)
                        .unwrap_or(0),
                );
            }
            _ => return self.set_default(dst_id, dst_ty_id),
        };
        let tag = self.tmp();
        writeln!(
            self.out,
            "  {tag} = extractvalue {} {}, {}",
            op.ty, op.repr, tag_idx
        )
        .map_err(|e| e.to_string())?;
        let tag_i32 = self.tmp();
        writeln!(self.out, "  {tag_i32} = zext i1 {tag} to i32").map_err(|e| e.to_string())?;
        self.emit_enum_is_from_tag(dst_id, dst_ty_id, tag_i32, expected)
    }

    fn emit_enum_is_from_tag(
        &mut self,
        dst_id: magpie_types::LocalId,
        dst_ty_id: TypeId,
        tag_i32: String,
        expected: i32,
    ) -> Result<(), String> {
        let is_variant = self.tmp();
        writeln!(
            self.out,
            "  {is_variant} = icmp eq i32 {tag_i32}, {expected}"
        )
        .map_err(|e| e.to_string())?;
        self.assign_cast_int(dst_id, dst_ty_id, is_variant, "i1")
    }

    fn emit_user_enum_new_value(
        &mut self,
        dst_id: magpie_types::LocalId,
        dst_ty_id: TypeId,
        sid: &Sid,
        variant: &str,
        args: &[(String, MpirValue)],
    ) -> Result<(), String> {
        let storage_ty = self.cg.llvm_storage_ty(dst_ty_id);
        let slot = self.tmp();
        writeln!(self.out, "  {slot} = alloca {storage_ty}").map_err(|e| e.to_string())?;
        writeln!(self.out, "  store {storage_ty} zeroinitializer, ptr {slot}")
            .map_err(|e| e.to_string())?;

        let tag = self
            .cg
            .type_ctx
            .user_enum_variant_tag(sid, variant)
            .unwrap_or(0);
        writeln!(self.out, "  store i32 {tag}, ptr {slot}").map_err(|e| e.to_string())?;

        let payload_offset = self.cg.type_ctx.user_enum_payload_offset(sid).unwrap_or(0);
        if let Some((_, fields)) = self
            .cg
            .type_ctx
            .user_enum_variants(sid)
            .and_then(|variants| variants.iter().find(|(name, _)| name == variant))
        {
            for (idx, (field_name, field_ty)) in fields.iter().enumerate() {
                let Some(value) = self.enum_variant_arg(args, field_name, idx) else {
                    continue;
                };
                if let Some((_, field_offset)) = self
                    .cg
                    .type_ctx
                    .user_enum_variant_field(sid, variant, field_name)
                {
                    self.store_value_at_offset(
                        &slot,
                        payload_offset.saturating_add(field_offset),
                        *field_ty,
                        value,
                    )?;
                }
            }
        }

        let dst = format!("%l{}", dst_id.0);
        writeln!(self.out, "  {dst} = load {storage_ty}, ptr {slot}").map_err(|e| e.to_string())?;
        self.set_local(dst_id, dst_ty_id, self.cg.llvm_ty(dst_ty_id), dst);
        Ok(())
    }

    fn emit_user_enum_new_heap(
        &mut self,
        dst_id: magpie_types::LocalId,
        dst_ty_id: TypeId,
        sid: &Sid,
        variant: &str,
        args: &[(String, MpirValue)],
    ) -> Result<(), String> {
        let layout = self
            .cg
            .type_ctx
            .user_enum_layout(sid)
            .unwrap_or(magpie_types::TypeLayout {
                size: 0,
                align: 1,
                fields: Vec::new(),
            });
        let payload_size = layout.size.max(1);
        let payload_align = layout.align.max(1);
        let dst = format!("%l{}", dst_id.0);
        writeln!(
            self.out,
            "  {dst} = call ptr @mp_rt_alloc(i32 {}, i64 {payload_size}, i64 {payload_align}, i32 1)",
            dst_ty_id.0
        )
        .map_err(|e| e.to_string())?;

        let payload_base = self.tmp();
        writeln!(
            self.out,
            "  {payload_base} = getelementptr i8, ptr {dst}, i64 {MP_RT_HEADER_SIZE}"
        )
        .map_err(|e| e.to_string())?;

        let tag = self
            .cg
            .type_ctx
            .user_enum_variant_tag(sid, variant)
            .unwrap_or(0);
        writeln!(self.out, "  store i32 {tag}, ptr {payload_base}").map_err(|e| e.to_string())?;

        let payload_offset = self.cg.type_ctx.user_enum_payload_offset(sid).unwrap_or(0);
        if let Some((_, fields)) = self
            .cg
            .type_ctx
            .user_enum_variants(sid)
            .and_then(|variants| variants.iter().find(|(name, _)| name == variant))
        {
            for (idx, (field_name, field_ty)) in fields.iter().enumerate() {
                let Some(value) = self.enum_variant_arg(args, field_name, idx) else {
                    continue;
                };
                if let Some((_, field_offset)) = self
                    .cg
                    .type_ctx
                    .user_enum_variant_field(sid, variant, field_name)
                {
                    self.store_value_at_offset(
                        &payload_base,
                        payload_offset.saturating_add(field_offset),
                        *field_ty,
                        value,
                    )?;
                }
            }
        }

        self.set_local(dst_id, dst_ty_id, self.cg.llvm_ty(dst_ty_id), dst);
        Ok(())
    }

    fn emit_user_enum_payload_value(
        &mut self,
        dst_id: magpie_types::LocalId,
        dst_ty_id: TypeId,
        sid: &Sid,
        variant: &str,
        op: &Operand,
    ) -> Result<(), String> {
        let Some((field_ty, byte_offset)) = self.first_variant_field_offset(sid, variant) else {
            return self.set_default(dst_id, dst_ty_id);
        };

        let slot = self.tmp();
        writeln!(self.out, "  {slot} = alloca {}", op.ty).map_err(|e| e.to_string())?;
        writeln!(self.out, "  store {} {}, ptr {slot}", op.ty, op.repr)
            .map_err(|e| e.to_string())?;
        self.load_field_from_offset(dst_id, dst_ty_id, field_ty, &slot, byte_offset)
    }

    fn emit_user_enum_payload_heap(
        &mut self,
        dst_id: magpie_types::LocalId,
        dst_ty_id: TypeId,
        sid: &Sid,
        variant: &str,
        op: &Operand,
    ) -> Result<(), String> {
        let Some((field_ty, payload_rel_offset)) = self.first_variant_field_offset(sid, variant)
        else {
            return self.set_default(dst_id, dst_ty_id);
        };
        let obj_ptr = self.ensure_ptr(op.clone())?;
        let payload_base = self.tmp();
        writeln!(
            self.out,
            "  {payload_base} = getelementptr i8, ptr {obj_ptr}, i64 {MP_RT_HEADER_SIZE}"
        )
        .map_err(|e| e.to_string())?;
        self.load_field_from_offset(
            dst_id,
            dst_ty_id,
            field_ty,
            &payload_base,
            payload_rel_offset,
        )
    }

    fn load_user_enum_tag_from_value(&mut self, op: &Operand) -> Result<String, String> {
        let slot = self.tmp();
        writeln!(self.out, "  {slot} = alloca {}", op.ty).map_err(|e| e.to_string())?;
        writeln!(self.out, "  store {} {}, ptr {slot}", op.ty, op.repr)
            .map_err(|e| e.to_string())?;
        let tag = self.tmp();
        writeln!(self.out, "  {tag} = load i32, ptr {slot}").map_err(|e| e.to_string())?;
        Ok(tag)
    }

    fn load_user_enum_tag_from_heap(&mut self, op: &Operand) -> Result<String, String> {
        let obj_ptr = self.ensure_ptr(op.clone())?;
        let payload_base = self.tmp();
        writeln!(
            self.out,
            "  {payload_base} = getelementptr i8, ptr {obj_ptr}, i64 {MP_RT_HEADER_SIZE}"
        )
        .map_err(|e| e.to_string())?;
        let tag = self.tmp();
        writeln!(self.out, "  {tag} = load i32, ptr {payload_base}").map_err(|e| e.to_string())?;
        Ok(tag)
    }

    fn first_variant_field_offset(&self, sid: &Sid, variant: &str) -> Option<(TypeId, u64)> {
        let payload_offset = self.cg.type_ctx.user_enum_payload_offset(sid).unwrap_or(0);
        let (_, fields) = self
            .cg
            .type_ctx
            .user_enum_variants(sid)?
            .iter()
            .find(|(name, _)| name == variant)?;
        let (field_name, field_ty) = fields.first()?;
        let (_, field_offset) = self
            .cg
            .type_ctx
            .user_enum_variant_field(sid, variant, field_name)?;
        Some((*field_ty, payload_offset.saturating_add(field_offset)))
    }

    fn enum_variant_arg<'b>(
        &self,
        args: &'b [(String, MpirValue)],
        field_name: &str,
        idx: usize,
    ) -> Option<&'b MpirValue> {
        args.iter()
            .find(|(name, _)| name == field_name)
            .or_else(|| args.get(idx))
            .map(|(_, value)| value)
    }

    fn load_field_from_offset(
        &mut self,
        dst_id: magpie_types::LocalId,
        dst_ty_id: TypeId,
        field_ty: TypeId,
        base_ptr: &str,
        byte_offset: u64,
    ) -> Result<(), String> {
        let field_ptr = self.tmp();
        writeln!(
            self.out,
            "  {field_ptr} = getelementptr i8, ptr {base_ptr}, i64 {byte_offset}"
        )
        .map_err(|e| e.to_string())?;
        let storage_ty = self.cg.llvm_storage_ty(field_ty);
        let loaded = self.tmp();
        writeln!(self.out, "  {loaded} = load {storage_ty}, ptr {field_ptr}")
            .map_err(|e| e.to_string())?;
        self.assign_or_copy(
            dst_id,
            dst_ty_id,
            Operand {
                ty: storage_ty,
                ty_id: field_ty,
                repr: loaded,
            },
        )
    }

    fn emit_gpu_buffer_load(
        &mut self,
        dst_id: magpie_types::LocalId,
        dst_ty_id: TypeId,
        buf: &MpirValue,
        idx: &MpirValue,
    ) -> Result<(), String> {
        let dst_ty = self.cg.llvm_ty(dst_ty_id);
        if dst_ty == "void" {
            self.set_default(dst_id, dst_ty_id)?;
            return Ok(());
        }

        let buf = self.ensure_ptr_value(buf)?;
        let idx = self.cast_i64_value(idx)?;
        let elem_size = self.cg.size_of_ty(dst_ty_id).max(1);
        let offset = self.tmp();
        writeln!(self.out, "  {offset} = mul i64 {idx}, {elem_size}").map_err(|e| e.to_string())?;

        let storage_ty = self.cg.llvm_storage_ty(dst_ty_id);
        let slot = self.tmp();
        writeln!(self.out, "  {slot} = alloca {storage_ty}").map_err(|e| e.to_string())?;
        writeln!(
            self.out,
            "  call i32 @mp_rt_gpu_buffer_read(ptr {buf}, i64 {offset}, ptr {slot}, i64 {elem_size})"
        )
        .map_err(|e| e.to_string())?;

        let loaded = self.tmp();
        writeln!(self.out, "  {loaded} = load {storage_ty}, ptr {slot}")
            .map_err(|e| e.to_string())?;
        self.assign_or_copy(
            dst_id,
            dst_ty_id,
            Operand {
                ty: storage_ty,
                ty_id: dst_ty_id,
                repr: loaded,
            },
        )
    }

    fn emit_gpu_buffer_store(
        &mut self,
        buf: &MpirValue,
        idx: &MpirValue,
        val: &MpirValue,
    ) -> Result<(), String> {
        let buf = self.ensure_ptr_value(buf)?;
        let idx = self.cast_i64_value(idx)?;
        let val = self.value(val)?;
        let elem_size = self.cg.size_of_ty(val.ty_id).max(1);
        let offset = self.tmp();
        writeln!(self.out, "  {offset} = mul i64 {idx}, {elem_size}").map_err(|e| e.to_string())?;

        if val.ty == "void" {
            let slot = self.tmp();
            writeln!(self.out, "  {slot} = alloca i8").map_err(|e| e.to_string())?;
            writeln!(self.out, "  store i8 0, ptr {slot}").map_err(|e| e.to_string())?;
            writeln!(
                self.out,
                "  call i32 @mp_rt_gpu_buffer_write(ptr {buf}, i64 {offset}, ptr {slot}, i64 1)"
            )
            .map_err(|e| e.to_string())?;
            return Ok(());
        }

        let slot = self.stack_slot(&val)?;
        writeln!(
            self.out,
            "  call i32 @mp_rt_gpu_buffer_write(ptr {buf}, i64 {offset}, ptr {slot}, i64 {elem_size})"
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    fn emit_gpu_shared(
        &mut self,
        dst_id: magpie_types::LocalId,
        dst_ty_id: TypeId,
        elem_ty: TypeId,
        size: &MpirValue,
    ) -> Result<(), String> {
        let dst_ty = self.cg.llvm_ty(dst_ty_id);
        if dst_ty == "void" {
            self.set_default(dst_id, dst_ty_id)?;
            return Ok(());
        }

        let count = self.cast_i64_value(size)?;
        let elem_size = self.cg.size_of_ty(elem_ty).max(1);
        let total = self.tmp();
        writeln!(self.out, "  {total} = mul i64 {count}, {elem_size}")
            .map_err(|e| e.to_string())?;

        let alloc = self.tmp();
        writeln!(self.out, "  {alloc} = alloca i8, i64 {total}").map_err(|e| e.to_string())?;

        if dst_ty == "ptr" {
            self.set_local(dst_id, dst_ty_id, dst_ty, alloc);
            return Ok(());
        }
        if is_int_ty(&dst_ty) {
            let casted = self.tmp();
            writeln!(self.out, "  {casted} = ptrtoint ptr {alloc} to {dst_ty}")
                .map_err(|e| e.to_string())?;
            self.set_local(dst_id, dst_ty_id, dst_ty, casted);
            return Ok(());
        }

        let casted = self.tmp();
        writeln!(self.out, "  {casted} = bitcast ptr {alloc} to {dst_ty}")
            .map_err(|e| e.to_string())?;
        self.set_local(dst_id, dst_ty_id, dst_ty, casted);
        Ok(())
    }

    fn emit_gpu_launch(
        &mut self,
        dst_id: magpie_types::LocalId,
        dst_ty_id: TypeId,
        device: &MpirValue,
        kernel: &Sid,
        groups: &MpirValue,
        threads: &MpirValue,
        args: &[MpirValue],
        is_async: bool,
    ) -> Result<(), String> {
        let device = self.ensure_ptr_value(device)?;
        let groups = self.cast_i32_value(groups)?;
        let threads = self.cast_i32_value(threads)?;
        let (args_blob, args_len) = self.build_gpu_launch_args_blob(kernel, args)?;

        let err_slot = self.tmp();
        writeln!(self.out, "  {err_slot} = alloca ptr").map_err(|e| e.to_string())?;
        writeln!(self.out, "  store ptr null, ptr {err_slot}").map_err(|e| e.to_string())?;

        let status = self.tmp();
        let sid_hash = sid_hash_64(kernel);

        if is_async {
            let fence_slot = self.tmp();
            writeln!(self.out, "  {fence_slot} = alloca ptr").map_err(|e| e.to_string())?;
            writeln!(self.out, "  store ptr null, ptr {fence_slot}").map_err(|e| e.to_string())?;
            writeln!(
                self.out,
                "  {status} = call i32 @mp_rt_gpu_launch_async(ptr {device}, i64 {sid_hash}, i32 {groups}, i32 1, i32 1, i32 {threads}, i32 1, i32 1, ptr {args_blob}, i64 {args_len}, ptr {fence_slot}, ptr {err_slot})"
            )
            .map_err(|e| e.to_string())?;
            let fence = self.tmp();
            writeln!(self.out, "  {fence} = load ptr, ptr {fence_slot}")
                .map_err(|e| e.to_string())?;
            let err = self.tmp();
            writeln!(self.out, "  {err} = load ptr, ptr {err_slot}").map_err(|e| e.to_string())?;
            self.assign_gpu_launch_result(dst_id, dst_ty_id, status, Some(fence), err)?;
        } else {
            writeln!(
                self.out,
                "  {status} = call i32 @mp_rt_gpu_launch_sync(ptr {device}, i64 {sid_hash}, i32 {groups}, i32 1, i32 1, i32 {threads}, i32 1, i32 1, ptr {args_blob}, i64 {args_len}, ptr {err_slot})"
            )
            .map_err(|e| e.to_string())?;
            let err = self.tmp();
            writeln!(self.out, "  {err} = load ptr, ptr {err_slot}").map_err(|e| e.to_string())?;
            self.assign_gpu_launch_result(dst_id, dst_ty_id, status, None, err)?;
        }

        Ok(())
    }

    fn assign_gpu_launch_result(
        &mut self,
        dst_id: magpie_types::LocalId,
        dst_ty_id: TypeId,
        status: String,
        ok_value: Option<String>,
        err_value: String,
    ) -> Result<(), String> {
        if let Some(TypeKind::BuiltinResult { ok, err }) = self.cg.kind_of(dst_ty_id) {
            let agg_ty = self.cg.llvm_ty(dst_ty_id);
            let ok_ty = self.cg.llvm_storage_ty(*ok);
            let err_ty = self.cg.llvm_storage_ty(*err);
            let ok_payload = ok_value.unwrap_or_else(|| self.zero_lit(&ok_ty));
            let err_zero = self.zero_lit(&err_ty);
            let err_payload = if err_ty == "ptr" {
                err_value
            } else {
                err_zero.clone()
            };

            let is_ok = self.tmp();
            writeln!(self.out, "  {is_ok} = icmp eq i32 {status}, 0").map_err(|e| e.to_string())?;

            let ok0 = self.tmp();
            writeln!(self.out, "  {ok0} = insertvalue {agg_ty} undef, i1 1, 0")
                .map_err(|e| e.to_string())?;
            let ok1 = self.tmp();
            writeln!(
                self.out,
                "  {ok1} = insertvalue {agg_ty} {ok0}, {ok_ty} {ok_payload}, 1"
            )
            .map_err(|e| e.to_string())?;
            let ok2 = self.tmp();
            writeln!(
                self.out,
                "  {ok2} = insertvalue {agg_ty} {ok1}, {err_ty} {err_zero}, 2"
            )
            .map_err(|e| e.to_string())?;

            let err0 = self.tmp();
            writeln!(self.out, "  {err0} = insertvalue {agg_ty} undef, i1 0, 0")
                .map_err(|e| e.to_string())?;
            let err1 = self.tmp();
            writeln!(
                self.out,
                "  {err1} = insertvalue {agg_ty} {err0}, {ok_ty} {}, 1",
                self.zero_lit(&ok_ty)
            )
            .map_err(|e| e.to_string())?;
            let err2 = self.tmp();
            writeln!(
                self.out,
                "  {err2} = insertvalue {agg_ty} {err1}, {err_ty} {err_payload}, 2"
            )
            .map_err(|e| e.to_string())?;

            let dst = format!("%l{}", dst_id.0);
            writeln!(
                self.out,
                "  {dst} = select i1 {is_ok}, {agg_ty} {ok2}, {agg_ty} {err2}"
            )
            .map_err(|e| e.to_string())?;
            self.set_local(dst_id, dst_ty_id, agg_ty, dst);
            return Ok(());
        }

        let dst_ty = self.cg.llvm_ty(dst_ty_id);
        if is_int_ty(&dst_ty) || dst_ty == "i1" {
            return self.assign_cast_int(dst_id, dst_ty_id, status, "i32");
        }
        self.set_default(dst_id, dst_ty_id)
    }

    fn build_gpu_launch_args_blob(
        &mut self,
        kernel: &Sid,
        args: &[MpirValue],
    ) -> Result<(String, u64), String> {
        if args.is_empty() {
            return Ok(("null".to_string(), 0));
        }

        let arg_ops = args
            .iter()
            .map(|arg| self.value(arg))
            .collect::<Result<Vec<_>, _>>()?;
        let kernel_params = self
            .cg
            .mpir
            .functions
            .iter()
            .find(|f| &f.sid == kernel)
            .map(|f| f.params.iter().map(|(_, ty)| *ty).collect::<Vec<_>>())
            .unwrap_or_default();

        #[derive(Clone, Copy)]
        struct ArgLayout {
            ty: TypeId,
            is_buffer: bool,
            offset: u64,
        }

        let mut layouts = Vec::with_capacity(arg_ops.len());
        let mut num_buffers = 0_u64;
        let mut scalar_offset = 0_u64;

        for (idx, op) in arg_ops.iter().enumerate() {
            let ty = kernel_params.get(idx).copied().unwrap_or(op.ty_id);
            if self.cg.is_gpu_buffer_param_ty(ty) {
                layouts.push(ArgLayout {
                    ty,
                    is_buffer: true,
                    offset: num_buffers.saturating_mul(8),
                });
                num_buffers = num_buffers.saturating_add(1);
                continue;
            }

            let size = self.cg.size_of_ty(ty).max(1);
            let align = size.clamp(1, 16);
            scalar_offset = align_up_u64(scalar_offset, align);
            layouts.push(ArgLayout {
                ty,
                is_buffer: false,
                offset: scalar_offset,
            });
            scalar_offset = scalar_offset.saturating_add(size);
        }

        let push_const_size = align_up_u64(scalar_offset, 16);
        let total_len = num_buffers
            .saturating_mul(8)
            .saturating_add(push_const_size);
        if total_len == 0 {
            return Ok(("null".to_string(), 0));
        }

        let blob = self.tmp();
        writeln!(self.out, "  {blob} = alloca i8, i64 {total_len}").map_err(|e| e.to_string())?;

        for (idx, layout) in layouts.iter().enumerate() {
            let arg = &arg_ops[idx];
            let offset = if layout.is_buffer {
                layout.offset
            } else {
                num_buffers.saturating_mul(8).saturating_add(layout.offset)
            };
            let slot = self.tmp();
            writeln!(
                self.out,
                "  {slot} = getelementptr i8, ptr {blob}, i64 {offset}"
            )
            .map_err(|e| e.to_string())?;

            if layout.is_buffer {
                let arg_ptr = self.ensure_ptr(arg.clone())?;
                writeln!(self.out, "  store ptr {arg_ptr}, ptr {slot}")
                    .map_err(|e| e.to_string())?;
                continue;
            }

            let storage_ty = self.cg.llvm_storage_ty(layout.ty);
            if storage_ty == "i8" && arg.ty == "void" {
                writeln!(self.out, "  store i8 0, ptr {slot}").map_err(|e| e.to_string())?;
                continue;
            }

            if arg.ty == storage_ty {
                writeln!(self.out, "  store {storage_ty} {}, ptr {slot}", arg.repr)
                    .map_err(|e| e.to_string())?;
            } else {
                writeln!(
                    self.out,
                    "  store {storage_ty} {}, ptr {slot}",
                    self.zero_lit(&storage_ty)
                )
                .map_err(|e| e.to_string())?;
            }
        }

        Ok((blob, total_len))
    }

    fn array_result_elem(&self, ty: TypeId) -> (TypeId, u64) {
        if let Some(TypeKind::HeapHandle {
            base: HeapBase::BuiltinArray { elem },
            ..
        }) = self.cg.kind_of(ty)
        {
            return (*elem, self.cg.size_of_ty(*elem));
        }
        (TypeId(0), 0)
    }

    fn array_elem_ty_for_value(&self, arr_ptr_repr: &str) -> Option<TypeId> {
        if !arr_ptr_repr.starts_with("%arg") && !arr_ptr_repr.starts_with("%l") {
            return None;
        }
        let digits = arr_ptr_repr
            .trim_start_matches("%arg")
            .trim_start_matches("%l");
        let local_id = digits.parse::<u32>().ok()?;
        let arr_ty = self.local_tys.get(&local_id).copied()?;
        if let Some(TypeKind::HeapHandle {
            base: HeapBase::BuiltinArray { elem },
            ..
        }) = self.cg.kind_of(arr_ty)
        {
            Some(*elem)
        } else {
            None
        }
    }

    fn call_args(&mut self, args: &[MpirValue]) -> Result<String, String> {
        let mut out = Vec::with_capacity(args.len());
        for a in args {
            let op = self.value(a)?;
            out.push(format!("{} {}", op.ty, op.repr));
        }
        Ok(out.join(", "))
    }

    fn assign_or_copy(
        &mut self,
        dst_id: magpie_types::LocalId,
        dst_ty: TypeId,
        src: Operand,
    ) -> Result<(), String> {
        let dst_ty_str = self.cg.llvm_ty(dst_ty);
        if src.ty == dst_ty_str {
            self.locals.insert(
                dst_id.0,
                Operand {
                    ty: dst_ty_str,
                    ty_id: dst_ty,
                    repr: src.repr,
                },
            );
            return Ok(());
        }
        let dst_name = format!("%l{}", dst_id.0);
        if src.ty == "ptr" && is_int_ty(&dst_ty_str) {
            writeln!(
                self.out,
                "  {dst_name} = ptrtoint ptr {} to {}",
                src.repr, dst_ty_str
            )
            .map_err(|e| e.to_string())?;
        } else if is_int_ty(&src.ty) && dst_ty_str == "ptr" {
            writeln!(
                self.out,
                "  {dst_name} = inttoptr {} {} to ptr",
                src.ty, src.repr
            )
            .map_err(|e| e.to_string())?;
        } else if is_int_ty(&src.ty) && is_int_ty(&dst_ty_str) {
            let sb = int_bits(&src.ty).unwrap_or(64);
            let db = int_bits(&dst_ty_str).unwrap_or(64);
            if sb == db {
                writeln!(self.out, "  {dst_name} = add {} {}, 0", src.ty, src.repr)
                    .map_err(|e| e.to_string())?;
            } else if sb < db {
                let ext = if self.cg.is_signed_int(src.ty_id) {
                    "sext"
                } else {
                    "zext"
                };
                writeln!(
                    self.out,
                    "  {dst_name} = {ext} {} {} to {}",
                    src.ty, src.repr, dst_ty_str
                )
                .map_err(|e| e.to_string())?;
            } else {
                writeln!(
                    self.out,
                    "  {dst_name} = trunc {} {} to {}",
                    src.ty, src.repr, dst_ty_str
                )
                .map_err(|e| e.to_string())?;
            }
        } else if is_float_ty(&src.ty) && is_float_ty(&dst_ty_str) {
            let sb = float_bits(&src.ty).unwrap_or(64);
            let db = float_bits(&dst_ty_str).unwrap_or(64);
            let op = if sb < db { "fpext" } else { "fptrunc" };
            writeln!(
                self.out,
                "  {dst_name} = {op} {} {} to {}",
                src.ty, src.repr, dst_ty_str
            )
            .map_err(|e| e.to_string())?;
        } else if is_int_ty(&src.ty) && is_float_ty(&dst_ty_str) {
            let op = if self.cg.is_signed_int(src.ty_id) {
                "sitofp"
            } else {
                "uitofp"
            };
            writeln!(
                self.out,
                "  {dst_name} = {op} {} {} to {}",
                src.ty, src.repr, dst_ty_str
            )
            .map_err(|e| e.to_string())?;
        } else if is_float_ty(&src.ty) && is_int_ty(&dst_ty_str) {
            let op = if self.cg.is_signed_int(dst_ty) {
                "fptosi"
            } else {
                "fptoui"
            };
            writeln!(
                self.out,
                "  {dst_name} = {op} {} {} to {}",
                src.ty, src.repr, dst_ty_str
            )
            .map_err(|e| e.to_string())?;
        } else {
            writeln!(
                self.out,
                "  {dst_name} = bitcast {} {} to {}",
                src.ty, src.repr, dst_ty_str
            )
            .map_err(|e| e.to_string())?;
        }
        self.locals.insert(
            dst_id.0,
            Operand {
                ty: dst_ty_str,
                ty_id: dst_ty,
                repr: dst_name,
            },
        );
        Ok(())
    }

    fn assign_or_copy_value(
        &mut self,
        dst_id: magpie_types::LocalId,
        dst_ty: TypeId,
        v: &MpirValue,
    ) -> Result<(), String> {
        let src = self.value(v)?;
        self.assign_or_copy(dst_id, dst_ty, src)
    }

    fn set_local(&mut self, id: magpie_types::LocalId, ty_id: TypeId, ty: String, repr: String) {
        self.locals.insert(id.0, Operand { ty, ty_id, repr });
    }

    fn set_default(&mut self, id: magpie_types::LocalId, ty_id: TypeId) -> Result<(), String> {
        let ty = self.cg.llvm_ty(ty_id);
        if ty == "void" {
            self.locals.insert(
                id.0,
                Operand {
                    ty: "i1".to_string(),
                    ty_id,
                    repr: "0".to_string(),
                },
            );
            return Ok(());
        }
        self.locals.insert(
            id.0,
            Operand {
                ty: ty.clone(),
                ty_id,
                repr: self.zero_lit(&ty),
            },
        );
        Ok(())
    }

    fn value(&mut self, v: &MpirValue) -> Result<Operand, String> {
        match v {
            MpirValue::Local(id) => self.locals.get(&id.0).cloned().ok_or_else(|| {
                let ty = self
                    .local_tys
                    .get(&id.0)
                    .map(|t| format!(" (declared ty {})", self.cg.llvm_ty(*t)))
                    .unwrap_or_default();
                format!("undefined local %{} in fn '{}'{}", id.0, self.f.name, ty)
            }),
            MpirValue::Const(c) => {
                let repr = match &c.lit {
                    HirConstLit::StringLit(s) => self.emit_string_literal_runtime(s)?,
                    _ => self.const_lit(c)?,
                };
                Ok(Operand {
                    ty: self.cg.llvm_ty(c.ty),
                    ty_id: c.ty,
                    repr,
                })
            }
        }
    }

    fn emit_string_literal_runtime(&mut self, s: &str) -> Result<String, String> {
        if s.is_empty() {
            let dst = self.tmp();
            writeln!(
                self.out,
                "  {dst} = call ptr @mp_rt_str_from_utf8(ptr null, i64 0)"
            )
            .map_err(|e| e.to_string())?;
            return Ok(dst);
        }

        let bytes = s.as_bytes();
        let arr_ty = format!("[{} x i8]", bytes.len());
        let arr_lit = llvm_bytes_literal(bytes);

        let slot = self.tmp();
        writeln!(self.out, "  {slot} = alloca {arr_ty}").map_err(|e| e.to_string())?;
        writeln!(self.out, "  store {arr_ty} c\"{arr_lit}\", ptr {slot}")
            .map_err(|e| e.to_string())?;

        let data_ptr = self.tmp();
        writeln!(
            self.out,
            "  {data_ptr} = getelementptr inbounds {arr_ty}, ptr {slot}, i64 0, i64 0"
        )
        .map_err(|e| e.to_string())?;

        let dst = self.tmp();
        writeln!(
            self.out,
            "  {dst} = call ptr @mp_rt_str_from_utf8(ptr {data_ptr}, i64 {})",
            bytes.len()
        )
        .map_err(|e| e.to_string())?;
        Ok(dst)
    }

    fn const_lit(&self, c: &HirConst) -> Result<String, String> {
        match &c.lit {
            HirConstLit::IntLit(v) => Ok(v.to_string()),
            HirConstLit::FloatLit(v) => Ok(float_lit(*v)),
            HirConstLit::BoolLit(v) => Ok(if *v { "1" } else { "0" }.to_string()),
            HirConstLit::StringLit(_) => Ok("null".to_string()),
            HirConstLit::Unit => {
                let ty = self.cg.llvm_ty(c.ty);
                if ty == "void" {
                    Ok("0".to_string())
                } else {
                    Ok(self.zero_lit(&ty))
                }
            }
        }
    }

    fn cond_i1(&mut self, v: &MpirValue) -> Result<String, String> {
        let op = self.value(v)?;
        if op.ty == "i1" {
            return Ok(op.repr);
        }
        let tmp = self.tmp();
        if is_int_ty(&op.ty) {
            writeln!(
                self.out,
                "  {tmp} = icmp ne {} {}, {}",
                op.ty,
                op.repr,
                self.zero_lit(&op.ty)
            )
            .map_err(|e| e.to_string())?;
            return Ok(tmp);
        }
        if is_float_ty(&op.ty) {
            writeln!(
                self.out,
                "  {tmp} = fcmp one {} {}, {}",
                op.ty,
                op.repr,
                self.zero_lit(&op.ty)
            )
            .map_err(|e| e.to_string())?;
            return Ok(tmp);
        }
        if op.ty == "ptr" {
            writeln!(self.out, "  {tmp} = icmp ne ptr {}, null", op.repr)
                .map_err(|e| e.to_string())?;
            return Ok(tmp);
        }
        Err(format!("cannot lower condition value of type {}", op.ty))
    }

    fn assign_cast_int(
        &mut self,
        dst: magpie_types::LocalId,
        dst_ty: TypeId,
        src_repr: String,
        src_ty: &str,
    ) -> Result<(), String> {
        let dst_ty_s = self.cg.llvm_ty(dst_ty);
        let dst_name = format!("%l{}", dst.0);
        if dst_ty_s == src_ty {
            writeln!(self.out, "  {dst_name} = add {src_ty} {src_repr}, 0")
                .map_err(|e| e.to_string())?;
        } else if is_int_ty(src_ty) && is_int_ty(&dst_ty_s) {
            let sb = int_bits(src_ty).unwrap_or(64);
            let db = int_bits(&dst_ty_s).unwrap_or(64);
            if sb > db {
                writeln!(
                    self.out,
                    "  {dst_name} = trunc {src_ty} {src_repr} to {dst_ty_s}"
                )
                .map_err(|e| e.to_string())?;
            } else if sb < db {
                writeln!(
                    self.out,
                    "  {dst_name} = zext {src_ty} {src_repr} to {dst_ty_s}"
                )
                .map_err(|e| e.to_string())?;
            } else {
                writeln!(self.out, "  {dst_name} = add {src_ty} {src_repr}, 0")
                    .map_err(|e| e.to_string())?;
            }
        } else {
            writeln!(
                self.out,
                "  {dst_name} = bitcast {src_ty} {src_repr} to {dst_ty_s}"
            )
            .map_err(|e| e.to_string())?;
        }
        self.set_local(dst, dst_ty, dst_ty_s, dst_name);
        Ok(())
    }

    fn cast_i64(&mut self, op: Operand) -> Result<String, String> {
        if op.ty == "i64" {
            return Ok(op.repr);
        }
        let tmp = self.tmp();
        if is_int_ty(&op.ty) {
            let bits = int_bits(&op.ty).unwrap_or(64);
            if bits < 64 {
                writeln!(self.out, "  {tmp} = zext {} {} to i64", op.ty, op.repr)
                    .map_err(|e| e.to_string())?;
            } else if bits > 64 {
                writeln!(self.out, "  {tmp} = trunc {} {} to i64", op.ty, op.repr)
                    .map_err(|e| e.to_string())?;
            } else {
                writeln!(self.out, "  {tmp} = add i64 {}, 0", op.repr)
                    .map_err(|e| e.to_string())?;
            }
            return Ok(tmp);
        }
        if is_float_ty(&op.ty) {
            writeln!(self.out, "  {tmp} = fptoui {} {} to i64", op.ty, op.repr)
                .map_err(|e| e.to_string())?;
            return Ok(tmp);
        }
        Err(format!("cannot cast {} to i64", op.ty))
    }

    fn cast_i32(&mut self, op: Operand) -> Result<String, String> {
        if op.ty == "i32" {
            return Ok(op.repr);
        }
        let tmp = self.tmp();
        if is_int_ty(&op.ty) {
            let bits = int_bits(&op.ty).unwrap_or(64);
            if bits < 32 {
                writeln!(self.out, "  {tmp} = zext {} {} to i32", op.ty, op.repr)
                    .map_err(|e| e.to_string())?;
            } else if bits > 32 {
                writeln!(self.out, "  {tmp} = trunc {} {} to i32", op.ty, op.repr)
                    .map_err(|e| e.to_string())?;
            } else {
                writeln!(self.out, "  {tmp} = add i32 {}, 0", op.repr)
                    .map_err(|e| e.to_string())?;
            }
            return Ok(tmp);
        }
        if is_float_ty(&op.ty) {
            writeln!(self.out, "  {tmp} = fptosi {} {} to i32", op.ty, op.repr)
                .map_err(|e| e.to_string())?;
            return Ok(tmp);
        }
        Err(format!("cannot cast {} to i32", op.ty))
    }

    fn cast_f64(&mut self, op: Operand) -> Result<String, String> {
        if op.ty == "double" {
            return Ok(op.repr);
        }
        let tmp = self.tmp();
        if is_float_ty(&op.ty) {
            let bits = float_bits(&op.ty).unwrap_or(64);
            if bits < 64 {
                writeln!(self.out, "  {tmp} = fpext {} {} to double", op.ty, op.repr)
                    .map_err(|e| e.to_string())?;
            } else {
                writeln!(
                    self.out,
                    "  {tmp} = fptrunc {} {} to double",
                    op.ty, op.repr
                )
                .map_err(|e| e.to_string())?;
            }
            return Ok(tmp);
        }
        if is_int_ty(&op.ty) {
            writeln!(self.out, "  {tmp} = sitofp {} {} to double", op.ty, op.repr)
                .map_err(|e| e.to_string())?;
            return Ok(tmp);
        }
        Err(format!("cannot cast {} to double", op.ty))
    }

    fn ensure_ptr(&mut self, op: Operand) -> Result<String, String> {
        if op.ty == "ptr" {
            return Ok(op.repr);
        }
        let tmp = self.tmp();
        if is_int_ty(&op.ty) {
            writeln!(self.out, "  {tmp} = inttoptr {} {} to ptr", op.ty, op.repr)
                .map_err(|e| e.to_string())?;
            return Ok(tmp);
        }
        // Aggregate types (structs like TResult/TOption) cannot be bitcast to ptr.
        // Return null as a sentinel — callers that handle aggregates should use
        // emit_arc_release_composite / emit_arc_retain_composite instead.
        if op.ty.starts_with('{') || op.ty.starts_with('[') || op.ty.starts_with('<') {
            writeln!(self.out, "  ; skip bitcast of aggregate {} to ptr", op.ty)
                .map_err(|e| e.to_string())?;
            return Ok("null".to_string());
        }
        writeln!(self.out, "  {tmp} = bitcast {} {} to ptr", op.ty, op.repr)
            .map_err(|e| e.to_string())?;
        Ok(tmp)
    }

    /// Extract all `ptr`-typed fields from an aggregate value and emit
    /// `mp_rt_release_strong` for each.  Inactive variant fields are `null`;
    /// the runtime null-guard handles them safely.
    fn emit_arc_release_composite(&mut self, op: &Operand) -> Result<(), String> {
        for (idx, field_ty) in parse_aggregate_fields(&op.ty).iter().enumerate() {
            if field_ty == "ptr" {
                let tmp = self.tmp();
                writeln!(
                    self.out,
                    "  {tmp} = extractvalue {} {}, {idx}",
                    op.ty, op.repr
                )
                .map_err(|e| e.to_string())?;
                writeln!(self.out, "  call void @mp_rt_release_strong(ptr {tmp})")
                    .map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }

    /// Extract all `ptr`-typed fields from an aggregate value and emit
    /// `mp_rt_retain_strong` for each.
    fn emit_arc_retain_composite(&mut self, op: &Operand) -> Result<(), String> {
        for (idx, field_ty) in parse_aggregate_fields(&op.ty).iter().enumerate() {
            if field_ty == "ptr" {
                let tmp = self.tmp();
                writeln!(
                    self.out,
                    "  {tmp} = extractvalue {} {}, {idx}",
                    op.ty, op.repr
                )
                .map_err(|e| e.to_string())?;
                writeln!(self.out, "  call void @mp_rt_retain_strong(ptr {tmp})")
                    .map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }

    fn stack_slot(&mut self, op: &Operand) -> Result<String, String> {
        let slot = self.tmp();
        writeln!(self.out, "  {slot} = alloca {}", op.ty).map_err(|e| e.to_string())?;
        writeln!(self.out, "  store {} {}, ptr {}", op.ty, op.repr, slot)
            .map_err(|e| e.to_string())?;
        Ok(slot)
    }

    fn cast_i64_value(&mut self, v: &MpirValue) -> Result<String, String> {
        let op = self.value(v)?;
        self.cast_i64(op)
    }

    fn cast_i32_value(&mut self, v: &MpirValue) -> Result<String, String> {
        let op = self.value(v)?;
        self.cast_i32(op)
    }

    fn cast_f64_value(&mut self, v: &MpirValue) -> Result<String, String> {
        let op = self.value(v)?;
        self.cast_f64(op)
    }

    fn ensure_ptr_value(&mut self, v: &MpirValue) -> Result<String, String> {
        let op = self.value(v)?;
        self.ensure_ptr(op)
    }

    fn tmp(&mut self) -> String {
        self.tmp_idx = self.tmp_idx.wrapping_add(1);
        format!("%t{}", self.tmp_idx)
    }

    fn label(&mut self, prefix: &str) -> String {
        self.tmp_idx = self.tmp_idx.wrapping_add(1);
        format!("{prefix}_{}", self.tmp_idx)
    }

    fn zero_lit(&self, ty: &str) -> String {
        match ty {
            "half" | "float" | "double" => "0.0".to_string(),
            "ptr" => "null".to_string(),
            "void" => "0".to_string(),
            t if t.starts_with('i') => "0".to_string(),
            t if t.starts_with('{')
                || t.starts_with('[')
                || t.starts_with('<')
                || t.starts_with("%mp_t") =>
            {
                "zeroinitializer".to_string()
            }
            _ => "0".to_string(),
        }
    }
}

fn mangle_fn(sid: &Sid) -> String {
    format!("mp$0$FN${}", sid_suffix(sid))
}

fn mangle_init_types(module_sid: &Sid) -> String {
    format!("mp$0$INIT_TYPES${}", sid_suffix(module_sid))
}

fn callback_hash_symbol(ty: TypeId) -> String {
    format!("mp_cb_hash_t{}", ty.0)
}

fn callback_eq_symbol(ty: TypeId) -> String {
    format!("mp_cb_eq_t{}", ty.0)
}

fn callback_cmp_symbol(ty: TypeId) -> String {
    format!("mp_cb_cmp_t{}", ty.0)
}

fn sid_suffix(sid: &Sid) -> &str {
    sid.0.split_once(':').map(|(_, suf)| suf).unwrap_or(&sid.0)
}

fn sid_hash_64(sid: &Sid) -> u64 {
    // Deterministic FNV-1a 64-bit (matches magpie_gpu::sid_hash_64).
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for b in sid.0.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn align_up_u64(value: u64, align: u64) -> u64 {
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

fn llvm_c_string_literal(s: &str) -> (usize, String) {
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0);
    let encoded = bytes
        .iter()
        .map(|b| format!("\\{:02X}", b))
        .collect::<String>();
    (bytes.len(), encoded)
}

fn llvm_bytes_literal(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|b| format!("\\{:02X}", b))
        .collect::<String>()
}

fn llvm_quote(s: &str) -> String {
    s.chars()
        .flat_map(|ch| match ch {
            '\\' => "\\5C".chars().collect::<Vec<_>>(),
            '"' => "\\22".chars().collect::<Vec<_>>(),
            '\n' => "\\0A".chars().collect::<Vec<_>>(),
            '\r' => "\\0D".chars().collect::<Vec<_>>(),
            '\t' => "\\09".chars().collect::<Vec<_>>(),
            c if c.is_ascii_graphic() || c == ' ' => vec![c],
            c => {
                let mut buf = [0u8; 4];
                let b = c.encode_utf8(&mut buf).as_bytes()[0];
                format!("\\{:02X}", b).chars().collect::<Vec<_>>()
            }
        })
        .collect()
}

fn float_lit(v: f64) -> String {
    if v.is_nan() {
        "0x7ff8000000000000".to_string()
    } else if v.is_infinite() {
        if v.is_sign_negative() {
            "-0x7ff0000000000000".to_string()
        } else {
            "0x7ff0000000000000".to_string()
        }
    } else {
        let mut s = format!("{v}");
        if !s.contains('.') && !s.contains('e') && !s.contains('E') {
            s.push_str(".0");
        }
        s
    }
}

fn normalize_icmp_pred(pred: &str) -> &str {
    match pred {
        "eq" | "ne" | "ugt" | "uge" | "ult" | "ule" | "sgt" | "sge" | "slt" | "sle" => pred,
        "gt" => "sgt",
        "ge" => "sge",
        "lt" => "slt",
        "le" => "sle",
        _ => "eq",
    }
}

/// Parse the field types of an LLVM aggregate type like `{ i1, i64, ptr }`.
/// Returns a vec of the field type strings (e.g. `["i1", "i64", "ptr"]`).
fn parse_aggregate_fields(ty: &str) -> Vec<String> {
    let trimmed = ty.trim();
    let inner = match trimmed.strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
        Some(s) => s,
        None => return vec![],
    };
    if inner.trim().is_empty() {
        return vec![];
    }
    inner.split(',').map(|f| f.trim().to_string()).collect()
}

fn normalize_fcmp_pred(pred: &str) -> &str {
    match pred {
        "false" | "oeq" | "ogt" | "oge" | "olt" | "ole" | "one" | "ord" | "uno" | "ueq" | "ugt"
        | "uge" | "ult" | "ule" | "une" | "true" => pred,
        "eq" => "oeq",
        "ne" => "one",
        "gt" => "ogt",
        "ge" => "oge",
        "lt" => "olt",
        "le" => "ole",
        _ => "oeq",
    }
}

fn int_bits(ty: &str) -> Option<u32> {
    ty.strip_prefix('i')?.parse::<u32>().ok()
}

fn float_bits(ty: &str) -> Option<u32> {
    match ty {
        "half" => Some(16),
        "float" => Some(32),
        "double" => Some(64),
        _ => None,
    }
}

fn is_int_ty(ty: &str) -> bool {
    int_bits(ty).is_some()
}

fn is_float_ty(ty: &str) -> bool {
    matches!(ty, "half" | "float" | "double")
}

use regex::Regex;
use std::path::Path;

#[derive(Clone, Debug, Default)]
pub struct LlvmGenCtx {
    pub locals: HashMap<u32, String>,
    pub llvm_tys: HashMap<u32, String>,
}

impl LlvmGenCtx {
    fn llvm_ty(&self, ty: TypeId) -> String {
        self.llvm_tys
            .get(&ty.0)
            .cloned()
            .unwrap_or_else(|| "ptr".to_string())
    }
}

#[derive(Clone, Debug, Default)]
pub struct ExternDecl {
    pub abi: String,
    pub module: String,
    pub functions: Vec<ExternFnDecl>,
}

#[derive(Clone, Debug, Default)]
pub struct ExternFnDecl {
    pub name: String,
    pub ret_ty: String,
    pub params: Vec<(String, String)>,
    pub lib: String,
}

pub fn lower_ptr_ops(op: &MpirOp, ctx: &mut LlvmGenCtx) -> String {
    match op {
        MpirOp::PtrNull { .. } => "null".to_string(),
        MpirOp::PtrAddr { p } => {
            let ptr = mpir_value_to_llvm(p, ctx);
            format!("ptrtoint ptr {ptr} to i64")
        }
        MpirOp::PtrFromAddr { addr, .. } => {
            let addr = mpir_value_to_llvm(addr, ctx);
            format!("inttoptr i64 {addr} to ptr")
        }
        MpirOp::PtrAdd { p, count } => {
            let p = mpir_value_to_llvm(p, ctx);
            let count = mpir_value_to_llvm(count, ctx);
            format!("getelementptr i8, ptr {p}, i64 {count}")
        }
        MpirOp::PtrLoad { to, p } => {
            let p = mpir_value_to_llvm(p, ctx);
            let to_ty = ctx.llvm_ty(*to);
            format!("load {to_ty}, ptr {p}")
        }
        MpirOp::PtrStore { to, p, v } => {
            let p = mpir_value_to_llvm(p, ctx);
            let v = mpir_value_to_llvm(v, ctx);
            let to_ty = ctx.llvm_ty(*to);
            format!("store {to_ty} {v}, ptr {p}")
        }
        _ => "; unsupported ptr opcode".to_string(),
    }
}

pub fn lower_extern_module(ext: &ExternDecl) -> String {
    let mut out = String::new();
    for decl in &ext.functions {
        let ret = mp_type_to_llvm(&decl.ret_ty);
        let params = decl
            .params
            .iter()
            .map(|(ty, _)| mp_type_to_llvm(ty))
            .collect::<Vec<_>>()
            .join(", ");
        let name = normalize_symbol_name(&decl.name);
        let _ = writeln!(out, "declare {ret} @{name}({params})");
    }
    out
}

pub fn parse_c_header(header_path: &Path) -> Result<Vec<ExternFnDecl>, String> {
    let src = std::fs::read_to_string(header_path)
        .map_err(|e| format!("failed to read {}: {e}", header_path.display()))?;

    let block_comment_re = Regex::new(r"(?s)/\*.*?\*/").map_err(|e| e.to_string())?;
    let line_comment_re = Regex::new(r"(?m)//.*$").map_err(|e| e.to_string())?;
    let preproc_re = Regex::new(r"(?m)^\s*#.*$").map_err(|e| e.to_string())?;
    let decl_re = Regex::new(
        r"(?m)^\s*(?:extern\s+)?([A-Za-z_][A-Za-z0-9_\s\*]*?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^;{}()]*)\)\s*;",
    )
    .map_err(|e| e.to_string())?;
    let param_re =
        Regex::new(r"^\s*(.+?)\s*([A-Za-z_][A-Za-z0-9_]*)\s*$").map_err(|e| e.to_string())?;

    let no_block = block_comment_re.replace_all(&src, "");
    let no_line = line_comment_re.replace_all(&no_block, "");
    let cleaned = preproc_re.replace_all(&no_line, "");

    let lib_name = header_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("c")
        .to_string();

    let mut out = Vec::new();
    for caps in decl_re.captures_iter(cleaned.as_ref()) {
        let raw_ret = caps
            .get(1)
            .map(|m| m.as_str())
            .ok_or_else(|| "missing return type".to_string())?;
        let name = caps
            .get(2)
            .map(|m| m.as_str().to_string())
            .ok_or_else(|| "missing function name".to_string())?;
        let raw_params = caps
            .get(3)
            .map(|m| m.as_str())
            .ok_or_else(|| "missing parameter list".to_string())?;

        let ret_ty = map_c_type(raw_ret)
            .ok_or_else(|| format!("unsupported C return type '{raw_ret}' in function '{name}'"))?;

        let mut params = Vec::new();
        if !raw_params.trim().is_empty() && raw_params.trim() != "void" {
            for (idx, raw_param) in raw_params.split(',').enumerate() {
                let raw_param = raw_param.trim();
                if raw_param.is_empty() {
                    continue;
                }
                if raw_param == "..." {
                    return Err(format!(
                        "variadic extern function '{name}' is not supported yet"
                    ));
                }

                let (raw_ty, raw_name) = if let Some(pm) = param_re.captures(raw_param) {
                    let pty = pm.get(1).map(|m| m.as_str()).unwrap_or(raw_param);
                    let pname = pm
                        .get(2)
                        .map(|m| m.as_str().to_string())
                        .unwrap_or_else(|| format!("arg{idx}"));
                    (pty.to_string(), pname)
                } else {
                    (raw_param.to_string(), format!("arg{idx}"))
                };

                let ty = map_c_type(raw_ty.as_str()).ok_or_else(|| {
                    format!("unsupported C param type '{raw_ty}' in function '{name}'")
                })?;
                params.push((ty, sanitize_mp_ident(&raw_name)));
            }
        }

        out.push(ExternFnDecl {
            name,
            ret_ty,
            params,
            lib: lib_name.clone(),
        });
    }

    Ok(out)
}

pub fn generate_extern_mp(decls: &[ExternFnDecl], lib_name: &str) -> String {
    let mut out = String::new();
    let module = sanitize_mp_ident(lib_name);
    let _ = writeln!(out, "extern \"c\" module {module} {{");

    for decl in decls {
        let fn_name = normalize_symbol_name(&decl.name);
        let params = decl
            .params
            .iter()
            .map(|(ty, name)| format!("%{}: {}", sanitize_mp_ident(name), ty))
            .collect::<Vec<_>>()
            .join(", ");
        let mut attrs = Vec::new();
        if is_pointer_mp_type(&decl.ret_ty) {
            attrs.push("returns=\"borrowed\"");
        }
        if decl.params.iter().any(|(ty, _)| is_pointer_mp_type(ty)) {
            attrs.push("params=\"borrowed\"");
        }
        let attrs = if attrs.is_empty() {
            String::new()
        } else {
            format!(" attrs {{ {} }}", attrs.join(" "))
        };
        let _ = writeln!(out, "  fn @{fn_name}({params}) -> {}{}", decl.ret_ty, attrs);
    }

    let _ = writeln!(out, "}}");
    out
}

fn mpir_value_to_llvm(v: &MpirValue, ctx: &LlvmGenCtx) -> String {
    match v {
        MpirValue::Local(id) => ctx
            .locals
            .get(&id.0)
            .cloned()
            .unwrap_or_else(|| format!("%l{}", id.0)),
        MpirValue::Const(c) => mp_const_to_llvm_lit(c),
    }
}

fn mp_const_to_llvm_lit(c: &HirConst) -> String {
    match &c.lit {
        HirConstLit::IntLit(v) => v.to_string(),
        HirConstLit::FloatLit(v) => float_lit(*v),
        HirConstLit::BoolLit(v) => {
            if *v {
                "1".to_string()
            } else {
                "0".to_string()
            }
        }
        HirConstLit::StringLit(_) => "null".to_string(),
        HirConstLit::Unit => "0".to_string(),
    }
}

fn normalize_symbol_name(name: &str) -> &str {
    name.strip_prefix('@').unwrap_or(name)
}

fn sanitize_mp_ident(name: &str) -> String {
    let mut out = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        return "arg".to_string();
    }
    if out.chars().next().is_some_and(|ch| ch.is_ascii_digit()) {
        out.insert(0, '_');
    }
    out
}

fn is_pointer_mp_type(ty: &str) -> bool {
    ty.starts_with("rawptr<") || ty == "ptr"
}

fn map_c_type(raw_ty: &str) -> Option<String> {
    let mut ty = raw_ty.trim().to_string();
    for qualifier in [
        "const", "volatile", "register", "static", "extern", "inline",
    ] {
        ty = Regex::new(&format!(r"\b{qualifier}\b"))
            .ok()?
            .replace_all(&ty, "")
            .to_string();
    }

    ty = ty.replace(" *", "*").replace("* ", "*");
    ty = ty.split_whitespace().collect::<Vec<_>>().join(" ");

    if ty.ends_with('*') {
        let base = ty.trim_end_matches('*').trim();
        if base == "void" {
            return Some("rawptr<u8>".to_string());
        }
        let inner = match base {
            "char" => "i8",
            "int" => "i32",
            "long" | "long long" => "i64",
            "size_t" => "u64",
            _ => "u8",
        };
        return Some(format!("rawptr<{inner}>"));
    }

    let mapped = match ty.as_str() {
        "void" => "unit",
        "char" => "i8",
        "int" => "i32",
        "long" | "long long" => "i64",
        "size_t" => "u64",
        "unsigned char" => "u8",
        "unsigned int" => "u32",
        "unsigned long" | "unsigned long long" => "u64",
        _ => return None,
    };
    Some(mapped.to_string())
}

fn mp_type_to_llvm(mp_ty: &str) -> String {
    match mp_ty.trim() {
        "unit" | "void" => "void".to_string(),
        "i1" | "bool" => "i1".to_string(),
        "i8" | "u8" => "i8".to_string(),
        "i16" | "u16" => "i16".to_string(),
        "i32" | "u32" => "i32".to_string(),
        "i64" | "u64" => "i64".to_string(),
        "i128" | "u128" => "i128".to_string(),
        "f16" => "half".to_string(),
        "f32" => "float".to_string(),
        "f64" => "double".to_string(),
        t if t.starts_with("rawptr<") => "ptr".to_string(),
        _ => "ptr".to_string(),
    }
}

pub fn lower_ptr_null() -> String {
    "  null".to_string()
}

pub fn lower_ptr_addr(reg: &str) -> String {
    format!("  %addr = ptrtoint ptr {} to i64", reg)
}

pub fn lower_ptr_from_addr(reg: &str) -> String {
    format!("  %ptr = inttoptr i64 {} to ptr", reg)
}

pub fn lower_ptr_add(ptr_reg: &str, count_reg: &str, elem_ty: &str) -> String {
    format!(
        "  %ptr = getelementptr {}, ptr {}, i64 {}",
        elem_ty, ptr_reg, count_reg
    )
}

pub fn lower_ptr_load(ptr_reg: &str, ty: &str) -> String {
    format!("  %v = load {}, ptr {}", ty, ptr_reg)
}

pub fn lower_ptr_store(ptr_reg: &str, val_reg: &str, ty: &str) -> String {
    format!("  store {} {}, ptr {}", ty, val_reg, ptr_reg)
}

pub fn lower_extern_fn_declare(name: &str, params: &[(String, String)], ret_ty: &str) -> String {
    let llvm_ret = mp_type_to_llvm(ret_ty);
    let llvm_params = params
        .iter()
        .map(|(ty, _)| mp_type_to_llvm(ty))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "declare {} @{}({})",
        llvm_ret,
        normalize_symbol_name(name),
        llvm_params
    )
}

pub fn parse_c_header_basic(source: &str) -> Vec<ExternFnDecl> {
    let block_comment_re = match Regex::new(r"(?s)/\*.*?\*/") {
        Ok(re) => re,
        Err(_) => return Vec::new(),
    };
    let line_comment_re = match Regex::new(r"(?m)//.*$") {
        Ok(re) => re,
        Err(_) => return Vec::new(),
    };
    let preproc_re = match Regex::new(r"(?m)^\s*#.*$") {
        Ok(re) => re,
        Err(_) => return Vec::new(),
    };
    let decl_re = match Regex::new(
        r"(?m)^\s*(?:extern\s+)?([A-Za-z_][A-Za-z0-9_\s\*]*?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^;{}()]*)\)\s*;",
    ) {
        Ok(re) => re,
        Err(_) => return Vec::new(),
    };
    let param_re = match Regex::new(r"^\s*(.+?)\s*([A-Za-z_][A-Za-z0-9_]*)\s*$") {
        Ok(re) => re,
        Err(_) => return Vec::new(),
    };

    let no_block = block_comment_re.replace_all(source, "");
    let no_line = line_comment_re.replace_all(&no_block, "");
    let cleaned = preproc_re.replace_all(&no_line, "");

    let mut out = Vec::new();
    'decls: for caps in decl_re.captures_iter(cleaned.as_ref()) {
        let Some(raw_ret) = caps.get(1).map(|m| m.as_str()) else {
            continue;
        };
        let Some(name) = caps.get(2).map(|m| m.as_str().to_string()) else {
            continue;
        };
        let Some(raw_params) = caps.get(3).map(|m| m.as_str()) else {
            continue;
        };

        let Some(ret_ty) = map_c_type(raw_ret) else {
            continue;
        };

        let mut params = Vec::new();
        if !raw_params.trim().is_empty() && raw_params.trim() != "void" {
            for (idx, raw_param) in raw_params.split(',').enumerate() {
                let raw_param = raw_param.trim();
                if raw_param.is_empty() {
                    continue;
                }
                if raw_param == "..." {
                    continue 'decls;
                }

                let (raw_ty, raw_name) = if let Some(pm) = param_re.captures(raw_param) {
                    let pty = pm.get(1).map(|m| m.as_str()).unwrap_or(raw_param);
                    let pname = pm
                        .get(2)
                        .map(|m| m.as_str().to_string())
                        .unwrap_or_else(|| format!("arg{idx}"));
                    (pty.to_string(), pname)
                } else {
                    (raw_param.to_string(), format!("arg{idx}"))
                };

                let Some(ty) = map_c_type(raw_ty.as_str()) else {
                    continue 'decls;
                };
                params.push((ty, sanitize_mp_ident(&raw_name)));
            }
        }

        out.push(ExternFnDecl {
            name,
            ret_ty,
            params,
            lib: "c".to_string(),
        });
    }

    out
}

pub fn generate_extern_mp_module(decls: &[ExternFnDecl], lib_name: &str) -> String {
    let mut out = String::new();
    let module = sanitize_mp_ident(lib_name);
    let _ = writeln!(out, "extern \"c\" module {module} {{");

    for decl in decls {
        let fn_name = normalize_symbol_name(&decl.name);
        let params = decl
            .params
            .iter()
            .map(|(ty, name)| format!("%{}: {}", sanitize_mp_ident(name), ty))
            .collect::<Vec<_>>()
            .join(", ");
        let mut attrs = Vec::new();
        if is_pointer_mp_type(&decl.ret_ty) {
            attrs.push("returns=\"borrowed\"");
        }
        if decl.params.iter().any(|(ty, _)| is_pointer_mp_type(ty)) {
            attrs.push("params=\"borrowed\"");
        }
        let attrs = if attrs.is_empty() {
            String::new()
        } else {
            format!(" attrs {{ {} }}", attrs.join(" "))
        };
        let _ = writeln!(out, "  fn @{fn_name}({params}) -> {}{}", decl.ret_ty, attrs);
    }

    let _ = writeln!(out, "}}");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use magpie_mpir::{MpirLocalDecl, MpirTypeTable};
    use magpie_types::HandleKind;

    #[test]
    fn test_codegen_hello_world() {
        let type_ctx = TypeCtx::new();
        let i32_ty = type_ctx.lookup_by_prim(PrimType::I32);

        let module = MpirModule {
            sid: Sid("M:HELLOWORLD0".to_string()),
            path: "hello.mp".to_string(),
            type_table: MpirTypeTable { types: vec![] },
            functions: vec![MpirFn {
                sid: Sid("F:HELLOWORLD0".to_string()),
                name: "main".to_string(),
                params: vec![],
                ret_ty: i32_ty,
                blocks: vec![MpirBlock {
                    id: magpie_types::BlockId(0),
                    instrs: vec![MpirInstr {
                        dst: magpie_types::LocalId(0),
                        ty: i32_ty,
                        op: MpirOp::Const(HirConst {
                            ty: i32_ty,
                            lit: HirConstLit::IntLit(42),
                        }),
                    }],
                    void_ops: vec![],
                    terminator: MpirTerminator::Ret(Some(MpirValue::Local(magpie_types::LocalId(
                        0,
                    )))),
                }],
                locals: vec![MpirLocalDecl {
                    id: magpie_types::LocalId(0),
                    ty: i32_ty,
                    name: "retv".to_string(),
                }],
                is_async: false,
            }],
            globals: vec![],
        };

        let llvm_ir = codegen_module(&module, &type_ctx).expect("codegen should succeed");
        assert!(
            llvm_ir.contains("define"),
            "expected function definitions in IR"
        );
        assert!(llvm_ir.contains("ret"), "expected return instruction in IR");
        assert!(llvm_ir.contains("call void @mp_gpu_register_all_kernels()"));
    }

    #[test]
    fn test_codegen_emits_generics_mode_marker() {
        let type_ctx = TypeCtx::new();
        let i32_ty = type_ctx.lookup_by_prim(PrimType::I32);
        let module = MpirModule {
            sid: Sid("M:GENMODE000".to_string()),
            path: "genmode.mp".to_string(),
            type_table: MpirTypeTable { types: vec![] },
            functions: vec![MpirFn {
                sid: Sid("F:GENMODE000".to_string()),
                name: "main".to_string(),
                params: vec![],
                ret_ty: i32_ty,
                blocks: vec![MpirBlock {
                    id: magpie_types::BlockId(0),
                    instrs: vec![MpirInstr {
                        dst: magpie_types::LocalId(0),
                        ty: i32_ty,
                        op: MpirOp::Const(HirConst {
                            ty: i32_ty,
                            lit: HirConstLit::IntLit(0),
                        }),
                    }],
                    void_ops: vec![],
                    terminator: MpirTerminator::Ret(Some(MpirValue::Local(magpie_types::LocalId(
                        0,
                    )))),
                }],
                locals: vec![MpirLocalDecl {
                    id: magpie_types::LocalId(0),
                    ty: i32_ty,
                    name: "retv".to_string(),
                }],
                is_async: false,
            }],
            globals: vec![],
        };

        let default_ir = codegen_module(&module, &type_ctx).expect("default codegen should pass");
        assert!(default_ir.contains("\"mp$0$ABI$generics_mode\""));
        assert!(default_ir.contains("constant i8 0"));

        let shared_ir = codegen_module_with_options(
            &module,
            &type_ctx,
            CodegenOptions {
                shared_generics: true,
            },
        )
        .expect("shared-generics codegen should pass");
        assert!(shared_ir.contains("\"mp$0$ABI$generics_mode\""));
        assert!(shared_ir.contains("constant i8 1"));
    }

    #[test]
    fn test_codegen_without_main_does_not_emit_c_main() {
        let type_ctx = TypeCtx::new();
        let i32_ty = type_ctx.lookup_by_prim(PrimType::I32);
        let module = MpirModule {
            sid: Sid("M:NOMAIN0000".to_string()),
            path: "no_main.mp".to_string(),
            type_table: MpirTypeTable { types: vec![] },
            functions: vec![MpirFn {
                sid: Sid("F:HELPER0000".to_string()),
                name: "helper".to_string(),
                params: vec![],
                ret_ty: i32_ty,
                blocks: vec![MpirBlock {
                    id: magpie_types::BlockId(0),
                    instrs: vec![MpirInstr {
                        dst: magpie_types::LocalId(0),
                        ty: i32_ty,
                        op: MpirOp::Const(HirConst {
                            ty: i32_ty,
                            lit: HirConstLit::IntLit(7),
                        }),
                    }],
                    void_ops: vec![],
                    terminator: MpirTerminator::Ret(Some(MpirValue::Local(magpie_types::LocalId(
                        0,
                    )))),
                }],
                locals: vec![MpirLocalDecl {
                    id: magpie_types::LocalId(0),
                    ty: i32_ty,
                    name: "retv".to_string(),
                }],
                is_async: false,
            }],
            globals: vec![],
        };

        let llvm_ir = codegen_module(&module, &type_ctx).expect("codegen should succeed");
        assert!(!llvm_ir.contains("define i32 @main("));
    }

    #[test]
    fn test_mangling() {
        let sid = Sid("F:ABCDEFGHIJ".to_string());
        assert_eq!(mangle_fn(&sid), "mp$0$FN$ABCDEFGHIJ");
    }

    #[test]
    fn test_codegen_gpu_ops_lowering() {
        let mut type_ctx = TypeCtx::new();
        let i32_ty = type_ctx.lookup_by_prim(PrimType::I32);
        let i64_ty = type_ctx.lookup_by_prim(PrimType::I64);
        let unit_ty = type_ctx.lookup_by_prim(PrimType::Unit);
        let raw_ptr_ty = type_ctx.intern(TypeKind::RawPtr {
            to: fixed_type_ids::U8,
        });
        let kernel_sid = Sid("F:KERNELGPU0".to_string());

        let kernel_fn = MpirFn {
            sid: kernel_sid.clone(),
            name: "kernel".to_string(),
            params: vec![(magpie_types::LocalId(0), raw_ptr_ty)],
            ret_ty: unit_ty,
            blocks: vec![],
            locals: vec![MpirLocalDecl {
                id: magpie_types::LocalId(0),
                ty: raw_ptr_ty,
                name: "buf".to_string(),
            }],
            is_async: false,
        };

        let main_fn = MpirFn {
            sid: Sid("F:GPUHOST00".to_string()),
            name: "main".to_string(),
            params: vec![],
            ret_ty: i32_ty,
            blocks: vec![MpirBlock {
                id: magpie_types::BlockId(0),
                instrs: vec![
                    MpirInstr {
                        dst: magpie_types::LocalId(0),
                        ty: raw_ptr_ty,
                        op: MpirOp::PtrNull { to: raw_ptr_ty },
                    },
                    MpirInstr {
                        dst: magpie_types::LocalId(1),
                        ty: i64_ty,
                        op: MpirOp::Const(HirConst {
                            ty: i64_ty,
                            lit: HirConstLit::IntLit(0),
                        }),
                    },
                    MpirInstr {
                        dst: magpie_types::LocalId(2),
                        ty: i64_ty,
                        op: MpirOp::GpuBufferLen {
                            buf: MpirValue::Local(magpie_types::LocalId(0)),
                        },
                    },
                    MpirInstr {
                        dst: magpie_types::LocalId(3),
                        ty: i32_ty,
                        op: MpirOp::GpuBufferLoad {
                            buf: MpirValue::Local(magpie_types::LocalId(0)),
                            idx: MpirValue::Local(magpie_types::LocalId(1)),
                        },
                    },
                    MpirInstr {
                        dst: magpie_types::LocalId(4),
                        ty: i32_ty,
                        op: MpirOp::GpuLaunch {
                            device: MpirValue::Local(magpie_types::LocalId(0)),
                            kernel: kernel_sid.clone(),
                            groups: MpirValue::Const(HirConst {
                                ty: i32_ty,
                                lit: HirConstLit::IntLit(1),
                            }),
                            threads: MpirValue::Const(HirConst {
                                ty: i32_ty,
                                lit: HirConstLit::IntLit(1),
                            }),
                            args: vec![MpirValue::Local(magpie_types::LocalId(0))],
                        },
                    },
                    MpirInstr {
                        dst: magpie_types::LocalId(5),
                        ty: i32_ty,
                        op: MpirOp::GpuLaunchAsync {
                            device: MpirValue::Local(magpie_types::LocalId(0)),
                            kernel: kernel_sid,
                            groups: MpirValue::Const(HirConst {
                                ty: i32_ty,
                                lit: HirConstLit::IntLit(1),
                            }),
                            threads: MpirValue::Const(HirConst {
                                ty: i32_ty,
                                lit: HirConstLit::IntLit(1),
                            }),
                            args: vec![MpirValue::Local(magpie_types::LocalId(0))],
                        },
                    },
                ],
                void_ops: vec![
                    MpirOpVoid::GpuBufferStore {
                        buf: MpirValue::Local(magpie_types::LocalId(0)),
                        idx: MpirValue::Local(magpie_types::LocalId(1)),
                        val: MpirValue::Const(HirConst {
                            ty: i32_ty,
                            lit: HirConstLit::IntLit(7),
                        }),
                    },
                    MpirOpVoid::GpuBarrier,
                ],
                terminator: MpirTerminator::Ret(Some(MpirValue::Local(magpie_types::LocalId(4)))),
            }],
            locals: vec![
                MpirLocalDecl {
                    id: magpie_types::LocalId(0),
                    ty: raw_ptr_ty,
                    name: "dev".to_string(),
                },
                MpirLocalDecl {
                    id: magpie_types::LocalId(1),
                    ty: i64_ty,
                    name: "idx".to_string(),
                },
                MpirLocalDecl {
                    id: magpie_types::LocalId(2),
                    ty: i64_ty,
                    name: "len".to_string(),
                },
                MpirLocalDecl {
                    id: magpie_types::LocalId(3),
                    ty: i32_ty,
                    name: "loaded".to_string(),
                },
                MpirLocalDecl {
                    id: magpie_types::LocalId(4),
                    ty: i32_ty,
                    name: "launch".to_string(),
                },
                MpirLocalDecl {
                    id: magpie_types::LocalId(5),
                    ty: i32_ty,
                    name: "launch_async".to_string(),
                },
            ],
            is_async: false,
        };

        let module = MpirModule {
            sid: Sid("M:GPULOWER00".to_string()),
            path: "gpu.mp".to_string(),
            type_table: MpirTypeTable { types: vec![] },
            functions: vec![kernel_fn, main_fn],
            globals: vec![],
        };

        let llvm_ir = codegen_module(&module, &type_ctx).expect("gpu lowering should succeed");
        assert!(llvm_ir.contains("@mp_rt_gpu_launch_sync"));
        assert!(llvm_ir.contains("@mp_rt_gpu_launch_async"));
        assert!(llvm_ir.contains("@mp_rt_gpu_buffer_len"));
        assert!(llvm_ir.contains("@mp_rt_gpu_buffer_read"));
        assert!(llvm_ir.contains("@mp_rt_gpu_buffer_write"));
    }

    #[test]
    fn test_codegen_builtin_result_unit_uses_storage_type() {
        let mut type_ctx = TypeCtx::new();
        let result_ty = type_ctx.intern(TypeKind::BuiltinResult {
            ok: fixed_type_ids::UNIT,
            err: fixed_type_ids::STR,
        });

        let module = MpirModule {
            sid: Sid("M:RESULTUNIT0".to_string()),
            path: "result_unit.mp".to_string(),
            type_table: MpirTypeTable { types: vec![] },
            functions: vec![MpirFn {
                sid: Sid("F:RESULTUNIT0".to_string()),
                name: "main".to_string(),
                params: vec![],
                ret_ty: result_ty,
                blocks: vec![],
                locals: vec![],
                is_async: false,
            }],
            globals: vec![],
        };

        let llvm_ir = codegen_module(&module, &type_ctx).expect("codegen should succeed");
        assert!(llvm_ir.contains("define { i1, i8, ptr }"));
    }

    #[test]
    fn test_codegen_checked_add_option_does_not_emit_invalid_aggregate_add() {
        let mut type_ctx = TypeCtx::new();
        let i64_ty = type_ctx.lookup_by_prim(PrimType::I64);
        let opt_i64_ty = type_ctx.intern(TypeKind::BuiltinOption { inner: i64_ty });

        let module = MpirModule {
            sid: Sid("M:CHECKEDOPT0".to_string()),
            path: "checked_opt.mp".to_string(),
            type_table: MpirTypeTable { types: vec![] },
            functions: vec![MpirFn {
                sid: Sid("F:CHECKEDOPT0".to_string()),
                name: "main".to_string(),
                params: vec![],
                ret_ty: opt_i64_ty,
                blocks: vec![MpirBlock {
                    id: magpie_types::BlockId(0),
                    instrs: vec![MpirInstr {
                        dst: magpie_types::LocalId(0),
                        ty: opt_i64_ty,
                        op: MpirOp::IAddChecked {
                            lhs: MpirValue::Const(HirConst {
                                ty: i64_ty,
                                lit: HirConstLit::IntLit(1),
                            }),
                            rhs: MpirValue::Const(HirConst {
                                ty: i64_ty,
                                lit: HirConstLit::IntLit(2),
                            }),
                        },
                    }],
                    void_ops: vec![],
                    terminator: MpirTerminator::Ret(Some(MpirValue::Local(magpie_types::LocalId(
                        0,
                    )))),
                }],
                locals: vec![MpirLocalDecl {
                    id: magpie_types::LocalId(0),
                    ty: opt_i64_ty,
                    name: "sum".to_string(),
                }],
                is_async: false,
            }],
            globals: vec![],
        };

        let llvm_ir = codegen_module(&module, &type_ctx).expect("codegen should succeed");
        assert!(llvm_ir.contains("@llvm.sadd.with.overflow.i64"));
        assert!(!llvm_ir.contains("add { i64, i1 }"));
    }

    #[test]
    fn test_codegen_const_str_materializes_runtime_string() {
        let type_ctx = TypeCtx::new();
        let str_ty = fixed_type_ids::STR;

        let module = MpirModule {
            sid: Sid("M:STRCONST00".to_string()),
            path: "str_const.mp".to_string(),
            type_table: MpirTypeTable { types: vec![] },
            functions: vec![MpirFn {
                sid: Sid("F:STRCONST00".to_string()),
                name: "make".to_string(),
                params: vec![],
                ret_ty: str_ty,
                blocks: vec![MpirBlock {
                    id: magpie_types::BlockId(0),
                    instrs: vec![MpirInstr {
                        dst: magpie_types::LocalId(0),
                        ty: str_ty,
                        op: MpirOp::Const(HirConst {
                            ty: str_ty,
                            lit: HirConstLit::StringLit("hello".to_string()),
                        }),
                    }],
                    void_ops: vec![],
                    terminator: MpirTerminator::Ret(Some(MpirValue::Local(magpie_types::LocalId(
                        0,
                    )))),
                }],
                locals: vec![MpirLocalDecl {
                    id: magpie_types::LocalId(0),
                    ty: str_ty,
                    name: "s".to_string(),
                }],
                is_async: false,
            }],
            globals: vec![],
        };

        let llvm_ir = codegen_module(&module, &type_ctx).expect("codegen should succeed");
        assert!(llvm_ir.contains("declare ptr @mp_rt_str_from_utf8(ptr, i64)"));
        assert!(llvm_ir.contains("alloca [5 x i8]"));
        assert!(llvm_ir.contains("call ptr @mp_rt_str_from_utf8"));
    }

    #[test]
    fn test_codegen_user_struct_new_getfield_setfield() {
        let mut type_ctx = TypeCtx::new();
        let i32_ty = type_ctx.lookup_by_prim(PrimType::I32);
        let sid = Sid("T:POINT00000".to_string());
        type_ctx.register_type_fqn(sid.clone(), "pkg.main.Point");
        type_ctx.register_value_struct_fields(
            sid.clone(),
            vec![("x".to_string(), i32_ty), ("y".to_string(), i32_ty)],
        );
        let point_ty = type_ctx.intern(TypeKind::HeapHandle {
            hk: HandleKind::Unique,
            base: HeapBase::UserType {
                type_sid: sid,
                targs: vec![],
            },
        });

        let module = MpirModule {
            sid: Sid("M:STRUCTOPS0".to_string()),
            path: "struct_ops.mp".to_string(),
            type_table: MpirTypeTable { types: vec![] },
            functions: vec![MpirFn {
                sid: Sid("F:STRUCTOPS0".to_string()),
                name: "main".to_string(),
                params: vec![],
                ret_ty: i32_ty,
                blocks: vec![MpirBlock {
                    id: magpie_types::BlockId(0),
                    instrs: vec![
                        MpirInstr {
                            dst: magpie_types::LocalId(0),
                            ty: point_ty,
                            op: MpirOp::New {
                                ty: point_ty,
                                fields: vec![
                                    (
                                        "x".to_string(),
                                        MpirValue::Const(HirConst {
                                            ty: i32_ty,
                                            lit: HirConstLit::IntLit(1),
                                        }),
                                    ),
                                    (
                                        "y".to_string(),
                                        MpirValue::Const(HirConst {
                                            ty: i32_ty,
                                            lit: HirConstLit::IntLit(2),
                                        }),
                                    ),
                                ],
                            },
                        },
                        MpirInstr {
                            dst: magpie_types::LocalId(1),
                            ty: i32_ty,
                            op: MpirOp::GetField {
                                obj: MpirValue::Local(magpie_types::LocalId(0)),
                                field: "x".to_string(),
                            },
                        },
                    ],
                    void_ops: vec![MpirOpVoid::SetField {
                        obj: MpirValue::Local(magpie_types::LocalId(0)),
                        field: "y".to_string(),
                        value: MpirValue::Const(HirConst {
                            ty: i32_ty,
                            lit: HirConstLit::IntLit(7),
                        }),
                    }],
                    terminator: MpirTerminator::Ret(Some(MpirValue::Local(magpie_types::LocalId(
                        1,
                    )))),
                }],
                locals: vec![
                    MpirLocalDecl {
                        id: magpie_types::LocalId(0),
                        ty: point_ty,
                        name: "p".to_string(),
                    },
                    MpirLocalDecl {
                        id: magpie_types::LocalId(1),
                        ty: i32_ty,
                        name: "x".to_string(),
                    },
                ],
                is_async: false,
            }],
            globals: vec![],
        };

        let llvm_ir = codegen_module(&module, &type_ctx).expect("codegen should succeed");
        assert!(llvm_ir.contains("@mp_rt_alloc(i32"));
        assert!(llvm_ir.contains("getelementptr i8, ptr %l0, i64 32"));
        assert!(llvm_ir.contains("load i32"));
    }

    #[test]
    fn test_codegen_emits_runtime_type_registry_for_user_heap_types() {
        let mut type_ctx = TypeCtx::new();
        let i32_ty = type_ctx.lookup_by_prim(PrimType::I32);
        let sid = Sid("T:REGTYPE000".to_string());
        type_ctx.register_type_fqn(sid.clone(), "pkg.main.RegType");
        type_ctx.register_value_struct_fields(sid.clone(), vec![("x".to_string(), i32_ty)]);
        let _heap_ty = type_ctx.intern(TypeKind::HeapHandle {
            hk: HandleKind::Unique,
            base: HeapBase::UserType {
                type_sid: sid,
                targs: vec![],
            },
        });

        let module = MpirModule {
            sid: Sid("M:TYPEREG00".to_string()),
            path: "typereg.mp".to_string(),
            type_table: MpirTypeTable { types: vec![] },
            functions: vec![MpirFn {
                sid: Sid("F:TYPEREG00".to_string()),
                name: "main".to_string(),
                params: vec![],
                ret_ty: i32_ty,
                blocks: vec![MpirBlock {
                    id: magpie_types::BlockId(0),
                    instrs: vec![MpirInstr {
                        dst: magpie_types::LocalId(0),
                        ty: i32_ty,
                        op: MpirOp::Const(HirConst {
                            ty: i32_ty,
                            lit: HirConstLit::IntLit(0),
                        }),
                    }],
                    void_ops: vec![],
                    terminator: MpirTerminator::Ret(Some(MpirValue::Local(magpie_types::LocalId(
                        0,
                    )))),
                }],
                locals: vec![MpirLocalDecl {
                    id: magpie_types::LocalId(0),
                    ty: i32_ty,
                    name: "ret".to_string(),
                }],
                is_async: false,
            }],
            globals: vec![],
        };

        let llvm_ir = codegen_module(&module, &type_ctx).expect("codegen should succeed");
        assert!(llvm_ir.contains("%MpRtTypeInfo = type"));
        assert!(llvm_ir.contains("mp_type_registry_"));
        assert!(llvm_ir.contains("call void @mp_rt_register_types("));
    }

    #[test]
    fn test_codegen_indirect_and_enum_user_ops() {
        let mut type_ctx = TypeCtx::new();
        let i32_ty = type_ctx.lookup_by_prim(PrimType::I32);
        let callable_ty = type_ctx.intern(TypeKind::HeapHandle {
            hk: HandleKind::Unique,
            base: HeapBase::Callable {
                sig_sid: Sid("S:CALLSIG000".to_string()),
            },
        });

        let enum_sid = Sid("T:ENUMUSER00".to_string());
        type_ctx.register_type_fqn(enum_sid.clone(), "pkg.main.E");
        type_ctx.register_value_enum_variants(
            enum_sid.clone(),
            vec![
                ("A".to_string(), vec![("v".to_string(), i32_ty)]),
                ("B".to_string(), vec![("e".to_string(), i32_ty)]),
            ],
        );
        let user_enum_ty = type_ctx.intern(TypeKind::ValueStruct {
            sid: enum_sid.clone(),
        });
        let heap_enum_ty = type_ctx.intern(TypeKind::HeapHandle {
            hk: HandleKind::Unique,
            base: HeapBase::UserType {
                type_sid: enum_sid,
                targs: vec![],
            },
        });

        let module = MpirModule {
            sid: Sid("M:INDENUM00".to_string()),
            path: "indirect_enum.mp".to_string(),
            type_table: MpirTypeTable { types: vec![] },
            functions: vec![MpirFn {
                sid: Sid("F:INDENUM00".to_string()),
                name: "main".to_string(),
                params: vec![],
                ret_ty: i32_ty,
                blocks: vec![MpirBlock {
                    id: magpie_types::BlockId(0),
                    instrs: vec![
                        MpirInstr {
                            dst: magpie_types::LocalId(0),
                            ty: callable_ty,
                            op: MpirOp::PtrNull { to: callable_ty },
                        },
                        MpirInstr {
                            dst: magpie_types::LocalId(1),
                            ty: i32_ty,
                            op: MpirOp::CallIndirect {
                                callee: MpirValue::Local(magpie_types::LocalId(0)),
                                args: vec![],
                            },
                        },
                        MpirInstr {
                            dst: magpie_types::LocalId(2),
                            ty: user_enum_ty,
                            op: MpirOp::EnumNew {
                                variant: "A".to_string(),
                                args: vec![(
                                    "v".to_string(),
                                    MpirValue::Const(HirConst {
                                        ty: i32_ty,
                                        lit: HirConstLit::IntLit(9),
                                    }),
                                )],
                            },
                        },
                        MpirInstr {
                            dst: magpie_types::LocalId(3),
                            ty: i32_ty,
                            op: MpirOp::EnumTag {
                                v: MpirValue::Local(magpie_types::LocalId(2)),
                            },
                        },
                        MpirInstr {
                            dst: magpie_types::LocalId(4),
                            ty: i32_ty,
                            op: MpirOp::EnumPayload {
                                variant: "A".to_string(),
                                v: MpirValue::Local(magpie_types::LocalId(2)),
                            },
                        },
                        MpirInstr {
                            dst: magpie_types::LocalId(5),
                            ty: i32_ty,
                            op: MpirOp::EnumIs {
                                variant: "A".to_string(),
                                v: MpirValue::Local(magpie_types::LocalId(2)),
                            },
                        },
                        MpirInstr {
                            dst: magpie_types::LocalId(6),
                            ty: heap_enum_ty,
                            op: MpirOp::EnumNew {
                                variant: "B".to_string(),
                                args: vec![(
                                    "e".to_string(),
                                    MpirValue::Const(HirConst {
                                        ty: i32_ty,
                                        lit: HirConstLit::IntLit(3),
                                    }),
                                )],
                            },
                        },
                    ],
                    void_ops: vec![MpirOpVoid::CallVoidIndirect {
                        callee: MpirValue::Local(magpie_types::LocalId(0)),
                        args: vec![],
                    }],
                    terminator: MpirTerminator::Ret(Some(MpirValue::Local(magpie_types::LocalId(
                        3,
                    )))),
                }],
                locals: vec![
                    MpirLocalDecl {
                        id: magpie_types::LocalId(0),
                        ty: callable_ty,
                        name: "cb".to_string(),
                    },
                    MpirLocalDecl {
                        id: magpie_types::LocalId(1),
                        ty: i32_ty,
                        name: "ind".to_string(),
                    },
                    MpirLocalDecl {
                        id: magpie_types::LocalId(2),
                        ty: user_enum_ty,
                        name: "e".to_string(),
                    },
                    MpirLocalDecl {
                        id: magpie_types::LocalId(3),
                        ty: i32_ty,
                        name: "tag".to_string(),
                    },
                    MpirLocalDecl {
                        id: magpie_types::LocalId(4),
                        ty: i32_ty,
                        name: "payload".to_string(),
                    },
                    MpirLocalDecl {
                        id: magpie_types::LocalId(5),
                        ty: i32_ty,
                        name: "is_a".to_string(),
                    },
                    MpirLocalDecl {
                        id: magpie_types::LocalId(6),
                        ty: heap_enum_ty,
                        name: "he".to_string(),
                    },
                ],
                is_async: false,
            }],
            globals: vec![],
        };

        let llvm_ir = codegen_module(&module, &type_ctx).expect("codegen should succeed");
        assert!(llvm_ir.contains("@mp_rt_callable_fn_ptr"));
        assert!(llvm_ir.contains("@mp_rt_callable_data_ptr"));
        assert!(llvm_ir.contains("@mp_rt_callable_capture_size"));
        assert!(llvm_ir.contains("icmp eq i32"));
        assert!(llvm_ir.contains("@mp_rt_alloc(i32"));
    }

    #[test]
    fn test_codegen_map_get_ref_emits_missing_key_panic() {
        let mut type_ctx = TypeCtx::new();
        let i32_ty = type_ctx.lookup_by_prim(PrimType::I32);
        let map_ty = type_ctx.intern(TypeKind::HeapHandle {
            hk: HandleKind::Unique,
            base: HeapBase::BuiltinMap {
                key: i32_ty,
                val: i32_ty,
            },
        });
        let ptr_i32 = type_ctx.intern(TypeKind::RawPtr { to: i32_ty });

        let module = MpirModule {
            sid: Sid("M:MAPGETRF0".to_string()),
            path: "map_get_ref.mp".to_string(),
            type_table: MpirTypeTable { types: vec![] },
            functions: vec![MpirFn {
                sid: Sid("F:MAPGETRF0".to_string()),
                name: "main".to_string(),
                params: vec![],
                ret_ty: i32_ty,
                blocks: vec![MpirBlock {
                    id: magpie_types::BlockId(0),
                    instrs: vec![
                        MpirInstr {
                            dst: magpie_types::LocalId(0),
                            ty: map_ty,
                            op: MpirOp::MapNew {
                                key_ty: i32_ty,
                                val_ty: i32_ty,
                            },
                        },
                        MpirInstr {
                            dst: magpie_types::LocalId(1),
                            ty: i32_ty,
                            op: MpirOp::Const(HirConst {
                                ty: i32_ty,
                                lit: HirConstLit::IntLit(7),
                            }),
                        },
                        MpirInstr {
                            dst: magpie_types::LocalId(2),
                            ty: ptr_i32,
                            op: MpirOp::MapGetRef {
                                map: MpirValue::Local(magpie_types::LocalId(0)),
                                key: MpirValue::Local(magpie_types::LocalId(1)),
                            },
                        },
                        MpirInstr {
                            dst: magpie_types::LocalId(3),
                            ty: i32_ty,
                            op: MpirOp::Const(HirConst {
                                ty: i32_ty,
                                lit: HirConstLit::IntLit(0),
                            }),
                        },
                    ],
                    void_ops: vec![],
                    terminator: MpirTerminator::Ret(Some(MpirValue::Local(magpie_types::LocalId(
                        3,
                    )))),
                }],
                locals: vec![
                    MpirLocalDecl {
                        id: magpie_types::LocalId(0),
                        ty: map_ty,
                        name: "m".to_string(),
                    },
                    MpirLocalDecl {
                        id: magpie_types::LocalId(1),
                        ty: i32_ty,
                        name: "k".to_string(),
                    },
                    MpirLocalDecl {
                        id: magpie_types::LocalId(2),
                        ty: ptr_i32,
                        name: "vref".to_string(),
                    },
                    MpirLocalDecl {
                        id: magpie_types::LocalId(3),
                        ty: i32_ty,
                        name: "ret".to_string(),
                    },
                ],
                is_async: false,
            }],
            globals: vec![],
        };

        let llvm_ir = codegen_module(&module, &type_ctx).expect("codegen should succeed");
        assert!(llvm_ir.contains("@mp_rt_map_get("));
        assert!(llvm_ir.contains("icmp eq ptr"));
        assert!(llvm_ir.contains("call void @mp_rt_panic(ptr null)"));
    }

    #[test]
    fn test_codegen_collection_callbacks_use_generated_wrappers() {
        let mut type_ctx = TypeCtx::new();
        let i32_ty = type_ctx.lookup_by_prim(PrimType::I32);
        let unit_ty = type_ctx.lookup_by_prim(PrimType::Unit);
        let arr_ty = type_ctx.intern(TypeKind::HeapHandle {
            hk: HandleKind::Unique,
            base: HeapBase::BuiltinArray { elem: i32_ty },
        });
        let map_ty = type_ctx.intern(TypeKind::HeapHandle {
            hk: HandleKind::Unique,
            base: HeapBase::BuiltinMap {
                key: i32_ty,
                val: i32_ty,
            },
        });

        let module = MpirModule {
            sid: Sid("M:CBWRAP0".to_string()),
            path: "cb_wrap.mp".to_string(),
            type_table: MpirTypeTable { types: vec![] },
            functions: vec![MpirFn {
                sid: Sid("F:CBWRAP0".to_string()),
                name: "main".to_string(),
                params: vec![],
                ret_ty: i32_ty,
                blocks: vec![MpirBlock {
                    id: magpie_types::BlockId(0),
                    instrs: vec![
                        MpirInstr {
                            dst: magpie_types::LocalId(0),
                            ty: i32_ty,
                            op: MpirOp::Const(HirConst {
                                ty: i32_ty,
                                lit: HirConstLit::IntLit(0),
                            }),
                        },
                        MpirInstr {
                            dst: magpie_types::LocalId(1),
                            ty: arr_ty,
                            op: MpirOp::ArrNew {
                                elem_ty: i32_ty,
                                cap: MpirValue::Local(magpie_types::LocalId(0)),
                            },
                        },
                        MpirInstr {
                            dst: magpie_types::LocalId(2),
                            ty: i32_ty,
                            op: MpirOp::Const(HirConst {
                                ty: i32_ty,
                                lit: HirConstLit::IntLit(7),
                            }),
                        },
                        MpirInstr {
                            dst: magpie_types::LocalId(3),
                            ty: i32_ty,
                            op: MpirOp::ArrContains {
                                arr: MpirValue::Local(magpie_types::LocalId(1)),
                                val: MpirValue::Local(magpie_types::LocalId(2)),
                            },
                        },
                        MpirInstr {
                            dst: magpie_types::LocalId(4),
                            ty: unit_ty,
                            op: MpirOp::ArrSort {
                                arr: MpirValue::Local(magpie_types::LocalId(1)),
                            },
                        },
                        MpirInstr {
                            dst: magpie_types::LocalId(5),
                            ty: map_ty,
                            op: MpirOp::MapNew {
                                key_ty: i32_ty,
                                val_ty: i32_ty,
                            },
                        },
                    ],
                    void_ops: vec![],
                    terminator: MpirTerminator::Ret(Some(MpirValue::Local(magpie_types::LocalId(
                        3,
                    )))),
                }],
                locals: vec![
                    MpirLocalDecl {
                        id: magpie_types::LocalId(0),
                        ty: i32_ty,
                        name: "cap".to_string(),
                    },
                    MpirLocalDecl {
                        id: magpie_types::LocalId(1),
                        ty: arr_ty,
                        name: "arr".to_string(),
                    },
                    MpirLocalDecl {
                        id: magpie_types::LocalId(2),
                        ty: i32_ty,
                        name: "val".to_string(),
                    },
                    MpirLocalDecl {
                        id: magpie_types::LocalId(3),
                        ty: i32_ty,
                        name: "contains".to_string(),
                    },
                    MpirLocalDecl {
                        id: magpie_types::LocalId(4),
                        ty: unit_ty,
                        name: "sorted".to_string(),
                    },
                    MpirLocalDecl {
                        id: magpie_types::LocalId(5),
                        ty: map_ty,
                        name: "map".to_string(),
                    },
                ],
                is_async: false,
            }],
            globals: vec![],
        };

        let llvm_ir = codegen_module(&module, &type_ctx).expect("codegen should succeed");
        assert!(llvm_ir.contains("define weak_odr i64 @mp_cb_hash_t4(ptr %x_bytes)"));
        assert!(llvm_ir.contains("define weak_odr i32 @mp_cb_eq_t4(ptr %a_bytes, ptr %b_bytes)"));
        assert!(llvm_ir.contains("define weak_odr i32 @mp_cb_cmp_t4(ptr %a_bytes, ptr %b_bytes)"));
        assert!(llvm_ir.contains("call i32 @mp_rt_arr_contains(ptr %l1, ptr"));
        assert!(llvm_ir.contains("ptr @mp_cb_eq_t4)"));
        assert!(llvm_ir.contains("call void @mp_rt_arr_sort(ptr %l1, ptr @mp_cb_cmp_t4)"));
        assert!(llvm_ir.contains("@mp_rt_map_new(i32 4, i32 4, i64 4, i64 4, i64 0, ptr @mp_cb_hash_t4, ptr @mp_cb_eq_t4)"));
    }

    #[test]
    fn test_codegen_collection_callbacks_for_str_use_string_semantics() {
        let mut type_ctx = TypeCtx::new();
        let str_ty = fixed_type_ids::STR;
        let i32_ty = type_ctx.lookup_by_prim(PrimType::I32);
        let unit_ty = type_ctx.lookup_by_prim(PrimType::Unit);
        let bool_ty = type_ctx.lookup_by_prim(PrimType::Bool);
        let arr_ty = type_ctx.intern(TypeKind::HeapHandle {
            hk: HandleKind::Unique,
            base: HeapBase::BuiltinArray { elem: str_ty },
        });
        let map_ty = type_ctx.intern(TypeKind::HeapHandle {
            hk: HandleKind::Unique,
            base: HeapBase::BuiltinMap {
                key: str_ty,
                val: bool_ty,
            },
        });

        let module = MpirModule {
            sid: Sid("M:CBSTR00".to_string()),
            path: "cb_str.mp".to_string(),
            type_table: MpirTypeTable { types: vec![] },
            functions: vec![MpirFn {
                sid: Sid("F:CBSTR00".to_string()),
                name: "main".to_string(),
                params: vec![],
                ret_ty: bool_ty,
                blocks: vec![MpirBlock {
                    id: magpie_types::BlockId(0),
                    instrs: vec![
                        MpirInstr {
                            dst: magpie_types::LocalId(0),
                            ty: arr_ty,
                            op: MpirOp::ArrNew {
                                elem_ty: str_ty,
                                cap: MpirValue::Const(HirConst {
                                    ty: i32_ty,
                                    lit: HirConstLit::IntLit(0),
                                }),
                            },
                        },
                        MpirInstr {
                            dst: magpie_types::LocalId(1),
                            ty: str_ty,
                            op: MpirOp::PtrNull { to: str_ty },
                        },
                        MpirInstr {
                            dst: magpie_types::LocalId(2),
                            ty: bool_ty,
                            op: MpirOp::ArrContains {
                                arr: MpirValue::Local(magpie_types::LocalId(0)),
                                val: MpirValue::Local(magpie_types::LocalId(1)),
                            },
                        },
                        MpirInstr {
                            dst: magpie_types::LocalId(3),
                            ty: unit_ty,
                            op: MpirOp::ArrSort {
                                arr: MpirValue::Local(magpie_types::LocalId(0)),
                            },
                        },
                        MpirInstr {
                            dst: magpie_types::LocalId(4),
                            ty: map_ty,
                            op: MpirOp::MapNew {
                                key_ty: str_ty,
                                val_ty: bool_ty,
                            },
                        },
                    ],
                    void_ops: vec![],
                    terminator: MpirTerminator::Ret(Some(MpirValue::Local(magpie_types::LocalId(
                        2,
                    )))),
                }],
                locals: vec![
                    MpirLocalDecl {
                        id: magpie_types::LocalId(0),
                        ty: arr_ty,
                        name: "arr".to_string(),
                    },
                    MpirLocalDecl {
                        id: magpie_types::LocalId(1),
                        ty: str_ty,
                        name: "s".to_string(),
                    },
                    MpirLocalDecl {
                        id: magpie_types::LocalId(2),
                        ty: bool_ty,
                        name: "contains".to_string(),
                    },
                    MpirLocalDecl {
                        id: magpie_types::LocalId(3),
                        ty: unit_ty,
                        name: "sorted".to_string(),
                    },
                    MpirLocalDecl {
                        id: magpie_types::LocalId(4),
                        ty: map_ty,
                        name: "map".to_string(),
                    },
                ],
                is_async: false,
            }],
            globals: vec![],
        };

        let llvm_ir = codegen_module(&module, &type_ctx).expect("codegen should succeed");
        assert!(llvm_ir.contains("call i64 @mp_std_hash_str(ptr %x)"));
        assert!(llvm_ir.contains("call i32 @mp_rt_str_eq(ptr %a, ptr %b)"));
        assert!(llvm_ir.contains("call i32 @mp_rt_str_cmp(ptr %a, ptr %b)"));
    }
}
