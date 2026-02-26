//! magpie_codegen_llvm

use magpie_mpir::{
    HirConst, HirConstLit, MpirBlock, MpirFn, MpirInstr, MpirModule, MpirOp, MpirOpVoid,
    MpirTerminator, MpirValue,
};
use magpie_types::{HeapBase, PrimType, Sid, TypeCtx, TypeId, TypeKind};
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;

pub fn codegen_module(mpir: &MpirModule, type_ctx: &TypeCtx) -> Result<String, String> {
    let mut cg = LlvmTextCodegen::new(mpir, type_ctx);
    cg.codegen_module()
}

struct LlvmTextCodegen<'a> {
    mpir: &'a MpirModule,
    type_ctx: &'a TypeCtx,
    type_map: BTreeMap<u32, TypeKind>,
}

impl<'a> LlvmTextCodegen<'a> {
    fn new(mpir: &'a MpirModule, type_ctx: &'a TypeCtx) -> Self {
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
        }
    }

    fn codegen_module(&mut self) -> Result<String, String> {
        let mut out = String::new();
        let module_init = mangle_init_types(&self.mpir.sid);
        let main_fn = self.mpir.functions.iter().find(|f| f.name == "main");

        writeln!(out, "; ModuleID = '{}'", llvm_quote(&self.mpir.path)).map_err(|e| e.to_string())?;
        writeln!(out, "source_filename = \"{}\"", llvm_quote(&self.mpir.path))
            .map_err(|e| e.to_string())?;
        writeln!(out).map_err(|e| e.to_string())?;

        let value_struct_ids = self
            .type_map
            .iter()
            .filter_map(|(id, kind)| matches!(kind, TypeKind::ValueStruct { .. }).then_some(*id))
            .collect::<Vec<_>>();
        for id in value_struct_ids {
            writeln!(out, "%mp_t{} = type {{}}", id).map_err(|e| e.to_string())?;
        }
        if !out.ends_with('\n') {
            writeln!(out).map_err(|e| e.to_string())?;
        }

        self.emit_declarations(&mut out)?;
        writeln!(out).map_err(|e| e.to_string())?;

        writeln!(out, "define internal void @{}() {{", module_init).map_err(|e| e.to_string())?;
        writeln!(out, "entry:").map_err(|e| e.to_string())?;
        writeln!(out, "  ret void").map_err(|e| e.to_string())?;
        writeln!(out, "}}").map_err(|e| e.to_string())?;
        writeln!(out).map_err(|e| e.to_string())?;

        for f in &self.mpir.functions {
            let body = self.codegen_fn(f)?;
            out.push_str(&body);
            out.push('\n');
        }

        self.emit_c_main(&mut out, module_init.as_str(), main_fn)?;
        Ok(out)
    }

    fn emit_declarations(&self, out: &mut String) -> Result<(), String> {
        let decls = [
            "declare void @mp_rt_init()",
            "declare void @mp_rt_retain_strong(ptr)",
            "declare void @mp_rt_release_strong(ptr)",
            "declare void @mp_rt_retain_weak(ptr)",
            "declare void @mp_rt_release_weak(ptr)",
            "declare ptr @mp_rt_weak_upgrade(ptr)",
            "declare noreturn void @mp_rt_panic(ptr)",
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
            "declare ptr @mp_rt_str_slice(ptr, i64, i64)",
            "declare ptr @mp_rt_str_bytes(ptr, ptr)",
            "declare ptr @mp_rt_strbuilder_new()",
            "declare void @mp_rt_strbuilder_append_str(ptr, ptr)",
            "declare void @mp_rt_strbuilder_append_i64(ptr, i64)",
            "declare void @mp_rt_strbuilder_append_i32(ptr, i32)",
            "declare void @mp_rt_strbuilder_append_f64(ptr, double)",
            "declare void @mp_rt_strbuilder_append_bool(ptr, i32)",
            "declare ptr @mp_rt_strbuilder_build(ptr)",
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
                format!("{{ {}, i1 }}", self.llvm_ty(*inner))
            }
            Some(TypeKind::BuiltinResult { ok, err }) => {
                format!("{{ i1, {}, {} }}", self.llvm_ty(*ok), self.llvm_ty(*err))
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
            Some(TypeKind::BuiltinResult { ok, err }) => 1 + self.size_of_ty(*ok) + self.size_of_ty(*err),
            Some(TypeKind::Arr { n, elem }) => (*n as u64).saturating_mul(self.size_of_ty(*elem)),
            Some(TypeKind::Vec { n, elem }) => (*n as u64).saturating_mul(self.size_of_ty(*elem)),
            Some(TypeKind::Tuple { elems }) => elems.iter().map(|e| self.size_of_ty(*e)).sum(),
            Some(TypeKind::ValueStruct { .. }) => 0,
            None => 8,
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
        writeln!(out, "define {} @{}({}) {{", ret_ty, mangle_fn(&f.sid), params)
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
                let lit = self.const_lit(c)?;
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
            MpirOp::IAdd { lhs, rhs }
            | MpirOp::IAddWrap { lhs, rhs } => self.emit_bin(i.dst, i.ty, "add", lhs, rhs)?,
            MpirOp::ISub { lhs, rhs }
            | MpirOp::ISubWrap { lhs, rhs } => self.emit_bin(i.dst, i.ty, "sub", lhs, rhs)?,
            MpirOp::IMul { lhs, rhs }
            | MpirOp::IMulWrap { lhs, rhs } => self.emit_bin(i.dst, i.ty, "mul", lhs, rhs)?,
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
                writeln!(self.out, "  {dst} = ptrtoint {} {} to {}", p.ty, p.repr, dst_ty)
                    .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::PtrFromAddr { .. } => {
                let addr = match &i.op {
                    MpirOp::PtrFromAddr { addr, .. } => self.value(addr)?,
                    _ => unreachable!(),
                };
                writeln!(self.out, "  {dst} = inttoptr {} {} to {}", addr.ty, addr.repr, dst_ty)
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
                writeln!(self.out, "  {dst} = load {load_ty}, ptr {p}").map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::PtrStore { to, p, v } => {
                let p = self.ensure_ptr_value(p)?;
                let v = self.value(v)?;
                writeln!(self.out, "  store {} {}, ptr {}", self.cg.llvm_ty(*to), v.repr, p)
                    .map_err(|e| e.to_string())?;
                self.set_default(i.dst, i.ty)?;
            }
            MpirOp::Call {
                callee_sid, args, ..
            } => {
                let args = self.call_args(args)?;
                let name = mangle_fn(callee_sid);
                if dst_ty == "void" {
                    writeln!(self.out, "  call void @{}({})", name, args).map_err(|e| e.to_string())?;
                    self.set_default(i.dst, i.ty)?;
                } else {
                    writeln!(self.out, "  {dst} = call {dst_ty} @{}({})", name, args)
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
            MpirOp::ArcRetain { v } => {
                let p = self.ensure_ptr_value(v)?;
                writeln!(self.out, "  call void @mp_rt_retain_strong(ptr {p})")
                    .map_err(|e| e.to_string())?;
                self.assign_or_copy_value(i.dst, i.ty, v)?;
            }
            MpirOp::ArcRelease { v } => {
                let p = self.ensure_ptr_value(v)?;
                writeln!(self.out, "  call void @mp_rt_release_strong(ptr {p})")
                    .map_err(|e| e.to_string())?;
                self.set_default(i.dst, i.ty)?;
            }
            MpirOp::ArcRetainWeak { v } => {
                let p = self.ensure_ptr_value(v)?;
                writeln!(self.out, "  call void @mp_rt_retain_weak(ptr {p})").map_err(|e| e.to_string())?;
                self.assign_or_copy_value(i.dst, i.ty, v)?;
            }
            MpirOp::ArcReleaseWeak { v } => {
                let p = self.ensure_ptr_value(v)?;
                writeln!(self.out, "  call void @mp_rt_release_weak(ptr {p})").map_err(|e| e.to_string())?;
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
                writeln!(self.out, "  {p} = call ptr @mp_rt_arr_get(ptr {arr}, i64 {idx})")
                    .map_err(|e| e.to_string())?;
                writeln!(self.out, "  {dst} = load {dst_ty}, ptr {p}").map_err(|e| e.to_string())?;
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
                writeln!(
                    self.out,
                    "  {stat} = call i32 @mp_rt_arr_contains(ptr {arr}, ptr {slot}, i64 {}, ptr null)",
                    self.cg.size_of_ty(val.ty_id)
                )
                .map_err(|e| e.to_string())?;
                self.assign_cast_int(i.dst, i.ty, stat, "i32")?;
            }
            MpirOp::ArrSort { arr } => {
                let arr = self.ensure_ptr_value(arr)?;
                writeln!(self.out, "  call void @mp_rt_arr_sort(ptr {arr}, ptr null)")
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
                writeln!(self.out, "  {dst} = call ptr @mp_rt_arr_filter(ptr {arr}, ptr {func})")
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
                writeln!(self.out, "  {dst} = load {dst_ty}, ptr {slot}").map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::ArrForeach { arr, func } => {
                let arr = self.ensure_ptr_value(arr)?;
                let func = self.ensure_ptr_value(func)?;
                writeln!(self.out, "  call void @mp_rt_arr_foreach(ptr {arr}, ptr {func})")
                    .map_err(|e| e.to_string())?;
                self.set_default(i.dst, i.ty)?;
            }
            MpirOp::MapNew { key_ty, val_ty } => {
                let key_size = self.cg.size_of_ty(*key_ty);
                let val_size = self.cg.size_of_ty(*val_ty);
                writeln!(
                    self.out,
                    "  {dst} = call ptr @mp_rt_map_new(i32 {}, i32 {}, i64 {}, i64 {}, i64 0, ptr null, ptr null)",
                    key_ty.0, val_ty.0, key_size, val_size
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
                writeln!(self.out, "  {dst} = load {dst_ty}, ptr {p}").map_err(|e| e.to_string())?;
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
                writeln!(self.out, "  {dst} = call ptr @mp_rt_map_keys(ptr {map})").map_err(|e| e.to_string())?;
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
                writeln!(self.out, "  {dst} = call ptr @mp_rt_str_concat(ptr {a}, ptr {b})")
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
                writeln!(self.out, "  {eq} = call i32 @mp_rt_str_eq(ptr {a}, ptr {b})")
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
                writeln!(self.out, "  {dst} = call ptr @mp_rt_str_bytes(ptr {s}, ptr {len_slot})")
                    .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::StrBuilderNew => {
                writeln!(self.out, "  {dst} = call ptr @mp_rt_strbuilder_new()")
                    .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::StrBuilderAppendStr { b, s } => {
                let b = self.ensure_ptr_value(b)?;
                let s = self.ensure_ptr_value(s)?;
                writeln!(self.out, "  call void @mp_rt_strbuilder_append_str(ptr {b}, ptr {s})")
                    .map_err(|e| e.to_string())?;
                self.set_default(i.dst, i.ty)?;
            }
            MpirOp::StrBuilderAppendI64 { b, v } => {
                let b = self.ensure_ptr_value(b)?;
                let v = self.cast_i64_value(v)?;
                writeln!(self.out, "  call void @mp_rt_strbuilder_append_i64(ptr {b}, i64 {v})")
                    .map_err(|e| e.to_string())?;
                self.set_default(i.dst, i.ty)?;
            }
            MpirOp::StrBuilderAppendI32 { b, v } => {
                let b = self.ensure_ptr_value(b)?;
                let v = self.cast_i32_value(v)?;
                writeln!(self.out, "  call void @mp_rt_strbuilder_append_i32(ptr {b}, i32 {v})")
                    .map_err(|e| e.to_string())?;
                self.set_default(i.dst, i.ty)?;
            }
            MpirOp::StrBuilderAppendF64 { b, v } => {
                let b = self.ensure_ptr_value(b)?;
                let v = self.cast_f64_value(v)?;
                writeln!(self.out, "  call void @mp_rt_strbuilder_append_f64(ptr {b}, double {v})")
                    .map_err(|e| e.to_string())?;
                self.set_default(i.dst, i.ty)?;
            }
            MpirOp::StrBuilderAppendBool { b, v } => {
                let b = self.ensure_ptr_value(b)?;
                let v = self.cast_i32_value(v)?;
                writeln!(self.out, "  call void @mp_rt_strbuilder_append_bool(ptr {b}, i32 {v})")
                    .map_err(|e| e.to_string())?;
                self.set_default(i.dst, i.ty)?;
            }
            MpirOp::StrBuilderBuild { b } => {
                let b = self.ensure_ptr_value(b)?;
                writeln!(self.out, "  {dst} = call ptr @mp_rt_strbuilder_build(ptr {b})")
                    .map_err(|e| e.to_string())?;
                self.set_local(i.dst, i.ty, dst_ty, dst);
            }
            MpirOp::Panic { msg } => {
                let msg = self.ensure_ptr_value(msg)?;
                writeln!(self.out, "  call noreturn void @mp_rt_panic(ptr {msg})")
                    .map_err(|e| e.to_string())?;
                self.set_default(i.dst, i.ty)?;
            }
            other => {
                return Err(format!(
                    "unsupported MPIR op in llvm text lowering for fn '{}': {:?}",
                    self.f.name, other
                ));
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
            MpirOpVoid::ArrSet { arr, idx, val } => self.emit_arr_set(arr, idx, val)?,
            MpirOpVoid::ArrPush { arr, val } => self.emit_arr_push(arr, val)?,
            MpirOpVoid::ArrSort { arr } => {
                let arr = self.ensure_ptr_value(arr)?;
                writeln!(self.out, "  call void @mp_rt_arr_sort(ptr {arr}, ptr null)")
                    .map_err(|e| e.to_string())?;
            }
            MpirOpVoid::ArrForeach { arr, func } => {
                let arr = self.ensure_ptr_value(arr)?;
                let func = self.ensure_ptr_value(func)?;
                writeln!(self.out, "  call void @mp_rt_arr_foreach(ptr {arr}, ptr {func})")
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
                writeln!(self.out, "  call void @mp_rt_strbuilder_append_str(ptr {b}, ptr {s})")
                    .map_err(|e| e.to_string())?;
            }
            MpirOpVoid::StrBuilderAppendI64 { b, v } => {
                let b = self.ensure_ptr_value(b)?;
                let v = self.cast_i64_value(v)?;
                writeln!(self.out, "  call void @mp_rt_strbuilder_append_i64(ptr {b}, i64 {v})")
                    .map_err(|e| e.to_string())?;
            }
            MpirOpVoid::StrBuilderAppendI32 { b, v } => {
                let b = self.ensure_ptr_value(b)?;
                let v = self.cast_i32_value(v)?;
                writeln!(self.out, "  call void @mp_rt_strbuilder_append_i32(ptr {b}, i32 {v})")
                    .map_err(|e| e.to_string())?;
            }
            MpirOpVoid::StrBuilderAppendF64 { b, v } => {
                let b = self.ensure_ptr_value(b)?;
                let v = self.cast_f64_value(v)?;
                writeln!(self.out, "  call void @mp_rt_strbuilder_append_f64(ptr {b}, double {v})")
                    .map_err(|e| e.to_string())?;
            }
            MpirOpVoid::StrBuilderAppendBool { b, v } => {
                let b = self.ensure_ptr_value(b)?;
                let v = self.cast_i32_value(v)?;
                writeln!(self.out, "  call void @mp_rt_strbuilder_append_bool(ptr {b}, i32 {v})")
                    .map_err(|e| e.to_string())?;
            }
            MpirOpVoid::PtrStore { to, p, v } => {
                let p = self.ensure_ptr_value(p)?;
                let v = self.value(v)?;
                writeln!(self.out, "  store {} {}, ptr {}", self.cg.llvm_ty(*to), v.repr, p)
                    .map_err(|e| e.to_string())?;
            }
            MpirOpVoid::Panic { msg } => {
                let msg = self.ensure_ptr_value(msg)?;
                writeln!(self.out, "  call noreturn void @mp_rt_panic(ptr {msg})")
                    .map_err(|e| e.to_string())?;
            }
            MpirOpVoid::ArcRetain { v } => {
                let p = self.ensure_ptr_value(v)?;
                writeln!(self.out, "  call void @mp_rt_retain_strong(ptr {p})")
                    .map_err(|e| e.to_string())?;
            }
            MpirOpVoid::ArcRelease { v } => {
                let p = self.ensure_ptr_value(v)?;
                writeln!(self.out, "  call void @mp_rt_release_strong(ptr {p})")
                    .map_err(|e| e.to_string())?;
            }
            MpirOpVoid::ArcRetainWeak { v } => {
                let p = self.ensure_ptr_value(v)?;
                writeln!(self.out, "  call void @mp_rt_retain_weak(ptr {p})").map_err(|e| e.to_string())?;
            }
            MpirOpVoid::ArcReleaseWeak { v } => {
                let p = self.ensure_ptr_value(v)?;
                writeln!(self.out, "  call void @mp_rt_release_weak(ptr {p})").map_err(|e| e.to_string())?;
            }
            other => {
                return Err(format!(
                    "unsupported MPIR void-op in llvm text lowering for fn '{}': {:?}",
                    self.f.name, other
                ));
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
        writeln!(self.out, "  {dst} = {op} {dst_ty} {}, {}", lhs.repr, rhs.repr)
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
            writeln!(self.out, "  {dst_name} = add {pair_ty} {call_tmp}, zeroinitializer")
                .map_err(|e| e.to_string())?;
            self.set_local(dst, dst_ty, expect, dst_name);
            return Ok(());
        }
        let val_tmp = self.tmp();
        writeln!(
            self.out,
            "  {val_tmp} = extractvalue {pair_ty} {call_tmp}, 0"
        )
        .map_err(|e| e.to_string())?;
        let ov_tmp = self.tmp();
        writeln!(self.out, "  {ov_tmp} = extractvalue {pair_ty} {call_tmp}, 1")
            .map_err(|e| e.to_string())?;
        let none_tag = self.tmp();
        writeln!(self.out, "  {none_tag} = xor i1 {ov_tmp}, true").map_err(|e| e.to_string())?;
        let agg0 = self.tmp();
        writeln!(self.out, "  {agg0} = insertvalue {expect} undef, {int_ty} {val_tmp}, 0")
            .map_err(|e| e.to_string())?;
        writeln!(self.out, "  {dst_name} = insertvalue {expect} {agg0}, i1 {none_tag}, 1")
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
            writeln!(self.out, "  {dst} = bitcast ptr {} to ptr", src.repr).map_err(|e| e.to_string())?;
        } else if src_ty == "ptr" && is_int_ty(&dst_ty) {
            writeln!(self.out, "  {dst} = ptrtoint ptr {} to {}", src.repr, dst_ty)
                .map_err(|e| e.to_string())?;
        } else if is_int_ty(&src_ty) && dst_ty == "ptr" {
            writeln!(self.out, "  {dst} = inttoptr {} {} to ptr", src_ty, src.repr)
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
                writeln!(self.out, "  {dst} = add {} {}, 0", dst_ty, src.repr).map_err(|e| e.to_string())?;
            } else {
                writeln!(self.out, "  {dst} = {op} {} {} to {}", src_ty, src.repr, dst_ty)
                    .map_err(|e| e.to_string())?;
            }
        } else if is_float_ty(&src_ty) && is_float_ty(&dst_ty) {
            let src_bits = float_bits(&src_ty).unwrap_or(64);
            let dst_bits = float_bits(&dst_ty).unwrap_or(64);
            let op = if src_bits < dst_bits { "fpext" } else { "fptrunc" };
            writeln!(self.out, "  {dst} = {op} {} {} to {}", src_ty, src.repr, dst_ty)
                .map_err(|e| e.to_string())?;
        } else if is_int_ty(&src_ty) && is_float_ty(&dst_ty) {
            let op = if self.cg.is_signed_int(src.ty_id) {
                "sitofp"
            } else {
                "uitofp"
            };
            writeln!(self.out, "  {dst} = {op} {} {} to {}", src_ty, src.repr, dst_ty)
                .map_err(|e| e.to_string())?;
        } else if is_float_ty(&src_ty) && is_int_ty(&dst_ty) {
            let op = if self.cg.is_signed_int(dst_ty_id) {
                "fptosi"
            } else {
                "fptoui"
            };
            writeln!(self.out, "  {dst} = {op} {} {} to {}", src_ty, src.repr, dst_ty)
                .map_err(|e| e.to_string())?;
        } else {
            writeln!(self.out, "  {dst} = bitcast {} {} to {}", src_ty, src.repr, dst_ty)
                .map_err(|e| e.to_string())?;
        }
        self.set_local(dst_id, dst_ty_id, dst_ty, dst);
        Ok(())
    }

    fn emit_arr_set(&mut self, arr: &MpirValue, idx: &MpirValue, val: &MpirValue) -> Result<(), String> {
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
                let inner_ty = self.cg.llvm_ty(*inner);
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
                writeln!(self.out, "  {loaded} = load {inner_ty}, ptr {out}").map_err(|e| e.to_string())?;
                let ok = self.tmp();
                writeln!(self.out, "  {ok} = icmp eq i32 {stat}, 1").map_err(|e| e.to_string())?;
                let agg0 = self.tmp();
                writeln!(
                    self.out,
                    "  {agg0} = insertvalue {dst_ty} undef, {inner_ty} {loaded}, 0"
                )
                .map_err(|e| e.to_string())?;
                let dst = format!("%l{}", dst_id.0);
                writeln!(self.out, "  {dst} = insertvalue {dst_ty} {agg0}, i1 {ok}, 1")
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
                writeln!(self.out, "  {dst} = load {dst_ty}, ptr {out}").map_err(|e| e.to_string())?;
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
            writeln!(self.out, "  {dst_name} = inttoptr {} {} to ptr", src.ty, src.repr)
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

    fn value(&self, v: &MpirValue) -> Result<Operand, String> {
        match v {
            MpirValue::Local(id) => self.locals.get(&id.0).cloned().ok_or_else(|| {
                let ty = self
                    .local_tys
                    .get(&id.0)
                    .map(|t| format!(" (declared ty {})", self.cg.llvm_ty(*t)))
                    .unwrap_or_default();
                format!(
                    "undefined local %{} in fn '{}'{}",
                    id.0, self.f.name, ty
                )
            }),
            MpirValue::Const(c) => Ok(Operand {
                ty: self.cg.llvm_ty(c.ty),
                ty_id: c.ty,
                repr: self.const_lit(c)?,
            }),
        }
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
            writeln!(self.out, "  {tmp} = icmp ne ptr {}, null", op.repr).map_err(|e| e.to_string())?;
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
            writeln!(self.out, "  {dst_name} = add {src_ty} {src_repr}, 0").map_err(|e| e.to_string())?;
        } else if is_int_ty(src_ty) && is_int_ty(&dst_ty_s) {
            let sb = int_bits(src_ty).unwrap_or(64);
            let db = int_bits(&dst_ty_s).unwrap_or(64);
            if sb > db {
                writeln!(self.out, "  {dst_name} = trunc {src_ty} {src_repr} to {dst_ty_s}")
                    .map_err(|e| e.to_string())?;
            } else if sb < db {
                writeln!(self.out, "  {dst_name} = zext {src_ty} {src_repr} to {dst_ty_s}")
                    .map_err(|e| e.to_string())?;
            } else {
                writeln!(self.out, "  {dst_name} = add {src_ty} {src_repr}, 0").map_err(|e| e.to_string())?;
            }
        } else {
            writeln!(self.out, "  {dst_name} = bitcast {src_ty} {src_repr} to {dst_ty_s}")
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
                writeln!(self.out, "  {tmp} = add i64 {}, 0", op.repr).map_err(|e| e.to_string())?;
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
                writeln!(self.out, "  {tmp} = add i32 {}, 0", op.repr).map_err(|e| e.to_string())?;
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
                writeln!(self.out, "  {tmp} = fptrunc {} {} to double", op.ty, op.repr)
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
        writeln!(self.out, "  {tmp} = bitcast {} {} to ptr", op.ty, op.repr)
            .map_err(|e| e.to_string())?;
        Ok(tmp)
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

    fn zero_lit(&self, ty: &str) -> String {
        match ty {
            "half" | "float" | "double" => "0.0".to_string(),
            "ptr" => "null".to_string(),
            "void" => "0".to_string(),
            t if t.starts_with('i') => "0".to_string(),
            t if t.starts_with('{') || t.starts_with('[') || t.starts_with('<') || t.starts_with("%mp_t") => {
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

fn sid_suffix(sid: &Sid) -> &str {
    sid.0.split_once(':').map(|(_, suf)| suf).unwrap_or(&sid.0)
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

fn normalize_fcmp_pred(pred: &str) -> &str {
    match pred {
        "false" | "oeq" | "ogt" | "oge" | "olt" | "ole" | "one" | "ord" | "uno" | "ueq"
        | "ugt" | "uge" | "ult" | "ule" | "une" | "true" => pred,
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
