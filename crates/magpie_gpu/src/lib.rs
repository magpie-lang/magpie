//! GPU kernel validation/layout helpers for Magpie GPU v0.1 (§31).

use magpie_diag::{Diagnostic, DiagnosticBag, Severity};
use magpie_mpir::{MpirFn, MpirOp, MpirOpVoid, MpirTerminator};
use magpie_types::{fixed_type_ids, HeapBase, PrimType, Sid, TypeCtx, TypeId, TypeKind};
use std::collections::HashSet;

pub const GPU_BACKEND_SPV: u32 = 1;

#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelParamKind {
    Buffer = 1,
    Scalar = 2,
}

/// Kernel parameter metadata mirroring `MpRtGpuParam` semantics.
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KernelParam {
    pub kind: KernelParamKind,
    pub type_id: u32,
    pub offset_or_binding: u32,
    pub size: u32,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct KernelLayout {
    pub params: Vec<KernelParam>,
    pub num_buffers: u32,
    pub push_const_size: u32,
}

/// Runtime entry layout matching `MpRtGpuKernelEntry` (§20.1.7).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct KernelEntry {
    pub sid_hash: u64,
    pub backend: u32,
    pub blob: *const u8,
    pub blob_len: u64,
    pub num_params: u32,
    pub params: *const KernelParam,
    pub num_buffers: u32,
    pub push_const_size: u32,
}

/// Enforce `gpu fn` restrictions from §31.3.
pub fn validate_kernel(func: &MpirFn, type_ctx: &TypeCtx, diag: &mut DiagnosticBag) -> Result<(), ()> {
    let before = diag.error_count();

    check_kernel_type(func.ret_ty, type_ctx, diag, "kernel return type");

    for (_, ty) in &func.params {
        check_kernel_type(*ty, type_ctx, diag, "kernel parameter type");
    }
    for local in &func.locals {
        check_kernel_type(local.ty, type_ctx, diag, "kernel local type");
    }

    for block in &func.blocks {
        for instr in &block.instrs {
            check_kernel_type(instr.ty, type_ctx, diag, "kernel instruction result type");
            check_op_types(&instr.op, type_ctx, diag);

            match &instr.op {
                MpirOp::New { .. }
                | MpirOp::ArrNew { .. }
                | MpirOp::MapNew { .. }
                | MpirOp::StrBuilderNew
                | MpirOp::CallableCapture { .. } => {
                    emit_kernel_error(
                        diag,
                        "MPG1100",
                        "heap allocation is forbidden in gpu kernels",
                    );
                }
                MpirOp::ArcRetain { .. }
                | MpirOp::ArcRelease { .. }
                | MpirOp::ArcRetainWeak { .. }
                | MpirOp::ArcReleaseWeak { .. }
                | MpirOp::Share { .. }
                | MpirOp::CloneShared { .. }
                | MpirOp::CloneWeak { .. }
                | MpirOp::WeakDowngrade { .. }
                | MpirOp::WeakUpgrade { .. } => {
                    emit_kernel_error(
                        diag,
                        "MPG1101",
                        "ARC/ownership runtime operations are forbidden in gpu kernels",
                    );
                }
                MpirOp::CallIndirect { .. }
                | MpirOp::CallVoidIndirect { .. }
                | MpirOp::ArrMap { .. }
                | MpirOp::ArrFilter { .. }
                | MpirOp::ArrReduce { .. }
                | MpirOp::ArrForeach { .. } => {
                    emit_kernel_error(
                        diag,
                        "MPG1102",
                        "TCallable/dynamic dispatch is forbidden in gpu kernels",
                    );
                }
                MpirOp::Call { callee_sid, .. } | MpirOp::SuspendCall { callee_sid, .. } => {
                    if callee_sid == &func.sid {
                        emit_kernel_error(diag, "MPG1103", "recursive kernel calls are forbidden");
                    }
                }
                _ => {}
            }

            if let MpirOp::Const(c) = &instr.op {
                check_kernel_type(c.ty, type_ctx, diag, "kernel constant type");
            }
        }

        for op in &block.void_ops {
            check_void_op_types(op, type_ctx, diag);

            match op {
                MpirOpVoid::CallVoid { callee_sid, .. } => {
                    if callee_sid == &func.sid {
                        emit_kernel_error(diag, "MPG1103", "recursive kernel calls are forbidden");
                    }
                }
                MpirOpVoid::ArcRetain { .. }
                | MpirOpVoid::ArcRelease { .. }
                | MpirOpVoid::ArcRetainWeak { .. }
                | MpirOpVoid::ArcReleaseWeak { .. } => {
                    emit_kernel_error(
                        diag,
                        "MPG1101",
                        "ARC operations are forbidden in gpu kernels",
                    );
                }
                _ => {}
            }

        }

        if let MpirTerminator::Switch { arms, .. } = &block.terminator {
            for (c, _) in arms {
                check_kernel_type(c.ty, type_ctx, diag, "kernel switch-arm constant type");
            }
        }
    }

    if diag.error_count() > before {
        Err(())
    } else {
        Ok(())
    }
}

/// Compute deterministic Vulkan/SPIR-V kernel parameter layout (§31.6).
pub fn compute_kernel_layout(func: &MpirFn, type_ctx: &TypeCtx) -> KernelLayout {
    let mut params = Vec::with_capacity(func.params.len());
    let mut num_buffers = 0_u32;
    let mut scalar_offset = 0_u32;

    for (_, ty) in &func.params {
        if is_buffer_param(*ty, type_ctx) {
            params.push(KernelParam {
                kind: KernelParamKind::Buffer,
                type_id: 0,
                offset_or_binding: num_buffers,
                size: 0,
            });
            num_buffers = num_buffers.saturating_add(1);
            continue;
        }

        let size = scalar_size_bytes(*ty, type_ctx);
        let align = size.clamp(1, 16);
        scalar_offset = align_up(scalar_offset, align);

        params.push(KernelParam {
            kind: KernelParamKind::Scalar,
            type_id: ty.0,
            offset_or_binding: scalar_offset,
            size,
        });

        scalar_offset = scalar_offset.saturating_add(size);
    }

    KernelLayout {
        params,
        num_buffers,
        push_const_size: align_up(scalar_offset, 16),
    }
}

/// Minimal SPIR-V binary stub: 5-word module header.
pub fn generate_spirv_stub(_func: &MpirFn) -> Vec<u8> {
    let words: [u32; 5] = [
        0x0723_0203, // Magic number
        0x0001_0000, // Version 1.0
        0,           // Generator magic
        1,           // Bound (at least 1)
        0,           // Reserved schema
    ];

    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

pub fn sid_hash_64(sid: &Sid) -> u64 {
    // Deterministic FNV-1a (64-bit).
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for b in sid.0.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn align_up(value: u32, align: u32) -> u32 {
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

fn is_buffer_param(ty: TypeId, type_ctx: &TypeCtx) -> bool {
    if ty == fixed_type_ids::GPU_BUFFER_BASE {
        return true;
    }

    match type_ctx.lookup(ty) {
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

fn scalar_size_bytes(ty: TypeId, type_ctx: &TypeCtx) -> u32 {
    scalar_size_bytes_inner(ty, type_ctx, &mut HashSet::new()).max(1)
}

fn scalar_size_bytes_inner(ty: TypeId, type_ctx: &TypeCtx, seen: &mut HashSet<TypeId>) -> u32 {
    if !seen.insert(ty) {
        return 0;
    }

    let size = match type_ctx.lookup(ty) {
        Some(TypeKind::Prim(p)) => prim_size_bytes(*p),
        Some(TypeKind::HeapHandle { .. }) | Some(TypeKind::RawPtr { .. }) => 8,
        Some(TypeKind::BuiltinOption { inner }) => scalar_size_bytes_inner(*inner, type_ctx, seen).saturating_add(1),
        Some(TypeKind::BuiltinResult { ok, err }) => {
            1_u32
                .saturating_add(scalar_size_bytes_inner(*ok, type_ctx, seen))
                .saturating_add(scalar_size_bytes_inner(*err, type_ctx, seen))
        }
        Some(TypeKind::Arr { n, elem }) | Some(TypeKind::Vec { n, elem }) => {
            n.saturating_mul(scalar_size_bytes_inner(*elem, type_ctx, seen))
        }
        Some(TypeKind::Tuple { elems }) => elems
            .iter()
            .fold(0_u32, |acc, e| acc.saturating_add(scalar_size_bytes_inner(*e, type_ctx, seen))),
        Some(TypeKind::ValueStruct { .. }) | None => 0,
    };

    seen.remove(&ty);
    size
}

fn prim_size_bytes(prim: PrimType) -> u32 {
    match prim {
        PrimType::Unit => 0,
        PrimType::I1 | PrimType::U1 | PrimType::Bool => 1,
        PrimType::I8 | PrimType::U8 => 1,
        PrimType::I16 | PrimType::U16 | PrimType::F16 => 2,
        PrimType::I32 | PrimType::U32 | PrimType::F32 => 4,
        PrimType::I64 | PrimType::U64 | PrimType::F64 => 8,
        PrimType::I128 | PrimType::U128 => 16,
    }
}

fn check_kernel_type(ty: TypeId, type_ctx: &TypeCtx, diag: &mut DiagnosticBag, where_: &str) {
    match forbidden_kernel_type(ty, type_ctx, &mut HashSet::new()) {
        Some("Str") => emit_kernel_error(diag, "MPG1104", &format!("{where_}: Str is not allowed in gpu kernels")),
        Some("Array") => emit_kernel_error(diag, "MPG1105", &format!("{where_}: Array is not allowed in gpu kernels")),
        Some("Map") => emit_kernel_error(diag, "MPG1106", &format!("{where_}: Map is not allowed in gpu kernels")),
        Some("TCallable") => {
            emit_kernel_error(diag, "MPG1107", &format!("{where_}: TCallable is not allowed in gpu kernels"))
        }
        _ => {}
    }
}

fn forbidden_kernel_type<'a>(
    ty: TypeId,
    type_ctx: &TypeCtx,
    seen: &mut HashSet<TypeId>,
) -> Option<&'a str> {
    if !seen.insert(ty) {
        return None;
    }

    let out = match type_ctx.lookup(ty) {
        Some(TypeKind::HeapHandle { base, .. }) => match base {
            HeapBase::BuiltinStr => Some("Str"),
            HeapBase::BuiltinArray { .. } => Some("Array"),
            HeapBase::BuiltinMap { .. } => Some("Map"),
            HeapBase::Callable { .. } => Some("TCallable"),
            HeapBase::BuiltinMutex { inner }
            | HeapBase::BuiltinRwLock { inner }
            | HeapBase::BuiltinCell { inner }
            | HeapBase::BuiltinFuture { result: inner }
            | HeapBase::BuiltinChannelSend { elem: inner }
            | HeapBase::BuiltinChannelRecv { elem: inner } => forbidden_kernel_type(*inner, type_ctx, seen),
            HeapBase::BuiltinStrBuilder | HeapBase::UserType { .. } => None,
        },
        Some(TypeKind::BuiltinOption { inner }) => forbidden_kernel_type(*inner, type_ctx, seen),
        Some(TypeKind::BuiltinResult { ok, err }) => {
            forbidden_kernel_type(*ok, type_ctx, seen).or_else(|| forbidden_kernel_type(*err, type_ctx, seen))
        }
        Some(TypeKind::RawPtr { to }) => forbidden_kernel_type(*to, type_ctx, seen),
        Some(TypeKind::Arr { elem, .. }) | Some(TypeKind::Vec { elem, .. }) => {
            forbidden_kernel_type(*elem, type_ctx, seen)
        }
        Some(TypeKind::Tuple { elems }) => elems.iter().find_map(|t| forbidden_kernel_type(*t, type_ctx, seen)),
        Some(TypeKind::Prim(_)) | Some(TypeKind::ValueStruct { .. }) | None => None,
    };

    seen.remove(&ty);
    out
}

fn check_op_types(op: &MpirOp, type_ctx: &TypeCtx, diag: &mut DiagnosticBag) {
    match op {
        MpirOp::New { ty, .. }
        | MpirOp::Cast { to: ty, .. }
        | MpirOp::PtrNull { to: ty }
        | MpirOp::PtrFromAddr { to: ty, .. }
        | MpirOp::PtrLoad { to: ty, .. }
        | MpirOp::GpuShared { ty, .. }
        | MpirOp::JsonEncode { ty, .. }
        | MpirOp::JsonDecode { ty, .. }
        | MpirOp::Phi { ty, .. } => check_kernel_type(*ty, type_ctx, diag, "kernel op type"),
        MpirOp::ArrNew { elem_ty, .. } => check_kernel_type(*elem_ty, type_ctx, diag, "kernel array element type"),
        MpirOp::MapNew { key_ty, val_ty } => {
            check_kernel_type(*key_ty, type_ctx, diag, "kernel map key type");
            check_kernel_type(*val_ty, type_ctx, diag, "kernel map value type");
        }
        MpirOp::Call { inst, .. } | MpirOp::SuspendCall { inst, .. } => {
            for ty in inst {
                check_kernel_type(*ty, type_ctx, diag, "kernel call instantiation type");
            }
        }
        _ => {}
    }
}

fn check_void_op_types(op: &MpirOpVoid, type_ctx: &TypeCtx, diag: &mut DiagnosticBag) {
    match op {
        MpirOpVoid::PtrStore { to, .. } => check_kernel_type(*to, type_ctx, diag, "kernel void op type"),
        MpirOpVoid::CallVoid { inst, .. } => {
            for ty in inst {
                check_kernel_type(*ty, type_ctx, diag, "kernel call instantiation type");
            }
        }
        _ => {}
    }
}

fn emit_kernel_error(diag: &mut DiagnosticBag, code: &str, message: &str) {
    diag.emit(Diagnostic {
        code: code.to_string(),
        severity: Severity::Error,
        title: "GPU kernel restriction violation".to_string(),
        primary_span: None,
        secondary_spans: vec![],
        message: message.to_string(),
        explanation_md: None,
        why: None,
        suggested_fixes: vec![],
    });
}
