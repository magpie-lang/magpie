//! GPU kernel validation/layout helpers for Magpie GPU v0.1 (§31).
#![allow(
    clippy::field_reassign_with_default,
    clippy::manual_is_multiple_of,
    clippy::result_unit_err,
    clippy::too_many_arguments
)]

use magpie_diag::{Diagnostic, DiagnosticBag, Severity};
use magpie_mpir::{HirConstLit, MpirFn, MpirOp, MpirOpVoid, MpirTerminator, MpirValue};
use magpie_types::{fixed_type_ids, HeapBase, PrimType, Sid, TypeCtx, TypeId, TypeKind};
use std::collections::{HashMap, HashSet};

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
pub fn validate_kernel(
    func: &MpirFn,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) -> Result<(), ()> {
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
                MpirOpVoid::CallVoidIndirect { .. } => {
                    emit_kernel_error(
                        diag,
                        "MPG1102",
                        "TCallable/dynamic dispatch is forbidden in gpu kernels",
                    );
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

/// SPIR-V generator for a GPU kernel MPIR function.
pub fn generate_spirv(func: &MpirFn) -> Vec<u8> {
    let mut builder = SpirvBuilder::new();
    builder.emit_kernel_module(func);
    builder.finalize()
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
        Some(TypeKind::BuiltinOption { inner }) => {
            scalar_size_bytes_inner(*inner, type_ctx, seen).saturating_add(1)
        }
        Some(TypeKind::BuiltinResult { ok, err }) => 1_u32
            .saturating_add(scalar_size_bytes_inner(*ok, type_ctx, seen))
            .saturating_add(scalar_size_bytes_inner(*err, type_ctx, seen)),
        Some(TypeKind::Arr { n, elem }) | Some(TypeKind::Vec { n, elem }) => {
            n.saturating_mul(scalar_size_bytes_inner(*elem, type_ctx, seen))
        }
        Some(TypeKind::Tuple { elems }) => elems.iter().fold(0_u32, |acc, e| {
            acc.saturating_add(scalar_size_bytes_inner(*e, type_ctx, seen))
        }),
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
        Some("Str") => emit_kernel_error(
            diag,
            "MPG1104",
            &format!("{where_}: Str is not allowed in gpu kernels"),
        ),
        Some("Array") => emit_kernel_error(
            diag,
            "MPG1105",
            &format!("{where_}: Array is not allowed in gpu kernels"),
        ),
        Some("Map") => emit_kernel_error(
            diag,
            "MPG1106",
            &format!("{where_}: Map is not allowed in gpu kernels"),
        ),
        Some("TCallable") => emit_kernel_error(
            diag,
            "MPG1107",
            &format!("{where_}: TCallable is not allowed in gpu kernels"),
        ),
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
            | HeapBase::BuiltinChannelRecv { elem: inner } => {
                forbidden_kernel_type(*inner, type_ctx, seen)
            }
            HeapBase::BuiltinStrBuilder | HeapBase::UserType { .. } => None,
        },
        Some(TypeKind::BuiltinOption { inner }) => forbidden_kernel_type(*inner, type_ctx, seen),
        Some(TypeKind::BuiltinResult { ok, err }) => forbidden_kernel_type(*ok, type_ctx, seen)
            .or_else(|| forbidden_kernel_type(*err, type_ctx, seen)),
        Some(TypeKind::RawPtr { to }) => forbidden_kernel_type(*to, type_ctx, seen),
        Some(TypeKind::Arr { elem, .. }) | Some(TypeKind::Vec { elem, .. }) => {
            forbidden_kernel_type(*elem, type_ctx, seen)
        }
        Some(TypeKind::Tuple { elems }) => elems
            .iter()
            .find_map(|t| forbidden_kernel_type(*t, type_ctx, seen)),
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
        MpirOp::ArrNew { elem_ty, .. } => {
            check_kernel_type(*elem_ty, type_ctx, diag, "kernel array element type")
        }
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
        MpirOpVoid::PtrStore { to, .. } => {
            check_kernel_type(*to, type_ctx, diag, "kernel void op type")
        }
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
        rag_bundle: Vec::new(),
        related_docs: Vec::new(),
    });
}

#[derive(Debug, Default)]
struct SpirvBuilder {
    words: Vec<u32>,
    next_id: u32,
}

#[derive(Debug, Default)]
struct SpirvKernelIds {
    void_ty: u32,
    bool_ty: u32,
    int_ty: u32,
    float_ty: u32,
    vec3_int_ty: u32,
    ptr_input_vec3_int_ty: u32,
    ptr_input_int_ty: u32,
    runtime_array_int_ty: u32,
    storage_buffer_struct_ty: u32,
    ptr_storage_buffer_struct_ty: u32,
    ptr_storage_buffer_int_ty: u32,
    void_fn_ty: u32,
    global_invocation_id_var: u32,
    const_int_0: u32,
    const_int_1: u32,
    const_float_0: u32,
    const_float_1: u32,
    const_bool_false: u32,
    const_bool_true: u32,
    interface_vars: Vec<u32>,
}

#[derive(Debug, Default)]
struct SpirvKernelState {
    local_types: HashMap<u32, TypeId>,
    value_ids: HashMap<u32, u32>,
    value_types: HashMap<u32, u32>,
    buffer_vars: HashMap<u32, u32>,
    block_labels: HashMap<u32, u32>,
}

impl SpirvBuilder {
    const SPIRV_MAGIC: u32 = 0x0723_0203;
    const SPIRV_VERSION_1_6: u32 = 0x0001_0600;

    const OP_MEMORY_MODEL: u16 = 14;
    const OP_ENTRY_POINT: u16 = 15;
    const OP_EXECUTION_MODE: u16 = 16;
    const OP_CAPABILITY: u16 = 17;
    const OP_TYPE_VOID: u16 = 19;
    const OP_TYPE_BOOL: u16 = 20;
    const OP_TYPE_INT: u16 = 21;
    const OP_TYPE_FLOAT: u16 = 22;
    const OP_TYPE_VECTOR: u16 = 23;
    const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
    const OP_TYPE_STRUCT: u16 = 30;
    const OP_TYPE_POINTER: u16 = 32;
    const OP_TYPE_FUNCTION: u16 = 33;
    const OP_CONSTANT_TRUE: u16 = 41;
    const OP_CONSTANT_FALSE: u16 = 42;
    const OP_CONSTANT: u16 = 43;
    const OP_FUNCTION: u16 = 54;
    const OP_FUNCTION_END: u16 = 56;
    const OP_VARIABLE: u16 = 59;
    const OP_LOAD: u16 = 61;
    const OP_STORE: u16 = 62;
    const OP_ACCESS_CHAIN: u16 = 65;
    const OP_DECORATE: u16 = 71;
    const OP_MEMBER_DECORATE: u16 = 72;
    const OP_BITCAST: u16 = 124;
    const OP_IADD: u16 = 128;
    const OP_FADD: u16 = 129;
    const OP_ISUB: u16 = 130;
    const OP_FSUB: u16 = 131;
    const OP_IMUL: u16 = 132;
    const OP_FMUL: u16 = 133;
    const OP_UDIV: u16 = 134;
    const OP_SDIV: u16 = 135;
    const OP_FDIV: u16 = 136;
    const OP_UMOD: u16 = 137;
    const OP_SREM: u16 = 138;
    const OP_FREM: u16 = 140;
    const OP_SELECT: u16 = 169;
    const OP_IEQUAL: u16 = 170;
    const OP_INOTEQUAL: u16 = 171;
    const OP_UGREATER_THAN: u16 = 172;
    const OP_SGREATER_THAN: u16 = 173;
    const OP_UGREATER_THAN_EQUAL: u16 = 174;
    const OP_SGREATER_THAN_EQUAL: u16 = 175;
    const OP_ULESS_THAN: u16 = 176;
    const OP_SLESS_THAN: u16 = 177;
    const OP_ULESS_THAN_EQUAL: u16 = 178;
    const OP_SLESS_THAN_EQUAL: u16 = 179;
    const OP_FORD_EQUAL: u16 = 180;
    const OP_FORD_NOT_EQUAL: u16 = 182;
    const OP_FORD_LESS_THAN: u16 = 184;
    const OP_FORD_GREATER_THAN: u16 = 186;
    const OP_FORD_LESS_THAN_EQUAL: u16 = 188;
    const OP_FORD_GREATER_THAN_EQUAL: u16 = 190;
    const OP_SHIFT_RIGHT_LOGICAL: u16 = 194;
    const OP_SHIFT_RIGHT_ARITHMETIC: u16 = 195;
    const OP_SHIFT_LEFT_LOGICAL: u16 = 196;
    const OP_BITWISE_OR: u16 = 197;
    const OP_BITWISE_XOR: u16 = 198;
    const OP_BITWISE_AND: u16 = 199;
    const OP_PHI: u16 = 245;
    const OP_LABEL: u16 = 248;
    const OP_BRANCH: u16 = 249;
    const OP_BRANCH_CONDITIONAL: u16 = 250;
    const OP_RETURN: u16 = 253;
    const OP_RETURN_VALUE: u16 = 254;

    const CAPABILITY_SHADER: u32 = 1;
    const ADDRESSING_MODEL_LOGICAL: u32 = 0;
    const MEMORY_MODEL_GLSL450: u32 = 1;
    const EXEC_MODEL_GL_COMPUTE: u32 = 5;
    const EXEC_MODE_LOCAL_SIZE: u32 = 17;
    const STORAGE_CLASS_INPUT: u32 = 1;
    const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;
    const DECORATION_BLOCK: u32 = 2;
    const DECORATION_ARRAY_STRIDE: u32 = 6;
    const DECORATION_BUILT_IN: u32 = 11;
    const DECORATION_BINDING: u32 = 33;
    const DECORATION_DESCRIPTOR_SET: u32 = 34;
    const DECORATION_OFFSET: u32 = 35;
    const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;

    fn new() -> Self {
        Self {
            words: Vec::new(),
            next_id: 1,
        }
    }

    fn new_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        id
    }

    fn emit_kernel_module(&mut self, func: &MpirFn) {
        self.emit_header(Self::SPIRV_VERSION_1_6, 0);
        self.emit_capability(Self::CAPABILITY_SHADER);
        self.emit_memory_model(Self::ADDRESSING_MODEL_LOGICAL, Self::MEMORY_MODEL_GLSL450);

        let mut state = SpirvKernelState {
            local_types: Self::collect_local_types(func),
            ..Default::default()
        };
        let buffer_param_locals = Self::collect_buffer_param_locals(func);
        let ids = self.emit_kernel_decls(func, &buffer_param_locals, &mut state);

        let function_id = self.new_id();
        self.emit_entry_point(
            Self::EXEC_MODEL_GL_COMPUTE,
            function_id,
            &sanitize_spirv_entry_name(&func.name),
            &ids.interface_vars,
        );
        self.emit_execution_mode(function_id, Self::EXEC_MODE_LOCAL_SIZE, &[64, 1, 1]);
        self.emit_kernel_function(function_id, func, &ids, &mut state);
    }

    fn emit_header(&mut self, version: u32, generator: u32) {
        self.words.clear();
        self.words.extend_from_slice(&[
            Self::SPIRV_MAGIC,
            version,
            generator,
            1, // Bound; patched in finalize.
            0, // Reserved schema.
        ]);
    }

    fn emit_capability(&mut self, capability: u32) {
        self.emit_inst(Self::OP_CAPABILITY, &[capability]);
    }

    fn emit_memory_model(&mut self, addressing_model: u32, memory_model: u32) {
        self.emit_inst(Self::OP_MEMORY_MODEL, &[addressing_model, memory_model]);
    }

    fn emit_entry_point(
        &mut self,
        execution_model: u32,
        fn_id: u32,
        name: &str,
        interface_vars: &[u32],
    ) {
        let mut ops = vec![execution_model, fn_id];
        ops.extend(Self::encode_literal_string(name));
        ops.extend_from_slice(interface_vars);
        self.emit_inst(Self::OP_ENTRY_POINT, &ops);
    }

    fn emit_execution_mode(&mut self, fn_id: u32, mode: u32, literals: &[u32]) {
        let mut ops = vec![fn_id, mode];
        ops.extend_from_slice(literals);
        self.emit_inst(Self::OP_EXECUTION_MODE, &ops);
    }

    fn collect_local_types(func: &MpirFn) -> HashMap<u32, TypeId> {
        let mut out = HashMap::new();
        for (lid, ty) in &func.params {
            out.insert(lid.0, *ty);
        }
        for local in &func.locals {
            out.insert(local.id.0, local.ty);
        }
        for block in &func.blocks {
            for instr in &block.instrs {
                out.insert(instr.dst.0, instr.ty);
            }
        }
        out
    }

    fn collect_buffer_param_locals(func: &MpirFn) -> Vec<u32> {
        let mut used_as_buffer = HashSet::new();
        for block in &func.blocks {
            for instr in &block.instrs {
                if let MpirOp::GpuBufferLoad {
                    buf: MpirValue::Local(local),
                    ..
                } = &instr.op
                {
                    used_as_buffer.insert(local.0);
                }
            }
            for op in &block.void_ops {
                if let MpirOpVoid::GpuBufferStore {
                    buf: MpirValue::Local(local),
                    ..
                } = op
                {
                    used_as_buffer.insert(local.0);
                }
            }
        }

        let mut out = Vec::new();
        for (local, ty) in &func.params {
            if *ty == fixed_type_ids::GPU_BUFFER_BASE || used_as_buffer.contains(&local.0) {
                out.push(local.0);
            }
        }
        out
    }

    fn emit_kernel_decls(
        &mut self,
        func: &MpirFn,
        buffer_param_locals: &[u32],
        state: &mut SpirvKernelState,
    ) -> SpirvKernelIds {
        let mut ids = SpirvKernelIds::default();

        ids.void_ty = self.new_id();
        self.emit_inst(Self::OP_TYPE_VOID, &[ids.void_ty]);

        ids.bool_ty = self.new_id();
        self.emit_inst(Self::OP_TYPE_BOOL, &[ids.bool_ty]);

        ids.int_ty = self.new_id();
        self.emit_inst(Self::OP_TYPE_INT, &[ids.int_ty, 32, 0]);

        ids.float_ty = self.new_id();
        self.emit_inst(Self::OP_TYPE_FLOAT, &[ids.float_ty, 32]);

        ids.vec3_int_ty = self.new_id();
        self.emit_inst(Self::OP_TYPE_VECTOR, &[ids.vec3_int_ty, ids.int_ty, 3]);

        ids.ptr_input_vec3_int_ty = self.new_id();
        self.emit_inst(
            Self::OP_TYPE_POINTER,
            &[
                ids.ptr_input_vec3_int_ty,
                Self::STORAGE_CLASS_INPUT,
                ids.vec3_int_ty,
            ],
        );

        ids.ptr_input_int_ty = self.new_id();
        self.emit_inst(
            Self::OP_TYPE_POINTER,
            &[ids.ptr_input_int_ty, Self::STORAGE_CLASS_INPUT, ids.int_ty],
        );

        ids.runtime_array_int_ty = self.new_id();
        self.emit_inst(
            Self::OP_TYPE_RUNTIME_ARRAY,
            &[ids.runtime_array_int_ty, ids.int_ty],
        );

        ids.storage_buffer_struct_ty = self.new_id();
        self.emit_inst(
            Self::OP_TYPE_STRUCT,
            &[ids.storage_buffer_struct_ty, ids.runtime_array_int_ty],
        );

        ids.ptr_storage_buffer_struct_ty = self.new_id();
        self.emit_inst(
            Self::OP_TYPE_POINTER,
            &[
                ids.ptr_storage_buffer_struct_ty,
                Self::STORAGE_CLASS_STORAGE_BUFFER,
                ids.storage_buffer_struct_ty,
            ],
        );

        ids.ptr_storage_buffer_int_ty = self.new_id();
        self.emit_inst(
            Self::OP_TYPE_POINTER,
            &[
                ids.ptr_storage_buffer_int_ty,
                Self::STORAGE_CLASS_STORAGE_BUFFER,
                ids.int_ty,
            ],
        );

        ids.void_fn_ty = self.new_id();
        self.emit_inst(Self::OP_TYPE_FUNCTION, &[ids.void_fn_ty, ids.void_ty]);

        ids.const_int_0 = self.new_id();
        self.emit_inst(Self::OP_CONSTANT, &[ids.int_ty, ids.const_int_0, 0]);

        ids.const_int_1 = self.new_id();
        self.emit_inst(Self::OP_CONSTANT, &[ids.int_ty, ids.const_int_1, 1]);

        ids.const_float_0 = self.new_id();
        self.emit_inst(
            Self::OP_CONSTANT,
            &[ids.float_ty, ids.const_float_0, 0.0_f32.to_bits()],
        );

        ids.const_float_1 = self.new_id();
        self.emit_inst(
            Self::OP_CONSTANT,
            &[ids.float_ty, ids.const_float_1, 1.0_f32.to_bits()],
        );

        ids.const_bool_false = self.new_id();
        self.emit_inst(
            Self::OP_CONSTANT_FALSE,
            &[ids.bool_ty, ids.const_bool_false],
        );

        ids.const_bool_true = self.new_id();
        self.emit_inst(Self::OP_CONSTANT_TRUE, &[ids.bool_ty, ids.const_bool_true]);

        self.emit_inst(
            Self::OP_DECORATE,
            &[ids.runtime_array_int_ty, Self::DECORATION_ARRAY_STRIDE, 4],
        );
        self.emit_inst(
            Self::OP_DECORATE,
            &[ids.storage_buffer_struct_ty, Self::DECORATION_BLOCK],
        );
        self.emit_inst(
            Self::OP_MEMBER_DECORATE,
            &[ids.storage_buffer_struct_ty, 0, Self::DECORATION_OFFSET, 0],
        );

        ids.global_invocation_id_var = self.new_id();
        self.emit_inst(
            Self::OP_VARIABLE,
            &[
                ids.ptr_input_vec3_int_ty,
                ids.global_invocation_id_var,
                Self::STORAGE_CLASS_INPUT,
            ],
        );
        self.emit_inst(
            Self::OP_DECORATE,
            &[
                ids.global_invocation_id_var,
                Self::DECORATION_BUILT_IN,
                Self::BUILTIN_GLOBAL_INVOCATION_ID,
            ],
        );
        ids.interface_vars.push(ids.global_invocation_id_var);

        for (binding, local_id) in buffer_param_locals.iter().enumerate() {
            let var_id = self.new_id();
            self.emit_inst(
                Self::OP_VARIABLE,
                &[
                    ids.ptr_storage_buffer_struct_ty,
                    var_id,
                    Self::STORAGE_CLASS_STORAGE_BUFFER,
                ],
            );
            self.emit_inst(
                Self::OP_DECORATE,
                &[var_id, Self::DECORATION_DESCRIPTOR_SET, 0],
            );
            self.emit_inst(
                Self::OP_DECORATE,
                &[var_id, Self::DECORATION_BINDING, binding as u32],
            );

            ids.interface_vars.push(var_id);
            state.buffer_vars.insert(*local_id, var_id);
        }

        for (local, ty) in &func.params {
            if state.buffer_vars.contains_key(&local.0) {
                continue;
            }
            let spirv_ty = self.spirv_scalar_for_type_id(*ty, &ids);
            let value_id = self.default_value_for_type(spirv_ty, &ids);
            state.value_ids.insert(local.0, value_id);
            state.value_types.insert(local.0, spirv_ty);
        }

        ids
    }

    fn emit_kernel_function(
        &mut self,
        function_id: u32,
        func: &MpirFn,
        ids: &SpirvKernelIds,
        state: &mut SpirvKernelState,
    ) {
        self.emit_inst(
            Self::OP_FUNCTION,
            &[ids.void_ty, function_id, 0, ids.void_fn_ty],
        );

        if func.blocks.is_empty() {
            let label = self.new_id();
            self.emit_inst(Self::OP_LABEL, &[label]);
            self.emit_inst(Self::OP_RETURN, &[]);
            self.emit_inst(Self::OP_FUNCTION_END, &[]);
            return;
        }

        for block in &func.blocks {
            state.block_labels.insert(block.id.0, self.new_id());
        }

        for block in &func.blocks {
            let label = state
                .block_labels
                .get(&block.id.0)
                .copied()
                .unwrap_or_else(|| self.new_id());
            self.emit_inst(Self::OP_LABEL, &[label]);

            for instr in &block.instrs {
                self.emit_kernel_instr(instr, ids, state, label);
            }
            for op in &block.void_ops {
                self.emit_kernel_void_op(op, ids, state);
            }
            self.emit_kernel_terminator(&block.terminator, ids, state, label);
        }

        self.emit_inst(Self::OP_FUNCTION_END, &[]);
    }

    fn emit_kernel_instr(
        &mut self,
        instr: &magpie_mpir::MpirInstr,
        ids: &SpirvKernelIds,
        state: &mut SpirvKernelState,
        current_label: u32,
    ) {
        match &instr.op {
            MpirOp::Const(c) => {
                let dst_ty = self.spirv_scalar_for_type_id(instr.ty, ids);
                let value_id = self.constant_from_literal(&c.lit, dst_ty, ids);
                self.set_local_value(instr.dst.0, value_id, dst_ty, state);
            }
            MpirOp::Move { v }
            | MpirOp::BorrowShared { v }
            | MpirOp::BorrowMut { v }
            | MpirOp::Share { v }
            | MpirOp::CloneShared { v }
            | MpirOp::CloneWeak { v }
            | MpirOp::WeakDowngrade { v }
            | MpirOp::WeakUpgrade { v } => {
                if Self::is_buffer_type(instr.ty) {
                    if let Some(buf_var) = self.resolve_buffer_var(v, state) {
                        state.buffer_vars.insert(instr.dst.0, buf_var);
                    }
                    return;
                }
                let dst_ty = self.spirv_scalar_for_type_id(instr.ty, ids);
                let value_id = self.resolve_value(v, dst_ty, ids, state);
                self.set_local_value(instr.dst.0, value_id, dst_ty, state);
            }

            MpirOp::IAdd { lhs, rhs } | MpirOp::IAddWrap { lhs, rhs } => {
                self.emit_binary_op(instr.dst.0, Self::OP_IADD, lhs, rhs, ids.int_ty, ids, state);
            }
            MpirOp::ISub { lhs, rhs } | MpirOp::ISubWrap { lhs, rhs } => {
                self.emit_binary_op(instr.dst.0, Self::OP_ISUB, lhs, rhs, ids.int_ty, ids, state);
            }
            MpirOp::IMul { lhs, rhs } | MpirOp::IMulWrap { lhs, rhs } => {
                self.emit_binary_op(instr.dst.0, Self::OP_IMUL, lhs, rhs, ids.int_ty, ids, state);
            }
            MpirOp::ISDiv { lhs, rhs } => {
                self.emit_binary_op(instr.dst.0, Self::OP_SDIV, lhs, rhs, ids.int_ty, ids, state);
            }
            MpirOp::IUDiv { lhs, rhs } => {
                self.emit_binary_op(instr.dst.0, Self::OP_UDIV, lhs, rhs, ids.int_ty, ids, state);
            }
            MpirOp::ISRem { lhs, rhs } => {
                self.emit_binary_op(instr.dst.0, Self::OP_SREM, lhs, rhs, ids.int_ty, ids, state);
            }
            MpirOp::IURem { lhs, rhs } => {
                self.emit_binary_op(instr.dst.0, Self::OP_UMOD, lhs, rhs, ids.int_ty, ids, state);
            }
            MpirOp::IAnd { lhs, rhs } => {
                self.emit_binary_op(
                    instr.dst.0,
                    Self::OP_BITWISE_AND,
                    lhs,
                    rhs,
                    ids.int_ty,
                    ids,
                    state,
                );
            }
            MpirOp::IOr { lhs, rhs } => {
                self.emit_binary_op(
                    instr.dst.0,
                    Self::OP_BITWISE_OR,
                    lhs,
                    rhs,
                    ids.int_ty,
                    ids,
                    state,
                );
            }
            MpirOp::IXor { lhs, rhs } => {
                self.emit_binary_op(
                    instr.dst.0,
                    Self::OP_BITWISE_XOR,
                    lhs,
                    rhs,
                    ids.int_ty,
                    ids,
                    state,
                );
            }
            MpirOp::IShl { lhs, rhs } => {
                self.emit_binary_op(
                    instr.dst.0,
                    Self::OP_SHIFT_LEFT_LOGICAL,
                    lhs,
                    rhs,
                    ids.int_ty,
                    ids,
                    state,
                );
            }
            MpirOp::ILshr { lhs, rhs } => {
                self.emit_binary_op(
                    instr.dst.0,
                    Self::OP_SHIFT_RIGHT_LOGICAL,
                    lhs,
                    rhs,
                    ids.int_ty,
                    ids,
                    state,
                );
            }
            MpirOp::IAshr { lhs, rhs } => {
                self.emit_binary_op(
                    instr.dst.0,
                    Self::OP_SHIFT_RIGHT_ARITHMETIC,
                    lhs,
                    rhs,
                    ids.int_ty,
                    ids,
                    state,
                );
            }

            MpirOp::FAdd { lhs, rhs } | MpirOp::FAddFast { lhs, rhs } => {
                self.emit_binary_op(
                    instr.dst.0,
                    Self::OP_FADD,
                    lhs,
                    rhs,
                    ids.float_ty,
                    ids,
                    state,
                );
            }
            MpirOp::FSub { lhs, rhs } | MpirOp::FSubFast { lhs, rhs } => {
                self.emit_binary_op(
                    instr.dst.0,
                    Self::OP_FSUB,
                    lhs,
                    rhs,
                    ids.float_ty,
                    ids,
                    state,
                );
            }
            MpirOp::FMul { lhs, rhs } | MpirOp::FMulFast { lhs, rhs } => {
                self.emit_binary_op(
                    instr.dst.0,
                    Self::OP_FMUL,
                    lhs,
                    rhs,
                    ids.float_ty,
                    ids,
                    state,
                );
            }
            MpirOp::FDiv { lhs, rhs } | MpirOp::FDivFast { lhs, rhs } => {
                self.emit_binary_op(
                    instr.dst.0,
                    Self::OP_FDIV,
                    lhs,
                    rhs,
                    ids.float_ty,
                    ids,
                    state,
                );
            }
            MpirOp::FRem { lhs, rhs } => {
                self.emit_binary_op(
                    instr.dst.0,
                    Self::OP_FREM,
                    lhs,
                    rhs,
                    ids.float_ty,
                    ids,
                    state,
                );
            }

            MpirOp::ICmp { pred, lhs, rhs } => {
                let lhs_id = self.resolve_value(lhs, ids.int_ty, ids, state);
                let rhs_id = self.resolve_value(rhs, ids.int_ty, ids, state);
                let opcode = match pred.to_ascii_lowercase().as_str() {
                    "eq" => Self::OP_IEQUAL,
                    "ne" => Self::OP_INOTEQUAL,
                    "slt" => Self::OP_SLESS_THAN,
                    "sgt" => Self::OP_SGREATER_THAN,
                    "sle" => Self::OP_SLESS_THAN_EQUAL,
                    "sge" => Self::OP_SGREATER_THAN_EQUAL,
                    "ult" => Self::OP_ULESS_THAN,
                    "ugt" => Self::OP_UGREATER_THAN,
                    "ule" => Self::OP_ULESS_THAN_EQUAL,
                    "uge" => Self::OP_UGREATER_THAN_EQUAL,
                    _ => Self::OP_IEQUAL,
                };
                let result_id = self.new_id();
                self.emit_inst(opcode, &[ids.bool_ty, result_id, lhs_id, rhs_id]);
                self.set_local_value(instr.dst.0, result_id, ids.bool_ty, state);
            }
            MpirOp::FCmp { pred, lhs, rhs } => {
                let lhs_id = self.resolve_value(lhs, ids.float_ty, ids, state);
                let rhs_id = self.resolve_value(rhs, ids.float_ty, ids, state);
                let opcode = match pred.to_ascii_lowercase().as_str() {
                    "eq" => Self::OP_FORD_EQUAL,
                    "ne" => Self::OP_FORD_NOT_EQUAL,
                    "lt" => Self::OP_FORD_LESS_THAN,
                    "gt" => Self::OP_FORD_GREATER_THAN,
                    "le" => Self::OP_FORD_LESS_THAN_EQUAL,
                    "ge" => Self::OP_FORD_GREATER_THAN_EQUAL,
                    _ => Self::OP_FORD_EQUAL,
                };
                let result_id = self.new_id();
                self.emit_inst(opcode, &[ids.bool_ty, result_id, lhs_id, rhs_id]);
                self.set_local_value(instr.dst.0, result_id, ids.bool_ty, state);
            }

            MpirOp::GpuGlobalId => {
                let ptr_id = self.new_id();
                self.emit_inst(
                    Self::OP_ACCESS_CHAIN,
                    &[
                        ids.ptr_input_int_ty,
                        ptr_id,
                        ids.global_invocation_id_var,
                        ids.const_int_0,
                    ],
                );
                let raw_id = self.new_id();
                self.emit_inst(Self::OP_LOAD, &[ids.int_ty, raw_id, ptr_id]);
                let dst_ty = self.spirv_scalar_for_type_id(instr.ty, ids);
                let out_id = self.cast_value(raw_id, ids.int_ty, dst_ty, ids);
                self.set_local_value(instr.dst.0, out_id, dst_ty, state);
            }

            MpirOp::GpuBufferLoad { buf, idx } => {
                let dst_ty = self.spirv_scalar_for_type_id(instr.ty, ids);
                if let Some(buf_var) = self.resolve_buffer_var(buf, state) {
                    let idx_id = self.resolve_value(idx, ids.int_ty, ids, state);
                    let ptr_id = self.new_id();
                    self.emit_inst(
                        Self::OP_ACCESS_CHAIN,
                        &[
                            ids.ptr_storage_buffer_int_ty,
                            ptr_id,
                            buf_var,
                            ids.const_int_0,
                            idx_id,
                        ],
                    );
                    let raw_id = self.new_id();
                    self.emit_inst(Self::OP_LOAD, &[ids.int_ty, raw_id, ptr_id]);
                    let out_id = self.cast_value(raw_id, ids.int_ty, dst_ty, ids);
                    self.set_local_value(instr.dst.0, out_id, dst_ty, state);
                } else {
                    let value_id = self.default_value_for_type(dst_ty, ids);
                    self.set_local_value(instr.dst.0, value_id, dst_ty, state);
                }
            }
            MpirOp::Phi { ty, incomings } => {
                let phi_ty = self.spirv_scalar_for_type_id(*ty, ids);
                let result_id = self.new_id();
                let mut operands = Vec::with_capacity(2 + incomings.len() * 2);
                operands.push(phi_ty);
                operands.push(result_id);

                for (pred_bb, incoming) in incomings {
                    let incoming_id = self.resolve_value(incoming, phi_ty, ids, state);
                    let pred_label = state
                        .block_labels
                        .get(&pred_bb.0)
                        .copied()
                        .unwrap_or(current_label);
                    operands.push(incoming_id);
                    operands.push(pred_label);
                }

                self.emit_inst(Self::OP_PHI, &operands);
                self.set_local_value(instr.dst.0, result_id, phi_ty, state);
            }

            _ => {
                if !Self::is_buffer_type(instr.ty) {
                    let dst_ty = self.spirv_scalar_for_type_id(instr.ty, ids);
                    let value_id = self.default_value_for_type(dst_ty, ids);
                    self.set_local_value(instr.dst.0, value_id, dst_ty, state);
                }
            }
        }
    }

    fn emit_binary_op(
        &mut self,
        dst_local: u32,
        opcode: u16,
        lhs: &MpirValue,
        rhs: &MpirValue,
        ty: u32,
        ids: &SpirvKernelIds,
        state: &mut SpirvKernelState,
    ) {
        let lhs_id = self.resolve_value(lhs, ty, ids, state);
        let rhs_id = self.resolve_value(rhs, ty, ids, state);
        let result_id = self.new_id();
        self.emit_inst(opcode, &[ty, result_id, lhs_id, rhs_id]);
        self.set_local_value(dst_local, result_id, ty, state);
    }

    fn emit_kernel_void_op(
        &mut self,
        op: &MpirOpVoid,
        ids: &SpirvKernelIds,
        state: &mut SpirvKernelState,
    ) {
        if let MpirOpVoid::GpuBufferStore { buf, idx, val } = op {
            let Some(buf_var) = self.resolve_buffer_var(buf, state) else {
                return;
            };

            let idx_id = self.resolve_value(idx, ids.int_ty, ids, state);
            let val_id = self.resolve_value(val, ids.int_ty, ids, state);

            let ptr_id = self.new_id();
            self.emit_inst(
                Self::OP_ACCESS_CHAIN,
                &[
                    ids.ptr_storage_buffer_int_ty,
                    ptr_id,
                    buf_var,
                    ids.const_int_0,
                    idx_id,
                ],
            );
            self.emit_inst(Self::OP_STORE, &[ptr_id, val_id]);
        }
    }

    fn emit_kernel_terminator(
        &mut self,
        term: &MpirTerminator,
        ids: &SpirvKernelIds,
        state: &mut SpirvKernelState,
        current_label: u32,
    ) {
        match term {
            MpirTerminator::Ret(Some(v)) => {
                let ret_ty = self.infer_value_type(v, ids, state);
                let ret_id = self.resolve_value(v, ret_ty, ids, state);
                self.emit_inst(Self::OP_RETURN_VALUE, &[ret_id]);
            }
            MpirTerminator::Ret(None) => {
                self.emit_inst(Self::OP_RETURN, &[]);
            }
            MpirTerminator::Br(bb) => {
                let target = state
                    .block_labels
                    .get(&bb.0)
                    .copied()
                    .unwrap_or(current_label);
                self.emit_inst(Self::OP_BRANCH, &[target]);
            }
            MpirTerminator::Cbr {
                cond,
                then_bb,
                else_bb,
            } => {
                let cond_id = self.resolve_value(cond, ids.bool_ty, ids, state);
                let then_label = state
                    .block_labels
                    .get(&then_bb.0)
                    .copied()
                    .unwrap_or(current_label);
                let else_label = state
                    .block_labels
                    .get(&else_bb.0)
                    .copied()
                    .unwrap_or(current_label);
                self.emit_inst(
                    Self::OP_BRANCH_CONDITIONAL,
                    &[cond_id, then_label, else_label],
                );
            }
            MpirTerminator::Switch { default, .. } => {
                let target = state
                    .block_labels
                    .get(&default.0)
                    .copied()
                    .unwrap_or(current_label);
                self.emit_inst(Self::OP_BRANCH, &[target]);
            }
            MpirTerminator::Unreachable => {
                self.emit_inst(Self::OP_RETURN, &[]);
            }
        }
    }

    fn resolve_buffer_var(&self, value: &MpirValue, state: &SpirvKernelState) -> Option<u32> {
        match value {
            MpirValue::Local(local) => state.buffer_vars.get(&local.0).copied(),
            MpirValue::Const(_) => None,
        }
    }

    fn resolve_value(
        &mut self,
        value: &MpirValue,
        expected_ty: u32,
        ids: &SpirvKernelIds,
        state: &mut SpirvKernelState,
    ) -> u32 {
        match value {
            MpirValue::Local(local) => {
                if let Some(value_id) = state.value_ids.get(&local.0).copied() {
                    let from_ty = state
                        .value_types
                        .get(&local.0)
                        .copied()
                        .unwrap_or(expected_ty);
                    return self.cast_value(value_id, from_ty, expected_ty, ids);
                }

                let inferred_ty = state
                    .local_types
                    .get(&local.0)
                    .copied()
                    .map(|ty| self.spirv_scalar_for_type_id(ty, ids))
                    .unwrap_or(expected_ty);
                let default_id = self.default_value_for_type(inferred_ty, ids);
                state.value_ids.insert(local.0, default_id);
                state.value_types.insert(local.0, inferred_ty);
                self.cast_value(default_id, inferred_ty, expected_ty, ids)
            }
            MpirValue::Const(c) => self.constant_from_literal(&c.lit, expected_ty, ids),
        }
    }

    fn infer_value_type(
        &self,
        value: &MpirValue,
        ids: &SpirvKernelIds,
        state: &SpirvKernelState,
    ) -> u32 {
        match value {
            MpirValue::Local(local) => state
                .value_types
                .get(&local.0)
                .copied()
                .or_else(|| {
                    state
                        .local_types
                        .get(&local.0)
                        .copied()
                        .map(|ty| self.spirv_scalar_for_type_id(ty, ids))
                })
                .unwrap_or(ids.int_ty),
            MpirValue::Const(c) => self.spirv_scalar_for_type_id(c.ty, ids),
        }
    }

    fn cast_value(&mut self, value_id: u32, from_ty: u32, to_ty: u32, ids: &SpirvKernelIds) -> u32 {
        if from_ty == to_ty {
            return value_id;
        }

        if to_ty == ids.bool_ty {
            let result_id = self.new_id();
            if from_ty == ids.float_ty {
                self.emit_inst(
                    Self::OP_FORD_NOT_EQUAL,
                    &[ids.bool_ty, result_id, value_id, ids.const_float_0],
                );
            } else {
                self.emit_inst(
                    Self::OP_INOTEQUAL,
                    &[ids.bool_ty, result_id, value_id, ids.const_int_0],
                );
            }
            return result_id;
        }

        if from_ty == ids.bool_ty && to_ty == ids.int_ty {
            let result_id = self.new_id();
            self.emit_inst(
                Self::OP_SELECT,
                &[
                    ids.int_ty,
                    result_id,
                    value_id,
                    ids.const_int_1,
                    ids.const_int_0,
                ],
            );
            return result_id;
        }

        if from_ty == ids.bool_ty && to_ty == ids.float_ty {
            let result_id = self.new_id();
            self.emit_inst(
                Self::OP_SELECT,
                &[
                    ids.float_ty,
                    result_id,
                    value_id,
                    ids.const_float_1,
                    ids.const_float_0,
                ],
            );
            return result_id;
        }

        if (from_ty == ids.int_ty && to_ty == ids.float_ty)
            || (from_ty == ids.float_ty && to_ty == ids.int_ty)
        {
            let result_id = self.new_id();
            self.emit_inst(Self::OP_BITCAST, &[to_ty, result_id, value_id]);
            return result_id;
        }

        value_id
    }

    fn set_local_value(
        &self,
        local_id: u32,
        value_id: u32,
        spirv_ty: u32,
        state: &mut SpirvKernelState,
    ) {
        state.value_ids.insert(local_id, value_id);
        state.value_types.insert(local_id, spirv_ty);
    }

    fn constant_from_literal(
        &mut self,
        lit: &HirConstLit,
        expected_ty: u32,
        ids: &SpirvKernelIds,
    ) -> u32 {
        match lit {
            HirConstLit::IntLit(v) => {
                if expected_ty == ids.bool_ty {
                    return if *v == 0 {
                        ids.const_bool_false
                    } else {
                        ids.const_bool_true
                    };
                }
                if expected_ty == ids.float_ty {
                    let id = self.new_id();
                    self.emit_inst(
                        Self::OP_CONSTANT,
                        &[ids.float_ty, id, (*v as f32).to_bits()],
                    );
                    return id;
                }
                let id = self.new_id();
                self.emit_inst(Self::OP_CONSTANT, &[expected_ty, id, *v as u32]);
                id
            }
            HirConstLit::FloatLit(v) => {
                if expected_ty == ids.float_ty {
                    let id = self.new_id();
                    self.emit_inst(
                        Self::OP_CONSTANT,
                        &[ids.float_ty, id, (*v as f32).to_bits()],
                    );
                    return id;
                }
                if expected_ty == ids.bool_ty {
                    return if *v == 0.0 {
                        ids.const_bool_false
                    } else {
                        ids.const_bool_true
                    };
                }
                let id = self.new_id();
                self.emit_inst(Self::OP_CONSTANT, &[expected_ty, id, *v as u32]);
                id
            }
            HirConstLit::BoolLit(v) => {
                if expected_ty == ids.bool_ty {
                    if *v {
                        ids.const_bool_true
                    } else {
                        ids.const_bool_false
                    }
                } else if expected_ty == ids.float_ty {
                    if *v {
                        ids.const_float_1
                    } else {
                        ids.const_float_0
                    }
                } else if *v {
                    ids.const_int_1
                } else {
                    ids.const_int_0
                }
            }
            HirConstLit::StringLit(_) | HirConstLit::Unit => {
                self.default_value_for_type(expected_ty, ids)
            }
        }
    }

    fn default_value_for_type(&self, ty: u32, ids: &SpirvKernelIds) -> u32 {
        if ty == ids.bool_ty {
            ids.const_bool_false
        } else if ty == ids.float_ty {
            ids.const_float_0
        } else {
            ids.const_int_0
        }
    }

    fn spirv_scalar_for_type_id(&self, ty: TypeId, ids: &SpirvKernelIds) -> u32 {
        if matches!(
            ty,
            fixed_type_ids::F16 | fixed_type_ids::F32 | fixed_type_ids::F64
        ) {
            ids.float_ty
        } else if matches!(ty, fixed_type_ids::BOOL | fixed_type_ids::U1) {
            ids.bool_ty
        } else {
            ids.int_ty
        }
    }

    fn is_buffer_type(ty: TypeId) -> bool {
        ty == fixed_type_ids::GPU_BUFFER_BASE
    }

    fn finalize(mut self) -> Vec<u8> {
        if self.words.len() >= 5 {
            self.words[3] = self.next_id.max(1);
        }

        let mut out = Vec::with_capacity(self.words.len() * 4);
        for w in self.words {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    fn emit_inst(&mut self, opcode: u16, operands: &[u32]) {
        let wc = (1 + operands.len()) as u32;
        self.words.push((wc << 16) | u32::from(opcode));
        self.words.extend_from_slice(operands);
    }

    fn encode_literal_string(s: &str) -> Vec<u32> {
        let mut bytes = s.as_bytes().to_vec();
        bytes.push(0);
        while bytes.len() % 4 != 0 {
            bytes.push(0);
        }

        bytes
            .chunks(4)
            .map(|chunk| {
                u32::from(chunk[0])
                    | (u32::from(chunk[1]) << 8)
                    | (u32::from(chunk[2]) << 16)
                    | (u32::from(chunk[3]) << 24)
            })
            .collect()
    }
}

pub fn generate_spirv_with_layout(
    func: &magpie_mpir::MpirFn,
    layout: &KernelLayout,
    type_ctx: &magpie_types::TypeCtx,
) -> Result<Vec<u8>, String> {
    if func.ret_ty != fixed_type_ids::UNIT {
        return Err(format!(
            "gpu kernel '{}' must return unit (type_id 0), found type_id {}",
            func.name, func.ret_ty.0
        ));
    }
    if layout.params.len() != func.params.len() {
        return Err(format!(
            "kernel layout parameter count mismatch for '{}': layout={}, fn={}",
            func.name,
            layout.params.len(),
            func.params.len()
        ));
    }
    for (_, ty) in &func.params {
        if type_ctx.lookup(*ty).is_none() {
            return Err(format!(
                "kernel '{}' references unknown type_id {} in parameters",
                func.name, ty.0
            ));
        }
    }

    Ok(generate_spirv(func))
}

pub fn embed_spirv_as_llvm_const(spirv_bytes: &[u8], symbol_name: &str) -> String {
    let symbol = sanitize_llvm_symbol(symbol_name);
    let escaped = spirv_bytes
        .iter()
        .map(|b| format!("\\{:02X}", b))
        .collect::<String>();
    format!(
        "@{symbol} = private constant [{} x i8] c\"{}\"",
        spirv_bytes.len(),
        escaped
    )
}

pub fn generate_kernel_registry_ir(kernels: &[(String, KernelLayout, Vec<u8>)]) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();

    writeln!(out, "%MpRtGpuParam = type {{ i8, i32, i32, i32 }}").unwrap();
    writeln!(
        out,
        "%MpRtGpuKernelEntry = type {{ i64, i32, ptr, i64, i32, ptr, i32, i32 }}"
    )
    .unwrap();
    writeln!(out).unwrap();

    for (idx, (_, layout, blob)) in kernels.iter().enumerate() {
        let blob_sym = format!("mp_gpu_spv_blob_{idx}");
        writeln!(out, "{}", embed_spirv_as_llvm_const(blob, &blob_sym)).unwrap();

        if !layout.params.is_empty() {
            let param_sym = format!("mp_gpu_kernel_params_{idx}");
            let elems = layout
                .params
                .iter()
                .map(llvm_kernel_param_const)
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(
                out,
                "@{param_sym} = private constant [{} x %MpRtGpuParam] [{}]",
                layout.params.len(),
                elems
            )
            .unwrap();
        }
        writeln!(out).unwrap();
    }

    let mut entries = Vec::with_capacity(kernels.len());
    for (idx, (sid_str, layout, blob)) in kernels.iter().enumerate() {
        let sid_hash = sid_hash_64(&Sid(sid_str.clone()));
        let blob_sym = format!("mp_gpu_spv_blob_{idx}");
        let blob_ptr = format!(
            "ptr getelementptr inbounds ([{} x i8], ptr @{}, i64 0, i64 0)",
            blob.len(),
            blob_sym
        );
        let params_ptr = if layout.params.is_empty() {
            "ptr null".to_string()
        } else {
            format!(
                "ptr getelementptr inbounds ([{} x %MpRtGpuParam], ptr @mp_gpu_kernel_params_{}, i64 0, i64 0)",
                layout.params.len(),
                idx
            )
        };

        entries.push(format!(
            "%MpRtGpuKernelEntry {{ i64 {sid_hash}, i32 {backend}, ptr {blob_ptr}, i64 {blob_len}, i32 {num_params}, ptr {params_ptr}, i32 {num_buffers}, i32 {push_const_size} }}",
            backend = GPU_BACKEND_SPV,
            blob_len = blob.len(),
            num_params = layout.params.len(),
            num_buffers = layout.num_buffers,
            push_const_size = layout.push_const_size,
        ));
    }

    writeln!(
        out,
        "@mp_gpu_kernel_registry = private constant [{} x %MpRtGpuKernelEntry] [{}]",
        kernels.len(),
        entries.join(", ")
    )
    .unwrap();
    writeln!(out).unwrap();

    writeln!(out, "declare void @mp_rt_gpu_register_kernels(ptr, i32)").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "define void @mp_gpu_register_all_kernels() {{").unwrap();
    writeln!(out, "entry:").unwrap();
    if !kernels.is_empty() {
        writeln!(
            out,
            "  call void @mp_rt_gpu_register_kernels(ptr getelementptr inbounds ([{} x %MpRtGpuKernelEntry], ptr @mp_gpu_kernel_registry, i64 0, i64 0), i32 {})",
            kernels.len(),
            kernels.len()
        )
        .unwrap();
    }
    writeln!(out, "  ret void").unwrap();
    writeln!(out, "}}").unwrap();

    out
}

fn sanitize_spirv_entry_name(name: &str) -> String {
    let mut out = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push_str("kernel");
    }
    out
}

fn sanitize_llvm_symbol(symbol_name: &str) -> String {
    let raw = symbol_name.strip_prefix('@').unwrap_or(symbol_name);
    let mut out = String::new();
    for (idx, ch) in raw.chars().enumerate() {
        let ok = ch.is_ascii_alphanumeric() || ch == '_' || ch == '$' || ch == '.';
        if ok {
            if idx == 0 && ch.is_ascii_digit() {
                out.push('_');
            }
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push_str("mp_gpu_spv_blob");
    }
    out
}

fn llvm_kernel_param_const(param: &KernelParam) -> String {
    format!(
        "%MpRtGpuParam {{ i8 {}, i32 {}, i32 {}, i32 {} }}",
        param.kind as u8, param.type_id, param.offset_or_binding, param.size
    )
}
