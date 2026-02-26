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

#[derive(Debug, Default)]
struct SpirvBuilder {
    words: Vec<u32>,
    next_id: u32,
}

#[derive(Debug, Default)]
struct SpirvTypeIds {
    void_ty: u32,
    void_fn_ty: u32,
    runtime_array_u32_ty: Option<u32>,
    storage_buffer_struct_ty: Option<u32>,
    buffer_vars: Vec<u32>,
    push_constant_struct_ty: Option<u32>,
    push_constant_var: Option<u32>,
    interface_vars: Vec<u32>,
}

impl SpirvBuilder {
    const SPIRV_MAGIC: u32 = 0x0723_0203;
    const SPIRV_VERSION_1_0: u32 = 0x0001_0000;
    const CAPABILITY_SHADER: u32 = 1;
    const ADDRESSING_MODEL_LOGICAL: u32 = 0;
    const MEMORY_MODEL_GLSL450: u32 = 1;
    const EXEC_MODEL_GL_COMPUTE: u32 = 5;
    const EXEC_MODE_LOCAL_SIZE: u32 = 17;
    const STORAGE_CLASS_PUSH_CONSTANT: u32 = 9;
    const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;
    const DECORATION_BLOCK: u32 = 2;
    const DECORATION_ARRAY_STRIDE: u32 = 6;
    const DECORATION_BINDING: u32 = 33;
    const DECORATION_DESCRIPTOR_SET: u32 = 34;
    const DECORATION_OFFSET: u32 = 35;

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
        self.emit_inst(17, &[capability]); // OpCapability
    }

    fn emit_memory_model(&mut self, addressing_model: u32, memory_model: u32) {
        self.emit_inst(14, &[addressing_model, memory_model]); // OpMemoryModel
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
        self.emit_inst(15, &ops); // OpEntryPoint
    }

    fn emit_execution_mode(&mut self, fn_id: u32, mode: u32, literals: &[u32]) {
        let mut ops = vec![fn_id, mode];
        ops.extend_from_slice(literals);
        self.emit_inst(16, &ops); // OpExecutionMode
    }

    fn emit_decorations(&mut self, layout: &KernelLayout, ids: &SpirvTypeIds) {
        if let Some(runtime_arr) = ids.runtime_array_u32_ty {
            self.emit_inst(71, &[runtime_arr, Self::DECORATION_ARRAY_STRIDE, 4]); // OpDecorate
        }
        if let Some(buf_struct) = ids.storage_buffer_struct_ty {
            self.emit_inst(71, &[buf_struct, Self::DECORATION_BLOCK]); // OpDecorate
        }

        let mut buffer_bindings = layout
            .params
            .iter()
            .filter_map(|p| (p.kind == KernelParamKind::Buffer).then_some(p.offset_or_binding));
        for var_id in &ids.buffer_vars {
            let binding = buffer_bindings.next().unwrap_or(0);
            self.emit_inst(71, &[*var_id, Self::DECORATION_DESCRIPTOR_SET, 0]); // OpDecorate
            self.emit_inst(71, &[*var_id, Self::DECORATION_BINDING, binding]); // OpDecorate
        }

        if let Some(push_struct) = ids.push_constant_struct_ty {
            self.emit_inst(71, &[push_struct, Self::DECORATION_BLOCK]); // OpDecorate
            self.emit_inst(72, &[push_struct, 0, Self::DECORATION_OFFSET, 0]); // OpMemberDecorate
        }
    }

    fn emit_types(&mut self, layout: &KernelLayout) -> SpirvTypeIds {
        let mut ids = SpirvTypeIds::default();

        ids.void_ty = self.new_id();
        self.emit_inst(19, &[ids.void_ty]); // OpTypeVoid

        let u8_ty = self.new_id();
        self.emit_inst(21, &[u8_ty, 8, 0]); // OpTypeInt

        let u32_ty = self.new_id();
        self.emit_inst(21, &[u32_ty, 32, 0]); // OpTypeInt

        ids.void_fn_ty = self.new_id();
        self.emit_inst(33, &[ids.void_fn_ty, ids.void_ty]); // OpTypeFunction

        if layout.num_buffers > 0 {
            let runtime_arr_u32_ty = self.new_id();
            self.emit_inst(29, &[runtime_arr_u32_ty, u32_ty]); // OpTypeRuntimeArray

            let storage_buffer_struct_ty = self.new_id();
            self.emit_inst(30, &[storage_buffer_struct_ty, runtime_arr_u32_ty]); // OpTypeStruct

            let storage_buffer_ptr_ty = self.new_id();
            self.emit_inst(
                32,
                &[
                    storage_buffer_ptr_ty,
                    Self::STORAGE_CLASS_STORAGE_BUFFER,
                    storage_buffer_struct_ty,
                ],
            ); // OpTypePointer

            for _ in 0..layout.num_buffers {
                let var_id = self.new_id();
                self.emit_inst(
                    59,
                    &[storage_buffer_ptr_ty, var_id, Self::STORAGE_CLASS_STORAGE_BUFFER],
                ); // OpVariable
                ids.buffer_vars.push(var_id);
                ids.interface_vars.push(var_id);
            }

            ids.runtime_array_u32_ty = Some(runtime_arr_u32_ty);
            ids.storage_buffer_struct_ty = Some(storage_buffer_struct_ty);
        }

        if layout.push_const_size > 0 {
            let push_size_const = self.new_id();
            self.emit_inst(43, &[u32_ty, push_size_const, layout.push_const_size]); // OpConstant

            let push_const_arr_ty = self.new_id();
            self.emit_inst(28, &[push_const_arr_ty, u8_ty, push_size_const]); // OpTypeArray

            let push_const_struct_ty = self.new_id();
            self.emit_inst(30, &[push_const_struct_ty, push_const_arr_ty]); // OpTypeStruct

            let push_const_ptr_ty = self.new_id();
            self.emit_inst(
                32,
                &[
                    push_const_ptr_ty,
                    Self::STORAGE_CLASS_PUSH_CONSTANT,
                    push_const_struct_ty,
                ],
            ); // OpTypePointer

            let push_const_var = self.new_id();
            self.emit_inst(
                59,
                &[push_const_ptr_ty, push_const_var, Self::STORAGE_CLASS_PUSH_CONSTANT],
            ); // OpVariable

            ids.push_constant_struct_ty = Some(push_const_struct_ty);
            ids.push_constant_var = Some(push_const_var);
            ids.interface_vars.push(push_const_var);
        }

        ids
    }

    fn emit_function(&mut self, function_id: u32, ids: &SpirvTypeIds) {
        self.emit_inst(54, &[ids.void_ty, function_id, 0, ids.void_fn_ty]); // OpFunction
        let label = self.new_id();
        self.emit_inst(248, &[label]); // OpLabel
        self.emit_inst(253, &[]); // OpReturn
        self.emit_inst(56, &[]); // OpFunctionEnd
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

pub fn generate_spirv(
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

    let mut builder = SpirvBuilder::new();
    builder.emit_header(SpirvBuilder::SPIRV_VERSION_1_0, 0);
    builder.emit_capability(SpirvBuilder::CAPABILITY_SHADER);
    builder.emit_memory_model(
        SpirvBuilder::ADDRESSING_MODEL_LOGICAL,
        SpirvBuilder::MEMORY_MODEL_GLSL450,
    );

    let ids = builder.emit_types(layout);
    let entry_fn = builder.new_id();
    builder.emit_entry_point(
        SpirvBuilder::EXEC_MODEL_GL_COMPUTE,
        entry_fn,
        &sanitize_spirv_entry_name(&func.name),
        &ids.interface_vars,
    );
    builder.emit_execution_mode(entry_fn, SpirvBuilder::EXEC_MODE_LOCAL_SIZE, &[1, 1, 1]);
    builder.emit_decorations(layout, &ids);
    builder.emit_function(entry_fn, &ids);

    Ok(builder.finalize())
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

    writeln!(out, "define internal void @mp_gpu_register_all_kernels() {{").unwrap();
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
        param.kind as u8,
        param.type_id,
        param.offset_or_binding,
        param.size
    )
}
