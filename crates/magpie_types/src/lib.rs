//! Magpie type system: TypeKind, HeapBase, type interning, TypeId assignment (§8, §16.2-16.3).

use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub struct PackageId(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub struct ModuleId(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub struct DefId(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub struct TypeId(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub struct InstId(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub struct FnId(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub struct GlobalId(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub struct LocalId(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub struct BlockId(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum PrimType {
    I1,
    I8,
    I16,
    I32,
    I64,
    I128,
    U1,
    U8,
    U16,
    U32,
    U64,
    U128,
    F16,
    F32,
    F64,
    Bool, // alias for I1
    Unit,
}

impl PrimType {
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            Self::I1
                | Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
                | Self::I128
                | Self::U1
                | Self::U8
                | Self::U16
                | Self::U32
                | Self::U64
                | Self::U128
                | Self::Bool
        )
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Self::F16 | Self::F32 | Self::F64)
    }

    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            Self::I1 | Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::I128
        )
    }

    pub fn bit_width(&self) -> u32 {
        match self {
            Self::I1 | Self::U1 | Self::Bool => 1,
            Self::I8 | Self::U8 => 8,
            Self::I16 | Self::U16 | Self::F16 => 16,
            Self::I32 | Self::U32 | Self::F32 => 32,
            Self::I64 | Self::U64 | Self::F64 => 64,
            Self::I128 | Self::U128 => 128,
            Self::Unit => 0,
        }
    }
}

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct Sid(pub String);

impl Sid {
    pub fn is_valid(&self) -> bool {
        self.0.len() == 12
            && matches!(self.0.as_bytes()[0], b'M' | b'F' | b'T' | b'G' | b'E')
            && self.0.as_bytes()[1] == b':'
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum HandleKind {
    Unique,
    Shared,
    Borrow,
    MutBorrow,
    Weak,
}

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub enum TypeKind {
    Prim(PrimType),
    HeapHandle { hk: HandleKind, base: HeapBase },
    BuiltinOption { inner: TypeId },
    BuiltinResult { ok: TypeId, err: TypeId },
    RawPtr { to: TypeId },
    Arr { n: u32, elem: TypeId },
    Vec { n: u32, elem: TypeId },
    Tuple { elems: Vec<TypeId> },
    ValueStruct { sid: Sid },
}

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub enum HeapBase {
    BuiltinStr,
    BuiltinArray { elem: TypeId },
    BuiltinMap { key: TypeId, val: TypeId },
    BuiltinStrBuilder,
    BuiltinMutex { inner: TypeId },
    BuiltinRwLock { inner: TypeId },
    BuiltinCell { inner: TypeId },
    BuiltinFuture { result: TypeId },
    BuiltinChannelSend { elem: TypeId },
    BuiltinChannelRecv { elem: TypeId },
    Callable { sig_sid: Sid },
    UserType { type_sid: Sid, targs: Vec<TypeId> },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TypeLayout {
    pub size: u64,
    pub align: u64,
    pub fields: Vec<(String, u64)>, // (field_name, byte_offset)
}

/// Fixed type_id table (§20.1.4)
pub mod fixed_type_ids {
    use super::TypeId;
    pub const UNIT: TypeId = TypeId(0);
    pub const BOOL: TypeId = TypeId(1);
    pub const I8: TypeId = TypeId(2);
    pub const I16: TypeId = TypeId(3);
    pub const I32: TypeId = TypeId(4);
    pub const I64: TypeId = TypeId(5);
    pub const I128: TypeId = TypeId(6);
    pub const U8: TypeId = TypeId(7);
    pub const U16: TypeId = TypeId(8);
    pub const U32: TypeId = TypeId(9);
    pub const U64: TypeId = TypeId(10);
    pub const U128: TypeId = TypeId(11);
    pub const U1: TypeId = TypeId(12);
    pub const F16: TypeId = TypeId(13);
    pub const F32: TypeId = TypeId(14);
    pub const F64: TypeId = TypeId(15);
    pub const STR: TypeId = TypeId(20);
    pub const STR_BUILDER: TypeId = TypeId(21);
    pub const ARRAY_BASE: TypeId = TypeId(22);
    pub const MAP_BASE: TypeId = TypeId(23);
    pub const TOPTION_BASE: TypeId = TypeId(24);
    pub const TRESULT_BASE: TypeId = TypeId(25);
    pub const TCALLABLE_BASE: TypeId = TypeId(26);
    pub const GPU_DEVICE: TypeId = TypeId(30);
    pub const GPU_BUFFER_BASE: TypeId = TypeId(31);
    pub const GPU_FENCE: TypeId = TypeId(32);
    pub const USER_TYPE_START: TypeId = TypeId(1000);
}

type ValueFieldDef = (String, TypeId);
type ValueStructFieldDefs = Vec<ValueFieldDef>;
type ValueEnumVariantDef = (String, ValueStructFieldDefs);
type ValueEnumVariantDefs = Vec<ValueEnumVariantDef>;

/// Type context for interning and layout computation.
#[derive(Debug, Default)]
pub struct TypeCtx {
    pub types: Vec<(TypeId, TypeKind)>,
    next_user_id: u32,
    type_fqns: HashMap<Sid, String>,
    value_struct_fields: HashMap<Sid, ValueStructFieldDefs>,
    value_enum_variants: HashMap<Sid, ValueEnumVariantDefs>,
    layout_cache: RefCell<HashMap<TypeId, TypeLayout>>,
}

impl TypeCtx {
    pub fn new() -> Self {
        let mut ctx = Self {
            types: Vec::with_capacity(64),
            next_user_id: fixed_type_ids::USER_TYPE_START.0,
            type_fqns: HashMap::new(),
            value_struct_fields: HashMap::new(),
            value_enum_variants: HashMap::new(),
            layout_cache: RefCell::new(HashMap::new()),
        };
        ctx.populate_fixed_types();
        ctx
    }

    fn populate_fixed_types(&mut self) {
        use fixed_type_ids::*;
        let fixed = [
            (UNIT, TypeKind::Prim(PrimType::Unit)),
            (BOOL, TypeKind::Prim(PrimType::Bool)),
            (I8, TypeKind::Prim(PrimType::I8)),
            (I16, TypeKind::Prim(PrimType::I16)),
            (I32, TypeKind::Prim(PrimType::I32)),
            (I64, TypeKind::Prim(PrimType::I64)),
            (I128, TypeKind::Prim(PrimType::I128)),
            (U8, TypeKind::Prim(PrimType::U8)),
            (U16, TypeKind::Prim(PrimType::U16)),
            (U32, TypeKind::Prim(PrimType::U32)),
            (U64, TypeKind::Prim(PrimType::U64)),
            (U128, TypeKind::Prim(PrimType::U128)),
            (U1, TypeKind::Prim(PrimType::U1)),
            (F16, TypeKind::Prim(PrimType::F16)),
            (F32, TypeKind::Prim(PrimType::F32)),
            (F64, TypeKind::Prim(PrimType::F64)),
            (
                STR,
                TypeKind::HeapHandle {
                    hk: HandleKind::Unique,
                    base: HeapBase::BuiltinStr,
                },
            ),
            (
                STR_BUILDER,
                TypeKind::HeapHandle {
                    hk: HandleKind::Unique,
                    base: HeapBase::BuiltinStrBuilder,
                },
            ),
        ];
        for (id, kind) in fixed {
            self.types.push((id, kind));
        }
    }

    pub fn intern(&mut self, kind: TypeKind) -> TypeId {
        // Fast path: check if already interned
        if let Some((id, _)) = self.types.iter().find(|(_, k)| k == &kind) {
            return *id;
        }
        let id = TypeId(self.next_user_id);
        self.next_user_id += 1;
        self.types.push((id, kind));
        self.layout_cache.borrow_mut().clear();
        id
    }

    pub fn lookup(&self, id: TypeId) -> Option<&TypeKind> {
        // Fast path for fixed IDs
        if id.0 < 100 {
            return self.types.iter().find(|(i, _)| i == &id).map(|(_, k)| k);
        }
        self.types.iter().find(|(i, _)| i == &id).map(|(_, k)| k)
    }

    pub fn lookup_by_prim(&self, prim: PrimType) -> TypeId {
        match prim {
            PrimType::Unit => fixed_type_ids::UNIT,
            PrimType::Bool | PrimType::I1 => fixed_type_ids::BOOL,
            PrimType::I8 => fixed_type_ids::I8,
            PrimType::I16 => fixed_type_ids::I16,
            PrimType::I32 => fixed_type_ids::I32,
            PrimType::I64 => fixed_type_ids::I64,
            PrimType::I128 => fixed_type_ids::I128,
            PrimType::U1 => fixed_type_ids::U1,
            PrimType::U8 => fixed_type_ids::U8,
            PrimType::U16 => fixed_type_ids::U16,
            PrimType::U32 => fixed_type_ids::U32,
            PrimType::U64 => fixed_type_ids::U64,
            PrimType::U128 => fixed_type_ids::U128,
            PrimType::F16 => fixed_type_ids::F16,
            PrimType::F32 => fixed_type_ids::F32,
            PrimType::F64 => fixed_type_ids::F64,
        }
    }

    pub fn register_type_fqn(&mut self, sid: Sid, fqn: impl Into<String>) {
        self.type_fqns.insert(sid, fqn.into());
    }

    pub fn register_value_struct_fields(&mut self, sid: Sid, fields: ValueStructFieldDefs) {
        self.value_struct_fields.insert(sid, fields);
        self.layout_cache.borrow_mut().clear();
    }

    pub fn register_value_enum_variants(&mut self, sid: Sid, variants: ValueEnumVariantDefs) {
        self.value_enum_variants.insert(sid, variants);
        self.layout_cache.borrow_mut().clear();
    }

    pub fn user_struct_fields(&self, sid: &Sid) -> Option<&ValueStructFieldDefs> {
        self.value_struct_fields.get(sid)
    }

    pub fn user_enum_variants(&self, sid: &Sid) -> Option<&ValueEnumVariantDefs> {
        self.value_enum_variants.get(sid)
    }

    pub fn user_struct_layout(&self, sid: &Sid) -> Option<TypeLayout> {
        let fields = self.value_struct_fields.get(sid)?;
        Some(self.compute_field_layout(fields))
    }

    pub fn user_enum_layout(&self, sid: &Sid) -> Option<TypeLayout> {
        let variants = self.value_enum_variants.get(sid)?;
        Some(self.compute_enum_layout_public(variants))
    }

    pub fn user_struct_field(&self, sid: &Sid, field: &str) -> Option<(TypeId, u64)> {
        let fields = self.value_struct_fields.get(sid)?;
        let mut offset = 0_u64;
        for (name, ty) in fields {
            let ty_layout = self.compute_layout(*ty);
            let align = ty_layout.align.max(1);
            offset = align_to(offset, align);
            if name == field {
                return Some((*ty, offset));
            }
            offset = offset.saturating_add(ty_layout.size);
        }
        None
    }

    pub fn user_enum_variant_tag(&self, sid: &Sid, variant: &str) -> Option<i32> {
        let variants = self.value_enum_variants.get(sid)?;
        variants
            .iter()
            .position(|(name, _)| name == variant)
            .map(|idx| idx as i32)
    }

    pub fn user_enum_variant_field(
        &self,
        sid: &Sid,
        variant: &str,
        field: &str,
    ) -> Option<(TypeId, u64)> {
        let variants = self.value_enum_variants.get(sid)?;
        let (_, fields) = variants.iter().find(|(name, _)| name == variant)?;
        let mut offset = 0_u64;
        for (name, ty) in fields {
            let ty_layout = self.compute_layout(*ty);
            let align = ty_layout.align.max(1);
            offset = align_to(offset, align);
            if name == field {
                return Some((*ty, offset));
            }
            offset = offset.saturating_add(ty_layout.size);
        }
        None
    }

    pub fn user_enum_payload_offset(&self, sid: &Sid) -> Option<u64> {
        self.user_enum_layout(sid)?
            .fields
            .iter()
            .find(|(name, _)| name == "payload")
            .map(|(_, offset)| *offset)
    }

    pub fn type_str(&self, type_id: TypeId) -> String {
        self.lookup(type_id)
            .map(|k| self.type_str_kind(k))
            .unwrap_or_else(|| format!("type#{}", type_id.0))
    }

    pub fn compute_layout(&self, type_id: TypeId) -> TypeLayout {
        if let Some(layout) = self.layout_cache.borrow().get(&type_id).cloned() {
            return layout;
        }
        let mut visiting = HashSet::new();
        self.compute_layout_inner(type_id, &mut visiting)
    }

    pub fn finalize_type_ids(&mut self) {
        let _ = self.finalize_type_ids_with_remap();
    }

    pub fn finalize_type_ids_with_remap(&mut self) -> HashMap<TypeId, TypeId> {
        let mut user_entries = self
            .types
            .iter()
            .filter(|(id, _)| id.0 >= fixed_type_ids::USER_TYPE_START.0)
            .map(|(id, _)| (*id, self.type_str(*id)))
            .collect::<Vec<_>>();
        user_entries.sort_by(|(lhs_id, lhs_key), (rhs_id, rhs_key)| {
            lhs_key.cmp(rhs_key).then(lhs_id.0.cmp(&rhs_id.0))
        });

        let mut remap: HashMap<TypeId, TypeId> = HashMap::with_capacity(user_entries.len());
        let mut changed_remap: HashMap<TypeId, TypeId> = HashMap::new();
        let mut next = fixed_type_ids::USER_TYPE_START.0;
        for (old_id, _) in user_entries {
            let new_id = TypeId(next);
            remap.insert(old_id, new_id);
            if old_id != new_id {
                changed_remap.insert(old_id, new_id);
            }
            next += 1;
        }

        if remap.is_empty() {
            self.next_user_id = fixed_type_ids::USER_TYPE_START.0;
            return changed_remap;
        }

        let mut rewritten = Vec::with_capacity(self.types.len());
        for (old_id, kind) in &self.types {
            let new_id = Self::remap_type_id(*old_id, &remap);
            let new_kind = Self::remap_type_kind(kind, &remap);
            rewritten.push((new_id, new_kind));
        }
        rewritten.sort_by_key(|(id, _)| id.0);
        self.types = rewritten;

        for fields in self.value_struct_fields.values_mut() {
            for (_, ty) in fields {
                *ty = Self::remap_type_id(*ty, &remap);
            }
        }
        for variants in self.value_enum_variants.values_mut() {
            for (_, fields) in variants {
                for (_, ty) in fields {
                    *ty = Self::remap_type_id(*ty, &remap);
                }
            }
        }

        self.next_user_id = self
            .types
            .iter()
            .map(|(id, _)| id.0)
            .max()
            .map_or(fixed_type_ids::USER_TYPE_START.0, |max_id| {
                max_id.saturating_add(1)
            })
            .max(fixed_type_ids::USER_TYPE_START.0);

        self.layout_cache.borrow_mut().clear();
        changed_remap
    }

    fn compute_layout_inner(&self, type_id: TypeId, visiting: &mut HashSet<TypeId>) -> TypeLayout {
        if let Some(layout) = self.layout_cache.borrow().get(&type_id).cloned() {
            return layout;
        }
        if !visiting.insert(type_id) {
            return TypeLayout {
                size: 0,
                align: 1,
                fields: Vec::new(),
            };
        }

        let layout = match self.lookup(type_id) {
            Some(TypeKind::Prim(p)) => self.prim_layout(*p),
            Some(TypeKind::HeapHandle { .. }) | Some(TypeKind::RawPtr { .. }) => TypeLayout {
                size: 8,
                align: 8,
                fields: Vec::new(),
            },
            Some(TypeKind::BuiltinOption { inner }) => self.compute_enum_layout(
                &[
                    ("None".to_string(), Vec::new()),
                    ("Some".to_string(), vec![("value".to_string(), *inner)]),
                ],
                visiting,
            ),
            Some(TypeKind::BuiltinResult { ok, err }) => self.compute_enum_layout(
                &[
                    ("Ok".to_string(), vec![("value".to_string(), *ok)]),
                    ("Err".to_string(), vec![("error".to_string(), *err)]),
                ],
                visiting,
            ),
            Some(TypeKind::Arr { n, elem }) | Some(TypeKind::Vec { n, elem }) => {
                let elem_layout = self.compute_layout_inner(*elem, visiting);
                TypeLayout {
                    size: elem_layout.size.saturating_mul(u64::from(*n)),
                    align: elem_layout.align.max(1),
                    fields: Vec::new(),
                }
            }
            Some(TypeKind::Tuple { elems }) => {
                let fields = elems
                    .iter()
                    .enumerate()
                    .map(|(idx, ty)| (idx.to_string(), *ty))
                    .collect::<Vec<_>>();
                self.compute_struct_layout(&fields, visiting)
            }
            Some(TypeKind::ValueStruct { sid }) => {
                if let Some(variants) = self.value_enum_variants.get(sid) {
                    self.compute_enum_layout(variants, visiting)
                } else if let Some(fields) = self.value_struct_fields.get(sid) {
                    self.compute_struct_layout(fields, visiting)
                } else {
                    TypeLayout {
                        size: 0,
                        align: 1,
                        fields: Vec::new(),
                    }
                }
            }
            None => TypeLayout {
                size: 0,
                align: 1,
                fields: Vec::new(),
            },
        };

        visiting.remove(&type_id);
        self.layout_cache
            .borrow_mut()
            .insert(type_id, layout.clone());
        layout
    }

    fn compute_field_layout(&self, fields: &[(String, TypeId)]) -> TypeLayout {
        let mut align = 1_u64;
        let mut offset = 0_u64;
        let mut field_offsets = Vec::with_capacity(fields.len());

        for (name, ty) in fields {
            let field_layout = self.compute_layout(*ty);
            let field_align = field_layout.align.max(1);
            offset = align_to(offset, field_align);
            field_offsets.push((name.clone(), offset));
            offset = offset.saturating_add(field_layout.size);
            align = align.max(field_align);
        }

        TypeLayout {
            size: align_to(offset, align),
            align,
            fields: field_offsets,
        }
    }

    fn compute_enum_layout_public(
        &self,
        variants: &[(String, Vec<(String, TypeId)>)],
    ) -> TypeLayout {
        let mut payload_size = 0_u64;
        let mut payload_align = 1_u64;

        for (_, fields) in variants {
            let payload_layout = self.compute_field_layout(fields);
            payload_size = payload_size.max(payload_layout.size);
            payload_align = payload_align.max(payload_layout.align);
        }

        let tag_size = 4_u64;
        let tag_align = 4_u64;
        let payload_offset = align_to(tag_size, payload_align);
        let align = tag_align.max(payload_align);
        let size = align_to(payload_offset.saturating_add(payload_size), align);
        let mut fields = vec![("tag".to_string(), 0)];
        if payload_size > 0 {
            fields.push(("payload".to_string(), payload_offset));
        }

        TypeLayout {
            size,
            align,
            fields,
        }
    }

    fn compute_struct_layout(
        &self,
        fields: &[(String, TypeId)],
        visiting: &mut HashSet<TypeId>,
    ) -> TypeLayout {
        let mut align = 1_u64;
        let mut offset = 0_u64;
        let mut field_offsets = Vec::with_capacity(fields.len());

        for (name, ty) in fields {
            let field_layout = self.compute_layout_inner(*ty, visiting);
            let field_align = field_layout.align.max(1);
            offset = align_to(offset, field_align);
            field_offsets.push((name.clone(), offset));
            offset = offset.saturating_add(field_layout.size);
            align = align.max(field_align);
        }

        TypeLayout {
            size: align_to(offset, align),
            align,
            fields: field_offsets,
        }
    }

    fn compute_enum_layout(
        &self,
        variants: &[(String, Vec<(String, TypeId)>)],
        visiting: &mut HashSet<TypeId>,
    ) -> TypeLayout {
        let mut payload_size = 0_u64;
        let mut payload_align = 1_u64;

        for (_, fields) in variants {
            let payload_layout = self.compute_struct_layout(fields, visiting);
            payload_size = payload_size.max(payload_layout.size);
            payload_align = payload_align.max(payload_layout.align);
        }

        let tag_size = 4_u64;
        let tag_align = 4_u64;
        let payload_offset = align_to(tag_size, payload_align);
        let align = tag_align.max(payload_align);
        let size = align_to(payload_offset.saturating_add(payload_size), align);
        let mut fields = vec![("tag".to_string(), 0)];
        if payload_size > 0 {
            fields.push(("payload".to_string(), payload_offset));
        }

        TypeLayout {
            size,
            align,
            fields,
        }
    }

    fn prim_layout(&self, prim: PrimType) -> TypeLayout {
        let (size, align) = match prim {
            PrimType::Unit => (0, 1),
            PrimType::Bool | PrimType::I1 | PrimType::U1 => (1, 1),
            PrimType::I8 | PrimType::U8 => (1, 1),
            PrimType::I16 | PrimType::U16 | PrimType::F16 => (2, 2),
            PrimType::I32 | PrimType::U32 | PrimType::F32 => (4, 4),
            PrimType::I64 | PrimType::U64 | PrimType::F64 => (8, 8),
            PrimType::I128 | PrimType::U128 => (16, 16),
        };
        TypeLayout {
            size,
            align,
            fields: Vec::new(),
        }
    }

    fn type_str_kind(&self, kind: &TypeKind) -> String {
        match kind {
            TypeKind::Prim(p) => self.prim_type_str(*p).to_string(),
            TypeKind::HeapHandle { hk, base } => {
                let base_s = self.heap_base_str(base);
                match hk {
                    HandleKind::Unique => base_s,
                    HandleKind::Shared => format!("shared {}", base_s),
                    HandleKind::Borrow => format!("borrow {}", base_s),
                    HandleKind::MutBorrow => format!("mutborrow {}", base_s),
                    HandleKind::Weak => format!("weak {}", base_s),
                }
            }
            TypeKind::BuiltinOption { inner } => format!("TOption<{}>", self.type_str(*inner)),
            TypeKind::BuiltinResult { ok, err } => {
                format!("TResult<{},{}>", self.type_str(*ok), self.type_str(*err))
            }
            TypeKind::RawPtr { to } => format!("rawptr<{}>", self.type_str(*to)),
            TypeKind::Arr { n, elem } => format!("arr<{},{}>", n, self.type_str(*elem)),
            TypeKind::Vec { n, elem } => format!("vec<{},{}>", n, self.type_str(*elem)),
            TypeKind::Tuple { elems } => {
                let joined = elems
                    .iter()
                    .map(|ty| self.type_str(*ty))
                    .collect::<Vec<_>>()
                    .join(",");
                format!("tuple<{}>", joined)
            }
            TypeKind::ValueStruct { sid } => self.user_type_name(sid),
        }
    }

    fn heap_base_str(&self, base: &HeapBase) -> String {
        match base {
            HeapBase::BuiltinStr => "Str".to_string(),
            HeapBase::BuiltinArray { elem } => format!("Array<{}>", self.type_str(*elem)),
            HeapBase::BuiltinMap { key, val } => {
                format!("Map<{},{}>", self.type_str(*key), self.type_str(*val))
            }
            HeapBase::BuiltinStrBuilder => "TStrBuilder".to_string(),
            HeapBase::BuiltinMutex { inner } => format!("TMutex<{}>", self.type_str(*inner)),
            HeapBase::BuiltinRwLock { inner } => format!("TRwLock<{}>", self.type_str(*inner)),
            HeapBase::BuiltinCell { inner } => format!("TCell<{}>", self.type_str(*inner)),
            HeapBase::BuiltinFuture { result } => format!("TFuture<{}>", self.type_str(*result)),
            HeapBase::BuiltinChannelSend { elem } => {
                format!("TChannelSend<{}>", self.type_str(*elem))
            }
            HeapBase::BuiltinChannelRecv { elem } => {
                format!("TChannelRecv<{}>", self.type_str(*elem))
            }
            HeapBase::Callable { sig_sid } => format!("TCallable<{}>", sig_sid.0),
            HeapBase::UserType { type_sid, targs } => {
                let base = self.user_type_name(type_sid);
                if targs.is_empty() {
                    base
                } else {
                    let joined = targs
                        .iter()
                        .map(|ty| self.type_str(*ty))
                        .collect::<Vec<_>>()
                        .join(",");
                    format!("{}<{}>", base, joined)
                }
            }
        }
    }

    fn user_type_name(&self, sid: &Sid) -> String {
        self.type_fqns
            .get(sid)
            .cloned()
            .unwrap_or_else(|| sid.0.clone())
    }

    fn prim_type_str(&self, prim: PrimType) -> &'static str {
        match prim {
            PrimType::I1 => "i1",
            PrimType::I8 => "i8",
            PrimType::I16 => "i16",
            PrimType::I32 => "i32",
            PrimType::I64 => "i64",
            PrimType::I128 => "i128",
            PrimType::U1 => "u1",
            PrimType::U8 => "u8",
            PrimType::U16 => "u16",
            PrimType::U32 => "u32",
            PrimType::U64 => "u64",
            PrimType::U128 => "u128",
            PrimType::F16 => "f16",
            PrimType::F32 => "f32",
            PrimType::F64 => "f64",
            PrimType::Bool => "bool",
            PrimType::Unit => "unit",
        }
    }

    fn remap_type_id(id: TypeId, remap: &HashMap<TypeId, TypeId>) -> TypeId {
        remap.get(&id).copied().unwrap_or(id)
    }

    fn remap_type_kind(kind: &TypeKind, remap: &HashMap<TypeId, TypeId>) -> TypeKind {
        match kind {
            TypeKind::Prim(p) => TypeKind::Prim(*p),
            TypeKind::HeapHandle { hk, base } => TypeKind::HeapHandle {
                hk: *hk,
                base: Self::remap_heap_base(base, remap),
            },
            TypeKind::BuiltinOption { inner } => TypeKind::BuiltinOption {
                inner: Self::remap_type_id(*inner, remap),
            },
            TypeKind::BuiltinResult { ok, err } => TypeKind::BuiltinResult {
                ok: Self::remap_type_id(*ok, remap),
                err: Self::remap_type_id(*err, remap),
            },
            TypeKind::RawPtr { to } => TypeKind::RawPtr {
                to: Self::remap_type_id(*to, remap),
            },
            TypeKind::Arr { n, elem } => TypeKind::Arr {
                n: *n,
                elem: Self::remap_type_id(*elem, remap),
            },
            TypeKind::Vec { n, elem } => TypeKind::Vec {
                n: *n,
                elem: Self::remap_type_id(*elem, remap),
            },
            TypeKind::Tuple { elems } => TypeKind::Tuple {
                elems: elems
                    .iter()
                    .map(|ty| Self::remap_type_id(*ty, remap))
                    .collect(),
            },
            TypeKind::ValueStruct { sid } => TypeKind::ValueStruct { sid: sid.clone() },
        }
    }

    fn remap_heap_base(base: &HeapBase, remap: &HashMap<TypeId, TypeId>) -> HeapBase {
        match base {
            HeapBase::BuiltinStr => HeapBase::BuiltinStr,
            HeapBase::BuiltinArray { elem } => HeapBase::BuiltinArray {
                elem: Self::remap_type_id(*elem, remap),
            },
            HeapBase::BuiltinMap { key, val } => HeapBase::BuiltinMap {
                key: Self::remap_type_id(*key, remap),
                val: Self::remap_type_id(*val, remap),
            },
            HeapBase::BuiltinStrBuilder => HeapBase::BuiltinStrBuilder,
            HeapBase::BuiltinMutex { inner } => HeapBase::BuiltinMutex {
                inner: Self::remap_type_id(*inner, remap),
            },
            HeapBase::BuiltinRwLock { inner } => HeapBase::BuiltinRwLock {
                inner: Self::remap_type_id(*inner, remap),
            },
            HeapBase::BuiltinCell { inner } => HeapBase::BuiltinCell {
                inner: Self::remap_type_id(*inner, remap),
            },
            HeapBase::BuiltinFuture { result } => HeapBase::BuiltinFuture {
                result: Self::remap_type_id(*result, remap),
            },
            HeapBase::BuiltinChannelSend { elem } => HeapBase::BuiltinChannelSend {
                elem: Self::remap_type_id(*elem, remap),
            },
            HeapBase::BuiltinChannelRecv { elem } => HeapBase::BuiltinChannelRecv {
                elem: Self::remap_type_id(*elem, remap),
            },
            HeapBase::Callable { sig_sid } => HeapBase::Callable {
                sig_sid: sig_sid.clone(),
            },
            HeapBase::UserType { type_sid, targs } => HeapBase::UserType {
                type_sid: type_sid.clone(),
                targs: targs
                    .iter()
                    .map(|ty| Self::remap_type_id(*ty, remap))
                    .collect(),
            },
        }
    }
}

fn align_to(value: u64, align: u64) -> u64 {
    if align <= 1 {
        value
    } else {
        let rem = value % align;
        if rem == 0 {
            value
        } else {
            value + (align - rem)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_for_primitives() {
        let ctx = TypeCtx::new();
        assert_eq!(
            ctx.compute_layout(fixed_type_ids::BOOL),
            TypeLayout {
                size: 1,
                align: 1,
                fields: Vec::new()
            }
        );
        assert_eq!(
            ctx.compute_layout(fixed_type_ids::I32),
            TypeLayout {
                size: 4,
                align: 4,
                fields: Vec::new()
            }
        );
        assert_eq!(
            ctx.compute_layout(fixed_type_ids::F64),
            TypeLayout {
                size: 8,
                align: 8,
                fields: Vec::new()
            }
        );
    }

    #[test]
    fn layout_for_simple_value_struct() {
        let mut ctx = TypeCtx::new();
        let sid = Sid("T:VALUE00001".to_string());
        let ty = ctx.intern(TypeKind::ValueStruct { sid: sid.clone() });
        ctx.register_value_struct_fields(
            sid,
            vec![
                ("age".to_string(), fixed_type_ids::I32),
                ("score".to_string(), fixed_type_ids::I64),
            ],
        );

        let layout = ctx.compute_layout(ty);
        assert_eq!(layout.size, 16);
        assert_eq!(layout.align, 8);
        assert_eq!(
            layout.fields,
            vec![("age".to_string(), 0), ("score".to_string(), 8)]
        );
    }

    #[test]
    fn finalize_type_ids_sorts_by_fqn() {
        let mut ctx = TypeCtx::new();
        let sid_z = Sid("T:ZZZZZZZZZZ".to_string());
        let sid_a = Sid("T:AAAAAAAAAA".to_string());
        let _z = ctx.intern(TypeKind::ValueStruct { sid: sid_z.clone() });
        let _a = ctx.intern(TypeKind::ValueStruct { sid: sid_a.clone() });
        ctx.register_type_fqn(sid_z.clone(), "pkg.zmod.TZed");
        ctx.register_type_fqn(sid_a.clone(), "pkg.amod.TAlpha");

        ctx.finalize_type_ids();

        let id_alpha = ctx
            .types
            .iter()
            .find_map(|(id, kind)| {
                if matches!(kind, TypeKind::ValueStruct { sid } if sid == &sid_a) {
                    Some(*id)
                } else {
                    None
                }
            })
            .expect("alpha type must exist");
        let id_zed = ctx
            .types
            .iter()
            .find_map(|(id, kind)| {
                if matches!(kind, TypeKind::ValueStruct { sid } if sid == &sid_z) {
                    Some(*id)
                } else {
                    None
                }
            })
            .expect("zed type must exist");

        assert_eq!(id_alpha, TypeId(1000));
        assert_eq!(id_zed, TypeId(1001));
    }

    #[test]
    fn finalize_type_ids_with_remap_reports_changed_ids() {
        let mut ctx = TypeCtx::new();
        let sid_z = Sid("T:ZZZZZZZZZZ".to_string());
        let sid_a = Sid("T:AAAAAAAAAA".to_string());
        let z = ctx.intern(TypeKind::ValueStruct { sid: sid_z.clone() });
        let a = ctx.intern(TypeKind::ValueStruct { sid: sid_a.clone() });
        ctx.register_type_fqn(sid_z.clone(), "pkg.zmod.TZed");
        ctx.register_type_fqn(sid_a.clone(), "pkg.amod.TAlpha");

        let remap = ctx.finalize_type_ids_with_remap();

        assert_eq!(remap.get(&z), Some(&TypeId(1001)));
        assert_eq!(remap.get(&a), Some(&TypeId(1000)));
    }

    #[test]
    fn finalize_type_ids_with_remap_empty_when_ids_already_canonical() {
        let mut ctx = TypeCtx::new();
        let sid_a = Sid("T:AAAAAAAAAA".to_string());
        let sid_z = Sid("T:ZZZZZZZZZZ".to_string());
        let _a = ctx.intern(TypeKind::ValueStruct { sid: sid_a.clone() });
        let _z = ctx.intern(TypeKind::ValueStruct { sid: sid_z.clone() });
        ctx.register_type_fqn(sid_a.clone(), "pkg.amod.TAlpha");
        ctx.register_type_fqn(sid_z.clone(), "pkg.zmod.TZed");

        let remap = ctx.finalize_type_ids_with_remap();

        assert!(remap.is_empty());
        let next = ctx.intern(TypeKind::RawPtr {
            to: fixed_type_ids::I32,
        });
        assert_eq!(next, TypeId(1002));
    }

    #[test]
    fn canonical_type_str_common_types() {
        let mut ctx = TypeCtx::new();
        let i32_ty = fixed_type_ids::I32;
        let str_ty = fixed_type_ids::STR;

        let shared_str = ctx.intern(TypeKind::HeapHandle {
            hk: HandleKind::Shared,
            base: HeapBase::BuiltinStr,
        });
        let borrow_map = ctx.intern(TypeKind::HeapHandle {
            hk: HandleKind::Borrow,
            base: HeapBase::BuiltinMap {
                key: i32_ty,
                val: str_ty,
            },
        });
        let opt_shared = ctx.intern(TypeKind::BuiltinOption { inner: shared_str });
        let result_ty = ctx.intern(TypeKind::BuiltinResult {
            ok: i32_ty,
            err: borrow_map,
        });
        let raw_i32 = ctx.intern(TypeKind::RawPtr { to: i32_ty });
        let callable = ctx.intern(TypeKind::HeapHandle {
            hk: HandleKind::Unique,
            base: HeapBase::Callable {
                sig_sid: Sid("E:CALLABLE01".to_string()),
            },
        });
        let user_sid = Sid("T:USERTYPE01".to_string());
        let user_type = ctx.intern(TypeKind::ValueStruct {
            sid: user_sid.clone(),
        });
        ctx.register_type_fqn(user_sid, "pkg.mod.TNode");

        assert_eq!(ctx.type_str(i32_ty), "i32");
        assert_eq!(ctx.type_str(str_ty), "Str");
        assert_eq!(ctx.type_str(shared_str), "shared Str");
        assert_eq!(ctx.type_str(borrow_map), "borrow Map<i32,Str>");
        assert_eq!(ctx.type_str(opt_shared), "TOption<shared Str>");
        assert_eq!(ctx.type_str(result_ty), "TResult<i32,borrow Map<i32,Str>>");
        assert_eq!(ctx.type_str(raw_i32), "rawptr<i32>");
        assert_eq!(ctx.type_str(callable), "TCallable<E:CALLABLE01>");
        assert_eq!(ctx.type_str(user_type), "pkg.mod.TNode");

        assert!(!ctx.type_str(result_ty).contains(", "));
        assert!(ctx.type_str(shared_str).starts_with("shared "));
        assert!(ctx.type_str(borrow_map).starts_with("borrow "));
    }
}
