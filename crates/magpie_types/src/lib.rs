//! Magpie type system: TypeKind, HeapBase, type interning, TypeId assignment (§8, §16.2-16.3).

use serde::{Deserialize, Serialize};

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
    I1, I8, I16, I32, I64, I128,
    U1, U8, U16, U32, U64, U128,
    F16, F32, F64,
    Bool, // alias for I1
    Unit,
}

impl PrimType {
    pub fn is_integer(&self) -> bool {
        matches!(self, Self::I1 | Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::I128
            | Self::U1 | Self::U8 | Self::U16 | Self::U32 | Self::U64 | Self::U128 | Self::Bool)
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Self::F16 | Self::F32 | Self::F64)
    }

    pub fn is_signed(&self) -> bool {
        matches!(self, Self::I1 | Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::I128)
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
    Unique, Shared, Borrow, MutBorrow, Weak,
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

/// Type context for interning and layout computation.
#[derive(Debug, Default)]
pub struct TypeCtx {
    pub types: Vec<(TypeId, TypeKind)>,
    next_user_id: u32,
}

impl TypeCtx {
    pub fn new() -> Self {
        let mut ctx = Self { types: Vec::with_capacity(64), next_user_id: 1000 };
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
            (STR, TypeKind::HeapHandle { hk: HandleKind::Unique, base: HeapBase::BuiltinStr }),
            (STR_BUILDER, TypeKind::HeapHandle { hk: HandleKind::Unique, base: HeapBase::BuiltinStrBuilder }),
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
}
