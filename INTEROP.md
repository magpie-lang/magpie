# INTEROP.md — Magpie Crate Interoperability Specification

This document defines all naming conventions, API boundaries, shared types, and contracts
between the 23 workspace crates. All implementors MUST follow these rules.

---

## 1. Crate Dependency Graph (layered)

```
Layer 0 (no deps):     magpie_ast, magpie_rt
Layer 1 (ast only):    magpie_diag
Layer 2 (ast+diag):    magpie_lex, magpie_types
Layer 3:               magpie_parse (lex+ast+diag), magpie_csnf (ast+diag),
                       magpie_hir (ast+types+diag)
Layer 4:               magpie_sema (ast+hir+types+diag),
                       magpie_mpir (hir+types+diag)
Layer 5:               magpie_mono (hir+types+diag),
                       magpie_own (hir+types+diag),
                       magpie_arc (mpir+types+diag)
Layer 6:               magpie_codegen_llvm (mpir+types+diag),
                       magpie_codegen_wasm (mpir+types+diag),
                       magpie_gpu (mpir+types+diag),
                       magpie_pkg (diag), magpie_memory (diag),
                       magpie_web (types+diag)
Layer 7:               magpie_jit (codegen_llvm+diag),
                       magpie_ctx (memory+diag)
Layer 8:               magpie_driver (ast+lex+parse+csnf+hir+sema+types+mono+own+mpir+arc+diag+pkg)
Layer 9:               magpie_cli (driver+diag)
```

Rule: No circular dependencies. Lower layers MUST NOT depend on higher layers.

---

## 2. Shared ID Types (defined in `magpie_types`)

All crates MUST use these exact types for cross-crate references:

| Type | Definition | Usage |
|------|-----------|-------|
| `FileId(u32)` | `magpie_ast::FileId` | Source file identifier |
| `Span` | `magpie_ast::Span` | Byte-offset source span |
| `Spanned<T>` | `magpie_ast::Spanned<T>` | Node with span |
| `PackageId(u32)` | `magpie_types::PackageId` | Package in build graph |
| `ModuleId(u32)` | `magpie_types::ModuleId` | Module in build graph |
| `DefId(u32)` | `magpie_types::DefId` | Definition (fn/type/global) |
| `TypeId(u32)` | `magpie_types::TypeId` | Interned type |
| `InstId(u32)` | `magpie_types::InstId` | Monomorphization instance |
| `FnId(u32)` | `magpie_types::FnId` | Function ID |
| `GlobalId(u32)` | `magpie_types::GlobalId` | Global variable ID |
| `LocalId(u32)` | `magpie_types::LocalId` | SSA local within a function |
| `BlockId(u32)` | `magpie_types::BlockId` | Basic block within a function |
| `Sid(String)` | `magpie_types::Sid` | Stable ID: `<Kind>:<10chars>` |

---

## 3. Naming Conventions

### 3.1 Rust Code

- Crate names: `magpie_<component>` (snake_case)
- Module names: snake_case
- Struct names: PascalCase
- Enum names: PascalCase
- Enum variant names: PascalCase
- Function names: snake_case
- Constants: UPPER_SNAKE_CASE
- Type parameters: single uppercase or PascalCase

### 3.2 Magpie Surface Language Symbols

- Functions: `@<snake_case>` (e.g., `@println`, `@hash_Str`)
- Types: `T<PascalCase>` (e.g., `TPerson`, `TOption`)
- SSA locals: `%<snake_case>` (e.g., `%msg`, `%age_val`)
- Block labels: `bb<digits>` (e.g., `bb0`, `bb1`)
- Modules: `<dotted.path>` (e.g., `pkg.sub.module`)

### 3.3 SID Format (§18.4)

- Module: `M:<10 base32 crockford chars>`
- Function: `F:<10 chars>`
- Type: `T:<10 chars>`
- Global: `G:<10 chars>`
- Extern: `E:<10 chars>`
- Instance: `I:<16 chars>`

Input: `"magpie:sid:v0.1|<kind>|<canonical_string>"`
Hash: `blake3(input)`, encode first 5 bytes (10 chars) as base32 crockford uppercase.

### 3.4 LLVM Symbol Mangling (§19)

- Prefix: `mp$0$`
- Functions: `mp$0$FN$<F_sid_suffix>`
- Mono instances: `mp$0$FN$<F_sid_suffix>$I$<inst_suffix16>`
- Globals: `mp$0$GL$<G_sid_suffix>`
- Type info: `mp$0$TI$<T_sid_suffix>`
- Drop fn: `mp$0$DROP$<T_sid_suffix>`
- Type init: `mp$0$INIT_TYPES$<M_sid_suffix>`
- Program entry: C `main` → `mp_rt_init()` → all `INIT_TYPES` → Magpie `@main`

### 3.5 Diagnostic Code Namespaces (§26.3)

| Prefix | Domain | Crate |
|--------|--------|-------|
| `MPP` | Parse/lex | `magpie_parse`, `magpie_lex` |
| `MPT` | Types | `magpie_types`, `magpie_sema` |
| `MPO` | Ownership | `magpie_own` |
| `MPA` | ARC | `magpie_arc` |
| `MPS` | SSA verification | `magpie_mpir`, `magpie_hir` |
| `MPAS` | Async | `magpie_sema` |
| `MPF` | FFI | `magpie_codegen_llvm` |
| `MPG` | GPU | `magpie_gpu` |
| `MPW` | Web | `magpie_web` |
| `MPK` | Package manager | `magpie_pkg` |
| `MPL` | Lint / LLM features | `magpie_diag` |

---

## 4. Key API Boundaries Between Crates

### 4.1 `magpie_lex` → `magpie_parse`

```rust
// magpie_lex public API
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    pub text: String, // only for identifiers/literals
}

pub enum TokenKind {
    // Keywords, punctuation, literals, identifiers, EOF
    // ~60 variants matching §7.1
}

pub fn lex(file_id: FileId, source: &str, diag: &mut DiagnosticBag) -> Vec<Token>;
```

### 4.2 `magpie_parse` → `magpie_driver`

```rust
// magpie_parse public API
pub fn parse_file(
    tokens: &[Token],
    file_id: FileId,
    diag: &mut DiagnosticBag,
) -> Result<AstFile, ()>;
```

### 4.3 `magpie_csnf` → `magpie_driver`

```rust
// magpie_csnf public API
pub fn format_csnf(ast: &AstFile, source_map: &SourceMap) -> String;
pub fn compute_digest(canonical_source: &str) -> String; // blake3 hex
```

### 4.4 `magpie_sema` → `magpie_driver`

```rust
// magpie_sema public API
pub struct SymbolTable { /* per-module symbols: fns, types, globals, sigs */ }
pub struct ResolvedModule { /* AST + resolved names + symbol table */ }

pub fn resolve_modules(
    files: &[AstFile],
    diag: &mut DiagnosticBag,
) -> Result<Vec<ResolvedModule>, ()>;
```

### 4.5 `magpie_types` → all type-aware crates

```rust
// magpie_types public API
pub struct TypeCtx { /* interning + layout */ }

impl TypeCtx {
    pub fn intern(&mut self, kind: TypeKind) -> TypeId;
    pub fn lookup(&self, id: TypeId) -> Option<&TypeKind>;
}
```

### 4.6 `magpie_hir` → `magpie_sema`, `magpie_own`, `magpie_mpir`

```rust
// magpie_hir public API
pub struct HirModule {
    pub module_id: ModuleId,
    pub functions: Vec<HirFunction>,
    pub globals: Vec<HirGlobal>,
    pub type_decls: Vec<HirTypeDecl>,
}

pub struct HirFunction {
    pub fn_id: FnId,
    pub sid: Sid,
    pub name: String,
    pub params: Vec<(LocalId, TypeId)>,
    pub ret_ty: TypeId,
    pub blocks: Vec<HirBlock>,
    pub is_async: bool,
}

pub fn lower_ast_to_hir(
    resolved: &ResolvedModule,
    type_ctx: &mut TypeCtx,
    diag: &mut DiagnosticBag,
) -> Result<HirModule, ()>;
```

### 4.7 `magpie_own` → `magpie_driver`

```rust
// magpie_own public API
pub fn check_ownership(
    module: &HirModule,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) -> Result<(), ()>;
```

### 4.8 `magpie_mpir` → `magpie_arc`, `magpie_codegen_llvm`

```rust
// magpie_mpir public API
pub struct MpirModule { /* §17.1 */ }

pub fn lower_hir_to_mpir(
    hir: &HirModule,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) -> Result<MpirModule, ()>;

pub fn verify_mpir(
    module: &MpirModule,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) -> Result<(), ()>;

pub fn print_mpir(module: &MpirModule, type_ctx: &TypeCtx) -> String;
```

### 4.9 `magpie_arc` → `magpie_driver`

```rust
// magpie_arc public API
pub fn insert_arc_ops(
    module: &mut MpirModule,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) -> Result<(), ()>;

pub fn optimize_arc(module: &mut MpirModule, type_ctx: &TypeCtx);
```

### 4.10 `magpie_codegen_llvm` → `magpie_driver`

```rust
// magpie_codegen_llvm public API
pub struct LlvmCodegen { /* LLVM module handle */ }

impl LlvmCodegen {
    pub fn new(target_triple: &str) -> Result<Self, String>;
    pub fn compile_module(
        &mut self,
        mpir: &MpirModule,
        type_ctx: &TypeCtx,
    ) -> Result<(), String>;
    pub fn emit_llvm_ir(&self) -> String;
    pub fn emit_object(&self, path: &std::path::Path) -> Result<(), String>;
    pub fn link_executable(
        objects: &[&std::path::Path],
        output: &std::path::Path,
        rt_lib: &std::path::Path,
    ) -> Result<(), String>;
}
```

### 4.11 `magpie_driver` → `magpie_cli`

```rust
// magpie_driver public API
pub struct DriverConfig {
    pub entry: String,
    pub profile: String,
    pub target: Option<String>,
    pub emit: Vec<String>,
    pub max_errors: u32,
    pub llm_mode: bool,
    pub token_budget: Option<u32>,
}

pub struct BuildResult {
    pub success: bool,
    pub diagnostics: Vec<Diagnostic>,
    pub artifacts: Vec<String>,
    pub timing_ms: std::collections::HashMap<String, u64>,
}

pub fn build(config: &DriverConfig) -> BuildResult;
pub fn format_files(paths: &[String], fix_meta: bool) -> BuildResult;
pub fn run_tests(filter: Option<&str>) -> BuildResult;
```

---

## 5. Runtime ABI Contract (`magpie_rt` ↔ `magpie_codegen_llvm`)

All functions use C ABI (`extern "C"`). Header layout:

```c
typedef struct MpRtHeader {
    _Atomic uint64_t strong;    // offset 0
    _Atomic uint64_t weak;      // offset 8
    uint32_t type_id;           // offset 16
    uint32_t flags;             // offset 20
    uint64_t reserved0;         // offset 24
} MpRtHeader;                   // sizeof=32, payload at offset 32
```

### 5.1 Core Runtime Functions

| Symbol | Signature |
|--------|-----------|
| `mp_rt_init` | `() -> void` |
| `mp_rt_register_types` | `(*const MpRtTypeInfo, u32) -> void` |
| `mp_rt_type_info` | `(u32) -> *const MpRtTypeInfo` |
| `mp_rt_alloc` | `(u32, u64, u64, u32) -> *mut MpRtHeader` |
| `mp_rt_retain_strong` | `(*mut MpRtHeader) -> void` |
| `mp_rt_release_strong` | `(*mut MpRtHeader) -> void` |
| `mp_rt_retain_weak` | `(*mut MpRtHeader) -> void` |
| `mp_rt_release_weak` | `(*mut MpRtHeader) -> void` |
| `mp_rt_weak_upgrade` | `(*mut MpRtHeader) -> *mut MpRtHeader` |
| `mp_rt_str_from_utf8` | `(*const u8, u64) -> *mut MpRtHeader` |
| `mp_rt_str_bytes` | `(*mut MpRtHeader, *mut u64) -> *const u8` |
| `mp_rt_panic` | `(*mut MpRtHeader) -> !` |

### 5.2 Collection Functions

All `arr.*`, `map.*`, `str.*` lower to `mp_rt_arr_*`, `mp_rt_map_*`, `mp_rt_str_*`.
See §20.1.5 for exact signatures.

### 5.3 Fixed TypeId Table

| ID | Type |
|----|------|
| 0 | unit |
| 1 | bool/i1 |
| 2-6 | i8, i16, i32, i64, i128 |
| 7-11 | u8, u16, u32, u64, u128 |
| 12 | u1 |
| 13-15 | f16, f32, f64 |
| 20 | Str |
| 21 | TStrBuilder |
| 22-26 | Array/Map/TOption/TResult/TCallable bases |
| 30-32 | GPU types |
| ≥1000 | User-defined types |

---

## 6. Compilation Pipeline Stage Contract (§22.1)

```
Stage 1:  Parse + CSNF    → AstFile + digest           [magpie_lex, magpie_parse, magpie_csnf]
Stage 2:  Resolve          → HirModule (resolved names)  [magpie_sema]
Stage 3:  Typecheck        → typed HIR (TypeIds, layouts) [magpie_types, magpie_sema]
Stage 3.5: Async lowering  → coroutine state machines    [magpie_hir]
Stage 4:  Verify HIR       → SSA + borrow-locality ok    [magpie_hir]
Stage 5:  Ownership check  → ownership proofs or errors  [magpie_own]
Stage 6:  Lower to MPIR    → MpirModule (no ARC ops)     [magpie_mpir]
Stage 7:  MPIR verify      → SSA + type refs + phi ok    [magpie_mpir]
Stage 8:  ARC insertion    → MPIR with ARC ops            [magpie_arc]
Stage 9:  ARC optimization → optimized retain/release     [magpie_arc]
Stage 10: LLVM codegen     → LLVM module                  [magpie_codegen_llvm]
Stage 11: Link             → executable/shared lib        [magpie_codegen_llvm]
Stage 12: MMS update       → updated retrieval index      [magpie_memory]
```

Each stage MUST:
- Accept input from the previous stage
- Collect up to `max_errors` diagnostics
- Skip if previous stage produced errors
- Return diagnostics via `DiagnosticBag`

---

## 7. Test Fixture Convention

- Source fixtures: `tests/fixtures/<name>.mp`
- Expected MPIR: `tests/fixtures/<name>.mpir`
- Expected JSON: `tests/fixtures/<name>.json`
- Snapshot tests use `insta` crate
- Golden tests compare output byte-for-byte after CSNF normalization
