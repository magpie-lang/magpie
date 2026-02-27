---
name: magpie-engineer
description: Extremely detailed, source-first guide for authoring Magpie (.mp), implementing compiler/runtime changes, and debugging parse/resolve/type/HIR/ownership/MPIR/codegen/link/runtime failures. Use for language grammar/semantics questions, opcode work, diagnostics triage, and production-grade validation.
---

# Magpie Engineer Skill (Source of Truth: Code)

Use this skill whenever you touch Magpie language files (`.mp`) or any compiler/runtime crate in this repo.

---

## 0) Non-negotiable rules

1. **Source code is canonical.**
   - Trust only implementation + tests in this repository.
   - Do **not** assume `SPEC.md` / `PLAN.md` are current.
2. **Diagnose by pipeline stage + diagnostic code first**, then patch.
3. **Smallest reproducer first** (fixture or tiny `.mp`).
4. **Patch the owning layer only** (parser vs sema vs ownership vs codegen vs runtime).
5. **Prove fixes with commands and exit codes.**

---

## 1) Where truth lives (read in this order)

### Language surface + grammar
- `crates/magpie_lex/src/lib.rs` — tokenization, keyword/op spelling
- `crates/magpie_parse/src/lib.rs` — actual grammar and key requirements
- `crates/magpie_csnf/src/lib.rs` — canonical pretty-printer (ground truth for canonical source form)

### AST/HIR/type model
- `crates/magpie_ast/src/lib.rs` — AST node set
- `crates/magpie_types/src/lib.rs` — primitive/handle/base type semantics
- `crates/magpie_hir/src/lib.rs` — HIR model + HIR verifier invariants

### Semantic legality
- `crates/magpie_sema/src/lib.rs` — resolve/lower/type/trait/v0.1 restrictions
- `crates/magpie_own/src/lib.rs` — ownership/borrow/move/send rules
- `crates/magpie_mpir/src/lib.rs` — MPIR model + verifier

### Pipeline + emitted artifacts
- `crates/magpie_driver/src/lib.rs`
- `crates/magpie_cli/src/main.rs`

### Runtime/codegen backends
- `crates/magpie_codegen_llvm/src/lib.rs`
- `crates/magpie_rt/src/lib.rs`
- `crates/magpie_gpu/src/lib.rs`
- `crates/magpie_codegen_wasm/src/lib.rs` (when wasm path is relevant)

### Canonical examples
- `tests/fixtures/*.mp`
- especially `tests/fixtures/feature_harness.mp`

---

## 2) CLI and execution workflow

Core commands:

- Build:
  - `cargo run -p magpie_cli -- build --entry <path> --emit <kinds>`
- Run:
  - `cargo run -p magpie_cli -- run --entry <path> --emit <kinds>`
- Parse only:
  - `cargo run -p magpie_cli -- parse --entry <path>`
- Test crate:
  - `cargo test -p <crate>`
- Integration fixtures:
  - `cargo test --test integration_test`
- Full sweep:
  - `cargo test --workspace`

Useful emit kinds from driver planning:
- `llvm-ir`, `llvm-bc`, `object`, `asm`, `spv`, `exe`, `shared-lib`, `mpir`, `mpd`, `mpdbg`, `symgraph`, `depsgraph`, `ownershipgraph`, `cfggraph`

---

## 3) Full lexical model (from lexer)

### Comments
- `; ...` line comment
- `;;; ...` doc comment token (`DocComment`)

### Name classes
- Module path segments: plain identifiers (`ident`)
- Function symbol: `@name` (`FnName`)
- SSA local: `%name` (`SsaName`)
- Type name: `TName` (`TypeName`)
- Block label: `bbN` (`BlockLabel`)

### Identifier rules
- start: ASCII alpha or `_`
- continue: start chars + digits

### Literals
- Int: decimal or hex (`0x...`)
- Float: `digits.digits` optionally with suffix `f32` / `f64`
- String escapes: `\n`, `\t`, `\\`, `\"`, `\u{...}`
- Bool: `true` / `false`
- Unit literal tokenized as identifier text `unit` and interpreted contextually

### Punctuation
`{ } ( ) < > [ ] = : , . ->`

---

## 4) Full grammar skeleton (parser-authoritative)

## 4.1 File structure (strict order)

```ebnf
file      := header decl*
header    := "module" module_path
             "exports" export_block
             "imports" import_block
             "digest" string_lit
```

`module/exports/imports/digest` order is mandatory in parser.

### Exports
```ebnf
export_block := "{" (export_item ("," export_item)*)? "}"
export_item  := fn_name | type_name
```

### Imports
```ebnf
import_block := "{" (import_group ("," import_group)*)? "}"
import_group := module_path "::" "{" (import_item ("," import_item)*)? "}"
import_item  := fn_name | type_name
```

## 4.2 Declarations

```ebnf
decl := fn_decl
      | async_fn_decl
      | unsafe_fn_decl
      | gpu_fn_decl
      | heap_struct_decl
      | value_struct_decl
      | heap_enum_decl
      | value_enum_decl
      | extern_decl
      | global_decl
      | impl_decl
      | sig_decl
```

Doc comments (`;;;`) may precede declarations and are attached.

### Functions
```ebnf
fn_decl         := "fn" fn_name "(" params ")" "->" type meta_opt blocks
async_fn_decl   := "async" "fn" ...
unsafe_fn_decl  := "unsafe" "fn" ...
gpu_fn_decl     := "gpu" "fn" ... "target" "(" ident ")" meta_opt blocks
```

### Function meta
```ebnf
meta_opt := ε | "meta" "{" meta_entry* "}"
meta_entry := "uses" "{" fqn_list "}"
            | "effects" "{" ident_list "}"
            | "cost" "{" (ident "=" int_lit ("," ... )*)? "}"
```

### Type declarations
```ebnf
heap_struct_decl  := "heap" "struct" TName type_params? "{" struct_fields "}"
value_struct_decl := "value" "struct" TName type_params? "{" struct_fields "}"
heap_enum_decl    := "heap" "enum"   TName type_params? "{" enum_variants "}"
value_enum_decl   := "value" "enum"  TName type_params? "{" enum_variants "}"
```

Struct field entry keyword is textual `field`.
Enum variant entry keyword is textual `variant`.

### Extern
```ebnf
extern_decl := "extern" string_lit "module" ident "{" extern_item* "}"
extern_item := "fn" fn_name "(" params ")" "->" type attrs_opt
attrs_opt   := ε | "attrs" "{" (ident "=" string_lit ("," ... )*)? "}"
```

### Global
```ebnf
global_decl := "global" fn_name ":" type "=" const_expr
```

### Impl/sig
```ebnf
impl_decl := "impl" ident "for" type "=" fn_ref
sig_decl  := "sig" TName "(" type_list ")" "->" type
```

## 4.3 Function body, blocks, instructions, terminators

```ebnf
blocks    := "{" block+ "}"
block     := block_label ":" instr* terminator
```

Parser requires at least one block; every block must end in terminator.

### Instruction forms
- SSA assignment:
  - `%name: Type = <value-op>`
- Void op:
  - `<void-op>`
- Unsafe sub-block:
  - `unsafe { <ssa-or-void-instr>+ }`

Unsafe sub-block currently allows only SSA assign and void ops.

### Terminators
```ebnf
terminator := "ret" value_ref?
            | "br" block_label
            | "cbr" value_ref block_label block_label
            | "switch" value_ref "{" ("case" const_lit "->" block_label)* "}" "else" block_label
            | "unreachable"
```

---

## 5) Type grammar + type semantics

## 5.1 Ownership prefix

Optional ownership modifiers before base type:
- `shared`
- `borrow`
- `mutborrow`
- `weak`

Parser accepts these as leading keywords.

## 5.2 Primitive types

`i1 i8 i16 i32 i64 i128 u1 u8 u16 u32 u64 u128 f16 f32 f64 bool unit`

## 5.3 Builtin types

- `Str`
- `Array<T>`
- `Map<K, V>`
- `TOption<T>`
- `TResult<Ok, Err>`
- `TStrBuilder`
- `TMutex<T>`
- `TRwLock<T>`
- `TCell<T>`
- `TFuture<T>`
- `TChannelSend<T>`
- `TChannelRecv<T>`
- `TCallable<module?.SigName>`

## 5.4 User + raw pointer types

- Named type:
  - `TName<...>` or `module.path.TName<...>`
- Raw pointer:
  - `rawptr<T>`

## 5.5 Important semantic mapping rules

From `ast_type_to_type_id` and `lower_builtin_type`:

1. Named local `value struct/enum` without ownership prefix maps to **value type** (`TypeKind::ValueStruct`) for local value declarations.
2. Heap-y things map to `TypeKind::HeapHandle { hk, base }` where `hk` comes from ownership modifier (default unique).
3. `TOption` and `TResult` are value enums; `shared`/`weak` on these are rejected (`MPT0002`, `MPT0003`).
4. Unknown primitive yields `MPT0001`.

---

## 6) Full opcode surface (parser + CSNF canonical spelling)

Use exactly these names and keys.

## 6.1 Value-producing ops

### Constants
- `const.<Type> <literal>`

### Integer arithmetic / bitwise
- `i.add { lhs=V, rhs=V }`
- `i.sub { lhs=V, rhs=V }`
- `i.mul { lhs=V, rhs=V }`
- `i.sdiv { lhs=V, rhs=V }`
- `i.udiv { lhs=V, rhs=V }`
- `i.srem { lhs=V, rhs=V }`
- `i.urem { lhs=V, rhs=V }`
- `i.add.wrap { lhs=V, rhs=V }`
- `i.sub.wrap { lhs=V, rhs=V }`
- `i.mul.wrap { lhs=V, rhs=V }`
- `i.add.checked { lhs=V, rhs=V }`
- `i.sub.checked { lhs=V, rhs=V }`
- `i.mul.checked { lhs=V, rhs=V }`
- `i.and { lhs=V, rhs=V }`
- `i.or { lhs=V, rhs=V }`
- `i.xor { lhs=V, rhs=V }`
- `i.shl { lhs=V, rhs=V }`
- `i.lshr { lhs=V, rhs=V }`
- `i.ashr { lhs=V, rhs=V }`

### Float arithmetic
- `f.add { lhs=V, rhs=V }`
- `f.sub { lhs=V, rhs=V }`
- `f.mul { lhs=V, rhs=V }`
- `f.div { lhs=V, rhs=V }`
- `f.rem { lhs=V, rhs=V }`
- `f.add.fast { lhs=V, rhs=V }`
- `f.sub.fast { lhs=V, rhs=V }`
- `f.mul.fast { lhs=V, rhs=V }`
- `f.div.fast { lhs=V, rhs=V }`

### Comparisons
- `icmp.eq { lhs=V, rhs=V }`
- `icmp.ne { lhs=V, rhs=V }`
- `icmp.slt { lhs=V, rhs=V }`
- `icmp.sgt { lhs=V, rhs=V }`
- `icmp.sle { lhs=V, rhs=V }`
- `icmp.sge { lhs=V, rhs=V }`
- `icmp.ult { lhs=V, rhs=V }`
- `icmp.ugt { lhs=V, rhs=V }`
- `icmp.ule { lhs=V, rhs=V }`
- `icmp.uge { lhs=V, rhs=V }`
- `fcmp.oeq { lhs=V, rhs=V }`
- `fcmp.one { lhs=V, rhs=V }`
- `fcmp.olt { lhs=V, rhs=V }`
- `fcmp.ogt { lhs=V, rhs=V }`
- `fcmp.ole { lhs=V, rhs=V }`
- `fcmp.oge { lhs=V, rhs=V }`

### Calls / async-related
- `call @fn<TypeArgs?> { key=Arg, ... }`
- `call.indirect V { key=Arg, ... }`
- `try @fn<TypeArgs?> { key=Arg, ... }`
- `suspend.call @fn<TypeArgs?> { key=Arg, ... }`
- `suspend.await { fut=V }`

### Struct / enum / SSA
- `new Type { field=V, ... }`
- `getfield { obj=V, field=name }`  (**strict key order obj,field in parser branch**)
- `phi Type { [bbN:V], [bbM:V], ... }`
- `enum.new<Variant> { key=V, ... }`
- `enum.tag { v=V }`
- `enum.payload<Variant> { v=V }`
- `enum.is<Variant> { v=V }`

### Ownership conversion
- `share { v=V }`
- `clone.shared { v=V }`
- `clone.weak { v=V }`
- `weak.downgrade { v=V }`
- `weak.upgrade { v=V }`
- `cast<PrimFrom, PrimTo> { v=V }`
- `borrow.shared { v=V }`
- `borrow.mut { v=V }`

### Raw pointer (unsafe context required)
- `ptr.null<Type>`
- `ptr.addr<Type> { p=V }`
- `ptr.from_addr<Type> { addr=V }`
- `ptr.add<Type> { p=V, count=V }`
- `ptr.load<Type> { p=V }`

### Callable capture
- `callable.capture @fn { captureName=V, ... }`

### Arrays
- `arr.new<T> { cap=V }`
- `arr.len { arr=V }`
- `arr.get { arr=V, idx=V }`
- `arr.pop { arr=V }`
- `arr.slice { arr=V, start=V, end=V }`
- `arr.contains { arr=V, val=V }`
- `arr.map { arr=V, fn=V }`
- `arr.filter { arr=V, fn=V }`
- `arr.reduce { arr=V, init=V, fn=V }`

### Maps
- `map.new<K, V> { }`
- `map.len { map=V }`
- `map.get { map=V, key=V }`
- `map.get_ref { map=V, key=V }`
- `map.delete { map=V, key=V }`
- `map.contains_key { map=V, key=V }`
- `map.keys { map=V }`
- `map.values { map=V }`

### Strings + JSON
- `str.concat { a=V, b=V }`
- `str.len { s=V }`
- `str.eq { a=V, b=V }`
- `str.slice { s=V, start=V, end=V }`
- `str.bytes { s=V }`
- `str.builder.new { }`
- `str.builder.build { b=V }`
- `str.parse_i64 { s=V }`
- `str.parse_u64 { s=V }`
- `str.parse_f64 { s=V }`
- `str.parse_bool { s=V }`
- `json.encode<Type> { v=V }`
- `json.decode<Type> { s=V }`

### GPU value ops
- `gpu.thread_id { dim=V }`
- `gpu.workgroup_id { dim=V }`
- `gpu.workgroup_size { dim=V }`
- `gpu.global_id { dim=V }`
- `gpu.buffer_load<Type> { buf=V, idx=V }`
- `gpu.buffer_len<Type> { buf=V }`
- `gpu.shared<count, Type>`
- `gpu.launch { device=V, kernel=@fn, grid=Arg, block=Arg, args=Arg }` (**strict key order**)
- `gpu.launch_async { device=V, kernel=@fn, grid=Arg, block=Arg, args=Arg }` (**strict key order**)

## 6.2 Void ops

- `call_void @fn<TypeArgs?> { key=Arg, ... }`
- `call_void.indirect V { key=Arg, ... }`
- `setfield { obj=V, field=name, val=V }` (**must use `val=`, not `value=`**)
- `panic { msg=V }`
- `ptr.store<Type> { p=V, v=V }` (unsafe context)
- `arr.set { arr=V, idx=V, val=V }`
- `arr.push { arr=V, val=V }`
- `arr.sort { arr=V }`
- `arr.foreach { arr=V, fn=V }`
- `map.set { map=V, key=V, val=V }`
- `map.delete_void { map=V, key=V }`
- `str.builder.append_str { b=V, s=V }`
- `str.builder.append_i64 { b=V, v=V }`
- `str.builder.append_i32 { b=V, v=V }`
- `str.builder.append_f64 { b=V, v=V }`
- `str.builder.append_bool { b=V, v=V }`
- `gpu.barrier`
- `gpu.buffer_store<Type> { buf=V, idx=V, v=V }`

## 6.3 Arg value grammar

`Arg` in call/gpu forms can be:
1. value ref (`%x` or `const...`)
2. list `[ArgElem, ...]` where `ArgElem` is value or fn ref
3. fn ref (`@fn` or `module.@fn`)

**Important lowering behavior:** call argument keys are currently ignored in lowering; arguments are flattened by pair order (`lower_call_args`). Keep key order stable and explicit anyway.

## 6.4 Internal-only op tokens

Lexer recognizes:
- `arc.retain`, `arc.release`, `arc.retain_weak`, `arc.release_weak`

These are **not** parser surface ops for `.mp` authoring; they are compiler-internal (ARC stages / MPIR).

---

## 7) Semantic invariants by stage

## 7.1 Resolve/symbol layer (`MPS*`, `MPF*`)

- Duplicate module path -> `MPS0001`
- Missing imported module -> `MPS0002`
- Unresolvable import item -> `MPS0003`
- Import conflicts with local symbols -> `MPS0004` / `MPS0005`
- Ambiguous imports -> `MPS0006`
- No overload in sig namespace -> `MPS0023`
- Raw pointer ops outside unsafe context -> `MPS0024`
- Unsafe fn call outside unsafe context -> `MPS0025`
- Extern rawptr return requires attrs `returns=owned|borrowed` -> `MPF0001`

## 7.2 Typecheck layer (`MPT*`)

Core checks from `typecheck_module` and helpers:

- Numeric family checks:
  - unknown lhs/rhs type `MPT2012/MPT2013`
  - mismatch lhs/rhs `MPT2014`
  - non-numeric primitive for family `MPT2015`
- Call checks:
  - arity `MPT2001`
  - unknown arg type `MPT2002`
  - arg type mismatch `MPT2003`
  - unknown callee sid `MPT2004`
  - invalid type arg `MPT2005`
- Projection/constructor checks:
  - `getfield` object unknown/wrong/not struct/missing field `MPT2006..MPT2009`
  - `cast` must be primitive->primitive `MPT2010/MPT2011`
  - `new` field duplicate/unknown/unknown arg type/type mismatch/missing field `MPT2016..MPT2020`
  - `new` non-struct target / unknown struct `MPT2021/MPT2022`
  - `enum.new` invalid variant for `TOption`/`TResult`/user enum etc `MPT2023..MPT2027`
- Trait impl signature checks `MPT2028..MPT2031`
- Explicit impl references unknown local function `MPT2032`

## 7.3 HIR invariants (`MPHIR*` + SSA)

From `verify_hir`:

- SSA single-def / use-before-def / dominance:
  - `MPS0001`, `MPS0002`, `MPS0003`
- `getfield` object must be borrow/mutborrow -> `MPHIR01`
- `setfield` object must be mutborrow -> `MPHIR02`
- Borrow values must not be returned / function ret type cannot be borrow -> `MPHIR03`
- Borrow in phi / cross-block borrow use also flagged (`MPO0102`, `MPO0101`)

## 7.4 Ownership (`MPO*`)

Major rules from `magpie_own`:

- Borrow escapes scope (globals, storing borrows into aggregates) -> `MPO0003`
- Shared mutation / wrong ownership mode for mut/read ops -> `MPO0004`
- Use-after-move -> `MPO0007`
- Move while borrowed / illegal borrow state -> `MPO0011`
- Borrow crosses block boundary -> `MPO0101`
- Borrow in phi -> `MPO0102`
- `map.get` requires Dupable V for by-value result -> `MPO0103`
- Spawn/send restrictions -> `MPO0201`
  - spawn-like callee (sid contains `spawn`) requires first arg TCallable
  - captured values must be send-safe under current `is_send_type` rules

Projection semantics enforced:
- `getfield`, `arr.get`, `map.get_ref` result type must match ownership-sensitive projection rules:
  - copy-like stored type -> by value
  - move-only strong handle -> borrow/mutborrow projection
  - weak -> weak clone

## 7.5 MPIR verifier (`MPS*`)

From `verify_mpir`:

- Valid SID formats and type references
- SSA rules (`MPS0001`, `MPS0002`, `MPS0003`)
- CFG duplicates (`MPS0009`)
- missing blocks/terminator violations (`MPS0010` context)
- call arity mismatch (`MPS0012`)
- phi type legality (`MPS0008`)
- arc ops forbidden pre-ARC stage (`MPS0014`)

## 7.6 v0.1 restriction checks

`check_v01_restrictions` enforces:

- deferred aggregate kinds (Arr/Vec/Tuple type kinds) -> `MPT1021`
- value enum deferred -> `MPT1020`
- value struct fields cannot contain heap handles -> `MPT1005`
- `suspend.call` on non-function target (TCallable form) forbidden -> `MPT1030`

Trait constraints for collection ops:
- `arr.contains` requires `eq`
- `arr.sort` requires `ord`
- `map.new<K,V>` requires `hash` and `eq` for `K`
- missing impl -> `MPT1023`

---

## 8) Unsafe, async, and control-flow semantics

## 8.1 Unsafe

Only in unsafe context (`unsafe fn` or `unsafe { ... }`):
- `ptr.null`, `ptr.addr`, `ptr.from_addr`, `ptr.add`, `ptr.load`, `ptr.store`
- calls to functions marked unsafe

Violations emit `MPS0024` / `MPS0025`.

## 8.2 Async lowering reality (driver)

Driver stage `stage3_5_async_lowering` lowers async functions by:
1. finding `suspend.call` / `suspend.await`
2. adding synthetic `%state: i32` param
3. splitting blocks around suspend points
4. inserting dispatch `switch` over resume states
5. rewriting callsites to lowered async sids to prepend state argument `0`

After lowering, function `is_async` is set false in HIR used downstream.

Note: diagnostic constant `MPAS0001` exists in diag codes, but current code path does not emit it directly.

---

## 9) Canonical .mp authoring templates

## 9.1 Minimal valid module

```mp
module demo.main
exports { @main }
imports { }
digest "0000000000000000"

fn @main() -> i64 {
bb0:
  ret const.i64 0
}
```

## 9.2 Borrow-safe struct mutation/read split

```mp
heap struct TPoint {
  field x: i64
  field y: i64
}

fn @f() -> i64 {
bb0:
  %p: TPoint = new TPoint { x=const.i64 1, y=const.i64 2 }
  %pm: mutborrow TPoint = borrow.mut { v=%p }
  setfield { obj=%pm, field=y, val=const.i64 3 }
  br bb1

bb1:
  %pb: borrow TPoint = borrow.shared { v=%p }
  %y: i64 = getfield { obj=%pb, field=y }
  ret %y
}
```

---

## 10) Pipeline model for debugging

Driver stage names (actual constants):
1. `stage1_read_lex_parse`
2. `stage2_resolve`
3. `stage3_typecheck`
4. `stage3_5_async_lowering`
5. `stage4_verify_hir`
6. `stage5_ownership_check`
7. `stage6_lower_mpir`
8. `stage7_verify_mpir`
9. `stage8_arc_insertion`
10. `stage9_arc_optimization`
11. `stage10_codegen`
12. `stage11_link`
13. `stage12_mms_update`

When triaging failures, always name the failing stage.

---

## 11) Error-driven fix playbooks

## 11.1 Parse/lex (`MPP*`)

- Check header order and block terminators first.
- Check key names and key order for strict ops (`getfield`, `setfield`, `gpu.launch*`).
- Confirm function names have `@`, locals have `%`, types use `T` prefix where required.

## 11.2 Resolve (`MPS0001..0006`, `MPS0024/25`, `MPF0001`)

- Fix imports/module path first, then symbol conflicts.
- Wrap ptr ops or unsafe calls in `unsafe {}` or mark caller `unsafe fn`.
- Add extern attrs for rawptr returns.

## 11.3 Type (`MPT*`)

- For constructor errors, verify field completeness and exact field names.
- For binary op errors, unify operand types before op.
- For trait requirement errors (`MPT1023`), add explicit impl function and impl declaration.

## 11.4 Ownership (`MPO*`)

- Never let borrow handles cross block boundaries.
- End mutborrow usage in one block, branch, then create shared borrow in successor.
- For `map.get` on non-Dupable V, use ref form (`map.get_ref`) + explicit handling.

## 11.5 MPIR/backend/link (`MPS*`, `MPG*`, `MPLINK*`, `MPL0002`)

- Emit `mpir` and `llvm-ir`, inspect for missing/invalid artifacts.
- If requested emit file is missing, treat as hard failure (`MPL0002` path).
- Check runtime symbol declaration/definition parity when link fails.

---

## 12) How to implement a new language feature correctly

For a new opcode/syntax/type feature:

1. **Lexer** (`magpie_lex`): add token spelling.
2. **Parser** (`magpie_parse`): add grammar branch + keys.
3. **AST/HIR/MPIR enums** as needed.
4. **Sema lowering** (`magpie_sema`): AST -> HIR conversion.
5. **Typecheck semantics** (`typecheck_module` + helpers).
6. **HIR verifier/ownership rules** (`magpie_hir`, `magpie_own`) if behavior affects borrows/moves.
7. **MPIR lower/verify** (`magpie_mpir`) keep invariants valid.
8. **Codegen** (`magpie_codegen_llvm` / wasm/gpu) + runtime ABI (`magpie_rt`) if needed.
9. **CSNF printing** (`magpie_csnf`) for canonical output.
10. **Tests** at parser + sema + ownership + codegen + integration layers.

Do not skip intermediate layers; partial implementation causes stage-mismatch regressions.

---

## 13) Comprehensive testing strategy (language + semantics)

## 13.1 Fast local loop

1. Reproducer file in `tests/fixtures/`.
2. `cargo test -p magpie_parse`
3. `cargo test -p magpie_sema`
4. `cargo test -p magpie_own`
5. `cargo test -p magpie_mpir`

## 13.2 End-to-end compile/run

- Build fixture with multiple emits:
  - `cargo run -p magpie_cli -- build --entry tests/fixtures/feature_harness.mp --emit mpir,llvm-ir,mpdbg,exe`
- Run:
  - `cargo run -p magpie_cli -- run --entry tests/fixtures/feature_harness.mp`

## 13.3 Regression expectations

Every bug fix should add one of:
- parser test,
- sema/ownership unit test,
- integration fixture,
- codegen assertion test,
- runtime execution check.

---

## 14) High-value gotchas (easy to miss)

1. **Header order is strict** in parser.
2. **Call argument keys are not semantic** today; order is what lowering preserves.
3. `setfield` uses `val=` key (not `value=`).
4. `getfield`/`setfield` have explicit key-order parsing logic.
5. Borrows cannot appear in phi or cross blocks.
6. `map.get` by-value result requires Dupable map value type.
7. `TOption`/`TResult` reject shared/weak ownership prefix.
8. Arc op tokens exist in lexer but are not surface source operations.
9. Async lowering rewrites function shape; debug post-lowering IR, not just source intent.

---

## 15) Production-ready completion checklist

Before declaring done:

- [ ] Reproducer failed before and passes after.
- [ ] No unresolved diagnostics for touched path.
- [ ] Changed crates have targeted tests passing.
- [ ] Integration fixture path passes (`integration_test` when surface changed).
- [ ] End-to-end compile (and run when relevant) completed.
- [ ] If codegen/runtime touched: validate emitted `.ll` + binary execution.
- [ ] Output includes exact commands + outcomes + residual risks.

---

## 16) Response protocol for failures/fixes

Always report in this order:
1. Stage + diagnostic code(s)
2. Root cause (source-grounded)
3. Exact files/functions changed
4. Validation commands and results
5. Remaining risk / next follow-up

Never claim “fixed” without command evidence.

---

## 17) Per-op type/ownership contract matrix (implementation-grounded)

Use this section when writing or reviewing `.mp` operations. It states what is currently enforced by code (not wishful design).

### 17.1 Interpreting result types

- In source, each value op appears in an SSA assignment:
  - `%dst: DeclTy = op ...`
- So the immediate “result type” is **declared by `DeclTy`**.
- Static checks then enforce compatibility at various layers:
  - sema/typecheck (`MPT*`),
  - HIR verifier (`MPHIR*`, `MPS*`),
  - ownership checker (`MPO*`),
  - MPIR verifier (`MPS*`).

If an op family lacks dedicated checker logic today, rely on downstream failures and backend expectations.

### 17.2 Arithmetic/comparison families

| Family | Operand contract | Result contract | Main enforcement |
|---|---|---|---|
| `i.*`, `icmp.*` | lhs/rhs must have known type, equal type, integer primitive | Declared SSA type must be coherent downstream | `MPT2012/2013/2014/2015` |
| `f.*`, `fcmp.*` | lhs/rhs must have known type, equal type, float primitive | Declared SSA type must be coherent downstream | `MPT2012/2013/2014/2015` |
| `cast<from,to>` | operand type known; both `from` and `to` primitive | Declared SSA type should align with cast target | `MPT2010/2011` |

### 17.3 Call families

| Op | Contract | Main enforcement |
|---|---|---|
| `call` / `call_void` | callee must resolve; arity must match; arg types must match params | `MPT2001..2005` |
| `suspend.call` | same as call family; extra v0.1 restriction in callable-target forms | `MPT2001..2005`, `MPT1030` |
| `call.indirect` / `call_void.indirect` | no direct callee sid lookup; argument ownership/move rules still apply | ownership call-mode checks + downstream |
| `try` | lowered like call path for callee resolution; rely on call compatibility + downstream | resolve/lowering + downstream |

Ownership-mode checks for call args:
- Param typed `borrow` => arg must be `borrow` or `mutborrow`.
- Param typed `mutborrow` => arg must be `mutborrow`.
- By-value params => arg must not be borrow-handle.
- Violations => `MPO0004`.

### 17.4 Struct/enum constructors and field ops

| Op | Contract | Main enforcement |
|---|---|---|
| `new` | target must be struct; provided fields exactly match declared fields | `MPT2021`, `MPT2022`, `MPT2016..2020` |
| `enum.new` | variant must exist for target enum (including `TOption`/`TResult` variants) and fields must match | `MPT2023..2027`, `MPT2016..2020` |
| `getfield` | object must be borrow/mutborrow struct; field must exist | `MPT2006..2009`, `MPHIR01` |
| `setfield` | object must be mutborrow | `MPHIR02` |

Projection result semantics (ownership layer):
- `getfield` over Copy-like field => by-value result expected.
- `getfield` over move-only strong handle => borrow/mutborrow projection expected.
- `getfield` over weak handle => weak clone style result expected.
- Mismatches => `MPO0004`.

### 17.5 Borrow/share/weak ops

| Op | Contract | Main enforcement |
|---|---|---|
| `borrow.shared`, `borrow.mut` | source value must satisfy borrow-state rules | `MPO0011` + borrow-state machine |
| `share`, `clone.shared`, `clone.weak`, `weak.downgrade`, `weak.upgrade` | no dedicated sema family checker; checked through ownership/type context and backend use | ownership + downstream |
| borrow values in `phi` | forbidden | `MPO0102` / `MPHIR` path |
| borrow values across blocks | forbidden | `MPO0101` |
| returning borrow from fn | forbidden | `MPHIR03` |

### 17.6 Raw pointer ops

| Op | Contract | Main enforcement |
|---|---|---|
| `ptr.null`, `ptr.addr`, `ptr.from_addr`, `ptr.add`, `ptr.load`, `ptr.store` | must be in unsafe context (`unsafe fn` or `unsafe {}`) | `MPS0024` |
| unsafe fn call | must be in unsafe context | `MPS0025` |

### 17.7 Array op matrix

| Op | Ownership contract on receiver | Extra semantic contract | Main enforcement |
|---|---|---|---|
| `arr.new<T>` | n/a | element type cannot be borrow type | `MPO0003` |
| `arr.len` | receiver must be borrow/mutborrow | — | `MPO0004` |
| `arr.get` | receiver must be borrow/mutborrow | projection result type must match element ownership model | `MPO0004` |
| `arr.set` | receiver must be unique/mutborrow | stored value must not be borrow escape | `MPO0004`, `MPO0003` |
| `arr.push` | receiver must be unique/mutborrow | stored value must not be borrow escape | `MPO0004`, `MPO0003` |
| `arr.pop` | receiver treated as mutating target | — | `MPO0004` |
| `arr.slice` | receiver must be borrow/mutborrow | — | `MPO0004` |
| `arr.contains` | receiver must be borrow/mutborrow | elem type must implement `eq` | `MPO0004`, `MPT1023` |
| `arr.sort` | mutating receiver (unique/mutborrow) | elem type must implement `ord` | `MPO0004`, `MPT1023` |
| `arr.map/filter/reduce/foreach` | receiver must be borrow/mutborrow (`foreach` void path checked too) | callable compatibility mostly downstream today | `MPO0004` + downstream |

### 17.8 Map op matrix

| Op | Ownership contract on receiver | Extra semantic contract | Main enforcement |
|---|---|---|---|
| `map.new<K,V>` | n/a | K must satisfy `hash` + `eq`; K/V cannot be borrow type | `MPT1023`, `MPO0003` |
| `map.len` | receiver must be borrow/mutborrow | — | `MPO0004` |
| `map.get` | receiver must be borrow/mutborrow | map value type must be Dupable; result must be `TOption<V>` | `MPO0103`, `MPO0004` |
| `map.get_ref` | receiver must be borrow/mutborrow | projection result type must follow ownership projection rules | `MPO0004` |
| `map.set` | receiver must be unique/mutborrow | key/value cannot be borrow escapes | `MPO0004`, `MPO0003` |
| `map.delete` / `map.delete_void` | mutating receiver (unique/mutborrow) | — | `MPO0004` |
| `map.contains_key` | receiver must be borrow/mutborrow | — | `MPO0004` |
| `map.keys` / `map.values` | receiver must be borrow/mutborrow | — | `MPO0004` |

### 17.9 String / builder / JSON matrix

| Op | Ownership contract | Main enforcement |
|---|---|---|
| `str.len`, `str.slice`, `str.bytes` | input `s` must be borrow/mutborrow | `MPO0004` |
| `str.eq` | both `a`, `b` must be borrow/mutborrow | `MPO0004` |
| `str.concat` | operands tracked as value consumers; downstream type compatibility | ownership + downstream |
| `str.parse_*` | currently limited dedicated static checks in ownership/typecheck layers | downstream/backend checks |
| `str.builder.new` | creates builder handle | downstream |
| `str.builder.append_*` | builder target is mutating => unique/mutborrow required | `MPO0004` |
| `str.builder.build` | builder target is mutating/consuming boundary | `MPO0004` + consumption model |
| `json.encode` / `json.decode` | generic type + value/string compatibility mostly downstream | sema type-id existence + downstream |

### 17.10 GPU matrix

| Op | Contract | Current enforcement |
|---|---|---|
| `gpu.thread_id`, `gpu.workgroup_id`, `gpu.workgroup_size`, `gpu.global_id` | `dim` argument required in parser | parser/lowering; limited dedicated type family checks |
| `gpu.buffer_load`, `gpu.buffer_len`, `gpu.buffer_store` | parser enforces type arg + keys | lowering + downstream/backend |
| `gpu.shared<count,T>` | parser enforces count/type syntax | lowering + backend |
| `gpu.launch`, `gpu.launch_async` | strict key order: `device,kernel,grid,block,args` | parser strict key checks |
| `gpu.barrier` | void synchronization op | parser/lowering |

When adding GPU semantics, extend sema/ownership checks explicitly; current static checking is lighter than core collection/struct ops.

### 17.11 Phi/control-flow contracts

| Construct | Contract | Enforced by |
|---|---|---|
| `phi` | incoming values must dominate use; borrow values forbidden | `MPS0002/0003`, `MPO0102`, `MPHIR` |
| block terminator | every block must end with one terminator | parser + HIR/MPIR structure |
| `switch` arms/default | block ids and constant forms must be valid | parser + MPIR verifier |

---

## 18) Move/consume semantics matrix (ownership checker model)

These are the values the ownership checker treats as consumed (move candidates), per `op_consumed_locals` / `op_void_consumed_locals`.

### 18.1 Always-consume patterns

- `move { v }` consumes `v`
- `share { v }` consumes `v`
- `new` consumes each field value
- `enum.new` consumes each variant payload value
- `callable.capture` consumes each captured value
- `str.concat` consumes `a`, `b`
- `str.builder.build` consumes builder value
- `arr.reduce` consumes `init`
- `setfield` consumes assigned `val`
- `arr.set`/`arr.push` consume `val`
- `map.set` consumes `key` and `val`
- `ptr.store` / `gpu.buffer_store` consume stored value
- `ret %x` consumes `%x` for move-only tracking

### 18.2 Conditional consume patterns

Calls (`call`, `call_void`, `suspend.call`, indirect variants):
- consumption is inferred from callee param modes when available:
  - by-value move param => consume arg
  - by-value copy param => not consumed as move
  - borrow/mutborrow param => not consumed as move
- if callee param metadata unavailable (indirect/unknown), fallback uses local type move-only heuristics.

### 18.3 Explicitly non-consuming by ownership model

Many read/projection/math ops do not directly mark args as consumed in ownership analysis (though they still must satisfy mode constraints), e.g.:
- arithmetic/cmp families,
- `getfield`, `arr.get`, `map.get_ref`,
- `borrow.shared`, `borrow.mut`,
- most parse/read-only intrinsics.

Use this when diagnosing “use of moved value” (`MPO0007`) vs “move while borrowed” (`MPO0011`).
