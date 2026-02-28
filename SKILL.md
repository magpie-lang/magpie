---
name: magpie-engineer
description: Extremely detailed, binary-first guide for authoring Magpie (.mp), operating with only the compiler CLI/binary, and debugging parse/resolve/type/HIR/ownership/MPIR/codegen/link/runtime failures. Includes grammar/semantics, ASCII diagrams, and error-code fix playbooks with incorrect/fixed examples.
---

# Magpie Engineer Skill (Binary-first: works without source access)

Use this skill whenever you write/debug Magpie `.mp` programs or triage Magpie compiler failures.

**Assumption for this skill:** the agent may only have access to the compiler binary/CLI outputs (no source tree).

---

## 0) Non-negotiable rules

1. **Binary-first operation:** treat compiler diagnostics and emitted artifacts as truth.
   - Use `magpie explain <CODE>` and `--output json` when available.
   - Do **not** rely on outdated design docs.
2. **Diagnose by pipeline stage + diagnostic code first**, then patch the program.
3. **Smallest reproducer first** (one tiny `.mp` file, then grow).
4. **Change one dimension at a time** (syntax, then types, then ownership, then backend).
5. **Prove fixes with commands and exit codes.**

---

## 1) Binary-only truth stack (no source required)

When source is unavailable, use this evidence hierarchy:

1. Compiler exit status + diagnostic codes (`--output json` preferred).
2. `magpie explain <CODE>` output for immediate remediation guidance.
3. Emitted artifacts (`.mpir`, `.ll`, `.mpdbg`, graphs) to localize failing stage.
4. Minimal reproducer program that isolates one failing behavior.
5. Only if source becomes available: inspect implementation details.

Binary-only triage loop:

1. Re-run build with machine-readable output and focused emits.
2. Group diagnostics by code prefix (`MPP`, `MPS`, `MPT`, `MPO`, `MPL`, ...).
3. Fix earliest-stage hard errors first.
4. Rebuild after each small fix.
5. Stop only when clean build + expected artifact set are produced.

### Diagnostic Triage Flowchart

```
Got an error code?
       |
       v
  +-----------+
  | Code      |
  | prefix?   |
  +-----+-----+
        |
   +----+----+----+----+----+
   |         |         |    |
  MPP*?     MPS*?    MPT*? MPHIR*?
   |         |         |    |
   v         v         v    v
Check      Check      Check  Check borrow
header     imports,   type   on getfield/
order,     SSA        match, setfield,
block      single-    trait  no borrow
terms,     def,       impls, returns,
key        use-       const  borrow not
names,     before-    suffix in phi,
commas     def,       match, borrow not
           domina-    ctor   crossing
           nce,       field  blocks
           unsafe     comp-
           context    lete
                |          |
               MPO*?      MPL*?
                |          |
                v          v
           Check         Check link
           ownership     toolchain,
           modes,        emit flags,
           move          token budget,
           ordering,     lint/policy
           use-after-
           move,
           borrow
           scope
```

---

## 2) Optional deep-reference map (use only if source is available)

### Language surface + grammar
- `crates/magpie_lex/src/lib.rs` -- tokenization, keyword/op spelling
- `crates/magpie_parse/src/lib.rs` -- actual grammar and key requirements
- `crates/magpie_csnf/src/lib.rs` -- canonical pretty-printer (ground truth for canonical source form)

### AST/HIR/type model
- `crates/magpie_ast/src/lib.rs` -- AST node set
- `crates/magpie_types/src/lib.rs` -- primitive/handle/base type semantics
- `crates/magpie_hir/src/lib.rs` -- HIR model + HIR verifier invariants

### Semantic legality
- `crates/magpie_sema/src/lib.rs` -- resolve/lower/type/trait/v0.1 restrictions
- `crates/magpie_own/src/lib.rs` -- ownership/borrow/move/send rules
- `crates/magpie_mpir/src/lib.rs` -- MPIR model + verifier

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

## 3) CLI and execution workflow

Core commands:

- CLI rule: place global flags before the subcommand (e.g., `magpie --entry src/main.mp --output json build`).
- Build:
  - `cargo run -p magpie_cli -- --entry <path> --emit <kinds> build`
- Run:
  - `cargo run -p magpie_cli -- --entry <path> --emit <kinds> run`
- Parse only:
  - `cargo run -p magpie_cli -- --entry <path> parse`
- Test crate:
  - `cargo test -p <crate>`
- Integration fixtures:
  - `cargo test --test integration_test`
- Full sweep:
  - `cargo test --workspace`

Useful emit kinds from driver planning:
- `llvm-ir`, `llvm-bc`, `object`, `asm`, `spv`, `exe`, `shared-lib`, `mpir`, `mpd`, `mpdbg`, `symgraph`, `depsgraph`, `ownershipgraph`, `cfggraph`

---

## 4) Full lexical model (from lexer)

### Comments
- `; ...` line comment
- `;;` doc comment token (`DocComment`) -- attached to the declaration that follows

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

## 5) Full grammar skeleton (parser-authoritative)

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

Doc comments (`;;`) may precede declarations and are attached.

### Functions
```ebnf
fn_decl         := "fn" fn_name "(" params ")" "->" type meta_opt blocks
async_fn_decl   := "async" "fn" ...
unsafe_fn_decl  := "unsafe" "fn" ...
gpu_fn_decl     := "gpu" "fn" ... "target" "(" ident ")" meta_opt blocks
```

### Function meta
```ebnf
meta_opt := epsilon | "meta" "{" meta_entry* "}"
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
attrs_opt   := epsilon | "attrs" "{" (ident "=" string_lit ("," ... )*)? "}"
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

## 6) Type grammar + type semantics

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

## 5.6 Builtin type trait requirements

| Type               | hash | eq  | ord | Notes                                       |
|--------------------|------|-----|-----|---------------------------------------------|
| `i*` / `u*` / `f*` | yes | yes | yes | All primitives satisfy all built-in traits  |
| `bool`             | yes  | yes | yes | Built-in                                    |
| `Str`              | yes  | yes | yes | Built-in -- no explicit `impl` needed       |
| User `heap struct` | no   | no  | no  | Must provide explicit `impl` declarations   |
| User `heap enum`   | no   | no  | no  | Must provide explicit `impl` declarations   |

**Key point:** `Str` has built-in `hash`, `eq`, and `ord` -- `Map<Str, V>` requires no explicit `impl hash for Str` or `impl eq for Str`. Only user-defined struct/enum types require explicit impl declarations.

---

## 7) Full opcode surface (parser + CSNF canonical spelling)

Use exactly these names and keys.

## 6.1 Value-producing ops

### Constants
- `const.<Type> <literal>`
  - The type suffix **must match** the declared SSA type exactly. Examples:
    - `const.i32 0` for an `i32` result
    - `const.i64 0` for an `i64` result
    - `const.bool true` for a `bool` result
    - `const.f64 1.0` for an `f64` result
  - Mismatch between declared type and const suffix causes `MPT2014`/`MPT2015` type errors.

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
- `getfield { obj=V, field=name }` (keys accepted in **any order** -- parser is flexible)
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
- `setfield { obj=V, field=name, val=V }` (**must use `val=`, not `value=`; keys in any order**)
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

## 6.5 Key-ordering summary for ops

| Op family          | Key order required? | Notes                                           |
|--------------------|---------------------|-------------------------------------------------|
| `getfield`         | No -- any order     | Parser accepts keys in any order                |
| `setfield`         | No -- any order     | Uses `val=` key (not `value=`)                  |
| `gpu.launch`       | Yes -- strict       | `device, kernel, grid, block, args`             |
| `gpu.launch_async` | Yes -- strict       | `device, kernel, grid, block, args`             |
| All other ops      | Stable recommended  | Keys ignored during lowering; order preserved   |

---

## 8) Semantic invariants by stage

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
- Parse/JSON migration checks:
  - result shape mismatch `MPT2033`
  - input string-handle mismatch/unknown `MPT2034`
  - `json.encode<T>` value-type mismatch/unknown `MPT2035`

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

## 9) Unsafe, async, and control-flow semantics

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

**Important:** After async lowering, `is_async` **remains `true`** in the HIR. Downstream
verifiers use this flag to skip SSA domination checks that would otherwise fire on the
split-block coroutine state machine structure. Do not expect `is_async` to become `false`
after lowering -- the flag is a permanent marker for verifier bypass.

Note: diagnostic constant `MPAS0001` exists in diag codes, but current code path does not emit it directly.

---

## 10) Canonical .mp authoring templates

## 9.1 Minimal valid module (returning i32)

```mp
module demo.main
exports { @main }
imports { }
digest "0000000000000000"

fn @main() -> i32 {
bb0:
  ret const.i32 0
}
```

## 9.2 Minimal valid module (returning i64)

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

**The `const` suffix must match the declared return type exactly.**

## 9.3 Borrow-safe struct mutation/read split

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

## 9.4 Trait impl for user struct (Map key)

```mp
;; hash impl for TKey -- required for Map<TKey, V>
fn @hash_key(%k: borrow TKey) -> u64 {
bb0:
  ret const.u64 0
}
impl hash for TKey = @hash_key

;; eq impl for TKey -- required for Map<TKey, V>
fn @eq_key(%a: borrow TKey, %b: borrow TKey) -> bool {
bb0:
  ret const.bool true
}
impl eq for TKey = @eq_key
```

Note: `Map<Str, V>` does NOT need these -- `Str` has built-in hash/eq.

## 9.5 Array with borrow receiver

```mp
fn @arr_demo() -> i64 {
bb0:
  %arr: Array<i64> = arr.new<i64> { cap=const.i64 8 }
  %arrm: mutborrow Array<i64> = borrow.mut { v=%arr }
  arr.push { arr=%arrm, val=const.i64 42 }
  br bb1

bb1:
  %arrb: borrow Array<i64> = borrow.shared { v=%arr }
  %len: i64 = arr.len { arr=%arrb }
  ret %len
}
```

Array mutation ops (`arr.push`, `arr.set`, `arr.sort`) require a `mutborrow` receiver.
Array read ops (`arr.len`, `arr.get`, `arr.contains`) require a `borrow` or `mutborrow` receiver.

## 9.6 Map with Str keys (no impl needed)

```mp
fn @map_demo() -> i64 {
bb0:
  %m: Map<Str, i64> = map.new<Str, i64> { }
  %mb: mutborrow Map<Str, i64> = borrow.mut { v=%m }
  map.set { map=%mb, key=const.Str "hello", val=const.i64 1 }
  br bb1

bb1:
  %mr: borrow Map<Str, i64> = borrow.shared { v=%m }
  %len: i64 = map.len { map=%mr }
  ret %len
}
```

## 9.7 Enum pattern with TOption

```mp
fn @maybe() -> i64 {
bb0:
  %o: TOption<i64> = enum.new<Some> { v=const.i64 99 }
  %is_some: bool = enum.is<Some> { v=%o }
  cbr %is_some bb1 bb2

bb1:
  %val: i64 = enum.payload<Some> { v=%o }
  ret %val

bb2:
  ret const.i64 0
}
```

## 9.8 Unsafe raw pointer block

```mp
unsafe fn @raw_demo() -> i64 {
bb0:
  %p: rawptr<i64> = ptr.null<i64>
  ret const.i64 0
}
```

Or inline unsafe block in a normal function:

```mp
fn @inline_unsafe() -> i64 {
bb0:
  unsafe {
    %p: rawptr<i64> = ptr.null<i64>
  }
  ret const.i64 0
}
```

---

## 11) Pipeline model for debugging

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

### Pipeline Stage -> Diagnostic Code Mapping

```
Stage                     | Primary Codes           | Secondary / Cross-stage
--------------------------+-------------------------+----------------------------
stage1 (lex/parse)        | MPP0001, MPP0002        | --
stage2 (resolve)          | MPS0001-MPS0006         | MPF0001, MPS0020-0025
stage3 (typecheck)        | MPT0001-MPT0003         | MPT1005, MPT1020, MPT1021
                          | MPT2001-MPT2035         | MPT1023, MPT1030, MPT1200
stage3_5 (async lower)    | MPAS0001 (reserved)     | is_async stays TRUE
stage4 (verify HIR)       | MPHIR01, MPHIR02        | MPHIR03, MPS0001-0003
stage5 (ownership)        | MPO0003, MPO0004        | MPO0007, MPO0011
                          | MPO0101, MPO0102        | MPO0103, MPO0201
stage6 (lower MPIR)       | MPS0008-0016            | MPM0001
stage7 (verify MPIR)      | MPS0008-0016            | MPS0001-0003
stage8-9 (ARC)            | MPS0014 (if pre-ARC)    | --
stage10 (codegen)         | MPG* (backend-specific) | --
stage11 (link)            | MPLINK01, MPLINK02      | MPL0001, MPL0002
stage12 (mms)             | --                      | MPL0801, MPL0802
lint pass                 | MPL2001-MPL2021         | (any stage, policy-driven)
```

---

## 12) Ownership model deep-dive

### Ownership State Machine

```
             +------------------------+
             |        Unique          |  <-- initial state after `new`
             |     (sole owner)       |
             +---+--------+-------+---+
                 |        |       |
        share{}  |        |       | borrow.mut{}
                 |        |       |
                 v        |       v
         +----------+     |  +------------------+
         |  Shared  |     |  |    MutBorrow     |
         |(ref-cnt) |     |  | (exclusive r/w)  |
         +----+-----+     |  +------------------+
              |           |    (block-scoped only;
              | clone      |     ends at block exit)
              | .shared{}  |
              v            | borrow.shared{}
         +----------+      |
         |  Shared  |      v
         |  (copy)  |  +----------+
         +----------+  |  Borrow  |
                       | (shared  |
                       |   r/o)   |
                       +----------+
                       (block-scoped only;
                        ends at block exit)

Key rules:
  Unique --share{}--> [Unique CONSUMED, new Shared created]
  Unique --borrow.mut{}--> MutBorrow [scoped to current block]
  Unique --borrow.shared{}--> Borrow  [scoped to current block]
  Shared --clone.shared{}--> new Shared [original survives]
  Borrow / MutBorrow: CANNOT cross block boundaries
  Borrow / MutBorrow: CANNOT appear in phi nodes
  Borrow / MutBorrow: CANNOT be returned from functions
```

### Borrow Lifecycle Per Block

```
  Block entry
      |
      |  %x: T = ...             (Unique value defined or received as param)
      |
      |  %xb: borrow T =         (Borrow created -- valid from this point
      |    borrow.shared{v=%x}    to the END of THIS block only)
      |
      |  getfield{obj=%xb,...}   (OK: use borrow in same block)
      |  arr.len{arr=%xb}        (OK: read op in same block)
      |
      |  [br / cbr / ret]        (block exit -- borrow scope ends here)
      |
      v
  Successor blocks
      |
      |  %xb is DEAD here. Cannot be referenced.
      |  If borrow is needed again, call borrow.shared{} again.
      |
      v
  Pattern for multi-block borrow use:
    bb0: %xb = borrow.shared{v=%x}; use %xb; br bb1
    bb1: %xb2 = borrow.shared{v=%x}; use %xb2; br bb2
    bb2: ...
```

### Mutation vs Read receiver summary

```
  Operation class               Required receiver mode
  -------------------------------------------------------
  arr.push / arr.set /          mutborrow Array<T>
  arr.sort / arr.pop /            (or unique)
  arr.foreach (mutating)

  arr.len / arr.get /           borrow Array<T>
  arr.slice / arr.contains /      (or mutborrow)
  arr.map / arr.filter /
  arr.reduce

  map.set / map.delete /        mutborrow Map<K,V>
  map.delete_void                 (or unique)

  map.len / map.get /           borrow Map<K,V>
  map.get_ref /                   (or mutborrow)
  map.contains_key /
  map.keys / map.values

  setfield                      mutborrow T

  getfield                      borrow T  OR  mutborrow T

  str.builder.append_*          mutborrow TStrBuilder
  str.builder.build               (or unique, consuming)
```

---

## 13) Error-driven fix playbooks

## 11.1 Parse/lex (`MPP*`)

- Check header order and block terminators first.
- Check key names for strict ops (note: `getfield`/`setfield` accept keys in any order).
- Confirm function names have `@`, locals have `%`, types use `T` prefix where required.
- Confirm commas separate op arguments within `{ }`.
- Confirm `gpu.launch`/`gpu.launch_async` use strict key order: `device, kernel, grid, block, args`.

## 11.2 Resolve (`MPS0001..0006`, `MPS0024/25`, `MPF0001`)

- Fix imports/module path first, then symbol conflicts.
- Wrap ptr ops or unsafe calls in `unsafe {}` or mark caller `unsafe fn`.
- Add extern attrs for rawptr returns.

## 11.3 Type (`MPT*`)

- For constructor errors, verify field completeness and exact field names.
- For binary op errors, unify operand types before op.
- For trait requirement errors (`MPT1023`), add explicit impl function and impl declaration.
- Remember: `Str` requires no explicit hash/eq impl -- only user-defined struct/enum types do.
- `const` suffix must match declared SSA type (e.g., `const.i32 0` for `i32`, `const.i64 0` for `i64`).

## 11.4 Ownership (`MPO*`)

- Never let borrow handles cross block boundaries.
- End mutborrow usage in one block, branch, then create shared borrow in successor.
- For `map.get` on non-Dupable V, use ref form (`map.get_ref`) + explicit handling.
- Array/map mutation ops require `mutborrow` receiver -- obtain it with `borrow.mut{}`.
- Array/map read ops require at least `borrow` receiver -- obtain it with `borrow.shared{}`.

## 11.5 MPIR/backend/link (`MPS*`, `MPG*`, `MPLINK*`, `MPL0002`)

- Emit `mpir` and `llvm-ir`, inspect for missing/invalid artifacts.
- If requested emit file is missing, treat as hard failure (`MPL0002` path).
- Check runtime symbol declaration/definition parity when link fails.

---

## 14) How to implement a new language feature correctly

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

## 15) Comprehensive testing strategy (language + semantics)

## 13.1 Fast local loop

1. Reproducer file in `tests/fixtures/`.
2. `cargo test -p magpie_parse`
3. `cargo test -p magpie_sema`
4. `cargo test -p magpie_own`
5. `cargo test -p magpie_mpir`

## 13.2 End-to-end compile/run

- Build fixture with multiple emits:
  - `cargo run -p magpie_cli -- --entry tests/fixtures/feature_harness.mp --emit mpir,llvm-ir,mpdbg,exe build`
- Run:
  - `cargo run -p magpie_cli -- --entry tests/fixtures/feature_harness.mp run`

## 13.3 Regression expectations

Every bug fix should add one of:
- parser test,
- sema/ownership unit test,
- integration fixture,
- codegen assertion test,
- runtime execution check.

---

## 16) High-value gotchas (easy to miss)

1. **Header order is strict** in parser (`module`, `exports`, `imports`, `digest` -- exactly this order).
2. **Call argument keys are not semantic** today; order is what lowering preserves.
3. `setfield` uses `val=` key (not `value=`).
4. `getfield`/`setfield` accept keys in **any order** -- no strict key ordering requirement for these ops.
5. Borrows cannot appear in phi or cross blocks. They are single-block scoped.
6. `map.get` by-value result requires Dupable map value type; use `map.get_ref` for non-Dupable.
7. `TOption`/`TResult` reject `shared`/`weak` ownership prefix.
8. Arc op tokens exist in lexer but are not surface source operations.
9. Async lowering rewrites function shape. `is_async` **stays `true`** after lowering so verifiers can skip SSA domination checks for async coroutine state machines.
10. **`const` suffix must match declared SSA type** -- `const.i32 0` for `i32`, `const.i64 0` for `i64`. Mismatched suffix causes type errors.
11. **`Str` has built-in `hash`/`eq`/`ord`** -- `Map<Str, V>` needs no explicit impl declarations.
12. Array/map mutation ops require `mutborrow` receiver; read ops require `borrow` or `mutborrow`.
13. `;;` is the doc comment syntax (NOT `;;;`).
14. `gpu.launch`/`gpu.launch_async` require strict key order: `device, kernel, grid, block, args`.

---

## 17) Production-ready completion checklist

Before declaring done:

- [ ] Reproducer failed before and passes after.
- [ ] No unresolved diagnostics for touched path.
- [ ] Changed crates have targeted tests passing.
- [ ] Integration fixture path passes (`integration_test` when surface changed).
- [ ] End-to-end compile (and run when relevant) completed.
- [ ] If codegen/runtime touched: validate emitted `.ll` + binary execution.
- [ ] Output includes exact commands + outcomes + residual risks.

---

## 18) Response protocol for failures/fixes

Always report in this order:
1. Stage + diagnostic code(s)
2. Root cause (source-grounded)
3. Exact files/functions changed
4. Validation commands and results
5. Remaining risk / next follow-up

Never claim "fixed" without command evidence.

---

## 19) Per-op type/ownership contract matrix (implementation-grounded)

Use this section when writing or reviewing `.mp` operations. It states what is currently enforced by code (not wishful design).

### 17.1 Interpreting result types

- In source, each value op appears in an SSA assignment:
  - `%dst: DeclTy = op ...`
- So the immediate "result type" is **declared by `DeclTy`**.
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
| `getfield` | object must be borrow/mutborrow struct; field must exist; keys in any order | `MPT2006..2009`, `MPHIR01` |
| `setfield` | object must be mutborrow; uses `val=` key; keys in any order | `MPHIR02` |

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
| `arr.len` | receiver must be borrow/mutborrow | -- | `MPO0004` |
| `arr.get` | receiver must be borrow/mutborrow | projection result type must match element ownership model | `MPO0004` |
| `arr.set` | receiver must be unique/mutborrow | stored value must not be borrow escape | `MPO0004`, `MPO0003` |
| `arr.push` | receiver must be unique/mutborrow | stored value must not be borrow escape | `MPO0004`, `MPO0003` |
| `arr.pop` | receiver treated as mutating target | -- | `MPO0004` |
| `arr.slice` | receiver must be borrow/mutborrow | -- | `MPO0004` |
| `arr.contains` | receiver must be borrow/mutborrow | elem type must implement `eq` | `MPO0004`, `MPT1023` |
| `arr.sort` | mutating receiver (unique/mutborrow) | elem type must implement `ord` | `MPO0004`, `MPT1023` |
| `arr.map/filter/reduce/foreach` | receiver must be borrow/mutborrow (`foreach` void path checked too) | callable compatibility mostly downstream today | `MPO0004` + downstream |

### 17.8 Map op matrix

| Op | Ownership contract on receiver | Extra semantic contract | Main enforcement |
|---|---|---|---|
| `map.new<K,V>` | n/a | K must satisfy `hash` + `eq`; K/V cannot be borrow type | `MPT1023`, `MPO0003` |
| `map.len` | receiver must be borrow/mutborrow | -- | `MPO0004` |
| `map.get` | receiver must be borrow/mutborrow | map value type must be Dupable; result must be `TOption<V>` | `MPO0103`, `MPO0004` |
| `map.get_ref` | receiver must be borrow/mutborrow | projection result type must follow ownership projection rules | `MPO0004` |
| `map.set` | receiver must be unique/mutborrow | key/value cannot be borrow escapes | `MPO0004`, `MPO0003` |
| `map.delete` / `map.delete_void` | mutating receiver (unique/mutborrow) | -- | `MPO0004` |
| `map.contains_key` | receiver must be borrow/mutborrow | -- | `MPO0004` |
| `map.keys` / `map.values` | receiver must be borrow/mutborrow | -- | `MPO0004` |

### 17.9 String / builder / JSON matrix

| Op | Ownership contract | Main enforcement |
|---|---|---|
| `str.len`, `str.slice`, `str.bytes` | input `s` must be borrow/mutborrow | `MPO0004` |
| `str.eq` | both `a`, `b` must be borrow/mutborrow | `MPO0004` |
| `str.concat` | operands tracked as value consumers; downstream type compatibility | ownership + downstream |
| `str.parse_*` | input must be `Str`/`borrow Str`; result must be legacy primitive or `TResult<ok, err>` | `MPT2033`, `MPT2034` |
| `str.builder.new` | creates builder handle | downstream |
| `str.builder.append_*` | builder target is mutating => unique/mutborrow required | `MPO0004` |
| `str.builder.build` | builder target is mutating/consuming boundary | `MPO0004` + consumption model |
| `json.encode` / `json.decode` | `encode`: value must match generic `T`; `decode`: input must be string handle; result must be legacy/rawptr or `TResult<rawptr<...>, err>` | `MPT2033`, `MPT2034`, `MPT2035` |

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

## 20) Move/consume semantics matrix (ownership checker model)

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

Use this when diagnosing "use of moved value" (`MPO0007`) vs "move while borrowed" (`MPO0011`).

---

## 21) Complete Error Code Quick Reference Table

| Code       | Stage         | Short description                                     |
|------------|---------------|-------------------------------------------------------|
| MPP0001    | parse         | Source I/O / lexer-level read issue                   |
| MPP0002    | parse         | Syntax/tokenization error (missing comma, etc.)       |
| MPP0003    | parse         | Artifact emission failure                             |
| MPS0000    | resolve       | Generic module resolution failure                     |
| MPS0001    | resolve/SSA   | Duplicate definition (module path or SSA local)       |
| MPS0002    | resolve/SSA   | Unresolved reference / use-before-def                 |
| MPS0003    | resolve/SSA   | Dominance violation                                   |
| MPS0004    | resolve       | Import/local namespace conflict                       |
| MPS0005    | resolve       | Type import/local type conflict                       |
| MPS0006    | resolve       | Ambiguous import name                                 |
| MPS0008    | MPIR verify   | Invalid CFG target / phi type legality                |
| MPS0009    | MPIR verify   | Duplicate block label                                 |
| MPS0010    | MPIR verify   | Structural/type invariant failure                     |
| MPS0011    | MPIR verify   | Duplicate SSA local in lowering                       |
| MPS0012    | MPIR verify   | Call arity mismatch                                   |
| MPS0013    | MPIR verify   | Expected single arg value / value-shape mismatch      |
| MPS0014    | MPIR verify   | Invalid fn ref in scalar position / arc ops pre-ARC   |
| MPS0015    | MPIR verify   | Invalid fn ref inside list lowered as plain values    |
| MPS0016    | MPIR verify   | Invalid plain-value argument uses fn ref              |
| MPS0017    | MPIR verify   | Invalid branch predicate type                         |
| MPS0020    | resolve       | Duplicate function/global symbol                      |
| MPS0021    | resolve       | Duplicate type symbol in module                       |
| MPS0022    | resolve       | Duplicate @ namespace symbol                          |
| MPS0023    | resolve       | Duplicate `sig` symbol                                |
| MPS0024    | resolve       | ptr.* outside unsafe context                          |
| MPS0025    | resolve       | Unsafe fn call outside unsafe context                 |
| MPF0001    | resolve       | Extern rawptr return missing ownership attr           |
| MPHIR01    | HIR verify    | getfield object must be borrow/mutborrow              |
| MPHIR02    | HIR verify    | setfield object must be mutborrow                     |
| MPHIR03    | HIR verify    | Borrow escapes via return                             |
| MPT0001    | typecheck     | Unknown primitive type                                |
| MPT0002    | typecheck     | shared/weak invalid on TOption                        |
| MPT0003    | typecheck     | shared/weak invalid on TResult                        |
| MPT1005    | typecheck v01 | Value struct contains heap handle                     |
| MPT1020    | typecheck v01 | Value enum deferred in v0.1                           |
| MPT1021    | typecheck v01 | Aggregate type deferred in v0.1                       |
| MPT1023    | typecheck v01 | Missing required trait impl (hash/eq/ord)             |
| MPT1030    | typecheck v01 | suspend.call on non-function target form in v0.1      |
| MPT1200    | typecheck     | Orphan impl                                           |
| MPT2001    | typecheck     | Call arity mismatch                                   |
| MPT2002    | typecheck     | Call arg unknown type                                 |
| MPT2003    | typecheck     | Call arg type mismatch                                |
| MPT2004    | typecheck     | Call target not found                                 |
| MPT2005    | typecheck     | Invalid generic type argument                         |
| MPT2006    | typecheck     | getfield object unknown type                          |
| MPT2007    | typecheck     | getfield requires borrow/mutborrow struct             |
| MPT2008    | typecheck     | getfield target not a struct                          |
| MPT2009    | typecheck     | Missing struct field in getfield                      |
| MPT2010    | typecheck     | cast operand unknown                                  |
| MPT2011    | typecheck     | cast only primitive->primitive                        |
| MPT2012    | typecheck     | Numeric lhs unknown type                              |
| MPT2013    | typecheck     | Numeric rhs unknown type                              |
| MPT2014    | typecheck     | Numeric operands have mismatched types                |
| MPT2015    | typecheck     | Wrong primitive family for numeric op                 |
| MPT2016    | typecheck     | Duplicate field arg in constructor                    |
| MPT2017    | typecheck     | Unknown field in constructor/variant args             |
| MPT2018    | typecheck     | Field value unknown type in constructor               |
| MPT2019    | typecheck     | Field type mismatch in constructor                    |
| MPT2020    | typecheck     | Missing required field in constructor                 |
| MPT2021    | typecheck     | `new` target must be struct                           |
| MPT2022    | typecheck     | Unknown struct target in `new`                        |
| MPT2023    | typecheck     | Invalid variant for TOption                           |
| MPT2024    | typecheck     | Invalid variant for TResult                           |
| MPT2025    | typecheck     | enum.new result type is not enum                      |
| MPT2026    | typecheck     | User enum variant not found                           |
| MPT2027    | typecheck     | enum.new target type must be enum                     |
| MPT2028    | typecheck     | Trait impl parameter count mismatch                   |
| MPT2029    | typecheck     | Trait impl return type mismatch                       |
| MPT2030    | typecheck     | Trait impl first param must be borrow target type     |
| MPT2031    | typecheck     | Trait impl params must both match borrow target       |
| MPT2032    | typecheck     | Impl target function missing                          |
| MPT2033    | typecheck     | Parse/JSON result type shape mismatch                 |
| MPT2034    | typecheck     | Parse/JSON input must be Str/borrow Str               |
| MPT2035    | typecheck     | json.encode<T> value type mismatch                    |
| MPO0003    | ownership     | Borrow escapes scope                                  |
| MPO0004    | ownership     | Wrong ownership mode for mut/read op                  |
| MPO0007    | ownership     | Use after move                                        |
| MPO0011    | ownership     | Move while borrowed                                   |
| MPO0101    | ownership     | Borrow crosses block boundary                         |
| MPO0102    | ownership     | Borrow in phi                                         |
| MPO0103    | ownership     | map.get requires Dupable V                            |
| MPO0201    | ownership     | Spawn/send capture rule violation                     |
| MPM0001    | MPIR lower    | MPIR lowering produced no modules                     |
| MPLINK01   | link          | Primary link path failed                              |
| MPLINK02   | link          | Fallback link also unavailable                        |
| MPL0001    | link/emit     | Unknown emit kind                                     |
| MPL0002    | link/emit     | Requested artifact missing                            |
| MPL0801    | llm/budget    | LLM budget too small                                  |
| MPL0802    | llm/budget    | Tokenizer fallback                                    |
| MPL2001    | lint          | Oversized function body                               |
| MPL2002    | lint          | Unused/dead symbol                                    |
| MPL2003    | lint          | Unnecessary borrow                                    |
| MPL2005    | lint          | Empty block                                           |
| MPL2007    | lint          | Unreachable code                                      |
| MPL2020    | lint          | Monomorphization pressure too high                    |
| MPL2021    | lint          | Mixed generics mode conflict                          |

---

## 22) Error-code cookbook (binary-only, with incorrect/fixed examples)

Use this whenever you only have the compiler binary.

### 22.0 Rapid workflow for any code

1. Reproduce:
   - `magpie --entry <file.mp> --output json --emit mpir,llvm-ir,mpdbg build`
2. Explain:
   - `magpie explain <CODE>`
3. Apply minimal fix.
4. Rebuild until the code disappears.

For each code below, **Bad** is a minimal failing pattern and **Fix** is the smallest stable correction.

### 22.1 Parse / IO / artifact codes

#### MPP0001 -- Source I/O / lexer-level read issue

Bad:
```bash
magpie --entry ./missing/main.mp build
```

Fix:
```bash
mkdir -p src
cat > src/main.mp <<'MP'
module demo.main
exports { @main }
imports { }
digest "0000000000000000"

fn @main() -> i32 {
bb0:
  ret const.i32 0
}
MP
magpie --entry ./src/main.mp build
```

#### MPP0002 -- Syntax/tokenization error

Bad (missing comma between args):
```mp
module demo.main
exports { @main }
imports { }
digest "0000000000000000"

fn @main() -> i64 {
bb0:
  %x: i64 = i.add { lhs=const.i64 1 rhs=const.i64 2 }
  ret %x
}
```

Fix:
```mp
module demo.main
exports { @main }
imports { }
digest "0000000000000000"

fn @main() -> i64 {
bb0:
  %x: i64 = i.add { lhs=const.i64 1, rhs=const.i64 2 }
  ret %x
}
```

#### MPP0003 -- Artifact emission failure

Bad:
```bash
magpie --entry src/main.mp --emit llvm-ir --output json build
# writing artifact fails due to unwritable destination/disk issues
```

Fix:
```bash
# ensure writable working/output dirs and free disk, then rebuild
magpie --entry src/main.mp --emit llvm-ir --output json build
```

### 22.2 Resolve / SSA / unsafe-context codes (MPS*)

#### MPS0001 -- duplicate definition (module path or SSA local)

Bad:
```mp
fn @main() -> i64 {
bb0:
  %x: i64 = const.i64 1
  %x: i64 = const.i64 2
  ret %x
}
```

Fix:
```mp
fn @main() -> i64 {
bb0:
  %x: i64 = const.i64 1
  %y: i64 = const.i64 2
  ret %y
}
```

#### MPS0002 -- unresolved reference / use-before-def

Bad:
```mp
fn @main() -> i64 {
bb0:
  %z: i64 = i.add { lhs=%x, rhs=const.i64 1 }
  ret %z
}
```

Fix:
```mp
fn @main() -> i64 {
bb0:
  %x: i64 = const.i64 10
  %z: i64 = i.add { lhs=%x, rhs=const.i64 1 }
  ret %z
}
```

#### MPS0003 -- dominance violation

Bad:
```mp
fn @f(%c: bool) -> i64 {
bb0:
  cbr %c bb1 bb2
bb1:
  %x: i64 = const.i64 1
  br bb3
bb2:
  br bb3
bb3:
  %y: i64 = i.add { lhs=%x, rhs=const.i64 1 }
  ret %y
}
```

Fix (use phi to merge definitions from both branches):
```mp
fn @f(%c: bool) -> i64 {
bb0:
  cbr %c bb1 bb2
bb1:
  %x1: i64 = const.i64 1
  br bb3
bb2:
  %x2: i64 = const.i64 2
  br bb3
bb3:
  %x: i64 = phi i64 { [bb1:%x1], [bb2:%x2] }
  %y: i64 = i.add { lhs=%x, rhs=const.i64 1 }
  ret %y
}
```

#### MPS0004 / MPS0005 / MPS0006 -- import conflict / ambiguity

Bad:
```mp
imports { a.mod::{@foo}, b.mod::{@foo} }
```

Fix:
```mp
imports { a.mod::{@foo_a}, b.mod::{@foo_b} }
; or call fully-qualified symbols instead of conflicting short names
```

#### MPS0008 -- invalid CFG target

Bad:
```mp
fn @main() -> i64 {
bb0:
  br bb9
}
```

Fix:
```mp
fn @main() -> i64 {
bb0:
  br bb1
bb1:
  ret const.i64 0
}
```

#### MPS0009 -- duplicate block label

Bad:
```mp
fn @main() -> i64 {
bb0:
  br bb0
bb0:
  ret const.i64 0
}
```

Fix:
```mp
fn @main() -> i64 {
bb0:
  br bb1
bb1:
  ret const.i64 0
}
```

#### MPS0010 -- structural/type invariant failure (verify stage)

Bad pattern:
```mp
; usually emitted after malformed IR shape or invalid transformed CFG
```

Fix pattern:
```mp
; restore canonical block structure and valid types, then rebuild from source
```

#### MPS0011 -- duplicate SSA local in lowering

Bad:
```mp
%x: i64 = const.i64 1
%x: i64 = const.i64 2
```

Fix:
```mp
%x: i64 = const.i64 1
%y: i64 = const.i64 2
```

#### MPS0012 -- call arity mismatch

Bad:
```mp
%r: i64 = call @add { a=const.i64 1 }
```

Fix:
```mp
%r: i64 = call @add { a=const.i64 1, b=const.i64 2 }
```

#### MPS0013 -- expected single arg value / value-shape mismatch

Bad:
```mp
; scalar-only site receives list argument
```

Fix:
```mp
; pass exactly one scalar value where scalar is required
```

#### MPS0014 / MPS0015 / MPS0016 -- fn-ref used where plain value required

Bad:
```mp
%r: i64 = call @f { x=@g }
```

Fix:
```mp
%r: i64 = call @f { x=const.i64 1 }
```

#### MPS0017 -- invalid branch predicate type

Bad:
```mp
cbr const.i64 1 bb1 bb2
```

Fix:
```mp
%cond: bool = icmp.eq { lhs=const.i64 1, rhs=const.i64 1 }
cbr %cond bb1 bb2
```

#### MPS0020 / MPS0021 / MPS0022 / MPS0023 -- no-overload namespace duplicates

Bad:
```mp
fn @dup() -> i64 { bb0: ret const.i64 0 }
fn @dup() -> i64 { bb0: ret const.i64 1 }
```

Fix:
```mp
fn @dup0() -> i64 { bb0: ret const.i64 0 }
fn @dup1() -> i64 { bb0: ret const.i64 1 }
```

#### MPS0024 -- ptr.* outside unsafe context

Bad:
```mp
%p: rawptr<i64> = ptr.null<i64>
```

Fix:
```mp
unsafe {
  %p: rawptr<i64> = ptr.null<i64>
}
```

#### MPS0025 -- unsafe fn call outside unsafe context

Bad:
```mp
%v: i64 = call @dangerous { }
```

Fix:
```mp
unsafe {
  %v: i64 = call @dangerous { }
}
```

### 22.3 HIR invariant codes (MPHIR*)

#### MPHIR01 -- getfield object must be borrow/mutborrow

Bad:
```mp
%p: TPoint = new TPoint { x=const.i64 1, y=const.i64 2 }
%x: i64 = getfield { obj=%p, field=x }
```

Fix:
```mp
%p: TPoint = new TPoint { x=const.i64 1, y=const.i64 2 }
%pb: borrow TPoint = borrow.shared { v=%p }
%x: i64 = getfield { obj=%pb, field=x }
```

#### MPHIR02 -- setfield object must be mutborrow

Bad:
```mp
%pb: borrow TPoint = borrow.shared { v=%p }
setfield { obj=%pb, field=x, val=const.i64 3 }
```

Fix:
```mp
%pm: mutborrow TPoint = borrow.mut { v=%p }
setfield { obj=%pm, field=x, val=const.i64 3 }
```

#### MPHIR03 -- borrow escapes via return

Bad:
```mp
fn @leak(%p: TPoint) -> borrow TPoint {
bb0:
  %pb: borrow TPoint = borrow.shared { v=%p }
  ret %pb
}
```

Fix (return the field value instead of the borrow):
```mp
fn @ok(%p: TPoint) -> i64 {
bb0:
  %pb: borrow TPoint = borrow.shared { v=%p }
  %x: i64 = getfield { obj=%pb, field=x }
  ret %x
}
```

### 22.4 Type system codes (MPT*)

#### MPT0001 -- unknown primitive

Bad:
```mp
%x: i99 = const.i64 0
```

Fix:
```mp
%x: i64 = const.i64 0
```

#### MPT0002 -- `shared`/`weak` invalid on `TOption`

Bad:
```mp
%x: shared TOption<i64> = enum.new<None> { }
```

Fix:
```mp
%x: TOption<i64> = enum.new<None> { }
```

#### MPT0003 -- `shared`/`weak` invalid on `TResult`

Bad:
```mp
%x: weak TResult<i64, i64> = enum.new<Ok> { v=const.i64 1 }
```

Fix:
```mp
%x: TResult<i64, i64> = enum.new<Ok> { v=const.i64 1 }
```

#### MPT1005 -- value struct contains heap handle

Bad:
```mp
value struct TBad {
  field s: Str
}
```

Fix:
```mp
heap struct TGood {
  field s: Str
}
```

#### MPT1020 -- value enum deferred in v0.1

Bad:
```mp
value enum TTag {
  variant A { }
}
```

Fix:
```mp
heap enum TTag {
  variant A { }
}
```

#### MPT1021 -- aggregate type deferred in v0.1

Bad pattern:
```mp
; uses deferred aggregate type forms not enabled for v0.1
```

Fix pattern:
```mp
; replace with supported builtins (Array/Map/struct) for v0.1
```

#### MPT1023 -- missing required trait impl

Bad (sorting Array<TPoint> without `ord` impl):
```mp
heap struct TPoint { field x: i64 }
; arr.sort over Array<TPoint> without ord impl
```

Fix:
```mp
sig TOrdPoint(borrow TPoint, borrow TPoint) -> i32
fn @ord_point(%a: borrow TPoint, %b: borrow TPoint) -> i32 {
bb0:
  ret const.i32 0
}
impl ord for TPoint = @ord_point
```

Note: For `Map<Str, V>` -- no impl needed because `Str` has built-in `hash` and `eq`.

#### MPT1030 -- suspend.call on non-function target form in v0.1

Bad:
```mp
; suspend.call through unsupported callable target in v0.1
```

Fix:
```mp
; call concrete function symbol directly or remove suspend.call pattern
```

#### MPT1200 -- orphan impl

Bad:
```mp
; impl hash for foreign type declared in another module, trait also foreign
```

Fix:
```mp
; either move impl to owning type module or define trait locally
```

#### MPT2001 -- call arity mismatch

Bad:
```mp
%r: i64 = call @sum2 { a=const.i64 1 }
```

Fix:
```mp
%r: i64 = call @sum2 { a=const.i64 1, b=const.i64 2 }
```

#### MPT2002 -- call arg unknown type

Bad:
```mp
%r: i64 = call @f { a=%missing }
```

Fix:
```mp
%a: i64 = const.i64 1
%r: i64 = call @f { a=%a }
```

#### MPT2003 -- call arg type mismatch

Bad:
```mp
%r: i64 = call @takes_i64 { a=const.bool true }
```

Fix:
```mp
%r: i64 = call @takes_i64 { a=const.i64 1 }
```

#### MPT2004 -- call target not found

Bad:
```mp
%r: i64 = call @does_not_exist { }
```

Fix:
```mp
fn @does_exist() -> i64 { bb0: ret const.i64 0 }
%r: i64 = call @does_exist { }
```

#### MPT2005 -- invalid generic type argument

Bad:
```mp
%r: i64 = call @id<TMissing> { x=const.i64 1 }
```

Fix:
```mp
%r: i64 = call @id<i64> { x=const.i64 1 }
```

#### MPT2006 -- getfield object unknown type

Bad:
```mp
%x: i64 = getfield { obj=%missing, field=x }
```

Fix:
```mp
%p: TPoint = new TPoint { x=const.i64 1, y=const.i64 2 }
%pb: borrow TPoint = borrow.shared { v=%p }
%x: i64 = getfield { obj=%pb, field=x }
```

#### MPT2007 -- getfield requires borrow/mutborrow struct

Bad:
```mp
%x: i64 = getfield { obj=const.i64 1, field=x }
```

Fix:
```mp
%pb: borrow TPoint = borrow.shared { v=%p }
%x: i64 = getfield { obj=%pb, field=x }
```

#### MPT2008 -- getfield target not a struct

Bad:
```mp
%b: borrow Str = borrow.shared { v=%s }
%x: i64 = getfield { obj=%b, field=x }
```

Fix:
```mp
; use a struct type with an actual field named x
%pb: borrow TPoint = borrow.shared { v=%p }
%x: i64 = getfield { obj=%pb, field=x }
```

#### MPT2009 -- missing struct field

Bad:
```mp
%x: i64 = getfield { obj=%pb, field=z }
```

Fix:
```mp
%x: i64 = getfield { obj=%pb, field=x }
```

#### MPT2010 -- cast operand unknown

Bad:
```mp
%x: i64 = cast<i64, i32> { v=%missing }
```

Fix:
```mp
%a: i64 = const.i64 1
%x: i32 = cast<i64, i32> { v=%a }
```

#### MPT2011 -- cast only primitive->primitive

Bad:
```mp
%x: i64 = cast<Str, i64> { v=%s }
```

Fix:
```mp
%x: i32 = cast<i64, i32> { v=const.i64 7 }
```

#### MPT2012 / MPT2013 / MPT2014 / MPT2015 -- numeric family typing

Bad (type mismatch -- `i64` mixed with `i32`):
```mp
%r: i64 = i.add { lhs=const.i64 1, rhs=const.i32 2 }
```

Fix:
```mp
%r: i64 = i.add { lhs=const.i64 1, rhs=const.i64 2 }
```

Bad (wrong family -- bool used with integer op):
```mp
%r: i64 = i.add { lhs=const.bool true, rhs=const.bool false }
```

Fix:
```mp
%r: i64 = i.add { lhs=const.i64 1, rhs=const.i64 2 }
```

Bad (unknown operand):
```mp
%r: i64 = i.add { lhs=%missing, rhs=const.i64 1 }
```

Fix:
```mp
%a: i64 = const.i64 10
%r: i64 = i.add { lhs=%a, rhs=const.i64 1 }
```

#### MPT2016 -- duplicate field arg

Bad:
```mp
%p: TPoint = new TPoint { x=const.i64 1, x=const.i64 2, y=const.i64 3 }
```

Fix:
```mp
%p: TPoint = new TPoint { x=const.i64 1, y=const.i64 3 }
```

#### MPT2017 -- unknown field in constructor/variant args

Bad:
```mp
%p: TPoint = new TPoint { x=const.i64 1, z=const.i64 3 }
```

Fix:
```mp
%p: TPoint = new TPoint { x=const.i64 1, y=const.i64 3 }
```

#### MPT2018 -- field value unknown type

Bad:
```mp
%p: TPoint = new TPoint { x=%missing, y=const.i64 3 }
```

Fix:
```mp
%x0: i64 = const.i64 1
%p: TPoint = new TPoint { x=%x0, y=const.i64 3 }
```

#### MPT2019 -- field type mismatch

Bad (bool used for an i64 field):
```mp
%p: TPoint = new TPoint { x=const.bool true, y=const.i64 3 }
```

Fix:
```mp
%p: TPoint = new TPoint { x=const.i64 1, y=const.i64 3 }
```

#### MPT2020 -- missing required field

Bad:
```mp
%p: TPoint = new TPoint { x=const.i64 1 }
```

Fix:
```mp
%p: TPoint = new TPoint { x=const.i64 1, y=const.i64 2 }
```

#### MPT2021 -- `new` target must be struct

Bad:
```mp
%x: i64 = new i64 { }
```

Fix:
```mp
%p: TPoint = new TPoint { x=const.i64 1, y=const.i64 2 }
```

#### MPT2022 -- unknown struct target

Bad:
```mp
%p: TMissing = new TMissing { x=const.i64 1 }
```

Fix:
```mp
heap struct TPoint { field x: i64 field y: i64 }
%p: TPoint = new TPoint { x=const.i64 1, y=const.i64 2 }
```

#### MPT2023 / MPT2024 -- invalid variant for TOption/TResult

Bad (using `Ok` on `TOption` -- that variant belongs to `TResult`):
```mp
%x: TOption<i64> = enum.new<Ok> { v=const.i64 1 }
```

Fix:
```mp
%x: TOption<i64> = enum.new<Some> { v=const.i64 1 }
```

Bad (using `Some` on `TResult` -- that variant belongs to `TOption`):
```mp
%r: TResult<i64, i64> = enum.new<Some> { v=const.i64 1 }
```

Fix:
```mp
%r: TResult<i64, i64> = enum.new<Ok> { v=const.i64 1 }
```

Valid variants:
- `TOption<T>`: `Some { v: T }`, `None { }`
- `TResult<Ok, Err>`: `Ok { v: Ok }`, `Err { v: Err }`

#### MPT2025 / MPT2026 / MPT2027 -- enum.new target/variant mismatch

Bad (using struct type as enum.new target):
```mp
%p: TPoint = enum.new<Some> { v=const.i64 1 }
```

Fix:
```mp
%o: TOption<i64> = enum.new<Some> { v=const.i64 1 }
```

Bad (variant does not exist in user enum):
```mp
%e: TMyEnum = enum.new<MissingVariant> { }
```

Fix:
```mp
%e: TMyEnum = enum.new<ExistingVariant> { }
```

#### MPT2028 / MPT2029 / MPT2030 / MPT2031 -- trait impl signature mismatch

Bad (`hash` impl has wrong arity -- takes 2 params, expected 1):
```mp
fn @hash_point(%a: borrow TPoint, %b: borrow TPoint) -> u64 { bb0: ret const.u64 0 }
impl hash for TPoint = @hash_point
```

Fix:
```mp
fn @hash_point(%a: borrow TPoint) -> u64 { bb0: ret const.u64 0 }
impl hash for TPoint = @hash_point
```

Bad (`eq` impl has wrong return type -- returns `i64`, expected `bool`):
```mp
fn @eq_point(%a: borrow TPoint, %b: borrow TPoint) -> i64 { bb0: ret const.i64 1 }
impl eq for TPoint = @eq_point
```

Fix:
```mp
fn @eq_point(%a: borrow TPoint, %b: borrow TPoint) -> bool { bb0: ret const.bool true }
impl eq for TPoint = @eq_point
```

Bad (`ord` impl -- first param is not borrow):
```mp
fn @ord_point(%a: TPoint, %b: borrow TPoint) -> i32 { bb0: ret const.i32 0 }
impl ord for TPoint = @ord_point
```

Fix:
```mp
fn @ord_point(%a: borrow TPoint, %b: borrow TPoint) -> i32 { bb0: ret const.i32 0 }
impl ord for TPoint = @ord_point
```

Bad (`eq` impl -- params have mismatched types):
```mp
fn @eq_point(%a: borrow TPoint, %b: borrow TOther) -> bool { bb0: ret const.bool true }
impl eq for TPoint = @eq_point
```

Fix:
```mp
fn @eq_point(%a: borrow TPoint, %b: borrow TPoint) -> bool { bb0: ret const.bool true }
impl eq for TPoint = @eq_point
```

#### MPT2032 -- impl target function missing

Bad:
```mp
impl hash for TPoint = @hash_point_missing
```

Fix:
```mp
fn @hash_point(%p: borrow TPoint) -> u64 {
bb0:
  ret const.u64 0
}
impl hash for TPoint = @hash_point
```

### 22.5 Ownership codes (MPO*)

#### MPO0003 -- borrow escapes scope

Bad (storing borrow into array):
```mp
%pb: borrow TPoint = borrow.shared { v=%p }
arr.push { arr=%arrm, val=%pb }
```

Fix:
```mp
; store owned/shared values, not borrow handles
arr.push { arr=%arrm, val=%p }
```

#### MPO0004 -- wrong ownership mode for mut/read op

Bad (using `shared` ref as mutating receiver for `arr.push`):
```mp
%sb: shared Array<i64> = share { v=%arr }
arr.push { arr=%sb, val=const.i64 1 }
```

Fix:
```mp
%arrm: mutborrow Array<i64> = borrow.mut { v=%arr }
arr.push { arr=%arrm, val=const.i64 1 }
```

Bad (using unique value directly as borrow receiver for `arr.len`):
```mp
%len: i64 = arr.len { arr=%arr }
```

Fix:
```mp
%arrb: borrow Array<i64> = borrow.shared { v=%arr }
%len: i64 = arr.len { arr=%arrb }
```

#### MPO0007 -- use after move

Bad (reusing `%p` after `share{}` consumed it):
```mp
%sp: shared TPoint = share { v=%p }
%pb: borrow TPoint = borrow.shared { v=%p }
```

Fix:
```mp
%sp: shared TPoint = share { v=%p }
%cp: shared TPoint = clone.shared { v=%sp }
; avoid reusing moved %p -- use %sp or clone it
```

#### MPO0011 -- move while borrowed

Bad (moving `%p` while borrow `%pb` is still active):
```mp
%pb: borrow TPoint = borrow.shared { v=%p }
%sp: shared TPoint = share { v=%p }
```

Fix (finish borrow uses, branch, then move in successor block):
```mp
bb0:
  %pb: borrow TPoint = borrow.shared { v=%p }
  %x: i64 = getfield { obj=%pb, field=x }
  br bb1
bb1:
  %sp: shared TPoint = share { v=%p }
  ret const.i64 0
```

#### MPO0101 -- borrow crosses block boundary

Bad:
```mp
bb0:
  %pb: borrow TPoint = borrow.shared { v=%p }
  br bb1
bb1:
  %x: i64 = getfield { obj=%pb, field=x }
```

Fix (re-borrow in the block where it is used):
```mp
bb0:
  br bb1
bb1:
  %pb: borrow TPoint = borrow.shared { v=%p }
  %x: i64 = getfield { obj=%pb, field=x }
```

#### MPO0102 -- borrow in phi

Bad:
```mp
%pb: borrow TPoint = phi borrow TPoint { [bb1:%p1b], [bb2:%p2b] }
```

Fix (phi the owned value, then borrow locally):
```mp
%p: TPoint = phi TPoint { [bb1:%p1], [bb2:%p2] }
%pb: borrow TPoint = borrow.shared { v=%p }
```

#### MPO0103 -- map.get requires Dupable V

Bad (map value type `TPoint` is not Dupable -- cannot return by value):
```mp
%v: TOption<TPoint> = map.get { map=%m_b, key=const.i64 1 }
```

Fix (use ref form instead):
```mp
%vref: borrow TPoint = map.get_ref { map=%m_b, key=const.i64 1 }
; or change map value type to Dupable (e.g., i64, Str, etc.)
```

#### MPO0201 -- spawn/send capture rule violation

Bad:
```mp
; spawn-like callable captures non-send borrow/mutborrow values
```

Fix:
```mp
; ensure callable captures only send-safe owned/shared data
; remove borrow captures before spawn boundary
```

### 22.6 FFI code

#### MPF0001 -- extern rawptr return missing ownership attr

Bad:
```mp
extern "c" module ffi {
  fn @open() -> rawptr<i64>
}
```

Fix:
```mp
extern "c" module ffi {
  fn @open() -> rawptr<i64> attrs { returns="owned" }
}
```

### 22.7 Link / emit / budget / lint codes (MPL*, MPLINK*)

#### MPL0001 -- unknown emit kind

Bad:
```bash
magpie --entry src/main.mp --emit foo build
```

Fix:
```bash
magpie --entry src/main.mp --emit exe,llvm-ir,mpir build
```

#### MPL0002 -- requested artifact missing

Bad:
```bash
magpie --entry src/main.mp --emit exe,shared-lib build
# build reports success path issue but one requested artifact absent
```

Fix:
```bash
magpie --entry src/main.mp --emit exe --output json build
# resolve upstream codegen/link errors until requested artifact exists
```

#### MPL0801 -- LLM budget too small

Bad:
```bash
magpie --entry src/main.mp --llm --llm-token-budget 100 build
```

Fix:
```bash
magpie --entry src/main.mp --llm --llm-token-budget 12000 build
# or use --llm-budget-policy minimal
```

#### MPL0802 -- tokenizer fallback

Bad:
```bash
magpie --entry src/main.mp --llm --llm-tokenizer custom:missing build
```

Fix:
```bash
magpie --entry src/main.mp --llm --llm-tokenizer approx:utf8_4chars build
```

#### MPL2001 / MPL2002 / MPL2003 / MPL2005 / MPL2007 / MPL2020 / MPL2021

Bad patterns:
```mp
; oversized functions, dead code, unnecessary borrows, empty blocks,
; generic explosion, mixed generics-mode usage
```

Fix patterns:
```mp
; split large functions, remove dead/empty code, simplify borrows,
; constrain generic instantiations, use one generics mode consistently
```

#### MPLINK01 -- primary link path failed

Bad:
```bash
magpie --entry src/main.mp --emit exe build
# native link toolchain missing/misconfigured
```

Fix:
```bash
# install/repair linker toolchain for target triple, then rebuild
magpie --entry src/main.mp --emit exe build
```

#### MPLINK02 -- fallback link also unavailable

Bad:
```bash
magpie --entry src/main.mp --emit exe build
```

Fix:
```bash
# ensure clang/llc/system linker availability for target
# until fixed, use --emit llvm-ir,mpir for non-native debugging
magpie --entry src/main.mp --emit llvm-ir,mpir build
```

### 22.8 MPIR pipeline code

#### MPM0001 -- MPIR lowering produced no modules

Bad pattern:
```bash
magpie --entry src/main.mp --emit mpir build
# lowering receives empty resolved module set
```

Fix pattern:
```bash
# verify entry file parses/resolves and exports expected module/function symbols
magpie --entry src/main.mp --emit mpir --output json build
```

### 22.9 Family fallback for codes not listed above

When `magpie explain <CODE>` returns family-level guidance only:

- `MPO*`: ownership lifetime/move/borrow violation. Reduce borrow scope, avoid borrow escapes, and use clones intentionally.
- `MPT*`: type/trait contract violation. Align types exactly and satisfy required trait impls.
- `MPS*`: SSA/CFG/resolve invariant violation. Ensure defs dominate uses and control-flow targets are valid.
- `MPL*`: lint/policy/artifact constraints. Adjust command flags, code structure, or output budget.

If unsure, create a tiny reproducer with one function and one failing op, then iterate.

### 22.10 Individual-code expansions for grouped ranges

#### MPS0000 -- generic module resolution failure

Bad:
```mp
; unresolved/invalid module/import graph with no narrower code selected
```

Fix:
```mp
; ensure module headers are valid and all imports resolve uniquely
```

#### MPS0004 -- import/local namespace conflict

Bad:
```mp
imports { util.math::{@sum} }
fn @sum() -> i64 { bb0: ret const.i64 0 }
```

Fix:
```mp
imports { util.math::{@sum_util} }
fn @sum() -> i64 { bb0: ret const.i64 0 }
```

#### MPS0005 -- type import/local type conflict

Bad:
```mp
imports { util.types::{TPoint} }
heap struct TPoint { field x: i64 field y: i64 }
```

Fix:
```mp
imports { util.types::{TPointExt} }
heap struct TPoint { field x: i64 field y: i64 }
```

#### MPS0006 -- ambiguous import name

Bad:
```mp
imports { a.mod::{@foo}, b.mod::{@foo} }
```

Fix:
```mp
imports { a.mod::{@foo_a}, b.mod::{@foo_b} }
```

#### MPS0014 -- invalid function reference in scalar-value position

Bad:
```mp
; scalar-only site receives fn ref
%r: i64 = call @f { x=@g }
```

Fix:
```mp
%r: i64 = call @f { x=const.i64 1 }
```

#### MPS0015 -- invalid function reference inside list lowered as plain values

Bad:
```mp
%r: i64 = call @f { xs=[@g] }
```

Fix:
```mp
%r: i64 = call @f { xs=[const.i64 1] }
```

#### MPS0016 -- invalid plain-value argument uses fn ref

Bad:
```mp
%r: i64 = call @f { x=@g }
```

Fix:
```mp
%r: i64 = call @f { x=const.i64 7 }
```

#### MPS0020 -- duplicate function/global symbol

Bad:
```mp
global @x: i64 = const.i64 1
fn @x() -> i64 { bb0: ret const.i64 0 }
```

Fix:
```mp
global @x_global: i64 = const.i64 1
fn @x() -> i64 { bb0: ret const.i64 0 }
```

#### MPS0021 -- duplicate type symbol in module

Bad:
```mp
heap struct TPoint { field x: i64 field y: i64 }
heap struct TPoint { field x: i64 field y: i64 }
```

Fix:
```mp
heap struct TPoint { field x: i64 field y: i64 }
heap struct TPoint2 { field x: i64 field y: i64 }
```

#### MPS0022 -- duplicate @ namespace symbol

Bad:
```mp
fn @main() -> i64 { bb0: ret const.i64 0 }
global @main: i64 = const.i64 1
```

Fix:
```mp
fn @main() -> i64 { bb0: ret const.i64 0 }
global @main_value: i64 = const.i64 1
```

#### MPS0023 -- duplicate `sig` symbol

Bad:
```mp
sig TOrdPoint(borrow TPoint, borrow TPoint) -> i32
sig TOrdPoint(borrow TPoint, borrow TPoint) -> i32
```

Fix:
```mp
sig TOrdPoint(borrow TPoint, borrow TPoint) -> i32
sig THashPoint(borrow TPoint) -> u64
```

#### MPT2012 -- numeric lhs unknown

Bad:
```mp
%r: i64 = i.add { lhs=%missing, rhs=const.i64 1 }
```

Fix:
```mp
%a: i64 = const.i64 10
%r: i64 = i.add { lhs=%a, rhs=const.i64 1 }
```

#### MPT2013 -- numeric rhs unknown

Bad:
```mp
%r: i64 = i.add { lhs=const.i64 1, rhs=%missing }
```

Fix:
```mp
%b: i64 = const.i64 2
%r: i64 = i.add { lhs=const.i64 1, rhs=%b }
```

#### MPT2014 -- numeric operands have mismatched types

Bad:
```mp
%r: i64 = i.add { lhs=const.i64 1, rhs=const.i32 2 }
```

Fix:
```mp
%r: i64 = i.add { lhs=const.i64 1, rhs=const.i64 2 }
```

#### MPT2015 -- wrong primitive family for numeric op

Bad:
```mp
%r: i64 = i.add { lhs=const.bool true, rhs=const.bool false }
```

Fix:
```mp
%r: i64 = i.add { lhs=const.i64 1, rhs=const.i64 2 }
```

#### MPT2023 -- invalid variant for TOption

Bad:
```mp
%o: TOption<i64> = enum.new<Ok> { v=const.i64 1 }
```

Fix:
```mp
%o: TOption<i64> = enum.new<Some> { v=const.i64 1 }
```

#### MPT2024 -- invalid variant for TResult

Bad:
```mp
%r: TResult<i64, i64> = enum.new<Some> { v=const.i64 1 }
```

Fix:
```mp
%r: TResult<i64, i64> = enum.new<Ok> { v=const.i64 1 }
```

#### MPT2025 -- enum.new result type is not enum

Bad:
```mp
%p: TPoint = enum.new<Any> { }
```

Fix:
```mp
%e: TMyEnum = enum.new<MyVariant> { }
```

#### MPT2026 -- user enum variant not found

Bad:
```mp
%e: TMyEnum = enum.new<MissingVariant> { }
```

Fix:
```mp
%e: TMyEnum = enum.new<ExistingVariant> { }
```

#### MPT2027 -- enum.new target type must be enum

Bad:
```mp
%x: i64 = enum.new<Any> { }
```

Fix:
```mp
%e: TMyEnum = enum.new<ExistingVariant> { }
```

#### MPT2028 -- trait impl parameter count mismatch

Bad:
```mp
fn @hash_point(%a: borrow TPoint, %b: borrow TPoint) -> u64 { bb0: ret const.u64 0 }
impl hash for TPoint = @hash_point
```

Fix:
```mp
fn @hash_point(%a: borrow TPoint) -> u64 { bb0: ret const.u64 0 }
impl hash for TPoint = @hash_point
```

#### MPT2029 -- trait impl return type mismatch

Bad:
```mp
fn @eq_point(%a: borrow TPoint, %b: borrow TPoint) -> i64 { bb0: ret const.i64 1 }
impl eq for TPoint = @eq_point
```

Fix:
```mp
fn @eq_point(%a: borrow TPoint, %b: borrow TPoint) -> bool { bb0: ret const.bool true }
impl eq for TPoint = @eq_point
```

#### MPT2030 -- trait impl first parameter must be borrow target type

Bad:
```mp
fn @ord_point(%a: TPoint, %b: borrow TPoint) -> i32 { bb0: ret const.i32 0 }
impl ord for TPoint = @ord_point
```

Fix:
```mp
fn @ord_point(%a: borrow TPoint, %b: borrow TPoint) -> i32 { bb0: ret const.i32 0 }
impl ord for TPoint = @ord_point
```

#### MPT2031 -- trait impl parameters must both match borrow target

Bad:
```mp
fn @eq_point(%a: borrow TPoint, %b: borrow TOther) -> bool { bb0: ret const.bool true }
impl eq for TPoint = @eq_point
```

Fix:
```mp
fn @eq_point(%a: borrow TPoint, %b: borrow TPoint) -> bool { bb0: ret const.bool true }
impl eq for TPoint = @eq_point
```

#### MPL2001 -- lint/code-quality violation (oversized function)

Bad:
```mp
; compiler-reported lint pattern (e.g., oversized or problematic function body)
```

Fix:
```mp
; refactor into smaller helpers as suggested by diagnostic text
```

#### MPL2002 -- lint/code-quality violation (unused/dead symbol class)

Bad:
```mp
fn @unused() -> i64 { bb0: ret const.i64 0 }
```

Fix:
```mp
; remove unused symbol or reference/export it intentionally
```

#### MPL2003 -- lint/code-quality violation (unnecessary borrow class)

Bad:
```mp
%pb: borrow TPoint = borrow.shared { v=%p }
; immediate pass-through where owned/shared would suffice
```

Fix:
```mp
; pass %p directly when borrow semantics are unnecessary
```

#### MPL2005 -- lint/code-quality violation (empty block class)

Bad:
```mp
bb1:
  unreachable
; structurally pointless empty flow region around it
```

Fix:
```mp
; remove dead/empty block and simplify control flow
```

#### MPL2007 -- lint/code-quality violation (unreachable code class)

Bad:
```mp
bb0:
  ret const.i64 0
  %x: i64 = const.i64 1
```

Fix:
```mp
bb0:
  %x: i64 = const.i64 1
  ret %x
```

#### MPL2020 -- monomorphization pressure too high

Bad:
```mp
; huge generic instantiation fan-out across many type arguments
```

Fix:
```mp
; reduce generic permutations or switch to shared generics mode
```

#### MPL2021 -- mixed generics mode conflict

Bad:
```bash
# build/config mixes incompatible generics strategies in one target/profile
```

Fix:
```bash
# pick one generics strategy consistently for this build profile
```

---

## 23) Trait signature reference

Precise signatures for all built-in traits, as enforced by `MPT2028..MPT2031`:

| Trait | Required signature | Return type | Notes |
|-------|-------------------|-------------|-------|
| `hash` | `(%self: borrow T) -> u64` | `u64` | 1 param |
| `eq`   | `(%a: borrow T, %b: borrow T) -> bool` | `bool` | 2 params, both borrow T |
| `ord`  | `(%a: borrow T, %b: borrow T) -> i32` | `i32` | 2 params, both borrow T |

When implementing for a type `TFoo`:

```mp
;; hash impl
fn @hash_foo(%self: borrow TFoo) -> u64 {
bb0:
  ret const.u64 0
}
impl hash for TFoo = @hash_foo

;; eq impl
fn @eq_foo(%a: borrow TFoo, %b: borrow TFoo) -> bool {
bb0:
  ret const.bool true
}
impl eq for TFoo = @eq_foo

;; ord impl
fn @ord_foo(%a: borrow TFoo, %b: borrow TFoo) -> i32 {
bb0:
  ret const.i32 0
}
impl ord for TFoo = @ord_foo
```

`Str`, all integer/float primitives, and `bool` satisfy all three traits built-in.
User-defined `heap struct` and `heap enum` types require explicit impl declarations.
`value enum` is deferred in v0.1; use `heap enum` instead.

---

## 24) Worked end-to-end examples

### 24.1 Counting elements in a Map<Str, i64>

```mp
module demo.counter
exports { @count_keys }
imports { }
digest "0000000000000000"

fn @count_keys() -> i64 {
bb0:
  ; Map<Str, i64> -- Str has built-in hash/eq, no impl needed
  %m: Map<Str, i64> = map.new<Str, i64> { }
  %mb: mutborrow Map<Str, i64> = borrow.mut { v=%m }
  map.set { map=%mb, key=const.Str "a", val=const.i64 1 }
  map.set { map=%mb, key=const.Str "b", val=const.i64 2 }
  br bb1

bb1:
  %mr: borrow Map<Str, i64> = borrow.shared { v=%m }
  %len: i64 = map.len { map=%mr }
  ret %len
}
```

### 24.2 Sorting integers (Array<i64> -- ord built-in)

```mp
module demo.sort
exports { @sort_ints }
imports { }
digest "0000000000000000"

fn @sort_ints() -> i64 {
bb0:
  %arr: Array<i64> = arr.new<i64> { cap=const.i64 4 }
  %arrm: mutborrow Array<i64> = borrow.mut { v=%arr }
  arr.push { arr=%arrm, val=const.i64 3 }
  arr.push { arr=%arrm, val=const.i64 1 }
  arr.push { arr=%arrm, val=const.i64 2 }
  arr.sort { arr=%arrm }
  br bb1

bb1:
  %arrb: borrow Array<i64> = borrow.shared { v=%arr }
  %len: i64 = arr.len { arr=%arrb }
  ret %len
}
```

### 24.3 i32 return (const suffix must match)

```mp
module demo.i32ret
exports { @main }
imports { }
digest "0000000000000000"

fn @main() -> i32 {
bb0:
  %a: i32 = const.i32 10
  %b: i32 = const.i32 20
  %c: i32 = i.add { lhs=%a, rhs=%b }
  ret %c
}
```

Note: `ret const.i64 0` would be wrong here -- it must be `const.i32 0` (or a value with declared type `i32`).

### 24.4 TResult error propagation pattern

```mp
module demo.result
exports { @might_fail }
imports { }
digest "0000000000000000"

fn @might_fail(%x: i64) -> TResult<i64, i64> {
bb0:
  %zero: i64 = const.i64 0
  %is_zero: bool = icmp.eq { lhs=%x, rhs=%zero }
  cbr %is_zero bb_err bb_ok

bb_err:
  %e: TResult<i64, i64> = enum.new<Err> { v=const.i64 1 }
  ret %e

bb_ok:
  %ok: TResult<i64, i64> = enum.new<Ok> { v=%x }
  ret %ok
}
```

### 24.5 Struct with hash/eq for use as Map key

```mp
module demo.custom_key
exports { @demo }
imports { }
digest "0000000000000000"

heap struct TId {
  field n: i64
}

fn @hash_id(%self: borrow TId) -> u64 {
bb0:
  ; getfield keys may be in any order
  %n: i64 = getfield { field=n, obj=%self }
  %h: u64 = cast<i64, u64> { v=%n }
  ret %h
}
impl hash for TId = @hash_id

fn @eq_id(%a: borrow TId, %b: borrow TId) -> bool {
bb0:
  %na: i64 = getfield { obj=%a, field=n }
  %nb_val: i64 = getfield { obj=%b, field=n }
  %eq: bool = icmp.eq { lhs=%na, rhs=%nb_val }
  ret %eq
}
impl eq for TId = @eq_id

fn @demo() -> i64 {
bb0:
  %m: Map<TId, i64> = map.new<TId, i64> { }
  %mb: mutborrow Map<TId, i64> = borrow.mut { v=%m }
  %id: TId = new TId { n=const.i64 1 }
  map.set { map=%mb, key=%id, val=const.i64 42 }
  br bb1

bb1:
  %mr: borrow Map<TId, i64> = borrow.shared { v=%m }
  %len: i64 = map.len { map=%mr }
  ret %len
}
```

---

## 25) Summary of all corrections vs. earlier documentation

This section consolidates the behavioral changes from older versions of this guide:

| Topic | Old (incorrect) | Correct |
|-------|----------------|---------|
| `const` suffix | Examples used `const.i64 0` for `i32` returns | Suffix must match declared type: `const.i32 0` for `i32` |
| `getfield`/`setfield` key order | "strict key order" warning present | Keys may be in **any order** -- no strict ordering for these ops |
| Doc comment syntax | `;;;` listed as doc comment | `;;` is the correct doc comment token |
| `is_async` after lowering | "After lowering, `is_async` is set false" | `is_async` **stays true** after async lowering; verifiers use this to skip SSA checks |
| `Str` trait impls | Not mentioned | `Str` has built-in `hash`/`eq`/`ord` -- no explicit impl needed for `Map<Str, V>` |
| Collection op receivers | Inconsistent | Mutation ops require `mutborrow`; read ops require `borrow` or `mutborrow` |
| Minimal example return type | Used `i64` with `const.i64` only | Both `i32` and `i64` templates provided; suffix must match declared return type |
