# Magpie v0.1 — Comprehensive Technical Documentation

> This document is a comprehensive, implementation-oriented reference for Magpie v0.1.
> It consolidates language grammar/syntax, core semantics, compiler architecture, CLI arguments,
> and the LLM-first design rationale.

---

## Table of Contents

1. [Design Goals and Core Decisions](#1-design-goals-and-core-decisions)
2. [LLM-First Rationale](#2-llm-first-rationale)
3. [Language Overview](#3-language-overview)
4. [Lexical Grammar](#4-lexical-grammar)
5. [File Grammar and Canonical Structure](#5-file-grammar-and-canonical-structure)
6. [Declarations](#6-declarations)
7. [Type System](#7-type-system)
8. [Instruction Model](#8-instruction-model)
9. [Opcode Reference (Value + Void)](#9-opcode-reference-value--void)
10. [Semantics and Safety Rules](#10-semantics-and-safety-rules)
11. [TCallable: Why It Exists and How It Works](#11-tcallable-why-it-exists-and-how-it-works)
12. [ARC + Ownership: Why Both](#12-arc--ownership-why-both)
13. [Compiler Architecture](#13-compiler-architecture)
14. [Compiler Pipeline](#14-compiler-pipeline)
15. [Artifacts and Emission Kinds](#15-artifacts-and-emission-kinds)
16. [Diagnostics Model](#16-diagnostics-model)
17. [CLI Arguments and Commands](#17-cli-arguments-and-commands)
18. [Configuration Resolution Order](#18-configuration-resolution-order)
19. [Build/Test/Run Playbooks](#19-buildtestrun-playbooks)
20. [Examples](#20-examples)

---

## 1) Design Goals and Core Decisions

Magpie v0.1 is designed around a few non-negotiable engineering constraints:

1. **Deterministic, explicit surface language**
  - Minimal hidden behavior.
  - Explicit control-flow and explicit ownership-related operations.
  - Canonical formatting (CSNF) for stable text representation.

2. **LLM-agent compatibility**
  - Structured, low-ambiguity syntax for deterministic generation and repair loops.
  - Progressive disclosure and compact machine artifacts.
  - Token-budget-aware output behavior.

3. **Static safety + predictable runtime**
  - Rust-like ownership/borrowing constraints for aliasing/mutation safety.
  - ARC-managed heap lifetimes for deterministic reclamation (no tracing GC pause model).

4. **Pipeline observability**
  - Explicit staged compiler pipeline.
  - Rich diagnostics, graph artifacts, and JSON envelopes for tool/agent integration.

5. **Separation of concerns**
  - Parse/resolve/type/HIR/ownership/MPIR/codegen/link are distinct phases with explicit failure surfaces.

---

## 2) LLM-First Rationale

This section summarizes the core LLM-first engineering rationale as applied in Magpie tooling and language design.

### 2.1 Finite context windows and bounded token budgets

**Claim (paraphrase):** Transformer-based systems operate under bounded context and practical token budgets.

**Evidence:**
- Original Transformer self-attention scaling with sequence length.
- Efficient Transformer survey on scaling constraints.
- LongBench: practical long-context degradation and retrieval/compression usefulness.

**Magpie design implications:**
- Budgeted machine-readable outputs.
- Small/local code and progressive disclosure.
- Prefer targeted artifacts (`.mpdbg`, graphs, `.mpir`) over full-context dumps.

### 2.2 Attention dilution / long-context retrieval failure

**Claim:** Long sequences reduce reliable focus on relevant middle-span details.

**Evidence:**
- “Lost in the Middle” shows severe position effects for relevant information in long contexts.

**Magpie implications:**
- Locality constraints, explicit dependencies, compact structured summaries.
- Encourage retrieval of only relevant slices rather than monolithic context.

### 2.3 Tokenization brittleness and canonical syntax

**Claim:** Tokenization and boundary behavior can destabilize model output.

**Evidence:**
- Tokenization robustness degradation results.
- Partial token boundary issues, including code domains.

**Magpie implications:**
- Canonical `key=value` forms.
- Regular opcode syntax with low branching.
- Reduced format variance across iterations.

### 2.4 Grammar-constrained generation and reduced syntax error rate

**Claim:** Less ambiguous, more regular grammars are easier for model generation and correction.

**Evidence:**
- Constrained decoding reduces syntax errors in code generation settings.
- Grammar-constrained decoding improves syntactic correctness in structured outputs.

**Magpie implications:**
- Explicit opcodes, explicit call forms, no operator overloading/method-syntax ambiguity.

### 2.5 Canonicalization to reduce drift and token waste

**Claim:** Format drift can alter model behavior and inflate token usage.

**Evidence:**
- Prompt formatting sensitivity and output variance (, ).
- Measured token savings from formatting minimization in code contexts.
- Deterministic canonicalization standards and tooling analogues (, ).

**Magpie implications:**
- CSNF canonical source normalization.
- Canonical JSON output for stable machine consumption.

### 2.6 Stable IDs and progressive disclosure

**Claim:** Stable identifiers enable compact references and incremental retrieval.

**Evidence:**
- Content-addressed object identity in Git.

**Magpie implications:**
- Stable symbol-oriented references and graph artifacts.
- Better cacheability and compact cross-reference workflows.

### 2.7 Retrieval-first workflows

**Claim:** Retrieval of relevant context is more effective than monolithic prompt stuffing.

**Evidence:**
- RAG , RETRO , in-context retrieval augmentation.
- LongBench’s retrieval/compression observations.

**Magpie implications:**
- Progressive disclosure artifacts and memory index workflows.

---

## 3) Language Overview

Magpie source files are module-centric and SSA-oriented.

### 3.1 Core shape

- Header with strict order:
 1. `module...`
 2. `exports {... }`
 3. `imports {... }`
 4. `digest "..."`
- Declarations (`fn`, `struct`, `enum`, `extern`, `impl`, `sig`, `global`, etc.)
- Function bodies are basic-block based (`bbN:` labels)
- Each block ends with one terminator (`ret`, `br`, `cbr`, `switch`, `unreachable`)

### 3.2 Programming model highlights

- Explicit SSA values (`%name`)
- Function symbols with `@name`
- Type symbols with `TName`
- Explicit ownership forms (`shared`, `borrow`, `mutborrow`, `weak`)
- Explicit heap and projection ops (e.g., `new`, `getfield`, `setfield`)

---

## 4) Lexical Grammar

### 4.1 Comments

- Line comment: `;...`
- Doc comment token: `;;...` (double semicolon, NOT triple)

### 4.2 Token classes

- Identifiers: `ident`
- Function names: `@fn_name`
- SSA names: `%local`
- Type names: `TType`
- Block labels: `bb0`, `bb1`,...

### 4.3 Literals

- Integer literals (decimal and hex `0x...`)
- Float literals (`123.45`, optional `f32` / `f64` suffix)
- String literals with escapes (`\n`, `\t`, `\\`, `\"`, `\u{...}`)
- Booleans: `true`, `false`
- Unit literal form: `unit`

### 4.4 Punctuation

`{ } ( ) < > [ ] = : ,. ->`

---

## 5) File Grammar and Canonical Structure

## 5.1 Header grammar

```ebnf
file   := header decl*
header  := "module" module_path
       "exports" export_block
       "imports" import_block
       "digest" string_lit
```

### Exports

```ebnf
export_block := "{" (export_item ("," export_item)*)? "}"
export_item := fn_name | type_name
```

### Imports

```ebnf
import_block := "{" (import_group ("," import_group)*)? "}"
import_group := module_path "::" "{" (import_item ("," import_item)*)? "}"
import_item := fn_name | type_name
```

## 5.2 Canonical source normal form (CSNF)

- Canonical ordering of exports/imports.
- Canonical printing of ops and types.
- Canonical block label remapping on formatting.
- Digest normalization via `update_digest`.

---

## 6) Declarations

## 6.1 Function declarations

- `fn @name(...) -> Type {... }`
- `async fn @name(...) -> Type {... }`
- `unsafe fn @name(...) -> Type {... }`
- `gpu fn @name(...) -> Type target(<ident>) {... }`

Optional metadata block:

```mp
meta {
 uses {... }
 effects {... }
 cost { key=123,... }
}
```

## 6.2 Type declarations

- `heap struct TName { field... }`
- `value struct TName { field... }`
- `heap enum TName { variant... }`
- `value enum TName { variant... }`

## 6.3 Extern module

```mp
extern "c" module ffi {
 fn @name(%x: i64) -> i64 attrs { link_name="...", returns="owned" }
}
```

## 6.4 Globals

```mp
global @g: i64 = const.i64 42
```

## 6.5 Trait-like signature + impl bindings

```mp
sig TOrdPoint(borrow TPoint, borrow TPoint) -> i32
impl ord for TPoint = @ord_point
```

---

## 7) Type System

## 7.1 Primitive types

- Signed ints: `i1 i8 i16 i32 i64 i128`
- Unsigned ints: `u1 u8 u16 u32 u64 u128`
- Floats: `f16 f32 f64`
- Others: `bool unit`

## 7.2 Ownership modifiers

Prefix modifiers:
- `shared`
- `borrow`
- `mutborrow`
- `weak`

## 7.3 Builtin types

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
- `TCallable<TSig>`

## 7.4 User and pointer types

- Named type: `TName` or `module.path.TName`
- Raw pointer: `rawptr<T>`

---

## 8) Instruction Model

Each basic block contains:

1. **SSA assignments** (`%dst: Ty = value_op...`)
2. **Void ops** (`setfield...`, `arr.push...`, etc.)
3. **Terminator** (required)

### 8.1 Terminators

- `ret [value]?`
- `br bbN`
- `cbr cond bbThen bbElse`
- `switch value { case lit -> bbN... } else bbDefault`
- `unreachable`

---

## 9) Opcode Reference (Value + Void)

This section lists major surface op families and canonical syntax forms.

## 9.1 Value-producing op families

### Constants
- `const.<Type> <literal>`

### Integer arithmetic/bitwise
- `i.add`, `i.sub`, `i.mul`, `i.sdiv`, `i.udiv`, `i.srem`, `i.urem`
- `i.add.wrap`, `i.sub.wrap`, `i.mul.wrap`
- `i.add.checked`, `i.sub.checked`, `i.mul.checked`
- `i.and`, `i.or`, `i.xor`, `i.shl`, `i.lshr`, `i.ashr`

All use:
```mp
{ lhs=<value>, rhs=<value> }
```

### Floating-point
- `f.add`, `f.sub`, `f.mul`, `f.div`, `f.rem`
- `f.add.fast`, `f.sub.fast`, `f.mul.fast`, `f.div.fast`

### Comparison
- Integer: `icmp.eq/ne/slt/sgt/sle/sge/ult/ugt/ule/uge`
- Float: `fcmp.oeq/one/olt/ogt/ole/oge`

### Call and control transfer values
- `call @fn...`
- `call.indirect <callable>...`
- `try @fn...`
- `suspend.call @fn...`
- `suspend.await { fut=... }`

### Heap/object/SSA
- `new Type { field=v,... }`
- `getfield { obj=..., field=name }` (keys accepted in any order)
- `phi Type { [bb1:v1], [bb2:v2],... }`

### Enum
- `enum.new<Variant> {... }`
- `enum.tag { v=... }`
- `enum.payload<Variant> { v=... }`
- `enum.is<Variant> { v=... }`

### Ownership conversion
- `share`, `clone.shared`, `clone.weak`, `weak.downgrade`, `weak.upgrade`
- `borrow.shared`, `borrow.mut`

### Cast/pointer
- `cast<PrimFrom, PrimTo> { v=... }`
- `ptr.null<T>`, `ptr.addr<T>`, `ptr.from_addr<T>`, `ptr.add<T>`, `ptr.load<T>`

### Callable
- `callable.capture @fn { capture=value,... }`

### Arrays
- `arr.new<T>`, `arr.len`, `arr.get`, `arr.pop`, `arr.slice`, `arr.contains`, `arr.map`, `arr.filter`, `arr.reduce`

### Maps
- `map.new<K,V>`, `map.len`, `map.get`, `map.get_ref`, `map.delete`, `map.contains_key`, `map.keys`, `map.values`

### Strings/JSON
- `str.concat`, `str.len`, `str.eq`, `str.slice`, `str.bytes`
- `str.builder.new`, `str.builder.build`
- `str.parse_i64`, `str.parse_u64`, `str.parse_f64`, `str.parse_bool`
- `json.encode<T>`, `json.decode<T>`

Runtime ABI note (current migration model):
- Compiler lowering uses fallible runtime calls (`mp_rt_str_try_parse_*`, `mp_rt_json_try_*`) and checks status codes.
- In the temporary compatibility path, non-OK status still routes to `mp_rt_panic` to preserve legacy user-visible behavior.
- Legacy runtime wrappers (`mp_rt_str_parse_*`, `mp_rt_json_encode/decode`) are compatibility shims and are deprecated. New integrations should call only `*_try_*` APIs.

### GPU values
- `gpu.thread_id`, `gpu.workgroup_id`, `gpu.workgroup_size`, `gpu.global_id`
- `gpu.buffer_load<T>`, `gpu.buffer_len<T>`
- `gpu.shared<count,T>`
- `gpu.launch`, `gpu.launch_async`

## 9.2 Void op families

- `call_void`, `call_void.indirect`
- `setfield { obj=..., field=..., val=... }` (keys accepted in any order)
- `panic { msg=... }`
- `ptr.store<T> { p=..., v=... }`
- `arr.set`, `arr.push`, `arr.sort`, `arr.foreach`
- `map.set`, `map.delete_void`
- `str.builder.append_str/i64/i32/f64/bool`
- `gpu.barrier`
- `gpu.buffer_store<T>`

---

## 10) Semantics and Safety Rules

## 10.1 SSA and CFG invariants

- Single definition per local id.
- Uses must be dominated by defs.
- Branch/switch targets must exist.
- Every block has one terminator.

## 10.2 Borrowing rules

- Borrow values cannot be stored into escaping locations.
- Borrow handles cannot cross block boundaries.
- Borrow handles cannot appear in `phi`.
- Returning borrow values is forbidden.

## 10.3 Projection constraints

- `getfield` requires `borrow`/`mutborrow` receiver.
- `setfield` requires `mutborrow` receiver.
- Collection read/write ops enforce ownership modes (read via borrow; write via unique/mutborrow).

## 10.4 Unsafe boundaries

Outside unsafe contexts, forbidden:
- raw pointer opcodes (`ptr.*`)
- calls to unsafe functions

## 10.5 Trait requirements (selected)

- `arr.contains` requires `eq` impl on element type.
- `arr.sort` requires `ord` impl on element type.
- `map.new<K,V>` requires `hash` + `eq` impl for `K`.

## 10.6 v0.1 restrictions (selected)

- Certain aggregate/deferred forms are restricted in v0.1 checks.
- `suspend.call` on non-function callable target forms is forbidden in v0.1.

---

## 11) TCallable: Why It Exists and How It Works

In Magpie v0.1, `TCallable` exists because the language **forbids closures as a primitive**,
but still needs first-class “callable later” behavior (callbacks, middleware, higher-order APIs)
without hidden captures or opaque syntax.

## 11.1 Conceptual representation

`TCallable<TSig>` is an ARC-managed heap object containing:

- target function identity (call target)
- optional captured environment pointer (`data_ptr`, nullable)
- callable metadata/vtable entries (`call_fn`, `drop_fn`, capture layout metadata)

Creation and invocation are explicit:

- create: `callable.capture @fn { capture1=%v1,... }`
- invoke: `call.indirect %callable {... }` or `call_void.indirect`

## 11.2 Why not closure syntax

Magpie intentionally avoids closure syntax because closure primitives often imply:

- implicit capture set inference,
- hidden environment layout,
- hidden destructor behavior,
- increased generation ambiguity for LLM-driven tool loops.

`TCallable` keeps these visible and explicit.

## 11.3 LLM and tooling benefits

1. **Visible dependencies**
  - Capture list is explicit in one line.
2. **Regular generation pattern**
  - `sig` + `callable.capture` + `call.indirect` are low-ambiguity templates.
3. **Deterministic ownership repair**
  - capture is a move boundary; clone/share fixes are mechanical.
4. **Storage-friendly behavior objects**
  - Router/middleware patterns can store callables in typed fields and arrays.
5. **Async boundary clarity**
  - v0.1 forbids problematic `suspend.call` callable-indirection patterns, reducing opaque failures.

## 11.4 Minimal example: capture + call.indirect

```mp
module demo.callable
exports { @main, @multiply_by }
imports { }
digest "0000000000000000"

sig TMulSig(i32) -> i32

fn @multiply_by(%x: i32, %factor: i32) -> i32 {
bb0:
 %y: i32 = i.mul { lhs=%x, rhs=%factor }
 ret %y
}

fn @main() -> i32 {
bb0:
 %factor: i32 = const.i32 3

 ; Create callable with captured factor
 %mul_by_3: TCallable<TMulSig> = callable.capture @multiply_by { factor=%factor }

 ; Invoke indirectly
 %result: i32 = call.indirect %mul_by_3 { args=[const.i32 7] }

 ret %result
}
```

---

## 12) ARC + Ownership: Why Both

Magpie intentionally combines:

- **ARC** for deterministic lifetime reclamation, and
- **Rust-like ownership/borrowing** for static alias/mutation safety.

## 12.1 Division of responsibilities

- ARC answers: “when can this heap object be released?”
- Ownership answers: “who may alias/mutate this value right now?”

This avoids both:
- manual memory management burden, and
- unconstrained shared-mutation hazards.

## 12.2 Performance model advantages

1. **Most values stay unique**
  - refcount often remains near 1 unless explicitly shared.
2. **Explicit sharing operations**
  - `share`, `clone.shared`, etc. make aliasing costs visible.
3. **Atomicity where needed**
  - shared/thread-crossing paths pay synchronization costs explicitly.

## 12.3 Safety and optimization advantages

- Mutations require exclusive pathways (unique or mutborrow).
- Borrow restrictions prevent common temporal aliasing mistakes.
- Explicit ownership boundaries make ARC optimization passes more reliable.

## 12.4 Conceptual ARC/ownership flow

```mp
%p: TPerson = new TPerson { name=%n, age=%a }
%s: shared TPerson = share { v=%p }
%s2: shared TPerson = clone.shared { v=%s }
; compiler/runtime manage release points for %s2 and %s
```

### 12.5 Ownership State Machine

```
                    ┌────────────┐
       new -------->│   Unique   │ (initial state, refcount=1)
                    │  (owned)   │
                    └──┬──┬──┬───┘
                       │  │  │
           share       │  │  │  borrow.shared
         (consumes)    │  │  │  (temporary)
                       │  │  │
        ┌──────────────┘  │  └──────────────┐
        v                 │                  v
  ┌───────────┐           │           ┌───────────┐
  │  Shared   │           │           │  Borrow   │
  │ (ARC, RC) │           │           │(read-only)│
  └──┬────┬───┘           │           └───────────┘
     │    │               │
     │    │ weak.         │ borrow.mut
     │    │ downgrade     │ (temporary, exclusive)
     │    │               │
     │  ┌─v────────┐   ┌─v───────────┐
     │  │   Weak   │   │  MutBorrow  │
     │  │(non-own) │   │ (exclusive) │
     │  └──────────┘   └─────────────┘
     │
     │ clone.shared (increments refcount)
     v
  ┌───────────┐
  │  Shared   │ (additional reference)
  │  (clone)  │
  └───────────┘

  RULES:
  - Borrows are block-scoped (cannot cross br/cbr boundaries)
  - Borrows cannot appear in phi nodes
  - Functions cannot return borrow values
  - mutborrow is exclusive: no other borrows or moves while active
  - share consumes the unique handle
  - ARC retain/release inserted automatically by Stage 8
```

### 12.6 Type System Hierarchy

```
  ┌─────────────────────────────────────────────────┐
  │                   All Types                      │
  ├──────────────────┬──────────────────────────────┤
  │   Value Types    │        Heap Types             │
  │  (stack/inline)  │     (ARC-managed handle)      │
  ├──────────────────┼──────────────────────────────┤
  │ Primitives:      │ Builtins:                     │
  │  i8..i128        │  Str                          │
  │  u8..u128        │  Array<T>                     │
  │  f16, f32, f64   │  Map<K, V>                    │
  │  bool, unit      │  TStrBuilder                  │
  │  i1, u1          │  TFuture<T>                   │
  │                  │  TMutex<T>, TRwLock<T>        │
  │ Value Structs:   │  TCell<T>                     │
  │  value struct T  │  TChannelSend/Recv<T>         │
  │                  │                               │
  │ Value Enums:     │ User Heap Types:              │
  │  TOption<T>      │  heap struct TName            │
  │  TResult<O, E>   │  heap enum TName              │
  │                  │                               │
  │                  │ Callable:                      │
  │                  │  TCallable<TSig>               │
  │                  │                               │
  │                  │ Pointer (unsafe):              │
  │                  │  rawptr<T>                     │
  └──────────────────┴──────────────────────────────┘

  Ownership modifiers (apply to heap types only):
    (none)     = unique owned handle
    shared     = reference-counted (ARC)
    borrow     = immutable reference
    mutborrow  = mutable exclusive reference
    weak       = non-owning reference

  Note: TOption and TResult are VALUE enums.
  shared/weak modifiers on TOption/TResult are rejected (MPT0002/MPT0003).
  Str has built-in hash, eq, ord impls (no explicit impl needed for Map keys).
```

---

## 13) Compiler Architecture

High-level crate roles:

- `magpie_cli`: command-line UX and config resolution
- `magpie_driver`: staged compilation orchestration
- `magpie_lex`: tokenization
- `magpie_parse`: recursive-descent parser
- `magpie_sema`: resolve/lowering/type checks/trait checks/v0.1 checks
- `magpie_hir`: HIR structures + verifier
- `magpie_own`: ownership checker
- `magpie_mpir`: MPIR + verifier + printer
- `magpie_arc`: ARC insertion/optimization passes
- `magpie_codegen_llvm`: LLVM lowering
- `magpie_codegen_wasm`: wasm lowering path
- `magpie_rt`: runtime ABI and support
- `magpie_gpu`: GPU codegen helpers/registries
- `magpie_web`: web framework + MCP integration paths
- `magpie_memory`: index/query workflows for memory/context artifacts

---

## 14) Compiler Pipeline

Driver stage names:

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

This staged structure is intentionally visible in output timing and diagnostics.

### 14.1 Pipeline Diagram

```
  .mp source
      │
      v
┌─────────────────────┐
│ Stage 1: Lex/Parse  │──> .ast.txt
│   magpie_lex         │
│   magpie_parse       │
│   magpie_csnf        │
└──────────┬──────────┘
           v
┌─────────────────────┐
│ Stage 2: Resolve    │    (imports, symbol tables)
│   magpie_sema        │
└──────────┬──────────┘
           v
┌─────────────────────┐
│ Stage 3: Typecheck  │    (AST -> HIR, type validation)
│   magpie_sema        │
│   magpie_types       │
└──────────┬──────────┘
           v
┌─────────────────────┐
│ Stage 3.5: Async    │    (coroutine state machines)
│   Lowering           │    suspend.call -> dispatch switch
└──────────┬──────────┘
           v
┌─────────────────────┐
│ Stage 4: Verify HIR │    (SSA, borrow invariants)
│   magpie_hir         │
└──────────┬──────────┘
           v
┌─────────────────────┐
│ Stage 5: Ownership  │    (move/borrow/alias rules)
│   magpie_own         │
└──────────┬──────────┘
           v
┌─────────────────────┐
│ Stage 6: Lower MPIR │──> .mpir
│   magpie_mpir        │
└──────────┬──────────┘
           v
┌─────────────────────┐
│ Stage 7: Verify MPIR│    (SID/CFG/type invariants)
│   magpie_mpir        │
└──────────┬──────────┘
           v
┌─────────────────────┐
│ Stage 8: ARC Insert │    (retain/release insertion)
│   magpie_arc         │
└──────────┬──────────┘
           v
┌─────────────────────┐
│ Stage 9: ARC Opt    │    (elide redundant refcounting)
│   magpie_arc         │
└──────────┬──────────┘
           v
┌─────────────────────┐
│ Stage 10: Codegen   │──> .ll (LLVM IR), .gpu_registry.ll
│   magpie_codegen_llvm│
│   magpie_gpu         │
└──────────┬──────────┘
           v
┌─────────────────────┐
│ Stage 11: Link      │──> native executable / shared lib
│   clang -x ir        │    (or lli for interpretation)
│   libmagpie_rt.a     │
└──────────┬──────────┘
           v
┌─────────────────────┐
│ Stage 12: MMS Update│──> .mms_index.json
│   magpie_memory      │
└─────────────────────┘
```

### 14.2 Crate Architecture

```
                    magpie_cli
                        │
                    magpie_driver ─────────────────────────────┐
                   /    │    \                                  │
          magpie_lex  magpie_sema  magpie_codegen_llvm    magpie_web
              │         │    │           │                     │
        magpie_parse  magpie_hir  magpie_arc            magpie_jit
              │         │           │
          magpie_ast  magpie_own  magpie_mpir
              │         │           │
          magpie_csnf magpie_types magpie_gpu
                        │
                    magpie_diag    magpie_rt (runtime)
                                   magpie_pkg
                                   magpie_memory
                                   magpie_ctx
                                   magpie_mono
```

---

## 15) Artifacts and Emission Kinds

Supported emit kinds include:

- `llvm-ir`
- `llvm-bc`
- `object`
- `asm`
- `spv`
- `exe`
- `shared-lib`
- `mpir`
- `mpd`
- `mpdbg`
- `symgraph`
- `depsgraph`
- `ownershipgraph`
- `cfggraph`

Typical usage:

```bash
magpie build --entry src/main.mp --emit mpir,llvm-ir,mpdbg
```

---

## 16) Diagnostics Model

Magpie diagnostics are structured objects with:

- code (`MPS0001`, `MPT2014`, etc.)
- severity
- message/title
- spans
- optional explanation / fix hints
- optional trace/rag/doc links

Common code families:

- `MPP*` parse/io/artifact
- `MPS*` resolve/SSA/structural invariants
- `MPT*` type/trait/v0.1 restrictions
- `MPO*` ownership/borrowing
- `MPF*` FFI
- `MPG*` GPU
- `MPL*` lint/link/LLM budget
- `MPW*` web
- `MPK*` package/dependency
- `MPM*` memory/index

Parse/JSON sema diagnostics (migration-focused):

| Code | Meaning | Trigger |
|---|---|---|
| `MPT2033` | Parse/JSON result shape mismatch | Result type is neither legacy shape nor `TResult<ok, err>` shape expected by the opcode |
| `MPT2034` | Parse/JSON input type mismatch | Parse/decode input is unknown or not `Str` / `borrow Str` |
| `MPT2035` | `json.encode<T>` value type mismatch | Encoded value type does not match generic target `T` (or value type is unknown) |

Explain command:

```bash
magpie explain MPT2014 --output json
```

---

## 17) CLI Arguments and Commands

This section is an implementation-level argument reference.

## 17.1 Global flags

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--output` | enum | `text` | `text`, `json`, `jsonl` |
| `--color` | enum | `auto` | `auto`, `always`, `never` |
| `--log-level` | enum | `warn` | `error`, `warn`, `info`, `debug`, `trace` |
| `--profile` | enum | `dev` | CLI parser accepts `dev`, `release`, `custom`; config maps non-release to dev |
| `--target` | string | host default | target triple |
| `--emit` | csv string | command-dependent | artifact kinds |
| `--entry` | path | manifest/default | source entry file |
| `--cache-dir` | path | none | cache path |
| `-j, --jobs` | int | none | parallel jobs |
| `--features` | csv string | empty | feature flags |
| `--no-default-features` | bool | false | disable default features |
| `--offline` | bool | false | dependency operations offline |
| `--llm` | bool | false | LLM-optimized output mode |
| `--no-auto-fmt` | bool | false | disable pre-build auto-format in llm mode |
| `--llm-token-budget` | int | resolved | output budget |
| `--llm-tokenizer` | string | resolved | tokenizer id |
| `--llm-budget-policy` | enum | resolved | `balanced`, `diagnostics_first`, `slices_first`, `minimal` |
| `--max-errors` | int | `20` | max diagnostics per pass |
| `--shared-generics` | bool | false | use shared generics mode |

## 17.2 Top-level commands

- `magpie new <name>`
- `magpie build`
- `magpie run [args...]`
- `magpie repl`
- `magpie fmt [--fix-meta]`
- `magpie parse [--emit ast]`
- `magpie lint`
- `magpie test [--filter <pattern>]`
- `magpie doc`
- `magpie mpir verify`
- `magpie explain <CODE>`
- `magpie pkg resolve|add|remove|why`
- `magpie web dev|build|serve`
- `magpie mcp serve`
- `magpie memory build|query --q <query> [--k 10]`
- `magpie ctx pack`
- `magpie ffi import --header <h> --out <mp>`
- `magpie graph symbols|deps|ownership|cfg`

## 17.3 Subcommand options

### `fmt`
- `--fix-meta`

### `parse`
- `--emit ast` (currently fixed to `ast`)

### `test`
- `--filter <pattern>`

### `memory query`
- `-q, --q <query>`
- `-k, --k <top_k>` (default 10)

### `ffi import`
- `--header <header-path>`
- `--out <output-path>`

## 17.4 Important defaults and behaviors

- `run` default emit:
 - release profile: `exe`
 - dev profile: `llvm-ir`
- `test` mode auto-adds `test` feature when absent.
- In `--llm` mode (unless `--no-auto-fmt`), auto-format precheck runs first.

---

## 18) Configuration Resolution Order

Selected settings are resolved from a combination of:

1. CLI flags
2. Environment variables (`MAGPIE_LLM`, `MAGPIE_LLM_TOKEN_BUDGET`)
3. `Magpie.toml` defaults (`[build]`, `[llm]`)
4. hardcoded defaults

### Manifest defaults (examples)

- `[build].entry`
- `[llm].mode_default`
- `[llm].token_budget`
- `[llm].tokenizer`
- `[llm].budget_policy`

---

## 19) Build/Test/Run Playbooks

## 19.1 Minimal compile

```bash
magpie build --entry src/main.mp --emit mpir,llvm-ir --output json
```

## 19.2 Run executable path

```bash
magpie run --profile release --entry src/main.mp --emit exe
```

## 19.3 Parse-only sanity

```bash
magpie parse --entry src/main.mp --output json
```

## 19.4 Graph-oriented introspection

```bash
magpie graph symbols --entry src/main.mp --output json
magpie graph deps --entry src/main.mp --output json
magpie graph ownership --entry src/main.mp --output json
magpie graph cfg --entry src/main.mp --output json
```

## 19.5 Diagnostic-driven iteration

```bash
magpie build --entry src/main.mp --output json --emit mpir,llvm-ir,mpdbg
magpie explain MPS0024 --output json
```

---

## 20) Examples

## 20.1 Minimal hello-style module

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

## 20.2 Struct mutate/read with proper borrows

```mp
module demo.point
exports { @main }
imports { }
digest "0000000000000000"

heap struct TPoint {
 field x: i64
 field y: i64
}

fn @main() -> i64 {
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

## 20.3 TCallable callback example

```mp
module demo.callable
exports { @main, @multiply_by }
imports { }
digest "0000000000000000"

sig TMulSig(i32) -> i32

fn @multiply_by(%x: i32, %factor: i32) -> i32 {
bb0:
 %y: i32 = i.mul { lhs=%x, rhs=%factor }
 ret %y
}

fn @main() -> i32 {
bb0:
 %factor: i32 = const.i32 3
 %mul_by_3: TCallable<TMulSig> = callable.capture @multiply_by { factor=%factor }
 %result: i32 = call.indirect %mul_by_3 { args=[const.i32 7] }
 ret %result
}
```

---

### Notes

- This document is intentionally comprehensive and practical.
- For binary-only operation, prioritize `--output json`, `magpie explain <CODE>`, and emitted artifacts.
- For source-level implementation changes, use stage-specific diagnostics and crate boundaries to localize fixes.

---

## Appendix A — Formal Grammar (Extended EBNF)

The following is an extended grammar sketch aligned with v0.1 parser behavior.

```ebnf
file     := header decl*
header    := module_decl exports_decl imports_decl digest_decl
module_decl  := "module" module_path
exports_decl := "exports" "{" export_item_list? "}"
imports_decl := "imports" "{" import_group_list? "}"
digest_decl  := "digest" string_lit

export_item_list := export_item ("," export_item)*
export_item   := fn_name | type_name

import_group_list := import_group ("," import_group)*
import_group   := module_path "::" "{" import_item_list? "}"
import_item_list := import_item ("," import_item)*
import_item    := fn_name | type_name

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

fn_decl    := doc* "fn" fn_name "(" params? ")" "->" type fn_meta? blocks
async_fn_decl := doc* "async" "fn" fn_name "(" params? ")" "->" type fn_meta? blocks
unsafe_fn_decl := doc* "unsafe" "fn" fn_name "(" params? ")" "->" type fn_meta? blocks
gpu_fn_decl  := doc* "gpu" "fn" fn_name "(" params? ")" "->" type "target" "(" ident ")" fn_meta? blocks

fn_meta := "meta" "{" (meta_uses | meta_effects | meta_cost)* "}"
meta_uses  := "uses" "{" fqn_list? "}"
meta_effects := "effects" "{" ident_list? "}"
meta_cost  := "cost" "{" kv_i64_list? "}"

params := param ("," param)*
param := ssa_name ":" type

heap_struct_decl := doc* "heap" "struct" type_name type_params? "{" field_decl* "}"
value_struct_decl := doc* "value" "struct" type_name type_params? "{" field_decl* "}"
field_decl    := "field" ident ":" type

heap_enum_decl := doc* "heap" "enum" type_name type_params? "{" variant_decl* "}"
value_enum_decl := doc* "value" "enum" type_name type_params? "{" variant_decl* "}"
variant_decl  := "variant" ident "{" field_decl_inline_list? "}"

extern_decl := doc* "extern" string_lit "module" ident "{" extern_item* "}"
extern_item := "fn" fn_name "(" params? ")" "->" type attrs_block?
attrs_block := "attrs" "{" kv_string_list? "}"

global_decl := doc* "global" fn_name ":" type "=" const_expr
impl_decl  := "impl" ident "for" type "=" fn_ref
sig_decl  := "sig" type_name "(" type_list? ")" "->" type

blocks := "{" block+ "}"
block := block_label ":" instr* terminator

instr := assign_instr
   | void_instr
   | unsafe_block

assign_instr := ssa_name ":" type "=" value_op
void_instr  := void_op
unsafe_block := "unsafe" "{" (assign_instr | void_instr)+ "}"

terminator := "ret" value_ref?
      | "br" block_label
      | "cbr" value_ref block_label block_label
      | "switch" value_ref "{" switch_arms* "}" "else" block_label
      | "unreachable"

switch_arms := "case" const_lit "->" block_label

value_ref := ssa_name | const_expr
const_expr := "const" "." type const_lit

type := ownership_mod? base_type
ownership_mod := "shared" | "borrow" | "mutborrow" | "weak"
base_type := prim_type
     | builtin_type
     | named_type
     | rawptr_type
     | callable_type

rawptr_type := "rawptr" "<" type ">"
callable_type := "TCallable" "<" type_ref ">"
named_type := (module_path ".")? type_name type_args?
type_args := "<" type_list ">"
type_params := "<" type_param_list ">"
type_param_list := type_param ("," type_param)*
type_param := ident ":" ident

module_path := ident ("." ident)*
fn_ref := fn_name | module_path "." fn_name
type_ref := type_name | module_path "." type_name

prim_type := "i1" | "i8" | "i16" | "i32" | "i64" | "i128"
      | "u1" | "u8" | "u16" | "u32" | "u64" | "u128"
      | "f16" | "f32" | "f64"
      | "bool" | "unit"

builtin_type := "Str"
       | "Array" "<" type ">"
       | "Map" "<" type "," type ">"
       | "TOption" "<" type ">"
       | "TResult" "<" type "," type ">"
       | "TStrBuilder"
       | "TMutex" "<" type ">"
       | "TRwLock" "<" type ">"
       | "TCell" "<" type ">"
       | "TFuture" "<" type ">"
       | "TChannelSend" "<" type ">"
       | "TChannelRecv" "<" type ">"
```

---

## Appendix B — Value Opcode Syntax Matrix (Exhaustive v0.1 Surface)

### B.1 Arithmetic and comparison

```mp
i.add { lhs=V, rhs=V }
i.sub { lhs=V, rhs=V }
i.mul { lhs=V, rhs=V }
i.sdiv { lhs=V, rhs=V }
i.udiv { lhs=V, rhs=V }
i.srem { lhs=V, rhs=V }
i.urem { lhs=V, rhs=V }
i.add.wrap { lhs=V, rhs=V }
i.sub.wrap { lhs=V, rhs=V }
i.mul.wrap { lhs=V, rhs=V }
i.add.checked { lhs=V, rhs=V }
i.sub.checked { lhs=V, rhs=V }
i.mul.checked { lhs=V, rhs=V }
i.and { lhs=V, rhs=V }
i.or { lhs=V, rhs=V }
i.xor { lhs=V, rhs=V }
i.shl { lhs=V, rhs=V }
i.lshr { lhs=V, rhs=V }
i.ashr { lhs=V, rhs=V }

f.add { lhs=V, rhs=V }
f.sub { lhs=V, rhs=V }
f.mul { lhs=V, rhs=V }
f.div { lhs=V, rhs=V }
f.rem { lhs=V, rhs=V }
f.add.fast { lhs=V, rhs=V }
f.sub.fast { lhs=V, rhs=V }
f.mul.fast { lhs=V, rhs=V }
f.div.fast { lhs=V, rhs=V }

icmp.eq { lhs=V, rhs=V }
icmp.ne { lhs=V, rhs=V }
icmp.slt { lhs=V, rhs=V }
icmp.sgt { lhs=V, rhs=V }
icmp.sle { lhs=V, rhs=V }
icmp.sge { lhs=V, rhs=V }
icmp.ult { lhs=V, rhs=V }
icmp.ugt { lhs=V, rhs=V }
icmp.ule { lhs=V, rhs=V }
icmp.uge { lhs=V, rhs=V }

fcmp.oeq { lhs=V, rhs=V }
fcmp.one { lhs=V, rhs=V }
fcmp.olt { lhs=V, rhs=V }
fcmp.ogt { lhs=V, rhs=V }
fcmp.ole { lhs=V, rhs=V }
fcmp.oge { lhs=V, rhs=V }
```

### B.2 Calls and async-related

```mp
call @fn<TypeArgs?> { key=Arg,... }
call.indirect V { key=Arg,... }
try @fn<TypeArgs?> { key=Arg,... }
suspend.call @fn<TypeArgs?> { key=Arg,... }
suspend.await { fut=V }
```

### B.3 Heap/object/enum

```mp
new Type { field=V,... }
getfield { obj=V, field=name }
phi Type { [bbN:V], [bbM:V],... }

enum.new<Variant> { key=V,... }
enum.tag { v=V }
enum.payload<Variant> { v=V }
enum.is<Variant> { v=V }
```

### B.4 Ownership/pointer/callable

```mp
share { v=V }
clone.shared { v=V }
clone.weak { v=V }
weak.downgrade { v=V }
weak.upgrade { v=V }
cast<PrimFrom, PrimTo> { v=V }
borrow.shared { v=V }
borrow.mut { v=V }

ptr.null<T>
ptr.addr<T> { p=V }
ptr.from_addr<T> { addr=V }
ptr.add<T> { p=V, count=V }
ptr.load<T> { p=V }

callable.capture @fn { cap_name=V,... }
```

### B.5 Collections, strings, JSON, GPU

```mp
arr.new<T> { cap=V }
arr.len { arr=V }
arr.get { arr=V, idx=V }
arr.pop { arr=V }
arr.slice { arr=V, start=V, end=V }
arr.contains { arr=V, val=V }
arr.map { arr=V, fn=V }
arr.filter { arr=V, fn=V }
arr.reduce { arr=V, init=V, fn=V }

map.new<K, V> { }
map.len { map=V }
map.get { map=V, key=V }
map.get_ref { map=V, key=V }
map.delete { map=V, key=V }
map.contains_key { map=V, key=V }
map.keys { map=V }
map.values { map=V }

str.concat { a=V, b=V }
str.len { s=V }
str.eq { a=V, b=V }
str.slice { s=V, start=V, end=V }
str.bytes { s=V }
str.builder.new { }
str.builder.build { b=V }
str.parse_i64 { s=V }
str.parse_u64 { s=V }
str.parse_f64 { s=V }
str.parse_bool { s=V }
json.encode<T> { v=V }
json.decode<T> { s=V }

gpu.thread_id { dim=V }
gpu.workgroup_id { dim=V }
gpu.workgroup_size { dim=V }
gpu.global_id { dim=V }
gpu.buffer_load<T> { buf=V, idx=V }
gpu.buffer_len<T> { buf=V }
gpu.shared<count, T>

gpu.launch { device=V, kernel=@fn, grid=Arg, block=Arg, args=Arg }
gpu.launch_async { device=V, kernel=@fn, grid=Arg, block=Arg, args=Arg }
```

Compatibility note:
- The source op names above are stable.
- Internally, parse/json codegen now targets `*_try_*` runtime symbols with explicit status branching at the ABI boundary.

---

## Appendix C — Void Opcode Syntax Matrix (Exhaustive v0.1 Surface)

```mp
call_void @fn<TypeArgs?> { key=Arg,... }
call_void.indirect V { key=Arg,... }
setfield { obj=V, field=name, val=V }
panic { msg=V }
ptr.store<T> { p=V, v=V }

arr.set { arr=V, idx=V, val=V }
arr.push { arr=V, val=V }
arr.sort { arr=V }
arr.foreach { arr=V, fn=V }

map.set { map=V, key=V, val=V }
map.delete_void { map=V, key=V }

str.builder.append_str { b=V, s=V }
str.builder.append_i64 { b=V, v=V }
str.builder.append_i32 { b=V, v=V }
str.builder.append_f64 { b=V, v=V }
str.builder.append_bool { b=V, v=V }

gpu.barrier
gpu.buffer_store<T> { buf=V, idx=V, v=V }
```

---

## Appendix D — Compiler Arguments: Complete Command Matrix

### D.1 Global invocation forms

```bash
magpie [GLOBAL_FLAGS] <command> [SUBCOMMAND_FLAGS]
```

### D.2 Global flag examples

```bash
magpie --output json --entry src/main.mp build
magpie --profile release --target x86_64-unknown-linux-gnu --emit exe run
magpie --llm --llm-token-budget 12000 --llm-budget-policy balanced build
magpie --features test,web --no-default-features test --filter callable
```

### D.3 Command examples

```bash
magpie new demo_project
magpie fmt --fix-meta
magpie parse --entry src/main.mp --emit ast
magpie lint --entry src/main.mp
magpie doc
magpie explain MPO0102
magpie mpir verify --entry src/main.mp
magpie graph symbols --entry src/main.mp
magpie graph deps --entry src/main.mp
magpie graph ownership --entry src/main.mp
magpie graph cfg --entry src/main.mp
magpie ffi import --header ffi.h --out ffi_bindings.mp
magpie pkg resolve
magpie pkg add serde_like
magpie pkg remove serde_like
magpie pkg why std
magpie web dev
magpie web build
magpie web serve
magpie mcp serve
magpie memory build --entry src/main.mp
magpie memory query -q "ownershipgraph borrow phi" -k 10 --entry src/main.mp
magpie ctx pack --entry src/main.mp
```

---

## Appendix E — Operational Rationale Matrix

| Claim | Operational basis in Magpie | Confidence |
|---|---|---|
| Bounded output design is necessary | Token-budget options, JSON envelopes, and progressive emit strategy | High |
| Locality improves practical reliability | Explicit control flow, explicit ownership ops, explicit call forms | High |
| Canonical formatting improves stability | CSNF formatting and deterministic output behavior | High |
| Progressive disclosure outperforms bulk dumps in workflows | `mpdbg`, graph emits, and memory/query workflows | High |
| Stable symbol identity helps incremental workflows | Symbol/dependency graphs and deterministic artifact surfaces | Medium-High |
| Canonicalization reduces iterative edit drift | Deterministic formatting and reduced syntactic variance | Medium |

---

## Appendix F — Specific Notes Requested for v0.1
