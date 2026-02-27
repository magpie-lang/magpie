# Magpie v0.1 — Comprehensive Quickstart Guide

> Fast path into the Magpie compiler. Full reference: [`DOCUMENTATION.md`](./DOCUMENTATION.md)

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Install and Build the CLI](#2-install-and-build-the-cli)
3. [Create Your First Project](#3-create-your-first-project)
4. [Minimal Program](#4-minimal-program)
5. [Compiler Pipeline (13 Stages)](#5-compiler-pipeline-13-stages)
6. [Project Structure](#6-project-structure)
7. [Build Artifact Flow](#7-build-artifact-flow)
8. [Type System Hierarchy](#8-type-system-hierarchy)
9. [Ownership and Borrow Lifecycle](#9-ownership-and-borrow-lifecycle)
10. [CLI Command Reference](#10-cli-command-reference)
11. [Complete Working Examples](#11-complete-working-examples)
12. [Test Fixtures Reference](#12-test-fixtures-reference)
13. [Common Pitfalls and Fixes](#13-common-pitfalls-and-fixes)
14. [Diagnostic Code Families](#14-diagnostic-code-families)
15. [LLM/Agent Mode](#15-llmagent-mode)
16. [Where to Go Next](#16-where-to-go-next)

---

## 1) Prerequisites

### Required

- **Rust toolchain** (stable, 1.75+)
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  rustup update stable
  ```

- **Cargo** (ships with Rust)

### Required for `magpie run` (uses `lli`)

`magpie run` invokes the LLVM interpreter (`lli`) to execute the compiled LLVM IR.
You must have LLVM tools on your PATH:

```bash
# macOS (Homebrew)
brew install llvm
export PATH="$(brew --prefix llvm)/bin:$PATH"

# Ubuntu/Debian
sudo apt-get install llvm

# Verify
lli --version
```

### Required for native executables (`--emit exe`)

`magpie build --emit exe` links via `clang`. Install clang:

```bash
# macOS: ships with Xcode command line tools
xcode-select --install

# Ubuntu/Debian
sudo apt-get install clang

# Verify
clang --version
```

---

## 2) Install and Build the CLI

```bash
# Clone the repo
git clone <magpie-repo-url>
cd magpie

# Build the CLI binary
cargo build -p magpie_cli

# Verify
cargo run -p magpie_cli -- --help
```

### Important: global flags go BEFORE the subcommand

```bash
# Correct: flags before subcommand
cargo run -p magpie_cli -- --entry src/main.mp --emit mpir,llvm-ir --output json build

# Wrong: flags after subcommand (will error or be ignored)
cargo run -p magpie_cli -- build --entry src/main.mp
```

The global flags `--entry`, `--emit`, `--output`, `--profile`, `--llm`, etc. must
all appear before the subcommand name.

### The `--entry` flag is relative to CWD

`--entry` resolves relative to your **current working directory**, not the project root.
If you are in `magpie/` and your file is `demo/src/main.mp`, use:

```bash
cargo run -p magpie_cli -- --entry demo/src/main.mp build
```

---

## 3) Create Your First Project

```bash
# Create a new project named "hello"
cargo run -p magpie_cli -- new hello

# The scaffolded structure:
hello/
  Magpie.toml
  src/
    main.mp

# Build it
cd hello
cargo run -p magpie_cli -- --output json build
```

### Magpie.toml example

```toml
[package]
name = "hello"
version = "0.1.0"

[build]
entry = "src/main.mp"

[llm]
mode_default = false
token_budget = 8000
budget_policy = "balanced"
```

---

## 4) Minimal Program

`src/main.mp`

```mp
module hello.main
exports { @main }
imports { }
digest "0000000000000000"

fn @main() -> i32 {
bb0:
  ret const.i32 0
}
```

**Key rule:** the return type determines the constant type.
- `-> i32` requires `const.i32 0`  (NOT `const.i64 0`)
- `-> i64` requires `const.i64 0`
- `-> bool` requires `true` or `false`

### Build and run

```bash
# Build only (emits MPIR)
cargo run -p magpie_cli -- --entry src/main.mp build

# Build with multiple artifact kinds
cargo run -p magpie_cli -- --entry src/main.mp --emit mpir,llvm-ir,mpdbg --output json build

# Build a native executable via clang
cargo run -p magpie_cli -- --entry src/main.mp --emit exe build

# Run via lli (LLVM interpreter) — requires lli on PATH
cargo run -p magpie_cli -- --entry src/main.mp run
```

### Header order is mandatory

Every `.mp` file must have this exact header sequence:

```
1.  module <path>
2.  exports { ... }
3.  imports { ... }
4.  digest "<hex>"
```

Any deviation causes an `MPP*` parse error.

---

## 5) Compiler Pipeline (13 Stages)

```
Source (.mp)
     |
     v
+----+------------------------------------------------------------+
|  Stage 1: stage1_read_lex_parse                                 |
|  Tokenize (magpie_lex) -> Parse (magpie_parse) -> CST           |
+----+------------------------------------------------------------+
     |
     v
+----+------------------------------------------------------------+
|  Stage 2: stage2_resolve                                        |
|  Symbol resolution, import linking, scope analysis              |
|  (magpie_sema)                                                  |
+----+------------------------------------------------------------+
     |
     v
+----+------------------------------------------------------------+
|  Stage 3: stage3_typecheck                                      |
|  Type inference, type compatibility, trait checking             |
|  (magpie_sema)                                                  |
+----+------------------------------------------------------------+
     |
     v
+----+------------------------------------------------------------+
|  Stage 3.5: stage3_5_async_lowering                             |
|  Desugar async fn / suspend.call into state machines            |
|  (magpie_sema)                                                  |
+----+------------------------------------------------------------+
     |
     v
+----+------------------------------------------------------------+
|  Stage 4: stage4_verify_hir                                     |
|  HIR structural invariants (magpie_hir verifier)                |
+----+------------------------------------------------------------+
     |
     v
+----+------------------------------------------------------------+
|  Stage 5: stage5_ownership_check                                |
|  Borrow/move/alias safety (magpie_own)                          |
+----+------------------------------------------------------------+
     |
     v
+----+------------------------------------------------------------+
|  Stage 6: stage6_lower_mpir                                     |
|  Lower HIR -> MPIR representation (magpie_mpir)                 |
+----+------------------------------------------------------------+
     |
     v
+----+------------------------------------------------------------+
|  Stage 7: stage7_verify_mpir                                    |
|  MPIR structural invariants, SSA form checks                    |
+----+------------------------------------------------------------+
     |
     v
+----+------------------------------------------------------------+
|  Stage 8: stage8_arc_insertion                                  |
|  Insert ARC retain/release operations (magpie_arc)              |
+----+------------------------------------------------------------+
     |
     v
+----+------------------------------------------------------------+
|  Stage 9: stage9_arc_optimization                               |
|  Optimize ARC operations, elide redundant ref counts            |
+----+------------------------------------------------------------+
     |
     v
+----+------------------------------------------------------------+
|  Stage 10: stage10_codegen                                      |
|  Lower MPIR -> LLVM IR / WASM / SPIR-V (magpie_codegen_*)      |
+----+------------------------------------------------------------+
     |
     v
+----+------------------------------------------------------------+
|  Stage 11: stage11_link                                         |
|  Link object files into exe/shared-lib via clang                |
+----+------------------------------------------------------------+
     |
     v
+----+------------------------------------------------------------+
|  Stage 12: stage12_mms_update                                   |
|  Update module memory/symbol index (.mpd files)                 |
+----+------------------------------------------------------------+
     |
     v
Artifacts: .mpir  .ll  .bc  .o  .exe  .mpdbg  .mpd  graphs
```

### Crate responsibilities

| Crate | Role |
|---|---|
| `magpie_cli` | CLI UX, config resolution |
| `magpie_driver` | Stage orchestration |
| `magpie_lex` | Tokenization |
| `magpie_parse` | Recursive-descent parser |
| `magpie_sema` | Resolve / type check / trait check / async lower |
| `magpie_hir` | HIR structures + verifier |
| `magpie_own` | Ownership/borrow checker |
| `magpie_mpir` | MPIR + verifier + printer |
| `magpie_arc` | ARC insertion + optimization passes |
| `magpie_codegen_llvm` | LLVM IR lowering |
| `magpie_codegen_wasm` | WASM lowering |
| `magpie_rt` | Runtime ABI / support |
| `magpie_gpu` | GPU codegen helpers |
| `magpie_memory` | Memory index / query workflows |

---

## 6) Project Structure

```
hello/                         <- project root
  Magpie.toml                  <- manifest (name, version, build.entry, llm config)
  src/
    main.mp                    <- entry module
    lib.mp                     <- optional library root
    utils/
      math.mp                  <- submodule (module hello.utils.math)
      strings.mp
  tests/
    unit.mp                    <- test modules
  out/                         <- build artifacts (auto-created)
    hello.mpir                 <- MPIR artifact (single module)
    hello.ll                   <- LLVM IR
    hello.mpdbg                <- debug symbol/context bundle
    hello.0.mpir               <- indexed artifact (multi-module builds)
    hello.1.mpir
  .magpie/
    mms/                       <- module memory store (.mpd files)
    cache/                     <- incremental build cache
```

### Multi-module builds produce indexed artifacts

When your entry module imports other modules, the build produces one artifact
per compilation unit, indexed numerically:

```
hello.0.mpir   <- entry module
hello.1.mpir   <- first imported module
hello.2.mpir   <- second imported module
hello.0.ll
hello.1.ll
```

---

## 7) Build Artifact Flow

```
src/main.mp
     |
     | --emit mpir
     v
  hello.mpir          <- Magpie IR (text, inspectable)

     |
     | --emit llvm-ir
     v
  hello.ll            <- LLVM IR (text, pass to llc/lli/clang)

     |
     | --emit llvm-bc
     v
  hello.bc            <- LLVM Bitcode (binary)

     |
     | --emit object
     v
  hello.o             <- Native object file

     |
     | --emit exe        (links via clang)
     v
  hello / hello.exe   <- Native executable

     |
     | --emit asm
     v
  hello.s             <- Native assembly

     |
     | --emit spv
     v
  hello.spv           <- SPIR-V (GPU compute)

     |
     | --emit mpdbg
     v
  hello.mpdbg         <- Debug bundle for LLM/agent workflows

     |
     | --emit symgraph / depsgraph / ownershipgraph / cfggraph
     v
  hello.*.dot         <- Graph artifacts for tooling
```

### Combining emit kinds

```bash
# Emit everything useful for debugging
cargo run -p magpie_cli -- \
  --entry src/main.mp \
  --emit mpir,llvm-ir,mpdbg,symgraph \
  --output json \
  build
```

### `magpie run` uses `lli`

`magpie run` compiles to LLVM IR then invokes `lli` (the LLVM interpreter).
It does NOT produce a native binary. For a native executable, use `--emit exe`.

```bash
# Interpreted execution (requires lli)
cargo run -p magpie_cli -- --entry src/main.mp run

# Native executable
cargo run -p magpie_cli -- --entry src/main.mp --emit exe build
./out/hello
```

---

## 8) Type System Hierarchy

```
Types
  |
  +-- Primitive types
  |     |
  |     +-- Signed integers:   i1  i8  i16  i32  i64  i128
  |     +-- Unsigned integers: u1  u8  u16  u32  u64  u128
  |     +-- Floats:            f16  f32  f64
  |     +-- Other:             bool  unit
  |
  +-- Ownership-qualified types
  |     |
  |     +-- shared T          (ARC-managed shared ownership)
  |     +-- borrow T          (immutable reference, no escape)
  |     +-- mutborrow T       (mutable reference, no escape)
  |     +-- weak T            (non-owning weak reference)
  |     +-- rawptr<T>         (unsafe raw pointer)
  |
  +-- Builtin aggregate types
  |     |
  |     +-- Str               (UTF-8 string; built-in hash/eq/ord)
  |     +-- Array<T>          (growable array)
  |     +-- Map<K, V>         (hash map; K needs hash + eq)
  |     +-- TOption<T>        (Some/None)
  |     +-- TResult<Ok, Err>  (Ok/Err)
  |     +-- TStrBuilder       (mutable string builder)
  |     +-- TMutex<T>         (mutex-guarded value)
  |     +-- TRwLock<T>        (reader-writer lock)
  |     +-- TCell<T>          (interior mutability cell)
  |     +-- TFuture<T>        (async future)
  |     +-- TChannelSend<T>   (channel sender)
  |     +-- TChannelRecv<T>   (channel receiver)
  |     +-- TCallable<TSig>   (first-class callable with explicit captures)
  |
  +-- User-defined types
        |
        +-- heap struct TName { field name: Type, ... }
        +-- value struct TName { field name: Type, ... }
        +-- heap enum TName { variant V { field ...; } ... }
        +-- value enum TName { variant V { field ...; } ... }
```

### `Str` has built-in trait implementations

`Str` automatically satisfies `hash`, `eq`, and `ord`. You do NOT need explicit
`impl` declarations to use `Str` as a `Map` key or in comparisons:

```mp
; This works with no impl boilerplate:
%m: Map<Str, i32> = map.new<Str, i32> { }
```

For user-defined types used as `Map` keys, you must provide:

```mp
sig TMyHash(borrow TMyKey) -> u64
impl hash for TMyKey = @my_hash

sig TMyEq(borrow TMyKey, borrow TMyKey) -> bool
impl eq for TMyKey = @my_eq
```

---

## 9) Ownership and Borrow Lifecycle

```
Value created (stack or heap)
        |
        v
   %p: TPerson = new TPerson { name=%n, age=%a }
        |
        | (unique ownership - one owner)
        v
   Unique value: can be read and mutated directly
        |
        +------ borrow.shared ----> %pb: borrow TPerson
        |                                  |
        |                           read-only access
        |                           getfield, arr.len, etc.
        |                           CANNOT cross block boundary
        |                           CANNOT appear in phi
        |                           CANNOT be returned
        |
        +------ borrow.mut -------> %pm: mutborrow TPerson
        |                                  |
        |                           read + write access
        |                           setfield, arr.push, map.set, etc.
        |                           same restrictions as borrow
        |
        +------ share ------------> %sp: shared TPerson
        |                                  |
        |                           ARC-managed shared ownership
        |                           refcount = 1+
        |                           can be cloned and stored
        |
        +------ (from shared) ----> clone.shared -> %sp2: shared TPerson
        |                                  |
        |                           new shared handle, refcount++
        |
        +------ (from shared) ----> weak.downgrade -> %w: weak TPerson
                                           |
                                    non-owning reference
                                    weak.upgrade -> TOption<shared TPerson>
```

### Borrow rules summary

| Rule | Consequence if violated |
|---|---|
| Borrows cannot cross basic block boundaries | `MPO*` error |
| Borrows cannot appear in `phi` | `MPO*` error |
| Borrows cannot be returned from functions | `MPO*` error |
| `getfield` requires `borrow` or `mutborrow` receiver | `MPT*` error |
| `setfield` requires `mutborrow` receiver | `MPT*` error |
| `arr.push`, `map.set` require `mutborrow` receiver | `MPT*` error |
| `arr.len`, `map.contains_key` require `borrow` receiver | `MPT*` error |

### Correct borrow pattern

```mp
fn @example() -> i64 {
bb0:
  %arr: Array<i64> = arr.new<i64> { cap=const.i64 8 }
  %arr_m: mutborrow Array<i64> = borrow.mut { v=%arr }
  arr.push { arr=%arr_m, val=const.i64 42 }
  br bb1          ; <-- drop the mutborrow before new block

bb1:
  %arr_b: borrow Array<i64> = borrow.shared { v=%arr }
  %len: i64 = arr.len { arr=%arr_b }
  ret %len        ; <-- borrow used and dropped in same block
}
```

---

## 10) CLI Command Reference

### Global flags (always before the subcommand)

| Flag | Values | Default | Description |
|---|---|---|---|
| `--output` | `text`, `json`, `jsonl` | `text` | Output format |
| `--color` | `auto`, `always`, `never` | `auto` | Terminal color |
| `--log-level` | `error`, `warn`, `info`, `debug`, `trace` | `warn` | Logging verbosity |
| `--profile` | `dev`, `release` | `dev` | Build profile |
| `--target` | triple string | host | Target triple |
| `--emit` | CSV of artifact kinds | command default | Artifacts to produce |
| `--entry` | file path | manifest `build.entry` | Entry `.mp` file (relative to CWD) |
| `--cache-dir` | path | none | Cache directory |
| `-j, --jobs` | int | none | Parallel jobs |
| `--features` | CSV string | empty | Feature flags |
| `--llm` | flag | false | LLM-optimized output |
| `--llm-token-budget` | int | resolved | Token budget |
| `--llm-tokenizer` | string | resolved | Tokenizer ID |
| `--llm-budget-policy` | `balanced`, `diagnostics_first`, `slices_first`, `minimal` | resolved | Budget policy |
| `--max-errors` | int | 20 | Max diagnostics per pass |
| `--no-auto-fmt` | flag | false | Disable pre-build auto-format in LLM mode |

### Subcommands

#### `new` — scaffold a project

```bash
cargo run -p magpie_cli -- new <project-name>
```

#### `build` — compile

```bash
# Default build
cargo run -p magpie_cli -- --entry src/main.mp build

# Full artifact set
cargo run -p magpie_cli -- --entry src/main.mp --emit mpir,llvm-ir,mpdbg,exe --output json build

# Release profile
cargo run -p magpie_cli -- --entry src/main.mp --profile release --emit exe build
```

Supported emit kinds: `mpir`, `llvm-ir`, `llvm-bc`, `object`, `asm`, `exe`,
`shared-lib`, `spv`, `mpd`, `mpdbg`, `symgraph`, `depsgraph`, `ownershipgraph`, `cfggraph`

#### `run` — interpret via lli

```bash
# Requires lli on PATH
cargo run -p magpie_cli -- --entry src/main.mp run

# Pass arguments to the program
cargo run -p magpie_cli -- --entry src/main.mp run -- arg1 arg2
```

#### `fmt` — format source

```bash
# Format and update digest hashes
cargo run -p magpie_cli -- fmt --fix-meta

# Format single file
cargo run -p magpie_cli -- --entry src/main.mp fmt --fix-meta
```

#### `parse` — parse only

```bash
cargo run -p magpie_cli -- --entry src/main.mp --output json parse
```

#### `lint` — lint checks

```bash
cargo run -p magpie_cli -- --entry src/main.mp --output json lint
```

#### `test` — run tests

```bash
cargo run -p magpie_cli -- test

# Filter by pattern
cargo run -p magpie_cli -- test --filter "arithmetic"
```

#### `mpir verify` — verify MPIR

```bash
cargo run -p magpie_cli -- --entry src/main.mp --output json mpir verify
```

#### `explain` — explain a diagnostic code

```bash
cargo run -p magpie_cli -- --output json explain MPT2014
cargo run -p magpie_cli -- explain MPS0001
```

#### `graph` — emit graph artifacts

```bash
cargo run -p magpie_cli -- --entry src/main.mp --output json graph symbols
cargo run -p magpie_cli -- --entry src/main.mp --output json graph deps
cargo run -p magpie_cli -- --entry src/main.mp --output json graph ownership
cargo run -p magpie_cli -- --entry src/main.mp --output json graph cfg
```

#### `memory` — build and query memory index

```bash
# Build index
cargo run -p magpie_cli -- --entry src/main.mp --output json memory build

# Query index
cargo run -p magpie_cli -- --entry src/main.mp --output json memory query -q "borrow phi" -k 10
```

#### `ffi import` — generate bindings from C header

```bash
cargo run -p magpie_cli -- --output json ffi import --header mylib.h --out ffi_bindings.mp
```

#### `doc` — generate documentation

```bash
cargo run -p magpie_cli -- --entry src/main.mp doc
```

#### `pkg` — package management

```bash
cargo run -p magpie_cli -- pkg resolve
cargo run -p magpie_cli -- pkg add <package>
cargo run -p magpie_cli -- pkg remove <package>
cargo run -p magpie_cli -- pkg why <package>
```

---

## 11) Complete Working Examples

### 11.1 Arithmetic

```mp
module test.arithmetic
exports { @add, @checked_add }
imports { }
digest "191ba41acdc8e7e544162b0c7d87738c9ca4c65c00b59098bb7ce18de0786c58"

fn @add(%a: i32, %b: i32) -> i32 meta { } {
bb0:
  %sum: i32 = i.add { lhs=%a, rhs=%b }
  ret %sum
}

fn @checked_add(%a: i32, %b: i32) -> TOption<i32> meta { } {
bb0:
  %result: TOption<i32> = i.add.checked { lhs=%a, rhs=%b }
  ret %result
}
```

Key arithmetic opcodes:

| Opcode | Description |
|---|---|
| `i.add { lhs=, rhs= }` | Integer add (wrapping) |
| `i.sub { lhs=, rhs= }` | Integer subtract |
| `i.mul { lhs=, rhs= }` | Integer multiply |
| `i.sdiv { lhs=, rhs= }` | Signed integer divide |
| `i.udiv { lhs=, rhs= }` | Unsigned integer divide |
| `i.add.checked { lhs=, rhs= }` | Returns `TOption<T>` (None on overflow) |
| `i.add.wrap { lhs=, rhs= }` | Explicit wrapping add |
| `f.add { lhs=, rhs= }` | Float add |
| `f.mul.fast { lhs=, rhs= }` | Float multiply (fast-math) |

### 11.2 Structs and Field Access

```mp
module demo.structs
exports { @make_point, @sum_point }
imports { }
digest "0000000000000000"

heap struct TPoint {
  field x: i64
  field y: i64
}

fn @make_point(%x: i64, %y: i64) -> shared TPoint meta { } {
bb0:
  %p: TPoint = new TPoint { x=%x, y=%y }
  %sp: shared TPoint = share { v=%p }
  ret %sp
}

fn @sum_point(%sp: shared TPoint) -> i64 meta { } {
bb0:
  %pb: borrow TPoint = borrow.shared { v=%sp }
  %x: i64 = getfield { field=x, obj=%pb }
  %y: i64 = getfield { field=y, obj=%pb }
  %s: i64 = i.add { lhs=%x, rhs=%y }
  ret %s
}
```

**Notes on `getfield` / `setfield`:**
- Key arguments accept ANY order (not strict ordering required)
- `getfield` requires a `borrow` or `mutborrow` receiver
- `setfield` requires a `mutborrow` receiver and uses `val=` (not `value=`)

```mp
; Reading a field
%x: i64 = getfield { field=x, obj=%pb }     ; any key order works
%x: i64 = getfield { obj=%pb, field=x }     ; also valid

; Writing a field (requires mutborrow)
setfield { field=y, obj=%pm, val=const.i64 22 }
setfield { obj=%pm, val=const.i64 22, field=y }  ; also valid
```

### 11.3 Enums and Match

```mp
module test.enum_match
exports { @classify, @safe_divide }
imports { }
digest "0000000000000000"

heap enum TShape {
  variant Circle { field radius: f64 }
  variant Rect { field w: f64, field h: f64 }
}

; Tag-based dispatch (integer tag comparison)
fn @classify(%s: borrow TShape) -> i32 meta { } {
bb0:
  %tag: i32 = enum.tag { v=%s }
  %is_circle: bool = icmp.eq { lhs=%tag, rhs=const.i32 0 }
  cbr %is_circle bb1 bb2

bb1:
  ret const.i32 1

bb2:
  ret const.i32 2
}

; TResult pattern
fn @safe_divide(%a: i32, %b: i32) -> TResult<i32, Str> meta { } {
bb0:
  %is_zero: bool = icmp.eq { lhs=%b, rhs=const.i32 0 }
  cbr %is_zero bb_err bb_ok

bb_err:
  %msg: Str = const.Str "division by zero"
  %err: TResult<i32, Str> = enum.new<Err> { e=%msg }
  ret %err

bb_ok:
  %result: i32 = i.sdiv { lhs=%a, rhs=%b }
  %ok: TResult<i32, Str> = enum.new<Ok> { v=%result }
  ret %ok
}
```

Enum opcodes:

| Opcode | Description |
|---|---|
| `enum.new<Variant> { field=val, ... }` | Construct an enum variant |
| `enum.tag { v= }` | Get integer discriminant of variant |
| `enum.payload<Variant> { v= }` | Extract variant payload |
| `enum.is<Variant> { v= }` | Bool test for variant |

### 11.4 Collections (Array and Map)

```mp
module demo.collections
exports { @test_array, @test_map }
imports { }
digest "0000000000000000"

fn @test_array() -> i64 meta { } {
bb0:
  %arr: Array<i32> = arr.new<i32> { cap=const.i64 10 }
  %arr_m: mutborrow Array<i32> = borrow.mut { v=%arr }
  arr.push { arr=%arr_m, val=const.i32 42 }
  arr.push { arr=%arr_m, val=const.i32 99 }
  br bb1                ; drop mutborrow before reading

bb1:
  %arr_b: borrow Array<i32> = borrow.shared { v=%arr }
  %len: i64 = arr.len { arr=%arr_b }
  ret %len
}

fn @test_map() -> bool meta { } {
bb0:
  %m: Map<Str, i32> = map.new<Str, i32> { }
  %key: Str = const.Str "hello"
  %m_mut: mutborrow Map<Str, i32> = borrow.mut { v=%m }
  map.set { key=%key, map=%m_mut, val=const.i32 42 }
  br bb1                ; drop mutborrow before reading

bb1:
  %key2: Str = const.Str "hello"
  %m_b: borrow Map<Str, i32> = borrow.shared { v=%m }
  %has: bool = map.contains_key { key=%key2, map=%m_b }
  ret %has
}
```

**Critical rule:** collection mutation ops (`arr.push`, `arr.sort`, `map.set`)
require a `mutborrow` receiver. Read ops (`arr.len`, `arr.contains`,
`map.contains_key`, `map.len`) require a `borrow` receiver.

Array opcodes:

| Opcode | Receiver | Description |
|---|---|---|
| `arr.new<T> { cap= }` | — | Create array with capacity hint |
| `arr.push { arr=mutborrow, val= }` | mutborrow | Append element |
| `arr.set { arr=mutborrow, idx=, val= }` | mutborrow | Set element at index |
| `arr.sort { arr=mutborrow }` | mutborrow | Sort in place (needs `ord` on T) |
| `arr.len { arr=borrow }` | borrow | Get length |
| `arr.get { arr=borrow, idx= }` | borrow | Get element |
| `arr.contains { arr=borrow, val= }` | borrow | Test membership (needs `eq` on T) |
| `arr.pop { arr=mutborrow }` | mutborrow | Remove last element |

Map opcodes:

| Opcode | Receiver | Description |
|---|---|---|
| `map.new<K,V> { }` | — | Create map (K needs `hash` + `eq`) |
| `map.set { map=mutborrow, key=, val= }` | mutborrow | Insert/update |
| `map.delete_void { map=mutborrow, key= }` | mutborrow | Delete key |
| `map.get { map=borrow, key= }` | borrow | Get value (returns TOption) |
| `map.contains_key { map=borrow, key= }` | borrow | Test key presence |
| `map.len { map=borrow }` | borrow | Get entry count |

### 11.5 Strings

```mp
module demo.strings
exports { @string_ops }
imports { }
digest "0000000000000000"

fn @string_ops() -> i64 meta { } {
bb0:
  ; String constant
  %s: Str = const.Str "abcdef"

  ; Borrow to call read ops
  %sb: borrow Str = borrow.shared { v=%s }
  %len: i64 = str.len { s=%sb }

  ; Parse a string to integer
  %nstr: Str = const.Str "123"
  %nb: borrow Str = borrow.shared { v=%nstr }
  %n: i64 = str.parse_i64 { s=%nb }

  %out: i64 = i.add { lhs=%len, rhs=%n }
  ret %out
}
```

Compatibility note:
- `str.parse_*` is currently exposed in source as value-producing ops for compatibility.
- Under the hood, runtime ABI calls are now fallible (`mp_rt_str_try_parse_*` / `mp_rt_json_try_*`) with explicit status checks in codegen.
- Legacy runtime wrappers (`mp_rt_str_parse_*`, `mp_rt_json_encode`, `mp_rt_json_decode`) are deprecated temporary shims and are planned for removal after migration.

String opcodes:

| Opcode | Description |
|---|---|
| `const.Str "text"` | String literal constant |
| `str.len { s=borrow }` | Length in bytes |
| `str.concat { lhs=, rhs= }` | Concatenate two strings |
| `str.eq { lhs=, rhs= }` | Equality test |
| `str.slice { s=, start=, end= }` | Substring slice |
| `str.parse_i64 { s=borrow }` | Parse to i64 |
| `str.parse_f64 { s=borrow }` | Parse to f64 |
| `str.parse_bool { s=borrow }` | Parse to bool |
| `str.builder.new { }` | Create TStrBuilder |
| `str.builder.append_str { b=, s= }` | Append string |
| `str.builder.build { b= }` | Finalize to Str |

### 11.6 Ownership: Share and Clone

```mp
module test.ownership
exports { @create_and_share }
imports { }
digest "0000000000000000"

heap struct TPerson {
  field name: Str
  field age: i32
}

fn @create_and_share() -> shared TPerson meta { } {
bb0:
  %name: Str = const.Str "Alice"
  %age: i32 = const.i32 30
  ; Field order in new {} does not matter
  %person: TPerson = new TPerson { age=%age, name=%name }
  %shared_p: shared TPerson = share { v=%person }
  %clone_p: shared TPerson = clone.shared { v=%shared_p }
  ret %shared_p
}
```

Ownership opcodes:

| Opcode | Description |
|---|---|
| `new T { field=val, ... }` | Allocate heap object |
| `share { v= }` | Move unique value into ARC shared handle |
| `clone.shared { v=shared }` | Clone a shared handle (refcount++) |
| `weak.downgrade { v=shared }` | Create weak reference |
| `weak.upgrade { v=weak }` | Upgrade weak to `TOption<shared T>` |
| `borrow.shared { v= }` | Create immutable borrow (block-scoped) |
| `borrow.mut { v= }` | Create mutable borrow (block-scoped) |

### 11.7 Async Functions

```mp
module test.async_example
exports { @fetch_data }
imports { }
digest "0000000000000000"

fn @get_connection() -> Str meta { } {
bb0:
  %s: Str = const.Str "conn:1"
  ret %s
}

fn @process(%conn: Str, %id: u64) -> Str meta { } {
bb0:
  ret %conn
}

; Async function: use suspend.call for awaited calls
async fn @fetch_data(%id: u64) -> Str meta { uses { @get_connection, @process } } {
bb0:
  %conn: Str = suspend.call @get_connection { }
  %result: Str = call @process { conn=%conn, id=%id }
  ret %result
}
```

**`suspend.call` vs `call`:**
- `suspend.call @fn { args }` — async await point; suspends until the called fn resolves
- `call @fn { args }` — synchronous call inside async fn

### 11.8 TCallable (First-Class Callables)

```mp
module demo.callable
exports { @main, @multiply_by }
imports { }
digest "0000000000000000"

; Declare the signature type
sig TMulSig(i32) -> i32

; The function to be captured
fn @multiply_by(%x: i32, %factor: i32) -> i32 meta { } {
bb0:
  %y: i32 = i.mul { lhs=%x, rhs=%factor }
  ret %y
}

fn @main() -> i32 meta { uses { @multiply_by } } {
bb0:
  %factor: i32 = const.i32 3

  ; Create a callable that captures %factor
  %mul_by_3: TCallable<TMulSig> = callable.capture @multiply_by { factor=%factor }

  ; Invoke indirectly — args= list matches the sig parameters
  %result: i32 = call.indirect %mul_by_3 { args=[const.i32 7] }

  ret %result
}
```

TCallable pattern steps:
1. Declare `sig TName(ArgTypes...) -> RetType`
2. Write `fn @impl(%arg: T, %captured: CaptureType) -> RetType`
3. Create: `callable.capture @impl { captured_name=value }`
4. Invoke: `call.indirect %callable { args=[...] }`

---

## 12) Test Fixtures Reference

All fixtures live in `tests/fixtures/`. Run them with:

```bash
cargo test
cargo run -p magpie_cli -- --entry tests/fixtures/<name>.mp --output json build
```

| Fixture | Module | Exports | What It Tests |
|---|---|---|---|
| `hello.mp` | `hello.main` | `@main` | Basic program, `Str` constant, `call_void`, external import (`std.io`) |
| `arithmetic.mp` | `test.arithmetic` | `@add`, `@checked_add` | `i.add`, `i.add.checked`, `TOption<i32>` return |
| `enum_match.mp` | `test.enum_match` | `@classify` | `heap enum`, `enum.tag`, `cbr` dispatch, `borrow` receiver |
| `collections.mp` | `test.collections` | `@test_array`, `@test_map` | `arr.new/push/len`, `map.new/set/contains_key`, mutborrow/borrow pattern |
| `ownership.mp` | `test.ownership` | `@create_and_share` | `heap struct`, `new`, `share`, `clone.shared`, field args any order |
| `async_fn.mp` | `test.async_example` | `@fetch_data` | `async fn`, `suspend.call` await pattern |
| `try_error.mp` | `test.try_error` | `@safe_divide` | `TResult<Ok,Err>`, `enum.new<Ok>`, `enum.new<Err>`, error branching |
| `feature_harness.mp` | `test.feature_harness` | `@main` + 5 helpers | Comprehensive: structs, collections, strings, sharing, checked math |

---

## 13) Common Pitfalls and Fixes

### Pitfall 1: Wrong constant type for return

```mp
; WRONG: returns i32 but uses i64 constant
fn @main() -> i32 {
bb0:
  ret const.i64 0    ; MPT* type mismatch
}

; CORRECT: match const type to return type
fn @main() -> i32 {
bb0:
  ret const.i32 0
}
```

### Pitfall 2: Comment syntax errors

```mp
; CORRECT: line comment uses single semicolon
; This is a line comment

;; CORRECT: doc comment uses double semicolon
;; This is a doc comment

;;; WRONG: triple semicolon causes parse errors
;;; Do not use this form
```

### Pitfall 3: Borrow crossing block boundary

```mp
; WRONG: borrow created in bb0 used in bb1
fn @bad() -> i64 {
bb0:
  %arr: Array<i64> = arr.new<i64> { cap=const.i64 4 }
  %arr_m: mutborrow Array<i64> = borrow.mut { v=%arr }
  arr.push { arr=%arr_m, val=const.i64 1 }
  br bb1

bb1:
  arr.push { arr=%arr_m, val=const.i64 2 }   ; MPO* - borrow from bb0
  ret const.i64 0
}

; CORRECT: re-borrow in each block that needs it
fn @good() -> i64 {
bb0:
  %arr: Array<i64> = arr.new<i64> { cap=const.i64 4 }
  %arr_m: mutborrow Array<i64> = borrow.mut { v=%arr }
  arr.push { arr=%arr_m, val=const.i64 1 }
  br bb1

bb1:
  %arr_m2: mutborrow Array<i64> = borrow.mut { v=%arr }
  arr.push { arr=%arr_m2, val=const.i64 2 }
  %arr_b: borrow Array<i64> = borrow.shared { v=%arr }
  %len: i64 = arr.len { arr=%arr_b }
  ret %len
}
```

### Pitfall 4: `getfield` without borrow receiver

```mp
; WRONG: using raw value as getfield receiver
fn @bad(%p: TPoint) -> i64 {
bb0:
  %x: i64 = getfield { field=x, obj=%p }   ; MPT* - needs borrow
  ret %x
}

; CORRECT: borrow first
fn @good(%p: TPoint) -> i64 {
bb0:
  %pb: borrow TPoint = borrow.shared { v=%p }
  %x: i64 = getfield { field=x, obj=%pb }
  ret %x
}
```

### Pitfall 5: `arr.push` with borrow instead of mutborrow

```mp
; WRONG: push requires mutborrow
fn @bad() -> i64 {
bb0:
  %arr: Array<i32> = arr.new<i32> { cap=const.i64 4 }
  %arr_b: borrow Array<i32> = borrow.shared { v=%arr }
  arr.push { arr=%arr_b, val=const.i32 1 }   ; MPT* - needs mutborrow
  ret const.i64 0
}

; CORRECT: use borrow.mut for mutation
fn @good() -> i64 {
bb0:
  %arr: Array<i32> = arr.new<i32> { cap=const.i64 4 }
  %arr_m: mutborrow Array<i32> = borrow.mut { v=%arr }
  arr.push { arr=%arr_m, val=const.i32 1 }
  br bb1

bb1:
  %arr_b: borrow Array<i32> = borrow.shared { v=%arr }
  %len: i64 = arr.len { arr=%arr_b }
  ret %len
}
```

### Pitfall 6: Wrong header order

```mp
; WRONG: digest before imports
module demo.main
exports { @main }
digest "0000000000000000"   ; MPP* - must come after imports
imports { }

; CORRECT: strict order
module demo.main
exports { @main }
imports { }
digest "0000000000000000"
```

### Pitfall 7: Missing block terminator

```mp
; WRONG: bb0 has no terminator
fn @bad() -> i32 {
bb0:
  %x: i32 = const.i32 1
  ; missing ret/br/cbr/switch/unreachable -- MPS* error
}

; CORRECT
fn @good() -> i32 {
bb0:
  %x: i32 = const.i32 1
  ret %x
}
```

### Pitfall 8: Using `Str` map key — no impl needed

```mp
; WRONG: attempting to add redundant impl for Str
impl hash for Str = @str_hash   ; MPT* - Str already has built-in hash

; CORRECT: just use Str directly as a map key
%m: Map<Str, i32> = map.new<Str, i32> { }
```

### Pitfall 9: `--entry` path is relative to CWD

```bash
# If you are in /project/magpie and your file is at demo/src/main.mp:

# CORRECT
cargo run -p magpie_cli -- --entry demo/src/main.mp build

# WRONG (if you cd into demo first but then use project-relative path)
cd demo
cargo run -p magpie_cli -- --entry src/main.mp build  # must be from demo/ CWD
```

### Pitfall 10: `setfield` uses `val=` not `value=`

```mp
; WRONG
setfield { field=x, obj=%pm, value=const.i64 5 }   ; MPP* parse error

; CORRECT
setfield { field=x, obj=%pm, val=const.i64 5 }
```

---

## 14) Diagnostic Code Families

| Prefix | Family | Examples |
|---|---|---|
| `MPP*` | Parse / IO / Artifact | Bad header order, syntax errors, malformed opcodes, missing terminators |
| `MPS*` | Resolve / SSA / Structural | Unknown symbols, duplicate definitions, SSA use-before-def, missing block |
| `MPT*` | Type / Trait / v0.1 restrictions | Type mismatch, missing trait impl, invalid receiver kind, v0.1 restricted form |
| `MPO*` | Ownership / Borrow / Move | Borrow crossing blocks, borrow in phi, returning borrow, use-after-move |
| `MPF*` | FFI | Invalid extern binding, unsupported ABI form |
| `MPG*` | GPU | Invalid GPU op context, buffer type mismatch |
| `MPL*` | Lint / Link / LLM budget | Lint warnings, linker errors, token budget exceeded |
| `MPW*` | Web | Web framework integration errors |
| `MPK*` | Package / Dependency | Manifest errors, missing dependency |
| `MPM*` | Memory / Index | Memory index query failures |
| `MPHIR*` | HIR invariants | Internal HIR structural violations |

### Parse/JSON migration diagnostics

| Code | Meaning |
|---|---|
| `MPT2033` | Parse/JSON result type shape is invalid (expected legacy or `TResult<ok, err>`) |
| `MPT2034` | Parse/JSON input is not `Str`/`borrow Str` (or input type is unknown) |
| `MPT2035` | `json.encode<T>` value type does not match `T` (or value type is unknown) |

### Fast diagnostic loop

```bash
# Step 1: Build with JSON output + debug artifacts
cargo run -p magpie_cli -- \
  --entry src/main.mp \
  --emit mpir,llvm-ir,mpdbg \
  --output json \
  build

# Step 2: Explain a specific code
cargo run -p magpie_cli -- --output json explain MPT2014

# Step 3: Fix the smallest error first and rebuild
# Repeat until clean
```

---

## 15) LLM/Agent Mode

Enable with `--llm` for structured, token-budget-aware output:

```bash
cargo run -p magpie_cli -- \
  --entry src/main.mp \
  --llm \
  --llm-token-budget 12000 \
  --llm-budget-policy balanced \
  --output json \
  build
```

### Budget policies

| Policy | Description |
|---|---|
| `balanced` | Even split between diagnostics and code slices |
| `diagnostics_first` | Prioritize error messages |
| `slices_first` | Prioritize source context |
| `minimal` | Minimal output, errors only |

### LLM-mode flags

```bash
--llm                              # Enable LLM mode
--llm-token-budget 8000            # Hard token ceiling
--llm-budget-policy diagnostics_first
--llm-tokenizer approx:utf8_4chars # Tokenizer heuristic
--no-auto-fmt                      # Skip pre-build format pass
```

### Environment variable overrides

```bash
export MAGPIE_LLM=true
export MAGPIE_LLM_TOKEN_BUDGET=10000
```

### Configuration in Magpie.toml

```toml
[llm]
mode_default = true
token_budget = 8000
tokenizer = "approx:utf8_4chars"
budget_policy = "balanced"
```

---

## 16) Where to Go Next

| Topic | Location |
|---|---|
| Language grammar and lexical rules | `DOCUMENTATION.md` §4–5 |
| All declarations (`fn`, `struct`, `enum`, `extern`, `sig`, `impl`, `global`) | `DOCUMENTATION.md` §6 |
| Full type system | `DOCUMENTATION.md` §7 |
| Complete opcode reference | `DOCUMENTATION.md` §9 |
| All safety and semantic rules | `DOCUMENTATION.md` §10 |
| TCallable deep dive | `DOCUMENTATION.md` §11 |
| ARC + ownership design rationale | `DOCUMENTATION.md` §12 |
| Compiler architecture details | `DOCUMENTATION.md` §13 |
| All emit kinds and artifacts | `DOCUMENTATION.md` §15 |
| Full diagnostics model | `DOCUMENTATION.md` §16 |
| All CLI flags (complete table) | `DOCUMENTATION.md` §17 |
| Configuration resolution order | `DOCUMENTATION.md` §18 |
| Build/test/run playbooks | `DOCUMENTATION.md` §19 |
| Extended examples | `DOCUMENTATION.md` §20 |
| Formal grammar appendix | `DOCUMENTATION.md` Appendix A |
| Full opcode appendix | `DOCUMENTATION.md` Appendix B–D |
| Evidence matrix | `DOCUMENTATION.md` Appendix E–F |
| Skill guide (LLM agent usage) | `SKILL.md` |
| All test fixtures | `tests/fixtures/` |
