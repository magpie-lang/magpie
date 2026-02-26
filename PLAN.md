# PLAN.md — Magpie v0.1 Implementation Plan (Agent-Optimized)

This plan is designed for coding agents (Codex-style) to implement Magpie v0.1 from **SPEC.md** in **phases**, with clear deliverables, file boundaries, and “definition of done” gates. It assumes implementation in **Rust** with an **LLVM frontend** that emits LLVM IR/bitcode and links to a final binary.

---

## 0) Global rules (apply to every phase)

### 0.1 Non-negotiable properties

* **Determinism:** same inputs + same toolchain config MUST produce identical:
  * CSNF source
  * digests
  * SIDs
  * TypeId tables
  * MPIR text
  * JSON outputs (canonical JSON)
  * final binaries in `--profile release` (within platform/linker limits)
* **LLM-first outputs:** every CLI command supports token budgets and produces structured JSON in `--llm` mode.
* **No “spec guessing”:** if the spec is ambiguous, open a SPEC issue and add a local “assumption note” in `docs/assumptions.md` before coding.

### 0.2 Repository skeleton (create immediately)

Rust workspace layout (per §36.1):

```
magpie/
  Cargo.toml                  # workspace
  crates/
    magpie_cli/
    magpie_driver/
    magpie_lex/
    magpie_parse/
    magpie_csnf/
    magpie_ast/
    magpie_hir/
    magpie_sema/
    magpie_types/
    magpie_mono/
    magpie_own/
    magpie_mpir/
    magpie_arc/
    magpie_codegen_llvm/
    magpie_codegen_wasm/
    magpie_jit/
    magpie_diag/
    magpie_pkg/
    magpie_memory/
    magpie_ctx/
    magpie_web/
    magpie_gpu/
    magpie_rt/
  std/
    std.core/
    std.io/
    std.hash/
    std.sync/
    std.async/
  tests/
    fixtures/
    e2e/
  docs/
    assumptions.md
    dev/
      coding_style.md
      determinism.md
```

### 0.3 “Golden” test strategy (start on day 1)

* Add `tests/fixtures/*.mp` (surface) and expected `*.mpir` (and sometimes `*.json`) outputs.
* Use snapshot tests for:
  * `magpie fmt` output (CSNF)
  * `--emit mpir` text
  * JSON diagnostics (canonical JSON)
* Tests MUST be stable under token budgeting (i.e., verify truncation ladders deterministically).

Recommended testing tools:
* `insta` for snapshots
* `pretty_assertions` for diffs
* a tiny `magpie_test_harness` helper to run the CLI in-process

### 0.4 Cross-cutting “shared types” crate conventions

Create a few “spine” modules early; many phases depend on them:

* `magpie_span` (either a small crate or inside `magpie_ast`): `FileId`, `Span`, `Spanned<T>`, `SourceMap`.
* `magpie_hash`: BLAKE3 digest helpers, base32 crockford SID helpers.
* `magpie_cjson`: canonical JSON encoder.

If you prefer not to create extra crates, place these in `magpie_diag` or `magpie_ast` and re-export.

---

## Phase 1 — Buildable skeleton + CLI plumbing (M0)

### Goal
`magpie --help` works; `magpie new/build/fmt/test/mpir verify` exist as subcommands (even if stubbed). Structured output framework exists.

### Work items
1. **Workspace + crates**
   * Create all crates listed in §36.1 (empty lib bins OK).
   * Establish common feature flags: `default`, `llvm`, `jit`, `gpu`, `web`.
2. **CLI surface (`magpie_cli`)**
   * Implement argument parsing for all global flags (§5.1) and commands (§5.2).
   * Create a typed `CliConfig` struct used by `magpie_driver`.
3. **Output modes**
   * Implement `--output text|json|jsonl` routing.
   * Implement `--llm` behavior defaults (JSON by default, budget enforcement enabled).
4. **Diagnostics backbone (`magpie_diag`)**
   * Implement the JSON root envelope (§26.1) and per-diagnostic schema (§26.2).
   * Implement token-budget enforcement tiers (§3.3) in one place.

### Definition of done
* `magpie new hello` creates the required files.
* `magpie build` runs the pipeline skeleton and returns a valid JSON envelope even if compilation is not implemented.
* `magpie fmt` rewrites input file to itself (no-op) but writes a correct digest line.

---

## Phase 2 — SourceMap + lexer + parser + AST (M1)

### Goal
Parse `.mp` files into a recoverable AST with precise spans. Validate the mandatory header (§6.3). No semantics yet.

### Work items
1. **SourceMap / spans** (per §36.2)
   * `SourceMap::add_file(path, bytes) -> FileId`
   * byte-offset spans only (UTF-8 bytes)
2. **Lexer (`magpie_lex`)**
   * Token types: identifiers, keywords, punctuation, literals, comments.
   * Preserve doc comments (`;;;`) and attach later.
   * Canonical string literal unescaping.
3. **Parser (`magpie_parse`)**
   * Handwritten recursive descent (§36.3).
   * Error recovery via sync tokens at decl boundaries and block boundaries.
   * Parse:
     * Header: `module`, `exports`, `imports`, `digest`.
     * Decls: `fn`, `async fn`, `unsafe fn`, `extern`, `global`, `heap/value struct`, `heap/value enum`, `sig`, `impl`.
     * Function bodies: basic blocks, SSA locals with explicit types, instructions + terminators.
   * Store everything in `magpie_ast` with spans.
4. **AST validation (shallow)**
   * Reject missing header / malformed module path.
   * Reject unknown keywords/opcodes with parse diagnostics `MPP*`.

### Definition of done
* `magpie parse --emit ast` (internal/dev flag) can print a debug AST.
* At least 10 fixtures parse successfully, including the Hello World example.

---

## Phase 3 — CSNF formatter + digest authority (M2)

### Goal
`magpie fmt` produces **canonical source** (CSNF) deterministically and updates `digest` using `BLAKE3(canonical_source_without_digest_line)` (§6.4, §2.1).

### Work items
1. **CSNF rules implementation (`magpie_csnf`)**
   * Canonical whitespace and indentation.
   * Canonical ordering for:
     * header lists (`exports`, `imports` groups/items)
     * key-value argument blocks (`{ key=value, ... }` sorted by key)
     * `enum.new` fields sorted by field name
   * Canonical block labels: rename to `bb0..bbN` and rewrite branch targets (§5.2.5).
   * Canonical numeric printing and const forms (e.g., `const.i32 0`).
2. **Digest update**
   * Strip the `digest "..."` line before hashing.
   * Reinsert correct digest in canonical form.

### Definition of done
* Running `magpie fmt` twice is a no-op.
* `digest` changes iff canonical content changes.
* Snapshot tests for CSNF formatting across representative syntax.

---

## Phase 4 — Symbol tables + module system + SIDs (M3)

### Goal
Resolve imports/exports, produce FQNs, and assign **Stable IDs (SIDs)** deterministically (§18). Generate a minimal symbol graph.

### Work items
1. **Module loading** (`magpie_driver` + `magpie_sema`)
   * Map `module pkg.sub.mod` to `src/sub/mod.mp` (§6.2).
   * Enforce “one module per file”.
   * Parse workspace starting from `[build].entry` and imports.
2. **Name resolution / symbol table**
   * Build per-module symbol tables for:
     * functions, types, globals, sigs
   * Enforce no overloads within namespaces (§6.6).
   * Resolve imported names to FQNs.
3. **SIDs**
   * Implement SID generator (base32-crockford(blake3(input))[0..10]) (§18.4).
   * Emit module/function/type/global SID maps.
4. **Minimal graph outputs**
   * `magpie graph symbols --json` (even if small) and `--llm` truncation.

### Definition of done
* A multi-module project builds a complete symbol table.
* Duplicate symbol errors produce stable diagnostics.

---

## Phase 5 — Type interning + canonical type strings + TypeId assignment (M4)

### Goal
Implement the v0.1 type system core (explicit types, ownership modifiers, builtins, generics skeleton) and deterministic `TypeId` assignment.

### Work items
1. **Type model (`magpie_types`)**
   * Implement `TypeKind` and related enums per §16.3.
   * Enforce v0.1 restrictions:
     * reject surface aggregates (`vec/arr/tuple`) with `MPT1021`.
     * reject value enums (`MPT1020`).
     * reject value structs containing heap handles (`MPT1005`).
2. **Canonical type strings** (§18.3)
   * Implement `TypeStr` printer (ownership modifiers + spacing rules).
3. **Deterministic TypeId table**
   * Reserve fixed IDs for primitives/builtins (§20.1.4).
   * For non-fixed types:
     * compute canonical type key (FQN for user types; canonical TypeStr for instantiations and qualified types)
     * assign IDs >= 1000 in lexicographic order **across the build graph** (§15.5).
   * Persist the final table into MPIR type section.
4. **Layouts (minimum viable)**
   * Compute payload size/align for heap structs/enums.
   * Provide enough layout info for codegen and runtime registration.

### Definition of done
* `magpie build --emit mpir` can emit a `types {}` table for a simple project.

---

## Phase 6 — HIR construction + SSA verification (M5)

### Goal
Lower AST into HIR with explicit SSA locals and CFG blocks; verify SSA invariants.

### Work items
1. **HIR data structures** (`magpie_hir`)
   * Implement the IDs, values, consts, op enums, terminators, blocks.
   * NOTE: the spec text currently contains a duplicated `MapGetRef` in §16.4; implement it **once**.
2. **AST → HIR lowering**
   * Each SSA assignment produces a `LocalId`.
   * Blocks preserve order and terminators.
   * Parse `try` as surface sugar but lower to the desugared form in MPIR later; in HIR you may keep `Try` op or desugar early—pick one and make it consistent.
3. **SSA verifier (`magpie_mpir` or `magpie_hir`)**
   * Dominance/use-before-def per §15.8 and §16.6.
   * Single terminator rule.

### Definition of done
* `magpie build` fails fast on malformed SSA with `MPS*` errors.

---

## Phase 7 — Type checking + trait binding resolution (M6)

### Goal
Assign types to all HIR ops (using explicit annotations), validate op typing rules, and resolve required trait impls (`hash/eq/ord/send/sync`).

### Work items
1. **Typecheck engine (`magpie_types` + `magpie_sema`)**
   * Validate every opcode signature (arg types, result types).
   * Validate calls:
     * function existence
     * arity
     * generics targs validity
   * Validate struct fields and enum variants.
2. **Trait impl binding (`magpie_sema`)**
   * Parse and store `impl trait for Type = @function`.
   * Enforce orphan rule (§9.5).
   * Enforce trait signature matches required function form (§9.2).
3. **Required traits for intrinsics**
   * Enforce §10.4.1:
     * `arr.contains` requires `eq`
     * `arr.sort` requires `ord`
     * `map.new` requires `hash` and `eq`
   * Emit `MPT1023` on missing impl.

### Definition of done
* A project using `Map<K,V>` refuses to compile unless `hash/eq` exist for K.

---

## Phase 8 — Ownership + borrow checking (M7)

### Goal
Implement move tracking and block-local borrow checking per §10 (moved-set analysis + linear scan borrow rules), producing rich traces for LLM diagnostics.

### Work items
1. **Moved-set dataflow (`magpie_own`)**
   * Compute `Moved_in/out` sets over CFG for move-only locals (`O`).
   * Treat phi as consuming incoming values on edges (§10.6).
2. **Borrow checker (block-local)**
   * Enforce:
     * no borrows across block boundaries (`MPO0101`)
     * no borrows in phi (`MPO0102`)
     * no borrow storage (`MPO0003`)
     * move-while-borrowed (`MPO0011`)
   * Track shared counts + mut-active within a block.
3. **Ownership rules for storage projections** (§10.4.1)
   * `getfield/arr.get/map.get_ref` return value vs borrow depending on field/elem type.
   * `map.get` restrictions (Dupable-only) + emit `MPO0103`.
   * Duplication-producing intrinsics require `Dupable` (`MPT1022`).
4. **Effectful diagnostics**
   * For each error, produce a `why.trace` that references:
     * value SID/local name
     * first move site
     * conflicting use site
     * active borrow sites

### Definition of done
* Add fixtures covering:
  * use-after-move
  * borrow crosses block
  * borrow-in-phi
  * map.get on non-dupable V
  * shared mutation errors

---

## Phase 9 — MPIR builder + verifier + printer (M8)

### Goal
Produce canonical MPIR (`--emit mpir`) that matches §15–§17, and implement `magpie mpir verify`.

### Work items
1. **MPIR structures (`magpie_mpir`)**
   * Implement `MpirModule`, `MpirFn`, blocks, instrs, void ops, terminators.
   * Ensure SIDs and TypeIds are fully resolved.
2. **MPIR text format writer**
   * Deterministic ordering and printing:
     * section order
     * type table
     * globals
     * functions sorted by FQN
3. **MPIR verifier** (§15.8)
   * SSA correctness
   * type refs exist
   * phi restrictions
   * terminator correctness

### Definition of done
* `magpie build --emit mpir` produces stable `.mpir` suitable for golden tests.

---

## Phase 10 — ARC insertion + drop elaboration + ARC opts (M9)

### Goal
Insert explicit ARC ops and drop logic into MPIR (then optimize), according to §11.

### Work items
1. **Drop elaboration**
   * Generate drop routines for heap structs/enums that release contained heap handles (including nested in `TOption/TResult`).
   * Ensure overwrite semantics: drop old value before move-in (§10.4.1, §11.4).
2. **ARC insertion** (`magpie_arc`)
   * Insert retains/releases for:
     * `clone.shared`, `clone.weak`, `weak.downgrade`
     * cloning weak on `getfield` of weak handle
     * end-of-scope releases for owned handles
   * Respect unique vs shared atomic rules (runtime handles atomicity).
3. **ARC optimization passes**
   * Pair elimination in straight-line code
   * CFG-aware sink/hoist basics

### Definition of done
* MPIR after ARC pass includes `arc.*` ops only after stage 8 in pipeline.
* ARC verifier asserts no `arc.*` earlier.

---

## Phase 11 — Runtime library (`magpie_rt`) (M10)

### Goal
A working runtime implementing §20 ABI for:
* allocation + refcounting
* type registry
* Str/Array/Map primitives
* panic
* callback ABI plumbing for `hash/eq/ord`

### Work items
1. **C ABI surface**
   * Provide `magpie_rt/include/magpie_rt.h` mirroring §20.
   * Implement `#[no_mangle] extern "C"` functions in Rust.
2. **Allocator + headers**
   * Implement `MpRtHeader` layout and payload alignment.
   * Implement strong/weak semantics (§20.3).
3. **Type registry**
   * `mp_rt_register_types`, `mp_rt_type_info`.
4. **Strings**
   * `mp_rt_str_from_utf8`, `mp_rt_str_bytes`, concat/slice/len/eq.
5. **Array**
   * contiguous buffer + element size
   * `get` panics on OOB
   * `slice` clones elements (requires Dupable at compile time)
6. **Map**
   * hash map storing key/value bytes
   * `mp_rt_map_new` takes `hash_fn/eq_fn`
   * `mp_rt_map_take` vs `mp_rt_map_delete` semantics

### Definition of done
* Runtime unit tests validate refcounts, weak upgrade behavior, array/map correctness.

---

## Phase 12 — LLVM codegen + link to final binary (M11)

### Goal
Compile MPIR → LLVM IR → object/exe via LLVM + linker. `magpie run` executes Hello World.

### Work items
1. **LLVM binding choice**
   * Use `llvm-sys` (spec recommends) with a pinned LLVM version.
   * Centralize all unsafe LLVM calls in `magpie_codegen_llvm`.
2. **LLVM module construction**
   * Type mapping (§21.1): value types → ints/floats, heap handles → `ptr`.
   * Function symbols: mangling rules (§19).
   * Emit per-module type registration init: `mp$0$INIT_TYPES$<M_sid>`.
3. **Lowering of core ops**
   * `const.*`, arithmetic, compares, control flow, phi.
   * Calls (direct/indirect) and void calls.
   * Heap ops (`new/getfield/setfield`) as runtime calls + GEP loads/stores.
   * `TOption/TResult` lowering:
     * follow representation rules in §8.1.4
     * implement `enum.tag/is/payload/new` for builtins and heap enums.
4. **Checked arithmetic traps** (§21.4)
   * Use LLVM overflow intrinsics; panic on overflow.
5. **Collections & strings** (§21.5)
   * Lower all `arr.*`, `map.*`, `str.*` to runtime ABI calls.
   * Ensure `map.get_ref` panics on missing key.
6. **Trait callback wrappers** (§9.5)
   * Generate `mp_cb_hash_T`, `mp_cb_eq_T`, `mp_cb_cmp_T` wrappers and export them.
7. **Link step**
   * Object emission + invoke `lld` if available; otherwise system linker (§36.4).
   * Always link `magpie_rt` statically.

### Definition of done
* `magpie build --emit llvm-ir,object,exe` works for Hello World.
* `magpie run` prints output and returns correct exit code.

---

## Phase 13 — End-to-end compiler pipeline + caching skeleton (M12)

### Goal
Implement pipeline stages (§22) with correct pass ordering and “skip dependent passes on error”. Add minimal incremental cache keys.

### Work items
1. **Driver graph**
   * Stage ordering exactly as §22.1.
   * Accumulate timing and diagnostics into JSON envelope.
2. **Artifacts**
   * Implement `--emit` handling for at least:
     * `mpir`, `llvm-ir`, `llvm-bc`, `object`, `exe`
   * Deterministic file naming.
3. **Cache**
   * Cache keys include compiler version, toolchain hash, module digests, target triple (§22.3).
   * Cache parsed AST + resolved HIR first; MPIR/LLVM later.

### Definition of done
* Rebuilding without changes hits cache and returns identical artifacts.

---

## Phase 14 — Diagnostics quality + suggested patches + `magpie explain` (M13)

### Goal
High-quality LLM diagnostics (structured JSON), with optional unified-diff patches (§27) and budget-aware truncation.

### Work items
1. **Diagnostics rendering**
   * Populate: `primary_span`, `secondary_spans`, `explanation_md`, `why.trace`.
   * Implement `--max-errors` collection.
2. **Suggested fixes**
   * Implement at least 5 core fixers:
     * add missing import
     * replace `map.get` with `map.contains_key + map.get_ref` pattern
     * insert `share` / `clone.shared`
     * split borrows so they don’t cross blocks
     * add missing trait impl stub
   * Wrap in patch envelope with `applies_to`/`produces` digests.
3. **`magpie explain <CODE>`**
   * Map error codes → deterministic remediation templates.

### Definition of done
* A known ownership error produces a patch that applies cleanly after `magpie fmt`.

---

## Phase 15 — Graph emitters + `.mpd` digests (M14)

### Goal
Implement the progressive-disclosure outputs (§2.2): symbol graph, dep graph, ownership graph, cfg graph; plus module digests (`.mpd`).

### Work items
1. **`.mpd` generator**
   * Public API summary: exports, signatures, docs, SIDs, digests.
2. **Graphs**
   * `magpie graph symbols` (nodes=defs, edges=refs)
   * `magpie graph deps` (package/module deps)
   * `magpie graph ownership` (ownership proof/trace summary)
   * `magpie graph cfg` (per-function blocks/edges)
3. **Token-budgeted graph reduction**
   * Implement the Tiered dropping rules (§3.3) for graphs.

### Definition of done
* In `--llm` mode, a failed build includes minimal graphs without exceeding budget.

---

## Phase 16 — Interpreter/JIT + REPL (Tier 2) (M15)

### Goal
Provide `magpie repl` with `:type`, `:ir`, `:llvm`, and `:diag last` (§5.2.4, §23).

### Work items
* ORC JIT wrapper in `magpie_jit`.
* Incremental compilation session state.
* REPL command parser.

---

## Phase 17 — Package manager + lockfile (Tier 2) (M16)

### Goal
Implement `magpie pkg` minimal engine: `Magpie.toml` parsing, git/path deps, lockfile canonical JSON (§28).

### Work items
* Manifest parser with unknown-key tolerance.
* Lockfile writer: canonical JSON.
* `--offline` support.

---

## Phase 18 — MCP server (Tier 2) (M17)

### Goal
Expose compiler actions over MCP (`magpie mcp serve`) with tool schemas (§29) and token budgets.

### Work items
* MCP transport + JSON-RPC.
* Tools: build/fmt/diagnostics/ctx pack.

---

## Phase 19 — MMS + ctx pack builder (Tier 2) (M18)

### Goal
Implement retrieval index and context pack builder (§24–§25) sufficient for augmenting diagnostics.

### Work items
* MMS capsule schema, digest staleness checks (§22.2).
* Simple lexical retrieval backend.
* Deterministic context-pack selection ladder.

---

## Phase 20 — Web frameworks (Tier 3) (M19)

### Goal
Implement `magpie web dev/build/serve` and the web runtime ABI bridge (§30, §20.1.6).

### Work items
* Define the required callback symbols and generate stubs.
* Runtime HTTP server integration.

---

## Phase 21 — GPU compilation + runtime integration (Tier 3) (M20)

### Goal
Implement Magpie GPU v0.1 subset (§31), SPIR-V backend emission, and runtime kernel registry ABI (§20.1.7).

### Work items
* Parse `gpu` modules + kernel subset.
* Lower GPU kernels to LLVM IR -> SPIR-V.
* `gpu.launch_sync/async` lowering.

---

## Phase 22 — Unsafe + C FFI import (M21)

### Goal
Support `unsafe {}` blocks and `magpie ffi import --header ...` (§32, §5.2.15).

### Work items
* `rawptr<T>` typing rules.
* `ptr.*` lowering.
* Basic header parser or clang-based importer.

---

## Phase 23 — Standard library surface + distribution packaging (M22)

### Goal
Ship `magpie`, `magpie_rt`, and `magpie_std` bundles that match §4 and §35.

### Work items
* Implement minimal `std.io.@println`, `std.hash` primitives, `std.async` wrappers.
* Bundle precompiled runtimes per target triple.

---

## Parallelization map (how to split agents safely)

To maximize parallel progress without conflicts:

* **Agent A (Frontend):** Phases 2–4 (lexer/parser/CSNF/sema)
* **Agent B (Types/Traits):** Phases 5–7
* **Agent C (Ownership/ARC):** Phases 8–10
* **Agent D (Runtime):** Phase 11
* **Agent E (LLVM/Link):** Phase 12
* **Agent F (Diagnostics/LLM):** Phases 1, 14–15

Rule: only one agent edits `magpie_types` at a time; it is the “hotspot” crate.

---

## Minimal “first executable” path (recommended)

If you want the fastest vertical slice to a runnable binary:

1. Phase 1 (CLI skeleton)
2. Phase 2 (parse)
3. Phase 3 (fmt + digest)
4. Phase 4 (sema)
5. Phase 5 (types enough for primitives + Str)
6. Phase 6 (HIR)
7. Phase 9 (MPIR emit + verify)
8. Phase 11 (runtime: str + panic + retain/release minimal)
9. Phase 12 (LLVM codegen + link)

This yields `Hello, world!` very early; then expand into ownership + ARC + collections.
