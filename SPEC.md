# SPEC.md — Magpie Language + Toolchain v0.1 (Authoritative Implementation Specification)

> **Primary constraint:** Magpie optimizes *only* for LLMs (Transformer/Attention architectures) and automated agent workflows. Human ergonomics are explicitly out of scope.
> **Normative keywords:** MUST/SHOULD/MAY are used in the RFC 2119 sense.
> **Scope:** This is a buildable spec: concrete syntax, invariants, IR schemas, algorithms, CLI/MCP contracts, packaging, runtime ABI, web frameworks (frontend SSR + backend services), GPU compute, and compiler-integrated memory/retrieval.

---

## 1. Document control

### 1.1 Versioning

* This spec defines **Magpie v0.1**.
* v0.1 is allowed to break compatibility at any time; once v1.0 ships, breaking changes require a migration tool (`magpie migrate`) and a compatibility policy.

### 1.2 Conformance levels

Implementations may claim conformance to:

* **Magpie Core v0.1** (compiler, IR, safety, ARC, CPU targets)
* **Magpie Web v0.1** (SSR + backend framework)
* **Magpie GPU v0.1** (kernel subset + at least one backend)
* **Magpie Pkg v0.1** (package manager + registry protocol)
* **Magpie MCP v0.1** (MCP server + tool schemas)

### 1.3 Edition system

The `edition` field in `Magpie.toml` controls LLM default settings:

* Edition `"2026"` is the v0.1 baseline with these defaults:
  * `token_budget = 12000`
  * `budget_policy = "balanced"`
  * `auto_fmt = true`
  * `meta_block_required = false` (SHOULD, not MUST)
* Future editions MAY change these defaults without breaking old code.
* Packages with different editions interoperate at the MPIR level.

### 1.4 Build priority (implementation ordering)

* **Tier 1 (ship first):** lexer, parser, CSNF formatter, type checker, ownership checker, ARC insertion, LLVM codegen, runtime, CLI (`build`/`run`/`fmt`/`test`)
* **Tier 2 (ship second):** MMS, diagnostics engine, MCP server, REPL, package manager, linter, doc generator
* **Tier 3 (ship third):** web framework (backend + SSR), GPU compute, WASM targets

All tiers are v0.1 scope with clear sequencing.

### 1.5 Glossary

* **Surface Magpie** (`.mp`): authoring format written by LLMs; LLVM-close; explicit SSA blocks.
* **MPIR** (`.mpir`): Magpie IR, canonical SSA+CFG+typed IR, the compiler's main internal representation and test output.
* **HIR**: High-level IR, the first fully-resolved representation after parsing and name resolution.
* **LLVM IR**: target-independent IR produced by Magpie for LLVM backends.
* **CSNF**: Canonical Source Normal Form (Magpie canonical formatting for stable tokenization).
* **ARC**: automatic reference counting inserted by compiler.
* **MMS**: Magpie Memory Store (compiler-integrated retrieval store).
* **RAG**: retrieval-augmented generation (here: retrieval-augmented compilation outputs for LLMs).
* **Token budget (TB)**: maximum "LLM token" cost allowed for compiler/tool outputs.
* **SID**: Stable ID for progressive disclosure and linking.
* **Unique handle**: an owned reference with exclusive aliasing guarantees (Rust-like), ARC-managed.
* **Shared handle**: explicitly shared ARC reference, clonable and thread-safe.
* **Borrow**: temporary non-owning reference that cannot escape its scope.
* **TCallable**: ARC-managed heap type wrapping a function pointer + optional captured environment.

---

## 2. LLM-first design principles (hard requirements)

These are language + toolchain requirements motivated by Transformer limitations: finite context window, attention dilution over long sequences, and token-level ambiguity.

### 2.1 Canonicalization and determinism (CSNF)

* Every Magpie artifact MUST have a **canonical form**:
  * source canonicalization (`magpie fmt`)
  * canonical symbol ordering
  * canonical numeric formatting
  * canonical type printing
* `magpie fmt` MUST rewrite `.mp` into CSNF — a canonical, deterministic form.
* Canonical output MUST be stable given the same compiler version/config and input semantics.
* **Canonical JSON**: When emitting JSON (`--output json`), Magpie MUST serialize canonical JSON:
  * no whitespace outside string literals
  * object keys lexicographically sorted at every level
  * floats printed with shortest-roundtrip canonical format
* **Stable IDs (SIDs)**: Compiler MUST emit SIDs for modules/symbols to allow retrieval and compact referencing (§18).

**Why (LLM):** canonicalization prevents "format drift" that wastes tokens and causes divergent edits. Stable IDs enable progressive disclosure.

### 2.2 Progressive disclosure built into the language

Magpie MUST support "compressed summaries" that can be loaded without reading full code:

* `.mpd` **Module Definition Digest** files
* `magpie graph symbols --json` **Symbol Graph**
* `magpie graph deps --json` **Dependency Graph**
* `magpie graph ownership --json` **Ownership Graph** (per function)
* `magpie graph cfg --json` **CFG graph** (per function)

All outputs MUST support token budgeting (§3).

**Why (LLM):** agents can retrieve minimal context to resolve unknowns without reloading everything.

### 2.3 Locality constraints (anti-attention-fade rules)

Surface Magpie MUST enforce:

* No implicit imports (lang items are compiler-recognized built-ins, not imports; see §2.4)
* No wildcard imports (except none — v0.1 has no wildcard imports)
* No operator overloading
* No precedence-sensitive operators in v0.1 (all operations are spelled opcodes)
* No hidden control flow (no exceptions; no implicit destructors beyond ARC codegen; explicit `suspend.call` for async)
* No method syntax (`obj.method()` is forbidden; all calls are `call @function { args=[...] }`)
* No closures as a language primitive (TCallable is an explicit heap type with visible captures)
* SSA locals MUST be explicitly typed
* Functions SHOULD be small; compiler MUST provide lint `MPL2001 FN_TOO_LARGE` if exceeding config

Optional but recommended:

* A function SHOULD declare `meta { uses {…} effects {…} cost {…} }` blocks; the compiler can auto-generate and verify them (§7.7).

### 2.4 Lang item auto-availability

The following types are **compiler-recognized built-ins** available in all modules without import:

* `TOption<T>` (from `std.core`)
* `TResult<T,E>` (from `std.core`)
* `bool` (alias of `i1`)
* `unit` (empty value)
* `Str` (heap string)
* `Array<T>` (heap array)
* `Map<K,V>` (heap map)

These are NOT imports. They are built-in types recognized by the compiler. The `imports` block only lists package-level imports.

**Why (LLM):** These types appear in virtually every file. Requiring explicit imports for them adds tokens that are always present and never informative.

---

## 3. LLM token budget system (TB) — global, enforced (MUST)

Magpie MUST treat token consumption of outputs as a first-class bounded resource.

### 3.1 Token budget parameter: sources and precedence (MUST)

Token budget configuration MUST be available via:

1. CLI: `--llm-token-budget <u32>`
2. MCP per-tool request: `llm.token_budget`
3. Env: `MAGPIE_LLM_TOKEN_BUDGET=<u32>`
4. Manifest: `Magpie.toml` `[llm].token_budget`

Precedence: CLI > MCP > Env > Manifest > edition default.

Defaults (edition "2026"):

* In `--llm` mode: `12000`
* Otherwise: unlimited (unless configured)

### 3.2 Tokenizer selection (MUST)

Token counting MUST be performed with a configured tokenizer:

* `approx:utf8_4chars` MUST be supported with zero dependencies:
  * `tokens = ceil(utf8_bytes / 4)`
* Optional tokenizers MAY be supported (plugins):
  * `openai:cl100k_base`, `openai:o200k_base`, `anthropic:claude`, etc.

Compiler MUST expose:

* CLI: `--llm-tokenizer <id>`
* Manifest: `[llm].tokenizer`
* MCP: `llm.tokenizer`

If tokenizer is unavailable, fallback to `approx:utf8_4chars` and emit warning `MPL0802 TOKENIZER_FALLBACK_USED`.

### 3.3 Budget enforcement: deterministic dropping/truncation (MUST)

If output would exceed budget, Magpie MUST reduce payload deterministically in this priority order:

**Tier 0 (MUST keep)**

* `success`
* tool/command metadata
* diagnostic `code`, `severity`, `title`
* `primary_span`
* short `message` (≤ 200 chars; truncated if necessary)

**Tier 1 (SHOULD keep)**

* top 1 suggested fix patch (if available)
* minimal `why.trace` events

**Tier 2 (MAY drop next)**

* secondary spans
* graphs reduced to summaries (node counts + root IDs)

**Tier 3 (drop after Tier 4)**

* long code excerpts, full slices, large retrieved items

**Tier 4 (drop first)**

* long explanations
* doc excerpts beyond top snippet
* verbose graphs

Drop order is: 4→3→2→1.

If Tier 0 alone exceeds budget:

* output MUST contain only:
  * `success=false`
  * a single diagnostic `MPL0801 TOKEN_BUDGET_TOO_SMALL`
  * recommended budget value (estimated Tier0 tokens × 2)

### 3.4 Budget report (MUST)

Every budgeted output MUST include:

```json
{"llm_budget":{"token_budget":12000,"tokenizer":"approx:utf8_4chars","estimated_tokens":11840,"policy":"balanced","dropped":[{"field":"diagnostics[0].explanation_md","reason":"budget"}]}}
```

---

## 4. Toolchain requirements

### 4.1 Delivered binaries

Magpie distribution MUST ship:

* `magpie` (CLI: compiler + package manager + web tooling)
* `magpie-mcp` (or `magpie mcp serve` subcommand)
* `magpie-rt` (runtime library: linked into Magpie programs)
* `magpie-std` (standard packages compiled/installed)

### 4.2 LLVM baseline and licensing

* Magpie v0.1 targets an LLVM release line selected by the project (recommend pinning a specific major/minor for reproducibility).
* Redistribution must respect LLVM's Apache-2.0 WITH LLVM-exception license text.

### 4.3 GPU baseline facts

* NVPTX backend conventions are defined by LLVM documentation.
* SPIR-V is an official LLVM backend since LLVM 20.
* Metal shader converter converts LLVM IR bytecode into Metal-loadable bytecode.
* WebGPU shaders are WGSL source strings; WGSL is the shader language for WebGPU.

---

## 5. CLI specification (`magpie`)

### 5.1 Global flags (MUST)

* `--output <text|json|jsonl>`
  * `json` = single JSON document
  * `jsonl` = JSON lines for streaming build logs
* `--color <auto|always|never>` (ignored in JSON modes)
* `--log-level <error|warn|info|debug|trace>`
* `--profile <dev|release|custom>`
* `--target <llvm-triple>`
* `--emit <artifact-list>` where artifacts include:
  * `llvm-ir` (`.ll`), `llvm-bc` (`.bc`), `object` (`.o`/`.obj`), `asm`, `spv` (`.spv`), `exe`, `shared-lib` (`.so`/`.dylib`), `mpir`, `mpd`, `symgraph`
* `--cache-dir <path>`
* `--jobs <n>`
* `--features <list>`
* `--no-default-features`
* `--offline` (package manager and web tooling must not use network)
* `--llm` (forces LLM-optimized behavior; see §5.3)
* `--llm-token-budget <u32>`
* `--llm-tokenizer <id>`
* `--llm-budget-policy balanced|diagnostics_first|slices_first|minimal`
* `--max-errors <n>` (default 20; max errors collected per compiler pass)
* `--shared-generics` (use vtable-based approach for generics to reduce binary size)

### 5.2 Commands

#### 5.2.1 `magpie new <name>`

Creates:

* `Magpie.toml`
* `src/main.mp`
* `Magpie.lock` (empty)
* `.magpie/` (local cache + generated digests)
* `tests/` (empty test directory)

#### 5.2.2 `magpie build`

Behavior:

* Resolves deps via `magpie pkg` engine
* Produces requested artifacts via `--emit`
* In `--profile dev`, defaults: fast incremental, minimal optimization, full diagnostics, DWARF debug info, `.mpdbg` structured debug
* In `--profile release`, defaults: full optimization, LTO optionally, deterministic build outputs
* Each pass collects up to `--max-errors` errors; dependent passes are skipped if earlier passes fail

#### 5.2.3 `magpie run`

* In dev: MAY use JIT (ORC) or compile+run based on `Magpie.toml` policy
* In release: MUST compile native and run executable
* Supports passing program args after `--`

#### 5.2.4 `magpie repl`

* Persistent session with incremental compilation
* Must expose:
  * `:type <expr>`
  * `:ir <fn>`
  * `:llvm <fn>`
  * `:diag last`

#### 5.2.5 `magpie fmt`

* MUST rewrite `.mp` into CSNF
* MUST update file digests
* MUST ensure every instruction uses explicit key-value argument form
* MUST ensure block labels are canonical `bb0..bbN` (renumbered if necessary)
* `--fix-meta`: auto-generate missing meta blocks in `--llm` mode

#### 5.2.6 `magpie lint`

* MUST provide structured, fixable diagnostics (same format as compiler)
* Full suite: style + complexity + safety + LLM-specific (see §33)
* Configurable severity per lint code in `Magpie.toml`

#### 5.2.7 `magpie test`

* Discovers functions prefixed with `@test_` in `tests/*.mp` and source modules
* Runs each test function, reports pass/fail
* Supports `--filter <pattern>` for selective test execution
* Output format follows §26 diagnostics schema

#### 5.2.8 `magpie doc`

* Generates:
  * `.mpd` digests
  * HTML/JSON docs of public API
  * "LLM doc pack" (compact symbol index, budget-aware)

#### 5.2.9 `magpie web ...`

See §30 (Web frameworks). Includes both frontend SSR and backend services.

#### 5.2.10 `magpie pkg ...`

See §28 (Package manager).

#### 5.2.11 `magpie mcp serve`

See §29.

#### 5.2.12 `magpie memory ...`

* `magpie memory build` (incremental MMS index update)
* `magpie memory query --q "<query>" --k <n> [--kinds ...]`

#### 5.2.13 `magpie ctx pack ...`

Generates prompt-ready context pack bounded by token budget. See §25.

#### 5.2.14 `magpie explain <CODE>`

* MUST output: explanation, minimal examples, canonical remediation templates

#### 5.2.15 `magpie ffi import --header <path> --out <file.mp>`

* Auto-generates extern module declarations from C headers
* Output MUST include TODO markers for unknown ownership

#### 5.2.16 `magpie mpir verify`

* Verifies MPIR correctness: SSA, type IDs, SIDs, call arity, arc ops

### 5.3 `--llm` mode behavior (MUST)

When `--llm` is set OR environment variable `MAGPIE_LLM=1`:

* Default output MUST be JSON (unless overridden).
* Diagnostics MUST include suggested patches whenever possible.
* Compiler MUST emit additional graphs on failure:
  * minimal symbol graph for the file(s) involved
  * ownership trace graph for the failing function(s)
* `magpie fmt` MUST run automatically before build unless `--no-auto-fmt`.
* Token budget is enforced on all outputs.
* MMS retrieval is enabled for diagnostic augmentation.

---

## 6. Source format and module system

### 6.1 File types

* `.mp`: Magpie source (Surface)
* `.mpir`: Core Magpie IR (debug/test)
* `.mpd`: Module definition digest (public API summary)
* `.mpdbg`: Structured debug metadata (JSON, LLM-friendly)
* `Magpie.lock`: lockfile (canonical JSON)

### 6.2 Module-to-file mapping (MUST)

Every `.mp` file is exactly one module. Module path maps to filesystem path:

* `module pkg.sub.module` → `src/sub/module.mp`
* No directory modules, no multiple modules per file.
* This mapping is strict, unambiguous, and LLM-friendly.

### 6.3 Mandatory module header (Surface `.mp`)

Every file MUST begin with a header block:

```
module <module_path>
exports { <export_list> }
imports { <import_list> }
digest "<hex>"
```

Rules:

* `module_path` is a dotted path: `pkg.sub.module`.
* `exports` MUST list all exported symbols (no implicit exports).
* `imports` MUST list all imported symbols using grouped syntax:
  * `imports { std.io::{@println, @readln}, pkg.other::{@foo, TBar} }`
* `digest` MUST equal the compiler-defined digest of the canonicalized file contents (excluding the digest line itself).
* `magpie fmt` is the authority that inserts/updates `digest`.
* Lang items (TOption, TResult, bool, unit, Str, Array, Map) do NOT need to appear in imports.

### 6.4 Digest algorithm (MUST)

* Digest MUST be computed as: `BLAKE3(canonical_source_without_digest_line)`
* Encoded as lowercase hex.
* File digest MUST change if and only if canonical source changes.

### 6.5 Name resolution model

* Every symbol has a **fully qualified name (FQN)**:
  * `pkg.module.@function`
  * `pkg.module.TType`
  * `pkg.module.@global`
* Within a module, local references MUST be either:
  * fully qualified, or
  * imported via `imports` header
* No implicit prelude (lang items are built-in, not prelude).

### 6.6 No overloads (MUST)

* A module MUST NOT define two symbols with the same unqualified name in the same namespace.
* Namespaces:
  * functions/globals: `@`
  * types: `T`
  * SSA locals: `%`
  * signatures: `sig`

### 6.7 Documentation comments

* `;;;` doc comments attach to the next declaration.
* Doc comments are included in `.mpd` output.

---

## 7. Surface language spec (syntax + semantics)

### 7.1 Lexical tokens

#### 7.1.1 Whitespace

* Space, tab, newline are whitespace.
* Newlines are significant only for diagnostics and canonical formatting; grammar is whitespace-insensitive.

#### 7.1.2 Comments

* Line comment: `; ...` to end of line
* Doc comment: `;;; ...` to end of line
* Block comments are NOT supported in v0.1.

#### 7.1.3 Identifiers

* Global function/global: `@` + `[A-Za-z_][A-Za-z0-9_]*`
* SSA value: `%` + same pattern
* Type name: `T` + `[A-Za-z_][A-Za-z0-9_]*`
* Basic block label: `bb` + digits (canonical), e.g. `bb0`, `bb1`

#### 7.1.4 Literals

* Integers: decimal `123`, hex `0x7f`
* Floats: `1.0`, `1.0f32` (canonical form prints via `const.f32 1.0`)
* Strings: `"..."` UTF-8 with escapes `\n \t \\ \" \u{...}`
* Booleans: `true`, `false`

#### 7.1.5 Keywords

`module exports imports digest fn async meta uses effects cost heap value struct enum extern global unsafe gpu target sig impl`

#### 7.1.6 Op keywords

`const.* i.add i.sub i.mul i.sdiv i.udiv i.srem i.urem i.add.wrap i.sub.wrap i.mul.wrap i.add.checked i.sub.checked i.mul.checked i.and i.or i.xor i.shl i.lshr i.ashr f.add f.sub f.mul f.div f.rem f.add.fast f.sub.fast f.mul.fast f.div.fast icmp.* fcmp.* call call_void call.indirect call_void.indirect try suspend.call suspend.await new getfield setfield enum.new enum.tag enum.payload enum.is share clone.shared clone.weak weak.downgrade weak.upgrade cast borrow.shared borrow.mut ptr.null ptr.addr ptr.from_addr ptr.add ptr.load ptr.store arr.new arr.len arr.get arr.set arr.push arr.pop arr.slice arr.contains arr.sort arr.map arr.filter arr.reduce arr.foreach map.new map.len map.get map.get_ref map.set map.delete map.delete_void map.contains_key map.keys map.values str.concat str.len str.eq str.slice str.bytes str.parse_i64 str.parse_u64 str.parse_f64 str.parse_bool str.builder.new str.builder.append_str str.builder.append_i64 str.builder.append_i32 str.builder.append_f64 str.builder.append_bool str.builder.build json.encode json.decode callable.capture arc.retain arc.release arc.retain_weak arc.release_weak panic phi gpu.thread_id gpu.workgroup_id gpu.workgroup_size gpu.global_id gpu.barrier gpu.shared gpu.buffer_load gpu.buffer_store gpu.buffer_len gpu.launch gpu.launch_async`

### 7.2 Grammar (EBNF, Surface `.mp`)

This grammar is intentionally verbose and unambiguous.

```
File          := Header Decl* EOF ;

Header        := "module" ModulePath
                 "exports" "{" ExportList? "}"
                 "imports" "{" ImportList? "}"
                 "digest" StringLit ;

ModulePath    := Ident ("." Ident)* ;

ExportList    := ExportItem ("," ExportItem)* ;
ExportItem    := "@" Ident | "T" Ident ;

ImportList    := ImportGroup ("," ImportGroup)* ;
ImportGroup   := ModulePath "::" "{" ImportItem ("," ImportItem)* "}" ;
ImportItem    := "@" Ident | "T" Ident ;

Decl          := FnDecl
              | AsyncFnDecl
              | UnsafeFnDecl
              | GpuFnDecl
              | TypeDecl
              | ExternModuleDecl
              | GlobalDecl
              | ImplDecl
              | SigDecl ;

FnDecl        := Doc? "fn" FnName "(" Params? ")" "->" Type FnMeta? "{" Block+ "}" ;
AsyncFnDecl   := Doc? "async" "fn" FnName "(" Params? ")" "->" Type FnMeta? "{" Block+ "}" ;
UnsafeFnDecl  := Doc? "unsafe" "fn" FnName "(" Params? ")" "->" Type FnMeta? "{" Block+ "}" ;
GpuFnDecl     := Doc? "gpu" "fn" FnName "(" Params? ")" "->" Type "target" "(" Ident ")" FnMeta? "{" Block+ "}" ;

FnName        := "@" Ident ;

FnMeta        := "meta" "{" UsesBlock? EffectsBlock? CostBlock? "}" ;
UsesBlock     := "uses" "{" FqnRef ("," FqnRef)* "}" ;
EffectsBlock  := "effects" "{" Ident ("," Ident)* "}" ;
CostBlock     := "cost" "{" CostItem ("," CostItem)* "}" ;
CostItem      := Ident "=" IntLit ;

Params        := Param ("," Param)* ;
Param         := SSAName ":" Type ;
SSAName       := "%" Ident ;

TypeDecl      := Doc? ("heap" | "value") ("struct" | "enum") TypeName TypeParams? "{" TypeBody "}" ;
TypeName      := "T" Ident ;
TypeParams    := "<" TypeParam ("," TypeParam)* ">" ;
TypeParam     := Ident ":" TypeConstraint ;
TypeConstraint:= "type" | "send" | "sync" | "hash" | "eq" | "ord" ;

ImplDecl      := "impl" Ident "for" Type "=" FnRef ;

SigDecl       := "sig" "T" Ident "(" TypeList? ")" "->" Type ;
TypeList      := Type ("," Type)* ;

ExternModuleDecl := Doc? "extern" StringLit "module" Ident "{" ExternItem* "}" ;
ExternItem    := "fn" FnName "(" ExternParams? ")" "->" Type ExternAttrs? ;
ExternParams  := ExternParam ("," ExternParam)* ;
ExternParam   := SSAName ":" Type ;
ExternAttrs   := "attrs" "{" ExternAttr* "}" ;

GlobalDecl    := Doc? "global" "@" Ident ":" Type "=" ConstExpr ;

Block         := BlockLabel ":" Instr* Terminator ;
BlockLabel    := "bb" Digits ;
Digits        := [0-9]+ ;

Instr         := SSAName ":" Type "=" Op
              | OpVoid
              | UnsafeBlock ;

UnsafeBlock   := "unsafe" "{" (SSAName ":" Type "=" Op | OpVoid)+ "}" ;

Terminator    := "ret" (ValueRef)?
              | "br" BlockLabel
              | "cbr" ValueRef BlockLabel BlockLabel
              | "switch" ValueRef "{" SwitchArm+ "}" "else" BlockLabel
              | "unreachable" ;

SwitchArm     := "case" ConstLit "->" BlockLabel ;

Op            := ConstExpr

              /* Integer arithmetic (checked by default) */
              | "i.add" BinArgs | "i.sub" BinArgs | "i.mul" BinArgs
              | "i.sdiv" BinArgs | "i.udiv" BinArgs
              | "i.srem" BinArgs | "i.urem" BinArgs

              /* Integer arithmetic (wrapping, safe) */
              | "i.add.wrap" BinArgs | "i.sub.wrap" BinArgs | "i.mul.wrap" BinArgs

              /* Integer arithmetic (checked, returns TOption) */
              | "i.add.checked" BinArgs | "i.sub.checked" BinArgs | "i.mul.checked" BinArgs

              /* Bitwise */
              | "i.and" BinArgs | "i.or" BinArgs | "i.xor" BinArgs
              | "i.shl" BinArgs | "i.lshr" BinArgs | "i.ashr" BinArgs

              /* Float (strict IEEE 754) */
              | "f.add" BinArgs | "f.sub" BinArgs | "f.mul" BinArgs | "f.div" BinArgs | "f.rem" BinArgs

              /* Float (fast-math opt-in) */
              | "f.add.fast" BinArgs | "f.sub.fast" BinArgs | "f.mul.fast" BinArgs | "f.div.fast" BinArgs

              /* Compare (dotted form: icmp.eq, fcmp.oeq, etc.) */
              | "icmp." IcmpPred CmpArgs
              | "fcmp." FcmpPred CmpArgs

              /* Calls */
              | "call" FnRef TypeArgs? "{" CallArgs? "}"
              | "call.indirect" ValueRef "{" CallArgs? "}"
              | "try" FnRef TypeArgs? "{" CallArgs? "}"
              | "suspend.call" FnRef TypeArgs? "{" CallArgs? "}"
              | "suspend.await" "{" "fut" "=" ValueRef "}"

              /* Heap and fields */
              | "new" TypeCtor
              | "getfield" "{" "obj" "=" ValueRef "," "field" "=" FieldName "}"
              | "phi" PhiArgs

              /* Enum operations */
              | "enum.new" "<" Ident ">" "{" CallArgs? "}"
              | "enum.tag" "{" "v" "=" ValueRef "}"
              | "enum.payload" "<" Ident ">" "{" "v" "=" ValueRef "}"
              | "enum.is" "<" Ident ">" "{" "v" "=" ValueRef "}"

              /* Ownership conversions */
              | "share" "{" "v" "=" ValueRef "}"
              | "clone.shared" "{" "v" "=" ValueRef "}"
              | "clone.weak" "{" "v" "=" ValueRef "}"
              | "weak.downgrade" "{" "v" "=" ValueRef "}"
              | "weak.upgrade" "{" "v" "=" ValueRef "}"
              | "cast" "<" PrimType "," PrimType ">" "{" "v" "=" ValueRef "}"

              /* Borrow creation (surface opcodes; lowered to BorrowShared/BorrowMut in HIR) */
              | "borrow.shared" "{" "v" "=" ValueRef "}"
              | "borrow.mut" "{" "v" "=" ValueRef "}"

              /* Raw pointer ops (unsafe-only; MUST appear inside `unsafe {}` blocks) */
              | "ptr.null" "<" Type ">"
              | "ptr.addr" "<" Type ">" "{" "p" "=" ValueRef "}"
              | "ptr.from_addr" "<" Type ">" "{" "addr" "=" ValueRef "}"
              | "ptr.add" "<" Type ">" "{" "p" "=" ValueRef "," "count" "=" ValueRef "}"
              | "ptr.load" "<" Type ">" "{" "p" "=" ValueRef "}"

              /* TCallable */
              | "callable.capture" FnRef "{" CaptureList "}"

              /* Array intrinsics (value-producing) */
              | "arr.new" "<" Type ">" "{" "cap" "=" ValueRef "}"
              | "arr.len" "{" "arr" "=" ValueRef "}"
              | "arr.get" "{" "arr" "=" ValueRef "," "idx" "=" ValueRef "}"
              | "arr.pop" "{" "arr" "=" ValueRef "}"
              | "arr.slice" "{" "arr" "=" ValueRef "," "start" "=" ValueRef "," "end" "=" ValueRef "}"
              | "arr.contains" "{" "arr" "=" ValueRef "," "val" "=" ValueRef "}"
              | "arr.map" "{" "arr" "=" ValueRef "," "fn" "=" ValueRef "}"
              | "arr.filter" "{" "arr" "=" ValueRef "," "fn" "=" ValueRef "}"
              | "arr.reduce" "{" "arr" "=" ValueRef "," "init" "=" ValueRef "," "fn" "=" ValueRef "}"

              /* Map intrinsics (value-producing) */
              | "map.new" "<" Type "," Type ">" "{" "}"
              | "map.len" "{" "map" "=" ValueRef "}"
              | "map.get" "{" "map" "=" ValueRef "," "key" "=" ValueRef "}"
              | "map.get_ref" "{" "map" "=" ValueRef "," "key" "=" ValueRef "}"
              | "map.delete" "{" "map" "=" ValueRef "," "key" "=" ValueRef "}"
              | "map.contains_key" "{" "map" "=" ValueRef "," "key" "=" ValueRef "}"
              | "map.keys" "{" "map" "=" ValueRef "}"
              | "map.values" "{" "map" "=" ValueRef "}"

              /* String intrinsics (value-producing) */
              | "str.concat" "{" "a" "=" ValueRef "," "b" "=" ValueRef "}"
              | "str.len" "{" "s" "=" ValueRef "}"
              | "str.eq" "{" "a" "=" ValueRef "," "b" "=" ValueRef "}"
              | "str.slice" "{" "s" "=" ValueRef "," "start" "=" ValueRef "," "end" "=" ValueRef "}"
              | "str.bytes" "{" "s" "=" ValueRef "}"
              | "str.builder.new" "{" "}"
              | "str.builder.build" "{" "b" "=" ValueRef "}"

              /* String parse intrinsics (value-producing, return TResult) */
              | "str.parse_i64" "{" "s" "=" ValueRef "}"
              | "str.parse_u64" "{" "s" "=" ValueRef "}"
              | "str.parse_f64" "{" "s" "=" ValueRef "}"
              | "str.parse_bool" "{" "s" "=" ValueRef "}"

              /* JSON intrinsics (value-producing, return TResult) */
              | "json.encode" "<" Type ">" "{" "v" "=" ValueRef "}"
              | "json.decode" "<" Type ">" "{" "s" "=" ValueRef "}"

              /* GPU device ops (value-producing; only valid inside gpu fn) */
              | "gpu.thread_id" "{" "dim" "=" ValueRef "}"
              | "gpu.workgroup_id" "{" "dim" "=" ValueRef "}"
              | "gpu.workgroup_size" "{" "dim" "=" ValueRef "}"
              | "gpu.global_id" "{" "dim" "=" ValueRef "}"
              | "gpu.buffer_load" "<" Type ">" "{" "buf" "=" ValueRef "," "idx" "=" ValueRef "}"
              | "gpu.buffer_len" "<" Type ">" "{" "buf" "=" ValueRef "}"
              | "gpu.shared" "<" IntLit "," Type ">"
              /* GPU host ops (value-producing; only valid outside gpu fn) */
              | "gpu.launch" "{" "device" "=" ValueRef "," "kernel" "=" FnRef "," "grid" "=" ArgValue "," "block" "=" ArgValue "," "args" "=" ArgValue "}"
              | "gpu.launch_async" "{" "device" "=" ValueRef "," "kernel" "=" FnRef "," "grid" "=" ArgValue "," "block" "=" ArgValue "," "args" "=" ArgValue "}" ;

OpVoid        := "call_void" FnRef TypeArgs? "{" CallArgs? "}"
              | "call_void.indirect" ValueRef "{" CallArgs? "}"
              | "setfield" "{" "obj" "=" ValueRef "," "field" "=" FieldName "," "val" "=" ValueRef "}"
              | "panic" "{" "msg" "=" ValueRef "}"
              | "ptr.store" "<" Type ">" "{" "p" "=" ValueRef "," "v" "=" ValueRef "}"

              /* Array (void) */
              | "arr.set" "{" "arr" "=" ValueRef "," "idx" "=" ValueRef "," "val" "=" ValueRef "}"
              | "arr.push" "{" "arr" "=" ValueRef "," "val" "=" ValueRef "}"
              | "arr.sort" "{" "arr" "=" ValueRef "}"
              | "arr.foreach" "{" "arr" "=" ValueRef "," "fn" "=" ValueRef "}"

              /* Map (void) */
              | "map.set" "{" "map" "=" ValueRef "," "key" "=" ValueRef "," "val" "=" ValueRef "}"
              | "map.delete_void" "{" "map" "=" ValueRef "," "key" "=" ValueRef "}"

              /* StringBuilder (void) */
              | "str.builder.append_str" "{" "b" "=" ValueRef "," "s" "=" ValueRef "}"
              | "str.builder.append_i64" "{" "b" "=" ValueRef "," "v" "=" ValueRef "}"
              | "str.builder.append_i32" "{" "b" "=" ValueRef "," "v" "=" ValueRef "}"
              | "str.builder.append_f64" "{" "b" "=" ValueRef "," "v" "=" ValueRef "}"
              | "str.builder.append_bool" "{" "b" "=" ValueRef "," "v" "=" ValueRef "}"

              /* GPU device ops (void; only valid inside gpu fn) */
              | "gpu.barrier"
              | "gpu.buffer_store" "<" Type ">" "{" "buf" "=" ValueRef "," "idx" "=" ValueRef "," "v" "=" ValueRef "}" ;

CallArgs      := CallArg ("," CallArg)* ;
CallArg       := Ident "=" ArgValue ;
ArgValue      := ValueRef | "[" (ArgListElem ("," ArgListElem)*)? "]" | FnRef ;
ArgListElem   := ValueRef | FnRef ;

BinArgs       := "{" "lhs" "=" ValueRef "," "rhs" "=" ValueRef "}" ;
CmpArgs       := "{" "lhs" "=" ValueRef "," "rhs" "=" ValueRef "}" ;
CaptureList   := CaptureItem ("," CaptureItem)* ;
CaptureItem   := Ident "=" ValueRef ;

TypeCtor      := NamedType "{" FieldInit ("," FieldInit)* "}" ;
FieldInit     := FieldName "=" ValueRef ;

IcmpPred      := "eq" | "ne" | "slt" | "sgt" | "sle" | "sge"
              | "ult" | "ugt" | "ule" | "uge" ;
FcmpPred      := "oeq" | "one" | "olt" | "ogt" | "ole" | "oge" ;

ExternAttr    := Ident "=" StringLit ;

FqnRef        := ModulePath "." ("@" Ident | "T" Ident)
              | "@" Ident
              | "T" Ident ;

ConstExpr     := "const." Type ConstLit ;
ConstLit      := IntLit | FloatLit | StringLit | "true" | "false" | "unit" ;

ValueRef      := SSAName | ConstExpr ;
FnRef         := FnName | QualifiedFnName ;
QualifiedFnName := ModulePath "." FnName ;
TypeRef       := TypeName | ModulePath "." TypeName ;

PhiArgs       := Type "{" PhiIncoming ("," PhiIncoming)* "}" ;
PhiIncoming   := "[" BlockLabel ":" ValueRef "]" ;

FieldName     := Ident ;
TypeArgs      := "<" Type ("," Type)* ">" ;

Type          := OwnershipMod? BaseType ;
OwnershipMod  := "shared" | "borrow" | "mutborrow" | "weak" ;
BaseType      := PrimType | NamedType | BuiltinType | CallableType | RawPtrType ;
RawPtrType    := "rawptr" "<" Type ">" ;
PrimType      := "i1" | "i8" | "i16" | "i32" | "i64" | "i128"
              | "u1" | "u8" | "u16" | "u32" | "u64" | "u128"
              | "f16" | "f32" | "f64" | "bool" | "unit" ;
BuiltinType   := "Str" | "Array" "<" Type ">" | "Map" "<" Type "," Type ">"
              | "TOption" "<" Type ">" | "TResult" "<" Type "," Type ">"
              | "TStrBuilder"
              | "TMutex" "<" Type ">" | "TRwLock" "<" Type ">" | "TCell" "<" Type ">"
              | "TFuture" "<" Type ">"
              | "TChannelSend" "<" Type ">" | "TChannelRecv" "<" Type ">" ;
NamedType     := TypeRef TypeArgs? ;
CallableType  := "TCallable" "<" TypeRef ">" ;

Doc           := (";;;" .* newline)+ ;
```

**Canonical style constraint:** operations MUST use the `{ key=value }` argument form, even when redundant, to prevent ambiguous tokenization.

**`new` example:**

```
%person: TPerson = new TPerson { name=%name_str, age=%age_val }
```

### 7.3 Semantics: evaluation order

* Instructions execute in program order within a basic block.
* SSA values are immutable once defined.
* `setfield` is a side-effecting instruction requiring mutable access.

### 7.4 Integer overflow (MUST)

* All default integer arithmetic (`i.add`, `i.sub`, `i.mul`) is **checked**: overflow panics with a diagnostic message including the values and operation.
* `i.add.wrap`, `i.sub.wrap`, `i.mul.wrap` perform two's complement wrapping (defined behavior, no UB).
* `i.add.checked`, `i.sub.checked`, `i.mul.checked` return `TOption<T>` — `Some` on success, `None` on overflow.
* Division by zero always panics (for both `i.sdiv`/`i.udiv`).

**Why:** Magpie guarantees no UB in safe code. Checked arithmetic is the default safety mechanism.

### 7.5 Float semantics (MUST)

* All default float operations (`f.add`, `f.sub`, etc.) follow **strict IEEE 754**: NaN propagates, signed zeros preserved, denormals supported. Every float operation has defined behavior.
* `f.add.fast`, `f.sub.fast`, etc. opt into LLVM fast-math flags (loses NaN/inf determinism).
* No UB — every float operation has defined behavior in both modes.

### 7.6 No implicit casts

* Any conversion must use an explicit opcode: `cast<i32, i64> { v=%x }`, etc. Cast is restricted to primitive types in v0.1.
* Canonical form always prints casts explicitly.

### 7.7 Function meta blocks (`meta {}`) (LLM locality feature)

A function MAY include:

```
meta {
  uses { pkg.mod.@foo, std.io.@println, pkg.mod.TThing }
  effects { io.write, alloc.heap, net.tcp, fs.read }
  cost { approx_instructions=120, approx_allocs=2 }
}
```

Rules:

* In `--llm` mode, the compiler SHOULD auto-generate and insert missing meta blocks (`magpie fmt --fix-meta`).
* If present, `uses` MUST be a complete set of referenced external symbols (excluding SSA locals).
* If present, `effects` MUST match inferred effects; mismatch is a warning in v0.1, error in v1.
* If present, `cost` is user-authored and **verified with tolerance**: if actual cost exceeds declared by >2x, emit warning `MPL2002 COST_UNDERESTIMATE`.

### 7.8 `try` opcode (error propagation)

The `try` opcode provides compact error propagation:

```
%user: TUser = try @get_user { id=%id }
```

The compiler desugars `try` to:

1. `%result: TResult<TUser, TErr> = call @get_user { id=%id }`
2. `%tag: i32 = enum.tag { v=%result }`
3. `%is_err: bool = icmp.eq { lhs=%tag, rhs=const.i32 1 }`
4. `cbr %is_err bb_err bb_ok`
5. In `bb_ok`: `%user: TUser = enum.payload<Ok> { v=%result }`
6. In `bb_err`: `ret %result` (propagate the full `TResult<_,E>` value unchanged)

The desugared form is what appears in MPIR. The function's return type MUST be `TResult<T,E>` to use `try`.

### 7.9 Hello world example

```
module hello.main
exports { @main }
imports { std.io::{@println} }
digest "0000000000000000"

fn @main() -> i32 {
bb0:
  %msg: Str = const.Str "Hello, world!"
  call_void std.io.@println { args=[%msg] }
  ret const.i32 0
}
```

---

## 8. Type system (strong + explicit)

### 8.1 Type categories

#### 8.1.1 Primitive value types

* Signed integers: `i1, i8, i16, i32, i64, i128`
* Unsigned: `u1, u8, u16, u32, u64, u128`
* Floats: `f16, f32, f64`
* `bool` is alias of `i1`
* `unit` is the empty value

#### 8.1.2 Aggregates (planned for v0.2)

* `vec<N, T>` fixed-size SIMD vector (value type) — **deferred to v0.2**
* `arr<N, T>` fixed-size array (value type) — **deferred to v0.2**
* `tuple<T0,T1,...>` (value type) — **deferred to v0.2**

> **v0.1 restriction:** These types exist in `TypeKind` as internal-only representations (the compiler may use them for ABI lowering), but MUST NOT appear in surface `.mp` syntax. The parser MUST reject them with `MPT1021 AGGREGATE_TYPE_DEFERRED`. Heap `Array<T>` (§8.1.3) is the v0.1 dynamic array type.

#### 8.1.3 Heap-managed types (ARC)

* `Str`
* `Array<T>`
* `Map<K,V>`
* `TStrBuilder`
* user-defined `heap struct TName { ... }`
* user-defined `heap enum TName { ... }`
* `TCallable<TSig>` (callable with captures)

#### 8.1.4 Lang items (compiler-known, auto-available)

* `TOption<T>` — builtin **value enum** with variants:
  * `None { }`
  * `Some { field v: T }`
* `TResult<T,E>` — builtin **value enum** with variants:
  * `Ok { field v: T }`
  * `Err { field e: E }`

These are the only value enums supported in v0.1. User-defined `value enum` remains deferred to v0.2.

Layout rules for lang items (codegen-visible; semantic model is **value enum**):

* `TOption<T>` when `T` is a heap handle type: niche optimization (NULL = None, non-NULL = Some)
* `TOption<T>` when `T` is a non-handle value type: tagged `{ i1 tag; T payload }` (0=None, 1=Some)
* `TResult<T,E>`: tagged `{ i1 tag; union { T ok; E err } }` (0=Ok, 1=Err)

Ownership + ARC notes:

* `TOption` / `TResult` values are **not** ARC-managed heap objects themselves, but they may contain heap handles.
* If the payload contains heap handles, the compiler MUST apply move/borrow rules + ARC insertion/drop elaboration to those payload handles.
* `enum.tag`, `enum.is`, `enum.payload`, and `enum.new` apply to both heap enums and these builtin value enums.

### 8.2 Ownership modifiers (compile-time)

Ownership is part of the type:

* `T` (unique owned handle — default for heap-managed types)
* `borrow T` (scoped shared borrow; v0.1 borrows are non-escaping)
* `mutborrow T` (scoped exclusive borrow; v0.1 borrows are non-escaping)
* `shared T` (explicit shareable ARC handle; clonable; thread-safe; immutable by default) — heap-managed only
* `weak T` (non-owning weak reference; may be null-like) — heap-managed only
* `rawptr<T>` (unsafe-only raw pointer type; only usable inside `unsafe {}`)

Rules:

* `shared` and `weak` are only valid when `T` is heap-managed.
* `borrow` / `mutborrow` are valid for **any** type `T` and represent a scoped reference to a value of type `T`.
  * For heap-managed `T`, `borrow T` / `mutborrow T` are non-owning references to the heap object.
  * For value types (including builtin value enums like `TOption`/`TResult`), `borrow T` / `mutborrow T` refer to an in-memory slot (stack/local spill, heap field, or container element).

### 8.3 Type declarations

#### 8.3.1 Heap struct

```
heap struct TPerson {
  field name: Str
  field age: i32
}
```

Rules:
* Heap structs are ARC-managed.
* Fields may be value types or heap types.

#### 8.3.2 Value struct

```
value struct TVec2 {
  field x: f32
  field y: f32
}
```

Rules:
* Value structs copy by value.
* No ARC actions for value types.
* In v0.1, value struct fields MUST NOT include heap handles. Violation: `MPT1005 VALUE_TYPE_CONTAINS_HEAP`.

#### 8.3.3 Enums

```
heap enum TShape {
  variant Circle { field radius: f64 }
  variant Rect { field w: f64, field h: f64 }
}
```

Rules:
* Heap enums are ARC-managed (payload may be heap/value).
* Enum access/construction is via dedicated opcodes: `enum.new<VariantName>`, `enum.tag`, `enum.payload<VariantName>`, `enum.is<VariantName>`.
* **User-defined value enums are deferred to v0.2.** The grammar accepts `value enum` but the compiler MUST reject it with `MPT1020 VALUE_ENUM_DEFERRED`. The only value enums available in v0.1 are the builtin lang items `TOption` and `TResult` (§8.1.4).


**`enum.new` construction rules (MUST):**

* Surface form: `%v: TEnum = enum.new<VariantName> { field1=%x, field2=%y }`
* The **result type annotation** determines the enum type (`TEnum`). The compiler MUST reject `enum.new` if the result type is not an enum type.
* `VariantName` MUST exist for that enum type.
* The provided `key=value` pairs MUST match the variant's fields exactly (same names, same count). Order is irrelevant; canonical formatting sorts by field name.
* For empty variants, the arg block MUST be present but MAY be empty: `enum.new<None> { }`.

**Semantics:**

* For **heap enums**: allocate a new enum object, set the tag, write payload fields, and return a **unique** handle to the new enum object.
* For builtin **value enums** (`TOption`, `TResult`): construct the value directly according to §8.1.4 layout rules (no heap allocation).

#### 8.3.4 Recursive types

* Heap struct/enum types can reference themselves or each other, but self-referential fields MUST use explicit indirection: `TOption<T>` or `weak T`.
* Direct `field next: TNode` within `TNode` is forbidden — use `field next: TOption<TNode>` instead.
* Value structs cannot be recursive (compiler error `MPT1010 RECURSIVE_VALUE_TYPE`).

### 8.4 Generics (v0.1 target)

Magpie MUST support **monomorphization** generics for:

* `Array<T>`, `Map<K,V>`
* user-defined types and functions

Constraints:

* `type` (any type)
* `send`, `sync` (concurrency markers)
* `hash`, `eq`, `ord` (behavioral traits with impl blocks; see §9)

### 8.5 Type inference policy

* Surface Magpie MUST NOT infer types of SSA values; they MUST be annotated.
* The compiler MAY infer some generic parameters only when the call site provides all needed information; canonical form SHOULD print all instantiated types explicitly.

### 8.6 Monomorphization controls

* Default is full monomorphization (fast, no vtables).
* `--shared-generics` flag switches to vtable-based approach for specific functions to reduce binary size.
* **ABI compatibility rule:** all packages in a binary MUST use the same generics mode. Mixed compilation (some packages with `--shared-generics`, others without) is forbidden. The linker MUST reject mixed-mode object files with `MPL2021 MIXED_GENERICS_MODE`. Detection mechanism: the compiler emits a global symbol `mp$0$ABI$generics_mode` with value 0 (monomorphized) or 1 (shared). The linker checks all object files for this symbol and rejects if values differ.
* `[build].max_mono_instances` in `Magpie.toml` sets a budget. Compiler errors `MPL2020 EXCESSIVE_MONO` if exceeded. This budget is **global** across all packages in the build — each package contributes to the shared count.
* In release profile, LTO deduplicates identical instantiations.

---

## 9. Trait system (minimal, explicit)

### 9.1 Overview

Magpie v0.1 has a minimal trait system supporting:

* Marker traits: `send`, `sync`
* Behavioral traits: `hash`, `eq`, `ord`

### 9.2 Trait declarations (built-in only in v0.1)

Traits are compiler-known in v0.1. The built-in traits are:

* `hash` — requires `fn @hash_<Type>(%self: borrow <Type>) -> u64`
* `eq` — requires `fn @eq_<Type>(%a: borrow <Type>, %b: borrow <Type>) -> bool`
* `ord` — requires `fn @ord_<Type>(%a: borrow <Type>, %b: borrow <Type>) -> i32` (returns -1, 0, 1)
* `send` — marker, no function required
* `sync` — marker, no function required

### 9.3 Impl declarations

Standalone binding syntax — functions defined separately:

```
;;; Bind trait implementations
impl hash for TPerson = @hash_TPerson
impl eq for TPerson = @eq_TPerson

;;; Implementation function (normal global fn)
fn @hash_TPerson(%self: borrow TPerson) -> u64 {
bb0:
  %name_hash: u64 = call std.hash.@hash_Str { args=[%self_name] }
  ; ...
  ret %h
}
```

Rules:
* `impl Trait for Type = @function` binds a trait to a type with a specific function.
* The function MUST match the trait's required signature.
* The function is a normal global function accessible by its FQN.

### 9.4 Dispatch model

* All trait dispatch is **monomorphized** (static).
* At monomorphization time, the compiler resolves `call @hash { k=%k }` to the concrete `call @hash_TMyKey { k=%k }` based on the impl binding.
* No vtables in the default mode.
* LLMs see the exact function being called in MPIR output.

### 9.5 Coherence (strict orphan rules)

* An `impl` declaration is only valid if the current package owns either the trait or the type (or both).
* No implementing foreign traits for foreign types.
* Violation: `MPT1200 ORPHAN_IMPL`.

---

### 9.5 Runtime callback ABI for `hash` / `eq` / `ord` (v0.1)

Some runtime collection operations (Map hashing/equality, Array `contains`, Array `sort`) call back into compiler-generated functions.

For each concrete type `T` that is used with these operations, the compiler MUST generate C-ABI wrappers with the following signatures:

```c
// The pointers point to an in-memory value with layout identical to Magpie type `T`.
uint64_t mp_cb_hash_T(const void* x_bytes);
int32_t  mp_cb_eq_T(const void* a_bytes, const void* b_bytes);   // 0=false, 1=true
int32_t  mp_cb_cmp_T(const void* a_bytes, const void* b_bytes);  // negative / 0 / positive
```

Lowering contracts (normative):

* `map.new<K,V>` passes `hash_fn = &mp_cb_hash_K` and `eq_fn = &mp_cb_eq_K` to `mp_rt_map_new`.
* `arr.contains<T>` passes `eq_fn = &mp_cb_eq_T` to `mp_rt_arr_contains`.
* `arr.sort<T>` passes `cmp = &mp_cb_cmp_T` to `mp_rt_arr_sort`.

Determinism requirements:

* These wrappers MUST be pure (no I/O, no mutation, no nondeterminism) and MUST return the same result for the same input bytes.
* For heap handle types (e.g., `Str`), the wrapper reads the handle and hashes/compares the underlying object content.

## 10. Ownership and borrow checking (Rust-like guarantees)

### 10.1 Core invariants

Magpie enforces Rust-like aliasing for **move-only** values, and allows free duplication for **copy** values.

Definitions (v0.1):

* **Copy types**: primitives, `rawptr<T>`, and `value struct` types whose fields are all Copy.
* **Move-only types**: all heap handles (`T`, `shared T`, `weak T`) and any value type that contains a move-only field (including many instantiations of `TOption` / `TResult`).

Invariants for move-only values:

* A **unique** handle `T` has exclusive ownership.
* A `borrow T` may coexist with other `borrow T` borrows, but not with `mutborrow T`.
* A `mutborrow T` is exclusive and cannot coexist with any other borrow.
* No borrow may outlive the scope in which it was created (non-escaping).
* A move-only value cannot be used after it is moved.
* Borrows are block-local: MUST NOT appear in `phi`, MUST NOT cross basic block boundaries.

Invariants for Copy values:

* Copy values may be used multiple times with no move tracking.
* Creating borrows of Copy values is allowed, but still subject to the borrow lifetime restrictions in §10.5.

### 10.2 Move semantics

* Assignment/consumption of any **move-only** value is a **move** (transfers ownership), not an implicit clone.
* After move, the source becomes invalid (use-after-move is an error).
* Copy types are exempt.

### 10.3 Explicit cloning/sharing

* Convert unique → shared: `%s: shared TPerson = share { v=%p }` (consumes %p)
* Clone shared: `%s2: shared TPerson = clone.shared { v=%s }` (does NOT consume %s)
* Clone weak: `%w2: weak TPerson = clone.weak { v=%w }` (does NOT consume %w)
* No implicit aliasing.

### 10.4 Mutation rules

* `setfield` requires operand to be `mutborrow T` or unique `T` with no active borrows.
* Mutating a `shared T` is forbidden in v0.1 (use `TMutex<T>` for interior mutability).


### 10.4.1 Storable types and projection reads (unique-in-storage)

Magpie v0.1 allows heap handles (including **unique** handles) to be stored inside heap fields and collections. Containers/fields **own** their stored values.

#### Storable types (MUST)

* A type is **storable** if it may appear in:
  * heap struct/enum fields
  * `Array<T>` elements
  * `Map<K,V>` keys/values
  * globals
* In v0.1, **all types are storable except** `borrow T` and `mutborrow T`.
  * Attempting to store a borrow MUST error: `MPO0003 BORROW_ESCAPES_SCOPE`.

#### Store/write semantics (MUST)

* Writing a value into a field/array/map **consumes** the input value (move-in). There are **no implicit clones**.
* Overwriting an existing stored value MUST **drop** the old value first (releasing any contained heap handles), then move-in the new value.
* To store multiple references to the same heap object, code MUST explicitly use `share`, `clone.shared`, `weak.downgrade`, and/or `clone.weak`.

#### Read/projection semantics (MUST)

Reading from storage MUST NOT transfer ownership. Therefore, read operations return:

* A **value copy** for Copy types.
* A **borrow** for move-only types (including heap handles and non-Copy value enums/structs).

##### `getfield` rules

* Operand type requirement:
  * `obj` MUST be `borrow TStruct` or `mutborrow TStruct`.
* Result type rule, based on the field type `F`:
  * If `F` is **Copy**: `getfield` returns `F` by value.
  * If `F` is a **strong heap handle**:
    * If `F` is `shared T`: `getfield` returns `borrow T` (shared borrow of the underlying heap object).
    * If `F` is `T` (a unique strong handle):
      * If `obj` is `borrow TStruct`: result is `borrow T`.
      * If `obj` is `mutborrow TStruct`: result is `mutborrow T` (exclusive borrow of the underlying heap object).
  * If `F` is a **weak heap handle** (`weak T`): `getfield` returns `weak T` by **cloning** the weak handle (compiler inserts `arc.retain_weak`).
  * Otherwise (`F` is move-only value type, e.g. `TOption<T>` with move-only payload): `getfield` returns `borrow F` (borrow of the field slot).

##### `arr.get` rules

* Operand type requirement:
  * `arr` MUST be `borrow Array<T>` or `mutborrow Array<T>`.
* Result type rule is the same as `getfield`, with `F := T` (the element type).

##### `map.get` rules (v0.1)

* Operand type requirement:
  * `map` MUST be `borrow Map<K,V>` or `mutborrow Map<K,V>`.
* Return type (v0.1):
  * If `V` is Copy: `TOption<V>`.
  * If `V` is `weak T`: `TOption<weak T>` (weak handle is cloned on success).
  * If `V` is `shared T`: `TOption<shared T>` (strong handle is cloned on success; compiler inserts `arc.retain_strong`).
  * Otherwise (including unique strong handles `T` and move-only value types): **forbidden in safe v0.1**, because it would require returning borrows inside `TOption`.
    * Violation: `MPO0103 MAP_GET_REQUIRES_DUPABLE_V`.
    * Use `map.contains_key` + `map.get_ref` (below), or `map.delete` to move the value out, or unsafe code.

##### `map.get_ref` rules (v0.1)

* Operand type requirement:
  * `map` MUST be `borrow Map<K,V>` or `mutborrow Map<K,V>`.
* Semantics:
  * Panics if the key is not present (mirrors `arr.get` panicking on out-of-bounds).
  * Otherwise returns a projection of the stored value without transferring ownership.
* Result type rule:
  * Same as `arr.get`, with `T := V`.

##### Duplication-producing collection intrinsics (v0.1 constraints)

The following intrinsics allocate a new collection and therefore duplicate elements/keys/values:

* `arr.slice`, `arr.filter`, `map.keys`, `map.values`.

They require the relevant element/key/value types to be **Dupable**.

* `Dupable` (v0.1): `Copy` types, `shared T` strong handles, and `weak T` handles.
* Unique strong handles `T` are **not** Dupable.
* Types that contain borrows (directly or indirectly) are never Dupable.

Violation: `MPT1022 COLLECTION_DUPLICATION_REQUIRES_DUPABLE`.

##### Trait requirements for collection algorithms (v0.1)

* `arr.contains` requires `impl eq for T`.
* `arr.sort` requires `impl ord for T`.
* `map.new` requires `impl hash for K` and `impl eq for K`.

Violation: `MPT1023 MISSING_REQUIRED_TRAIT_IMPL`.

#### Move-out operations

* `arr.pop` moves an element out of the array (returns an owned `TOption<T>`).
* `map.delete` moves the removed value out of the map (returns an owned `TOption<V>`).
* `map.delete_void` deletes the entry and drops the removed value (returns `unit`).
* These are the only stable, v0.1-supported ways to extract owned move-only values from collections without unsafe code.

### 10.5 Lifetime model (v0.1)

* Lifetimes are lexical scopes only (no named lifetimes).
* Borrows cannot be returned from functions in v0.1, **except** compiler-known interior-mutability intrinsics (`@mutex_lock`, `@rwlock_read`, `@rwlock_write`, `@rwlock_unlock`) which return scoped borrows with automatic scope-end release semantics (§13.3.1). These are not considered function-return borrows for the purposes of this rule — the compiler inserts the corresponding unlock/release at the borrow's scope end.
* Borrows cannot be stored into heap fields, arrays, or globals. Violation: `MPO0003 BORROW_ESCAPES_SCOPE`.
* Borrows cannot cross basic block boundaries. Violation: `MPO0101 BORROW_CROSSES_BLOCK`.
* Borrows cannot appear in phi nodes. Violation: `MPO0102 BORROW_IN_PHI`.

### 10.6 Formal ownership dataflow (moved-set analysis)

The complete formal specification follows:

* Domain: `Moved ⊆ O` where `O` = set of all locals whose types are **move-only** (heap handles and non-Copy value types). Copy-typed locals are excluded from `O`.
* Transfer rules: Rule U (use-after-move check), Rule C (consume and add to moved set)
* Join rule at CFG merge: `Moved_in[B] = ⋃ Moved_out[Pi]` (union = "maybe moved")
* Phi incoming use semantics: phi is treated as consuming the incoming value on each edge

### 10.7 Formal borrow checking (block-local, linear scan)

For each owned local `v ∈ O`, maintain within each block:

* `SharedCount[v]: u32`
* `MutActive[v]: bool`

Borrow creation:
* `BorrowShared`: requires `MutActive[v] == false`, increments `SharedCount[v]`
* `BorrowMut`: requires `SharedCount[v] == 0` and `MutActive[v] == false`, sets `MutActive[v] = true`

Borrow end (at last use):
* Shared: `SharedCount[v] -= 1`
* Mut: `MutActive[v] = false`

Moves while borrowed: any `Consume(v)` requires `SharedCount[v] == 0` and `MutActive[v] == false`. Violation: `MPO0011 MOVE_WHILE_BORROWED`.

### 10.8 Call argument mode

* By-value parameter whose type is **move-only** (heap handles and non-Copy value types): argument is **consumed**.
* By-value parameter whose type is **Copy**: argument is copied (not consumed).
* `borrow` parameter: temporary shared borrow lasting only for the call.
* `mutborrow` parameter: temporary mutable borrow lasting only for the call.

### 10.9 Collection intrinsic ownership requirements

Collection-mutating intrinsics follow the same ownership rules as `setfield`:

| Ownership required | Intrinsics |
|-------------------|------------|
| **unique or mutborrow** | `arr.set`, `arr.push`, `arr.pop`, `arr.sort`, `map.set`, `map.delete`, `map.delete_void`, `str.builder.append_*` |
| **borrow (read-only)** | `arr.len`, `arr.get`, `arr.slice`, `arr.contains`, `arr.map`, `arr.filter`, `arr.reduce`, `arr.foreach`, `map.len`, `map.get`, `map.get_ref`, `map.contains_key`, `map.keys`, `map.values`, `str.len`, `str.eq`, `str.slice`, `str.bytes` |
| **consumes (moves)** | `arr.new`, `map.new`, `str.concat`, `str.builder.new`, `str.builder.build` |

* Mutating intrinsics on `shared` references are forbidden — the ownership checker MUST reject them with `MPO0004 SHARED_MUTATION`.
* `arr.map`/`arr.filter`/`arr.reduce`/`arr.foreach` only borrow the array; the `func` (TCallable) argument is borrowed for the call duration.

---

## 11. ARC memory model (Swift-like insertion, with Magpie ownership)

### 11.1 Key idea

ARC manages *lifetime*, while ownership checking manages *aliasing safety*.

* Unique handles still use refcounted allocation under the hood.
* Most refcounts are 1 unless explicitly shared.

### 11.2 Runtime representation

Every heap object has:
* header: strong refcount (atomic), weak refcount, type id, flags, reserved
* payload bytes

### 11.3 ARC operations

* `arc.retain` / `arc.release` (strong)
* `arc.retain_weak` / `arc.release_weak`

### 11.4 ARC insertion rules

* `new` initializes strong=1, no retain needed.
* `share` consumes Unique, produces Shared WITHOUT changing count (type reclassification).
* `clone.shared` emits `arc.retain`.
* `clone.weak` emits `arc.retain_weak`.
* `weak.downgrade` emits `arc.retain_weak`.
* End of scope for strong handle: emit `arc.release`.
* Field/element overwrite: **drop old**, then **move in new**. There is no implicit retain of the new value; retains only occur when the IR explicitly clones (`clone.shared`, `clone.weak`, `weak.downgrade`) or when a spec-defined read clones a weak handle (see §10.4.1).

### 11.5 Drop elaboration

* Drop functions are **compiler-generated only** — no user-defined destructors.
* Drop recursively releases all contained heap handles (including handles nested inside value enums like `TOption`/`TResult` and inside nested value structs).
* Types needing custom cleanup (files, connections) require explicit `call @close { handle=%h }` before scope exit.
* For values (including `TOption`/`TResult`) that die without being moved: insert the appropriate releases for any contained handles (`arc.release` for strong, `arc.release_weak` for weak), using the compiler-generated drop logic.
* For values that are moved: no release inserted (ownership transferred).

### 11.6 ARC optimization passes

MUST implement:
* retain/release pairing elimination in straight-line code
* CFG-aware ARC: sink releases, hoist retains, eliminate redundant retains on same value
* Optional: no-escape stack promotion
* **Drop function deduplication:** LLVM ICF (Identical Code Folding) during LTO SHOULD deduplicate structurally identical drop functions. The compiler MAY also detect identical drop bodies at the MPIR level and emit a single shared drop function with multiple type_id aliases.

### 11.7 Cycle handling

* `weak T` type modifier for cycle breaking.
* Weak references do not keep object alive.
* Weak can be upgraded to `TOption<shared T>` via `weak.upgrade`.

### 11.8 Threading + ARC

* `shared T` strong refcount operations MUST be atomic.
* Unique `T` strong refcount operations MAY be non-atomic.
* Converting `T -> shared T` MUST upgrade to atomic refcounting.

---

## 12. Async model (stackless coroutines)

### 12.1 Overview

Magpie uses stackless coroutines with explicit `suspend.call` and `suspend.await` opcodes. Every suspension point is visible in the source code. The compiler generates state machines.

### 12.2 Async function declaration

```
async fn @fetch_user(%id: u64) -> TUser {
bb0:
  %conn: TDbConn = suspend.call @db.@connect {}
  %row: TRow = suspend.call @db.@query { conn=%conn, id=%id }
  %user: TUser = call @parse_user { row=%row }
  ret %user
}
```

Rules:
* `async fn` declares an async function.
* `suspend.call` is a suspension opcode — the function yields to the executor at this point.
* `suspend.call` and `suspend.await` MUST appear only inside `async fn` bodies. Violation: `MPAS0001 SUSPEND_IN_NON_ASYNC`.
* `suspend.await` is a suspension opcode that awaits an existing `TFuture<T>` value (see §12.4).
* The compiler generates a state machine that saves/restores state across suspend points.
* Every suspension point is a visible `suspend.call` instruction — no hidden suspend.

### 12.3 Async + ARC interaction

* The compiler generates a heap struct per async function containing all SSA values that are live across suspend points.
* On suspend, live values are moved into the state struct (ARC-managed).
* On resume, values are moved back out.
* The state struct IS the future object.
* Drop of the future releases all captured values (compiler-generated drop).
* The runtime provides the executor that manages scheduling and polling.

### 12.4 Async return type

* `async fn @f(...) -> T` actually returns a future type internally.
* Callers use `suspend.call` to await an **async function call**.
* Callers use `suspend.await { fut=... }` to await an **existing** `TFuture<T>` value.
* Non-async callers cannot call async functions directly — use `call std.async.@block_on { fn=@f, args=[...] }`.

---


#### 12.4.1 `suspend.await` (MUST)

`suspend.await` awaits an existing future value.

* Surface form: `%v: T = suspend.await { fut=%f }`
* Requirements:
  * MUST appear only inside an `async fn`.
  * `%f` MUST have type `TFuture<T>` (or `shared TFuture<T>`), and is **consumed** by the await.
* Semantics:
  * If the future is not ready, the current async state machine stores `%f` in its state and yields control to the executor.
  * When resumed, it re-polls `%f` until ready, then produces `%v` and drops the awaited future.

## 13. Concurrency

### 13.1 Threading model

Magpie exposes OS threads + channels:

* `std.thread.@spawn` — spawns an OS thread, takes a `TCallable` or function pointer
* `std.sync.TChannel<T>` — typed channel for message passing
* `std.sync.TMutex<T>` — mutex for shared mutable state
* `std.sync.TRwLock<T>` — reader-writer lock
* `std.sync.TCell<T>` — single-threaded interior mutability

### 13.2 Send/Sync enforcement

* `send` trait: type can be transferred across thread boundaries.
* `sync` trait: type can be shared (via `shared`) across threads.
* The type checker enforces `send`/`sync` at thread API boundaries:
  * `std.thread.@spawn` requires `TCallable<TSig>` where captured values are `send`.
  * `shared T` requires `T: sync` for cross-thread sharing.

### 13.3 Interior mutability types

* `TMutex<T>`, `TRwLock<T>`, `TCell<T>` are compiler-known types with special runtime support.
* They allow controlled mutation of `shared` references.
* `TMutex` and `TRwLock` are `sync` (safe for cross-thread sharing).
* `TCell` is NOT `sync` (single-threaded only).

#### 13.3.1 Interior mutability intrinsic signatures

| Intrinsic | Signature | Semantics |
|-----------|-----------|-----------|
| `@mutex_new<T>` | `(T) -> TMutex<T>` | Create mutex wrapping initial value |
| `@mutex_lock<T>` | `(borrow TMutex<T>) -> mutborrow T` | Acquire lock; returns scoped mutable borrow |
| `@mutex_unlock<T>` | `(borrow TMutex<T>) -> unit` | Release lock (compiler-inserted at borrow scope end) |
| `@rwlock_new<T>` | `(T) -> TRwLock<T>` | Create reader-writer lock |
| `@rwlock_read<T>` | `(borrow TRwLock<T>) -> borrow T` | Acquire read lock |
| `@rwlock_write<T>` | `(borrow TRwLock<T>) -> mutborrow T` | Acquire write lock |
| `@rwlock_unlock<T>` | `(borrow TRwLock<T>) -> unit` | Releases the held lock (read or write). The compiler MUST also insert an implicit unlock when the borrow/mutborrow returned from `@rwlock_read`/`@rwlock_write` ends (v0.1: end of basic block). |
| `@cell_get<T>` | `(borrow TCell<T>) -> T` | Copy value out (T must be Copy-like value type) |
| `@cell_set<T>` | `(borrow TCell<T>, T) -> unit` | Replace interior value |

`@mutex_lock` returns a `mutborrow T` with a scoped guard pattern: the compiler inserts `@mutex_unlock` at the end of the borrow's scope. This mirrors Rust's `MutexGuard` but is explicit in the IR.

#### 13.3.2 Runtime ABI for interior mutability

```c
MpRtHeader* mp_rt_mutex_new(uint32_t type_id, void* initial_value, uint64_t size);
void*       mp_rt_mutex_lock(MpRtHeader* mutex);   // returns pointer to payload
void        mp_rt_mutex_unlock(MpRtHeader* mutex);
MpRtHeader* mp_rt_rwlock_new(uint32_t type_id, void* initial_value, uint64_t size);
const void* mp_rt_rwlock_read(MpRtHeader* rwlock);
void*       mp_rt_rwlock_write(MpRtHeader* rwlock);
void        mp_rt_rwlock_unlock(MpRtHeader* rwlock);
void        mp_rt_cell_get(MpRtHeader* cell, void* out, uint64_t size);
void        mp_rt_cell_set(MpRtHeader* cell, const void* val, uint64_t size);
```

### 13.4 Typed channels (`TChannel<T>`)

Channels provide typed message passing between threads.

```
%pair: TChannelPair<i32> = call std.sync.@channel_new<i32> {}
%pair_b: borrow TChannelPair<i32> = borrow.shared { v=%pair }
%sender: borrow TChannelSend<i32> = getfield { obj=%pair_b, field=send }
%receiver: borrow TChannelRecv<i32> = getfield { obj=%pair_b, field=recv }
```

`TChannelPair<T>` is a heap struct: `heap struct TChannelPair<T: type> { field send: TChannelSend<T>, field recv: TChannelRecv<T> }`

| Intrinsic | Signature | Semantics |
|-----------|-----------|-----------|
| `@channel_new<T>` | `() -> TChannelPair<T>` | Create an unbounded MPSC channel pair |
| `@channel_send<T>` | `(borrow TChannelSend<T>, T) -> unit` | Send a value (moves T into channel) |
| `@channel_recv<T>` | `(borrow TChannelRecv<T>) -> TOption<T>` | Receive a value (blocks until available; returns None if sender dropped) |

* `TChannelSend<T>` is `send + sync` (clonable for multiple producers).
* `TChannelRecv<T>` is `send` but NOT `sync` (single consumer).
* `T` must be `send`.

Runtime ABI:
```c
MpRtHeader* mp_rt_channel_new(uint32_t elem_type_id, uint64_t elem_size);  // returns pair struct
void        mp_rt_channel_send(MpRtHeader* sender, const void* val, uint64_t elem_size);
int32_t     mp_rt_channel_recv(MpRtHeader* receiver, void* out, uint64_t elem_size); // 0=closed, 1=ok
```

---

## 14. TCallable and signature types

### 14.1 Signature declarations

```
sig THandlerSig(TRequest, TContext) -> TResponse
```

* Defines a named function signature type.
* Referenced as `TCallable<THandlerSig>`.
* Signatures appear in the `sig` namespace.

### 14.2 TCallable type

* `TCallable<TSig>` is an ARC-managed heap type wrapping a function pointer + optional captured environment.
* Supports vtable-dispatched calls.

#### 14.2.1 TCallable memory layout

```
TCallable<TSig> runtime layout:
┌─────────────────────────┐
│ MpRtHeader (32 bytes)   │  ← standard ARC header
├─────────────────────────┤
│ vtable_ptr: *const TCallableVtable │
│ data_ptr: *mut u8       │  ← pointer to captured environment (or null)
└─────────────────────────┘

TCallableVtable:
┌─────────────────────────┐
│ call_fn: fn_ptr          │  ← matches TSig; first implicit arg is data_ptr
│ drop_fn: fn_ptr          │  ← drops captured environment
│ size: u64                │  ← size of captured data (for debug/introspection)
└─────────────────────────┘
```

* `call_fn` signature: the TSig parameter types prepended with `*mut u8` (the data_ptr for captures).
* `drop_fn` signature: `fn(*mut u8) -> void` — releases any ARC-managed captures.
* When there are no captures, `data_ptr` is NULL and `drop_fn` is a no-op.

#### 14.2.2 TCallable + async restriction (v0.1)

**`suspend.call` on TCallable is forbidden in v0.1.** The compiler MUST reject `suspend.call %callable { ... }` where the callee is a TCallable with error `MPT1030 TCALLABLE_SUSPEND_FORBIDDEN`.

Rationale: TCallable uses vtable dispatch, and the async state machine generator cannot determine the coroutine frame layout through a vtable indirection. Async middleware MUST use a regular async function reference, not TCallable. This restriction may be lifted in v0.2 with runtime-typed coroutine frames.

### 14.3 Creating TCallable

User code can create TCallable via `callable.capture`:

```
%mul_fn: TCallable<TMulSig> = callable.capture @multiply_by { n=%n }
```

* Captured values are **moved** into the TCallable (consuming them).
* To keep using a value after capture, clone it first.
* The runtime can also create TCallable instances with opaque captures (e.g., for middleware chains).

### 14.4 Calling TCallable

```
%result: TReturn = call.indirect %my_callable { args=[%arg1, %arg2] }
```

TCallable values are callable via `call.indirect` (or `call_void.indirect`).

### 14.5 Functional intrinsics with TCallable

Array/Map functional operations accept `TCallable`:

```
%mul_fn: TCallable<TMapI32Sig> = callable.capture @multiply_by { factor=%n }
%result: Array<i32> = arr.map { arr=%a, fn=%mul_fn }
```

---

## 15. Core Magpie IR (MPIR) spec

### 15.1 Overview

MPIR is the compiler's canonical mid-level IR:
* SSA values
* Explicit basic blocks and terminators
* Ownership states attached to values
* ARC operations explicit after insertion pass
* Resolved SIDs and TypeIds

### 15.2 MPIR file format

* Extension: `.mpir`
* Encoding: UTF-8
* Newlines: `\n` only (CSNF normalized)

### 15.3 MPIR file header (MUST)

```
mpir.version 0.1
module pkg.sub.module
module_sid "M:XXXXXXXXXX"
module_digest "<blake3hex>"
target "<llvm-triple>"      ; optional
```

### 15.4 Sections and order (MUST)

1. header
2. `types { ... }`
3. `externs { ... }`
4. `globals { ... }`
5. `fns { ... }`

Omitted sections MUST still appear empty.

### 15.5 MPIR type table

```
types {
  type_id 4 = prim i32
  type_id 20 = heap_builtin Str
  type_id 1000 = heap_struct TPerson {
    field name : type_id 20
    field age  : type_id 4
  } layout { size=16 align=8 fields { name=0 age=8 } }
}
```

Type table entry forms (MUST):

* Primitives:
  * `type_id N = prim i32`
* Heap builtins (no type parameters):
  * `type_id N = heap_builtin Str`
  * `type_id N = heap_builtin StrBuilder`
* Heap builtins (with type parameters):
  * Arrays: `type_id N = heap_builtin Array { elem=type_id X }`
  * Maps: `type_id N = heap_builtin Map { key=type_id K, val=type_id V }`
  * Futures: `type_id N = heap_builtin TFuture { result=type_id X }`
  * Channels: `type_id N = heap_builtin ChannelSend { elem=type_id X }`, `type_id N = heap_builtin ChannelRecv { elem=type_id X }`
  * Interior mutability: `type_id N = heap_builtin Mutex { inner=type_id X }`, `RwLock { inner=type_id X }`, `Cell { inner=type_id X }`
  * Callables: `type_id N = heap_builtin TCallable { sig=@SigName, caps=[type_id A, ...] }`
* Builtin value enums (lang items):
  * `type_id N = builtin TOption { inner=type_id X }`
  * `type_id N = builtin TResult { ok=type_id A, err=type_id B }`
* Raw pointers:
  * `type_id N = rawptr { to=type_id X }`
* Handle qualifiers (for heap handle types):
  * `type_id N = shared { inner=type_id X }`
  * `type_id N = weak { inner=type_id X }`
* Borrow qualifiers:
  * `type_id N = borrow { inner=type_id X }`
  * `type_id N = mutborrow { inner=type_id X }`
* User types:
  * `heap_struct` / `heap_enum` entries as shown above. Generic instantiations MAY include `targs=[...]` for debugging.

Notes:
* `@SigName` is the fully-qualified signature name (a Sid) declared via `sig`.

Type IDs MUST be assigned deterministically **across the build graph**: fixed IDs for primitives/builtins, then user types by fully-qualified name (FQN) lexicographic order, then monomorphized instances by canonical type string.

### 15.6 MPIR functions

```
fns {
  fn @main sid "F:XXXXXXXXXX" sigdigest "<blake3hex>" inst_id "I:base" ( ) -> type_id 1
  meta { uses { ... } effects { ... } cost { ... } }
  {
  bb0:
    %msg : type_id 20 = const.type_id 20 "hello"
    call_void sid "F:YYYYYYYYYY" @std.io.@println { targs=[], args=[%msg] }
    ret const.type_id 4 0
  }
}
```

### 15.7 Required opcodes in MPIR v0.1

* `const.*`
* Integer ops: `i.add i.sub i.mul i.sdiv i.udiv i.srem i.urem i.add.wrap i.sub.wrap i.mul.wrap i.add.checked i.sub.checked i.mul.checked i.and i.or i.xor i.shl i.lshr i.ashr`
* Float ops: `f.add f.sub f.mul f.div f.rem f.add.fast f.sub.fast f.mul.fast f.div.fast`
* Compares: `icmp.* fcmp.*`
* Control flow: `br cbr switch ret unreachable phi`
* Calls: `call call_void call.indirect call_void.indirect try suspend.call suspend.await`
* Heap: `new getfield setfield`
* Enum: `enum.new enum.tag enum.payload enum.is`
* Ownership: `share clone.shared clone.weak weak.downgrade weak.upgrade`
* ARC: `arc.retain arc.release arc.retain_weak arc.release_weak`
* Unsafe pointers: `ptr.null ptr.addr ptr.from_addr ptr.add ptr.load ptr.store`
* Collections: `arr.* map.* str.*`
* JSON: `json.encode json.decode`
* GPU: `gpu.*` (required when a module uses `gpu` or when `--emit=spv` is requested)
* Callable: `callable.capture`
* Error: `panic`

### 15.8 MPIR verifier (MUST)

`magpie mpir verify` MUST check:
* SSA correctness (each name defined once, all uses dominated by defs)
* Type IDs exist in type table
* SIDs are valid format
* Call arity matches signature in symbol table
* ARC ops appear only after ARC insertion stage
* Phi nodes only for value types and Unique/Shared/Weak handles (not Borrow/MutBorrow)
* Block labels canonical ascending
* Each block ends with exactly one terminator

---

## 16. HIR specification (Rust data model)

### 16.1 Core ID types

```rust
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct PackageId(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ModuleId(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct DefId(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct TypeId(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct InstId(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct FnId(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct GlobalId(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct LocalId(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct BlockId(pub u32);
```

### 16.2 Primitive types and stable IDs

```rust
/// Primitive type enum — covers all surface-level primitive types.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum PrimType {
    I1, I8, I16, I32, I64, I128,
    U1, U8, U16, U32, U64, U128,
    F16, F32, F64,
    Bool, // alias for I1
    Unit,
}

/// Stable ID — a content-addressed identifier for modules, functions, types, globals.
/// Format: `<Kind>:<10 chars>` where Kind ∈ {M, F, T, G, E}.
/// The suffix is `base32_crockford(blake3(input))[0..10]`.
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct Sid(pub String);

impl Sid {
    /// Regex for validation: `^[MFTGE]:[0-9A-Z]{10}$`
    pub fn is_valid(&self) -> bool {
        // runtime check against the regex
        self.0.len() == 12
            && matches!(self.0.as_bytes()[0], b'M' | b'F' | b'T' | b'G' | b'E')
            && self.0.as_bytes()[1] == b':'
    }
}

/// A reference to an SSA value — either a local (SSA name) or an inline constant.
#[derive(Clone, Debug)]
pub enum HirValue {
    Local(LocalId),
    Const(HirConst),
}

/// A compile-time constant value.
#[derive(Clone, Debug)]
pub struct HirConst {
    pub ty: TypeId,
    pub lit: HirConstLit,
}

#[derive(Clone, Debug)]
pub enum HirConstLit {
    IntLit(i128),
    FloatLit(f64),
    BoolLit(bool),
    StringLit(String),
    Unit,
}
```

### 16.3 Type structure

```rust
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum HandleKind {
    Unique, Shared, Borrow, MutBorrow, Weak,
}

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub enum TypeKind {
    Prim(PrimType),

    // Heap-managed handles
    HeapHandle { hk: HandleKind, base: HeapBase },

    // Builtin value enums (lang items)
    BuiltinOption { inner: TypeId },
    BuiltinResult { ok: TypeId, err: TypeId },

    // Unsafe raw pointer type
    RawPtr { to: TypeId },

    // Aggregate value types (internal-only in v0.1 surface syntax)
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
```

### 16.4 HIR operations (v0.1 set)

> **Note on `Move`, `BorrowShared`, `BorrowMut`:** These are compiler-inserted during HIR lowering from the ownership checker, not surface opcodes. Surface code uses `borrow.shared` and `borrow.mut` explicitly to create borrows; the compiler then verifies and lowers them to the HIR variants below.

```rust
pub enum HirOp {
    Const(HirConst),

    // Ownership (compiler-inserted during HIR lowering)
    Move { v: HirValue },
    BorrowShared { v: HirValue },
    BorrowMut { v: HirValue },

    // Heap and fields
    New { ty: TypeId, fields: Vec<(String, HirValue)> },
    GetField { obj: HirValue, field: String },
    SetField { obj: HirValue, field: String, value: HirValue },

    // Integer arithmetic (checked — default, traps on overflow)
    IAdd { lhs: HirValue, rhs: HirValue },
    ISub { lhs: HirValue, rhs: HirValue },
    IMul { lhs: HirValue, rhs: HirValue },
    ISDiv { lhs: HirValue, rhs: HirValue },
    IUDiv { lhs: HirValue, rhs: HirValue },
    ISRem { lhs: HirValue, rhs: HirValue },
    IURem { lhs: HirValue, rhs: HirValue },

    // Integer arithmetic (wrapping — safe, no trap)
    IAddWrap { lhs: HirValue, rhs: HirValue },
    ISubWrap { lhs: HirValue, rhs: HirValue },
    IMulWrap { lhs: HirValue, rhs: HirValue },

    // Integer arithmetic (checked → TOption)
    IAddChecked { lhs: HirValue, rhs: HirValue },
    ISubChecked { lhs: HirValue, rhs: HirValue },
    IMulChecked { lhs: HirValue, rhs: HirValue },

    // Bitwise
    IAnd { lhs: HirValue, rhs: HirValue },
    IOr { lhs: HirValue, rhs: HirValue },
    IXor { lhs: HirValue, rhs: HirValue },
    IShl { lhs: HirValue, rhs: HirValue },
    ILshr { lhs: HirValue, rhs: HirValue },
    IAshr { lhs: HirValue, rhs: HirValue },

    // Compare
    ICmp { pred: String, lhs: HirValue, rhs: HirValue },
    FCmp { pred: String, lhs: HirValue, rhs: HirValue },

    // Float (strict IEEE 754)
    FAdd { lhs: HirValue, rhs: HirValue },
    FSub { lhs: HirValue, rhs: HirValue },
    FMul { lhs: HirValue, rhs: HirValue },
    FDiv { lhs: HirValue, rhs: HirValue },
    FRem { lhs: HirValue, rhs: HirValue },

    // Float (fast-math opt-in)
    FAddFast { lhs: HirValue, rhs: HirValue },
    FSubFast { lhs: HirValue, rhs: HirValue },
    FMulFast { lhs: HirValue, rhs: HirValue },
    FDivFast { lhs: HirValue, rhs: HirValue },

    // Cast
    Cast { to: TypeId, v: HirValue },

    // Unsafe raw pointer ops (surface: `ptr.*`, restricted to `unsafe {}`)
    PtrNull { to: TypeId },
    PtrAddr { p: HirValue },
    PtrFromAddr { to: TypeId, addr: HirValue },
    PtrAdd { p: HirValue, count: HirValue },
    PtrLoad { to: TypeId, p: HirValue },
    PtrStore { to: TypeId, p: HirValue, v: HirValue },

    // Calls
    Call { callee_sid: Sid, inst: Vec<TypeId>, args: Vec<HirValue> },
    CallIndirect { callee: HirValue, args: Vec<HirValue> },
    CallVoidIndirect { callee: HirValue, args: Vec<HirValue> },
    SuspendCall { callee_sid: Sid, inst: Vec<TypeId>, args: Vec<HirValue> },
    SuspendAwait { fut: HirValue },

    // Control flow
    Phi { ty: TypeId, incomings: Vec<(BlockId, HirValue)> },

    // Ownership conversions
    Share { v: HirValue },
    CloneShared { v: HirValue },
    CloneWeak { v: HirValue },
    WeakDowngrade { v: HirValue },
    WeakUpgrade { v: HirValue },           // returns TOption<shared T>

    // Enum operations
    EnumNew { variant: String, args: Vec<(String, HirValue)> },
    EnumTag { v: HirValue },           // always returns i32 (§16.5)
    EnumPayload { variant: String, v: HirValue },
    EnumIs { variant: String, v: HirValue },

    // TCallable
    CallableCapture { fn_ref: Sid, captures: Vec<(String, HirValue)> },

    // Array intrinsics
    ArrNew { elem_ty: TypeId, cap: HirValue },
    ArrLen { arr: HirValue },
    ArrGet { arr: HirValue, idx: HirValue },
    ArrSet { arr: HirValue, idx: HirValue, val: HirValue },
    ArrPush { arr: HirValue, val: HirValue },
    ArrPop { arr: HirValue },
    ArrSlice { arr: HirValue, start: HirValue, end: HirValue },
    ArrContains { arr: HirValue, val: HirValue },
    ArrSort { arr: HirValue },
    ArrMap { arr: HirValue, func: HirValue },
    ArrFilter { arr: HirValue, func: HirValue },
    ArrReduce { arr: HirValue, init: HirValue, func: HirValue },
    ArrForeach { arr: HirValue, func: HirValue },

    // Map intrinsics
    MapNew { key_ty: TypeId, val_ty: TypeId },
    MapLen { map: HirValue },
    MapGet { map: HirValue, key: HirValue },
    MapGetRef { map: HirValue, key: HirValue },
    MapGetRef { map: HirValue, key: HirValue },
    MapSet { map: HirValue, key: HirValue, val: HirValue },
    MapDelete { map: HirValue, key: HirValue },
    MapContainsKey { map: HirValue, key: HirValue },
    MapDeleteVoid { map: HirValue, key: HirValue },
    MapKeys { map: HirValue },
    MapValues { map: HirValue },

    // String intrinsics
    StrConcat { a: HirValue, b: HirValue },
    StrLen { s: HirValue },
    StrEq { a: HirValue, b: HirValue },
    StrSlice { s: HirValue, start: HirValue, end: HirValue },
    StrBytes { s: HirValue },
    StrBuilderNew,
    StrBuilderAppendStr { b: HirValue, s: HirValue },
    StrBuilderAppendI64 { b: HirValue, v: HirValue },
    StrBuilderAppendI32 { b: HirValue, v: HirValue },
    StrBuilderAppendF64 { b: HirValue, v: HirValue },
    StrBuilderAppendBool { b: HirValue, v: HirValue },
    StrBuilderBuild { b: HirValue },

    // Error
    Panic { msg: HirValue },
}
```

Surface borrow opcodes (parsed from `.mp`, lowered to `BorrowShared`/`BorrowMut` above):

```
%ref: borrow TPerson = borrow.shared { v=%person }
%mref: mutborrow TPerson = borrow.mut { v=%person }
```

### 16.5 `enum.tag` return type (P1-10)

* `EnumTag` / `enum.tag` ALWAYS returns `i32`, regardless of the number of variants.
* Tag values are assigned sequentially starting from 0 in declaration order.
* For `TOption<T>` with heap handles: niche optimization applies at the representation level (NULL = None), but `enum.tag` still returns `i32` (0=None, 1=Some). The niche representation is an LLVM codegen optimization, not visible at HIR/MPIR level.
* For `TResult<T,E>`: tag 0=Ok, tag 1=Err.

### 16.6 HIR invariants (MUST)

1. **SSA well-formed**: each LocalId defined exactly once, all uses dominated by defs, each block ends with exactly one terminator.
2. **Borrow locality**: borrows MUST NOT appear in phi, MUST NOT cross blocks, last use before terminator.
3. **Field access requires explicit borrow**: GetField requires borrow/mutborrow, SetField requires mutborrow.
4. **No returning/storing borrows**: return type MUST NOT be borrow/mutborrow, borrows MUST NOT be stored into heap.

---

## 17. Typed MPIR (Rust data model)

MPIR adds to HIR:
* Explicit ARC ops after ARC pass
* Verified type tables and symbol table snapshots
* Field overwrite expansions
* Resolved SIDs for all symbols

### 17.1 MPIR Rust data structures

```rust
/// A single MPIR value reference — extends HirValue with resolved SIDs.
#[derive(Clone, Debug)]
pub enum MpirValue {
    Local(LocalId),
    Const(HirConst),
}

/// MPIR instruction — an SSA assignment.
#[derive(Clone, Debug)]
pub struct MpirInstr {
    pub dst: LocalId,
    pub ty: TypeId,
    pub op: MpirOp,
}

/// MPIR void operation (no result value).
#[derive(Clone, Debug)]
pub enum MpirOpVoid {
    CallVoid { callee_sid: Sid, inst: Vec<TypeId>, args: Vec<MpirValue> },
    CallVoidIndirect { callee: MpirValue, args: Vec<MpirValue> },
    SetField { obj: MpirValue, field: String, val: MpirValue },
    ArrPush { arr: MpirValue, val: MpirValue },
    ArrForeach { arr: MpirValue, func: MpirValue },
    MapDeleteVoid { map: MpirValue, key: MpirValue },
    PtrStore { to: TypeId, p: MpirValue, v: MpirValue },
    Panic { msg: MpirValue },
    // ARC operations (inserted by ARC pass)
    ArcRetain { v: MpirValue },
    ArcRelease { v: MpirValue },
    ArcRetainWeak { v: MpirValue },
    ArcReleaseWeak { v: MpirValue },
}

/// MPIR operation — extends HirOp with ARC ops and resolved SIDs.
#[derive(Clone, Debug)]
pub enum MpirOp {
    // All HirOp variants carry over with MpirValue instead of HirValue.
    // Only MPIR-specific additions listed here:

    // ARC operations (not present in HIR, inserted by ARC insertion pass)
    ArcRetain { v: MpirValue },
    ArcRelease { v: MpirValue },
    ArcRetainWeak { v: MpirValue },
    ArcReleaseWeak { v: MpirValue },
    WeakUpgrade { v: MpirValue },   // returns TOption<shared T>

    // All other ops mirror HirOp with MpirValue substituted for HirValue
    // Every HirOp variant (§16.4) has a 1:1 MpirOp counterpart with
    // HirValue replaced by MpirValue. The canonical list is:
    // Const, Move, BorrowShared, BorrowMut, New, GetField, SetField,
    // IAdd, ISub, IMul, ISDiv, IUDiv, ISRem, IURem,
    // IAddWrap, ISubWrap, IMulWrap, IAddChecked, ISubChecked, IMulChecked,
    // IAnd, IOr, IXor, IShl, ILshr, IAshr, ICmp, FCmp,
    // FAdd, FSub, FMul, FDiv, FRem, FAddFast, FSubFast, FMulFast, FDivFast,
    // Cast, PtrNull, PtrAddr, PtrFromAddr, PtrAdd, PtrLoad, PtrStore,
    // Call, CallIndirect, SuspendCall, SuspendAwait, Phi,
    // Share, CloneShared, CloneWeak, WeakDowngrade, WeakUpgrade,
    // EnumNew, EnumTag, EnumPayload, EnumIs, CallableCapture,
    // ArrNew, ArrLen, ArrGet, ArrSet, ArrPush, ArrPop, ArrSlice,
    // ArrContains, ArrSort, ArrMap, ArrFilter, ArrReduce, ArrForeach,
    // MapNew, MapLen, MapGet, MapSet, MapDelete, MapContainsKey, MapDeleteVoid,
    // MapKeys, MapValues,
    // StrConcat, StrLen, StrEq, StrSlice, StrBytes,
    // StrBuilderNew, StrBuilderAppendStr, StrBuilderAppendI64,
    // StrBuilderAppendI32, StrBuilderAppendF64, StrBuilderAppendBool,
    // StrBuilderBuild, Panic
}

/// MPIR block terminator.
#[derive(Clone, Debug)]
pub enum MpirTerminator {
    Ret(Option<MpirValue>),
    Br(BlockId),
    Cbr { cond: MpirValue, then_bb: BlockId, else_bb: BlockId },
    Switch { val: MpirValue, arms: Vec<(HirConst, BlockId)>, default: BlockId },
    Unreachable,
}

/// MPIR basic block.
#[derive(Clone, Debug)]
pub struct MpirBlock {
    pub id: BlockId,
    pub instrs: Vec<MpirInstr>,
    pub void_ops: Vec<MpirOpVoid>,
    pub terminator: MpirTerminator,
}

/// MPIR function.
#[derive(Clone, Debug)]
pub struct MpirFn {
    pub sid: Sid,
    pub name: String,
    pub params: Vec<(LocalId, TypeId)>,
    pub ret_ty: TypeId,
    pub blocks: Vec<MpirBlock>,
    pub locals: Vec<MpirLocalDecl>,
    pub is_async: bool,
}

/// MPIR local declaration.
#[derive(Clone, Debug)]
pub struct MpirLocalDecl {
    pub id: LocalId,
    pub ty: TypeId,
    pub name: String,   // debug name from SSA (e.g., "msg")
}

/// MPIR module — the unit of compilation output.
#[derive(Clone, Debug)]
pub struct MpirModule {
    pub sid: Sid,
    pub path: String,
    pub type_table: MpirTypeTable,
    pub functions: Vec<MpirFn>,
    pub globals: Vec<(GlobalId, TypeId, HirConst)>,
}

/// MPIR type table snapshot.
#[derive(Clone, Debug)]
pub struct MpirTypeTable {
    pub types: Vec<(TypeId, TypeKind)>,
}
```

### 17.2 MPIR invariants

MPIR invariants include all HIR invariants (§16.6) plus:
* ARC ops only after ARC insertion pass
* TypeRef correctness — all TypeIds resolve in the type table
* Phi only for value types and Unique/Shared/Weak handles

---

## 18. Canonical strings, stable IDs, and digests

### 18.1 Canonical module path string

* `ModulePathStr = "pkg.sub.mod"` (dotted, no whitespace)

### 18.2 Canonical FQN strings

* Function: `ModulePathStr + ".@" + Ident`
* Type: `ModulePathStr + ".T" + Ident`
* Global: `ModulePathStr + ".@" + Ident`

### 18.3 Canonical type strings (TypeStr)

Complete grammar and key rules:
* Ownership modifiers followed by exactly one space
* No extra whitespace
* Type args without spaces after commas

### 18.4 Stable IDs (SIDs)

Format: `<Kind>:<10 chars>` where Kind is M/F/T/G/E.

Suffix: `base32_crockford(blake3(input))[0..10]`

Input strings (all prefixed with `"magpie:sid:v0.1|"`):
* Module: `"magpie:sid:v0.1|module|" + ModulePathStr`
* Function: `"magpie:sid:v0.1|fn|" + FqnFnStr`
* Type: `"magpie:sid:v0.1|type|" + FqnTypeStr`
* Global: `"magpie:sid:v0.1|global|" + FqnGlobalStr`

### 18.5 Signature core string (`SigCoreStr`)

`SigCoreStr(fn)` is the canonical string representation of a function signature used for digest computation:

```
SigCoreStr(fn) = "fn " + FqnFnStr + "(" + Join(",", ParamTypeStrs) + ") -> " + RetTypeStr
```

Where:
* `FqnFnStr` is the fully qualified function name (e.g., `mymod.@my_func`)
* `ParamTypeStrs` is the ordered list of canonical type strings (§18.3) for each parameter
* `RetTypeStr` is the canonical type string for the return type
* `Join(",", list)` concatenates elements with `,` separator (no spaces)

Example: `fn mymod.@add(i32,i32) -> i32`

### 18.6 Signature digests

`SigDigestHex = blake3_hex("magpie:sigdigest:v0.1|" + SigCoreStr(fn))`

### 18.7 Monomorphized instance IDs

Format: `I:<16 chars>`

Input: `"magpie:inst:v0.1|" + SymbolSID + "|" + Join("|", TypeArgTypeStrs)`

---

## 19. LLVM symbol mangling (exact, deterministic)

### 19.1 Global conventions

* All Magpie-generated LLVM symbols start with: `mp$0$` (ABI version 0)
* `$` is the delimiter

### 19.2 Function symbols

* Non-generic: `mp$0$FN$<F_sid_suffix>`
* Monomorphized: `mp$0$FN$<F_sid_suffix>$I$<inst_suffix16>`

### 19.3 Other symbols

* Globals: `mp$0$GL$<G_sid_suffix>`
* Type info: `mp$0$TI$<T_sid_suffix>`
* Drop function: `mp$0$DROP$<T_sid_suffix>`
* Type init: `mp$0$INIT_TYPES$<M_sid_suffix>`

### 19.4 Program entrypoint

C `main` MUST: call `mp_rt_init()`, call all `INIT_TYPES` in deterministic order, call Magpie `@main`, return exit code.

---

## 20. Runtime ABI

### 20.1 64-bit ABI (primary, LP64)

#### 20.1.1 Object header (C layout)

```c
typedef struct MpRtHeader {
  _Atomic uint64_t strong;
  _Atomic uint64_t weak;
  uint32_t type_id;
  uint32_t flags;
  uint64_t reserved0;
} MpRtHeader;
// sizeof == 32, alignof >= 8
// Payload at byte offset 32
```

#### 20.1.2 Required runtime functions

```c
void mp_rt_init(void);
void mp_rt_register_types(const MpRtTypeInfo* infos, uint32_t count);
const MpRtTypeInfo* mp_rt_type_info(uint32_t type_id);
MpRtHeader* mp_rt_alloc(uint32_t type_id, uint64_t payload_size, uint64_t payload_align, uint32_t flags);
void mp_rt_retain_strong(MpRtHeader* obj);
void mp_rt_release_strong(MpRtHeader* obj);
void mp_rt_retain_weak(MpRtHeader* obj);
void mp_rt_release_weak(MpRtHeader* obj);
MpRtHeader* mp_rt_weak_upgrade(MpRtHeader* obj);
MpRtHeader* mp_rt_str_from_utf8(const uint8_t* bytes, uint64_t len);
const uint8_t* mp_rt_str_bytes(MpRtHeader* str, uint64_t* out_len);
void mp_rt_panic(MpRtHeader* str_msg) __attribute__((noreturn));
```

For parse/json boundary hardening, the runtime MUST also expose fallible status-returning APIs:

```c
#define MP_RT_OK 0
#define MP_RT_ERR_INVALID_UTF8 1
#define MP_RT_ERR_INVALID_FORMAT 2
#define MP_RT_ERR_UNSUPPORTED_TYPE 3
#define MP_RT_ERR_NULL_OUT_PTR 4
#define MP_RT_ERR_NULL_INPUT 5

int32_t mp_rt_str_try_parse_i64(MpRtHeader* s, int64_t* out, MpRtHeader** out_errmsg);
int32_t mp_rt_str_try_parse_u64(MpRtHeader* s, uint64_t* out, MpRtHeader** out_errmsg);
int32_t mp_rt_str_try_parse_f64(MpRtHeader* s, double* out, MpRtHeader** out_errmsg);
int32_t mp_rt_str_try_parse_bool(MpRtHeader* s, int32_t* out, MpRtHeader** out_errmsg);
int32_t mp_rt_json_try_encode(uint8_t* obj, uint32_t type_id, MpRtHeader** out_str, MpRtHeader** out_errmsg);
int32_t mp_rt_json_try_decode(MpRtHeader* json_str, uint32_t type_id, uint8_t** out_val, MpRtHeader** out_errmsg);
```

Error ownership contract:

* On success: `status == MP_RT_OK` and `*out_errmsg == NULL` (when `out_errmsg` is non-NULL).
* On error: runtime MAY allocate an error `Str` and store it in `*out_errmsg`; caller owns it and MUST release via `mp_rt_release_strong`.
* If `out_errmsg == NULL`, runtime MUST still return status and drop any temporary error string internally.
* Legacy panic-oriented entry points (`mp_rt_str_parse_*`, `mp_rt_json_encode`, `mp_rt_json_decode`) remain as compatibility wrappers over `*_try_*`.

#### 20.1.3 `MpRtTypeInfo` struct

```c
typedef void (*MpRtDropFn)(MpRtHeader* obj);

typedef struct MpRtTypeInfo {
    uint32_t    type_id;
    uint32_t    flags;          // bitfield: 0x1=heap, 0x2=has_drop, 0x4=send, 0x8=sync
    uint64_t    payload_size;
    uint64_t    payload_align;
    MpRtDropFn  drop_fn;        // NULL if no custom drop (see rules below)
    const char* debug_fqn;      // e.g. "mymod.TPerson", for diagnostics only
} MpRtTypeInfo;
```

**`drop_fn` rules:**
* `drop_fn` is NULL for: primitive types, value structs with no heap fields, and builtin types whose drop is handled entirely by the runtime (Str, Array, Map, TStrBuilder).
* `drop_fn` is non-NULL for: user-defined heap structs/enums whose fields include other heap handles (the compiler generates a drop function that releases each heap field).
* Value structs do NOT get `MpRtTypeInfo` entries — they have no runtime header and no ARC lifecycle.
* Drop function deduplication (§11.6): if two types have structurally identical drop functions, LLVM ICF or compiler-level dedup may merge them. The `type_id` in the header still distinguishes the types; only the function pointer is shared.

#### 20.1.4 Fixed type_id table for primitives and builtins

Primitive and builtin type_ids are deterministic and fixed across all compilations:

| type_id | Type | Notes |
|---------|------|-------|
| 0 | `unit` | zero-size |
| 1 | `bool` / `i1` | |
| 2 | `i8` | |
| 3 | `i16` | |
| 4 | `i32` | |
| 5 | `i64` | |
| 6 | `i128` | |
| 7 | `u8` | |
| 8 | `u16` | |
| 9 | `u32` | |
| 10 | `u64` | |
| 11 | `u128` | |
| 12 | `u1` | |
| 13 | `f16` | |
| 14 | `f32` | |
| 15 | `f64` | |
| 20 | `Str` | heap builtin |
| 21 | `TStrBuilder` | heap builtin |
| 22 | `Array<?>` | base; instantiations get unique IDs ≥ 1000 |
| 23 | `Map<?,?>` | base; instantiations get unique IDs ≥ 1000 |
| 24 | `TOption<?>` | base; instantiations get unique IDs ≥ 1000 |
| 25 | `TResult<?,?>` | base; instantiations get unique IDs ≥ 1000 |
| 26 | `TCallable<?>` | base; instantiations get unique IDs ≥ 1000 |
| 30 | `gpu.TDevice` | heap builtin |
| 31 | `gpu.TBuffer<?>` | base; instantiations get unique IDs ≥ 1000 |
| 32 | `gpu.TFence` | heap builtin |

All non-fixed types (user-defined heap types, generic instantiations, and ownership-qualified derived types) receive `type_id`s starting at 1000, assigned deterministically across the entire build graph by lexicographic order of their canonical type key (see §15.5).

#### 20.1.5 Collection runtime ABI

All `arr.*`, `map.*`, `str.*` intrinsics lower to these C functions:

```c
typedef uint64_t (*MpRtHashFn)(const void* key_bytes);
typedef int32_t  (*MpRtEqFn)(const void* a_bytes, const void* b_bytes);
typedef int32_t  (*MpRtCmpFn)(const void* a_bytes, const void* b_bytes);

// Array operations
MpRtHeader* mp_rt_arr_new(uint32_t elem_type_id, uint64_t elem_size, uint64_t capacity);
uint64_t    mp_rt_arr_len(MpRtHeader* arr);
void*       mp_rt_arr_get(MpRtHeader* arr, uint64_t idx);          // returns ptr to element; panics on OOB
void        mp_rt_arr_set(MpRtHeader* arr, uint64_t idx, const void* val, uint64_t elem_size);
void        mp_rt_arr_push(MpRtHeader* arr, const void* val, uint64_t elem_size);
int32_t     mp_rt_arr_pop(MpRtHeader* arr, void* out, uint64_t elem_size); // returns 0=empty, 1=ok
MpRtHeader* mp_rt_arr_slice(MpRtHeader* arr, uint64_t start, uint64_t end);
int32_t     mp_rt_arr_contains(MpRtHeader* arr, const void* val, uint64_t elem_size, MpRtEqFn eq_fn); // 0=no, 1=yes
void        mp_rt_arr_sort(MpRtHeader* arr, MpRtCmpFn cmp);
void        mp_rt_arr_foreach(MpRtHeader* arr, MpRtHeader* callable);
MpRtHeader* mp_rt_arr_map(MpRtHeader* arr, MpRtHeader* callable, uint32_t result_elem_type_id, uint64_t result_elem_size);
MpRtHeader* mp_rt_arr_filter(MpRtHeader* arr, MpRtHeader* callable);
void        mp_rt_arr_reduce(MpRtHeader* arr, void* acc_inout, uint64_t acc_size, MpRtHeader* callable);

// Map operations
MpRtHeader* mp_rt_map_new(
    uint32_t key_type_id,
    uint32_t val_type_id,
    uint64_t key_size,
    uint64_t val_size,
    uint64_t capacity,
    MpRtHashFn hash_fn,
    MpRtEqFn eq_fn
);
uint64_t    mp_rt_map_len(MpRtHeader* map);
void*       mp_rt_map_get(MpRtHeader* map, const void* key, uint64_t key_size);   // returns ptr or NULL
void        mp_rt_map_set(MpRtHeader* map, const void* key, uint64_t key_size, const void* val, uint64_t val_size);
int32_t     mp_rt_map_take(MpRtHeader* map, const void* key, uint64_t key_size, void* out_val, uint64_t val_size); // 0=not_found, 1=ok (moves out; does NOT drop)
int32_t     mp_rt_map_delete(MpRtHeader* map, const void* key, uint64_t key_size); // 0=not_found, 1=deleted (drops removed value)
int32_t     mp_rt_map_contains_key(MpRtHeader* map, const void* key, uint64_t key_size);
MpRtHeader* mp_rt_map_keys(MpRtHeader* map);     // returns Array of keys
MpRtHeader* mp_rt_map_values(MpRtHeader* map);   // returns Array of values

// String operations
MpRtHeader* mp_rt_str_concat(MpRtHeader* a, MpRtHeader* b);
uint64_t    mp_rt_str_len(MpRtHeader* s);
int32_t     mp_rt_str_eq(MpRtHeader* a, MpRtHeader* b);
MpRtHeader* mp_rt_str_slice(MpRtHeader* s, uint64_t start, uint64_t end);
// mp_rt_str_bytes and mp_rt_str_from_utf8 defined in §20.1.2

// StringBuilder operations
MpRtHeader* mp_rt_strbuilder_new(void);
void        mp_rt_strbuilder_append_str(MpRtHeader* b, MpRtHeader* s);
void        mp_rt_strbuilder_append_i64(MpRtHeader* b, int64_t v);
void        mp_rt_strbuilder_append_i32(MpRtHeader* b, int32_t v);
void        mp_rt_strbuilder_append_f64(MpRtHeader* b, double v);
void        mp_rt_strbuilder_append_bool(MpRtHeader* b, int32_t v);
MpRtHeader* mp_rt_strbuilder_build(MpRtHeader* b);   // consumes builder, returns Str

// Async executor
int32_t     mp_rt_future_poll(MpRtHeader* state);     // 0=Pending, 1=Ready
void        mp_rt_future_take(MpRtHeader* state, void* out_result);
```

Notes (async executor):

* `mp_rt_future_poll` is called by the executor to drive a `TFuture<T>` state machine.
* When `mp_rt_future_poll` reports Ready, the executor (or `std.async.block_on`) MAY call `mp_rt_future_take` to move the completed result value into `out_result` (a caller-allocated buffer). Calling `mp_rt_future_take` before Ready is undefined behavior.

#### 20.1.6 Web runtime ABI

The web runtime provides an HTTP/1.1 server loop and bridges requests/responses to Magpie via fixed
callback symbols (see §30.1.7).

```c
// Starts a blocking HTTP server.
//
// Returns 0 on clean shutdown.
// Returns non-zero on startup/fatal error and sets *out_errmsg to an owned Str.
//
// The runtime MUST invoke the callback symbols:
//   __magpie_web_handle_request
//   __magpie_web_stream_next
//
// Ownership:
//   On error, out_errmsg is an owned Str and must be released by the caller.
//   On success, *out_errmsg MUST be NULL.
int32_t mp_rt_web_serve(
  MpRtHeader* svc,              // web.router.TService
  MpRtHeader* addr,             // Str (bind address, e.g. "127.0.0.1")
  uint16_t    port,             // bind port
  uint8_t     keep_alive,       // 0/1
  uint32_t    threads,          // worker thread count (>=1)
  uint64_t    max_body_bytes,   // request body limit
  uint64_t    read_timeout_ms,
  uint64_t    write_timeout_ms,
  uint8_t     log_requests,     // 0/1
  MpRtHeader** out_errmsg       // Str* (nullable)
);
```

Request parsing rules (MUST):
* The runtime MUST parse:
  * method (uppercase)
  * path (no query string)
  * query string into `Map<Str,Str>` (first value wins)
  * headers into `Map<Str,Str>` with lowercase names (first value wins)
  * body into `Array<u8>` (buffered; may be empty)
* The runtime MUST generate a `request_id: Str` that is unique within the process lifetime.
* The runtime MUST call `__magpie_web_handle_request(...)` once per request.
* The runtime MUST write all chunks returned by `__magpie_web_stream_next` in order.
* The runtime MAY close the connection early on I/O error.

Static assets (MUST in Magpie Web v0.1):
* `magpie web dev` MUST serve `/assets/*` from `app/assets` (no caching).
* `magpie web serve` MUST serve `/assets/*` from `dist/assets` (caching enabled).
The asset handler lives in the runtime and does not require Magpie file I/O in v0.1.

#### 20.1.7 GPU runtime ABI

The GPU runtime is responsible for device discovery, buffer management, kernel registry, and dispatch.

```c
// --- Kernel registry ---

typedef enum MpRtGpuBackend {
  MP_GPU_BACKEND_SPV = 1,   // Vulkan SPIR-V (required for Magpie GPU v0.1)
} MpRtGpuBackend;

typedef enum MpRtGpuParamKind {
  MP_GPU_PARAM_BUFFER = 1,  // gpu.TBuffer<T>
  MP_GPU_PARAM_SCALAR = 2,  // primitive scalar packed into push constants
} MpRtGpuParamKind;

typedef struct MpRtGpuParam {
  uint8_t  kind;            // MpRtGpuParamKind
  uint8_t  _reserved0;
  uint16_t _reserved1;
  uint32_t type_id;         // primitive TypeId for scalars; 0 for buffers
  uint32_t offset_or_binding; // scalar: byte offset in push constants; buffer: binding index
  uint32_t size;            // scalar: size in bytes; buffer: 0
} MpRtGpuParam;

typedef struct MpRtGpuKernelEntry {
  uint64_t sid_hash;        // hash of kernel SID string
  uint32_t backend;         // MpRtGpuBackend
  const uint8_t* blob;
  uint64_t blob_len;
  uint32_t num_params;
  const MpRtGpuParam* params;
  uint32_t num_buffers;     // convenience: count of buffer params
  uint32_t push_const_size; // size of scalar block (bytes; multiple of 16)
} MpRtGpuKernelEntry;

void mp_rt_gpu_register_kernels(const MpRtGpuKernelEntry* entries, uint32_t count);

// --- Device discovery ---

uint32_t mp_rt_gpu_device_count(void);

int32_t mp_rt_gpu_device_default(MpRtHeader** out_dev, MpRtHeader** out_errmsg);     // Str errmsg
int32_t mp_rt_gpu_device_by_index(uint32_t idx, MpRtHeader** out_dev, MpRtHeader** out_errmsg);

MpRtHeader* mp_rt_gpu_device_name(MpRtHeader* dev); // returns owned Str

// --- Buffers ---

int32_t mp_rt_gpu_buffer_new(
  MpRtHeader* dev,
  uint32_t elem_type_id,
  uint64_t elem_size,
  uint64_t len,
  uint32_t usage_flags,
  MpRtHeader** out_buf,
  MpRtHeader** out_errmsg
);

int32_t mp_rt_gpu_buffer_from_array(
  MpRtHeader* dev,
  MpRtHeader* host_arr,     // Array<T>
  uint32_t usage_flags,
  MpRtHeader** out_buf,
  MpRtHeader** out_errmsg
);

int32_t mp_rt_gpu_buffer_to_array(
  MpRtHeader* buf,
  MpRtHeader** out_arr,     // Array<T>
  MpRtHeader** out_errmsg
);

uint64_t mp_rt_gpu_buffer_len(MpRtHeader* buf);

int32_t mp_rt_gpu_buffer_copy(MpRtHeader* src, MpRtHeader* dst, MpRtHeader** out_errmsg);

int32_t mp_rt_gpu_device_sync(MpRtHeader* dev, MpRtHeader** out_errmsg);

// --- Launch ---

// args_blob layout (MUST):
//   [8 * num_buffers bytes]  buffer pointers (MpRtHeader*) in binding order
//   [push_const_size bytes]  scalar push-constant bytes
//
// The compiler MUST pass args_len = 8*num_buffers + push_const_size.
int32_t mp_rt_gpu_launch_sync(
  MpRtHeader* dev,
  uint64_t kernel_sid_hash,
  uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
  uint32_t block_x, uint32_t block_y, uint32_t block_z,
  const uint8_t* args_blob,
  uint64_t args_len,
  MpRtHeader** out_errmsg
);

// Optional async launch/fence.
int32_t mp_rt_gpu_launch_async(
  MpRtHeader* dev,
  uint64_t kernel_sid_hash,
  uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
  uint32_t block_x, uint32_t block_y, uint32_t block_z,
  const uint8_t* args_blob,
  uint64_t args_len,
  MpRtHeader** out_fence,
  MpRtHeader** out_errmsg
);

int32_t mp_rt_gpu_fence_wait(
  MpRtHeader* fence,
  uint64_t timeout_ms,
  uint8_t* out_done,        // 0/1
  MpRtHeader** out_errmsg
);
```

The compiler lowers `gpu.host.*` intrinsics (§31.2) and `gpu.launch*` (§31.6) to these runtime calls.
### 20.2 wasm32 ABI variant

For WASM targets (wasm32-unknown-unknown):

```c
typedef struct MpRtHeader32 {
  _Atomic uint32_t strong;
  _Atomic uint32_t weak;
  uint32_t type_id;
  uint32_t flags;
} MpRtHeader32;
// sizeof == 16, alignof >= 4
// Payload at byte offset 16
```

* Pointers are 4 bytes.
* All runtime functions have the same signatures but with 32-bit pointer arguments.
* The compiler selects the ABI variant based on target triple.

### 20.3 Refcount semantics

* Strong retain: increment `strong` by 1.
* Strong release: decrement `strong`; if 0, call drop, then release weak once (implicit weak).
* Weak retain: increment `weak`.
* Weak release: decrement `weak`; if 0, free memory.
* Weak upgrade: atomically check strong > 0, increment, return ptr or NULL.
* Memory ordering: increments may use relaxed; decrements that hit zero use release + acquire.

---

## 21. LLVM IR lowering

### 21.1 Type mapping

* Value types → LLVM integer/float types
* Heap handles → `ptr` (opaque pointer)
* Value structs → LLVM struct types

### 21.2 Calls ABI

* Parameters: value types by value, heap handles as `ptr`
* Returns: value types by value, heap handles as `ptr`

### 21.3 ARC lowering

`arc.retain/release` lower to runtime calls: `@mp_rt_retain_strong(ptr)`, `@mp_rt_release_strong(ptr)`

### 21.4 Checked arithmetic lowering

`i.add` lowers to LLVM intrinsic `llvm.sadd.with.overflow.i32` (or equivalent), with branch to panic on overflow.

### 21.5 Collection intrinsic lowering

All `arr.*`, `map.*`, `str.*` intrinsics lower to runtime function calls (e.g., `@mp_rt_arr_push`, `@mp_rt_map_get`, etc.).

---

## 22. Compilation pipeline

### 22.1 Stages

1. **Parse + CSNF**: raw `.mp` → CSNF source + `AstFile` + file digest
2. **Resolve**: AST modules + import/export headers → `HirPackage`
3. **Typecheck**: HIR → typed HIR (TypeIds, layouts)
3.5. **Async lowering**: async functions → coroutine state machines (see below)
4. **Verify HIR**: SSA and borrow-locality invariants
5. **Ownership check**: typed HIR → ownership proof traces or errors
6. **Lower to MPIR**: typed HIR → MPIR (no ARC ops yet)
7. **MPIR verify**: SSA + type refs + phi restrictions
8. **ARC insertion**: MPIR → MPIR with ARC ops + field-overwrite expansions
9. **ARC optimization**: eliminate redundant retain/release
10. **LLVM codegen**: MPIR + type layouts → LLVM module
11. **Link**: output final exe/shared-lib
12. **MMS update**: update capsules, signatures, repair episodes

#### Stage 3.5: Async lowering (detail)

For each `async fn`, the compiler:

1. **Generates a coroutine state struct** (heap-allocated, ARC-managed):
   * Contains all SSA values live across any `suspend.call` point.
   * Contains a `state_index: i32` field tracking which resume point to jump to.
   * Has a compiler-generated `type_id` (registered via `mp_rt_register_types`).

2. **Rewrites the function body** into a resume function:
   * Entry dispatches on `state_index` to the correct resume point.
   * Each `suspend.call` saves live values into the state struct and returns `Pending`.
   * On final completion, the result is written into the state struct and returns `Ready`.

3. **Resume function ABI**:
   ```c
   // Returns: 0 = Pending (suspended), 1 = Ready (completed)
   int32_t mp_rt_future_poll(MpRtHeader* state);
   ```
   The result value is read from the state struct payload after `Ready` is returned.

4. **Caller-side**: `suspend.call` lowers to creating the state struct, then yielding to the executor which polls via `mp_rt_future_poll`.

**Async lowering MPIR example** — given:
```
async fn @fetch_user(%id: u64) -> TUser {
bb0:
  %conn: TDbConn = suspend.call @db.@connect {}
  %user: TUser = call @parse_user { row=%conn }
  ret %user
}
```

The compiler generates (conceptual MPIR):
```
;; Compiler-generated state struct (heap, ARC-managed)
heap struct T__fetch_user_state {
  field state_index: i32
  field id: u64              ;; captured parameter
  field conn: TDbConn        ;; live across suspend point 0
  field result: TUser         ;; final result
}

;; Resume function — called by executor via mp_rt_future_poll
fn @__fetch_user_resume(%state: mutborrow T__fetch_user_state) -> i32 {
bb0:
  %idx: i32 = getfield { obj=%state, field=state_index }
  switch %idx {
    case 0 -> bb_start
    case 1 -> bb_resume_0
  } else bb_unreachable

bb_start:
  ;; Start async call to @db.@connect, save state, return Pending
  %future: TFuture<TDbConn> = call @db.@connect {}
  setfield { obj=%state, field=state_index, val=const.i32 1 }
  ret const.i32 0            ;; 0 = Pending

bb_resume_0:
  ;; Resumed: read result from completed sub-future
  %conn: borrow TDbConn = getfield { obj=%state, field=conn }
  %user: TUser = call @parse_user { row=%conn }
  setfield { obj=%state, field=result, val=%user }
  ret const.i32 1            ;; 1 = Ready

bb_unreachable:
  unreachable
}
```

### 22.2 Error recovery

* Each pass collects up to `--max-errors` (default 20) errors.
* If a pass produces errors, dependent passes are skipped (e.g., no ownership check if type check failed).
* All collected diagnostics are emitted together.
* **MMS query timing:** MMS queries for diagnostic augmentation (§24) run after stage 5 (ownership check) and before diagnostic output, using the existing MMS index. This ensures repair suggestions can reference ownership errors without blocking the critical compilation path. **Staleness handling:** the MMS index may reference code from a previous compilation. The query layer MUST compare the module digest (BLAKE3) of each MMS capsule against the current compilation's digests. Capsules with stale digests are excluded from repair suggestions and marked for re-indexing at stage 12 (MMS update).

### 22.3 Incremental compilation

Cache keys MUST include: compiler version, toolchain hash, module digests (BLAKE3), dependency `.mpd` digests, feature set + target triple.

Cache layers: parsed AST, resolved HIR, MPIR, LLVM bitcode.

---

## 23. Interpreter/JIT and REPL

### 23.1 JIT engine

* Use LLVM ORC JIT.
* Maintain per-session symbol table.
* JIT and AOT must produce semantically equivalent results.

### 23.2 REPL cell model

Each cell compiles into a hidden module `repl.cell.N`. Expressions compile into `@__repl_eval_N`.

### 23.3 REPL over MCP (stateful sessions)

MCP exposes:
* `magpie.repl.create` → returns `session_id`
* `magpie.repl.eval` → takes `session_id` + code, returns result (budget-aware)
* `magpie.repl.inspect` → type/ir/llvm queries on session state

Sessions persist in memory. Session state is serializable for checkpointing.

### 23.4 Hot reload (`magpie web dev`)

* JIT-based hot swap via LLVM ORC.
* File watcher detects changes.
* Changed functions are recompiled and swapped in-place.
* Sub-second iteration. Request state preserved across reloads.
* Zero downtime.

#### 23.4.1 Hot reload mechanism (concrete)

1. **File watcher**: an OS-level watcher (inotify/kqueue/FSEvents) monitors `.mp` source files.
2. **Incremental recompile**: on change, only the affected module(s) are re-parsed, type-checked, and lowered to LLVM IR. The incremental cache (§22.3) is used to skip unchanged dependencies.
3. **ORC JIT replacement**: the new LLVM IR module is compiled via LLVM ORC JIT. The runtime calls `LLVMOrcReplaceObjectFiles` (or equivalent ORC API) to atomically replace function bodies in the running process. Old function bodies are freed after all in-flight calls complete.
4. **State preservation**: request-scoped state (in-flight HTTP contexts, channel buffers) is not affected because only function code is replaced — heap data and ARC objects remain valid.
5. **Constraints**: hot reload does NOT support changes to type layouts (struct field additions/removals). Such changes require a full restart. The compiler detects layout-breaking changes and logs a warning instead of attempting hot swap.

---

## 24. Compiler-integrated memory and RAG (MMS)

### 24.1 MMS storage layout

```
.magpie/
  memory/
    mms_meta.json
    items/
      <item_id>.json
    index_lex/
      vocab.bin
      postings.bin
      doclens.bin
      itemmap.bin
      bm25_meta.json
    episodes/
      <episode_id>.json
```

### 24.2 MMS item schema

```json
{
  "schema": 1,
  "item_id": "I:xxxxxxxxxxxxxxxx",
  "kind": "symbol_capsule|mpd_signature|doc_excerpt|spec_excerpt|diag_template|repair_episode|test_case",
  "sid": "F:XXXXXXXXXX",
  "fqn": "pkg.module.@fn",
  "module_sid": "M:XXXXXXXXXX",
  "source_digest": "<blake3hex>",
  "body_digest": "<blake3hex>",
  "text": "<canonical text>",
  "tags": ["MPO0007", "ownership", "web.http"],
  "priority": 50,
  "token_cost": {"approx:utf8_4chars": 350}
}
```

### 24.3 MMS retrieval (budget-aware, BM25)

BM25 with k1=1.2, b=0.75. Field boosts and deterministic tie-breaking. See Appendix E for exact lexical tokenizer spec and Appendix F for BM25 scoring formula, IDF definition, and boost values.

### 24.4 MMS integration

In `--llm` mode, `magpie build` MUST:
* Augment diagnostics with retrieved items
* Provide minimal symbol graph + ownership trace for failing functions
* Emit `rag_bundle` per diagnostic (budgeted)

### 24.5 MMS commands

* `magpie memory build` (incremental index update)
* `magpie memory query --q "<query>" --k <n> [--kinds ...]`
* `magpie ctx pack ...` (prompt-ready context pack builder)

---

## 25. Context pack builder (`magpie ctx pack`)

Generates prompt-ready context pack bounded by token budget. See Appendix G for complete specification including:

* Chunk types (structural, problem-focused, code capsules, retrieved)
* Scoring formula (base_priority + relevance + proximity + retrieval_score - size_penalty)
* Budget partitioning policies (balanced, diagnostics_first, slices_first, minimal)
* Multi-variant compression ladder (v3 full → v0 one-line identity)
* Deterministic selection algorithm

---

## 26. Diagnostics specification (LLM-grade)

### 26.1 JSON root schema

```json
{
  "magpie_version": "0.1.0",
  "command": "build",
  "target": "x86_64-unknown-linux-gnu",
  "success": false,
  "artifacts": [],
  "diagnostics": [],
  "graphs": { "symbols": {}, "deps": {}, "ownership": {}, "cfg": {} },
  "timing_ms": { "parse": 12, "typecheck": 31, "owncheck": 20, "arc": 5, "codegen": 40, "link": 60 },
  "llm_budget": {}
}
```

### 26.2 Per-diagnostic schema

```json
{
  "code": "MPO0007",
  "severity": "error",
  "title": "use of moved value",
  "primary_span": { "file": "src/main.mp", "start": 120, "end": 135 },
  "secondary_spans": [],
  "message": "...",
  "explanation_md": "...",
  "why": { "kind": "ownership_conflict", "trace": [] },
  "suggested_fixes": [{ "title": "clone into shared handle", "patch_format": "unified-diff", "patch": "...", "confidence": 0.82 }],
  "rag_bundle": [],
  "related_docs": []
}
```

### 26.3 Diagnostic code namespaces

* `MPP` parse/lex
* `MPT` types
* `MPO` ownership
* `MPA` ARC
* `MPF` FFI
* `MPG` GPU
* `MPW` Web
* `MPK` Package manager
* `MPL` Lint / LLM features
* `MPS` SSA verification

### 26.4 Debug information

* **Dev profile**: Emit DWARF debug info. SSA values map to debug variables. Basic blocks map to scope ranges.
* **All profiles**: Emit `.mpdbg` structured JSON — maps SIDs + block labels to source spans. Budget-aware for MMS retrieval.

---

## 27. Unified diff patch contract

### 27.1 Format

* MUST be Git-style unified diff
* Paths relative to workspace root
* No absolute paths, no out-of-root modifications

### 27.2 Patch JSON envelope

```json
{
  "title": "clone into shared handle",
  "patch_format": "unified-diff",
  "patch": "diff --git a/src/main.mp b/src/main.mp\n...",
  "applies_to": {"src/main.mp": "<pre_digest>"},
  "produces": {"src/main.mp": "<post_digest>"},
  "requires_fmt": true,
  "confidence": 0.82
}
```

---

## 28. Package manager (`magpie pkg`)

### 28.1 Manifest (`Magpie.toml`)

```toml
[package]
name = "my_pkg"
version = "0.1.0"
edition = "2026"

[build]
entry = "src/main.mp"
profile_default = "dev"
max_mono_instances = 10000

[dependencies]
std = { version = "^0.1" }

[features]
gui = { modules = ["src/gui/*.mp"] }

[toolchain.aarch64-unknown-linux-gnu]
sysroot = "/path/to/sysroot"
linker = "aarch64-linux-gnu-ld"

[llm]
mode_default = true
token_budget = 12000
tokenizer = "approx:utf8_4chars"
budget_policy = "balanced"
max_module_lines = 800
max_fn_lines = 80

[llm.rag]
enabled = true
backend = "lexical"
top_k = 12

[web]
addr = "127.0.0.1"
port = 3000
open_browser = false
max_body_bytes = 10000000
threads = 0  # 0 => auto

[gpu]
enabled = true
backend = "spv"   # currently: "spv" (Vulkan SPIR-V)
device_index = -1 # -1 => default device
```

Tool-specific tables (v0.1):

* `[web]` configures `magpie web dev/build/serve` (defaults are used when omitted).
  * `addr` (Str), `port` (u16), `open_browser` (bool)
  * `max_body_bytes` (u64), `threads` (u32; `0` means auto)
* `[gpu]` configures GPU defaults.
  * `enabled` (bool), `backend` (Str; `"spv"` in v0.1), `device_index` (i32; `-1` means default)

Unknown keys and unknown tables MUST be ignored to preserve forwards compatibility.


### 28.2 Registry model

* Git-based, no central hosted registry in v0.1.
* Packages referenced by git URL or local path.
* Curated index repository for discovery.
* Registry HTTP protocol defined for future self-hosted registries.
* `magpie pkg why <pkg>` outputs dependency reason tree.

### 28.3 Lockfile (`Magpie.lock`)

Canonical JSON. See Appendix C and Appendix D for JSON Schema definitions.

### 28.4 Feature flags

* Features gate entire modules via `[features]` section.
* `gui = { modules = ["src/gui/*.mp"] }` — when `gui` feature is inactive, those modules are excluded.
* No function-level or type-level conditionals in v0.1.

---

## 29. MCP server (`magpie mcp serve`)

### 29.1 Tools exposed

* `magpie.build`, `magpie.run`, `magpie.test`, `magpie.fmt`, `magpie.lint`, `magpie.explain`
* `magpie.pkg.resolve`, `magpie.pkg.add`, `magpie.pkg.remove`, `magpie.pkg.plan`
* `magpie.memory.build`, `magpie.memory.query`
* `magpie.ctx.pack`
* `magpie.repl.create`, `magpie.repl.eval`, `magpie.repl.inspect`
* `magpie.graph.symbols`, `magpie.graph.deps`, `magpie.graph.ownership`, `magpie.graph.cfg`

Each tool request MUST accept an optional `llm` object controlling budget/tokenizer/policy.

### 29.2 Security model

* Config file controlling: allowed filesystem roots, allowed network access (default deny), allowed subprocesses (default deny except linker/llvm tools).
* MUST never execute arbitrary scripts unless explicitly enabled.

---

## 30. Web frameworks

Magpie Web v0.1 is a **batteries-included** server-side web stack optimized for automated agents:
deterministic routing, explicit types, explicit serialization, and minimal ambient magic.

### 30.0 Scope and guarantees (v0.1)

**Included (MUST):**
* HTTP/1.1 server (cleartext) with keep-alive (configurable)
* Routing with **compile-time validated** route patterns (const-str only)
* Typed path parameters via compiler-generated wrappers (no runtime reflection)
* Middleware chain with explicit call to `next`
* Request/response bodies as bytes (buffered requests) and **streaming responses**
* JSON encode/decode intrinsics for heap structs (already defined in §30.1.8 / §7 ops)
* Testing harness that invokes router without opening sockets

**Out of scope (MAY in future):**
* TLS termination (use a reverse proxy in v0.1)
* HTTP/2, WebSockets, SSE, multipart/form-data
* Zero-copy request bodies (v0.1 buffers the body into memory)

### 30.1 Backend: Magpie Web Service Framework (MWSF)

#### 30.1.1 Core web types (authoritative)

All web framework types live under the `web.*` package namespace and are part of **Magpie Web v0.1**.

```mp
; -------- web.stream --------

sig web.stream.TNextSig() -> TOption<Array<u8>>

heap struct web.stream.TByteStream {
  field next: TCallable<web.stream.TNextSig>   ;; yields next chunk; None => end-of-stream
}

; -------- web.http --------

heap struct web.http.TRequest {
  field method: Str                 ;; "GET", "POST", etc. (uppercase)
  field path: Str                   ;; URL path only (e.g., "/users/42")
  field query: Map<Str, Str>        ;; parsed query params (first value wins)
  field headers: Map<Str, Str>      ;; lowercase header names; first value wins
  field body: Array<u8>             ;; fully-buffered request body (may be empty)
  field path_params: Map<Str, Str>  ;; extracted path params (strings, even if typed)
  field remote_addr: Str            ;; e.g. "203.0.113.10:54321"
}

heap struct web.http.TResponse {
  field status: i32                 ;; HTTP status code (e.g., 200, 404)
  field headers: Map<Str, Str>      ;; lowercase header names
  field body_kind: i32              ;; 0=bytes, 1=stream
  field body_bytes: Array<u8>       ;; used when body_kind==0
  field body_stream: web.stream.TByteStream  ;; used when body_kind==1
}

heap struct web.http.TContext {
  field state: Map<Str, Str>        ;; per-request scratch state (middleware may write)
  field request_id: Str             ;; unique request identifier (opaque string)
}
```

**Normalization rules (MUST):**
* `TRequest.method` MUST be uppercase.
* `TRequest.headers` keys MUST be lowercase ASCII.
* `TResponse.headers` keys MUST be lowercase ASCII.
* The server MUST set/overwrite header `x-request-id` on every response to `ctx.request_id`.

#### 30.1.2 Router/service types

```mp
sig web.router.THandlerSig(web.http.TRequest, web.http.TContext) -> web.http.TResponse

sig web.router.TMiddlewareSig(
  web.http.TRequest,
  web.http.TContext,
  TCallable<web.router.THandlerSig>    ;; next
) -> web.http.TResponse

heap struct web.router.TRoute {
  field method: Str                    ;; uppercase
  field pattern: Str                   ;; canonical route pattern string
  field handler: TCallable<web.router.THandlerSig>  ;; normalized to base signature via wrapper
}

heap struct web.router.TService {
  field prefix: Str                    ;; base path prefix (e.g., "/api")
  field routes: Array<web.router.TRoute>
  field middleware: Array<TCallable<web.router.TMiddlewareSig>>
}
```

#### 30.1.3 Route pattern syntax (const-only, compile-time validated)

A route pattern is a **const.Str** with this grammar:

* Literal segments: `/users`
* Typed parameters: `/{name:type}`
  * `name` = `[A-Za-z_][A-Za-z0-9_]*`
  * `type` ∈ `{ i32, i64, u32, u64, bool, Str }`
* Wildcard tail (optional, at end only): `/*{name}` (captures the remainder, unescaped)

Examples:
* `"/"`  
* `"/users/{id:u64}"`
* `"/assets/*{path}"`

**v0.1 restrictions (MUST):**
* `method` and `pattern` arguments to `web.router.@route_add` MUST be `const.Str`.
* If `pattern` contains typed params, the handler function MUST be compatible (see below).
* Duplicate route registrations (same method + canonical pattern) are a hard error: `MPW1001 DUPLICATE_ROUTE`.

#### 30.1.4 Handler signature matching + compiler-generated wrappers (typed params)

User handlers MAY include typed parameters after `(TRequest, TContext)`.

Given:
* method = `"GET"`
* pattern = `"/users/{id:u64}/posts/{slug:Str}"`

Then a compatible handler is:

```mp
fn @get_posts(%req: web.http.TRequest, %ctx: web.http.TContext, %id: u64, %slug: Str) -> web.http.TResponse { ... }
```

**Matching rule (MUST):**
* The handler's first two parameters MUST be `(web.http.TRequest, web.http.TContext)`.
* The remaining parameters MUST match the typed params in the pattern **left-to-right**.

**Wrapper generation (MUST):**
For every `@route_add(..., handler=@H)` where `%H` has typed params, the compiler generates a wrapper:

```
fn @__web_route_wrap_<SID>(%req: TRequest, %ctx: TContext) -> TResponse
```

Wrapper semantics:
1. Match the route pattern against `req.path` (after stripping `svc.prefix`).
2. Extract param substrings.
3. Store string params into `req.path_params` (always).
4. Parse each typed param:
   * `u64/u32/i64/i32/bool` via `str.parse_*`
   * `Str` uses the raw substring (no decoding in v0.1)
5. On parse failure: return a **400** response with JSON body:
   `{"error":"bad_request","param":"<name>","request_id":"..."}`
6. Call the user handler with typed values and return its response.

The route table stores the wrapper as `TCallable<web.router.THandlerSig>`.

#### 30.1.5 Middleware model (synchronous, production-safe)

Middleware is synchronous in v0.1 (no `suspend.call`), but may still perform blocking I/O.

Rules (MUST):
* Middleware MUST call `next` at most once.
* Middleware MAY short-circuit by returning a response without calling `next`.
* Middleware MAY read/write `ctx.state`.
* Middleware MUST NOT mutate `req.path` or `req.method` (compiler lint `MPW1102 REQUEST_MUTATION_FORBIDDEN`).

Example:

```mp
fn @auth_mw(
  %req: web.http.TRequest,
  %ctx: web.http.TContext,
  %next: TCallable<web.router.THandlerSig>
) -> web.http.TResponse {
bb0:
  ; auth check omitted (may short-circuit here)
  %resp: web.http.TResponse = call.indirect %next { args=[%req, %ctx] }
  ret %resp
}
```


#### 30.1.6 Required library functions (web.http / web.stream / web.router / web.server)

These functions are part of the `web.*` standard packages and MUST exist.

**`web.stream` helpers:**
* `web.stream.@from_bytes(bytes: Array<u8>) -> web.stream.TByteStream`
* `web.stream.@empty() -> web.stream.TByteStream`
* `web.stream.@concat(a: web.stream.TByteStream, b: web.stream.TByteStream) -> web.stream.TByteStream`

**`web.http` helpers:**
* `web.http.@response_bytes(status: i32, headers: Map<Str,Str>, body: Array<u8>) -> web.http.TResponse`
* `web.http.@response_stream(status: i32, headers: Map<Str,Str>, body: web.stream.TByteStream) -> web.http.TResponse`
* `web.http.@text(status: i32, body: Str) -> web.http.TResponse` (sets `content-type: text/plain; charset=utf-8`)
* `web.http.@json<T>(status: i32, v: borrow T) -> web.http.TResponse` (sets `content-type: application/json`)
* `web.http.@bad_request(msg: Str, request_id: Str) -> web.http.TResponse`
* `web.http.@not_found(request_id: Str) -> web.http.TResponse`
* `web.http.@internal_error(request_id: Str) -> web.http.TResponse`
* `web.http.@serialize_http1(resp: web.http.TResponse, request_id: Str) -> web.stream.TByteStream`
  * Produces the full HTTP/1.1 response bytes (status line + headers + body). Uses `Content-Length` for `body_kind==0` and `Transfer-Encoding: chunked` for `body_kind==1`, and injects/overwrites `x-request-id`.

**`web.router` entry points:**
* `web.router.@service_new(prefix: Str) -> web.router.TService`
* `web.router.@route_add(svc: web.router.TService, method: Str, pattern: Str, handler: TCallable<web.router.THandlerSig>) -> web.router.TService`
  * In surface code, the final argument MAY be a `FnRef` (e.g. `@get_user`). The compiler MUST coerce a `FnRef` to a captureless `TCallable<web.router.THandlerSig>`.
  * If the `FnRef` has typed params matching the route pattern (§30.1.4), the compiler MUST generate a wrapper first, then coerce the wrapper to `TCallable<web.router.THandlerSig>`.

  * **MUST** be called with `method` and `pattern` as `const.Str` (compile-time validated).
* `web.router.@middleware_add(svc: web.router.TService, mw: TCallable<web.router.TMiddlewareSig>) -> web.router.TService`
* `web.router.@dispatch(svc: web.router.TService, req: web.http.TRequest, ctx: web.http.TContext) -> web.http.TResponse`
  * Matches route + runs middleware chain + handler wrapper.

#### 30.1.7 HTTP server entry point (`web.server.@serve`) and runtime contract

`web.server.@serve` is the **only** socket-facing API in v0.1.

```mp
heap struct web.server.TServeOpts {
  field keep_alive: bool          ;; default true
  field threads: u32              ;; default = number of logical CPUs (min 1)
  field max_body_bytes: u64       ;; default 10_000_000
  field read_timeout_ms: u64      ;; default 30_000
  field write_timeout_ms: u64     ;; default 30_000
  field log_requests: bool        ;; default false
}

fn web.server.@serve(
  %svc: web.router.TService,
  %addr: Str,
  %port: u16,
  %opts: web.server.TServeOpts
) -> TResult<unit, Str>   ;; Err contains a human-opaque error string
```

**Lowering rule (MUST):**
The compiler lowers `web.server.@serve` to the runtime function `mp_rt_web_serve` (defined in §20.1.6).
`Ok(())` is returned only when the server exits cleanly. `Err(msg)` indicates startup failure.

**Runtime → Magpie callbacks (MUST):**
The runtime MUST call these exported symbols provided by the web standard package:

```c
// For each request: returns an owned TByteStream producing the full HTTP response bytes.
MpRtHeader* __magpie_web_handle_request(
  MpRtHeader* svc,          // web.router.TService
  MpRtHeader* method,       // Str
  MpRtHeader* path,         // Str (path only, no query)
  MpRtHeader* query,        // Map<Str,Str>
  MpRtHeader* headers,      // Map<Str,Str>
  MpRtHeader* body_bytes,   // Array<u8>
  MpRtHeader* remote_addr,  // Str
  MpRtHeader* request_id    // Str
);

// Pull next chunk. Returns 1 and sets *out_chunk on success, 0 when finished.
int32_t __magpie_web_stream_next(MpRtHeader* stream, MpRtHeader** out_chunk);
```

**Ownership contract (MUST):**
* Runtime transfers ownership of all request objects passed to `__magpie_web_handle_request`.
* The callback transfers ownership of returned `stream` to the runtime.
* Each `out_chunk` returned by `__magpie_web_stream_next` is owned by the runtime, which MUST release it after write.
* Runtime MUST release the stream after completion (and on error).

This design keeps the runtime oblivious to Magpie struct field layouts.

#### 30.1.8 JSON encode/decode intrinsics (backend)

The `json.encode<T>` / `json.decode<T>` intrinsics described in §7 are the canonical JSON mechanism.

Additional web JSON rules (MUST):
* JSON for `web.http` errors MUST use UTF-8 and set `content-type: application/json; charset=utf-8`.
* Encoding/decoding MUST be deterministic (stable key order for maps).

#### 30.1.9 Testing harness (no sockets)

`web.test.@request` MUST exist:

```mp
fn web.test.@request(
  %svc: web.router.TService,
  %method: Str,
  %path: Str,
  %query: Map<Str,Str>,
  %headers: Map<Str,Str>,
  %body: Array<u8>
) -> web.http.TResponse
```

Semantics:
* Constructs a `TRequest`/`TContext` with a deterministic `request_id` (e.g. `"test-<counter>"`).
* Calls `web.router.@dispatch`.
* Returns the response.

### 30.2 Frontend: Magpie Web App Framework (MWAF) (SSR)

MWAF is a **code generator + conventions** that builds on MWSF. It is required for Magpie Web v0.1 conformance.

#### 30.2.1 Project layout (MUST)

```
app/
  routes/
    index.mp
    about.mp
    users/
      [id:u64].mp
    _layout.mp          ;; optional root layout
  assets/
    ... static files ...
```

#### 30.2.2 File-based routing rules (MUST)

* `app/routes/index.mp` maps to path `/`.
* `app/routes/<name>.mp` maps to path `/<name>`.
* Nested directories map to nested paths.
* Dynamic segment file name: `[param:type].mp` maps to `/{param:type}`.
  * `type` is the same set as §30.1.3.

#### 30.2.3 Route module contract (MUST)

A route file MUST export:

* `fn @render(%req: web.http.TRequest, %ctx: web.http.TContext) -> web.ui.TNode`

Optional:
* `fn @data(%req: web.http.TRequest, %ctx: web.http.TContext) -> TResult<TProps, Str>`
  * `TProps` MUST be a JSON-serializable heap struct.
  * If present, the generator calls `@data`, and passes the resulting props to `@render` via an overload:
    `fn @render(%req, %ctx, %props: TProps) -> web.ui.TNode`

Optional root layout:
* `app/routes/_layout.mp` MAY export:
  * `fn @wrap(%req: web.http.TRequest, %ctx: web.http.TContext, %child: web.ui.TNode) -> web.ui.TNode`

#### 30.2.4 UI representation (`web.ui.TNode`) (authoritative)

```mp
heap struct web.ui.TAttr {
  field key: Str
  field value: Str
}

heap struct web.ui.TNode {
  field tag: Str                 ;; "div", "span", "#text", "#raw"
  field text: Str                ;; used when tag == "#text" or "#raw"
  field attrs: Array<web.ui.TAttr>
  field children: Array<web.ui.TNode>
}
```

Rules (MUST):
* `tag == "#text"`: `text` is HTML-escaped.
* `tag == "#raw"`: `text` is inserted verbatim (unsafe; only allowed in `unsafe` context in v0.1).
* Otherwise: `tag` is a lower-case HTML tag name; `text` MUST be empty; children render recursively.

#### 30.2.5 SSR rendering (bytes and streaming)

Required functions:

* `web.ui.@render_bytes(node: web.ui.TNode) -> Array<u8>`
* `web.ui.@render_stream(node: web.ui.TNode) -> web.stream.TByteStream`

Streaming rules (MUST):
* `@render_stream` MUST produce valid incremental HTML. It MAY chunk at arbitrary boundaries.
* The default chunk size SHOULD be 16KiB (configurable).

#### 30.2.6 MWAF code generation (`magpie web build` / `magpie web dev`)

The `magpie web` tool scans `app/routes/**` and generates a module:

* `.magpie/gen/webapp_routes.mp` containing:
  * `fn web.app.@service(prefix: Str) -> web.router.TService` which registers:
    * page routes (GET)

Static assets are served by the **web runtime** (not by generated Magpie routes):
* In `magpie web dev`, `/assets/*` is served from `app/assets` with caching disabled.
* In `magpie web serve`, `/assets/*` is served from `dist/assets` with caching enabled.

  * wrappers for typed path params (via §30.1.4)
  * wrappers for `@data`/`@render`/`_layout.@wrap`
  * SSR responses using `web.ui.@render_stream`

The generated service is deterministic and stable under CSNF.

#### 30.2.7 Islands (WASM) (optional in v0.1)

Islands are **optional** in Magpie Web v0.1. If implemented, they MUST follow this contract:

* A function annotated with `meta { effects { web.client } }` and returning `web.ui.TNode` is a client island entry.
* `magpie web build` compiles islands to wasm32 and emits:
  * `dist/islands/<sid>.wasm`
  * `dist/islands/<sid>.js` loader
* SSR rendering inserts placeholders:
  `<div data-magpie-island="<sid>" data-magpie-props="<json>"></div>`
* The JS loader MUST:
  * find placeholders
  * fetch wasm
  * call an exported function `magpie_island_mount(ptr, len)` where `(ptr,len)` is UTF-8 JSON props in WASM memory
  * mount/hydrate into the DOM element

### 30.3 `magpie web` commands and artifacts (v0.1)

#### 30.3.1 `magpie web dev`

* Runs the web server on `addr/port` from `[web]` manifest config (defaults: `127.0.0.1:3000`).
* Enables hot reload (§23.4).
* Serves assets from `app/assets` with no caching.
* Rebuilds generated MWAF module on file changes under `app/routes` and `app/assets`.

#### 30.3.2 `magpie web build`

Produces `dist/`:

* `dist/server/<name>` — native executable
* `dist/assets/**` — copied static assets (hash in filename optional)
* `dist/openapi.json` — OpenAPI (MWSF routes only; pages excluded)
* `dist/routes.json` — route manifest (for tooling)

#### 30.3.3 `magpie web serve`

Runs `dist/server/<name>` and serves `dist/assets/**` with caching headers.

---
## 31. GPU specification

Magpie GPU v0.1 defines a **compute-only** kernel subset and a host dispatch API designed for LLM-written code:
explicit buffers, explicit launches, and deterministic binding rules.

### 31.0 Conformance and backends (v0.1)

A toolchain claiming **Magpie GPU v0.1** conformance MUST implement:

* **Vulkan Compute backend via SPIR-V** (`target(spv)`)

The following backends are OPTIONAL (MAY be implemented later):
* CUDA via LLVM NVPTX (`target(ptx)`)
* Metal (`target(msl)`)
* WebGPU/WGSL (`target(wgsl)`)

If a backend is not available at runtime, GPU operations MUST return `Err(<message>)` (not panic).

### 31.1 Host-visible GPU types (builtins)

GPU handles are builtin heap types implemented by the runtime (opaque payloads):

* `gpu.TDevice` — a GPU device handle
* `gpu.TBuffer<T>` — a typed device buffer
* `gpu.TFence` — completion handle for async launches (optional; may be a no-op in sync mode)

Errors are represented as `Str` in v0.1 (opaque, for machine consumption).

### 31.2 Host API surface (compiler-known intrinsics)

All host GPU functions live in the `gpu.host` namespace and are compiler-known intrinsics lowered to runtime ABI calls (§20.1.7).

```mp
; Device discovery
fn gpu.host.@device_default() -> TResult<gpu.TDevice, Str>
fn gpu.host.@device_count() -> u32
fn gpu.host.@device_by_index(%idx: u32) -> TResult<gpu.TDevice, Str>
fn gpu.host.@device_name(%dev: borrow gpu.TDevice) -> Str

; Buffers (device-local by default)
fn gpu.host.@buffer_new<T: type>(
  %dev: borrow gpu.TDevice,
  %len: u64,
  %usage_flags: u32
) -> TResult<gpu.TBuffer<T>, Str>

fn gpu.host.@buffer_from_array<T: type>(
  %dev: borrow gpu.TDevice,
  %src: borrow Array<T>,
  %usage_flags: u32
) -> TResult<gpu.TBuffer<T>, Str>

fn gpu.host.@buffer_to_array<T: type>(
  %buf: borrow gpu.TBuffer<T>
) -> TResult<Array<T>, Str>

fn gpu.host.@buffer_copy<T: type>(
  %src: borrow gpu.TBuffer<T>,
  %dst: borrow gpu.TBuffer<T>
) -> TResult<unit, Str>

fn gpu.host.@buffer_len<T: type>(%buf: borrow gpu.TBuffer<T>) -> u64

; Synchronization
fn gpu.host.@fence_wait(%f: borrow gpu.TFence, %timeout_ms: u64) -> TResult<bool, Str> ;; Ok(true)=done, Ok(false)=timeout
fn gpu.host.@device_sync(%dev: borrow gpu.TDevice) -> TResult<unit, Str>
```

**Usage flags (v0.1):**
* Bit 0 (`1<<0`) = `STORAGE` (read/write in kernels) — default
* Bit 1 (`1<<1`) = `UNIFORM` (read-only uniform) — optional
* Bit 2 (`1<<2`) = `TRANSFER_SRC`
* Bit 3 (`1<<3`) = `TRANSFER_DST`

The compiler MAY infer missing transfer bits when needed for copies.

### 31.3 Kernel declaration (`gpu fn`) and restrictions

A kernel is declared with `gpu fn` and a required backend target:

```mp
gpu fn @kernel_add(
  %in: gpu.TBuffer<f32>,
  %out: gpu.TBuffer<f32>,
  %n: u32
) -> unit target(spv) {
bb0:
  %gid: u32 = gpu.global_id { dim=const.u32 0 }
  %in_bounds: bool = icmp.ult { lhs=%gid, rhs=%n }
  cbr %in_bounds bb1 bb2

bb1:
  %x: f32 = gpu.buffer_load<f32> { buf=%in, idx=%gid }
  %y: f32 = f.add { a=%x, b=const.f32 1.0 }
  gpu.buffer_store<f32> { buf=%out, idx=%gid, v=%y }
  br bb2

bb2:
  ret
}
```

**Kernel restrictions (MUST):**
* No heap allocation, no ARC operations, no Str/Array/Map, no TCallable.
* No recursion.
* No dynamic dispatch (no TCallable calls; no `call.indirect`).
* Allowed local types:
  * primitives (`i*`, `u*`, `f*`, `bool`)
  * value structs containing only primitives
  * `gpu.TBuffer<T>` handles
* All out-of-bounds checks MUST be explicit unless the kernel body is inside `unsafe {}`.

### 31.4 Device-side buffer access ops (MUST)

These ops are valid **only inside `gpu fn`**:

```
gpu.buffer_load<T>  { buf=<ValueRef>, idx=<ValueRef> }  -> T
gpu.buffer_store<T> { buf=<ValueRef>, idx=<ValueRef>, v=<ValueRef> } -> unit
gpu.buffer_len<T>   { buf=<ValueRef> } -> u32
```

Semantics:
* `idx` is an element index (not bytes).
* `gpu.buffer_load` reads `buf[idx]`.
* `gpu.buffer_store` writes `buf[idx]`.
* Bounds are NOT implicit: out-of-bounds behavior is undefined unless compiler inserts checks.

### 31.5 GPU builtins (thread/workgroup) (MUST)

Available inside `gpu fn`:

* `gpu.thread_id { dim=<0|1|2> } -> u32`
* `gpu.workgroup_id { dim=<0|1|2> } -> u32`
* `gpu.workgroup_size { dim=<0|1|2> } -> u32`
* `gpu.global_id { dim=<0|1|2> } -> u32`
* `gpu.barrier` (workgroup barrier)
* `gpu.shared<N,T>` — shared memory allocation (optional in v0.1; MAY be unimplemented, in which case compiler errors with `MPG1201 SHARED_UNSUPPORTED`)

### 31.6 Kernel launch (`gpu.launch` / `gpu.launch_async`)

Host code dispatches kernels via explicit launch ops:

```mp
%dev: gpu.TDevice = try gpu.host.@device_default { args=[] }
%buf_in: gpu.TBuffer<f32> = try gpu.host.@buffer_from_array<f32> { args=[%dev, %a, const.u32 0] }
%buf_out: gpu.TBuffer<f32> = try gpu.host.@buffer_new<f32> { args=[%dev, %n, const.u32 0] }

%launch: TResult<unit, Str> = gpu.launch {
  device=%dev,
  kernel=@kernel_add,
  grid=[%gx, %gy, %gz],
  block=[%bx, %by, %bz],
  args=[%buf_in, %buf_out, %n]
}
```


Launch forms:

* **Synchronous** (MUST): `gpu.launch { ... } -> TResult<unit, Str>`
  * The runtime MUST not return `Ok(())` until the kernel has completed and all writes are visible to subsequent host reads/copies on the same device.
* **Asynchronous** (OPTIONAL): `gpu.launch_async { ... } -> TResult<gpu.TFence, Str>`
  * Completion is tested via `gpu.host.@fence_wait`.

**Deterministic binding rules (MUST):**
* Buffer parameters are bound in parameter order to set=0, binding=b where b increments for each buffer param.
* Scalar parameters are packed into a single push-constant block in parameter order using std430-like alignment:
  * scalars aligned to their size, capped at 16 bytes
  * the block size is rounded up to 16
* The compiler MUST generate the same layout across builds given the same kernel signature.

### 31.7 Kernel compilation and embedding (MUST)

During `magpie build`:

1. Each `gpu fn ... target(spv)` is lowered to a GPU LLVM module.
2. The module is compiled to a SPIR-V binary blob.
3. The blob is embedded in the host object as a `const` byte array.
4. The compiler emits a kernel registry section:

```c
typedef struct MpRtGpuKernelEntry {
  uint64_t sid_hash;        // hash of the kernel's SID string
  uint32_t backend;         // 1=SPV
  const uint8_t* blob;
  uint64_t blob_len;
  uint32_t num_params;
  const MpRtGpuParam* params;
} MpRtGpuKernelEntry;
```

At program startup, the runtime is called to register all kernels:
`mp_rt_gpu_register_kernels(entries, count)` (§20.1.7).

### 31.8 CLI integration (v0.1)

* `magpie build` MUST compile and embed GPU kernels when they are reachable from the build graph.
* `--emit spv` MUST emit per-kernel SPIR-V blobs to `target/<profile>/gpu/<sid>.spv`.
* `--features gpu` MAY be used to gate GPU-only modules.

---
## 32. Unsafe and C FFI

### 32.1 Unsafe blocks and functions

```
unsafe {
  %p: rawptr<u8> = call @malloc { size=%sz }
}

unsafe fn @dangerous_op(%p: rawptr<u8>) -> i32 {
bb0:
  ; entire body is unsafe context
  ...
}
```

Rules:
* `rawptr<T>` operations outside `unsafe` is a hard error.
* `unsafe fn` makes the entire function body an unsafe context.
* Callers of `unsafe fn` must be in an unsafe context.

#### 32.1.1 Raw pointer opcodes (`ptr.*`)

`rawptr<T>` is an unsafe, non-owning pointer to a value of type `T`. `rawptr<T>` is a **Copy** type and never participates in ARC.

All `ptr.*` opcodes MUST appear in an unsafe context (inside `unsafe {}` or inside an `unsafe fn`). Violations are a hard error.

Semantics (LLVM-like; UB is allowed):

* `%p: rawptr<T> = ptr.null<T>` produces a null pointer.
* `%addr: u64 = ptr.addr<T> { p=%p }` converts a raw pointer to an address.
* `%p: rawptr<T> = ptr.from_addr<T> { addr=%addr }` converts an address to a raw pointer.
* `%q: rawptr<T> = ptr.add<T> { p=%p, count=%n }` performs element-wise pointer arithmetic (`q = p + n*sizeof(T)`).
* `%v: T = ptr.load<T> { p=%p }` loads a `T` from memory.
* `ptr.store<T> { p=%p, v=%v }` stores a `T` to memory.

**Undefined behavior (UB):** `ptr.load`/`ptr.store` on an invalid address, misaligned address for `T`, or violating aliasing rules is UB.


### 32.2 Extern modules

```
extern "c" module libc {
  fn @puts(%s: borrow Str) -> i32 attrs { returns="borrowed" }
}
```

Every extern function returning a pointer MUST specify ownership in attrs. Missing: `MPF0001 FFI_RETURN_OWNERSHIP_REQUIRED`.

---

## 33. Testing and linting

### 33.1 Test discovery

* Functions prefixed with `@test_` are test cases.
* Test modules live in `tests/*.mp`.
* `magpie test` discovers and runs all test functions.
* Assertions via `std.test.@assert` (panics with message) and `std.test.@assert_eq` (panics with expected/actual).

### 33.2 Lint categories

**Style + complexity:**
* `MPL2001 FN_TOO_LARGE` — function exceeds `max_fn_lines`
* `MPL2003 MODULE_TOO_LARGE` — module exceeds `max_module_lines`
* `MPL2004 UNUSED_IMPORT` — import not referenced
* `MPL2005 UNUSED_LOCAL` — SSA value defined but never used
* `MPL2006 MISSING_META` — function lacks meta block in `--llm` mode
* `MPL2007 NAMING_CONVENTION` — identifier doesn't match convention

**Safety:**
* `MPL2010 POTENTIAL_CYCLE` — type can form ownership cycles without weak
* `MPL2011 UNCHECKED_RESULT` — Result value used without checking error
* `MPL2012 DEAD_CODE` — unreachable code detected
* `MPL2013 UNBOUNDED_RECURSION` — recursive call with no base case detected
* `MPL2014 PANIC_REACHABLE` — panic instruction reachable in production code

**LLM-specific:**
* `MPL2002 COST_UNDERESTIMATE` — actual cost >2x declared
* `MPL2020 EXCESSIVE_MONO` — monomorphized instances exceed budget
* `MPL0801 TOKEN_BUDGET_TOO_SMALL` — budget insufficient for minimal output

All lints configurable in `Magpie.toml`:

```toml
[lint]
MPL2001 = "error"    # promote to error
MPL2010 = "allow"    # silence
MPL2006 = "warn"     # default
```

---

## 34. Standard library surface

Minimal std (Go-like):

| Package | Contents |
|---------|----------|
| `std.core` | TOption, TResult (lang items, auto-available) |
| `std.io` | @println, @readln, @stdin, @stdout, @stderr, file read/write |
| `std.str` | String operations (intrinsics: concat, len, eq, slice, bytes) + TStrBuilder |
| `std.os` | @env_var, @args, @exit, @cwd |
| `std.math` | Basic numeric functions (abs, min, max, pow, sqrt, floor, ceil) |
| `std.test` | @assert, @assert_eq, @assert_ne, @fail |
| `std.sync` | TMutex, TRwLock, TChannel, TCell |
| `std.thread` | @spawn, @sleep, @yield_now |
| `std.hash` | @hash_Str, @hash_i32, @hash_i64, etc. |
| `std.async` | @block_on, @spawn_task, TFuture\<T\> — async executor entry points |
| `std.parse` | Intrinsic opcodes: `str.parse_i64`, `str.parse_u64`, `str.parse_f64`, `str.parse_bool` (§34.2) |

Extended packages (v0.1):

| Package | Summary |
|---|---|
| `gpu.host` | Host-side GPU API (device discovery, buffers, dispatch) (§31) |
| `web.http` | HTTP request/response types and helpers (§30) |
| `web.router` | Routing and middleware (§30) |
| `web.server` | HTTP server entry point (`@serve`) (§30) |
| `web.stream` | Byte streams for streaming responses (§30) |
| `web.ui` | SSR UI nodes and HTML renderer (§30) |
| `web.test` | Socketless request testing harness (§30) |
| `web.app` | MWAF generated glue module (not authored by hand) (§30) |

All core type operations (Str, Array, Map) are compiler intrinsics lowered to runtime ABI calls. No Magpie source needed for these.

### 34.1 `std.async` package

| Intrinsic | Call form | Semantics |
|-----------|----------|-----------|
| `@block_on<T>` | `call std.async.@block_on<T> { fn=@async_fn, args=[...] }` | Runs an async function to completion on the default executor (blocking the current thread). |
| `@spawn_task<T>` | `call std.async.@spawn_task<T> { fn=@async_fn, args=[...] }` | Spawns an async task on the default executor; returns `TFuture<T>`. |

Notes:

* These are compiler intrinsics. The `fn`/`args` call form is validated by the compiler (arity and types).
 `TFuture<T>` is the handle returned by `@spawn_task`. It is ARC-managed and `send + sync`.
* `@block_on` is the entry point for calling async code from non-async contexts.

### 34.2 `str.parse_*` intrinsics

| Intrinsic | Signature | Semantics |
|-----------|-----------|-----------|
| `str.parse_i64` | `(Str) -> TResult<i64, TParseError>` | Parse string to i64 |
| `str.parse_u64` | `(Str) -> TResult<u64, TParseError>` | Parse string to u64 |
| `str.parse_f64` | `(Str) -> TResult<f64, TParseError>` | Parse string to f64 |
| `str.parse_bool` | `(Str) -> TResult<bool, TParseError>` | Parse "true"/"false" to bool |

`TParseError` is a heap struct: `heap struct TParseError { field message: Str }`

Transitional compatibility note (v0.1 implementation status):

* The language target model remains fallible (`TResult<_, TParseError>`).
* Current lowering keeps legacy success-only parse/json op shapes for compatibility, but codegen MUST call fallible runtime ABI (`mp_rt_*_try_*`) and branch on status.
* On non-OK status in this compatibility path, codegen currently calls `mp_rt_panic` with the runtime-provided error string.
* Direct Rust `panic!`/`expect!` behavior at parse/json FFI boundaries is not allowed for recoverable input failures.

---

## 35. Distribution and agent packaging

### 35.1 Codex skill pack

* `SKILL.md` with YAML front matter
* `agents/openai.yaml`
* scripts: `bin/magpie-build`, `bin/magpie-run`

### 35.2 Claude Code plugin pack

* `.claude-plugin/plugin.json` optional manifest

---

## 36. Compiler implementation requirements (Rust)

### 36.1 Workspace crates

* `magpie_cli` — CLI entry point
* `magpie_driver` — compilation orchestration
* `magpie_lex` — lexer
* `magpie_parse` — parser (hand-written recursive descent)
* `magpie_csnf` — formatter/canonicalizer
* `magpie_ast` — AST types
* `magpie_hir` — HIR types and lowering
* `magpie_sema` — name resolution, symbol tables
* `magpie_types` — type interning/layout
* `magpie_mono` — monomorphization
* `magpie_own` — ownership checker (dataflow)
* `magpie_mpir` — MPIR builder + verifier
* `magpie_arc` — ARC insertion + optimization
* `magpie_codegen_llvm` — LLVM codegen (llvm-sys bindings)
* `magpie_codegen_wasm` — WASM-specific codegen adjustments
* `magpie_jit` — ORC JIT
* `magpie_diag` — diagnostics + patches
* `magpie_pkg` — manifest/lock/registry
* `magpie_memory` — MMS indexing + retrieval
* `magpie_ctx` — ctx pack builder
* `magpie_web` — backend + SSR frameworks
* `magpie_gpu` — GPU compilation
* `magpie_rt` — runtime library

### 36.2 SourceMap and spans

All parsed nodes MUST carry byte spans for precise diagnostics.

```rust
pub struct FileId(pub u32);

pub struct Span {
    pub file: FileId,
    pub start: u32,
    pub end: u32,
}

pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}
```

### 36.3 Parsing strategy

* Hand-written recursive descent with recovery by synchronization tokens at block/decl boundaries.
* All parse errors produce diagnostics with precise span, expected tokens set, and recovery action.

### 36.4 Linking

* Default: static linking, all `.o` into one binary.
* Optional: `--emit shared-lib` produces `.so`/`.dylib` with C ABI exports.
* `magpie_rt` is always statically linked.
* Prefer `lld` if available; otherwise system linker.
* Release mode: deterministic builds.

### 36.5 Cross-compilation

* Toolchain bundles defined in `Magpie.toml` under `[toolchain.<triple>]`.
* Specifies sysroot, linker, and flags.
* `magpie_rt` must be pre-compiled for each target and bundled.

---

## 37. Appendices

### Appendix A — Complete instruction set table (v0.1)

**Constants:** `const.*`

**Integer (checked):** `i.add i.sub i.mul i.sdiv i.udiv i.srem i.urem`

**Integer (wrapping):** `i.add.wrap i.sub.wrap i.mul.wrap`

**Integer (checked → TOption):** `i.add.checked i.sub.checked i.mul.checked`

**Bitwise:** `i.and i.or i.xor i.shl i.lshr i.ashr`

**Float (strict IEEE 754):** `f.add f.sub f.mul f.div f.rem`

**Float (fast-math):** `f.add.fast f.sub.fast f.mul.fast f.div.fast`

**Compare:** `icmp.eq icmp.ne icmp.slt icmp.sgt icmp.sle icmp.sge icmp.ult icmp.ugt icmp.ule icmp.uge` / `fcmp.oeq fcmp.one fcmp.olt fcmp.ogt fcmp.ole fcmp.oge`

**Control:** `br cbr switch ret unreachable phi`

**Calls:** `call call_void try suspend.call`

**Heap:** `new getfield setfield`

**Enum:** `enum.tag enum.payload<V> enum.is<V>`

**Ownership:** `share clone.shared weak.downgrade weak.upgrade`

**ARC:** `arc.retain arc.release arc.retain_weak arc.release_weak`

**Callable:** `callable.capture`

**Array:** `arr.new arr.len arr.get arr.set arr.push arr.pop arr.slice arr.contains arr.sort arr.map arr.filter arr.reduce arr.foreach`

**Map:** `map.new map.len map.get map.set map.delete map.contains_key map.keys map.values`

**String:** `str.concat str.len str.eq str.slice str.bytes str.builder.new str.builder.append_str str.builder.append_i64 str.builder.append_i32 str.builder.append_f64 str.builder.append_bool str.builder.build`

**GPU:** `gpu.thread_id gpu.workgroup_id gpu.workgroup_size gpu.global_id gpu.barrier gpu.shared`

**Error:** `panic`

**Cast:** `cast<From,To>` (primitive types only in v0.1)

### Appendix B — Backend framework MVP checklist

1. `web.http` request/response types
2. `web.router` route matcher + typed param extraction (compiler-generated parsers)
3. `web.server` minimal async server (Rust runtime)
4. JSON encode/decode for heap structs
5. Route registration builder function pattern
6. `magpie web dev` wiring: build + JIT hot-swap
7. OpenAPI JSON generation from route table + types

### Appendix C — Manifest JSON Schema (`magpie-manifest.schema.json`, Draft 2020-12)

```json
{
  "$schema":"https://json-schema.org/draft/2020-12/schema",
  "$id":"magpie-manifest.schema.json",
  "type":"object",
  "required":["package","build","dependencies"],
  "properties":{
    "package":{
      "type":"object",
      "required":["name","version","edition"],
      "properties":{
        "name":{"type":"string","pattern":"^[a-z][a-z0-9_\\-]{1,63}$"},
        "version":{"type":"string","pattern":"^[0-9]+\\.[0-9]+\\.[0-9]+.*$"},
        "edition":{"type":"string"}
      },
      "additionalProperties":true
    },
    "build":{
      "type":"object",
      "required":["entry","profile_default"],
      "properties":{
        "entry":{"type":"string"},
        "profile_default":{"type":"string","enum":["dev","release","custom"]}
      },
      "additionalProperties":true
    },
    "dependencies":{
      "type":"object",
      "additionalProperties":{
        "type":"object",
        "required":["version"],
        "properties":{
          "version":{"type":"string"},
          "registry":{"type":"string"},
          "path":{"type":"string"},
          "git":{"type":"string"},
          "rev":{"type":"string"},
          "features":{"type":"array","items":{"type":"string"}},
          "optional":{"type":"boolean"}
        },
        "additionalProperties":true,
        "allOf":[{"not":{"anyOf":[{"required":["path","git"]},{"required":["path","registry"]},{"required":["git","registry"]}]}}]
      }
    },
    "llm":{
      "type":"object",
      "properties":{
        "mode_default":{"type":"boolean"},
        "token_budget":{"type":"integer","minimum":256,"maximum":1000000},
        "tokenizer":{"type":"string"},
        "budget_policy":{"type":"string","enum":["balanced","diagnostics_first","slices_first","minimal"]},
        "max_module_lines":{"type":"integer","minimum":50,"maximum":200000},
        "max_fn_lines":{"type":"integer","minimum":10,"maximum":200000},
        "auto_split_on_budget_violation":{"type":"boolean"},
        "rag":{
          "type":"object",
          "properties":{
            "enabled":{"type":"boolean"},
            "backend":{"type":"string","enum":["lexical","vector","hybrid"]},
            "top_k":{"type":"integer","minimum":1,"maximum":200},
            "max_items_per_diag":{"type":"integer","minimum":0,"maximum":50},
            "include_repair_episodes":{"type":"boolean"}
          },
          "additionalProperties":true
        }
      },
      "additionalProperties":true
    }
  },
  "additionalProperties":true
}
```

### Appendix D — Lockfile JSON Schema (`magpie-lock.schema.json`, Draft 2020-12)

```json
{
  "$schema":"https://json-schema.org/draft/2020-12/schema",
  "$id":"magpie-lock.schema.json",
  "type":"object",
  "required":["lock_version","generated_by","packages"],
  "properties":{
    "lock_version":{"type":"integer","enum":[1]},
    "generated_by":{
      "type":"object",
      "required":["magpie_version","toolchain_hash"],
      "properties":{
        "magpie_version":{"type":"string"},
        "toolchain_hash":{"type":"string"}
      },
      "additionalProperties":true
    },
    "packages":{
      "type":"array",
      "items":{
        "type":"object",
        "required":["name","version","source","content_hash","deps"],
        "properties":{
          "name":{"type":"string"},
          "version":{"type":"string"},
          "source":{
            "type":"object",
            "required":["kind"],
            "properties":{
              "kind":{"type":"string","enum":["registry","path","git"]},
              "registry":{"type":"string"},
              "url":{"type":"string"},
              "path":{"type":"string"},
              "rev":{"type":"string"}
            },
            "additionalProperties":true
          },
          "content_hash":{"type":"string"},
          "deps":{
            "type":"array",
            "items":{
              "type":"object",
              "required":["name","req"],
              "properties":{
                "name":{"type":"string"},
                "req":{"type":"string"},
                "features":{"type":"array","items":{"type":"string"}}
              },
              "additionalProperties":true
            }
          },
          "resolved_features":{"type":"array","items":{"type":"string"}},
          "targets":{"type":"array","items":{"type":"string"}}
        },
        "additionalProperties":true
      }
    }
  },
  "additionalProperties":true
}
```

### Appendix E — MMS Lexical Tokenizer (exact spec)

This tokenizer is used for MMS lexical indexing and BM25 scoring. It MUST be deterministic and platform-independent.

**Input normalization (MUST):** Given UTF-8 input `S`: (1) Convert to Unicode NFKC normalization. (2) Apply Unicode case folding (full). (3) Replace `\r\n` and `\r` with `\n`. (4) Collapse 2+ whitespace to single space. (5) Trim leading/trailing whitespace.

**Token categories:** Each term has `term_text`, `term_kind` (`word|symbol|number|code|diag|path`), `position` (0-based).

**Character classes:** `ALNUM` (Unicode letters/digits), `UNDERSCORE` (`_`), `DOT` (`.`), `COLON` (`:`), `SLASH` (`/`), `AT` (`@`), `PERCENT` (`%`), `HASH` (`#`), `DASH` (`-`), `PLUS` (`+`), `OTHER`.

**Joiner set:** `@ % . : / _ -` (inside certain tokens).

**Primary scan (left-to-right, longest match):**

* **Pattern A — Diagnostic code (`diag`):** `MP[A-Z][0-9]{4}` (e.g., `MPO0007`)
* **Pattern B — Symbol (`symbol`):** Starts with `@` or `%` or contains `.@`/`.%`. Allowed: ALNUM + joiners. Must contain `@` or `%`.
* **Pattern C — Path (`path`):** Contains `/`. Allowed: ALNUM + `. : / _ -`.
* **Pattern D — Number (`number`):** `0x[0-9a-f]+` or `[0-9]+`.
* **Pattern E — Word (`word`):** Longest run of ALNUM or `_`.
* All other characters are delimiters.

**Identifier decomposition (MUST):** For `symbol` and `word` terms, emit sub-terms by splitting on `_`, camelCase transitions (using pre-casefold string), and digit boundaries. Parts with length >= 2 emitted as `code`. Also emit compressed form (remove `_` and `-`).

**Stopword filtering (MUST):** Remove for `word` and `code` terms only. Set: `a an and are as at be by for from has have if in is it its of on or that the to was were with true false unit ret br cbr switch bb fn module imports exports digest const call call_void new`

**Term length limits:** Drop terms < 2 chars (except `diag`). Truncate > 64 chars.

**Query tokenization:** Same algorithm; stopword removal MAY be disabled if query contains only stopwords.

### Appendix F — BM25 Defaults (exact spec)

**Document model:** Each MMS item is a document. Document length `|D|` = number of retained terms.

**Scoring formula:**

```
score(D,Q) = Σ_{t in Q} IDF(t) * ((f(t,D) * (k1 + 1)) / (f(t,D) + k1 * (1 - b + b * |D|/avgdl)))
```

Where `f(t,D)` = term frequency, `avgdl` = average doc length.

**IDF (Robertson/Sparck Jones with +1 smoothing):**

```
IDF(t) = ln(1 + (N - df(t) + 0.5)/(df(t) + 0.5))
```

**Default parameters:** `k1 = 1.2`, `b = 0.75`

**Field boosts (multiplicative, applied after BM25):**

* `boost_kind`: diag_template=1.40, spec_excerpt=1.25, mpd_signature=1.20, symbol_capsule=1.15, test_case=1.10, repair_episode=1.05, default=1.0
* `boost_tags`: query contains exact diag code matching doc tag: x1.30; query contains module terms matching doc module: x1.15
* `boost_priority`: `x (0.5 + priority/100)` clamped to `[0.75, 1.50]`

**Deterministic tie-break:** (1) smaller token_cost, (2) lexicographically smaller item_id.

**Index persistence:** `index_lex/` stores: `vocab.bin`, `postings.bin`, `doclens.bin`, `itemmap.bin`, `bm25_meta.json` (N, avgdl, k1, b, tokenizer id, schema version).

### Appendix G — Context Pack Scoring and Selection (exact spec)

**Chunk types:** Core structural (module_header, mpd_public_api, symgraph_summary, deps_summary), Problem-focused (diagnostics, ownership_trace, cfg_summary), Code capsules (symbol_capsule, snippet), Retrieved (rag_item).

**Chunk IDs:** `chunk_id = "C:" + base32(blake3(kind + "|" + subject_id + "|" + variant + "|" + body_digest))[:16]`

**Scoring formula:**

```
score = base_priority(kind) + relevance(kind, scope) + proximity(kind, failing_sid) + retrieval_score(kind, MMS) - size_penalty(token_cost)
```

**base_priority:** module_header=100, mpd_public_api=90, symgraph_summary=85, diagnostics=80, ownership_trace=78, cfg_summary=72, symbol_capsule=70, snippet=60, rag_item=55, deps_summary=50.

**relevance:** symbol scope=+30, module=+20, files=+15, pkg=+10.

**proximity:** subject==failing_sid=+25, same module=+10, direct dep=+5, else=+0.

**retrieval_score:** rag_item only: +min(25, floor(mms_score)).

**size_penalty:** `token_cost / 200` (integer division).

**Tie-break:** (1) higher score, (2) lower token_cost, (3) smaller chunk_id.

**Budget partitioning policies:**

* `balanced`: 25% structural, 45% problem+capsules, 30% retrieved
* `diagnostics_first`: 30% structural, 60% problem+capsules, 10% retrieved
* `slices_first`: 35% structural, 55% capsules/snippets, 10% retrieved
* `minimal`: 60% structural, 40% problem, 0% retrieved

Spillover (default in `balanced`): structural -> problem -> retrieved.

**Multi-variant compression ladder:** v3 (full text) -> v2 (trimmed, signatures+key lines) -> v1 (signatures+bullets, <=20 lines) -> v0 (one-line identity: SID+name+type). Selection picks highest variant that fits budget.

**Selection algorithm:** Per bucket: build candidates, sort by score desc/token_cost asc/chunk_id asc, greedily select highest fitting variant. Merge bucket results in order: structural -> problem -> retrieved.

---

*End of specification.*
