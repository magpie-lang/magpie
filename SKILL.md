---
name: magpie-engineer
description: Comprehensive source-first workflow for writing Magpie code, changing the Magpie compiler/runtime, and diagnosing/fixing build or runtime failures. Use when tasks involve creating or editing .mp programs, implementing language/runtime features in Rust, triaging diagnostics (MPP/MPS/MPT/MPO/MPHIR/MPG/MPL*), reproducing crashes, and delivering validated fixes.
---

# Magpie Engineer Skill

Follow this skill when implementing or fixing anything in this repository.

## Source-of-truth policy

- Treat **source code and tests** as canonical truth.
- Do **not** rely on `SPEC.md` or `PLAN.md` for behavior.
- Prefer behavior validated by:
  - crate implementation,
  - unit/integration tests,
  - emitted artifacts (`.ll`, `.mpir`, `.mpdbg`, graphs),
  - executable runs.

## How Codex skills work (apply this while maintaining this skill)

- Keep YAML frontmatter minimal (`name`, `description`) because skill triggering depends on these fields.
- Put triggering conditions in `description`; put operational instructions in body.
- Keep instructions procedural and executable.
- Prefer deterministic workflows with concrete commands.
- Prefer progressive disclosure: start with core steps, then branch to specialized playbooks.

## Fast operating checklist

1. Classify task:
   - `.mp` authoring,
   - compiler pipeline bug,
   - codegen/link failure,
   - runtime crash,
   - web/MCP path.
2. Identify likely pipeline stage from diagnostics.
3. Reproduce with smallest case (fixture or isolated temp program).
4. Patch the owning crate only.
5. Run targeted tests first, then broader verification.
6. Report root cause + exact validations (commands + exit codes).

## Repository map (high-value crates)

- `crates/magpie_cli`: user-facing CLI and command routing.
- `crates/magpie_driver`: build pipeline orchestration, stage timings, artifact planning, diagnostic envelope.
- `crates/magpie_lex`: tokenization and opcode keywords.
- `crates/magpie_parse`: CSNF parser (`module/exports/imports/digest`, fn/block/op grammar).
- `crates/magpie_sema`: resolve/typechecking/lowering constraints.
- `crates/magpie_hir`: HIR validation (SSA + borrow invariants).
- `crates/magpie_own`: ownership checker and MPO diagnostics.
- `crates/magpie_mpir`: MPIR model + verifier.
- `crates/magpie_codegen_llvm`: LLVM IR lowering.
- `crates/magpie_rt`: runtime ABI, ARC, collections, strings, GPU runtime interfaces.
- `crates/magpie_gpu`: GPU kernel lowering/registry generation.
- `crates/magpie_web`: web/MCP service behavior.
- `tests/fixtures/*.mp`: canonical syntax/program examples.
- `tests/integration_test.rs`: parse/build fixture validation flow.

## Canonical commands

Use these as defaults:

- Build CLI/toolchain components:
  - `cargo build -p magpie_cli`
- Run CLI:
  - `cargo run -p magpie_cli -- <command> ...`
- Build a program:
  - `cargo run -p magpie_cli -- build --entry <path/to/main.mp> --emit <kinds>`
- Run a program:
  - `cargo run -p magpie_cli -- run --entry <path> [--emit ...]`
- Run tests:
  - `cargo test -p <crate>`
  - `cargo test --test integration_test`
  - `cargo test --workspace` (final sweep)

Useful emits:

- `mpir` for pipeline smoke checks without full native execution dependency.
- `llvm-ir` for debugging codegen and link/runtime interactions.
- `mpdbg` for machine-readable diagnostic bundles.
- graphs: `symgraph`, `depsgraph`, `ownershipgraph`, `cfggraph` for analysis.

## Build pipeline mental model (from driver)

Pipeline stages (in order):

1. package resolve
2. read/lex/parse
3. resolve
4. type/lower
5. async lowering
6. HIR verify
7. ownership
8. MPIR lower
9. MPIR verify
10. ARC insertion
11. ARC optimization
12. LLVM/GPU codegen
13. link
14. memory index update

Map every failure to one stage before patching.

## Writing `.mp` programs correctly

### Required file skeleton

Use this exact header order:

```mp
module <path>
exports { @main }
imports { }
digest "0000000000000000"
```

Then function definitions with explicit blocks:

```mp
fn @main() -> i64 {
bb0:
  %x: i64 = const.i64 1
  ret %x
}
```

### Core syntax rules

- Use SSA locals `%name: Type = op`.
- End every block with a terminator (`ret`, `br`, `cbr`, `switch`, `unreachable`).
- Prefer explicit named args in op calls.
- Keep op key ordering canonical even when parser can recover.

### Key ordering gotchas (important)

Use canonical keys/order for strict ops:

- `getfield { obj=..., field=... }`
- `setfield { obj=..., field=..., val=... }`
- GPU launch forms in declared key order (`device`, `kernel`, `grid`, `block`, `args`).

Do not use `value=` for surface `setfield`; use `val=`.

### Borrow/ownership-safe authoring pattern

- Mutate through `mutborrow` only.
- Read fields through `borrow`/`mutborrow` objects.
- Keep borrow scopes short and block-local.
- Avoid borrows in phi edges and across branch joins.
- Re-borrow after merges instead of carrying borrow handles through control-flow joins.

### Fixture-first style

When unsure, clone style from:

- `tests/fixtures/hello.mp`
- `tests/fixtures/arithmetic.mp`
- `tests/fixtures/collections.mp`
- `tests/fixtures/feature_harness.mp`

## Implementing language/runtime changes

### Add or modify an opcode end-to-end

Touch in this order:

1. Lexer keyword/token mapping (`magpie_lex`).
2. Parser op branch (`magpie_parse`).
3. Sema typing/lowering (`magpie_sema`).
4. HIR invariants if needed (`magpie_hir`).
5. Ownership rules (`magpie_own`).
6. MPIR mapping/verification (`magpie_mpir`).
7. Backend lowering (`magpie_codegen_llvm` and/or wasm/gpu).
8. Runtime ABI implementation (`magpie_rt`) if new runtime call is required.
9. Add/adjust tests at each layer.

### If changing runtime ABI

- Update runtime function implementation in `magpie_rt`.
- Ensure declaration parity in codegen (`magpie_codegen_llvm` extern declarations).
- Verify call-site argument sizes/types.
- Add regression tests for both codegen text and runtime behavior.

## Diagnostic triage playbook

Use codes first, then stage, then fix.

### Parse/read/artifact I/O

- `MPP0001`, `MPP0002`, `MPP0003`
- Actions:
  - verify file exists/paths,
  - fix syntax/header order,
  - check artifact output path permissions.

### Resolve/sema/type

- `MPS0001..0006`, `MPT*`, `MPF0001`, `MPS0024`, `MPS0025`
- Actions:
  - fix module/import names and conflicts,
  - align call signatures and types,
  - add required trait impl bindings,
  - respect unsafe context requirements,
  - add FFI return ownership attrs for rawptr.

### HIR invariants

- `MPHIR01`, `MPHIR02`, `MPHIR03`, plus SSA-related `MPS0001/2/3` at HIR verify time
- Actions:
  - borrow before `getfield`, mutborrow before `setfield`,
  - avoid returning borrows,
  - repair single-def/use-before-def/dominance violations.

### Ownership

- `MPO0003`, `MPO0004`, `MPO0007`, `MPO0011`, `MPO0101`, `MPO0102`, `MPO0103`, `MPO0201`
- Actions:
  - avoid storing borrows in aggregates,
  - avoid moving values still borrowed,
  - clone/share when aliasing is intended,
  - split control flow to prevent borrow-phi/cross-block violations,
  - for non-Dupable `map.get`, switch to contains/get_ref style where appropriate.

### MPIR/backend/link

- MPIR invariants: `MPS0010`, `MPS0008`, `MPS0009`, `MPS0012`, `MPS0014`, etc.
- Backend/link: `MPG*`, `MPL2021`, `MPLINK01`, `MPLINK02`, `MPL0002`, `MPL0001`
- Actions:
  - validate MPIR forms before codegen,
  - inspect emitted `.ll` for malformed IR,
  - confirm runtime/library link inputs,
  - verify requested emits actually exist.

## Error-response protocol (how to respond when failures happen)

Always respond in this structure:

1. **Stage + code**: “Failed in stage X with CODE”.
2. **Root cause**: one-sentence source-grounded explanation.
3. **Patch summary**: exact files/functions changed.
4. **Validation**: commands run + pass/fail + exit code.
5. **Residual risk**: what is not covered yet.
6. **Next action**: precise follow-up command.

Do not provide vague “fixed” claims without command evidence.

## Runtime crash isolation workflow

When binary crashes or exits unexpectedly:

1. Build with `--emit llvm-ir`.
2. Inspect generated `target/<triple>/<profile>/<module>.ll`.
3. Identify entry and init symbols.
4. If needed, create a wrapper `main` in a combined LLVM module calling:
   - `mp_rt_init`,
   - module init types function,
   - target function.
5. Compile with `clang` and runtime staticlib (`libmagpie_rt-*.a`).
6. Run binary, capture exit code.
7. Run lldb non-interactively for backtrace.
8. Minimize reproducer by pruning functions from harness until crash disappears.

Use `tests/fixtures/feature_harness.mp` as the initial broad reproducer.

## Codegen debugging checklist

- Verify runtime decl exists for every runtime call emitted.
- Verify struct/option/result LLVM types match emitted ops.
- For aggregate-typed values, avoid scalar-only instructions.
- Verify string constants are materialized as runtime `Str` objects (not null placeholders) before string ops.
- Verify explicit retains/releases where ownership model requires them.

## Quality gates before declaring completion

Minimum:

- Targeted crate tests for changed crates.
- Integration fixture test if language surface changed:
  - `cargo test --test integration_test`
- If codegen/runtime changed:
  - run codegen crate tests,
  - run at least one compiled program end-to-end (feature harness preferred).

Recommended final sweep:

- `cargo test --workspace`

## Patch discipline

- Make smallest correct fix.
- Preserve existing behavior outside failing path.
- Add regression tests for every bug fix.
- Avoid broad refactors while diagnosing correctness bugs.
- Keep messages and commits tied to one root cause each.

## Common high-value fix patterns

- Split one block into `bb0 -> bb1` to end mutable borrow before shared borrow.
- Replace ambiguous import usage with explicit module-qualified symbols.
- Convert unsafe-sensitive op use into `unsafe` context.
- Add missing runtime declaration + implementation pair.
- Add parser `expect_key` order alignment if formatter or generator drifted.
- Add codegen regression tests that assert exact IR snippets for bug boundaries.

## Definition of done

Consider a task done only when all are true:

- Reproducer fails before patch and passes after patch.
- New/updated tests cover the fix.
- Diagnostics (if any) are expected and explained.
- Commands and exit codes are recorded.
- No unrelated files were changed.
