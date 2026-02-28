# Repository Guidelines

## Project Structure & Module Organization
Magpie is a Rust workspace (`Cargo.toml` at repo root) with compiler/tooling crates in `crates/`. Core flow is split across frontend crates (`magpie_lex`, `magpie_parse`, `magpie_ast`), analysis crates (`magpie_sema`, `magpie_hir`, `magpie_types`, `magpie_own`), and backend crates (`magpie_mpir`, `magpie_arc`, `magpie_codegen_llvm`, `magpie_codegen_wasm`). CLI entrypoint is `crates/magpie_cli/src/main.rs`.  
Pipeline orchestration is centered in `magpie_driver`; runtime and platform/tooling crates (for example `magpie_rt`, `magpie_pkg`, `magpie_memory`, `magpie_web`, `magpie_gpu`) are also workspace-critical. Use root `Cargo.toml` workspace members as the source of truth for full crate inventory.  
Language fixtures live in `tests/fixtures/*.mp`, integration tests in `tests/integration_test.rs`, and standard library modules in `std/std.*/*.mp`.

## Build, Test, and Development Commands
Use Cargo from repo root:

```bash
cargo build -p magpie_cli              # build CLI
cargo build --workspace                # build all crates
cargo run -p magpie_cli -- --help      # inspect CLI
cargo test --workspace                 # full test suite
cargo test --test integration_test     # fixture-driven integration tests
cargo test -p magpie_parse snapshot_hello_parser_output
```

For CLI usage, global flags must come before subcommands:

```bash
cargo run -p magpie_cli -- --entry tests/fixtures/hello.mp --emit mpir,llvm-ir build
```

Toolchain notes (important for backend/link workflows):
- `lli` is needed for common `magpie run` LLVM-IR execution flow.
- `clang` (and often `llc`) is needed for `--emit exe` / native linking.
- Missing native tools usually surface as link diagnostics such as `MPLINK01` / `MPLINK02`.
- In `--llm` mode, builds may auto-run formatting unless `--no-auto-fmt`; expect possible source rewrites when this mode is enabled.

CLI path resolution pitfall:
- `--entry` is resolved relative to the current working directory (CWD), not always the repo root.
- If a command fails to find sources, check CWD first before changing code.

## Coding Style & Naming Conventions
Workspace is Rust 2021 with MSRV 1.80. Follow standard Rust formatting (`cargo fmt`) and keep code idiomatic: `snake_case` for functions/tests, `PascalCase` for types, `SCREAMING_SNAKE_CASE` for constants.  
MSRV source of truth is root `Cargo.toml` (`workspace.package.rust-version`) if other docs drift.
Crate names follow `magpie_*`. Keep crate boundaries clean: avoid cross-crate leakage of responsibilities.  
For `.mp` sources, keep the header order strict: `module`, `exports`, `imports`, `digest`.
For user-visible/compiler-emitted output (diagnostics, JSON, graphs, symbol lists), preserve deterministic ordering and canonical forms; avoid introducing iteration-order nondeterminism.

## Testing Guidelines
Prefer focused unit tests in crate-local `#[cfg(test)]` modules and end-to-end behavior checks in `tests/integration_test.rs`.  
Parser output stability is validated with `insta` snapshots in `crates/magpie_parse/src/snapshots/`.  
When changing language behavior, update or add fixtures under `tests/fixtures/` and run both targeted crate tests and `cargo test --workspace`.

Verification matrix (run what matches your change):
- Parser/grammar/AST changes: `cargo test -p magpie_parse` and `cargo test --test integration_test`.
- Driver/codegen/emit changes: `cargo test -p magpie_driver`, plus a fixture build such as  
  `cargo run -p magpie_cli -- --entry tests/fixtures/feature_harness.mp --emit mpir,llvm-ir --output json build`.
- CLI behavior/flag parsing changes: `cargo run -p magpie_cli -- --help` and at least one end-to-end command with global flags before subcommands.
- `.mp` source edits in fixtures/std examples: run `cargo run -p magpie_cli -- fmt --fix-meta` before verification so digest/meta stay canonical.
- Parser snapshot-affecting changes: run parser snapshot tests and review snapshot updates under `crates/magpie_parse/src/snapshots/`.

Cross-crate impact checklist:
- If you change public types/contracts in a crate, run tests for likely dependents (at minimum `magpie_driver` and `magpie_cli`, plus directly affected crates).
- Do not assume a crate-local green test run is sufficient for workspace-level compiler changes.

Snapshot update policy:
- Treat snapshot diffs as code changes, not mechanical updates.
- In PR notes, state why changed snapshots are correct and what behavior changed.

Preferred staged verification (fast failure localization):
1. `parse` (syntax/header correctness)
2. `lint` (policy/quality)
3. `mpir verify` (IR invariants)
4. `build` with required emits

Diagnostic triage rule:
- Prefer `--output json` for machine-readable diagnostics.
- Use `cargo run -p magpie_cli -- explain <CODE>` for remediation details.
- Fix earliest-stage failures first (parse/resolution/type/ownership/backend) to avoid chasing downstream noise.

Runtime/FFI risk gate:
- Changes in runtime/FFI paths (`magpie_rt`, FFI import/link surfaces, ABI-facing code) require at least one integration fixture run and recorded verification evidence in the PR.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects, often with optional conventional prefixes (for example `fix(driver): ...`, `docs(benchmark): ...`).  
Prefer small, stage-focused PRs (parse/type/ownership/codegen/CLI) over wide multi-stage refactors when feasible.
PRs should include:
- a concise problem/solution summary,
- affected crates/fixtures,
- exact verification commands run,
- sample CLI output or diagnostics when behavior changes.

Documentation sync rule:
- If user-visible behavior changes (flags, diagnostics, command flow, emitted artifacts), update `README.md` and `DOCUMENTATION_QUICKSTART.md` in the same PR, or link a follow-up docs PR in the original PR description.

## Security & Configuration Tips
Do not commit generated artifacts or local caches (`target/`, `.magpie/`, `.omc/`, `.omx/`, `.cargo-home/`, `.cargo-local/`).
For `.mp` files, do not hand-edit `digest`; use `magpie fmt --fix-meta` as the source of truth for canonical metadata.
