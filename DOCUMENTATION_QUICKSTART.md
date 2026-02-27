# Magpie v0.1 Quickstart

This is the fast path into `DOCUMENTATION.md`.

- Full reference: [`DOCUMENTATION.md`](./DOCUMENTATION.md)
- Comprehensive skill guide: [`SKILL.md`](./SKILL.md)

---

## 1) Install/Build CLI

```bash
cargo build -p magpie_cli
cargo run -p magpie_cli -- --help
```

Important: CLI global flags (like `--entry`, `--emit`, `--output`) go **before** the subcommand.

---

## 2) Create a Project

```bash
cargo run -p magpie_cli -- new demo
cd demo
cargo run -p magpie_cli -- --output json build
```

---

## 3) Minimal Magpie Program

`src/main.mp`

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

Build/run:

```bash
cargo run -p magpie_cli -- --entry src/main.mp --emit mpir,llvm-ir,exe --output json build
cargo run -p magpie_cli -- --entry src/main.mp run
```

---

## 4) Fast Diagnostic Loop (Binary-only Friendly)

1. Build with machine output + debug artifacts:

```bash
cargo run -p magpie_cli -- --entry src/main.mp --emit mpir,llvm-ir,mpdbg --output json build
```

2. Explain code:

```bash
cargo run -p magpie_cli -- --output json explain MPT2014
```

3. Fix smallest issue first and rebuild.

---

## 5) Most Common Failure Families

- `MPP*` parse/io/artifact
- `MPS*` resolve/SSA/unsafe context
- `MPT*` type/trait/constructor rules
- `MPHIR*` HIR invariants
- `MPO*` ownership/borrow/move
- `MPL*` lint/link/LLM budget

Detailed bad/fix examples are in:
- `SKILL.md` section **20+**

---

## 6) Canonical Syntax Rules That Prevent Many Errors

- Header order must be:
  1. `module`
  2. `exports`
  3. `imports`
  4. `digest`
- Every block must end with a terminator.
- Use explicit key/value arg forms: `{ key=value }`.
- `setfield` uses key `val=` (not `value=`).
- `getfield`/`setfield` key order is strict.

---

## 7) Ownership Rules (Practical)

- Read fields through `borrow` / `mutborrow` handles.
- Mutate through `mutborrow` only.
- Don’t let borrows cross blocks.
- Don’t put borrows into `phi`.
- Don’t return borrow values.

---

## 8) TCallable Quick Pattern

```mp
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
  %r: i32 = call.indirect %mul_by_3 { args=[const.i32 7] }
  ret %r
}
```

---

## 9) High-Value Commands

### Format
```bash
cargo run -p magpie_cli -- fmt --fix-meta
```

### Parse only
```bash
cargo run -p magpie_cli -- --entry src/main.mp --output json parse
```

### Graphs
```bash
cargo run -p magpie_cli -- --entry src/main.mp --output json graph symbols
cargo run -p magpie_cli -- --entry src/main.mp --output json graph deps
cargo run -p magpie_cli -- --entry src/main.mp --output json graph ownership
cargo run -p magpie_cli -- --entry src/main.mp --output json graph cfg
```

### MPIR verify
```bash
cargo run -p magpie_cli -- --entry src/main.mp --output json mpir verify
```

### FFI import
```bash
cargo run -p magpie_cli -- --output json ffi import --header mylib.h --out ffi_bindings.mp
```

### Memory index/query
```bash
cargo run -p magpie_cli -- --entry src/main.mp --output json memory build
cargo run -p magpie_cli -- --entry src/main.mp --output json memory query -q "borrow phi" -k 10
```

---

## 10) LLM/Agent Mode Flags

```bash
cargo run -p magpie_cli -- \
  --entry src/main.mp \
  --llm \
  --llm-token-budget 12000 \
  --llm-budget-policy balanced \
  --output json \
  build
```

Useful:
- `--no-auto-fmt` (disable llm pre-format)
- `--llm-tokenizer approx:utf8_4chars`

---

## 11) Where to Read Next in DOCUMENTATION.md

- Language and grammar: sections **4–9**
- Semantics and safety: section **10**
- TCallable + ARC/ownership: sections **11–12**
- Compiler and pipeline: sections **13–15**
- Diagnostics + CLI args: sections **16–18**
- Formal grammar/opcode appendices: **Appendix A–D**
- Evidence matrix: **Appendix E–F**
