#!/usr/bin/env bash
set -euo pipefail

# Benchmark runner for Magpie vs Rust vs TypeScript
# Measures: compilation time, execution time, memory usage, binary size

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_FILE="$SCRIPT_DIR/results.json"
RUNS=10  # Number of runs for timing

cd "$PROJECT_DIR"

echo "=== Magpie vs Rust vs TypeScript Benchmark ==="
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Runs per measurement: $RUNS"
echo ""

# --- Helper: median of sorted values ---
median() {
    local arr=("$@")
    local n=${#arr[@]}
    local mid=$((n / 2))
    if (( n % 2 == 0 )); then
        echo "scale=3; (${arr[$((mid-1))]} + ${arr[$mid]}) / 2" | bc
    else
        echo "${arr[$mid]}"
    fi
}

# --- Helper: time command in ms ---
time_cmd_ms() {
    local cmd="$1"
    local start end
    start=$(python3 -c "import time; print(int(time.time_ns()))")
    eval "$cmd" > /dev/null 2>&1
    end=$(python3 -c "import time; print(int(time.time_ns()))")
    echo "scale=3; ($end - $start) / 1000000" | bc
}

# --- Helper: peak RSS in KB ---
peak_rss_kb() {
    local cmd="$1"
    /usr/bin/time -l sh -c "$cmd" 2>&1 | grep "maximum resident set size" | awk '{print $1}'
}

echo "=== 1. Source Code Metrics ==="
python3 "$SCRIPT_DIR/count_tokens.py" \
    "$SCRIPT_DIR/benchmark.mp" \
    "$SCRIPT_DIR/benchmark.rs" \
    "$SCRIPT_DIR/benchmark.ts"
echo ""

echo "=== 2. Compilation Time (${RUNS} runs, milliseconds) ==="

# Clean artifacts
rm -f target/aarch64-apple-macos/dev/benchmark target/aarch64-apple-macos/dev/benchmark.ll
rm -f "$SCRIPT_DIR/benchmark_rs"
rm -f "$SCRIPT_DIR/benchmark.js"

# Magpie compilation times
echo "--- Magpie ---"
mp_times=()
for i in $(seq 1 $RUNS); do
    t=$(time_cmd_ms "./target/release/magpie --entry benchmarks/benchmark.mp --emit exe build")
    mp_times+=("$t")
done
IFS=$'\n' mp_sorted=($(sort -g <<<"${mp_times[*]}")); unset IFS
mp_median=$(median "${mp_sorted[@]}")
echo "  Magpie compilation: min=${mp_sorted[0]}ms median=${mp_median}ms max=${mp_sorted[-1]}ms"

# Rust compilation times
echo "--- Rust ---"
rs_times=()
for i in $(seq 1 $RUNS); do
    rm -f "$SCRIPT_DIR/benchmark_rs"
    t=$(time_cmd_ms "rustc -O -o $SCRIPT_DIR/benchmark_rs $SCRIPT_DIR/benchmark.rs")
    rs_times+=("$t")
done
IFS=$'\n' rs_sorted=($(sort -g <<<"${rs_times[*]}")); unset IFS
rs_median=$(median "${rs_sorted[@]}")
echo "  Rust compilation: min=${rs_sorted[0]}ms median=${rs_median}ms max=${rs_sorted[-1]}ms"

# TypeScript: tsc type-check time
echo "--- TypeScript (tsc --noEmit) ---"
ts_times=()
for i in $(seq 1 $RUNS); do
    t=$(time_cmd_ms "npx tsc --noEmit --strict $SCRIPT_DIR/benchmark.ts 2>/dev/null || true")
    ts_times+=("$t")
done
IFS=$'\n' ts_sorted=($(sort -g <<<"${ts_times[*]}")); unset IFS
ts_median=$(median "${ts_sorted[@]}")
echo "  TypeScript type-check: min=${ts_sorted[0]}ms median=${ts_median}ms max=${ts_sorted[-1]}ms"
echo ""

echo "=== 3. Execution Time (${RUNS} runs, milliseconds) ==="

# Ensure executables exist
./target/release/magpie --entry benchmarks/benchmark.mp --emit exe build > /dev/null 2>&1
rustc -O -o "$SCRIPT_DIR/benchmark_rs" "$SCRIPT_DIR/benchmark.rs" 2>/dev/null

echo "--- Magpie (native binary) ---"
mp_exec_times=()
for i in $(seq 1 $RUNS); do
    t=$(time_cmd_ms "./target/aarch64-apple-macos/dev/benchmark || true")
    mp_exec_times+=("$t")
done
IFS=$'\n' mp_exec_sorted=($(sort -g <<<"${mp_exec_times[*]}")); unset IFS
mp_exec_median=$(median "${mp_exec_sorted[@]}")
echo "  Magpie execution: min=${mp_exec_sorted[0]}ms median=${mp_exec_median}ms max=${mp_exec_sorted[-1]}ms"

echo "--- Rust (native binary) ---"
rs_exec_times=()
for i in $(seq 1 $RUNS); do
    t=$(time_cmd_ms "$SCRIPT_DIR/benchmark_rs || true")
    rs_exec_times+=("$t")
done
IFS=$'\n' rs_exec_sorted=($(sort -g <<<"${rs_exec_times[*]}")); unset IFS
rs_exec_median=$(median "${rs_exec_sorted[@]}")
echo "  Rust execution: min=${rs_exec_sorted[0]}ms median=${rs_exec_median}ms max=${rs_exec_sorted[-1]}ms"

echo "--- TypeScript (node) ---"
ts_exec_times=()
for i in $(seq 1 $RUNS); do
    t=$(time_cmd_ms "node $SCRIPT_DIR/benchmark.ts || true")
    ts_exec_times+=("$t")
done
IFS=$'\n' ts_exec_sorted=($(sort -g <<<"${ts_exec_times[*]}")); unset IFS
ts_exec_median=$(median "${ts_exec_sorted[@]}")
echo "  TypeScript execution: min=${ts_exec_sorted[0]}ms median=${ts_exec_median}ms max=${ts_exec_sorted[-1]}ms"
echo ""

echo "=== 4. Memory Usage (peak RSS, KB) ==="
mp_rss=$(peak_rss_kb "./target/aarch64-apple-macos/dev/benchmark || true")
rs_rss=$(peak_rss_kb "$SCRIPT_DIR/benchmark_rs || true")
ts_rss=$(peak_rss_kb "node $SCRIPT_DIR/benchmark.ts || true")
echo "  Magpie: ${mp_rss} KB"
echo "  Rust:   ${rs_rss} KB"
echo "  TypeScript: ${ts_rss} KB"
echo ""

echo "=== 5. Binary / Artifact Size ==="
mp_bin_size=$(stat -f%z target/aarch64-apple-macos/dev/benchmark 2>/dev/null || echo 0)
rs_bin_size=$(stat -f%z "$SCRIPT_DIR/benchmark_rs" 2>/dev/null || echo 0)
mp_ll_size=$(stat -f%z target/aarch64-apple-macos/dev/benchmark.ll 2>/dev/null || echo 0)
echo "  Magpie binary: ${mp_bin_size} bytes"
echo "  Magpie LLVM IR: ${mp_ll_size} bytes"
echo "  Rust binary: ${rs_bin_size} bytes"
echo "  TypeScript: N/A (interpreted)"
echo ""

echo "=== 6. Correctness Verification ==="
mp_exit=$(./target/aarch64-apple-macos/dev/benchmark 2>/dev/null; echo $?)
rs_out=$($SCRIPT_DIR/benchmark_rs 2>/dev/null; echo "EXIT:$?")
ts_out=$(node $SCRIPT_DIR/benchmark.ts 2>/dev/null; echo "EXIT:$?")
echo "  Magpie exit code: $mp_exit (expected: 70 = 4934 % 256)"
echo "  Rust output: $rs_out"
echo "  TypeScript output: $ts_out"
echo ""

echo "=== Benchmark Complete ==="
