//! Magpie CLI entry point (§5).

use clap::{Parser, Subcommand};
use magpie_ctx::{build_context_pack, BudgetPolicy, Chunk};
use magpie_diag::{
    canonical_json_encode, canonical_json_string, Diagnostic, OutputEnvelope, Severity,
};
use magpie_driver::{BuildProfile, BuildResult, DriverConfig, TestResult};
use magpie_memory::{query_bm25, validate_index_staleness, MmsIndex};
use magpie_pkg::{parse_manifest, read_lockfile, resolve_deps, write_lockfile};
use magpie_web::{create_mcp_server, handle_web_command, run_mcp_stdio, WebCommand};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Parser, Debug)]
#[command(
    name = "magpie",
    version = "0.1.0",
    about = "Magpie language toolchain"
)]
struct Cli {
    /// Output format
    #[arg(long, default_value = "text", value_parser = ["text", "json", "jsonl"])]
    output: String,

    /// Color mode
    #[arg(long, default_value = "auto", value_parser = ["auto", "always", "never"])]
    color: String,

    /// Log level
    #[arg(long, default_value = "warn", value_parser = ["error", "warn", "info", "debug", "trace"])]
    log_level: String,

    /// Build profile
    #[arg(long, default_value = "dev", value_parser = ["dev", "release", "custom"])]
    profile: String,

    /// Target triple
    #[arg(long)]
    target: Option<String>,

    /// Emit artifact types (comma-separated)
    #[arg(long)]
    emit: Option<String>,

    /// Entry source file path (overrides Magpie.toml [build].entry)
    #[arg(long)]
    entry: Option<String>,

    /// Cache directory
    #[arg(long)]
    cache_dir: Option<String>,

    /// Parallel jobs
    #[arg(long, short = 'j')]
    jobs: Option<u32>,

    /// Feature flags
    #[arg(long)]
    features: Option<String>,

    /// Disable default features
    #[arg(long)]
    no_default_features: bool,

    /// Offline mode
    #[arg(long)]
    offline: bool,

    /// LLM-optimized output
    #[arg(long)]
    llm: bool,

    /// Disable automatic formatting in --llm mode
    #[arg(long)]
    no_auto_fmt: bool,

    /// LLM token budget
    #[arg(long)]
    llm_token_budget: Option<u32>,

    /// LLM tokenizer
    #[arg(long)]
    llm_tokenizer: Option<String>,

    /// LLM budget policy
    #[arg(long, value_parser = ["balanced", "diagnostics_first", "slices_first", "minimal"])]
    llm_budget_policy: Option<String>,

    /// Maximum errors per pass
    #[arg(long, default_value = "20")]
    max_errors: u32,

    /// Use shared generics (vtable-based)
    #[arg(long)]
    shared_generics: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Create a new Magpie project
    New {
        /// Project name
        name: String,
    },
    /// Build the project
    Build,
    /// Build and run the project
    Run {
        /// Arguments to pass to the program
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },
    /// Start the REPL
    Repl,
    /// Format source files (CSNF)
    Fmt {
        /// Auto-generate missing meta blocks
        #[arg(long)]
        fix_meta: bool,
    },
    /// Parse entry source and emit AST debug artifact
    Parse {
        /// Parse emission kind
        #[arg(long, default_value = "ast", value_parser = ["ast"])]
        emit: String,
    },
    /// Run linter
    Lint,
    /// Run tests
    Test {
        /// Filter pattern
        #[arg(long)]
        filter: Option<String>,
    },
    /// Generate documentation
    Doc,
    /// Verify MPIR
    Mpir {
        #[command(subcommand)]
        subcmd: MpirSubcommand,
    },
    /// Explain a diagnostic code
    Explain {
        /// Diagnostic code (e.g., MPO0007)
        code: String,
    },
    /// Package manager
    Pkg {
        #[command(subcommand)]
        subcmd: PkgSubcommand,
    },
    /// Web framework commands
    Web {
        #[command(subcommand)]
        subcmd: WebSubcommand,
    },
    /// MCP server
    Mcp {
        #[command(subcommand)]
        subcmd: McpSubcommand,
    },
    /// Memory store commands
    Memory {
        #[command(subcommand)]
        subcmd: MemorySubcommand,
    },
    /// Context pack builder
    Ctx {
        #[command(subcommand)]
        subcmd: CtxSubcommand,
    },
    /// FFI import
    Ffi {
        #[command(subcommand)]
        subcmd: FfiSubcommand,
    },
    /// Graph outputs
    Graph {
        #[command(subcommand)]
        subcmd: GraphSubcommand,
    },
}

#[derive(Subcommand, Debug)]
enum MpirSubcommand {
    /// Verify MPIR correctness
    Verify,
}

#[derive(Subcommand, Debug)]
enum PkgSubcommand {
    /// Resolve dependencies
    Resolve,
    /// Add a dependency
    Add { name: String },
    /// Remove a dependency
    Remove { name: String },
    /// Show dependency tree
    Why { name: String },
}

#[derive(Subcommand, Debug)]
enum WebSubcommand {
    /// Start dev server with hot reload
    Dev,
    /// Build for production
    Build,
    /// Serve production build
    Serve,
}

#[derive(Subcommand, Debug)]
enum McpSubcommand {
    /// Start MCP server
    Serve,
}

#[derive(Subcommand, Debug)]
enum MemorySubcommand {
    /// Build/update MMS index
    Build,
    /// Query MMS
    Query {
        #[arg(long, short)]
        q: String,
        #[arg(long, short, default_value = "10")]
        k: u32,
    },
}

#[derive(Subcommand, Debug)]
enum CtxSubcommand {
    /// Generate context pack
    Pack,
}

#[derive(Subcommand, Debug)]
enum FfiSubcommand {
    /// Import C headers
    Import {
        #[arg(long)]
        header: String,
        #[arg(long)]
        out: String,
    },
}

#[derive(Subcommand, Debug)]
enum GraphSubcommand {
    /// Symbol graph
    Symbols,
    /// Dependency graph
    Deps,
    /// Ownership graph
    Ownership,
    /// CFG graph
    Cfg,
}

fn main() {
    let cli = Cli::parse();
    let output_mode = effective_output_mode(&cli);

    let exit_code = match &cli.command {
        Commands::New { name } => {
            let mut result = BuildResult::default();
            match magpie_driver::create_project(name) {
                Ok(()) => {
                    result.success = true;
                    result.artifacts.push(name.clone());
                }
                Err(err) => {
                    result.success = false;
                    result
                        .diagnostics
                        .push(error_diag("MPC0001", "project creation failed", err));
                }
            }

            let config = driver_config_from_cli(&cli, false);
            emit_command_output("new", &config, &result, output_mode);
            if result.success {
                0
            } else {
                1
            }
        }
        Commands::Build => {
            let config = build_driver_config(&cli);
            if let Some(exit_code) =
                run_llm_auto_fmt_precheck(&config, cli.no_auto_fmt, output_mode)
            {
                exit_code
            } else {
                let result = magpie_driver::build(&config);
                emit_command_output("build", &config, &result, output_mode);
                if result.success {
                    0
                } else {
                    1
                }
            }
        }
        Commands::Run { args } => {
            let config = run_driver_config(&cli);
            if let Some(exit_code) =
                run_llm_auto_fmt_precheck(&config, cli.no_auto_fmt, output_mode)
            {
                exit_code
            } else {
                let result = magpie_driver::build(&config);
                emit_command_output("run", &config, &result, output_mode);
                if !result.success {
                    1
                } else {
                    execute_run_artifact(&config, &result, args)
                }
            }
        }
        Commands::Fmt { fix_meta } => {
            let paths = collect_project_fmt_paths(Path::new("."));
            let result = magpie_driver::format_files(&paths, *fix_meta);
            let config = driver_config_from_cli(&cli, false);
            emit_command_output("fmt", &config, &result, output_mode);
            if result.success {
                0
            } else {
                1
            }
        }
        Commands::Parse { emit } => {
            let mut config = driver_config_from_cli(&cli, false);
            config.emit = vec![emit.clone()];
            let result = magpie_driver::parse_entry(&config);
            emit_command_output("parse", &config, &result, output_mode);
            if result.success {
                0
            } else {
                1
            }
        }
        Commands::Lint => {
            let config = driver_config_from_cli(&cli, false);
            let result = magpie_driver::lint(&config);
            emit_command_output("lint", &config, &result, output_mode);
            if result.success {
                0
            } else {
                1
            }
        }
        Commands::Test { filter } => {
            let config = test_driver_config(&cli);
            let test_result = magpie_driver::run_tests(&config, filter.as_deref());
            let discovery_only_mode = !config.emit.iter().any(|kind| kind == "exe");

            match output_mode {
                OutputMode::Text => print_test_result(&test_result, discovery_only_mode),
                OutputMode::Json | OutputMode::Jsonl => {
                    let result = build_result_from_test_result(
                        &test_result,
                        filter.as_deref(),
                        discovery_only_mode,
                    );
                    emit_command_output("test", &config, &result, output_mode);
                }
            }

            if test_result.failed == 0 {
                0
            } else {
                1
            }
        }
        Commands::Doc => {
            let config = driver_config_from_cli(&cli, false);
            let mut doc_paths = Vec::new();
            collect_doc_paths(Path::new("src"), &mut doc_paths);
            if doc_paths.is_empty() {
                doc_paths.push(Path::new(&config.entry_path).to_path_buf());
            }
            doc_paths.sort();
            doc_paths.dedup();
            let paths = doc_paths
                .iter()
                .map(|path| path.to_string_lossy().to_string())
                .collect::<Vec<_>>();
            let result = magpie_driver::generate_docs(&paths);
            emit_command_output("doc", &config, &result, output_mode);
            if result.success {
                0
            } else {
                1
            }
        }
        Commands::Explain { code } => {
            let mut result = BuildResult::default();
            match magpie_diag::explain_code(code) {
                Some(explanation) => {
                    result.success = true;
                    result.diagnostics.push(info_diag(
                        code.clone(),
                        "diagnostic explanation",
                        explanation,
                    ));
                }
                None => {
                    result.success = false;
                    result.diagnostics.push(error_diag(
                        code.clone(),
                        "unknown diagnostic code",
                        "No explanation is available for this code.",
                    ));
                }
            }

            let config = driver_config_from_cli(&cli, false);
            emit_command_output("explain", &config, &result, output_mode);
            if result.success {
                0
            } else {
                1
            }
        }
        Commands::Graph { subcmd } => {
            let mut config = build_driver_config(&cli);
            config.emit = vec![graph_emit_kind(subcmd).to_string()];
            let result = magpie_driver::build(&config);
            let command = match subcmd {
                GraphSubcommand::Symbols => "graph.symbols",
                GraphSubcommand::Deps => "graph.deps",
                GraphSubcommand::Ownership => "graph.ownership",
                GraphSubcommand::Cfg => "graph.cfg",
            };
            emit_command_output(command, &config, &result, output_mode);
            if result.success {
                0
            } else {
                1
            }
        }
        Commands::Ffi { subcmd } => match subcmd {
            FfiSubcommand::Import { header, out } => {
                let result = magpie_driver::import_c_header(header, out);
                let config = driver_config_from_cli(&cli, false);
                emit_command_output("ffi.import", &config, &result, output_mode);
                if result.success {
                    0
                } else {
                    1
                }
            }
        },
        Commands::Repl => match magpie_jit::run_repl() {
            Ok(()) => 0,
            Err(err) => {
                eprintln!("REPL failed: {err}");
                1
            }
        },
        Commands::Mpir { subcmd } => match subcmd {
            MpirSubcommand::Verify => {
                let mut config = build_driver_config(&cli);
                if !config.emit.iter().any(|kind| kind == "mpir") {
                    config.emit.push("mpir".to_string());
                }
                let result = magpie_driver::build(&config);
                emit_command_output("mpir.verify", &config, &result, output_mode);
                if result.success {
                    0
                } else {
                    1
                }
            }
        },
        Commands::Pkg { subcmd } => {
            let config = driver_config_from_cli(&cli, false);
            let result = handle_pkg_command(subcmd, cli.offline);
            let command = match subcmd {
                PkgSubcommand::Resolve => "pkg.resolve",
                PkgSubcommand::Add { .. } => "pkg.add",
                PkgSubcommand::Remove { .. } => "pkg.remove",
                PkgSubcommand::Why { .. } => "pkg.why",
            };
            emit_command_output(command, &config, &result, output_mode);
            if result.success {
                0
            } else {
                1
            }
        }
        Commands::Web { subcmd } => {
            let config = driver_config_from_cli(&cli, false);
            let web_cmd = match subcmd {
                WebSubcommand::Dev => WebCommand::Dev,
                WebSubcommand::Build => WebCommand::Build,
                WebSubcommand::Serve => WebCommand::Serve,
            };
            let mut result = BuildResult::default();
            match handle_web_command(web_cmd, Path::new(".")) {
                Ok(()) => {
                    result.success = true;
                    result
                        .artifacts
                        .push(".magpie/gen/webapp_routes.mp".to_string());
                }
                Err(err) => {
                    result.success = false;
                    result
                        .diagnostics
                        .push(error_diag("MPW0001", "web command failed", err));
                }
            }
            let command = match subcmd {
                WebSubcommand::Dev => "web.dev",
                WebSubcommand::Build => "web.build",
                WebSubcommand::Serve => "web.serve",
            };
            emit_command_output(command, &config, &result, output_mode);
            if result.success {
                0
            } else {
                1
            }
        }
        Commands::Mcp { subcmd } => match subcmd {
            McpSubcommand::Serve => {
                let server = create_mcp_server();
                run_mcp_stdio(&server);
                0
            }
        },
        Commands::Memory { subcmd } => {
            let config = build_driver_config(&cli);
            let result = match subcmd {
                MemorySubcommand::Build => magpie_driver::build(&config),
                MemorySubcommand::Query { q, k } => handle_memory_query(&config, q, *k as usize),
            };
            let command = match subcmd {
                MemorySubcommand::Build => "memory.build",
                MemorySubcommand::Query { .. } => "memory.query",
            };
            emit_command_output(command, &config, &result, output_mode);
            if result.success {
                0
            } else {
                1
            }
        }
        Commands::Ctx { subcmd } => {
            let config = build_driver_config(&cli);
            let result = match subcmd {
                CtxSubcommand::Pack => handle_ctx_pack(&config),
            };
            emit_command_output("ctx.pack", &config, &result, output_mode);
            if result.success {
                0
            } else {
                1
            }
        }
    };

    std::process::exit(exit_code);
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum OutputMode {
    Text,
    Json,
    Jsonl,
}

fn effective_output_mode(cli: &Cli) -> OutputMode {
    if cli.output == "jsonl" {
        OutputMode::Jsonl
    } else if cli.llm || cli.output == "json" {
        OutputMode::Json
    } else {
        OutputMode::Text
    }
}

fn build_driver_config(cli: &Cli) -> DriverConfig {
    driver_config_from_cli(cli, false)
}

fn test_driver_config(cli: &Cli) -> DriverConfig {
    let mut config = driver_config_from_cli(cli, true);
    if cli.emit.is_none() {
        config.emit = vec!["exe".to_string()];
    }
    config
}

fn run_driver_config(cli: &Cli) -> DriverConfig {
    let mut config = build_driver_config(cli);
    if cli.emit.is_none() {
        // Keep `run` zero-config: release runs native executables, dev runs via LLVM IR.
        config.emit = if matches!(config.profile, BuildProfile::Release) {
            vec!["exe".to_string()]
        } else {
            vec!["llvm-ir".to_string()]
        };
    }
    config
}

fn graph_emit_kind(subcmd: &GraphSubcommand) -> &'static str {
    match subcmd {
        GraphSubcommand::Symbols => "symgraph",
        GraphSubcommand::Deps => "depsgraph",
        GraphSubcommand::Ownership => "ownershipgraph",
        GraphSubcommand::Cfg => "cfggraph",
    }
}

fn run_llm_auto_fmt_precheck(
    config: &DriverConfig,
    no_auto_fmt: bool,
    output_mode: OutputMode,
) -> Option<i32> {
    // In LLM mode we auto-format first so downstream output stays deterministic for model consumption.
    if config.llm_mode && !no_auto_fmt {
        let paths = collect_source_paths_for_fmt(&config.entry_path);
        let fmt_result = magpie_driver::format_files(&paths, true);
        if !fmt_result.success {
            emit_command_output("fmt.auto", config, &fmt_result, output_mode);
            return Some(1);
        }
    }
    None
}

#[derive(Clone, Debug)]
struct ResolvedLlmSettings {
    mode: bool,
    token_budget: Option<u32>,
    tokenizer: String,
    policy: String,
}

fn resolve_llm_settings(cli: &Cli) -> ResolvedLlmSettings {
    let manifest_llm = load_manifest_llm_defaults();
    let env_llm_mode = std::env::var("MAGPIE_LLM")
        .ok()
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false);
    let env_token_budget = std::env::var("MAGPIE_LLM_TOKEN_BUDGET")
        .ok()
        .and_then(|value| value.parse::<u32>().ok());

    let mode = cli.llm || env_llm_mode || manifest_llm.mode_default.unwrap_or(false);
    let token_budget = cli
        .llm_token_budget
        .or(env_token_budget)
        .or(manifest_llm.token_budget)
        .or_else(|| mode.then_some(magpie_driver::DEFAULT_LLM_TOKEN_BUDGET));
    let tokenizer = cli
        .llm_tokenizer
        .clone()
        .or(manifest_llm.tokenizer)
        .unwrap_or_else(|| magpie_driver::DEFAULT_LLM_TOKENIZER.to_string());
    let policy = cli
        .llm_budget_policy
        .clone()
        .or(manifest_llm.policy)
        .unwrap_or_else(|| magpie_driver::DEFAULT_LLM_BUDGET_POLICY.to_string());

    ResolvedLlmSettings {
        mode,
        token_budget,
        tokenizer,
        policy,
    }
}

#[derive(Clone, Debug, Default)]
struct ManifestLlmDefaults {
    mode_default: Option<bool>,
    token_budget: Option<u32>,
    tokenizer: Option<String>,
    policy: Option<String>,
}

#[derive(Clone, Debug, Default)]
struct ManifestBuildDefaults {
    entry_path: Option<String>,
}

fn find_manifest_path() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let candidate = dir.join("Magpie.toml");
        if candidate.is_file() {
            return Some(candidate);
        }
        if !dir.pop() {
            break;
        }
    }
    None
}

fn load_manifest_build_defaults() -> ManifestBuildDefaults {
    let Some(path) = find_manifest_path() else {
        return ManifestBuildDefaults::default();
    };
    let Ok(manifest) = parse_manifest(&path) else {
        return ManifestBuildDefaults::default();
    };

    let entry_raw = manifest.build.entry.trim();
    if entry_raw.is_empty() {
        return ManifestBuildDefaults::default();
    }

    let entry = if Path::new(entry_raw).is_absolute() {
        PathBuf::from(entry_raw)
    } else {
        path.parent()
            .unwrap_or_else(|| Path::new("."))
            .join(entry_raw)
    };

    ManifestBuildDefaults {
        entry_path: Some(entry.to_string_lossy().to_string()),
    }
}

fn load_manifest_llm_defaults() -> ManifestLlmDefaults {
    let Some(path) = find_manifest_path() else {
        return ManifestLlmDefaults::default();
    };
    let Ok(manifest) = parse_manifest(&path) else {
        return ManifestLlmDefaults::default();
    };
    let Some(llm) = manifest.llm else {
        return ManifestLlmDefaults::default();
    };

    ManifestLlmDefaults {
        mode_default: llm.mode_default,
        token_budget: llm.token_budget.map(|value| value as u32),
        tokenizer: llm.tokenizer,
        policy: llm.budget_policy,
    }
}

fn driver_config_from_cli(cli: &Cli, test_mode: bool) -> DriverConfig {
    let mut config = DriverConfig::default();
    let manifest_build = load_manifest_build_defaults();

    config.profile = match cli.profile.as_str() {
        "release" => BuildProfile::Release,
        _ => BuildProfile::Dev,
    };
    if let Some(entry) = &cli.entry {
        config.entry_path = entry.clone();
    } else if let Some(entry) = manifest_build.entry_path {
        config.entry_path = entry;
    }
    if let Some(target) = &cli.target {
        config.target_triple = target.clone();
    }
    if let Some(emit) = &cli.emit {
        let emit_items = parse_csv(emit);
        if !emit_items.is_empty() {
            config.emit = emit_items;
        }
    }
    config.cache_dir = cli.cache_dir.clone();
    config.jobs = cli.jobs;
    config.offline = cli.offline;
    config.no_default_features = cli.no_default_features;

    config.max_errors = cli.max_errors as usize;
    let llm = resolve_llm_settings(cli);
    config.llm_mode = llm.mode;
    config.token_budget = llm.token_budget;
    config.llm_tokenizer = Some(llm.tokenizer);
    config.llm_budget_policy = Some(llm.policy);
    config.shared_generics = cli.shared_generics;
    config.features = cli.features.as_deref().map(parse_csv).unwrap_or_default();

    if test_mode && !config.features.iter().any(|feature| feature == "test") {
        config.features.push("test".to_string());
    }

    config
}

fn execute_run_artifact(config: &DriverConfig, result: &BuildResult, extra_args: &[String]) -> i32 {
    let emit_exe = config.emit.iter().any(|kind| kind == "exe");
    if emit_exe {
        if let Some(path) = find_executable_artifact(config, &result.artifacts) {
            return execute_binary(path, extra_args);
        }
        eprintln!("Error: build produced no executable artifact");
        return 1;
    }

    let ll_path = find_llvm_ir_artifact(&result.artifacts);
    if let Some(path) = ll_path {
        return execute_with_lli(path, extra_args);
    }

    eprintln!("Error: build produced no runnable artifacts");
    eprintln!("Hint: use --emit llvm-ir (default for run) or --emit exe");
    1
}

fn find_executable_artifact<'a>(config: &DriverConfig, artifacts: &'a [String]) -> Option<&'a str> {
    let is_windows = config.target_triple.contains("windows");
    artifacts.iter().find_map(|artifact| {
        let path = Path::new(artifact);
        if is_windows {
            (path.extension().and_then(|ext| ext.to_str()) == Some("exe"))
                .then_some(artifact.as_str())
        } else {
            path.extension().is_none().then_some(artifact.as_str())
        }
    })
}

fn find_llvm_ir_artifact(artifacts: &[String]) -> Option<&str> {
    artifacts
        .iter()
        .find(|artifact| artifact.ends_with(".ll"))
        .map(String::as_str)
}

fn execute_binary(path: &str, extra_args: &[String]) -> i32 {
    match Command::new(path).args(extra_args).status() {
        Ok(status) => status.code().unwrap_or(1),
        Err(err) => {
            eprintln!("Error: could not execute binary '{}': {}", path, err);
            1
        }
    }
}

fn execute_with_lli(path: &str, extra_args: &[String]) -> i32 {
    let mut cmd = Command::new("lli");

    // Load the runtime shared library if available, otherwise try the static lib
    if let Some(dylib) = find_runtime_dylib() {
        cmd.arg("-load").arg(&dylib);
    }

    // Also load any GPU registry .ll files alongside the main .ll
    let main_path = Path::new(path);
    if let Some(parent) = main_path.parent() {
        if let Some(stem) = main_path.file_stem().and_then(|s| s.to_str()) {
            let gpu_registry = parent.join(format!("{}.gpu_registry.ll", stem));
            if gpu_registry.exists() {
                cmd.arg("-extra-module").arg(&gpu_registry);
            }
        }
    }

    cmd.arg(path).args(extra_args);

    match cmd.status() {
        Ok(status) => status.code().unwrap_or(1),
        Err(err) => {
            eprintln!("Error: could not execute program: {}", err);
            eprintln!("Hint: install LLVM tools (lli) or use --emit exe");
            1
        }
    }
}

fn find_runtime_dylib() -> Option<String> {
    let search_paths = [
        "target/debug",
        "target/release",
    ];
    let dylib_names = if cfg!(target_os = "macos") {
        vec!["libmagpie_rt.dylib"]
    } else if cfg!(target_os = "windows") {
        vec!["magpie_rt.dll"]
    } else {
        vec!["libmagpie_rt.so"]
    };

    for dir in &search_paths {
        for name in &dylib_names {
            let path = Path::new(dir).join(name);
            if path.exists() {
                return Some(path.to_string_lossy().to_string());
            }
        }
    }
    None
}

fn parse_csv(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(str::to_string)
        .collect()
}

fn collect_doc_paths(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut paths = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .collect::<Vec<_>>();
    paths.sort();

    for path in paths {
        if path.is_dir() {
            collect_doc_paths(&path, out);
            continue;
        }
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext == "mp")
        {
            out.push(path);
        }
    }
}

fn collect_source_paths_for_fmt(entry_path: &str) -> Vec<String> {
    let mut paths = Vec::new();
    let entry = Path::new(entry_path);
    if entry.is_file() {
        paths.push(entry.to_path_buf());
    }
    collect_doc_paths(Path::new("src"), &mut paths);
    collect_doc_paths(Path::new("tests"), &mut paths);
    paths.sort();
    paths.dedup();
    if paths.is_empty() {
        paths.push(PathBuf::from(entry_path));
    }
    paths
        .into_iter()
        .map(|path| path.to_string_lossy().to_string())
        .collect()
}

fn manifest_path_or_default() -> PathBuf {
    find_manifest_path().unwrap_or_else(|| PathBuf::from("Magpie.toml"))
}

fn handle_pkg_command(subcmd: &PkgSubcommand, offline: bool) -> BuildResult {
    match subcmd {
        PkgSubcommand::Resolve => {
            let mut result = BuildResult::default();
            let manifest_path = manifest_path_or_default();
            let manifest = match parse_manifest(&manifest_path) {
                Ok(manifest) => manifest,
                Err(err) => {
                    result.success = false;
                    result
                        .diagnostics
                        .push(error_diag("MPK0001", "manifest parse failed", err));
                    return result;
                }
            };
            match resolve_deps(&manifest, offline) {
                Ok(lock) => {
                    let lock_path = manifest_path
                        .parent()
                        .unwrap_or_else(|| Path::new("."))
                        .join("Magpie.lock");
                    match write_lockfile(&lock, &lock_path) {
                        Ok(()) => {
                            result.success = true;
                            result
                                .artifacts
                                .push(lock_path.to_string_lossy().to_string());
                            result.diagnostics.push(info_diag(
                                "MPK0000",
                                "dependencies resolved",
                                format!("Resolved {} package(s).", lock.packages.len()),
                            ));
                        }
                        Err(err) => {
                            result.success = false;
                            result.diagnostics.push(error_diag(
                                "MPK0002",
                                "lockfile write failed",
                                err,
                            ));
                        }
                    }
                }
                Err(err) => {
                    result.success = false;
                    result.diagnostics.push(error_diag(
                        "MPK0003",
                        "dependency resolution failed",
                        err,
                    ));
                }
            }
            result
        }
        PkgSubcommand::Add { name } => update_manifest_dependency(name, true),
        PkgSubcommand::Remove { name } => update_manifest_dependency(name, false),
        PkgSubcommand::Why { name } => {
            let mut result = BuildResult::default();
            let lock_path = manifest_path_or_default()
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("Magpie.lock");
            let lock = match read_lockfile(&lock_path) {
                Ok(lock) => lock,
                Err(err) => {
                    result.success = false;
                    result
                        .diagnostics
                        .push(error_diag("MPK0004", "lockfile read failed", err));
                    return result;
                }
            };

            let mut reasons = Vec::new();
            for pkg in &lock.packages {
                for dep in &pkg.deps {
                    if dep.name == *name {
                        reasons.push(format!("{} -> {}", pkg.name, dep.name));
                    }
                }
            }
            reasons.sort();
            reasons.dedup();

            result.success = true;
            if reasons.is_empty() {
                result.diagnostics.push(info_diag(
                    "MPK0000",
                    "dependency reason",
                    format!(
                        "No reverse dependency entries found for '{}' in '{}'.",
                        name,
                        lock_path.display()
                    ),
                ));
            } else {
                result.diagnostics.push(info_diag(
                    "MPK0000",
                    "dependency reason tree",
                    reasons.join(", "),
                ));
            }
            result
                .artifacts
                .push(lock_path.to_string_lossy().to_string());
            result
        }
    }
}

fn update_manifest_dependency(name: &str, add: bool) -> BuildResult {
    let mut result = BuildResult::default();
    if name.trim().is_empty() {
        result.success = false;
        result.diagnostics.push(error_diag(
            "MPK0005",
            "invalid dependency name",
            "Dependency name cannot be empty.",
        ));
        return result;
    }

    let manifest_path = manifest_path_or_default();
    let manifest_raw = match fs::read_to_string(&manifest_path) {
        Ok(raw) => raw,
        Err(err) => {
            result.success = false;
            result.diagnostics.push(error_diag(
                "MPK0001",
                "manifest read failed",
                format!("Could not read '{}': {}", manifest_path.display(), err),
            ));
            return result;
        }
    };
    let mut root_value = match manifest_raw.parse::<toml::Value>() {
        Ok(value) => value,
        Err(err) => {
            result.success = false;
            result.diagnostics.push(error_diag(
                "MPK0001",
                "manifest parse failed",
                format!("Could not parse '{}': {}", manifest_path.display(), err),
            ));
            return result;
        }
    };

    let Some(root_table) = root_value.as_table_mut() else {
        result.success = false;
        result.diagnostics.push(error_diag(
            "MPK0001",
            "manifest shape invalid",
            "Manifest root must be a TOML table.",
        ));
        return result;
    };
    let deps_value = root_table
        .entry("dependencies".to_string())
        .or_insert_with(|| toml::Value::Table(toml::map::Map::new()));
    let Some(deps_table) = deps_value.as_table_mut() else {
        result.success = false;
        result.diagnostics.push(error_diag(
            "MPK0001",
            "manifest dependencies invalid",
            "'dependencies' must be a TOML table.",
        ));
        return result;
    };

    if add {
        let dep_value = toml::Value::Table(toml::map::Map::from_iter([(
            "version".to_string(),
            toml::Value::String("^0.1".to_string()),
        )]));
        deps_table.insert(name.to_string(), dep_value);
    } else {
        deps_table.remove(name);
    }

    let encoded = match toml::to_string_pretty(&root_value) {
        Ok(encoded) => encoded,
        Err(err) => {
            result.success = false;
            result.diagnostics.push(error_diag(
                "MPK0006",
                "manifest serialization failed",
                err.to_string(),
            ));
            return result;
        }
    };
    match fs::write(&manifest_path, encoded) {
        Ok(()) => {
            result.success = true;
            result
                .artifacts
                .push(manifest_path.to_string_lossy().to_string());
            result.diagnostics.push(info_diag(
                "MPK0000",
                if add {
                    "dependency added"
                } else {
                    "dependency removed"
                },
                format!(
                    "Dependency '{}' {}.",
                    name,
                    if add { "was added" } else { "was removed" }
                ),
            ));
        }
        Err(err) => {
            result.success = false;
            result.diagnostics.push(error_diag(
                "MPK0007",
                "manifest write failed",
                format!("Could not write '{}': {}", manifest_path.display(), err),
            ));
        }
    }
    result
}

fn handle_memory_query(config: &DriverConfig, query: &str, k: usize) -> BuildResult {
    let mut result = BuildResult::default();
    let Some(index_path) = find_latest_mms_index() else {
        result.success = false;
        result.diagnostics.push(error_diag(
            "MPM1000",
            "memory index not found",
            "No MMS index found. Run `magpie memory build` first.",
        ));
        return result;
    };

    let raw = match fs::read_to_string(&index_path) {
        Ok(raw) => raw,
        Err(err) => {
            result.success = false;
            result.diagnostics.push(error_diag(
                "MPM1001",
                "memory index read failed",
                format!("Could not read '{}': {}", index_path.display(), err),
            ));
            return result;
        }
    };
    let index = match serde_json::from_str::<MmsIndex>(&raw) {
        Ok(index) => index,
        Err(err) => {
            result.success = false;
            result.diagnostics.push(error_diag(
                "MPM1002",
                "memory index parse failed",
                format!("Could not parse '{}': {}", index_path.display(), err),
            ));
            return result;
        }
    };

    let base_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let stale_issues = validate_index_staleness(&index, &base_dir, 8);
    if !stale_issues.is_empty() {
        let stale_count = stale_issues.len();
        let sample = stale_issues
            .first()
            .map(|issue| format!("{} ({})", issue.path, issue.reason))
            .unwrap_or_else(|| "unknown artifact".to_string());
        result.success = false;
        result.diagnostics.push(error_diag(
            "MPM1003",
            "memory index is stale",
            format!(
                "Detected {stale_count} stale MMS artifact(s), first: {sample}. Run `magpie memory build` to refresh the index."
            ),
        ));
        return result;
    }

    let hits = query_bm25(&index, query, k.max(1));
    result.success = true;
    result
        .artifacts
        .push(index_path.to_string_lossy().to_string());
    result.diagnostics.push(info_diag(
        "MPM0000",
        "memory query results",
        format!("query='{}' returned {} hit(s)", query, hits.len()),
    ));
    for hit in hits {
        result.diagnostics.push(info_diag(
            "MPM0000",
            format!("memory hit {}", hit.item_id),
            format!(
                "score={:.4} token_cost={} fqn={}",
                hit.score, hit.token_cost, hit.item.fqn
            ),
        ));
    }
    let _ = config;
    result
}

fn handle_ctx_pack(config: &DriverConfig) -> BuildResult {
    let mut result = BuildResult::default();
    let Some(index_path) = find_latest_mms_index() else {
        result.success = false;
        result.diagnostics.push(error_diag(
            "MPCX001",
            "memory index not found",
            "No MMS index found. Run `magpie memory build` first.",
        ));
        return result;
    };

    let raw = match fs::read_to_string(&index_path) {
        Ok(raw) => raw,
        Err(err) => {
            result.success = false;
            result.diagnostics.push(error_diag(
                "MPCX002",
                "memory index read failed",
                format!("Could not read '{}': {}", index_path.display(), err),
            ));
            return result;
        }
    };
    let index = match serde_json::from_str::<MmsIndex>(&raw) {
        Ok(index) => index,
        Err(err) => {
            result.success = false;
            result.diagnostics.push(error_diag(
                "MPCX003",
                "memory index parse failed",
                format!("Could not parse '{}': {}", index_path.display(), err),
            ));
            return result;
        }
    };

    let base_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let stale_issues = validate_index_staleness(&index, &base_dir, 8);
    if !stale_issues.is_empty() {
        let stale_count = stale_issues.len();
        let sample = stale_issues
            .first()
            .map(|issue| format!("{} ({})", issue.path, issue.reason))
            .unwrap_or_else(|| "unknown artifact".to_string());
        result.success = false;
        result.diagnostics.push(error_diag(
            "MPCX007",
            "memory index is stale",
            format!(
                "Detected {stale_count} stale MMS artifact(s), first: {sample}. Run `magpie memory build` to refresh the index."
            ),
        ));
        return result;
    }

    let chunks = index
        .items
        .iter()
        .map(|item| Chunk {
            chunk_id: item.item_id.clone(),
            kind: item.kind.clone(),
            subject_id: item.sid.clone(),
            body: item.text.clone(),
            token_cost: item
                .token_cost
                .get(magpie_driver::DEFAULT_LLM_TOKENIZER)
                .copied()
                .unwrap_or(0),
            score: item.priority as f64,
        })
        .collect::<Vec<_>>();
    let budget = config
        .token_budget
        .unwrap_or(magpie_driver::DEFAULT_LLM_TOKEN_BUDGET);
    let policy = parse_budget_policy(config.llm_budget_policy.as_deref());
    let pack = build_context_pack(chunks, budget, policy);
    let payload = match canonical_json_encode(&pack) {
        Ok(payload) => payload,
        Err(err) => {
            result.success = false;
            result.diagnostics.push(error_diag(
                "MPCX004",
                "context pack serialization failed",
                err.to_string(),
            ));
            return result;
        }
    };

    let out_path = Path::new(".magpie").join("ctx").join("pack.json");
    if let Some(parent) = out_path.parent() {
        if let Err(err) = fs::create_dir_all(parent) {
            result.success = false;
            result.diagnostics.push(error_diag(
                "MPCX005",
                "context pack directory creation failed",
                format!("Could not create '{}': {}", parent.display(), err),
            ));
            return result;
        }
    }
    match fs::write(&out_path, payload) {
        Ok(()) => {
            result.success = true;
            result
                .artifacts
                .push(out_path.to_string_lossy().to_string());
            result.diagnostics.push(info_diag(
                "MPCX000",
                "context pack generated",
                format!("Generated context pack with {} chunks.", pack.chunks.len()),
            ));
        }
        Err(err) => {
            result.success = false;
            result.diagnostics.push(error_diag(
                "MPCX006",
                "context pack write failed",
                format!("Could not write '{}': {}", out_path.display(), err),
            ));
        }
    }
    result
}

fn collect_project_fmt_paths(root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    collect_project_fmt_paths_recursive(root, &mut out);
    out.sort();
    out.dedup();
    out
}

fn collect_project_fmt_paths_recursive(dir: &Path, out: &mut Vec<String>) {
    let Ok(read_dir) = fs::read_dir(dir) else {
        return;
    };

    let mut entries = read_dir.filter_map(|entry| entry.ok()).collect::<Vec<_>>();
    entries.sort_by_key(|a| a.file_name());

    for entry in entries {
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(_) => continue,
        };

        if file_type.is_dir() {
            let skip_dir = path
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name| {
                    matches!(
                        name,
                        ".git" | ".hg" | ".svn" | ".magpie" | "target" | "node_modules"
                    )
                })
                .unwrap_or(false);
            if !skip_dir {
                collect_project_fmt_paths_recursive(&path, out);
            }
            continue;
        }

        if !file_type.is_file() {
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) != Some("mp") {
            continue;
        }

        out.push(path.to_string_lossy().to_string());
    }
}

fn parse_budget_policy(policy: Option<&str>) -> BudgetPolicy {
    match policy.unwrap_or(magpie_driver::DEFAULT_LLM_BUDGET_POLICY) {
        "diagnostics_first" => BudgetPolicy::DiagnosticsFirst,
        "slices_first" => BudgetPolicy::SlicesFirst,
        "minimal" => BudgetPolicy::Minimal,
        _ => BudgetPolicy::Balanced,
    }
}

fn find_latest_mms_index() -> Option<PathBuf> {
    let preferred_dir = Path::new(".magpie").join("memory");
    find_latest_mms_index_in_dir(&preferred_dir)
        .or_else(|| find_latest_mms_index_in_dir(Path::new("target")))
}

fn find_latest_mms_index_in_dir(dir: &Path) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    collect_mms_index_paths(dir, &mut candidates);
    candidates.sort();
    candidates.pop()
}

fn collect_mms_index_paths(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_mms_index_paths(&path, out);
            continue;
        }
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".mms_index.json"))
        {
            out.push(path);
        }
    }
}

fn print_test_result(result: &TestResult, discovery_only_mode: bool) {
    println!("running {} tests...", result.total);
    for (name, passed) in &result.test_names {
        println!("test {name} ... {}", if *passed { "ok" } else { "FAILED" });
    }
    println!();
    println!(
        "test result: {}. {} passed; {} failed; 0 ignored",
        if result.failed == 0 { "ok" } else { "FAILED" },
        result.passed,
        result.failed
    );
    if discovery_only_mode {
        println!("note: test discovery only mode (--emit exe not enabled)");
    }
}

fn build_result_from_test_result(
    test_result: &TestResult,
    filter: Option<&str>,
    discovery_only_mode: bool,
) -> BuildResult {
    let mut result = BuildResult {
        success: test_result.failed == 0,
        diagnostics: Vec::new(),
        artifacts: Vec::new(),
        timing_ms: Default::default(),
    };

    result.diagnostics.push(info_diag(
        "MPT0000",
        "test summary",
        format!(
            "{} tests discovered; {} passed; {} failed",
            test_result.total, test_result.passed, test_result.failed
        ),
    ));

    if let Some(filter) = filter {
        result.diagnostics.push(info_diag(
            "MPT0000",
            "test filter applied",
            format!("Applied filter pattern '{filter}'."),
        ));
    }

    if discovery_only_mode {
        result.diagnostics.push(info_diag(
            "MPT0000",
            "test discovery only",
            "Test execution skipped because --emit exe is not enabled.",
        ));
    }

    for (name, passed) in &test_result.test_names {
        let diag = if *passed {
            info_diag("MPT0000", "test passed", format!("test {name} ... ok"))
        } else {
            error_diag("MPT0001", "test failed", format!("test {name} ... FAILED"))
        };
        result.diagnostics.push(diag);
    }

    result
}

fn emit_command_output(
    command: &str,
    config: &DriverConfig,
    result: &BuildResult,
    mode: OutputMode,
) {
    match mode {
        OutputMode::Text => print_human_output(command, result),
        OutputMode::Json => {
            let mut envelope = magpie_driver::json_output_envelope(command, config, result);
            magpie_driver::apply_llm_budget(config, &mut envelope);
            match canonical_json_encode(&envelope) {
                Ok(payload) => println!("{payload}"),
                Err(err) => eprintln!("Failed to serialize output envelope: {err}"),
            }
        }
        OutputMode::Jsonl => {
            let mut envelope = magpie_driver::json_output_envelope(command, config, result);
            magpie_driver::apply_llm_budget(config, &mut envelope);
            emit_jsonl_envelope(&envelope);
        }
    }
}

fn emit_jsonl_envelope(envelope: &OutputEnvelope) {
    fn emit_line(value: &serde_json::Value) {
        println!("{}", canonical_json_string(value));
    }

    emit_line(&serde_json::json!({
        "type": "meta",
        "magpie_version": envelope.magpie_version,
        "command": envelope.command,
        "target": envelope.target,
        "success": envelope.success,
    }));

    if let Some(llm_budget) = &envelope.llm_budget {
        emit_line(&serde_json::json!({
            "type": "llm_budget",
            "value": llm_budget,
        }));
    }

    for artifact in &envelope.artifacts {
        emit_line(&serde_json::json!({
            "type": "artifact",
            "path": artifact,
        }));
    }

    for diag in &envelope.diagnostics {
        match serde_json::to_value(diag) {
            Ok(mut row) => {
                if let Some(obj) = row.as_object_mut() {
                    obj.insert(
                        "type".to_string(),
                        serde_json::Value::String("diagnostic".to_string()),
                    );
                }
                emit_line(&row);
            }
            Err(err) => {
                emit_line(&serde_json::json!({
                    "type": "diagnostic",
                    "code": diag.code,
                    "severity": "error",
                    "message": format!("failed to serialize diagnostic payload: {err}"),
                }));
            }
        }
    }

    if let Some(graphs) = envelope.graphs.as_object() {
        for key in ["symbols", "deps", "ownership", "cfg"] {
            let Some(payload) = graphs.get(key) else {
                continue;
            };
            let is_empty_object = payload
                .as_object()
                .map(|obj| obj.is_empty())
                .unwrap_or(false);
            let is_empty_array = payload
                .as_array()
                .map(|arr| arr.is_empty())
                .unwrap_or(false);
            if payload.is_null() || is_empty_object || is_empty_array {
                continue;
            }
            emit_line(&serde_json::json!({
                "type": "graph",
                "graph": key,
                "payload": payload,
            }));
        }
    }

    if let Some(timing) = envelope.timing_ms.as_object() {
        let mut entries = timing.iter().collect::<Vec<_>>();
        entries.sort_by(|(lhs, _), (rhs, _)| lhs.cmp(rhs));
        for (stage, ms) in entries {
            emit_line(&serde_json::json!({
                "type": "timing",
                "stage": stage,
                "ms": ms,
            }));
        }
    }

    emit_line(&serde_json::json!({
        "type": "end",
        "success": envelope.success,
        "diagnostic_count": envelope.diagnostics.len(),
        "artifact_count": envelope.artifacts.len(),
    }));
}

fn print_human_output(command: &str, result: &BuildResult) {
    let status = if result.success { "ok" } else { "failed" };
    println!("{command}: {status}");

    if !result.artifacts.is_empty() {
        println!("Artifacts:");
        for artifact in &result.artifacts {
            println!("  - {artifact}");
        }
    }

    if !result.diagnostics.is_empty() {
        println!("Diagnostics:");
        for diag in &result.diagnostics {
            println!(
                "  - [{}] {}: {}",
                diag.code,
                severity_label(&diag.severity),
                diag.message
            );
        }
    }

    if !result.timing_ms.is_empty() {
        println!("Timing (ms):");
        let mut items: Vec<(&str, u64)> = result
            .timing_ms
            .iter()
            .map(|(stage, ms)| (stage.as_str(), *ms))
            .collect();
        items.sort_by_key(|(stage, _)| *stage);
        for (stage, ms) in items {
            println!("  - {stage}: {ms}");
        }
    }
}

fn severity_label(severity: &Severity) -> &'static str {
    match severity {
        Severity::Error => "error",
        Severity::Warning => "warning",
        Severity::Info => "info",
        Severity::Hint => "hint",
    }
}

fn info_diag(
    code: impl Into<String>,
    title: impl Into<String>,
    message: impl Into<String>,
) -> Diagnostic {
    simple_diag(code, Severity::Info, title, message)
}

fn error_diag(
    code: impl Into<String>,
    title: impl Into<String>,
    message: impl Into<String>,
) -> Diagnostic {
    simple_diag(code, Severity::Error, title, message)
}

fn simple_diag(
    code: impl Into<String>,
    severity: Severity,
    title: impl Into<String>,
    message: impl Into<String>,
) -> Diagnostic {
    Diagnostic {
        code: code.into(),
        severity,
        title: title.into(),
        primary_span: None,
        secondary_spans: Vec::new(),
        message: message.into(),
        explanation_md: None,
        why: None,
        suggested_fixes: Vec::new(),
        rag_bundle: Vec::new(),
        related_docs: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::collections::BTreeMap;
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};

    static CWD_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    #[test]
    fn test_driver_config_defaults_emit_exe_for_test_command() {
        let cli = Cli::try_parse_from(["magpie", "test"]).expect("CLI should parse");
        let config = test_driver_config(&cli);
        assert_eq!(config.emit, vec!["exe".to_string()]);
        assert!(config.features.iter().any(|feature| feature == "test"));
    }

    #[test]
    fn test_driver_config_respects_explicit_emit_for_test_command() {
        let cli =
            Cli::try_parse_from(["magpie", "--emit", "mpir", "test"]).expect("CLI should parse");
        let config = test_driver_config(&cli);
        assert_eq!(config.emit, vec!["mpir".to_string()]);
        assert!(config.features.iter().any(|feature| feature == "test"));
    }

    #[test]
    fn collect_project_fmt_paths_recurses_and_skips_target() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("magpie_cli_fmt_{}_{}", std::process::id(), nonce));
        std::fs::create_dir_all(root.join("src/nested")).expect("src dir should exist");
        std::fs::create_dir_all(root.join("target/debug")).expect("target dir should exist");

        let src_main = root.join("src/main.mp");
        let src_nested = root.join("src/nested/mod.mp");
        let target_file = root.join("target/debug/generated.mp");
        let txt_file = root.join("notes.txt");
        std::fs::write(&src_main, "module demo.main").expect("main source should be written");
        std::fs::write(&src_nested, "module demo.nested").expect("nested source should be written");
        std::fs::write(&target_file, "module generated").expect("target source should be written");
        std::fs::write(&txt_file, "not magpie source").expect("txt file should be written");

        let paths = collect_project_fmt_paths(&root);
        assert!(paths.iter().any(|path| path.ends_with("src/main.mp")));
        assert!(paths.iter().any(|path| path.ends_with("src/nested/mod.mp")));
        assert!(!paths.iter().any(|path| path.contains("/target/")));
        assert!(!paths.iter().any(|path| path.ends_with("notes.txt")));

        std::fs::remove_dir_all(root).expect("temp tree should be removable");
    }

    #[test]
    fn handle_memory_query_reports_stale_index() {
        let _guard = CWD_TEST_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("cwd test lock should be available");

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "magpie_cli_memory_stale_{}_{}",
            std::process::id(),
            nonce
        ));
        std::fs::create_dir_all(root.join(".magpie/memory")).expect("memory dir should exist");
        std::fs::create_dir_all(root.join("src")).expect("src dir should exist");

        let source = root.join("src/main.mp");
        std::fs::write(&source, "module test.main\ndigest \"deadbeef\"\n")
            .expect("source file should be written");

        let mut token_cost = BTreeMap::new();
        token_cost.insert("approx:utf8_4chars".to_string(), 1);
        let item = magpie_memory::MmsItem {
            item_id: "I:stale".to_string(),
            kind: "symbol_capsule".to_string(),
            sid: "S:stale".to_string(),
            fqn: source.to_string_lossy().to_string(),
            module_sid: "M:stale".to_string(),
            source_digest: "0000000000000000".to_string(),
            body_digest: "0000000000000000".to_string(),
            text: "stale".to_string(),
            tags: vec!["test".to_string()],
            priority: 1,
            token_cost,
        };
        let index = magpie_memory::build_index_with_sources(
            &[item],
            &[magpie_memory::MmsSourceFingerprint {
                path: source.to_string_lossy().to_string(),
                digest: "0000000000000000".to_string(),
            }],
        );
        let encoded = serde_json::to_string_pretty(&index).expect("index should serialize");
        std::fs::write(root.join(".magpie/memory/stale.mms_index.json"), encoded)
            .expect("index fixture should be written");

        let old_cwd = std::env::current_dir().expect("cwd should resolve");
        std::env::set_current_dir(&root).expect("cwd should change");
        let result = handle_memory_query(&DriverConfig::default(), "main", 5);
        std::env::set_current_dir(old_cwd).expect("cwd should restore");

        assert!(!result.success);
        assert!(result.diagnostics.iter().any(|diag| diag.code == "MPM1003"));

        std::fs::remove_dir_all(root).expect("temp tree should be removable");
    }
}
