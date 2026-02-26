//! Magpie CLI entry point (§5).

use clap::{Parser, Subcommand};
use magpie_diag::{Diagnostic, Severity};
use magpie_driver::{BuildProfile, BuildResult, DriverConfig};

#[derive(Parser, Debug)]
#[command(name = "magpie", version = "0.1.0", about = "Magpie language toolchain")]
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
                    result.diagnostics.push(error_diag(
                        "MPC0001",
                        "project creation failed",
                        err,
                    ));
                }
            }

            let config = driver_config_from_cli(&cli, false);
            emit_command_output("new", &config, &result, output_mode);
            if result.success { 0 } else { 1 }
        }
        Commands::Build => {
            let config = driver_config_from_cli(&cli, false);
            let result = magpie_driver::build(&config);
            emit_command_output("build", &config, &result, output_mode);
            if result.success { 0 } else { 1 }
        }
        Commands::Run { args } => {
            let config = driver_config_from_cli(&cli, false);
            let mut result = magpie_driver::build(&config);
            if result.success {
                let message = if args.is_empty() {
                    "run placeholder: execution is not implemented yet.".to_string()
                } else {
                    format!(
                        "run placeholder: execution is not implemented yet (args: {}).",
                        args.join(" ")
                    )
                };
                result
                    .diagnostics
                    .push(info_diag("MPR0000", "run placeholder", message));
            }

            emit_command_output("run", &config, &result, output_mode);
            if result.success { 0 } else { 1 }
        }
        Commands::Fmt { fix_meta } => {
            let paths = vec!["src/main.mp".to_string()];
            let result = magpie_driver::format_files(&paths, *fix_meta);
            let config = driver_config_from_cli(&cli, false);
            emit_command_output("fmt", &config, &result, output_mode);
            if result.success { 0 } else { 1 }
        }
        Commands::Test { filter } => {
            let config = driver_config_from_cli(&cli, true);
            let mut result = magpie_driver::build(&config);
            if let Some(filter) = filter {
                result.diagnostics.push(info_diag(
                    "MPT0000",
                    "test filter accepted",
                    format!("Test filter '{filter}' is accepted but not yet applied."),
                ));
            }

            emit_command_output("test", &config, &result, output_mode);
            if result.success { 0 } else { 1 }
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
            if result.success { 0 } else { 1 }
        }
        _ => {
            println!("Command not yet implemented: {:?}", cli.command);
            2
        }
    };

    std::process::exit(exit_code);
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum OutputMode {
    Text,
    Json,
}

fn effective_output_mode(cli: &Cli) -> OutputMode {
    if cli.llm || cli.output == "json" || cli.output == "jsonl" {
        OutputMode::Json
    } else {
        OutputMode::Text
    }
}

fn driver_config_from_cli(cli: &Cli, test_mode: bool) -> DriverConfig {
    let mut config = DriverConfig::default();

    config.profile = match cli.profile.as_str() {
        "release" => BuildProfile::Release,
        _ => BuildProfile::Dev,
    };
    if let Some(target) = &cli.target {
        config.target_triple = target.clone();
    }
    if let Some(emit) = &cli.emit {
        let emit_items = parse_csv(emit);
        if !emit_items.is_empty() {
            config.emit = emit_items;
        }
    }

    config.max_errors = cli.max_errors as usize;
    config.llm_mode = cli.llm;
    config.token_budget = cli.llm_token_budget;
    config.shared_generics = cli.shared_generics;
    config.features = cli.features.as_deref().map(parse_csv).unwrap_or_default();

    if test_mode && !config.features.iter().any(|feature| feature == "test") {
        config.features.push("test".to_string());
    }

    config
}

fn parse_csv(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(str::to_string)
        .collect()
}

fn emit_command_output(command: &str, config: &DriverConfig, result: &BuildResult, mode: OutputMode) {
    match mode {
        OutputMode::Text => print_human_output(command, result),
        OutputMode::Json => {
            let envelope = magpie_driver::json_output_envelope(command, config, result);
            match serde_json::to_string_pretty(&envelope) {
                Ok(payload) => println!("{payload}"),
                Err(err) => eprintln!("Failed to serialize output envelope: {err}"),
            }
        }
    }
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

fn info_diag(code: impl Into<String>, title: impl Into<String>, message: impl Into<String>) -> Diagnostic {
    simple_diag(code, Severity::Info, title, message)
}

fn error_diag(code: impl Into<String>, title: impl Into<String>, message: impl Into<String>) -> Diagnostic {
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
    }
}
