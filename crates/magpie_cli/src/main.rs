//! Magpie CLI entry point (§5).

use clap::{Parser, Subcommand};

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

    match &cli.command {
        Commands::New { name } => {
            println!("Creating new Magpie project: {}", name);
            create_project(name);
        }
        Commands::Build => {
            println!("Building...");
            // TODO: Wire up magpie_driver
        }
        Commands::Run { args: _ } => {
            println!("Building and running...");
        }
        Commands::Fmt { fix_meta: _ } => {
            println!("Formatting (CSNF)...");
        }
        Commands::Test { filter: _ } => {
            println!("Running tests...");
        }
        _ => {
            println!("Command not yet implemented: {:?}", cli.command);
        }
    }
}

fn create_project(name: &str) {
    use std::fs;
    use std::path::Path;

    let base = Path::new(name);
    fs::create_dir_all(base.join("src")).expect("Failed to create src/");
    fs::create_dir_all(base.join("tests")).expect("Failed to create tests/");
    fs::create_dir_all(base.join(".magpie")).expect("Failed to create .magpie/");

    // Magpie.toml
    let manifest = format!(
        r#"[package]
name = "{name}"
version = "0.1.0"
edition = "2026"

[build]
entry = "src/main.mp"
profile_default = "dev"
max_mono_instances = 10000

[dependencies]
std = {{ version = "^0.1" }}

[llm]
mode_default = true
token_budget = 12000
tokenizer = "approx:utf8_4chars"
budget_policy = "balanced"
max_module_lines = 800
max_fn_lines = 80
"#
    );
    fs::write(base.join("Magpie.toml"), manifest).expect("Failed to write Magpie.toml");

    // src/main.mp
    let main_mp = format!(
        r#"module {name}.main
exports {{ @main }}
imports {{ std.io::{{@println}} }}
digest "0000000000000000"

fn @main() -> i32 {{
bb0:
  %msg: Str = const.Str "Hello, world!"
  call_void std.io.@println {{ args=[%msg] }}
  ret const.i32 0
}}
"#
    );
    fs::write(base.join("src/main.mp"), main_mp).expect("Failed to write main.mp");

    // Magpie.lock (empty)
    fs::write(base.join("Magpie.lock"), "{}").expect("Failed to write Magpie.lock");

    println!("Created project '{name}' with Magpie.toml, src/main.mp, and tests/");
}
