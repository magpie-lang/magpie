use magpie_ast::FileId;
use magpie_diag::DiagnosticBag;
use magpie_driver::{build, BuildProfile, DriverConfig};
use magpie_lex::lex;
use magpie_parse::parse_file;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

fn parse_fixture(path: &Path) {
    let source = std::fs::read_to_string(path).expect("fixture source should be readable");
    let mut diag = DiagnosticBag::new(32);
    let tokens = lex(FileId(0), &source, &mut diag);
    let _ = parse_file(&tokens, FileId(0), &mut diag).expect("fixture should parse");
    assert!(
        !diag.has_errors(),
        "unexpected parse diagnostics: {:?}",
        diag.diagnostics
            .iter()
            .map(|d| d.code.as_str())
            .collect::<Vec<_>>()
    );
}

fn unique_temp_dir(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before unix epoch")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "magpie_integration_{label}_{}_{}",
        std::process::id(),
        nonce
    ));
    std::fs::create_dir_all(&dir).expect("temp dir should be created");
    dir
}

fn prepare_fixture_for_build(path: &Path) -> PathBuf {
    let source = std::fs::read_to_string(path).expect("fixture source should be readable");
    // Driver builds a standalone file in this mode; remove external imports for fixture-only e2e.
    let source = source.replace("imports { std.io::{@println} }", "imports { }");
    let dir = unique_temp_dir("build");
    let entry = dir.join("main.mp");
    std::fs::write(&entry, source).expect("prepared fixture should be written");
    entry
}

fn build_entry(entry_path: &Path) {
    let config = DriverConfig {
        entry_path: entry_path.to_string_lossy().to_string(),
        profile: BuildProfile::Dev,
        emit: vec!["mpir".to_string()],
        ..DriverConfig::default()
    };
    let result = build(&config);
    assert!(
        result.success,
        "build failed with diagnostics: {:?}",
        result
            .diagnostics
            .iter()
            .map(|d| format!("{}: {}", d.code, d.message))
            .collect::<Vec<_>>()
    );
}

#[test]
fn hello_fixture_parses_and_builds() {
    let fixture = fixture_path("hello.mp");
    parse_fixture(&fixture);
    let entry = prepare_fixture_for_build(&fixture);
    build_entry(&entry);
}

#[test]
fn arithmetic_fixture_parses_and_builds() {
    let fixture = fixture_path("arithmetic.mp");
    parse_fixture(&fixture);
    let entry = prepare_fixture_for_build(&fixture);
    build_entry(&entry);
}

#[test]
fn fixtures_build_via_driver_pipeline() {
    let hello_entry = prepare_fixture_for_build(&fixture_path("hello.mp"));
    let arithmetic_entry = prepare_fixture_for_build(&fixture_path("arithmetic.mp"));
    build_entry(&hello_entry);
    build_entry(&arithmetic_entry);
}
