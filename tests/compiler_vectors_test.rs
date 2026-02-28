use magpie_ast::FileId;
use magpie_diag::{DiagnosticBag, Severity};
use magpie_driver::{
    build, format_files, generate_docs, lint, parse_entry, BuildProfile, BuildResult, DriverConfig,
};
use magpie_lex::lex;
use magpie_parse::parse_file;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

fn fixture_paths() -> Vec<PathBuf> {
    let mut paths = fs::read_dir(fixtures_dir())
        .expect("fixtures directory should exist")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("mp"))
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

fn fixture_path(name: &str) -> PathBuf {
    fixtures_dir().join(name)
}

fn nonce() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before unix epoch")
        .as_nanos()
}

fn unique_target(label: &str) -> String {
    format!("it-{label}-{}-{}", std::process::id(), nonce())
}

fn build_fixture_with(
    entry_path: &Path,
    profile: BuildProfile,
    emit: &[&str],
    llm_mode: bool,
    shared_generics: bool,
    target_triple: String,
) -> (BuildResult, String) {
    let config = DriverConfig {
        entry_path: entry_path.to_string_lossy().to_string(),
        profile,
        target_triple: target_triple.clone(),
        emit: emit.iter().map(|kind| (*kind).to_string()).collect(),
        llm_mode,
        token_budget: llm_mode.then_some(8_000),
        llm_tokenizer: llm_mode.then_some("approx:utf8_4chars".to_string()),
        llm_budget_policy: llm_mode.then_some("balanced".to_string()),
        shared_generics,
        ..DriverConfig::default()
    };
    (build(&config), target_triple)
}

fn parse_fixture(path: &Path) {
    let source = fs::read_to_string(path).expect("fixture source should be readable");
    let mut diag = DiagnosticBag::new(32);
    let tokens = lex(FileId(0), &source, &mut diag);
    let ast = parse_file(&tokens, FileId(0), &mut diag);
    assert!(ast.is_ok(), "fixture should parse: {}", path.display());
    let error_diags = diag
        .diagnostics
        .iter()
        .filter(|diag| matches!(diag.severity, Severity::Error))
        .map(|diag| format!("{}: {}", diag.code, diag.message))
        .collect::<Vec<_>>();
    assert!(
        error_diags.is_empty(),
        "unexpected parse diagnostics for {}: {:?}",
        path.display(),
        error_diags
    );
}

fn assert_build_success(result: &BuildResult, context: &str) {
    let errors = result
        .diagnostics
        .iter()
        .filter(|diag| matches!(diag.severity, Severity::Error))
        .map(|diag| format!("{}: {}", diag.code, diag.message))
        .collect::<Vec<_>>();
    assert!(
        result.success,
        "build failed for {context}; diagnostics={:?}",
        errors
    );
}

fn assert_has_artifact_suffix(result: &BuildResult, suffix: &str) {
    assert!(
        result
            .artifacts
            .iter()
            .any(|artifact| artifact.ends_with(suffix)),
        "expected artifact suffix `{suffix}`; got {:?}",
        result.artifacts
    );
}

fn assert_artifacts_exist(result: &BuildResult) {
    for artifact in &result.artifacts {
        let artifact_path = Path::new(artifact);
        assert!(artifact_path.exists(), "artifact missing: {}", artifact);
    }
}

fn cleanup_target(target_triple: &str) {
    let _ = fs::remove_dir_all(Path::new("target").join(target_triple));
}

fn write_temp_source(label: &str, source: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "magpie_test_{label}_{}_{}",
        std::process::id(),
        nonce()
    ));
    fs::create_dir_all(&root).expect("temp source dir should be created");
    let entry = root.join("main.mp");
    fs::write(&entry, source).expect("temp source should be written");
    entry
}

#[test]
fn vector_parse_all_language_fixtures() {
    let fixtures = fixture_paths();
    assert!(
        !fixtures.is_empty(),
        "expected at least one fixture under {}",
        fixtures_dir().display()
    );

    for fixture in fixtures {
        parse_fixture(&fixture);
    }
}

#[test]
fn vector_build_all_fixtures_to_mpir_dev() {
    let target = unique_target("mpir-dev");
    for fixture in fixture_paths() {
        let (result, _) = build_fixture_with(
            &fixture,
            BuildProfile::Dev,
            &["mpir"],
            false,
            false,
            target.clone(),
        );
        let context = format!("{} (mpir/dev)", fixture.display());
        assert_build_success(&result, &context);
        assert_has_artifact_suffix(&result, ".mpir");
        assert_artifacts_exist(&result);
    }
    cleanup_target(&target);
}

#[test]
fn vector_build_feature_harness_emit_matrix() {
    let feature_harness = fixture_path("feature_harness.mp");
    let target = unique_target("emit-matrix");
    let (result, target_triple) = build_fixture_with(
        &feature_harness,
        BuildProfile::Dev,
        &[
            "mpir",
            "llvm-ir",
            "mpdbg",
            "symgraph",
            "depsgraph",
            "ownershipgraph",
            "cfggraph",
        ],
        false,
        false,
        target,
    );

    assert_build_success(&result, "feature_harness emit matrix");
    for suffix in [
        ".mpir",
        ".ll",
        ".mpdbg",
        ".symgraph.json",
        ".depsgraph.json",
        ".ownershipgraph.json",
        ".cfggraph.json",
    ] {
        assert_has_artifact_suffix(&result, suffix);
    }
    assert_artifacts_exist(&result);

    for graph_suffix in [
        ".symgraph.json",
        ".depsgraph.json",
        ".ownershipgraph.json",
        ".cfggraph.json",
    ] {
        let graph = result
            .artifacts
            .iter()
            .find(|artifact| artifact.ends_with(graph_suffix))
            .expect("graph artifact should exist");
        let payload = fs::read_to_string(graph).expect("graph artifact should be readable");
        let value: Value =
            serde_json::from_str(&payload).expect("graph artifact should be valid JSON");
        assert!(
            value.is_object(),
            "graph payload should be object for {graph}"
        );
    }

    cleanup_target(&target_triple);
}

#[test]
fn vector_build_release_shared_generics_and_llm_mode() {
    let feature_harness = fixture_path("feature_harness.mp");
    let target = unique_target("release-shared-llm");
    let (result, target_triple) = build_fixture_with(
        &feature_harness,
        BuildProfile::Release,
        &["mpir", "llvm-ir"],
        true,
        true,
        target,
    );
    assert_build_success(&result, "feature_harness release/shared_generics/llm");
    assert_has_artifact_suffix(&result, ".mpir");
    assert_has_artifact_suffix(&result, ".ll");
    assert_artifacts_exist(&result);
    cleanup_target(&target_triple);
}

#[test]
fn vector_parse_entry_and_lint_commands() {
    let fixture = fixture_path("feature_harness.mp");
    let target = unique_target("parse-lint");
    let config = DriverConfig {
        entry_path: fixture.to_string_lossy().to_string(),
        target_triple: target.clone(),
        profile: BuildProfile::Dev,
        emit: vec!["ast".to_string()],
        ..DriverConfig::default()
    };

    let parse_result = parse_entry(&config);
    assert_build_success(&parse_result, "parse_entry feature_harness");
    assert_has_artifact_suffix(&parse_result, ".ast.txt");
    assert_artifacts_exist(&parse_result);

    let lint_result = lint(&DriverConfig {
        entry_path: fixture.to_string_lossy().to_string(),
        target_triple: target.clone(),
        profile: BuildProfile::Dev,
        ..DriverConfig::default()
    });
    assert_build_success(&lint_result, "lint feature_harness");
    assert!(
        lint_result
            .diagnostics
            .iter()
            .all(|diag| !matches!(diag.severity, Severity::Error)),
        "lint emitted errors: {:?}",
        lint_result
            .diagnostics
            .iter()
            .map(|diag| format!("{}: {}", diag.code, diag.message))
            .collect::<Vec<_>>()
    );

    cleanup_target(&target);
}

#[test]
fn vector_generate_docs_for_all_fixtures() {
    let fixture_list = fixture_paths();
    let input_paths = fixture_list
        .iter()
        .map(|path| path.to_string_lossy().to_string())
        .collect::<Vec<_>>();

    let docs_result = generate_docs(&input_paths);
    assert_build_success(&docs_result, "generate_docs fixtures");
    assert_eq!(
        docs_result.artifacts.len(),
        input_paths.len(),
        "expected one mpd output per fixture"
    );
    for artifact in &docs_result.artifacts {
        assert!(
            artifact.ends_with(".mpd"),
            "unexpected doc artifact: {artifact}"
        );
        assert!(
            Path::new(artifact).exists(),
            "doc artifact missing: {artifact}"
        );
        let payload = fs::read_to_string(artifact).expect("doc payload should be readable");
        assert!(
            payload.contains("module "),
            "mpd should include module stanza"
        );
    }

    for input in input_paths {
        let source_path = Path::new(&input);
        let mpd_path = source_path.with_extension("mpd");
        let _ = fs::remove_file(mpd_path);
    }
}

#[test]
fn vector_formatter_round_trip_with_fix_meta() {
    let source = r#"module test.format_case
exports { @main }
imports { }
digest "0000000000000000"

fn @main() -> i32 {
bb0:
ret const.i32 0
}
"#;
    let entry = write_temp_source("format", source);
    let entry_str = entry.to_string_lossy().to_string();

    let first = format_files(std::slice::from_ref(&entry_str), true);
    assert_build_success(&first, "format first pass");

    let formatted_once = fs::read_to_string(&entry).expect("formatted file should be readable");
    assert!(
        formatted_once.contains("meta { }"),
        "formatter with --fix-meta should insert meta block"
    );

    let second = format_files(std::slice::from_ref(&entry_str), true);
    assert_build_success(&second, "format second pass");
    let formatted_twice = fs::read_to_string(&entry).expect("formatted file should be readable");
    assert_eq!(
        formatted_once, formatted_twice,
        "formatter should be idempotent on second pass"
    );

    let _ = fs::remove_dir_all(entry.parent().expect("temp file parent should exist"));
}

#[test]
fn vector_diagnostic_parse_error() {
    let entry = write_temp_source(
        "parse_error",
        r#"module bad.parse
exports { @main }
imports { }
digest "0000000000000000"

fn @main() -> i32 {
bb0:
  %x: i32 = const.i32 1
  ret %x
"#,
    );
    let target = unique_target("diag-parse");
    let (result, target_triple) =
        build_fixture_with(&entry, BuildProfile::Dev, &["mpir"], false, false, target);

    assert!(!result.success, "parse-error source should fail build");
    assert!(
        result
            .diagnostics
            .iter()
            .any(|diag| matches!(diag.severity, Severity::Error)),
        "expected at least one error diagnostic, got {:?}",
        result
            .diagnostics
            .iter()
            .map(|diag| format!("{}: {}", diag.code, diag.message))
            .collect::<Vec<_>>()
    );

    cleanup_target(&target_triple);
    let _ = fs::remove_dir_all(entry.parent().expect("temp file parent should exist"));
}

#[test]
fn vector_diagnostic_type_error() {
    let entry = write_temp_source(
        "type_error",
        r#"module bad.type
exports { @takes_i32, @main }
imports { }
digest "0000000000000000"

fn @takes_i32(%x: i32) -> i32 {
bb0:
  ret %x
}

fn @main() -> i32 {
bb0:
  %bad: i32 = call @takes_i32 { x=const.bool true }
  ret %bad
}
"#,
    );
    let target = unique_target("diag-type");
    let (result, target_triple) =
        build_fixture_with(&entry, BuildProfile::Dev, &["mpir"], false, false, target);

    assert!(!result.success, "type-error source should fail build");
    assert!(
        result
            .diagnostics
            .iter()
            .any(|diag| matches!(diag.severity, Severity::Error)),
        "expected at least one error diagnostic, got {:?}",
        result
            .diagnostics
            .iter()
            .map(|diag| format!("{}: {}", diag.code, diag.message))
            .collect::<Vec<_>>()
    );

    cleanup_target(&target_triple);
    let _ = fs::remove_dir_all(entry.parent().expect("temp file parent should exist"));
}

#[test]
fn vector_diagnostic_ownership_error() {
    let entry = write_temp_source(
        "ownership_error",
        r#"module bad.ownership
exports { @main }
imports { }
digest "0000000000000000"

heap struct TPoint {
  field x: i32
}

fn @main() -> i32 {
bb0:
  %p: TPoint = new TPoint { x=const.i32 1 }
  %sp: shared TPoint = share { v=%p }
  %bad_use_after_move: borrow TPoint = borrow.shared { v=%p }
  %x: i32 = getfield { field=x, obj=%bad_use_after_move }
  ret const.i32 0
}
"#,
    );
    let target = unique_target("diag-ownership");
    let (result, target_triple) =
        build_fixture_with(&entry, BuildProfile::Dev, &["mpir"], false, false, target);

    assert!(!result.success, "ownership-error source should fail build");
    assert!(
        result
            .diagnostics
            .iter()
            .any(|diag| matches!(diag.severity, Severity::Error)),
        "expected at least one error diagnostic, got {:?}",
        result
            .diagnostics
            .iter()
            .map(|diag| format!("{}: {}", diag.code, diag.message))
            .collect::<Vec<_>>()
    );

    cleanup_target(&target_triple);
    let _ = fs::remove_dir_all(entry.parent().expect("temp file parent should exist"));
}

#[test]
fn vector_unknown_emit_is_reported_but_valid_emit_still_produces_artifact() {
    let fixture = fixture_path("hello.mp");
    let target = unique_target("unknown-emit");
    let (result, target_triple) = build_fixture_with(
        &fixture,
        BuildProfile::Dev,
        &["mpir", "definitely-not-real"],
        false,
        false,
        target,
    );

    assert!(result.success, "known emit should still succeed build");
    assert_has_artifact_suffix(&result, ".mpir");
    assert!(
        result.diagnostics.iter().any(|diag| diag.code == "MPL0001"),
        "expected unknown emit warning MPL0001, got {:?}",
        result
            .diagnostics
            .iter()
            .map(|diag| format!("{}: {}", diag.code, diag.message))
            .collect::<Vec<_>>()
    );

    cleanup_target(&target_triple);
}
