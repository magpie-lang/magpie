//! magpie_pkg

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub package: PackageSection,
    pub build: BuildSection,
    #[serde(default)]
    pub dependencies: HashMap<String, DependencySpec>,
    #[serde(default)]
    pub features: HashMap<String, FeatureSpec>,
    #[serde(default)]
    pub llm: Option<LlmSection>,
    #[serde(default)]
    pub web: Option<WebSection>,
    #[serde(default)]
    pub gpu: Option<GpuSection>,
    #[serde(default)]
    pub toolchain: HashMap<String, ToolchainTargetSection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageSection {
    pub name: String,
    pub version: String,
    pub edition: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildSection {
    pub entry: String,
    pub profile_default: String,
    #[serde(default)]
    pub max_mono_instances: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DependencySpec {
    Detail(Dependency),
    Version(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub registry: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub git: Option<String>,
    #[serde(default)]
    pub rev: Option<String>,
    #[serde(default)]
    pub features: Vec<String>,
    #[serde(default)]
    pub optional: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FeatureSpec {
    #[serde(default)]
    pub modules: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolchainTargetSection {
    #[serde(default)]
    pub sysroot: Option<String>,
    #[serde(default)]
    pub linker: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LlmSection {
    #[serde(default)]
    pub mode_default: Option<bool>,
    #[serde(default)]
    pub token_budget: Option<u64>,
    #[serde(default)]
    pub tokenizer: Option<String>,
    #[serde(default)]
    pub budget_policy: Option<String>,
    #[serde(default)]
    pub max_module_lines: Option<u64>,
    #[serde(default)]
    pub max_fn_lines: Option<u64>,
    #[serde(default)]
    pub auto_split_on_budget_violation: Option<bool>,
    #[serde(default)]
    pub rag: Option<LlmRagSection>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LlmRagSection {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub backend: Option<String>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub max_items_per_diag: Option<u32>,
    #[serde(default)]
    pub include_repair_episodes: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WebSection {
    #[serde(default)]
    pub addr: Option<String>,
    #[serde(default)]
    pub port: Option<u16>,
    #[serde(default)]
    pub open_browser: Option<bool>,
    #[serde(default)]
    pub max_body_bytes: Option<u64>,
    #[serde(default)]
    pub threads: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpuSection {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub backend: Option<String>,
    #[serde(default)]
    pub device_index: Option<i32>,
}

pub fn parse_manifest(path: &Path) -> Result<Manifest, String> {
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("failed to read manifest '{}': {}", path.display(), e))?;
    toml::from_str::<Manifest>(&raw)
        .map_err(|e| format!("failed to parse manifest '{}': {}", path.display(), e))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockFile {
    pub lock_version: u32,
    pub generated_by: GeneratedBy,
    #[serde(default)]
    pub packages: Vec<LockPackage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedBy {
    pub magpie_version: String,
    pub toolchain_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockPackage {
    pub name: String,
    pub version: String,
    pub source: LockSource,
    pub content_hash: String,
    #[serde(default)]
    pub deps: Vec<LockDependency>,
    #[serde(default)]
    pub resolved_features: Vec<String>,
    #[serde(default)]
    pub targets: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockSource {
    pub kind: String,
    #[serde(default)]
    pub registry: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub rev: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockDependency {
    pub name: String,
    pub req: String,
    #[serde(default)]
    pub features: Vec<String>,
}

pub fn read_lockfile(path: &Path) -> Result<LockFile, String> {
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("failed to read lockfile '{}': {}", path.display(), e))?;
    serde_json::from_str::<LockFile>(&raw)
        .map_err(|e| format!("failed to parse lockfile '{}': {}", path.display(), e))
}

pub fn write_lockfile(lock: &LockFile, path: &Path) -> Result<(), String> {
    let value =
        serde_json::to_value(lock).map_err(|e| format!("failed to encode lockfile JSON: {}", e))?;
    let canonical = canonical_json(&value);
    fs::write(path, canonical)
        .map_err(|e| format!("failed to write lockfile '{}': {}", path.display(), e))
}

pub fn resolve_deps(manifest: &Manifest, offline: bool) -> Result<LockFile, String> {
    let mut packages = Vec::new();

    let mut dep_names = manifest.dependencies.keys().cloned().collect::<Vec<_>>();
    dep_names.sort_unstable();

    for dep_name in dep_names {
        let dep_spec = manifest
            .dependencies
            .get(&dep_name)
            .ok_or_else(|| format!("internal resolver error: missing dependency '{}'", dep_name))?;

        let dep = match dep_spec {
            DependencySpec::Detail(dep) => dep,
            DependencySpec::Version(_) => {
                if offline {
                    return Err(format!(
                        "offline mode cannot resolve non-path dependency '{}'",
                        dep_name
                    ));
                }
                return Err(format!(
                    "dependency '{}' must use a detailed spec with either `path` or `git` in v0.1",
                    dep_name
                ));
            }
        };

        let (dep_path, source_kind, source_url, source_rev) = if let Some(path) = dep.path.as_ref()
        {
            (PathBuf::from(path), "path".to_string(), None, None)
        } else if let Some(git_url) = dep.git.as_ref() {
            let rev = dep
                .rev
                .clone()
                .or_else(|| dep.version.clone())
                .unwrap_or_else(|| "main".to_string());
            let cloned_dir = resolve_git_dependency(&dep_name, git_url, &rev, offline)?;
            (cloned_dir, "git".to_string(), Some(git_url.clone()), Some(rev))
        } else {
            if offline {
                return Err(format!(
                    "offline mode cannot resolve non-path dependency '{}'",
                    dep_name
                ));
            }
            return Err(format!(
                "dependency '{}' must set either `path` or `git` in v0.1",
                dep_name
            ));
        };

        let dep_manifest_path = dep_path.join("Magpie.toml");
        let (resolved_name, resolved_version) = if dep_manifest_path.is_file() {
            match parse_manifest(&dep_manifest_path) {
                Ok(m) => (m.package.name, m.package.version),
                Err(_) => (
                    dep_name.clone(),
                    dep.version.clone().unwrap_or_else(|| "0.0.0".to_string()),
                ),
            }
        } else {
            (
                dep_name.clone(),
                dep.version.clone().unwrap_or_else(|| "0.0.0".to_string()),
            )
        };

        let content_hash = stable_hash_hex(&format!(
            "{}|{}|{}",
            resolved_name,
            resolved_version,
            dep_path.to_string_lossy()
        ));

        packages.push(LockPackage {
            name: resolved_name,
            version: resolved_version,
            source: LockSource {
                kind: source_kind,
                registry: None,
                url: source_url,
                path: Some(dep_path.to_string_lossy().to_string()),
                rev: source_rev,
            },
            content_hash,
            deps: Vec::new(),
            resolved_features: dep.features.clone(),
            targets: Vec::new(),
        });
    }

    Ok(LockFile {
        lock_version: 1,
        generated_by: GeneratedBy {
            magpie_version: env!("CARGO_PKG_VERSION").to_string(),
            toolchain_hash: "unknown".to_string(),
        },
        packages,
    })
}

fn resolve_git_dependency(
    dep_name: &str,
    git_url: &str,
    rev: &str,
    offline: bool,
) -> Result<PathBuf, String> {
    if offline {
        return Err(format!(
            "offline mode cannot resolve git dependency '{}'",
            dep_name
        ));
    }

    let hash = stable_hash_hex(&format!("{dep_name}|{git_url}|{rev}"));
    let cache_dir = PathBuf::from(".magpie").join("deps");
    let target_dir = cache_dir.join(format!("{dep_name}-{hash}"));

    if target_dir.is_dir() {
        return Ok(target_dir);
    }
    if target_dir.exists() {
        return Err(format!(
            "dependency cache path '{}' exists but is not a directory",
            target_dir.display()
        ));
    }

    fs::create_dir_all(&cache_dir)
        .map_err(|e| format!("failed to create cache directory '{}': {}", cache_dir.display(), e))?;

    let status = Command::new("git")
        .arg("clone")
        .arg("--depth")
        .arg("1")
        .arg("--branch")
        .arg(rev)
        .arg(git_url)
        .arg(&target_dir)
        .status()
        .map_err(|e| format!("failed to run git clone for dependency '{}': {}", dep_name, e))?;

    if !status.success() {
        return Err(format!(
            "git clone failed for dependency '{}' from '{}' at '{}' with status {}",
            dep_name, git_url, rev, status
        ));
    }

    Ok(target_dir)
}

pub fn resolve_features(manifest: &Manifest, active: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut modules = Vec::new();

    for feature_name in active {
        if let Some(feature) = manifest.features.get(feature_name) {
            for module in &feature.modules {
                if seen.insert(module.clone()) {
                    modules.push(module.clone());
                }
            }
        }
    }

    modules
}

fn canonical_json(value: &Value) -> String {
    let mut out = String::new();
    write_canonical_json(value, &mut out);
    out
}

fn write_canonical_json(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(v) => {
            if *v {
                out.push_str("true");
            } else {
                out.push_str("false");
            }
        }
        Value::Number(v) => out.push_str(&v.to_string()),
        Value::String(v) => {
            let escaped = serde_json::to_string(v).expect("string JSON encoding cannot fail");
            out.push_str(&escaped);
        }
        Value::Array(items) => {
            out.push('[');
            for (idx, item) in items.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                write_canonical_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            let mut keys = map.keys().cloned().collect::<Vec<_>>();
            keys.sort_unstable();
            for (idx, key) in keys.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                let encoded_key =
                    serde_json::to_string(key).expect("object key JSON encoding cannot fail");
                out.push_str(&encoded_key);
                out.push(':');
                if let Some(item) = map.get(key) {
                    write_canonical_json(item, out);
                }
            }
            out.push('}');
        }
    }
}

fn stable_hash_hex(input: &str) -> String {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    let mut hash = OFFSET_BASIS;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    format!("{:016x}", hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn write_temp_manifest(contents: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "magpie_pkg_manifest_{}_{}.toml",
            std::process::id(),
            unique
        ));
        fs::write(&path, contents).expect("failed to write temporary manifest");
        path
    }

    #[test]
    fn parse_manifest_reads_core_sections_and_dependencies() {
        let manifest_text = r#"
[package]
name = "demo"
version = "0.1.0"
edition = "2024"

[build]
entry = "src/main.mp"
profile_default = "dev"
max_mono_instances = 128

[dependencies]
util = { path = "../util", version = "1.2.3", features = ["serde"], optional = true }

[features]
default = { modules = ["core.main", "core.util"] }
"#;
        let path = write_temp_manifest(manifest_text);
        let parsed = parse_manifest(&path).expect("manifest should parse");

        assert_eq!(parsed.package.name, "demo");
        assert_eq!(parsed.package.version, "0.1.0");
        assert_eq!(parsed.build.entry, "src/main.mp");
        assert_eq!(parsed.build.max_mono_instances, Some(128));
        assert!(parsed.dependencies.contains_key("util"));

        match parsed.dependencies.get("util") {
            Some(DependencySpec::Detail(dep)) => {
                assert_eq!(dep.path.as_deref(), Some("../util"));
                assert_eq!(dep.version.as_deref(), Some("1.2.3"));
                assert_eq!(dep.features, vec!["serde".to_string()]);
                assert!(dep.optional);
            }
            _ => panic!("expected detailed dependency spec for util"),
        }

        let enabled_modules = resolve_features(&parsed, &[String::from("default")]);
        assert_eq!(enabled_modules, vec!["core.main", "core.util"]);

        fs::remove_file(path).expect("failed to remove temporary manifest");
    }

    #[test]
    fn parse_manifest_reports_invalid_toml() {
        let path = write_temp_manifest("[package]\nname = 42\n");
        let err = parse_manifest(&path).expect_err("invalid manifest should fail");

        assert!(err.contains("failed to parse manifest"));
        assert!(
            err.contains(&path.display().to_string()),
            "error should include the manifest path"
        );

        fs::remove_file(path).expect("failed to remove temporary manifest");
    }
}
