//! magpie_pkg

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

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
                    "dependency '{}' is not a path dependency; only path dependencies are supported in v0.1",
                    dep_name
                ));
            }
        };

        let dep_path = match dep.path.as_ref() {
            Some(path) => path.clone(),
            None => {
                if offline {
                    return Err(format!(
                        "offline mode cannot resolve non-path dependency '{}'",
                        dep_name
                    ));
                }
                return Err(format!(
                    "dependency '{}' is not a path dependency; only path dependencies are supported in v0.1",
                    dep_name
                ));
            }
        };

        let dep_manifest_path = PathBuf::from(&dep_path).join("Magpie.toml");
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
            resolved_name, resolved_version, dep_path
        ));

        packages.push(LockPackage {
            name: resolved_name,
            version: resolved_version,
            source: LockSource {
                kind: "path".to_string(),
                registry: None,
                url: None,
                path: Some(dep_path),
                rev: None,
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
