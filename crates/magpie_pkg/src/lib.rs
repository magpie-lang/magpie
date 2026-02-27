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
    let mut resolver = DependencyResolver::new(offline);
    resolver.resolve_manifest_dependencies(manifest, Path::new("."), None)?;
    let mut packages = resolver.into_packages();
    packages.sort_by(cmp_lock_package);

    Ok(LockFile {
        lock_version: 1,
        generated_by: GeneratedBy {
            magpie_version: env!("CARGO_PKG_VERSION").to_string(),
            toolchain_hash: "unknown".to_string(),
        },
        packages,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VisitState {
    Visiting,
    Visited,
}

#[derive(Debug, Clone)]
struct StackFrame {
    key: String,
    name: String,
}

#[derive(Debug, Clone)]
struct ResolvedNode {
    key: String,
    name: String,
    version: String,
    req: String,
    features: Vec<String>,
    source: LockSource,
    content_hash: String,
    path: PathBuf,
    manifest: Option<Manifest>,
}

struct DependencyResolver {
    offline: bool,
    packages: HashMap<String, LockPackage>,
    states: HashMap<String, VisitState>,
    stack: Vec<StackFrame>,
}

impl DependencyResolver {
    fn new(offline: bool) -> Self {
        Self {
            offline,
            packages: HashMap::new(),
            states: HashMap::new(),
            stack: Vec::new(),
        }
    }

    fn resolve_manifest_dependencies(
        &mut self,
        manifest: &Manifest,
        manifest_dir: &Path,
        parent_key: Option<&str>,
    ) -> Result<(), String> {
        let mut dep_names = manifest.dependencies.keys().cloned().collect::<Vec<_>>();
        dep_names.sort_unstable();

        let mut resolved_deps = Vec::new();
        for dep_name in dep_names {
            let dep_spec = manifest.dependencies.get(&dep_name).ok_or_else(|| {
                format!("internal resolver error: missing dependency '{}'", dep_name)
            })?;

            let dep = self.normalize_dependency_spec(dep_spec);
            let mut resolved = self.resolve_node(&dep_name, &dep, manifest_dir)?;
            self.insert_or_merge_package(&resolved);

            if let Some(existing) = self.packages.get(&resolved.key) {
                resolved.name = existing.name.clone();
                resolved.version = existing.version.clone();
            }

            resolved_deps.push(LockDependency {
                name: resolved.name.clone(),
                req: resolved.req.clone(),
                features: resolved.features.clone(),
            });

            self.visit_node(&resolved)?;
        }

        resolved_deps.sort_by(cmp_lock_dependency);
        if let Some(key) = parent_key {
            if let Some(pkg) = self.packages.get_mut(key) {
                pkg.deps = resolved_deps;
            } else {
                return Err(format!(
                    "internal resolver error: missing package state for '{}'",
                    key
                ));
            }
        }

        Ok(())
    }

    fn normalize_dependency_spec(&self, dep_spec: &DependencySpec) -> Dependency {
        match dep_spec {
            DependencySpec::Detail(dep) => dep.clone(),
            DependencySpec::Version(version_req) => Dependency {
                version: Some(version_req.clone()),
                registry: Some("default".to_string()),
                path: None,
                git: None,
                rev: None,
                features: Vec::new(),
                optional: false,
            },
        }
    }

    fn resolve_node(
        &self,
        dep_name: &str,
        dep: &Dependency,
        manifest_dir: &Path,
    ) -> Result<ResolvedNode, String> {
        let mut features = dep.features.clone();
        features.sort_unstable();
        features.dedup();

        let (dep_path, source) = if let Some(path) = dep.path.as_ref() {
            let dep_path = resolve_path_dependency(manifest_dir, path);
            let source = LockSource {
                kind: "path".to_string(),
                registry: None,
                url: None,
                path: Some(dep_path.to_string_lossy().to_string()),
                rev: None,
            };
            (dep_path, source)
        } else if let Some(git_url) = dep.git.as_ref() {
            let rev = dep
                .rev
                .clone()
                .or_else(|| dep.version.clone())
                .unwrap_or_else(|| "main".to_string());
            let dep_path = resolve_git_dependency(dep_name, git_url, &rev, self.offline)?;
            let source = LockSource {
                kind: "git".to_string(),
                registry: None,
                url: Some(git_url.clone()),
                path: Some(dep_path.to_string_lossy().to_string()),
                rev: Some(rev),
            };
            (dep_path, source)
        } else {
            let version_tag = dep.version.as_deref().unwrap_or("0.0.0");
            let dep_path = PathBuf::from(".magpie")
                .join("registry")
                .join(dep_name)
                .join(version_tag);
            let source = LockSource {
                kind: "registry".to_string(),
                registry: dep.registry.clone().or_else(|| Some("default".to_string())),
                url: None,
                path: None,
                rev: None,
            };
            (dep_path, source)
        };

        let dep_manifest_path = dep_path.join("Magpie.toml");
        let (resolved_name, resolved_version, resolved_manifest) = if dep_manifest_path.is_file() {
            match parse_manifest(&dep_manifest_path) {
                Ok(m) => {
                    let name = m.package.name.clone();
                    let version = m.package.version.clone();
                    (name, version, Some(m))
                }
                Err(_) => (
                    dep_name.to_string(),
                    dep.version.clone().unwrap_or_else(|| "0.0.0".to_string()),
                    None,
                ),
            }
        } else {
            (
                dep_name.to_string(),
                dep.version.clone().unwrap_or_else(|| "0.0.0".to_string()),
                None,
            )
        };

        let req = dep
            .version
            .clone()
            .unwrap_or_else(|| resolved_version.clone());

        let content_hash = stable_hash_hex(&format!(
            "{}|{}|{}",
            resolved_name,
            resolved_version,
            source.path.as_deref().unwrap_or("")
        ));

        let key = source_key(&source, &dep_path);

        Ok(ResolvedNode {
            key,
            name: resolved_name,
            version: resolved_version,
            req,
            features,
            source,
            content_hash,
            path: dep_path,
            manifest: resolved_manifest,
        })
    }

    fn insert_or_merge_package(&mut self, resolved: &ResolvedNode) {
        let entry = self
            .packages
            .entry(resolved.key.clone())
            .or_insert_with(|| LockPackage {
                name: resolved.name.clone(),
                version: resolved.version.clone(),
                source: resolved.source.clone(),
                content_hash: resolved.content_hash.clone(),
                deps: Vec::new(),
                resolved_features: resolved.features.clone(),
                targets: Vec::new(),
            });

        if entry.resolved_features.is_empty() {
            entry.resolved_features = resolved.features.clone();
        } else {
            entry
                .resolved_features
                .extend(resolved.features.iter().cloned());
            entry.resolved_features.sort_unstable();
            entry.resolved_features.dedup();
        }
    }

    fn visit_node(&mut self, resolved: &ResolvedNode) -> Result<(), String> {
        match self.states.get(&resolved.key) {
            Some(VisitState::Visited) => return Ok(()),
            Some(VisitState::Visiting) => {
                return Err(format!(
                    "dependency cycle detected: {}",
                    self.describe_cycle(&resolved.key, &resolved.name)
                ));
            }
            None => {}
        }

        self.states
            .insert(resolved.key.clone(), VisitState::Visiting);
        self.stack.push(StackFrame {
            key: resolved.key.clone(),
            name: resolved.name.clone(),
        });

        let result = if let Some(manifest) = resolved.manifest.as_ref() {
            self.resolve_manifest_dependencies(manifest, &resolved.path, Some(&resolved.key))
        } else {
            Ok(())
        };

        self.stack.pop();

        match result {
            Ok(()) => {
                self.states
                    .insert(resolved.key.clone(), VisitState::Visited);
                Ok(())
            }
            Err(err) => {
                self.states.remove(&resolved.key);
                Err(err)
            }
        }
    }

    fn describe_cycle(&self, key: &str, fallback_name: &str) -> String {
        if let Some(start) = self.stack.iter().position(|frame| frame.key == key) {
            let mut names = self.stack[start..]
                .iter()
                .map(|frame| frame.name.clone())
                .collect::<Vec<_>>();
            if let Some(first) = names.first().cloned() {
                names.push(first);
            }
            return names.join(" -> ");
        }

        format!("{fallback_name} -> {fallback_name}")
    }

    fn into_packages(mut self) -> Vec<LockPackage> {
        let mut packages = self
            .packages
            .drain()
            .map(|(_, pkg)| pkg)
            .collect::<Vec<_>>();
        for pkg in &mut packages {
            pkg.deps.sort_by(cmp_lock_dependency);
            pkg.resolved_features.sort_unstable();
            pkg.resolved_features.dedup();
        }
        packages
    }
}

fn resolve_path_dependency(base_dir: &Path, path: &str) -> PathBuf {
    let raw = PathBuf::from(path);
    if raw.is_absolute() || base_dir == Path::new(".") {
        raw
    } else {
        base_dir.join(raw)
    }
}

fn normalize_path(path: &Path) -> PathBuf {
    use std::path::Component;

    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                let can_pop = normalized
                    .components()
                    .next_back()
                    .map(|last| {
                        !matches!(
                            last,
                            Component::RootDir | Component::Prefix(_) | Component::ParentDir
                        )
                    })
                    .unwrap_or(false);
                if can_pop {
                    normalized.pop();
                } else {
                    normalized.push(component.as_os_str());
                }
            }
            _ => normalized.push(component.as_os_str()),
        }
    }

    if normalized.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        normalized
    }
}

fn source_key(source: &LockSource, path: &Path) -> String {
    let identity_path = fs::canonicalize(path)
        .unwrap_or_else(|_| normalize_path(path))
        .to_string_lossy()
        .to_string();

    format!(
        "{}|{}|{}|{}|{}",
        source.kind,
        source.registry.as_deref().unwrap_or(""),
        source.url.as_deref().unwrap_or(""),
        identity_path,
        source.rev.as_deref().unwrap_or(""),
    )
}

fn cmp_lock_dependency(a: &LockDependency, b: &LockDependency) -> std::cmp::Ordering {
    a.name
        .cmp(&b.name)
        .then_with(|| a.req.cmp(&b.req))
        .then_with(|| a.features.cmp(&b.features))
}

fn cmp_lock_source(a: &LockSource, b: &LockSource) -> std::cmp::Ordering {
    a.kind
        .cmp(&b.kind)
        .then_with(|| a.registry.cmp(&b.registry))
        .then_with(|| a.url.cmp(&b.url))
        .then_with(|| a.path.cmp(&b.path))
        .then_with(|| a.rev.cmp(&b.rev))
}

fn cmp_lock_package(a: &LockPackage, b: &LockPackage) -> std::cmp::Ordering {
    a.name
        .cmp(&b.name)
        .then_with(|| a.version.cmp(&b.version))
        .then_with(|| cmp_lock_source(&a.source, &b.source))
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

    fs::create_dir_all(&cache_dir).map_err(|e| {
        format!(
            "failed to create cache directory '{}': {}",
            cache_dir.display(),
            e
        )
    })?;

    let status = Command::new("git")
        .arg("clone")
        .arg("--depth")
        .arg("1")
        .arg("--branch")
        .arg(rev)
        .arg(git_url)
        .arg(&target_dir)
        .status()
        .map_err(|e| {
            format!(
                "failed to run git clone for dependency '{}': {}",
                dep_name, e
            )
        })?;

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
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEST_TEMP_UNIQUE_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_temp_token(prefix: &str) -> String {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        let counter = TEST_TEMP_UNIQUE_COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("{prefix}_{}_{}_{}", std::process::id(), unique, counter)
    }

    fn write_temp_manifest(contents: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "magpie_pkg_manifest_{}.toml",
            unique_temp_token("manifest")
        ));
        fs::write(&path, contents).expect("failed to write temporary manifest");
        path
    }

    fn create_temp_dir(prefix: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("magpie_pkg_{}", unique_temp_token(prefix)));
        fs::create_dir_all(&dir).expect("failed to create temporary directory");
        dir
    }

    fn write_package_manifest(dir: &Path, name: &str, version: &str, deps_block: Option<&str>) {
        fs::create_dir_all(dir).expect("failed to create package directory");

        let mut manifest = format!(
            r#"[package]
name = "{name}"
version = "{version}"
edition = "2024"

[build]
entry = "src/lib.mp"
profile_default = "dev"
"#
        );

        if let Some(deps) = deps_block {
            manifest.push_str("\n[dependencies]\n");
            manifest.push_str(deps);
            manifest.push('\n');
        }

        fs::write(dir.join("Magpie.toml"), manifest).expect("failed to write package manifest");
    }

    fn root_manifest_with_path_deps(
        deps: Vec<(&str, PathBuf, Option<&str>, Vec<&str>)>,
    ) -> Manifest {
        let mut dependencies = HashMap::new();
        for (dep_name, dep_path, version, features) in deps {
            dependencies.insert(
                dep_name.to_string(),
                DependencySpec::Detail(Dependency {
                    version: version.map(ToString::to_string),
                    registry: None,
                    path: Some(dep_path.to_string_lossy().to_string()),
                    git: None,
                    rev: None,
                    features: features.into_iter().map(ToString::to_string).collect(),
                    optional: false,
                }),
            );
        }

        Manifest {
            package: PackageSection {
                name: "root".to_string(),
                version: "0.1.0".to_string(),
                edition: "2024".to_string(),
            },
            build: BuildSection {
                entry: "src/main.mp".to_string(),
                profile_default: "dev".to_string(),
                max_mono_instances: None,
            },
            dependencies,
            features: HashMap::new(),
            llm: None,
            web: None,
            gpu: None,
            toolchain: HashMap::new(),
        }
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

    #[test]
    fn resolve_deps_resolves_transitive_path_dependencies() {
        let workspace = create_temp_dir("transitive");
        let dep_a = workspace.join("dep-a");
        let dep_b = workspace.join("dep-b");

        write_package_manifest(&dep_b, "dep_b_pkg", "0.2.0", None);
        write_package_manifest(
            &dep_a,
            "dep_a_pkg",
            "0.1.0",
            Some(r#"dep_b = { path = "../dep-b", version = "0.2.0", features = ["feat-b"] }"#),
        );

        let manifest = root_manifest_with_path_deps(vec![(
            "dep_a",
            dep_a.clone(),
            Some("0.1.0"),
            vec!["feat-a"],
        )]);

        let lock = resolve_deps(&manifest, true).expect("path dependencies should resolve");
        assert_eq!(lock.packages.len(), 2);

        let dep_a_pkg = lock
            .packages
            .iter()
            .find(|pkg| pkg.name == "dep_a_pkg")
            .expect("dep_a_pkg should be present");
        assert_eq!(dep_a_pkg.deps.len(), 1);
        assert_eq!(dep_a_pkg.deps[0].name, "dep_b_pkg");
        assert_eq!(dep_a_pkg.deps[0].req, "0.2.0");
        assert_eq!(dep_a_pkg.deps[0].features, vec!["feat-b"]);
        assert_eq!(dep_a_pkg.resolved_features, vec!["feat-a"]);

        let dep_b_pkg = lock
            .packages
            .iter()
            .find(|pkg| pkg.name == "dep_b_pkg")
            .expect("dep_b_pkg should be present");
        assert!(dep_b_pkg.deps.is_empty());

        fs::remove_dir_all(workspace).expect("failed to clean temporary workspace");
    }

    #[test]
    fn resolve_deps_sorts_packages_deterministically() {
        let workspace = create_temp_dir("ordering");
        let alpha_a = workspace.join("alpha-a");
        let alpha_b = workspace.join("alpha-b");
        let alpha_v2 = workspace.join("alpha-v2");
        let zeta = workspace.join("zeta");

        write_package_manifest(&alpha_a, "alpha", "1.0.0", None);
        write_package_manifest(&alpha_b, "alpha", "1.0.0", None);
        write_package_manifest(&alpha_v2, "alpha", "2.0.0", None);
        write_package_manifest(&zeta, "zeta", "0.1.0", None);

        let manifest = root_manifest_with_path_deps(vec![
            ("dep_z", zeta.clone(), Some("0.1.0"), vec![]),
            ("dep_alpha_v2", alpha_v2.clone(), Some("2.0.0"), vec![]),
            ("dep_alpha_b", alpha_b.clone(), Some("1.0.0"), vec![]),
            ("dep_alpha_a", alpha_a.clone(), Some("1.0.0"), vec![]),
        ]);

        let lock = resolve_deps(&manifest, true).expect("path dependencies should resolve");
        let actual = lock
            .packages
            .iter()
            .map(|pkg| {
                (
                    pkg.name.clone(),
                    pkg.version.clone(),
                    pkg.source.path.clone().unwrap_or_default(),
                )
            })
            .collect::<Vec<_>>();

        let expected = vec![
            (
                "alpha".to_string(),
                "1.0.0".to_string(),
                alpha_a.to_string_lossy().to_string(),
            ),
            (
                "alpha".to_string(),
                "1.0.0".to_string(),
                alpha_b.to_string_lossy().to_string(),
            ),
            (
                "alpha".to_string(),
                "2.0.0".to_string(),
                alpha_v2.to_string_lossy().to_string(),
            ),
            (
                "zeta".to_string(),
                "0.1.0".to_string(),
                zeta.to_string_lossy().to_string(),
            ),
        ];

        assert_eq!(actual, expected);

        fs::remove_dir_all(workspace).expect("failed to clean temporary workspace");
    }

    #[test]
    fn resolve_deps_accepts_version_only_registry_dependencies() {
        let mut dependencies = HashMap::new();
        dependencies.insert(
            "std".to_string(),
            DependencySpec::Version("^0.1".to_string()),
        );

        let manifest = Manifest {
            package: PackageSection {
                name: "root".to_string(),
                version: "0.1.0".to_string(),
                edition: "2024".to_string(),
            },
            build: BuildSection {
                entry: "src/main.mp".to_string(),
                profile_default: "dev".to_string(),
                max_mono_instances: None,
            },
            dependencies,
            features: HashMap::new(),
            llm: None,
            web: None,
            gpu: None,
            toolchain: HashMap::new(),
        };

        let lock = resolve_deps(&manifest, true).expect("version-only deps should resolve");
        assert_eq!(lock.packages.len(), 1);
        let std_pkg = &lock.packages[0];
        assert_eq!(std_pkg.name, "std");
        assert_eq!(std_pkg.version, "^0.1");
        assert_eq!(std_pkg.source.kind, "registry");
        assert_eq!(std_pkg.source.registry.as_deref(), Some("default"));
        assert!(std_pkg.source.path.is_none());
    }

    #[test]
    fn resolve_deps_reports_dependency_cycles() {
        let workspace = create_temp_dir("cycle");
        let dep_a = workspace.join("dep-a");
        let dep_b = workspace.join("dep-b");

        write_package_manifest(
            &dep_a,
            "dep_a_pkg",
            "0.1.0",
            Some(r#"dep_b = { path = "../dep-b", version = "0.1.0" }"#),
        );
        write_package_manifest(
            &dep_b,
            "dep_b_pkg",
            "0.1.0",
            Some(r#"dep_a = { path = "../dep-a", version = "0.1.0" }"#),
        );

        let manifest =
            root_manifest_with_path_deps(vec![("dep_a", dep_a.clone(), Some("0.1.0"), vec![])]);

        let err = resolve_deps(&manifest, true).expect_err("cycle should be reported");
        assert!(err.contains("dependency cycle detected"));
        assert!(err.contains("dep_a_pkg"));
        assert!(err.contains("dep_b_pkg"));

        fs::remove_dir_all(workspace).expect("failed to clean temporary workspace");
    }
}
