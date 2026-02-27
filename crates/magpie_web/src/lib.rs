//! magpie_web
#![allow(clippy::result_large_err)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TRequest {
    pub method: String,
    pub path: String,
    pub query: HashMap<String, String>,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
    pub path_params: HashMap<String, String>,
    pub remote_addr: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TResponse {
    pub status: i32,
    pub headers: HashMap<String, String>,
    pub body_kind: i32,
    pub body_bytes: Vec<u8>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TContext {
    pub state: HashMap<String, String>,
    pub request_id: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TRoute {
    pub method: String,
    pub pattern: String,
    pub handler_name: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TService {
    pub prefix: String,
    pub routes: Vec<TRoute>,
    pub middleware: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum RouteParamType {
    I32,
    I64,
    U32,
    U64,
    Bool,
    Str,
}

impl RouteParamType {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "i32" => Some(Self::I32),
            "i64" => Some(Self::I64),
            "u32" => Some(Self::U32),
            "u64" => Some(Self::U64),
            "bool" => Some(Self::Bool),
            "Str" => Some(Self::Str),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum RouteSegment {
    Literal(String),
    Param { name: String, ty: RouteParamType },
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct RoutePattern {
    pub segments: Vec<RouteSegment>,
    pub wildcard: Option<String>,
}

fn is_valid_ident(input: &str) -> bool {
    let mut chars = input.chars();
    match chars.next() {
        Some(c) if c == '_' || c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

pub fn parse_route_pattern(pattern: &str) -> Result<RoutePattern, String> {
    if pattern.is_empty() {
        return Err("route pattern cannot be empty".to_string());
    }
    if !pattern.starts_with('/') {
        return Err("route pattern must start with '/'".to_string());
    }
    if pattern == "/" {
        return Ok(RoutePattern::default());
    }

    let raw_segments: Vec<&str> = pattern[1..].split('/').collect();
    if raw_segments.iter().any(|seg| seg.is_empty()) {
        return Err("route pattern cannot contain empty segments".to_string());
    }

    let mut parsed = RoutePattern::default();
    let mut seen_names: HashMap<String, ()> = HashMap::new();

    for (idx, seg) in raw_segments.iter().enumerate() {
        if seg.starts_with("*{") && seg.ends_with('}') {
            if idx != raw_segments.len() - 1 {
                return Err("wildcard segment must be the final segment".to_string());
            }
            let name = &seg[2..seg.len() - 1];
            if !is_valid_ident(name) {
                return Err(format!("invalid wildcard parameter name '{name}'"));
            }
            if seen_names.insert(name.to_string(), ()).is_some() {
                return Err(format!("duplicate parameter name '{name}'"));
            }
            parsed.wildcard = Some(name.to_string());
            continue;
        }

        if seg.starts_with('{') && seg.ends_with('}') {
            let inner = &seg[1..seg.len() - 1];
            let (name, ty_name) = inner
                .split_once(':')
                .ok_or_else(|| "typed parameter must use {name:type}".to_string())?;
            if !is_valid_ident(name) {
                return Err(format!("invalid parameter name '{name}'"));
            }
            let ty = RouteParamType::parse(ty_name)
                .ok_or_else(|| format!("unsupported route param type '{ty_name}'"))?;
            if seen_names.insert(name.to_string(), ()).is_some() {
                return Err(format!("duplicate parameter name '{name}'"));
            }
            parsed.segments.push(RouteSegment::Param {
                name: name.to_string(),
                ty,
            });
            continue;
        }

        if seg.contains('{') || seg.contains('}') {
            return Err(format!("invalid literal segment '{seg}'"));
        }

        parsed
            .segments
            .push(RouteSegment::Literal((*seg).to_string()));
    }

    Ok(parsed)
}

pub fn match_route(pattern: &RoutePattern, path: &str) -> Option<HashMap<String, String>> {
    if !path.starts_with('/') {
        return None;
    }

    let path_segments: Vec<&str> = if path == "/" {
        Vec::new()
    } else {
        let parts: Vec<&str> = path[1..].split('/').collect();
        if parts.iter().any(|seg| seg.is_empty()) {
            return None;
        }
        parts
    };

    let fixed_len = pattern.segments.len();
    if path_segments.len() < fixed_len {
        return None;
    }
    if pattern.wildcard.is_none() && path_segments.len() != fixed_len {
        return None;
    }

    let mut captures = HashMap::new();

    for (idx, seg) in pattern.segments.iter().enumerate() {
        match seg {
            RouteSegment::Literal(expected) => {
                if path_segments[idx] != expected {
                    return None;
                }
            }
            RouteSegment::Param { name, .. } => {
                captures.insert(name.clone(), path_segments[idx].to_string());
            }
        }
    }

    if let Some(name) = &pattern.wildcard {
        let rest = if path_segments.len() == fixed_len {
            String::new()
        } else {
            path_segments[fixed_len..].join("/")
        };
        captures.insert(name.clone(), rest);
    }

    Some(captures)
}

fn normalize_prefix(prefix: &str) -> String {
    if prefix.is_empty() || prefix == "/" {
        return String::new();
    }
    let mut normalized = if prefix.starts_with('/') {
        prefix.to_string()
    } else {
        format!("/{prefix}")
    };
    while normalized.ends_with('/') {
        normalized.pop();
    }
    if normalized == "/" {
        String::new()
    } else {
        normalized
    }
}

fn strip_prefix<'a>(path: &'a str, prefix: &str) -> Option<&'a str> {
    if prefix.is_empty() {
        return Some(path);
    }
    if path == prefix {
        return Some("/");
    }
    let suffix = path.strip_prefix(prefix)?;
    if !suffix.starts_with('/') {
        return None;
    }
    Some(suffix)
}

fn json_response(status: i32, request_id: &str, body: serde_json::Value) -> TResponse {
    let mut headers = HashMap::new();
    headers.insert(
        "content-type".to_string(),
        "application/json; charset=utf-8".to_string(),
    );
    headers.insert("x-request-id".to_string(), request_id.to_string());

    TResponse {
        status,
        headers,
        body_kind: 0,
        body_bytes: serde_json::to_vec(&body).unwrap_or_default(),
    }
}

pub fn dispatch(service: &TService, req: &TRequest, ctx: &TContext) -> TResponse {
    let method = req.method.to_ascii_uppercase();
    let req_path = if req.path.starts_with('/') {
        req.path.as_str()
    } else {
        return json_response(
            400,
            &ctx.request_id,
            serde_json::json!({
                "error": "bad_request",
                "message": "request path must start with '/'",
                "request_id": ctx.request_id,
            }),
        );
    };

    let prefix = normalize_prefix(&service.prefix);
    let effective_path = match strip_prefix(req_path, &prefix) {
        Some(path) => path,
        None => {
            return json_response(
                404,
                &ctx.request_id,
                serde_json::json!({
                    "error": "not_found",
                    "request_id": ctx.request_id,
                }),
            );
        }
    };

    for route in &service.routes {
        if route.method.to_ascii_uppercase() != method {
            continue;
        }

        let pattern = match parse_route_pattern(&route.pattern) {
            Ok(p) => p,
            Err(_) => continue,
        };

        if let Some(params) = match_route(&pattern, effective_path) {
            return json_response(
                200,
                &ctx.request_id,
                serde_json::json!({
                    "handler": route.handler_name,
                    "path_params": params,
                    "request_id": ctx.request_id,
                }),
            );
        }
    }

    json_response(
        404,
        &ctx.request_id,
        serde_json::json!({
            "error": "not_found",
            "request_id": ctx.request_id,
        }),
    )
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TNode {
    pub tag: String,
    pub text: String,
    pub attrs: Vec<(String, String)>,
    pub children: Vec<TNode>,
}

fn escape_html(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(ch),
        }
    }
    out
}

pub fn render_html(node: &TNode) -> String {
    match node.tag.as_str() {
        "#text" => return escape_html(&node.text),
        "#raw" => return node.text.clone(),
        _ => {}
    }

    let mut out = String::new();
    out.push('<');
    out.push_str(&node.tag);
    for (key, value) in &node.attrs {
        out.push(' ');
        out.push_str(key);
        out.push_str("=\"");
        out.push_str(&escape_html(value));
        out.push('"');
    }
    out.push('>');

    if !node.text.is_empty() {
        out.push_str(&escape_html(&node.text));
    }

    for child in &node.children {
        out.push_str(&render_html(child));
    }

    out.push_str("</");
    out.push_str(&node.tag);
    out.push('>');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_match_typed_pattern() {
        let pattern = parse_route_pattern("/users/{id:u64}/posts/{slug:Str}").unwrap();
        let params = match_route(&pattern, "/users/42/posts/hello").unwrap();
        assert_eq!(params.get("id").map(String::as_str), Some("42"));
        assert_eq!(params.get("slug").map(String::as_str), Some("hello"));
    }

    #[test]
    fn wildcard_matches_tail() {
        let pattern = parse_route_pattern("/assets/*{path}").unwrap();
        let params = match_route(&pattern, "/assets/css/app.css").unwrap();
        assert_eq!(params.get("path").map(String::as_str), Some("css/app.css"));
    }

    #[test]
    fn dispatch_injects_request_id() {
        let service = TService {
            prefix: "/api".to_string(),
            routes: vec![TRoute {
                method: "GET".to_string(),
                pattern: "/users/{id:u64}".to_string(),
                handler_name: "get_user".to_string(),
            }],
            middleware: Vec::new(),
        };

        let req = TRequest {
            method: "GET".to_string(),
            path: "/api/users/7".to_string(),
            ..TRequest::default()
        };
        let ctx = TContext {
            request_id: "req-123".to_string(),
            ..TContext::default()
        };

        let resp = dispatch(&service, &req, &ctx);
        assert_eq!(resp.status, 200);
        assert_eq!(
            resp.headers.get("x-request-id").map(String::as_str),
            Some("req-123")
        );
    }

    #[test]
    fn render_html_escapes_text() {
        let node = TNode {
            tag: "div".to_string(),
            text: String::new(),
            attrs: vec![("data-x".to_string(), "a&b".to_string())],
            children: vec![TNode {
                tag: "#text".to_string(),
                text: "<hello>".to_string(),
                attrs: Vec::new(),
                children: Vec::new(),
            }],
        };

        assert_eq!(
            render_html(&node),
            "<div data-x=\"a&amp;b\">&lt;hello&gt;</div>"
        );
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct McpConfig {
    pub allowed_roots: Vec<String>,
    pub allow_network: bool,
    pub allow_subprocess: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct McpServer {
    pub tools: HashMap<String, McpTool>,
    pub config: McpConfig,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct McpRequest {
    pub jsonrpc: String,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct McpError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct McpResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<McpError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<serde_json::Value>,
}

fn default_tool_input_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "llm": {
                "type": "object",
                "properties": {
                    "mode": { "type": "boolean" },
                    "token_budget": { "type": "integer", "minimum": 0 },
                    "tokenizer": { "type": "string" },
                    "policy": { "type": "string" }
                },
                "additionalProperties": false
            }
        },
        "additionalProperties": true
    })
}

fn register_mcp_tool(
    tools: &mut HashMap<String, McpTool>,
    name: &str,
    description: &str,
    input_schema: serde_json::Value,
) {
    tools.insert(
        name.to_string(),
        McpTool {
            name: name.to_string(),
            description: description.to_string(),
            input_schema,
        },
    );
}

fn parse_env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => default,
        },
        Err(_) => default,
    }
}

fn default_mcp_allowed_roots() -> Vec<String> {
    match std::env::var("MAGPIE_MCP_ALLOWED_ROOTS") {
        Ok(value) => {
            let mut roots = value
                .split(',')
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            roots.sort();
            roots.dedup();
            if roots.is_empty() {
                vec![".".to_string()]
            } else {
                roots
            }
        }
        Err(_) => vec![".".to_string()],
    }
}

fn default_mcp_config() -> McpConfig {
    McpConfig {
        allowed_roots: default_mcp_allowed_roots(),
        allow_network: parse_env_bool("MAGPIE_MCP_ALLOW_NETWORK", false),
        allow_subprocess: parse_env_bool("MAGPIE_MCP_ALLOW_SUBPROCESS", true),
    }
}

pub fn create_mcp_server() -> McpServer {
    let mut tools = HashMap::new();
    let schema = default_tool_input_schema();

    register_mcp_tool(
        &mut tools,
        "magpie.build",
        "Build a Magpie project.",
        schema.clone(),
    );
    register_mcp_tool(
        &mut tools,
        "magpie.run",
        "Run a Magpie project.",
        schema.clone(),
    );
    register_mcp_tool(
        &mut tools,
        "magpie.test",
        "Run Magpie tests.",
        schema.clone(),
    );
    register_mcp_tool(
        &mut tools,
        "magpie.fmt",
        "Format Magpie source.",
        schema.clone(),
    );
    register_mcp_tool(
        &mut tools,
        "magpie.lint",
        "Lint Magpie source.",
        schema.clone(),
    );
    register_mcp_tool(
        &mut tools,
        "magpie.explain",
        "Explain diagnostics or code.",
        schema.clone(),
    );
    register_mcp_tool(
        &mut tools,
        "magpie.pkg.resolve",
        "Resolve package graph and lockfile.",
        schema.clone(),
    );
    register_mcp_tool(
        &mut tools,
        "magpie.pkg.plan",
        "Plan package resolution without writing a lockfile.",
        schema.clone(),
    );
    register_mcp_tool(
        &mut tools,
        "magpie.pkg.add",
        "Add a package dependency.",
        schema.clone(),
    );
    register_mcp_tool(
        &mut tools,
        "magpie.pkg.remove",
        "Remove a package dependency.",
        schema.clone(),
    );
    register_mcp_tool(
        &mut tools,
        "magpie.pkg.why",
        "Explain why a dependency is present.",
        schema.clone(),
    );
    register_mcp_tool(
        &mut tools,
        "magpie.memory.build",
        "Build compiler memory index.",
        schema.clone(),
    );
    register_mcp_tool(
        &mut tools,
        "magpie.memory.query",
        "Query compiler memory index.",
        schema.clone(),
    );
    register_mcp_tool(
        &mut tools,
        "magpie.ctx.pack",
        "Build an LLM context pack.",
        schema.clone(),
    );
    register_mcp_tool(
        &mut tools,
        "magpie.repl.create",
        "Create a REPL session.",
        schema.clone(),
    );
    register_mcp_tool(
        &mut tools,
        "magpie.repl.eval",
        "Evaluate a REPL cell.",
        schema.clone(),
    );
    register_mcp_tool(
        &mut tools,
        "magpie.repl.inspect",
        "Inspect REPL session state.",
        schema.clone(),
    );
    register_mcp_tool(
        &mut tools,
        "magpie.graph.symbols",
        "Return symbol graph information.",
        schema.clone(),
    );
    register_mcp_tool(
        &mut tools,
        "magpie.graph.deps",
        "Return dependency graph information.",
        schema.clone(),
    );
    register_mcp_tool(
        &mut tools,
        "magpie.graph.ownership",
        "Return ownership graph information.",
        schema.clone(),
    );
    register_mcp_tool(
        &mut tools,
        "magpie.graph.cfg",
        "Return control-flow graph information.",
        schema,
    );

    McpServer {
        tools,
        config: default_mcp_config(),
    }
}

fn mcp_error_response(
    id: Option<serde_json::Value>,
    code: i32,
    message: &str,
    data: Option<serde_json::Value>,
) -> McpResponse {
    McpResponse {
        jsonrpc: "2.0".to_string(),
        result: None,
        error: Some(McpError {
            code,
            message: message.to_string(),
            data,
        }),
        id,
    }
}

fn json_u32(value: &serde_json::Value) -> Option<u32> {
    value.as_u64().and_then(|raw| u32::try_from(raw).ok())
}

fn parse_cwd_param(params: Option<&serde_json::Value>) -> Result<Option<PathBuf>, String> {
    let Some(params) = params else {
        return Ok(None);
    };
    let Some(cwd_value) = params.get("cwd") else {
        return Ok(None);
    };
    let Some(cwd_str) = cwd_value.as_str() else {
        return Err("invalid params: 'cwd' must be a string".to_string());
    };

    let cwd_path = PathBuf::from(cwd_str);
    if cwd_path.is_absolute() {
        return Ok(Some(cwd_path));
    }

    let base = std::env::current_dir().map_err(|err| format!("failed to resolve cwd: {err}"))?;
    Ok(Some(base.join(cwd_path)))
}

fn resolve_path_from_base(base_dir: &Path, raw: &str) -> PathBuf {
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        path
    } else {
        base_dir.join(path)
    }
}

fn path_within_allowed_roots(path: &Path, allowed_roots: &[String]) -> bool {
    if allowed_roots.is_empty() {
        return false;
    }

    let Ok(path) = std::fs::canonicalize(path) else {
        return false;
    };
    let base_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    allowed_roots.iter().any(|root| {
        let root_path = resolve_path_from_base(&base_dir, root);
        std::fs::canonicalize(&root_path)
            .map(|canonical_root| path.starts_with(&canonical_root))
            .unwrap_or(false)
    })
}

fn tool_requires_network(method: &str) -> bool {
    method.starts_with("magpie.web.")
}

fn tool_requires_subprocess(method: &str) -> bool {
    matches!(
        method,
        "magpie.build"
            | "magpie.run"
            | "magpie.test"
            | "magpie.fmt"
            | "magpie.lint"
            | "magpie.memory.build"
            | "magpie.ctx.pack"
            | "magpie.graph.symbols"
            | "magpie.graph.deps"
            | "magpie.graph.ownership"
            | "magpie.graph.cfg"
    )
}

fn collect_request_paths_for_security(
    method: &str,
    params: &serde_json::Value,
    base_dir: &Path,
) -> Result<Vec<(String, PathBuf)>, String> {
    let mut out = Vec::new();
    let mut push_str_path = |label: &str, raw: &serde_json::Value| -> Result<(), String> {
        let Some(path) = raw.as_str() else {
            return Err(format!("invalid params: '{label}' must be a string"));
        };
        out.push((label.to_string(), resolve_path_from_base(base_dir, path)));
        Ok(())
    };

    let checks_entry_path = matches!(
        method,
        "magpie.build"
            | "magpie.run"
            | "magpie.test"
            | "magpie.fmt"
            | "magpie.lint"
            | "magpie.memory.build"
            | "magpie.ctx.pack"
            | "magpie.graph.symbols"
            | "magpie.graph.deps"
            | "magpie.graph.ownership"
            | "magpie.graph.cfg"
    );
    if checks_entry_path {
        if let Some(value) = params.get("entry_path") {
            push_str_path("entry_path", value)?;
        }
    }

    if matches!(
        method,
        "magpie.pkg.resolve"
            | "magpie.pkg.plan"
            | "magpie.pkg.add"
            | "magpie.pkg.remove"
            | "magpie.pkg.why"
    ) {
        if let Some(value) = params.get("manifest_path") {
            push_str_path("manifest_path", value)?;
        }
    }

    if method == "magpie.fmt" {
        if let Some(value) = params.get("path") {
            push_str_path("path", value)?;
        }
        if let Some(values) = params.get("paths") {
            let Some(values) = values.as_array() else {
                return Err("invalid params: 'paths' must be an array of strings".to_string());
            };
            for (idx, value) in values.iter().enumerate() {
                let label = format!("paths[{idx}]");
                push_str_path(&label, value)?;
            }
        }
    }

    Ok(out)
}

fn enforce_mcp_security(
    server: &McpServer,
    request: &McpRequest,
) -> Result<Option<PathBuf>, McpResponse> {
    let cwd = match parse_cwd_param(request.params.as_ref()) {
        Ok(cwd) => cwd,
        Err(message) => {
            return Err(mcp_error_response(
                request.id.clone(),
                -32602,
                &message,
                request.params.clone(),
            ));
        }
    };

    if let Some(cwd_path) = cwd.as_ref() {
        if !path_within_allowed_roots(cwd_path, &server.config.allowed_roots) {
            return Err(mcp_error_response(
                request.id.clone(),
                -32001,
                "security policy denied cwd",
                Some(serde_json::json!({
                    "method": request.method.as_str(),
                    "cwd": cwd_path.to_string_lossy().to_string(),
                    "allowed_roots": server.config.allowed_roots.clone(),
                })),
            ));
        }
    }

    if let Some(params) = request.params.as_ref() {
        let base_dir = mcp_base_dir(cwd.as_deref());
        let param_paths =
            match collect_request_paths_for_security(&request.method, params, &base_dir) {
                Ok(paths) => paths,
                Err(message) => {
                    return Err(mcp_error_response(
                        request.id.clone(),
                        -32602,
                        &message,
                        request.params.clone(),
                    ));
                }
            };

        for (label, path) in param_paths {
            if !path_within_allowed_roots(&path, &server.config.allowed_roots) {
                return Err(mcp_error_response(
                    request.id.clone(),
                    -32001,
                    "security policy denied path parameter",
                    Some(serde_json::json!({
                        "method": request.method.as_str(),
                        "param": label,
                        "path": path.to_string_lossy().to_string(),
                        "allowed_roots": server.config.allowed_roots.clone(),
                    })),
                ));
            }
        }
    }

    if tool_requires_network(&request.method) && !server.config.allow_network {
        return Err(mcp_error_response(
            request.id.clone(),
            -32002,
            "security policy denied network tool",
            Some(serde_json::json!({
                "method": request.method.as_str(),
                "allow_network": server.config.allow_network,
            })),
        ));
    }

    if tool_requires_subprocess(&request.method) && !server.config.allow_subprocess {
        return Err(mcp_error_response(
            request.id.clone(),
            -32003,
            "security policy denied subprocess tool",
            Some(serde_json::json!({
                "method": request.method.as_str(),
                "allow_subprocess": server.config.allow_subprocess,
            })),
        ));
    }

    Ok(cwd)
}

fn mcp_emit_contains(config: &magpie_driver::DriverConfig, emit_kind: &str) -> bool {
    config.emit.iter().any(|emit| emit == emit_kind)
}

fn mcp_has_explicit_emit(params: Option<&serde_json::Value>) -> bool {
    params
        .and_then(|value| value.get("emit"))
        .is_some_and(|emit| !emit.is_null())
}

fn find_runnable_artifact(target_triple: &str, artifacts: &[String]) -> Option<String> {
    let is_windows = target_triple.contains("windows");
    artifacts.iter().find_map(|artifact| {
        let path = Path::new(artifact);
        let is_executable = if is_windows {
            path.extension().and_then(|ext| ext.to_str()) == Some("exe")
        } else {
            path.extension().is_none()
        };
        (is_executable && path.exists()).then(|| artifact.clone())
    })
}

fn parse_mcp_run_args(
    request_id: Option<serde_json::Value>,
    params: Option<&serde_json::Value>,
) -> Result<Vec<String>, McpResponse> {
    let Some(raw_args) = params.and_then(|params| params.get("args")) else {
        return Ok(Vec::new());
    };
    let Some(items) = raw_args.as_array() else {
        return Err(mcp_error_response(
            request_id,
            -32602,
            "invalid params: 'args' must be an array of strings",
            params.cloned(),
        ));
    };
    let mut parsed = Vec::with_capacity(items.len());
    for value in items {
        let Some(arg) = value.as_str() else {
            return Err(mcp_error_response(
                request_id,
                -32602,
                "invalid params: 'args' must be an array of strings",
                params.cloned(),
            ));
        };
        parsed.push(arg.to_string());
    }
    Ok(parsed)
}

fn execute_runnable_artifact(
    path: &str,
    args: &[String],
) -> Result<std::process::ExitStatus, String> {
    Command::new(path)
        .args(args)
        .status()
        .map_err(|err| format!("failed to execute '{}': {}", path, err))
}

fn mcp_base_dir(cwd: Option<&Path>) -> PathBuf {
    cwd.map(Path::to_path_buf)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

fn mcp_repl_sessions_dir(cwd: Option<&Path>) -> PathBuf {
    mcp_base_dir(cwd).join(".magpie").join("mcp").join("repl")
}

fn is_valid_repl_session_id(session_id: &str) -> bool {
    !session_id.is_empty()
        && session_id.len() <= 128
        && session_id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
}

fn generate_repl_session_id() -> String {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!("repl-{}-{nonce:x}", std::process::id())
}

fn repl_session_path(cwd: Option<&Path>, session_id: &str) -> Result<PathBuf, String> {
    if !is_valid_repl_session_id(session_id) {
        return Err(
            "invalid params: 'session_id' must be non-empty and contain only [A-Za-z0-9_-]."
                .to_string(),
        );
    }
    Ok(mcp_repl_sessions_dir(cwd).join(format!("{session_id}.session")))
}

fn save_repl_session(path: &Path, session: &magpie_jit::ReplSession) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create '{}': {}", parent.display(), err))?;
    }
    let serialized = magpie_jit::save_session(session);
    std::fs::write(path, serialized)
        .map_err(|err| format!("failed to write session '{}': {}", path.display(), err))
}

fn load_repl_session(path: &Path) -> Result<magpie_jit::ReplSession, String> {
    let payload = std::fs::read_to_string(path)
        .map_err(|err| format!("failed to read session '{}': {}", path.display(), err))?;
    magpie_jit::load_session(&payload)
        .map_err(|err| format!("failed to parse session '{}': {}", path.display(), err))
}

fn create_repl_session(
    cwd: Option<&Path>,
    requested_session_id: Option<&str>,
) -> Result<(String, PathBuf), String> {
    let (session_id, session_path) = if let Some(requested) = requested_session_id {
        let path = repl_session_path(cwd, requested)?;
        if path.exists() {
            return Err(format!("session '{}' already exists", requested));
        }
        (requested.to_string(), path)
    } else {
        let mut generated: Option<(String, PathBuf)> = None;
        for _ in 0..16 {
            let id = generate_repl_session_id();
            let path = repl_session_path(cwd, &id)?;
            if !path.exists() {
                generated = Some((id, path));
                break;
            }
        }
        generated.ok_or_else(|| "failed to allocate unique repl session id".to_string())?
    };

    let session = magpie_jit::create_repl_session();
    save_repl_session(&session_path, &session)?;
    Ok((session_id, session_path))
}

fn repl_result_to_json(result: &magpie_jit::ReplResult) -> serde_json::Value {
    serde_json::json!({
        "output": result.output,
        "ty": result.ty,
        "diagnostics": result.diagnostics,
        "llvm_ir": result.llvm_ir,
    })
}

fn repl_result_has_errors(result: &magpie_jit::ReplResult) -> bool {
    result
        .diagnostics
        .iter()
        .any(|diag| matches!(diag.severity, magpie_diag::Severity::Error))
}

fn collect_fmt_source_paths_recursive(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut paths = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .collect::<Vec<_>>();
    paths.sort();

    for path in paths {
        if path.is_dir() {
            collect_fmt_source_paths_recursive(&path, out);
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

fn collect_source_paths_for_fmt(entry_path: &str, cwd: Option<&Path>) -> Vec<String> {
    let base_dir = cwd
        .map(Path::to_path_buf)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let entry = resolve_path_from_base(&base_dir, entry_path);

    let mut paths = Vec::new();
    if entry.is_file() {
        paths.push(entry.clone());
    }
    collect_fmt_source_paths_recursive(&base_dir.join("src"), &mut paths);
    collect_fmt_source_paths_recursive(&base_dir.join("tests"), &mut paths);
    paths.sort();
    paths.dedup();
    if paths.is_empty() {
        paths.push(entry);
    }
    paths
        .into_iter()
        .map(|path| path.to_string_lossy().to_string())
        .collect()
}

fn collect_mms_index_paths(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
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

fn find_latest_mms_index(base_dir: &Path) -> Option<PathBuf> {
    let preferred_dir = base_dir.join(".magpie").join("memory");
    find_latest_mms_index_in_dir(&preferred_dir)
        .or_else(|| find_latest_mms_index_in_dir(&base_dir.join("target")))
}

fn find_latest_mms_index_in_dir(dir: &Path) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    collect_mms_index_paths(dir, &mut candidates);
    candidates.sort();
    candidates.pop()
}

fn load_latest_mms_index(base_dir: &Path) -> Result<(PathBuf, magpie_memory::MmsIndex), String> {
    let index_path = find_latest_mms_index(base_dir)
        .ok_or_else(|| "No MMS index found. Run `magpie.memory.build` first.".to_string())?;
    let raw = std::fs::read_to_string(&index_path)
        .map_err(|err| format!("Could not read '{}': {}", index_path.display(), err))?;
    let index = serde_json::from_str::<magpie_memory::MmsIndex>(&raw)
        .map_err(|err| format!("Could not parse '{}': {}", index_path.display(), err))?;
    Ok((index_path, index))
}

fn parse_ctx_budget_policy(policy: Option<&str>) -> magpie_ctx::BudgetPolicy {
    match policy.unwrap_or(magpie_driver::DEFAULT_LLM_BUDGET_POLICY) {
        "diagnostics_first" => magpie_ctx::BudgetPolicy::DiagnosticsFirst,
        "slices_first" => magpie_ctx::BudgetPolicy::SlicesFirst,
        "minimal" => magpie_ctx::BudgetPolicy::Minimal,
        _ => magpie_ctx::BudgetPolicy::Balanced,
    }
}

fn find_manifest_path_from(start_dir: &Path) -> Option<PathBuf> {
    let mut dir = if start_dir.is_dir() {
        start_dir.to_path_buf()
    } else {
        start_dir.parent()?.to_path_buf()
    };
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

fn manifest_path_for_request(params: Option<&serde_json::Value>, cwd: Option<&Path>) -> PathBuf {
    let base_dir = cwd
        .map(Path::to_path_buf)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    if let Some(manifest_path) = params
        .and_then(|params| params.get("manifest_path"))
        .and_then(|value| value.as_str())
    {
        return resolve_path_from_base(&base_dir, manifest_path);
    }
    find_manifest_path_from(&base_dir).unwrap_or_else(|| base_dir.join("Magpie.toml"))
}

fn update_manifest_dependency(manifest_path: &Path, name: &str, add: bool) -> Result<(), String> {
    if name.trim().is_empty() {
        return Err("Dependency name cannot be empty.".to_string());
    }

    let manifest_raw = std::fs::read_to_string(manifest_path)
        .map_err(|err| format!("Could not read '{}': {}", manifest_path.display(), err))?;
    let mut root_value = manifest_raw
        .parse::<toml::Value>()
        .map_err(|err| format!("Could not parse '{}': {}", manifest_path.display(), err))?;

    let Some(root_table) = root_value.as_table_mut() else {
        return Err("Manifest root must be a TOML table.".to_string());
    };
    let deps_value = root_table
        .entry("dependencies".to_string())
        .or_insert_with(|| toml::Value::Table(toml::map::Map::new()));
    let Some(deps_table) = deps_value.as_table_mut() else {
        return Err("'dependencies' must be a TOML table.".to_string());
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

    let encoded = toml::to_string_pretty(&root_value)
        .map_err(|err| format!("failed to serialize manifest TOML: {err}"))?;
    std::fs::write(manifest_path, encoded)
        .map_err(|err| format!("Could not write '{}': {}", manifest_path.display(), err))
}

fn mcp_driver_config(
    params: Option<&serde_json::Value>,
    cwd: Option<&Path>,
) -> magpie_driver::DriverConfig {
    let mut config = magpie_driver::DriverConfig::default();
    let Some(params) = params else {
        if let Some(cwd) = cwd {
            if Path::new(&config.entry_path).is_relative() {
                config.entry_path = cwd.join(&config.entry_path).to_string_lossy().to_string();
            }
        }
        return config;
    };

    if let Some(entry_path) = params.get("entry_path").and_then(|v| v.as_str()) {
        config.entry_path = entry_path.to_string();
    }
    if let Some(target_triple) = params.get("target").and_then(|v| v.as_str()) {
        config.target_triple = target_triple.to_string();
    }
    if let Some(profile) = params.get("profile").and_then(|v| v.as_str()) {
        config.profile = match profile {
            "release" => magpie_driver::BuildProfile::Release,
            _ => magpie_driver::BuildProfile::Dev,
        };
    }
    if let Some(max_errors) = params
        .get("max_errors")
        .and_then(|v| v.as_u64())
        .and_then(|raw| usize::try_from(raw).ok())
    {
        config.max_errors = max_errors;
    }
    if let Some(cache_dir) = params.get("cache_dir").and_then(|value| value.as_str()) {
        config.cache_dir = Some(cache_dir.to_string());
    }
    if let Some(jobs) = params.get("jobs").and_then(|value| value.as_u64()) {
        if let Ok(jobs) = u32::try_from(jobs) {
            config.jobs = Some(jobs);
        }
    }
    if let Some(offline) = params.get("offline").and_then(|value| value.as_bool()) {
        config.offline = offline;
    }
    if let Some(no_default_features) = params
        .get("no_default_features")
        .and_then(|value| value.as_bool())
    {
        config.no_default_features = no_default_features;
    }
    if let Some(shared_generics) = params.get("shared_generics").and_then(|v| v.as_bool()) {
        config.shared_generics = shared_generics;
    }
    if let Some(emit) = params.get("emit") {
        let emit_values = if let Some(single) = emit.as_str() {
            vec![single.to_string()]
        } else if let Some(list) = emit.as_array() {
            list.iter()
                .filter_map(|value| value.as_str().map(ToString::to_string))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        if !emit_values.is_empty() {
            config.emit = emit_values;
        }
    }
    if let Some(features) = params.get("features").and_then(|v| v.as_array()) {
        config.features = features
            .iter()
            .filter_map(|value| value.as_str().map(ToString::to_string))
            .collect::<Vec<_>>();
    }

    if let Some(llm) = params.get("llm") {
        if let Some(llm_mode) = llm.get("mode").and_then(|value| value.as_bool()) {
            config.llm_mode = llm_mode;
        }
        if let Some(token_budget) = llm.get("token_budget").and_then(json_u32) {
            config.token_budget = Some(token_budget);
        }
        if let Some(tokenizer) = llm.get("tokenizer").and_then(|value| value.as_str()) {
            config.llm_tokenizer = Some(tokenizer.to_string());
        }
        if let Some(policy) = llm.get("policy").and_then(|value| value.as_str()) {
            config.llm_budget_policy = Some(policy.to_string());
        }
    }

    if let Some(llm_mode) = params.get("llm_mode").and_then(|v| v.as_bool()) {
        config.llm_mode = llm_mode;
    }
    if let Some(token_budget) = params.get("token_budget").and_then(json_u32) {
        config.token_budget = Some(token_budget);
    }
    if let Some(tokenizer) = params.get("llm_tokenizer").and_then(|value| value.as_str()) {
        config.llm_tokenizer = Some(tokenizer.to_string());
    }
    if let Some(policy) = params
        .get("llm_budget_policy")
        .and_then(|value| value.as_str())
    {
        config.llm_budget_policy = Some(policy.to_string());
    }

    if let Some(cwd) = cwd {
        if Path::new(&config.entry_path).is_relative() {
            config.entry_path = cwd.join(&config.entry_path).to_string_lossy().to_string();
        }
    }

    config
}

pub fn handle_mcp_request(server: &McpServer, request: &McpRequest) -> McpResponse {
    if request.jsonrpc != "2.0" {
        return mcp_error_response(
            request.id.clone(),
            -32600,
            "invalid request: jsonrpc must be '2.0'",
            None,
        );
    }

    if request.method == "mcp.tools.list" {
        let mut tool_list: Vec<serde_json::Value> = server
            .tools
            .values()
            .map(|tool| {
                serde_json::json!({
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                })
            })
            .collect();
        tool_list.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));

        return McpResponse {
            jsonrpc: "2.0".to_string(),
            result: Some(serde_json::json!({
                "tools": tool_list,
            })),
            error: None,
            id: request.id.clone(),
        };
    }

    let Some(_tool) = server.tools.get(&request.method) else {
        return mcp_error_response(
            request.id.clone(),
            -32601,
            "method not found",
            Some(serde_json::json!({
                "method": request.method.as_str(),
            })),
        );
    };

    let cwd = match enforce_mcp_security(server, request) {
        Ok(cwd) => cwd,
        Err(response) => return response,
    };
    let llm = request
        .params
        .as_ref()
        .and_then(|params| params.get("llm"))
        .cloned();
    let params = request.params.as_ref();

    let result = match request.method.as_str() {
        "magpie.build" => {
            let config = mcp_driver_config(params, cwd.as_deref());
            let build = magpie_driver::build(&config);
            Ok(serde_json::json!({
                "tool": "magpie.build",
                "status": if build.success { "ok" } else { "failed" },
                "build": build,
                "config": config,
                "llm": llm,
            }))
        }
        "magpie.run" => {
            let mut config = mcp_driver_config(params, cwd.as_deref());
            if !mcp_emit_contains(&config, "exe") {
                config.emit.push("exe".to_string());
            }
            let run_args = match parse_mcp_run_args(request.id.clone(), params) {
                Ok(args) => args,
                Err(response) => return response,
            };
            let execute = params
                .and_then(|params| params.get("execute"))
                .and_then(|value| value.as_bool())
                .unwrap_or(true);
            let build = magpie_driver::build(&config);
            let runnable = find_runnable_artifact(&config.target_triple, &build.artifacts);
            let (status, run_payload) = if !build.success {
                (
                    "failed".to_string(),
                    serde_json::json!({
                        "executed": false,
                        "runnable_artifact": runnable,
                    }),
                )
            } else if let Some(path) = runnable.as_ref() {
                if execute {
                    match execute_runnable_artifact(path, &run_args) {
                        Ok(exit_status) => {
                            let exit_code = exit_status.code();
                            (
                                if exit_status.success() {
                                    "ok".to_string()
                                } else {
                                    "failed".to_string()
                                },
                                serde_json::json!({
                                    "executed": true,
                                    "runnable_artifact": path,
                                    "exit_code": exit_code,
                                    "args": run_args,
                                }),
                            )
                        }
                        Err(err) => (
                            "execution_error".to_string(),
                            serde_json::json!({
                                "executed": false,
                                "runnable_artifact": path,
                                "error": err,
                                "args": run_args,
                            }),
                        ),
                    }
                } else {
                    (
                        "ok".to_string(),
                        serde_json::json!({
                            "executed": false,
                            "runnable_artifact": path,
                            "reason": "execution disabled by request parameter",
                            "args": run_args,
                        }),
                    )
                }
            } else {
                (
                    "no_runnable_artifact".to_string(),
                    serde_json::json!({
                        "executed": false,
                        "runnable_artifact": serde_json::Value::Null,
                        "args": run_args,
                    }),
                )
            };
            Ok(serde_json::json!({
                "tool": "magpie.run",
                "status": status,
                "build": build,
                "config": config,
                "run": run_payload,
                "llm": llm,
            }))
        }
        "magpie.test" => {
            let mut config = mcp_driver_config(params, cwd.as_deref());
            if !mcp_has_explicit_emit(params) {
                config.emit = vec!["exe".to_string()];
            }
            let filter = match params.and_then(|params| params.get("filter")) {
                Some(value) => match value.as_str() {
                    Some(value) => Some(value.to_string()),
                    None => {
                        return mcp_error_response(
                            request.id.clone(),
                            -32602,
                            "invalid params: 'filter' must be a string",
                            request.params.clone(),
                        );
                    }
                },
                None => None,
            };
            let tests = magpie_driver::run_tests(&config, filter.as_deref());
            Ok(serde_json::json!({
                "tool": "magpie.test",
                "status": if tests.failed == 0 { "ok" } else { "failed" },
                "summary": {
                    "total": tests.total,
                    "passed": tests.passed,
                    "failed": tests.failed,
                },
                "tests": tests,
                "config": config,
                "llm": llm,
            }))
        }
        "magpie.fmt" => {
            let config = mcp_driver_config(params, cwd.as_deref());
            let fix_meta = match params.and_then(|params| params.get("fix_meta")) {
                Some(value) => match value.as_bool() {
                    Some(value) => value,
                    None => {
                        return mcp_error_response(
                            request.id.clone(),
                            -32602,
                            "invalid params: 'fix_meta' must be a boolean",
                            request.params.clone(),
                        );
                    }
                },
                None => false,
            };

            let base_dir = cwd
                .as_deref()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
            let explicit_paths =
                if let Some(paths_value) = params.and_then(|params| params.get("paths")) {
                    let Some(values) = paths_value.as_array() else {
                        return mcp_error_response(
                            request.id.clone(),
                            -32602,
                            "invalid params: 'paths' must be an array of strings",
                            request.params.clone(),
                        );
                    };
                    let mut out = Vec::with_capacity(values.len());
                    for value in values {
                        let Some(path) = value.as_str() else {
                            return mcp_error_response(
                                request.id.clone(),
                                -32602,
                                "invalid params: 'paths' must be an array of strings",
                                request.params.clone(),
                            );
                        };
                        out.push(
                            resolve_path_from_base(&base_dir, path)
                                .to_string_lossy()
                                .to_string(),
                        );
                    }
                    out
                } else if let Some(path) = params
                    .and_then(|params| params.get("path"))
                    .and_then(|value| value.as_str())
                {
                    vec![resolve_path_from_base(&base_dir, path)
                        .to_string_lossy()
                        .to_string()]
                } else {
                    Vec::new()
                };
            let paths = if explicit_paths.is_empty() {
                collect_source_paths_for_fmt(&config.entry_path, cwd.as_deref())
            } else {
                explicit_paths
            };

            let fmt = magpie_driver::format_files(&paths, fix_meta);
            Ok(serde_json::json!({
                "tool": "magpie.fmt",
                "status": if fmt.success { "ok" } else { "failed" },
                "format": fmt,
                "paths": paths,
                "fix_meta": fix_meta,
                "llm": llm,
            }))
        }
        "magpie.lint" => {
            let config = mcp_driver_config(params, cwd.as_deref());
            let lint = magpie_driver::lint(&config);
            Ok(serde_json::json!({
                "tool": "magpie.lint",
                "status": if lint.success { "ok" } else { "failed" },
                "lint": lint,
                "config": config,
                "llm": llm,
            }))
        }
        "magpie.explain" => {
            let code = params
                .and_then(|params| params.get("code"))
                .and_then(|value| value.as_str());

            let Some(code) = code else {
                return mcp_error_response(
                    request.id.clone(),
                    -32602,
                    "invalid params: 'code' is required",
                    request.params.clone(),
                );
            };

            let explanation = magpie_driver::explain_code(code);
            Ok(serde_json::json!({
                "tool": "magpie.explain",
                "status": if explanation.is_some() { "ok" } else { "unknown_code" },
                "code": code,
                "explanation": explanation,
                "llm": llm,
            }))
        }
        "magpie.pkg.resolve" => {
            let manifest_path = manifest_path_for_request(params, cwd.as_deref());
            let offline = params
                .and_then(|params| params.get("offline"))
                .and_then(|value| value.as_bool())
                .unwrap_or(false);

            match magpie_pkg::parse_manifest(&manifest_path) {
                Ok(manifest) => match magpie_pkg::resolve_deps(&manifest, offline) {
                    Ok(lock) => {
                        let lock_path = manifest_path
                            .parent()
                            .unwrap_or_else(|| Path::new("."))
                            .join("Magpie.lock");
                        match magpie_pkg::write_lockfile(&lock, &lock_path) {
                            Ok(()) => Ok(serde_json::json!({
                                "tool": "magpie.pkg.resolve",
                                "status": "ok",
                                "manifest_path": manifest_path.to_string_lossy().to_string(),
                                "lockfile_path": lock_path.to_string_lossy().to_string(),
                                "package_count": lock.packages.len(),
                                "offline": offline,
                                "llm": llm,
                            })),
                            Err(error) => Ok(serde_json::json!({
                                "tool": "magpie.pkg.resolve",
                                "status": "failed",
                                "error": error,
                                "manifest_path": manifest_path.to_string_lossy().to_string(),
                                "llm": llm,
                            })),
                        }
                    }
                    Err(error) => Ok(serde_json::json!({
                        "tool": "magpie.pkg.resolve",
                        "status": "failed",
                        "error": error,
                        "manifest_path": manifest_path.to_string_lossy().to_string(),
                        "llm": llm,
                    })),
                },
                Err(error) => Ok(serde_json::json!({
                    "tool": "magpie.pkg.resolve",
                    "status": "failed",
                    "error": error,
                    "manifest_path": manifest_path.to_string_lossy().to_string(),
                    "llm": llm,
                })),
            }
        }
        "magpie.pkg.plan" => {
            let manifest_path = manifest_path_for_request(params, cwd.as_deref());
            let offline = params
                .and_then(|params| params.get("offline"))
                .and_then(|value| value.as_bool())
                .unwrap_or(false);

            match magpie_pkg::parse_manifest(&manifest_path) {
                Ok(manifest) => match magpie_pkg::resolve_deps(&manifest, offline) {
                    Ok(lock) => {
                        let planned_packages = lock
                            .packages
                            .iter()
                            .map(|package| {
                                serde_json::json!({
                                    "name": package.name.clone(),
                                    "version": package.version.clone(),
                                    "source_kind": package.source.kind.clone(),
                                })
                            })
                            .collect::<Vec<_>>();
                        Ok(serde_json::json!({
                            "tool": "magpie.pkg.plan",
                            "status": "ok",
                            "dry_run": true,
                            "manifest_path": manifest_path.to_string_lossy().to_string(),
                            "offline": offline,
                            "summary": {
                                "package_count": planned_packages.len(),
                                "packages": planned_packages,
                            },
                            "llm": llm,
                        }))
                    }
                    Err(error) => Ok(serde_json::json!({
                        "tool": "magpie.pkg.plan",
                        "status": "failed",
                        "dry_run": true,
                        "error": error,
                        "manifest_path": manifest_path.to_string_lossy().to_string(),
                        "offline": offline,
                        "llm": llm,
                    })),
                },
                Err(error) => Ok(serde_json::json!({
                    "tool": "magpie.pkg.plan",
                    "status": "failed",
                    "dry_run": true,
                    "error": error,
                    "manifest_path": manifest_path.to_string_lossy().to_string(),
                    "offline": offline,
                    "llm": llm,
                })),
            }
        }
        "magpie.pkg.add" => {
            let name = match params
                .and_then(|params| params.get("name"))
                .and_then(|value| value.as_str())
            {
                Some(name) => name,
                None => {
                    return mcp_error_response(
                        request.id.clone(),
                        -32602,
                        "invalid params: 'name' is required",
                        request.params.clone(),
                    );
                }
            };

            let manifest_path = manifest_path_for_request(params, cwd.as_deref());
            match update_manifest_dependency(&manifest_path, name, true) {
                Ok(()) => Ok(serde_json::json!({
                    "tool": "magpie.pkg.add",
                    "status": "ok",
                    "manifest_path": manifest_path.to_string_lossy().to_string(),
                    "name": name,
                    "llm": llm,
                })),
                Err(error) => Ok(serde_json::json!({
                    "tool": "magpie.pkg.add",
                    "status": "failed",
                    "manifest_path": manifest_path.to_string_lossy().to_string(),
                    "name": name,
                    "error": error,
                    "llm": llm,
                })),
            }
        }
        "magpie.pkg.remove" => {
            let name = match params
                .and_then(|params| params.get("name"))
                .and_then(|value| value.as_str())
            {
                Some(name) => name,
                None => {
                    return mcp_error_response(
                        request.id.clone(),
                        -32602,
                        "invalid params: 'name' is required",
                        request.params.clone(),
                    );
                }
            };

            let manifest_path = manifest_path_for_request(params, cwd.as_deref());
            match update_manifest_dependency(&manifest_path, name, false) {
                Ok(()) => Ok(serde_json::json!({
                    "tool": "magpie.pkg.remove",
                    "status": "ok",
                    "manifest_path": manifest_path.to_string_lossy().to_string(),
                    "name": name,
                    "llm": llm,
                })),
                Err(error) => Ok(serde_json::json!({
                    "tool": "magpie.pkg.remove",
                    "status": "failed",
                    "manifest_path": manifest_path.to_string_lossy().to_string(),
                    "name": name,
                    "error": error,
                    "llm": llm,
                })),
            }
        }
        "magpie.pkg.why" => {
            let name = match params
                .and_then(|params| params.get("name"))
                .and_then(|value| value.as_str())
            {
                Some(name) => name,
                None => {
                    return mcp_error_response(
                        request.id.clone(),
                        -32602,
                        "invalid params: 'name' is required",
                        request.params.clone(),
                    );
                }
            };

            let manifest_path = manifest_path_for_request(params, cwd.as_deref());
            let lock_path = manifest_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("Magpie.lock");
            match magpie_pkg::read_lockfile(&lock_path) {
                Ok(lock) => {
                    let mut reasons = Vec::new();
                    for package in lock.packages {
                        for dependency in package.deps {
                            if dependency.name == name {
                                reasons.push(format!("{} -> {}", package.name, dependency.name));
                            }
                        }
                    }
                    reasons.sort();
                    reasons.dedup();
                    Ok(serde_json::json!({
                        "tool": "magpie.pkg.why",
                        "status": "ok",
                        "name": name,
                        "lockfile_path": lock_path.to_string_lossy().to_string(),
                        "reasons": reasons,
                        "llm": llm,
                    }))
                }
                Err(error) => Ok(serde_json::json!({
                    "tool": "magpie.pkg.why",
                    "status": "failed",
                    "name": name,
                    "lockfile_path": lock_path.to_string_lossy().to_string(),
                    "error": error,
                    "llm": llm,
                })),
            }
        }
        "magpie.memory.build" => {
            let config = mcp_driver_config(params, cwd.as_deref());
            let build = magpie_driver::build(&config);
            Ok(serde_json::json!({
                "tool": "magpie.memory.build",
                "status": if build.success { "ok" } else { "failed" },
                "build": build,
                "config": config,
                "llm": llm,
            }))
        }
        "magpie.memory.query" => {
            let query = params
                .and_then(|params| params.get("query").or_else(|| params.get("q")))
                .and_then(|value| value.as_str());
            let Some(query) = query else {
                return mcp_error_response(
                    request.id.clone(),
                    -32602,
                    "invalid params: 'query' (or 'q') is required",
                    request.params.clone(),
                );
            };

            let k = params
                .and_then(|params| params.get("k"))
                .and_then(|value| value.as_u64())
                .and_then(|raw| usize::try_from(raw).ok())
                .unwrap_or(10)
                .max(1);
            let base_dir = cwd
                .as_deref()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

            match load_latest_mms_index(&base_dir) {
                Ok((index_path, index)) => {
                    let stale_issues =
                        magpie_memory::validate_index_staleness(&index, &base_dir, 8);
                    if !stale_issues.is_empty() {
                        let sample = stale_issues
                            .first()
                            .map(|issue| format!("{} ({})", issue.path, issue.reason))
                            .unwrap_or_else(|| "unknown artifact".to_string());
                        return McpResponse {
                            jsonrpc: "2.0".to_string(),
                            result: Some(serde_json::json!({
                                "tool": "magpie.memory.query",
                                "status": "failed",
                                "query": query,
                                "k": k,
                                "index_path": index_path.to_string_lossy().to_string(),
                                "error": format!(
                                    "MMS index is stale ({} issue(s)); first: {}. Run `magpie memory build` to refresh.",
                                    stale_issues.len(),
                                    sample
                                ),
                                "stale_issues": stale_issues,
                                "llm": llm,
                            })),
                            error: None,
                            id: request.id.clone(),
                        };
                    }

                    let hits = magpie_memory::query_bm25(&index, query, k);
                    Ok(serde_json::json!({
                        "tool": "magpie.memory.query",
                        "status": "ok",
                        "query": query,
                        "k": k,
                        "index_path": index_path.to_string_lossy().to_string(),
                        "hits": hits,
                        "llm": llm,
                    }))
                }
                Err(error) => Ok(serde_json::json!({
                    "tool": "magpie.memory.query",
                    "status": "failed",
                    "query": query,
                    "k": k,
                    "error": error,
                    "llm": llm,
                })),
            }
        }
        "magpie.ctx.pack" => {
            let config = mcp_driver_config(params, cwd.as_deref());
            let base_dir = cwd
                .as_deref()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

            match load_latest_mms_index(&base_dir) {
                Ok((index_path, index)) => {
                    let stale_issues =
                        magpie_memory::validate_index_staleness(&index, &base_dir, 8);
                    if !stale_issues.is_empty() {
                        let sample = stale_issues
                            .first()
                            .map(|issue| format!("{} ({})", issue.path, issue.reason))
                            .unwrap_or_else(|| "unknown artifact".to_string());
                        Ok(serde_json::json!({
                            "tool": "magpie.ctx.pack",
                            "status": "failed",
                            "index_path": index_path.to_string_lossy().to_string(),
                            "error": format!(
                                "MMS index is stale ({} issue(s)); first: {}. Run `magpie memory build` to refresh.",
                                stale_issues.len(),
                                sample
                            ),
                            "stale_issues": stale_issues,
                            "config": config,
                            "llm": llm,
                        }))
                    } else {
                        let tokenizer = config
                            .llm_tokenizer
                            .as_deref()
                            .unwrap_or(magpie_driver::DEFAULT_LLM_TOKENIZER);
                        let chunks = index
                            .items
                            .iter()
                            .map(|item| magpie_ctx::Chunk {
                                chunk_id: item.item_id.clone(),
                                kind: item.kind.clone(),
                                subject_id: item.sid.clone(),
                                body: item.text.clone(),
                                token_cost: item
                                    .token_cost
                                    .get(tokenizer)
                                    .copied()
                                    .or_else(|| {
                                        item.token_cost
                                            .get(magpie_driver::DEFAULT_LLM_TOKENIZER)
                                            .copied()
                                    })
                                    .unwrap_or(0),
                                score: item.priority as f64,
                            })
                            .collect::<Vec<_>>();
                        let budget = config
                            .token_budget
                            .unwrap_or(magpie_driver::DEFAULT_LLM_TOKEN_BUDGET);
                        let policy = parse_ctx_budget_policy(config.llm_budget_policy.as_deref());
                        let pack = magpie_ctx::build_context_pack(chunks, budget, policy);
                        let out_path = base_dir.join(".magpie").join("ctx").join("pack.json");

                        let write_result = (|| -> Result<(), String> {
                            if let Some(parent) = out_path.parent() {
                                std::fs::create_dir_all(parent).map_err(|err| {
                                    format!("Could not create '{}': {}", parent.display(), err)
                                })?;
                            }
                            let payload = serde_json::to_string(&pack).map_err(|err| {
                                format!("failed to serialize context pack: {err}")
                            })?;
                            std::fs::write(&out_path, payload).map_err(|err| {
                                format!("Could not write '{}': {}", out_path.display(), err)
                            })?;
                            Ok(())
                        })();

                        match write_result {
                            Ok(()) => Ok(serde_json::json!({
                                "tool": "magpie.ctx.pack",
                                "status": "ok",
                                "index_path": index_path.to_string_lossy().to_string(),
                                "pack_path": out_path.to_string_lossy().to_string(),
                                "pack": pack,
                                "config": config,
                                "llm": llm,
                            })),
                            Err(error) => Ok(serde_json::json!({
                                "tool": "magpie.ctx.pack",
                                "status": "failed",
                                "index_path": index_path.to_string_lossy().to_string(),
                                "pack_path": out_path.to_string_lossy().to_string(),
                                "error": error,
                                "config": config,
                                "llm": llm,
                            })),
                        }
                    }
                }
                Err(error) => Ok(serde_json::json!({
                    "tool": "magpie.ctx.pack",
                    "status": "failed",
                    "error": error,
                    "config": config,
                    "llm": llm,
                })),
            }
        }
        "magpie.graph.symbols"
        | "magpie.graph.deps"
        | "magpie.graph.ownership"
        | "magpie.graph.cfg" => {
            let (emit_kind, suffix) = match request.method.as_str() {
                "magpie.graph.symbols" => ("symgraph", ".symgraph.json"),
                "magpie.graph.deps" => ("depsgraph", ".depsgraph.json"),
                "magpie.graph.ownership" => ("ownershipgraph", ".ownershipgraph.json"),
                "magpie.graph.cfg" => ("cfggraph", ".cfggraph.json"),
                _ => unreachable!(),
            };
            let mut config = mcp_driver_config(params, cwd.as_deref());
            config.emit = vec![emit_kind.to_string()];
            let build = magpie_driver::build(&config);
            let graph_artifacts = build
                .artifacts
                .iter()
                .filter(|artifact| artifact.ends_with(suffix))
                .cloned()
                .collect::<Vec<_>>();
            let graph = graph_artifacts.first().and_then(|path| {
                std::fs::read_to_string(path)
                    .ok()
                    .and_then(|payload| serde_json::from_str::<serde_json::Value>(&payload).ok())
            });

            Ok(serde_json::json!({
                "tool": request.method.as_str(),
                "status": if build.success { "ok" } else { "failed" },
                "build": build,
                "config": config,
                "graph_artifacts": graph_artifacts,
                "graph": graph,
                "llm": llm,
            }))
        }
        "magpie.repl.create" => {
            let requested_session_id = match params.and_then(|params| params.get("session_id")) {
                Some(value) => match value.as_str() {
                    Some(session_id) => Some(session_id),
                    None => {
                        return mcp_error_response(
                            request.id.clone(),
                            -32602,
                            "invalid params: 'session_id' must be a string",
                            request.params.clone(),
                        );
                    }
                },
                None => None,
            };

            match create_repl_session(cwd.as_deref(), requested_session_id) {
                Ok((session_id, session_path)) => Ok(serde_json::json!({
                    "tool": "magpie.repl.create",
                    "status": "ok",
                    "session_id": session_id,
                    "session_path": session_path.to_string_lossy().to_string(),
                    "llm": llm,
                })),
                Err(error) => Ok(serde_json::json!({
                    "tool": "magpie.repl.create",
                    "status": "failed",
                    "error": error,
                    "llm": llm,
                })),
            }
        }
        "magpie.repl.eval" => {
            let session_id = match params
                .and_then(|params| params.get("session_id"))
                .and_then(|value| value.as_str())
            {
                Some(session_id) => session_id,
                None => {
                    return mcp_error_response(
                        request.id.clone(),
                        -32602,
                        "invalid params: 'session_id' is required",
                        request.params.clone(),
                    );
                }
            };
            let code = match params
                .and_then(|params| params.get("code"))
                .and_then(|value| value.as_str())
            {
                Some(code) => code,
                None => {
                    return mcp_error_response(
                        request.id.clone(),
                        -32602,
                        "invalid params: 'code' is required",
                        request.params.clone(),
                    );
                }
            };
            let session_path = match repl_session_path(cwd.as_deref(), session_id) {
                Ok(path) => path,
                Err(message) => {
                    return mcp_error_response(
                        request.id.clone(),
                        -32602,
                        &message,
                        request.params.clone(),
                    );
                }
            };

            match load_repl_session(&session_path) {
                Ok(mut session) => {
                    let mut diag = magpie_diag::DiagnosticBag::new(200);
                    let eval = magpie_jit::eval_cell(&mut session, code, &mut diag);
                    match save_repl_session(&session_path, &session) {
                        Ok(()) => Ok(serde_json::json!({
                            "tool": "magpie.repl.eval",
                            "status": if repl_result_has_errors(&eval) { "failed" } else { "ok" },
                            "session_id": session_id,
                            "result": repl_result_to_json(&eval),
                            "llm": llm,
                        })),
                        Err(error) => Ok(serde_json::json!({
                            "tool": "magpie.repl.eval",
                            "status": "failed",
                            "session_id": session_id,
                            "result": repl_result_to_json(&eval),
                            "error": error,
                            "llm": llm,
                        })),
                    }
                }
                Err(error) => Ok(serde_json::json!({
                    "tool": "magpie.repl.eval",
                    "status": "failed",
                    "session_id": session_id,
                    "error": error,
                    "llm": llm,
                })),
            }
        }
        "magpie.repl.inspect" => {
            let session_id = match params
                .and_then(|params| params.get("session_id"))
                .and_then(|value| value.as_str())
            {
                Some(session_id) => session_id,
                None => {
                    return mcp_error_response(
                        request.id.clone(),
                        -32602,
                        "invalid params: 'session_id' is required",
                        request.params.clone(),
                    );
                }
            };
            let kind = match params.and_then(|params| params.get("kind")) {
                Some(value) => match value.as_str() {
                    Some(value) => value,
                    None => {
                        return mcp_error_response(
                            request.id.clone(),
                            -32602,
                            "invalid params: 'kind' must be a string",
                            request.params.clone(),
                        );
                    }
                },
                None => "session",
            };
            let query = match params
                .and_then(|params| params.get("query").or_else(|| params.get("expr")))
            {
                Some(value) => match value.as_str() {
                    Some(value) => Some(value),
                    None => {
                        return mcp_error_response(
                            request.id.clone(),
                            -32602,
                            "invalid params: 'query'/'expr' must be a string",
                            request.params.clone(),
                        );
                    }
                },
                None => None,
            };
            let session_path = match repl_session_path(cwd.as_deref(), session_id) {
                Ok(path) => path,
                Err(message) => {
                    return mcp_error_response(
                        request.id.clone(),
                        -32602,
                        &message,
                        request.params.clone(),
                    );
                }
            };

            match load_repl_session(&session_path) {
                Ok(session) => {
                    let inspect = match kind {
                        "session" => serde_json::json!({
                            "cell_counter": session.cell_counter,
                            "symbol_count": session.symbol_table.len(),
                            "diagnostics_count": session.diagnostics_history.len(),
                            "compiled_module_count": session.compiled_modules.len(),
                            "last_compiled_module": session.compiled_modules.last().map(|module| module.module_name.clone()),
                        }),
                        "type" => serde_json::json!({
                            "query": query.unwrap_or_default(),
                            "value": magpie_jit::inspect_type(&session, query.unwrap_or_default()),
                        }),
                        "ir" => serde_json::json!({
                            "query": query.unwrap_or_default(),
                            "value": magpie_jit::inspect_ir(&session, query.unwrap_or_default()),
                        }),
                        "llvm" => serde_json::json!({
                            "query": query.unwrap_or_default(),
                            "value": magpie_jit::inspect_llvm_ir(&session, query.unwrap_or_default()),
                        }),
                        "diag_last" => serde_json::json!({
                            "value": session.diagnostics_history.last().cloned(),
                        }),
                        _ => {
                            return mcp_error_response(
                                request.id.clone(),
                                -32602,
                                "invalid params: 'kind' must be one of session|type|ir|llvm|diag_last",
                                request.params.clone(),
                            );
                        }
                    };

                    Ok(serde_json::json!({
                        "tool": "magpie.repl.inspect",
                        "status": "ok",
                        "session_id": session_id,
                        "kind": kind,
                        "inspect": inspect,
                        "llm": llm,
                    }))
                }
                Err(error) => Ok(serde_json::json!({
                    "tool": "magpie.repl.inspect",
                    "status": "failed",
                    "session_id": session_id,
                    "kind": kind,
                    "error": error,
                    "llm": llm,
                })),
            }
        }
        other => Err(mcp_error_response(
            request.id.clone(),
            -32004,
            "internal dispatch mismatch",
            Some(serde_json::json!({
                "tool": other,
                "message": "Tool was registered but no handler branch matched.",
            })),
        )),
    };

    let result = match result {
        Ok(result) => result,
        Err(error) => return error,
    };

    McpResponse {
        jsonrpc: "2.0".to_string(),
        result: Some(result),
        error: None,
        id: request.id.clone(),
    }
}

pub fn run_mcp_stdio(server: &McpServer) {
    use std::io::{self, BufRead, Write};

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let Ok(line) = line else {
            break;
        };
        let payload = line.trim();
        if payload.is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<McpRequest>(payload) {
            Ok(request) => handle_mcp_request(server, &request),
            Err(error) => mcp_error_response(
                None,
                -32700,
                "parse error",
                Some(serde_json::json!({ "detail": error.to_string() })),
            ),
        };

        if serde_json::to_writer(&mut stdout, &response).is_err() {
            break;
        }
        if stdout.write_all(b"\n").is_err() {
            break;
        }
        if stdout.flush().is_err() {
            break;
        }
    }
}

#[cfg(test)]
mod mcp_tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_server_with_subprocess() -> McpServer {
        let mut server = create_mcp_server();
        server.config.allow_subprocess = true;
        server
    }

    fn test_server_with_subprocess_for_entry(entry: &Path) -> McpServer {
        let mut server = test_server_with_subprocess();
        if let Some(root) = entry.parent() {
            server.config.allowed_roots = vec![root.to_string_lossy().to_string()];
        }
        server
    }

    fn temp_entry_path() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("magpie_web_mcp_{}_{}", std::process::id(), nonce));
        std::fs::create_dir_all(&dir).expect("temp dir should exist");
        let entry = dir.join("main.mp");
        std::fs::write(
            &entry,
            r#"module test.main
exports { @main }
imports { }
digest "0000000000000000"

fn @main() -> i32 {
bb0:
  ret const.i32 0
}
"#,
        )
        .expect("entry fixture should be written");
        entry
    }

    fn temp_stale_mms_root() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "magpie_web_mms_stale_{}_{}",
            std::process::id(),
            nonce
        ));
        std::fs::create_dir_all(root.join(".magpie/memory")).expect("memory dir should exist");
        std::fs::create_dir_all(root.join("src")).expect("src dir should exist");
        let source = root.join("src").join("main.mp");
        std::fs::write(&source, "module test.main\ndigest \"cafebabe\"\n")
            .expect("source fixture should be written");

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
        let source_fingerprints = vec![magpie_memory::MmsSourceFingerprint {
            path: source.to_string_lossy().to_string(),
            digest: "0000000000000000".to_string(),
        }];
        let index = magpie_memory::build_index_with_sources(&[item], &source_fingerprints);
        let encoded =
            serde_json::to_string_pretty(&index).expect("mms index fixture should serialize");
        std::fs::write(root.join(".magpie/memory/stale.mms_index.json"), encoded)
            .expect("mms index fixture should be written");
        root
    }

    #[test]
    fn mcp_build_calls_driver_build() {
        let entry = temp_entry_path();
        let server = test_server_with_subprocess_for_entry(&entry);
        let request = McpRequest {
            jsonrpc: "2.0".to_string(),
            method: "magpie.build".to_string(),
            params: Some(serde_json::json!({
                "entry_path": entry.to_string_lossy().to_string(),
                "emit": ["mpir"],
            })),
            id: Some(serde_json::json!(1)),
        };
        let response = handle_mcp_request(&server, &request);
        assert!(
            response.error.is_none(),
            "response error: {:?}",
            response.error
        );
        let result = response.result.expect("result should be present");
        assert_eq!(result["tool"], "magpie.build");
        assert_eq!(result["status"], "ok");
        assert_eq!(result["build"]["success"], true);
    }

    #[test]
    fn mcp_explain_calls_driver_explain() {
        let server = test_server_with_subprocess();
        let request = McpRequest {
            jsonrpc: "2.0".to_string(),
            method: "magpie.explain".to_string(),
            params: Some(serde_json::json!({
                "code": "MPO0003",
            })),
            id: Some(serde_json::json!(2)),
        };
        let response = handle_mcp_request(&server, &request);
        assert!(
            response.error.is_none(),
            "response error: {:?}",
            response.error
        );
        let result = response.result.expect("result should be present");
        assert_eq!(result["tool"], "magpie.explain");
        assert_eq!(result["status"], "ok");
        assert!(result["explanation"].is_string());
    }

    #[test]
    fn mcp_test_defaults_emit_exe_when_not_explicitly_provided() {
        let entry = temp_entry_path();
        let server = test_server_with_subprocess_for_entry(&entry);
        let request = McpRequest {
            jsonrpc: "2.0".to_string(),
            method: "magpie.test".to_string(),
            params: Some(serde_json::json!({
                "entry_path": entry.to_string_lossy().to_string(),
            })),
            id: Some(serde_json::json!(3)),
        };
        let response = handle_mcp_request(&server, &request);
        assert!(
            response.error.is_none(),
            "response error: {:?}",
            response.error
        );
        let result = response.result.expect("result should be present");
        assert_eq!(result["tool"], "magpie.test");
        assert_eq!(result["config"]["emit"], serde_json::json!(["exe"]));
    }

    #[test]
    fn mcp_run_supports_execute_false_intent_mode() {
        let entry = temp_entry_path();
        let server = test_server_with_subprocess_for_entry(&entry);
        let request = McpRequest {
            jsonrpc: "2.0".to_string(),
            method: "magpie.run".to_string(),
            params: Some(serde_json::json!({
                "entry_path": entry.to_string_lossy().to_string(),
                "execute": false,
                "args": ["--example"],
            })),
            id: Some(serde_json::json!(4)),
        };
        let response = handle_mcp_request(&server, &request);
        assert!(
            response.error.is_none(),
            "response error: {:?}",
            response.error
        );
        let result = response.result.expect("result should be present");
        assert_eq!(result["tool"], "magpie.run");
        assert_eq!(result["run"]["executed"], false);
    }

    #[test]
    fn mcp_repl_tools_work_without_subprocess_permission() {
        let mut server = create_mcp_server();
        server.config.allow_subprocess = false;

        let entry = temp_entry_path();
        let cwd = entry
            .parent()
            .expect("temp entry should have parent")
            .to_path_buf();
        server.config.allowed_roots = vec![cwd.to_string_lossy().to_string()];

        let create_request = McpRequest {
            jsonrpc: "2.0".to_string(),
            method: "magpie.repl.create".to_string(),
            params: Some(serde_json::json!({
                "cwd": cwd.to_string_lossy().to_string(),
            })),
            id: Some(serde_json::json!(5)),
        };
        let create_response = handle_mcp_request(&server, &create_request);
        assert!(
            create_response.error.is_none(),
            "create response error: {:?}",
            create_response.error
        );
        let create_result = create_response.result.expect("create result");
        assert_eq!(create_result["status"], "ok");
        let session_id = create_result["session_id"]
            .as_str()
            .expect("session id string")
            .to_string();

        let eval_request = McpRequest {
            jsonrpc: "2.0".to_string(),
            method: "magpie.repl.eval".to_string(),
            params: Some(serde_json::json!({
                "cwd": cwd.to_string_lossy().to_string(),
                "session_id": session_id,
                "code": "ret const.i32 7",
            })),
            id: Some(serde_json::json!(6)),
        };
        let eval_response = handle_mcp_request(&server, &eval_request);
        assert!(
            eval_response.error.is_none(),
            "eval response error: {:?}",
            eval_response.error
        );
        let eval_result = eval_response.result.expect("eval result");
        assert_eq!(eval_result["status"], "ok", "eval result: {eval_result:#?}");

        let inspect_request = McpRequest {
            jsonrpc: "2.0".to_string(),
            method: "magpie.repl.inspect".to_string(),
            params: Some(serde_json::json!({
                "cwd": cwd.to_string_lossy().to_string(),
                "session_id": session_id,
                "kind": "session",
            })),
            id: Some(serde_json::json!(7)),
        };
        let inspect_response = handle_mcp_request(&server, &inspect_request);
        assert!(
            inspect_response.error.is_none(),
            "inspect response error: {:?}",
            inspect_response.error
        );
        let inspect_result = inspect_response.result.expect("inspect result");
        assert_eq!(inspect_result["status"], "ok");
        assert_eq!(
            inspect_result["inspect"]["cell_counter"],
            serde_json::json!(1)
        );
    }

    #[test]
    fn mcp_memory_query_fails_on_stale_index() {
        let mut server = create_mcp_server();
        let root = temp_stale_mms_root();
        server.config.allowed_roots = vec![root.to_string_lossy().to_string()];

        let request = McpRequest {
            jsonrpc: "2.0".to_string(),
            method: "magpie.memory.query".to_string(),
            params: Some(serde_json::json!({
                "cwd": root.to_string_lossy().to_string(),
                "query": "main",
                "k": 5,
            })),
            id: Some(serde_json::json!(8)),
        };
        let response = handle_mcp_request(&server, &request);
        assert!(
            response.error.is_none(),
            "memory.query response error: {:?}",
            response.error
        );
        let result = response
            .result
            .expect("memory.query result should be present");
        assert_eq!(result["tool"], "magpie.memory.query");
        assert_eq!(result["status"], "failed");
        let err = result["error"].as_str().unwrap_or_default().to_string();
        assert!(
            err.contains("MMS index is stale"),
            "expected stale-index message, got: {err}"
        );
        assert!(
            result["stale_issues"]
                .as_array()
                .is_some_and(|issues| !issues.is_empty()),
            "expected non-empty stale issues payload: {result:#?}"
        );

        std::fs::remove_dir_all(root).expect("stale mms root should be removable");
    }

    #[test]
    fn mcp_build_denies_entry_path_outside_allowed_roots() {
        let mut server = create_mcp_server();
        server.config.allow_subprocess = true;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        let allowed_root = std::env::temp_dir().join(format!(
            "magpie_web_allowed_root_{}_{}",
            std::process::id(),
            nonce
        ));
        std::fs::create_dir_all(&allowed_root).expect("allowed root should be created");
        server.config.allowed_roots = vec![allowed_root.to_string_lossy().to_string()];

        let entry = temp_entry_path();
        let request = McpRequest {
            jsonrpc: "2.0".to_string(),
            method: "magpie.build".to_string(),
            params: Some(serde_json::json!({
                "entry_path": entry.to_string_lossy().to_string(),
            })),
            id: Some(serde_json::json!(9)),
        };
        let response = handle_mcp_request(&server, &request);
        let error = response.error.expect("security error expected");
        assert_eq!(error.code, -32001);
        assert!(
            error
                .message
                .contains("security policy denied path parameter"),
            "unexpected error message: {}",
            error.message
        );
        let data = error.data.expect("security error data should be present");
        assert_eq!(data["param"], "entry_path");

        let entry_root = entry
            .parent()
            .expect("temp entry should have parent")
            .to_path_buf();
        std::fs::remove_dir_all(allowed_root).expect("allowed root should be removable");
        std::fs::remove_dir_all(entry_root).expect("temp entry root should be removable");
    }
}

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TServeOpts {
    pub keep_alive: bool,
    pub threads: u32,
    pub max_body_bytes: u64,
    pub read_timeout_ms: u64,
    pub write_timeout_ms: u64,
    pub log_requests: bool,
}

impl Default for TServeOpts {
    fn default() -> Self {
        let threads = std::thread::available_parallelism()
            .map(|value| value.get() as u32)
            .unwrap_or(1);
        Self {
            keep_alive: true,
            threads: threads.max(1),
            max_body_bytes: 10_000_000,
            read_timeout_ms: 30_000,
            write_timeout_ms: 30_000,
            log_requests: false,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ServeConfig {
    pub service: TService,
    pub addr: String,
    pub port: u16,
    pub opts: TServeOpts,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct WebRouteMapping {
    pub file: String,
    pub pattern: String,
}

fn parse_route_file_segment(segment: &str) -> Result<String, String> {
    if segment.is_empty() {
        return Err("route segment cannot be empty".to_string());
    }
    if segment.starts_with('[') && segment.ends_with(']') {
        let inner = &segment[1..segment.len() - 1];
        let (name, ty) = inner
            .split_once(':')
            .ok_or_else(|| format!("invalid dynamic route segment '{segment}'"))?;
        if !is_valid_ident(name) {
            return Err(format!("invalid route parameter name '{name}'"));
        }
        if RouteParamType::parse(ty).is_none() {
            return Err(format!("unsupported route parameter type '{ty}'"));
        }
        return Ok(format!("{{{name}:{ty}}}"));
    }
    if segment.contains('[') || segment.contains(']') {
        return Err(format!("invalid route segment '{segment}'"));
    }
    Ok(segment.to_string())
}

fn scan_webapp_routes_recursive(
    routes_root: &Path,
    current_dir: &Path,
    out: &mut Vec<WebRouteMapping>,
) -> Result<(), String> {
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(current_dir)
        .map_err(|err| format!("failed to read '{}': {err}", current_dir.display()))?
    {
        entries.push(entry.map_err(|err| {
            format!("failed to read entry in '{}': {err}", current_dir.display())
        })?);
    }
    entries.sort_by_key(|a| a.file_name());

    for entry in entries {
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|err| format!("failed to read metadata for '{}': {err}", path.display()))?;

        if file_type.is_dir() {
            scan_webapp_routes_recursive(routes_root, &path, out)?;
            continue;
        }
        if !file_type.is_file() || path.extension() != Some(OsStr::new("mp")) {
            continue;
        }

        let rel_path = path
            .strip_prefix(routes_root)
            .map_err(|_| format!("failed to relativize '{}'", path.display()))?;
        let file_stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("invalid utf-8 route file name '{}'", path.display()))?;
        if file_stem == "_layout" {
            continue;
        }

        let mut segments = Vec::new();
        if let Some(parent) = rel_path.parent() {
            for component in parent.components() {
                if let std::path::Component::Normal(seg) = component {
                    segments.push(parse_route_file_segment(&seg.to_string_lossy())?);
                }
            }
        }
        if file_stem != "index" {
            segments.push(parse_route_file_segment(file_stem)?);
        }

        let pattern = if segments.is_empty() {
            "/".to_string()
        } else {
            format!("/{}", segments.join("/"))
        };
        let rel_file = rel_path.to_string_lossy().replace('\\', "/");
        out.push(WebRouteMapping {
            file: format!("routes/{rel_file}"),
            pattern,
        });
    }

    Ok(())
}

fn scan_webapp_routes(app_dir: &Path) -> Result<Vec<WebRouteMapping>, String> {
    let routes_dir = app_dir.join("routes");
    if !routes_dir.is_dir() {
        return Err(format!(
            "routes directory not found at '{}'",
            routes_dir.display()
        ));
    }

    let mut mappings = Vec::new();
    scan_webapp_routes_recursive(&routes_dir, &routes_dir, &mut mappings)?;
    mappings.sort_by(|a, b| a.pattern.cmp(&b.pattern).then(a.file.cmp(&b.file)));
    Ok(mappings)
}

fn render_webapp_routes_source(mappings: &[WebRouteMapping]) -> String {
    let mut out = String::new();
    out.push_str(";; @generated by magpie_web::generate_webapp_routes\n");
    out.push_str(";; File-based routes per SPEC 30.2.2\n");
    for mapping in mappings {
        out.push_str(&format!("GET {} <- {}\n", mapping.pattern, mapping.file));
    }
    out
}

pub fn generate_webapp_routes(app_dir: &Path) -> Result<String, String> {
    let mappings = scan_webapp_routes(app_dir)?;
    Ok(render_webapp_routes_source(&mappings))
}

fn route_param_type_name(ty: &RouteParamType) -> &'static str {
    match ty {
        RouteParamType::I32 => "i32",
        RouteParamType::I64 => "i64",
        RouteParamType::U32 => "u32",
        RouteParamType::U64 => "u64",
        RouteParamType::Bool => "bool",
        RouteParamType::Str => "Str",
    }
}

fn openapi_schema_for_param_type(ty: &RouteParamType) -> serde_json::Value {
    match ty {
        RouteParamType::I32 => serde_json::json!({ "type": "integer", "format": "int32" }),
        RouteParamType::I64 => serde_json::json!({ "type": "integer", "format": "int64" }),
        RouteParamType::U32 => {
            serde_json::json!({ "type": "integer", "minimum": 0, "format": "int64" })
        }
        RouteParamType::U64 => {
            serde_json::json!({ "type": "integer", "minimum": 0, "format": "int64" })
        }
        RouteParamType::Bool => serde_json::json!({ "type": "boolean" }),
        RouteParamType::Str => serde_json::json!({ "type": "string" }),
    }
}

fn openapi_path_from_pattern(pattern: &str) -> (String, Vec<serde_json::Value>) {
    let parsed = match parse_route_pattern(pattern) {
        Ok(value) => value,
        Err(_) => return (pattern.to_string(), Vec::new()),
    };

    let mut path = String::new();
    let mut parameters = Vec::new();

    for seg in parsed.segments {
        path.push('/');
        match seg {
            RouteSegment::Literal(value) => path.push_str(&value),
            RouteSegment::Param { name, ty } => {
                path.push('{');
                path.push_str(&name);
                path.push('}');
                parameters.push(serde_json::json!({
                    "name": name,
                    "in": "path",
                    "required": true,
                    "schema": openapi_schema_for_param_type(&ty),
                }));
            }
        }
    }

    if let Some(name) = parsed.wildcard {
        path.push('/');
        path.push('{');
        path.push_str(&name);
        path.push('}');
        parameters.push(serde_json::json!({
            "name": name,
            "in": "path",
            "required": true,
            "schema": { "type": "string" },
            "description": "Wildcard tail capture",
        }));
    }

    if path.is_empty() {
        path.push('/');
    }

    (path, parameters)
}

fn join_prefix_and_pattern(prefix: &str, pattern: &str) -> String {
    let normalized_prefix = normalize_prefix(prefix);
    let normalized_pattern = if pattern.is_empty() { "/" } else { pattern };

    if normalized_prefix.is_empty() {
        return normalized_pattern.to_string();
    }
    if normalized_pattern == "/" {
        return normalized_prefix;
    }
    format!("{normalized_prefix}{normalized_pattern}")
}

pub fn generate_openapi(service: &TService) -> String {
    // Current v0.1 OpenAPI emission covers route/path metadata and generic 200 responses.
    let mut paths: BTreeMap<String, serde_json::Map<String, serde_json::Value>> = BTreeMap::new();

    for route in &service.routes {
        let method = route.method.to_ascii_lowercase();
        let (openapi_path, parameters) = openapi_path_from_pattern(&route.pattern);
        let full_path = join_prefix_and_pattern(&service.prefix, &openapi_path);

        let mut operation = serde_json::Map::new();
        operation.insert(
            "operationId".to_string(),
            serde_json::Value::String(route.handler_name.clone()),
        );
        operation.insert(
            "responses".to_string(),
            serde_json::json!({
                "200": { "description": "OK" }
            }),
        );
        if !parameters.is_empty() {
            operation.insert(
                "parameters".to_string(),
                serde_json::Value::Array(parameters),
            );
        }
        operation.insert(
            "x-magpie-route-pattern".to_string(),
            serde_json::Value::String(route.pattern.clone()),
        );
        operation.insert(
            "x-magpie-handler".to_string(),
            serde_json::Value::String(route.handler_name.clone()),
        );

        paths
            .entry(full_path)
            .or_default()
            .insert(method, serde_json::Value::Object(operation));
    }

    let mut paths_json = serde_json::Map::new();
    for (path, operations) in paths {
        paths_json.insert(path, serde_json::Value::Object(operations));
    }

    serde_json::to_string(&serde_json::json!({
        "openapi": "3.1.0",
        "info": {
            "title": "Magpie Web Service",
            "version": "0.1.0",
        },
        "paths": paths_json,
    }))
    .unwrap_or_else(|_| {
        "{\"openapi\":\"3.1.0\",\"info\":{\"title\":\"Magpie Web Service\",\"version\":\"0.1.0\"},\"paths\":{}}".to_string()
    })
}

fn generate_openapi_build_artifact(service: &TService) -> String {
    // Build artifact currently mirrors the in-memory OpenAPI generation contract.
    generate_openapi(service)
}

pub fn generate_routes_json(service: &TService) -> String {
    #[derive(Serialize)]
    struct ManifestParam {
        name: String,
        ty: String,
    }

    #[derive(Serialize)]
    struct ManifestRoute {
        method: String,
        pattern: String,
        full_path: String,
        handler: String,
        params: Vec<ManifestParam>,
        wildcard: Option<String>,
    }

    let mut routes = Vec::new();
    for route in &service.routes {
        let mut params = Vec::new();
        let mut wildcard = None;
        if let Ok(parsed) = parse_route_pattern(&route.pattern) {
            for segment in parsed.segments {
                if let RouteSegment::Param { name, ty } = segment {
                    params.push(ManifestParam {
                        name,
                        ty: route_param_type_name(&ty).to_string(),
                    });
                }
            }
            wildcard = parsed.wildcard;
        }

        routes.push(ManifestRoute {
            method: route.method.clone(),
            pattern: route.pattern.clone(),
            full_path: join_prefix_and_pattern(&service.prefix, &route.pattern),
            handler: route.handler_name.clone(),
            params,
            wildcard,
        });
    }
    routes.sort_by(|a, b| {
        a.full_path
            .cmp(&b.full_path)
            .then(a.method.cmp(&b.method))
            .then(a.handler.cmp(&b.handler))
    });

    serde_json::to_string(&serde_json::json!({
        "prefix": service.prefix,
        "middleware": service.middleware,
        "routes": routes,
    }))
    .unwrap_or_else(|_| "{\"prefix\":\"\",\"middleware\":[],\"routes\":[]}".to_string())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WebCommand {
    Dev,
    Build,
    Serve,
}

fn write_generated_routes(manifest_dir: &Path, generated_routes: &str) -> Result<(), String> {
    let output_path = manifest_dir
        .join(".magpie")
        .join("gen")
        .join("webapp_routes.mp");
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create '{}': {err}", parent.display()))?;
    }
    std::fs::write(&output_path, generated_routes)
        .map_err(|err| format!("failed to write '{}': {err}", output_path.display()))
}

fn compile_project(manifest_dir: &Path, release: bool) -> Result<(), String> {
    let manifest_path = manifest_dir.join("Cargo.toml");
    if !manifest_path.is_file() {
        return Err(format!(
            "project manifest not found at '{}'",
            manifest_path.display()
        ));
    }

    let mut command = Command::new("cargo");
    command.arg("build");
    if release {
        command.arg("--release");
    }
    command.current_dir(manifest_dir);

    let status = command
        .status()
        .map_err(|err| format!("failed to run cargo build: {err}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("cargo build failed with status {status}"))
    }
}

fn dev_server_port() -> u16 {
    std::env::var("MAGPIE_WEB_PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(3000)
}

fn parse_http_request_line(request: &str) -> Option<(String, String)> {
    let mut lines = request.lines();
    let first = lines.next()?.trim();
    let mut parts = first.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();
    Some((method, path))
}

fn write_http_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> Result<(), String> {
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(header.as_bytes())
        .map_err(|err| format!("failed to write HTTP header: {err}"))?;
    stream
        .write_all(body)
        .map_err(|err| format!("failed to write HTTP body: {err}"))?;
    stream
        .flush()
        .map_err(|err| format!("failed to flush HTTP response: {err}"))
}

fn content_type_for_path(path: &Path) -> &'static str {
    match path.extension().and_then(|ext| ext.to_str()).unwrap_or("") {
        "html" | "htm" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" => "application/javascript; charset=utf-8",
        "json" => "application/json; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "mp" | "txt" | "md" => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

fn read_dev_request_path(stream: &mut TcpStream) -> Option<(String, String)> {
    let mut request_buf = [0_u8; 4096];
    let read_bytes = stream.read(&mut request_buf).ok()?;
    let request = std::str::from_utf8(&request_buf[..read_bytes]).ok()?;
    let (method, path_with_query) = parse_http_request_line(request)?;
    let path = path_with_query
        .split_once('?')
        .map(|(path, _)| path.to_string())
        .unwrap_or(path_with_query);
    Some((method, path))
}

fn resolve_asset_path(app_dir: &Path, request_path: &str) -> Option<PathBuf> {
    let asset_rel = request_path.strip_prefix("/assets/")?;
    if asset_rel.contains("..") || asset_rel.contains('\\') {
        return None;
    }
    let assets_root = app_dir.join("assets");
    let candidate = assets_root.join(asset_rel);
    if candidate.is_file() {
        Some(candidate)
    } else {
        None
    }
}

fn resolve_route_file(app_dir: &Path, request_path: &str) -> Option<PathBuf> {
    let mappings = scan_webapp_routes(app_dir).ok()?;
    for mapping in mappings {
        let Ok(pattern) = parse_route_pattern(&mapping.pattern) else {
            continue;
        };
        if match_route(&pattern, request_path).is_some() {
            let route_path = app_dir.join(&mapping.file);
            if route_path.is_file() {
                return Some(route_path);
            }
        }
    }
    None
}

fn respond_dev_request(stream: &mut TcpStream, app_dir: &Path) -> Result<(), String> {
    let Some((method, request_path)) = read_dev_request_path(stream) else {
        return write_http_response(
            stream,
            "400 Bad Request",
            "text/plain; charset=utf-8",
            b"invalid HTTP request",
        );
    };

    if method != "GET" {
        return write_http_response(
            stream,
            "405 Method Not Allowed",
            "text/plain; charset=utf-8",
            b"method not allowed",
        );
    }

    if let Some(asset_path) = resolve_asset_path(app_dir, &request_path) {
        return match std::fs::read(&asset_path) {
            Ok(bytes) => {
                write_http_response(stream, "200 OK", content_type_for_path(&asset_path), &bytes)
            }
            Err(err) => write_http_response(
                stream,
                "500 Internal Server Error",
                "text/plain; charset=utf-8",
                format!("failed to read asset '{}': {err}", asset_path.display()).as_bytes(),
            ),
        };
    }

    if let Some(route_file) = resolve_route_file(app_dir, &request_path) {
        return match std::fs::read(&route_file) {
            Ok(bytes) => {
                write_http_response(stream, "200 OK", content_type_for_path(&route_file), &bytes)
            }
            Err(err) => write_http_response(
                stream,
                "500 Internal Server Error",
                "text/plain; charset=utf-8",
                format!(
                    "failed to read route source '{}': {err}",
                    route_file.display()
                )
                .as_bytes(),
            ),
        };
    }

    let mappings = scan_webapp_routes(app_dir).unwrap_or_default();
    let mut body = String::new();
    body.push_str(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Magpie Dev</title></head><body>",
    );
    body.push_str("<h1>Magpie web dev server</h1>");
    body.push_str("<p>Routes are served from <code>app/routes</code> and assets from <code>app/assets</code>.</p>");
    body.push_str("<h2>Known routes</h2><ul>");
    for mapping in &mappings {
        body.push_str("<li><code>");
        body.push_str(&escape_html(&mapping.pattern));
        body.push_str("</code> &rarr; <code>");
        body.push_str(&escape_html(&mapping.file));
        body.push_str("</code></li>");
    }
    body.push_str("</ul></body></html>");
    write_http_response(
        stream,
        "404 Not Found",
        "text/html; charset=utf-8",
        body.as_bytes(),
    )
}

fn run_dev_server(app_dir: &Path) -> Result<(), String> {
    let port = dev_server_port();
    let bind_addr = format!("127.0.0.1:{port}");
    let listener = TcpListener::bind(&bind_addr)
        .map_err(|err| format!("failed to bind dev server on {bind_addr}: {err}"))?;

    println!("magpie web dev: server started on {bind_addr}");
    println!(
        "magpie web dev: watching '{}' and '{}'",
        app_dir.join("routes").display(),
        app_dir.join("assets").display()
    );
    println!("magpie web dev: file updates are reflected on next request");

    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                if let Err(err) = respond_dev_request(&mut stream, app_dir) {
                    eprintln!("magpie web dev: request handling error: {err}");
                }
            }
            Err(err) => {
                eprintln!("magpie web dev: accept error: {err}");
            }
        }
    }

    Ok(())
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<(), String> {
    if !src.is_dir() {
        return Ok(());
    }
    std::fs::create_dir_all(dst)
        .map_err(|err| format!("failed to create '{}': {err}", dst.display()))?;

    let mut entries = Vec::new();
    for entry in std::fs::read_dir(src)
        .map_err(|err| format!("failed to read '{}': {err}", src.display()))?
    {
        entries.push(
            entry.map_err(|err| format!("failed to read entry in '{}': {err}", src.display()))?,
        );
    }
    entries.sort_by_key(|a| a.file_name());

    for entry in entries {
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        let file_type = entry.file_type().map_err(|err| {
            format!(
                "failed to read metadata for '{}': {err}",
                src_path.display()
            )
        })?;
        if file_type.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else if file_type.is_file() {
            std::fs::copy(&src_path, &dst_path).map_err(|err| {
                format!(
                    "failed to copy '{}' to '{}': {err}",
                    src_path.display(),
                    dst_path.display()
                )
            })?;
        }
    }

    Ok(())
}

fn handler_name_from_route_file(file: &str) -> String {
    let base = file.strip_suffix(".mp").unwrap_or(file);
    let mut out = String::from("page_");
    for ch in base.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('_');
        }
    }
    while out.contains("__") {
        out = out.replace("__", "_");
    }
    out.trim_end_matches('_').to_string()
}

fn webapp_service_from_mappings(mappings: &[WebRouteMapping]) -> TService {
    let mut routes = Vec::new();
    for mapping in mappings {
        routes.push(TRoute {
            method: "GET".to_string(),
            pattern: mapping.pattern.clone(),
            handler_name: handler_name_from_route_file(&mapping.file),
        });
    }
    routes.sort_by(|a, b| {
        a.pattern
            .cmp(&b.pattern)
            .then(a.method.cmp(&b.method))
            .then(a.handler_name.cmp(&b.handler_name))
    });

    TService {
        prefix: String::new(),
        routes,
        middleware: Vec::new(),
    }
}

fn project_name(manifest_dir: &Path) -> String {
    manifest_dir
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.to_string())
        .unwrap_or_else(|| "server".to_string())
}

fn ensure_dist_server_binary(manifest_dir: &Path, dist_dir: &Path) -> Result<(), String> {
    let server_dir = dist_dir.join("server");
    std::fs::create_dir_all(&server_dir)
        .map_err(|err| format!("failed to create '{}': {err}", server_dir.display()))?;

    let name = project_name(manifest_dir);
    let dist_binary = server_dir.join(&name);
    let source_binary = manifest_dir.join("target").join("release").join(&name);

    if source_binary.is_file() {
        std::fs::copy(&source_binary, &dist_binary).map_err(|err| {
            format!(
                "failed to copy '{}' to '{}': {err}",
                source_binary.display(),
                dist_binary.display()
            )
        })?;
        return Ok(());
    }
    Err(format!(
        "compiled server binary not found at '{}'; expected project binary '{}' after `cargo build --release`",
        source_binary.display(),
        name
    ))
}

fn discover_server_binary(manifest_dir: &Path) -> Result<PathBuf, String> {
    let server_dir = manifest_dir.join("dist").join("server");
    if !server_dir.is_dir() {
        return Err(format!(
            "server directory not found at '{}'",
            server_dir.display()
        ));
    }

    let preferred = server_dir.join(project_name(manifest_dir));
    if preferred.is_file() {
        return Ok(preferred);
    }

    let mut entries = Vec::new();
    for entry in std::fs::read_dir(&server_dir)
        .map_err(|err| format!("failed to read '{}': {err}", server_dir.display()))?
    {
        entries.push(
            entry.map_err(|err| {
                format!("failed to read entry in '{}': {err}", server_dir.display())
            })?,
        );
    }
    entries.sort_by_key(|a| a.file_name());

    for entry in entries {
        let path = entry.path();
        if path.is_file() {
            return Ok(path);
        }
    }

    Err(format!(
        "no server binary found in '{}'",
        server_dir.display()
    ))
}

pub fn handle_web_command(cmd: WebCommand, manifest_dir: &Path) -> Result<(), String> {
    let app_dir = manifest_dir.join("app");
    let mappings = scan_webapp_routes(&app_dir)?;
    let generated_routes = render_webapp_routes_source(&mappings);
    write_generated_routes(manifest_dir, &generated_routes)?;

    match cmd {
        WebCommand::Dev => {
            compile_project(manifest_dir, false)?;
            run_dev_server(&app_dir)
        }
        WebCommand::Build => {
            compile_project(manifest_dir, true)?;

            let dist_dir = manifest_dir.join("dist");
            let assets_src = app_dir.join("assets");
            let assets_dst = dist_dir.join("assets");
            copy_dir_recursive(&assets_src, &assets_dst)?;

            std::fs::create_dir_all(&dist_dir)
                .map_err(|err| format!("failed to create '{}': {err}", dist_dir.display()))?;

            let page_service = webapp_service_from_mappings(&mappings);
            let openapi_json = generate_openapi_build_artifact(&page_service);
            std::fs::write(dist_dir.join("openapi.json"), openapi_json).map_err(|err| {
                format!(
                    "failed to write '{}': {err}",
                    dist_dir.join("openapi.json").display()
                )
            })?;

            let routes_json = generate_routes_json(&page_service);
            std::fs::write(dist_dir.join("routes.json"), routes_json).map_err(|err| {
                format!(
                    "failed to write '{}': {err}",
                    dist_dir.join("routes.json").display()
                )
            })?;

            ensure_dist_server_binary(manifest_dir, &dist_dir)?;
            Ok(())
        }
        WebCommand::Serve => {
            let binary = discover_server_binary(manifest_dir)?;
            let status = Command::new(&binary)
                .current_dir(manifest_dir)
                .status()
                .map_err(|err| format!("failed to run '{}': {err}", binary.display()))?;
            if status.success() {
                Ok(())
            } else {
                Err(format!(
                    "server binary '{}' exited with status {status}",
                    binary.display()
                ))
            }
        }
    }
}
