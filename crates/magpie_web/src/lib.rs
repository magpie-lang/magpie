//! magpie_web

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
        config: McpConfig {
            allowed_roots: Vec::new(),
            allow_network: false,
            allow_subprocess: false,
        },
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

fn mcp_driver_config(params: Option<&serde_json::Value>) -> magpie_driver::DriverConfig {
    let mut config = magpie_driver::DriverConfig::default();
    let Some(params) = params else {
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
    if let Some(max_errors) = params.get("max_errors").and_then(|v| v.as_u64()) {
        config.max_errors = max_errors as usize;
    }
    if let Some(token_budget) = params.get("token_budget").and_then(|v| v.as_u64()) {
        config.token_budget = Some(token_budget as u32);
    }
    if let Some(llm_mode) = params.get("llm_mode").and_then(|v| v.as_bool()) {
        config.llm_mode = llm_mode;
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
                "method": request.method,
            })),
        );
    };

    let llm = request
        .params
        .as_ref()
        .and_then(|params| params.get("llm"))
        .cloned();

    let result = match request.method.as_str() {
        "magpie.build" => {
            let config = mcp_driver_config(request.params.as_ref());
            let build = magpie_driver::build(&config);
            serde_json::json!({
                "tool": "magpie.build",
                "status": if build.success { "ok" } else { "failed" },
                "build": build,
                "config": config,
                "llm": llm,
            })
        }
        "magpie.explain" => {
            let code = request
                .params
                .as_ref()
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
            serde_json::json!({
                "tool": "magpie.explain",
                "status": if explanation.is_some() { "ok" } else { "unknown_code" },
                "code": code,
                "explanation": explanation,
                "llm": llm,
            })
        }
        // TODO(G62): wire remaining MCP tools to their real subsystems.
        other => serde_json::json!({
            "tool": other,
            "status": "stub",
            "message": "Tool registered but backend execution is not implemented yet.",
            "todo": "TODO(G62): connect this MCP tool to a concrete Magpie subsystem.",
            "llm": llm,
        }),
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
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_entry_path() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "magpie_web_mcp_{}_{}",
            std::process::id(),
            nonce
        ));
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

    #[test]
    fn mcp_build_calls_driver_build() {
        let server = create_mcp_server();
        let entry = temp_entry_path();
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
        assert!(response.error.is_none(), "response error: {:?}", response.error);
        let result = response.result.expect("result should be present");
        assert_eq!(result["tool"], "magpie.build");
        assert_eq!(result["status"], "ok");
        assert_eq!(result["build"]["success"], true);
    }

    #[test]
    fn mcp_explain_calls_driver_explain() {
        let server = create_mcp_server();
        let request = McpRequest {
            jsonrpc: "2.0".to_string(),
            method: "magpie.explain".to_string(),
            params: Some(serde_json::json!({
                "code": "MPO0003",
            })),
            id: Some(serde_json::json!(2)),
        };
        let response = handle_mcp_request(&server, &request);
        assert!(response.error.is_none(), "response error: {:?}", response.error);
        let result = response.result.expect("result should be present");
        assert_eq!(result["tool"], "magpie.explain");
        assert_eq!(result["status"], "ok");
        assert!(result["explanation"].is_string());
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
    entries.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

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
    // TODO(G63): enrich OpenAPI with request/response schemas inferred from typed handlers.
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

    serde_json::to_string_pretty(&serde_json::json!({
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
    // TODO(G63): split this into a full spec generator once handler-level schema metadata exists.
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

    serde_json::to_string_pretty(&serde_json::json!({
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

fn respond_dev_placeholder(stream: &mut TcpStream) -> Result<(), String> {
    let mut request_buf = [0_u8; 4096];
    let _ = stream.read(&mut request_buf);

    let body = "<!doctype html><html><head><meta charset=\"utf-8\"><title>Magpie Dev</title></head><body><h1>Magpie dev server</h1><p>Placeholder page from magpie web dev.</p></body></html>";
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );

    stream
        .write_all(response.as_bytes())
        .map_err(|err| format!("failed to write HTTP response: {err}"))?;
    stream
        .flush()
        .map_err(|err| format!("failed to flush HTTP response: {err}"))?;
    Ok(())
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
    // TODO(G67): replace placeholder loop with file-watcher based hot reload + restart.
    println!("magpie web dev: hot reload stub enabled");

    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                if let Err(err) = respond_dev_placeholder(&mut stream) {
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
    entries.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

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

    if !dist_binary.exists() {
        std::fs::write(
            &dist_binary,
            "#!/usr/bin/env sh\nprintf '%s\\n' 'magpie web serve stub: replace with dist/server binary'\n",
        )
        .map_err(|err| format!("failed to write '{}': {err}", dist_binary.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&dist_binary)
                .map_err(|err| format!("failed to stat '{}': {err}", dist_binary.display()))?
                .permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&dist_binary, perms)
                .map_err(|err| format!("failed to chmod '{}': {err}", dist_binary.display()))?;
        }
    }

    Ok(())
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
    entries.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

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

            let openapi_json = generate_openapi_build_artifact(&TService::default());
            std::fs::write(dist_dir.join("openapi.json"), openapi_json).map_err(|err| {
                format!(
                    "failed to write '{}': {err}",
                    dist_dir.join("openapi.json").display()
                )
            })?;

            let page_service = webapp_service_from_mappings(&mappings);
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
