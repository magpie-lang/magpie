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

        parsed.segments.push(RouteSegment::Literal((*seg).to_string()));
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
        assert_eq!(resp.headers.get("x-request-id").map(String::as_str), Some("req-123"));
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

        assert_eq!(render_html(&node), "<div data-x=\"a&amp;b\">&lt;hello&gt;</div>");
    }
}
