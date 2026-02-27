//! magpie_jit
//!
//! REPL/JIT scaffolding for SPEC §23.
#![allow(clippy::result_unit_err)]

use magpie_ast::FileId;
use magpie_codegen_llvm::codegen_module;
use magpie_diag::{Diagnostic, DiagnosticBag, Severity};
use magpie_driver::lower_hir_module_to_mpir;
use magpie_lex::lex;
use magpie_mpir::print_mpir;
use magpie_parse::parse_file;
use magpie_sema::{lower_to_hir, resolve_modules};
use magpie_types::TypeCtx;
use std::collections::HashMap;
use std::io::{self, Write};

const SESSION_HEADER: &str = "MAGPIE_REPL_V1";

#[derive(Clone, Debug, Default)]
pub struct ReplSession {
    pub cell_counter: u64,
    pub symbol_table: HashMap<String, String>,
    pub diagnostics_history: Vec<Diagnostic>,
    pub compiled_modules: Vec<CompiledModule>,
}

#[derive(Clone, Debug, Default)]
pub struct ReplResult {
    pub output: String,
    pub ty: Option<String>,
    pub diagnostics: Vec<Diagnostic>,
    pub llvm_ir: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplCommand {
    Eval(String),
    Type(String),
    Ir(String),
    Llvm(String),
    DiagLast,
    Quit,
}

#[derive(Clone, Debug, Default)]
pub struct CompiledModule {
    pub module_name: String,
    pub fn_name: String,
    pub source: String,
    pub ir: String,
    pub llvm_ir: String,
    pub output: String,
    pub ty: Option<String>,
}

#[derive(Clone, Debug)]
struct EvalArtifacts {
    mpir: String,
    llvm_ir: String,
}

pub fn parse_repl_command(input: &str) -> ReplCommand {
    let trimmed = input.trim();

    if trimmed == ":quit" {
        return ReplCommand::Quit;
    }
    if trimmed == ":diag" || trimmed == ":diag last" {
        return ReplCommand::DiagLast;
    }
    if let Some(rest) = trimmed.strip_prefix(":type") {
        return ReplCommand::Type(rest.trim().to_string());
    }
    if let Some(rest) = trimmed.strip_prefix(":ir") {
        return ReplCommand::Ir(rest.trim().to_string());
    }
    if let Some(rest) = trimmed.strip_prefix(":llvm") {
        return ReplCommand::Llvm(rest.trim().to_string());
    }

    ReplCommand::Eval(input.to_string())
}

pub fn create_repl_session() -> ReplSession {
    ReplSession::default()
}

pub fn run_repl() -> Result<(), String> {
    let mut session = create_repl_session();
    let mut diag = DiagnosticBag::new(200);
    let stdin = io::stdin();
    let mut line = String::new();

    loop {
        print!("magpie> ");
        io::stdout()
            .flush()
            .map_err(|err| format!("failed to flush repl prompt: {err}"))?;
        line.clear();

        let read = stdin
            .read_line(&mut line)
            .map_err(|err| format!("failed to read repl input: {err}"))?;
        if read == 0 {
            break;
        }

        let input = line.trim_end_matches(&['\n', '\r'][..]);
        match parse_repl_command(input) {
            ReplCommand::Quit => break,
            ReplCommand::DiagLast => {
                if let Some(last) = session.diagnostics_history.last() {
                    print_diagnostic(last);
                } else {
                    println!("no diagnostics");
                }
            }
            ReplCommand::Type(expr) => match inspect_type(&session, &expr) {
                Some(ty) => println!("{ty}"),
                None => println!("unknown"),
            },
            ReplCommand::Ir(expr) => match inspect_ir(&session, &expr) {
                Some(ir) => println!("{ir}"),
                None => println!("no IR available"),
            },
            ReplCommand::Llvm(expr) => match inspect_llvm_ir(&session, &expr) {
                Some(ir) => println!("{ir}"),
                None => println!("no LLVM IR available"),
            },
            ReplCommand::Eval(code) => {
                let result = eval_cell(&mut session, &code, &mut diag);
                if let Some(llvm_ir) = &result.llvm_ir {
                    println!("{llvm_ir}");
                } else if result.diagnostics.is_empty() {
                    println!("{}", result.output);
                } else {
                    for diagnostic in &result.diagnostics {
                        print_diagnostic(diagnostic);
                    }
                }
            }
        }
    }

    Ok(())
}

pub fn eval_cell(session: &mut ReplSession, code: &str, diag: &mut DiagnosticBag) -> ReplResult {
    let diag_start = diag.diagnostics.len();
    session.cell_counter = session.cell_counter.saturating_add(1);
    let cell_id = session.cell_counter;
    let module_name = format!("repl.cell.c{cell_id}");
    let fn_name = format!("__repl_eval_{cell_id}");

    let wrapped_source = wrap_cell_source(&module_name, &fn_name, code);
    let mut ok = true;

    if code.trim().is_empty() {
        emit_diag(
            diag,
            "MPJ0001",
            Severity::Error,
            "empty repl cell",
            "REPL cell is empty; provide an expression or function body.",
        );
        ok = false;
    }
    if !balanced_delimiters(code) {
        emit_diag(
            diag,
            "MPJ0002",
            Severity::Error,
            "unbalanced delimiters",
            "REPL cell appears to contain unbalanced (), {}, or [].",
        );
        ok = false;
    }

    let compiled = if ok {
        eval_source_to_artifacts(&wrapped_source, diag).ok()
    } else {
        None
    };

    let inferred_ty = infer_expression_type(code);
    if let Some(ty) = inferred_ty.clone() {
        session.symbol_table.insert(fn_name.clone(), ty.clone());
        if !code.trim().is_empty() {
            session.symbol_table.insert(code.trim().to_string(), ty);
        }
    }

    let llvm_ir_out = compiled.as_ref().map(|artifacts| artifacts.llvm_ir.clone());
    let output = if compiled.is_some() {
        format!("compiled {module_name}::@{fn_name}")
    } else {
        format!("failed to compile {module_name}::@{fn_name}")
    };

    if let Some(artifacts) = compiled {
        session.compiled_modules.push(CompiledModule {
            module_name: module_name.clone(),
            fn_name: fn_name.clone(),
            source: wrapped_source,
            ir: artifacts.mpir,
            llvm_ir: artifacts.llvm_ir,
            output: output.clone(),
            ty: inferred_ty.clone(),
        });
    }

    let diagnostics = collect_new_diagnostics(diag, diag_start);
    session
        .diagnostics_history
        .extend(diagnostics.iter().cloned());

    ReplResult {
        output,
        ty: inferred_ty,
        diagnostics,
        llvm_ir: llvm_ir_out,
    }
}

pub fn inspect_type(session: &ReplSession, expr: &str) -> Option<String> {
    let expr = expr.trim();
    if expr.is_empty() {
        return None;
    }

    session
        .symbol_table
        .get(expr)
        .cloned()
        .or_else(|| {
            session
                .symbol_table
                .get(expr.trim_start_matches('@'))
                .cloned()
        })
        .or_else(|| infer_expression_type(expr))
}

pub fn inspect_ir(session: &ReplSession, fn_name: &str) -> Option<String> {
    let query = fn_name.trim().trim_start_matches('@');
    if query.is_empty() {
        return session.compiled_modules.last().map(|m| m.ir.clone());
    }

    session
        .compiled_modules
        .iter()
        .rev()
        .find(|m| m.fn_name == query || m.module_name == query)
        .map(|m| m.ir.clone())
}

pub fn inspect_llvm_ir(session: &ReplSession, fn_name: &str) -> Option<String> {
    let query = fn_name.trim().trim_start_matches('@');
    if query.is_empty() {
        return session.compiled_modules.last().map(|m| m.llvm_ir.clone());
    }

    session
        .compiled_modules
        .iter()
        .rev()
        .find(|m| m.fn_name == query || m.module_name == query)
        .map(|m| m.llvm_ir.clone())
}

pub fn save_session(session: &ReplSession) -> String {
    let mut out = String::new();
    out.push_str(SESSION_HEADER);
    out.push('\n');
    out.push_str(&format!("cell_counter\t{}\n", session.cell_counter));
    out.push_str(&format!("symbols\t{}\n", session.symbol_table.len()));

    let mut symbols = session
        .symbol_table
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect::<Vec<_>>();
    symbols.sort_by(|a, b| a.0.cmp(b.0).then(a.1.cmp(b.1)));

    for (key, value) in symbols {
        out.push_str("sym\t");
        out.push_str(&escape_field(key));
        out.push('\t');
        out.push_str(&escape_field(value));
        out.push('\n');
    }

    out.push_str(&format!(
        "diagnostics\t{}\n",
        session.diagnostics_history.len()
    ));
    for diag in &session.diagnostics_history {
        out.push_str("diag\t");
        out.push_str(&escape_field(&diag.code));
        out.push('\t');
        out.push_str(severity_to_str(&diag.severity));
        out.push('\t');
        out.push_str(&escape_field(&diag.title));
        out.push('\t');
        out.push_str(&escape_field(&diag.message));
        out.push('\n');
    }

    out.push_str(&format!("modules\t{}\n", session.compiled_modules.len()));
    for module in &session.compiled_modules {
        out.push_str("mod\t");
        out.push_str(&escape_field(&module.module_name));
        out.push('\t');
        out.push_str(&escape_field(&module.fn_name));
        out.push('\t');
        out.push_str(&escape_field(module.ty.as_deref().unwrap_or("")));
        out.push('\t');
        out.push_str(&escape_field(&module.output));
        out.push('\t');
        out.push_str(&escape_field(&module.source));
        out.push('\t');
        out.push_str(&escape_field(&module.ir));
        out.push('\t');
        out.push_str(&escape_field(&module.llvm_ir));
        out.push('\n');
    }

    out
}

pub fn load_session(serialized: &str) -> Result<ReplSession, String> {
    let mut lines = serialized.lines();
    let Some(header) = lines.next() else {
        return Err("session payload is empty".to_string());
    };
    if header != SESSION_HEADER {
        return Err("invalid session payload header".to_string());
    }

    let mut session = create_repl_session();

    for line in lines {
        if line.trim().is_empty() {
            continue;
        }

        let parts = line.split('\t').collect::<Vec<_>>();
        match parts.first().copied().unwrap_or_default() {
            "cell_counter" => {
                if parts.len() != 2 {
                    return Err("invalid cell_counter record".to_string());
                }
                session.cell_counter = parts[1]
                    .parse::<u64>()
                    .map_err(|_| "invalid cell_counter value".to_string())?;
            }
            "symbols" | "diagnostics" | "modules" => {
                // Count records are informational and accepted as-is.
            }
            "sym" => {
                if parts.len() != 3 {
                    return Err("invalid sym record".to_string());
                }
                let key = unescape_field(parts[1])?;
                let value = unescape_field(parts[2])?;
                session.symbol_table.insert(key, value);
            }
            "diag" => {
                if parts.len() != 5 {
                    return Err("invalid diag record".to_string());
                }
                let code = unescape_field(parts[1])?;
                let severity =
                    str_to_severity(parts[2]).ok_or_else(|| "invalid diag severity".to_string())?;
                let title = unescape_field(parts[3])?;
                let message = unescape_field(parts[4])?;

                session.diagnostics_history.push(Diagnostic {
                    code,
                    severity,
                    title,
                    primary_span: None,
                    secondary_spans: Vec::new(),
                    message,
                    explanation_md: None,
                    why: None,
                    suggested_fixes: Vec::new(),
                    rag_bundle: Vec::new(),
                    related_docs: Vec::new(),
                });
            }
            "mod" => {
                if parts.len() != 8 {
                    return Err("invalid mod record".to_string());
                }
                let module_name = unescape_field(parts[1])?;
                let fn_name = unescape_field(parts[2])?;
                let ty = unescape_field(parts[3])?;
                let output = unescape_field(parts[4])?;
                let source = unescape_field(parts[5])?;
                let ir = unescape_field(parts[6])?;
                let llvm_ir = unescape_field(parts[7])?;

                session.compiled_modules.push(CompiledModule {
                    module_name,
                    fn_name,
                    source,
                    ir,
                    llvm_ir,
                    output,
                    ty: if ty.is_empty() { None } else { Some(ty) },
                });
            }
            _ => return Err("unknown session record kind".to_string()),
        }
    }

    Ok(session)
}

fn wrap_cell_source(module_name: &str, fn_name: &str, code: &str) -> String {
    let body = if code.trim().is_empty() {
        "  ret const.i32 0".to_string()
    } else {
        code.lines()
            .map(|line| format!("  {line}"))
            .collect::<Vec<_>>()
            .join("\n")
    };

    format!(
        "module {module_name}\nexports {{ @{fn_name} }}\nimports {{ }}\ndigest \"\"\n\nfn @{fn_name}() -> i32 {{\nbb0:\n{body}\n}}\n"
    )
}

pub fn eval_source_to_llvm_ir(source: &str, diag: &mut DiagnosticBag) -> Result<String, ()> {
    eval_source_to_artifacts(source, diag).map(|artifacts| artifacts.llvm_ir)
}

fn eval_source_to_artifacts(source: &str, diag: &mut DiagnosticBag) -> Result<EvalArtifacts, ()> {
    let file_id = FileId(0);
    let lex_errors = diag.error_count();
    let tokens = lex(file_id, source, diag);
    if diag.error_count() > lex_errors {
        return Err(());
    }

    let ast = parse_file(&tokens, file_id, diag)?;
    let resolved_modules = resolve_modules(std::slice::from_ref(&ast), diag)?;

    let mut type_ctx = TypeCtx::new();
    if let Some(resolved) = resolved_modules.first() {
        let hir_module = lower_to_hir(resolved, &mut type_ctx, diag)?;
        let mpir_module = lower_hir_module_to_mpir(&hir_module, &type_ctx);
        let mpir = print_mpir(&mpir_module, &type_ctx);
        let llvm_ir = match codegen_module(&mpir_module, &type_ctx) {
            Ok(llvm_ir) => llvm_ir,
            Err(err) => {
                emit_diag(diag, "MPG0001", Severity::Error, "llvm codegen failed", err);
                return Err(());
            }
        };
        return Ok(EvalArtifacts { mpir, llvm_ir });
    }

    emit_diag(
        diag,
        "MPJ0003",
        Severity::Error,
        "empty lowered module",
        "REPL evaluation produced no lowered modules.",
    );
    Err(())
}

fn infer_expression_type(expr: &str) -> Option<String> {
    let s = expr.trim();
    if s.is_empty() {
        return None;
    }
    if s == "true" || s == "false" {
        return Some("bool".to_string());
    }
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        return Some("String".to_string());
    }
    if s.parse::<i32>().is_ok() {
        return Some("i32".to_string());
    }
    if s.parse::<i64>().is_ok() {
        return Some("i64".to_string());
    }
    if s.parse::<f64>().is_ok() {
        return Some("f64".to_string());
    }
    None
}

fn balanced_delimiters(source: &str) -> bool {
    let mut stack: Vec<char> = Vec::new();
    for ch in source.chars() {
        match ch {
            '(' | '{' | '[' => stack.push(ch),
            ')' => {
                if stack.pop() != Some('(') {
                    return false;
                }
            }
            '}' => {
                if stack.pop() != Some('{') {
                    return false;
                }
            }
            ']' => {
                if stack.pop() != Some('[') {
                    return false;
                }
            }
            _ => {}
        }
    }
    stack.is_empty()
}

fn emit_diag(
    bag: &mut DiagnosticBag,
    code: &str,
    severity: Severity,
    title: impl Into<String>,
    message: impl Into<String>,
) {
    bag.emit(Diagnostic {
        code: code.to_string(),
        severity,
        title: title.into(),
        primary_span: None,
        secondary_spans: Vec::new(),
        message: message.into(),
        explanation_md: None,
        why: None,
        suggested_fixes: Vec::new(),
        rag_bundle: Vec::new(),
        related_docs: Vec::new(),
    });
}

fn print_diagnostic(diag: &Diagnostic) {
    println!(
        "{} [{}] {}",
        severity_to_str(&diag.severity),
        diag.code,
        diag.message
    );
}

fn collect_new_diagnostics(diag: &DiagnosticBag, from: usize) -> Vec<Diagnostic> {
    diag.diagnostics
        .get(from..)
        .map(|slice| slice.to_vec())
        .unwrap_or_default()
}

fn severity_to_str(severity: &Severity) -> &'static str {
    match severity {
        Severity::Error => "error",
        Severity::Warning => "warning",
        Severity::Info => "info",
        Severity::Hint => "hint",
    }
}

fn str_to_severity(s: &str) -> Option<Severity> {
    match s {
        "error" => Some(Severity::Error),
        "warning" => Some(Severity::Warning),
        "info" => Some(Severity::Info),
        "hint" => Some(Severity::Hint),
        _ => None,
    }
}

fn escape_field(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

fn unescape_field(s: &str) -> Result<String, String> {
    let mut out = String::new();
    let mut chars = s.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }

        let Some(next) = chars.next() else {
            return Err("trailing escape in session payload".to_string());
        };
        match next {
            '\\' => out.push('\\'),
            't' => out.push('\t'),
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            _ => return Err("invalid escape sequence in session payload".to_string()),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_repl_command_accepts_diag_last_alias() {
        assert_eq!(parse_repl_command(":diag"), ReplCommand::DiagLast);
        assert_eq!(parse_repl_command(":diag last"), ReplCommand::DiagLast);
    }
}
