//! Canonical Source Normal Form (CSNF) formatter for Magpie.

use std::collections::{BTreeMap, HashMap};

use magpie_ast::*;

pub fn format_csnf(ast: &AstFile, _source_map: &SourceMap) -> String {
    let mut out = String::new();
    out.push_str("module ");
    out.push_str(&ast.header.node.module_path.node.to_string());
    out.push('\n');

    out.push_str("exports ");
    out.push_str(&format_braced_list(sorted_export_items(&ast.header.node.exports)));
    out.push('\n');

    out.push_str("imports ");
    out.push_str(&format_braced_list(sorted_import_groups(&ast.header.node.imports)));
    out.push('\n');

    // Placeholder; update_digest will replace with the canonical value.
    out.push_str("digest \"\"\n");

    if !ast.decls.is_empty() {
        out.push('\n');
        for (idx, decl) in ast.decls.iter().enumerate() {
            out.push_str(&print_decl(&decl.node));
            if idx + 1 < ast.decls.len() {
                out.push_str("\n\n");
            }
        }
    }

    update_digest(&ensure_single_trailing_newline(out))
}

pub fn compute_digest(canonical_source: &str) -> String {
    let stripped = strip_digest_line(canonical_source);
    blake3::hash(stripped.as_bytes()).to_hex().to_string()
}

pub fn update_digest(source: &str) -> String {
    let digest = compute_digest(source);

    let mut lines: Vec<String> = source.lines().map(str::to_owned).collect();
    let mut insert_idx = None;
    lines.retain(|line| {
        let is_digest = line.trim_start().starts_with("digest ");
        if is_digest && insert_idx.is_none() {
            insert_idx = Some(0);
        }
        !is_digest
    });

    if insert_idx.is_none() {
        insert_idx = lines
            .iter()
            .position(|line| line.trim_start().starts_with("imports "))
            .map(|i| i + 1)
            .or(Some(lines.len().min(3)));
    } else {
        // Insert where digest line canonically belongs: after imports if present.
        insert_idx = lines
            .iter()
            .position(|line| line.trim_start().starts_with("imports "))
            .map(|i| i + 1)
            .or(insert_idx);
    }

    lines.insert(
        insert_idx.unwrap_or(0),
        format!("digest \"{}\"", digest),
    );

    ensure_single_trailing_newline(lines.join("\n"))
}

pub fn print_type(ty: &AstType) -> String {
    let mut out = String::new();
    if let Some(ownership) = &ty.ownership {
        out.push_str(match ownership {
            OwnershipMod::Shared => "shared",
            OwnershipMod::Borrow => "borrow",
            OwnershipMod::MutBorrow => "mutborrow",
            OwnershipMod::Weak => "weak",
        });
        out.push(' ');
    }

    out.push_str(&match &ty.base {
        AstBaseType::Prim(p) => p.clone(),
        AstBaseType::Named { path, name, targs } => {
            let mut s = String::new();
            if let Some(path) = path {
                s.push_str(&path.to_string());
                s.push('.');
            }
            s.push_str(name);
            s.push_str(&print_type_args(targs));
            s
        }
        AstBaseType::Builtin(b) => print_builtin_type(b),
        AstBaseType::Callable { sig_ref } => format!("TCallable<{}>", sig_ref),
        AstBaseType::RawPtr(inner) => format!("rawptr<{}>", print_type(inner)),
    });

    out
}

pub fn print_value_ref(v: &AstValueRef) -> String {
    match v {
        AstValueRef::Local(name) => format!("%{}", name),
        AstValueRef::Const(c) => print_const_expr(c),
    }
}

pub fn print_op(op: &AstOp) -> String {
    print_op_with_bb_map(op, &HashMap::new())
}

pub fn print_op_void(op: &AstOpVoid) -> String {
    print_op_void_with_bb_map(op, &HashMap::new())
}

pub fn print_terminator(t: &AstTerminator) -> String {
    print_terminator_with_bb_map(t, &HashMap::new())
}

fn print_decl(decl: &AstDecl) -> String {
    match decl {
        AstDecl::Fn(f) => print_fn_decl(f, "fn", None),
        AstDecl::AsyncFn(f) => print_fn_decl(f, "async fn", None),
        AstDecl::UnsafeFn(f) => print_fn_decl(f, "unsafe fn", None),
        AstDecl::GpuFn(g) => print_fn_decl(&g.inner, "gpu fn", Some(&g.target)),
        AstDecl::HeapStruct(s) => print_struct_decl(s, "heap struct"),
        AstDecl::ValueStruct(s) => print_struct_decl(s, "value struct"),
        AstDecl::HeapEnum(e) => print_enum_decl(e, "heap enum"),
        AstDecl::ValueEnum(e) => print_enum_decl(e, "value enum"),
        AstDecl::Extern(e) => print_extern_decl(e),
        AstDecl::Global(g) => print_global_decl(g),
        AstDecl::Impl(i) => print_impl_decl(i),
        AstDecl::Sig(s) => print_sig_decl(s),
    }
}

fn print_doc_lines(doc: &Option<String>) -> String {
    match doc {
        None => String::new(),
        Some(text) => {
            let mut out = String::new();
            for line in text.lines() {
                out.push_str(";;; ");
                out.push_str(line);
                out.push('\n');
            }
            out
        }
    }
}

fn print_fn_decl(decl: &AstFnDecl, prefix: &str, gpu_target: Option<&str>) -> String {
    let mut out = String::new();
    out.push_str(&print_doc_lines(&decl.doc));

    out.push_str(prefix);
    out.push(' ');
    out.push_str(&decl.name);
    out.push('(');
    out.push_str(
        &decl
            .params
            .iter()
            .map(|p| format!("%{}: {}", p.name, print_type(&p.ty.node)))
            .collect::<Vec<_>>()
            .join(", "),
    );
    out.push(')');
    out.push_str(" -> ");
    out.push_str(&print_type(&decl.ret_ty.node));

    if let Some(target) = gpu_target {
        out.push_str(" target(");
        out.push_str(target);
        out.push(')');
    }

    if let Some(meta) = &decl.meta {
        out.push(' ');
        out.push_str(&print_fn_meta(meta));
    }

    out.push_str(" {\n");

    let bb_map = canonical_block_map(&decl.blocks);
    for (idx, block) in decl.blocks.iter().enumerate() {
        let new_label = remap_bb_label(block.node.label, &bb_map);
        out.push_str("  bb");
        out.push_str(&new_label.to_string());
        out.push_str(":\n");

        for instr in &block.node.instrs {
            out.push_str(&print_instr_with_bb_map(&instr.node, &bb_map, 4));
            out.push('\n');
        }

        out.push_str("    ");
        out.push_str(&print_terminator_with_bb_map(
            &block.node.terminator.node,
            &bb_map,
        ));
        out.push('\n');

        if idx + 1 < decl.blocks.len() {
            out.push('\n');
        }
    }

    out.push('}');
    out
}

fn print_fn_meta(meta: &AstFnMeta) -> String {
    let mut uses = meta.uses.clone();
    uses.sort();

    let mut effects = meta.effects.clone();
    effects.sort();

    let mut cost = meta.cost.clone();
    cost.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let cost_items = cost
        .into_iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>();

    format!(
        "meta {{ uses {} effects {} cost {} }}",
        format_braced_list(uses),
        format_braced_list(effects),
        format_braced_list(cost_items),
    )
}

fn print_struct_decl(decl: &AstStructDecl, prefix: &str) -> String {
    let mut out = String::new();
    out.push_str(&print_doc_lines(&decl.doc));
    out.push_str(prefix);
    out.push(' ');
    out.push_str(&decl.name);
    out.push_str(&print_type_params(&decl.type_params));
    out.push_str(" {\n");

    for field in &decl.fields {
        out.push_str("  field ");
        out.push_str(&field.name);
        out.push_str(": ");
        out.push_str(&print_type(&field.ty.node));
        out.push('\n');
    }

    out.push('}');
    out
}

fn print_enum_decl(decl: &AstEnumDecl, prefix: &str) -> String {
    let mut out = String::new();
    out.push_str(&print_doc_lines(&decl.doc));
    out.push_str(prefix);
    out.push(' ');
    out.push_str(&decl.name);
    out.push_str(&print_type_params(&decl.type_params));
    out.push_str(" {\n");

    for variant in &decl.variants {
        out.push_str("  variant ");
        out.push_str(&variant.name);
        out.push_str(" { ");
        out.push_str(
            &variant
                .fields
                .iter()
                .map(|f| format!("field {}: {}", f.name, print_type(&f.ty.node)))
                .collect::<Vec<_>>()
                .join(", "),
        );
        out.push_str(" }\n");
    }

    out.push('}');
    out
}

fn print_extern_decl(decl: &AstExternModule) -> String {
    let mut out = String::new();
    out.push_str(&print_doc_lines(&decl.doc));
    out.push_str("extern ");
    out.push_str(&print_string_literal(&decl.abi));
    out.push_str(" module ");
    out.push_str(&decl.name);
    out.push_str(" {\n");

    for item in &decl.items {
        out.push_str("  fn ");
        out.push_str(&item.name);
        out.push('(');
        out.push_str(
            &item
                .params
                .iter()
                .map(|p| format!("%{}: {}", p.name, print_type(&p.ty.node)))
                .collect::<Vec<_>>()
                .join(", "),
        );
        out.push(')');
        out.push_str(" -> ");
        out.push_str(&print_type(&item.ret_ty.node));

        if !item.attrs.is_empty() {
            let mut attrs = item.attrs.clone();
            attrs.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            out.push_str(" attrs ");
            out.push_str(&format_braced_list(
                attrs
                    .into_iter()
                    .map(|(k, v)| format!("{k}={}", print_string_literal(&v)))
                    .collect::<Vec<_>>(),
            ));
        }

        out.push('\n');
    }

    out.push('}');
    out
}

fn print_global_decl(decl: &AstGlobalDecl) -> String {
    let mut out = String::new();
    out.push_str(&print_doc_lines(&decl.doc));
    out.push_str("global ");
    out.push_str(&decl.name);
    out.push_str(": ");
    out.push_str(&print_type(&decl.ty.node));
    out.push_str(" = ");
    out.push_str(&print_const_expr(&decl.init));
    out
}

fn print_impl_decl(decl: &AstImplDecl) -> String {
    format!(
        "impl {} for {} = {}",
        decl.trait_name,
        print_type(&decl.for_type),
        decl.fn_ref
    )
}

fn print_sig_decl(decl: &AstSigDecl) -> String {
    format!(
        "sig {}({}) -> {}",
        decl.name,
        decl.param_types
            .iter()
            .map(print_type)
            .collect::<Vec<_>>()
            .join(", "),
        print_type(&decl.ret_ty)
    )
}

fn print_instr_with_bb_map(instr: &AstInstr, bb_map: &HashMap<u32, u32>, indent: usize) -> String {
    let pad = " ".repeat(indent);
    match instr {
        AstInstr::Assign { name, ty, op } => format!(
            "{pad}%{}: {} = {}",
            name,
            print_type(&ty.node),
            print_op_with_bb_map(op, bb_map)
        ),
        AstInstr::Void(op) => format!("{pad}{}", print_op_void_with_bb_map(op, bb_map)),
        AstInstr::UnsafeBlock(inner) => {
            let mut out = String::new();
            out.push_str(&pad);
            out.push_str("unsafe {\n");
            for i in inner {
                out.push_str(&print_instr_with_bb_map(&i.node, bb_map, indent + 2));
                out.push('\n');
            }
            out.push_str(&pad);
            out.push('}');
            out
        }
    }
}

fn print_op_with_bb_map(op: &AstOp, bb_map: &HashMap<u32, u32>) -> String {
    match op {
        AstOp::Const(c) => print_const_expr(c),
        AstOp::BinOp { kind, lhs, rhs } => format!(
            "{} {}",
            print_bin_op(kind),
            print_kv_args(vec![
                ("lhs".to_string(), print_value_ref(lhs)),
                ("rhs".to_string(), print_value_ref(rhs)),
            ])
        ),
        AstOp::Cmp {
            kind,
            pred,
            lhs,
            rhs,
        } => {
            let op_name = match kind {
                CmpKind::ICmp => format!("icmp.{pred}"),
                CmpKind::FCmp => format!("fcmp.{pred}"),
            };
            format!(
                "{} {}",
                op_name,
                print_kv_args(vec![
                    ("lhs".to_string(), print_value_ref(lhs)),
                    ("rhs".to_string(), print_value_ref(rhs)),
                ])
            )
        }
        AstOp::Call { callee, targs, args } => format!(
            "call {}{} {}",
            callee,
            print_type_args(targs),
            print_arg_kv_args(args)
        ),
        AstOp::CallIndirect { callee, args } => format!(
            "call.indirect {} {}",
            print_value_ref(callee),
            print_arg_kv_args(args)
        ),
        AstOp::Try { callee, targs, args } => format!(
            "try {}{} {}",
            callee,
            print_type_args(targs),
            print_arg_kv_args(args)
        ),
        AstOp::SuspendCall { callee, targs, args } => format!(
            "suspend.call {}{} {}",
            callee,
            print_type_args(targs),
            print_arg_kv_args(args)
        ),
        AstOp::SuspendAwait { fut } => format!(
            "suspend.await {}",
            print_kv_args(vec![("fut".to_string(), print_value_ref(fut)),])
        ),
        AstOp::New { ty, fields } => format!(
            "new {} {}",
            print_type(ty),
            print_kv_args(
                fields
                    .iter()
                    .map(|(k, v)| (k.clone(), print_value_ref(v)))
                    .collect(),
            )
        ),
        AstOp::GetField { obj, field } => format!(
            "getfield {}",
            print_kv_args(vec![
                ("obj".to_string(), print_value_ref(obj)),
                ("field".to_string(), field.clone()),
            ])
        ),
        AstOp::Phi { ty, incomings } => {
            let mut incoming = incomings
                .iter()
                .map(|(bb, v)| (remap_bb_label(*bb, bb_map), print_value_ref(v)))
                .collect::<Vec<_>>();
            incoming.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            format!(
                "phi {} {}",
                print_type(ty),
                if incoming.is_empty() {
                    "{ }".to_string()
                } else {
                    format!(
                        "{{ {} }}",
                        incoming
                            .into_iter()
                            .map(|(bb, v)| format!("[bb{}:{}]", bb, v))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
            )
        }
        AstOp::EnumNew { variant, args } => {
            let mut sorted = args.clone();
            sorted.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| print_value_ref(&a.1).cmp(&print_value_ref(&b.1))));
            format!(
                "enum.new<{}> {}",
                variant,
                print_kv_args(
                    sorted
                        .into_iter()
                        .map(|(k, v)| (k, print_value_ref(&v)))
                        .collect(),
                )
            )
        }
        AstOp::EnumTag { v } => format!(
            "enum.tag {}",
            print_kv_args(vec![("v".to_string(), print_value_ref(v)),])
        ),
        AstOp::EnumPayload { variant, v } => format!(
            "enum.payload<{}> {}",
            variant,
            print_kv_args(vec![("v".to_string(), print_value_ref(v)),])
        ),
        AstOp::EnumIs { variant, v } => format!(
            "enum.is<{}> {}",
            variant,
            print_kv_args(vec![("v".to_string(), print_value_ref(v)),])
        ),
        AstOp::Share { v } => format!(
            "share {}",
            print_kv_args(vec![("v".to_string(), print_value_ref(v)),])
        ),
        AstOp::CloneShared { v } => format!(
            "clone.shared {}",
            print_kv_args(vec![("v".to_string(), print_value_ref(v)),])
        ),
        AstOp::CloneWeak { v } => format!(
            "clone.weak {}",
            print_kv_args(vec![("v".to_string(), print_value_ref(v)),])
        ),
        AstOp::WeakDowngrade { v } => format!(
            "weak.downgrade {}",
            print_kv_args(vec![("v".to_string(), print_value_ref(v)),])
        ),
        AstOp::WeakUpgrade { v } => format!(
            "weak.upgrade {}",
            print_kv_args(vec![("v".to_string(), print_value_ref(v)),])
        ),
        AstOp::Cast { from, to, v } => format!(
            "cast<{}, {}> {}",
            print_type(from),
            print_type(to),
            print_kv_args(vec![("v".to_string(), print_value_ref(v)),])
        ),
        AstOp::BorrowShared { v } => format!(
            "borrow.shared {}",
            print_kv_args(vec![("v".to_string(), print_value_ref(v)),])
        ),
        AstOp::BorrowMut { v } => format!(
            "borrow.mut {}",
            print_kv_args(vec![("v".to_string(), print_value_ref(v)),])
        ),
        AstOp::PtrNull { ty } => format!("ptr.null<{}>", print_type(ty)),
        AstOp::PtrAddr { ty, p } => format!(
            "ptr.addr<{}> {}",
            print_type(ty),
            print_kv_args(vec![("p".to_string(), print_value_ref(p)),])
        ),
        AstOp::PtrFromAddr { ty, addr } => format!(
            "ptr.from_addr<{}> {}",
            print_type(ty),
            print_kv_args(vec![("addr".to_string(), print_value_ref(addr)),])
        ),
        AstOp::PtrAdd { ty, p, count } => format!(
            "ptr.add<{}> {}",
            print_type(ty),
            print_kv_args(vec![
                ("p".to_string(), print_value_ref(p)),
                ("count".to_string(), print_value_ref(count)),
            ])
        ),
        AstOp::PtrLoad { ty, p } => format!(
            "ptr.load<{}> {}",
            print_type(ty),
            print_kv_args(vec![("p".to_string(), print_value_ref(p)),])
        ),
        AstOp::CallableCapture { fn_ref, captures } => format!(
            "callable.capture {} {}",
            fn_ref,
            print_kv_args(
                captures
                    .iter()
                    .map(|(k, v)| (k.clone(), print_value_ref(v)))
                    .collect(),
            )
        ),
        AstOp::ArrNew { elem_ty, cap } => format!(
            "arr.new<{}> {}",
            print_type(elem_ty),
            print_kv_args(vec![("cap".to_string(), print_value_ref(cap)),])
        ),
        AstOp::ArrLen { arr } => format!(
            "arr.len {}",
            print_kv_args(vec![("arr".to_string(), print_value_ref(arr)),])
        ),
        AstOp::ArrGet { arr, idx } => format!(
            "arr.get {}",
            print_kv_args(vec![
                ("arr".to_string(), print_value_ref(arr)),
                ("idx".to_string(), print_value_ref(idx)),
            ])
        ),
        AstOp::ArrPop { arr } => format!(
            "arr.pop {}",
            print_kv_args(vec![("arr".to_string(), print_value_ref(arr)),])
        ),
        AstOp::ArrSlice { arr, start, end } => format!(
            "arr.slice {}",
            print_kv_args(vec![
                ("arr".to_string(), print_value_ref(arr)),
                ("start".to_string(), print_value_ref(start)),
                ("end".to_string(), print_value_ref(end)),
            ])
        ),
        AstOp::ArrContains { arr, val } => format!(
            "arr.contains {}",
            print_kv_args(vec![
                ("arr".to_string(), print_value_ref(arr)),
                ("val".to_string(), print_value_ref(val)),
            ])
        ),
        AstOp::ArrMap { arr, func } => format!(
            "arr.map {}",
            print_kv_args(vec![
                ("arr".to_string(), print_value_ref(arr)),
                ("fn".to_string(), print_value_ref(func)),
            ])
        ),
        AstOp::ArrFilter { arr, func } => format!(
            "arr.filter {}",
            print_kv_args(vec![
                ("arr".to_string(), print_value_ref(arr)),
                ("fn".to_string(), print_value_ref(func)),
            ])
        ),
        AstOp::ArrReduce { arr, init, func } => format!(
            "arr.reduce {}",
            print_kv_args(vec![
                ("arr".to_string(), print_value_ref(arr)),
                ("init".to_string(), print_value_ref(init)),
                ("fn".to_string(), print_value_ref(func)),
            ])
        ),
        AstOp::MapNew { key_ty, val_ty } => {
            format!("map.new<{}, {}> {{ }}", print_type(key_ty), print_type(val_ty))
        }
        AstOp::MapLen { map } => format!(
            "map.len {}",
            print_kv_args(vec![("map".to_string(), print_value_ref(map)),])
        ),
        AstOp::MapGet { map, key } => format!(
            "map.get {}",
            print_kv_args(vec![
                ("map".to_string(), print_value_ref(map)),
                ("key".to_string(), print_value_ref(key)),
            ])
        ),
        AstOp::MapGetRef { map, key } => format!(
            "map.get_ref {}",
            print_kv_args(vec![
                ("map".to_string(), print_value_ref(map)),
                ("key".to_string(), print_value_ref(key)),
            ])
        ),
        AstOp::MapDelete { map, key } => format!(
            "map.delete {}",
            print_kv_args(vec![
                ("map".to_string(), print_value_ref(map)),
                ("key".to_string(), print_value_ref(key)),
            ])
        ),
        AstOp::MapContainsKey { map, key } => format!(
            "map.contains_key {}",
            print_kv_args(vec![
                ("map".to_string(), print_value_ref(map)),
                ("key".to_string(), print_value_ref(key)),
            ])
        ),
        AstOp::MapKeys { map } => format!(
            "map.keys {}",
            print_kv_args(vec![("map".to_string(), print_value_ref(map)),])
        ),
        AstOp::MapValues { map } => format!(
            "map.values {}",
            print_kv_args(vec![("map".to_string(), print_value_ref(map)),])
        ),
        AstOp::StrConcat { a, b } => format!(
            "str.concat {}",
            print_kv_args(vec![
                ("a".to_string(), print_value_ref(a)),
                ("b".to_string(), print_value_ref(b)),
            ])
        ),
        AstOp::StrLen { s } => format!(
            "str.len {}",
            print_kv_args(vec![("s".to_string(), print_value_ref(s)),])
        ),
        AstOp::StrEq { a, b } => format!(
            "str.eq {}",
            print_kv_args(vec![
                ("a".to_string(), print_value_ref(a)),
                ("b".to_string(), print_value_ref(b)),
            ])
        ),
        AstOp::StrSlice { s, start, end } => format!(
            "str.slice {}",
            print_kv_args(vec![
                ("s".to_string(), print_value_ref(s)),
                ("start".to_string(), print_value_ref(start)),
                ("end".to_string(), print_value_ref(end)),
            ])
        ),
        AstOp::StrBytes { s } => format!(
            "str.bytes {}",
            print_kv_args(vec![("s".to_string(), print_value_ref(s)),])
        ),
        AstOp::StrBuilderNew => "str.builder.new { }".to_string(),
        AstOp::StrBuilderBuild { b } => format!(
            "str.builder.build {}",
            print_kv_args(vec![("b".to_string(), print_value_ref(b)),])
        ),
        AstOp::StrParseI64 { s } => format!(
            "str.parse_i64 {}",
            print_kv_args(vec![("s".to_string(), print_value_ref(s)),])
        ),
        AstOp::StrParseU64 { s } => format!(
            "str.parse_u64 {}",
            print_kv_args(vec![("s".to_string(), print_value_ref(s)),])
        ),
        AstOp::StrParseF64 { s } => format!(
            "str.parse_f64 {}",
            print_kv_args(vec![("s".to_string(), print_value_ref(s)),])
        ),
        AstOp::StrParseBool { s } => format!(
            "str.parse_bool {}",
            print_kv_args(vec![("s".to_string(), print_value_ref(s)),])
        ),
        AstOp::JsonEncode { ty, v } => format!(
            "json.encode<{}> {}",
            print_type(ty),
            print_kv_args(vec![("v".to_string(), print_value_ref(v)),])
        ),
        AstOp::JsonDecode { ty, s } => format!(
            "json.decode<{}> {}",
            print_type(ty),
            print_kv_args(vec![("s".to_string(), print_value_ref(s)),])
        ),
        AstOp::GpuThreadId { dim } => format!(
            "gpu.thread_id {}",
            print_kv_args(vec![("dim".to_string(), print_value_ref(dim)),])
        ),
        AstOp::GpuWorkgroupId { dim } => format!(
            "gpu.workgroup_id {}",
            print_kv_args(vec![("dim".to_string(), print_value_ref(dim)),])
        ),
        AstOp::GpuWorkgroupSize { dim } => format!(
            "gpu.workgroup_size {}",
            print_kv_args(vec![("dim".to_string(), print_value_ref(dim)),])
        ),
        AstOp::GpuGlobalId { dim } => format!(
            "gpu.global_id {}",
            print_kv_args(vec![("dim".to_string(), print_value_ref(dim)),])
        ),
        AstOp::GpuBufferLoad { ty, buf, idx } => format!(
            "gpu.buffer_load<{}> {}",
            print_type(ty),
            print_kv_args(vec![
                ("buf".to_string(), print_value_ref(buf)),
                ("idx".to_string(), print_value_ref(idx)),
            ])
        ),
        AstOp::GpuBufferLen { ty, buf } => format!(
            "gpu.buffer_len<{}> {}",
            print_type(ty),
            print_kv_args(vec![("buf".to_string(), print_value_ref(buf)),])
        ),
        AstOp::GpuShared { count, ty } => format!("gpu.shared<{}, {}>", count, print_type(ty)),
        AstOp::GpuLaunch {
            device,
            kernel,
            grid,
            block,
            args,
        } => format!(
            "gpu.launch {}",
            print_arg_kv_args(&[
                ("device".to_string(), AstArgValue::Value(device.clone())),
                ("kernel".to_string(), AstArgValue::FnRef(kernel.clone())),
                ("grid".to_string(), grid.clone()),
                ("block".to_string(), block.clone()),
                ("args".to_string(), args.clone()),
            ])
        ),
        AstOp::GpuLaunchAsync {
            device,
            kernel,
            grid,
            block,
            args,
        } => format!(
            "gpu.launch_async {}",
            print_arg_kv_args(&[
                ("device".to_string(), AstArgValue::Value(device.clone())),
                ("kernel".to_string(), AstArgValue::FnRef(kernel.clone())),
                ("grid".to_string(), grid.clone()),
                ("block".to_string(), block.clone()),
                ("args".to_string(), args.clone()),
            ])
        ),
    }
}

fn print_op_void_with_bb_map(op: &AstOpVoid, _bb_map: &HashMap<u32, u32>) -> String {
    match op {
        AstOpVoid::CallVoid { callee, targs, args } => format!(
            "call_void {}{} {}",
            callee,
            print_type_args(targs),
            print_arg_kv_args(args)
        ),
        AstOpVoid::CallVoidIndirect { callee, args } => format!(
            "call_void.indirect {} {}",
            print_value_ref(callee),
            print_arg_kv_args(args)
        ),
        AstOpVoid::SetField { obj, field, val } => format!(
            "setfield {}",
            print_kv_args(vec![
                ("obj".to_string(), print_value_ref(obj)),
                ("field".to_string(), field.clone()),
                ("val".to_string(), print_value_ref(val)),
            ])
        ),
        AstOpVoid::Panic { msg } => format!(
            "panic {}",
            print_kv_args(vec![("msg".to_string(), print_value_ref(msg)),])
        ),
        AstOpVoid::PtrStore { ty, p, v } => format!(
            "ptr.store<{}> {}",
            print_type(ty),
            print_kv_args(vec![
                ("p".to_string(), print_value_ref(p)),
                ("v".to_string(), print_value_ref(v)),
            ])
        ),
        AstOpVoid::ArrSet { arr, idx, val } => format!(
            "arr.set {}",
            print_kv_args(vec![
                ("arr".to_string(), print_value_ref(arr)),
                ("idx".to_string(), print_value_ref(idx)),
                ("val".to_string(), print_value_ref(val)),
            ])
        ),
        AstOpVoid::ArrPush { arr, val } => format!(
            "arr.push {}",
            print_kv_args(vec![
                ("arr".to_string(), print_value_ref(arr)),
                ("val".to_string(), print_value_ref(val)),
            ])
        ),
        AstOpVoid::ArrSort { arr } => format!(
            "arr.sort {}",
            print_kv_args(vec![("arr".to_string(), print_value_ref(arr)),])
        ),
        AstOpVoid::ArrForeach { arr, func } => format!(
            "arr.foreach {}",
            print_kv_args(vec![
                ("arr".to_string(), print_value_ref(arr)),
                ("fn".to_string(), print_value_ref(func)),
            ])
        ),
        AstOpVoid::MapSet { map, key, val } => format!(
            "map.set {}",
            print_kv_args(vec![
                ("map".to_string(), print_value_ref(map)),
                ("key".to_string(), print_value_ref(key)),
                ("val".to_string(), print_value_ref(val)),
            ])
        ),
        AstOpVoid::MapDeleteVoid { map, key } => format!(
            "map.delete_void {}",
            print_kv_args(vec![
                ("map".to_string(), print_value_ref(map)),
                ("key".to_string(), print_value_ref(key)),
            ])
        ),
        AstOpVoid::StrBuilderAppendStr { b, s } => format!(
            "str.builder.append_str {}",
            print_kv_args(vec![
                ("b".to_string(), print_value_ref(b)),
                ("s".to_string(), print_value_ref(s)),
            ])
        ),
        AstOpVoid::StrBuilderAppendI64 { b, v } => format!(
            "str.builder.append_i64 {}",
            print_kv_args(vec![
                ("b".to_string(), print_value_ref(b)),
                ("v".to_string(), print_value_ref(v)),
            ])
        ),
        AstOpVoid::StrBuilderAppendI32 { b, v } => format!(
            "str.builder.append_i32 {}",
            print_kv_args(vec![
                ("b".to_string(), print_value_ref(b)),
                ("v".to_string(), print_value_ref(v)),
            ])
        ),
        AstOpVoid::StrBuilderAppendF64 { b, v } => format!(
            "str.builder.append_f64 {}",
            print_kv_args(vec![
                ("b".to_string(), print_value_ref(b)),
                ("v".to_string(), print_value_ref(v)),
            ])
        ),
        AstOpVoid::StrBuilderAppendBool { b, v } => format!(
            "str.builder.append_bool {}",
            print_kv_args(vec![
                ("b".to_string(), print_value_ref(b)),
                ("v".to_string(), print_value_ref(v)),
            ])
        ),
        AstOpVoid::GpuBarrier => "gpu.barrier".to_string(),
        AstOpVoid::GpuBufferStore { ty, buf, idx, v } => format!(
            "gpu.buffer_store<{}> {}",
            print_type(ty),
            print_kv_args(vec![
                ("buf".to_string(), print_value_ref(buf)),
                ("idx".to_string(), print_value_ref(idx)),
                ("v".to_string(), print_value_ref(v)),
            ])
        ),
    }
}

fn print_terminator_with_bb_map(t: &AstTerminator, bb_map: &HashMap<u32, u32>) -> String {
    match t {
        AstTerminator::Ret(None) => "ret".to_string(),
        AstTerminator::Ret(Some(v)) => format!("ret {}", print_value_ref(v)),
        AstTerminator::Br(bb) => format!("br bb{}", remap_bb_label(*bb, bb_map)),
        AstTerminator::Cbr {
            cond,
            then_bb,
            else_bb,
        } => format!(
            "cbr {} bb{} bb{}",
            print_value_ref(cond),
            remap_bb_label(*then_bb, bb_map),
            remap_bb_label(*else_bb, bb_map)
        ),
        AstTerminator::Switch { val, arms, default } => {
            let mut sorted_arms = arms
                .iter()
                .map(|(lit, bb)| (print_const_lit(lit), remap_bb_label(*bb, bb_map)))
                .collect::<Vec<_>>();
            sorted_arms.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

            let arms_str = if sorted_arms.is_empty() {
                "{ }".to_string()
            } else {
                format!(
                    "{{ {} }}",
                    sorted_arms
                        .into_iter()
                        .map(|(lit, bb)| format!("case {} -> bb{}", lit, bb))
                        .collect::<Vec<_>>()
                        .join(" ")
                )
            };

            format!(
                "switch {} {} else bb{}",
                print_value_ref(val),
                arms_str,
                remap_bb_label(*default, bb_map)
            )
        }
        AstTerminator::Unreachable => "unreachable".to_string(),
    }
}

fn print_const_expr(c: &AstConstExpr) -> String {
    format!("const.{} {}", print_type(&c.ty), print_typed_const_lit(&c.lit))
}

fn print_typed_const_lit(lit: &AstConstLit) -> String {
    match lit {
        AstConstLit::Int(v) => v.to_string(),
        AstConstLit::Float(v) => canonical_float(*v),
        AstConstLit::Str(s) => print_string_literal(s),
        AstConstLit::Bool(v) => v.to_string(),
        AstConstLit::Unit => "unit".to_string(),
    }
}

fn print_const_lit(lit: &AstConstLit) -> String {
    match lit {
        AstConstLit::Int(v) => v.to_string(),
        AstConstLit::Float(v) => canonical_float(*v),
        AstConstLit::Str(s) => print_string_literal(s),
        AstConstLit::Bool(v) => v.to_string(),
        AstConstLit::Unit => "unit".to_string(),
    }
}

fn print_string_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                out.push_str("\\u{");
                out.push_str(&format!("{:x}", c as u32));
                out.push('}');
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn canonical_float(v: f64) -> String {
    let mut s = v.to_string();
    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
        s.push_str(".0");
    }
    s
}

fn print_type_args(targs: &[AstType]) -> String {
    if targs.is_empty() {
        String::new()
    } else {
        format!(
            "<{}>",
            targs.iter().map(print_type).collect::<Vec<_>>().join(", ")
        )
    }
}

fn print_type_params(params: &[AstTypeParam]) -> String {
    if params.is_empty() {
        String::new()
    } else {
        format!(
            "<{}>",
            params
                .iter()
                .map(|p| format!("{}: {}", p.name, p.constraint))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

fn print_builtin_type(b: &AstBuiltinType) -> String {
    match b {
        AstBuiltinType::Str => "Str".to_string(),
        AstBuiltinType::Array(t) => format!("Array<{}>", print_type(t)),
        AstBuiltinType::Map(k, v) => format!("Map<{}, {}>", print_type(k), print_type(v)),
        AstBuiltinType::TOption(t) => format!("TOption<{}>", print_type(t)),
        AstBuiltinType::TResult(a, b) => format!("TResult<{}, {}>", print_type(a), print_type(b)),
        AstBuiltinType::TStrBuilder => "TStrBuilder".to_string(),
        AstBuiltinType::TMutex(t) => format!("TMutex<{}>", print_type(t)),
        AstBuiltinType::TRwLock(t) => format!("TRwLock<{}>", print_type(t)),
        AstBuiltinType::TCell(t) => format!("TCell<{}>", print_type(t)),
        AstBuiltinType::TFuture(t) => format!("TFuture<{}>", print_type(t)),
        AstBuiltinType::TChannelSend(t) => format!("TChannelSend<{}>", print_type(t)),
        AstBuiltinType::TChannelRecv(t) => format!("TChannelRecv<{}>", print_type(t)),
    }
}

fn print_bin_op(kind: &BinOpKind) -> &'static str {
    match kind {
        BinOpKind::IAdd => "i.add",
        BinOpKind::ISub => "i.sub",
        BinOpKind::IMul => "i.mul",
        BinOpKind::ISDiv => "i.sdiv",
        BinOpKind::IUDiv => "i.udiv",
        BinOpKind::ISRem => "i.srem",
        BinOpKind::IURem => "i.urem",
        BinOpKind::IAddWrap => "i.add.wrap",
        BinOpKind::ISubWrap => "i.sub.wrap",
        BinOpKind::IMulWrap => "i.mul.wrap",
        BinOpKind::IAddChecked => "i.add.checked",
        BinOpKind::ISubChecked => "i.sub.checked",
        BinOpKind::IMulChecked => "i.mul.checked",
        BinOpKind::IAnd => "i.and",
        BinOpKind::IOr => "i.or",
        BinOpKind::IXor => "i.xor",
        BinOpKind::IShl => "i.shl",
        BinOpKind::ILshr => "i.lshr",
        BinOpKind::IAshr => "i.ashr",
        BinOpKind::FAdd => "f.add",
        BinOpKind::FSub => "f.sub",
        BinOpKind::FMul => "f.mul",
        BinOpKind::FDiv => "f.div",
        BinOpKind::FRem => "f.rem",
        BinOpKind::FAddFast => "f.add.fast",
        BinOpKind::FSubFast => "f.sub.fast",
        BinOpKind::FMulFast => "f.mul.fast",
        BinOpKind::FDivFast => "f.div.fast",
    }
}

fn print_kv_args(mut args: Vec<(String, String)>) -> String {
    args.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    if args.is_empty() {
        "{ }".to_string()
    } else {
        format!(
            "{{ {} }}",
            args.into_iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

fn print_arg_kv_args(args: &[(String, AstArgValue)]) -> String {
    let mapped = args
        .iter()
        .map(|(k, v)| (k.clone(), print_arg_value(v)))
        .collect::<Vec<_>>();
    print_kv_args(mapped)
}

fn print_arg_value(v: &AstArgValue) -> String {
    match v {
        AstArgValue::Value(v) => print_value_ref(v),
        AstArgValue::List(items) => format!(
            "[{}]",
            items
                .iter()
                .map(print_arg_list_elem)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        AstArgValue::FnRef(f) => f.clone(),
    }
}

fn print_arg_list_elem(v: &AstArgListElem) -> String {
    match v {
        AstArgListElem::Value(v) => print_value_ref(v),
        AstArgListElem::FnRef(f) => f.clone(),
    }
}

fn sorted_export_items(items: &[Spanned<ExportItem>]) -> Vec<String> {
    let mut out = items
        .iter()
        .map(|item| match &item.node {
            ExportItem::Fn(f) => f.clone(),
            ExportItem::Type(t) => t.clone(),
        })
        .collect::<Vec<_>>();
    out.sort();
    out
}

fn sorted_import_groups(groups: &[Spanned<ImportGroup>]) -> Vec<String> {
    let mut grouped: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for g in groups {
        let path = g.node.module_path.to_string();
        let entry = grouped.entry(path).or_default();
        for item in &g.node.items {
            match item {
                ImportItem::Fn(f) => entry.push(f.clone()),
                ImportItem::Type(t) => entry.push(t.clone()),
            }
        }
    }

    grouped
        .into_iter()
        .map(|(path, mut items)| {
            items.sort();
            items.dedup();
            format!("{}::{}", path, format_braced_list(items))
        })
        .collect()
}

fn format_braced_list(items: Vec<String>) -> String {
    if items.is_empty() {
        "{ }".to_string()
    } else {
        format!("{{ {} }}", items.join(", "))
    }
}

fn canonical_block_map(blocks: &[Spanned<AstBlock>]) -> HashMap<u32, u32> {
    let mut map = HashMap::new();
    for (idx, block) in blocks.iter().enumerate() {
        map.insert(block.node.label, idx as u32);
    }
    map
}

fn remap_bb_label(label: u32, bb_map: &HashMap<u32, u32>) -> u32 {
    bb_map.get(&label).copied().unwrap_or(label)
}

fn strip_digest_line(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut i = 0usize;

    while i < source.len() {
        let rest = &source[i..];
        let (line, next_i, had_newline) = if let Some(rel) = rest.find('\n') {
            (&rest[..rel], i + rel + 1, true)
        } else {
            (rest, source.len(), false)
        };

        if !line.trim_start().starts_with("digest ") {
            out.push_str(line);
            if had_newline {
                out.push('\n');
            }
        }

        i = next_i;
    }

    out
}

fn ensure_single_trailing_newline(mut s: String) -> String {
    while s.ends_with('\n') {
        s.pop();
    }
    s.push('\n');
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_span<T>(node: T) -> Spanned<T> {
        Spanned::new(node, Span::dummy())
    }

    #[test]
    fn digest_updates_and_stabilizes() {
        let src = "module a.b\nexports { @x }\nimports { }\ndigest \"old\"\n\nfn @x() -> i32 {\n  bb0:\n    ret const.i32 0\n}\n";
        let once = update_digest(src);
        let twice = update_digest(&once);
        assert_eq!(once, twice);
        assert!(once.contains("digest \""));
    }

    #[test]
    fn format_is_idempotent_and_renumbers_blocks() {
        let ast = AstFile {
            header: dummy_span(AstHeader {
                module_path: dummy_span(ModulePath {
                    segments: vec!["demo".to_string(), "main".to_string()],
                }),
                exports: vec![
                    dummy_span(ExportItem::Fn("@z".to_string())),
                    dummy_span(ExportItem::Fn("@a".to_string())),
                ],
                imports: vec![dummy_span(ImportGroup {
                    module_path: ModulePath {
                        segments: vec!["std".to_string(), "io".to_string()],
                    },
                    items: vec![ImportItem::Fn("@println".to_string())],
                })],
                digest: dummy_span(String::new()),
            }),
            decls: vec![dummy_span(AstDecl::Fn(AstFnDecl {
                name: "@main".to_string(),
                params: Vec::new(),
                ret_ty: dummy_span(AstType {
                    ownership: None,
                    base: AstBaseType::Prim("i32".to_string()),
                }),
                meta: None,
                blocks: vec![
                    dummy_span(AstBlock {
                        label: 10,
                        instrs: Vec::new(),
                        terminator: dummy_span(AstTerminator::Br(20)),
                    }),
                    dummy_span(AstBlock {
                        label: 20,
                        instrs: Vec::new(),
                        terminator: dummy_span(AstTerminator::Ret(Some(AstValueRef::Const(
                            AstConstExpr {
                                ty: AstType {
                                    ownership: None,
                                    base: AstBaseType::Prim("i32".to_string()),
                                },
                                lit: AstConstLit::Int(0),
                            },
                        )))),
                    }),
                ],
                doc: None,
            }))],
        };

        let sm = SourceMap::new();
        let once = format_csnf(&ast, &sm);
        let twice = update_digest(&once);
        assert_eq!(once, twice);
        assert!(once.contains("  bb0:"));
        assert!(once.contains("  bb1:"));
        assert!(once.contains("    br bb1"));
    }
}
