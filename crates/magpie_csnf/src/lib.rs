//! Canonical Source Normal Form (CSNF) formatting and digest helpers.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use magpie_ast::{
    AstArgListElem, AstArgValue, AstBaseType, AstBlock, AstBuiltinType, AstConstExpr, AstConstLit,
    AstDecl, AstEnumDecl, AstExternModule, AstFieldDecl, AstFile, AstFnDecl, AstFnMeta,
    AstGlobalDecl, AstGpuFnDecl, AstImplDecl, AstInstr, AstOp, AstOpVoid, AstParam, AstSigDecl,
    AstStructDecl, AstTerminator, AstType, AstTypeParam, AstValueRef, BinOpKind, CmpKind,
    ExportItem, ImportGroup, ImportItem, OwnershipMod, Spanned,
};

const BLOCK_INDENT: &str = "  ";
const INSTR_INDENT: &str = "    ";

/// Prints AST back as canonical `.mp` source.
pub fn format_csnf(ast: &AstFile) -> String {
    let mut out = String::new();
    let header = &ast.header.node;

    out.push_str("module ");
    out.push_str(&header.module_path.node.to_string());
    out.push('\n');

    out.push_str("exports ");
    out.push_str(&format_exports(&header.exports));
    out.push('\n');

    out.push_str("imports ");
    out.push_str(&format_imports(&header.imports));
    out.push('\n');

    out.push_str("digest ");
    out.push_str(&quote_string(&header.digest.node));
    out.push('\n');

    if !ast.decls.is_empty() {
        out.push('\n');
    }

    for (idx, decl) in ast.decls.iter().enumerate() {
        out.push_str(&print_decl(&decl.node));
        if idx + 1 != ast.decls.len() {
            out.push_str("\n\n");
        }
    }

    ensure_single_trailing_newline(&out)
}

/// BLAKE3 hash (hex) of source minus digest line.
pub fn compute_digest(canonical_source: &str) -> String {
    let stripped = strip_digest_lines(canonical_source);
    blake3::hash(stripped.as_bytes()).to_hex().to_string()
}

/// Replace digest line with the correct one.
pub fn update_digest(source: &str) -> String {
    let normalized = normalize_newlines(source);
    let mut lines: Vec<String> = normalized.lines().map(ToOwned::to_owned).collect();

    if let Some(idx) = lines
        .iter()
        .position(|line| line.trim_start().starts_with("digest "))
    {
        lines[idx] = "digest \"\"".to_string();
    } else {
        let insert_idx = lines
            .iter()
            .position(|line| line.starts_with("imports "))
            .map(|idx| idx + 1)
            .unwrap_or(3.min(lines.len()));
        lines.insert(insert_idx, "digest \"\"".to_string());
    }

    let interim = ensure_single_trailing_newline(&lines.join("\n"));
    let digest = compute_digest(&interim);
    let digest_line = format!("digest \"{}\"", digest);

    let final_lines = interim
        .lines()
        .map(|line| {
            if line.trim_start().starts_with("digest ") {
                digest_line.clone()
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>();

    ensure_single_trailing_newline(&final_lines.join("\n"))
}

/// Canonical type printing helper.
pub fn print_type(ty: &AstType) -> String {
    let mut out = String::new();

    if let Some(ownership) = &ty.ownership {
        out.push_str(match ownership {
            OwnershipMod::Shared => "shared ",
            OwnershipMod::Borrow => "borrow ",
            OwnershipMod::MutBorrow => "mutborrow ",
            OwnershipMod::Weak => "weak ",
        });
    }

    out.push_str(&print_base_type(&ty.base));
    out
}

/// Canonical value-ref printing helper.
pub fn print_value_ref(v: &AstValueRef) -> String {
    match v {
        AstValueRef::Local(name) => format!("%{}", name),
        AstValueRef::Const(c) => print_const_expr(c),
    }
}

/// Canonical value-producing op printing helper.
pub fn print_op(op: &AstOp) -> String {
    print_op_with_label_map(op, &HashMap::new())
}

/// Canonical void op printing helper.
pub fn print_op_void(op: &AstOpVoid) -> String {
    print_op_void_with_label_map(op, &HashMap::new())
}

/// Canonical terminator printing helper.
pub fn print_terminator(term: &AstTerminator) -> String {
    print_terminator_with_label_map(term, &HashMap::new())
}

/// Canonical declaration printing helper.
pub fn print_decl(decl: &AstDecl) -> String {
    match decl {
        AstDecl::Fn(f) => print_fn_decl("fn", f, None),
        AstDecl::AsyncFn(f) => print_fn_decl("async fn", f, None),
        AstDecl::UnsafeFn(f) => print_fn_decl("unsafe fn", f, None),
        AstDecl::GpuFn(g) => print_gpu_fn_decl(g),
        AstDecl::HeapStruct(s) => print_struct_decl("heap", s),
        AstDecl::ValueStruct(s) => print_struct_decl("value", s),
        AstDecl::HeapEnum(e) => print_enum_decl("heap", e),
        AstDecl::ValueEnum(e) => print_enum_decl("value", e),
        AstDecl::Extern(extern_mod) => print_extern_decl(extern_mod),
        AstDecl::Global(g) => print_global_decl(g),
        AstDecl::Impl(i) => print_impl_decl(i),
        AstDecl::Sig(s) => print_sig_decl(s),
    }
}

fn format_exports(exports: &[Spanned<ExportItem>]) -> String {
    let mut items = BTreeSet::new();
    for item in exports {
        match &item.node {
            ExportItem::Fn(name) | ExportItem::Type(name) => {
                items.insert(name.clone());
            }
        }
    }

    if items.is_empty() {
        return "{ }".to_string();
    }

    format!("{{ {} }}", join_comma(items))
}

fn format_imports(imports: &[Spanned<ImportGroup>]) -> String {
    let mut grouped: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    for group in imports {
        let module = group.node.module_path.to_string();
        let items = grouped.entry(module).or_default();
        for item in &group.node.items {
            match item {
                ImportItem::Fn(name) | ImportItem::Type(name) => {
                    items.insert(name.clone());
                }
            }
        }
    }

    if grouped.is_empty() {
        return "{ }".to_string();
    }

    let groups = grouped
        .into_iter()
        .map(|(module, items)| format!("{}::{{{}}}", module, join_comma(items)));

    format!("{{ {} }}", join_comma(groups))
}

fn print_fn_decl(prefix: &str, f: &AstFnDecl, gpu_target: Option<&str>) -> String {
    let mut out = String::new();
    push_doc(&mut out, f.doc.as_deref());

    out.push_str(prefix);
    out.push(' ');
    out.push_str(&f.name);
    out.push('(');
    out.push_str(&join_comma(f.params.iter().map(print_param)));
    out.push_str(") -> ");
    out.push_str(&print_type(&f.ret_ty.node));

    if let Some(target) = gpu_target {
        out.push_str(" target(");
        out.push_str(target);
        out.push(')');
    }

    if let Some(meta) = &f.meta {
        out.push(' ');
        out.push_str(&print_meta(meta));
    }

    out.push_str(" {\n");

    let blocks = f.blocks.iter().map(|b| &b.node).collect::<Vec<_>>();
    out.push_str(&print_blocks(&blocks));

    out.push_str("\n}");
    out
}

fn print_gpu_fn_decl(gpu: &AstGpuFnDecl) -> String {
    print_fn_decl("gpu fn", &gpu.inner, Some(&gpu.target))
}

fn print_blocks(blocks: &[&AstBlock]) -> String {
    let label_map = build_block_label_map(blocks);
    let mut out = String::new();

    for (idx, block) in blocks.iter().enumerate() {
        out.push_str(BLOCK_INDENT);
        out.push_str(&format!("bb{}:\n", remap_bb(block.label, &label_map)));

        for instr in &block.instrs {
            out.push_str(&print_instr(&instr.node, &label_map, INSTR_INDENT));
            out.push('\n');
        }

        out.push_str(INSTR_INDENT);
        out.push_str(&print_terminator_with_label_map(
            &block.terminator.node,
            &label_map,
        ));

        if idx + 1 != blocks.len() {
            out.push_str("\n\n");
        }
    }

    out
}

fn build_block_label_map(blocks: &[&AstBlock]) -> HashMap<u32, u32> {
    let mut map = HashMap::new();
    for (idx, block) in blocks.iter().enumerate() {
        map.insert(block.label, idx as u32);
    }
    map
}

fn print_instr(instr: &AstInstr, label_map: &HashMap<u32, u32>, indent: &str) -> String {
    match instr {
        AstInstr::Assign { name, ty, op } => format!(
            "{}%{}: {} = {}",
            indent,
            name,
            print_type(&ty.node),
            print_op_with_label_map(op, label_map)
        ),
        AstInstr::Void(op) => format!("{}{}", indent, print_op_void_with_label_map(op, label_map)),
        AstInstr::UnsafeBlock(instrs) => {
            let mut out = String::new();
            out.push_str(indent);
            out.push_str("unsafe {\n");

            for inner in instrs {
                out.push_str(&print_instr(&inner.node, label_map, "      "));
                out.push('\n');
            }

            out.push_str(indent);
            out.push('}');
            out
        }
    }
}

fn print_terminator_with_label_map(term: &AstTerminator, label_map: &HashMap<u32, u32>) -> String {
    match term {
        AstTerminator::Ret(Some(v)) => format!("ret {}", print_value_ref(v)),
        AstTerminator::Ret(None) => "ret".to_string(),
        AstTerminator::Br(bb) => format!("br bb{}", remap_bb(*bb, label_map)),
        AstTerminator::Cbr {
            cond,
            then_bb,
            else_bb,
        } => format!(
            "cbr {} bb{} bb{}",
            print_value_ref(cond),
            remap_bb(*then_bb, label_map),
            remap_bb(*else_bb, label_map)
        ),
        AstTerminator::Switch { val, arms, default } => {
            let arm_text = if arms.is_empty() {
                "".to_string()
            } else {
                format!(
                    " {}",
                    arms.iter()
                        .map(|(lit, bb)| format!(
                            "case {} -> bb{}",
                            print_const_lit(lit),
                            remap_bb(*bb, label_map)
                        ))
                        .collect::<Vec<_>>()
                        .join(" ")
                )
            };

            format!(
                "switch {} {{{} }} else bb{}",
                print_value_ref(val),
                arm_text,
                remap_bb(*default, label_map)
            )
        }
        AstTerminator::Unreachable => "unreachable".to_string(),
    }
}

fn print_op_with_label_map(op: &AstOp, label_map: &HashMap<u32, u32>) -> String {
    match op {
        AstOp::Const(c) => print_const_expr(c),
        AstOp::BinOp { kind, lhs, rhs } => print_op_with_pairs(
            bin_op_name(kind),
            vec![("lhs", print_value_ref(lhs)), ("rhs", print_value_ref(rhs))],
        ),
        AstOp::Cmp {
            kind,
            pred,
            lhs,
            rhs,
        } => {
            let op_name = match kind {
                CmpKind::ICmp => format!("icmp.{}", pred),
                CmpKind::FCmp => format!("fcmp.{}", pred),
            };
            print_op_with_pairs(
                &op_name,
                vec![("lhs", print_value_ref(lhs)), ("rhs", print_value_ref(rhs))],
            )
        }
        AstOp::Call {
            callee,
            targs,
            args,
        } => {
            format!(
                "call {}{} {}",
                callee,
                print_type_args(targs),
                print_arg_pairs(args)
            )
        }
        AstOp::CallIndirect { callee, args } => {
            format!(
                "call.indirect {} {}",
                print_value_ref(callee),
                print_arg_pairs(args)
            )
        }
        AstOp::Try {
            callee,
            targs,
            args,
        } => {
            format!(
                "try {}{} {}",
                callee,
                print_type_args(targs),
                print_arg_pairs(args)
            )
        }
        AstOp::SuspendCall {
            callee,
            targs,
            args,
        } => format!(
            "suspend.call {}{} {}",
            callee,
            print_type_args(targs),
            print_arg_pairs(args)
        ),
        AstOp::SuspendAwait { fut } => {
            print_op_with_pairs("suspend.await", vec![("fut", print_value_ref(fut))])
        }
        AstOp::New { ty, fields } => {
            format!("new {} {}", print_type(ty), print_value_pairs(fields))
        }
        AstOp::GetField { obj, field } => print_op_with_pairs(
            "getfield",
            vec![("obj", print_value_ref(obj)), ("field", field.clone())],
        ),
        AstOp::Phi { ty, incomings } => {
            let mut incoming = incomings
                .iter()
                .map(|(bb, v)| (remap_bb(*bb, label_map), print_value_ref(v)))
                .collect::<Vec<_>>();
            incoming.sort_by(|a, b| a.0.cmp(&b.0));

            let body = if incoming.is_empty() {
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
            };
            format!("phi {} {}", print_type(ty), body)
        }
        AstOp::EnumNew { variant, args } => {
            format!("enum.new<{}> {}", variant, print_value_pairs(args))
        }
        AstOp::EnumTag { v } => print_op_with_pairs("enum.tag", vec![("v", print_value_ref(v))]),
        AstOp::EnumPayload { variant, v } => format!(
            "enum.payload<{}> {}",
            variant,
            format_pairs(vec![("v", print_value_ref(v))])
        ),
        AstOp::EnumIs { variant, v } => {
            format!(
                "enum.is<{}> {}",
                variant,
                format_pairs(vec![("v", print_value_ref(v))])
            )
        }
        AstOp::Share { v } => print_op_with_pairs("share", vec![("v", print_value_ref(v))]),
        AstOp::CloneShared { v } => {
            print_op_with_pairs("clone.shared", vec![("v", print_value_ref(v))])
        }
        AstOp::CloneWeak { v } => {
            print_op_with_pairs("clone.weak", vec![("v", print_value_ref(v))])
        }
        AstOp::WeakDowngrade { v } => {
            print_op_with_pairs("weak.downgrade", vec![("v", print_value_ref(v))])
        }
        AstOp::WeakUpgrade { v } => {
            print_op_with_pairs("weak.upgrade", vec![("v", print_value_ref(v))])
        }
        AstOp::Cast { from, to, v } => format!(
            "cast<{}, {}> {}",
            print_type(from),
            print_type(to),
            format_pairs(vec![("v", print_value_ref(v))])
        ),
        AstOp::BorrowShared { v } => {
            print_op_with_pairs("borrow.shared", vec![("v", print_value_ref(v))])
        }
        AstOp::BorrowMut { v } => {
            print_op_with_pairs("borrow.mut", vec![("v", print_value_ref(v))])
        }
        AstOp::PtrNull { ty } => format!("ptr.null<{}>", print_type(ty)),
        AstOp::PtrAddr { ty, p } => format!(
            "ptr.addr<{}> {}",
            print_type(ty),
            format_pairs(vec![("p", print_value_ref(p))])
        ),
        AstOp::PtrFromAddr { ty, addr } => format!(
            "ptr.from_addr<{}> {}",
            print_type(ty),
            format_pairs(vec![("addr", print_value_ref(addr))])
        ),
        AstOp::PtrAdd { ty, p, count } => format!(
            "ptr.add<{}> {}",
            print_type(ty),
            format_pairs(vec![
                ("p", print_value_ref(p)),
                ("count", print_value_ref(count))
            ])
        ),
        AstOp::PtrLoad { ty, p } => format!(
            "ptr.load<{}> {}",
            print_type(ty),
            format_pairs(vec![("p", print_value_ref(p))])
        ),
        AstOp::CallableCapture { fn_ref, captures } => {
            format!(
                "callable.capture {} {}",
                fn_ref,
                print_value_pairs(captures)
            )
        }
        AstOp::ArrNew { elem_ty, cap } => format!(
            "arr.new<{}> {}",
            print_type(elem_ty),
            format_pairs(vec![("cap", print_value_ref(cap))])
        ),
        AstOp::ArrLen { arr } => {
            print_op_with_pairs("arr.len", vec![("arr", print_value_ref(arr))])
        }
        AstOp::ArrGet { arr, idx } => print_op_with_pairs(
            "arr.get",
            vec![("arr", print_value_ref(arr)), ("idx", print_value_ref(idx))],
        ),
        AstOp::ArrPop { arr } => {
            print_op_with_pairs("arr.pop", vec![("arr", print_value_ref(arr))])
        }
        AstOp::ArrSlice { arr, start, end } => print_op_with_pairs(
            "arr.slice",
            vec![
                ("arr", print_value_ref(arr)),
                ("start", print_value_ref(start)),
                ("end", print_value_ref(end)),
            ],
        ),
        AstOp::ArrContains { arr, val } => print_op_with_pairs(
            "arr.contains",
            vec![("arr", print_value_ref(arr)), ("val", print_value_ref(val))],
        ),
        AstOp::ArrMap { arr, func } => print_op_with_pairs(
            "arr.map",
            vec![("arr", print_value_ref(arr)), ("fn", print_value_ref(func))],
        ),
        AstOp::ArrFilter { arr, func } => print_op_with_pairs(
            "arr.filter",
            vec![("arr", print_value_ref(arr)), ("fn", print_value_ref(func))],
        ),
        AstOp::ArrReduce { arr, init, func } => print_op_with_pairs(
            "arr.reduce",
            vec![
                ("arr", print_value_ref(arr)),
                ("init", print_value_ref(init)),
                ("fn", print_value_ref(func)),
            ],
        ),
        AstOp::MapNew { key_ty, val_ty } => {
            format!(
                "map.new<{}, {}> {{ }}",
                print_type(key_ty),
                print_type(val_ty)
            )
        }
        AstOp::MapLen { map } => {
            print_op_with_pairs("map.len", vec![("map", print_value_ref(map))])
        }
        AstOp::MapGet { map, key } => print_op_with_pairs(
            "map.get",
            vec![("map", print_value_ref(map)), ("key", print_value_ref(key))],
        ),
        AstOp::MapGetRef { map, key } => print_op_with_pairs(
            "map.get_ref",
            vec![("map", print_value_ref(map)), ("key", print_value_ref(key))],
        ),
        AstOp::MapDelete { map, key } => print_op_with_pairs(
            "map.delete",
            vec![("map", print_value_ref(map)), ("key", print_value_ref(key))],
        ),
        AstOp::MapContainsKey { map, key } => print_op_with_pairs(
            "map.contains_key",
            vec![("map", print_value_ref(map)), ("key", print_value_ref(key))],
        ),
        AstOp::MapKeys { map } => {
            print_op_with_pairs("map.keys", vec![("map", print_value_ref(map))])
        }
        AstOp::MapValues { map } => {
            print_op_with_pairs("map.values", vec![("map", print_value_ref(map))])
        }
        AstOp::StrConcat { a, b } => print_op_with_pairs(
            "str.concat",
            vec![("a", print_value_ref(a)), ("b", print_value_ref(b))],
        ),
        AstOp::StrLen { s } => print_op_with_pairs("str.len", vec![("s", print_value_ref(s))]),
        AstOp::StrEq { a, b } => print_op_with_pairs(
            "str.eq",
            vec![("a", print_value_ref(a)), ("b", print_value_ref(b))],
        ),
        AstOp::StrSlice { s, start, end } => print_op_with_pairs(
            "str.slice",
            vec![
                ("s", print_value_ref(s)),
                ("start", print_value_ref(start)),
                ("end", print_value_ref(end)),
            ],
        ),
        AstOp::StrBytes { s } => print_op_with_pairs("str.bytes", vec![("s", print_value_ref(s))]),
        AstOp::StrBuilderNew => "str.builder.new { }".to_string(),
        AstOp::StrBuilderBuild { b } => {
            print_op_with_pairs("str.builder.build", vec![("b", print_value_ref(b))])
        }
        AstOp::StrParseI64 { s } => {
            print_op_with_pairs("str.parse_i64", vec![("s", print_value_ref(s))])
        }
        AstOp::StrParseU64 { s } => {
            print_op_with_pairs("str.parse_u64", vec![("s", print_value_ref(s))])
        }
        AstOp::StrParseF64 { s } => {
            print_op_with_pairs("str.parse_f64", vec![("s", print_value_ref(s))])
        }
        AstOp::StrParseBool { s } => {
            print_op_with_pairs("str.parse_bool", vec![("s", print_value_ref(s))])
        }
        AstOp::JsonEncode { ty, v } => format!(
            "json.encode<{}> {}",
            print_type(ty),
            format_pairs(vec![("v", print_value_ref(v))])
        ),
        AstOp::JsonDecode { ty, s } => format!(
            "json.decode<{}> {}",
            print_type(ty),
            format_pairs(vec![("s", print_value_ref(s))])
        ),
        AstOp::GpuThreadId { dim } => {
            print_op_with_pairs("gpu.thread_id", vec![("dim", print_value_ref(dim))])
        }
        AstOp::GpuWorkgroupId { dim } => {
            print_op_with_pairs("gpu.workgroup_id", vec![("dim", print_value_ref(dim))])
        }
        AstOp::GpuWorkgroupSize { dim } => {
            print_op_with_pairs("gpu.workgroup_size", vec![("dim", print_value_ref(dim))])
        }
        AstOp::GpuGlobalId { dim } => {
            print_op_with_pairs("gpu.global_id", vec![("dim", print_value_ref(dim))])
        }
        AstOp::GpuBufferLoad { ty, buf, idx } => format!(
            "gpu.buffer_load<{}> {}",
            print_type(ty),
            format_pairs(vec![
                ("buf", print_value_ref(buf)),
                ("idx", print_value_ref(idx))
            ])
        ),
        AstOp::GpuBufferLen { ty, buf } => format!(
            "gpu.buffer_len<{}> {}",
            print_type(ty),
            format_pairs(vec![("buf", print_value_ref(buf))])
        ),
        AstOp::GpuShared { count, ty } => format!("gpu.shared<{}, {}>", count, print_type(ty)),
        AstOp::GpuLaunch {
            device,
            kernel,
            grid,
            block,
            args,
        } => print_op_with_pairs(
            "gpu.launch",
            vec![
                ("device", print_value_ref(device)),
                ("kernel", kernel.clone()),
                ("grid", print_arg_value(grid)),
                ("block", print_arg_value(block)),
                ("args", print_arg_value(args)),
            ],
        ),
        AstOp::GpuLaunchAsync {
            device,
            kernel,
            grid,
            block,
            args,
        } => print_op_with_pairs(
            "gpu.launch_async",
            vec![
                ("device", print_value_ref(device)),
                ("kernel", kernel.clone()),
                ("grid", print_arg_value(grid)),
                ("block", print_arg_value(block)),
                ("args", print_arg_value(args)),
            ],
        ),
    }
}

fn print_op_void_with_label_map(op: &AstOpVoid, _label_map: &HashMap<u32, u32>) -> String {
    match op {
        AstOpVoid::CallVoid {
            callee,
            targs,
            args,
        } => format!(
            "call_void {}{} {}",
            callee,
            print_type_args(targs),
            print_arg_pairs(args)
        ),
        AstOpVoid::CallVoidIndirect { callee, args } => format!(
            "call_void.indirect {} {}",
            print_value_ref(callee),
            print_arg_pairs(args)
        ),
        AstOpVoid::SetField { obj, field, val } => print_op_with_pairs(
            "setfield",
            vec![
                ("obj", print_value_ref(obj)),
                ("field", field.clone()),
                ("val", print_value_ref(val)),
            ],
        ),
        AstOpVoid::Panic { msg } => {
            print_op_with_pairs("panic", vec![("msg", print_value_ref(msg))])
        }
        AstOpVoid::PtrStore { ty, p, v } => format!(
            "ptr.store<{}> {}",
            print_type(ty),
            format_pairs(vec![("p", print_value_ref(p)), ("v", print_value_ref(v))])
        ),
        AstOpVoid::ArrSet { arr, idx, val } => print_op_with_pairs(
            "arr.set",
            vec![
                ("arr", print_value_ref(arr)),
                ("idx", print_value_ref(idx)),
                ("val", print_value_ref(val)),
            ],
        ),
        AstOpVoid::ArrPush { arr, val } => print_op_with_pairs(
            "arr.push",
            vec![("arr", print_value_ref(arr)), ("val", print_value_ref(val))],
        ),
        AstOpVoid::ArrSort { arr } => {
            print_op_with_pairs("arr.sort", vec![("arr", print_value_ref(arr))])
        }
        AstOpVoid::ArrForeach { arr, func } => print_op_with_pairs(
            "arr.foreach",
            vec![("arr", print_value_ref(arr)), ("fn", print_value_ref(func))],
        ),
        AstOpVoid::MapSet { map, key, val } => print_op_with_pairs(
            "map.set",
            vec![
                ("map", print_value_ref(map)),
                ("key", print_value_ref(key)),
                ("val", print_value_ref(val)),
            ],
        ),
        AstOpVoid::MapDeleteVoid { map, key } => print_op_with_pairs(
            "map.delete_void",
            vec![("map", print_value_ref(map)), ("key", print_value_ref(key))],
        ),
        AstOpVoid::StrBuilderAppendStr { b, s } => print_op_with_pairs(
            "str.builder.append_str",
            vec![("b", print_value_ref(b)), ("s", print_value_ref(s))],
        ),
        AstOpVoid::StrBuilderAppendI64 { b, v } => print_op_with_pairs(
            "str.builder.append_i64",
            vec![("b", print_value_ref(b)), ("v", print_value_ref(v))],
        ),
        AstOpVoid::StrBuilderAppendI32 { b, v } => print_op_with_pairs(
            "str.builder.append_i32",
            vec![("b", print_value_ref(b)), ("v", print_value_ref(v))],
        ),
        AstOpVoid::StrBuilderAppendF64 { b, v } => print_op_with_pairs(
            "str.builder.append_f64",
            vec![("b", print_value_ref(b)), ("v", print_value_ref(v))],
        ),
        AstOpVoid::StrBuilderAppendBool { b, v } => print_op_with_pairs(
            "str.builder.append_bool",
            vec![("b", print_value_ref(b)), ("v", print_value_ref(v))],
        ),
        AstOpVoid::GpuBarrier => "gpu.barrier".to_string(),
        AstOpVoid::GpuBufferStore { ty, buf, idx, v } => format!(
            "gpu.buffer_store<{}> {}",
            print_type(ty),
            format_pairs(vec![
                ("buf", print_value_ref(buf)),
                ("idx", print_value_ref(idx)),
                ("v", print_value_ref(v)),
            ])
        ),
    }
}

fn print_base_type(base: &AstBaseType) -> String {
    match base {
        AstBaseType::Prim(name) => name.clone(),
        AstBaseType::Named { path, name, targs } => {
            let mut out = String::new();
            if let Some(path) = path {
                out.push_str(&path.to_string());
                out.push('.');
            }
            out.push_str(name);
            out.push_str(&print_type_args(targs));
            out
        }
        AstBaseType::Builtin(builtin) => print_builtin_type(builtin),
        AstBaseType::Callable { sig_ref } => format!("TCallable<{}>", sig_ref),
        AstBaseType::RawPtr(inner) => format!("rawptr<{}>", print_type(inner)),
    }
}

fn print_builtin_type(builtin: &AstBuiltinType) -> String {
    match builtin {
        AstBuiltinType::Str => "Str".to_string(),
        AstBuiltinType::Array(elem) => format!("Array<{}>", print_type(elem)),
        AstBuiltinType::Map(k, v) => format!("Map<{}, {}>", print_type(k), print_type(v)),
        AstBuiltinType::TOption(t) => format!("TOption<{}>", print_type(t)),
        AstBuiltinType::TResult(ok, err) => {
            format!("TResult<{}, {}>", print_type(ok), print_type(err))
        }
        AstBuiltinType::TStrBuilder => "TStrBuilder".to_string(),
        AstBuiltinType::TMutex(t) => format!("TMutex<{}>", print_type(t)),
        AstBuiltinType::TRwLock(t) => format!("TRwLock<{}>", print_type(t)),
        AstBuiltinType::TCell(t) => format!("TCell<{}>", print_type(t)),
        AstBuiltinType::TFuture(t) => format!("TFuture<{}>", print_type(t)),
        AstBuiltinType::TChannelSend(t) => format!("TChannelSend<{}>", print_type(t)),
        AstBuiltinType::TChannelRecv(t) => format!("TChannelRecv<{}>", print_type(t)),
    }
}

fn print_const_expr(c: &AstConstExpr) -> String {
    format!("const.{} {}", print_type(&c.ty), print_const_lit(&c.lit))
}

fn print_const_lit(lit: &AstConstLit) -> String {
    match lit {
        AstConstLit::Int(v) => v.to_string(),
        AstConstLit::Float(v) => canonical_float(*v),
        AstConstLit::Str(s) => quote_string(s),
        AstConstLit::Bool(v) => v.to_string(),
        AstConstLit::Unit => "unit".to_string(),
    }
}

fn print_struct_decl(kind: &str, s: &AstStructDecl) -> String {
    let mut out = String::new();
    push_doc(&mut out, s.doc.as_deref());

    out.push_str(kind);
    out.push_str(" struct ");
    out.push_str(&s.name);
    out.push_str(&print_type_params(&s.type_params));
    out.push_str(" {\n");

    for field in &s.fields {
        out.push_str(BLOCK_INDENT);
        out.push_str(&print_field_decl(field));
        out.push('\n');
    }

    out.push('}');
    out
}

fn print_enum_decl(kind: &str, e: &AstEnumDecl) -> String {
    let mut out = String::new();
    push_doc(&mut out, e.doc.as_deref());

    out.push_str(kind);
    out.push_str(" enum ");
    out.push_str(&e.name);
    out.push_str(&print_type_params(&e.type_params));
    out.push_str(" {\n");

    for variant in &e.variants {
        out.push_str(BLOCK_INDENT);
        out.push_str("variant ");
        out.push_str(&variant.name);
        if variant.fields.is_empty() {
            out.push_str(" { }");
        } else {
            out.push_str(" { ");
            out.push_str(&join_comma(variant.fields.iter().map(print_field_decl)));
            out.push_str(" }");
        }
        out.push('\n');
    }

    out.push('}');
    out
}

fn print_extern_decl(extern_mod: &AstExternModule) -> String {
    let mut out = String::new();
    push_doc(&mut out, extern_mod.doc.as_deref());

    out.push_str("extern ");
    out.push_str(&quote_string(&extern_mod.abi));
    out.push_str(" module ");
    out.push_str(&extern_mod.name);
    out.push_str(" {\n");

    for item in &extern_mod.items {
        out.push_str(BLOCK_INDENT);
        out.push_str("fn ");
        out.push_str(&item.name);
        out.push('(');
        out.push_str(&join_comma(item.params.iter().map(print_param)));
        out.push_str(") -> ");
        out.push_str(&print_type(&item.ret_ty.node));

        if !item.attrs.is_empty() {
            let attrs = item
                .attrs
                .iter()
                .map(|(k, v)| (k.as_str(), quote_string(v)))
                .collect::<Vec<_>>();
            out.push(' ');
            out.push_str("attrs ");
            out.push_str(&format_pairs(attrs));
        }

        out.push('\n');
    }

    out.push('}');
    out
}

fn print_global_decl(g: &AstGlobalDecl) -> String {
    let mut out = String::new();
    push_doc(&mut out, g.doc.as_deref());

    out.push_str("global ");
    out.push_str(&g.name);
    out.push_str(": ");
    out.push_str(&print_type(&g.ty.node));
    out.push_str(" = ");
    out.push_str(&print_const_expr(&g.init));
    out
}

fn print_impl_decl(i: &AstImplDecl) -> String {
    format!(
        "impl {} for {} = {}",
        i.trait_name,
        print_type(&i.for_type),
        i.fn_ref
    )
}

fn print_sig_decl(s: &AstSigDecl) -> String {
    format!(
        "sig {}({}) -> {}",
        s.name,
        join_comma(s.param_types.iter().map(print_type)),
        print_type(&s.ret_ty)
    )
}

fn print_meta(meta: &AstFnMeta) -> String {
    let mut parts = Vec::new();

    if !meta.uses.is_empty() {
        let mut uses = meta.uses.clone();
        uses.sort();
        uses.dedup();
        parts.push(format!("uses {{ {} }}", join_comma(uses)));
    }

    if !meta.effects.is_empty() {
        let mut effects = meta.effects.clone();
        effects.sort();
        effects.dedup();
        parts.push(format!("effects {{ {} }}", join_comma(effects)));
    }

    if !meta.cost.is_empty() {
        let mut cost = meta.cost.clone();
        cost.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        cost.dedup_by(|a, b| a.0 == b.0);
        let cost_text = cost
            .into_iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(", ");
        parts.push(format!("cost {{ {} }}", cost_text));
    }

    if parts.is_empty() {
        "meta { }".to_string()
    } else {
        format!("meta {{ {} }}", parts.join(" "))
    }
}

fn print_param(param: &AstParam) -> String {
    format!("%{}: {}", param.name, print_type(&param.ty.node))
}

fn print_field_decl(field: &AstFieldDecl) -> String {
    format!("field {}: {}", field.name, print_type(&field.ty.node))
}

fn print_type_params(params: &[AstTypeParam]) -> String {
    if params.is_empty() {
        return String::new();
    }

    let body = params
        .iter()
        .map(|p| format!("{}: {}", p.name, p.constraint))
        .collect::<Vec<_>>()
        .join(", ");
    format!("<{}>", body)
}

fn print_type_args(types: &[AstType]) -> String {
    if types.is_empty() {
        return String::new();
    }

    format!("<{}>", join_comma(types.iter().map(print_type)))
}

fn print_arg_pairs(args: &[(String, AstArgValue)]) -> String {
    let pairs = args
        .iter()
        .map(|(k, v)| (k.as_str(), print_arg_value(v)))
        .collect::<Vec<_>>();
    format_pairs(pairs)
}

fn print_value_pairs(args: &[(String, AstValueRef)]) -> String {
    let pairs = args
        .iter()
        .map(|(k, v)| (k.as_str(), print_value_ref(v)))
        .collect::<Vec<_>>();
    format_pairs(pairs)
}

fn print_arg_value(v: &AstArgValue) -> String {
    match v {
        AstArgValue::Value(v) => print_value_ref(v),
        AstArgValue::List(items) => {
            if items.is_empty() {
                "[]".to_string()
            } else {
                format!(
                    "[{}]",
                    items
                        .iter()
                        .map(print_arg_list_elem)
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        }
        AstArgValue::FnRef(name) => name.clone(),
    }
}

fn print_arg_list_elem(elem: &AstArgListElem) -> String {
    match elem {
        AstArgListElem::Value(v) => print_value_ref(v),
        AstArgListElem::FnRef(name) => name.clone(),
    }
}

fn print_op_with_pairs(op_name: &str, pairs: Vec<(&str, String)>) -> String {
    format!("{} {}", op_name, format_pairs(pairs))
}

fn format_pairs(mut pairs: Vec<(&str, String)>) -> String {
    pairs.sort_by(|a, b| a.0.cmp(b.0).then_with(|| a.1.cmp(&b.1)));
    if pairs.is_empty() {
        return "{ }".to_string();
    }

    let body = pairs
        .into_iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{ {} }}", body)
}

fn bin_op_name(kind: &BinOpKind) -> &'static str {
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

fn remap_bb(old: u32, map: &HashMap<u32, u32>) -> u32 {
    map.get(&old).copied().unwrap_or(old)
}

fn canonical_float(v: f64) -> String {
    if v.is_nan() {
        return "0.0".to_string();
    }

    let mut s = v.to_string();
    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
        s.push_str(".0");
    }
    s
}

fn quote_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{{{:x}}}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn join_comma<I>(iter: I) -> String
where
    I: IntoIterator,
    I::Item: ToString,
{
    iter.into_iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn push_doc(out: &mut String, doc: Option<&str>) {
    let Some(doc) = doc else {
        return;
    };

    for line in doc.lines() {
        if line.is_empty() {
            out.push_str(";;;\n");
        } else {
            out.push_str(";;; ");
            out.push_str(line);
            out.push('\n');
        }
    }
}

fn normalize_newlines(source: &str) -> String {
    source.replace("\r\n", "\n").replace('\r', "\n")
}

fn strip_digest_lines(source: &str) -> String {
    let normalized = normalize_newlines(source);
    let mut out = String::new();

    for chunk in normalized.split_inclusive('\n') {
        let line = chunk.strip_suffix('\n').unwrap_or(chunk);
        if line.trim_start().starts_with("digest ") {
            continue;
        }
        out.push_str(chunk);
    }

    out
}

fn ensure_single_trailing_newline(source: &str) -> String {
    let trimmed = source.trim_end_matches('\n');
    let mut out = trimmed.to_string();
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use magpie_ast::{AstHeader, FileId, ModulePath, Span};

    fn spanned<T>(node: T) -> Spanned<T> {
        Spanned::new(node, Span::new(FileId(0), 0, 0))
    }

    #[test]
    fn format_csnf_sorts_and_deduplicates_header_items() {
        let ast = AstFile {
            header: spanned(AstHeader {
                module_path: spanned(ModulePath {
                    segments: vec!["demo".to_string(), "core".to_string()],
                }),
                exports: vec![
                    spanned(ExportItem::Fn("beta".to_string())),
                    spanned(ExportItem::Type("Alpha".to_string())),
                    spanned(ExportItem::Fn("beta".to_string())),
                ],
                imports: vec![
                    spanned(ImportGroup {
                        module_path: ModulePath {
                            segments: vec!["dep".to_string(), "beta".to_string()],
                        },
                        items: vec![
                            ImportItem::Fn("run".to_string()),
                            ImportItem::Type("Other".to_string()),
                        ],
                    }),
                    spanned(ImportGroup {
                        module_path: ModulePath {
                            segments: vec!["dep".to_string(), "alpha".to_string()],
                        },
                        items: vec![
                            ImportItem::Fn("make".to_string()),
                            ImportItem::Type("Thing".to_string()),
                            ImportItem::Fn("make".to_string()),
                        ],
                    }),
                ],
                digest: spanned("placeholder".to_string()),
            }),
            decls: Vec::new(),
        };

        let formatted = format_csnf(&ast);
        let expected = concat!(
            "module demo.core\n",
            "exports { Alpha, beta }\n",
            "imports { dep.alpha::{Thing, make}, dep.beta::{Other, run} }\n",
            "digest \"placeholder\"\n"
        );

        assert_eq!(formatted, expected);
    }

    #[test]
    fn update_digest_inserts_and_stabilizes_digest_line() {
        let source = "module demo\nexports { }\nimports { }\n";

        let updated_once = update_digest(source);
        let updated_twice = update_digest(&updated_once);
        assert_eq!(
            updated_once, updated_twice,
            "digest update should be idempotent"
        );

        let digest_line = updated_once
            .lines()
            .find(|line| line.starts_with("digest "))
            .expect("digest line should exist");

        let digest = digest_line
            .strip_prefix("digest \"")
            .and_then(|rest| rest.strip_suffix('"'))
            .expect("digest line should be quoted");
        assert_eq!(digest.len(), 64, "blake3 hex digest should be 64 chars");
        assert!(digest.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
