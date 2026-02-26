//! Semantic analysis, module resolution, symbol metadata, and AST -> HIR lowering.

use std::collections::{HashMap, HashSet};

use base32::Alphabet;
use magpie_ast::{
    AstArgListElem, AstArgValue, AstBaseType, AstBuiltinType, AstConstExpr, AstConstLit, AstDecl,
    AstFile, AstInstr, AstOp, AstOpVoid, AstSigDecl, AstTerminator, AstType, AstValueRef,
    BinOpKind, ImportItem, ModulePath, OwnershipMod, Span,
};
use magpie_diag::{Diagnostic, DiagnosticBag, Severity};
use magpie_hir::{
    BlockId, FnId, GlobalId, HirBlock, HirConst, HirConstLit, HirEnumVariant, HirFunction,
    HirGlobal, HirInstr, HirModule, HirOp, HirOpVoid, HirTerminator, HirTypeDecl, HirValue,
    LocalId,
};
use magpie_types::{
    fixed_type_ids, HandleKind, HeapBase, ModuleId, PrimType, Sid, TypeCtx, TypeId, TypeKind,
};

pub type FQN = String;

#[derive(Clone, Debug)]
pub struct FnSymbol {
    pub name: String,
    pub fqn: FQN,
    pub sid: Sid,
    pub params: Vec<TypeId>,
    pub ret_ty: TypeId,
}

#[derive(Clone, Debug)]
pub struct TypeSymbol {
    pub name: String,
    pub fqn: FQN,
    pub sid: Sid,
    pub type_id: TypeId,
}

#[derive(Clone, Debug)]
pub struct GlobalSymbol {
    pub name: String,
    pub fqn: FQN,
    pub sid: Sid,
    pub ty: TypeId,
}

#[derive(Clone, Debug)]
pub struct SigSymbol {
    pub name: String,
    pub fqn: FQN,
    pub sid: Sid,
    pub param_types: Vec<TypeId>,
    pub ret_ty: TypeId,
    pub digest: String,
}

#[derive(Clone, Debug, Default)]
pub struct SymbolTable {
    pub functions: HashMap<String, FnSymbol>,
    pub types: HashMap<String, TypeSymbol>,
    pub globals: HashMap<String, GlobalSymbol>,
    pub sigs: HashMap<String, SigSymbol>,
}

#[derive(Clone, Debug)]
pub struct ResolvedModule<'a> {
    pub module_id: ModuleId,
    pub path: String,
    pub ast: &'a AstFile,
    pub symbol_table: SymbolTable,
    pub resolved_imports: Vec<(String, FQN)>,
}

pub fn resolve_modules<'a>(
    files: &'a [AstFile],
    diag: &mut DiagnosticBag,
) -> Result<Vec<ResolvedModule<'a>>, ()> {
    let before = diag.error_count();

    let mut seen_modules: HashMap<String, usize> = HashMap::new();
    let mut modules = Vec::with_capacity(files.len());

    for (idx, file) in files.iter().enumerate() {
        let module_path = module_path_str(file);
        if let Some(prev) = seen_modules.insert(module_path.clone(), idx) {
            emit_error(
                diag,
                "MPS0001",
                Some(file.header.span),
                format!(
                    "Duplicate module path '{}'; first declaration is module index {}.",
                    module_path, prev
                ),
            );
        }

        let mut symbol_table = SymbolTable::default();
        collect_module_symbols(file, &module_path, &mut symbol_table, diag);

        modules.push(ResolvedModule {
            module_id: ModuleId(idx as u32),
            path: module_fs_path(&file.header.node.module_path.node),
            ast: file,
            symbol_table,
            resolved_imports: default_lang_item_imports(),
        });
    }

    let mut fn_index: HashMap<(String, String), String> = HashMap::new();
    let mut ty_index: HashMap<(String, String), String> = HashMap::new();
    let mut global_index: HashMap<(String, String), String> = HashMap::new();

    for module in &modules {
        let module_path = module_path_str(module.ast);
        for (name, sym) in &module.symbol_table.functions {
            fn_index.insert((module_path.clone(), name.clone()), sym.fqn.clone());
        }
        for (name, sym) in &module.symbol_table.types {
            ty_index.insert((module_path.clone(), name.clone()), sym.fqn.clone());
        }
        for (name, sym) in &module.symbol_table.globals {
            global_index.insert((module_path.clone(), name.clone()), sym.fqn.clone());
        }
    }

    for module in &mut modules {
        let mut imports = default_lang_item_import_map();
        let module_path = module_path_str(module.ast);

        for group in &module.ast.header.node.imports {
            let imported_module = group.node.module_path.to_string();
            let module_exists = seen_modules.contains_key(&imported_module);
            if !module_exists {
                emit_error(
                    diag,
                    "MPS0002",
                    Some(group.span),
                    format!(
                        "Imported module '{}' is not present in this compilation unit.",
                        imported_module
                    ),
                );
            }

            for item in &group.node.items {
                let (name, fqn_opt) = match item {
                    ImportItem::Fn(name) => {
                        let key = (imported_module.clone(), name.clone());
                        let fqn = fn_index
                            .get(&key)
                            .cloned()
                            .or_else(|| global_index.get(&key).cloned());
                        (name.clone(), fqn)
                    }
                    ImportItem::Type(name) => {
                        let key = (imported_module.clone(), name.clone());
                        let fqn = ty_index.get(&key).cloned();
                        (name.clone(), fqn)
                    }
                };

                let Some(fqn) = fqn_opt else {
                    emit_error(
                        diag,
                        "MPS0003",
                        Some(group.span),
                        format!("Cannot resolve import '{}::{}'.", imported_module, name),
                    );
                    continue;
                };

                match item {
                    ImportItem::Fn(name) => {
                        if module.symbol_table.functions.contains_key(name)
                            || module.symbol_table.globals.contains_key(name)
                        {
                            emit_error(
                                diag,
                                "MPS0004",
                                Some(group.span),
                                format!(
                                    "Import '{}' conflicts with local function/global in module '{}'.",
                                    name, module_path
                                ),
                            );
                            continue;
                        }
                    }
                    ImportItem::Type(name) => {
                        if module.symbol_table.types.contains_key(name) {
                            emit_error(
                                diag,
                                "MPS0005",
                                Some(group.span),
                                format!(
                                    "Import '{}' conflicts with local type in module '{}'.",
                                    name, module_path
                                ),
                            );
                            continue;
                        }
                    }
                }

                if let Some(existing) = imports.get(&name) {
                    if existing != &fqn {
                        emit_error(
                            diag,
                            "MPS0006",
                            Some(group.span),
                            format!("Ambiguous import '{}': '{}' vs '{}'.", name, existing, fqn),
                        );
                    }
                    continue;
                }

                imports.insert(name, fqn);
            }
        }

        let mut sorted_imports: Vec<(String, FQN)> = imports.into_iter().collect();
        sorted_imports.sort_by(|a, b| a.0.cmp(&b.0));
        module.resolved_imports = sorted_imports;
    }

    for module in &mut modules {
        let module_path = module_path_str(module.ast);
        let import_map: HashMap<String, String> = module
            .resolved_imports
            .iter()
            .cloned()
            .collect::<HashMap<_, _>>();
        let value_types = collect_local_value_types(module.ast);
        let mut type_ctx = TypeCtx::new();

        for decl in &module.ast.decls {
            match &decl.node {
                AstDecl::Fn(f)
                | AstDecl::AsyncFn(f)
                | AstDecl::UnsafeFn(f)
                | AstDecl::GpuFn(magpie_ast::AstGpuFnDecl { inner: f, .. }) => {
                    let params = f
                        .params
                        .iter()
                        .map(|p| {
                            ast_type_to_type_id(
                                &p.ty.node,
                                &module_path,
                                &module.symbol_table,
                                &import_map,
                                &value_types,
                                &mut type_ctx,
                                diag,
                            )
                        })
                        .collect::<Vec<_>>();
                    let ret_ty = ast_type_to_type_id(
                        &f.ret_ty.node,
                        &module_path,
                        &module.symbol_table,
                        &import_map,
                        &value_types,
                        &mut type_ctx,
                        diag,
                    );

                    if let Some(sym) = module.symbol_table.functions.get_mut(&f.name) {
                        sym.params = params;
                        sym.ret_ty = ret_ty;
                    }
                }
                AstDecl::Extern(ext) => {
                    for item in &ext.items {
                        let params = item
                            .params
                            .iter()
                            .map(|p| {
                                ast_type_to_type_id(
                                    &p.ty.node,
                                    &module_path,
                                    &module.symbol_table,
                                    &import_map,
                                    &value_types,
                                    &mut type_ctx,
                                    diag,
                                )
                            })
                            .collect::<Vec<_>>();
                        let ret_ty = ast_type_to_type_id(
                            &item.ret_ty.node,
                            &module_path,
                            &module.symbol_table,
                            &import_map,
                            &value_types,
                            &mut type_ctx,
                            diag,
                        );

                        if let Some(sym) = module.symbol_table.functions.get_mut(&item.name) {
                            sym.params = params;
                            sym.ret_ty = ret_ty;
                        }
                    }
                }
                AstDecl::Global(g) => {
                    let ty = ast_type_to_type_id(
                        &g.ty.node,
                        &module_path,
                        &module.symbol_table,
                        &import_map,
                        &value_types,
                        &mut type_ctx,
                        diag,
                    );
                    if let Some(sym) = module.symbol_table.globals.get_mut(&g.name) {
                        sym.ty = ty;
                    }
                }
                AstDecl::Sig(sig) => {
                    let (params, ret_ty, digest) = lower_sig_symbol(
                        sig,
                        &module_path,
                        &module.symbol_table,
                        &import_map,
                        &value_types,
                        &mut type_ctx,
                        diag,
                    );

                    if let Some(sym) = module.symbol_table.sigs.get_mut(&sig.name) {
                        sym.param_types = params;
                        sym.ret_ty = ret_ty;
                        sym.digest = digest;
                    }
                }
                _ => {}
            }
        }
    }

    if diag.error_count() > before {
        Err(())
    } else {
        Ok(modules)
    }
}

pub fn generate_sid(kind: char, input: &str) -> Sid {
    let kind_word = match kind {
        'M' => "module",
        'F' => "fn",
        'T' => "type",
        'G' => "global",
        'E' => "sig",
        _ => "unknown",
    };

    let payload = format!("magpie:sid:v0.1|{}|{}", kind_word, input);
    let digest = blake3::hash(payload.as_bytes());
    let encoded = base32::encode(Alphabet::Crockford, digest.as_bytes());

    let mut suffix: String = encoded.chars().take(10).collect();
    if suffix.len() < 10 {
        suffix.push_str(&"0".repeat(10 - suffix.len()));
    }

    Sid(format!("{}:{}", kind, suffix))
}

pub fn type_str(ty: &TypeKind, type_ctx: &TypeCtx) -> String {
    match ty {
        TypeKind::Prim(p) => prim_type_str(*p).to_string(),
        TypeKind::HeapHandle { hk, base } => {
            let base_s = heap_base_str(base, type_ctx);
            match hk {
                HandleKind::Unique => base_s,
                HandleKind::Shared => format!("shared {}", base_s),
                HandleKind::Borrow => format!("borrow {}", base_s),
                HandleKind::MutBorrow => format!("mutborrow {}", base_s),
                HandleKind::Weak => format!("weak {}", base_s),
            }
        }
        TypeKind::BuiltinOption { inner } => {
            format!("TOption<{}>", type_id_str(*inner, type_ctx))
        }
        TypeKind::BuiltinResult { ok, err } => {
            format!(
                "TResult<{},{}>",
                type_id_str(*ok, type_ctx),
                type_id_str(*err, type_ctx)
            )
        }
        TypeKind::RawPtr { to } => format!("rawptr<{}>", type_id_str(*to, type_ctx)),
        TypeKind::Arr { n, elem } => format!("arr<{},{}>", n, type_id_str(*elem, type_ctx)),
        TypeKind::Vec { n, elem } => format!("vec<{},{}>", n, type_id_str(*elem, type_ctx)),
        TypeKind::Tuple { elems } => {
            let elems = elems
                .iter()
                .map(|t| type_id_str(*t, type_ctx))
                .collect::<Vec<_>>()
                .join(",");
            format!("tuple<{}>", elems)
        }
        TypeKind::ValueStruct { sid } => sid.0.clone(),
    }
}

pub fn sig_core_str(
    fqn: &str,
    param_types: &[TypeId],
    ret_ty: TypeId,
    type_ctx: &TypeCtx,
) -> String {
    let params = param_types
        .iter()
        .map(|ty| type_id_str(*ty, type_ctx))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "fn {}({}) -> {}",
        fqn,
        params,
        type_id_str(ret_ty, type_ctx)
    )
}

pub fn sig_digest(sig_core: &str) -> String {
    let payload = format!("magpie:sigdigest:v0.1|{}", sig_core);
    blake3::hash(payload.as_bytes()).to_hex().to_string()
}

pub fn lower_to_hir(
    resolved: &ResolvedModule<'_>,
    type_ctx: &mut TypeCtx,
    diag: &mut DiagnosticBag,
) -> Result<HirModule, ()> {
    let before = diag.error_count();
    let module_path = module_path_str(resolved.ast);
    let import_map: HashMap<String, String> = resolved
        .resolved_imports
        .iter()
        .cloned()
        .collect::<HashMap<_, _>>();
    let value_types = collect_local_value_types(resolved.ast);

    let mut type_decls = Vec::new();
    for decl in &resolved.ast.decls {
        match &decl.node {
            AstDecl::HeapStruct(s) | AstDecl::ValueStruct(s) => {
                let sid = resolve_type_sid(&s.name, &module_path, resolved);
                let fields = s
                    .fields
                    .iter()
                    .map(|f| {
                        (
                            f.name.clone(),
                            ast_type_to_type_id(
                                &f.ty.node,
                                &module_path,
                                &resolved.symbol_table,
                                &import_map,
                                &value_types,
                                type_ctx,
                                diag,
                            ),
                        )
                    })
                    .collect::<Vec<_>>();
                type_decls.push(HirTypeDecl::Struct {
                    sid,
                    name: s.name.clone(),
                    fields,
                });
            }
            AstDecl::HeapEnum(e) | AstDecl::ValueEnum(e) => {
                let sid = resolve_type_sid(&e.name, &module_path, resolved);
                let variants = e
                    .variants
                    .iter()
                    .enumerate()
                    .map(|(tag, v)| HirEnumVariant {
                        name: v.name.clone(),
                        tag: tag as i32,
                        fields: v
                            .fields
                            .iter()
                            .map(|f| {
                                (
                                    f.name.clone(),
                                    ast_type_to_type_id(
                                        &f.ty.node,
                                        &module_path,
                                        &resolved.symbol_table,
                                        &import_map,
                                        &value_types,
                                        type_ctx,
                                        diag,
                                    ),
                                )
                            })
                            .collect(),
                    })
                    .collect::<Vec<_>>();
                type_decls.push(HirTypeDecl::Enum {
                    sid,
                    name: e.name.clone(),
                    variants,
                });
            }
            _ => {}
        }
    }

    let mut globals = Vec::new();
    let mut next_global = 0_u32;
    for decl in &resolved.ast.decls {
        if let AstDecl::Global(g) = &decl.node {
            let ty = ast_type_to_type_id(
                &g.ty.node,
                &module_path,
                &resolved.symbol_table,
                &import_map,
                &value_types,
                type_ctx,
                diag,
            );
            let init = lower_const_expr(
                &g.init,
                &module_path,
                &resolved.symbol_table,
                &import_map,
                &value_types,
                type_ctx,
                diag,
            );

            globals.push(HirGlobal {
                id: GlobalId(next_global),
                name: g.name.clone(),
                ty,
                init,
            });
            next_global += 1;
        }
    }

    let mut functions = Vec::new();
    let mut next_fn = 0_u32;

    for decl in &resolved.ast.decls {
        let (f, is_async, is_unsafe) = match &decl.node {
            AstDecl::Fn(f) => (f, false, false),
            AstDecl::AsyncFn(f) => (f, true, false),
            AstDecl::UnsafeFn(f) => (f, false, true),
            AstDecl::GpuFn(g) => (&g.inner, false, false),
            _ => continue,
        };

        let mut next_local = 0_u32;
        let mut local_ids: HashMap<String, LocalId> = HashMap::new();
        let mut params = Vec::with_capacity(f.params.len());

        for param in &f.params {
            let id = LocalId(next_local);
            next_local += 1;

            if local_ids.insert(param.name.clone(), id).is_some() {
                emit_error(
                    diag,
                    "MPS0010",
                    Some(param.ty.span),
                    format!("Duplicate parameter name '{}'.", param.name),
                );
            }

            let ty = ast_type_to_type_id(
                &param.ty.node,
                &module_path,
                &resolved.symbol_table,
                &import_map,
                &value_types,
                type_ctx,
                diag,
            );
            params.push((id, ty));
        }

        let ret_ty = ast_type_to_type_id(
            &f.ret_ty.node,
            &module_path,
            &resolved.symbol_table,
            &import_map,
            &value_types,
            type_ctx,
            diag,
        );

        let sid = resolved
            .symbol_table
            .functions
            .get(&f.name)
            .map(|s| s.sid.clone())
            .unwrap_or_else(|| generate_sid('F', &format!("{}.{}", module_path, f.name)));

        let mut blocks = Vec::with_capacity(f.blocks.len());
        for block in &f.blocks {
            let mut instrs = Vec::new();
            let mut void_ops = Vec::new();

            for instr in &block.node.instrs {
                lower_instr(
                    &instr.node,
                    instr.span,
                    &module_path,
                    resolved,
                    &import_map,
                    &value_types,
                    &mut local_ids,
                    &mut next_local,
                    &mut instrs,
                    &mut void_ops,
                    type_ctx,
                    diag,
                );
            }

            let terminator = lower_terminator(
                &block.node.terminator.node,
                &module_path,
                resolved,
                &import_map,
                &value_types,
                &local_ids,
                type_ctx,
                diag,
            );

            blocks.push(HirBlock {
                id: BlockId(block.node.label),
                instrs,
                void_ops,
                terminator,
            });
        }

        functions.push(HirFunction {
            fn_id: FnId(next_fn),
            sid,
            name: f.name.clone(),
            params,
            ret_ty,
            blocks,
            is_async,
            is_unsafe,
        });

        next_fn += 1;
    }

    let hir = HirModule {
        module_id: resolved.module_id,
        sid: generate_sid('M', &module_path),
        path: module_path,
        functions,
        globals,
        type_decls,
    };

    if diag.error_count() > before {
        Err(())
    } else {
        Ok(hir)
    }
}

fn lower_instr(
    instr: &AstInstr,
    span: Span,
    module_path: &str,
    resolved: &ResolvedModule<'_>,
    import_map: &HashMap<String, String>,
    value_types: &HashSet<String>,
    local_ids: &mut HashMap<String, LocalId>,
    next_local: &mut u32,
    out_instrs: &mut Vec<HirInstr>,
    out_void_ops: &mut Vec<HirOpVoid>,
    type_ctx: &mut TypeCtx,
    diag: &mut DiagnosticBag,
) {
    match instr {
        AstInstr::Assign { name, ty, op } => {
            let dst = if let Some(existing) = local_ids.get(name) {
                emit_error(
                    diag,
                    "MPS0011",
                    Some(span),
                    format!("SSA local '{}' is defined more than once.", name),
                );
                *existing
            } else {
                let id = LocalId(*next_local);
                *next_local += 1;
                local_ids.insert(name.clone(), id);
                id
            };

            let ty = ast_type_to_type_id(
                &ty.node,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            );

            let op = lower_op(
                op,
                module_path,
                resolved,
                import_map,
                value_types,
                local_ids,
                type_ctx,
                diag,
            );

            out_instrs.push(HirInstr { dst, ty, op });
        }
        AstInstr::Void(v) => {
            let op = lower_op_void(
                v,
                module_path,
                resolved,
                import_map,
                value_types,
                local_ids,
                type_ctx,
                diag,
            );
            out_void_ops.push(op);
        }
        AstInstr::UnsafeBlock(inner) => {
            for i in inner {
                lower_instr(
                    &i.node,
                    i.span,
                    module_path,
                    resolved,
                    import_map,
                    value_types,
                    local_ids,
                    next_local,
                    out_instrs,
                    out_void_ops,
                    type_ctx,
                    diag,
                );
            }
        }
    }
}

fn lower_op(
    op: &AstOp,
    module_path: &str,
    resolved: &ResolvedModule<'_>,
    import_map: &HashMap<String, String>,
    value_types: &HashSet<String>,
    locals: &HashMap<String, LocalId>,
    type_ctx: &mut TypeCtx,
    diag: &mut DiagnosticBag,
) -> HirOp {
    match op {
        AstOp::Const(c) => HirOp::Const(lower_const_expr(
            c,
            module_path,
            &resolved.symbol_table,
            import_map,
            value_types,
            type_ctx,
            diag,
        )),
        AstOp::BinOp { kind, lhs, rhs } => {
            let lhs = lower_value_ref(
                lhs,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            );
            let rhs = lower_value_ref(
                rhs,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            );
            match kind {
                BinOpKind::IAdd => HirOp::IAdd { lhs, rhs },
                BinOpKind::ISub => HirOp::ISub { lhs, rhs },
                BinOpKind::IMul => HirOp::IMul { lhs, rhs },
                BinOpKind::ISDiv => HirOp::ISDiv { lhs, rhs },
                BinOpKind::IUDiv => HirOp::IUDiv { lhs, rhs },
                BinOpKind::ISRem => HirOp::ISRem { lhs, rhs },
                BinOpKind::IURem => HirOp::IURem { lhs, rhs },
                BinOpKind::IAddWrap => HirOp::IAddWrap { lhs, rhs },
                BinOpKind::ISubWrap => HirOp::ISubWrap { lhs, rhs },
                BinOpKind::IMulWrap => HirOp::IMulWrap { lhs, rhs },
                BinOpKind::IAddChecked => HirOp::IAddChecked { lhs, rhs },
                BinOpKind::ISubChecked => HirOp::ISubChecked { lhs, rhs },
                BinOpKind::IMulChecked => HirOp::IMulChecked { lhs, rhs },
                BinOpKind::IAnd => HirOp::IAnd { lhs, rhs },
                BinOpKind::IOr => HirOp::IOr { lhs, rhs },
                BinOpKind::IXor => HirOp::IXor { lhs, rhs },
                BinOpKind::IShl => HirOp::IShl { lhs, rhs },
                BinOpKind::ILshr => HirOp::ILshr { lhs, rhs },
                BinOpKind::IAshr => HirOp::IAshr { lhs, rhs },
                BinOpKind::FAdd => HirOp::FAdd { lhs, rhs },
                BinOpKind::FSub => HirOp::FSub { lhs, rhs },
                BinOpKind::FMul => HirOp::FMul { lhs, rhs },
                BinOpKind::FDiv => HirOp::FDiv { lhs, rhs },
                BinOpKind::FRem => HirOp::FRem { lhs, rhs },
                BinOpKind::FAddFast => HirOp::FAddFast { lhs, rhs },
                BinOpKind::FSubFast => HirOp::FSubFast { lhs, rhs },
                BinOpKind::FMulFast => HirOp::FMulFast { lhs, rhs },
                BinOpKind::FDivFast => HirOp::FDivFast { lhs, rhs },
            }
        }
        AstOp::Cmp {
            kind,
            pred,
            lhs,
            rhs,
        } => {
            let lhs = lower_value_ref(
                lhs,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            );
            let rhs = lower_value_ref(
                rhs,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            );
            match kind {
                magpie_ast::CmpKind::ICmp => HirOp::ICmp {
                    pred: pred.clone(),
                    lhs,
                    rhs,
                },
                magpie_ast::CmpKind::FCmp => HirOp::FCmp {
                    pred: pred.clone(),
                    lhs,
                    rhs,
                },
            }
        }
        AstOp::Call {
            callee,
            targs,
            args,
        } => HirOp::Call {
            callee_sid: resolve_fn_sid(callee, module_path, resolved, import_map),
            inst: targs
                .iter()
                .map(|t| {
                    ast_type_to_type_id(
                        t,
                        module_path,
                        &resolved.symbol_table,
                        import_map,
                        value_types,
                        type_ctx,
                        diag,
                    )
                })
                .collect(),
            args: lower_call_args(
                args,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::CallIndirect { callee, args } => HirOp::CallIndirect {
            callee: lower_value_ref(
                callee,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            args: lower_call_args(
                args,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::Try {
            callee,
            targs,
            args,
        } => HirOp::Call {
            callee_sid: resolve_fn_sid(callee, module_path, resolved, import_map),
            inst: targs
                .iter()
                .map(|t| {
                    ast_type_to_type_id(
                        t,
                        module_path,
                        &resolved.symbol_table,
                        import_map,
                        value_types,
                        type_ctx,
                        diag,
                    )
                })
                .collect(),
            args: lower_call_args(
                args,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::SuspendCall {
            callee,
            targs,
            args,
        } => HirOp::SuspendCall {
            callee_sid: resolve_fn_sid(callee, module_path, resolved, import_map),
            inst: targs
                .iter()
                .map(|t| {
                    ast_type_to_type_id(
                        t,
                        module_path,
                        &resolved.symbol_table,
                        import_map,
                        value_types,
                        type_ctx,
                        diag,
                    )
                })
                .collect(),
            args: lower_call_args(
                args,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::SuspendAwait { fut } => HirOp::SuspendAwait {
            fut: lower_value_ref(
                fut,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::New { ty, fields } => HirOp::New {
            ty: ast_type_to_type_id(
                ty,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            ),
            fields: fields
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        lower_value_ref(
                            v,
                            module_path,
                            &resolved.symbol_table,
                            import_map,
                            value_types,
                            locals,
                            type_ctx,
                            diag,
                        ),
                    )
                })
                .collect(),
        },
        AstOp::GetField { obj, field } => HirOp::GetField {
            obj: lower_value_ref(
                obj,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            field: field.clone(),
        },
        AstOp::Phi { ty, incomings } => HirOp::Phi {
            ty: ast_type_to_type_id(
                ty,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            ),
            incomings: incomings
                .iter()
                .map(|(bb, v)| {
                    (
                        BlockId(*bb),
                        lower_value_ref(
                            v,
                            module_path,
                            &resolved.symbol_table,
                            import_map,
                            value_types,
                            locals,
                            type_ctx,
                            diag,
                        ),
                    )
                })
                .collect(),
        },
        AstOp::EnumNew { variant, args } => HirOp::EnumNew {
            variant: variant.clone(),
            args: args
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        lower_value_ref(
                            v,
                            module_path,
                            &resolved.symbol_table,
                            import_map,
                            value_types,
                            locals,
                            type_ctx,
                            diag,
                        ),
                    )
                })
                .collect(),
        },
        AstOp::EnumTag { v } => HirOp::EnumTag {
            v: lower_value_ref(
                v,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::EnumPayload { variant, v } => HirOp::EnumPayload {
            variant: variant.clone(),
            v: lower_value_ref(
                v,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::EnumIs { variant, v } => HirOp::EnumIs {
            variant: variant.clone(),
            v: lower_value_ref(
                v,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::Share { v } => HirOp::Share {
            v: lower_value_ref(
                v,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::CloneShared { v } => HirOp::CloneShared {
            v: lower_value_ref(
                v,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::CloneWeak { v } => HirOp::CloneWeak {
            v: lower_value_ref(
                v,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::WeakDowngrade { v } => HirOp::WeakDowngrade {
            v: lower_value_ref(
                v,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::WeakUpgrade { v } => HirOp::WeakUpgrade {
            v: lower_value_ref(
                v,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::Cast { to, v, .. } => HirOp::Cast {
            to: ast_type_to_type_id(
                to,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            ),
            v: lower_value_ref(
                v,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::BorrowShared { v } => HirOp::BorrowShared {
            v: lower_value_ref(
                v,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::BorrowMut { v } => HirOp::BorrowMut {
            v: lower_value_ref(
                v,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::PtrNull { ty } => HirOp::PtrNull {
            to: ast_type_to_type_id(
                ty,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            ),
        },
        AstOp::PtrAddr { p, .. } => HirOp::PtrAddr {
            p: lower_value_ref(
                p,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::PtrFromAddr { ty, addr } => HirOp::PtrFromAddr {
            to: ast_type_to_type_id(
                ty,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            ),
            addr: lower_value_ref(
                addr,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::PtrAdd { p, count, .. } => HirOp::PtrAdd {
            p: lower_value_ref(
                p,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            count: lower_value_ref(
                count,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::PtrLoad { ty, p } => HirOp::PtrLoad {
            to: ast_type_to_type_id(
                ty,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            ),
            p: lower_value_ref(
                p,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::CallableCapture { fn_ref, captures } => HirOp::CallableCapture {
            fn_ref: resolve_fn_sid(fn_ref, module_path, resolved, import_map),
            captures: captures
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        lower_value_ref(
                            v,
                            module_path,
                            &resolved.symbol_table,
                            import_map,
                            value_types,
                            locals,
                            type_ctx,
                            diag,
                        ),
                    )
                })
                .collect(),
        },
        AstOp::ArrNew { elem_ty, cap } => HirOp::ArrNew {
            elem_ty: ast_type_to_type_id(
                elem_ty,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            ),
            cap: lower_value_ref(
                cap,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::ArrLen { arr } => HirOp::ArrLen {
            arr: lower_value_ref(
                arr,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::ArrGet { arr, idx } => HirOp::ArrGet {
            arr: lower_value_ref(
                arr,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            idx: lower_value_ref(
                idx,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::ArrPop { arr } => HirOp::ArrPop {
            arr: lower_value_ref(
                arr,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::ArrSlice { arr, start, end } => HirOp::ArrSlice {
            arr: lower_value_ref(
                arr,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            start: lower_value_ref(
                start,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            end: lower_value_ref(
                end,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::ArrContains { arr, val } => HirOp::ArrContains {
            arr: lower_value_ref(
                arr,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            val: lower_value_ref(
                val,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::ArrMap { arr, func } => HirOp::ArrMap {
            arr: lower_value_ref(
                arr,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            func: lower_value_ref(
                func,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::ArrFilter { arr, func } => HirOp::ArrFilter {
            arr: lower_value_ref(
                arr,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            func: lower_value_ref(
                func,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::ArrReduce { arr, init, func } => HirOp::ArrReduce {
            arr: lower_value_ref(
                arr,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            init: lower_value_ref(
                init,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            func: lower_value_ref(
                func,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::MapNew { key_ty, val_ty } => HirOp::MapNew {
            key_ty: ast_type_to_type_id(
                key_ty,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            ),
            val_ty: ast_type_to_type_id(
                val_ty,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            ),
        },
        AstOp::MapLen { map } => HirOp::MapLen {
            map: lower_value_ref(
                map,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::MapGet { map, key } => HirOp::MapGet {
            map: lower_value_ref(
                map,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            key: lower_value_ref(
                key,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::MapGetRef { map, key } => HirOp::MapGetRef {
            map: lower_value_ref(
                map,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            key: lower_value_ref(
                key,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::MapDelete { map, key } => HirOp::MapDelete {
            map: lower_value_ref(
                map,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            key: lower_value_ref(
                key,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::MapContainsKey { map, key } => HirOp::MapContainsKey {
            map: lower_value_ref(
                map,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            key: lower_value_ref(
                key,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::MapKeys { map } => HirOp::MapKeys {
            map: lower_value_ref(
                map,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::MapValues { map } => HirOp::MapValues {
            map: lower_value_ref(
                map,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::StrConcat { a, b } => HirOp::StrConcat {
            a: lower_value_ref(
                a,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            b: lower_value_ref(
                b,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::StrLen { s } => HirOp::StrLen {
            s: lower_value_ref(
                s,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::StrEq { a, b } => HirOp::StrEq {
            a: lower_value_ref(
                a,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            b: lower_value_ref(
                b,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::StrSlice { s, start, end } => HirOp::StrSlice {
            s: lower_value_ref(
                s,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            start: lower_value_ref(
                start,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            end: lower_value_ref(
                end,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::StrBytes { s } => HirOp::StrBytes {
            s: lower_value_ref(
                s,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::StrBuilderNew => HirOp::StrBuilderNew,
        AstOp::StrBuilderBuild { b } => HirOp::StrBuilderBuild {
            b: lower_value_ref(
                b,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::StrParseI64 { s } => HirOp::StrParseI64 {
            s: lower_value_ref(
                s,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::StrParseU64 { s } => HirOp::StrParseU64 {
            s: lower_value_ref(
                s,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::StrParseF64 { s } => HirOp::StrParseF64 {
            s: lower_value_ref(
                s,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::StrParseBool { s } => HirOp::StrParseBool {
            s: lower_value_ref(
                s,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::JsonEncode { ty, v } => HirOp::JsonEncode {
            ty: ast_type_to_type_id(
                ty,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            ),
            v: lower_value_ref(
                v,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::JsonDecode { ty, s } => HirOp::JsonDecode {
            ty: ast_type_to_type_id(
                ty,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            ),
            s: lower_value_ref(
                s,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::GpuThreadId { .. } => HirOp::GpuThreadId,
        AstOp::GpuWorkgroupId { .. } => HirOp::GpuWorkgroupId,
        AstOp::GpuWorkgroupSize { .. } => HirOp::GpuWorkgroupSize,
        AstOp::GpuGlobalId { .. } => HirOp::GpuGlobalId,
        AstOp::GpuBufferLoad { buf, idx, .. } => HirOp::GpuBufferLoad {
            buf: lower_value_ref(
                buf,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            idx: lower_value_ref(
                idx,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::GpuBufferLen { buf, .. } => HirOp::GpuBufferLen {
            buf: lower_value_ref(
                buf,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::GpuShared { count, ty } => HirOp::GpuShared {
            ty: ast_type_to_type_id(
                ty,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            ),
            size: HirValue::Const(HirConst {
                ty: fixed_type_ids::I64,
                lit: HirConstLit::IntLit(*count as i128),
            }),
        },
        AstOp::GpuLaunch {
            device,
            kernel,
            grid,
            block,
            args,
        } => HirOp::GpuLaunch {
            device: lower_value_ref(
                device,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            kernel: resolve_fn_sid(kernel, module_path, resolved, import_map),
            groups: lower_arg_value_single(
                grid,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            )
            .unwrap_or_else(unit_hir_value),
            threads: lower_arg_value_single(
                block,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            )
            .unwrap_or_else(unit_hir_value),
            args: lower_arg_value_list(
                args,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOp::GpuLaunchAsync {
            device,
            kernel,
            grid,
            block,
            args,
        } => HirOp::GpuLaunchAsync {
            device: lower_value_ref(
                device,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            kernel: resolve_fn_sid(kernel, module_path, resolved, import_map),
            groups: lower_arg_value_single(
                grid,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            )
            .unwrap_or_else(unit_hir_value),
            threads: lower_arg_value_single(
                block,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            )
            .unwrap_or_else(unit_hir_value),
            args: lower_arg_value_list(
                args,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
    }
}

fn lower_op_void(
    op: &AstOpVoid,
    module_path: &str,
    resolved: &ResolvedModule<'_>,
    import_map: &HashMap<String, String>,
    value_types: &HashSet<String>,
    locals: &HashMap<String, LocalId>,
    type_ctx: &mut TypeCtx,
    diag: &mut DiagnosticBag,
) -> HirOpVoid {
    match op {
        AstOpVoid::CallVoid {
            callee,
            targs,
            args,
        } => HirOpVoid::CallVoid {
            callee_sid: resolve_fn_sid(callee, module_path, resolved, import_map),
            inst: targs
                .iter()
                .map(|t| {
                    ast_type_to_type_id(
                        t,
                        module_path,
                        &resolved.symbol_table,
                        import_map,
                        value_types,
                        type_ctx,
                        diag,
                    )
                })
                .collect(),
            args: lower_call_args(
                args,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOpVoid::CallVoidIndirect { .. } => {
            emit_error(
                diag,
                "MPS0018",
                None,
                "HIR has no void-indirect-call opcode in HirOpVoid; lowered as panic placeholder."
                    .to_string(),
            );
            HirOpVoid::Panic {
                msg: unit_hir_value(),
            }
        }
        AstOpVoid::SetField { obj, field, val } => HirOpVoid::SetField {
            obj: lower_value_ref(
                obj,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            field: field.clone(),
            value: lower_value_ref(
                val,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOpVoid::Panic { msg } => HirOpVoid::Panic {
            msg: lower_value_ref(
                msg,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOpVoid::PtrStore { ty, p, v } => HirOpVoid::PtrStore {
            to: ast_type_to_type_id(
                ty,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            ),
            p: lower_value_ref(
                p,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            v: lower_value_ref(
                v,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOpVoid::ArrSet { arr, idx, val } => HirOpVoid::ArrSet {
            arr: lower_value_ref(
                arr,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            idx: lower_value_ref(
                idx,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            val: lower_value_ref(
                val,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOpVoid::ArrPush { arr, val } => HirOpVoid::ArrPush {
            arr: lower_value_ref(
                arr,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            val: lower_value_ref(
                val,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOpVoid::ArrSort { arr } => HirOpVoid::ArrSort {
            arr: lower_value_ref(
                arr,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOpVoid::ArrForeach { arr, func } => HirOpVoid::ArrForeach {
            arr: lower_value_ref(
                arr,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            func: lower_value_ref(
                func,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOpVoid::MapSet { map, key, val } => HirOpVoid::MapSet {
            map: lower_value_ref(
                map,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            key: lower_value_ref(
                key,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            val: lower_value_ref(
                val,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOpVoid::MapDeleteVoid { map, key } => HirOpVoid::MapDeleteVoid {
            map: lower_value_ref(
                map,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            key: lower_value_ref(
                key,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOpVoid::StrBuilderAppendStr { b, s } => HirOpVoid::StrBuilderAppendStr {
            b: lower_value_ref(
                b,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            s: lower_value_ref(
                s,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOpVoid::StrBuilderAppendI64 { b, v } => HirOpVoid::StrBuilderAppendI64 {
            b: lower_value_ref(
                b,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            v: lower_value_ref(
                v,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOpVoid::StrBuilderAppendI32 { b, v } => HirOpVoid::StrBuilderAppendI32 {
            b: lower_value_ref(
                b,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            v: lower_value_ref(
                v,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOpVoid::StrBuilderAppendF64 { b, v } => HirOpVoid::StrBuilderAppendF64 {
            b: lower_value_ref(
                b,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            v: lower_value_ref(
                v,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOpVoid::StrBuilderAppendBool { b, v } => HirOpVoid::StrBuilderAppendBool {
            b: lower_value_ref(
                b,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            v: lower_value_ref(
                v,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
        },
        AstOpVoid::GpuBarrier => HirOpVoid::GpuBarrier,
        AstOpVoid::GpuBufferStore { ty, buf, idx, v } => {
            let _ = ast_type_to_type_id(
                ty,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            );
            HirOpVoid::GpuBufferStore {
                buf: lower_value_ref(
                    buf,
                    module_path,
                    &resolved.symbol_table,
                    import_map,
                    value_types,
                    locals,
                    type_ctx,
                    diag,
                ),
                idx: lower_value_ref(
                    idx,
                    module_path,
                    &resolved.symbol_table,
                    import_map,
                    value_types,
                    locals,
                    type_ctx,
                    diag,
                ),
                val: lower_value_ref(
                    v,
                    module_path,
                    &resolved.symbol_table,
                    import_map,
                    value_types,
                    locals,
                    type_ctx,
                    diag,
                ),
            }
        }
    }
}

fn lower_terminator(
    term: &AstTerminator,
    module_path: &str,
    resolved: &ResolvedModule<'_>,
    import_map: &HashMap<String, String>,
    value_types: &HashSet<String>,
    locals: &HashMap<String, LocalId>,
    type_ctx: &mut TypeCtx,
    diag: &mut DiagnosticBag,
) -> HirTerminator {
    match term {
        AstTerminator::Ret(v) => HirTerminator::Ret(v.as_ref().map(|v| {
            lower_value_ref(
                v,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            )
        })),
        AstTerminator::Br(bb) => HirTerminator::Br(BlockId(*bb)),
        AstTerminator::Cbr {
            cond,
            then_bb,
            else_bb,
        } => HirTerminator::Cbr {
            cond: lower_value_ref(
                cond,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            then_bb: BlockId(*then_bb),
            else_bb: BlockId(*else_bb),
        },
        AstTerminator::Switch { val, arms, default } => HirTerminator::Switch {
            val: lower_value_ref(
                val,
                module_path,
                &resolved.symbol_table,
                import_map,
                value_types,
                locals,
                type_ctx,
                diag,
            ),
            arms: arms
                .iter()
                .map(|(lit, bb)| (lower_switch_lit(lit), BlockId(*bb)))
                .collect(),
            default: BlockId(*default),
        },
        AstTerminator::Unreachable => HirTerminator::Unreachable,
    }
}

fn lower_switch_lit(lit: &AstConstLit) -> HirConst {
    match lit {
        AstConstLit::Int(v) => HirConst {
            ty: fixed_type_ids::I64,
            lit: HirConstLit::IntLit(*v),
        },
        AstConstLit::Float(v) => HirConst {
            ty: fixed_type_ids::F64,
            lit: HirConstLit::FloatLit(*v),
        },
        AstConstLit::Str(v) => HirConst {
            ty: fixed_type_ids::STR,
            lit: HirConstLit::StringLit(v.clone()),
        },
        AstConstLit::Bool(v) => HirConst {
            ty: fixed_type_ids::BOOL,
            lit: HirConstLit::BoolLit(*v),
        },
        AstConstLit::Unit => HirConst {
            ty: fixed_type_ids::UNIT,
            lit: HirConstLit::Unit,
        },
    }
}

fn lower_call_args(
    args: &[(String, AstArgValue)],
    module_path: &str,
    symbol_table: &SymbolTable,
    import_map: &HashMap<String, String>,
    value_types: &HashSet<String>,
    locals: &HashMap<String, LocalId>,
    type_ctx: &mut TypeCtx,
    diag: &mut DiagnosticBag,
) -> Vec<HirValue> {
    let mut out = Vec::new();
    for (_, arg) in args {
        out.extend(lower_arg_value_list(
            arg,
            module_path,
            symbol_table,
            import_map,
            value_types,
            locals,
            type_ctx,
            diag,
        ));
    }
    out
}

fn lower_arg_value_single(
    arg: &AstArgValue,
    module_path: &str,
    symbol_table: &SymbolTable,
    import_map: &HashMap<String, String>,
    value_types: &HashSet<String>,
    locals: &HashMap<String, LocalId>,
    type_ctx: &mut TypeCtx,
    diag: &mut DiagnosticBag,
) -> Option<HirValue> {
    match arg {
        AstArgValue::Value(v) => Some(lower_value_ref(
            v,
            module_path,
            symbol_table,
            import_map,
            value_types,
            locals,
            type_ctx,
            diag,
        )),
        AstArgValue::List(v) => {
            if v.len() == 1 {
                match &v[0] {
                    AstArgListElem::Value(v) => Some(lower_value_ref(
                        v,
                        module_path,
                        symbol_table,
                        import_map,
                        value_types,
                        locals,
                        type_ctx,
                        diag,
                    )),
                    AstArgListElem::FnRef(name) => {
                        emit_error(
                            diag,
                            "MPS0012",
                            None,
                            format!(
                                "Function reference '{}' cannot be lowered as a value in this position.",
                                name
                            ),
                        );
                        None
                    }
                }
            } else {
                emit_error(
                    diag,
                    "MPS0013",
                    None,
                    "Expected a single argument value.".to_string(),
                );
                None
            }
        }
        AstArgValue::FnRef(name) => {
            emit_error(
                diag,
                "MPS0014",
                None,
                format!(
                    "Function reference '{}' cannot be lowered as a value in this position.",
                    name
                ),
            );
            None
        }
    }
}

fn lower_arg_value_list(
    arg: &AstArgValue,
    module_path: &str,
    symbol_table: &SymbolTable,
    import_map: &HashMap<String, String>,
    value_types: &HashSet<String>,
    locals: &HashMap<String, LocalId>,
    type_ctx: &mut TypeCtx,
    diag: &mut DiagnosticBag,
) -> Vec<HirValue> {
    match arg {
        AstArgValue::Value(v) => vec![lower_value_ref(
            v,
            module_path,
            symbol_table,
            import_map,
            value_types,
            locals,
            type_ctx,
            diag,
        )],
        AstArgValue::List(vs) => {
            let mut out = Vec::new();
            for v in vs {
                match v {
                    AstArgListElem::Value(v) => out.push(lower_value_ref(
                        v,
                        module_path,
                        symbol_table,
                        import_map,
                        value_types,
                        locals,
                        type_ctx,
                        diag,
                    )),
                    AstArgListElem::FnRef(name) => {
                        emit_error(
                            diag,
                            "MPS0015",
                            None,
                            format!(
                                "Function reference '{}' cannot be lowered as a plain value argument.",
                                name
                            ),
                        );
                    }
                }
            }
            out
        }
        AstArgValue::FnRef(name) => {
            emit_error(
                diag,
                "MPS0016",
                None,
                format!(
                    "Function reference '{}' cannot be lowered as a plain value argument.",
                    name
                ),
            );
            Vec::new()
        }
    }
}

fn lower_value_ref(
    v: &AstValueRef,
    module_path: &str,
    symbol_table: &SymbolTable,
    import_map: &HashMap<String, String>,
    value_types: &HashSet<String>,
    locals: &HashMap<String, LocalId>,
    type_ctx: &mut TypeCtx,
    diag: &mut DiagnosticBag,
) -> HirValue {
    match v {
        AstValueRef::Local(name) => {
            if let Some(id) = locals.get(name) {
                HirValue::Local(*id)
            } else {
                emit_error(
                    diag,
                    "MPS0017",
                    None,
                    format!("Unknown SSA local '{}'.", name),
                );
                unit_hir_value()
            }
        }
        AstValueRef::Const(c) => HirValue::Const(lower_const_expr(
            c,
            module_path,
            symbol_table,
            import_map,
            value_types,
            type_ctx,
            diag,
        )),
    }
}

fn lower_const_expr(
    c: &AstConstExpr,
    module_path: &str,
    symbol_table: &SymbolTable,
    import_map: &HashMap<String, String>,
    value_types: &HashSet<String>,
    type_ctx: &mut TypeCtx,
    diag: &mut DiagnosticBag,
) -> HirConst {
    let ty = ast_type_to_type_id(
        &c.ty,
        module_path,
        symbol_table,
        import_map,
        value_types,
        type_ctx,
        diag,
    );

    let lit = match &c.lit {
        AstConstLit::Int(v) => HirConstLit::IntLit(*v),
        AstConstLit::Float(v) => HirConstLit::FloatLit(*v),
        AstConstLit::Str(v) => HirConstLit::StringLit(v.clone()),
        AstConstLit::Bool(v) => HirConstLit::BoolLit(*v),
        AstConstLit::Unit => HirConstLit::Unit,
    };

    HirConst { ty, lit }
}

fn lower_sig_symbol(
    sig: &AstSigDecl,
    module_path: &str,
    symbol_table: &SymbolTable,
    import_map: &HashMap<String, String>,
    value_types: &HashSet<String>,
    type_ctx: &mut TypeCtx,
    diag: &mut DiagnosticBag,
) -> (Vec<TypeId>, TypeId, String) {
    let params = sig
        .param_types
        .iter()
        .map(|t| {
            ast_type_to_type_id(
                t,
                module_path,
                symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            )
        })
        .collect::<Vec<_>>();

    let ret_ty = ast_type_to_type_id(
        &sig.ret_ty,
        module_path,
        symbol_table,
        import_map,
        value_types,
        type_ctx,
        diag,
    );

    let fqn = format!("{}.sig.{}", module_path, sig.name);
    let core = sig_core_str(&fqn, &params, ret_ty, type_ctx);
    let digest = sig_digest(&core);

    (params, ret_ty, digest)
}

fn ast_type_to_type_id(
    ty: &AstType,
    module_path: &str,
    symbol_table: &SymbolTable,
    import_map: &HashMap<String, String>,
    value_types: &HashSet<String>,
    type_ctx: &mut TypeCtx,
    diag: &mut DiagnosticBag,
) -> TypeId {
    match &ty.base {
        AstBaseType::Prim(name) => {
            if let Some(prim) = prim_type_from_name(name) {
                type_ctx.lookup_by_prim(prim)
            } else {
                emit_error(
                    diag,
                    "MPT0001",
                    None,
                    format!("Unknown primitive type '{}'.", name),
                );
                fixed_type_ids::UNIT
            }
        }
        AstBaseType::Named { path, name, targs } => {
            let targs = targs
                .iter()
                .map(|t| {
                    ast_type_to_type_id(
                        t,
                        module_path,
                        symbol_table,
                        import_map,
                        value_types,
                        type_ctx,
                        diag,
                    )
                })
                .collect::<Vec<_>>();

            let fqn = if let Some(path) = path {
                format!("{}.{}", path.to_string(), name)
            } else if let Some(local) = symbol_table.types.get(name) {
                local.fqn.clone()
            } else if let Some(imported) = import_map.get(name) {
                imported.clone()
            } else {
                format!("{}.{}", module_path, name)
            };

            let sid = symbol_table
                .types
                .get(name)
                .map(|t| t.sid.clone())
                .unwrap_or_else(|| generate_sid('T', &fqn));

            if ty.ownership.is_none() && value_types.contains(name) && path.is_none() {
                type_ctx.intern(TypeKind::ValueStruct { sid })
            } else {
                let hk = ownership_to_handle(ty.ownership.as_ref());
                type_ctx.intern(TypeKind::HeapHandle {
                    hk,
                    base: HeapBase::UserType {
                        type_sid: sid,
                        targs,
                    },
                })
            }
        }
        AstBaseType::Builtin(b) => lower_builtin_type(
            b,
            ty.ownership.as_ref(),
            module_path,
            symbol_table,
            import_map,
            value_types,
            type_ctx,
            diag,
        ),
        AstBaseType::Callable { sig_ref } => {
            let sig_sid = resolve_sig_sid(sig_ref, module_path, symbol_table, import_map);
            let hk = ownership_to_handle(ty.ownership.as_ref());
            type_ctx.intern(TypeKind::HeapHandle {
                hk,
                base: HeapBase::Callable { sig_sid },
            })
        }
        AstBaseType::RawPtr(inner) => {
            let inner = ast_type_to_type_id(
                inner,
                module_path,
                symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            );
            type_ctx.intern(TypeKind::RawPtr { to: inner })
        }
    }
}

fn lower_builtin_type(
    b: &AstBuiltinType,
    ownership: Option<&OwnershipMod>,
    module_path: &str,
    symbol_table: &SymbolTable,
    import_map: &HashMap<String, String>,
    value_types: &HashSet<String>,
    type_ctx: &mut TypeCtx,
    diag: &mut DiagnosticBag,
) -> TypeId {
    match b {
        AstBuiltinType::Str => {
            let hk = ownership_to_handle(ownership);
            type_ctx.intern(TypeKind::HeapHandle {
                hk,
                base: HeapBase::BuiltinStr,
            })
        }
        AstBuiltinType::Array(elem) => {
            let elem = ast_type_to_type_id(
                elem,
                module_path,
                symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            );
            let hk = ownership_to_handle(ownership);
            type_ctx.intern(TypeKind::HeapHandle {
                hk,
                base: HeapBase::BuiltinArray { elem },
            })
        }
        AstBuiltinType::Map(key, val) => {
            let key = ast_type_to_type_id(
                key,
                module_path,
                symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            );
            let val = ast_type_to_type_id(
                val,
                module_path,
                symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            );
            let hk = ownership_to_handle(ownership);
            type_ctx.intern(TypeKind::HeapHandle {
                hk,
                base: HeapBase::BuiltinMap { key, val },
            })
        }
        AstBuiltinType::TOption(inner) => {
            if matches!(
                ownership,
                Some(OwnershipMod::Shared) | Some(OwnershipMod::Weak)
            ) {
                emit_error(
                    diag,
                    "MPT0002",
                    None,
                    "`shared`/`weak` are not valid on `TOption` (value enum).".to_string(),
                );
            }
            let inner = ast_type_to_type_id(
                inner,
                module_path,
                symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            );
            type_ctx.intern(TypeKind::BuiltinOption { inner })
        }
        AstBuiltinType::TResult(ok, err) => {
            if matches!(
                ownership,
                Some(OwnershipMod::Shared) | Some(OwnershipMod::Weak)
            ) {
                emit_error(
                    diag,
                    "MPT0003",
                    None,
                    "`shared`/`weak` are not valid on `TResult` (value enum).".to_string(),
                );
            }
            let ok = ast_type_to_type_id(
                ok,
                module_path,
                symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            );
            let err = ast_type_to_type_id(
                err,
                module_path,
                symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            );
            type_ctx.intern(TypeKind::BuiltinResult { ok, err })
        }
        AstBuiltinType::TStrBuilder => {
            let hk = ownership_to_handle(ownership);
            type_ctx.intern(TypeKind::HeapHandle {
                hk,
                base: HeapBase::BuiltinStrBuilder,
            })
        }
        AstBuiltinType::TMutex(inner) => {
            let inner = ast_type_to_type_id(
                inner,
                module_path,
                symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            );
            let hk = ownership_to_handle(ownership);
            type_ctx.intern(TypeKind::HeapHandle {
                hk,
                base: HeapBase::BuiltinMutex { inner },
            })
        }
        AstBuiltinType::TRwLock(inner) => {
            let inner = ast_type_to_type_id(
                inner,
                module_path,
                symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            );
            let hk = ownership_to_handle(ownership);
            type_ctx.intern(TypeKind::HeapHandle {
                hk,
                base: HeapBase::BuiltinRwLock { inner },
            })
        }
        AstBuiltinType::TCell(inner) => {
            let inner = ast_type_to_type_id(
                inner,
                module_path,
                symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            );
            let hk = ownership_to_handle(ownership);
            type_ctx.intern(TypeKind::HeapHandle {
                hk,
                base: HeapBase::BuiltinCell { inner },
            })
        }
        AstBuiltinType::TFuture(result) => {
            let result = ast_type_to_type_id(
                result,
                module_path,
                symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            );
            let hk = ownership_to_handle(ownership);
            type_ctx.intern(TypeKind::HeapHandle {
                hk,
                base: HeapBase::BuiltinFuture { result },
            })
        }
        AstBuiltinType::TChannelSend(elem) => {
            let elem = ast_type_to_type_id(
                elem,
                module_path,
                symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            );
            let hk = ownership_to_handle(ownership);
            type_ctx.intern(TypeKind::HeapHandle {
                hk,
                base: HeapBase::BuiltinChannelSend { elem },
            })
        }
        AstBuiltinType::TChannelRecv(elem) => {
            let elem = ast_type_to_type_id(
                elem,
                module_path,
                symbol_table,
                import_map,
                value_types,
                type_ctx,
                diag,
            );
            let hk = ownership_to_handle(ownership);
            type_ctx.intern(TypeKind::HeapHandle {
                hk,
                base: HeapBase::BuiltinChannelRecv { elem },
            })
        }
    }
}

fn prim_type_from_name(name: &str) -> Option<PrimType> {
    match name {
        "i1" => Some(PrimType::I1),
        "i8" => Some(PrimType::I8),
        "i16" => Some(PrimType::I16),
        "i32" => Some(PrimType::I32),
        "i64" => Some(PrimType::I64),
        "i128" => Some(PrimType::I128),
        "u1" => Some(PrimType::U1),
        "u8" => Some(PrimType::U8),
        "u16" => Some(PrimType::U16),
        "u32" => Some(PrimType::U32),
        "u64" => Some(PrimType::U64),
        "u128" => Some(PrimType::U128),
        "f16" => Some(PrimType::F16),
        "f32" => Some(PrimType::F32),
        "f64" => Some(PrimType::F64),
        "bool" => Some(PrimType::Bool),
        "unit" => Some(PrimType::Unit),
        _ => None,
    }
}

fn prim_type_str(p: PrimType) -> &'static str {
    match p {
        PrimType::I1 => "i1",
        PrimType::I8 => "i8",
        PrimType::I16 => "i16",
        PrimType::I32 => "i32",
        PrimType::I64 => "i64",
        PrimType::I128 => "i128",
        PrimType::U1 => "u1",
        PrimType::U8 => "u8",
        PrimType::U16 => "u16",
        PrimType::U32 => "u32",
        PrimType::U64 => "u64",
        PrimType::U128 => "u128",
        PrimType::F16 => "f16",
        PrimType::F32 => "f32",
        PrimType::F64 => "f64",
        PrimType::Bool => "bool",
        PrimType::Unit => "unit",
    }
}

fn type_id_str(ty: TypeId, type_ctx: &TypeCtx) -> String {
    type_ctx
        .lookup(ty)
        .map(|k| type_str(k, type_ctx))
        .unwrap_or_else(|| format!("type#{}", ty.0))
}

fn heap_base_str(base: &HeapBase, type_ctx: &TypeCtx) -> String {
    match base {
        HeapBase::BuiltinStr => "Str".to_string(),
        HeapBase::BuiltinArray { elem } => format!("Array<{}>", type_id_str(*elem, type_ctx)),
        HeapBase::BuiltinMap { key, val } => {
            format!(
                "Map<{},{}>",
                type_id_str(*key, type_ctx),
                type_id_str(*val, type_ctx)
            )
        }
        HeapBase::BuiltinStrBuilder => "TStrBuilder".to_string(),
        HeapBase::BuiltinMutex { inner } => format!("TMutex<{}>", type_id_str(*inner, type_ctx)),
        HeapBase::BuiltinRwLock { inner } => {
            format!("TRwLock<{}>", type_id_str(*inner, type_ctx))
        }
        HeapBase::BuiltinCell { inner } => format!("TCell<{}>", type_id_str(*inner, type_ctx)),
        HeapBase::BuiltinFuture { result } => {
            format!("TFuture<{}>", type_id_str(*result, type_ctx))
        }
        HeapBase::BuiltinChannelSend { elem } => {
            format!("TChannelSend<{}>", type_id_str(*elem, type_ctx))
        }
        HeapBase::BuiltinChannelRecv { elem } => {
            format!("TChannelRecv<{}>", type_id_str(*elem, type_ctx))
        }
        HeapBase::Callable { sig_sid } => format!("TCallable<{}>", sig_sid.0),
        HeapBase::UserType { type_sid, targs } => {
            if targs.is_empty() {
                type_sid.0.clone()
            } else {
                let targs = targs
                    .iter()
                    .map(|t| type_id_str(*t, type_ctx))
                    .collect::<Vec<_>>()
                    .join(",");
                format!("{}<{}>", type_sid.0, targs)
            }
        }
    }
}

fn ownership_to_handle(ownership: Option<&OwnershipMod>) -> HandleKind {
    match ownership {
        Some(OwnershipMod::Shared) => HandleKind::Shared,
        Some(OwnershipMod::Borrow) => HandleKind::Borrow,
        Some(OwnershipMod::MutBorrow) => HandleKind::MutBorrow,
        Some(OwnershipMod::Weak) => HandleKind::Weak,
        None => HandleKind::Unique,
    }
}

fn module_path_str(file: &AstFile) -> String {
    file.header.node.module_path.node.to_string()
}

fn module_fs_path(path: &ModulePath) -> String {
    let segments = &path.segments;
    let rel_segments = if segments.len() > 1 {
        &segments[1..]
    } else {
        segments.as_slice()
    };
    format!("src/{}.mp", rel_segments.join("/"))
}

fn collect_module_symbols(
    file: &AstFile,
    module_path: &str,
    table: &mut SymbolTable,
    diag: &mut DiagnosticBag,
) {
    let mut type_ctx = TypeCtx::new();

    for decl in &file.decls {
        match &decl.node {
            AstDecl::Fn(f)
            | AstDecl::AsyncFn(f)
            | AstDecl::UnsafeFn(f)
            | AstDecl::GpuFn(magpie_ast::AstGpuFnDecl { inner: f, .. }) => {
                insert_fn_symbol(table, module_path, &f.name, decl.span, diag);
            }
            AstDecl::Extern(ext) => {
                for item in &ext.items {
                    insert_fn_symbol(table, module_path, &item.name, decl.span, diag);
                }
            }
            AstDecl::HeapStruct(s) => {
                insert_type_symbol(
                    table,
                    module_path,
                    &s.name,
                    true,
                    &mut type_ctx,
                    decl.span,
                    diag,
                );
            }
            AstDecl::HeapEnum(e) => {
                insert_type_symbol(
                    table,
                    module_path,
                    &e.name,
                    true,
                    &mut type_ctx,
                    decl.span,
                    diag,
                );
            }
            AstDecl::ValueStruct(s) => {
                insert_type_symbol(
                    table,
                    module_path,
                    &s.name,
                    false,
                    &mut type_ctx,
                    decl.span,
                    diag,
                );
            }
            AstDecl::ValueEnum(e) => {
                insert_type_symbol(
                    table,
                    module_path,
                    &e.name,
                    false,
                    &mut type_ctx,
                    decl.span,
                    diag,
                );
            }
            AstDecl::Global(g) => {
                insert_global_symbol(table, module_path, &g.name, decl.span, diag);
            }
            AstDecl::Sig(sig) => {
                insert_sig_symbol(table, module_path, &sig.name, decl.span, diag);
            }
            AstDecl::Impl(_) => {}
        }
    }
}

fn insert_fn_symbol(
    table: &mut SymbolTable,
    module_path: &str,
    name: &str,
    span: Span,
    diag: &mut DiagnosticBag,
) {
    if table.functions.contains_key(name) || table.globals.contains_key(name) {
        emit_error(
            diag,
            "MPS0020",
            Some(span),
            format!(
                "No overloads allowed in @ namespace; symbol '{}' is already defined.",
                name
            ),
        );
        return;
    }

    let fqn = format!("{}.{}", module_path, name);
    table.functions.insert(
        name.to_string(),
        FnSymbol {
            name: name.to_string(),
            fqn: fqn.clone(),
            sid: generate_sid('F', &fqn),
            params: Vec::new(),
            ret_ty: fixed_type_ids::UNIT,
        },
    );
}

fn insert_type_symbol(
    table: &mut SymbolTable,
    module_path: &str,
    name: &str,
    is_heap: bool,
    type_ctx: &mut TypeCtx,
    span: Span,
    diag: &mut DiagnosticBag,
) {
    if table.types.contains_key(name) {
        emit_error(
            diag,
            "MPS0021",
            Some(span),
            format!(
                "No overloads allowed in T namespace; type '{}' is already defined.",
                name
            ),
        );
        return;
    }

    let fqn = format!("{}.{}", module_path, name);
    let sid = generate_sid('T', &fqn);

    let type_id = if is_heap {
        type_ctx.intern(TypeKind::HeapHandle {
            hk: HandleKind::Unique,
            base: HeapBase::UserType {
                type_sid: sid.clone(),
                targs: Vec::new(),
            },
        })
    } else {
        type_ctx.intern(TypeKind::ValueStruct { sid: sid.clone() })
    };

    table.types.insert(
        name.to_string(),
        TypeSymbol {
            name: name.to_string(),
            fqn,
            sid,
            type_id,
        },
    );
}

fn insert_global_symbol(
    table: &mut SymbolTable,
    module_path: &str,
    name: &str,
    span: Span,
    diag: &mut DiagnosticBag,
) {
    if table.functions.contains_key(name) || table.globals.contains_key(name) {
        emit_error(
            diag,
            "MPS0022",
            Some(span),
            format!(
                "No overloads allowed in @ namespace; symbol '{}' is already defined.",
                name
            ),
        );
        return;
    }

    let fqn = format!("{}.{}", module_path, name);
    table.globals.insert(
        name.to_string(),
        GlobalSymbol {
            name: name.to_string(),
            fqn: fqn.clone(),
            sid: generate_sid('G', &fqn),
            ty: fixed_type_ids::UNIT,
        },
    );
}

fn insert_sig_symbol(
    table: &mut SymbolTable,
    module_path: &str,
    name: &str,
    span: Span,
    diag: &mut DiagnosticBag,
) {
    if table.sigs.contains_key(name) {
        emit_error(
            diag,
            "MPS0023",
            Some(span),
            format!(
                "No overloads allowed in sig namespace; signature '{}' is already defined.",
                name
            ),
        );
        return;
    }

    let fqn = format!("{}.sig.{}", module_path, name);
    table.sigs.insert(
        name.to_string(),
        SigSymbol {
            name: name.to_string(),
            fqn: fqn.clone(),
            sid: generate_sid('E', &fqn),
            param_types: Vec::new(),
            ret_ty: fixed_type_ids::UNIT,
            digest: String::new(),
        },
    );
}

fn resolve_fn_sid(
    callee: &str,
    module_path: &str,
    resolved: &ResolvedModule<'_>,
    import_map: &HashMap<String, String>,
) -> Sid {
    if let Some(sym) = resolved.symbol_table.functions.get(callee) {
        return sym.sid.clone();
    }

    if let Some(fqn) = import_map.get(callee) {
        return generate_sid('F', fqn);
    }

    if callee.contains('.') {
        return generate_sid('F', callee);
    }

    generate_sid('F', &format!("{}.{}", module_path, callee))
}

fn resolve_sig_sid(
    sig_ref: &str,
    module_path: &str,
    symbol_table: &SymbolTable,
    import_map: &HashMap<String, String>,
) -> Sid {
    if let Some(sym) = symbol_table.sigs.get(sig_ref) {
        return sym.sid.clone();
    }

    let fqn = if sig_ref.contains('.') {
        sig_ref.to_string()
    } else if let Some(fqn) = import_map.get(sig_ref) {
        fqn.clone()
    } else {
        format!("{}.sig.{}", module_path, sig_ref)
    };

    generate_sid('E', &fqn)
}

fn resolve_type_sid(name: &str, module_path: &str, resolved: &ResolvedModule<'_>) -> Sid {
    resolved
        .symbol_table
        .types
        .get(name)
        .map(|t| t.sid.clone())
        .unwrap_or_else(|| generate_sid('T', &format!("{}.{}", module_path, name)))
}

fn default_lang_item_imports() -> Vec<(String, FQN)> {
    let mut items = default_lang_item_import_map()
        .into_iter()
        .collect::<Vec<(String, FQN)>>();
    items.sort_by(|a, b| a.0.cmp(&b.0));
    items
}

fn default_lang_item_import_map() -> HashMap<String, FQN> {
    HashMap::from([
        ("TOption".to_string(), "magpie.lang.TOption".to_string()),
        ("TResult".to_string(), "magpie.lang.TResult".to_string()),
        ("bool".to_string(), "magpie.lang.bool".to_string()),
        ("unit".to_string(), "magpie.lang.unit".to_string()),
        ("Str".to_string(), "magpie.lang.Str".to_string()),
        ("Array".to_string(), "magpie.lang.Array".to_string()),
        ("Map".to_string(), "magpie.lang.Map".to_string()),
    ])
}

fn collect_local_value_types(ast: &AstFile) -> HashSet<String> {
    let mut out = HashSet::new();
    for decl in &ast.decls {
        match &decl.node {
            AstDecl::ValueStruct(s) => {
                out.insert(s.name.clone());
            }
            AstDecl::ValueEnum(e) => {
                out.insert(e.name.clone());
            }
            _ => {}
        }
    }
    out
}

fn unit_hir_value() -> HirValue {
    HirValue::Const(HirConst {
        ty: fixed_type_ids::UNIT,
        lit: HirConstLit::Unit,
    })
}

fn emit_error(diag: &mut DiagnosticBag, code: &str, span: Option<Span>, message: String) {
    diag.emit(Diagnostic {
        code: code.to_string(),
        severity: Severity::Error,
        title: message.clone(),
        primary_span: span,
        secondary_spans: Vec::new(),
        message,
        explanation_md: None,
        why: None,
        suggested_fixes: Vec::new(),
    });
}
