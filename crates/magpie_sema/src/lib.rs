//! Semantic analysis, module resolution, symbol metadata, and AST -> HIR lowering.
#![allow(clippy::result_unit_err, clippy::too_many_arguments)]

use std::collections::{HashMap, HashSet};

use base32::Alphabet;
use magpie_ast::{
    AstArgListElem, AstArgValue, AstBaseType, AstBuiltinType, AstConstExpr, AstConstLit, AstDecl,
    AstFile, AstImplDecl, AstInstr, AstOp, AstOpVoid, AstSigDecl, AstTerminator, AstType,
    AstValueRef, BinOpKind, ImportItem, OwnershipMod, Span,
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
    pub is_unsafe: bool,
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
pub struct ResolvedModule {
    pub module_id: ModuleId,
    pub path: String,
    pub ast: AstFile,
    pub symbol_table: SymbolTable,
    pub resolved_imports: Vec<(String, FQN)>,
    pub unsafe_fn_sids: HashSet<Sid>,
}

pub fn resolve_modules(
    files: &[AstFile],
    diag: &mut DiagnosticBag,
) -> Result<Vec<ResolvedModule>, ()> {
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
            path: module_path.clone(),
            ast: file.clone(),
            symbol_table,
            resolved_imports: default_lang_item_imports(),
            unsafe_fn_sids: HashSet::new(),
        });
    }

    let mut fn_index: HashMap<(String, String), String> = HashMap::new();
    let mut ty_index: HashMap<(String, String), String> = HashMap::new();
    let mut global_index: HashMap<(String, String), String> = HashMap::new();

    for module in &modules {
        let module_path = module_path_str(&module.ast);
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
        let module_path = module_path_str(&module.ast);

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
        let module_path = module_path_str(&module.ast);
        let import_map: HashMap<String, String> = module
            .resolved_imports
            .iter()
            .cloned()
            .collect::<HashMap<_, _>>();
        let value_types = collect_local_value_types(&module.ast);
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

                        if ast_type_returns_rawptr(&item.ret_ty.node) {
                            match item
                                .attrs
                                .iter()
                                .find(|(key, _)| key == "returns")
                                .map(|(_, value)| value.as_str())
                            {
                                Some("owned") | Some("borrowed") => {}
                                Some(other) => emit_error(
                                    diag,
                                    "MPF0001",
                                    Some(item.ret_ty.span),
                                    format!(
                                        "extern fn '{}' returns rawptr but attrs {{ returns=... }} must be 'owned' or 'borrowed' (got '{}').",
                                        item.name, other
                                    ),
                                ),
                                None => emit_error(
                                    diag,
                                    "MPF0001",
                                    Some(item.ret_ty.span),
                                    format!(
                                        "extern fn '{}' returns rawptr but is missing attrs {{ returns=\"owned|borrowed\" }}.",
                                        item.name
                                    ),
                                ),
                            }
                        }

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
                AstDecl::Impl(impl_decl) => {
                    check_impl_orphan_rule(
                        &module_path,
                        impl_decl,
                        &module.symbol_table,
                        &import_map,
                        diag,
                    );
                }
                _ => {}
            }
        }
    }

    let known_unsafe_fn_sids: HashSet<Sid> = modules
        .iter()
        .flat_map(|m| m.symbol_table.functions.values())
        .filter(|sym| sym.is_unsafe)
        .map(|sym| sym.sid.clone())
        .collect();
    for module in &mut modules {
        module.unsafe_fn_sids = known_unsafe_fn_sids.clone();
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
    resolved: &ResolvedModule,
    type_ctx: &mut TypeCtx,
    diag: &mut DiagnosticBag,
) -> Result<HirModule, ()> {
    let before = diag.error_count();
    let module_path = module_path_str(&resolved.ast);
    let import_map: HashMap<String, String> = resolved
        .resolved_imports
        .iter()
        .cloned()
        .collect::<HashMap<_, _>>();
    let value_types = collect_local_value_types(&resolved.ast);

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
                type_ctx.register_type_fqn(sid.clone(), format!("{}.{}", module_path, s.name));
                type_ctx.register_value_struct_fields(sid.clone(), fields.clone());
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
                type_ctx.register_type_fqn(sid.clone(), format!("{}.{}", module_path, e.name));
                type_ctx.register_value_enum_variants(
                    sid.clone(),
                    variants
                        .iter()
                        .map(|v| (v.name.clone(), v.fields.clone()))
                        .collect(),
                );
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
                    is_unsafe,
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
    in_unsafe_context: bool,
    module_path: &str,
    resolved: &ResolvedModule,
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
            if !in_unsafe_context {
                if op_requires_unsafe(op) {
                    emit_error(
                        diag,
                        "MPS0024",
                        Some(span),
                        "raw pointer opcodes (`ptr.*`) are only allowed inside `unsafe {}` or `unsafe fn`.".to_string(),
                    );
                }
                if op_calls_unsafe_fn(op, module_path, resolved, import_map) {
                    emit_error(
                        diag,
                        "MPS0025",
                        Some(span),
                        "calling an `unsafe fn` requires an unsafe context (`unsafe {}` or `unsafe fn`).".to_string(),
                    );
                }
            }

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
            if !in_unsafe_context {
                if op_void_requires_unsafe(v) {
                    emit_error(
                        diag,
                        "MPS0024",
                        Some(span),
                        "raw pointer opcodes (`ptr.*`) are only allowed inside `unsafe {}` or `unsafe fn`.".to_string(),
                    );
                }
                if op_void_calls_unsafe_fn(v, module_path, resolved, import_map) {
                    emit_error(
                        diag,
                        "MPS0025",
                        Some(span),
                        "calling an `unsafe fn` requires an unsafe context (`unsafe {}` or `unsafe fn`).".to_string(),
                    );
                }
            }

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
                    true,
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

fn op_requires_unsafe(op: &AstOp) -> bool {
    matches!(
        op,
        AstOp::PtrNull { .. }
            | AstOp::PtrAddr { .. }
            | AstOp::PtrFromAddr { .. }
            | AstOp::PtrAdd { .. }
            | AstOp::PtrLoad { .. }
    )
}

fn op_void_requires_unsafe(op: &AstOpVoid) -> bool {
    matches!(op, AstOpVoid::PtrStore { .. })
}

fn op_calls_unsafe_fn(
    op: &AstOp,
    module_path: &str,
    resolved: &ResolvedModule,
    import_map: &HashMap<String, String>,
) -> bool {
    match op {
        AstOp::Call { callee, .. }
        | AstOp::Try { callee, .. }
        | AstOp::SuspendCall { callee, .. } => {
            callee_is_unsafe(callee, module_path, resolved, import_map)
        }
        _ => false,
    }
}

fn op_void_calls_unsafe_fn(
    op: &AstOpVoid,
    module_path: &str,
    resolved: &ResolvedModule,
    import_map: &HashMap<String, String>,
) -> bool {
    match op {
        AstOpVoid::CallVoid { callee, .. } => {
            callee_is_unsafe(callee, module_path, resolved, import_map)
        }
        _ => false,
    }
}

fn callee_is_unsafe(
    callee: &str,
    module_path: &str,
    resolved: &ResolvedModule,
    import_map: &HashMap<String, String>,
) -> bool {
    let sid = resolve_fn_sid(callee, module_path, resolved, import_map);
    resolved.unsafe_fn_sids.contains(&sid)
}

fn lower_op(
    op: &AstOp,
    module_path: &str,
    resolved: &ResolvedModule,
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
    resolved: &ResolvedModule,
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
        AstOpVoid::CallVoidIndirect { callee, args } => HirOpVoid::CallVoidIndirect {
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
    resolved: &ResolvedModule,
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
                format!("{}.{}", path, name)
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
            | AstDecl::GpuFn(magpie_ast::AstGpuFnDecl { inner: f, .. }) => {
                insert_fn_symbol(table, module_path, &f.name, false, decl.span, diag);
            }
            AstDecl::UnsafeFn(f) => {
                insert_fn_symbol(table, module_path, &f.name, true, decl.span, diag);
            }
            AstDecl::Extern(ext) => {
                for item in &ext.items {
                    insert_fn_symbol(table, module_path, &item.name, false, decl.span, diag);
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
    is_unsafe: bool,
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
            is_unsafe,
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
    resolved: &ResolvedModule,
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

fn resolve_type_sid(name: &str, module_path: &str, resolved: &ResolvedModule) -> Sid {
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

fn check_impl_orphan_rule(
    module_path: &str,
    impl_decl: &AstImplDecl,
    symbol_table: &SymbolTable,
    import_map: &HashMap<String, String>,
    diag: &mut DiagnosticBag,
) {
    let trait_is_local = symbol_table.sigs.contains_key(&impl_decl.trait_name);
    let type_owner =
        impl_target_owner_module(module_path, &impl_decl.for_type, symbol_table, import_map);
    let is_orphan = type_owner
        .as_deref()
        .is_some_and(|owner| owner != module_path)
        && !trait_is_local;

    if is_orphan {
        let target = impl_target_display_name(&impl_decl.for_type);
        emit_error(
            diag,
            "MPT1200",
            None,
            format!(
                "orphan impl: trait '{}' for foreign type '{}'. Impl must be in module '{}' or define the trait locally.",
                impl_decl.trait_name,
                target,
                type_owner.unwrap_or_else(|| "<unknown>".to_string())
            ),
        );
    }
}

fn impl_target_owner_module(
    module_path: &str,
    ty: &AstType,
    symbol_table: &SymbolTable,
    import_map: &HashMap<String, String>,
) -> Option<String> {
    match &ty.base {
        AstBaseType::Named { path, name, .. } => {
            if let Some(path) = path {
                return Some(path.to_string());
            }
            if symbol_table.types.contains_key(name) {
                return Some(module_path.to_string());
            }
            if let Some(imported) = import_map.get(name) {
                return imported
                    .rsplit_once('.')
                    .map(|(module, _)| module.to_string());
            }
            Some(module_path.to_string())
        }
        AstBaseType::Prim(_) | AstBaseType::Builtin(_) | AstBaseType::Callable { .. } => None,
        AstBaseType::RawPtr(_) => None,
    }
}

fn ast_type_returns_rawptr(ty: &AstType) -> bool {
    matches!(ty.base, AstBaseType::RawPtr(_))
}

fn impl_target_display_name(ty: &AstType) -> String {
    match &ty.base {
        AstBaseType::Named { path, name, .. } => match path {
            Some(path) if !path.segments.is_empty() => format!("{}.{}", path, name),
            _ => name.clone(),
        },
        AstBaseType::Prim(name) => name.clone(),
        AstBaseType::Builtin(builtin) => match builtin {
            AstBuiltinType::Str => "Str".to_string(),
            AstBuiltinType::Array(_) => "Array".to_string(),
            AstBuiltinType::Map(_, _) => "Map".to_string(),
            AstBuiltinType::TOption(_) => "TOption".to_string(),
            AstBuiltinType::TResult(_, _) => "TResult".to_string(),
            AstBuiltinType::TStrBuilder => "TStrBuilder".to_string(),
            AstBuiltinType::TMutex(_) => "TMutex".to_string(),
            AstBuiltinType::TRwLock(_) => "TRwLock".to_string(),
            AstBuiltinType::TCell(_) => "TCell".to_string(),
            AstBuiltinType::TFuture(_) => "TFuture".to_string(),
            AstBuiltinType::TChannelSend(_) => "TChannelSend".to_string(),
            AstBuiltinType::TChannelRecv(_) => "TChannelRecv".to_string(),
        },
        AstBaseType::Callable { sig_ref } => format!("TCallable<{}>", sig_ref),
        AstBaseType::RawPtr(_) => "rawptr".to_string(),
    }
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
        rag_bundle: Vec::new(),
        related_docs: Vec::new(),
    });
}

pub fn typecheck_module(
    module: &HirModule,
    type_ctx: &TypeCtx,
    sym: &SymbolTable,
    diag: &mut DiagnosticBag,
) -> Result<(), ()> {
    let before = diag.error_count();

    let fn_by_sid: HashMap<String, &FnSymbol> = sym
        .functions
        .values()
        .map(|f| (f.sid.0.clone(), f))
        .collect();
    let struct_fields = sema_struct_fields_by_sid(module);
    let enum_variants = sema_enum_variants_by_sid(module);

    for func in &module.functions {
        let local_types = sema_collect_local_types(func);

        for block in &func.blocks {
            for instr in &block.instrs {
                match &instr.op {
                    HirOp::IAdd { lhs, rhs }
                    | HirOp::ISub { lhs, rhs }
                    | HirOp::IMul { lhs, rhs }
                    | HirOp::ISDiv { lhs, rhs }
                    | HirOp::IUDiv { lhs, rhs }
                    | HirOp::ISRem { lhs, rhs }
                    | HirOp::IURem { lhs, rhs }
                    | HirOp::IAddWrap { lhs, rhs }
                    | HirOp::ISubWrap { lhs, rhs }
                    | HirOp::IMulWrap { lhs, rhs }
                    | HirOp::IAddChecked { lhs, rhs }
                    | HirOp::ISubChecked { lhs, rhs }
                    | HirOp::IMulChecked { lhs, rhs }
                    | HirOp::IAnd { lhs, rhs }
                    | HirOp::IOr { lhs, rhs }
                    | HirOp::IXor { lhs, rhs }
                    | HirOp::IShl { lhs, rhs }
                    | HirOp::ILshr { lhs, rhs }
                    | HirOp::IAshr { lhs, rhs }
                    | HirOp::ICmp { lhs, rhs, .. } => {
                        sema_check_binary_numeric("integer", lhs, rhs, &local_types, type_ctx, diag)
                    }
                    HirOp::FAdd { lhs, rhs }
                    | HirOp::FSub { lhs, rhs }
                    | HirOp::FMul { lhs, rhs }
                    | HirOp::FDiv { lhs, rhs }
                    | HirOp::FRem { lhs, rhs }
                    | HirOp::FAddFast { lhs, rhs }
                    | HirOp::FSubFast { lhs, rhs }
                    | HirOp::FMulFast { lhs, rhs }
                    | HirOp::FDivFast { lhs, rhs }
                    | HirOp::FCmp { lhs, rhs, .. } => {
                        sema_check_binary_numeric("float", lhs, rhs, &local_types, type_ctx, diag)
                    }
                    HirOp::Call {
                        callee_sid,
                        inst,
                        args,
                    } => {
                        if let Some(callee) = fn_by_sid.get(&callee_sid.0) {
                            if callee.params.len() != args.len() {
                                emit_error(
                                    diag,
                                    "MPT2001",
                                    None,
                                    format!(
                                        "call arity mismatch: callee expects {} args, got {}.",
                                        callee.params.len(),
                                        args.len()
                                    ),
                                );
                            }

                            for (idx, arg) in args.iter().enumerate() {
                                let Some(arg_ty) = sema_value_type(arg, &local_types) else {
                                    emit_error(
                                        diag,
                                        "MPT2002",
                                        None,
                                        format!("call argument {} has unknown type.", idx),
                                    );
                                    continue;
                                };
                                if let Some(param_ty) = callee.params.get(idx) {
                                    if arg_ty != *param_ty {
                                        emit_error(
                                            diag,
                                            "MPT2003",
                                            None,
                                            format!(
                                                "call argument {} type mismatch: expected {}, got {}.",
                                                idx,
                                                type_id_str(*param_ty, type_ctx),
                                                type_id_str(arg_ty, type_ctx)
                                            ),
                                        );
                                    }
                                }
                            }
                        } else {
                            emit_error(
                                diag,
                                "MPT2004",
                                None,
                                format!(
                                    "call target '{}' not found in symbol table.",
                                    callee_sid.0
                                ),
                            );
                        }

                        for targ in inst {
                            if type_ctx.lookup(*targ).is_none() {
                                emit_error(
                                    diag,
                                    "MPT2005",
                                    None,
                                    format!("invalid generic type argument '{}'.", targ.0),
                                );
                            }
                        }
                    }
                    HirOp::GetField { obj, field } => {
                        let Some(obj_ty) = sema_value_type(obj, &local_types) else {
                            emit_error(
                                diag,
                                "MPT2006",
                                None,
                                "getfield object has unknown type.".to_string(),
                            );
                            continue;
                        };

                        let Some(struct_sid) = sema_borrowed_struct_sid(obj_ty, type_ctx) else {
                            emit_error(
                                diag,
                                "MPT2007",
                                None,
                                "getfield requires object type `borrow TStruct` or `mutborrow TStruct`."
                                    .to_string(),
                            );
                            continue;
                        };

                        let Some(fields) = struct_fields.get(&struct_sid) else {
                            emit_error(
                                diag,
                                "MPT2008",
                                None,
                                format!("getfield target type '{}' is not a struct.", struct_sid),
                            );
                            continue;
                        };

                        if !fields.contains_key(field) {
                            emit_error(
                                diag,
                                "MPT2009",
                                None,
                                format!("struct '{}' has no field '{}'.", struct_sid, field),
                            );
                        }
                    }
                    HirOp::EnumNew { variant, args } => {
                        sema_check_enum_new(
                            instr.ty,
                            variant,
                            args,
                            &local_types,
                            type_ctx,
                            &enum_variants,
                            diag,
                        );
                    }
                    HirOp::New { ty, fields } => {
                        sema_check_new_struct(
                            *ty,
                            fields,
                            &local_types,
                            type_ctx,
                            &struct_fields,
                            diag,
                        );
                    }
                    HirOp::Cast { to, v } => {
                        let Some(from_ty) = sema_value_type(v, &local_types) else {
                            emit_error(
                                diag,
                                "MPT2010",
                                None,
                                "cast operand has unknown type.".to_string(),
                            );
                            continue;
                        };
                        if !sema_is_primitive_type(from_ty, type_ctx)
                            || !sema_is_primitive_type(*to, type_ctx)
                        {
                            emit_error(
                                diag,
                                "MPT2011",
                                None,
                                format!(
                                    "cast is only allowed between primitive types (from {} to {}).",
                                    type_id_str(from_ty, type_ctx),
                                    type_id_str(*to, type_ctx)
                                ),
                            );
                        }
                    }
                    HirOp::StrParseI64 { s } => {
                        sema_check_parse_op_shape(
                            "str.parse_i64",
                            instr.ty,
                            fixed_type_ids::I64,
                            s,
                            &local_types,
                            type_ctx,
                            diag,
                        );
                    }
                    HirOp::StrParseU64 { s } => {
                        sema_check_parse_op_shape(
                            "str.parse_u64",
                            instr.ty,
                            fixed_type_ids::U64,
                            s,
                            &local_types,
                            type_ctx,
                            diag,
                        );
                    }
                    HirOp::StrParseF64 { s } => {
                        sema_check_parse_op_shape(
                            "str.parse_f64",
                            instr.ty,
                            fixed_type_ids::F64,
                            s,
                            &local_types,
                            type_ctx,
                            diag,
                        );
                    }
                    HirOp::StrParseBool { s } => {
                        sema_check_parse_op_shape(
                            "str.parse_bool",
                            instr.ty,
                            fixed_type_ids::BOOL,
                            s,
                            &local_types,
                            type_ctx,
                            diag,
                        );
                    }
                    HirOp::JsonEncode { ty, v } => {
                        let Some(v_ty) = sema_value_type(v, &local_types) else {
                            emit_error(
                                diag,
                                "MPT2035",
                                None,
                                "json.encode value has unknown type.".to_string(),
                            );
                            continue;
                        };
                        if v_ty != *ty {
                            emit_error(
                                diag,
                                "MPT2035",
                                None,
                                format!(
                                    "json.encode<T> requires value type T (expected {}, got {}).",
                                    type_id_str(*ty, type_ctx),
                                    type_id_str(v_ty, type_ctx)
                                ),
                            );
                        }

                        if !sema_is_legacy_or_result_shape(instr.ty, fixed_type_ids::STR, type_ctx)
                        {
                            emit_error(
                                diag,
                                "MPT2033",
                                None,
                                format!(
                                    "json.encode result type must be Str (legacy) or TResult<Str, E>; got {}.",
                                    type_id_str(instr.ty, type_ctx)
                                ),
                            );
                        }
                    }
                    HirOp::JsonDecode { ty, s } => {
                        let Some(src_ty) = sema_value_type(s, &local_types) else {
                            emit_error(
                                diag,
                                "MPT2034",
                                None,
                                "json.decode input has unknown type.".to_string(),
                            );
                            continue;
                        };
                        if !sema_is_str_handle(src_ty, type_ctx) {
                            emit_error(
                                diag,
                                "MPT2034",
                                None,
                                format!(
                                    "json.decode input must be Str/borrow Str, got {}.",
                                    type_id_str(src_ty, type_ctx)
                                ),
                            );
                        }

                        if !sema_is_legacy_or_result_rawptr_shape(instr.ty, type_ctx) {
                            emit_error(
                                diag,
                                "MPT2033",
                                None,
                                format!(
                                    "json.decode<T> result type must be rawptr<...> (legacy) or TResult<rawptr<...>, E>; got {} (decode target T is {}).",
                                    type_id_str(instr.ty, type_ctx),
                                    type_id_str(*ty, type_ctx)
                                ),
                            );
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    if diag.error_count() > before {
        Err(())
    } else {
        Ok(())
    }
}

pub fn check_trait_impls(
    module: &HirModule,
    type_ctx: &TypeCtx,
    sym: &SymbolTable,
    impl_decls: &[AstImplDecl],
    resolved_imports: &[(String, FQN)],
    diag: &mut DiagnosticBag,
) -> Result<(), ()> {
    let before = diag.error_count();

    let local_type_names: HashMap<String, Sid> = module
        .type_decls
        .iter()
        .map(|decl| match decl {
            HirTypeDecl::Struct { sid, name, .. } => (name.clone(), sid.clone()),
            HirTypeDecl::Enum { sid, name, .. } => (name.clone(), sid.clone()),
        })
        .collect();
    let fn_names: HashSet<&str> = sym.functions.keys().map(String::as_str).collect();
    let fn_by_sid: HashMap<String, &HirFunction> = module
        .functions
        .iter()
        .map(|func| (func.sid.0.clone(), func))
        .collect();
    let import_map = resolved_imports
        .iter()
        .cloned()
        .collect::<HashMap<String, String>>();

    let mut impls: HashSet<(String, String)> = HashSet::new();
    sema_seed_builtin_trait_impls(&mut impls);

    for impl_decl in impl_decls {
        let trait_name = impl_decl.trait_name.as_str();
        if !matches!(trait_name, "hash" | "eq" | "ord") {
            continue;
        }

        let target_name = impl_trait_target_name(&impl_decl.for_type);
        let fn_sid = sema_resolve_impl_fn_sid(&impl_decl.fn_ref, &module.path, sym, &import_map);
        let Some(func) = fn_by_sid.get(&fn_sid.0) else {
            emit_error(
                diag,
                "MPT2032",
                None,
                format!(
                    "impl '{}' for '{}' references unknown local function '{}'.",
                    trait_name, target_name, impl_decl.fn_ref
                ),
            );
            continue;
        };

        sema_check_trait_sig(
            func,
            trait_name,
            &target_name,
            &local_type_names,
            type_ctx,
            diag,
        );

        if let Some(key) =
            sema_trait_key_from_ast_type(&impl_decl.for_type, &local_type_names, type_ctx)
        {
            impls.insert((trait_name.to_string(), key));
        }
    }

    for func in &module.functions {
        let trait_kind = func
            .name
            .strip_prefix("hash_")
            .map(|target| ("hash", target))
            .or_else(|| func.name.strip_prefix("eq_").map(|target| ("eq", target)))
            .or_else(|| func.name.strip_prefix("ord_").map(|target| ("ord", target)));

        let Some((trait_name, target_name)) = trait_kind else {
            continue;
        };
        if target_name.is_empty() {
            continue;
        }

        if !fn_names.contains(func.name.as_str()) {
            continue;
        }

        sema_check_trait_sig(
            func,
            trait_name,
            target_name,
            &local_type_names,
            type_ctx,
            diag,
        );

        if !local_type_names.contains_key(target_name) && !sema_is_lang_owned_type_name(target_name)
        {
            emit_error(
                diag,
                "MPT1200",
                None,
                format!(
                    "orphan impl: trait '{}' for foreign type '{}'.",
                    trait_name, target_name
                ),
            );
        }

        if let Some(key) = sema_trait_key_from_type_name(target_name, &local_type_names, type_ctx) {
            impls.insert((trait_name.to_string(), key));
        }
    }

    for func in &module.functions {
        let local_types = sema_collect_local_types(func);

        for block in &func.blocks {
            for instr in &block.instrs {
                match &instr.op {
                    HirOp::ArrContains { arr, .. } => {
                        if let Some(elem_ty) = sema_array_elem_type(arr, &local_types, type_ctx) {
                            sema_require_trait_impl("eq", elem_ty, type_ctx, &impls, diag);
                        }
                    }
                    HirOp::ArrSort { arr } => {
                        if let Some(elem_ty) = sema_array_elem_type(arr, &local_types, type_ctx) {
                            sema_require_trait_impl("ord", elem_ty, type_ctx, &impls, diag);
                        }
                    }
                    HirOp::MapNew { key_ty, .. } => {
                        sema_require_trait_impl("hash", *key_ty, type_ctx, &impls, diag);
                        sema_require_trait_impl("eq", *key_ty, type_ctx, &impls, diag);
                    }
                    _ => {}
                }
            }

            for op in &block.void_ops {
                if let HirOpVoid::ArrSort { arr } = op {
                    if let Some(elem_ty) = sema_array_elem_type(arr, &local_types, type_ctx) {
                        sema_require_trait_impl("ord", elem_ty, type_ctx, &impls, diag);
                    }
                }
            }
        }
    }

    if diag.error_count() > before {
        Err(())
    } else {
        Ok(())
    }
}

pub fn check_v01_restrictions(
    module: &HirModule,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) -> Result<(), ()> {
    let before = diag.error_count();

    for (_, kind) in &type_ctx.types {
        if matches!(
            kind,
            TypeKind::Arr { .. } | TypeKind::Vec { .. } | TypeKind::Tuple { .. }
        ) {
            emit_error(
                diag,
                "MPT1021",
                None,
                format!(
                    "aggregate type '{}' is deferred in v0.1.",
                    type_str(kind, type_ctx)
                ),
            );
        }
    }

    let value_sids: HashSet<String> = type_ctx
        .types
        .iter()
        .filter_map(|(_, kind)| match kind {
            TypeKind::ValueStruct { sid } => Some(sid.0.clone()),
            _ => None,
        })
        .collect();

    for decl in &module.type_decls {
        match decl {
            HirTypeDecl::Enum { sid, name, .. } => {
                if value_sids.contains(&sid.0) {
                    emit_error(
                        diag,
                        "MPT1020",
                        None,
                        format!("value enum '{}' is deferred in v0.1.", name),
                    );
                }
            }
            HirTypeDecl::Struct { sid, name, fields } => {
                if !value_sids.contains(&sid.0) {
                    continue;
                }
                for (field_name, field_ty) in fields {
                    let mut visiting = HashSet::new();
                    if sema_contains_heap_handle(*field_ty, type_ctx, &mut visiting) {
                        emit_error(
                            diag,
                            "MPT1005",
                            None,
                            format!(
                                "value struct '{}' field '{}' contains heap handle type '{}'.",
                                name,
                                field_name,
                                type_id_str(*field_ty, type_ctx)
                            ),
                        );
                    }
                }
            }
        }
    }

    let known_fn_sids: HashSet<&str> = module.functions.iter().map(|f| f.sid.0.as_str()).collect();
    for func in &module.functions {
        for block in &func.blocks {
            for instr in &block.instrs {
                if let HirOp::SuspendCall { callee_sid, .. } = &instr.op {
                    if !known_fn_sids.contains(callee_sid.0.as_str()) {
                        emit_error(
                            diag,
                            "MPT1030",
                            None,
                            format!(
                                "`suspend.call` on non-function target '{}' (TCallable form) is forbidden in v0.1.",
                                callee_sid.0
                            ),
                        );
                    }
                }
            }
        }
    }

    if diag.error_count() > before {
        Err(())
    } else {
        Ok(())
    }
}

fn sema_collect_local_types(func: &HirFunction) -> HashMap<LocalId, TypeId> {
    let mut locals = HashMap::new();
    for (local, ty) in &func.params {
        locals.insert(*local, *ty);
    }
    for block in &func.blocks {
        for instr in &block.instrs {
            locals.insert(instr.dst, instr.ty);
        }
    }
    locals
}

fn sema_value_type(v: &HirValue, locals: &HashMap<LocalId, TypeId>) -> Option<TypeId> {
    match v {
        HirValue::Local(id) => locals.get(id).copied(),
        HirValue::Const(c) => Some(c.ty),
    }
}

fn sema_is_primitive_type(ty: TypeId, type_ctx: &TypeCtx) -> bool {
    matches!(type_ctx.lookup(ty), Some(TypeKind::Prim(_)))
}

fn sema_is_str_handle(ty: TypeId, type_ctx: &TypeCtx) -> bool {
    matches!(
        type_ctx.lookup(ty),
        Some(TypeKind::HeapHandle {
            base: HeapBase::BuiltinStr,
            ..
        })
    )
}

fn sema_is_legacy_or_result_shape(dst_ty: TypeId, expected_ok: TypeId, type_ctx: &TypeCtx) -> bool {
    if dst_ty == expected_ok {
        return true;
    }
    matches!(
        type_ctx.lookup(dst_ty),
        Some(TypeKind::BuiltinResult { ok, .. }) if *ok == expected_ok
    )
}

fn sema_is_raw_ptr(ty: TypeId, type_ctx: &TypeCtx) -> bool {
    matches!(type_ctx.lookup(ty), Some(TypeKind::RawPtr { .. }))
}

fn sema_is_legacy_or_result_rawptr_shape(dst_ty: TypeId, type_ctx: &TypeCtx) -> bool {
    if sema_is_raw_ptr(dst_ty, type_ctx) {
        return true;
    }
    matches!(
        type_ctx.lookup(dst_ty),
        Some(TypeKind::BuiltinResult { ok, .. }) if sema_is_raw_ptr(*ok, type_ctx)
    )
}

fn sema_check_parse_op_shape(
    op_name: &str,
    dst_ty: TypeId,
    expected_ok: TypeId,
    s: &HirValue,
    local_types: &HashMap<LocalId, TypeId>,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) {
    let Some(src_ty) = sema_value_type(s, local_types) else {
        emit_error(
            diag,
            "MPT2034",
            None,
            format!("{} input has unknown type.", op_name),
        );
        return;
    };
    if !sema_is_str_handle(src_ty, type_ctx) {
        emit_error(
            diag,
            "MPT2034",
            None,
            format!(
                "{} input must be Str/borrow Str, got {}.",
                op_name,
                type_id_str(src_ty, type_ctx)
            ),
        );
    }

    if !sema_is_legacy_or_result_shape(dst_ty, expected_ok, type_ctx) {
        emit_error(
            diag,
            "MPT2033",
            None,
            format!(
                "{} result type must be {} (legacy) or TResult<{}, E>; got {}.",
                op_name,
                type_id_str(expected_ok, type_ctx),
                type_id_str(expected_ok, type_ctx),
                type_id_str(dst_ty, type_ctx)
            ),
        );
    }
}

fn sema_check_binary_numeric(
    family: &str,
    lhs: &HirValue,
    rhs: &HirValue,
    local_types: &HashMap<LocalId, TypeId>,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) {
    let Some(lhs_ty) = sema_value_type(lhs, local_types) else {
        emit_error(
            diag,
            "MPT2012",
            None,
            format!("{} binary op lhs has unknown type.", family),
        );
        return;
    };
    let Some(rhs_ty) = sema_value_type(rhs, local_types) else {
        emit_error(
            diag,
            "MPT2013",
            None,
            format!("{} binary op rhs has unknown type.", family),
        );
        return;
    };

    if lhs_ty != rhs_ty {
        emit_error(
            diag,
            "MPT2014",
            None,
            format!(
                "{} binary op requires both operands to have same type (lhs={}, rhs={}).",
                family,
                type_id_str(lhs_ty, type_ctx),
                type_id_str(rhs_ty, type_ctx)
            ),
        );
        return;
    }

    let ok = match type_ctx.lookup(lhs_ty) {
        Some(TypeKind::Prim(p)) if family == "integer" => p.is_integer(),
        Some(TypeKind::Prim(p)) if family == "float" => p.is_float(),
        _ => false,
    };
    if !ok {
        emit_error(
            diag,
            "MPT2015",
            None,
            format!(
                "{} binary op requires {} primitive operands, got '{}'.",
                family,
                family,
                type_id_str(lhs_ty, type_ctx)
            ),
        );
    }
}

fn sema_struct_fields_by_sid(module: &HirModule) -> HashMap<String, HashMap<String, TypeId>> {
    let mut out = HashMap::new();
    for decl in &module.type_decls {
        if let HirTypeDecl::Struct { sid, fields, .. } = decl {
            out.insert(sid.0.clone(), fields.iter().cloned().collect());
        }
    }
    out
}

fn sema_enum_variants_by_sid(
    module: &HirModule,
) -> HashMap<String, HashMap<String, HashMap<String, TypeId>>> {
    let mut out = HashMap::new();
    for decl in &module.type_decls {
        if let HirTypeDecl::Enum { sid, variants, .. } = decl {
            let mut by_variant = HashMap::new();
            for v in variants {
                by_variant.insert(v.name.clone(), v.fields.iter().cloned().collect());
            }
            out.insert(sid.0.clone(), by_variant);
        }
    }
    out
}

fn sema_borrowed_struct_sid(ty: TypeId, type_ctx: &TypeCtx) -> Option<String> {
    match type_ctx.lookup(ty) {
        Some(TypeKind::HeapHandle {
            hk: HandleKind::Borrow | HandleKind::MutBorrow,
            base: HeapBase::UserType { type_sid, .. },
        }) => Some(type_sid.0.clone()),
        _ => None,
    }
}

fn sema_check_field_arg_list(
    kind_name: &str,
    owner_name: &str,
    args: &[(String, HirValue)],
    expected_fields: &HashMap<String, TypeId>,
    local_types: &HashMap<LocalId, TypeId>,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) {
    let mut seen = HashSet::new();
    for (field, value) in args {
        if !seen.insert(field.clone()) {
            emit_error(
                diag,
                "MPT2016",
                None,
                format!(
                    "{} '{}' has duplicate field argument '{}'.",
                    kind_name, owner_name, field
                ),
            );
            continue;
        }

        let Some(expected_ty) = expected_fields.get(field).copied() else {
            emit_error(
                diag,
                "MPT2017",
                None,
                format!("{} '{}' has no field '{}'.", kind_name, owner_name, field),
            );
            continue;
        };

        let Some(actual_ty) = sema_value_type(value, local_types) else {
            emit_error(
                diag,
                "MPT2018",
                None,
                format!(
                    "{} '{}' field '{}' value has unknown type.",
                    kind_name, owner_name, field
                ),
            );
            continue;
        };

        if actual_ty != expected_ty {
            emit_error(
                diag,
                "MPT2019",
                None,
                format!(
                    "{} '{}' field '{}' type mismatch: expected {}, got {}.",
                    kind_name,
                    owner_name,
                    field,
                    type_id_str(expected_ty, type_ctx),
                    type_id_str(actual_ty, type_ctx)
                ),
            );
        }
    }

    for field in expected_fields.keys() {
        if !seen.contains(field) {
            emit_error(
                diag,
                "MPT2020",
                None,
                format!(
                    "{} '{}' is missing required field '{}'.",
                    kind_name, owner_name, field
                ),
            );
        }
    }
}

fn sema_check_new_struct(
    new_ty: TypeId,
    fields: &[(String, HirValue)],
    local_types: &HashMap<LocalId, TypeId>,
    type_ctx: &TypeCtx,
    struct_fields: &HashMap<String, HashMap<String, TypeId>>,
    diag: &mut DiagnosticBag,
) {
    let sid = match type_ctx.lookup(new_ty) {
        Some(TypeKind::HeapHandle {
            base: HeapBase::UserType { type_sid, .. },
            ..
        }) => type_sid.0.clone(),
        Some(TypeKind::ValueStruct { sid }) => sid.0.clone(),
        _ => {
            emit_error(
                diag,
                "MPT2021",
                None,
                format!(
                    "`new` target must be a struct type, got '{}'.",
                    type_id_str(new_ty, type_ctx)
                ),
            );
            return;
        }
    };

    let Some(expected) = struct_fields.get(&sid) else {
        emit_error(
            diag,
            "MPT2022",
            None,
            format!("`new` target '{}' is not a known struct.", sid),
        );
        return;
    };

    sema_check_field_arg_list(
        "struct",
        &sid,
        fields,
        expected,
        local_types,
        type_ctx,
        diag,
    );
}

fn sema_check_enum_new(
    enum_ty: TypeId,
    variant: &str,
    args: &[(String, HirValue)],
    local_types: &HashMap<LocalId, TypeId>,
    type_ctx: &TypeCtx,
    enum_variants: &HashMap<String, HashMap<String, HashMap<String, TypeId>>>,
    diag: &mut DiagnosticBag,
) {
    match type_ctx.lookup(enum_ty) {
        Some(TypeKind::BuiltinOption { inner }) => {
            let expected: HashMap<String, TypeId> = match variant {
                "None" => HashMap::new(),
                "Some" => HashMap::from([("v".to_string(), *inner)]),
                _ => {
                    emit_error(
                        diag,
                        "MPT2023",
                        None,
                        format!("variant '{}' is invalid for TOption.", variant),
                    );
                    return;
                }
            };
            sema_check_field_arg_list(
                "variant",
                "TOption",
                args,
                &expected,
                local_types,
                type_ctx,
                diag,
            );
        }
        Some(TypeKind::BuiltinResult { ok, err }) => {
            let expected: HashMap<String, TypeId> = match variant {
                "Ok" => HashMap::from([("v".to_string(), *ok)]),
                "Err" => HashMap::from([("e".to_string(), *err)]),
                _ => {
                    emit_error(
                        diag,
                        "MPT2024",
                        None,
                        format!("variant '{}' is invalid for TResult.", variant),
                    );
                    return;
                }
            };
            sema_check_field_arg_list(
                "variant",
                "TResult",
                args,
                &expected,
                local_types,
                type_ctx,
                diag,
            );
        }
        Some(TypeKind::HeapHandle {
            base: HeapBase::UserType { type_sid, .. },
            ..
        })
        | Some(TypeKind::ValueStruct { sid: type_sid }) => {
            let sid = type_sid.0.clone();
            let Some(variants) = enum_variants.get(&sid) else {
                emit_error(
                    diag,
                    "MPT2025",
                    None,
                    format!(
                        "`enum.new` result type '{}' is not an enum.",
                        type_id_str(enum_ty, type_ctx)
                    ),
                );
                return;
            };

            let Some(expected) = variants.get(variant) else {
                emit_error(
                    diag,
                    "MPT2026",
                    None,
                    format!("enum '{}' has no variant '{}'.", sid, variant),
                );
                return;
            };

            sema_check_field_arg_list("variant", &sid, args, expected, local_types, type_ctx, diag);
        }
        _ => {
            emit_error(
                diag,
                "MPT2027",
                None,
                format!(
                    "`enum.new` result type must be enum, got '{}'.",
                    type_id_str(enum_ty, type_ctx)
                ),
            );
        }
    }
}

fn sema_check_trait_sig(
    func: &HirFunction,
    trait_name: &str,
    target_name: &str,
    local_type_names: &HashMap<String, Sid>,
    type_ctx: &TypeCtx,
    diag: &mut DiagnosticBag,
) {
    let (expected_params, expected_ret) = match trait_name {
        "hash" => (1usize, fixed_type_ids::U64),
        "eq" => (2usize, fixed_type_ids::BOOL),
        "ord" => (2usize, fixed_type_ids::I32),
        _ => return,
    };

    if func.params.len() != expected_params {
        emit_error(
            diag,
            "MPT2028",
            None,
            format!(
                "trait impl '{}' must have {} parameter(s), got {}.",
                func.name,
                expected_params,
                func.params.len()
            ),
        );
    }

    if func.ret_ty != expected_ret {
        emit_error(
            diag,
            "MPT2029",
            None,
            format!(
                "trait impl '{}' return type mismatch: expected {}, got {}.",
                func.name,
                type_id_str(expected_ret, type_ctx),
                type_id_str(func.ret_ty, type_ctx)
            ),
        );
    }

    let Some((_, first_ty)) = func.params.first() else {
        return;
    };

    if !sema_is_borrow_for_trait_target(*first_ty, target_name, local_type_names, type_ctx) {
        emit_error(
            diag,
            "MPT2030",
            None,
            format!(
                "trait impl '{}' first parameter must be `borrow {}`.",
                func.name, target_name
            ),
        );
    }

    if let Some((_, second_ty)) = func.params.get(1) {
        if *first_ty != *second_ty {
            emit_error(
                diag,
                "MPT2031",
                None,
                format!(
                    "trait impl '{}' parameters must both be `borrow {}`.",
                    func.name, target_name
                ),
            );
        }
    }
}

fn sema_is_borrow_for_trait_target(
    ty: TypeId,
    target_name: &str,
    local_type_names: &HashMap<String, Sid>,
    type_ctx: &TypeCtx,
) -> bool {
    match type_ctx.lookup(ty) {
        Some(TypeKind::HeapHandle {
            hk: HandleKind::Borrow,
            base: HeapBase::UserType { type_sid, .. },
        }) => local_type_names
            .get(target_name)
            .map(|sid| sid == type_sid)
            .unwrap_or(false),
        Some(TypeKind::HeapHandle {
            hk: HandleKind::Borrow,
            base: HeapBase::BuiltinStr,
        }) => target_name == "Str",
        Some(TypeKind::Prim(p)) => target_name == prim_type_str(*p),
        _ => false,
    }
}

fn sema_seed_builtin_trait_impls(impls: &mut HashSet<(String, String)>) {
    let prims = [
        "bool", "i8", "i16", "i32", "i64", "i128", "u1", "u8", "u16", "u32", "u64", "u128", "f16",
        "f32", "f64",
    ];
    for p in prims {
        let key = format!("prim:{}", p);
        impls.insert(("hash".to_string(), key.clone()));
        impls.insert(("eq".to_string(), key.clone()));
        impls.insert(("ord".to_string(), key));
    }
    impls.insert(("hash".to_string(), "str".to_string()));
    impls.insert(("eq".to_string(), "str".to_string()));
    impls.insert(("ord".to_string(), "str".to_string()));
}

fn sema_trait_key_from_type_name(
    type_name: &str,
    local_type_names: &HashMap<String, Sid>,
    _type_ctx: &TypeCtx,
) -> Option<String> {
    if let Some(sid) = local_type_names.get(type_name) {
        return Some(format!("user:{}", sid.0));
    }
    if type_name == "Str" {
        return Some("str".to_string());
    }
    if prim_type_from_name(type_name).is_some() {
        return Some(format!("prim:{}", type_name));
    }
    None
}

fn sema_trait_key_from_ast_type(
    ty: &AstType,
    local_type_names: &HashMap<String, Sid>,
    type_ctx: &TypeCtx,
) -> Option<String> {
    match &ty.base {
        AstBaseType::Named { name, .. } => {
            sema_trait_key_from_type_name(name, local_type_names, type_ctx)
        }
        AstBaseType::Prim(name) => sema_trait_key_from_type_name(name, local_type_names, type_ctx),
        AstBaseType::Builtin(AstBuiltinType::Str) => Some("str".to_string()),
        _ => None,
    }
}

fn impl_trait_target_name(ty: &AstType) -> String {
    match &ty.base {
        AstBaseType::Named { name, .. } => name.clone(),
        AstBaseType::Prim(name) => name.clone(),
        AstBaseType::Builtin(AstBuiltinType::Str) => "Str".to_string(),
        _ => impl_target_display_name(ty),
    }
}

fn sema_resolve_impl_fn_sid(
    fn_ref: &str,
    module_path: &str,
    symbol_table: &SymbolTable,
    import_map: &HashMap<String, String>,
) -> Sid {
    if let Some(sym) = symbol_table.functions.get(fn_ref) {
        return sym.sid.clone();
    }
    if let Some(fqn) = import_map.get(fn_ref) {
        return generate_sid('F', fqn);
    }
    if fn_ref.contains('.') {
        return generate_sid('F', fn_ref);
    }
    generate_sid('F', &format!("{}.{}", module_path, fn_ref))
}

fn sema_trait_key_from_type_id(ty: TypeId, type_ctx: &TypeCtx) -> Option<String> {
    match type_ctx.lookup(ty) {
        Some(TypeKind::Prim(p)) => Some(format!("prim:{}", prim_type_str(*p))),
        Some(TypeKind::HeapHandle {
            base: HeapBase::BuiltinStr,
            ..
        }) => Some("str".to_string()),
        Some(TypeKind::HeapHandle {
            base: HeapBase::UserType { type_sid, targs },
            ..
        }) => {
            if targs.is_empty() {
                Some(format!("user:{}", type_sid.0))
            } else {
                let mut arg_keys = Vec::new();
                for targ in targs {
                    arg_keys.push(
                        sema_trait_key_from_type_id(*targ, type_ctx)
                            .unwrap_or_else(|| format!("type#{}", targ.0)),
                    );
                }
                Some(format!("user:{}<{}>", type_sid.0, arg_keys.join(",")))
            }
        }
        Some(TypeKind::ValueStruct { sid }) => Some(format!("user:{}", sid.0)),
        _ => None,
    }
}

fn sema_require_trait_impl(
    trait_name: &str,
    ty: TypeId,
    type_ctx: &TypeCtx,
    impls: &HashSet<(String, String)>,
    diag: &mut DiagnosticBag,
) {
    let key = sema_trait_key_from_type_id(ty, type_ctx).unwrap_or_else(|| format!("type#{}", ty.0));
    if !impls.contains(&(trait_name.to_string(), key.clone())) {
        emit_error(
            diag,
            "MPT1023",
            None,
            format!(
                "missing required trait impl: `impl {} for {}`.",
                trait_name,
                type_id_str(ty, type_ctx)
            ),
        );
    }
}

fn sema_array_elem_type(
    arr_value: &HirValue,
    locals: &HashMap<LocalId, TypeId>,
    type_ctx: &TypeCtx,
) -> Option<TypeId> {
    let arr_ty = sema_value_type(arr_value, locals)?;
    match type_ctx.lookup(arr_ty) {
        Some(TypeKind::HeapHandle {
            base: HeapBase::BuiltinArray { elem },
            ..
        }) => Some(*elem),
        _ => None,
    }
}

fn sema_is_lang_owned_type_name(name: &str) -> bool {
    matches!(
        name,
        "Str"
            | "bool"
            | "unit"
            | "i1"
            | "i8"
            | "i16"
            | "i32"
            | "i64"
            | "i128"
            | "u1"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "u128"
            | "f16"
            | "f32"
            | "f64"
    )
}

fn sema_contains_heap_handle(
    ty: TypeId,
    type_ctx: &TypeCtx,
    visiting: &mut HashSet<TypeId>,
) -> bool {
    if !visiting.insert(ty) {
        return false;
    }
    let out = match type_ctx.lookup(ty) {
        Some(TypeKind::HeapHandle { .. }) => true,
        Some(TypeKind::BuiltinOption { inner }) => {
            sema_contains_heap_handle(*inner, type_ctx, visiting)
        }
        Some(TypeKind::BuiltinResult { ok, err }) => {
            sema_contains_heap_handle(*ok, type_ctx, visiting)
                || sema_contains_heap_handle(*err, type_ctx, visiting)
        }
        Some(TypeKind::Arr { elem, .. }) | Some(TypeKind::Vec { elem, .. }) => {
            sema_contains_heap_handle(*elem, type_ctx, visiting)
        }
        Some(TypeKind::Tuple { elems }) => elems
            .iter()
            .any(|elem| sema_contains_heap_handle(*elem, type_ctx, visiting)),
        _ => false,
    };
    visiting.remove(&ty);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use magpie_ast::{
        AstBlock, AstConstExpr, AstConstLit, AstDecl, AstExternItem, AstExternModule, AstFieldDecl,
        AstFile, AstFnDecl, AstHeader, AstImplDecl, AstInstr, AstOp, AstParam, AstStructDecl,
        AstTerminator, AstType, AstTypeParam, AstValueRef, ExportItem, ImportGroup, ImportItem,
        ModulePath, Span, Spanned,
    };

    fn sp<T>(node: T) -> Spanned<T> {
        Spanned::new(node, Span::dummy())
    }

    fn i32_ty() -> AstType {
        AstType {
            ownership: None,
            base: AstBaseType::Prim("i32".to_string()),
        }
    }

    fn rawptr_i32_ty() -> AstType {
        AstType {
            ownership: None,
            base: AstBaseType::RawPtr(Box::new(i32_ty())),
        }
    }

    fn const_i32(v: i128) -> AstConstExpr {
        AstConstExpr {
            ty: i32_ty(),
            lit: AstConstLit::Int(v),
        }
    }

    fn mk_fn_decl(
        name: &str,
        is_unsafe: bool,
        instrs: Vec<Spanned<AstInstr>>,
        terminator: AstTerminator,
    ) -> Spanned<AstDecl> {
        let f = AstFnDecl {
            name: name.to_string(),
            params: vec![],
            ret_ty: sp(i32_ty()),
            meta: None,
            blocks: vec![sp(AstBlock {
                label: 0,
                instrs,
                terminator: sp(terminator),
            })],
            doc: None,
        };
        if is_unsafe {
            sp(AstDecl::UnsafeFn(f))
        } else {
            sp(AstDecl::Fn(f))
        }
    }

    fn mk_module(decls: Vec<Spanned<AstDecl>>) -> AstFile {
        AstFile {
            header: sp(AstHeader {
                module_path: sp(ModulePath {
                    segments: vec!["demo".to_string(), "unsafe_checks".to_string()],
                }),
                exports: vec![],
                imports: vec![],
                digest: sp(String::new()),
            }),
            decls,
        }
    }

    #[test]
    fn test_generate_sid() {
        let sid = generate_sid('M', "demo.main");
        assert_eq!(sid.0.len(), 12);
        assert!(sid.0.starts_with("M:"));
        assert!(sid.is_valid());
        assert!(sid.0[2..]
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit()));
    }

    #[test]
    fn test_resolve_simple_module() {
        let i32_ty = AstType {
            ownership: None,
            base: AstBaseType::Prim("i32".to_string()),
        };

        let ast = AstFile {
            header: sp(AstHeader {
                module_path: sp(ModulePath {
                    segments: vec!["demo".to_string(), "main".to_string()],
                }),
                exports: vec![sp(ExportItem::Fn("main".to_string()))],
                imports: vec![],
                digest: sp(String::new()),
            }),
            decls: vec![sp(AstDecl::Fn(AstFnDecl {
                name: "main".to_string(),
                params: vec![],
                ret_ty: sp(i32_ty),
                meta: None,
                blocks: vec![],
                doc: None,
            }))],
        };

        let mut diag = DiagnosticBag::new(16);
        let resolved = resolve_modules(&[ast], &mut diag).expect("module should resolve");

        assert!(
            !diag.has_errors(),
            "unexpected diagnostics: {:?}",
            diag.diagnostics
        );
        assert_eq!(resolved.len(), 1);
        assert!(
            resolved[0].symbol_table.functions.contains_key("main"),
            "expected function symbol table entry for 'main'"
        );
    }

    #[test]
    fn test_orphan_impl_decl_rejected() {
        let local_struct = AstFile {
            header: sp(AstHeader {
                module_path: sp(ModulePath {
                    segments: vec!["pkg".to_string(), "types".to_string()],
                }),
                exports: vec![],
                imports: vec![],
                digest: sp(String::new()),
            }),
            decls: vec![sp(AstDecl::HeapStruct(AstStructDecl {
                name: "TForeign".to_string(),
                type_params: Vec::<AstTypeParam>::new(),
                fields: Vec::<AstFieldDecl>::new(),
                doc: None,
            }))],
        };

        let importer = AstFile {
            header: sp(AstHeader {
                module_path: sp(ModulePath {
                    segments: vec!["pkg".to_string(), "consumer".to_string()],
                }),
                exports: vec![],
                imports: vec![sp(ImportGroup {
                    module_path: ModulePath {
                        segments: vec!["pkg".to_string(), "types".to_string()],
                    },
                    items: vec![ImportItem::Type("TForeign".to_string())],
                })],
                digest: sp(String::new()),
            }),
            decls: vec![sp(AstDecl::Impl(AstImplDecl {
                trait_name: "hash".to_string(),
                for_type: AstType {
                    ownership: None,
                    base: AstBaseType::Named {
                        path: None,
                        name: "TForeign".to_string(),
                        targs: Vec::new(),
                    },
                },
                fn_ref: "pkg.consumer.hash_foreign".to_string(),
            }))],
        };

        let mut diag = DiagnosticBag::new(32);
        let resolved = resolve_modules(&[local_struct, importer], &mut diag);
        assert!(
            resolved.is_err(),
            "expected orphan impl rule to reject module set"
        );
        assert!(
            diag.diagnostics.iter().any(|d| d.code == "MPT1200"),
            "expected MPT1200 diagnostics, got {:?}",
            diag.diagnostics
                .iter()
                .map(|d| d.code.as_str())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn ptr_op_outside_unsafe_context_is_rejected() {
        let module = mk_module(vec![mk_fn_decl(
            "main",
            false,
            vec![sp(AstInstr::Assign {
                name: "p".to_string(),
                ty: sp(rawptr_i32_ty()),
                op: AstOp::PtrNull { ty: i32_ty() },
            })],
            AstTerminator::Ret(Some(AstValueRef::Const(const_i32(0)))),
        )]);

        let mut diag = DiagnosticBag::new(32);
        let resolved = resolve_modules(&[module], &mut diag).expect("resolve succeeds");
        let mut type_ctx = TypeCtx::new();
        let lower = lower_to_hir(&resolved[0], &mut type_ctx, &mut diag);
        assert!(
            lower.is_err(),
            "lowering should fail outside unsafe context"
        );
        assert!(
            diag.diagnostics.iter().any(|d| d.code == "MPS0024"),
            "expected MPS0024 diagnostics, got {:?}",
            diag.diagnostics
                .iter()
                .map(|d| d.code.as_str())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn ptr_op_inside_unsafe_block_is_allowed() {
        let module = mk_module(vec![mk_fn_decl(
            "main",
            false,
            vec![sp(AstInstr::UnsafeBlock(vec![sp(AstInstr::Assign {
                name: "p".to_string(),
                ty: sp(rawptr_i32_ty()),
                op: AstOp::PtrNull { ty: i32_ty() },
            })]))],
            AstTerminator::Ret(Some(AstValueRef::Const(const_i32(0)))),
        )]);

        let mut diag = DiagnosticBag::new(32);
        let resolved = resolve_modules(&[module], &mut diag).expect("resolve succeeds");
        let mut type_ctx = TypeCtx::new();
        let lower = lower_to_hir(&resolved[0], &mut type_ctx, &mut diag);
        assert!(lower.is_ok(), "unsafe block should permit ptr ops");
        assert!(
            !diag.diagnostics.iter().any(|d| d.code == "MPS0024"),
            "unexpected MPS0024 diagnostics: {:?}",
            diag.diagnostics
        );
    }

    #[test]
    fn unsafe_fn_call_outside_unsafe_context_is_rejected() {
        let module = mk_module(vec![
            mk_fn_decl(
                "dangerous",
                true,
                vec![],
                AstTerminator::Ret(Some(AstValueRef::Const(const_i32(1)))),
            ),
            mk_fn_decl(
                "main",
                false,
                vec![sp(AstInstr::Assign {
                    name: "v".to_string(),
                    ty: sp(i32_ty()),
                    op: AstOp::Call {
                        callee: "dangerous".to_string(),
                        targs: vec![],
                        args: vec![],
                    },
                })],
                AstTerminator::Ret(Some(AstValueRef::Local("v".to_string()))),
            ),
        ]);

        let mut diag = DiagnosticBag::new(32);
        let resolved = resolve_modules(&[module], &mut diag).expect("resolve succeeds");
        let mut type_ctx = TypeCtx::new();
        let lower = lower_to_hir(&resolved[0], &mut type_ctx, &mut diag);
        assert!(lower.is_err(), "unsafe call should fail in safe context");
        assert!(
            diag.diagnostics.iter().any(|d| d.code == "MPS0025"),
            "expected MPS0025 diagnostics, got {:?}",
            diag.diagnostics
                .iter()
                .map(|d| d.code.as_str())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn unsafe_fn_call_inside_unsafe_block_is_allowed() {
        let module = mk_module(vec![
            mk_fn_decl(
                "dangerous",
                true,
                vec![],
                AstTerminator::Ret(Some(AstValueRef::Const(const_i32(1)))),
            ),
            mk_fn_decl(
                "main",
                false,
                vec![sp(AstInstr::UnsafeBlock(vec![sp(AstInstr::Assign {
                    name: "v".to_string(),
                    ty: sp(i32_ty()),
                    op: AstOp::Call {
                        callee: "dangerous".to_string(),
                        targs: vec![],
                        args: vec![],
                    },
                })]))],
                AstTerminator::Ret(Some(AstValueRef::Local("v".to_string()))),
            ),
        ]);

        let mut diag = DiagnosticBag::new(32);
        let resolved = resolve_modules(&[module], &mut diag).expect("resolve succeeds");
        let mut type_ctx = TypeCtx::new();
        let lower = lower_to_hir(&resolved[0], &mut type_ctx, &mut diag);
        assert!(lower.is_ok(), "unsafe block should permit unsafe calls");
        assert!(
            !diag.diagnostics.iter().any(|d| d.code == "MPS0025"),
            "unexpected MPS0025 diagnostics: {:?}",
            diag.diagnostics
        );
    }

    #[test]
    fn extern_rawptr_return_requires_returns_attr() {
        let module = mk_module(vec![sp(AstDecl::Extern(AstExternModule {
            abi: "c".to_string(),
            name: "ffi".to_string(),
            items: vec![AstExternItem {
                name: "open_handle".to_string(),
                params: vec![],
                ret_ty: sp(rawptr_i32_ty()),
                attrs: vec![("link_name".to_string(), "open_handle".to_string())],
            }],
            doc: None,
        }))]);

        let mut diag = DiagnosticBag::new(32);
        let resolved = resolve_modules(&[module], &mut diag);
        assert!(
            resolved.is_err(),
            "rawptr extern without returns attr should fail"
        );
        assert!(
            diag.diagnostics.iter().any(|d| d.code == "MPF0001"),
            "expected MPF0001 diagnostics, got {:?}",
            diag.diagnostics
                .iter()
                .map(|d| d.code.as_str())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn extern_rawptr_return_accepts_borrowed_attr() {
        let module = mk_module(vec![sp(AstDecl::Extern(AstExternModule {
            abi: "c".to_string(),
            name: "ffi".to_string(),
            items: vec![AstExternItem {
                name: "open_handle".to_string(),
                params: vec![AstParam {
                    name: "arg".to_string(),
                    ty: sp(i32_ty()),
                }],
                ret_ty: sp(rawptr_i32_ty()),
                attrs: vec![
                    ("link_name".to_string(), "open_handle".to_string()),
                    ("returns".to_string(), "borrowed".to_string()),
                ],
            }],
            doc: None,
        }))]);

        let mut diag = DiagnosticBag::new(32);
        let resolved = resolve_modules(&[module], &mut diag);
        assert!(resolved.is_ok(), "valid rawptr returns attr should pass");
        assert!(
            !diag.diagnostics.iter().any(|d| d.code == "MPF0001"),
            "unexpected MPF0001 diagnostics: {:?}",
            diag.diagnostics
        );
    }

    #[test]
    fn typecheck_str_parse_i64_rejects_invalid_result_type() {
        let type_ctx = TypeCtx::new();
        let fn_sid = generate_sid('F', "demo.parse_shape");

        let module = HirModule {
            module_id: ModuleId(0),
            sid: generate_sid('M', "demo.parse_shape"),
            path: "demo.parse_shape".to_string(),
            functions: vec![HirFunction {
                fn_id: FnId(0),
                sid: fn_sid.clone(),
                name: "main".to_string(),
                params: vec![],
                ret_ty: fixed_type_ids::I32,
                blocks: vec![HirBlock {
                    id: BlockId(0),
                    instrs: vec![
                        HirInstr {
                            dst: LocalId(0),
                            ty: fixed_type_ids::STR,
                            op: HirOp::Const(HirConst {
                                ty: fixed_type_ids::STR,
                                lit: HirConstLit::StringLit("123".to_string()),
                            }),
                        },
                        HirInstr {
                            dst: LocalId(1),
                            ty: fixed_type_ids::I32,
                            op: HirOp::StrParseI64 {
                                s: HirValue::Local(LocalId(0)),
                            },
                        },
                    ],
                    void_ops: vec![],
                    terminator: HirTerminator::Ret(Some(HirValue::Const(HirConst {
                        ty: fixed_type_ids::I32,
                        lit: HirConstLit::IntLit(0),
                    }))),
                }],
                is_async: false,
                is_unsafe: false,
            }],
            globals: vec![],
            type_decls: vec![],
        };

        let mut sym = SymbolTable::default();
        sym.functions.insert(
            "main".to_string(),
            FnSymbol {
                name: "main".to_string(),
                fqn: "demo.parse_shape.main".to_string(),
                sid: fn_sid,
                params: vec![],
                ret_ty: fixed_type_ids::I32,
                is_unsafe: false,
            },
        );

        let mut diag = DiagnosticBag::new(32);
        let result = typecheck_module(&module, &type_ctx, &sym, &mut diag);
        assert!(result.is_err(), "expected parse result shape to be rejected");
        assert!(
            diag.diagnostics.iter().any(|d| d.code == "MPT2033"),
            "expected MPT2033 diagnostics, got {:?}",
            diag.diagnostics
                .iter()
                .map(|d| d.code.as_str())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn typecheck_str_parse_i64_accepts_tresult_shape() {
        let mut type_ctx = TypeCtx::new();
        let fn_sid = generate_sid('F', "demo.parse_shape_ok");
        let parse_result_ty = type_ctx.intern(TypeKind::BuiltinResult {
            ok: fixed_type_ids::I64,
            err: fixed_type_ids::STR,
        });

        let module = HirModule {
            module_id: ModuleId(0),
            sid: generate_sid('M', "demo.parse_shape_ok"),
            path: "demo.parse_shape_ok".to_string(),
            functions: vec![HirFunction {
                fn_id: FnId(0),
                sid: fn_sid.clone(),
                name: "main".to_string(),
                params: vec![],
                ret_ty: fixed_type_ids::I32,
                blocks: vec![HirBlock {
                    id: BlockId(0),
                    instrs: vec![
                        HirInstr {
                            dst: LocalId(0),
                            ty: fixed_type_ids::STR,
                            op: HirOp::Const(HirConst {
                                ty: fixed_type_ids::STR,
                                lit: HirConstLit::StringLit("123".to_string()),
                            }),
                        },
                        HirInstr {
                            dst: LocalId(1),
                            ty: parse_result_ty,
                            op: HirOp::StrParseI64 {
                                s: HirValue::Local(LocalId(0)),
                            },
                        },
                    ],
                    void_ops: vec![],
                    terminator: HirTerminator::Ret(Some(HirValue::Const(HirConst {
                        ty: fixed_type_ids::I32,
                        lit: HirConstLit::IntLit(0),
                    }))),
                }],
                is_async: false,
                is_unsafe: false,
            }],
            globals: vec![],
            type_decls: vec![],
        };

        let mut sym = SymbolTable::default();
        sym.functions.insert(
            "main".to_string(),
            FnSymbol {
                name: "main".to_string(),
                fqn: "demo.parse_shape_ok.main".to_string(),
                sid: fn_sid,
                params: vec![],
                ret_ty: fixed_type_ids::I32,
                is_unsafe: false,
            },
        );

        let mut diag = DiagnosticBag::new(32);
        let result = typecheck_module(&module, &type_ctx, &sym, &mut diag);
        assert!(
            result.is_ok(),
            "expected TResult parse shape to pass typecheck: {:?}",
            diag.diagnostics
        );
    }

    #[test]
    fn typecheck_json_decode_rejects_non_rawptr_result_shape() {
        let type_ctx = TypeCtx::new();
        let fn_sid = generate_sid('F', "demo.json_decode_shape");

        let module = HirModule {
            module_id: ModuleId(0),
            sid: generate_sid('M', "demo.json_decode_shape"),
            path: "demo.json_decode_shape".to_string(),
            functions: vec![HirFunction {
                fn_id: FnId(0),
                sid: fn_sid.clone(),
                name: "main".to_string(),
                params: vec![],
                ret_ty: fixed_type_ids::I32,
                blocks: vec![HirBlock {
                    id: BlockId(0),
                    instrs: vec![
                        HirInstr {
                            dst: LocalId(0),
                            ty: fixed_type_ids::STR,
                            op: HirOp::Const(HirConst {
                                ty: fixed_type_ids::STR,
                                lit: HirConstLit::StringLit("123".to_string()),
                            }),
                        },
                        HirInstr {
                            dst: LocalId(1),
                            ty: fixed_type_ids::I32,
                            op: HirOp::JsonDecode {
                                ty: fixed_type_ids::I32,
                                s: HirValue::Local(LocalId(0)),
                            },
                        },
                    ],
                    void_ops: vec![],
                    terminator: HirTerminator::Ret(Some(HirValue::Const(HirConst {
                        ty: fixed_type_ids::I32,
                        lit: HirConstLit::IntLit(0),
                    }))),
                }],
                is_async: false,
                is_unsafe: false,
            }],
            globals: vec![],
            type_decls: vec![],
        };

        let mut sym = SymbolTable::default();
        sym.functions.insert(
            "main".to_string(),
            FnSymbol {
                name: "main".to_string(),
                fqn: "demo.json_decode_shape.main".to_string(),
                sid: fn_sid,
                params: vec![],
                ret_ty: fixed_type_ids::I32,
                is_unsafe: false,
            },
        );

        let mut diag = DiagnosticBag::new(32);
        let result = typecheck_module(&module, &type_ctx, &sym, &mut diag);
        assert!(
            result.is_err(),
            "expected json.decode non-rawptr result shape to be rejected"
        );
        assert!(
            diag.diagnostics.iter().any(|d| d.code == "MPT2033"),
            "expected MPT2033 diagnostics, got {:?}",
            diag.diagnostics
                .iter()
                .map(|d| d.code.as_str())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn trait_impl_decl_bindings_satisfy_map_new_requirements() {
        let key_sid = generate_sid('T', "pkg.consumer.TKey");
        let key_unique = TypeId(3000);
        let key_borrow = TypeId(3001);
        let map_ty = TypeId(3002);
        let mut type_ctx = TypeCtx::new();
        type_ctx.types.push((
            key_unique,
            TypeKind::HeapHandle {
                hk: HandleKind::Unique,
                base: HeapBase::UserType {
                    type_sid: key_sid.clone(),
                    targs: Vec::new(),
                },
            },
        ));
        type_ctx.types.push((
            key_borrow,
            TypeKind::HeapHandle {
                hk: HandleKind::Borrow,
                base: HeapBase::UserType {
                    type_sid: key_sid.clone(),
                    targs: Vec::new(),
                },
            },
        ));
        type_ctx.types.push((
            map_ty,
            TypeKind::HeapHandle {
                hk: HandleKind::Unique,
                base: HeapBase::BuiltinMap {
                    key: key_unique,
                    val: fixed_type_ids::I32,
                },
            },
        ));

        let hash_sid = generate_sid('F', "pkg.consumer.my_hash_key");
        let eq_sid = generate_sid('F', "pkg.consumer.my_eq_key");
        let main_sid = generate_sid('F', "pkg.consumer.main");

        let module = HirModule {
            module_id: ModuleId(0),
            sid: generate_sid('M', "pkg.consumer"),
            path: "pkg.consumer".to_string(),
            type_decls: vec![HirTypeDecl::Struct {
                sid: key_sid.clone(),
                name: "TKey".to_string(),
                fields: Vec::new(),
            }],
            globals: Vec::new(),
            functions: vec![
                HirFunction {
                    fn_id: FnId(0),
                    sid: hash_sid.clone(),
                    name: "my_hash_key".to_string(),
                    params: vec![(LocalId(0), key_borrow)],
                    ret_ty: fixed_type_ids::U64,
                    blocks: vec![HirBlock {
                        id: BlockId(0),
                        instrs: Vec::new(),
                        void_ops: Vec::new(),
                        terminator: HirTerminator::Ret(Some(HirValue::Const(HirConst {
                            ty: fixed_type_ids::U64,
                            lit: HirConstLit::IntLit(1),
                        }))),
                    }],
                    is_async: false,
                    is_unsafe: false,
                },
                HirFunction {
                    fn_id: FnId(1),
                    sid: eq_sid.clone(),
                    name: "my_eq_key".to_string(),
                    params: vec![(LocalId(0), key_borrow), (LocalId(1), key_borrow)],
                    ret_ty: fixed_type_ids::BOOL,
                    blocks: vec![HirBlock {
                        id: BlockId(0),
                        instrs: Vec::new(),
                        void_ops: Vec::new(),
                        terminator: HirTerminator::Ret(Some(HirValue::Const(HirConst {
                            ty: fixed_type_ids::BOOL,
                            lit: HirConstLit::BoolLit(true),
                        }))),
                    }],
                    is_async: false,
                    is_unsafe: false,
                },
                HirFunction {
                    fn_id: FnId(2),
                    sid: main_sid.clone(),
                    name: "main".to_string(),
                    params: Vec::new(),
                    ret_ty: fixed_type_ids::I32,
                    blocks: vec![HirBlock {
                        id: BlockId(0),
                        instrs: vec![HirInstr {
                            dst: LocalId(0),
                            ty: map_ty,
                            op: HirOp::MapNew {
                                key_ty: key_unique,
                                val_ty: fixed_type_ids::I32,
                            },
                        }],
                        void_ops: Vec::new(),
                        terminator: HirTerminator::Ret(Some(HirValue::Const(HirConst {
                            ty: fixed_type_ids::I32,
                            lit: HirConstLit::IntLit(0),
                        }))),
                    }],
                    is_async: false,
                    is_unsafe: false,
                },
            ],
        };

        let mut sym = SymbolTable::default();
        sym.functions.insert(
            "my_hash_key".to_string(),
            FnSymbol {
                name: "my_hash_key".to_string(),
                fqn: "pkg.consumer.my_hash_key".to_string(),
                sid: hash_sid,
                params: vec![key_borrow],
                ret_ty: fixed_type_ids::U64,
                is_unsafe: false,
            },
        );
        sym.functions.insert(
            "my_eq_key".to_string(),
            FnSymbol {
                name: "my_eq_key".to_string(),
                fqn: "pkg.consumer.my_eq_key".to_string(),
                sid: eq_sid,
                params: vec![key_borrow, key_borrow],
                ret_ty: fixed_type_ids::BOOL,
                is_unsafe: false,
            },
        );
        sym.functions.insert(
            "main".to_string(),
            FnSymbol {
                name: "main".to_string(),
                fqn: "pkg.consumer.main".to_string(),
                sid: main_sid,
                params: Vec::new(),
                ret_ty: fixed_type_ids::I32,
                is_unsafe: false,
            },
        );

        let impl_decls = vec![
            AstImplDecl {
                trait_name: "hash".to_string(),
                for_type: AstType {
                    ownership: None,
                    base: AstBaseType::Named {
                        path: None,
                        name: "TKey".to_string(),
                        targs: Vec::new(),
                    },
                },
                fn_ref: "my_hash_key".to_string(),
            },
            AstImplDecl {
                trait_name: "eq".to_string(),
                for_type: AstType {
                    ownership: None,
                    base: AstBaseType::Named {
                        path: None,
                        name: "TKey".to_string(),
                        targs: Vec::new(),
                    },
                },
                fn_ref: "my_eq_key".to_string(),
            },
        ];

        let mut diag = DiagnosticBag::new(64);
        let check = check_trait_impls(&module, &type_ctx, &sym, &impl_decls, &[], &mut diag);
        assert!(
            check.is_ok(),
            "expected trait check success: {:?}",
            diag.diagnostics
        );
        assert!(
            !diag.diagnostics.iter().any(|d| d.code == "MPT1023"),
            "unexpected MPT1023 diagnostics: {:?}",
            diag.diagnostics
                .iter()
                .map(|d| d.code.as_str())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn trait_impl_decl_reports_missing_function_binding() {
        let key_sid = generate_sid('T', "pkg.consumer.TKey");
        let key_unique = TypeId(3010);
        let mut type_ctx = TypeCtx::new();
        type_ctx.types.push((
            key_unique,
            TypeKind::HeapHandle {
                hk: HandleKind::Unique,
                base: HeapBase::UserType {
                    type_sid: key_sid.clone(),
                    targs: Vec::new(),
                },
            },
        ));
        let module = HirModule {
            module_id: ModuleId(0),
            sid: generate_sid('M', "pkg.consumer"),
            path: "pkg.consumer".to_string(),
            type_decls: vec![HirTypeDecl::Struct {
                sid: key_sid,
                name: "TKey".to_string(),
                fields: Vec::new(),
            }],
            globals: Vec::new(),
            functions: Vec::new(),
        };
        let sym = SymbolTable::default();
        let impl_decls = vec![AstImplDecl {
            trait_name: "hash".to_string(),
            for_type: AstType {
                ownership: None,
                base: AstBaseType::Named {
                    path: None,
                    name: "TKey".to_string(),
                    targs: Vec::new(),
                },
            },
            fn_ref: "missing_hash".to_string(),
        }];

        let mut diag = DiagnosticBag::new(32);
        let check = check_trait_impls(&module, &type_ctx, &sym, &impl_decls, &[], &mut diag);
        assert!(check.is_err(), "expected missing impl fn to fail");
        assert!(
            diag.diagnostics.iter().any(|d| d.code == "MPT2032"),
            "expected MPT2032 diagnostics, got {:?}",
            diag.diagnostics
                .iter()
                .map(|d| d.code.as_str())
                .collect::<Vec<_>>()
        );
    }
}
