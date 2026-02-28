//! Magpie recursive-descent parser.

use magpie_ast::*;
use magpie_diag::{Diagnostic, DiagnosticBag, Severity};
use magpie_lex::{Token, TokenKind};

#[allow(clippy::result_unit_err)]
pub fn parse_file(
    tokens: &[Token],
    file_id: FileId,
    diag: &mut DiagnosticBag,
) -> Result<AstFile, ()> {
    let before = diag.error_count();
    let mut parser = Parser::new(tokens, file_id, diag);
    let ast = parser.parse_file();
    if parser.diag.error_count() > before {
        Err(())
    } else {
        Ok(ast)
    }
}

struct Parser<'a, 'd> {
    tokens: &'a [Token],
    pos: usize,
    file_id: FileId,
    diag: &'d mut DiagnosticBag,
    fallback_eof: Token,
}

impl<'a, 'd> Parser<'a, 'd> {
    fn new(tokens: &'a [Token], file_id: FileId, diag: &'d mut DiagnosticBag) -> Self {
        Self {
            tokens,
            pos: 0,
            file_id,
            diag,
            fallback_eof: Token {
                kind: TokenKind::Eof,
                span: Span::new(file_id, 0, 0),
                text: String::new(),
            },
        }
    }

    fn parse_file(&mut self) -> AstFile {
        let header = self.parse_header();
        let mut decls = Vec::new();

        while !self.at(TokenKind::Eof) {
            let before = self.pos;
            if let Some(decl) = self.parse_decl() {
                decls.push(decl);
                continue;
            }

            if self.pos == before {
                self.error_here("Expected declaration.");
                self.advance();
            }
            self.recover_to_next_decl();
        }

        AstFile { header, decls }
    }

    fn parse_header(&mut self) -> Spanned<AstHeader> {
        let start = self.peek().span;

        self.expect(TokenKind::Module);
        let module_path = self.parse_module_path().unwrap_or(ModulePath {
            segments: Vec::new(),
        });

        self.expect(TokenKind::Exports);
        let exports = self.parse_exports_block();

        self.expect(TokenKind::Imports);
        let imports = self.parse_imports_block();

        self.expect(TokenKind::Digest);
        let digest_start = self.peek().span;
        let digest = self.parse_string_lit().unwrap_or_default();
        let digest = Spanned::new(digest, self.span_from(digest_start));

        Spanned::new(
            AstHeader {
                module_path: Spanned::new(module_path, self.span_from(start)),
                exports,
                imports,
                digest,
            },
            self.span_from(start),
        )
    }

    fn parse_exports_block(&mut self) -> Vec<Spanned<ExportItem>> {
        let mut items = Vec::new();
        self.expect(TokenKind::LBrace);

        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            let start = self.peek().span;
            match self.peek().kind {
                TokenKind::FnName => {
                    let name = self.advance().text;
                    items.push(Spanned::new(ExportItem::Fn(name), self.span_from(start)));
                }
                TokenKind::TypeName => {
                    let name = self.advance().text;
                    items.push(Spanned::new(ExportItem::Type(name), self.span_from(start)));
                }
                _ => {
                    self.error_here("Expected export item (`@name` or `TName`).");
                    self.advance();
                }
            }

            if self.eat(TokenKind::Comma).is_none() {
                break;
            }
        }

        self.expect(TokenKind::RBrace);
        items
    }

    fn parse_imports_block(&mut self) -> Vec<Spanned<ImportGroup>> {
        let mut groups = Vec::new();
        self.expect(TokenKind::LBrace);

        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            let start = self.peek().span;
            let module_path = self.parse_module_path().unwrap_or(ModulePath {
                segments: Vec::new(),
            });
            self.expect_double_colon();
            self.expect(TokenKind::LBrace);

            let mut items = Vec::new();
            while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
                let item_start = self.peek().span;
                match self.peek().kind {
                    TokenKind::FnName => {
                        let name = self.advance().text;
                        items.push(ImportItem::Fn(name));
                    }
                    TokenKind::TypeName => {
                        let name = self.advance().text;
                        items.push(ImportItem::Type(name));
                    }
                    _ => {
                        self.error_here("Expected import item (`@name` or `TName`).");
                        self.advance();
                    }
                }

                let _ = item_start;
                if self.eat(TokenKind::Comma).is_none() {
                    break;
                }
            }

            self.expect(TokenKind::RBrace);
            groups.push(Spanned::new(
                ImportGroup { module_path, items },
                self.span_from(start),
            ));

            if self.eat(TokenKind::Comma).is_none() {
                break;
            }
        }

        self.expect(TokenKind::RBrace);
        groups
    }

    fn parse_decl(&mut self) -> Option<Spanned<AstDecl>> {
        if self.at(TokenKind::Eof) {
            return None;
        }

        let start = self.peek().span;
        let doc = self.parse_doc_opt();

        let decl = match self.peek().kind {
            TokenKind::Fn => self.parse_fn_decl(start, doc, FnFlavor::Regular),
            TokenKind::Async => self.parse_fn_decl(start, doc, FnFlavor::Async),
            TokenKind::Unsafe if self.peek_n_kind(1) == TokenKind::Fn => {
                self.parse_fn_decl(start, doc, FnFlavor::Unsafe)
            }
            TokenKind::Gpu => self.parse_fn_decl(start, doc, FnFlavor::Gpu),
            TokenKind::Heap => self.parse_type_decl(start, doc, true),
            TokenKind::Value => self.parse_type_decl(start, doc, false),
            TokenKind::Extern => self.parse_extern_decl(start, doc),
            TokenKind::Global => self.parse_global_decl(start, doc),
            TokenKind::Impl => self.parse_impl_decl(start),
            TokenKind::Sig => self.parse_sig_decl(start),
            _ => {
                if doc.is_some() {
                    self.error_here("Doc comment must be followed by a declaration.");
                }
                return None;
            }
        };

        if decl.is_none() {
            self.recover_to_next_decl();
        }

        decl
    }

    fn parse_doc_opt(&mut self) -> Option<String> {
        if !self.at(TokenKind::DocComment) {
            return None;
        }

        let mut lines = Vec::new();
        while self.at(TokenKind::DocComment) {
            lines.push(self.advance().text);
        }
        Some(lines.join("\n"))
    }

    fn parse_fn_decl(
        &mut self,
        start: Span,
        doc: Option<String>,
        flavor: FnFlavor,
    ) -> Option<Spanned<AstDecl>> {
        match flavor {
            FnFlavor::Regular => {
                self.expect(TokenKind::Fn);
            }
            FnFlavor::Async => {
                self.expect(TokenKind::Async);
                self.expect(TokenKind::Fn);
            }
            FnFlavor::Unsafe => {
                self.expect(TokenKind::Unsafe);
                self.expect(TokenKind::Fn);
            }
            FnFlavor::Gpu => {
                self.expect(TokenKind::Gpu);
                self.expect(TokenKind::Fn);
            }
        }

        let name = self
            .parse_fn_name()
            .unwrap_or_else(|| "@<error>".to_string());

        self.expect(TokenKind::LParen);
        let params = self.parse_params_list(TokenKind::RParen);
        self.expect(TokenKind::RParen);

        self.expect(TokenKind::Arrow);
        let ret_ty = self.parse_type();

        let target = if matches!(flavor, FnFlavor::Gpu) {
            self.expect(TokenKind::Target);
            self.expect(TokenKind::LParen);
            let target = self.parse_ident().unwrap_or_default();
            self.expect(TokenKind::RParen);
            Some(target)
        } else {
            None
        };

        let meta = if self.at(TokenKind::Meta) {
            self.parse_fn_meta()
        } else {
            None
        };

        let blocks = self.parse_blocks();

        let inner = AstFnDecl {
            name,
            params,
            ret_ty,
            meta,
            blocks,
            doc,
        };

        let decl = match flavor {
            FnFlavor::Regular => AstDecl::Fn(inner),
            FnFlavor::Async => AstDecl::AsyncFn(inner),
            FnFlavor::Unsafe => AstDecl::UnsafeFn(inner),
            FnFlavor::Gpu => AstDecl::GpuFn(AstGpuFnDecl {
                inner,
                target: target.unwrap_or_default(),
            }),
        };

        Some(Spanned::new(decl, self.span_from(start)))
    }

    fn parse_fn_meta(&mut self) -> Option<AstFnMeta> {
        self.expect(TokenKind::Meta);
        self.expect(TokenKind::LBrace);

        let mut uses = Vec::new();
        let mut effects = Vec::new();
        let mut cost = Vec::new();

        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            if self.at(TokenKind::Uses) {
                self.advance();
                self.expect(TokenKind::LBrace);
                while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
                    if let Some(r) = self.parse_fqn_ref() {
                        uses.push(r);
                    } else {
                        self.advance();
                    }
                    if self.eat(TokenKind::Comma).is_none() {
                        break;
                    }
                }
                self.expect(TokenKind::RBrace);
                continue;
            }

            if self.at(TokenKind::Effects) {
                self.advance();
                self.expect(TokenKind::LBrace);
                while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
                    if let Some(effect) = self.parse_ident() {
                        effects.push(effect);
                    } else {
                        self.advance();
                    }
                    if self.eat(TokenKind::Comma).is_none() {
                        break;
                    }
                }
                self.expect(TokenKind::RBrace);
                continue;
            }

            if self.at(TokenKind::Cost) {
                self.advance();
                self.expect(TokenKind::LBrace);
                while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
                    let key = self.parse_ident().unwrap_or_default();
                    self.expect(TokenKind::Eq);
                    let value = self.parse_i64_lit().unwrap_or_default();
                    cost.push((key, value));
                    if self.eat(TokenKind::Comma).is_none() {
                        break;
                    }
                }
                self.expect(TokenKind::RBrace);
                continue;
            }

            self.error_here("Expected `uses`, `effects`, or `cost` inside `meta`.");
            self.advance();
        }

        self.expect(TokenKind::RBrace);

        Some(AstFnMeta {
            uses,
            effects,
            cost,
        })
    }

    fn parse_type_decl(
        &mut self,
        start: Span,
        doc: Option<String>,
        is_heap: bool,
    ) -> Option<Spanned<AstDecl>> {
        if is_heap {
            self.expect(TokenKind::Heap);
        } else {
            self.expect(TokenKind::Value);
        }

        let is_struct = if self.at(TokenKind::Struct) {
            self.advance();
            true
        } else if self.at(TokenKind::Enum) {
            self.advance();
            false
        } else {
            self.error_here("Expected `struct` or `enum`.");
            return None;
        };

        let name = self
            .parse_type_name()
            .unwrap_or_else(|| "T<error>".to_string());
        let type_params = self.parse_type_params_opt();

        self.expect(TokenKind::LBrace);

        let decl = if is_struct {
            let mut fields = Vec::new();
            while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
                if self.eat(TokenKind::Comma).is_some() {
                    continue;
                }

                if !self.expect_word("field") {
                    self.recover_to_type_body();
                    if self.at(TokenKind::RBrace) {
                        break;
                    }
                }

                let field_name = self.parse_ident().unwrap_or_default();
                self.expect(TokenKind::Colon);
                let ty = self.parse_type();
                fields.push(AstFieldDecl {
                    name: field_name,
                    ty,
                });
            }

            AstDecl::HeapStruct(AstStructDecl {
                name,
                type_params,
                fields,
                doc,
            })
        } else {
            let mut variants = Vec::new();
            while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
                if self.eat(TokenKind::Comma).is_some() {
                    continue;
                }

                if !self.expect_word("variant") {
                    self.recover_to_type_body();
                    if self.at(TokenKind::RBrace) {
                        break;
                    }
                }

                let variant_name = self.parse_ident_or_type_name().unwrap_or_default();
                self.expect(TokenKind::LBrace);
                let mut fields = Vec::new();

                while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
                    if self.eat(TokenKind::Comma).is_some() {
                        continue;
                    }

                    if !self.expect_word("field") {
                        self.recover_to_variant_body();
                        if self.at(TokenKind::RBrace) {
                            break;
                        }
                    }

                    let field_name = self.parse_ident().unwrap_or_default();
                    self.expect(TokenKind::Colon);
                    let ty = self.parse_type();
                    fields.push(AstFieldDecl {
                        name: field_name,
                        ty,
                    });
                }

                self.expect(TokenKind::RBrace);
                variants.push(AstVariantDecl {
                    name: variant_name,
                    fields,
                });
            }

            if is_heap {
                AstDecl::HeapEnum(AstEnumDecl {
                    name,
                    type_params,
                    variants,
                    doc,
                })
            } else {
                AstDecl::ValueEnum(AstEnumDecl {
                    name,
                    type_params,
                    variants,
                    doc,
                })
            }
        };

        self.expect(TokenKind::RBrace);

        if is_struct {
            let out = if is_heap {
                match decl {
                    AstDecl::HeapStruct(s) => AstDecl::HeapStruct(s),
                    _ => unreachable!(),
                }
            } else {
                match decl {
                    AstDecl::HeapStruct(s) => AstDecl::ValueStruct(s),
                    _ => unreachable!(),
                }
            };
            return Some(Spanned::new(out, self.span_from(start)));
        }

        Some(Spanned::new(decl, self.span_from(start)))
    }

    fn parse_extern_decl(&mut self, start: Span, doc: Option<String>) -> Option<Spanned<AstDecl>> {
        self.expect(TokenKind::Extern);
        let abi = self.parse_string_lit().unwrap_or_default();
        self.expect(TokenKind::Module);
        let name = self.parse_ident().unwrap_or_default();

        self.expect(TokenKind::LBrace);
        let mut items = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            if !self.at(TokenKind::Fn) {
                self.error_here("Expected extern item `fn`.");
                self.advance();
                continue;
            }

            self.advance();
            let item_name = self.parse_fn_name().unwrap_or_default();
            self.expect(TokenKind::LParen);
            let params = self.parse_params_list(TokenKind::RParen);
            self.expect(TokenKind::RParen);
            self.expect(TokenKind::Arrow);
            let ret_ty = self.parse_type();

            let mut attrs = Vec::new();
            if self.at_word("attrs") {
                self.advance();
                self.expect(TokenKind::LBrace);
                while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
                    let key = self.parse_ident().unwrap_or_default();
                    self.expect(TokenKind::Eq);
                    let val = self.parse_string_lit().unwrap_or_default();
                    attrs.push((key, val));
                    let _ = self.eat(TokenKind::Comma);
                }
                self.expect(TokenKind::RBrace);
            }

            items.push(AstExternItem {
                name: item_name,
                params,
                ret_ty,
                attrs,
            });
        }

        self.expect(TokenKind::RBrace);

        Some(Spanned::new(
            AstDecl::Extern(AstExternModule {
                abi,
                name,
                items,
                doc,
            }),
            self.span_from(start),
        ))
    }

    fn parse_global_decl(&mut self, start: Span, doc: Option<String>) -> Option<Spanned<AstDecl>> {
        self.expect(TokenKind::Global);
        let name = self.parse_fn_name().unwrap_or_default();
        self.expect(TokenKind::Colon);
        let ty = self.parse_type();
        self.expect(TokenKind::Eq);
        let init = self
            .parse_const_expr_with_hint(Some(ty.node.clone()))
            .unwrap_or_else(|| AstConstExpr {
                ty: ty.node.clone(),
                lit: AstConstLit::Unit,
            });

        Some(Spanned::new(
            AstDecl::Global(AstGlobalDecl {
                name,
                ty,
                init,
                doc,
            }),
            self.span_from(start),
        ))
    }

    fn parse_impl_decl(&mut self, start: Span) -> Option<Spanned<AstDecl>> {
        self.expect(TokenKind::Impl);
        let trait_name = self.parse_ident().unwrap_or_default();
        self.expect_word("for");
        let for_type = self.parse_type().node;
        self.expect(TokenKind::Eq);
        let fn_ref = self.parse_fn_ref().unwrap_or_default();

        Some(Spanned::new(
            AstDecl::Impl(AstImplDecl {
                trait_name,
                for_type,
                fn_ref,
            }),
            self.span_from(start),
        ))
    }

    fn parse_sig_decl(&mut self, start: Span) -> Option<Spanned<AstDecl>> {
        self.expect(TokenKind::Sig);
        let name = self.parse_type_name().unwrap_or_default();
        self.expect(TokenKind::LParen);

        let mut param_types = Vec::new();
        while !self.at(TokenKind::RParen) && !self.at(TokenKind::Eof) {
            param_types.push(self.parse_type().node);
            if self.eat(TokenKind::Comma).is_none() {
                break;
            }
        }

        self.expect(TokenKind::RParen);
        self.expect(TokenKind::Arrow);
        let ret_ty = self.parse_type().node;

        Some(Spanned::new(
            AstDecl::Sig(AstSigDecl {
                name,
                param_types,
                ret_ty,
            }),
            self.span_from(start),
        ))
    }

    fn parse_blocks(&mut self) -> Vec<Spanned<AstBlock>> {
        let mut blocks = Vec::new();
        self.expect(TokenKind::LBrace);

        while self.at(TokenKind::BlockLabel) {
            if let Some(block) = self.parse_block() {
                blocks.push(block);
            } else {
                self.recover_to_block_boundary();
                if self.at(TokenKind::RBrace) || self.at(TokenKind::Eof) {
                    break;
                }
            }
        }

        if blocks.is_empty() {
            self.error_here("Function body must contain at least one basic block.");
        }

        self.expect(TokenKind::RBrace);
        blocks
    }

    fn parse_block(&mut self) -> Option<Spanned<AstBlock>> {
        let start = self.peek().span;
        let label = self.parse_block_label_num()?;
        self.expect(TokenKind::Colon);

        let mut instrs = Vec::new();
        while !self.at_terminator_start() {
            if self.at(TokenKind::BlockLabel)
                || self.at(TokenKind::RBrace)
                || self.at(TokenKind::Eof)
            {
                self.error_here("Missing block terminator.");
                let term =
                    Spanned::new(AstTerminator::Unreachable, self.span_from(self.peek().span));
                return Some(Spanned::new(
                    AstBlock {
                        label,
                        instrs,
                        terminator: term,
                    },
                    self.span_from(start),
                ));
            }

            let before = self.pos;
            if let Some(instr) = self.parse_instr() {
                instrs.push(instr);
            } else {
                if self.pos == before {
                    self.error_here("Expected instruction.");
                    self.advance();
                }
                self.recover_to_block_stmt_boundary();
            }
        }

        let terminator = self.parse_terminator().unwrap_or_else(|| {
            Spanned::new(AstTerminator::Unreachable, self.span_from(self.peek().span))
        });

        Some(Spanned::new(
            AstBlock {
                label,
                instrs,
                terminator,
            },
            self.span_from(start),
        ))
    }

    fn parse_instr(&mut self) -> Option<Spanned<AstInstr>> {
        if self.at(TokenKind::SsaName) {
            return self.parse_assign_instr();
        }

        if self.at(TokenKind::Unsafe) && self.peek_n_kind(1) == TokenKind::LBrace {
            return self.parse_unsafe_block_instr();
        }

        if self.at_op_void_start() {
            let start = self.peek().span;
            let op = self.parse_op_void()?;
            return Some(Spanned::new(AstInstr::Void(op), self.span_from(start)));
        }

        None
    }

    fn parse_assign_instr(&mut self) -> Option<Spanned<AstInstr>> {
        let start = self.peek().span;
        let name = self.parse_ssa_name()?;
        self.expect(TokenKind::Colon);
        let ty = self.parse_type();
        self.expect(TokenKind::Eq);
        let op = self.parse_op(Some(ty.node.clone()))?;

        Some(Spanned::new(
            AstInstr::Assign { name, ty, op },
            self.span_from(start),
        ))
    }

    fn parse_unsafe_block_instr(&mut self) -> Option<Spanned<AstInstr>> {
        let start = self.peek().span;
        self.expect(TokenKind::Unsafe);
        self.expect(TokenKind::LBrace);

        let mut inner = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            let before = self.pos;
            if self.at(TokenKind::SsaName) {
                if let Some(i) = self.parse_assign_instr() {
                    inner.push(i);
                    continue;
                }
            } else if self.at_op_void_start() {
                let op_start = self.peek().span;
                if let Some(v) = self.parse_op_void() {
                    inner.push(Spanned::new(AstInstr::Void(v), self.span_from(op_start)));
                    continue;
                }
            }

            if self.pos == before {
                self.error_here("Unsafe block supports only SSA assignments and void ops.");
                self.advance();
            }
            self.recover_to_block_stmt_boundary();
        }

        if inner.is_empty() {
            self.error_here("Unsafe block must contain at least one instruction.");
        }

        self.expect(TokenKind::RBrace);
        Some(Spanned::new(
            AstInstr::UnsafeBlock(inner),
            self.span_from(start),
        ))
    }

    fn parse_terminator(&mut self) -> Option<Spanned<AstTerminator>> {
        let start = self.peek().span;

        if self.at_word("ret") {
            self.advance();
            let value = if self.at(TokenKind::SsaName) || self.at(TokenKind::ConstOp) {
                self.parse_value_ref(None)
            } else {
                None
            };
            return Some(Spanned::new(
                AstTerminator::Ret(value),
                self.span_from(start),
            ));
        }

        if self.at_word("br") {
            self.advance();
            let bb = self.parse_block_label_num().unwrap_or_default();
            return Some(Spanned::new(AstTerminator::Br(bb), self.span_from(start)));
        }

        if self.at_word("cbr") {
            self.advance();
            let cond = self.parse_value_ref(None)?;
            let then_bb = self.parse_block_label_num().unwrap_or_default();
            let else_bb = self.parse_block_label_num().unwrap_or_default();
            return Some(Spanned::new(
                AstTerminator::Cbr {
                    cond,
                    then_bb,
                    else_bb,
                },
                self.span_from(start),
            ));
        }

        if self.at_word("switch") {
            self.advance();
            let val = self.parse_value_ref(None)?;
            self.expect(TokenKind::LBrace);

            let mut arms = Vec::new();
            while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
                if !self.expect_word("case") {
                    self.advance();
                    continue;
                }
                let lit = self.parse_const_lit().unwrap_or(AstConstLit::Unit);
                self.expect(TokenKind::Arrow);
                let bb = self.parse_block_label_num().unwrap_or_default();
                arms.push((lit, bb));
            }

            self.expect(TokenKind::RBrace);
            self.expect_word("else");
            let default = self.parse_block_label_num().unwrap_or_default();
            return Some(Spanned::new(
                AstTerminator::Switch { val, arms, default },
                self.span_from(start),
            ));
        }

        if self.at_word("unreachable") {
            self.advance();
            return Some(Spanned::new(
                AstTerminator::Unreachable,
                self.span_from(start),
            ));
        }

        self.error_here("Expected terminator (`ret`, `br`, `cbr`, `switch`, `unreachable`).");
        None
    }

    fn parse_op(&mut self, const_hint: Option<AstType>) -> Option<AstOp> {
        match self.peek().kind {
            TokenKind::ConstOp => {
                let c = self.parse_const_expr_with_hint(const_hint)?;
                Some(AstOp::Const(c))
            }

            TokenKind::IAdd
            | TokenKind::ISub
            | TokenKind::IMul
            | TokenKind::ISdiv
            | TokenKind::IUdiv
            | TokenKind::ISrem
            | TokenKind::IUrem
            | TokenKind::IAddWrap
            | TokenKind::ISubWrap
            | TokenKind::IMulWrap
            | TokenKind::IAddChecked
            | TokenKind::ISubChecked
            | TokenKind::IMulChecked
            | TokenKind::IAnd
            | TokenKind::IOr
            | TokenKind::IXor
            | TokenKind::IShl
            | TokenKind::ILshr
            | TokenKind::IAshr
            | TokenKind::FAdd
            | TokenKind::FSub
            | TokenKind::FMul
            | TokenKind::FDiv
            | TokenKind::FRem
            | TokenKind::FAddFast
            | TokenKind::FSubFast
            | TokenKind::FMulFast
            | TokenKind::FDivFast => {
                let tok = self.advance().kind;
                let kind = self.bin_op_kind(tok)?;
                let args = self.parse_value_pairs_braced()?;
                let lhs = self.require_named(&args, "lhs")?;
                let rhs = self.require_named(&args, "rhs")?;
                Some(AstOp::BinOp { kind, lhs, rhs })
            }

            TokenKind::IcmpEq
            | TokenKind::IcmpNe
            | TokenKind::IcmpSlt
            | TokenKind::IcmpSgt
            | TokenKind::IcmpSle
            | TokenKind::IcmpSge
            | TokenKind::IcmpUlt
            | TokenKind::IcmpUgt
            | TokenKind::IcmpUle
            | TokenKind::IcmpUge
            | TokenKind::FcmpOeq
            | TokenKind::FcmpOne
            | TokenKind::FcmpOlt
            | TokenKind::FcmpOgt
            | TokenKind::FcmpOle
            | TokenKind::FcmpOge => {
                let tok = self.advance().kind;
                let (kind, pred) = self.cmp_kind(tok)?;
                let args = self.parse_value_pairs_braced()?;
                let lhs = self.require_named(&args, "lhs")?;
                let rhs = self.require_named(&args, "rhs")?;
                Some(AstOp::Cmp {
                    kind,
                    pred: pred.to_string(),
                    lhs,
                    rhs,
                })
            }

            TokenKind::Call => {
                self.advance();
                let callee = self.parse_fn_ref()?;
                let targs = self.parse_type_args_opt();
                let args = self.parse_arg_pairs_braced()?;
                Some(AstOp::Call {
                    callee,
                    targs,
                    args,
                })
            }

            TokenKind::CallIndirect => {
                self.advance();
                let callee = self.parse_value_ref(None)?;
                let args = self.parse_arg_pairs_braced()?;
                Some(AstOp::CallIndirect { callee, args })
            }

            TokenKind::Try => {
                self.advance();
                let callee = self.parse_fn_ref()?;
                let targs = self.parse_type_args_opt();
                let args = self.parse_arg_pairs_braced()?;
                Some(AstOp::Try {
                    callee,
                    targs,
                    args,
                })
            }

            TokenKind::SuspendCall => {
                self.advance();
                let callee = self.parse_fn_ref()?;
                let targs = self.parse_type_args_opt();
                let args = self.parse_arg_pairs_braced()?;
                Some(AstOp::SuspendCall {
                    callee,
                    targs,
                    args,
                })
            }

            TokenKind::SuspendAwait => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let fut = self.require_named(&args, "fut")?;
                Some(AstOp::SuspendAwait { fut })
            }

            TokenKind::New => {
                self.advance();
                let ty = self.parse_type().node;
                let fields = self.parse_value_pairs_braced()?;
                Some(AstOp::New { ty, fields })
            }

            TokenKind::GetField => {
                self.advance();
                self.expect(TokenKind::LBrace);
                let mut obj = None;
                let mut field = None;
                while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
                    let key = self.parse_ident().unwrap_or_default();
                    self.expect(TokenKind::Eq);
                    match key.as_str() {
                        "obj" => obj = self.parse_value_ref(None),
                        "field" => field = Some(self.parse_ident().unwrap_or_default()),
                        _ => {
                            self.error_here(format!("Unknown key `{}` in getfield", key));
                            break;
                        }
                    }
                    self.eat(TokenKind::Comma);
                }
                self.expect(TokenKind::RBrace);
                let obj = obj.or_else(|| {
                    self.error_here("Missing key `obj` in getfield");
                    None
                })?;
                let field = field.unwrap_or_else(|| {
                    self.error_here("Missing key `field` in getfield");
                    String::new()
                });
                Some(AstOp::GetField { obj, field })
            }

            TokenKind::Phi => {
                self.advance();
                let ty = self.parse_type().node;
                self.expect(TokenKind::LBrace);
                let mut incomings = Vec::new();
                while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
                    self.expect(TokenKind::LBracket);
                    let bb = self.parse_block_label_num().unwrap_or_default();
                    self.expect(TokenKind::Colon);
                    let val = self.parse_value_ref(None)?;
                    self.expect(TokenKind::RBracket);
                    incomings.push((bb, val));
                    if self.eat(TokenKind::Comma).is_none() {
                        break;
                    }
                }
                self.expect(TokenKind::RBrace);
                Some(AstOp::Phi { ty, incomings })
            }

            TokenKind::EnumNew => {
                self.advance();
                self.expect(TokenKind::LAngle);
                let variant = self.parse_ident_or_type_name().unwrap_or_default();
                self.expect(TokenKind::RAngle);
                let args = self.parse_value_pairs_braced()?;
                Some(AstOp::EnumNew { variant, args })
            }

            TokenKind::EnumTag => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let v = self.require_named(&args, "v")?;
                Some(AstOp::EnumTag { v })
            }

            TokenKind::EnumPayload => {
                self.advance();
                self.expect(TokenKind::LAngle);
                let variant = self.parse_ident_or_type_name().unwrap_or_default();
                self.expect(TokenKind::RAngle);
                let args = self.parse_value_pairs_braced()?;
                let v = self.require_named(&args, "v")?;
                Some(AstOp::EnumPayload { variant, v })
            }

            TokenKind::EnumIs => {
                self.advance();
                self.expect(TokenKind::LAngle);
                let variant = self.parse_ident_or_type_name().unwrap_or_default();
                self.expect(TokenKind::RAngle);
                let args = self.parse_value_pairs_braced()?;
                let v = self.require_named(&args, "v")?;
                Some(AstOp::EnumIs { variant, v })
            }

            TokenKind::Share => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let v = self.require_named(&args, "v")?;
                Some(AstOp::Share { v })
            }

            TokenKind::CloneShared => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let v = self.require_named(&args, "v")?;
                Some(AstOp::CloneShared { v })
            }

            TokenKind::CloneWeak => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let v = self.require_named(&args, "v")?;
                Some(AstOp::CloneWeak { v })
            }

            TokenKind::WeakDowngrade => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let v = self.require_named(&args, "v")?;
                Some(AstOp::WeakDowngrade { v })
            }

            TokenKind::WeakUpgrade => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let v = self.require_named(&args, "v")?;
                Some(AstOp::WeakUpgrade { v })
            }

            TokenKind::Cast => {
                self.advance();
                self.expect(TokenKind::LAngle);
                let from = self.parse_prim_type_only();
                self.expect(TokenKind::Comma);
                let to = self.parse_prim_type_only();
                self.expect(TokenKind::RAngle);
                let args = self.parse_value_pairs_braced()?;
                let v = self.require_named(&args, "v")?;
                Some(AstOp::Cast { from, to, v })
            }

            TokenKind::BorrowShared => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let v = self.require_named(&args, "v")?;
                Some(AstOp::BorrowShared { v })
            }

            TokenKind::BorrowMut => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let v = self.require_named(&args, "v")?;
                Some(AstOp::BorrowMut { v })
            }

            TokenKind::PtrNull => {
                self.advance();
                self.expect(TokenKind::LAngle);
                let ty = self.parse_type().node;
                self.expect(TokenKind::RAngle);
                Some(AstOp::PtrNull { ty })
            }

            TokenKind::PtrAddr => {
                self.advance();
                self.expect(TokenKind::LAngle);
                let ty = self.parse_type().node;
                self.expect(TokenKind::RAngle);
                let args = self.parse_value_pairs_braced()?;
                let p = self.require_named(&args, "p")?;
                Some(AstOp::PtrAddr { ty, p })
            }

            TokenKind::PtrFromAddr => {
                self.advance();
                self.expect(TokenKind::LAngle);
                let ty = self.parse_type().node;
                self.expect(TokenKind::RAngle);
                let args = self.parse_value_pairs_braced()?;
                let addr = self.require_named(&args, "addr")?;
                Some(AstOp::PtrFromAddr { ty, addr })
            }

            TokenKind::PtrAdd => {
                self.advance();
                self.expect(TokenKind::LAngle);
                let ty = self.parse_type().node;
                self.expect(TokenKind::RAngle);
                let args = self.parse_value_pairs_braced()?;
                let p = self.require_named(&args, "p")?;
                let count = self.require_named(&args, "count")?;
                Some(AstOp::PtrAdd { ty, p, count })
            }

            TokenKind::PtrLoad => {
                self.advance();
                self.expect(TokenKind::LAngle);
                let ty = self.parse_type().node;
                self.expect(TokenKind::RAngle);
                let args = self.parse_value_pairs_braced()?;
                let p = self.require_named(&args, "p")?;
                Some(AstOp::PtrLoad { ty, p })
            }

            TokenKind::CallableCapture => {
                self.advance();
                let fn_ref = self.parse_fn_ref()?;
                let captures = self.parse_value_pairs_braced()?;
                Some(AstOp::CallableCapture { fn_ref, captures })
            }

            TokenKind::ArrNew => {
                self.advance();
                self.expect(TokenKind::LAngle);
                let elem_ty = self.parse_type().node;
                self.expect(TokenKind::RAngle);
                let args = self.parse_value_pairs_braced()?;
                let cap = self.require_named(&args, "cap")?;
                Some(AstOp::ArrNew { elem_ty, cap })
            }

            TokenKind::ArrLen => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let arr = self.require_named(&args, "arr")?;
                Some(AstOp::ArrLen { arr })
            }

            TokenKind::ArrGet => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let arr = self.require_named(&args, "arr")?;
                let idx = self.require_named(&args, "idx")?;
                Some(AstOp::ArrGet { arr, idx })
            }

            TokenKind::ArrPop => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let arr = self.require_named(&args, "arr")?;
                Some(AstOp::ArrPop { arr })
            }

            TokenKind::ArrSlice => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let arr = self.require_named(&args, "arr")?;
                let start_v = self.require_named(&args, "start")?;
                let end = self.require_named(&args, "end")?;
                Some(AstOp::ArrSlice {
                    arr,
                    start: start_v,
                    end,
                })
            }

            TokenKind::ArrContains => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let arr = self.require_named(&args, "arr")?;
                let val = self.require_named(&args, "val")?;
                Some(AstOp::ArrContains { arr, val })
            }

            TokenKind::ArrMap => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let arr = self.require_named(&args, "arr")?;
                let func = self.require_named(&args, "fn")?;
                Some(AstOp::ArrMap { arr, func })
            }

            TokenKind::ArrFilter => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let arr = self.require_named(&args, "arr")?;
                let func = self.require_named(&args, "fn")?;
                Some(AstOp::ArrFilter { arr, func })
            }

            TokenKind::ArrReduce => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let arr = self.require_named(&args, "arr")?;
                let init = self.require_named(&args, "init")?;
                let func = self.require_named(&args, "fn")?;
                Some(AstOp::ArrReduce { arr, init, func })
            }

            TokenKind::MapNew => {
                self.advance();
                self.expect(TokenKind::LAngle);
                let key_ty = self.parse_type().node;
                self.expect(TokenKind::Comma);
                let val_ty = self.parse_type().node;
                self.expect(TokenKind::RAngle);
                self.parse_empty_braces();
                Some(AstOp::MapNew { key_ty, val_ty })
            }

            TokenKind::MapLen => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let map = self.require_named(&args, "map")?;
                Some(AstOp::MapLen { map })
            }

            TokenKind::MapGet => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let map = self.require_named(&args, "map")?;
                let key = self.require_named(&args, "key")?;
                Some(AstOp::MapGet { map, key })
            }

            TokenKind::MapGetRef => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let map = self.require_named(&args, "map")?;
                let key = self.require_named(&args, "key")?;
                Some(AstOp::MapGetRef { map, key })
            }

            TokenKind::MapDelete => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let map = self.require_named(&args, "map")?;
                let key = self.require_named(&args, "key")?;
                Some(AstOp::MapDelete { map, key })
            }

            TokenKind::MapContainsKey => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let map = self.require_named(&args, "map")?;
                let key = self.require_named(&args, "key")?;
                Some(AstOp::MapContainsKey { map, key })
            }

            TokenKind::MapKeys => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let map = self.require_named(&args, "map")?;
                Some(AstOp::MapKeys { map })
            }

            TokenKind::MapValues => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let map = self.require_named(&args, "map")?;
                Some(AstOp::MapValues { map })
            }

            TokenKind::StrConcat => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let a = self.require_named(&args, "a")?;
                let b = self.require_named(&args, "b")?;
                Some(AstOp::StrConcat { a, b })
            }

            TokenKind::StrLen => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let s = self.require_named(&args, "s")?;
                Some(AstOp::StrLen { s })
            }

            TokenKind::StrEq => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let a = self.require_named(&args, "a")?;
                let b = self.require_named(&args, "b")?;
                Some(AstOp::StrEq { a, b })
            }

            TokenKind::StrSlice => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let s = self.require_named(&args, "s")?;
                let start_v = self.require_named(&args, "start")?;
                let end = self.require_named(&args, "end")?;
                Some(AstOp::StrSlice {
                    s,
                    start: start_v,
                    end,
                })
            }

            TokenKind::StrBytes => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let s = self.require_named(&args, "s")?;
                Some(AstOp::StrBytes { s })
            }

            TokenKind::StrBuilderNew => {
                self.advance();
                self.parse_empty_braces();
                Some(AstOp::StrBuilderNew)
            }

            TokenKind::StrBuilderBuild => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let b = self.require_named(&args, "b")?;
                Some(AstOp::StrBuilderBuild { b })
            }

            TokenKind::StrParseI64 => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let s = self.require_named(&args, "s")?;
                Some(AstOp::StrParseI64 { s })
            }

            TokenKind::StrParseU64 => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let s = self.require_named(&args, "s")?;
                Some(AstOp::StrParseU64 { s })
            }

            TokenKind::StrParseF64 => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let s = self.require_named(&args, "s")?;
                Some(AstOp::StrParseF64 { s })
            }

            TokenKind::StrParseBool => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let s = self.require_named(&args, "s")?;
                Some(AstOp::StrParseBool { s })
            }

            TokenKind::JsonEncode => {
                self.advance();
                self.expect(TokenKind::LAngle);
                let ty = self.parse_type().node;
                self.expect(TokenKind::RAngle);
                let args = self.parse_value_pairs_braced()?;
                let v = self.require_named(&args, "v")?;
                Some(AstOp::JsonEncode { ty, v })
            }

            TokenKind::JsonDecode => {
                self.advance();
                self.expect(TokenKind::LAngle);
                let ty = self.parse_type().node;
                self.expect(TokenKind::RAngle);
                let args = self.parse_value_pairs_braced()?;
                let s = self.require_named(&args, "s")?;
                Some(AstOp::JsonDecode { ty, s })
            }

            TokenKind::GpuThreadId => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let dim = self.require_named(&args, "dim")?;
                Some(AstOp::GpuThreadId { dim })
            }

            TokenKind::GpuWorkgroupId => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let dim = self.require_named(&args, "dim")?;
                Some(AstOp::GpuWorkgroupId { dim })
            }

            TokenKind::GpuWorkgroupSize => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let dim = self.require_named(&args, "dim")?;
                Some(AstOp::GpuWorkgroupSize { dim })
            }

            TokenKind::GpuGlobalId => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let dim = self.require_named(&args, "dim")?;
                Some(AstOp::GpuGlobalId { dim })
            }

            TokenKind::GpuBufferLoad => {
                self.advance();
                self.expect(TokenKind::LAngle);
                let ty = self.parse_type().node;
                self.expect(TokenKind::RAngle);
                let args = self.parse_value_pairs_braced()?;
                let buf = self.require_named(&args, "buf")?;
                let idx = self.require_named(&args, "idx")?;
                Some(AstOp::GpuBufferLoad { ty, buf, idx })
            }

            TokenKind::GpuBufferLen => {
                self.advance();
                self.expect(TokenKind::LAngle);
                let ty = self.parse_type().node;
                self.expect(TokenKind::RAngle);
                let args = self.parse_value_pairs_braced()?;
                let buf = self.require_named(&args, "buf")?;
                Some(AstOp::GpuBufferLen { ty, buf })
            }

            TokenKind::GpuShared => {
                self.advance();
                self.expect(TokenKind::LAngle);
                let count = self.parse_i64_lit().unwrap_or_default();
                self.expect(TokenKind::Comma);
                let ty = self.parse_type().node;
                self.expect(TokenKind::RAngle);
                Some(AstOp::GpuShared { count, ty })
            }

            TokenKind::GpuLaunch => {
                self.advance();
                self.expect(TokenKind::LBrace);
                self.expect_key("device");
                self.expect(TokenKind::Eq);
                let device = self.parse_value_ref(None)?;
                self.expect(TokenKind::Comma);
                self.expect_key("kernel");
                self.expect(TokenKind::Eq);
                let kernel = self.parse_fn_ref()?;
                self.expect(TokenKind::Comma);
                self.expect_key("grid");
                self.expect(TokenKind::Eq);
                let grid = self.parse_arg_value()?;
                self.expect(TokenKind::Comma);
                self.expect_key("block");
                self.expect(TokenKind::Eq);
                let block = self.parse_arg_value()?;
                self.expect(TokenKind::Comma);
                self.expect_key("args");
                self.expect(TokenKind::Eq);
                let args = self.parse_arg_value()?;
                self.expect(TokenKind::RBrace);
                Some(AstOp::GpuLaunch {
                    device,
                    kernel,
                    grid,
                    block,
                    args,
                })
            }

            TokenKind::GpuLaunchAsync => {
                self.advance();
                self.expect(TokenKind::LBrace);
                self.expect_key("device");
                self.expect(TokenKind::Eq);
                let device = self.parse_value_ref(None)?;
                self.expect(TokenKind::Comma);
                self.expect_key("kernel");
                self.expect(TokenKind::Eq);
                let kernel = self.parse_fn_ref()?;
                self.expect(TokenKind::Comma);
                self.expect_key("grid");
                self.expect(TokenKind::Eq);
                let grid = self.parse_arg_value()?;
                self.expect(TokenKind::Comma);
                self.expect_key("block");
                self.expect(TokenKind::Eq);
                let block = self.parse_arg_value()?;
                self.expect(TokenKind::Comma);
                self.expect_key("args");
                self.expect(TokenKind::Eq);
                let args = self.parse_arg_value()?;
                self.expect(TokenKind::RBrace);
                Some(AstOp::GpuLaunchAsync {
                    device,
                    kernel,
                    grid,
                    block,
                    args,
                })
            }

            _ => {
                self.error_here("Expected value-producing operation.");
                None
            }
        }
    }

    fn parse_op_void(&mut self) -> Option<AstOpVoid> {
        match self.peek().kind {
            TokenKind::CallVoid => {
                self.advance();
                let callee = self.parse_fn_ref()?;
                let targs = self.parse_type_args_opt();
                let args = self.parse_arg_pairs_braced()?;
                Some(AstOpVoid::CallVoid {
                    callee,
                    targs,
                    args,
                })
            }
            TokenKind::CallVoidIndirect => {
                self.advance();
                let callee = self.parse_value_ref(None)?;
                let args = self.parse_arg_pairs_braced()?;
                Some(AstOpVoid::CallVoidIndirect { callee, args })
            }
            TokenKind::SetField => {
                self.advance();
                self.expect(TokenKind::LBrace);
                let mut obj = None;
                let mut field = None;
                let mut val = None;
                while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
                    let key = self.parse_ident().unwrap_or_default();
                    self.expect(TokenKind::Eq);
                    match key.as_str() {
                        "obj" => obj = self.parse_value_ref(None),
                        "field" => field = Some(self.parse_ident().unwrap_or_default()),
                        "val" => val = self.parse_value_ref(None),
                        _ => {
                            self.error_here(format!("Unknown key `{}` in setfield", key));
                            break;
                        }
                    }
                    self.eat(TokenKind::Comma);
                }
                self.expect(TokenKind::RBrace);
                let obj = obj.or_else(|| {
                    self.error_here("Missing key `obj` in setfield");
                    None
                })?;
                let field = field.unwrap_or_else(|| {
                    self.error_here("Missing key `field` in setfield");
                    String::new()
                });
                let val = val.or_else(|| {
                    self.error_here("Missing key `val` in setfield");
                    None
                })?;
                Some(AstOpVoid::SetField { obj, field, val })
            }
            TokenKind::Panic => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let msg = self.require_named(&args, "msg")?;
                Some(AstOpVoid::Panic { msg })
            }
            TokenKind::PtrStore => {
                self.advance();
                self.expect(TokenKind::LAngle);
                let ty = self.parse_type().node;
                self.expect(TokenKind::RAngle);
                let args = self.parse_value_pairs_braced()?;
                let p = self.require_named(&args, "p")?;
                let v = self.require_named(&args, "v")?;
                Some(AstOpVoid::PtrStore { ty, p, v })
            }
            TokenKind::ArrSet => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let arr = self.require_named(&args, "arr")?;
                let idx = self.require_named(&args, "idx")?;
                let val = self.require_named(&args, "val")?;
                Some(AstOpVoid::ArrSet { arr, idx, val })
            }
            TokenKind::ArrPush => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let arr = self.require_named(&args, "arr")?;
                let val = self.require_named(&args, "val")?;
                Some(AstOpVoid::ArrPush { arr, val })
            }
            TokenKind::ArrSort => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let arr = self.require_named(&args, "arr")?;
                Some(AstOpVoid::ArrSort { arr })
            }
            TokenKind::ArrForeach => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let arr = self.require_named(&args, "arr")?;
                let func = self.require_named(&args, "fn")?;
                Some(AstOpVoid::ArrForeach { arr, func })
            }
            TokenKind::MapSet => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let map = self.require_named(&args, "map")?;
                let key = self.require_named(&args, "key")?;
                let val = self.require_named(&args, "val")?;
                Some(AstOpVoid::MapSet { map, key, val })
            }
            TokenKind::MapDeleteVoid => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let map = self.require_named(&args, "map")?;
                let key = self.require_named(&args, "key")?;
                Some(AstOpVoid::MapDeleteVoid { map, key })
            }
            TokenKind::StrBuilderAppendStr => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let b = self.require_named(&args, "b")?;
                let s = self.require_named(&args, "s")?;
                Some(AstOpVoid::StrBuilderAppendStr { b, s })
            }
            TokenKind::StrBuilderAppendI64 => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let b = self.require_named(&args, "b")?;
                let v = self.require_named(&args, "v")?;
                Some(AstOpVoid::StrBuilderAppendI64 { b, v })
            }
            TokenKind::StrBuilderAppendI32 => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let b = self.require_named(&args, "b")?;
                let v = self.require_named(&args, "v")?;
                Some(AstOpVoid::StrBuilderAppendI32 { b, v })
            }
            TokenKind::StrBuilderAppendF64 => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let b = self.require_named(&args, "b")?;
                let v = self.require_named(&args, "v")?;
                Some(AstOpVoid::StrBuilderAppendF64 { b, v })
            }
            TokenKind::StrBuilderAppendBool => {
                self.advance();
                let args = self.parse_value_pairs_braced()?;
                let b = self.require_named(&args, "b")?;
                let v = self.require_named(&args, "v")?;
                Some(AstOpVoid::StrBuilderAppendBool { b, v })
            }
            TokenKind::GpuBarrier => {
                self.advance();
                Some(AstOpVoid::GpuBarrier)
            }
            TokenKind::GpuBufferStore => {
                self.advance();
                self.expect(TokenKind::LAngle);
                let ty = self.parse_type().node;
                self.expect(TokenKind::RAngle);
                let args = self.parse_value_pairs_braced()?;
                let buf = self.require_named(&args, "buf")?;
                let idx = self.require_named(&args, "idx")?;
                let v = self.require_named(&args, "v")?;
                Some(AstOpVoid::GpuBufferStore { ty, buf, idx, v })
            }
            _ => {
                self.error_here("Expected void operation.");
                None
            }
        }
    }

    fn parse_type(&mut self) -> Spanned<AstType> {
        let start = self.peek().span;
        let ty = self.parse_type_node().unwrap_or_else(|| self.unit_type());
        Spanned::new(ty, self.span_from(start))
    }

    fn parse_type_node(&mut self) -> Option<AstType> {
        let ownership = if self.at_word("shared") {
            self.advance();
            Some(OwnershipMod::Shared)
        } else if self.at_word("borrow") {
            self.advance();
            Some(OwnershipMod::Borrow)
        } else if self.at_word("mutborrow") {
            self.advance();
            Some(OwnershipMod::MutBorrow)
        } else if self.at_word("weak") {
            self.advance();
            Some(OwnershipMod::Weak)
        } else {
            None
        };

        let base = self.parse_base_type()?;
        Some(AstType { ownership, base })
    }

    fn parse_base_type(&mut self) -> Option<AstBaseType> {
        match self.peek().kind {
            TokenKind::Ident => {
                let text = self.peek().text.clone();
                if is_prim_type(&text) {
                    self.advance();
                    return Some(AstBaseType::Prim(text));
                }

                match text.as_str() {
                    "Str" => {
                        self.advance();
                        Some(AstBaseType::Builtin(AstBuiltinType::Str))
                    }
                    "Array" => {
                        self.advance();
                        self.expect(TokenKind::LAngle);
                        let elem = self.parse_type().node;
                        self.expect(TokenKind::RAngle);
                        Some(AstBaseType::Builtin(AstBuiltinType::Array(Box::new(elem))))
                    }
                    "Map" => {
                        self.advance();
                        self.expect(TokenKind::LAngle);
                        let key = self.parse_type().node;
                        self.expect(TokenKind::Comma);
                        let val = self.parse_type().node;
                        self.expect(TokenKind::RAngle);
                        Some(AstBaseType::Builtin(AstBuiltinType::Map(
                            Box::new(key),
                            Box::new(val),
                        )))
                    }
                    "rawptr" => {
                        self.advance();
                        self.expect(TokenKind::LAngle);
                        let inner = self.parse_type().node;
                        self.expect(TokenKind::RAngle);
                        Some(AstBaseType::RawPtr(Box::new(inner)))
                    }
                    _ => self.parse_named_type_from_ident_path(),
                }
            }

            TokenKind::TypeName => {
                let name = self.peek().text.clone();
                match name.as_str() {
                    "TOption" => {
                        self.advance();
                        self.expect(TokenKind::LAngle);
                        let t = self.parse_type().node;
                        self.expect(TokenKind::RAngle);
                        Some(AstBaseType::Builtin(AstBuiltinType::TOption(Box::new(t))))
                    }
                    "TResult" => {
                        self.advance();
                        self.expect(TokenKind::LAngle);
                        let a = self.parse_type().node;
                        self.expect(TokenKind::Comma);
                        let b = self.parse_type().node;
                        self.expect(TokenKind::RAngle);
                        Some(AstBaseType::Builtin(AstBuiltinType::TResult(
                            Box::new(a),
                            Box::new(b),
                        )))
                    }
                    "TStrBuilder" => {
                        self.advance();
                        Some(AstBaseType::Builtin(AstBuiltinType::TStrBuilder))
                    }
                    "TMutex" => {
                        self.advance();
                        self.expect(TokenKind::LAngle);
                        let t = self.parse_type().node;
                        self.expect(TokenKind::RAngle);
                        Some(AstBaseType::Builtin(AstBuiltinType::TMutex(Box::new(t))))
                    }
                    "TRwLock" => {
                        self.advance();
                        self.expect(TokenKind::LAngle);
                        let t = self.parse_type().node;
                        self.expect(TokenKind::RAngle);
                        Some(AstBaseType::Builtin(AstBuiltinType::TRwLock(Box::new(t))))
                    }
                    "TCell" => {
                        self.advance();
                        self.expect(TokenKind::LAngle);
                        let t = self.parse_type().node;
                        self.expect(TokenKind::RAngle);
                        Some(AstBaseType::Builtin(AstBuiltinType::TCell(Box::new(t))))
                    }
                    "TFuture" => {
                        self.advance();
                        self.expect(TokenKind::LAngle);
                        let t = self.parse_type().node;
                        self.expect(TokenKind::RAngle);
                        Some(AstBaseType::Builtin(AstBuiltinType::TFuture(Box::new(t))))
                    }
                    "TChannelSend" => {
                        self.advance();
                        self.expect(TokenKind::LAngle);
                        let t = self.parse_type().node;
                        self.expect(TokenKind::RAngle);
                        Some(AstBaseType::Builtin(AstBuiltinType::TChannelSend(
                            Box::new(t),
                        )))
                    }
                    "TChannelRecv" => {
                        self.advance();
                        self.expect(TokenKind::LAngle);
                        let t = self.parse_type().node;
                        self.expect(TokenKind::RAngle);
                        Some(AstBaseType::Builtin(AstBuiltinType::TChannelRecv(
                            Box::new(t),
                        )))
                    }
                    "TCallable" => {
                        self.advance();
                        self.expect(TokenKind::LAngle);
                        let (path, sig_name) = self.parse_type_ref()?;
                        self.expect(TokenKind::RAngle);
                        let sig_ref = if let Some(p) = path {
                            format!("{}.{}", p, sig_name)
                        } else {
                            sig_name
                        };
                        Some(AstBaseType::Callable { sig_ref })
                    }
                    _ => {
                        self.advance();
                        let targs = self.parse_type_args_opt();
                        Some(AstBaseType::Named {
                            path: None,
                            name,
                            targs,
                        })
                    }
                }
            }

            _ => {
                self.error_here("Expected type.");
                None
            }
        }
    }

    fn parse_named_type_from_ident_path(&mut self) -> Option<AstBaseType> {
        let mut segments = vec![self.parse_ident()?];

        loop {
            if self.eat(TokenKind::Dot).is_none() {
                self.error_here("Expected `.` followed by type name.");
                return None;
            }

            if self.at(TokenKind::TypeName) {
                let name = self.advance().text;
                let targs = self.parse_type_args_opt();
                return Some(AstBaseType::Named {
                    path: Some(ModulePath { segments }),
                    name,
                    targs,
                });
            }

            if self.at(TokenKind::Ident) {
                segments.push(self.advance().text);
                continue;
            }

            self.error_here("Expected module path segment or type name.");
            return None;
        }
    }

    fn parse_type_ref(&mut self) -> Option<(Option<ModulePath>, String)> {
        if self.at(TokenKind::TypeName) {
            return Some((None, self.advance().text));
        }

        if !self.at(TokenKind::Ident) {
            self.error_here("Expected type reference.");
            return None;
        }

        let mut segments = vec![self.advance().text];
        loop {
            self.expect(TokenKind::Dot);
            if self.at(TokenKind::TypeName) {
                let name = self.advance().text;
                return Some((Some(ModulePath { segments }), name));
            }
            if self.at(TokenKind::Ident) {
                segments.push(self.advance().text);
                continue;
            }
            self.error_here("Expected module segment or type name.");
            return None;
        }
    }

    fn parse_type_args_opt(&mut self) -> Vec<AstType> {
        if self.eat(TokenKind::LAngle).is_none() {
            return Vec::new();
        }

        let mut targs = Vec::new();
        while !self.at(TokenKind::RAngle) && !self.at(TokenKind::Eof) {
            targs.push(self.parse_type().node);
            if self.eat(TokenKind::Comma).is_none() {
                break;
            }
        }

        self.expect(TokenKind::RAngle);
        targs
    }

    fn parse_type_params_opt(&mut self) -> Vec<AstTypeParam> {
        if self.eat(TokenKind::LAngle).is_none() {
            return Vec::new();
        }

        let mut params = Vec::new();
        while !self.at(TokenKind::RAngle) && !self.at(TokenKind::Eof) {
            let name = self.parse_ident().unwrap_or_default();
            self.expect(TokenKind::Colon);
            let constraint = self.parse_ident().unwrap_or_else(|| "type".to_string());
            params.push(AstTypeParam { name, constraint });
            if self.eat(TokenKind::Comma).is_none() {
                break;
            }
        }

        self.expect(TokenKind::RAngle);
        params
    }

    fn parse_params_list(&mut self, until: TokenKind) -> Vec<AstParam> {
        let mut params = Vec::new();

        while !self.at(until) && !self.at(TokenKind::Eof) {
            let name = self.parse_ssa_name().unwrap_or_default();
            self.expect(TokenKind::Colon);
            let ty = self.parse_type();
            params.push(AstParam { name, ty });
            if self.eat(TokenKind::Comma).is_none() {
                break;
            }
        }

        params
    }

    fn parse_value_pairs_braced(&mut self) -> Option<Vec<(String, AstValueRef)>> {
        self.expect(TokenKind::LBrace);
        let mut pairs = Vec::new();

        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            let key = self.parse_key_name()?;
            self.expect(TokenKind::Eq);
            let value = self.parse_value_ref(None)?;
            pairs.push((key, value));
            if self.eat(TokenKind::Comma).is_none() {
                break;
            }
        }

        self.expect(TokenKind::RBrace);
        Some(pairs)
    }

    fn parse_arg_pairs_braced(&mut self) -> Option<Vec<(String, AstArgValue)>> {
        self.expect(TokenKind::LBrace);
        let mut pairs = Vec::new();

        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            let key = self.parse_key_name()?;
            self.expect(TokenKind::Eq);
            let value = self.parse_arg_value()?;
            pairs.push((key, value));
            if self.eat(TokenKind::Comma).is_none() {
                break;
            }
        }

        self.expect(TokenKind::RBrace);
        Some(pairs)
    }

    fn parse_arg_value(&mut self) -> Option<AstArgValue> {
        if self.at(TokenKind::LBracket) {
            self.advance();
            let mut elems = Vec::new();
            while !self.at(TokenKind::RBracket) && !self.at(TokenKind::Eof) {
                elems.push(self.parse_arg_list_elem()?);
                if self.eat(TokenKind::Comma).is_none() {
                    break;
                }
            }
            self.expect(TokenKind::RBracket);
            return Some(AstArgValue::List(elems));
        }

        if self.at(TokenKind::SsaName) || self.at(TokenKind::ConstOp) {
            return Some(AstArgValue::Value(self.parse_value_ref(None)?));
        }

        Some(AstArgValue::FnRef(self.parse_fn_ref()?))
    }

    fn parse_arg_list_elem(&mut self) -> Option<AstArgListElem> {
        if self.at(TokenKind::SsaName) || self.at(TokenKind::ConstOp) {
            return Some(AstArgListElem::Value(self.parse_value_ref(None)?));
        }

        Some(AstArgListElem::FnRef(self.parse_fn_ref()?))
    }

    fn parse_value_ref(&mut self, hint: Option<AstType>) -> Option<AstValueRef> {
        match self.peek().kind {
            TokenKind::SsaName => {
                let name = self.parse_ssa_name()?;
                Some(AstValueRef::Local(name))
            }
            TokenKind::ConstOp => {
                let c = self.parse_const_expr_with_hint(hint)?;
                Some(AstValueRef::Const(c))
            }
            _ => {
                self.error_here("Expected value reference (`%name` or const expression).");
                None
            }
        }
    }

    fn parse_const_expr_with_hint(&mut self, hint: Option<AstType>) -> Option<AstConstExpr> {
        let const_tok = self.expect(TokenKind::ConstOp);

        // Extract type suffix from `const.i32`, `const.f64`, `const.Str`, etc.
        let suffix_type = const_tok.as_ref().and_then(|tok| {
            let suffix = tok.text.as_str();
            if suffix.is_empty() {
                None
            } else if is_prim_type(suffix) {
                Some(AstType {
                    ownership: None,
                    base: AstBaseType::Prim(suffix.to_string()),
                })
            } else if suffix == "Str" {
                Some(AstType {
                    ownership: None,
                    base: AstBaseType::Builtin(AstBuiltinType::Str),
                })
            } else if suffix == "bool" {
                Some(AstType {
                    ownership: None,
                    base: AstBaseType::Prim("bool".to_string()),
                })
            } else {
                None
            }
        });

        let explicit_type = if self.const_expr_has_inline_type() {
            self.parse_type_node()
        } else {
            None
        };

        let lit = self.parse_const_lit()?;
        let ty = explicit_type
            .or(suffix_type)
            .or(hint)
            .unwrap_or_else(|| self.infer_const_type(&lit));
        Some(AstConstExpr { ty, lit })
    }

    fn const_expr_has_inline_type(&self) -> bool {
        match self.peek().kind {
            TokenKind::IntLit
            | TokenKind::FloatLit
            | TokenKind::StringLit
            | TokenKind::True
            | TokenKind::False => false,
            TokenKind::Ident if self.peek().text == "unit" => false,
            _ => self.looks_like_type_start(),
        }
    }

    fn looks_like_type_start(&self) -> bool {
        match self.peek().kind {
            TokenKind::TypeName => true,
            TokenKind::Ident => {
                let t = self.peek().text.as_str();
                is_prim_type(t)
                    || matches!(
                        t,
                        "shared"
                            | "borrow"
                            | "mutborrow"
                            | "weak"
                            | "Str"
                            | "Array"
                            | "Map"
                            | "rawptr"
                    )
                    || t.chars().next().is_some_and(|c| c.is_ascii_lowercase())
            }
            _ => false,
        }
    }

    fn parse_const_lit(&mut self) -> Option<AstConstLit> {
        match self.peek().kind {
            TokenKind::IntLit => {
                let tok = self.advance();
                let value = parse_i128_lit(&tok.text).unwrap_or_else(|| {
                    self.error_at(tok.span, "Invalid integer literal.");
                    0
                });
                Some(AstConstLit::Int(value))
            }
            TokenKind::FloatLit => {
                let tok = self.advance();
                let text = tok
                    .text
                    .trim_end_matches("f32")
                    .trim_end_matches("f64")
                    .to_string();
                let value = text.parse::<f64>().unwrap_or_else(|_| {
                    self.error_at(tok.span, "Invalid float literal.");
                    0.0
                });
                Some(AstConstLit::Float(value))
            }
            TokenKind::StringLit => Some(AstConstLit::Str(self.advance().text)),
            TokenKind::True => {
                self.advance();
                Some(AstConstLit::Bool(true))
            }
            TokenKind::False => {
                self.advance();
                Some(AstConstLit::Bool(false))
            }
            TokenKind::Ident if self.peek().text == "unit" => {
                self.advance();
                Some(AstConstLit::Unit)
            }
            _ => {
                self.error_here("Expected constant literal.");
                None
            }
        }
    }

    fn parse_fn_ref(&mut self) -> Option<String> {
        if self.at(TokenKind::FnName) {
            return Some(self.advance().text);
        }

        if !self.at(TokenKind::Ident) {
            self.error_here("Expected function reference.");
            return None;
        }

        let mut segments = vec![self.advance().text];
        loop {
            self.expect(TokenKind::Dot);
            if self.at(TokenKind::FnName) {
                let fn_name = self.advance().text;
                return Some(format!("{}.{}", segments.join("."), fn_name));
            }
            if self.at(TokenKind::Ident) {
                segments.push(self.advance().text);
                continue;
            }
            self.error_here("Expected module segment or function name.");
            return None;
        }
    }

    fn parse_fqn_ref(&mut self) -> Option<String> {
        if self.at(TokenKind::FnName) || self.at(TokenKind::TypeName) {
            return Some(self.advance().text);
        }

        if !self.at(TokenKind::Ident) {
            self.error_here("Expected fully-qualified symbol reference.");
            return None;
        }

        let mut segments = vec![self.advance().text];
        loop {
            self.expect(TokenKind::Dot);
            if self.at(TokenKind::FnName) || self.at(TokenKind::TypeName) {
                let tail = self.advance().text;
                return Some(format!("{}.{}", segments.join("."), tail));
            }
            if self.at(TokenKind::Ident) {
                segments.push(self.advance().text);
                continue;
            }
            self.error_here("Expected module segment, `@fn`, or `TType`.");
            return None;
        }
    }

    fn parse_module_path(&mut self) -> Option<ModulePath> {
        let mut segments = Vec::new();
        segments.push(self.parse_ident_or_keyword()?);
        while self.eat(TokenKind::Dot).is_some() {
            segments.push(self.parse_ident_or_keyword()?);
        }
        Some(ModulePath { segments })
    }

    /// Accept an identifier or a keyword token as a name (for module paths where
    /// keywords like `async` may appear as segments).
    fn parse_ident_or_keyword(&mut self) -> Option<String> {
        let kind = self.peek().kind;
        match kind {
            TokenKind::Ident => Some(self.advance().text),
            _ if Self::keyword_as_ident(kind).is_some() => {
                self.advance();
                Some(Self::keyword_as_ident(kind).unwrap().to_string())
            }
            _ => {
                self.error_here("Expected identifier.");
                None
            }
        }
    }

    fn keyword_as_ident(kind: TokenKind) -> Option<&'static str> {
        match kind {
            TokenKind::Async => Some("async"),
            TokenKind::Unsafe => Some("unsafe"),
            TokenKind::Gpu => Some("gpu"),
            TokenKind::Target => Some("target"),
            TokenKind::Heap => Some("heap"),
            TokenKind::Value => Some("value"),
            TokenKind::Fn => Some("fn"),
            TokenKind::Meta => Some("meta"),
            _ => None,
        }
    }

    fn parse_fn_name(&mut self) -> Option<String> {
        if !self.at(TokenKind::FnName) {
            self.error_here("Expected function name (`@name`).");
            return None;
        }
        Some(self.advance().text)
    }

    fn parse_ssa_name(&mut self) -> Option<String> {
        if !self.at(TokenKind::SsaName) {
            self.error_here("Expected SSA name (`%name`).");
            return None;
        }
        let text = self.advance().text;
        Some(text.strip_prefix('%').unwrap_or(&text).to_string())
    }

    fn parse_type_name(&mut self) -> Option<String> {
        if !self.at(TokenKind::TypeName) {
            self.error_here("Expected type name (`TName`).");
            return None;
        }
        Some(self.advance().text)
    }

    fn parse_ident(&mut self) -> Option<String> {
        if !self.at(TokenKind::Ident) {
            self.error_here("Expected identifier.");
            return None;
        }
        Some(self.advance().text)
    }

    fn parse_ident_or_type_name(&mut self) -> Option<String> {
        match self.peek().kind {
            TokenKind::Ident | TokenKind::TypeName => Some(self.advance().text),
            _ => {
                self.error_here("Expected identifier.");
                None
            }
        }
    }

    fn parse_string_lit(&mut self) -> Option<String> {
        if !self.at(TokenKind::StringLit) {
            self.error_here("Expected string literal.");
            return None;
        }
        Some(self.advance().text)
    }

    fn parse_i64_lit(&mut self) -> Option<i64> {
        if !self.at(TokenKind::IntLit) {
            self.error_here("Expected integer literal.");
            return None;
        }
        let tok = self.advance();
        let value = parse_i128_lit(&tok.text)
            .ok_or(())
            .and_then(|v| i64::try_from(v).map_err(|_| ()))
            .unwrap_or_else(|_| {
                self.error_at(tok.span, "Integer literal out of i64 range.");
                0
            });
        Some(value)
    }

    fn parse_block_label_num(&mut self) -> Option<u32> {
        if !self.at(TokenKind::BlockLabel) {
            self.error_here("Expected block label (`bbN`).");
            return None;
        }

        let tok = self.advance();
        let digits = tok.text.strip_prefix("bb").unwrap_or_default();
        match digits.parse::<u32>() {
            Ok(v) => Some(v),
            Err(_) => {
                self.error_at(tok.span, "Invalid block label.");
                Some(0)
            }
        }
    }

    fn parse_key_name(&mut self) -> Option<String> {
        match self.peek().kind {
            TokenKind::Ident => Some(self.advance().text),
            TokenKind::Fn => {
                self.advance();
                Some("fn".to_string())
            }
            _ => {
                self.error_here("Expected argument key.");
                None
            }
        }
    }

    fn parse_prim_type_only(&mut self) -> AstType {
        if self.at(TokenKind::Ident) && is_prim_type(&self.peek().text) {
            let prim = self.advance().text;
            return AstType {
                ownership: None,
                base: AstBaseType::Prim(prim),
            };
        }

        self.error_here("Expected primitive type.");
        self.unit_type()
    }

    fn require_named<T: Clone>(&mut self, pairs: &[(String, T)], key: &str) -> Option<T> {
        if let Some((_, value)) = pairs.iter().find(|(k, _)| k == key) {
            return Some(value.clone());
        }
        self.error_here(format!("Missing required key `{}`.", key));
        None
    }

    fn parse_empty_braces(&mut self) {
        self.expect(TokenKind::LBrace);
        self.expect(TokenKind::RBrace);
    }

    fn unit_type(&self) -> AstType {
        AstType {
            ownership: None,
            base: AstBaseType::Prim("unit".to_string()),
        }
    }

    fn infer_const_type(&self, lit: &AstConstLit) -> AstType {
        match lit {
            AstConstLit::Int(_) => AstType {
                ownership: None,
                base: AstBaseType::Prim("i64".to_string()),
            },
            AstConstLit::Float(_) => AstType {
                ownership: None,
                base: AstBaseType::Prim("f64".to_string()),
            },
            AstConstLit::Str(_) => AstType {
                ownership: None,
                base: AstBaseType::Builtin(AstBuiltinType::Str),
            },
            AstConstLit::Bool(_) => AstType {
                ownership: None,
                base: AstBaseType::Prim("bool".to_string()),
            },
            AstConstLit::Unit => self.unit_type(),
        }
    }

    fn bin_op_kind(&mut self, kind: TokenKind) -> Option<BinOpKind> {
        Some(match kind {
            TokenKind::IAdd => BinOpKind::IAdd,
            TokenKind::ISub => BinOpKind::ISub,
            TokenKind::IMul => BinOpKind::IMul,
            TokenKind::ISdiv => BinOpKind::ISDiv,
            TokenKind::IUdiv => BinOpKind::IUDiv,
            TokenKind::ISrem => BinOpKind::ISRem,
            TokenKind::IUrem => BinOpKind::IURem,
            TokenKind::IAddWrap => BinOpKind::IAddWrap,
            TokenKind::ISubWrap => BinOpKind::ISubWrap,
            TokenKind::IMulWrap => BinOpKind::IMulWrap,
            TokenKind::IAddChecked => BinOpKind::IAddChecked,
            TokenKind::ISubChecked => BinOpKind::ISubChecked,
            TokenKind::IMulChecked => BinOpKind::IMulChecked,
            TokenKind::IAnd => BinOpKind::IAnd,
            TokenKind::IOr => BinOpKind::IOr,
            TokenKind::IXor => BinOpKind::IXor,
            TokenKind::IShl => BinOpKind::IShl,
            TokenKind::ILshr => BinOpKind::ILshr,
            TokenKind::IAshr => BinOpKind::IAshr,
            TokenKind::FAdd => BinOpKind::FAdd,
            TokenKind::FSub => BinOpKind::FSub,
            TokenKind::FMul => BinOpKind::FMul,
            TokenKind::FDiv => BinOpKind::FDiv,
            TokenKind::FRem => BinOpKind::FRem,
            TokenKind::FAddFast => BinOpKind::FAddFast,
            TokenKind::FSubFast => BinOpKind::FSubFast,
            TokenKind::FMulFast => BinOpKind::FMulFast,
            TokenKind::FDivFast => BinOpKind::FDivFast,
            _ => {
                self.error_here("Internal parser error: expected binary op token.");
                return None;
            }
        })
    }

    fn cmp_kind(&mut self, kind: TokenKind) -> Option<(CmpKind, &'static str)> {
        Some(match kind {
            TokenKind::IcmpEq => (CmpKind::ICmp, "eq"),
            TokenKind::IcmpNe => (CmpKind::ICmp, "ne"),
            TokenKind::IcmpSlt => (CmpKind::ICmp, "slt"),
            TokenKind::IcmpSgt => (CmpKind::ICmp, "sgt"),
            TokenKind::IcmpSle => (CmpKind::ICmp, "sle"),
            TokenKind::IcmpSge => (CmpKind::ICmp, "sge"),
            TokenKind::IcmpUlt => (CmpKind::ICmp, "ult"),
            TokenKind::IcmpUgt => (CmpKind::ICmp, "ugt"),
            TokenKind::IcmpUle => (CmpKind::ICmp, "ule"),
            TokenKind::IcmpUge => (CmpKind::ICmp, "uge"),
            TokenKind::FcmpOeq => (CmpKind::FCmp, "oeq"),
            TokenKind::FcmpOne => (CmpKind::FCmp, "one"),
            TokenKind::FcmpOlt => (CmpKind::FCmp, "olt"),
            TokenKind::FcmpOgt => (CmpKind::FCmp, "ogt"),
            TokenKind::FcmpOle => (CmpKind::FCmp, "ole"),
            TokenKind::FcmpOge => (CmpKind::FCmp, "oge"),
            _ => {
                self.error_here("Internal parser error: expected cmp op token.");
                return None;
            }
        })
    }

    fn expect_double_colon(&mut self) {
        self.expect(TokenKind::Colon);
        self.expect(TokenKind::Colon);
    }

    fn expect_word(&mut self, word: &str) -> bool {
        if self.at_word(word) {
            self.advance();
            true
        } else {
            self.error_here(format!("Expected `{}`.", word));
            false
        }
    }

    fn expect_key(&mut self, key: &str) -> bool {
        if let Some(actual) = self.parse_key_name() {
            if actual == key {
                return true;
            }
            self.error_here(format!("Expected key `{}` but found `{}`.", key, actual));
        }
        false
    }

    fn at_word(&self, word: &str) -> bool {
        self.at(TokenKind::Ident) && self.peek().text == word
    }

    fn at_terminator_start(&self) -> bool {
        self.at_word("ret")
            || self.at_word("br")
            || self.at_word("cbr")
            || self.at_word("switch")
            || self.at_word("unreachable")
    }

    fn at_op_void_start(&self) -> bool {
        matches!(
            self.peek().kind,
            TokenKind::CallVoid
                | TokenKind::CallVoidIndirect
                | TokenKind::SetField
                | TokenKind::Panic
                | TokenKind::PtrStore
                | TokenKind::ArrSet
                | TokenKind::ArrPush
                | TokenKind::ArrSort
                | TokenKind::ArrForeach
                | TokenKind::MapSet
                | TokenKind::MapDeleteVoid
                | TokenKind::StrBuilderAppendStr
                | TokenKind::StrBuilderAppendI64
                | TokenKind::StrBuilderAppendI32
                | TokenKind::StrBuilderAppendF64
                | TokenKind::StrBuilderAppendBool
                | TokenKind::GpuBarrier
                | TokenKind::GpuBufferStore
        )
    }

    fn at_decl_start(&self) -> bool {
        matches!(
            self.peek().kind,
            TokenKind::DocComment
                | TokenKind::Fn
                | TokenKind::Async
                | TokenKind::Unsafe
                | TokenKind::Gpu
                | TokenKind::Heap
                | TokenKind::Value
                | TokenKind::Extern
                | TokenKind::Global
                | TokenKind::Impl
                | TokenKind::Sig
        )
    }

    fn recover_to_next_decl(&mut self) {
        while !self.at(TokenKind::Eof) {
            if self.at_decl_start() {
                break;
            }
            self.advance();
        }
    }

    fn recover_to_type_body(&mut self) {
        while !self.at(TokenKind::Eof) {
            if self.at_word("field") || self.at_word("variant") || self.at(TokenKind::RBrace) {
                break;
            }
            self.advance();
        }
    }

    fn recover_to_variant_body(&mut self) {
        while !self.at(TokenKind::Eof) {
            if self.at_word("field") || self.at(TokenKind::RBrace) {
                break;
            }
            self.advance();
        }
    }

    fn recover_to_block_boundary(&mut self) {
        while !self.at(TokenKind::Eof) {
            if self.at(TokenKind::BlockLabel) || self.at(TokenKind::RBrace) {
                break;
            }
            self.advance();
        }
    }

    fn recover_to_block_stmt_boundary(&mut self) {
        while !self.at(TokenKind::Eof) {
            if self.at(TokenKind::BlockLabel)
                || self.at(TokenKind::RBrace)
                || self.at_terminator_start()
                || self.at(TokenKind::SsaName)
                || self.at_op_void_start()
                || (self.at(TokenKind::Unsafe) && self.peek_n_kind(1) == TokenKind::LBrace)
            {
                break;
            }
            self.advance();
        }
    }

    fn error_here(&mut self, message: impl Into<String>) {
        self.error_at(self.peek().span, message);
    }

    fn error_at(&mut self, span: Span, message: impl Into<String>) {
        let message = message.into();
        self.diag.emit(Diagnostic {
            code: "MPP0001".to_string(),
            severity: Severity::Error,
            title: "Parse error".to_string(),
            primary_span: Some(span),
            secondary_spans: Vec::new(),
            message,
            explanation_md: None,
            why: None,
            suggested_fixes: Vec::new(),
            rag_bundle: Vec::new(),
            related_docs: Vec::new(),
        });
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&self.fallback_eof)
    }

    fn peek_n_kind(&self, n: usize) -> TokenKind {
        self.tokens
            .get(self.pos.saturating_add(n))
            .map(|t| t.kind)
            .unwrap_or(TokenKind::Eof)
    }

    fn advance(&mut self) -> Token {
        let tok = self.peek().clone();
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    fn at(&self, kind: TokenKind) -> bool {
        self.peek().kind == kind
    }

    fn eat(&mut self, kind: TokenKind) -> Option<Token> {
        if self.at(kind) {
            Some(self.advance())
        } else {
            None
        }
    }

    fn expect(&mut self, kind: TokenKind) -> Option<Token> {
        if self.at(kind) {
            Some(self.advance())
        } else {
            self.error_here(format!("Expected token `{:?}`.", kind));
            None
        }
    }

    fn prev_span(&self) -> Span {
        if self.pos == 0 {
            self.peek().span
        } else {
            self.tokens
                .get(self.pos - 1)
                .map(|t| t.span)
                .unwrap_or(self.peek().span)
        }
    }

    fn span_from(&self, start: Span) -> Span {
        Span::new(self.file_id, start.start, self.prev_span().end)
    }
}

enum FnFlavor {
    Regular,
    Async,
    Unsafe,
    Gpu,
}

fn is_prim_type(name: &str) -> bool {
    matches!(
        name,
        "i1" | "i8"
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
            | "bool"
            | "unit"
    )
}

fn parse_i128_lit(text: &str) -> Option<i128> {
    if text.starts_with("0x") || text.starts_with("0X") {
        i128::from_str_radix(&text[2..], 16).ok()
    } else {
        text.parse::<i128>().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::parse_file;
    use magpie_ast::{AstDecl, FileId};
    use magpie_csnf::format_csnf;
    use magpie_diag::DiagnosticBag;

    fn parse_fixture(path: &str) -> (magpie_ast::AstFile, DiagnosticBag) {
        let src = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", path, e));
        let mut diag = DiagnosticBag::new(100);
        let file_id = FileId(0);
        let tokens = magpie_lex::lex(file_id, &src, &mut diag);
        assert!(
            !diag.has_errors(),
            "lexer diagnostics for {}: {:?}",
            path,
            diag.diagnostics
        );

        let ast = parse_file(&tokens, file_id, &mut diag)
            .unwrap_or_else(|_| panic!("parse failed for {}: {:?}", path, diag.diagnostics));
        (ast, diag)
    }

    #[test]
    fn test_parse_hello() {
        let (ast, _) = parse_fixture("../../tests/fixtures/hello.mp");
        assert_eq!(
            ast.header.node.module_path.node.segments,
            vec!["hello", "main"]
        );
        assert_eq!(ast.decls.len(), 1);
        assert!(matches!(ast.decls[0].node, AstDecl::Fn(_)));
    }

    #[test]
    fn test_parse_arithmetic() {
        let (ast, _) = parse_fixture("../../tests/fixtures/arithmetic.mp");
        assert_eq!(
            ast.header.node.module_path.node.segments,
            vec!["test", "arithmetic"]
        );
        assert_eq!(ast.decls.len(), 2);
    }

    #[test]
    fn test_parse_ownership() {
        let (ast, _) = parse_fixture("../../tests/fixtures/ownership.mp");
        assert_eq!(
            ast.header.node.module_path.node.segments,
            vec!["test", "ownership"]
        );
        // 1 struct + 1 fn
        assert_eq!(ast.decls.len(), 2);
        assert!(matches!(ast.decls[0].node, AstDecl::HeapStruct(_)));
    }

    #[test]
    fn test_parse_enum_match() {
        let (ast, _) = parse_fixture("../../tests/fixtures/enum_match.mp");
        assert_eq!(
            ast.header.node.module_path.node.segments,
            vec!["test", "enum_match"]
        );
        // 1 enum + 1 fn
        assert_eq!(ast.decls.len(), 2);
        assert!(matches!(ast.decls[0].node, AstDecl::HeapEnum(_)));
    }

    #[test]
    fn test_parse_collections() {
        let (ast, _) = parse_fixture("../../tests/fixtures/collections.mp");
        assert_eq!(
            ast.header.node.module_path.node.segments,
            vec!["test", "collections"]
        );
        // 2 fn (Str has built-in hash/eq impls)
        assert_eq!(ast.decls.len(), 2);
        assert!(matches!(ast.decls[0].node, AstDecl::Fn(_)));
        assert!(matches!(ast.decls[1].node, AstDecl::Fn(_)));
    }

    #[test]
    fn test_parse_async_fn() {
        let (ast, _) = parse_fixture("../../tests/fixtures/async_fn.mp");
        assert_eq!(
            ast.header.node.module_path.node.segments,
            vec!["test", "async_example"]
        );
        // 2 helper fns + 1 async fn
        assert_eq!(ast.decls.len(), 3);
        assert!(matches!(ast.decls[0].node, AstDecl::Fn(_)));
        assert!(matches!(ast.decls[1].node, AstDecl::Fn(_)));
        assert!(matches!(ast.decls[2].node, AstDecl::AsyncFn(_)));
    }

    #[test]
    fn test_parse_try_error() {
        let (ast, _) = parse_fixture("../../tests/fixtures/try_error.mp");
        assert_eq!(
            ast.header.node.module_path.node.segments,
            vec!["test", "try_error"]
        );
        assert_eq!(ast.decls.len(), 1);
    }

    fn canonical_fixture(path: &str) -> String {
        let (ast, diag) = parse_fixture(path);
        assert!(
            !diag.has_errors(),
            "fixture should parse without errors: {:?}",
            diag.diagnostics
                .iter()
                .map(|d| d.code.as_str())
                .collect::<Vec<_>>()
        );
        format_csnf(&ast)
    }

    #[test]
    fn snapshot_hello_parser_output() {
        insta::assert_snapshot!(canonical_fixture("../../tests/fixtures/hello.mp"));
    }

    #[test]
    fn snapshot_arithmetic_parser_output() {
        insta::assert_snapshot!(canonical_fixture("../../tests/fixtures/arithmetic.mp"));
    }
}
