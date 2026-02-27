//! Magpie lexer.

use magpie_ast::{FileId, Span};
use magpie_diag::{Diagnostic, DiagnosticBag, Severity};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    pub text: String,
}

impl Token {
    fn new(kind: TokenKind, span: Span, text: String) -> Self {
        Self { kind, span, text }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TokenKind {
    // Header / declaration keywords.
    Module,
    Exports,
    Imports,
    Digest,
    Fn,
    Async,
    Meta,
    Uses,
    Effects,
    Cost,
    Heap,
    Value,
    Struct,
    Enum,
    Extern,
    Global,
    Unsafe,
    Gpu,
    Target,
    Sig,
    Impl,

    // Op keywords.
    ConstOp,
    IAdd,
    ISub,
    IMul,
    ISdiv,
    IUdiv,
    ISrem,
    IUrem,
    IAddWrap,
    ISubWrap,
    IMulWrap,
    IAddChecked,
    ISubChecked,
    IMulChecked,
    IAnd,
    IOr,
    IXor,
    IShl,
    ILshr,
    IAshr,
    FAdd,
    FSub,
    FMul,
    FDiv,
    FRem,
    FAddFast,
    FSubFast,
    FMulFast,
    FDivFast,
    IcmpEq,
    IcmpNe,
    IcmpSlt,
    IcmpSgt,
    IcmpSle,
    IcmpSge,
    IcmpUlt,
    IcmpUgt,
    IcmpUle,
    IcmpUge,
    FcmpOeq,
    FcmpOne,
    FcmpOlt,
    FcmpOgt,
    FcmpOle,
    FcmpOge,
    Call,
    CallVoid,
    CallIndirect,
    CallVoidIndirect,
    Try,
    SuspendCall,
    SuspendAwait,
    New,
    GetField,
    SetField,
    Phi,
    EnumNew,
    EnumTag,
    EnumPayload,
    EnumIs,
    Share,
    CloneShared,
    CloneWeak,
    WeakDowngrade,
    WeakUpgrade,
    Cast,
    BorrowShared,
    BorrowMut,
    PtrNull,
    PtrAddr,
    PtrFromAddr,
    PtrAdd,
    PtrLoad,
    PtrStore,
    CallableCapture,
    ArrNew,
    ArrLen,
    ArrGet,
    ArrSet,
    ArrPush,
    ArrPop,
    ArrSlice,
    ArrContains,
    ArrSort,
    ArrMap,
    ArrFilter,
    ArrReduce,
    ArrForeach,
    MapNew,
    MapLen,
    MapGet,
    MapGetRef,
    MapSet,
    MapDelete,
    MapDeleteVoid,
    MapContainsKey,
    MapKeys,
    MapValues,
    StrConcat,
    StrLen,
    StrEq,
    StrSlice,
    StrBytes,
    StrParseI64,
    StrParseU64,
    StrParseF64,
    StrParseBool,
    StrBuilderNew,
    StrBuilderAppendStr,
    StrBuilderAppendI64,
    StrBuilderAppendI32,
    StrBuilderAppendF64,
    StrBuilderAppendBool,
    StrBuilderBuild,
    JsonEncode,
    JsonDecode,
    GpuThreadId,
    GpuWorkgroupId,
    GpuWorkgroupSize,
    GpuGlobalId,
    GpuBarrier,
    GpuShared,
    GpuBufferLoad,
    GpuBufferStore,
    GpuBufferLen,
    GpuLaunch,
    GpuLaunchAsync,
    ArcRetain,
    ArcRelease,
    ArcRetainWeak,
    ArcReleaseWeak,
    Panic,

    // Punctuation.
    LBrace,
    RBrace,
    LParen,
    RParen,
    LAngle,
    RAngle,
    LBracket,
    RBracket,
    Eq,
    Colon,
    Comma,
    Dot,
    At,
    Percent,
    Arrow,

    // Literals.
    IntLit,
    FloatLit,
    StringLit,
    True,
    False,

    // Identifiers.
    Ident,
    FnName,
    SsaName,
    TypeName,
    BlockLabel,
    DocComment,

    Eof,
}

pub fn lex(file_id: FileId, source: &str, diag: &mut DiagnosticBag) -> Vec<Token> {
    Lexer::new(file_id, source, diag).lex_all()
}

struct Lexer<'a> {
    file_id: FileId,
    source: &'a str,
    bytes: &'a [u8],
    pos: usize,
    diag: &'a mut DiagnosticBag,
}

impl<'a> Lexer<'a> {
    fn new(file_id: FileId, source: &'a str, diag: &'a mut DiagnosticBag) -> Self {
        Self {
            file_id,
            source,
            bytes: source.as_bytes(),
            pos: 0,
            diag,
        }
    }

    fn lex_all(mut self) -> Vec<Token> {
        let mut tokens = Vec::new();

        while !self.is_eof() {
            self.skip_whitespace();
            if self.is_eof() {
                break;
            }

            if self.peek_byte() == Some(b';') {
                if self.starts_with(";;;") {
                    tokens.push(self.lex_doc_comment());
                } else {
                    self.skip_line_comment();
                }
                continue;
            }

            let start = self.pos;
            let token = match self.peek_byte() {
                Some(b'{') => self.single_char_token(TokenKind::LBrace, 1),
                Some(b'}') => self.single_char_token(TokenKind::RBrace, 1),
                Some(b'(') => self.single_char_token(TokenKind::LParen, 1),
                Some(b')') => self.single_char_token(TokenKind::RParen, 1),
                Some(b'<') => self.single_char_token(TokenKind::LAngle, 1),
                Some(b'>') => self.single_char_token(TokenKind::RAngle, 1),
                Some(b'[') => self.single_char_token(TokenKind::LBracket, 1),
                Some(b']') => self.single_char_token(TokenKind::RBracket, 1),
                Some(b'=') => self.single_char_token(TokenKind::Eq, 1),
                Some(b':') => self.single_char_token(TokenKind::Colon, 1),
                Some(b',') => self.single_char_token(TokenKind::Comma, 1),
                Some(b'.') => self.single_char_token(TokenKind::Dot, 1),
                Some(b'-') if self.peek_next_byte() == Some(b'>') => {
                    self.single_char_token(TokenKind::Arrow, 2)
                }
                Some(b'@') => self.lex_sigiled_name(TokenKind::FnName, TokenKind::At),
                Some(b'%') => self.lex_sigiled_name(TokenKind::SsaName, TokenKind::Percent),
                Some(b'"') => self.lex_string(),
                Some(b'0'..=b'9') => self.lex_number(),
                Some(ch) if is_ident_start(ch) => self.lex_ident_or_keyword(),
                Some(_) => {
                    let ch = self.current_char();
                    let end = start + ch.len_utf8();
                    self.emit_mpp0001(start, end, format!("Unknown character `{}`.", ch));
                    self.pos = end;
                    continue;
                }
                None => break,
            };

            tokens.push(token);
        }

        let end = self.pos as u32;
        tokens.push(Token::new(
            TokenKind::Eof,
            Span::new(self.file_id, end, end),
            String::new(),
        ));
        tokens
    }

    fn lex_doc_comment(&mut self) -> Token {
        let start = self.pos;
        self.pos += 3; // ;;;
        let text_start = self.pos;
        while let Some(b) = self.peek_byte() {
            if b == b'\n' {
                break;
            }
            self.pos += 1;
        }
        let text = self.source[text_start..self.pos].trim_start().to_string();
        Token::new(TokenKind::DocComment, self.span(start, self.pos), text)
    }

    fn skip_line_comment(&mut self) {
        self.pos += 1; // ;
        while let Some(b) = self.peek_byte() {
            if b == b'\n' {
                break;
            }
            self.pos += 1;
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(b) = self.peek_byte() {
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn lex_sigiled_name(&mut self, kind: TokenKind, fallback: TokenKind) -> Token {
        let start = self.pos;
        self.pos += 1;
        if let Some(b) = self.peek_byte() {
            if is_ident_start(b) {
                self.pos += 1;
                while let Some(next) = self.peek_byte() {
                    if is_ident_continue(next) {
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
                let text = self.source[start..self.pos].to_string();
                return Token::new(kind, self.span(start, self.pos), text);
            }
        }
        Token::new(fallback, self.span(start, self.pos), String::new())
    }

    fn lex_number(&mut self) -> Token {
        let start = self.pos;

        if self.starts_with("0x") || self.starts_with("0X") {
            self.pos += 2;
            let hex_start = self.pos;
            while let Some(b) = self.peek_byte() {
                if b.is_ascii_hexdigit() {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            if self.pos == hex_start {
                self.emit_mpp0001(
                    start,
                    self.pos,
                    "Hex literal requires at least one digit.".into(),
                );
            }
            let text = self.source[start..self.pos].to_string();
            return Token::new(TokenKind::IntLit, self.span(start, self.pos), text);
        }

        while let Some(b) = self.peek_byte() {
            if b.is_ascii_digit() {
                self.pos += 1;
            } else {
                break;
            }
        }

        let mut kind = TokenKind::IntLit;
        if self.peek_byte() == Some(b'.')
            && self.peek_next_byte().is_some_and(|b| b.is_ascii_digit())
        {
            kind = TokenKind::FloatLit;
            self.pos += 1;
            while let Some(b) = self.peek_byte() {
                if b.is_ascii_digit() {
                    self.pos += 1;
                } else {
                    break;
                }
            }
        }

        if kind == TokenKind::FloatLit && (self.starts_with("f32") || self.starts_with("f64")) {
            self.pos += 3;
        }

        let text = self.source[start..self.pos].to_string();
        Token::new(kind, self.span(start, self.pos), text)
    }

    fn lex_string(&mut self) -> Token {
        let start = self.pos;
        self.pos += 1; // opening quote
        let mut out = String::new();

        while !self.is_eof() {
            let ch = self.current_char();
            self.pos += ch.len_utf8();

            match ch {
                '"' => {
                    return Token::new(TokenKind::StringLit, self.span(start, self.pos), out);
                }
                '\\' => {
                    if self.is_eof() {
                        self.emit_mpp0001(start, self.pos, "Unterminated escape sequence.".into());
                        break;
                    }
                    if let Some(escaped) = self.lex_escape_char() {
                        out.push(escaped);
                    }
                }
                _ => out.push(ch),
            }
        }

        self.emit_mpp0001(start, self.pos, "Unterminated string literal.".into());
        Token::new(TokenKind::StringLit, self.span(start, self.pos), out)
    }

    fn lex_escape_char(&mut self) -> Option<char> {
        let esc_start = self.pos.saturating_sub(1);
        let ch = self.current_char();
        self.pos += ch.len_utf8();

        match ch {
            'n' => Some('\n'),
            't' => Some('\t'),
            '\\' => Some('\\'),
            '"' => Some('"'),
            'u' => self.lex_unicode_escape(esc_start),
            _ => {
                self.emit_mpp0001(
                    esc_start,
                    self.pos,
                    format!("Unknown escape sequence `\\{}`.", ch),
                );
                None
            }
        }
    }

    fn lex_unicode_escape(&mut self, esc_start: usize) -> Option<char> {
        if self.peek_byte() != Some(b'{') {
            self.emit_mpp0001(esc_start, self.pos, "Expected `{` after `\\u`.".into());
            return None;
        }
        self.pos += 1; // {

        let digits_start = self.pos;
        while let Some(b) = self.peek_byte() {
            if b.is_ascii_hexdigit() {
                self.pos += 1;
            } else {
                break;
            }
        }
        let digits_end = self.pos;

        if self.peek_byte() != Some(b'}') {
            self.emit_mpp0001(
                esc_start,
                self.pos,
                "Expected `}` to close unicode escape.".into(),
            );
            return None;
        }
        self.pos += 1; // }

        if digits_start == digits_end {
            self.emit_mpp0001(
                esc_start,
                self.pos,
                "Unicode escape requires digits.".into(),
            );
            return None;
        }

        let digits = &self.source[digits_start..digits_end];
        match u32::from_str_radix(digits, 16)
            .ok()
            .and_then(char::from_u32)
        {
            Some(ch) => Some(ch),
            None => {
                self.emit_mpp0001(esc_start, self.pos, "Invalid unicode scalar value.".into());
                None
            }
        }
    }

    fn lex_ident_or_keyword(&mut self) -> Token {
        let start = self.pos;
        self.pos += 1;
        while let Some(b) = self.peek_byte() {
            if is_ident_continue(b) {
                self.pos += 1;
            } else {
                break;
            }
        }
        let plain_end = self.pos;
        let plain = &self.source[start..plain_end];

        if plain_end < self.bytes.len() && self.bytes[plain_end] == b'.' {
            let dotted_end = self.scan_dotted_end(plain_end);
            if dotted_end > plain_end {
                let dotted = &self.source[start..dotted_end];
                if let Some(kind) = op_keyword_kind(dotted) {
                    self.pos = dotted_end;
                    return Token::new(kind, self.span(start, self.pos), String::new());
                }
                if dotted.starts_with("const.") {
                    self.pos = dotted_end;
                    let suffix = &dotted["const.".len()..];
                    return Token::new(
                        TokenKind::ConstOp,
                        self.span(start, self.pos),
                        suffix.to_string(),
                    );
                }
            }
        }

        if let Some(kind) = keyword_kind(plain).or_else(|| op_keyword_kind(plain)) {
            return Token::new(kind, self.span(start, plain_end), keyword_text(kind, plain));
        }

        if plain.starts_with('T') && plain.len() > 1 && is_ident_start(plain.as_bytes()[1]) {
            return Token::new(
                TokenKind::TypeName,
                self.span(start, plain_end),
                plain.to_string(),
            );
        }

        if is_block_label(plain) {
            return Token::new(
                TokenKind::BlockLabel,
                self.span(start, plain_end),
                plain.to_string(),
            );
        }

        Token::new(
            TokenKind::Ident,
            self.span(start, plain_end),
            plain.to_string(),
        )
    }

    fn scan_dotted_end(&self, mut end: usize) -> usize {
        while end < self.bytes.len() && self.bytes[end] == b'.' {
            let seg_start = end + 1;
            if seg_start >= self.bytes.len() || !is_ident_start(self.bytes[seg_start]) {
                break;
            }
            end = seg_start + 1;
            while end < self.bytes.len() && is_ident_continue(self.bytes[end]) {
                end += 1;
            }
        }
        end
    }

    fn single_char_token(&mut self, kind: TokenKind, width: usize) -> Token {
        let start = self.pos;
        self.pos += width;
        Token::new(kind, self.span(start, self.pos), String::new())
    }

    fn span(&self, start: usize, end: usize) -> Span {
        Span::new(self.file_id, start as u32, end as u32)
    }

    fn emit_mpp0001(&mut self, start: usize, end: usize, message: String) {
        self.diag.emit(Diagnostic {
            code: "MPP0001".to_string(),
            severity: Severity::Error,
            title: "Lexer error".to_string(),
            primary_span: Some(self.span(start, end)),
            secondary_spans: Vec::new(),
            message,
            explanation_md: None,
            why: None,
            suggested_fixes: Vec::new(),
            rag_bundle: Vec::new(),
            related_docs: Vec::new(),
        });
    }

    fn current_char(&self) -> char {
        self.source[self.pos..].chars().next().unwrap_or('\0')
    }

    fn starts_with(&self, needle: &str) -> bool {
        self.source[self.pos..].starts_with(needle)
    }

    fn peek_byte(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn peek_next_byte(&self) -> Option<u8> {
        self.bytes.get(self.pos + 1).copied()
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.bytes.len()
    }
}

fn keyword_text(kind: TokenKind, text: &str) -> String {
    if matches!(
        kind,
        TokenKind::IntLit
            | TokenKind::FloatLit
            | TokenKind::StringLit
            | TokenKind::True
            | TokenKind::False
            | TokenKind::FnName
            | TokenKind::SsaName
            | TokenKind::TypeName
            | TokenKind::BlockLabel
            | TokenKind::Ident
            | TokenKind::DocComment
    ) {
        text.to_string()
    } else {
        String::new()
    }
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident_continue(b: u8) -> bool {
    is_ident_start(b) || b.is_ascii_digit()
}

fn is_block_label(text: &str) -> bool {
    let bytes = text.as_bytes();
    bytes.len() >= 3
        && bytes[0] == b'b'
        && bytes[1] == b'b'
        && bytes[2..].iter().all(|b| b.is_ascii_digit())
}

fn keyword_kind(text: &str) -> Option<TokenKind> {
    Some(match text {
        "module" => TokenKind::Module,
        "exports" => TokenKind::Exports,
        "imports" => TokenKind::Imports,
        "digest" => TokenKind::Digest,
        "fn" => TokenKind::Fn,
        "async" => TokenKind::Async,
        "meta" => TokenKind::Meta,
        "uses" => TokenKind::Uses,
        "effects" => TokenKind::Effects,
        "cost" => TokenKind::Cost,
        "heap" => TokenKind::Heap,
        "value" => TokenKind::Value,
        "struct" => TokenKind::Struct,
        "enum" => TokenKind::Enum,
        "extern" => TokenKind::Extern,
        "global" => TokenKind::Global,
        "unsafe" => TokenKind::Unsafe,
        "gpu" => TokenKind::Gpu,
        "target" => TokenKind::Target,
        "sig" => TokenKind::Sig,
        "impl" => TokenKind::Impl,
        "true" => TokenKind::True,
        "false" => TokenKind::False,
        _ => return None,
    })
}

fn op_keyword_kind(text: &str) -> Option<TokenKind> {
    Some(match text {
        "i.add" => TokenKind::IAdd,
        "i.sub" => TokenKind::ISub,
        "i.mul" => TokenKind::IMul,
        "i.sdiv" => TokenKind::ISdiv,
        "i.udiv" => TokenKind::IUdiv,
        "i.srem" => TokenKind::ISrem,
        "i.urem" => TokenKind::IUrem,
        "i.add.wrap" => TokenKind::IAddWrap,
        "i.sub.wrap" => TokenKind::ISubWrap,
        "i.mul.wrap" => TokenKind::IMulWrap,
        "i.add.checked" => TokenKind::IAddChecked,
        "i.sub.checked" => TokenKind::ISubChecked,
        "i.mul.checked" => TokenKind::IMulChecked,
        "i.and" => TokenKind::IAnd,
        "i.or" => TokenKind::IOr,
        "i.xor" => TokenKind::IXor,
        "i.shl" => TokenKind::IShl,
        "i.lshr" => TokenKind::ILshr,
        "i.ashr" => TokenKind::IAshr,
        "f.add" => TokenKind::FAdd,
        "f.sub" => TokenKind::FSub,
        "f.mul" => TokenKind::FMul,
        "f.div" => TokenKind::FDiv,
        "f.rem" => TokenKind::FRem,
        "f.add.fast" => TokenKind::FAddFast,
        "f.sub.fast" => TokenKind::FSubFast,
        "f.mul.fast" => TokenKind::FMulFast,
        "f.div.fast" => TokenKind::FDivFast,
        "icmp.eq" => TokenKind::IcmpEq,
        "icmp.ne" => TokenKind::IcmpNe,
        "icmp.slt" => TokenKind::IcmpSlt,
        "icmp.sgt" => TokenKind::IcmpSgt,
        "icmp.sle" => TokenKind::IcmpSle,
        "icmp.sge" => TokenKind::IcmpSge,
        "icmp.ult" => TokenKind::IcmpUlt,
        "icmp.ugt" => TokenKind::IcmpUgt,
        "icmp.ule" => TokenKind::IcmpUle,
        "icmp.uge" => TokenKind::IcmpUge,
        "fcmp.oeq" => TokenKind::FcmpOeq,
        "fcmp.one" => TokenKind::FcmpOne,
        "fcmp.olt" => TokenKind::FcmpOlt,
        "fcmp.ogt" => TokenKind::FcmpOgt,
        "fcmp.ole" => TokenKind::FcmpOle,
        "fcmp.oge" => TokenKind::FcmpOge,
        "call" => TokenKind::Call,
        "call_void" => TokenKind::CallVoid,
        "call.indirect" => TokenKind::CallIndirect,
        "call_void.indirect" => TokenKind::CallVoidIndirect,
        "try" => TokenKind::Try,
        "suspend.call" => TokenKind::SuspendCall,
        "suspend.await" => TokenKind::SuspendAwait,
        "new" => TokenKind::New,
        "getfield" => TokenKind::GetField,
        "setfield" => TokenKind::SetField,
        "phi" => TokenKind::Phi,
        "enum.new" => TokenKind::EnumNew,
        "enum.tag" => TokenKind::EnumTag,
        "enum.payload" => TokenKind::EnumPayload,
        "enum.is" => TokenKind::EnumIs,
        "share" => TokenKind::Share,
        "clone.shared" => TokenKind::CloneShared,
        "clone.weak" => TokenKind::CloneWeak,
        "weak.downgrade" => TokenKind::WeakDowngrade,
        "weak.upgrade" => TokenKind::WeakUpgrade,
        "cast" => TokenKind::Cast,
        "borrow.shared" => TokenKind::BorrowShared,
        "borrow.mut" => TokenKind::BorrowMut,
        "ptr.null" => TokenKind::PtrNull,
        "ptr.addr" => TokenKind::PtrAddr,
        "ptr.from_addr" => TokenKind::PtrFromAddr,
        "ptr.add" => TokenKind::PtrAdd,
        "ptr.load" => TokenKind::PtrLoad,
        "ptr.store" => TokenKind::PtrStore,
        "callable.capture" => TokenKind::CallableCapture,
        "arr.new" => TokenKind::ArrNew,
        "arr.len" => TokenKind::ArrLen,
        "arr.get" => TokenKind::ArrGet,
        "arr.set" => TokenKind::ArrSet,
        "arr.push" => TokenKind::ArrPush,
        "arr.pop" => TokenKind::ArrPop,
        "arr.slice" => TokenKind::ArrSlice,
        "arr.contains" => TokenKind::ArrContains,
        "arr.sort" => TokenKind::ArrSort,
        "arr.map" => TokenKind::ArrMap,
        "arr.filter" => TokenKind::ArrFilter,
        "arr.reduce" => TokenKind::ArrReduce,
        "arr.foreach" => TokenKind::ArrForeach,
        "map.new" => TokenKind::MapNew,
        "map.len" => TokenKind::MapLen,
        "map.get" => TokenKind::MapGet,
        "map.get_ref" => TokenKind::MapGetRef,
        "map.set" => TokenKind::MapSet,
        "map.delete" => TokenKind::MapDelete,
        "map.delete_void" => TokenKind::MapDeleteVoid,
        "map.contains_key" => TokenKind::MapContainsKey,
        "map.keys" => TokenKind::MapKeys,
        "map.values" => TokenKind::MapValues,
        "str.concat" => TokenKind::StrConcat,
        "str.len" => TokenKind::StrLen,
        "str.eq" => TokenKind::StrEq,
        "str.slice" => TokenKind::StrSlice,
        "str.bytes" => TokenKind::StrBytes,
        "str.parse_i64" => TokenKind::StrParseI64,
        "str.parse_u64" => TokenKind::StrParseU64,
        "str.parse_f64" => TokenKind::StrParseF64,
        "str.parse_bool" => TokenKind::StrParseBool,
        "str.builder.new" => TokenKind::StrBuilderNew,
        "str.builder.append_str" => TokenKind::StrBuilderAppendStr,
        "str.builder.append_i64" => TokenKind::StrBuilderAppendI64,
        "str.builder.append_i32" => TokenKind::StrBuilderAppendI32,
        "str.builder.append_f64" => TokenKind::StrBuilderAppendF64,
        "str.builder.append_bool" => TokenKind::StrBuilderAppendBool,
        "str.builder.build" => TokenKind::StrBuilderBuild,
        "json.encode" => TokenKind::JsonEncode,
        "json.decode" => TokenKind::JsonDecode,
        "gpu.thread_id" => TokenKind::GpuThreadId,
        "gpu.workgroup_id" => TokenKind::GpuWorkgroupId,
        "gpu.workgroup_size" => TokenKind::GpuWorkgroupSize,
        "gpu.global_id" => TokenKind::GpuGlobalId,
        "gpu.barrier" => TokenKind::GpuBarrier,
        "gpu.shared" => TokenKind::GpuShared,
        "gpu.buffer_load" => TokenKind::GpuBufferLoad,
        "gpu.buffer_store" => TokenKind::GpuBufferStore,
        "gpu.buffer_len" => TokenKind::GpuBufferLen,
        "gpu.launch" => TokenKind::GpuLaunch,
        "gpu.launch_async" => TokenKind::GpuLaunchAsync,
        "arc.retain" => TokenKind::ArcRetain,
        "arc.release" => TokenKind::ArcRelease,
        "arc.retain_weak" => TokenKind::ArcRetainWeak,
        "arc.release_weak" => TokenKind::ArcReleaseWeak,
        "panic" => TokenKind::Panic,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use magpie_diag::DiagnosticBag;

    #[test]
    fn lexes_simple_program_into_expected_tokens() {
        let source = "module demo\nfn @main(%x: TNum) -> TNum { bb0: true }";
        let mut diag = DiagnosticBag::new(16);

        let tokens = lex(FileId(1), source, &mut diag);
        let kinds: Vec<TokenKind> = tokens.iter().map(|t| t.kind).collect();

        assert_eq!(
            kinds,
            vec![
                TokenKind::Module,
                TokenKind::Ident,
                TokenKind::Fn,
                TokenKind::FnName,
                TokenKind::LParen,
                TokenKind::SsaName,
                TokenKind::Colon,
                TokenKind::TypeName,
                TokenKind::RParen,
                TokenKind::Arrow,
                TokenKind::TypeName,
                TokenKind::LBrace,
                TokenKind::BlockLabel,
                TokenKind::Colon,
                TokenKind::True,
                TokenKind::RBrace,
                TokenKind::Eof,
            ]
        );

        assert_eq!(tokens[1].text, "demo");
        assert_eq!(tokens[3].text, "@main");
        assert_eq!(tokens[5].text, "%x");
        assert_eq!(tokens[7].text, "TNum");
        assert_eq!(tokens[14].text, "true");
        assert_eq!(
            diag.error_count(),
            0,
            "valid source should lex without errors"
        );
    }

    #[test]
    fn lexes_doc_comments_and_skips_line_comments() {
        let source = ";;; top level doc\n; plain comment\nmodule demo";
        let mut diag = DiagnosticBag::new(8);

        let tokens = lex(FileId(2), source, &mut diag);

        assert_eq!(tokens[0].kind, TokenKind::DocComment);
        assert_eq!(tokens[0].text, "top level doc");
        assert_eq!(tokens[1].kind, TokenKind::Module);
        assert_eq!(tokens[2].kind, TokenKind::Ident);
        assert_eq!(tokens[2].text, "demo");
        assert_eq!(tokens[3].kind, TokenKind::Eof);
        assert_eq!(diag.error_count(), 0);
    }
}
