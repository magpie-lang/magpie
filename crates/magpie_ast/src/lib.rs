//! Magpie AST types and span infrastructure.

use serde::{Deserialize, Serialize};

// ── Span infrastructure (§36.2) ──

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub struct FileId(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub struct Span {
    pub file: FileId,
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub fn new(file: FileId, start: u32, end: u32) -> Self {
        Self { file, start, end }
    }

    pub fn dummy() -> Self {
        Self {
            file: FileId(0),
            start: 0,
            end: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    pub fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }
}

// ── Source map ──

#[derive(Debug, Default)]
pub struct SourceMap {
    files: Vec<SourceFile>,
}

#[derive(Debug)]
pub struct SourceFile {
    pub id: FileId,
    pub path: String,
    pub source: String,
    pub line_starts: Vec<u32>,
}

impl SourceMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_file(&mut self, path: String, source: String) -> FileId {
        let id = FileId(self.files.len() as u32);
        let line_starts = std::iter::once(0)
            .chain(source.match_indices('\n').map(|(i, _)| (i + 1) as u32))
            .collect();
        self.files.push(SourceFile {
            id,
            path,
            source,
            line_starts,
        });
        id
    }

    pub fn get_file(&self, id: FileId) -> Option<&SourceFile> {
        self.files.get(id.0 as usize)
    }

    pub fn get_source(&self, id: FileId) -> Option<&str> {
        self.get_file(id).map(|f| f.source.as_str())
    }

    pub fn lookup_line_col(&self, span: Span) -> Option<(usize, usize)> {
        let file = self.get_file(span.file)?;
        let line = file.line_starts.partition_point(|&s| s <= span.start);
        let col = span.start
            - file
                .line_starts
                .get(line.saturating_sub(1))
                .copied()
                .unwrap_or(0);
        Some((line, col as usize))
    }
}

// ── AST node types (§7.2) ──

#[derive(Clone, Debug)]
pub struct AstFile {
    pub header: Spanned<AstHeader>,
    pub decls: Vec<Spanned<AstDecl>>,
}

#[derive(Clone, Debug)]
pub struct AstHeader {
    pub module_path: Spanned<ModulePath>,
    pub exports: Vec<Spanned<ExportItem>>,
    pub imports: Vec<Spanned<ImportGroup>>,
    pub digest: Spanned<String>,
}

#[derive(Clone, Debug)]
pub struct ModulePath {
    pub segments: Vec<String>,
}

impl ModulePath {
    pub fn to_string(&self) -> String {
        self.segments.join(".")
    }
}

#[derive(Clone, Debug)]
pub enum ExportItem {
    Fn(String),   // @name
    Type(String), // TName
}

#[derive(Clone, Debug)]
pub struct ImportGroup {
    pub module_path: ModulePath,
    pub items: Vec<ImportItem>,
}

#[derive(Clone, Debug)]
pub enum ImportItem {
    Fn(String),
    Type(String),
}

#[derive(Clone, Debug)]
pub enum AstDecl {
    Fn(AstFnDecl),
    AsyncFn(AstFnDecl),
    UnsafeFn(AstFnDecl),
    GpuFn(AstGpuFnDecl),
    HeapStruct(AstStructDecl),
    ValueStruct(AstStructDecl),
    HeapEnum(AstEnumDecl),
    ValueEnum(AstEnumDecl),
    Extern(AstExternModule),
    Global(AstGlobalDecl),
    Impl(AstImplDecl),
    Sig(AstSigDecl),
}

#[derive(Clone, Debug)]
pub struct AstFnDecl {
    pub name: String,
    pub params: Vec<AstParam>,
    pub ret_ty: Spanned<AstType>,
    pub meta: Option<AstFnMeta>,
    pub blocks: Vec<Spanned<AstBlock>>,
    pub doc: Option<String>,
}

#[derive(Clone, Debug)]
pub struct AstGpuFnDecl {
    pub inner: AstFnDecl,
    pub target: String,
}

#[derive(Clone, Debug)]
pub struct AstParam {
    pub name: String,
    pub ty: Spanned<AstType>,
}

#[derive(Clone, Debug)]
pub struct AstFnMeta {
    pub uses: Vec<String>,
    pub effects: Vec<String>,
    pub cost: Vec<(String, i64)>,
}

#[derive(Clone, Debug)]
pub struct AstBlock {
    pub label: u32,
    pub instrs: Vec<Spanned<AstInstr>>,
    pub terminator: Spanned<AstTerminator>,
}

#[derive(Clone, Debug)]
pub enum AstInstr {
    Assign {
        name: String,
        ty: Spanned<AstType>,
        op: AstOp,
    },
    Void(AstOpVoid),
    UnsafeBlock(Vec<Spanned<AstInstr>>),
}

#[derive(Clone, Debug)]
pub enum AstTerminator {
    Ret(Option<AstValueRef>),
    Br(u32),
    Cbr {
        cond: AstValueRef,
        then_bb: u32,
        else_bb: u32,
    },
    Switch {
        val: AstValueRef,
        arms: Vec<(AstConstLit, u32)>,
        default: u32,
    },
    Unreachable,
}

#[derive(Clone, Debug)]
pub enum AstValueRef {
    Local(String),
    Const(AstConstExpr),
}

#[derive(Clone, Debug)]
pub struct AstConstExpr {
    pub ty: AstType,
    pub lit: AstConstLit,
}

#[derive(Clone, Debug)]
pub enum AstConstLit {
    Int(i128),
    Float(f64),
    Str(String),
    Bool(bool),
    Unit,
}

// Simplified Op/OpVoid — full variants will be populated during implementation
#[derive(Clone, Debug)]
pub enum AstOp {
    Const(AstConstExpr),
    BinOp {
        kind: BinOpKind,
        lhs: AstValueRef,
        rhs: AstValueRef,
    },
    Cmp {
        kind: CmpKind,
        pred: String,
        lhs: AstValueRef,
        rhs: AstValueRef,
    },
    Call {
        callee: String,
        targs: Vec<AstType>,
        args: Vec<(String, AstArgValue)>,
    },
    CallIndirect {
        callee: AstValueRef,
        args: Vec<(String, AstArgValue)>,
    },
    Try {
        callee: String,
        targs: Vec<AstType>,
        args: Vec<(String, AstArgValue)>,
    },
    SuspendCall {
        callee: String,
        targs: Vec<AstType>,
        args: Vec<(String, AstArgValue)>,
    },
    SuspendAwait {
        fut: AstValueRef,
    },
    New {
        ty: AstType,
        fields: Vec<(String, AstValueRef)>,
    },
    GetField {
        obj: AstValueRef,
        field: String,
    },
    Phi {
        ty: AstType,
        incomings: Vec<(u32, AstValueRef)>,
    },
    EnumNew {
        variant: String,
        args: Vec<(String, AstValueRef)>,
    },
    EnumTag {
        v: AstValueRef,
    },
    EnumPayload {
        variant: String,
        v: AstValueRef,
    },
    EnumIs {
        variant: String,
        v: AstValueRef,
    },
    Share {
        v: AstValueRef,
    },
    CloneShared {
        v: AstValueRef,
    },
    CloneWeak {
        v: AstValueRef,
    },
    WeakDowngrade {
        v: AstValueRef,
    },
    WeakUpgrade {
        v: AstValueRef,
    },
    Cast {
        from: AstType,
        to: AstType,
        v: AstValueRef,
    },
    BorrowShared {
        v: AstValueRef,
    },
    BorrowMut {
        v: AstValueRef,
    },
    PtrNull {
        ty: AstType,
    },
    PtrAddr {
        ty: AstType,
        p: AstValueRef,
    },
    PtrFromAddr {
        ty: AstType,
        addr: AstValueRef,
    },
    PtrAdd {
        ty: AstType,
        p: AstValueRef,
        count: AstValueRef,
    },
    PtrLoad {
        ty: AstType,
        p: AstValueRef,
    },
    CallableCapture {
        fn_ref: String,
        captures: Vec<(String, AstValueRef)>,
    },
    ArrNew {
        elem_ty: AstType,
        cap: AstValueRef,
    },
    ArrLen {
        arr: AstValueRef,
    },
    ArrGet {
        arr: AstValueRef,
        idx: AstValueRef,
    },
    ArrPop {
        arr: AstValueRef,
    },
    ArrSlice {
        arr: AstValueRef,
        start: AstValueRef,
        end: AstValueRef,
    },
    ArrContains {
        arr: AstValueRef,
        val: AstValueRef,
    },
    ArrMap {
        arr: AstValueRef,
        func: AstValueRef,
    },
    ArrFilter {
        arr: AstValueRef,
        func: AstValueRef,
    },
    ArrReduce {
        arr: AstValueRef,
        init: AstValueRef,
        func: AstValueRef,
    },
    MapNew {
        key_ty: AstType,
        val_ty: AstType,
    },
    MapLen {
        map: AstValueRef,
    },
    MapGet {
        map: AstValueRef,
        key: AstValueRef,
    },
    MapGetRef {
        map: AstValueRef,
        key: AstValueRef,
    },
    MapDelete {
        map: AstValueRef,
        key: AstValueRef,
    },
    MapContainsKey {
        map: AstValueRef,
        key: AstValueRef,
    },
    MapKeys {
        map: AstValueRef,
    },
    MapValues {
        map: AstValueRef,
    },
    StrConcat {
        a: AstValueRef,
        b: AstValueRef,
    },
    StrLen {
        s: AstValueRef,
    },
    StrEq {
        a: AstValueRef,
        b: AstValueRef,
    },
    StrSlice {
        s: AstValueRef,
        start: AstValueRef,
        end: AstValueRef,
    },
    StrBytes {
        s: AstValueRef,
    },
    StrBuilderNew,
    StrBuilderBuild {
        b: AstValueRef,
    },
    StrParseI64 {
        s: AstValueRef,
    },
    StrParseU64 {
        s: AstValueRef,
    },
    StrParseF64 {
        s: AstValueRef,
    },
    StrParseBool {
        s: AstValueRef,
    },
    JsonEncode {
        ty: AstType,
        v: AstValueRef,
    },
    JsonDecode {
        ty: AstType,
        s: AstValueRef,
    },
    GpuThreadId {
        dim: AstValueRef,
    },
    GpuWorkgroupId {
        dim: AstValueRef,
    },
    GpuWorkgroupSize {
        dim: AstValueRef,
    },
    GpuGlobalId {
        dim: AstValueRef,
    },
    GpuBufferLoad {
        ty: AstType,
        buf: AstValueRef,
        idx: AstValueRef,
    },
    GpuBufferLen {
        ty: AstType,
        buf: AstValueRef,
    },
    GpuShared {
        count: i64,
        ty: AstType,
    },
    GpuLaunch {
        device: AstValueRef,
        kernel: String,
        grid: AstArgValue,
        block: AstArgValue,
        args: AstArgValue,
    },
    GpuLaunchAsync {
        device: AstValueRef,
        kernel: String,
        grid: AstArgValue,
        block: AstArgValue,
        args: AstArgValue,
    },
}

#[derive(Clone, Debug)]
pub enum AstOpVoid {
    CallVoid {
        callee: String,
        targs: Vec<AstType>,
        args: Vec<(String, AstArgValue)>,
    },
    CallVoidIndirect {
        callee: AstValueRef,
        args: Vec<(String, AstArgValue)>,
    },
    SetField {
        obj: AstValueRef,
        field: String,
        val: AstValueRef,
    },
    Panic {
        msg: AstValueRef,
    },
    PtrStore {
        ty: AstType,
        p: AstValueRef,
        v: AstValueRef,
    },
    ArrSet {
        arr: AstValueRef,
        idx: AstValueRef,
        val: AstValueRef,
    },
    ArrPush {
        arr: AstValueRef,
        val: AstValueRef,
    },
    ArrSort {
        arr: AstValueRef,
    },
    ArrForeach {
        arr: AstValueRef,
        func: AstValueRef,
    },
    MapSet {
        map: AstValueRef,
        key: AstValueRef,
        val: AstValueRef,
    },
    MapDeleteVoid {
        map: AstValueRef,
        key: AstValueRef,
    },
    StrBuilderAppendStr {
        b: AstValueRef,
        s: AstValueRef,
    },
    StrBuilderAppendI64 {
        b: AstValueRef,
        v: AstValueRef,
    },
    StrBuilderAppendI32 {
        b: AstValueRef,
        v: AstValueRef,
    },
    StrBuilderAppendF64 {
        b: AstValueRef,
        v: AstValueRef,
    },
    StrBuilderAppendBool {
        b: AstValueRef,
        v: AstValueRef,
    },
    GpuBarrier,
    GpuBufferStore {
        ty: AstType,
        buf: AstValueRef,
        idx: AstValueRef,
        v: AstValueRef,
    },
}

#[derive(Clone, Debug)]
pub enum AstArgValue {
    Value(AstValueRef),
    List(Vec<AstArgListElem>),
    FnRef(String),
}

#[derive(Clone, Debug)]
pub enum AstArgListElem {
    Value(AstValueRef),
    FnRef(String),
}

#[derive(Clone, Debug)]
pub enum BinOpKind {
    IAdd,
    ISub,
    IMul,
    ISDiv,
    IUDiv,
    ISRem,
    IURem,
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
}

#[derive(Clone, Debug)]
pub enum CmpKind {
    ICmp,
    FCmp,
}

// ── Type AST (§7.2 Type grammar) ──

#[derive(Clone, Debug)]
pub struct AstType {
    pub ownership: Option<OwnershipMod>,
    pub base: AstBaseType,
}

#[derive(Clone, Debug)]
pub enum OwnershipMod {
    Shared,
    Borrow,
    MutBorrow,
    Weak,
}

#[derive(Clone, Debug)]
pub enum AstBaseType {
    Prim(String),
    Named {
        path: Option<ModulePath>,
        name: String,
        targs: Vec<AstType>,
    },
    Builtin(AstBuiltinType),
    Callable {
        sig_ref: String,
    },
    RawPtr(Box<AstType>),
}

#[derive(Clone, Debug)]
pub enum AstBuiltinType {
    Str,
    Array(Box<AstType>),
    Map(Box<AstType>, Box<AstType>),
    TOption(Box<AstType>),
    TResult(Box<AstType>, Box<AstType>),
    TStrBuilder,
    TMutex(Box<AstType>),
    TRwLock(Box<AstType>),
    TCell(Box<AstType>),
    TFuture(Box<AstType>),
    TChannelSend(Box<AstType>),
    TChannelRecv(Box<AstType>),
}

// ── Struct/Enum declarations ──

#[derive(Clone, Debug)]
pub struct AstStructDecl {
    pub name: String,
    pub type_params: Vec<AstTypeParam>,
    pub fields: Vec<AstFieldDecl>,
    pub doc: Option<String>,
}

#[derive(Clone, Debug)]
pub struct AstTypeParam {
    pub name: String,
    pub constraint: String,
}

#[derive(Clone, Debug)]
pub struct AstFieldDecl {
    pub name: String,
    pub ty: Spanned<AstType>,
}

#[derive(Clone, Debug)]
pub struct AstEnumDecl {
    pub name: String,
    pub type_params: Vec<AstTypeParam>,
    pub variants: Vec<AstVariantDecl>,
    pub doc: Option<String>,
}

#[derive(Clone, Debug)]
pub struct AstVariantDecl {
    pub name: String,
    pub fields: Vec<AstFieldDecl>,
}

#[derive(Clone, Debug)]
pub struct AstExternModule {
    pub abi: String,
    pub name: String,
    pub items: Vec<AstExternItem>,
    pub doc: Option<String>,
}

#[derive(Clone, Debug)]
pub struct AstExternItem {
    pub name: String,
    pub params: Vec<AstParam>,
    pub ret_ty: Spanned<AstType>,
    pub attrs: Vec<(String, String)>,
}

#[derive(Clone, Debug)]
pub struct AstGlobalDecl {
    pub name: String,
    pub ty: Spanned<AstType>,
    pub init: AstConstExpr,
    pub doc: Option<String>,
}

#[derive(Clone, Debug)]
pub struct AstImplDecl {
    pub trait_name: String,
    pub for_type: AstType,
    pub fn_ref: String,
}

#[derive(Clone, Debug)]
pub struct AstSigDecl {
    pub name: String,
    pub param_types: Vec<AstType>,
    pub ret_ty: AstType,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prim_i32() -> AstType {
        AstType {
            ownership: None,
            base: AstBaseType::Prim("i32".to_string()),
        }
    }

    #[test]
    fn constructs_ast_file_with_function_decl() {
        let span = Span::new(FileId(7), 10, 20);
        let module = ModulePath {
            segments: vec!["core".to_string(), "math".to_string()],
        };

        let fn_decl = AstFnDecl {
            name: "add".to_string(),
            params: vec![AstParam {
                name: "lhs".to_string(),
                ty: Spanned::new(prim_i32(), span),
            }],
            ret_ty: Spanned::new(prim_i32(), span),
            meta: None,
            blocks: vec![Spanned::new(
                AstBlock {
                    label: 0,
                    instrs: Vec::new(),
                    terminator: Spanned::new(AstTerminator::Ret(None), span),
                },
                span,
            )],
            doc: Some("Adds two integers".to_string()),
        };

        let ast = AstFile {
            header: Spanned::new(
                AstHeader {
                    module_path: Spanned::new(module, span),
                    exports: vec![Spanned::new(ExportItem::Fn("add".to_string()), span)],
                    imports: Vec::new(),
                    digest: Spanned::new(String::new(), span),
                },
                span,
            ),
            decls: vec![Spanned::new(AstDecl::Fn(fn_decl), span)],
        };

        assert_eq!(ast.header.node.module_path.node.to_string(), "core.math");
        assert_eq!(ast.decls.len(), 1);

        match &ast.decls[0].node {
            AstDecl::Fn(func) => {
                assert_eq!(func.name, "add");
                assert_eq!(func.params.len(), 1);
                assert_eq!(func.blocks.len(), 1);
                assert!(matches!(func.blocks[0].node.terminator.node, AstTerminator::Ret(None)));
            }
            other => panic!("expected fn declaration, got {other:?}"),
        }
    }

    #[test]
    fn source_map_lookup_line_col_uses_file_offsets() {
        let mut source_map = SourceMap::new();
        let file_id = source_map.add_file("demo.mp".to_string(), "alpha\nbeta\ngamma".to_string());
        let span = Span::new(file_id, 7, 8);

        let (line, col) = source_map
            .lookup_line_col(span)
            .expect("span should resolve to line/column");

        assert_eq!(line, 2);
        assert_eq!(col, 1);
        assert_eq!(
            source_map.get_source(file_id),
            Some("alpha\nbeta\ngamma"),
            "source text should round-trip via source map",
        );
    }
}
