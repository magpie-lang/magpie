//! magpie_rt — Magpie runtime ABI (§20 Runtime ABI)
//!
//! Implements ARC memory management, type registry, strings, StringBuilder,
//! and panic as specified in SPEC.md §20.1.

#![allow(clippy::missing_safety_doc)]

use std::alloc::{self, Layout};
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::ffi::c_char;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Mutex, Once, OnceLock, RwLock};
use std::time::Duration;

// ---------------------------------------------------------------------------
// §20.1.1  Object header
// ---------------------------------------------------------------------------

/// Object header placed before every heap-allocated Magpie value.
/// sizeof == 32, payload at byte offset 32.
#[repr(C)]
pub struct MpRtHeader {
    pub strong: AtomicU64, // offset  0
    pub weak: AtomicU64,   // offset  8
    pub type_id: u32,      // offset 16
    pub flags: u32,        // offset 20
    pub reserved0: u64,    // offset 24
}

// Compile-time layout assertions.
const _: () = assert!(std::mem::size_of::<MpRtHeader>() == 32);
const _: () = assert!(std::mem::align_of::<MpRtHeader>() >= 8);
const _: () = {
    // Check field offsets via repr(C) rules.
    let base = 0usize;
    // strong at 0
    assert!(std::mem::offset_of!(MpRtHeader, strong) == base);
    // weak at 8
    assert!(std::mem::offset_of!(MpRtHeader, weak) == 8);
    // type_id at 16
    assert!(std::mem::offset_of!(MpRtHeader, type_id) == 16);
    // flags at 20
    assert!(std::mem::offset_of!(MpRtHeader, flags) == 20);
    // reserved0 at 24
    assert!(std::mem::offset_of!(MpRtHeader, reserved0) == 24);
};

// ---------------------------------------------------------------------------
// §20.1.3  MpRtTypeInfo
// ---------------------------------------------------------------------------

pub const FLAG_HEAP: u32 = 0x1;
pub const FLAG_HAS_DROP: u32 = 0x2;
pub const FLAG_SEND: u32 = 0x4;
pub const FLAG_SYNC: u32 = 0x8;

/// Type descriptor registered by the compiler.
#[repr(C)]
pub struct MpRtTypeInfo {
    pub type_id: u32,
    pub flags: u32,
    pub payload_size: u64,
    pub payload_align: u64,
    pub drop_fn: Option<unsafe extern "C" fn(*mut MpRtHeader)>,
    pub debug_fqn: *const c_char,
}

// SAFETY: MpRtTypeInfo is only read after registration; the raw pointer fields
// are expected to be 'static (string literals / function pointers).
unsafe impl Send for MpRtTypeInfo {}
unsafe impl Sync for MpRtTypeInfo {}

pub type MpRtHashFn = Option<unsafe extern "C" fn(*const u8) -> u64>;
pub type MpRtEqFn = Option<unsafe extern "C" fn(*const u8, *const u8) -> i32>;
pub type MpRtCmpFn = Option<unsafe extern "C" fn(*const u8, *const u8) -> i32>;

// ---------------------------------------------------------------------------
// §20.1.4  Fixed type_ids
// ---------------------------------------------------------------------------

pub const TYPE_ID_STR: u32 = 20;
pub const TYPE_ID_STRBUILDER: u32 = 21;
pub const TYPE_ID_ARRAY: u32 = 22;
pub const TYPE_ID_MAP: u32 = 23;
pub const TYPE_ID_BOOL: u32 = 1;
pub const TYPE_ID_I8: u32 = 2;
pub const TYPE_ID_I16: u32 = 3;
pub const TYPE_ID_I32: u32 = 4;
pub const TYPE_ID_I64: u32 = 5;
pub const TYPE_ID_U8: u32 = 7;
pub const TYPE_ID_U16: u32 = 8;
pub const TYPE_ID_U32: u32 = 9;
pub const TYPE_ID_U64: u32 = 10;
pub const TYPE_ID_F32: u32 = 14;
pub const TYPE_ID_F64: u32 = 15;
pub const TYPE_ID_TCALLABLE: u32 = 26;

pub const MP_RT_OK: i32 = 0;
pub const MP_RT_ERR_INVALID_UTF8: i32 = 1;
pub const MP_RT_ERR_INVALID_FORMAT: i32 = 2;
pub const MP_RT_ERR_UNSUPPORTED_TYPE: i32 = 3;
pub const MP_RT_ERR_NULL_OUT_PTR: i32 = 4;
pub const MP_RT_ERR_NULL_INPUT: i32 = 5;

// ---------------------------------------------------------------------------
// Global type registry
// ---------------------------------------------------------------------------

struct TypeRegistry {
    entries: Vec<MpRtTypeInfo>,
}

impl TypeRegistry {
    const fn new() -> Self {
        TypeRegistry {
            entries: Vec::new(),
        }
    }

    fn find(&self, type_id: u32) -> Option<&MpRtTypeInfo> {
        self.entries.iter().find(|e| e.type_id == type_id)
    }

    fn find_mut(&mut self, type_id: u32) -> Option<&mut MpRtTypeInfo> {
        self.entries.iter_mut().find(|e| e.type_id == type_id)
    }
}

static TYPE_REGISTRY: OnceLock<Mutex<TypeRegistry>> = OnceLock::new();

fn registry() -> &'static Mutex<TypeRegistry> {
    TYPE_REGISTRY.get_or_init(|| Mutex::new(TypeRegistry::new()))
}

// ---------------------------------------------------------------------------
// §20.1.2  Core functions
// ---------------------------------------------------------------------------

/// Initialise the runtime. Idempotent; safe to call multiple times.
#[no_mangle]
pub extern "C" fn mp_rt_init() {
    let _ = registry();
}

/// Register an array of type descriptors.
///
/// # Safety
/// `infos` must point to `count` valid, initialised `MpRtTypeInfo` values.
#[no_mangle]
pub unsafe extern "C" fn mp_rt_register_types(infos: *const MpRtTypeInfo, count: u32) {
    if infos.is_null() || count == 0 {
        return;
    }
    let slice = std::slice::from_raw_parts(infos, count as usize);
    let mut reg = registry().lock().unwrap();
    for info in slice {
        // Overwrite existing entry for same type_id.
        if let Some(existing) = reg.find_mut(info.type_id) {
            existing.flags = info.flags;
            existing.payload_size = info.payload_size;
            existing.payload_align = info.payload_align;
            existing.drop_fn = info.drop_fn;
            existing.debug_fqn = info.debug_fqn;
        } else {
            reg.entries.push(MpRtTypeInfo {
                type_id: info.type_id,
                flags: info.flags,
                payload_size: info.payload_size,
                payload_align: info.payload_align,
                drop_fn: info.drop_fn,
                debug_fqn: info.debug_fqn,
            });
        }
    }
}

/// Return a pointer to the type info for `type_id`, or null if not registered.
#[no_mangle]
pub extern "C" fn mp_rt_type_info(type_id: u32) -> *const MpRtTypeInfo {
    let reg = registry().lock().unwrap();
    match reg.find(type_id) {
        Some(info) => info as *const MpRtTypeInfo,
        None => std::ptr::null(),
    }
}

// ---------------------------------------------------------------------------
// Allocation helpers
// ---------------------------------------------------------------------------

/// Compute the layout for a combined header+payload allocation.
///
/// Layout: [MpRtHeader (32 bytes)] [padding] [payload (payload_size bytes, payload_align)]
fn alloc_layout(payload_size: u64, payload_align: u64) -> Option<(Layout, usize)> {
    let header_layout = Layout::new::<MpRtHeader>();
    let payload_align = (payload_align as usize).max(1);
    let payload_size = payload_size as usize;

    // Build payload layout.
    let payload_layout = Layout::from_size_align(payload_size, payload_align).ok()?;

    // Extend header layout to accommodate payload alignment.
    let (combined, payload_offset) = header_layout.extend(payload_layout).ok()?;
    Some((combined, payload_offset))
}

/// Allocate a new object with strong=1, weak=1.
///
/// # Safety
/// `payload_align` must be a power of two.
#[no_mangle]
pub unsafe extern "C" fn mp_rt_alloc(
    type_id: u32,
    payload_size: u64,
    payload_align: u64,
    flags: u32,
) -> *mut MpRtHeader {
    let (layout, _payload_offset) =
        alloc_layout(payload_size, payload_align).expect("mp_rt_alloc: invalid layout");

    let ptr = alloc::alloc_zeroed(layout);
    if ptr.is_null() {
        alloc::handle_alloc_error(layout);
    }

    let header = ptr as *mut MpRtHeader;
    (*header).strong = AtomicU64::new(1);
    (*header).weak = AtomicU64::new(1);
    (*header).type_id = type_id;
    (*header).flags = flags;
    (*header).reserved0 = layout.size() as u64;

    header
}

// ---------------------------------------------------------------------------
// Retain / release
// ---------------------------------------------------------------------------

/// Increment the strong reference count (Relaxed).
///
/// # Safety
/// `obj` must be a live heap object.
#[no_mangle]
pub unsafe extern "C" fn mp_rt_retain_strong(obj: *mut MpRtHeader) {
    if obj.is_null() {
        return;
    }
    (*obj).strong.fetch_add(1, Ordering::Relaxed);
}

/// Decrement the strong reference count.
///
/// When strong hits 0: call drop_fn (if any), then release the implicit weak.
///
/// # Safety
/// `obj` must be a live heap object whose strong count >= 1.
#[no_mangle]
pub unsafe extern "C" fn mp_rt_release_strong(obj: *mut MpRtHeader) {
    if obj.is_null() {
        return;
    }
    let prev = (*obj).strong.fetch_sub(1, Ordering::Release);
    if prev == 1 {
        // Acquire fence so we observe all writes before the release.
        std::sync::atomic::fence(Ordering::Acquire);

        // Call the type-registered drop_fn if present.
        let type_id = (*obj).type_id;
        {
            let reg = registry().lock().unwrap();
            if let Some(info) = reg.find(type_id) {
                if let Some(drop_fn) = info.drop_fn {
                    drop_fn(obj);
                }
            }
        }

        // For builtin types (Str, StringBuilder) we have internal cleanup.
        builtin_drop(obj);

        // Release the implicit weak reference that was set during alloc.
        mp_rt_release_weak(obj);
    }
}

/// Perform builtin-type-specific cleanup on drop (before dealloc).
unsafe fn builtin_drop(obj: *mut MpRtHeader) {
    match (*obj).type_id {
        TYPE_ID_STRBUILDER => {
            // The payload holds a `*mut Vec<u8>` (box raw pointer).
            let payload = str_payload_base(obj);
            let vec_ptr = *(payload as *mut *mut Vec<u8>);
            if !vec_ptr.is_null() {
                drop(Box::from_raw(vec_ptr));
                // Zero the pointer so double-free is a null deref, not UB.
                *(payload as *mut *mut Vec<u8>) = std::ptr::null_mut();
            }
        }
        TYPE_ID_TCALLABLE => {
            let payload = str_payload_base(obj) as *mut MpRtCallablePayload;
            let vtable = (*payload).vtable_ptr;
            let data_ptr = (*payload).data_ptr;
            if !vtable.is_null() {
                if let Some(drop_fn) = (*vtable).drop_fn {
                    drop_fn(data_ptr);
                } else if !data_ptr.is_null() && (*vtable).size > 0 {
                    let layout = Layout::from_size_align((*vtable).size as usize, 8)
                        .expect("callable capture layout");
                    alloc::dealloc(data_ptr, layout);
                }
                drop(Box::from_raw(vtable as *mut MpRtCallableVtable));
            }
            (*payload).vtable_ptr = std::ptr::null();
            (*payload).data_ptr = std::ptr::null_mut();
        }
        _ => {}
    }
}

/// Increment the weak reference count (Relaxed).
///
/// # Safety
/// `obj` must be a live heap object.
#[no_mangle]
pub unsafe extern "C" fn mp_rt_retain_weak(obj: *mut MpRtHeader) {
    if obj.is_null() {
        return;
    }
    (*obj).weak.fetch_add(1, Ordering::Relaxed);
}

/// Decrement the weak reference count.  When weak hits 0 the memory is freed.
///
/// # Safety
/// `obj` must be a live heap object whose weak count >= 1.
#[no_mangle]
pub unsafe extern "C" fn mp_rt_release_weak(obj: *mut MpRtHeader) {
    if obj.is_null() {
        return;
    }
    let prev = (*obj).weak.fetch_sub(1, Ordering::Release);
    if prev == 1 {
        std::sync::atomic::fence(Ordering::Acquire);
        dealloc_object(obj);
    }
}

/// Free the raw memory backing `obj`.  Must only be called when weak == 0.
unsafe fn dealloc_object(obj: *mut MpRtHeader) {
    // We need the layout to call dealloc.  We stash payload_size / payload_align
    // in the header flags / reserved fields?  No — per spec those fields are fixed.
    // Instead we re-derive the layout from the type registry (or from the
    // payload_size stored in reserved0 for convenience).
    //
    // For robustness: store the combined allocation size in reserved0 at alloc time.
    // We set reserved0 to the combined allocation size during alloc.
    let alloc_size = (*obj).reserved0 as usize;
    if alloc_size == 0 {
        // Cannot free — this should not happen in correct usage.
        return;
    }
    // We also need alignment; use the type registry.
    let type_id = (*obj).type_id;
    let align = {
        let reg = registry().lock().unwrap();
        reg.find(type_id)
            .map(|i| (i.payload_align as usize).max(std::mem::align_of::<MpRtHeader>()))
            .unwrap_or(std::mem::align_of::<MpRtHeader>())
    };
    let layout = Layout::from_size_align(alloc_size, align).unwrap();
    alloc::dealloc(obj as *mut u8, layout);
}

/// Attempt to atomically upgrade a weak reference to a strong reference.
/// Returns null if the object has already been destroyed (strong == 0).
///
/// # Safety
/// `obj` must be a live heap object (weak count >= 1).
#[no_mangle]
pub unsafe extern "C" fn mp_rt_weak_upgrade(obj: *mut MpRtHeader) -> *mut MpRtHeader {
    let strong = &(*obj).strong;
    loop {
        let current = strong.load(Ordering::Relaxed);
        if current == 0 {
            return std::ptr::null_mut();
        }
        match strong.compare_exchange_weak(
            current,
            current + 1,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => return obj,
            Err(_) => continue,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal allocation helper that also stores layout info in reserved0.
// Used by all builtin allocators so that dealloc_object can reconstruct
// the layout without extra metadata.
// ---------------------------------------------------------------------------

/// Allocate a header + payload of exactly `payload_size` bytes with
/// `payload_align` alignment.  Stores the total allocation size in
/// `(*header).reserved0` for later use by `dealloc_object`.
///
/// The returned pointer is the header; payload starts at offset 32
/// (assuming payload_align <= 8, which is true for all builtin types).
unsafe fn alloc_builtin(
    type_id: u32,
    flags: u32,
    payload_size: usize,
    payload_align: usize,
) -> *mut MpRtHeader {
    let header_layout = Layout::new::<MpRtHeader>();
    let payload_layout = Layout::from_size_align(payload_size.max(1), payload_align).unwrap();
    let (combined, _payload_offset) = header_layout.extend(payload_layout).unwrap();
    let combined = combined.pad_to_align();

    let ptr = alloc::alloc_zeroed(combined);
    if ptr.is_null() {
        alloc::handle_alloc_error(combined);
    }

    let header = ptr as *mut MpRtHeader;
    (*header).strong = AtomicU64::new(1);
    (*header).weak = AtomicU64::new(1);
    (*header).type_id = type_id;
    (*header).flags = flags;
    (*header).reserved0 = combined.size() as u64;

    header
}

// ---------------------------------------------------------------------------
// String (type_id = 20)
//
// Payload layout: [len: u64 (8 bytes)] [bytes: u8 * len]
// Total payload = 8 + len bytes, alignment = 8.
// ---------------------------------------------------------------------------

/// Return a pointer to the base of the object's payload (header + 32).
/// Valid for all builtin types because payload_align <= 8.
#[inline]
unsafe fn str_payload_base(obj: *mut MpRtHeader) -> *mut u8 {
    (obj as *mut u8).add(std::mem::size_of::<MpRtHeader>())
}

/// Allocate a new Str object from a UTF-8 byte slice.
///
/// # Safety
/// `bytes` must point to `len` valid bytes.
#[no_mangle]
pub unsafe extern "C" fn mp_rt_str_from_utf8(bytes: *const u8, len: u64) -> *mut MpRtHeader {
    let payload_size = (std::mem::size_of::<u64>() as u64 + len) as usize;
    let obj = alloc_builtin(
        TYPE_ID_STR,
        FLAG_HEAP | FLAG_SEND | FLAG_SYNC,
        payload_size,
        8,
    );

    let base = str_payload_base(obj);
    // Write length.
    *(base as *mut u64) = len;
    // Write bytes.
    if len > 0 && !bytes.is_null() {
        std::ptr::copy_nonoverlapping(bytes, base.add(8), len as usize);
    }

    obj
}

/// Return a pointer to the string's bytes and write the length to `out_len`.
///
/// # Safety
/// `str_obj` must be a valid Str object.  `out_len` must be non-null.
#[no_mangle]
pub unsafe extern "C" fn mp_rt_str_bytes(str_obj: *mut MpRtHeader, out_len: *mut u64) -> *const u8 {
    let base = str_payload_base(str_obj);
    let len = *(base as *const u64);
    *out_len = len;
    base.add(8) as *const u8
}

/// Return the UTF-8 length of a Str object (byte count).
#[no_mangle]
pub unsafe extern "C" fn mp_rt_str_len(s: *mut MpRtHeader) -> u64 {
    let base = str_payload_base(s);
    *(base as *const u64)
}

/// Return 1 if the two Str objects contain equal bytes, 0 otherwise.
#[no_mangle]
pub unsafe extern "C" fn mp_rt_str_eq(a: *mut MpRtHeader, b: *mut MpRtHeader) -> i32 {
    let base_a = str_payload_base(a);
    let base_b = str_payload_base(b);
    let len_a = *(base_a as *const u64);
    let len_b = *(base_b as *const u64);
    if len_a != len_b {
        return 0;
    }
    let bytes_a = std::slice::from_raw_parts(base_a.add(8), len_a as usize);
    let bytes_b = std::slice::from_raw_parts(base_b.add(8), len_b as usize);
    if bytes_a == bytes_b {
        1
    } else {
        0
    }
}

/// Lexicographic compare for Str objects.
///
/// Returns negative / 0 / positive according to byte ordering.
#[no_mangle]
pub unsafe extern "C" fn mp_rt_str_cmp(a: *mut MpRtHeader, b: *mut MpRtHeader) -> i32 {
    let base_a = str_payload_base(a);
    let base_b = str_payload_base(b);
    let len_a = *(base_a as *const u64) as usize;
    let len_b = *(base_b as *const u64) as usize;
    let shared = std::cmp::min(len_a, len_b);
    let bytes_a = std::slice::from_raw_parts(base_a.add(8), shared);
    let bytes_b = std::slice::from_raw_parts(base_b.add(8), shared);
    for i in 0..shared {
        if bytes_a[i] < bytes_b[i] {
            return -1;
        }
        if bytes_a[i] > bytes_b[i] {
            return 1;
        }
    }
    if len_a < len_b {
        -1
    } else if len_a > len_b {
        1
    } else {
        0
    }
}

/// Concatenate two Str objects and return a new Str.
///
/// # Safety
/// Both `a` and `b` must be valid Str objects.
#[no_mangle]
pub unsafe extern "C" fn mp_rt_str_concat(
    a: *mut MpRtHeader,
    b: *mut MpRtHeader,
) -> *mut MpRtHeader {
    let base_a = str_payload_base(a);
    let base_b = str_payload_base(b);
    let len_a = *(base_a as *const u64);
    let len_b = *(base_b as *const u64);
    let new_len = len_a + len_b;

    let payload_size = (std::mem::size_of::<u64>() as u64 + new_len) as usize;
    let obj = alloc_builtin(
        TYPE_ID_STR,
        FLAG_HEAP | FLAG_SEND | FLAG_SYNC,
        payload_size,
        8,
    );

    let base = str_payload_base(obj);
    *(base as *mut u64) = new_len;
    if len_a > 0 {
        std::ptr::copy_nonoverlapping(base_a.add(8), base.add(8), len_a as usize);
    }
    if len_b > 0 {
        std::ptr::copy_nonoverlapping(base_b.add(8), base.add(8 + len_a as usize), len_b as usize);
    }

    obj
}

/// Return a new Str that is the byte slice `[start, end)` of `s`.
///
/// Panics if `start > end` or `end > len`.
///
/// # Safety
/// `s` must be a valid Str object.
#[no_mangle]
pub unsafe extern "C" fn mp_rt_str_slice(
    s: *mut MpRtHeader,
    start: u64,
    end: u64,
) -> *mut MpRtHeader {
    let base = str_payload_base(s);
    let len = *(base as *const u64);
    assert!(start <= end && end <= len, "mp_rt_str_slice: out of bounds");
    let slice_len = end - start;
    let src = base.add(8 + start as usize);
    mp_rt_str_from_utf8(src as *const u8, slice_len)
}

// ---------------------------------------------------------------------------
// StringBuilder (type_id = 21)
//
// Payload layout: one pointer-sized slot holding a `*mut Vec<u8>`.
// The Vec is heap-allocated via Box<Vec<u8>>.
// ---------------------------------------------------------------------------

#[inline]
unsafe fn strbuilder_vec(obj: *mut MpRtHeader) -> *mut Vec<u8> {
    let base = str_payload_base(obj);
    *(base as *const *mut Vec<u8>)
}

/// Create a new empty StringBuilder.
#[no_mangle]
pub unsafe extern "C" fn mp_rt_strbuilder_new() -> *mut MpRtHeader {
    // Payload = one pointer (8 bytes on 64-bit).
    let obj = alloc_builtin(
        TYPE_ID_STRBUILDER,
        FLAG_HEAP | FLAG_HAS_DROP,
        std::mem::size_of::<*mut Vec<u8>>(),
        std::mem::align_of::<*mut Vec<u8>>(),
    );
    let vec = Box::into_raw(Box::new(Vec::<u8>::new()));
    let base = str_payload_base(obj);
    *(base as *mut *mut Vec<u8>) = vec;
    obj
}

/// Append a Str to a StringBuilder.
///
/// # Safety
/// `b` must be a valid StringBuilder, `s` a valid Str.
#[no_mangle]
pub unsafe extern "C" fn mp_rt_strbuilder_append_str(b: *mut MpRtHeader, s: *mut MpRtHeader) {
    let vec = strbuilder_vec(b);
    let base = str_payload_base(s);
    let len = *(base as *const u64);
    let bytes = std::slice::from_raw_parts(base.add(8), len as usize);
    (*vec).extend_from_slice(bytes);
}

/// Append an i64 decimal representation to a StringBuilder.
#[no_mangle]
pub unsafe extern "C" fn mp_rt_strbuilder_append_i64(b: *mut MpRtHeader, v: i64) {
    let vec = strbuilder_vec(b);
    let s = v.to_string();
    (*vec).extend_from_slice(s.as_bytes());
}

/// Append an i32 decimal representation to a StringBuilder.
#[no_mangle]
pub unsafe extern "C" fn mp_rt_strbuilder_append_i32(b: *mut MpRtHeader, v: i32) {
    let vec = strbuilder_vec(b);
    let s = v.to_string();
    (*vec).extend_from_slice(s.as_bytes());
}

/// Append an f64 representation to a StringBuilder.
#[no_mangle]
pub unsafe extern "C" fn mp_rt_strbuilder_append_f64(b: *mut MpRtHeader, v: f64) {
    let vec = strbuilder_vec(b);
    let s = v.to_string();
    (*vec).extend_from_slice(s.as_bytes());
}

/// Append "true" or "false" to a StringBuilder.  `v` is 0 for false, non-zero for true.
#[no_mangle]
pub unsafe extern "C" fn mp_rt_strbuilder_append_bool(b: *mut MpRtHeader, v: i32) {
    let vec = strbuilder_vec(b);
    let s = if v != 0 { "true" } else { "false" };
    (*vec).extend_from_slice(s.as_bytes());
}

/// Consume the StringBuilder and return an owned Str.
///
/// After calling this function the StringBuilder's internal Vec pointer is
/// zeroed; the caller should release the StringBuilder header normally.
///
/// # Safety
/// `b` must be a valid StringBuilder.
#[no_mangle]
pub unsafe extern "C" fn mp_rt_strbuilder_build(b: *mut MpRtHeader) -> *mut MpRtHeader {
    let base = str_payload_base(b);
    let vec_ptr = *(base as *const *mut Vec<u8>);
    // Take ownership of the Vec.
    let vec = Box::from_raw(vec_ptr);
    // Zero the pointer so builtin_drop won't double-free.
    *(base as *mut *mut Vec<u8>) = std::ptr::null_mut();

    mp_rt_str_from_utf8(vec.as_ptr(), vec.len() as u64)
}

// ---------------------------------------------------------------------------
// §20.1.2  mp_rt_panic
// ---------------------------------------------------------------------------

/// Print the string message to stderr and abort the process.
///
/// # Safety
/// `str_msg` must be a valid Str object.
#[no_mangle]
pub unsafe extern "C" fn mp_rt_panic(str_msg: *mut MpRtHeader) -> ! {
    if str_msg.is_null() {
        eprintln!("magpie panic");
        std::process::abort();
    }
    let base = str_payload_base(str_msg);
    let len = *(base as *const u64);
    let bytes = std::slice::from_raw_parts(base.add(8), len as usize);
    let msg = std::str::from_utf8(bytes).unwrap_or("<invalid utf-8>");
    eprintln!("magpie panic: {}", msg);
    std::process::abort()
}

#[no_mangle]
pub unsafe extern "C" fn mp_std_println(s: *mut MpRtHeader) {
    if s.is_null() {
        println!();
    } else {
        println!("{}", str_to_rust_str(s));
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_std_eprintln(s: *const u8, len: usize) {
    let slice = std::slice::from_raw_parts(s, len);
    let text = std::str::from_utf8(slice).unwrap_or("<invalid utf8>");
    eprintln!("{}", text);
}

#[no_mangle]
pub extern "C" fn mp_std_readln() -> *mut u8 {
    use std::io::{self, BufRead};

    let mut line = String::new();
    io::stdin().lock().read_line(&mut line).unwrap_or(0);
    if line.ends_with('\n') {
        line.pop();
    }
    if line.ends_with('\r') {
        line.pop();
    }
    unsafe { mp_rt_str_from_utf8(line.as_ptr(), line.len() as u64) as *mut u8 }
}

#[no_mangle]
pub unsafe extern "C" fn mp_std_assert(cond: i32, msg: *mut MpRtHeader) {
    if cond == 0 {
        let text = if msg.is_null() {
            "assertion failed"
        } else {
            str_to_rust_str(msg)
        };
        eprintln!("assertion failed: {}", text);
        std::process::abort();
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_std_assert_eq(a: i64, b: i64, msg: *mut MpRtHeader) {
    if a != b {
        if msg.is_null() {
            eprintln!("assertion failed: {} != {}", a, b);
        } else {
            eprintln!(
                "assertion failed: {} != {} ({})",
                a,
                b,
                str_to_rust_str(msg)
            );
        }
        std::process::abort();
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_std_assert_ne(a: i64, b: i64, msg: *mut MpRtHeader) {
    if a == b {
        if msg.is_null() {
            eprintln!("assertion failed: {} == {}", a, b);
        } else {
            eprintln!(
                "assertion failed: {} == {} ({})",
                a,
                b,
                str_to_rust_str(msg)
            );
        }
        std::process::abort();
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_std_fail(msg: *mut MpRtHeader) {
    if msg.is_null() {
        eprintln!("test failed");
    } else {
        eprintln!("test failed: {}", str_to_rust_str(msg));
    }
    std::process::abort();
}

#[no_mangle]
pub extern "C" fn mp_std_exit(code: i32) {
    std::process::exit(code);
}

#[no_mangle]
pub extern "C" fn mp_std_cwd() -> *mut MpRtHeader {
    match std::env::current_dir() {
        Ok(path) => {
            let text = path.to_string_lossy();
            unsafe { mp_rt_str_from_utf8(text.as_ptr(), text.len() as u64) }
        }
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_std_env_var(name: *mut MpRtHeader) -> *mut MpRtHeader {
    if name.is_null() {
        return std::ptr::null_mut();
    }
    let key = str_to_rust_str(name);
    match std::env::var(key) {
        Ok(val) => mp_rt_str_from_utf8(val.as_ptr(), val.len() as u64),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn mp_std_args() -> *mut MpRtHeader {
    let args = std::env::args().collect::<Vec<_>>();
    let arr = unsafe {
        mp_rt_arr_new(
            TYPE_ID_STR,
            std::mem::size_of::<*mut MpRtHeader>() as u64,
            args.len() as u64,
        )
    };

    for arg in args {
        let arg_obj = unsafe { mp_rt_str_from_utf8(arg.as_ptr(), arg.len() as u64) };
        let arg_ptr = &arg_obj as *const *mut MpRtHeader as *const u8;
        unsafe {
            mp_rt_arr_push(arr, arg_ptr, std::mem::size_of::<*mut MpRtHeader>() as u64);
        }
    }

    arr
}

#[no_mangle]
pub unsafe extern "C" fn mp_std_println_bytes(s: *const u8, len: usize) {
    let slice = std::slice::from_raw_parts(s, len);
    let text = std::str::from_utf8(slice).unwrap_or("<invalid utf8>");
    println!("{}", text);
}

#[no_mangle]
pub unsafe extern "C" fn mp_std_assert_bytes(cond: i32, msg: *const u8, msg_len: usize) {
    if cond == 0 {
        let slice = std::slice::from_raw_parts(msg, msg_len);
        let text = std::str::from_utf8(slice).unwrap_or("assertion failed");
        eprintln!("assertion failed: {}", text);
        std::process::abort();
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_std_hash_str(s: *mut MpRtHeader) -> u64 {
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    str_to_rust_str(s).hash(&mut hasher);
    hasher.finish()
}

#[no_mangle]
pub unsafe extern "C" fn mp_std_hash_Str(s: *mut MpRtHeader) -> u64 {
    mp_std_hash_str(s)
}

#[no_mangle]
pub extern "C" fn mp_std_hash_i32(v: i32) -> u64 {
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    v.hash(&mut hasher);
    hasher.finish()
}

#[no_mangle]
pub extern "C" fn mp_std_hash_i64(v: i64) -> u64 {
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    v.hash(&mut hasher);
    hasher.finish()
}

#[no_mangle]
pub unsafe extern "C" fn mp_std_block_on(fut: *mut u8) {
    if fut.is_null() {
        return;
    }

    let mut spins: u32 = 0;
    while mp_rt_future_poll(fut) == 0 {
        std::thread::yield_now();
        spins = spins.saturating_add(1);
        if spins > 1_000_000 {
            eprintln!("magpie: mp_std_block_on timed out waiting for future readiness");
            return;
        }
    }

    mp_rt_future_take(fut, std::ptr::null_mut());
}

#[no_mangle]
pub unsafe extern "C" fn mp_std_spawn_task(fut: *mut u8) -> *mut u8 {
    fut
}

#[no_mangle]
pub extern "C" fn mp_std_abs_i32(v: i32) -> i32 {
    v.abs()
}

#[no_mangle]
pub extern "C" fn mp_std_abs_i64(v: i64) -> i64 {
    v.abs()
}

#[no_mangle]
pub extern "C" fn mp_std_sqrt_f64(v: f64) -> f64 {
    v.sqrt()
}

#[no_mangle]
pub extern "C" fn mp_std_min_i32(a: i32, b: i32) -> i32 {
    a.min(b)
}

#[no_mangle]
pub extern "C" fn mp_std_max_i32(a: i32, b: i32) -> i32 {
    a.max(b)
}

// ---------------------------------------------------------------------------
// Phase B runtime functions
// ---------------------------------------------------------------------------

#[repr(C)]
struct MpRtCallablePayload {
    vtable_ptr: *const MpRtCallableVtable,
    data_ptr: *mut u8,
}

#[repr(C)]
struct MpRtCallableVtable {
    call_fn: *mut u8,
    drop_fn: Option<unsafe extern "C" fn(*mut u8)>,
    size: u64,
}

type ArrForeachCallFn = unsafe extern "C" fn(*mut u8, *const u8);
type ArrMapCallFn = unsafe extern "C" fn(*mut u8, *const u8, *mut u8);
type ArrFilterCallFn = unsafe extern "C" fn(*mut u8, *const u8) -> i32;
type ArrReduceCallFn = unsafe extern "C" fn(*mut u8, *mut u8, *const u8);

#[no_mangle]
pub unsafe extern "C" fn mp_rt_callable_new(fn_ptr: *mut u8, captures_ptr: *mut u8) -> *mut u8 {
    let mut captures_size: usize = 0;
    let mut captures_data = std::ptr::null_mut::<u8>();
    if !captures_ptr.is_null() {
        captures_size = std::ptr::read_unaligned(captures_ptr as *const u64) as usize;
        if captures_size > 0 {
            let layout =
                Layout::from_size_align(captures_size, 8).expect("callable captures layout");
            captures_data = alloc::alloc_zeroed(layout);
            if captures_data.is_null() {
                alloc::handle_alloc_error(layout);
            }
            std::ptr::copy_nonoverlapping(
                (captures_ptr as *const u8).add(8),
                captures_data,
                captures_size,
            );
        }
    }

    let callable = alloc_builtin(
        TYPE_ID_TCALLABLE,
        FLAG_HEAP,
        std::mem::size_of::<MpRtCallablePayload>(),
        std::mem::align_of::<MpRtCallablePayload>(),
    );
    let payload = str_payload_base(callable) as *mut MpRtCallablePayload;
    let vtable = Box::into_raw(Box::new(MpRtCallableVtable {
        call_fn: fn_ptr,
        drop_fn: None,
        size: captures_size as u64,
    }));
    (*payload).vtable_ptr = vtable;
    (*payload).data_ptr = captures_data;
    callable as *mut u8
}

#[inline]
unsafe fn callable_payload(callable: *mut u8) -> *mut MpRtCallablePayload {
    assert!(!callable.is_null(), "callable is null");
    str_payload_base(callable as *mut MpRtHeader) as *mut MpRtCallablePayload
}

#[inline]
unsafe fn callable_parts(callable: *mut u8) -> (*mut u8, *mut u8) {
    let payload = callable_payload(callable);
    let vtable = (*payload).vtable_ptr;
    assert!(!vtable.is_null(), "callable vtable is null");
    let call_fn = (*vtable).call_fn;
    assert!(!call_fn.is_null(), "callable call_fn is null");
    (call_fn, (*payload).data_ptr)
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_callable_fn_ptr(callable: *mut u8) -> *mut u8 {
    if callable.is_null() {
        return std::ptr::null_mut();
    }
    let payload = callable_payload(callable);
    let vtable = (*payload).vtable_ptr;
    if vtable.is_null() {
        return std::ptr::null_mut();
    }
    (*vtable).call_fn
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_callable_capture_size(callable: *mut u8) -> u64 {
    if callable.is_null() {
        return 0;
    }
    let payload = callable_payload(callable);
    let vtable = (*payload).vtable_ptr;
    if vtable.is_null() {
        return 0;
    }
    (*vtable).size
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_callable_data_ptr(callable: *mut u8) -> *mut u8 {
    if callable.is_null() {
        return std::ptr::null_mut();
    }
    let payload = callable_payload(callable);
    (*payload).data_ptr
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_arr_foreach(arr: *mut MpRtHeader, callable: *mut u8) {
    let payload = arr_payload(arr);
    let (call_fn, data_ptr) = callable_parts(callable);
    let callback: ArrForeachCallFn = std::mem::transmute(call_fn);

    let len = usize_from_u64((*payload).len, "array len too large");
    let elem_size = usize_from_u64((*payload).elem_size, "array elem_size too large");
    for i in 0..len {
        let elem_ptr = if elem_size == 0 {
            std::ptr::NonNull::<u8>::dangling().as_ptr() as *const u8
        } else {
            (*payload)
                .data_ptr
                .add(mul_usize(i, elem_size, "array index overflow")) as *const u8
        };
        callback(data_ptr, elem_ptr);
    }
}

#[inline]
unsafe fn read_u64_unaligned(ptr: *const u8) -> u64 {
    std::ptr::read_unaligned(ptr as *const u64)
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_arr_map(
    arr: *mut MpRtHeader,
    callable: *mut u8,
    result_elem_type_id: u32,
    result_elem_size: u64,
) -> *mut MpRtHeader {
    let payload = arr_payload(arr);
    let out = mp_rt_arr_new(result_elem_type_id, result_elem_size, (*payload).len);
    if (*payload).len == 0 {
        return out;
    }

    let (call_fn, data_ptr) = callable_parts(callable);
    let callback: ArrMapCallFn = std::mem::transmute(call_fn);

    let len = usize_from_u64((*payload).len, "array len too large");
    let elem_size = usize_from_u64((*payload).elem_size, "array elem_size too large");
    let result_elem_size_usize =
        usize_from_u64(result_elem_size, "array result elem_size too large");
    let mut scratch = vec![0_u8; result_elem_size_usize.max(1)];

    for i in 0..len {
        let elem_ptr = if elem_size == 0 {
            std::ptr::NonNull::<u8>::dangling().as_ptr() as *const u8
        } else {
            (*payload)
                .data_ptr
                .add(mul_usize(i, elem_size, "array index overflow")) as *const u8
        };
        if result_elem_size_usize == 0 {
            callback(data_ptr, elem_ptr, std::ptr::null_mut());
            mp_rt_arr_push(out, std::ptr::null(), 0);
        } else {
            callback(data_ptr, elem_ptr, scratch.as_mut_ptr());
            mp_rt_arr_push(out, scratch.as_ptr(), result_elem_size);
        }
    }

    out
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_arr_filter(
    arr: *mut MpRtHeader,
    callable: *mut u8,
) -> *mut MpRtHeader {
    let payload = arr_payload(arr);
    let out = mp_rt_arr_new(
        (*payload).elem_type_id,
        (*payload).elem_size,
        (*payload).len,
    );
    if (*payload).len == 0 {
        return out;
    }

    let (call_fn, data_ptr) = callable_parts(callable);
    let callback: ArrFilterCallFn = std::mem::transmute(call_fn);

    let len = usize_from_u64((*payload).len, "array len too large");
    let elem_size = usize_from_u64((*payload).elem_size, "array elem_size too large");
    for i in 0..len {
        let elem_ptr = if elem_size == 0 {
            std::ptr::NonNull::<u8>::dangling().as_ptr() as *const u8
        } else {
            (*payload)
                .data_ptr
                .add(mul_usize(i, elem_size, "array index overflow")) as *const u8
        };
        if callback(data_ptr, elem_ptr) != 0 {
            mp_rt_arr_push(out, elem_ptr, (*payload).elem_size);
        }
    }

    out
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_arr_reduce(
    arr: *mut MpRtHeader,
    acc_inout: *mut u8,
    acc_size: u64,
    callable: *mut u8,
) {
    let _ = acc_size;

    let payload = arr_payload(arr);
    if (*payload).len == 0 {
        return;
    }

    let (call_fn, data_ptr) = callable_parts(callable);
    let callback: ArrReduceCallFn = std::mem::transmute(call_fn);

    let len = usize_from_u64((*payload).len, "array len too large");
    let elem_size = usize_from_u64((*payload).elem_size, "array elem_size too large");
    for i in 0..len {
        let elem_ptr = if elem_size == 0 {
            std::ptr::NonNull::<u8>::dangling().as_ptr() as *const u8
        } else {
            (*payload)
                .data_ptr
                .add(mul_usize(i, elem_size, "array index overflow")) as *const u8
        };
        callback(data_ptr, acc_inout, elem_ptr);
    }
}

struct MpRtMutexState {
    mutex: Mutex<()>,
    locked: std::sync::atomic::AtomicBool,
}

struct MpRtRwLockState {
    rwlock: RwLock<()>,
    state: std::sync::atomic::AtomicI32, // 0=unlocked, 1=read, 2=write
}

struct MpRtCellState {
    value: UnsafeCell<*mut u8>,
}

#[repr(C)]
struct MpRtFutureState {
    ready: bool,
    value: *mut u8,
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mutex_new() -> *mut u8 {
    Box::into_raw(Box::new(MpRtMutexState {
        mutex: Mutex::new(()),
        locked: std::sync::atomic::AtomicBool::new(false),
    })) as *mut u8
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mutex_lock(m: *mut u8) {
    assert!(!m.is_null(), "mp_rt_mutex_lock: null mutex");
    let state = m as *mut MpRtMutexState;

    // Surface poison state to callers while keeping the runtime running.
    drop(match (*state).mutex.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    });

    assert!(
        (*state)
            .locked
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok(),
        "mp_rt_mutex_lock: already locked"
    );
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mutex_unlock(m: *mut u8) {
    assert!(!m.is_null(), "mp_rt_mutex_unlock: null mutex");
    let state = m as *mut MpRtMutexState;
    assert!(
        (*state).locked.swap(false, Ordering::AcqRel),
        "mp_rt_mutex_unlock: mutex not locked"
    );
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_rwlock_new() -> *mut u8 {
    Box::into_raw(Box::new(MpRtRwLockState {
        rwlock: RwLock::new(()),
        state: std::sync::atomic::AtomicI32::new(0),
    })) as *mut u8
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_rwlock_read(rw: *mut u8) {
    assert!(!rw.is_null(), "mp_rt_rwlock_read: null rwlock");
    let state = rw as *mut MpRtRwLockState;

    drop(match (*state).rwlock.read() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    });

    assert!(
        (*state)
            .state
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_ok(),
        "mp_rt_rwlock_read: lock already held"
    );
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_rwlock_write(rw: *mut u8) {
    assert!(!rw.is_null(), "mp_rt_rwlock_write: null rwlock");
    let state = rw as *mut MpRtRwLockState;

    drop(match (*state).rwlock.write() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    });

    assert!(
        (*state)
            .state
            .compare_exchange(0, 2, Ordering::AcqRel, Ordering::Acquire)
            .is_ok(),
        "mp_rt_rwlock_write: lock already held"
    );
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_rwlock_unlock(rw: *mut u8) {
    assert!(!rw.is_null(), "mp_rt_rwlock_unlock: null rwlock");
    let state = rw as *mut MpRtRwLockState;
    assert_ne!(
        (*state).state.swap(0, Ordering::AcqRel),
        0,
        "mp_rt_rwlock_unlock: lock not held"
    );
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_cell_new(init: *mut u8) -> *mut u8 {
    Box::into_raw(Box::new(MpRtCellState {
        value: UnsafeCell::new(init),
    })) as *mut u8
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_cell_get(cell: *mut u8) -> *mut u8 {
    assert!(!cell.is_null(), "mp_rt_cell_get: null cell");
    *(*(cell as *mut MpRtCellState)).value.get()
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_cell_set(cell: *mut u8, val: *mut u8) {
    assert!(!cell.is_null(), "mp_rt_cell_set: null cell");
    *(*(cell as *mut MpRtCellState)).value.get() = val;
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_future_poll(future: *mut u8) -> i32 {
    assert!(!future.is_null(), "mp_rt_future_poll: null future");
    if (*(future as *mut MpRtFutureState)).ready {
        1
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_future_take(future: *mut u8, out_result: *mut u8) {
    assert!(!future.is_null(), "mp_rt_future_take: null future");
    let state = future as *mut MpRtFutureState;
    assert!((*state).ready, "mp_rt_future_take: future not ready");
    let value = (*state).value;
    (*state).value = std::ptr::null_mut();
    if !out_result.is_null() {
        std::ptr::write(out_result as *mut *mut u8, value);
    }
}

#[inline]
unsafe fn str_to_rust_str<'a>(s: *mut MpRtHeader) -> &'a str {
    let base = str_payload_base(s);
    let len = *(base as *const u64) as usize;
    let bytes = std::slice::from_raw_parts(base.add(8), len);
    std::str::from_utf8(bytes).expect("invalid utf-8 string")
}

#[inline]
unsafe fn str_to_rust_str_checked<'a>(s: *mut MpRtHeader) -> Result<&'a str, i32> {
    if s.is_null() {
        return Err(MP_RT_ERR_NULL_INPUT);
    }
    if (*s).type_id != TYPE_ID_STR {
        return Err(MP_RT_ERR_UNSUPPORTED_TYPE);
    }
    let base = str_payload_base(s);
    let len = *(base as *const u64) as usize;
    let bytes = std::slice::from_raw_parts(base.add(8), len);
    std::str::from_utf8(bytes).map_err(|_| MP_RT_ERR_INVALID_UTF8)
}

#[inline]
unsafe fn set_errmsg(out_errmsg: *mut *mut MpRtHeader, msg: &str) {
    let owned = mp_rt_str_from_utf8(msg.as_ptr(), msg.len() as u64);
    if out_errmsg.is_null() {
        mp_rt_release_strong(owned);
        return;
    }
    *out_errmsg = owned;
}

#[inline]
unsafe fn clear_slot<T>(slot: *mut *mut T) {
    if !slot.is_null() {
        *slot = std::ptr::null_mut();
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_str_parse_i64(s: *mut MpRtHeader) -> i64 {
    str_to_rust_str(s)
        .parse::<i64>()
        .expect("str.parse_i64 failed")
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_str_parse_u64(s: *mut MpRtHeader) -> u64 {
    str_to_rust_str(s)
        .parse::<u64>()
        .expect("str.parse_u64 failed")
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_str_parse_f64(s: *mut MpRtHeader) -> f64 {
    str_to_rust_str(s)
        .parse::<f64>()
        .expect("str.parse_f64 failed")
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_str_parse_bool(s: *mut MpRtHeader) -> i32 {
    if str_to_rust_str(s)
        .parse::<bool>()
        .expect("str.parse_bool failed")
    {
        1
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_str_try_parse_i64(
    s: *mut MpRtHeader,
    out: *mut i64,
    out_errmsg: *mut *mut MpRtHeader,
) -> i32 {
    clear_slot(out_errmsg);
    if out.is_null() {
        set_errmsg(out_errmsg, "out must not be null");
        return MP_RT_ERR_NULL_OUT_PTR;
    }
    let src = match str_to_rust_str_checked(s) {
        Ok(src) => src,
        Err(code) => {
            match code {
                MP_RT_ERR_NULL_INPUT => set_errmsg(out_errmsg, "input string must not be null"),
                MP_RT_ERR_UNSUPPORTED_TYPE => set_errmsg(out_errmsg, "input must be a Str handle"),
                MP_RT_ERR_INVALID_UTF8 => set_errmsg(out_errmsg, "input string is not valid UTF-8"),
                _ => set_errmsg(out_errmsg, "string parse failed"),
            }
            return code;
        }
    };
    match src.parse::<i64>() {
        Ok(value) => {
            *out = value;
            MP_RT_OK
        }
        Err(err) => {
            set_errmsg(out_errmsg, &format!("str.parse_i64 failed: {err}"));
            MP_RT_ERR_INVALID_FORMAT
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_str_try_parse_u64(
    s: *mut MpRtHeader,
    out: *mut u64,
    out_errmsg: *mut *mut MpRtHeader,
) -> i32 {
    clear_slot(out_errmsg);
    if out.is_null() {
        set_errmsg(out_errmsg, "out must not be null");
        return MP_RT_ERR_NULL_OUT_PTR;
    }
    let src = match str_to_rust_str_checked(s) {
        Ok(src) => src,
        Err(code) => {
            match code {
                MP_RT_ERR_NULL_INPUT => set_errmsg(out_errmsg, "input string must not be null"),
                MP_RT_ERR_UNSUPPORTED_TYPE => set_errmsg(out_errmsg, "input must be a Str handle"),
                MP_RT_ERR_INVALID_UTF8 => set_errmsg(out_errmsg, "input string is not valid UTF-8"),
                _ => set_errmsg(out_errmsg, "string parse failed"),
            }
            return code;
        }
    };
    match src.parse::<u64>() {
        Ok(value) => {
            *out = value;
            MP_RT_OK
        }
        Err(err) => {
            set_errmsg(out_errmsg, &format!("str.parse_u64 failed: {err}"));
            MP_RT_ERR_INVALID_FORMAT
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_str_try_parse_f64(
    s: *mut MpRtHeader,
    out: *mut f64,
    out_errmsg: *mut *mut MpRtHeader,
) -> i32 {
    clear_slot(out_errmsg);
    if out.is_null() {
        set_errmsg(out_errmsg, "out must not be null");
        return MP_RT_ERR_NULL_OUT_PTR;
    }
    let src = match str_to_rust_str_checked(s) {
        Ok(src) => src,
        Err(code) => {
            match code {
                MP_RT_ERR_NULL_INPUT => set_errmsg(out_errmsg, "input string must not be null"),
                MP_RT_ERR_UNSUPPORTED_TYPE => set_errmsg(out_errmsg, "input must be a Str handle"),
                MP_RT_ERR_INVALID_UTF8 => set_errmsg(out_errmsg, "input string is not valid UTF-8"),
                _ => set_errmsg(out_errmsg, "string parse failed"),
            }
            return code;
        }
    };
    match src.parse::<f64>() {
        Ok(value) => {
            *out = value;
            MP_RT_OK
        }
        Err(err) => {
            set_errmsg(out_errmsg, &format!("str.parse_f64 failed: {err}"));
            MP_RT_ERR_INVALID_FORMAT
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_str_try_parse_bool(
    s: *mut MpRtHeader,
    out: *mut i32,
    out_errmsg: *mut *mut MpRtHeader,
) -> i32 {
    clear_slot(out_errmsg);
    if out.is_null() {
        set_errmsg(out_errmsg, "out must not be null");
        return MP_RT_ERR_NULL_OUT_PTR;
    }
    let src = match str_to_rust_str_checked(s) {
        Ok(src) => src,
        Err(code) => {
            match code {
                MP_RT_ERR_NULL_INPUT => set_errmsg(out_errmsg, "input string must not be null"),
                MP_RT_ERR_UNSUPPORTED_TYPE => set_errmsg(out_errmsg, "input must be a Str handle"),
                MP_RT_ERR_INVALID_UTF8 => set_errmsg(out_errmsg, "input string is not valid UTF-8"),
                _ => set_errmsg(out_errmsg, "string parse failed"),
            }
            return code;
        }
    };
    match src.parse::<bool>() {
        Ok(value) => {
            *out = if value { 1 } else { 0 };
            MP_RT_OK
        }
        Err(err) => {
            set_errmsg(out_errmsg, &format!("str.parse_bool failed: {err}"));
            MP_RT_ERR_INVALID_FORMAT
        }
    }
}

fn json_escape_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out
}

fn json_unescape_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        match chars.next().expect("JSON: invalid escape sequence") {
            '"' => out.push('"'),
            '\\' => out.push('\\'),
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            't' => out.push('\t'),
            _ => panic!("JSON: unsupported escape"),
        }
    }
    out
}

unsafe fn json_encode_fallible(
    obj: *mut u8,
    type_id: u32,
) -> Result<*mut MpRtHeader, (i32, String)> {
    if obj.is_null() {
        return Err((
            MP_RT_ERR_NULL_INPUT,
            "input value must not be null".to_string(),
        ));
    }
    let json = match type_id {
        TYPE_ID_BOOL => {
            if *(obj as *const u8) != 0 {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        TYPE_ID_I8 => (*(obj as *const i8)).to_string(),
        TYPE_ID_I16 => (*(obj as *const i16)).to_string(),
        TYPE_ID_I32 => (*(obj as *const i32)).to_string(),
        TYPE_ID_I64 => (*(obj as *const i64)).to_string(),
        TYPE_ID_U8 => (*(obj as *const u8)).to_string(),
        TYPE_ID_U16 => (*(obj as *const u16)).to_string(),
        TYPE_ID_U32 => (*(obj as *const u32)).to_string(),
        TYPE_ID_U64 => (*(obj as *const u64)).to_string(),
        TYPE_ID_F32 => (*(obj as *const f32)).to_string(),
        TYPE_ID_F64 => (*(obj as *const f64)).to_string(),
        TYPE_ID_STR => {
            let s = match str_to_rust_str_checked(obj as *mut MpRtHeader) {
                Ok(s) => s,
                Err(MP_RT_ERR_INVALID_UTF8) => {
                    return Err((
                        MP_RT_ERR_INVALID_UTF8,
                        "input string is not valid UTF-8".to_string(),
                    ))
                }
                Err(_) => {
                    return Err((
                        MP_RT_ERR_UNSUPPORTED_TYPE,
                        "input must be a Str handle".to_string(),
                    ))
                }
            };
            format!("\"{}\"", json_escape_str(s))
        }
        _ => {
            return Err((
                MP_RT_ERR_UNSUPPORTED_TYPE,
                format!("JSON encode unsupported type_id: {type_id}"),
            ))
        }
    };
    Ok(mp_rt_str_from_utf8(json.as_ptr(), json.len() as u64))
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_json_encode(obj: *mut u8, type_id: u32) -> *mut MpRtHeader {
    match json_encode_fallible(obj, type_id) {
        Ok(encoded) => encoded,
        Err((_code, msg)) => panic!("{msg}"),
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_json_try_encode(
    obj: *mut u8,
    type_id: u32,
    out_str: *mut *mut MpRtHeader,
    out_errmsg: *mut *mut MpRtHeader,
) -> i32 {
    clear_slot(out_str);
    clear_slot(out_errmsg);
    if out_str.is_null() {
        set_errmsg(out_errmsg, "out_str must not be null");
        return MP_RT_ERR_NULL_OUT_PTR;
    }
    match json_encode_fallible(obj, type_id) {
        Ok(encoded) => {
            *out_str = encoded;
            MP_RT_OK
        }
        Err((code, msg)) => {
            set_errmsg(out_errmsg, &msg);
            code
        }
    }
}

unsafe fn json_decode_fallible(
    json_str: *mut MpRtHeader,
    type_id: u32,
) -> Result<*mut u8, (i32, String)> {
    let src = str_to_rust_str_checked(json_str).map_err(|code| match code {
        MP_RT_ERR_NULL_INPUT => (code, "json input must not be null".to_string()),
        MP_RT_ERR_UNSUPPORTED_TYPE => (code, "json input must be a Str handle".to_string()),
        MP_RT_ERR_INVALID_UTF8 => (code, "json input is not valid UTF-8".to_string()),
        _ => (code, "json decode failed".to_string()),
    })?;
    let src = src.trim();
    let value = match type_id {
        TYPE_ID_BOOL => Box::into_raw(Box::new(src.parse::<bool>().map_err(|err| {
            (
                MP_RT_ERR_INVALID_FORMAT,
                format!("JSON bool parse failed: {err}"),
            )
        })? as u8)),
        TYPE_ID_I8 => Box::into_raw(Box::new(src.parse::<i8>().map_err(|err| {
            (
                MP_RT_ERR_INVALID_FORMAT,
                format!("JSON i8 parse failed: {err}"),
            )
        })?)) as *mut u8,
        TYPE_ID_I16 => Box::into_raw(Box::new(src.parse::<i16>().map_err(|err| {
            (
                MP_RT_ERR_INVALID_FORMAT,
                format!("JSON i16 parse failed: {err}"),
            )
        })?)) as *mut u8,
        TYPE_ID_I32 => Box::into_raw(Box::new(src.parse::<i32>().map_err(|err| {
            (
                MP_RT_ERR_INVALID_FORMAT,
                format!("JSON i32 parse failed: {err}"),
            )
        })?)) as *mut u8,
        TYPE_ID_I64 => Box::into_raw(Box::new(src.parse::<i64>().map_err(|err| {
            (
                MP_RT_ERR_INVALID_FORMAT,
                format!("JSON i64 parse failed: {err}"),
            )
        })?)) as *mut u8,
        TYPE_ID_U8 => Box::into_raw(Box::new(src.parse::<u8>().map_err(|err| {
            (
                MP_RT_ERR_INVALID_FORMAT,
                format!("JSON u8 parse failed: {err}"),
            )
        })?)) as *mut u8,
        TYPE_ID_U16 => Box::into_raw(Box::new(src.parse::<u16>().map_err(|err| {
            (
                MP_RT_ERR_INVALID_FORMAT,
                format!("JSON u16 parse failed: {err}"),
            )
        })?)) as *mut u8,
        TYPE_ID_U32 => Box::into_raw(Box::new(src.parse::<u32>().map_err(|err| {
            (
                MP_RT_ERR_INVALID_FORMAT,
                format!("JSON u32 parse failed: {err}"),
            )
        })?)) as *mut u8,
        TYPE_ID_U64 => Box::into_raw(Box::new(src.parse::<u64>().map_err(|err| {
            (
                MP_RT_ERR_INVALID_FORMAT,
                format!("JSON u64 parse failed: {err}"),
            )
        })?)) as *mut u8,
        TYPE_ID_F32 => Box::into_raw(Box::new(src.parse::<f32>().map_err(|err| {
            (
                MP_RT_ERR_INVALID_FORMAT,
                format!("JSON f32 parse failed: {err}"),
            )
        })?)) as *mut u8,
        TYPE_ID_F64 => Box::into_raw(Box::new(src.parse::<f64>().map_err(|err| {
            (
                MP_RT_ERR_INVALID_FORMAT,
                format!("JSON f64 parse failed: {err}"),
            )
        })?)) as *mut u8,
        TYPE_ID_STR => {
            if src.len() < 2 || !src.starts_with('"') || !src.ends_with('"') {
                return Err((
                    MP_RT_ERR_INVALID_FORMAT,
                    "JSON string must be quoted".to_string(),
                ));
            }
            let unescaped = json_unescape_str(&src[1..src.len() - 1]);
            mp_rt_str_from_utf8(unescaped.as_ptr(), unescaped.len() as u64) as *mut u8
        }
        _ => {
            return Err((
                MP_RT_ERR_UNSUPPORTED_TYPE,
                format!("JSON decode unsupported type_id: {type_id}"),
            ))
        }
    };
    Ok(value)
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_json_decode(json_str: *mut MpRtHeader, type_id: u32) -> *mut u8 {
    match json_decode_fallible(json_str, type_id) {
        Ok(decoded) => decoded,
        Err((_code, msg)) => panic!("{msg}"),
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_json_try_decode(
    json_str: *mut MpRtHeader,
    type_id: u32,
    out_val: *mut *mut u8,
    out_errmsg: *mut *mut MpRtHeader,
) -> i32 {
    clear_slot(out_val);
    clear_slot(out_errmsg);
    if out_val.is_null() {
        set_errmsg(out_errmsg, "out_val must not be null");
        return MP_RT_ERR_NULL_OUT_PTR;
    }
    match json_decode_fallible(json_str, type_id) {
        Ok(decoded) => {
            *out_val = decoded;
            MP_RT_OK
        }
        Err((code, msg)) => {
            set_errmsg(out_errmsg, &msg);
            code
        }
    }
}

// ---------------------------------------------------------------------------
// Runtime support: channels / web bridge / GPU CPU-fallback bridge
// ---------------------------------------------------------------------------

struct MpRtChannelState {
    elem_size: u64,
    sender: Sender<Vec<u8>>,
    receiver: Mutex<Receiver<Vec<u8>>>,
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_channel_new(_elem_type_id: u32, elem_size: u64) -> *mut MpRtHeader {
    let (sender, receiver) = channel::<Vec<u8>>();
    Box::into_raw(Box::new(MpRtChannelState {
        elem_size,
        sender,
        receiver: Mutex::new(receiver),
    })) as *mut MpRtHeader
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_channel_send(
    sender: *mut MpRtHeader,
    val: *const u8,
    elem_size: u64,
) {
    if sender.is_null() || val.is_null() {
        return;
    }
    let state = sender as *mut MpRtChannelState;
    let copy_len = std::cmp::min((*state).elem_size, elem_size) as usize;
    let bytes = std::slice::from_raw_parts(val, copy_len).to_vec();
    let _ = (*state).sender.send(bytes);
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_channel_recv(
    receiver: *mut MpRtHeader,
    out: *mut u8,
    elem_size: u64,
) -> i32 {
    if receiver.is_null() || out.is_null() {
        return 0;
    }
    let state = receiver as *mut MpRtChannelState;
    let guard = match (*state).receiver.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    match guard.recv() {
        Ok(value) => {
            let copy_len = std::cmp::min(value.len(), elem_size as usize);
            std::ptr::copy_nonoverlapping(value.as_ptr(), out, copy_len);
            1
        }
        Err(_) => 0,
    }
}

unsafe fn str_obj_to_string(value: *mut MpRtHeader) -> Option<String> {
    if value.is_null() || (*value).type_id != TYPE_ID_STR {
        return None;
    }
    let mut len = 0_u64;
    let ptr = mp_rt_str_bytes(value, &mut len);
    if ptr.is_null() {
        return Some(String::new());
    }
    let bytes = std::slice::from_raw_parts(ptr, len as usize);
    Some(String::from_utf8_lossy(bytes).to_string())
}

fn parse_http_line(req: &str) -> Option<(&str, &str)> {
    let mut lines = req.lines();
    let first = lines.next()?.trim();
    let mut parts = first.split_whitespace();
    let method = parts.next()?;
    let path = parts.next()?;
    Some((method, path))
}

fn write_simple_http_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> Result<(), String> {
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(header.as_bytes())
        .map_err(|err| format!("failed to write response header: {err}"))?;
    stream
        .write_all(body)
        .map_err(|err| format!("failed to write response body: {err}"))?;
    stream
        .flush()
        .map_err(|err| format!("failed to flush response: {err}"))
}

fn handle_web_connection(
    stream: &mut TcpStream,
    max_body_bytes: usize,
    log_requests: bool,
) -> Result<(), String> {
    let mut request = vec![0_u8; max_body_bytes.clamp(1024, 64 * 1024)];
    let read = stream
        .read(&mut request)
        .map_err(|err| format!("failed to read request: {err}"))?;
    let text = std::str::from_utf8(&request[..read]).unwrap_or_default();
    let (method, path) = parse_http_line(text).unwrap_or(("INVALID", "/"));

    if log_requests {
        eprintln!("magpie_rt web: {method} {path}");
    }

    if method != "GET" {
        return write_simple_http_response(
            stream,
            "405 Method Not Allowed",
            "text/plain; charset=utf-8",
            b"method not allowed",
        );
    }

    let body = format!("{{\"ok\":true,\"path\":\"{}\"}}", path);
    write_simple_http_response(
        stream,
        "200 OK",
        "application/json; charset=utf-8",
        body.as_bytes(),
    )
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_web_serve(
    svc: *mut MpRtHeader,
    addr: *mut MpRtHeader,
    port: u16,
    keep_alive: u8,
    _threads: u32,
    max_body_bytes: u64,
    read_timeout_ms: u64,
    write_timeout_ms: u64,
    log_requests: u8,
    out_errmsg: *mut *mut MpRtHeader,
) -> i32 {
    gpu_clear_handle(out_errmsg);

    if svc.is_null() {
        gpu_set_error(out_errmsg, "invalid web service handle");
        return -1;
    }

    let host = str_obj_to_string(addr)
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "127.0.0.1".to_string());
    let bind_addr = format!("{host}:{port}");

    let listener = match TcpListener::bind(&bind_addr) {
        Ok(listener) => listener,
        Err(err) => {
            gpu_set_error(
                out_errmsg,
                &format!("failed to bind web runtime listener '{bind_addr}': {err}"),
            );
            return -1;
        }
    };

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(stream) => stream,
            Err(err) => {
                gpu_set_error(out_errmsg, &format!("failed to accept connection: {err}"));
                return -1;
            }
        };

        if read_timeout_ms > 0 {
            let _ = stream.set_read_timeout(Some(Duration::from_millis(read_timeout_ms)));
        }
        if write_timeout_ms > 0 {
            let _ = stream.set_write_timeout(Some(Duration::from_millis(write_timeout_ms)));
        }

        if let Err(err) = handle_web_connection(
            &mut stream,
            usize::try_from(max_body_bytes).unwrap_or(usize::MAX),
            log_requests != 0,
        ) {
            gpu_set_error(out_errmsg, &err);
            return -1;
        }

        if keep_alive == 0 {
            break;
        }
    }

    0
}

const TYPE_ID_GPU_DEVICE_RT: u32 = 9001;
const TYPE_ID_GPU_BUFFER_RT: u32 = 9002;
const TYPE_ID_GPU_FENCE_RT: u32 = 9003;
const TYPE_ID_GPU_KERNEL_RT: u32 = 9004;

#[repr(C)]
#[derive(Clone, Copy)]
struct MpRtGpuParam {
    kind: u8,
    _reserved0: u8,
    _reserved1: u16,
    type_id: u32,
    offset_or_binding: u32,
    size: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MpRtGpuKernelEntry {
    sid_hash: u64,
    backend: u32,
    blob: *const u8,
    blob_len: u64,
    num_params: u32,
    params: *const MpRtGpuParam,
    num_buffers: u32,
    push_const_size: u32,
}

#[repr(C)]
struct MpRtGpuDevicePayload {
    index: u32,
    _reserved: u32,
}

#[repr(C)]
struct MpRtGpuBufferPayload {
    elem_type_id: u32,
    _reserved: u32,
    elem_size: u64,
    len: u64,
    bytes: UnsafeCell<Vec<u8>>,
}

#[repr(C)]
struct MpRtGpuFencePayload {
    done: u8,
    _reserved: [u8; 7],
}

#[repr(C)]
struct MpRtGpuKernelPayload {
    sid_hash: u64,
    blob: UnsafeCell<Vec<u8>>,
}

#[derive(Clone, Copy)]
struct GpuKernelMeta {
    num_buffers: u32,
    push_const_size: u32,
}

static GPU_REGISTERED_KERNEL_COUNT: AtomicU64 = AtomicU64::new(0);
static GPU_TYPES_ONCE: Once = Once::new();
static GPU_KERNEL_REGISTRY: OnceLock<RwLock<HashMap<u64, GpuKernelMeta>>> = OnceLock::new();

fn gpu_kernel_registry() -> &'static RwLock<HashMap<u64, GpuKernelMeta>> {
    GPU_KERNEL_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

unsafe extern "C" fn mp_rt_gpu_buffer_drop(obj: *mut MpRtHeader) {
    let payload = gpu_buffer_payload(obj);
    std::ptr::drop_in_place((*payload).bytes.get());
}

unsafe extern "C" fn mp_rt_gpu_kernel_drop(obj: *mut MpRtHeader) {
    let payload = gpu_kernel_payload(obj);
    std::ptr::drop_in_place((*payload).blob.get());
}

fn ensure_gpu_types_registered() {
    GPU_TYPES_ONCE.call_once(|| unsafe {
        let infos = [
            MpRtTypeInfo {
                type_id: TYPE_ID_GPU_DEVICE_RT,
                flags: FLAG_HEAP | FLAG_SEND | FLAG_SYNC,
                payload_size: std::mem::size_of::<MpRtGpuDevicePayload>() as u64,
                payload_align: std::mem::align_of::<MpRtGpuDevicePayload>() as u64,
                drop_fn: None,
                debug_fqn: c"gpu.Device".as_ptr(),
            },
            MpRtTypeInfo {
                type_id: TYPE_ID_GPU_BUFFER_RT,
                flags: FLAG_HEAP | FLAG_HAS_DROP | FLAG_SEND | FLAG_SYNC,
                payload_size: std::mem::size_of::<MpRtGpuBufferPayload>() as u64,
                payload_align: std::mem::align_of::<MpRtGpuBufferPayload>() as u64,
                drop_fn: Some(mp_rt_gpu_buffer_drop),
                debug_fqn: c"gpu.Buffer".as_ptr(),
            },
            MpRtTypeInfo {
                type_id: TYPE_ID_GPU_FENCE_RT,
                flags: FLAG_HEAP | FLAG_SEND | FLAG_SYNC,
                payload_size: std::mem::size_of::<MpRtGpuFencePayload>() as u64,
                payload_align: std::mem::align_of::<MpRtGpuFencePayload>() as u64,
                drop_fn: None,
                debug_fqn: c"gpu.Fence".as_ptr(),
            },
            MpRtTypeInfo {
                type_id: TYPE_ID_GPU_KERNEL_RT,
                flags: FLAG_HEAP | FLAG_HAS_DROP | FLAG_SEND | FLAG_SYNC,
                payload_size: std::mem::size_of::<MpRtGpuKernelPayload>() as u64,
                payload_align: std::mem::align_of::<MpRtGpuKernelPayload>() as u64,
                drop_fn: Some(mp_rt_gpu_kernel_drop),
                debug_fqn: c"gpu.Kernel".as_ptr(),
            },
        ];
        mp_rt_register_types(infos.as_ptr(), infos.len() as u32);
    });
}

#[inline]
unsafe fn gpu_device_payload(dev: *mut MpRtHeader) -> *mut MpRtGpuDevicePayload {
    str_payload_base(dev) as *mut MpRtGpuDevicePayload
}

#[inline]
unsafe fn gpu_buffer_payload(buf: *mut MpRtHeader) -> *mut MpRtGpuBufferPayload {
    str_payload_base(buf) as *mut MpRtGpuBufferPayload
}

#[inline]
unsafe fn gpu_fence_payload(fence: *mut MpRtHeader) -> *mut MpRtGpuFencePayload {
    str_payload_base(fence) as *mut MpRtGpuFencePayload
}

#[inline]
unsafe fn gpu_kernel_payload(kernel: *mut MpRtHeader) -> *mut MpRtGpuKernelPayload {
    str_payload_base(kernel) as *mut MpRtGpuKernelPayload
}

#[inline]
unsafe fn gpu_is_device(dev: *mut MpRtHeader) -> bool {
    !dev.is_null() && (*dev).type_id == TYPE_ID_GPU_DEVICE_RT
}

#[inline]
unsafe fn gpu_is_buffer(buf: *mut MpRtHeader) -> bool {
    !buf.is_null() && (*buf).type_id == TYPE_ID_GPU_BUFFER_RT
}

#[inline]
unsafe fn gpu_is_fence(fence: *mut MpRtHeader) -> bool {
    !fence.is_null() && (*fence).type_id == TYPE_ID_GPU_FENCE_RT
}

#[inline]
unsafe fn gpu_is_kernel(kernel: *mut MpRtHeader) -> bool {
    !kernel.is_null() && (*kernel).type_id == TYPE_ID_GPU_KERNEL_RT
}

unsafe fn gpu_new_device(index: u32) -> *mut MpRtHeader {
    ensure_gpu_types_registered();
    let obj = alloc_builtin(
        TYPE_ID_GPU_DEVICE_RT,
        FLAG_HEAP | FLAG_SEND | FLAG_SYNC,
        std::mem::size_of::<MpRtGpuDevicePayload>(),
        std::mem::align_of::<MpRtGpuDevicePayload>(),
    );
    let payload = gpu_device_payload(obj);
    (*payload).index = index;
    (*payload)._reserved = 0;
    obj
}

unsafe fn gpu_new_buffer(
    elem_type_id: u32,
    elem_size: u64,
    len: u64,
) -> Result<*mut MpRtHeader, String> {
    ensure_gpu_types_registered();
    let total_bytes = len
        .checked_mul(elem_size)
        .ok_or_else(|| "gpu buffer byte length overflow".to_string())?;
    let total_bytes = usize::try_from(total_bytes)
        .map_err(|_| "gpu buffer byte length does not fit host usize".to_string())?;

    let obj = alloc_builtin(
        TYPE_ID_GPU_BUFFER_RT,
        FLAG_HEAP | FLAG_HAS_DROP | FLAG_SEND | FLAG_SYNC,
        std::mem::size_of::<MpRtGpuBufferPayload>(),
        std::mem::align_of::<MpRtGpuBufferPayload>(),
    );
    let payload = gpu_buffer_payload(obj);
    (*payload).elem_type_id = elem_type_id;
    (*payload)._reserved = 0;
    (*payload).elem_size = elem_size;
    (*payload).len = len;
    std::ptr::write((*payload).bytes.get(), vec![0_u8; total_bytes]);
    Ok(obj)
}

unsafe fn gpu_new_fence(done: u8) -> *mut MpRtHeader {
    ensure_gpu_types_registered();
    let obj = alloc_builtin(
        TYPE_ID_GPU_FENCE_RT,
        FLAG_HEAP | FLAG_SEND | FLAG_SYNC,
        std::mem::size_of::<MpRtGpuFencePayload>(),
        std::mem::align_of::<MpRtGpuFencePayload>(),
    );
    let payload = gpu_fence_payload(obj);
    (*payload).done = done;
    (*payload)._reserved = [0; 7];
    obj
}

fn gpu_hash_bytes(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x00000100000001b3;
    let mut hash = OFFSET;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

unsafe fn gpu_new_kernel(blob: &[u8]) -> *mut MpRtHeader {
    ensure_gpu_types_registered();
    let obj = alloc_builtin(
        TYPE_ID_GPU_KERNEL_RT,
        FLAG_HEAP | FLAG_HAS_DROP | FLAG_SEND | FLAG_SYNC,
        std::mem::size_of::<MpRtGpuKernelPayload>(),
        std::mem::align_of::<MpRtGpuKernelPayload>(),
    );
    let payload = gpu_kernel_payload(obj);
    (*payload).sid_hash = gpu_hash_bytes(blob);
    std::ptr::write((*payload).blob.get(), blob.to_vec());
    obj
}

#[no_mangle]
pub extern "C" fn mp_rt_gpu_device_count() -> u32 {
    1
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_gpu_register_kernels(entries: *const u8, count: i32) {
    let mut registry = gpu_kernel_registry().write().unwrap();
    registry.clear();

    if entries.is_null() || count <= 0 {
        GPU_REGISTERED_KERNEL_COUNT.store(0, Ordering::Relaxed);
        return;
    }

    let entries = std::slice::from_raw_parts(entries as *const MpRtGpuKernelEntry, count as usize);
    for entry in entries {
        registry.insert(
            entry.sid_hash,
            GpuKernelMeta {
                num_buffers: entry.num_buffers,
                push_const_size: entry.push_const_size,
            },
        );
    }
    GPU_REGISTERED_KERNEL_COUNT.store(registry.len() as u64, Ordering::Relaxed);
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_gpu_device_default(
    out_dev: *mut *mut MpRtHeader,
    out_errmsg: *mut *mut MpRtHeader,
) -> i32 {
    gpu_clear_handle(out_dev);
    gpu_clear_handle(out_errmsg);
    if out_dev.is_null() {
        gpu_set_error(out_errmsg, "out_dev must not be null");
        return -1;
    }
    *out_dev = gpu_new_device(0);
    0
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_gpu_device_by_index(
    idx: u32,
    out_dev: *mut *mut MpRtHeader,
    out_errmsg: *mut *mut MpRtHeader,
) -> i32 {
    gpu_clear_handle(out_dev);
    gpu_clear_handle(out_errmsg);
    if out_dev.is_null() {
        gpu_set_error(out_errmsg, "out_dev must not be null");
        return -1;
    }
    if idx != 0 {
        gpu_set_error(out_errmsg, &format!("gpu device index out of range: {idx}"));
        return -1;
    }
    *out_dev = gpu_new_device(idx);
    0
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_gpu_device_name(dev: *mut MpRtHeader) -> *mut MpRtHeader {
    let label = if gpu_is_device(dev) {
        "cpu-fallback-gpu:0"
    } else {
        "invalid-gpu-device"
    };
    mp_rt_str_from_utf8(label.as_ptr(), label.len() as u64)
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_gpu_buffer_new(
    dev: *mut MpRtHeader,
    elem_type_id: u32,
    elem_size: u64,
    len: u64,
    _usage_flags: u32,
    out_buf: *mut *mut MpRtHeader,
    out_errmsg: *mut *mut MpRtHeader,
) -> i32 {
    gpu_clear_handle(out_buf);
    gpu_clear_handle(out_errmsg);
    if out_buf.is_null() {
        gpu_set_error(out_errmsg, "out_buf must not be null");
        return -1;
    }
    if !gpu_is_device(dev) {
        gpu_set_error(out_errmsg, "invalid gpu device handle");
        return -1;
    }
    match gpu_new_buffer(elem_type_id, elem_size, len) {
        Ok(buf) => {
            *out_buf = buf;
            0
        }
        Err(err) => {
            gpu_set_error(out_errmsg, &err);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_gpu_buffer_from_array(
    dev: *mut MpRtHeader,
    host_arr: *mut MpRtHeader,
    _usage_flags: u32,
    out_buf: *mut *mut MpRtHeader,
    out_errmsg: *mut *mut MpRtHeader,
) -> i32 {
    gpu_clear_handle(out_buf);
    gpu_clear_handle(out_errmsg);
    if out_buf.is_null() {
        gpu_set_error(out_errmsg, "out_buf must not be null");
        return -1;
    }
    if !gpu_is_device(dev) {
        gpu_set_error(out_errmsg, "invalid gpu device handle");
        return -1;
    }
    if host_arr.is_null() || (*host_arr).type_id != TYPE_ID_ARRAY {
        gpu_set_error(out_errmsg, "host_arr must be a valid Array handle");
        return -1;
    }
    let arr = arr_payload(host_arr);
    let buf = match gpu_new_buffer((*arr).elem_type_id, (*arr).elem_size, (*arr).len) {
        Ok(buf) => buf,
        Err(err) => {
            gpu_set_error(out_errmsg, &err);
            return -1;
        }
    };
    let payload = gpu_buffer_payload(buf);
    let bytes_len = usize::try_from((*arr).len.saturating_mul((*arr).elem_size)).unwrap_or(0);
    if bytes_len > 0 && !(*arr).data_ptr.is_null() {
        let target = &mut *(*payload).bytes.get();
        std::ptr::copy_nonoverlapping((*arr).data_ptr, target.as_mut_ptr(), bytes_len);
    }
    *out_buf = buf;
    0
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_gpu_buffer_to_array(
    buf: *mut MpRtHeader,
    out_arr: *mut *mut MpRtHeader,
    out_errmsg: *mut *mut MpRtHeader,
) -> i32 {
    gpu_clear_handle(out_arr);
    gpu_clear_handle(out_errmsg);
    if out_arr.is_null() {
        gpu_set_error(out_errmsg, "out_arr must not be null");
        return -1;
    }
    if !gpu_is_buffer(buf) {
        gpu_set_error(out_errmsg, "invalid gpu buffer handle");
        return -1;
    }

    let payload = gpu_buffer_payload(buf);
    let arr = mp_rt_arr_new(
        (*payload).elem_type_id,
        (*payload).elem_size,
        (*payload).len,
    );
    let bytes = &*(*payload).bytes.get();
    let elem_size = usize::try_from((*payload).elem_size).unwrap_or(0);
    if elem_size == 0 {
        for _ in 0..(*payload).len {
            mp_rt_arr_push(arr, std::ptr::null(), 0);
        }
    } else {
        for idx in 0..(*payload).len {
            let idx = usize::try_from(idx).unwrap_or(0);
            let off = idx.saturating_mul(elem_size);
            mp_rt_arr_push(arr, bytes.as_ptr().add(off), (*payload).elem_size);
        }
    }
    *out_arr = arr;
    0
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_gpu_buffer_len(buf: *mut MpRtHeader) -> u64 {
    if !gpu_is_buffer(buf) {
        return 0;
    }
    (*gpu_buffer_payload(buf)).len
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_gpu_buffer_copy(
    src: *mut MpRtHeader,
    dst: *mut MpRtHeader,
    out_errmsg: *mut *mut MpRtHeader,
) -> i32 {
    gpu_clear_handle(out_errmsg);
    if !gpu_is_buffer(src) || !gpu_is_buffer(dst) {
        gpu_set_error(out_errmsg, "invalid gpu buffer handle");
        return -1;
    }
    let src_payload = gpu_buffer_payload(src);
    let dst_payload = gpu_buffer_payload(dst);
    if (*src_payload).elem_size != (*dst_payload).elem_size
        || (*src_payload).len != (*dst_payload).len
    {
        gpu_set_error(out_errmsg, "buffer shape mismatch");
        return -1;
    }
    let src_vec = &*(*src_payload).bytes.get();
    let dst_vec = &mut *(*dst_payload).bytes.get();
    dst_vec.clone_from_slice(src_vec);
    0
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_gpu_device_sync(
    dev: *mut MpRtHeader,
    out_errmsg: *mut *mut MpRtHeader,
) -> i32 {
    gpu_clear_handle(out_errmsg);
    if !gpu_is_device(dev) {
        gpu_set_error(out_errmsg, "invalid gpu device handle");
        return -1;
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_gpu_launch_sync(
    dev: *mut MpRtHeader,
    kernel_sid_hash: u64,
    _grid_x: u32,
    _grid_y: u32,
    _grid_z: u32,
    _block_x: u32,
    _block_y: u32,
    _block_z: u32,
    args_blob: *const u8,
    args_len: u64,
    out_errmsg: *mut *mut MpRtHeader,
) -> i32 {
    gpu_clear_handle(out_errmsg);
    if !gpu_is_device(dev) {
        gpu_set_error(out_errmsg, "invalid gpu device handle");
        return -1;
    }
    let Some(meta) = gpu_kernel_registry()
        .read()
        .unwrap()
        .get(&kernel_sid_hash)
        .copied()
    else {
        gpu_set_error(
            out_errmsg,
            &format!("kernel sid hash not registered: {kernel_sid_hash}"),
        );
        return -1;
    };

    let expected_len = u64::from(meta.num_buffers)
        .checked_mul(8)
        .and_then(|v| v.checked_add(u64::from(meta.push_const_size)))
        .unwrap_or(u64::MAX);
    if args_len != expected_len {
        gpu_set_error(
            out_errmsg,
            &format!("args_len mismatch: expected {expected_len}, got {args_len}"),
        );
        return -1;
    }
    if args_len > 0 && args_blob.is_null() {
        gpu_set_error(out_errmsg, "args_blob must not be null when args_len > 0");
        return -1;
    }

    if meta.num_buffers > 0 {
        for idx in 0..(meta.num_buffers as usize) {
            let off = idx
                .checked_mul(std::mem::size_of::<u64>())
                .expect("gpu launch arg offset overflow");
            let raw = read_u64_unaligned(args_blob.add(off));
            let ptr = raw as usize as *mut MpRtHeader;
            if !gpu_is_buffer(ptr) {
                gpu_set_error(out_errmsg, "launch args contain invalid gpu buffer handle");
                return -1;
            }
        }
    }

    0
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_gpu_launch_async(
    dev: *mut MpRtHeader,
    kernel_sid_hash: u64,
    _grid_x: u32,
    _grid_y: u32,
    _grid_z: u32,
    _block_x: u32,
    _block_y: u32,
    _block_z: u32,
    args_blob: *const u8,
    args_len: u64,
    out_fence: *mut *mut MpRtHeader,
    out_errmsg: *mut *mut MpRtHeader,
) -> i32 {
    gpu_clear_handle(out_fence);
    gpu_clear_handle(out_errmsg);
    if out_fence.is_null() {
        gpu_set_error(out_errmsg, "out_fence must not be null");
        return -1;
    }
    let sync = mp_rt_gpu_launch_sync(
        dev,
        kernel_sid_hash,
        _grid_x,
        _grid_y,
        _grid_z,
        _block_x,
        _block_y,
        _block_z,
        args_blob,
        args_len,
        out_errmsg,
    );
    if sync != 0 {
        return -1;
    }
    *out_fence = gpu_new_fence(1);
    0
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_gpu_fence_wait(
    fence: *mut MpRtHeader,
    _timeout_ms: u64,
    out_done: *mut u8,
    out_errmsg: *mut *mut MpRtHeader,
) -> i32 {
    gpu_clear_handle(out_errmsg);
    if !out_done.is_null() {
        *out_done = 0;
    }
    if !gpu_is_fence(fence) {
        gpu_set_error(out_errmsg, "invalid gpu fence handle");
        return -1;
    }
    if !out_done.is_null() {
        *out_done = (*gpu_fence_payload(fence)).done;
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_gpu_fence_free(fence: *mut MpRtHeader) {
    if !fence.is_null() {
        mp_rt_release_strong(fence);
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_gpu_device_open(idx: i32) -> *mut u8 {
    if idx < 0 {
        return std::ptr::null_mut();
    }
    let mut out_dev: *mut MpRtHeader = std::ptr::null_mut();
    let mut out_err: *mut MpRtHeader = std::ptr::null_mut();
    let _ = mp_rt_gpu_device_by_index(idx as u32, &mut out_dev, &mut out_err);
    if !out_err.is_null() {
        mp_rt_release_strong(out_err);
    }
    out_dev as *mut u8
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_gpu_device_close(dev: *mut u8) {
    if !dev.is_null() {
        mp_rt_release_strong(dev as *mut MpRtHeader);
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_gpu_buffer_alloc(dev: *mut u8, size: u64) -> *mut u8 {
    let mut out_buf: *mut MpRtHeader = std::ptr::null_mut();
    let mut out_err: *mut MpRtHeader = std::ptr::null_mut();
    let rc = mp_rt_gpu_buffer_new(
        dev as *mut MpRtHeader,
        TYPE_ID_U8,
        1,
        size,
        0,
        &mut out_buf,
        &mut out_err,
    );
    if !out_err.is_null() {
        mp_rt_release_strong(out_err);
    }
    if rc == 0 {
        out_buf as *mut u8
    } else {
        std::ptr::null_mut()
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_gpu_buffer_free(buf: *mut u8) {
    if !buf.is_null() {
        mp_rt_release_strong(buf as *mut MpRtHeader);
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_gpu_buffer_write(
    buf: *mut u8,
    offset: u64,
    data: *const u8,
    len: u64,
) -> i32 {
    if data.is_null() {
        return -1;
    }
    let buf = buf as *mut MpRtHeader;
    if !gpu_is_buffer(buf) {
        return -1;
    }
    let payload = gpu_buffer_payload(buf);
    let bytes = &mut *(*payload).bytes.get();
    let offset = match usize::try_from(offset) {
        Ok(offset) => offset,
        Err(_) => return -1,
    };
    let len = match usize::try_from(len) {
        Ok(len) => len,
        Err(_) => return -1,
    };
    if offset > bytes.len() || len > bytes.len().saturating_sub(offset) {
        return -1;
    }
    std::ptr::copy_nonoverlapping(data, bytes.as_mut_ptr().add(offset), len);
    0
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_gpu_buffer_read(
    buf: *mut u8,
    offset: u64,
    out: *mut u8,
    len: u64,
) -> i32 {
    if out.is_null() {
        return -1;
    }
    let buf = buf as *mut MpRtHeader;
    if !gpu_is_buffer(buf) {
        return -1;
    }
    let payload = gpu_buffer_payload(buf);
    let bytes = &*(*payload).bytes.get();
    let offset = match usize::try_from(offset) {
        Ok(offset) => offset,
        Err(_) => return -1,
    };
    let len = match usize::try_from(len) {
        Ok(len) => len,
        Err(_) => return -1,
    };
    if offset > bytes.len() || len > bytes.len().saturating_sub(offset) {
        return -1;
    }
    std::ptr::copy_nonoverlapping(bytes.as_ptr().add(offset), out, len);
    0
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_gpu_kernel_load(
    dev: *mut u8,
    spv: *const u8,
    spv_len: u64,
) -> *mut u8 {
    if !gpu_is_device(dev as *mut MpRtHeader) || spv.is_null() {
        return std::ptr::null_mut();
    }
    let len = match usize::try_from(spv_len) {
        Ok(len) => len,
        Err(_) => return std::ptr::null_mut(),
    };
    let blob = std::slice::from_raw_parts(spv, len);
    gpu_new_kernel(blob) as *mut u8
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_gpu_kernel_free(kernel: *mut u8) {
    if !kernel.is_null() {
        mp_rt_release_strong(kernel as *mut MpRtHeader);
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_gpu_launch(
    kernel: *mut u8,
    _groups_x: u32,
    _groups_y: u32,
    _groups_z: u32,
    _args: *const *mut u8,
    _arg_count: u32,
) -> i32 {
    if !gpu_is_kernel(kernel as *mut MpRtHeader) {
        return -1;
    }
    0
}

unsafe fn gpu_clear_handle<T>(slot: *mut *mut T) {
    if !slot.is_null() {
        *slot = std::ptr::null_mut();
    }
}

unsafe fn gpu_set_error(out_errmsg: *mut *mut MpRtHeader, msg: &str) {
    let owned = mp_rt_str_from_utf8(msg.as_ptr(), msg.len() as u64);
    if out_errmsg.is_null() {
        mp_rt_release_strong(owned);
        return;
    }
    *out_errmsg = owned;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create a Str from a &str literal.
    unsafe fn make_str(s: &str) -> *mut MpRtHeader {
        mp_rt_str_from_utf8(s.as_ptr(), s.len() as u64)
    }

    // Helper: read a Str back as a Rust String.
    unsafe fn read_str(obj: *mut MpRtHeader) -> String {
        let mut len: u64 = 0;
        let ptr = mp_rt_str_bytes(obj, &mut len);
        let bytes = std::slice::from_raw_parts(ptr, len as usize);
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    unsafe fn make_test_callable(call_fn: *mut u8) -> *mut u8 {
        let callable = alloc_builtin(
            TYPE_ID_TCALLABLE,
            FLAG_HEAP,
            std::mem::size_of::<MpRtCallablePayload>(),
            std::mem::align_of::<MpRtCallablePayload>(),
        );
        let payload = str_payload_base(callable) as *mut MpRtCallablePayload;
        let vtable = Box::into_raw(Box::new(MpRtCallableVtable {
            call_fn,
            drop_fn: None,
            size: 0,
        }));
        (*payload).vtable_ptr = vtable;
        (*payload).data_ptr = std::ptr::null_mut();
        callable as *mut u8
    }

    unsafe extern "C" fn arr_map_double(_ctx: *mut u8, elem: *const u8, out: *mut u8) {
        let v = *(elem as *const i32);
        *(out as *mut i32) = v * 2;
    }

    unsafe extern "C" fn arr_filter_even(_ctx: *mut u8, elem: *const u8) -> i32 {
        if *(elem as *const i32) % 2 == 0 {
            1
        } else {
            0
        }
    }

    unsafe extern "C" fn arr_reduce_sum(_ctx: *mut u8, acc: *mut u8, elem: *const u8) {
        let acc = acc as *mut i32;
        *acc += *(elem as *const i32);
    }

    fn gpu_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static GPU_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        GPU_TEST_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    // -----------------------------------------------------------------------
    // ARC: alloc + retain + release cycle
    // -----------------------------------------------------------------------

    #[test]
    fn test_std_println_does_not_crash() {
        unsafe {
            let msg = make_str("Hello, world!");
            mp_std_println(msg);
            mp_rt_release_strong(msg);
        }
    }

    #[test]
    fn test_callable_introspection_reports_fn_and_capture_data() {
        unsafe {
            let fn_ptr = arr_filter_even as *const () as *mut u8;
            let mut capture_blob = Vec::new();
            capture_blob.extend_from_slice(&(4_u64).to_le_bytes());
            capture_blob.extend_from_slice(&[1_u8, 2, 3, 4]);

            let callable = mp_rt_callable_new(fn_ptr, capture_blob.as_ptr() as *mut u8);
            assert_eq!(mp_rt_callable_fn_ptr(callable), fn_ptr);
            assert_eq!(mp_rt_callable_capture_size(callable), 4);

            let data_ptr = mp_rt_callable_data_ptr(callable);
            assert!(!data_ptr.is_null());
            let captured = std::slice::from_raw_parts(data_ptr as *const u8, 4);
            assert_eq!(captured, &[1_u8, 2, 3, 4]);

            mp_rt_release_strong(callable as *mut MpRtHeader);

            let nocap = mp_rt_callable_new(fn_ptr, std::ptr::null_mut());
            assert_eq!(mp_rt_callable_capture_size(nocap), 0);
            assert!(mp_rt_callable_data_ptr(nocap).is_null());
            mp_rt_release_strong(nocap as *mut MpRtHeader);
        }
    }

    #[test]
    fn test_callable_new_accepts_unaligned_capture_blob() {
        unsafe {
            let fn_ptr = arr_filter_even as *const () as *mut u8;
            let mut raw = vec![0_u8; 8 + 4 + 1];
            let blob = raw.as_mut_ptr().add(1);
            std::ptr::write_unaligned(blob as *mut u64, 4_u64);
            std::ptr::copy_nonoverlapping([9_u8, 8, 7, 6].as_ptr(), blob.add(8), 4);

            let callable = mp_rt_callable_new(fn_ptr, blob);
            assert_eq!(mp_rt_callable_fn_ptr(callable), fn_ptr);
            assert_eq!(mp_rt_callable_capture_size(callable), 4);
            let data = mp_rt_callable_data_ptr(callable);
            assert!(!data.is_null());
            assert_eq!(
                std::slice::from_raw_parts(data as *const u8, 4),
                &[9, 8, 7, 6]
            );
            mp_rt_release_strong(callable as *mut MpRtHeader);
        }
    }

    #[test]
    fn test_std_assert_success_paths() {
        unsafe {
            let msg = make_str("ok");
            mp_std_assert(1, msg);
            mp_std_assert_eq(7, 7, msg);
            mp_std_assert_ne(7, 8, msg);
            mp_rt_release_strong(msg);
        }
    }

    #[test]
    fn test_std_os_functions() {
        unsafe {
            let cwd = mp_std_cwd();
            assert!(!cwd.is_null());
            let cwd_text = read_str(cwd);
            assert!(!cwd_text.is_empty());
            mp_rt_release_strong(cwd);

            let key = if std::env::var("PATH").is_ok() {
                "PATH"
            } else if std::env::var("HOME").is_ok() {
                "HOME"
            } else {
                // Environment can be highly constrained in sandboxed runners.
                // Validate missing-key behavior in that case.
                let missing = make_str("MAGPIE_RT_TEST_MISSING_ENV");
                let none = mp_std_env_var(missing);
                assert!(none.is_null());
                mp_rt_release_strong(missing);
                return;
            };
            let expected = std::env::var(key).expect("selected env key should exist");
            let key_obj = make_str(key);
            let env_val = mp_std_env_var(key_obj);
            assert!(!env_val.is_null());
            assert_eq!(read_str(env_val), expected);
            mp_rt_release_strong(env_val);
            mp_rt_release_strong(key_obj);
        }
    }

    #[test]
    fn test_std_hash_functions() {
        use std::hash::{Hash, Hasher};

        unsafe {
            let s = make_str("hash-me");
            let actual = mp_std_hash_str(s);
            let actual_pascal = mp_std_hash_Str(s);
            let mut expected_hasher = std::collections::hash_map::DefaultHasher::new();
            "hash-me".hash(&mut expected_hasher);
            let expected = expected_hasher.finish();
            assert_eq!(actual, expected);
            assert_eq!(actual_pascal, expected);
            mp_rt_release_strong(s);
        }

        let mut i32_hasher = std::collections::hash_map::DefaultHasher::new();
        123_i32.hash(&mut i32_hasher);
        assert_eq!(mp_std_hash_i32(123), i32_hasher.finish());

        let mut i64_hasher = std::collections::hash_map::DefaultHasher::new();
        456_i64.hash(&mut i64_hasher);
        assert_eq!(mp_std_hash_i64(456), i64_hasher.finish());
    }

    #[test]
    fn test_std_async_shims() {
        unsafe {
            mp_std_block_on(std::ptr::null_mut());

            let state = Box::into_raw(Box::new(MpRtFutureState {
                ready: true,
                value: std::ptr::null_mut(),
            }));
            let future = state as *mut u8;
            assert_eq!(mp_std_spawn_task(future), future);
            mp_std_block_on(future);
            drop(Box::from_raw(state));
        }
    }

    #[test]
    fn test_std_math_functions() {
        assert_eq!(mp_std_abs_i32(-9), 9);
        assert_eq!(mp_std_abs_i64(-11), 11);
        assert_eq!(mp_std_min_i32(2, -5), -5);
        assert_eq!(mp_std_max_i32(2, -5), 2);
        assert_eq!(mp_std_sqrt_f64(81.0), 9.0);
    }

    #[test]
    fn test_gpu_runtime_cpu_fallback_roundtrip() {
        let _guard = gpu_test_lock();
        unsafe {
            assert_eq!(mp_rt_gpu_device_count(), 1_u32);
            mp_rt_gpu_register_kernels(std::ptr::null(), 0);
            assert_eq!(GPU_REGISTERED_KERNEL_COUNT.load(Ordering::Relaxed), 0);

            let params: [MpRtGpuParam; 1] = [MpRtGpuParam {
                kind: 1,
                _reserved0: 0,
                _reserved1: 0,
                type_id: 0,
                offset_or_binding: 0,
                size: 0,
            }];
            let entries = [MpRtGpuKernelEntry {
                sid_hash: 42,
                backend: 1,
                blob: std::ptr::null(),
                blob_len: 0,
                num_params: 1,
                params: params.as_ptr(),
                num_buffers: 1,
                push_const_size: 16,
            }];
            mp_rt_gpu_register_kernels(entries.as_ptr().cast::<u8>(), entries.len() as i32);
            assert_eq!(GPU_REGISTERED_KERNEL_COUNT.load(Ordering::Relaxed), 1);

            let compat_dev = mp_rt_gpu_device_open(0);
            assert!(!compat_dev.is_null());
            mp_rt_gpu_device_close(compat_dev);

            let mut dev: *mut MpRtHeader = std::ptr::null_mut();
            let mut gpu_err: *mut MpRtHeader = std::ptr::null_mut();
            assert_eq!(mp_rt_gpu_device_default(&mut dev, &mut gpu_err), 0);
            assert!(!dev.is_null());
            assert!(gpu_err.is_null());

            let mut bad_dev: *mut MpRtHeader = std::ptr::null_mut();
            gpu_err = std::ptr::null_mut();
            assert_eq!(mp_rt_gpu_device_by_index(1, &mut bad_dev, &mut gpu_err), -1);
            assert!(bad_dev.is_null());
            assert!(!gpu_err.is_null());
            assert!(read_str(gpu_err).contains("out of range"));
            mp_rt_release_strong(gpu_err);

            let mut dev2: *mut MpRtHeader = std::ptr::null_mut();
            gpu_err = std::ptr::null_mut();
            assert_eq!(mp_rt_gpu_device_by_index(0, &mut dev2, &mut gpu_err), 0);
            assert!(!dev2.is_null());
            assert!(gpu_err.is_null());

            let dev_name = mp_rt_gpu_device_name(std::ptr::null_mut());
            assert!(!dev_name.is_null());
            assert_eq!(read_str(dev_name), "invalid-gpu-device");
            mp_rt_release_strong(dev_name);

            let dev_name_ok = mp_rt_gpu_device_name(dev);
            assert_eq!(read_str(dev_name_ok), "cpu-fallback-gpu:0");
            mp_rt_release_strong(dev_name_ok);

            let compat_buf = mp_rt_gpu_buffer_alloc(dev as *mut u8, 8);
            assert!(!compat_buf.is_null());
            let write_word: u64 = 0xAABBCCDDEEFF0011;
            assert_eq!(
                mp_rt_gpu_buffer_write(
                    compat_buf,
                    0,
                    (&write_word as *const u64).cast::<u8>(),
                    std::mem::size_of::<u64>() as u64,
                ),
                0
            );
            let mut read_word: u64 = 0;
            assert_eq!(
                mp_rt_gpu_buffer_read(
                    compat_buf,
                    0,
                    (&mut read_word as *mut u64).cast::<u8>(),
                    std::mem::size_of::<u64>() as u64,
                ),
                0
            );
            assert_eq!(read_word, write_word);
            mp_rt_gpu_buffer_free(compat_buf);

            assert_eq!(
                mp_rt_gpu_buffer_write(std::ptr::null_mut(), 0, std::ptr::null(), 0),
                -1
            );
            assert_eq!(
                mp_rt_gpu_buffer_read(std::ptr::null_mut(), 0, std::ptr::null_mut(), 0),
                -1
            );

            let kernel_blob = [0x03_u8, 0x02, 0x23, 0x07];
            let kernel = mp_rt_gpu_kernel_load(
                dev as *mut u8,
                kernel_blob.as_ptr(),
                kernel_blob.len() as u64,
            );
            assert!(!kernel.is_null());
            let args: [*mut u8; 0] = [];
            assert_eq!(mp_rt_gpu_launch(kernel, 1, 1, 1, args.as_ptr(), 0), 0);
            mp_rt_gpu_kernel_free(kernel);

            let mut out_buf: *mut MpRtHeader = std::ptr::null_mut();
            let mut out_err: *mut MpRtHeader = std::ptr::null_mut();
            assert_eq!(
                mp_rt_gpu_buffer_new(
                    dev,
                    TYPE_ID_I32,
                    std::mem::size_of::<i32>() as u64,
                    4,
                    0,
                    &mut out_buf,
                    &mut out_err,
                ),
                0
            );
            assert!(!out_buf.is_null());
            assert!(out_err.is_null());
            assert_eq!(mp_rt_gpu_buffer_len(out_buf), 4);

            let host_arr = mp_rt_arr_new(TYPE_ID_I32, std::mem::size_of::<i32>() as u64, 0);
            for value in [10_i32, 20, 30, 40] {
                mp_rt_arr_push(
                    host_arr,
                    (&value as *const i32).cast::<u8>(),
                    std::mem::size_of::<i32>() as u64,
                );
            }

            let mut from_arr_buf: *mut MpRtHeader = std::ptr::null_mut();
            out_err = std::ptr::null_mut();
            assert_eq!(
                mp_rt_gpu_buffer_from_array(dev, host_arr, 0, &mut from_arr_buf, &mut out_err,),
                0
            );
            assert!(!from_arr_buf.is_null());
            assert!(out_err.is_null());

            let mut out_arr: *mut MpRtHeader = std::ptr::null_mut();
            out_err = std::ptr::null_mut();
            assert_eq!(
                mp_rt_gpu_buffer_to_array(from_arr_buf, &mut out_arr, &mut out_err),
                0
            );
            assert!(!out_arr.is_null());
            assert!(out_err.is_null());
            assert_eq!(mp_rt_arr_len(out_arr), 4);
            assert_eq!(*(mp_rt_arr_get(out_arr, 0) as *const i32), 10);
            assert_eq!(*(mp_rt_arr_get(out_arr, 1) as *const i32), 20);
            assert_eq!(*(mp_rt_arr_get(out_arr, 2) as *const i32), 30);
            assert_eq!(*(mp_rt_arr_get(out_arr, 3) as *const i32), 40);

            let mut dst_buf: *mut MpRtHeader = std::ptr::null_mut();
            out_err = std::ptr::null_mut();
            assert_eq!(
                mp_rt_gpu_buffer_new(
                    dev,
                    TYPE_ID_I32,
                    std::mem::size_of::<i32>() as u64,
                    4,
                    0,
                    &mut dst_buf,
                    &mut out_err,
                ),
                0
            );
            assert!(!dst_buf.is_null());
            assert!(out_err.is_null());

            out_err = std::ptr::null_mut();
            assert_eq!(
                mp_rt_gpu_buffer_copy(from_arr_buf, dst_buf, &mut out_err),
                0
            );
            assert!(out_err.is_null());

            let mut copied_arr: *mut MpRtHeader = std::ptr::null_mut();
            out_err = std::ptr::null_mut();
            assert_eq!(
                mp_rt_gpu_buffer_to_array(dst_buf, &mut copied_arr, &mut out_err),
                0
            );
            assert!(out_err.is_null());
            assert_eq!(*(mp_rt_arr_get(copied_arr, 0) as *const i32), 10);
            assert_eq!(*(mp_rt_arr_get(copied_arr, 1) as *const i32), 20);

            out_err = std::ptr::null_mut();
            assert_eq!(mp_rt_gpu_device_sync(dev, &mut out_err), 0);
            assert!(out_err.is_null());

            let mut args_blob = [0_u8; 24];
            let ptr_word = (from_arr_buf as usize as u64).to_ne_bytes();
            args_blob[0..8].copy_from_slice(&ptr_word);
            let mut sync_err: *mut MpRtHeader = std::ptr::null_mut();
            assert_eq!(
                mp_rt_gpu_launch_sync(
                    dev,
                    42,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    args_blob.as_ptr(),
                    args_blob.len() as u64,
                    &mut sync_err,
                ),
                0
            );
            assert!(sync_err.is_null());

            let mut fence: *mut MpRtHeader = std::ptr::null_mut();
            let mut launch_err: *mut MpRtHeader = std::ptr::null_mut();
            assert_eq!(
                mp_rt_gpu_launch_async(
                    dev,
                    42,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    args_blob.as_ptr(),
                    args_blob.len() as u64,
                    &mut fence,
                    &mut launch_err,
                ),
                0
            );
            assert!(!fence.is_null());
            assert!(launch_err.is_null());

            let mut done = 0_u8;
            let mut wait_err: *mut MpRtHeader = std::ptr::null_mut();
            assert_eq!(mp_rt_gpu_fence_wait(fence, 0, &mut done, &mut wait_err), 0);
            assert_eq!(done, 1);
            assert!(wait_err.is_null());
            mp_rt_gpu_fence_free(fence);

            let addr = make_str("127.0.0.1");
            let mut err_msg: *mut MpRtHeader = std::ptr::null_mut();
            assert_eq!(
                mp_rt_web_serve(
                    std::ptr::null_mut(),
                    addr,
                    8080,
                    1,
                    1,
                    1_000_000,
                    1_000,
                    1_000,
                    0,
                    &mut err_msg as *mut *mut MpRtHeader,
                ),
                -1
            );
            assert!(!err_msg.is_null());
            let err_text = read_str(err_msg);
            assert!(err_text.contains("invalid web service handle"));
            mp_rt_release_strong(err_msg);
            mp_rt_release_strong(addr);

            mp_rt_release_strong(out_buf);
            mp_rt_release_strong(from_arr_buf);
            mp_rt_release_strong(dst_buf);
            mp_rt_release_strong(out_arr);
            mp_rt_release_strong(copied_arr);
            mp_rt_release_strong(host_arr);
            mp_rt_release_strong(dev);
            mp_rt_release_strong(dev2);
        }
    }

    #[test]
    fn test_gpu_launch_sync_accepts_unaligned_args_blob() {
        let _guard = gpu_test_lock();
        unsafe {
            let params: [MpRtGpuParam; 1] = [MpRtGpuParam {
                kind: 1,
                _reserved0: 0,
                _reserved1: 0,
                type_id: 0,
                offset_or_binding: 0,
                size: 0,
            }];
            let entries = [MpRtGpuKernelEntry {
                sid_hash: 77,
                backend: 1,
                blob: std::ptr::null(),
                blob_len: 0,
                num_params: 1,
                params: params.as_ptr(),
                num_buffers: 1,
                push_const_size: 0,
            }];
            mp_rt_gpu_register_kernels(entries.as_ptr().cast::<u8>(), entries.len() as i32);

            let mut dev: *mut MpRtHeader = std::ptr::null_mut();
            let mut err: *mut MpRtHeader = std::ptr::null_mut();
            assert_eq!(mp_rt_gpu_device_default(&mut dev, &mut err), 0);
            assert!(err.is_null());
            assert!(!dev.is_null());

            let mut buf: *mut MpRtHeader = std::ptr::null_mut();
            assert_eq!(
                mp_rt_gpu_buffer_new(
                    dev,
                    TYPE_ID_I32,
                    std::mem::size_of::<i32>() as u64,
                    1,
                    0,
                    &mut buf,
                    &mut err
                ),
                0
            );
            assert!(err.is_null());
            assert!(!buf.is_null());

            let mut blob = vec![0_u8; 8 + 1];
            let unaligned = blob.as_mut_ptr().add(1);
            let ptr_word = (buf as usize as u64).to_ne_bytes();
            std::ptr::copy_nonoverlapping(ptr_word.as_ptr(), unaligned, ptr_word.len());

            let mut launch_err: *mut MpRtHeader = std::ptr::null_mut();
            assert_eq!(
                mp_rt_gpu_launch_sync(
                    dev,
                    77,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    unaligned as *const u8,
                    8,
                    &mut launch_err
                ),
                0
            );
            assert!(launch_err.is_null());

            mp_rt_release_strong(buf);
            mp_rt_release_strong(dev);
        }
    }

    #[test]
    fn test_channel_runtime_roundtrip() {
        unsafe {
            let elem_size = std::mem::size_of::<i32>() as u64;
            let ch = mp_rt_channel_new(TYPE_ID_I32, elem_size);
            assert!(!ch.is_null());

            let input = 0x1234_i32;
            mp_rt_channel_send(ch, (&input as *const i32).cast::<u8>(), elem_size);
            let mut output = 0_i32;
            assert_eq!(
                mp_rt_channel_recv(ch, (&mut output as *mut i32).cast::<u8>(), elem_size),
                1
            );
            assert_eq!(output, input);

            mp_rt_channel_send(std::ptr::null_mut(), std::ptr::null(), elem_size);
            assert_eq!(
                mp_rt_channel_recv(std::ptr::null_mut(), std::ptr::null_mut(), elem_size),
                0
            );
        }
    }

    #[test]
    fn test_mp_rt_alloc_sets_reserved_layout_size() {
        unsafe {
            mp_rt_init();
            let info = MpRtTypeInfo {
                type_id: 9901,
                flags: FLAG_HEAP,
                payload_size: 8,
                payload_align: 8,
                drop_fn: None,
                debug_fqn: c"test.ReservedAlloc".as_ptr(),
            };
            mp_rt_register_types(&info as *const MpRtTypeInfo, 1);

            let obj = mp_rt_alloc(9901, 8, 8, FLAG_HEAP);
            assert!(!obj.is_null());
            assert!(
                (*obj).reserved0 >= (std::mem::size_of::<MpRtHeader>() + 8) as u64,
                "reserved0 must include full allocation size for dealloc"
            );
            mp_rt_release_strong(obj);
        }
    }

    #[test]
    fn test_arc_retain_release_cycle() {
        unsafe {
            mp_rt_init();

            // Register a dummy type so dealloc works.
            let info = MpRtTypeInfo {
                type_id: 1000,
                flags: FLAG_HEAP,
                payload_size: 8,
                payload_align: 8,
                drop_fn: None,
                debug_fqn: std::ptr::null(),
            };
            mp_rt_register_types(&info as *const MpRtTypeInfo, 1);

            // Use the string allocator as a concrete builtin that sets reserved0 correctly.
            let obj = make_str("hello");

            // Initial strong == 1.
            assert_eq!((*obj).strong.load(Ordering::Relaxed), 1);

            // Retain -> strong == 2.
            mp_rt_retain_strong(obj);
            assert_eq!((*obj).strong.load(Ordering::Relaxed), 2);

            // Release once -> strong == 1.
            mp_rt_release_strong(obj);
            assert_eq!((*obj).strong.load(Ordering::Relaxed), 1);

            // Release again -> strong == 0 (triggers drop + dealloc).
            // We can't safely read from obj after this, but as long as we
            // don't crash the test passes.
            mp_rt_release_strong(obj);
            // obj is now freed; do not access it.
        }
    }

    // -----------------------------------------------------------------------
    // Weak: upgrade success and failure
    // -----------------------------------------------------------------------

    #[test]
    fn test_weak_upgrade_success_and_failure() {
        unsafe {
            mp_rt_init();

            let obj = make_str("weak-test");
            // Initial: strong=1, weak=1.

            // Take a weak reference (increment weak manually).
            mp_rt_retain_weak(obj);
            // Now strong=1, weak=2.

            // Upgrade while strong > 0 — should succeed.
            let upgraded = mp_rt_weak_upgrade(obj);
            assert!(!upgraded.is_null());
            assert_eq!(upgraded, obj);
            // Now strong=2.

            // Release the upgraded strong reference.
            mp_rt_release_strong(upgraded);
            // strong=1.

            // Release the original strong reference — triggers drop but NOT dealloc
            // because weak is still 2.
            mp_rt_release_strong(obj);
            // strong=0, weak=2.

            // Now try to upgrade — strong is 0, so should fail.
            let failed = mp_rt_weak_upgrade(obj);
            assert!(failed.is_null());

            // Only the explicit weak remains here:
            // strong drop consumed the implicit weak reference.
            mp_rt_release_weak(obj);
        }
    }

    // -----------------------------------------------------------------------
    // String: creation, bytes, len, eq, concat, slice
    // -----------------------------------------------------------------------

    #[test]
    fn test_str_creation_and_bytes() {
        unsafe {
            let s = make_str("hello");
            let content = read_str(s);
            assert_eq!(content, "hello");
            mp_rt_release_strong(s);
        }
    }

    #[test]
    fn test_str_len() {
        unsafe {
            let s = make_str("world");
            assert_eq!(mp_rt_str_len(s), 5);
            mp_rt_release_strong(s);
        }
    }

    #[test]
    fn test_str_eq() {
        unsafe {
            let a = make_str("foo");
            let b = make_str("foo");
            let c = make_str("bar");
            assert_eq!(mp_rt_str_eq(a, b), 1);
            assert_eq!(mp_rt_str_eq(a, c), 0);
            assert_eq!(mp_rt_str_cmp(a, b), 0);
            assert!(mp_rt_str_cmp(c, a) < 0);
            assert!(mp_rt_str_cmp(a, c) > 0);
            mp_rt_release_strong(a);
            mp_rt_release_strong(b);
            mp_rt_release_strong(c);
        }
    }

    #[test]
    fn test_bytes_helpers() {
        unsafe {
            let a = [1_u8, 2, 3];
            let b = [1_u8, 2, 3];
            let c = [1_u8, 2, 4];
            assert_eq!(mp_rt_bytes_eq(a.as_ptr(), b.as_ptr(), 3), 1);
            assert_eq!(mp_rt_bytes_eq(a.as_ptr(), c.as_ptr(), 3), 0);
            assert_eq!(mp_rt_bytes_cmp(a.as_ptr(), b.as_ptr(), 3), 0);
            assert!(mp_rt_bytes_cmp(a.as_ptr(), c.as_ptr(), 3) < 0);
            assert_ne!(mp_rt_bytes_hash(a.as_ptr(), 3), 0);
        }
    }

    #[test]
    fn test_str_concat() {
        unsafe {
            let a = make_str("hello");
            let b = make_str(", world");
            let c = mp_rt_str_concat(a, b);
            assert_eq!(read_str(c), "hello, world");
            mp_rt_release_strong(a);
            mp_rt_release_strong(b);
            mp_rt_release_strong(c);
        }
    }

    #[test]
    fn test_str_slice() {
        unsafe {
            let s = make_str("hello, world");
            let sl = mp_rt_str_slice(s, 7, 12);
            assert_eq!(read_str(sl), "world");
            mp_rt_release_strong(s);
            mp_rt_release_strong(sl);
        }
    }

    #[test]
    fn test_str_empty() {
        unsafe {
            let s = make_str("");
            assert_eq!(mp_rt_str_len(s), 0);
            assert_eq!(read_str(s), "");
            mp_rt_release_strong(s);
        }
    }

    // -----------------------------------------------------------------------
    // StringBuilder: append + build
    // -----------------------------------------------------------------------

    #[test]
    fn test_strbuilder_append_and_build() {
        unsafe {
            let b = mp_rt_strbuilder_new();

            let hello = make_str("hello");
            mp_rt_strbuilder_append_str(b, hello);
            mp_rt_release_strong(hello);

            mp_rt_strbuilder_append_i64(b, 42_i64);
            mp_rt_strbuilder_append_i32(b, -7_i32);
            mp_rt_strbuilder_append_f64(b, 2.5_f64);
            mp_rt_strbuilder_append_bool(b, 1);
            mp_rt_strbuilder_append_bool(b, 0);

            let result = mp_rt_strbuilder_build(b);
            let s = read_str(result);

            assert!(s.starts_with("hello"));
            assert!(s.contains("42"));
            assert!(s.contains("-7"));
            assert!(s.contains("2.5"));
            assert!(s.contains("true"));
            assert!(s.contains("false"));

            mp_rt_release_strong(result);
            mp_rt_release_strong(b);
        }
    }

    #[test]
    fn test_strbuilder_empty_build() {
        unsafe {
            let b = mp_rt_strbuilder_new();
            let result = mp_rt_strbuilder_build(b);
            assert_eq!(read_str(result), "");
            mp_rt_release_strong(result);
            mp_rt_release_strong(b);
        }
    }

    #[test]
    fn test_arr_map_filter_reduce() {
        unsafe {
            let arr = mp_rt_arr_new(4, std::mem::size_of::<i32>() as u64, 0);
            for v in [1_i32, 2_i32, 3_i32, 4_i32] {
                mp_rt_arr_push(
                    arr,
                    &v as *const i32 as *const u8,
                    std::mem::size_of::<i32>() as u64,
                );
            }

            let map_callable = make_test_callable(arr_map_double as *const () as *mut u8);
            let mapped = mp_rt_arr_map(arr, map_callable, 4, std::mem::size_of::<i32>() as u64);
            assert_eq!(mp_rt_arr_len(mapped), 4);
            assert_eq!(*(mp_rt_arr_get(mapped, 0) as *const i32), 2);
            assert_eq!(*(mp_rt_arr_get(mapped, 1) as *const i32), 4);
            assert_eq!(*(mp_rt_arr_get(mapped, 2) as *const i32), 6);
            assert_eq!(*(mp_rt_arr_get(mapped, 3) as *const i32), 8);

            let filter_callable = make_test_callable(arr_filter_even as *const () as *mut u8);
            let filtered = mp_rt_arr_filter(arr, filter_callable);
            assert_eq!(mp_rt_arr_len(filtered), 2);
            assert_eq!(*(mp_rt_arr_get(filtered, 0) as *const i32), 2);
            assert_eq!(*(mp_rt_arr_get(filtered, 1) as *const i32), 4);

            let reduce_callable = make_test_callable(arr_reduce_sum as *const () as *mut u8);
            let mut sum = 0_i32;
            mp_rt_arr_reduce(
                filtered,
                &mut sum as *mut i32 as *mut u8,
                std::mem::size_of::<i32>() as u64,
                reduce_callable,
            );
            assert_eq!(sum, 6);

            mp_rt_release_strong(arr);
            mp_rt_release_strong(mapped);
            mp_rt_release_strong(filtered);
            mp_rt_release_strong(map_callable as *mut MpRtHeader);
            mp_rt_release_strong(filter_callable as *mut MpRtHeader);
            mp_rt_release_strong(reduce_callable as *mut MpRtHeader);
        }
    }

    #[test]
    fn test_mutex_lock_unlock() {
        unsafe {
            let m = mp_rt_mutex_new();
            mp_rt_mutex_lock(m);
            mp_rt_mutex_unlock(m);
            mp_rt_mutex_lock(m);
            mp_rt_mutex_unlock(m);
            drop(Box::from_raw(m as *mut MpRtMutexState));
        }
    }

    #[test]
    fn test_str_parse_i64() {
        unsafe {
            let s = make_str("-12345");
            assert_eq!(mp_rt_str_parse_i64(s), -12345);
            mp_rt_release_strong(s);
        }
    }

    #[test]
    fn test_str_try_parse_i64_success_and_failure() {
        unsafe {
            let good = make_str("-12345");
            let mut parsed = 0_i64;
            let mut errmsg: *mut MpRtHeader = std::ptr::null_mut();
            assert_eq!(
                mp_rt_str_try_parse_i64(good, &mut parsed as *mut i64, &mut errmsg),
                MP_RT_OK
            );
            assert_eq!(parsed, -12345);
            assert!(errmsg.is_null());
            mp_rt_release_strong(good);

            let bad = make_str("oops");
            let mut parsed_bad = 0_i64;
            let mut bad_errmsg: *mut MpRtHeader = std::ptr::null_mut();
            assert_eq!(
                mp_rt_str_try_parse_i64(bad, &mut parsed_bad as *mut i64, &mut bad_errmsg),
                MP_RT_ERR_INVALID_FORMAT
            );
            assert!(!bad_errmsg.is_null());
            mp_rt_release_strong(bad_errmsg);
            mp_rt_release_strong(bad);
        }
    }

    #[test]
    fn test_json_try_encode_decode_roundtrip_i64() {
        unsafe {
            let input = 987_i64;
            let mut json: *mut MpRtHeader = std::ptr::null_mut();
            let mut errmsg: *mut MpRtHeader = std::ptr::null_mut();
            assert_eq!(
                mp_rt_json_try_encode(
                    (&input as *const i64).cast::<u8>() as *mut u8,
                    TYPE_ID_I64,
                    &mut json,
                    &mut errmsg,
                ),
                MP_RT_OK
            );
            assert!(!json.is_null());
            assert!(errmsg.is_null());

            let mut decoded: *mut u8 = std::ptr::null_mut();
            assert_eq!(
                mp_rt_json_try_decode(json, TYPE_ID_I64, &mut decoded, &mut errmsg),
                MP_RT_OK
            );
            assert!(!decoded.is_null());
            assert!(errmsg.is_null());
            assert_eq!(*(decoded as *const i64), input);

            drop(Box::from_raw(decoded as *mut i64));
            mp_rt_release_strong(json);
        }
    }

    #[test]
    fn test_future_poll_take() {
        unsafe {
            let state = Box::into_raw(Box::new(MpRtFutureState {
                ready: false,
                value: std::ptr::null_mut(),
            }));
            let future = state as *mut u8;
            assert_eq!(mp_rt_future_poll(future), 0);

            let result_ptr = Box::into_raw(Box::new(77_i64)) as *mut u8;
            (*state).ready = true;
            (*state).value = result_ptr;

            assert_eq!(mp_rt_future_poll(future), 1);
            let mut taken: *mut u8 = std::ptr::null_mut();
            mp_rt_future_take(future, (&mut taken as *mut *mut u8).cast::<u8>());
            assert_eq!(*(taken as *const i64), 77);

            drop(Box::from_raw(taken as *mut i64));
            drop(Box::from_raw(state));
        }
    }

    // -----------------------------------------------------------------------
    // Type registry
    // -----------------------------------------------------------------------

    #[test]
    fn test_type_registry() {
        unsafe {
            mp_rt_init();

            assert!(mp_rt_type_info(9999).is_null());

            let fqn = b"test.TFoo\0";
            let info = MpRtTypeInfo {
                type_id: 9999,
                flags: FLAG_HEAP | FLAG_HAS_DROP,
                payload_size: 16,
                payload_align: 8,
                drop_fn: None,
                debug_fqn: fqn.as_ptr() as *const c_char,
            };
            mp_rt_register_types(&info as *const MpRtTypeInfo, 1);

            let found = mp_rt_type_info(9999);
            assert!(!found.is_null());
            assert_eq!((*found).type_id, 9999);
            assert_eq!((*found).payload_size, 16);
        }
    }

    #[test]
    fn test_public_c_header_contains_core_abi_surface() {
        let header = include_str!("../include/magpie_rt.h");
        assert!(header.contains("typedef struct MpRtHeader"));
        assert!(header.contains("typedef struct MpRtTypeInfo"));
        assert!(header.contains("mp_rt_alloc("));
        assert!(header.contains("mp_rt_arr_new("));
        assert!(header.contains("mp_rt_map_new("));
        assert!(header.contains("mp_rt_str_concat("));
        assert!(header.contains("mp_rt_arr_foreach("));
        assert!(header.contains("mp_rt_callable_new("));
        assert!(header.contains("mp_rt_mutex_new("));
        assert!(header.contains("mp_rt_channel_new("));
        assert!(header.contains("mp_rt_web_serve("));
        assert!(header.contains("typedef struct MpRtGpuKernelEntry"));
        assert!(header.contains("mp_rt_gpu_register_kernels("));
        assert!(header.contains("mp_rt_gpu_launch_sync("));
        assert!(header.contains("mp_rt_gpu_launch_async("));
        assert!(header.contains("mp_rt_gpu_kernel_load("));
        assert!(header.contains("mp_std_hash_str("));
        assert!(header.contains("mp_std_block_on("));
        assert!(header.contains("mp_std_spawn_task("));
    }
}

// ---------------------------------------------------------------------------
// §20.1.5  Collection runtime ABI
// ---------------------------------------------------------------------------

#[repr(C)]
struct MpRtArrayPayload {
    len: u64,
    cap: u64,
    elem_size: u64,
    data_ptr: *mut u8,
    elem_type_id: u32,
    _reserved: u32,
}

#[repr(C)]
struct MpRtMapPayload {
    len: u64,
    cap: u64,
    key_size: u64,
    val_size: u64,
    hash_fn: MpRtHashFn,
    eq_fn: MpRtEqFn,
    data_ptr: *mut u8,
    key_type_id: u32,
    val_type_id: u32,
}

const COLLECTION_DATA_ALIGN: usize = 8;
const MAP_SLOT_EMPTY: u8 = 0;
const MAP_SLOT_FULL: u8 = 1;
const MAP_SLOT_TOMBSTONE: u8 = 2;

static COLLECTION_TYPES_ONCE: Once = Once::new();

#[inline]
fn usize_from_u64(v: u64, ctx: &str) -> usize {
    usize::try_from(v).expect(ctx)
}

#[inline]
fn align_up(v: usize, align: usize) -> usize {
    let mask = align - 1;
    v.checked_add(mask).expect("align overflow") & !mask
}

#[inline]
fn mul_usize(a: usize, b: usize, ctx: &str) -> usize {
    a.checked_mul(b).expect(ctx)
}

#[inline]
fn add_usize(a: usize, b: usize, ctx: &str) -> usize {
    a.checked_add(b).expect(ctx)
}

#[inline]
unsafe fn bytes_eq(a: *const u8, b: *const u8, len: usize) -> bool {
    if len == 0 {
        return true;
    }
    for i in 0..len {
        if *a.add(i) != *b.add(i) {
            return false;
        }
    }
    true
}

#[inline]
unsafe fn bytes_cmp(a: *const u8, b: *const u8, len: usize) -> i32 {
    for i in 0..len {
        let lhs = *a.add(i);
        let rhs = *b.add(i);
        if lhs < rhs {
            return -1;
        }
        if lhs > rhs {
            return 1;
        }
    }
    0
}

#[inline]
unsafe fn bytes_hash(data: *const u8, len: usize) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    for i in 0..len {
        hash ^= *data.add(i) as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_bytes_hash(data: *const u8, len: u64) -> u64 {
    let len = usize_from_u64(len, "byte length too large");
    if len == 0 {
        return bytes_hash(std::ptr::null(), 0);
    }
    assert!(
        !data.is_null(),
        "mp_rt_bytes_hash: null data with non-zero len"
    );
    bytes_hash(data, len)
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_bytes_eq(a: *const u8, b: *const u8, len: u64) -> i32 {
    let len = usize_from_u64(len, "byte length too large");
    if len == 0 {
        return 1;
    }
    assert!(!a.is_null(), "mp_rt_bytes_eq: null lhs with non-zero len");
    assert!(!b.is_null(), "mp_rt_bytes_eq: null rhs with non-zero len");
    if bytes_eq(a, b, len) {
        1
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_bytes_cmp(a: *const u8, b: *const u8, len: u64) -> i32 {
    let len = usize_from_u64(len, "byte length too large");
    if len == 0 {
        return 0;
    }
    assert!(!a.is_null(), "mp_rt_bytes_cmp: null lhs with non-zero len");
    assert!(!b.is_null(), "mp_rt_bytes_cmp: null rhs with non-zero len");
    bytes_cmp(a, b, len)
}

#[inline]
unsafe fn slot_load_handle(slot: *const u8) -> *mut MpRtHeader {
    if slot.is_null() {
        return std::ptr::null_mut();
    }
    *(slot as *const *mut MpRtHeader)
}

#[inline]
unsafe fn str_handle_bytes(handle: *mut MpRtHeader) -> Option<(*const u8, usize)> {
    if handle.is_null() || (*handle).type_id != TYPE_ID_STR {
        return None;
    }
    let mut len = 0_u64;
    let ptr = mp_rt_str_bytes(handle, &mut len);
    Some((ptr, usize_from_u64(len, "string length too large")))
}

#[inline]
unsafe fn str_slot_eq(a_slot: *const u8, b_slot: *const u8) -> bool {
    let a = slot_load_handle(a_slot);
    let b = slot_load_handle(b_slot);
    match (str_handle_bytes(a), str_handle_bytes(b)) {
        (Some((a_ptr, a_len)), Some((b_ptr, b_len))) => {
            a_len == b_len && bytes_eq(a_ptr, b_ptr, a_len)
        }
        _ => a == b,
    }
}

#[inline]
unsafe fn str_slot_cmp(a_slot: *const u8, b_slot: *const u8) -> i32 {
    let a = slot_load_handle(a_slot);
    let b = slot_load_handle(b_slot);
    match (str_handle_bytes(a), str_handle_bytes(b)) {
        (Some((a_ptr, a_len)), Some((b_ptr, b_len))) => {
            let shared = std::cmp::min(a_len, b_len);
            let prefix = bytes_cmp(a_ptr, b_ptr, shared);
            if prefix != 0 {
                prefix
            } else if a_len < b_len {
                -1
            } else if a_len > b_len {
                1
            } else {
                0
            }
        }
        _ => {
            let ap = a as usize;
            let bp = b as usize;
            if ap < bp {
                -1
            } else if ap > bp {
                1
            } else {
                0
            }
        }
    }
}

#[inline]
unsafe fn str_slot_hash(slot: *const u8) -> u64 {
    let handle = slot_load_handle(slot);
    match str_handle_bytes(handle) {
        Some((ptr, len)) => bytes_hash(ptr, len),
        None => {
            let addr = (handle as usize as u64).to_le_bytes();
            bytes_hash(addr.as_ptr(), addr.len())
        }
    }
}

#[inline]
fn primitive_type_size(type_id: u32) -> Option<usize> {
    match type_id {
        TYPE_ID_BOOL => Some(std::mem::size_of::<u8>()),
        TYPE_ID_I8 => Some(std::mem::size_of::<i8>()),
        TYPE_ID_I16 => Some(std::mem::size_of::<i16>()),
        TYPE_ID_I32 => Some(std::mem::size_of::<i32>()),
        TYPE_ID_I64 => Some(std::mem::size_of::<i64>()),
        TYPE_ID_U8 => Some(std::mem::size_of::<u8>()),
        TYPE_ID_U16 => Some(std::mem::size_of::<u16>()),
        TYPE_ID_U32 => Some(std::mem::size_of::<u32>()),
        TYPE_ID_U64 => Some(std::mem::size_of::<u64>()),
        TYPE_ID_F32 => Some(std::mem::size_of::<f32>()),
        TYPE_ID_F64 => Some(std::mem::size_of::<f64>()),
        _ => None,
    }
}

#[inline]
unsafe fn primitive_slot_cmp(
    type_id: u32,
    elem_size: usize,
    a: *const u8,
    b: *const u8,
) -> Option<i32> {
    if primitive_type_size(type_id)? != elem_size {
        return None;
    }
    let out = match type_id {
        TYPE_ID_BOOL => {
            let av = *a != 0;
            let bv = *b != 0;
            if av == bv {
                0
            } else if !av && bv {
                -1
            } else {
                1
            }
        }
        TYPE_ID_I8 => {
            let av = *(a as *const i8);
            let bv = *(b as *const i8);
            if av < bv {
                -1
            } else if av > bv {
                1
            } else {
                0
            }
        }
        TYPE_ID_I16 => {
            let av = *(a as *const i16);
            let bv = *(b as *const i16);
            if av < bv {
                -1
            } else if av > bv {
                1
            } else {
                0
            }
        }
        TYPE_ID_I32 => {
            let av = *(a as *const i32);
            let bv = *(b as *const i32);
            if av < bv {
                -1
            } else if av > bv {
                1
            } else {
                0
            }
        }
        TYPE_ID_I64 => {
            let av = *(a as *const i64);
            let bv = *(b as *const i64);
            if av < bv {
                -1
            } else if av > bv {
                1
            } else {
                0
            }
        }
        TYPE_ID_U8 => {
            let av = *a;
            let bv = *b;
            if av < bv {
                -1
            } else if av > bv {
                1
            } else {
                0
            }
        }
        TYPE_ID_U16 => {
            let av = *(a as *const u16);
            let bv = *(b as *const u16);
            if av < bv {
                -1
            } else if av > bv {
                1
            } else {
                0
            }
        }
        TYPE_ID_U32 => {
            let av = *(a as *const u32);
            let bv = *(b as *const u32);
            if av < bv {
                -1
            } else if av > bv {
                1
            } else {
                0
            }
        }
        TYPE_ID_U64 => {
            let av = *(a as *const u64);
            let bv = *(b as *const u64);
            if av < bv {
                -1
            } else if av > bv {
                1
            } else {
                0
            }
        }
        TYPE_ID_F32 => {
            let av = *(a as *const f32);
            let bv = *(b as *const f32);
            if av < bv {
                -1
            } else if av > bv {
                1
            } else if av == bv {
                0
            } else {
                let ab = av.to_bits();
                let bb = bv.to_bits();
                if ab < bb {
                    -1
                } else if ab > bb {
                    1
                } else {
                    0
                }
            }
        }
        TYPE_ID_F64 => {
            let av = *(a as *const f64);
            let bv = *(b as *const f64);
            if av < bv {
                -1
            } else if av > bv {
                1
            } else if av == bv {
                0
            } else {
                let ab = av.to_bits();
                let bb = bv.to_bits();
                if ab < bb {
                    -1
                } else if ab > bb {
                    1
                } else {
                    0
                }
            }
        }
        _ => unreachable!("primitive type size check should gate non-primitive ids"),
    };
    Some(out)
}

#[inline]
unsafe fn collection_alloc_zeroed(size: usize) -> *mut u8 {
    let layout = Layout::from_size_align(size.max(1), COLLECTION_DATA_ALIGN)
        .expect("bad collection alloc layout");
    let ptr = alloc::alloc_zeroed(layout);
    if ptr.is_null() {
        alloc::handle_alloc_error(layout);
    }
    ptr
}

#[inline]
unsafe fn collection_realloc(ptr: *mut u8, old_size: usize, new_size: usize) -> *mut u8 {
    let old_layout = Layout::from_size_align(old_size.max(1), COLLECTION_DATA_ALIGN)
        .expect("bad collection realloc old layout");
    let new_ptr = alloc::realloc(ptr, old_layout, new_size.max(1));
    if new_ptr.is_null() {
        let new_layout = Layout::from_size_align(new_size.max(1), COLLECTION_DATA_ALIGN)
            .expect("bad collection realloc new layout");
        alloc::handle_alloc_error(new_layout);
    }
    new_ptr
}

#[inline]
unsafe fn collection_dealloc(ptr: *mut u8, size: usize) {
    if ptr.is_null() {
        return;
    }
    let layout = Layout::from_size_align(size.max(1), COLLECTION_DATA_ALIGN)
        .expect("bad collection dealloc layout");
    alloc::dealloc(ptr, layout);
}

#[inline]
unsafe fn arr_payload(arr: *mut MpRtHeader) -> *mut MpRtArrayPayload {
    str_payload_base(arr) as *mut MpRtArrayPayload
}

#[inline]
unsafe fn map_payload(map: *mut MpRtHeader) -> *mut MpRtMapPayload {
    str_payload_base(map) as *mut MpRtMapPayload
}

#[inline]
fn array_bytes(cap: u64, elem_size: u64) -> usize {
    let cap = usize_from_u64(cap, "array cap too large");
    let elem_size = usize_from_u64(elem_size, "array elem_size too large");
    mul_usize(cap, elem_size, "array size overflow")
}

fn map_layout(cap: u64, key_size: u64, val_size: u64) -> (usize, usize, usize) {
    let cap = usize_from_u64(cap, "map cap too large");
    let key_size = usize_from_u64(key_size, "map key_size too large");
    let val_size = usize_from_u64(val_size, "map val_size too large");

    let keys_off = align_up(cap, COLLECTION_DATA_ALIGN);
    let keys_bytes = mul_usize(cap, key_size, "map keys bytes overflow");
    let vals_off = align_up(
        add_usize(keys_off, keys_bytes, "map keys offset overflow"),
        COLLECTION_DATA_ALIGN,
    );
    let vals_bytes = mul_usize(cap, val_size, "map vals bytes overflow");
    let total = add_usize(vals_off, vals_bytes, "map total bytes overflow");

    (total, keys_off, vals_off)
}

#[inline]
unsafe fn map_state_ptr(payload: *mut MpRtMapPayload, idx: usize) -> *mut u8 {
    (*payload).data_ptr.add(idx)
}

#[inline]
unsafe fn map_key_ptr(payload: *mut MpRtMapPayload, idx: usize) -> *mut u8 {
    let (_, keys_off, _) = map_layout((*payload).cap, (*payload).key_size, (*payload).val_size);
    let key_size = usize_from_u64((*payload).key_size, "map key_size too large");
    (*payload)
        .data_ptr
        .add(keys_off + mul_usize(idx, key_size, "map key idx overflow"))
}

#[inline]
unsafe fn map_val_ptr(payload: *mut MpRtMapPayload, idx: usize) -> *mut u8 {
    let (_, _, vals_off) = map_layout((*payload).cap, (*payload).key_size, (*payload).val_size);
    let val_size = usize_from_u64((*payload).val_size, "map val_size too large");
    (*payload)
        .data_ptr
        .add(vals_off + mul_usize(idx, val_size, "map val idx overflow"))
}

unsafe extern "C" fn mp_rt_arr_drop(obj: *mut MpRtHeader) {
    let payload = arr_payload(obj);
    if (*payload).elem_size == 0 || (*payload).cap == 0 {
        (*payload).data_ptr = std::ptr::null_mut();
        return;
    }
    let bytes = array_bytes((*payload).cap, (*payload).elem_size);
    collection_dealloc((*payload).data_ptr, bytes);
    (*payload).data_ptr = std::ptr::null_mut();
}

unsafe extern "C" fn mp_rt_map_drop(obj: *mut MpRtHeader) {
    let payload = map_payload(obj);
    if (*payload).cap == 0 {
        (*payload).data_ptr = std::ptr::null_mut();
        return;
    }
    let (bytes, _, _) = map_layout((*payload).cap, (*payload).key_size, (*payload).val_size);
    collection_dealloc((*payload).data_ptr, bytes);
    (*payload).data_ptr = std::ptr::null_mut();
}

fn ensure_collection_types_registered() {
    COLLECTION_TYPES_ONCE.call_once(|| unsafe {
        let infos = [
            MpRtTypeInfo {
                type_id: TYPE_ID_ARRAY,
                flags: FLAG_HEAP | FLAG_HAS_DROP,
                payload_size: std::mem::size_of::<MpRtArrayPayload>() as u64,
                payload_align: std::mem::align_of::<MpRtArrayPayload>() as u64,
                drop_fn: Some(mp_rt_arr_drop),
                debug_fqn: c"core.Array".as_ptr(),
            },
            MpRtTypeInfo {
                type_id: TYPE_ID_MAP,
                flags: FLAG_HEAP | FLAG_HAS_DROP,
                payload_size: std::mem::size_of::<MpRtMapPayload>() as u64,
                payload_align: std::mem::align_of::<MpRtMapPayload>() as u64,
                drop_fn: Some(mp_rt_map_drop),
                debug_fqn: c"core.Map".as_ptr(),
            },
        ];
        mp_rt_register_types(infos.as_ptr(), infos.len() as u32);
    });
}

unsafe fn arr_reserve(payload: *mut MpRtArrayPayload, needed: u64) {
    if needed <= (*payload).cap {
        return;
    }
    let mut new_cap = (*payload).cap.max(4);
    while new_cap < needed {
        new_cap = new_cap
            .checked_mul(2)
            .expect("mp_rt_arr_push: capacity overflow");
    }

    if (*payload).elem_size == 0 {
        (*payload).cap = new_cap;
        if (*payload).data_ptr.is_null() {
            (*payload).data_ptr = std::ptr::NonNull::<u8>::dangling().as_ptr();
        }
        return;
    }

    let new_bytes = array_bytes(new_cap, (*payload).elem_size);
    if (*payload).cap == 0 || (*payload).data_ptr.is_null() {
        (*payload).data_ptr = collection_alloc_zeroed(new_bytes);
    } else {
        let old_bytes = array_bytes((*payload).cap, (*payload).elem_size);
        (*payload).data_ptr = collection_realloc((*payload).data_ptr, old_bytes, new_bytes);
    }
    (*payload).cap = new_cap;
}

unsafe fn map_find_slot(payload: *mut MpRtMapPayload, key: *const u8) -> (bool, usize) {
    let cap = usize_from_u64((*payload).cap, "map cap too large");
    if cap == 0 || (*payload).data_ptr.is_null() {
        return (false, 0);
    }

    let mut first_tombstone = None;
    let key_size = usize_from_u64((*payload).key_size, "map key_size too large");
    let hash = match (*payload).hash_fn {
        Some(hash_fn) => hash_fn(key),
        None => {
            if (*payload).key_type_id == TYPE_ID_STR && key_size == std::mem::size_of::<usize>() {
                str_slot_hash(key)
            } else {
                bytes_hash(key, key_size)
            }
        }
    };
    let start = hash as usize % cap;

    for step in 0..cap {
        let idx = (start + step) % cap;
        let state = *map_state_ptr(payload, idx);
        match state {
            MAP_SLOT_EMPTY => return (false, first_tombstone.unwrap_or(idx)),
            MAP_SLOT_TOMBSTONE => {
                if first_tombstone.is_none() {
                    first_tombstone = Some(idx);
                }
            }
            MAP_SLOT_FULL => {
                let existing_key = map_key_ptr(payload, idx) as *const u8;
                let is_eq = match (*payload).eq_fn {
                    Some(eq_fn) => eq_fn(existing_key, key) != 0,
                    None => {
                        if (*payload).key_type_id == TYPE_ID_STR
                            && key_size == std::mem::size_of::<usize>()
                        {
                            str_slot_eq(existing_key, key)
                        } else {
                            bytes_eq(existing_key, key, key_size)
                        }
                    }
                };
                if is_eq {
                    return (true, idx);
                }
            }
            _ => unreachable!("invalid map slot state"),
        }
    }

    (false, first_tombstone.unwrap_or(start))
}

unsafe fn map_resize(payload: *mut MpRtMapPayload, new_cap: u64) {
    let new_cap = new_cap.max(8);
    let (new_bytes, _, _) = map_layout(new_cap, (*payload).key_size, (*payload).val_size);
    let new_data = collection_alloc_zeroed(new_bytes);

    let old_cap = (*payload).cap;
    let old_data = (*payload).data_ptr;

    (*payload).cap = new_cap;
    (*payload).data_ptr = new_data;

    if old_cap == 0 || old_data.is_null() {
        return;
    }

    let old_payload = MpRtMapPayload {
        len: (*payload).len,
        cap: old_cap,
        key_size: (*payload).key_size,
        val_size: (*payload).val_size,
        hash_fn: (*payload).hash_fn,
        eq_fn: (*payload).eq_fn,
        data_ptr: old_data,
        key_type_id: (*payload).key_type_id,
        val_type_id: (*payload).val_type_id,
    };

    let old_cap_usize = usize_from_u64(old_cap, "old map cap too large");
    let key_size = usize_from_u64((*payload).key_size, "map key_size too large");
    let val_size = usize_from_u64((*payload).val_size, "map val_size too large");

    for i in 0..old_cap_usize {
        if *old_data.add(i) != MAP_SLOT_FULL {
            continue;
        }

        let old_key = map_key_ptr(&old_payload as *const _ as *mut _, i) as *const u8;
        let old_val = map_val_ptr(&old_payload as *const _ as *mut _, i) as *const u8;

        let (_, insert_idx) = map_find_slot(payload, old_key);
        *map_state_ptr(payload, insert_idx) = MAP_SLOT_FULL;

        if key_size > 0 {
            std::ptr::copy_nonoverlapping(old_key, map_key_ptr(payload, insert_idx), key_size);
        }
        if val_size > 0 {
            std::ptr::copy_nonoverlapping(old_val, map_val_ptr(payload, insert_idx), val_size);
        }
    }

    let (old_bytes, _, _) = map_layout(old_cap, (*payload).key_size, (*payload).val_size);
    collection_dealloc(old_data, old_bytes);
}

unsafe fn map_ensure_capacity(payload: *mut MpRtMapPayload) {
    if (*payload).cap == 0 {
        map_resize(payload, 8);
        return;
    }

    // Grow around 70% load factor.
    let len_after = (*payload).len.checked_add(1).expect("map length overflow");
    if len_after.checked_mul(10).expect("map load factor overflow")
        > (*payload)
            .cap
            .checked_mul(7)
            .expect("map load factor overflow")
    {
        map_resize(
            payload,
            (*payload)
                .cap
                .checked_mul(2)
                .expect("map capacity overflow"),
        );
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_arr_new(
    elem_type_id: u32,
    elem_size: u64,
    capacity: u64,
) -> *mut MpRtHeader {
    ensure_collection_types_registered();

    let obj = alloc_builtin(
        TYPE_ID_ARRAY,
        FLAG_HEAP | FLAG_HAS_DROP,
        std::mem::size_of::<MpRtArrayPayload>(),
        std::mem::align_of::<MpRtArrayPayload>(),
    );

    let payload = arr_payload(obj);
    (*payload).len = 0;
    (*payload).cap = 0;
    (*payload).elem_size = elem_size;
    (*payload).data_ptr = std::ptr::null_mut();
    (*payload).elem_type_id = elem_type_id;
    (*payload)._reserved = 0;

    if capacity > 0 {
        arr_reserve(payload, capacity);
    }

    obj
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_arr_len(arr: *mut MpRtHeader) -> u64 {
    (*arr_payload(arr)).len
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_arr_get(arr: *mut MpRtHeader, idx: u64) -> *mut u8 {
    let payload = arr_payload(arr);
    assert!(idx < (*payload).len, "mp_rt_arr_get: out of bounds");

    if (*payload).elem_size == 0 {
        if (*payload).data_ptr.is_null() {
            (*payload).data_ptr = std::ptr::NonNull::<u8>::dangling().as_ptr();
        }
        return (*payload).data_ptr;
    }

    let idx = usize_from_u64(idx, "array index too large");
    let elem_size = usize_from_u64((*payload).elem_size, "array elem_size too large");
    (*payload)
        .data_ptr
        .add(mul_usize(idx, elem_size, "array index overflow"))
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_arr_set(
    arr: *mut MpRtHeader,
    idx: u64,
    val: *const u8,
    elem_size: u64,
) {
    let payload = arr_payload(arr);
    assert_eq!(
        elem_size,
        (*payload).elem_size,
        "mp_rt_arr_set: elem_size mismatch"
    );
    assert!(idx < (*payload).len, "mp_rt_arr_set: out of bounds");

    if elem_size == 0 {
        return;
    }

    let dst = mp_rt_arr_get(arr, idx);
    std::ptr::copy_nonoverlapping(
        val,
        dst,
        usize_from_u64(elem_size, "array elem_size too large"),
    );
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_arr_push(arr: *mut MpRtHeader, val: *const u8, elem_size: u64) {
    let payload = arr_payload(arr);
    assert_eq!(
        elem_size,
        (*payload).elem_size,
        "mp_rt_arr_push: elem_size mismatch"
    );

    let new_len = (*payload)
        .len
        .checked_add(1)
        .expect("mp_rt_arr_push: length overflow");
    arr_reserve(payload, new_len);

    if elem_size > 0 {
        let dst = (*payload)
            .data_ptr
            .add(array_bytes((*payload).len, elem_size));
        std::ptr::copy_nonoverlapping(
            val,
            dst,
            usize_from_u64(elem_size, "array elem_size too large"),
        );
    }
    (*payload).len = new_len;
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_arr_pop(arr: *mut MpRtHeader, out: *mut u8, elem_size: u64) -> i32 {
    let payload = arr_payload(arr);
    assert_eq!(
        elem_size,
        (*payload).elem_size,
        "mp_rt_arr_pop: elem_size mismatch"
    );
    if (*payload).len == 0 {
        return 0;
    }

    (*payload).len -= 1;
    if elem_size > 0 && !out.is_null() {
        let src = (*payload)
            .data_ptr
            .add(array_bytes((*payload).len, elem_size));
        std::ptr::copy_nonoverlapping(
            src,
            out,
            usize_from_u64(elem_size, "array elem_size too large"),
        );
    }
    1
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_arr_slice(
    arr: *mut MpRtHeader,
    start: u64,
    end: u64,
) -> *mut MpRtHeader {
    let payload = arr_payload(arr);
    assert!(
        start <= end && end <= (*payload).len,
        "mp_rt_arr_slice: out of bounds"
    );

    let out = mp_rt_arr_new((*payload).elem_type_id, (*payload).elem_size, end - start);
    let out_payload = arr_payload(out);
    let count = end - start;

    if count > 0 && (*payload).elem_size > 0 {
        let elem_size = usize_from_u64((*payload).elem_size, "array elem_size too large");
        let start_off = mul_usize(
            usize_from_u64(start, "array start too large"),
            elem_size,
            "array start overflow",
        );
        let total = mul_usize(
            usize_from_u64(count, "array count too large"),
            elem_size,
            "array slice overflow",
        );
        std::ptr::copy_nonoverlapping(
            (*payload).data_ptr.add(start_off),
            (*out_payload).data_ptr,
            total,
        );
    }

    (*out_payload).len = count;
    out
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_arr_contains(
    arr: *mut MpRtHeader,
    val: *const u8,
    elem_size: u64,
    eq_fn: MpRtEqFn,
) -> i32 {
    let payload = arr_payload(arr);
    assert_eq!(
        elem_size,
        (*payload).elem_size,
        "mp_rt_arr_contains: elem_size mismatch"
    );

    if (*payload).len == 0 {
        return 0;
    }

    let elem_size = usize_from_u64(elem_size, "array elem_size too large");
    for i in 0..usize_from_u64((*payload).len, "array len too large") {
        let elem_ptr = if elem_size == 0 {
            std::ptr::NonNull::<u8>::dangling().as_ptr()
        } else {
            (*payload)
                .data_ptr
                .add(mul_usize(i, elem_size, "array index overflow"))
        };
        let is_eq = match eq_fn {
            Some(eq_fn) => eq_fn(elem_ptr as *const u8, val) != 0,
            None => {
                if (*payload).elem_type_id == TYPE_ID_STR
                    && elem_size == std::mem::size_of::<*mut MpRtHeader>()
                {
                    str_slot_eq(elem_ptr as *const u8, val)
                } else {
                    bytes_eq(elem_ptr as *const u8, val, elem_size)
                }
            }
        };
        if is_eq {
            return 1;
        }
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_arr_sort(arr: *mut MpRtHeader, cmp: MpRtCmpFn) {
    let payload = arr_payload(arr);
    if (*payload).len < 2 || (*payload).elem_size == 0 {
        return;
    }

    let len = usize_from_u64((*payload).len, "array len too large");
    let elem_size = usize_from_u64((*payload).elem_size, "array elem_size too large");

    for i in 1..len {
        let mut j = i;
        while j > 0 {
            let lhs =
                (*payload)
                    .data_ptr
                    .add(mul_usize(j - 1, elem_size, "array lhs index overflow"));
            let rhs = (*payload)
                .data_ptr
                .add(mul_usize(j, elem_size, "array rhs index overflow"));
            let ord = match cmp {
                Some(cmp_fn) => cmp_fn(lhs as *const u8, rhs as *const u8),
                None => {
                    if (*payload).elem_type_id == TYPE_ID_STR
                        && elem_size == std::mem::size_of::<*mut MpRtHeader>()
                    {
                        str_slot_cmp(lhs as *const u8, rhs as *const u8)
                    } else if let Some(ord) = primitive_slot_cmp(
                        (*payload).elem_type_id,
                        elem_size,
                        lhs as *const u8,
                        rhs as *const u8,
                    ) {
                        ord
                    } else {
                        bytes_cmp(lhs as *const u8, rhs as *const u8, elem_size)
                    }
                }
            };
            if ord <= 0 {
                break;
            }
            std::ptr::swap_nonoverlapping(lhs, rhs, elem_size);
            j -= 1;
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_map_new(
    key_type_id: u32,
    val_type_id: u32,
    key_size: u64,
    val_size: u64,
    capacity: u64,
    hash_fn: MpRtHashFn,
    eq_fn: MpRtEqFn,
) -> *mut MpRtHeader {
    ensure_collection_types_registered();

    let obj = alloc_builtin(
        TYPE_ID_MAP,
        FLAG_HEAP | FLAG_HAS_DROP,
        std::mem::size_of::<MpRtMapPayload>(),
        std::mem::align_of::<MpRtMapPayload>(),
    );

    let payload = map_payload(obj);
    (*payload).len = 0;
    (*payload).cap = 0;
    (*payload).key_size = key_size;
    (*payload).val_size = val_size;
    (*payload).hash_fn = hash_fn;
    (*payload).eq_fn = eq_fn;
    (*payload).data_ptr = std::ptr::null_mut();
    (*payload).key_type_id = key_type_id;
    (*payload).val_type_id = val_type_id;

    if capacity > 0 {
        map_resize(payload, capacity.max(8).next_power_of_two());
    }

    obj
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_map_len(map: *mut MpRtHeader) -> u64 {
    (*map_payload(map)).len
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_map_get(
    map: *mut MpRtHeader,
    key: *const u8,
    key_size: u64,
) -> *mut u8 {
    let payload = map_payload(map);
    assert_eq!(
        key_size,
        (*payload).key_size,
        "mp_rt_map_get: key_size mismatch"
    );

    if (*payload).cap == 0 || (*payload).data_ptr.is_null() {
        return std::ptr::null_mut();
    }

    let (found, idx) = map_find_slot(payload, key);
    if found {
        map_val_ptr(payload, idx)
    } else {
        std::ptr::null_mut()
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_map_set(
    map: *mut MpRtHeader,
    key: *const u8,
    key_size: u64,
    val: *const u8,
    val_size: u64,
) {
    let payload = map_payload(map);
    assert_eq!(
        key_size,
        (*payload).key_size,
        "mp_rt_map_set: key_size mismatch"
    );
    assert_eq!(
        val_size,
        (*payload).val_size,
        "mp_rt_map_set: val_size mismatch"
    );

    map_ensure_capacity(payload);
    let (found, idx) = map_find_slot(payload, key);

    let key_size = usize_from_u64(key_size, "map key_size too large");
    let val_size = usize_from_u64(val_size, "map val_size too large");
    if !found {
        *map_state_ptr(payload, idx) = MAP_SLOT_FULL;
        if key_size > 0 {
            std::ptr::copy_nonoverlapping(key, map_key_ptr(payload, idx), key_size);
        }
        (*payload).len = (*payload).len.checked_add(1).expect("map len overflow");
    }

    if val_size > 0 {
        std::ptr::copy_nonoverlapping(val, map_val_ptr(payload, idx), val_size);
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_map_take(
    map: *mut MpRtHeader,
    key: *const u8,
    key_size: u64,
    out_val: *mut u8,
    val_size: u64,
) -> i32 {
    let payload = map_payload(map);
    assert_eq!(
        key_size,
        (*payload).key_size,
        "mp_rt_map_take: key_size mismatch"
    );
    assert_eq!(
        val_size,
        (*payload).val_size,
        "mp_rt_map_take: val_size mismatch"
    );

    if (*payload).cap == 0 || (*payload).data_ptr.is_null() {
        return 0;
    }

    let (found, idx) = map_find_slot(payload, key);
    if !found {
        return 0;
    }

    let val_size = usize_from_u64(val_size, "map val_size too large");
    if val_size > 0 && !out_val.is_null() {
        std::ptr::copy_nonoverlapping(map_val_ptr(payload, idx), out_val, val_size);
    }
    *map_state_ptr(payload, idx) = MAP_SLOT_TOMBSTONE;
    (*payload).len -= 1;
    1
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_map_delete(
    map: *mut MpRtHeader,
    key: *const u8,
    key_size: u64,
) -> i32 {
    let payload = map_payload(map);
    assert_eq!(
        key_size,
        (*payload).key_size,
        "mp_rt_map_delete: key_size mismatch"
    );
    if (*payload).cap == 0 || (*payload).data_ptr.is_null() {
        return 0;
    }

    let (found, idx) = map_find_slot(payload, key);
    if !found {
        return 0;
    }

    *map_state_ptr(payload, idx) = MAP_SLOT_TOMBSTONE;
    (*payload).len -= 1;
    1
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_map_contains_key(
    map: *mut MpRtHeader,
    key: *const u8,
    key_size: u64,
) -> i32 {
    if mp_rt_map_get(map, key, key_size).is_null() {
        0
    } else {
        1
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_map_keys(map: *mut MpRtHeader) -> *mut MpRtHeader {
    let payload = map_payload(map);
    let out = mp_rt_arr_new((*payload).key_type_id, (*payload).key_size, (*payload).len);
    if (*payload).len == 0 {
        return out;
    }

    let cap = usize_from_u64((*payload).cap, "map cap too large");
    for i in 0..cap {
        if *map_state_ptr(payload, i) == MAP_SLOT_FULL {
            let key_ptr = map_key_ptr(payload, i) as *const u8;
            mp_rt_arr_push(out, key_ptr, (*payload).key_size);
        }
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_map_values(map: *mut MpRtHeader) -> *mut MpRtHeader {
    let payload = map_payload(map);
    let out = mp_rt_arr_new((*payload).val_type_id, (*payload).val_size, (*payload).len);
    if (*payload).len == 0 {
        return out;
    }

    let cap = usize_from_u64((*payload).cap, "map cap too large");
    for i in 0..cap {
        if *map_state_ptr(payload, i) == MAP_SLOT_FULL {
            let val_ptr = map_val_ptr(payload, i) as *const u8;
            mp_rt_arr_push(out, val_ptr, (*payload).val_size);
        }
    }
    out
}

#[cfg(test)]
mod collection_tests {
    use super::*;

    unsafe extern "C" fn i32_eq(a: *const u8, b: *const u8) -> i32 {
        let av = *(a as *const i32);
        let bv = *(b as *const i32);
        if av == bv {
            1
        } else {
            0
        }
    }

    unsafe extern "C" fn i32_cmp(a: *const u8, b: *const u8) -> i32 {
        let av = *(a as *const i32);
        let bv = *(b as *const i32);
        if av < bv {
            -1
        } else if av > bv {
            1
        } else {
            0
        }
    }

    unsafe extern "C" fn u64_hash(a: *const u8) -> u64 {
        let mut x = *(a as *const u64);
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^= x >> 33;
        x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
        x ^ (x >> 33)
    }

    unsafe extern "C" fn u64_eq(a: *const u8, b: *const u8) -> i32 {
        if *(a as *const u64) == *(b as *const u64) {
            1
        } else {
            0
        }
    }

    #[test]
    fn test_array_push_pop_get() {
        unsafe {
            let arr = mp_rt_arr_new(1, std::mem::size_of::<i32>() as u64, 0);

            let v1 = 10_i32;
            let v2 = 30_i32;
            let v3 = 20_i32;
            mp_rt_arr_push(
                arr,
                &v1 as *const i32 as *const u8,
                std::mem::size_of::<i32>() as u64,
            );
            mp_rt_arr_push(
                arr,
                &v2 as *const i32 as *const u8,
                std::mem::size_of::<i32>() as u64,
            );
            mp_rt_arr_push(
                arr,
                &v3 as *const i32 as *const u8,
                std::mem::size_of::<i32>() as u64,
            );

            assert_eq!(mp_rt_arr_len(arr), 3);
            assert_eq!(*(mp_rt_arr_get(arr, 0) as *const i32), 10);
            assert_eq!(*(mp_rt_arr_get(arr, 1) as *const i32), 30);
            assert_eq!(*(mp_rt_arr_get(arr, 2) as *const i32), 20);
            assert_eq!(
                mp_rt_arr_contains(
                    arr,
                    &v2 as *const i32 as *const u8,
                    std::mem::size_of::<i32>() as u64,
                    Some(i32_eq)
                ),
                1
            );

            mp_rt_arr_sort(arr, Some(i32_cmp));
            assert_eq!(*(mp_rt_arr_get(arr, 0) as *const i32), 10);
            assert_eq!(*(mp_rt_arr_get(arr, 1) as *const i32), 20);
            assert_eq!(*(mp_rt_arr_get(arr, 2) as *const i32), 30);

            let mut out = 0_i32;
            assert_eq!(
                mp_rt_arr_pop(
                    arr,
                    &mut out as *mut i32 as *mut u8,
                    std::mem::size_of::<i32>() as u64
                ),
                1
            );
            assert_eq!(out, 30);
            assert_eq!(mp_rt_arr_len(arr), 2);

            mp_rt_release_strong(arr);
        }
    }

    #[test]
    fn test_array_contains_and_sort_with_null_callbacks() {
        unsafe {
            let arr = mp_rt_arr_new(1, std::mem::size_of::<i32>() as u64, 0);
            for value in [3_i32, 1_i32, 2_i32] {
                mp_rt_arr_push(
                    arr,
                    &value as *const i32 as *const u8,
                    std::mem::size_of::<i32>() as u64,
                );
            }

            let needle = 2_i32;
            assert_eq!(
                mp_rt_arr_contains(
                    arr,
                    &needle as *const i32 as *const u8,
                    std::mem::size_of::<i32>() as u64,
                    None,
                ),
                1
            );

            mp_rt_arr_sort(arr, None);
            assert_eq!(*(mp_rt_arr_get(arr, 0) as *const i32), 1);
            assert_eq!(*(mp_rt_arr_get(arr, 1) as *const i32), 2);
            assert_eq!(*(mp_rt_arr_get(arr, 2) as *const i32), 3);

            mp_rt_release_strong(arr);
        }
    }

    #[test]
    fn test_array_sort_null_callback_uses_numeric_order_for_i32() {
        unsafe {
            let arr = mp_rt_arr_new(TYPE_ID_I32, std::mem::size_of::<i32>() as u64, 0);
            for value in [2_i32, -1_i32, 1_i32] {
                mp_rt_arr_push(
                    arr,
                    &value as *const i32 as *const u8,
                    std::mem::size_of::<i32>() as u64,
                );
            }

            mp_rt_arr_sort(arr, None);
            assert_eq!(*(mp_rt_arr_get(arr, 0) as *const i32), -1);
            assert_eq!(*(mp_rt_arr_get(arr, 1) as *const i32), 1);
            assert_eq!(*(mp_rt_arr_get(arr, 2) as *const i32), 2);

            mp_rt_release_strong(arr);
        }
    }

    #[test]
    fn test_array_null_callbacks_use_string_value_semantics() {
        unsafe {
            let ptr_size = std::mem::size_of::<*mut MpRtHeader>() as u64;
            let arr = mp_rt_arr_new(TYPE_ID_STR, ptr_size, 0);

            let alpha = b"alpha";
            let beta = b"beta";
            let alpha_h = mp_rt_str_from_utf8(alpha.as_ptr(), alpha.len() as u64);
            let beta_h = mp_rt_str_from_utf8(beta.as_ptr(), beta.len() as u64);
            let probe_h = mp_rt_str_from_utf8(alpha.as_ptr(), alpha.len() as u64);

            mp_rt_arr_push(
                arr,
                &beta_h as *const *mut MpRtHeader as *const u8,
                ptr_size,
            );
            mp_rt_arr_push(
                arr,
                &alpha_h as *const *mut MpRtHeader as *const u8,
                ptr_size,
            );

            assert_eq!(
                mp_rt_arr_contains(
                    arr,
                    &probe_h as *const *mut MpRtHeader as *const u8,
                    ptr_size,
                    None
                ),
                1
            );

            mp_rt_arr_sort(arr, None);
            let first = *(mp_rt_arr_get(arr, 0) as *const *mut MpRtHeader);
            let second = *(mp_rt_arr_get(arr, 1) as *const *mut MpRtHeader);
            assert_eq!(mp_rt_str_eq(first, alpha_h), 1);
            assert_eq!(mp_rt_str_eq(second, beta_h), 1);

            mp_rt_release_strong(arr);
            mp_rt_release_strong(alpha_h);
            mp_rt_release_strong(beta_h);
            mp_rt_release_strong(probe_h);
        }
    }

    #[test]
    fn test_map_set_get_delete() {
        unsafe {
            let map = mp_rt_map_new(
                1,
                2,
                std::mem::size_of::<u64>() as u64,
                std::mem::size_of::<u64>() as u64,
                4,
                Some(u64_hash),
                Some(u64_eq),
            );

            let k1 = 11_u64;
            let v1 = 101_u64;
            let k2 = 22_u64;
            let v2 = 202_u64;

            mp_rt_map_set(
                map,
                &k1 as *const u64 as *const u8,
                std::mem::size_of::<u64>() as u64,
                &v1 as *const u64 as *const u8,
                std::mem::size_of::<u64>() as u64,
            );
            mp_rt_map_set(
                map,
                &k2 as *const u64 as *const u8,
                std::mem::size_of::<u64>() as u64,
                &v2 as *const u64 as *const u8,
                std::mem::size_of::<u64>() as u64,
            );

            assert_eq!(mp_rt_map_len(map), 2);

            let p1 = mp_rt_map_get(
                map,
                &k1 as *const u64 as *const u8,
                std::mem::size_of::<u64>() as u64,
            );
            assert!(!p1.is_null());
            assert_eq!(*(p1 as *const u64), 101);

            assert_eq!(
                mp_rt_map_contains_key(
                    map,
                    &k2 as *const u64 as *const u8,
                    std::mem::size_of::<u64>() as u64
                ),
                1
            );
            assert_eq!(
                mp_rt_map_delete(
                    map,
                    &k2 as *const u64 as *const u8,
                    std::mem::size_of::<u64>() as u64
                ),
                1
            );
            assert_eq!(
                mp_rt_map_contains_key(
                    map,
                    &k2 as *const u64 as *const u8,
                    std::mem::size_of::<u64>() as u64
                ),
                0
            );
            assert!(mp_rt_map_get(
                map,
                &k2 as *const u64 as *const u8,
                std::mem::size_of::<u64>() as u64
            )
            .is_null());

            let keys = mp_rt_map_keys(map);
            let values = mp_rt_map_values(map);
            assert_eq!(mp_rt_arr_len(keys), 1);
            assert_eq!(mp_rt_arr_len(values), 1);
            assert_eq!(*(mp_rt_arr_get(keys, 0) as *const u64), 11);
            assert_eq!(*(mp_rt_arr_get(values, 0) as *const u64), 101);

            mp_rt_release_strong(keys);
            mp_rt_release_strong(values);
            mp_rt_release_strong(map);
        }
    }

    #[test]
    fn test_map_set_get_delete_with_null_callbacks() {
        unsafe {
            let map = mp_rt_map_new(
                1,
                2,
                std::mem::size_of::<u64>() as u64,
                std::mem::size_of::<u64>() as u64,
                4,
                None,
                None,
            );

            let key = 7_u64;
            let val = 70_u64;
            mp_rt_map_set(
                map,
                &key as *const u64 as *const u8,
                std::mem::size_of::<u64>() as u64,
                &val as *const u64 as *const u8,
                std::mem::size_of::<u64>() as u64,
            );

            let out = mp_rt_map_get(
                map,
                &key as *const u64 as *const u8,
                std::mem::size_of::<u64>() as u64,
            );
            assert!(!out.is_null());
            assert_eq!(*(out as *const u64), 70_u64);
            assert_eq!(
                mp_rt_map_contains_key(
                    map,
                    &key as *const u64 as *const u8,
                    std::mem::size_of::<u64>() as u64
                ),
                1
            );
            assert_eq!(
                mp_rt_map_delete(
                    map,
                    &key as *const u64 as *const u8,
                    std::mem::size_of::<u64>() as u64
                ),
                1
            );
            assert_eq!(mp_rt_map_len(map), 0);

            mp_rt_release_strong(map);
        }
    }

    #[test]
    fn test_map_null_callbacks_use_string_key_value_semantics() {
        unsafe {
            let ptr_size = std::mem::size_of::<*mut MpRtHeader>() as u64;
            let map = mp_rt_map_new(
                TYPE_ID_STR,
                1,
                ptr_size,
                std::mem::size_of::<u64>() as u64,
                4,
                None,
                None,
            );

            let alpha = b"alpha";
            let key_insert = mp_rt_str_from_utf8(alpha.as_ptr(), alpha.len() as u64);
            let key_lookup = mp_rt_str_from_utf8(alpha.as_ptr(), alpha.len() as u64);
            let val = 42_u64;

            mp_rt_map_set(
                map,
                &key_insert as *const *mut MpRtHeader as *const u8,
                ptr_size,
                &val as *const u64 as *const u8,
                std::mem::size_of::<u64>() as u64,
            );

            let out = mp_rt_map_get(
                map,
                &key_lookup as *const *mut MpRtHeader as *const u8,
                ptr_size,
            );
            assert!(!out.is_null());
            assert_eq!(*(out as *const u64), 42_u64);
            assert_eq!(
                mp_rt_map_contains_key(
                    map,
                    &key_lookup as *const *mut MpRtHeader as *const u8,
                    ptr_size
                ),
                1
            );
            assert_eq!(
                mp_rt_map_delete(
                    map,
                    &key_lookup as *const *mut MpRtHeader as *const u8,
                    ptr_size
                ),
                1
            );
            assert_eq!(mp_rt_map_len(map), 0);

            mp_rt_release_strong(key_insert);
            mp_rt_release_strong(key_lookup);
            mp_rt_release_strong(map);
        }
    }
}
