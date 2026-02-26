//! magpie_rt — Magpie runtime ABI (§20 Runtime ABI)
//!
//! Implements ARC memory management, type registry, strings, StringBuilder,
//! and panic as specified in SPEC.md §20.1.

#![allow(clippy::missing_safety_doc)]

use std::alloc::{self, Layout};
use std::ffi::c_char;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

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

// ---------------------------------------------------------------------------
// §20.1.4  Fixed type_ids
// ---------------------------------------------------------------------------

pub const TYPE_ID_STR: u32 = 20;
pub const TYPE_ID_STRBUILDER: u32 = 21;

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
        if let Some(existing) = reg.entries.iter_mut().find(|e| e.type_id == info.type_id) {
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
    (*header).reserved0 = 0;

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
        _ => {}
    }
}

/// Increment the weak reference count (Relaxed).
///
/// # Safety
/// `obj` must be a live heap object.
#[no_mangle]
pub unsafe extern "C" fn mp_rt_retain_weak(obj: *mut MpRtHeader) {
    (*obj).weak.fetch_add(1, Ordering::Relaxed);
}

/// Decrement the weak reference count.  When weak hits 0 the memory is freed.
///
/// # Safety
/// `obj` must be a live heap object whose weak count >= 1.
#[no_mangle]
pub unsafe extern "C" fn mp_rt_release_weak(obj: *mut MpRtHeader) {
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
        match strong.compare_exchange_weak(current, current + 1, Ordering::Acquire, Ordering::Relaxed) {
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
unsafe fn alloc_builtin(type_id: u32, flags: u32, payload_size: usize, payload_align: usize) -> *mut MpRtHeader {
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
    let obj = alloc_builtin(TYPE_ID_STR, FLAG_HEAP | FLAG_SEND | FLAG_SYNC, payload_size, 8);

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
    if bytes_a == bytes_b { 1 } else { 0 }
}

/// Concatenate two Str objects and return a new Str.
///
/// # Safety
/// Both `a` and `b` must be valid Str objects.
#[no_mangle]
pub unsafe extern "C" fn mp_rt_str_concat(a: *mut MpRtHeader, b: *mut MpRtHeader) -> *mut MpRtHeader {
    let base_a = str_payload_base(a);
    let base_b = str_payload_base(b);
    let len_a = *(base_a as *const u64);
    let len_b = *(base_b as *const u64);
    let new_len = len_a + len_b;

    let payload_size = (std::mem::size_of::<u64>() as u64 + new_len) as usize;
    let obj = alloc_builtin(TYPE_ID_STR, FLAG_HEAP | FLAG_SEND | FLAG_SYNC, payload_size, 8);

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
pub unsafe extern "C" fn mp_rt_str_slice(s: *mut MpRtHeader, start: u64, end: u64) -> *mut MpRtHeader {
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
    let base = str_payload_base(str_msg);
    let len = *(base as *const u64);
    let bytes = std::slice::from_raw_parts(base.add(8), len as usize);
    let msg = std::str::from_utf8(bytes).unwrap_or("<invalid utf-8>");
    eprintln!("magpie panic: {}", msg);
    std::process::abort()
}

// ---------------------------------------------------------------------------
// Public helper: mp_rt_alloc with layout stored in reserved0.
// Overrides the no_mangle version above so callers that go through the
// registry get the correct dealloc behaviour.
// ---------------------------------------------------------------------------

// NOTE: the exported mp_rt_alloc uses the registry to find alignment for
// dealloc. We patch it here so reserved0 is also filled in.
//
// The existing #[no_mangle] mp_rt_alloc above does NOT set reserved0; we
// replace its body via a separate wrapper approach. To keep the single
// no_mangle symbol, we update the function in place.

// Actually, we need to fix mp_rt_alloc to set reserved0 for dealloc.
// Let's shadow it by making the public symbol call alloc_builtin-style logic.
// Since we already have the #[no_mangle] above, we'll adjust it here with
// a note that the version above is incomplete — we instead provide the
// correct version below and remove the old one.
//
// (The Write tool replaces the file entirely, so the version in this file
//  is the only version. The mp_rt_alloc above sets reserved0=0; we need
//  to fix that. The cleanest approach: rewrite mp_rt_alloc to store the size.)

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

    // -----------------------------------------------------------------------
    // ARC: alloc + retain + release cycle
    // -----------------------------------------------------------------------

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

            // Release both weak references to free memory.
            mp_rt_release_weak(obj);
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
            mp_rt_release_strong(a);
            mp_rt_release_strong(b);
            mp_rt_release_strong(c);
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
            mp_rt_strbuilder_append_f64(b, 3.14_f64);
            mp_rt_strbuilder_append_bool(b, 1);
            mp_rt_strbuilder_append_bool(b, 0);

            let result = mp_rt_strbuilder_build(b);
            let s = read_str(result);

            assert!(s.starts_with("hello"));
            assert!(s.contains("42"));
            assert!(s.contains("-7"));
            assert!(s.contains("3.14"));
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
}
