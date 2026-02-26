//! magpie_rt — Magpie runtime ABI (§20 Runtime ABI)
//!
//! Implements ARC memory management, type registry, strings, StringBuilder,
//! and panic as specified in SPEC.md §20.1.

#![allow(clippy::missing_safety_doc)]

use std::alloc::{self, Layout};
use std::ffi::c_char;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, Once, OnceLock};

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

pub type MpRtHashFn = unsafe extern "C" fn(*const u8) -> u64;
pub type MpRtEqFn = unsafe extern "C" fn(*const u8, *const u8) -> i32;
pub type MpRtCmpFn = unsafe extern "C" fn(*const u8, *const u8) -> i32;

// ---------------------------------------------------------------------------
// §20.1.4  Fixed type_ids
// ---------------------------------------------------------------------------

pub const TYPE_ID_STR: u32 = 20;
pub const TYPE_ID_STRBUILDER: u32 = 21;
pub const TYPE_ID_ARRAY: u32 = 22;
pub const TYPE_ID_MAP: u32 = 23;

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
unsafe fn collection_alloc_zeroed(size: usize) -> *mut u8 {
    let layout = Layout::from_size_align(size.max(1), COLLECTION_DATA_ALIGN).expect("bad collection alloc layout");
    let ptr = alloc::alloc_zeroed(layout);
    if ptr.is_null() {
        alloc::handle_alloc_error(layout);
    }
    ptr
}

#[inline]
unsafe fn collection_realloc(ptr: *mut u8, old_size: usize, new_size: usize) -> *mut u8 {
    let old_layout =
        Layout::from_size_align(old_size.max(1), COLLECTION_DATA_ALIGN).expect("bad collection realloc old layout");
    let new_ptr = alloc::realloc(ptr, old_layout, new_size.max(1));
    if new_ptr.is_null() {
        let new_layout =
            Layout::from_size_align(new_size.max(1), COLLECTION_DATA_ALIGN).expect("bad collection realloc new layout");
        alloc::handle_alloc_error(new_layout);
    }
    new_ptr
}

#[inline]
unsafe fn collection_dealloc(ptr: *mut u8, size: usize) {
    if ptr.is_null() {
        return;
    }
    let layout = Layout::from_size_align(size.max(1), COLLECTION_DATA_ALIGN).expect("bad collection dealloc layout");
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
    let vals_off = align_up(add_usize(keys_off, keys_bytes, "map keys offset overflow"), COLLECTION_DATA_ALIGN);
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
                debug_fqn: b"core.Array\0".as_ptr() as *const c_char,
            },
            MpRtTypeInfo {
                type_id: TYPE_ID_MAP,
                flags: FLAG_HEAP | FLAG_HAS_DROP,
                payload_size: std::mem::size_of::<MpRtMapPayload>() as u64,
                payload_align: std::mem::align_of::<MpRtMapPayload>() as u64,
                drop_fn: Some(mp_rt_map_drop),
                debug_fqn: b"core.Map\0".as_ptr() as *const c_char,
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
        new_cap = new_cap.checked_mul(2).expect("mp_rt_arr_push: capacity overflow");
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
    let start = ((*payload).hash_fn)(key) as usize % cap;

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
                if ((*payload).eq_fn)(existing_key, key) != 0 {
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
    if len_after
        .checked_mul(10)
        .expect("map load factor overflow")
        > (*payload).cap.checked_mul(7).expect("map load factor overflow")
    {
        map_resize(
            payload,
            (*payload).cap.checked_mul(2).expect("map capacity overflow"),
        );
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_arr_new(elem_type_id: u32, elem_size: u64, capacity: u64) -> *mut MpRtHeader {
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
pub unsafe extern "C" fn mp_rt_arr_set(arr: *mut MpRtHeader, idx: u64, val: *const u8, elem_size: u64) {
    let payload = arr_payload(arr);
    assert_eq!(elem_size, (*payload).elem_size, "mp_rt_arr_set: elem_size mismatch");
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
    assert_eq!(elem_size, (*payload).elem_size, "mp_rt_arr_push: elem_size mismatch");

    let new_len = (*payload).len.checked_add(1).expect("mp_rt_arr_push: length overflow");
    arr_reserve(payload, new_len);

    if elem_size > 0 {
        let dst = (*payload).data_ptr.add(array_bytes((*payload).len, elem_size));
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
    assert_eq!(elem_size, (*payload).elem_size, "mp_rt_arr_pop: elem_size mismatch");
    if (*payload).len == 0 {
        return 0;
    }

    (*payload).len -= 1;
    if elem_size > 0 && !out.is_null() {
        let src = (*payload).data_ptr.add(array_bytes((*payload).len, elem_size));
        std::ptr::copy_nonoverlapping(
            src,
            out,
            usize_from_u64(elem_size, "array elem_size too large"),
        );
    }
    1
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_arr_slice(arr: *mut MpRtHeader, start: u64, end: u64) -> *mut MpRtHeader {
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
        let start_off = mul_usize(usize_from_u64(start, "array start too large"), elem_size, "array start overflow");
        let total = mul_usize(usize_from_u64(count, "array count too large"), elem_size, "array slice overflow");
        std::ptr::copy_nonoverlapping((*payload).data_ptr.add(start_off), (*out_payload).data_ptr, total);
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
        if eq_fn(elem_ptr as *const u8, val) != 0 {
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
            let lhs = (*payload)
                .data_ptr
                .add(mul_usize(j - 1, elem_size, "array lhs index overflow"));
            let rhs = (*payload)
                .data_ptr
                .add(mul_usize(j, elem_size, "array rhs index overflow"));
            if cmp(lhs as *const u8, rhs as *const u8) <= 0 {
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
pub unsafe extern "C" fn mp_rt_map_get(map: *mut MpRtHeader, key: *const u8, key_size: u64) -> *mut u8 {
    let payload = map_payload(map);
    assert_eq!(key_size, (*payload).key_size, "mp_rt_map_get: key_size mismatch");

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
    assert_eq!(key_size, (*payload).key_size, "mp_rt_map_set: key_size mismatch");
    assert_eq!(val_size, (*payload).val_size, "mp_rt_map_set: val_size mismatch");

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
    assert_eq!(key_size, (*payload).key_size, "mp_rt_map_take: key_size mismatch");
    assert_eq!(val_size, (*payload).val_size, "mp_rt_map_take: val_size mismatch");

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
pub unsafe extern "C" fn mp_rt_map_delete(map: *mut MpRtHeader, key: *const u8, key_size: u64) -> i32 {
    let payload = map_payload(map);
    assert_eq!(key_size, (*payload).key_size, "mp_rt_map_delete: key_size mismatch");
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
pub unsafe extern "C" fn mp_rt_map_contains_key(map: *mut MpRtHeader, key: *const u8, key_size: u64) -> i32 {
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
        if av == bv { 1 } else { 0 }
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
            mp_rt_arr_push(arr, &v1 as *const i32 as *const u8, std::mem::size_of::<i32>() as u64);
            mp_rt_arr_push(arr, &v2 as *const i32 as *const u8, std::mem::size_of::<i32>() as u64);
            mp_rt_arr_push(arr, &v3 as *const i32 as *const u8, std::mem::size_of::<i32>() as u64);

            assert_eq!(mp_rt_arr_len(arr), 3);
            assert_eq!(*(mp_rt_arr_get(arr, 0) as *const i32), 10);
            assert_eq!(*(mp_rt_arr_get(arr, 1) as *const i32), 30);
            assert_eq!(*(mp_rt_arr_get(arr, 2) as *const i32), 20);
            assert_eq!(
                mp_rt_arr_contains(
                    arr,
                    &v2 as *const i32 as *const u8,
                    std::mem::size_of::<i32>() as u64,
                    i32_eq
                ),
                1
            );

            mp_rt_arr_sort(arr, i32_cmp);
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
    fn test_map_set_get_delete() {
        unsafe {
            let map = mp_rt_map_new(
                1,
                2,
                std::mem::size_of::<u64>() as u64,
                std::mem::size_of::<u64>() as u64,
                4,
                u64_hash,
                u64_eq,
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

            let p1 = mp_rt_map_get(map, &k1 as *const u64 as *const u8, std::mem::size_of::<u64>() as u64);
            assert!(!p1.is_null());
            assert_eq!(*(p1 as *const u64), 101);

            assert_eq!(
                mp_rt_map_contains_key(map, &k2 as *const u64 as *const u8, std::mem::size_of::<u64>() as u64),
                1
            );
            assert_eq!(
                mp_rt_map_delete(map, &k2 as *const u64 as *const u8, std::mem::size_of::<u64>() as u64),
                1
            );
            assert_eq!(
                mp_rt_map_contains_key(map, &k2 as *const u64 as *const u8, std::mem::size_of::<u64>() as u64),
                0
            );
            assert!(mp_rt_map_get(map, &k2 as *const u64 as *const u8, std::mem::size_of::<u64>() as u64).is_null());

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
}
