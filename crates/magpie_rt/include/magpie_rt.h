#ifndef MAGPIE_RT_H
#define MAGPIE_RT_H

#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MpRtHeader {
  _Atomic uint64_t strong;
  _Atomic uint64_t weak;
  uint32_t type_id;
  uint32_t flags;
  uint64_t reserved0;
} MpRtHeader;

typedef struct MpRtTypeInfo {
  uint32_t type_id;
  uint32_t flags;
  uint64_t payload_size;
  uint64_t payload_align;
  void (*drop_fn)(MpRtHeader* obj);
  const char* debug_fqn;
} MpRtTypeInfo;

typedef uint64_t (*MpRtHashFn)(const uint8_t* x);
typedef int32_t (*MpRtEqFn)(const uint8_t* a, const uint8_t* b);
typedef int32_t (*MpRtCmpFn)(const uint8_t* a, const uint8_t* b);

#define MP_RT_OK 0
#define MP_RT_ERR_INVALID_UTF8 1
#define MP_RT_ERR_INVALID_FORMAT 2
#define MP_RT_ERR_UNSUPPORTED_TYPE 3
#define MP_RT_ERR_NULL_OUT_PTR 4
#define MP_RT_ERR_NULL_INPUT 5

void mp_rt_init(void);
void mp_rt_register_types(const MpRtTypeInfo* infos, uint32_t count);
const MpRtTypeInfo* mp_rt_type_info(uint32_t type_id);

MpRtHeader* mp_rt_alloc(uint32_t type_id, uint64_t payload_size, uint64_t payload_align, uint32_t flags);
void mp_rt_retain_strong(MpRtHeader* obj);
void mp_rt_release_strong(MpRtHeader* obj);
void mp_rt_retain_weak(MpRtHeader* obj);
void mp_rt_release_weak(MpRtHeader* obj);
MpRtHeader* mp_rt_weak_upgrade(MpRtHeader* obj);

MpRtHeader* mp_rt_str_from_utf8(const uint8_t* bytes, uint64_t len);
const uint8_t* mp_rt_str_bytes(MpRtHeader* s, uint64_t* out_len);
MpRtHeader* mp_rt_str_concat(MpRtHeader* a, MpRtHeader* b);
uint64_t mp_rt_str_len(MpRtHeader* s);
int32_t mp_rt_str_eq(MpRtHeader* a, MpRtHeader* b);
int32_t mp_rt_str_cmp(MpRtHeader* a, MpRtHeader* b);
MpRtHeader* mp_rt_str_slice(MpRtHeader* s, uint64_t start, uint64_t end);

MpRtHeader* mp_rt_strbuilder_new(void);
void mp_rt_strbuilder_append_str(MpRtHeader* b, MpRtHeader* s);
void mp_rt_strbuilder_append_i64(MpRtHeader* b, int64_t v);
void mp_rt_strbuilder_append_i32(MpRtHeader* b, int32_t v);
void mp_rt_strbuilder_append_f64(MpRtHeader* b, double v);
void mp_rt_strbuilder_append_bool(MpRtHeader* b, int32_t v);
MpRtHeader* mp_rt_strbuilder_build(MpRtHeader* b);

void mp_rt_panic(MpRtHeader* msg);

void mp_std_println(MpRtHeader* s);
void mp_std_eprintln(const uint8_t* s, size_t len);
uint8_t* mp_std_readln(void);
void mp_std_assert(int32_t cond, MpRtHeader* msg);
void mp_std_assert_eq(int64_t a, int64_t b, MpRtHeader* msg);
void mp_std_assert_ne(int64_t a, int64_t b, MpRtHeader* msg);
void mp_std_fail(MpRtHeader* msg);
void mp_std_exit(int32_t code);
MpRtHeader* mp_std_cwd(void);
MpRtHeader* mp_std_env_var(MpRtHeader* name);
MpRtHeader* mp_std_args(void);
void mp_std_println_bytes(const uint8_t* s, size_t len);
void mp_std_assert_bytes(int32_t cond, const uint8_t* msg, size_t msg_len);
uint64_t mp_std_hash_str(MpRtHeader* s);
uint64_t mp_std_hash_Str(MpRtHeader* s);
uint64_t mp_std_hash_i32(int32_t v);
uint64_t mp_std_hash_i64(int64_t v);
void mp_std_block_on(uint8_t* fut);
uint8_t* mp_std_spawn_task(uint8_t* fut);
int32_t mp_std_abs_i32(int32_t v);
int64_t mp_std_abs_i64(int64_t v);
double mp_std_sqrt_f64(double v);
int32_t mp_std_min_i32(int32_t a, int32_t b);
int32_t mp_std_max_i32(int32_t a, int32_t b);

MpRtHeader* mp_rt_arr_new(uint32_t elem_type_id, uint64_t elem_size, uint64_t capacity);
uint64_t mp_rt_arr_len(MpRtHeader* arr);
uint8_t* mp_rt_arr_get(MpRtHeader* arr, uint64_t idx);
void mp_rt_arr_set(MpRtHeader* arr, uint64_t idx, const uint8_t* val, uint64_t elem_size);
void mp_rt_arr_push(MpRtHeader* arr, const uint8_t* val, uint64_t elem_size);
int32_t mp_rt_arr_pop(MpRtHeader* arr, uint8_t* out, uint64_t elem_size);
MpRtHeader* mp_rt_arr_slice(MpRtHeader* arr, uint64_t start, uint64_t end);
int32_t mp_rt_arr_contains(MpRtHeader* arr, const uint8_t* val, uint64_t elem_size, MpRtEqFn eq_fn);
void mp_rt_arr_sort(MpRtHeader* arr, MpRtCmpFn cmp_fn);
void mp_rt_arr_foreach(MpRtHeader* arr, uint8_t* callable);
MpRtHeader* mp_rt_arr_map(MpRtHeader* arr, uint8_t* callable, uint32_t out_elem_type_id, uint64_t out_elem_size);
MpRtHeader* mp_rt_arr_filter(MpRtHeader* arr, uint8_t* callable);
void mp_rt_arr_reduce(MpRtHeader* arr, uint8_t* acc_inout, uint64_t acc_size, uint8_t* callable);

uint8_t* mp_rt_callable_new(uint8_t* fn_ptr, uint8_t* captures_ptr);
uint8_t* mp_rt_callable_fn_ptr(uint8_t* callable);
uint64_t mp_rt_callable_capture_size(uint8_t* callable);
uint8_t* mp_rt_callable_data_ptr(uint8_t* callable);

MpRtHeader* mp_rt_map_new(
  uint32_t key_type_id,
  uint32_t val_type_id,
  uint64_t key_size,
  uint64_t val_size,
  uint64_t capacity,
  MpRtHashFn hash_fn,
  MpRtEqFn eq_fn
);
uint64_t mp_rt_map_len(MpRtHeader* map);
uint8_t* mp_rt_map_get(MpRtHeader* map, const uint8_t* key, uint64_t key_size);
void mp_rt_map_set(MpRtHeader* map, const uint8_t* key, uint64_t key_size, const uint8_t* val, uint64_t val_size);
int32_t mp_rt_map_take(MpRtHeader* map, const uint8_t* key, uint64_t key_size, uint8_t* out_val, uint64_t val_size);
int32_t mp_rt_map_delete(MpRtHeader* map, const uint8_t* key, uint64_t key_size);
int32_t mp_rt_map_contains_key(MpRtHeader* map, const uint8_t* key, uint64_t key_size);
MpRtHeader* mp_rt_map_keys(MpRtHeader* map);
MpRtHeader* mp_rt_map_values(MpRtHeader* map);

uint8_t* mp_rt_mutex_new(void);
void mp_rt_mutex_lock(uint8_t* m);
void mp_rt_mutex_unlock(uint8_t* m);
uint8_t* mp_rt_rwlock_new(void);
void mp_rt_rwlock_read(uint8_t* rw);
void mp_rt_rwlock_write(uint8_t* rw);
void mp_rt_rwlock_unlock(uint8_t* rw);
uint8_t* mp_rt_cell_new(uint8_t* init);
uint8_t* mp_rt_cell_get(uint8_t* cell);
void mp_rt_cell_set(uint8_t* cell, uint8_t* val);
int32_t mp_rt_future_poll(uint8_t* future);
void mp_rt_future_take(uint8_t* future, uint8_t* out_result);

/*
 * DEPRECATED (compatibility shim): use mp_rt_str_try_parse_* instead.
 * Legacy parse wrappers preserve historical fatal-on-error behavior.
 */
int64_t mp_rt_str_parse_i64(MpRtHeader* s);
uint64_t mp_rt_str_parse_u64(MpRtHeader* s);
double mp_rt_str_parse_f64(MpRtHeader* s);
int32_t mp_rt_str_parse_bool(MpRtHeader* s);
/*
 * Preferred fallible parse API:
 * - Returns MP_RT_OK on success and writes to `out`.
 * - On error, returns nonzero status and optionally writes owned Str to `out_errmsg`.
 * - Caller releases `*out_errmsg` via mp_rt_release_strong when non-NULL.
 */
int32_t mp_rt_str_try_parse_i64(MpRtHeader* s, int64_t* out, MpRtHeader** out_errmsg);
int32_t mp_rt_str_try_parse_u64(MpRtHeader* s, uint64_t* out, MpRtHeader** out_errmsg);
int32_t mp_rt_str_try_parse_f64(MpRtHeader* s, double* out, MpRtHeader** out_errmsg);
int32_t mp_rt_str_try_parse_bool(MpRtHeader* s, int32_t* out, MpRtHeader** out_errmsg);
/*
 * DEPRECATED (compatibility shim): use mp_rt_json_try_encode/decode instead.
 * Legacy JSON wrappers preserve historical fatal-on-error behavior.
 */
MpRtHeader* mp_rt_json_encode(uint8_t* obj, uint32_t type_id);
uint8_t* mp_rt_json_decode(MpRtHeader* json_str, uint32_t type_id);
/*
 * Preferred fallible JSON API:
 * - Returns MP_RT_OK on success and writes to `out_str` / `out_val`.
 * - On error, returns nonzero status and optionally writes owned Str to `out_errmsg`.
 * - Caller releases `*out_errmsg` via mp_rt_release_strong when non-NULL.
 */
int32_t mp_rt_json_try_encode(uint8_t* obj, uint32_t type_id, MpRtHeader** out_str, MpRtHeader** out_errmsg);
int32_t mp_rt_json_try_decode(MpRtHeader* json_str, uint32_t type_id, uint8_t** out_val, MpRtHeader** out_errmsg);

MpRtHeader* mp_rt_channel_new(uint32_t elem_type_id, uint64_t elem_size);
void mp_rt_channel_send(MpRtHeader* sender, const uint8_t* val, uint64_t elem_size);
int32_t mp_rt_channel_recv(MpRtHeader* receiver, uint8_t* out, uint64_t elem_size);

int32_t mp_rt_web_serve(
  MpRtHeader* svc,
  MpRtHeader* addr,
  uint16_t port,
  uint8_t keep_alive,
  uint32_t threads,
  uint64_t max_body_bytes,
  uint64_t read_timeout_ms,
  uint64_t write_timeout_ms,
  uint8_t log_requests,
  MpRtHeader** out_errmsg
);

typedef struct MpRtGpuKernelEntry {
  uint64_t sid_hash;
  uint32_t num_buffers;
  uint32_t push_const_size;
} MpRtGpuKernelEntry;

uint32_t mp_rt_gpu_device_count(void);
void mp_rt_gpu_register_kernels(const uint8_t* entries, int32_t count);
int32_t mp_rt_gpu_device_default(MpRtHeader** out_dev, MpRtHeader** out_errmsg);
int32_t mp_rt_gpu_device_by_index(uint32_t idx, MpRtHeader** out_dev, MpRtHeader** out_errmsg);
MpRtHeader* mp_rt_gpu_device_name(MpRtHeader* dev);
int32_t mp_rt_gpu_buffer_new(
  MpRtHeader* dev,
  uint32_t elem_type_id,
  uint64_t elem_size,
  uint64_t len,
  uint32_t usage_flags,
  MpRtHeader** out_buf,
  MpRtHeader** out_errmsg
);
int32_t mp_rt_gpu_buffer_from_array(
  MpRtHeader* dev,
  MpRtHeader* host_arr,
  uint32_t usage_flags,
  MpRtHeader** out_buf,
  MpRtHeader** out_errmsg
);
int32_t mp_rt_gpu_buffer_to_array(MpRtHeader* buf, MpRtHeader** out_arr, MpRtHeader** out_errmsg);
uint64_t mp_rt_gpu_buffer_len(MpRtHeader* buf);
int32_t mp_rt_gpu_buffer_copy(MpRtHeader* src, MpRtHeader* dst, MpRtHeader** out_errmsg);
int32_t mp_rt_gpu_device_sync(MpRtHeader* dev, MpRtHeader** out_errmsg);
int32_t mp_rt_gpu_launch_sync(
  MpRtHeader* dev,
  uint64_t kernel_sid_hash,
  uint32_t grid_x,
  uint32_t grid_y,
  uint32_t grid_z,
  uint32_t block_x,
  uint32_t block_y,
  uint32_t block_z,
  const uint8_t* args_blob,
  uint64_t args_len,
  MpRtHeader** out_errmsg
);
int32_t mp_rt_gpu_launch_async(
  MpRtHeader* dev,
  uint64_t kernel_sid_hash,
  uint32_t grid_x,
  uint32_t grid_y,
  uint32_t grid_z,
  uint32_t block_x,
  uint32_t block_y,
  uint32_t block_z,
  const uint8_t* args_blob,
  uint64_t args_len,
  MpRtHeader** out_fence,
  MpRtHeader** out_errmsg
);
int32_t mp_rt_gpu_fence_wait(
  MpRtHeader* fence,
  uint64_t timeout_ms,
  uint8_t* out_done,
  MpRtHeader** out_errmsg
);
void mp_rt_gpu_fence_free(MpRtHeader* fence);
uint8_t* mp_rt_gpu_device_open(int32_t idx);
void mp_rt_gpu_device_close(uint8_t* dev);
uint8_t* mp_rt_gpu_buffer_alloc(uint8_t* dev, uint64_t size);
void mp_rt_gpu_buffer_free(uint8_t* buf);
int32_t mp_rt_gpu_buffer_write(uint8_t* buf, uint64_t offset, const uint8_t* data, uint64_t len);
int32_t mp_rt_gpu_buffer_read(uint8_t* buf, uint64_t offset, uint8_t* out, uint64_t len);
uint8_t* mp_rt_gpu_kernel_load(uint8_t* dev, const uint8_t* spv, uint64_t spv_len);
void mp_rt_gpu_kernel_free(uint8_t* kernel);
int32_t mp_rt_gpu_launch(
  uint8_t* kernel,
  uint32_t groups_x,
  uint32_t groups_y,
  uint32_t groups_z,
  uint8_t* const* args,
  uint32_t arg_count
);

uint64_t mp_rt_bytes_hash(const uint8_t* data, uint64_t len);
int32_t mp_rt_bytes_eq(const uint8_t* a, const uint8_t* b, uint64_t len);
int32_t mp_rt_bytes_cmp(const uint8_t* a, const uint8_t* b, uint64_t len);

#ifdef __cplusplus
}
#endif

#endif /* MAGPIE_RT_H */
