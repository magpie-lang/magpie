module std.hash
exports { @hash_Str, @hash_i32, @hash_i64, @eq_Str }
imports { }
digest "0000000000000000"

extern "C" module hash_rt {
  fn @hash_Str(%v: Str) -> u64 attrs { link_name="mp_std_hash_Str" }
  fn @hash_i32(%v: i32) -> u64 attrs { link_name="mp_std_hash_i32" }
  fn @hash_i64(%v: i64) -> u64 attrs { link_name="mp_std_hash_i64" }
  fn @eq_Str(%a: Str, %b: Str) -> bool attrs { link_name="mp_std_eq_Str" }
}
