module std.math
exports { @abs_i32, @min_i32, @max_i32, @sqrt_f64, @floor_f64, @ceil_f64 }
imports { }
digest "0000000000000000"

extern "C" module math_rt {
  fn @abs_i32(%v: i32) -> i32 attrs { link_name="mp_std_abs_i32" }
  fn @min_i32(%a: i32, %b: i32) -> i32 attrs { link_name="mp_std_min_i32" }
  fn @max_i32(%a: i32, %b: i32) -> i32 attrs { link_name="mp_std_max_i32" }
  fn @sqrt_f64(%v: f64) -> f64 attrs { link_name="mp_std_sqrt_f64" }
  fn @floor_f64(%v: f64) -> f64 attrs { link_name="mp_std_floor_f64" }
  fn @ceil_f64(%v: f64) -> f64 attrs { link_name="mp_std_ceil_f64" }
}
