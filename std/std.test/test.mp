module std.test
exports { @assert, @assert_eq, @assert_ne, @fail }
imports { }
digest "0000000000000000"

extern "C" module test_rt {
  fn @assert(%cond: bool, %msg: Str) -> unit attrs { link_name="mp_std_assert" }
  fn @assert_eq(%lhs: i64, %rhs: i64, %msg: Str) -> unit attrs { link_name="mp_std_assert_eq" }
  fn @assert_ne(%lhs: i64, %rhs: i64, %msg: Str) -> unit attrs { link_name="mp_std_assert_ne" }
  fn @fail(%msg: Str) -> unit attrs { link_name="mp_std_fail" }
}
