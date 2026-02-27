module std.test
exports { @assert, @assert_eq, @assert_ne, @fail }
imports { }
digest "90004bb11e1f57a29cf4d5fb3c61e26d2df3f172c077e7e2efa5fc286e4ef7ba"

extern "C" module test_rt {
  fn @assert(%cond: bool, %msg: Str) -> unit attrs { link_name="mp_std_assert" }
  fn @assert_eq(%lhs: i64, %rhs: i64, %msg: Str) -> unit attrs { link_name="mp_std_assert_eq" }
  fn @assert_ne(%lhs: i64, %rhs: i64, %msg: Str) -> unit attrs { link_name="mp_std_assert_ne" }
  fn @fail(%msg: Str) -> unit attrs { link_name="mp_std_fail" }
}
