module std.io
exports { @println, @readln }
imports { }
digest "0000000000000000"

extern "C" module io_rt {
  fn @println(%s: Str) -> unit attrs { link_name="mp_std_println" }
  fn @readln() -> Str attrs { link_name="mp_std_readln" }
}
