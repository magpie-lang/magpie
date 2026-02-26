module std.os
exports { @env_var, @args, @exit, @cwd }
imports { }
digest "0000000000000000"

extern "C" module os_rt {
  fn @env_var(%name: Str) -> TOption<Str> attrs { link_name="mp_std_env_var" }
  fn @args() -> Array<Str> attrs { link_name="mp_std_args" }
  fn @exit(%code: i32) -> unit attrs { link_name="mp_std_exit" }
  fn @cwd() -> Str attrs { link_name="mp_std_cwd" }
}
