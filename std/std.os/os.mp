module std.os
exports { @args, @cwd, @env_var, @exit }
imports { }
digest "6883f4fe38ccb83b8113178264694d5a32209c1553838e614d04f80c043fc2bf"

extern "C" module os_rt {
  fn @env_var(%name: Str) -> TOption<Str> attrs { link_name="mp_std_env_var" }
  fn @args() -> Array<Str> attrs { link_name="mp_std_args" }
  fn @exit(%code: i32) -> unit attrs { link_name="mp_std_exit" }
  fn @cwd() -> Str attrs { link_name="mp_std_cwd" }
}
