module std.io
exports { @println, @readln }
imports { }
digest "51cfbf618408a9a12ec7c0c116bb084367922f48c307a4dd17728c62368cc48f"

extern "C" module io_rt {
  fn @println(%s: Str) -> unit attrs { link_name="mp_std_println" }
  fn @readln() -> Str attrs { link_name="mp_std_readln" }
}
