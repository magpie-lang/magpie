module hello.main
exports { @main }
imports { std.io::{@println} }
digest "0000000000000000"

fn @main() -> i32 {
bb0:
  %msg: Str = const.Str "Hello, world!"
  call_void std.io.@println { args=[%msg] }
  ret const.i32 0
}
