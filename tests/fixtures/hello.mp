module hello.main
exports { @main }
imports { std.io::{@println} }
digest "6c3f84f0c3585679a972eecaee7772a8a1e8b5be130c57be074f2e6f12923ee6"

fn @main() -> i32 meta { uses { std.io.@println } } {
  bb0:
    %msg: Str = const.Str "Hello, world!"
    call_void std.io.@println { args=[%msg] }
    ret const.i32 0
}
