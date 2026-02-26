module test.ownership
exports { @create_and_share }
imports { std.io::{@println} }
digest "0000000000000000"

heap struct TPerson {
  field name: Str
  field age: i32
}

fn @create_and_share() -> shared TPerson {
bb0:
  %name: Str = const.Str "Alice"
  %age: i32 = const.i32 30
  %person: TPerson = new TPerson { name=%name, age=%age }
  %shared_p: shared TPerson = share { v=%person }
  %clone_p: shared TPerson = clone.shared { v=%shared_p }
  ret %shared_p
}
