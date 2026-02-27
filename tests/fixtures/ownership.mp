module test.ownership
exports { @create_and_share }
imports { std.io::{@println} }
digest "1e4eafe0782cb7e4c723a85292330608d529aa872b02c81b553f4e27b82c1bde"

heap struct TPerson {
  field name: Str
  field age: i32
}

fn @create_and_share() -> shared TPerson meta { } {
  bb0:
    %name: Str = const.Str "Alice"
    %age: i32 = const.i32 30
    %person: TPerson = new TPerson { age=%age, name=%name }
    %shared_p: shared TPerson = share { v=%person }
    %clone_p: shared TPerson = clone.shared { v=%shared_p }
    ret %shared_p
}
