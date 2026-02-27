module test.try_error
exports { @safe_divide }
imports { }
digest "6f93d8096af8855fc6edd08442802377dbb895d7ba6d500294e1862285c9d5b9"

fn @safe_divide(%a: i32, %b: i32) -> TResult<i32, Str> meta { } {
  bb0:
    %is_zero: bool = icmp.eq { lhs=%b, rhs=const.i32 0 }
    cbr %is_zero bb1 bb2

  bb1:
    %err_msg: Str = const.Str "division by zero"
    %err: TResult<i32, Str> = enum.new<Err> { e=%err_msg }
    ret %err

  bb2:
    %result: i32 = i.sdiv { lhs=%a, rhs=%b }
    %ok: TResult<i32, Str> = enum.new<Ok> { v=%result }
    ret %ok
}
