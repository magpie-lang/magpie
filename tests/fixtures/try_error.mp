module test.try_error
exports { @safe_divide }
imports { }
digest "0000000000000000"

fn @safe_divide(%a: i32, %b: i32) -> TResult<i32, Str> {
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
