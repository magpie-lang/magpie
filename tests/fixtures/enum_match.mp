module test.enum_match
exports { @classify }
imports { }
digest "0000000000000000"

heap enum TShape {
  variant Circle { field radius: f64 }
  variant Rect { field w: f64, field h: f64 }
}

fn @classify(%s: borrow TShape) -> i32 {
bb0:
  %tag: i32 = enum.tag { v=%s }
  %is_circle: bool = icmp.eq { lhs=%tag, rhs=const.i32 0 }
  cbr %is_circle bb1 bb2

bb1:
  ret const.i32 1

bb2:
  ret const.i32 2
}
