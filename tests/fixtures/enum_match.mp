module test.enum_match
exports { @classify }
imports { }
digest "75bab834cb9f9b4c770e47f9c10f6c7b62732e9cca4fadf1e31bcc916d2f700b"

heap enum TShape {
  variant Circle { field radius: f64 }
  variant Rect { field w: f64, field h: f64 }
}

fn @classify(%s: borrow TShape) -> i32 meta { } {
  bb0:
    %tag: i32 = enum.tag { v=%s }
    %is_circle: bool = icmp.eq { lhs=%tag, rhs=const.i32 0 }
    cbr %is_circle bb1 bb2

  bb1:
    ret const.i32 1

  bb2:
    ret const.i32 2
}
