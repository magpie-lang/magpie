module test.feature_harness
exports { @main }
imports { }
digest "c31c74a8044837d869403ace3f2c19c22d07be668b6772fdeb20536e85712d62"

heap struct TPoint {
  field x: i64
  field y: i64
}

fn @sum_add(%a: i64, %b: i64) -> i64 meta { } {
  bb0:
    %sum: i64 = i.add { lhs=%a, rhs=%b }
    ret %sum
}

fn @sum_checked(%a: i64, %b: i64) -> TOption<i64> meta { } {
  bb0:
    %sum: TOption<i64> = i.add.checked { lhs=%a, rhs=%b }
    ret %sum
}

fn @exercise_struct_fields() -> i64 meta { } {
  bb0:
    %x0: i64 = const.i64 10
    %y0: i64 = const.i64 20
    %p: TPoint = new TPoint { x=%x0, y=%y0 }
    %pm: mutborrow TPoint = borrow.mut { v=%p }
    setfield { obj=%pm, field=y, val=const.i64 22 }
    br bb1

  bb1:
    %pb: borrow TPoint = borrow.shared { v=%p }
    %x1: i64 = getfield { obj=%pb, field=x }
    %y1: i64 = getfield { obj=%pb, field=y }
    %s: i64 = i.add { lhs=%x1, rhs=%y1 }
    ret %s
}

fn @exercise_share_clone() -> i64 meta { } {
  bb0:
    %x0: i64 = const.i64 1
    %y0: i64 = const.i64 2
    %p: TPoint = new TPoint { x=%x0, y=%y0 }
    %sp: shared TPoint = share { v=%p }
    %cp: shared TPoint = clone.shared { v=%sp }
    ret const.i64 1
}

fn @exercise_strings() -> i64 meta { } {
  bb0:
    %s: Str = const.Str "abcdef"
    %sb: borrow Str = borrow.shared { v=%s }
    %len: i64 = str.len { s=%sb }
    %nstr: Str = const.Str "123"
    %nb: borrow Str = borrow.shared { v=%nstr }
    %n: i64 = str.parse_i64 { s=%nb }
    %out: i64 = i.add { lhs=%len, rhs=%n }
    ret %out
}

fn @exercise_collections() -> i64 meta { } {
  bb0:
    %arr: Array<i64> = arr.new<i64> { cap=const.i64 8 }
    %m: Map<i64, i64> = map.new<i64, i64> { }
    arr.push { arr=%arr, val=const.i64 3 }
    arr.push { arr=%arr, val=const.i64 1 }
    arr.push { arr=%arr, val=const.i64 2 }
    arr.sort { arr=%arr }
    map.set { map=%m, key=const.i64 1, val=const.i64 7 }
    br bb1

  bb1:
    %arr_b: borrow Array<i64> = borrow.shared { v=%arr }
    %contains: bool = arr.contains { arr=%arr_b, val=const.i64 2 }
    %len: i64 = arr.len { arr=%arr_b }
    %m_b: borrow Map<i64, i64> = borrow.shared { v=%m }
    %has: bool = map.contains_key { map=%m_b, key=const.i64 1 }
    %mlen: i64 = map.len { map=%m_b }
    %r0: i64 = i.add { lhs=%len, rhs=%mlen }
    ret %r0
}

fn @main() -> i64 meta { uses { @exercise_collections, @exercise_share_clone, @exercise_strings, @exercise_struct_fields, @sum_add, @sum_checked } } {
  bb0:
    %sum: i64 = call @sum_add { a=const.i64 10, b=const.i64 20 }
    %checked: TOption<i64> = call @sum_checked { a=const.i64 4, b=const.i64 5 }
    %a: i64 = call @exercise_struct_fields { }
    %b: i64 = call @exercise_share_clone { }
    %c: i64 = call @exercise_collections { }
    %d: i64 = call @exercise_strings { }
    %t0: i64 = i.add { lhs=%sum, rhs=%a }
    %t1: i64 = i.add { lhs=%t0, rhs=%b }
    %t2: i64 = i.add { lhs=%t1, rhs=%c }
    %t3: i64 = i.add { lhs=%t2, rhs=%d }
    ret %t3
}
