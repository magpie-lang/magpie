module test.collections
exports { @test_array, @test_map }
imports { }
digest "0000000000000000"

fn @test_array() -> i64 meta { } {
bb0:
  %arr: Array<i32> = arr.new<i32> { cap=const.i64 10 }
  %arr_m: mutborrow Array<i32> = borrow.mut { v=%arr }
  arr.push { arr=%arr_m, val=const.i32 42 }
  arr.push { arr=%arr_m, val=const.i32 99 }
  br bb1

bb1:
  %arr_b: borrow Array<i32> = borrow.shared { v=%arr }
  %len: i64 = arr.len { arr=%arr_b }
  ret %len
}

fn @test_map() -> bool meta { } {
bb0:
  %m: Map<Str, i32> = map.new<Str, i32> { }
  %key: Str = const.Str "hello"
  %m_mut: mutborrow Map<Str, i32> = borrow.mut { v=%m }
  map.set { key=%key, map=%m_mut, val=const.i32 42 }
  br bb1

bb1:
  %key2: Str = const.Str "hello"
  %m_b: borrow Map<Str, i32> = borrow.shared { v=%m }
  %has: bool = map.contains_key { key=%key2, map=%m_b }
  ret %has
}
