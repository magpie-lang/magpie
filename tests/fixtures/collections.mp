module test.collections
exports { @test_array, @test_map }
imports { }
digest "0000000000000000"

impl hash for Str = std.hash.@hash_Str
impl eq for Str = std.hash.@eq_Str

fn @test_array() -> i64 {
bb0:
  %arr: Array<i32> = arr.new<i32> { cap=const.i64 10 }
  arr.push { arr=%arr, val=const.i32 42 }
  arr.push { arr=%arr, val=const.i32 99 }
  %len: i64 = arr.len { arr=%arr }
  ret %len
}

fn @test_map() -> bool {
bb0:
  %m: Map<Str, i32> = map.new<Str, i32> { }
  %key: Str = const.Str "hello"
  map.set { map=%m, key=%key, val=const.i32 42 }
  %has: bool = map.contains_key { map=%m, key=%key }
  ret %has
}
