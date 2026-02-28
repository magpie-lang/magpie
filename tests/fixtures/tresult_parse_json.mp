module test.tresult_parse_json
exports { @main }
imports { }
digest "d302f64980fa086af3547f18e4fc4358c44cf6a8e7d0f681bc0b2f1453c93e62"

fn @main() -> i32 meta { } {
  bb0:
    %s_num: Str = const.Str "123"
    %sb_num: borrow Str = borrow.shared { v=%s_num }
    %parsed: TResult<i64, Str> = str.parse_i64 { s=%sb_num }

    %value: i32 = const.i32 7
    %encoded: TResult<Str, Str> = json.encode<i32> { v=%value }

    %json_raw: Str = const.Str "42"
    %json_raw_b: borrow Str = borrow.shared { v=%json_raw }
    %decoded: TResult<rawptr<u8>, Str> = json.decode<i32> { s=%json_raw_b }

    ret const.i32 0
}
