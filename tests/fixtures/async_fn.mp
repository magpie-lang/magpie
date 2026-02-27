module test.async_example
exports { @fetch_data }
imports { }
digest "0000000000000000"

fn @get_connection() -> Str meta { } {
bb0:
  %s: Str = const.Str "conn:1"
  ret %s
}

fn @process(%conn: Str, %id: u64) -> Str meta { } {
bb0:
  ret %conn
}

async fn @fetch_data(%id: u64) -> Str meta { uses { @get_connection, @process } } {
bb0:
  %conn: Str = suspend.call @get_connection { }
  %result: Str = call @process { conn=%conn, id=%id }
  ret %result
}
