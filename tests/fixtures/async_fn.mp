module test.async_example
exports { @fetch_data }
imports { }
digest "0000000000000000"

async fn @fetch_data(%id: u64) -> Str {
bb0:
  %conn: Str = suspend.call @get_connection {}
  %result: Str = call @process { conn=%conn, id=%id }
  ret %result
}
