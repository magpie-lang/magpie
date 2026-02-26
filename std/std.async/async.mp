module std.async
exports { @block_on, @spawn_task }
imports { }
digest "0000000000000000"

extern "C" module async_rt {
  fn @block_on(%fut: TFuture<unit>) -> unit attrs { link_name="mp_std_block_on" }
  fn @spawn_task(%fut: TFuture<unit>) -> TFuture<unit> attrs { link_name="mp_std_spawn_task" }
}
