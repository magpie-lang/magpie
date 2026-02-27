module std.async
exports { @block_on, @spawn_task }
imports { }
digest "607060ada9a3b5570cbb4786b55a0742c501bb10aebd3393ecfff5d813114872"

extern "C" module async_rt {
  fn @block_on(%fut: TFuture<unit>) -> unit attrs { link_name="mp_std_block_on" }
  fn @spawn_task(%fut: TFuture<unit>) -> TFuture<unit> attrs { link_name="mp_std_spawn_task" }
}
