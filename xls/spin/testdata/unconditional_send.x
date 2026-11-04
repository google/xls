proc Writer {
  out_s: chan<u32> out;
  config(out_s: chan<u32> out) { (out_s,) }
  init { u32:0 }
  next(state: u32) {
    send(join(), out_s, state);
    state
  }
}
