proc CondWriter {
  out_s: chan<u32> out;
  config(out_s: chan<u32> out) { (out_s,) }
  init { u32:0 }
  next(state: u32) {
    send_if(join(), out_s, true, state);
    state
  }
}
