proc CondReader {
  in_r: chan<u32> in;
  config(in_r: chan<u32> in) { (in_r,) }
  init { u1:0 }
  next(state: u1) {
    let (_, _) = recv_if(join(), in_r, true, u32:0);
    state
  }
}
