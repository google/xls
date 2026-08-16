proc NbReader {
  in_r: chan<u32> in;
  config(in_r: chan<u32> in) { (in_r,) }
  init { u1:0 }
  next(state: u1) {
    let (_, _, _) = recv_non_blocking(join(), in_r, u32:0);
    state
  }
}
