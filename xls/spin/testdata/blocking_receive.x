proc Reader {
  in_r: chan<u32> in;
  config(in_r: chan<u32> in) { (in_r,) }
  init { u1:0 }
  next(state: u1) {
    let (_, _) = recv(join(), in_r);
    state
  }
}
