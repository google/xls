proc RwProc {
  in_ch: chan<u32> in;
  out_ch: chan<u32> out;
  config(in_ch: chan<u32> in, out_ch: chan<u32> out) { (in_ch, out_ch) }
  init { u1:0 }
  next(state: u1) { state }
}
