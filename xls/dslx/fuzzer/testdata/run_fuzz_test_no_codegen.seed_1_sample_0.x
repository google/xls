type x4 = uN[0xa];
type x8 = uN[0x2];
fn main(x0: s23) -> u60 {
  let x1: u20 = (x0)[:0x14];
  let x2: u60 = ((x1) ++ (x1)) ++ (x1);
  let x3: x4[0x2] = ((x1) as x4[0x2]);
  let x5: u20 = rev(x1);
  let x6: u60 = clz(x2);
  let x7: x8[0xa] = ((x1) as x8[0xa]);
  let x9: u60 = (x6) + (((x5) as u60));
  let x10: u20 = one_hot_sel(u1:0x0, [x5]);
  let x11: u35 = u35:0x10;
  let x12: u15 = (x1)[0x5+:u15];
  let x13: u61 = one_hot(x2, u1:0x1);
  let x14: u60 = for (i, x): (u4, u60) in range(u4:0x0, u4:0x3) {
    x
  }(x6);
  let x15: u1 = or_reduce(x14);
  let x16: u60 = one_hot_sel(x15, [x14]);
  let x17: u1 = xor_reduce(x9);
  let x18: s23 = one_hot_sel(x17, [x0]);
  let x19: u1 = one_hot_sel(x17, [x15]);
  let x20: u60 = for (i, x): (u4, u60) in range(u4:0x0, u4:0x5) {
    x
  }(x6);
  let x21: u61 = one_hot_sel(x19, [x13]);
  let x22: u47 = (x16)[-0x33:-0x4];
  x9
}