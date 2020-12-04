type x4 = u10;
type x8 = u2;
fn main(x0: s23) -> u1 {
  let x1: u20 = (x0)[:0x14];
  let x2: u60 = ((x1) ++ (x1)) ++ (x1);
  let x3: x4[0x2] = ((x1) as x4[0x2]);
  let x5: u20 = rev(x1);
  let x6: u60 = clz(x2);
  let x7: x8[0xa] = ((x1) as x8[0xa]);
  let x9: u60 = (x6) << (((x5) as u60));
  let x10: u60 = for (i, x): (u4, u60) in range(u4:0x0, u4:0x3) {
    x
  }(x9);
  let x11: (u60, u60, u60, s23, u60, u60) = (x6, x6, x10, x0, x2, x6);
  let x12: u20 = rev(x5);
  let x13: u60 = for (i, x): (u4, u60) in range(u4:0x0, u4:0x6) {
    x
  }(x10);
  let x14: u60 = for (i, x): (u4, u60) in range(u4:0x0, u4:0x3) {
    x
  }(x6);
  let x15: u1 = or_reduce(x2);
  let x16: u1 = one_hot_sel(x15, [x15]);
  let x17: u1 = and_reduce(x5);
  let x18: s23 = -(x0);
  let x19: u23 = (((x15) ++ (x16)) ++ (x15)) ++ (x1);
  let x20: s23 = one_hot_sel(x16, [x18]);
  let x21: u1 = xor_reduce(x9);
  let x22: x4[0x4] = (x3) ++ (x3);
  let x23: u60 = (x14) << ((u60:0x21) if ((x2) >= (u60:0x21)) else (x2));
  let x24: u8 = u8:0x20;
  x17
}