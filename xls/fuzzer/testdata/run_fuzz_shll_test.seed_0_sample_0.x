fn main(x0: u48, x1: s18, x2: u10, x3: u63, x4: u44, x5: u20, x6: u61) -> (u20, u61, u61, u10, u27) {
  let x7: u58 = (x2) ++ (x0);
  let x8: u61 = ctz(x6);
  let x9: u1 = or_reduce(x6);
  let x10: u61 = (((x3) as u61)) << ((u61:0x3a) if ((x8) >= (u61:0x3a)) else (x8));
  let x11: u43 = u43:0x4;
  let x12: u63 = for (i, x): (u4, u63) in range(u4:0x0, u4:0x7) {
    x
  }(x3);
  let x13: u1 = xor_reduce(x12);
  let x14: (u1,) = (x13,);
  let x15: u1 = rev(x13);
  let x16: u63 = (x3)[:];
  let x17: u1 = (x13) << ((u1:0x0) if ((x15) >= (u1:0x0)) else (x15));
  let x18: u27 = (x12)[0x24:];
  let x19: u61 = x6;
  let x20: u1 = and_reduce(x17);
  let x21: u54 = (x2) ++ (x4);
  let x22: u10 = !(x2);
  let x23: u1 = (x9)[:];
  let x24: u61 = for (i, x): (u4, u61) in range(u4:0x0, u4:0x1) {
    x
  }(x8);
  let x25: u1 = and_reduce(x5);
  let x26: u19 = (x6)[-0x13:];
  (x5, x8, x19, x22, x18)
}