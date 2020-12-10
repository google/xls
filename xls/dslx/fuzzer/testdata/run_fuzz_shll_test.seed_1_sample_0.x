const W2_V2 = u2:0x2;
type x4 = u10;
type x13 = u1;
type x28 = u1;
fn main(x0: s23) -> (u1, u20, x28[0x1], u60, s23, u1, s23, u10, u62, x13[0x2], u1) {
  let x1: u20 = (x0)[:0x14];
  let x2: u60 = ((x1) ++ (x1)) ++ (x1);
  let x3: x4[W2_V2] = ((x1) as x4[W2_V2]);
  let x5: x4[0x4] = (x3) ++ (x3);
  let x6: u60 = !(x2);
  let x7: u1 = xor_reduce(x1);
  let x8: u49 = u49:0x200000;
  let x9: u1 = (((x6) as u1)) << ((u1:0x0) if ((x7) >= (u1:0x0)) else (x7));
  let x10: s23 = one_hot_sel(x9, [x0]);
  let x11: u10 = (x8)[0x0+:u10];
  let x12: x13[0x1] = ((x7) as x13[0x1]);
  let x14: u1 = for (i, x): (u4, u1) in range(u4:0x0, u4:0x3) {
    x
  }(x9);
  let x15: u1 = or_reduce(x7);
  let x16: u60 = one_hot_sel(x14, [x2]);
  let x17: u1 = and_reduce(x8);
  let x18: s23 = -(x0);
  let x19: u62 = ((x17) ++ (x2)) ++ (x17);
  let x20: u62 = one_hot_sel(x7, [x19]);
  let x21: u1 = xor_reduce(x9);
  let x22: x13[0x2] = (x12) ++ (x12);
  let x23: u1 = (((x15) as u62)) != (x20);
  let x24: u19 = u19:0x200;
  let x25: (u60,) = (x16,);
  let x26: u27 = (x16)[x1+:u27];
  let x27: x28[0x1] = ((x7) as x28[0x1]);
  (x17, x1, x27, x6, x10, x14, x10, x11, x19, x22, x14)
}