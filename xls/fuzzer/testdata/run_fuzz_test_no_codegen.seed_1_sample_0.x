const W2_V2 = u2:0x2;
type x4 = u10;
fn main(x0: s23) -> u1 {
  let x1: u20 = (x0)[:0x14];
  let x2: u60 = ((x1) ++ (x1)) ++ (x1);
  let x3: x4[W2_V2] = ((x1) as x4[W2_V2]);
  let x5: x4[0x4] = (x3) ++ (x3);
  let x6: u60 = !(x2);
  let x7: u1 = xor_reduce(x1);
  let x8: u49 = u49:0x200000;
  let x9: u1 = (((x6) as u1)) & (x7);
  let x10: u49 = !(x8);
  let x11: u1 = (x7)[:];
  let x12: u49 = for (i, x): (u4, u49) in range(u4:0x0, u4:0x6) {
    x
  }(x10);
  let x13: u49 = for (i, x): (u4, u49) in range(u4:0x0, u4:0x3) {
    x
  }(x8);
  let x14: u1 = or_reduce(x2);
  let x15: u1 = one_hot_sel(x11, [x14]);
  let x16: u1 = xor_reduce(x9);
  let x17: s23 = one_hot_sel(x16, [x0]);
  let x18: u1 = one_hot_sel(x7, [x14]);
  let x19: u1 = (x18) << ((u1:0x0) if ((((x8) as u1)) >= (u1:0x0)) else (((x8) as u1)));
  let x20: u1 = (x16)[-0x2:0x1];
  x14
}