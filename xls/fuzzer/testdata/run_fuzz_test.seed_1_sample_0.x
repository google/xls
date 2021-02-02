type x7 = u1;
fn main(x0: s23) -> u1 {
  let x1: s23 = !(x0);
  let x2: bool = bool:true;
  let x3: u5 = ((((x2) ++ (x2)) ++ (x2)) ++ (x2)) ++ (x2);
  let x4: u5 = for (i, x): (u4, u5) in range(u4:0x0, u4:0x3) {
    x
  }(x3);
  let x5: u18 = u18:0x3ffff;
  let x6: x7[0x1] = ((x2) as x7[0x1]);
  let x8: u9 = u9:0x80;
  let x9: (u5, u9, u5, u5, u5, u9, s23, u5) = (x3, x8, x4, x3, x4, x8, x0, x3);
  let x10: s23 = -(x1);
  let x11: s23 = one_hot_sel(x3, [x1, x1, x10, x10, x0]);
  let x12: s23 = (x0) - (((x5) as s23));
  let x13: x7[0x2] = (x6) ++ (x6);
  let x14: u1 = (x2)[0x0+:u1];
  let x15: u1 = (x4)[0x0:0x1];
  let x16: (u1, u1) = (x14, x15);
  let x17: u1 = one_hot_sel(x4, [x14, x14, x14, x14, x14]);
  let x18: s23 = for (i, x): (u4, s23) in range(u4:0x0, u4:0x4) {
    x
  }(x0);
  let x19: u9 = for (i, x): (u4, u9) in range(u4:0x0, u4:0x0) {
    x
  }(x8);
  x14
}