type x7 = uN[0x1];
fn main(x0: s23) -> u5 {
  let x1: s23 = !(x0);
  let x2: bool = bool:true;
  let x3: u5 = ((((x2) ++ (x2)) ++ (x2)) ++ (x2)) ++ (x2);
  let x4: u5 = for (i, x): (u4, u5) in range(u4:0x0, u4:0x3) {
    x
  }(x3);
  let x5: u18 = u18:0x3ffff;
  let x6: x7[0x1] = ((x2) as x7[0x1]);
  let x8: u51 = ((((x5) ++ (x3)) ++ (x4)) ++ (x4)) ++ (x5);
  let x9: u5 = one_hot_sel(x3, [x4, x4, x3, x4, x3]);
  let x10: s23 = (x0) - (((x9) as s23));
  let x11: x7[0x2] = (x6) ++ (x6);
  let x12: u2 = (x4)[0x3+:u2];
  let x13: u5 = rev(x4);
  let x14: u19 = one_hot(x5, u1:0x1);
  let x15: u19 = clz(x14);
  let x16: u1 = (x13)[-0x5:0x1];
  x3
}