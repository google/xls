type x7 = uN[0x1];fn main(x0: s23) -> uN[0x1] {
    let x1: s23 = !(x0);
    let x2: bool = (bool:0x1);
    let x3: uN[0x5] = ((((x2) ++ (x2)) ++ (x2)) ++ (x2)) ++ (x2);
    let x4: s23 = for (i, x): (u4, s23) in range((u4:0x0), (u4:0x3)) {
    x
  }(x1)
  ;
    let x5: u18 = (u18:0x3ffff);
    let x6: x7[0x1] = (x2 as x7[0x1]);
    let x8: uN[0x37] = (((x5) ++ (x2)) ++ (x5)) ++ (x5);
    let x9: u18 = one_hot_sel(x2, [x5]);
    let x10: s29 = (s29:0x20);
    let x11: (u18) = (x5,);
    let x12: u18 = clz(x5);
    let x13: u18 = for (i, x): (u4, u18) in range((u4:0x0), (u4:0x3)) {
    x
  }(x9)
  ;
    let x14: uN[0x1] = (x2)[0x0+:uN[0x1]];
    let x15: uN[0x3] = (x9)[0x8:0xb];
    let x16: (u18, bool) = (x13, x2);
    let x17: u18 = one_hot_sel(x2, [x5]);
    let x18: s23 = one_hot_sel(x2, [x1]);
    let x19: uN[0x5] = (x13)[-0xe:0x9];
    x14
}