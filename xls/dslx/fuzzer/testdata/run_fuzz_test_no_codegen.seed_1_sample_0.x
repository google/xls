fn main(x0: s23) -> s51 {
    let x1: uN[0x14] = (x0)[:0x14];
    let x2: uN[0x28] = (x1) ++ (x1);
    let x3: s23 = for (i, x): (u4, s23) in range((u4:0x0), (u4:0x6)) {
    x
  }(x0)
  ;
    let x4: s23 = (x3) - (x3);
    let x5: s51 = (s51:0x8000);
    let x6: u9 = (u9:0x80);
    let x7: uN[0x36] = (((((x6) ++ (x6)) ++ (x6)) ++ (x6)) ++ (x6)) ++ (x6);
    let x8: s23 = one_hot_sel((uN[0x6]:0x15), [x4, x0, x3, x4, x4, x0]);
    let x9: uN[0xa] = one_hot(x6, (u1:0x1));
    let x10: s51 = one_hot_sel((uN[0x5]:0x15), [x5, x5, x5, x5, x5]);
    let x11: uN[0x3f] = ((((((x6) ++ (x6)) ++ (x6)) ++ (x6)) ++ (x6)) ++ (x6)) ++ (x6);
    let x12: s23 = ((x5 as s23)) & (x8);
    let x13: s51 = one_hot_sel((uN[0x2]:0x3), [x10, x5]);
    let x14: u9 = for (i, x): (u4, u9) in range((u4:0x0), (u4:0x0)) {
    x
  }(x6)
  ;
    x5
}