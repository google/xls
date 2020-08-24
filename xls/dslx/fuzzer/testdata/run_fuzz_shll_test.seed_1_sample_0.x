type x20 = uN[0x1];fn main(x0: s23) -> x20[0x2f] {
    let x1: uN[0x14] = (x0)[:0x14];
    let x2: uN[0x28] = (x1) ++ (x1);
    let x3: s23 = for (i, x): (u4, s23) in range((u4:0x0), (u4:0x6)) {
    x
  }(x0)
  ;
    let x4: s23 = (x3) << (x3);
    let x5: s23 = -(x4);
    let x6: u47 = (u47:0x200);
    let x7: u37 = (u37:0x1000000000);
    let x8: uN[0x2f] = (x6)[0x0+:uN[0x2f]];
    let x9: u47 = (x6) << (((u47:0xe)) if ((x6) >= ((u47:0xe))) else (x6));
    let x10: uN[0x2f] = x9;
    let x11: s23 = one_hot_sel((uN[0x4]:0x1), [x0, x5, x4, x3]);
    let x12: s23 = -(x0);
    let x13: uN[0x5e] = (x10) ++ (x10);
    let x14: uN[0x6] = (x4)[0xf:-0x2];
    let x15: uN[0x5e] = (x8) ++ (x8);
    let x16: u8 = (u8:0x20);
    let x17: s23 = (x3) << (x11);
    let x18: uN[0x2f] = x6;
    let x19: x20[0x2f] = (x9 as x20[0x2f]);
    let x21: u25 = (u25:0x100);
    let x22: bool = (x0) == (x17);
    let x23: uN[0x1] = (x22)[x21+:uN[0x1]];
    let x24: uN[0x15] = (x5)[x16+:uN[0x15]];
    let x25: u8 = -(x16);
    x19
}