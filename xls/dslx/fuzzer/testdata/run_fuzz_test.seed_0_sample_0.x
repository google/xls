type x24 = uN[0x9];fn main(x0: u48, x1: s18, x2: u10, x3: u63, x4: u44, x5: u20, x6: u61) -> (uN[0x3d], s6, uN[0xc], u61, u20, u20, u10, uN[0x3d], uN[0x3d], u20, uN[0x2c], uN[0x3d], s6, uN[0x2c], u44, u10, uN[0x3d], s6, uN[0x3a], x24[0x1], u61, x24[0x1]) {
    let x7: uN[0x3a] = (x2) ++ (x0);
    let x8: u61 = clz(x6);
    let x9: uN[0x9] = (x5)[0xb+:uN[0x9]];
    let x10: uN[0x3d] = x8;
    let x11: bool = (bool:0x0);
    let x12: uN[0x3d] = x10;
    let x13: (u10,) = (x2,);
    let x14: uN[0x3d] = -(x12);
    let x15: uN[0x2c] = (x4)[:];
    let x16: bool = (x10) <= ((x11 as uN[0x3d]));
    let x17: uN[0x3d] = one_hot_sel(x11, [x12]);
    let x18: uN[0xc] = (x5)[x7+:uN[0xc]];
    let x19: uN[0x3d] = one_hot_sel(x16, [x10]);
    let x20: u10 = clz(x2);
    let x21: u10 = (x13)[(u32:0x0)];
    let x22: s6 = (s6:0x3f);
    let x23: x24[0x1] = (x9 as x24[0x1]);
    let x25: (uN[0x3d],) = (x10,);
    let x26: uN[0x3d] = for (i, x): (u4, uN[0x3d]) in range((u4:0x0), (u4:0x0)) {
    x
  }(x14)
  ;
    let x27: uN[0x2c] = (x8)[:0x2c];
    let x28: u10 = one_hot_sel(x16, [x21]);
    let x29: u20 = for (i, x): (u4, u20) in range((u4:0x0), (u4:0x1)) {
    x
  }(x5)
  ;
    let x30: uN[0x7] = (x29)[-0x12:-0xb];
    let x31: uN[0x2c] = clz(x27);
    (x19, x22, x18, x8, x5, x5, x2, x26, x19, x29, x31, x14, x22, x31, x4, x20, x17, x22, x7, x23, x6, x23)
}