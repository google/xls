type x23 = uN[0x1];fn main(x0: u48, x1: s18, x2: u10, x3: u63, x4: u44, x5: u20, x6: u61) -> (x23[0x3d], u61, u48, s18) {
    let x7: uN[0x3a] = (x2) ++ (x0);
    let x8: u61 = clz(x6);
    let x9: uN[0x9] = (x5)[0xb+:uN[0x9]];
    let x10: uN[0x14] = x5;
    let x11: u48 = for (i, x): (u4, u48) in range((u4:0x0), (u4:0x6)) {
    x
  }(x0)
  ;
    let x12: s18 = (x1) << ((x2 as s18));
    let x13: s18 = ((x5 as s18)) | (x1);
    let x14: uN[0x2d] = one_hot(x4, (u1:0x1));
    let x15: u44 = clz(x4);
    let x16: s18 = one_hot_sel((uN[0x6]:0x15), [x13, x12, x1, x1, x13, x1]);
    let x17: uN[0x3f] = (x3)[x8+:uN[0x3f]];
    let x18: uN[0x2c] = (x4)[0x0+:uN[0x2c]];
    let x19: u33 = (u33:0x100000000);
    let x20: u63 = clz(x3);
    let x21: (u44, s18, u61, u48, s18, u20, u48, u61, u44, s18, u48) = (x15, x16, x6, x0, x1, x5, x11, x6, x4, x12, x0);
    let x22: x23[0x3d] = (x6 as x23[0x3d]);
    (x22, x6, x11, x16)
}