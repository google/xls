type x24 = uN[0x1];fn main(x0: u48, x1: s18, x2: u10, x3: u63, x4: u44, x5: u20, x6: u61) -> (s6, u61, bool, bool) {
    let x7: uN[58] = (x2) ++ (x0);
    let x8: u61 = clz(x6);
    let x9: uN[19] = (x5)[0x1:];
    let x10: uN[61] = x8;
    let x11: bool = (bool:0x0);
    let x12: uN[61] = x10;
    let x13: (u10,) = (x2,);
    let x14: uN[61] = -(x12);
    let x15: uN[44] = (x4)[0x0+:uN[44]];
    let x16: bool = (x10) <= ((x11 as uN[61]));
    let x17: uN[61] = one_hot_sel(x11, [x12]);
    let x18: uN[8] = (x5)[-0x8:];
    let x19: uN[61] = one_hot_sel(x16, [x10]);
    let x20: u10 = clz(x2);
    let x21: u10 = (x13)[(u32:0x0)];
    let x22: s6 = (s6:0x3f);
    let x23: x24[0x13] = (x9 as x24[0x13]);
    (x22, x6, x11, x16)
}