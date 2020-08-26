type x22 = uN[0xa];fn main(x0: u48, x1: s18, x2: u10, x3: u63, x4: u44, x5: u20, x6: u61) -> u2 {
    let x7: u58 = (x2) ++ (x0);
    let x8: u61 = clz(x6);
    let x9: u1 = or_reduce(x6);
    let x10: u61 = ((x3 as u61)) * (x8);
    let x11: s18 = one_hot_sel(x9, [x1]);
    let x12: u1 = xor_reduce(x0);
    let x13: u61 = x10;
    let x14: (u61) = (x13,);
    let x15: u11 = one_hot(x2, (u1:0x1));
    let x16: u2 = one_hot(x9, (u1:0x0));
    let x17: u1 = (x13) <= ((x15 as u61));
    let x18: s39 = (s39:0x400000000);
    let x19: u1 = xor_reduce(x9);
    let x20: u1 = clz(x19);
    let x21: x22[0x1] = (x2 as x22[0x1]);
    let x23: u33 = (u33:0x100000000);
    let x24: u11 = one_hot_sel(x17, [x15]);
    let x25: u61 = (x14)[(u32:0x0)];
    let x26: u1 = (x9)[-0x2:];
    x16
}