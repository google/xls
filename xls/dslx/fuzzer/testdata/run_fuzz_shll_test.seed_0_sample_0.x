fn main(x0: u48, x1: s18, x2: u10, x3: u63, x4: u44, x5: u20, x6: u61) -> bool {
    let x7: uN[58] = (x2) ++ (x0);
    let x8: u61 = clz(x6);
    let x9: u1 = or_reduce(x6);
    let x10: u61 = ((x3 as u61)) << (((u61:0x3a)) if ((x8) >= ((u61:0x3a))) else (x8));
    let x11: u43 = (u43:0x4);
    let x12: u63 = for (i, x): (u4, u63) in range((u4:0x0), (u4:0x7)) {
    x
  }(x3)
   in
    let x13: u1 = xor_reduce(x12) in
    let x14: (u1,) = (x13,) in
    let x15: uN[2] = one_hot(x13, (u1:1)) in
    let x16: uN[62] = one_hot(x8, (u1:0)) in
    let x17: bool = (x13) <= ((x15 as u1)) in
    let x18: s39 = (s39:0x400000000) in
    let x19: u1 = xor_reduce(x6) in
    let x20: uN[62] = (x6) ++ (x19) in
    let x21: u33 = (u33:0x100000000) in
    let x22: uN[2] = one_hot_sel(x17, [x15]) in
    let x23: u1 = (x14)[(u32:0x0)] in
    let x24: uN[1] = (x9)[-0x2:] in
    let x25: (u1,) = (x13,) in
    let x26: s22 = (s22:0x20) in
    x17
}