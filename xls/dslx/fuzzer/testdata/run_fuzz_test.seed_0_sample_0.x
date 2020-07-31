type x24 = uN[0x9];fn main(x0: u48, x1: s18, x2: u10, x3: u63, x4: u44, x5: u20, x6: u61) -> (uN[61], s6, uN[12], u61, u20, u20, u10, uN[61], uN[61], u20, uN[44], uN[61], s6, uN[44], u44, u10, uN[61], s6, uN[58], x24[0x1], u61, x24[0x1]) {
    let x7: uN[58] = (x2) ++ (x0) in
    let x8: u61 = clz(x6) in
    let x9: uN[9] = (x5)[0xb+:uN[9]] in
    let x10: uN[61] = x8 in
    let x11: bool = (bool:0x0) in
    let x12: uN[61] = x10 in
    let x13: (u10,) = (x2,) in
    let x14: uN[61] = -(x12) in
    let x15: uN[44] = (x4)[:] in
    let x16: bool = (x10) <= ((x11 as uN[61])) in
    let x17: uN[61] = one_hot_sel(x11, [x12]) in
    let x18: uN[12] = (x5)[x7+:uN[12]] in
    let x19: uN[61] = one_hot_sel(x16, [x10]) in
    let x20: u10 = clz(x2) in
    let x21: u10 = (x13)[(u32:0x0)] in
    let x22: s6 = (s6:0x3f) in
    let x23: x24[0x1] = (x9 as x24[0x1]) in
    let x25: (uN[61],) = (x10,) in
    let x26: uN[61] = for (i, x): (u4, uN[61]) in range((u4:0x0), (u4:0x0)) {
    x
  }(x14)
   in
    let x27: uN[44] = (x8)[:0x2c] in
    let x28: u10 = one_hot_sel(x16, [x21]) in
    let x29: u20 = for (i, x): (u4, u20) in range((u4:0x0), (u4:0x1)) {
    x
  }(x5)
   in
    let x30: uN[7] = (x29)[-0x12:-0xb] in
    let x31: uN[44] = clz(x27) in
    (x19, x22, x18, x8, x5, x5, x2, x26, x19, x29, x31, x14, x22, x31, x4, x20, x17, x22, x7, x23, x6, x23)
}