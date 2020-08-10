type x24 = uN[0x1];fn main(x0: u48, x1: s18, x2: u10, x3: u63, x4: u44, x5: u20, x6: u61) -> (s6, u61, bool, bool) {
    let x7: uN[58] = (x2) ++ (x0) in
    let x8: u61 = clz(x6) in
    let x9: uN[19] = (x5)[0x1:] in
    let x10: uN[61] = x8 in
    let x11: bool = (bool:0x0) in
    let x12: uN[61] = x10 in
    let x13: (u10,) = (x2,) in
    let x14: uN[61] = -(x12) in
    let x15: uN[44] = (x4)[0x0+:uN[44]] in
    let x16: bool = (x10) <= ((x11 as uN[61])) in
    let x17: uN[61] = one_hot_sel(x11, [x12]) in
    let x18: uN[8] = (x5)[-0x8:] in
    let x19: uN[61] = one_hot_sel(x16, [x10]) in
    let x20: u10 = clz(x2) in
    let x21: u10 = (x13)[(u32:0x0)] in
    let x22: s6 = (s6:0x3f) in
    let x23: x24[0x13] = (x9 as x24[0x13]) in
    (x22, x6, x11, x16)
}
