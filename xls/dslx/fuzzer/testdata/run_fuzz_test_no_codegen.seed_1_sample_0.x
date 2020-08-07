type x4 = uN[0xa];
type x7 = uN[0x5];fn main(x0: s23) -> uN[21] {
    let x1: uN[20] = (x0)[:0x14] in
    let x2: uN[60] = ((x1) ++ (x1)) ++ (x1) in
    let x3: x4[0x2] = (x1 as x4[0x2]) in
    let x5: uN[21] = one_hot(x1, (u1:1)) in
    let x6: x7[0x4] = (x1 as x7[0x4]) in
    let x8: s51 = (s51:0x8000) in
    let x9: u9 = (u9:0x80) in
    let x10: uN[47] = (((x9) ++ (x9)) ++ (x1)) ++ (x9) in
    let x11: u9 = one_hot_sel((uN[6]:0x4), [x9, x9, x9, x9, x9, x9]) in
    let x12: uN[9] = (x9)[:] in
    let x13: x4[0x4] = (x3) ++ (x3) in
    let x14: u1 = or_reduce(x11) in
    let x15: uN[60] = one_hot_sel(x14, [x2]) in
    let x16: u1 = and_reduce(x9) in
    let x17: s23 = -(x0) in
    let x18: uN[61] = (x15) ++ (x16) in
    let x19: uN[61] = one_hot_sel(x16, [x18]) in
    let x20: u1 = or_reduce(x19) in
    x5
}