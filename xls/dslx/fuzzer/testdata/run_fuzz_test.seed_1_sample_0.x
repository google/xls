type x7 = uN[0x1];fn main(x0: s23) -> uN[5] {
    let x1: s23 = !(x0) in
    let x2: bool = (bool:0x1) in
    let x3: uN[5] = ((((x2) ++ (x2)) ++ (x2)) ++ (x2)) ++ (x2) in
    let x4: uN[5] = for (i, x): (u4, uN[5]) in range((u4:0x0), (u4:0x3)) {
    x
  }(x3)
   in
    let x5: u18 = (u18:0x3ffff) in
    let x6: x7[0x1] = (x2 as x7[0x1]) in
    let x8: uN[51] = ((((x5) ++ (x3)) ++ (x4)) ++ (x4)) ++ (x5) in
    let x9: uN[5] = one_hot_sel(x3, [x4, x4, x3, x4, x3]) in
    let x10: s23 = (x0) - ((x9 as s23)) in
    let x11: uN[56] = (x3) ++ (x8) in
    let x12: x13[0x6] = (x5 as x13[0x6]) in
    let x14: uN[51] = clz(x8) in
    let x15: x16[0x1] = (x2 as x16[0x1]) in
    let x17: uN[39] = (x8)[x9+:uN[39]] in
    let x18: uN[5] = for (i, x): (u4, uN[5]) in range((u4:0x0), (u4:0x5)) {
    x
  }(x0)
   in
    let x16: bool = for (i, x): (u4, bool) in range((u4:0x0), (u4:0x0)) {
    x
  }(x2)
   in
    x15
}
