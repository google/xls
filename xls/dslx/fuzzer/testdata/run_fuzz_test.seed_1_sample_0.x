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
    let x11: x7[0x2] = (x6) ++ (x6) in
    let x12: uN[2] = (x4)[0x3+:uN[2]] in
    let x13: uN[6] = one_hot(x4, (u1:0)) in
    let x14: bool = one_hot_sel(x4, [x2, x2, x2, x2, x2]) in
    let x15: s23 = for (i, x): (u4, s23) in range((u4:0x0), (u4:0x4)) {
    x
  }(x0)
   in
    let x16: bool = for (i, x): (u4, bool) in range((u4:0x0), (u4:0x0)) {
    x
  }(x2)
   in
    x3
}