type x24 = uN[0x9];
fn main(x0: u48, x1: s18, x2: u10, x3: u63, x4: u44, x5: u20, x6: u61) -> (u61, s6, u12, u61, u20, u20, u10, u61, u61, u20, u44, u61, s6, u44, u44, u10, u61, s6, u58, x24[0x1], u61, x24[0x1]) {
  let x7: u58 = (x2) ++ (x0);
  let x8: u61 = ctz(x6);
  let x9: u9 = (x5)[0xb+:u9];
  let x10: u61 = x8;
  let x11: bool = bool:false;
  let x12: u61 = x10;
  let x13: (u10,) = (x2,);
  let x14: u61 = -(x12);
  let x15: u44 = (x4)[:];
  let x16: u1 = (x10) <= (((x11) as u61));
  let x17: u61 = one_hot_sel(x11, [x12]);
  let x18: u12 = (x5)[x7+:u12];
  let x19: u61 = one_hot_sel(x16, [x10]);
  let x20: u10 = ctz(x2);
  let x21: u10 = (x13)[u32:0x0];
  let x22: s6 = s6:0x3f;
  let x23: x24[0x1] = ((x9) as x24[0x1]);
  let x25: (u61,) = (x10,);
  let x26: u61 = for (i, x): (u4, u61) in range(u4:0x0, u4:0x0) {
    x
  }(x14);
  let x27: u44 = (x8)[:0x2c];
  let x28: u10 = one_hot_sel(x16, [x21]);
  let x29: u20 = for (i, x): (u4, u20) in range(u4:0x0, u4:0x1) {
    x
  }(x5);
  let x30: u7 = (x29)[-0x12:-0xb];
  let x31: u44 = ctz(x27);
  (x19, x22, x18, x8, x5, x5, x2, x26, x19, x29, x31, x14, x22, x31, x4, x20, x17, x22, x7, x23, x6, x23)
}