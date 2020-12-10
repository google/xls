const W1_V1 = u1:0x1;
const W6_V44 = u6:0x2c;
type x22 = u10;
type x24 = u1;
type x31 = u1;
fn main(x0: u48, x1: s18, x2: u10, x3: u63, x4: u44, x5: u20, x6: u61) -> (x22[0x2], u1, x31[W1_V1]) {
  let x7: u58 = (x2) ++ (x0);
  let x8: u61 = ctz(x6);
  let x9: u1 = or_reduce(x6);
  let x10: u61 = (((x3) as u61)) * (x8);
  let x11: s18 = one_hot_sel(x9, [x1]);
  let x12: u1 = xor_reduce(x0);
  let x13: u61 = x10;
  let x14: (u61,) = (x13,);
  let x15: u10 = rev(x2);
  let x16: u63 = (x3)[:];
  let x17: u1 = (x13) <= (((x15) as u61));
  let x18: s39 = s39:0x400000000;
  let x19: u1 = xor_reduce(x9);
  let x20: u1 = ctz(x19);
  let x21: x22[W1_V1] = ((x2) as x22[W1_V1]);
  let x23: x24[W6_V44] = ((x4) as x24[W6_V44]);
  let x25: u48 = (x0) ^ (((x17) as u48));
  let x26: u2 = (x11)[:-0x10];
  let x27: x22[0x2] = (x21) ++ (x21);
  let x28: u61 = clz(x10);
  let x29: u61 = x8;
  let x30: x31[W1_V1] = ((x9) as x31[W1_V1]);
  let x32: u57 = u57:0x4000;
  (x27, x19, x30)
}