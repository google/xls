import std

// Create some wrappers around the builtins so we validate that we can
// typecheck them (typechecking doesn't apply to test code at the moment).
fn umul_2(x: u2, y: u2) -> u4 {
  std::umul(x, y)
}
fn smul_3(x: s3, y: s3) -> s6 {
  std::smul(x, y)
}

fn main(x: u3, y: u3) -> s6 {
  (umul_2(x as u2, y as u2) as s6) + smul_3(x as s3, y as s3)
}

test multiplies {
  let _ = assert_eq(u4:0b1001, umul_2(u2:0b11, u2:0b11)) in
  let _ = assert_eq(u4:0b0001, umul_2(u2:0b01, u2:0b01)) in
  let _ = assert_eq(s4:0b1111, std::smul(s2:0b11, s2:0b01)) in
  let _ = assert_eq(s6:6,  smul_3(s3:-3, s3:-2)) in
  let _ = assert_eq(s6:-6, smul_3(s3:-3, s3:2)) in
  let _ = assert_eq(s6:-6, smul_3(s3:3,  s3:-2)) in
  let _ = assert_eq(s6:6,  smul_3(s3:3,  s3:2)) in
  let _ = assert_eq(s6:1,  smul_3(s3:-1, s3:-1)) in
  let _ = assert_eq(u6:49, std::umul(u3:-1, u3:-1)) in
  ()
}
