fn main(x: u32, y: u32) -> u32 {
  x-y
}

test subtract_to_negative {
  let x: u32 = u32:5 in
  let y: u32 = u32:6 in
  assert_eq(u32:-1, main(x, y))
}
