struct Point {
  x: u32,
  y: u32,
}

test struct_equality {
  let p0 = Point { x: u32:42, y: u32:64 } in
  let p1 = Point { y: u32:64, x: u32:42 } in
  assert_eq(p0, p1)
}
