struct Point {
  x: u32,
  y: u32,
}

fn main(xy: u32) -> Point {
  Point { x: xy, y: xy }
}

test f {
  let p: Point = main(u32:42) in
  let _ = assert_eq(u32:42, p.x) in
  let _ = assert_eq(u32:42, p.y) in
  ()
}

test alternative_forms_equal {
  let p0 = Point { x: u32:42, y: u32:64 } in
  let p1 = Point { y: u32:64, x: u32:42 } in
  assert_eq(p0, p1)
}
