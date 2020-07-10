struct Point3 {
  x: u32,
  y: u32,
  z: u32,
}

fn update_y(p: Point3, new_y: u32) -> Point3 {
  Point3 { y: new_y, ..p }
}

fn main() -> Point3 {
  let p = Point3 { x: u32:42, y: u32:64, z: u32:256 } in
  update_y(p, u32:128)
}

test main {
  let want = Point3 { x: u32:42, y: u32:128, z: u32:256 } in
  assert_eq(want, main())
}
