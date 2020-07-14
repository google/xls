struct Point3 {
  x: u32,
  y: u32,
  z: u32,
}

fn update_yz(p: Point3, new_y: u32, new_z: u32) -> Point3 {
  Point3 { y: new_y, z: new_z, ..p }
}

fn main() -> Point3 {
  let p = Point3 { x: u32:42, y: u32:0, z: u32:0 } in
  update_yz(p, u32:128, u32:256)
}

test main {
  let want = Point3 { x: u32:42, y: u32:128, z: u32:256 } in
  assert_eq(want, main())
}
