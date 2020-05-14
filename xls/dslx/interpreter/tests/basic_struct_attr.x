struct Point {
  x: u32,
  y: u32,
}

fn f(p: Point) -> u32 {
  p.x + p.y
}

fn main() -> u32 {
  f(Point { x: u32:42, y: u32:64 })
}

test main {
  assert_eq(u32:106, main())
}
