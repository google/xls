struct Point {
  x: u32,
  y: u32,
}

fn id(x: Point) -> Point {
  x
}

fn main(x: Point) -> Point {
  id(x)
}

test id {
  let x = Point { x: u32:42, y: u32:64 } in
  let y = main(x) in
  assert_eq(x, y)
}
