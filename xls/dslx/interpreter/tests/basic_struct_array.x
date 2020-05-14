struct Point {
  x: u32,
  y: u32,
}

fn main(ps: Point[2], x: u1) -> Point {
  ps[x]
}

test main {
  let p0 = Point { x: u32:42, y: u32:64 } in
  let p1 = Point { y: u32:64, x: u32:42 } in
  let ps: Point[2] = [p0, p1] in
  let _ = assert_eq(p0, main(ps, u1:0)) in
  let _ = assert_eq(p1, main(ps, u1:1)) in
  ()
}
