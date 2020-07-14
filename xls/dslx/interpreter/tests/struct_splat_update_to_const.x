struct Point {
  x: u32,
  y: u32,
}

const BEST_Y = u32:42;

fn update_y(p: Point) -> Point {
  Point{ y: BEST_Y, ..p }
}
