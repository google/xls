import xls.dslx.interpreter.tests.mod_imported

fn fully_qualified(x: u32) -> mod_imported::Point {
  mod_imported::Point { x: x, y: u32:64 }
}

// A version that goes through a type alias.

type PointAlias = mod_imported::Point;

fn main(x: u32) -> PointAlias {
  PointAlias { x: x, y: u32:64 }
}

test main {
  let p: PointAlias = fully_qualified(u32:42) in
  let _ = assert_eq(u32:42, p.x) in
  let _ = assert_eq(u32:64, p.y) in
  let _ = assert_eq(main(u32:128), fully_qualified(u32:128)) in
  ()
}
