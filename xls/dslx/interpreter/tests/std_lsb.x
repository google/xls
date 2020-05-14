import std

fn main(x: u2) -> u1 {
  std::lsb(x)
}

test main {
  let _ = assert_eq(main(u2:0b01), u1:1) in
  let _ = assert_eq(main(u2:0b10), u1:0) in
  ()
}
