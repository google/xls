fn main(x: s8, y: s8) -> s8 {
  x - y
}

test main {
  let x: s8 = s8:2 in
  let y: s8 = s8:3 in
  let z: s8 = main(x, y) in
  let _ = assert_lt(z, s8:0) in
  let _ = assert_eq(false, z >= s8:0) in
  ()
}
