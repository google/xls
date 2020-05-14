fn main(a: u32[2], i: u1) -> u32 {
  a[i]
}

test main {
  let x = u32:42 in
  let y = u32:64 in
  // Make an array with "bracket notation".
  let my_array: u32[2] = [x, y] in
  let _ = assert_eq(main(my_array, u1:0), x) in
  let _ = assert_eq(main(my_array, u1:1), y) in
  ()
}
