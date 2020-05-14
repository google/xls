fn make_array(x: u32) -> u32[3] {
  u32[3]:[u32:42, x, ...]
}

test make_array {
  let _ = assert_eq(u32[3]:[u32:42, u32:42, u32:42], make_array(u32:42)) in
  let _ = assert_eq(u32[3]:[u32:42, u32:64, u32:64], make_array(u32:64)) in
  ()
}
