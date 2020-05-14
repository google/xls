fn match_multi(x: u32) -> u32 {
  match x {
    u32:24 | u32:42 => u32:42;
    _ => u32:64
  }
}

test match_multi {
  let _ = assert_eq(u32:42, match_multi(u32:24)) in
  let _ = assert_eq(u32:42, match_multi(u32:42)) in
  let _ = assert_eq(u32:64, match_multi(u32:41)) in
  ()
}
