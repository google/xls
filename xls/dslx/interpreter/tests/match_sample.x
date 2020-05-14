fn match_sample(s: bool, x: u32, y: u32) -> u32 {
  match (s, x, y) {
    (true, _, _) => x;
    (false, u32:7, b) => b;
    _ => u32:42;
  }
}
test match_wildcard {
  let _ = assert_eq(u32:7, match_sample(true, u32:7, u32:-1)) in
  let _ = assert_eq(u32:-1, match_sample(false, u32:7, u32:-1)) in
  let _ = assert_eq(u32:42, match_sample(false, u32:8, u32:-1)) in
  ()
}

fn match_wrapper(x: u32) -> u8 {
  match x {
    u32:42 => u8:1;
    u32:64 => u8:2;
    u32:77 => u8:3;
    _ => u8:4
  }
}

test match_wrapper {
  let _: () = assert_eq(u8:1, match_wrapper(u32:42)) in
  let _: () = assert_eq(u8:2, match_wrapper(u32:64)) in
  let _: () = assert_eq(u8:3, match_wrapper(u32:77)) in
  let _: () = assert_eq(u8:4, match_wrapper(u32:128)) in
  ()
}
