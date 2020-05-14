const FOO = u8:42;

fn match_const(x: u8) -> u8 {
  match x {
    FOO => u8:0;
    _ => u8:42;
  }
}

test match_const_not_binding {
  let _ = assert_eq(u8:42, match_const(u8:0)) in
  let _ = assert_eq(u8:42, match_const(u8:1)) in
  let _ = assert_eq(u8:0, match_const(u8:42)) in
  ()
}

fn h(t: (u8, (u16, u32))) -> u32 {
  match t {
    (FOO, (x, y)) => (x as u32) + y;
    (_, (y, u32:42)) => y as u32;
    _ => u32:7;
  }
}

test match_nested {
  let _ = assert_eq(u32:3, h((u8:42, (u16:1, u32:2)))) in
  let _ = assert_eq(u32:1, h((u8:0, (u16:1, u32:42)))) in
  let _ = assert_eq(u32:7, h((u8:0, (u16:1, u32:0)))) in
  ()
}
