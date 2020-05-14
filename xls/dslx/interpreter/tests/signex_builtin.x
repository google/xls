test signex_builtin {
  let x = u8:-1 in
  let s: s32 = signex(x, s32:0) in
  let u: u32 = signex(x, u32:0) in
  assert_eq(s as u32, u)
}
