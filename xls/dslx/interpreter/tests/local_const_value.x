fn f() -> u32[4] {
  const FOO = u32:4 in
  u32[FOO]:[0, ...]
}

test f {
  assert_eq(u32[4]:[0, ...], f())
}
