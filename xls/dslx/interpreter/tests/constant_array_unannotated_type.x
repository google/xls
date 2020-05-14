fn main() -> u32[3] {
  // Note: no need for type annotations on the literal numbers here.
  u32[3]:[1, 2, 3]
}

test main {
  assert_eq(u32[3]:[u32:1, u32:2, u32:3], main())
}
