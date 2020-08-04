fn main() -> s32[4] {
  s32:[1, 2] ++ s32:[3, 4]
}

test main {
  assert_eq(s32[4]:[1, 2, 3, 4], main())
}
