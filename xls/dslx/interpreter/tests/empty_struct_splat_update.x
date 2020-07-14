struct Empty {}

fn main() -> Empty {
  let orig = Empty{} in
  Empty{..orig}
}

test main {
  assert_eq(main(), Empty {})
}
