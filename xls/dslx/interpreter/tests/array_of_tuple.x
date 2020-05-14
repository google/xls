type Foo = (
  u8,
  u32,
);

test array_of_tuple_literal_test {
  let xs = Foo[2]:[(u8:1, u32:2), (u8:3, u32:4)]
  let x0 = xs[u32:0] in
  let x1 = xs[u32:1] in
  let _ = assert_eq(x0[u32:0], u8:1) in
  let _ = assert_eq(x0[u32:1], u32:2) in
  let _ = assert_eq(x0[u32:2], u8:3) in
  let _ = assert_eq(x1[u32:1], u32:4) in
  ()
}
