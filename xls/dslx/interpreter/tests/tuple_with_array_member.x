type Foo = (
  u32,
  u8[4],
);

fn flatten(x: u8[4]) -> u32 {
  x[u32:0] ++ x[u32:1] ++ x[u32:2] ++ x[u32:3]
}

fn add_members(x: Foo) -> u32 {
  x[u32:0] + flatten(x[u32:1])
}

test add_members {
  let x: Foo = (u32:0xdeadbeef, u8[4]:[0, 0, 0, 1]) in
  assert_eq(u32:0xdeadbef0, add_members(x))
}

fn main(x: Foo) -> u32 {
  add_members(x)
}
