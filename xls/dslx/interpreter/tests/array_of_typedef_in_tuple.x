type Foo = (u8, u32);
type Bar = (
  u16,
  Foo[2],
);
type Foo2 = Foo[2];

fn foo_add_all(x: Foo) -> u32 {
  (x[u32:0] as u32) + x[u32:1]
}

fn bar_add_all(x: Bar) -> u32 {
  let foos: Foo2 = x[u32:1] in
  let foo0: Foo = foos[u32:0] in
  let foo1: Foo = foos[u32:1] in
  (x[u32:0] as u32) + foo_add_all(foo0) + foo_add_all(foo1)
}

test bar_add_all {
  let foo0: Foo = (u8:1, u32:2) in
  let bar: Bar = (u16:3, [foo0, foo0]) in
  assert_eq(u32:9, bar_add_all(bar))
}

fn main(x: Bar) -> u32 {
  bar_add_all(x)
}

