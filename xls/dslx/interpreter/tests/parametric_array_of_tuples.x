type MyTuple = (u8,);

fn [N: u32] f(xs: MyTuple[N]) -> u32 {
  N
}

fn main() -> u32 {
  let xs: MyTuple[3] = MyTuple[3]:[(u8:0,), (u8:1,), (u8:2,)] in
  f(xs)
}

test main {
  assert_eq(u32:3, main())
}
