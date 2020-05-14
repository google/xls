enum MyEnum : u2 {
  FOO = 0,
  BAR = 1
}

fn main(xs: MyEnum[2], i: u1) -> MyEnum {
  xs[i]
}

test main {
  let xs: MyEnum[2] = MyEnum[2]:[MyEnum::FOO, MyEnum::BAR] in
  let _ = assert_eq(MyEnum::FOO, main(xs, u1:0)) in
  let _ = assert_eq(MyEnum::BAR, main(xs, u1:1)) in
  ()
}
