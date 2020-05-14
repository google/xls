import xls.dslx.interpreter.tests.mod_imported

type MyEnum = mod_imported::MyEnum;

fn main(x: u8) -> MyEnum {
  x as MyEnum
}

test main {
  let _ = assert_eq(main(u8:42), MyEnum::FOO) in
  let _ = assert_eq(main(u8:64), MyEnum::BAR) in
  ()
}
