import xls.dslx.interpreter.tests.mod_imported_aliases

type MyEnum = mod_imported_aliases::MyEnumAlias;
type MyStruct = mod_imported_aliases::MyStructAlias;
type MyTuple = mod_imported_aliases::MyTupleAlias;

fn main(x: u8) -> MyTuple {
  (MyStruct { me: x as MyEnum }, MyEnum::FOO)
}

test main {
  let (ms, me) = main(u8:64) in
  let _ = assert_eq(MyEnum::BAR, ms.me) in
  assert_eq(MyEnum::FOO, me)
}
