enum MyEnum : u8 {
  FOO = 42,
  BAR = 64,
}

struct MyStruct {
  me: MyEnum
}

pub type MyEnumAlias = MyEnum;
pub type MyStructAlias = MyStruct;
pub type MyTupleAlias = (MyStruct, MyEnum);
