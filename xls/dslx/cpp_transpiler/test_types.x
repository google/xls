// Copyright 2023 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import xls.dslx.cpp_transpiler.test_types_dep as x;

// TODO:
//   * add width constants for type aliases and enums
type MyType = u37;
type MyTypeAlias = MyType;
type MySignedType = s20;
type MyBitsType = bits[42];
type MyTuple = (u35, s4, MyType, MySignedType);
type MyTupleOfTuples = (u3, MyTuple);
type MyTupleAlias = MyTuple;
type MyTupleAliasAlias = MyTupleAlias;
type MyEmptyTuple = ();
type MyArray = u17[2];
type MyU5 = u5;
type MyArrayOfArrays = MyU5[2][3];
type MyTupleArray = MyTuple[2];
type OtherTupleArray = (u32, u2)[2];

enum MyEnum : u7 {
  kA = 0,
  kB = 1,
  kC = u7:20 + u7:22,
}

struct InnerStruct {
  x: u17,
  y: MyEnum
}

type InnerStructAlias = InnerStruct;

struct OuterStruct {
  a: InnerStructAlias,
  b: InnerStruct,
  c: MyType,
  v: MyEnum,
}

struct EmptyStruct {}

struct OuterOuterStruct {
  q: EmptyStruct,
  some_array: u5[u32:1 + u32:2],
  s: OuterStruct,
}

struct StructWithTuple {
  t: MyTuple,
  t2: MyTupleAlias,
  t3: MyTupleAliasAlias,
}

struct StructWithArray {
 a: MyArray,
}

type InnerStructArray = InnerStruct[1];

struct StructWithStructArray {
  x: InnerStructArray,
}

struct StructWithTuplesArray {
  x: (),
  y: (u2, u4)
}

struct StructWithLotsOfTypes {
  v: bool,
  w: bits[3],
  x: u1,
  y: uN[44],
  z: sN[11],
}

type TupleOfStructs = (InnerStruct, InnerStruct);

struct FatType {
  x: u32[1000],
}

type snake_case_type_t = u13;

enum snake_case_enum_t : u7 {
  kA = 0,
  kB = 1,
}

struct snake_case_struct_t {
  some_field: snake_case_type_t,
  some_other_field: snake_case_enum_t,
}

struct StructWithKeywordFields {
  float: u32,
  int: u42,
}

enum EnumWithKeywordValues : u8 {
  float = 0,
  static = 1,
}

type float = u33;
