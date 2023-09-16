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

#include "gtest/gtest.h"
#include "xls/dslx/cpp_transpiler/test_types_lib.h"

namespace xls {
namespace {

TEST(TestTypesTest, EnumToString) {
  EXPECT_EQ(MyEnumToString(test::MyEnum::kA), "MyEnum::kA");
  EXPECT_EQ(MyEnumToString(test::MyEnum::kB), "MyEnum::kB");
  EXPECT_EQ(MyEnumToString(test::MyEnum::kC), "MyEnum::kC");
  EXPECT_EQ(MyEnumToString(test::MyEnum(1234)), "<unknown>");
}

TEST(TestTypesTest, SimpleStructToString) {
  test::InnerStruct s{.x = 42, .y = test::MyEnum::kB};
  EXPECT_EQ(s.ToString(), R"(InnerStruct {
  x: bits[32]:0x2a,
  y: MyEnum::kB,
})");
}

TEST(TestTypesTest, SimpleStructEq) {
  test::InnerStruct a{.x = 42, .y = test::MyEnum::kB};
  test::InnerStruct b{.x = 123, .y = test::MyEnum::kB};
  test::InnerStruct c{.x = 42, .y = test::MyEnum::kC};
  EXPECT_EQ(a, a);
  EXPECT_EQ(b, b);
  EXPECT_EQ(c, c);
  EXPECT_NE(a, b);
  EXPECT_NE(a, c);
}

TEST(TestTypesTest, EmptyStructToString) {
  test::EmptyStruct s;
  EXPECT_EQ(s.ToString(), R"(EmptyStruct {
})");
}

TEST(TestTypesTest, EmptyStructEq) {
  test::EmptyStruct s;
  test::EmptyStruct t;
  EXPECT_EQ(s, s);
  EXPECT_EQ(s, t);
}

TEST(TestTypesTest, NestedStructToString) {
  test::InnerStruct a{.x = 42, .y = test::MyEnum::kB};
  test::InnerStruct b{.x = 123, .y = test::MyEnum::kC};
  test::OuterStruct s{.a = a, .b = b, .c = 0xdead, .v = test::MyEnum::kA};
  EXPECT_EQ(s.ToString(), R"(OuterStruct {
  a: InnerStruct {
      x: bits[32]:0x2a,
      y: MyEnum::kB,
    },
  b: InnerStruct {
      x: bits[32]:0x7b,
      y: MyEnum::kC,
    },
  c: bits[37]:0xdead,
  v: MyEnum::kA,
})");
}

TEST(TestTypesTest, NestedStructEq) {
  test::InnerStruct a{.x = 42, .y = test::MyEnum::kB};
  test::InnerStruct b{.x = 42, .y = test::MyEnum::kC};
  test::InnerStruct c{.x = 123, .y = test::MyEnum::kB};
  test::OuterStruct x{.a = a, .b = b, .c = 0xdead, .v = test::MyEnum::kA};
  test::OuterStruct y{.a = a, .b = c, .c = 0xdead, .v = test::MyEnum::kA};
  test::OuterStruct z{.a = a, .b = b, .c = 0x1111, .v = test::MyEnum::kA};

  EXPECT_EQ(x, x);
  EXPECT_NE(x, y);
  EXPECT_NE(x, z);
}

TEST(TestTypesTest, DoublyNestedStructToString) {
  test::InnerStruct a{.x = 42, .y = test::MyEnum::kB};
  test::InnerStruct b{.x = 123, .y = test::MyEnum::kC};
  test::OuterStruct o{.a = a, .b = b, .c = 0xdead, .v = test::MyEnum::kA};
  test::OuterOuterStruct s{.q = test::EmptyStruct(), .s = o};
  EXPECT_EQ(s.ToString(), R"(OuterOuterStruct {
  q: EmptyStruct {
    },
  s: OuterStruct {
      a: InnerStruct {
          x: bits[32]:0x2a,
          y: MyEnum::kB,
        },
      b: InnerStruct {
          x: bits[32]:0x7b,
          y: MyEnum::kC,
        },
      c: bits[37]:0xdead,
      v: MyEnum::kA,
    },
})");
}

}  // namespace
}  // namespace xls
