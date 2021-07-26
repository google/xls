// Copyright 2020 The XLS Authors
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
#include "xls/dslx/cpp_transpiler.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {
namespace {

// Verifies that the transpiler can convert a basic enum into C++.
TEST(CppTranspilerTest, BasicEnums) {
  const std::string kModule = R"(
pub enum MyEnum : u32 {
  A = 0,
  B = 1,
  C = 42,
  // D = 4294967296,
  E = 4294967295
}
)";

  const std::string kExpected = R"(enum class MyEnum {
  kA = 0,
  kB = 1,
  kC = 42,
  kE = 4294967295,
};)";

  ImportData import_data;
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           TranspileToCpp(module.module, &import_data));
  ASSERT_EQ(result.header, kExpected);
}

// Verifies we can use a constexpr evaluated constant in our enum.
TEST(CppTranspilerTest, EnumWithConstexprValues) {
  const std::string kModule = R"(
const MY_CONST = u48:17;

fn constexpr_fn(x: u16) -> u16 {
  x * x
}

pub enum MyEnum : u32 {
  A = 0,
  B = MY_CONST as u32,
  C = constexpr_fn(MY_CONST as u16) as u32
}
)";

  const std::string kExpected = R"(enum class MyEnum {
  kA = 0,
  kB = 17,
  kC = 289,
};)";

  ImportData import_data;
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           TranspileToCpp(module.module, &import_data));
  ASSERT_EQ(result.header, kExpected);
}

// Basic typedef support.
TEST(CppTranspilerTest, BasicTypedefs) {
  const std::string kModule = R"(
const CONST_1 = u32:4;

type MyType = u6;
type MySignedType = s8;
type MyThirdType = s9;

type MyArrayType1 = u31[8];
type MyArrayType2 = u31[CONST_1];
type MyArrayType3 = MySignedType[CONST_1];
type MyArrayType4 = s8[CONST_1];

type MyFirstTuple = (u7, s8, MyType, MySignedType, MyArrayType1, MyArrayType2);
)";

  const std::string kExpected = R"(using MyType = uint8_t;
using MySignedType = int8_t;
using MyThirdType = int16_t;
using MyArrayType1 = uint32_t[8];
using MyArrayType2 = uint32_t[4];
using MyArrayType3 = MySignedType[4];
using MyArrayType4 = int8_t[4];
using MyFirstTuple = std::tuple<uint8_t, int8_t, MyType, MySignedType, MyArrayType1, MyArrayType2>;)";
  ImportData import_data;
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           TranspileToCpp(module.module, &import_data));
  ASSERT_EQ(result.header, kExpected);
}

TEST(CppTranspilerTest, BasicStruct) {
  const std::string kModule = R"(
struct MyStruct {
  x: u32,
  y: u15,
  z: u8,
  w: s63,
}
)";

  const std::string kExpectedHeader = R"(struct MyStruct {
  static absl::StatusOr<MyStruct> FromValue(const Value& value) {
    absl::Span<const xls::Value> elements = value.elements();
    if (elements.size() != 4) {
      return absl::InvalidArgumentError(
          "MyStruct::FromValue input must be a 4-tuple.");
    }

    MyStruct result;
    result.x = elements[0].ToBits().ToUint64().value();
    result.y = elements[1].ToBits().ToUint64().value();
    result.z = elements[2].ToBits().ToUint64().value();
    result.w = elements[3].ToBits().ToUint64().value();
    return result;
  }

  Value ToValue() const {
    std::vector<Value> elements;
    Value x_value(UBits(x, /*bit_count=*/32));
    elements.push_back(x_value);
    Value y_value(UBits(y, /*bit_count=*/15));
    elements.push_back(y_value);
    Value z_value(UBits(z, /*bit_count=*/8));
    elements.push_back(z_value);
    Value w_value(SBits(w, /*bit_count=*/63));
    elements.push_back(w_value);
    return Value::Tuple(elements);
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const MyStruct& data);

  uint32_t x;
  uint16_t y;
  uint8_t z;
  int64_t w;
};
)";

  constexpr absl::string_view kExpectedBody =
      R"(std::ostream& operator<<(std::ostream& os, const MyStruct& data) {
  xls::Value value = data.ToValue();
  absl::Span<const xls::Value> elements = value.elements();
  os << "(\n";
  os << "  x: " << elements[0].ToString() << "\n";
  os << "  y: " << elements[1].ToString() << "\n";
  os << "  z: " << elements[2].ToString() << "\n";
  os << "  w: " << elements[3].ToString() << "\n";
  os << ")\n";
  return os;
})";

  ImportData import_data;
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           TranspileToCpp(module.module, &import_data));
  ASSERT_EQ(result.header, kExpectedHeader);
  ASSERT_EQ(result.body, kExpectedBody);
}

}  // namespace
}  // namespace xls::dslx
