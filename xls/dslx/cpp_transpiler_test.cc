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
})";

  const std::string kExpectedHeader = R"(struct MyStruct {
  static absl::StatusOr<MyStruct> FromValue(const Value& value);

  Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const MyStruct& data);

  uint32_t x;
  uint16_t y;
  uint8_t z;
  int64_t w;
};
)";

  constexpr absl::string_view kExpectedBody =
      R"(absl::StatusOr<MyStruct> MyStruct::FromValue(const Value& value) {
  absl::Span<const xls::Value> elements = value.elements();
  if (elements.size() != 4) {
    return absl::InvalidArgumentError(
        "MyStruct::FromValue input must be a 4-tuple.");
  }

  MyStruct result;
  result.x = elements[0].ToBits().ToUint64().value();
  result.y = elements[1].ToBits().ToUint64().value();
  result.z = elements[2].ToBits().ToUint64().value();
  result.w = elements[3].ToBits().ToInt64().value();
  return result;
}

Value MyStruct::ToValue() const {
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

std::ostream& operator<<(std::ostream& os, const MyStruct& data) {
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

TEST(CppTranspilerTest, BasicArray) {
  constexpr absl::string_view kModule = R"(
struct MyStruct {
  x: u32[32],
  y: s7[8],
  z: u8[7],
})";

  constexpr absl::string_view kExpectedHeader = R"(struct MyStruct {
  static absl::StatusOr<MyStruct> FromValue(const Value& value);

  Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const MyStruct& data);

  uint32_t x[32];
  int8_t y[8];
  uint8_t z[7];
};
)";

  constexpr absl::string_view kExpectedBody =
      R"(absl::StatusOr<MyStruct> MyStruct::FromValue(const Value& value) {
  absl::Span<const xls::Value> elements = value.elements();
  if (elements.size() != 3) {
    return absl::InvalidArgumentError(
        "MyStruct::FromValue input must be a 3-tuple.");
  }

  MyStruct result;
  for (int i = 0; i < 32; i++) {
    result.x[i] = elements[0].element(i).ToBits().ToUint64().value();
  }
  for (int i = 0; i < 8; i++) {
    result.y[i] = elements[1].element(i).ToBits().ToInt64().value();
  }
  for (int i = 0; i < 7; i++) {
    result.z[i] = elements[2].element(i).ToBits().ToUint64().value();
  }
  return result;
}

Value MyStruct::ToValue() const {
  std::vector<Value> elements;
  std::vector<Value> x_elements;
  for (int i = 0; i < ABSL_ARRAYSIZE(x); i++) {
    Value x_element(UBits(x[i], /*bit_count=*/32));
    x_elements.push_back(x_element);
  }
  Value x_value = Value::ArrayOrDie(x_elements);
  elements.push_back(x_value);
  std::vector<Value> y_elements;
  for (int i = 0; i < ABSL_ARRAYSIZE(y); i++) {
    Value y_element(SBits(y[i], /*bit_count=*/7));
    y_elements.push_back(y_element);
  }
  Value y_value = Value::ArrayOrDie(y_elements);
  elements.push_back(y_value);
  std::vector<Value> z_elements;
  for (int i = 0; i < ABSL_ARRAYSIZE(z); i++) {
    Value z_element(UBits(z[i], /*bit_count=*/8));
    z_elements.push_back(z_element);
  }
  Value z_value = Value::ArrayOrDie(z_elements);
  elements.push_back(z_value);
  return Value::Tuple(elements);
}

std::ostream& operator<<(std::ostream& os, const MyStruct& data) {
  xls::Value value = data.ToValue();
  absl::Span<const xls::Value> elements = value.elements();
  os << "(\n";
  os << "  x: " << elements[0].ToString() << "\n";
  os << "  y: " << elements[1].ToString() << "\n";
  os << "  z: " << elements[2].ToString() << "\n";
  os << ")\n";
  return os;
})";

  ImportData import_data;
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           TranspileToCpp(module.module, &import_data));
  EXPECT_EQ(result.header, kExpectedHeader);
  EXPECT_EQ(result.body, kExpectedBody);
}

TEST(CppTranspilerTest, StructWithStruct) {
  constexpr absl::string_view kModule = R"(
struct InnerStruct {
  x: u32,
  y: u16
}

struct OuterStruct {
  x: u32,
  a: InnerStruct,
  b: InnerStruct
})";

  constexpr absl::string_view kExpectedHeader = R"(struct InnerStruct {
  static absl::StatusOr<InnerStruct> FromValue(const Value& value);

  Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const InnerStruct& data);

  uint32_t x;
  uint16_t y;
};

struct OuterStruct {
  static absl::StatusOr<OuterStruct> FromValue(const Value& value);

  Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const OuterStruct& data);

  uint32_t x;
  InnerStruct a;
  InnerStruct b;
};
)";
  constexpr absl::string_view kExpectedBody =
      R"(absl::StatusOr<InnerStruct> InnerStruct::FromValue(const Value& value) {
  absl::Span<const xls::Value> elements = value.elements();
  if (elements.size() != 2) {
    return absl::InvalidArgumentError(
        "InnerStruct::FromValue input must be a 2-tuple.");
  }

  InnerStruct result;
  result.x = elements[0].ToBits().ToUint64().value();
  result.y = elements[1].ToBits().ToUint64().value();
  return result;
}

Value InnerStruct::ToValue() const {
  std::vector<Value> elements;
  Value x_value(UBits(x, /*bit_count=*/32));
  elements.push_back(x_value);
  Value y_value(UBits(y, /*bit_count=*/16));
  elements.push_back(y_value);
  return Value::Tuple(elements);
}

std::ostream& operator<<(std::ostream& os, const InnerStruct& data) {
  xls::Value value = data.ToValue();
  absl::Span<const xls::Value> elements = value.elements();
  os << "(\n";
  os << "  x: " << elements[0].ToString() << "\n";
  os << "  y: " << elements[1].ToString() << "\n";
  os << ")\n";
  return os;
}

absl::StatusOr<OuterStruct> OuterStruct::FromValue(const Value& value) {
  absl::Span<const xls::Value> elements = value.elements();
  if (elements.size() != 3) {
    return absl::InvalidArgumentError(
        "OuterStruct::FromValue input must be a 3-tuple.");
  }

  OuterStruct result;
  result.x = elements[0].ToBits().ToUint64().value();
  XLS_ASSIGN_OR_RETURN(result.a, InnerStruct::FromValue(elements[1]));
  XLS_ASSIGN_OR_RETURN(result.b, InnerStruct::FromValue(elements[2]));
  return result;
}

Value OuterStruct::ToValue() const {
  std::vector<Value> elements;
  Value x_value(UBits(x, /*bit_count=*/32));
  elements.push_back(x_value);
  elements.push_back(a.ToValue());
  elements.push_back(b.ToValue());
  return Value::Tuple(elements);
}

std::ostream& operator<<(std::ostream& os, const OuterStruct& data) {
  xls::Value value = data.ToValue();
  absl::Span<const xls::Value> elements = value.elements();
  os << "(\n";
  os << "  x: " << elements[0].ToString() << "\n";
  os << "  a: " << elements[1].ToString() << "\n";
  os << "  b: " << elements[2].ToString() << "\n";
  os << ")\n";
  return os;
})";

  ImportData import_data;
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           TranspileToCpp(module.module, &import_data));
  EXPECT_EQ(result.header, kExpectedHeader);
  EXPECT_EQ(result.body, kExpectedBody);
}

TEST(CppTranspilerTest, StructWithStructWithStruct) {
  constexpr absl::string_view kModule = R"(
struct InnerStruct {
  x: u32,
  y: u16
}

struct MiddleStruct {
  z: u48,
  a: InnerStruct,
}

struct OtherMiddleStruct {
  b: InnerStruct,
  w: u64,
}

struct OuterStruct {
  a: InnerStruct,
  b: MiddleStruct,
  c: OtherMiddleStruct,
  v: u8,
})";

  constexpr absl::string_view kExpectedHeader = R"(struct InnerStruct {
  static absl::StatusOr<InnerStruct> FromValue(const Value& value);

  Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const InnerStruct& data);

  uint32_t x;
  uint16_t y;
};

struct MiddleStruct {
  static absl::StatusOr<MiddleStruct> FromValue(const Value& value);

  Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const MiddleStruct& data);

  uint64_t z;
  InnerStruct a;
};

struct OtherMiddleStruct {
  static absl::StatusOr<OtherMiddleStruct> FromValue(const Value& value);

  Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const OtherMiddleStruct& data);

  InnerStruct b;
  uint64_t w;
};

struct OuterStruct {
  static absl::StatusOr<OuterStruct> FromValue(const Value& value);

  Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const OuterStruct& data);

  InnerStruct a;
  MiddleStruct b;
  OtherMiddleStruct c;
  uint8_t v;
};
)";
  constexpr absl::string_view kExpectedBody =
      R"(absl::StatusOr<InnerStruct> InnerStruct::FromValue(const Value& value) {
  absl::Span<const xls::Value> elements = value.elements();
  if (elements.size() != 2) {
    return absl::InvalidArgumentError(
        "InnerStruct::FromValue input must be a 2-tuple.");
  }

  InnerStruct result;
  result.x = elements[0].ToBits().ToUint64().value();
  result.y = elements[1].ToBits().ToUint64().value();
  return result;
}

Value InnerStruct::ToValue() const {
  std::vector<Value> elements;
  Value x_value(UBits(x, /*bit_count=*/32));
  elements.push_back(x_value);
  Value y_value(UBits(y, /*bit_count=*/16));
  elements.push_back(y_value);
  return Value::Tuple(elements);
}

std::ostream& operator<<(std::ostream& os, const InnerStruct& data) {
  xls::Value value = data.ToValue();
  absl::Span<const xls::Value> elements = value.elements();
  os << "(\n";
  os << "  x: " << elements[0].ToString() << "\n";
  os << "  y: " << elements[1].ToString() << "\n";
  os << ")\n";
  return os;
}

absl::StatusOr<MiddleStruct> MiddleStruct::FromValue(const Value& value) {
  absl::Span<const xls::Value> elements = value.elements();
  if (elements.size() != 2) {
    return absl::InvalidArgumentError(
        "MiddleStruct::FromValue input must be a 2-tuple.");
  }

  MiddleStruct result;
  result.z = elements[0].ToBits().ToUint64().value();
  XLS_ASSIGN_OR_RETURN(result.a, InnerStruct::FromValue(elements[1]));
  return result;
}

Value MiddleStruct::ToValue() const {
  std::vector<Value> elements;
  Value z_value(UBits(z, /*bit_count=*/48));
  elements.push_back(z_value);
  elements.push_back(a.ToValue());
  return Value::Tuple(elements);
}

std::ostream& operator<<(std::ostream& os, const MiddleStruct& data) {
  xls::Value value = data.ToValue();
  absl::Span<const xls::Value> elements = value.elements();
  os << "(\n";
  os << "  z: " << elements[0].ToString() << "\n";
  os << "  a: " << elements[1].ToString() << "\n";
  os << ")\n";
  return os;
}

absl::StatusOr<OtherMiddleStruct> OtherMiddleStruct::FromValue(const Value& value) {
  absl::Span<const xls::Value> elements = value.elements();
  if (elements.size() != 2) {
    return absl::InvalidArgumentError(
        "OtherMiddleStruct::FromValue input must be a 2-tuple.");
  }

  OtherMiddleStruct result;
  XLS_ASSIGN_OR_RETURN(result.b, InnerStruct::FromValue(elements[0]));
  result.w = elements[1].ToBits().ToUint64().value();
  return result;
}

Value OtherMiddleStruct::ToValue() const {
  std::vector<Value> elements;
  elements.push_back(b.ToValue());
  Value w_value(UBits(w, /*bit_count=*/64));
  elements.push_back(w_value);
  return Value::Tuple(elements);
}

std::ostream& operator<<(std::ostream& os, const OtherMiddleStruct& data) {
  xls::Value value = data.ToValue();
  absl::Span<const xls::Value> elements = value.elements();
  os << "(\n";
  os << "  b: " << elements[0].ToString() << "\n";
  os << "  w: " << elements[1].ToString() << "\n";
  os << ")\n";
  return os;
}

absl::StatusOr<OuterStruct> OuterStruct::FromValue(const Value& value) {
  absl::Span<const xls::Value> elements = value.elements();
  if (elements.size() != 4) {
    return absl::InvalidArgumentError(
        "OuterStruct::FromValue input must be a 4-tuple.");
  }

  OuterStruct result;
  XLS_ASSIGN_OR_RETURN(result.a, InnerStruct::FromValue(elements[0]));
  XLS_ASSIGN_OR_RETURN(result.b, MiddleStruct::FromValue(elements[1]));
  XLS_ASSIGN_OR_RETURN(result.c, OtherMiddleStruct::FromValue(elements[2]));
  result.v = elements[3].ToBits().ToUint64().value();
  return result;
}

Value OuterStruct::ToValue() const {
  std::vector<Value> elements;
  elements.push_back(a.ToValue());
  elements.push_back(b.ToValue());
  elements.push_back(c.ToValue());
  Value v_value(UBits(v, /*bit_count=*/8));
  elements.push_back(v_value);
  return Value::Tuple(elements);
}

std::ostream& operator<<(std::ostream& os, const OuterStruct& data) {
  xls::Value value = data.ToValue();
  absl::Span<const xls::Value> elements = value.elements();
  os << "(\n";
  os << "  a: " << elements[0].ToString() << "\n";
  os << "  b: " << elements[1].ToString() << "\n";
  os << "  c: " << elements[2].ToString() << "\n";
  os << "  v: " << elements[3].ToString() << "\n";
  os << ")\n";
  return os;
})";

  ImportData import_data;
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           TranspileToCpp(module.module, &import_data));
  EXPECT_EQ(result.header, kExpectedHeader);
  EXPECT_EQ(result.body, kExpectedBody);
}

}  // namespace
}  // namespace xls::dslx
