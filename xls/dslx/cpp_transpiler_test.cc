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
#include "xls/dslx/create_import_data.h"
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

  const std::string kExpected =
      R"(// AUTOMATICALLY GENERATED FILE. DO NOT EDIT!
#ifndef FAKE_PATH_H_
#define FAKE_PATH_H_
#include <cstdint>
#include <ostream>

#include "absl/status/statusor.h"
#include "xls/public/value.h"

enum class MyEnum {
  kA = 0,
  kB = 1,
  kC = 42,
  kE = 4294967295,
};
constexpr int64_t kMyEnumNumElements = 4;

#endif  // FAKE_PATH_H_
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, TranspileToCpp(module.module, &import_data, "fake_path.h"));
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

  const std::string kExpected = R"(// AUTOMATICALLY GENERATED FILE. DO NOT EDIT!
#ifndef FAKE_PATH_H_
#define FAKE_PATH_H_
#include <cstdint>
#include <ostream>

#include "absl/status/statusor.h"
#include "xls/public/value.h"

enum class MyEnum {
  kA = 0,
  kB = 17,
  kC = 289,
};
constexpr int64_t kMyEnumNumElements = 3;

#endif  // FAKE_PATH_H_
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, TranspileToCpp(module.module, &import_data, "fake_path.h"));
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

  const std::string kExpected = R"(// AUTOMATICALLY GENERATED FILE. DO NOT EDIT!
#ifndef FAKE_PATH_H_
#define FAKE_PATH_H_
#include <cstdint>
#include <ostream>

#include "absl/status/statusor.h"
#include "xls/public/value.h"

namespace robs::secret::space {

using MyType = uint8_t;

using MySignedType = int8_t;

using MyThirdType = int16_t;

using MyArrayType1 = uint32_t[8];

using MyArrayType2 = uint32_t[4];

using MyArrayType3 = MySignedType[4];

using MyArrayType4 = int8_t[4];

using MyFirstTuple = std::tuple<uint8_t, int8_t, MyType, MySignedType, MyArrayType1, MyArrayType2>;

}  // namespace robs::secret::space

#endif  // FAKE_PATH_H_
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, TranspileToCpp(module.module, &import_data, "fake_path.h",
                                  "robs::secret::space"));
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

  const std::string kExpectedHeader =
      R"(// AUTOMATICALLY GENERATED FILE. DO NOT EDIT!
#ifndef FAKE_PATH_H_
#define FAKE_PATH_H_
#include <cstdint>
#include <ostream>

#include "absl/status/statusor.h"
#include "xls/public/value.h"

struct MyStruct {
  static absl::StatusOr<MyStruct> FromValue(const xls::Value& value);

  xls::Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const MyStruct& data);

  uint32_t x;
  uint16_t y;
  uint8_t z;
  int64_t w;

  static constexpr int64_t kXWidth = 32;
  static constexpr int64_t kYWidth = 15;
  static constexpr int64_t kZWidth = 8;
  static constexpr int64_t kWWidth = 63;
};

#endif  // FAKE_PATH_H_
)";

  constexpr std::string_view kExpectedBody =
      R"(// AUTOMATICALLY GENERATED FILE. DO NOT EDIT!
#include <vector>

#include "fake_path.h"
#include "absl/base/macros.h"
#include "absl/status/status.h"
#include "absl/types/span.h"

absl::StatusOr<MyStruct> MyStruct::FromValue(const xls::Value& value) {
  absl::Span<const xls::Value> elements = value.elements();
  if (elements.size() != 4) {
    return absl::InvalidArgumentError(
        "MyStruct::FromValue input must be a 4-tuple.");
  }

  MyStruct result;
  result.x = elements[0].bits().ToUint64().value();
  result.y = elements[1].bits().ToUint64().value();
  result.z = elements[2].bits().ToUint64().value();
  result.w = elements[3].bits().ToInt64().value();
  return result;
}

xls::Value MyStruct::ToValue() const {
  std::vector<xls::Value> elements;
  xls::Value x_value(xls::UBits(x, /*bit_count=*/32));
  elements.push_back(x_value);
  xls::Value y_value(xls::UBits(y, /*bit_count=*/15));
  elements.push_back(y_value);
  xls::Value z_value(xls::UBits(z, /*bit_count=*/8));
  elements.push_back(z_value);
  xls::Value w_value(xls::SBits(w, /*bit_count=*/63));
  elements.push_back(w_value);
  return xls::Value::Tuple(elements);
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
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, TranspileToCpp(module.module, &import_data, "fake_path.h"));
  ASSERT_EQ(result.header, kExpectedHeader);
  ASSERT_EQ(result.body, kExpectedBody);
}

TEST(CppTranspilerTest, BasicArray) {
  constexpr std::string_view kModule = R"(
struct MyStruct {
  x: u32[32],
  y: s7[8],
  z: u8[7],
})";

  constexpr std::string_view kExpectedHeader =
      R"(// AUTOMATICALLY GENERATED FILE. DO NOT EDIT!
#ifndef FAKE_PATH_H_
#define FAKE_PATH_H_
#include <cstdint>
#include <ostream>

#include "absl/status/statusor.h"
#include "xls/public/value.h"

struct MyStruct {
  static absl::StatusOr<MyStruct> FromValue(const xls::Value& value);

  xls::Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const MyStruct& data);

  uint32_t x[32];
  int8_t y[8];
  uint8_t z[7];
};

#endif  // FAKE_PATH_H_
)";

  constexpr std::string_view kExpectedBody =
      R"(// AUTOMATICALLY GENERATED FILE. DO NOT EDIT!
#include <vector>

#include "fake_path.h"
#include "absl/base/macros.h"
#include "absl/status/status.h"
#include "absl/types/span.h"

absl::StatusOr<MyStruct> MyStruct::FromValue(const xls::Value& value) {
  absl::Span<const xls::Value> elements = value.elements();
  if (elements.size() != 3) {
    return absl::InvalidArgumentError(
        "MyStruct::FromValue input must be a 3-tuple.");
  }

  MyStruct result;
  for (int i = 0; i < 32; i++) {
    result.x[i] = elements[0].element(i).bits().ToUint64().value();
  }
  for (int i = 0; i < 8; i++) {
    result.y[i] = elements[1].element(i).bits().ToInt64().value();
  }
  for (int i = 0; i < 7; i++) {
    result.z[i] = elements[2].element(i).bits().ToUint64().value();
  }
  return result;
}

xls::Value MyStruct::ToValue() const {
  std::vector<xls::Value> elements;
  std::vector<xls::Value> x_elements;
  for (int i = 0; i < ABSL_ARRAYSIZE(x); i++) {
    xls::Value x_element(xls::UBits(x[i], /*bit_count=*/32));
    x_elements.push_back(x_element);
  }
  xls::Value x_value = xls::Value::ArrayOrDie(x_elements);
  elements.push_back(x_value);
  std::vector<xls::Value> y_elements;
  for (int i = 0; i < ABSL_ARRAYSIZE(y); i++) {
    xls::Value y_element(xls::SBits(y[i], /*bit_count=*/7));
    y_elements.push_back(y_element);
  }
  xls::Value y_value = xls::Value::ArrayOrDie(y_elements);
  elements.push_back(y_value);
  std::vector<xls::Value> z_elements;
  for (int i = 0; i < ABSL_ARRAYSIZE(z); i++) {
    xls::Value z_element(xls::UBits(z[i], /*bit_count=*/8));
    z_elements.push_back(z_element);
  }
  xls::Value z_value = xls::Value::ArrayOrDie(z_elements);
  elements.push_back(z_value);
  return xls::Value::Tuple(elements);
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
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, TranspileToCpp(module.module, &import_data, "fake_path.h"));
  EXPECT_EQ(result.header, kExpectedHeader);
  EXPECT_EQ(result.body, kExpectedBody);
}

TEST(CppTranspilerTest, StructWithStruct) {
  constexpr std::string_view kModule = R"(
struct InnerStruct {
  x: u32,
  y: u16
}

struct OuterStruct {
  x: u32,
  a: InnerStruct,
  b: InnerStruct
})";

  constexpr std::string_view kExpectedHeader =
      R"(// AUTOMATICALLY GENERATED FILE. DO NOT EDIT!
#ifndef FAKE_PATH_H_
#define FAKE_PATH_H_
#include <cstdint>
#include <ostream>

#include "absl/status/statusor.h"
#include "xls/public/value.h"

struct InnerStruct {
  static absl::StatusOr<InnerStruct> FromValue(const xls::Value& value);

  xls::Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const InnerStruct& data);

  uint32_t x;
  uint16_t y;

  static constexpr int64_t kXWidth = 32;
  static constexpr int64_t kYWidth = 16;
};

struct OuterStruct {
  static absl::StatusOr<OuterStruct> FromValue(const xls::Value& value);

  xls::Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const OuterStruct& data);

  uint32_t x;
  InnerStruct a;
  InnerStruct b;

  static constexpr int64_t kXWidth = 32;
};

#endif  // FAKE_PATH_H_
)";
  constexpr std::string_view kExpectedBody =
      R"(// AUTOMATICALLY GENERATED FILE. DO NOT EDIT!
#include <vector>

#include "fake_path.h"
#include "absl/base/macros.h"
#include "absl/status/status.h"
#include "absl/types/span.h"

absl::StatusOr<InnerStruct> InnerStruct::FromValue(const xls::Value& value) {
  absl::Span<const xls::Value> elements = value.elements();
  if (elements.size() != 2) {
    return absl::InvalidArgumentError(
        "InnerStruct::FromValue input must be a 2-tuple.");
  }

  InnerStruct result;
  result.x = elements[0].bits().ToUint64().value();
  result.y = elements[1].bits().ToUint64().value();
  return result;
}

xls::Value InnerStruct::ToValue() const {
  std::vector<xls::Value> elements;
  xls::Value x_value(xls::UBits(x, /*bit_count=*/32));
  elements.push_back(x_value);
  xls::Value y_value(xls::UBits(y, /*bit_count=*/16));
  elements.push_back(y_value);
  return xls::Value::Tuple(elements);
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

absl::StatusOr<OuterStruct> OuterStruct::FromValue(const xls::Value& value) {
  absl::Span<const xls::Value> elements = value.elements();
  if (elements.size() != 3) {
    return absl::InvalidArgumentError(
        "OuterStruct::FromValue input must be a 3-tuple.");
  }

  OuterStruct result;
  result.x = elements[0].bits().ToUint64().value();
  auto a_or = InnerStruct::FromValue(elements[1]);
  if (!a_or.ok()) {
    return a_or.status();
  }
  result.a = a_or.value();

  auto b_or = InnerStruct::FromValue(elements[2]);
  if (!b_or.ok()) {
    return b_or.status();
  }
  result.b = b_or.value();

  return result;
}

xls::Value OuterStruct::ToValue() const {
  std::vector<xls::Value> elements;
  xls::Value x_value(xls::UBits(x, /*bit_count=*/32));
  elements.push_back(x_value);
  elements.push_back(a.ToValue());
  elements.push_back(b.ToValue());
  return xls::Value::Tuple(elements);
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
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, TranspileToCpp(module.module, &import_data, "fake_path.h"));
  EXPECT_EQ(result.header, kExpectedHeader);
  EXPECT_EQ(result.body, kExpectedBody);
}

TEST(CppTranspilerTest, StructWithStructWithStruct) {
  constexpr std::string_view kModule = R"(
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

  constexpr std::string_view kExpectedHeader =
      R"(// AUTOMATICALLY GENERATED FILE. DO NOT EDIT!
#ifndef FAKE_PATH_H_
#define FAKE_PATH_H_
#include <cstdint>
#include <ostream>

#include "absl/status/statusor.h"
#include "xls/public/value.h"

struct InnerStruct {
  static absl::StatusOr<InnerStruct> FromValue(const xls::Value& value);

  xls::Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const InnerStruct& data);

  uint32_t x;
  uint16_t y;

  static constexpr int64_t kXWidth = 32;
  static constexpr int64_t kYWidth = 16;
};

struct MiddleStruct {
  static absl::StatusOr<MiddleStruct> FromValue(const xls::Value& value);

  xls::Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const MiddleStruct& data);

  uint64_t z;
  InnerStruct a;

  static constexpr int64_t kZWidth = 48;
};

struct OtherMiddleStruct {
  static absl::StatusOr<OtherMiddleStruct> FromValue(const xls::Value& value);

  xls::Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const OtherMiddleStruct& data);

  InnerStruct b;
  uint64_t w;

  static constexpr int64_t kWWidth = 64;
};

struct OuterStruct {
  static absl::StatusOr<OuterStruct> FromValue(const xls::Value& value);

  xls::Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const OuterStruct& data);

  InnerStruct a;
  MiddleStruct b;
  OtherMiddleStruct c;
  uint8_t v;

  static constexpr int64_t kVWidth = 8;
};

#endif  // FAKE_PATH_H_
)";

  constexpr std::string_view kExpectedBody =
      R"(// AUTOMATICALLY GENERATED FILE. DO NOT EDIT!
#include <vector>

#include "fake_path.h"
#include "absl/base/macros.h"
#include "absl/status/status.h"
#include "absl/types/span.h"

absl::StatusOr<InnerStruct> InnerStruct::FromValue(const xls::Value& value) {
  absl::Span<const xls::Value> elements = value.elements();
  if (elements.size() != 2) {
    return absl::InvalidArgumentError(
        "InnerStruct::FromValue input must be a 2-tuple.");
  }

  InnerStruct result;
  result.x = elements[0].bits().ToUint64().value();
  result.y = elements[1].bits().ToUint64().value();
  return result;
}

xls::Value InnerStruct::ToValue() const {
  std::vector<xls::Value> elements;
  xls::Value x_value(xls::UBits(x, /*bit_count=*/32));
  elements.push_back(x_value);
  xls::Value y_value(xls::UBits(y, /*bit_count=*/16));
  elements.push_back(y_value);
  return xls::Value::Tuple(elements);
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

absl::StatusOr<MiddleStruct> MiddleStruct::FromValue(const xls::Value& value) {
  absl::Span<const xls::Value> elements = value.elements();
  if (elements.size() != 2) {
    return absl::InvalidArgumentError(
        "MiddleStruct::FromValue input must be a 2-tuple.");
  }

  MiddleStruct result;
  result.z = elements[0].bits().ToUint64().value();
  auto a_or = InnerStruct::FromValue(elements[1]);
  if (!a_or.ok()) {
    return a_or.status();
  }
  result.a = a_or.value();

  return result;
}

xls::Value MiddleStruct::ToValue() const {
  std::vector<xls::Value> elements;
  xls::Value z_value(xls::UBits(z, /*bit_count=*/48));
  elements.push_back(z_value);
  elements.push_back(a.ToValue());
  return xls::Value::Tuple(elements);
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

absl::StatusOr<OtherMiddleStruct> OtherMiddleStruct::FromValue(const xls::Value& value) {
  absl::Span<const xls::Value> elements = value.elements();
  if (elements.size() != 2) {
    return absl::InvalidArgumentError(
        "OtherMiddleStruct::FromValue input must be a 2-tuple.");
  }

  OtherMiddleStruct result;
  auto b_or = InnerStruct::FromValue(elements[0]);
  if (!b_or.ok()) {
    return b_or.status();
  }
  result.b = b_or.value();

  result.w = elements[1].bits().ToUint64().value();
  return result;
}

xls::Value OtherMiddleStruct::ToValue() const {
  std::vector<xls::Value> elements;
  elements.push_back(b.ToValue());
  xls::Value w_value(xls::UBits(w, /*bit_count=*/64));
  elements.push_back(w_value);
  return xls::Value::Tuple(elements);
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

absl::StatusOr<OuterStruct> OuterStruct::FromValue(const xls::Value& value) {
  absl::Span<const xls::Value> elements = value.elements();
  if (elements.size() != 4) {
    return absl::InvalidArgumentError(
        "OuterStruct::FromValue input must be a 4-tuple.");
  }

  OuterStruct result;
  auto a_or = InnerStruct::FromValue(elements[0]);
  if (!a_or.ok()) {
    return a_or.status();
  }
  result.a = a_or.value();

  auto b_or = MiddleStruct::FromValue(elements[1]);
  if (!b_or.ok()) {
    return b_or.status();
  }
  result.b = b_or.value();

  auto c_or = OtherMiddleStruct::FromValue(elements[2]);
  if (!c_or.ok()) {
    return c_or.status();
  }
  result.c = c_or.value();

  result.v = elements[3].bits().ToUint64().value();
  return result;
}

xls::Value OuterStruct::ToValue() const {
  std::vector<xls::Value> elements;
  elements.push_back(a.ToValue());
  elements.push_back(b.ToValue());
  elements.push_back(c.ToValue());
  xls::Value v_value(xls::UBits(v, /*bit_count=*/8));
  elements.push_back(v_value);
  return xls::Value::Tuple(elements);
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
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, TranspileToCpp(module.module, &import_data, "fake_path.h"));
  EXPECT_EQ(result.header, kExpectedHeader);
  EXPECT_EQ(result.body, kExpectedBody);
}

TEST(CppTranspilerTest, HandlesAbsolutePaths) {
  const std::string kModule = R"(
pub enum MyEnum : u32 {
  A = 0,
  B = 1,
  C = 42,
  // D = 4294967296,
  E = 4294967295
}
)";

  const std::string kExpected =
      R"(// AUTOMATICALLY GENERATED FILE. DO NOT EDIT!
#ifndef TMP_FAKE_PATH_H_
#define TMP_FAKE_PATH_H_
#include <cstdint>
#include <ostream>

#include "absl/status/statusor.h"
#include "xls/public/value.h"

enum class MyEnum {
  kA = 0,
  kB = 1,
  kC = 42,
  kE = 4294967295,
};
constexpr int64_t kMyEnumNumElements = 4;

#endif  // TMP_FAKE_PATH_H_
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result,
      TranspileToCpp(module.module, &import_data, "/tmp/fake_path.h"));
  ASSERT_EQ(result.header, kExpected);
}

}  // namespace
}  // namespace xls::dslx
