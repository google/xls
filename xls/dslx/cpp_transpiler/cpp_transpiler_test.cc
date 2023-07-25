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

#include "xls/dslx/cpp_transpiler/cpp_transpiler.h"

#include <string>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {

using testing::HasSubstr;
using xls::status_testing::StatusIs;

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
      R"(// AUTOMATICALLY GENERATED FILE FROM `xls/dslx/cpp_transpiler`. DO NOT EDIT!
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

  const std::string kExpected =
      R"(// AUTOMATICALLY GENERATED FILE FROM `xls/dslx/cpp_transpiler`. DO NOT EDIT!
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

TEST(CppTranspilerTest, EnumWithS64) {
  const std::string kModule = R"(
pub enum MyEnum : s64 {
  MIN = s64:0x8000000000000000,
  MID = s64:1 << 62,
  MAX = s64::MAX,
}
)";

  const std::string kExpected =
      R"(// AUTOMATICALLY GENERATED FILE FROM `xls/dslx/cpp_transpiler`. DO NOT EDIT!
#ifndef FAKE_PATH_H_
#define FAKE_PATH_H_
#include <cstdint>
#include <ostream>

#include "absl/status/statusor.h"
#include "xls/public/value.h"

enum class MyEnum {
  kMIN = -9223372036854775808,
  kMID = 4611686018427387904,
  kMAX = 9223372036854775807,
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

TEST(CppTranspilerTest, EnumWithU64) {
  const std::string kModule = R"(
pub enum MyEnum : u64 {
  MIN = u64:0,
  MID = u64:1 << 63,
  MAX = u64::MAX,
}
)";

  const std::string kExpected =
      R"(// AUTOMATICALLY GENERATED FILE FROM `xls/dslx/cpp_transpiler`. DO NOT EDIT!
#ifndef FAKE_PATH_H_
#define FAKE_PATH_H_
#include <cstdint>
#include <ostream>

#include "absl/status/statusor.h"
#include "xls/public/value.h"

enum class MyEnum {
  kMIN = 0,
  kMID = 9223372036854775808,
  kMAX = 18446744073709551615,
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
type MyArrayType5 = bits[1];

type MyFirstTuple = (u7, s8, MyType, MySignedType, MyArrayType1, MyArrayType2);
)";

  const std::string kExpected =
      R"(// AUTOMATICALLY GENERATED FILE FROM `xls/dslx/cpp_transpiler`. DO NOT EDIT!
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

using MyArrayType5 = bool;

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
  v: u1,
})";

  const std::string kExpectedHeader =
      R"(// AUTOMATICALLY GENERATED FILE FROM `xls/dslx/cpp_transpiler`. DO NOT EDIT!
#ifndef FAKE_PATH_H_
#define FAKE_PATH_H_
#include <cstdint>
#include <ostream>

#include "absl/status/statusor.h"
#include "xls/public/value.h"

struct MyStruct {
  static absl::StatusOr<MyStruct> FromValue(const ::xls::Value& value);

  ::xls::Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const MyStruct& data);

  uint32_t x;
  uint16_t y;
  uint8_t z;
  int64_t w;
  bool v;

  static constexpr int64_t kXWidth = 32;
  static constexpr int64_t kYWidth = 15;
  static constexpr int64_t kZWidth = 8;
  static constexpr int64_t kWWidth = 63;
  static constexpr int64_t kVWidth = 1;
};

#endif  // FAKE_PATH_H_
)";

  constexpr std::string_view kExpectedBody =
      R"(// AUTOMATICALLY GENERATED FILE FROM `xls/dslx/cpp_transpiler`. DO NOT EDIT!
#include <vector>

#include "fake_path.h"
#include "absl/base/macros.h"
#include "absl/status/status.h"
#include "absl/types/span.h"

absl::StatusOr<MyStruct> MyStruct::FromValue(const ::xls::Value& value) {
  if (value.size() != 5) {
    return absl::InvalidArgumentError(
        "MyStruct::FromValue input must be a 5-tuple.");
  }

  MyStruct result;
  result.x = value.element(0).bits().ToUint64().value();
  result.y = value.element(1).bits().ToUint64().value();
  result.z = value.element(2).bits().ToUint64().value();
  result.w = value.element(3).bits().ToInt64().value();
  result.v = value.element(4).bits().ToUint64().value();
  return result;
}

::xls::Value MyStruct::ToValue() const {
  std::vector<::xls::Value> members;
  members.reserve(5);
  ::xls::Value x_value;
  x_value = ::xls::Value(::xls::UBits(x, /*bit_count=*/32));
  members.push_back(x_value);
  ::xls::Value y_value;
  y_value = ::xls::Value(::xls::UBits(y, /*bit_count=*/15));
  members.push_back(y_value);
  ::xls::Value z_value;
  z_value = ::xls::Value(::xls::UBits(z, /*bit_count=*/8));
  members.push_back(z_value);
  ::xls::Value w_value;
  w_value = ::xls::Value(::xls::SBits(w, /*bit_count=*/63));
  members.push_back(w_value);
  ::xls::Value v_value;
  v_value = ::xls::Value(::xls::UBits(v, /*bit_count=*/1));
  members.push_back(v_value);
  return ::xls::Value::Tuple(members);
}

std::ostream& operator<<(std::ostream& os, const MyStruct& data) {
  ::xls::Value value = data.ToValue();
  absl::Span<const ::xls::Value> elements = value.elements();
  os << "(\n";
  os << "  x: " << elements[0].ToString() << "\n";
  os << "  y: " << elements[1].ToString() << "\n";
  os << "  z: " << elements[2].ToString() << "\n";
  os << "  w: " << elements[3].ToString() << "\n";
  os << "  v: " << elements[4].ToString() << "\n";
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
      R"(// AUTOMATICALLY GENERATED FILE FROM `xls/dslx/cpp_transpiler`. DO NOT EDIT!
#ifndef FAKE_PATH_H_
#define FAKE_PATH_H_
#include <cstdint>
#include <ostream>

#include "absl/status/statusor.h"
#include "xls/public/value.h"

struct MyStruct {
  static absl::StatusOr<MyStruct> FromValue(const ::xls::Value& value);

  ::xls::Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const MyStruct& data);

  uint32_t x[32];
  int8_t y[8];
  uint8_t z[7];
};

#endif  // FAKE_PATH_H_
)";

  constexpr std::string_view kExpectedBody =
      R"(// AUTOMATICALLY GENERATED FILE FROM `xls/dslx/cpp_transpiler`. DO NOT EDIT!
#include <vector>

#include "fake_path.h"
#include "absl/base/macros.h"
#include "absl/status/status.h"
#include "absl/types/span.h"

absl::StatusOr<MyStruct> MyStruct::FromValue(const ::xls::Value& value) {
  if (value.size() != 3) {
    return absl::InvalidArgumentError(
        "MyStruct::FromValue input must be a 3-tuple.");
  }

  MyStruct result;
  for (int idx_1 = 0; idx_1 < 32; idx_1++) {
    result.x[idx_1] = value.element(0).element(idx_1).bits().ToUint64().value();
  }
  for (int idx_1 = 0; idx_1 < 8; idx_1++) {
    result.y[idx_1] = value.element(1).element(idx_1).bits().ToInt64().value();
  }
  for (int idx_1 = 0; idx_1 < 7; idx_1++) {
    result.z[idx_1] = value.element(2).element(idx_1).bits().ToUint64().value();
  }
  return result;
}

::xls::Value MyStruct::ToValue() const {
  std::vector<::xls::Value> members;
  members.reserve(3);
  ::xls::Value x_value;
  std::vector<::xls::Value> x_elements;
  x_elements.reserve(ABSL_ARRAYSIZE(x));
  for (int idx_1 = 0; idx_1 < ABSL_ARRAYSIZE(x); idx_1++) {
    ::xls::Value x_element;
    x_element = ::xls::Value(::xls::UBits(x[idx_1], /*bit_count=*/32));
    x_elements.push_back(x_element);
  }
  x_value = ::xls::Value::ArrayOrDie(x_elements);
  members.push_back(x_value);
  ::xls::Value y_value;
  std::vector<::xls::Value> y_elements;
  y_elements.reserve(ABSL_ARRAYSIZE(y));
  for (int idx_1 = 0; idx_1 < ABSL_ARRAYSIZE(y); idx_1++) {
    ::xls::Value y_element;
    y_element = ::xls::Value(::xls::SBits(y[idx_1], /*bit_count=*/7));
    y_elements.push_back(y_element);
  }
  y_value = ::xls::Value::ArrayOrDie(y_elements);
  members.push_back(y_value);
  ::xls::Value z_value;
  std::vector<::xls::Value> z_elements;
  z_elements.reserve(ABSL_ARRAYSIZE(z));
  for (int idx_1 = 0; idx_1 < ABSL_ARRAYSIZE(z); idx_1++) {
    ::xls::Value z_element;
    z_element = ::xls::Value(::xls::UBits(z[idx_1], /*bit_count=*/8));
    z_elements.push_back(z_element);
  }
  z_value = ::xls::Value::ArrayOrDie(z_elements);
  members.push_back(z_value);
  return ::xls::Value::Tuple(members);
}

std::ostream& operator<<(std::ostream& os, const MyStruct& data) {
  ::xls::Value value = data.ToValue();
  absl::Span<const ::xls::Value> elements = value.elements();
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
      R"(// AUTOMATICALLY GENERATED FILE FROM `xls/dslx/cpp_transpiler`. DO NOT EDIT!
#ifndef FAKE_PATH_H_
#define FAKE_PATH_H_
#include <cstdint>
#include <ostream>

#include "absl/status/statusor.h"
#include "xls/public/value.h"

struct InnerStruct {
  static absl::StatusOr<InnerStruct> FromValue(const ::xls::Value& value);

  ::xls::Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const InnerStruct& data);

  uint32_t x;
  uint16_t y;

  static constexpr int64_t kXWidth = 32;
  static constexpr int64_t kYWidth = 16;
};

struct OuterStruct {
  static absl::StatusOr<OuterStruct> FromValue(const ::xls::Value& value);

  ::xls::Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const OuterStruct& data);

  uint32_t x;
  InnerStruct a;
  InnerStruct b;

  static constexpr int64_t kXWidth = 32;
};

#endif  // FAKE_PATH_H_
)";
  constexpr std::string_view kExpectedBody =
      R"(// AUTOMATICALLY GENERATED FILE FROM `xls/dslx/cpp_transpiler`. DO NOT EDIT!
#include <vector>

#include "fake_path.h"
#include "absl/base/macros.h"
#include "absl/status/status.h"
#include "absl/types/span.h"

absl::StatusOr<InnerStruct> InnerStruct::FromValue(const ::xls::Value& value) {
  if (value.size() != 2) {
    return absl::InvalidArgumentError(
        "InnerStruct::FromValue input must be a 2-tuple.");
  }

  InnerStruct result;
  result.x = value.element(0).bits().ToUint64().value();
  result.y = value.element(1).bits().ToUint64().value();
  return result;
}

::xls::Value InnerStruct::ToValue() const {
  std::vector<::xls::Value> members;
  members.reserve(2);
  ::xls::Value x_value;
  x_value = ::xls::Value(::xls::UBits(x, /*bit_count=*/32));
  members.push_back(x_value);
  ::xls::Value y_value;
  y_value = ::xls::Value(::xls::UBits(y, /*bit_count=*/16));
  members.push_back(y_value);
  return ::xls::Value::Tuple(members);
}

std::ostream& operator<<(std::ostream& os, const InnerStruct& data) {
  ::xls::Value value = data.ToValue();
  absl::Span<const ::xls::Value> elements = value.elements();
  os << "(\n";
  os << "  x: " << elements[0].ToString() << "\n";
  os << "  y: " << elements[1].ToString() << "\n";
  os << ")\n";
  return os;
}

absl::StatusOr<OuterStruct> OuterStruct::FromValue(const ::xls::Value& value) {
  if (value.size() != 3) {
    return absl::InvalidArgumentError(
        "OuterStruct::FromValue input must be a 3-tuple.");
  }

  OuterStruct result;
  result.x = value.element(0).bits().ToUint64().value();
  auto result_a_or = InnerStruct::FromValue(value.element(1));
  if (!result_a_or.ok()) {
    return result_a_or.status();
  }
  result.a = result_a_or.value();
  auto result_b_or = InnerStruct::FromValue(value.element(2));
  if (!result_b_or.ok()) {
    return result_b_or.status();
  }
  result.b = result_b_or.value();
  return result;
}

::xls::Value OuterStruct::ToValue() const {
  std::vector<::xls::Value> members;
  members.reserve(3);
  ::xls::Value x_value;
  x_value = ::xls::Value(::xls::UBits(x, /*bit_count=*/32));
  members.push_back(x_value);
  ::xls::Value a_value;
  a_value = a.ToValue();
  members.push_back(a_value);
  ::xls::Value b_value;
  b_value = b.ToValue();
  members.push_back(b_value);
  return ::xls::Value::Tuple(members);
}

std::ostream& operator<<(std::ostream& os, const OuterStruct& data) {
  ::xls::Value value = data.ToValue();
  absl::Span<const ::xls::Value> elements = value.elements();
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
      R"(// AUTOMATICALLY GENERATED FILE FROM `xls/dslx/cpp_transpiler`. DO NOT EDIT!
#ifndef FAKE_PATH_H_
#define FAKE_PATH_H_
#include <cstdint>
#include <ostream>

#include "absl/status/statusor.h"
#include "xls/public/value.h"

struct InnerStruct {
  static absl::StatusOr<InnerStruct> FromValue(const ::xls::Value& value);

  ::xls::Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const InnerStruct& data);

  uint32_t x;
  uint16_t y;

  static constexpr int64_t kXWidth = 32;
  static constexpr int64_t kYWidth = 16;
};

struct MiddleStruct {
  static absl::StatusOr<MiddleStruct> FromValue(const ::xls::Value& value);

  ::xls::Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const MiddleStruct& data);

  uint64_t z;
  InnerStruct a;

  static constexpr int64_t kZWidth = 48;
};

struct OtherMiddleStruct {
  static absl::StatusOr<OtherMiddleStruct> FromValue(const ::xls::Value& value);

  ::xls::Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const OtherMiddleStruct& data);

  InnerStruct b;
  uint64_t w;

  static constexpr int64_t kWWidth = 64;
};

struct OuterStruct {
  static absl::StatusOr<OuterStruct> FromValue(const ::xls::Value& value);

  ::xls::Value ToValue() const;

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
      R"(// AUTOMATICALLY GENERATED FILE FROM `xls/dslx/cpp_transpiler`. DO NOT EDIT!
#include <vector>

#include "fake_path.h"
#include "absl/base/macros.h"
#include "absl/status/status.h"
#include "absl/types/span.h"

absl::StatusOr<InnerStruct> InnerStruct::FromValue(const ::xls::Value& value) {
  if (value.size() != 2) {
    return absl::InvalidArgumentError(
        "InnerStruct::FromValue input must be a 2-tuple.");
  }

  InnerStruct result;
  result.x = value.element(0).bits().ToUint64().value();
  result.y = value.element(1).bits().ToUint64().value();
  return result;
}

::xls::Value InnerStruct::ToValue() const {
  std::vector<::xls::Value> members;
  members.reserve(2);
  ::xls::Value x_value;
  x_value = ::xls::Value(::xls::UBits(x, /*bit_count=*/32));
  members.push_back(x_value);
  ::xls::Value y_value;
  y_value = ::xls::Value(::xls::UBits(y, /*bit_count=*/16));
  members.push_back(y_value);
  return ::xls::Value::Tuple(members);
}

std::ostream& operator<<(std::ostream& os, const InnerStruct& data) {
  ::xls::Value value = data.ToValue();
  absl::Span<const ::xls::Value> elements = value.elements();
  os << "(\n";
  os << "  x: " << elements[0].ToString() << "\n";
  os << "  y: " << elements[1].ToString() << "\n";
  os << ")\n";
  return os;
}

absl::StatusOr<MiddleStruct> MiddleStruct::FromValue(const ::xls::Value& value) {
  if (value.size() != 2) {
    return absl::InvalidArgumentError(
        "MiddleStruct::FromValue input must be a 2-tuple.");
  }

  MiddleStruct result;
  result.z = value.element(0).bits().ToUint64().value();
  auto result_a_or = InnerStruct::FromValue(value.element(1));
  if (!result_a_or.ok()) {
    return result_a_or.status();
  }
  result.a = result_a_or.value();
  return result;
}

::xls::Value MiddleStruct::ToValue() const {
  std::vector<::xls::Value> members;
  members.reserve(2);
  ::xls::Value z_value;
  z_value = ::xls::Value(::xls::UBits(z, /*bit_count=*/48));
  members.push_back(z_value);
  ::xls::Value a_value;
  a_value = a.ToValue();
  members.push_back(a_value);
  return ::xls::Value::Tuple(members);
}

std::ostream& operator<<(std::ostream& os, const MiddleStruct& data) {
  ::xls::Value value = data.ToValue();
  absl::Span<const ::xls::Value> elements = value.elements();
  os << "(\n";
  os << "  z: " << elements[0].ToString() << "\n";
  os << "  a: " << elements[1].ToString() << "\n";
  os << ")\n";
  return os;
}

absl::StatusOr<OtherMiddleStruct> OtherMiddleStruct::FromValue(const ::xls::Value& value) {
  if (value.size() != 2) {
    return absl::InvalidArgumentError(
        "OtherMiddleStruct::FromValue input must be a 2-tuple.");
  }

  OtherMiddleStruct result;
  auto result_b_or = InnerStruct::FromValue(value.element(0));
  if (!result_b_or.ok()) {
    return result_b_or.status();
  }
  result.b = result_b_or.value();
  result.w = value.element(1).bits().ToUint64().value();
  return result;
}

::xls::Value OtherMiddleStruct::ToValue() const {
  std::vector<::xls::Value> members;
  members.reserve(2);
  ::xls::Value b_value;
  b_value = b.ToValue();
  members.push_back(b_value);
  ::xls::Value w_value;
  w_value = ::xls::Value(::xls::UBits(w, /*bit_count=*/64));
  members.push_back(w_value);
  return ::xls::Value::Tuple(members);
}

std::ostream& operator<<(std::ostream& os, const OtherMiddleStruct& data) {
  ::xls::Value value = data.ToValue();
  absl::Span<const ::xls::Value> elements = value.elements();
  os << "(\n";
  os << "  b: " << elements[0].ToString() << "\n";
  os << "  w: " << elements[1].ToString() << "\n";
  os << ")\n";
  return os;
}

absl::StatusOr<OuterStruct> OuterStruct::FromValue(const ::xls::Value& value) {
  if (value.size() != 4) {
    return absl::InvalidArgumentError(
        "OuterStruct::FromValue input must be a 4-tuple.");
  }

  OuterStruct result;
  auto result_a_or = InnerStruct::FromValue(value.element(0));
  if (!result_a_or.ok()) {
    return result_a_or.status();
  }
  result.a = result_a_or.value();
  auto result_b_or = MiddleStruct::FromValue(value.element(1));
  if (!result_b_or.ok()) {
    return result_b_or.status();
  }
  result.b = result_b_or.value();
  auto result_c_or = OtherMiddleStruct::FromValue(value.element(2));
  if (!result_c_or.ok()) {
    return result_c_or.status();
  }
  result.c = result_c_or.value();
  result.v = value.element(3).bits().ToUint64().value();
  return result;
}

::xls::Value OuterStruct::ToValue() const {
  std::vector<::xls::Value> members;
  members.reserve(4);
  ::xls::Value a_value;
  a_value = a.ToValue();
  members.push_back(a_value);
  ::xls::Value b_value;
  b_value = b.ToValue();
  members.push_back(b_value);
  ::xls::Value c_value;
  c_value = c.ToValue();
  members.push_back(c_value);
  ::xls::Value v_value;
  v_value = ::xls::Value(::xls::UBits(v, /*bit_count=*/8));
  members.push_back(v_value);
  return ::xls::Value::Tuple(members);
}

std::ostream& operator<<(std::ostream& os, const OuterStruct& data) {
  ::xls::Value value = data.ToValue();
  absl::Span<const ::xls::Value> elements = value.elements();
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
      R"(// AUTOMATICALLY GENERATED FILE FROM `xls/dslx/cpp_transpiler`. DO NOT EDIT!
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

TEST(CppTranspilerTest, UnhandledTupleType) {
  constexpr std::string_view kModule = R"(
struct Foo {
    a: (u32, u32),
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  EXPECT_THAT(TranspileToCpp(module.module, &import_data, "/tmp/fake_path.h"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("Unknown/unsupported")));
}

TEST(CppTranspilerTest, ArrayOfTyperefs) {
  constexpr std::string_view kModule = R"(
struct Foo {
    a: u32,
    b: u64,
}

struct Bar {
    c: Foo[2],
}
)";

  const std::string kExpectedHeader =
      R"(// AUTOMATICALLY GENERATED FILE FROM `xls/dslx/cpp_transpiler`. DO NOT EDIT!
#ifndef TMP_FAKE_PATH_H_
#define TMP_FAKE_PATH_H_
#include <cstdint>
#include <ostream>

#include "absl/status/statusor.h"
#include "xls/public/value.h"

struct Foo {
  static absl::StatusOr<Foo> FromValue(const ::xls::Value& value);

  ::xls::Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const Foo& data);

  uint32_t a;
  uint64_t b;

  static constexpr int64_t kAWidth = 32;
  static constexpr int64_t kBWidth = 64;
};

struct Bar {
  static absl::StatusOr<Bar> FromValue(const ::xls::Value& value);

  ::xls::Value ToValue() const;

  friend std::ostream& operator<<(std::ostream& os, const Bar& data);

  Foo c[2];
};

#endif  // TMP_FAKE_PATH_H_
)";

  constexpr std::string_view kExpectedSource =
      R"(// AUTOMATICALLY GENERATED FILE FROM `xls/dslx/cpp_transpiler`. DO NOT EDIT!
#include <vector>

#include "/tmp/fake_path.h"
#include "absl/base/macros.h"
#include "absl/status/status.h"
#include "absl/types/span.h"

absl::StatusOr<Foo> Foo::FromValue(const ::xls::Value& value) {
  if (value.size() != 2) {
    return absl::InvalidArgumentError(
        "Foo::FromValue input must be a 2-tuple.");
  }

  Foo result;
  result.a = value.element(0).bits().ToUint64().value();
  result.b = value.element(1).bits().ToUint64().value();
  return result;
}

::xls::Value Foo::ToValue() const {
  std::vector<::xls::Value> members;
  members.reserve(2);
  ::xls::Value a_value;
  a_value = ::xls::Value(::xls::UBits(a, /*bit_count=*/32));
  members.push_back(a_value);
  ::xls::Value b_value;
  b_value = ::xls::Value(::xls::UBits(b, /*bit_count=*/64));
  members.push_back(b_value);
  return ::xls::Value::Tuple(members);
}

std::ostream& operator<<(std::ostream& os, const Foo& data) {
  ::xls::Value value = data.ToValue();
  absl::Span<const ::xls::Value> elements = value.elements();
  os << "(\n";
  os << "  a: " << elements[0].ToString() << "\n";
  os << "  b: " << elements[1].ToString() << "\n";
  os << ")\n";
  return os;
}

absl::StatusOr<Bar> Bar::FromValue(const ::xls::Value& value) {
  if (value.size() != 1) {
    return absl::InvalidArgumentError(
        "Bar::FromValue input must be a 1-tuple.");
  }

  Bar result;
  for (int idx_1 = 0; idx_1 < 2; idx_1++) {
    auto result_c_idx_1__or = Foo::FromValue(value.element(0).element(idx_1));
    if (!result_c_idx_1__or.ok()) {
      return result_c_idx_1__or.status();
    }
    result.c[idx_1] = result_c_idx_1__or.value();
  }
  return result;
}

::xls::Value Bar::ToValue() const {
  std::vector<::xls::Value> members;
  members.reserve(1);
  ::xls::Value c_value;
  std::vector<::xls::Value> c_elements;
  c_elements.reserve(ABSL_ARRAYSIZE(c));
  for (int idx_1 = 0; idx_1 < ABSL_ARRAYSIZE(c); idx_1++) {
    ::xls::Value c_element;
    c_element = c[idx_1].ToValue();
    c_elements.push_back(c_element);
  }
  c_value = ::xls::Value::ArrayOrDie(c_elements);
  members.push_back(c_value);
  return ::xls::Value::Tuple(members);
}

std::ostream& operator<<(std::ostream& os, const Bar& data) {
  ::xls::Value value = data.ToValue();
  absl::Span<const ::xls::Value> elements = value.elements();
  os << "(\n";
  os << "  c: " << elements[0].ToString() << "\n";
  os << ")\n";
  return os;
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result,
      TranspileToCpp(module.module, &import_data, "/tmp/fake_path.h"));
  ASSERT_EQ(result.header, kExpectedHeader);
  ASSERT_EQ(result.body, kExpectedSource);
}

}  // namespace
}  // namespace xls::dslx
