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

#include <filesystem>  // NOLINT
#include <string>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_format.h"
#include "xls/common/golden_files.h"
#include "xls/common/source_location.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/cpp_transpiler/cpp_type_generator.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {

namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

constexpr char kTestdataPath[] = "xls/dslx/cpp_transpiler/testdata";

void ExpectEqualToGoldenFiles(
    const CppSource& sources,
    xabsl::SourceLocation loc = xabsl::SourceLocation::current()) {
  const std::filesystem::path header_path = absl::StrFormat(
      "%s/%s.htxt", kTestdataPath,
      ::testing::UnitTest::GetInstance()->current_test_info()->name());
  ExpectEqualToGoldenFile(header_path, sources.header, loc);
  const std::filesystem::path source_path = absl::StrFormat(
      "%s/%s.cctxt", kTestdataPath,
      ::testing::UnitTest::GetInstance()->current_test_info()->name());
  ExpectEqualToGoldenFile(source_path, sources.source, loc);
}

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

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, TranspileToCpp(module.module, &import_data, "fake_path.h"));
  ExpectEqualToGoldenFiles(result);
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

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, TranspileToCpp(module.module, &import_data, "fake_path.h"));
  ExpectEqualToGoldenFiles(result);
}

TEST(CppTranspilerTest, EnumWithS64) {
  const std::string kModule = R"(
pub enum MyEnum : s64 {
  MIN = s64:0x8000000000000000,
  MID = s64:1 << 62,
  MAX = s64::MAX,
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, TranspileToCpp(module.module, &import_data, "fake_path.h"));
  ExpectEqualToGoldenFiles(result);
}

TEST(CppTranspilerTest, EnumWithU64) {
  const std::string kModule = R"(
pub enum MyEnum : u64 {
  MIN = u64:0,
  MID = u64:1 << 63,
  MAX = u64::MAX,
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, TranspileToCpp(module.module, &import_data, "fake_path.h"));
  ExpectEqualToGoldenFiles(result);
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

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, TranspileToCpp(module.module, &import_data, "fake_path.h",
                                  "robs::secret::space"));
  ExpectEqualToGoldenFiles(result);
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

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, TranspileToCpp(module.module, &import_data, "fake_path.h"));
  ExpectEqualToGoldenFiles(result);
}

TEST(CppTranspilerTest, BasicArray) {
  constexpr std::string_view kModule = R"(
struct MyStruct {
  x: u32[32],
  y: s7[8],
  z: u8[7],
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, TranspileToCpp(module.module, &import_data, "fake_path.h"));
  ExpectEqualToGoldenFiles(result);
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

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, TranspileToCpp(module.module, &import_data, "fake_path.h"));
  ExpectEqualToGoldenFiles(result);
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

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, TranspileToCpp(module.module, &import_data, "fake_path.h"));
  ExpectEqualToGoldenFiles(result);
}

TEST(CppTranspilerTest, HandlesAbsolutePaths) {
  const std::string kModule = R"(
pub enum MyEnum : u34 {
  A = 0,
  B = 1,
  C = 42,
  // D = 4294967296,
  E = 4294967295
}
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result,
      TranspileToCpp(module.module, &import_data, "/tmp/fake_path.h"));
  ExpectEqualToGoldenFiles(result);
}

TEST(CppTranspilerTest, StructWithTuples) {
  constexpr std::string_view kModule = R"(
struct Foo {
    a: (u32, u32),
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result,
      TranspileToCpp(module.module, &import_data, "/tmp/fake_path.h"));
  ExpectEqualToGoldenFiles(result);
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

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result,
      TranspileToCpp(module.module, &import_data, "/tmp/fake_path.h"));
  ExpectEqualToGoldenFiles(result);
}

TEST(CppTranspilerTest, UnsupportedS1) {
  constexpr std::string_view kModule = R"(
type MyUnsupportedSignedBit = s1;
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  EXPECT_THAT(TranspileToCpp(module.module, &import_data, "/tmp/fake_path.h"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("Signed one-bit numbers are not supported")));
}

TEST(CppTranspilerTest, UnsupportedTypes) {
  constexpr std::string_view kModule = R"(
type MyUnsupportedWideAlias = sN[123];

enum MyUnsupportedWideEnum : uN[555] {
  A = 0,
  B = 1,
}

struct MyUnsupportedWideStruct {
  wide_field: bits[100],
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kModule, "fake_path", "MyModule", &import_data));
  // The types compile, but the code is wrong.
  // TODO(https://github.com/google/xls/issues/1135): Fix this.
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result,
      TranspileToCpp(module.module, &import_data, "/tmp/fake_path.h"));
  EXPECT_THAT(result.header,
              HasSubstr("using MyUnsupportedWideAlias = int64_t;"));
  EXPECT_THAT(result.header, HasSubstr("uint64_t wide_field"));
  EXPECT_THAT(result.header, HasSubstr("uint64_t wide_field"));
  EXPECT_THAT(result.header,
              HasSubstr("enum class MyUnsupportedWideEnum : uint64_t"));
}

}  // namespace
}  // namespace xls::dslx
