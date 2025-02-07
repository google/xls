// Copyright 2024 The XLS Authors
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

#include "xls/dslx/translators/dslx_to_verilog.h"

#include <filesystem>  // NOLINT
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "xls/common/golden_files.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/virtualizable_file_system.h"

namespace xls::dslx {
namespace {

constexpr std::string_view kTestdataPath = "xls/dslx/translators/testdata";

class DslxToVerilogTest : public ::testing::Test {
 public:
  static std::string TestName() {
    // If we try to run the program it can't have the '/' in its name. Remove
    // them so this pattern works.
    std::string name =
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    absl::StrReplaceAll(std::vector{std::pair{"/", "_"}}, &name);
    return name;
  }

  std::filesystem::path GoldenFilePath(std::string_view file_ext) {
    return absl::StrFormat("%s/dslx_to_verilog_test_%s.%s", kTestdataPath,
                           TestName(), file_ext);
  }
};

TEST_F(DslxToVerilogTest, BasicTypesInFunctions) {
  constexpr std::string_view program =
      R"(
struct Point {
  x: u16,
  y: u32,
}

enum Option : u5 {
  ZERO = 0,
  ONE = 1,
}

type AliasType = Point;
type AliasType1 = Point[1];

fn add_point_elements(p : Point, o : Option, v : u5, a : Point[3], b: u34[5], c: bits[9], d: bits[431], e: AliasType, f: AliasType[1], g: AliasType1) -> (u16, u32, u64) {
  let additional = if o == Option::ZERO { u5:0  } else  { v };
  let sum = p.x as u64 + p.y as u64 + additional as u64;
  (p.x, p.y, sum)
}
)";

  // Parse and typecheck program.
  dslx::ImportData import_data = dslx::CreateImportDataForTest();

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      dslx::ParseAndTypecheck(program, "test_module.x", "test_module",
                              &import_data));

  // Create package, add function types, and check output
  XLS_ASSERT_OK_AND_ASSIGN(DslxTypeToVerilogManager type_to_verilog,
                           DslxTypeToVerilogManager::Create("test_pkg"));

  Function* func = tm.module->GetFunction("add_point_elements").value();

  for (Param* p : func->params()) {
    XLS_ASSERT_OK(type_to_verilog.AddTypeForFunctionParam(
        func, &import_data, p->name_def()->identifier()));
  }

  XLS_ASSERT_OK(type_to_verilog.AddTypeForFunctionOutput(
      func, &import_data, "user_defined_output_type_t"));

  ExpectEqualToGoldenFile(GoldenFilePath("vtxt"), type_to_verilog.Emit());
}

TEST_F(DslxToVerilogTest, BasicTypeDefinition) {
  constexpr std::string_view program =
      R"(
struct Point {
  x: u16,
  y: u32,
}

enum Option : u5 {
  ZERO = 0,
  ONE = 1,
}

type AliasType = Point;
type AliasType1 = Point[1];
)";

  // Parse and typecheck program.
  dslx::ImportData import_data = dslx::CreateImportDataForTest();

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      dslx::ParseAndTypecheck(program, "test_module.x", "test_module",
                              &import_data));

  // Create package, add function types, and check output
  XLS_ASSERT_OK_AND_ASSIGN(DslxTypeToVerilogManager type_to_verilog,
                           DslxTypeToVerilogManager::Create("test_pkg"));

  for (const TypeDefinition& def : tm.module->GetTypeDefinitions()) {
    XLS_ASSERT_OK(type_to_verilog.AddTypeForTypeDefinition(def, &import_data));
  }

  ExpectEqualToGoldenFile(GoldenFilePath("vtxt"), type_to_verilog.Emit());
}

TEST_F(DslxToVerilogTest, NestedTypeDefinition) {
  constexpr std::string_view program =
      R"(
struct Point {
  x: u16,
  y: u32,
}

enum Option : u5 {
  ZERO = 0,
  ONE = 1,
}

type AliasType = Point;
type AliasType1 = Point[1];
type AliasType2 = uN[100];

struct TopType {
  a: Point,
  b: Option,
  c: AliasType,
  d: AliasType1,
  e: AliasType2,
}
)";

  // Parse and typecheck program.
  dslx::ImportData import_data = dslx::CreateImportDataForTest();

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      dslx::ParseAndTypecheck(program, "test_module.x", "test_module",
                              &import_data));

  // Create package, add function types, and check output
  XLS_ASSERT_OK_AND_ASSIGN(DslxTypeToVerilogManager type_to_verilog,
                           DslxTypeToVerilogManager::Create("test_pkg"));

  for (const TypeDefinition& def : tm.module->GetTypeDefinitions()) {
    XLS_ASSERT_OK(type_to_verilog.AddTypeForTypeDefinition(def, &import_data));
  }

  ExpectEqualToGoldenFile(GoldenFilePath("vtxt"), type_to_verilog.Emit());
}

TEST_F(DslxToVerilogTest, TypeWithNestedTuple) {
  constexpr std::string_view program =
      R"(
struct NestedType {
  x: (u16, u32),
  y: u32[4],
}

fn f() -> NestedType {
  zero!<NestedType>()
}
)";

  // Parse and typecheck program.
  dslx::ImportData import_data = dslx::CreateImportDataForTest();

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      dslx::ParseAndTypecheck(program, "test_module.x", "test_module",
                              &import_data));

  // Create package, add function types, and check output
  XLS_ASSERT_OK_AND_ASSIGN(DslxTypeToVerilogManager type_to_verilog,
                           DslxTypeToVerilogManager::Create("test_pkg"));

  Function* func = tm.module->GetFunction("f").value();

  XLS_ASSERT_OK(type_to_verilog.AddTypeForFunctionOutput(
      func, &import_data, "user_defined_name_t"));

  ExpectEqualToGoldenFile(GoldenFilePath("vtxt"), type_to_verilog.Emit());
}

TEST_F(DslxToVerilogTest, MultiDimTest) {
  constexpr std::string_view program =
      R"(
struct StructType {
  x: u16,
}

fn f(a : StructType[4][7], b : u32[5][8], c : bits[300][8][9]) -> u32 {
  u32:0
}
)";

  // Parse and typecheck program.
  dslx::ImportData import_data = dslx::CreateImportDataForTest();

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      dslx::ParseAndTypecheck(program, "test_module.x", "test_module",
                              &import_data));

  // Create package, add function types, and check output
  XLS_ASSERT_OK_AND_ASSIGN(DslxTypeToVerilogManager type_to_verilog,
                           DslxTypeToVerilogManager::Create("test_pkg"));

  Function* func = tm.module->GetFunction("f").value();

  XLS_ASSERT_OK(
      type_to_verilog.AddTypeForFunctionParam(func, &import_data, "a"));

  XLS_ASSERT_OK(
      type_to_verilog.AddTypeForFunctionParam(func, &import_data, "b"));

  XLS_ASSERT_OK(
      type_to_verilog.AddTypeForFunctionParam(func, &import_data, "c"));

  ExpectEqualToGoldenFile(GoldenFilePath("vtxt"), type_to_verilog.Emit());
}

TEST_F(DslxToVerilogTest, ArrayTypedefTest) {
  constexpr std::string_view program =
      R"(
struct StructType {
  x: u16,
}

type ArrayOfStructType = StructType[5];

fn f(a : ArrayOfStructType, b : ArrayOfStructType[2]) -> u32 {
  u32:0
}
)";

  // Parse and typecheck program.
  dslx::ImportData import_data = dslx::CreateImportDataForTest();

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      dslx::ParseAndTypecheck(program, "test_module.x", "test_module",
                              &import_data));

  // Create package, add function types, and check output
  XLS_ASSERT_OK_AND_ASSIGN(DslxTypeToVerilogManager type_to_verilog,
                           DslxTypeToVerilogManager::Create("test_pkg"));

  Function* func = tm.module->GetFunction("f").value();

  XLS_ASSERT_OK(
      type_to_verilog.AddTypeForFunctionParam(func, &import_data, "a"));
  XLS_ASSERT_OK(
      type_to_verilog.AddTypeForFunctionParam(func, &import_data, "b"));

  ExpectEqualToGoldenFile(GoldenFilePath("vtxt"), type_to_verilog.Emit());
}

TEST_F(DslxToVerilogTest, ImportedTypesWithTheSameName) {
  constexpr std::string_view program =
      R"(
import a;
import b;

type AType = a::ArrayOfStructType;
type BType = b::ArrayOfStructType;

)";

  constexpr std::string_view a_import =
      R"(
struct StructType {
  x: u16,
}

pub type ArrayOfStructType = StructType[5];
)";
  constexpr std::string_view b_import =
      R"(
struct StructType {
  x: u32,
}

pub type ArrayOfStructType = StructType[10];
)";

  absl::flat_hash_map<std::filesystem::path, std::string> files;
  files[std::filesystem::path("/a.x")] = a_import;
  files[std::filesystem::path("/b.x")] = b_import;
  auto vfs =
      std::make_unique<FakeFilesystem>(files, std::filesystem::path("/"));
  // Parse and typecheck program.
  dslx::ImportData import_data = dslx::CreateImportDataForTest(std::move(vfs));
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      dslx::ParseAndTypecheck(program, "test_module.x", "test_module",
                              &import_data));

  // Create package, add function types, and check output
  XLS_ASSERT_OK_AND_ASSIGN(DslxTypeToVerilogManager type_to_verilog,
                           DslxTypeToVerilogManager::Create("test_pkg"));

  for (const TypeDefinition& def : tm.module->GetTypeDefinitions()) {
    XLS_ASSERT_OK(type_to_verilog.AddTypeForTypeDefinition(def, &import_data));
  }

  ExpectEqualToGoldenFile(GoldenFilePath("vtxt"), type_to_verilog.Emit());
}

}  // namespace
}  // namespace xls::dslx
