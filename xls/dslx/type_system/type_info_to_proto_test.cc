// Copyright 2021 The XLS Authors
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

#include "xls/dslx/type_system/type_info_to_proto.h"

#include <filesystem>  // NOLINT
#include <optional>
#include <string>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/golden_files.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/type_info.pb.h"

namespace xls::dslx {
namespace {

std::string TestName() {
  return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

void DoRun(std::string_view program, const std::string& test_name,
           TypeInfoProto* proto_out = nullptr,
           ImportData* import_data = nullptr) {
  std::optional<ImportData> local_import_data;
  if (import_data == nullptr) {
    local_import_data.emplace(CreateImportDataForTest());
    import_data = &local_import_data.value();
  }
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "fake.x", "fake", import_data));

  XLS_ASSERT_OK_AND_ASSIGN(TypeInfoProto tip, TypeInfoToProto(*tm.type_info));

  std::string nodes_text = absl::StrJoin(
      tip.nodes(), "\n",
      [&](std::string* out, const AstNodeTypeInfoProto& node) {
        absl::StrAppend(
            out, ToHumanString(node, *import_data, import_data->file_table())
                     .value());
      });

  std::filesystem::path golden_file_path = absl::StrFormat(
      "xls/dslx/type_system/testdata/type_info_to_proto_test_%s.txt",
      test_name);
  ExpectEqualToGoldenFile(golden_file_path, nodes_text);

  if (proto_out != nullptr) {
    *proto_out = tip;
  }
}

TEST(TypeInfoToProtoTest, IdentityFunction) {
  std::string program = R"(fn id(x: u32) -> u32 { x })";
  DoRun(program, TestName());
}

TEST(TypeInfoToProtoTest, ParametricIdentityFunction) {
  std::string program = R"(
fn pid<N: u32>(x: bits[N]) -> bits[N] { x }
fn id(x: u32) -> u32 { pid<u32:32>(x) }
)";
  DoRun(program, TestName());
}

TEST(TypeInfoToProtoTest, UnitFunction) {
  std::string program = R"(fn f() -> () { () })";
  DoRun(program, TestName());
}

TEST(TypeInfoToProtoTest, ArrayFunction) {
  std::string program = R"(fn f() -> u8[2] { u8[2]:[u8:1, u8:2] })";
  DoRun(program, TestName());
}

TEST(TypeInfoToProtoTest, TokenFunction) {
  std::string program = R"(fn f(x: token) -> token { x })";
  DoRun(program, TestName());
}

TEST(TypeInfoToProtoTest, MakeStructInstanceFunction) {
  std::string program = R"(
struct S { x: u32 }
fn f() -> S { S { x: u32:42 } }
)";
  TypeInfoProto tip;
  DoRun(program, TestName(), &tip);
  EXPECT_THAT(
      tip.ShortDebugString(),
      ::testing::ContainsRegex(
          R"(struct_def \{ span \{ .*? \} identifier: "S" member_names: "x" is_public: false \})"));
}

TEST(TypeInfoToProtoTest, MakeEnumFunction) {
  std::string program = R"(
enum E : u32 { A = 42 }
fn f() -> E { E::A }
)";
  DoRun(program, TestName());
}

TEST(TypeInfoToProtoTest, ImportModuleAndTypeAliasAnEnum) {
  std::string imported = R"(
pub enum Foo : u32 {
  A = 42,
}
)";

  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(imported, "my_imported_module.x", "my_imported_module",
                        &import_data));
  (void)tm;

  std::string program = R"(
import my_imported_module;

type MyFoo = my_imported_module::Foo;
)";
  DoRun(program, TestName(), /*proto_out=*/nullptr,
        /*import_data=*/&import_data);
}

TEST(TypeInfoToProtoTest, ProcWithImpl) {
  std::string program = R"(
proc Foo { a: u32 }
)";
  DoRun(program, TestName());
}

TEST(TypeInfoToProtoTest, BitsConstructorTypeProto) {
  std::string program = R"(
fn distinct<COUNT: u32, N: u32, S: bool>(items: xN[S][N][COUNT], valid: bool[COUNT]) -> bool { fail!("unimplemented", zero!<bool>()) }

#[test]
fn test_simple_nondistinct() {
    assert_eq(distinct(u2[2]:[1, 1], bool[2]:[true, true]), false)
}
)";
  DoRun(program, TestName());
}

}  // namespace
}  // namespace xls::dslx
