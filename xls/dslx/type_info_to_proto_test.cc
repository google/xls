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

#include "xls/dslx/type_info_to_proto.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {
namespace {

void DoRun(std::string_view program, absl::Span<const std::string> want,
           TypeInfoProto* proto_out = nullptr,
           ImportData* import_data = nullptr) {
  std::optional<ImportData> local_import_data;
  if (import_data == nullptr) {
    local_import_data = CreateImportDataForTest();
    import_data = &local_import_data.value();
  }
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "fake.x", "fake", import_data));

  XLS_ASSERT_OK_AND_ASSIGN(TypeInfoProto tip, TypeInfoToProto(*tm.type_info));
  ASSERT_THAT(want, ::testing::SizeIs(tip.nodes_size()))
      << "want size: " << want.size() << " got size: " << tip.nodes_size();
  std::vector<std::string> got;
  for (int64_t i = 0; i < tip.nodes_size(); ++i) {
    XLS_ASSERT_OK_AND_ASSIGN(std::string node_str,
                             ToHumanString(tip.nodes(i), *import_data));
    EXPECT_EQ(node_str, want[i]) << "at index: " << i;
  }
  if (proto_out != nullptr) {
    *proto_out = tip;
  }
}

TEST(TypeInfoToProtoTest, IdentityFunction) {
  std::string program = R"(fn id(x: u32) -> u32 { x })";
  std::vector<std::string> want = {
      /*0=*/
      "0:0-0:26: FUNCTION :: `fn id(x: u32) -> u32 {\n  x\n}` :: (uN[32]) -> "
      "uN[32]",
      /*1=*/"0:3-0:5: NAME_DEF :: `id` :: (uN[32]) -> uN[32]",
      /*2=*/"0:6-0:7: NAME_DEF :: `x` :: uN[32]",
      /*3=*/"0:6-0:12: PARAM :: `x: u32` :: uN[32]",
      /*4=*/"0:9-0:12: TYPE_ANNOTATION :: `u32` :: uN[32]",
      /*5=*/"0:17-0:20: TYPE_ANNOTATION :: `u32` :: uN[32]",
      /*6=*/"0:21-0:26: BLOCK :: `{\n  x\n}` :: uN[32]",
      /*7=*/"0:23-0:24: NAME_REF :: `x` :: uN[32]",
  };
  DoRun(program, want);
}

TEST(TypeInfoToProtoTest, ParametricIdentityFunction) {
  std::string program = R"(
fn pid<N: u32>(x: bits[N]) -> bits[N] { x }
fn id(x: u32) -> u32 { pid<u32:32>(x) }
)";
  std::vector<std::string> want = {
      /*0=*/"1:3-1:6: NAME_DEF :: `pid` :: (uN[32]) -> uN[32]",
      /*1=*/"1:7-1:8: NAME_DEF :: `N` :: uN[32]",
      /*2=*/"1:10-1:13: TYPE_ANNOTATION :: `u32` :: uN[32]",
      /*3=*/"1:15-1:16: NAME_DEF :: `x` :: uN[N]",
      /*4=*/"1:15-1:25: PARAM :: `x: bits[N]` :: uN[N]",
      /*5=*/"1:18-1:25: TYPE_ANNOTATION :: `bits[N]` :: uN[N]",
      /*6=*/"1:30-1:37: TYPE_ANNOTATION :: `bits[N]` :: uN[N]",
      /*7=*/
      "2:0-2:39: FUNCTION :: `fn id(x: u32) -> u32 {\n  pid<u32:32>(x)\n}` :: "
      "(uN[32]) -> uN[32]",
      /*8=*/"2:3-2:5: NAME_DEF :: `id` :: (uN[32]) -> uN[32]",
      /*9=*/"2:6-2:7: NAME_DEF :: `x` :: uN[32]",
      /*10=*/"2:6-2:12: PARAM :: `x: u32` :: uN[32]",
      /*11=*/"2:9-2:12: TYPE_ANNOTATION :: `u32` :: uN[32]",
      /*12=*/"2:17-2:20: TYPE_ANNOTATION :: `u32` :: uN[32]",
      /*13=*/"2:21-2:39: BLOCK :: `{\n  pid<u32:32>(x)\n}` :: uN[32]",
      /*14=*/"2:23-2:26: NAME_REF :: `pid` :: (uN[32]) -> uN[32]",
      /*15=*/"2:26-2:37: INVOCATION :: `pid<u32:32>(x)` :: uN[32]",
      /*16=*/"2:27-2:30: TYPE_ANNOTATION :: `u32` :: uN[32]",
      /*17=*/"2:31-2:33: NUMBER :: `u32:32` :: uN[32]",
      /*18=*/"2:35-2:36: NAME_REF :: `x` :: uN[32]",
  };
  DoRun(program, want);
}

TEST(TypeInfoToProtoTest, UnitFunction) {
  std::string program = R"(fn f() -> () { () })";
  std::vector<std::string> want = {
      "0:0-0:19: FUNCTION :: `fn f() -> () {\n  ()\n}` :: () -> ()",
      "0:3-0:4: NAME_DEF :: `f` :: () -> ()",
      "0:10-0:12: TYPE_ANNOTATION :: `()` :: ()",
      "0:13-0:19: BLOCK :: `{\n  ()\n}` :: ()",
      "0:15-0:18: XLS_TUPLE :: `()` :: ()",
  };
  DoRun(program, want);
}

TEST(TypeInfoToProtoTest, ArrayFunction) {
  std::string program = R"(fn f() -> u8[2] { u8[2]:[u8:1, u8:2] })";
  std::vector<std::string> want = {
      /*0=*/
      "0:0-0:38: FUNCTION :: `fn f() -> u8[2] {\n  u8[2]:[u8:1, u8:2]\n}` :: "
      "() -> uN[8][2]",
      /*1=*/"0:3-0:4: NAME_DEF :: `f` :: () -> uN[8][2]",
      /*2=*/"0:10-0:12: TYPE_ANNOTATION :: `u8` :: uN[8]",
      /*3=*/"0:10-0:15: TYPE_ANNOTATION :: `u8[2]` :: uN[8][2]",
      /*4=*/"0:13-0:14: NUMBER :: `2` :: uN[32]",
      /*5=*/"0:16-0:38: BLOCK :: `{\n  u8[2]:[u8:1, u8:2]\n}` :: uN[8][2]",
      /*6=*/"0:18-0:20: TYPE_ANNOTATION :: `u8` :: uN[8]",
      /*7=*/"0:18-0:23: TYPE_ANNOTATION :: `u8[2]` :: uN[8][2]",
      /*8=*/"0:21-0:22: NUMBER :: `2` :: uN[32]",
      /*9=*/"0:24-0:36: ARRAY :: `u8[2]:[u8:1, u8:2]` :: uN[8][2]",
      /*10=*/"0:25-0:27: TYPE_ANNOTATION :: `u8` :: uN[8]",
      /*11=*/"0:28-0:29: NUMBER :: `u8:1` :: uN[8]",
      /*12=*/"0:31-0:33: TYPE_ANNOTATION :: `u8` :: uN[8]",
      /*13=*/"0:34-0:35: NUMBER :: `u8:2` :: uN[8]",
  };
  DoRun(program, want);
}

TEST(TypeInfoToProtoTest, TokenFunction) {
  std::string program = R"(fn f(x: token) -> token { x })";
  std::vector<std::string> want = {
      /*0=*/
      "0:0-0:29: FUNCTION :: `fn f(x: token) -> token {\n  x\n}` :: (token) -> "
      "token",
      /*1=*/"0:3-0:4: NAME_DEF :: `f` :: (token) -> token",
      /*2=*/"0:5-0:6: NAME_DEF :: `x` :: token",
      /*3=*/"0:5-0:13: PARAM :: `x: token` :: token",
      /*4=*/"0:8-0:13: TYPE_ANNOTATION :: `token` :: token",
      /*5=*/"0:18-0:23: TYPE_ANNOTATION :: `token` :: token",
      /*6=*/"0:24-0:29: BLOCK :: `{\n  x\n}` :: token",
      /*7=*/"0:26-0:27: NAME_REF :: `x` :: token",
  };
  DoRun(program, want);
}

TEST(TypeInfoToProtoTest, MakeStructInstanceFunction) {
  std::string program = R"(
struct S { x: u32 }
fn f() -> S { S { x: u32:42 } }
)";
  std::vector<std::string> want = {
      /*0=*/
      "1:0-1:19: STRUCT_DEF :: `struct S {\n  x: u32,\n}` :: S { x: uN[32] }",
      /*1=*/"1:7-1:8: NAME_DEF :: `S` :: S { x: uN[32] }",
      /*2=*/"1:14-1:17: TYPE_ANNOTATION :: `u32` :: uN[32]",
      /*3=*/
      "2:0-2:31: FUNCTION :: `fn f() -> S {\n  S { x: u32:42 }\n}` :: () -> S "
      "{ x: uN[32] }",
      /*4=*/"2:3-2:4: NAME_DEF :: `f` :: () -> S { x: uN[32] }",
      /*5=*/"2:10-2:11: TYPE_REF :: `S` :: S { x: uN[32] }",
      /*6=*/"2:10-2:12: TYPE_ANNOTATION :: `S` :: S { x: uN[32] }",
      /*7=*/"2:12-2:31: BLOCK :: `{\n  S { x: u32:42 }\n}` :: S { x: uN[32] }",
      /*8=*/
      "2:16-2:29: STRUCT_INSTANCE :: `S { x: u32:42 }` :: S { x: uN[32] }",
      /*9=*/"2:21-2:24: TYPE_ANNOTATION :: `u32` :: uN[32]",
      /*10=*/"2:25-2:27: NUMBER :: `u32:42` :: uN[32]",
  };
  TypeInfoProto tip;
  DoRun(program, want, &tip);
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
  std::vector<std::string> want = {
      /*0=*/"1:0-1:4: ENUM_DEF :: `enum E : u32 {\n  A = 42,\n}` :: E",
      /*1=*/"1:5-1:6: NAME_DEF :: `E` :: E",
      /*2=*/"1:9-1:12: TYPE_ANNOTATION :: `u32` :: uN[32]",
      /*3=*/"1:15-1:16: NAME_DEF :: `A` :: E",
      /*4=*/"1:19-1:21: NUMBER :: `42` :: uN[32]",
      /*5=*/"2:0-2:20: FUNCTION :: `fn f() -> E {\n  E::A\n}` :: () -> E",
      /*6=*/"2:3-2:4: NAME_DEF :: `f` :: () -> E",
      /*7=*/"2:10-2:11: TYPE_REF :: `E` :: E",
      /*8=*/"2:10-2:12: TYPE_ANNOTATION :: `E` :: E",
      /*9=*/"2:12-2:20: BLOCK :: `{\n  E::A\n}` :: E",
      /*10=*/"2:15-2:18: COLON_REF :: `E::A` :: E",
  };
  DoRun(program, want);
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
import my_imported_module

type MyFoo = my_imported_module::Foo;
)";
  std::vector<std::string> want = {
      /*0=*/
      "3:0-3:37: TYPE_DEF :: `type MyFoo = my_imported_module::Foo;` :: Foo",
      /*1=*/"3:5-3:10: NAME_DEF :: `MyFoo` :: Foo",
      /*2=*/"3:13-3:36: TYPE_ANNOTATION :: `my_imported_module::Foo` :: Foo",
      /*3=*/"3:13-3:36: TYPE_REF :: `my_imported_module::Foo` :: Foo",
      /*4=*/"3:13-3:36: COLON_REF :: `my_imported_module::Foo` :: Foo",
  };
  DoRun(program, want, /*proto_out=*/nullptr, /*import_data=*/&import_data);
}

}  // namespace
}  // namespace xls::dslx
