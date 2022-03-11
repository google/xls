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
#include "xls/dslx/constexpr_evaluator.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/concrete_type.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/deduce.h"
#include "xls/dslx/deduce_ctx.h"
#include "xls/dslx/default_dslx_stdlib_path.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/parser.h"
#include "xls/dslx/type_info.h"
#include "xls/dslx/typecheck.h"

namespace xls::dslx {
namespace {

struct TestData {
  std::unique_ptr<Module> module;
  ImportData import_data;
  TypeInfo* type_info;
};

absl::StatusOr<TestData> CreateTestData(absl::string_view module_text) {
  Scanner s("test.x", std::string(module_text));
  Parser parser{"test", &s};

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Module> module, parser.ParseModule());
  TestData test_data{std::move(module), CreateImportDataForTest()};
  XLS_ASSIGN_OR_RETURN(
      test_data.type_info,
      CheckModule(test_data.module.get(), &test_data.import_data));
  return std::move(test_data);
}

TEST(ConstexprEvaluatorTest, HandleAttr_Simple) {
  constexpr absl::string_view kModule = R"(
struct MyStruct {
  x: u32,
  y: u64
}

const kMyConstStruct = MyStruct { x: u32:7, y: u64:14 };

fn Foo() -> u64 {
  kMyConstStruct.y
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(TestData test_data, CreateTestData(kModule));
  Module* module = test_data.module.get();
  TypeInfo* type_info = test_data.type_info;

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, module->GetFunctionOrError("Foo"));

  Attr* attr = down_cast<Attr*>(f->body());
  absl::optional<InterpValue> maybe_value = type_info->GetConstExpr(attr);
  ASSERT_TRUE(maybe_value.has_value());
  EXPECT_EQ(maybe_value.value().GetBitValueInt64().value(), 14);
}

TEST(ConstexprEvaluatorTest, HandleNumber_Simple) {
  constexpr absl::string_view kModule = R"(
const kFoo = u32:7;
)";

  XLS_ASSERT_OK_AND_ASSIGN(TestData test_data, CreateTestData(kModule));
  Module* module = test_data.module.get();
  TypeInfo* type_info = test_data.type_info;
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * constant_def,
                           module->GetConstantDef("kFoo"));
  Number* number = down_cast<Number*>(constant_def->value());

  absl::optional<InterpValue> maybe_value = type_info->GetConstExpr(number);
  ASSERT_TRUE(maybe_value.has_value());
  EXPECT_EQ(maybe_value.value().GetBitValueInt64().value(), 7);
}

TEST(ConstexprEvaluatorTest, HandleCast_Simple) {
  constexpr absl::string_view kModule = R"(
const kFoo = u32:13;

fn Foo() -> u64 {
  kFoo as u64
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(TestData test_data, CreateTestData(kModule));
  Module* module = test_data.module.get();
  TypeInfo* type_info = test_data.type_info;

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, module->GetFunctionOrError("Foo"));

  Cast* cast = down_cast<Cast*>(f->body());
  absl::optional<InterpValue> maybe_value = type_info->GetConstExpr(cast);
  ASSERT_TRUE(maybe_value.has_value());
  EXPECT_EQ(maybe_value.value().GetBitValueInt64().value(), 13);
}

TEST(ConstexprEvaluatorTest, HandleStructInstance_Simple) {
  constexpr absl::string_view kModule = R"(
struct MyStruct {
  x: u32,
  y: u64
}

fn Foo() -> MyStruct {
  MyStruct { x: u32:666, y: u64:777 }
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(TestData test_data, CreateTestData(kModule));
  Module* module = test_data.module.get();
  TypeInfo* type_info = test_data.type_info;

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, module->GetFunctionOrError("Foo"));

  StructInstance* struct_instance = down_cast<StructInstance*>(f->body());
  absl::optional<InterpValue> maybe_value =
      type_info->GetConstExpr(struct_instance);
  ASSERT_TRUE(maybe_value.has_value());
  InterpValue element_value =
      maybe_value.value().Index(InterpValue::MakeUBits(1, 0)).value();
  EXPECT_EQ(element_value.GetBitValueInt64().value(), 666);
  element_value =
      maybe_value.value().Index(InterpValue::MakeUBits(1, 1)).value();
  EXPECT_EQ(element_value.GetBitValueInt64().value(), 777);
}

TEST(ConstexprEvaluatorTest, HandleColonRefToConstant) {
  constexpr absl::string_view kImported = R"(
pub const MY_CONST = u32:100;
  )";

  constexpr absl::string_view kProgram = R"(
import imported
fn main() -> u32 {
  imported::MY_CONST
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK(
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data)
          .status());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, tm.module->GetFunctionOrError("main"));
  ColonRef* colon_ref = down_cast<ColonRef*>(f->body());
  absl::optional<InterpValue> maybe_value =
      tm.type_info->GetConstExpr(colon_ref);
  ASSERT_TRUE(maybe_value.has_value());
  EXPECT_EQ(maybe_value.value().GetBitValueInt64().value(), 100);
}

TEST(ConstexprEvaluatorTest, HandleColonRefToEnum) {
  constexpr absl::string_view kImported = R"(
pub enum MyEnum : u4 {
  HELLO = 3,
  DEAR = 4,
  FRIENDS = 5,
})";

  constexpr absl::string_view kProgram = R"(
import imported
fn main() -> imported::MyEnum {
  imported::MyEnum::HELLO
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK(
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data)
          .status());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, tm.module->GetFunctionOrError("main"));
  ColonRef* colon_ref = down_cast<ColonRef*>(f->body());
  absl::optional<InterpValue> maybe_value =
      tm.type_info->GetConstExpr(colon_ref);
  ASSERT_TRUE(maybe_value.has_value());
  EXPECT_EQ(maybe_value.value().GetBitValueInt64().value(), 3);
}

TEST(ConstexprEvaluatorTest, HandleIndex) {
  constexpr absl::string_view kProgram = R"(
const MY_ARRAY = u32[4]:[u32:100, u32:200, u32:300, u32:400];

fn main() -> u32 {
  MY_ARRAY[u32:2]
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, tm.module->GetFunctionOrError("main"));
  Index* index = down_cast<Index*>(f->body());
  absl::optional<InterpValue> maybe_value = tm.type_info->GetConstExpr(index);
  ASSERT_TRUE(maybe_value.has_value());
  EXPECT_EQ(maybe_value.value().GetBitValueInt64().value(), 300);
}

TEST(ConstexprEvaluatorTest, HandleSlice) {
  constexpr absl::string_view kProgram = R"(
const MY_VALUE = u32:0xdeadbeef;

fn main() -> u16 {
  MY_VALUE[8:24]
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, tm.module->GetFunctionOrError("main"));
  Index* index = down_cast<Index*>(f->body());
  absl::optional<InterpValue> maybe_value = tm.type_info->GetConstExpr(index);
  ASSERT_TRUE(maybe_value.has_value());
  EXPECT_EQ(maybe_value.value().GetBitValueUint64().value(), 0xadbe);
}

TEST(ConstexprEvaluatorTest, HandleWidthSlice) {
  constexpr absl::string_view kProgram = R"(
const MY_VALUE = u32:0xdeadbeef;

fn main() -> u16 {
  MY_VALUE[8 +: u16]
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, tm.module->GetFunctionOrError("main"));
  Index* index = down_cast<Index*>(f->body());
  absl::optional<InterpValue> maybe_value = tm.type_info->GetConstExpr(index);
  ASSERT_TRUE(maybe_value.has_value());
  EXPECT_EQ(maybe_value.value().GetBitValueUint64().value(), 0xadbe);
}

}  // namespace
}  // namespace xls::dslx
