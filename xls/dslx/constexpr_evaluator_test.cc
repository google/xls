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

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/casts.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/typecheck_module.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"

namespace xls::dslx {
namespace {

Expr* GetSingleBodyExpr(Function* f) {
  Block* body = f->body();
  CHECK_EQ(body->statements().size(), 1);
  return std::get<Expr*>(body->statements().at(0)->wrapped());
}

struct TestData {
  std::unique_ptr<Module> module;
  ImportData import_data;
  TypeInfo* type_info;
};

absl::StatusOr<TestData> CreateTestData(std::string_view module_text) {
  Scanner s("test.x", std::string(module_text));
  Parser parser{"test", &s};

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Module> module, parser.ParseModule());
  TestData test_data{std::move(module), CreateImportDataForTest()};
  WarningCollector warnings(kAllWarningsSet);
  XLS_ASSIGN_OR_RETURN(test_data.type_info,
                       TypecheckModule(test_data.module.get(),
                                       &test_data.import_data, &warnings));
  return std::move(test_data);
}

absl::StatusOr<Type*> GetType(TypeInfo* ti, Expr* expr) {
  auto maybe_type = ti->GetItem(expr);
  if (!maybe_type.has_value()) {
    return absl::NotFoundError("");
  }
  return maybe_type.value();
}

TEST(ConstexprEvaluatorTest, HandleAttr_Simple) {
  constexpr std::string_view kModule = R"(
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

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("Foo"));

  Attr* attr = down_cast<Attr*>(
      std::get<Expr*>(f->body()->statements().at(0)->wrapped()));

  XLS_ASSERT_OK_AND_ASSIGN(Type * type, GetType(type_info, attr));
  WarningCollector warnings(kAllWarningsSet);
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(&test_data.import_data, type_info,
                                             &warnings, ParametricEnv(), attr,
                                             type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, type_info->GetConstExpr(attr));
  EXPECT_EQ(value.GetBitValueViaSign().value(), 14);
  EXPECT_TRUE(warnings.warnings().empty());
}

TEST(ConstexprEvaluatorTest, HandleNumber_Simple) {
  constexpr std::string_view kModule = R"(
const kFoo = u32:7;
)";

  XLS_ASSERT_OK_AND_ASSIGN(TestData test_data, CreateTestData(kModule));
  Module* module = test_data.module.get();
  TypeInfo* type_info = test_data.type_info;
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * constant_def,
                           module->GetConstantDef("kFoo"));
  Number* number = down_cast<Number*>(constant_def->value());

  XLS_ASSERT_OK_AND_ASSIGN(Type * type, GetType(type_info, number));
  WarningCollector warnings(kAllWarningsSet);
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(&test_data.import_data, type_info,
                                             &warnings, ParametricEnv(), number,
                                             type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, type_info->GetConstExpr(number));
  EXPECT_EQ(value.GetBitValueViaSign().value(), 7);
  EXPECT_TRUE(warnings.warnings().empty());
}

TEST(ConstexprEvaluatorTest, HandleCast_Simple) {
  constexpr std::string_view kModule = R"(
const kFoo = u32:13;

fn Foo() -> u64 {
  kFoo as u64
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(TestData test_data, CreateTestData(kModule));
  Module* module = test_data.module.get();
  TypeInfo* type_info = test_data.type_info;

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("Foo"));

  Cast* cast = down_cast<Cast*>(GetSingleBodyExpr(f));
  XLS_ASSERT_OK_AND_ASSIGN(Type * type, GetType(type_info, cast));
  WarningCollector warnings(kAllWarningsSet);
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(&test_data.import_data, type_info,
                                             &warnings, ParametricEnv(), cast,
                                             type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, type_info->GetConstExpr(cast));
  EXPECT_EQ(value.GetBitValueViaSign().value(), 13);
}

TEST(ConstexprEvaluatorTest, HandleTernary) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32 {
  if false { u32:100 } else { u32:500 }
})";

  XLS_ASSERT_OK_AND_ASSIGN(TestData test_data, CreateTestData(kProgram));
  Module* module = test_data.module.get();
  TypeInfo* type_info = test_data.type_info;

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));

  Conditional* conditional = down_cast<Conditional*>(GetSingleBodyExpr(f));
  XLS_ASSERT_OK_AND_ASSIGN(Type * type, GetType(type_info, conditional));
  WarningCollector warnings(kAllWarningsSet);
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(&test_data.import_data, type_info,
                                             &warnings, ParametricEnv(),
                                             conditional, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           type_info->GetConstExpr(conditional));
  EXPECT_EQ(value.GetBitValueViaSign().value(), 500);
}

TEST(ConstexprEvaluatorTest, HandleStructInstance_Simple) {
  constexpr std::string_view kModule = R"(
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

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("Foo"));

  StructInstance* struct_instance =
      down_cast<StructInstance*>(GetSingleBodyExpr(f));
  XLS_ASSERT_OK_AND_ASSIGN(Type * type, GetType(type_info, struct_instance));
  WarningCollector warnings(kAllWarningsSet);
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(&test_data.import_data, type_info,
                                             &warnings, ParametricEnv(),
                                             struct_instance, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           type_info->GetConstExpr(struct_instance));
  InterpValue element_value = value.Index(InterpValue::MakeUBits(1, 0)).value();
  EXPECT_EQ(element_value.GetBitValueViaSign().value(), 666);
  element_value = value.Index(InterpValue::MakeUBits(1, 1)).value();
  EXPECT_EQ(element_value.GetBitValueViaSign().value(), 777);
}

TEST(ConstexprEvaluatorTest, HandleSplatStructInstance) {
  constexpr std::string_view kProgram = R"(
struct MyStruct {
  x: u32,
  y: u64
}

pub const my_struct = MyStruct { x: u32:1000, y: u64:100000 };

fn main() -> MyStruct {
  MyStruct { y: u64:200000, ..my_struct }
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(TestData test_data, CreateTestData(kProgram));
  Module* module = test_data.module.get();
  TypeInfo* type_info = test_data.type_info;

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           module->GetMemberOrError<Function>("main"));

  SplatStructInstance* struct_instance =
      down_cast<SplatStructInstance*>(GetSingleBodyExpr(f));
  XLS_ASSERT_OK_AND_ASSIGN(Type * type, GetType(type_info, struct_instance));
  WarningCollector warnings(kAllWarningsSet);
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(&test_data.import_data, type_info,
                                             &warnings, ParametricEnv(),
                                             struct_instance, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           type_info->GetConstExpr(struct_instance));
  InterpValue element_value = value.Index(InterpValue::MakeUBits(1, 0)).value();
  EXPECT_EQ(element_value.GetBitValueUnsigned().value(), 1000);
  element_value = value.Index(InterpValue::MakeUBits(1, 1)).value();
  EXPECT_EQ(element_value.GetBitValueViaSign().value(), 200000);
}

TEST(ConstexprEvaluatorTest, HandleColonRefToConstant) {
  constexpr std::string_view kImported = R"(
pub const MY_CONST = u32:100;
  )";

  constexpr std::string_view kProgram = R"(
import imported;
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

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  ColonRef* colon_ref = down_cast<ColonRef*>(GetSingleBodyExpr(f));
  XLS_ASSERT_OK_AND_ASSIGN(Type * type, GetType(tm.type_info, colon_ref));
  WarningCollector warnings(kAllWarningsSet);
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(
      &import_data, tm.type_info, &warnings, ParametricEnv(), colon_ref, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           tm.type_info->GetConstExpr(colon_ref));
  EXPECT_EQ(value.GetBitValueViaSign().value(), 100);
}

TEST(ConstexprEvaluatorTest, HandleColonRefToEnum) {
  constexpr std::string_view kImported = R"(
pub enum MyEnum : u4 {
  HELLO = 3,
  DEAR = 4,
  FRIENDS = 5,
})";

  constexpr std::string_view kProgram = R"(
import imported;
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

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  ColonRef* colon_ref = down_cast<ColonRef*>(GetSingleBodyExpr(f));
  XLS_ASSERT_OK_AND_ASSIGN(Type * type, GetType(tm.type_info, colon_ref));
  WarningCollector warnings(kAllWarningsSet);
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(
      &import_data, tm.type_info, &warnings, ParametricEnv(), colon_ref, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           tm.type_info->GetConstExpr(colon_ref));
  EXPECT_EQ(value.GetBitValueViaSign().value(), 3);
}

TEST(ConstexprEvaluatorTest, HandleIndex) {
  constexpr std::string_view kProgram = R"(
const MY_ARRAY = u32[4]:[u32:100, u32:200, u32:300, u32:400];

fn main() -> u32 {
  MY_ARRAY[u32:2]
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  Index* index = down_cast<Index*>(GetSingleBodyExpr(f));
  XLS_ASSERT_OK_AND_ASSIGN(Type * type, GetType(tm.type_info, index));
  WarningCollector warnings(kAllWarningsSet);
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(
      &import_data, tm.type_info, &warnings, ParametricEnv(), index, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           tm.type_info->GetConstExpr(index));
  EXPECT_EQ(value.GetBitValueViaSign().value(), 300);
}

TEST(ConstexprEvaluatorTest, HandleSlice) {
  constexpr std::string_view kProgram = R"(
const MY_VALUE = u32:0xdeadbeef;

fn main() -> u16 {
  MY_VALUE[8:24]
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  Expr* body_expr = GetSingleBodyExpr(f);
  Index* index = down_cast<Index*>(body_expr);
  XLS_ASSERT_OK_AND_ASSIGN(Type * type, GetType(tm.type_info, index));
  WarningCollector warnings(kAllWarningsSet);
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(
      &import_data, tm.type_info, &warnings, ParametricEnv(), index, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           tm.type_info->GetConstExpr(index));
  EXPECT_EQ(value.GetBitValueUnsigned().value(), 0xadbe);
}

TEST(ConstexprEvaluatorTest, HandleWidthSlice) {
  constexpr std::string_view kProgram = R"(
const MY_VALUE = u32:0xdeadbeef;

fn main() -> u16 {
  MY_VALUE[8 +: u16]
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  Index* index = down_cast<Index*>(GetSingleBodyExpr(f));
  XLS_ASSERT_OK_AND_ASSIGN(Type * type, GetType(tm.type_info, index));
  WarningCollector warnings(kAllWarningsSet);
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(
      &import_data, tm.type_info, &warnings, ParametricEnv(), index, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           tm.type_info->GetConstExpr(index));
  EXPECT_EQ(value.GetBitValueUnsigned().value(), 0xadbe);
}

TEST(ConstexprEvaluatorTest, HandleXlsTuple) {
  constexpr std::string_view kProgram = R"(
fn main() -> (u32, u32, u32) {
  (u32:1, u32:2, u32:3)
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  XlsTuple* xls_tuple = down_cast<XlsTuple*>(GetSingleBodyExpr(f));
  std::vector<InterpValue> elements;
  elements.push_back(InterpValue::MakeU32(1));
  elements.push_back(InterpValue::MakeU32(2));
  elements.push_back(InterpValue::MakeU32(3));

  XLS_ASSERT_OK_AND_ASSIGN(Type * type, GetType(tm.type_info, xls_tuple));
  WarningCollector warnings(kAllWarningsSet);
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(
      &import_data, tm.type_info, &warnings, ParametricEnv(), xls_tuple, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           tm.type_info->GetConstExpr(xls_tuple));
  EXPECT_EQ(value, InterpValue::MakeTuple(elements));
}

TEST(ConstexprEvaluatorTest, HandleMatch) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32 {
  match u32:200 {
    u32:100 => u32:1,
    u32:200 => u32:2,
    u32:300 => u32:3,
    _ => u32:4
  }
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  Match* match = down_cast<Match*>(GetSingleBodyExpr(f));
  XLS_ASSERT_OK_AND_ASSIGN(Type * type, GetType(tm.type_info, match));
  WarningCollector warnings(kAllWarningsSet);
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(
      &import_data, tm.type_info, &warnings, ParametricEnv(), match, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           tm.type_info->GetConstExpr(match));
  EXPECT_EQ(value.GetBitValueUnsigned().value(), 2);
}

TEST(ConstexprEvaluatorTest, HandleUnop) {
  constexpr std::string_view kProgram = R"(
fn main() -> s32 {
  -s32:1337
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  Unop* unop = down_cast<Unop*>(GetSingleBodyExpr(f));
  XLS_ASSERT_OK_AND_ASSIGN(Type * type, GetType(tm.type_info, unop));
  WarningCollector warnings(kAllWarningsSet);
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(
      &import_data, tm.type_info, &warnings, ParametricEnv(), unop, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, tm.type_info->GetConstExpr(unop));
  EXPECT_EQ(value.GetBitValueViaSign().value(), -1337);
}

TEST(ConstexprEvaluatorTest, BasicTupleIndex) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32 {
  (u64:64, u32:32, u16:16, u8:8).1
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  WarningCollector warnings(kAllWarningsSet);
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(&import_data, tm.type_info,
                                             &warnings, ParametricEnv(),
                                             f->body(), nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           tm.type_info->GetConstExpr(f->body()));
  EXPECT_EQ(value.GetBitValueViaSign().value(), 32);
}

TEST(ConstexprEvaluatorTest, ZeroMacro) {
  constexpr std::string_view kProgram = R"(
struct MyStruct {
  field: u32
}

fn main() -> MyStruct {
  zero!<MyStruct>()
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  WarningCollector warnings(kAllWarningsSet);
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(&import_data, tm.type_info,
                                             &warnings, ParametricEnv(),
                                             f->body(), nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           tm.type_info->GetConstExpr(f->body()));
  ASSERT_TRUE(value.IsTuple());
  ASSERT_THAT(value.GetLength(), status_testing::IsOkAndHolds(1));
  EXPECT_EQ(value.GetValuesOrDie().at(0).GetBitValueViaSign().value(), 0);
}

TEST(ConstexprEvaluatorTest, BuiltinArraySize) {
  constexpr std::string_view kProgram = R"(
fn p<N: u32>() -> u32 {
  N
}

fn main() -> u32 {
  array_size(u32[5]:[0, ...])
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  WarningCollector warnings(kAllWarningsSet);
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(&import_data, tm.type_info,
                                             &warnings, ParametricEnv(),
                                             f->body(), nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           tm.type_info->GetConstExpr(f->body()));
  EXPECT_EQ(value.GetBitValueViaSign().value(), 5);
}

}  // namespace
}  // namespace xls::dslx
