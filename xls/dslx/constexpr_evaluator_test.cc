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
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/parser.h"
#include "xls/dslx/symbolic_bindings.h"
#include "xls/dslx/type_info.h"
#include "xls/dslx/typecheck.h"

namespace xls::dslx {
namespace {

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
  WarningCollector warnings;
  XLS_ASSIGN_OR_RETURN(
      test_data.type_info,
      CheckModule(test_data.module.get(), &test_data.import_data, &warnings));
  return std::move(test_data);
}

absl::StatusOr<ConcreteType*> GetConcreteType(TypeInfo* ti, Expr* expr) {
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

  Attr* attr = down_cast<Attr*>(f->body()->body());

  XLS_ASSERT_OK_AND_ASSIGN(ConcreteType * type,
                           GetConcreteType(type_info, attr));
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(&test_data.import_data, type_info,
                                             SymbolicBindings(), attr, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, type_info->GetConstExpr(attr));
  EXPECT_EQ(value.GetBitValueInt64().value(), 14);
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

  XLS_ASSERT_OK_AND_ASSIGN(ConcreteType * type,
                           GetConcreteType(type_info, number));
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(&test_data.import_data, type_info,
                                             SymbolicBindings(), number, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, type_info->GetConstExpr(number));
  EXPECT_EQ(value.GetBitValueInt64().value(), 7);
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

  Cast* cast = down_cast<Cast*>(f->body()->body());
  XLS_ASSERT_OK_AND_ASSIGN(ConcreteType * type,
                           GetConcreteType(type_info, cast));
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(&test_data.import_data, type_info,
                                             SymbolicBindings(), cast, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, type_info->GetConstExpr(cast));
  EXPECT_EQ(value.GetBitValueInt64().value(), 13);
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

  Ternary* ternary = down_cast<Ternary*>(f->body()->body());
  XLS_ASSERT_OK_AND_ASSIGN(ConcreteType * type,
                           GetConcreteType(type_info, ternary));
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(
      &test_data.import_data, type_info, SymbolicBindings(), ternary, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, type_info->GetConstExpr(ternary));
  EXPECT_EQ(value.GetBitValueInt64().value(), 500);
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
      down_cast<StructInstance*>(f->body()->body());
  XLS_ASSERT_OK_AND_ASSIGN(ConcreteType * type,
                           GetConcreteType(type_info, struct_instance));
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(&test_data.import_data, type_info,
                                             SymbolicBindings(),
                                             struct_instance, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           type_info->GetConstExpr(struct_instance));
  InterpValue element_value = value.Index(InterpValue::MakeUBits(1, 0)).value();
  EXPECT_EQ(element_value.GetBitValueInt64().value(), 666);
  element_value = value.Index(InterpValue::MakeUBits(1, 1)).value();
  EXPECT_EQ(element_value.GetBitValueInt64().value(), 777);
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
      down_cast<SplatStructInstance*>(f->body()->body());
  XLS_ASSERT_OK_AND_ASSIGN(ConcreteType * type,
                           GetConcreteType(type_info, struct_instance));
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(&test_data.import_data, type_info,
                                             SymbolicBindings(),
                                             struct_instance, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           type_info->GetConstExpr(struct_instance));
  InterpValue element_value = value.Index(InterpValue::MakeUBits(1, 0)).value();
  EXPECT_EQ(element_value.GetBitValueUint64().value(), 1000);
  element_value = value.Index(InterpValue::MakeUBits(1, 1)).value();
  EXPECT_EQ(element_value.GetBitValueInt64().value(), 200000);
}

TEST(ConstexprEvaluatorTest, HandleColonRefToConstant) {
  constexpr std::string_view kImported = R"(
pub const MY_CONST = u32:100;
  )";

  constexpr std::string_view kProgram = R"(
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

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  ColonRef* colon_ref = down_cast<ColonRef*>(f->body()->body());
  XLS_ASSERT_OK_AND_ASSIGN(ConcreteType * type,
                           GetConcreteType(tm.type_info, colon_ref));
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(
      &import_data, tm.type_info, SymbolicBindings(), colon_ref, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           tm.type_info->GetConstExpr(colon_ref));
  EXPECT_EQ(value.GetBitValueInt64().value(), 100);
}

TEST(ConstexprEvaluatorTest, HandleColonRefToEnum) {
  constexpr std::string_view kImported = R"(
pub enum MyEnum : u4 {
  HELLO = 3,
  DEAR = 4,
  FRIENDS = 5,
})";

  constexpr std::string_view kProgram = R"(
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

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  ColonRef* colon_ref = down_cast<ColonRef*>(f->body()->body());
  XLS_ASSERT_OK_AND_ASSIGN(ConcreteType * type,
                           GetConcreteType(tm.type_info, colon_ref));
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(
      &import_data, tm.type_info, SymbolicBindings(), colon_ref, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           tm.type_info->GetConstExpr(colon_ref));
  EXPECT_EQ(value.GetBitValueInt64().value(), 3);
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
  Index* index = down_cast<Index*>(f->body()->body());
  XLS_ASSERT_OK_AND_ASSIGN(ConcreteType * type,
                           GetConcreteType(tm.type_info, index));
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(&import_data, tm.type_info,
                                             SymbolicBindings(), index, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           tm.type_info->GetConstExpr(index));
  EXPECT_EQ(value.GetBitValueInt64().value(), 300);
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
  Index* index = down_cast<Index*>(f->body()->body());
  XLS_ASSERT_OK_AND_ASSIGN(ConcreteType * type,
                           GetConcreteType(tm.type_info, index));
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(&import_data, tm.type_info,
                                             SymbolicBindings(), index, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           tm.type_info->GetConstExpr(index));
  EXPECT_EQ(value.GetBitValueUint64().value(), 0xadbe);
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
  Index* index = down_cast<Index*>(f->body()->body());
  XLS_ASSERT_OK_AND_ASSIGN(ConcreteType * type,
                           GetConcreteType(tm.type_info, index));
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(&import_data, tm.type_info,
                                             SymbolicBindings(), index, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           tm.type_info->GetConstExpr(index));
  EXPECT_EQ(value.GetBitValueUint64().value(), 0xadbe);
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
  XlsTuple* xls_tuple = down_cast<XlsTuple*>(f->body()->body());
  std::vector<InterpValue> elements;
  elements.push_back(InterpValue::MakeU32(1));
  elements.push_back(InterpValue::MakeU32(2));
  elements.push_back(InterpValue::MakeU32(3));

  XLS_ASSERT_OK_AND_ASSIGN(ConcreteType * type,
                           GetConcreteType(tm.type_info, xls_tuple));
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(
      &import_data, tm.type_info, SymbolicBindings(), xls_tuple, type));
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
  Match* match = down_cast<Match*>(f->body()->body());
  XLS_ASSERT_OK_AND_ASSIGN(ConcreteType * type,
                           GetConcreteType(tm.type_info, match));
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(&import_data, tm.type_info,
                                             SymbolicBindings(), match, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           tm.type_info->GetConstExpr(match));
  EXPECT_EQ(value.GetBitValueUint64().value(), 2);
}

TEST(ConstexprEvaluatorTest, HandleFor_Simple) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32 {
  for (i, acc) : (u32, u32) in range(u32:0, u32:8) {
    acc + i
  } (u32:0)
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  For* xls_for = down_cast<For*>(f->body()->body());
  XLS_ASSERT_OK_AND_ASSIGN(ConcreteType * type,
                           GetConcreteType(tm.type_info, xls_for));
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(
      &import_data, tm.type_info, SymbolicBindings(), xls_for, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           tm.type_info->GetConstExpr(xls_for));
  EXPECT_EQ(value.GetBitValueUint64().value(), 0x1c);
}

TEST(ConstexprEvaluatorTest, HandleFor_OutsideRefs) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32 {
  let beef = u32:0xbeef;
  for (i, acc) : (u32, u32) in range(u32:0, u32:8) {
    acc + i + beef
  } (u32:0)
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  For* xls_for = down_cast<For*>(down_cast<Let*>(f->body()->body())->body());
  XLS_ASSERT_OK_AND_ASSIGN(ConcreteType * type,
                           GetConcreteType(tm.type_info, xls_for));
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(
      &import_data, tm.type_info, SymbolicBindings(), xls_for, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           tm.type_info->GetConstExpr(xls_for));
  EXPECT_EQ(value.GetBitValueUint64().value(), 0x5f794);
}

TEST(ConstexprEvaluatorTest, HandleFor_InitShadowed) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32 {
  let beef = u32:0xbeef;
  for (i, beef) : (u32, u32) in range(u32:0, u32:8) {
    beef + i
  } (beef)
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  For* xls_for = down_cast<For*>(down_cast<Let*>(f->body()->body())->body());
  XLS_ASSERT_OK_AND_ASSIGN(ConcreteType * type,
                           GetConcreteType(tm.type_info, xls_for));
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(
      &import_data, tm.type_info, SymbolicBindings(), xls_for, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           tm.type_info->GetConstExpr(xls_for));
  EXPECT_EQ(value.GetBitValueUint64().value(), 0xbf0b);
}

// Tests constexpr evaluation of `for` loops with misc. internal expressions (to
// exercise NameRefCollector).
TEST(ConstexprEvaluatorTest, HandleFor_MiscExprs) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32 {
  let beef = u32:0xbeef;
  for (i, beef) : (u32, u32) in range(u32:0, u32:8) {
    let upbeef = beef + i;
    let beeves = u32[4]:[0, 1, 2, 3];
    let beef_tuple = (upbeef, beeves[u32:2]);
    upbeef + beeves[u32:1] + beef_tuple.1
  } (beef)
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  For* xls_for = down_cast<For*>(down_cast<Let*>(f->body()->body())->body());
  XLS_ASSERT_OK_AND_ASSIGN(ConcreteType * type,
                           GetConcreteType(tm.type_info, xls_for));
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(
      &import_data, tm.type_info, SymbolicBindings(), xls_for, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           tm.type_info->GetConstExpr(xls_for));
  EXPECT_EQ(value.GetBitValueUint64().value(), 0xbf23);
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
  Unop* unop = down_cast<Unop*>(f->body()->body());
  XLS_ASSERT_OK_AND_ASSIGN(ConcreteType * type,
                           GetConcreteType(tm.type_info, unop));
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(&import_data, tm.type_info,
                                             SymbolicBindings(), unop, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, tm.type_info->GetConstExpr(unop));
  EXPECT_EQ(value.GetBitValueInt64().value(), -1337);
}

// Verifies that we can still constexpr evaluate, even when a variable declared
// inside a for loop shadows a var outside.
TEST(ConstexprEvaluatorTest, ShadowedVar) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32 {
  let my_var = u32:32;
  for (i, my_var) : (u32, u32) in range(u32:0, u32:4) {
    let my_var = my_var - u32:8;
    i + my_var
  }(my_var)
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  Let* let = down_cast<Let*>(f->body()->body());
  For* xls_for = down_cast<For*>(let->body());
  XLS_ASSERT_OK_AND_ASSIGN(ConcreteType * type,
                           GetConcreteType(tm.type_info, xls_for));
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(
      &import_data, tm.type_info, SymbolicBindings(), xls_for, type));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           tm.type_info->GetConstExpr(xls_for));
  EXPECT_EQ(value.GetBitValueInt64().value(), 6);
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
  XLS_ASSERT_OK(ConstexprEvaluator::Evaluate(
      &import_data, tm.type_info, SymbolicBindings(), f->body(), nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           tm.type_info->GetConstExpr(f->body()));
  EXPECT_EQ(value.GetBitValueInt64().value(), 32);
}

}  // namespace
}  // namespace xls::dslx
