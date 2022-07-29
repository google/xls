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
#include "xls/dslx/bytecode_interpreter.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/bytecode_cache.h"
#include "xls/dslx/bytecode_emitter.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using testing::HasSubstr;

TEST(BytecodeInterpreterTest, TraceDataToString) {
  std::vector<InterpValue> stack = {
      InterpValue::MakeUBits(8, /*value=*/0x42),
      InterpValue::MakeUBits(3, /*value=*/4),
  };
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<FormatStep> steps,
                           ParseFormatString("x: {:x} y: {}"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string result, BytecodeInterpreter::TraceDataToString(steps, stack));
  EXPECT_TRUE(stack.empty());
  EXPECT_EQ("x: 42 y: 4", result);
}

absl::StatusOr<InterpValue> Interpret(ImportData* import_data,
                                      absl::string_view program,
                                      absl::string_view entry,
                                      std::vector<InterpValue> args = {}) {
  XLS_ASSIGN_OR_RETURN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", import_data));

  XLS_ASSIGN_OR_RETURN(Function * f,
                       tm.module->GetMemberOrError<Function>(entry));
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::Emit(import_data, tm.type_info, f, SymbolicBindings()));

  return BytecodeInterpreter::Interpret(import_data, bf.get(), args);
}

static const Pos kFakePos("fake.x", 0, 0);
static const Span kFakeSpan = Span(kFakePos, kFakePos);

TEST(BytecodeInterpreterTest, DupLiteral) {
  std::vector<Bytecode> bytecodes;
  bytecodes.emplace_back(kFakeSpan, Bytecode::Op::kLiteral,
                         InterpValue::MakeU32(42));
  bytecodes.emplace_back(kFakeSpan, Bytecode::Op::kDup);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto bfunc,
      BytecodeFunction::Create(/*owner=*/nullptr, /*source_fn=*/nullptr,
                               /*type_info=*/nullptr, std::move(bytecodes)));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue result,
                           BytecodeInterpreter::Interpret(
                               /*import_data=*/nullptr, bfunc.get(), {}));
  EXPECT_EQ(result.ToString(), "u32:42");
}

TEST(BytecodeInterpreterTest, DupEmptyStack) {
  std::vector<Bytecode> bytecodes;
  bytecodes.emplace_back(kFakeSpan, Bytecode::Op::kDup);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto bfunc,
      BytecodeFunction::Create(/*owner=*/nullptr, /*source_fn=*/nullptr,
                               /*type_info=*/nullptr, std::move(bytecodes)));
  ASSERT_THAT(
      BytecodeInterpreter::Interpret(/*import_data=*/nullptr, bfunc.get(), {}),
      StatusIs(absl::StatusCode::kInternal,
               ::testing::HasSubstr("!stack_.empty()")));
}

// Interprets a nearly-minimal bytecode program; the same from
// BytecodeEmitterTest.SimpleTranslation.
TEST(BytecodeInterpreterTest, PositiveSmokeTest) {
  constexpr absl::string_view kProgram = R"(
fn main() -> u32 {
  let a = u32:1;
  a + u32:2
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "main"));
  EXPECT_EQ(value, InterpValue::MakeU32(3));
}

// Tests that a failing assert_eq is interpreted correctly. Again, a
// continuation of a test from BytecodeEmitterTest. Get used to it.
TEST(BytecodeInterpreterTest, AssertEqFail) {
  constexpr absl::string_view kProgram = R"(
fn main() -> u32{
  let a = u32:3;
  let _ = assert_eq(a, u32:2);
  a
}
)";

  ImportData import_data(CreateImportDataForTest());
  absl::StatusOr<InterpValue> value = Interpret(&import_data, kProgram, "main");
  EXPECT_THAT(value.status(), StatusIs(absl::StatusCode::kInternal,
                                       HasSubstr("were not equal")));
}

TEST(BytecodeInterpreterTest, AssertLtFail) {
  constexpr absl::string_view kProgram = R"(
fn main() -> u32{
  let a = u32:3;
  let _ = assert_lt(a, u32:2);
  a
}
)";

  ImportData import_data(CreateImportDataForTest());
  absl::StatusOr<InterpValue> value = Interpret(&import_data, kProgram, "main");
  EXPECT_THAT(value.status(),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("want")));
}

// This test won't work unless BytecodeEmitterTest.DestructuringLet works!
TEST(BytecodeInterpreterTest, DestructuringLet) {
  constexpr absl::string_view kProgram = R"(
fn has_name_def_tree() -> (u32, u64, uN[128]) {
  let (a, b, (c, d)) = (u4:0, u8:1, (u16:2, (u32:3, u64:4, uN[128]:5)));
  let _ = assert_eq(a, u4:0);
  let _ = assert_eq(b, u8:1);
  let _ = assert_eq(c, u16:2);
  let _ = assert_eq(d, (u32:3, u64:4, uN[128]:5));
  d
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(&import_data, kProgram,
                                                        "has_name_def_tree"));

  ASSERT_TRUE(value.IsTuple());
  XLS_ASSERT_OK_AND_ASSIGN(int64_t num_elements, value.GetLength());
  ASSERT_EQ(num_elements, 3);
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue element,
                           value.Index(InterpValue::MakeU32(0)));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t bit_value, element.GetBitValueInt64());
  EXPECT_EQ(bit_value, 3);

  XLS_ASSERT_OK_AND_ASSIGN(element, value.Index(InterpValue::MakeU32(1)));
  XLS_ASSERT_OK_AND_ASSIGN(bit_value, element.GetBitValueInt64());
  EXPECT_EQ(bit_value, 4);

  XLS_ASSERT_OK_AND_ASSIGN(element, value.Index(InterpValue::MakeU32(2)));
  XLS_ASSERT_OK_AND_ASSIGN(bit_value, element.GetBitValueInt64());
  EXPECT_EQ(bit_value, 5);
}

TEST(BytecodeInterpreterTest, RunMatchArms) {
  constexpr absl::string_view kProgram = R"(
fn main(x: u32) -> u32 {
  match x {
    u32:42 => u32:64,
    u32:64 => u32:42,
    _ => x + u32:1
  }
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, /*path=*/"test.x", /*module_name=*/"test",
                        &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::Emit(&import_data, tm.type_info, f, SymbolicBindings()));

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, BytecodeInterpreter::Interpret(
                                                  &import_data, bf.get(),
                                                  {InterpValue::MakeU32(42)}));
  EXPECT_EQ(value, InterpValue::MakeU32(64)) << value.ToString();

  XLS_ASSERT_OK_AND_ASSIGN(
      value, BytecodeInterpreter::Interpret(&import_data, bf.get(),
                                            {InterpValue::MakeU32(64)}));
  EXPECT_EQ(value, InterpValue::MakeU32(42)) << value.ToString();

  XLS_ASSERT_OK_AND_ASSIGN(
      value, BytecodeInterpreter::Interpret(&import_data, bf.get(),
                                            {InterpValue::MakeU32(77)}));
  EXPECT_EQ(value, InterpValue::MakeU32(78)) << value.ToString();
}

TEST(BytecodeInterpreterTest, RunMatchArmsTuplePattern) {
  constexpr absl::string_view kProgram = R"(
fn main(t: (u32, u32)) -> u32 {
  match t {
    (u32:42, u32:64) => u32:1,
    (u32:64, u32:42) => u32:2,
    _ => u32:3,
  }
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, /*path=*/"test.x", /*module_name=*/"test",
                        &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::Emit(&import_data, tm.type_info, f, SymbolicBindings()));

  auto tuple_one = InterpValue::MakeTuple(
      {InterpValue::MakeU32(42), InterpValue::MakeU32(64)});
  auto tuple_two = InterpValue::MakeTuple(
      {InterpValue::MakeU32(64), InterpValue::MakeU32(42)});
  auto tuple_three = InterpValue::MakeTuple(
      {InterpValue::MakeU32(64), InterpValue::MakeU32(64)});
  auto tuple_four = InterpValue::MakeTuple(
      {InterpValue::MakeU32(42), InterpValue::MakeU32(42)});

  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue value,
      BytecodeInterpreter::Interpret(&import_data, bf.get(), {tuple_one}));
  EXPECT_EQ(value, InterpValue::MakeU32(1)) << value.ToString();

  XLS_ASSERT_OK_AND_ASSIGN(value, BytecodeInterpreter::Interpret(
                                      &import_data, bf.get(), {tuple_two}));
  EXPECT_EQ(value, InterpValue::MakeU32(2)) << value.ToString();

  XLS_ASSERT_OK_AND_ASSIGN(value, BytecodeInterpreter::Interpret(
                                      &import_data, bf.get(), {tuple_three}));
  EXPECT_EQ(value, InterpValue::MakeU32(3)) << value.ToString();

  XLS_ASSERT_OK_AND_ASSIGN(value, BytecodeInterpreter::Interpret(
                                      &import_data, bf.get(), {tuple_four}));
  EXPECT_EQ(value, InterpValue::MakeU32(3)) << value.ToString();
}

TEST(BytecodeInterpreterTest, RunMatchArmIrrefutablePattern) {
  constexpr absl::string_view kProgram = R"(
fn main(x: u32) -> u32 {
  match x {
    u32:42 => u32:64,
    y => y + u32:1,
    _ => u32:128,
  }
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, /*path=*/"test.x", /*module_name=*/"test",
                        &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::Emit(&import_data, tm.type_info, f, SymbolicBindings()));

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, BytecodeInterpreter::Interpret(
                                                  &import_data, bf.get(),
                                                  {InterpValue::MakeU32(42)}));
  EXPECT_EQ(value, InterpValue::MakeU32(64)) << value.ToString();

  XLS_ASSERT_OK_AND_ASSIGN(
      value, BytecodeInterpreter::Interpret(&import_data, bf.get(),
                                            {InterpValue::MakeU32(43)}));
  EXPECT_EQ(value, InterpValue::MakeU32(44)) << value.ToString();
}

TEST(BytecodeInterpreterTest, RunMatchNoTrailingWildcard) {
  constexpr absl::string_view kProgram = R"(
fn main(x: u32) -> u32 {
  match x {
    y => y + u32:1,
  }
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, /*path=*/"test.x", /*module_name=*/"test",
                        &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::Emit(&import_data, tm.type_info, f, SymbolicBindings()));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, BytecodeInterpreter::Interpret(
                                                  &import_data, bf.get(),
                                                  {InterpValue::MakeU32(1)}));
  EXPECT_EQ(value, InterpValue::MakeU32(2));
}

TEST(BytecodeInterpreterTest, RunMatchNoMatch) {
  constexpr absl::string_view kProgram = R"(
fn main(x: u32) -> u32 {
  match x {
    u32:1 => u32:2,
  }
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, /*path=*/"test.x", /*module_name=*/"test",
                        &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::Emit(&import_data, tm.type_info, f, SymbolicBindings()));
  absl::StatusOr<InterpValue> value = BytecodeInterpreter::Interpret(
      &import_data, bf.get(), {InterpValue::MakeU32(2)});
  EXPECT_THAT(value.status(), StatusIs(absl::StatusCode::kInternal,
                                       HasSubstr("The value was not matched")));
}

TEST(BytecodeInterpreterTest, RunMatchWithNameRefs) {
  constexpr absl::string_view kProgram = R"(
fn main(x: u32, y: u32, z: u32) -> u32 {
  match x {
    y => x + y,
    z => x + z,
    _ => u32:0xdeadbeef,
  }
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, /*path=*/"test.x", /*module_name=*/"test",
                        &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::Emit(&import_data, tm.type_info, f, SymbolicBindings()));

  InterpValue one(InterpValue::MakeU32(1));
  InterpValue two(InterpValue::MakeU32(2));
  InterpValue three(InterpValue::MakeU32(3));
  InterpValue four(InterpValue::MakeU32(4));
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue value,
      BytecodeInterpreter::Interpret(&import_data, bf.get(), {one, one, two}));
  EXPECT_EQ(value, two) << value.ToString();

  XLS_ASSERT_OK_AND_ASSIGN(value, BytecodeInterpreter::Interpret(
                                      &import_data, bf.get(), {two, one, two}));
  EXPECT_EQ(value, four) << value.ToString();

  XLS_ASSERT_OK_AND_ASSIGN(
      value, BytecodeInterpreter::Interpret(&import_data, bf.get(),
                                            {three, one, two}));
  EXPECT_EQ(value, InterpValue::MakeU32(0xdeadbeef)) << value.ToString();
}

TEST(BytecodeInterpreterTest, RunTernaryConsequent) {
  constexpr absl::string_view kProgram = R"(
fn main() -> u32 {
  if true { u32:42 } else { u32:64 }
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "main"));
  EXPECT_EQ(value, InterpValue::MakeU32(42)) << value.ToString();
}

TEST(BytecodeInterpreterTest, RunTernaryAlternate) {
  constexpr absl::string_view kProgram = R"(
fn main() -> u32 {
  if false { u32:42 } else { u32:64 }
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "main"));
  EXPECT_EQ(value, InterpValue::MakeU32(64)) << value.ToString();
}

TEST(BytecodeInterpreterTest, BinopAnd) {
  constexpr absl::string_view kProgram = R"(
fn do_and() -> u32 {
  let a = u32:0xa5a5a5a5;
  let b = u32:0xffffffff;
  a & b
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "do_and"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0xa5a5a5a5ll);
}

TEST(BytecodeInterpreterTest, BinopConcat) {
  constexpr absl::string_view kProgram = R"(
fn do_concat() -> u64 {
  let a = u32:0xa5a5a5a5;
  let b = u32:0xffffffff;
  a ++ b
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "do_concat"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0xa5a5a5a5ffffffffll);
}

TEST(BytecodeInterpreterTest, BinopDiv) {
  constexpr absl::string_view kProgram = R"(
fn do_div() -> u32 {
  let a = u32:0x84208420;
  let b = u32:0x4;
  a / b
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "do_div"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0x21082108);
}

TEST(BytecodeInterpreterTest, BinopMul) {
  constexpr absl::string_view kProgram = R"(
fn do_mul() -> u32 {
  let a = u32:0x21082108;
  let b = u32:0x4;
  a * b
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "do_mul"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0x84208420);
}

TEST(BytecodeInterpreterTest, BinopOr) {
  constexpr absl::string_view kProgram = R"(
fn do_or() -> u32 {
  let a = u32:0xa5a5a5a5;
  let b = u32:0x5a5a5a5a;
  a | b
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "do_or"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0xffffffff);
}

TEST(BytecodeInterpreterTest, BinopShll) {
  constexpr absl::string_view kProgram = R"(
fn do_shll() -> u32 {
  let a = u32:0x21082108;
  let b = u32:0x2;
  a << b
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "do_shll"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0x84208420);
}

TEST(BytecodeInterpreterTest, BinopShra) {
  constexpr absl::string_view kProgram = R"(
fn do_shrl() -> s32 {
  let a = s32:-128;
  let b = u32:2;
  a >> b
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "do_shrl"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_val, bits.ToInt64());
  EXPECT_EQ(int_val, -32);
}

TEST(BytecodeInterpreterTest, BinopShrl) {
  constexpr absl::string_view kProgram = R"(
fn do_shrl() -> u32 {
  let a = u32:0x84208420;
  let b = u32:0x2;
  a >> b
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "do_shrl"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0x21082108);
}

TEST(BytecodeInterpreterTest, BinopSub) {
  constexpr absl::string_view kProgram = R"(
fn do_sub() -> u32 {
  let a = u32:0xa5a5a5a5;
  let b = u32:0x5a5a5a5a;
  a - b
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "do_sub"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0x4b4b4b4b);
}

TEST(BytecodeInterpreterTest, BinopXor) {
  constexpr absl::string_view kProgram = R"(
fn do_xor() -> u32 {
  let a = u32:0xa5a5ffff;
  let b = u32:0x5a5affff;
  a ^ b
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "do_xor"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0xffff0000);
}

TEST(BytecodeInterpreterTest, Unops) {
  constexpr absl::string_view kProgram = R"(
fn unops() -> s32 {
  let a = s32:1;
  let b = !a;
  -b
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "unops"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0x2);
}

TEST(BytecodeInterpreterTest, CreateArray) {
  constexpr absl::string_view kProgram = R"(
fn arrays() -> u32[3] {
  let a = u32:32;
  u32[3]:[u32:0, u32:1, a]
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "arrays"));
  ASSERT_TRUE(value.IsArray());
  XLS_ASSERT_OK_AND_ASSIGN(int64_t num_elements, value.GetLength());
  ASSERT_EQ(num_elements, 3);
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue element,
                           value.Index(InterpValue::MakeU32(0)));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t bit_value, element.GetBitValueInt64());
  EXPECT_EQ(bit_value, 0);

  XLS_ASSERT_OK_AND_ASSIGN(element, value.Index(InterpValue::MakeU32(1)));
  XLS_ASSERT_OK_AND_ASSIGN(bit_value, element.GetBitValueInt64());
  EXPECT_EQ(bit_value, 1);

  XLS_ASSERT_OK_AND_ASSIGN(element, value.Index(InterpValue::MakeU32(2)));
  XLS_ASSERT_OK_AND_ASSIGN(bit_value, element.GetBitValueInt64());
  EXPECT_EQ(bit_value, 32);
}

TEST(BytecodeInterpreterTest, IndexArray) {
  constexpr absl::string_view kProgram = R"(
fn index_array() -> u32 {
  let a = u32[3]:[0, 1, 2];
  let b = bits[32][3]:[3, 4, 5];

  a[u32:0] + b[u32:1]
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "index_array"));
  ASSERT_TRUE(value.IsBits());
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitsOrDie().ToInt64());
  EXPECT_EQ(int_value, 4);
}

TEST(BytecodeInterpreterTest, IndexTuple) {
  constexpr absl::string_view kProgram = R"(
fn index_tuple() -> u32 {
  let a = (u32:0, (u32:1, u32:2));
  let b = ((u32:3, (u32:4,)), u32:5);

  a.1.1 + b.0.1.0
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "index_tuple"));
  ASSERT_TRUE(value.IsBits());
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitsOrDie().ToInt64());
  EXPECT_EQ(int_value, 6);
}

TEST(BytecodeInterpreterTest, SimpleBitSlice) {
  constexpr absl::string_view kProgram = R"(
fn simple_slice() -> u16 {
  let a = u32:0xdeadbeef;
  a[16:32]
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "simple_slice"));
  ASSERT_TRUE(value.IsBits());
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitsOrDie().ToUint64());
  EXPECT_EQ(int_value, 0xdead);
}

// Tests a slice from the start: a[-x:].
TEST(BytecodeInterpreterTest, NegativeStartSlice) {
  constexpr absl::string_view kProgram = R"(
fn negative_start_slice() -> u16 {
  let a = u32:0xdeadbeef;
  a[-16:]
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue value,
      Interpret(&import_data, kProgram, "negative_start_slice"));
  ASSERT_TRUE(value.IsBits());
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitsOrDie().ToUint64());
  EXPECT_EQ(int_value, 0xdead);
}

// Tests a slice from the end: a[:-x].
TEST(BytecodeInterpreterTest, NegativeEndSlice) {
  constexpr absl::string_view kProgram = R"(
fn negative_end_slice() -> u16 {
  let a = u32:0xdeadbeef;
  a[:-16]
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(&import_data, kProgram,
                                                        "negative_end_slice"));
  ASSERT_TRUE(value.IsBits());
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitsOrDie().ToUint64());
  EXPECT_EQ(int_value, 0xbeef);
}

TEST(BytecodeInterpreterTest, WidthSlice) {
  constexpr absl::string_view kProgram = R"(
fn width_slice() -> s16 {
  let a = u32:0xdeadbeef;
  a[u32:8 +: s16]
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "width_slice"));
  ASSERT_TRUE(value.IsBits());
  ASSERT_TRUE(value.IsSigned());
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitsOrDie().ToUint64());
  EXPECT_EQ(int_value, 0xadbe);
}

// Makes sure we properly handle an OOB width slice and don't leave an extra
// value on the stack.
TEST(BytecodeInterpreterTest, OobWidthSlice) {
  constexpr absl::string_view kProgram = R"(
fn oob_slicer(a: u32) -> (u32, u32) {
  let b = u32:0xfeedf00d;
  let c = uN[128]:0xffffffffffffffffffffffffffffffff;
  let d = (u32:0xffffffff)[c +: u16];
  (a, b)
}

fn oob_width_slice() -> (u32, u32)[4] {
  let a = u32[4]:[u32:0, u32:1, u32:2, u32:3];
  map(a, oob_slicer)
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue result, Interpret(&import_data, kProgram, "oob_width_slice"));
  XLS_ASSERT_OK_AND_ASSIGN(const std::vector<InterpValue>* array_elements,
                           result.GetValues());
  for (int i = 0; i < array_elements->size(); i++) {
    XLS_ASSERT_OK_AND_ASSIGN(const std::vector<InterpValue>* tuple_elements,
                             array_elements->at(i).GetValues());
    XLS_ASSERT_OK_AND_ASSIGN(uint64_t value,
                             tuple_elements->at(0).GetBitValueUint64());
    EXPECT_EQ(value, i);
    XLS_ASSERT_OK_AND_ASSIGN(value, tuple_elements->at(1).GetBitValueUint64());
    EXPECT_EQ(value, 0xfeedf00d);
  }
}

TEST(BytecodeInterpreterTest, WidthSliceWithZext) {
  constexpr absl::string_view kProgram = R"(
fn width_slice() -> u32 {
  let a = u32:0xdeadbeef;
  a[u32:16 +: u32]
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "width_slice"));
  ASSERT_TRUE(value.IsBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_value, value.GetBitsOrDie().ToUint64());
  EXPECT_EQ(int_value, 0x0000dead);
}

// Tests a slice from both ends: a[-x:-y].
TEST(BytecodeInterpreterTest, BothNegativeSlice) {
  constexpr absl::string_view kProgram = R"(
fn both_negative_slice() -> u8 {
  let a = u32:0xdeadbeef;
  a[-16:-8]
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(&import_data, kProgram,
                                                        "both_negative_slice"));
  ASSERT_TRUE(value.IsBits());
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitsOrDie().ToUint64());
  EXPECT_EQ(int_value, 0xad);
}

TEST(BytecodeInterpreterTest, CastBits_Extend) {
  constexpr absl::string_view kProgram = R"(
fn cast_extend() -> u32 {
  let a = u16:0xa5a5;
  let b = s16:0x8000;
  a as u32 + ((b as s32) as u32)
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "cast_extend"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0x25a5);
}

TEST(BytecodeInterpreterTest, CastBits_SignExtend) {
  constexpr absl::string_view kProgram = R"(
fn cast_sign_extend() -> s32 {
  let a = s16:0xffff;
  a as s32
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue value, Interpret(&import_data, kProgram, "cast_sign_extend"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToInt64());
  EXPECT_EQ(int_val, -1);
}

TEST(BytecodeInterpreterTest, CastBits_Shrink) {
  constexpr absl::string_view kProgram = R"(
fn cast_shrink() -> u16 {
  let a = u32:0x0000a5a5;
  let b = s32:0x8fff5a5a;
  a as u16 + b as u16
})";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "cast_shrink"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0xffff);
}

TEST(BytecodeInterpreterTest, CastArrayToBits) {
  constexpr absl::string_view kProgram = R"(
fn cast_array_to_bits() -> u32 {
  let a = u8[4]:[0xc, 0xa, 0xf, 0xe];
  a as u32
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(&import_data, kProgram,
                                                        "cast_array_to_bits"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0x0c0a0f0e);
}

TEST(BytecodeInterpreterTest, CastBitsToArray) {
  constexpr absl::string_view kProgram = R"(
fn cast_bits_to_array() -> u8 {
  let a = u32:0x0c0a0f0e;
  let b = a as u8[4];
  b[u32:2]
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(&import_data, kProgram,
                                                        "cast_bits_to_array"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0x0f);
}

TEST(BytecodeInterpreterTest, CastEnumToBits) {
  constexpr absl::string_view kProgram = R"(enum MyEnum : u3 {
  VAL_0 = 0,
  VAL_1 = 1,
  VAL_2 = 2,
  VAL_3 = 3,
}

fn cast_enum_to_bits() -> u3 {
  let a = MyEnum::VAL_3;
  a as u3
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(&import_data, kProgram,
                                                        "cast_enum_to_bits"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 3);
}

TEST(BytecodeInterpreterTest, CastBitsToEnum) {
  constexpr absl::string_view kProgram = R"(enum MyEnum : u3 {
  VAL_0 = 0,
  VAL_1 = 1,
  VAL_2 = 2,
  VAL_3 = 3,
}

fn cast_bits_to_enum() -> MyEnum {
  let a = u3:2;
  a as MyEnum
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(&import_data, kProgram,
                                                        "cast_bits_to_enum"));
  InterpValue::EnumData enum_data = value.GetEnumData().value();
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, enum_data.value.ToUint64());
  EXPECT_EQ(int_val, 2);
}

TEST(BytecodeInterpreterTest, CastWithMissingData) {
  constexpr absl::string_view kProgram = R"(enum MyEnum : u3 {
  VAL_0 = 0,
  VAL_1 = 1,
  VAL_2 = 2,
  VAL_3 = 3,
}

fn cast_bits_to_enum() -> MyEnum {
  let a = u3:2;
  a as MyEnum
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, tm.module->GetMemberOrError<Function>("cast_bits_to_enum"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::Emit(&import_data, tm.type_info, f, SymbolicBindings()));

  // Get a modifiable copy of the bytecodes.
  std::vector<Bytecode> bytecodes = bf->CloneBytecodes();

  // Clear out the data element of the last bytecode, the cast op.
  bytecodes[bytecodes.size() - 1] = Bytecode(Span::Fake(), Bytecode::Op::kCast);
  XLS_ASSERT_OK_AND_ASSIGN(bf,
                           BytecodeFunction::Create(f->owner(), f, tm.type_info,
                                                    std::move(bytecodes)));
  absl::StatusOr<InterpValue> value =
      BytecodeInterpreter::Interpret(&import_data, bf.get(), {});
  EXPECT_THAT(value.status(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Cast op requires ConcreteType data.")));
}

TEST(BytecodeInterpreterTest, Params) {
  constexpr absl::string_view kProgram = R"(
fn has_params(x: u32, y: u64) -> u48 {
  let a = u48:100;
  let x = x as u48 + a;
  let y = x + y as u48;
  x + y
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("has_params"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::Emit(&import_data, tm.type_info, f, SymbolicBindings()));

  std::vector<InterpValue> params;
  params.push_back(InterpValue::MakeU32(1));
  params.push_back(InterpValue::MakeU64(10));
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue value,
      BytecodeInterpreter::Interpret(&import_data, bf.get(), params));

  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 212);
}

TEST(BytecodeInterpreterTest, SimpleFnCall) {
  constexpr absl::string_view kProgram = R"(
fn callee(x: u32, y: u32) -> u32 {
  x + y
}

fn caller() -> u32{
  let a = u32:100;
  let b = u32:200;
  callee(a, b)
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "caller"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 300);
}

TEST(BytecodeInterpreterTest, NestedFnCalls) {
  constexpr absl::string_view kProgram = R"(
fn callee_callee(x: u32) -> u32 {
  x + u32:100
}

fn callee(x: u32, y: u32) -> u32 {
  x + callee_callee(y)
}

fn caller(a: u32) -> u32{
  let b = u32:200;
  callee(a, b)
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("caller"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::Emit(&import_data, tm.type_info, f, SymbolicBindings()));
  std::vector<InterpValue> params;
  params.push_back(InterpValue::MakeU32(100));
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue value,
      BytecodeInterpreter::Interpret(&import_data, bf.get(), params));

  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 400);
}

TEST(BytecodeInterpreterTest, SimpleParametric) {
  constexpr absl::string_view kProgram = R"(
fn foo<N: u32>(x: uN[N]) -> uN[N] {
  x * x
}

fn main() -> u32 {
  let a = foo<u32:16>(u16:4);
  let b = foo(u32:8);
  a as u32 + b
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitValueInt64());
  EXPECT_EQ(int_value, 80);
}

TEST(BytecodeInterpreterTest, NestedParametric) {
  constexpr absl::string_view kProgram = R"(
fn second<N: u32, M: u32, O: u32>(x: uN[N], y: uN[M]) -> uN[O] {
  x as uN[O] + y as uN[O]
}

fn first<N:u32>(x: uN[N]) -> uN[N] {
  second<N, u32:48, u32:64>(x, u48:7) as uN[N]
}

fn main() -> u32 {
  first(u32:5)
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitValueInt64());
  EXPECT_EQ(int_value, 12);
}

TEST(BytecodeInterpreterTest, ParametricStruct) {
  constexpr absl::string_view kProgram = R"(
struct MyStruct<N: u32, M: u32 = N * u32:2> {
  x: uN[N],
  y: uN[M]
}

fn foo<N: u32, M: u32>(x: MyStruct<N, M>) -> u32 {
  x.x as u32 + x.y as u32
}

fn main() -> u32 {
  foo(MyStruct { x: u16:100, y: u32:200 })
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitValueInt64());
  EXPECT_EQ(int_value, 300);
}

TEST(BytecodeInterpreterTest, BuiltinAddWithCarry) {
  constexpr absl::string_view kProgram = R"(
fn main() -> (u1, u8) {
  let x = u8:0xff;
  let y = u8:0x2;
  add_with_carry(x, y)
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue actual,
                           Interpret(&import_data, kProgram, "main"));
  InterpValue expected(InterpValue::MakeTuple(
      {InterpValue::MakeUBits(1, 1), InterpValue::MakeUBits(8, 1)}));
  EXPECT_TRUE(expected.Eq(actual));
}

TEST(BytecodeInterpreterTest, BuiltinBitSlice) {
  constexpr absl::string_view kProgram = R"(
fn main() -> u16 {
  bit_slice(u32:0xdeadbeef, u16:8, u16:16)
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitValueUint64());
  EXPECT_EQ(int_value, 0xadbe);
}

TEST(BytecodeInterpreterTest, BuiltinBitSliceUpdate) {
  constexpr absl::string_view kProgram = R"(
fn main() -> u32 {
  bit_slice_update(u32:0xbeefbeef, u32:16, u32:0xdead)
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitValueUint64());
  EXPECT_EQ(int_value, 0xdeadbeef);
}

TEST(BytecodeInterpreterTest, BuiltinClz) {
  constexpr absl::string_view kProgram = R"(
fn main() -> u32 {
  clz(u32:0xbeef)
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitValueUint64());
  EXPECT_EQ(int_value, 16);
}

TEST(BytecodeInterpreterTest, BuiltinCtz) {
  constexpr absl::string_view kProgram = R"(
fn main() -> u32 {
  ctz(u32:0xbeef0000)
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitValueUint64());
  EXPECT_EQ(int_value, 16);
}

TEST(BytecodeInterpreterTest, BuiltinOneHot) {
  constexpr absl::string_view kProgram = R"(
fn main() -> u8 {
  let input = u3:0x5;
  let r0 = one_hot(input, false);
  let r1 = one_hot(input, true);
  r0 ++ r1
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitValueUint64());
  EXPECT_EQ(int_value, 0x41);
}

TEST(BytecodeInterpreterTest, BuiltinOneHotSel) {
  constexpr absl::string_view kProgram = R"(
fn main() -> u32 {
  let cases = u32[8]:[u32:0x1, u32:0x20, u32:0x300, u32:0x4000,
                      u32:0x50000, u32:0x600000, u32:0x7000000, u32:0x80000000];
  let selector = u8:0xaa;
  one_hot_sel(selector, cases)
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitValueUint64());
  EXPECT_EQ(int_value, 0x80604020);
}

TEST(BytecodeInterpreterTest, BuiltinPrioritySel) {
  constexpr absl::string_view kProgram = R"(
fn main() -> u32 {
  let cases = u32[8]:[u32:0x1, u32:0x20, u32:0x300, u32:0x4000,
                      u32:0x50000, u32:0x600000, u32:0x7000000, u32:0x80000000];
  let selector = u8:0xaa;
  priority_sel(selector, cases)
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitValueUint64());
  EXPECT_EQ(int_value, 0x00000020);
}

TEST(BytecodeInterpreterTest, BuiltinRange) {
  constexpr absl::string_view kProgram = R"(
fn main() -> u32[5] {
  range(u32:100, u32:105)
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(const std::vector<InterpValue>* elements,
                           value.GetValues());
  EXPECT_EQ(elements->size(), 5);
  for (int i = 0; i < elements->size(); i++) {
    XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value,
                             elements->at(i).GetBitValueUint64());
    EXPECT_EQ(int_value, i + 100);
  }
}

TEST(BytecodeInterpreterTest, BuiltinGate) {
  constexpr absl::string_view kProgram = R"(
fn main(p: bool, x: u32) -> u32 {
  gate!(p, x)
})";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::Emit(&import_data, tm.type_info, f, SymbolicBindings()));

  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue value,
      BytecodeInterpreter::Interpret(
          &import_data, bf.get(),
          {InterpValue::MakeBool(true), InterpValue::MakeU32(0xbeef)}));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitValueUint64());
  EXPECT_EQ(int_value, 0xbeef);

  XLS_ASSERT_OK_AND_ASSIGN(
      value, BytecodeInterpreter::Interpret(
                 &import_data, bf.get(),
                 {InterpValue::MakeBool(false), InterpValue::MakeU32(0xbeef)}));
  XLS_ASSERT_OK_AND_ASSIGN(int_value, value.GetBitValueUint64());
  EXPECT_EQ(int_value, 0x0);
}

TEST(BytecodeInterpreterTest, BuiltinSMulp) {
  constexpr absl::string_view kProgram = R"(
fn main(x: s10, y: s10) -> s10 {
  let mulp = smulp(x, y);
  mulp.0 + mulp.1
})";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::Emit(&import_data, tm.type_info, f, SymbolicBindings()));

  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue value,
      BytecodeInterpreter::Interpret(
          &import_data, bf.get(),
          {InterpValue::MakeSBits(10, 3), InterpValue::MakeSBits(10, -5)}));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  EXPECT_THAT(bits.ToInt64(), IsOkAndHolds(-15));

  XLS_ASSERT_OK_AND_ASSIGN(
      value, BytecodeInterpreter::Interpret(&import_data, bf.get(),
                                            {InterpValue::MakeSBits(10, 511),
                                             InterpValue::MakeSBits(10, -5)}));
  XLS_ASSERT_OK_AND_ASSIGN(bits, value.GetBits());
  EXPECT_THAT(bits.ToInt64(), IsOkAndHolds(-507));
}

TEST(BytecodeInterpreterTest, BuiltinUMulp) {
  constexpr absl::string_view kProgram = R"(
fn main(x: u10, y: u10) -> u10 {
  let mulp = umulp(x, y);
  mulp.0 + mulp.1
})";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::Emit(&import_data, tm.type_info, f, SymbolicBindings()));

  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue value,
      BytecodeInterpreter::Interpret(
          &import_data, bf.get(),
          {InterpValue::MakeUBits(10, 3), InterpValue::MakeUBits(10, 5)}));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  EXPECT_THAT(bits.ToInt64(), IsOkAndHolds(15));

  XLS_ASSERT_OK_AND_ASSIGN(value, BytecodeInterpreter::Interpret(
                                      &import_data, bf.get(),
                                      {InterpValue::MakeUBits(10, 1023),
                                       InterpValue::MakeUBits(10, 1000)}));
  XLS_ASSERT_OK_AND_ASSIGN(bits, value.GetBits());
  EXPECT_THAT(bits.ToInt64(), IsOkAndHolds(24));
}

TEST(BytecodeInterpreterTest, RangeExpr) {
  constexpr absl::string_view kProgram = R"(
fn main() -> u32[8] {
  u32:8..u32:16
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(&import_data, kProgram, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(const std::vector<InterpValue>* elements,
                           value.GetValues());
  EXPECT_EQ(elements->size(), 8);
  for (int i = 0; i < elements->size(); i++) {
    XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value,
                             elements->at(i).GetBitValueUint64());
    EXPECT_EQ(int_value, i + 8);
  }
}

}  // namespace
}  // namespace xls::dslx
