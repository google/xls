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
#include "xls/dslx/bytecode/bytecode_interpreter.h"

#include <cstdint>
#include <filesystem>  // NOLINT
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/bytecode/bytecode_emitter.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/run_routines.h"

namespace xls::dslx {
namespace {

absl::StatusOr<TypecheckedModule> ParseAndTypecheckOrPrintError(
    std::string_view program, ImportData* import_data) {
  // Parse/typecheck and print a helpful error.
  absl::StatusOr<TypecheckedModule> tm_or =
      ParseAndTypecheck(program, "test.x", "test", import_data);
  if (!tm_or.ok()) {
    TryPrintError(tm_or.status(),
                  [&](std::string_view) -> absl::StatusOr<std::string> {
                    return std::string{program};
                  });
  }
  return tm_or;
}

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
  XLS_ASSERT_OK_AND_ASSIGN(std::string result,
                           BytecodeInterpreter::TraceDataToString(
                               Bytecode::TraceData(steps, {}), stack));
  EXPECT_TRUE(stack.empty());
  EXPECT_EQ("x: 42 y: 4", result);
}

// Helper that runs the bytecode interpreter after emitting an entry function as
// bytecode.
static absl::StatusOr<InterpValue> Interpret(
    std::string_view program, std::string_view entry,
    std::vector<InterpValue> args = {},
    const BytecodeInterpreterOptions& options = BytecodeInterpreterOptions()) {
  ImportData import_data(CreateImportDataForTest());
  XLS_ASSIGN_OR_RETURN(TypecheckedModule tm,
                       ParseAndTypecheckOrPrintError(program, &import_data));

  XLS_ASSIGN_OR_RETURN(Function * f,
                       tm.module->GetMemberOrError<Function>(entry));
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::Emit(&import_data, tm.type_info, f, ParametricEnv()));

  return BytecodeInterpreter::Interpret(&import_data, bf.get(), args, options);
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

TEST(BytecodeInterpreterTest, TraceFmtStructValue) {
  constexpr std::string_view kProgram = R"(
struct Point {
  x: u32,
  y: u32,
}
fn main() -> () {
  let p = Point{x: u32:42, y: u32:64};
  let _ = trace_fmt!("{}", p);
  ()
}
)";
  std::vector<std::string> trace_output;
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue value, Interpret(kProgram, "main", /*args=*/{},
                                   BytecodeInterpreterOptions().trace_hook(
                                       [&](std::string_view s) {
                                         trace_output.push_back(std::string{s});
                                       })));
  EXPECT_THAT(trace_output, testing::ElementsAre(R"(Point {
  x: 42,
  y: 64
})"));
  EXPECT_EQ(value, InterpValue::MakeUnit());
}

TEST(BytecodeInterpreterTest, NestedTraceFmtStructValue) {
  constexpr std::string_view kProgram = R"(
struct Point {
  x: u32,
  y: u32,
}

struct Foo {
  p: Point,
  z: u32,
}

fn main() -> () {
  let p = Foo { p: Point{x: u32:42, y: u32:64}, z: u32: 123 };
  let _ = trace_fmt!("{}", p);
  ()
}
)";
  std::vector<std::string> trace_output;
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue value, Interpret(kProgram, "main", /*args=*/{},
                                   BytecodeInterpreterOptions().trace_hook(
                                       [&](std::string_view s) {
                                         trace_output.push_back(std::string{s});
                                       })));
  EXPECT_THAT(trace_output, testing::ElementsAre(R"(Foo {
  p: Point {
    x: 42,
    y: 64
  },
  z: 123
})"));
  EXPECT_EQ(value, InterpValue::MakeUnit());
}

// Interprets a nearly-minimal bytecode program; the same from
// BytecodeEmitterTest.SimpleTranslation.
TEST(BytecodeInterpreterTest, PositiveSmokeTest) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32 {
  let a = u32:1;
  a + u32:2
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "main"));
  EXPECT_EQ(value, InterpValue::MakeU32(3));
}

// Tests that a failing assert_eq is interpreted correctly. Again, a
// continuation of a test from BytecodeEmitterTest. Get used to it.
TEST(BytecodeInterpreterTest, AssertEqFail) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32{
  let a = u32:3;
  let _ = assert_eq(a, u32:2);
  a
}
)";

  absl::StatusOr<InterpValue> value = Interpret(kProgram, "main");
  EXPECT_THAT(value.status(), StatusIs(absl::StatusCode::kInternal,
                                       HasSubstr("were not equal")));
}

TEST(BytecodeInterpreterTest, AssertLtFail) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32{
  let a = u32:3;
  let _ = assert_lt(a, u32:2);
  a
}
)";

  absl::StatusOr<InterpValue> value = Interpret(kProgram, "main");
  EXPECT_THAT(value.status(), StatusIs(absl::StatusCode::kInternal,
                                       HasSubstr("not less than")));
}

// This test won't work unless BytecodeEmitterTest.DestructuringLet works!
TEST(BytecodeInterpreterTest, DestructuringLet) {
  constexpr std::string_view kProgram = R"(
fn has_name_def_tree() -> (u32, u64, uN[128]) {
  let (a, b, (c, d)) = (u4:0, u8:1, (u16:2, (u32:3, u64:4, uN[128]:5)));
  let _ = assert_eq(a, u4:0);
  let _ = assert_eq(b, u8:1);
  let _ = assert_eq(c, u16:2);
  let _ = assert_eq(d, (u32:3, u64:4, uN[128]:5));
  d
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(kProgram, "has_name_def_tree"));

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
  constexpr std::string_view kProgram = R"(
fn main(x: u32) -> u32 {
  match x {
    u32:42 => u32:64,
    u32:64 => u32:42,
    _ => x + u32:1
  }
})";

  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue value,
      Interpret(kProgram, "main", {InterpValue::MakeU32(42)}));
  EXPECT_EQ(value, InterpValue::MakeU32(64)) << value.ToString();

  XLS_ASSERT_OK_AND_ASSIGN(
      value, Interpret(kProgram, "main", {InterpValue::MakeU32(64)}));
  EXPECT_EQ(value, InterpValue::MakeU32(42)) << value.ToString();

  XLS_ASSERT_OK_AND_ASSIGN(
      value, Interpret(kProgram, "main", {InterpValue::MakeU32(77)}));
  EXPECT_EQ(value, InterpValue::MakeU32(78)) << value.ToString();
}

TEST(BytecodeInterpreterTest, RunMatchArmsTuplePattern) {
  constexpr std::string_view kProgram = R"(
fn main(t: (u32, u32)) -> u32 {
  match t {
    (u32:42, u32:64) => u32:1,
    (u32:64, u32:42) => u32:2,
    _ => u32:3,
  }
})";

  auto tuple_one = InterpValue::MakeTuple(
      {InterpValue::MakeU32(42), InterpValue::MakeU32(64)});
  auto tuple_two = InterpValue::MakeTuple(
      {InterpValue::MakeU32(64), InterpValue::MakeU32(42)});
  auto tuple_three = InterpValue::MakeTuple(
      {InterpValue::MakeU32(64), InterpValue::MakeU32(64)});
  auto tuple_four = InterpValue::MakeTuple(
      {InterpValue::MakeU32(42), InterpValue::MakeU32(42)});

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(kProgram, "main", {tuple_one}));
  EXPECT_EQ(value, InterpValue::MakeU32(1)) << value.ToString();

  XLS_ASSERT_OK_AND_ASSIGN(value, Interpret(kProgram, "main", {tuple_two}));
  EXPECT_EQ(value, InterpValue::MakeU32(2)) << value.ToString();

  XLS_ASSERT_OK_AND_ASSIGN(value, Interpret(kProgram, "main", {tuple_three}));
  EXPECT_EQ(value, InterpValue::MakeU32(3)) << value.ToString();

  XLS_ASSERT_OK_AND_ASSIGN(value, Interpret(kProgram, "main", {tuple_four}));
  EXPECT_EQ(value, InterpValue::MakeU32(3)) << value.ToString();
}

TEST(BytecodeInterpreterTest, RunMatchArmIrrefutablePattern) {
  constexpr std::string_view kProgram = R"(
fn main(x: u32) -> u32 {
  match x {
    u32:42 => u32:64,
    y => y + u32:1,
    _ => u32:128,
  }
})";

  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue value,
      Interpret(kProgram, "main", {InterpValue::MakeU32(42)}));
  EXPECT_EQ(value, InterpValue::MakeU32(64)) << value.ToString();

  XLS_ASSERT_OK_AND_ASSIGN(
      value, Interpret(kProgram, "main", {InterpValue::MakeU32(43)}));
  EXPECT_EQ(value, InterpValue::MakeU32(44)) << value.ToString();
}

TEST(BytecodeInterpreterTest, RunMatchNoTrailingWildcard) {
  constexpr std::string_view kProgram = R"(
fn main(x: u32) -> u32 {
  match x {
    y => y + u32:1,
  }
})";

  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue value,
      Interpret(kProgram, "main", {InterpValue::MakeU32(1)}));
  EXPECT_EQ(value, InterpValue::MakeU32(2));
}

TEST(BytecodeInterpreterTest, RunMatchNoMatch) {
  constexpr std::string_view kProgram = R"(
fn main(x: u32) -> u32 {
  match x {
    u32:1 => u32:2,
  }
})";

  absl::StatusOr<InterpValue> value =
      Interpret(kProgram, "main", {InterpValue::MakeU32(2)});
  EXPECT_THAT(value.status(), StatusIs(absl::StatusCode::kInternal,
                                       HasSubstr("The value was not matched")));
}

TEST(BytecodeInterpreterTest, RunMatchWithNameRefs) {
  constexpr std::string_view kProgram = R"(
fn main(x: u32, y: u32, z: u32) -> u32 {
  match x {
    y => x + y,
    z => x + z,
    _ => u32:0xdeadbeef,
  }
}
)";

  InterpValue one(InterpValue::MakeU32(1));
  InterpValue two(InterpValue::MakeU32(2));
  InterpValue three(InterpValue::MakeU32(3));
  InterpValue four(InterpValue::MakeU32(4));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(kProgram, "main", {one, one, two}));
  EXPECT_EQ(value, two) << value.ToString();

  XLS_ASSERT_OK_AND_ASSIGN(value, Interpret(kProgram, "main", {two, one, two}));
  EXPECT_EQ(value, four) << value.ToString();

  XLS_ASSERT_OK_AND_ASSIGN(value,
                           Interpret(kProgram, "main", {three, one, two}));
  EXPECT_EQ(value, InterpValue::MakeU32(0xdeadbeef)) << value.ToString();
}

TEST(BytecodeInterpreterTest, RunTernaryConsequent) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32 {
  if true { u32:42 } else { u32:64 }
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "main"));
  EXPECT_EQ(value, InterpValue::MakeU32(42)) << value.ToString();
}

TEST(BytecodeInterpreterTest, RunTernaryAlternate) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32 {
  if false { u32:42 } else { u32:64 }
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "main"));
  EXPECT_EQ(value, InterpValue::MakeU32(64)) << value.ToString();
}

TEST(BytecodeInterpreterTest, BinopAnd) {
  constexpr std::string_view kProgram = R"(
fn do_and() -> u32 {
  let a = u32:0xa5a5a5a5;
  let b = u32:0xffffffff;
  a & b
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "do_and"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0xa5a5a5a5ll);
}

TEST(BytecodeInterpreterTest, BinopConcat) {
  constexpr std::string_view kProgram = R"(
fn do_concat() -> u64 {
  let a = u32:0xa5a5a5a5;
  let b = u32:0xffffffff;
  a ++ b
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "do_concat"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0xa5a5a5a5ffffffffll);
}

TEST(BytecodeInterpreterTest, BinopDiv) {
  constexpr std::string_view kProgram = R"(
fn do_div() -> u32 {
  let a = u32:0x84208420;
  let b = u32:0x4;
  a / b
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "do_div"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0x21082108);
}

TEST(BytecodeInterpreterTest, BinopMul) {
  constexpr std::string_view kProgram = R"(
fn do_mul() -> u32 {
  let a = u32:0x21082108;
  let b = u32:0x4;
  a * b
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "do_mul"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0x84208420);
}

TEST(BytecodeInterpreterTest, BinopOr) {
  constexpr std::string_view kProgram = R"(
fn do_or() -> u32 {
  let a = u32:0xa5a5a5a5;
  let b = u32:0x5a5a5a5a;
  a | b
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "do_or"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0xffffffff);
}

TEST(BytecodeInterpreterTest, BinopShll) {
  constexpr std::string_view kProgram = R"(
fn do_shll() -> u32 {
  let a = u32:0x21082108;
  let b = u32:0x2;
  a << b
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "do_shll"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0x84208420);
}

TEST(BytecodeInterpreterTest, BinopShra) {
  constexpr std::string_view kProgram = R"(
fn do_shrl() -> s32 {
  let a = s32:-128;
  let b = u32:2;
  a >> b
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "do_shrl"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_val, bits.ToInt64());
  EXPECT_EQ(int_val, -32);
}

TEST(BytecodeInterpreterTest, BinopShrl) {
  constexpr std::string_view kProgram = R"(
fn do_shrl() -> u32 {
  let a = u32:0x84208420;
  let b = u32:0x2;
  a >> b
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "do_shrl"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0x21082108);
}

TEST(BytecodeInterpreterTest, BinopSub) {
  constexpr std::string_view kProgram = R"(
fn do_sub() -> u32 {
  let a = u32:0xa5a5a5a5;
  let b = u32:0x5a5a5a5a;
  a - b
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "do_sub"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0x4b4b4b4b);
}

TEST(BytecodeInterpreterTest, BinopXor) {
  constexpr std::string_view kProgram = R"(
fn do_xor() -> u32 {
  let a = u32:0xa5a5ffff;
  let b = u32:0x5a5affff;
  a ^ b
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "do_xor"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0xffff0000);
}

TEST(BytecodeInterpreterTest, Unops) {
  constexpr std::string_view kProgram = R"(
fn unops() -> s32 {
  let a = s32:1;
  let b = !a;
  -b
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "unops"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0x2);
}

TEST(BytecodeInterpreterTest, CreateArray) {
  constexpr std::string_view kProgram = R"(
fn arrays() -> u32[3] {
  let a = u32:32;
  u32[3]:[u32:0, u32:1, a]
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "arrays"));
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
  constexpr std::string_view kProgram = R"(
fn index_array() -> u32 {
  let a = u32[3]:[0, 1, 2];
  let b = bits[32][3]:[3, 4, 5];

  a[u32:0] + b[u32:1]
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(kProgram, "index_array"));
  ASSERT_TRUE(value.IsBits());
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitsOrDie().ToInt64());
  EXPECT_EQ(int_value, 4);
}

TEST(BytecodeInterpreterTest, IndexTuple) {
  constexpr std::string_view kProgram = R"(
fn index_tuple() -> u32 {
  let a = (u32:0, (u32:1, u32:2));
  let b = ((u32:3, (u32:4,)), u32:5);

  a.1.1 + b.0.1.0
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(kProgram, "index_tuple"));
  ASSERT_TRUE(value.IsBits());
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitsOrDie().ToInt64());
  EXPECT_EQ(int_value, 6);
}

TEST(BytecodeInterpreterTest, SimpleBitSlice) {
  constexpr std::string_view kProgram = R"(
fn simple_slice() -> u16 {
  let a = u32:0xdeadbeef;
  a[16:32]
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(kProgram, "simple_slice"));
  ASSERT_TRUE(value.IsBits());
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitsOrDie().ToUint64());
  EXPECT_EQ(int_value, 0xdead);
}

// Tests a slice from the start: a[-x:].
TEST(BytecodeInterpreterTest, NegativeStartSlice) {
  constexpr std::string_view kProgram = R"(
fn negative_start_slice() -> u16 {
  let a = u32:0xdeadbeef;
  a[-16:]
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(kProgram, "negative_start_slice"));
  ASSERT_TRUE(value.IsBits());
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitsOrDie().ToUint64());
  EXPECT_EQ(int_value, 0xdead);
}

// Tests a slice from the end: a[:-x].
TEST(BytecodeInterpreterTest, NegativeEndSlice) {
  constexpr std::string_view kProgram = R"(
fn negative_end_slice() -> u16 {
  let a = u32:0xdeadbeef;
  a[:-16]
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(kProgram, "negative_end_slice"));
  ASSERT_TRUE(value.IsBits());
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitsOrDie().ToUint64());
  EXPECT_EQ(int_value, 0xbeef);
}

TEST(BytecodeInterpreterTest, WidthSlice) {
  constexpr std::string_view kProgram = R"(
fn width_slice() -> s16 {
  let a = u32:0xdeadbeef;
  a[u32:8 +: s16]
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(kProgram, "width_slice"));
  ASSERT_TRUE(value.IsBits());
  ASSERT_TRUE(value.IsSigned());
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitsOrDie().ToUint64());
  EXPECT_EQ(int_value, 0xadbe);
}

// Makes sure we properly handle an OOB width slice and don't leave an extra
// value on the stack.
TEST(BytecodeInterpreterTest, OobWidthSlice) {
  constexpr std::string_view kProgram = R"(
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

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue result,
                           Interpret(kProgram, "oob_width_slice"));
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
  constexpr std::string_view kProgram = R"(
fn width_slice() -> u32 {
  let a = u32:0xdeadbeef;
  a[u32:16 +: u32]
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(kProgram, "width_slice"));
  ASSERT_TRUE(value.IsBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_value, value.GetBitsOrDie().ToUint64());
  EXPECT_EQ(int_value, 0x0000dead);
}

// Tests a slice from both ends: a[-x:-y].
TEST(BytecodeInterpreterTest, BothNegativeSlice) {
  constexpr std::string_view kProgram = R"(
fn both_negative_slice() -> u8 {
  let a = u32:0xdeadbeef;
  a[-16:-8]
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(kProgram, "both_negative_slice"));
  ASSERT_TRUE(value.IsBits());
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitsOrDie().ToUint64());
  EXPECT_EQ(int_value, 0xad);
}

TEST(BytecodeInterpreterTest, CastBits_Extend) {
  constexpr std::string_view kProgram = R"(
fn cast_extend() -> u32 {
  let a = u16:0xa5a5;
  let b = s16:0x8000;
  a as u32 + ((b as s32) as u32)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(kProgram, "cast_extend"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0x25a5);
}

TEST(BytecodeInterpreterTest, CastBits_SignExtend) {
  constexpr std::string_view kProgram = R"(
fn cast_sign_extend() -> s32 {
  let a = s16:0xffff;
  a as s32
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(kProgram, "cast_sign_extend"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToInt64());
  EXPECT_EQ(int_val, -1);
}

TEST(BytecodeInterpreterTest, CastBits_Shrink) {
  constexpr std::string_view kProgram = R"(
fn cast_shrink() -> u16 {
  let a = u32:0x0000a5a5;
  let b = s32:0x8fff5a5a;
  a as u16 + b as u16
})";
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(kProgram, "cast_shrink"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0xffff);
}

TEST(BytecodeInterpreterTest, CastArrayToBits) {
  constexpr std::string_view kProgram = R"(
fn cast_array_to_bits() -> u32 {
  let a = u8[4]:[0xc, 0xa, 0xf, 0xe];
  a as u32
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(kProgram, "cast_array_to_bits"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0x0c0a0f0e);
}

TEST(BytecodeInterpreterTest, CastBitsToArray) {
  constexpr std::string_view kProgram = R"(
fn cast_bits_to_array() -> u8 {
  let a = u32:0x0c0a0f0e;
  let b = a as u8[4];
  b[u32:2]
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(kProgram, "cast_bits_to_array"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 0x0f);
}

TEST(BytecodeInterpreterTest, CastEnumToBits) {
  constexpr std::string_view kProgram = R"(enum MyEnum : u3 {
  VAL_0 = 0,
  VAL_1 = 1,
  VAL_2 = 2,
  VAL_3 = 3,
}

fn cast_enum_to_bits() -> u3 {
  let a = MyEnum::VAL_3;
  a as u3
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(kProgram, "cast_enum_to_bits"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 3);
}

TEST(BytecodeInterpreterTest, CastBitsToEnum) {
  constexpr std::string_view kProgram = R"(enum MyEnum : u3 {
  VAL_0 = 0,
  VAL_1 = 1,
  VAL_2 = 2,
  VAL_3 = 3,
}

fn cast_bits_to_enum() -> MyEnum {
  let a = u3:2;
  a as MyEnum
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(kProgram, "cast_bits_to_enum"));
  InterpValue::EnumData enum_data = value.GetEnumData().value();
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, enum_data.value.ToUint64());
  EXPECT_EQ(int_val, 2);
}

TEST(BytecodeInterpreterTest, CastWithMissingData) {
  constexpr std::string_view kProgram = R"(enum MyEnum : u3 {
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
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckedModule tm, ParseAndTypecheckOrPrintError(
                                                     kProgram, &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, tm.module->GetMemberOrError<Function>("cast_bits_to_enum"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::Emit(&import_data, tm.type_info, f, ParametricEnv()));

  // Get a modifiable copy of the bytecodes.
  std::vector<Bytecode> bytecodes = bf->CloneBytecodes();

  // Clear out the data element of the last bytecode, the cast op.
  bytecodes[bytecodes.size() - 1] = Bytecode(Span::Fake(), Bytecode::Op::kCast);
  XLS_ASSERT_OK_AND_ASSIGN(bf,
                           BytecodeFunction::Create(f->owner(), f, tm.type_info,
                                                    std::move(bytecodes)));
  absl::StatusOr<InterpValue> result =
      BytecodeInterpreter::Interpret(&import_data, bf.get(), {});
  EXPECT_THAT(result.status(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Cast op requires ConcreteType data.")));
}

TEST(BytecodeInterpreterTest, Params) {
  constexpr std::string_view kProgram = R"(
fn has_params(x: u32, y: u64) -> u48 {
  let a = u48:100;
  let x = x as u48 + a;
  let y = x + y as u48;
  x + y
})";

  std::vector<InterpValue> params;
  params.push_back(InterpValue::MakeU32(1));
  params.push_back(InterpValue::MakeU64(10));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(kProgram, "has_params", params));

  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 212);
}

TEST(BytecodeInterpreterTest, SimpleFnCall) {
  constexpr std::string_view kProgram = R"(
fn callee(x: u32, y: u32) -> u32 {
  x + y
}

fn caller() -> u32{
  let a = u32:100;
  let b = u32:200;
  callee(a, b)
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "caller"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 300);
}

TEST(BytecodeInterpreterTest, NestedFnCalls) {
  constexpr std::string_view kProgram = R"(
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

  std::vector<InterpValue> params;
  params.push_back(InterpValue::MakeU32(100));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(kProgram, "caller", params));

  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_val, bits.ToUint64());
  EXPECT_EQ(int_val, 400);
}

TEST(BytecodeInterpreterTest, SimpleParametric) {
  constexpr std::string_view kProgram = R"(
fn foo<N: u32>(x: uN[N]) -> uN[N] {
  x * x
}

fn main() -> u32 {
  let a = foo<u32:16>(u16:4);
  let b = foo(u32:8);
  a as u32 + b
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitValueInt64());
  EXPECT_EQ(int_value, 80);
}

TEST(BytecodeInterpreterTest, NestedParametric) {
  constexpr std::string_view kProgram = R"(
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

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitValueInt64());
  EXPECT_EQ(int_value, 12);
}

TEST(BytecodeInterpreterTest, ParametricStruct) {
  constexpr std::string_view kProgram = R"(
struct MyStruct<N: u32, M: u32 = {N * u32:2}> {
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

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitValueInt64());
  EXPECT_EQ(int_value, 300);
}

TEST(BytecodeInterpreterTest, BuiltinAddWithCarry) {
  constexpr std::string_view kProgram = R"(
fn main() -> (u1, u8) {
  let x = u8:0xff;
  let y = u8:0x2;
  add_with_carry(x, y)
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue actual, Interpret(kProgram, "main"));
  InterpValue expected(InterpValue::MakeTuple(
      {InterpValue::MakeUBits(1, 1), InterpValue::MakeUBits(8, 1)}));
  EXPECT_TRUE(expected.Eq(actual));
}

TEST(BytecodeInterpreterTest, BuiltinBitSlice) {
  constexpr std::string_view kProgram = R"(
fn main() -> u16 {
  bit_slice(u32:0xdeadbeef, u16:8, u16:16)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitValueUint64());
  EXPECT_EQ(int_value, 0xadbe);
}

TEST(BytecodeInterpreterTest, BuiltinBitSliceUpdate) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32 {
  bit_slice_update(u32:0xbeefbeef, u32:16, u32:0xdead)
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitValueUint64());
  EXPECT_EQ(int_value, 0xdeadbeef);
}

TEST(BytecodeInterpreterTest, BuiltinClz) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32 {
  clz(u32:0xbeef)
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitValueUint64());
  EXPECT_EQ(int_value, 16);
}

TEST(BytecodeInterpreterTest, BuiltinCtz) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32 {
  ctz(u32:0xbeef0000)
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitValueUint64());
  EXPECT_EQ(int_value, 16);
}

TEST(BytecodeInterpreterTest, BuiltinOneHot) {
  constexpr std::string_view kProgram = R"(
fn main() -> u8 {
  let input = u3:0x5;
  let r0 = one_hot(input, false);
  let r1 = one_hot(input, true);
  r0 ++ r1
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitValueUint64());
  EXPECT_EQ(int_value, 0x41);
}

TEST(BytecodeInterpreterTest, BuiltinOneHotSel) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32 {
  let cases = u32[8]:[u32:0x1, u32:0x20, u32:0x300, u32:0x4000,
                      u32:0x50000, u32:0x600000, u32:0x7000000, u32:0x80000000];
  let selector = u8:0xaa;
  one_hot_sel(selector, cases)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitValueUint64());
  EXPECT_EQ(int_value, 0x80604020);
}

TEST(BytecodeInterpreterTest, BuiltinPrioritySel) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32 {
  let cases = u32[8]:[u32:0x1, u32:0x20, u32:0x300, u32:0x4000,
                      u32:0x50000, u32:0x600000, u32:0x7000000, u32:0x80000000];
  let selector = u8:0xaa;
  priority_sel(selector, cases)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitValueUint64());
  EXPECT_EQ(int_value, 0x00000020);
}

TEST(BytecodeInterpreterTest, BuiltinRange) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32[5] {
  range(u32:100, u32:105)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "main"));
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
  constexpr std::string_view kProgram = R"(
fn main(p: bool, x: u32) -> u32 {
  gate!(p, x)
})";
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue value,
      Interpret(kProgram, "main",
                {InterpValue::MakeBool(true), InterpValue::MakeU32(0xbeef)}));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitValueUint64());
  EXPECT_EQ(int_value, 0xbeef);

  XLS_ASSERT_OK_AND_ASSIGN(value, Interpret(kProgram, "main",
                                            {InterpValue::MakeBool(false),
                                             InterpValue::MakeU32(0xbeef)}));
  XLS_ASSERT_OK_AND_ASSIGN(int_value, value.GetBitValueUint64());
  EXPECT_EQ(int_value, 0x0);
}

TEST(BytecodeInterpreterTest, BuiltinSMulp) {
  constexpr std::string_view kProgram = R"(
fn main(x: s10, y: s10) -> s10 {
  let mulp = smulp(x, y);
  mulp.0 + mulp.1
})";
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(kProgram, "main",
                                     {InterpValue::MakeSBits(10, 3),
                                      InterpValue::MakeSBits(10, -5)}));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  EXPECT_THAT(bits.ToInt64(), IsOkAndHolds(-15));

  XLS_ASSERT_OK_AND_ASSIGN(value, Interpret(kProgram, "main",
                                            {InterpValue::MakeSBits(10, 511),
                                             InterpValue::MakeSBits(10, -5)}));
  XLS_ASSERT_OK_AND_ASSIGN(bits, value.GetBits());
  EXPECT_THAT(bits.ToInt64(), IsOkAndHolds(-507));
}

TEST(BytecodeInterpreterTest, BuiltinUMulp) {
  constexpr std::string_view kProgram = R"(
fn main(x: u10, y: u10) -> u10 {
  let mulp = umulp(x, y);
  mulp.0 + mulp.1
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           Interpret(kProgram, "main",
                                     {InterpValue::MakeUBits(10, 3),
                                      InterpValue::MakeUBits(10, 5)}));
  XLS_ASSERT_OK_AND_ASSIGN(Bits bits, value.GetBits());
  EXPECT_THAT(bits.ToInt64(), IsOkAndHolds(15));

  XLS_ASSERT_OK_AND_ASSIGN(value,
                           Interpret(kProgram, "main",
                                     {InterpValue::MakeUBits(10, 1023),
                                      InterpValue::MakeUBits(10, 1000)}));
  XLS_ASSERT_OK_AND_ASSIGN(bits, value.GetBits());
  EXPECT_THAT(bits.ToInt64(), IsOkAndHolds(24));
}

TEST(BytecodeInterpreterTest, RangeExpr) {
  constexpr std::string_view kProgram = R"(
fn main() -> u32[8] {
  u32:8..u32:16
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "main"));
  XLS_ASSERT_OK_AND_ASSIGN(const std::vector<InterpValue>* elements,
                           value.GetValues());
  EXPECT_EQ(elements->size(), 8);
  for (int i = 0; i < elements->size(); i++) {
    XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value,
                             elements->at(i).GetBitValueUint64());
    EXPECT_EQ(int_value, i + 8);
  }
}

TEST(BytecodeInterpreterTest, TypeMaxExprU7) {
  constexpr std::string_view kProgram = R"(
fn main() -> u7 {
  u7::MAX
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "main"));
  EXPECT_THAT(value.GetBitValueUint64(), IsOkAndHolds(0x7f));
}

TEST(BytecodeInterpreterTest, TypeMaxExprS7) {
  constexpr std::string_view kProgram = R"(
fn main() -> s3 {
  s3::MAX
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "main"));
  EXPECT_THAT(value.GetBitValueInt64(), IsOkAndHolds(3));
}

TEST(BytecodeInterpreterTest, TypeMaxExprTypeAlias) {
  constexpr std::string_view kProgram = R"(
type MyU9 = uN[9];
fn main() -> MyU9 {
  MyU9::MAX
})";

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, Interpret(kProgram, "main"));
  EXPECT_THAT(value.GetBitValueUint64(), IsOkAndHolds(0x1ff));
}

TEST(BytecodeInterpreterTest, DistinctNestedParametricProcs) {
  // Tests that B, which has one set of parameters, can instantiate A, which has
  // a different set of parameters.
  // TODO(rspringer): Once this goes in, open a bug: if init() is changed to
  // "{ N }", it fails, because N can't be evaluated to a value: it's computed
  // without applying the caller bindings.
  constexpr std::string_view kProgram = R"(
proc A<N: u32> {
    data_in: chan<u32> in;
    data_out: chan<u32> out;

    init {
        N
    }
    config(data_in: chan<u32> in, data_out: chan<u32> out) {
        (data_in, data_out)
    }
    next(tok: token, state: u32) {
        let (tok, x) = recv(tok, data_in);
        let tok = send(tok, data_out, x + N + state);
        state + u32:1
    }
}

proc B<M: u32, N: u32> {
    data_in: chan<u32> in;
    data_out: chan<u32> out;

    init { () }
    config(data_in: chan<u32> in, data_out: chan<u32> out) {
        spawn A<N>(data_in, data_out);
        (data_in, data_out)
    }
    next(tok: token, state: ()) {
        ()
    }
}

#[test_proc]
proc BTester {
    data_in: chan<u32> out;
    data_out: chan<u32> in;
    terminator: chan<bool> out;

    init { () }

    config(terminator: chan<bool> out) {
        let (data_in_p, data_in_c) = chan<u32>;
        let (data_out_p, data_out_c) = chan<u32>;
        spawn B<u32:5, u32:3>(data_in_c, data_out_p);
        (data_in_p, data_out_c, terminator)
    }

    next(tok: token, state: ()) {
        let tok = send(tok, data_in, u32:3);
        let (tok, result) = recv(tok, data_out);
        let _ = assert_eq(result, u32:9);
        let tok = send(tok, terminator, true);
        ()
    }
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto temp_file,
                           TempFile::CreateWithContent(kProgram, "_test.x"));
  constexpr std::string_view kModuleName = "test";
  ParseAndTestOptions options;
  absl::StatusOr<TestResult> result = ParseAndTest(
      kProgram, kModuleName, std::string{temp_file.path()}, options);
  EXPECT_THAT(result, status_testing::IsOkAndHolds(TestResult::kAllPassed));
}

TEST(BytecodeInterpreterTest, PrettyPrintsStructs) {
  constexpr std::string_view kProgram = R"(
struct InnerStruct {
    x: u32,
    y: u32,
}

struct MyStruct {
    a: u32,
    b: u16[4],
    c: InnerStruct,
}

fn doomed() {
    let a = MyStruct {
        a: u32:0,
        b: u16[4]:[u16:1, u16:2, u16:3, u16:4],
        c: InnerStruct {
            x: u32:5,
            y: u32:6,
        }
    };

    let b = MyStruct {
        a: u32:7,
        b: u16[4]:[u16:8, ...],
        c: InnerStruct {
            x: u32:12,
            y: u32:13,
        }
    };

    assert_eq(a, b)
})";

  absl::StatusOr<InterpValue> value = Interpret(kProgram, "doomed");
  EXPECT_THAT(value.status(), StatusIs(absl::StatusCode::kInternal,
                                       HasSubstr("were not equal")));
  EXPECT_TRUE(absl::StrContains(value.status().message(),
                                R"(lhs: MyStruct {
    a: u32:0
    b: uN[16][4]:[u16:1, u16:2, u16:3, u16:4]
    c: InnerStruct {
        x: u32:5
        y: u32:6
    }
})"));
}

TEST(BytecodeInterpreterTest, PrettyPrintsArrays) {
  constexpr std::string_view kProgram = R"(
struct InnerStruct {
    x: u32,
    y: u32,
}

struct MyStruct {
    c: InnerStruct[2],
}

fn doomed() {
    let a = MyStruct {
        c: InnerStruct[2]:[
            InnerStruct {
                x: u32:1,
                y: u32:2,
            },
            InnerStruct {
                x: u32:3,
                y: u32:4,
            }
        ],
    };
    let b = MyStruct {
        c: InnerStruct[2]:[
            InnerStruct {
                x: u32:5,
                y: u32:6,
            },
            InnerStruct {
                x: u32:7,
                y: u32:8,
            },
        ],
    };

    assert_eq(a, b)
})";

  absl::StatusOr<InterpValue> value = Interpret(kProgram, "doomed");
  EXPECT_THAT(value.status(), StatusIs(absl::StatusCode::kInternal,
                                       HasSubstr("were not equal")));
  EXPECT_TRUE(absl::StrContains(value.status().message(),
                                R"(lhs: MyStruct {
    c: InnerStruct[2]:[
        InnerStruct {
            x: u32:1
            y: u32:2
        },
        InnerStruct {
            x: u32:3
            y: u32:4
        }
    ])"));
}

TEST(BytecodeInterpreterTest, TraceChannels) {
  constexpr std::string_view kProgram = R"(
proc incrementer {
  in_ch: chan<u32> in;
  out_ch: chan<u32> out;

  init { () }

  config(in_ch: chan<u32> in,
         out_ch: chan<u32> out) {
    (in_ch, out_ch)
  }
  next(tok: token, _: ()) {
    let (tok, i) = recv(tok, in_ch);
    let tok = send(tok, out_ch, i + u32:1);
    ()
  }
}

#[test_proc]
proc tester_proc {
  data_out: chan<u32> out;
  data_in: chan<u32> in;
  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (input_p, input_c) = chan<u32>;
    let (output_p, output_c) = chan<u32>;
    spawn incrementer(input_c, output_p);
    (input_p, output_c, terminator)
  }

  next(tok: token, state: ()) {
    let tok = send(tok, data_out, u32:42);
    let (tok, result) = recv(tok, data_in);

    let tok = send(tok, data_out, u32:100);
    let (tok, result) = recv(tok, data_in);

    let tok = send(tok, terminator, true);
    ()
 }
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckedModule tm, ParseAndTypecheckOrPrintError(
                                                     kProgram, &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(TestProc * test_proc,
                           tm.module->GetTestProc("tester_proc"));
  XLS_ASSERT_OK_AND_ASSIGN(
      TypeInfo * ti, tm.type_info->GetTopLevelProcTypeInfo(test_proc->proc()));
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue terminator,
      ti->GetConstExpr(test_proc->proc()->config()->params()[0]));
  std::vector<ProcInstance> proc_instances;
  std::vector<std::string> trace_output;
  XLS_ASSERT_OK(ProcConfigBytecodeInterpreter::InitializeProcNetwork(
      &import_data, ti, test_proc->proc(), terminator, &proc_instances,
      BytecodeInterpreterOptions().trace_channels(true).trace_hook(
          [&](std::string_view s) {
            trace_output.push_back(std::string{s});
          })));
  std::shared_ptr<InterpValue::Channel> term_chan =
      terminator.GetChannelOrDie();
  while (term_chan->empty()) {
    for (auto& p : proc_instances) {
      XLS_ASSERT_OK(p.Run());
    }
  }
  EXPECT_THAT(trace_output,
              testing::ElementsAre(
                  "Sent data on channel `tester_proc::data_out`:\n  42",
                  "Received data on channel `incrementer::in_ch`:\n  42",
                  "Sent data on channel `incrementer::out_ch`:\n  43",
                  "Received data on channel `tester_proc::data_in`:\n  43",
                  "Sent data on channel `tester_proc::data_out`:\n  100",
                  "Received data on channel `incrementer::in_ch`:\n  100",
                  "Sent data on channel `incrementer::out_ch`:\n  101",
                  "Received data on channel `tester_proc::data_in`:\n  101",
                  "Sent data on channel `tester_proc::terminator`:\n  1"));
}

TEST(BytecodeInterpreterTest, TraceChannelsWithNonblockingReceive) {
  constexpr std::string_view kProgram = R"(
proc incrementer {
  in_ch: chan<u32> in;
  out_ch: chan<u32> out;

  init { () }

  config(in_ch: chan<u32> in,
         out_ch: chan<u32> out) {
    (in_ch, out_ch)
  }
  next(tok: token, _: ()) {
    let (tok, i, valid) = recv_non_blocking(tok, in_ch, u32:0);
    let tok = send(tok, out_ch, i + u32:1);
    ()
  }
}

#[test_proc]
proc tester_proc {
  data_out: chan<u32> out;
  data_in: chan<u32> in;
  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (input_p, input_c) = chan<u32>;
    let (output_p, output_c) = chan<u32>;
    spawn incrementer(input_c, output_p);
    (input_p, output_c, terminator)
  }

  next(tok: token, state: ()) {
    let tok = send_if(tok, data_out, false, u32:42);
    let (tok, result) = recv(tok, data_in);
    let tok = send(tok, terminator, true);
    ()
 }
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckedModule tm, ParseAndTypecheckOrPrintError(
                                                     kProgram, &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(TestProc * test_proc,
                           tm.module->GetTestProc("tester_proc"));
  XLS_ASSERT_OK_AND_ASSIGN(
      TypeInfo * ti, tm.type_info->GetTopLevelProcTypeInfo(test_proc->proc()));
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue terminator,
      ti->GetConstExpr(test_proc->proc()->config()->params()[0]));
  std::vector<ProcInstance> proc_instances;
  std::vector<std::string> trace_output;
  XLS_ASSERT_OK(ProcConfigBytecodeInterpreter::InitializeProcNetwork(
      &import_data, ti, test_proc->proc(), terminator, &proc_instances,
      BytecodeInterpreterOptions().trace_channels(true).trace_hook(
          [&](std::string_view s) {
            trace_output.push_back(std::string{s});
          })));
  std::shared_ptr<InterpValue::Channel> term_chan =
      terminator.GetChannelOrDie();
  while (term_chan->empty()) {
    for (auto& p : proc_instances) {
      XLS_ASSERT_OK(p.Run());
    }
  }
  EXPECT_THAT(trace_output,
              testing::ElementsAre(
                  "Sent data on channel `incrementer::out_ch`:\n  1",
                  "Received data on channel `tester_proc::data_in`:\n  1",
                  "Sent data on channel `tester_proc::terminator`:\n  1"));
}

TEST(BytecodeInterpreterTest, TraceStructChannels) {
  constexpr std::string_view kProgram = R"(
struct Foo {
  a: u32,
  b: u16
}

proc incrementer {
  in_ch: chan<Foo> in;
  out_ch: chan<Foo> out;

  init { () }

  config(in_ch: chan<Foo> in,
         out_ch: chan<Foo> out) {
    (in_ch, out_ch)
  }
  next(tok: token, _: ()) {
    let (tok, i) = recv(tok, in_ch);
    let tok = send(tok, out_ch, Foo { a:i.a + u32:1, b:i.b + u16:1 });
    ()
  }
}

#[test_proc]
proc tester_proc {
  data_out: chan<Foo> out;
  data_in: chan<Foo> in;
  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (input_p, input_c) = chan<Foo>;
    let (output_p, output_c) = chan<Foo>;
    spawn incrementer(input_c, output_p);
    (input_p, output_c, terminator)
  }

  next(tok: token, state: ()) {

    let tok = send(tok, data_out, Foo { a:u32:42, b:u16:100 });
    let (tok, result) = recv(tok, data_in);

    let tok = send(tok, data_out, Foo{ a:u32:555, b:u16:123 });
    let (tok, result) = recv(tok, data_in);

    let tok = send(tok, terminator, true);
    ()
 }
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckedModule tm, ParseAndTypecheckOrPrintError(
                                                     kProgram, &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(TestProc * test_proc,
                           tm.module->GetTestProc("tester_proc"));
  XLS_ASSERT_OK_AND_ASSIGN(
      TypeInfo * ti, tm.type_info->GetTopLevelProcTypeInfo(test_proc->proc()));
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue terminator,
      ti->GetConstExpr(test_proc->proc()->config()->params()[0]));
  std::vector<ProcInstance> proc_instances;
  std::vector<std::string> trace_output;
  XLS_ASSERT_OK(ProcConfigBytecodeInterpreter::InitializeProcNetwork(
      &import_data, ti, test_proc->proc(), terminator, &proc_instances,
      BytecodeInterpreterOptions().trace_channels(true).trace_hook(
          [&](std::string_view s) {
            trace_output.push_back(std::string{s});
          })));
  std::shared_ptr<InterpValue::Channel> term_chan =
      terminator.GetChannelOrDie();
  while (term_chan->empty()) {
    for (auto& p : proc_instances) {
      XLS_ASSERT_OK(p.Run());
    }
  }
  EXPECT_EQ(trace_output[0],
            R"(Sent data on channel `tester_proc::data_out`:
  Foo {
    a: 42,
    b: 100
  })");
  EXPECT_EQ(trace_output[1],
            R"(Received data on channel `incrementer::in_ch`:
  Foo {
    a: 42,
    b: 100
  })");
}

TEST(BytecodeInterpreterTest, TraceArrayOfChannels) {
  constexpr std::string_view kProgram = R"(
proc incrementer {
  in_ch: chan<u32> in;
  out_ch: chan<u32> out;

  init { () }

  config(in_ch: chan<u32> in,
         out_ch: chan<u32> out) {
    (in_ch, out_ch)
  }
  next(tok: token, _: ()) {
    let (tok, i) = recv(tok, in_ch);
    let tok = send(tok, out_ch, i + u32:1);
    ()
  }
}

#[test_proc]
proc tester_proc {
  data_out: chan<u32>[1] out;
  data_in: chan<u32>[1] in;
  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (input_p, input_c) = chan<u32>[1];
    let (output_p, output_c) = chan<u32>[1];
    spawn incrementer(input_c[0], output_p[0]);
    (input_p, output_c, terminator)
  }

  next(tok: token, state: ()) {

    let tok = send(tok, data_out[0], u32:42);
    let (tok, result) = recv(tok, data_in[0]);

    let tok = send(tok, data_out[0], u32:100);
    let (tok, result) = recv(tok, data_in[0]);

    let tok = send(tok, terminator, true);
    ()
 }
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckedModule tm, ParseAndTypecheckOrPrintError(
                                                     kProgram, &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(TestProc * test_proc,
                           tm.module->GetTestProc("tester_proc"));
  XLS_ASSERT_OK_AND_ASSIGN(
      TypeInfo * ti, tm.type_info->GetTopLevelProcTypeInfo(test_proc->proc()));
  XLS_ASSERT_OK_AND_ASSIGN(
      InterpValue terminator,
      ti->GetConstExpr(test_proc->proc()->config()->params()[0]));
  std::vector<ProcInstance> proc_instances;
  std::vector<std::string> trace_output;
  XLS_ASSERT_OK(ProcConfigBytecodeInterpreter::InitializeProcNetwork(
      &import_data, ti, test_proc->proc(), terminator, &proc_instances,
      BytecodeInterpreterOptions().trace_channels(true).trace_hook(
          [&](std::string_view s) {
            trace_output.push_back(std::string{s});
          })));
  std::shared_ptr<InterpValue::Channel> term_chan =
      terminator.GetChannelOrDie();
  while (term_chan->empty()) {
    for (auto& p : proc_instances) {
      XLS_ASSERT_OK(p.Run());
    }
  }
  EXPECT_THAT(
      trace_output,
      testing::ElementsAre(
          "Sent data on channel `tester_proc::(data_out)[0]`:\n  42",
          "Received data on channel `incrementer::in_ch`:\n  42",
          "Sent data on channel `incrementer::out_ch`:\n  43",
          "Received data on channel `tester_proc::(data_in)[0]`:\n  43",
          "Sent data on channel `tester_proc::(data_out)[0]`:\n  100",
          "Received data on channel `incrementer::in_ch`:\n  100",
          "Sent data on channel `incrementer::out_ch`:\n  101",
          "Received data on channel `tester_proc::(data_in)[0]`:\n  101",
          "Sent data on channel `tester_proc::terminator`:\n  1"));
}

}  // namespace
}  // namespace xls::dslx
