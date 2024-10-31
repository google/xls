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
#include "xls/dslx/bytecode/bytecode_emitter.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "xls/common/casts.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"
#include "re2/re2.h"

namespace xls::dslx {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;

absl::StatusOr<std::unique_ptr<BytecodeFunction>> EmitBytecodes(
    ImportData* import_data, std::string_view program,
    std::string_view fn_name) {
  XLS_ASSIGN_OR_RETURN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", import_data));

  XLS_ASSIGN_OR_RETURN(TestFunction * tf, tm.module->GetTest(fn_name));

  return BytecodeEmitter::Emit(import_data, tm.type_info, tf->fn(),
                               std::nullopt);
}

// Verifies that a baseline translation - of a nearly-minimal test case -
// succeeds.
TEST(BytecodeEmitterTest, SimpleTranslation) {
  constexpr std::string_view kProgram = R"(fn one_plus_one() -> u32 {
  let foo = u32:1;
  foo + u32:2
})";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, tm.module->GetMemberOrError<Function>("one_plus_one"));
  ASSERT_TRUE(f != nullptr);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::Emit(&import_data, tm.type_info, *f, ParametricEnv()));

  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 5);
  const Bytecode* bc = bytecodes.data();
  ASSERT_EQ(bc->op(), Bytecode::Op::kLiteral);
  ASSERT_TRUE(bc->has_data());
  ASSERT_EQ(bc->value_data().value(), InterpValue::MakeU32(1));

  bc = &bytecodes[1];
  ASSERT_EQ(bc->op(), Bytecode::Op::kStore);
  ASSERT_TRUE(bc->has_data());
  XLS_ASSERT_OK_AND_ASSIGN(Bytecode::SlotIndex slot_index, bc->slot_index());
  ASSERT_EQ(slot_index.value(), 0);

  bc = &bytecodes[2];
  ASSERT_EQ(bc->op(), Bytecode::Op::kLoad);
  ASSERT_TRUE(bc->has_data());
  XLS_ASSERT_OK_AND_ASSIGN(slot_index, bc->slot_index());
  ASSERT_EQ(slot_index.value(), 0);

  bc = &bytecodes[3];
  ASSERT_EQ(bc->op(), Bytecode::Op::kLiteral);
  ASSERT_TRUE(bc->has_data());
  ASSERT_EQ(bc->value_data().value(), InterpValue::MakeU32(2));

  bc = &bytecodes[4];
  ASSERT_EQ(bc->op(), Bytecode::Op::kUAdd);
  ASSERT_FALSE(bc->has_data());
}

// Validates emission of AssertEq builtins.
TEST(BytecodeEmitterTest, AssertEq) {
  constexpr std::string_view kProgram = R"(#[test]
fn expect_fail() -> u32{
  let foo = u32:3;
  assert_eq(foo, u32:2);
  foo
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      EmitBytecodes(&import_data, kProgram, "expect_fail"));

  EXPECT_EQ(BytecodesToString(bf->bytecodes(), /*source_locs=*/false,
                              import_data.file_table()),
            R"(000 literal u32:3
001 store 0
002 load 0
003 literal u32:2
004 literal builtin:assert_eq
005 call assert_eq(foo, u32:2)
006 pop
007 load 0)");
}

TEST(BytecodeEmitterTest, DestructuringLet) {
  constexpr std::string_view kProgram = R"(#[test]
fn has_name_def_tree() -> (u32, u64, uN[128]) {
  let (a, b, (c, d)) = (u4:0, u8:1, (u16:2, (u32:3, u64:4, uN[128]:5)));
  assert_eq(a, u4:0);
  assert_eq(b, u8:1);
  assert_eq(c, u16:2);
  assert_eq(d, (u32:3, u64:4, uN[128]:5));
  d
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      EmitBytecodes(&import_data, kProgram, "has_name_def_tree"));

  EXPECT_EQ(BytecodesToString(bf->bytecodes(), /*source_locs=*/false,
                              import_data.file_table()),
            R"(000 literal u4:0
001 literal u8:1
002 literal u16:2
003 literal u32:3
004 literal u64:4
005 literal u128:0x5
006 create_tuple 3
007 create_tuple 2
008 create_tuple 3
009 expand_tuple
010 store 0
011 store 1
012 expand_tuple
013 store 2
014 store 3
015 load 0
016 literal u4:0
017 literal builtin:assert_eq
018 call assert_eq(a, u4:0)
019 pop
020 load 1
021 literal u8:1
022 literal builtin:assert_eq
023 call assert_eq(b, u8:1)
024 pop
025 load 2
026 literal u16:2
027 literal builtin:assert_eq
028 call assert_eq(c, u16:2)
029 pop
030 load 3
031 literal u32:3
032 literal u64:4
033 literal u128:0x5
034 create_tuple 3
035 literal builtin:assert_eq
036 call assert_eq(d, (u32:3, u64:4, uN[128]:5))
037 pop
038 load 3)");
}

TEST(BytecodeEmitterTest, DestructuringLetWithRestOfTuple) {
  constexpr std::string_view kProgram = R"(#[test]
fn destructuring_let_with_rest_of_tuple() -> (u32, u64, uN[128]) {
  let (a, b, .., (c, d)) = (u4:0, u8:1, u9:2, (u16:3, (u32:4, u64:5, uN[128]:6)));
  assert_eq(a, u4:0);
  assert_eq(b, u8:1);
  assert_eq(c, u16:3);
  assert_eq(d, (u32:4, u64:5, uN[128]:6));
  d
})";

  ImportData import_data(CreateImportDataForTest());
  // Asserts that we can generate bytecode for this situation. We shouldn't
  // assert on the contents of the bytecode generated, since that is fragile
  // and is essentially a "Change Detector" test.
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      EmitBytecodes(&import_data, kProgram,
                    "destructuring_let_with_rest_of_tuple"));
}

TEST(BytecodeEmitterTest, DestructuringLetWithRestOfTupleNested) {
  constexpr std::string_view kProgram = R"(#[test]
fn destructuring_let_with_rest_of_tuple_nested() -> (u32, u64, uN[128]) {
  let (a, b, .., (c, .., d)) = (u4:0, u8:1, u9:2, (u16:2, u10:4, (u32:5, u64:6, uN[128]:7)));
  assert_eq(a, u4:0);
  assert_eq(b, u8:1);
  assert_eq(c, u16:2);
  assert_eq(d, (u32:5, u64:6, uN[128]:7));
  d
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK(EmitBytecodes(&import_data, kProgram,
                              "destructuring_let_with_rest_of_tuple_nested"));
}

TEST(BytecodeEmitterTest, DestructuringNonConstantTuple) {
  constexpr std::string_view kProgram = R"(#[test]
fn destructuring_non_constant_tuple() -> (u32, u64, uN[128]) {
  let t = (u4:0, u8:1, u9:2, (u16:2, u10:4, (u32:5, u64:6, uN[128]:7)));
  let (a, b, .., (c, .., d)) = t;
  assert_eq(a, u4:0);
  assert_eq(b, u8:1);
  assert_eq(c, u16:2);
  assert_eq(d, (u32:5, u64:6, uN[128]:7));
  d
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK(EmitBytecodes(&import_data, kProgram,
                              "destructuring_non_constant_tuple"));
}

TEST(BytecodeEmitterTest, Ternary) {
  constexpr std::string_view kProgram = R"(#[test]
fn do_ternary() -> u32 {
  if true { u32:42 } else { u32:64 }
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BytecodeFunction> bf,
                           EmitBytecodes(&import_data, kProgram, "do_ternary"));

  EXPECT_EQ(BytecodesToString(bf->bytecodes(), /*source_locs=*/false,
                              import_data.file_table()),
            R"(000 literal u1:1
001 jump_rel_if +3
002 literal u32:64
003 jump_rel +3
004 jump_dest
005 literal u32:42
006 jump_dest)");
}

TEST(BytecodeEmitterTest, CastToXbits) {
  constexpr std::string_view kProgram = R"(#[test]
fn main(x: u1) -> s1 {
  x as xN[true][1]
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BytecodeFunction> bf,
                           EmitBytecodes(&import_data, kProgram, "main"));

  EXPECT_EQ(BytecodesToString(bf->bytecodes(), /*source_locs=*/false,
                              import_data.file_table()),
            R"(000 load 0
001 cast xN[is_signed=1][1])");
}

TEST(BytecodeEmitterTest, Shadowing) {
  constexpr std::string_view kProgram = R"(#[test]
fn f() -> u32 {
  let x = u32:42;
  let x = u32:64;
  x
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BytecodeFunction> bf,
                           EmitBytecodes(&import_data, kProgram, "f"));

  EXPECT_EQ(BytecodesToString(bf->bytecodes(), /*source_locs=*/false,
                              import_data.file_table()),
            R"(000 literal u32:42
001 store 0
002 literal u32:64
003 store 1
004 load 1)");
}

TEST(BytecodeEmitterTest, MatchTuplesWithRestOfTuple) {
  constexpr std::string_view kProgram = R"(#[test]
fn match_tuple() -> u32 {
  let x = (u32:42, u32:64, u32:128);
  let y = match x {
    (u32:42, u32:64, u32:128) => u32:42,
    (u32:41, u32:63, ..) => u32:63,
    (u32:40, ..) => u32:40,
    _ => u32:0
  };
  y
})";

  ImportData import_data(CreateImportDataForTest());
  // Asserts that we can generate bytecode for this situation. We shouldn't
  // assert on the contents of the bytecode generated, since that is fragile
  // and is essentially a "Change Detector" test.
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      EmitBytecodes(&import_data, kProgram, "match_tuple"));
}

TEST(BytecodeEmitterTest, MatchSimpleArms) {
  constexpr std::string_view kProgram = R"(#[test]
fn do_match() -> u32 {
  let x = u32:77;
  match x {
    u32:42 => u32:64,
    u32:64 => u32:42,
    _ => x + u32:1
  }
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BytecodeFunction> bf,
                           EmitBytecodes(&import_data, kProgram, "do_match"));

  EXPECT_EQ(BytecodesToString(bf->bytecodes(), /*source_locs=*/false,
                              import_data.file_table()),
            R"(000 literal u32:77
001 store 0
002 load 0
003 dup
004 match_arm value:u32:42
005 invert
006 jump_rel_if +4
007 pop
008 literal u32:64
009 jump_rel +21
010 jump_dest
011 dup
012 match_arm value:u32:64
013 invert
014 jump_rel_if +4
015 pop
016 literal u32:42
017 jump_rel +13
018 jump_dest
019 dup
020 match_arm wildcard
021 invert
022 jump_rel_if +6
023 pop
024 load 0
025 literal u32:1
026 uadd
027 jump_rel +3
028 jump_dest
029 fail trace data: The value was not matched: value: , default
030 jump_dest)");
}

TEST(BytecodeEmitterTest, BytecodesFromString) {
  std::string s = R"(000 literal u2:1
001 literal s2:-1
002 literal s2:-2
003 literal s3:-1
004 literal u32:42)";
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Bytecode> bytecodes,
                           BytecodesFromString(s));
  EXPECT_THAT(bytecodes.at(3).value_data(),
              IsOkAndHolds(InterpValue::MakeSBits(3, -1)));
  FileTable file_table;
  EXPECT_EQ(BytecodesToString(bytecodes, /*source_locs=*/false, file_table), s);
}

// Tests emission of all of the supported binary operators.
TEST(BytecodeEmitterTest, Binops) {
  constexpr std::string_view kProgram = R"(#[test]
fn binops_galore() {
  let a = u32:4;
  let b = u32:2;

  let uadd = a + b;
  let uand = a & b;
  let uconcat = a ++ b;
  let udiv = a / b;
  let ueq = a == b;
  let uge = a >= b;
  let ugt = a > b;
  let ule = a <= b;
  let ult = a < b;
  let umul = a * b;
  let une = a != b;
  let uor = a | b;
  let ushl = a << b;
  let ushr = a >> b;
  let usub = a - b;
  let uxor = a ^ b;

  let c = s32:4;
  let d = s32:2;
  let sadd = c + d;
  let sand = c & d;
  let sdiv = c / d;
  let seq = c == d;
  let sge = c >= d;
  let sgt = c > d;
  let sle = c <= d;
  let slt = c < d;
  let smul = c * d;
  let sne = c != d;
  let sor = c | d;
  let ssub = c - d;
  let sxor = c ^ d;
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      EmitBytecodes(&import_data, kProgram, "binops_galore"));

  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  std::string got = BytecodesToString(bytecodes, /*source_locs=*/false,
                                      import_data.file_table());
  std::vector<std::string_view> opcodes;
  for (std::string_view line : absl::StrSplit(got, '\n')) {
    if (RE2::FullMatch(line, RE2(R"(^\d+ (literal|store|load).*$)"))) {
      continue;
    }
    opcodes.push_back(line);
  }
  EXPECT_EQ(absl::StrJoin(opcodes, "\n"), R"(006 uadd
010 and
014 concat
018 div
022 eq
026 ge
030 gt
034 le
038 lt
042 umul
046 ne
050 or
054 shl
058 shr
062 usub
066 xor
074 sadd
078 and
082 div
086 eq
090 ge
094 gt
098 le
102 lt
106 smul
110 ne
114 or
118 ssub
122 xor)");
}

// Tests emission of all of the supported binary operators.
TEST(BytecodeEmitterTest, Unops) {
  constexpr std::string_view kProgram = R"(#[test]
fn unops() {
  let a = s32:32;
  let b = !a;
  let c = -b;
  ()
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BytecodeFunction> bf,
                           EmitBytecodes(&import_data, kProgram, "unops"));

  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 9);
  const Bytecode* bc = &bytecodes[3];
  ASSERT_EQ(bc->op(), Bytecode::Op::kInvert);

  bc = &bytecodes[6];
  ASSERT_EQ(bc->op(), Bytecode::Op::kNegate);
}

// Tests array creation.
TEST(BytecodeEmitterTest, Arrays) {
  constexpr std::string_view kProgram = R"(#[test]
fn arrays() -> u32[3] {
  let a = u32:32;
  u32[3]:[u32:0, u32:1, a]
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BytecodeFunction> bf,
                           EmitBytecodes(&import_data, kProgram, "arrays"));

  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 6);
  const Bytecode* bc = &bytecodes[5];
  ASSERT_EQ(bc->op(), Bytecode::Op::kCreateArray);
  ASSERT_TRUE(bc->has_data());
  XLS_ASSERT_OK_AND_ASSIGN(Bytecode::NumElements num_elements,
                           bc->num_elements());
  ASSERT_EQ(num_elements.value(), 3);
}

// Tests large constexpr 2D array creation doesn't create a skillion bytecodes.
TEST(BytecodeEmitterTest, TwoDimensionalArrayLiteral) {
  constexpr std::string_view kProgram = R"(#[test]
fn make_2d_array() -> u32[1024][1024] {
  const A: u32[1024][1024] = u32[1024][1024]:[u32[1024]:[0, ...], ...];
  A
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      EmitBytecodes(&import_data, kProgram, "make_2d_array"));

  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 3);
}

// Tests emission of kIndex ops on arrays.
TEST(BytecodeEmitterTest, IndexArray) {
  constexpr std::string_view kProgram = R"(#[test]
fn index_array() -> u32 {
  let a = u32[3]:[0, 1, 2];
  let b = bits[32][3]:[3, 4, 5];

  a[u32:0] + b[u32:1]
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      EmitBytecodes(&import_data, kProgram, "index_array"));

  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 11);

  const std::string_view kWant =
      R"(literal [u32:0, u32:1, u32:2]
store 0
literal [u32:3, u32:4, u32:5]
store 1
load 0
literal u32:0
index
load 1
literal u32:1
index
uadd)";
  std::string got = absl::StrJoin(
      bf->bytecodes(), "\n",
      [&import_data](std::string* out, const Bytecode& b) {
        absl::StrAppend(out, b.ToString(import_data.file_table()));
      });

  EXPECT_EQ(kWant, got);
}

// Tests emission of kIndex ops on tuples.
TEST(BytecodeEmitterTest, IndexTuple) {
  constexpr std::string_view kProgram = R"(#[test]
fn index_tuple() -> u32 {
  let a = (u16:0, u32:1, u64:2);
  let b = (bits[128]:3, bits[32]:4);

  a.1 + b.1
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      EmitBytecodes(&import_data, kProgram, "index_tuple"));

  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 16);
  const Bytecode* bc = &bytecodes[11];
  ASSERT_EQ(bc->op(), Bytecode::Op::kIndex);

  bc = &bytecodes[14];
  ASSERT_EQ(bc->op(), Bytecode::Op::kIndex);
}

// Tests a regular a[x:y] slice op.
TEST(BytecodeEmitterTest, SimpleSlice) {
  constexpr std::string_view kProgram = R"(#[test]
fn simple_slice() -> u16 {
  let a = u32:0xdeadbeef;
  a[16:32]
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      EmitBytecodes(&import_data, kProgram, "simple_slice"));

  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 6);
  const Bytecode* bc = &bytecodes[3];
  ASSERT_EQ(bc->op(), Bytecode::Op::kLiteral);

  bc = &bytecodes[4];
  ASSERT_EQ(bc->op(), Bytecode::Op::kLiteral);

  bc = &bytecodes[5];
  ASSERT_EQ(bc->op(), Bytecode::Op::kSlice);
}

// Tests a slice from the start: a[-x:].
TEST(BytecodeEmitterTest, NegativeStartSlice) {
  constexpr std::string_view kProgram = R"(#[test]
fn negative_start_slice() -> u16 {
  let a = u32:0xdeadbeef;
  a[-16:]
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      EmitBytecodes(&import_data, kProgram, "negative_start_slice"));

  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 6);
  const Bytecode* bc = &bytecodes[3];
  ASSERT_EQ(bc->op(), Bytecode::Op::kLiteral);

  bc = &bytecodes[4];
  ASSERT_EQ(bc->op(), Bytecode::Op::kLiteral);

  bc = &bytecodes[5];
  ASSERT_EQ(bc->op(), Bytecode::Op::kSlice);
}

// Tests a slice from the end: a[:-x].
TEST(BytecodeEmitterTest, NegativeEndSlice) {
  constexpr std::string_view kProgram = R"(#[test]
fn negative_end_slice() -> u16 {
  let a = u32:0xdeadbeef;
  a[:-16]
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      EmitBytecodes(&import_data, kProgram, "negative_end_slice"));

  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 6);
  const Bytecode* bc = &bytecodes[3];
  ASSERT_EQ(bc->op(), Bytecode::Op::kLiteral);

  bc = &bytecodes[4];
  ASSERT_EQ(bc->op(), Bytecode::Op::kLiteral);

  bc = &bytecodes[5];
  ASSERT_EQ(bc->op(), Bytecode::Op::kSlice);
}

// Tests a slice from both ends: a[-x:-y].
TEST(BytecodeEmitterTest, BothNegativeSlice) {
  constexpr std::string_view kProgram = R"(#[test]
fn both_negative_slice() -> u8 {
  let a = u32:0xdeadbeef;
  a[-16:-8]
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      EmitBytecodes(&import_data, kProgram, "both_negative_slice"));

  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 6);
  const Bytecode* bc = &bytecodes[3];
  ASSERT_EQ(bc->op(), Bytecode::Op::kLiteral);

  bc = &bytecodes[4];
  ASSERT_EQ(bc->op(), Bytecode::Op::kLiteral);

  bc = &bytecodes[5];
  ASSERT_EQ(bc->op(), Bytecode::Op::kSlice);
}

// Tests the width slice op.
TEST(BytecodeEmitterTest, WidthSlice) {
  constexpr std::string_view kProgram = R"(#[test]
fn width_slice() -> u16 {
  let a = u32:0xdeadbeef;
  a[u32:8 +: bits[16]]
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      EmitBytecodes(&import_data, kProgram, "width_slice"));
  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 5);

  const Bytecode* bc = &bytecodes[3];
  ASSERT_EQ(bc->op(), Bytecode::Op::kLiteral);

  bc = &bytecodes[4];
  ASSERT_EQ(bc->op(), Bytecode::Op::kWidthSlice);
}

TEST(BytecodeEmitterTest, LocalEnumRef) {
  constexpr std::string_view kProgram = R"(enum MyEnum : u23 {
  VAL_0 = 0,
  VAL_1 = 1,
  VAL_2 = 2,
}

#[test]
fn local_enum_ref() -> MyEnum {
  MyEnum::VAL_1
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      EmitBytecodes(&import_data, kProgram, "local_enum_ref"));
  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 1);

  const Bytecode* bc = bytecodes.data();
  ASSERT_EQ(bc->op(), Bytecode::Op::kLiteral);
  ASSERT_TRUE(bc->has_data());
  EXPECT_THAT(bytecodes.at(0).value_data(),
              IsOkAndHolds(InterpValue::MakeSBits(23, 1)));
}

TEST(BytecodeEmitterTest, ImportedEnumRef) {
  constexpr std::string_view kImportedProgram = R"(pub enum ImportedEnum : u4 {
  VAL_0 = 0,
  VAL_1 = 1,
  VAL_2 = 2,
  VAL_3 = 3,
}
)";
  constexpr std::string_view kBaseProgram = R"(
import import_0;

#[test]
fn imported_enum_ref() -> import_0::ImportedEnum {
  import_0::ImportedEnum::VAL_2
}
)";

  auto import_data = CreateImportDataForTest();

  {
    XLS_ASSERT_OK_AND_ASSIGN(TypecheckedModule tm,
                             ParseAndTypecheck(kImportedProgram, "import_0.x",
                                               "import_0", &import_data));
  }

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kBaseProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(TestFunction * tf,
                           tm.module->GetTest("imported_enum_ref"));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BytecodeFunction> bf,
                           BytecodeEmitter::Emit(&import_data, tm.type_info,
                                                 tf->fn(), ParametricEnv()));

  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 1);

  const Bytecode* bc = bytecodes.data();
  ASSERT_EQ(bc->op(), Bytecode::Op::kLiteral);
  ASSERT_TRUE(bc->has_data());
  EXPECT_THAT(bytecodes.at(0).value_data(),
              IsOkAndHolds(InterpValue::MakeSBits(4, 2)));
}

TEST(BytecodeEmitterTest, StructImplConstant) {
  constexpr std::string_view kBaseProgram = R"(
struct Empty {}

impl Empty {
  const MY_CONST = u4:7;
}

#[test]
fn struct_const_ref() -> u4 {
  Empty::MY_CONST
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kBaseProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(TestFunction * tf,
                           tm.module->GetTest("struct_const_ref"));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BytecodeFunction> bf,
                           BytecodeEmitter::Emit(&import_data, tm.type_info,
                                                 tf->fn(), ParametricEnv()));

  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 1);

  const Bytecode* bc = bytecodes.data();
  ASSERT_EQ(bc->op(), Bytecode::Op::kLiteral);
  ASSERT_TRUE(bc->has_data());
  EXPECT_THAT(bytecodes.at(0).value_data(),
              IsOkAndHolds(InterpValue::MakeSBits(4, 7)));
}

TEST(BytecodeEmitterTest, StructImplConstantParametric) {
  constexpr std::string_view kBaseProgram = R"(
struct Empty<N: u32> {}

impl Empty<N> {
  const MY_CONST = uN[N]:7;
}

#[test]
fn struct_const_ref() -> u4 {
  type MyEmpty = Empty<u32:4>;
  MyEmpty::MY_CONST
}
)";

  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kBaseProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(TestFunction * tf,
                           tm.module->GetTest("struct_const_ref"));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BytecodeFunction> bf,
                           BytecodeEmitter::Emit(&import_data, tm.type_info,
                                                 tf->fn(), ParametricEnv()));

  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 1);

  const Bytecode* bc = bytecodes.data();
  ASSERT_EQ(bc->op(), Bytecode::Op::kLiteral);
  ASSERT_TRUE(bc->has_data());
  EXPECT_THAT(bytecodes.at(0).value_data(),
              IsOkAndHolds(InterpValue::MakeSBits(4, 7)));
}

TEST(BytecodeEmitterTest, ImportedConstant) {
  constexpr std::string_view kImportedProgram = R"(pub const MY_CONST = u3:2;)";
  constexpr std::string_view kBaseProgram = R"(
import import_0;

#[test]
fn imported_enum_ref() -> u3 {
  import_0::MY_CONST
}
)";

  auto import_data = CreateImportDataForTest();

  {
    XLS_ASSERT_OK_AND_ASSIGN(TypecheckedModule tm,
                             ParseAndTypecheck(kImportedProgram, "import_0.x",
                                               "import_0", &import_data));
  }

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kBaseProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(TestFunction * tf,
                           tm.module->GetTest("imported_enum_ref"));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BytecodeFunction> bf,
                           BytecodeEmitter::Emit(&import_data, tm.type_info,
                                                 tf->fn(), ParametricEnv()));

  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 1);

  const Bytecode* bc = bytecodes.data();
  ASSERT_EQ(bc->op(), Bytecode::Op::kLiteral);
  ASSERT_TRUE(bc->has_data());
  EXPECT_THAT(bytecodes.at(0).value_data(),
              IsOkAndHolds(InterpValue::MakeSBits(3, 2)));
}

TEST(BytecodeEmitterTest, HandlesConstRefs) {
  constexpr std::string_view kProgram = R"(const kFoo = u32:100;

#[test]
fn handles_const_refs() -> u32 {
  let a = u32:200;
  a + kFoo
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      EmitBytecodes(&import_data, kProgram, "handles_const_refs"));
  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 5);
  const Bytecode* bc = &bytecodes[3];
  ASSERT_EQ(bc->op(), Bytecode::Op::kLiteral);
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, bc->value_data());
  XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitValueViaSign());
  ASSERT_EQ(int_value, 100);
}

TEST(BytecodeEmitterTest, HandlesStructInstances) {
  constexpr std::string_view kProgram = R"(struct MyStruct {
  x: u32,
  y: u64,
}

#[test]
fn handles_struct_instances() -> MyStruct {
  let x = u32:2;
  MyStruct { x: x, y: u64:3 }
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      EmitBytecodes(&import_data, kProgram, "handles_struct_instances"));
  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 5);
  const Bytecode* bc = &bytecodes[4];
  ASSERT_EQ(bc->op(), Bytecode::Op::kCreateTuple);
}

TEST(BytecodeEmitterTest, HandlesAttr) {
  constexpr std::string_view kProgram = R"(struct MyStruct {
  x: u32,
  y: u64,
}

#[test]
fn handles_attr() -> u64 {
  MyStruct { x: u32:0, y: u64:0xbeef }.y
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      EmitBytecodes(&import_data, kProgram, "handles_attr"));
  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 5);
  const Bytecode* bc = &bytecodes[4];
  ASSERT_EQ(bc->op(), Bytecode::Op::kTupleIndex);
}

TEST(BytecodeEmitterTest, CastBitsToBits) {
  constexpr std::string_view kProgram = R"(#[test]
fn cast_bits_to_bits() -> u64 {
  let a = s16:-4;
  a as u64
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      EmitBytecodes(&import_data, kProgram, "cast_bits_to_bits"));
  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 4);
  const Bytecode* bc = &bytecodes[3];
  ASSERT_EQ(bc->op(), Bytecode::Op::kCast);
}

TEST(BytecodeEmitterTest, CastArrayToBits) {
  constexpr std::string_view kProgram = R"(#[test]
fn cast_array_to_bits() -> u32 {
  let a = u8[4]:[0xc, 0xa, 0xf, 0xe];
  a as u32
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      EmitBytecodes(&import_data, kProgram, "cast_array_to_bits"));
  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 4);

  const std::string_view kWant =
      R"(literal [u8:12, u8:10, u8:15, u8:14]
store 0
load 0
cast uN[32])";
  std::string got = absl::StrJoin(
      bf->bytecodes(), "\n",
      [&import_data](std::string* out, const Bytecode& b) {
        absl::StrAppend(out, b.ToString(import_data.file_table()));
      });

  EXPECT_EQ(kWant, got);
}

TEST(BytecodeEmitterTest, CastBitsToArray) {
  constexpr std::string_view kProgram = R"(#[test]
fn cast_bits_to_array() -> u8 {
  let a = u32:0x0c0a0f0e;
  let b = a as u8[4];
  b[u32:2]
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      EmitBytecodes(&import_data, kProgram, "cast_bits_to_array"));
  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 8);
  const Bytecode* bc = &bytecodes[3];
  ASSERT_EQ(bc->op(), Bytecode::Op::kCast);
}

TEST(BytecodeEmitterTest, CastEnumToBits) {
  constexpr std::string_view kProgram = R"(enum MyEnum : u3 {
  VAL_0 = 0,
  VAL_1 = 1,
  VAL_2 = 2,
  VAL_3 = 3,
}

#[test]
fn cast_enum_to_bits() -> u3 {
  let a = MyEnum::VAL_3;
  a as u3
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      EmitBytecodes(&import_data, kProgram, "cast_enum_to_bits"));
  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 4);
  const Bytecode* bc = &bytecodes[3];
  ASSERT_EQ(bc->op(), Bytecode::Op::kCast);
}

TEST(BytecodeEmitterTest, CastBitsToEnum) {
  constexpr std::string_view kProgram = R"(enum MyEnum : u3 {
  VAL_0 = 0,
  VAL_1 = 1,
  VAL_2 = 2,
  VAL_3 = 3,
}

#[test]
fn cast_bits_to_enum() -> MyEnum {
  let a = u3:2;
  a as MyEnum
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      EmitBytecodes(&import_data, kProgram, "cast_bits_to_enum"));
  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 4);
  const Bytecode* bc = &bytecodes[3];
  ASSERT_EQ(bc->op(), Bytecode::Op::kCast);
}

TEST(BytecodeEmitterTest, HandlesSplatStructInstances) {
  constexpr std::string_view kProgram = R"(struct MyStruct {
  x: u16,
  y: u32,
  z: u64,
}

#[test]
fn handles_struct_instances() -> MyStruct {
  let a = u16:2;
  let b = MyStruct { z: u64:0xbeef, x: a, y: u32:3 };
  MyStruct { y:u32:0xf00d, ..b }
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      EmitBytecodes(&import_data, kProgram, "handles_struct_instances"));

  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  const Bytecode* bc = &bytecodes[7];
  ASSERT_EQ(bc->op(), Bytecode::Op::kLoad);
  bc = &bytecodes[8];
  ASSERT_EQ(bc->op(), Bytecode::Op::kLiteral);
  bc = &bytecodes[9];
  ASSERT_EQ(bc->op(), Bytecode::Op::kIndex);

  bc = &bytecodes[10];
  ASSERT_EQ(bc->op(), Bytecode::Op::kLiteral);

  bc = &bytecodes[11];
  ASSERT_EQ(bc->op(), Bytecode::Op::kLoad);
  bc = &bytecodes[12];
  ASSERT_EQ(bc->op(), Bytecode::Op::kLiteral);
  bc = &bytecodes[13];
  ASSERT_EQ(bc->op(), Bytecode::Op::kIndex);
}

TEST(BytecodeEmitterTest, Params) {
  constexpr std::string_view kProgram = R"(
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
  ASSERT_TRUE(f != nullptr);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::Emit(&import_data, tm.type_info, *f, ParametricEnv()));

  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 15);

  const Bytecode* bc = &bytecodes[2];
  EXPECT_EQ(bc->op(), Bytecode::Op::kLoad);
  ASSERT_TRUE(bc->has_data());
  XLS_ASSERT_OK_AND_ASSIGN(Bytecode::SlotIndex slot_index, bc->slot_index());
  ASSERT_EQ(slot_index.value(), 0);

  bc = &bytecodes[7];
  EXPECT_EQ(bc->op(), Bytecode::Op::kLoad);
  ASSERT_TRUE(bc->has_data());
  XLS_ASSERT_OK_AND_ASSIGN(slot_index, bc->slot_index());
  ASSERT_EQ(slot_index.value(), 3);

  bc = &bytecodes[8];
  EXPECT_EQ(bc->op(), Bytecode::Op::kLoad);
  ASSERT_TRUE(bc->has_data());
  XLS_ASSERT_OK_AND_ASSIGN(slot_index, bc->slot_index());
  ASSERT_EQ(slot_index.value(), 1);

  bc = &bytecodes[12];
  EXPECT_EQ(bc->op(), Bytecode::Op::kLoad);
  ASSERT_TRUE(bc->has_data());
  XLS_ASSERT_OK_AND_ASSIGN(slot_index, bc->slot_index());
  ASSERT_EQ(slot_index.value(), 3);

  bc = &bytecodes[13];
  EXPECT_EQ(bc->op(), Bytecode::Op::kLoad);
  ASSERT_TRUE(bc->has_data());
  XLS_ASSERT_OK_AND_ASSIGN(slot_index, bc->slot_index());
  ASSERT_EQ(slot_index.value(), 4);
}

TEST(BytecodeEmitterTest, Strings) {
  constexpr std::string_view kProgram = R"(
#[test]
fn main() -> u8[13] {
  "tofu sandwich"
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BytecodeFunction> bf,
                           EmitBytecodes(&import_data, kProgram, "main"));

  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 1);
  const Bytecode* bc = bytecodes.data();
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, bc->value_data());
  XLS_ASSERT_OK_AND_ASSIGN(int64_t length, value.GetLength());
  EXPECT_EQ(13, length);
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t char_value,
                           value.GetValuesOrDie().at(0).GetBitValueUnsigned());
  EXPECT_EQ(char_value, 't');
}

TEST(BytecodeEmitterTest, SimpleParametric) {
  constexpr std::string_view kProgram = R"(
fn foo<N: u32>(x: uN[N]) -> uN[N] {
  x * x
}

#[test]
fn main() -> u32 {
  let a = foo<u32:16>(u16:4);
  let b = foo(u32:8);
  a as u32 + b
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BytecodeFunction> bf,
                           EmitBytecodes(&import_data, kProgram, "main"));

  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 12);
  const Bytecode* bc = &bytecodes[2];
  EXPECT_EQ(bc->op(), Bytecode::Op::kCall);
  XLS_ASSERT_OK_AND_ASSIGN(Bytecode::InvocationData id, bc->invocation_data());
  NameRef* name_ref = dynamic_cast<NameRef*>(id.invocation()->callee());
  ASSERT_NE(name_ref, nullptr);
  EXPECT_EQ(name_ref->identifier(), "foo");

  bc = &bytecodes[6];
  EXPECT_EQ(bc->op(), Bytecode::Op::kCall);
  XLS_ASSERT_OK_AND_ASSIGN(id, bc->invocation_data());
  name_ref = dynamic_cast<NameRef*>(id.invocation()->callee());
  ASSERT_NE(name_ref, nullptr);
  EXPECT_EQ(name_ref->identifier(), "foo");
}

TEST(BytecodeEmitterTest, SimpleFor) {
  constexpr std::string_view kProgram = R"(#[test]
fn main() -> u32 {
  for (i, accum) : (u32, u32) in range(u32:0, u32:8) {
    accum + i
  }(u32:1)
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BytecodeFunction> bf,
                           EmitBytecodes(&import_data, kProgram, "main"));

  // Since `for` generates a complex set of bytecodes, we test. every. one.
  // To make that a bit easier, we do string comparison.
  const std::vector<std::string> kExpected = {
      "literal u32:0",
      "literal u32:8",
      "literal builtin:range",
      "call range(u32:0, u32:8)",
      "store 0",
      "literal u32:0",
      "store 1",
      "literal u32:1",
      "jump_dest",
      "load 1",
      "literal u32:8",
      "eq",
      "jump_rel_if +17",
      "load 0",
      "load 1",
      "index",
      "swap",
      "create_tuple 2",
      "expand_tuple",
      "store 2",
      "store 3",
      "load 3",
      "load 2",
      "uadd",
      "load 1",
      "literal u32:1",
      "uadd",
      "store 1",
      "jump_rel -20",
      "jump_dest",
  };

  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 30);
  for (int i = 0; i < bytecodes.size(); i++) {
    ASSERT_EQ(
        bytecodes[i].ToString(import_data.file_table(), /*source_locs=*/false),
        kExpected[i]);
  }
}

TEST(BytecodeEmitterTest, ForWithCover) {
  constexpr std::string_view kProgram = R"(
struct SomeStruct {
  some_bool: bool
}

#[test]
fn test_main(s: SomeStruct) {
  for  (_, ()) in u32:0..u32:4 {
    let _ = cover!("whee", s.some_bool);
    ()
  }(())
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BytecodeFunction> bf,
                           EmitBytecodes(&import_data, kProgram, "test_main"));

  const std::string_view kWant = R"(literal u32:0
literal u32:4
range
store 1
literal u32:0
store 2
create_tuple 0
jump_dest
load 2
literal u32:4
eq
jump_rel_if +22
load 1
load 2
index
swap
create_tuple 2
expand_tuple
pop
expand_tuple
literal [u8:119, u8:104, u8:101, u8:101]
load 0
literal u64:0
tuple_index
literal builtin:cover!
call cover!("whee", s.some_bool)
pop
create_tuple 0
load 2
literal u32:1
uadd
store 2
jump_rel -25
jump_dest)";
  std::string got = absl::StrJoin(
      bf->bytecodes(), "\n",
      [&import_data](std::string* out, const Bytecode& b) {
        absl::StrAppend(
            out, b.ToString(import_data.file_table(), /*source_locs=*/false));
      });

  EXPECT_EQ(kWant, got);
}

TEST(BytecodeEmitterTest, Range) {
  constexpr std::string_view kProgram = R"(#[test]
fn main() -> u32[8] {
  let x = u32:8;
  let y = u32:16;
  x..y
})";
  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BytecodeFunction> bf,
                           EmitBytecodes(&import_data, kProgram, "main"));

  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 7);
  const Bytecode* bc = &bytecodes[6];
  ASSERT_EQ(bc->op(), Bytecode::Op::kRange);
}

TEST(BytecodeEmitterTest, ShlAndShr) {
  constexpr std::string_view kProgram = R"(#[test]
fn main() -> u32 {
  let x = u32:8;
  let y = u32:16;
  x << y >> y
})";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BytecodeFunction> bf,
                           EmitBytecodes(&import_data, kProgram, "main"));

  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 9);
  const Bytecode* bc = &bytecodes[6];
  ASSERT_EQ(bc->op(), Bytecode::Op::kShl);

  bc = &bytecodes[8];
  ASSERT_EQ(bc->op(), Bytecode::Op::kShr);
}

TEST(BytecodeEmitterTest, ParameterizedTypeDefToImportedEnum) {
  constexpr std::string_view kImported = R"(
pub struct ImportedStruct<X: u32> {
  x: uN[X],
}

pub enum ImportedEnum : u32 {
  EAT = 0,
  YOUR = 1,
  VEGGIES = 2
})";

  constexpr std::string_view kProgram = R"(
import imported;

type MyEnum = imported::ImportedEnum;
type MyStruct = imported::ImportedStruct<16>;

#[test]
fn main() -> u32 {
  let foo = MyStruct { x: u16:100 };
  foo.x as u32 + (MyEnum::VEGGIES as u32)
}

)";

  ImportData import_data(CreateImportDataForTest());

  {
    XLS_ASSERT_OK_AND_ASSIGN(
        TypecheckedModule tm,
        ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));
  }

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(TestFunction * tf, tm.module->GetTest("main"));

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BytecodeFunction> bf,
                           BytecodeEmitter::Emit(&import_data, tm.type_info,
                                                 tf->fn(), ParametricEnv()));

  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 10);
}

TEST(BytecodeEmitterTest, BasicProc) {
  // We can only test 0-arg procs (both config and next), since procs are only
  // typechecked if spawned by a top-level (i.e., 0-arg) proc.
  constexpr std::string_view kProgram = R"(
proc Foo {
  x: chan<u32> in;
  y: u32;
  init { () }
  config() {
    let (p, c) = chan<u32>("my_chan");
    (c, u32:100)
  }

  next(state: ()) {
    ()
  }
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * p, tm.module->GetMemberOrError<Proc>("Foo"));
  XLS_ASSERT_OK_AND_ASSIGN(TypeInfo * ti,
                           tm.type_info->GetTopLevelProcTypeInfo(p));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::Emit(&import_data, ti, p->config(), ParametricEnv()));
  const std::vector<Bytecode>& config_bytecodes = bf->bytecodes();
  ASSERT_EQ(config_bytecodes.size(), 7);
  const std::vector<std::string> kConfigExpected = {
      "literal (channel, channel)",
      "expand_tuple",
      "store 0",
      "store 1",
      "load 1",
      "literal u32:100",
      "create_tuple 2"};

  for (int i = 0; i < config_bytecodes.size(); i++) {
    ASSERT_EQ(config_bytecodes[i].ToString(import_data.file_table()),
              kConfigExpected[i]);
  }
}

TEST(BytecodeEmitterTest, SpawnedProc) {
  constexpr std::string_view kProgram = R"(
proc Child {
  c: chan<u32> in;
  x: u32;
  y: u64;

  config(c: chan<u32> in, a: u64, b: uN[128]) {
    (c, a as u32, (a + b as u64))
  }

  init {
    u64:1234
  }

  next(a: u64) {
    let (tok, b) = recv(join(), c);
    a + x as u64 + y + b as u64
  }
}

proc Parent {
  p: chan<u32> out;
  init { () }
  config() {
    let (p, c) = chan<u32>("my_chan");
    spawn Child(c, u64:100, uN[128]:200);
    (p,)
  }

  next(state: ()) {
    ()
  }
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * parent,
                           tm.module->GetMemberOrError<Proc>("Parent"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * child,
                           tm.module->GetMemberOrError<Proc>("Child"));

  StatementBlock* config_body = parent->config().body();
  EXPECT_EQ(config_body->statements().size(), 3);
  Spawn* spawn = down_cast<Spawn*>(
      std::get<Expr*>(config_body->statements().at(1)->wrapped()));
  XLS_ASSERT_OK_AND_ASSIGN(TypeInfo * parent_ti,
                           tm.type_info->GetTopLevelProcTypeInfo(parent));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> parent_config_bf,
      BytecodeEmitter::Emit(&import_data, parent_ti, parent->config(),
                            ParametricEnv()));
  const std::vector<Bytecode>& parent_config_bytecodes =
      parent_config_bf->bytecodes();
  const std::vector<std::string> kParentConfigExpected = {
      "literal (channel, channel)",
      "expand_tuple",
      "store 0",
      "store 1",
      "load 1",
      "literal u64:100",
      "literal u128:0xc8",
      "spawn spawn (c, u64:100, uN[128]:200)",
      "pop",
      "load 0",
      "create_tuple 1",
  };
  ASSERT_EQ(parent_config_bytecodes.size(), kParentConfigExpected.size());
  for (int i = 0; i < parent_config_bytecodes.size(); i++) {
    ASSERT_EQ(parent_config_bytecodes[i].ToString(import_data.file_table()),
              kParentConfigExpected[i]);
  }

  TypeInfo* child_ti =
      parent_ti->GetInvocationTypeInfo(spawn->config(), ParametricEnv())
          .value();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::Emit(&import_data, child_ti, child->config(),
                            ParametricEnv()));
  const std::vector<Bytecode>& config_bytecodes = bf->bytecodes();
  ASSERT_EQ(config_bytecodes.size(), 8);
  const std::vector<std::string> kConfigExpected = {
      "load 0",          //
      "load 1",          //
      "cast uN[32]",     //
      "load 1",          //
      "load 2",          //
      "cast uN[64]",     //
      "uadd",            //
      "create_tuple 3",  //
  };
  for (int i = 0; i < config_bytecodes.size(); i++) {
    ASSERT_EQ(config_bytecodes[i].ToString(import_data.file_table()),
              kConfigExpected[i]);
  }

  std::vector<NameDef*> members;
  for (const ProcMember* member : child->members()) {
    members.push_back(member->name_def());
  }
  child_ti =
      parent_ti->GetInvocationTypeInfo(spawn->next(), ParametricEnv()).value();
  XLS_ASSERT_OK_AND_ASSIGN(
      bf, BytecodeEmitter::EmitProcNext(&import_data, child_ti, child->next(),
                                        ParametricEnv(), members));
  const std::vector<Bytecode>& next_bytecodes = bf->bytecodes();
  std::vector<std::string> next_bytecode_strings;
  absl::c_transform(next_bytecodes, std::back_inserter(next_bytecode_strings),
                    [&import_data](const Bytecode& bc) {
                      return bc.ToString(import_data.file_table());
                    });
  EXPECT_THAT(next_bytecode_strings,
              ElementsAre(testing::MatchesRegex("literal token:0x[0-9a-f]+"),
                          "load 0",         //
                          "literal u1:1",   //
                          "literal u32:0",  //
                          "recv Child::c",  //
                          "expand_tuple",   //
                          "store 4",        //
                          "store 5",        //
                          "load 3",         //
                          "load 1",         //
                          "cast uN[64]",    //
                          "uadd",           //
                          "load 2",         //
                          "uadd",           //
                          "load 5",         //
                          "cast uN[64]",    //
                          "uadd"));
}

// Verifies no explosions when calling BytecodeEmitter::EmitExpression with an
// import in the NameDef environment.
TEST(BytecodeEmitterTest, EmitExpressionWithImport) {
  constexpr std::string_view kImported = R"(
pub const MY_CONST = u32:4;
)";
  constexpr std::string_view kProgram = R"(
import imported as mod;

#[test]
fn main() -> u32 {
  mod::MY_CONST + u32:1
}
)";

  ImportData import_data(CreateImportDataForTest());

  {
    XLS_ASSERT_OK_AND_ASSIGN(
        TypecheckedModule tm,
        ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));
  }

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(TestFunction * tf, tm.module->GetTest("main"));
  Function& f = tf->fn();
  Expr* body = f.body();

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BytecodeFunction> bf,
                           BytecodeEmitter::EmitExpression(
                               &import_data, tm.type_info, body, /*env=*/{},
                               /*caller_bindings=*/std::nullopt));
  const std::vector<Bytecode>& bytecodes = bf->bytecodes();
  ASSERT_EQ(bytecodes.size(), 3);
  const std::vector<std::string> kNextExpected = {
      "literal u32:4",  //
      "literal u32:1",  //
      "uadd"            //
  };
  for (int i = 0; i < bytecodes.size(); i++) {
    ASSERT_EQ(bytecodes[i].ToString(import_data.file_table()),
              kNextExpected[i]);
  }
}

}  // namespace
}  // namespace xls::dslx
