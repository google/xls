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
#include "xls/dslx/bytecode_emitter.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/string_view.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {
namespace {

using status_testing::IsOkAndHolds;

// Verifies that a baseline translation - of a nearly-minimal test case -
// succeeds.
TEST(BytecodeEmitterTest, SimpleTranslation) {
  constexpr absl::string_view kProgram = R"(fn one_plus_one() -> u32 {
  let foo = u32:1;
  foo + u32:2
})";

  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  absl::flat_hash_map<const NameDef*, int64_t> namedef_to_slot;
  BytecodeEmitter emitter(&import_data, tm.type_info, &namedef_to_slot);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetFunctionOrError("one_plus_one"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Bytecode> bytecodes, emitter.Emit(f));

  ASSERT_EQ(bytecodes.size(), 5);
  Bytecode bc = bytecodes[0];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLiteral);
  ASSERT_TRUE(bc.has_data());
  ASSERT_EQ(bc.value_data().value(), InterpValue::MakeU32(1));

  bc = bytecodes[1];
  ASSERT_EQ(bc.op(), Bytecode::Op::kStore);
  ASSERT_TRUE(bc.has_data());
  ASSERT_EQ(bc.integer_data().value(), 0);

  bc = bytecodes[2];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLoad);
  ASSERT_TRUE(bc.has_data());
  ASSERT_EQ(bc.integer_data().value(), 0);

  bc = bytecodes[3];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLiteral);
  ASSERT_TRUE(bc.has_data());
  ASSERT_EQ(bc.value_data().value(), InterpValue::MakeU32(2));

  bc = bytecodes[4];
  ASSERT_EQ(bc.op(), Bytecode::Op::kAdd);
  ASSERT_FALSE(bc.has_data());
}

// Validates emission of AssertEq builtins.
TEST(BytecodeEmitterTest, AssertEq) {
  constexpr absl::string_view kProgram = R"(#![test]
fn expect_fail() -> u32{
  let foo = u32:3;
  let _ = assert_eq(foo, u32:2);
  foo
})";

  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  absl::flat_hash_map<const NameDef*, int64_t> namedef_to_slot;
  BytecodeEmitter emitter(&import_data, tm.type_info, &namedef_to_slot);

  XLS_ASSERT_OK_AND_ASSIGN(TestFunction * tf,
                           tm.module->GetTest("expect_fail"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Bytecode> bytecodes,
                           emitter.Emit(tf->fn()));

  ASSERT_EQ(bytecodes.size(), 7);
  Bytecode bc = bytecodes[0];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLiteral);
  ASSERT_TRUE(bc.has_data());
  ASSERT_EQ(bc.value_data().value(), InterpValue::MakeU32(3));

  bc = bytecodes[1];
  ASSERT_EQ(bc.op(), Bytecode::Op::kStore);
  ASSERT_TRUE(bc.has_data());
  ASSERT_EQ(bc.integer_data().value(), 0);

  bc = bytecodes[2];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLoad);
  ASSERT_TRUE(bc.has_data());
  ASSERT_EQ(bc.integer_data().value(), 0);

  bc = bytecodes[3];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLiteral);
  ASSERT_TRUE(bc.has_data());
  ASSERT_EQ(bc.value_data().value(), InterpValue::MakeU32(2));

  bc = bytecodes[4];
  ASSERT_EQ(bc.op(), Bytecode::Op::kCall);
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue call_fn, bc.value_data());
  ASSERT_TRUE(call_fn.IsBuiltinFunction());
  // How meta!
  ASSERT_EQ(absl::get<Builtin>(call_fn.GetFunctionOrDie()), Builtin::kAssertEq);

  bc = bytecodes[5];
  ASSERT_EQ(bc.op(), Bytecode::Op::kStore);
  ASSERT_TRUE(bc.has_data());
  ASSERT_EQ(bc.integer_data().value(), 1);

  bc = bytecodes[6];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLoad);
  ASSERT_TRUE(bc.has_data());
  ASSERT_EQ(bc.integer_data().value(), 0);
}

// Validates emission of Let nodes with structured bindings.
TEST(BytecodeEmitterTest, DestructuringLet) {
  constexpr absl::string_view kProgram = R"(#![test]
fn has_name_def_tree() -> (u32, u64, uN[128]) {
  let (a, b, (c, d)) = (u4:0, u8:1, (u16:2, (u32:3, u64:4, uN[128]:5)));
  let _ = assert_eq(a, u4:0);
  let _ = assert_eq(b, u8:1);
  let _ = assert_eq(c, u16:2);
  let _ = assert_eq(d, (u32:3, u64:4, uN[128]:5));
  d
})";

  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  absl::flat_hash_map<const NameDef*, int64_t> namedef_to_slot;
  BytecodeEmitter emitter(&import_data, tm.type_info, &namedef_to_slot);

  XLS_ASSERT_OK_AND_ASSIGN(TestFunction * tf,
                           tm.module->GetTest("has_name_def_tree"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Bytecode> bytecodes,
                           emitter.Emit(tf->fn()));

  ASSERT_EQ(bytecodes.size(), 35);
  Bytecode bc = bytecodes[0];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLiteral);
  ASSERT_TRUE(bc.has_data());
  ASSERT_EQ(bc.value_data().value(), InterpValue::MakeUBits(4, 0));

  bc = bytecodes[5];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLiteral);
  ASSERT_TRUE(bc.has_data());
  ASSERT_EQ(bc.value_data().value(), InterpValue::MakeUBits(128, 5));

  bc = bytecodes[6];
  ASSERT_EQ(bc.op(), Bytecode::Op::kCreateTuple);
  ASSERT_TRUE(bc.has_data());
  ASSERT_EQ(bc.integer_data().value(), 3);

  bc = bytecodes[7];
  ASSERT_EQ(bc.op(), Bytecode::Op::kCreateTuple);
  ASSERT_TRUE(bc.has_data());
  ASSERT_EQ(bc.integer_data().value(), 2);

  bc = bytecodes[8];
  ASSERT_EQ(bc.op(), Bytecode::Op::kCreateTuple);
  ASSERT_TRUE(bc.has_data());
  ASSERT_EQ(bc.integer_data().value(), 3);

  bc = bytecodes[9];
  ASSERT_EQ(bc.op(), Bytecode::Op::kExpandTuple);
  ASSERT_FALSE(bc.has_data());

  bc = bytecodes[10];
  ASSERT_EQ(bc.op(), Bytecode::Op::kStore);
  ASSERT_TRUE(bc.has_data());
  ASSERT_EQ(bc.integer_data().value(), 0);

  bc = bytecodes[11];
  ASSERT_EQ(bc.op(), Bytecode::Op::kStore);
  ASSERT_TRUE(bc.has_data());
  ASSERT_EQ(bc.integer_data().value(), 1);

  bc = bytecodes[12];
  ASSERT_EQ(bc.op(), Bytecode::Op::kExpandTuple);
  ASSERT_FALSE(bc.has_data());

  bc = bytecodes[13];
  ASSERT_EQ(bc.op(), Bytecode::Op::kStore);
  ASSERT_TRUE(bc.has_data());
  ASSERT_EQ(bc.integer_data().value(), 2);

  bc = bytecodes[14];
  ASSERT_EQ(bc.op(), Bytecode::Op::kStore);
  ASSERT_TRUE(bc.has_data());
  ASSERT_EQ(bc.integer_data().value(), 3);

  // Skip the uninteresting comparisons.
  bc = bytecodes[27];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLoad);
  ASSERT_TRUE(bc.has_data());
  ASSERT_EQ(bc.integer_data().value(), 3);

  bc = bytecodes[31];
  ASSERT_EQ(bc.op(), Bytecode::Op::kCreateTuple);
  ASSERT_TRUE(bc.has_data());
  ASSERT_EQ(bc.integer_data().value(), 3);

  bc = bytecodes[34];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLoad);
  ASSERT_TRUE(bc.has_data());
  ASSERT_EQ(bc.integer_data().value(), 3);
}

TEST(BytecodeEmitterTest, Ternary) {
  constexpr absl::string_view kProgram = R"(fn do_ternary() -> u32 {
  if true { u32:42 } else { u32:64 }
})";

  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  absl::flat_hash_map<const NameDef*, int64_t> namedef_to_slot;
  BytecodeEmitter emitter(&import_data, tm.type_info, &namedef_to_slot);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetFunctionOrError("do_ternary"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Bytecode> bytecodes, emitter.Emit(f));

  EXPECT_EQ(BytecodesToString(bytecodes, /*source_locs=*/false),
            R"(000 literal u1:1
001 jump_rel_if +3
002 literal u32:64
003 jump_rel +3
004 jump_dest
005 literal u32:42
006 jump_dest)");
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
  EXPECT_EQ(BytecodesToString(bytecodes, /*source_locs=*/false), s);
}

// Tests emission of all of the supported binary operators.
TEST(BytecodeEmitterTest, Binops) {
  constexpr absl::string_view kProgram = R"(#![test]
fn binops_galore() {
  let a = u32:4;
  let b = u32:2;

  let add = a + b;
  let and = a & b;
  let concat = a ++ b;
  let div = a / b;
  let eq = a == b;
  let ge = a >= b;
  let gt = a > b;
  let le = a <= b;
  let lt = a < b;
  let mul = a * b;
  let ne = a != b;
  let or = a | b;
  let shl = a << b;
  let shr = a >> b;
  let sub = a - b;
  let xor = a ^ b;

  ()
})";

  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  absl::flat_hash_map<const NameDef*, int64_t> namedef_to_slot;
  BytecodeEmitter emitter(&import_data, tm.type_info, &namedef_to_slot);

  XLS_ASSERT_OK_AND_ASSIGN(TestFunction * tf,
                           tm.module->GetTest("binops_galore"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Bytecode> bytecodes,
                           emitter.Emit(tf->fn()));

  ASSERT_EQ(bytecodes.size(), 69);
  Bytecode bc = bytecodes[6];
  ASSERT_EQ(bc.op(), Bytecode::Op::kAdd);

  bc = bytecodes[10];
  ASSERT_EQ(bc.op(), Bytecode::Op::kAnd);

  bc = bytecodes[14];
  ASSERT_EQ(bc.op(), Bytecode::Op::kConcat);

  bc = bytecodes[18];
  ASSERT_EQ(bc.op(), Bytecode::Op::kDiv);

  bc = bytecodes[22];
  ASSERT_EQ(bc.op(), Bytecode::Op::kEq);

  bc = bytecodes[26];
  ASSERT_EQ(bc.op(), Bytecode::Op::kGe);

  bc = bytecodes[30];
  ASSERT_EQ(bc.op(), Bytecode::Op::kGt);

  bc = bytecodes[34];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLe);

  bc = bytecodes[38];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLt);

  bc = bytecodes[42];
  ASSERT_EQ(bc.op(), Bytecode::Op::kMul);

  bc = bytecodes[46];
  ASSERT_EQ(bc.op(), Bytecode::Op::kNe);

  bc = bytecodes[50];
  ASSERT_EQ(bc.op(), Bytecode::Op::kOr);

  bc = bytecodes[54];
  ASSERT_EQ(bc.op(), Bytecode::Op::kShll);

  bc = bytecodes[58];
  ASSERT_EQ(bc.op(), Bytecode::Op::kShrl);

  bc = bytecodes[62];
  ASSERT_EQ(bc.op(), Bytecode::Op::kSub);

  bc = bytecodes[66];
  ASSERT_EQ(bc.op(), Bytecode::Op::kXor);
}

// Tests emission of all of the supported binary operators.
TEST(BytecodeEmitterTest, Unops) {
  constexpr absl::string_view kProgram = R"(#![test]
fn unops() {
  let a = s32:32;
  let b = !a;
  let c = -b;
  ()
})";

  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  absl::flat_hash_map<const NameDef*, int64_t> namedef_to_slot;
  BytecodeEmitter emitter(&import_data, tm.type_info, &namedef_to_slot);

  XLS_ASSERT_OK_AND_ASSIGN(TestFunction * tf, tm.module->GetTest("unops"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Bytecode> bytecodes,
                           emitter.Emit(tf->fn()));

  ASSERT_EQ(bytecodes.size(), 9);
  Bytecode bc = bytecodes[3];
  ASSERT_EQ(bc.op(), Bytecode::Op::kInvert);

  bc = bytecodes[6];
  ASSERT_EQ(bc.op(), Bytecode::Op::kNegate);
}

// Tests array creation.
TEST(BytecodeEmitterTest, Arrays) {
  constexpr absl::string_view kProgram = R"(#![test]
fn arrays() -> u32[3] {
  let a = u32:32;
  u32[3]:[u32:0, u32:1, a]
}
)";

  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  absl::flat_hash_map<const NameDef*, int64_t> namedef_to_slot;
  BytecodeEmitter emitter(&import_data, tm.type_info, &namedef_to_slot);

  XLS_ASSERT_OK_AND_ASSIGN(TestFunction * tf, tm.module->GetTest("arrays"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Bytecode> bytecodes,
                           emitter.Emit(tf->fn()));

  ASSERT_EQ(bytecodes.size(), 6);
  Bytecode bc = bytecodes[5];
  ASSERT_EQ(bc.op(), Bytecode::Op::kCreateArray);
  ASSERT_TRUE(bc.has_data());
  ASSERT_EQ(bc.integer_data().value(), 3);
}

// Tests emission of kIndex ops on arrays.
TEST(BytecodeEmitterTest, IndexArray) {
  constexpr absl::string_view kProgram = R"(#![test]
fn index_array() -> u32 {
  let a = u32[3]:[0, 1, 2];
  let b = bits[32][3]:[3, 4, 5];

  a[u32:0] + b[u32:1]
}
)";

  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  absl::flat_hash_map<const NameDef*, int64_t> namedef_to_slot;
  BytecodeEmitter emitter(&import_data, tm.type_info, &namedef_to_slot);

  XLS_ASSERT_OK_AND_ASSIGN(TestFunction * tf,
                           tm.module->GetTest("index_array"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Bytecode> bytecodes,
                           emitter.Emit(tf->fn()));

  ASSERT_EQ(bytecodes.size(), 17);
  Bytecode bc = bytecodes[12];
  ASSERT_EQ(bc.op(), Bytecode::Op::kIndex);

  bc = bytecodes[15];
  ASSERT_EQ(bc.op(), Bytecode::Op::kIndex);
}

// Tests emission of kIndex ops on tuples.
TEST(BytecodeEmitterTest, IndexTuple) {
  constexpr absl::string_view kProgram = R"(#![test]
fn index_tuple() -> u32 {
  let a = (u16:0, u32:1, u64:2);
  let b = (bits[128]:3, bits[32]:4);

  a[1] + b[1]
}
)";

  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  absl::flat_hash_map<const NameDef*, int64_t> namedef_to_slot;
  BytecodeEmitter emitter(&import_data, tm.type_info, &namedef_to_slot);

  XLS_ASSERT_OK_AND_ASSIGN(TestFunction * tf,
                           tm.module->GetTest("index_tuple"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Bytecode> bytecodes,
                           emitter.Emit(tf->fn()));

  ASSERT_EQ(bytecodes.size(), 16);
  Bytecode bc = bytecodes[11];
  ASSERT_EQ(bc.op(), Bytecode::Op::kIndex);

  bc = bytecodes[14];
  ASSERT_EQ(bc.op(), Bytecode::Op::kIndex);
}

// Tests a regular a[x:y] slice op.
TEST(BytecodeEmitterTest, SimpleSlice) {
  constexpr absl::string_view kProgram = R"(#![test]
fn simple_slice() -> u16 {
  let a = u32:0xdeadbeef;
  a[16:32]
}
)";

  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  absl::flat_hash_map<const NameDef*, int64_t> namedef_to_slot;
  BytecodeEmitter emitter(&import_data, tm.type_info, &namedef_to_slot);

  XLS_ASSERT_OK_AND_ASSIGN(TestFunction * tf,
                           tm.module->GetTest("simple_slice"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Bytecode> bytecodes,
                           emitter.Emit(tf->fn()));

  ASSERT_EQ(bytecodes.size(), 6);
  Bytecode bc = bytecodes[3];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLiteral);

  bc = bytecodes[4];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLiteral);

  bc = bytecodes[5];
  ASSERT_EQ(bc.op(), Bytecode::Op::kSlice);
}

// Tests a slice from the start: a[-x:].
TEST(BytecodeEmitterTest, NegativeStartSlice) {
  constexpr absl::string_view kProgram = R"(#![test]
fn negative_start_slice() -> u16 {
  let a = u32:0xdeadbeef;
  a[-16:]
}
)";

  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  absl::flat_hash_map<const NameDef*, int64_t> namedef_to_slot;
  BytecodeEmitter emitter(&import_data, tm.type_info, &namedef_to_slot);

  XLS_ASSERT_OK_AND_ASSIGN(TestFunction * tf,
                           tm.module->GetTest("negative_start_slice"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Bytecode> bytecodes,
                           emitter.Emit(tf->fn()));

  ASSERT_EQ(bytecodes.size(), 6);
  Bytecode bc = bytecodes[3];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLiteral);

  bc = bytecodes[4];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLiteral);

  bc = bytecodes[5];
  ASSERT_EQ(bc.op(), Bytecode::Op::kSlice);
}

// Tests a slice from the end: a[:-x].
TEST(BytecodeEmitterTest, NegativeEndSlice) {
  constexpr absl::string_view kProgram = R"(#![test]
fn negative_end_slice() -> u16 {
  let a = u32:0xdeadbeef;
  a[:-16]
}
)";

  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  absl::flat_hash_map<const NameDef*, int64_t> namedef_to_slot;
  BytecodeEmitter emitter(&import_data, tm.type_info, &namedef_to_slot);

  XLS_ASSERT_OK_AND_ASSIGN(TestFunction * tf,
                           tm.module->GetTest("negative_end_slice"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Bytecode> bytecodes,
                           emitter.Emit(tf->fn()));

  ASSERT_EQ(bytecodes.size(), 6);
  Bytecode bc = bytecodes[3];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLiteral);

  bc = bytecodes[4];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLiteral);

  bc = bytecodes[5];
  ASSERT_EQ(bc.op(), Bytecode::Op::kSlice);
}

// Tests a slice from both ends: a[-x:-y].
TEST(BytecodeEmitterTest, BothNegativeSlice) {
  constexpr absl::string_view kProgram = R"(#![test]
fn both_negative_slice() -> u8 {
  let a = u32:0xdeadbeef;
  a[-16:-8]
}
)";

  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  absl::flat_hash_map<const NameDef*, int64_t> namedef_to_slot;
  BytecodeEmitter emitter(&import_data, tm.type_info, &namedef_to_slot);

  XLS_ASSERT_OK_AND_ASSIGN(TestFunction * tf,
                           tm.module->GetTest("both_negative_slice"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Bytecode> bytecodes,
                           emitter.Emit(tf->fn()));

  ASSERT_EQ(bytecodes.size(), 6);
  Bytecode bc = bytecodes[3];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLiteral);

  bc = bytecodes[4];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLiteral);

  bc = bytecodes[5];
  ASSERT_EQ(bc.op(), Bytecode::Op::kSlice);
}

// Tests the width slice op.
TEST(BytecodeEmitterTest, WidthSlice) {
  constexpr absl::string_view kProgram = R"(#![test]
fn width_slice() -> u16 {
  let a = u32:0xdeadbeef;
  a[u32:8 +: bits[16]]
})";

  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  absl::flat_hash_map<const NameDef*, int64_t> namedef_to_slot;
  BytecodeEmitter emitter(&import_data, tm.type_info, &namedef_to_slot);

  XLS_ASSERT_OK_AND_ASSIGN(TestFunction * tf,
                           tm.module->GetTest("width_slice"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Bytecode> bytecodes,
                           emitter.Emit(tf->fn()));

  ASSERT_EQ(bytecodes.size(), 6);

  Bytecode bc = bytecodes[4];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLiteral);

  bc = bytecodes[5];
  ASSERT_EQ(bc.op(), Bytecode::Op::kWidthSlice);
}

TEST(BytecodeEmitterTest, LocalEnumRef) {
  constexpr absl::string_view kProgram = R"(enum MyEnum : u23 {
  VAL_0 = 0,
  VAL_1 = 1,
  VAL_2 = 2,
}

#![test]
fn local_enum_ref() -> MyEnum {
  MyEnum::VAL_1
}
)";

  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  absl::flat_hash_map<const NameDef*, int64_t> namedef_to_slot;
  BytecodeEmitter emitter(&import_data, tm.type_info, &namedef_to_slot);

  XLS_ASSERT_OK_AND_ASSIGN(TestFunction * tf,
                           tm.module->GetTest("local_enum_ref"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Bytecode> bytecodes,
                           emitter.Emit(tf->fn()));

  ASSERT_EQ(bytecodes.size(), 1);

  Bytecode bc = bytecodes[0];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLiteral);
  ASSERT_TRUE(bc.has_data());
  EXPECT_THAT(bytecodes.at(0).value_data(),
              IsOkAndHolds(InterpValue::MakeSBits(23, 1)));
}

TEST(BytecodeEmitterTest, ImportedEnumRef) {
  constexpr absl::string_view kImportedProgram = R"(pub enum ImportedEnum : u4 {
  VAL_0 = 0,
  VAL_1 = 1,
  VAL_2 = 2,
  VAL_3 = 3,
}
)";
  constexpr absl::string_view kBaseProgram = R"(
import import_0

#![test]
fn imported_enum_ref() -> import_0::ImportedEnum {
  import_0::ImportedEnum::VAL_2
}
)";

  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckedModule tm,
                           ParseAndTypecheck(kImportedProgram, "import_0.x",
                                             "import_0", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(
      tm, ParseAndTypecheck(kBaseProgram, "test.x", "test", &import_data));

  absl::flat_hash_map<const NameDef*, int64_t> namedef_to_slot;
  BytecodeEmitter emitter(&import_data, tm.type_info, &namedef_to_slot);

  XLS_ASSERT_OK_AND_ASSIGN(TestFunction * tf,
                           tm.module->GetTest("imported_enum_ref"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Bytecode> bytecodes,
                           emitter.Emit(tf->fn()));

  ASSERT_EQ(bytecodes.size(), 1);

  Bytecode bc = bytecodes[0];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLiteral);
  ASSERT_TRUE(bc.has_data());
  EXPECT_THAT(bytecodes.at(0).value_data(),
              IsOkAndHolds(InterpValue::MakeSBits(4, 2)));
}

TEST(BytecodeEmitterTest, ImportedConstant) {
  constexpr absl::string_view kImportedProgram =
      R"(pub const MY_CONST = u3:2;)";
  constexpr absl::string_view kBaseProgram = R"(
import import_0

#![test]
fn imported_enum_ref() -> u3 {
  import_0::MY_CONST
}
)";

  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckedModule tm,
                           ParseAndTypecheck(kImportedProgram, "import_0.x",
                                             "import_0", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(
      tm, ParseAndTypecheck(kBaseProgram, "test.x", "test", &import_data));

  absl::flat_hash_map<const NameDef*, int64_t> namedef_to_slot;
  BytecodeEmitter emitter(&import_data, tm.type_info, &namedef_to_slot);

  XLS_ASSERT_OK_AND_ASSIGN(TestFunction * tf,
                           tm.module->GetTest("imported_enum_ref"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Bytecode> bytecodes,
                           emitter.Emit(tf->fn()));

  ASSERT_EQ(bytecodes.size(), 1);

  Bytecode bc = bytecodes[0];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLiteral);
  ASSERT_TRUE(bc.has_data());
  EXPECT_THAT(bytecodes.at(0).value_data(),
              IsOkAndHolds(InterpValue::MakeSBits(3, 2)));
}

TEST(BytecodeEmitterTest, HandlesConstRefs) {
  constexpr absl::string_view kProgram = R"(const kFoo = u32:100;

#![test]
fn handles_const_refs() -> u32 {
  let a = u32:200;
  a + kFoo
}
)";

  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  absl::flat_hash_map<const NameDef*, int64_t> namedef_to_slot;
  for (const ConstantDef* def : tm.module->GetConstantDefs()) {
    namedef_to_slot[def->name_def()] = namedef_to_slot.size();
  }
  BytecodeEmitter emitter(&import_data, tm.type_info, &namedef_to_slot);

  XLS_ASSERT_OK_AND_ASSIGN(TestFunction * tf,
                           tm.module->GetTest("handles_const_refs"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Bytecode> bytecodes,
                           emitter.Emit(tf->fn()));

  ASSERT_EQ(bytecodes.size(), 5);
  Bytecode bc = bytecodes[2];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLoad);

  bc = bytecodes[3];
  ASSERT_EQ(bc.op(), Bytecode::Op::kLiteral);
}

TEST(BytecodeEmitterTest, HandlesStructInstances) {
  constexpr absl::string_view kProgram = R"(struct MyStruct {
  x: u32,
  y: u64,
}

#![test]
fn handles_struct_instances() -> MyStruct {
  let x = u32:2;
  MyStruct { x: x, y: u64:3 }
})";

  auto import_data = ImportData::CreateForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  absl::flat_hash_map<const NameDef*, int64_t> namedef_to_slot;
  for (const ConstantDef* def : tm.module->GetConstantDefs()) {
    namedef_to_slot[def->name_def()] = namedef_to_slot.size();
  }
  BytecodeEmitter emitter(&import_data, tm.type_info, &namedef_to_slot);

  XLS_ASSERT_OK_AND_ASSIGN(TestFunction * tf,
                           tm.module->GetTest("handles_struct_instances"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Bytecode> bytecodes,
                           emitter.Emit(tf->fn()));

  ASSERT_EQ(bytecodes.size(), 5);
  Bytecode bc = bytecodes[4];
  ASSERT_EQ(bc.op(), Bytecode::Op::kCreateTuple);
}

}  // namespace
}  // namespace xls::dslx
