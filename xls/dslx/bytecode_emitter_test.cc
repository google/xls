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

}  // namespace
}  // namespace xls::dslx
