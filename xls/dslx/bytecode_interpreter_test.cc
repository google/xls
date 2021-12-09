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
#include "xls/dslx/bytecode_emitter.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {
namespace {

using status_testing::StatusIs;
using testing::HasSubstr;

// Interprets a nearly-minimal bytecode program; the same from
// BytecodeEmitterTest.SimpleTranslation.
TEST(BytecodeInterpreterTest, PositiveSmokeTest) {
  std::vector<Bytecode> bytecodes;

  Bytecode bc(Span::Fake(), Bytecode::Op::kLiteral, InterpValue::MakeU32(1));
  bytecodes.push_back(bc);

  bc = Bytecode(Span::Fake(), Bytecode::Op::kStore, 0);
  bytecodes.push_back(bc);

  bc = Bytecode(Span::Fake(), Bytecode::Op::kLoad, 0);
  bytecodes.push_back(bc);

  bc = Bytecode(Span::Fake(), Bytecode::Op::kLiteral, InterpValue::MakeU32(2));
  bytecodes.push_back(bc);

  bc = Bytecode(Span::Fake(), Bytecode::Op::kAdd);
  bytecodes.push_back(bc);

  std::vector<InterpValue> env;
  env.push_back(InterpValue::MakeUnit());
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           BytecodeInterpreter::Interpret(bytecodes, &env));
  EXPECT_EQ(value, InterpValue::MakeU32(3));
}

// Tests that a failing assert_eq is interpreted correctly. Again, a
// continuation of a test from BytecodeEmitterTest. Get used to it.
TEST(BytecodeInterpreterTest, AssertEqFail) {
  std::vector<Bytecode> bytecodes;
  Bytecode bc(Span::Fake(), Bytecode::Op::kLiteral, InterpValue::MakeU32(3));
  bytecodes.push_back(bc);

  bc = Bytecode(Span::Fake(), Bytecode::Op::kStore, 0);
  bytecodes.push_back(bc);

  bc = Bytecode(Span::Fake(), Bytecode::Op::kLoad, 0);
  bytecodes.push_back(bc);

  bc = Bytecode(Span::Fake(), Bytecode::Op::kLiteral, InterpValue::MakeU32(2));
  bytecodes.push_back(bc);

  InterpValue fn_value(InterpValue::MakeFunction(Builtin::kAssertEq));
  bc = Bytecode(Span::Fake(), Bytecode::Op::kCall, fn_value);
  bytecodes.push_back(bc);

  bc = Bytecode(Span::Fake(), Bytecode::Op::kStore, 1);
  bytecodes.push_back(bc);

  bc = Bytecode(Span::Fake(), Bytecode::Op::kLoad, 0);
  bytecodes.push_back(bc);

  std::vector<InterpValue> env;
  env.push_back(InterpValue::MakeUnit());
  env.push_back(InterpValue::MakeUnit());
  absl::StatusOr<InterpValue> value =
      BytecodeInterpreter::Interpret(bytecodes, &env);
  EXPECT_THAT(value.status(), StatusIs(absl::StatusCode::kInternal,
                                       HasSubstr("were not equal")));
}

}  // namespace
}  // namespace xls::dslx
