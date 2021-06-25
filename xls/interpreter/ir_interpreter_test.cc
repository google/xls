// Copyright 2020 The XLS Authors
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

#include "xls/interpreter/ir_interpreter.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/ir_evaluator_test.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/verifier.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::HasSubstr;

INSTANTIATE_TEST_SUITE_P(
    IrInterpreterTest, IrEvaluatorTest,
    testing::Values(IrEvaluatorTestParam(
        [](Function* function, const std::vector<Value>& args) {
          return IrInterpreter::Run(function, args);
        },
        [](Function* function,
           const absl::flat_hash_map<std::string, Value>& kwargs) {
          return IrInterpreter::RunKwargs(function, kwargs);
        })));

// Fixture for IrInterpreter-only tests (i.e., those that aren't common to all
// IR evaluators).
class IrInterpreterOnlyTest : public IrTestBase {};

TEST_F(IrInterpreterOnlyTest, EvaluateNode) {
  Package package("my_package");
  std::string fn_text = R"(
    fn f(x: bits[4]) -> bits[4] {
      literal.1: bits[4] = literal(value=6)
      literal.2: bits[4] = literal(value=3)
      and.3: bits[4] = and(literal.1, x)
      ret or.4: bits[4] = or(literal.2, and.3)
    }
    )";

  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(fn_text, &package));

  Value a = Value(UBits(0b0011, 4));
  Value b = Value(UBits(0b1010, 4));
  EXPECT_THAT(
      IrInterpreter::EvaluateNode(FindNode("and.3", function), {&a, &b}),
      IsOkAndHolds(Value(UBits(0b0010, 4))));
  EXPECT_THAT(IrInterpreter::EvaluateNode(FindNode("or.4", function), {&a, &b}),
              IsOkAndHolds(Value(UBits(0b1011, 4))));
  EXPECT_THAT(IrInterpreter::EvaluateNode(FindNode("literal.1", function), {}),
              IsOkAndHolds(Value(UBits(6, 4))));
}

TEST_F(IrInterpreterOnlyTest, AssertTest) {
  auto p = CreatePackage();
  FunctionBuilder b(TestName(), p.get());
  auto p0 = b.Param("tkn", p->GetTokenType());
  auto p1 = b.Param("cond", p->GetBitsType(1));
  b.Assert(p0, p1, "the assertion error message", {});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());

  EXPECT_THAT(IrInterpreter::Run(f, {Value::Token(), Value(UBits(1, 1))}),
              IsOkAndHolds(Value::Token()));
  EXPECT_THAT(IrInterpreter::Run(f, {Value::Token(), Value(UBits(0, 1))}),
              StatusIs(absl::StatusCode::kAborted,
                       HasSubstr("the assertion error message")));
}

TEST_F(IrInterpreterOnlyTest, AssertTestWithData) {
  auto p = CreatePackage();
  FunctionBuilder b(TestName(), p.get());
  auto p0 = b.Param("tkn", p->GetTokenType());
  auto p1 = b.Param("cond", p->GetBitsType(1));
  auto p2 = b.Param("value", p->GetBitsType(8));

  b.Assert(p0, p1, "default: {}, hex: {:x}, bits: {:b}, decimal: {:d}",
           {p2, p2, p2, p2});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());

  EXPECT_THAT(
      IrInterpreter::Run(
          f, {Value::Token(), Value(UBits(0, 1)), Value(UBits(67, 8))}),
      StatusIs(absl::StatusCode::kAborted,
               HasSubstr("default: bits[8]:67, hex: bits[8]:0x43, bits: "
                         "bits[8]:0b100_0011, decimal: bits[8]:67")));
}

TEST_F(IrInterpreterOnlyTest, AssertTestNotEnoughOperands) {
  auto p = CreatePackage();
  FunctionBuilder b(TestName(), p.get());
  auto p0 = b.Param("tkn", p->GetTokenType());
  auto p1 = b.Param("cond", p->GetBitsType(1));

  b.Assert(p0, p1, "{}", {});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());

  EXPECT_THAT(IrInterpreter::Run(f, {Value::Token(), Value(UBits(0, 1))}),
              StatusIs(absl::StatusCode::kAborted, HasSubstr("{}")));
}

}  // namespace
}  // namespace xls
