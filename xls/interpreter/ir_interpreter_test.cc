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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/interpreter/ir_evaluator_test_base.h"
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
using testing::HasSubstr;

INSTANTIATE_TEST_SUITE_P(
    IrInterpreterTest, IrEvaluatorTestBase,
    testing::Values(IrEvaluatorTestParam(
        [](Function* function, absl::Span<const Value> args) {
          return InterpretFunction(function, args);
        },
        [](Function* function,
           const absl::flat_hash_map<std::string, Value>& kwargs) {
          return InterpretFunctionKwargs(function, kwargs);
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

  EXPECT_THAT(InterpretNode(FindNode("and.3", function),
                            {Value(UBits(0b0011, 4)), Value(UBits(0b1010, 4))}),
              IsOkAndHolds(Value(UBits(0b0010, 4))));
  EXPECT_THAT(InterpretNode(FindNode("or.4", function),
                            {Value(UBits(0b0011, 4)), Value(UBits(0b1010, 4))}),
              IsOkAndHolds(Value(UBits(0b1011, 4))));
  EXPECT_THAT(InterpretNode(FindNode("literal.1", function), {}),
              IsOkAndHolds(Value(UBits(6, 4))));
}

TEST_F(IrInterpreterOnlyTest, SideEffectingNodes) {
  Package package("my_package");
  const std::string fn_text = R"(
    fn bar(tkn: token, cond: bits[1], x: bits[5]) -> bits[5] {
      trace.1: token = trace(tkn, cond, format="x is {}", data_operands=[x], id=1)
      umul.2 : bits[5] = umul(x, x, id=2)
      ret gate.3: bits[5] = gate(cond, umul.2, id=3)
    }
    )";

  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(fn_text, &package));

  EXPECT_THAT(
      InterpretNode(FindNode("trace.1", function),
                    {Value::Token(), Value::Bool(true), Value(UBits(17, 5))}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Cannot interpret side-effecting op")));

  Node* gate_node = FindNode("gate.3", function);
  EXPECT_THAT(
      InterpretNode(gate_node, {Value::Bool(true), Value(UBits(17, 5))}),
      IsOkAndHolds(Value(UBits(0, 5))));
  EXPECT_THAT(
      InterpretNode(gate_node, {Value::Bool(false), Value(UBits(17, 5))}),
      IsOkAndHolds(Value(UBits(17, 5))));
}

// This is a smoke test of the trace plumbing.
// TODO(amfv): 2021-10-05 Move these to the common IR evaluator tests and make
// them more comprehensive once the JIT supports generating events.
TEST_F(IrInterpreterOnlyTest, Traces) {
  const std::string pkg_text = R"(
package trace_test

fn a (tkn:token, cond: bits[1]) -> (token, bits[8]) {
  trace.1 : token = trace(tkn, cond, format = "trace a", data_operands=[])
  literal.2: bits[8] = literal(value=17)
  ret tuple.3: (token, bits[8]) = tuple(trace.1, literal.2)
}

fn b (tkn:token, cond: bits[1]) -> (token, bits[8]) {
  trace.4: token = trace(tkn, cond, format = "trace b", data_operands=[])
  literal.5: bits[8] = literal(value=23)
  ret tuple.6: (token, bits[8]) = tuple(trace.4, literal.5)
}

fn ab (in_token:token, a_cond: bits[1], b_cond: bits[1]) -> bits[8] {
  a_result: (token, bits[8]) = invoke(in_token, a_cond, to_apply=a)
  a_token: token = tuple_index(a_result, index=0)
  a_value: bits[8] = tuple_index(a_result, index=1)
  b_result: (token, bits[8]) = invoke(a_token, b_cond, to_apply=b)
  b_value: bits[8] = tuple_index(b_result, index=1)
  ret sum: bits[8] = add(a_value, b_value)
}

fn ba (in_token:token, a_cond: bits[1], b_cond: bits[1]) -> bits[8] {
  b_result: (token, bits[8]) = invoke(in_token, b_cond, to_apply=b)
  b_token: token = tuple_index(b_result, index=0)
  b_value: bits[8] = tuple_index(b_result, index=1)
  a_result: (token, bits[8]) = invoke(b_token, a_cond, to_apply=a)
  a_value: bits[8] = tuple_index(a_result, index=1)
  ret sum: bits[8] = add(a_value, b_value)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, ParsePackage(pkg_text));

  Function* a = FindFunction("a", package.get());
  Function* b = FindFunction("b", package.get());
  Function* ab = FindFunction("ab", package.get());
  Function* ba = FindFunction("ba", package.get());

  auto run_for_traces = [](Function* f, absl::Span<const Value> args)
      -> absl::StatusOr<std::vector<std::string>> {
    XLS_ASSIGN_OR_RETURN(auto result, InterpretFunctionWithEvents(f, args));
    return result.events.trace_msgs;
  };

  std::vector<std::string> no_traces = {};
  std::vector<std::string> trace_a = {"trace a"};
  std::vector<std::string> trace_b = {"trace b"};
  std::vector<std::string> trace_ab = {"trace a", "trace b"};
  std::vector<std::string> trace_ba = {"trace b", "trace a"};

  EXPECT_THAT(run_for_traces(a, {Value::Token(), Value(UBits(1, 1))}),
              IsOkAndHolds(trace_a));

  EXPECT_THAT(run_for_traces(b, {Value::Token(), Value(UBits(0, 1))}),
              IsOkAndHolds(no_traces));

  EXPECT_THAT(run_for_traces(
                  ab, {Value::Token(), Value(UBits(1, 1)), Value(UBits(1, 1))}),
              IsOkAndHolds(trace_ab));

  EXPECT_THAT(run_for_traces(
                  ab, {Value::Token(), Value(UBits(0, 1)), Value(UBits(1, 1))}),
              IsOkAndHolds(trace_b));

  EXPECT_THAT(run_for_traces(
                  ba, {Value::Token(), Value(UBits(1, 1)), Value(UBits(1, 1))}),
              IsOkAndHolds(trace_ba));

  EXPECT_THAT(run_for_traces(
                  ba, {Value::Token(), Value(UBits(0, 1)), Value(UBits(0, 1))}),
              IsOkAndHolds(no_traces));
}

// A variant of the TwoAssert test that checks if both assertions were recorded.
TEST_F(IrInterpreterOnlyTest, BothAsserts) {
  Package p("assert_test");
  FunctionBuilder b("fun", &p);
  auto p0 = b.Param("tkn", p.GetTokenType());
  auto p1 = b.Param("cond1", p.GetBitsType(1));
  auto p2 = b.Param("cond2", p.GetBitsType(1));

  BValue token1 = b.Assert(p0, p1, "first assertion error message");
  b.Assert(token1, p2, "second assertion error message");

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());

  std::vector<Value> failboth_args = {Value::Token(), Value(UBits(0, 1)),
                                      Value(UBits(0, 1))};

  auto run_for_asserts = [](Function* f, absl::Span<const Value> args)
      -> absl::StatusOr<std::vector<std::string>> {
    XLS_ASSIGN_OR_RETURN(auto result, InterpretFunctionWithEvents(f, args));
    return result.events.assert_msgs;
  };

  std::vector<std::string> both_asserts = {"first assertion error message",
                                           "second assertion error message"};
  EXPECT_THAT(run_for_asserts(f, failboth_args), IsOkAndHolds(both_asserts));
}

}  // namespace
}  // namespace xls
