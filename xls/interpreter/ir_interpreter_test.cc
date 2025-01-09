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

#include <optional>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/interpreter/ir_evaluator_test_base.h"
#include "xls/interpreter/observer.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/ir/verifier.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::FieldsAre;
using ::testing::HasSubstr;
using ::testing::UnorderedElementsAre;

INSTANTIATE_TEST_SUITE_P(
    IrInterpreterTest, IrEvaluatorTestBase,
    testing::Values(IrEvaluatorTestParam(
        [](Function* function, absl::Span<const Value> args,
           std::optional<EvaluationObserver*> obs) {
          return InterpretFunction(function, args, obs);
        },
        [](Function* function,
           const absl::flat_hash_map<std::string, Value>& kwargs,
           std::optional<EvaluationObserver*> obs) {
          return InterpretFunctionKwargs(function, kwargs, obs);
        },
        true, "IrInterpreter")),
    testing::PrintToStringParamName());

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
      IsOkAndHolds(Value(UBits(17, 5))));
  EXPECT_THAT(
      InterpretNode(gate_node, {Value::Bool(false), Value(UBits(17, 5))}),
      IsOkAndHolds(Value(UBits(0, 5))));
}

// TODO(https://github.com/google/xls/issues/506): 2021-10-05 Move these to the
// common IR evaluator tests and make them more comprehensive once the JIT
// supports the full range of trace operations.

// Test collecting traces across static and dynamic counted for loops.
TEST_F(IrInterpreterOnlyTest, TraceCountedFor) {
  const std::string pkg_text = R"(
package trace_counted_for_test

fn accum_body(i: bits[32], accum: bits[32]) -> bits[32] {
  after_all.0: token = after_all()
  literal.1: bits[1] = literal(value=1)
  trace.2: token = trace(after_all.0, literal.1, format = "accum is {}", data_operands=[accum])
  ret add.3: bits[32] = add(accum, i)
}

fn accum_fixed() -> bits[32] {
  literal.4: bits[32] = literal(value=0)
  ret counted_for.5: bits[32] = counted_for(literal.4, trip_count=5, stride=1, body=accum_body)
}

fn accum_dynamic(trips: bits[8]) -> bits[32] {
    literal.6: bits[32] = literal(value=0)
    literal.7: bits[32] = literal(value=1)
    ret dynamic_counted_for.8: bits[32] = dynamic_counted_for(literal.6, trips, literal.7, body=accum_body)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, ParsePackage(pkg_text));

  Function* accum_fixed = FindFunction("accum_fixed", package.get());
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> accum_fixed_result,
                           InterpretFunction(accum_fixed, {}));

  Value accum_5_value = Value(UBits(10, 32));
  auto accum_5_traces = {FieldsAre("accum is 0", 0), FieldsAre("accum is 0", 0),
                         FieldsAre("accum is 1", 0), FieldsAre("accum is 3", 0),
                         FieldsAre("accum is 6", 0)};
  EXPECT_EQ(accum_fixed_result.value, accum_5_value);
  EXPECT_THAT(accum_fixed_result.events.trace_msgs,
              ElementsAreArray(accum_5_traces));

  Function* accum_dynamic = FindFunction("accum_dynamic", package.get());

  XLS_ASSERT_OK_AND_ASSIGN(
      InterpreterResult<Value> accum_dynamic_5,
      InterpretFunction(accum_dynamic, {Value(UBits(5, 8))}));
  EXPECT_EQ(accum_dynamic_5.value, accum_5_value);
  EXPECT_THAT(accum_dynamic_5.events.trace_msgs,
              ElementsAreArray(accum_5_traces));

  XLS_ASSERT_OK_AND_ASSIGN(
      InterpreterResult<Value> accum_dynamic_0,
      InterpretFunction(accum_dynamic, {Value(UBits(0, 8))}));
  EXPECT_EQ(accum_dynamic_0.value, Value(UBits(0, 32)));
  EXPECT_THAT(accum_dynamic_0.events.trace_msgs, ElementsAre());

  XLS_ASSERT_OK_AND_ASSIGN(
      InterpreterResult<Value> accum_dynamic_1,
      InterpretFunction(accum_dynamic, {Value(UBits(1, 8))}));
  EXPECT_EQ(accum_dynamic_1.value, Value(UBits(0, 32)));
  EXPECT_THAT(accum_dynamic_1.events.trace_msgs,
              ElementsAre(FieldsAre("accum is 0", 0)));

  XLS_ASSERT_OK_AND_ASSIGN(
      InterpreterResult<Value> accum_dynamic_7,
      InterpretFunction(accum_dynamic, {Value(UBits(7, 8))}));
  EXPECT_EQ(accum_dynamic_7.value, Value(UBits(21, 32)));
  EXPECT_THAT(
      accum_dynamic_7.events.trace_msgs,
      ElementsAre(FieldsAre("accum is 0", 0), FieldsAre("accum is 0", 0),
                  FieldsAre("accum is 1", 0), FieldsAre("accum is 3", 0),
                  FieldsAre("accum is 6", 0), FieldsAre("accum is 10", 0),
                  FieldsAre("accum is 15", 0)));
}

// Test collecting traces across a map.
TEST_F(IrInterpreterOnlyTest, TraceMap) {
  const std::string pkg_text = R"(
package trace_map_test

fn square_with_trace_odd(x: bits[32]) -> bits[32] {
  after_all.0: token = after_all()
  bit_slice.1: bits[1] = bit_slice(x, start=0, width=1)
  trace.2: token = trace(after_all.0, bit_slice.1, format = "{:x} is odd", data_operands=[x])
  ret umul.3: bits[32] = umul(x, x)
}

fn map_trace() -> bits[32][5]{
  literal.11: bits[32] = literal(value=11)
  literal.12: bits[32] = literal(value=12)
  literal.13: bits[32] = literal(value=13)
  literal.14: bits[32] = literal(value=14)
  literal.15: bits[32] = literal(value=15)
  array.4: bits[32][5] = array(literal.11, literal.12, literal.13, literal.14, literal.15)
  ret map.5: bits[32][5] = map(array.4, to_apply=square_with_trace_odd)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, ParsePackage(pkg_text));
  Function* map_trace = FindFunction("map_trace", package.get());

  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> map_trace_result,
                           InterpretFunction(map_trace, {}));

  XLS_ASSERT_OK_AND_ASSIGN(Value map_trace_expected,
                           Value::UBitsArray({121, 144, 169, 196, 225}, 32));
  EXPECT_EQ(map_trace_result.value, map_trace_expected);
  EXPECT_THAT(
      map_trace_result.events.trace_msgs,
      UnorderedElementsAre(FieldsAre("f is odd", 0), FieldsAre("d is odd", 0),
                           FieldsAre("b is odd", 0)));
}

}  // namespace
}  // namespace xls
