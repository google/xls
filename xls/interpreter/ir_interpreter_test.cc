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
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/verifier.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

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

// TODO(meheff): Move these tests to ir_evaluator_test when JIT support for
// multiarray index/update operations is supported.
TEST_F(IrInterpreterOnlyTest, MultiArrayIndex) {
  auto package = CreatePackage();
  std::string input = R"(
  fn main(a: bits[13][3], idx: (bits[3])) -> bits[13] {
    ret result: bits[13] = multiarray_index(a, idx)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseFunction(input, package.get()));

  EXPECT_THAT(
      IrInterpreter::Run(
          function, {Value::UBitsArray({11, 22, 33}, /*bit_count=*/13).value(),
                     Value::Tuple({Value(UBits(1, 3))})}),
      IsOkAndHolds(Value(UBits(22, 13))));
  EXPECT_THAT(
      IrInterpreter::Run(
          function, {Value::UBitsArray({11, 22, 33}, /*bit_count=*/13).value(),
                     Value::Tuple({Value(UBits(2, 3))})}),
      IsOkAndHolds(Value(UBits(33, 13))));
  // Out of bounds access should return last element.
  EXPECT_THAT(
      IrInterpreter::Run(
          function, {Value::UBitsArray({11, 22, 33}, /*bit_count=*/13).value(),
                     Value::Tuple({Value(UBits(6, 3))})}),
      IsOkAndHolds(Value(UBits(33, 13))));
}

TEST_F(IrInterpreterOnlyTest, MultiArrayIndex2DArray) {
  auto package = CreatePackage();
  std::string input = R"(
  fn main(a: bits[13][3][2], idx: (bits[3], bits[7])) -> bits[13] {
    ret result: bits[13] = multiarray_index(a, idx)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseFunction(input, package.get()));

  EXPECT_THAT(
      IrInterpreter::Run(
          function,
          {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
               .value(),
           Value::Tuple({Value(UBits(1, 3)), Value(UBits(2, 7))})}),
      IsOkAndHolds(Value(UBits(33, 13))));
  EXPECT_THAT(
      IrInterpreter::Run(
          function,
          {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
               .value(),
           Value::Tuple({Value(UBits(0, 3)), Value(UBits(1, 7))})}),
      IsOkAndHolds(Value(UBits(55, 13))));

  // Out of bounds access should return last element of respective array.
  EXPECT_THAT(
      IrInterpreter::Run(
          function,
          {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
               .value(),
           Value::Tuple({Value(UBits(6, 3)), Value(UBits(1, 7))})}),
      IsOkAndHolds(Value(UBits(22, 13))));
  EXPECT_THAT(
      IrInterpreter::Run(
          function,
          {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
               .value(),
           Value::Tuple({Value(UBits(0, 3)), Value(UBits(100, 7))})}),
      IsOkAndHolds(Value(UBits(66, 13))));
}

TEST_F(IrInterpreterOnlyTest, MultiArrayIndex2DArrayReturnArray) {
  auto package = CreatePackage();
  std::string input = R"(
  fn main(a: bits[13][3][2], idx: (bits[3])) -> bits[13][3] {
    ret result: bits[13][3] = multiarray_index(a, idx)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseFunction(input, package.get()));

  EXPECT_THAT(
      IrInterpreter::Run(
          function,
          {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
               .value(),
           Value::Tuple({Value(UBits(1, 3))})}),
      IsOkAndHolds(Value::UBitsArray({11, 22, 33}, /*bit_count=*/13).value()));
  EXPECT_THAT(
      IrInterpreter::Run(
          function,
          {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
               .value(),
           Value::Tuple({Value(UBits(0, 3))})}),
      IsOkAndHolds(Value::UBitsArray({44, 55, 66}, /*bit_count=*/13).value()));

  // Out of bounds access should return last element of respective array.
  EXPECT_THAT(
      IrInterpreter::Run(
          function,
          {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
               .value(),
           Value::Tuple({Value(UBits(7, 3))})}),
      IsOkAndHolds(Value::UBitsArray({11, 22, 33}, /*bit_count=*/13).value()));
}

TEST_F(IrInterpreterOnlyTest, MultiArrayIndex2DArrayNilIndex) {
  auto package = CreatePackage();
  std::string input = R"(
  fn main(a: bits[13][3][2], idx: ()) -> bits[13][3][2] {
    ret result: bits[13][3][2] = multiarray_index(a, idx)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseFunction(input, package.get()));

  XLS_ASSERT_OK_AND_ASSIGN(
      Value array,
      Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13));
  EXPECT_THAT(IrInterpreter::Run(function, {array, Value::Tuple({})}),
              IsOkAndHolds(array));
}

TEST_F(IrInterpreterOnlyTest, MultiArrayUpdate) {
  auto package = CreatePackage();
  std::string input = R"(
  fn main(a: bits[13][3], idx: (bits[7]), value: bits[13]) -> bits[13][3] {
    ret result: bits[13][3] = multiarray_update(a, idx, value)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseFunction(input, package.get()));

  EXPECT_THAT(
      IrInterpreter::Run(
          function,
          {Value::UBitsArray({11, 22, 33}, /*bit_count=*/13).value(),
           Value::Tuple({Value(UBits(1, 7))}), Value(UBits(123, 13))}),
      IsOkAndHolds(Value::UBitsArray({11, 123, 33}, /*bit_count=*/13).value()));
  EXPECT_THAT(
      IrInterpreter::Run(
          function,
          {Value::UBitsArray({11, 22, 33}, /*bit_count=*/13).value(),
           Value::Tuple({Value(UBits(2, 7))}), Value(UBits(123, 13))}),
      IsOkAndHolds(Value::UBitsArray({11, 22, 123}, /*bit_count=*/13).value()));
  // Out of bounds access should update no element.
  EXPECT_THAT(
      IrInterpreter::Run(
          function,
          {Value::UBitsArray({11, 22, 33}, /*bit_count=*/13).value(),
           Value::Tuple({Value(UBits(3, 7))}), Value(UBits(123, 13))}),
      IsOkAndHolds(Value::UBitsArray({11, 22, 33}, /*bit_count=*/13).value()));
  EXPECT_THAT(
      IrInterpreter::Run(
          function,
          {Value::UBitsArray({11, 22, 33}, /*bit_count=*/13).value(),
           Value::Tuple({Value(UBits(55, 7))}), Value(UBits(123, 13))}),
      IsOkAndHolds(Value::UBitsArray({11, 22, 33}, /*bit_count=*/13).value()));
}

TEST_F(IrInterpreterOnlyTest, MultiArrayUpdate2DArray) {
  auto package = CreatePackage();
  std::string input = R"(
  fn main(a: bits[13][3][2], idx: (bits[3], bits[7]), value: bits[13]) -> bits[13][3][2] {
    ret result: bits[13][3][2] = multiarray_update(a, idx, value)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseFunction(input, package.get()));

  EXPECT_THAT(
      IrInterpreter::Run(
          function,
          {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
               .value(),
           Value::Tuple({Value(UBits(1, 3)), Value(UBits(2, 7))}),
           Value(UBits(999, 13))}),
      IsOkAndHolds(
          Value::UBits2DArray({{44, 55, 66}, {11, 22, 999}}, /*bit_count=*/13)
              .value()));
  EXPECT_THAT(
      IrInterpreter::Run(
          function,
          {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
               .value(),
           Value::Tuple({Value(UBits(0, 3)), Value(UBits(1, 7))}),
           Value(UBits(999, 13))}),
      IsOkAndHolds(
          Value::UBits2DArray({{44, 999, 66}, {11, 22, 33}}, /*bit_count=*/13)
              .value()));
  // Out of bounds on either index should update no element.
  EXPECT_THAT(
      IrInterpreter::Run(
          function,
          {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
               .value(),
           Value::Tuple({Value(UBits(4, 3)), Value(UBits(1, 7))}),
           Value(UBits(999, 13))}),
      IsOkAndHolds(
          Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
              .value()));
  EXPECT_THAT(
      IrInterpreter::Run(
          function,
          {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
               .value(),
           Value::Tuple({Value(UBits(1, 3)), Value(UBits(3, 7))}),
           Value(UBits(999, 13))}),
      IsOkAndHolds(
          Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
              .value()));
}

TEST_F(IrInterpreterOnlyTest, MultiArrayUpdate2DArrayWithArray) {
  auto package = CreatePackage();
  std::string input = R"(
  fn main(a: bits[13][3][2], idx: (bits[3]), value: bits[13][3]) -> bits[13][3][2] {
    ret result: bits[13][3][2] = multiarray_update(a, idx, value)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseFunction(input, package.get()));

  EXPECT_THAT(
      IrInterpreter::Run(
          function,
          {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
               .value(),
           Value::Tuple({Value(UBits(1, 3))}),
           Value::UBitsArray({999, 888, 777}, /*bit_count=*/13).value()}),
      IsOkAndHolds(
          Value::UBits2DArray({{44, 55, 66}, {999, 888, 777}}, /*bit_count=*/13)
              .value()));

  // Out of bounds should update no element.
  EXPECT_THAT(
      IrInterpreter::Run(
          function,
          {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
               .value(),
           Value::Tuple({Value(UBits(2, 3))}),
           Value::UBitsArray({999, 888, 777}, /*bit_count=*/13).value()}),
      IsOkAndHolds(
          Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
              .value()));
}

TEST_F(IrInterpreterOnlyTest, MultiArrayUpdate2DArrayNilIndex) {
  auto package = CreatePackage();
  std::string input = R"(
  fn main(a: bits[13][3][2], idx: (), value: bits[13][3][2]) -> bits[13][3][2] {
    ret result: bits[13][3][2] = multiarray_update(a, idx, value)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseFunction(input, package.get()));

  EXPECT_THAT(
      IrInterpreter::Run(
          function,
          {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
               .value(),
           Value::Tuple({}),
           Value::UBits2DArray({{99, 123, 4}, {432, 7, 42}}, /*bit_count=*/13)
               .value()}),
      IsOkAndHolds(
          Value::UBits2DArray({{99, 123, 4}, {432, 7, 42}}, /*bit_count=*/13)
              .value()));
}

TEST_F(IrInterpreterOnlyTest, MultiArrayUpdateBitsValueNilIndex) {
  auto package = CreatePackage();
  std::string input = R"(
  fn main(a: bits[1234], idx: (), value: bits[1234]) -> bits[1234] {
    ret result: bits[1234] = multiarray_update(a, idx, value)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseFunction(input, package.get()));

  EXPECT_THAT(
      IrInterpreter::Run(function, {Value(UBits(42, 1234)), Value::Tuple({}),
                                    Value(UBits(888, 1234))}),
      IsOkAndHolds(Value(UBits(888, 1234))));
}

}  // namespace
}  // namespace xls
