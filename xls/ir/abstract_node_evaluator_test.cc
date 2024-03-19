// Copyright 2024 The XLS Authors
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

#include "xls/ir/abstract_node_evaluator.h"

#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/abstract_evaluator.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"

namespace xls {
namespace {

// Simple wrapper to avoid std::vector<bool> specialization.
struct BoxedBool {
  bool value;
  bool operator!=(const BoxedBool& other) const { return value != other.value; }
  bool operator==(const BoxedBool& other) const { return value == other.value; }
};

Bits FromBoxedVector(absl::Span<BoxedBool const> input) {
  BitsRope rope(input.size());
  for (BoxedBool bit : input) {
    rope.push_back(bit.value);
  }
  return rope.Build();
}

class TestAbstractEvaluator
    : public AbstractEvaluator<BoxedBool, TestAbstractEvaluator> {
 public:
  BoxedBool One() const { return {true}; }
  BoxedBool Zero() const { return {false}; }
  BoxedBool Not(const BoxedBool& input) const { return {!input.value}; }
  BoxedBool And(const BoxedBool& a, const BoxedBool& b) const {
    return {static_cast<bool>(static_cast<int>(a.value) &
                              static_cast<int>(b.value))};
  }
  BoxedBool Or(const BoxedBool& a, const BoxedBool& b) const {
    return {static_cast<bool>(static_cast<int>(a.value) |
                              static_cast<int>(b.value))};
  }
};

using BoxedBoolNodeEvaluator = AbstractNodeEvaluator<TestAbstractEvaluator>;

absl::StatusOr<Bits> AbstractEvaluate(BValue v) {
  TestAbstractEvaluator tea;
  BoxedBoolNodeEvaluator eval(tea);
  XLS_RETURN_IF_ERROR(v.node()->function_base()->Accept(&eval));
  XLS_ASSIGN_OR_RETURN(auto out, eval.GetValue(v.node()));
  return FromBoxedVector(out);
}

class AbstractNodeEvaluatorTest : public IrTestBase {};

TEST_F(AbstractNodeEvaluatorTest, TestUMul) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue res = fb.UMul(fb.Literal(UBits(2, 4)), fb.Literal(UBits(4, 4)),
                       /*result_width=*/32);
  XLS_ASSERT_OK(fb.Build().status());
  EXPECT_THAT(AbstractEvaluate(res),
              status_testing::IsOkAndHolds(UBits(8, 32)));
}

TEST_F(AbstractNodeEvaluatorTest, TestSMul) {
  // -1 * 4
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto res = fb.SMul(fb.Literal(UBits(0b1111, 4)), fb.Literal(UBits(4, 4)),
                     /*result_width=*/16);
  XLS_ASSERT_OK(fb.Build().status());
  EXPECT_THAT(AbstractEvaluate(res),
              status_testing::IsOkAndHolds(UBits(0b1111111111111100, 16)));
}

TEST_F(AbstractNodeEvaluatorTest, TestAdd) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto res = fb.Add(fb.Literal(UBits(2, 4)), fb.Literal(UBits(4, 4)));
  XLS_ASSERT_OK(fb.Build().status());
  EXPECT_THAT(AbstractEvaluate(res), status_testing::IsOkAndHolds(UBits(6, 4)));
}

TEST_F(AbstractNodeEvaluatorTest, TestUSub) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto res = fb.Subtract(fb.Literal(UBits(4, 4)), fb.Literal(UBits(2, 4)));
  XLS_ASSERT_OK(fb.Build().status());
  EXPECT_THAT(AbstractEvaluate(res), status_testing::IsOkAndHolds(UBits(2, 4)));
}

TEST_F(AbstractNodeEvaluatorTest, TestUDiv) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto res = fb.UDiv(fb.Literal(UBits(9, 32)), fb.Literal(UBits(3, 32)));
  XLS_ASSERT_OK(fb.Build().status());
  EXPECT_THAT(AbstractEvaluate(res),
              status_testing::IsOkAndHolds(UBits(3, 32)));
}

TEST_F(AbstractNodeEvaluatorTest, TestSDiv) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto res = fb.SDiv(fb.Literal(UBits(-9, 64)), fb.Literal(UBits(3, 64)));
  XLS_ASSERT_OK(fb.Build().status());
  EXPECT_THAT(AbstractEvaluate(res),
              status_testing::IsOkAndHolds(UBits(-3, 64)));
}

TEST_F(AbstractNodeEvaluatorTest, TestShll) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto res = fb.Shll(fb.Literal(UBits(0b1010, 4)), fb.Literal(UBits(2, 4)));
  XLS_ASSERT_OK(fb.Build().status());
  EXPECT_THAT(AbstractEvaluate(res),
              status_testing::IsOkAndHolds(UBits(0b1000, 4)));
}

TEST_F(AbstractNodeEvaluatorTest, TestShrl) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto res = fb.Shrl(fb.Literal(UBits(0b1010, 4)), fb.Literal(UBits(2, 4)));
  XLS_ASSERT_OK(fb.Build().status());
  EXPECT_THAT(AbstractEvaluate(res),
              status_testing::IsOkAndHolds(UBits(0b0010, 4)));
}

TEST_F(AbstractNodeEvaluatorTest, TestShra) {
  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    auto res = fb.Shra(fb.Literal(UBits(0b1010, 4)), fb.Literal(UBits(2, 4)));
    XLS_ASSERT_OK(fb.Build().status());
    EXPECT_THAT(AbstractEvaluate(res),
                status_testing::IsOkAndHolds(UBits(0b1110, 4)));
  }
  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    auto res = fb.Shra(fb.Literal(UBits(0b0101, 4)), fb.Literal(UBits(2, 4)));
    XLS_ASSERT_OK(fb.Build().status());
    EXPECT_THAT(AbstractEvaluate(res),
                status_testing::IsOkAndHolds(UBits(0b0001, 4)));
  }
}

TEST_F(AbstractNodeEvaluatorTest, TestUMod) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto res = fb.UMod(fb.Literal(UBits(9, 32)), fb.Literal(UBits(4, 32)));
  XLS_ASSERT_OK(fb.Build().status());
  EXPECT_THAT(AbstractEvaluate(res),
              status_testing::IsOkAndHolds(UBits(1, 32)));
}

}  // namespace
}  // namespace xls
