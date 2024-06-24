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

#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/abstract_evaluator.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/value.h"
#include "xls/ir/value_builder.h"

namespace xls {
namespace {

using VB = ValueBuilder;

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

VB BuilderFromBoxedValue(LeafTypeTreeView<std::vector<BoxedBool>> input) {
  CHECK(!input.type()->IsToken());
  if (input.type()->IsBits()) {
    return VB::Bits(FromBoxedVector(input.Get({})));
  }
  if (input.type()->IsTuple()) {
    std::vector<VB> elements;
    for (int64_t i = 0; i < input.type()->AsTupleOrDie()->size(); ++i) {
      elements.push_back(BuilderFromBoxedValue(input.AsView({i})));
    }
    return VB::TupleB(elements);
  }
  CHECK(input.type()->IsArray());
  std::vector<VB> elements;
  for (int64_t i = 0; i < input.type()->AsArrayOrDie()->size(); ++i) {
    elements.push_back(BuilderFromBoxedValue(input.AsView({i})));
  }
  return VB::ArrayB(elements);
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

absl::StatusOr<Value> AbstractEvaluate(BValue v) {
  TestAbstractEvaluator tea;
  BoxedBoolNodeEvaluator eval(tea);
  XLS_RETURN_IF_ERROR(v.node()->function_base()->Accept(&eval));
  if (v.node()->GetType()->IsBits()) {
    XLS_ASSIGN_OR_RETURN(auto out, eval.GetValue(v.node()));
    return Value(FromBoxedVector(out));
  }
  XLS_ASSIGN_OR_RETURN(auto out, eval.GetCompoundValue(v.node()));
  return BuilderFromBoxedValue(out).Build();
}

class AbstractNodeEvaluatorTest : public IrTestBase {};

TEST_F(AbstractNodeEvaluatorTest, TestUMul) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue res = fb.UMul(fb.Literal(UBits(2, 4)), fb.Literal(UBits(4, 4)),
                       /*result_width=*/32);
  XLS_ASSERT_OK(fb.Build().status());
  EXPECT_THAT(AbstractEvaluate(res),
              status_testing::IsOkAndHolds(Value(UBits(8, 32))));
}

TEST_F(AbstractNodeEvaluatorTest, TestSMul) {
  // -1 * 4
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto res = fb.SMul(fb.Literal(UBits(0b1111, 4)), fb.Literal(UBits(4, 4)),
                     /*result_width=*/16);
  XLS_ASSERT_OK(fb.Build().status());
  EXPECT_THAT(AbstractEvaluate(res), status_testing::IsOkAndHolds(
                                         Value(UBits(0b1111111111111100, 16))));
}

TEST_F(AbstractNodeEvaluatorTest, TestAdd) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto res = fb.Add(fb.Literal(UBits(2, 4)), fb.Literal(UBits(4, 4)));
  XLS_ASSERT_OK(fb.Build().status());
  EXPECT_THAT(AbstractEvaluate(res),
              status_testing::IsOkAndHolds(Value(UBits(6, 4))));
}

TEST_F(AbstractNodeEvaluatorTest, TestUSub) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto res = fb.Subtract(fb.Literal(UBits(4, 4)), fb.Literal(UBits(2, 4)));
  XLS_ASSERT_OK(fb.Build().status());
  EXPECT_THAT(AbstractEvaluate(res),
              status_testing::IsOkAndHolds(Value(UBits(2, 4))));
}

TEST_F(AbstractNodeEvaluatorTest, TestUDiv) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto res = fb.UDiv(fb.Literal(UBits(9, 32)), fb.Literal(UBits(3, 32)));
  XLS_ASSERT_OK(fb.Build().status());
  EXPECT_THAT(AbstractEvaluate(res),
              status_testing::IsOkAndHolds(Value(UBits(3, 32))));
}

TEST_F(AbstractNodeEvaluatorTest, TestSDiv) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto res = fb.SDiv(fb.Literal(UBits(-9, 64)), fb.Literal(UBits(3, 64)));
  XLS_ASSERT_OK(fb.Build().status());
  EXPECT_THAT(AbstractEvaluate(res),
              status_testing::IsOkAndHolds(Value(UBits(-3, 64))));
}

TEST_F(AbstractNodeEvaluatorTest, TestShll) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto res = fb.Shll(fb.Literal(UBits(0b1010, 4)), fb.Literal(UBits(2, 4)));
  XLS_ASSERT_OK(fb.Build().status());
  EXPECT_THAT(AbstractEvaluate(res),
              status_testing::IsOkAndHolds(Value(UBits(0b1000, 4))));
}

TEST_F(AbstractNodeEvaluatorTest, TestShrl) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto res = fb.Shrl(fb.Literal(UBits(0b1010, 4)), fb.Literal(UBits(2, 4)));
  XLS_ASSERT_OK(fb.Build().status());
  EXPECT_THAT(AbstractEvaluate(res),
              status_testing::IsOkAndHolds(Value(UBits(0b0010, 4))));
}

TEST_F(AbstractNodeEvaluatorTest, TestShra) {
  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    auto res = fb.Shra(fb.Literal(UBits(0b1010, 4)), fb.Literal(UBits(2, 4)));
    XLS_ASSERT_OK(fb.Build().status());
    EXPECT_THAT(AbstractEvaluate(res),
                status_testing::IsOkAndHolds(Value(UBits(0b1110, 4))));
  }
  {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    auto res = fb.Shra(fb.Literal(UBits(0b0101, 4)), fb.Literal(UBits(2, 4)));
    XLS_ASSERT_OK(fb.Build().status());
    EXPECT_THAT(AbstractEvaluate(res),
                status_testing::IsOkAndHolds(Value(UBits(0b0001, 4))));
  }
}

TEST_F(AbstractNodeEvaluatorTest, TestUMod) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto res = fb.UMod(fb.Literal(UBits(9, 32)), fb.Literal(UBits(4, 32)));
  XLS_ASSERT_OK(fb.Build().status());
  EXPECT_THAT(AbstractEvaluate(res),
              status_testing::IsOkAndHolds(Value(UBits(1, 32))));
}

TEST_F(AbstractNodeEvaluatorTest, TestGate) {
  auto test_with = [&](Value cond, Value value) -> absl::StatusOr<Value> {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    auto res = fb.Gate(fb.Literal(cond), fb.Literal(value));
    XLS_RETURN_IF_ERROR(fb.Build().status());
    return AbstractEvaluate(res);
  };
  EXPECT_THAT(test_with(Value(UBits(0, 1)), Value(UBits(0b111, 3))),
              status_testing::IsOkAndHolds(Value(UBits(0b000, 3))));
  EXPECT_THAT(test_with(Value(UBits(1, 1)), Value(UBits(0b111, 3))),
              status_testing::IsOkAndHolds(Value(UBits(0b111, 3))));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value target,
      VB::Tuple({VB::Tuple({VB::Bits(UBits(3, 4)), VB::Bits(UBits(5, 6))}),
                 VB::UBits2DArray({{1, 2, 3}, {4, 5, 6}}, 8)})
          .Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      Value zero,
      VB::Tuple({VB::Tuple({VB::Bits(UBits(0, 4)), VB::Bits(UBits(0, 6))}),
                 VB::UBits2DArray({{0, 0, 0}, {0, 0, 0}}, 8)})
          .Build());
  EXPECT_THAT(test_with(Value(UBits(0, 1)), target),
              status_testing::IsOkAndHolds(zero));
  EXPECT_THAT(test_with(Value(UBits(1, 1)), target),
              status_testing::IsOkAndHolds(target));
}

TEST_F(AbstractNodeEvaluatorTest, Identity) {
  auto test_with = [&](Value value) -> absl::StatusOr<Value> {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    auto res = fb.Identity(fb.Literal(std::move(value)));
    XLS_RETURN_IF_ERROR(fb.Build().status());
    return AbstractEvaluate(res);
  };
  EXPECT_THAT(test_with(Value(UBits(0b111, 3))),
              status_testing::IsOkAndHolds(Value(UBits(0b111, 3))));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value target,
      VB::Tuple({VB::Tuple({VB::Bits(UBits(3, 4)), VB::Bits(UBits(5, 6))}),
                 VB::UBits2DArray({{1, 2, 3}, {4, 5, 6}}, 8)})
          .Build());
  EXPECT_THAT(test_with(target), status_testing::IsOkAndHolds(target));
  EXPECT_THAT(test_with(Value::Tuple({})),
              status_testing::IsOkAndHolds(Value::Tuple({})));
}

TEST_F(AbstractNodeEvaluatorTest, Eq) {
  auto test_with = [&](Value v1, Value v2) -> absl::StatusOr<Value> {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    auto res = fb.Eq(fb.Literal(std::move(v1)), fb.Literal(std::move(v2)));
    XLS_RETURN_IF_ERROR(fb.Build().status());
    return AbstractEvaluate(res);
  };
  auto v_true = Value::Bool(true);
  auto v_false = Value::Bool(false);
  EXPECT_THAT(test_with(Value(UBits(3, 4)), Value(UBits(3, 4))),
              status_testing::IsOkAndHolds(v_true));
  EXPECT_THAT(test_with(Value(UBits(0, 4)), Value(UBits(3, 4))),
              status_testing::IsOkAndHolds(v_false));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value target,
      VB::Tuple({VB::Tuple({VB::Bits(UBits(3, 4)), VB::Bits(UBits(5, 6))}),
                 VB::UBits2DArray({{1, 2, 3}, {4, 5, 6}}, 8)})
          .Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      Value zero,
      VB::Tuple({VB::Tuple({VB::Bits(UBits(0, 4)), VB::Bits(UBits(0, 6))}),
                 VB::UBits2DArray({{0, 0, 0}, {0, 0, 0}}, 8)})
          .Build());
  EXPECT_THAT(test_with(target, target), status_testing::IsOkAndHolds(v_true));
  EXPECT_THAT(test_with(target, zero), status_testing::IsOkAndHolds(v_false));
}

TEST_F(AbstractNodeEvaluatorTest, Ne) {
  auto test_with = [&](Value v1, Value v2) -> absl::StatusOr<Value> {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    auto res = fb.Ne(fb.Literal(std::move(v1)), fb.Literal(std::move(v2)));
    XLS_RETURN_IF_ERROR(fb.Build().status());
    return AbstractEvaluate(res);
  };
  auto v_true = Value::Bool(true);
  auto v_false = Value::Bool(false);
  EXPECT_THAT(test_with(Value(UBits(3, 4)), Value(UBits(3, 4))),
              status_testing::IsOkAndHolds(v_false));
  EXPECT_THAT(test_with(Value(UBits(0, 4)), Value(UBits(3, 4))),
              status_testing::IsOkAndHolds(v_true));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value target,
      VB::Tuple({VB::Tuple({VB::Bits(UBits(3, 4)), VB::Bits(UBits(5, 6))}),
                 VB::UBits2DArray({{1, 2, 3}, {4, 5, 6}}, 8)})
          .Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      Value zero,
      VB::Tuple({VB::Tuple({VB::Bits(UBits(0, 4)), VB::Bits(UBits(0, 6))}),
                 VB::UBits2DArray({{0, 0, 0}, {0, 0, 0}}, 8)})
          .Build());
  EXPECT_THAT(test_with(target, target), status_testing::IsOkAndHolds(v_false));
  EXPECT_THAT(test_with(target, zero), status_testing::IsOkAndHolds(v_true));
}

TEST_F(AbstractNodeEvaluatorTest, Sel) {
  auto test_with = [&](Value selector, absl::Span<Value const> cases,
                       std::optional<Value> default_value =
                           std::nullopt) -> absl::StatusOr<Value> {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    std::vector<BValue> vals;
    vals.reserve(cases.size());
    for (const auto& v : cases) {
      vals.push_back(fb.Literal(v));
    }
    auto res = fb.Select(
        fb.Literal(std::move(selector)), vals,
        default_value
            ? std::make_optional(fb.Literal(*std::move(default_value)))
            : std::nullopt);
    XLS_RETURN_IF_ERROR(fb.Build().status());
    return AbstractEvaluate(res);
  };
  EXPECT_THAT(
      test_with(Value(UBits(2, 2)), {Value(UBits(0, 4)), Value(UBits(1, 4)),
                                     Value(UBits(2, 4)), Value(UBits(3, 4))}),
      status_testing::IsOkAndHolds(Value(UBits(2, 4))));
  EXPECT_THAT(test_with(Value(UBits(6, 20)),
                        {Value(UBits(0, 4)), Value(UBits(1, 4)),
                         Value(UBits(2, 4)), Value(UBits(3, 4))},
                        Value(UBits(4, 4))),
              status_testing::IsOkAndHolds(Value(UBits(4, 4))));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value zero,
      VB::Tuple({VB::Tuple({VB::Bits(UBits(0, 4)), VB::Bits(UBits(0, 6))}),
                 VB::UBits2DArray({{0, 0, 0}, {0, 0, 0}}, 8)})
          .Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      Value one,
      VB::Tuple({VB::Tuple({VB::Bits(UBits(1, 4)), VB::Bits(UBits(1, 6))}),
                 VB::UBits2DArray({{1, 1, 1}, {1, 1, 1}}, 8)})
          .Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      Value two,
      VB::Tuple({VB::Tuple({VB::Bits(UBits(2, 4)), VB::Bits(UBits(2, 6))}),
                 VB::UBits2DArray({{2, 2, 2}, {2, 2, 2}}, 8)})
          .Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      Value three,
      VB::Tuple({VB::Tuple({VB::Bits(UBits(3, 4)), VB::Bits(UBits(3, 6))}),
                 VB::UBits2DArray({{3, 3, 3}, {3, 3, 3}}, 8)})
          .Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      Value four,
      VB::Tuple({VB::Tuple({VB::Bits(UBits(4, 4)), VB::Bits(UBits(4, 6))}),
                 VB::UBits2DArray({{4, 4, 4}, {4, 4, 4}}, 8)})
          .Build());

  EXPECT_THAT(test_with(Value(UBits(2, 2)), {zero, one, two, three}),
              status_testing::IsOkAndHolds(two));
  EXPECT_THAT(test_with(Value(UBits(6, 20)), {zero, one, two, three}, four),
              status_testing::IsOkAndHolds(four));
}

TEST_F(AbstractNodeEvaluatorTest, OneHotSel) {
  auto test_with = [&](Value selector,
                       absl::Span<Value const> cases) -> absl::StatusOr<Value> {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    std::vector<BValue> vals;
    vals.reserve(cases.size());
    for (const auto& v : cases) {
      vals.push_back(fb.Literal(v));
    }
    auto res = fb.OneHotSelect(fb.Literal(std::move(selector)), vals);
    XLS_RETURN_IF_ERROR(fb.Build().status());
    return AbstractEvaluate(res);
  };
  EXPECT_THAT(test_with(Value(UBits(0b0010, 4)),
                        {Value(UBits(0, 4)), Value(UBits(1, 4)),
                         Value(UBits(2, 4)), Value(UBits(3, 4))}),
              status_testing::IsOkAndHolds(Value(UBits(1, 4))));
  EXPECT_THAT(test_with(Value(UBits(0b00, 2)),
                        {Value(UBits(1, 4)), Value(UBits(2, 4))}),
              status_testing::IsOkAndHolds(Value(UBits(0, 4))));
  EXPECT_THAT(test_with(Value(UBits(0b11, 2)),
                        {Value(UBits(1, 4)), Value(UBits(2, 4))}),
              status_testing::IsOkAndHolds(Value(UBits(3, 4))));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value zero,
      VB::Tuple({VB::Tuple({VB::Bits(UBits(0, 4)), VB::Bits(UBits(0, 6))}),
                 VB::UBits2DArray({{0, 0, 0}, {0, 0, 0}}, 8)})
          .Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      Value one,
      VB::Tuple({VB::Tuple({VB::Bits(UBits(1, 4)), VB::Bits(UBits(1, 6))}),
                 VB::UBits2DArray({{1, 1, 1}, {1, 1, 1}}, 8)})
          .Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      Value two,
      VB::Tuple({VB::Tuple({VB::Bits(UBits(2, 4)), VB::Bits(UBits(2, 6))}),
                 VB::UBits2DArray({{2, 2, 2}, {2, 2, 2}}, 8)})
          .Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      Value three,
      VB::Tuple({VB::Tuple({VB::Bits(UBits(3, 4)), VB::Bits(UBits(3, 6))}),
                 VB::UBits2DArray({{3, 3, 3}, {3, 3, 3}}, 8)})
          .Build());
  EXPECT_THAT(test_with(Value(UBits(0b0010, 4)), {zero, one, two, three}),
              status_testing::IsOkAndHolds(one));
  EXPECT_THAT(test_with(Value(UBits(0, 2)), {one, two}),
              status_testing::IsOkAndHolds(zero));
  EXPECT_THAT(test_with(Value(UBits(0b11, 2)), {one, two}),
              status_testing::IsOkAndHolds(three));
}

TEST_F(AbstractNodeEvaluatorTest, PrioritySel) {
  auto test_with = [&](Value selector, absl::Span<Value const> cases,
                       Value default_value) -> absl::StatusOr<Value> {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    std::vector<BValue> vals;
    vals.reserve(cases.size());
    for (const auto& v : cases) {
      vals.push_back(fb.Literal(v));
    }
    auto res = fb.PrioritySelect(fb.Literal(std::move(selector)), vals,
                                 fb.Literal(std::move(default_value)));
    XLS_RETURN_IF_ERROR(fb.Build().status());
    return AbstractEvaluate(res);
  };
  EXPECT_THAT(test_with(Value(UBits(0b0010, 4)),
                        {Value(UBits(0, 4)), Value(UBits(1, 4)),
                         Value(UBits(2, 4)), Value(UBits(3, 4))},
                        Value(UBits(8, 4))),
              status_testing::IsOkAndHolds(Value(UBits(1, 4))));
  EXPECT_THAT(
      test_with(Value(UBits(0b00, 2)), {Value(UBits(1, 4)), Value(UBits(2, 4))},
                Value(UBits(8, 4))),
      status_testing::IsOkAndHolds(Value(UBits(8, 4))));
  EXPECT_THAT(
      test_with(Value(UBits(0b11, 2)), {Value(UBits(1, 4)), Value(UBits(2, 4))},
                Value(UBits(8, 4))),
      status_testing::IsOkAndHolds(Value(UBits(1, 4))));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value zero,
      VB::Tuple({VB::Tuple({VB::Bits(UBits(0, 4)), VB::Bits(UBits(0, 6))}),
                 VB::UBits2DArray({{0, 0, 0}, {0, 0, 0}}, 8)})
          .Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      Value one,
      VB::Tuple({VB::Tuple({VB::Bits(UBits(1, 4)), VB::Bits(UBits(1, 6))}),
                 VB::UBits2DArray({{1, 1, 1}, {1, 1, 1}}, 8)})
          .Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      Value two,
      VB::Tuple({VB::Tuple({VB::Bits(UBits(2, 4)), VB::Bits(UBits(2, 6))}),
                 VB::UBits2DArray({{2, 2, 2}, {2, 2, 2}}, 8)})
          .Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      Value three,
      VB::Tuple({VB::Tuple({VB::Bits(UBits(3, 4)), VB::Bits(UBits(3, 6))}),
                 VB::UBits2DArray({{3, 3, 3}, {3, 3, 3}}, 8)})
          .Build());
  EXPECT_THAT(test_with(Value(UBits(0b0010, 4)), {zero, one, two, three}, zero),
              status_testing::IsOkAndHolds(one));
  EXPECT_THAT(test_with(Value(UBits(0, 2)), {one, two}, zero),
              status_testing::IsOkAndHolds(zero));
  EXPECT_THAT(test_with(Value(UBits(0b11, 2)), {one, two}, zero),
              status_testing::IsOkAndHolds(one));
}

TEST_F(AbstractNodeEvaluatorTest, Array) {
  auto test_with = [&](absl::Span<VB const> values) -> absl::StatusOr<Value> {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    std::vector<BValue> elements;
    elements.reserve(values.size());
    for (const auto& v : values) {
      XLS_ASSIGN_OR_RETURN(auto e, v.Build());
      elements.push_back(fb.Literal(e));
    }
    auto res = fb.Array(elements, elements.front().GetType());
    XLS_RETURN_IF_ERROR(fb.Build().status());
    return AbstractEvaluate(res);
  };
  std::array<VB, 4> t1 = {VB::Bits(UBits(0, 4)), VB::Bits(UBits(1, 4)),
                          VB::Bits(UBits(2, 4)), VB::Bits(UBits(3, 4))};
  XLS_ASSERT_OK_AND_ASSIGN(auto a1, VB::ArrayB(t1).Build());
  EXPECT_THAT(test_with(t1), status_testing::IsOkAndHolds(a1));

  auto zero =
      VB::Tuple({VB::Tuple({VB::Bits(UBits(0, 4)), VB::Bits(UBits(0, 6))}),
                 VB::UBits2DArray({{0, 0, 0}, {0, 0, 0}}, 8)});
  auto one =
      VB::Tuple({VB::Tuple({VB::Bits(UBits(1, 4)), VB::Bits(UBits(1, 6))}),
                 VB::UBits2DArray({{1, 1, 1}, {1, 1, 1}}, 8)});
  auto two =
      VB::Tuple({VB::Tuple({VB::Bits(UBits(2, 4)), VB::Bits(UBits(2, 6))}),
                 VB::UBits2DArray({{2, 2, 2}, {2, 2, 2}}, 8)});
  auto three =
      VB::Tuple({VB::Tuple({VB::Bits(UBits(3, 4)), VB::Bits(UBits(3, 6))}),
                 VB::UBits2DArray({{3, 3, 3}, {3, 3, 3}}, 8)});
  XLS_ASSERT_OK_AND_ASSIGN(auto a2, VB::Array({zero, one, two, three}).Build());
  EXPECT_THAT(test_with({zero, one, two, three}),
              status_testing::IsOkAndHolds(a2));
}

TEST_F(AbstractNodeEvaluatorTest, ArrayConcat) {
  auto test_with =
      [&](absl::Span<absl::Span<VB const> const> vs) -> absl::StatusOr<Value> {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    std::vector<BValue> arrs;
    arrs.reserve(vs.size());
    for (auto v : vs) {
      XLS_ASSIGN_OR_RETURN(auto a, VB::ArrayB(v).Build());
      arrs.push_back(fb.Literal(a));
    }
    auto res = fb.ArrayConcat(arrs);
    XLS_RETURN_IF_ERROR(fb.Build().status());
    return AbstractEvaluate(res);
  };
  auto bit_build = [](int64_t v) { return VB::Bits(UBits(v, 32)); };
  auto comp_build = [](uint64_t v) {
    return VB::Tuple(
        {VB::Tuple({VB::Bits(UBits(v, 48)), VB::Bits(UBits(v, 64))}),
         VB::UBits2DArray({{v, v, v}, {v, v, v}}, 80)});
  };
  std::array<VB, 12> all_ints{
      bit_build(0), bit_build(1), bit_build(2),  bit_build(3),
      bit_build(4), bit_build(5), bit_build(6),  bit_build(7),
      bit_build(8), bit_build(9), bit_build(10), bit_build(11),
  };
  absl::Span<VB const> ints = all_ints;
  std::array<VB, 12> all_comp{
      comp_build(0), comp_build(1), comp_build(2),  comp_build(3),
      comp_build(4), comp_build(5), comp_build(6),  comp_build(7),
      comp_build(8), comp_build(9), comp_build(10), comp_build(11),
  };
  absl::Span<VB const> comps = all_comp;

  XLS_ASSERT_OK_AND_ASSIGN(auto ints_result, VB::ArrayB(ints).Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto comps_result, VB::ArrayB(comps).Build());
  EXPECT_THAT(
      test_with({ints.subspan(0, 4), ints.subspan(4, 2), ints.subspan(6)}),
      status_testing::IsOkAndHolds(ints_result));
  EXPECT_THAT(
      test_with({ints.subspan(0, 2), ints.subspan(2, 2), ints.subspan(4, 2),
                 ints.subspan(6, 2), ints.subspan(8, 2), ints.subspan(10)}),
      status_testing::IsOkAndHolds(ints_result));
  EXPECT_THAT(
      test_with({comps.subspan(0, 4), comps.subspan(4, 2), comps.subspan(6)}),
      status_testing::IsOkAndHolds(comps_result));
  EXPECT_THAT(
      test_with({comps.subspan(0, 2), comps.subspan(2, 2), comps.subspan(4, 2),
                 comps.subspan(6, 2), comps.subspan(8, 2), comps.subspan(10)}),
      status_testing::IsOkAndHolds(comps_result));
}

TEST_F(AbstractNodeEvaluatorTest, Tuple) {
  auto test_with = [&](absl::Span<VB const> values) -> absl::StatusOr<Value> {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    std::vector<BValue> elements;
    elements.reserve(values.size());
    for (const auto& v : values) {
      XLS_ASSIGN_OR_RETURN(auto e, v.Build());
      elements.push_back(fb.Literal(e));
    }
    auto res = fb.Tuple(elements);
    XLS_RETURN_IF_ERROR(fb.Build().status());
    return AbstractEvaluate(res);
  };
  std::array<VB, 4> t1 = {VB::Bits(UBits(0, 4)), VB::Bits(UBits(1, 4)),
                          VB::Bits(UBits(2, 4)), VB::Bits(UBits(3, 4))};
  XLS_ASSERT_OK_AND_ASSIGN(auto a1, VB::TupleB(t1).Build());
  EXPECT_THAT(test_with(t1), status_testing::IsOkAndHolds(a1));

  auto zero =
      VB::Tuple({VB::Tuple({VB::Bits(UBits(0, 4)), VB::Bits(UBits(0, 6))}),
                 VB::UBits2DArray({{0, 0, 0}, {0, 0, 0}}, 8)});
  auto one =
      VB::Tuple({VB::Tuple({VB::Bits(UBits(1, 4)), VB::Bits(UBits(1, 6))}),
                 VB::UBits2DArray({{1, 1, 1}, {1, 1, 1}}, 8)});
  auto two =
      VB::Tuple({VB::Tuple({VB::Bits(UBits(2, 4)), VB::Bits(UBits(2, 6))}),
                 VB::UBits2DArray({{2, 2, 2}, {2, 2, 2}}, 8)});
  auto three =
      VB::Tuple({VB::Tuple({VB::Bits(UBits(3, 4)), VB::Bits(UBits(3, 6))}),
                 VB::UBits2DArray({{3, 3, 3}, {3, 3, 3}}, 8)});
  XLS_ASSERT_OK_AND_ASSIGN(auto a2, VB::Tuple({zero, one, two, three}).Build());
  EXPECT_THAT(test_with({zero, one, two, three}),
              status_testing::IsOkAndHolds(a2));
  auto foo = VB::UBits2DArray({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, 16);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto a3, VB::Tuple({zero, foo, one, foo, two, foo, three, foo}).Build());
  EXPECT_THAT(test_with({zero, foo, one, foo, two, foo, three, foo}),
              status_testing::IsOkAndHolds(a3));
}

TEST_F(AbstractNodeEvaluatorTest, TupleIndex) {
  auto test_with =
      [&](VB tuple,
          absl::Span<int64_t const> indexes) -> absl::StatusOr<Value> {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    XLS_RET_CHECK(tuple.IsTuple());
    XLS_ASSIGN_OR_RETURN(auto tv, tuple.Build());
    auto tup = fb.Literal(tv);
    for (int64_t i : indexes) {
      tup = fb.TupleIndex(tup, i);
    }
    XLS_RETURN_IF_ERROR(fb.Build().status());
    return AbstractEvaluate(tup);
  };

  {
    auto res = Value(UBits(3, 4));
    EXPECT_THAT(test_with(VB::Tuple({VB::Bits(UBits(1, 1)),
                                     VB::Tuple({VB::Bits(UBits(1, 1)), res})}),
                          {1, 1}),
                status_testing::IsOkAndHolds(res));
  }
  {
    auto res = Value::Tuple({Value(UBits(3, 4)), Value(UBits(5, 6))});
    EXPECT_THAT(test_with(VB::Tuple({VB::Bits(UBits(1, 1)),
                                     VB::Tuple({VB::Bits(UBits(1, 1)), res})}),
                          {1, 1}),
                status_testing::IsOkAndHolds(res));
  }
  {
    auto res = Value::Tuple({});
    EXPECT_THAT(test_with(VB::Tuple({VB::Bits(UBits(1, 1)),
                                     VB::Tuple({VB::Bits(UBits(1, 1)), res})}),
                          {1, 1}),
                status_testing::IsOkAndHolds(res));
  }
}
}  // namespace
}  // namespace xls
