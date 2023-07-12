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

#include "xls/delay_model/delay_estimator.h"

#include <cstdint>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/op.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::ElementsAre;

// A test delay estimator that returns a fixed delay for every node.
class TestDelayEstimator : public DelayEstimator {
 public:
  explicit TestDelayEstimator(int64_t delay, std::string_view name)
      : DelayEstimator(name), delay_(delay) {}

  absl::StatusOr<int64_t> GetOperationDelayInPs(Node* node) const override {
    return delay_;
  }

 private:
  int64_t delay_;
};

class DelayEstimatorTest : public IrTestBase {};

TEST_F(DelayEstimatorTest, DelayEstimatorManager) {
  DelayEstimatorManager manager;
  EXPECT_THAT(manager.estimator_names(), ElementsAre());

  XLS_ASSERT_OK(manager.RegisterDelayEstimator(
      std::make_unique<TestDelayEstimator>(42, "forty_two"),
      DelayEstimatorPrecedence::kLow));
  XLS_ASSERT_OK(manager.RegisterDelayEstimator(
      std::make_unique<TestDelayEstimator>(1, "one"),
      DelayEstimatorPrecedence::kLow));

  EXPECT_THAT(manager.estimator_names(), ElementsAre("forty_two", "one"));

  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Literal(UBits(42, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * forty_two,
                           manager.GetDelayEstimator("forty_two"));
  EXPECT_THAT(forty_two->GetOperationDelayInPs(f->return_value()),
              IsOkAndHolds(42));

  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * one,
                           manager.GetDelayEstimator("one"));
  EXPECT_THAT(one->GetOperationDelayInPs(f->return_value()), IsOkAndHolds(1));

  EXPECT_THAT(manager.GetDelayEstimator("foo"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(DelayEstimatorTest, LogicalEffortForXors) {
  {
    // Two-input XOR.
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    BValue x = fb.Param("x", p->GetBitsType(1));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                             fb.BuildWithReturnValue(fb.Xor({x, x})));
    EXPECT_THAT(
        DelayEstimator::GetLogicalEffortDelayInPs(f->return_value(), 10),
        IsOkAndHolds(40));
  }

  {
    // Many-input XOR.
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    BValue x = fb.Param("x", p->GetBitsType(1));
    std::vector<BValue> xor_inputs(100, x);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                             fb.BuildWithReturnValue(fb.Xor(xor_inputs)));
    EXPECT_THAT(
        DelayEstimator::GetLogicalEffortDelayInPs(f->return_value(), 10),
        IsOkAndHolds(280));
  }
}

TEST_F(DelayEstimatorTest, DecoratingDelayEstimator) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Xor({x, x})));
  DelayEstimatorManager manager;
  XLS_ASSERT_OK(manager.RegisterDelayEstimator(
      std::make_unique<TestDelayEstimator>(1, "one"),
      DelayEstimatorPrecedence::kLow));
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * one,
                           manager.GetDelayEstimator("one"));
  DecoratingDelayEstimator decorating("decorating", *one,
                                      [](Node* n, int64_t original) -> int64_t {
                                        EXPECT_EQ(n->GetName(), "xor.2");
                                        EXPECT_EQ(original, 1);
                                        return 42;
                                      });
  EXPECT_THAT(decorating.GetOperationDelayInPs(f->return_value()),
              IsOkAndHolds(42));
}

// A Delay Estimator that can only handle one kind of operation.
class TestNodeMatchEstimator : public DelayEstimator {
 public:
  TestNodeMatchEstimator(Op match_op, int64_t delay, std::string_view name)
      : DelayEstimator(name), delay_(delay), match_op_(match_op) {}

  absl::StatusOr<int64_t> GetOperationDelayInPs(Node* node) const override {
    if (node->op() == match_op_) {
      return delay_;
    }
    return absl::UnimplementedError("not a matching op.");
  }

 private:
  int64_t delay_;
  Op match_op_;
};

TEST_F(DelayEstimatorTest, FirstMatchDelegationEstimator) {
  TestNodeMatchEstimator only_xor(Op::kXor, 10, "only_xor");
  TestNodeMatchEstimator only_add(Op::kAdd, 20, "only_add");
  FirstMatchDelayEstimator match_estimator("match_first",
                                           {&only_xor, &only_add});

  auto p = CreatePackage();
  {
    FunctionBuilder fb("xor", p.get());
    BValue x = fb.Param("x", p->GetBitsType(1));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                             fb.BuildWithReturnValue(fb.Xor({x, x})));
    EXPECT_THAT(match_estimator.GetOperationDelayInPs(f->return_value()),
                IsOkAndHolds(10));
  }
  {
    FunctionBuilder fb("add", p.get());
    BValue x = fb.Param("x", p->GetBitsType(1));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                             fb.BuildWithReturnValue(fb.Add(x, x)));
    EXPECT_THAT(match_estimator.GetOperationDelayInPs(f->return_value()),
                IsOkAndHolds(20));
  }
  {
    FunctionBuilder fb("or", p.get());
    BValue x = fb.Param("x", p->GetBitsType(1));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                             fb.BuildWithReturnValue(fb.Or({x, x})));
    EXPECT_THAT(match_estimator.GetOperationDelayInPs(f->return_value()),
                StatusIs(absl::StatusCode::kUnimplemented));
  }
}

}  // namespace
}  // namespace xls
