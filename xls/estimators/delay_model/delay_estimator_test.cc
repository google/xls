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

#include "xls/estimators/delay_model/delay_estimator.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "google/protobuf/text_format.h"
#include "xls/common/status/matchers.h"
#include "xls/estimators/estimator_model.pb.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/op.h"

namespace xls {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

// A test delay estimator that returns a fixed delay for every node.
class FakeDelayEstimator : public DelayEstimator {
 public:
  explicit FakeDelayEstimator(int64_t delay, std::string_view name)
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
      std::make_unique<FakeDelayEstimator>(42, "forty_two"),
      DelayEstimatorPrecedence::kLow));
  XLS_ASSERT_OK(manager.RegisterDelayEstimator(
      std::make_unique<FakeDelayEstimator>(1, "one"),
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

TEST_F(DelayEstimatorTest, LogicalEffortForAndReduces) {
  {
    // 10bit AndReduce.
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    BValue x = fb.Param("x", p->GetBitsType(10));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                             fb.BuildWithReturnValue(fb.AndReduce(x)));
    EXPECT_THAT(
        DelayEstimator::GetLogicalEffortDelayInPs(f->return_value(), 10),
        IsOkAndHolds(50));
  }

  {
    // 100bit AndReduce.
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    BValue x = fb.Param("x", p->GetBitsType(100));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                             fb.BuildWithReturnValue(fb.AndReduce(x)));
    EXPECT_THAT(
        DelayEstimator::GetLogicalEffortDelayInPs(f->return_value(), 10),
        IsOkAndHolds(210));
  }
}

TEST_F(DelayEstimatorTest, LogicalEffortForOrReduces) {
  {
    // 10bit OrReduce.
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    BValue x = fb.Param("x", p->GetBitsType(10));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                             fb.BuildWithReturnValue(fb.OrReduce(x)));
    EXPECT_THAT(
        DelayEstimator::GetLogicalEffortDelayInPs(f->return_value(), 10),
        IsOkAndHolds(80));
  }

  {
    // 100bit OrReduce.
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    BValue x = fb.Param("x", p->GetBitsType(100));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                             fb.BuildWithReturnValue(fb.OrReduce(x)));
    EXPECT_THAT(
        DelayEstimator::GetLogicalEffortDelayInPs(f->return_value(), 10),
        IsOkAndHolds(210));
  }
}

TEST_F(DelayEstimatorTest, LogicalEffortForXorReduces) {
  {
    // 10bit XORReduce.
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    BValue x = fb.Param("x", p->GetBitsType(10));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                             fb.BuildWithReturnValue(fb.XorReduce(x)));
    EXPECT_THAT(
        DelayEstimator::GetLogicalEffortDelayInPs(f->return_value(), 10),
        IsOkAndHolds(160));
  }

  {
    // 100bit xorreduce.
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    BValue x = fb.Param("x", p->GetBitsType(100));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                             fb.BuildWithReturnValue(fb.XorReduce(x)));
    EXPECT_THAT(
        DelayEstimator::GetLogicalEffortDelayInPs(f->return_value(), 10),
        IsOkAndHolds(280));
  }
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

TEST_F(DelayEstimatorTest, LogicalEffortForOneHotSelects) {
  {
    // 3-input OneHotSelect.
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    BValue s = fb.Param("s", p->GetBitsType(3));
    BValue a = fb.Param("a", p->GetBitsType(10));
    BValue b = fb.Param("b", p->GetBitsType(10));
    BValue c = fb.Param("c", p->GetBitsType(10));
    XLS_ASSERT_OK_AND_ASSIGN(
        Function * f, fb.BuildWithReturnValue(fb.OneHotSelect(s, {a, b, c})));
    EXPECT_THAT(
        DelayEstimator::GetLogicalEffortDelayInPs(f->return_value(), 10),
        IsOkAndHolds(40));
  }

  {
    // 3-input OneHotSelect with literal selector (selecting 2 inputs)
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    BValue s = fb.Literal(UBits(0b101, 3));
    BValue a = fb.Param("a", p->GetBitsType(10));
    BValue b = fb.Param("b", p->GetBitsType(10));
    BValue c = fb.Param("c", p->GetBitsType(10));
    XLS_ASSERT_OK_AND_ASSIGN(
        Function * f, fb.BuildWithReturnValue(fb.OneHotSelect(s, {a, b, c})));
    EXPECT_THAT(
        DelayEstimator::GetLogicalEffortDelayInPs(f->return_value(), 10),
        IsOkAndHolds(30));
  }

  {
    // 3-input OneHotSelect with literal selector (selecting 1 input)
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    BValue s = fb.Literal(UBits(0b010, 3));
    BValue a = fb.Param("a", p->GetBitsType(10));
    BValue b = fb.Param("b", p->GetBitsType(10));
    BValue c = fb.Param("c", p->GetBitsType(10));
    XLS_ASSERT_OK_AND_ASSIGN(
        Function * f, fb.BuildWithReturnValue(fb.OneHotSelect(s, {a, b, c})));
    EXPECT_THAT(
        DelayEstimator::GetLogicalEffortDelayInPs(f->return_value(), 10),
        IsOkAndHolds(0));
  }

  {
    // 3-input OneHotSelect with literal selector (selecting no inputs)
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    BValue s = fb.Literal(UBits(0b000, 3));
    BValue a = fb.Param("a", p->GetBitsType(10));
    BValue b = fb.Param("b", p->GetBitsType(10));
    BValue c = fb.Param("c", p->GetBitsType(10));
    XLS_ASSERT_OK_AND_ASSIGN(
        Function * f, fb.BuildWithReturnValue(fb.OneHotSelect(s, {a, b, c})));
    EXPECT_THAT(
        DelayEstimator::GetLogicalEffortDelayInPs(f->return_value(), 10),
        IsOkAndHolds(0));
  }

  {
    // 1-input OneHotSelect
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    BValue s = fb.Param("s", p->GetBitsType(1));
    BValue a = fb.Param("a", p->GetBitsType(10));
    XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                             fb.BuildWithReturnValue(fb.OneHotSelect(s, {a})));
    EXPECT_THAT(
        DelayEstimator::GetLogicalEffortDelayInPs(f->return_value(), 10),
        IsOkAndHolds(30));
  }
  {
    // 200-input OneHotSelect
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    BValue s = fb.Param("s", p->GetBitsType(200));
    std::vector<BValue> cases;
    cases.reserve(200);
    for (int i = 0; i < 200; ++i) {
      cases.push_back(
          fb.Param(absl::StrFormat("param_%d", i), p->GetBitsType(10)));
    }
    fb.OneHotSelect(s, cases);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
    EXPECT_THAT(
        DelayEstimator::GetLogicalEffortDelayInPs(f->return_value(), 10),
        IsOkAndHolds(180));
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
      std::make_unique<FakeDelayEstimator>(1, "one"),
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

  DecoratingDelayEstimator decorating_deduced_int(
      "decorating_deduced_int", *one, [](Node*, int64_t) { return 7; });
  EXPECT_THAT(decorating_deduced_int.GetOperationDelayInPs(f->return_value()),
              IsOkAndHolds(7));

  std::function<int64_t(Node*, int64_t)> int_modifier =
      [](Node*, int64_t original) { return original + 100; };
  DecoratingDelayEstimator decorating_int_fn_lvalue("decorating_int_fn_lvalue",
                                                    *one, int_modifier);
  EXPECT_THAT(decorating_int_fn_lvalue.GetOperationDelayInPs(f->return_value()),
              IsOkAndHolds(101));
  DecoratingDelayEstimator decorating_int_fn_rvalue(
      "decorating_int_fn_rvalue", *one, std::move(int_modifier));
  EXPECT_THAT(decorating_int_fn_rvalue.GetOperationDelayInPs(f->return_value()),
              IsOkAndHolds(101));

  std::function<absl::StatusOr<int64_t>(Node*, int64_t)> status_or_modifier =
      [](Node*, int64_t original) -> absl::StatusOr<int64_t> {
    return original + 200;
  };
  DecoratingDelayEstimator decorating_status_or_fn_lvalue(
      "decorating_status_or_fn_lvalue", *one, status_or_modifier);
  EXPECT_THAT(
      decorating_status_or_fn_lvalue.GetOperationDelayInPs(f->return_value()),
      IsOkAndHolds(201));
  DecoratingDelayEstimator decorating_status_or_fn_rvalue(
      "decorating_status_or_fn_rvalue", *one, std::move(status_or_modifier));
  EXPECT_THAT(
      decorating_status_or_fn_rvalue.GetOperationDelayInPs(f->return_value()),
      IsOkAndHolds(201));

  DecoratingDelayEstimator decorating_status_or_lambda(
      "decorating_status_or_lambda", *one,
      [](Node* n, int64_t original) -> absl::StatusOr<int64_t> {
        EXPECT_EQ(n->GetName(), "xor.2");
        EXPECT_EQ(original, 1);
        return 84;
      });
  EXPECT_THAT(
      decorating_status_or_lambda.GetOperationDelayInPs(f->return_value()),
      IsOkAndHolds(84));

  DecoratingDelayEstimator decorating_error(
      "decorating_error", *one,
      [](Node* n, int64_t original) -> absl::StatusOr<int64_t> {
        return absl::InvalidArgumentError("modifier failed");
      });
  EXPECT_THAT(decorating_error.GetOperationDelayInPs(f->return_value()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("modifier failed")));

  // Also verify propagating an error from the underlying estimator.
  DecoratingDelayEstimator decorating_unimplemented(
      "decorating_unimplemented", *one,
      [](Node* n, int64_t original) -> absl::StatusOr<int64_t> {
        if (n->op() == Op::kParam) {
          return absl::UnimplementedError("param delay unsupported");
        }
        return original + 10;
      });
  bool modifier_called = false;
  DecoratingDelayEstimator nested_decorating("nested_decorating",
                                             decorating_unimplemented,
                                             [&](Node* n, int64_t original) {
                                               modifier_called = true;
                                               return original + 5;
                                             });
  EXPECT_THAT(nested_decorating.GetOperationDelayInPs(x.node()),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("param delay unsupported")));
  EXPECT_FALSE(modifier_called);
  EXPECT_THAT(nested_decorating.GetOperationDelayInPs(f->return_value()),
              IsOkAndHolds(16));
  EXPECT_TRUE(modifier_called);
}

using DelayEstimatorDeathTest = DelayEstimatorTest;

TEST_F(DelayEstimatorDeathTest, DecoratingDelayEstimatorNullModifierDies) {
  FakeDelayEstimator base(1, "one");
  std::function<int64_t(Node*, int64_t)> null_int_modifier;
  EXPECT_DEATH(DecoratingDelayEstimator("null_int", base, null_int_modifier),
               HasSubstr("modifier"));
  std::function<absl::StatusOr<int64_t>(Node*, int64_t)>
      null_status_or_modifier;
  EXPECT_DEATH(
      DecoratingDelayEstimator("null_status_or", base, null_status_or_modifier),
      HasSubstr("modifier"));
}

TEST_F(DelayEstimatorTest, CachingDelayEstimator) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue node = fb.Xor({x, x});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(node));
  DelayEstimatorManager manager;
  XLS_ASSERT_OK(manager.RegisterDelayEstimator(
      std::make_unique<FakeDelayEstimator>(1, "one"),
      DelayEstimatorPrecedence::kLow));
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * one,
                           manager.GetDelayEstimator("one"));
  CachingDelayEstimator caching("caching", *one);
  EXPECT_THAT(caching.GetOperationDelayInPs(f->return_value()),
              IsOkAndHolds(1));
  EXPECT_THAT(caching.ContainsNodeDelay(f->return_value()), true);
  EXPECT_THAT(caching.GetNodeDelay(f->return_value()), 1);
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

TEST_F(DelayEstimatorTest, DelayAnnotator) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(32));
  BValue b = fb.Param("b", p->GetBitsType(32));
  BValue add = fb.Add(a, b);
  BValue neg = fb.Negate(add);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(neg));

  FakeDelayEstimator estimator(10, "ten");
  XLS_ASSERT_OK_AND_ASSIGN(DelayAnnotator annotator,
                           DelayAnnotator::Create(f, estimator));

  EXPECT_EQ(annotator.NodeAnnotation(a.node()).suffix, "[10ps (+10ps)]");
  EXPECT_EQ(annotator.NodeAnnotation(b.node()).suffix, "[10ps (+10ps)]");
  EXPECT_EQ(annotator.NodeAnnotation(add.node()).suffix, "[20ps (+10ps)]");
  EXPECT_EQ(annotator.NodeAnnotation(neg.node()).suffix, "[30ps (+10ps)]");
}

TEST_F(DelayEstimatorTest, ProtoDelayEstimator) {
  estimator_model::CompiledEstimatorModel model_proto;
  QCHECK(google::protobuf::TextFormat::ParseFromString(
      R"pb(
        metric: DELAY_METRIC
        op_models {
          op: "kAdd"
          estimator {
            regression {
              constant: 10.0
              terms {
                expression { factor { source: RESULT_BIT_COUNT } }
                linear: 2.0
                log2: 0.0
              }
            }
          }
          specializations {
            kind: OPERANDS_IDENTICAL
            estimator { fixed: 5 }
          }
        }
        op_models {
          op: "kSub"
          estimator { alias_op: "kAdd" }
        }
        op_models {
          op: "kAndReduce"
          estimator { logical_effort { tau_in_ps: 10 } }
        }
        op_models {
          op: "kUMul"
          estimator {
            bounding_box {
              points {
                bounds {
                  factor { source: RESULT_BIT_COUNT }
                  max_value: 8
                }
                delay_ps: 40
              }
              points {
                bounds {
                  factor { source: RESULT_BIT_COUNT }
                  max_value: 16
                }
                delay_ps: 120
              }
            }
          }
        }
      )pb",
      &model_proto));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProtoDelayEstimator> estimator,
      ProtoDelayEstimator::Create("test_proto", model_proto));

  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue b = fb.Param("b", p->GetBitsType(8));
  BValue add_diff = fb.Add(a, b);
  BValue add_same = fb.Add(a, a);
  BValue sub_diff = fb.Subtract(a, b);
  BValue sub_same = fb.Subtract(a, a);
  BValue and_reduce = fb.AndReduce(fb.Param("c", p->GetBitsType(10)));
  BValue umul_8 = fb.UMul(a, b, 8);
  BValue umul_32 = fb.UMul(a, b, 32);
  BValue neg = fb.Negate(a);
  XLS_ASSERT_OK(fb.Build().status());

  EXPECT_THAT(estimator->GetOperationDelayInPs(add_diff.node()),
              IsOkAndHolds(26));
  EXPECT_THAT(estimator->GetOperationDelayInPs(add_same.node()),
              IsOkAndHolds(5));
  EXPECT_THAT(estimator->GetOperationDelayInPs(sub_diff.node()),
              IsOkAndHolds(26));
  EXPECT_THAT(estimator->GetOperationDelayInPs(sub_same.node()),
              IsOkAndHolds(5));
  EXPECT_THAT(estimator->GetOperationDelayInPs(and_reduce.node()),
              IsOkAndHolds(50));
  EXPECT_THAT(estimator->GetOperationDelayInPs(umul_8.node()),
              IsOkAndHolds(40));
  EXPECT_THAT(estimator->GetOperationDelayInPs(umul_32.node()),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(estimator->GetOperationDelayInPs(neg.node()),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(DelayEstimatorTest, ProtoDelayEstimatorInvalidFactors) {
  estimator_model::CompiledEstimatorModel model_proto;
  QCHECK(google::protobuf::TextFormat::ParseFromString(
      R"pb(
        metric: DELAY_METRIC
        op_models {
          op: "kAdd"
          estimator {
            regression {
              constant: 1.0
              terms {
                expression {
                  factor { source: OPERAND_BIT_COUNT operand_number: 5 }
                }
                linear: 1.0
              }
            }
          }
        }
        op_models {
          op: "kSub"
          estimator {
            regression {
              constant: 1.0
              terms {
                expression {
                  factor { source: OPERAND_ELEMENT_COUNT operand_number: 0 }
                }
                linear: 1.0
              }
            }
          }
        }
        op_models {
          op: "kUMul"
          estimator {
            regression {
              constant: 1.0
              terms {
                expression {
                  factor { source: OPERAND_ELEMENT_BIT_COUNT operand_number: 0 }
                }
                linear: 1.0
              }
            }
          }
        }
      )pb",
      &model_proto));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProtoDelayEstimator> estimator,
      ProtoDelayEstimator::Create("test_invalid_factors", model_proto));

  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue b = fb.Param("b", p->GetBitsType(8));
  BValue add = fb.Add(a, b);
  BValue sub = fb.Subtract(a, b);
  BValue umul = fb.UMul(a, b, 8);
  XLS_ASSERT_OK(fb.Build().status());

  EXPECT_THAT(estimator->GetOperationDelayInPs(add.node()),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(estimator->GetOperationDelayInPs(sub.node()),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(estimator->GetOperationDelayInPs(umul.node()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace xls
