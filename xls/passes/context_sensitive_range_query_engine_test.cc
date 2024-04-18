// Copyright 2023 The XLS Authors
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

#include "xls/passes/context_sensitive_range_query_engine.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/predicate_state.h"
#include "xls/passes/range_query_engine.h"

namespace xls {
namespace {
using testing::AnyOf;
using testing::Eq;

class ContextSensitiveRangeQueryEngineTest : public IrTestBase {
 public:
  static constexpr PredicateState::ArmT kConsequentArm = 1;
  static constexpr PredicateState::ArmT kAlternateArm = 0;
  static const Bits kTrue;
  static const Bits kFalse;
};
const Bits ContextSensitiveRangeQueryEngineTest::kFalse = Bits(1);
const Bits ContextSensitiveRangeQueryEngineTest::kTrue = Bits::AllOnes(1);

enum class Signedness : int8_t {
  kSigned,
  kUnsigned,
};
template <typename Sink>
void AbslStringify(Sink& sink, const Signedness& src) {
  switch (src) {
    case Signedness::kSigned:
      sink.Append("kSigned");
      break;
    case Signedness::kUnsigned:
      sink.Append("kUnsigned");
      break;
  }
}

class BaseSignedContextSensitiveRangeQueryEngineTest {
 public:
  virtual ~BaseSignedContextSensitiveRangeQueryEngineTest() = default;
  virtual bool IsSigned() const = 0;
  Bits MinValue(int64_t bits) const {
    return IsSigned() ? Bits::MinSigned(bits) : Bits(bits);
  }
  Bits MaxValue(int64_t bits) const {
    return IsSigned() ? Bits::MaxSigned(bits) : Bits::AllOnes(bits);
  }
  BValue Gt(FunctionBuilder& fb, BValue l, BValue r) {
    if (IsSigned()) {
      return fb.SGt(l, r);
    }
    return fb.UGt(l, r);
  }
  BValue Lt(FunctionBuilder& fb, BValue l, BValue r) {
    if (IsSigned()) {
      return fb.SLt(l, r);
    }
    return fb.ULt(l, r);
  }
  BValue Ge(FunctionBuilder& fb, BValue l, BValue r) {
    if (IsSigned()) {
      return fb.SGe(l, r);
    }
    return fb.UGe(l, r);
  }
  BValue Le(FunctionBuilder& fb, BValue l, BValue r) {
    if (IsSigned()) {
      return fb.SLe(l, r);
    }
    return fb.ULe(l, r);
  }
};

class SignedContextSensitiveRangeQueryEngineTest
    : public ContextSensitiveRangeQueryEngineTest,
      public BaseSignedContextSensitiveRangeQueryEngineTest,
      public testing::WithParamInterface<Signedness> {
 public:
  bool IsSigned() const final { return GetParam() == Signedness::kSigned; }
};

enum class AndOrder : int8_t {
  kLeftFirst,
  kRightFirst,
};

enum class ComparisonType : int8_t {
  kOpen,
  kClosed,
};
template <typename Sink>
void AbslStringify(Sink& sink, const ComparisonType& src) {
  switch (src) {
    case ComparisonType::kOpen:
      sink.Append("kOpen");
      break;
    case ComparisonType::kClosed:
      sink.Append("kClosed");
      break;
  }
}

enum class ComparisonOrder : int8_t {
  kParamFirst,
  kLiteralFirst,
};

template <typename Sink>
void AbslStringify(Sink& sink, const ComparisonOrder& src) {
  switch (src) {
    case ComparisonOrder::kParamFirst:
      sink.Append("kParamFirst");
      break;
    case ComparisonOrder::kLiteralFirst:
      sink.Append("kLiteralFirst");
      break;
  }
}

struct SignedRangeComparison {
  Signedness sign;
  AndOrder order;
  ComparisonType left_cmp_type;
  ComparisonOrder left_cmp_order;
  ComparisonType right_cmp_type;
  ComparisonOrder right_cmp_order;

  using Tuple = std::tuple<Signedness, AndOrder, ComparisonType,
                           ComparisonOrder, ComparisonType, ComparisonOrder>;
  SignedRangeComparison(Tuple vals) {  // NOLINT: explicit
    std::tie(sign, order, left_cmp_type, left_cmp_order, right_cmp_type,
             right_cmp_order) = vals;
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const SignedRangeComparison& src) {
    absl::Format(&sink, "%v_", src.sign);
    if (src.order == AndOrder::kLeftFirst) {
      absl::Format(&sink, "LEFT_%v_%v__AND__RIGHT_%v_%v", src.left_cmp_type,
                   src.left_cmp_order, src.right_cmp_type, src.right_cmp_order);
    } else {
      absl::Format(&sink, "RIGHT_%v_%v__AND__LEFT_%v_%v", src.right_cmp_type,
                   src.right_cmp_order, src.left_cmp_type, src.left_cmp_order);
    }
  }
};

template <typename Sink>
void AbslStringify(Sink& sink, const SignedRangeComparison::Tuple& src) {
  absl::Format(&sink, "%v", SignedRangeComparison(src));
}

class SignedRangeComparisonContextSensitiveRangeQueryEngineTest
    : public ContextSensitiveRangeQueryEngineTest,
      public BaseSignedContextSensitiveRangeQueryEngineTest,
      public testing::WithParamInterface<SignedRangeComparison> {
 public:
  bool IsSigned() const final { return GetParam().sign == Signedness::kSigned; }

  BValue RangeComparison(FunctionBuilder& fb, BValue left, BValue param,
                         BValue right) {
    BValue left_cmp = LeftComparison(fb, left, param);
    BValue right_cmp = RightComparison(fb, right, param);
    if (GetParam().order == AndOrder::kLeftFirst) {
      return fb.And({left_cmp, right_cmp});
    }
    return fb.And({right_cmp, left_cmp});
  }

  std::vector<Interval> ParamInterval(const Bits& left, const Bits& right) {
    SignedRangeComparison cmp = GetParam();
    if (cmp.left_cmp_type == ComparisonType::kOpen) {
      if (cmp.right_cmp_type == ComparisonType::kOpen) {
        return {Interval::Open(left, right)};
      }
      return {Interval::LeftOpen(left, right)};
    }
    if (cmp.right_cmp_type == ComparisonType::kOpen) {
      return {Interval::RightOpen(left, right)};
    }
    return {Interval::Closed(left, right)};
  }
  std::vector<Interval> InverseParamInterval(const Bits& left,
                                             const Bits& right) {
    IntervalSet set = IntervalSet::Of(ParamInterval(left, right));
    auto complement = IntervalSet::Complement(set);
    auto intervals = complement.Intervals();
    return std::vector<Interval>(intervals.begin(), intervals.end());
  }

 private:
  BValue LessCmp(FunctionBuilder& fb, BValue left, BValue right,
                 ComparisonType type) {
    if (type == ComparisonType::kOpen) {
      return Lt(fb, left, right);
    }
    return Le(fb, left, right);
  }
  BValue GreaterCmp(FunctionBuilder& fb, BValue left, BValue right,
                    ComparisonType type) {
    if (type == ComparisonType::kOpen) {
      return Gt(fb, left, right);
    }
    return Ge(fb, left, right);
  }
  BValue LeftComparison(FunctionBuilder& fb, BValue literal, BValue param) {
    ComparisonType cmp = GetParam().left_cmp_type;
    if (GetParam().left_cmp_order == ComparisonOrder::kParamFirst) {
      return GreaterCmp(fb, param, literal, cmp);
    }
    return LessCmp(fb, literal, param, cmp);
  }
  BValue RightComparison(FunctionBuilder& fb, BValue literal, BValue param) {
    ComparisonType cmp = GetParam().right_cmp_type;
    if (GetParam().right_cmp_order == ComparisonOrder::kParamFirst) {
      return LessCmp(fb, param, literal, cmp);
    }
    return GreaterCmp(fb, literal, param, cmp);
  }
};

LeafTypeTree<IntervalSet> BitsLTT(Node* node,
                                  absl::Span<const Interval> intervals) {
  CHECK(!intervals.empty());
  IntervalSet interval_set = IntervalSet::Of(intervals);
  CHECK(node->GetType()->IsBits());
  LeafTypeTree<IntervalSet> result(node->GetType());
  result.Set({}, interval_set);
  return result;
}

TEST_F(ContextSensitiveRangeQueryEngineTest, Eq) {
  Bits max_bits = UBits(12, 8);
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // if (x == 12) { x + 10 } else { x }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue cond = fb.Eq(x, fb.Literal(UBits(12, 8)));
  BValue add_ten = fb.Add(x, fb.Literal(UBits(10, 8)));
  BValue res = fb.Select(cond, {x, add_ten});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree x_ist = BitsLTT(x.node(), {Interval::Precise(UBits(12, 8))});
  IntervalSetTree x_ist_global = BitsLTT(x.node(), {Interval::Maximal(8)});

  IntervalSetTree add_ten_ist =
      BitsLTT(x.node(), {Interval::Precise(UBits(22, 8))});
  IntervalSetTree add_ten_ist_global =
      BitsLTT(x.node(), {Interval::Maximal(8)});
  IntervalSetTree cond_ist = BitsLTT(cond.node(), {Interval::Maximal(1)});
  IntervalSetTree res_ist = BitsLTT(res.node(), {Interval::Maximal(8)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_ist_global);
  EXPECT_EQ(engine.GetIntervals(add_ten.node()), add_ten_ist_global);
  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(engine.GetIntervals(res.node()), res_ist);

  EXPECT_EQ(consequent_arm_range->GetIntervals(x.node()), x_ist);
  EXPECT_EQ(consequent_arm_range->GetIntervals(add_ten.node()), add_ten_ist);

  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(kFalse)}));

  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(kTrue)}));
  // NB There is a restricted value for res given cond == 0 but its not clear
  // that we actually want to bother to calculate it. Instead just verify that
  // the result is less than or equal to the unconstrained case.
  EXPECT_THAT(consequent_arm_range->GetIntervals(res.node()),
              AnyOf(Eq(x_ist), Eq(res_ist)));
}

TEST_F(ContextSensitiveRangeQueryEngineTest, Ne) {
  Bits max_bits = UBits(12, 8);
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // if (x != 12) { x } else { x + 10 }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue cond = fb.Ne(x, fb.Literal(UBits(12, 8)));
  BValue add_ten = fb.Add(x, fb.Literal(UBits(10, 8)));
  BValue res = fb.Select(cond, {add_ten, x});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree x_ist = BitsLTT(x.node(), {Interval::Precise(UBits(12, 8))});
  IntervalSetTree x_ist_global = BitsLTT(x.node(), {Interval::Maximal(8)});

  IntervalSetTree add_ten_ist =
      BitsLTT(x.node(), {Interval::Precise(UBits(22, 8))});
  IntervalSetTree add_ten_ist_global =
      BitsLTT(x.node(), {Interval::Maximal(8)});
  IntervalSetTree cond_ist = BitsLTT(cond.node(), {Interval::Maximal(1)});
  IntervalSetTree res_ist = BitsLTT(res.node(), {Interval::Maximal(8)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_ist_global);
  EXPECT_EQ(engine.GetIntervals(add_ten.node()), add_ten_ist_global);
  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(engine.GetIntervals(res.node()), res_ist);

  EXPECT_EQ(alternate_arm_range->GetIntervals(x.node()), x_ist);

  EXPECT_EQ(alternate_arm_range->GetIntervals(add_ten.node()), add_ten_ist);

  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(kFalse)}));

  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(kTrue)}));
  // NB There is a restricted value for res given cond == 0 but its not clear
  // that we actually want to bother to calculate it. Instead just verify that
  // the result is less than or equal to the unconstrained case.
  EXPECT_THAT(alternate_arm_range->GetIntervals(res.node()),
              AnyOf(Eq(x_ist), Eq(res_ist)));
}

TEST_F(ContextSensitiveRangeQueryEngineTest, DeadBranchEq) {
  Bits max_bits = UBits(12, 8);
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // x: u2; if ((x as u8) == 12) { (x as u8) + 12 } else { (x as u8) }
  BValue x = fb.Param("x", p->GetBitsType(2));
  BValue u8_x = fb.ZeroExtend(x, 8);
  BValue cond = fb.Eq(u8_x, fb.Literal(UBits(12, 8)));
  BValue add_ten = fb.Add(u8_x, fb.Literal(UBits(10, 8)));
  BValue res = fb.Select(cond, {u8_x, add_ten});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree cond_ist =
      BitsLTT(cond.node(), {Interval::Precise(UBits(0, 1))});

  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(UBits(1, 1))}));
  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()), cond_ist);
}

TEST_F(ContextSensitiveRangeQueryEngineTest, DeadBranchNe) {
  Bits max_bits = UBits(12, 8);
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // x: u2; if ((x as u8) != 12) { (x as u8) + 12 } else { (x as u8) }
  BValue x = fb.Param("x", p->GetBitsType(2));
  BValue u8_x = fb.ZeroExtend(x, 8);
  BValue cond = fb.Ne(u8_x, fb.Literal(UBits(12, 8)));
  BValue add_ten = fb.Add(u8_x, fb.Literal(UBits(10, 8)));
  BValue res = fb.Select(cond, {u8_x, add_ten});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree cond_ist =
      BitsLTT(cond.node(), {Interval::Precise(UBits(1, 1))});

  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(UBits(0, 1))}));
  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()), cond_ist);
}

TEST_F(ContextSensitiveRangeQueryEngineTest, DeadBranchCmp) {
  Bits max_bits = UBits(12, 8);
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // x: u2; if ((x as u8) > 12) { (x as u8) + 12 } else { (x as u8) }
  BValue x = fb.Param("x", p->GetBitsType(2));
  BValue u8_x = fb.ZeroExtend(x, 8);
  BValue cond = fb.UGt(u8_x, fb.Literal(UBits(12, 8)));
  BValue add_ten = fb.Add(u8_x, fb.Literal(UBits(10, 8)));
  BValue res = fb.Select(cond, {u8_x, add_ten});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree cond_ist =
      BitsLTT(cond.node(), {Interval::Precise(UBits(0, 1))});
  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(UBits(1, 1))}));
}

TEST_P(SignedContextSensitiveRangeQueryEngineTest, VariableLtConstantUseInIf) {
  Bits max_bits = bits_ops::Sub(UBits(12, 8), UBits(1, 8));
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // if (x < 12) { x } else { y }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue cond = Lt(fb, x, fb.Literal(UBits(12, 8)));
  BValue res = fb.Select(cond, {y, x});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree x_ist = BitsLTT(x.node(), {Interval(MinValue(8), max_bits)});
  IntervalSetTree x_ist_global = BitsLTT(x.node(), {Interval::Maximal(8)});

  IntervalSetTree y_ist = BitsLTT(y.node(), {Interval::Maximal(8)});
  IntervalSetTree cond_ist = BitsLTT(cond.node(), {Interval::Maximal(1)});
  IntervalSetTree res_ist = BitsLTT(res.node(), {Interval::Maximal(8)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_ist_global);
  EXPECT_EQ(engine.GetIntervals(y.node()), y_ist);
  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(engine.GetIntervals(res.node()), res_ist);

  EXPECT_EQ(consequent_arm_range->GetIntervals(x.node()), x_ist);

  // NB Result is same as base since the conditional values can't affect the
  // arm so no need to compute them.
  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()),
            engine.GetIntervals(cond.node()));

  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(kTrue)}));
  // NB There is a restricted value for res given cond == 0 but its not clear
  // that we actually want to bother to calculate it. Instead just verify that
  // the result is less than or equal to the unconstrained case.
  EXPECT_THAT(consequent_arm_range->GetIntervals(res.node()),
              AnyOf(Eq(x_ist), Eq(res_ist)));
}

TEST_P(SignedContextSensitiveRangeQueryEngineTest,
       VariableLtConstantUseInElse) {
  Bits max_bits = UBits(12, 8);
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // if (x < 12) { y } else { x }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue cond = Lt(fb, x, fb.Literal(UBits(12, 8)));
  BValue res = fb.Select(cond, {x, y});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RecordProperty("graph", f->DumpIr());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree x_ist = BitsLTT(x.node(), {Interval(max_bits, MaxValue(8))});
  IntervalSetTree x_ist_global = BitsLTT(x.node(), {Interval::Maximal(8)});

  IntervalSetTree y_ist = BitsLTT(y.node(), {Interval::Maximal(8)});
  IntervalSetTree cond_ist = BitsLTT(cond.node(), {Interval::Maximal(1)});
  IntervalSetTree res_ist = BitsLTT(res.node(), {Interval::Maximal(8)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_ist_global);
  EXPECT_EQ(engine.GetIntervals(y.node()), y_ist);
  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(engine.GetIntervals(res.node()), res_ist);

  EXPECT_EQ(alternate_arm_range->GetIntervals(x.node()), x_ist);

  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(kFalse)}));

  // NB Result is same as base since the conditional values can't affect the
  // arm so no need to compute them.
  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()),
            engine.GetIntervals(cond.node()));
  // NB There is a restricted value for res given cond == 0 but its not clear
  // that we actually want to bother to calculate it. Instead just verify that
  // the result is less than or equal to the unconstrained case.
  EXPECT_THAT(alternate_arm_range->GetIntervals(res.node()),
              AnyOf(Eq(x_ist), Eq(res_ist)));
}

TEST_P(SignedContextSensitiveRangeQueryEngineTest, VariableLeConstantUseInIf) {
  Bits max_bits = UBits(12, 8);
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // if (x <= 12) { x } else { y }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue cond = Le(fb, x, fb.Literal(UBits(12, 8)));
  BValue res = fb.Select(cond, {y, x});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree x_ist = BitsLTT(x.node(), {Interval(MinValue(8), max_bits)});
  IntervalSetTree x_ist_global = BitsLTT(x.node(), {Interval::Maximal(8)});

  IntervalSetTree y_ist = BitsLTT(y.node(), {Interval::Maximal(8)});
  IntervalSetTree cond_ist = BitsLTT(cond.node(), {Interval::Maximal(1)});
  IntervalSetTree res_ist = BitsLTT(res.node(), {Interval::Maximal(8)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_ist_global);
  EXPECT_EQ(engine.GetIntervals(y.node()), y_ist);
  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(engine.GetIntervals(res.node()), res_ist);

  EXPECT_EQ(consequent_arm_range->GetIntervals(x.node()), x_ist);

  // NB Result is same as base since the conditional values can't affect the
  // arm so no need to compute them.
  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()),
            engine.GetIntervals(cond.node()));

  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(kTrue)}));
  // NB There is a restricted value for res given cond == 0 but its not clear
  // that we actually want to bother to calculate it. Instead just verify that
  // the result is less than or equal to the unconstrained case.
  EXPECT_THAT(consequent_arm_range->GetIntervals(res.node()),
              AnyOf(Eq(x_ist), Eq(res_ist)));
}

TEST_P(SignedContextSensitiveRangeQueryEngineTest,
       VariableLeConstantUseInElse) {
  Bits max_bits = bits_ops::Add(UBits(12, 8), UBits(1, 8));
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // if (x <= 12) { y } else { x }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue cond = Le(fb, x, fb.Literal(UBits(12, 8)));
  BValue res = fb.Select(cond, {x, y});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree x_ist = BitsLTT(x.node(), {Interval(max_bits, MaxValue(8))});
  IntervalSetTree x_ist_global = BitsLTT(x.node(), {Interval::Maximal(8)});

  IntervalSetTree y_ist = BitsLTT(y.node(), {Interval::Maximal(8)});
  IntervalSetTree cond_ist = BitsLTT(cond.node(), {Interval::Maximal(1)});
  IntervalSetTree res_ist = BitsLTT(res.node(), {Interval::Maximal(8)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_ist_global);
  EXPECT_EQ(engine.GetIntervals(y.node()), y_ist);
  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(engine.GetIntervals(res.node()), res_ist);

  EXPECT_EQ(alternate_arm_range->GetIntervals(x.node()), x_ist);

  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(kFalse)}));

  // NB Result is same as base since the conditional values can't affect the
  // arm so no need to compute them.
  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()),
            engine.GetIntervals(cond.node()));
  // NB There is a restricted value for res given cond == 0 but its not clear
  // that we actually want to bother to calculate it. Instead just verify that
  // the result is less than or equal to the unconstrained case.
  EXPECT_THAT(alternate_arm_range->GetIntervals(res.node()),
              AnyOf(Eq(x_ist), Eq(res_ist)));
}

TEST_P(SignedContextSensitiveRangeQueryEngineTest, VariableGtConstantUseInIf) {
  Bits max_bits = bits_ops::Add(UBits(12, 8), UBits(1, 8));
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // if (x > 12) { x } else { y }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue cond = Gt(fb, x, fb.Literal(UBits(12, 8)));
  BValue res = fb.Select(cond, {y, x});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree x_ist = BitsLTT(x.node(), {Interval(max_bits, MaxValue(8))});
  IntervalSetTree x_ist_global = BitsLTT(x.node(), {Interval::Maximal(8)});

  IntervalSetTree y_ist = BitsLTT(y.node(), {Interval::Maximal(8)});
  IntervalSetTree cond_ist = BitsLTT(cond.node(), {Interval::Maximal(1)});
  IntervalSetTree res_ist = BitsLTT(res.node(), {Interval::Maximal(8)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_ist_global);
  EXPECT_EQ(engine.GetIntervals(y.node()), y_ist);
  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(engine.GetIntervals(res.node()), res_ist);

  EXPECT_EQ(consequent_arm_range->GetIntervals(x.node()), x_ist);

  // NB Result is same as base since the conditional values can't affect the
  // arm so no need to compute them.
  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()),
            engine.GetIntervals(cond.node()));

  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(UBits(1, 1))}));
  // NB There is a restricted value for res given cond == 0 but its not clear
  // that we actually want to bother to calculate it. Instead just verify that
  // the result is less than or equal to the unconstrained case.
  EXPECT_THAT(consequent_arm_range->GetIntervals(res.node()),
              AnyOf(Eq(x_ist), Eq(res_ist)));
}

TEST_P(SignedContextSensitiveRangeQueryEngineTest,
       VariableGtConstantUseInElse) {
  Bits max_bits = UBits(12, 8);
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // if (x > 12) { y } else { x }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue cond = Gt(fb, x, fb.Literal(UBits(12, 8)));
  BValue res = fb.Select(cond, {x, y});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree x_ist = BitsLTT(x.node(), {Interval(MinValue(8), max_bits)});
  IntervalSetTree x_ist_global = BitsLTT(x.node(), {Interval::Maximal(8)});

  IntervalSetTree y_ist = BitsLTT(y.node(), {Interval::Maximal(8)});
  IntervalSetTree cond_ist = BitsLTT(cond.node(), {Interval::Maximal(1)});
  IntervalSetTree res_ist = BitsLTT(res.node(), {Interval::Maximal(8)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_ist_global);
  EXPECT_EQ(engine.GetIntervals(y.node()), y_ist);
  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(engine.GetIntervals(res.node()), res_ist);

  EXPECT_EQ(alternate_arm_range->GetIntervals(x.node()), x_ist);

  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(UBits(0, 1))}));

  // NB Result is same as base since the conditional values can't affect the
  // arm so no need to compute them.
  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()),
            engine.GetIntervals(cond.node()));
  // NB There is a restricted value for res given cond == 0 but its not clear
  // that we actually want to bother to calculate it. Instead just verify that
  // the result is less than or equal to the unconstrained case.
  EXPECT_THAT(alternate_arm_range->GetIntervals(res.node()),
              AnyOf(Eq(x_ist), Eq(res_ist)));
}

TEST_P(SignedContextSensitiveRangeQueryEngineTest, VariableGeConstantUseInIf) {
  Bits max_bits = UBits(12, 8);
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // if (x >= 12) { x } else { y }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue cond = Ge(fb, x, fb.Literal(UBits(12, 8)));
  BValue res = fb.Select(cond, {y, x});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree x_ist = BitsLTT(x.node(), {Interval(max_bits, MaxValue(8))});
  IntervalSetTree x_ist_global = BitsLTT(x.node(), {Interval::Maximal(8)});

  IntervalSetTree y_ist = BitsLTT(y.node(), {Interval::Maximal(8)});
  IntervalSetTree cond_ist = BitsLTT(cond.node(), {Interval::Maximal(1)});
  IntervalSetTree res_ist = BitsLTT(res.node(), {Interval::Maximal(8)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_ist_global);
  EXPECT_EQ(engine.GetIntervals(y.node()), y_ist);
  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(engine.GetIntervals(res.node()), res_ist);

  EXPECT_EQ(consequent_arm_range->GetIntervals(x.node()), x_ist);

  // NB Result is same as base since the conditional values can't affect the
  // arm so no need to compute them.
  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()),
            engine.GetIntervals(cond.node()));

  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(UBits(1, 1))}));
  // NB There is a restricted value for res given cond == 0 but its not clear
  // that we actually want to bother to calculate it. Instead just verify that
  // the result is less than or equal to the unconstrained case.
  EXPECT_THAT(consequent_arm_range->GetIntervals(res.node()),
              AnyOf(Eq(x_ist), Eq(res_ist)));
}

TEST_P(SignedContextSensitiveRangeQueryEngineTest,
       VariableGeConstantUseInElse) {
  Bits max_bits = bits_ops::Sub(UBits(12, 8), UBits(1, 8));
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // if (x >= 12) { y } else { x }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue cond = Ge(fb, x, fb.Literal(UBits(12, 8)));
  BValue res = fb.Select(cond, {x, y});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree x_ist = BitsLTT(x.node(), {Interval(MinValue(8), max_bits)});
  IntervalSetTree x_ist_global = BitsLTT(x.node(), {Interval::Maximal(8)});

  IntervalSetTree y_ist = BitsLTT(y.node(), {Interval::Maximal(8)});
  IntervalSetTree cond_ist = BitsLTT(cond.node(), {Interval::Maximal(1)});
  IntervalSetTree res_ist = BitsLTT(res.node(), {Interval::Maximal(8)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_ist_global);
  EXPECT_EQ(engine.GetIntervals(y.node()), y_ist);
  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(engine.GetIntervals(res.node()), res_ist);

  EXPECT_EQ(alternate_arm_range->GetIntervals(x.node()), x_ist);

  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(UBits(0, 1))}));

  // NB Result is same as base since the conditional values can't affect the
  // arm so no need to compute them.
  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()),
            engine.GetIntervals(cond.node()));
  // NB There is a restricted value for res given cond == 0 but its not clear
  // that we actually want to bother to calculate it. Instead just verify that
  // the result is less than or equal to the unconstrained case.
  EXPECT_THAT(alternate_arm_range->GetIntervals(res.node()),
              AnyOf(Eq(x_ist), Eq(res_ist)));
}

TEST_P(SignedContextSensitiveRangeQueryEngineTest, ConstantLtVariableUseInIf) {
  Bits max_bits = bits_ops::Add(UBits(12, 8), UBits(1, 8));
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // if (12 < x) { x } else { y }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue cond = Lt(fb, fb.Literal(UBits(12, 8)), x);
  BValue res = fb.Select(cond, {y, x});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree x_ist = BitsLTT(x.node(), {Interval(max_bits, MaxValue(8))});
  IntervalSetTree x_ist_global = BitsLTT(x.node(), {Interval::Maximal(8)});

  IntervalSetTree y_ist = BitsLTT(y.node(), {Interval::Maximal(8)});
  IntervalSetTree cond_ist = BitsLTT(cond.node(), {Interval::Maximal(1)});
  IntervalSetTree res_ist = BitsLTT(res.node(), {Interval::Maximal(8)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_ist_global);
  EXPECT_EQ(engine.GetIntervals(y.node()), y_ist);
  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(engine.GetIntervals(res.node()), res_ist);

  EXPECT_EQ(consequent_arm_range->GetIntervals(x.node()), x_ist);

  // NB Result is same as base since the conditional values can't affect this
  // arm so no need to compute them.
  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()),
            engine.GetIntervals(cond.node()));

  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(kTrue)}));
  // NB There is a restricted value for res given cond == 0 but its not clear
  // that we actually want to bother to calculate it. Instead just verify that
  // the result is less than or equal to the unconstrained case.
  EXPECT_THAT(consequent_arm_range->GetIntervals(res.node()),
              AnyOf(Eq(x_ist), Eq(res_ist)));
}

TEST_P(SignedContextSensitiveRangeQueryEngineTest,
       ConstantLtVariableUseInElse) {
  Bits max_bits = UBits(12, 8);
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // if (12 < x) { y } else { x }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue cond = Lt(fb, fb.Literal(UBits(12, 8)), x);
  BValue res = fb.Select(cond, {x, y});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RecordProperty("graph", f->DumpIr());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree x_ist = BitsLTT(x.node(), {Interval(MinValue(8), max_bits)});
  IntervalSetTree x_ist_global = BitsLTT(x.node(), {Interval::Maximal(8)});

  IntervalSetTree y_ist = BitsLTT(y.node(), {Interval::Maximal(8)});
  IntervalSetTree cond_ist = BitsLTT(cond.node(), {Interval::Maximal(1)});
  IntervalSetTree res_ist = BitsLTT(res.node(), {Interval::Maximal(8)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_ist_global);
  EXPECT_EQ(engine.GetIntervals(y.node()), y_ist);
  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(engine.GetIntervals(res.node()), res_ist);

  EXPECT_EQ(alternate_arm_range->GetIntervals(x.node()), x_ist);

  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(kFalse)}));

  // NB Result is same as base since the conditional values can't affect the
  // arm so no need to compute them.
  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()),
            engine.GetIntervals(cond.node()));
  // NB There is a restricted value for res given cond == 0 but its not clear
  // that we actually want to bother to calculate it. Instead just verify that
  // the result is less than or equal to the unconstrained case.
  EXPECT_THAT(alternate_arm_range->GetIntervals(res.node()),
              AnyOf(Eq(x_ist), Eq(res_ist)));
}

TEST_P(SignedContextSensitiveRangeQueryEngineTest, ConstantLeVariableUseInIf) {
  Bits max_bits = UBits(12, 8);
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // if (12 <= x) { x } else { y }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue cond = Le(fb, fb.Literal(UBits(12, 8)), x);
  BValue res = fb.Select(cond, {y, x});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree x_ist = BitsLTT(x.node(), {Interval(max_bits, MaxValue(8))});
  IntervalSetTree x_ist_global = BitsLTT(x.node(), {Interval::Maximal(8)});

  IntervalSetTree y_ist = BitsLTT(y.node(), {Interval::Maximal(8)});
  IntervalSetTree cond_ist = BitsLTT(cond.node(), {Interval::Maximal(1)});
  IntervalSetTree res_ist = BitsLTT(res.node(), {Interval::Maximal(8)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_ist_global);
  EXPECT_EQ(engine.GetIntervals(y.node()), y_ist);
  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(engine.GetIntervals(res.node()), res_ist);

  EXPECT_EQ(consequent_arm_range->GetIntervals(x.node()), x_ist);

  // NB Result is same as base since the conditional values can't affect the
  // arm so no need to compute them.
  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()),
            engine.GetIntervals(cond.node()));

  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(kTrue)}));
  // NB There is a restricted value for res given cond == 0 but its not clear
  // that we actually want to bother to calculate it. Instead just verify that
  // the result is less than or equal to the unconstrained case.
  EXPECT_THAT(consequent_arm_range->GetIntervals(res.node()),
              AnyOf(Eq(x_ist), Eq(res_ist)));
}

TEST_P(SignedContextSensitiveRangeQueryEngineTest,
       ConstantLeVariableUseInElse) {
  Bits max_bits = bits_ops::Sub(UBits(12, 8), UBits(1, 8));
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // if (12 <= x) { y } else { x }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue cond = Le(fb, fb.Literal(UBits(12, 8)), x);
  BValue res = fb.Select(cond, {x, y});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree x_ist = BitsLTT(x.node(), {Interval(MinValue(8), max_bits)});
  IntervalSetTree x_ist_global = BitsLTT(x.node(), {Interval::Maximal(8)});

  IntervalSetTree y_ist = BitsLTT(y.node(), {Interval::Maximal(8)});
  IntervalSetTree cond_ist = BitsLTT(cond.node(), {Interval::Maximal(1)});
  IntervalSetTree res_ist = BitsLTT(res.node(), {Interval::Maximal(8)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_ist_global);
  EXPECT_EQ(engine.GetIntervals(y.node()), y_ist);
  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(engine.GetIntervals(res.node()), res_ist);

  EXPECT_EQ(alternate_arm_range->GetIntervals(x.node()), x_ist);

  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(kFalse)}));

  // NB Result is same as base since the conditional values can't affect the
  // arm so no need to compute them.
  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()),
            engine.GetIntervals(cond.node()));
  // NB There is a restricted value for res given cond == 0 but its not clear
  // that we actually want to bother to calculate it. Instead just verify that
  // the result is less than or equal to the unconstrained case.
  EXPECT_THAT(alternate_arm_range->GetIntervals(res.node()),
              AnyOf(Eq(x_ist), Eq(res_ist)));
}

TEST_P(SignedContextSensitiveRangeQueryEngineTest, ConstantGtVariableUseInIf) {
  Bits max_bits = bits_ops::Sub(UBits(12, 8), UBits(1, 8));
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // if (12 > x) { x } else { y }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue cond = Gt(fb, fb.Literal(UBits(12, 8)), x);
  BValue res = fb.Select(cond, {y, x});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree x_ist = BitsLTT(x.node(), {Interval(MinValue(8), max_bits)});
  IntervalSetTree x_ist_global = BitsLTT(x.node(), {Interval::Maximal(8)});

  IntervalSetTree y_ist = BitsLTT(y.node(), {Interval::Maximal(8)});
  IntervalSetTree cond_ist = BitsLTT(cond.node(), {Interval::Maximal(1)});
  IntervalSetTree res_ist = BitsLTT(res.node(), {Interval::Maximal(8)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_ist_global);
  EXPECT_EQ(engine.GetIntervals(y.node()), y_ist);
  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(engine.GetIntervals(res.node()), res_ist);

  EXPECT_EQ(consequent_arm_range->GetIntervals(x.node()), x_ist);

  // NB Result is same as base since the conditional values can't affect the
  // arm so no need to compute them.
  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()),
            engine.GetIntervals(cond.node()));

  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(UBits(1, 1))}));
  // NB There is a restricted value for res given cond == 0 but its not clear
  // that we actually want to bother to calculate it. Instead just verify that
  // the result is less than or equal to the unconstrained case.
  EXPECT_THAT(consequent_arm_range->GetIntervals(res.node()),
              AnyOf(Eq(x_ist), Eq(res_ist)));
}

TEST_P(SignedContextSensitiveRangeQueryEngineTest,
       ConstantGtVariableUseInElse) {
  Bits max_bits = UBits(12, 8);
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // if (12 > x) { y } else { x }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue cond = Gt(fb, fb.Literal(UBits(12, 8)), x);
  BValue res = fb.Select(cond, {x, y});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree x_ist = BitsLTT(x.node(), {Interval(max_bits, MaxValue(8))});
  IntervalSetTree x_ist_global = BitsLTT(x.node(), {Interval::Maximal(8)});

  IntervalSetTree y_ist = BitsLTT(y.node(), {Interval::Maximal(8)});
  IntervalSetTree cond_ist = BitsLTT(cond.node(), {Interval::Maximal(1)});
  IntervalSetTree res_ist = BitsLTT(res.node(), {Interval::Maximal(8)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_ist_global);
  EXPECT_EQ(engine.GetIntervals(y.node()), y_ist);
  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(engine.GetIntervals(res.node()), res_ist);

  EXPECT_EQ(alternate_arm_range->GetIntervals(x.node()), x_ist);

  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(UBits(0, 1))}));

  // NB Result is same as base since the conditional values can't affect the
  // arm so no need to compute them.
  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()),
            engine.GetIntervals(cond.node()));
  // NB There is a restricted value for res given cond == 0 but its not clear
  // that we actually want to bother to calculate it. Instead just verify that
  // the result is less than or equal to the unconstrained case.
  EXPECT_THAT(alternate_arm_range->GetIntervals(res.node()),
              AnyOf(Eq(x_ist), Eq(res_ist)));
}

TEST_P(SignedContextSensitiveRangeQueryEngineTest, ConstantGeVariableUseInIf) {
  Bits max_bits = UBits(12, 8);
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // if (12 >= x) { x } else { y }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue cond = Ge(fb, fb.Literal(UBits(12, 8)), x);
  BValue res = fb.Select(cond, {y, x});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree x_ist = BitsLTT(x.node(), {Interval(MinValue(8), max_bits)});
  IntervalSetTree x_ist_global = BitsLTT(x.node(), {Interval::Maximal(8)});

  IntervalSetTree y_ist = BitsLTT(y.node(), {Interval::Maximal(8)});
  IntervalSetTree cond_ist = BitsLTT(cond.node(), {Interval::Maximal(1)});
  IntervalSetTree res_ist = BitsLTT(res.node(), {Interval::Maximal(8)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_ist_global);
  EXPECT_EQ(engine.GetIntervals(y.node()), y_ist);
  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(engine.GetIntervals(res.node()), res_ist);

  EXPECT_EQ(consequent_arm_range->GetIntervals(x.node()), x_ist);

  // NB Result is same as base since the conditional values can't affect the
  // arm so no need to compute them.
  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()),
            engine.GetIntervals(cond.node()));

  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(UBits(1, 1))}));
  // NB There is a restricted value for res given cond == 0 but its not clear
  // that we actually want to bother to calculate it. Instead just verify that
  // the result is less than or equal to the unconstrained case.
  EXPECT_THAT(consequent_arm_range->GetIntervals(res.node()),
              AnyOf(Eq(x_ist), Eq(res_ist)));
}

TEST_P(SignedContextSensitiveRangeQueryEngineTest,
       ConstantGeVariableUseInElse) {
  Bits max_bits = bits_ops::Add(UBits(12, 8), UBits(1, 8));
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // if (12 >= x) { y } else { x }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue cond = Ge(fb, fb.Literal(UBits(12, 8)), x);
  BValue res = fb.Select(cond, {x, y});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree x_ist = BitsLTT(x.node(), {Interval(max_bits, MaxValue(8))});
  IntervalSetTree x_ist_global = BitsLTT(x.node(), {Interval::Maximal(8)});

  IntervalSetTree y_ist = BitsLTT(y.node(), {Interval::Maximal(8)});
  IntervalSetTree cond_ist = BitsLTT(cond.node(), {Interval::Maximal(1)});
  IntervalSetTree res_ist = BitsLTT(res.node(), {Interval::Maximal(8)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_ist_global);
  EXPECT_EQ(engine.GetIntervals(y.node()), y_ist);
  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(engine.GetIntervals(res.node()), res_ist);

  EXPECT_EQ(alternate_arm_range->GetIntervals(x.node()), x_ist);

  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(UBits(0, 1))}));

  // NB Result is same as base since the conditional values can't affect the
  // arm so no need to compute them.
  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()),
            engine.GetIntervals(cond.node()));
  // NB There is a restricted value for res given cond == 0 but its not clear
  // that we actually want to bother to calculate it. Instead just verify that
  // the result is less than or equal to the unconstrained case.
  EXPECT_THAT(alternate_arm_range->GetIntervals(res.node()),
              AnyOf(Eq(x_ist), Eq(res_ist)));
}

TEST_F(ContextSensitiveRangeQueryEngineTest, OpenOpenRangeUseInIf) {
  Bits max_bits = UBits(12, 8);
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // if (4 < x && x < 12) { x } else { y }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue left = fb.ULt(fb.Literal(UBits(4, 8)), x);
  BValue right = fb.ULt(x, fb.Literal(UBits(12, 8)));
  BValue cond = fb.And({left, right});
  BValue res = fb.Select(cond, {y, x});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree x_ist =
      BitsLTT(x.node(), {Interval(UBits(5, 8), UBits(11, 8))});
  IntervalSetTree x_ist_global = BitsLTT(x.node(), {Interval::Maximal(8)});

  IntervalSetTree y_ist = BitsLTT(y.node(), {Interval::Maximal(8)});
  IntervalSetTree left_ist = BitsLTT(left.node(), {Interval::Maximal(1)});
  IntervalSetTree right_ist = BitsLTT(right.node(), {Interval::Maximal(1)});
  IntervalSetTree cond_ist = BitsLTT(cond.node(), {Interval::Maximal(1)});
  IntervalSetTree res_ist = BitsLTT(res.node(), {Interval::Maximal(8)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_ist_global);
  EXPECT_EQ(engine.GetIntervals(y.node()), y_ist);
  EXPECT_EQ(engine.GetIntervals(left.node()), left_ist);
  EXPECT_EQ(engine.GetIntervals(right.node()), right_ist);
  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(engine.GetIntervals(res.node()), res_ist);

  EXPECT_EQ(consequent_arm_range->GetIntervals(x.node()), x_ist);

  EXPECT_EQ(consequent_arm_range->GetIntervals(left.node()),
            BitsLTT(left.node(), {Interval::Precise(UBits(1, 1))}));
  EXPECT_EQ(consequent_arm_range->GetIntervals(right.node()),
            BitsLTT(right.node(), {Interval::Precise(UBits(1, 1))}));
  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(UBits(1, 1))}));

  EXPECT_EQ(alternate_arm_range->GetIntervals(left.node()),
            BitsLTT(left.node(), {Interval::Maximal(1)}));
  EXPECT_EQ(alternate_arm_range->GetIntervals(right.node()),
            BitsLTT(right.node(), {Interval::Maximal(1)}));
  // NB Result is same as base since the conditional values can't affect the
  // arm so no need to compute them.
  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()),
            engine.GetIntervals(cond.node()));
  // NB There is a restricted value for res given cond == 0 but its not clear
  // that we actually want to bother to calculate it. Instead just verify that
  // the result is less than or equal to the unconstrained case.
  EXPECT_THAT(consequent_arm_range->GetIntervals(res.node()),
              AnyOf(Eq(x_ist), Eq(res_ist)));
}

TEST_F(ContextSensitiveRangeQueryEngineTest, OpenClosedRangeUseInIf) {
  Bits max_bits = UBits(12, 8);
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // if (4 < x && x <= 12) { x } else { y }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue left = fb.ULt(fb.Literal(UBits(4, 8)), x);
  BValue right = fb.ULe(x, fb.Literal(UBits(12, 8)));
  BValue cond = fb.And({left, right});
  BValue res = fb.Select(cond, {y, x});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree x_ist =
      BitsLTT(x.node(), {Interval(UBits(5, 8), UBits(12, 8))});
  IntervalSetTree x_ist_global = BitsLTT(x.node(), {Interval::Maximal(8)});

  IntervalSetTree y_ist = BitsLTT(y.node(), {Interval::Maximal(8)});
  IntervalSetTree left_ist = BitsLTT(left.node(), {Interval::Maximal(1)});
  IntervalSetTree right_ist = BitsLTT(right.node(), {Interval::Maximal(1)});
  IntervalSetTree cond_ist = BitsLTT(cond.node(), {Interval::Maximal(1)});
  IntervalSetTree res_ist = BitsLTT(res.node(), {Interval::Maximal(8)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_ist_global);
  EXPECT_EQ(engine.GetIntervals(y.node()), y_ist);
  EXPECT_EQ(engine.GetIntervals(left.node()), left_ist);
  EXPECT_EQ(engine.GetIntervals(right.node()), right_ist);
  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(engine.GetIntervals(res.node()), res_ist);

  EXPECT_EQ(consequent_arm_range->GetIntervals(x.node()), x_ist);

  EXPECT_EQ(consequent_arm_range->GetIntervals(left.node()),
            BitsLTT(left.node(), {Interval::Precise(UBits(1, 1))}));
  EXPECT_EQ(consequent_arm_range->GetIntervals(right.node()),
            BitsLTT(right.node(), {Interval::Precise(UBits(1, 1))}));
  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(UBits(1, 1))}));

  EXPECT_EQ(alternate_arm_range->GetIntervals(left.node()),
            BitsLTT(left.node(), {Interval::Maximal(1)}));
  EXPECT_EQ(alternate_arm_range->GetIntervals(right.node()),
            BitsLTT(right.node(), {Interval::Maximal(1)}));
  // NB Result is same as base since the conditional values can't affect the
  // arm so no need to compute them.
  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()),
            engine.GetIntervals(cond.node()));
  // NB There is a restricted value for res given cond == 0 but its not clear
  // that we actually want to bother to calculate it. Instead just verify that
  // the result is less than or equal to the unconstrained case.
  EXPECT_THAT(consequent_arm_range->GetIntervals(res.node()),
              AnyOf(Eq(x_ist), Eq(res_ist)));
}

TEST_F(ContextSensitiveRangeQueryEngineTest, ClosedOpenRangeUseInIf) {
  Bits max_bits = UBits(12, 8);
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // if (4 <= x && x < 12) { x } else { y }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue left = fb.ULe(fb.Literal(UBits(4, 8)), x);
  BValue right = fb.ULt(x, fb.Literal(UBits(12, 8)));
  BValue cond = fb.And({left, right});
  BValue res = fb.Select(cond, {y, x});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree x_ist =
      BitsLTT(x.node(), {Interval(UBits(4, 8), UBits(11, 8))});
  IntervalSetTree x_ist_global = BitsLTT(x.node(), {Interval::Maximal(8)});

  IntervalSetTree y_ist = BitsLTT(y.node(), {Interval::Maximal(8)});
  IntervalSetTree left_ist = BitsLTT(left.node(), {Interval::Maximal(1)});
  IntervalSetTree right_ist = BitsLTT(right.node(), {Interval::Maximal(1)});
  IntervalSetTree cond_ist = BitsLTT(cond.node(), {Interval::Maximal(1)});
  IntervalSetTree res_ist = BitsLTT(res.node(), {Interval::Maximal(8)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_ist_global);
  EXPECT_EQ(engine.GetIntervals(y.node()), y_ist);
  EXPECT_EQ(engine.GetIntervals(left.node()), left_ist);
  EXPECT_EQ(engine.GetIntervals(right.node()), right_ist);
  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(engine.GetIntervals(res.node()), res_ist);

  EXPECT_EQ(consequent_arm_range->GetIntervals(x.node()), x_ist);

  EXPECT_EQ(consequent_arm_range->GetIntervals(left.node()),
            BitsLTT(left.node(), {Interval::Precise(UBits(1, 1))}));
  EXPECT_EQ(consequent_arm_range->GetIntervals(right.node()),
            BitsLTT(right.node(), {Interval::Precise(UBits(1, 1))}));
  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(UBits(1, 1))}));

  EXPECT_EQ(alternate_arm_range->GetIntervals(left.node()),
            BitsLTT(left.node(), {Interval::Maximal(1)}));
  EXPECT_EQ(alternate_arm_range->GetIntervals(right.node()),
            BitsLTT(right.node(), {Interval::Maximal(1)}));
  // NB Result is same as base since the conditional values can't affect the
  // arm so no need to compute them.
  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()),
            engine.GetIntervals(cond.node()));
  // NB There is a restricted value for res given cond == 0 but its not clear
  // that we actually want to bother to calculate it. Instead just verify that
  // the result is less than or equal to the unconstrained case.
  EXPECT_THAT(consequent_arm_range->GetIntervals(res.node()),
              AnyOf(Eq(x_ist), Eq(res_ist)));
}
TEST_F(ContextSensitiveRangeQueryEngineTest, ClosedClosedRangeUseInIf) {
  Bits max_bits = UBits(12, 8);
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // if (4 <= x && x <= 12) { x } else { y }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue left = fb.ULe(fb.Literal(UBits(4, 8)), x);
  BValue right = fb.ULe(x, fb.Literal(UBits(12, 8)));
  BValue cond = fb.And({left, right});
  BValue res = fb.Select(cond, {y, x});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree x_ist =
      BitsLTT(x.node(), {Interval(UBits(4, 8), UBits(12, 8))});
  IntervalSetTree x_ist_global = BitsLTT(x.node(), {Interval::Maximal(8)});

  IntervalSetTree y_ist = BitsLTT(y.node(), {Interval::Maximal(8)});
  IntervalSetTree left_ist = BitsLTT(left.node(), {Interval::Maximal(1)});
  IntervalSetTree right_ist = BitsLTT(right.node(), {Interval::Maximal(1)});
  IntervalSetTree cond_ist = BitsLTT(cond.node(), {Interval::Maximal(1)});
  IntervalSetTree res_ist = BitsLTT(res.node(), {Interval::Maximal(8)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_ist_global);
  EXPECT_EQ(engine.GetIntervals(y.node()), y_ist);
  EXPECT_EQ(engine.GetIntervals(left.node()), left_ist);
  EXPECT_EQ(engine.GetIntervals(right.node()), right_ist);
  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(engine.GetIntervals(res.node()), res_ist);

  EXPECT_EQ(consequent_arm_range->GetIntervals(x.node()), x_ist);

  EXPECT_EQ(consequent_arm_range->GetIntervals(left.node()),
            BitsLTT(left.node(), {Interval::Precise(UBits(1, 1))}));
  EXPECT_EQ(consequent_arm_range->GetIntervals(right.node()),
            BitsLTT(right.node(), {Interval::Precise(UBits(1, 1))}));
  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(UBits(1, 1))}));

  EXPECT_EQ(alternate_arm_range->GetIntervals(left.node()),
            BitsLTT(left.node(), {Interval::Maximal(1)}));
  EXPECT_EQ(alternate_arm_range->GetIntervals(right.node()),
            BitsLTT(right.node(), {Interval::Maximal(1)}));
  // NB Result is same as base since the conditional values can't affect the
  // arm so no need to compute them.
  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()),
            engine.GetIntervals(cond.node()));
  // NB There is a restricted value for res given cond == 0 but its not clear
  // that we actually want to bother to calculate it. Instead just verify that
  // the result is less than or equal to the unconstrained case.
  EXPECT_THAT(consequent_arm_range->GetIntervals(res.node()),
              AnyOf(Eq(x_ist), Eq(res_ist)));
}

TEST_F(ContextSensitiveRangeQueryEngineTest,
       UseInComplicatedExpressionBubblesDown) {
  Bits max_bits = bits_ops::Sub(UBits(12, 64), UBits(1, 64));
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // if (x < 12) { (x + 6) } else { y }
  BValue x = fb.Param("x", p->GetBitsType(64));
  BValue y = fb.Param("y", p->GetBitsType(64));
  BValue cond = fb.ULt(x, fb.Literal(UBits(12, 64)));
  BValue x_full = fb.Add(x, fb.Literal(UBits(6, 64)));
  BValue res = fb.Select(cond, {y, x_full});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  IntervalSetTree x_ist = BitsLTT(x.node(), {Interval(Bits(64), max_bits)});
  IntervalSetTree x_ist_global = BitsLTT(x.node(), {Interval::Maximal(64)});
  IntervalSetTree x_full_ist =
      BitsLTT(x_full.node(), {Interval(UBits(6, 64), UBits(17, 64))});
  IntervalSetTree x_full_ist_global =
      BitsLTT(x_full.node(), {Interval::Maximal(64)});

  IntervalSetTree y_ist = BitsLTT(y.node(), {Interval::Maximal(64)});
  IntervalSetTree cond_ist = BitsLTT(cond.node(), {Interval::Maximal(1)});
  IntervalSetTree res_ist = BitsLTT(res.node(), {Interval::Maximal(64)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_ist_global);
  EXPECT_EQ(engine.GetIntervals(x_full.node()), x_full_ist_global);
  EXPECT_EQ(engine.GetIntervals(y.node()), y_ist);
  EXPECT_EQ(engine.GetIntervals(cond.node()), cond_ist);
  EXPECT_EQ(engine.GetIntervals(res.node()), res_ist);

  EXPECT_EQ(consequent_arm_range->GetIntervals(x.node()), x_ist);
  EXPECT_EQ(consequent_arm_range->GetIntervals(x_full.node()), x_full_ist);

  // NB Result is same as base since the conditional values can't affect the
  // arm so no need to compute them.
  EXPECT_EQ(alternate_arm_range->GetIntervals(cond.node()),
            engine.GetIntervals(cond.node()));

  EXPECT_EQ(consequent_arm_range->GetIntervals(cond.node()),
            BitsLTT(cond.node(), {Interval::Precise(UBits(1, 1))}));
  // NB There is a restricted value for res given cond == 0 but its not clear
  // that we actually want to bother to calculate it. Instead just verify that
  // the result is less than or equal to the unconstrained case.
  EXPECT_THAT(consequent_arm_range->GetIntervals(res.node()),
              AnyOf(Eq(x_ist), Eq(res_ist)));
}

TEST_P(SignedRangeComparisonContextSensitiveRangeQueryEngineTest,
       UsedInTrueRange) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // if (x in [10, 15]) { x + 10 } else { y }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue cmp = RangeComparison(fb, fb.Literal(UBits(10, 8)), x,
                               fb.Literal(UBits(15, 8)));
  BValue add_ten = fb.Add(x, fb.Literal(UBits(10, 8)));
  BValue res = fb.Select(cmp, {y, add_ten});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RecordProperty("f", f->DumpIr());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));

  IntervalSetTree x_ist =
      BitsLTT(x.node(), ParamInterval(UBits(10, 8), UBits(15, 8)));
  IntervalSetTree x_ist_global = BitsLTT(x.node(), {Interval::Maximal(8)});
  IntervalSetTree add_ten_ist =
      BitsLTT(add_ten.node(), ParamInterval(UBits(20, 8), UBits(25, 8)));
  IntervalSetTree add_ten_ist_global =
      BitsLTT(add_ten.node(), {Interval::Maximal(8)});

  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_ist_global);
  EXPECT_EQ(engine.GetIntervals(add_ten.node()), add_ten_ist_global);

  EXPECT_EQ(consequent_arm_range->GetIntervals(x.node()), x_ist);

  EXPECT_EQ(consequent_arm_range->GetIntervals(add_ten.node()), add_ten_ist);
}

TEST_P(SignedRangeComparisonContextSensitiveRangeQueryEngineTest,
       UsedInFalseRange) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // if (x in [10, 15]) { y } else { x + 10 }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue cmp = RangeComparison(fb, fb.Literal(UBits(10, 8)), x,
                               fb.Literal(UBits(15, 8)));
  BValue add_ten = fb.Add(x, fb.Literal(UBits(10, 8)));
  BValue res = fb.Select(cmp, {add_ten, y});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RecordProperty("f", f->DumpIr());
  ContextSensitiveRangeQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  IntervalSetTree x_ist =
      BitsLTT(x.node(), InverseParamInterval(UBits(10, 8), UBits(15, 8)));
  IntervalSetTree x_ist_global = BitsLTT(x.node(), {Interval::Maximal(8)});
  IntervalSetTree add_ten_ist =
      BitsLTT(add_ten.node(), InverseParamInterval(UBits(20, 8), UBits(25, 8)));
  IntervalSetTree add_ten_ist_global =
      BitsLTT(add_ten.node(), {Interval::Maximal(8)});

  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});
  EXPECT_EQ(engine.GetIntervals(x.node()), x_ist_global);
  EXPECT_EQ(engine.GetIntervals(add_ten.node()), add_ten_ist_global);

  EXPECT_EQ(alternate_arm_range->GetIntervals(x.node()), x_ist);
}

TEST_F(ContextSensitiveRangeQueryEngineTest, DirectUse) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // Check we can get results for direct uses on many arms
  // match (x) { 0 => 0, 1 -> 0, 2 -> 0, 3 -> x + 1, 4 -> x + 2, _ -> 0 }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue zero = fb.Literal(UBits(0, 8));
  BValue x_plus_one = fb.Add(x, fb.Literal(UBits(1, 8)));
  BValue x_plus_two = fb.Add(x, fb.Literal(UBits(2, 8)));
  BValue res = fb.Select(x, {zero, zero, zero, x_plus_one, x_plus_two}, zero);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));

  IntervalSetTree x_global_ist = BitsLTT(x.node(), {Interval::Maximal(8)});
  IntervalSetTree x_plus_one_global_ist =
      BitsLTT(x_plus_one.node(), {Interval::Maximal(8)});
  IntervalSetTree x_plus_two_global_ist =
      BitsLTT(x_plus_two.node(), {Interval::Maximal(8)});
  IntervalSetTree res_global_ist = BitsLTT(res.node(), {Interval::Maximal(8)});

  IntervalSetTree x_3_ist = BitsLTT(x.node(), {Interval::Precise(UBits(3, 8))});
  IntervalSetTree x_plus_one_3_ist =
      BitsLTT(x_plus_one.node(), {Interval::Precise(UBits(4, 8))});

  IntervalSetTree x_4_ist = BitsLTT(x.node(), {Interval::Precise(UBits(4, 8))});
  IntervalSetTree x_plus_two_4_ist =
      BitsLTT(x_plus_two.node(), {Interval::Precise(UBits(6, 8))});

  auto arm_3_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), 3)});
  auto arm_4_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), 4)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_global_ist);
  EXPECT_EQ(engine.GetIntervals(x_plus_one.node()), x_plus_one_global_ist);
  EXPECT_EQ(engine.GetIntervals(x_plus_two.node()), x_plus_two_global_ist);
  EXPECT_EQ(engine.GetIntervals(res.node()), res_global_ist);

  EXPECT_EQ(arm_3_range->GetIntervals(x.node()), x_3_ist);
  EXPECT_EQ(arm_3_range->GetIntervals(x_plus_one.node()), x_plus_one_3_ist);

  EXPECT_EQ(arm_4_range->GetIntervals(x.node()), x_4_ist);
  EXPECT_EQ(arm_4_range->GetIntervals(x_plus_two.node()), x_plus_two_4_ist);
}

TEST_F(ContextSensitiveRangeQueryEngineTest, HandlesAllArms) {
  static constexpr uint64_t kMaxU8 =
      static_cast<uint64_t>(std::numeric_limits<uint8_t>::max());
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // Check we can get results for more than a single arm
  // if (x <= 8) { (10 - x) as u64 } else { (x as u64) + 10 }
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue sub_ten = fb.ZeroExtend(fb.Subtract(fb.Literal(UBits(10, 8)), x), 64);
  BValue add_ten = fb.Add(fb.ZeroExtend(x, 64), fb.Literal(UBits(10, 64)));
  BValue cond = fb.ULe(x, fb.Literal(UBits(8, 8)));
  BValue res = fb.Select(cond, {add_ten, sub_ten});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ContextSensitiveRangeQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));

  IntervalSetTree x_global_ist = BitsLTT(x.node(), {Interval::Maximal(8)});
  IntervalSetTree x_consequent_ist =  // [0, 8]
      BitsLTT(x.node(), {Interval(UBits(0, 8), UBits(8, 8))});
  IntervalSetTree x_alternate_ist =  // [9, max<u8>]
      BitsLTT(x.node(), {Interval(UBits(9, 8), Bits::AllOnes(8))});

  IntervalSetTree sub_ten_global_ist =
      BitsLTT(sub_ten.node(), {Interval::Maximal(8).ZeroExtend(64)});
  IntervalSetTree sub_ten_consequent_ist =  // [2, 10]
      BitsLTT(sub_ten.node(),
              {Interval(UBits(2, 8), UBits(10, 8)).ZeroExtend(64)});

  IntervalSetTree add_ten_global_ist = BitsLTT(
      add_ten.node(), {Interval(UBits(10, 64), UBits(kMaxU8 + 10, 64))});
  IntervalSetTree add_ten_alternate_ist = BitsLTT(
      add_ten.node(), {Interval(UBits(19, 64), UBits(kMaxU8 + 10, 64))});

  auto consequent_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kConsequentArm)});
  auto alternate_arm_range = engine.SpecializeGivenPredicate(
      {PredicateState(res.node()->As<Select>(), kAlternateArm)});

  EXPECT_EQ(engine.GetIntervals(x.node()), x_global_ist);
  EXPECT_EQ(consequent_arm_range->GetIntervals(x.node()), x_consequent_ist);
  EXPECT_EQ(alternate_arm_range->GetIntervals(x.node()), x_alternate_ist);
  EXPECT_EQ(engine.GetIntervals(sub_ten.node()), sub_ten_global_ist);
  EXPECT_EQ(consequent_arm_range->GetIntervals(sub_ten.node()),
            sub_ten_consequent_ist);
  EXPECT_EQ(engine.GetIntervals(add_ten.node()), add_ten_global_ist);
  EXPECT_EQ(alternate_arm_range->GetIntervals(add_ten.node()),
            add_ten_alternate_ist);
}

INSTANTIATE_TEST_SUITE_P(Signed, SignedContextSensitiveRangeQueryEngineTest,
                         testing::Values(Signedness::kSigned,
                                         Signedness::kUnsigned),
                         testing::PrintToStringParamName());
INSTANTIATE_TEST_SUITE_P(
    Range, SignedRangeComparisonContextSensitiveRangeQueryEngineTest,
    testing::ConvertGenerator<SignedRangeComparison::Tuple>(testing::Combine(
        testing::Values(Signedness::kSigned, Signedness::kUnsigned),
        testing::Values(AndOrder::kLeftFirst, AndOrder::kRightFirst),
        testing::Values(ComparisonType::kOpen, ComparisonType::kClosed),
        testing::Values(ComparisonOrder::kLiteralFirst,
                        ComparisonOrder::kParamFirst),
        testing::Values(ComparisonType::kOpen, ComparisonType::kClosed),
        testing::Values(ComparisonOrder::kLiteralFirst,
                        ComparisonOrder::kParamFirst))),
    testing::PrintToStringParamName());

}  // namespace
}  // namespace xls
