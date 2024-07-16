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

#include "xls/passes/back_propagate_range_analysis.h"

#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/op.h"
#include "xls/passes/range_query_engine.h"

namespace m = xls::op_matchers;
namespace xls {
namespace {

using testing::ElementsAre;
using testing::Pair;
using testing::UnorderedElementsAre;

class BackPropagateRangeAnalysisTest : public IrTestBase {
 public:
  auto LiteralPair(const Bits& value) {
    return Pair(m::Literal(value), IntervalSet::Precise(value));
  }
};

// Super basic check that we can call this without issues.
TEST_F(BackPropagateRangeAnalysisTest, PropagateNothing) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto arg = fb.Param("arg", p->GetBitsType(4));
  // There is nothing that can be gained from (== 0 (and-reduce X)) since all it
  // means is at least one bit is 0.
  auto target = fb.AndReduce(arg);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  RangeQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(f).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto results, PropagateOneGivenBackwards(qe, target.node(), UBits(0, 1)));
  EXPECT_THAT(results, ElementsAre(Pair(target.node(),
                                        IntervalSet::Precise(UBits(0, 1)))));
}

TEST_F(BackPropagateRangeAnalysisTest, SignedLessThanX) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto arg = fb.Param("arg", p->GetBitsType(4));
  auto target = fb.SLt(arg, fb.Literal(UBits(2, 4)));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  RangeQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(f).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto results, PropagateOneGivenBackwards(qe, target.node(), UBits(1, 1)));
  EXPECT_THAT(
      results,
      UnorderedElementsAre(
          LiteralPair(UBits(2, 4)),
          Pair(target.node(), IntervalSet::Precise(UBits(1, 1))),
          Pair(arg.node(),
               IntervalSet::Of(
                   {Interval::Closed(UBits(0, 4), UBits(1, 4)),
                    Interval::Closed(Bits::MinSigned(4), SBits(-1, 4))}))));
}

TEST_F(BackPropagateRangeAnalysisTest, LessThanX) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto arg = fb.Param("arg", p->GetBitsType(4));
  auto target = fb.ULt(arg, fb.Literal(UBits(2, 4)));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  RangeQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(f).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto results, PropagateOneGivenBackwards(qe, target.node(), UBits(1, 1)));
  EXPECT_THAT(
      results,
      UnorderedElementsAre(
          LiteralPair(UBits(2, 4)),
          Pair(target.node(), IntervalSet::Precise(UBits(1, 1))),
          Pair(arg.node(),
               IntervalSet::Of({Interval::Closed(UBits(0, 4), UBits(1, 4))}))));
}

TEST_F(BackPropagateRangeAnalysisTest, Between) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto arg = fb.Param("arg", p->GetBitsType(4));
  auto target = fb.And(fb.UGt(arg, fb.Literal(UBits(0, 4))),
                       fb.ULt(arg, fb.Literal(UBits(5, 4))));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  RangeQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(f).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto results, PropagateOneGivenBackwards(qe, target.node(), UBits(1, 1)));
  EXPECT_THAT(
      results,
      UnorderedElementsAre(
          LiteralPair(UBits(0, 4)), LiteralPair(UBits(5, 4)),
          Pair(target.node(), IntervalSet::Precise(UBits(1, 1))),
          Pair(arg.node(),
               IntervalSet::Of({Interval::Closed(UBits(1, 4), UBits(4, 4))})),
          Pair(target.node()->operand(0), IntervalSet::Precise(UBits(1, 1))),
          Pair(target.node()->operand(1), IntervalSet::Precise(UBits(1, 1)))));
}

TEST_F(BackPropagateRangeAnalysisTest, UnsignedAdd) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto param = fb.Param("foo", p->GetBitsType(8));
  auto add_four = fb.Add(param, fb.Literal(UBits(4, 8)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  RangeQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(f).status());
  auto add_four_interval =
      IntervalSet::Of({Interval(UBits(4, 8), UBits(32, 8))});
  XLS_ASSERT_OK_AND_ASSIGN(
      auto results,
      PropagateOneGivenBackwards(qe, add_four.node(), add_four_interval));

  EXPECT_THAT(
      results,
      UnorderedElementsAre(
          LiteralPair(UBits(4, 8)), Pair(add_four.node(), add_four_interval),
          Pair(param.node(),
               IntervalSet::Of({Interval(UBits(0, 8), UBits(28, 8))}))));
}

TEST_F(BackPropagateRangeAnalysisTest, UnsignedAddCompare) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto param = fb.Param("foo", p->GetBitsType(8));
  auto add_four = fb.Add(param, fb.Literal(UBits(4, 8)));
  auto compare = fb.ULe(add_four, fb.Literal(UBits(32, 8)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  RangeQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(f).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto results, PropagateOneGivenBackwards(
                        qe, compare.node(), IntervalSet::Precise(UBits(1, 1))));

  EXPECT_THAT(
      results,
      UnorderedElementsAre(
          LiteralPair(UBits(4, 8)), LiteralPair(UBits(32, 8)),
          Pair(compare.node(), IntervalSet::Precise(UBits(1, 1))),
          Pair(add_four.node(),
               IntervalSet::Of({Interval(UBits(0, 8), UBits(32, 8))})),
          Pair(param.node(),
               IntervalSet::Of({Interval(UBits(0, 8), UBits(28, 8)),
                                Interval(SBits(-4, 8), UBits(0xff, 8))}))));
}

TEST_F(BackPropagateRangeAnalysisTest, UnsignedAddMulti) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto param = fb.Param("foo", p->GetBitsType(8));
  auto l1 = fb.Literal(UBits(1, 8));
  auto a1 = fb.Add(param, l1);
  auto a2 = fb.Add(a1, l1);
  auto a3 = fb.Add(a2, l1);
  auto a4 = fb.Add(a3, l1);
  auto a5 = fb.Add(a4, l1);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  RangeQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(f).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto results,
      PropagateOneGivenBackwards(
          qe, a5.node(),
          IntervalSet::Of({Interval(UBits(10, 8), UBits(20, 8))})));

  EXPECT_THAT(
      results,
      UnorderedElementsAre(
          LiteralPair(UBits(1, 8)),
          Pair(a5.node(),
               IntervalSet::Of({Interval(UBits(10, 8), UBits(20, 8))})),
          Pair(a4.node(),
               IntervalSet::Of({Interval(UBits(9, 8), UBits(19, 8))})),
          Pair(a3.node(),
               IntervalSet::Of({Interval(UBits(8, 8), UBits(18, 8))})),
          Pair(a2.node(),
               IntervalSet::Of({Interval(UBits(7, 8), UBits(17, 8))})),
          Pair(a1.node(),
               IntervalSet::Of({Interval(UBits(6, 8), UBits(16, 8))})),
          Pair(param.node(),
               IntervalSet::Of({Interval(UBits(5, 8), UBits(15, 8))}))));
}

TEST_F(BackPropagateRangeAnalysisTest, MultipleGivens) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto param = fb.Param("foo", p->GetBitsType(8));
  auto secret_limit = fb.Param("secret_limit", p->GetBitsType(8));
  auto compare = fb.ULe(param, secret_limit);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  RangeQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(f).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto results,
      PropagateGivensBackwards(
          qe, f,
          {{compare.node(), IntervalSet::Precise(UBits(1, 1))},
           {secret_limit.node(), IntervalSet::Precise(UBits(32, 8))}}));

  EXPECT_THAT(
      results,
      UnorderedElementsAre(
          Pair(secret_limit.node(), IntervalSet::Precise(UBits(32, 8))),
          Pair(compare.node(), IntervalSet::Precise(UBits(1, 1))),
          Pair(param.node(),
               IntervalSet::Of({Interval(UBits(0, 8), UBits(32, 8))}))));
}

TEST_F(BackPropagateRangeAnalysisTest, And) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a1 = fb.Param("a1", p->GetBitsType(1));
  auto a2 = fb.Param("a2", p->GetBitsType(1));
  auto a3 = fb.Param("a3", p->GetBitsType(1));
  auto a4 = fb.Param("a4", p->GetBitsType(1));
  auto a5 = fb.Param("a5", p->GetBitsType(1));
  auto a6 = fb.Param("a6", p->GetBitsType(1));
  auto comp = fb.And({a1, a2, a3, a4, a5, a6});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  RangeQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(f).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto results_true,
      PropagateGivensBackwards(
          qe, f, {{comp.node(), IntervalSet::Precise(UBits(1, 1))}}));

  XLS_ASSERT_OK_AND_ASSIGN(
      auto results_false,
      PropagateGivensBackwards(
          qe, f, {{comp.node(), IntervalSet::Precise(UBits(0, 1))}}));

  EXPECT_THAT(
      results_true,
      UnorderedElementsAre(Pair(comp.node(), IntervalSet::Precise(UBits(1, 1))),
                           Pair(a1.node(), IntervalSet::Precise(UBits(1, 1))),
                           Pair(a2.node(), IntervalSet::Precise(UBits(1, 1))),
                           Pair(a3.node(), IntervalSet::Precise(UBits(1, 1))),
                           Pair(a4.node(), IntervalSet::Precise(UBits(1, 1))),
                           Pair(a5.node(), IntervalSet::Precise(UBits(1, 1))),
                           Pair(a6.node(), IntervalSet::Precise(UBits(1, 1)))));
  EXPECT_THAT(results_false,
              UnorderedElementsAre(
                  Pair(comp.node(), IntervalSet::Precise(UBits(0, 1)))));
}
TEST_F(BackPropagateRangeAnalysisTest, Or) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a1 = fb.Param("a1", p->GetBitsType(1));
  auto a2 = fb.Param("a2", p->GetBitsType(1));
  auto a3 = fb.Param("a3", p->GetBitsType(1));
  auto a4 = fb.Param("a4", p->GetBitsType(1));
  auto a5 = fb.Param("a5", p->GetBitsType(1));
  auto a6 = fb.Param("a6", p->GetBitsType(1));
  auto comp = fb.Or({a1, a2, a3, a4, a5, a6});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  RangeQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(f).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto results_true,
      PropagateGivensBackwards(
          qe, f, {{comp.node(), IntervalSet::Precise(UBits(1, 1))}}));

  XLS_ASSERT_OK_AND_ASSIGN(
      auto results_false,
      PropagateGivensBackwards(
          qe, f, {{comp.node(), IntervalSet::Precise(UBits(0, 1))}}));

  EXPECT_THAT(
      results_false,
      UnorderedElementsAre(Pair(comp.node(), IntervalSet::Precise(UBits(0, 1))),
                           Pair(a1.node(), IntervalSet::Precise(UBits(0, 1))),
                           Pair(a2.node(), IntervalSet::Precise(UBits(0, 1))),
                           Pair(a3.node(), IntervalSet::Precise(UBits(0, 1))),
                           Pair(a4.node(), IntervalSet::Precise(UBits(0, 1))),
                           Pair(a5.node(), IntervalSet::Precise(UBits(0, 1))),
                           Pair(a6.node(), IntervalSet::Precise(UBits(0, 1)))));
  EXPECT_THAT(results_true,
              UnorderedElementsAre(
                  Pair(comp.node(), IntervalSet::Precise(UBits(1, 1)))));
}

TEST_F(BackPropagateRangeAnalysisTest, Nand) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a1 = fb.Param("a1", p->GetBitsType(1));
  auto a2 = fb.Param("a2", p->GetBitsType(1));
  auto a3 = fb.Param("a3", p->GetBitsType(1));
  auto a4 = fb.Param("a4", p->GetBitsType(1));
  auto a5 = fb.Param("a5", p->GetBitsType(1));
  auto a6 = fb.Param("a6", p->GetBitsType(1));
  auto comp = fb.AddNaryOp(Op::kNand, {a1, a2, a3, a4, a5, a6});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  RangeQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(f).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto results_true,
      PropagateGivensBackwards(
          qe, f, {{comp.node(), IntervalSet::Precise(UBits(1, 1))}}));

  XLS_ASSERT_OK_AND_ASSIGN(
      auto results_false,
      PropagateGivensBackwards(
          qe, f, {{comp.node(), IntervalSet::Precise(UBits(0, 1))}}));

  EXPECT_THAT(
      results_false,
      UnorderedElementsAre(Pair(comp.node(), IntervalSet::Precise(UBits(0, 1))),
                           Pair(a1.node(), IntervalSet::Precise(UBits(1, 1))),
                           Pair(a2.node(), IntervalSet::Precise(UBits(1, 1))),
                           Pair(a3.node(), IntervalSet::Precise(UBits(1, 1))),
                           Pair(a4.node(), IntervalSet::Precise(UBits(1, 1))),
                           Pair(a5.node(), IntervalSet::Precise(UBits(1, 1))),
                           Pair(a6.node(), IntervalSet::Precise(UBits(1, 1)))));
  EXPECT_THAT(results_true,
              UnorderedElementsAre(
                  Pair(comp.node(), IntervalSet::Precise(UBits(1, 1)))));
}

TEST_F(BackPropagateRangeAnalysisTest, Nor) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a1 = fb.Param("a1", p->GetBitsType(1));
  auto a2 = fb.Param("a2", p->GetBitsType(1));
  auto a3 = fb.Param("a3", p->GetBitsType(1));
  auto a4 = fb.Param("a4", p->GetBitsType(1));
  auto a5 = fb.Param("a5", p->GetBitsType(1));
  auto a6 = fb.Param("a6", p->GetBitsType(1));
  auto comp = fb.AddNaryOp(Op::kNor, {a1, a2, a3, a4, a5, a6});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  RangeQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(f).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto results_true,
      PropagateGivensBackwards(
          qe, f, {{comp.node(), IntervalSet::Precise(UBits(1, 1))}}));

  XLS_ASSERT_OK_AND_ASSIGN(
      auto results_false,
      PropagateGivensBackwards(
          qe, f, {{comp.node(), IntervalSet::Precise(UBits(0, 1))}}));

  EXPECT_THAT(
      results_true,
      UnorderedElementsAre(Pair(comp.node(), IntervalSet::Precise(UBits(1, 1))),
                           Pair(a1.node(), IntervalSet::Precise(UBits(0, 1))),
                           Pair(a2.node(), IntervalSet::Precise(UBits(0, 1))),
                           Pair(a3.node(), IntervalSet::Precise(UBits(0, 1))),
                           Pair(a4.node(), IntervalSet::Precise(UBits(0, 1))),
                           Pair(a5.node(), IntervalSet::Precise(UBits(0, 1))),
                           Pair(a6.node(), IntervalSet::Precise(UBits(0, 1)))));
  EXPECT_THAT(results_false,
              UnorderedElementsAre(
                  Pair(comp.node(), IntervalSet::Precise(UBits(0, 1)))));
}

TEST_F(BackPropagateRangeAnalysisTest, Concat) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a1 = fb.Param("a1", p->GetBitsType(4));
  auto cmp = fb.ULe(a1, fb.Literal(UBits(2, 4)));
  auto cc = fb.Concat({fb.Literal(UBits(0b1010, 4)), cmp});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  RangeQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(f).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto results_true,
      PropagateGivensBackwards(
          qe, f, {{cc.node(), IntervalSet::Precise(UBits(0b10101, 5))}}));

  XLS_ASSERT_OK_AND_ASSIGN(
      auto results_false,
      PropagateGivensBackwards(
          qe, f, {{cc.node(), IntervalSet::Precise(UBits(0b10100, 5))}}));

  EXPECT_THAT(results_true,
              UnorderedElementsAre(
                  LiteralPair(UBits(2, 4)), LiteralPair(UBits(0b1010, 4)),
                  Pair(cc.node(), IntervalSet::Precise(UBits(0b10101, 5))),
                  Pair(cmp.node(), IntervalSet::Precise(UBits(1, 1))),
                  Pair(a1.node(),
                       IntervalSet::Of({Interval(UBits(0, 4), UBits(2, 4))}))));
  EXPECT_THAT(
      results_false,
      UnorderedElementsAre(
          LiteralPair(UBits(2, 4)), LiteralPair(UBits(0b1010, 4)),
          Pair(cc.node(), IntervalSet::Precise(UBits(0b10100, 5))),
          Pair(cmp.node(), IntervalSet::Precise(UBits(0, 1))),
          Pair(a1.node(),
               IntervalSet::Of({Interval(UBits(3, 4), Bits::AllOnes(4))}))));
}

TEST_F(BackPropagateRangeAnalysisTest, SignExtend) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a1 = fb.Param("a1", p->GetBitsType(4));
  auto target = fb.SignExtend(a1, 64);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  RangeQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(f).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto results,
      PropagateGivensBackwards(
          qe, f,
          {{target.node(),
            IntervalSet::Of({Interval(SBits(-6, 64), SBits(-4, 64)),
                             Interval(SBits(-2, 64), SBits(-1, 64))})}}));
  EXPECT_THAT(
      results,
      UnorderedElementsAre(
          Pair(target.node(),
               IntervalSet::Of({Interval(SBits(-6, 64), SBits(-4, 64)),
                                Interval(SBits(-2, 64), SBits(-1, 64))})),
          Pair(a1.node(),
               IntervalSet::Of({Interval(SBits(-6, 4), SBits(-4, 4)),
                                Interval(SBits(-2, 4), SBits(-1, 4))}))));
}

TEST_F(BackPropagateRangeAnalysisTest, ZeroExtend) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a1 = fb.Param("a1", p->GetBitsType(4));
  auto target = fb.ZeroExtend(a1, 64);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  RangeQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(f).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto results,
      PropagateGivensBackwards(
          qe, f,
          {{target.node(),
            IntervalSet::Of({Interval(SBits(-6, 64), SBits(-4, 64)),
                             Interval(SBits(-2, 64), SBits(-1, 64))})}}));
  EXPECT_THAT(
      results,
      UnorderedElementsAre(
          Pair(target.node(),
               IntervalSet::Of({Interval(SBits(-6, 64), SBits(-4, 64)),
                                Interval(SBits(-2, 64), SBits(-1, 64))})),
          Pair(a1.node(),
               IntervalSet::Of({Interval(SBits(-6, 4), SBits(-4, 4)),
                                Interval(SBits(-2, 4), SBits(-1, 4))}))));
}

TEST_F(BackPropagateRangeAnalysisTest, AndReduce) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a1 = fb.Param("a1", p->GetBitsType(4));
  auto target = fb.AndReduce(a1);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  RangeQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(f).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto results_true,
      PropagateGivensBackwards(
          qe, f, {{target.node(), IntervalSet::Precise(UBits(1, 1))}}));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto results_false,
      PropagateGivensBackwards(
          qe, f, {{target.node(), IntervalSet::Precise(UBits(0, 1))}}));
  EXPECT_THAT(results_true,
              UnorderedElementsAre(
                  Pair(target.node(), IntervalSet::Precise(UBits(1, 1))),
                  Pair(a1.node(), IntervalSet::Precise(Bits::AllOnes(4)))));
  EXPECT_THAT(results_false,
              UnorderedElementsAre(
                  Pair(target.node(), IntervalSet::Precise(UBits(0, 1)))));
}

TEST_F(BackPropagateRangeAnalysisTest, OrReduce) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a1 = fb.Param("a1", p->GetBitsType(4));
  auto target = fb.OrReduce(a1);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  RangeQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(f).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto results_true,
      PropagateGivensBackwards(
          qe, f, {{target.node(), IntervalSet::Precise(UBits(1, 1))}}));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto results_false,
      PropagateGivensBackwards(
          qe, f, {{target.node(), IntervalSet::Precise(UBits(0, 1))}}));
  EXPECT_THAT(results_false,
              UnorderedElementsAre(
                  Pair(target.node(), IntervalSet::Precise(UBits(0, 1))),
                  Pair(a1.node(), IntervalSet::Precise(Bits(4)))));
  EXPECT_THAT(results_true,
              UnorderedElementsAre(
                  Pair(target.node(), IntervalSet::Precise(UBits(1, 1)))));
}

TEST_F(BackPropagateRangeAnalysisTest, ImpossibleSignedCmp) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  // Impossible comparison.
  auto cmp = fb.SGt(fb.Literal(UBits(1, 32)), fb.Literal(UBits(9, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  RangeQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(f).status());
  EXPECT_THAT(PropagateGivensBackwards(
                  qe, f, {{cmp.node(), IntervalSet::Precise(UBits(1, 1))}}),
              status_testing::IsOk());
}

}  // namespace
}  // namespace xls
