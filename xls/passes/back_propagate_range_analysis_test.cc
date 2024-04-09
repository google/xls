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

#include <optional>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/ternary.h"
#include "xls/passes/range_query_engine.h"

namespace xls {
namespace {

class BackPropagateRangeAnalysisTest : public IrTestBase {};

using testing::ElementsAre;
using testing::Pair;
using testing::UnorderedElementsAre;

MATCHER_P(TernaryIs, tern,
          absl::StrFormat("ternary %smatch", negation ? "does not " : "")) {
  std::optional<TernaryVector> vec = arg;
  return testing::ExplainMatchResult(
      tern, vec ? std::make_optional(ToString(*vec)) : std::nullopt,
      result_listener);
}

MATCHER_P2(HasRangeData, tern, ist,
           absl::StrFormat("range data which is%s 'ternary: %s', 'ist: %s'",
                           negation ? " not" : "", tern,
                           testing::DescribeMatcher<IntervalSet>(ist))) {
  const RangeData& r = arg;
  IntervalSet is = r.interval_set.Get({});
  return testing::ExplainMatchResult(TernaryIs(tern), r.ternary,
                                     result_listener) &&
         testing::ExplainMatchResult(ist, is, result_listener);
}

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
      auto results, PropagateGivensBackwards(qe, target.node(), UBits(0, 1)));
  EXPECT_THAT(results,
              ElementsAre(Pair(
                  target.node(),
                  HasRangeData("0b0", IntervalSet::Precise(UBits(0, 1))))));
}

IntervalSet Intervals(absl::Span<Interval const> intervals) {
  IntervalSet o(intervals.front().BitCount());
  for (const auto& i : intervals) {
    o.AddInterval(i);
  }
  return o;
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
      auto results, PropagateGivensBackwards(qe, target.node(), UBits(1, 1)));
  EXPECT_THAT(
      results,
      UnorderedElementsAre(
          Pair(target.node(),
               HasRangeData("0b1", IntervalSet::Precise(UBits(1, 1)))),
          Pair(arg.node(),
               HasRangeData("0b000X", Intervals({Interval::Closed(
                                          UBits(0, 4), UBits(1, 4))})))));
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
      auto results, PropagateGivensBackwards(qe, target.node(), UBits(1, 1)));
  EXPECT_THAT(
      results,
      UnorderedElementsAre(
          Pair(target.node(),
               HasRangeData("0b1", IntervalSet::Precise(UBits(1, 1)))),
          Pair(arg.node(),
               HasRangeData("0b0XXX", Intervals({Interval::Closed(
                                          UBits(1, 4), UBits(4, 4))}))),
          Pair(target.node()->operand(0),
               HasRangeData("0b1", IntervalSet::Precise(UBits(1, 1)))),
          Pair(target.node()->operand(1),
               HasRangeData("0b1", IntervalSet::Precise(UBits(1, 1))))));
}

}  // namespace
}  // namespace xls
