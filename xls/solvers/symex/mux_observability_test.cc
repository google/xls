// Copyright 2026 The XLS Authors
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

#include "xls/solvers/symex/mux_observability.h"

#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"

namespace xls::solvers::symex {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

class MuxObservabilityTest : public IrTestBase {};

TEST_F(MuxObservabilityTest, FunctionWithoutMuxesHasNoMuxes) {
  Package p(TestName());
  FunctionBuilder fb(TestName(), &p);
  BValue a = fb.Param("a", p.GetBitsType(8));
  BValue b = fb.Param("b", p.GetBitsType(8));
  fb.Add(a, b);
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(MuxObservability observability,
                           MuxObservability::Create(fn));

  EXPECT_THAT(observability.muxes(), IsEmpty());
}

TEST_F(MuxObservabilityTest, MuxesAreOrderedConsumersBeforeProducers) {
  Package p(TestName());
  FunctionBuilder fb(TestName(), &p);
  BValue s0 = fb.Param("s0", p.GetBitsType(1));
  BValue s1 = fb.Param("s1", p.GetBitsType(1));
  BValue inner =
      fb.Select(s0, {fb.Literal(UBits(0, 8)), fb.Literal(UBits(1, 8))});

  BValue outer = fb.Select(s1, {inner, fb.Literal(UBits(2, 8))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(MuxObservability observability,
                           MuxObservability::Create(fn));

  EXPECT_THAT(observability.muxes(), ElementsAre(outer.node(), inner.node()));
}

TEST_F(MuxObservabilityTest, ReturnValueMuxIsObservableWithoutAnyDecision) {
  Package p(TestName());
  FunctionBuilder fb(TestName(), &p);
  BValue s = fb.Param("s", p.GetBitsType(1));
  BValue a = fb.Param("a", p.GetBitsType(8));
  BValue b = fb.Param("b", p.GetBitsType(8));
  BValue mux = fb.Select(s, {a, b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(MuxObservability observability,
                           MuxObservability::Create(fn));

  EXPECT_TRUE(observability.IsObservable(mux.node()));
}

TEST_F(MuxObservabilityTest, UpstreamMuxIsObservableOnlyUnderTheArmFeedingIt) {
  // `outer` selects between `inner` (arm 0) and a literal (arm 1), so `inner`
  // matters on the first arm and is a don't care on the second.
  Package p(TestName());
  FunctionBuilder fb(TestName(), &p);
  BValue s0 = fb.Param("s0", p.GetBitsType(1));
  BValue s1 = fb.Param("s1", p.GetBitsType(1));
  BValue inner =
      fb.Select(s0, {fb.Literal(UBits(0, 8)), fb.Literal(UBits(1, 8))});
  BValue unrelated = fb.Literal(UBits(2, 8));
  BValue outer = fb.Select(s1, {inner, unrelated});
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(MuxObservability observability,
                           MuxObservability::Create(fn));
  ASSERT_FALSE(observability.IsObservable(inner.node()));

  int64_t token = observability.MarkArmChosen(outer.node(), unrelated.node());
  EXPECT_FALSE(observability.IsObservable(inner.node()));
  observability.Rollback(token);

  token = observability.MarkArmChosen(outer.node(), inner.node());
  EXPECT_TRUE(observability.IsObservable(inner.node()));
  observability.Rollback(token);

  EXPECT_FALSE(observability.IsObservable(inner.node()));
}

TEST_F(MuxObservabilityTest, MuxFeedingASelectorIsObservableUnderEveryArm) {
  // `inner` computes the selector of `outer`, so no arm choice at `outer` can
  // make `inner` irrelevant.
  Package p(TestName());
  FunctionBuilder fb(TestName(), &p);
  BValue s = fb.Param("s", p.GetBitsType(1));
  BValue a = fb.Param("a", p.GetBitsType(8));
  BValue b = fb.Param("b", p.GetBitsType(8));
  BValue inner =
      fb.Select(s, {fb.Literal(UBits(0, 1)), fb.Literal(UBits(1, 1))});
  BValue outer = fb.Select(inner, {a, b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(MuxObservability observability,
                           MuxObservability::Create(fn));
  ASSERT_FALSE(observability.IsObservable(inner.node()));

  int64_t token = observability.MarkArmChosen(outer.node(), a.node());
  EXPECT_TRUE(observability.IsObservable(inner.node()));
  observability.Rollback(token);

  token = observability.MarkArmChosen(outer.node(), b.node());
  EXPECT_TRUE(observability.IsObservable(inner.node()));
  observability.Rollback(token);
}

TEST_F(MuxObservabilityTest, RootObservabilitySurvivesRollback) {
  Package p(TestName());
  FunctionBuilder fb(TestName(), &p);
  BValue s = fb.Param("s", p.GetBitsType(1));
  BValue a = fb.Param("a", p.GetBitsType(8));
  BValue b = fb.Param("b", p.GetBitsType(8));
  BValue mux = fb.Select(s, {a, b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(MuxObservability observability,
                           MuxObservability::Create(fn));

  int64_t token = observability.MarkArmChosen(mux.node(), a.node());
  observability.Rollback(token);

  // The return value is observable on every path, so rolling back the only
  // decision must not retract it.
  EXPECT_TRUE(observability.IsObservable(mux.node()));

  // Rolling back to 0 must also not retract the root return value multiplexer.
  observability.Rollback(0);
  EXPECT_TRUE(observability.IsObservable(mux.node()));
}

TEST_F(MuxObservabilityTest, DiamondFanInDoesNotDuplicateBoundaryMuxes) {
  Package p(TestName());
  FunctionBuilder fb(TestName(), &p);
  BValue s = fb.Param("s", p.GetBitsType(1));
  BValue a = fb.Param("a", p.GetBitsType(8));
  BValue b = fb.Param("b", p.GetBitsType(8));
  BValue inner_mux = fb.Select(s, {a, b});
  // Diamond fan-in: inner_mux feeds both operands of an add before reaching
  // outer_mux.
  BValue left = fb.Add(inner_mux, fb.Literal(UBits(1, 8)));
  BValue right = fb.Add(inner_mux, fb.Literal(UBits(2, 8)));
  BValue sum = fb.Add(left, right);
  BValue s_outer = fb.Param("s_outer", p.GetBitsType(1));
  BValue outer_mux = fb.Select(s_outer, {sum, fb.Literal(UBits(0, 8))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(MuxObservability observability,
                           MuxObservability::Create(fn));
  int64_t token = observability.MarkArmChosen(outer_mux.node(), sum.node());
  EXPECT_TRUE(observability.IsObservable(inner_mux.node()));
  observability.Rollback(token);
}

}  // namespace
}  // namespace xls::solvers::symex
