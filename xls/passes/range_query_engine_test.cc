// Copyright 2021 The XLS Authors
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

#include "xls/passes/range_query_engine.h"

#include <stdint.h>

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "xls/common/logging/log_message.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/passes/query_engine.h"

namespace xls {
namespace {

class RangeQueryEngineTest : public IrTestBase {};

// TODO(taktoa): replace this with a proper property-based testing library
// once we have such a thing in XLS

Interval RandomInterval(uint32_t seed, int64_t bit_count) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<uint8_t> distrib(0, 255);
  int64_t num_bytes = (bit_count / 8) + ((bit_count % 8 == 0) ? 0 : 1);
  std::vector<uint8_t> start_bytes(num_bytes);
  for (int64_t i = 0; i < num_bytes; ++i) {
    start_bytes[i] = distrib(gen);
  }
  std::vector<uint8_t> end_bytes(num_bytes);
  for (int64_t i = 0; i < num_bytes; ++i) {
    end_bytes[i] = distrib(gen);
  }
  return Interval(Bits::FromBytes(start_bytes, bit_count),
                  Bits::FromBytes(end_bytes, bit_count));
}

IntervalSet RandomIntervalSet(uint32_t seed, int64_t bit_count) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int64_t> distrib(0, 30);
  int64_t num_intervals = distrib(gen);
  IntervalSet result(bit_count);
  for (int64_t i = 0; i < num_intervals; ++i) {
    result.AddInterval(RandomInterval(gen(), bit_count));
  }
  result.Normalize();
  return result;
}

LeafTypeTree<IntervalSet> BitsLTT(Node* node,
                                  absl::Span<const Interval> intervals) {
  XLS_CHECK(!intervals.empty());
  int64_t bit_count = intervals[0].BitCount();
  IntervalSet interval_set(bit_count);
  for (const Interval& interval : intervals) {
    XLS_CHECK_EQ(interval.BitCount(), bit_count);
    interval_set.AddInterval(interval);
  }
  interval_set.Normalize();
  XLS_CHECK(node->GetType()->IsBits());
  LeafTypeTree<IntervalSet> result(node->GetType());
  result.Set({}, interval_set);
  return result;
}

TEST_F(RangeQueryEngineTest, MinimizeIntervals) {
  for (int64_t size = 1; size < 20; ++size) {
    for (int64_t bits = 1; bits < 10; ++bits) {
      for (int64_t i = 0; i < 10; ++i) {
        uint32_t seed = 802103005;
        IntervalSet interval_set = RandomIntervalSet(seed, bits);
        IntervalSet minimized = MinimizeIntervals(interval_set, size);
        EXPECT_EQ(interval_set.BitCount(), minimized.BitCount());
        EXPECT_LE(minimized.NumberOfIntervals(), size)
            << "interval_set = " << interval_set.ToString() << "\n"
            << "minimized    = " << minimized.ToString() << "\n";
      }
    }
  }
}

TEST_F(RangeQueryEngineTest, Add) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", fb.package()->GetBitsType(20));
  BValue y = fb.Param("y", fb.package()->GetBitsType(20));
  BValue expr = fb.Add(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(0, 20), UBits(48, 20))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(0, 20), UBits(15, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));

  // We know that at most 6 bits are used, since 48 + 15 = 63 <= 2^6 - 1
  EXPECT_EQ(UBits(0b11111111111111000000, 20),
            engine.GetKnownBits(expr.node()));
  EXPECT_EQ(UBits(0b00000000000000000000, 20),
            engine.GetKnownBitsValues(expr.node()));

  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(),
      BitsLTT(x.node(), {Interval(UBits(1048566, 20), Bits::AllOnes(20))}));
  engine.SetIntervalSetTree(
      y.node(),
      BitsLTT(y.node(), {Interval(UBits(1048566, 20), UBits(1048570, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));

  // Nothing is known because of the potential for overflow, even though the
  // result of additions in those ranges have a common suffix (little-endian).
  EXPECT_EQ(UBits(0, 20), engine.GetKnownBits(expr.node()));
  EXPECT_EQ(UBits(0, 20), engine.GetKnownBitsValues(expr.node()));
}

TEST_F(RangeQueryEngineTest, Eq) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", fb.package()->GetBitsType(20));
  BValue y = fb.Param("y", fb.package()->GetBitsType(20));
  BValue expr = fb.Eq(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;

  // When the interval sets are precise and equivalent, we know they are equal.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(560, 20), UBits(560, 20))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(560, 20), UBits(560, 20)),
                                   Interval(UBits(560, 20), UBits(560, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetKnownBits(expr.node()), UBits(1, 1));
  EXPECT_EQ(engine.GetKnownBitsValues(expr.node()), UBits(1, 1));

  // When the interval sets overlap, we don't know anything about them.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(100, 20), UBits(200, 20))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(150, 20), UBits(300, 20)),
                                   Interval(UBits(400, 20), UBits(500, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetKnownBits(expr.node()), UBits(0, 1));
  EXPECT_EQ(engine.GetKnownBitsValues(expr.node()), UBits(0, 1));

  // When the interval sets are disjoint, we know they are not equal.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {
                                      Interval(UBits(100, 20), UBits(200, 20)),
                                      Interval(UBits(550, 20), UBits(600, 20)),
                                      Interval(UBits(850, 20), UBits(900, 20)),
                                  }));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(400, 20), UBits(500, 20)),
                                   Interval(UBits(700, 20), UBits(800, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetKnownBits(expr.node()), UBits(1, 1));
  EXPECT_EQ(engine.GetKnownBitsValues(expr.node()), UBits(0, 1));
}

TEST_F(RangeQueryEngineTest, Gate) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue cond = fb.Literal(UBits(1, 1));
  BValue x = fb.Param("x", fb.package()->GetBitsType(20));
  BValue expr = fb.Gate(cond, x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  LeafTypeTree<IntervalSet> interval_set =
      BitsLTT(x.node(), {Interval(UBits(600, 20), UBits(700, 20))});
  engine.SetIntervalSetTree(x.node(), interval_set);
  XLS_ASSERT_OK(engine.Populate(f));

  // Gate works like an identity function with respect to range analysis.
  EXPECT_EQ(engine.GetIntervalSetTree(expr.node()), interval_set);
}

TEST_F(RangeQueryEngineTest, Identity) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", fb.package()->GetBitsType(20));
  BValue expr = fb.Identity(x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  LeafTypeTree<IntervalSet> interval_set =
      BitsLTT(x.node(), {Interval(UBits(600, 20), UBits(700, 20))});
  engine.SetIntervalSetTree(x.node(), interval_set);
  XLS_ASSERT_OK(engine.Populate(f));

  // Identity function should leave intervals unaffected.
  EXPECT_EQ(engine.GetIntervalSetTree(expr.node()), interval_set);
}

TEST_F(RangeQueryEngineTest, Literal) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue lit = fb.Literal(UBits(57, 20));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  // The interval set of a literal should be a single interval starting and
  // ending at its value.
  EXPECT_EQ(engine.GetIntervalSetTree(lit.node()),
            BitsLTT(lit.node(), {Interval(UBits(57, 20), UBits(57, 20))}));
}

TEST_F(RangeQueryEngineTest, Ne) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", fb.package()->GetBitsType(20));
  BValue y = fb.Param("y", fb.package()->GetBitsType(20));
  BValue expr = fb.Ne(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;

  // When the interval sets are precise and equivalent, we know they are equal.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(560, 20), UBits(560, 20))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(560, 20), UBits(560, 20)),
                                   Interval(UBits(560, 20), UBits(560, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetKnownBits(expr.node()), UBits(1, 1));
  EXPECT_EQ(engine.GetKnownBitsValues(expr.node()), UBits(0, 1));

  // When the interval sets overlap, we don't know anything about them.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(100, 20), UBits(200, 20))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(150, 20), UBits(300, 20)),
                                   Interval(UBits(400, 20), UBits(500, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetKnownBits(expr.node()), UBits(0, 1));
  EXPECT_EQ(engine.GetKnownBitsValues(expr.node()), UBits(0, 1));

  // When the interval sets are disjoint, we know they are not equal.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {
                                      Interval(UBits(100, 20), UBits(200, 20)),
                                      Interval(UBits(550, 20), UBits(600, 20)),
                                      Interval(UBits(850, 20), UBits(900, 20)),
                                  }));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(400, 20), UBits(500, 20)),
                                   Interval(UBits(700, 20), UBits(800, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetKnownBits(expr.node()), UBits(1, 1));
  EXPECT_EQ(engine.GetKnownBitsValues(expr.node()), UBits(1, 1));
}

TEST_F(RangeQueryEngineTest, Neg) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", fb.package()->GetBitsType(20));
  BValue expr = fb.Negate(x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(SBits(600, 20), SBits(700, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));

  // Negation is antitone.
  EXPECT_EQ(engine.GetIntervalSetTree(expr.node()),
            BitsLTT(x.node(), {Interval(SBits(-700, 20), SBits(-600, 20))}));
}

TEST_F(RangeQueryEngineTest, Param) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", fb.package()->GetBitsType(20));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  // The interval set of a param should be a single interval covering all bit
  // patterns with the bit count of the param.
  EXPECT_EQ(engine.GetIntervalSetTree(x.node()),
            BitsLTT(x.node(), {Interval(Bits(20), Bits::AllOnes(20))}));
}

TEST_F(RangeQueryEngineTest, SignExtend) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", fb.package()->GetBitsType(20));
  BValue expr = fb.SignExtend(x, 40);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(SBits(-500, 20), SBits(700, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));

  // Sign extension is monotone.
  EXPECT_EQ(engine.GetIntervalSetTree(expr.node()),
            BitsLTT(expr.node(), {Interval(SBits(-500, 40), SBits(700, 40))}));
}

TEST_F(RangeQueryEngineTest, Sub) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", fb.package()->GetBitsType(20));
  BValue y = fb.Param("y", fb.package()->GetBitsType(20));
  BValue expr = fb.Subtract(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(500, 20), UBits(1200, 20))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(200, 20), UBits(400, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));

  // Subtraction is monotone-antitone.
  EXPECT_EQ(engine.GetIntervalSetTree(expr.node()),
            BitsLTT(expr.node(), {Interval(UBits(100, 20), UBits(1000, 20))}));
}

TEST_F(RangeQueryEngineTest, UDiv) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", fb.package()->GetBitsType(20));
  BValue y = fb.Param("y", fb.package()->GetBitsType(20));
  BValue expr = fb.UDiv(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  engine.SetIntervalSetTree(
      x.node(),
      BitsLTT(x.node(), {Interval(UBits(16384, 20), UBits(32768, 20))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(4, 20), UBits(16, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));

  // Unsigned division is monotone-antitone.
  EXPECT_EQ(engine.GetIntervalSetTree(expr.node()),
            BitsLTT(expr.node(), {Interval(UBits(1024, 20), UBits(8192, 20))}));
}

TEST_F(RangeQueryEngineTest, UGe) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", fb.package()->GetBitsType(20));
  BValue y = fb.Param("y", fb.package()->GetBitsType(20));
  BValue expr = fb.UGe(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;

  // When the interval sets are disjoint and greater than, returns true.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(200, 20), UBits(300, 20)),
                                   Interval(UBits(500, 20), UBits(600, 20))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(50, 20), UBits(100, 20)),
                                   Interval(UBits(120, 20), UBits(180, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetKnownBits(expr.node()), UBits(1, 1));
  EXPECT_EQ(engine.GetKnownBitsValues(expr.node()), UBits(1, 1));

  // When the interval sets are precise and equal, returns true.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {
                                      Interval(UBits(200, 20), UBits(200, 20)),
                                  }));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {
                                      Interval(UBits(200, 20), UBits(200, 20)),
                                  }));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetKnownBits(expr.node()), UBits(1, 1));
  EXPECT_EQ(engine.GetKnownBitsValues(expr.node()), UBits(1, 1));

  // When the interval sets are disjoint and less than, returns false.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(50, 20), UBits(100, 20)),
                                   Interval(UBits(120, 20), UBits(180, 20))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(200, 20), UBits(300, 20)),
                                   Interval(UBits(500, 20), UBits(600, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetKnownBits(expr.node()), UBits(1, 1));
  EXPECT_EQ(engine.GetKnownBitsValues(expr.node()), UBits(0, 1));

  // When the convex hulls overlap, we don't know anything.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {
                                      Interval(UBits(100, 20), UBits(200, 20)),
                                      Interval(UBits(550, 20), UBits(600, 20)),
                                      Interval(UBits(850, 20), UBits(900, 20)),
                                  }));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(400, 20), UBits(500, 20)),
                                   Interval(UBits(700, 20), UBits(800, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetKnownBits(expr.node()), UBits(0, 1));
  EXPECT_EQ(engine.GetKnownBitsValues(expr.node()), UBits(0, 1));
}

TEST_F(RangeQueryEngineTest, UGt) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", fb.package()->GetBitsType(20));
  BValue y = fb.Param("y", fb.package()->GetBitsType(20));
  BValue expr = fb.UGt(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;

  // When the interval sets are disjoint and greater than, returns true.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(200, 20), UBits(300, 20)),
                                   Interval(UBits(500, 20), UBits(600, 20))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(50, 20), UBits(100, 20)),
                                   Interval(UBits(120, 20), UBits(180, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetKnownBits(expr.node()), UBits(1, 1));
  EXPECT_EQ(engine.GetKnownBitsValues(expr.node()), UBits(1, 1));

  // When the interval sets are precise and equal, returns unknown.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {
                                      Interval(UBits(200, 20), UBits(200, 20)),
                                  }));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {
                                      Interval(UBits(200, 20), UBits(200, 20)),
                                  }));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetKnownBits(expr.node()), UBits(0, 1));
  EXPECT_EQ(engine.GetKnownBitsValues(expr.node()), UBits(0, 1));

  // When the interval sets are disjoint and less than, returns false.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(50, 20), UBits(100, 20)),
                                   Interval(UBits(120, 20), UBits(180, 20))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(200, 20), UBits(300, 20)),
                                   Interval(UBits(500, 20), UBits(600, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetKnownBits(expr.node()), UBits(1, 1));
  EXPECT_EQ(engine.GetKnownBitsValues(expr.node()), UBits(0, 1));

  // When the convex hulls overlap, we don't know anything.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {
                                      Interval(UBits(100, 20), UBits(200, 20)),
                                      Interval(UBits(550, 20), UBits(600, 20)),
                                      Interval(UBits(850, 20), UBits(900, 20)),
                                  }));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(400, 20), UBits(500, 20)),
                                   Interval(UBits(700, 20), UBits(800, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetKnownBits(expr.node()), UBits(0, 1));
  EXPECT_EQ(engine.GetKnownBitsValues(expr.node()), UBits(0, 1));
}

TEST_F(RangeQueryEngineTest, ULe) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", fb.package()->GetBitsType(20));
  BValue y = fb.Param("y", fb.package()->GetBitsType(20));
  BValue expr = fb.ULe(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;

  // When the interval sets are disjoint and greater than, returns false.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(200, 20), UBits(300, 20)),
                                   Interval(UBits(500, 20), UBits(600, 20))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(50, 20), UBits(100, 20)),
                                   Interval(UBits(120, 20), UBits(180, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetKnownBits(expr.node()), UBits(1, 1));
  EXPECT_EQ(engine.GetKnownBitsValues(expr.node()), UBits(0, 1));

  // When the interval sets are precise and equal, returns true.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {
                                      Interval(UBits(200, 20), UBits(200, 20)),
                                  }));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {
                                      Interval(UBits(200, 20), UBits(200, 20)),
                                  }));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetKnownBits(expr.node()), UBits(1, 1));
  EXPECT_EQ(engine.GetKnownBitsValues(expr.node()), UBits(1, 1));

  // When the interval sets are disjoint and less than, returns true.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(50, 20), UBits(100, 20)),
                                   Interval(UBits(120, 20), UBits(180, 20))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(200, 20), UBits(300, 20)),
                                   Interval(UBits(500, 20), UBits(600, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetKnownBits(expr.node()), UBits(1, 1));
  EXPECT_EQ(engine.GetKnownBitsValues(expr.node()), UBits(1, 1));

  // When the convex hulls overlap, we don't know anything.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {
                                      Interval(UBits(100, 20), UBits(200, 20)),
                                      Interval(UBits(550, 20), UBits(600, 20)),
                                      Interval(UBits(850, 20), UBits(900, 20)),
                                  }));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(400, 20), UBits(500, 20)),
                                   Interval(UBits(700, 20), UBits(800, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetKnownBits(expr.node()), UBits(0, 1));
  EXPECT_EQ(engine.GetKnownBitsValues(expr.node()), UBits(0, 1));
}

TEST_F(RangeQueryEngineTest, ULt) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", fb.package()->GetBitsType(20));
  BValue y = fb.Param("y", fb.package()->GetBitsType(20));
  BValue expr = fb.ULt(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;

  // When the interval sets are disjoint and greater than, returns false.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(200, 20), UBits(300, 20)),
                                   Interval(UBits(500, 20), UBits(600, 20))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(50, 20), UBits(100, 20)),
                                   Interval(UBits(120, 20), UBits(180, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetKnownBits(expr.node()), UBits(1, 1));
  EXPECT_EQ(engine.GetKnownBitsValues(expr.node()), UBits(0, 1));

  // When the interval sets are precise and equal, returns unknown.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {
                                      Interval(UBits(200, 20), UBits(200, 20)),
                                  }));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {
                                      Interval(UBits(200, 20), UBits(200, 20)),
                                  }));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetKnownBits(expr.node()), UBits(0, 1));
  EXPECT_EQ(engine.GetKnownBitsValues(expr.node()), UBits(0, 1));

  // When the interval sets are disjoint and less than, returns true.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(50, 20), UBits(100, 20)),
                                   Interval(UBits(120, 20), UBits(180, 20))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(200, 20), UBits(300, 20)),
                                   Interval(UBits(500, 20), UBits(600, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetKnownBits(expr.node()), UBits(1, 1));
  EXPECT_EQ(engine.GetKnownBitsValues(expr.node()), UBits(1, 1));

  // When the convex hulls overlap, we don't know anything.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {
                                      Interval(UBits(100, 20), UBits(200, 20)),
                                      Interval(UBits(550, 20), UBits(600, 20)),
                                      Interval(UBits(850, 20), UBits(900, 20)),
                                  }));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(400, 20), UBits(500, 20)),
                                   Interval(UBits(700, 20), UBits(800, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetKnownBits(expr.node()), UBits(0, 1));
  EXPECT_EQ(engine.GetKnownBitsValues(expr.node()), UBits(0, 1));
}

TEST_F(RangeQueryEngineTest, UMul) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", fb.package()->GetBitsType(7));
  BValue y = fb.Param("y", fb.package()->GetBitsType(7));
  BValue expr = fb.UMul(x, y, 14);
  BValue overflow = fb.UMul(x, y, 12);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(50, 7), UBits(100, 7))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(10, 7), UBits(20, 7))}));
  XLS_ASSERT_OK(engine.Populate(f));

  // Unsigned multiplication is monotone-monotone.
  EXPECT_EQ(engine.GetIntervalSetTree(expr.node()),
            BitsLTT(expr.node(), {Interval(UBits(500, 14), UBits(2000, 14))}));

  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(62, 7), UBits(67, 7))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(61, 7), UBits(70, 7))}));
  XLS_ASSERT_OK(engine.Populate(f));

  // If the multiplication can overflow, the maximal range is inferred
  EXPECT_EQ(engine.GetIntervalSetTree(overflow.node()),
            BitsLTT(overflow.node(), {Interval::Maximal(12)}));
}

TEST_F(RangeQueryEngineTest, ZeroExtend) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", fb.package()->GetBitsType(20));
  BValue expr = fb.ZeroExtend(x, 40);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(500, 20), UBits(700, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));

  // Zero extension is monotone.
  EXPECT_EQ(engine.GetIntervalSetTree(expr.node()),
            BitsLTT(expr.node(), {Interval(UBits(500, 40), UBits(700, 40))}));
}

}  // namespace
}  // namespace xls
