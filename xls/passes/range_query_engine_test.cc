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

IntervalSet RandomIntervalSet(uint32_t seed, int64_t bit_count) {
  return IntervalSet::Random(seed, bit_count, 30);
}

IntervalSet CreateIntervalSet(
    int64_t bit_count, absl::Span<const std::pair<int64_t, int64_t>> bounds) {
  XLS_CHECK(!bounds.empty());
  IntervalSet interval_set(bit_count);
  for (const auto& [lb, ub] : bounds) {
    interval_set.AddInterval(
        Interval(UBits(lb, bit_count), UBits(ub, bit_count)));
  }
  interval_set.Normalize();
  return interval_set;
}

LeafTypeTree<IntervalSet> BitsLTT(
    Node* node, absl::Span<const std::pair<int64_t, int64_t>> bounds) {
  XLS_CHECK(node->GetType()->IsBits());
  LeafTypeTree<IntervalSet> result(node->GetType());
  result.Set({}, CreateIntervalSet(node->BitCountOrDie(), bounds));
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

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue y = fb.Param("y", p->GetBitsType(20));
  BValue expr = fb.Add(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(0, 20), UBits(48, 20))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(0, 20), UBits(15, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));

  // We know that at most 6 bits are used, since 48 + 15 = 63 <= 2^6 - 1
  EXPECT_EQ("0b0000_0000_0000_00XX_XXXX", engine.ToString(expr.node()));

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
  EXPECT_EQ("0bXXXX_XXXX_XXXX_XXXX_XXXX", engine.ToString(expr.node()));

  EXPECT_EQ(IntervalSetTreeToString(engine.GetIntervalSetTree(expr.node())),
            "[[0, 1048575]]");
}

TEST_F(RangeQueryEngineTest, Array) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(10));
  BValue y = fb.Param("y", p->GetBitsType(10));
  BValue expr = fb.Array({x, y}, p->GetBitsType(10));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;

  IntervalSetTree x_ist =
      BitsLTT(x.node(), {Interval(UBits(100, 10), UBits(150, 10))});
  IntervalSetTree y_ist =
      BitsLTT(y.node(), {Interval(UBits(200, 10), UBits(250, 10))});

  engine.SetIntervalSetTree(x.node(), x_ist);
  engine.SetIntervalSetTree(y.node(), y_ist);
  XLS_ASSERT_OK(engine.Populate(f));

  EXPECT_EQ(x_ist.Get({}), engine.GetIntervalSetTree(expr.node()).Get({0}));
  EXPECT_EQ(y_ist.Get({}), engine.GetIntervalSetTree(expr.node()).Get({1}));

  EXPECT_EQ(IntervalSetTreeToString(engine.GetIntervalSetTree(expr.node())),
            R"([
  [[100, 150]]
  [[200, 250]]
]
)");
}

TEST_F(RangeQueryEngineTest, ArrayConcat) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(10));
  BValue y = fb.Param("y", p->GetBitsType(10));
  BValue z = fb.Param("z", p->GetBitsType(10));
  BValue w = fb.Param("w", p->GetBitsType(10));
  BValue array1 = fb.Array({x, y}, p->GetBitsType(10));
  BValue array2 = fb.Array({z, w}, p->GetBitsType(10));
  BValue expr = fb.ArrayConcat({array1, array2, array1});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;

  IntervalSetTree x_ist =
      BitsLTT(x.node(), {Interval(UBits(100, 10), UBits(150, 10))});
  IntervalSetTree y_ist =
      BitsLTT(y.node(), {Interval(UBits(200, 10), UBits(250, 10))});
  IntervalSetTree z_ist =
      BitsLTT(z.node(), {Interval(UBits(300, 10), UBits(350, 10))});
  IntervalSetTree w_ist =
      BitsLTT(w.node(), {Interval(UBits(400, 10), UBits(450, 10))});

  engine.SetIntervalSetTree(x.node(), x_ist);
  engine.SetIntervalSetTree(y.node(), y_ist);
  engine.SetIntervalSetTree(z.node(), z_ist);
  engine.SetIntervalSetTree(w.node(), w_ist);
  XLS_ASSERT_OK(engine.Populate(f));

  EXPECT_EQ(x_ist.Get({}), engine.GetIntervalSetTree(expr.node()).Get({0}));
  EXPECT_EQ(y_ist.Get({}), engine.GetIntervalSetTree(expr.node()).Get({1}));
  EXPECT_EQ(z_ist.Get({}), engine.GetIntervalSetTree(expr.node()).Get({2}));
  EXPECT_EQ(w_ist.Get({}), engine.GetIntervalSetTree(expr.node()).Get({3}));
  EXPECT_EQ(x_ist.Get({}), engine.GetIntervalSetTree(expr.node()).Get({4}));
  EXPECT_EQ(y_ist.Get({}), engine.GetIntervalSetTree(expr.node()).Get({5}));
}

TEST_F(RangeQueryEngineTest, ArrayIndex1D) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(10));
  BValue y = fb.Param("y", p->GetBitsType(10));
  BValue z = fb.Param("z", p->GetBitsType(10));
  BValue array = fb.Array({x, y, z}, p->GetBitsType(10));
  BValue index0 = fb.ArrayIndex(array, {fb.Literal(UBits(0, 2))});
  BValue index1 = fb.ArrayIndex(array, {fb.Literal(UBits(1, 2))});
  BValue index2 = fb.ArrayIndex(array, {fb.Literal(UBits(2, 2))});
  BValue oob_index = fb.ArrayIndex(array, {fb.Literal(UBits(3, 2))});
  BValue i = fb.Param("i", p->GetBitsType(2));
  BValue index12 = fb.ArrayIndex(array, {i});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;

  IntervalSetTree x_ist =
      BitsLTT(x.node(), {Interval(UBits(100, 10), UBits(150, 10))});
  IntervalSetTree y_ist =
      BitsLTT(y.node(), {Interval(UBits(200, 10), UBits(250, 10))});
  IntervalSetTree z_ist =
      BitsLTT(z.node(), {Interval(UBits(300, 10), UBits(350, 10))});
  IntervalSetTree i_ist =
      BitsLTT(i.node(), {Interval(UBits(1, 2), UBits(2, 2))});

  engine.SetIntervalSetTree(x.node(), x_ist);
  engine.SetIntervalSetTree(y.node(), y_ist);
  engine.SetIntervalSetTree(z.node(), z_ist);
  engine.SetIntervalSetTree(i.node(), i_ist);
  XLS_ASSERT_OK(engine.Populate(f));

  EXPECT_EQ(x_ist.Get({}), engine.GetIntervalSetTree(index0.node()).Get({}));
  EXPECT_EQ(y_ist.Get({}), engine.GetIntervalSetTree(index1.node()).Get({}));
  EXPECT_EQ(z_ist.Get({}), engine.GetIntervalSetTree(index2.node()).Get({}));
  // An out-of-bounds index returns the highest-index element.
  EXPECT_EQ(z_ist.Get({}), engine.GetIntervalSetTree(oob_index.node()).Get({}));
  EXPECT_EQ(IntervalSet::Combine(y_ist.Get({}), z_ist.Get({})),
            engine.GetIntervalSetTree(index12.node()).Get({}));
}

TEST_F(RangeQueryEngineTest, ArrayIndex3D) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(10));
  BValue y = fb.Param("y", p->GetBitsType(10));
  BValue z = fb.Param("z", p->GetBitsType(10));
  // Dimension = [5, 6, 3]
  BValue array = fb.Literal(Value::ArrayOrDie({*Value::UBits2DArray(
                                                   {
                                                       {1, 2, 3},
                                                       {4, 5, 6},
                                                       {7, 8, 9},
                                                       {10, 11, 12},
                                                       {13, 14, 15},
                                                       {16, 17, 18},
                                                   },
                                                   32),
                                               *Value::UBits2DArray(
                                                   {
                                                       {19, 20, 21},
                                                       {22, 23, 24},
                                                       {25, 26, 27},
                                                       {28, 29, 30},
                                                       {31, 32, 33},
                                                       {34, 35, 36},
                                                   },
                                                   32),
                                               *Value::UBits2DArray(
                                                   {
                                                       {37, 38, 39},
                                                       {40, 41, 42},
                                                       {43, 44, 45},
                                                       {46, 47, 48},
                                                       {49, 50, 51},
                                                       {52, 53, 54},
                                                   },
                                                   32),
                                               *Value::UBits2DArray(
                                                   {
                                                       {55, 56, 57},
                                                       {58, 59, 60},
                                                       {61, 62, 63},
                                                       {64, 65, 66},
                                                       {67, 68, 69},
                                                       {70, 71, 72},
                                                   },
                                                   32),
                                               *Value::UBits2DArray(
                                                   {
                                                       {73, 74, 75},
                                                       {76, 77, 78},
                                                       {79, 80, 81},
                                                       {82, 83, 84},
                                                       {85, 86, 87},
                                                       {88, 89, 90},
                                                   },
                                                   32)}));
  BValue index = fb.ArrayIndex(array, {x, y, z});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;

  IntervalSetTree x_ist =
      BitsLTT(x.node(), {Interval(UBits(2, 10), UBits(3, 10))});
  IntervalSetTree y_ist =
      BitsLTT(y.node(), {Interval(UBits(2, 10), UBits(4, 10))});
  IntervalSetTree z_ist =
      BitsLTT(z.node(), {Interval(UBits(1, 10), UBits(1, 10))});

  engine.SetIntervalSetTree(x.node(), x_ist);
  engine.SetIntervalSetTree(y.node(), y_ist);
  engine.SetIntervalSetTree(z.node(), z_ist);
  XLS_ASSERT_OK(engine.Populate(f));

  // array[2, 2, 1] = 44
  // array[2, 3, 1] = 47
  // array[2, 4, 1] = 50
  // array[3, 2, 1] = 62
  // array[3, 3, 1] = 65
  // array[3, 4, 1] = 68

  IntervalSet expected(32);
  expected.AddInterval(Interval::Precise(UBits(44, 32)));
  expected.AddInterval(Interval::Precise(UBits(47, 32)));
  expected.AddInterval(Interval::Precise(UBits(50, 32)));
  expected.AddInterval(Interval::Precise(UBits(62, 32)));
  expected.AddInterval(Interval::Precise(UBits(65, 32)));
  expected.AddInterval(Interval::Precise(UBits(68, 32)));

  EXPECT_EQ(expected, engine.GetIntervalSetTree(index.node()).Get({}));
}

TEST_F(RangeQueryEngineTest, AndReduce) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue expr = fb.AndReduce(x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;

  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(0, 20), UBits(48, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));

  // Interval does not cover 2^20 - 1, so the result is known to be 0.
  EXPECT_EQ("0b0", engine.ToString(expr.node()));

  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(),
      BitsLTT(x.node(), {Interval(UBits(500, 20), Bits::AllOnes(20))}));
  XLS_ASSERT_OK(engine.Populate(f));

  // Interval does cover 2^20 - 1, so we don't know the value.
  EXPECT_EQ("0bX", engine.ToString(expr.node()));

  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(),
      BitsLTT(x.node(), {Interval(Bits::AllOnes(20), Bits::AllOnes(20))}));
  XLS_ASSERT_OK(engine.Populate(f));

  // Interval only covers 2^20 - 1, so the result is known to be 1.
  EXPECT_EQ("0b1", engine.ToString(expr.node()));
}

TEST_F(RangeQueryEngineTest, Concat) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(6));
  BValue y = fb.Param("y", p->GetBitsType(5));
  BValue z = fb.Param("z", p->GetBitsType(3));
  BValue expr = fb.Concat({x, y, z});

  IntervalSet x_intervals = RandomIntervalSet(802103005, 6);
  IntervalSet y_intervals = RandomIntervalSet(802103006, 5);
  IntervalSet z_intervals = RandomIntervalSet(802103007, 3);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  engine.SetIntervalSetTree(x.node(),
                            BitsLTT(x.node(), x_intervals.Intervals()));
  engine.SetIntervalSetTree(y.node(),
                            BitsLTT(y.node(), y_intervals.Intervals()));
  engine.SetIntervalSetTree(z.node(),
                            BitsLTT(z.node(), z_intervals.Intervals()));
  XLS_ASSERT_OK(engine.Populate(f));

  x_intervals.ForEachElement([&](const Bits& bits_x) -> bool {
    y_intervals.ForEachElement([&](const Bits& bits_y) -> bool {
      z_intervals.ForEachElement([&](const Bits& bits_z) -> bool {
        Bits concatenated = bits_ops::Concat({bits_x, bits_y, bits_z});
        EXPECT_TRUE(engine.GetIntervalSetTree(expr.node())
                        .Get({})
                        .Covers(concatenated));
        return false;
      });
      return false;
    });
    return false;
  });
}

TEST_F(RangeQueryEngineTest, Eq) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue y = fb.Param("y", p->GetBitsType(20));
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
  EXPECT_EQ("0b1", engine.ToString(expr.node()));

  // When the interval sets overlap, we don't know anything about them.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(100, 20), UBits(200, 20))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(150, 20), UBits(300, 20)),
                                   Interval(UBits(400, 20), UBits(500, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ("0bX", engine.ToString(expr.node()));

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
  EXPECT_EQ("0b0", engine.ToString(expr.node()));
}

TEST_F(RangeQueryEngineTest, Gate) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue cond = fb.Param("cond", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue expr = fb.Gate(cond, x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  LeafTypeTree<IntervalSet> interval_set =
      BitsLTT(x.node(), {Interval(UBits(600, 20), UBits(700, 20))});
  engine.SetIntervalSetTree(x.node(), interval_set);
  XLS_ASSERT_OK(engine.Populate(f));

  // Gate can either produce zero or the data operand value.
  EXPECT_EQ(engine.GetIntervalSetTree(expr.node()),
            BitsLTT(x.node(), {{0, 0}, {600, 700}}));
}

TEST_F(RangeQueryEngineTest, GateWithConditionTrue) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue cond = fb.Literal(Value(UBits(1, 1)));
  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue expr = fb.Gate(cond, x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  engine.SetIntervalSetTree(x.node(), BitsLTT(x.node(), {{600, 700}}));
  XLS_ASSERT_OK(engine.Populate(f));

  // With the gated condition true the gate operation produces the original
  // value.
  EXPECT_EQ(engine.GetIntervalSetTree(expr.node()),
            BitsLTT(x.node(), {{600, 700}}));
}

TEST_F(RangeQueryEngineTest, GateWithConditionFalse) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue cond = fb.Literal(Value(UBits(0, 1)));
  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue expr = fb.Gate(cond, x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  engine.SetIntervalSetTree(x.node(), BitsLTT(x.node(), {{600, 700}}));
  XLS_ASSERT_OK(engine.Populate(f));

  // With the gated condition false the gate operation is the identity.
  EXPECT_EQ(engine.GetIntervalSetTree(expr.node()),
            BitsLTT(x.node(), {{0, 0}}));
}

TEST_F(RangeQueryEngineTest, GateWithCompoundType) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue cond = fb.Param("cond", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(10));
  BValue y = fb.Param("y", p->GetBitsType(20));
  BValue tuple = fb.Tuple({x, y, y});
  BValue gate = fb.Gate(cond, tuple);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;

  engine.SetIntervalSetTree(x.node(), BitsLTT(x.node(), {{100, 150}}));
  engine.SetIntervalSetTree(y.node(), BitsLTT(y.node(), {{20000, 25000}}));
  XLS_ASSERT_OK(engine.Populate(f));

  EXPECT_EQ(engine.GetIntervalSetTree(gate.node()).Get({0}),
            CreateIntervalSet(10, {{0, 0}, {100, 150}}));
  EXPECT_EQ(engine.GetIntervalSetTree(gate.node()).Get({1}),
            CreateIntervalSet(20, {{0, 0}, {20000, 25000}}));
  EXPECT_EQ(engine.GetIntervalSetTree(gate.node()).Get({2}),
            CreateIntervalSet(20, {{0, 0}, {20000, 25000}}));
}

TEST_F(RangeQueryEngineTest, Identity) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
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

TEST_F(RangeQueryEngineTest, LiteralSimple) {
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

TEST_F(RangeQueryEngineTest, LiteralNonBits) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue lit = fb.Literal(
      Value::TupleOwned({Value(UBits(100, 123)), Value(UBits(140, 200)),
                         *Value::UBitsArray({27, 53, 62}, 10)}));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  // The interval set of a literal should be a single interval starting and
  // ending at its value.
  LeafTypeTree<IntervalSet> expected(lit.node()->GetType());
  expected.Set({0}, IntervalSet::Precise(UBits(100, 123)));
  expected.Set({1}, IntervalSet::Precise(UBits(140, 200)));
  expected.Set({2, 0}, IntervalSet::Precise(UBits(27, 10)));
  expected.Set({2, 1}, IntervalSet::Precise(UBits(53, 10)));
  expected.Set({2, 2}, IntervalSet::Precise(UBits(62, 10)));

  EXPECT_EQ(engine.GetIntervalSetTree(lit.node()), expected);
}

TEST_F(RangeQueryEngineTest, Ne) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue y = fb.Param("y", p->GetBitsType(20));
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
  EXPECT_EQ("0b0", engine.ToString(expr.node()));

  // When the interval sets overlap, we don't know anything about them.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(100, 20), UBits(200, 20))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(150, 20), UBits(300, 20)),
                                   Interval(UBits(400, 20), UBits(500, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ("0bX", engine.ToString(expr.node()));

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
  EXPECT_EQ("0b1", engine.ToString(expr.node()));
}

TEST_F(RangeQueryEngineTest, Neg) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
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

TEST_F(RangeQueryEngineTest, OrReduce) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue expr = fb.OrReduce(x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;

  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(1, 20), UBits(48, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));

  // Interval does not cover 0, so the result is known to be 1.
  EXPECT_EQ("0b1", engine.ToString(expr.node()));

  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(0, 20), UBits(48, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));

  // Interval does cover 0, so we don't know the value.
  EXPECT_EQ("0bX", engine.ToString(expr.node()));

  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(0, 20), UBits(0, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));

  // Interval only covers 0, so the result is known to be 0.
  EXPECT_EQ("0b0", engine.ToString(expr.node()));
}

TEST_F(RangeQueryEngineTest, Param) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  // The interval set of a param should be a single interval covering all bit
  // patterns with the bit count of the param.
  EXPECT_EQ(engine.GetIntervalSetTree(x.node()),
            BitsLTT(x.node(), {Interval(Bits(20), Bits::AllOnes(20))}));
}

TEST_F(RangeQueryEngineTest, Sel) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue selector = fb.Param("selector", p->GetBitsType(10));
  BValue x = fb.Param("x", p->GetBitsType(10));
  BValue y = fb.Param("y", p->GetBitsType(10));
  BValue z = fb.Param("z", p->GetBitsType(10));
  BValue def = fb.Param("def", p->GetBitsType(10));
  BValue expr = fb.Select(selector, {x, y, z}, def);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;

  IntervalSetTree x_ist =
      BitsLTT(x.node(), {Interval(UBits(100, 10), UBits(150, 10))});
  IntervalSetTree y_ist =
      BitsLTT(y.node(), {Interval(UBits(200, 10), UBits(250, 10))});
  IntervalSetTree z_ist =
      BitsLTT(y.node(), {Interval(UBits(300, 10), UBits(350, 10))});
  IntervalSetTree def_ist =
      BitsLTT(y.node(), {Interval(UBits(1023, 10), UBits(1023, 10))});

  engine.SetIntervalSetTree(
      selector.node(),
      BitsLTT(selector.node(), {Interval(UBits(0, 10), UBits(1, 10))}));
  engine.SetIntervalSetTree(x.node(), x_ist);
  engine.SetIntervalSetTree(y.node(), y_ist);
  engine.SetIntervalSetTree(z.node(), z_ist);
  engine.SetIntervalSetTree(def.node(), def_ist);
  XLS_ASSERT_OK(engine.Populate(f));

  // Usual test case where only part of the valid inputs are covered.
  EXPECT_EQ(IntervalSet::Combine(x_ist.Get({}), y_ist.Get({})),
            engine.GetIntervalSetTree(expr.node()).Get({}));

  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      selector.node(),
      BitsLTT(selector.node(), {Interval(UBits(2, 10), UBits(10, 10))}));
  engine.SetIntervalSetTree(x.node(), x_ist);
  engine.SetIntervalSetTree(y.node(), y_ist);
  engine.SetIntervalSetTree(z.node(), z_ist);
  engine.SetIntervalSetTree(def.node(), def_ist);
  XLS_ASSERT_OK(engine.Populate(f));

  // Test case where default is covered.
  EXPECT_EQ(IntervalSet::Combine(z_ist.Get({}), def_ist.Get({})),
            engine.GetIntervalSetTree(expr.node()).Get({}));
}

TEST_F(RangeQueryEngineTest, PrioritySel) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue selector = fb.Param("selector", p->GetBitsType(3));
  BValue x = fb.Param("x", p->GetBitsType(10));
  BValue y = fb.Param("y", p->GetBitsType(10));
  BValue z = fb.Param("z", p->GetBitsType(10));
  BValue expr = fb.PrioritySelect(selector, {x, y, z});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  // Usual test case where only part of the valid inputs are covered.
  RangeQueryEngine engine;

  IntervalSetTree x_ist =
      BitsLTT(x.node(), {Interval(UBits(100, 10), UBits(150, 10))});
  IntervalSetTree y_ist =
      BitsLTT(y.node(), {Interval(UBits(200, 10), UBits(250, 10))});
  IntervalSetTree z_ist =
      BitsLTT(y.node(), {Interval(UBits(300, 10), UBits(350, 10))});

  engine.SetIntervalSetTree(
      selector.node(),
      BitsLTT(selector.node(), {Interval(UBits(1, 3), UBits(2, 3))}));
  engine.SetIntervalSetTree(x.node(), x_ist);
  engine.SetIntervalSetTree(y.node(), y_ist);
  engine.SetIntervalSetTree(z.node(), z_ist);
  XLS_ASSERT_OK(engine.Populate(f));

  EXPECT_EQ(IntervalSet::Combine(x_ist.Get({}), y_ist.Get({})),
            engine.GetIntervalSetTree(expr.node()).Get({}));

  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      selector.node(),
      BitsLTT(selector.node(), {Interval(UBits(1, 3), UBits(4, 3))}));
  engine.SetIntervalSetTree(x.node(), x_ist);
  engine.SetIntervalSetTree(y.node(), y_ist);
  engine.SetIntervalSetTree(z.node(), z_ist);
  XLS_ASSERT_OK(engine.Populate(f));

  EXPECT_EQ(
      IntervalSet::Combine(IntervalSet::Combine(x_ist.Get({}), y_ist.Get({})),
                           z_ist.Get({})),
      engine.GetIntervalSetTree(expr.node()).Get({}));

  // Test case with overlapping bits for selector.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      selector.node(),
      BitsLTT(selector.node(), {Interval(UBits(2, 3), UBits(6, 3))}));
  engine.SetIntervalSetTree(x.node(), x_ist);
  engine.SetIntervalSetTree(y.node(), y_ist);
  engine.SetIntervalSetTree(z.node(), z_ist);
  XLS_ASSERT_OK(engine.Populate(f));

  EXPECT_EQ(IntervalSet::Combine(y_ist.Get({}), z_ist.Get({})),
            engine.GetIntervalSetTree(expr.node()).Get({}));

  // Test case where default is covered.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      selector.node(),
      BitsLTT(selector.node(), {Interval(UBits(0, 3), UBits(1, 3))}));
  engine.SetIntervalSetTree(x.node(), x_ist);
  engine.SetIntervalSetTree(y.node(), y_ist);
  engine.SetIntervalSetTree(z.node(), z_ist);
  XLS_ASSERT_OK(engine.Populate(f));

  EXPECT_EQ(
      IntervalSet::Combine(x_ist.Get({}), IntervalSet::Precise(UBits(0, 10))),
      engine.GetIntervalSetTree(expr.node()).Get({}));
}

TEST_F(RangeQueryEngineTest, SignExtend) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
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

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue y = fb.Param("y", p->GetBitsType(20));
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

TEST_F(RangeQueryEngineTest, Tuple) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(10));
  BValue y = fb.Param("y", p->GetBitsType(20));
  BValue expr = fb.Tuple({x, y, y});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;

  IntervalSetTree x_ist =
      BitsLTT(x.node(), {Interval(UBits(100, 10), UBits(150, 10))});
  IntervalSetTree y_ist =
      BitsLTT(y.node(), {Interval(UBits(20000, 20), UBits(25000, 20))});

  engine.SetIntervalSetTree(x.node(), x_ist);
  engine.SetIntervalSetTree(y.node(), y_ist);
  XLS_ASSERT_OK(engine.Populate(f));

  EXPECT_EQ(x_ist.Get({}), engine.GetIntervalSetTree(expr.node()).Get({0}));
  EXPECT_EQ(y_ist.Get({}), engine.GetIntervalSetTree(expr.node()).Get({1}));
  EXPECT_EQ(y_ist.Get({}), engine.GetIntervalSetTree(expr.node()).Get({2}));

  EXPECT_EQ(IntervalSetTreeToString(engine.GetIntervalSetTree(expr.node())),
            R"((
  [[100, 150]]
  [[20000, 25000]]
  [[20000, 25000]]
)
)");
}

TEST_F(RangeQueryEngineTest, TupleIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(10));
  BValue y = fb.Param("y", p->GetBitsType(20));
  BValue tuple = fb.Tuple({x, y, y});
  BValue index0 = fb.TupleIndex(tuple, 0);
  BValue index1 = fb.TupleIndex(tuple, 1);
  BValue index2 = fb.TupleIndex(tuple, 2);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;

  IntervalSetTree x_ist =
      BitsLTT(x.node(), {Interval(UBits(100, 10), UBits(150, 10))});
  IntervalSetTree y_ist =
      BitsLTT(y.node(), {Interval(UBits(20000, 20), UBits(25000, 20))});

  engine.SetIntervalSetTree(x.node(), x_ist);
  engine.SetIntervalSetTree(y.node(), y_ist);
  XLS_ASSERT_OK(engine.Populate(f));

  EXPECT_EQ(x_ist, engine.GetIntervalSetTree(index0.node()));
  EXPECT_EQ(y_ist, engine.GetIntervalSetTree(index1.node()));
  EXPECT_EQ(y_ist, engine.GetIntervalSetTree(index2.node()));
}

TEST_F(RangeQueryEngineTest, UDiv) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue y = fb.Param("y", p->GetBitsType(20));
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

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue y = fb.Param("y", p->GetBitsType(20));
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
  EXPECT_EQ("0b1", engine.ToString(expr.node()));

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
  EXPECT_EQ("0b1", engine.ToString(expr.node()));

  // When the interval sets are disjoint and less than, returns false.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(50, 20), UBits(100, 20)),
                                   Interval(UBits(120, 20), UBits(180, 20))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(200, 20), UBits(300, 20)),
                                   Interval(UBits(500, 20), UBits(600, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ("0b0", engine.ToString(expr.node()));

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
  EXPECT_EQ("0bX", engine.ToString(expr.node()));
}

TEST_F(RangeQueryEngineTest, UGt) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue y = fb.Param("y", p->GetBitsType(20));
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
  EXPECT_EQ("0b1", engine.ToString(expr.node()));

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
  EXPECT_EQ("0bX", engine.ToString(expr.node()));

  // When the interval sets are disjoint and less than, returns false.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(50, 20), UBits(100, 20)),
                                   Interval(UBits(120, 20), UBits(180, 20))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(200, 20), UBits(300, 20)),
                                   Interval(UBits(500, 20), UBits(600, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ("0b0", engine.ToString(expr.node()));

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
  EXPECT_EQ("0bX", engine.ToString(expr.node()));
}

TEST_F(RangeQueryEngineTest, ULe) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue y = fb.Param("y", p->GetBitsType(20));
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
  EXPECT_EQ("0b0", engine.ToString(expr.node()));

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
  EXPECT_EQ("0b1", engine.ToString(expr.node()));

  // When the interval sets are disjoint and less than, returns true.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(50, 20), UBits(100, 20)),
                                   Interval(UBits(120, 20), UBits(180, 20))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(200, 20), UBits(300, 20)),
                                   Interval(UBits(500, 20), UBits(600, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ("0b1", engine.ToString(expr.node()));

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
  EXPECT_EQ("0bX", engine.ToString(expr.node()));
}

TEST_F(RangeQueryEngineTest, ULt) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue y = fb.Param("y", p->GetBitsType(20));
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
  EXPECT_EQ("0b0", engine.ToString(expr.node()));

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
  EXPECT_EQ("0bX", engine.ToString(expr.node()));

  // When the interval sets are disjoint and less than, returns true.
  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(50, 20), UBits(100, 20)),
                                   Interval(UBits(120, 20), UBits(180, 20))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(200, 20), UBits(300, 20)),
                                   Interval(UBits(500, 20), UBits(600, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ("0b1", engine.ToString(expr.node()));

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
  EXPECT_EQ("0bX", engine.ToString(expr.node()));
}

TEST_F(RangeQueryEngineTest, UMul) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(7));
  BValue y = fb.Param("y", p->GetBitsType(7));
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

TEST_F(RangeQueryEngineTest, XorReduce) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue expr = fb.XorReduce(x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;

  engine.SetIntervalSetTree(
      x.node(),
      BitsLTT(x.node(), {Interval(UBits(0b1, 20), UBits(0b1, 20)),
                         Interval(UBits(0b1110, 20), UBits(0b1110, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));

  // Interval set covers only numbers with an odd number of 1s, so result is 1.
  EXPECT_EQ("0b1", engine.ToString(expr.node()));

  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(0, 20), UBits(48, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));

  // Interval is not precise, so result is unknown.
  EXPECT_EQ("0bX", engine.ToString(expr.node()));

  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(),
      BitsLTT(x.node(), {Interval(UBits(0b1110, 20), UBits(0b1110, 20)),
                         Interval(UBits(0b1001, 20), UBits(0b1001, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));

  // Intervals are precise, but don't match parity of 1s, so result is unknown.
  EXPECT_EQ("0bX", engine.ToString(expr.node()));

  engine = RangeQueryEngine();
  engine.SetIntervalSetTree(
      x.node(),
      BitsLTT(x.node(), {Interval(UBits(0b11, 20), UBits(0b11, 20))}));
  XLS_ASSERT_OK(engine.Populate(f));

  // Interval only covers numbers with an even number of 1s, so result is 0.
  EXPECT_EQ("0b0", engine.ToString(expr.node()));
}

TEST_F(RangeQueryEngineTest, ZeroExtend) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
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
