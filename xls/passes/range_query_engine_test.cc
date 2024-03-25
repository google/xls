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

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fuzztest/fuzztest.h"
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/interval_set_test_utils.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/ternary.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"

namespace xls {
namespace {

class RangeQueryEngineTest : public IrTestBase {};

IntervalSet CreateIntervalSet(
    int64_t bit_count, absl::Span<const std::pair<int64_t, int64_t>> bounds) {
  CHECK(!bounds.empty());
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
  CHECK(node->GetType()->IsBits());
  LeafTypeTree<IntervalSet> result(node->GetType());
  result.Set({}, CreateIntervalSet(node->BitCountOrDie(), bounds));
  return result;
}

LeafTypeTree<IntervalSet> BitsLTT(Node* node,
                                  absl::Span<const Interval> intervals) {
  CHECK(!intervals.empty());
  int64_t bit_count = intervals[0].BitCount();
  IntervalSet interval_set(bit_count);
  for (const Interval& interval : intervals) {
    CHECK_EQ(interval.BitCount(), bit_count);
    interval_set.AddInterval(interval);
  }
  interval_set.Normalize();
  CHECK(node->GetType()->IsBits());
  LeafTypeTree<IntervalSet> result(node->GetType());
  result.Set({}, interval_set);
  return result;
}

void MinimizeIntervalsSatisfiesInvariants(const IntervalSet& interval_set,
                                          int64_t size) {
  IntervalSet minimized = MinimizeIntervals(interval_set, size);
  EXPECT_EQ(interval_set.BitCount(), minimized.BitCount());
  EXPECT_LE(minimized.NumberOfIntervals(), size)
      << "interval_set = " << interval_set.ToString() << "\n"
      << "minimized    = " << minimized.ToString() << "\n";

  IntervalSet normalized = interval_set;
  normalized.Normalize();
  EXPECT_LE(minimized.NumberOfIntervals(), normalized.NumberOfIntervals())
      << "normalized interval_set = " << normalized.ToString() << "\n"
      << "              minimized = " << minimized.ToString() << "\n";
}
FUZZ_TEST(RangeQueryEngineFuzzTest, MinimizeIntervalsSatisfiesInvariants)
    .WithDomains(fuzztest::FlatMap(
                     [](int64_t bit_count) {
                       return ArbitraryIntervalSet(bit_count);
                     },
                     fuzztest::InRange(1, 32)),
                 fuzztest::InRange(1, 20));

MATCHER_P(IntervalsAre, expected,
          absl::StrFormat("%smatch interval set of %s",
                          negation ? "doesn't " : "does ",
                          expected.ToString())) {
  return testing::ExplainMatchResult(testing::Eq(expected.type()), arg.type(),
                                     result_listener) &&
         testing::ExplainMatchResult(
             testing::ElementsAreArray(expected.elements()), arg.elements(),
             result_listener);
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

TEST_F(RangeQueryEngineTest, ArraySlice) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue a = fb.Param("a", p->GetBitsType(10));
  BValue b = fb.Param("b", p->GetBitsType(10));
  BValue c = fb.Param("c", p->GetBitsType(10));
  BValue w = fb.Param("w", p->GetBitsType(10));
  BValue x = fb.Param("x", p->GetBitsType(10));
  BValue y = fb.Param("y", p->GetBitsType(10));
  BValue z = fb.Param("z", p->GetBitsType(10));
  BValue array = fb.Array({a, b, c, w, x, y, z}, p->GetBitsType(10));
  BValue slice0_3 = fb.ArraySlice(array, fb.Literal(UBits(0, 3)), 3);
  BValue slice3_4 = fb.ArraySlice(array, fb.Literal(UBits(3, 3)), 4);
  BValue slice4_8 = fb.ArraySlice(array, fb.Literal(UBits(4, 3)), 8);
  BValue i = fb.Param("i", p->GetBitsType(2));
  BValue slice12_2 = fb.ArraySlice(array, i, 2);
  BValue big_i = fb.Param("big_i", p->GetBitsType(20));
  BValue slice_oob = fb.ArraySlice(array, big_i, 2);
  BValue unbound_i = fb.Param("unbound_i", p->GetBitsType(20));
  BValue slice_unbound = fb.ArraySlice(array, unbound_i, 1);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;

  IntervalSetTree a_ist =
      BitsLTT(a.node(), {Interval(UBits(400, 10), UBits(450, 10))});
  IntervalSetTree b_ist =
      BitsLTT(b.node(), {Interval(UBits(500, 10), UBits(550, 10))});
  IntervalSetTree c_ist =
      BitsLTT(c.node(), {Interval(UBits(600, 10), UBits(650, 10))});
  IntervalSetTree w_ist =
      BitsLTT(w.node(), {Interval(UBits(0, 10), UBits(50, 10))});
  IntervalSetTree x_ist =
      BitsLTT(x.node(), {Interval(UBits(100, 10), UBits(150, 10))});
  IntervalSetTree y_ist =
      BitsLTT(y.node(), {Interval(UBits(200, 10), UBits(250, 10))});
  IntervalSetTree z_ist =
      BitsLTT(z.node(), {Interval(UBits(300, 10), UBits(350, 10))});
  IntervalSetTree i_ist =
      BitsLTT(i.node(), {Interval(UBits(1, 2), UBits(2, 2))});
  IntervalSetTree big_i_ist =
      BitsLTT(big_i.node(), {Interval(UBits(20, 20), UBits(30, 20))});

  IntervalSetTree abc_ist =
      IntervalSetTree(p->GetArrayType(3, p->GetBitsType(10)));
  abc_ist.Set({0}, a_ist.Get({}));
  abc_ist.Set({1}, b_ist.Get({}));
  abc_ist.Set({2}, c_ist.Get({}));

  IntervalSetTree wxyz_ist =
      IntervalSetTree(p->GetArrayType(4, p->GetBitsType(10)));
  wxyz_ist.Set({0}, w_ist.Get({}));
  wxyz_ist.Set({1}, x_ist.Get({}));
  wxyz_ist.Set({2}, y_ist.Get({}));
  wxyz_ist.Set({3}, z_ist.Get({}));

  IntervalSetTree xyzzzzzz_ist =
      IntervalSetTree(p->GetArrayType(8, p->GetBitsType(10)));
  xyzzzzzz_ist.Set({0}, x_ist.Get({}));
  xyzzzzzz_ist.Set({1}, y_ist.Get({}));
  xyzzzzzz_ist.Set({2}, z_ist.Get({}));
  xyzzzzzz_ist.Set({3}, z_ist.Get({}));
  xyzzzzzz_ist.Set({4}, z_ist.Get({}));
  xyzzzzzz_ist.Set({5}, z_ist.Get({}));
  xyzzzzzz_ist.Set({6}, z_ist.Get({}));
  xyzzzzzz_ist.Set({7}, z_ist.Get({}));

  IntervalSetTree bc_or_cw_ist =
      IntervalSetTree(p->GetArrayType(2, p->GetBitsType(10)));
  bc_or_cw_ist.Set({0}, IntervalSet::Combine(b_ist.Get({}), c_ist.Get({})));
  bc_or_cw_ist.Set({1}, IntervalSet::Combine(c_ist.Get({}), w_ist.Get({})));

  IntervalSetTree zz_ist =
      IntervalSetTree(p->GetArrayType(2, p->GetBitsType(10)));
  zz_ist.Set({0}, z_ist.Get({}));
  zz_ist.Set({1}, z_ist.Get({}));

  IntervalSetTree unbound_ist =
      IntervalSetTree(p->GetArrayType(1, p->GetBitsType(10)));
  unbound_ist.Set({0}, CreateIntervalSet(10, {{0, 50},
                                              {100, 150},
                                              {200, 250},
                                              {300, 350},
                                              {400, 450},
                                              {500, 550},
                                              {600, 650}}));

  engine.SetIntervalSetTree(a.node(), a_ist);
  engine.SetIntervalSetTree(b.node(), b_ist);
  engine.SetIntervalSetTree(c.node(), c_ist);
  engine.SetIntervalSetTree(w.node(), w_ist);
  engine.SetIntervalSetTree(x.node(), x_ist);
  engine.SetIntervalSetTree(y.node(), y_ist);
  engine.SetIntervalSetTree(z.node(), z_ist);
  engine.SetIntervalSetTree(i.node(), i_ist);
  engine.SetIntervalSetTree(big_i.node(), big_i_ist);
  XLS_ASSERT_OK(engine.Populate(f));

  EXPECT_THAT(engine.GetIntervalSetTree(slice0_3.node()),
              IntervalsAre(abc_ist));
  EXPECT_THAT(engine.GetIntervalSetTree(slice3_4.node()),
              IntervalsAre(wxyz_ist));
  EXPECT_THAT(engine.GetIntervalSetTree(slice4_8.node()),
              IntervalsAre(xyzzzzzz_ist));
  EXPECT_THAT(engine.GetIntervalSetTree(slice12_2.node()),
              IntervalsAre(bc_or_cw_ist));
  EXPECT_THAT(engine.GetIntervalSetTree(slice_unbound.node()),
              IntervalsAre(unbound_ist));
  EXPECT_THAT(engine.GetIntervalSetTree(slice_oob.node()),
              IntervalsAre(zz_ist));
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

void ConcatIsCorrect(const IntervalSet& x_intervals,
                     const IntervalSet& y_intervals,
                     const IntervalSet& z_intervals) {
  constexpr std::string_view kTestName =
      "RangeQueryEngineFuzzTest.ConcatIsCorrect";

  auto p = std::make_unique<VerifiedPackage>(kTestName);
  FunctionBuilder fb(kTestName, p.get());
  BValue x = fb.Param("x", p->GetBitsType(x_intervals.BitCount()));
  BValue y = fb.Param("y", p->GetBitsType(y_intervals.BitCount()));
  BValue z = fb.Param("z", p->GetBitsType(z_intervals.BitCount()));
  BValue expr = fb.Concat({x, y, z});
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
FUZZ_TEST(RangeQueryEngineFuzzTest, ConcatIsCorrect)
    .WithDomains(NonemptyNormalizedIntervalSet(6),
                 NonemptyNormalizedIntervalSet(5),
                 NonemptyNormalizedIntervalSet(3));

TEST_F(RangeQueryEngineTest, Eq) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue y = fb.Param("y", p->GetBitsType(20));
  BValue expr = fb.Eq(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;

  // When the interval sets are precise and equivalent, we know they are
  // equal.
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

  // When the interval sets are precise and equivalent, we know they are
  // equal.
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

TEST_F(RangeQueryEngineTest, UMulOverflow) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue expr = fb.UMul(x, y, 8);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(0, 8), UBits(4, 8))}));
  engine.SetIntervalSetTree(y.node(),
                            BitsLTT(y.node(), {Interval::Maximal(8)}));

  XLS_ASSERT_OK(engine.Populate(f));

  // Overflow so we have maximal range
  EXPECT_EQ(engine.GetIntervalSetTree(expr.node()),
            BitsLTT(expr.node(), {Interval::Maximal(8)}));
}

TEST_F(RangeQueryEngineTest, UMulNoOverflow) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue expr = fb.UMul(x, y, 8);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(0, 8), UBits(4, 8))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(0, 8), UBits(4, 8))}));

  XLS_ASSERT_OK(engine.Populate(f));

  // None of the results can overflow so we have a real range.
  EXPECT_EQ(engine.GetIntervalSetTree(expr.node()),
            BitsLTT(expr.node(), {Interval(UBits(0, 8), UBits(16, 8))}));
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

  // Interval set covers only numbers with an odd number of 1s, so result
  // is 1.
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

  // Intervals are precise, but don't match parity of 1s, so result is
  // unknown.
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

class IntervalRangeGivens : public RangeDataProvider {
 public:
  explicit IntervalRangeGivens(absl::Span<Node* const> topo_sort)
      : topo_sort_(topo_sort) {}
  std::optional<RangeData> GetKnownIntervals(Node* node) override {
    return std::nullopt;
  }

  absl::Status IterateFunction(DfsVisitor* visitor) override {
    for (Node* n : topo_sort_) {
      XLS_RETURN_IF_ERROR(n->VisitSingleNode(visitor));
    }
    return absl::OkStatus();
  }

 private:
  absl::Span<Node* const> topo_sort_;
};

// Make sure we can bail out of a range analysis if we have enough data.
TEST_F(RangeQueryEngineTest, EarlyBailout) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(10));
  BValue y = fb.Param("y", p->GetBitsType(10));
  BValue z = fb.Param("z", p->GetBitsType(10));
  // pretend we only care about this value for some reason and are ok with not
  // tracking data after it.
  BValue xy = fb.Add(x, y);
  BValue xyz = fb.Add(xy, z);
  // Always true (200 + 20 + 5) < 250
  BValue ltxyz = fb.ULt(xyz, fb.Literal(UBits(250, 10)));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  // Give inputs ranges.
  engine.SetIntervalSetTree(
      x.node(), BitsLTT(x.node(), {Interval(UBits(100, 10), UBits(200, 10))}));
  engine.SetIntervalSetTree(
      y.node(), BitsLTT(y.node(), {Interval(UBits(10, 10), UBits(20, 10))}));
  engine.SetIntervalSetTree(
      z.node(), BitsLTT(z.node(), {Interval(UBits(1, 10), UBits(5, 10))}));

  std::vector<Node*> sort = TopoSort(f);
  // Get the topological sort list up to and including xy
  IntervalRangeGivens test_givens(
      absl::MakeSpan(&*sort.begin(), &*(absl::c_find(sort, xy.node()) + 1)));
  XLS_ASSERT_OK(engine.PopulateWithGivens(test_givens));

  // We should stop after calculating xy so xyz should not have any info
  // beyond type based.
  EXPECT_EQ(engine.GetIntervalSetTree(xyz.node()),
            BitsLTT(xyz.node(), {Interval::Maximal(10)}));
  EXPECT_EQ(engine.GetIntervalSetTree(ltxyz.node()),
            BitsLTT(ltxyz.node(), {Interval::Maximal(1)}));

  // XY should have correct information though.
  EXPECT_EQ(engine.GetIntervalSetTree(xy.node()),
            BitsLTT(xy.node(), {Interval(UBits(110, 10), UBits(220, 10))}));
}

template <typename FKnown>
class LambdaRangeGivens : public RangeDataProvider {
 public:
  LambdaRangeGivens(FunctionBase* func, FKnown known)
      : func_(func), known_func_(known) {}
  std::optional<RangeData> GetKnownIntervals(Node* node) override {
    return known_func_(node);
  }

  absl::Status IterateFunction(DfsVisitor* visitor) override {
    return func_->Accept(visitor);
  }

 private:
  FunctionBase* func_;
  FKnown known_func_;
};

template <typename F>
LambdaRangeGivens(FunctionBase*, F) -> LambdaRangeGivens<F>;

TEST_F(RangeQueryEngineTest, ExactGivenValue) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  // NB z is [0,4]
  BValue z = fb.Param("z", p->GetBitsType(2));
  BValue zext = fb.ZeroExtend(z, 8);
  // We will have a precise known-value of 12 for this.
  BValue xy = fb.Add(x, y);
  BValue xyz = fb.Add(xy, zext);
  // Always true: 12 + [0,3] == [12,15] < 25
  BValue ltxyz = fb.ULt(xyz, fb.Literal(UBits(25, 8)));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  // Inputs have Maximal range.
  // Given xy an a-priori exact value.
  auto intervals = [&](Node* n) -> std::optional<RangeData> {
    if (n == xy.node()) {
      return RangeData{
          .ternary = ternary_ops::FromKnownBits(
              /*known_bits=*/UBits(0xff, 8),
              /*known_bits_values=*/UBits(/* decimal 12 */ 0b00001100, 8)),
          .interval_set = BitsLTT(n, {Interval::Precise(UBits(12, 8))})};
    }
    return std::nullopt;
  };
  LambdaRangeGivens test_givens(f, intervals);
  XLS_ASSERT_OK(engine.PopulateWithGivens(test_givens));
  EXPECT_EQ(engine.GetIntervalSetTree(x.node()),
            BitsLTT(x.node(), {Interval::Maximal(8)}));
  EXPECT_EQ(engine.GetIntervalSetTree(y.node()),
            BitsLTT(y.node(), {Interval::Maximal(8)}));
  EXPECT_EQ(engine.GetIntervalSetTree(z.node()),
            BitsLTT(z.node(), {Interval::Maximal(2)}));
  EXPECT_EQ(engine.GetIntervalSetTree(zext.node()),
            BitsLTT(zext.node(), {Interval::Maximal(2).ZeroExtend(8)}));
  EXPECT_EQ(engine.GetIntervalSetTree(xy.node()),
            BitsLTT(xy.node(), {Interval::Precise(UBits(12, 8))}));
  EXPECT_EQ(engine.GetIntervalSetTree(xyz.node()),
            BitsLTT(xyz.node(), {Interval(UBits(12, 8), UBits(15, 8))}));
  EXPECT_EQ(engine.GetIntervalSetTree(ltxyz.node()),
            BitsLTT(ltxyz.node(), {Interval::Precise(UBits(1, 1))}));
}

TEST_F(RangeQueryEngineTest, RangeGivenValue) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  // NB z is [0,4]
  BValue z = fb.Param("z", p->GetBitsType(2));
  BValue zext = fb.ZeroExtend(z, 8);
  // We will have a known range of [0, 12] for this
  BValue xy = fb.Add(x, y);
  BValue xyz = fb.Add(xy, zext);
  // Always true: [0, 12] + [0,3] == [0,15] < 25
  BValue ltxyz = fb.ULt(xyz, fb.Literal(UBits(25, 8)));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  // Inputs have Maximal range.
  // Given xy an a-priori range.
  auto intervals = [&](Node* n) -> std::optional<RangeData> {
    if (n == xy.node()) {
      return RangeData{
          .ternary = ternary_ops::FromKnownBits(
              /*known_bits=*/UBits(0xf0, 8),
              /*known_bits_values=*/UBits(0b00000000, 8)),
          .interval_set = BitsLTT(n, {Interval(UBits(0, 8), UBits(12, 8))})};
    }
    return std::nullopt;
  };
  LambdaRangeGivens test_givens(f, intervals);
  XLS_ASSERT_OK(engine.PopulateWithGivens(test_givens));
  EXPECT_EQ(engine.GetIntervalSetTree(x.node()),
            BitsLTT(x.node(), {Interval::Maximal(8)}));
  EXPECT_EQ(engine.GetIntervalSetTree(y.node()),
            BitsLTT(y.node(), {Interval::Maximal(8)}));
  EXPECT_EQ(engine.GetIntervalSetTree(z.node()),
            BitsLTT(z.node(), {Interval::Maximal(2)}));
  EXPECT_EQ(engine.GetIntervalSetTree(zext.node()),
            BitsLTT(zext.node(), {Interval::Maximal(2).ZeroExtend(8)}));
  EXPECT_EQ(engine.GetIntervalSetTree(xy.node()),
            BitsLTT(xy.node(), {Interval(UBits(0, 8), UBits(12, 8))}));
  EXPECT_EQ(engine.GetIntervalSetTree(xyz.node()),
            BitsLTT(xyz.node(), {Interval(UBits(0, 8), UBits(15, 8))}));
  EXPECT_EQ(engine.GetIntervalSetTree(ltxyz.node()),
            BitsLTT(ltxyz.node(), {Interval::Precise(UBits(1, 1))}));
}

TEST_F(RangeQueryEngineTest, KnownGateValueZero) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue param = fb.Param("x", p->GetBitsType(1));
  BValue gate = fb.Gate(param, fb.Literal(UBits(0, 8)));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetIntervalSetTree(gate.node()),
            BitsLTT(gate.node(), {Interval::Precise(UBits(0, 8))}));
}

TEST_F(RangeQueryEngineTest, KnownGateValueConditionOne) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue param = fb.Param("x", p->GetBitsType(8));
  BValue gate = fb.Gate(fb.Literal(UBits(1, 1)), param);

  auto intervals = [&](Node* n) -> std::optional<RangeData> {
    if (n == param.node()) {
      return RangeData{
          .ternary = ternary_ops::FromKnownBits(
              /*known_bits=*/UBits(0xff, 8),
              /*known_bits_values=*/UBits(/* decimal 12 */ 0b00001100, 8)),
          .interval_set = BitsLTT(n, {Interval::Precise(UBits(12, 8))})};
    }
    return std::nullopt;
  };
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  LambdaRangeGivens test_givens(f, intervals);
  RangeQueryEngine engine;
  XLS_ASSERT_OK(engine.PopulateWithGivens(test_givens));
  EXPECT_EQ(engine.GetIntervalSetTree(gate.node()),
            BitsLTT(gate.node(), {Interval::Precise(UBits(12, 8))}));
}

TEST_F(RangeQueryEngineTest, KnownGateValueConditionZero) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue param = fb.Param("x", p->GetBitsType(8));
  BValue gate = fb.Gate(fb.Literal(UBits(0, 1)), param);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetIntervalSetTree(gate.node()),
            BitsLTT(gate.node(), {Interval::Precise(UBits(0, 8))}));
}

TEST_F(RangeQueryEngineTest, TupleRangeGivenValue) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  // let x = ...
  // let y = ...
  // let xy = (x, y) -- given ranges [[0,12], [0,3]]
  // let (x2, y2) = xy
  // x2 + y2 -- range [0,15]
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue xy = fb.Tuple({x, y});
  BValue x2 = fb.TupleIndex(xy, 0);
  BValue y2 = fb.TupleIndex(xy, 1);
  BValue ret = fb.Add(x2, y2);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  // Inputs have Maximal range.
  // Given xy an a-priori range.

  IntervalSetTree xy_tree(fb.GetType(xy));
  IntervalSet x_interval(8);
  x_interval.AddInterval(Interval(UBits(0, 8), UBits(12, 8)));
  x_interval.Normalize();
  IntervalSet y_interval(8);
  y_interval.AddInterval(Interval(UBits(0, 8), UBits(3, 8)));
  y_interval.Normalize();
  xy_tree.Set({0}, x_interval);
  xy_tree.Set({1}, y_interval);
  auto intervals = [&](Node* n) -> std::optional<RangeData> {
    if (n == xy.node()) {
      return RangeData{
          .ternary = std::nullopt,
          .interval_set = xy_tree,
      };
    }
    return std::nullopt;
  };
  LambdaRangeGivens test_givens(f, intervals);
  XLS_ASSERT_OK(engine.PopulateWithGivens(test_givens));

  EXPECT_EQ(engine.GetIntervalSetTree(x.node()),
            BitsLTT(x.node(), {Interval::Maximal(8)}));
  EXPECT_EQ(engine.GetIntervalSetTree(y.node()),
            BitsLTT(y.node(), {Interval::Maximal(8)}));
  EXPECT_EQ(engine.GetIntervalSetTree(xy.node()), xy_tree);
  EXPECT_EQ(engine.GetIntervalSetTree(x2.node()),
            BitsLTT(x2.node(), {Interval(UBits(0, 8), UBits(12, 8))}));
  EXPECT_EQ(engine.GetIntervalSetTree(y2.node()),
            BitsLTT(y2.node(), {Interval(UBits(0, 8), UBits(3, 8))}));
  EXPECT_EQ(engine.GetIntervalSetTree(ret.node()),
            BitsLTT(ret.node(), {Interval(UBits(0, 8), UBits(15, 8))}));
}

TEST_F(RangeQueryEngineTest, MultipleRangeGivenValue) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  // We will have a known range of [0,3] for this.
  BValue z = fb.Param("z", p->GetBitsType(8));
  // We will have a known range of [0, 12] for this
  BValue xy = fb.Add(x, y);
  BValue xyz = fb.Add(xy, z);
  // Always true: [0, 12] + [0,3] == [0,15] < 25
  BValue ltxyz = fb.ULt(xyz, fb.Literal(UBits(25, 8)));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  RangeQueryEngine engine;
  // Inputs have Maximal range.
  // Give xy and z an a-priori range
  auto intervals = [&](Node* n) -> std::optional<RangeData> {
    if (n == xy.node()) {
      return RangeData{
          .ternary = ternary_ops::FromKnownBits(
              /*known_bits=*/UBits(0xf0, 8),
              /*known_bits_values=*/UBits(0b00000000, 8)),
          .interval_set = BitsLTT(n, {Interval(UBits(0, 8), UBits(12, 8))})};
    }
    if (n == z.node()) {
      return RangeData{
          .ternary = ternary_ops::FromKnownBits(
              /*known_bits=*/UBits(0xfb, 8),
              /*known_bits_values=*/UBits(0b00000000, 8)),
          .interval_set =
              BitsLTT(z.node(), {Interval::Maximal(2).ZeroExtend(8)}),
      };
    }
    return std::nullopt;
  };
  LambdaRangeGivens test_givens(f, intervals);
  XLS_ASSERT_OK(engine.PopulateWithGivens(test_givens));
  EXPECT_EQ(engine.GetIntervalSetTree(x.node()),
            BitsLTT(x.node(), {Interval::Maximal(8)}));
  EXPECT_EQ(engine.GetIntervalSetTree(y.node()),
            BitsLTT(y.node(), {Interval::Maximal(8)}));
  EXPECT_EQ(engine.GetIntervalSetTree(z.node()),
            BitsLTT(z.node(), {Interval::Maximal(2).ZeroExtend(8)}));
  EXPECT_EQ(engine.GetIntervalSetTree(xy.node()),
            BitsLTT(xy.node(), {Interval(UBits(0, 8), UBits(12, 8))}));
  EXPECT_EQ(engine.GetIntervalSetTree(xyz.node()),
            BitsLTT(xyz.node(), {Interval(UBits(0, 8), UBits(15, 8))}));
  EXPECT_EQ(engine.GetIntervalSetTree(ltxyz.node()),
            BitsLTT(ltxyz.node(), {Interval::Precise(UBits(1, 1))}));
}

}  // namespace
}  // namespace xls
