// Copyright 2025 The XLS Authors
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

#include "xls/passes/partial_info_query_engine.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/interval_set_test_utils.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/partial_information.h"
#include "xls/ir/partial_ops.h"
#include "xls/ir/ternary.h"
#include "xls/ir/value.h"
#include "xls/ir/value_builder.h"
#include "xls/passes/range_query_engine.h"

namespace xls {
namespace {

using PartialInformationTree = LeafTypeTree<PartialInformation>;

class PartialInfoQueryEngineTest : public IrTestBase {};

IntervalSet CreateIntervalSet(
    int64_t bit_count, absl::Span<const std::pair<int64_t, int64_t>> bounds) {
  CHECK(!bounds.empty());
  IntervalSet interval_set(bit_count);
  for (const auto& [lb, ub] : bounds) {
    Bits lb_bits = lb < 0 ? SBits(lb, bit_count) : UBits(lb, bit_count);
    Bits ub_bits = ub < 0 ? SBits(ub, bit_count) : UBits(ub, bit_count);
    interval_set.AddInterval(Interval(lb_bits, ub_bits));
  }
  interval_set.Normalize();
  return interval_set;
}

PartialInformation CreateRangePartialInformation(
    int64_t bit_count, absl::Span<const std::pair<int64_t, int64_t>> bounds) {
  return PartialInformation(CreateIntervalSet(bit_count, bounds));
}

PartialInformationTree BitsLTT(
    Node* node, absl::Span<const std::pair<int64_t, int64_t>> bounds) {
  CHECK(node->GetType()->IsBits());
  return PartialInformationTree::CreateSingleElementTree(
      node->GetType(),
      CreateRangePartialInformation(node->BitCountOrDie(), bounds));
}

PartialInformationTree BitsLTT(Node* node,
                               absl::Span<const Interval> intervals) {
  CHECK(!intervals.empty());
  IntervalSet interval_set = IntervalSet::Of(intervals);
  CHECK(node->GetType()->IsBits());
  return PartialInformationTree::CreateSingleElementTree(
      node->GetType(), PartialInformation(std::move(interval_set)));
}

MATCHER_P(IntervalsMatch, expected,
          absl::StrFormat("%smatch interval set of %s",
                          negation ? "doesn't " : "does ",
                          expected.ToString())) {
  std::vector<IntervalSet> expected_elements;
  expected_elements.reserve(expected.elements().size());
  for (const PartialInformation& p : expected.elements()) {
    expected_elements.push_back(p.RangeOrMaximal());
  }
  return testing::ExplainMatchResult(testing::Eq(expected.type()), arg.type(),
                                     result_listener) &&
         testing::ExplainMatchResult(
             testing::ElementsAreArray(expected_elements), expected_elements,
             result_listener);
}

TEST_F(PartialInfoQueryEngineTest, Add) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue y = fb.Param("y", p->GetBitsType(20));
  BValue expr = fb.Add(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(UBits(0, 20), UBits(48, 20))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(0, 20), UBits(15, 20))})));

  // We know that at most 6 bits are used, since 48 + 15 = 63 <= 2^6 - 1
  EXPECT_EQ("0b0000_0000_0000_00XX_XXXX", engine.ToString(expr.node()));

  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(),
      BitsLTT(x.node(), {Interval(UBits(1048566, 20), Bits::AllOnes(20))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(),
      BitsLTT(y.node(), {Interval(UBits(1048566, 20), UBits(1048570, 20))})));

  EXPECT_EQ("0b1111_1111_1111_111X_XXXX", engine.ToString(expr.node()));
  EXPECT_EQ(IntervalSetTreeToString(engine.GetIntervals(expr.node())),
            "[[1048544, 1048575]]");
}

TEST_F(PartialInfoQueryEngineTest, Array) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(10));
  BValue y = fb.Param("y", p->GetBitsType(10));
  BValue expr = fb.Array({x, y}, p->GetBitsType(10));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  PartialInformationTree x_given =
      BitsLTT(x.node(), {Interval(UBits(100, 10), UBits(150, 10))});
  PartialInformationTree y_given =
      BitsLTT(y.node(), {Interval(UBits(200, 10), UBits(250, 10))});

  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), x_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(y.node(), y_given));

  EXPECT_EQ(x_given.Get({}).Range(), engine.GetIntervals(expr.node()).Get({0}));
  EXPECT_EQ(y_given.Get({}).Range(), engine.GetIntervals(expr.node()).Get({1}));

  EXPECT_EQ(IntervalSetTreeToString(engine.GetIntervals(expr.node())),
            R"([
  [[100, 150]]
  [[200, 250]]
]
)");
}

TEST_F(PartialInfoQueryEngineTest, ArrayConcat) {
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
  PartialInfoQueryEngine engine;

  PartialInformationTree x_given =
      BitsLTT(x.node(), {Interval(UBits(100, 10), UBits(150, 10))});
  PartialInformationTree y_given =
      BitsLTT(y.node(), {Interval(UBits(200, 10), UBits(250, 10))});
  PartialInformationTree z_given =
      BitsLTT(z.node(), {Interval(UBits(300, 10), UBits(350, 10))});
  PartialInformationTree w_given =
      BitsLTT(w.node(), {Interval(UBits(400, 10), UBits(450, 10))});

  XLS_ASSERT_OK(engine.Populate(f));
  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), x_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(y.node(), y_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(z.node(), z_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(w.node(), w_given));

  EXPECT_EQ(x_given.Get({}).Range(), engine.GetIntervals(expr.node()).Get({0}));
  EXPECT_EQ(y_given.Get({}).Range(), engine.GetIntervals(expr.node()).Get({1}));
  EXPECT_EQ(z_given.Get({}).Range(), engine.GetIntervals(expr.node()).Get({2}));
  EXPECT_EQ(w_given.Get({}).Range(), engine.GetIntervals(expr.node()).Get({3}));
  EXPECT_EQ(x_given.Get({}).Range(), engine.GetIntervals(expr.node()).Get({4}));
  EXPECT_EQ(y_given.Get({}).Range(), engine.GetIntervals(expr.node()).Get({5}));
}

TEST_F(PartialInfoQueryEngineTest, ArrayIndex1D) {
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
  PartialInfoQueryEngine engine;

  PartialInformationTree x_given =
      BitsLTT(x.node(), {Interval(UBits(100, 10), UBits(150, 10))});
  PartialInformationTree y_given =
      BitsLTT(y.node(), {Interval(UBits(200, 10), UBits(250, 10))});
  PartialInformationTree z_given =
      BitsLTT(z.node(), {Interval(UBits(300, 10), UBits(350, 10))});
  PartialInformationTree i_given =
      BitsLTT(i.node(), {Interval(UBits(1, 2), UBits(2, 2))});

  XLS_ASSERT_OK(engine.Populate(f));
  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), x_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(y.node(), y_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(z.node(), z_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(i.node(), i_given));

  EXPECT_EQ(x_given.Get({}).Range(),
            engine.GetIntervals(index0.node()).Get({}));
  EXPECT_EQ(y_given.Get({}).Range(),
            engine.GetIntervals(index1.node()).Get({}));
  EXPECT_EQ(z_given.Get({}).Range(),
            engine.GetIntervals(index2.node()).Get({}));
  // An out-of-bounds index returns the highest-index element.
  EXPECT_EQ(z_given.Get({}).Range(),
            engine.GetIntervals(oob_index.node()).Get({}));
  EXPECT_EQ(
      IntervalSet::Combine(*y_given.Get({}).Range(), *z_given.Get({}).Range()),
      engine.GetIntervals(index12.node()).Get({}));
}

TEST_F(PartialInfoQueryEngineTest, ArrayIndex3D) {
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
  PartialInfoQueryEngine engine;

  PartialInformationTree x_given =
      BitsLTT(x.node(), {Interval(UBits(2, 10), UBits(3, 10))});
  PartialInformationTree y_given =
      BitsLTT(y.node(), {Interval(UBits(2, 10), UBits(4, 10))});
  PartialInformationTree z_given =
      BitsLTT(z.node(), {Interval(UBits(1, 10), UBits(1, 10))});

  XLS_ASSERT_OK(engine.Populate(f));
  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), x_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(y.node(), y_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(z.node(), z_given));

  // array[2, 2, 1] = 44
  // array[2, 3, 1] = 47
  // array[2, 4, 1] = 50
  // array[3, 2, 1] = 62
  // array[3, 3, 1] = 65
  // array[3, 4, 1] = 68

  IntervalSet expected = IntervalSet::Of({
      Interval::Precise(UBits(44, 32)),
      Interval::Precise(UBits(47, 32)),
      Interval::Precise(UBits(50, 32)),
      Interval::Precise(UBits(62, 32)),
      Interval::Precise(UBits(65, 32)),
      Interval::Precise(UBits(68, 32)),
  });

  EXPECT_EQ(expected, engine.GetIntervals(index.node()).Get({}));
}

TEST_F(PartialInfoQueryEngineTest, ArraySlice) {
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
  PartialInfoQueryEngine engine;

  PartialInformationTree a_given =
      BitsLTT(a.node(), {Interval(UBits(400, 10), UBits(450, 10))});
  PartialInformationTree b_given =
      BitsLTT(b.node(), {Interval(UBits(500, 10), UBits(550, 10))});
  PartialInformationTree c_given =
      BitsLTT(c.node(), {Interval(UBits(600, 10), UBits(650, 10))});
  PartialInformationTree w_given =
      BitsLTT(w.node(), {Interval(UBits(0, 10), UBits(50, 10))});
  PartialInformationTree x_given =
      BitsLTT(x.node(), {Interval(UBits(100, 10), UBits(150, 10))});
  PartialInformationTree y_given =
      BitsLTT(y.node(), {Interval(UBits(200, 10), UBits(250, 10))});
  PartialInformationTree z_given =
      BitsLTT(z.node(), {Interval(UBits(300, 10), UBits(350, 10))});
  PartialInformationTree i_given =
      BitsLTT(i.node(), {Interval(UBits(1, 2), UBits(2, 2))});
  PartialInformationTree big_i_given =
      BitsLTT(big_i.node(), {Interval(UBits(20, 20), UBits(30, 20))});

  PartialInformationTree abc_given = PartialInformationTree::CreateFromVector(
      p->GetArrayType(3, p->GetBitsType(10)),
      {a_given.Get({}), b_given.Get({}), c_given.Get({})});

  PartialInformationTree wxyz_given = PartialInformationTree::CreateFromVector(
      p->GetArrayType(4, p->GetBitsType(10)),
      {w_given.Get({}), x_given.Get({}), y_given.Get({}), z_given.Get({})});

  PartialInformationTree xyzzzzzz_given =
      PartialInformationTree::CreateFromVector(
          p->GetArrayType(8, p->GetBitsType(10)),
          {x_given.Get({}), y_given.Get({}), z_given.Get({}), z_given.Get({}),
           z_given.Get({}), z_given.Get({}), z_given.Get({}), z_given.Get({})});

  PartialInformationTree bc_or_cw_given =
      PartialInformationTree::CreateFromVector(
          p->GetArrayType(2, p->GetBitsType(10)),
          {partial_ops::Meet(b_given.Get({}), c_given.Get({})),
           partial_ops::Meet(c_given.Get({}), w_given.Get({}))});

  PartialInformationTree zz_given = PartialInformationTree::CreateFromVector(
      p->GetArrayType(2, p->GetBitsType(10)),
      {z_given.Get({}), z_given.Get({})});

  PartialInformationTree unbound_given =
      PartialInformationTree::CreateFromVector(
          p->GetArrayType(1, p->GetBitsType(10)),
          {CreateRangePartialInformation(10, {{0, 50},
                                              {100, 150},
                                              {200, 250},
                                              {300, 350},
                                              {400, 450},
                                              {500, 550},
                                              {600, 650}})});

  XLS_ASSERT_OK(engine.Populate(f));
  XLS_ASSERT_OK(engine.ReplaceGiven(a.node(), a_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(b.node(), b_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(c.node(), c_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(w.node(), w_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), x_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(y.node(), y_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(z.node(), z_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(i.node(), i_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(big_i.node(), big_i_given));

  EXPECT_THAT(engine.GetIntervals(slice0_3.node()), IntervalsMatch(abc_given));
  EXPECT_THAT(engine.GetIntervals(slice3_4.node()), IntervalsMatch(wxyz_given));
  EXPECT_THAT(engine.GetIntervals(slice4_8.node()),
              IntervalsMatch(xyzzzzzz_given));
  EXPECT_THAT(engine.GetIntervals(slice12_2.node()),
              IntervalsMatch(bc_or_cw_given));
  EXPECT_THAT(engine.GetIntervals(slice_unbound.node()),
              IntervalsMatch(unbound_given));
  EXPECT_THAT(engine.GetIntervals(slice_oob.node()), IntervalsMatch(zz_given));
}

TEST_F(PartialInfoQueryEngineTest, ArrayUpdate1D) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(10));
  BValue y = fb.Param("y", p->GetBitsType(10));
  BValue z = fb.Param("z", p->GetBitsType(10));
  BValue value = fb.Param("value", p->GetBitsType(10));
  BValue array = fb.Array({x, y, z}, p->GetBitsType(10));
  BValue update0 = fb.ArrayUpdate(array, value, {fb.Literal(UBits(0, 2))});
  BValue update1 = fb.ArrayUpdate(array, value, {fb.Literal(UBits(1, 2))});
  BValue update2 = fb.ArrayUpdate(array, value, {fb.Literal(UBits(2, 2))});
  BValue oob_update = fb.ArrayUpdate(array, value, {fb.Literal(UBits(3, 2))});
  BValue i = fb.Param("i", p->GetBitsType(2));
  BValue update12 = fb.ArrayUpdate(array, value, {i});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;

  PartialInformationTree x_given =
      BitsLTT(x.node(), {Interval(UBits(100, 10), UBits(150, 10))});
  PartialInformationTree y_given =
      BitsLTT(y.node(), {Interval(UBits(200, 10), UBits(250, 10))});
  PartialInformationTree z_given =
      BitsLTT(z.node(), {Interval(UBits(300, 10), UBits(350, 10))});
  PartialInformationTree value_given =
      BitsLTT(value.node(), {Interval(UBits(400, 10), UBits(450, 10))});
  PartialInformationTree i_given =
      BitsLTT(i.node(), {Interval(UBits(1, 2), UBits(2, 2))});

  XLS_ASSERT_OK(engine.Populate(f));
  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), x_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(y.node(), y_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(z.node(), z_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(value.node(), value_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(i.node(), i_given));

  auto array_expected = [array_type = array.GetType()](
                            const PartialInformation& x,
                            const PartialInformation& y,
                            const PartialInformation& z) {
    return IntervalSetTree::CreateFromVector(
        array_type,
        {x.RangeOrMaximal(), y.RangeOrMaximal(), z.RangeOrMaximal()});
  };

  EXPECT_EQ(
      array_expected(value_given.Get({}), y_given.Get({}), z_given.Get({})),
      engine.GetIntervals(update0.node()));
  EXPECT_EQ(
      array_expected(x_given.Get({}), value_given.Get({}), z_given.Get({})),
      engine.GetIntervals(update1.node()));
  EXPECT_EQ(
      array_expected(x_given.Get({}), y_given.Get({}), value_given.Get({})),
      engine.GetIntervals(update2.node()));
  // An out-of-bounds update leaves the array unchanged.
  EXPECT_EQ(array_expected(x_given.Get({}), y_given.Get({}), z_given.Get({})),
            engine.GetIntervals(oob_update.node()));
  // If multiple elements could be updated, we get the union of each element's
  // initial value & the updated value.
  EXPECT_EQ(
      array_expected(x_given.Get({}),
                     partial_ops::Meet(y_given.Get({}), value_given.Get({})),
                     partial_ops::Meet(z_given.Get({}), value_given.Get({}))),
      engine.GetIntervals(update12.node()));
}

TEST_F(PartialInfoQueryEngineTest, ArrayUpdateBigArray) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  constexpr int64_t kMaxX = 3;
  constexpr int64_t kSetValue = 100;
  BValue x = fb.Param("x", p->GetBitsType(2));
  BValue idx = fb.ZeroExtend(x, 3);
  BValue y = fb.Literal(UBits(kSetValue, 8));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value arr, ValueBuilder::UBitsArray(
                     {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 8)
                     .Build());
  BValue target = fb.Literal(arr);
  BValue update = fb.ArrayUpdate(target, y, {idx});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));
  XLS_ASSERT_OK_AND_ASSIGN(
      PartialInformationTree given,
      PartialInformationTree::CreateFromFunction(
          update.GetType(),
          [&](Type* t, absl::Span<const int64_t> idx) -> PartialInformation {
            Interval existing = Interval::Precise(arr.element(idx[0]).bits());
            if (idx[0] <= kMaxX) {
              return PartialInformation(IntervalSet::Of(
                  {Interval::Precise(UBits(kSetValue, 8)), existing}));
            }
            return PartialInformation(IntervalSet::Of({existing}));
          }));

  EXPECT_THAT(engine.GetIntervals(update.node()), IntervalsMatch(given));
}

TEST_F(PartialInfoQueryEngineTest, ArrayUpdateBigArrayIsAlwaysConstrained) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  constexpr int64_t kMaxX = 3;
  constexpr int64_t kSetValue = 100;
  BValue x = fb.Param("x", p->GetBitsType(2));
  BValue y = fb.Literal(UBits(kSetValue, 8));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value arr, ValueBuilder::UBitsArray(
                     {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 8)
                     .Build());
  BValue target = fb.Literal(arr);
  BValue update = fb.ArrayUpdate(target, y, {x});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));
  XLS_ASSERT_OK_AND_ASSIGN(
      PartialInformationTree given,
      PartialInformationTree::CreateFromFunction(
          update.GetType(),
          [&](Type* t, absl::Span<const int64_t> idx) -> PartialInformation {
            Interval existing = Interval::Precise(arr.element(idx[0]).bits());
            if (idx[0] <= kMaxX) {
              return PartialInformation(IntervalSet::Of(
                  {Interval::Precise(UBits(kSetValue, 8)), existing}));
            }
            return PartialInformation(IntervalSet::Of({existing}));
          }));

  EXPECT_THAT(engine.GetIntervals(update.node()), IntervalsMatch(given));
}

TEST_F(PartialInfoQueryEngineTest, ArrayUpdate3D) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // Dimension = [5, 6, 3]
  XLS_ASSERT_OK_AND_ASSIGN(Value array_value,
                           ValueBuilder::Array({ValueBuilder::UBits2DArray(
                                                    {
                                                        {1, 2, 3},
                                                        {4, 5, 6},
                                                        {7, 8, 9},
                                                        {10, 11, 12},
                                                        {13, 14, 15},
                                                        {16, 17, 18},
                                                    },
                                                    32),
                                                ValueBuilder::UBits2DArray(
                                                    {
                                                        {19, 20, 21},
                                                        {22, 23, 24},
                                                        {25, 26, 27},
                                                        {28, 29, 30},
                                                        {31, 32, 33},
                                                        {34, 35, 36},
                                                    },
                                                    32),
                                                ValueBuilder::UBits2DArray(
                                                    {
                                                        {37, 38, 39},
                                                        {40, 41, 42},
                                                        {43, 44, 45},
                                                        {46, 47, 48},
                                                        {49, 50, 51},
                                                        {52, 53, 54},
                                                    },
                                                    32),
                                                ValueBuilder::UBits2DArray(
                                                    {
                                                        {55, 56, 57},
                                                        {58, 59, 60},
                                                        {61, 62, 63},
                                                        {64, 65, 66},
                                                        {67, 68, 69},
                                                        {70, 71, 72},
                                                    },
                                                    32),
                                                ValueBuilder::UBits2DArray(
                                                    {
                                                        {73, 74, 75},
                                                        {76, 77, 78},
                                                        {79, 80, 81},
                                                        {82, 83, 84},
                                                        {85, 86, 87},
                                                        {88, 89, 90},
                                                    },
                                                    32)})
                               .Build());

  BValue x = fb.Param("x", p->GetBitsType(10));
  BValue y = fb.Param("y", p->GetBitsType(10));
  BValue z = fb.Param("z", p->GetBitsType(10));
  BValue array = fb.Literal(array_value);
  BValue value = fb.Param("value", p->GetBitsType(32));
  BValue update = fb.ArrayUpdate(array, value, {x, y, z});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;

  PartialInformationTree x_given =
      BitsLTT(x.node(), {Interval(UBits(2, 10), UBits(3, 10))});
  PartialInformationTree y_given =
      BitsLTT(y.node(), {Interval(UBits(2, 10), UBits(4, 10))});
  PartialInformationTree z_given =
      BitsLTT(z.node(), {Interval(UBits(1, 10), UBits(1, 10))});
  PartialInformationTree value_given =
      BitsLTT(value.node(), {Interval(UBits(500, 32), UBits(500, 32))});

  XLS_ASSERT_OK(engine.Populate(f));
  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), x_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(y.node(), y_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(z.node(), z_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(value.node(), value_given));

  // Could update any of:
  //   array[2, 2, 1],
  //   array[2, 3, 1],
  //   array[2, 4, 1],
  //   array[3, 2, 1],
  //   array[3, 3, 1], or
  //   array[3, 4, 1].
  XLS_ASSERT_OK_AND_ASSIGN(
      IntervalSetTree expected,
      IntervalSetTree::CreateFromFunction(
          array.GetType(),
          [&](Type*,
              absl::Span<const int64_t> index) -> absl::StatusOr<IntervalSet> {
            return IntervalSet::Precise(array_value.element(index[0])
                                            .element(index[1])
                                            .element(index[2])
                                            .bits());
          }));
  expected.Set({2, 2, 1},
               IntervalSet::Combine(expected.Get({2, 2, 1}),
                                    value_given.Get({}).RangeOrMaximal()));
  expected.Set({2, 3, 1},
               IntervalSet::Combine(expected.Get({2, 3, 1}),
                                    value_given.Get({}).RangeOrMaximal()));
  expected.Set({2, 4, 1},
               IntervalSet::Combine(expected.Get({2, 4, 1}),
                                    value_given.Get({}).RangeOrMaximal()));
  expected.Set({3, 2, 1},
               IntervalSet::Combine(expected.Get({3, 2, 1}),
                                    value_given.Get({}).RangeOrMaximal()));
  expected.Set({3, 3, 1},
               IntervalSet::Combine(expected.Get({3, 3, 1}),
                                    value_given.Get({}).RangeOrMaximal()));
  expected.Set({3, 4, 1},
               IntervalSet::Combine(expected.Get({3, 4, 1}),
                                    value_given.Get({}).RangeOrMaximal()));

  EXPECT_EQ(expected, engine.GetIntervals(update.node()));
}

TEST_F(PartialInfoQueryEngineTest, AndReduce) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue expr = fb.AndReduce(x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;

  XLS_ASSERT_OK(engine.Populate(f));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(UBits(0, 20), UBits(48, 20))})));

  // Interval does not cover 2^20 - 1, so the result is known to be 0.
  EXPECT_EQ("0b0", engine.ToString(expr.node()));

  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(),
      BitsLTT(x.node(), {Interval(UBits(500, 20), Bits::AllOnes(20))})));

  // Interval does cover 2^20 - 1, so we don't know the value.
  EXPECT_EQ("0bX", engine.ToString(expr.node()));

  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(),
      BitsLTT(x.node(), {Interval(Bits::AllOnes(20), Bits::AllOnes(20))})));

  // Interval only covers 2^20 - 1, so the result is known to be 1.
  EXPECT_EQ("0b1", engine.ToString(expr.node()));
}

void ConcatIsCorrect(const IntervalSet& x_intervals,
                     const IntervalSet& y_intervals,
                     const IntervalSet& z_intervals) {
  constexpr std::string_view kTestName =
      "PartialInfoQueryEngineFuzzTest.ConcatIsCorrect";

  auto p = std::make_unique<VerifiedPackage>(kTestName);
  FunctionBuilder fb(kTestName, p.get());
  BValue x = fb.Param("x", p->GetBitsType(x_intervals.BitCount()));
  BValue y = fb.Param("y", p->GetBitsType(y_intervals.BitCount()));
  BValue z = fb.Param("z", p->GetBitsType(z_intervals.BitCount()));
  BValue expr = fb.Concat({x, y, z});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), x_intervals.Intervals())));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), y_intervals.Intervals())));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      z.node(), BitsLTT(z.node(), z_intervals.Intervals())));

  x_intervals.ForEachElement([&](const Bits& bits_x) -> bool {
    y_intervals.ForEachElement([&](const Bits& bits_y) -> bool {
      z_intervals.ForEachElement([&](const Bits& bits_z) -> bool {
        Bits concatenated = bits_ops::Concat({bits_x, bits_y, bits_z});
        EXPECT_TRUE(
            engine.GetIntervals(expr.node()).Get({}).Covers(concatenated));
        return false;
      });
      return false;
    });
    return false;
  });
}
FUZZ_TEST(PartialInfoQueryEngineFuzzTest, ConcatIsCorrect)
    .WithDomains(NonemptyNormalizedIntervalSet(6),
                 NonemptyNormalizedIntervalSet(5),
                 NonemptyNormalizedIntervalSet(3));

TEST_F(PartialInfoQueryEngineTest, Decode) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue expr = fb.Decode(x);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(UBits(2, 4), UBits(4, 4)),
                                   Interval::Precise(UBits(9, 4))})));
  EXPECT_EQ(engine.GetIntervals(expr.node()).Get({}),
            IntervalSet::Of({Interval::Precise(Bits::PowerOfTwo(2, 16)),
                             Interval::Precise(Bits::PowerOfTwo(3, 16)),
                             Interval::Precise(Bits::PowerOfTwo(4, 16)),
                             Interval::Precise(Bits::PowerOfTwo(9, 16))}));
}

TEST_F(PartialInfoQueryEngineTest, DecodePrecise) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue expr = fb.Decode(x);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval::Precise(UBits(9, 4))})));
  EXPECT_EQ("0b0000_0010_0000_0000", engine.ToString(expr.node()));
}

TEST_F(PartialInfoQueryEngineTest, DecodePreciseOverflow) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue expr = fb.Decode(x, /*width=*/10);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval::Precise(UBits(10, 4))})));
  EXPECT_EQ("0b00_0000_0000", engine.ToString(expr.node()));

  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval::Precise(UBits(13, 4))})));
  EXPECT_EQ("0b00_0000_0000", engine.ToString(expr.node()));
}

TEST_F(PartialInfoQueryEngineTest, DecodeUnconstrained) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue expr = fb.Decode(x);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  // With an unconstrained input (and only 16 possible outputs), we get a full
  // interval set.
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetIntervals(expr.node()).Get({}),
            IntervalSet::Of({
                Interval(Bits::PowerOfTwo(0, 16), Bits::PowerOfTwo(1, 16)),
                Interval::Precise(Bits::PowerOfTwo(2, 16)),
                Interval::Precise(Bits::PowerOfTwo(3, 16)),
                Interval::Precise(Bits::PowerOfTwo(4, 16)),
                Interval::Precise(Bits::PowerOfTwo(5, 16)),
                Interval::Precise(Bits::PowerOfTwo(6, 16)),
                Interval::Precise(Bits::PowerOfTwo(7, 16)),
                Interval::Precise(Bits::PowerOfTwo(8, 16)),
                Interval::Precise(Bits::PowerOfTwo(9, 16)),
                Interval::Precise(Bits::PowerOfTwo(10, 16)),
                Interval::Precise(Bits::PowerOfTwo(11, 16)),
                Interval::Precise(Bits::PowerOfTwo(12, 16)),
                Interval::Precise(Bits::PowerOfTwo(13, 16)),
                Interval::Precise(Bits::PowerOfTwo(14, 16)),
                Interval::Precise(Bits::PowerOfTwo(15, 16)),
            }));
}

TEST_F(PartialInfoQueryEngineTest, DecodeUnconstrainedOverflow) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue expr = fb.Decode(x, /*width=*/10);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  // With an unconstrained input (and only 10 possible outputs), we get a full
  // interval set - including 0, since overflow is possible.
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetIntervals(expr.node()).Get({}),
            IntervalSet::Of({
                Interval(UBits(0, 10), Bits::PowerOfTwo(1, 10)),
                Interval::Precise(Bits::PowerOfTwo(2, 10)),
                Interval::Precise(Bits::PowerOfTwo(3, 10)),
                Interval::Precise(Bits::PowerOfTwo(4, 10)),
                Interval::Precise(Bits::PowerOfTwo(5, 10)),
                Interval::Precise(Bits::PowerOfTwo(6, 10)),
                Interval::Precise(Bits::PowerOfTwo(7, 10)),
                Interval::Precise(Bits::PowerOfTwo(8, 10)),
                Interval::Precise(Bits::PowerOfTwo(9, 10)),
            }));
}

TEST_F(PartialInfoQueryEngineTest, DecodeUnconstrainedWide) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(5));
  BValue expr = fb.Decode(x);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  // With an unconstrained input (and 32 possible outputs), we get a minimized
  // interval set, collapsing the first several intervals.
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetIntervals(expr.node()).Get({}),
            IntervalSet::Of({
                Interval(Bits::PowerOfTwo(0, 32), Bits::PowerOfTwo(16, 32)),
                Interval::Precise(Bits::PowerOfTwo(17, 32)),
                Interval::Precise(Bits::PowerOfTwo(18, 32)),
                Interval::Precise(Bits::PowerOfTwo(19, 32)),
                Interval::Precise(Bits::PowerOfTwo(20, 32)),
                Interval::Precise(Bits::PowerOfTwo(21, 32)),
                Interval::Precise(Bits::PowerOfTwo(22, 32)),
                Interval::Precise(Bits::PowerOfTwo(23, 32)),
                Interval::Precise(Bits::PowerOfTwo(24, 32)),
                Interval::Precise(Bits::PowerOfTwo(25, 32)),
                Interval::Precise(Bits::PowerOfTwo(26, 32)),
                Interval::Precise(Bits::PowerOfTwo(27, 32)),
                Interval::Precise(Bits::PowerOfTwo(28, 32)),
                Interval::Precise(Bits::PowerOfTwo(29, 32)),
                Interval::Precise(Bits::PowerOfTwo(30, 32)),
                Interval::Precise(Bits::PowerOfTwo(31, 32)),
            }));
}

TEST_F(PartialInfoQueryEngineTest, Eq) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue y = fb.Param("y", p->GetBitsType(20));
  BValue expr = fb.Eq(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  // When the interval sets are precise and equivalent, we know they are
  // equal.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(UBits(560, 20), UBits(560, 20))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(560, 20), UBits(560, 20)),
                                   Interval(UBits(560, 20), UBits(560, 20))})));
  EXPECT_EQ("0b1", engine.ToString(expr.node()));

  // When the interval sets overlap, we don't know anything about them.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(UBits(100, 20), UBits(200, 20))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(150, 20), UBits(300, 20)),
                                   Interval(UBits(400, 20), UBits(500, 20))})));
  EXPECT_EQ("0bX", engine.ToString(expr.node()));

  // When the interval sets are disjoint, we know they are not equal.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {
                                      Interval(UBits(100, 20), UBits(200, 20)),
                                      Interval(UBits(550, 20), UBits(600, 20)),
                                      Interval(UBits(850, 20), UBits(900, 20)),
                                  })));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(400, 20), UBits(500, 20)),
                                   Interval(UBits(700, 20), UBits(800, 20))})));
  EXPECT_EQ("0b0", engine.ToString(expr.node()));
}

TEST_F(PartialInfoQueryEngineTest, Gate) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue cond = fb.Param("cond", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue expr = fb.Gate(cond, x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(UBits(600, 20), UBits(700, 20))})));

  // Gate can either produce zero or the data operand value.
  EXPECT_EQ(engine.GetIntervals(expr.node()).Get({}),
            CreateIntervalSet(20, {{0, 0}, {600, 700}}));
}

TEST_F(PartialInfoQueryEngineTest, GateWithConditionTrue) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue cond = fb.Literal(Value(UBits(1, 1)));
  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue expr = fb.Gate(cond, x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));
  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), BitsLTT(x.node(), {{600, 700}})));

  // With the gated condition true the gate operation produces the original
  // value.
  EXPECT_EQ(engine.GetIntervals(expr.node()).Get({}),
            CreateIntervalSet(20, {{600, 700}}));
}

TEST_F(PartialInfoQueryEngineTest, GateWithConditionFalse) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue cond = fb.Literal(Value(UBits(0, 1)));
  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue expr = fb.Gate(cond, x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));
  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), BitsLTT(x.node(), {{600, 700}})));

  // With the gated condition false the gate results is identically zero.
  EXPECT_EQ(engine.GetIntervals(expr.node()).Get({}),
            IntervalSet::Precise(UBits(0, 20)));
}

TEST_F(PartialInfoQueryEngineTest, GateWithCompoundType) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue cond = fb.Param("cond", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(10));
  BValue y = fb.Param("y", p->GetBitsType(20));
  BValue tuple = fb.Tuple({x, y, y});
  BValue gate = fb.Gate(cond, tuple);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), BitsLTT(x.node(), {{100, 150}})));
  XLS_ASSERT_OK(
      engine.ReplaceGiven(y.node(), BitsLTT(y.node(), {{20000, 25000}})));

  EXPECT_EQ(engine.GetIntervals(gate.node()).Get({0}),
            CreateIntervalSet(10, {{0, 0}, {100, 150}}));
  EXPECT_EQ(engine.GetIntervals(gate.node()).Get({1}),
            CreateIntervalSet(20, {{0, 0}, {20000, 25000}}));
  EXPECT_EQ(engine.GetIntervals(gate.node()).Get({2}),
            CreateIntervalSet(20, {{0, 0}, {20000, 25000}}));
}

TEST_F(PartialInfoQueryEngineTest, Identity) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue expr = fb.Identity(x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  PartialInformationTree given =
      BitsLTT(x.node(), {Interval(UBits(600, 20), UBits(700, 20))});
  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), given));

  // Identity function should leave intervals unaffected.
  EXPECT_EQ(engine.GetIntervals(expr.node()).Get({}), *given.Get({}).Range());
}

TEST_F(PartialInfoQueryEngineTest, LiteralSimple) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue lit = fb.Literal(UBits(57, 20));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  // The interval set of a literal should be a single interval starting and
  // ending at its value.
  EXPECT_EQ(engine.GetIntervals(lit.node()).Get({}),
            IntervalSet::Precise(UBits(57, 20)));
}

TEST_F(PartialInfoQueryEngineTest, LiteralNonBits) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue lit = fb.Literal(
      Value::TupleOwned({Value(UBits(100, 123)), Value(UBits(140, 200)),
                         *Value::UBitsArray({27, 53, 62}, 10)}));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  // The interval set of a literal should be a single interval starting and
  // ending at its value.
  LeafTypeTree<IntervalSet> expected(lit.node()->GetType());
  expected.Set({0}, IntervalSet::Precise(UBits(100, 123)));
  expected.Set({1}, IntervalSet::Precise(UBits(140, 200)));
  expected.Set({2, 0}, IntervalSet::Precise(UBits(27, 10)));
  expected.Set({2, 1}, IntervalSet::Precise(UBits(53, 10)));
  expected.Set({2, 2}, IntervalSet::Precise(UBits(62, 10)));

  EXPECT_EQ(engine.GetIntervals(lit.node()), expected);
}

TEST_F(PartialInfoQueryEngineTest, And) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue z = fb.Param("z", p->GetBitsType(8));
  BValue expr = fb.And({x, y, z});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  // When the interval sets are precise, we know the exact result.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval::Precise(UBits(0b1111'0000, 8))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval::Precise(UBits(0b1100'1100, 8))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      z.node(), BitsLTT(z.node(), {Interval::Precise(UBits(0b1010'1010, 8))})));
  EXPECT_EQ("0b1000_0000", engine.ToString(expr.node()));

  // With one imprecise interval, things are less certain.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(),
      BitsLTT(x.node(), {Interval(UBits(0b1111'0000, 8), Bits::AllOnes(8))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval::Precise(UBits(0b1100'1100, 8))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      z.node(), BitsLTT(z.node(), {Interval::Precise(UBits(0b1010'1010, 8))})));
  EXPECT_EQ("0b1000_X000", engine.ToString(expr.node()));
}

TEST_F(PartialInfoQueryEngineTest, Nand) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue z = fb.Param("z", p->GetBitsType(8));
  BValue expr = fb.Nand({x, y, z});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  // When the interval sets are precise, we know the exact result.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval::Precise(UBits(0b1111'0000, 8))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval::Precise(UBits(0b1100'1100, 8))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      z.node(), BitsLTT(z.node(), {Interval::Precise(UBits(0b1010'1010, 8))})));
  EXPECT_EQ("0b0111_1111", engine.ToString(expr.node()));

  // With one imprecise interval, things are less certain.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(),
      BitsLTT(x.node(), {Interval(UBits(0b1111'0000, 8), Bits::AllOnes(8))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval::Precise(UBits(0b1100'1100, 8))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      z.node(), BitsLTT(z.node(), {Interval::Precise(UBits(0b1010'1010, 8))})));
  EXPECT_EQ("0b0111_X111", engine.ToString(expr.node()));
}

TEST_F(PartialInfoQueryEngineTest, Nor) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue z = fb.Param("z", p->GetBitsType(8));
  BValue expr = fb.Nor({x, y, z});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  // When the interval sets are precise, we know the exact result.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval::Precise(UBits(0b1111'0000, 8))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval::Precise(UBits(0b1100'1100, 8))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      z.node(), BitsLTT(z.node(), {Interval::Precise(UBits(0b1010'1010, 8))})));
  EXPECT_EQ("0b0000_0001", engine.ToString(expr.node()));

  // With one imprecise interval, things are less certain.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(),
      BitsLTT(x.node(), {Interval(UBits(0b1111'0000, 8), Bits::AllOnes(8))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval::Precise(UBits(0b1100'1100, 8))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      z.node(), BitsLTT(z.node(), {Interval::Precise(UBits(0b1010'1010, 8))})));
  EXPECT_EQ("0b0000_000X", engine.ToString(expr.node()));
}

TEST_F(PartialInfoQueryEngineTest, Or) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue z = fb.Param("z", p->GetBitsType(8));
  BValue expr = fb.Or({x, y, z});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  // When the interval sets are precise, we know the exact result.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval::Precise(UBits(0b1111'0000, 8))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval::Precise(UBits(0b1100'1100, 8))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      z.node(), BitsLTT(z.node(), {Interval::Precise(UBits(0b1010'1010, 8))})));
  EXPECT_EQ("0b1111_1110", engine.ToString(expr.node()));

  // With one imprecise interval, things are less certain.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(),
      BitsLTT(x.node(), {Interval(UBits(0b1111'0000, 8), Bits::AllOnes(8))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval::Precise(UBits(0b1100'1100, 8))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      z.node(), BitsLTT(z.node(), {Interval::Precise(UBits(0b1010'1010, 8))})));
  EXPECT_EQ("0b1111_111X", engine.ToString(expr.node()));
}

TEST_F(PartialInfoQueryEngineTest, Xor) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue z = fb.Param("z", p->GetBitsType(8));
  BValue expr = fb.Xor({x, y, z});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  // When the interval sets are precise, we know the exact result.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval::Precise(UBits(0b1111'0000, 8))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval::Precise(UBits(0b1100'1100, 8))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      z.node(), BitsLTT(z.node(), {Interval::Precise(UBits(0b1010'1010, 8))})));
  EXPECT_EQ("0b1001_0110", engine.ToString(expr.node()));

  // With one imprecise interval, things are less certain.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(),
      BitsLTT(x.node(), {Interval(UBits(0b1111'0000, 8), Bits::AllOnes(8))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval::Precise(UBits(0b1100'1100, 8))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      z.node(), BitsLTT(z.node(), {Interval::Precise(UBits(0b1010'1010, 8))})));
  EXPECT_EQ("0b1001_XXXX", engine.ToString(expr.node()));
}

TEST_F(PartialInfoQueryEngineTest, Ne) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue y = fb.Param("y", p->GetBitsType(20));
  BValue expr = fb.Ne(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  // When the interval sets are precise and equivalent, we know they are
  // equal.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(UBits(560, 20), UBits(560, 20))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(560, 20), UBits(560, 20)),
                                   Interval(UBits(560, 20), UBits(560, 20))})));
  EXPECT_EQ("0b0", engine.ToString(expr.node()));

  // When the interval sets overlap, we don't know anything about them.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(UBits(100, 20), UBits(200, 20))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(150, 20), UBits(300, 20)),
                                   Interval(UBits(400, 20), UBits(500, 20))})));
  EXPECT_EQ("0bX", engine.ToString(expr.node()));

  // When the interval sets are disjoint, we know they are not equal.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {
                                      Interval(UBits(100, 20), UBits(200, 20)),
                                      Interval(UBits(550, 20), UBits(600, 20)),
                                      Interval(UBits(850, 20), UBits(900, 20)),
                                  })));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(400, 20), UBits(500, 20)),
                                   Interval(UBits(700, 20), UBits(800, 20))})));
  EXPECT_EQ("0b1", engine.ToString(expr.node()));
}

TEST_F(PartialInfoQueryEngineTest, Neg) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue expr = fb.Negate(x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  // Negation is antitone.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(SBits(600, 20), SBits(700, 20))})));
  EXPECT_EQ(engine.GetIntervals(expr.node()).Get({}),
            CreateIntervalSet(20, {{-700, -600}}));
}

TEST_F(PartialInfoQueryEngineTest, OrReduce) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue expr = fb.OrReduce(x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(UBits(1, 20), UBits(48, 20))})));

  // Interval does not cover 0, so the result is known to be 1.
  EXPECT_EQ("0b1", engine.ToString(expr.node()));

  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(UBits(0, 20), UBits(48, 20))})));

  // Interval does cover 0, so we don't know the value.
  EXPECT_EQ("0bX", engine.ToString(expr.node()));

  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(UBits(0, 20), UBits(0, 20))})));

  // Interval only covers 0, so the result is known to be 0.
  EXPECT_EQ("0b0", engine.ToString(expr.node()));
}

TEST_F(PartialInfoQueryEngineTest, Param) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  // The interval set of a param should be a single interval covering all bit
  // patterns with the bit count of the param.
  EXPECT_EQ(engine.GetIntervals(x.node()).Get({}), IntervalSet::Maximal(20));
}

TEST_F(PartialInfoQueryEngineTest, Sel) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue selector = fb.Param("selector", p->GetBitsType(10));
  BValue x = fb.Param("x", p->GetBitsType(10));
  BValue y = fb.Param("y", p->GetBitsType(10));
  BValue z = fb.Param("z", p->GetBitsType(10));
  BValue def = fb.Param("def", p->GetBitsType(10));
  BValue expr = fb.Select(selector, {x, y, z}, def);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  PartialInformationTree x_given =
      BitsLTT(x.node(), {Interval(UBits(100, 10), UBits(150, 10))});
  PartialInformationTree y_given =
      BitsLTT(y.node(), {Interval(UBits(200, 10), UBits(250, 10))});
  PartialInformationTree z_given =
      BitsLTT(y.node(), {Interval(UBits(300, 10), UBits(350, 10))});
  PartialInformationTree def_given =
      BitsLTT(y.node(), {Interval(UBits(1023, 10), UBits(1023, 10))});

  XLS_ASSERT_OK(engine.ReplaceGiven(
      selector.node(),
      BitsLTT(selector.node(), {Interval(UBits(0, 10), UBits(1, 10))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), x_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(y.node(), y_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(z.node(), z_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(def.node(), def_given));

  // Usual test case where only part of the valid inputs are covered.
  EXPECT_EQ(IntervalSet::Combine(x_given.Get({}).RangeOrMaximal(),
                                 y_given.Get({}).RangeOrMaximal()),
            engine.GetIntervals(expr.node()).Get({}));

  XLS_ASSERT_OK(engine.ReplaceGiven(
      selector.node(),
      BitsLTT(selector.node(), {Interval(UBits(2, 10), UBits(10, 10))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), x_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(y.node(), y_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(z.node(), z_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(def.node(), def_given));

  // Test case where default is covered.
  EXPECT_EQ(IntervalSet::Combine(z_given.Get({}).RangeOrMaximal(),
                                 def_given.Get({}).RangeOrMaximal()),
            engine.GetIntervals(expr.node()).Get({}));
}

TEST_F(PartialInfoQueryEngineTest, SelHugeSelector) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue selector = fb.Param("selector", p->GetBitsType(100));
  BValue x = fb.Param("x", p->GetBitsType(10));
  BValue y = fb.Param("y", p->GetBitsType(10));
  BValue z = fb.Param("z", p->GetBitsType(10));
  BValue def = fb.Param("def", p->GetBitsType(10));
  BValue expr = fb.Select(selector, {x, y, z}, def);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  PartialInformationTree x_given =
      BitsLTT(x.node(), {Interval(UBits(100, 10), UBits(150, 10))});
  PartialInformationTree y_given =
      BitsLTT(y.node(), {Interval(UBits(200, 10), UBits(250, 10))});
  PartialInformationTree z_given =
      BitsLTT(y.node(), {Interval(UBits(300, 10), UBits(350, 10))});
  PartialInformationTree def_given =
      BitsLTT(y.node(), {Interval(UBits(1023, 10), UBits(1023, 10))});

  XLS_ASSERT_OK(engine.ReplaceGiven(
      selector.node(),
      BitsLTT(selector.node(),
              {Interval::Precise(bits_ops::Concat(
                  {UBits(0, 10), UBits(1, 1), UBits(0, 89)}))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), x_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(y.node(), y_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(z.node(), z_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(def.node(), def_given));

  // Test case where default is covered.
  EXPECT_EQ(def_given.Get({}).RangeOrMaximal(),
            engine.GetIntervals(expr.node()).Get({}));
}

TEST_F(PartialInfoQueryEngineTest, OneHotSelect) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue selector = fb.Param("selector", p->GetBitsType(3));
  BValue x = fb.Param("x", p->GetBitsType(10));
  BValue y = fb.Param("y", p->GetBitsType(10));
  BValue z = fb.Param("z", p->GetBitsType(10));
  BValue expr = fb.OneHotSelect(selector, {x, y, z});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  // Usual test case where only part of the valid inputs are covered.
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  PartialInformationTree x_given =
      BitsLTT(x.node(), {Interval(UBits(32, 10), UBits(33, 10))});
  PartialInformationTree y_given =
      BitsLTT(y.node(), {Interval(UBits(64, 10), UBits(66, 10))});
  PartialInformationTree z_given =
      BitsLTT(y.node(), {Interval(UBits(128, 10), UBits(135, 10))});

  XLS_ASSERT_OK(engine.ReplaceGiven(
      selector.node(),
      BitsLTT(selector.node(), {Interval(UBits(1, 3), UBits(3, 3))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), x_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(y.node(), y_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(z.node(), z_given));

  EXPECT_EQ(engine.GetIntervals(expr.node()).Get({}),
            CreateIntervalSet(10, {{0, 3}, {32, 35}, {64, 67}, {96, 99}}));

  // Test case with non-overlapping bits for selector
  // TODO(epastor): Fix test once this is better-supported by the query engine.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      selector.node(),
      BitsLTT(selector.node(), {Interval(UBits(1, 3), UBits(2, 3))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), x_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(y.node(), y_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(z.node(), z_given));

  EXPECT_EQ(engine.GetIntervals(expr.node()).Get({}),
            CreateIntervalSet(10, {{0, 3}, {32, 35}, {64, 67}, {96, 99}}));

  XLS_ASSERT_OK(engine.ReplaceGiven(
      selector.node(),
      BitsLTT(selector.node(), {Interval(UBits(2, 3), UBits(6, 3))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), x_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(y.node(), y_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(z.node(), z_given));

  EXPECT_EQ(engine.GetIntervals(expr.node()).Get({}),
            CreateIntervalSet(10, {{0, 7},
                                   {32, 39},
                                   {64, 71},
                                   {96, 103},
                                   {128, 135},
                                   {160, 167},
                                   {192, 199},
                                   {224, 231}}));

  // Test case where default is covered.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      selector.node(),
      BitsLTT(selector.node(), {Interval(UBits(0, 3), UBits(1, 3))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), x_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(y.node(), y_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(z.node(), z_given));

  EXPECT_EQ(engine.GetIntervals(expr.node()).Get({}),
            CreateIntervalSet(10, {{0, 0}, {32, 33}}));

  // Test case where x is always selected, and y may or may not be.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      selector.node(),
      BitsLTT(selector.node(), {Interval(UBits(1, 3), UBits(1, 3)),
                                Interval(UBits(3, 3), UBits(3, 3))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), x_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(y.node(), y_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(z.node(), z_given));

  EXPECT_EQ(engine.GetIntervals(expr.node()).Get({}),
            CreateIntervalSet(10, {{32, 35}, {96, 99}}));
}

TEST_F(PartialInfoQueryEngineTest, PrioritySel) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue selector = fb.Param("selector", p->GetBitsType(3));
  BValue x = fb.Param("x", p->GetBitsType(10));
  BValue y = fb.Param("y", p->GetBitsType(10));
  BValue z = fb.Param("z", p->GetBitsType(10));
  BValue d = fb.Literal(UBits(0, 10));
  BValue expr = fb.PrioritySelect(selector, {x, y, z}, d);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  // Usual test case where only part of the valid inputs are covered.
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  PartialInformationTree x_given =
      BitsLTT(x.node(), {Interval(UBits(100, 10), UBits(150, 10))});
  PartialInformationTree y_given =
      BitsLTT(y.node(), {Interval(UBits(200, 10), UBits(250, 10))});
  PartialInformationTree z_given =
      BitsLTT(y.node(), {Interval(UBits(300, 10), UBits(350, 10))});

  XLS_ASSERT_OK(engine.ReplaceGiven(
      selector.node(),
      BitsLTT(selector.node(), {Interval(UBits(1, 3), UBits(2, 3))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), x_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(y.node(), y_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(z.node(), z_given));

  EXPECT_EQ(IntervalSet::Combine(x_given.Get({}).RangeOrMaximal(),
                                 y_given.Get({}).RangeOrMaximal()),
            engine.GetIntervals(expr.node()).Get({}));

  XLS_ASSERT_OK(engine.ReplaceGiven(
      selector.node(),
      BitsLTT(selector.node(), {Interval::Precise(UBits(4, 3))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), x_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(y.node(), y_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(z.node(), z_given));

  EXPECT_EQ(z_given.Get({}).RangeOrMaximal(),
            engine.GetIntervals(expr.node()).Get({}));

  XLS_ASSERT_OK(engine.ReplaceGiven(
      selector.node(),
      BitsLTT(selector.node(), {Interval(UBits(1, 3), UBits(4, 3))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), x_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(y.node(), y_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(z.node(), z_given));

  EXPECT_EQ(IntervalSet::Combine(
                IntervalSet::Combine(x_given.Get({}).RangeOrMaximal(),
                                     y_given.Get({}).RangeOrMaximal()),
                z_given.Get({}).RangeOrMaximal()),
            engine.GetIntervals(expr.node()).Get({}));

  // Test case with overlapping bits for selector.
  // TODO(epastor): Fix test once this is better supported.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      selector.node(),
      BitsLTT(selector.node(), {Interval(UBits(5, 3), UBits(7, 3))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), x_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(y.node(), y_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(z.node(), z_given));

  EXPECT_EQ(IntervalSet::Combine(x_given.Get({}).RangeOrMaximal(),
                                 y_given.Get({}).RangeOrMaximal()),
            engine.GetIntervals(expr.node()).Get({}));

  // Test case where default is covered.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      selector.node(),
      BitsLTT(selector.node(), {Interval(UBits(0, 3), UBits(1, 3))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), x_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(y.node(), y_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(z.node(), z_given));

  EXPECT_EQ(IntervalSet::Combine(x_given.Get({}).RangeOrMaximal(),
                                 IntervalSet::Precise(UBits(0, 10))),
            engine.GetIntervals(expr.node()).Get({}));
}

TEST_F(PartialInfoQueryEngineTest, SignExtend) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue expr = fb.SignExtend(x, 40);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(),
      BitsLTT(x.node(), {Interval(SBits(-500, 20), SBits(700, 20))})));

  // Sign extension is monotone.
  EXPECT_EQ(engine.GetIntervals(expr.node()).Get({}),
            CreateIntervalSet(40, {{-500, 700}}));
}

TEST_F(PartialInfoQueryEngineTest, SignExtendFromUnknown) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue expr = fb.SignExtend(x, 40);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  // Sign extension is monotone.
  EXPECT_EQ(engine.GetIntervals(expr.node()).Get({}),
            CreateIntervalSet(40, {{-128, 127}}));
}

TEST_F(PartialInfoQueryEngineTest, Sub) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue y = fb.Param("y", p->GetBitsType(20));
  BValue expr = fb.Subtract(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(),
      BitsLTT(x.node(), {Interval(UBits(500, 20), UBits(1200, 20))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(200, 20), UBits(400, 20))})));

  // Subtraction is monotone-antitone.
  EXPECT_EQ(engine.GetIntervals(expr.node()).Get({}),
            CreateIntervalSet(20, {{100, 1000}}));
}

TEST_F(PartialInfoQueryEngineTest, Tuple) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(10));
  BValue y = fb.Param("y", p->GetBitsType(20));
  BValue expr = fb.Tuple({x, y, y});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  PartialInformationTree x_given =
      BitsLTT(x.node(), {Interval(UBits(100, 10), UBits(150, 10))});
  PartialInformationTree y_given =
      BitsLTT(y.node(), {Interval(UBits(20000, 20), UBits(25000, 20))});

  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), x_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(y.node(), y_given));

  EXPECT_EQ(x_given.Get({}).RangeOrMaximal(),
            engine.GetIntervals(expr.node()).Get({0}));
  EXPECT_EQ(y_given.Get({}).RangeOrMaximal(),
            engine.GetIntervals(expr.node()).Get({1}));
  EXPECT_EQ(y_given.Get({}).RangeOrMaximal(),
            engine.GetIntervals(expr.node()).Get({2}));

  EXPECT_EQ(IntervalSetTreeToString(engine.GetIntervals(expr.node())),
            R"((
  [[100, 150]]
  [[20000, 25000]]
  [[20000, 25000]]
)
)");
}

TEST_F(PartialInfoQueryEngineTest, TupleIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(10));
  BValue y = fb.Param("y", p->GetBitsType(20));
  BValue tuple = fb.Tuple({x, y, y});
  BValue index0 = fb.TupleIndex(tuple, 0);
  BValue index1 = fb.TupleIndex(tuple, 1);
  BValue index2 = fb.TupleIndex(tuple, 2);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  PartialInformationTree x_given =
      BitsLTT(x.node(), {Interval(UBits(100, 10), UBits(150, 10))});
  PartialInformationTree y_given =
      BitsLTT(y.node(), {Interval(UBits(20000, 20), UBits(25000, 20))});

  XLS_ASSERT_OK(engine.ReplaceGiven(x.node(), x_given));
  XLS_ASSERT_OK(engine.ReplaceGiven(y.node(), y_given));

  EXPECT_EQ(x_given.Get({}).RangeOrMaximal(),
            engine.GetIntervals(index0.node()).Get({}));
  EXPECT_EQ(y_given.Get({}).RangeOrMaximal(),
            engine.GetIntervals(index1.node()).Get({}));
  EXPECT_EQ(y_given.Get({}).RangeOrMaximal(),
            engine.GetIntervals(index2.node()).Get({}));
}

TEST_F(PartialInfoQueryEngineTest, UDiv) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue y = fb.Param("y", p->GetBitsType(20));
  BValue expr = fb.UDiv(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(),
      BitsLTT(x.node(), {Interval(UBits(16384, 20), UBits(32768, 20))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(4, 20), UBits(16, 20))})));

  // Unsigned division is monotone-antitone.
  EXPECT_EQ(engine.GetIntervals(expr.node()).Get({}),
            CreateIntervalSet(20, {{1024, 8192}}));
}

TEST_F(PartialInfoQueryEngineTest, SDiv) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue y = fb.Param("y", p->GetBitsType(20));
  BValue expr = fb.SDiv(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(),
      BitsLTT(x.node(), {Interval(UBits(16384, 20), UBits(32768, 20))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(4, 20), UBits(16, 20)),
                                   Interval(SBits(-16, 20), SBits(-4, 20))})));

  EXPECT_EQ(engine.GetIntervals(expr.node()).Get({}),
            CreateIntervalSet(20, {{-8192, -1024}, {1024, 8192}}));
}

TEST_F(PartialInfoQueryEngineTest, UGe) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue y = fb.Param("y", p->GetBitsType(20));
  BValue expr = fb.UGe(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  // When the interval sets are disjoint and greater than, returns true.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(UBits(200, 20), UBits(300, 20)),
                                   Interval(UBits(500, 20), UBits(600, 20))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(50, 20), UBits(100, 20)),
                                   Interval(UBits(120, 20), UBits(180, 20))})));
  EXPECT_EQ("0b1", engine.ToString(expr.node()));

  // When the interval sets are precise and equal, returns true.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {
                                      Interval(UBits(200, 20), UBits(200, 20)),
                                  })));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {
                                      Interval(UBits(200, 20), UBits(200, 20)),
                                  })));
  EXPECT_EQ("0b1", engine.ToString(expr.node()));

  // When the interval sets are disjoint and less than, returns false.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(UBits(50, 20), UBits(100, 20)),
                                   Interval(UBits(120, 20), UBits(180, 20))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(200, 20), UBits(300, 20)),
                                   Interval(UBits(500, 20), UBits(600, 20))})));
  EXPECT_EQ("0b0", engine.ToString(expr.node()));

  // When the convex hulls overlap, we don't know anything.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {
                                      Interval(UBits(100, 20), UBits(200, 20)),
                                      Interval(UBits(550, 20), UBits(600, 20)),
                                      Interval(UBits(850, 20), UBits(900, 20)),
                                  })));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(400, 20), UBits(500, 20)),
                                   Interval(UBits(700, 20), UBits(800, 20))})));
  EXPECT_EQ("0bX", engine.ToString(expr.node()));
}

TEST_F(PartialInfoQueryEngineTest, UGt) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue y = fb.Param("y", p->GetBitsType(20));
  BValue expr = fb.UGt(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  // When the interval sets are disjoint and greater than, returns true.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(UBits(200, 20), UBits(300, 20)),
                                   Interval(UBits(500, 20), UBits(600, 20))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(50, 20), UBits(100, 20)),
                                   Interval(UBits(120, 20), UBits(180, 20))})));
  EXPECT_EQ("0b1", engine.ToString(expr.node()));

  // When the interval sets are precise and equal, returns false.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {
                                      Interval(UBits(200, 20), UBits(200, 20)),
                                  })));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {
                                      Interval(UBits(200, 20), UBits(200, 20)),
                                  })));
  EXPECT_EQ("0b0", engine.ToString(expr.node()));

  // When the interval sets are disjoint and less than, returns false.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(UBits(50, 20), UBits(100, 20)),
                                   Interval(UBits(120, 20), UBits(180, 20))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(200, 20), UBits(300, 20)),
                                   Interval(UBits(500, 20), UBits(600, 20))})));
  EXPECT_EQ("0b0", engine.ToString(expr.node()));

  // When the convex hulls overlap, we don't know anything.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {
                                      Interval(UBits(100, 20), UBits(200, 20)),
                                      Interval(UBits(550, 20), UBits(600, 20)),
                                      Interval(UBits(850, 20), UBits(900, 20)),
                                  })));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(400, 20), UBits(500, 20)),
                                   Interval(UBits(700, 20), UBits(800, 20))})));
  EXPECT_EQ("0bX", engine.ToString(expr.node()));
}

TEST_F(PartialInfoQueryEngineTest, ULe) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue y = fb.Param("y", p->GetBitsType(20));
  BValue expr = fb.ULe(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  // When the interval sets are disjoint and greater than, returns false.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(UBits(200, 20), UBits(300, 20)),
                                   Interval(UBits(500, 20), UBits(600, 20))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(50, 20), UBits(100, 20)),
                                   Interval(UBits(120, 20), UBits(180, 20))})));
  EXPECT_EQ("0b0", engine.ToString(expr.node()));

  // When the interval sets are precise and equal, returns true.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {
                                      Interval(UBits(200, 20), UBits(200, 20)),
                                  })));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {
                                      Interval(UBits(200, 20), UBits(200, 20)),
                                  })));
  EXPECT_EQ("0b1", engine.ToString(expr.node()));

  // When the interval sets are disjoint and less than, returns true.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(UBits(50, 20), UBits(100, 20)),
                                   Interval(UBits(120, 20), UBits(180, 20))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(200, 20), UBits(300, 20)),
                                   Interval(UBits(500, 20), UBits(600, 20))})));
  EXPECT_EQ("0b1", engine.ToString(expr.node()));

  // When the convex hulls overlap, we don't know anything.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {
                                      Interval(UBits(100, 20), UBits(200, 20)),
                                      Interval(UBits(550, 20), UBits(600, 20)),
                                      Interval(UBits(850, 20), UBits(900, 20)),
                                  })));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(400, 20), UBits(500, 20)),
                                   Interval(UBits(700, 20), UBits(800, 20))})));
  EXPECT_EQ("0bX", engine.ToString(expr.node()));
}

TEST_F(PartialInfoQueryEngineTest, ULt) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue y = fb.Param("y", p->GetBitsType(20));
  BValue expr = fb.ULt(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  // When the interval sets are disjoint and greater than, returns false.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(UBits(200, 20), UBits(300, 20)),
                                   Interval(UBits(500, 20), UBits(600, 20))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(50, 20), UBits(100, 20)),
                                   Interval(UBits(120, 20), UBits(180, 20))})));
  EXPECT_EQ("0b0", engine.ToString(expr.node()));

  // When the interval sets are precise and equal, returns false.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {
                                      Interval(UBits(200, 20), UBits(200, 20)),
                                  })));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {
                                      Interval(UBits(200, 20), UBits(200, 20)),
                                  })));
  EXPECT_EQ("0b0", engine.ToString(expr.node()));

  // When the interval sets are disjoint and less than, returns true.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(UBits(50, 20), UBits(100, 20)),
                                   Interval(UBits(120, 20), UBits(180, 20))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(200, 20), UBits(300, 20)),
                                   Interval(UBits(500, 20), UBits(600, 20))})));
  EXPECT_EQ("0b1", engine.ToString(expr.node()));

  // When the convex hulls overlap, we don't know anything.
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {
                                      Interval(UBits(100, 20), UBits(200, 20)),
                                      Interval(UBits(550, 20), UBits(600, 20)),
                                      Interval(UBits(850, 20), UBits(900, 20)),
                                  })));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(400, 20), UBits(500, 20)),
                                   Interval(UBits(700, 20), UBits(800, 20))})));
  EXPECT_EQ("0bX", engine.ToString(expr.node()));
}

TEST_F(PartialInfoQueryEngineTest, SMul) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue expr = fb.SMul(x, y, 8);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(SBits(-2, 8), SBits(-1, 8)),
                                   Interval(UBits(0, 8), UBits(2, 8))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(10, 8), UBits(20, 8))})));

  // Unsigned multiplication is monotone-monotone.
  EXPECT_EQ(engine.GetIntervals(expr.node()).Get({}),
            CreateIntervalSet(8, {{-40, -10}, {0, 0}, {10, 40}}));
}

TEST_F(PartialInfoQueryEngineTest, UMul) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(7));
  BValue y = fb.Param("y", p->GetBitsType(7));
  BValue expr = fb.UMul(x, y, 14);
  BValue overflow = fb.UMul(x, y, 12);
  BValue overflow2 = fb.UMul(x, y, 8);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(UBits(50, 7), UBits(100, 7))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(10, 7), UBits(20, 7))})));

  // Unsigned multiplication is monotone-monotone.
  EXPECT_EQ(engine.GetIntervals(expr.node()).Get({}),
            CreateIntervalSet(14, {{500, 2000}}));

  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(UBits(62, 7), UBits(67, 7))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(61, 7), UBits(70, 7))})));

  // If the multiplication can overflow without covering the whole range.
  // 62 * 61 == 3782, 67 * 70 == 4690, 1 << 12 == 4096
  // Overflows top once but not bottom so range can be split
  EXPECT_EQ(engine.GetIntervals(overflow.node()).Get({}),
            CreateIntervalSet(12, {{0, 67 * 70 % (1 << 12)}, {3782, 4095}}));
  // If the multiplication can overflow on both the low and the high side we
  // currently don't try to determine the range and fallback to maximal.
  EXPECT_EQ(engine.GetIntervals(overflow2.node()).Get({}),
            IntervalSet::Maximal(8));
}

TEST_F(PartialInfoQueryEngineTest, UMulPowerOfTwo) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u3 = p->GetBitsType(3);
  BValue z = fb.Param("z", u3);
  BValue index = fb.UMul(fb.ZeroExtend(z, 5), fb.Literal(UBits(4, 5)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));
  EXPECT_EQ(engine.GetTernary(index.node())->Get({}),
            *StringToTernaryVector("0bXXX00"));
}

TEST_F(PartialInfoQueryEngineTest, UMulOverflow) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue expr = fb.UMul(x, y, 8);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(UBits(0, 8), UBits(4, 8))})));
  XLS_ASSERT_OK(
      engine.ReplaceGiven(y.node(), BitsLTT(y.node(), {Interval::Maximal(8)})));

  // Overflow so we have maximal range
  EXPECT_EQ(engine.GetIntervals(expr.node()).Get({}), IntervalSet::Maximal(8));
}

TEST_F(PartialInfoQueryEngineTest, UMulNoOverflow) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue expr = fb.UMul(x, y, 8);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(UBits(0, 8), UBits(4, 8))})));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      y.node(), BitsLTT(y.node(), {Interval(UBits(0, 8), UBits(4, 8))})));

  // None of the results can overflow so we have a real range.
  EXPECT_EQ(engine.GetIntervals(expr.node()).Get({}),
            CreateIntervalSet(8, {{0, 16}}));
}

TEST_F(PartialInfoQueryEngineTest, XorReduce) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue expr = fb.XorReduce(x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));

  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(),
      BitsLTT(x.node(), {Interval(UBits(0b1, 20), UBits(0b1, 20)),
                         Interval(UBits(0b1110, 20), UBits(0b1110, 20))})));

  // Interval set covers only numbers with an odd number of 1s, so result
  // is 1.
  EXPECT_EQ("0b1", engine.ToString(expr.node()));

  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(UBits(0, 20), UBits(48, 20))})));

  // Interval is not precise, so result is unknown.
  EXPECT_EQ("0bX", engine.ToString(expr.node()));

  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(),
      BitsLTT(x.node(), {Interval(UBits(0b1110, 20), UBits(0b1110, 20)),
                         Interval(UBits(0b1001, 20), UBits(0b1001, 20))})));

  // Intervals are precise, but don't match parity of 1s, so result is
  // unknown.
  EXPECT_EQ("0bX", engine.ToString(expr.node()));

  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(),
      BitsLTT(x.node(), {Interval(UBits(0b11, 20), UBits(0b11, 20))})));

  // Interval only covers numbers with an even number of 1s, so result is 0.
  EXPECT_EQ("0b0", engine.ToString(expr.node()));
}

TEST_F(PartialInfoQueryEngineTest, ZeroExtend) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue expr = fb.ZeroExtend(x, 40);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  PartialInfoQueryEngine engine;
  XLS_ASSERT_OK(engine.Populate(f));
  XLS_ASSERT_OK(engine.ReplaceGiven(
      x.node(), BitsLTT(x.node(), {Interval(UBits(500, 20), UBits(700, 20))})));

  // Zero extension is monotone.
  EXPECT_EQ(engine.GetIntervals(expr.node()).Get({}),
            CreateIntervalSet(40, {{500, 700}}));
}

}  // namespace
}  // namespace xls
