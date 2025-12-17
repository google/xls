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

#include "xls/passes/bit_provenance_analysis.h"

#include <memory>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/passes/query_engine.h"

namespace xls {
namespace {
using ::testing::ElementsAre;

class BitProvenanceAnalysisTest : public IrTestBase {};

MATCHER_P(
    IsTreeBitSources, ranges,
    absl::StrFormat(
        "Tree bit span with ranges [%s]",
        testing::DescribeMatcher<absl::Span<const TreeBitSources::BitRange>>(
            ranges))) {
  const TreeBitSources& tbs = arg;
  return testing::ExplainMatchResult(ranges, tbs.ranges(), result_listener);
}

MATCHER_P4(IsBitRangeM, node, source_bit_index_low, dest_bit_index_low,
           bit_width,
           absl::StrFormat("BitRange source_node: %s source_bit_index_low: %s  "
                           "dest_bit_index_low: %s width: %s",
                           testing::DescribeMatcher<Node*>(node),
                           testing::DescribeMatcher<int>(source_bit_index_low),
                           testing::DescribeMatcher<int>(dest_bit_index_low),
                           testing::DescribeMatcher<int>(bit_width))) {
  const TreeBitSources::BitRange& r = arg;
  return testing::ExplainMatchResult(node, r.source_node(), result_listener) &&
         testing::ExplainMatchResult(
             source_bit_index_low, r.source_bit_index_low(), result_listener) &&
         testing::ExplainMatchResult(dest_bit_index_low, r.dest_bit_index_low(),
                                     result_listener) &&
         testing::ExplainMatchResult(bit_width, r.bit_width(), result_listener);
}
auto IsBitRange(auto node, auto source_bit_index_low, auto dest_bit_index_low,
                auto bit_width) {
  return IsBitRangeM(node, source_bit_index_low, dest_bit_index_low, bit_width);
}

TEST_F(BitProvenanceAnalysisTest, SelfGenerated) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue high = fb.Param("high_bits", p->GetBitsType(8));
  BValue low = fb.Param("low_bits", p->GetBitsType(8));
  BValue res = fb.Add(high, low);
  XLS_ASSERT_OK(fb.Build());
  BitProvenanceAnalysis bpa;
  for (int i = 0; i < 8; ++i) {
    XLS_ASSERT_OK_AND_ASSIGN(TreeBitLocation source,
                             bpa.GetSource(TreeBitLocation(res.node(), i)));
    EXPECT_EQ(source, TreeBitLocation(res.node(), i));
  }
  for (int i = 0; i < 8; ++i) {
    XLS_ASSERT_OK_AND_ASSIGN(TreeBitLocation source,
                             bpa.GetSource(TreeBitLocation(high.node(), i)));
    EXPECT_EQ(source, TreeBitLocation(high.node(), i));
  }
  for (int i = 0; i < 8; ++i) {
    XLS_ASSERT_OK_AND_ASSIGN(TreeBitLocation source,
                             bpa.GetSource(TreeBitLocation(low.node(), i)));
    EXPECT_EQ(source, TreeBitLocation(low.node(), i));
  }
  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTreeView<TreeBitSources> res_sources,
                           bpa.GetBitSources(res.node()));
  EXPECT_THAT(res_sources.Get({}),
              IsTreeBitSources(ElementsAre(
                  IsBitRange(res.node(), /*source_bit_index_low=*/0,
                             /*dest_bit_index_low=*/0, /*bit_width=*/8))));
  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTreeView<TreeBitSources> low_sources,
                           bpa.GetBitSources(low.node()));
  EXPECT_THAT(low_sources.Get({}),
              IsTreeBitSources(ElementsAre(
                  IsBitRange(low.node(), /*source_bit_index_low=*/0,
                             /*dest_bit_index_low=*/0, /*bit_width=*/8))));
  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTreeView<TreeBitSources> high_sources,
                           bpa.GetBitSources(high.node()));
  EXPECT_THAT(high_sources.Get({}),
              IsTreeBitSources(ElementsAre(
                  IsBitRange(high.node(), /*source_bit_index_low=*/0,
                             /*dest_bit_index_low=*/0, /*bit_width=*/8))));
  RecordProperty("res", absl::StrCat(res_sources.Get({})));
}

TEST_F(BitProvenanceAnalysisTest, Concat) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue high = fb.Param("high_bits", p->GetBitsType(8));
  BValue low = fb.Param("low_bits", p->GetBitsType(24));
  BValue res = fb.Concat({high, low});
  XLS_ASSERT_OK(fb.Build());
  BitProvenanceAnalysis bpa;
  for (int i = 0; i < 24; ++i) {
    XLS_ASSERT_OK_AND_ASSIGN(TreeBitLocation source,
                             bpa.GetSource(TreeBitLocation(res.node(), i)));
    EXPECT_EQ(source, TreeBitLocation(low.node(), i));
  }
  for (int i = 0; i < 8; ++i) {
    XLS_ASSERT_OK_AND_ASSIGN(
        TreeBitLocation source,
        bpa.GetSource(TreeBitLocation(res.node(), i + 24)));
    EXPECT_EQ(source, TreeBitLocation(high.node(), i));
  }
  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTreeView<TreeBitSources> res_sources,
                           bpa.GetBitSources(res.node()));
  EXPECT_THAT(res_sources.Get({}),
              IsTreeBitSources(ElementsAre(
                  IsBitRange(low.node(), /*source_bit_index_low=*/0,
                             /*dest_bit_index_low=*/0, /*bit_width=*/24),
                  IsBitRange(high.node(), /*source_bit_index_low=*/0,
                             /*dest_bit_index_low=*/24, /*bit_width=*/8))));
  RecordProperty("res", absl::StrCat(res_sources.Get({})));
}

TEST_F(BitProvenanceAnalysisTest, BitSlice) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue high = fb.Param("high_bits", p->GetBitsType(32));
  BValue res = fb.BitSlice(high, 4, 8);

  XLS_ASSERT_OK(fb.Build());
  BitProvenanceAnalysis bpa;
  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTreeView<TreeBitSources> res_sources,
                           bpa.GetBitSources(res.node()));
  EXPECT_THAT(res_sources.Get({}),
              IsTreeBitSources(ElementsAre(
                  IsBitRange(high.node(), /*source_bit_index_low=*/4,
                             /*dest_bit_index_low=*/0, /*bit_width=*/8))));
  RecordProperty("res", absl::StrCat(res_sources.Get({})));
}

TEST_F(BitProvenanceAnalysisTest, SignExtend) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue bits = fb.Param("bits", p->GetBitsType(10));
  BValue bitslice = fb.BitSlice(bits, 2, 8);
  BValue res = fb.SignExtend(bitslice, 12);

  XLS_ASSERT_OK(fb.Build());
  BitProvenanceAnalysis bpa;
  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTreeView<TreeBitSources> res_sources,
                           bpa.GetBitSources(res.node()));
  RecordProperty("res", absl::StrCat(res_sources.Get({})));
  EXPECT_THAT(res_sources.Get({}),
              IsTreeBitSources(ElementsAre(
                  IsBitRange(bits.node(), /*source_bit_index_low=*/2,
                             /*dest_bit_index_low=*/0, /*bit_width=*/8),
                  IsBitRange(bits.node(), /*source_bit_index_low=*/9,
                             /*dest_bit_index_low=*/8, /*bit_width=*/1),
                  IsBitRange(bits.node(), /*source_bit_index_low=*/9,
                             /*dest_bit_index_low=*/9, /*bit_width=*/1),
                  IsBitRange(bits.node(), /*source_bit_index_low=*/9,
                             /*dest_bit_index_low=*/10, /*bit_width=*/1),
                  IsBitRange(bits.node(), /*source_bit_index_low=*/9,
                             /*dest_bit_index_low=*/11, /*bit_width=*/1))));
}

TEST_F(BitProvenanceAnalysisTest, ZeroExtend) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue bits = fb.Param("bits", p->GetBitsType(8));
  BValue res = fb.ZeroExtend(bits, 32);

  XLS_ASSERT_OK(fb.Build());
  BitProvenanceAnalysis bpa;
  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTreeView<TreeBitSources> res_sources,
                           bpa.GetBitSources(res.node()));
  RecordProperty("res", absl::StrCat(res_sources.Get({})));
  EXPECT_THAT(res_sources.Get({}),
              IsTreeBitSources(ElementsAre(
                  IsBitRange(bits.node(), /*source_bit_index_low=*/0,
                             /*dest_bit_index_low=*/0, /*bit_width=*/8),
                  IsBitRange(res.node(), /*source_bit_index_low=*/8,
                             /*dest_bit_index_low=*/8, /*bit_width=*/24))));
}

TEST_F(BitProvenanceAnalysisTest, Sel) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue start = fb.Param("start", p->GetBitsType(8));
  BValue left = fb.Param("left", p->GetBitsType(16));
  BValue right = fb.Param("right", p->GetBitsType(16));
  BValue end = fb.Param("end", p->GetBitsType(8));
  BValue left_concat = fb.Concat({end, left, start});
  BValue right_concat = fb.Concat({end, right, start});
  BValue res = fb.Select(fb.Param("sel", p->GetBitsType(1)),
                         {left_concat, right_concat});

  XLS_ASSERT_OK(fb.Build());
  BitProvenanceAnalysis bpa;
  RecordProperty("ir", p->DumpIr());

  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTreeView<TreeBitSources> res_sources,
                           bpa.GetBitSources(res.node()));
  RecordProperty("res", absl::StrCat(res_sources.Get({})));
  EXPECT_THAT(res_sources.Get({}),
              IsTreeBitSources(ElementsAre(
                  IsBitRange(start.node(), /*source_bit_index_low=*/0,
                             /*dest_bit_index_low=*/0, /*bit_width=*/8),
                  IsBitRange(res.node(), /*source_bit_index_low=*/8,
                             /*dest_bit_index_low=*/8, /*bit_width=*/16),
                  IsBitRange(end.node(), /*source_bit_index_low=*/0,
                             /*dest_bit_index_low=*/24, /*bit_width=*/8))));
  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTreeView<TreeBitSources> left_sources,
                           bpa.GetBitSources(left_concat.node()));
  EXPECT_THAT(left_sources.Get({}),
              IsTreeBitSources(ElementsAre(
                  IsBitRange(start.node(), /*source_bit_index_low=*/0,
                             /*dest_bit_index_low=*/0, /*bit_width=*/8),
                  IsBitRange(left.node(), /*source_bit_index_low=*/0,
                             /*dest_bit_index_low=*/8, /*bit_width=*/16),
                  IsBitRange(end.node(), /*source_bit_index_low=*/0,
                             /*dest_bit_index_low=*/24, /*bit_width=*/8))));
  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTreeView<TreeBitSources> right_sources,
                           bpa.GetBitSources(right_concat.node()));
  EXPECT_THAT(right_sources.Get({}),
              IsTreeBitSources(ElementsAre(
                  IsBitRange(start.node(), /*source_bit_index_low=*/0,
                             /*dest_bit_index_low=*/0, /*bit_width=*/8),
                  IsBitRange(right.node(), /*source_bit_index_low=*/0,
                             /*dest_bit_index_low=*/8, /*bit_width=*/16),
                  IsBitRange(end.node(), /*source_bit_index_low=*/0,
                             /*dest_bit_index_low=*/24, /*bit_width=*/8))));
}

TEST_F(BitProvenanceAnalysisTest, RangesMinimized) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue start = fb.Param("start", p->GetBitsType(32));
  BValue low = fb.BitSlice(start, 0, 16);
  BValue high = fb.BitSlice(start, 16, 16);
  BValue res = fb.Concat({high, low});
  XLS_ASSERT_OK(fb.Build());
  BitProvenanceAnalysis bpa;
  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTreeView<TreeBitSources> res_sources,
                           bpa.GetBitSources(res.node()));
  EXPECT_THAT(res_sources.Get({}),
              IsTreeBitSources(ElementsAre(
                  IsBitRange(start.node(), /*source_bit_index_low=*/0,
                             /*dest_bit_index_low=*/0, /*bit_width=*/32))));
  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTreeView<TreeBitSources> low_sources,
                           bpa.GetBitSources(low.node()));
  EXPECT_THAT(low_sources.Get({}),
              IsTreeBitSources(ElementsAre(
                  IsBitRange(start.node(), /*source_bit_index_low=*/0,
                             /*dest_bit_index_low=*/0, /*bit_width=*/16))));
  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTreeView<TreeBitSources> high_sources,
                           bpa.GetBitSources(high.node()));
  EXPECT_THAT(high_sources.Get({}),
              IsTreeBitSources(ElementsAre(
                  IsBitRange(start.node(), /*source_bit_index_low=*/16,
                             /*dest_bit_index_low=*/0, /*bit_width=*/16))));
}

TEST_F(BitProvenanceAnalysisTest, Literal) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue st = fb.Literal(UBits(0, 32));
  XLS_ASSERT_OK(fb.Build());
  BitProvenanceAnalysis bpa;
  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTreeView<TreeBitSources> st_sources,
                           bpa.GetBitSources(st.node()));
  EXPECT_THAT(st_sources.Get({}),
              IsTreeBitSources(ElementsAre(IsBitRange(st.node(), 0, 0, 32))));
}

TEST_F(BitProvenanceAnalysisTest, ZeroLenBits) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue l1 = fb.Literal(UBits(0, 1));
  BValue lz = fb.Literal(UBits(0, 0));
  BValue conc = fb.Concat({l1, l1, lz, l1});
  BValue slice = fb.BitSlice(conc, 0, 2);
  XLS_ASSERT_OK(fb.Build());
  BitProvenanceAnalysis bpa;
  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTreeView<TreeBitSources> slice_sources,
                           bpa.GetBitSources(slice.node()));
  EXPECT_THAT(slice_sources.Get({}),
              IsTreeBitSources(ElementsAre(IsBitRange(l1.node(), 0, 0, 1),
                                           IsBitRange(l1.node(), 0, 1, 1))));
}

TEST_F(BitProvenanceAnalysisTest, MultipleConcat) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue st = fb.Param("foo", p->GetBitsType(128));
  BValue value = fb.Param("value", p->GetBitsType(32));
  BValue top_64 = fb.BitSlice(st, 64, 64);
  BValue bottom_32 = fb.BitSlice(st, 0, 32);
  BValue res = fb.Concat({top_64, value, bottom_32});

  XLS_ASSERT_OK(fb.Build());
  BitProvenanceAnalysis bpa;
  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTreeView<TreeBitSources> res_sources,
                           bpa.GetBitSources(res.node()));
  EXPECT_THAT(res_sources.Get({}),
              IsTreeBitSources(ElementsAre(
                  IsBitRange(st.node(), /*source_bit_index_low=*/0,
                             /*dest_bit_index_low=*/0, /*bit_width=*/32),
                  IsBitRange(value.node(), /*source_bit_index_low=*/0,
                             /*dest_bit_index_low=*/32, /*bit_width=*/32),
                  IsBitRange(st.node(), /*source_bit_index_low=*/64,
                             /*dest_bit_index_low=*/64, /*bit_width=*/64))));
}

TEST_F(BitProvenanceAnalysisTest, ComputesLazily) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue x_bit0 = fb.BitSlice(x, 0, 1);
  BValue concat_xy = fb.Concat({x, y});
  BValue concat_xy_bits345 = fb.BitSlice(concat_xy, 3, 3);
  BValue concat_xy_bit4 = fb.BitSlice(concat_xy_bits345, 1, 1);
  XLS_ASSERT_OK(fb.BuildWithReturnValue(fb.Tuple({x_bit0, concat_xy_bit4})));
  BitProvenanceAnalysis bpa;

  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTreeView<TreeBitSources> bit4_source,
                           bpa.GetBitSources(concat_xy_bit4.node()));
  EXPECT_TRUE(bpa.IsTracked(concat_xy.node()));
  EXPECT_TRUE(bpa.IsTracked(concat_xy_bits345.node()));
  EXPECT_TRUE(bpa.IsTracked(concat_xy_bit4.node()));
  EXPECT_FALSE(bpa.IsTracked(x_bit0.node()));
  EXPECT_THAT(bit4_source.Get({}),
              IsTreeBitSources(ElementsAre(
                  IsBitRange(y.node(), /*source_bit_index_low=*/4,
                             /*dest_bit_index_low=*/0, /*bit_width=*/1))));

  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTreeView<TreeBitSources> x_bit0_sources,
                           bpa.GetBitSources(x_bit0.node()));
  EXPECT_TRUE(bpa.IsTracked(x_bit0.node()));
  EXPECT_THAT(x_bit0_sources.Get({}),
              IsTreeBitSources(ElementsAre(
                  IsBitRange(x.node(), /*source_bit_index_low=*/0,
                             /*dest_bit_index_low=*/0, /*bit_width=*/1))));
}

TEST_F(BitProvenanceAnalysisTest, TrimLeadingAndTrailingRepeatedSourceBits) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue x_5 = fb.BitSlice(x, 5, 1);
  BValue x_mid_16 = fb.BitSlice(x, 8, 16);
  BValue padded = fb.Concat({fb.SignExtend(x_mid_16, 19), x_5, x_5});
  XLS_ASSERT_OK(fb.Build());
  BitProvenanceAnalysis bpa;
  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTreeView<TreeBitSources> padded_sources,
                           bpa.GetBitSources(padded.node()));
  // Confirm padded = {x[23], x[23], x[23], x[8:24], x[5], x[5]}
  EXPECT_THAT(padded_sources.Get({}),
              IsTreeBitSources(ElementsAre(
                  IsBitRange(x.node(), /*source_bit_index_low=*/5,
                             /*dest_bit_index_low=*/0, /*bit_width=*/1),
                  IsBitRange(x.node(), /*source_bit_index_low=*/5,
                             /*dest_bit_index_low=*/1, /*bit_width=*/1),
                  IsBitRange(x.node(), /*source_bit_index_low=*/8,
                             /*dest_bit_index_low=*/2, /*bit_width=*/16),
                  IsBitRange(x.node(), /*source_bit_index_low=*/23,
                             /*dest_bit_index_low=*/18, /*bit_width=*/1),
                  IsBitRange(x.node(), /*source_bit_index_low=*/23,
                             /*dest_bit_index_low=*/19, /*bit_width=*/1),
                  IsBitRange(x.node(), /*source_bit_index_low=*/23,
                             /*dest_bit_index_low=*/20, /*bit_width=*/1))));

  // Confirm trimmed = {x[8:24], x[5]}
  EXPECT_THAT(
      BitProvenanceAnalysis::TrimRepeatedSourceBits(std::move(padded_sources))
          .Get({}),
      IsTreeBitSources(
          ElementsAre(IsBitRange(x.node(), /*source_bit_index_low=*/5,
                                 /*dest_bit_index_low=*/1, /*bit_width=*/1),
                      IsBitRange(x.node(), /*source_bit_index_low=*/8,
                                 /*dest_bit_index_low=*/2, /*bit_width=*/16))));
}

TEST_F(BitProvenanceAnalysisTest, NoRepeatedSourceBitsToTrim) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue x_3 = fb.BitSlice(x, 3, 1);
  XLS_ASSERT_OK(fb.Build());
  BitProvenanceAnalysis bpa;
  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTreeView<TreeBitSources> no_repeated,
                           bpa.GetBitSources(x_3.node()));
  EXPECT_THAT(
      BitProvenanceAnalysis::TrimRepeatedSourceBits(std::move(no_repeated))
          .Get({}),
      IsTreeBitSources(
          ElementsAre(IsBitRange(x.node(), /*source_bit_index_low=*/3,
                                 /*dest_bit_index_low=*/0, /*bit_width=*/1))));
}

TEST_F(BitProvenanceAnalysisTest, SingleSourceBitRepeatedMultipleTimes) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue x_4 = fb.Concat({fb.BitSlice(x, 4, 1), fb.BitSlice(x, 4, 1),
                          fb.BitSlice(x, 4, 1), fb.BitSlice(x, 4, 1)});
  XLS_ASSERT_OK(fb.Build());
  BitProvenanceAnalysis bpa;
  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTreeView<TreeBitSources> x_4_sources,
                           bpa.GetBitSources(x_4.node()));
  EXPECT_THAT(
      BitProvenanceAnalysis::TrimRepeatedSourceBits(std::move(x_4_sources))
          .Get({}),
      IsTreeBitSources(
          ElementsAre(IsBitRange(x.node(), /*source_bit_index_low=*/4,
                                 /*dest_bit_index_low=*/3, /*bit_width=*/1))));
}

}  // namespace
}  // namespace xls
