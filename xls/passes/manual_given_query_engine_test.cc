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

#include "xls/passes/manual_given_query_engine.h"

#include <optional>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/interval_ops.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/ternary.h"
#include "xls/passes/lazy_ternary_query_engine.h"
#include "xls/passes/partial_info_query_engine.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/range_query_engine.h"

namespace xls {
namespace {

class ManualGivenQueryEngineTest : public IrTestBase {
 protected:
  ManualGivenQueryEngineTest() = default;
};

class DirectQueryEngine final : public QueryEngine {
 public:
  bool HasOverride(Node* node) const { return override_.contains(node); }
  DirectQueryEngine& Override(Node* n, const TernaryVector& ternary) {
    CHECK(n->GetType()->IsBits());
    auto interval = interval_ops::FromTernary(ternary);
    override_[n] = ValueKnowledge{
        .ternary =
            LeafTypeTree<TernaryVector>(n->GetType(), {std::move(ternary)}),
        .intervals =
            LeafTypeTree<IntervalSet>(n->GetType(), {std::move(interval)}),
    };
    return *this;
  }
  DirectQueryEngine& Override(Node* n, ValueKnowledge&& knowledge) {
    override_[n] = std::move(knowledge);
    return *this;
  }
  DirectQueryEngine& Override(Node* n, const ValueKnowledge& knowledge) {
    override_[n] = knowledge;
    return *this;
  }

  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override {
    return ReachedFixpoint::Changed;
  }

  std::optional<SharedLeafTypeTree<TernaryVector>> GetTernary(
      Node* node) const override {
    return HasOverride(node)
               ? std::make_optional<SharedLeafTypeTree<TernaryVector>>(
                     override_.at(node).ternary->AsView().AsShared())
               : std::nullopt;
  }

  LeafTypeTree<IntervalSet> GetIntervals(Node* node) const override {
    if (HasOverride(node) && override_.at(node).intervals.has_value()) {
      return *override_.at(node).intervals;
    }
    return QueryEngine::GetIntervals(node);
  }

  bool AtMostOneTrue(absl::Span<const TreeBitLocation> bits) const override {
    return false;
  }

  bool AtLeastOneTrue(absl::Span<const TreeBitLocation> bits) const override {
    return false;
  }

  bool Implies(const TreeBitLocation& a,
               const TreeBitLocation& b) const override {
    return false;
  }

  std::optional<Bits> ImpliedNodeValue(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override {
    return std::nullopt;
  }

  std::optional<TernaryVector> ImpliedNodeTernary(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override {
    auto tree = GetTernary(node);
    if (tree && node->GetType()->IsBits()) {
      return tree->Get({});
    }
    return std::nullopt;
  }

  bool KnownEquals(const TreeBitLocation& a,
                   const TreeBitLocation& b) const override {
    return false;
  }

  bool KnownNotEquals(const TreeBitLocation& a,
                      const TreeBitLocation& b) const override {
    return false;
  }

  bool IsTracked(Node* node) const override { return true; }

  bool IsAllOnes(Node* node) const override {
    return HasOverride(node) && override_.at(node).ternary
               ? absl::c_all_of(override_.at(node).ternary->elements(),
                                ternary_ops::IsKnownOne)
               : false;
  }

  bool IsAllZeros(Node* node) const override {
    return HasOverride(node) && override_.at(node).ternary
               ? absl::c_all_of(override_.at(node).ternary->elements(),
                                ternary_ops::IsKnownZero)
               : false;
  }

 private:
  absl::flat_hash_map<Node* const, ValueKnowledge> override_;
};

TEST_F(ManualGivenQueryEngineTest, TestOverrideAll) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue y = fb.Param("y", p->GetBitsType(4));
  fb.Add(fb.ZeroExtend(x, 5), fb.ZeroExtend(y, 5));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  DirectQueryEngine base;
  XLS_ASSERT_OK_AND_ASSIGN(auto x_ternary, StringToTernaryVector("0bX00X"));
  XLS_ASSERT_OK_AND_ASSIGN(auto y_ternary, StringToTernaryVector("0bX00X"));
  base.Override(x.node(), x_ternary).Override(y.node(), y_ternary);
  LazyTernaryQueryEngine orig;
  XLS_ASSERT_OK(orig.Populate(f).status());
  ManualGivenQueryEngine mg =
      ManualGivenQueryEngine::Of(LazyTernaryQueryEngine());
  XLS_ASSERT_OK(mg.Populate(f).status());
  EXPECT_FALSE(mg.has_givens());
  XLS_ASSERT_OK(mg.SetGivens({x.node(), y.node()}, base));
  EXPECT_TRUE(mg.has_givens());
  RecordProperty("res", mg.GetTernary(f->return_value())->ToString());
  EXPECT_THAT(mg.GetTernary(f->return_value()),
              testing::Optional(LttIs<TernaryVector>(
                  testing::ElementsAre(TernaryIs("0bXX0XX")))));
}

TEST_F(ManualGivenQueryEngineTest, OverrideSubsetOfNodes) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue y = fb.Param("y", p->GetBitsType(4));
  fb.Add(fb.ZeroExtend(x, 5), fb.ZeroExtend(y, 5));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  DirectQueryEngine base;
  XLS_ASSERT_OK_AND_ASSIGN(auto x_ternary, StringToTernaryVector("0b001X"));
  XLS_ASSERT_OK_AND_ASSIGN(auto y_ternary, StringToTernaryVector("0b0100"));
  base.Override(x.node(), x_ternary).Override(y.node(), y_ternary);

  ManualGivenQueryEngine mg =
      ManualGivenQueryEngine::Of(LazyTernaryQueryEngine());
  XLS_ASSERT_OK(mg.Populate(f).status());
  XLS_ASSERT_OK(mg.SetGivens({x.node()}, base));

  EXPECT_THAT(mg.GetTernary(x.node()),
              testing::Optional(LttIs<TernaryVector>(
                  testing::ElementsAre(TernaryIs("0b001X")))));
  EXPECT_THAT(mg.GetTernary(y.node()),
              testing::Optional(LttIs<TernaryVector>(
                  testing::ElementsAre(TernaryIs("0bXXXX")))));
  EXPECT_THAT(mg.GetTernary(f->return_value()),
              testing::Optional(LttIs<TernaryVector>(
                  testing::ElementsAre(TernaryIs("0bXXXXX")))));
}

TEST_F(ManualGivenQueryEngineTest, ClearGivensResetsToBase) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue y = fb.Param("y", p->GetBitsType(4));
  fb.Add(fb.ZeroExtend(x, 5), fb.ZeroExtend(y, 5));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  DirectQueryEngine base;
  XLS_ASSERT_OK_AND_ASSIGN(auto x_ternary, StringToTernaryVector("0bX00X"));
  XLS_ASSERT_OK_AND_ASSIGN(auto y_ternary, StringToTernaryVector("0bX00X"));
  base.Override(x.node(), x_ternary).Override(y.node(), y_ternary);

  ManualGivenQueryEngine mg =
      ManualGivenQueryEngine::Of(LazyTernaryQueryEngine());
  XLS_ASSERT_OK(mg.Populate(f).status());
  EXPECT_FALSE(mg.has_givens());

  XLS_ASSERT_OK(mg.SetGivens({x.node(), y.node()}, base));
  EXPECT_TRUE(mg.has_givens());
  EXPECT_THAT(mg.GetTernary(f->return_value()),
              testing::Optional(LttIs<TernaryVector>(
                  testing::ElementsAre(TernaryIs("0bXX0XX")))));

  XLS_ASSERT_OK(mg.ClearGivens());
  EXPECT_FALSE(mg.has_givens());
  EXPECT_THAT(mg.GetTernary(f->return_value()),
              testing::Optional(LttIs<TernaryVector>(
                  testing::ElementsAre(TernaryIs("0bXXXXX")))));
}

TEST_F(ManualGivenQueryEngineTest, PopulateResetsGivens) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue y = fb.Param("y", p->GetBitsType(4));
  fb.Add(fb.ZeroExtend(x, 5), fb.ZeroExtend(y, 5));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  DirectQueryEngine base;
  XLS_ASSERT_OK_AND_ASSIGN(auto x_ternary, StringToTernaryVector("0bX00X"));
  XLS_ASSERT_OK_AND_ASSIGN(auto y_ternary, StringToTernaryVector("0bX00X"));
  base.Override(x.node(), x_ternary).Override(y.node(), y_ternary);

  ManualGivenQueryEngine mg =
      ManualGivenQueryEngine::Of(LazyTernaryQueryEngine());
  XLS_ASSERT_OK(mg.Populate(f).status());
  XLS_ASSERT_OK(mg.SetGivens({x.node(), y.node()}, base));
  EXPECT_TRUE(mg.has_givens());
  EXPECT_THAT(mg.GetTernary(f->return_value()),
              testing::Optional(LttIs<TernaryVector>(
                  testing::ElementsAre(TernaryIs("0bXX0XX")))));

  XLS_ASSERT_OK(mg.Populate(f).status());
  EXPECT_FALSE(mg.has_givens());
  EXPECT_THAT(mg.GetTernary(f->return_value()),
              testing::Optional(LttIs<TernaryVector>(
                  testing::ElementsAre(TernaryIs("0bXXXXX")))));
}

TEST_F(ManualGivenQueryEngineTest, RepeatedSetGivensReplacesPriorGivens) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue y = fb.Param("y", p->GetBitsType(4));
  fb.Add(fb.ZeroExtend(x, 5), fb.ZeroExtend(y, 5));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  DirectQueryEngine base;
  XLS_ASSERT_OK_AND_ASSIGN(auto x_ternary, StringToTernaryVector("0bX00X"));
  XLS_ASSERT_OK_AND_ASSIGN(auto y_ternary, StringToTernaryVector("0bX00X"));
  base.Override(x.node(), x_ternary).Override(y.node(), y_ternary);

  ManualGivenQueryEngine mg =
      ManualGivenQueryEngine::Of(LazyTernaryQueryEngine());
  XLS_ASSERT_OK(mg.Populate(f).status());

  XLS_ASSERT_OK(mg.SetGivens({x.node()}, base));
  EXPECT_THAT(mg.GetTernary(x.node()),
              testing::Optional(LttIs<TernaryVector>(
                  testing::ElementsAre(TernaryIs("0bX00X")))));
  EXPECT_THAT(mg.GetTernary(y.node()),
              testing::Optional(LttIs<TernaryVector>(
                  testing::ElementsAre(TernaryIs("0bXXXX")))));

  XLS_ASSERT_OK(mg.SetGivens({y.node()}, base));
  EXPECT_THAT(mg.GetTernary(x.node()),
              testing::Optional(LttIs<TernaryVector>(
                  testing::ElementsAre(TernaryIs("0bXXXX")))));
  EXPECT_THAT(mg.GetTernary(y.node()),
              testing::Optional(LttIs<TernaryVector>(
                  testing::ElementsAre(TernaryIs("0bX00X")))));
}

TEST_F(ManualGivenQueryEngineTest, IntervalsWithPartialInfoQueryEngine) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue y = fb.Param("y", p->GetBitsType(4));
  fb.Add(fb.ZeroExtend(x, 8), fb.ZeroExtend(y, 8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  DirectQueryEngine base;
  XLS_ASSERT_OK_AND_ASSIGN(auto x_ternary, StringToTernaryVector("0b00XX"));
  XLS_ASSERT_OK_AND_ASSIGN(auto y_ternary, StringToTernaryVector("0b0100"));
  base.Override(x.node(), x_ternary).Override(y.node(), y_ternary);

  ManualGivenQueryEngine mg =
      ManualGivenQueryEngine::Of(PartialInfoQueryEngine());
  XLS_ASSERT_OK(mg.Populate(f).status());
  EXPECT_EQ(IntervalSetTreeToString(mg.GetIntervals(f->return_value())),
            "[[0, 30]]");

  XLS_ASSERT_OK(mg.SetGivens({x.node(), y.node()}, base));
  EXPECT_EQ(IntervalSetTreeToString(mg.GetIntervals(f->return_value())),
            "[[4, 7]]");

  XLS_ASSERT_OK(mg.ClearGivens());
  EXPECT_EQ(IntervalSetTreeToString(mg.GetIntervals(f->return_value())),
            "[[0, 30]]");
}

}  // namespace
}  // namespace xls
