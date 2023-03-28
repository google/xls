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

#include "xls/passes/union_query_engine.h"

#include <cstdint>
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

class FakeQueryEngine : public QueryEngine {
 public:
  FakeQueryEngine() = default;

  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override {
    return ReachedFixpoint::Unchanged;
  }

  bool IsTracked(Node* node) const override { return tracked_.contains(node); }

  LeafTypeTree<TernaryVector> GetTernary(Node* node) const override {
    XLS_CHECK(node->GetType()->IsBits());
    TernaryVector ternary = ternary_ops::FromKnownBits(
        known_bits_.at(node), known_bit_values_.at(node));
    LeafTypeTree<TernaryVector> result(node->GetType());
    result.Set({}, ternary);
    return result;
  }

  LeafTypeTree<IntervalSet> GetIntervals(Node* node) const override {
    return intervals_.at(node);
  }

  bool AtMostOneTrue(absl::Span<TreeBitLocation const> bits) const override {
    return at_most_one_true_.contains(
        std::vector<TreeBitLocation>(bits.begin(), bits.end()));
  }

  bool AtLeastOneTrue(absl::Span<TreeBitLocation const> bits) const override {
    return at_least_one_true_.contains(
        std::vector<TreeBitLocation>(bits.begin(), bits.end()));
  }

  bool Implies(const TreeBitLocation& a,
               const TreeBitLocation& b) const override {
    return implications_.contains({a, b});
  }

  std::optional<Bits> ImpliedNodeValue(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override {
    std::vector<std::pair<TreeBitLocation, bool>> vec(
        predicate_bit_values.begin(), predicate_bit_values.end());
    if (implied_node_values_.contains({vec, node})) {
      return implied_node_values_.at({vec, node});
    }
    return std::nullopt;
  }

  bool KnownEquals(const TreeBitLocation& a,
                   const TreeBitLocation& b) const override {
    if (equality_states_.contains({a, b})) {
      if (equality_states_.at({a, b}) == EqualityState::Equal) {
        return true;
      }
    }
    if (equality_states_.contains({b, a})) {
      if (equality_states_.at({b, a}) == EqualityState::Equal) {
        return true;
      }
    }
    return false;
  }

  bool KnownNotEquals(const TreeBitLocation& a,
                      const TreeBitLocation& b) const override {
    if (equality_states_.contains({a, b})) {
      if (equality_states_.at({a, b}) == EqualityState::NotEqual) {
        return true;
      }
    }
    if (equality_states_.contains({b, a})) {
      if (equality_states_.at({b, a}) == EqualityState::NotEqual) {
        return true;
      }
    }
    return false;
  }

  void AddTracked(Node* node) {
    if (!tracked_.contains(node)) {
      tracked_.insert(node);
      known_bits_[node] = Bits(node->GetType()->GetFlatBitCount());
      known_bit_values_[node] = Bits(node->GetType()->GetFlatBitCount());
    }
  }

  void AddIntervals(Node* node, const LeafTypeTree<IntervalSet>& intervals) {
    AddTracked(node);
    intervals_[node] = intervals;
  }

  void AddKnownBit(const TreeBitLocation& location, bool value) {
    Node* node = location.node();
    AddTracked(node);
    XLS_CHECK(node->GetType()->IsBits());
    int64_t index = location.bit_index();
    int64_t width = node->GetType()->GetFlatBitCount();
    XLS_CHECK_LT(index, width);
    XLS_CHECK_EQ(known_bits_[node].bit_count(), width);
    XLS_CHECK_EQ(known_bit_values_[node].bit_count(), width);
    known_bits_[node] = known_bits_[node].UpdateWithSet(index, true);
    known_bit_values_[node] =
        known_bit_values_[node].UpdateWithSet(index, value);
  }

  void AddAtMostOneTrue(absl::Span<const TreeBitLocation> span) {
    for (const TreeBitLocation& location : span) {
      AddTracked(location.node());
    }
    at_most_one_true_.insert(
        std::vector<TreeBitLocation>(span.begin(), span.end()));
  }

  void AddAtLeastOneTrue(absl::Span<const TreeBitLocation> span) {
    for (const TreeBitLocation& location : span) {
      AddTracked(location.node());
    }
    at_least_one_true_.insert(
        std::vector<TreeBitLocation>(span.begin(), span.end()));
  }

  void AddImplication(const TreeBitLocation& x, const TreeBitLocation& y) {
    AddTracked(x.node());
    AddTracked(y.node());
    implications_.insert({x, y});
  }

  void AddImpliedNodeValue(
      absl::Span<const std::pair<TreeBitLocation, bool>> span, Node* node,
      const Bits& bits) {
    AddTracked(node);
    for (const auto& [location, value] : span) {
      AddTracked(location.node());
    }
    std::vector<std::pair<TreeBitLocation, bool>> vec(span.begin(), span.end());
    implied_node_values_[{vec, node}] = bits;
  }

  void AddEquality(const TreeBitLocation& x, const TreeBitLocation& y) {
    AddTracked(x.node());
    AddTracked(y.node());
    equality_states_[{x, y}] = EqualityState::Equal;
    equality_states_[{y, x}] = EqualityState::Equal;
  }

  void AddInequality(const TreeBitLocation& x, const TreeBitLocation& y) {
    AddTracked(x.node());
    AddTracked(y.node());
    equality_states_[{x, y}] = EqualityState::NotEqual;
    equality_states_[{y, x}] = EqualityState::NotEqual;
  }

 private:
  enum class EqualityState {
    Equal,
    NotEqual,
  };

  absl::flat_hash_set<Node*> tracked_;
  absl::flat_hash_map<Node*, Bits> known_bits_;
  absl::flat_hash_map<Node*, Bits> known_bit_values_;
  absl::flat_hash_map<Node*, LeafTypeTree<IntervalSet>> intervals_;
  absl::flat_hash_set<std::vector<TreeBitLocation>> at_most_one_true_;
  absl::flat_hash_set<std::vector<TreeBitLocation>> at_least_one_true_;
  absl::flat_hash_set<std::pair<TreeBitLocation, TreeBitLocation>>
      implications_;
  absl::flat_hash_map<
      std::pair<std::vector<std::pair<TreeBitLocation, bool>>, Node*>, Bits>
      implied_node_values_;
  absl::flat_hash_map<std::pair<TreeBitLocation, TreeBitLocation>,
                      EqualityState>
      equality_states_;
};

class UnionQueryEngineTest : public IrTestBase {};

TEST_F(UnionQueryEngineTest, Simple) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue param = fb.Param("param", fb.package()->GetBitsType(8));
  XLS_ASSERT_OK(fb.Build());

  Node* node = param.node();

  FakeQueryEngine query_engine_a;
  query_engine_a.AddKnownBit(TreeBitLocation(node, 4), true);
  query_engine_a.AddEquality(TreeBitLocation(node, 0),
                             TreeBitLocation(node, 1));
  query_engine_a.AddImplication(TreeBitLocation(node, 3),
                                TreeBitLocation(node, 7));
  FakeQueryEngine query_engine_b;
  query_engine_b.AddInequality(TreeBitLocation(node, 2),
                               TreeBitLocation(node, 3));
  query_engine_b.AddAtMostOneTrue(
      {TreeBitLocation(node, 2), TreeBitLocation(node, 3)});
  query_engine_b.AddAtLeastOneTrue(
      {TreeBitLocation(node, 2), TreeBitLocation(node, 3)});
  query_engine_b.AddImpliedNodeValue(
      {{TreeBitLocation(node, 7), true}, {TreeBitLocation(node, 3), true}},
      node, UBits(0b10011000, 8));
  query_engine_b.AddImplication(TreeBitLocation(node, 7),
                                TreeBitLocation(node, 3));
  std::vector<std::unique_ptr<QueryEngine>> engines;
  engines.push_back(std::make_unique<FakeQueryEngine>(query_engine_a));
  engines.push_back(std::make_unique<FakeQueryEngine>(query_engine_b));
  UnionQueryEngine union_query_engine(std::move(engines));
  // No need to Populate, since FakeQueryEngine doesn't use that interface

  EXPECT_FALSE(union_query_engine.IsTracked(nullptr));
  EXPECT_TRUE(union_query_engine.IsTracked(node));
  EXPECT_EQ(ToString(union_query_engine.GetTernary(node).Get({})),
            "0bXXX1_XXXX");
  // Query the same ones again to test the caching of known bits
  EXPECT_EQ(ToString(union_query_engine.GetTernary(node).Get({})),
            "0bXXX1_XXXX");
  EXPECT_TRUE(union_query_engine.AtMostOneTrue(
      {TreeBitLocation(node, 2), TreeBitLocation(node, 3)}));
  EXPECT_FALSE(union_query_engine.AtMostOneTrue(
      {TreeBitLocation(node, 5), TreeBitLocation(node, 6)}));
  EXPECT_TRUE(union_query_engine.AtLeastOneTrue(
      {TreeBitLocation(node, 2), TreeBitLocation(node, 3)}));
  EXPECT_FALSE(union_query_engine.AtLeastOneTrue(
      {TreeBitLocation(node, 5), TreeBitLocation(node, 6)}));
  EXPECT_TRUE(union_query_engine.KnownEquals(TreeBitLocation(node, 0),
                                             TreeBitLocation(node, 1)));
  EXPECT_FALSE(union_query_engine.KnownEquals(TreeBitLocation(node, 5),
                                              TreeBitLocation(node, 6)));
  EXPECT_TRUE(union_query_engine.KnownNotEquals(TreeBitLocation(node, 2),
                                                TreeBitLocation(node, 3)));
  EXPECT_FALSE(union_query_engine.KnownNotEquals(TreeBitLocation(node, 5),
                                                 TreeBitLocation(node, 6)));
  EXPECT_TRUE(union_query_engine.Implies(TreeBitLocation(node, 3),
                                         TreeBitLocation(node, 7)));
  EXPECT_TRUE(union_query_engine.Implies(TreeBitLocation(node, 7),
                                         TreeBitLocation(node, 3)));
  EXPECT_FALSE(union_query_engine.Implies(TreeBitLocation(node, 5),
                                          TreeBitLocation(node, 6)));
  EXPECT_EQ(
      union_query_engine.ImpliedNodeValue(
          {{TreeBitLocation(node, 7), true}, {TreeBitLocation(node, 3), true}},
          node),
      UBits(0b10011000, 8));
  EXPECT_FALSE(union_query_engine.ImpliedNodeValue(
      {{TreeBitLocation(node, 5), true}, {TreeBitLocation(node, 6), true}},
      node));
}

TEST_F(UnionQueryEngineTest, Intervals) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", fb.package()->GetBitsType(8));
  BValue y = fb.Param("y", fb.package()->GetBitsType(8));
  BValue tuple = fb.Tuple({x, y});
  XLS_ASSERT_OK(fb.Build());

  FakeQueryEngine query_engine_a;
  IntervalSet x_a(8);
  x_a.AddInterval(Interval(UBits(20, 8), UBits(40, 8)));
  x_a.Normalize();
  query_engine_a.AddIntervals(
      tuple.node(), LeafTypeTree<IntervalSet>(tuple.node()->GetType(),
                                              {x_a, IntervalSet::Maximal(8)}));
  FakeQueryEngine query_engine_b;
  IntervalSet y_b(8);
  y_b.AddInterval(Interval(UBits(200, 8), UBits(240, 8)));
  y_b.Normalize();
  query_engine_b.AddIntervals(
      tuple.node(), LeafTypeTree<IntervalSet>(tuple.node()->GetType(),
                                              {IntervalSet::Maximal(8), y_b}));
  std::vector<std::unique_ptr<QueryEngine>> engines;
  engines.push_back(std::make_unique<FakeQueryEngine>(query_engine_a));
  engines.push_back(std::make_unique<FakeQueryEngine>(query_engine_b));
  UnionQueryEngine union_query_engine(std::move(engines));
  // No need to Populate, since FakeQueryEngine doesn't use that interface
  LeafTypeTree<IntervalSet> tuple_intervals =
      union_query_engine.GetIntervals(tuple.node());
  EXPECT_EQ(tuple_intervals.elements()[0], x_a);
  EXPECT_EQ(tuple_intervals.elements()[1], y_b);
}

}  // namespace
}  // namespace xls
