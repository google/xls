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
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"

namespace xls {
namespace {

class FakeQueryEngine : public QueryEngine {
 public:
  FakeQueryEngine() = default;

  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override {
    return ReachedFixpoint::Unchanged;
  }

  bool IsTracked(Node* node) const override { return tracked_.contains(node); }

  std::optional<SharedLeafTypeTree<TernaryVector>> GetTernary(
      Node* node) const override {
    if (auto it = ternaries_.find(node); it != ternaries_.end()) {
      return it->second.AsView().AsShared();
    }
    return std::nullopt;
  }

  LeafTypeTree<IntervalSet> GetIntervals(Node* node) const override {
    if (auto it = intervals_.find(node); it != intervals_.end()) {
      return it->second;
    }
    return QueryEngine::GetIntervals(node);
  }

  std::unique_ptr<QueryEngine> SpecializeGiven(
      const absl::btree_map<Node*, ValueKnowledge, Node::NodeIdLessThan>&
          givens) const override {
    auto copy = std::make_unique<FakeQueryEngine>(*this);
    for (const auto& [node, knowledge] : givens) {
      if (knowledge.ternary.has_value()) {
        if (auto it = copy->ternaries_.find(node);
            it != copy->ternaries_.end()) {
          LeafTypeTree<TernaryVector> merged = it->second;
          leaf_type_tree::SimpleUpdateFrom<TernaryVector, TernaryVector>(
              merged.AsMutableView(), knowledge.ternary->AsView(),
              [](TernaryVector& lhs, const TernaryVector& rhs) {
                CHECK_OK(ternary_ops::UpdateWithUnion(lhs, rhs));
              });
          copy->AddTernary(node, merged);
        } else {
          copy->AddTernary(node, *knowledge.ternary);
        }
      }
      if (knowledge.intervals.has_value()) {
        copy->AddIntervals(node, *knowledge.intervals);
      }
    }
    return copy;
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

  std::optional<TernaryVector> ImpliedNodeTernary(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override {
    std::vector<std::pair<TreeBitLocation, bool>> vec(
        predicate_bit_values.begin(), predicate_bit_values.end());
    if (implied_node_ternaries_.contains({vec, node})) {
      return implied_node_ternaries_.at({vec, node});
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

  void AddTernary(Node* node, const LeafTypeTree<TernaryVector>& ternary) {
    AddTracked(node);
    ternaries_[node] = ternary;
    if (node->GetType()->IsBits()) {
      known_bits_[node] = ternary_ops::ToKnownBits(ternary.Get({}));
      known_bit_values_[node] = ternary_ops::ToKnownBitsValues(ternary.Get({}));
    }
  }

  void AddIntervals(Node* node, const LeafTypeTree<IntervalSet>& intervals) {
    AddTracked(node);
    intervals_[node] = intervals;
  }

  Bits MaxUnsignedValue(Node* n) const override {
    CHECK(n->GetType()->IsBits()) << n;
    if (intervals_.contains(n)) {
      return intervals_.at(n).Get({}).ConvexHull()->UpperBound();
    } else {
      return Bits::AllOnes(n->BitCountOrDie());
    }
  }
  Bits MinUnsignedValue(Node* n) const override {
    CHECK(n->GetType()->IsBits()) << n;
    if (intervals_.contains(n)) {
      return intervals_.at(n).Get({}).ConvexHull()->LowerBound();
    } else {
      return Bits(n->BitCountOrDie());
    }
  }

  void AddKnownBit(const TreeBitLocation& location, bool value) {
    Node* node = location.node();
    AddTracked(node);
    CHECK(node->GetType()->IsBits());
    int64_t index = location.bit_index();
    int64_t width = node->GetType()->GetFlatBitCount();
    CHECK_LT(index, width);
    CHECK_EQ(known_bits_[node].bit_count(), width);
    CHECK_EQ(known_bit_values_[node].bit_count(), width);
    known_bits_[node] = known_bits_[node].UpdateWithSet(index, true);
    known_bit_values_[node] =
        known_bit_values_[node].UpdateWithSet(index, value);
    auto ternary_it = ternaries_.find(node);
    if (ternary_it == ternaries_.end()) {
      bool inserted;
      std::tie(ternary_it, inserted) = ternaries_.insert(
          {node, *LeafTypeTree<TernaryVector>::CreateFromFunction(
                     node->GetType(), [](Type* leaf_type) {
                       return TernaryVector(leaf_type->GetFlatBitCount(),
                                            TernaryValue::kUnknown);
                     })});
      CHECK(inserted);
    }
    ternary_it->second.Get({})[index] =
        value ? TernaryValue::kKnownOne : TernaryValue::kKnownZero;
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
  absl::flat_hash_map<Node*, LeafTypeTree<TernaryVector>> ternaries_;
  absl::flat_hash_map<Node*, LeafTypeTree<IntervalSet>> intervals_;
  absl::flat_hash_set<std::vector<TreeBitLocation>> at_most_one_true_;
  absl::flat_hash_set<std::vector<TreeBitLocation>> at_least_one_true_;
  absl::flat_hash_set<std::pair<TreeBitLocation, TreeBitLocation>>
      implications_;
  absl::flat_hash_map<
      std::pair<std::vector<std::pair<TreeBitLocation, bool>>, Node*>, Bits>
      implied_node_values_;
  absl::flat_hash_map<
      std::pair<std::vector<std::pair<TreeBitLocation, bool>>, Node*>,
      TernaryVector>
      implied_node_ternaries_;
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
  EXPECT_EQ(ToString(union_query_engine.GetTernary(node)->Get({})),
            "0bXXX1_XXXX");
  // Query the same ones again to test the caching of known bits
  EXPECT_EQ(ToString(union_query_engine.GetTernary(node)->Get({})),
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
  IntervalSet x_a = IntervalSet::Of({
      Interval(UBits(20, 8), UBits(40, 8)),
  });
  query_engine_a.AddIntervals(
      tuple.node(), LeafTypeTree<IntervalSet>(tuple.node()->GetType(),
                                              {x_a, IntervalSet::Maximal(8)}));
  FakeQueryEngine query_engine_b;
  IntervalSet y_b = IntervalSet::Of({Interval(UBits(200, 8), UBits(240, 8))});
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

TEST_F(UnionQueryEngineTest, MaxMinUnsignedValue) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue x = fb.Param("x", fb.package()->GetBitsType(8));
  BValue y = fb.Param("y", fb.package()->GetBitsType(8));
  fb.Tuple({x, y});
  XLS_ASSERT_OK(fb.Build());

  FakeQueryEngine query_engine_a;
  IntervalSet x_a = IntervalSet::Of({
      Interval(UBits(20, 8), UBits(40, 8)),
  });
  query_engine_a.AddIntervals(
      x.node(), LeafTypeTree<IntervalSet>(x.node()->GetType(), {x_a}));
  FakeQueryEngine query_engine_b;
  IntervalSet y_b = IntervalSet::Of({Interval(UBits(200, 8), UBits(240, 8))});
  query_engine_b.AddIntervals(
      y.node(), LeafTypeTree<IntervalSet>(y.node()->GetType(), {y_b}));
  std::vector<std::unique_ptr<QueryEngine>> engines;
  engines.push_back(std::make_unique<FakeQueryEngine>(query_engine_a));
  engines.push_back(std::make_unique<FakeQueryEngine>(query_engine_b));
  UnionQueryEngine union_query_engine(std::move(engines));
  // No need to Populate, since FakeQueryEngine doesn't use that interface
  EXPECT_EQ(union_query_engine.MinUnsignedValue(x.node()), UBits(20, 8));
  EXPECT_EQ(union_query_engine.MinUnsignedValue(y.node()), UBits(200, 8));
  EXPECT_EQ(union_query_engine.MaxUnsignedValue(x.node()), UBits(40, 8));
  EXPECT_EQ(union_query_engine.MaxUnsignedValue(y.node()), UBits(240, 8));
}

TEST_F(UnionQueryEngineTest, TokenValueUnknown) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue tok = fb.Param("tok", p->GetTokenType());
  XLS_ASSERT_OK(fb.Build());

  FakeQueryEngine query_engine_a;
  query_engine_a.AddTernary(
      tok.node(), *LeafTypeTree<TernaryVector>::CreateFromFunction(
                      tok.GetType(), [](Type* leaf_type) {
                        return TernaryVector(leaf_type->GetFlatBitCount(),
                                             TernaryValue::kUnknown);
                      }));
  FakeQueryEngine query_engine_b;
  query_engine_b.AddTracked(tok.node());
  std::vector<std::unique_ptr<QueryEngine>> engines;
  engines.push_back(std::make_unique<FakeQueryEngine>(query_engine_a));
  engines.push_back(std::make_unique<FakeQueryEngine>(query_engine_b));
  UnionQueryEngine union_query_engine(std::move(engines));

  EXPECT_EQ(union_query_engine.KnownValue(tok.node()), std::nullopt);
}

TEST_F(UnionQueryEngineTest, OfGeneric) {
  auto p = CreatePackage();
  auto tern =
      [&](std::string_view sv) -> absl::StatusOr<LeafTypeTree<TernaryVector>> {
    XLS_ASSIGN_OR_RETURN(TernaryVector tv, StringToTernaryVector(sv));
    return LeafTypeTree<TernaryVector>::CreateSingleElementTree(
        p->GetBitsType(tv.size()), tv);
  };
  FunctionBuilder fb(TestName(), p.get());
  BValue res = fb.Param("res", p->GetBitsType(8));
  XLS_ASSERT_OK(fb.Build());
  FakeQueryEngine a;
  XLS_ASSERT_OK_AND_ASSIGN(auto tt, tern("0b1XXXXXXX"));
  a.AddTernary(res.node(), tt);
  auto b = std::make_unique<FakeQueryEngine>();
  XLS_ASSERT_OK_AND_ASSIGN(tt, tern("0bX1XXXXXX"));
  b->AddTernary(res.node(), tt);
  FakeQueryEngine c;
  XLS_ASSERT_OK_AND_ASSIGN(tt, tern("0bXX1XXXXX"));
  c.AddTernary(res.node(), tt);
  FakeQueryEngine d;
  XLS_ASSERT_OK_AND_ASSIGN(tt, tern("0bXXX1XXXX"));
  d.AddTernary(res.node(), tt);

  UnionQueryEngine uqe =
      UnionQueryEngine::Of(&a, std::move(b), c, std::move(d));

  EXPECT_THAT(b.get(), testing::IsNull());
  XLS_ASSERT_OK_AND_ASSIGN(auto exp, tern("0b1111XXXX"));
  EXPECT_THAT(uqe.GetTernary(res.node()), testing::Optional(exp));
  // a is referenced by pointer so this changes the result.
  XLS_ASSERT_OK_AND_ASSIGN(tt, tern("0b1XXXXXX1"));
  a.AddTernary(res.node(), tt);
  // C is copied so this has no effect.
  XLS_ASSERT_OK_AND_ASSIGN(tt, tern("0bX1XXXX1X"));
  c.AddTernary(res.node(), tt);
  XLS_ASSERT_OK_AND_ASSIGN(exp, tern("0b1111XXX1"));
  EXPECT_THAT(uqe.GetTernary(res.node()), testing::Optional(exp))
      << uqe.GetTernary(res.node())->ToString();
}

TEST_F(UnionQueryEngineTest, SpecializeOnNodes) {
  auto p = CreatePackage();
  auto tern =
      [&](std::string_view sv) -> absl::StatusOr<LeafTypeTree<TernaryVector>> {
    XLS_ASSIGN_OR_RETURN(TernaryVector tv, StringToTernaryVector(sv));
    return LeafTypeTree<TernaryVector>::CreateSingleElementTree(
        p->GetBitsType(tv.size()), tv);
  };
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  XLS_ASSERT_OK(fb.Build());

  FakeQueryEngine a;
  XLS_ASSERT_OK_AND_ASSIGN(auto tt, tern("0b1XXXXXXX"));
  a.AddTernary(x.node(), tt);
  XLS_ASSERT_OK_AND_ASSIGN(tt, tern("0b0XXXXXXX"));
  a.AddTernary(y.node(), tt);

  FakeQueryEngine b;
  XLS_ASSERT_OK_AND_ASSIGN(tt, tern("0bX1XXXXXX"));
  b.AddTernary(x.node(), tt);
  XLS_ASSERT_OK_AND_ASSIGN(tt, tern("0bX0XXXXXX"));
  b.AddTernary(y.node(), tt);

  UnionQueryEngine uqe = UnionQueryEngine::Of(a, b);

  FakeQueryEngine info_source;
  XLS_ASSERT_OK_AND_ASSIGN(tt, tern("0bXX1XXXXX"));
  info_source.AddTernary(x.node(), tt);
  XLS_ASSERT_OK_AND_ASSIGN(tt, tern("0bXX0XXXXX"));
  info_source.AddTernary(y.node(), tt);

  // Specialize only on x. Both child engines receive the given for x,
  // while y keeps the unioned knowledge from a and b.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<QueryEngine> specialized,
                           uqe.SpecializeOnNodes({x.node()}, info_source));
  XLS_ASSERT_OK_AND_ASSIGN(auto exp_x, tern("0b111XXXXX"));
  EXPECT_THAT(specialized->GetTernary(x.node()), testing::Optional(exp_x));
  XLS_ASSERT_OK_AND_ASSIGN(auto exp_y, tern("0b00XXXXXX"));
  EXPECT_THAT(specialized->GetTernary(y.node()), testing::Optional(exp_y));

  // Original union query engine is unmodified.
  XLS_ASSERT_OK_AND_ASSIGN(auto orig_x, tern("0b11XXXXXX"));
  EXPECT_THAT(uqe.GetTernary(x.node()), testing::Optional(orig_x));
}

// A FakeQueryEngine that records calls to IsFullyKnown, GetTernary,
// IsAllZeros, and IsAllOnes, while answering IsAllZeros/IsAllOnes without
// going through GetTernary (matching engines like BddQueryEngine,
// BitCountQueryEngine, and PartialInfoQueryEngine).
class CountingQueryEngine : public FakeQueryEngine {
 public:
  bool IsFullyKnown(Node* n) const override {
    ++is_fully_known_calls_;
    return FakeQueryEngine::IsFullyKnown(n);
  }

  std::optional<SharedLeafTypeTree<TernaryVector>> GetTernary(
      Node* node) const override {
    ++get_ternary_calls_;
    return FakeQueryEngine::GetTernary(node);
  }

  bool IsAllZeros(Node* n) const override {
    ++is_all_zeros_calls_;
    if (!IsTracked(n) || TypeHasToken(n->GetType())) {
      return false;
    }
    std::optional<SharedLeafTypeTree<TernaryVector>> ternary =
        FakeQueryEngine::GetTernary(n);
    return ternary.has_value() &&
           absl::c_all_of(ternary->elements(), [](const TernaryVector& v) {
             return ternary_ops::IsKnownZero(v);
           });
  }

  bool IsAllOnes(Node* n) const override {
    ++is_all_ones_calls_;
    if (!IsTracked(n) || TypeHasToken(n->GetType())) {
      return false;
    }
    std::optional<SharedLeafTypeTree<TernaryVector>> ternary =
        FakeQueryEngine::GetTernary(n);
    return ternary.has_value() &&
           absl::c_all_of(ternary->elements(), [](const TernaryVector& v) {
             return ternary_ops::IsKnownOne(v);
           });
  }

  int64_t is_fully_known_calls() const { return is_fully_known_calls_; }
  int64_t get_ternary_calls() const { return get_ternary_calls_; }
  int64_t is_all_zeros_calls() const { return is_all_zeros_calls_; }
  int64_t is_all_ones_calls() const { return is_all_ones_calls_; }

  void ResetCounts() {
    is_fully_known_calls_ = 0;
    get_ternary_calls_ = 0;
    is_all_zeros_calls_ = 0;
    is_all_ones_calls_ = 0;
  }

 private:
  mutable int64_t is_fully_known_calls_ = 0;
  mutable int64_t get_ternary_calls_ = 0;
  mutable int64_t is_all_zeros_calls_ = 0;
  mutable int64_t is_all_ones_calls_ = 0;
};

TEST_F(UnionQueryEngineTest, IsAllZerosAndOnesTokenTyped) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue tok_param = fb.Param("tok", p->GetTokenType());
  BValue tok_lit = fb.Literal(Value::Token());
  BValue tok_tuple_zero =
      fb.Literal(Value::Tuple({Value(UBits(0, 4)), Value::Token()}));
  BValue tok_tuple_ones =
      fb.Literal(Value::Tuple({Value(Bits::AllOnes(4)), Value::Token()}));
  BValue tok_param_tuple = fb.Tuple({fb.Literal(UBits(0, 4)), tok_param});
  fb.Tuple(
      {tok_param, tok_lit, tok_tuple_zero, tok_tuple_ones, tok_param_tuple});
  XLS_ASSERT_OK(fb.Build());

  FakeQueryEngine fake;
  fake.AddTracked(tok_param.node());
  fake.AddTracked(tok_lit.node());
  fake.AddTracked(tok_tuple_zero.node());
  fake.AddTracked(tok_tuple_ones.node());
  fake.AddTracked(tok_param_tuple.node());

  UnionQueryEngine single_stateless =
      UnionQueryEngine::Of(StatelessQueryEngine());
  UnionQueryEngine union_qe =
      UnionQueryEngine::Of(StatelessQueryEngine(), fake);

  for (Node* n : {tok_param.node(), tok_lit.node(), tok_tuple_zero.node(),
                  tok_tuple_ones.node(), tok_param_tuple.node()}) {
    EXPECT_FALSE(single_stateless.IsAllZeros(n));
    EXPECT_FALSE(single_stateless.IsAllOnes(n));
    EXPECT_FALSE(union_qe.IsAllZeros(n));
    EXPECT_FALSE(union_qe.IsAllOnes(n));
  }
}

TEST_F(UnionQueryEngineTest, IsAllZerosAndOnesBitsUnionedAcrossEngines) {
  auto p = CreatePackage();
  auto make_bits_ternary =
      [&](std::string_view sv) -> absl::StatusOr<LeafTypeTree<TernaryVector>> {
    XLS_ASSIGN_OR_RETURN(TernaryVector tv, StringToTernaryVector(sv));
    return LeafTypeTree<TernaryVector>::CreateSingleElementTree(
        p->GetBitsType(tv.size()), tv);
  };

  FunctionBuilder fb(TestName(), p.get());
  BValue all_zero = fb.Param("all_zero", p->GetBitsType(4));
  BValue all_ones = fb.Param("all_ones", p->GetBitsType(4));
  BValue partial = fb.Param("partial", p->GetBitsType(4));
  BValue mixed = fb.Param("mixed", p->GetBitsType(4));
  BValue three_way = fb.Param("three_way", p->GetBitsType(6));
  fb.Tuple({all_zero, all_ones, partial, mixed, three_way});
  XLS_ASSERT_OK(fb.Build());

  FakeQueryEngine engine_a;
  FakeQueryEngine engine_b;
  FakeQueryEngine engine_c;

  XLS_ASSERT_OK_AND_ASSIGN(auto t, make_bits_ternary("0b00XX"));
  engine_a.AddTernary(all_zero.node(), t);
  XLS_ASSERT_OK_AND_ASSIGN(t, make_bits_ternary("0bXX00"));
  engine_b.AddTernary(all_zero.node(), t);

  XLS_ASSERT_OK_AND_ASSIGN(t, make_bits_ternary("0b11XX"));
  engine_a.AddTernary(all_ones.node(), t);
  XLS_ASSERT_OK_AND_ASSIGN(t, make_bits_ternary("0bXX11"));
  engine_b.AddTernary(all_ones.node(), t);

  XLS_ASSERT_OK_AND_ASSIGN(t, make_bits_ternary("0b00XX"));
  engine_a.AddTernary(partial.node(), t);
  XLS_ASSERT_OK_AND_ASSIGN(t, make_bits_ternary("0bXXX0"));
  engine_b.AddTernary(partial.node(), t);

  XLS_ASSERT_OK_AND_ASSIGN(t, make_bits_ternary("0b00XX"));
  engine_a.AddTernary(mixed.node(), t);
  XLS_ASSERT_OK_AND_ASSIGN(t, make_bits_ternary("0bXX10"));
  engine_b.AddTernary(mixed.node(), t);

  XLS_ASSERT_OK_AND_ASSIGN(t, make_bits_ternary("0b00XXXX"));
  engine_a.AddTernary(three_way.node(), t);
  XLS_ASSERT_OK_AND_ASSIGN(t, make_bits_ternary("0bXX00XX"));
  engine_b.AddTernary(three_way.node(), t);
  XLS_ASSERT_OK_AND_ASSIGN(t, make_bits_ternary("0bXXXX00"));
  engine_c.AddTernary(three_way.node(), t);

  UnionQueryEngine uqe = UnionQueryEngine::Of(&engine_a, &engine_b, &engine_c);

  EXPECT_TRUE(uqe.IsAllZeros(all_zero.node()));
  EXPECT_FALSE(uqe.IsAllOnes(all_zero.node()));

  EXPECT_TRUE(uqe.IsAllOnes(all_ones.node()));
  EXPECT_FALSE(uqe.IsAllZeros(all_ones.node()));

  EXPECT_FALSE(uqe.IsAllZeros(partial.node()));
  EXPECT_FALSE(uqe.IsAllOnes(partial.node()));

  EXPECT_FALSE(uqe.IsAllZeros(mixed.node()));
  EXPECT_FALSE(uqe.IsAllOnes(mixed.node()));

  EXPECT_TRUE(uqe.IsAllZeros(three_way.node()));
  EXPECT_FALSE(uqe.IsAllOnes(three_way.node()));
}

TEST_F(UnionQueryEngineTest, IsAllZerosAndOnesTuplesAndArrays) {
  auto p = CreatePackage();
  Type* u4 = p->GetBitsType(4);
  Type* tup_ty = p->GetTupleType({u4, u4});
  Type* arr_ty = p->GetArrayType(2, u4);

  FunctionBuilder fb(TestName(), p.get());
  BValue tup_one_engine = fb.Param("tup_one_engine", tup_ty);
  BValue tup_split_zero = fb.Param("tup_split_zero", tup_ty);
  BValue tup_split_ones = fb.Param("tup_split_ones", tup_ty);
  BValue tup_mixed = fb.Param("tup_mixed", tup_ty);
  BValue arr_split_zero = fb.Param("arr_split_zero", arr_ty);
  BValue arr_split_ones = fb.Param("arr_split_ones", arr_ty);
  BValue arr_partial = fb.Param("arr_partial", arr_ty);
  BValue zero_bits = fb.Param("zero_bits", p->GetBitsType(0));
  BValue empty_tup = fb.Param("empty_tup", p->GetTupleType({}));
  BValue untracked_zero_bits =
      fb.Param("untracked_zero_bits", p->GetBitsType(0));
  fb.Tuple({tup_one_engine, tup_split_zero, tup_split_ones, tup_mixed,
            arr_split_zero, arr_split_ones, arr_partial, zero_bits, empty_tup,
            untracked_zero_bits});
  XLS_ASSERT_OK(fb.Build());

  auto tv = [](std::string_view sv) {
    absl::StatusOr<TernaryVector> result = StringToTernaryVector(sv);
    CHECK_OK(result);
    return *std::move(result);
  };

  FakeQueryEngine engine_a;
  FakeQueryEngine engine_b;

  engine_b.AddTernary(
      tup_one_engine.node(),
      LeafTypeTree<TernaryVector>(tup_ty, {tv("0b0000"), tv("0b0000")}));

  // Tuple split both across elements and within each element.
  engine_a.AddTernary(
      tup_split_zero.node(),
      LeafTypeTree<TernaryVector>(tup_ty, {tv("0b00XX"), tv("0bXXXX")}));
  engine_b.AddTernary(
      tup_split_zero.node(),
      LeafTypeTree<TernaryVector>(tup_ty, {tv("0bXX00"), tv("0b0000")}));

  engine_a.AddTernary(
      tup_split_ones.node(),
      LeafTypeTree<TernaryVector>(tup_ty, {tv("0b1111"), tv("0b11XX")}));
  engine_b.AddTernary(
      tup_split_ones.node(),
      LeafTypeTree<TernaryVector>(tup_ty, {tv("0bXXXX"), tv("0bXX11")}));

  engine_a.AddTernary(
      tup_mixed.node(),
      LeafTypeTree<TernaryVector>(tup_ty, {tv("0b0000"), tv("0bXXXX")}));
  engine_b.AddTernary(
      tup_mixed.node(),
      LeafTypeTree<TernaryVector>(tup_ty, {tv("0bXXXX"), tv("0b1111")}));

  // Array split across two engines.
  engine_a.AddTernary(
      arr_split_zero.node(),
      LeafTypeTree<TernaryVector>(arr_ty, {tv("0b0000"), tv("0bXX00")}));
  engine_b.AddTernary(
      arr_split_zero.node(),
      LeafTypeTree<TernaryVector>(arr_ty, {tv("0bXXXX"), tv("0b00XX")}));

  engine_a.AddTernary(
      arr_split_ones.node(),
      LeafTypeTree<TernaryVector>(arr_ty, {tv("0b11XX"), tv("0b1111")}));
  engine_b.AddTernary(
      arr_split_ones.node(),
      LeafTypeTree<TernaryVector>(arr_ty, {tv("0bXX11"), tv("0bXXXX")}));

  engine_a.AddTernary(
      arr_partial.node(),
      LeafTypeTree<TernaryVector>(arr_ty, {tv("0b0000"), tv("0bXX00")}));
  engine_b.AddTernary(
      arr_partial.node(),
      LeafTypeTree<TernaryVector>(arr_ty, {tv("0bXXXX"), tv("0bX0XX")}));

  // 0-bit types tracked with ternary vs untracked.
  engine_a.AddTernary(
      zero_bits.node(),
      LeafTypeTree<TernaryVector>(p->GetBitsType(0), TernaryVector{}));
  engine_b.AddTernary(
      empty_tup.node(),
      LeafTypeTree<TernaryVector>(p->GetTupleType({}), TernaryVector{}));

  UnionQueryEngine uqe = UnionQueryEngine::Of(&engine_a, &engine_b);

  EXPECT_TRUE(uqe.IsAllZeros(tup_one_engine.node()));
  EXPECT_FALSE(uqe.IsAllOnes(tup_one_engine.node()));

  EXPECT_TRUE(uqe.IsAllZeros(tup_split_zero.node()));
  EXPECT_FALSE(uqe.IsAllOnes(tup_split_zero.node()));

  EXPECT_TRUE(uqe.IsAllOnes(tup_split_ones.node()));
  EXPECT_FALSE(uqe.IsAllZeros(tup_split_ones.node()));

  EXPECT_FALSE(uqe.IsAllZeros(tup_mixed.node()));
  EXPECT_FALSE(uqe.IsAllOnes(tup_mixed.node()));

  EXPECT_TRUE(uqe.IsAllZeros(arr_split_zero.node()));
  EXPECT_FALSE(uqe.IsAllOnes(arr_split_zero.node()));

  EXPECT_TRUE(uqe.IsAllOnes(arr_split_ones.node()));
  EXPECT_FALSE(uqe.IsAllZeros(arr_split_ones.node()));

  EXPECT_FALSE(uqe.IsAllZeros(arr_partial.node()));
  EXPECT_FALSE(uqe.IsAllOnes(arr_partial.node()));

  EXPECT_TRUE(uqe.IsAllZeros(zero_bits.node()));
  EXPECT_TRUE(uqe.IsAllOnes(zero_bits.node()));
  EXPECT_TRUE(uqe.IsAllZeros(empty_tup.node()));
  EXPECT_TRUE(uqe.IsAllOnes(empty_tup.node()));
  EXPECT_FALSE(uqe.IsAllZeros(untracked_zero_bits.node()));
  EXPECT_FALSE(uqe.IsAllOnes(untracked_zero_bits.node()));
}

TEST_F(UnionQueryEngineTest, IsAllZerosAndOnesQueriesEnginesDirectly) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue lit_zero = fb.Literal(UBits(0, 4));
  BValue lit_ones = fb.Literal(Bits::AllOnes(4));
  BValue lit_mixed = fb.Literal(UBits(0b0101, 4));
  BValue known_zero = fb.Param("known_zero", p->GetBitsType(4));
  BValue known_ones = fb.Param("known_ones", p->GetBitsType(4));
  BValue unknown_wide = fb.Param("unknown_wide", p->GetBitsType(4));
  BValue unknown_bit = fb.Param("unknown_bit", p->GetBitsType(1));
  BValue concat_zero =
      fb.Concat({fb.Literal(UBits(0, 2)), fb.Param("low", p->GetBitsType(2))});
  BValue concat_nonzero =
      fb.Concat({fb.Literal(UBits(1, 2)), fb.Param("low2", p->GetBitsType(2))});
  fb.Tuple({lit_zero, lit_ones, lit_mixed, known_zero, known_ones, unknown_wide,
            unknown_bit, concat_zero, concat_nonzero});
  XLS_ASSERT_OK(fb.Build());

  auto make_bits_ternary = [&](std::string_view sv) {
    absl::StatusOr<TernaryVector> tv = StringToTernaryVector(sv);
    CHECK_OK(tv);
    const int64_t width = tv->size();
    return LeafTypeTree<TernaryVector>::CreateSingleElementTree(
        p->GetBitsType(width), *std::move(tv));
  };

  // Empty union always returns false.
  UnionQueryEngine empty_uqe;
  EXPECT_FALSE(empty_uqe.IsAllZeros(lit_zero.node()));
  EXPECT_FALSE(empty_uqe.IsAllOnes(lit_ones.node()));

  CountingQueryEngine counting;
  counting.AddTernary(known_zero.node(), make_bits_ternary("0b0000"));
  counting.AddTernary(known_ones.node(), make_bits_ternary("0b1111"));
  counting.AddTernary(unknown_wide.node(), make_bits_ternary("0b00XX"));
  counting.AddTernary(unknown_bit.node(), make_bits_ternary("0bX"));
  counting.AddTernary(concat_zero.node(), make_bits_ternary("0bXX00"));

  // Single-engine union queries IsAllZeros/IsAllOnes directly without
  // IsFullyKnown or GetTernary.
  UnionQueryEngine single_uqe = UnionQueryEngine::Of(&counting);
  EXPECT_TRUE(single_uqe.IsAllZeros(known_zero.node()));
  EXPECT_TRUE(single_uqe.IsAllOnes(known_ones.node()));
  EXPECT_FALSE(single_uqe.IsAllZeros(unknown_wide.node()));
  EXPECT_EQ(counting.is_fully_known_calls(), 0);
  EXPECT_EQ(counting.get_ternary_calls(), 0);
  EXPECT_EQ(counting.is_all_zeros_calls(), 2);
  EXPECT_EQ(counting.is_all_ones_calls(), 1);

  // Mix of StatelessQueryEngine + CountingQueryEngine.
  UnionQueryEngine mixed_uqe =
      UnionQueryEngine::Of(StatelessQueryEngine(), &counting);

  // StatelessQueryEngine resolves literals without touching counting.
  counting.ResetCounts();
  EXPECT_TRUE(mixed_uqe.IsAllZeros(lit_zero.node()));
  EXPECT_TRUE(mixed_uqe.IsAllOnes(lit_ones.node()));
  EXPECT_FALSE(mixed_uqe.IsAllZeros(lit_mixed.node()));
  EXPECT_FALSE(mixed_uqe.IsAllOnes(lit_mixed.node()));
  EXPECT_EQ(counting.is_fully_known_calls(), 0);
  EXPECT_EQ(counting.get_ternary_calls(), 0);
  EXPECT_EQ(counting.is_all_zeros_calls(), 0);
  EXPECT_EQ(counting.is_all_ones_calls(), 0);

  // Only counting knows known_zero/known_ones: no IsFullyKnown or GetTernary.
  counting.ResetCounts();
  EXPECT_TRUE(mixed_uqe.IsAllZeros(known_zero.node()));
  EXPECT_TRUE(mixed_uqe.IsAllOnes(known_ones.node()));
  EXPECT_EQ(counting.is_fully_known_calls(), 0);
  EXPECT_EQ(counting.get_ternary_calls(), 0);
  EXPECT_EQ(counting.is_all_zeros_calls(), 1);
  EXPECT_EQ(counting.is_all_ones_calls(), 1);

  // Neither engine proves unknown_wide or unknown_bit: skips counting's
  // GetTernary because StatelessQueryEngine returned no ternary info (for
  // unknown_wide) and single-bit values cannot split bits across engines.
  counting.ResetCounts();
  EXPECT_FALSE(mixed_uqe.IsAllZeros(unknown_wide.node()));
  EXPECT_FALSE(mixed_uqe.IsAllOnes(unknown_wide.node()));
  EXPECT_FALSE(mixed_uqe.IsAllZeros(unknown_bit.node()));
  EXPECT_FALSE(mixed_uqe.IsAllOnes(unknown_bit.node()));
  EXPECT_EQ(counting.is_fully_known_calls(), 0);
  EXPECT_EQ(counting.get_ternary_calls(), 0);
  EXPECT_EQ(counting.is_all_zeros_calls(), 2);
  EXPECT_EQ(counting.is_all_ones_calls(), 2);

  // Stateless contributes upper 0b00XX of concat_zero, counting contributes
  // lower 0bXX00: combined ternary proves IsAllZeros without IsFullyKnown.
  counting.ResetCounts();
  EXPECT_TRUE(mixed_uqe.IsAllZeros(concat_zero.node()));
  EXPECT_EQ(counting.is_fully_known_calls(), 0);
  EXPECT_EQ(counting.get_ternary_calls(), 1);

  // Stateless sees a known 1 bit in concat_nonzero (0b01XX) and fails fast on
  // IsAllZeros without calling counting.GetTernary.
  counting.AddTernary(concat_nonzero.node(), make_bits_ternary("0bXX00"));
  counting.ResetCounts();
  EXPECT_FALSE(mixed_uqe.IsAllZeros(concat_nonzero.node()));
  EXPECT_EQ(counting.is_fully_known_calls(), 0);
  EXPECT_EQ(counting.get_ternary_calls(), 0);

  // Earlier engine returning an all-unknown ternary tree (as TernaryQueryEngine
  // and RangeQueryEngine do for tracked unknown nodes) contributes no target
  // bits, so the last engine's GetTernary is skipped.
  FakeQueryEngine all_unknown;
  all_unknown.AddTernary(unknown_wide.node(), make_bits_ternary("0bXXXX"));
  UnionQueryEngine unknown_then_counting =
      UnionQueryEngine::Of(&all_unknown, &counting);
  counting.ResetCounts();
  EXPECT_FALSE(unknown_then_counting.IsAllZeros(unknown_wide.node()));
  EXPECT_FALSE(unknown_then_counting.IsAllOnes(unknown_wide.node()));
  EXPECT_EQ(counting.is_fully_known_calls(), 0);
  EXPECT_EQ(counting.get_ternary_calls(), 0);
  EXPECT_EQ(counting.is_all_zeros_calls(), 1);
  EXPECT_EQ(counting.is_all_ones_calls(), 1);

  // An all-unknown engine followed by two partial engines still combines the
  // partial engines' bits.
  FakeQueryEngine partial_high;
  all_unknown.AddTernary(concat_zero.node(), make_bits_ternary("0bXXXX"));
  partial_high.AddTernary(concat_zero.node(), make_bits_ternary("0b00XX"));
  UnionQueryEngine three_engines =
      UnionQueryEngine::Of(&all_unknown, &partial_high, &counting);
  counting.ResetCounts();
  EXPECT_TRUE(three_engines.IsAllZeros(concat_zero.node()));
  EXPECT_EQ(counting.is_fully_known_calls(), 0);
  EXPECT_EQ(counting.get_ternary_calls(), 1);
}

}  // namespace
}  // namespace xls
