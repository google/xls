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

class FakeQueryEngine : public QueryEngine {
 public:
  FakeQueryEngine() {}

  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override {
    return ReachedFixpoint::Unchanged;
  }

  bool IsTracked(Node* node) const override { return tracked_.contains(node); }

  const Bits& GetKnownBits(Node* node) const override {
    return known_bits_.at(node);
  }

  const Bits& GetKnownBitsValues(Node* node) const override {
    return known_bit_values_.at(node);
  }

  bool AtMostOneTrue(absl::Span<BitLocation const> bits) const override {
    return at_most_one_true_.contains(
        std::vector<BitLocation>(bits.begin(), bits.end()));
  }

  bool AtLeastOneTrue(absl::Span<BitLocation const> bits) const override {
    return at_least_one_true_.contains(
        std::vector<BitLocation>(bits.begin(), bits.end()));
  }

  bool Implies(const BitLocation& a, const BitLocation& b) const override {
    return implications_.contains({a, b});
  }

  absl::optional<Bits> ImpliedNodeValue(
      absl::Span<const std::pair<BitLocation, bool>> predicate_bit_values,
      Node* node) const override {
    std::vector<std::pair<BitLocation, bool>> vec(predicate_bit_values.begin(),
                                                  predicate_bit_values.end());
    if (implied_node_values_.contains({vec, node})) {
      return implied_node_values_.at({vec, node});
    }
    return absl::nullopt;
  }

  bool KnownEquals(const BitLocation& a, const BitLocation& b) const override {
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

  bool KnownNotEquals(const BitLocation& a,
                      const BitLocation& b) const override {
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

  void AddKnownBit(const BitLocation& location, bool value) {
    Node* node = location.node;
    AddTracked(node);
    int64_t index = location.bit_index;
    int64_t width = node->GetType()->GetFlatBitCount();
    XLS_CHECK_LT(index, width);
    XLS_CHECK_EQ(known_bits_[node].bit_count(), width);
    XLS_CHECK_EQ(known_bit_values_[node].bit_count(), width);
    known_bits_[node] = known_bits_[node].UpdateWithSet(index, true);
    known_bit_values_[node] =
        known_bit_values_[node].UpdateWithSet(index, value);
  }

  void AddAtMostOneTrue(absl::Span<const BitLocation> span) {
    for (const BitLocation& location : span) {
      AddTracked(location.node);
    }
    at_most_one_true_.insert(
        std::vector<BitLocation>(span.begin(), span.end()));
  }

  void AddAtLeastOneTrue(absl::Span<const BitLocation> span) {
    for (const BitLocation& location : span) {
      AddTracked(location.node);
    }
    at_least_one_true_.insert(
        std::vector<BitLocation>(span.begin(), span.end()));
  }

  void AddImplication(const BitLocation& x, const BitLocation& y) {
    AddTracked(x.node);
    AddTracked(y.node);
    implications_.insert({x, y});
  }

  void AddImpliedNodeValue(absl::Span<const std::pair<BitLocation, bool>> span,
                           Node* node, const Bits& bits) {
    AddTracked(node);
    for (const auto& [location, value] : span) {
      AddTracked(location.node);
    }
    std::vector<std::pair<BitLocation, bool>> vec(span.begin(), span.end());
    implied_node_values_[{vec, node}] = bits;
  }

  void AddEquality(const BitLocation& x, const BitLocation& y) {
    AddTracked(x.node);
    AddTracked(y.node);
    equality_states_[{x, y}] = EqualityState::Equal;
    equality_states_[{y, x}] = EqualityState::Equal;
  }

  void AddInequality(const BitLocation& x, const BitLocation& y) {
    AddTracked(x.node);
    AddTracked(y.node);
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
  absl::flat_hash_set<std::vector<BitLocation>> at_most_one_true_;
  absl::flat_hash_set<std::vector<BitLocation>> at_least_one_true_;
  absl::flat_hash_set<std::pair<BitLocation, BitLocation>> implications_;
  absl::flat_hash_map<
      std::pair<std::vector<std::pair<BitLocation, bool>>, Node*>, Bits>
      implied_node_values_;
  absl::flat_hash_map<std::pair<BitLocation, BitLocation>, EqualityState>
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
  query_engine_a.AddKnownBit(BitLocation(node, 4), true);
  query_engine_a.AddEquality(BitLocation(node, 0), BitLocation(node, 1));
  query_engine_a.AddImplication(BitLocation(node, 3), BitLocation(node, 7));
  FakeQueryEngine query_engine_b;
  query_engine_b.AddInequality(BitLocation(node, 2), BitLocation(node, 3));
  query_engine_b.AddAtMostOneTrue({BitLocation(node, 2), BitLocation(node, 3)});
  query_engine_b.AddAtLeastOneTrue(
      {BitLocation(node, 2), BitLocation(node, 3)});
  query_engine_b.AddImpliedNodeValue(
      {{BitLocation(node, 7), true}, {BitLocation(node, 3), true}}, node,
      UBits(0b10011000, 8));
  query_engine_b.AddImplication(BitLocation(node, 7), BitLocation(node, 3));
  std::vector<std::unique_ptr<QueryEngine>> engines;
  engines.push_back(std::make_unique<FakeQueryEngine>(query_engine_a));
  engines.push_back(std::make_unique<FakeQueryEngine>(query_engine_b));
  UnionQueryEngine union_query_engine(std::move(engines));
  // No need to Populate, since FakeQueryEngine doesn't use that interface

  EXPECT_FALSE(union_query_engine.IsTracked(nullptr));
  EXPECT_TRUE(union_query_engine.IsTracked(node));
  EXPECT_EQ(union_query_engine.GetKnownBits(node), UBits(0b00010000, 8));
  EXPECT_EQ(union_query_engine.GetKnownBitsValues(node), UBits(0b00010000, 8));
  // Query the same ones again to test the caching of known bits
  EXPECT_EQ(union_query_engine.GetKnownBits(node), UBits(0b00010000, 8));
  EXPECT_EQ(union_query_engine.GetKnownBitsValues(node), UBits(0b00010000, 8));
  EXPECT_TRUE(union_query_engine.AtMostOneTrue(
      {BitLocation(node, 2), BitLocation(node, 3)}));
  EXPECT_FALSE(union_query_engine.AtMostOneTrue(
      {BitLocation(node, 5), BitLocation(node, 6)}));
  EXPECT_TRUE(union_query_engine.AtLeastOneTrue(
      {BitLocation(node, 2), BitLocation(node, 3)}));
  EXPECT_FALSE(union_query_engine.AtLeastOneTrue(
      {BitLocation(node, 5), BitLocation(node, 6)}));
  EXPECT_TRUE(union_query_engine.KnownEquals(BitLocation(node, 0),
                                             BitLocation(node, 1)));
  EXPECT_FALSE(union_query_engine.KnownEquals(BitLocation(node, 5),
                                              BitLocation(node, 6)));
  EXPECT_TRUE(union_query_engine.KnownNotEquals(BitLocation(node, 2),
                                                BitLocation(node, 3)));
  EXPECT_FALSE(union_query_engine.KnownNotEquals(BitLocation(node, 5),
                                                 BitLocation(node, 6)));
  EXPECT_TRUE(
      union_query_engine.Implies(BitLocation(node, 3), BitLocation(node, 7)));
  EXPECT_TRUE(
      union_query_engine.Implies(BitLocation(node, 7), BitLocation(node, 3)));
  EXPECT_FALSE(
      union_query_engine.Implies(BitLocation(node, 5), BitLocation(node, 6)));
  EXPECT_EQ(
      union_query_engine.ImpliedNodeValue(
          {{BitLocation(node, 7), true}, {BitLocation(node, 3), true}}, node),
      UBits(0b10011000, 8));
  EXPECT_FALSE(union_query_engine.ImpliedNodeValue(
      {{BitLocation(node, 5), true}, {BitLocation(node, 6), true}}, node));
}

}  // namespace
}  // namespace xls
