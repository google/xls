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

#ifndef XLS_PASSES_FORWARDING_QUERY_ENGINE_H_
#define XLS_PASSES_FORWARDING_QUERY_ENGINE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/ternary.h"
#include "xls/ir/value.h"
#include "xls/passes/predicate_state.h"
#include "xls/passes/query_engine.h"
namespace xls {

// A forwarder for query engine.
//
// This overrides every virtual function of QueryEngine and forwards it to the
// real implementation.
class ForwardingQueryEngine : public QueryEngine {
 public:
  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override {
    return real().Populate(f);
  }

  bool IsTracked(Node* node) const override { return real().IsTracked(node); }

  std::optional<SharedLeafTypeTree<TernaryVector>> GetTernary(
      Node* node) const override {
    return real().GetTernary(node);
  };

  std::unique_ptr<QueryEngine> SpecializeGivenPredicate(
      const absl::btree_set<PredicateState>& state) const override {
    return real().SpecializeGivenPredicate(state);
  }
  std::unique_ptr<QueryEngine> SpecializeGiven(
      const absl::btree_map<Node*, ValueKnowledge, Node::NodeIdLessThan>&
          givens) const override {
    return real().SpecializeGiven(givens);
  }

  bool IsPredicatePossible(PredicateState state) const override {
    return real().IsPredicatePossible(state);
  }

  LeafTypeTree<IntervalSet> GetIntervals(Node* node) const override {
    return real().GetIntervals(node);
  }

  bool AtMostOneTrue(absl::Span<TreeBitLocation const> bits) const override {
    return real().AtMostOneTrue(bits);
  }

  bool AtLeastOneTrue(absl::Span<TreeBitLocation const> bits) const override {
    return real().AtLeastOneTrue(bits);
  }

  bool Implies(const TreeBitLocation& a,
               const TreeBitLocation& b) const override {
    return real().Implies(a, b);
  }

  std::optional<Bits> ImpliedNodeValue(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override {
    return real().ImpliedNodeValue(predicate_bit_values, node);
  }

  std::optional<TernaryVector> ImpliedNodeTernary(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override {
    return real().ImpliedNodeTernary(predicate_bit_values, node);
  }

  bool KnownEquals(const TreeBitLocation& a,
                   const TreeBitLocation& b) const override {
    return real().KnownEquals(a, b);
  }

  bool KnownNotEquals(const TreeBitLocation& a,
                      const TreeBitLocation& b) const override {
    return real().KnownNotEquals(a, b);
  }

  bool AtMostOneBitTrue(Node* node) const override {
    return real().AtMostOneBitTrue(node);
  }
  bool AtLeastOneBitTrue(Node* node) const override {
    return real().AtLeastOneBitTrue(node);
  }
  bool ExactlyOneBitTrue(Node* node) const override {
    return real().ExactlyOneBitTrue(node);
  }
  bool IsKnown(const TreeBitLocation& bit) const override {
    return real().IsKnown(bit);
  }
  std::optional<bool> KnownValue(const TreeBitLocation& bit) const override {
    return real().KnownValue(bit);
  }
  std::optional<Value> KnownValue(Node* node) const override {
    return real().KnownValue(node);
  }
  bool Covers(Node* n, const Bits& value) const override {
    return real().Covers(n, value);
  }
  bool IsAllZeros(Node* n) const override { return real().IsAllZeros(n); }
  bool IsAllOnes(Node* n) const override { return real().IsAllOnes(n); }
  bool IsFullyKnown(Node* n) const override { return real().IsFullyKnown(n); }
  Bits MaxUnsignedValue(Node* node) const override {
    return real().MaxUnsignedValue(node);
  }
  Bits MinUnsignedValue(Node* node) const override {
    return real().MinUnsignedValue(node);
  }

  std::optional<int64_t> KnownLeadingZeros(Node* node) const override {
    return real().KnownLeadingZeros(node);
  }
  std::optional<int64_t> KnownLeadingOnes(Node* node) const override {
    return real().KnownLeadingOnes(node);
  }
  std::optional<int64_t> KnownLeadingSignBits(Node* node) const override {
    return real().KnownLeadingSignBits(node);
  }

 protected:
  virtual QueryEngine& real() = 0;
  virtual const QueryEngine& real() const = 0;
};

}  // namespace xls

#endif  // XLS_PASSES_FORWARDING_QUERY_ENGINE_H_
