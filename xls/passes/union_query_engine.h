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

#ifndef XLS_PASSES_UNION_QUERY_ENGINE_H_
#define XLS_PASSES_UNION_QUERY_ENGINE_H_

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/ternary.h"
#include "xls/passes/predicate_state.h"
#include "xls/passes/query_engine.h"

namespace xls {

// A query engine that combines the results of multiple (unowned) given query
// engines.
//
// `GetKnownBits` and `GetKnownBitsValues` use `const_cast<...>(this)` under the
// hood, so it is undefined behavior to define a `const UnionQueryEngine`
// variable (but `const UnionQueryEngine*` is fine, the storage location just
// must be mutable). This is due to an infelicity in the QueryEngine API that
// will be fixed at some point.
//
// The unioned query engines are not owned and must live at least as long as
// this query engine does.
class UnownedUnionQueryEngine : public QueryEngine {
 public:
  explicit UnownedUnionQueryEngine(std::vector<QueryEngine*> engines)
      : engines_(std::move(engines)) {}
  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override;

  bool IsTracked(Node* node) const override;

  std::optional<LeafTypeTree<TernaryVector>> GetTernary(
      Node* node) const override;

  LeafTypeTree<IntervalSet> GetIntervals(Node* node) const override;

  std::unique_ptr<QueryEngine> SpecializeGivenPredicate(
      const absl::flat_hash_set<PredicateState>& state) const override;

  bool AtMostOneTrue(absl::Span<TreeBitLocation const> bits) const override;

  bool AtLeastOneTrue(absl::Span<TreeBitLocation const> bits) const override;

  bool KnownEquals(const TreeBitLocation& a,
                   const TreeBitLocation& b) const override;

  bool KnownNotEquals(const TreeBitLocation& a,
                      const TreeBitLocation& b) const override;

  bool Implies(const TreeBitLocation& a,
               const TreeBitLocation& b) const override;

  std::optional<Bits> ImpliedNodeValue(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override;

  std::optional<TernaryVector> ImpliedNodeTernary(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override;

  bool IsKnown(const TreeBitLocation& bit) const override;
  std::optional<bool> KnownValue(const TreeBitLocation& bit) const override;
  bool IsAllZeros(Node* n) const override;
  bool IsAllOnes(Node* n) const override;

 private:
  absl::flat_hash_map<Node*, Bits> known_bits_;
  absl::flat_hash_map<Node*, Bits> known_bit_values_;
  std::vector<QueryEngine*> engines_;
};

// A query engine that combines the results of multiple given query engines.
//
// `GetKnownBits` and `GetKnownBitsValues` use `const_cast<...>(this)` under the
// hood, so it is undefined behavior to define a `const UnionQueryEngine`
// variable (but `const UnionQueryEngine*` is fine, the storage location just
// must be mutable). This is due to an infelicity in the QueryEngine API that
// will be fixed at some point.
class UnionQueryEngine : public UnownedUnionQueryEngine {
 public:
  explicit UnionQueryEngine(std::vector<std::unique_ptr<QueryEngine>> engines)
      : UnownedUnionQueryEngine(ToUnownedVector(engines)),
        owned_engines_(std::move(engines)) {}

 private:
  static std::vector<QueryEngine*> ToUnownedVector(
      absl::Span<std::unique_ptr<QueryEngine> const> ptrs) {
    std::vector<QueryEngine*> result;
    result.reserve(ptrs.size());
    for (const auto& ptr : ptrs) {
      result.push_back(ptr.get());
    }
    return result;
  }
  std::vector<std::unique_ptr<QueryEngine>> owned_engines_;
};

}  // namespace xls

#endif  // XLS_PASSES_UNION_QUERY_ENGINE_H_
