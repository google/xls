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

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/status/status.h"
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
// The unioned query engines are not owned and must live at least as long as
// this query engine does.
class UnownedUnionQueryEngine : public QueryEngine {
 public:
  explicit UnownedUnionQueryEngine(std::vector<QueryEngine*> engines)
      : engines_(std::move(engines)) {}
  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override;

  bool IsTracked(Node* node) const override;

  std::optional<SharedLeafTypeTree<TernaryVector>> GetTernary(
      Node* node) const override;

  LeafTypeTree<IntervalSet> GetIntervals(Node* node) const override;

  std::unique_ptr<QueryEngine> SpecializeGivenPredicate(
      const absl::btree_set<PredicateState>& state) const override;

  std::unique_ptr<QueryEngine> SpecializeGiven(
      const absl::btree_map<Node*, ValueKnowledge, Node::NodeIdLessThan>&
          givens) const override;

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
  using QueryEngine::KnownValue;
  std::optional<bool> KnownValue(const TreeBitLocation& bit) const override;
  bool IsAllZeros(Node* n) const override;
  bool IsAllOnes(Node* n) const override;

  Bits MaxUnsignedValue(Node* node) const override;
  Bits MinUnsignedValue(Node* node) const override;
  std::optional<int64_t> KnownLeadingZeros(Node* node) const override;
  std::optional<int64_t> KnownLeadingOnes(Node* node) const override;
  std::optional<int64_t> KnownLeadingSignBits(Node* node) const override;

 private:
  std::vector<QueryEngine*> engines_;
};

class UnownedConstUnionQueryEngine : public UnownedUnionQueryEngine {
 public:
  explicit UnownedConstUnionQueryEngine(std::vector<QueryEngine const*> engines)
      : UnownedUnionQueryEngine(UnsafeConstCast(std::move(engines))) {}

  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override {
    return absl::InternalError("Cannot repopulate const engines.");
  }

 private:
  static std::vector<QueryEngine*> UnsafeConstCast(
      std::vector<QueryEngine const*> engines) {
    std::vector<QueryEngine*> result;
    result.reserve(engines.size());
    for (const auto& engine : engines) {
      result.push_back(const_cast<QueryEngine*>(engine));
    }
    return result;
  }
};

// A query engine that combines the results of multiple given query engines.
class UnionQueryEngine : public UnownedUnionQueryEngine {
 public:
  // If any `unowned_engines` are provided, they must live at least as long as
  // this query engine does.
  explicit UnionQueryEngine(std::vector<std::unique_ptr<QueryEngine>> engines,
                            std::vector<QueryEngine*> unowned_engines = {})
      : UnownedUnionQueryEngine(ToUnownedVector(engines, unowned_engines)),
        owned_engines_(std::move(engines)) {}

  // Helper to create a union-query-engine with a compile constant set of
  // engines. All engines must be movable.
  template <typename... Engines>
  static UnionQueryEngine Of(Engines... e) {
    std::vector<std::unique_ptr<QueryEngine>> vec =
        MakeVec<sizeof...(Engines), Engines...>(std::forward<Engines>(e)...);
    // Reverse the list so that the order of arguments is the same as the order
    // in the list of unique_ptr<QueryEngine> we use to construct the actual
    // UnionQueryEngine. NB Assuming well-behaved QEs this should never be
    // semantically meaningful but it makes debugging easier.
    absl::c_reverse(vec);
    return UnionQueryEngine(std::move(vec));
  }

 private:
  static std::vector<QueryEngine*> ToUnownedVector(
      absl::Span<std::unique_ptr<QueryEngine> const> ptrs,
      absl::Span<QueryEngine* const> unowned_ptrs) {
    std::vector<QueryEngine*> result;
    result.reserve(ptrs.size() + unowned_ptrs.size());
    for (const auto& ptr : ptrs) {
      result.push_back(ptr.get());
    }
    absl::c_copy(unowned_ptrs, std::back_inserter(result));
    return result;
  }
  std::vector<std::unique_ptr<QueryEngine>> owned_engines_;

  template <size_t kCnt, typename E>
    requires(std::is_base_of_v<QueryEngine, E>)
  static std::vector<std::unique_ptr<QueryEngine>> MakeVec(E e) {
    std::vector<std::unique_ptr<QueryEngine>> res;
    res.reserve(kCnt);
    res.push_back(std::make_unique<E>(std::move(e)));
    return res;
  }

  template <size_t kCnt, typename E, typename... Engines>
    requires(std::is_base_of_v<QueryEngine, E>)
  static std::vector<std::unique_ptr<QueryEngine>> MakeVec(E e, Engines... es) {
    std::vector<std::unique_ptr<QueryEngine>> res =
        MakeVec<kCnt, Engines...>(std::forward<Engines>(es)...);
    res.emplace_back(std::make_unique<E>(std::move(e)));
    return res;
  }
};

}  // namespace xls

#endif  // XLS_PASSES_UNION_QUERY_ENGINE_H_
