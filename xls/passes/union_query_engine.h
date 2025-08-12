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
  UnownedUnionQueryEngine(UnownedUnionQueryEngine&&) = default;
  UnownedUnionQueryEngine& operator=(UnownedUnionQueryEngine&&) = default;
  UnownedUnionQueryEngine(const UnownedUnionQueryEngine&) = default;
  UnownedUnionQueryEngine& operator=(const UnownedUnionQueryEngine&) = default;

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

  bool IsPredicatePossible(PredicateState state) const override;

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

  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) final {
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
 private:
  struct QueryEngineReferences {
    std::vector<std::unique_ptr<QueryEngine>> owned;
    std::vector<QueryEngine*> pointers;
  };

 public:
  // If any `unowned_engines` are provided, they must live at least as long as
  // this query engine does.
  explicit UnionQueryEngine(
      std::vector<std::unique_ptr<QueryEngine>> engines = {},
      std::vector<QueryEngine*> unowned_engines = {})
      : UnownedUnionQueryEngine(ToUnownedVector(engines, unowned_engines)),
        owned_engines_(std::move(engines)) {}

  UnionQueryEngine(UnionQueryEngine&&) = default;
  UnionQueryEngine& operator=(UnionQueryEngine&&) = default;
  UnionQueryEngine(const UnionQueryEngine&) = delete;
  UnionQueryEngine& operator=(const UnionQueryEngine&) = delete;

  // Helper to create a union-query-engine with a compile constant set of
  // engines. All engines must be movable or a pointer which bounds this engines
  // lifetime.
  template <typename... Engines>
  static UnionQueryEngine Of(Engines... e) {
    QueryEngineReferences vecs =
        MakeVecs<sizeof...(Engines), Engines...>(std::forward<Engines>(e)...);
    auto [owned, unowned] = std::move(vecs);
    // Reverse the list so that the order of arguments is the same as the order
    // in the list of unique_ptr<QueryEngine> we use to construct the actual
    // UnionQueryEngine. NB Assuming well-behaved QEs this should never be
    // semantically meaningful but it makes debugging easier.
    absl::c_reverse(unowned);
    return UnionQueryEngine(std::in_place, std::move(owned),
                            std::move(unowned));
  }

 private:
  // Private constructor with a set order.
  UnionQueryEngine(std::in_place_t,
                   std::vector<std::unique_ptr<QueryEngine>> engines,
                   std::vector<QueryEngine*> all_engines)
      : UnownedUnionQueryEngine(std::move(all_engines)),
        owned_engines_(std::move(engines)) {}
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
    requires(std::is_base_of_v<QueryEngine, E> ||
             std::is_convertible_v<E, std::unique_ptr<QueryEngine>> ||
             (std::is_pointer_v<E> &&
              std::is_base_of_v<QueryEngine, std::remove_pointer_t<E>>))
  static QueryEngineReferences MakeVecs(E e) {
    QueryEngineReferences res;
    res.owned.reserve(kCnt);
    res.pointers.reserve(kCnt);
    if constexpr (std::is_pointer_v<E>) {
      res.pointers.push_back(e);
    } else if constexpr (std::is_base_of_v<QueryEngine, E>) {
      std::unique_ptr<QueryEngine> e_out = std::make_unique<E>(std::move(e));
      QueryEngine* e_ptr = e_out.get();
      res.owned.push_back(std::move(e_out));
      res.pointers.push_back(e_ptr);
    } else {
      res.owned.push_back(std::unique_ptr<QueryEngine>(std::move(e)));
      QueryEngine* e_ptr = res.owned.back().get();
      res.pointers.push_back(e_ptr);
    }
    return res;
  }

  template <size_t kCnt, typename E, typename... Engines>
    requires(std::is_base_of_v<QueryEngine, E> ||
             std::is_convertible_v<E, std::unique_ptr<QueryEngine>> ||
             (std::is_pointer_v<E> &&
              std::is_base_of_v<QueryEngine, std::remove_pointer_t<E>>))
  static QueryEngineReferences MakeVecs(E e, Engines... es) {
    QueryEngineReferences res =
        MakeVecs<kCnt, Engines...>(std::forward<Engines>(es)...);
    if constexpr (std::is_pointer_v<E>) {
      res.pointers.push_back(e);
    } else if constexpr (std::is_base_of_v<QueryEngine, E>) {
      auto e_out = std::make_unique<E>(std::move(e));
      QueryEngine* e_ptr = e_out.get();
      res.owned.push_back(std::move(e_out));
      res.pointers.push_back(e_ptr);
    } else {
      res.owned.push_back(std::unique_ptr<QueryEngine>(std::move(e)));
      res.pointers.push_back(res.owned.back().get());
    }
    return res;
  }
};

}  // namespace xls

#endif  // XLS_PASSES_UNION_QUERY_ENGINE_H_
