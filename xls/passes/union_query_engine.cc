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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/predicate_state.h"
#include "xls/passes/query_engine.h"

namespace xls {
namespace {

bool IsAllTernaryValue(absl::Span<QueryEngine* const> engines, Node* n,
                       bool target_bit) {
  if (TypeHasToken(n->GetType())) {
    return false;
  }
  // A literal whose value is not all `target_bit` cannot be all `target_bit`,
  // so fail fast before querying slower engines (such as BDDs or ternary
  // caches).
  if (n->Is<Literal>()) {
    const Value& value = n->As<Literal>()->value();
    if (target_bit) {
      if (!value.IsAllOnes()) {
        return false;
      }
    } else {
      if (!value.IsAllZeros()) {
        return false;
      }
    }
  }
  for (const QueryEngine* engine : engines) {
    if (target_bit ? engine->IsAllOnes(n) : engine->IsAllZeros(n)) {
      return true;
    }
  }
  // A single engine cannot combine partial knowledge with another engine, and a
  // 0- or 1-bit value has no distinct bits to split across multiple engines.
  if (engines.size() <= 1 || n->GetType()->GetFlatBitCount() <= 1) {
    return false;
  }
  // No single engine proved `n` is all `target_bit`. Check whether combining
  // partial ternary knowledge across at least two engines proves it, failing
  // fast on any known opposite bit and ignoring engines that contribute no
  // `target` bits (so the last engine is skipped if no earlier engine knew any
  // `target` bit).
  const TernaryValue target =
      target_bit ? TernaryValue::kKnownOne : TernaryValue::kKnownZero;
  const TernaryValue opposite =
      target_bit ? TernaryValue::kKnownZero : TernaryValue::kKnownOne;
  auto is_all_target =
      [target_bit](const LeafTypeTreeView<TernaryVector>& tree) {
        return absl::c_all_of(tree.elements(),
                              [target_bit](const TernaryVector& v) {
                                return target_bit ? ternary_ops::IsKnownOne(v)
                                                  : ternary_ops::IsKnownZero(v);
                              });
      };
  std::optional<SharedLeafTypeTree<TernaryVector>> shared;
  std::optional<LeafTypeTree<TernaryVector>> owned;
  for (size_t i = 0; i < engines.size(); ++i) {
    if (i + 1 == engines.size() && !shared.has_value() && !owned.has_value()) {
      return false;
    }
    const QueryEngine* engine = engines[i];
    if (!engine->IsTracked(n)) {
      continue;
    }
    std::optional<SharedLeafTypeTree<TernaryVector>> ternary =
        engine->GetTernary(n);
    if (!ternary.has_value()) {
      continue;
    }
    bool has_target = false;
    for (const TernaryVector& v : ternary->elements()) {
      for (TernaryValue bit : v) {
        if (bit == opposite) {
          return false;
        }
        if (bit == target) {
          has_target = true;
        }
      }
    }
    if (!has_target) {
      continue;
    }
    if (owned.has_value() || shared.has_value()) {
      if (shared.has_value()) {
        owned = std::move(shared).value().ToOwned();
        shared.reset();
      }
      leaf_type_tree::SimpleUpdateFrom<TernaryVector, TernaryVector>(
          owned->AsMutableView(), ternary->AsView(),
          [](TernaryVector& lhs, const TernaryVector& rhs) {
            CHECK_OK(ternary_ops::UpdateWithUnion(lhs, rhs));
          });
      if (is_all_target(owned->AsView())) {
        return true;
      }
    } else {
      shared = *std::move(ternary);
    }
  }
  return false;
}

}  // namespace

bool UnownedUnionQueryEngine::IsKnown(const TreeBitLocation& bit) const {
  return absl::c_any_of(engines_, [&](const QueryEngine* qe) -> bool {
    return qe->IsKnown(bit);
  });
}

bool UnownedUnionQueryEngine::IsFullyKnown(Node* n) const {
  // Fastpath check if any qe happens to know the value all on its own.
  if (absl::c_any_of(engines_, [&](const QueryEngine* qe) -> bool {
        return qe->IsFullyKnown(n);
      })) {
    return true;
  }
  // Fallback to the ternary based one.
  return QueryEngine::IsFullyKnown(n);
}

std::optional<Value> UnownedUnionQueryEngine::KnownValue(Node* node) const {
  for (const QueryEngine* engine : engines_) {
    std::optional<Value> res = engine->KnownValue(node);
    if (res) {
      return res;
    }
  }
  // Fallback to the ternary-based union of everything.
  return QueryEngine::KnownValue(node);
}

std::optional<bool> UnownedUnionQueryEngine::KnownValue(
    const TreeBitLocation& bit) const {
  for (const QueryEngine* engine : engines_) {
    std::optional<bool> res = engine->KnownValue(bit);
    if (res) {
      return res;
    }
  }
  return std::nullopt;
}

bool UnownedUnionQueryEngine::IsAllOnes(Node* n) const {
  return IsAllTernaryValue(engines_, n, /*target_bit=*/true);
}

bool UnownedUnionQueryEngine::IsAllZeros(Node* n) const {
  return IsAllTernaryValue(engines_, n, /*target_bit=*/false);
}

absl::StatusOr<ReachedFixpoint> UnownedUnionQueryEngine::Populate(
    FunctionBase* f) {
  ReachedFixpoint result = ReachedFixpoint::Unchanged;
  for (QueryEngine* engine : engines_) {
    XLS_ASSIGN_OR_RETURN(ReachedFixpoint rf, engine->Populate(f));
    // Unchanged is the top of the lattice so it's an identity
    if (result == ReachedFixpoint::Unchanged) {
      result = rf;
    }
    // Changed can only degrade to Unknown
    if ((result == ReachedFixpoint::Changed) &&
        (rf == ReachedFixpoint::Unknown)) {
      result = ReachedFixpoint::Unknown;
    }
    // No case needed for ReachedFixpoint::Unknown since it's already the bottom
    // of the lattice
  }
  return result;
}

bool UnownedUnionQueryEngine::IsTracked(Node* node) const {
  for (const auto& engine : engines_) {
    if (engine->IsTracked(node)) {
      return true;
    }
  }
  return false;
}

std::optional<SharedLeafTypeTree<TernaryVector>>
UnownedUnionQueryEngine::GetTernary(Node* node) const {
  std::optional<LeafTypeTree<TernaryVector>> owned = std::nullopt;
  std::optional<SharedLeafTypeTree<TernaryVector>> shared = std::nullopt;
  for (const auto& engine : engines_) {
    CHECK(!owned || !shared) << "Both owned and unowned ltts present.";
    if (engine->IsTracked(node)) {
      std::optional<SharedLeafTypeTree<TernaryVector>> ternary =
          engine->GetTernary(node);
      if (!ternary) {
        continue;
      }
      if (owned || shared) {
        if (shared) {
          // Actually merging things so do the copy.
          owned = std::move(shared).value().ToOwned();
          shared.reset();
        }
        leaf_type_tree::SimpleUpdateFrom<TernaryVector, TernaryVector>(
            owned->AsMutableView(), ternary->AsView(),
            [](TernaryVector& lhs, const TernaryVector& rhs) {
              CHECK_OK(ternary_ops::UpdateWithUnion(lhs, rhs));
            });
      } else {
        // Avoid copying if we can.
        shared = *std::move(ternary);
      }
    }
  }
  CHECK(!owned || !shared) << "Both owned and unowned ltts present.";
  if (owned) {
    return std::move(*owned).AsShared();
  }
  return shared;
}

std::unique_ptr<QueryEngine> UnownedUnionQueryEngine::SpecializeGivenPredicate(
    const absl::btree_set<PredicateState>& state) const {
  std::vector<std::unique_ptr<QueryEngine>> engines;
  engines.reserve(engines_.size());
  for (const auto& engine : engines_) {
    engines.push_back(engine->SpecializeGivenPredicate(state));
  }
  return std::make_unique<UnionQueryEngine>(std::move(engines));
}

bool UnownedUnionQueryEngine::IsPredicatePossible(PredicateState state) const {
  for (const auto& engine : engines_) {
    if (!engine->IsPredicatePossible(state)) {
      return false;
    }
  }
  return QueryEngine::IsPredicatePossible(state);
}

std::unique_ptr<QueryEngine> UnownedUnionQueryEngine::SpecializeGiven(
    const absl::btree_map<Node*, ValueKnowledge, Node::NodeIdLessThan>& givens)
    const {
  std::vector<std::unique_ptr<QueryEngine>> engines;
  engines.reserve(engines_.size());
  for (const auto& engine : engines_) {
    engines.push_back(engine->SpecializeGiven(givens));
  }
  return std::make_unique<UnionQueryEngine>(std::move(engines));
}

LeafTypeTree<IntervalSet> UnownedUnionQueryEngine::GetIntervals(
    Node* node) const {
  LeafTypeTree<IntervalSet> result(node->GetType());
  for (int64_t i = 0; i < result.size(); ++i) {
    result.elements()[i] =
        IntervalSet::Maximal(result.leaf_types()[i]->GetFlatBitCount());
  }
  for (const auto& engine : engines_) {
    if (engine->IsTracked(node)) {
      leaf_type_tree::SimpleUpdateFrom<IntervalSet, IntervalSet>(
          result.AsMutableView(), engine->GetIntervals(node).AsView(),
          [](IntervalSet& lhs, const IntervalSet& rhs) {
            lhs = IntervalSet::Intersect(lhs, rhs);
          });
    }
  }
  return result;
}

absl::StatusOr<std::unique_ptr<QueryEngine>>
UnownedUnionQueryEngine::SpecializeOnNodes(
    absl::Span<Node* const> nodes,
    const QueryEngine& information_source) const {
  std::vector<std::unique_ptr<QueryEngine>> engines;
  engines.reserve(engines_.size());
  for (const auto& engine : engines_) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<QueryEngine> specialized,
                         engine->SpecializeOnNodes(nodes, information_source));
    engines.emplace_back(std::move(specialized));
  }
  return std::make_unique<UnionQueryEngine>(std::move(engines));
}

bool UnownedUnionQueryEngine::AtMostOneTrue(
    absl::Span<TreeBitLocation const> bits) const {
  for (const auto& engine : engines_) {
    if (engine->AtMostOneTrue(bits)) {
      return true;
    }
  }
  return false;
}

bool UnownedUnionQueryEngine::AtLeastOneTrue(
    absl::Span<TreeBitLocation const> bits) const {
  for (const auto& engine : engines_) {
    if (engine->AtLeastOneTrue(bits)) {
      return true;
    }
  }
  return false;
}

bool UnownedUnionQueryEngine::AtMostOneBitTrue(Node* node) const {
  return std::any_of(engines_.begin(), engines_.end(),
                     [&](const QueryEngine* engine) {
                       return engine->AtMostOneBitTrue(node);
                     });
}

bool UnownedUnionQueryEngine::AtLeastOneBitTrue(Node* node) const {
  return std::any_of(engines_.begin(), engines_.end(),
                     [&](const QueryEngine* engine) {
                       return engine->AtLeastOneBitTrue(node);
                     });
}

bool UnownedUnionQueryEngine::ExactlyOneBitTrue(Node* node) const {
  return std::any_of(engines_.begin(), engines_.end(),
                     [&](const QueryEngine* engine) {
                       return engine->ExactlyOneBitTrue(node);
                     });
}

bool UnownedUnionQueryEngine::KnownEquals(const TreeBitLocation& a,
                                          const TreeBitLocation& b) const {
  for (const auto& engine : engines_) {
    if (engine->KnownEquals(a, b)) {
      return true;
    }
  }
  return false;
}

bool UnownedUnionQueryEngine::KnownNotEquals(const TreeBitLocation& a,
                                             const TreeBitLocation& b) const {
  for (const auto& engine : engines_) {
    if (engine->KnownNotEquals(a, b)) {
      return true;
    }
  }
  return false;
}

bool UnownedUnionQueryEngine::Implies(const TreeBitLocation& a,
                                      const TreeBitLocation& b) const {
  for (const auto& engine : engines_) {
    if (engine->Implies(a, b)) {
      return true;
    }
  }
  return false;
}

std::optional<Bits> UnownedUnionQueryEngine::ImpliedNodeValue(
    absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
    Node* node) const {
  for (const auto& engine : engines_) {
    if (auto i = engine->ImpliedNodeValue(predicate_bit_values, node)) {
      return i;
    }
  }
  return std::nullopt;
}

std::optional<TernaryVector> UnownedUnionQueryEngine::ImpliedNodeTernary(
    absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
    Node* node) const {
  std::optional<TernaryVector> result = std::nullopt;
  for (const auto& engine : engines_) {
    if (std::optional<TernaryVector> implied =
            engine->ImpliedNodeTernary(predicate_bit_values, node);
        implied.has_value()) {
      if (result.has_value()) {
        CHECK_OK(ternary_ops::UpdateWithUnion(*result, *implied));
      } else {
        result = std::move(implied);
      }
    }
  }
  return result;
}

Bits UnownedUnionQueryEngine::MaxUnsignedValue(Node* node) const {
  CHECK(node->GetType()->IsBits()) << node;
  Bits result = engines_.front()->IsTracked(node)
                    ? engines_.front()->MaxUnsignedValue(node)
                    : Bits::AllOnes(node->BitCountOrDie());
  for (const auto& engine : absl::MakeConstSpan(engines_).subspan(1)) {
    Bits eng_res = engine->IsTracked(node)
                       ? engine->MaxUnsignedValue(node)
                       : Bits::AllOnes(node->BitCountOrDie());
    if (bits_ops::ULessThan(eng_res, result)) {
      result = std::move(eng_res);
    }
  }
  return result;
}

Bits UnownedUnionQueryEngine::MinUnsignedValue(Node* node) const {
  CHECK(node->GetType()->IsBits()) << node;
  CHECK(node->GetType()->IsBits()) << node;
  Bits result = engines_.front()->IsTracked(node)
                    ? engines_.front()->MinUnsignedValue(node)
                    : Bits(node->BitCountOrDie());
  for (const auto& engine : absl::MakeConstSpan(engines_).subspan(1)) {
    Bits eng_res = engine->IsTracked(node) ? engine->MinUnsignedValue(node)
                                           : Bits(node->BitCountOrDie());
    if (bits_ops::UGreaterThan(eng_res, result)) {
      result = std::move(eng_res);
    }
  }
  return result;
}

// TODO(allight): This and the other ones should consider also calling the base
// QueryEngine::KnownLeading... functions to try to take advantage of any cases
// where different QEs might know things about different bits. OTOH its unlikely
// to get much more info in most cases.
std::optional<int64_t> UnownedUnionQueryEngine::KnownLeadingZeros(
    Node* node) const {
  std::optional<int64_t> best = std::nullopt;
  for (const auto& engine : engines_) {
    std::optional<int64_t> cur = engine->KnownLeadingZeros(node);
    if (!best || (cur && *cur > *best)) {
      best = cur;
    }
  }
  return best;
}
std::optional<int64_t> UnownedUnionQueryEngine::KnownLeadingOnes(
    Node* node) const {
  std::optional<int64_t> best = std::nullopt;
  for (const auto& engine : engines_) {
    std::optional<int64_t> cur = engine->KnownLeadingOnes(node);
    if (!best || (cur && *cur > *best)) {
      best = cur;
    }
  }
  return best;
}
std::optional<int64_t> UnownedUnionQueryEngine::KnownLeadingSignBits(
    Node* node) const {
  std::optional<int64_t> best = std::nullopt;
  for (const auto& engine : engines_) {
    std::optional<int64_t> cur = engine->KnownLeadingSignBits(node);
    if (!best || (cur && *cur > *best)) {
      best = cur;
    }
  }
  return best;
}

}  // namespace xls
