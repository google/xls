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
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/passes/predicate_state.h"
#include "xls/passes/query_engine.h"

namespace xls {

bool UnownedUnionQueryEngine::IsKnown(const TreeBitLocation& bit) const {
  return absl::c_any_of(engines_, [&](const QueryEngine* qe) -> bool {
    return qe->IsKnown(bit);
  });
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
  for (const QueryEngine* engine : engines_) {
    if (engine->IsFullyKnown(n)) {
      return engine->IsAllOnes(n);
    }
  }
  if (engines_.size() <= 1) {
    return false;
  }
  return QueryEngine::IsAllOnes(n);
}

bool UnownedUnionQueryEngine::IsAllZeros(Node* n) const {
  for (const QueryEngine* engine : engines_) {
    if (engine->IsFullyKnown(n)) {
      return engine->IsAllZeros(n);
    }
  }
  if (engines_.size() <= 1) {
    return false;
  }
  return QueryEngine::IsAllZeros(n);
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

std::optional<LeafTypeTree<TernaryVector>> UnownedUnionQueryEngine::GetTernary(
    Node* node) const {
  std::optional<LeafTypeTree<TernaryVector>> result = std::nullopt;
  for (const auto& engine : engines_) {
    if (engine->IsTracked(node)) {
      std::optional<LeafTypeTree<TernaryVector>> ternary =
          engine->GetTernary(node);
      if (!ternary.has_value()) {
        continue;
      }
      if (!result.has_value()) {
        result = std::move(ternary);
        continue;
      }
      leaf_type_tree::SimpleUpdateFrom<TernaryVector, TernaryVector>(
          result->AsMutableView(), ternary->AsView(),
          [](TernaryVector& lhs, const TernaryVector& rhs) {
            CHECK_OK(ternary_ops::UpdateWithUnion(lhs, rhs));
          });
    }
  }
  return result;
}

std::unique_ptr<QueryEngine> UnownedUnionQueryEngine::SpecializeGivenPredicate(
    const absl::flat_hash_set<PredicateState>& state) const {
  std::vector<std::unique_ptr<QueryEngine>> engines;
  engines.reserve(engines_.size());
  for (const auto& engine : engines_) {
    engines.push_back(engine->SpecializeGivenPredicate(state));
  }
  return std::make_unique<UnionQueryEngine>(std::move(engines));
}

std::unique_ptr<QueryEngine> UnownedUnionQueryEngine::SpecializeGiven(
    const absl::flat_hash_map<Node*, ValueKnowledge>& givens) const {
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

}  // namespace xls
