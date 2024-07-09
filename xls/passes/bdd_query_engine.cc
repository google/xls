// Copyright 2020 The XLS Authors
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

#include "xls/passes/bdd_query_engine.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/binary_decision_diagram.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/node.h"
#include "xls/ir/ternary.h"
#include "xls/passes/bdd_function.h"
#include "xls/passes/query_engine.h"

namespace xls {

absl::StatusOr<ReachedFixpoint> BddQueryEngine::Populate(FunctionBase* f) {
  XLS_ASSIGN_OR_RETURN(bdd_function_,
                       BddFunction::Run(f, path_limit_, node_filter_));
  // Construct the Bits objects indication which bit values are statically known
  // for each node and what those values are (0 or 1) if known.
  BinaryDecisionDiagram& bdd = this->bdd();
  ReachedFixpoint rf = ReachedFixpoint::Unchanged;
  for (Node* node : f->nodes()) {
    if (node->GetType()->IsBits()) {
      absl::InlinedVector<bool, 1> known_bits;
      absl::InlinedVector<bool, 1> bits_values;
      for (int64_t i = 0; i < node->BitCountOrDie(); ++i) {
        if (GetBddNode(TreeBitLocation(node, i)) == bdd.zero()) {
          known_bits.push_back(true);
          bits_values.push_back(false);
        } else if (GetBddNode(TreeBitLocation(node, i)) == bdd.one()) {
          known_bits.push_back(true);
          bits_values.push_back(true);
        } else {
          known_bits.push_back(false);
          bits_values.push_back(false);
        }
      }
      if (!known_bits_.contains(node)) {
        known_bits_[node] = Bits(known_bits.size());
        bits_values_[node] = Bits(bits_values.size());
      }
      Bits new_known_bits(known_bits);
      Bits new_bits_values(bits_values);
      // TODO(taktoa): check for inconsistency
      Bits ored_known_bits = bits_ops::Or(known_bits_[node], new_known_bits);
      Bits ored_bits_values = bits_ops::Or(bits_values_[node], new_bits_values);
      if ((ored_known_bits != known_bits_[node]) ||
          (ored_bits_values != bits_values_[node])) {
        rf = ReachedFixpoint::Changed;
      }
      known_bits_[node] = ored_known_bits;
      bits_values_[node] = ored_bits_values;
    }
  }
  return rf;
}

bool BddQueryEngine::AtMostOneTrue(
    absl::Span<TreeBitLocation const> bits) const {
  // Computing this property is quadratic (at least) so limit the width.
  const int64_t kMaxWidth = 64;
  if (bits.size() > kMaxWidth) {
    return false;
  }

  BddNodeIndex result = bdd().zero();
  for (const TreeBitLocation& loc : bits) {
    if (!IsTracked(loc.node())) {
      return false;
    }
  }

  // Compute the OR-reduction of a pairwise AND of all bits. If this value is
  // zero then no two bits can be simultaneously true. Equivalently: at most one
  // bit is true.
  for (int64_t i = 0; i < bits.size(); ++i) {
    std::optional<BddNodeIndex> i_bdd = GetBddNode(bits[i]);
    if (!i_bdd.has_value()) {
      return false;
    }
    for (int64_t j = i + 1; j < bits.size(); ++j) {
      std::optional<BddNodeIndex> j_bdd = GetBddNode(bits[j]);
      if (!j_bdd.has_value()) {
        return false;
      }
      result = bdd().Or(result, bdd().And(*i_bdd, *j_bdd));
      if (ExceedsPathLimit(result)) {
        VLOG(3) << "AtMostOneTrue exceeded path limit of " << path_limit_;
        return false;
      }
    }
  }
  return result == bdd().zero();
}

bool BddQueryEngine::AtLeastOneTrue(
    absl::Span<TreeBitLocation const> bits) const {
  BddNodeIndex result = bdd().zero();
  // At least one bit is true is equivalent to an OR-reduction of all the bits.
  for (const TreeBitLocation& location : bits) {
    if (!IsTracked(location.node())) {
      return false;
    }
    std::optional<BddNodeIndex> bdd_node = GetBddNode(location);
    if (!bdd_node.has_value()) {
      return false;
    }
    result = bdd().Or(result, *bdd_node);
    if (ExceedsPathLimit(result)) {
      VLOG(3) << "AtLeastOneTrue exceeded path limit of " << path_limit_;
      return false;
    }
  }
  return result == bdd().one();
}

bool BddQueryEngine::Implies(const BddNodeIndex& a,
                             const BddNodeIndex& b) const {
  // A implies B  <=>  !(A && !B)
  return bdd().And(a, bdd().Not(b)) == bdd().zero();
}

bool BddQueryEngine::Implies(const TreeBitLocation& a,
                             const TreeBitLocation& b) const {
  if (!IsTracked(a.node()) || !IsTracked(b.node())) {
    return false;
  }
  std::optional<BddNodeIndex> a_bdd = GetBddNode(a);
  if (!a_bdd.has_value()) {
    return false;
  }
  std::optional<BddNodeIndex> b_bdd = GetBddNode(b);
  if (!b_bdd.has_value()) {
    return false;
  }
  return Implies(*a_bdd, *b_bdd);
}

std::optional<Bits> BddQueryEngine::ImpliedNodeValue(
    absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
    Node* node) const {
  std::optional<TernaryVector> implied_ternary =
      ImpliedNodeTernary(predicate_bit_values, node);
  if (!implied_ternary.has_value() ||
      !ternary_ops::IsFullyKnown(*implied_ternary)) {
    return std::nullopt;
  }
  return ternary_ops::ToKnownBitsValues(*implied_ternary);
}

std::optional<TernaryVector> BddQueryEngine::ImpliedNodeTernary(
    absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
    Node* node) const {
  if (!IsTracked(node) || !node->GetType()->IsBits()) {
    return std::nullopt;
  }

  // Create a Bdd node for the predicate_bit_values.
  BddNodeIndex bdd_predicate_bit = bdd().one();
  for (const auto& [conjuction_bit_location, conjunction_value] :
       predicate_bit_values) {
    std::optional<BddNodeIndex> conjuction_bit =
        GetBddNode(conjuction_bit_location);
    if (!conjuction_bit.has_value()) {
      // Skip this predicate; we don't recognize the node, so we can't see the
      // effects of assuming it.
      continue;
    }
    conjuction_bit =
        conjunction_value ? *conjuction_bit : bdd().Not(*conjuction_bit);
    bdd_predicate_bit = bdd().And(bdd_predicate_bit, *conjuction_bit);
  }
  // If the predicate evaluates to false, we can't determine
  // what node value it implies. That is, !predicate || node_bit
  // evaluates to true for both node_bit == 1 and == 0.
  if (bdd_predicate_bit == bdd().zero()) {
    return std::nullopt;
  }

  enum class ImpliedValue : std::uint8_t {
    kNotAnalyzable,
    kUnknown,
    kImpliedTrue,
    kImpliedFalse
  };
  auto implied_value = [&](int node_idx) -> std::optional<TernaryValue> {
    std::optional<BddNodeIndex> bdd_node_bit =
        GetBddNode(TreeBitLocation(node, node_idx));
    if (!bdd_node_bit.has_value()) {
      return std::nullopt;
    }
    if (Implies(bdd_predicate_bit, *bdd_node_bit)) {
      return TernaryValue::kKnownOne;
    }
    if (Implies(bdd_predicate_bit, bdd().Not(*bdd_node_bit))) {
      return TernaryValue::kKnownZero;
    }
    return TernaryValue::kUnknown;
  };

  // Check if bdd_predicate_bit implies anything about the node.
  CHECK(node->GetType()->IsBits());
  TernaryVector ternary(node->BitCountOrDie(), TernaryValue::kUnknown);
  for (int node_idx = 0; node_idx < node->BitCountOrDie(); ++node_idx) {
    std::optional<TernaryValue> bit_value = implied_value(node_idx);
    if (!bit_value.has_value()) {
      // Missing information; we can't determine the node value.
      return std::nullopt;
    }
    ternary[node_idx] = *bit_value;
  }
  if (ternary_ops::AllUnknown(ternary)) {
    return std::nullopt;
  }
  return ternary;
}

bool BddQueryEngine::KnownEquals(const TreeBitLocation& a,
                                 const TreeBitLocation& b) const {
  if (!IsTracked(a.node()) || !IsTracked(b.node())) {
    return false;
  }
  std::optional<BddNodeIndex> a_bdd = GetBddNode(a);
  if (!a_bdd.has_value()) {
    return false;
  }
  std::optional<BddNodeIndex> b_bdd = GetBddNode(b);
  if (!b_bdd.has_value()) {
    return false;
  }
  return *a_bdd == *b_bdd;
}

bool BddQueryEngine::KnownNotEquals(const TreeBitLocation& a,
                                    const TreeBitLocation& b) const {
  if (!IsTracked(a.node()) || !IsTracked(b.node())) {
    return false;
  }
  std::optional<BddNodeIndex> a_bdd = GetBddNode(a);
  if (!a_bdd.has_value()) {
    return false;
  }
  std::optional<BddNodeIndex> b_bdd = GetBddNode(b);
  if (!b_bdd.has_value()) {
    return false;
  }
  return *a_bdd == bdd().Not(*b_bdd);
}

}  // namespace xls
