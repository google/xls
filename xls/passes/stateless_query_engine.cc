// Copyright 2024 The XLS Authors
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

#include "xls/passes/stateless_query_engine.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "cppitertools/reversed.hpp"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/query_engine.h"

namespace xls {

std::optional<bool> StatelessQueryEngine::KnownValue(
    const TreeBitLocation& bit) const {
  if (!bit.node()->Is<Literal>()) {
    return std::nullopt;
  }
  auto value_tree = ValueToLeafTypeTree(bit.node()->As<Literal>()->value(),
                                        bit.node()->GetType());
  if (!value_tree.ok()) {
    return std::nullopt;
  }
  const Value& value = value_tree->Get(bit.tree_index());
  CHECK(value.IsBits());
  CHECK_GT(value.bits().bit_count(), bit.bit_index());
  return value.bits().Get(bit.bit_index());
}
std::optional<Value> StatelessQueryEngine::KnownValue(Node* node) const {
  if (node->Is<Literal>()) {
    return node->As<Literal>()->value();
  }

  return std::nullopt;
}

bool StatelessQueryEngine::IsAllZeros(Node* node) const {
  if (node->Is<Literal>()) {
    return node->As<Literal>()->value().IsAllZeros();
  }
  return false;
}
bool StatelessQueryEngine::IsAllOnes(Node* node) const {
  if (node->Is<Literal>()) {
    return node->As<Literal>()->value().IsAllOnes();
  }
  return false;
}

std::optional<SharedLeafTypeTree<TernaryVector>>
StatelessQueryEngine::GetTernary(Node* node) const {
  if (node->Is<Literal>()) {
    LeafTypeTree<Value> values =
        ValueToLeafTypeTree(node->As<Literal>()->value(), node->GetType())
            .value();
    return leaf_type_tree::Map<TernaryVector, Value>(
               values.AsView(),
               [](const Value& v) -> TernaryVector {
                 if (v.IsToken()) {
                   return TernaryVector();
                 }
                 return ternary_ops::BitsToTernary(v.bits());
               })
        .AsShared();
  }

  if (node->op() == Op::kConcat) {
    TernaryVector vec(node->BitCountOrDie(), TernaryValue::kUnknown);
    auto it = vec.begin();
    for (Node* operand : iter::reversed(node->operands())) {
      std::optional<SharedLeafTypeTree<TernaryVector>> ternary =
          GetTernary(operand);
      if (!ternary.has_value()) {
        it += operand->BitCountOrDie();
        continue;
      }
      it = absl::c_copy(ternary->Get({}), it);
    }
    return LeafTypeTree<TernaryVector>::CreateSingleElementTree(node->GetType(),
                                                                std::move(vec))
        .AsShared();
  }

  if (node->op() == Op::kZeroExt) {
    CHECK(node->GetType()->IsBits());
    TernaryVector ternary;
    if (auto ternary_tree = GetTernary(node->operand(0))) {
      ternary = ternary_tree->Get({});
    } else {
      ternary = TernaryVector(node->operand(0)->BitCountOrDie(),
                              TernaryValue::kUnknown);
    }
    ternary.resize(node->BitCountOrDie(), TernaryValue::kKnownZero);
    return LeafTypeTree<TernaryVector>::CreateSingleElementTree(
               node->GetType(), std::move(ternary))
        .AsShared();
  }

  if (node->op() == Op::kSignExt) {
    if (node->operand(0)->BitCountOrDie() == 0) {
      // Zero-len value has unset sign bit.
      return LeafTypeTree<TernaryVector>::CreateSingleElementTree(
                 node->GetType(),
                 TernaryVector(node->BitCountOrDie(), TernaryValue::kKnownZero))
          .AsShared();
    }
    TernaryVector ternary;
    if (auto ternary_tree = GetTernary(node->operand(0))) {
      ternary = ternary_tree->Get({});
    } else {
      ternary = TernaryVector(node->operand(0)->BitCountOrDie(),
                              TernaryValue::kUnknown);
    }
    TernaryValue sign = ternary.back();
    ternary.resize(node->BitCountOrDie(), sign);
    return LeafTypeTree<TernaryVector>::CreateSingleElementTree(
               node->GetType(), std::move(ternary))
        .AsShared();
  }

  return std::nullopt;
}

StatelessQueryEngine::BitCounts StatelessQueryEngine::KnownBitCounts(
    absl::Span<TreeBitLocation const> bits) const {
  absl::flat_hash_map<Node*,
                      absl::flat_hash_map<TreeBitLocation, /*count=*/int64_t>>
      by_node;
  for (const TreeBitLocation& bit : bits) {
    by_node[bit.node()][bit]++;
  }

  BitCounts counts;
  for (const auto& [node, node_bits] : by_node) {
    if (node->Is<OneHot>()) {
      int64_t total_count = 0;
      int64_t min_count = std::numeric_limits<int64_t>::max();
      int64_t max_count = 0;
      for (const auto& [bit, count] : node_bits) {
        total_count += count;
        min_count = std::min(min_count, count);
        max_count = std::max(max_count, count);
      }
      // Exactly one output bit from this node is enabled, so all but one is
      // false; in the worst case, it's the one named the most times.
      counts.known_false += total_count - max_count;
      if (node_bits.size() == node->BitCountOrDie()) {
        // If every bit from this node is named, then one of the named bits is
        // true; in the worst case, it's the one named the fewest times.
        counts.known_true += min_count;
      }
      continue;
    }

    std::optional<SharedLeafTypeTree<TernaryVector>> ternary = GetTernary(node);
    if (!ternary.has_value()) {
      continue;
    }
    for (const auto& [bit, count] : node_bits) {
      switch (ternary->Get(bit.tree_index()).at(bit.bit_index())) {
        case TernaryValue::kKnownZero:
          counts.known_false += count;
          break;
        case TernaryValue::kKnownOne:
          counts.known_true += count;
          break;
        case TernaryValue::kUnknown:
          break;
      }
    }
  }
  return counts;
}

bool StatelessQueryEngine::AtMostOneTrue(
    absl::Span<TreeBitLocation const> bits) const {
  return bits.size() <= 1 ||
         KnownBitCounts(bits).known_false >= bits.size() - 1;
}

bool StatelessQueryEngine::AtLeastOneTrue(
    absl::Span<TreeBitLocation const> bits) const {
  return !bits.empty() && KnownBitCounts(bits).known_true >= 1;
}

bool StatelessQueryEngine::KnownEquals(const TreeBitLocation& a,
                                       const TreeBitLocation& b) const {
  if (a == b) {
    return true;
  }

  if (a.node() == b.node() && a.node()->Is<OneHot>()) {
    // No two distinct bits from a OneHot node can be equal.
    return false;
  }

  std::optional<bool> a_value = QueryEngine::KnownValue(a);
  if (!a_value.has_value()) {
    return false;
  }

  std::optional<bool> b_value = QueryEngine::KnownValue(b);
  if (!b_value.has_value()) {
    return false;
  }

  return a_value == b_value;
}

bool StatelessQueryEngine::KnownNotEquals(const TreeBitLocation& a,
                                          const TreeBitLocation& b) const {
  if (a == b) {
    return false;
  }

  if (a.node() == b.node() && a.node()->Is<OneHot>()) {
    // No two distinct bits from a OneHot node can be equal.
    return true;
  }

  std::optional<bool> a_value = QueryEngine::KnownValue(a);
  if (!a_value.has_value()) {
    return false;
  }

  std::optional<bool> b_value = QueryEngine::KnownValue(b);
  if (!b_value.has_value()) {
    return false;
  }

  return a_value != b_value;
}

bool StatelessQueryEngine::AtMostOneBitTrue(Node* node) const {
  if (node->Is<OneHot>()) {
    return true;
  }

  return QueryEngine::AtMostOneBitTrue(node);
}
bool StatelessQueryEngine::AtLeastOneBitTrue(Node* node) const {
  if (node->Is<OneHot>()) {
    return true;
  }

  return QueryEngine::AtLeastOneBitTrue(node);
}
bool StatelessQueryEngine::ExactlyOneBitTrue(Node* node) const {
  if (node->Is<OneHot>()) {
    return true;
  }

  return QueryEngine::ExactlyOneBitTrue(node);
}

bool StatelessQueryEngine::Implies(const TreeBitLocation& a,
                                   const TreeBitLocation& b) const {
  return a == b || IsZero(a) || IsOne(b);
}

bool StatelessQueryEngine::IsFullyKnown(Node* n) const {
  return n->Is<Literal>();
}

std::optional<int64_t> StatelessQueryEngine::KnownLeadingSignBits(
    Node* node) const {
  if (!node->GetType()->IsBits()) {
    return std::nullopt;
  }
  int64_t lead_zero = KnownLeadingZeros(node).value_or(0);
  int64_t lead_one = KnownLeadingOnes(node).value_or(0);
  int64_t lead_sign_ext =
      // NB The top bit of the operand is also equal to the sign bit.
      node->op() == Op::kSignExt
          ? 1 + node->BitCountOrDie() - node->operand(0)->BitCountOrDie()
          : 0;
  return std::max({lead_zero, lead_one, lead_sign_ext});
}

}  // namespace xls
