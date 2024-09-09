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

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
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

std::optional<Value> StatelessQueryEngine::KnownValue(Node* node) const {
  if (node->Is<Literal>()) {
    return node->As<Literal>()->value();
  }

  return std::nullopt;
}

LeafTypeTree<TernaryVector> StatelessQueryEngine::GetTernary(Node* node) const {
  if (node->Is<Literal>()) {
    LeafTypeTree<Value> values =
        ValueToLeafTypeTree(node->As<Literal>()->value(), node->GetType())
            .value();
    return leaf_type_tree::Map<TernaryVector, Value>(
        values.AsView(), [](const Value& v) -> TernaryVector {
          return ternary_ops::BitsToTernary(v.bits());
        });
  }

  if (node->op() == Op::kZeroExt) {
    CHECK(node->GetType()->IsBits());
    TernaryVector ternary(node->BitCountOrDie(), TernaryValue::kUnknown);
    for (int64_t i = node->operand(ExtendOp::kArgOperand)->BitCountOrDie();
         i < ternary.size(); ++i) {
      ternary[i] = TernaryValue::kKnownZero;
    }
    return LeafTypeTree<TernaryVector>::CreateSingleElementTree(
        node->GetType(), std::move(ternary));
  }

  return LeafTypeTree<TernaryVector>::CreateFromFunction(
             node->GetType(),
             [](Type* leaf_type,
                absl::Span<const int64_t>) -> absl::StatusOr<TernaryVector> {
               return TernaryVector(leaf_type->GetFlatBitCount(),
                                    TernaryValue::kUnknown);
             })
      .value();
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

    LeafTypeTree<TernaryVector> ternary = GetTernary(node);
    for (const auto& [bit, count] : node_bits) {
      switch (ternary.Get(bit.tree_index()).at(bit.bit_index())) {
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

  LeafTypeTree<TernaryVector> a_ternary = GetTernary(a.node());
  const TernaryValue& a_value = a_ternary.Get(a.tree_index()).at(a.bit_index());
  if (a_value == TernaryValue::kUnknown) {
    return false;
  }

  LeafTypeTree<TernaryVector> b_ternary =
      a.node() == b.node() ? a_ternary : GetTernary(b.node());
  const TernaryValue& b_value = b_ternary.Get(b.tree_index()).at(b.bit_index());

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

  LeafTypeTree<TernaryVector> a_ternary = GetTernary(a.node());
  const TernaryValue& a_value = a_ternary.Get(a.tree_index()).at(a.bit_index());
  if (a_value == TernaryValue::kUnknown) {
    return false;
  }

  LeafTypeTree<TernaryVector> b_ternary =
      a.node() == b.node() ? a_ternary : GetTernary(b.node());
  const TernaryValue& b_value = b_ternary.Get(b.tree_index()).at(b.bit_index());
  if (b_value == TernaryValue::kUnknown) {
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

}  // namespace xls
