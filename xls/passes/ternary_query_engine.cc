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

#include "xls/passes/ternary_query_engine.h"

#include <limits>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/abstract_node_evaluator.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/node_iterator.h"
#include "xls/passes/ternary_evaluator.h"

namespace xls {

// Returns whether the operation will be computationally expensive to
// compute. The ternary query engine is intended to be fast so the analysis of
// these expensive operations is skipped with the effect being all bits are
// considered unknown. Operations and limits can be added as needed when
// pathological cases are encountered.
static bool IsExpensiveToEvaluate(Node* node) {
  // Shifts are quadratic in the width of the operand so wide shifts are very
  // slow to evaluate in the abstract evaluator.
  return (node->op() == Op::kShrl || node->op() == Op::kShra ||
          node->op() == Op::kShll) &&
         node->GetType()->GetFlatBitCount() > 256;
}

Bits TernaryVectorToKnownBits(const TernaryEvaluator::Vector& ternary_vector) {
  // Use InlinedVector to avoid std::vector<bool> specialization madness.
  absl::InlinedVector<bool, 1> bits(ternary_vector.size());
  for (int64_t i = 0; i < bits.size(); ++i) {
    bits[i] = (ternary_vector[i] == TernaryValue::kKnownOne ||
               ternary_vector[i] == TernaryValue::kKnownZero);
  }
  return Bits(bits);
}

Bits TernaryVectorToValueBits(const TernaryEvaluator::Vector& ternary_vector) {
  // Use InlinedVector to avoid std::vector<bool> specialization madness.
  absl::InlinedVector<bool, 1> bits(ternary_vector.size());
  for (int64_t i = 0; i < bits.size(); ++i) {
    bits[i] = ternary_vector[i] == TernaryValue::kKnownOne;
  }
  return Bits(bits);
}

absl::StatusOr<ReachedFixpoint> TernaryQueryEngine::Populate(FunctionBase* f) {
  TernaryEvaluator evaluator;
  absl::flat_hash_map<Node*, TernaryEvaluator::Vector> values;
  for (Node* node : TopoSort(f)) {
    if (!node->GetType()->IsBits()) {
      continue;
    }
    auto create_unknown_vector = [](Node* n) {
      return TernaryEvaluator::Vector(n->BitCountOrDie(),
                                      TernaryValue::kUnknown);
    };
    if (IsExpensiveToEvaluate(node) ||
        std::any_of(node->operands().begin(), node->operands().end(),
                    [](Node* o) { return !o->GetType()->IsBits(); })) {
      values[node] = create_unknown_vector(node);
      continue;
    }

    std::vector<TernaryEvaluator::Vector> operand_values;
    for (Node* operand : node->operands()) {
      operand_values.push_back(values.at(operand));
    }
    XLS_ASSIGN_OR_RETURN(
        values[node],
        AbstractEvaluate(node, operand_values, &evaluator,
                         /*default_handler=*/create_unknown_vector));
  }

  ReachedFixpoint rf = ReachedFixpoint::Unchanged;
  for (Node* node : f->nodes()) {
    // TODO(meheff): Handle types other than bits.
    if (node->GetType()->IsBits()) {
      if (!known_bits_.contains(node)) {
        known_bits_[node] = Bits(values.at(node).size());
        bits_values_[node] = Bits(values.at(node).size());
      }
      Bits combined_known_bits = bits_ops::Or(
          known_bits_[node], TernaryVectorToKnownBits(values.at(node)));
      Bits combined_bits_values = bits_ops::Or(
          bits_values_[node], TernaryVectorToValueBits(values.at(node)));
      if ((combined_known_bits != known_bits_[node]) ||
          (combined_bits_values != bits_values_[node])) {
        rf = ReachedFixpoint::Changed;
      }
      known_bits_[node] = combined_known_bits;
      bits_values_[node] = combined_bits_values;
    }
  }
  return rf;
}

bool TernaryQueryEngine::AtMostOneTrue(
    absl::Span<TreeBitLocation const> bits) const {
  int64_t maybe_one_count = 0;
  for (const TreeBitLocation& location : bits) {
    if (!IsKnown(location) || IsOne(location)) {
      maybe_one_count++;
    }
  }
  return maybe_one_count <= 1;
}

bool TernaryQueryEngine::AtLeastOneTrue(
    absl::Span<TreeBitLocation const> bits) const {
  for (const TreeBitLocation& location : bits) {
    if (IsOne(location)) {
      return true;
    }
  }
  return false;
}

bool TernaryQueryEngine::KnownEquals(const TreeBitLocation& a,
                                     const TreeBitLocation& b) const {
  return IsKnown(a) && IsKnown(b) && IsOne(a) == IsOne(b);
}

bool TernaryQueryEngine::KnownNotEquals(const TreeBitLocation& a,
                                        const TreeBitLocation& b) const {
  return IsKnown(a) && IsKnown(b) && IsOne(a) != IsOne(b);
}

}  // namespace xls
