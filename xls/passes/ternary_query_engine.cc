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

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/abstract_node_evaluator.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/ternary.h"
#include "xls/ir/topo_sort.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/ternary_evaluator.h"

namespace xls {
namespace {
// Returns whether the operation will be computationally expensive to
// compute. The ternary query engine is intended to be fast so the analysis of
// these expensive operations is skipped with the effect being all bits are
// considered unknown. Operations and limits can be added as needed when
// pathological cases are encountered.
bool IsExpensiveToEvaluate(Node* node) {
  // Shifts are quadratic in the width of the operand so wide shifts are very
  // slow to evaluate in the abstract evaluator.
  return (node->op() == Op::kShrl || node->op() == Op::kShra ||
          node->op() == Op::kShll || node->op() == Op::kBitSliceUpdate) &&
         node->GetType()->GetFlatBitCount() > 256;
}

class TernaryNodeEvaluator : public AbstractNodeEvaluator<TernaryEvaluator> {
 public:
  using AbstractNodeEvaluator<TernaryEvaluator>::AbstractNodeEvaluator;
  absl::Status DefaultHandler(Node* n) override {
    // We just consider any unhandled value entirely unconstrained.
    return SetValue(n, TernaryEvaluator::Vector(n->BitCountOrDie(),
                                                TernaryValue::kUnknown));
  }
};

}  // namespace

absl::StatusOr<ReachedFixpoint> TernaryQueryEngine::Populate(FunctionBase* f) {
  TernaryEvaluator evaluator;
  TernaryNodeEvaluator ternary_visitor(evaluator);
  for (Node* n : TopoSort(f)) {
    if (!n->GetType()->IsBits()) {
      continue;
    }
    if (IsExpensiveToEvaluate(n) ||
        std::any_of(n->operands().begin(), n->operands().end(),
                    [](Node* o) { return !o->GetType()->IsBits(); })) {
      XLS_RETURN_IF_ERROR(ternary_visitor.DefaultHandler(n));
      continue;
    }
    XLS_RETURN_IF_ERROR(n->VisitSingleNode(&ternary_visitor));
  }

  const auto& values = ternary_visitor.values();
  ReachedFixpoint rf = ReachedFixpoint::Unchanged;
  for (Node* node : f->nodes()) {
    // TODO(meheff): Handle types other than bits.
    if (node->GetType()->IsBits()) {
      if (!known_bits_.contains(node)) {
        known_bits_[node] = Bits(values.at(node).size());
        bits_values_[node] = Bits(values.at(node).size());
      }
      Bits combined_known_bits = bits_ops::Or(
          known_bits_[node], ternary_ops::ToKnownBits(values.at(node)));
      Bits combined_bits_values = bits_ops::Or(
          bits_values_[node], ternary_ops::ToKnownBitsValues(values.at(node)));
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
