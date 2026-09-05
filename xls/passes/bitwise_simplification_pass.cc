// Copyright 2026 The XLS Authors
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

#include "xls/passes/bitwise_simplification_pass.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "cppitertools/zip.hpp"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/ternary.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/partial_info_query_engine.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {
namespace {

// Simplifies an AND or OR operation using known bits across all operands.
//
// Full collapse to a known value is always performed even when splits are
// disabled. When splits are enabled, the operation is decomposed into concats
// of literals and bit-slices of non-constant operands.
absl::StatusOr<bool> SimplifyBitwiseLogic(Node* n,
                                          const QueryEngine& query_engine,
                                          bool splits_enabled) {
  if (n->IsDead()) {
    return false;
  }
  if (n->op() != Op::kAnd && n->op() != Op::kOr) {
    return false;
  }
  if (!n->GetType()->IsBits()) {
    return false;
  }
  if (std::optional<Bits> known_value = query_engine.KnownValueAsBits(n);
      known_value.has_value()) {
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(Value(*known_value)).status());
    return true;
  }

  const int64_t bit_count = n->BitCountOrDie();
  const bool is_and = (n->op() == Op::kAnd);

  // Accumulate a mask of known bits, remapping unknown bits to the identity
  // value (1 for AND, 0 for OR), while recording the non-constant operands.
  InlineBitmap mask(bit_count, /*fill=*/is_and);
  std::vector<Node*> non_constant_operands;
  std::vector<std::optional<SharedLeafTypeTree<TernaryVector>>>
      non_constant_ternaries;
  non_constant_operands.reserve(n->operand_count());
  non_constant_ternaries.reserve(n->operand_count());
  for (Node* op : n->operands()) {
    std::optional<SharedLeafTypeTree<TernaryVector>> ternary =
        query_engine.GetTernary(op);
    if (ternary.has_value()) {
      Bits operand_as_mask = ternary_ops::ToKnownBitsValues(
          ternary->Get({}), /*default_set=*/is_and);
      if (is_and) {
        mask.Intersect(operand_as_mask.bitmap());
      } else {
        mask.Union(operand_as_mask.bitmap());
      }
    }
    if (!ternary.has_value() || !ternary_ops::IsFullyKnown(ternary->Get({}))) {
      non_constant_operands.push_back(op);
      non_constant_ternaries.push_back(std::move(ternary));
    }
  }

  // If the composite mask is the identity, no bits are forced.
  if (is_and ? mask.IsAllOnes() : mask.IsAllZeroes()) {
    return false;
  }

  // If splits are not enabled, do not decompose into slices.
  if (!splits_enabled) {
    return false;
  }

  // If there are no non-constant operands, this should have already been
  // handled above as a fully-known value - and failing that, it'll be handled
  // by strength reduction & constant folding.
  if (non_constant_operands.empty()) {
    return false;
  }

  FunctionBase* f = n->function_base();
  std::vector<Node*> slices;
  int64_t pos = 0;
  while (pos < bit_count) {
    const bool mask_val = mask.Get(pos);
    const bool is_absorbing = is_and ^ mask_val;
    int64_t end = pos + 1;
    while (end < bit_count && mask.Get(end) == mask_val) {
      ++end;
    }

    if (is_absorbing) {
      Bits const_bits = is_and ? UBits(0, end - pos) : Bits::AllOnes(end - pos);
      XLS_ASSIGN_OR_RETURN(
          Node * lit,
          f->MakeNode<Literal>(n->loc(), Value(std::move(const_bits))));
      slices.push_back(lit);
    } else {
      absl::InlinedVector<Node*, 1> sub_slices;
      sub_slices.reserve(non_constant_operands.size());
      for (const auto& [op, ternary] :
           iter::zip(non_constant_operands, non_constant_ternaries)) {
        if (ternary.has_value() &&
            ternary_ops::IsFullyKnown(absl::MakeConstSpan(ternary->Get({}))
                                          .subspan(pos, end - pos))) {
          continue;
        }
        if (pos == 0 && end == op->BitCountOrDie()) {
          sub_slices.push_back(op);
        } else {
          XLS_ASSIGN_OR_RETURN(Node * slice, f->MakeNode<BitSlice>(
                                                 n->loc(), op, pos, end - pos));
          sub_slices.push_back(slice);
        }
      }
      if (sub_slices.empty()) {
        // All the non-constant operands are the identity on this slice.
        Bits const_bits =
            is_and ? Bits::AllOnes(end - pos) : UBits(0, end - pos);
        XLS_ASSIGN_OR_RETURN(
            Node * const_slice,
            f->MakeNode<Literal>(n->loc(), Value(std::move(const_bits))));
        slices.push_back(const_slice);
      } else if (sub_slices.size() == 1) {
        slices.push_back(sub_slices.front());
      } else {
        XLS_ASSIGN_OR_RETURN(
            Node * sub_op, f->MakeNode<NaryOp>(n->loc(), sub_slices, n->op()));
        slices.push_back(sub_op);
      }
    }

    pos = end;
  }

  if (slices.size() == 1) {
    XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(slices[0]));
  } else {
    // Concat operands are ordered from MSB to LSB.
    absl::c_reverse(slices);
    XLS_RETURN_IF_ERROR(n->ReplaceUsesWithNew<Concat>(slices).status());
  }
  return true;
}

}  // namespace

RedundancyGuard BitwiseSimplificationPass::GetRedundancyGuard(
    const OptimizationPassOptions& options,
    OptimizationContext& context) const {
  return RedundancyGuard::CanSkip(options.splits_enabled() ? "splitting"
                                                           : "absorbing");
}

absl::StatusOr<bool> BitwiseSimplificationPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  UnionQueryEngine query_engine = UnionQueryEngine::Of(
      StatelessQueryEngine(),
      context.SharedQueryEngine<PartialInfoQueryEngine>(f));
  XLS_RETURN_IF_ERROR(query_engine.Populate(f).status());

  bool changed = false;
  XLS_ASSIGN_OR_RETURN(std::vector<Node*> topo_sort_nodes, context.TopoSort(f));
  for (Node* node : topo_sort_nodes) {
    XLS_ASSIGN_OR_RETURN(
        bool node_changed,
        SimplifyBitwiseLogic(node, query_engine, options.splits_enabled()));
    changed |= node_changed;
  }
  return changed;
}

}  // namespace xls
