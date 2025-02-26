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

#include "xls/passes/cse_pass.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/topo_sort.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"

namespace xls {

namespace {

// Returns the operands of the given node for the purposes of the CSE
// optimization. The order of the nodes may not match the order of the node's
// actual operands. Motivation: generally for nodes to be considered equivalent
// the operands must be in the same order. However, commutative operations are
// agnostic to operand order. So to expand the CSE optimization, compare
// operands as an unordered set for commutative operands. This is done be
// ordered commutative operation operands by id prior to comparison. To avoid
// having to construct a vector every time, a span is returned by this
// function. If the operation is not commutative, the node's own operand span is
// simply returned. For the commutative case, a vector of sorted operands is
// constructed in span_backing_store from which a span is constructed.
absl::Span<Node* const> GetOperandsForCse(
    Node* node, std::vector<Node*>* span_backing_store) {
  CHECK(span_backing_store->empty());
  if (!OpIsCommutative(node->op())) {
    return node->operands();
  }
  span_backing_store->insert(span_backing_store->begin(),
                             node->operands().begin(), node->operands().end());
  SortByNodeId(span_backing_store);
  return *span_backing_store;
}

}  // namespace

absl::StatusOr<bool> RunCse(FunctionBase* f,
                            absl::flat_hash_map<Node*, Node*>* replacements,
                            bool common_literals) {
  // To improve efficiency, bucket potentially common nodes together. The
  // bucketing is done via an int64_t hash value which is constructed from the
  // op() of the node and the uid's of the node's operands.
  auto hasher = absl::Hash<std::vector<int64_t>>();
  auto node_hash = [&](Node* n) {
    std::vector<int64_t> values_to_hash = {static_cast<int64_t>(n->op())};
    std::vector<Node*> span_backing_store;
    for (Node* operand : GetOperandsForCse(n, &span_backing_store)) {
      values_to_hash.push_back(operand->id());
    }
    // If this is slow because of many literals, the Literal values could be
    // combined into the hash. As is, all literals get the same hash value.
    return hasher(values_to_hash);
  };

  bool changed = false;
  absl::flat_hash_map<int64_t, std::vector<Node*>> node_buckets;
  node_buckets.reserve(f->node_count());
  for (Node* node : TopoSort(f)) {
    if (OpIsSideEffecting(node->op())) {
      continue;
    }

    if (node->Is<Literal>() && !common_literals) {
      continue;
    }

    // Normally, dead nodes are removed by the DCE pass. However, if the node is
    // (e.g.) an invoke, DCE won't touch it, waiting for inlining to remove
    // it... and if we try to replace it, we'll think we changed the IR when we
    // actually didn't.
    if (node->IsDead()) {
      continue;
    }

    int64_t hash = node_hash(node);
    if (!node_buckets.contains(hash)) {
      node_buckets[hash].push_back(node);
      continue;
    }
    bool replaced = false;
    std::vector<Node*> node_span_backing_store;
    absl::Span<Node* const> node_operands_for_cse =
        GetOperandsForCse(node, &node_span_backing_store);
    for (Node* candidate : node_buckets.at(hash)) {
      std::vector<Node*> candidate_span_backing_store;
      if (node_operands_for_cse ==
              GetOperandsForCse(candidate, &candidate_span_backing_store) &&
          node->IsDefinitelyEqualTo(candidate)) {
        VLOG(3) << absl::StreamFormat("Replacing %s with equivalent node %s",
                                      node->GetName(), candidate->GetName());
        XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(candidate));
        if (replacements != nullptr) {
          (*replacements)[node] = candidate;
        }
        changed = true;
        replaced = true;
        break;
      }
    }
    if (!replaced) {
      node_buckets[hash].push_back(node);
    }
  }

  return changed;
}

absl::StatusOr<bool> CsePass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext* context) const {
  return RunCse(f, nullptr, common_literals_);
}

REGISTER_OPT_PASS(CsePass);

}  // namespace xls
