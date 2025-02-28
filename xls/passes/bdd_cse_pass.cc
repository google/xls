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

#include "xls/passes/bdd_cse_pass.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/passes/bdd_function.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"

namespace xls {

namespace {

// Returns the order in which to visit the nodes when performing the
// optimization. If a pair of equivalent nodes is found during the optimization
// then the earlier visited node replaces the later visited node so this order
// is constructed with the following properties:
//
// (1) Order is a topological sort. This is necessary to avoid introducing
//     cycles in the graph.
//
// (2) Critical-path delay through the graph to the node increases monotonically
//     in the list. This ensures that the CSE replacement does not increase
//     critical-path
//
absl::StatusOr<std::vector<Node*>> GetNodeOrder(FunctionBase* f,
                                                OptimizationContext& context) {
  // Index of each node in the topological sort.
  absl::flat_hash_map<Node*, int64_t> topo_index;
  // Critical-path distance from root in the graph to each node.
  absl::flat_hash_map<Node*, int64_t> node_cp_delay;
  int64_t i = 0;

  // Return an estimate of the delay of the given node. Because BDD-CSE may be
  // run at any point in the pipeline, some nodes with no delay model may be
  // present (these would be eliminated before codegen) so return zero for these
  // cases.
  // TODO(meheff): Replace with the actual model being used when the delay model
  // is threaded through the pass pipeline.
  auto get_node_delay = [&](Node* n) {
    absl::StatusOr<int64_t> delay_status =
        GetStandardDelayEstimator().GetOperationDelayInPs(n);
    return delay_status.ok() ? delay_status.value() : 0;
  };
  for (Node* node : context.TopoSort(f)) {
    topo_index[node] = i;
    int64_t node_start = 0;
    for (Node* operand : node->operands()) {
      node_start = std::max(
          node_start, node_cp_delay.at(operand) + get_node_delay(operand));
    }
    node_cp_delay[node] = node_start + get_node_delay(node);
    ++i;
  }
  std::vector<Node*> nodes(f->nodes().begin(), f->nodes().end());
  std::sort(nodes.begin(), nodes.end(), [&](Node* a, Node* b) {
    return (node_cp_delay.at(a) < node_cp_delay.at(b) ||
            (node_cp_delay.at(a) == node_cp_delay.at(b) &&
             topo_index.at(a) < topo_index.at(b)));
  });
  // The node order must be a topological sort in order to avoid introducing
  // cycles in the graph.
  for (Node* node : nodes) {
    for (Node* operand : node->operands()) {
      XLS_RET_CHECK(topo_index.at(operand) < topo_index.at(node));
    }
  }
  return nodes;
}

}  // namespace

absl::StatusOr<bool> BddCsePass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<BddFunction> bdd_function,
      BddFunction::Run(f, BddFunction::kDefaultPathLimit, IsCheapForBdds));

  // To improve efficiency, bucket potentially common nodes together. The
  // bucketing is done via a int64_t hash value of the BDD node indices of each
  // bit of the node.
  auto hasher = absl::Hash<std::vector<int64_t>>();
  auto node_hash = [&](Node* n) {
    CHECK(n->GetType()->IsBits());
    std::vector<int64_t> values_to_hash;
    values_to_hash.reserve(n->BitCountOrDie());
    for (int64_t i = 0; i < n->BitCountOrDie(); ++i) {
      values_to_hash.push_back(bdd_function->GetBddNode(n, i).value());
    }
    return hasher(values_to_hash);
  };

  auto is_same_value = [&](Node* a, Node* b) {
    if (a->BitCountOrDie() != b->BitCountOrDie()) {
      return false;
    }
    for (int64_t i = 0; i < a->BitCountOrDie(); ++i) {
      if (bdd_function->GetBddNode(a, i) != bdd_function->GetBddNode(b, i)) {
        return false;
      }
    }
    return true;
  };

  bool changed = false;
  absl::flat_hash_map<int64_t, std::vector<Node*>> node_buckets;
  node_buckets.reserve(f->node_count());
  XLS_ASSIGN_OR_RETURN(std::vector<Node*> node_order, GetNodeOrder(f, context));
  for (Node* node : node_order) {
    if (!node->GetType()->IsBits() || node->Is<Literal>()) {
      continue;
    }

    int64_t hash = node_hash(node);
    if (!node_buckets.contains(hash)) {
      node_buckets[hash].push_back(node);
      continue;
    }
    bool replaced = false;
    for (Node* candidate : node_buckets.at(hash)) {
      if (is_same_value(node, candidate)) {
        XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(candidate));
        VLOG(4) << "Found identical value:";
        VLOG(4) << "  Node: " << node->ToString();
        VLOG(4) << "  Replacement: " << candidate->ToString();
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

REGISTER_OPT_PASS(BddCsePass);

}  // namespace xls
