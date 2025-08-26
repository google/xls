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
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/critical_path_delay_analysis.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"

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
absl::StatusOr<std::vector<Node*>> GetNodeOrder(
    FunctionBase* f, OptimizationContext& context,
    const OptimizationPassOptions& options) {
  CriticalPathDelayAnalysis& critical_path_analysis =
      *ABSL_DIE_IF_NULL(context.SharedNodeData<CriticalPathDelayAnalysis>(
          f, {.delay_model_name = options.delay_model}));

  // Index of each node in the topological sort.
  absl::flat_hash_map<Node*, int64_t> topo_index;
  int64_t i = 0;
  for (Node* node : context.TopoSort(f)) {
    topo_index[node] = i++;
  }
  std::vector<Node*> nodes(f->nodes().begin(), f->nodes().end());
  auto sort_key = [&](Node* n) {
    std::optional<int64_t> delay = critical_path_analysis.GetDelay(n);
    return std::make_pair(delay.value_or(0), topo_index.at(n));
  };
  std::sort(nodes.begin(), nodes.end(),
            [&](Node* a, Node* b) { return sort_key(a) < sort_key(b); });
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
  BddQueryEngine* query_engine = context.SharedQueryEngine<BddQueryEngine>(f);
  auto get_bdd_node = [&](Node* n, int64_t bit_index) -> int64_t {
    return query_engine->GetBddNode(TreeBitLocation(n, bit_index))->value();
  };

  // To improve efficiency, bucket potentially common nodes together. The
  // bucketing is done via a int64_t hash value of the BDD node indices of each
  // bit of the node.
  auto hasher = absl::Hash<std::vector<int64_t>>();
  auto node_hash = [&](Node* n) {
    CHECK(n->GetType()->IsBits());
    std::vector<int64_t> values_to_hash;
    values_to_hash.reserve(n->BitCountOrDie());
    for (int64_t i = 0; i < n->BitCountOrDie(); ++i) {
      values_to_hash.push_back(get_bdd_node(n, i));
    }
    return hasher(values_to_hash);
  };

  auto is_same_value = [&](Node* a, Node* b) {
    if (a->BitCountOrDie() != b->BitCountOrDie()) {
      return false;
    }
    for (int64_t i = 0; i < a->BitCountOrDie(); ++i) {
      if (get_bdd_node(a, i) != get_bdd_node(b, i)) {
        return false;
      }
    }
    return true;
  };

  bool changed = false;
  absl::flat_hash_map<int64_t, std::vector<Node*>> node_buckets;
  node_buckets.reserve(f->node_count());
  XLS_ASSIGN_OR_RETURN(std::vector<Node*> node_order,
                       GetNodeOrder(f, context, options));
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

}  // namespace xls
