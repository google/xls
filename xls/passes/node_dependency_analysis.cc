// Copyright 2023 The XLS Authors
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

#include "xls/passes/node_dependency_analysis.h"

#include <cstdint>
#include <tuple>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/topo_sort.h"

namespace xls {

namespace {

// Perform actual analysis.
// f is the function to analyze. We only care about getting results for
// 'interesting_nodes' if the set is non-empty (otherwise all nodes are
// searched). Preds returns the nodes the argument depends on. iter is the
// iterator to walk the function in the topological order defined by preds.
template <typename Predecessors>
std::tuple<absl::flat_hash_map<Node*, InlineBitmap>,
           absl::flat_hash_map<Node*, int64_t>>
AnalyzeDependents(FunctionBase* f,
                  const absl::flat_hash_set<Node*>& interesting_nodes,
                  Predecessors preds, absl::Span<Node* const> topo_sort) {
  absl::flat_hash_map<Node*, int64_t> node_ids;
  node_ids.reserve(f->node_count());
  int64_t cnt = 0;
  for (Node* n : f->nodes()) {
    node_ids[n] = cnt++;
  }
  auto is_interesting = [&](Node* n) {
    return interesting_nodes.empty() || interesting_nodes.contains(n);
  };
  int64_t seen_interesting_nodes_count = 0;
  auto is_last_interesting_node = [&](Node* node) {
    if (interesting_nodes.empty()) {
      return false;
    }
    if (interesting_nodes.contains(node)) {
      ++seen_interesting_nodes_count;
    }
    return seen_interesting_nodes_count == interesting_nodes.size();
  };
  int64_t bitmap_size = f->node_count();
  absl::flat_hash_map<Node*, InlineBitmap> results;
  results.reserve(f->node_count());
  for (Node* n : topo_sort) {
    InlineBitmap& bm = results.emplace(n, bitmap_size).first->second;
    bm.Set(node_ids[n]);
    for (Node* pred : preds(n)) {
      bm.Union(results.at(pred));
    }
    if (is_last_interesting_node(n)) {
      break;
    }
  }
  // To avoid any bugs delete everything that's not specifically requested.
  absl::erase_if(results,
                 [&](auto& pair) { return !is_interesting(pair.first); });
  return {results, node_ids};
}

}  // namespace

absl::StatusOr<DependencyBitmap> NodeDependencyAnalysis::GetDependents(
    Node* node) const {
  if (!IsAnalyzed(node)) {
    return absl::InvalidArgumentError("Node is not analyzed");
  }
  return DependencyBitmap(dependents_.at(node), node_indices_);
}

NodeDependencyAnalysis NodeDependencyAnalysis::BackwardDependents(
    FunctionBase* fb, absl::Span<Node* const> nodes) {
  absl::flat_hash_set<Node*> interesting(nodes.begin(), nodes.end());
  auto [dependents, node_ids] = AnalyzeDependents(
      fb, interesting, [](Node* node) { return node->operands(); },
      TopoSort(fb));
  return NodeDependencyAnalysis(/*is_forwards=*/false, dependents, node_ids);
}

NodeDependencyAnalysis NodeDependencyAnalysis::ForwardDependents(
    FunctionBase* fb, absl::Span<Node* const> nodes) {
  absl::flat_hash_set<Node*> interesting(nodes.begin(), nodes.end());
  auto [dependents, node_ids] = AnalyzeDependents(
      fb, interesting, [](Node* node) { return node->users(); },
      ReverseTopoSort(fb));
  return NodeDependencyAnalysis(/*is_forwards=*/true, dependents, node_ids);
}

}  // namespace xls
