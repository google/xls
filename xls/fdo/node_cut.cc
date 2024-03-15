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

#include "xls/fdo/node_cut.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/topo_sort.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {

NodeCut NodeCut::GetMergedCut(Node *root, const NodeCut &lhs,
                              const NodeCut &rhs) {
  absl::flat_hash_set<Node *> merged(lhs.leaves().begin(), lhs.leaves().end());
  merged.insert(rhs.leaves().begin(), rhs.leaves().end());
  return NodeCut(root, merged);
}

bool NodeCut::Includes(const NodeCut &other) const {
  return root_ == other.root_ &&
         std::all_of(other.leaves().begin(), other.leaves().end(),
                     [&](const Node *leaf) { return leaves_.contains(leaf); });
}

absl::flat_hash_set<Node *> NodeCut::GetNodeCone() const {
  absl::flat_hash_set<Node *> cone({root_});
  if (IsTrivial()) {
    return cone;
  }

  std::vector<Node *> worklist({root_});
  while (!worklist.empty()) {
    Node *current_node = worklist.back();
    worklist.pop_back();
    for (Node *operand : current_node->operands()) {
      // The current node is a leaf or has been traversed.
      if (cone.contains(operand) || leaves_.contains(operand)) {
        continue;
      }
      // Otherwise, add it to the cone and push back into the worklist.
      cone.emplace(operand);
      worklist.emplace_back(operand);
    }
  }
  return cone;
}

// Add a new cut into the cut set. This method ensures there are no duplicated
// or dominated cuts.
static absl::Status AddCut(const NodeCut &cut, std::vector<NodeCut> &cuts) {
  // Replace an existing cut with the incoming one if the existing cut is a
  // superset. Because a superset means the existing cut has redundant leaves.
  auto super_cut = std::find_if(
      cuts.begin(), cuts.end(),
      [&](const NodeCut &exist_cut) { return exist_cut.Includes(cut); });
  if (super_cut != cuts.end()) {
    *super_cut = cut;
    return absl::OkStatus();
  }

  // Do nothing and return if the existing cut is a subset of the incoming one.
  auto sub_cut = std::find_if(
      cuts.begin(), cuts.end(),
      [&](const NodeCut &exist_cut) { return cut.Includes(exist_cut); });
  if (sub_cut != cuts.end()) {
    return absl::OkStatus();
  }

  // Otherwise, add the upcoming cut.
  cuts.emplace_back(cut);
  return absl::OkStatus();
}

absl::StatusOr<NodeCutMap> EnumerateMaxCutInSchedule(
    FunctionBase *f, int64_t pipeline_length,
    const ScheduleCycleMap &cycle_map) {
  XLS_ASSIGN_OR_RETURN(NodeCutsMap cuts_map,
                       EnumerateCutsInSchedule(f, pipeline_length, cycle_map,
                                               /*input_leaves_only=*/true));
  NodeCutMap cut_map;
  for (const auto &[node, cuts] : cuts_map) {
    XLS_RET_CHECK(cuts.size() == 1);
    cut_map.emplace(node, cuts.front());
  }
  return cut_map;
}

absl::StatusOr<NodeCutsMap> EnumerateCutsInSchedule(
    FunctionBase *f, int64_t pipeline_length, const ScheduleCycleMap &cycle_map,
    bool input_leaves_only) {
  // First, we topologically sort the nodes in every cycle of the schedule.
  std::vector<std::vector<Node *>> cycle_to_sorted_nodes;
  cycle_to_sorted_nodes.resize(pipeline_length);
  for (Node *node : TopoSort(f)) {
    cycle_to_sorted_nodes[cycle_map.at(node)].emplace_back(node);
  }

  // Then, we traverse the nodes in every cycle to enumerate the cuts of every
  // node. As we have topologically sorted the nodes, for each node, we can
  // enumerate and merge every combination of its operands' cuts. For the input
  // nodes, a.k.a. a node with only live-in operands of the current pipeline
  // cycle or no operand, the cuts should only contain the trivial cut.
  int64_t cycle = 0;
  NodeCutsMap cuts_map;
  for (const std::vector<Node *> &sorted_nodes : cycle_to_sorted_nodes) {
    for (Node *node : sorted_nodes) {
      // Param operation is considered as a primary input (PI).
      if (node->Is<Param>()) {
        continue;
      }

      // Holds the cuts owned by the current node.
      std::vector<NodeCut> cuts;

      // If we only allow PI to be cut leaves, we will not add the trivial cut
      // of any internal node.
      if (!input_leaves_only) {
        XLS_RET_CHECK_OK(AddCut(NodeCut::GetTrivialCut(node), cuts));
      }

      // Enumerate and merge every combination of operands' cuts. We adapt a
      // worklist algorithm for the enumeration.
      std::vector<std::pair<NodeCut, Node *const *>> worklist(
          {std::make_pair(NodeCut(node), node->operands().begin())});

      while (!worklist.empty()) {
        NodeCut current_cut = worklist.back().first;
        Node *const *operand = worklist.back().second;
        worklist.pop_back();

        // Continue if we already finished the merging.
        if (operand == node->operands().end()) {
          XLS_RET_CHECK_OK(AddCut(current_cut, cuts));
          continue;
        }

        // The operand cycle should not be larger than the current cycle.
        int64_t operand_cycle = cycle_map.at(*operand);
        XLS_RET_CHECK_LE(operand_cycle, cycle);
        if (operand_cycle < cycle || (*operand)->Is<Param>()) {
          // If the operand cycle is smaller than the current cycle, the operand
          // is considered as a PI.
          worklist.emplace_back(std::make_pair(
              NodeCut::GetMergedCut(node, current_cut,
                                    NodeCut::GetTrivialCut(*operand)),
              std::next(operand)));
        } else {
          // Otherwise, the operand is internal node whose cuts are merged.
          for (const NodeCut &cut : cuts_map.at(*operand)) {
            worklist.emplace_back(
                std::make_pair(NodeCut::GetMergedCut(node, current_cut, cut),
                               std::next(operand)));
          }
        }
      }
      cuts_map.emplace(node, cuts);
    }
    cycle++;
  }
  return cuts_map;
}

}  // namespace xls
