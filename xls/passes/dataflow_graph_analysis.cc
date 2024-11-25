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

#include "xls/passes/dataflow_graph_analysis.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/ternary.h"
#include "xls/ir/topo_sort.h"
#include "xls/passes/query_engine.h"
#include "ortools/graph/ebert_graph.h"
#include "ortools/graph/max_flow.h"

namespace xls {

namespace {

// Returns whether the given node *originates* data that is potentially unknown.
bool IsDataOriginating(Node* node) {
  return node->OpIn({Op::kReceive, Op::kRegisterRead, Op::kParam,
                     Op::kStateRead, Op::kInputPort, Op::kInstantiationInput});
}

}  // namespace

DataflowGraphAnalysis::DataflowGraphAnalysis(FunctionBase* f,
                                             const QueryEngine* query_engine)
    : nodes_(TopoSort(f)) {
  CHECK_LT(
      nodes_.size(),
      static_cast<size_t>(
          (std::numeric_limits<operations_research::NodeIndex>::max() >> 1) -
          1));
  node_to_index_.reserve(nodes_.size());
  for (size_t i = 0; i < nodes_.size(); ++i) {
    node_to_index_[nodes_[i]] = i;
  }

  graph_ = std::make_unique<Graph>();
  graph_->AddNode(kSourceIndex);
  absl::flat_hash_map<operations_research::ArcIndex, int64_t> arc_capacities;
  absl::flat_hash_map<Node*, operations_research::ArcIndex> internal_arcs;
  absl::flat_hash_map<Node*, operations_research::ArcIndex> source_arcs;
  absl::flat_hash_map<Node*, operations_research::ArcIndex> sink_arcs;
  for (size_t i = 0; i < nodes_.size(); ++i) {
    Node* node = nodes_[i];

    operations_research::NodeIndex v_in = InIndex(i);
    operations_research::NodeIndex v_out = OutIndex(i);
    graph_->AddNode(v_in);
    graph_->AddNode(v_out);

    int64_t bit_count = node->GetType()->GetFlatBitCount();

    int64_t unknown_bits = bit_count;
    if (node->Is<Literal>()) {
      unknown_bits = 0;
    } else if (IsDataOriginating(node)) {
      // These nodes originate (potentially) variable data; they are the points
      // of entry for unknown data.
      source_arcs[node] = graph_->AddArc(kSourceIndex, v_in);
      arc_capacities[source_arcs[node]] = bit_count;
    } else if (query_engine != nullptr) {
      if (std::optional<SharedLeafTypeTree<TernaryVector>> ternary_value =
              query_engine->GetTernary(node);
          ternary_value.has_value()) {
        unknown_bits = 0;
        leaf_type_tree::ForEach(
            ternary_value->AsView(), [&](const TernaryVector& ternary) {
              unknown_bits += absl::c_count(ternary, TernaryValue::kUnknown);
            });
      }
    }

    internal_arcs[node] = graph_->AddArc(v_in, v_out);
    arc_capacities[internal_arcs[node]] = unknown_bits;
    sink_arcs[node] = graph_->AddArc(v_in, kSinkIndex);
    arc_capacities[sink_arcs[node]] = 0;
    for (Node* user : node->users()) {
      arc_capacities[graph_->AddArc(v_out, InIndex(user))] = unknown_bits;
    }
  }

  std::vector<operations_research::ArcIndex> arc_permutation;
  graph_->Build(&arc_permutation);
  auto permuted = [&](operations_research::ArcIndex arc) {
    return arc < arc_permutation.size() ? arc_permutation[arc] : arc;
  };
  for (const auto& [arc, capacity] : arc_capacities) {
    arc_capacities_[permuted(arc)] = capacity;
  }
  for (const auto& [node, arc] : source_arcs) {
    source_arcs_[node] = permuted(arc);
  }
  for (const auto& [node, arc] : internal_arcs) {
    internal_arcs_[node] = permuted(arc);
  }
  for (const auto& [node, arc] : sink_arcs) {
    sink_arcs_[node] = permuted(arc);
  }
  max_flow_ = std::make_unique<operations_research::GenericMaxFlow<Graph>>(
      graph_.get(), kSourceIndex, kSinkIndex);
  for (auto& [arc, capacity] : arc_capacities_) {
    max_flow_->SetArcCapacity(arc, capacity);
  }
}

absl::Status DataflowGraphAnalysis::SolveFor(Node* node) {
  if (current_sink_ == node) {
    return absl::OkStatus();
  }
  if (current_sink_ != nullptr) {
    max_flow_->SetArcCapacity(sink_arcs_[current_sink_], 0);
  }
  max_flow_->SetArcCapacity(sink_arcs_[node],
                            std::numeric_limits<int64_t>::max());
  current_sink_ = node;
  if (max_flow_->Solve()) {
    return absl::OkStatus();
  }
  switch (max_flow_->status()) {
    case operations_research::GenericMaxFlow<Graph>::NOT_SOLVED:
      return absl::InternalError("Max flow solver failed to solve");
    case operations_research::GenericMaxFlow<Graph>::OPTIMAL:
      return absl::InternalError("Max flow solver reported an unknown failure");
    case operations_research::GenericMaxFlow<Graph>::INT_OVERFLOW:
      return absl::InternalError("Possible overflow in max flow solver");
    case operations_research::GenericMaxFlow<Graph>::BAD_INPUT:
      return absl::InternalError("Bad input to max flow solver");
    case operations_research::GenericMaxFlow<Graph>::BAD_RESULT:
      return absl::InternalError("Bad result from max flow solver");
  }
  return absl::InternalError(
      absl::StrCat("Unknown max flow solver status: ", max_flow_->status()));
}

absl::StatusOr<int64_t> DataflowGraphAnalysis::GetUnknownBitsFor(Node* node) {
  XLS_RETURN_IF_ERROR(SolveFor(node));
  return max_flow_->GetOptimalFlow();
}

absl::StatusOr<std::vector<Node*>> DataflowGraphAnalysis::GetMinCutFor(
    Node* node, std::optional<int64_t> max_unknown_bits,
    int64_t* unknown_bits) {
  XLS_RETURN_IF_ERROR(SolveFor(node));

  if (max_unknown_bits.has_value() &&
      max_flow_->GetOptimalFlow() > *max_unknown_bits) {
    // The max flow has too many unknown bits, so return nothing.
    return std::vector<Node*>({});
  }

  std::vector<operations_research::NodeIndex> min_cut_indices;
  max_flow_->GetSourceSideMinCut(&min_cut_indices);

  absl::flat_hash_set<Node*> min_cut_nodes;
  for (operations_research::NodeIndex index : min_cut_indices) {
    if (index == kSourceIndex) {
      for (const auto& [source, arc_index] : source_arcs_) {
        if (max_flow_->Flow(arc_index) > 0) {
          min_cut_nodes.insert(source);
        }
      }
      continue;
    }

    Node* cut_node = nodes_[TopoIndex(index)];
    if (max_flow_->Flow(internal_arcs_.at(cut_node)) > 0) {
      min_cut_nodes.insert(cut_node);
    }
  }

  std::vector<Node*> boundary_nodes;
  for (Node* min_cut_node : min_cut_nodes) {
    // A node is on the boundary if it has an active outgoing arc to a node that
    // is not in the min cut.
    if (absl::c_any_of(
            max_flow_->graph()->OutgoingArcs(OutIndex(min_cut_node)),
            [&](operations_research::ArcIndex out_arc) {
              Node* target =
                  nodes_[TopoIndex(max_flow_->graph()->Head(out_arc))];
              return !min_cut_nodes.contains(target) &&
                     max_flow_->Flow(out_arc) > 0;
            })) {
      boundary_nodes.push_back(min_cut_node);
    }
  }
  absl::c_sort(boundary_nodes, Node::NodeIdLessThan());

  int64_t unknown_bits_storage = 0;
  if (unknown_bits != nullptr) {
    *unknown_bits = 0;
  } else {
    unknown_bits = &unknown_bits_storage;
  }
  for (Node* boundary_node : boundary_nodes) {
    *unknown_bits += max_flow_->Capacity(internal_arcs_[boundary_node]);
  }
  if (max_unknown_bits.has_value() && *unknown_bits > *max_unknown_bits) {
    // The min cut has too many unknown bits, so return nothing.
    return std::vector<Node*>({});
  }

  return boundary_nodes;
}

}  // namespace xls
