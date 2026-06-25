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

#include "xls/contrib/eco/xls_ir_to_graph.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/eco/graph.h"
#include "xls/ir/format_strings.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/visualization/ir_viz/node_attribute_visitor.h"

namespace xls {
namespace {

absl::Status AddStateReadAttributes(
    Node* node, NodeCostAttributes* attrs) {
  if (!node->Is<StateRead>()) {
    return absl::OkStatus();
  }

  StateRead* state_read = node->As<StateRead>();
  attrs->state_element = state_read->state_element()->name();
  XLS_ASSIGN_OR_RETURN(attrs->state_initial_value,
                       state_read->state_element()->initial_value().AsProto());
  if (node->function_base()->IsProc()) {
    XLS_ASSIGN_OR_RETURN(
        int64_t index,
        node->function_base()->AsProcOrDie()->GetStateElementIndex(
            state_read->state_element()));
    attrs->state_index = index;
  }
  return absl::OkStatus();
}

void AddTraceAttributes(Node* node, NodeCostAttributes* attrs) {
  if (!node->Is<Trace>()) {
    return;
  }
  attrs->trace_xls_format = StepsToXlsFormatString(node->As<Trace>()->format());
}

absl::StatusOr<NodeCostAttributes> GetNodeCostAttributes(Node* node) {
  NodeCostAttributes attrs;
  attrs.op = node->op();
  attrs.data_type = node->GetType()->ToProto();
  attrs.operand_data_types.reserve(node->operand_count());
  for (Node* operand : node->operands()) {
    attrs.operand_data_types.push_back(operand->GetType()->ToProto());
  }
  if (node->Is<Literal>()) {
    XLS_ASSIGN_OR_RETURN(attrs.literal_value,
                         node->As<Literal>()->value().AsProto());
  }
  if (node->Is<ArrayIndex>()) {
    attrs.array_assumed_in_bounds = node->As<ArrayIndex>()->assumed_in_bounds();
  } else if (node->Is<ArrayUpdate>()) {
    attrs.array_assumed_in_bounds = node->As<ArrayUpdate>()->assumed_in_bounds();
  }
  XLS_RETURN_IF_ERROR(AddStateReadAttributes(node, &attrs));
  AddTraceAttributes(node, &attrs);

  AttributeVisitor visitor;
  XLS_RETURN_IF_ERROR(node->VisitSingleNode(&visitor));
  attrs.node_attributes = visitor.attributes();
  return attrs;
}

EdgeCostAttributes GetEdgeCostAttributes(Node* operand, Node* user,
                                         int64_t index) {
  EdgeCostAttributes attrs;
  attrs.source_data_type = operand->GetType()->ToProto();
  attrs.source_op = operand->op();
  attrs.sink_data_type = user->GetType()->ToProto();
  attrs.sink_op = user->op();
  if (!OpIsCommutative(user->op())) {
    attrs.index = index;
  }
  return attrs;
}

void SortEdgesAndRefresh(XLSGraph& graph) {
  absl::c_sort(graph.edges, [](const XLSEdge& a, const XLSEdge& b) {
    if (a.endpoints.first != b.endpoints.first) {
      return a.endpoints.first < b.endpoints.first;
    }
    if (a.endpoints.second != b.endpoints.second) {
      return a.endpoints.second < b.endpoints.second;
    }
    return a.index < b.index;
  });
  graph.RefreshAdjacency();
  graph.RefreshEdgeCounts();
}

}  // namespace

absl::StatusOr<XLSGraph> XlsIrToGraph(FunctionBase* function_base) {
  XLS_RET_CHECK(function_base != nullptr);

  XLSGraph graph;
  for (Node* node : function_base->nodes()) {
    XLS_ASSIGN_OR_RETURN(NodeCostAttributes cost_attributes,
                         GetNodeCostAttributes(node));
    XLSNode graph_node(node->GetName(), cost_attributes);
    graph_node.all_attributes = {
        {"id", absl::StrCat(node->id())},
        {"name", node->GetName()},
        {"op", OpToString(node->op())},
        {"ir", node->ToStringWithOperandTypes()},
        {"cost_attributes", graph_node.cost_attributes.DebugString()},
    };
    graph.add_node(graph_node);
  }

  // TODO(xls-eco): Model channels as first-class graph nodes so channel
  // additions, removals, and type changes are detected by the differencer
  // instead of requiring the manual fixup documented in test/BUILD.
  for (Node* node : function_base->nodes()) {
    XLS_RET_CHECK(graph.node_name_to_index.contains(node->GetName()))
        << "Missing graph node for IR node " << node->GetName();
    const int sink = graph.node_name_to_index.at(node->GetName());
    for (int64_t index = 0; index < node->operand_count(); ++index) {
      Node* operand = node->operand(index);
      XLS_RET_CHECK(graph.node_name_to_index.contains(operand->GetName()))
          << "Missing graph node for operand " << operand->GetName();
      const int source = graph.node_name_to_index.at(operand->GetName());
      graph.add_edge(
          XLSEdge(source, sink, GetEdgeCostAttributes(operand, node, index),
                  static_cast<int>(index)));
    }
  }

  if (function_base->IsFunction()) {
    Function* function = function_base->AsFunctionOrDie();
    if (function->return_value() != nullptr) {
      graph.return_node_name = function->return_value()->GetName();
    }
  }

  SortEdgesAndRefresh(graph);
  graph.populate_node_signatures();
  graph.RefreshReturnAndIndex();
  return graph;
}

absl::StatusOr<XLSGraph> ParseIrFileToGraph(std::string_view ir_path) {
  if (ir_path == "-") {
    ir_path = "/dev/stdin";
  }

  XLS_ASSIGN_OR_RETURN(std::string ir, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackage(ir, ir_path));
  std::optional<FunctionBase*> top = package->GetTop();
  XLS_RET_CHECK(top.has_value()) << "IR package has no top entity";
  return XlsIrToGraph(*top);
}

}  // namespace xls
