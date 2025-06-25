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

#include "xls/visualization/ir_viz/ir_to_proto.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/area_model/area_estimator.h"
#include "xls/estimators/delay_model/analyze_critical_path.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/block.h"  // IWYU pragma: keep
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/partial_info_query_engine.h"
#include "xls/passes/proc_state_range_query_engine.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/token_provenance_analysis.h"
#include "xls/passes/union_query_engine.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/visualization/ir_viz/node_attribute_visitor.h"
#include "xls/visualization/ir_viz/visualization.pb.h"
#include "re2/re2.h"

namespace xls {
namespace {

// Returns a globally unique identifier for the node.
std::string GetNodeUniqueId(
    Node* node,
    const absl::flat_hash_map<FunctionBase*, std::string>& function_ids) {
  return absl::StrFormat("%s_%d", function_ids.at(node->function_base()),
                         node->id());
}

// Returns a globally unique identifier for the edge.
std::string GetEdgeUniqueId(
    Node* source, Node* target,
    const absl::flat_hash_map<FunctionBase*, std::string>& function_ids) {
  return absl::StrFormat("%s_to_%s", GetNodeUniqueId(source, function_ids),
                         GetNodeUniqueId(target, function_ids));
}

// Create a short unique id for each function. We avoid using function names
// because they can be very long (e.g., mangled names from xlscc) and because
// names can be shared between blocks and functions/procs.
absl::flat_hash_map<FunctionBase*, std::string> GetFunctionIds(
    Package* package) {
  absl::flat_hash_map<FunctionBase*, std::string> function_ids;
  std::vector<FunctionBase*> function_bases = package->GetFunctionBases();
  for (int64_t i = 0; i < function_bases.size(); ++i) {
    function_ids[function_bases[i]] = absl::StrCat("f", i);
  }
  return function_ids;
}

std::optional<int64_t> MaybeGetStateReadIndex(Node* node) {
  if (node->Is<StateRead>() && node->function_base()->IsProc()) {
    return node->function_base()
        ->AsProcOrDie()
        ->GetStateElementIndex(node->As<StateRead>()->state_element())
        .value();
  }
  return std::nullopt;
}

// Returns the attributes of a node (e.g., the index value of a kTupleIndex
// instruction) as a proto which is to be serialized to JSON.
absl::StatusOr<viz::NodeAttributes> NodeAttributes(
    Node* node,
    const absl::flat_hash_map<Node*, CriticalPathEntry*>& critical_path_map,
    const QueryEngine& query_engine, const PipelineSchedule* schedule,
    const DelayEstimator& delay_estimator,
    const AreaEstimator& area_estimator) {
  AttributeVisitor visitor;
  XLS_RETURN_IF_ERROR(node->VisitSingleNode(&visitor));
  viz::NodeAttributes attributes = visitor.attributes();
  auto it = critical_path_map.find(node);
  if (it != critical_path_map.end()) {
    attributes.set_on_critical_path(true);
  }
  if (query_engine.IsTracked(node)) {
    attributes.set_known_bits(query_engine.ToString(node));
    attributes.set_ranges(query_engine.GetIntervals(node).ToString());
  }
  if (std::optional<int64_t> state_index = MaybeGetStateReadIndex(node);
      state_index.has_value()) {
    attributes.set_state_param_index(state_index.value());
    attributes.set_initial_value(
        node->As<StateRead>()->state_element()->initial_value().ToString());
  }

  absl::StatusOr<int64_t> delay_ps_status =
      delay_estimator.GetOperationDelayInPs(node);
  // The delay model may not have an estimate for this node. This can occur, for
  // example, when viewing a graph before optimizations and optimizations may
  // eliminate the node kind in question so it never is characterized in the
  // delay model. In this case, don't show any delay estimate.
  if (delay_ps_status.ok()) {
    attributes.set_delay_ps(delay_ps_status.value());
  }
  absl::StatusOr<double> area_um_status =
      area_estimator.GetOperationAreaInSquareMicrons(node);
  if (area_um_status.ok()) {
    attributes.set_area_um(*area_um_status);
  }

  if (schedule != nullptr) {
    attributes.set_cycle(schedule->cycle(node));
  }

  return attributes;
}

absl::StatusOr<viz::FunctionBase> FunctionBaseToVisualizationProto(
    FunctionBase* function, const DelayEstimator& delay_estimator,
    const AreaEstimator& area_estimator, const PipelineSchedule* schedule,
    const absl::flat_hash_map<FunctionBase*, std::string>& function_ids,
    bool token_dag) {
  viz::FunctionBase proto;
  proto.set_name(function->name());
  if (function->IsFunction()) {
    proto.set_kind("function");
  } else if (function->IsProc()) {
    proto.set_kind("proc");
  } else {
    XLS_RET_CHECK(function->IsBlock());
    proto.set_kind("block");
  }
  proto.set_id(function_ids.at(function));
  absl::StatusOr<std::vector<CriticalPathEntry>> critical_path =
      AnalyzeCriticalPath(function, /*clock_period_ps=*/std::nullopt,
                          delay_estimator);
  absl::flat_hash_map<Node*, CriticalPathEntry*> node_to_critical_path_entry;
  if (critical_path.ok()) {
    for (CriticalPathEntry& entry : critical_path.value()) {
      node_to_critical_path_entry[entry.node] = &entry;
    }
  } else {
    LOG(WARNING) << "Could not analyze critical path for function: "
                 << critical_path.status();
  }

  auto query_engine = UnionQueryEngine::Of(
      BddQueryEngine(BddQueryEngine::kDefaultPathLimit),
      PartialInfoQueryEngine(), ProcStateRangeQueryEngine());
  XLS_RETURN_IF_ERROR(query_engine.Populate(function).status());

  using NodeDAG =
      absl::flat_hash_map<xls::Node*, absl::flat_hash_set<xls::Node*>>;
  NodeDAG node_dag;
  if (token_dag) {
    XLS_ASSIGN_OR_RETURN(NodeDAG token_dag, ComputeTokenDAG(function));
    for (const auto& [node, predecessors] : token_dag) {
      // account for nodes w/ no predecessors.
      node_dag.try_emplace(node);
      for (const auto& p : predecessors) {
        node_dag[node].insert(p);
        // ensure all precessors have a top-level entry in the dag.
        node_dag.try_emplace(p);
      }
    }
  } else {
    for (Node* node : function->nodes()) {
      // account for nodes w/ no operands.
      node_dag.try_emplace(node);
      for (Node* op : node->operands()) {
        node_dag[node].insert(op);
      }
    }
  }

  for (auto& [node, operands] : node_dag) {
    viz::Node* graph_node = proto.add_nodes();
    graph_node->set_name(node->GetName());
    graph_node->set_id(GetNodeUniqueId(node, function_ids));
    graph_node->set_opcode(OpToString(node->op()));
    graph_node->set_ir(node->ToStringWithOperandTypes());
    for (const auto& loc : node->loc().locations) {
      viz::SourceLocation* graph_loc = graph_node->add_loc();
      graph_loc->set_file(node->package()
                              ->GetFilename(loc.fileno())
                              .value_or(absl::StrCat(loc.fileno().value())));
      graph_loc->set_line(loc.lineno().value());
      graph_loc->set_column(loc.colno().value());
    }
    XLS_ASSIGN_OR_RETURN(
        *graph_node->mutable_attributes(),
        NodeAttributes(node, node_to_critical_path_entry, query_engine,
                       schedule, delay_estimator, area_estimator));
  }
  viz::Node* implicit_sink = nullptr;
  auto get_implicit_sink = [&]() {
    if (implicit_sink == nullptr) {
      implicit_sink = proto.add_nodes();
      implicit_sink->set_name(absl::StrCat(function->name(), "_sink"));
      implicit_sink->set_id(
          absl::StrCat("f", function_ids.at(function), "_sink"));
      implicit_sink->set_opcode("ret");
      implicit_sink->mutable_attributes()->set_on_critical_path(false);
    }
    return implicit_sink;
  };

  for (auto& [node, operands] : node_dag) {
    bool node_on_critical_path = node_to_critical_path_entry.contains(node);
    for (auto& operand : operands) {
      viz::Edge* graph_edge = proto.add_edges();
      graph_edge->set_id(GetEdgeUniqueId(operand, node, function_ids));
      graph_edge->set_source_id(GetNodeUniqueId(operand, function_ids));
      graph_edge->set_target_id(GetNodeUniqueId(node, function_ids));
      graph_edge->set_type(operand->GetType()->ToString());
      graph_edge->set_bit_width(operand->GetType()->GetFlatBitCount());
      graph_edge->set_on_critical_path(
          node_on_critical_path &&
          node_to_critical_path_entry.contains(operand));
    }
    if (function->HasImplicitUse(node)) {
      viz::Node* sink = get_implicit_sink();
      if (node_on_critical_path) {
        sink->mutable_attributes()->set_on_critical_path(true);
      }

      viz::Edge* sink_edge = proto.add_edges();
      sink_edge->set_id(absl::StrFormat(
          "%s_to_%s", GetNodeUniqueId(node, function_ids), sink->id()));
      sink_edge->set_source_id(GetNodeUniqueId(node, function_ids));
      sink_edge->set_target_id(sink->id());
      sink_edge->set_on_critical_path(node_on_critical_path);
    }
  }
  return std::move(proto);
}

// Wraps the given text in a span with the given id, classes, and data. The
// string `str` is modified in place.
absl::Status WrapTextInSpan(
    std::string_view text, std::optional<std::string> dom_id,
    absl::Span<const std::string> classes,
    absl::Span<const std::pair<std::string, std::string>> data,
    std::string* str) {
  std::string open_span = "<span";
  if (dom_id.has_value()) {
    absl::StrAppendFormat(&open_span, " id=\"%s\"", dom_id.value());
  }
  if (!classes.empty()) {
    absl::StrAppendFormat(&open_span, " class=\"%s\"",
                          absl::StrJoin(classes, " "));
  }
  for (const auto& [key, value] : data) {
    absl::StrAppendFormat(&open_span, " data-%s=\"%s\"", key, value);
  }
  open_span.append(">");
  XLS_RET_CHECK(RE2::Replace(str, absl::StrCat("\\b", text, "\\b"),
                             absl::StrFormat("%s%s</span>", open_span, text)));
  return absl::OkStatus();
}

// Wraps the definition of the given node in `str` with an appropriate span.
absl::Status WrapNodeDefInSpan(
    Node* node,
    const absl::flat_hash_map<FunctionBase*, std::string>& function_ids,
    std::string* str) {
  std::string node_id = GetNodeUniqueId(node, function_ids);
  std::vector<std::string> classes = {
      "ir-node-identifier", absl::StrFormat("ir-node-identifier-%s", node_id)};
  std::vector<std::pair<std::string, std::string>> data = {
      {"node-id", node_id},
      {"function-id", function_ids.at(node->function_base())}};
  XLS_RETURN_IF_ERROR(WrapTextInSpan(node->GetName(),
                                     /*dom_id=*/
                                     absl::StrFormat("ir-node-def-%s", node_id),
                                     classes, data, str));
  return absl::OkStatus();
}

// Wraps the use of the given node in `str` with an appropriate span.
absl::Status WrapNodeUseInSpan(
    Node* def, Node* use,
    const absl::flat_hash_map<FunctionBase*, std::string>& function_ids,
    std::string* str) {
  std::string def_id = GetNodeUniqueId(def, function_ids);
  std::string use_id = GetNodeUniqueId(use, function_ids);
  XLS_RETURN_IF_ERROR(WrapTextInSpan(
      def->GetName(),
      /*dom_id=*/std::nullopt,
      /*classes=*/
      {"ir-node-identifier", absl::StrFormat("ir-node-identifier-%s", def_id),
       absl::StrFormat("ir-edge-%s-%s", def_id, use_id)},
      /*data=*/
      {{"node-id", def_id},
       {"function-id", function_ids.at(def->function_base())}},
      str));
  return absl::OkStatus();
}

// Wraps the name of the given function in `str` with an appropriate function
// identifier span.
absl::Status WrapFunctionNameInSpan(std::string_view function_name,
                                    std::string_view function_id,
                                    std::optional<std::string> dom_id,
                                    std::string* str) {
  return WrapTextInSpan(function_name,
                        /*dom_id=*/std::move(dom_id),
                        /*classes=*/{"ir-function-identifier"},
                        /*data=*/{{"identifier", std::string{function_id}}},
                        str);
}

absl::StatusOr<std::string> MarkUpIrText(Package* package) {
  absl::flat_hash_map<FunctionBase*, std::string> function_ids =
      GetFunctionIds(package);

  std::vector<std::string> lines;
  FunctionBase* current_function = nullptr;
  for (std::string_view line_view : absl::StrSplit(package->DumpIr(), '\n')) {
    std::string line{line_view};

    // Match function/proc/block signature. Put spans around function name and
    // parameter names.
    //
    //   fn foo(a: bits[32]) {
    //
    // =>
    //
    //   <span>fn <span>foo</span>(<span>a</span>: bits[32]) {
    //
    std::string is_top;
    std::string kind;
    std::string function_name;
    if (RE2::PartialMatch(line, R"(^\s*(top|)\s*(fn|proc|block)\s+(\w+))",
                          &is_top, &kind, &function_name)) {
      std::vector<Node*> args;
      if (kind == "fn") {
        XLS_ASSIGN_OR_RETURN(current_function,
                             package->GetFunction(function_name));
      } else if (kind == "proc") {
        XLS_ASSIGN_OR_RETURN(current_function, package->GetProc(function_name));
      } else {
        XLS_RET_CHECK_EQ(kind, "block");
        XLS_ASSIGN_OR_RETURN(current_function,
                             package->GetBlock(function_name));
      }
      XLS_RETURN_IF_ERROR(WrapFunctionNameInSpan(
          current_function->name(), function_ids.at(current_function),
          /*dom_id=*/
          absl::StrFormat("ir-function-def-%s",
                          function_ids.at(current_function)),
          &line));

      // Wrap the parameters in spans.
      for (Param* node : current_function->params()) {
        XLS_RETURN_IF_ERROR(WrapNodeDefInSpan(node, function_ids, &line));
      }

      // Prefix the line with a opening span which spans the entire function.
      lines.push_back(absl::StrFormat(
          "<span id=\"ir-function-%s\" class=\"ir-function\">%s",
          function_ids.at(current_function), line));
      continue;
    }

    // Match node definitions:
    //
    //   bar: bits[32] = op(x, y, ...) {
    //
    // =>
    //
    //   <span>bar</span>: bits[32] = op(<span>x</span>, <span>y</span>, ...)
    std::string node_name;
    if (RE2::PartialMatch(line, R"(^\s*(?:ret\s+)?([_a-zA-Z0-9.]+)\s*:)",
                          &node_name)) {
      XLS_ASSIGN_OR_RETURN(Node * node, current_function->GetNode(node_name));
      XLS_RETURN_IF_ERROR(WrapNodeDefInSpan(node, function_ids, &line));

      // Wrap the operands in spans.
      for (Node* operand : node->operands()) {
        XLS_RETURN_IF_ERROR(
            WrapNodeUseInSpan(operand, node, function_ids, &line));
      }

      // If the node calls another function then wrap the function identifier.
      std::string callee_name;
      if (RE2::PartialMatch(line, R"(^.*to_apply=([_a-zA-Z0-9.]+))",
                            &callee_name)) {
        XLS_ASSIGN_OR_RETURN(Function * callee,
                             package->GetFunction(callee_name));
        XLS_RETURN_IF_ERROR(
            WrapFunctionNameInSpan(callee_name, function_ids.at(callee),
                                   /*dom_id=*/std::nullopt, &line));
      }

      lines.push_back(std::string{line});
      continue;
    }

    // Add a </span> after a closing '}' for the entire function/proc/block
    // span.
    if (RE2::PartialMatch(line, R"(^\s*}\s*$)")) {
      lines.push_back(absl::StrFormat("%s</span>", line));
      continue;
    }

    lines.push_back(std::string{line});
  }
  return absl::StrJoin(lines, "\n");
}

struct NoAreaEstimator final : public AreaEstimator {
 public:
  explicit NoAreaEstimator() : AreaEstimator("NoArea") {}
  absl::StatusOr<double> GetOneBitRegisterAreaInSquareMicrons() const override {
    return absl::UnknownError("No area estimation");
  }
  absl::StatusOr<double> GetOperationAreaInSquareMicrons(
      Node* node) const override {
    return absl::UnknownError("No area estimation");
  }
};
}  // namespace

absl::StatusOr<viz::Package> IrToProto(
    Package* package, const DelayEstimator& delay_estimator,
    const PipelineSchedule* schedule,
    std::optional<std::string_view> entry_name, bool token_dag) {
  NoAreaEstimator no_area;
  return IrToProto(package, delay_estimator, no_area, schedule, entry_name,
                   token_dag);
}

absl::StatusOr<viz::Package> IrToProto(
    Package* package, const DelayEstimator& delay_estimator,
    const AreaEstimator& area_estimator, const PipelineSchedule* schedule,
    std::optional<std::string_view> entry_name, bool token_dag) {
  viz::Package proto;

  absl::flat_hash_map<FunctionBase*, std::string> function_ids =
      GetFunctionIds(package);

  std::optional<FunctionBase*> entry_function_base;
  for (FunctionBase* fb : package->GetFunctionBases()) {
    XLS_ASSIGN_OR_RETURN(
        *proto.add_function_bases(),
        FunctionBaseToVisualizationProto(
            fb, delay_estimator, area_estimator,
            schedule != nullptr && schedule->function_base() == fb ? schedule
                                                                   : nullptr,
            function_ids, token_dag));
    if (entry_name.has_value() && fb->name() == entry_name.value()) {
      entry_function_base = fb;
    }
  }
  proto.set_name(package->name());
  XLS_ASSIGN_OR_RETURN(std::string ir_html, MarkUpIrText(package));
  proto.set_ir_html(ir_html);
  if (entry_function_base.has_value()) {
    proto.set_entry_id(function_ids.at(entry_function_base.value()));
  } else {
    std::optional<FunctionBase*> top = package->GetTop();
    if (top.has_value()) {
      proto.set_entry_id(function_ids.at(top.value()));
    }
  }

  return proto;
}

}  // namespace xls
