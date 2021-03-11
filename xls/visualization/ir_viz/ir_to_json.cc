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

#include "xls/visualization/ir_viz/ir_to_json.h"

#include "google/protobuf/util/json_util.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/delay_model/analyze_critical_path.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/visualization/ir_viz/ir_for_visualization.pb.h"

namespace xls {
namespace {

// Visitor which constructs the attributes (if any) of a node and returns them
// as a JSON object.
class AttributeVisitor : public DfsVisitorWithDefault {
 public:
  absl::Status DefaultHandler(Node* node) override { return absl::OkStatus(); }

  absl::Status HandleLiteral(Literal* literal) override {
    attributes_.set_value(
        literal->value().ToHumanString(FormatPreference::kHex));
    return absl::OkStatus();
  }

  absl::Status HandleBitSlice(BitSlice* bit_slice) override {
    attributes_.set_start(bit_slice->start());
    attributes_.set_width(bit_slice->width());
    return absl::OkStatus();
  }

  absl::Status HandleTupleIndex(TupleIndex* tuple_index) override {
    attributes_.set_index(tuple_index->index());
    return absl::OkStatus();
  }

  const IrForVisualization::Attributes& attributes() const {
    return attributes_;
  }

 private:
  IrForVisualization::Attributes attributes_;
};

// Returns the attributes of a node (e.g., the index value of a kTupleIndex
// instruction) as a proto which is to be serialized to JSON.
absl::StatusOr<IrForVisualization::Attributes> NodeAttributes(
    Node* node,
    const absl::flat_hash_map<Node*, CriticalPathEntry*>& critical_path_map,
    const QueryEngine& query_engine, const PipelineSchedule* schedule) {
  AttributeVisitor visitor;
  XLS_RETURN_IF_ERROR(node->VisitSingleNode(&visitor));
  IrForVisualization::Attributes attributes = visitor.attributes();
  auto it = critical_path_map.find(node);
  if (it != critical_path_map.end()) {
    attributes.set_on_critical_path(true);
  }
  if (query_engine.IsTracked(node)) {
    attributes.set_known_bits(query_engine.ToString(node));
  }

  absl::StatusOr<int64_t> delay_ps_status =
      GetStandardDelayEstimator().GetOperationDelayInPs(node);
  if (delay_ps_status.ok()) {
    attributes.set_delay_ps(delay_ps_status.value());
  }

  if (schedule != nullptr) {
    attributes.set_cycle(schedule->cycle(node));
  }

  return attributes;
}

}  // namespace

absl::StatusOr<std::string> IrToJson(FunctionBase* function,
                                     const DelayEstimator& delay_estimator,
                                     const PipelineSchedule* schedule) {
  IrForVisualization ir;
  absl::StatusOr<std::vector<CriticalPathEntry>> critical_path =
      AnalyzeCriticalPath(function, /*clock_period_ps=*/absl::nullopt,
                          delay_estimator);
  absl::flat_hash_map<Node*, CriticalPathEntry*> node_to_critical_path_entry;
  if (critical_path.ok()) {
    for (CriticalPathEntry& entry : critical_path.value()) {
      node_to_critical_path_entry[entry.node] = &entry;
    }
  } else {
    XLS_LOG(WARNING) << "Could not analyze critical path for function: "
                     << critical_path.status();
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<BddQueryEngine> query_engine,
                       BddQueryEngine::Run(function, /*minterm_limit=*/4096));

  auto sanitize_name = [](absl::string_view name) {
    return absl::StrReplaceAll(name, {{".", "_"}});
  };
  for (Node* node : function->nodes()) {
    IrForVisualization::Node* graph_node = ir.add_nodes();
    graph_node->set_name(node->GetName());
    graph_node->set_id(sanitize_name(node->GetName()));
    graph_node->set_opcode(OpToString(node->op()));
    graph_node->set_ir(node->ToStringWithOperandTypes());
    XLS_ASSIGN_OR_RETURN(*graph_node->mutable_attributes(),
                         NodeAttributes(node, node_to_critical_path_entry,
                                        *query_engine, schedule));
  }

  for (Node* node : function->nodes()) {
    for (int64_t i = 0; i < node->operand_count(); ++i) {
      Node* operand = node->operand(i);
      IrForVisualization::Edge* graph_edge = ir.add_edges();
      std::string source = sanitize_name(operand->GetName());
      std::string target = sanitize_name(node->GetName());
      graph_edge->set_id(absl::StrFormat("%s_to_%s_%d", source, target, i));
      graph_edge->set_source(sanitize_name(operand->GetName()));
      graph_edge->set_target(sanitize_name(node->GetName()));
      graph_edge->set_type(operand->GetType()->ToString());
      graph_edge->set_bit_width(operand->GetType()->GetFlatBitCount());
    }
  }

  std::string serialized_json;
  google::protobuf::util::JsonPrintOptions print_options;
  print_options.add_whitespace = true;
  print_options.preserve_proto_field_names = true;

  auto status =
      google::protobuf::util::MessageToJsonString(ir, &serialized_json, print_options);
  if (!status.ok()) {
    return absl::InternalError(std::string{status.message()});
  }
  return serialized_json;
}

}  // namespace xls
