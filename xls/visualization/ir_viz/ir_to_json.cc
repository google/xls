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
#include "xls/visualization/ir_viz/visualization.pb.h"

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

  const visualization::NodeAttributes& attributes() const {
    return attributes_;
  }

 private:
  visualization::NodeAttributes attributes_;
};

// Returns the attributes of a node (e.g., the index value of a kTupleIndex
// instruction) as a proto which is to be serialized to JSON.
absl::StatusOr<visualization::NodeAttributes> NodeAttributes(
    Node* node,
    const absl::flat_hash_map<Node*, CriticalPathEntry*>& critical_path_map,
    const QueryEngine& query_engine, const PipelineSchedule* schedule,
    const DelayEstimator& delay_estimator) {
  AttributeVisitor visitor;
  XLS_RETURN_IF_ERROR(node->VisitSingleNode(&visitor));
  visualization::NodeAttributes attributes = visitor.attributes();
  auto it = critical_path_map.find(node);
  if (it != critical_path_map.end()) {
    attributes.set_on_critical_path(true);
  }
  if (query_engine.IsTracked(node)) {
    attributes.set_known_bits(query_engine.ToString(node));
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

  if (schedule != nullptr) {
    attributes.set_cycle(schedule->cycle(node));
  }

  return attributes;
}

absl::StatusOr<visualization::FunctionBase> FunctionBaseToVisualizationProto(
    FunctionBase* function, const DelayEstimator& delay_estimator,
    const PipelineSchedule* schedule) {
  visualization::FunctionBase proto;
  proto.set_name(function->name());
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

  BddQueryEngine query_engine(BddFunction::kDefaultPathLimit);
  XLS_RETURN_IF_ERROR(query_engine.Populate(function).status());

  auto sanitize_name = [](absl::string_view name) {
    return absl::StrReplaceAll(name, {{".", "_"}});
  };
  for (Node* node : function->nodes()) {
    visualization::Node* graph_node = proto.add_nodes();
    graph_node->set_name(node->GetName());
    graph_node->set_id(sanitize_name(node->GetName()));
    graph_node->set_opcode(OpToString(node->op()));
    graph_node->set_ir(node->ToStringWithOperandTypes());
    XLS_ASSIGN_OR_RETURN(
        *graph_node->mutable_attributes(),
        NodeAttributes(node, node_to_critical_path_entry, query_engine,
                       schedule, delay_estimator));
  }

  for (Node* node : function->nodes()) {
    for (int64_t i = 0; i < node->operand_count(); ++i) {
      Node* operand = node->operand(i);
      visualization::Edge* graph_edge = proto.add_edges();
      std::string source = sanitize_name(operand->GetName());
      std::string target = sanitize_name(node->GetName());
      graph_edge->set_id(absl::StrFormat("%s_to_%s_%d", source, target, i));
      graph_edge->set_source(sanitize_name(operand->GetName()));
      graph_edge->set_target(sanitize_name(node->GetName()));
      graph_edge->set_type(operand->GetType()->ToString());
      graph_edge->set_bit_width(operand->GetType()->GetFlatBitCount());
    }
  }
  return std::move(proto);
}

}  // namespace

absl::StatusOr<std::string> IrToJson(
    Package* package, const DelayEstimator& delay_estimator,
    const PipelineSchedule* schedule,
    absl::optional<absl::string_view> entry_name) {
  visualization::Package proto;

  proto.set_name(package->name());
  for (FunctionBase* fb : package->GetFunctionBases()) {
    XLS_ASSIGN_OR_RETURN(
        *proto.add_function_bases(),
        FunctionBaseToVisualizationProto(
            fb, delay_estimator,
            schedule != nullptr && schedule->function_base() == fb ? schedule
                                                                   : nullptr));
  }
  if (entry_name.has_value()) {
    proto.set_entry(std::string{entry_name.value()});
  } else {
    absl::StatusOr<Function*> entry_status = package->EntryFunction();
    if (entry_status.ok()) {
      proto.set_entry(entry_status.value()->name());
    }
  }

  std::string serialized_json;
  google::protobuf::util::JsonPrintOptions print_options;
  print_options.add_whitespace = true;
  print_options.preserve_proto_field_names = true;

  auto status =
      google::protobuf::util::MessageToJsonString(proto, &serialized_json, print_options);
  if (!status.ok()) {
    return absl::InternalError(std::string{status.message()});
  }
  return serialized_json;
}

}  // namespace xls
