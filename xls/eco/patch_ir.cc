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
#include "xls/eco/patch_ir.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/status_macros.h"
#include "xls/eco/ir_patch.pb.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/ir/xls_type.pb.h"
#include "xls/ir/xls_value.pb.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/run_pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/tools/codegen_flags.h"
#include "xls/tools/scheduling_options_flags.h"

namespace xls {

PatchIr::PatchIr(FunctionBase* function_base, xls_eco::IrPatchProto& patch)
    : patch_(patch), function_base_(function_base), schedule_(std::nullopt) {
  std::copy(patch_.edit_paths().begin(), patch_.edit_paths().end(),
            std::back_inserter(sorted_edit_paths_));
  std::sort(sorted_edit_paths_.begin(), sorted_edit_paths_.end(),
            [this](const xls_eco::EditPathProto& lhs,
                   const xls_eco::EditPathProto& rhs) {
              return this->CompareEditPaths(lhs, rhs);
            });
  package_ = function_base_->package();
}
absl::StatusOr<std::vector<Node*>> PatchIr::MakeDummyNodes(
    absl::Span<Type*> types) {
  std::vector<Node*> dummy_nodes;
  for (Type* type : types) {
    XLS_ASSIGN_OR_RETURN(Node * dummy_node, MakeDummyNode(type));
    dummy_nodes.push_back(dummy_node);
  }
  return dummy_nodes;
}
absl::StatusOr<Node*> PatchIr::MakeDummyNode(Type* type) {
  XLS_ASSIGN_OR_RETURN(Node * dummy_node,
                       function_base_->MakeNodeWithName<Literal>(
                           SourceInfo(), ZeroOfType(type), "Dummy"));
  return dummy_node;
}
absl::Status PatchIr::UpdateNodeMaps(Node* n, absl::Span<Node*> dummy_operands,
                                     std::string_view node_name) {
  auto& dummy_nodes = dummy_nodes_map_[n];
  dummy_nodes.insert(dummy_nodes.begin(), dummy_operands.begin(),
                     dummy_operands.end());
  patch_to_ir_node_map_[node_name] = n->GetName();
  return absl::OkStatus();
}
absl::Status PatchIr::CleanupDummyNodes(Node* node) {
  auto& dummy_nodes = dummy_nodes_map_[node];
  for (auto it = dummy_nodes.begin(); it != dummy_nodes.end();) {
    Node* dummy_node = *it;
    XLS_RETURN_IF_ERROR(function_base_->RemoveNode(dummy_node));
    it = dummy_nodes.erase(it);
  }
  return absl::OkStatus();
}

absl::StatusOr<int64_t> PatchIr::GetProtoBitCount(const TypeProto& type) {
  XLS_ASSIGN_OR_RETURN(Type * t, package_->GetTypeFromProto(type));
  XLS_ASSIGN_OR_RETURN(BitsType * b, t->AsBits());
  return b->bit_count();
}

absl::Status PatchIr::PatchContainsNode(std::string_view node_name) {
  if (patch_to_ir_node_map_.find(node_name) != patch_to_ir_node_map_.end())
    return absl::OkStatus();
  return absl::NotFoundError("Patch does not contain the node.");
}

absl::Status PatchIr::ApplyPatch() {
  // XLS_RETURN_IF_ERROR(IsolateReturnNode());
  for (const xls_eco::EditPathProto& edit_path : sorted_edit_paths_) {
    XLS_RETURN_IF_ERROR(ApplyPath(edit_path));
  }
  // XLS_RETURN_IF_ERROR(RestoreReturnNode());
  XLS_RETURN_IF_ERROR(ValidatePatch());
  return absl::OkStatus();
}

absl::Status PatchIr::ApplyPath(const xls_eco::EditPathProto& edit_path) {
  switch (edit_path.operation()) {
    case xls_eco::Operation::DELETE:
      XLS_RETURN_IF_ERROR(edit_path.has_node_edit_path()
                              ? ApplyDeletePath(edit_path.node_edit_path())
                              : ApplyDeletePath(edit_path.edge_edit_path()));
      break;
    case xls_eco::Operation::INSERT:
      XLS_RETURN_IF_ERROR(edit_path.has_node_edit_path()
                              ? ApplyInsertPath(edit_path.node_edit_path())
                              : ApplyInsertPath(edit_path.edge_edit_path()));
      break;
    case xls_eco::Operation::UPDATE:
      XLS_RETURN_IF_ERROR(edit_path.has_node_edit_path()
                              ? ApplyUpdatePath(edit_path.node_edit_path())
                              : ApplyUpdatePath(edit_path.edge_edit_path()));
      break;
    default:
      return absl::InvalidArgumentError("Invalid operation");
  }
  return absl::OkStatus();
}
absl::Status PatchIr::ApplyDeletePath(
    const xls_eco::NodeEditPathProto& node_delete) {
  XLS_ASSIGN_OR_RETURN(Node * n,
                       function_base_->GetNode(node_delete.node().name()));
  XLS_RETURN_IF_ERROR(function_base_->RemoveNode(n));
  XLS_RETURN_IF_ERROR(CleanupDummyNodes(n));
  return absl::OkStatus();
}
absl::Status PatchIr::ApplyDeletePath(
    const xls_eco::EdgeEditPathProto& edge_delete) {
  XLS_ASSIGN_OR_RETURN(Node * from_node,
                       function_base_->GetNode(edge_delete.edge().from_node()));
  XLS_ASSIGN_OR_RETURN(Node * to_node,
                       function_base_->GetNode(edge_delete.edge().to_node()));
  XLS_ASSIGN_OR_RETURN(Node * dummy_node, MakeDummyNode(from_node->GetType()));
  dummy_nodes_map_[to_node].push_back(dummy_node);
  XLS_RETURN_IF_ERROR(to_node->ReplaceOperandNumber(edge_delete.edge().index(),
                                                    dummy_node, false));
  return absl::OkStatus();
}
absl::Status PatchIr::ApplyInsertPath(
    const xls_eco::NodeEditPathProto& node_insert) {
  const xls_eco::NodeProto& patch_node = node_insert.node();
  if (patch_to_ir_op_map_.find(patch_node.op()) == patch_to_ir_op_map_.end()) {
    std::cerr << "Error! Unsupported operation: " << patch_node.op() << '\n';
    return absl::InvalidArgumentError("Unsupported operation");
  }
  const Op op = patch_to_ir_op_map_.at(patch_node.op());
  std::vector<Type*> operand_types = {};
  if (patch_node.operand_data_types_size() > 0) {
    std::transform(patch_node.operand_data_types().begin(),
                   patch_node.operand_data_types().end(),
                   std::back_inserter(operand_types),
                   [&](const TypeProto& type) {
                     return package_->GetTypeFromProto(type).value();
                   });
  }
  XLS_ASSIGN_OR_RETURN(std::vector<Node*> dummy_operands,
                       MakeDummyNodes(absl::MakeSpan(operand_types)));
  Node* n = nullptr;
  switch (op) {
    case (Op::kLiteral): {
      XLS_ASSIGN_OR_RETURN(Value v,
                           Value::FromProto(patch_node.unique_args(0).value()));
      XLS_ASSIGN_OR_RETURN(n,
                           function_base_->MakeNode<Literal>(SourceInfo(), v));
      XLS_RETURN_IF_ERROR(
          UpdateNodeMaps(n, absl::MakeSpan(dummy_operands), patch_node.name()));
      break;
    }
    case (Op::kSignExt): {
      XLS_ASSIGN_OR_RETURN(n,
                           function_base_->MakeNode<ExtendOp>(
                               SourceInfo(), dummy_operands.front(),
                               patch_node.unique_args(0).new_bit_count(), op));
      XLS_RETURN_IF_ERROR(
          UpdateNodeMaps(n, absl::MakeSpan(dummy_operands), patch_node.name()));
      break;
    }
    case (Op::kBitSlice): {
      XLS_ASSIGN_OR_RETURN(int64_t width,
                           GetProtoBitCount(patch_node.data_type()));
      XLS_ASSIGN_OR_RETURN(n, function_base_->MakeNode<BitSlice>(
                                  SourceInfo(), dummy_operands.front(),
                                  patch_node.unique_args(0).start(), width));
      XLS_RETURN_IF_ERROR(
          UpdateNodeMaps(n, absl::MakeSpan(dummy_operands), patch_node.name()));
      break;
    }
    case (Op::kTuple): {
      XLS_ASSIGN_OR_RETURN(
          n, function_base_->MakeNode<Tuple>(SourceInfo(),
                                             absl::MakeSpan(dummy_operands)));
      XLS_RETURN_IF_ERROR(
          UpdateNodeMaps(n, absl::MakeSpan(dummy_operands), patch_node.name()));
      break;
    }
    case (Op::kSMul):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kUMul): {
      XLS_ASSIGN_OR_RETURN(int64_t width,
                           GetProtoBitCount(patch_node.data_type()));
      XLS_ASSIGN_OR_RETURN(n, function_base_->MakeNode<ArithOp>(
                                  SourceInfo(), dummy_operands.front(),
                                  dummy_operands[1], width, op));
      XLS_RETURN_IF_ERROR(
          UpdateNodeMaps(n, absl::MakeSpan(dummy_operands), patch_node.name()));
      break;
    }
    case (Op::kAdd):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kSub):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kULe):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kULt):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kUGt):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kNe):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kEq):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kShll): {
      XLS_ASSIGN_OR_RETURN(
          n, function_base_->MakeNode<BinOp>(
                 SourceInfo(), dummy_operands.front(), dummy_operands[1], op));
      XLS_RETURN_IF_ERROR(
          UpdateNodeMaps(n, absl::MakeSpan(dummy_operands), patch_node.name()));
      break;
    }
    case (Op::kConcat): {
      XLS_ASSIGN_OR_RETURN(
          n, function_base_->MakeNode<Concat>(SourceInfo(),
                                              absl::MakeSpan(dummy_operands)));
      XLS_RETURN_IF_ERROR(
          UpdateNodeMaps(n, absl::MakeSpan(dummy_operands), patch_node.name()));
      break;
    }
    case (Op::kOrReduce): {
      XLS_ASSIGN_OR_RETURN(n, function_base_->MakeNode<BitwiseReductionOp>(
                                  SourceInfo(), dummy_operands.front(), op));
      XLS_RETURN_IF_ERROR(
          UpdateNodeMaps(n, absl::MakeSpan(dummy_operands), patch_node.name()));
      break;
    }
    case (Op::kNot): {
      XLS_ASSIGN_OR_RETURN(n, function_base_->MakeNode<UnOp>(
                                  SourceInfo(), dummy_operands.front(), op));
      XLS_RETURN_IF_ERROR(
          UpdateNodeMaps(n, absl::MakeSpan(dummy_operands), patch_node.name()));
      break;
    }
    case (Op::kOr):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kNor):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kNand):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kAnd): {
      XLS_ASSIGN_OR_RETURN(
          n, function_base_->MakeNode<NaryOp>(
                 SourceInfo(), absl::MakeSpan(dummy_operands), op));
      XLS_RETURN_IF_ERROR(
          UpdateNodeMaps(n, absl::MakeSpan(dummy_operands), patch_node.name()));
      break;
    }
    case (Op::kSel): {
      std::vector<Node*> cases;
      std::optional<Node*> default_value;
      if (patch_node.unique_args_size() > 0 &&
          patch_node.unique_args(0).has_default_value()) {
        default_value = dummy_operands.back();
        dummy_operands.pop_back();
      }
      for (auto it = dummy_operands.begin() + 1; it != dummy_operands.end();
           ++it) {
        cases.push_back(*it);
      }
      XLS_ASSIGN_OR_RETURN(n, function_base_->MakeNode<Select>(
                                  SourceInfo(), dummy_operands.front(),
                                  absl::MakeSpan(cases), default_value));
      XLS_RETURN_IF_ERROR(
          UpdateNodeMaps(n, absl::MakeSpan(dummy_operands), patch_node.name()));
      break;
    }
    case (Op::kOneHotSel): {
      std::vector<Node*> cases;
      for (auto it = dummy_operands.begin() + 1; it != dummy_operands.end();
           ++it) {
        cases.push_back(*it);
      }
      XLS_ASSIGN_OR_RETURN(
          n, function_base_->MakeNode<OneHotSelect>(
                 SourceInfo(), dummy_operands.front(), absl::MakeSpan(cases)));
      XLS_RETURN_IF_ERROR(
          UpdateNodeMaps(n, absl::MakeSpan(dummy_operands), patch_node.name()));
      break;
    }
    case (Op::kParam): {
      XLS_ASSIGN_OR_RETURN(int64_t bit_count,
                           GetProtoBitCount(patch_node.data_type()));
      XLS_ASSIGN_OR_RETURN(
          n,
          function_base_->MakeNodeWithName<Param>(
              SourceInfo(), function_base_->package()->GetBitsType(bit_count),
              patch_node.name()));
      patch_to_ir_node_map_[patch_node.name()] = n->GetName();
      break;
    }
    default:
      return absl::InvalidArgumentError("Invalid operation");
  }
  inserted_node_names_.insert(n->GetName());
  return absl::OkStatus();
}
absl::Status PatchIr::ApplyInsertPath(
    const xls_eco::EdgeEditPathProto& edge_insert) {
  const xls_eco::EdgeProto& patch_edge = edge_insert.edge();
  XLS_ASSIGN_OR_RETURN(Node * from_node,
                       PatchContainsNode(patch_edge.from_node()).ok()
                           ? function_base_->GetNode(
                                 patch_to_ir_node_map_[patch_edge.from_node()])
                           : function_base_->GetNode(patch_edge.from_node()));
  XLS_ASSIGN_OR_RETURN(
      Node * to_node,
      PatchContainsNode(patch_edge.to_node()).ok()
          ? function_base_->GetNode(patch_to_ir_node_map_[patch_edge.to_node()])
          : function_base_->GetNode(patch_edge.to_node()));
  uint position;
  if (commiutative_edge_index_map_.find({to_node, patch_edge.index()}) !=
      commiutative_edge_index_map_.end()) {
    position = commiutative_edge_index_map_.at({to_node, patch_edge.index()});
  } else {
    position = patch_edge.index();
  }
  Node* node_to_remove = to_node->operands()[position];
  XLS_RETURN_IF_ERROR(
      to_node->ReplaceOperandNumber(position, from_node, false));
  XLS_RETURN_IF_ERROR(function_base_->RemoveNode(node_to_remove));
  auto it = std::remove(dummy_nodes_map_[to_node].begin(),
                        dummy_nodes_map_[to_node].end(), node_to_remove);
  dummy_nodes_map_[to_node].erase(it, dummy_nodes_map_[to_node].end());
  return absl::OkStatus();
}

absl::Status PatchIr::ApplyUpdatePath(
    const xls_eco::NodeEditPathProto& node_update) {
  patch_to_ir_node_map_[node_update.updated_node().name()] =
      node_update.node().name();
  return absl::OkStatus();
}
absl::Status PatchIr::ApplyUpdatePath(
    const xls_eco::EdgeEditPathProto& edge_update) {
  if (edge_update.edge().index() != edge_update.updated_edge().index()) {
    XLS_ASSIGN_OR_RETURN(
        Node * n,
        function_base_->GetNode(
            patch_to_ir_node_map_[edge_update.updated_edge().to_node()]));
    commiutative_edge_index_map_[{n, edge_update.edge().index()}] =
        edge_update.updated_edge().index();
  }

  return absl::OkStatus();
}
absl::Status PatchIr::IsolateReturnNode() {
  for (Node* n : function_base_->nodes()) {
    if (function_base_->HasImplicitUse(n)) {
      XLS_ASSIGN_OR_RETURN(
          dummy_return_node_,
          function_base_->MakeNode<Literal>(
              SourceInfo(), Value(UBits(0, n->BitCountOrDie()))));
      XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(dummy_return_node_));
      std::cout << "Isolated return node.\n";
      return absl::OkStatus();
    }
  }
  return absl::InternalError("No return node found");
}

absl::Status PatchIr::RestoreReturnNode() {
  Node* return_node =
      PatchContainsNode(patch_.return_node().name()).ok()
          ? function_base_
                ->GetNode(patch_to_ir_node_map_[patch_.return_node().name()])
                .value()
          : function_base_->GetNode(patch_.return_node().name()).value();
  XLS_RETURN_IF_ERROR(dummy_return_node_->ReplaceUsesWith(return_node));
  XLS_RETURN_IF_ERROR(function_base_->RemoveNode(dummy_return_node_));
  std::cout << "Reestablished return node: " << return_node->GetName() << '\n';
  return absl::OkStatus();
}
absl::Status PatchIr::ValidatePatch() {
  for (const auto& key : dummy_nodes_map_) {
    Node* n = key.first;
    const std::vector<Node*>& d = key.second;
    if (!d.empty()) {
      std::cout << "Warning! Dummy nodes in IR -> " << n->GetName()
                << " -> Related nodes: \n";
      for (Node* node : d) {
        std::cout << node->GetName() << "\n";
      }
    }
  }
  return absl::OkStatus();
}

absl::Status PatchIr::PatchSchedule(const PipelineSchedule& schedule) {
  XLS_ASSIGN_OR_RETURN(
      SchedulingOptionsFlagsProto scheduling_options_flags_proto,
      GetSchedulingOptionsFlagsProto());
  XLS_ASSIGN_OR_RETURN(
      SchedulingOptions scheduling_options,
      SetUpSchedulingOptions(scheduling_options_flags_proto, package_));
  SchedulingOptions tmp_scheduling_options = scheduling_options;
  XLS_ASSIGN_OR_RETURN(DelayEstimator * delay_estimator,
                       SetUpDelayEstimator(scheduling_options_flags_proto));
  decltype(package_->GetNodeCount()) constraint_count = 0;
  for (const auto& [node, cycle] : schedule.GetCycleMap()) {
    if (inserted_node_names_.contains(node->GetName())) {
      std::cout << "Skipping constraint for node: " << node->GetName()
                << " due to node being newly inserted.\n";
      continue;
    }
    if (std::find(function_base_->nodes().begin(),
                  function_base_->nodes().end(),
                  node) == function_base_->nodes().end()) {
      std::cout << "Skipping constraint for node: " << node->GetName()
                << " due to node not found in IR.\n";
      continue;
    }
    if (node->op() == Op::kLiteral) {
      std::cout << "Skipping constraint for node: " << node->GetName()
                << " due to node being a literal.\n";
      continue;
    }
    tmp_scheduling_options.add_constraint(NodeInCycleConstraint(node, cycle));
    // check if schedule is feasible; if not, then we need remove the constraint
    if (!RunPipelineSchedule(function_base_, *delay_estimator,
                             tmp_scheduling_options)
             .ok()) {
      tmp_scheduling_options.clear_constraints();
      std::cout << "Skipping constraint for node: " << node->GetName()
                << " due to schedule infeasibility.\n";
    } else {
      scheduling_options = tmp_scheduling_options;
      constraint_count++;
    }
    tmp_scheduling_options = scheduling_options;
    XLS_ASSIGN_OR_RETURN(schedule_,
                         RunPipelineSchedule(function_base_, *delay_estimator,
                                             scheduling_options));
  }
  XLS_RETURN_IF_ERROR(schedule_->Verify());
  std::cout << "Total nodes: " << package_->GetNodeCount();
  std::cout << "constrained nodes count: " << constraint_count;
  return absl::OkStatus();
}

absl::StatusOr<PipelineSchedule> PatchIr::GetPatchedSchedule() {
  if (schedule_.has_value()) {
    return schedule_.value();
  }
  return absl::InternalError("No schedule found");
}

absl::Status PatchIr::ExportIr(const std::string& export_path) const {
  std::string ir_data = package_->DumpIr();
  std::ofstream out_file(export_path);
  if (out_file.is_open()) {
    out_file << ir_data;
    out_file.close();
    return absl::OkStatus();
  }
  return absl::InternalError("Failed to open file: " + export_path);
}
absl::Status PatchIr::ExportScheduleProto() {
  XLS_ASSIGN_OR_RETURN(
      SchedulingOptionsFlagsProto scheduling_options_flags_proto,
      GetSchedulingOptionsFlagsProto());
  XLS_ASSIGN_OR_RETURN(DelayEstimator * delay_estimator,
                       SetUpDelayEstimator(scheduling_options_flags_proto));
  PackagePipelineSchedules package_pipeline_schedules = {
      {function_base_, schedule_.value()}};
  XLS_RETURN_IF_ERROR(
      SetTextProtoFile(absl::GetFlag(FLAGS_output_schedule_path),
                       PackagePipelineSchedulesToProto(
                           package_pipeline_schedules, *delay_estimator)));
  return absl::OkStatus();
}
bool PatchIr::CompareEditPaths(const xls_eco::EditPathProto& lhs,
                               const xls_eco::EditPathProto& rhs) {
  EditPathPriority lhs_priority =
      edit_path_priority_map_[{lhs.has_node_edit_path(), lhs.operation()}];
  EditPathPriority rhs_priority =
      edit_path_priority_map_[{rhs.has_node_edit_path(), rhs.operation()}];
  if (lhs_priority == rhs_priority) {
    return lhs.id() < rhs.id();
  }
  return lhs_priority < rhs_priority;
}
}  // namespace xls
