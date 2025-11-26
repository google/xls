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

#include "xls/ir/function_base.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "google/protobuf/text_format.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/change_listener.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_scanner.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"

namespace xls {

bool Stage::AddNode(Node* node) {
  switch (node->op()) {
    case Op::kReceive:
    case Op::kStateRead:
    case Op::kRegisterRead:
      return active_inputs_.insert(node).second;
    case Op::kSend:
    case Op::kNext:
    case Op::kRegisterWrite:
      return active_outputs_.insert(node).second;
    default:
      return logic_.insert(node).second;
  }
}

absl::StatusOr<Stage> Stage::Clone(
    const absl::flat_hash_map<Node*, Node*>& node_mapping) const {
  auto map_node = [&](Node* node) -> absl::StatusOr<Node*> {
    auto it = node_mapping.find(node);
    if (it == node_mapping.end()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Node %s not found in node mapping.", node->GetName()));
    }
    return it->second;
  };

  Stage cloned_stage;
  for (Node* node : active_inputs_) {
    XLS_ASSIGN_OR_RETURN(Node * mapped_node, map_node(node));
    cloned_stage.active_inputs_.insert(mapped_node);
  }
  for (Node* node : logic_) {
    XLS_ASSIGN_OR_RETURN(Node * mapped_node, map_node(node));
    cloned_stage.logic_.insert(mapped_node);
  }
  for (Node* node : active_outputs_) {
    XLS_ASSIGN_OR_RETURN(Node * mapped_node, map_node(node));
    cloned_stage.active_outputs_.insert(mapped_node);
  }
  if (inputs_valid_ != nullptr) {
    XLS_ASSIGN_OR_RETURN(cloned_stage.inputs_valid_, map_node(inputs_valid_));
  }
  if (outputs_ready_ != nullptr) {
    XLS_ASSIGN_OR_RETURN(cloned_stage.outputs_ready_, map_node(outputs_ready_));
  }
  if (active_inputs_valid_ != nullptr) {
    XLS_ASSIGN_OR_RETURN(cloned_stage.active_inputs_valid_,
                         map_node(active_inputs_valid_));
  }
  if (outputs_valid_ != nullptr) {
    XLS_ASSIGN_OR_RETURN(cloned_stage.outputs_valid_, map_node(outputs_valid_));
  }
  return cloned_stage;
}

std::ostream& operator<<(std::ostream& os, const FunctionBase::Kind& kind) {
  switch (kind) {
    case FunctionBase::Kind::kFunction:
      return os << "function";
    case FunctionBase::Kind::kProc:
      return os << "proc";
    case FunctionBase::Kind::kBlock:
      return os << "block";
  }
}

void FunctionBase::MoveFrom(FunctionBase& other) {
  for (std::unique_ptr<Node>& node : other.nodes_) {
    node->function_base_ = this;
    AddNodeInternal(std::move(node));
  }
  node_to_stage_ = std::move(other.node_to_stage_);

  other.nodes_.clear();
  other.node_iterators_.clear();
  other.next_values_by_state_read_.clear();
  other.params_.clear();
  other.node_to_stage_.clear();
}

std::vector<std::string> FunctionBase::AttributeIrStrings() const {
  std::vector<std::string> attribute_strings;
  if (ForeignFunctionData().has_value()) {
    std::string serialized;
    CHECK(
        google::protobuf::TextFormat::PrintToString(*ForeignFunctionData(), &serialized));
    // Triple-quoted attribute strings allow for newlines.
    attribute_strings.push_back(
        absl::StrCat("ffi_proto(\"\"\"", serialized, "\"\"\")"));
  }
  if (initiation_interval_.has_value()) {
    attribute_strings.push_back(
        absl::StrFormat("initiation_interval(%d)", *initiation_interval_));
  }

  return attribute_strings;
}

absl::StatusOr<Param*> FunctionBase::GetParamByName(
    std::string_view param_name) const {
  for (Param* param : params()) {
    if (param->name() == param_name) {
      return param;
    }
  }
  return absl::NotFoundError(
      absl::StrFormat("Function '%s' does not have a parameter named '%s'",
                      name(), param_name));
}

absl::StatusOr<int64_t> FunctionBase::GetParamIndex(Param* param) const {
  auto it = std::find(params_.begin(), params_.end(), param);
  if (it == params_.end()) {
    return absl::InvalidArgumentError(
        "Given param is not a member of this function: " + param->ToString());
  }
  return std::distance(params_.begin(), it);
}

absl::Status FunctionBase::MoveParamToIndex(Param* param, int64_t index) {
  XLS_RET_CHECK_LT(index, params_.size());
  auto it = std::find(params_.begin(), params_.end(), param);
  if (it == params_.end()) {
    return absl::InvalidArgumentError(
        "Given param is not a member of this function base: " +
        param->ToString());
  }
  params_.erase(it);
  params_.insert(params_.begin() + index, param);
  return absl::OkStatus();
}

absl::StatusOr<Node*> FunctionBase::GetNodeById(int64_t id) const {
  for (Node* node : nodes()) {
    if (node->id() == id) {
      return node;
    }
  }
  return absl::NotFoundError(absl::StrFormat("No node found with id %d.", id));
}

std::optional<Node*> FunctionBase::MaybeGetNode(
    std::string_view standard_node_name) const {
  for (Node* node : nodes()) {
    if (node->GetName() == standard_node_name) {
      return node;
    }
  }
  for (auto& param : params()) {
    if (param->As<Param>()->name() == standard_node_name) {
      return param;
    }
  }
  return std::nullopt;
}

absl::StatusOr<Node*> FunctionBase::GetNode(
    std::string_view standard_node_name) const {
  if (auto node = MaybeGetNode(standard_node_name); node.has_value()) {
    return *node;
  }
  return absl::NotFoundError(
      absl::StrFormat("GetNode(%s) failed.", standard_node_name));
}

absl::Status FunctionBase::RemoveNode(Node* node) {
  XLS_RET_CHECK(node->users().empty()) << node->GetName();
  XLS_RET_CHECK(!HasImplicitUse(node)) << node->GetName();
  VLOG(4) << absl::StrFormat("Removing node from FunctionBase %s: %s", name(),
                             node->ToString());
  ++package()->transform_metrics().nodes_removed;
  std::vector<Node*> unique_operands;
  for (Node* operand : node->operands()) {
    if (!absl::c_linear_search(unique_operands, operand)) {
      unique_operands.push_back(operand);
    }
  }
  for (Node* operand : unique_operands) {
    operand->RemoveUser(node);
  }
  if (node->Is<Param>()) {
    params_.erase(std::remove(params_.begin(), params_.end(), node),
                  params_.end());
  }
  if (node->Is<StateRead>()) {
    next_values_by_state_read_.erase(node->As<StateRead>());
  }
  if (node->Is<Next>()) {
    Next* next = node->As<Next>();
    if (next->state_read()->Is<StateRead>()) {  // Could've been replaced.
      StateRead* state_read = next->state_read()->As<StateRead>();
      next_values_by_state_read_.at(state_read).erase(next);
    }
    std::erase(next_values_, next);
  }
  for (ChangeListener* listener : change_listeners_) {
    listener->NodeDeleted(node);
  }
  // Clear the name.
  if (node->HasAssignedName()) {
    XLS_RETURN_IF_ERROR(node_name_uniquer_.ReleaseIdentifier(node->GetName()));
  }
  if (IsScheduled()) {
    auto it = node_to_stage_.find(node);
    CHECK_NE(it, node_to_stage_.end());
    stages_[it->second].erase(node);
    node_to_stage_.erase(it);
  }
  auto node_it = node_iterators_.find(node);
  XLS_RET_CHECK(node_it != node_iterators_.end());
  nodes_.erase(node_it->second);
  node_iterators_.erase(node_it);
  return absl::OkStatus();
}

absl::Status FunctionBase::Accept(DfsVisitor* visitor) {
  for (Node* node : nodes()) {
    if (node->users().empty()) {
      XLS_RETURN_IF_ERROR(node->Accept(visitor));
    }
  }
  if (visitor->GetVisitedCount() < node_count()) {
    // Not all nodes were visited. This indicates a cycle. Create a separate
    // trivial DFS visitor to find the cycle.
    class CycleChecker : public DfsVisitorWithDefault {
      absl::Status DefaultHandler(Node* node) override {
        return absl::OkStatus();
      }
    };
    CycleChecker cycle_checker;
    for (Node* node : nodes()) {
      if (!cycle_checker.IsVisited(node)) {
        XLS_RETURN_IF_ERROR(node->Accept(&cycle_checker));
      }
    }
    return absl::InternalError(absl::StrFormat(
        "Expected to find cycle in function base %s, but none was found.",
        name()));
  }

  return absl::OkStatus();
}

const Function* FunctionBase::AsFunctionOrDie() const {
  CHECK(IsFunction());
  return down_cast<const Function*>(this);
}

const Proc* FunctionBase::AsProcOrDie() const {
  CHECK(IsProc());
  return down_cast<const Proc*>(this);
}

const Block* FunctionBase::AsBlockOrDie() const {
  CHECK(IsBlock());
  return down_cast<const Block*>(this);
}
Function* FunctionBase::AsFunctionOrDie() {
  CHECK(IsFunction());
  return down_cast<Function*>(this);
}

Proc* FunctionBase::AsProcOrDie() {
  CHECK(IsProc());
  return down_cast<Proc*>(this);
}

Block* FunctionBase::AsBlockOrDie() {
  CHECK(IsBlock());
  return down_cast<Block*>(this);
}

Node* FunctionBase::AddNodeInternal(std::unique_ptr<Node> node) {
  VLOG(4) << absl::StrFormat("Adding node to FunctionBase %s: %s", name(),
                             node->ToString());
  ++package()->transform_metrics().nodes_added;
  if (node->Is<Param>()) {
    params_.push_back(node->As<Param>());
  }
  if (node->Is<StateRead>()) {
    next_values_by_state_read_[node->As<StateRead>()];
  }
  if (node->Is<Next>()) {
    Next* next = node->As<Next>();
    StateRead* state_read = next->state_read()->As<StateRead>();
    next_values_.push_back(node->As<Next>());
    next_values_by_state_read_.at(state_read).insert(next);
  }
  Node* ptr = node.get();
  node_iterators_[ptr] = nodes_.insert(nodes_.end(), std::move(node));
  for (ChangeListener* listener : change_listeners_) {
    listener->NodeAdded(ptr);
  }
  return ptr;
}

/* static */ std::vector<std::string> FunctionBase::GetIrReservedWords() {
  std::vector<std::string> words(Token::GetKeywords().begin(),
                                 Token::GetKeywords().end());
  // Sort to avoid nondeterminism because GetKeywords returns a flat hashmap.
  std::sort(words.begin(), words.end());
  return words;
}

std::ostream& operator<<(std::ostream& os, const FunctionBase& function) {
  os << function.DumpIr();
  return os;
}

absl::Span<const Stage> FunctionBase::stages() const {
  CHECK(IsScheduled());
  return stages_;
}

absl::Span<Stage> FunctionBase::stages() {
  CHECK(IsScheduled());
  return absl::MakeSpan(stages_);
}

void FunctionBase::AddStage(Stage stage) {
  CHECK(IsScheduled());
  int64_t stage_index = stages_.size();
  auto add_node_to_map = [&](Node* node) {
    CHECK(!node_to_stage_.contains(node))
        << "Node " << node->GetName() << " is already in a stage.";
    node_to_stage_[node] = stage_index;
  };
  for (Node* node : stage.active_inputs()) {
    add_node_to_map(node);
  }
  for (Node* node : stage.logic()) {
    add_node_to_map(node);
  }
  for (Node* node : stage.active_outputs()) {
    add_node_to_map(node);
  }
  stages_.push_back(std::move(stage));
}

void FunctionBase::AddEmptyStages(int64_t n) {
  CHECK(IsScheduled());
  stages_.resize(stages_.size() + n);
}

void FunctionBase::ClearStages() {
  CHECK(IsScheduled());
  stages_.clear();
  node_to_stage_.clear();
}

absl::StatusOr<int64_t> FunctionBase::GetStageIndex(Node* node) const {
  XLS_RET_CHECK(IsScheduled());
  auto it = node_to_stage_.find(node);
  if (it == node_to_stage_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("Node %s not found in any stage of FunctionBase %s",
                        node->GetName(), name()));
  }
  return it->second;
}

absl::StatusOr<bool> FunctionBase::AddNodeToStage(int64_t stage_index,
                                                  Node* node) {
  XLS_RET_CHECK(IsScheduled());
  XLS_RET_CHECK_LE(stage_index, stages_.size());
  if (stage_index == stages_.size()) {
    stages_.push_back({});
  }
  bool added = stages_[stage_index].AddNode(node);
  if (added) {
    XLS_RET_CHECK(!node_to_stage_.contains(node))
        << "Node " << node->GetName() << " is already in a stage.";
    node_to_stage_[node] = stage_index;
  }
  return added;
}

absl::Status FunctionBase::RebuildStageSideTables() {
  if (!IsScheduled()) {
    return absl::OkStatus();
  }
  node_to_stage_.clear();
  for (int64_t i = 0; i < stages_.size(); ++i) {
    for (Node* node : stages_[i].active_inputs()) {
      node_to_stage_[node] = i;
    }
    for (Node* node : stages_[i].logic()) {
      node_to_stage_[node] = i;
    }
    for (Node* node : stages_[i].active_outputs()) {
      node_to_stage_[node] = i;
    }
  }
  return absl::OkStatus();
}

absl::Status FunctionBase::RebuildSideTables() {
  // TODO(allight): The fact that there is so much crap in the function_base
  // itself is a problem. Having next's and params' in the function base doesn't
  // make a ton of sense.
  // NB Because of above the next-values/next_values_by_state_read_ and params
  // lists are updated in proc and function respectively.
  // NB We assume that node_iterators_ never gets invalidated.
  XLS_RETURN_IF_ERROR(InternalRebuildSideTables());
  XLS_RETURN_IF_ERROR(RebuildStageSideTables());
  return absl::OkStatus();
}

}  // namespace xls
