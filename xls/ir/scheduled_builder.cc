// Copyright 2025 The XLS Authors
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

#include "xls/ir/scheduled_builder.h"

#include <cstdint>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"

namespace xls {

void ScheduledFunctionBuilder::SetCurrentStage(int64_t stage) {
  current_stage_ = stage;
}

void ScheduledFunctionBuilder::EndStage() { current_stage_++; }

void ScheduledFunctionBuilder::OnNodeAdded(Node* node) {
  node_to_stage_[node] = current_stage_;
}

BValue ScheduledFunctionBuilder::AssignNodeToStage(BValue node, int64_t stage) {
  node_to_stage_[node.node()] = stage;
  return node;
}

absl::StatusOr<ScheduledFunction*> ScheduledFunctionBuilder::Build() {
  const int64_t max_stage =
      absl::c_max_element(node_to_stage_, [](std::pair<Node*, int64_t> a,
                                             std::pair<Node*, int64_t> b) {
        return a.second < b.second;
      })->second;
  function()->AddEmptyStages(max_stage + 1);
  for (const auto& [node, stage] : node_to_stage_) {
    XLS_RETURN_IF_ERROR(function()->AddNodeToStage(stage, node).status());
  }
  XLS_ASSIGN_OR_RETURN(Function * f, FunctionBuilder::Build());
  return down_cast<ScheduledFunction*>(f);
}

absl::StatusOr<ScheduledFunction*>
ScheduledFunctionBuilder::BuildWithReturnValue(BValue return_value) {
  const int64_t max_stage =
      absl::c_max_element(node_to_stage_, [](std::pair<Node*, int64_t> a,
                                             std::pair<Node*, int64_t> b) {
        return a.second < b.second;
      })->second;
  function()->AddEmptyStages(max_stage + 1);
  for (const auto& [node, stage] : node_to_stage_) {
    XLS_RETURN_IF_ERROR(function()->AddNodeToStage(stage, node).status());
  }
  XLS_ASSIGN_OR_RETURN(Function * f,
                       FunctionBuilder::BuildWithReturnValue(return_value));
  return down_cast<ScheduledFunction*>(f);
}

void ScheduledProcBuilder::SetCurrentStage(int64_t stage) {
  current_stage_ = stage;
}

void ScheduledProcBuilder::EndStage() { current_stage_++; }

void ScheduledProcBuilder::OnNodeAdded(Node* node) {
  node_to_stage_[node] = current_stage_;
}

BValue ScheduledProcBuilder::AssignNodeToStage(BValue node, int64_t stage) {
  node_to_stage_[node.node()] = stage;
  return node;
}

absl::StatusOr<ScheduledProc*> ScheduledProcBuilder::Build() {
  const int64_t max_stage =
      absl::c_max_element(node_to_stage_, [](std::pair<Node*, int64_t> a,
                                             std::pair<Node*, int64_t> b) {
        return a.second < b.second;
      })->second;
  function()->AddEmptyStages(max_stage + 1);
  for (const auto& [node, stage] : node_to_stage_) {
    XLS_RETURN_IF_ERROR(function()->AddNodeToStage(stage, node).status());
  }
  XLS_ASSIGN_OR_RETURN(Proc * p, ProcBuilder::Build());
  return down_cast<ScheduledProc*>(p);
}

absl::StatusOr<ScheduledProc*> ScheduledProcBuilder::Build(
    absl::Span<const BValue> next_state) {
  if (!next_state.empty()) {
    if (next_state.size() != proc()->GetStateElementCount()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Number of recurrent state elements given (%d) does "
                          "not equal the number of state elements in the proc "
                          "(%d)",
                          next_state.size(), proc()->GetStateElementCount()));
    }
    if (!proc()->next_values().empty()) {
      return absl::InvalidArgumentError(
          "Cannot use Build(next_state) when also using next_value nodes.");
    }
    for (int64_t index = 0; index < next_state.size(); ++index) {
      if (GetType(next_state[index]) != GetType(GetStateParam(index))) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Recurrent state type %s does not match provided "
                            "state type %s for element %d.",
                            GetType(GetStateParam(index))->ToString(),
                            GetType(next_state[index])->ToString(), index));
      }
      Next(GetStateParam(index), next_state[index]);
    }
  }
  return Build();
}

void ScheduledBlockBuilder::StartStage(BValue stage_inputs_valid,
                                       BValue stage_outputs_ready) {
  CHECK(stage_inputs_valid.valid());
  staging_nodes_ = true;
  current_stage_inputs_valid_ = stage_inputs_valid.node();
  current_stage_outputs_ready_ = stage_outputs_ready.node();
}

void ScheduledBlockBuilder::EndStage(BValue active_inputs_valid,
                                     BValue stage_outputs_valid) {
  CHECK(active_inputs_valid.valid());
  CHECK(stage_outputs_valid.valid());
  CHECK_NE(current_stage_inputs_valid_, nullptr);
  CHECK_NE(current_stage_outputs_ready_, nullptr);

  staging_nodes_ = false;

  Stage stage{current_stage_inputs_valid_, current_stage_outputs_ready_,
              active_inputs_valid.node(), stage_outputs_valid.node()};
  current_stage_inputs_valid_ = nullptr;
  current_stage_outputs_ready_ = nullptr;
  for (Node* node : current_stage_nodes_) {
    stage.AddNode(node);
  }
  current_stage_nodes_.clear();

  block()->AddStage(stage);
  current_stage_++;
}

void ScheduledBlockBuilder::SuspendStaging() {
  CHECK(staging_nodes_);
  staging_nodes_ = false;
}

void ScheduledBlockBuilder::ResumeStaging() {
  CHECK(!staging_nodes_);
  staging_nodes_ = true;
}

absl::StatusOr<ScheduledBlock*> ScheduledBlockBuilder::Build() {
  XLS_RET_CHECK(!staging_nodes_) << "Build() called without ending all stages.";
  XLS_ASSIGN_OR_RETURN(Block * b, BlockBuilder::Build());
  return down_cast<ScheduledBlock*>(b);
}

BValue ScheduledBlockBuilder::AssignNodeToStage(BValue node, int64_t stage) {
  Node* raw_node = node.node();
  if (block()->IsStaged(raw_node)) {
    int64_t stage_index = *block()->GetStageIndex(raw_node);
    if (stage_index == stage) {
      // Node is already in the correct stage, so we're done.
      return node;
    }
    CHECK_OK(block()->RemoveNodeFromStage(raw_node));
  }
  CHECK_OK(block()->AddNodeToStage(stage, node.node()));
  return node;
}

BValue ScheduledBlockBuilder::RemoveNodeFromStage(BValue node) {
  CHECK_OK(block()->RemoveNodeFromStage(node.node()));
  return node;
}

void ScheduledBlockBuilder::OnNodeAdded(Node* node) {
  if (staging_nodes_) {
    current_stage_nodes_.push_back(node);
  }
}

}  // namespace xls
