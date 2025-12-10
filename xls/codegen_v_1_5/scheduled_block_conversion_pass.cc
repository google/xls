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

#include "xls/codegen_v_1_5/scheduled_block_conversion_pass.h"

#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/casts.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.pb.h"

namespace xls::codegen {
namespace {

absl::StatusOr<Node*> AddPlaceholder(Block& block, std::string_view name) {
  return block.MakeNodeWithName<Literal>(SourceInfo(), Value(UBits(1, 1)),
                                         name);
}

absl::Status AddClockAndResetPorts(const verilog::CodegenOptions& options,
                                   Block& block) {
  if (!options.clock_name().has_value()) {
    return absl::InvalidArgumentError(
        "Clock name must be specified when generating a pipelined block");
  }
  XLS_RETURN_IF_ERROR(block.AddClockPort(options.clock_name().value()));

  if (options.reset().has_value()) {
    XLS_RETURN_IF_ERROR(block
                            .AddResetPort(options.reset()->name(),
                                          options.GetResetBehavior().value())
                            .status());
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> ScheduledBlockConversionPass::RunInternal(
    Package* package, const BlockConversionPassOptions& options,
    PassResults* results) const {
  for (FunctionBase* old_fb : package->GetFunctionBases()) {
    if (!old_fb->IsScheduled() ||
        (!old_fb->IsFunction() && !old_fb->IsProc())) {
      continue;
    }

    ScheduledBlock* block = down_cast<ScheduledBlock*>(package->AddBlock(
        std::make_unique<ScheduledBlock>(old_fb->name(), package)));

    XLS_RETURN_IF_ERROR(AddClockAndResetPorts(options.codegen_options, *block));

    for (int i = 0; i < old_fb->stages().size(); ++i) {
      XLS_ASSIGN_OR_RETURN(
          Node * stage_inputs_valid,
          AddPlaceholder(*block, absl::StrCat("stage_inputs_valid_", i)));
      XLS_ASSIGN_OR_RETURN(
          Node * stage_outputs_ready,
          AddPlaceholder(*block, absl::StrCat("stage_outputs_ready_", i)));
      XLS_ASSIGN_OR_RETURN(
          Node * active_inputs_valid,
          AddPlaceholder(*block, absl::StrCat("active_inputs_valid_", i)));
      XLS_ASSIGN_OR_RETURN(
          Node * stage_outputs_valid,
          block->MakeNodeWithName<NaryOp>(
              SourceInfo(),
              std::vector<Node*>{stage_inputs_valid, active_inputs_valid},
              Op::kAnd, absl::StrCat("stage_outputs_valid_", i)));

      Stage stage(stage_inputs_valid, stage_outputs_ready, active_inputs_valid,
                  stage_outputs_valid);

      stage.AddNode(active_inputs_valid);
      const Stage& old_stage = old_fb->stages()[i];
      for (Node* node : old_stage) {
        stage.AddNode(node);
      }

      stage.AddNode(stage_outputs_valid);
      block->AddStage(std::move(stage));
    }

    if (old_fb->IsFunction()) {
      auto source_fn = std::make_unique<Function>(
          absl::StrCat(old_fb->name(), "__src"), package);
      Node* return_value = down_cast<Function*>(old_fb)->return_value();
      source_fn->MoveParamsFrom(*down_cast<Function*>(old_fb));
      source_fn->set_return_type(return_value->GetType());
      block->SetSource(std::move(source_fn));
      block->SetSourceReturnValue(return_value);
      block->MoveLogicFrom(*old_fb);
    } else if (old_fb->IsProc()) {
      auto source_proc = std::make_unique<Proc>(
          absl::StrCat(old_fb->name(), "__src"), package);
      source_proc->MoveNonLogicFrom(down_cast<Proc&>(*old_fb));
      block->SetSource(std::move(source_proc));
      block->MoveLogicFrom(*old_fb);
    }

    std::optional<FunctionBase*> top = package->GetTop();
    if (top.has_value() && top == old_fb) {
      XLS_RETURN_IF_ERROR(package->SetTop(block));
    }
    XLS_RETURN_IF_ERROR(package->RemoveFunctionBase(old_fb));
  }

  return true;
}

}  // namespace xls::codegen
