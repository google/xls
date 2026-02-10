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

#include "xls/codegen_v_1_5/flow_control_insertion_pass.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/container/fixed_array.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/casts.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {

absl::StatusOr<bool> FlowControlInsertionPass::InsertFlowControl(
    ScheduledBlock* block, const BlockConversionPassOptions& options) const {
  bool changed = false;

  if (block->stages().empty()) {
    return false;
  }

  if (block->stages().size() == 1) {
    const Stage& stage = block->stages().front();
    if (!IsLiteralUnsignedOne(stage.inputs_valid())) {
      XLS_RETURN_IF_ERROR(stage.inputs_valid()
                              ->ReplaceUsesWithNew<Literal>(Value(UBits(1, 1)))
                              .status());
      changed = true;
    }
    if (!IsLiteralUnsignedOne(stage.outputs_ready())) {
      XLS_RETURN_IF_ERROR(stage.outputs_ready()
                              ->ReplaceUsesWithNew<Literal>(Value(UBits(1, 1)))
                              .status());
      changed = true;
    }
    return changed;
  }

  const bool is_function =
      block->source() != nullptr && block->source()->IsFunction();

  const bool uses_valid =
      options.codegen_options.valid_control().has_value() &&
      (options.codegen_options.valid_control()->has_input_name() ||
       options.codegen_options.valid_control()->has_output_name());
  if (is_function && !uses_valid) {
    // If this block represents a function and neither gates on valid nor
    // signals valid, then the pipeline registers can just be considered
    // always-valid and always-ready.
    for (Stage& stage : block->stages()) {
      if (!IsLiteralUnsignedOne(stage.inputs_valid())) {
        XLS_RETURN_IF_ERROR(
            stage.inputs_valid()
                ->ReplaceUsesWithNew<Literal>(Value(UBits(1, 1)))
                .status());
        changed = true;
      }
      if (!IsLiteralUnsignedOne(stage.outputs_ready())) {
        XLS_RETURN_IF_ERROR(
            stage.outputs_ready()
                ->ReplaceUsesWithNew<Literal>(Value(UBits(1, 1)))
                .status());
        changed = true;
      }
    }
    return changed;
  }

  absl::FixedArray<Node*> stage_done(block->stages().size());
  for (int64_t stage_index = 0; stage_index < block->stages().size();
       ++stage_index) {
    const Stage& stage = block->stages()[stage_index];
    XLS_ASSIGN_OR_RETURN(
        stage_done[stage_index],
        block->MakeNode<NaryOp>(
            SourceInfo(),
            absl::MakeConstSpan({stage.outputs_valid(), stage.outputs_ready()}),
            Op::kAnd));
  }

  // Replace all `inputs_valid` entries with appropriate valid registers:
  // - Stage 0 has no previous stage, so its inputs are always valid.
  // - Stage i > 0: The inputs for stage i become valid if stage i-1 completes
  //                in the current cycle, and stop being valid if stage i
  //                completes in the current cycle. Otherwise, the validity
  //                doesn't change.
  //     - Therefore, the `inputs_valid` register updates if either
  //       `prev_stage_done` or `cur_stage_done` is true.
  //     - If prev_stage_done is true, it loads 1.
  //     - If cur_stage_done is true (and prev_stage_done is false), it loads 0.
  if (!IsLiteralUnsignedOne(block->stages().front().inputs_valid())) {
    XLS_RETURN_IF_ERROR(block->stages()
                            .front()
                            .inputs_valid()
                            ->ReplaceUsesWithNew<Literal>(Value(UBits(1, 1)))
                            .status());
    changed = true;
  }

  for (int64_t stage_index = 1; stage_index < block->stages().size();
       ++stage_index) {
    const Stage& stage = block->stages()[stage_index];

    Node* prev_stage_done = stage_done[stage_index - 1];
    Node* cur_stage_done = stage_done[stage_index];

    const SourceInfo& loc = stage.inputs_valid()->loc();
    std::string inputs_valid_name =
        absl::StrFormat("p%d_inputs_valid", stage_index);
    XLS_ASSIGN_OR_RETURN(
        Register * inputs_valid_reg,
        block->GetResetPort().has_value()
            ? block->AddRegisterWithZeroResetValue(
                  inputs_valid_name, stage.inputs_valid()->GetType())
            : block->AddRegister(inputs_valid_name,
                                 stage.inputs_valid()->GetType()));
    Node* data = prev_stage_done;
    std::optional<Node*> load_enable;
    if (is_function) {
      // If this is a function, then the pipeline can never stall; we can just
      // update the inputs_valid register every cycle.
      load_enable = std::nullopt;
    } else {
      XLS_ASSIGN_OR_RETURN(
          load_enable,
          block->MakeNode<NaryOp>(
              loc, absl::MakeConstSpan({prev_stage_done, cur_stage_done}),
              Op::kOr));
    }
    XLS_RETURN_IF_ERROR(block
                            ->MakeNodeWithName<RegisterWrite>(
                                loc, data, load_enable,
                                /*reset=*/block->GetResetPort(),
                                /*reg=*/inputs_valid_reg,
                                block->UniquifyNodeName(
                                    absl::StrCat(inputs_valid_name, "_write")))
                            .status());
    XLS_ASSIGN_OR_RETURN(
        RegisterRead * inputs_valid,
        block->MakeNodeWithName<RegisterRead>(
            loc, inputs_valid_reg, block->UniquifyNodeName(inputs_valid_name)));
    XLS_RETURN_IF_ERROR(stage.inputs_valid()->ReplaceUsesWith(inputs_valid));
    changed = true;
  }

  // Replace all `outputs_ready` entries with the appropriate signal:
  // - The last stage's outputs are always ready, since there is no next stage
  //   that might be stalled.
  // - Stage N's outputs are ready if stage N+1 can accept data without loss;
  //   i.e., either stage N+1's inputs are empty (filling a bubble) or stage N+1
  //   is finishing in the current cycle (not stalled).
  const Stage& last_stage = block->stages().back();
  if (!IsLiteralUnsignedOne(last_stage.outputs_ready())) {
    XLS_RETURN_IF_ERROR(last_stage.outputs_ready()
                            ->ReplaceUsesWithNew<Literal>(Value(UBits(1, 1)))
                            .status());
    changed = true;
  }
  for (int64_t stage_index = block->stages().size() - 2; stage_index >= 0;
       --stage_index) {
    const Stage& stage = block->stages()[stage_index];

    if (is_function) {
      // As currently implemented, functions can never stall and do not accept
      // output backpressure, so all stages are always ready.
      XLS_RETURN_IF_ERROR(stage.outputs_ready()
                              ->ReplaceUsesWithNew<Literal>(Value(UBits(1, 1)))
                              .status());
      changed = true;
      continue;
    }

    const Stage& next_stage = block->stages()[stage_index + 1];
    const SourceInfo& loc = stage.outputs_ready()->loc();
    XLS_ASSIGN_OR_RETURN(
        Node * next_stage_empty,
        block->MakeNode<UnOp>(loc, next_stage.inputs_valid(), Op::kNot));
    Node* next_stage_done = stage_done[stage_index + 1];
    XLS_RETURN_IF_ERROR(
        stage.outputs_ready()
            ->ReplaceUsesWithNew<NaryOp>(
                absl::MakeConstSpan({next_stage_empty, next_stage_done}),
                Op::kOr)
            .status());
    changed = true;
  }

  return changed;
}

absl::StatusOr<bool> FlowControlInsertionPass::RunInternal(
    Package* package, const BlockConversionPassOptions& options,
    PassResults* results) const {
  bool changed = false;
  for (const std::unique_ptr<Block>& block : package->blocks()) {
    if (block->IsScheduled()) {
      ScheduledBlock* scheduled_block =
          absl::down_cast<ScheduledBlock*>(block.get());
      XLS_ASSIGN_OR_RETURN(bool changed_block,
                           InsertFlowControl(scheduled_block, options));
      changed |= changed_block;
    }
  }
  return changed;
}

}  // namespace xls::codegen
