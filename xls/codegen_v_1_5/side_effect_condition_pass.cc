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

#include "xls/codegen_v_1_5/side_effect_condition_pass.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen/conversion_utils.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {
namespace {

absl::StatusOr<bool> OpShouldBeRewritten(Op op) {
  if (!OpIsSideEffecting(op)) {
    return false;
  }
  // Channel, port, param, state, register, and instantiation operations are
  // handled elsewhere. Gate ops are special and not conditional, so we ignore
  // them here. That leaves the following ops to rewrite.
  switch (op) {
    case Op::kAssert:
    case Op::kCover:
    case Op::kTrace:
      return true;

    case Op::kReceive:
    case Op::kSend:
    case Op::kInputPort:
    case Op::kOutputPort:
    case Op::kParam:
    case Op::kNext:
    case Op::kRegisterRead:
    case Op::kRegisterWrite:
    case Op::kInstantiationOutput:
    case Op::kInstantiationInput:
    case Op::kGate:
      return false;
    default:
      return absl::InternalError(
          absl::StrFormat("Unexpected side-effecting op %s", OpToString(op)));
  }
}

absl::StatusOr<int64_t> GetConditionOperandNumber(Node* node) {
  switch (node->op()) {
    case Op::kAssert:
      return ::xls::Assert::kConditionOperand;
    case Op::kCover:
      return ::xls::Cover::kConditionOperand;
    case Op::kTrace:
      return ::xls::Trace::kConditionOperand;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Expected %v to be assert, cover, or trace.", *node));
  }
}

absl::StatusOr<Node*> MakeGuardedConditionForOp(Op op, Node* condition,
                                                Node* stage_guard,
                                                Block* block) {
  XLS_RET_CHECK(stage_guard->GetType()->IsBits() &&
                stage_guard->GetType()->AsBitsOrDie()->bit_count() == 1);
  switch (op) {
    case Op::kAssert: {  // Asserts are !stage_guard || condition
                         //  [|| asserted(reset)]
      XLS_ASSIGN_OR_RETURN(Node * not_stage_guard,
                           block->MakeNode<xls::UnOp>(/*loc=*/SourceInfo(),
                                                      stage_guard, Op::kNot));
      std::vector<Node*> new_conditions = {not_stage_guard, condition};
      XLS_ASSIGN_OR_RETURN(std::optional<Node*> reset_asserted,
                           verilog::ResetAsserted(block));
      if (reset_asserted.has_value()) {
        new_conditions.push_back(*reset_asserted);
      }
      return NaryOrIfNeeded(block, new_conditions);
    }
    case Op::kCover:    // Cover and trace have the same condition guard:
    case Op::kTrace: {  // stage_guard && condition [&& !asserted(reset)]
      std::vector<Node*> new_conditions = {stage_guard, condition};
      XLS_ASSIGN_OR_RETURN(std::optional<Node*> reset_not_asserted,
                           verilog::ResetNotAsserted(block));
      if (reset_not_asserted.has_value()) {
        new_conditions.push_back(*reset_not_asserted);
      }
      return NaryAndIfNeeded(block, new_conditions);
    }
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Expected %s to be assert, cover, or trace.", OpToString(op)));
  }
}

}  // namespace

absl::StatusOr<bool> SideEffectConditionPass::RunInternal(
    Package* package, const BlockConversionPassOptions& options,
    PassResults* results) const {
  bool changed = false;
  for (std::unique_ptr<Block>& block : package->blocks()) {
    if (!block->IsScheduled()) {
      continue;
    }
    ScheduledBlock* scheduled_block = down_cast<ScheduledBlock*>(block.get());

    // Don't rewrite side-effecting ops for functions without valid control:
    // every input is presumed valid and the op should fire every cycle.
    bool is_function = scheduled_block->source() != nullptr &&
                       scheduled_block->source()->IsFunction();
    if (is_function && !options.codegen_options.valid_control().has_value()) {
      continue;
    }

    absl::FixedArray<Node*> internal_condition_by_stage(
        scheduled_block->stages().size(), nullptr);
    for (Node* node : scheduled_block->nodes()) {
      XLS_ASSIGN_OR_RETURN(bool should_be_rewritten,
                           OpShouldBeRewritten(node->op()));
      if (!should_be_rewritten || !scheduled_block->IsStaged(node)) {
        continue;
      }
      VLOG(3) << absl::StreamFormat("Rewriting condition for %v", *node);
      XLS_ASSIGN_OR_RETURN(int64_t stage_index,
                           scheduled_block->GetStageIndex(node));
      VLOG(5) << absl::StreamFormat("Node %s is in stage %d.", node->GetName(),
                                    stage_index);
      if (internal_condition_by_stage[stage_index] == nullptr) {
        if (is_function || options.codegen_options.generate_combinational()) {
          XLS_ASSIGN_OR_RETURN(
              internal_condition_by_stage[stage_index],
              scheduled_block->MakeNodeWithName<NaryOp>(
                  SourceInfo(),
                  absl::MakeConstSpan(
                      {scheduled_block->stages()[stage_index].inputs_valid(),
                       scheduled_block->stages()[stage_index]
                           .active_inputs_valid()}),
                  Op::kAnd,
                  absl::StrFormat("p%d_all_inputs_valid", stage_index)));
        } else {
          XLS_ASSIGN_OR_RETURN(
              internal_condition_by_stage[stage_index],
              scheduled_block->MakeNodeWithName<NaryOp>(
                  SourceInfo(),
                  absl::MakeConstSpan(
                      {scheduled_block->stages()[stage_index].outputs_valid(),
                       scheduled_block->stages()[stage_index].outputs_ready()}),
                  Op::kAnd, absl::StrFormat("p%d_stage_done", stage_index)));
        }
      }

      XLS_ASSIGN_OR_RETURN(int64_t condition_operand,
                           GetConditionOperandNumber(node));
      XLS_ASSIGN_OR_RETURN(
          Node * guarded_condition,
          MakeGuardedConditionForOp(
              node->op(), node->operand(condition_operand),
              internal_condition_by_stage[stage_index], block.get()));
      XLS_RETURN_IF_ERROR(
          node->ReplaceOperandNumber(condition_operand, guarded_condition));
      changed = true;
    }
  }
  return changed;
}
}  // namespace xls::codegen
