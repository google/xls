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
#include <string>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/fixed_array.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen/conversion_utils.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/status/ret_check.h"
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

absl::StatusOr<Node*> GetOrCreateResetDrainValid(
    ScheduledBlock* scheduled_block, int64_t stage_index, bool has_input_flops,
    std::vector<Node*>& reset_drain_valid_chain) {
  int64_t drain_stages = stage_index + (has_input_flops ? 1 : 0);
  if (drain_stages == 0) {
    return scheduled_block->stages()[stage_index].inputs_valid();
  }
  if (reset_drain_valid_chain.empty()) {
    XLS_ASSIGN_OR_RETURN(Node * lit_one,
                         scheduled_block->MakeNode<xls::Literal>(
                             SourceInfo(), Value(UBits(1, 1))));
    reset_drain_valid_chain.push_back(lit_one);
  }
  while (reset_drain_valid_chain.size() <= drain_stages) {
    int64_t hop = reset_drain_valid_chain.size();
    int64_t valid_stage_index = has_input_flops ? (hop - 1) : hop;
    std::string reg_name =
        absl::StrFormat("p%d_inputs_valid", valid_stage_index);
    XLS_ASSIGN_OR_RETURN(
        Register * valid_reg,
        scheduled_block->AddRegisterWithZeroResetValue(
            reg_name, scheduled_block->package()->GetBitsType(1)));
    XLS_RETURN_IF_ERROR(
        scheduled_block
            ->MakeNodeWithName<RegisterWrite>(
                SourceInfo(), /*data=*/reset_drain_valid_chain.back(),
                /*load_enable=*/std::nullopt,
                /*reset=*/scheduled_block->GetResetPort(),
                /*reg=*/valid_reg, absl::StrCat(reg_name, "_write"))
            .status());
    XLS_ASSIGN_OR_RETURN(RegisterRead * valid_read,
                         scheduled_block->MakeNodeWithName<RegisterRead>(
                             SourceInfo(), valid_reg, reg_name));
    reset_drain_valid_chain.push_back(valid_read);
  }
  return reset_drain_valid_chain[drain_stages];
}

}  // namespace

absl::StatusOr<bool> SideEffectConditionPass::RunInternal(
    Package* package, const BlockConversionPassOptions& options,
    PassResults* results, BlockConversionContext& context) const {
  bool changed = false;
  for (std::unique_ptr<Block>& block : package->blocks()) {
    if (!block->IsScheduled()) {
      continue;
    }
    ScheduledBlock* scheduled_block =
        absl::down_cast<ScheduledBlock*>(block.get());

    // For functions, side-effecting ops are rewritten according to the
    // pipeline and valid/reset configuration:
    // - Functions with `valid_control` (with or without reset): rewritten to
    //   be gated by the stage's `inputs_valid` signal (and reset if present).
    // - Pipelined functions without `valid_control` but with reset: rewritten
    //   to be gated by a synthesized reset-drain valid chain (and reset) so ops
    //   do not fire on transient post-reset pipeline state.
    // - Combinational functions without `valid_control`, or pipelined functions
    //   without `valid_control` and without reset: not rewritten, as every
    //   input is presumed valid on every cycle.
    bool is_function = scheduled_block->source() != nullptr &&
                       scheduled_block->source()->IsFunction();
    bool has_valid_control =
        options.codegen_options.valid_control().has_value();
    if (is_function && !has_valid_control &&
        (options.codegen_options.generate_combinational() ||
         !scheduled_block->GetResetPort().has_value())) {
      continue;
    }

    const bool has_input_flops =
        is_function && options.codegen_options.flop_inputs() &&
        !scheduled_block->source()->AsFunctionOrDie()->params().empty();
    std::vector<Node*> reset_drain_valid_chain;

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
          Node* stage_inputs_valid =
              scheduled_block->stages()[stage_index].inputs_valid();
          if (is_function && !has_valid_control) {
            XLS_ASSIGN_OR_RETURN(stage_inputs_valid,
                                 GetOrCreateResetDrainValid(
                                     scheduled_block, stage_index,
                                     has_input_flops, reset_drain_valid_chain));
          }
          XLS_ASSIGN_OR_RETURN(
              internal_condition_by_stage[stage_index],
              scheduled_block->MakeNodeWithName<NaryOp>(
                  SourceInfo(),
                  absl::MakeConstSpan({stage_inputs_valid,
                                       scheduled_block->stages()[stage_index]
                                           .active_inputs_valid()}),
                  Op::kAnd,
                  absl::StrFormat("p%d_all_inputs_valid", stage_index)));
        } else {
          XLS_ASSIGN_OR_RETURN(
              internal_condition_by_stage[stage_index],
              scheduled_block->GetOrCreateStageDone(stage_index));
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
