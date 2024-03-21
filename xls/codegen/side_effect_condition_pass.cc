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

#include "xls/codegen/side_effect_condition_pass.h"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {
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
      XLS_ASSIGN_OR_RETURN(Node * not_stage_guard,
                           block->MakeNode<xls::UnOp>(/*loc=*/SourceInfo(),
                                                      stage_guard, Op::kNot));
      return block->MakeNode<xls::NaryOp>(
          /*loc=*/SourceInfo(),
          std::initializer_list<Node*>{not_stage_guard, condition}, Op::kOr);
    }
    case Op::kCover:    // Cover and trace have the same condition guard:
    case Op::kTrace: {  // stage_guard && condition
      return block->MakeNode<xls::NaryOp>(
          /*loc=*/SourceInfo(),
          std::initializer_list<Node*>{stage_guard, condition}, Op::kAnd);
    }
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Expected %s to be assert, cover, or trace.", OpToString(op)));
  }
}
}  // namespace

absl::StatusOr<bool> SideEffectConditionPass::RunInternal(
    CodegenPassUnit* unit, const CodegenPassOptions& options,
    PassResults* results) const {
  // Don't rewrite side-effecting ops for:
  //  1) functions without valid control: every input is presumed valid and the
  //     op should fire every cycle.
  //  2) combinational functions: ops should fire every cycle. Check for
  //     this by looking for a schedule. If there's no schedule, assume that
  //     we're looking at something produced by the combinational generator.
  bool changed = false;
  for (std::unique_ptr<Block>& block : unit->package->blocks()) {
    auto metadata_itr = unit->metadata.find(block.get());
    if (metadata_itr == unit->metadata.end()) {
      continue;
    }
    const CodegenMetadata& metadata = metadata_itr->second;
    bool is_function = std::holds_alternative<FunctionConversionMetadata>(
        metadata.conversion_metadata);
    if (is_function && (!options.codegen_options.valid_control().has_value() ||
                        !options.schedule.has_value())) {
      continue;
    }
    // We need to use different signals as a guard on the op's condition for
    // functions and procs. Procs have extra control signals to manage the
    // channel operations. For procs, use stage_done which is asserted when all
    // sends and receives have completed. For functions, stage_done does not
    // exist, so use pipeline_valid.
    // TODO(google/xls#1060): revisit this when function- and proc-specific
    // metadata are refactored.
    absl::Span<std::optional<Node*> const> stage_guards;
    if (is_function) {
      stage_guards = metadata.streaming_io_and_pipeline.pipeline_valid;
    } else if (metadata.streaming_io_and_pipeline.stage_done.empty()) {
      // If we're looking at a proc, stage_done is used for pipelined procs
      // and stage_valid is used for combinational procs. Check if
      // stage_done is empty- if it is, use stage_valid.
      stage_guards = metadata.streaming_io_and_pipeline.stage_valid;
    } else {
      stage_guards = metadata.streaming_io_and_pipeline.stage_done;
    }
    if (stage_guards.empty()) {
      return absl::InternalError(
          "No stage guards found for side-effecting ops.");
    }
    for (Node* node : block->nodes()) {
      XLS_ASSIGN_OR_RETURN(bool should_be_rewritten,
                           OpShouldBeRewritten(node->op()));
      if (!should_be_rewritten) {
        continue;
      }
      VLOG(3) << absl::StreamFormat("Rewriting condition for %v", *node);
      auto itr =
          metadata.streaming_io_and_pipeline.node_to_stage_map.find(node);
      XLS_RET_CHECK(itr !=
                    metadata.streaming_io_and_pipeline.node_to_stage_map.end());
      int64_t condition_stage = itr->second;
      VLOG(5) << absl::StreamFormat("Condition is in stage %d.",
                                    condition_stage);
      std::optional<Node*> stage_guard = stage_guards[condition_stage];
      XLS_RET_CHECK(stage_guard.has_value()) << absl::StreamFormat(
          "Stage guard not found for stage %d.", condition_stage);
      XLS_ASSIGN_OR_RETURN(int64_t condition_operand,
                           GetConditionOperandNumber(node));
      XLS_ASSIGN_OR_RETURN(Node * guarded_condition,
                           MakeGuardedConditionForOp(
                               node->op(), node->operand(condition_operand),
                               *stage_guard, block.get()));
      XLS_RETURN_IF_ERROR(
          node->ReplaceOperandNumber(condition_operand, guarded_condition));
      changed = true;
    }
  }
  return changed;
}
}  // namespace xls::verilog
