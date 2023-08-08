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

#include "xls/codegen/assert_condition_pass.h"

#include <cstdint>
#include <initializer_list>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
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
absl::StatusOr<bool> AssertConditionPass::RunInternal(
    CodegenPassUnit* unit, const CodegenPassOptions& options,
    PassResults* results) const {
  // Don't rewrite assertions for:
  //  1) functions without valid control: every input is presumed valid and the
  //     assertion should fire every cycle.
  //  2) combinational functions: assertions should fire every cycle. Check for
  //     this by looking for a schedule. If there's no schedule, assume that
  //     we're looking at something produced by the combinational generator.
  bool is_function = std::holds_alternative<FunctionConversionMetadata>(
      unit->conversion_metadata);
  if (is_function && (!options.codegen_options.valid_control().has_value() ||
                      !options.schedule.has_value())) {
    return false;
  }
  bool changed = false;
  Block* const block = unit->block;
  const StreamingIOPipeline& streaming_io = unit->streaming_io_and_pipeline;
  // We need to use different signals as a guard on the assert condition for
  // functions and procs. Procs have extra control signals to manage the channel
  // operations. For procs, use stage_done which is asserted when all sends and
  // receives have completed. For functions, stage_done does not exist, so use
  // pipeline_valid.
  // TODO(google/xls#1060): revisit this when function- and proc-specific
  // metadata are refactored.
  const std::vector<Node*>& stage_guards =
      is_function
          ? streaming_io.pipeline_valid
          // If we're looking at a proc, stage_done is used for pipelined procs
          // and stage_valid is used for combinational procs. Check if
          // stage_done is empty- if it is, use stage_valid.
          : (streaming_io.stage_done.empty() ? streaming_io.stage_valid
                                             : streaming_io.stage_done);
  if (stage_guards.empty()) {
    return absl::InternalError("No stage guards found for assertions.");
  }
  for (Node* node : block->nodes()) {
    if (!node->Is<xls::Assert>()) {
      continue;
    }
    xls::Assert* assert_node = node->As<xls::Assert>();
    XLS_VLOG(3) << absl::StreamFormat("Rewriting condition for assert %s",
                                      assert_node->GetName());
    auto itr = streaming_io.node_to_stage_map.find(assert_node);
    XLS_RET_CHECK(itr != streaming_io.node_to_stage_map.end());
    int64_t condition_stage = itr->second;
    XLS_VLOG(5) << absl::StreamFormat("Condition is in stage %d.",
                                      condition_stage);
    Node* stage_guard = stage_guards[condition_stage];
    XLS_RET_CHECK(stage_guard->GetType()->IsBits() &&
                  stage_guard->GetType()->AsBitsOrDie()->bit_count() == 1);
    XLS_ASSIGN_OR_RETURN(Node * not_stage_guard,
                         block->MakeNode<xls::UnOp>(/*loc=*/SourceInfo(),
                                                    stage_guard, Op::kNot));
    XLS_ASSIGN_OR_RETURN(Node * new_condition,
                         block->MakeNode<xls::NaryOp>(
                             /*loc=*/SourceInfo(),
                             std::initializer_list<Node*>{
                                 not_stage_guard, assert_node->condition()},
                             Op::kOr));
    XLS_RETURN_IF_ERROR(assert_node->ReplaceOperandNumber(
        xls::Assert::kConditionOperand, new_condition));
    changed = true;
  }
  return changed;
}
}  // namespace xls::verilog
