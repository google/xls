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

#include "xls/contrib/xlscc/generate_fsm.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/translator_types.h"
#include "xls/contrib/xlscc/xlscc_logging.h"
#include "xls/ir/nodes.h"
#include "xls/ir/source_location.h"

namespace xlscc {

NewFSMGenerator::NewFSMGenerator(TranslatorTypeInterface& translator,
                                 DebugIrTraceFlags debug_ir_trace_flags)
    : GeneratorBase(translator), debug_ir_trace_flags_(debug_ir_trace_flags) {}

absl::Status NewFSMGenerator::SetupNewFSMGenerationContext(
    const GeneratedFunction& func, NewFSMLayout& layout,
    const xls::SourceInfo& body_loc) {
  int64_t slice_index = 0;
  for (const GeneratedFunctionSlice& slice : func.slices) {
    layout.slice_by_index[slice_index] = &slice;
    layout.index_by_slice[&slice] = slice_index;

    if (slice.after_op != nullptr) {
      layout.slice_index_by_after_op[slice.after_op] = slice_index;
    }

    for (const ContinuationValue& continuation_out : slice.continuations_out) {
      layout.output_slice_index_by_value[&continuation_out] = slice_index;
    }
    ++slice_index;
  }

  return absl::OkStatus();
}

absl::StatusOr<NewFSMLayout> NewFSMGenerator::LayoutNewFSM(
    const GeneratedFunction& func, const xls::SourceInfo& body_loc) {
  NewFSMLayout ret;

  XLS_RETURN_IF_ERROR(SetupNewFSMGenerationContext(func, ret, body_loc));

  // Record transitions across activations
  // TODO(seanhaskell): Add from last to first for statics
  for (const GeneratedFunctionSlice& slice : func.slices) {
    if (slice.after_op != nullptr &&
        slice.after_op->op == OpType::kLoopEndJump) {
      const int64_t end_jump_slice_index = ret.index_by_slice.at(&slice);
      XLSCC_CHECK_GE(end_jump_slice_index, 1, body_loc);
      const IOOp* const loop_begin_op = slice.after_op->loop_op_paired;
      XLSCC_CHECK_NE(loop_begin_op, nullptr, body_loc);
      const int64_t begin_slice_index =
          ret.slice_index_by_after_op.at(loop_begin_op);

      // Feedback is from the slice after begin to the slice before end jump
      NewFSMActivationTransition transition;
      transition.from_slice = end_jump_slice_index - 1;
      transition.to_slice = begin_slice_index;
      ret.transition_by_slice_from_index[transition.from_slice] = transition;
      ret.state_transitions.push_back(transition);
      ret.all_jump_from_slice_indices.push_back(transition.from_slice);
    }
  }

  XLS_RETURN_IF_ERROR(LayoutNewFSMStates(ret, func, body_loc));

  XLS_RETURN_IF_ERROR(LayoutValuesToSaveForNewFSMStates(ret, func, body_loc));

  // Remove unused states
  std::vector<NewFSMState> all_states = std::move(ret.states);
  ret.states.clear();
  for (NewFSMState& state : all_states) {
    bool ignore_state = false;
    for (const JumpInfo& jump_info : state.jumped_from_slice_indices) {
      if (jump_info.count < 2) {
        ignore_state = true;
        continue;
      }
    }

    if (ignore_state) {
      continue;
    }

    ret.states.push_back(std::move(state));
  }

  if (debug_ir_trace_flags_ & DebugIrTraceFlags_FSMStates) {
    LOG(INFO) << "FSM states after filtering:";
    PrintNewFSMStates(ret);
  }

  return ret;
}
absl::Status NewFSMGenerator::LayoutNewFSMStates(
    NewFSMLayout& layout, const GeneratedFunction& func,
    const xls::SourceInfo& body_loc) {
  // Get inputs for each state, resolving phis
  // Elaborates 3 iterations of each loop by taking each jump twice
  // This means that nested loops produce 3^N states, where N is the number
  // of nesting levels
  // The third iteration is filtered out later, but is necessary to
  // expose feedbacks from the second iteration to itself.

  std::vector<JumpInfo> jumped_from_slice;
  absl::flat_hash_map<const ContinuationValue*, int64_t> step_produced_by_value;

  int64_t step = 0;

  for (int64_t slice_index = 0; slice_index < func.slices.size(); ++step) {
    NewFSMState& new_state =
        layout.states.emplace_back(NewFSMState{.slice_index = slice_index});

    const GeneratedFunctionSlice* slice = layout.slice_by_index.at(slice_index);

    // Get current input for each parameter (resolving phis)
    absl::flat_hash_map<xls::Param*, std::vector<const ContinuationInput*>>
        inputs_by_param;
    for (const ContinuationInput& continuation_in : slice->continuations_in) {
      inputs_by_param[continuation_in.input_node].push_back(&continuation_in);
    }

    for (const auto& [input_param, continuation_ins] : inputs_by_param) {
      XLSCC_CHECK_GE(continuation_ins.size(), 1, body_loc);
      int64_t latest_step_produced = -1;
      const ContinuationInput* latest_continuation_in = nullptr;
      for (const ContinuationInput* continuation_in : continuation_ins) {
        const ContinuationValue* continuation_out =
            continuation_in->continuation_out;
        // Ignore if not produced yet
        if (!step_produced_by_value.contains(continuation_out)) {
          continue;
        }
        const int64_t step_produced =
            step_produced_by_value.at(continuation_out);
        XLSCC_CHECK_NE(latest_step_produced, step_produced, body_loc);
        if (step_produced > latest_step_produced) {
          latest_step_produced = step_produced;
          latest_continuation_in = continuation_in;
        }
      }
      XLSCC_CHECK_GE(latest_step_produced, 0, body_loc);
      XLSCC_CHECK_NE(latest_continuation_in, nullptr, body_loc);
      new_state.current_inputs_by_input_param[input_param] =
          latest_continuation_in->continuation_out;
    }

    for (const ContinuationValue& continuation_out : slice->continuations_out) {
      step_produced_by_value[&continuation_out] = step;
    }

    for (const JumpInfo& jump_info : jumped_from_slice) {
      new_state.jumped_from_slice_indices.push_back(jump_info);
    }

    // Next slice
    const bool is_slice_top_of_stack =
        !jumped_from_slice.empty() &&
        jumped_from_slice.back().from_slice == slice_index;

    if (is_slice_top_of_stack) {
      XLSCC_CHECK(!jumped_from_slice.empty(), body_loc);
      JumpInfo& jump_info = jumped_from_slice.back();
      if (jump_info.count == 1) {
        jumped_from_slice.pop_back();
        ++slice_index;
      } else {
        --jump_info.count;
        slice_index = jump_info.to_slice;
      }
    } else {
      if (layout.transition_by_slice_from_index.contains(slice_index)) {
        const NewFSMActivationTransition& transition =
            layout.transition_by_slice_from_index.at(slice_index);
        slice_index = transition.to_slice;
        jumped_from_slice.push_back(JumpInfo{
            .from_slice = transition.from_slice,
            .to_slice = transition.to_slice,
            .count = 2,
        });
      } else {
        ++slice_index;
      }
    }
  }

  XLSCC_CHECK(jumped_from_slice.empty(), body_loc);

  return absl::OkStatus();
}

absl::Status NewFSMGenerator::LayoutValuesToSaveForNewFSMStates(
    NewFSMLayout& layout, const GeneratedFunction& func,
    const xls::SourceInfo& body_loc) {
  // Fill in values to save after each state, in case of an activation
  // transition.
  // States are now flattened and linear, so this can be handled as if
  // phis / jumps / loops don't exist, with a simple reference count for each
  // continuation value, initialized when the value is produced to the total
  // number of references in the graph.

  absl::flat_hash_map<const ContinuationValue*, int64_t>
      total_input_count_by_value;

  for (const NewFSMState& state : layout.states) {
    for (const auto& [input_param, continuation_out] :
         state.current_inputs_by_input_param) {
      ++total_input_count_by_value[continuation_out];
    }
  }

  absl::flat_hash_map<const ContinuationValue*, int64_t>
      remaining_input_count_by_value;

  for (NewFSMState& state : layout.states) {
    const GeneratedFunctionSlice* slice =
        layout.slice_by_index.at(state.slice_index);
    // Decrement input counts
    for (const auto& [input_param, continuation_out] :
         state.current_inputs_by_input_param) {
      XLSCC_CHECK(remaining_input_count_by_value.contains(continuation_out),
                  body_loc);
      --remaining_input_count_by_value[continuation_out];
    }

    // Record output counts, including all future uses
    for (const ContinuationValue& continuation_out : slice->continuations_out) {
      // Due to virtual unrolling, continuation values may be produced
      // multiple times
      if (remaining_input_count_by_value.contains(&continuation_out)) {
        continue;
      }
      remaining_input_count_by_value[&continuation_out] =
          total_input_count_by_value.at(&continuation_out);
    }

    for (const auto& [value, count] : remaining_input_count_by_value) {
      if (count == 0) {
        continue;
      }
      state.values_to_save.insert(value);
    }
  }

  for (const auto& [value, count] : remaining_input_count_by_value) {
    XLSCC_CHECK_EQ(count, 0, body_loc);
  }

  return absl::OkStatus();
}

void NewFSMGenerator::PrintNewFSMStates(const NewFSMLayout& layout) {
  auto jump_infos_string =
      [](const std::vector<JumpInfo>& jump_infos) -> std::string {
    std::vector<std::string> ret;
    for (const JumpInfo& jump_info : jump_infos) {
      std::string str;
      absl::StrAppendFormat(&str, "{%li,c = %li}", jump_info.from_slice,
                            jump_info.count);
      ret.push_back(str);
    }
    return absl::StrJoin(ret, ",");
  };

  LOG(INFO) << absl::StrFormat("States %li, inputs:", layout.states.size());
  for (const NewFSMState& state : layout.states) {
    LOG(INFO) << absl::StrFormat(
        "  [slc %li j %s]: ", state.slice_index,
        jump_infos_string(state.jumped_from_slice_indices).c_str());
    for (const auto& [input_param, continuation_out] :
         state.current_inputs_by_input_param) {
      LOG(INFO) << absl::StrFormat(
          "    p %s: slice %li %s", input_param->name().data(),
          layout.output_slice_index_by_value.at(continuation_out),
          continuation_out->name.c_str());
    }
  }
  LOG(INFO) << absl::StrFormat("States %li, values to save:",
                               layout.states.size());
  for (const NewFSMState& state : layout.states) {
    LOG(INFO) << absl::StrFormat(
        "  [slc %li j %s]: ", state.slice_index,
        jump_infos_string(state.jumped_from_slice_indices).c_str());
    for (const ContinuationValue* value : state.values_to_save) {
      LOG(INFO) << absl::StrFormat(
          "    v %s slice %li", value->name.c_str(),
          layout.output_slice_index_by_value.at(value));
    }
  }
}

}  // namespace xlscc
