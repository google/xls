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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/math_util.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/tracked_bvalue.h"
#include "xls/contrib/xlscc/translator_types.h"
#include "xls/contrib/xlscc/xlscc_logging.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/nodes.h"
#include "xls/ir/source_location.h"
#include "xls/ir/state_element.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"

namespace xlscc {

NewFSMGenerator::NewFSMGenerator(TranslatorTypeInterface& translator_types,
                                 TranslatorIOInterface& translator_io,
                                 DebugIrTraceFlags debug_ir_trace_flags)
    : GeneratorBase(translator_types),
      translator_io_(translator_io),
      debug_ir_trace_flags_(debug_ir_trace_flags) {}

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
          "    v %s slice %li (%li bits)", value->name.c_str(),
          layout.output_slice_index_by_value.at(value),
          value->output_node->GetType()->GetFlatBitCount());
    }
  }
}

absl::StatusOr<GenerateFSMInvocationReturn>
NewFSMGenerator::GenerateNewFSMInvocation(
    const GeneratedFunction* xls_func,
    const std::vector<TrackedBValue>& direct_in_args, xls::ProcBuilder& pb,
    const xls::SourceInfo& body_loc) {
  XLSCC_CHECK_NE(xls_func, nullptr, body_loc);
  const GeneratedFunction& func = *xls_func;

  if (func.slices.size() != (func.io_ops.size() + 1)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "New FSM is only applicable with N+1 slices, where N is the number of "
        "IO Ops. Called with %i ops and %i slices. Subroutine call incorrectly "
        "translated?",
        func.io_ops.size() + 1, xls_func->slices.size()));
  }

  NewFSMLayout layout;
  XLS_ASSIGN_OR_RETURN(layout, LayoutNewFSM(func, body_loc));

  const int64_t num_slice_index_bits =
      xls::CeilOfLog2(1 + layout.states.size());

  TrackedBValue next_activation_slice_index = pb.StateElement(
      "__next_activation_slice",
      xls::Value(xls::UBits(0, num_slice_index_bits)), body_loc);

  TrackedBValue first_slice_index =
      pb.Literal(xls::UBits(0, num_slice_index_bits), body_loc);
  TrackedBValue current_slice_index = next_activation_slice_index;

  absl::btree_multimap<const xls::StateElement*, NextStateValue>
      extra_next_state_values;

  // Also contains null bundle for sequencing non-channel ops, such as traces
  absl::btree_map<ChannelBundle, TrackedBValue> token_by_channel_bundle;
  auto token_ref_by_channel_bundle =
      [&pb, body_loc, &token_by_channel_bundle](
          const ChannelBundle& channel_bundle) -> TrackedBValue& {
    if (!token_by_channel_bundle.contains(channel_bundle)) {
      std::string channel_name = "unknown";
      if (channel_bundle.regular != nullptr) {
        channel_name = channel_bundle.regular->name();
      } else if (channel_bundle.read_request != nullptr) {
        channel_name = channel_bundle.read_request->name();
      }
      token_by_channel_bundle[channel_bundle] =
          pb.Literal(xls::Value::Token(), body_loc,
                     /*name=*/absl::StrFormat("token_%s", channel_name));
    }
    return token_by_channel_bundle.at(channel_bundle);
  };

  // Create state elements for jumps (jumped vs didn't jump yet)
  absl::flat_hash_map<int64_t, TrackedBValue> state_element_by_jump_slice_index;
  for (int64_t jump_slice_index : layout.all_jump_from_slice_indices) {
    TrackedBValue state_element =
        pb.StateElement(absl::StrFormat("__jump_state_%li", jump_slice_index),
                        xls::Value(xls::UBits(0, 1)), body_loc);

    state_element_by_jump_slice_index[jump_slice_index] = state_element;
  }

  // Analyze phis
  absl::flat_hash_map<int64_t, std::vector<PhiElement>>
      phi_elements_by_param_node_id;

  XLS_ASSIGN_OR_RETURN(
      phi_elements_by_param_node_id,
      GeneratePhiConditions(layout, state_element_by_jump_slice_index, pb,
                            body_loc));

  // The value from the current activation's perspective,
  // either outputted from invoke or state element.
  absl::flat_hash_map<const ContinuationValue*, TrackedBValue>
      value_by_continuation_value;

  // TODO(seanhaskell): Re-use these by same decl
  absl::flat_hash_map<const ContinuationValue*, TrackedBValue>
      state_element_by_continuation_value;

  {
    absl::flat_hash_set<const ContinuationValue*> all_values_to_save;
    for (const NewFSMState& state : layout.states) {
      for (const ContinuationValue* value : state.values_to_save) {
        all_values_to_save.insert(value);
      }
    }

    int64_t slice_index = 0;
    for (const GeneratedFunctionSlice& slice : func.slices) {
      for (const ContinuationValue& continuation_out :
           slice.continuations_out) {
        // Create state elements only for values to save from analysis
        if (!all_values_to_save.contains(&continuation_out)) {
          continue;
        }

        TrackedBValue state_element = pb.StateElement(
            /*name=*/absl::StrFormat("__slice_%li_cont_%s_%li", slice_index,
                                     continuation_out.name,
                                     layout.index_by_slice.at(&slice)),
            xls::ZeroOfType(continuation_out.output_node->GetType()), body_loc);

        state_element_by_continuation_value[&continuation_out] = state_element;
      }
      ++slice_index;
    }
  }

  // Set values initially to state elements, so that feedbacks
  // come from state. These will be overwritten for feedforwards as slices
  // are generated.
  value_by_continuation_value = state_element_by_continuation_value;

  TrackedBValue last_op_out_value;

  for (int64_t slice_index = 0; slice_index < func.slices.size();
       ++slice_index) {
    TrackedBValue slice_active = pb.Eq(
        current_slice_index,
        pb.Literal(xls::UBits(slice_index, num_slice_index_bits)), body_loc,
        /*name=*/absl::StrFormat("slice_%li_active", slice_index));
    TrackedBValue next_slice_index =
        pb.Literal(xls::UBits(slice_index + 1, num_slice_index_bits), body_loc);
    TrackedBValue debug_did_jump = pb.Literal(xls::UBits(0, 1), body_loc);

    // To avoid needing to store the IO op's received value,
    // the after_op is always in the same activation as the invoke for the
    // function slice.
    //
    // The output value for the op doesn't need to be stored because
    // it is available from the invoke for the previous function slice.
    //
    // NOTE: After an activation transition, the function slice must be
    // invoked again
    const GeneratedFunctionSlice& slice =
        *layout.slice_by_index.at(slice_index);

    // Gather invoke params, except IO input
    std::vector<TrackedBValue> invoke_params;
    invoke_params.reserve(slice.continuations_in.size() + 1);

    // Add direct-ins to first slice params
    if (slice_index == 0) {
      invoke_params.insert(invoke_params.end(), direct_in_args.begin(),
                           direct_in_args.end());
    }

    // Order for determinism
    for (const xls::Param* param : slice.function->params()) {
      std::optional<TrackedBValue> input_value;
      XLS_ASSIGN_OR_RETURN(
          input_value,
          GenerateInputValueInContext(param, phi_elements_by_param_node_id,
                                      value_by_continuation_value, slice_index,
                                      pb, body_loc));

      if (!input_value.has_value()) {
        continue;
      }

      invoke_params.push_back(input_value.value());
    }

    const bool loop_op = slice.after_op != nullptr &&
                         (slice.after_op->op == OpType::kLoopBegin ||
                          slice.after_op->op == OpType::kLoopEndJump);

    if (slice.after_op != nullptr && !loop_op) {
      const IOOp* after_op = slice.after_op;
      XLSCC_CHECK(after_op->op != OpType::kLoopBegin, body_loc);
      XLSCC_CHECK(after_op->op != OpType::kLoopEndJump, body_loc);
      std::optional<ChannelBundle> optional_bundle =
          translator_io_.GetChannelBundleForOp(*after_op, body_loc);
      ChannelBundle bundle = optional_bundle.value_or(ChannelBundle{});
      TrackedBValue& token = token_ref_by_channel_bundle(bundle);
      XLSCC_CHECK(last_op_out_value.valid(), body_loc);
      XLS_ASSIGN_OR_RETURN(
          GenerateIOReturn io_return,
          translator_io_.GenerateIO(*after_op, token, last_op_out_value, pb,
                                    optional_bundle,
                                    /*extra_condition=*/slice_active));
      token = io_return.token_out;

      // Add IO parameter if applicable
      if (io_return.received_value.valid()) {
        invoke_params.push_back(io_return.received_value);
      }
    }

    XLSCC_CHECK_NE(slice.function, nullptr, body_loc);
    XLSCC_CHECK_EQ(invoke_params.size(), slice.function->params().size(),
                   body_loc);
    TrackedBValue ret_tup =
        pb.Invoke(ToNativeBValues(invoke_params), slice.function, body_loc,
                  /*name=*/
                  absl::StrFormat("invoke_%s", slice.function->name()));
    XLSCC_CHECK(ret_tup.valid(), body_loc);

    // Set last_op_out_value if not the last slice
    if (slice_index < (func.slices.size() - 1)) {
      const GeneratedFunctionSlice& next_slice =
          *layout.slice_by_index.at(slice_index + 1);
      XLSCC_CHECK_NE(next_slice.after_op, nullptr, body_loc);
      TrackedBValue op_out_value = pb.TupleIndex(
          ret_tup, ret_tup.GetType()->AsTupleOrDie()->size() - 1, body_loc,
          /*name=*/
          absl::StrFormat("%s_io_out_value", slice.function->name()));

      last_op_out_value = op_out_value;
    }

    // Update value_by_continuation_value, set next values
    auto continuation_out_it = slice.continuations_out.begin();
    for (int64_t out_i = 0; out_i < slice.continuations_out.size();
         ++out_i, ++continuation_out_it) {
      const ContinuationValue& continuation_out = *continuation_out_it;
      TrackedBValue value_out =
          pb.TupleIndex(ret_tup, out_i, body_loc,
                        /*name=*/
                        absl::StrFormat("%s_out_%s", slice.function->name(),
                                        continuation_out.name));

      if (!state_element_by_continuation_value.contains(&continuation_out)) {
        value_by_continuation_value[&continuation_out] = value_out;
        continue;
      }

      value_by_continuation_value[&continuation_out] = pb.Select(
          slice_active,
          /*on_true=*/value_out,
          /*on_false=*/
          state_element_by_continuation_value.at(&continuation_out), body_loc,
          /*name=*/
          absl::StrFormat("select_active_or_prev_slice_%li_cont_%s",
                          slice_index, continuation_out.name));

      // Generate next values for state elements
      NextStateValue next_value = {
          .value = value_out,
          .condition = slice_active,
      };

      // Enable narrowing by including the loop jump condition in the next value
      // condition.
      //
      // This is safe to do because after the loop, values are fed forward.
      // Any feedback via state elements is from the iteration before, via the
      // loop body logic.
      //
      // TODO(seanhaskell): Either use values_to_save with next values from
      // jump state, or apply loop condition to all slices in the loop body.
      //
      // This doesn't work for a non-trivial loop body, as there may be
      // feedbacks that are not from the last state in the body.
      if (layout.transition_by_slice_from_index.contains(slice_index)) {
        next_value.condition = pb.And(next_value.condition, last_op_out_value);
      }

      // Generate next values
      extra_next_state_values.insert(
          {state_element_by_continuation_value.at(&continuation_out)
               .node()
               ->As<xls::StateRead>()
               ->state_element(),
           next_value});
    }

    if (layout.transition_by_slice_from_index.contains(slice_index)) {
      const NewFSMActivationTransition& transition =
          layout.transition_by_slice_from_index.at(slice_index);
      TrackedBValue jump_condition =
          pb.And(last_op_out_value, slice_active, body_loc, /*name=*/
                 absl::StrFormat("%s_jump_condition", slice.function->name()));
      XLSCC_CHECK(jump_condition.valid(), body_loc);
      XLSCC_CHECK(jump_condition.GetType()->IsBits(), body_loc);
      XLSCC_CHECK_EQ(jump_condition.GetType()->GetFlatBitCount(), 1, body_loc);
      const TrackedBValue jump_state_elem =
          state_element_by_jump_slice_index.at(slice_index);
      extra_next_state_values.insert(
          {jump_state_elem.node()->As<xls::StateRead>()->state_element(),
           NextStateValue{
               .value = last_op_out_value,
               .condition = slice_active,
           }});
      debug_did_jump = jump_condition;
      TrackedBValue jump_to_slice_index = pb.Literal(
          xls::UBits(transition.to_slice, num_slice_index_bits), body_loc,
          /*name=*/
          absl::StrFormat("%s_jump_to_slice_index", slice.function->name()));

      // TODO(seanhaskell): Force next activation on loop fall through?
      // (Make IOs mutually exclusive, ordered, can use tokens?)
      next_slice_index = pb.Select(
          jump_condition,
          /*on_true=*/jump_to_slice_index,
          /*on_false=*/next_slice_index, body_loc,
          /*name=*/
          absl::StrFormat("%s_select_did_jump", slice.function->name()));
    }

    // Calculate next state index
    if (debug_ir_trace_flags_ & DebugIrTraceFlags_LoopControl) {
      (void)pb.Trace(
          pb.Literal(xls::Value::Token(), body_loc),
          /*condition=*/pb.Literal(xls::Value(xls::UBits(1, 1)), body_loc),
          /*args=*/
          {slice_active, debug_did_jump, current_slice_index, next_slice_index},
          /*format_string=*/
          absl::StrFormat("--- [%li] slice_active {:u} did_jump {:u} current "
                          "{:u} next {:u}",
                          slice_index));
    }
    current_slice_index = pb.Select(
        slice_active,
        /*on_true=*/next_slice_index,
        /*on_false=*/current_slice_index, body_loc,
        /*name=*/
        absl::StrFormat("%s_next_slice_index", slice.function->name()));
  }

  // Set next slice index
  TrackedBValue after_last_slice =
      pb.Eq(current_slice_index,
            pb.Literal(xls::UBits(func.slices.size(), num_slice_index_bits),
                       body_loc),
            body_loc,
            /*name=*/"after_last_slice");

  extra_next_state_values.insert(
      {next_activation_slice_index.node()
           ->As<xls::StateRead>()
           ->state_element(),
       NextStateValue{
           .priority = std::numeric_limits<int64_t>::max(),
           .value = current_slice_index,
           .condition = pb.Not(after_last_slice),
       }});
  extra_next_state_values.insert(
      {next_activation_slice_index.node()
           ->As<xls::StateRead>()
           ->state_element(),
       NextStateValue{
           .priority = std::numeric_limits<int64_t>::max(),
           .value = first_slice_index,
           .condition = after_last_slice,
       }});

  return GenerateFSMInvocationReturn{
      .return_value = pb.Literal(
          xls::ZeroOfType(
              xls_func->slices.back().function->return_value()->GetType()),
          body_loc),
      .returns_this_activation = after_last_slice,
      .extra_next_state_values = extra_next_state_values};
}

absl::StatusOr<
    absl::flat_hash_map<int64_t, std::vector<NewFSMGenerator::PhiElement>>>
NewFSMGenerator::GeneratePhiConditions(
    const NewFSMLayout& layout,
    const absl::flat_hash_map<int64_t, TrackedBValue>&
        state_element_by_jump_slice_index,
    xls::ProcBuilder& pb, const xls::SourceInfo& body_loc) {
  absl::flat_hash_map<int64_t, std::vector<PhiElement>>
      phi_elements_by_param_node_id;

  // Use param node ID for determinism
  absl::btree_map<int64_t, std::vector<const NewFSMState*>>
      states_by_param_node_id;
  absl::flat_hash_map<int64_t, const xls::Param*> param_by_node_id;

  for (const NewFSMState& state : layout.states) {
    for (const auto& [param, value] : state.current_inputs_by_input_param) {
      states_by_param_node_id[param->id()].push_back(&state);
      param_by_node_id[param->id()] = param;
    }
  }
  for (const auto& [param_id, virtual_states] : states_by_param_node_id) {
    XLSCC_CHECK(!virtual_states.empty(), body_loc);
    // Sorted for determinism
    absl::btree_set<int64_t> from_jump_slice_indices;
    for (const NewFSMState* state : virtual_states) {
      for (const JumpInfo& jump_info : state->jumped_from_slice_indices) {
        from_jump_slice_indices.insert(jump_info.from_slice);
      }
    }

    std::vector<PhiElement>& phi_elements =
        phi_elements_by_param_node_id[param_id];

    for (const NewFSMState* state : virtual_states) {
      const xls::Param* param = param_by_node_id.at(param_id);
      absl::btree_set<int64_t> jumped_from_slice_indices_this_state;
      for (const JumpInfo& jump_info : state->jumped_from_slice_indices) {
        jumped_from_slice_indices_this_state.insert(jump_info.from_slice);
      }
      TrackedBValue condition = pb.Literal(xls::UBits(1, 1), body_loc);

      // Include all jump slices in each condition
      for (int64_t from_jump_slice_index : from_jump_slice_indices) {
        TrackedBValue jump_state_element =
            state_element_by_jump_slice_index.at(from_jump_slice_index);
        const int64_t active_value =
            jumped_from_slice_indices_this_state.contains(from_jump_slice_index)
                ? 1
                : 0;
        TrackedBValue condition_part = pb.Eq(
            jump_state_element,
            pb.Literal(
                xls::UBits(active_value, 1), body_loc,
                /*name=*/
                absl::StrFormat("slice_%li_param_%s_from_jump_slice_%li_active",
                                state->slice_index, param->name(),
                                from_jump_slice_index)));

        condition = pb.And(
            condition, condition_part, body_loc,
            /*name=*/
            absl::StrFormat(
                "slice_%li_jump__%s__phi_condition", state->slice_index,
                absl::StrJoin(jumped_from_slice_indices_this_state, "_")));
      }
      PhiElement& phi_element = phi_elements.emplace_back();
      phi_element.value = state->current_inputs_by_input_param.at(param);
      phi_element.condition = condition;
    }
  }

  return phi_elements_by_param_node_id;
}

absl::StatusOr<std::optional<TrackedBValue>>
NewFSMGenerator::GenerateInputValueInContext(
    const xls::Param* param,
    const absl::flat_hash_map<int64_t, std::vector<PhiElement>>&
        phi_elements_by_param_node_id,
    const absl::flat_hash_map<const ContinuationValue*, TrackedBValue>&
        value_by_continuation_value,
    const int64_t slice_index, xls::ProcBuilder& pb,
    const xls::SourceInfo& body_loc) {
  // Ignore IO params
  if (!phi_elements_by_param_node_id.contains(param->id())) {
    return std::nullopt;
  }
  const std::vector<PhiElement>& phi_elements =
      phi_elements_by_param_node_id.at(param->id());

  std::vector<TrackedBValue> phi_conditions;
  std::vector<TrackedBValue> phi_values;

  phi_conditions.reserve(phi_elements.size());
  phi_values.reserve(phi_elements.size());
  for (const PhiElement& phi_element : phi_elements) {
    phi_conditions.push_back(phi_element.condition);
    phi_values.push_back(value_by_continuation_value.at(phi_element.value));
  }

  std::reverse(phi_conditions.begin(), phi_conditions.end());

  XLSCC_CHECK_GT(phi_values.size(), 0, body_loc);
  XLSCC_CHECK_EQ(phi_values.size(), phi_conditions.size(), body_loc);

  if (phi_values.size() == 1) {
    return phi_values.at(0);
  }

  TrackedBValue one_hot_select = pb.OneHotSelect(
      pb.Concat(ToNativeBValues(phi_conditions), body_loc),
      ToNativeBValues(phi_values), body_loc,
      /*name=*/
      absl::StrFormat("slice_%li_param_%s_phi", slice_index, param->name()));

  return one_hot_select;
}

}  // namespace xlscc
