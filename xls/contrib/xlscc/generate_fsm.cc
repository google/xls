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
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "clang/include/clang/AST/Decl.h"
#include "xls/common/math_util.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/tracked_bvalue.h"
#include "xls/contrib/xlscc/translator_types.h"
#include "xls/contrib/xlscc/xlscc_logging.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/lsb_or_msb.h"
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
    const std::list<GeneratedFunctionSlice>& slices, NewFSMLayout& layout,
    const xls::SourceInfo& body_loc) {
  int64_t slice_index = 0;
  for (const GeneratedFunctionSlice& slice : slices) {
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

absl::Status NewFSMGenerator::LayoutNewFSMNoStateElements(
    NewFSMLayout& layout, const std::list<GeneratedFunctionSlice>& slices,
    const xls::SourceInfo& body_loc) {
  XLS_RETURN_IF_ERROR(SetupNewFSMGenerationContext(slices, layout, body_loc));

  // Record transitions across activations
  XLS_RETURN_IF_ERROR(LayoutNewFSMTransitions(layout, slices, body_loc));

  if (debug_ir_trace_flags_ & DebugIrTraceFlags_FSMStates) {
    LOG(INFO) << "FSM transitions:";
    for (const NewFSMActivationTransition& transition :
         layout.state_transitions) {
      LOG(INFO) << absl::StrFormat(
          "  %li -> %li (conditional? %i start_op? %s)", transition.from_slice,
          transition.to_slice, (int)transition.conditional,
          Debug_OpName(transition.start_op_type));
    }
  }

  XLS_RETURN_IF_ERROR(LayoutNewFSMStates(layout, slices, body_loc));

  return absl::OkStatus();
}

absl::Status NewFSMGenerator::ValidateStateInputs(
    const GeneratedFunction& func, const NewFSMLayout& layout,
    const xls::SourceInfo& body_loc) const {
  absl::flat_hash_map<int64_t, const GeneratedFunctionSlice*> slice_by_index;
  {
    int64_t slice_index = 0;
    for (const GeneratedFunctionSlice& slice : func.slices) {
      slice_by_index[slice_index] = &slice;
      ++slice_index;
    }
  }
  // Check that there's an input for each slice param
  for (const NewFSMState& state : layout.states) {
    const GeneratedFunctionSlice* slice = slice_by_index.at(state.slice_index);
    for (const ContinuationInput& continuation_in : slice->continuations_in) {
      const xls::Param* param = continuation_in.input_node;
      if (!state.current_inputs_by_input_param.contains(param)) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "State %s has no input for param %p/%s",
            state.GetStateId().ToString(), param, param->name()));
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<NewFSMLayout> NewFSMGenerator::LayoutNewFSM(
    const GeneratedFunction& func,
    const absl::flat_hash_map<DeclLeaf, xls::StateElement*>&
        state_element_for_static,
    const xls::SourceInfo& body_loc) {
  NewFSMLayout ret;

  XLS_RETURN_IF_ERROR(LayoutNewFSMNoStateElements(ret, func.slices, body_loc));

  XLS_RETURN_IF_ERROR(ValidateStateInputs(func, ret, body_loc));

  // Set NewFSMState::current_inputs_by_input_param from
  // ContinuationInput::choose_in_states. This allows phi selection order to be
  // preserved through optimization.
  absl::btree_map<StateId, NewFSMState*> state_by_state_id;
  for (NewFSMState& state : ret.states) {
    state.current_inputs_by_input_param.clear();
    const StateId state_id = state.GetStateId();
    XLSCC_CHECK(!state_by_state_id.contains(state_id), body_loc);
    state_by_state_id[state_id] = &state;
  }

  for (const GeneratedFunctionSlice& slice : func.slices) {
    for (const ContinuationInput& continuation_in : slice.continuations_in) {
      for (const StateId& state_id : continuation_in.choose_in_states) {
        XLSCC_CHECK(state_by_state_id.contains(state_id), body_loc);
        NewFSMState* state = state_by_state_id.at(state_id);
        XLSCC_CHECK(!state->current_inputs_by_input_param.contains(
                        continuation_in.input_node),
                    body_loc);
        state->current_inputs_by_input_param[continuation_in.input_node] =
            continuation_in.continuation_out;
      }
    }
  }

  XLS_RETURN_IF_ERROR(ValidateStateInputs(func, ret, body_loc));

  XLS_RETURN_IF_ERROR(LayoutValuesToSaveForNewFSMStates(ret, body_loc));

  // Remove unused states
  std::erase_if(ret.states, [](const NewFSMState& state) {
    const bool ignore_state = absl::c_any_of(
        state.jumped_from_slice_indices,
        [](const JumpInfo& jump_info) { return jump_info.count < 2; });
    return ignore_state;
  });

  XLS_RETURN_IF_ERROR(
      LayoutNewFSMStateElements(ret, func, state_element_for_static, body_loc));

  if (debug_ir_trace_flags_ & DebugIrTraceFlags_FSMStates) {
    LOG(INFO) << "FSM states after state element allocation:";
    PrintNewFSMStates(ret);
  }

  if (debug_ir_trace_flags_ & DebugIrTraceFlags_FSMStates) {
    int64_t total_bits = 0;
    for (const NewFSMStateElement& elem : ret.state_elements) {
      total_bits += elem.type->GetFlatBitCount();
    }
    LOG(INFO) << "State elements allocated: " << ret.state_elements.size()
              << ", total " << total_bits << " bits:";
    for (int64_t elem_idx = 0; elem_idx < ret.state_elements.size();
         ++elem_idx) {
      const NewFSMStateElement& elem = ret.state_elements.at(elem_idx);
      std::vector<std::string> value_names;
      for (const auto& [value, value_elem_index] :
           ret.state_element_by_continuation_value) {
        if (value_elem_index != elem_idx) {
          continue;
        }
        value_names.push_back(
            absl::StrFormat("%s from slice %li", value->name,
                            ret.output_slice_index_by_value.at(value)));
      }
      LOG(INFO) << absl::StrFormat("    %s type %s (%i bits), values: %s",
                                   elem.name, elem.type->ToString(),
                                   elem.type->GetFlatBitCount(),
                                   absl::StrJoin(value_names, ", "));
    }
  }

  return ret;
}

absl::Status NewFSMGenerator::LayoutNewFSMTransitions(
    NewFSMLayout& layout, const std::list<GeneratedFunctionSlice>& slices,
    const xls::SourceInfo& body_loc) {
  // TODO(seanhaskell): Add transition from last to first for statics.

  absl::flat_hash_set<const IOChannel*> channels_used_this_activation;

  auto insert_transition_safely =
      [&](const NewFSMActivationTransition& transition,
          const xls::SourceInfo& body_loc) -> void {
    XLSCC_CHECK(
        !layout.transition_by_slice_from_index.contains(transition.from_slice),
        body_loc);
    layout.transition_by_slice_from_index[transition.from_slice] = transition;
  };

  bool first_slice = true;
  for (const GeneratedFunctionSlice& slice : slices) {
    if (first_slice) {
      first_slice = false;
      continue;
    }
    if (slice.is_slice_before) {
      const int64_t before_io_slice_index = layout.index_by_slice.at(&slice);
      const int64_t after_io_slice_index = before_io_slice_index + 1;
      const IOOp* op_after = nullptr;

      // This is the "before" slice that can be transitioned to in an IO
      // activation transition.
      const GeneratedFunctionSlice* after_io_slice =
          layout.slice_by_index.at(after_io_slice_index);
      op_after = after_io_slice->after_op;

      XLSCC_CHECK_NE(op_after, nullptr, body_loc);

      if (!channels_used_this_activation.contains(op_after->channel)) {
        channels_used_this_activation.insert(op_after->channel);
        continue;
      }

      // Implied barrier
      XLSCC_CHECK_GT(before_io_slice_index, 0, body_loc);
      NewFSMActivationTransition transition = {
          .from_slice = before_io_slice_index - 1,
          .to_slice = before_io_slice_index,
          .conditional = false,
          .start_op_type = OpType::kActivationBarrier,
      };
      insert_transition_safely(transition, body_loc);
      layout.state_transitions.push_back(transition);
      layout.all_jump_from_slice_indices.push_back(transition.from_slice);
      // All channels are cleared after the transition
      channels_used_this_activation.clear();
      // The barrier is before this op, so this channel must be added
      channels_used_this_activation.insert(op_after->channel);
      continue;
    }

    const IOOp* after_op = slice.after_op;
    XLSCC_CHECK_NE(after_op, nullptr, body_loc);

    if (after_op->op == OpType::kActivationBarrier &&
        after_op->activation_barrier_type !=
            ActivationBarrierType::kConditionalEnd) {
      const int64_t before_io_slice_index = layout.index_by_slice.at(&slice);

      NewFSMActivationTransition transition = {
          .from_slice = before_io_slice_index - 1,
          .to_slice = before_io_slice_index,
          .conditional = after_op->activation_barrier_type ==
                         ActivationBarrierType::kConditionalBegin,
          .start_op_type = after_op->op,
      };
      insert_transition_safely(transition, body_loc);
      layout.state_transitions.push_back(transition);
      layout.all_jump_from_slice_indices.push_back(transition.from_slice);
      continue;
    }
    // This is optional, so doesn't reset channels_used_this_activation
    if (after_op->op == OpType::kLoopEndJump) {
      const int64_t end_jump_slice_index = layout.index_by_slice.at(&slice);
      XLSCC_CHECK_GE(end_jump_slice_index, 1, body_loc);
      const IOOp* const loop_begin_op = after_op->loop_op_paired;
      XLSCC_CHECK_NE(loop_begin_op, nullptr, body_loc);
      const int64_t begin_slice_index =
          layout.slice_index_by_after_op.at(loop_begin_op);

      // Feedback is from the slice after begin to the slice before end jump
      NewFSMActivationTransition transition = {
          .from_slice = end_jump_slice_index - 1,
          .to_slice = begin_slice_index,
          .conditional = true,
          .start_op_type = after_op->op,
      };
      insert_transition_safely(transition, body_loc);
      layout.state_transitions.push_back(transition);
      layout.all_jump_from_slice_indices.push_back(transition.from_slice);
      continue;
    }
  }

  return absl::OkStatus();
}

absl::Status NewFSMGenerator::LayoutNewFSMStates(
    NewFSMLayout& layout, const std::list<GeneratedFunctionSlice>& slices,
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

  for (int64_t slice_index = 0; slice_index < slices.size(); ++step) {
    NewFSMState& new_state =
        layout.states.emplace_back(NewFSMState{.slice_index = slice_index});

    const GeneratedFunctionSlice* slice = layout.slice_by_index.at(slice_index);

    // Get current input for each parameter (resolving phis)

    // First slice cannot have continuation inputs
    XLSCC_CHECK(!(slice_index == 0 && !slice->continuations_in.empty()),
                body_loc);

    absl::flat_hash_map<xls::Param*, std::vector<const ContinuationInput*>>
        inputs_by_param;
    for (const ContinuationInput& continuation_in : slice->continuations_in) {
      inputs_by_param[continuation_in.input_node].push_back(&continuation_in);
    }

    for (const auto& [input_param, continuation_ins] : inputs_by_param) {
      XLSCC_CHECK_GE(continuation_ins.size(), 1, body_loc);
      int64_t num_upstream_inputs_this_param = 0;
      int64_t latest_step_produced = -1;
      const ContinuationInput* latest_continuation_in = nullptr;
      for (const ContinuationInput* continuation_in : continuation_ins) {
        const ContinuationValue* continuation_out =
            continuation_in->continuation_out;

        const int64_t output_slice_index =
            layout.output_slice_index_by_value.at(continuation_out);
        if (output_slice_index < slice_index) {
          ++num_upstream_inputs_this_param;
        }

        // Ignore if not produced yet
        if (!step_produced_by_value.contains(continuation_out)) {
          continue;
        }
        const int64_t step_produced =
            step_produced_by_value.at(continuation_out);
        if (step_produced > latest_step_produced) {
          latest_step_produced = step_produced;
          latest_continuation_in = continuation_in;
        }
      }

      // We don't handle generating phis with multiple upstream inputs.
      XLSCC_CHECK_LE(num_upstream_inputs_this_param, 1, body_loc);
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

        if (transition.forward()) {
          // Jumping forwards
          CHECK_GT(transition.to_slice, transition.from_slice);
        } else {
          // Jumping backwards
          CHECK_GE(transition.from_slice, transition.to_slice);
          jumped_from_slice.push_back(JumpInfo{
              .from_slice = transition.from_slice,
              .to_slice = transition.to_slice,
              .count = 2,
          });
        }
      } else {
        ++slice_index;
      }
    }
  }

  XLSCC_CHECK(jumped_from_slice.empty(), body_loc);

  return absl::OkStatus();
}

absl::Status NewFSMGenerator::LayoutNewFSMStateElements(
    NewFSMLayout& layout, const GeneratedFunction& func,
    const absl::flat_hash_map<DeclLeaf, xls::StateElement*>&
        state_element_for_static,
    const xls::SourceInfo& body_loc) {
  // Plan the state elements
  //
  // State elements may be shared by multiple continuation values if the values
  // have the same type and are not saved in the same state (slice + jump flags)
  //
  // For optimization purposes, such as narrowing, it is better that the values
  // saved in a state element share semantics. Therefore, Clang NamedDecls are
  // used to identify values that may share a state element.
  absl::flat_hash_map<DeclLeaf, std::vector<int64_t>>
      state_element_indices_by_decl;

  // Inject statics: this enables state element sharing with statics
  for (const auto& [decl_leaf, existing_state_element] :
       state_element_for_static) {
    NewFSMStateElement state_element = {
        .name = existing_state_element->name(),
        .type = existing_state_element->type(),
        .existing_state_element = existing_state_element,
    };
    layout.state_elements.push_back(state_element);
    state_element_indices_by_decl[decl_leaf].push_back(
        layout.state_elements.size() - 1);
  }

  for (const NewFSMState& state : layout.states) {
    // Only need to save continuation values on activation transitions
    if (!layout.transition_by_slice_from_index.contains(state.slice_index)) {
      continue;
    }

    // A state element can only be used once in a given transition
    absl::flat_hash_set<int64_t> used_state_element_indices;

    // Mark reserved elements
    for (const ContinuationValue* value : state.values_to_save) {
      if (!layout.state_element_by_continuation_value.contains(value)) {
        continue;
      }
      const int64_t element_index =
          layout.state_element_by_continuation_value.at(value);
      // A state element can be used for different continuation values in
      // different states, typically when they share the same declaration.
      // However, since the allocation of state elements happens within a given
      // state, it is possible to end up in a scenario in which two continuation
      // values are saved into the same state element in one state, which can
      // cause incorrect results / logic errors.
      // This if statement fixes that by detecting the scenario and un-assigning
      // the state element from one of them.
      // For example, the problem:
      // - state A: save value A (decl X) into element 0
      // - state B: save value B (decl X) into element 0   * Element 0 re-used!
      // - state C: save value A (decl X) into element 0   *
      //            save value B (decl X) into element 0   *
      //        * Element 0 was already assigned to values A and B.
      // With this solution:
      // - state A: save value A (decl X) into element 0
      // - state B: save value B (decl X) into element 1
      // - state C: save value A (decl X) into element 0
      //            save value B (decl X) into element 1
      if (used_state_element_indices.contains(element_index)) {
        layout.state_element_by_continuation_value.erase(value);
        continue;
      }
      used_state_element_indices.insert(element_index);
    }

    for (const ContinuationValue* value : state.values_to_save) {
      if (layout.state_element_by_continuation_value.contains(value)) {
        // Already in used_state_element_indices
        continue;
      }
      // This value has not already been assigned a state element
      // Try to find state elements to share by decl
      std::optional<int64_t> found_element_by_decl = std::nullopt;
      std::vector<DeclLeaf> decls;

      for (const DeclLeaf& decl : value->decls) {
        decls.push_back(decl);
      }
      func.SortNamesDeterministically(decls);

      for (const DeclLeaf& decl : decls) {
        if (!state_element_indices_by_decl.contains(decl)) {
          continue;
        }
        const std::vector<int64_t>& elements_for_this_decl =
            state_element_indices_by_decl.at(decl);

        for (const int64_t element_for_decl_index : elements_for_this_decl) {
          if (used_state_element_indices.contains(element_for_decl_index)) {
            continue;
          }

          if (!layout.state_elements.at(element_for_decl_index)
                   .type->IsEqualTo(value->output_node->GetType())) {
            continue;
          }

          found_element_by_decl = element_for_decl_index;
          break;
        }
        if (found_element_by_decl.has_value()) {
          break;
        }
      }
      int64_t element_index = found_element_by_decl.value_or(-1);

      // Create a new state element if none were found to share
      if (!found_element_by_decl.has_value()) {
        std::string elem_name = value->name;
        if (element_index >= 0) {
          elem_name = absl::StrFormat("%s_el%li", elem_name, element_index);
        }
        NewFSMStateElement state_element = {
            .name = absl::StrFormat("%s_slc%li", elem_name, state.slice_index),
            .type = value->output_node->GetType(),
        };
        layout.state_elements.push_back(state_element);
        element_index = layout.state_elements.size() - 1;
      }

      // Mark element used for this value in this transition
      XLSCC_CHECK_GE(element_index, 0, body_loc);
      XLSCC_CHECK_LT(element_index, layout.state_elements.size(), body_loc);
      layout.state_element_by_continuation_value[value] = element_index;
      used_state_element_indices.insert(element_index);
      for (const DeclLeaf& decl : decls) {
        state_element_indices_by_decl[decl].push_back(element_index);
      }
    }
  }

  // Verify
  for (const NewFSMState& state : layout.states) {
    // Only need to save continuation values on activation transitions
    if (!layout.transition_by_slice_from_index.contains(state.slice_index)) {
      continue;
    }
    absl::flat_hash_set<int64_t> state_element_indices;
    for (const ContinuationValue* value : state.values_to_save) {
      auto found_elem = layout.state_element_by_continuation_value.find(value);
      if (found_elem == layout.state_element_by_continuation_value.end()) {
        continue;
      }
      const int64_t element_index = found_elem->second;
      XLSCC_CHECK(!state_element_indices.contains(element_index), body_loc);
      state_element_indices.insert(element_index);
    }
  }

  return absl::OkStatus();
}

absl::Status NewFSMGenerator::LayoutValuesToSaveForNewFSMStates(
    NewFSMLayout& layout, const xls::SourceInfo& body_loc) {
  // Fill in values to save after each state, in case of an activation
  // transition.
  //
  // States are now flattened and linear, so this can be handled as if
  // phis / jumps / loops don't exist, with a simple reference count for each
  // continuation value, initialized when the value is produced to the total
  // number of references in the graph.
  //
  // One caveat exists to the above: ContinuationValue* pointers are now no
  // longer sufficient to identify a given value. Using them alone will produce
  // a lot of false positives for values to save due to aliasing in later
  // loop iteration states.

  struct ValueKey {
    const ContinuationValue* value = nullptr;
    const NewFSMState* state_produced = nullptr;

    bool operator<(const ValueKey& other) const {
      if (value != other.value) {
        return value < other.value;
      }
      return state_produced < other.state_produced;
    }
  };

  absl::btree_map<ValueKey, int64_t> total_input_count_by_value;

  absl::flat_hash_map<const ContinuationValue*, const NewFSMState*>
      last_state_produced_value;

  for (const NewFSMState& state : layout.states) {
    for (const auto& [input_param, continuation_out] :
         state.current_inputs_by_input_param) {
      ++total_input_count_by_value[ValueKey{
          .value = continuation_out,
          .state_produced = last_state_produced_value.at(continuation_out),
      }];
    }

    const GeneratedFunctionSlice* slice =
        layout.slice_by_index.at(state.slice_index);

    for (const ContinuationValue& continuation_out : slice->continuations_out) {
      last_state_produced_value[&continuation_out] = &state;
      total_input_count_by_value[ValueKey{
          .value = &continuation_out,
          .state_produced = &state,
      }] = 0;
    }
  }

  last_state_produced_value.clear();

  absl::btree_map<ValueKey, int64_t> remaining_input_count_by_value;

  for (NewFSMState& state : layout.states) {
    const GeneratedFunctionSlice* slice =
        layout.slice_by_index.at(state.slice_index);
    // Decrement input counts
    for (const auto& [input_param, continuation_out] :
         state.current_inputs_by_input_param) {
      const ValueKey key = {
          .value = continuation_out,
          .state_produced = last_state_produced_value.at(continuation_out),
      };
      XLSCC_CHECK(remaining_input_count_by_value.contains(key), body_loc);
      --remaining_input_count_by_value[key];
    }

    // Record output counts, including all future uses
    for (const ContinuationValue& continuation_out : slice->continuations_out) {
      // Due to virtual unrolling, continuation values may be produced
      // multiple times
      const ValueKey key = {
          .value = &continuation_out,
          .state_produced = &state,
      };
      XLSCC_CHECK(!remaining_input_count_by_value.contains(key), body_loc);
      XLSCC_CHECK(total_input_count_by_value.contains(key), body_loc);
      const auto total_count = total_input_count_by_value.at(key);
      remaining_input_count_by_value.emplace(key, total_count);
      last_state_produced_value[&continuation_out] = &state;
    }

    for (const auto& [key, count] : remaining_input_count_by_value) {
      if (count == 0) {
        continue;
      }
      if (key.value->direct_in) {
        continue;
      }
      if (key.value->literal.has_value()) {
        continue;
      }
      state.values_to_save.insert(key.value);
    }
  }

  for (const auto& [value, count] : remaining_input_count_by_value) {
    XLSCC_CHECK_EQ(count, 0, body_loc);
  }

  return absl::OkStatus();
}

std::string NewFSMGenerator::GetStateName(const NewFSMState& state) {
  auto jump_infos_string =
      [](const std::vector<JumpInfo>& jump_infos) -> std::string {
    return absl::StrJoin(
        jump_infos, ", ", [](std::string* out, const JumpInfo& jump_info) {
          absl::StrAppendFormat(out, "{%li,c = %li}", jump_info.from_slice,
                                jump_info.count);
        });
  };
  return absl::StrFormat("  [slc %li j %s]: ", state.slice_index,
                         jump_infos_string(state.jumped_from_slice_indices));
}

std::string NewFSMGenerator::GetIRStateName(const NewFSMState& state) {
  auto jump_infos_string =
      [](const std::vector<JumpInfo>& jump_infos) -> std::string {
    std::vector<std::string> ret;
    for (const JumpInfo& jump_info : jump_infos) {
      std::string str;
      absl::StrAppendFormat(&str, "{%li,c = %li}", jump_info.from_slice,
                            jump_info.count);
      ret.push_back(str);
    }
    return absl::StrJoin(ret, "_");
  };
  return absl::StrFormat("state_%li__%s", state.slice_index,
                         jump_infos_string(state.jumped_from_slice_indices));
}

void NewFSMGenerator::PrintNewFSMStates(const NewFSMLayout& layout) {
  LOG(INFO) << absl::StrFormat("States %li, inputs:", layout.states.size());
  for (const NewFSMState& state : layout.states) {
    LOG(INFO) << GetStateName(state);
    for (const auto& [input_param, continuation_out] :
         state.current_inputs_by_input_param) {
      LOG(INFO) << absl::StrFormat(
          "    p %p/%s: slice %li %p/%s", input_param,
          input_param->name().data(),
          layout.output_slice_index_by_value.at(continuation_out),
          continuation_out, continuation_out->name.c_str());
    }
  }
  LOG(INFO) << absl::StrFormat("States %li, values to save:",
                               layout.states.size());
  for (const NewFSMState& state : layout.states) {
    int64_t num_bits = 0;
    for (const ContinuationValue* value : state.values_to_save) {
      num_bits += value->output_node->GetType()->GetFlatBitCount();
    }
    LOG(INFO) << GetStateName(state) << ": " << num_bits << " bits to save";
    for (const ContinuationValue* value : state.values_to_save) {
      LOG(INFO) << absl::StrFormat(
          "    v %s (%p) slice %li (%li bits) element %li", value->name.c_str(),
          value, layout.output_slice_index_by_value.at(value),
          value->output_node->GetType()->GetFlatBitCount(),
          layout.state_element_by_continuation_value.contains(value)
              ? layout.state_element_by_continuation_value.at(value)
              : -1L);
    }
  }
}

absl::StatusOr<
    absl::flat_hash_map<const IOOp*, NewFSMGenerator::BarrierScopeMask>>
NewFSMGenerator::GenerateBarrierScopeMasksByStartOp(
    int64_t num_slice_index_bits, TrackedBValue next_activation_slice_index,
    const GeneratedFunction& func, const NewFSMLayout& layout,
    xls::ProcBuilder& pb, const xls::SourceInfo& body_loc) {
  absl::flat_hash_map<const IOOp*, BarrierScopeMask> scopes_by_start_op;

  for (const IOOp& op : func.io_ops) {
    if (op.op != OpType::kActivationBarrier) {
      continue;
    }
    if (op.activation_barrier_type ==
        ActivationBarrierType::kConditionalBegin) {
      scopes_by_start_op[&op] = BarrierScopeMask{
          .start_slice = layout.slice_index_by_after_op.at(&op),
          .end_slice = -1};
    } else if (op.activation_barrier_type ==
               ActivationBarrierType::kConditionalEnd) {
      XLSCC_CHECK_NE(op.barrier_begin_op, nullptr, body_loc);
      XLSCC_CHECK(scopes_by_start_op.contains(op.barrier_begin_op), body_loc);
      scopes_by_start_op.at(op.barrier_begin_op).end_slice =
          layout.slice_index_by_after_op.at(&op);
    }
  }

  absl::flat_hash_map<const IOOp*, TrackedBValue> scope_masks_by_start_op;

  // Loop over scopes. Use ops for determinism.
  for (const IOOp& op : func.io_ops) {
    auto found_scope = scopes_by_start_op.find(&op);
    if (found_scope == scopes_by_start_op.end()) {
      continue;
    }
    const int64_t slice_index = layout.slice_index_by_after_op.at(&op);
    BarrierScopeMask& scope = found_scope->second;

    XLSCC_CHECK_NE(scope.start_slice, -1, body_loc);
    XLSCC_CHECK_NE(scope.end_slice, -1, body_loc);
    XLSCC_CHECK_LT(scope.start_slice, scope.end_slice, body_loc);

    TrackedBValue start_slice_index_bval = pb.Literal(
        xls::UBits(scope.start_slice, num_slice_index_bits), body_loc);

    TrackedBValue end_slice_index_bval =
        pb.Literal(xls::UBits(scope.end_slice, num_slice_index_bits), body_loc);

    TrackedBValue slice_lower_bound =
        pb.UGe(next_activation_slice_index, start_slice_index_bval, body_loc,
               /*name=*/absl::StrFormat("scope_before_slice_%li", slice_index));
    TrackedBValue slice_is_after =
        pb.ULt(next_activation_slice_index, end_slice_index_bval, body_loc,
               /*name=*/absl::StrFormat("scope_after_slice_%li", slice_index));

    TrackedBValue slice_is_in_scope =
        pb.And({slice_lower_bound, slice_is_after}, body_loc,
               /*name=*/absl::StrFormat("in_scope_slice_%li", slice_index));

    scope.mask_bval = slice_is_in_scope;
  }

  return scopes_by_start_op;
}

absl::StatusOr<TrackedBValue> NewFSMGenerator::GenerateBarrierSliceMask(
    int64_t slice_index,
    const absl::flat_hash_map<const IOOp*, BarrierScopeMask>&
        scopes_by_start_op,
    const GeneratedFunction& func, const NewFSMLayout& layout,
    xls::ProcBuilder& pb, const xls::SourceInfo& body_loc) {
  absl::InlinedVector<NATIVE_BVAL, 1> scope_masks_this_slice;

  // Loop over scopes. Use ops for determinism.
  // TODO(seanhaskell): Optimize this with log(N) lookup once the feature
  // is mature.
  for (const IOOp& op : func.io_ops) {
    auto found_scope = scopes_by_start_op.find(&op);
    if (found_scope == scopes_by_start_op.end()) {
      continue;
    }
    const BarrierScopeMask& scope = found_scope->second;

    XLSCC_CHECK_NE(scope.start_slice, -1, body_loc);
    XLSCC_CHECK_NE(scope.end_slice, -1, body_loc);
    XLSCC_CHECK_LT(scope.start_slice, scope.end_slice, body_loc);

    // Start slice activity is used to calculate after barrier at end of
    // scope.
    // End slice activity may be used to calculate finished iteration.
    if (slice_index <= scope.start_slice || slice_index >= scope.end_slice) {
      continue;
    }

    scope_masks_this_slice.push_back(scope.mask_bval);
  }

  if (scope_masks_this_slice.empty()) {
    return pb.Literal(xls::UBits(1, 1), body_loc);
  }

  return pb.And(scope_masks_this_slice, body_loc,
                absl::StrFormat("barrier_slice_mask_%li", slice_index));
}

absl::Status NewFSMGenerator::GenerateExtractStaticReturns(
    TrackedBValue last_slice_return_value,
    const absl::flat_hash_map<const clang::NamedDecl*, int64_t>&
        return_index_for_static,
    std::vector<TrackedBValue>& return_values, xls::ProcBuilder& pb,
    const xls::SourceInfo& body_loc) {
  struct DeclAndReturnIndex {
    const clang::NamedDecl* decl;
    int64_t return_index = -1;

    bool operator<(const DeclAndReturnIndex& other) const {
      return return_index < other.return_index;
    }
  };

  std::vector<DeclAndReturnIndex> all_static_values;

  for (const auto& [name, _] : return_index_for_static) {
    all_static_values.push_back(
        {.decl = name, .return_index = return_index_for_static.at(name)});
  }

  // Determinism
  std::sort(all_static_values.begin(), all_static_values.end());

  for (int64_t static_index = 0; static_index < all_static_values.size();
       ++static_index) {
    const DeclAndReturnIndex& static_decl_and_return_index =
        all_static_values.at(static_index);
    const clang::NamedDecl* static_decl = static_decl_and_return_index.decl;
    const int64_t return_index = static_decl_and_return_index.return_index;

    TrackedBValue output_value;
    // TODO(seanhaskell): Normalize last slice return
    if (all_static_values.size() == 1) {
      output_value = last_slice_return_value;
    } else {
      output_value = pb.TupleIndex(
          last_slice_return_value, static_index, body_loc,
          /*name=*/
          absl::StrFormat("return__%s", static_decl->getNameAsString()));
    }

    XLSCC_CHECK(output_value.GetType()->IsEqualTo(
                    return_values.at(return_index).GetType()),
                body_loc);

    return_values.at(return_index) = output_value;
  }
  return absl::OkStatus();
}

void NewFSMGenerator::ResetValuesToStateElements(
    const absl::flat_hash_map<const ContinuationValue*, TrackedBValue>&
        state_element_by_continuation_value,
    std::vector<ConditionalBarrierScope>& conditional_barrier_scope_stack) {
  ContinuationValueBValMap& value_by_continuation_value =
      conditional_barrier_scope_stack.back().value_by_continuation_value;
  for (auto& [value, state_element] : state_element_by_continuation_value) {
    value_by_continuation_value[value] = state_element;
  }
}

void NewFSMGenerator::AddToAfterConditionalActivationTransition(
    TrackedBValue condition, const xls::SourceInfo& loc, int64_t slice_index,
    std::vector<ConditionalBarrierScope>& conditional_barrier_scope_stack,
    xls::ProcBuilder& pb) {
  for (ConditionalBarrierScope& scope : conditional_barrier_scope_stack) {
    TrackedBValue& after_conditional_activation_transition =
        scope.after_conditional_activation_transition;
    after_conditional_activation_transition = pb.Or(
        after_conditional_activation_transition, condition, loc,
        /*name=*/
        absl::StrFormat("after_%li_after_conditional_activation_transition",
                        slice_index));
  }
};

absl::StatusOr<GenerateFSMInvocationReturn>
NewFSMGenerator::GenerateNewFSMInvocation(
    const GeneratedFunction* xls_func,
    const std::vector<TrackedBValue>& direct_in_args,
    const absl::flat_hash_map<DeclLeaf, xls::StateElement*>&
        state_element_for_static,
    const absl::flat_hash_map<const clang::NamedDecl*, xls::Type*>&
        type_for_static,
    const absl::flat_hash_map<const clang::NamedDecl*, int64_t>&
        return_index_for_static,
    xls::ProcBuilder& pb, const xls::SourceInfo& body_loc) {
  XLSCC_CHECK_NE(xls_func, nullptr, body_loc);
  const GeneratedFunction& func = *xls_func;
  NewFSMLayout layout;
  XLS_ASSIGN_OR_RETURN(layout,
                       LayoutNewFSM(func, state_element_for_static, body_loc));

  absl::flat_hash_map<PhiConditionCacheKey, TrackedBValue> generated_conditions;

  const int64_t num_slice_index_bits =
      xls::CeilOfLog2(1 + xls_func->slices.size());

  // TODO(seanhaskell): Clean this up once the old FSM is removed
  xls::Type* top_return_type =
      xls_func->slices.back().function->return_value()->GetType();

  std::vector<TrackedBValue> return_values;
  if (return_index_for_static.size() > 1) {
    for (int64_t i = 0; i < top_return_type->AsTupleOrDie()->size(); ++i) {
      return_values.push_back(pb.Literal(
          xls::ZeroOfType(top_return_type->AsTupleOrDie()->element_type(i)),
          body_loc));
    }
  } else {
    return_values.push_back(
        pb.Literal(xls::ZeroOfType(top_return_type), body_loc));
  }

  TrackedBValue next_activation_slice_index =
      pb.StateElement("__next_activation_slice",
                      xls::Value(xls::UBits(0, num_slice_index_bits)),
                      /*non_synthesizable=*/false, body_loc);

  TrackedBValue first_slice_index =
      pb.Literal(xls::UBits(0, num_slice_index_bits), body_loc);

  absl::btree_multimap<const xls::StateElement*, NextStateValue>
      extra_next_state_values;

  // Create state elements for jumps (jumped vs didn't jump yet)
  absl::flat_hash_map<int64_t, TrackedBValue> state_element_by_jump_slice_index;
  for (int64_t jump_slice_index : layout.all_jump_from_slice_indices) {
    const NewFSMActivationTransition& transition =
        layout.transition_by_slice_from_index.at(jump_slice_index);

    if (transition.start_op_type == OpType::kActivationBarrier) {
      continue;
    }

    TrackedBValue state_element =
        pb.StateElement(absl::StrFormat("__jump_state_%li", jump_slice_index),
                        xls::Value(xls::UBits(0, 1)),
                        /*non_synthesizable=*/false, body_loc);

    state_element_by_jump_slice_index[jump_slice_index] = state_element;
  }

  // Analyze phis
  absl::flat_hash_map<int64_t, std::vector<PhiElement>>
      phi_elements_by_param_node_id;

  XLS_ASSIGN_OR_RETURN(
      phi_elements_by_param_node_id,
      GeneratePhiConditions(layout, state_element_by_jump_slice_index, pb,
                            body_loc, generated_conditions));

  absl::flat_hash_map<const ContinuationValue*, TrackedBValue>
      state_element_by_continuation_value;

  absl::flat_hash_map<int64_t, std::vector<const ContinuationValue*>>
      values_by_state_element_index;

  // State elements by static may not go through any continuations
  for (int64_t i = 0; i < layout.state_elements.size(); ++i) {
    values_by_state_element_index[i] = {};
  }

  for (const auto& [value, index] :
       layout.state_element_by_continuation_value) {
    values_by_state_element_index[index].push_back(value);
  }

  // Create state elements
  // Loop over vector for determinism
  for (int64_t state_element_index = 0;
       state_element_index < layout.state_elements.size();
       ++state_element_index) {
    const NewFSMStateElement& state_element =
        layout.state_elements.at(state_element_index);
    TrackedBValue xls_state_element;

    if (state_element.existing_state_element == nullptr) {
      xls_state_element = pb.StateElement(
          state_element.name, xls::ZeroOfType(state_element.type),
          /*non_synthesizable=*/false, body_loc);
    } else {
      xls::StateRead* state_read = pb.proc()->GetStateReadByStateElement(
          state_element.existing_state_element);
      xls_state_element = TrackedBValue(state_read, &pb);
    }

    for (const ContinuationValue* value :
         values_by_state_element_index.at(state_element_index)) {
      state_element_by_continuation_value[value] = xls_state_element;
    }
  }

  std::vector<ConditionalBarrierScope> conditional_barrier_scope_stack;
  conditional_barrier_scope_stack.push_back(
      ConditionalBarrierScope{.after_conditional_activation_transition =
                                  pb.Literal(xls::UBits(0, 1), body_loc)});

  // Set values initially to state elements, so that feedbacks
  // come from state. These will be overwritten for feedforwards as slices
  // are generated.

  ResetValuesToStateElements(state_element_by_continuation_value,
                             conditional_barrier_scope_stack);

  auto assign_bval_to_continuation_value = [&](const ContinuationValue* value,
                                               TrackedBValue bval) {
    for (ConditionalBarrierScope& scope_map : conditional_barrier_scope_stack) {
      scope_map.value_by_continuation_value[value] = bval;
    }
  };

  absl::btree_map<std::tuple<xls::StateElement*, xls::Node*>,
                  absl::btree_set<xls::Node*, NodeIdLessThan>,
                  StateElementAndNodeLessThan>
      next_value_conditions_by_state_element_and_value;

  // Get a sorted list of unconditional transition from slice indices.
  std::vector<int64_t> unconditional_from_slice_indices_ordered;
  unconditional_from_slice_indices_ordered.reserve(
      layout.transition_by_slice_from_index.size());

  for (const auto& [from_slice_index, transition] :
       layout.transition_by_slice_from_index) {
    if (transition.conditional) {
      continue;
    }
    unconditional_from_slice_indices_ordered.push_back(from_slice_index);
  }

  std::sort(unconditional_from_slice_indices_ordered.begin(),
            unconditional_from_slice_indices_ordered.end());

  // Generate the slice active masks for scoped barriers.
  absl::flat_hash_map<const IOOp*, BarrierScopeMask> scopes_by_start_op;
  XLS_ASSIGN_OR_RETURN(scopes_by_start_op,
                       GenerateBarrierScopeMasksByStartOp(
                           num_slice_index_bits, next_activation_slice_index,
                           func, layout, pb, body_loc));

  if (debug_ir_trace_flags_ & DebugIrTraceFlags_ActivationBarriers) {
    TrackedBValue token = pb.Literal(xls::Value::Token(), body_loc,
                                     /*name=*/"token");
    pb.Trace(token, pb.Literal(xls::UBits(1, 1)),
             /*args=*/
             {next_activation_slice_index},
             absl::StrFormat("---- next activation slice index {:d} ---"));
  }

  // Use this index to step through the unconditional transition indices.
  int64_t next_unconditional_transition_index = 0;

  TrackedBValue last_slice_active = pb.Literal(xls::UBits(1, 1), body_loc);
  TrackedBValue last_op_out_value;

  absl::flat_hash_map<int64_t, TrackedBValue>
      jump_conditions_by_begin_slice_index;

  std::vector<SharedFunctionCall> shared_function_calls;

  for (int64_t slice_index = 0; slice_index < func.slices.size();
       ++slice_index) {
    const GeneratedFunctionSlice& slice =
        *layout.slice_by_index.at(slice_index);

    // Pop the value reference stack before preparing inputs for the end
    // barrier slice.
    if (slice.after_op != nullptr &&
        slice.after_op->op == OpType::kActivationBarrier &&
        slice.after_op->activation_barrier_type ==
            ActivationBarrierType::kConditionalEnd) {
      // Should never pop the base context
      XLSCC_CHECK_GT(conditional_barrier_scope_stack.size(), 1L, body_loc);
      // Pop the continuation value reference stack
      // State elements should be popped before end slice invoke, as it contains
      // the phi selects for the end of the scope.
      conditional_barrier_scope_stack.pop_back();

      const int64_t begin_slice_index =
          layout.slice_index_by_after_op.at(slice.after_op->barrier_begin_op);
      XLSCC_CHECK_GT(begin_slice_index, 0, body_loc);

      // Transition is from the slice before.
      // Need to also consider transitions in inner scopes.
      AddToAfterConditionalActivationTransition(
          jump_conditions_by_begin_slice_index.at(begin_slice_index - 1),
          body_loc, slice_index, conditional_barrier_scope_stack, pb);

      if (debug_ir_trace_flags_ & DebugIrTraceFlags_ActivationBarriers) {
        TrackedBValue token = pb.Literal(xls::Value::Token(), body_loc,
                                         /*name=*/"token");
        pb.Trace(
            token, pb.Literal(xls::UBits(1, 1)),
            /*args=*/
            {conditional_barrier_scope_stack.back()
                 .after_conditional_activation_transition},
            absl::StrFormat("end_scope[%li]: after_act {:b}", slice_index));
      }
    }

    const bool is_last_slice = (slice_index == func.slices.size() - 1);

    TrackedBValue slice_index_bval =
        pb.Literal(xls::UBits(slice_index, num_slice_index_bits), body_loc,
                   /*name=*/absl::StrFormat("slice_%li_index", slice_index));

    TrackedBValue slice_is_current = pb.Literal(xls::UBits(1, 1), body_loc);
    if (next_unconditional_transition_index > 0) {
      const int64_t from_slice_index =
          unconditional_from_slice_indices_ordered.at(
              next_unconditional_transition_index - 1);
      const NewFSMActivationTransition& transition =
          layout.transition_by_slice_from_index.at(from_slice_index);

      // next_activation_slice_index > from_slice
      slice_is_current = pb.And(
          slice_is_current,
          pb.UGt(next_activation_slice_index,
                 pb.Literal(
                     xls::UBits(transition.from_slice, num_slice_index_bits)),
                 body_loc,
                 /*name=*/
                 absl::StrFormat("slice_%li_is_current_lower", slice_index)));
    }

    // Upper bound needs to be this slice index, as there can be jumps to
    // anywhere. next_activation_slice_index <= from_slice
    slice_is_current = pb.And(
        slice_is_current,
        pb.ULe(next_activation_slice_index, slice_index_bval, body_loc,
               /*name=*/
               absl::StrFormat("slice_%li_is_current_upper", slice_index)));

    if (next_unconditional_transition_index <
        unconditional_from_slice_indices_ordered.size()) {
      const int64_t from_slice_index =
          unconditional_from_slice_indices_ordered.at(
              next_unconditional_transition_index);
      const NewFSMActivationTransition& transition =
          layout.transition_by_slice_from_index.at(from_slice_index);

      if (slice_index == transition.from_slice) {
        ++next_unconditional_transition_index;
      }
    }

    XLS_ASSIGN_OR_RETURN(
        TrackedBValue slice_barrier_mask,
        GenerateBarrierSliceMask(slice_index, scopes_by_start_op, func, layout,
                                 pb, body_loc));

    TrackedBValue slice_active = pb.And(
        {slice_is_current, slice_barrier_mask,
         pb.Not(conditional_barrier_scope_stack.back()
                    .after_conditional_activation_transition,
                body_loc, /*name=*/
                absl::StrFormat(
                    "slice_%li_not_after_conditional_activation_transition",
                    slice_index))},
        body_loc,
        /*name=*/absl::StrFormat("slice_%li_active", slice_index));

    last_slice_active = slice_active;

    if (debug_ir_trace_flags_ & DebugIrTraceFlags_ActivationBarriers) {
      TrackedBValue token = pb.Literal(xls::Value::Token(), body_loc,
                                       /*name=*/"token");
      pb.Trace(token, pb.Literal(xls::UBits(1, 1)),
               /*args=*/
               {slice_active, slice_barrier_mask, slice_is_current,
                conditional_barrier_scope_stack.back()
                    .after_conditional_activation_transition},
               absl::StrFormat("slice[%li]: active {:b} mask {:b} current {:b} "
                               "after barrier {:b}",
                               slice_index));
    }

    // Gather invoke params, except IO input
    std::vector<TrackedBValue> invoke_params;
    invoke_params.reserve(slice.continuations_in.size() + 1);

    // Add direct-ins (and top class input) to first slice params
    if (slice_index == 0) {
      invoke_params.insert(invoke_params.end(), direct_in_args.begin(),
                           direct_in_args.end());
    }

    // Order for determinism
    for (const xls::Param* param : slice.function->params()) {
      std::optional<TrackedBValue> input_value;
      XLS_ASSIGN_OR_RETURN(input_value,
                           GenerateInputValueInContext(
                               param, phi_elements_by_param_node_id,
                               conditional_barrier_scope_stack.back()
                                   .value_by_continuation_value,
                               state_element_by_continuation_value, slice_index,
                               slice_active, slice_is_current, pb, body_loc));

      if (!input_value.has_value()) {
        continue;
      }

      invoke_params.push_back(input_value.value());
    }

    const bool loop_op = slice.after_op != nullptr &&
                         (slice.after_op->op == OpType::kLoopBegin ||
                          slice.after_op->op == OpType::kLoopEndJump);

    // To avoid needing to store the IO op's received value,
    // the after_op is always in the same activation as the invoke for the
    // function slice.
    //
    // The output value for the op doesn't need to be stored because
    // it is available from the invoke for the previous function slice.
    //
    // NOTE: After an activation transition, the function slice must be
    // invoked again
    if (slice.after_op != nullptr && !loop_op) {
      const IOOp* after_op = slice.after_op;
      XLSCC_CHECK(after_op->op != OpType::kLoopBegin, body_loc);
      XLSCC_CHECK(after_op->op != OpType::kLoopEndJump, body_loc);

      std::optional<ChannelBundle> optional_bundle =
          translator_io_.GetChannelBundleForOp(*after_op, body_loc);
      // Do not generate an explicit token network.
      // Tokens edges will still be inserted downstream for data dependencies.
      // Ordering can be imposed via activation transitions.
      TrackedBValue token = pb.Literal(xls::Value::Token(), body_loc,
                                       /*name=*/"token");
      XLSCC_CHECK(last_op_out_value.valid(), body_loc);
      // No IO before first slice
      TrackedBValue io_active = slice_active;
      XLSCC_CHECK(io_active.valid(), body_loc);
      XLSCC_CHECK(slice_is_current.valid(), body_loc);

      // The IO op is predicated on the slice after it being active.
      // This is because the predicate/value to send, outputted from the slice
      // before it, remains valid even when the slice outputting it is inactive,
      // but the received value is only valid in the activation in which the IO
      // op happens, so it must be consumed in that same activation by the slice
      // after it.
      GenerateIOReturn io_return;

      if (after_op->op == OpType::kSharedCall) {
        XLS_RETURN_IF_ERROR(
            InterceptSharedCall(*after_op, last_op_out_value, io_active,
                                &shared_function_calls, &io_return, pb));
      } else {
        XLS_ASSIGN_OR_RETURN(io_return, translator_io_.GenerateIO(
                                            *after_op, token, last_op_out_value,
                                            pb, optional_bundle,
                                            /*extra_condition=*/
                                            io_active));
      }

      // Add IO parameter if applicable
      if (io_return.received_value.valid()) {
        TrackedBValue received_value = io_return.received_value;

        // Help the optimizer to remove references to the IO op from downstream
        // slices where they are spurious. This improves IO merging and
        // throughput.
        received_value = pb.Select(
            io_active,
            /*on_true=*/received_value,
            /*on_false=*/
            pb.Literal(xls::ZeroOfType(received_value.GetType()), body_loc),
            body_loc,
            /*name=*/
            absl::StrFormat("select_io_out_or_0_%li_%s", slice_index,
                            after_op->final_param_name));

        invoke_params.push_back(received_value);
      }
    }

    // Add statics
    for (const clang::NamedDecl* decl : slice.static_values) {
      TrackedBValue prev_val;
      XLS_ASSIGN_OR_RETURN(
          prev_val, ComposeStaticValueInput(decl,
                                            /*generate_new_fsm=*/true,
                                            state_element_for_static,
                                            type_for_static, pb, body_loc));
      invoke_params.push_back(prev_val);
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
      if (next_slice.after_op != nullptr) {
        XLS_ASSIGN_OR_RETURN(
            TrackedBValue op_out_value,
            translator_io_.GetIOOpRetValueFromSlice(ret_tup, slice, body_loc));
        last_op_out_value = op_out_value;
      } else {
        last_op_out_value = TrackedBValue();
      }
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
        assign_bval_to_continuation_value(&continuation_out, value_out);
        continue;
      }

      // A continuation value is available directly if the slice that produces
      // it is active this activation.
      TrackedBValue select_bval = pb.Select(
          slice_active,
          /*on_true=*/value_out,
          /*on_false=*/
          state_element_by_continuation_value.at(&continuation_out), body_loc,
          /*name=*/
          absl::StrFormat("select_active_or_prev_slice_%li_cont_%s",
                          slice_index, continuation_out.name));
      assign_bval_to_continuation_value(&continuation_out, select_bval);
    }

    if (is_last_slice) {
      XLS_RETURN_IF_ERROR(GenerateExtractStaticReturns(
          ret_tup, return_index_for_static, return_values, pb, body_loc));
    }

    if (layout.transition_by_slice_from_index.contains(slice_index)) {
      XLS_RETURN_IF_ERROR(GenerateTransitionFromThisSlice(
          /*from_slice_index=*/slice_index, num_slice_index_bits, slice_active,
          last_op_out_value, next_activation_slice_index, layout, slice,
          state_element_by_jump_slice_index,
          state_element_by_continuation_value, extra_next_state_values,
          jump_conditions_by_begin_slice_index, conditional_barrier_scope_stack,
          generated_conditions,
          next_value_conditions_by_state_element_and_value, pb, body_loc));
    }
  }  // slices

  // Shared function calls
  XLS_RETURN_IF_ERROR(GenerateSharedCalls(shared_function_calls, pb));

  for (auto& [key, or_nodes] :
       next_value_conditions_by_state_element_and_value) {
    xls::StateElement* state_elem = std::get<0>(key);
    xls::Node* next_value_node = std::get<1>(key);
    std::vector<NATIVE_BVAL> or_bvals;
    for (xls::Node* or_node : or_nodes) {
      or_bvals.push_back(NATIVE_BVAL(or_node, &pb));
    }

    TrackedBValue or_bval =
        pb.Or(absl::MakeSpan(or_bvals), body_loc,
              /*name=*/
              absl::StrFormat("%s_v_%s_or_bval", state_elem->name(),
                              next_value_node->GetName()));

    NextStateValue next_value = {
        .priority = 0,
        .value = TrackedBValue(next_value_node, &pb),
        .condition = or_bval,
    };
    extra_next_state_values.insert({state_elem, next_value});
  }

  // Set next slice index
  const TrackedBValue finished_iteration =
      pb.And({last_slice_active,
              pb.Not(conditional_barrier_scope_stack.back()
                         .after_conditional_activation_transition,
                     body_loc, /*name=*/
                     "last_slice_not_after_conditional_activation_transition")},
             body_loc,
             /*name=*/"finished_iteration");

  extra_next_state_values.insert(
      {next_activation_slice_index.node()
           ->As<xls::StateRead>()
           ->state_element(),
       NextStateValue{
           .priority = std::numeric_limits<int64_t>::max(),
           .value = first_slice_index,
           .condition = finished_iteration,
       }});

  if (debug_ir_trace_flags_ & DebugIrTraceFlags_ActivationBarriers) {
    TrackedBValue token = pb.Literal(xls::Value::Token(), body_loc,
                                     /*name=*/"token");
    pb.Trace(
        token, pb.Literal(xls::UBits(1, 1)),
        /*args=*/
        {finished_iteration, last_slice_active,
         conditional_barrier_scope_stack.back()
             .after_conditional_activation_transition},
        absl::StrFormat(
            "finished_iteration? {:b} last_slice_active {:b} after_act {:b}"));
  }

  TrackedBValue return_value;

  if (return_index_for_static.size() > 1) {
    return_value = pb.Tuple(ToNativeBValues(return_values), body_loc);
  } else {
    return_value = return_values.at(0);
  }

  return GenerateFSMInvocationReturn{
      .return_value = return_value,
      .returns_this_activation = finished_iteration,
      .extra_next_state_values = extra_next_state_values};
}

absl::Status NewFSMGenerator::InterceptSharedCall(
    const IOOp& op, TrackedBValue op_out_value, TrackedBValue io_active,
    std::vector<SharedFunctionCall>* shared_function_calls,
    GenerateIOReturn* io_return, xls::ProcBuilder& pb) {
  const GeneratedFunction* shared_call_func = op.shared_call_func;
  const std::shared_ptr<CType>& shared_call_param_type =
      op.shared_call_param_type;

  const xls::SourceInfo& func_loc =
      translator_types().GetLoc(*shared_call_func->clang_decl);

  XLSCC_CHECK(shared_call_func != nullptr, func_loc);
  XLSCC_CHECK(shared_call_param_type != nullptr, func_loc);

  io_return->io_condition = io_active;

  TrackedBValue val =
      pb.TupleIndex(op_out_value, 0, func_loc,
                    /*name=*/
                    absl::StrFormat("%s_value", Debug_OpName(op)));
  XLSCC_CHECK(val.valid(), func_loc);

  XLS_ASSIGN_OR_RETURN(
      xls::Type * ret_type,
      translator_types().TranslateTypeToXLS(shared_call_param_type, func_loc));

  io_return->received_value = pb.Literal(xls::ZeroOfType(ret_type), func_loc);

  shared_function_calls->push_back(
      SharedFunctionCall{.func = shared_call_func,
                         .input = val,
                         .output = io_return->received_value,
                         .condition = io_active});

  return absl::OkStatus();
}

absl::Status NewFSMGenerator::GenerateSharedCalls(
    const std::vector<SharedFunctionCall>& shared_function_calls,
    xls::ProcBuilder& pb) {
  absl::flat_hash_map<const GeneratedFunction*,
                      std::vector<const SharedFunctionCall*>>
      shared_calls_by_func;
  std::vector<const GeneratedFunction*> shared_funcs_in_order;
  for (const SharedFunctionCall& call : shared_function_calls) {
    if (!shared_calls_by_func.contains(call.func)) {
      shared_funcs_in_order.push_back(call.func);
    }
    shared_calls_by_func[call.func].push_back(&call);
  }
  // Ordered for determinism
  for (const GeneratedFunction* shared_func : shared_funcs_in_order) {
    if (shared_func->slices.size() != 1) {
      return absl::InternalError(
          absl::StrFormat("Shared function's should have exactly 1 slice (no "
                          "side effects), %s has %li slices",
                          shared_func->clang_decl->getNameAsString().c_str(),
                          shared_func->slices.size()));
    }

    const GeneratedFunctionSlice& only_slice = shared_func->slices.front();
    const xls::SourceInfo& func_loc =
        translator_types().GetLoc(*shared_func->clang_decl);

    // Route input
    std::vector<TrackedBValue> input_conditions;
    std::vector<TrackedBValue> input_values;

    for (const SharedFunctionCall* call : shared_calls_by_func[shared_func]) {
      input_conditions.push_back(call->condition);
      input_values.push_back(call->input);
    }

    std::reverse(input_values.begin(), input_values.end());

    TrackedBValue selector =
        pb.Concat(ToNativeBValues(input_conditions), func_loc);

    (void)pb.Assert(
        pb.Literal(xls::Value::Token(), func_loc),
        /*condition=*/
        pb.Eq(
            selector,
            pb.BitSlice(pb.OneHot(selector, /*priority=*/xls::LsbOrMsb::kLsb),
                        /*start=*/0, /*end=*/input_conditions.size(), func_loc),
            func_loc),
        /*message=*/
        "Shared function input selector is not one hot, two calls in one "
        "activation?",
        /*label=*/std::nullopt, func_loc);

    TrackedBValue input_select = pb.PrioritySelect(
        selector, ToNativeBValues(input_values),
        /*default_value=*/
        pb.Literal(xls::ZeroOfType(input_values.at(0).GetType()), func_loc),
        func_loc,
        /*name=*/
        absl::StrFormat("shared_input_select_%s",
                        shared_func->clang_decl->getNameAsString()));

    xls::Type* input_type = input_select.GetType();
    XLSCC_CHECK(input_type->IsTuple(), func_loc);
    xls::TupleType* input_tuple_type = input_type->AsTupleOrDie();

    std::vector<TrackedBValue> expanded_args;
    XLSCC_CHECK_EQ(only_slice.function->params().size(),
                   input_tuple_type->size(), func_loc);
    for (int64_t i = 0; i < only_slice.function->params().size(); ++i) {
      expanded_args.push_back(
          pb.TupleIndex(input_select, i, func_loc,
                        /*name=*/
                        absl::StrFormat("expanded_arg_%li", i)));
    }

    TrackedBValue invoke =
        pb.Invoke(ToNativeBValues(expanded_args), only_slice.function, func_loc,
                  /*name*/
                  absl::StrFormat("shared_invoke_%s",
                                  shared_func->clang_decl->getNameAsString()));

    // Route output
    for (const SharedFunctionCall* call : shared_calls_by_func[shared_func]) {
      xls::Node* output_node = call->output.node();
      XLSCC_CHECK_NE(output_node, nullptr, func_loc);
      XLSCC_CHECK(output_node->GetType()->IsEqualTo(invoke.node()->GetType()),
                  func_loc);
      XLS_RETURN_IF_ERROR(output_node->ReplaceUsesWith(invoke.node()));
    }
  }

  return absl::OkStatus();
}

absl::Status NewFSMGenerator::GenerateTransitionFromThisSlice(
    const int64_t from_slice_index, const int64_t num_slice_index_bits,
    TrackedBValue slice_active, TrackedBValue last_op_out_value,
    TrackedBValue next_activation_slice_index, const NewFSMLayout& layout,
    const GeneratedFunctionSlice& slice,
    const absl::flat_hash_map<int64_t, TrackedBValue>&
        state_element_by_jump_slice_index,
    const absl::flat_hash_map<const ContinuationValue*, TrackedBValue>&
        state_element_by_continuation_value,
    absl::btree_multimap<const xls::StateElement*, NextStateValue>&
        extra_next_state_values,
    absl::flat_hash_map<int64_t, TrackedBValue>&
        jump_conditions_by_begin_slice_index,
    std::vector<ConditionalBarrierScope>& conditional_barrier_scope_stack,
    absl::flat_hash_map<PhiConditionCacheKey, TrackedBValue>&
        generated_conditions,
    absl::btree_map<std::tuple<xls::StateElement*, xls::Node*>,
                    absl::btree_set<xls::Node*, NodeIdLessThan>,
                    StateElementAndNodeLessThan>&
        next_value_conditions_by_state_element_and_value,
    xls::ProcBuilder& pb, const xls::SourceInfo& body_loc) {
  const NewFSMActivationTransition& transition =
      layout.transition_by_slice_from_index.at(from_slice_index);
  // This is the activity of the slice before the begin slice for activation
  // transitions.
  TrackedBValue jump_condition = slice_active;

  // Save values before transition for forming state element next values.
  auto value_by_continuation_value_before_transition =
      conditional_barrier_scope_stack.back().value_by_continuation_value;

  if (transition.forward()) {
    XLSCC_CHECK_GE(transition.to_slice, transition.from_slice, body_loc);
  } else {
    XLSCC_CHECK_GE(transition.from_slice, transition.to_slice, body_loc);
    const TrackedBValue jump_state_elem =
        state_element_by_jump_slice_index.at(from_slice_index);
    extra_next_state_values.insert(
        {jump_state_elem.node()->As<xls::StateRead>()->state_element(),
         NextStateValue{
             .value = last_op_out_value,
             .condition = slice_active,
         }});
  }

  if (transition.conditional) {
    XLSCC_CHECK(last_op_out_value.valid(), body_loc);
    XLSCC_CHECK(jump_condition.valid(), body_loc);
    jump_condition =
        pb.And(last_op_out_value, jump_condition, body_loc, /*name=*/
               absl::StrFormat("%s_jump_condition", slice.function->name()));
    XLSCC_CHECK(jump_condition.valid(), body_loc);
    XLSCC_CHECK(jump_condition.GetType()->IsBits(), body_loc);
    XLSCC_CHECK_EQ(jump_condition.GetType()->GetFlatBitCount(), 1, body_loc);

    XLSCC_CHECK(jump_condition.valid(), body_loc);

    if (transition.start_op_type == OpType::kActivationBarrier) {
      // Push the stack of continuation value references
      conditional_barrier_scope_stack.push_back(ConditionalBarrierScope{
          .value_by_continuation_value = conditional_barrier_scope_stack.back()
                                             .value_by_continuation_value,
          .after_conditional_activation_transition =
              pb.Literal(xls::UBits(0, 1), body_loc),
      });

      // Reference state values after the barrier (within the scope).
      // State elements should be applied before begin slice invoke.
      ResetValuesToStateElements(state_element_by_continuation_value,
                                 conditional_barrier_scope_stack);
    } else {
      XLSCC_CHECK_EQ(transition.start_op_type, OpType::kLoopEndJump, body_loc);
      AddToAfterConditionalActivationTransition(
          jump_condition, body_loc, from_slice_index,
          conditional_barrier_scope_stack, pb);
    }
  } else {
    // Can't unmask from slice as it could drive an IO op..
    if (conditional_barrier_scope_stack.size() > 1) {
      return absl::UnimplementedError(
          absl::StrFormat("Unconditional transition from slice %li within "
                          "scope of conditional transition",
                          from_slice_index));
    }

    TrackedBValue zero_cond = pb.Literal(xls::UBits(0, 1), body_loc);
    for (ConditionalBarrierScope& scope : conditional_barrier_scope_stack) {
      scope.after_conditional_activation_transition = zero_cond;
    }

    // Remove direct references from downstream slices to raw outputs
    // of upstream slices by referring directly to the state element,
    // not a select over the state element and raw value.
    //
    // This is only done for values that have state elements. Others,
    // such as direct-ins, still need to be fed through directly.
    //
    // This makes it clear to the optimizer that dependencies don't exist
    // across the unconditional transition.
    ResetValuesToStateElements(state_element_by_continuation_value,
                               conditional_barrier_scope_stack);
  }

  TrackedBValue jump_to_slice_index = pb.Literal(
      xls::UBits(transition.to_slice, num_slice_index_bits), body_loc,
      /*name=*/
      absl::StrFormat("%s_jump_to_slice_index", slice.function->name()));

  extra_next_state_values.insert(
      {next_activation_slice_index.node()
           ->As<xls::StateRead>()
           ->state_element(),
       NextStateValue{
           .priority = std::numeric_limits<int64_t>::max(),
           .value = jump_to_slice_index,
           .condition = jump_condition,
       }});

  jump_conditions_by_begin_slice_index[from_slice_index] = jump_condition;

  if (debug_ir_trace_flags_ & DebugIrTraceFlags_ActivationBarriers) {
    TrackedBValue token = pb.Literal(xls::Value::Token(), body_loc,
                                     /*name=*/"token");
    pb.Trace(token, pb.Literal(xls::UBits(1, 1)),
             /*args=*/
             {jump_condition},
             absl::StrFormat("transition[%li]: jump {:b}", from_slice_index));
  }

  // Sorted for determinism
  absl::btree_set<int64_t> from_jump_slice_indices;
  for (const NewFSMState& state : layout.states) {
    if (state.slice_index != from_slice_index) {
      continue;
    }
    for (const JumpInfo& jump_info : state.jumped_from_slice_indices) {
      from_jump_slice_indices.insert(jump_info.from_slice);
    }
  }

  // Create a next value for each state for this slice
  for (const NewFSMState& state : layout.states) {
    if (state.slice_index != from_slice_index) {
      continue;
    }
    absl::btree_set<int64_t> jumped_from_slice_indices_this_state;
    for (const JumpInfo& jump_info : state.jumped_from_slice_indices) {
      jumped_from_slice_indices_this_state.insert(jump_info.from_slice);
    }

    XLS_ASSIGN_OR_RETURN(
        TrackedBValue state_active_condition,
        GeneratePhiCondition(
            from_jump_slice_indices, jumped_from_slice_indices_this_state,
            state_element_by_jump_slice_index, pb, state.slice_index, body_loc,
            generated_conditions));

    TrackedBValue next_value_condition =
        pb.And(state_active_condition, jump_condition, body_loc,
               /*name=*/GetIRStateName(state));

    for (const ContinuationValue* continuation_out : state.values_to_save) {
      xls::StateElement* state_elem =
          state_element_by_continuation_value.at(continuation_out)
              .node()
              ->As<xls::StateRead>()
              ->state_element();

      std::tuple<xls::StateElement*, xls::Node*> key = {
          state_elem,
          value_by_continuation_value_before_transition.at(continuation_out)
              .node()};

      // Generate next values
      next_value_conditions_by_state_element_and_value[key].insert(
          next_value_condition.node());
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<TrackedBValue> NewFSMGenerator::GeneratePhiCondition(
    const absl::btree_set<int64_t>& from_jump_slice_indices,
    const absl::btree_set<int64_t>& jumped_from_slice_indices_this_state,
    const absl::flat_hash_map<int64_t, TrackedBValue>&
        state_element_by_jump_slice_index,
    xls::ProcBuilder& pb, int64_t slice_index, const xls::SourceInfo& body_loc,
    absl::flat_hash_map<PhiConditionCacheKey, TrackedBValue>&
        phi_condition_cache) {
  PhiConditionCacheKey key = {from_jump_slice_indices,
                              jumped_from_slice_indices_this_state};

  if (phi_condition_cache.contains(key)) {
    return phi_condition_cache.at(key);
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

    TrackedBValue condition_part =
        pb.Eq(jump_state_element,
              pb.Literal(xls::UBits(active_value, 1), body_loc,
                         /*name=*/
                         absl::StrFormat("slice_%li_from_jump_slice_%li_active",
                                         slice_index, from_jump_slice_index)));

    condition =
        pb.And(condition, condition_part, body_loc,
               /*name=*/
               absl::StrFormat(
                   "slice_%li_%s__phi_condition", slice_index,
                   absl::StrJoin(jumped_from_slice_indices_this_state, "_")));
  }

  phi_condition_cache[key] = condition;
  return condition;
}

absl::StatusOr<
    absl::flat_hash_map<int64_t, std::vector<NewFSMGenerator::PhiElement>>>
NewFSMGenerator::GeneratePhiConditions(
    const NewFSMLayout& layout,
    const absl::flat_hash_map<int64_t, TrackedBValue>&
        state_element_by_jump_slice_index,
    xls::ProcBuilder& pb, const xls::SourceInfo& body_loc,
    absl::flat_hash_map<PhiConditionCacheKey, TrackedBValue>&
        phi_condition_cache) {
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

      XLS_ASSIGN_OR_RETURN(
          TrackedBValue condition,
          GeneratePhiCondition(
              from_jump_slice_indices, jumped_from_slice_indices_this_state,
              state_element_by_jump_slice_index, pb, state->slice_index,
              body_loc, phi_condition_cache));

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
    const absl::flat_hash_map<const ContinuationValue*, TrackedBValue>&
        state_element_by_continuation_value,
    const int64_t slice_index, TrackedBValue slice_active,
    TrackedBValue slice_is_current, xls::ProcBuilder& pb,
    const xls::SourceInfo& body_loc) {
  // Ignore IO params
  if (!phi_elements_by_param_node_id.contains(param->id())) {
    return std::nullopt;
  }
  const std::vector<PhiElement>& phi_elements =
      phi_elements_by_param_node_id.at(param->id());

  std::vector<TrackedBValue> phi_conditions;
  std::vector<TrackedBValue> phi_values;

  // Sort by Node ID for determinism.
  struct NodeIdLessThan {
    bool operator()(const xls::Node* a, const xls::Node* b) const {
      return a->id() < b->id();
    }
  };
  struct BValueIdLessThan {
    bool operator()(const TrackedBValue& a, const TrackedBValue& b) const {
      return a.node()->id() < b.node()->id();
    }
  };
  absl::btree_map<xls::Node*, absl::btree_set<TrackedBValue, BValueIdLessThan>,
                  NodeIdLessThan>
      conditions_by_value_node;

  phi_conditions.reserve(phi_elements.size());
  phi_values.reserve(phi_elements.size());

  for (const PhiElement& phi_element : phi_elements) {
    XLSCC_CHECK(value_by_continuation_value.contains(phi_element.value),
                phi_element.value->output_node->loc());

    TrackedBValue value_from_output =
        value_by_continuation_value.at(phi_element.value);

    xls::Node* value_node = value_from_output.node();

    conditions_by_value_node[value_node].insert(phi_element.condition);
  }

  for (auto& [value_node, or_nodes] : conditions_by_value_node) {
    std::vector<NATIVE_BVAL> or_bvals;
    or_bvals.reserve(or_nodes.size());
    for (const TrackedBValue& or_node : or_nodes) {
      or_bvals.push_back(or_node);
    }

    TrackedBValue or_bval =
        pb.Or(absl::MakeSpan(or_bvals), body_loc,
              /*name=*/
              absl::StrFormat("%s_v_%s_or_bval", param->name(),
                              value_node->GetName()));
    phi_conditions.push_back(or_bval);
    phi_values.push_back(TrackedBValue(value_node, &pb));
  }

  std::reverse(phi_conditions.begin(), phi_conditions.end());

  XLSCC_CHECK_GT(phi_values.size(), 0, body_loc);
  XLSCC_CHECK_EQ(phi_values.size(), phi_conditions.size(), body_loc);

  if (phi_values.size() == 1) {
    XLSCC_CHECK(phi_values.at(0).valid(), body_loc);
    return phi_values.at(0);
  }

  TrackedBValue one_hot_select = pb.OneHotSelect(
      pb.Concat(ToNativeBValues(phi_conditions), body_loc),
      ToNativeBValues(phi_values), body_loc,
      /*name=*/
      absl::StrFormat("slice_%li_param_%s_phi", slice_index, param->name()));

  XLSCC_CHECK(one_hot_select.valid(), body_loc);
  return one_hot_select;
}

}  // namespace xlscc
