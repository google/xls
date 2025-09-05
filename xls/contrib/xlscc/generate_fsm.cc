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
#include "clang/include/clang/AST/Decl.h"
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
                                 DebugIrTraceFlags debug_ir_trace_flags,
                                 bool split_states_on_channel_ops)
    : GeneratorBase(translator_types),
      translator_io_(translator_io),
      debug_ir_trace_flags_(debug_ir_trace_flags),
      split_states_on_channel_ops_(split_states_on_channel_ops) {}

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

  absl::flat_hash_set<const IOChannel*> channels_used_this_activation;

  // Record transitions across activations
  // TODO(seanhaskell): Add from last to first for statics
  bool first_slice = true;
  for (const GeneratedFunctionSlice& slice : func.slices) {
    if (first_slice) {
      first_slice = false;
      continue;
    }
    const IOOp* after_op = slice.after_op;
    if (after_op == nullptr) {
      // This is the "before" slice that can be transitioned to in an IO
      // activation transition.
      const int64_t before_io_slice_index = ret.index_by_slice.at(&slice);
      const int64_t after_io_slice_index = before_io_slice_index + 1;
      const GeneratedFunctionSlice* after_io_slice =
          ret.slice_by_index.at(after_io_slice_index);
      const IOOp* op_after = after_io_slice->after_op;
      CHECK_NE(op_after, nullptr);
      if (!channels_used_this_activation.contains(op_after->channel)) {
        channels_used_this_activation.insert(op_after->channel);
        continue;
      }
      NewFSMActivationTransition transition;
      CHECK_GT(before_io_slice_index, 0);
      transition.from_slice = before_io_slice_index - 1;
      transition.to_slice = before_io_slice_index;
      transition.unconditional_forward = true;
      ret.transition_by_slice_from_index[transition.from_slice] = transition;
      ret.state_transitions.push_back(transition);
      ret.all_jump_from_slice_indices.push_back(transition.from_slice);

      // All channels are cleared after the transition
      channels_used_this_activation.clear();
      continue;
    }
    // This is optional, so doesn't reset channels_used_this_activation
    if (after_op->op == OpType::kLoopEndJump) {
      const int64_t end_jump_slice_index = ret.index_by_slice.at(&slice);
      XLSCC_CHECK_GE(end_jump_slice_index, 1, body_loc);
      const IOOp* const loop_begin_op = after_op->loop_op_paired;
      XLSCC_CHECK_NE(loop_begin_op, nullptr, body_loc);
      const int64_t begin_slice_index =
          ret.slice_index_by_after_op.at(loop_begin_op);

      // Feedback is from the slice after begin to the slice before end jump
      NewFSMActivationTransition transition;
      transition.from_slice = end_jump_slice_index - 1;
      transition.to_slice = begin_slice_index;
      transition.unconditional_forward = false;
      ret.transition_by_slice_from_index[transition.from_slice] = transition;
      ret.state_transitions.push_back(transition);
      ret.all_jump_from_slice_indices.push_back(transition.from_slice);
      continue;
    }
  }

  if (debug_ir_trace_flags_ & DebugIrTraceFlags_FSMStates) {
    LOG(INFO) << "FSM transitions:";
    for (const NewFSMActivationTransition& transition : ret.state_transitions) {
      LOG(INFO) << absl::StrFormat("  %li -> %li (unconditional? %i)",
                                   transition.from_slice, transition.to_slice,
                                   (int)transition.unconditional_forward);
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

  // Plan the state elements
  //
  // State elements may be shared by multiple continuation values if the values
  // have the same type and are not saved in the same state (slice + jump flags)
  //
  // For optimization purposes, such as narrowing, it is better that the values
  // saved in a state element share semantics. Therefore, Clang NamedDecls are
  // used to identify values that may share a state element.
  absl::flat_hash_map<const clang::NamedDecl*, std::vector<int64_t>>
      state_element_indices_by_decl;

  for (const NewFSMState& state : ret.states) {
    // Only need to save continuation values on activation transitions
    if (!ret.transition_by_slice_from_index.contains(state.slice_index)) {
      continue;
    }

    // A state element can only be used once in a given transition
    absl::flat_hash_set<int64_t> used_state_element_indices;

    // Mark reserved elements
    for (const ContinuationValue* value : state.values_to_save) {
      if (!ret.state_element_by_continuation_value.contains(value)) {
        continue;
      }
      used_state_element_indices.insert(
          ret.state_element_by_continuation_value.at(value));
    }

    for (const ContinuationValue* value : state.values_to_save) {
      if (ret.state_element_by_continuation_value.contains(value)) {
        continue;
      }
      // This value has not already been assigned a state element
      // Try to find state elements to share by decl
      std::optional<int64_t> found_element_by_decl = std::nullopt;
      std::vector<const clang::NamedDecl*> decls;
      for (const clang::NamedDecl* decl : value->decls) {
        decls.push_back(decl);
      }
      func.SortNamesDeterministically(decls);

      for (const clang::NamedDecl* decl : decls) {
        if (!state_element_indices_by_decl.contains(decl)) {
          continue;
        }
        const std::vector<int64_t>& elements_for_this_decl =
            state_element_indices_by_decl.at(decl);
        for (const int64_t element_for_decl_index : elements_for_this_decl) {
          if (used_state_element_indices.contains(element_for_decl_index)) {
            continue;
          }
          XLSCC_CHECK(ret.state_elements.at(element_for_decl_index)
                          .type->IsEqualTo(value->output_node->GetType()),
                      body_loc);
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
        NewFSMStateElement state_element = {
            .name =
                absl::StrFormat("%s_slc_%li", value->name, state.slice_index),
            .type = value->output_node->GetType(),
        };
        ret.state_elements.push_back(state_element);
        element_index = ret.state_elements.size() - 1;
        ret.state_element_by_continuation_value[value] = element_index;
      }

      // Mark element used for this value in this transition
      XLSCC_CHECK_GE(element_index, 0, body_loc);
      XLSCC_CHECK_LT(element_index, ret.state_elements.size(), body_loc);
      ret.state_element_by_continuation_value[value] = element_index;
      used_state_element_indices.insert(element_index);
      for (const clang::NamedDecl* decl : decls) {
        state_element_indices_by_decl[decl].push_back(element_index);
      }
    }
  }

  if (debug_ir_trace_flags_ & DebugIrTraceFlags_FSMStates) {
    int64_t total_bits = 0;
    for (const NewFSMStateElement& elem : ret.state_elements) {
      total_bits += elem.type->GetFlatBitCount();
    }
    LOG(INFO) << "State elements allocated, total " << total_bits << " bits:";
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
      LOG(INFO) << absl::StrFormat("    %s (%s), values: %s", elem.name,
                                   elem.type->ToString(),
                                   absl::StrJoin(value_names, ", "));
    }
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

        if (transition.unconditional_forward) {
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

absl::Status NewFSMGenerator::LayoutValuesToSaveForNewFSMStates(
    NewFSMLayout& layout, const GeneratedFunction& func,
    const xls::SourceInfo& body_loc) {
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
    std::vector<std::string> ret;
    for (const JumpInfo& jump_info : jump_infos) {
      std::string str;
      absl::StrAppendFormat(&str, "{%li,c = %li}", jump_info.from_slice,
                            jump_info.count);
      ret.push_back(str);
    }
    return absl::StrJoin(ret, ",");
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
          "    p %s: slice %li %s", input_param->name().data(),
          layout.output_slice_index_by_value.at(continuation_out),
          continuation_out->name.c_str());
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
          "    v %s (%p) slice %li (%li bits)", value->name.c_str(), value,
          layout.output_slice_index_by_value.at(value),
          value->output_node->GetType()->GetFlatBitCount());
    }
  }
}

absl::StatusOr<GenerateFSMInvocationReturn>
NewFSMGenerator::GenerateNewFSMInvocation(
    const GeneratedFunction* xls_func,
    const std::vector<TrackedBValue>& direct_in_args,
    const absl::flat_hash_map<const clang::NamedDecl*, xls::StateElement*>&
        state_element_for_static,
    const absl::flat_hash_map<const clang::NamedDecl*, int64_t>&
        return_index_for_static,
    xls::ProcBuilder& pb, const xls::SourceInfo& body_loc) {
  XLSCC_CHECK_NE(xls_func, nullptr, body_loc);
  const GeneratedFunction& func = *xls_func;

  NewFSMLayout layout;
  XLS_ASSIGN_OR_RETURN(layout, LayoutNewFSM(func, body_loc));

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

  TrackedBValue next_activation_slice_index = pb.StateElement(
      "__next_activation_slice",
      xls::Value(xls::UBits(0, num_slice_index_bits)), body_loc);

  TrackedBValue first_slice_index =
      pb.Literal(xls::UBits(0, num_slice_index_bits), body_loc);

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
    const NewFSMActivationTransition& transition =
        layout.transition_by_slice_from_index.at(jump_slice_index);

    if (transition.unconditional_forward) {
      continue;
    }

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

  absl::flat_hash_map<const ContinuationValue*, TrackedBValue>
      state_element_by_continuation_value;

  absl::flat_hash_map<int64_t, std::vector<const ContinuationValue*>>
      values_by_state_element_index;

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
    TrackedBValue xls_state_element = pb.StateElement(
        state_element.name, xls::ZeroOfType(state_element.type), body_loc);

    for (const ContinuationValue* value :
         values_by_state_element_index.at(state_element_index)) {
      state_element_by_continuation_value[value] = xls_state_element;
    }
  }

  // Set values initially to state elements, so that feedbacks
  // come from state. These will be overwritten for feedforwards as slices
  // are generated.
  value_by_continuation_value = state_element_by_continuation_value;

  TrackedBValue last_op_out_value;
  TrackedBValue after_activation_transition =
      pb.Literal(xls::UBits(0, 1), body_loc);

  for (int64_t slice_index = 0; slice_index < func.slices.size();
       ++slice_index) {
    const bool is_last_slice = (slice_index == func.slices.size() - 1);

    const TrackedBValue slice_is_current = pb.ULe(
        next_activation_slice_index,
        pb.Literal(xls::UBits(slice_index, num_slice_index_bits)), body_loc,
        /*name=*/absl::StrFormat("slice_%li_is_current", slice_index));

    TrackedBValue slice_active = pb.And(
        {slice_is_current,
         pb.Not(after_activation_transition, body_loc, /*name=*/
                absl::StrFormat("slice_%li_not_after_activation_transition",
                                slice_index))},
        body_loc,
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

    // Add direct-ins (and top class input) to first slice params
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

      // With split states mode on, ordering within a channel is enforced via
      // activation barriers.
      if (!split_states_on_channel_ops_) {
        token = io_return.token_out;
      }

      // Add IO parameter if applicable
      if (io_return.received_value.valid()) {
        invoke_params.push_back(io_return.received_value);
      }
    }

    // Add statics
    for (const clang::NamedDecl* decl : slice.static_values) {
      xls::StateElement* state_element = state_element_for_static.at(decl);
      xls::StateRead* state_read = pb.proc()->GetStateRead(state_element);
      TrackedBValue prev_val(state_read, &pb);
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
    }

    if (is_last_slice) {
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
          output_value = ret_tup;
        } else {
          output_value = pb.TupleIndex(
              ret_tup, static_index, body_loc,
              /*name=*/
              absl::StrFormat("return__%s", static_decl->getNameAsString()));
        }

        XLSCC_CHECK(output_value.GetType()->IsEqualTo(
                        return_values.at(return_index).GetType()),
                    body_loc);

        return_values.at(return_index) = output_value;
      }
    }

    if (layout.transition_by_slice_from_index.contains(slice_index)) {
      const NewFSMActivationTransition& transition =
          layout.transition_by_slice_from_index.at(slice_index);
      TrackedBValue jump_condition = slice_active;
      if (transition.unconditional_forward) {
        XLSCC_CHECK_GE(transition.to_slice, transition.from_slice, body_loc);
      } else {
        XLSCC_CHECK_GE(transition.from_slice, transition.to_slice, body_loc);
        XLSCC_CHECK(last_op_out_value.valid(), body_loc);
        jump_condition = pb.And(
            last_op_out_value, jump_condition, body_loc, /*name=*/
            absl::StrFormat("%s_jump_condition", slice.function->name()));
        XLSCC_CHECK(jump_condition.valid(), body_loc);
        XLSCC_CHECK(jump_condition.GetType()->IsBits(), body_loc);
        XLSCC_CHECK_EQ(jump_condition.GetType()->GetFlatBitCount(), 1,
                       body_loc);
        const TrackedBValue jump_state_elem =
            state_element_by_jump_slice_index.at(slice_index);
        extra_next_state_values.insert(
            {jump_state_elem.node()->As<xls::StateRead>()->state_element(),
             NextStateValue{
                 .value = last_op_out_value,
                 .condition = slice_active,
             }});
      }

      XLSCC_CHECK(jump_condition.valid(), body_loc);

      after_activation_transition =
          pb.Or(after_activation_transition, jump_condition, body_loc,
                /*name=*/
                absl::StrFormat("after_%li_after_activation_transition",
                                slice_index));

      debug_did_jump = jump_condition;

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

      // Sorted for determinism
      absl::btree_set<int64_t> from_jump_slice_indices;
      for (const NewFSMState& state : layout.states) {
        if (state.slice_index != slice_index) {
          continue;
        }
        for (const JumpInfo& jump_info : state.jumped_from_slice_indices) {
          from_jump_slice_indices.insert(jump_info.from_slice);
        }
      }

      // Create a next value for each state for this slice
      for (const NewFSMState& state : layout.states) {
        if (state.slice_index != slice_index) {
          continue;
        }

        absl::btree_set<int64_t> jumped_from_slice_indices_this_state;
        for (const JumpInfo& jump_info : state.jumped_from_slice_indices) {
          jumped_from_slice_indices_this_state.insert(jump_info.from_slice);
        }

        XLS_ASSIGN_OR_RETURN(
            TrackedBValue state_active_condition,
            GeneratePhiCondition(from_jump_slice_indices,
                                 jumped_from_slice_indices_this_state,
                                 state_element_by_jump_slice_index, pb,
                                 state.slice_index, body_loc));

        TrackedBValue next_value_condition =
            pb.And(state_active_condition, jump_condition, body_loc,
                   /*name=*/GetIRStateName(state));

        for (const ContinuationValue* continuation_out : state.values_to_save) {
          // Generate next values for state elements
          NextStateValue next_value = {
              .priority = 0,
              .value = value_by_continuation_value.at(continuation_out),
              .condition = next_value_condition,
          };

          xls::StateElement* state_elem =
              state_element_by_continuation_value.at(continuation_out)
                  .node()
                  ->As<xls::StateRead>()
                  ->state_element();

          // Generate next values
          extra_next_state_values.insert({state_elem, next_value});
        }
      }
    }
  }

  // Set next slice index
  const TrackedBValue finished_iteration =
      pb.Not(after_activation_transition, body_loc,
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

absl::StatusOr<TrackedBValue> NewFSMGenerator::GeneratePhiCondition(
    const absl::btree_set<int64_t>& from_jump_slice_indices,
    const absl::btree_set<int64_t>& jumped_from_slice_indices_this_state,
    const absl::flat_hash_map<int64_t, TrackedBValue>&
        state_element_by_jump_slice_index,
    xls::ProcBuilder& pb, int64_t slice_index,
    const xls::SourceInfo& body_loc) {
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

  return condition;
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

      XLS_ASSIGN_OR_RETURN(
          TrackedBValue condition,
          GeneratePhiCondition(from_jump_slice_indices,
                               jumped_from_slice_indices_this_state,
                               state_element_by_jump_slice_index, pb,
                               state->slice_index, body_loc));

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
    XLSCC_CHECK(value_by_continuation_value.contains(phi_element.value),
                phi_element.value->output_node->loc());
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
