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

#ifndef XLS_CONTRIB_XLSCC_GENERATE_FSM_H_
#define XLS_CONTRIB_XLSCC_GENERATE_FSM_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "clang/include/clang/AST/Decl.h"
#include "xls/contrib/xlscc/tracked_bvalue.h"
#include "xls/contrib/xlscc/translator_types.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/nodes.h"
#include "xls/ir/source_location.h"
#include "xls/ir/state_element.h"

namespace xlscc {

struct JumpInfo {
  int64_t from_slice = -1;
  int64_t to_slice = -1;

  // Only used internally in layout, should be 0 after layout.
  int64_t count = 0;
};

struct NewFSMState {
  // Conditions to be in the state
  int64_t slice_index = -1;
  std::vector<JumpInfo> jumped_from_slice_indices;

  // Values needed for this state
  absl::flat_hash_map<const xls::Param*, const ContinuationValue*>
      current_inputs_by_input_param;

  // Values used after this state. Ordered for determinism.
  std::vector<const ContinuationValue*> values_to_save;
};

struct NewFSMActivationTransition {
  int64_t from_slice = -1;
  int64_t to_slice = -1;
  bool unconditional_forward = false;
};

struct NewFSMStateElement {
  std::string name;
  xls::Type* type = nullptr;
};

// Provides the necessary information to generate an FSM.
// This includes the states, transitions between states, and the values
// used by and passed between the states.
struct NewFSMLayout {
  std::vector<NewFSMState> states;
  std::vector<NewFSMActivationTransition> state_transitions;
  std::vector<int64_t> all_jump_from_slice_indices;

  absl::flat_hash_map<const IOOp*, int64_t> slice_index_by_after_op;
  absl::flat_hash_map<int64_t, const GeneratedFunctionSlice*> slice_by_index;
  absl::flat_hash_map<const GeneratedFunctionSlice*, int64_t> index_by_slice;
  absl::flat_hash_map<const ContinuationValue*, int64_t>
      output_slice_index_by_value;
  absl::flat_hash_map<int64_t, NewFSMActivationTransition>
      transition_by_slice_from_index;

  std::vector<NewFSMStateElement> state_elements;
  absl::flat_hash_map<const ContinuationValue*, int64_t>
      state_element_by_continuation_value;
};

// This class implements the New FSM in a separate module from the monolithic
// Translator class. GeneratorBase provides necessary common functionality,
// such as error handling.
class NewFSMGenerator : public GeneratorBase {
 public:
  NewFSMGenerator(TranslatorTypeInterface& translator_types,
                  TranslatorIOInterface& translator_io,
                  DebugIrTraceFlags debug_ir_trace_flags,
                  bool split_states_on_channel_ops);

  // Analyzes the control and data flow graphs, ie function slices and
  // continuations, for a translated function, and generates a "layout"
  // for the FSM to implement it.
  //
  // This layout is then intended to be followed in generating the FSM in
  // XLS IR.
  absl::StatusOr<NewFSMLayout> LayoutNewFSM(const GeneratedFunction& func,
                                            const xls::SourceInfo& body_loc);

  // Generate the XLS IR implementation of the FSM for a translated function.
  absl::StatusOr<GenerateFSMInvocationReturn> GenerateNewFSMInvocation(
      const GeneratedFunction* xls_func,
      const std::vector<TrackedBValue>& direct_in_args,
      const absl::flat_hash_map<const clang::NamedDecl*, xls::StateElement*>&
          state_element_for_static,
      const absl::flat_hash_map<const clang::NamedDecl*, int64_t>&
          return_index_for_static,
      xls::ProcBuilder& pb, const xls::SourceInfo& body_loc);

 protected:
  absl::Status LayoutNewFSMStates(NewFSMLayout& layout,
                                  const GeneratedFunction& func,
                                  const xls::SourceInfo& body_loc);

  absl::Status LayoutValuesToSaveForNewFSMStates(
      NewFSMLayout& layout, const GeneratedFunction& func,
      const xls::SourceInfo& body_loc);

  struct PhiElement {
    TrackedBValue condition;
    const ContinuationValue* value;
  };

  absl::StatusOr<absl::flat_hash_map<int64_t, std::vector<PhiElement>>>
  GeneratePhiConditions(const NewFSMLayout& layout,
                        const absl::flat_hash_map<int64_t, TrackedBValue>&
                            state_element_by_jump_slice_index,
                        xls::ProcBuilder& pb, const xls::SourceInfo& body_loc);

  absl::StatusOr<TrackedBValue> GeneratePhiCondition(
      const absl::btree_set<int64_t>& from_jump_slice_indices,
      const absl::btree_set<int64_t>& jumped_from_slice_indices_this_state,
      const absl::flat_hash_map<int64_t, TrackedBValue>&
          state_element_by_jump_slice_index,
      xls::ProcBuilder& pb, int64_t slice_index,
      const xls::SourceInfo& body_loc);

  absl::StatusOr<std::optional<TrackedBValue>> GenerateInputValueInContext(
      const xls::Param* param,
      const absl::flat_hash_map<int64_t, std::vector<PhiElement>>&
          phi_elements_by_param_node_id,
      const absl::flat_hash_map<const ContinuationValue*, TrackedBValue>&
          value_by_continuation_value,
      int64_t slice_index, xls::ProcBuilder& pb,
      const xls::SourceInfo& body_loc);

  absl::Status SetupNewFSMGenerationContext(const GeneratedFunction& func,
                                            NewFSMLayout& layout,
                                            const xls::SourceInfo& body_loc);

  void PrintNewFSMStates(const NewFSMLayout& layout);
  std::string GetStateName(const NewFSMState& state);
  std::string GetIRStateName(const NewFSMState& state);

 private:
  TranslatorIOInterface& translator_io_;
  DebugIrTraceFlags debug_ir_trace_flags_;
  bool split_states_on_channel_ops_;
};

}  // namespace xlscc

#endif  // XLS_CONTRIB_XLSCC_GENERATE_FSM_H_
