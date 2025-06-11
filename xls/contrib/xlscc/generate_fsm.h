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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/contrib/xlscc/tracked_bvalue.h"
#include "xls/contrib/xlscc/translator_types.h"
#include "xls/ir/function_builder.h"

namespace xlscc {

struct NextStateValue {
  // When the condition is true for multiple next state values,
  // the one with the lower priority is taken.
  // Whenever more than one next value is specified,
  // a priority must be specified, and all conditions must be valid.
  int64_t priority = -1L;
  std::string extra_label = "";
  TrackedBValue value;
  // condition being invalid indicates unconditional update (literal 1)
  TrackedBValue condition;
};

struct GenerateFSMInvocationReturn {
  TrackedBValue return_value;
  TrackedBValue returns_this_activation;
  absl::btree_multimap<const xls::StateElement*, NextStateValue>
      extra_next_state_values;
};

// This class implements the New FSM in a separate module from the monolithic
// Translator class. GeneratorBase provides necessary common functionality,
// such as error handling.
class NewFSMGenerator : public GeneratorBase {
 public:
  NewFSMGenerator(TranslatorTypeInterface& translator_types,
                  TranslatorIOInterface& translator_io,
                  DebugIrTraceFlags debug_ir_trace_flags);

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
      const std::vector<TrackedBValue>& direct_in_args, xls::ProcBuilder& pb,
      const xls::SourceInfo& body_loc);

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

  absl::StatusOr<std::optional<TrackedBValue>> GenerateInputValueInContext(
      const xls::Param* param,
      const absl::flat_hash_map<int64_t, std::vector<PhiElement>>&
          phi_elements_by_param_node_id,
      const absl::flat_hash_map<const ContinuationValue*, TrackedBValue>&
          value_by_continuation_value,
      const int64_t slice_index, xls::ProcBuilder& pb,
      const xls::SourceInfo& body_loc);

  absl::Status SetupNewFSMGenerationContext(const GeneratedFunction& func,
                                            NewFSMLayout& layout,
                                            const xls::SourceInfo& body_loc);

  void PrintNewFSMStates(const NewFSMLayout& layout);

 private:
  TranslatorIOInterface& translator_io_;
  DebugIrTraceFlags debug_ir_trace_flags_;
};

}  // namespace xlscc

#endif  // XLS_CONTRIB_XLSCC_GENERATE_FSM_H_
