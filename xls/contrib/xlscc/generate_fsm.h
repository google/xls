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
#include <string>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
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

// For determinism
struct ContinuationValuePointerComparator {
  bool operator()(const ContinuationValue* lhs,
                  const ContinuationValue* rhs) const {
    return lhs->output_node->id() < rhs->output_node->id();
  }
};

struct NewFSMState {
  // Conditions to be in the state
  int64_t slice_index = -1;
  std::vector<JumpInfo> jumped_from_slice_indices;

  // Values needed for this state
  absl::flat_hash_map<const xls::Param*, const ContinuationValue*>
      current_inputs_by_input_param;

  // Values used after this state. Ordered for determinism.
  absl::btree_set<const ContinuationValue*, ContinuationValuePointerComparator>
      values_to_save;

  StateId GetStateId() const {
    StateId state_id = {
        .slice_index = slice_index,
    };
    for (const JumpInfo& jump_info : jumped_from_slice_indices) {
      state_id.from_jump_slice_indices.insert(JumpId{
          .from_slice_index = jump_info.from_slice,
          .count = jump_info.count,
      });
    }
    return state_id;
  }
};

struct NewFSMActivationTransition {
  int64_t from_slice = -1;
  int64_t to_slice = -1;
  bool conditional = false;
  OpType start_op_type = OpType::kNull;

  bool forward() const {
    CHECK_NE(start_op_type, OpType::kNull);
    return start_op_type == OpType::kActivationBarrier;
  }
};

struct NewFSMStateElement {
  std::string name;
  xls::Type* type = nullptr;
  xls::StateElement* existing_state_element = nullptr;
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
                  DebugIrTraceFlags debug_ir_trace_flags);

  // Analyzes the control and data flow graphs, ie function slices and
  // continuations, for a translated function, and generates a "layout"
  // for the FSM to implement it.
  //
  // This layout is then intended to be followed in generating the FSM in
  // XLS IR.
  absl::StatusOr<NewFSMLayout> LayoutNewFSM(
      const GeneratedFunction& func,
      const absl::flat_hash_map<DeclLeaf, xls::StateElement*>&
          state_element_for_static,
      const xls::SourceInfo& body_loc);

  absl::Status LayoutNewFSMNoStateElements(
      NewFSMLayout& layout, const std::list<GeneratedFunctionSlice>& slices,
      const xls::SourceInfo& body_loc);

  absl::Status ValidateStateInputs(const GeneratedFunction& func,
                                   const NewFSMLayout& layout,
                                   const xls::SourceInfo& body_loc) const;

  // Generate the XLS IR implementation of the FSM for a translated function.
  absl::StatusOr<GenerateFSMInvocationReturn> GenerateNewFSMInvocation(
      const GeneratedFunction* xls_func,
      const std::vector<TrackedBValue>& direct_in_args,
      const absl::flat_hash_map<DeclLeaf, xls::StateElement*>&
          state_element_for_static,
      const absl::flat_hash_map<const clang::NamedDecl*, xls::Type*>&
          type_for_static,
      const absl::flat_hash_map<const clang::NamedDecl*, int64_t>&
          return_index_for_static,
      xls::ProcBuilder& pb, const xls::SourceInfo& body_loc);

  absl::Status GenerateExtractStaticReturns(
      TrackedBValue last_slice_return_value,
      const absl::flat_hash_map<const clang::NamedDecl*, int64_t>&
          return_index_for_static,
      std::vector<TrackedBValue>& return_values, xls::ProcBuilder& pb,
      const xls::SourceInfo& body_loc);

  void PrintNewFSMStates(const NewFSMLayout& layout);

 protected:
  absl::Status LayoutNewFSMTransitions(
      NewFSMLayout& layout, const std::list<GeneratedFunctionSlice>& slices,
      const xls::SourceInfo& body_loc);

  absl::Status LayoutNewFSMStates(
      NewFSMLayout& layout, const std::list<GeneratedFunctionSlice>& slices,
      const xls::SourceInfo& body_loc);

  absl::Status LayoutNewFSMStateElements(
      NewFSMLayout& layout, const GeneratedFunction& func,
      const absl::flat_hash_map<DeclLeaf, xls::StateElement*>&
          state_element_for_static,
      const xls::SourceInfo& body_loc);

  absl::Status LayoutValuesToSaveForNewFSMStates(
      NewFSMLayout& layout, const xls::SourceInfo& body_loc);

  struct PhiElement {
    TrackedBValue condition;
    const ContinuationValue* value;
  };

  typedef std::tuple<absl::btree_set<int64_t>, absl::btree_set<int64_t>>
      PhiConditionCacheKey;

  absl::StatusOr<absl::flat_hash_map<int64_t, std::vector<PhiElement>>>
  GeneratePhiConditions(
      const NewFSMLayout& layout,
      const absl::flat_hash_map<int64_t, TrackedBValue>&
          state_element_by_jump_slice_index,
      xls::ProcBuilder& pb, const xls::SourceInfo& body_loc,
      absl::flat_hash_map<PhiConditionCacheKey, TrackedBValue>&
          phi_condition_cache);

  absl::StatusOr<TrackedBValue> GeneratePhiCondition(
      const absl::btree_set<int64_t>& from_jump_slice_indices,
      const absl::btree_set<int64_t>& jumped_from_slice_indices_this_state,
      const absl::flat_hash_map<int64_t, TrackedBValue>&
          state_element_by_jump_slice_index,
      xls::ProcBuilder& pb, int64_t slice_index,
      const xls::SourceInfo& body_loc,
      absl::flat_hash_map<PhiConditionCacheKey, TrackedBValue>&
          phi_condition_cache);

  absl::StatusOr<std::optional<TrackedBValue>> GenerateInputValueInContext(
      const xls::Param* param,
      const absl::flat_hash_map<int64_t, std::vector<PhiElement>>&
          phi_elements_by_param_node_id,
      const absl::flat_hash_map<const ContinuationValue*, TrackedBValue>&
          value_by_continuation_value,
      const absl::flat_hash_map<const ContinuationValue*, TrackedBValue>&
          state_element_by_continuation_value,
      int64_t slice_index, TrackedBValue slice_active,
      TrackedBValue slice_is_current, xls::ProcBuilder& pb,
      const xls::SourceInfo& body_loc);

  absl::Status SetupNewFSMGenerationContext(
      const std::list<GeneratedFunctionSlice>& slices, NewFSMLayout& layout,
      const xls::SourceInfo& body_loc);

  std::string GetStateName(const NewFSMState& state);
  std::string GetIRStateName(const NewFSMState& state);

  // The value from the current activation's perspective,
  // either outputted from invoke or state element.
  typedef absl::flat_hash_map<const ContinuationValue*, TrackedBValue>
      ContinuationValueBValMap;

  struct ConditionalBarrierScope {
    ContinuationValueBValMap value_by_continuation_value;
    TrackedBValue after_conditional_activation_transition;
  };

  // Sort by Node ID and StateElement name for determinism.
  struct StateElementAndNodeLessThan {
    bool operator()(const std::tuple<xls::StateElement*, xls::Node*>& a,
                    const std::tuple<xls::StateElement*, xls::Node*>& b) const {
      const auto& [a_elem, a_node] = a;
      const auto& [b_elem, b_node] = b;
      if (a_elem->name() != b_elem->name()) {
        return a_elem->name() < b_elem->name();
      }
      return a_node->id() < b_node->id();
    }
  };

  struct NodeIdLessThan {
    bool operator()(const xls::Node* a, const xls::Node* b) const {
      return a->id() < b->id();
    }
  };

  absl::Status GenerateTransitionFromThisSlice(
      int64_t from_slice_index, int64_t num_slice_index_bits,
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
      xls::ProcBuilder& pb, const xls::SourceInfo& body_loc);

  void AddToAfterConditionalActivationTransition(
      TrackedBValue condition, const xls::SourceInfo& loc, int64_t slice_index,
      std::vector<ConditionalBarrierScope>& conditional_barrier_scope_stack,
      xls::ProcBuilder& pb);

  void ResetValuesToStateElements(
      const absl::flat_hash_map<const ContinuationValue*, TrackedBValue>&
          state_element_by_continuation_value,
      std::vector<ConditionalBarrierScope>& conditional_barrier_scope_stack);

  struct BarrierScopeMask {
    // scope includes start_slice, but not end_slice
    int64_t start_slice;
    int64_t end_slice;
    TrackedBValue mask_bval;
  };

  absl::StatusOr<absl::flat_hash_map<const IOOp*, BarrierScopeMask>>
  GenerateBarrierScopeMasksByStartOp(int64_t num_slice_index_bits,
                                     TrackedBValue next_activation_slice_index,
                                     const GeneratedFunction& func,
                                     const NewFSMLayout& layout,
                                     xls::ProcBuilder& pb,
                                     const xls::SourceInfo& body_loc);

  absl::StatusOr<TrackedBValue> GenerateBarrierSliceMask(
      int64_t slice_index,
      const absl::flat_hash_map<const IOOp*, BarrierScopeMask>&
          scopes_by_start_op,
      const GeneratedFunction& func, const NewFSMLayout& layout,
      xls::ProcBuilder& pb, const xls::SourceInfo& body_loc);

 private:
  TranslatorIOInterface& translator_io_;
  DebugIrTraceFlags debug_ir_trace_flags_;
};

}  // namespace xlscc

#endif  // XLS_CONTRIB_XLSCC_GENERATE_FSM_H_
