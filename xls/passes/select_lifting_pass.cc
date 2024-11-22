// Copyright 2024 The XLS Authors
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

#include "xls/passes/select_lifting_pass.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"

namespace xls {

namespace {

struct TransformationResult {
  bool was_code_modified;
  absl::btree_set<Node *, Node::NodeIdLessThan> new_selects_to_consider;
  absl::flat_hash_set<Node *> nodes_to_delete;

  TransformationResult() : was_code_modified{false} {}
};

struct LiftableSelectOperandInfo {
  Node *shared_input;
  Op op;
};

std::optional<Node *> GetDefaultValue(Node *select) {
  if (select->Is<PrioritySelect>()) {
    return select->As<PrioritySelect>()->default_value();
  }
  CHECK(select->Is<Select>());
  return select->As<Select>()->default_value();
}

absl::Span<Node *const> GetCases(Node *select) {
  if (select->Is<PrioritySelect>()) {
    return select->As<PrioritySelect>()->cases();
  }
  CHECK(select->Is<Select>());
  return select->As<Select>()->cases();
}

bool MatchesIndexBitwidth(ArrayIndex *ai, int64_t shared_index_bitwidth) {
  absl::Span<Node *const> current_case_indices = ai->indices();
  if (current_case_indices.length() != 1) {
    // Property 1 does not hold
    VLOG(3) << "        The input \"" << ai->ToString()
            << "\" uses more than one index";
    return false;
  }
  Node *current_case_index = current_case_indices.at(0);
  Type *current_case_index_type = current_case_index->GetType();
  int64_t current_index_bitwidth = current_case_index_type->GetFlatBitCount();
  if (current_index_bitwidth != shared_index_bitwidth) {
    // Property 1 does not hold
    VLOG(3) << "        The input \"" << ai->ToString()
            << "\" uses an index with a different bitwidth than the one used "
               "by the other cases of the \"select\" node";
    return false;
  }

  return true;
}

std::optional<Node *> ApplicabilityGuardForArrayIndex(
    absl::Span<Node *const> cases, std::optional<Node *> default_case) {
  // Only "select" nodes with the following properties can be optimized by this
  // transformations.
  //
  // Property 0: only "select" nodes with accesses to the same array as cases
  //             can be optimized by this transformation
  //
  // Property 1: the indices of the ArrayIndex nodes that are
  //             the input of the select have the same bitwidth.
  //             Note: it is possible to generalize the transformation to handle
  //             the case where these bitwidths differ. Doing so will remove the
  //             need for this property.
  //
  // The code below checks these properties for the target "select" node.
  //
  // Fetch the aspects of the first case of the "select" node that will have to
  // be shared between all the rest of the "select" inputs.
  ArrayIndex *first_case = cases[0]->As<ArrayIndex>();
  Node *shared_array_ref = first_case->operand(0);
  absl::Span<Node *const> first_case_indices = first_case->indices();
  Node *first_case_first_index = first_case_indices.at(0);
  Type *first_case_first_index_type = first_case_first_index->GetType();
  int64_t shared_index_bitwidth =
      first_case_first_index_type->GetFlatBitCount();

  // Check Property 0
  VLOG(3) << "      Array = " << shared_array_ref->ToString();
  for (uint32_t index = 1; index < cases.length(); index++) {
    // Notice that this case is guaranteed to succeed as all inputs of the
    // "select" are guaranteed to have the same operation and type at this
    // point.
    ArrayIndex *current_case = cases[index]->As<ArrayIndex>();

    // Check Property 0
    if (current_case->operand(0) != shared_array_ref) {
      // Property 0 does not hold
      VLOG(3) << "        The case " << index
              << " accesses an array that is different than the other inputs "
                 "of the \"select\" node";
      return std::nullopt;
    }

    // Check Property 1
    if (!MatchesIndexBitwidth(current_case, shared_index_bitwidth)) {
      // Property 1 does not hold
      VLOG(3) << "        The case " << index << " uses more than one index";
      return std::nullopt;
    }
  }
  if (default_case) {
    ArrayIndex *default_case_as_array_index = (*default_case)->As<ArrayIndex>();

    // Check Property 0
    if (default_case_as_array_index->operand(0) != shared_array_ref) {
      // Property 0 does not hold
      VLOG(3) << "        The default case accesses an array that is different "
                 "than the other inputs of the \"select\" node";
      return std::nullopt;
    }

    // Check Property 1
    if (!MatchesIndexBitwidth(default_case_as_array_index,
                              shared_index_bitwidth)) {
      // Property 1 does not hold
      VLOG(3) << "        Property 1 (see comments in the code) does not hold "
                 "for the default case of the \"select\" node";
      return std::nullopt;
    }
  }
  VLOG(3) << "        Passed the check";

  return shared_array_ref;
}

std::optional<Op> SharedOperation(absl::Span<Node *const> cases,
                                  std::optional<Node *> default_case) {
  // All inputs of a "select" node must have the same operation (e.g.,
  // ArrayIndex) and the same type. Notice that a future improvement could relax
  // the same-type constraint.
  //
  // Fetch the operation and the type of the first element
  Type *shared_type = cases[0]->GetType();
  Op shared_op = cases[0]->op();

  // Check that all inputs within the cases of the "select" have the same type
  // and op.
  for (Node *current_input : cases) {
    if (current_input->GetType() != shared_type) {
      return std::nullopt;
    }
    if (current_input->op() != shared_op) {
      return std::nullopt;
    }
  }

  // Check that the default value has the same type and op of the input cases of
  // the "select".
  if (default_case) {
    Node *default_case_value = default_case.value();
    if (default_case_value->GetType() != shared_type) {
      return std::nullopt;
    }
    if (default_case_value->op() != shared_op) {
      return std::nullopt;
    }
  }

  return shared_op;
}

absl::StatusOr<std::optional<LiftableSelectOperandInfo>> CanLiftSelect(
    FunctionBase *func, Node *select_to_optimize) {
  VLOG(3) << "  Checking the applicability guard";

  // Only "select" nodes with specific properties can be optimized by this
  // transformation.
  //
  // Shared properties that must hold for all cases:
  //
  // Property A:
  //   Only "select" nodes with at least one input case can be optimized.
  //
  // Property B:
  //   Only "select" nodes with the same node type for all its inputs can
  //   be optimized.
  //
  //
  //
  // There are more properties that must hold for the transformation to be
  // applicable. Such properties are specific to the node type of the inputs of
  // the select.
  //
  // The code below checks these properties for the "select" node given as input
  //
  // We first collect all input nodes of the select we must check.
  // Then, we check them all by first checking the shared property above, and
  // then by applying pattern matching that is specific to the type of the
  // inputs of the target "select".
  absl::Span<Node *const> select_cases = GetCases(select_to_optimize);
  std::optional<Node *> default_value = GetDefaultValue(select_to_optimize);

  // Check the shared property A
  if (select_cases.empty()) {
    VLOG(3) << "    The transformation is not applicable: the select does not "
               "have input cases";
    return std::nullopt;
  }

  // Check the shared property B
  std::optional<Op> shared_input_op =
      SharedOperation(select_cases, default_value);
  if (!shared_input_op) {
    VLOG(3) << "    The transformation is not applicable: not all inputs have "
               "the same type and operation";
    return std::nullopt;
  }

  // Check the type-specific constraints
  std::optional<Node *> shared_input_node;
  switch (*shared_input_op) {
    case Op::kArrayIndex:
      shared_input_node =
          ApplicabilityGuardForArrayIndex(select_cases, default_value);
      break;

    default:
      VLOG(3) << "    The current input of the select is not handled";
      return std::nullopt;
  }
  if (!shared_input_node) {
    return std::nullopt;
  }

  return LiftableSelectOperandInfo{.shared_input = *shared_input_node,
                                   .op = *shared_input_op};
}

bool ProfitabilityGuardForArrayIndex(FunctionBase *func,
                                     Node *select_to_optimize,
                                     Node *array_reference) {
  // The next properties when hold guarantee that it is profitable to transform
  // the "select" node.
  //
  // Property 0: array accesses (i.e., ArrayIndex) within the cases of the
  // select are not all literals. This is because ArrayIndex with only literals
  // as indices are free.
  //
  // Property 1: array accesses (i.e., ArrayIndex) within the cases must be only
  // used by the "select" given as input. This is because this property
  // guarantees that such "select" node becomes dead after applying this
  // transformation.
  //
  // Property 2: the bitwidth for the indices is less or equal to the bitwidth
  //             of a single element of the array.
  //
  //             When property 2 holds, this transformation is always
  //             beneficial. To understand why, let's look at the
  //             transformation. This transformation aims to reduce area and it
  //             does so by transforming
  //                  v = sel(c, a[i], a[j])
  //             to
  //                  t = a[sel(c, i, j)]
  //                  v = a[t]
  //
  //             To understand when it is profitable to do this transformation,
  //             let's define a few terms:
  //             - BWE = bitwdith of a single element of the array
  //             - BWI = bitwidth of the indices
  //             - SC = log of the number of clauses in the select
  //             - AS = log of number of elements of the array
  //
  //             Next, let's analyze the area of the two versions of the
  //             code, original and transformed.
  //             - Original code.
  //               The area of the select is a function of BWE and SC; so
  //                  s = f(BWE, SC)
  //               The area of each array index is a function of BWE and AS; so
  //                  a = g(BWE, AS)
  //               The total area of the original version of the code is:
  //                  s + 2*a
  //
  //              - Transformed code:
  //               The area of the select is a function of BWE and SC; so
  //                  s' = f(BWI, SC)
  //               The area of each array index is a function of BWE and AS; so
  //                  a = g(BWE, AS)
  //               The total area of the transformed version of the code is:
  //                  s' + a
  //
  //               Finally, both f and g are monotonic with their inputs.
  //               Therefore, if BWI <= BWE, then the transformed code will lead
  //               to a smaller area.
  //               Notice that this approach is conservative; in other
  //               words, there might be situations where BWI > BWE and this
  //               transformation could still save area. In other to become less
  //               conservative, we need to rely on an area model.
  //
  //
  // The code below checks all the above properties for the "select" node given
  // as input
  //
  // Check property 2
  Type *array_reference_type = array_reference->GetType();
  ArrayType *array_reference_type_as_array_type =
      array_reference_type->AsArrayOrDie();
  absl::Span<Node *const> select_cases = GetCases(select_to_optimize);
  Type *array_element_type = array_reference_type_as_array_type->element_type();
  int64_t array_element_bitwidth = array_element_type->GetFlatBitCount();
  for (Node *current_select_case_as_node : select_cases) {
    ArrayIndex *current_select_case =
        current_select_case_as_node->As<ArrayIndex>();
    absl::Span<Node *const> current_select_case_indices =
        current_select_case->indices();
    for (Node *current_select_case_index : current_select_case_indices) {
      Type *current_select_case_index_type =
          current_select_case_index->GetType();
      if (current_select_case_index_type->GetFlatBitCount() >
          array_element_bitwidth) {
        return false;
      }
    }
  }

  // Check properties 0 and 1
  for (Node *current_select_case_as_node : select_cases) {
    // Fetch the current array access (i.e., ArrayIndex)
    ArrayIndex *current_select_case =
        current_select_case_as_node->As<ArrayIndex>();

    // Check the users
    if (!HasSingleUse(current_select_case)) {
      return false;
    }

    // Check if all indices are literals
    if (AreAllLiteral(current_select_case->indices())) {
      return false;
    }
  }

  // The transformation is profitable
  return true;
}

bool ShouldLiftSelect(FunctionBase *func, Node *select_to_optimize,
                      const LiftableSelectOperandInfo &shared_between_inputs) {
  VLOG(3) << "  Checking the profitability guard";

  // Check if the transformation is profitable.
  //
  // Only "select" nodes with specific properties should be optimized.
  // Such properties depend on the inputs of the "select" node.
  //
  // The next code checks to see if the "select" node given as input should be
  // transformed.
  switch (shared_between_inputs.op) {
    case Op::kArrayIndex:
      return ProfitabilityGuardForArrayIndex(
          func, select_to_optimize, shared_between_inputs.shared_input);

    default:
      VLOG(3) << "    The current input of the select is not handled";
      return false;
  }
}

absl::StatusOr<TransformationResult> LiftSelectForArrayIndex(
    FunctionBase *func, Node *select_to_optimize, Node *array_reference) {
  TransformationResult result;

  // Step 0: add a new "select" for the indices
  VLOG(3) << "    Step 0: create a new \"select\" between the indices of the "
             "various arrayIndex nodes";
  std::optional<Node *> new_default_value = std::nullopt;
  std::optional<Node *> default_value = GetDefaultValue(select_to_optimize);
  if (default_value.has_value()) {
    ArrayIndex *default_case_array_index =
        default_value.value()->As<ArrayIndex>();
    absl::Span<Node *const> default_case_array_index_indices =
        default_case_array_index->indices();
    new_default_value = default_case_array_index_indices.at(0);
    VLOG(3) << "      Default case index " << new_default_value.value();
  }
  absl::Span<Node *const> select_cases = GetCases(select_to_optimize);
  std::vector<Node *> new_cases;
  for (uint32_t case_index = 0; case_index < select_cases.length();
       ++case_index) {
    Node *current_case_node = select_cases.at(case_index);
    ArrayIndex *current_case = current_case_node->As<ArrayIndex>();
    absl::Span<Node *const> current_case_indices = current_case->indices();
    VLOG(3) << "      Case " << case_index << ": " << *current_case;
    VLOG(3) << "        Index: " << current_case_indices.at(0);
    new_cases.push_back(current_case_indices.at(0));
  }
  Node *new_select;
  if (select_to_optimize->Is<PrioritySelect>()) {
    XLS_ASSIGN_OR_RETURN(
        new_select,
        func->MakeNode<PrioritySelect>(
            SourceInfo(), select_to_optimize->As<PrioritySelect>()->selector(),
            new_cases, *new_default_value));
  } else {
    XLS_ASSIGN_OR_RETURN(
        new_select,
        func->MakeNode<Select>(SourceInfo(),
                               select_to_optimize->As<Select>()->selector(),
                               new_cases, new_default_value));
  }

  // Step 1: add the new array access
  VLOG(3) << "    Step 1: add the new arrayIndex node";
  std::vector<Node *> new_indices;
  new_indices.push_back(new_select);
  Node *new_array_index = func->AddNode(std::make_unique<ArrayIndex>(
      SourceInfo(), array_reference, absl::Span<Node *const>(new_indices),
      "array_index_after_select", func));

  // Step 2: replace the uses of the original "select" node with the only
  //         exception of the new array access
  VLOG(3) << "    Step 2: replace the uses of the original \"select\"";
  XLS_RETURN_IF_ERROR(select_to_optimize->ReplaceUsesWith(new_array_index));
  VLOG(3) << "      New select     : " << select_to_optimize->ToString();
  VLOG(3) << "      New array index: " << new_array_index->ToString();

  // Step 3: remove the original "select" node as it just became dead. This is
  // done by adding such node to the list of nodes to delete at the end of the
  // main loop of this transformation.
  VLOG(3) << "    Step 3: mark the old \"select\" to be deleted";
  result.nodes_to_delete.insert(select_to_optimize);

  // Step 4: check if new "select" nodes become optimizable. These are users of
  // the new arrayIndex node
  VLOG(3) << "    Step 4: check if more \"select\" nodes should be considered";
  for (Node *user : new_array_index->users()) {
    if (user->OpIn({Op::kSel, Op::kPrioritySel})) {
      result.new_selects_to_consider.insert(user);
    }
  }
  result.was_code_modified = true;

  return result;
}

absl::StatusOr<TransformationResult> LiftSelect(
    FunctionBase *func, Node *select_to_optimize,
    const LiftableSelectOperandInfo &shared_between_inputs) {
  TransformationResult result;
  VLOG(3) << "  Apply the transformation";

  // The transformation depends on the specific inputs of the "select" node
  switch (shared_between_inputs.op) {
    case Op::kArrayIndex:
      return LiftSelectForArrayIndex(func, select_to_optimize,
                                     shared_between_inputs.shared_input);

    default:

      // If the execution arrives here, then the applicability guard has a bug.
      VLOG(3) << "    The current input of the select is not handled";
      return absl::InternalError(
          "The applicability guard incorrectly classified a \"select\" as "
          "applicable.");
  }
}

absl::StatusOr<TransformationResult> LiftSelect(FunctionBase *func,
                                                Node *select_to_optimize) {
  TransformationResult result;

  // Check if it is safe to apply the transformation
  XLS_ASSIGN_OR_RETURN(
      std::optional<LiftableSelectOperandInfo> applicability_guard_result,
      CanLiftSelect(func, select_to_optimize));
  if (!applicability_guard_result) {
    VLOG(3) << "  It is not safe to apply the transformation for this select";

    // The transformation is not applicable
    return result;
  }
  LiftableSelectOperandInfo shared_between_inputs = *applicability_guard_result;

  // It is safe to apply the transformation
  //
  // Check if it is profitable to apply the transformation
  if (!ShouldLiftSelect(func, select_to_optimize, shared_between_inputs)) {
    VLOG(3) << "  This transformation is not profitable for this select";

    // The transformation is not profitable
    return result;
  }

  // The transformation is safe and profitable.
  // It is now the time to apply it.
  VLOG(3) << "  This transformation is applicable and profitable for this "
             "select";
  XLS_ASSIGN_OR_RETURN(
      result, LiftSelect(func, select_to_optimize, shared_between_inputs));

  return result;
}

absl::StatusOr<TransformationResult> LiftSelects(
    FunctionBase *func,
    const absl::btree_set<Node *, Node::NodeIdLessThan> &selects_to_consider) {
  TransformationResult result;

  // Try to optimize all "select" nodes
  //
  // Step 0: try to shift the "select" nodes
  for (Node *select_node : selects_to_consider) {
    if (select_node->IsDead()) {
      continue;
    }
    VLOG(3) << "Select: " << select_node->ToString();

    // Try to optimize the current "select" node
    XLS_ASSIGN_OR_RETURN(TransformationResult current_transformation_result,
                         LiftSelect(func, select_node));

    // Accumulate the result of the transformation
    result.was_code_modified |= current_transformation_result.was_code_modified;
    result.new_selects_to_consider.insert(
        current_transformation_result.new_selects_to_consider.begin(),
        current_transformation_result.new_selects_to_consider.end());
    result.nodes_to_delete.insert(
        current_transformation_result.nodes_to_delete.begin(),
        current_transformation_result.nodes_to_delete.end());
  }

  // Step 1: delete the old selects
  for (Node *old_select : result.nodes_to_delete) {
    XLS_RETURN_IF_ERROR(func->RemoveNode(old_select));
  }

  return result;
}

}  // namespace

absl::StatusOr<bool> SelectLiftingPass::RunOnFunctionBaseInternal(
    FunctionBase *func, const OptimizationPassOptions &options,
    PassResults *results) const {
  absl::btree_set<Node *, Node::NodeIdLessThan> selects_to_consider;
  bool was_code_modified = false;

  // Collect the "select" nodes that might be optimizable
  VLOG(3) << "Optimizing the function at level " << options.opt_level;
  for (Node *node : func->nodes()) {
    // Only consider selects.
    if (!node->OpIn({Op::kSel, Op::kPrioritySel})) {
      continue;
    }

    // Do not consider selects that have no uses. These selects will get deleted
    // by the DeadCodeEliminationPass pass.
    if (node->IsDead()) {
      continue;
    }

    // Consider the current select.
    selects_to_consider.insert(node);
  }

  // Try to optimize all the "select" nodes of the function.
  while (!selects_to_consider.empty()) {
    VLOG(3) << "  New optimization iteration";

    // Optimize all "select" nodes.
    XLS_ASSIGN_OR_RETURN(TransformationResult current_result,
                         LiftSelects(func, selects_to_consider));

    // Check if we have modified the code.
    was_code_modified |= current_result.was_code_modified;
    if (!current_result.was_code_modified) {
      // The code did not get modified. So we can end the pass.
      VLOG(3) << "    No changes";
      break;
    }
    if (options.opt_level <= 1) {
      // The code got modified, but only higher level of optimizations (compared
      // to the current opt level) are allowed to repeat the transformation.
      break;
    }
    VLOG(3) << "    " << current_result.new_selects_to_consider.size()
            << " more select nodes need to be considered";

    // Consider the new "select" nodes that might have became optimizable.
    selects_to_consider = std::move(current_result.new_selects_to_consider);
  }

  return was_code_modified;
}

REGISTER_OPT_PASS(SelectLiftingPass);

}  // namespace xls
