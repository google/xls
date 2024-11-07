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
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
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

using CheckResult = std::optional<std::tuple<Node *, int64_t>>;

struct TransformationResult {
  bool was_code_modified;
  absl::btree_set<Node *, Node::NodeIdLessThan> new_selects_to_consider;
  absl::flat_hash_set<Node *> nodes_to_delete;

  TransformationResult() : was_code_modified{false} {}
};

CheckResult CheckSelectCase(Node *select_case, Node *array_ref,
                            std::optional<int64_t> index_bitwidth) {
  // Make sure the select case is an array index
  if (!select_case->Is<ArrayIndex>()) {
    VLOG(3) << "    This case is not an ArrayIndex";
    return std::nullopt;
  }
  ArrayIndex *array_index_case = select_case->As<ArrayIndex>();

  // Check the array base
  Node *current_array_ref = array_index_case->operand(0);
  if (array_ref != nullptr) {
    if (array_ref != current_array_ref) {
      VLOG(3) << "    This case accesses an array that is different than the "
                 "other cases";
      return std::nullopt;
    }
  }
  VLOG(3) << "    Array = " << current_array_ref->ToString();

  // Check the indices
  absl::Span<Node *const> current_case_indices = array_index_case->indices();
  if (current_case_indices.length() != 1) {
    return std::nullopt;
  }
  Node *index = current_case_indices.at(0);
  Type *index_type = index->GetType();
  int64_t current_index_bitwidth = index_type->GetFlatBitCount();
  if (index_bitwidth && (current_index_bitwidth != index_bitwidth.value())) {
    return std::nullopt;
  }

  return std::make_tuple(current_array_ref, current_index_bitwidth);
}

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

std::optional<Node *> ApplicabilityGuard(FunctionBase *func,
                                         Node *select_to_optimize) {
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
  // The code below checks these properties for the "select" node given as input
  //
  // Check Property 0 for the default case
  Node *array_ref = nullptr;
  std::optional<Node *> default_value = GetDefaultValue(select_to_optimize);
  std::optional<int64_t> index_bitwidth = std::nullopt;
  if (default_value.has_value()) {
    CheckResult check_result =
        CheckSelectCase(default_value.value(), nullptr, {});
    if (!check_result) {
      return std::nullopt;
    }
    auto [default_array_ref, default_index_bitwidth] = check_result.value();
    array_ref = default_array_ref;
    index_bitwidth = default_index_bitwidth;
  }

  // Check Property 0 for all cases excluding the default one
  absl::Span<Node *const> select_cases = GetCases(select_to_optimize);
  for (Node *select_case : select_cases) {
    CheckResult check_result =
        CheckSelectCase(select_case, array_ref, index_bitwidth);
    if (!check_result) {
      return std::nullopt;
    }
    auto [current_array_ref, current_index_bitwidth] = check_result.value();
    array_ref = current_array_ref;
    index_bitwidth = current_index_bitwidth;
  }

  // It is safe to apply the transformation
  return array_ref;
}

bool ProfitabilityGuard(FunctionBase *func, Node *select_to_optimize,
                        Node *array_reference) {
  // Check if the transformation is profitable
  //
  // Only "select" nodes with the following properties should be optimized
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

absl::StatusOr<TransformationResult> ApplyTransformation(
    FunctionBase *func, Node *select_to_optimize, Node *array_reference) {
  TransformationResult result;
  VLOG(3) << "  Apply the transformation";

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

absl::StatusOr<TransformationResult> LiftSelect(FunctionBase *func,
                                                Node *select_to_optimize) {
  TransformationResult result;

  // Check if it is safe to apply the transformation
  std::optional<Node *> applicability_guard_result =
      ApplicabilityGuard(func, select_to_optimize);
  if (!applicability_guard_result) {
    VLOG(3) << "  It is not safe to apply the transformation for this select";

    // The transformation is not applicable
    return result;
  }
  Node *array_reference = applicability_guard_result.value();

  // It is safe to apply the transformation
  //
  // Check if it is profitable to apply the transformation
  if (!ProfitabilityGuard(func, select_to_optimize, array_reference)) {
    VLOG(3) << "  This transformation is not profitable for this select";

    // The transformation is not profitable
    return result;
  }

  // The transformation is safe and profitable.
  // It is now the time to apply it.
  VLOG(3) << "  This transformation is applicable and profitable for this "
             "select";
  XLS_ASSIGN_OR_RETURN(
      result, ApplyTransformation(func, select_to_optimize, array_reference));

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
  VLOG(3) << "Optimizing the function at level " << this->opt_level_;
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
    if (opt_level_ <= 1) {
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

REGISTER_OPT_PASS(SelectLiftingPass, pass_config::kOptLevel);

}  // namespace xls
