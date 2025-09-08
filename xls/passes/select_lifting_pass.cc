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
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/critical_path_delay_analysis.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

namespace {

struct TransformationResult {
  bool was_code_modified;
  absl::btree_set<Node *, Node::NodeIdLessThan> new_selects_to_consider;
  absl::flat_hash_set<Node *> nodes_to_delete;

  TransformationResult() : was_code_modified{false} {}
};

// Information about a liftable binary operation from a Select node.
struct LiftedOpInfo {
  Op lifted_op;        // The binary operation to be lifted.
  Node* shared_node;   // The operand shared across all cases.
  bool shared_is_lhs;  // True if the shared node is the LHS operand.
  std::vector<Node*> other_operands;  // Operands for non-identity cases.
  std::optional<Node*>
      default_other_operand;  // Operand for non-identity default.
  // Indices of cases that were identity (i.e., equal to shared_node).
  absl::flat_hash_set<int64_t> identity_case_indices;
  bool default_is_identity;  // True if the default case is an identity.
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

// Checks if all cases are ArrayIndexes on the same array with compatible
// indices. Returns the shared array node if liftable.
std::optional<Node*> CheckArrayIndexLiftable(
    absl::Span<Node* const> cases, std::optional<Node*> default_case) {
  if (cases.empty()) {
    return std::nullopt;
  }
  // Check if all cases and default are kArrayIndex
  if (absl::c_any_of(cases, [](Node* case_node) {
        return case_node->op() != Op::kArrayIndex;
      })) {
    return std::nullopt;
  }
  if (default_case.has_value() && (*default_case)->op() != Op::kArrayIndex) {
    return std::nullopt;
  }

  return ApplicabilityGuardForArrayIndex(cases, default_case);
}

// Creates a Literal node representing the right-identity for the given
// operation. e.g., 0 for Add/Sub/Xor, -1 for And, 1 for Mul.
// Note that all operations in kLiftableBinaryOps have a constant
// right-identity, which simplifies lifting when the shared node is on the LHS.
absl::StatusOr<Node*> GetIdentityLiteral(Op op, Type* type,
                                         FunctionBase* func) {
  if (!type->IsBits()) {
    return absl::InvalidArgumentError(
        "Identity literal requested for non-Bits type.");
  }
  int64_t bit_count = type->GetFlatBitCount();
  Value identity_value;
  switch (op) {
    case Op::kAdd:
    case Op::kSub:
    case Op::kOr:
    case Op::kXor:
    case Op::kShll:
    case Op::kShrl:
    case Op::kShra:
      identity_value = Value(UBits(0, bit_count));
      break;
    case Op::kAnd:
      identity_value = Value(Bits::AllOnes(bit_count));
      break;
    case Op::kUMul:
    case Op::kSMul:
      identity_value = Value(UBits(1, bit_count));
      break;
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported operation for identity literal: ", OpToString(op)));
  }
  return func->MakeNode<Literal>(SourceInfo(), identity_value);
}

// Given a set of cases, a potential shared node, and a test operation, this
// function checks if the operation can be lifted. It returns a LiftedOpInfo
// struct if the operation is liftable, otherwise std::nullopt.
std::optional<LiftedOpInfo> GetLiftableOperationInfoForOp(
    absl::Span<Node* const> cases, std::optional<Node*> default_case,
    Node* potential_shared_node, Op test_op) {
  std::optional<bool> shared_is_lhs;
  std::vector<Node*> other_operands;
  absl::flat_hash_set<int64_t> identity_case_indices;
  int64_t other_operand_bitwidth = -1;
  bool default_is_identity = false;
  std::optional<Node*> default_other_operand;
  bool op_is_commutative = OpIsCommutative(test_op);

  // Helper lambda to check a single node (case or default) for liftability.
  // Returns the "other" operand and a boolean indicating if the shared node
  // was the LHS.
  struct LiftableCase {
    Node* other_operand;
    bool shared_is_lhs;
  };
  auto check_liftable = [&](Node* node) -> std::optional<LiftableCase> {
    if (node->op() != test_op || node->operand_count() != 2) {
      return std::nullopt;
    }

    Node* op0 = node->operand(0);
    Node* op1 = node->operand(1);
    Node* current_other_operand = nullptr;
    bool current_shared_is_lhs = false;

    if (op0 == potential_shared_node) {
      current_shared_is_lhs = true;
      current_other_operand = op1;
    } else if (op1 == potential_shared_node) {
      current_shared_is_lhs = false;
      current_other_operand = op0;
    } else {
      return std::nullopt;  // potential_shared_node not in this case
    }

    if (!shared_is_lhs.has_value()) {
      shared_is_lhs = current_shared_is_lhs;
    } else if (!op_is_commutative && *shared_is_lhs != current_shared_is_lhs) {
      return std::nullopt;  // Inconsistent shared side for non-commutative op
    }

    Type* other_type = current_other_operand->GetType();
    if (!other_type->IsBits()) {
      return std::nullopt;  // Other operand must be Bits type.
    }
    int64_t current_bw = other_type->GetFlatBitCount();
    if (other_operand_bitwidth == -1) {
      other_operand_bitwidth = current_bw;
    } else if (other_operand_bitwidth != current_bw) {
      return std::nullopt;  // Inconsistent bitwidth for other operands
    }
    return LiftableCase{current_other_operand, current_shared_is_lhs};
  };

  for (int64_t i = 0; i < cases.size(); ++i) {
    Node* case_node = cases[i];
    if (case_node == potential_shared_node) {
      identity_case_indices.insert(i);
      continue;
    }

    std::optional<LiftableCase> liftable_case = check_liftable(case_node);
    if (!liftable_case.has_value()) {
      return std::nullopt;
    }
    other_operands.push_back(liftable_case->other_operand);
  }

  if (default_case.has_value()) {
    Node* def_node = *default_case;
    if (def_node == potential_shared_node) {
      default_is_identity = true;
    } else {
      std::optional<LiftableCase> liftable_case = check_liftable(def_node);
      if (!liftable_case.has_value()) {
        return std::nullopt;
      }
      default_other_operand = liftable_case->other_operand;
    }
  }

  // If no non-identity cases were found, there's nothing to lift.
  if (!shared_is_lhs.has_value()) {
    return std::nullopt;
  }

  // If the shared node is on the RHS for a non-commutative op, we can only
  // lift if identity cases are present if the op has a left-identity L
  // such that `L op shared = shared`. The non-commutative operations
  // currently supported for lifting (subtraction and shifts) do not possess
  // a left-identity, so we disallow lifting in this configuration.
  if (!op_is_commutative && !*shared_is_lhs &&
      (!identity_case_indices.empty() || default_is_identity)) {
    VLOG(3) << "Cannot lift: shared node is RHS of non-commutative op "
            << OpToString(test_op)
            << ", and identity cases are present. Lifting in this "
               "configuration requires a left identity, which "
            << OpToString(test_op) << " lacks.";
    return std::nullopt;
  }

  // If the op is commutative, we can always act as if the shared node was on
  // the LHS.
  if (op_is_commutative) {
    shared_is_lhs = true;
  }

  return LiftedOpInfo{
      .lifted_op = test_op,
      .shared_node = potential_shared_node,
      .shared_is_lhs = *shared_is_lhs,
      .other_operands = std::move(other_operands),
      .default_other_operand = default_other_operand,
      .identity_case_indices = std::move(identity_case_indices),
      .default_is_identity = default_is_identity,
  };
}

// Attempts to find a liftable operation in the select cases.
std::optional<LiftedOpInfo> GetLiftableOperationInfo(
    absl::Span<Node* const> cases, std::optional<Node*> default_case,
    Node* potential_shared_node) {
  constexpr Op kLiftableBinaryOps[] = {
      Op::kAdd,  Op::kSub,  Op::kAnd,  Op::kOr,   Op::kXor,
      Op::kUMul, Op::kSMul, Op::kShll, Op::kShrl, Op::kShra};

  for (Op test_op : kLiftableBinaryOps) {
    std::optional<LiftedOpInfo> info = GetLiftableOperationInfoForOp(
        cases, default_case, potential_shared_node, test_op);
    if (info.has_value()) {
      return info;
    }
  }
  return std::nullopt;
}

absl::StatusOr<std::optional<LiftedOpInfo>> CanLiftSelect(
    FunctionBase* func, Node* select_to_optimize) {
  VLOG(3) << "  Checking the applicability guard";

  // Only "select" nodes with specific properties can be optimized by this
  // transformation.
  absl::Span<Node* const> cases = GetCases(select_to_optimize);
  std::optional<Node*> default_case = GetDefaultValue(select_to_optimize);

  if (cases.empty()) {
    VLOG(3) << "    Select has no cases, not liftable.";
    return std::nullopt;
  }

  // Identify potential shared nodes.
  absl::btree_set<Node*, Node::NodeIdLessThan> potential_shared_nodes;
  // Gather other potential shared nodes from other cases and default.
  auto add_potential_shared = [&](Node* node) {
    if (node->operand_count() == 2 && node->op() != Op::kArrayIndex) {
      potential_shared_nodes.insert(node->operand(0));
      potential_shared_nodes.insert(node->operand(1));
    } else {
      potential_shared_nodes.insert(node);
    }
  };
  for (Node* case_node : cases) {
    add_potential_shared(case_node);
  }
  if (default_case.has_value()) {
    add_potential_shared(*default_case);
  }

  for (Node* potential_shared : potential_shared_nodes) {
    std::optional<LiftedOpInfo> info =
        GetLiftableOperationInfo(cases, default_case, potential_shared);
    if (info.has_value()) {
      return info;
    }
  }

  // Check for ArrayIndex lifting opportunity.
  std::optional<Node*> shared_array =
      CheckArrayIndexLiftable(cases, default_case);
  if (shared_array.has_value()) {
    // Build LiftedOpInfo for ArrayIndex
    std::vector<Node*> index_operands;
    for (Node* case_node : cases) {
      index_operands.push_back(case_node->As<ArrayIndex>()->indices()[0]);
    }
    std::optional<Node*> default_index_operand;
    if (default_case.has_value()) {
      default_index_operand = (*default_case)->As<ArrayIndex>()->indices()[0];
    }

    return LiftedOpInfo{
        .lifted_op = Op::kArrayIndex,
        .shared_node = *shared_array,
        .shared_is_lhs = true,  // Not really applicable for ArrayIndex
        .other_operands = std::move(index_operands),
        .default_other_operand = default_index_operand,
        .identity_case_indices = {},
        .default_is_identity = false,
    };
  }

  return std::nullopt;
}

absl::StatusOr<Node*> MakeSelectNode(FunctionBase* func, Node* old_select,
                                     const std::vector<Node*>& new_cases,
                                     std::optional<Node*> new_default) {
  if (old_select->Is<PrioritySelect>()) {
    return func->MakeNode<PrioritySelect>(
        SourceInfo(), old_select->As<PrioritySelect>()->selector(), new_cases,
        *new_default);
  } else {
    return func->MakeNode<Select>(SourceInfo(),
                                  old_select->As<Select>()->selector(),
                                  new_cases, new_default);
  }
}

absl::StatusOr<bool> CheckLatencyIncrease(
    FunctionBase* func, Node* select_to_optimize, const LiftedOpInfo& info,
    const OptimizationPassOptions& options, OptimizationContext& context) {
  CriticalPathDelayAnalysis* analysis =
      context.SharedNodeData<CriticalPathDelayAnalysis>(func, options);
  if (analysis == nullptr) {
    return absl::InternalError(absl::StrCat(
        "Failed to get CriticalPathDelayAnalysis for delay model: ",
        *options.delay_model));
  }

  // Check the (unscheduled) critical path through the select we're optimizing
  int64_t t_before = *analysis->GetInfo(select_to_optimize);

  // To make it easy to estimate the critical path after lifting the select, we
  // add nodes to represent the post-optimization result.
  Node* tmp_new_select = nullptr;
  Node* tmp_lifted_op = nullptr;
  absl::flat_hash_set<Node*> tmp_identity_literals;
  int64_t original_next_node_id = func->package()->next_node_id();

  absl::Cleanup cleanup = [&] {
    if (tmp_lifted_op != nullptr) {
      CHECK_OK(func->RemoveNode(tmp_lifted_op));
    }
    if (tmp_new_select != nullptr) {
      CHECK_OK(func->RemoveNode(tmp_new_select));
    }
    for (Node* literal : tmp_identity_literals) {
      CHECK_OK(func->RemoveNode(literal));
    }
    // Restore the package's next node ID to its original value.
    func->package()->set_next_node_id(original_next_node_id);
  };

  if (info.lifted_op == Op::kArrayIndex) {
    XLS_ASSIGN_OR_RETURN(
        tmp_new_select,
        MakeSelectNode(func, select_to_optimize, info.other_operands,
                       info.default_other_operand));
    XLS_ASSIGN_OR_RETURN(
        tmp_lifted_op,
        func->MakeNode<ArrayIndex>(SourceInfo(), info.shared_node,
                                   absl::Span<Node* const>{tmp_new_select}));
  } else {
    Type* other_operand_type = nullptr;
    if (!info.other_operands.empty()) {
      other_operand_type = info.other_operands[0]->GetType();
    } else {
      other_operand_type = (*info.default_other_operand)->GetType();
    }

    std::vector<Node*> tmp_new_cases;
    std::optional<Node*> tmp_new_default;
    absl::Span<Node* const> original_cases = GetCases(select_to_optimize);
    int64_t other_operand_idx = 0;
    for (int64_t i = 0; i < original_cases.size(); ++i) {
      if (info.identity_case_indices.contains(i)) {
        XLS_ASSIGN_OR_RETURN(
            Node * identity_literal,
            GetIdentityLiteral(info.lifted_op, other_operand_type, func));
        tmp_new_cases.push_back(identity_literal);
        tmp_identity_literals.insert(identity_literal);
      } else {
        tmp_new_cases.push_back(info.other_operands[other_operand_idx++]);
      }
    }
    std::optional<Node*> original_default = GetDefaultValue(select_to_optimize);
    if (original_default.has_value()) {
      if (info.default_is_identity) {
        XLS_ASSIGN_OR_RETURN(
            Node * identity_literal,
            GetIdentityLiteral(info.lifted_op, other_operand_type, func));
        tmp_new_default = identity_literal;
        tmp_identity_literals.insert(identity_literal);
      } else {
        tmp_new_default = *info.default_other_operand;
      }
    }
    XLS_ASSIGN_OR_RETURN(tmp_new_select,
                         MakeSelectNode(func, select_to_optimize, tmp_new_cases,
                                        tmp_new_default));
    Node* lhs = info.shared_is_lhs ? info.shared_node : tmp_new_select;
    Node* rhs = info.shared_is_lhs ? tmp_new_select : info.shared_node;
    switch (info.lifted_op) {
      case Op::kAdd:
      case Op::kSub:
      case Op::kShll:
      case Op::kShrl:
      case Op::kShra: {
        XLS_ASSIGN_OR_RETURN(
            tmp_lifted_op,
            func->MakeNode<BinOp>(SourceInfo(), lhs, rhs, info.lifted_op));
        break;
      }
      case Op::kAnd:
      case Op::kOr:
      case Op::kXor: {
        XLS_ASSIGN_OR_RETURN(
            tmp_lifted_op,
            func->MakeNode<NaryOp>(SourceInfo(), std::vector<Node*>{lhs, rhs},
                                   info.lifted_op));
        break;
      }
      case Op::kUMul:
      case Op::kSMul: {
        XLS_ASSIGN_OR_RETURN(
            tmp_lifted_op, func->MakeNode<ArithOp>(
                               SourceInfo(), lhs, rhs,
                               select_to_optimize->GetType()->GetFlatBitCount(),
                               info.lifted_op));
        break;
      }
      default:
        return absl::InternalError(
            absl::StrCat("Unsupported binary operation in latency check: ",
                         OpToString(info.lifted_op)));
    }
  }

  int64_t t_after = *analysis->GetInfo(tmp_lifted_op);
  return t_after > t_before;
}

absl::StatusOr<bool> ProfitabilityGuardForArrayIndex(FunctionBase* func,
                                                     Node* select_to_optimize,
                                                     Node* array_reference) {
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

absl::StatusOr<bool> ProfitabilityGuardForBinaryOperation(
    FunctionBase* func, Node* select_to_optimize, const LiftedOpInfo& info,
    const OptimizationPassOptions& options, OptimizationContext& context) {
  // Heuristically: If the selector depends on `shared_node`, lifting will
  // likely serialize more operations & worsen the critical path.
  Node* selector = select_to_optimize->Is<Select>()
                       ? select_to_optimize->As<Select>()->selector()
                       : select_to_optimize->As<PrioritySelect>()->selector();
  if (IsAncestorOf(info.shared_node, selector)) {
    VLOG(3) << "    Selector depends on shared node, avoiding lift due to "
               "potential latency increase.";
    return false;
  }

  // Calculate Cost Before:
  // Sum of bitwidths of the original select and any single-use non-identity
  // case nodes.
  int64_t initial_bitwidths = select_to_optimize->GetType()->GetFlatBitCount();
  absl::Span<Node* const> cases = GetCases(select_to_optimize);
  for (int64_t i = 0; i < cases.size(); ++i) {
    if (!info.identity_case_indices.contains(i)) {
      Node* case_node = cases[i];
      if (HasSingleUse(case_node)) {
        initial_bitwidths += case_node->GetType()->GetFlatBitCount();
      }
    }
  }
  std::optional<Node*> default_case = GetDefaultValue(select_to_optimize);
  if (default_case.has_value() && !info.default_is_identity) {
    if (HasSingleUse(*default_case)) {
      initial_bitwidths += (*default_case)->GetType()->GetFlatBitCount();
    }
  }

  // Calculate Cost After:
  // Bitwidth of the new select + bitwidth of the lifted binary operation
  // output.
  Type* other_operand_type = nullptr;
  if (!info.other_operands.empty()) {
    other_operand_type = info.other_operands[0]->GetType();
  } else if (info.default_other_operand.has_value()) {
    other_operand_type = (*info.default_other_operand)->GetType();
  } else {
    // This should not happen if CanLiftSelect passed.
    return false;
  }
  int64_t new_select_width = other_operand_type->GetFlatBitCount();

  // The output width of the lifted op is the same as the original select.
  int64_t lifted_op_output_width =
      select_to_optimize->GetType()->GetFlatBitCount();

  int64_t remaining_bitwidths = new_select_width + lifted_op_output_width;

  VLOG(3) << "    Profitability: Initial bitwidths: " << initial_bitwidths
          << ", Remaining bitwidths: " << remaining_bitwidths;
  return initial_bitwidths >= remaining_bitwidths;
}

absl::StatusOr<bool> ShouldLiftSelect(FunctionBase* func,
                                      Node* select_to_optimize,
                                      const LiftedOpInfo& info,
                                      const OptimizationPassOptions& options,
                                      OptimizationContext& context) {
  VLOG(3) << "  Checking the profitability guard";

  if (options.delay_model.has_value()) {
    // If delay model is provided, check for latency increase.
    XLS_ASSIGN_OR_RETURN(
        bool latency_increases,
        CheckLatencyIncrease(func, select_to_optimize, info, options, context));
    if (latency_increases) {
      VLOG(3) << "    Not lifting " << OpToString(info.lifted_op)
              << " because latency increases.";
      return false;
    }
  } else if ((!info.identity_case_indices.empty() ||
              info.default_is_identity) &&
             (info.lifted_op == Op::kUMul || info.lifted_op == Op::kSMul)) {
    // If no delay model, apply fallback heuristic for binary ops.
    VLOG(3) << "    Not lifting high-latency op " << OpToString(info.lifted_op)
            << " with identity cases because no delay model is provided.";
    return false;
  }

  // If we pass latency checks or they don't apply, proceed to
  // op-specific profitability guards for bitwidth/cost checks.
  if (info.lifted_op == Op::kArrayIndex) {
    return ProfitabilityGuardForArrayIndex(func, select_to_optimize,
                                           info.shared_node);
  }
  return ProfitabilityGuardForBinaryOperation(func, select_to_optimize, info,
                                              options, context);
}

absl::StatusOr<TransformationResult> LiftSelectForArrayIndex(
    FunctionBase* func, Node* select_to_optimize, const LiftedOpInfo& info) {
  TransformationResult result;
  Node* array_reference = info.shared_node;

  // Step 0: add a new "select" for the indices
  VLOG(3) << "    Step 0: create a new \"select\" between the indices of the "
             "various arrayIndex nodes";
  std::optional<Node*> new_default_value = info.default_other_operand;
  const std::vector<Node*>& new_cases = info.other_operands;

  Node *new_select;
  XLS_ASSIGN_OR_RETURN(
      new_select,
      MakeSelectNode(func, select_to_optimize, new_cases, new_default_value));

  // Step 1: add the new array access
  VLOG(3) << "    Step 1: add the new arrayIndex node";
  std::vector<Node *> new_indices;
  new_indices.push_back(new_select);
  XLS_ASSIGN_OR_RETURN(
      Node * new_array_index,
      func->MakeNode<ArrayIndex>(SourceInfo(), array_reference,
                                 absl::Span<Node* const>(new_indices)));

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

absl::StatusOr<TransformationResult> LiftSelectForBinaryOperation(
    FunctionBase* func, Node* select_to_optimize, const LiftedOpInfo& info) {
  TransformationResult result;
  absl::Span<Node* const> original_cases = GetCases(select_to_optimize);

  VLOG(3) << "    Step 1: Build new cases for the inner select";
  std::vector<Node*> new_cases;
  new_cases.reserve(original_cases.size());
  int64_t other_operand_idx = 0;

  Type* other_operand_type = nullptr;
  if (!info.other_operands.empty()) {
    other_operand_type = info.other_operands[0]->GetType();
  } else if (info.default_other_operand.has_value()) {
    other_operand_type = (*info.default_other_operand)->GetType();
  }
  if (other_operand_type == nullptr) {
    // This can only happen if all cases and default are identity.
    // But CanLiftSelect should have returned nullopt in this case.
    return absl::InternalError(
        "Cannot determine other operand type in LiftSelectForBinaryOperation.");
  }

  for (int64_t i = 0; i < original_cases.size(); ++i) {
    if (info.identity_case_indices.contains(i)) {
      XLS_ASSIGN_OR_RETURN(
          Node * identity_literal,
          GetIdentityLiteral(info.lifted_op, other_operand_type, func));
      new_cases.push_back(identity_literal);
    } else {
      new_cases.push_back(info.other_operands[other_operand_idx++]);
    }
  }

  VLOG(3) << "    Step 2: Build new default for the inner select";
  std::optional<Node*> new_default;
  std::optional<Node*> original_default = GetDefaultValue(select_to_optimize);
  if (original_default.has_value()) {
    if (info.default_is_identity) {
      XLS_ASSIGN_OR_RETURN(
          new_default,
          GetIdentityLiteral(info.lifted_op, other_operand_type, func));
    } else {
      new_default = *info.default_other_operand;
    }
  } else {
    new_default = std::nullopt;
  }

  XLS_ASSIGN_OR_RETURN(
      Node * new_select,
      MakeSelectNode(func, select_to_optimize, new_cases, new_default));

  VLOG(3) << "    Step 3: Create the lifted binary operation";
  Node* lhs = info.shared_is_lhs ? info.shared_node : new_select;
  Node* rhs = info.shared_is_lhs ? new_select : info.shared_node;

  Node* new_binop;
  switch (info.lifted_op) {
    case Op::kAdd:
    case Op::kSub:
    case Op::kShll:
    case Op::kShrl:
    case Op::kShra: {
      XLS_ASSIGN_OR_RETURN(
          new_binop,
          func->MakeNode<BinOp>(SourceInfo(), lhs, rhs, info.lifted_op));
      break;
    }
    case Op::kAnd:
    case Op::kOr:
    case Op::kXor: {
      XLS_ASSIGN_OR_RETURN(
          new_binop,
          func->MakeNode<NaryOp>(SourceInfo(), std::vector<Node*>{lhs, rhs},
                                 info.lifted_op));
      break;
    }
    case Op::kUMul:
    case Op::kSMul: {
      XLS_ASSIGN_OR_RETURN(new_binop,
                           func->MakeNode<ArithOp>(
                               SourceInfo(), lhs, rhs,
                               select_to_optimize->GetType()->GetFlatBitCount(),
                               info.lifted_op));
    } break;
    default:
      return absl::InternalError(absl::StrCat(
          "Unsupported binary operation in LiftSelectForBinaryOperation: ",
          OpToString(info.lifted_op)));
  }

  VLOG(3) << "    Step 4: Replace uses of the original \"select\"";
  XLS_RETURN_IF_ERROR(select_to_optimize->ReplaceUsesWith(new_binop));
  VLOG(3) << "      New select: " << new_select->ToString();
  VLOG(3) << "      New binop : " << new_binop->ToString();

  VLOG(3) << "    Step 5: mark the old \"select\" to be deleted";
  result.nodes_to_delete.insert(select_to_optimize);

  VLOG(3) << "    Step 6: check if more \"select\" nodes should be considered";
  for (Node *user : new_binop->users()) {
    if (user->OpIn({Op::kSel, Op::kPrioritySel})) {
      result.new_selects_to_consider.insert(user);
    }
  }

  result.was_code_modified = true;

  return result;
}

absl::StatusOr<TransformationResult> LiftSelect(FunctionBase* func,
                                                Node* select_to_optimize,
                                                const LiftedOpInfo& info) {
  TransformationResult result;
  VLOG(3) << "  Apply the transformation";

  // The transformation depends on the specific inputs of the "select" node
  switch (info.lifted_op) {
    case Op::kArrayIndex:
      return LiftSelectForArrayIndex(func, select_to_optimize, info);

    // The applicability guard checked the operation has exactly 2 operands.
    case Op::kAnd:
    case Op::kOr:
    case Op::kNand:
    case Op::kNor:
    case Op::kXor:
    case Op::kAdd:
    case Op::kSub:
    case Op::kUMul:
    case Op::kSMul:
    case Op::kUDiv:
    case Op::kSDiv:
    case Op::kShll:
    case Op::kShrl:
    case Op::kShra:
      return LiftSelectForBinaryOperation(func, select_to_optimize, info);

    default:

      // If the execution arrives here, then the applicability guard has a bug.
      VLOG(3) << "    The current input of the select is not handled";
      return absl::InternalError(
          "The applicability guard incorrectly classified a \"select\" as "
          "applicable.");
  }
}

absl::StatusOr<TransformationResult> LiftSelect(
    FunctionBase* func, Node* select_to_optimize,
    const OptimizationPassOptions& options, OptimizationContext& context) {
  TransformationResult result;

  // Check if it is safe to apply the transformation
  XLS_ASSIGN_OR_RETURN(std::optional<LiftedOpInfo> applicability_guard_result,
                       CanLiftSelect(func, select_to_optimize));
  if (!applicability_guard_result) {
    VLOG(3) << "  It is not safe to apply the transformation for this select";

    // The transformation is not applicable
    return result;
  }
  LiftedOpInfo info = *applicability_guard_result;

  // It is safe to apply the transformation
  //
  // Check if it is profitable to apply the transformation
  XLS_ASSIGN_OR_RETURN(
      bool should_lift,
      ShouldLiftSelect(func, select_to_optimize, info, options, context));
  if (!should_lift) {
    VLOG(3) << "  This transformation is not profitable for this select";

    // The transformation is not profitable
    return result;
  }

  // The transformation is safe and profitable.
  // It is now the time to apply it.
  VLOG(3) << "  This transformation is applicable and profitable for this "
             "select";
  XLS_ASSIGN_OR_RETURN(result, LiftSelect(func, select_to_optimize, info));

  return result;
}

absl::StatusOr<TransformationResult> LiftSelects(
    FunctionBase* func,
    const absl::btree_set<Node*, Node::NodeIdLessThan>& selects_to_consider,
    const OptimizationPassOptions& options, OptimizationContext& context) {
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
                         LiftSelect(func, select_node, options, context));

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
    result.new_selects_to_consider.erase(old_select);
    XLS_RETURN_IF_ERROR(func->RemoveNode(old_select));
  }

  return result;
}

}  // namespace

absl::StatusOr<bool> SelectLiftingPass::RunOnFunctionBaseInternal(
    FunctionBase *func, const OptimizationPassOptions &options,
    PassResults *results, OptimizationContext &context) const {
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
    XLS_ASSIGN_OR_RETURN(
        TransformationResult current_result,
        LiftSelects(func, selects_to_consider, options, context));

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


}  // namespace xls
