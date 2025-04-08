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

#include "xls/passes/resource_sharing_pass.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "cppitertools/enumerate.hpp"
#include "cppitertools/zip.hpp"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"

namespace xls {

// This class represents a single folding action from a node into another one.
//
// Folding is the code transformation that re-uses a given hardware resource
// (e.g., multiplier) for multiple IR operations.
//
// Consider for example the following IR:
//   a2 = umul a1, a0
//   b2 = umul b1, b0
//   c  = sel(s, a2, b2)
//
// The code above can be rewritten into
//   lhs = sel(s, a1, b1)
//   rhs = sel(s, a0, b0)
//   c   = umul lhs, rhs
//
// The last version of the code is obtained by folding the two multiplications
// together.
class FoldingAction {
 public:
  FoldingAction(Node *from, Node *to, Node *select, uint32_t from_case_number,
                uint32_t to_case_number);

  Node *GetFrom(void) const;

  Node *GetTo(void) const;

  Node *GetSelect(void) const;

  Node *GetSelector(void) const;

  uint32_t GetFromCaseNumber(void) const;

  uint32_t GetToCaseNumber(void) const;

 private:
  Node *from_;
  Node *to_;
  Node *select_;
  uint32_t from_case_number_;
  uint32_t to_case_number_;
};

absl::Span<Node *const> GetCases(Node *select) {
  if (select->Is<PrioritySelect>()) {
    return select->As<PrioritySelect>()->cases();
  }
  if (select->Is<OneHotSelect>()) {
    return select->As<OneHotSelect>()->cases();
  }
  CHECK(select->Is<Select>());
  return select->As<Select>()->cases();
}

Node *GetSelector(Node *select) {
  if (select->Is<PrioritySelect>()) {
    return select->As<PrioritySelect>()->selector();
  }
  if (select->Is<OneHotSelect>()) {
    return select->As<OneHotSelect>()->selector();
  }
  CHECK(select->Is<Select>());
  return select->As<Select>()->selector();
}

std::optional<uint32_t> GetSelectCaseNumberOfNode(
    const absl::flat_hash_map<Node *, absl::flat_hash_set<Node *>>
        &reachability_result,
    Node *node, Node *select_case, uint32_t select_case_number) {
  // Check if @node reaches the select case given as input
  if (!reachability_result.at(select_case).contains(node)) {
    return {};
  }

  // @node reaches the select case given as input.
  // Update the case number
  return select_case_number;
}

// This function check if @node_to_map can be folded into @folding_destination.
// If it can, then it returns an object including the information needed to
// perform the folding operation.
//
// The @select parameter is assumed to be what makes the two nodes mutually
// exclusive. The current analysis succeeds at declaring the two nodes can be
// folded together if the condition to select which of the two nodes need to
// compute (and therefore which inputs to forward to the single resulting node
// can be determined) can be determined by @select.
std::optional<FoldingAction *> CanMapInto(
    Node *node_to_map, Node *folding_destination, Node *select,
    const absl::flat_hash_map<Node *, absl::flat_hash_set<Node *>>
        &reachability_result) {
  // We currently only handled PrioritySelect
  if (!select->Is<PrioritySelect>()) {
    return {};
  }

  // Check @node_to_map and @folding_destination reach only one case of the
  // select
  bool node_to_map_found = false;
  uint32_t node_to_map_case_number = 0;
  bool folding_destination_found = false;
  uint32_t folding_destination_case_number = 0;
  for (const auto &[case_number, current_case] :
       iter::enumerate(GetCases(select))) {
    // Check @node_to_map
    std::optional<uint32_t> node_to_map_case_number_opt =
        GetSelectCaseNumberOfNode(reachability_result, node_to_map,
                                  current_case, case_number);
    if (node_to_map_case_number_opt.has_value()) {
      if (node_to_map_found) {
        return {};
      }
      node_to_map_found = true;
      node_to_map_case_number = *node_to_map_case_number_opt;
    }

    // Check @folding_destination
    std::optional<uint32_t> folding_destination_case_number_opt =
        GetSelectCaseNumberOfNode(reachability_result, folding_destination,
                                  current_case, case_number);
    if (folding_destination_case_number_opt.has_value()) {
      if (folding_destination_found) {
        return {};
      }
      folding_destination_found = true;
      folding_destination_case_number = *folding_destination_case_number_opt;
    }
  }
  if (folding_destination_case_number == node_to_map_case_number) {
    return {};
  }

  // @node_to_map can fold into @folding_destination
  FoldingAction *f = new FoldingAction(node_to_map, folding_destination, select,
                                       node_to_map_case_number,
                                       folding_destination_case_number);

  return f;
}

// Check if the two nodes given as input are compatible for folding.
bool AreCompatible(Node *n0, Node *n1) {
  ArithOp *from_mul = n0->As<ArithOp>();
  ArithOp *to_mul = n1->As<ArithOp>();

  // We currently handle only folding between multiplications with the same
  // bitwidth
  //
  // - Check if the result has the same bitwidth
  if (from_mul->BitCountOrDie() != to_mul->BitCountOrDie()) {
    return false;
  }

  // - Check if the operands have the same bitwidth
  CHECK_EQ(from_mul->operand_count(), to_mul->operand_count());
  for (auto [operand_from_mul, operand_to_mul] :
       iter::zip(from_mul->operands(), to_mul->operands())) {
    if (operand_from_mul->BitCountOrDie() != operand_to_mul->BitCountOrDie()) {
      return false;
    }
  }

  return true;
}

// Check if we are currently capable to potentially handle the node given as
// input for folding.
bool CanTarget(Node *n) {
  // We currently handle only multiplications
  if (!n->Is<ArithOp>()) {
    return false;
  }
  ArithOp *binop = n->As<ArithOp>();
  return binop->OpIn({Op::kUMul, Op::kSMul});
}

// Compute the reachability analysis.
// This analysis associates a node n with the set of nodes that belong to the
// def-use chain reaching n (including n)
absl::flat_hash_map<Node *, absl::flat_hash_set<Node *>>
ComputeReachabilityAnalysis(FunctionBase *f, OptimizationContext &context) {
  absl::flat_hash_map<Node *, absl::flat_hash_set<Node *>> reachability_result;
  for (Node *node : context.TopoSort(f)) {
    // A node always reaches itself
    reachability_result[node].insert(node);

    // Iterate over the users of the current node and propagate the fact that
    // @node reaches them.
    for (Node *const user : node->users()) {
      // Add everything that reached @node to all its users
      reachability_result[user].insert(reachability_result[node].begin(),
                                       reachability_result[node].end());

      // Add @node to it
      reachability_result[user].insert(node);
    }
  }

  return reachability_result;
}

// This function computes the set of mutual exclusive pairs of instructions.
// Each pair of instruction is associated with the select (of any kind) that
// made them mutually exclusive.
// We later use this association to extract the conditions to decide which
// inputs to use at the folded remaining node.
//
// This analysis is conservative and as such it might generate false negatives.
// In other words, some mutually-exclusive pairs of instructions might not be
// detected by this analysis.
// Hence, this analysis can be improved in the future.
absl::flat_hash_map<Node *,
                    absl::flat_hash_map<Node *, absl::flat_hash_set<Node *>>>
ComputeMutualExclusionAnalysis(
    FunctionBase *f, OptimizationContext &context,
    absl::flat_hash_map<Node *, absl::flat_hash_set<Node *>>
        &reachability_result) {
  absl::flat_hash_map<Node *,
                      absl::flat_hash_map<Node *, absl::flat_hash_set<Node *>>>
      mutual_exclusivity_relation;

  // Compute the mutual exclusion binary relation between instructions
  for (const auto &[n, s] : reachability_result) {
    // Find the next select
    if (!n->OpIn({Op::kSel, Op::kPrioritySel, Op::kOneHotSel})) {
      continue;
    }

    // Compute the mutually-exclusive instructions
    absl::Span<Node *const> cases = GetCases(n);
    for (uint32_t case_number = 0; case_number < cases.length();
         case_number++) {
      Node *current_case = cases[case_number];

      // Check if any of the nodes that reach the current case (including it)
      // are mutually exclusive with the nodes that reach the next cases
      for (Node *current_case_reaching_node :
           reachability_result[current_case]) {
        // Check if the current reaching node reaches the other cases
        for (uint32_t case_number_2 = case_number + 1;
             case_number_2 < cases.length(); case_number_2++) {
          Node *current_case_2 = cases[case_number_2];
          if (reachability_result[current_case_2].contains(
                  current_case_reaching_node)) {
            continue;
          }

          // The current reaching node does not reach the current other case.
          // Add as mutually-exclusive all reaching nodes of the current other
          // case that also do not reach @current_case_reaching_node
          for (Node *other_case_reaching_node :
               reachability_result[current_case_2]) {
            if (reachability_result[current_case].contains(
                    other_case_reaching_node)) {
              continue;
            }
            if (current_case_reaching_node < other_case_reaching_node) {
              mutual_exclusivity_relation[n][current_case_reaching_node].insert(
                  other_case_reaching_node);
            } else {
              mutual_exclusivity_relation[n][other_case_reaching_node].insert(
                  current_case_reaching_node);
            }
          }
        }
      }
    }
  }

  // Print the mutual exclusivity relation
  VLOG(4) << "Mutually exclusive graph";
  for (const auto &mer : mutual_exclusivity_relation) {
    VLOG(4) << "  Select: " << mer.first->ToString();
    for (const auto &[n0, s0] : mer.second) {
      VLOG(4) << "  " << n0->ToString();
      for (auto n1 : s0) {
        VLOG(4) << "    -> " << n1->ToString();
      }
    }
  }

  return mutual_exclusivity_relation;
}

// This function returns all possible folding actions that we can legally
// perform.
absl::flat_hash_set<FoldingAction *> ComputeFoldableActions(
    absl::flat_hash_map<
        Node *, absl::flat_hash_map<Node *, absl::flat_hash_set<Node *>>>
        &mutual_exclusivity_relation,
    absl::flat_hash_map<Node *, absl::flat_hash_set<Node *>>
        &reachability_result) {
  absl::flat_hash_set<FoldingAction *> foldable_actions;

  // Compute all possible foldable actions
  for (const auto &mer : mutual_exclusivity_relation) {
    Node *select = mer.first;
    for (const auto &[n0, s0] : mer.second) {
      if (!CanTarget(n0)) {
        continue;
      }
      for (auto n1 : s0) {
        if (!CanTarget(n1)) {
          continue;
        }

        // Both nodes can be targeted by resource sharing
        //
        // Check if they are compatible
        if (!AreCompatible(n0, n1)) {
          continue;
        }

        // The nodes can be targeted by resource sharing and they are
        // compatible.
        //
        // Check if we know enough the fold one into the other
        std::optional<FoldingAction *> f =
            CanMapInto(n0, n1, select, reachability_result);
        if (f.has_value()) {
          foldable_actions.insert(*f);
        }
        f = CanMapInto(n1, n0, select, reachability_result);
        if (f.has_value()) {
          foldable_actions.insert(*f);
        }
      }
    }
  }

  // Print the folding actions found
  VLOG(3) << "Possible folding actions";
  for (FoldingAction *folding : foldable_actions) {
    VLOG(3) << "  From " << folding->GetFrom()->ToString();
    VLOG(3) << "  To " << folding->GetTo()->ToString();
    VLOG(3) << "    Select = " << folding->GetSelect()->ToString();
    VLOG(3) << "      From is case " << folding->GetFromCaseNumber();
    VLOG(3) << "      To is case " << folding->GetToCaseNumber();
  }

  return foldable_actions;
}

// This function chooses the subset of foldable actions to perform and decide
// their total order to perform them.
std::vector<FoldingAction *> SelectFoldingActions(
    absl::flat_hash_set<FoldingAction *> &foldable_actions) {
  std::vector<FoldingAction *> folding_actions_to_perform;

  // Choose all of them
  for (FoldingAction *folding : foldable_actions) {
    folding_actions_to_perform.push_back(folding);
  }

  return folding_actions_to_perform;
}

// This function performs the folding actions specified in its input following
// the order specified.
absl::StatusOr<bool> PerformFoldingActions(
    FunctionBase *f, std::vector<FoldingAction *> &folding_actions_to_perform) {
  bool modified = false;

  // Perform the folding actions specified
  absl::flat_hash_set<Node *> node_modified;
  for (FoldingAction *folding : folding_actions_to_perform) {
    modified = true;

    // Fetch the nodes to fold
    Node *from_node = folding->GetFrom();
    Node *to_node = folding->GetTo();

    // Check if any of these nodes have been involved in a folding action
    // already performed.
    if (node_modified.contains(from_node) || node_modified.contains(to_node)) {
      continue;
    }
    node_modified.insert(from_node);
    node_modified.insert(to_node);
    VLOG(2) << "  From " << folding->GetFrom()->ToString();
    VLOG(2) << "  To " << folding->GetTo()->ToString();
    VLOG(2) << "    Select = " << folding->GetSelect()->ToString();
    VLOG(2) << "      From is case " << folding->GetFromCaseNumber();
    VLOG(2) << "      To is case " << folding->GetToCaseNumber();

    // Fold
    //
    // - Step 0: Get the subset of the bits of the selector that are relevant
    Node *selector = folding->GetSelector();
    XLS_ASSIGN_OR_RETURN(Node * from_bit, f->MakeNode<BitSlice>(
                                              selector->loc(), selector,
                                              folding->GetFromCaseNumber(), 1));
    Node *new_selector = from_bit;

    // - Step 1: Create a new selector for each input
    std::vector<Node *> new_operands;
    for (uint32_t op_id = 0; op_id < from_node->operand_count(); op_id++) {
      // Fetch the current operand for both nodes
      Node *from_operand = from_node->operand(op_id);
      Node *to_operand = to_node->operand(op_id);

      // Generate a select between them
      std::vector<Node *> operand_select_cases = {from_operand};
      XLS_ASSIGN_OR_RETURN(
          Node * operand_select,
          f->MakeNode<PrioritySelect>(selector->loc(), new_selector,
                                      operand_select_cases, to_operand));
      new_operands.push_back(operand_select);
    }
    CHECK_EQ(new_operands.size(), 2);

    // - Step 2: Replace the operands of the @to_node to use the results of the
    //           new selectors computed at Step 1.
    for (uint32_t op_id = 0; op_id < to_node->operand_count(); op_id++) {
      XLS_RETURN_IF_ERROR(
          to_node->ReplaceOperandNumber(op_id, new_operands[op_id], true));
    }

    // - Step 3: Replace @from_node uses with the new @to_node
    XLS_RETURN_IF_ERROR(from_node->ReplaceUsesWith(to_node));

    // - Step 4: Remove the now-dead @from_node
    XLS_RETURN_IF_ERROR(f->RemoveNode(from_node));
  }

  return modified;
}

// This function computes the resource sharing optimization for multiplication
// instructions. In more detail, this function folds a multiplication
// instruction into another multiplication instruction that has the same
// bitwidth for all operands as well as for the result.
//
// This folding operation is performed for all multiplication instructions that
// allow it (i.e., the transformation is legal).
absl::StatusOr<bool> ResourceSharingPass::RunOnFunctionBaseInternal(
    FunctionBase *f, const OptimizationPassOptions &options,
    PassResults *results, OptimizationContext &context) const {
  // Check if the pass is enabled
  if (!options.enable_resource_sharing) {
    return false;
  }
  bool modified = false;

  // Perform the reachability analysis.
  absl::flat_hash_map<Node *, absl::flat_hash_set<Node *>> reachability_result =
      ComputeReachabilityAnalysis(f, context);

  // Compute the mutually exclusive binary relation between IR instructions
  absl::flat_hash_map<Node *,
                      absl::flat_hash_map<Node *, absl::flat_hash_set<Node *>>>
      mutual_exclusivity_relation =
          ComputeMutualExclusionAnalysis(f, context, reachability_result);

  // Identify the set of legal folding actions
  absl::flat_hash_set<FoldingAction *> foldable_actions =
      ComputeFoldableActions(mutual_exclusivity_relation, reachability_result);

  // Select the folding actions to perform
  std::vector<FoldingAction *> folding_actions_to_perform =
      SelectFoldingActions(foldable_actions);

  // Perform the folding
  VLOG(2) << "Folding actions to perform";
  XLS_ASSIGN_OR_RETURN(modified,
                       PerformFoldingActions(f, folding_actions_to_perform));

  // Free the memory
  for (FoldingAction *a : foldable_actions) {
    delete a;
  }

  return modified;
}

FoldingAction::FoldingAction(Node *from, Node *to, Node *select,
                             uint32_t from_case_number, uint32_t to_case_number)
    : from_{from},
      to_{to},
      select_{select},
      from_case_number_{from_case_number},
      to_case_number_{to_case_number} {}

Node *FoldingAction::GetFrom(void) const { return from_; }

Node *FoldingAction::GetTo(void) const { return to_; }

Node *FoldingAction::GetSelect(void) const { return select_; }

Node *FoldingAction::GetSelector(void) const {
  Node *s = ::xls::GetSelector(select_);
  return s;
}

uint32_t FoldingAction::GetFromCaseNumber(void) const {
  return from_case_number_;
}

uint32_t FoldingAction::GetToCaseNumber(void) const { return to_case_number_; }

REGISTER_OPT_PASS(ResourceSharingPass);

}  // namespace xls
