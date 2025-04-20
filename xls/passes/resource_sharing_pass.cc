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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
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
#include "xls/passes/post_dominator_analysis.h"
#include "ortools/graph/graph.h"
#include "ortools/graph/cliques.h"

namespace xls {

// This class represents a single folding action where the destination of the
// folding operation is a single IR node.
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
//
// This class is meant to be extended with sub-classes that specify the sources
// (e.g., a single IR node, or a set of nodes) of the folding operation.
class FoldingAction {
 public:
  Node *GetTo(void) const;

  Node *GetSelect(void) const;

  Node *GetSelector(void) const;

  uint32_t GetToCaseNumber(void) const;

 protected:
  FoldingAction(Node *to, Node *select, uint32_t to_case_number);

 private:
  Node *to_;
  Node *select_;
  uint32_t to_case_number_;
};

// This class represents a single folding action from an IR node into another IR
// node.
//
// An example of such folding action is the following.
// Consider the following IR:
//   a2 = umul a1, a0
//   b2 = umul b1, b0
//   r  = priority_sel(s, cases=[a2, b2])
//
// An instance of the class BinaryFoldingAction where
// - @from is "a2"
// - @to is "b2"
// - @to_case_number is 1 and
// - @select is "r"
// is the folding that transforms the code above into the code below:
//   custom_s = bit_slice(s, start=0, width=1)
//   lhs = priority_sel(custom_s, cases=[a1], default=b1)
//   rhs = priority_sel(custom_s, cases=[a0], default=b0)
//   r   = umul lhs, rhs
class BinaryFoldingAction : public FoldingAction {
 public:
  BinaryFoldingAction(Node *from, Node *to, Node *select,
                      uint32_t from_case_number, uint32_t to_case_number);

  Node *GetFrom(void) const;

  uint32_t GetFromCaseNumber(void) const;

 private:
  Node *from_;
  uint32_t from_case_number_;
};

// This class represents a single folding action from a set of IR nodes into
// another IR node.
//
// An example of such folding action is the following.
// Consider the following IR:
//   a2 = umul a1, a0
//   b2 = umul b1, b0
//   c2 = umul c1, c0
//   d2 = umul d1, d0
//   r  = priority_sel(s, cases=[a2, b2, c2, d2])
//
// An instance of the class NaryFoldingAction where
// - @from is "<a2, 0>, <b2, 1>, <c2, 2>"
// - @to is "d2"
// - @to_case_number is "3" and
// - @select is "r"
// is the folding the transforms the code above into the code below:
//   custom_s = bit_slice(s, start=0, width=3)
//   lhs = priority_sel(custom_s, cases=[a1, b1, c1], default=d1)
//   rhs = priority_sel(custom_s, cases=[a0, b0, c0], default=d0)
//   r   = umul lhs, rhs
class NaryFoldingAction : public FoldingAction {
 public:
  NaryFoldingAction(
      const absl::flat_hash_set<std::pair<Node *, uint32_t>> &from, Node *to,
      Node *select, uint32_t to_case_number);

  absl::flat_hash_set<std::pair<Node *, uint32_t>> GetFrom(void) const;

 private:
  absl::flat_hash_set<std::pair<Node *, uint32_t>> from_;
};

// This class organizes the set of binary folding given as input into a graph
// where a binary folding from node ni to the node nj is an edge from ni to nj.
class FoldingGraph {
 public:
  FoldingGraph(
      std::vector<std::unique_ptr<BinaryFoldingAction>> foldable_actions,
      const absl::flat_hash_map<
          Node *, absl::flat_hash_map<Node *, absl::flat_hash_set<Node *>>>
          &mutual_exclusivity_relation);

  // This function returns a set of sets of edges, one outermost set per clique
  // found in the graph.
  // Each outermost set includes the set of edges of the graph that creates a
  // clique within that graph.
  absl::flat_hash_set<absl::flat_hash_set<BinaryFoldingAction *>>
  GetEdgeCliques(void) const;

 private:
  using NodeIndex = int32_t;
  using EdgeIndex = int32_t;
  using Graph = ::util::ReverseArcStaticGraph<NodeIndex, EdgeIndex>;
  std::unique_ptr<Graph> graph_;
  std::vector<Node *> nodes_;
  absl::flat_hash_map<Node *, NodeIndex> node_to_index_;
  std::vector<std::unique_ptr<BinaryFoldingAction>> edges_;
  absl::flat_hash_set<absl::flat_hash_set<NodeIndex>> cliques_;

  void AddNodes(
      absl::Span<const std::unique_ptr<BinaryFoldingAction>> foldable_actions);
  void AddEdges(
      std::vector<std::unique_ptr<BinaryFoldingAction>> foldable_actions);
  void IdentifyCliques(void);
};

// This function permutes the order of the elements of the array
// @array_to_permute following the permutations listed in @permutation.
//
// This function is similar to util::Permute. There are only two differences
// between this function and util::Permute:
// 1) This function uses std::move rather than relying on the copy constructor.
//    This is important when using smart pointers.
// 2) This function relies on "typeof" to find the type of the elements of the
//    array to permute.
template <class IntVector, class Array>
void Permute(const IntVector &permutation, Array *array_to_permute) {
  if (permutation.empty()) {
    return;
  }
  std::vector<std::remove_reference_t<decltype((*array_to_permute)[0])>> temp(
      permutation.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    temp[i] = std::move((*array_to_permute)[i]);
  }
  for (size_t i = 0; i < permutation.size(); ++i) {
    (*array_to_permute)[static_cast<size_t>(permutation[i])] =
        std::move(temp[i]);
  }
}

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

// This function returns true if @n0 is mutually exclusive with @n1 thanks to
// the select @select, false otherwise.
bool AreMutuallyExclusive(
    const absl::flat_hash_map<
        Node *, absl::flat_hash_map<Node *, absl::flat_hash_set<Node *>>>
        &mutual_exclusivity_relation,
    Node *n0, Node *n1, Node *select) {
  // Fetch the relation created by the select given as input
  if (!mutual_exclusivity_relation.contains(select)) {
    return false;
  }
  const absl::flat_hash_map<Node *, absl::flat_hash_set<Node *>>
      &relation_due_to_select = mutual_exclusivity_relation.at(select);

  // Get the nodes that are mutually exclusive with @n0
  if (!relation_due_to_select.contains(n0)) {
    return false;
  }
  const absl::flat_hash_set<Node *> &mutually_exclusive_nodes_of_n0 =
      relation_due_to_select.at(n0);

  // Check if the nodes are mutually exclusive
  if (mutually_exclusive_nodes_of_n0.count(n1) > 0) {
    return true;
  }
  return false;
}

// This function returns true if @node_to_check reaches @point_in_the_graph,
// false otherwise.
bool DoesReach(const absl::flat_hash_map<Node *, absl::flat_hash_set<Node *>>
                   &reachability_result,
               Node *point_in_the_graph, Node *node_to_check) {
  // Fetch the set of nodes that reach @point_in_the_graph
  auto iter = reachability_result.find(point_in_the_graph);
  CHECK(iter != reachability_result.end());
  const absl::flat_hash_set<Node *> &reaching_nodes = iter->second;

  // Check if the specified node reaches @point_in_the_graph
  return reaching_nodes.contains(node_to_check);
}

std::optional<uint32_t> GetSelectCaseNumberOfNode(
    const absl::flat_hash_map<Node *, absl::flat_hash_set<Node *>>
        &reachability_result,
    Node *node, Node *select_case, uint32_t select_case_number) {
  // Check if @node reaches the select case given as input
  if (!DoesReach(reachability_result, select_case, node)) {
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
std::optional<std::unique_ptr<BinaryFoldingAction>> CanMapInto(
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
  std::unique_ptr<BinaryFoldingAction> f =
      std::make_unique<BinaryFoldingAction>(node_to_map, folding_destination,
                                            select, node_to_map_case_number,
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
absl::StatusOr<absl::flat_hash_map<
    Node *, absl::flat_hash_map<Node *, absl::flat_hash_set<Node *>>>>
ComputeMutualExclusionAnalysis(
    FunctionBase *f, OptimizationContext &context,
    absl::flat_hash_map<Node *, absl::flat_hash_set<Node *>>
        &reachability_result) {
  absl::flat_hash_map<Node *,
                      absl::flat_hash_map<Node *, absl::flat_hash_set<Node *>>>
      mutual_exclusivity_relation;

  // Run the post-dominance analysis.
  //
  // The result of this analysis is used to determine whether a given node @n is
  // mutually exclusive with another node @m where both of them reach a select
  // node @s.
  //
  // In more detail, the post-dominance analysis result is used to guarantee the
  // following code will lead to conclude @n isn't mutually exclusive with @m:
  // n = ...
  // m = ...
  // s = priority_sel(selector, cases=[n, m])
  // i = add(n, s)
  // return i
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<PostDominatorAnalysis> post_dominators,
                       PostDominatorAnalysis::Run(f));

  // Compute the mutual exclusion binary relation between instructions
  for (const auto &[n, s] : reachability_result) {
    // Find the next select
    if (!n->OpIn({Op::kSel, Op::kPrioritySel, Op::kOneHotSel})) {
      continue;
    }

    // Compute the mutually-exclusive instructions created by the select @n
    absl::Span<Node *const> cases = GetCases(n);
    for (uint32_t case_number = 0; case_number < cases.length();
         case_number++) {
      Node *current_case = cases[case_number];

      // Check if any of the nodes that reach the current case (including it)
      // are mutually exclusive with the nodes that reach the next cases
      for (Node *current_case_reaching_node :
           reachability_result[current_case]) {
        // Do not bother looking at nodes that we will not be able to fold
        if (!CanTarget(current_case_reaching_node)) {
          continue;
        }

        // Only nodes that are post-dominated by the select are considered to be
        // mutually exclusive with another node
        if (!post_dominators->NodeIsPostDominatedBy(current_case_reaching_node,
                                                    n)) {
          continue;
        }

        // Check if the current reaching node reaches the other cases
        for (uint32_t case_number_2 = case_number + 1;
             case_number_2 < cases.length(); case_number_2++) {
          Node *current_case_2 = cases[case_number_2];
          if (DoesReach(reachability_result, current_case_2,
                        current_case_reaching_node)) {
            continue;
          }

          // The current reaching node @current_case_reaching_node does not
          // reach the other case @current_case_2.
          //
          // Add as mutually-exclusive all reaching nodes of the current other
          // case @current_case_2 that also do not reach
          // @current_case_reaching_node.
          for (Node *other_case_reaching_node :
               reachability_result[current_case_2]) {
            // Do not bother looking at nodes that we will not be able to fold
            if (!CanTarget(other_case_reaching_node)) {
              continue;
            }

            // Only nodes that are post-dominated by the select are considered
            // to be mutually exclusive with another node
            if (!post_dominators->NodeIsPostDominatedBy(
                    other_case_reaching_node, n)) {
              continue;
            }

            // If @other_case_reaching_node reaches @current_case, then it
            // cannot be mutually exclusive with @current_case_reaching_node
            if (DoesReach(reachability_result, current_case,
                          other_case_reaching_node)) {
              continue;
            }

            // @current_case_reaching_node and @other_case_reaching_node are
            // mutually exclusive.
            mutual_exclusivity_relation[n][current_case_reaching_node].insert(
                other_case_reaching_node);
            mutual_exclusivity_relation[n][other_case_reaching_node].insert(
                current_case_reaching_node);
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
        VLOG(4) << "    <-> " << n1->ToString();
      }
    }
  }

  return mutual_exclusivity_relation;
}

// This function returns all possible folding actions that we can legally
// perform.
std::vector<std::unique_ptr<BinaryFoldingAction>> ComputeFoldableActions(
    absl::flat_hash_map<
        Node *, absl::flat_hash_map<Node *, absl::flat_hash_set<Node *>>>
        &mutual_exclusivity_relation,
    absl::flat_hash_map<Node *, absl::flat_hash_set<Node *>>
        &reachability_result) {
  std::vector<std::unique_ptr<BinaryFoldingAction>> foldable_actions;

  // Identify as many folding actions that are legal as possible by our current
  // analyses.
  for (const auto &mer : mutual_exclusivity_relation) {
    Node *select = mer.first;
    for (const auto &[n0, s0] : mer.second) {
      // Skip nodes that cannot be folded
      if (!CanTarget(n0)) {
        continue;
      }

      // Find nodes that n0 can fold into
      for (Node *n1 : s0) {
        // Since the mutual exclusive relation is symmetric, use only one side
        // of it
        if (n1->id() < n0->id()) {
          continue;
        }

        // Skip nodes that cannot be folded
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
        std::optional<std::unique_ptr<BinaryFoldingAction>> f_0_1 =
            CanMapInto(n0, n1, select, reachability_result);
        if (f_0_1.has_value()) {
          foldable_actions.push_back(std::move(*f_0_1));
        }
        std::optional<std::unique_ptr<BinaryFoldingAction>> f_1_0 =
            CanMapInto(n1, n0, select, reachability_result);
        if (f_1_0.has_value()) {
          foldable_actions.push_back(std::move(*f_1_0));
        }
      }
    }
  }

  // Print the folding actions found
  VLOG(3) << "Possible folding actions";
  for (const std::unique_ptr<BinaryFoldingAction> &folding : foldable_actions) {
    VLOG(3) << "  From " << folding->GetFrom()->ToString();
    VLOG(3) << "  To " << folding->GetTo()->ToString();
    VLOG(3) << "    Select = " << folding->GetSelect()->ToString();
    VLOG(3) << "    From is case " << folding->GetFromCaseNumber();
    VLOG(3) << "    To is case " << folding->GetToCaseNumber();
  }

  return foldable_actions;
}

// This function chooses the subset of foldable actions to perform and decide
// their total order to perform them.
std::vector<NaryFoldingAction *> SelectFoldingActions(
    FoldingGraph *folding_graph) {
  std::vector<NaryFoldingAction *> folding_actions_to_perform;

  // Choose all of them matching the maximum cliques of the folding graph
  for (const absl::flat_hash_set<BinaryFoldingAction *> &clique :
       folding_graph->GetEdgeCliques()) {
    // Get the destination that is shared among all elements in the clique
    BinaryFoldingAction *first_action = *clique.begin();
    Node *to = first_action->GetTo();
    uint32_t to_case_number = first_action->GetToCaseNumber();

    // Get the select
    Node *select = first_action->GetSelect();

    // Get the nodes to map into the destination
    //
    // This step requires to translate a clique into a star where the center is
    // the common destination chosen for the entire clique.
    // For now, we choose a random element of the clique as the center of the
    // start. Later, we should find a good heuristic to pick a candidate that
    // lead to a better PPA.
    absl::flat_hash_set<std::pair<Node *, uint32_t>> froms;
    for (BinaryFoldingAction *binary_folding : clique) {
      CHECK_EQ(binary_folding->GetSelect(), select);
      if (binary_folding->GetTo() != to) {
        // We can skip this binary folding because it doesn't target the
        // destination we chose.
        continue;
      }
      froms.insert(std::make_pair(binary_folding->GetFrom(),
                                  binary_folding->GetFromCaseNumber()));
    }

    // Create a single n-ary folding action for the whole clique
    NaryFoldingAction *new_action =
        new NaryFoldingAction(froms, to, select, to_case_number);
    folding_actions_to_perform.push_back(new_action);
  }

  return folding_actions_to_perform;
}

// This function performs the folding actions specified in its input following
// the order specified.
absl::StatusOr<bool> PerformFoldingActions(
    FunctionBase *f,
    const std::vector<NaryFoldingAction *> &folding_actions_to_perform) {
  bool modified = false;
  VLOG(2) << "There are " << folding_actions_to_perform.size()
          << " folding actions to perform";

  // Perform the folding actions specified
  absl::flat_hash_set<Node *> node_modified;
  for (NaryFoldingAction *folding : folding_actions_to_perform) {
    modified = true;
    VLOG(2) << "  Next folding to perform:\n";
    for (auto [from_node, from_node_case_number] : folding->GetFrom()) {
      VLOG(2) << "    From [" << from_node_case_number << "] "
              << from_node->ToString();
    }
    VLOG(2) << "    To [" << folding->GetToCaseNumber() << "] "
            << folding->GetTo()->ToString();
    VLOG(2) << "    Select " << folding->GetSelect()->ToString();

    // Fetch the nodes to fold that have not been folded already
    std::vector<std::pair<Node *, uint32_t>> froms_to_use;
    for (auto [from_node, from_node_case_number] : folding->GetFrom()) {
      if (!node_modified.contains(from_node)) {
        froms_to_use.push_back(
            std::make_pair(from_node, from_node_case_number));
      }
    }

    // Check if we have a folding action to perform with the current nodes.
    Node *to_node = folding->GetTo();
    if ((froms_to_use.empty()) || node_modified.contains(to_node)) {
      VLOG(2) << "    Not possible because of prior folding already performed";
      continue;
    }

    // Tag the nodes involved in the current folding action as modified.
    // This is to avoid folding the same node multiple times.
    for (auto [from_node, from_node_case_number] : froms_to_use) {
      node_modified.insert(from_node);
    }
    node_modified.insert(to_node);

    // Sort the from nodes in ascending order based on their select case number.
    // This will help us synthesize the correct selector to use
    auto from_comparator = [](std::pair<Node *, uint32_t> p0,
                              std::pair<Node *, uint32_t> p1) -> bool {
      return p0.second < p1.second;
    };
    std::sort(froms_to_use.begin(), froms_to_use.end(), from_comparator);
    VLOG(2) << "    Froms that were not involved in prior foldings";
    for (auto [from_node, from_node_case_number] : froms_to_use) {
      VLOG(2) << "      [" << from_node_case_number << "] "
              << from_node->ToString();
    }

    // Fold
    //
    // - Step 0: Get the subset of the bits of the selector that are relevant
    VLOG(3) << "    Step 0: generate the new selector";
    Node *selector = folding->GetSelector();
    std::vector<Node *> from_bits;
    from_bits.reserve(froms_to_use.size());
    for (auto [from_node, from_node_case_number] : froms_to_use) {
      XLS_ASSIGN_OR_RETURN(Node * from_bit,
                           f->MakeNode<BitSlice>(selector->loc(), selector,
                                                 from_node_case_number, 1));
      from_bits.push_back(from_bit);
    }
    absl::c_reverse(from_bits);
    XLS_ASSIGN_OR_RETURN(Node * new_selector,
                         f->MakeNode<Concat>(selector->loc(), from_bits));
    VLOG(3) << "      " << new_selector->ToString();

    // - Step 1: Create a new select for each input
    VLOG(3) << "    Step 1: generate the priority selects, one per input of "
               "the folding target";
    std::vector<Node *> new_operands;
    for (uint32_t op_id = 0; op_id < to_node->operand_count(); op_id++) {
      // Fetch the current operand for the target of the folding action.
      Node *to_operand = to_node->operand(op_id);

      // Generate all select cases, one for each source of the folding action
      std::vector<Node *> operand_select_cases;
      for (auto [from_node, from_node_case_number] : froms_to_use) {
        Node *from_operand = from_node->operand(op_id);
        operand_select_cases.push_back(from_operand);
      }

      // Generate a select between them
      XLS_ASSIGN_OR_RETURN(
          Node * operand_select,
          f->MakeNode<PrioritySelect>(selector->loc(), new_selector,
                                      operand_select_cases, to_operand));
      new_operands.push_back(operand_select);
      VLOG(3) << "      " << operand_select->ToString();
    }
    CHECK_EQ(new_operands.size(), 2);

    // - Step 2: Replace the operands of the @to_node to use the results of the
    //           new selectors computed at Step 1.
    VLOG(3) << "    Step 2: update the target of the folding transformation";
    for (int64_t op_id = 0LL; op_id < to_node->operand_count(); op_id++) {
      XLS_RETURN_IF_ERROR(
          to_node->ReplaceOperandNumber(op_id, new_operands[op_id], true));
    }
    VLOG(3) << "      " << to_node->ToString();

    // - Step 3: Replace every source of the folding action with the new
    // @to_node
    VLOG(3)
        << "    Step 3: update the def-use chains to use the new folded node";
    for (auto [from_node, from_node_case_number] : froms_to_use) {
      XLS_RETURN_IF_ERROR(from_node->ReplaceUsesWith(to_node));
    }

    // - Step 4: Remove all the sources of the folding action as they are now
    //           dead
    VLOG(3) << "    Step 4: remove the sources of the folding transformation";
    for (auto [from_node, from_node_case_number] : froms_to_use) {
      XLS_RETURN_IF_ERROR(f->RemoveNode(from_node));
    }
    VLOG(3) << "    Folding completed";
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
      mutual_exclusivity_relation;
  XLS_ASSIGN_OR_RETURN(
      mutual_exclusivity_relation,
      ComputeMutualExclusionAnalysis(f, context, reachability_result));

  // Identify the set of legal folding actions
  std::vector<std::unique_ptr<BinaryFoldingAction>> foldable_actions =
      ComputeFoldableActions(mutual_exclusivity_relation, reachability_result);

  // Organize the folding actions into a graph
  FoldingGraph folding_graph{std::move(foldable_actions),
                             mutual_exclusivity_relation};

  // Select the folding actions to perform
  std::vector<NaryFoldingAction *> folding_actions_to_perform =
      SelectFoldingActions(&folding_graph);

  // Perform the folding
  XLS_ASSIGN_OR_RETURN(modified,
                       PerformFoldingActions(f, folding_actions_to_perform));

  // Free the memory
  for (NaryFoldingAction *a : folding_actions_to_perform) {
    delete a;
  }

  return modified;
}

FoldingAction::FoldingAction(Node *to, Node *select, uint32_t to_case_number)
    : to_{to}, select_{select}, to_case_number_{to_case_number} {}

Node *FoldingAction::GetTo(void) const { return to_; }

Node *FoldingAction::GetSelect(void) const { return select_; }

Node *FoldingAction::GetSelector(void) const {
  Node *s = ::xls::GetSelector(select_);
  return s;
}

uint32_t FoldingAction::GetToCaseNumber(void) const { return to_case_number_; }

BinaryFoldingAction::BinaryFoldingAction(Node *from, Node *to, Node *select,
                                         uint32_t from_case_number,
                                         uint32_t to_case_number)
    : FoldingAction{to, select, to_case_number},
      from_{from},
      from_case_number_{from_case_number} {}

Node *BinaryFoldingAction::GetFrom(void) const { return from_; }

uint32_t BinaryFoldingAction::GetFromCaseNumber(void) const {
  return from_case_number_;
}

NaryFoldingAction::NaryFoldingAction(
    const absl::flat_hash_set<std::pair<Node *, uint32_t>> &from, Node *to,
    Node *select, uint32_t to_case_number)
    : FoldingAction{to, select, to_case_number}, from_{from} {}

absl::flat_hash_set<std::pair<Node *, uint32_t>> NaryFoldingAction::GetFrom(
    void) const {
  return from_;
}

FoldingGraph::FoldingGraph(
    std::vector<std::unique_ptr<BinaryFoldingAction>> foldable_actions,
    const absl::flat_hash_map<
        Node *, absl::flat_hash_map<Node *, absl::flat_hash_set<Node *>>>
        &mutual_exclusivity_relation) {
  // Allocate the graph
  graph_ = std::make_unique<Graph>();

  // Add the nodes
  AddNodes(foldable_actions);

  // Add the edges
  AddEdges(std::move(foldable_actions));

  // Build the graph
  std::vector<EdgeIndex> edge_permutations;
  graph_->Build(&edge_permutations);
  Permute(edge_permutations, &edges_);

  // Print the folding graph
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "Folding graph";
    for (NodeIndex node_index : graph_->AllNodes()) {
      Node *from_node = nodes_[node_index];
      VLOG(2) << "  [" << node_index << "] " << from_node->ToString();
      for (EdgeIndex edge_index : graph_->OutgoingArcs(node_index)) {
        NodeIndex to_node_index = graph_->Head(edge_index);
        Node *to_node = nodes_[to_node_index];
        CHECK_EQ(edges_[edge_index]->GetFrom(), from_node);
        CHECK_EQ(edges_[edge_index]->GetTo(), to_node);
        VLOG(2) << "    -> [" << to_node_index << "] " << to_node->ToString();
      }
    }
  }

  // Identify the cliques within the graph
  IdentifyCliques();
}

void FoldingGraph::AddNodes(
    absl::Span<const std::unique_ptr<BinaryFoldingAction>> foldable_actions) {
  // Add all nodes involved in folding actions into the internal
  // representation.
  absl::flat_hash_set<Node *> already_added;
  for (const std::unique_ptr<BinaryFoldingAction> &f : foldable_actions) {
    // Add the nodes to our internal representation if they were not added
    // already
    Node *from_node = f->GetFrom();
    Node *to_node = f->GetTo();
    if (!already_added.contains(from_node)) {
      already_added.insert(from_node);
      nodes_.push_back(from_node);
    }
    if (!already_added.contains(to_node)) {
      already_added.insert(to_node);
      nodes_.push_back(to_node);
    }
  }

  // Add the mapping from Node to its index
  node_to_index_.reserve(nodes_.size());
  for (size_t i = 0; i < nodes_.size(); ++i) {
    node_to_index_[nodes_[i]] = i;
  }

  // Add the nodes to the graph
  for (size_t i = 0; i < nodes_.size(); ++i) {
    graph_->AddNode(i);
  }
}

void FoldingGraph::AddEdges(
    std::vector<std::unique_ptr<BinaryFoldingAction>> foldable_actions) {
  // Add all edges to the graph
  for (std::unique_ptr<BinaryFoldingAction> &f : foldable_actions) {
    // Add a new edge into the graph to represent the current folding action
    NodeIndex from_index = node_to_index_.at(f->GetFrom());
    NodeIndex to_index = node_to_index_.at(f->GetTo());
    CHECK(graph_->IsNodeValid(from_index));
    CHECK(graph_->IsNodeValid(to_index));
    graph_->AddArc(from_index, to_index);

    // Add the current folding action to our internal representation
    edges_.push_back(std::move(f));
  }
}

void FoldingGraph::IdentifyCliques(void) {
  auto graph_descriptor = [this](int from_node, int to_node) -> bool {
    for (EdgeIndex outgoing_edge : this->graph_->OutgoingArcs(from_node)) {
      if (this->graph_->Head(outgoing_edge) == to_node) {
        return true;
      }
    }
    return false;
  };
  auto found_clique = [this](const std::vector<int> &clique) -> bool {
    absl::flat_hash_set<NodeIndex> clique_to_add;
    VLOG(2) << "New clique:";
    for (NodeIndex node : clique) {
      VLOG(2) << "  " << node;
      clique_to_add.insert(node);
    }
    this->cliques_.insert(clique_to_add);

    return false;
  };
  ::operations_research::FindCliques(graph_descriptor, nodes_.size(),
                                     found_clique);
}

absl::flat_hash_set<absl::flat_hash_set<BinaryFoldingAction *>>
FoldingGraph::GetEdgeCliques(void) const {
  absl::flat_hash_set<absl::flat_hash_set<BinaryFoldingAction *>> cliques;

  // Find all the cliques of edges
  for (const absl::flat_hash_set<NodeIndex> &node_clique : cliques_) {
    // Find all the edges within the clique
    absl::flat_hash_set<BinaryFoldingAction *> edge_clique;
    for (NodeIndex from_node_index : node_clique) {
      for (EdgeIndex outgoing_edge : graph_->OutgoingArcs(from_node_index)) {
        CHECK_LT(outgoing_edge, edges_.size());

        // Fetch the destination of the current outgoing edge
        NodeIndex to_node_index = graph_->Head(outgoing_edge);
        CHECK_NE(from_node_index, to_node_index);

        // Check if the current edge belongs to the clique
        if (!node_clique.contains(to_node_index)) {
          continue;
        }

        // We found a new edge that belongs to the clique
        BinaryFoldingAction *new_folding_action_within_clique =
            edges_[outgoing_edge].get();
        CHECK_NE(new_folding_action_within_clique, nullptr);
        edge_clique.insert(new_folding_action_within_clique);
      }
    }

    // Add the new edge clique
    cliques.insert(edge_clique);
  }

  return cliques;
}

REGISTER_OPT_PASS(ResourceSharingPass);

}  // namespace xls
