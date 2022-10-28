// Copyright 2022 The XLS Authors
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

#include "xls/passes/mutual_exclusion_pass.h"

#include <random>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/graph_coloring.h"
#include "xls/data_structures/transitive_closure.h"
#include "xls/data_structures/union_find_map.h"
#include "xls/interpreter/random_value.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/node_util.h"
#include "xls/ir/op.h"
#include "xls/ir/value.h"
#include "xls/ir/value_helpers.h"
#include "xls/passes/cse_pass.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/post_dominator_analysis.h"
#include "xls/passes/token_provenance_analysis.h"
#include "xls/solvers/z3_ir_translator.h"
#include "xls/solvers/z3_utils.h"
#include "../z3/src/api/z3.h"

namespace xls {

namespace {

bool FunctionIsOneBit(Node* node) {
  return node->GetType()->IsBits() &&
         node->GetType()->AsBitsOrDie()->bit_count() == 1;
}

// This stores a mapping from nodes in a FunctionBase to 1-bit nodes that are
// the "predicate" of that node. The idea is that it should always be sound to
// replace a node `N` that has predicate `P` with `gate(P, N)` (where `gate` is
// extended from its usual semantics to also gate the effect of side-effectful
// operations).
//
// It also contains a relation on the set of predicate nodes that describes
// whether any given pair of predicates is known to be mutually exclusive; i.e.:
// for a pair `(A, B)` whether `A NAND B` is known to be valid. If `A NAND B` is
// known not to be valid, that information is tracked too.
class Predicates {
 public:
  // Set the predicate of the given node to the given predicate node.
  void SetPredicate(Node* node, Node* pred) {
    if (predicated_by_.contains(node)) {
      Node* replaced_predicate = predicated_by_.at(node);
      predicate_of_[replaced_predicate].erase(node);
      if (predicate_of_[replaced_predicate].empty()) {
        predicate_of_.erase(replaced_predicate);
      }
    }
    predicated_by_[node] = pred;
    predicate_of_[pred].insert(node);
  }

  // Get the predicate of the given node.
  std::optional<Node*> GetPredicate(Node* node) const {
    return predicated_by_.contains(node)
               ? std::make_optional(predicated_by_.at(node))
               : std::nullopt;
  }

  // Get all nodes predicated by a given predicate node.
  absl::flat_hash_set<Node*> GetNodesPredicatedBy(Node* node) const {
    return predicate_of_.contains(node) ? predicate_of_.at(node)
                                        : absl::flat_hash_set<Node*>();
  }

  // Assert that the two given predicates are mutually exclusive.
  absl::Status MarkMutuallyExclusive(Node* pred_a, Node* pred_b) {
    XLS_RET_CHECK_NE(pred_a, pred_b);
    XLS_RET_CHECK(FunctionIsOneBit(pred_a));
    XLS_RET_CHECK(FunctionIsOneBit(pred_b));
    mutual_exclusion_[pred_a][pred_b] = true;
    mutual_exclusion_[pred_b][pred_a] = true;
    return absl::OkStatus();
  }

  // Assert that the two given predicates are not mutually exclusive.
  absl::Status MarkNotMutuallyExclusive(Node* pred_a, Node* pred_b) {
    XLS_RET_CHECK_NE(pred_a, pred_b);
    XLS_RET_CHECK(FunctionIsOneBit(pred_a));
    XLS_RET_CHECK(FunctionIsOneBit(pred_b));
    mutual_exclusion_[pred_a][pred_b] = false;
    mutual_exclusion_[pred_b][pred_a] = false;
    return absl::OkStatus();
  }

  // Query whether the two given predicates are known to be mutually exclusive
  // (`true`), known to not be mutually exclusive (`false`), or nothing is known
  // about them (`std::nullopt`).
  //
  // For all `P` and `Q`,
  // `QueryMutuallyExclusive(P, Q) == QueryMutuallyExclusive(Q, P)`.
  std::optional<bool> QueryMutuallyExclusive(Node* pred_a, Node* pred_b) const {
    if (!mutual_exclusion_.contains(pred_a)) {
      return std::nullopt;
    }
    if (!mutual_exclusion_.at(pred_a).contains(pred_b)) {
      return std::nullopt;
    }
    return mutual_exclusion_.at(pred_a).at(pred_b);
  }

  // Returns all neighbors of the given predicate in the mutual exclusion graph.
  // The return value of `MutualExclusionNeighbors(P)` should be all `Q` such
  // that `QueryMutuallyExclusive(P, Q).has_value()`.
  absl::flat_hash_map<Node*, bool> MutualExclusionNeighbors(Node* pred) const {
    return mutual_exclusion_.contains(pred)
               ? mutual_exclusion_.at(pred)
               : absl::flat_hash_map<Node*, bool>();
  }

  // Update the metadata contained within the `Predicates` to respect the
  // replacement of a node by another node.
  void ReplaceNode(Node* original, Node* replacement) {
    if (predicated_by_.contains(original)) {
      Node* predicate = predicated_by_.at(original);
      predicated_by_.erase(original);
      predicated_by_[replacement] = predicate;
      predicate_of_.at(predicate).erase(original);
      predicate_of_.at(predicate).insert(replacement);
    }
    if (predicate_of_.contains(original)) {
      for (Node* node : predicate_of_.at(original)) {
        predicate_of_[replacement].insert(node);
      }
      predicate_of_.erase(original);
    }
    if (mutual_exclusion_.contains(original)) {
      absl::flat_hash_map<Node*, bool> neighbors =
          mutual_exclusion_.at(original);
      mutual_exclusion_.erase(original);
      mutual_exclusion_[replacement] = neighbors;
      for (const auto& [neighbor, boolean] : neighbors) {
        mutual_exclusion_.at(neighbor).erase(original);
        mutual_exclusion_.at(neighbor)[replacement] = boolean;
      }
    }
  }

 private:
  absl::flat_hash_map<Node*, Node*> predicated_by_;
  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>> predicate_of_;

  // The `bool` represents knowledge about the mutual exclusion of two nodes;
  // if it is true then the two nodes are mutually exclusive, if it is false
  // then they are known to not be mutually exclusive.
  //
  // Invariant: if mutual_exclusion_.at(a).contains(b) then
  // mutual_exclusion.at(b).contains(a), i.e.: this is a symmetric relation;
  // this fact is used to make garbage collection efficient.
  //
  // Invariant: all nodes must have Bits type and bitwidth 1
  absl::flat_hash_map<Node*, absl::flat_hash_map<Node*, bool>>
      mutual_exclusion_;
};

// Add a predicate to a node. If the node does not already have a predicate,
// this will simply set the predicate of the node to the given predicate and
// then return the given predicate. Otherwise, this will replace the predicate
// of the node with AND of the given predicate and the existing predicate,
// returning this new predicate.
absl::StatusOr<Node*> AddPredicate(Predicates* p, Node* node, Node* pred) {
  XLS_CHECK_EQ(node->function_base(), pred->function_base());
  FunctionBase* f = node->function_base();

  std::optional<Node*> existing_predicate_maybe = p->GetPredicate(node);

  if (existing_predicate_maybe.has_value()) {
    Node* existing_predicate = existing_predicate_maybe.value();
    XLS_ASSIGN_OR_RETURN(
        Node * pred_and_existing,
        f->MakeNode<NaryOp>(SourceInfo(),
                            std::vector<Node*>({existing_predicate, pred}),
                            Op::kAnd));
    p->SetPredicate(node, pred_and_existing);

    for (const auto [neighbor, boolean] :
         p->MutualExclusionNeighbors(existing_predicate)) {
      if (boolean) {
        XLS_RETURN_IF_ERROR(
            p->MarkMutuallyExclusive(pred_and_existing, neighbor));
      }
    }

    return pred_and_existing;
  }

  p->SetPredicate(node, pred);
  return pred;
}

absl::Status AddSendReceivePredicates(Predicates* p, FunctionBase* f) {
  for (Node* node : f->nodes()) {
    if (node->Is<Send>()) {
      if (std::optional<Node*> pred = node->As<Send>()->predicate()) {
        XLS_RETURN_IF_ERROR(AddPredicate(p, node, pred.value()).status());
      }
    } else if (node->Is<Receive>()) {
      if (std::optional<Node*> pred = node->As<Receive>()->predicate()) {
        XLS_RETURN_IF_ERROR(AddPredicate(p, node, pred.value()).status());
      }
    }
  }
  return absl::OkStatus();
}

template <typename T>
bool HasIntersection(const absl::flat_hash_set<T>& lhs,
                     const absl::flat_hash_set<T>& rhs) {
  const absl::flat_hash_set<T>& smaller = lhs.size() > rhs.size() ? rhs : lhs;
  const absl::flat_hash_set<T>& bigger = lhs.size() > rhs.size() ? lhs : rhs;
  for (const T& element : smaller) {
    if (bigger.contains(element)) {
      return true;
    }
  }
  return false;
}

Z3_lbool RunSolver(Z3_context c, Z3_ast asserted) {
  Z3_solver solver = solvers::z3::CreateSolver(c, 1);
  Z3_solver_assert(c, solver, asserted);
  Z3_lbool satisfiable = Z3_solver_check(c, solver);
  Z3_solver_dec_ref(c, solver);
  return satisfiable;
}

// Returns a list of all predicates in a deterministic order, paired with their
// index in the list.
std::vector<std::pair<Node*, int64_t>> PredicateNodes(Predicates* p,
                                                      FunctionBase* f) {
  std::vector<std::pair<Node*, int64_t>> result;

  int64_t i = 0;
  for (Node* node : f->nodes()) {
    if (!p->GetNodesPredicatedBy(node).empty()) {
      result.push_back({node, i});
      ++i;
    }
  }

  return result;
}

bool IsHeavyOp(Op op) {
  return op == Op::kUMul || op == Op::kSMul || op == Op::kSend ||
         op == Op::kReceive;
}

absl::Status ComputeMutualExclusion(Predicates* p, FunctionBase* f) {
  if (f->IsBlock()) {
    return absl::OkStatus();
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<solvers::z3::IrTranslator> translator,
                       solvers::z3::IrTranslator::CreateAndTranslate(f, true));

  Z3_context ctx = translator->ctx();

  solvers::z3::ScopedErrorHandler seh(ctx);

  std::vector<std::pair<Node*, int64_t>> predicate_nodes = PredicateNodes(p, f);

  Z3_global_param_set("rlimit", "500000");

  // Determine for each predicate whether it is always false using Z3.
  // Dead nodes are mutually exclusive with all other nodes, so this can reduce
  // the runtime  by doing only a linear amount of Z3 calls to remove
  // quadratically many Z3 calls.
  for (const auto& [node, index] : predicate_nodes) {
    Z3_ast translated = translator->GetTranslation(node);
    if (RunSolver(ctx, solvers::z3::BitVectorToBoolean(ctx, translated)) ==
        Z3_L_FALSE) {
      XLS_VLOG(3) << "Proved that " << node << " is always false";
      // A constant false node is mutually exclusive with all other nodes.
      for (const auto& [other, other_index] : predicate_nodes) {
        if (index != other_index) {
          XLS_RETURN_IF_ERROR(p->MarkMutuallyExclusive(node, other));
        }
      }
    }
  }

  for (const auto& [node, index] : predicate_nodes) {
    XLS_VLOG(3) << "Predicate: " << node;
  }

  Z3_global_param_set("rlimit", "500000");

  int64_t known_false = 0;
  int64_t known_true = 0;
  int64_t unknown = 0;

  absl::flat_hash_map<Node*, absl::flat_hash_set<Op>> ops_for_pred;
  for (const auto& [node, index] : predicate_nodes) {
    for (Node* predicated_by : p->GetNodesPredicatedBy(node)) {
      ops_for_pred[node].insert(predicated_by->op());
    }
  }

  {
    std::vector<Node*> irrelevant;
    for (const auto& [pred, ops] : ops_for_pred) {
      if (!std::any_of(ops.begin(), ops.end(), IsHeavyOp)) {
        irrelevant.push_back(pred);
      }
    }
    for (Node* pred : irrelevant) {
      ops_for_pred.erase(pred);
    }
  }

  for (const auto& [node_a, index_a] : predicate_nodes) {
    for (const auto& [node_b, index_b] : predicate_nodes) {
      // This prevents checking `a NAND b` and then later checking `b NAND a`.
      if (index_a >= index_b) {
        continue;
      }

      // Skip this pair if we already know whether they are mutually exclusive.
      if (p->QueryMutuallyExclusive(node_a, node_b).has_value()) {
        continue;
      }

      if (!ops_for_pred.contains(node_a) || !ops_for_pred.contains(node_b) ||
          !HasIntersection(ops_for_pred.at(node_a), ops_for_pred.at(node_b))) {
        continue;
      }

      Z3_ast z3_a = translator->GetTranslation(node_a);
      Z3_ast z3_b = translator->GetTranslation(node_b);

      // We try to find out if `a ∧ b` is satisfiable, which is true iff
      // `a NAND b` is not valid.
      Z3_ast a_and_b =
          solvers::z3::BitVectorToBoolean(ctx, Z3_mk_bvand(ctx, z3_a, z3_b));

      Z3_lbool satisfiable = RunSolver(ctx, a_and_b);

      if (satisfiable == Z3_L_FALSE) {
        known_true += 1;
        XLS_RETURN_IF_ERROR(p->MarkMutuallyExclusive(node_a, node_b));
      } else if (satisfiable == Z3_L_TRUE) {
        known_false += 1;
        XLS_RETURN_IF_ERROR(p->MarkNotMutuallyExclusive(node_a, node_b));
      } else {
        unknown += 1;
        XLS_VLOG(3) << "Z3 ran out of time checking mutual exclusion of "
                    << node_a->GetName() << " and " << node_b->GetName();
      }
    }
  }

  XLS_VLOG(3) << "known_false = " << known_false;
  XLS_VLOG(3) << "known_true  = " << known_true;
  XLS_VLOG(3) << "unknown     = " << unknown;

  XLS_RETURN_IF_ERROR(seh.status());

  return absl::OkStatus();
}

// A map from side-effecting nodes (and AfterAll) to the set of side-effecting
// nodes (/ AfterAll) that their token inputs immediately came from. Note that
// this skips over intermediate movement of tokens through tuples or `identity`.
using TokenDAG = absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>>;

absl::StatusOr<TokenDAG> ComputeTokenDAG(FunctionBase* f) {
  XLS_ASSIGN_OR_RETURN(TokenProvenance provenance, TokenProvenanceAnalysis(f));

  TokenDAG dag;
  for (Node* node : f->nodes()) {
    if (OpIsSideEffecting(node->op()) || node->op() == Op::kAfterAll) {
      for (Node* operand : node->operands()) {
        if (operand->GetType()->IsToken()) {
          Node* child = provenance.at(operand).Get({});
          if (child != nullptr) {
            dag[node].insert(child);
          }
        }
      }
    }
  }

  return dag;
}

using NodeRelation = absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>>;

// Computes a symmetric relation that decides whether two side-effectful nodes
// can be merged. The principle is as follows:
//
// 1. A connected subgraph consisting of all the same kind of effect (sends on
//    all the same channel; receives on all the same channel) of the token DAG
//    can be merged. This is not yet implemented.
// 2. Nodes of the same type that are unrelated in the transitive token
//    dependency relation can be merged.
absl::StatusOr<NodeRelation> ComputeMergableEffects(FunctionBase* f) {
  XLS_ASSIGN_OR_RETURN(TokenDAG token_dag, ComputeTokenDAG(f));
  absl::flat_hash_set<Node*> token_nodes;
  for (const auto& [node, children] : token_dag) {
    token_nodes.insert(node);
    for (const auto& child : children) {
      token_nodes.insert(child);
    }
  }

  auto dfs = [&](Node* root, const NodeRelation& dag) {
    std::vector<Node*> stack;
    stack.push_back(root);
    absl::flat_hash_set<Node*> discovered;
    while (!stack.empty()) {
      Node* popped = stack.back();
      stack.pop_back();
      if (!discovered.contains(popped)) {
        discovered.insert(popped);
        if (dag.contains(popped)) {
          for (Node* child : dag.at(popped)) {
            stack.push_back(child);
          }
        }
      }
    }
    return discovered;
  };

  NodeRelation data_deps;
  for (Node* node : f->nodes()) {
    for (Node* child : node->operands()) {
      data_deps[node].insert(child);
    }
  }

  NodeRelation transitive_closure = TransitiveClosure<Node*>(token_dag);

  // If a receive uses data from another receive in its predicate, they cannot
  // be merged.
  for (Node* effectful_node : token_nodes) {
    if (effectful_node->Is<Receive>()) {
      for (Node* other_node : dfs(effectful_node, data_deps)) {
        if (other_node->Is<Receive>()) {
          transitive_closure[effectful_node].insert(other_node);
        }
      }
    }
  }

  NodeRelation result;

  for (Node* x : token_nodes) {
    for (Node* y : token_nodes) {
      if (!(transitive_closure.contains(x) &&
            transitive_closure.at(x).contains(y)) &&
          !(transitive_closure.contains(y) &&
            transitive_closure.at(y).contains(x))) {
        result[x].insert(y);
        result[y].insert(x);
      }
    }
  }
  return result;
}

// This computes a partition of a subset of all nodes into merge classes.
// Nodes that are not in this partition can be assumed to be in a merge class of
// size 1 including only themselves.
// A merge class is a set of nodes that are all jointly mutually exclusive.
absl::StatusOr<std::vector<absl::flat_hash_set<Node*>>> ComputeMergeClasses(
    Predicates* p, FunctionBase* f) {
  absl::flat_hash_set<Node*> nodes;
  std::vector<Node*> ordered_nodes;
  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>> neighborhoods;

  for (Node* node : TopoSort(f)) {
    if (IsHeavyOp(node->op())) {
      nodes.insert(node);
      neighborhoods[node];
      ordered_nodes.push_back(node);
    }
  }

  XLS_ASSIGN_OR_RETURN(NodeRelation mergable_effects,
                       ComputeMergableEffects(f));
  auto is_mergable = [&](Node* x, Node* y) -> bool {
    return mergable_effects.contains(x) && mergable_effects.at(x).contains(y);
  };

  for (Node* x : nodes) {
    if (!(p->GetPredicate(x).has_value())) {
      continue;
    }
    Node* px = p->GetPredicate(x).value();
    for (Node* y : nodes) {
      if (!(p->GetPredicate(y).has_value())) {
        continue;
      }
      Node* py = p->GetPredicate(y).value();
      if (x->op() != y->op()) {
        continue;
      }
      if ((x->op() == Op::kSend) &&
          (!is_mergable(x, y) ||
           (x->As<Send>()->channel_id() != y->As<Send>()->channel_id()))) {
        continue;
      }
      if ((x->op() == Op::kReceive) &&
          (!is_mergable(x, y) || (x->As<Receive>()->channel_id() !=
                                  y->As<Receive>()->channel_id()))) {
        continue;
      }
      if (p->QueryMutuallyExclusive(px, py) == std::make_optional(true)) {
        neighborhoods[x].insert(y);
        neighborhoods[y].insert(x);
      }
    }
  }

  // The complement of the `neighborhoods` graph
  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>> inverted_neighborhoods;

  for (Node* node : nodes) {
    inverted_neighborhoods[node] = nodes;
    for (Node* neighbor : neighborhoods.at(node)) {
      inverted_neighborhoods[node].erase(neighbor);
    }
  }

  absl::flat_hash_map<Node*, int64_t> node_to_index;
  for (int64_t i = 0; i < ordered_nodes.size(); ++i) {
    node_to_index[ordered_nodes[i]] = i;
  }

  std::vector<int64_t> iota(ordered_nodes.size());
  std::iota(iota.begin(), iota.end(), 0);
  std::vector<absl::flat_hash_set<int64_t>> coloring_indices =
      RecursiveLargestFirstColoring<int64_t>(
          absl::flat_hash_set<int64_t>(iota.begin(), iota.end()),
          [&](int64_t node_index) -> absl::flat_hash_set<int64_t> {
            absl::flat_hash_set<Node*> inv_neighbors =
                inverted_neighborhoods.at(ordered_nodes.at(node_index));
            absl::flat_hash_set<int64_t> result;
            for (Node* inv_neighbor : inv_neighbors) {
              result.insert(node_to_index.at(inv_neighbor));
            }
            return result;
          });

  std::vector<absl::flat_hash_set<Node*>> coloring;
  for (const absl::flat_hash_set<int64_t>& color_class : coloring_indices) {
    absl::flat_hash_set<Node*> color_node_class;
    for (int64_t index : color_class) {
      color_node_class.insert(ordered_nodes[index]);
    }
    coloring.push_back(color_node_class);
  }

  for (const absl::flat_hash_set<Node*>& color_class : coloring) {
    XLS_CHECK(!color_class.empty());
    Op op = (*(color_class.cbegin()))->op();
    for (Node* node : color_class) {
      XLS_CHECK_EQ(node->op(), op);
    }
  }

  return coloring;
}

// Given a sequence of nodes, returns a new sequence of nodes that comprises the
// predicates of each of the input nodes. If an input node does not have a
// predicate, a literal `true` (1-bit value containing 1) node is used.
absl::StatusOr<std::vector<Node*>> PredicateVectorFromNodes(
    Predicates* p, FunctionBase* f, absl::Span<Node* const> nodes) {
  XLS_ASSIGN_OR_RETURN(Node * literal_true,
                       f->MakeNode<Literal>(SourceInfo(), Value(UBits(1, 1))));

  std::vector<Node*> predicates;
  predicates.reserve(nodes.size());
  for (Node* node : nodes) {
    if (std::optional<Node*> pred = p->GetPredicate(node)) {
      predicates.push_back(pred.value());
    } else {
      predicates.push_back(literal_true);
    }
  }

  return predicates;
}

// Get the token produced by the given receive node. This avoids creating a new
// `tuple_index` node if one already exists.
absl::StatusOr<Node*> GetTokenOfReceive(Node* receive) {
  FunctionBase* f = receive->function_base();
  for (Node* user : receive->users()) {
    if (user->Is<TupleIndex>()) {
      if (user->As<TupleIndex>()->index() == 0) {
        return user;
      }
    }
  }
  return f->MakeNode<TupleIndex>(SourceInfo(), receive, 0);
}

// Given a set of nodes, returns all nodes in the token dag that feed into this
// set but are not contained within it. The ordering of the result is guaranteed
// to be deterministic.
absl::StatusOr<std::vector<Node*>> ComputeTokenInputs(
    FunctionBase* f, absl::Span<Node* const> nodes) {
  XLS_ASSIGN_OR_RETURN(TokenDAG token_dag, ComputeTokenDAG(f));

  absl::flat_hash_set<Node*> token_inputs_unsorted;

  {
    absl::flat_hash_set<Node*> nodes_set(nodes.begin(), nodes.end());

    for (Node* node : nodes) {
      if (token_dag.contains(node)) {
        absl::flat_hash_set<Node*> children = token_dag.at(node);
        for (Node* child : children) {
          if (!nodes_set.contains(child)) {
            Node* token = child;
            if (child->Is<Receive>()) {
              XLS_ASSIGN_OR_RETURN(token, GetTokenOfReceive(child));
            }
            token_inputs_unsorted.insert(token);
          }
        }
      }
    }
  }

  // Ensure determinism of output.
  return SetToSortedVector(token_inputs_unsorted);
}

absl::StatusOr<bool> MergeSends(Predicates* p, FunctionBase* f,
                                absl::Span<Node* const> to_merge) {
  int64_t channel_id = to_merge.front()->As<Send>()->channel_id();
  for (Node* send : to_merge) {
    XLS_CHECK_EQ(channel_id, send->As<Send>()->channel_id());
  }

  XLS_ASSIGN_OR_RETURN(std::vector<Node*> token_inputs,
                       ComputeTokenInputs(f, to_merge));

  XLS_ASSIGN_OR_RETURN(Node * token,
                       f->MakeNode<AfterAll>(SourceInfo(), token_inputs));

  XLS_ASSIGN_OR_RETURN(std::vector<Node*> predicates,
                       PredicateVectorFromNodes(p, f, to_merge));

  XLS_ASSIGN_OR_RETURN(Node * selector,
                       f->MakeNode<Concat>(SourceInfo(), predicates));

  XLS_ASSIGN_OR_RETURN(
      Node * predicate,
      f->MakeNode<BitwiseReductionOp>(SourceInfo(), selector, Op::kOrReduce));

  std::vector<Node*> args;
  args.reserve(to_merge.size());
  for (Node* node : to_merge) {
    args.push_back(node->As<Send>()->data());
  }
  // OneHotSelect takes the cases in reverse order, confusingly
  std::reverse(args.begin(), args.end());

  XLS_ASSIGN_OR_RETURN(Node * data,
                       f->MakeNode<OneHotSelect>(SourceInfo(), selector, args));

  XLS_ASSIGN_OR_RETURN(Node * send, f->MakeNode<Send>(SourceInfo(), token, data,
                                                      predicate, channel_id));

  for (Node* node : to_merge) {
    XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(send));
    XLS_RETURN_IF_ERROR(f->RemoveNode(node));
  }

  return true;
}

absl::StatusOr<bool> MergeReceives(Predicates* p, FunctionBase* f,
                                   absl::Span<Node* const> to_merge) {
  int64_t channel_id = to_merge.front()->As<Receive>()->channel_id();
  bool is_blocking = to_merge.front()->As<Receive>()->is_blocking();

  for (Node* send : to_merge) {
    XLS_CHECK_EQ(channel_id, send->As<Receive>()->channel_id());
    XLS_CHECK_EQ(is_blocking, send->As<Receive>()->is_blocking());
  }

  XLS_ASSIGN_OR_RETURN(std::vector<Node*> token_inputs,
                       ComputeTokenInputs(f, to_merge));

  XLS_ASSIGN_OR_RETURN(Node * token,
                       f->MakeNode<AfterAll>(SourceInfo(), token_inputs));

  XLS_ASSIGN_OR_RETURN(std::vector<Node*> predicates,
                       PredicateVectorFromNodes(p, f, to_merge));

  XLS_ASSIGN_OR_RETURN(Node * predicate,
                       f->MakeNode<NaryOp>(SourceInfo(), predicates, Op::kOr));

  XLS_ASSIGN_OR_RETURN(Node * receive,
                       f->MakeNode<Receive>(SourceInfo(), token, predicate,
                                            channel_id, is_blocking));

  XLS_ASSIGN_OR_RETURN(Node * token_output,
                       f->MakeNode<TupleIndex>(SourceInfo(), receive, 0));

  XLS_ASSIGN_OR_RETURN(Node * value_output,
                       f->MakeNode<TupleIndex>(SourceInfo(), receive, 1));

  XLS_ASSIGN_OR_RETURN(
      Node * zero,
      f->MakeNode<Literal>(SourceInfo(), ZeroOfType(value_output->GetType())));

  std::vector<Node*> gated_output;
  for (int64_t i = 0; i < to_merge.size(); ++i) {
    XLS_ASSIGN_OR_RETURN(
        Node * gated,
        f->MakeNode<Select>(SourceInfo(), predicates[i],
                            std::vector<Node*>{zero, value_output},
                            std::nullopt));
    XLS_ASSIGN_OR_RETURN(
        Node * gated_with_token,
        f->MakeNode<Tuple>(SourceInfo(),
                           std::vector<Node*>{token_output, gated}));
    gated_output.push_back(gated_with_token);
  }

  for (int64_t i = 0; i < to_merge.size(); ++i) {
    XLS_RETURN_IF_ERROR(to_merge[i]->ReplaceUsesWith(gated_output[i]));
    XLS_RETURN_IF_ERROR(f->RemoveNode(to_merge[i]));
  }

  return true;
}

absl::StatusOr<bool> MergeNodes(Predicates* p, FunctionBase* f,
                                const absl::flat_hash_set<Node*>& merge_class) {
  if (merge_class.size() <= 1) {
    return false;
  }

  std::vector<Node*> to_merge = SetToSortedVector(merge_class);

  Op op = to_merge.front()->op();
  if (op == Op::kSend) {
    return MergeSends(p, f, to_merge);
  }
  if (op == Op::kReceive) {
    return MergeReceives(p, f, to_merge);
  }

  return false;
}

}  // namespace

absl::StatusOr<bool> MutualExclusionPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const PassOptions& options, PassResults* results) const {
  Predicates p;
  XLS_RETURN_IF_ERROR(AddSendReceivePredicates(&p, f));
  XLS_RETURN_IF_ERROR(ComputeMutualExclusion(&p, f));
  XLS_ASSIGN_OR_RETURN(std::vector<absl::flat_hash_set<Node*>> merge_classes,
                       ComputeMergeClasses(&p, f));

  if (XLS_VLOG_IS_ON(3)) {
    for (const absl::flat_hash_set<Node*>& merge_class : merge_classes) {
      if (merge_class.size() <= 1) {
        continue;
      }
      Op op = (*(merge_class.cbegin()))->op();
      std::vector<std::string> name_pred_pairs;
      for (Node* node : merge_class) {
        XLS_CHECK_EQ(node->op(), op);
        name_pred_pairs.push_back(
            absl::StrFormat("%s [%s]", node->GetName(),
                            p.GetPredicate(node).value()->GetName()));
      }
      XLS_VLOG(3) << "Merge class: " << merge_class.size() << ", op = " << op;
      XLS_VLOG(3) << "    " << absl::StrJoin(name_pred_pairs, ", ");
    }
  }

  XLS_VLOG(3) << "Successfully computed mutual exclusion for " << f->name();

  bool changed = false;
  for (const absl::flat_hash_set<Node*>& merge_class : merge_classes) {
    XLS_ASSIGN_OR_RETURN(bool subpass_changed, MergeNodes(&p, f, merge_class));
    changed |= subpass_changed;
  }
  return changed;
}

}  // namespace xls
