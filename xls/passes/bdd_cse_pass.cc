// Copyright 2020 The XLS Authors
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

#include "xls/passes/bdd_cse_pass.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/predicate_state.h"
#include "xls/passes/query_engine.h"

namespace xls {

namespace {

// Returns the order in which to visit the nodes when performing the
// optimization. If a pair of equivalent nodes is found during the optimization
// then the earlier visited node replaces the later visited node so this order
// is constructed with the following properties:
//
// (1) Order is a topological sort. This is necessary to avoid introducing
//     cycles in the graph.
//
// (2) Critical-path delay through the graph to the node increases monotonically
//     in the list. This ensures that the CSE replacement does not increase
//     critical-path
//
absl::StatusOr<std::vector<Node*>> GetNodeOrder(FunctionBase* f,
                                                OptimizationContext& context) {
  // Index of each node in the topological sort.
  absl::flat_hash_map<Node*, int64_t> topo_index;
  // Critical-path distance from root in the graph to each node.
  absl::flat_hash_map<Node*, int64_t> node_cp_delay;
  int64_t i = 0;

  // Return an estimate of the delay of the given node. Because BDD-CSE may be
  // run at any point in the pipeline, some nodes with no delay model may be
  // present (these would be eliminated before codegen) so return zero for these
  // cases.
  // TODO(meheff): Replace with the actual model being used when the delay model
  // is threaded through the pass pipeline.
  auto get_node_delay = [&](Node* n) {
    absl::StatusOr<int64_t> delay_status =
        GetStandardDelayEstimator().GetOperationDelayInPs(n);
    return delay_status.ok() ? delay_status.value() : 0;
  };
  XLS_ASSIGN_OR_RETURN(std::vector<Node*> topo_sort_nodes, context.TopoSort(f));
  for (Node* node : topo_sort_nodes) {
    topo_index[node] = i;
    int64_t node_start = 0;
    for (Node* operand : node->operands()) {
      node_start = std::max(
          node_start, node_cp_delay.at(operand) + get_node_delay(operand));
    }
    node_cp_delay[node] = node_start + get_node_delay(node);
    ++i;
  }
  std::vector<Node*> nodes(f->nodes().begin(), f->nodes().end());
  std::sort(nodes.begin(), nodes.end(), [&](Node* a, Node* b) {
    return (node_cp_delay.at(a) < node_cp_delay.at(b) ||
            (node_cp_delay.at(a) == node_cp_delay.at(b) &&
             topo_index.at(a) < topo_index.at(b)));
  });
  // The node order must be a topological sort in order to avoid introducing
  // cycles in the graph.
  for (Node* node : nodes) {
    for (Node* operand : node->operands()) {
      XLS_RET_CHECK(topo_index.at(operand) < topo_index.at(node));
    }
  }
  return nodes;
}

// Returns whether bits nodes `a` and `b` are known equal under the query engine
// `qe` (which may already encode a guarding assumption on a select arm).
bool ValueEqualUnder(const QueryEngine& qe, Node* a, Node* b) {
  if (!a->GetType()->IsBits() || !b->GetType()->IsBits() ||
      a->BitCountOrDie() != b->BitCountOrDie()) {
    return false;
  }
  for (int64_t i = 0; i < a->BitCountOrDie(); ++i) {
    if (!qe.KnownEquals(TreeBitLocation(a, i), TreeBitLocation(b, i))) {
      return false;
    }
  }
  return true;
}

// Proves `a == b` under the per-bit guard `lhs[i] == rhs[i]` for every `i`.
bool ValueEqualUnderPerBitEq(const BddQueryEngine& qe, Node* a, Node* b,
                             Node* lhs, Node* rhs) {
  if (!a->GetType()->IsBits() || !b->GetType()->IsBits() ||
      a->BitCountOrDie() != b->BitCountOrDie() || !lhs->GetType()->IsBits() ||
      !rhs->GetType()->IsBits() ||
      lhs->BitCountOrDie() != rhs->BitCountOrDie()) {
    return false;
  }
  const int64_t width = a->BitCountOrDie();
  for (int64_t i = 0; i < width; ++i) {
    // Assumption: lhs[i] == rhs[i]  <=>  Xnor(lhs[i], rhs[i]).
    std::optional<BddNodeIndex> lhs_bit =
        qe.GetBddNode(TreeBitLocation(lhs, i));
    std::optional<BddNodeIndex> rhs_bit =
        qe.GetBddNode(TreeBitLocation(rhs, i));
    if (!lhs_bit.has_value() || !rhs_bit.has_value()) {
      return false;
    }
    // Xnor(x, y) = (x && y) || (!x && !y).
    BddNodeIndex assumption = qe.bdd().Or(
        qe.bdd().And(*lhs_bit, *rhs_bit),
        qe.bdd().And(qe.bdd().Not(*lhs_bit), qe.bdd().Not(*rhs_bit)));
    if (!qe.KnownEquals(TreeBitLocation(a, i), TreeBitLocation(b, i),
                        assumption)) {
      return false;
    }
  }
  return true;
}

// Maximum number of preceding nodes scanned as guarded-CSE candidates for a
// given arm value.
static constexpr int64_t kGuardedMaxCandidatesToScan = 16;

// Maximum bit width of `Eq(lhs, rhs)` guard operands handled by the exact full
// specialization path (`SpecializeGivenPredicate`). Above this width the full
// `eq(lhs, rhs)` BDD can saturate the engine's default path limit, so wider
// guards are rewritten via the per-bit weak surrogate instead.
static constexpr int64_t kFullSpecEqMaxWidth = 4;

// True if `arm` of a 1-bit `Eq`/`Ne`-selected `sel` is live exactly when the
// comparator operands are equal (`selector == 1` for `Eq` arm 1, `selector ==
// 0` for `Ne` arm 0).
bool ArmIsLiveWhenOperandsEqual(Select* sel, const PredicateState& state) {
  if (state.IsDefaultArm() || sel->cases().size() != 2) {
    return false;
  }
  Node* selector = state.selector();
  if (selector == nullptr || !selector->GetType()->IsBits() ||
      selector->BitCountOrDie() != 1 || !selector->Is<CompareOp>()) {
    return false;
  }
  Op compare_op = selector->As<CompareOp>()->op();
  int64_t arm_id = state.arm_index();
  return (compare_op == Op::kEq && arm_id == 1) ||
         (compare_op == Op::kNe && arm_id == 0);
}

// True if the guarded rewrite should use the per-bit weak surrogate: a 1-bit
// `Eq`/`Ne`-selected arm (see `ArmIsLiveWhenOperandsEqual`) wide enough that
// materializing the full `eq` BDD would saturate the path limit.
bool ShouldUsePerBitEq(Select* sel, const PredicateState& state) {
  Node* selector = state.selector();
  if (selector->operands().size() != 2) {
    return false;
  }
  Node* lhs = selector->operand(0);
  Node* rhs = selector->operand(1);
  return ArmIsLiveWhenOperandsEqual(sel, state) && lhs->GetType()->IsBits() &&
         rhs->GetType()->IsBits() &&
         lhs->BitCountOrDie() == rhs->BitCountOrDie() &&
         lhs->BitCountOrDie() > kFullSpecEqMaxWidth;
}

// Rewrites one guarded select-arm operand edge: `arm_value` (value of arm
// `arm` of `sel`, position `operand_number`) to another node within the last
// kGuardedMaxCandidatesToScan nodes of `node_order` that is BDD-equal under
// `arm`'s guard; returns true if the edge is rewired.
//
// Scan only a small window of preceding nodes (see kGuardedMaxCandidatesToScan)
// rather than the whole graph. Anchoring the window at `arm_value` keeps the
// replacement no later in the schedule and cycle-safe.
absl::StatusOr<bool> MaybeReplaceGuardedSelectArm(
    BddQueryEngine* guarded_engine, absl::Span<Node* const> node_order,
    int64_t arm_value_pos, Select* sel, int64_t operand_number, Node* arm_value,
    const PredicateState& state) {
  if (arm_value == nullptr || arm_value->Is<Literal>() ||
      !arm_value->GetType()->IsBits()) {
    return false;
  }
  // For a 1-bit `Eq`/`Ne` select guard pick between two sound rewrite
  // strategies:
  //  * Narrow operands (width <= kFullSpecEqMaxWidth): materialize the whole
  //    `eq(lhs, rhs)` BDD once and compare arms by canonical BDD index. Exact
  //    and cheap; the path width-1 guards take.
  //  * Wide operands (width > kFullSpecEqMaxWidth): the full `eq` BDD would
  //    saturate the shared engine's 1024 path limit, so prove each bit under
  //    the tiny single-bit surrogate `lhs[i] == rhs[i]`
  //    (`ValueEqualUnderPerBitEq`).
  //
  // The per-bit surrogate assumes `lhs == rhs` and is sound only where that is
  // the arm's live predicate (`ArmIsLiveWhenOperandsEqual`); elsewhere it would
  // be a false positive, so those arms fall back to the exact path.
  int64_t window_start =
      std::max(int64_t{0}, arm_value_pos - kGuardedMaxCandidatesToScan);
  if (!ShouldUsePerBitEq(sel, state)) {
    std::unique_ptr<QueryEngine> assumed =
        guarded_engine->SpecializeGivenPredicate({state});
    for (int64_t ci = window_start; ci < arm_value_pos; ++ci) {
      Node* cand = node_order[ci];
      if (cand == arm_value) {
        continue;
      }
      if (ValueEqualUnder(*assumed, arm_value, cand)) {
        VLOG(4) << "Guarded cond-value-prop: " << sel->GetName()
                << " arm operand == " << cand->GetName();
        XLS_RETURN_IF_ERROR(sel->ReplaceOperandNumber(operand_number, cand));
        return true;
      }
    }
    return false;
  }
  Node* eq_lhs = state.selector()->operand(0);
  Node* eq_rhs = state.selector()->operand(1);
  for (int64_t ci = window_start; ci < arm_value_pos; ++ci) {
    Node* cand = node_order[ci];
    if (cand == arm_value) {
      continue;
    }
    if (ValueEqualUnderPerBitEq(*guarded_engine, arm_value, cand, eq_lhs,
                                eq_rhs)) {
      VLOG(4) << "Guarded cond-value-prop (per-bit eq): " << sel->GetName()
              << " arm operand == " << cand->GetName();
      XLS_RETURN_IF_ERROR(sel->ReplaceOperandNumber(operand_number, cand));
      return true;
    }
  }
  return false;
}

// Width gates for guarded value propagation. A wide selector or data path can
// explode BDD re-derivation even at a bounded path limit, so skip anything
// wider than these.
static constexpr int64_t kGuardedMaxSelectorWidth = 64;
static constexpr int64_t kGuardedMaxValueWidth = 64;

// Tries to rewrite every arm of a plain `Select`.
absl::StatusOr<bool> TryGuardedArmRewrite(
    BddQueryEngine* guarded_engine, absl::Span<Node* const> node_order,
    const absl::flat_hash_map<Node*, int64_t>& node_pos, Select* sel) {
  if (!sel->GetType()->IsBits()) {
    return false;
  }
  if (!sel->selector()->GetType()->IsBits() ||
      sel->selector()->BitCountOrDie() > kGuardedMaxSelectorWidth ||
      sel->BitCountOrDie() > kGuardedMaxValueWidth) {
    return false;
  }

  auto guard_one_arm =
      [&](Node* arm_value, int64_t operand_number,
          const PredicateState& state) -> absl::StatusOr<bool> {
    if (!node_pos.contains(arm_value)) {
      return false;  // Arm not in node_order (e.g. a shared default); skip.
    }
    return MaybeReplaceGuardedSelectArm(guarded_engine, node_order,
                                        node_pos.at(arm_value), sel,
                                        operand_number, arm_value, state);
  };

  bool sel_changed = false;
  for (int64_t i = 0; i < sel->cases().size(); ++i) {
    XLS_ASSIGN_OR_RETURN(
        bool arm_changed,
        guard_one_arm(sel->get_case(i), /*operand_number=*/i + 1,
                      PredicateState(sel, i)));
    sel_changed = sel_changed || arm_changed;
  }
  if (sel->default_value().has_value()) {
    XLS_ASSIGN_OR_RETURN(
        bool arm_changed,
        guard_one_arm(*sel->default_value(),
                      /*operand_number=*/sel->operand_count() - 1,
                      PredicateState(sel, PredicateState::kDefaultArm)));
    sel_changed = sel_changed || arm_changed;
  }
  return sel_changed;
}

// Runs guarded CSE over every `Select` in `node_order` (in order) and returns
// whether any arm was rewritten.
absl::StatusOr<bool> RunGuardedSelectCse(
    BddQueryEngine* guarded_engine, absl::Span<Node* const> node_order,
    const absl::flat_hash_map<Node*, int64_t>& node_pos) {
  bool guarded_changed = false;
  for (Node* node : node_order) {
    if (!node->Is<Select>()) {
      continue;
    }
    Select* sel = node->As<Select>();
    XLS_ASSIGN_OR_RETURN(
        bool sel_changed,
        TryGuardedArmRewrite(guarded_engine, node_order, node_pos, sel));
    guarded_changed = guarded_changed || sel_changed;
  }
  return guarded_changed;
}

}  // namespace

absl::StatusOr<bool> BddCsePass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  BddQueryEngine* query_engine = context.SharedQueryEngine<BddQueryEngine>(f);
  auto get_bdd_node = [&](Node* n, int64_t bit_index) -> int64_t {
    return query_engine->GetBddNode(TreeBitLocation(n, bit_index))->value();
  };

  // To improve efficiency, bucket potentially common nodes together. The
  // bucketing is done via a int64_t hash value of the BDD node indices of each
  // bit of the node.
  auto hasher = absl::Hash<std::vector<int64_t>>();
  auto node_hash = [&](Node* n) {
    CHECK(n->GetType()->IsBits());
    std::vector<int64_t> values_to_hash;
    values_to_hash.reserve(n->BitCountOrDie());
    for (int64_t i = 0; i < n->BitCountOrDie(); ++i) {
      values_to_hash.push_back(get_bdd_node(n, i));
    }
    return hasher(values_to_hash);
  };

  auto is_same_value = [&](Node* a, Node* b) {
    if (a->BitCountOrDie() != b->BitCountOrDie()) {
      return false;
    }
    for (int64_t i = 0; i < a->BitCountOrDie(); ++i) {
      if (get_bdd_node(a, i) != get_bdd_node(b, i)) {
        return false;
      }
    }
    return true;
  };

  bool changed = false;
  absl::flat_hash_map<int64_t, std::vector<Node*>> node_buckets;
  node_buckets.reserve(f->node_count());
  XLS_ASSIGN_OR_RETURN(std::vector<Node*> node_order, GetNodeOrder(f, context));
  for (Node* node : node_order) {
    if (!node->GetType()->IsBits() || node->Is<Literal>()) {
      continue;
    }

    int64_t hash = node_hash(node);
    if (!node_buckets.contains(hash)) {
      node_buckets[hash].push_back(node);
      continue;
    }
    bool replaced = false;
    for (Node* candidate : node_buckets.at(hash)) {
      if (is_same_value(node, candidate)) {
        XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(candidate));
        VLOG(4) << "Found identical value:";
        VLOG(4) << "  Node: " << node->ToString();
        VLOG(4) << "  Replacement: " << candidate->ToString();
        changed = true;
        replaced = true;
        break;
      }
    }
    if (!replaced) {
      node_buckets[hash].push_back(node);
    }
  }

  // Guarded (conditional) value propagation over select arms: like classic CSE
  // above but rewrites an arm operand to a node that is BDD-equal to it only
  // under that arm's guarding predicate. The rewrite is per-edge (only the
  // guarded arm operand changes), so it stays sound even if the arm value has
  // other, unguarded uses.
  //
  // The guard predicate can grow exponentially in BDD size and saturate the
  // shared engine's default path limit; this causes false negatives, never
  // false positives.
  absl::flat_hash_map<Node*, int64_t> node_pos;
  node_pos.reserve(node_order.size());
  for (int64_t i = 0; i < node_order.size(); ++i) {
    node_pos[node_order[i]] = i;
  }
  XLS_ASSIGN_OR_RETURN(bool guard_changed,
                       RunGuardedSelectCse(query_engine, node_order, node_pos));
  changed = changed || guard_changed;

  return changed;
}

}  // namespace xls
