// Copyright 2023 The XLS Authors
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

#include "xls/passes/context_sensitive_range_query_engine.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function_base.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_ops.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/ternary.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"
#include "xls/passes/back_propagate_range_analysis.h"
#include "xls/passes/node_dependency_analysis.h"
#include "xls/passes/predicate_state.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/range_query_engine.h"

namespace xls {

namespace {

// Class which identifies which nodes BackPropagator might be able to extract
// more precise bounds for.
//
// TODO(allight): This should maybe belong in the
// back_propagate_range_analysis.h
class InterestingNodeFinder final : public DfsVisitorWithDefault {
 public:
  explicit InterestingNodeFinder(const RangeQueryEngine& base) : base_(base) {}
  static absl::StatusOr<std::vector<Node*>> Execute(
      const RangeQueryEngine& base, Node* cond) {
    InterestingNodeFinder inf(base);
    inf.interesting_.push_back(cond);
    XLS_RETURN_IF_ERROR(cond->VisitSingleNode(&inf));
    return std::move(inf.interesting_);
  }

  absl::Status DefaultHandler(Node* node) final { return absl::OkStatus(); }
  absl::Status HandleNe(CompareOp* ne) final { return UnifyExactMatch(ne); }
  absl::Status HandleEq(CompareOp* eq) final { return UnifyExactMatch(eq); }
  absl::Status HandleSGe(CompareOp* cmp) final { return UnifyComparison(cmp); }
  absl::Status HandleSGt(CompareOp* cmp) final { return UnifyComparison(cmp); }
  absl::Status HandleSLe(CompareOp* cmp) final { return UnifyComparison(cmp); }
  absl::Status HandleSLt(CompareOp* cmp) final { return UnifyComparison(cmp); }
  absl::Status HandleUGe(CompareOp* cmp) final { return UnifyComparison(cmp); }
  absl::Status HandleUGt(CompareOp* cmp) final { return UnifyComparison(cmp); }
  absl::Status HandleULe(CompareOp* cmp) final { return UnifyComparison(cmp); }
  absl::Status HandleULt(CompareOp* cmp) final { return UnifyComparison(cmp); }
  absl::Status HandleNaryAnd(NaryOp* and_op) final {
    if (and_op->operand_count() == 2) {
      interesting_.push_back(and_op->operand(0));
      interesting_.push_back(and_op->operand(1));
      AddImpreciseOperands(and_op->operand(0));
      AddImpreciseOperands(and_op->operand(1));
    }
    return absl::OkStatus();
  }

 private:
  absl::Status UnifyComparison(CompareOp* cmp) {
    AddImpreciseOperands(cmp);
    return absl::OkStatus();
  }
  absl::Status UnifyExactMatch(CompareOp* eq) {
    AddImpreciseOperands(eq);
    return absl::OkStatus();
  }

  void AddImpreciseOperands(Node* n) {
    for (auto* op : n->operands()) {
      if (absl::c_any_of(
              base_.GetIntervals(op).elements(),
              [](const IntervalSet& is) { return !is.IsPrecise(); })) {
        interesting_.push_back(op);
      }
    }
  }

  const RangeQueryEngine& base_;
  std::vector<Node*> interesting_;
};

// Class to hold givens extracted from select context.
//
// This also restricts the range analysis to end at the select we are
// specializing on since nodes below it can only be specialized to this select
// if we moved them into the selects branches. This sort of transform is not one
// we currently perform.
class ContextGivens final : public RangeDataProvider {
 public:
  ContextGivens(const std::vector<Node*>& topo_sort, const Node* finish,
                const absl::flat_hash_map<Node*, RangeData>& data,
                std::function<std::optional<RangeData>(Node*)> memoized_data)
      : topo_sort_(topo_sort),
        finish_(finish),
        data_(data),
        memoized_data_(std::move(memoized_data)) {}

  std::optional<RangeData> GetKnownIntervals(Node* node) final {
    if (data_.contains(node)) {
      return data_.at(node);
    }
    return memoized_data_(node);
  }

  absl::Status IterateFunction(DfsVisitor* visitor) final {
    for (Node* n : topo_sort_) {
      if (n == finish_) {
        break;
      }
      XLS_RETURN_IF_ERROR(n->VisitSingleNode(visitor));
    }
    return absl::OkStatus();
  }

 private:
  const std::vector<Node*>& topo_sort_;
  const Node* finish_;
  const absl::flat_hash_map<Node*, RangeData> data_;
  std::function<std::optional<RangeData>(Node*)> memoized_data_;
};

// A pair of selector values and selected arms.
struct SelectorAndArm {
  // The node which is the selector in some selects.
  Node* selector;
  // The arm value we assign the selector.
  PredicateState::ArmT arm;

  friend bool operator==(const SelectorAndArm& x, const SelectorAndArm& y) {
    return (x.selector == y.selector) && (x.arm == y.arm);
  }

  template <typename H>
  friend H AbslHashValue(H h, const SelectorAndArm& s) {
    return H::combine(std::move(h), s.selector, s.arm);
  }
};

struct EquivalenceSet {
  std::vector<PredicateState> equivalent_states;
  InlineBitmap interesting_nodes;
};

// Helper to perform the actual analysis and hold together all data needed.
// This is used to fill in the fields of the actual query engine and therefore
// does not own the arena/map that it fills in.
class Analysis {
 public:
  struct InterestingStatesAndNodeList {
    absl::flat_hash_map<Node*, int64_t> node_indices;
    std::vector<std::pair<PredicateState, InlineBitmap>> state_and_nodes;
  };
  Analysis(
      RangeQueryEngine& base_range,
      std::vector<std::unique_ptr<const RangeQueryEngine>>& arena,
      absl::flat_hash_map<PredicateState, const RangeQueryEngine*>& engines)
      : base_range_(base_range), arena_(arena), engines_(engines) {}

  absl::StatusOr<ReachedFixpoint> Execute(FunctionBase* f) {
    // Get the topological sort once so we don't recalculate it each time.
    topo_sort_ = TopoSort(f);
    // Get the base case.
    absl::flat_hash_map<Node*, RangeData> empty;
    ContextGivens base_givens(
        topo_sort_, /*finish=*/nullptr,
        /* data=*/empty,
        [](auto n) -> std::optional<RangeData> { return std::nullopt; });
    XLS_RETURN_IF_ERROR(base_range_.PopulateWithGivens(base_givens).status());

    // Get every possible one-hot state.

    std::vector<PredicateState> all_states;
    // Iterate in same order we walk.
    for (Node* n : topo_sort_) {
      if (n->Is<Select>()) {
        for (int64_t idx = 0; idx < n->As<Select>()->cases().size(); ++idx) {
          all_states.push_back(PredicateState(n->As<Select>(), idx));
        }
        if (n->As<Select>()->default_value().has_value()) {
          all_states.push_back(
              PredicateState(n->As<Select>(), PredicateState::kDefaultArm));
        }
      }
    }
    XLS_ASSIGN_OR_RETURN(auto interesting,
                         FilterUninterestingStates(f, all_states));
    // Bucket states into equivalence classes. Any predicate-states where the
    // arm and selector are identical
    absl::flat_hash_map<SelectorAndArm, EquivalenceSet> equivalences;
    equivalences.reserve(interesting.state_and_nodes.size());
    for (auto [state, interesting_nodes] : interesting.state_and_nodes) {
      EquivalenceSet& cur =
          equivalences
              .try_emplace(
                  SelectorAndArm{
                      .selector = state.node()->As<Select>()->selector(),
                      .arm = state.arm()},
                  EquivalenceSet{
                      .equivalent_states = {},
                      .interesting_nodes = InlineBitmap(f->node_count())})
              .first->second;
      cur.equivalent_states.push_back(state);
      cur.interesting_nodes.Union(interesting_nodes);
    }
    // NB We don't care what order we examine each equivalence (since all are
    // disjoint).
    for (const auto& [_, states] : equivalences) {
      // Since the all_states_ is in topo the last equiv state is usable for
      // everything.
      // We don't care what order we calculate the equivalences because each is
      // fully disjoint from one another as we consider only a single condition
      // to be true at a time.
      XLS_ASSIGN_OR_RETURN(auto tmp,
                           CalculateRangeGiven(states.equivalent_states.back(),
                                               states.interesting_nodes,
                                               interesting.node_indices));
      auto result =
          arena_
              .emplace_back(std::make_unique<RangeQueryEngine>(std::move(tmp)))
              .get();
      for (const PredicateState& ps : states.equivalent_states) {
        engines_[ps] = result;
      }
    }
    return ReachedFixpoint::Changed;
  }

 private:
  // Get rid of any predicate-states that we can statically tell won't affect
  // anything. A predicate state where the values that impact the selector don't
  // impact the selected value in any meaningful way will not show any
  // differences in the calculated ranges so no need to calculate them at all.
  absl::StatusOr<InterestingStatesAndNodeList> FilterUninterestingStates(
      FunctionBase* f, const std::vector<PredicateState>& states) {
    std::vector<Node*> select_nodes;
    std::vector<Node*> selectee_nodes;
    // Calculate all nodes which depend on or are depended on by either selector
    // values or the selected values.
    //
    // We do this in one pass so find all the interesting nodes first.
    for (const PredicateState& s : states) {
      XLS_ASSIGN_OR_RETURN(
          std::vector<Node*> interesting,
          InterestingNodeFinder::Execute(base_range_, s.selector()));
      absl::c_copy(interesting, std::back_inserter(select_nodes));
      selectee_nodes.push_back(s.value());
    }
    NodeDependencyAnalysis forward_interesting(
        NodeDependencyAnalysis::ForwardDependents(f, select_nodes));
    NodeDependencyAnalysis backwards_interesting(
        NodeDependencyAnalysis::BackwardDependents(f, selectee_nodes));
    std::vector<std::pair<PredicateState, InlineBitmap>> interesting_states;
    interesting_states.reserve(states.size());
    for (const PredicateState& ps : states) {
      // If there's any node which is both an input into the select value and
      // affected by something the conditional specialization can discover we
      // consider it interesting.
      InlineBitmap forward_bm(f->node_count(), false);
      // What nodes do we care about for this specific run. Since this basically
      // only depends on the input node no need to memoize it.
      XLS_ASSIGN_OR_RETURN(
          std::vector<Node*> interesting,
          InterestingNodeFinder::Execute(base_range_, ps.selector()));
      for (Node* n : interesting) {
        XLS_ASSIGN_OR_RETURN(auto deps, forward_interesting.GetDependents(n));
        forward_bm.Union(deps.bitmap());
      }
      XLS_ASSIGN_OR_RETURN(auto backwards_bm,
                           backwards_interesting.GetDependents(ps.value()));
      // Nodes that the selector affects & nodes the selected value is affected
      // by is the set of nodes with potentially changed conditional ranges.
      InlineBitmap final_bm = forward_bm;
      final_bm.Intersect(backwards_bm.bitmap());
      if (!final_bm.IsAllZeroes()) {
        // nodes affected by the known data are the ones we need to recalculate.
        interesting_states.push_back({ps, std::move(forward_bm)});
      }
    }
    return InterestingStatesAndNodeList{
        .node_indices = backwards_interesting.node_indices(),
        .state_and_nodes = interesting_states};
  }
  absl::StatusOr<RangeQueryEngine> CalculateRangeGiven(
      PredicateState s, const InlineBitmap& interesting_nodes,
      const absl::flat_hash_map<Node*, int64_t>& node_ids) const {
    RangeQueryEngine result;
    absl::flat_hash_map<Node*, RangeData> known_data;
    XLS_ASSIGN_OR_RETURN(known_data, ExtractKnownData(s));
    ContextGivens givens(
        topo_sort_, s.node(), known_data,
        [&](Node* n) -> std::optional<RangeData> {
          if (interesting_nodes.Get(node_ids.at(n))) {
            // Affected by known data.
            return std::nullopt;
          }
          // return memoized value from base
          return RangeData{
              .ternary =
                  n->GetType()->IsBits()
                      ? std::make_optional(base_range_.GetTernary(n).Get({}))
                      : std::nullopt,
              .interval_set = base_range_.GetIntervals(n),
          };
        });
    XLS_RETURN_IF_ERROR(result.PopulateWithGivens(givens).status());
    return result;
  }

  absl::StatusOr<absl::flat_hash_map<Node*, RangeData>> ExtractKnownData(
      PredicateState s) const {
    XLS_RET_CHECK(!s.IsBasePredicate())
        << "Can't back-propagate base predicate!";
    Select* select_node = s.node()->As<Select>();
    Node* selector = select_node->selector();
    CHECK(selector->GetType()->IsBits()) << "Non-bits select: " << *selector;
    IntervalSet given(selector->GetType()->GetFlatBitCount());
    if (s.IsDefaultArm()) {
      given.AddInterval(Interval::Open(
          UBits(select_node->cases().size(), selector->BitCountOrDie()),
          Bits::AllOnes(selector->BitCountOrDie())));
    } else {
      Bits value = UBits(s.arm_index(), selector->BitCountOrDie());
      given = IntervalSet::Precise(value);
    }
    XLS_ASSIGN_OR_RETURN(
        (absl::flat_hash_map<Node*, IntervalSet> intervals),
        PropagateOneGivenBackwards(base_range_, selector, given));
    absl::flat_hash_map<Node*, RangeData> ranges;
    ranges.reserve(intervals.size());
    for (auto [node, interval] : std::move(intervals)) {
      if (interval.IsEmpty()) {
        // This case is actually impossible? For now just ignore.
        // TODO: Figure out some way to communicate this.
        ranges[node] = RangeData{
            .ternary = base_range_.GetTernary(node).Get({}),
            .interval_set = base_range_.GetIntervals(node),
        };
      } else {
        ranges[node] =
            RangeData{.ternary = interval_ops::ExtractTernaryVector(interval),
                      .interval_set = IntervalSetTree::CreateSingleElementTree(
                          node->GetType(), std::move(interval))};
      }
    }
    return ranges;
  }

  std::vector<Node*> topo_sort_;
  RangeQueryEngine& base_range_;
  std::vector<std::unique_ptr<const RangeQueryEngine>>& arena_;
  absl::flat_hash_map<PredicateState, const RangeQueryEngine*>& engines_;
};

// A proxy query engine which specializes using select context.
class ProxyContextQueryEngine final : public QueryEngine {
 public:
  ProxyContextQueryEngine(const ContextSensitiveRangeQueryEngine& base,
                          const RangeQueryEngine& range_data)
      : base_(base), range_data_(range_data) {}

  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override {
    return absl::UnimplementedError(
        "Cannot populate proxy query engine. Populate must be called on "
        "original engine only.");
  }
  bool IsTracked(Node* node) const override { return base_.IsTracked(node); }

  LeafTypeTree<TernaryVector> GetTernary(Node* node) const override {
    return MostSpecific(node).GetTernary(node);
  }

  LeafTypeTree<IntervalSet> GetIntervals(Node* node) const override {
    return MostSpecific(node).GetIntervals(node);
  }

  bool AtMostOneTrue(absl::Span<TreeBitLocation const> bits) const override {
    TernaryVector ternary = GetTernaryOf(bits);
    return std::count_if(ternary.cbegin(), ternary.cend(), [&](TernaryValue v) {
             return v == TernaryValue::kKnownOne || v == TernaryValue::kUnknown;
           }) <= 1;
  }

  bool AtLeastOneTrue(absl::Span<TreeBitLocation const> bits) const override {
    TernaryVector ternary = GetTernaryOf(bits);
    return std::count_if(ternary.cbegin(), ternary.cend(), [&](TernaryValue v) {
             return v == TernaryValue::kKnownOne;
           }) >= 1;
  }

  bool Implies(const TreeBitLocation& a,
               const TreeBitLocation& b) const override {
    return MostSpecific(a.node(), b.node()).Implies(a, b);
  }
  // We're a range-analysis so no data here.
  std::optional<Bits> ImpliedNodeValue(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override {
    return std::nullopt;
  }

  bool KnownEquals(const TreeBitLocation& a,
                   const TreeBitLocation& b) const override {
    if (!IsKnown(a) || !IsKnown(b)) {
      return false;
    }
    TernaryValue av = GetTernary(a.node()).Get(a.tree_index())[a.bit_index()];
    TernaryValue bv = GetTernary(b.node()).Get(b.tree_index())[b.bit_index()];
    return av != TernaryValue::kUnknown && av == bv;
  }

  // Returns true if 'a' is the inverse of 'b'
  bool KnownNotEquals(const TreeBitLocation& a,
                      const TreeBitLocation& b) const override {
    TernaryValue av = GetTernary(a.node()).Get(a.tree_index())[a.bit_index()];
    TernaryValue bv = GetTernary(b.node()).Get(b.tree_index())[b.bit_index()];
    return av != TernaryValue::kUnknown && bv != TernaryValue::kUnknown &&
           av != bv;
  }

 private:
  TernaryVector GetTernaryOf(absl::Span<TreeBitLocation const> bits) const {
    // TODO(allight): Very inefficient but the AtMost/AtLeastOne don't seem to
    // actually be used?
    InlineBitmap known(bits.size());
    InlineBitmap values(bits.size());
    for (int64_t i = 0; i < bits.size(); ++i) {
      bool bit_known = IsKnown(bits[i]);
      if (bit_known) {
        known.Set(i, true);
        values.Set(i, IsOne(bits[i]));
      }
    }
    return ternary_ops::FromKnownBits(Bits::FromBitmap(known),
                                      Bits::FromBitmap(values));
  }
  const QueryEngine& MostSpecific(Node* nodeA, Node* nodeB) const {
    if (range_data_.HasKnownIntervals(nodeA) &&
        range_data_.HasKnownIntervals(nodeB)) {
      return range_data_;
    }
    return base_;
  }
  const QueryEngine& MostSpecific(Node* node) const {
    if (range_data_.HasKnownIntervals(node)) {
      return range_data_;
    }
    return base_;
  }
  const ContextSensitiveRangeQueryEngine& base_;
  const RangeQueryEngine& range_data_;
};

}  // namespace

absl::StatusOr<ReachedFixpoint> ContextSensitiveRangeQueryEngine::Populate(
    FunctionBase* f) {
  Analysis analysis(base_case_ranges_, arena_, one_hot_ranges_);
  return analysis.Execute(f);
}

std::unique_ptr<QueryEngine>
ContextSensitiveRangeQueryEngine::SpecializeGivenPredicate(
    const absl::flat_hash_set<PredicateState>& state) const {
  // Currently only single element states are supported.
  // We do check for consistency here but really we just ignore extra state
  // elements since something that is true for A is also true for A && B. We
  // don't have any particular strategy for picking which one gets to be the
  // 'real' state just using 'begin'.
  CHECK_LE(state.size(), 1);
  if (state.empty() || !one_hot_ranges_.contains(*state.cbegin())) {
    return QueryEngine::SpecializeGivenPredicate(state);
  }
  return std::make_unique<ProxyContextQueryEngine>(
      *this, *one_hot_ranges_.at(*state.cbegin()));
}

}  // namespace xls
