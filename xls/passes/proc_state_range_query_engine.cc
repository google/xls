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

#include "xls/passes/proc_state_range_query_engine.h"

#include <cstdint>
#include <optional>
#include <queue>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function_base.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_ops.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/proc.h"
#include "xls/ir/state_element.h"
#include "xls/ir/ternary.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/back_propagate_range_analysis.h"
#include "xls/passes/dataflow_visitor.h"
#include "xls/passes/node_dependency_analysis.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/range_query_engine.h"
#include "xls/passes/ternary_query_engine.h"

namespace xls {

namespace {

class ProcStateGivens : public RangeDataProvider, public TernaryDataProvider {
 public:
  ProcStateGivens(Proc* proc, absl::flat_hash_map<Node*, IntervalSet> intervals)
      : proc_(proc), intervals_(std::move(intervals)) {}
  absl::Status IterateFunction(DfsVisitor* visitor) override {
    return proc_->Accept(visitor);
  }

  std::optional<RangeData> GetKnownIntervals(Node* node) final {
    if (intervals_.contains(node) && !intervals_.at(node).IsEmpty()) {
      return RangeData{
          .ternary = interval_ops::ExtractTernaryVector(intervals_.at(node)),
          .interval_set = IntervalSetTree::CreateSingleElementTree(
              node->GetType(), intervals_.at(node))};
    }
    return std::nullopt;
  }

  std::optional<LeafTypeTree<TernaryVector>> GetKnownTernary(
      Node* node) const final {
    if (intervals_.contains(node) && !intervals_.at(node).IsEmpty()) {
      CHECK(node->GetType()->IsBits());
      return LeafTypeTree<TernaryVector>(
          node->GetType(),
          interval_ops::ExtractTernaryVector(intervals_.at(node)));
    }
    return std::nullopt;
  }

 private:
  Proc* proc_;
  absl::flat_hash_map<Node*, IntervalSet> intervals_;
};

// A givens that restricts the iteration to only values that hit the proc-state
// directly.
class ProcStateEvolutionGivens : public ProcStateGivens {
 public:
  ProcStateEvolutionGivens(absl::Span<Node* const> reverse_topo_sort,
                           Node* target,
                           absl::flat_hash_map<Node*, IntervalSet> intervals,
                           const DependencyBitmap& interesting_nodes)
      : ProcStateGivens(target->function_base()->AsProcOrDie(),
                        std::move(intervals)),
        reverse_topo_sort_(reverse_topo_sort),
        target_(target),
        interesting_nodes_(interesting_nodes) {}

  absl::Status IterateFunction(DfsVisitor* visitor) final {
    for (auto it = reverse_topo_sort_.crbegin();
         it != reverse_topo_sort_.crend(); ++it) {
      // Don't bother filling in information for nodes which don't lead to the
      // 'next' we're looking at.
      XLS_ASSIGN_OR_RETURN(bool interesting,
                           interesting_nodes_.IsDependent(*it));
      if (interesting) {
        XLS_RETURN_IF_ERROR((*it)->VisitSingleNode(visitor)) << *it;
      }
      if (*it == target_) {
        // We got the actual next value, no need to continue;
        break;
      }
    }
    return absl::OkStatus();
  }

 private:
  absl::Span<Node* const> reverse_topo_sort_;
  Node* target_;
  const DependencyBitmap& interesting_nodes_;
};

absl::StatusOr<std::optional<std::pair<TernaryVector, IntervalSet>>>
ExtractContextSensitiveRange(
    Proc* proc, Next* next, const RangeQueryEngine& rqe,
    absl::Span<Node* const> reverse_topo_sort,
    const NodeDependencyAnalysis& next_dependent_information) {
  Node* pred = *next->predicate();
  XLS_ASSIGN_OR_RETURN(
      (absl::flat_hash_map<Node*, IntervalSet> results),
      PropagateGivensBackwards(rqe, proc,
                               {{pred, IntervalSet::Precise(UBits(1, 1))}},
                               reverse_topo_sort));
  // Check if anything interesting was found.
  if (absl::c_all_of(results,
                     [&](const std::pair<Node*, IntervalSet>& entry) -> bool {
                       const auto& [node, interval] = entry;
                       return node == pred || node->Is<Literal>() ||
                              interval.IsMaximal() ||
                              interval == rqe.GetIntervals(node).Get({});
                     })) {
    // Nothing except for literals, unconstrained values or already discovered
    // values found. There's no point in doing anything more.
    return std::nullopt;
  }
  // Some new info found from the back-prop. Apply it.
  // TODO(allight): A heuristic to avoid doing this in some cases (such as none
  // of the discovered facts are in the predecessors of value) might be
  // worthwhile here.
  XLS_ASSIGN_OR_RETURN(DependencyBitmap dependencies,
                       next_dependent_information.GetDependents(next));
  ProcStateEvolutionGivens givens(reverse_topo_sort, next->value(),
                                  std::move(results), dependencies);
  RangeQueryEngine contextual_range;
  XLS_RETURN_IF_ERROR(contextual_range.PopulateWithGivens(givens).status());
  std::optional<LeafTypeTree<TernaryVector>> ternary =
      contextual_range.GetTernary(next->value());
  TernaryVector ternary_vec =
      ternary.has_value() ? ternary->Get({})
                          : TernaryVector(next->value()->BitCountOrDie(),
                                          TernaryValue::kUnknown);
  return std::make_pair(ternary_vec,
                        contextual_range.GetIntervals(next->value()).Get({}));
}

class SegmentRangeData : public RangeDataProvider {
 public:
  static absl::StatusOr<SegmentRangeData> Create(
      const NodeDependencyAnalysis& nda,
      const absl::flat_hash_map<StateElement*, RangeData>& ground_truth,
      StateRead* data_source, absl::Span<Node* const> topo_sort) {
    XLS_RET_CHECK(!nda.IsForward());
    std::vector<DependencyBitmap> bitmaps;
    auto nexts =
        data_source->function_base()->AsProcOrDie()->next_values(data_source);
    bitmaps.reserve(nexts.size());
    for (Next* n : nexts) {
      XLS_ASSIGN_OR_RETURN(DependencyBitmap bm, nda.GetDependents(n));
      bitmaps.push_back(bm);
    }
    return SegmentRangeData(bitmaps, ground_truth, data_source, topo_sort);
  }

  void SetParamIntervals(const IntervalSet& is) { current_segments_ = is; }

  bool IsInteresting(Node* n) const {
    return (n->Is<Next>() && n->As<Next>()->state_read() == data_source_) ||
           absl::c_any_of(dependencies_,
                          [&](const DependencyBitmap& d) -> bool {
                            return d.IsDependent(n).value_or(false);
                          });
  }

  std::optional<RangeData> GetKnownIntervals(Node* node) final {
    if (node == data_source_) {
      CHECK(!current_segments_.IsEmpty());
      return RangeData{.ternary = interval_ops::ExtractTernaryVector(
                           current_segments_, node),
                       .interval_set = IntervalSetTree::CreateSingleElementTree(
                           node->GetType(), current_segments_)};
    }
    if (node->Is<StateRead>()) {
      StateElement* state_element = node->As<StateRead>()->state_element();
      if (!node->GetType()->IsBits()) {
        return std::nullopt;
      }
      if (auto it = ground_truth_.find(state_element);
          it != ground_truth_.end()) {
        return it->second;
      }
      return std::nullopt;
    }
    // TODO(allight) We could be a bit more efficient by pre-calculating the
    // nodes which feed the next node but not the fed from the param by running
    // a TQE on the initial narrowing and using those values. Not clear its
    // worth the complication.
    return std::nullopt;
  }

  absl::Status IterateFunction(DfsVisitor* visitor) final {
    for (Node* node : topo_sort_) {
      // Don't bother to calculate anything nodes which don't reach a next
      // instruction.
      if (IsInteresting(node)) {
        XLS_RETURN_IF_ERROR(node->VisitSingleNode(visitor)) << node;
      }
    }
    return absl::OkStatus();
  }

 private:
  SegmentRangeData(
      std::vector<DependencyBitmap> dependencies,
      const absl::flat_hash_map<StateElement*, RangeData>& ground_truth,
      StateRead* data_source, absl::Span<Node* const> topo_sort)
      : dependencies_(std::move(dependencies)),
        ground_truth_(ground_truth),
        data_source_(data_source),
        current_segments_(data_source->BitCountOrDie()),
        topo_sort_(topo_sort) {}
  std::vector<DependencyBitmap> dependencies_;
  const absl::flat_hash_map<StateElement*, RangeData>& ground_truth_;
  StateRead* data_source_;
  IntervalSet current_segments_;
  absl::Span<Node* const> topo_sort_;
};
bool AbsoluteValueLessThan(const Bits& l, const Bits& r) {
  CHECK_EQ(l.bit_count(), r.bit_count());
  Bits max_int = Bits::MaxSigned(l.bit_count());
  bool l_pos = !l.GetFromMsb(0);
  bool r_pos = !r.GetFromMsb(0);
  if (l_pos && r_pos) {
    return bits_ops::ULessThan(l, r);
  }
  if (!l_pos && !r_pos) {
    return bits_ops::UGreaterThan(l, r);
  }
  if (!l_pos) {
    return bits_ops::ULessThan(bits_ops::Negate(l), r);
  }
  CHECK(!r_pos);
  return bits_ops::ULessThan(l, bits_ops::Negate(r));
}

// An interpreter that finds the values of nodes assuming that only Literal
// constant values are selected.
class ConstantValueIrInterpreter
    : public DataflowVisitor<absl::flat_hash_set<Bits>> {
 public:
  // How many values we will track at most. We will prioritize the values
  // closest to zero in the 2s-complement signed integer space. We prioritize
  // values close to signed zero since values close to that value can be
  // narrowed more than ones further away. 8 was picked pretty arbitrarily but
  // it's hard to imagine many procs with more than a handful of constant set
  // values which are still narrowable.
  static constexpr int64_t kSegmentLimit = 8;
  const absl::flat_hash_map<Node*, LeafTypeTree<absl::flat_hash_set<Bits>>>&
  values() const {
    return map_;
  }
  absl::Status DefaultHandler(Node* n) override {
    if ((OpIsSideEffecting(n->op()) || n->op() == Op::kAfterAll) &&
        n->op() != Op::kGate) {
      // Side effecting ops (eg send, recv, trace, cover etc) either don't
      // return anything or don't return any constants regardless of inputs and
      // cannot be interpreted. These are considered sources of unconstrained
      // values.
      return HandleNonConst(n);
    }
    // Non-bits ops are more complicated to handle. The only ones here should be
    // things like mulp which it's not clear if we should care much about...
    if (!n->GetType()->IsBits()) {
      VLOG(2) << "Ignoring non-bits type op " << n;
      return HandleNonConst(n);
    }
    bool args_are_all_bits = absl::c_all_of(
        n->operands(), [](Node* o) { return o->GetType()->IsBits(); });
    if (n->OpIn({Op::kEq, Op::kNe}) && !args_are_all_bits) {
      VLOG(2) << "Ignoring eq/ne of non-bits values " << n;
      return HandleNonConst(n);
    }
    XLS_RET_CHECK(args_are_all_bits) << n;
    // Heap keeping the current kSegmentLimit closest elements to 0
    std::vector<Bits> result_heap;
    auto insert_result = [&](Bits b) {
      if (result_heap.size() >= kSegmentLimit &&
          !AbsoluteValueLessThan(b, result_heap.front())) {
        // Bigger than existing values.
        return;
      }
      if (absl::c_find(result_heap, b) != result_heap.cend()) {
        // Already contained.
        return;
      }
      if (result_heap.size() == kSegmentLimit) {
        absl::c_pop_heap(result_heap, AbsoluteValueLessThan);
        result_heap.pop_back();
      }
      result_heap.emplace_back(std::move(b));
      absl::c_push_heap(result_heap, AbsoluteValueLessThan);
    };
    struct ArgSet {
      const absl::flat_hash_set<Bits>& values;
      absl::flat_hash_set<Bits>::const_iterator cur_value;
    };
    std::vector<ArgSet> inputs;
    for (Node* o : n->operands()) {
      if (GetValue(o).Get({}).empty()) {
        // Some input has no constant values, so this node has no constant
        // derived value.
        return HandleNonConst(n);
      }
      const auto& v = GetValue(o).Get({});
      inputs.push_back(ArgSet{.values = v, .cur_value = v.cbegin()});
    }
    if (inputs.empty()) {
      // Only zero-arg node that we care about is literal which is handled
      // elsewhere.
      return absl::OkStatus();
    }
    auto current_value = [&]() -> std::vector<Value> {
      std::vector<Value> res;
      res.reserve(inputs.size());
      for (const auto& v : inputs) {
        res.push_back(Value(*v.cur_value));
      }
      return res;
    };
    auto next_input = [&]() {
      for (auto it = inputs.rbegin(); it != inputs.rend(); ++it) {
        if (it != inputs.rbegin()) {
          (it - 1)->cur_value = (it - 1)->values.cbegin();
        }
        ++it->cur_value;
        if (it->cur_value != it->values.cend()) {
          break;
        }
      }
    };
    // Don't try more than ~4000 constants. This is pretty quick but avoids
    // combinatorial explosions caused by things like a concat of a ton of
    // (possibly correlated) selects.
    // TODO(allight): This ends up checking mostly just the values of the last
    // operand. This might or might not be desirable depending on the operation
    // and it would be good to be more intelligent.
    static constexpr int64_t kMaxConstantValues = 1 << 12;
    for (int64_t cnt = 0;
         cnt < kMaxConstantValues &&
         inputs.front().cur_value != inputs.front().values.cend();
         next_input(), ++cnt) {
      XLS_ASSIGN_OR_RETURN(Value r, InterpretNode(n, current_value()));
      insert_result(std::move(r).bits());
    }
    VLOG(3) << "Node " << n << " can have constant values of ["
            << absl::StrJoin(result_heap, ", ") << "]";
    absl::flat_hash_set<Bits> res;
    res.insert(result_heap.begin(), result_heap.end());
    return SetValue(
        n, LeafTypeTree<absl::flat_hash_set<Bits>>::CreateSingleElementTree(
               n->GetType(), std::move(res)));
  }

  absl::Status HandleNext(Next* n) override {
    return absl::InternalError(absl::StrFormat(
        "Unexpected invoke of %s. Next nodes should not feed into anything.",
        n->ToString()));
  }

  absl::Status HandleNonConst(Node* n) {
    return SetValue(n, LeafTypeTree<absl::flat_hash_set<Bits>>(
                           n->GetType(), absl::flat_hash_set<Bits>{}));
  }

  absl::Status HandleParam(Param* p) override { return HandleNonConst(p); }

  absl::Status HandleLiteral(Literal* l) override {
    XLS_ASSIGN_OR_RETURN(LeafTypeTree<Value> value_ltt,
                         ValueToLeafTypeTree(l->value(), l->GetType()));
    return SetValue(l, leaf_type_tree::Map<absl::flat_hash_set<Bits>, Value>(
                           value_ltt.AsView(),
                           [](const Value& v) -> absl::flat_hash_set<Bits> {
                             if (v.IsToken()) {
                               return {UBits(0, 0)};
                             }
                             return {v.bits()};
                           }));
  }

  // TODO(allight): Technically we could go through this but its hard to see
  // what the benefit would be.
  absl::Status HandleArraySlice(ArraySlice* a) override {
    return HandleNonConst(a);
  }

 protected:
  absl::StatusOr<absl::flat_hash_set<Bits>> JoinElements(
      Type* element_type,
      absl::Span<const absl::flat_hash_set<Bits>* const> data_sources,
      absl::Span<const LeafTypeTreeView<absl::flat_hash_set<Bits>>>
          control_sources,
      Node* node, absl::Span<const int64_t> index) const override {
    if (!element_type->IsBits()) {
      return absl::flat_hash_set<Bits>{};
    }
    struct NotAbsSignedCompare {
      bool operator()(const Bits& l, const Bits& r) {
        return AbsoluteValueLessThan(r, l);
      }
    };
    // Priority queue ordered small to large in absolute value.
    std::priority_queue<Bits, std::vector<Bits>, NotAbsSignedCompare> res;
    for (const absl::flat_hash_set<Bits>* v : data_sources) {
      for (const Bits& b : *v) {
        res.push(b);
      }
    }
    absl::flat_hash_set<Bits> out;
    out.reserve(kSegmentLimit);
    while (out.size() < kSegmentLimit && !res.empty()) {
      out.insert(res.top());
      res.pop();
    }
    return out;
  }
};

// Find all values where the antecedents (excepting selector values) are purely
// constants which update the given param.
absl::StatusOr<absl::flat_hash_set<Bits>> FindConstantUpdateValues(
    Proc* proc, StateElement* orig_state_element,
    absl::Span<Node* const> topo_sort, const NodeDependencyAnalysis& nda) {
  ConstantValueIrInterpreter interp;
  std::vector<DependencyBitmap> next_deps;
  absl::flat_hash_set<Node*> visited_next_values;
  XLS_RET_CHECK(orig_state_element->type()->IsBits());
  StateRead* orig_state_read = proc->GetStateRead(orig_state_element);
  next_deps.reserve(proc->next_values(orig_state_read).size());
  visited_next_values.reserve(proc->next_values(orig_state_read).size());
  for (Next* n : proc->next_values(orig_state_read)) {
    if (auto [it, inserted] = visited_next_values.insert(n->value());
        !inserted) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(DependencyBitmap bm, nda.GetDependents(n->value()));
    next_deps.push_back(bm);
  }
  auto is_interesting = [&](Node* n) {
    return absl::c_any_of(next_deps, [&](const DependencyBitmap& d) {
      return *d.IsDependent(n);
    });
  };
  for (Node* n : topo_sort) {
    if (!is_interesting(n)) {
      continue;
    }
    XLS_RETURN_IF_ERROR(n->VisitSingleNode(&interp));
  }
  absl::flat_hash_set<Bits> param_values;
  param_values.insert(orig_state_element->initial_value().bits());
  visited_next_values.clear();
  for (Next* n : proc->next_values(orig_state_read)) {
    if (auto [it, inserted] = visited_next_values.insert(n->value());
        !inserted) {
      continue;
    }
    const auto& values = interp.values();
    if (!values.contains(n->value())) {
      continue;
    }
    LeafTypeTreeView<absl::flat_hash_set<Bits>> v =
        values.at(n->value()).AsView();
    XLS_RET_CHECK(v.type()->IsBits());
    for (const Bits& b : v.Get({})) {
      param_values.insert(b);
    }
  }
  return std::move(param_values);
}

// Snip off segments of the interval set if possible.
//
// We do this by running the update function against the interval which contains
// the initial value. Any resulting interval is intersected with the given true
// intervals. Any interval that overlaps is added to the active set and the
// process is repeated until there are no new active intervals. Note that we do
// not take into account the size of the overlap. We assume that the update
// function is relatively dense and any entry into a segment makes the entire
// segment live. This enables us to do this state exploration with a relatively
// small number of runs.
absl::StatusOr<std::optional<RangeData>> NarrowUsingSegments(
    Proc* proc, StateElement* state_element, const IntervalSet& intervals,
    absl::Span<Node* const> topo_sort, const NodeDependencyAnalysis& nda,
    const absl::flat_hash_map<StateElement*, RangeData>& ground_truth) {
  VLOG(3) << "Doing segment walk for " << state_element->ToString() << " on "
          << intervals;
  absl::flat_hash_set<Interval> remaining_intervals(
      intervals.Intervals().begin(), intervals.Intervals().end());
  // Split each interval which is reachable using only constants into 3 segments
  // [<Before value>, <value>, <after value>]. We start with only the initial
  // value in the active segment. After this we assume that any value in a
  // segment makes the entire segment active. This is to handle values which go
  // down to 0.
  XLS_ASSIGN_OR_RETURN(
      absl::flat_hash_set<Bits> constant_update_values,
      FindConstantUpdateValues(proc, state_element, topo_sort, nda));
  VLOG(3) << "  Constant-derived values for updates are ["
          << absl::StrJoin(constant_update_values, ", ") << "]";
  for (const Bits& v : constant_update_values) {
    auto it = absl::c_find_if(remaining_intervals,
                              [&](const Interval& i) { return i.Covers(v); });
    if (it == remaining_intervals.cend()) {
      // Apparently this value is unreachable. Odd but possible.
      continue;
    }
    if (it->IsPrecise()) {
      // There's nothing to split.
      continue;
    }
    Interval interval = *it;  // NOLINT: Cannot use a reference since
                              // inserting values invalidates the reference.
    remaining_intervals.erase(it);
    if (v != interval.LowerBound()) {
      remaining_intervals.insert(
          Interval::Closed(interval.LowerBound(), bits_ops::Decrement(v)));
    }
    remaining_intervals.insert(Interval::Precise(v));
    if (v != interval.UpperBound()) {
      remaining_intervals.insert(
          Interval::Closed(bits_ops::Increment(v), interval.UpperBound()));
    }
  }
  VLOG(3) << "  state space separated into ["
          << absl::StrJoin(remaining_intervals, ", ") << "]";
  Value init_value = state_element->initial_value();
  XLS_RET_CHECK(intervals.Covers(init_value.bits()))
      << "Invalid interval calculation for " << state_element->ToString()
      << ". Initial value " << init_value << " was marked unreachable.";
  IntervalSet active_intervals = IntervalSet::Precise(init_value.bits());
  CHECK(remaining_intervals.contains(Interval::Precise(init_value.bits())))
      << "Initial value not included in constant values.";
  remaining_intervals.erase(Interval::Precise(init_value.bits()));
  StateRead* state_read = proc->GetStateRead(state_element);
  XLS_ASSIGN_OR_RETURN(
      SegmentRangeData limiter,
      SegmentRangeData::Create(nda, ground_truth, state_read, topo_sort));
  while (!remaining_intervals.empty()) {
    // Get the ranges of every node (which leads to a 'next' of the param)
    limiter.SetParamIntervals(active_intervals);
    RangeQueryEngine rqe;
    XLS_RETURN_IF_ERROR(rqe.PopulateWithGivens(limiter).status());

    // Get what this says all ranges are.
    IntervalSet run_intervals = active_intervals;
    absl::flat_hash_set<Node*> visited_next_values;
    visited_next_values.reserve(proc->next_values(state_read).size());
    for (Next* n : proc->next_values(state_read)) {
      // Nexts which don't update anything (either due to just being passthrough
      // or having a known-false predicate) don't need to be taken into account.
      if (n->value() == n->state_read() ||
          (n->predicate() && rqe.IsAllZeros(*n->predicate()))) {
        continue;
      }
      // Skip analyzing this value if we've already seen it.
      if (auto [it, inserted] = visited_next_values.insert(n->value());
          !inserted) {
        continue;
      }
      if (!rqe.HasExplicitIntervals(n->value())) {
        // Unconstrained result. All segments active.
        return std::nullopt;
      }
      // This next node might participate in the selection of the next value
      // and is not a no-op.
      run_intervals = IntervalSet::Combine(
          run_intervals,
          rqe.GetIntervalSetTreeView(n->value()).value().Get({}));
    }

    // Does this reveal new states?
    auto overlap = absl::c_find_if(remaining_intervals, [&](const Interval& i) {
      auto found_list = run_intervals.Intervals();
      return absl::c_any_of(found_list, [&](const Interval& r) {
        return Interval::Overlaps(i, r);
      });
    });
    if (overlap == remaining_intervals.cend()) {
      // Didn't discover anything new. The current active intervals are the
      // final result.
      return RangeData{
          .ternary = interval_ops::ExtractTernaryVector(active_intervals),
          .interval_set = IntervalSetTree::CreateSingleElementTree(
              state_element->type(), active_intervals)};
    }
    active_intervals.AddInterval(*overlap);
    active_intervals.Normalize();
    remaining_intervals.erase(overlap);
  }
  // Always able to find an element to expand to in the intervals so we aren't
  // able to reduce it at all.
  XLS_RET_CHECK_EQ(active_intervals, intervals);
  return std::nullopt;
}
// Narrow ranges using the contextual information of the next predicates.
absl::StatusOr<absl::flat_hash_map<StateElement*, RangeData>>
FindContextualRanges(Proc* proc, const QueryEngine& qe,
                     const RangeQueryEngine& rqe,
                     const NodeDependencyAnalysis& dependency_analysis,
                     absl::Span<Node* const> reverse_topo_sort) {
  // List of all the next instructions that change the param for each param.
  absl::flat_hash_map<StateElement*, std::vector<Next*>>
      modifying_nexts_for_state;
  for (StateElement* state_element : proc->StateElements()) {
    // TODO(allight): Being able to narrow inside a compound value would be
    // nice. Since we unpack tuple state elements in other passes however the
    // actual impact would likely be negligible so no reason to bother with it
    // for now.
    if (!state_element->type()->IsBits()) {
      continue;
    }
    std::vector<Next*>& nexts = modifying_nexts_for_state[state_element];
    StateRead* state_read = proc->GetStateRead(state_element);
    for (Next* n : proc->next_values(state_read)) {
      // TODO(allight): We might want to use data-flow to better track whether
      // things have changed. This should probably be good enough in practice
      // however.
      if (n->state_read() != n->value()) {
        nexts.push_back(n);
      }
    }
  }
  // To avoid issues where changes to the param values leads to invalidating the
  // TernaryQueryEngine we do all the modifications at the end.
  absl::flat_hash_map<StateElement*, RangeData> transforms;
  for (const auto& [orig_state_element, updates] : modifying_nexts_for_state) {
    if (updates.empty()) {
      // The state only has identity updates? Strange but this will be cleaned
      // up by NextValueOptimizationPass so we can ignore it.
      continue;
    }
    Value orig_init_value = orig_state_element->initial_value();
    TernaryVector possible_values =
        ternary_ops::BitsToTernary(orig_init_value.bits());

    IntervalSet contextual_intervals =
        IntervalSet::Precise(orig_init_value.bits());
    for (Next* next : updates) {
      std::optional<LeafTypeTree<TernaryVector>> context_free_ltt =
          qe.GetTernary(next->value());
      TernaryVector context_free =
          context_free_ltt.has_value()
              ? context_free_ltt->Get({})
              : TernaryVector(next->value()->BitCountOrDie(),
                              TernaryValue::kUnknown);
      // NB Only doing context-sensitive range analysis is a heuristic to avoid
      // performing the (somewhat) expensive range propagation when we have
      // already narrowed using static analysis. While its possible that better
      // bounds could be obtained by context-sensitive analysis it seems likely
      // this will generally not be true (since most operations that
      // ternary-analysis is able to narrow high bits on are not handled well by
      // range analysis).
      // TODO(allight): Once signed bounds are supported better we should check
      // this again and determine more precisely the sort of performance impact
      // always doing (non-trivial) range analysis would have.
      if (ternary_ops::ToKnownBits(context_free).CountLeadingOnes() == 0 &&
          next->predicate()) {
        // Context-free query engine wasn't able to narrow this at all and we do
        // have additional information in the form of a predicate. Try again
        // with contextual information.
        XLS_ASSIGN_OR_RETURN(
            (std::optional<std::pair<TernaryVector, IntervalSet>>
                 contextual_result),
            ExtractContextSensitiveRange(proc, next, rqe, reverse_topo_sort,
                                         dependency_analysis),
            _ << next);
        // Keep track of all the values that we can update to using ranges.
        if (contextual_result) {
          const auto& [contextual_tern, contextual_range] = *contextual_result;
          possible_values =
              ternary_ops::Intersection(possible_values, contextual_tern);
          contextual_intervals =
              IntervalSet::Combine(contextual_intervals, contextual_range);
        } else {
          possible_values =
              ternary_ops::Intersection(possible_values, context_free);
          contextual_intervals = IntervalSet::Combine(
              contextual_intervals, interval_ops::FromTernary(context_free));
        }
      } else {
        possible_values =
            ternary_ops::Intersection(possible_values, context_free);
        contextual_intervals = IntervalSet::Combine(
            contextual_intervals, interval_ops::FromTernary(context_free));
      }
    }
    transforms[orig_state_element] = RangeData{
        .ternary = possible_values,
        .interval_set = IntervalSetTree::CreateSingleElementTree(
            orig_state_element->type(), contextual_intervals),
    };
  }
  return transforms;
}
}  // namespace

// TODO(allight): Technically we'd probably want to run this whole populate to
// fixed point (incorporating the results into later runs) to get optimal
// results. It's not clear how much we'd gain there though. For now we will
// just run it once assuming that params are relatively independent of one
// another/additional information won't reveal more opportunities.
absl::StatusOr<ReachedFixpoint> ProcStateRangeQueryEngine ::Populate(
    FunctionBase* f) {
  // Start with a basic range and ternary analysis to get base cases.
  XLS_ASSIGN_OR_RETURN(ReachedFixpoint fixpoint, inner_.Populate(f));
  // If we aren't able to actually analyze proc-state we are done here.
  if (!ProcStateRangeQueryEngine::CanAnalyzeProcStateEvolution(f)) {
    return fixpoint;
  }
  XLS_RET_CHECK(f->IsProc());
  Proc* proc = f->AsProcOrDie();
  std::vector<Node*> reverse_topo_sort = ReverseTopoSort(proc);
  std::vector<Node*> topo_sort = reverse_topo_sort;
  absl::c_reverse(topo_sort);
  // Get the nodes which actually affect the next-value nodes. We don't really
  // care about anything else.
  std::vector<Node*> interesting_nodes;
  interesting_nodes.reserve(2 * proc->next_values().size());
  for (Next* n : proc->next_values()) {
    interesting_nodes.push_back(n);
    interesting_nodes.push_back(n->value());
  }
  NodeDependencyAnalysis next_node_sources =
      NodeDependencyAnalysis::BackwardDependents(proc, interesting_nodes);

  // TODO(allight): We could repeat the below and the loop until we hit a
  // fixed-point to fully incorporate all cross-param knowledge. This could be
  // quite slow however.
  XLS_ASSIGN_OR_RETURN(
      (absl::flat_hash_map<StateElement*, RangeData> initial_transforms),
      FindContextualRanges(proc, inner_, *range_, next_node_sources,
                           reverse_topo_sort));
  // Find implied ranges for each param. Note that we consider each parameter in
  // isolation. Technically we could go to fixed-point and maybe get better
  // bounds but that could take a while.
  absl::flat_hash_map<StateElement*, RangeData> final_range_data;
  for (const auto& [orig_state_element, t] : initial_transforms) {
    const auto& [ternary, interval_set] = t;
    int64_t known_leading =
        ternary_ops::ToKnownBits(*ternary).CountLeadingOnes();
    // If we have known leading bits from the ternary analysis and only care
    // about state params use that. These are usually good enough except with
    // signed integer things (identified as only being able to eliminate the
    // sign bit or not being able to eliminate anything).
    if (known_leading > 1) {
      VLOG(2) << "Narrowed range of " << orig_state_element->ToString()
              << " to "
              << (orig_state_element->type()->GetFlatBitCount() - known_leading)
              << " bits (savings: " << known_leading
              << ") using back-prop/ternary. Interval is: "
              << interval_set.Get({});
      final_range_data[orig_state_element] = t;
      continue;
    }
    // We can't remove segments from a 1 bit value.
    if (interval_set.Get({}).BitCount() < 2) {
      VLOG(2) << "Unable to narrow range of " << orig_state_element->ToString()
              << ". Value is unconstrained. Interval is: "
              << interval_set.Get({});
      continue;
    }
    // Try for signed value compression. We *only* do this if there are no
    // narrowings we can do without checking this. This is under the assumption
    // that in most cases the only thing that would case discontinuous range
    // results without having known high-bits is signed value comparisons
    // somewhere in the next-value predicates. This is likely to cause ranges
    // where there's a low bits region which is real and then all of the
    // negative numbers are also included.

    // Interval set is partitioned so we might be able to prove that no value
    // can move from one partition to another cutting down in the possible
    // values.
    XLS_ASSIGN_OR_RETURN(
        std::optional<RangeData> narrowed,
        NarrowUsingSegments(proc, orig_state_element, interval_set.Get({}),
                            topo_sort, next_node_sources, initial_transforms));
    if (narrowed) {
      VLOG(2)
          << "Narrowed range of " << orig_state_element->ToString() << " to "
          << (orig_state_element->type()->GetFlatBitCount() -
              ternary_ops::ToKnownBits(*narrowed->ternary).CountLeadingOnes())
          << " bits (savings: "
          << ternary_ops::ToKnownBits(*narrowed->ternary).CountLeadingOnes()
          << ") using segment walking. Interval is "
          << narrowed->interval_set.Get({});
      final_range_data[orig_state_element] = *narrowed;
    } else {
      VLOG(2) << "Unable to narrow range " << orig_state_element->ToString()
              << ". Segment walking unable to eliminate high bits. Interval is "
              << interval_set.Get({});
    }
  }

  // We now have intervals for all params. We run one more query-engine run to
  // get final results.
  TernaryQueryEngine spec_ternary;
  RangeQueryEngine spec_range;

  absl::flat_hash_map<Node*, IntervalSet> state_read_intervals;
  state_read_intervals.reserve(final_range_data.size());
  for (const auto& [state_element, range] : final_range_data) {
    state_read_intervals[proc->GetStateRead(state_element)] =
        range.interval_set.Get({});
  }
  ProcStateGivens givens(proc, std::move(state_read_intervals));
  XLS_RETURN_IF_ERROR(spec_ternary.PopulateWithGivens(proc, givens).status());
  XLS_RETURN_IF_ERROR(spec_range.PopulateWithGivens(givens).status());

  *ternary_ = std::move(spec_ternary);
  *range_ = std::move(spec_range);

  return ReachedFixpoint::Changed;
}
}  // namespace xls
