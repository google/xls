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

#include "xls/passes/proc_state_narrowing_pass.h"

#include <array>
#include <cstdint>
#include <iterator>
#include <optional>
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
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_ops.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/proc.h"
#include "xls/ir/ternary.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/value.h"
#include "xls/passes/back_propagate_range_analysis.h"
#include "xls/passes/node_dependency_analysis.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/range_query_engine.h"
#include "xls/passes/ternary_query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {
namespace {

// Struct which transforms a param into a slice of its trailing bits.
struct ProcStateNarrowTransform : public Proc::StateElementTransformer {
 public:
  explicit ProcStateNarrowTransform(Bits known_leading)
      : Proc::StateElementTransformer(),
        known_leading_(std::move(known_leading)) {}

  absl::StatusOr<Node*> TransformParamRead(Proc* proc, Param* new_param,
                                           Param* old_param) final {
    XLS_RET_CHECK_EQ(
        new_param->GetType()->GetFlatBitCount() + known_leading_.bit_count(),
        old_param->GetType()->GetFlatBitCount());
    XLS_ASSIGN_OR_RETURN(
        Node * leading,
        proc->MakeNodeWithName<Literal>(
            old_param->loc(), Value(known_leading_),
            absl::StrFormat("leading_bits_%s", old_param->name())));
    return proc->MakeNodeWithName<Concat>(
        new_param->loc(), std::array<Node*, 2>{leading, new_param},
        absl::StrFormat("extended_%s", old_param->name()));
  }
  absl::StatusOr<Node*> TransformNextValue(Proc* proc, Param* new_param,
                                           Next* old_next) final {
    XLS_RET_CHECK_EQ(
        new_param->GetType()->GetFlatBitCount() + known_leading_.bit_count(),
        old_next->param()->GetType()->GetFlatBitCount());
    return proc->MakeNodeWithName<BitSlice>(
        old_next->loc(), old_next->value(), /*start=*/0,
        /*width=*/new_param->GetType()->GetFlatBitCount(),
        absl::StrFormat("unexpand_for_%s", old_next->GetName()));
  }

 private:
  Bits known_leading_;
};

class NarrowGivens final : public RangeDataProvider {
 public:
  NarrowGivens(absl::Span<Node* const> reverse_topo_sort, Node* target,
               absl::flat_hash_map<Node*, IntervalSet> intervals,
               const DependencyBitmap& interesting_nodes)
      : reverse_topo_sort_(reverse_topo_sort),
        target_(target),
        intervals_(std::move(intervals)),
        interesting_nodes_(interesting_nodes) {}

  std::optional<RangeData> GetKnownIntervals(Node* node) final {
    // TODO(allight): We might want to return base-intervals for nodes which
    // don't come from these calculated ones.
    if (intervals_.contains(node) && !intervals_.at(node).IsEmpty()) {
      return RangeData{
          .ternary = interval_ops::ExtractTernaryVector(intervals_.at(node)),
          .interval_set = IntervalSetTree::CreateSingleElementTree(
              node->GetType(), intervals_.at(node))};
    }
    return std::nullopt;
  }
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
  absl::flat_hash_map<Node*, IntervalSet> intervals_;
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
  NarrowGivens givens(reverse_topo_sort, next->value(), std::move(results),
                      dependencies);
  RangeQueryEngine contextual_range;
  XLS_RETURN_IF_ERROR(contextual_range.PopulateWithGivens(givens).status());
  return std::make_pair(contextual_range.GetTernary(next->value()).Get({}),
                        contextual_range.GetIntervals(next->value()).Get({}));
}

absl::Status RemoveLeadingBits(Param* param, const Value& orig_init_value,
                               const Bits& known_leading) {
  Value new_init_value(orig_init_value.bits().Slice(
      0, orig_init_value.bits().bit_count() - known_leading.bit_count()));
  ProcStateNarrowTransform transform(known_leading);
  return param->function_base()
      ->AsProcOrDie()
      ->TransformStateElement(param, new_init_value, transform)
      .status();
}

class SegmentRangeData : public RangeDataProvider {
 public:
  static absl::StatusOr<SegmentRangeData> Create(
      const NodeDependencyAnalysis& nda,
      const absl::flat_hash_map<Param*, RangeData>& ground_truth,
      Param* data_source, absl::Span<Node* const> topo_sort) {
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
    return (n->Is<Next>() && n->As<Next>()->param() == data_source_) ||
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
    if (node->Is<Param>()) {
      return node->GetType()->IsBits() &&
                     ground_truth_.contains(node->As<Param>())
                 ? std::make_optional(ground_truth_.at(node->As<Param>()))
                 : std::nullopt;
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
  SegmentRangeData(std::vector<DependencyBitmap> dependencies,
                   const absl::flat_hash_map<Param*, RangeData>& ground_truth,
                   Param* data_source, absl::Span<Node* const> topo_sort)
      : dependencies_(std::move(dependencies)),
        ground_truth_(ground_truth),
        data_source_(data_source),
        topo_sort_(topo_sort) {}
  std::vector<DependencyBitmap> dependencies_;
  const absl::flat_hash_map<Param*, RangeData>& ground_truth_;
  Param* data_source_;
  IntervalSet current_segments_;
  absl::Span<Node* const> topo_sort_;
};

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
    Proc* proc, Param* param, const IntervalSet& intervals,
    absl::Span<Node* const> topo_sort, const NodeDependencyAnalysis& nda,
    const absl::flat_hash_map<Param*, RangeData>& ground_truth) {
  XLS_ASSIGN_OR_RETURN(Value init_value, proc->GetInitValue(param));
  XLS_RET_CHECK(intervals.Covers(init_value.bits()))
      << "Invalid interval calculation for " << param << ".";
  VLOG(3) << "Doing segment walk for " << param << " on " << intervals;
  absl::flat_hash_set<Interval> remaining_intervals(
      intervals.Intervals().begin(), intervals.Intervals().end());
  Interval initial_interval = *absl::c_find_if(
      remaining_intervals,
      [&](const Interval& i) { return i.Covers(init_value.bits()); });
  remaining_intervals.erase(initial_interval);
  IntervalSet active_intervals = IntervalSet::Of({initial_interval});
  XLS_ASSIGN_OR_RETURN(
      SegmentRangeData limiter,
      SegmentRangeData::Create(nda, ground_truth, param, topo_sort));
  while (!remaining_intervals.empty()) {
    // Get the ranges of every node (which leads to a 'next' of the param)
    limiter.SetParamIntervals(active_intervals);
    RangeQueryEngine rqe;
    XLS_RETURN_IF_ERROR(rqe.PopulateWithGivens(limiter).status());

    // Get what this says all ranges are.
    IntervalSet run_intervals = active_intervals;
    for (Next* n : proc->next_values(param)) {
      // Nexts which don't update anything don't need to be taken into account.
      if (n->value() != n->param() &&
          (!n->predicate() || !rqe.IsAllZeros(*n->predicate()))) {
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
              param->GetType(), active_intervals)};
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
absl::StatusOr<absl::flat_hash_map<Param*, RangeData>> FindContextualRanges(
    Proc* proc, const QueryEngine& qe, const RangeQueryEngine& rqe,
    const NodeDependencyAnalysis& dependency_analysis,
    absl::Span<Node* const> reverse_topo_sort) {
  // List of all the next instructions that change the param for each param.
  absl::flat_hash_map<Param*, std::vector<Next*>> modifying_nexts_for_param;
  for (Param* param : proc->StateParams()) {
    // TODO(allight): Being able to narrow inside a compound value would be
    // nice. Since we unpack tuple state elements in other passes however the
    // actual impact would likely be negligible so no reason to bother with it
    // for now.
    if (!param->GetType()->IsBits()) {
      continue;
    }
    std::vector<Next*>& nexts = modifying_nexts_for_param[param];
    for (Next* n : proc->next_values(param)) {
      // TODO(allight): We might want to use data-flow to better track whether
      // things have changed. This should probably be good enough in practice
      // however.
      if (n->param() != n->value()) {
        nexts.push_back(n);
      }
    }
  }
  // To avoid issues where changes to the param values leads to invalidating the
  // TernaryQueryEngine we do all the modifications at the end.
  absl::flat_hash_map<Param*, RangeData> transforms;
  for (const auto& [orig_param, updates] : modifying_nexts_for_param) {
    if (updates.empty()) {
      // The state only has identity updates? Strange but this will be cleaned
      // up by NextValueOptimizationPass so we can ignore it.
      continue;
    }
    XLS_ASSIGN_OR_RETURN(Value orig_init_value, proc->GetInitValue(orig_param));
    TernaryVector possible_values =
        ternary_ops::BitsToTernary(orig_init_value.bits());

    IntervalSet contextual_intervals =
        IntervalSet::Precise(orig_init_value.bits());
    for (Next* next : updates) {
      TernaryVector context_free = qe.GetTernary(next->value()).Get({});
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
    transforms[orig_param] = RangeData{
        .ternary = possible_values,
        .interval_set = IntervalSetTree::CreateSingleElementTree(
            orig_param->GetType(), contextual_intervals),
    };
  }
  return transforms;
}

}  // namespace

// TODO(allight): Technically we'd probably want to run this whole pass to fixed
// point (incorporating the results into later runs) to get optimal results.
// It's not clear how much we'd gain there though. For now we will just run it
// once assuming that params are relatively independent of one
// another/additional information won't reveal more opportunities.
absl::StatusOr<bool> ProcStateNarrowingPass::RunOnProcInternal(
    Proc* proc, const OptimizationPassOptions& options,
    PassResults* results) const {
  // Find basic ternary limits
  TernaryQueryEngine tqe;
  // Use for more complicated range analysis.
  RangeQueryEngine rqe;
  UnownedUnionQueryEngine qe({&tqe, &rqe});
  XLS_RETURN_IF_ERROR(qe.Populate(proc).status());
  std::vector<Node*> reverse_topo_sort = ReverseTopoSort(proc);
  // Get the nodes which actually affect the next-value nodes. We don't really
  // care about anything else.
  NodeDependencyAnalysis next_node_sources =
      NodeDependencyAnalysis::BackwardDependents(
          proc,
          // Annoyingly absl doesn't seem to like conversion even when its
          // pointers.
          absl::MakeConstSpan(
              reinterpret_cast<Node* const*>(proc->next_values().begin()),
              proc->next_values().size()));

  // Data for doing state exploration. optional since we usually don't need them
  // so no need to create them.
  std::optional<std::vector<Node*>> topo_sort;

  XLS_ASSIGN_OR_RETURN(
      (absl::flat_hash_map<Param*, RangeData> initial_transforms),
      FindContextualRanges(proc, qe, rqe, next_node_sources,
                           reverse_topo_sort));
  absl::flat_hash_map<Param*, RangeData> final_transformation_list;
  for (const auto& [orig_param, t] : initial_transforms) {
    const auto& [ternary, interval_set] = t;
    int64_t known_leading =
        ternary_ops::ToKnownBits(*ternary).CountLeadingOnes();
    // If we have known leading bits from the ternary analysis use that. These
    // are usually good enough except with signed integer things.
    if (known_leading > 0) {
      VLOG(2) << "Narrowed " << orig_param << " to "
              << (orig_param->BitCountOrDie() - known_leading)
              << " bits (savings: " << known_leading
              << ") using back-prop/ternary. Interval is: "
              << interval_set.Get({});
      final_transformation_list[orig_param] = t;
      continue;
    }
    // We can't remove segments if there are no segments. Since there was also
    // no known high bits no need to do anything.
    if (interval_set.Get({}).Intervals().size() <= 1) {
      VLOG(2) << "Unable to narrow " << orig_param << ". "
              << (interval_set.Get({}).IsMaximal()
                      ? "Value is unconstrained."
                      : "High bits set and unable to walk segments due to "
                        "single interval.")
              << " Interval is: " << interval_set.Get({});
      continue;
    }
    // Try for signed value compression. We *only* do this if there are no
    // narrowings we can do without checking this. This is under the assumption
    // that in most cases the only thing that would case discontinuous range
    // results without having known high-bits is signed value comparisons
    // somewhere in the next-value predicates. This is likely to cause ranges
    // where there's a low bits region which is real and then all of the
    // negative numbers are also included.
    if (!topo_sort) {
      // Avoid creating updated range-query-engine & topo sort until required.
      topo_sort.emplace();
      topo_sort->reserve(reverse_topo_sort.size());
      absl::c_reverse_copy(reverse_topo_sort, std::back_inserter(*topo_sort));
      CHECK(topo_sort);
    }
    // Interval set is partitioned so we might be able to prove that no value
    // can move from one partition to another cutting down in the possible
    // values.
    XLS_ASSIGN_OR_RETURN(
        std::optional<RangeData> narrowed,
        NarrowUsingSegments(proc, orig_param, interval_set.Get({}), *topo_sort,
                            next_node_sources, initial_transforms));
    if (narrowed &&
        ternary_ops::ToKnownBits(*narrowed->ternary).CountLeadingOnes() > 1) {
      VLOG(2)
          << "Narrowed " << orig_param << " to "
          << (orig_param->BitCountOrDie() -
              ternary_ops::ToKnownBits(*narrowed->ternary).CountLeadingOnes())
          << " bits (savings: "
          << ternary_ops::ToKnownBits(*narrowed->ternary).CountLeadingOnes()
          << ") using segment walking. Interval is "
          << narrowed->interval_set.Get({});
      final_transformation_list[orig_param] = *narrowed;
    } else {
      VLOG(2) << "Unable to narrow " << orig_param
              << ". Segment walking unable to eliminate high bits. Interval is "
              << interval_set.Get({});
    }
  }

  bool made_changes = false;
  for (const auto& [orig_param, t] : final_transformation_list) {
    const auto& [ternary, _] = t;
    int64_t known_leading =
        ternary_ops::ToKnownBits(*ternary).CountLeadingOnes();
    XLS_RET_CHECK_GT(known_leading, 0);
    TernarySpan known_leading_tern =
        absl::MakeConstSpan(*ternary).last(known_leading);
    XLS_RET_CHECK(ternary_ops::IsFullyKnown(known_leading_tern));
    XLS_ASSIGN_OR_RETURN(Value orig_init_value, proc->GetInitValue(orig_param));
    XLS_RETURN_IF_ERROR(
        RemoveLeadingBits(orig_param, orig_init_value,
                          ternary_ops::ToKnownBitsValues(known_leading_tern)));
    made_changes = true;
  }

  return made_changes;
}

REGISTER_OPT_PASS(ProcStateNarrowingPass);

}  // namespace xls
