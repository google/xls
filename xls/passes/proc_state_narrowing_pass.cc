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
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
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

absl::StatusOr<std::optional<TernaryVector>> ExtractContextSensitiveRange(
    Proc* proc, Next* next, RangeQueryEngine& rqe,
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
  return contextual_range.GetTernary(next->value()).Get({});
}

}  // namespace

absl::StatusOr<bool> ProcStateNarrowingPass::RunOnProcInternal(
    Proc* proc, const OptimizationPassOptions& options,
    PassResults* results) const {
  // Find basic ternary limits
  TernaryQueryEngine tqe;
  // Use for more complicated range analysis.
  RangeQueryEngine rqe;
  UnownedUnionQueryEngine qe({&tqe, &rqe});
  XLS_RETURN_IF_ERROR(qe.Populate(proc).status());
  // Get the nodes which actually affect the next-value nodes. We don't really
  // care about anything else.
  NodeDependencyAnalysis dependency_analysis =
      NodeDependencyAnalysis::BackwardDependents(
          // Annoyingly absl doesn't seem to like conversion even when its
          // pointers.
          proc, absl::MakeConstSpan(
                    reinterpret_cast<Node* const*>(proc->next_values().begin()),
                    proc->next_values().size()));

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
  bool made_changes = false;
  // To avoid issues where changes to the param values leads to invalidating the
  // TernaryQueryEngine we do all the modifications at the end.
  struct ToTransform {
    Param* orig_param;
    Value new_init_value;
    ProcStateNarrowTransform transformer;
  };
  std::vector<ToTransform> transforms;
  std::vector<Node*> reverse_topo_sort = ReverseTopoSort(proc);
  for (const auto& [orig_param, updates] : modifying_nexts_for_param) {
    if (updates.empty()) {
      // The state only has identity updates? Strange but this will be cleaned
      // up by NextValueOptimizationPass so we can ignore it.
      continue;
    }
    XLS_ASSIGN_OR_RETURN(Value orig_init_value, proc->GetInitValue(orig_param));
    TernaryVector possible_values =
        ternary_ops::BitsToTernary(orig_init_value.bits());
    for (Next* next : updates) {
      TernaryVector context_free = qe.GetTernary(next->value()).Get({});
      TernaryVector restricted_value;
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
            std::optional<TernaryVector> contextual_range,
            ExtractContextSensitiveRange(proc, next, rqe, reverse_topo_sort,
                                         dependency_analysis),
            _ << next);
        restricted_value =
            std::move(contextual_range).value_or(std::move(context_free));
      } else {
        restricted_value = std::move(context_free);
      }
      possible_values =
          ternary_ops::Intersection(possible_values, restricted_value);
    }
    int64_t initial_width = possible_values.size();
    int64_t known_leading =
        ternary_ops::ToKnownBits(possible_values).CountLeadingOnes();
    if (known_leading == 0) {
      continue;
    }

    transforms.push_back(
        {.orig_param = orig_param,
         // Remove the known leading bits from the proc state.
         .new_init_value = Value(
             orig_init_value.bits().Slice(0, initial_width - known_leading)),
         .transformer = ProcStateNarrowTransform(ternary_ops::ToKnownBitsValues(
             absl::MakeSpan(possible_values)
                 .subspan(initial_width - known_leading)))});
  }
  for (auto t : std::move(transforms)) {
    XLS_RETURN_IF_ERROR(proc->TransformStateElement(
                                t.orig_param, t.new_init_value, t.transformer)
                            .status());
    made_changes = true;
  }

  return made_changes;
}

REGISTER_OPT_PASS(ProcStateNarrowingPass);

}  // namespace xls
