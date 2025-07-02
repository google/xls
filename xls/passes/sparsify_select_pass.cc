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

#include "xls/passes/sparsify_select_pass.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function_base.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/partial_info_query_engine.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"

namespace xls {

// Returns a vector of the intervals contained within the given `IntervalSet`,
// sorted from smallest to largest in terms of number of points covered.
static std::vector<Interval> IntervalsSortedBySize(IntervalSet set) {
  set.Normalize();
  std::vector<Interval> intervals(set.Intervals().begin(),
                                  set.Intervals().end());
  // This is needed because we want to be deterministic when there are ties.
  // For example, if we had {[2, 2], [5, 5], [6, 6]} (all size 1) and we used
  // `std::sort`, the generated code would not be deterministic.
  std::stable_sort(intervals.begin(), intervals.end(),
                   [](const Interval& a, const Interval& b) -> bool {
                     return bits_ops::ULessThan(a.SizeBits(), b.SizeBits());
                   });
  return intervals;
}

// Given a select and a set of intervals that the selector of the select is
// guaranteed to be in, replace the select with a new nodes that only use the
// cases that are covered by the intervals given.
//
// Currently, this builds a chain of selects, each of which determines whether
// or not the given element is within one of the given intervals. If it is, it
// dispatches to another select that determines which element it is within the
// interval. Otherwise, it dispatches to the next select in the chain.
//
// TODO(taktoa): build a binary search tree based on interval lower bounds
// rather than a linear chain.
static absl::Status SparsifySelect(FunctionBase* f, Select* select,
                                   const IntervalSet& selector_intervals) {
  // As we build up the select chain, this represents the rest of the chain,
  // i.e.: what we should use when the current select we are adding is false.
  Node* other = nullptr;
  // Are we at the bottom of the select chain? If so, we don't need to check
  // whether the selector is in range; it must be in the remaining interval.
  bool is_leaf = true;

  for (const Interval& interval : IntervalsSortedBySize(selector_intervals)) {
    std::vector<Node*> cases_in_range;

    {
      absl::Status failure = absl::OkStatus();
      interval.ForEachElement([&](const Bits& bits) -> bool {
        std::optional<uint64_t> index =
            bits.FitsInUint64() ? std::make_optional(*bits.ToUint64())
                                : std::nullopt;
        if (index.has_value() && *index < select->cases().size()) {
          cases_in_range.push_back(select->cases()[*index]);
        } else if (select->default_value().has_value()) {
          cases_in_range.push_back(select->default_value().value());
        } else {
          failure = absl::InternalError(
              "SparsifySelectPass: select had not enough cases and no default");
          return true;
        }
        return false;
      });
      XLS_RETURN_IF_ERROR(failure);
    }

    XLS_ASSIGN_OR_RETURN(
        Node * lower_bound,
        f->MakeNode<Literal>(select->loc(), Value(interval.LowerBound())));
    XLS_ASSIGN_OR_RETURN(
        Node * upper_bound,
        f->MakeNode<Literal>(select->loc(), Value(interval.UpperBound())));
    XLS_ASSIGN_OR_RETURN(Node * selector_minus_lower_bound,
                         f->MakeNode<BinOp>(select->loc(), select->selector(),
                                            lower_bound, Op::kSub));

    // TODO(taktoa): would be good to use assume(false) here instead of 0
    XLS_ASSIGN_OR_RETURN(
        Node * zero,
        f->MakeNode<Literal>(select->loc(), ZeroOfType(select->GetType())));

    XLS_ASSIGN_OR_RETURN(
        Node * if_in_range,
        f->MakeNode<Select>(select->loc(), selector_minus_lower_bound,
                            cases_in_range, /*default_value=*/zero));

    if (is_leaf) {
      other = if_in_range;
    } else {
      XLS_ASSIGN_OR_RETURN(
          Node * ge_lower_bound,
          f->MakeNode<CompareOp>(select->loc(), select->selector(), lower_bound,
                                 Op::kUGe));
      XLS_ASSIGN_OR_RETURN(
          Node * le_upper_bound,
          f->MakeNode<CompareOp>(select->loc(), select->selector(), upper_bound,
                                 Op::kULe));
      XLS_ASSIGN_OR_RETURN(
          Node * is_in_range,
          f->MakeNode<NaryOp>(
              select->loc(),
              std::vector<Node*>({ge_lower_bound, le_upper_bound}), Op::kAnd));

      XLS_ASSIGN_OR_RETURN(
          other, f->MakeNode<Select>(select->loc(), is_in_range,
                                     std::vector<Node*>({other, if_in_range}),
                                     /*default_value=*/std::nullopt));
    }

    is_leaf = false;
  }

  if (other != nullptr) {
    XLS_RETURN_IF_ERROR(select->ReplaceUsesWith(other));
  }

  return absl::OkStatus();
}

absl::StatusOr<bool> SparsifySelectPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  PartialInfoQueryEngine* engine =
      context.SharedQueryEngine<PartialInfoQueryEngine>(f);
  XLS_RETURN_IF_ERROR(engine->Populate(f).status());

  bool changed = false;
  for (Node* node : context.TopoSort(f)) {
    if (node->Is<Select>()) {
      Select* select = node->As<Select>();
      Node* selector = select->selector();
      IntervalSetTree selector_ist = engine->GetIntervals(selector);
      IntervalSet selector_intervals = selector_ist.Get({});
      if (std::optional<int64_t> size = selector_intervals.Size()) {
        if (size >= select->cases().size()) {
          continue;
        }
      } else {
        continue;
      }

      changed = true;
      XLS_RETURN_IF_ERROR(SparsifySelect(f, select, selector_intervals));
    }
  }

  return changed;
}

}  // namespace xls
