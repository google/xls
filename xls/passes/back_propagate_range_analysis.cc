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

#include "xls/passes/back_propagate_range_analysis.h"

#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_ops.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"
#include "xls/passes/range_query_engine.h"

namespace xls {

namespace {

// Class which can back-propagate node ranges.
//
// This walks the tree in reverse topo order updating values as it goes. It
// only visits a node if (1) it is possible to actually find more information
// based on that node and (2) the inputs to that node have updated information.
class BackPropagate : public DfsVisitorWithDefault {
 public:
  explicit BackPropagate(const RangeQueryEngine& query_engine,
                         absl::flat_hash_map<Node*, IntervalSet> givens)
      : query_engine_(query_engine), result_(std::move(givens)) {
    for (const auto& [node, _] : result_) {
      if (CanUpdateWith(node)) {
        waiting_to_see_.emplace(node);
      }
    }
  }

  // Note that we are handling some node. Used to allow us to bail out early.
  void NoteVisit(Node* n) { waiting_to_see_.erase(n); }
  // Bail out if we've seen all the givens already. That means there's no new
  // information to discover.
  bool CanContinue() const { return !waiting_to_see_.empty(); }
  // Will we be able to propagate information from this node into its
  // predecessors?
  bool CanUpdateWith(Node* n) {
    // need to have givens and be a computable operation.
    //
    // NB We could track through array and tuple but Dataflow will usually
    // eliminate those anyway so no need to bother.
    return result_.contains(n) && n->GetType()->IsBits() &&
           !result_[n].IsMaximal() &&
           absl::c_all_of(n->operands(),
                          [](Node* op) { return op->GetType()->IsBits(); }) &&
           n->OpIn({
               // Basic arithmetic can push down limits.
               Op::kAdd,
               Op::kSub,
               // Merge ranges
               Op::kEq,
               Op::kNe,
               Op::kUGt,
               Op::kUGe,
               Op::kULt,
               Op::kULe,
               Op::kSGt,
               Op::kSGe,
               Op::kSLt,
               Op::kSLe,
               // On some values we can distribute
               Op::kNot,
               Op::kAnd,
               Op::kNand,
               Op::kOr,
               Op::kNor,
               // Bits ops we can make use of on (at least some) some values
               Op::kConcat,
               Op::kSignExt,
               Op::kZeroExt,
               Op::kAndReduce,
               Op::kOrReduce,
           });
  }
  absl::Status DefaultHandler(Node* node) final { return absl::OkStatus(); }
  absl::Status HandleSignExtend(ExtendOp* ext) final {
    return UnifyExtend(ext);
  }
  absl::Status HandleZeroExtend(ExtendOp* ext) final {
    return UnifyExtend(ext);
  }
  absl::Status HandleAdd(BinOp* add) final { return UnifyMath(add); }
  absl::Status HandleSub(BinOp* sub) final { return UnifyMath(sub); }
  absl::Status HandleNot(UnOp* not_op) final {
    const IntervalSet& value = GetIntervals(not_op);
    if (value.IsEmpty()) {
      // We're already in an impossible case; don't try to do anything more.
      return absl::OkStatus();
    }
    if (value.IsMaximal()) {
      // If we have no bounds there's no additional info we can get by looking
      // at our arguments.
      return absl::OkStatus();
    }
    return MergeIn(not_op->operand(0), interval_ops::Not(value));
  }
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
    return UnifyAndLike(and_op);
  }
  absl::Status HandleNaryNand(NaryOp* nand_op) final {
    return UnifyAndLike(nand_op);
  }
  absl::Status HandleNaryNor(NaryOp* nor_op) final {
    return UnifyAndLike(nor_op);
  }
  absl::Status HandleNaryOr(NaryOp* or_op) final { return UnifyAndLike(or_op); }
  absl::Status HandleConcat(Concat* concat) final {
    IntervalSet end_intervals = GetIntervals(concat);
    if (!end_intervals.IsPrecise()) {
      // TODO: allight - Technically we can get some stuff from this (especially
      // around the higher bits) but it seems unlikely we'd get much of worth
      // since range-representation is not very good at representing individual
      // bits.
      return absl::OkStatus();
    }
    int64_t bit_off = 0;
    Bits real_value = *end_intervals.GetPreciseValue();
    for (int64_t i = 0; i < concat->operand_count(); ++i) {
      Node* operand = concat->operand(concat->operand_count() - (i + 1));
      int64_t operand_width = operand->GetType()->GetFlatBitCount();
      XLS_RETURN_IF_ERROR(MergeIn(
          operand,
          IntervalSet::Precise(real_value.Slice(bit_off, operand_width))));
      bit_off += operand_width;
    }
    return absl::OkStatus();
  }

  absl::Status HandleAndReduce(BitwiseReductionOp* and_reduce) final {
    return UnifyReductionOp(and_reduce);
  }
  absl::Status HandleOrReduce(BitwiseReductionOp* or_reduce) final {
    return UnifyReductionOp(or_reduce);
  }

  const absl::flat_hash_map<Node*, IntervalSet>& ranges() const& {
    return result_;
  }
  absl::flat_hash_map<Node*, IntervalSet>&& ranges() && {
    return std::move(result_);
  }

 private:
  // Get the current range of the given node, either from the givens, calculated
  // from a parent or from the base query-engine.
  IntervalSet GetIntervals(Node* node) {
    CHECK(node->GetType()->IsBits());
    if (result_.contains(node)) {
      return result_[node];
    }
    if (query_engine_.HasExplicitIntervals(node)) {
      // Try to avoid allocating LTTs needlessly.
      return query_engine_.GetIntervalSetTreeView(node)->Get({});
    }
    return query_engine_.GetIntervalSetTree(node).Get({});
  }

  // Merge the given 'new_data' with the already known facts about the given
  // node. Since this data is (transitively) based on the givens it is
  // Intersected with the existing knowledge.
  absl::Status MergeIn(Node* node, const IntervalSet& new_data) {
    XLS_RET_CHECK(node->GetType()->IsBits());
    XLS_RET_CHECK(new_data.IsNormalized());
    if (!result_.contains(node)) {
      result_[node] = query_engine_.GetIntervalSetTree(node).Get({});
    }
    IntervalSet old_data = std::move(result_[node]);
    result_[node] = IntervalSet::Intersect(old_data, new_data);
    if (result_[node] == old_data) {
      VLOG(3) << "Calculated range of " << node << " did not change.";
      return absl::OkStatus();
    }
    if (!CanUpdateWith(node)) {
      VLOG(3) << "Calculated range of " << node << " is now " << result_[node]
              << " but cannot propagate further.";
      return absl::OkStatus();
    }
    VLOG(3) << "Calculated range of " << node << " is now " << result_[node]
            << " and will propagate down";
    waiting_to_see_.emplace(node);
    return absl::OkStatus();
  }

  absl::Status UnifyReductionOp(BitwiseReductionOp* op) {
    XLS_RET_CHECK(op->OpIn({Op::kAndReduce, Op::kOrReduce}));
    IntervalSet res = GetIntervals(op);
    // Exactly 0/1 is the only place we can ever get anything.
    if (!res.IsPrecise()) {
      // Can't do anything
      return absl::OkStatus();
    }
    int64_t bit_count = op->operand(0)->BitCountOrDie();
    Bits argument_value;
    if (op->op() == Op::kAndReduce && res.CoversOne()) {
      // Only all-1s maps to 1 with and_reduce.
      argument_value = Bits::AllOnes(bit_count);
    } else if (op->op() == Op::kOrReduce && res.CoversZero()) {
      // Only 0 maps to 0 with or_reduce.
      argument_value = Bits(bit_count);
    } else {
      // Can't get anything. Mix of 1s and 0s but no additional info on which
      // particular bits are which.
      return absl::OkStatus();
    }
    return MergeIn(op->operand(0), IntervalSet::Precise(argument_value));
  }

  // Both sign and zero extend imply that the argument is just the truncation of
  // the extended value.
  absl::Status UnifyExtend(ExtendOp* ext) {
    return MergeIn(ext->operand(0),
                   interval_ops::Truncate(GetIntervals(ext),
                                          ext->operand(0)->BitCountOrDie()));
  }

  // Propagate ranges through a math +/- expression.
  absl::Status UnifyMath(Node* math) {
    // Mul and Div lose too much information to be easily reversible.
    XLS_RET_CHECK(math->OpIn({Op::kAdd, Op::kSub})) << math << " not supported";
    IntervalSet res = GetIntervals(math);
    IntervalSet l_interval = GetIntervals(math->operand(0));
    IntervalSet r_interval = GetIntervals(math->operand(1));
    IntervalSet new_l;
    IntervalSet new_r;
    switch (math->op()) {
      // We can prove that iterating this logic is a no-op; on repetition, we
      // would just end up intersecting with supersets of the original
      // r_intervals and l_intervals.
      case Op::kAdd: {
        // l + r = res
        // l = res - r
        // r = res - l
        new_l = interval_ops::Sub(res, r_interval);
        new_r = interval_ops::Sub(res, l_interval);
        break;
      }
      case Op::kSub: {
        // l - r = res
        // l = res + r
        // -r = res - l
        // r = l - res
        new_l = interval_ops::Add(res, r_interval);
        new_r = interval_ops::Sub(l_interval, res);
        break;
      }
      default:
        return absl::InternalError("Unexpected op");
    }
    XLS_RETURN_IF_ERROR(MergeIn(math->operand(0), new_l));
    XLS_RETURN_IF_ERROR(MergeIn(math->operand(1), new_r));
    return absl::OkStatus();
  }

  // Propagate values through a comparison.
  absl::Status UnifyComparison(CompareOp* cmp) {
    XLS_RET_CHECK(cmp->op() == Op::kSLe || cmp->op() == Op::kSLt ||
                  cmp->op() == Op::kSGe || cmp->op() == Op::kSGt ||
                  cmp->op() == Op::kULe || cmp->op() == Op::kULt ||
                  cmp->op() == Op::kUGe || cmp->op() == Op::kUGt)
        << cmp;
    IntervalSet cmp_intervals = GetIntervals(cmp);
    if (!cmp_intervals.IsPrecise()) {
      // Not able to propagate down any more.
      return absl::OkStatus();
    }
    // Normalize the comparison to be a 'true' result of either the form `X > Y`
    // or the form `X >= Y`.
    XLS_ASSIGN_OR_RETURN(Op invert, InvertComparisonOp(cmp->op()));
    // If the operation results in false invert it.
    Op op = cmp_intervals.CoversOne() ? cmp->op() : invert;
    Node* l_op = cmp->operand(0);
    Node* r_op = cmp->operand(1);
    // If the operation is a less-than (<) reverse the operand order and swap
    // the comparison.
    if (op == Op::kSLt || op == Op::kSLe || op == Op::kULt || op == Op::kULe) {
      std::swap(l_op, r_op);
      XLS_ASSIGN_OR_RETURN(op, ReverseComparisonOp(op));
    }
    return UnifyTrueComparison(l_op, GetIntervals(l_op), r_op,
                               GetIntervals(r_op), op);
  }

  // Analyze a normalized comparison which is a 'true' greater-than or
  // greater-than-or-equal operation.
  absl::Status UnifyTrueComparison(Node* left, const IntervalSet& left_range,
                                   Node* right, const IntervalSet& right_range,
                                   Op op) {
    CHECK(op == Op::kUGe || op == Op::kUGt || op == Op::kSGe || op == Op::kSGt)
        << op;
    if (left_range.IsEmpty() || right_range.IsEmpty()) {
      // We're already in an impossible case; don't try to do anything more.
      return absl::OkStatus();
    }
    // Perform the comparison for specifically unsigned operations.
    auto range_unsigned = [](Op op, const IntervalSet& left_range,
                             const IntervalSet& right_range)
        -> std::pair<IntervalSet, IntervalSet> {
      // NB Result is TRUE.
      CHECK(op == Op::kUGe || op == Op::kUGt);
      if (op == Op::kUGe) {
        // left >= right
        // left can't be lower than right.LowerBound
        IntervalSet new_left = IntervalSet::Intersect(
            IntervalSet::Of(
                {Interval::Closed(*right_range.LowerBound(),
                                  Bits::AllOnes(left_range.BitCount()))}),
            left_range);
        // right can't be higher than left.UpperBound
        IntervalSet new_right = IntervalSet::Intersect(
            IntervalSet::Of({Interval::Closed(Bits(right_range.BitCount()),
                                              *left_range.UpperBound())}),
            right_range);
        return {new_left, new_right};
      }
      // left > right
      // left can't be lower than right.LowerBound
      IntervalSet new_left = IntervalSet::Intersect(
          IntervalSet::Of(
              {Interval::LeftOpen(*right_range.LowerBound(),
                                  Bits::AllOnes(left_range.BitCount()))}),
          left_range);
      // right can't be higher than left.UpperBound
      IntervalSet new_right = IntervalSet::Intersect(
          IntervalSet::Of({Interval::RightOpen(Bits(right_range.BitCount()),
                                               *left_range.UpperBound())}),
          right_range);
      return {new_left, new_right};
    };
    IntervalSet new_left;
    IntervalSet new_right;
    switch (op) {
      case Op::kUGt:
      case Op::kUGe: {
        std::tie(new_left, new_right) =
            range_unsigned(op, left_range, right_range);
        break;
      }
      case Op::kSGe:
      case Op::kSGt: {
        // Scale interval up so INT_MIN == 0. Overflows negative to be less than
        // positive unsigned. The interval_ops ensure that this operation is
        // reversible - modulo precision losses due to potentially splitting
        // ranges. We could extend the interval_ops::Add/Sub functions to have
        // them ignore the precision limit and operate with perfect precision
        // but that seems unlikely to produce enough benefits to be worth the
        // performance cost at the moment.
        IntervalSet offset =
            IntervalSet::Precise(Bits::MinSigned(left_range.BitCount()));
        XLS_ASSIGN_OR_RETURN(Op unsigned_op, SignedCompareToUnsigned(op));
        auto [scaled_new_left, scaled_new_right] =
            range_unsigned(unsigned_op, interval_ops::Add(offset, left_range),
                           interval_ops::Add(offset, right_range));
        new_left = interval_ops::Sub(scaled_new_left, offset);
        new_right = interval_ops::Sub(scaled_new_right, offset);
        break;
      }
      default:
        return absl::InternalError("Unexpected op");
    }
    XLS_RETURN_IF_ERROR(MergeIn(left, new_left));
    XLS_RETURN_IF_ERROR(MergeIn(right, new_right));
    return absl::OkStatus();
  }

  absl::Status UnifyExactMatch(CompareOp* eq) {
    XLS_RET_CHECK(eq->GetType()->GetFlatBitCount() == 1);
    CHECK(eq->op() == Op::kEq || eq->op() == Op::kNe) << eq;
    // Implies that ranges must be the same (since otherwise there would be
    // some numbers where condition is false)
    Node* a = eq->operand(0);
    Node* b = eq->operand(1);
    if (!a->GetType()->IsBits() || !b->GetType()->IsBits()) {
      // We don't want to have to deal with non-bits types and we learn very
      // little anyway since dataflow removes most of them.
      return absl::OkStatus();
    }
    IntervalSet eq_interval = GetIntervals(eq);
    if (!eq_interval.IsPrecise()) {
      // Don't know the result of this computation. Can't continue.
      return absl::OkStatus();
    }
    IntervalSet a_intervals = GetIntervals(a);
    IntervalSet b_intervals = GetIntervals(b);

    if (eq_interval.CoversOne() == (eq->op() == Op::kEq)) {
      // Case: (L == R) == TRUE
      // Case: (L != R) == FALSE
      IntervalSet unified = IntervalSet::Intersect(a_intervals, b_intervals);

      if (unified.NumberOfIntervals() == 0) {
        // This implies the condition is actually unreachable impossible
        // (since we unify to bottom on an element). For now just leave
        // unconstrained and continue.
        // TODO(allight): 2023-09-25: We can do better and should probably try
        // to communicate and remove the impossible cases here. This would
        // need to be done in narrowing or strength reduction by removing the
        // associated branches.
        return absl::OkStatus();
      }
      XLS_RETURN_IF_ERROR(MergeIn(a, unified));
      XLS_RETURN_IF_ERROR(MergeIn(b, unified));
    } else {
      // Case: (L == R) == FALSE
      // Case: (L != R) == TRUE
      // Basically only have any information if a or b is precise.
      if (a_intervals.IsPrecise() || b_intervals.IsPrecise()) {
        Node* precise = a_intervals.IsPrecise() ? a : b;
        const IntervalSet& precise_intervals =
            precise == a ? a_intervals : b_intervals;
        Node* imprecise = precise == a ? b : a;

        IntervalSet imprecise_complement_interval =
            IntervalSet::Complement(imprecise == a ? a_intervals : b_intervals);
        // Remove the single known precise value from the imprecise values
        // // range.
        imprecise_complement_interval.AddInterval(
            precise_intervals.Intervals().front());
        imprecise_complement_interval.Normalize();
        IntervalSet imprecise_interval =
            IntervalSet::Complement(imprecise_complement_interval);
        if (imprecise_interval.NumberOfIntervals() == 0) {
          // This implies the condition is actually unreachable
          // (since we unify to bottom on some element). For now just leave
          // unconstrained.
          // TODO(allight): 2023-09-25: We can do better and should probably try
          // to communicate and remove the impossible cases here. This would
          // need to be done in narrowing or strength reduction by removing the
          // associated branches.
          return absl::OkStatus();
        }
        XLS_RETURN_IF_ERROR(MergeIn(imprecise, imprecise_interval));
      } else {
        // TODO(allight): 2023-08-10 Technically there is information to be
        // gleaned here if |L \intersect R| == 1 but probably not worth it. For
        // now just ignore this case.
        return absl::OkStatus();
      }
    }
    return absl::OkStatus();
  }

  // Operations that, like the boolean and operation, match the behavior where
  // there are some values X & Y such that (eq? (op A B C ...) X) iff (all-of?
  // (eq? A Y) (eq? B Y) (eq? C Y) ...). This is true of 'and' where (eq? (and A
  // B C) #t) iff A B & C are all #t.
  absl::Status UnifyAndLike(NaryOp* and_op) {
    XLS_RET_CHECK(and_op->OpIn({Op::kAnd, Op::kNor, Op::kOr, Op::kNand}));
    IntervalSet interval = GetIntervals(and_op);
    if (interval.BitCount() != 1 || !interval.IsPrecise()) {
      // We could analyze higher bit counts but we often wouldn't get very much
      // (range analysis does not work well with bit-vectors). If the range is
      // not precise we of course cannot say anything at all about the inputs.
      return absl::OkStatus();
    }
    switch (and_op->op()) {
      case Op::kAnd: {
        if (interval.CoversOne()) {
          return UnifyAllOperandsMatch(and_op, /*boolean_value_is=*/true);
        }
        // (and x y z) == false doesn't imply anything except that at least one
        // is false, so we can't continue.
        return absl::OkStatus();
      }
      case Op::kOr: {
        if (interval.CoversZero()) {
          return UnifyAllOperandsMatch(and_op, /*boolean_value_is=*/false);
        }
        return absl::OkStatus();
      }
      case Op::kNand: {
        if (interval.CoversZero()) {
          return UnifyAllOperandsMatch(and_op, /*boolean_value_is=*/true);
        }
        return absl::OkStatus();
      }
      case Op::kNor: {
        if (interval.CoversOne()) {
          return UnifyAllOperandsMatch(and_op, /*boolean_value_is=*/false);
        }
        return absl::OkStatus();
      }
      default:
        return absl::InternalError("unsuported op");
    }
  }

  absl::Status UnifyAllOperandsMatch(NaryOp* op, bool boolean_value_is) {
    IntervalSet value =
        IntervalSet::Precise(boolean_value_is ? UBits(1, 1) : UBits(0, 1));
    for (Node* operand : op->operands()) {
      XLS_RETURN_IF_ERROR(MergeIn(operand, value));
    }
    return absl::OkStatus();
  }

  // Underlying query-engine providing base ranges.
  const RangeQueryEngine& query_engine_;
  // Set of all givens and any calculated refined ranges.
  absl::flat_hash_map<Node*, IntervalSet> result_;
  // Set of nodes which we have updated data for which might be possible to
  // propagate.
  absl::flat_hash_set<Node*> waiting_to_see_;
};

}  // namespace

absl::StatusOr<absl::flat_hash_map<Node*, IntervalSet>>
PropagateGivensBackwards(
    const RangeQueryEngine& engine, FunctionBase* function,
    absl::flat_hash_map<Node*, IntervalSet> givens,
    std::optional<absl::Span<Node* const>> reverse_topo_sort) {
  XLS_RET_CHECK(!givens.empty());
  BackPropagate prop(engine, std::move(givens));
  std::vector<Node*> nodes_mem =
      reverse_topo_sort ? std::vector<Node*>{} : ReverseTopoSort(function);
  absl::Span<Node* const> nodes =
      reverse_topo_sort.value_or(absl::MakeConstSpan(nodes_mem));
  for (Node* n : nodes) {
    // Check if we've got any more nodes with updated values we can propagate.
    if (!prop.CanContinue()) {
      break;
    }
    // Remove this node from the waiting-to-visit set.
    prop.NoteVisit(n);
    if (prop.CanUpdateWith(n)) {
      VLOG(3) << "Propagating backwards through " << n;
      XLS_RETURN_IF_ERROR(n->VisitSingleNode(&prop));
    }
  }
  return std::move(prop).ranges();
}

absl::StatusOr<absl::flat_hash_map<Node*, IntervalSet>>
PropagateOneGivenBackwards(const RangeQueryEngine& engine, Node* node,
                           const Bits& given) {
  return PropagateOneGivenBackwards(engine, node, IntervalSet::Precise(given));
}

}  // namespace xls
