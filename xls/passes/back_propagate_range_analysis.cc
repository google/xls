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

#include <array>
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/type.h"
#include "xls/passes/range_query_engine.h"

namespace xls {

namespace {

// A canonical representation of a range check holding the low-boundary,
// variable, and high-boundary.
struct CanonicalRange {
  // The low value which 'param' is compared to.
  Node* low_value;
  // The cmp used to compare low-value with param. Executed as `(low_cmp
  // param low_value)`. That is low_value is on the left. This is one of 'SGt',
  // 'UGt', 'SGe', or 'UGe'.
  Op low_cmp;
  // The parameter which is being constrained by the range.
  Node* param;
  // The cmp used to compare low-value with param. Executed as `(high_cmp
  // param high_value)`. That is low_value is on the left. This is one of 'SLt',
  // 'ULt', 'SLe', or 'ULe'.
  Op high_cmp;
  // The high value which 'param' is compared to.
  Node* high_value;

  // The actual instruction which implements the (low_cmp low_value param)
  // operation.
  CompareOp* low_range;
  // The actual instruction which implements the (high_cmp high_value param)
  // operation.
  CompareOp* high_range;
};

// Class which can back-propagate node ranges.
//
// This is currently limited to a single step.
class BackPropagate : public DfsVisitorWithDefault {
 public:
  explicit BackPropagate(const RangeQueryEngine& query_engine)
      : query_engine_(query_engine) {}
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
    return MaybeUnifyAnd(and_op);
  }
  // absl::Status HandleNaryOr(NaryOp* or_op) final {
  //   return MaybeUnifyOr(or_op);
  // }

  const absl::flat_hash_map<Node*, IntervalSet>& ranges() const& {
    return result_;
  }
  absl::flat_hash_map<Node*, IntervalSet>&& ranges() && {
    return std::move(result_);
  }
  void AddGiven(Node* n, IntervalSet data) { result_[n] = std::move(data); }

 private:
  absl::Status UnifyComparison(CompareOp* cmp) {
    XLS_RET_CHECK(cmp->op() == Op::kSLe || cmp->op() == Op::kSLt ||
                  cmp->op() == Op::kSGe || cmp->op() == Op::kSGt ||
                  cmp->op() == Op::kULe || cmp->op() == Op::kULt ||
                  cmp->op() == Op::kUGe || cmp->op() == Op::kUGt)
        << cmp;
    XLS_RET_CHECK(result_[cmp].IsPrecise())
        << "selector " << cmp
        << " not given actual value during context sensitive range analysis!";
    // Standardize so we are assuming the comparison is true.
    XLS_ASSIGN_OR_RETURN(Op invert, InvertComparisonOp(cmp->op()));
    Op op = result_[cmp].CoversOne() ? cmp->op() : invert;
    Node* l_op = cmp->operand(0);
    Node* r_op = cmp->operand(1);
    IntervalSet l_interval = query_engine_.GetIntervalSetTree(l_op).Get({});
    IntervalSet r_interval = query_engine_.GetIntervalSetTree(r_op).Get({});
    if (!l_interval.IsPrecise() && !r_interval.IsPrecise()) {
      return UnifyImpreciseComparison(l_op, r_op, l_interval, r_interval);
    }
    // Standardize so right side is always precise.
    bool is_signed =
        op == Op::kSLe || op == Op::kSLt || op == Op::kSGe || op == Op::kSGt;
    bool is_or_equals =
        op == Op::kULe || op == Op::kUGe || op == Op::kSLe || op == Op::kSGe;
    bool is_less_than =
        op == Op::kSLe || op == Op::kULe || op == Op::kSLt || op == Op::kULt;
    if (l_interval.IsPrecise()) {
      // We want to ensure that the constant is always on the right to simplify
      // UnifyLiteralComparison. This requires doing a transform 'ReverseOp'
      // such that '(op L R) == ((ReverseOp op) R L)'. The transform that works
      // for this is replacing '</<=' with '>/>=' and vice versa.
      is_less_than = !is_less_than;
      std::swap(l_op, r_op);
      std::swap(l_interval, r_interval);
    }
    return UnifyLiteralComparison(l_op, l_interval,
                                  r_interval.GetPreciseValue().value(),
                                  is_or_equals, is_less_than, is_signed);
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
    IntervalSet a_intervals = query_engine_.GetIntervalSetTree(a).Get({});
    IntervalSet b_intervals = query_engine_.GetIntervalSetTree(b).Get({});

    if (result_[eq].CoversOne() == (eq->op() == Op::kEq)) {
      // Case: (L == R) == TRUE
      // Case: (L != R) == FALSE
      IntervalSet unified = IntervalSet::Intersect(a_intervals, b_intervals);

      if (unified.NumberOfIntervals() == 0) {
        // This implies the condition is actually unreachable impossible
        // (since we unify to bottom on an element). For now just leave
        // unconstrained.
        // TODO(allight): 2023-09-25: We can do better and should probably try
        // to communicate and remove the impossible cases here. This would
        // need to be done in narrowing or strength reduction by removing the
        // associated branches.
        return absl::OkStatus();
      }
      result_[a] = unified;
      result_[b] = unified;
    } else {
      // Case: (L == R) == FALSE
      // Case: (L != R) == TRUE
      // Basically only have any information if a or b is precise.
      if (a_intervals.IsPrecise() || b_intervals.IsPrecise()) {
        Node* precise = a_intervals.IsPrecise() ? a : b;
        const IntervalSet& precise_intervals =
            precise == a ? a_intervals : b_intervals;
        Node* imprecise = precise == a ? b : a;

        IntervalSet imprecise_complement_interval = IntervalSet::Complement(
            query_engine_.GetIntervalSetTree(imprecise).AsView().Get({}));
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
        result_[imprecise] = imprecise_interval;
      } else {
        // TODO(allight): 2023-08-10 Technically there is information to be
        // gleaned here if |L \intersect R| == 1 but probably not worth it. For
        // now just ignore this case.
        return absl::OkStatus();
      }
    }
    return absl::OkStatus();
  }

  absl::Status MaybeUnifyAnd(NaryOp* and_op) {
    // To simplify the unification logic we only handle a single 'range' op
    // This is to catch the dslx a..b match type.
    // TODO(allight): 2023-09-07 We could do better in the positive case since
    // we know everything is true.
    if (and_op->operand_count() != 2) {
      return absl::OkStatus();
    }
    auto canonical_range = ExtractRange(and_op->operand(0), and_op->operand(1));
    if (canonical_range) {
      return UnifyRangeComparison(*canonical_range,
                                  result_[and_op].CoversOne());
    }
    return absl::OkStatus();
  }

  // Extract the CanonicalRange comparison out of the two and'd comparisons.
  //
  // Returns nullopt if the elements do not form a range check.
  std::optional<CanonicalRange> ExtractRange(Node* element_one,
                                             Node* element_two) {
    const std::array<Op, 8> cmp_ops{
        Op::kSLe, Op::kSLt, Op::kSGe, Op::kSGt,
        Op::kULe, Op::kULt, Op::kUGe, Op::kUGt,
    };
    if (!element_one->OpIn(cmp_ops) || !element_two->OpIn(cmp_ops)) {
      return std::nullopt;
    }
    // canonicalize both to
    // (<OP> <COMMON> <DIFFERENT>)
    // A range check is 'x in range(start, end)' this is represented in the IR
    // as (and (< start x) (< x end)). To simplify handling we make both ends
    // ordered in the 'x' 'start/end' direction.
    Op e1_op;
    Op e2_op;
    Node* e1_comparator;
    Node* e2_comparator;
    Node* common;
    if (element_one->operand(0) == element_two->operand(0)) {
      // Already in canonical order
      common = element_one->operand(0);
      e1_op = element_one->op();
      e2_op = element_two->op();
      e1_comparator = element_one->operand(1);
      e2_comparator = element_two->operand(1);
    } else if (element_one->operand(1) == element_two->operand(0)) {
      // element2 in canonical order
      common = element_one->operand(1);
      e1_op = *ReverseComparisonOp(element_one->op());
      e2_op = element_two->op();
      e1_comparator = element_one->operand(0);
      e2_comparator = element_two->operand(1);
    } else if (element_one->operand(0) == element_two->operand(1)) {
      // element1 in canonical order
      common = element_one->operand(0);
      e1_op = element_one->op();
      e2_op = *ReverseComparisonOp(element_two->op());
      e1_comparator = element_one->operand(1);
      e2_comparator = element_two->operand(0);
    } else if (element_one->operand(1) == element_two->operand(1)) {
      // both in reversed order
      common = element_one->operand(1);
      e1_op = *ReverseComparisonOp(element_one->op());
      e2_op = *ReverseComparisonOp(element_two->op());
      e1_comparator = element_one->operand(0);
      e2_comparator = element_two->operand(0);
    } else {
      // Not a range, no common comparator.
      return std::nullopt;
    }
    // order the operations
    std::array<Op, 4> low_ops{Op::kSGe, Op::kSGt, Op::kUGe, Op::kUGt};
    std::array<Op, 4> high_ops{Op::kSLe, Op::kSLt, Op::kULe, Op::kULt};
    if (absl::c_find(low_ops, e1_op) != low_ops.cend() &&
        absl::c_find(high_ops, e2_op) != high_ops.cend()) {
      return CanonicalRange{
          .low_value = e1_comparator,
          .low_cmp = e1_op,
          .param = common,
          .high_cmp = e2_op,
          .high_value = e2_comparator,
          .low_range = element_one->As<CompareOp>(),
          .high_range = element_two->As<CompareOp>(),
      };
    }
    if (absl::c_find(high_ops, e1_op) != high_ops.cend() &&
        absl::c_find(low_ops, e2_op) != low_ops.cend()) {
      return CanonicalRange{
          .low_value = e2_comparator,
          .low_cmp = e2_op,
          .param = common,
          .high_cmp = e1_op,
          .high_value = e1_comparator,
          .low_range = element_two->As<CompareOp>(),
          .high_range = element_one->As<CompareOp>(),
      };
    }
    return std::nullopt;
  }

 private:
  // Extract interval sets from the range given the range check succeeds or
  // fails (value_is_in_range).
  absl::Status UnifyRangeComparison(const CanonicalRange& range,
                                    bool value_is_in_range) {
    IntervalSet low_interval =
        query_engine_.GetIntervalSetTree(range.low_value).Get({});
    IntervalSet high_interval =
        query_engine_.GetIntervalSetTree(range.high_value).Get({});
    IntervalSet base_interval =
        query_engine_.GetIntervalSetTree(range.param).Get({});
    bool left_is_open = range.low_cmp == Op::kSGt || range.low_cmp == Op::kUGt;
    bool right_is_open =
        range.high_cmp == Op::kSLt || range.high_cmp == Op::kULt;
    IntervalSet range_interval(base_interval.BitCount());
    if (left_is_open && right_is_open) {
      range_interval.AddInterval(Interval::Open(*low_interval.LowerBound(),
                                                *high_interval.UpperBound()));
    } else if (left_is_open && !right_is_open) {
      range_interval.AddInterval(Interval::LeftOpen(
          *low_interval.LowerBound(), *high_interval.UpperBound()));
    } else if (!left_is_open && right_is_open) {
      range_interval.AddInterval(Interval::RightOpen(
          *low_interval.LowerBound(), *high_interval.UpperBound()));
    } else {
      range_interval.AddInterval(Interval::Closed(*low_interval.LowerBound(),
                                                  *high_interval.UpperBound()));
    }
    range_interval.Normalize();
    if (value_is_in_range) {
      // Value is in range, intersect with range.
      IntervalSet true_range = IntervalSet::Precise(Bits::AllOnes(1));
      result_[range.low_range] = true_range;
      result_[range.high_range] = true_range;
      IntervalSet constrained_param =
          IntervalSet::Intersect(base_interval, range_interval);
      result_[range.param] = constrained_param;
    } else {
      // Outside of range, add the inverse.
      IntervalSet constrained_param = IntervalSet::Intersect(
          base_interval, IntervalSet::Complement(range_interval));
      result_[range.param] = constrained_param;
    }
    return absl::OkStatus();
  }

  absl::Status UnifyLiteralComparison(Node* variable, const IntervalSet& base,
                                      Bits literal, bool is_or_equals,
                                      bool is_less_than, bool is_signed) {
    // Invert so we can remove elements from the interval.
    IntervalSet invert_base = IntervalSet::Complement(base);
    Bits min_value = is_signed ? Bits::MinSigned(literal.bit_count())
                               : Bits(literal.bit_count());
    Bits max_value = is_signed ? Bits::MaxSigned(literal.bit_count())
                               : Bits::AllOnes(literal.bit_count());
    Bits epsilon = UBits(1, literal.bit_count());
    if (is_less_than) {
      if (is_or_equals) {
        // variable <= literal
        if (literal == max_value) {
          // Nothing to restrict.
          // is V <= std::numeric_limits::max(). Always true.
          return absl::OkStatus();
        }
        literal = bits_ops::Add(literal, epsilon);
      }
      // variable < literal
      invert_base.AddInterval(Interval(literal, max_value));
    } else {
      if (is_or_equals) {
        // variable >= literal
        if (literal == min_value) {
          // nothing to restrict
          // is v >= std::numeric_limits::min(). Always true.
          return absl::OkStatus();
        }
        literal = bits_ops::Sub(literal, epsilon);
      }
      // variable > literal
      invert_base.AddInterval(Interval(min_value, literal));
    }
    invert_base.Normalize();
    IntervalSet restricted_set = IntervalSet::Complement(invert_base);
    if (restricted_set.Intervals().empty()) {
      // This implies the condition is actually unreachable impossible (since
      // we unify to bottom). For now just leave unconstrained.
      // TODO(allight): 2023-09-25: We can do better and should probably try to
      // communicate and remove the impossible cases here. This would need to be
      // done in narrowing or strength reduction by removing the associated
      // branches.
      return absl::OkStatus();
    }
    result_[variable] = restricted_set;
    return absl::OkStatus();
  }
  absl::Status UnifyImpreciseComparison(Node* l_op, Node* r_op,
                                        const IntervalSet& l_interval,
                                        const IntervalSet& r_interval) {
    // TODO(allight): 2023-08-10 This is much more complex and will be
    // implemented later.
    return absl::OkStatus();
  }

  const RangeQueryEngine& query_engine_;
  absl::flat_hash_map<Node*, IntervalSet> result_;
};

}  // namespace

absl::StatusOr<absl::flat_hash_map<Node*, IntervalSet>>
PropagateGivensBackwards(const RangeQueryEngine& engine, Node* node,
                         const IntervalSet& given) {
  BackPropagate prop(engine);
  prop.AddGiven(node, given);
  // We could back-propagate arbitrarily but (1) writing the rules for that is
  // tricky and time consuming since we need to do a reverse-topo sort and
  // unification between different users and (2) a single propagation is
  // likely good enough for most things.  This makes sure we figure out that
  // stuff like 'x < 4 == true' implies that x \in [0, 3] and such but we
  // don't need to deal with those tricky issues.
  XLS_RETURN_IF_ERROR(node->VisitSingleNode(&prop));
  return std::move(prop).ranges();
}

absl::StatusOr<absl::flat_hash_map<Node*, IntervalSet>>
PropagateGivensBackwards(const RangeQueryEngine& engine, Node* node,
                         const Bits& given) {
  return PropagateGivensBackwards(engine, node, IntervalSet::Precise(given));
}

}  // namespace xls
