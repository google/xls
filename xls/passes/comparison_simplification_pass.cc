// Copyright 2021 The XLS Authors
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

#include "xls/passes/comparison_simplification_pass.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

// An abstraction representing the condition that node `node` holds a value in
// `range`.
struct RangeEquivalence {
  Node* node;
  IntervalSet range;
};

// Reduce the ranges of the given equivalences using the given reduction
// function. If the equivalence `node` fields are not all the same, return
// nullopt. If `complement` is true then the range is complemented before
// returning.
std::optional<RangeEquivalence> ReduceEquivalence(
    Node* node, absl::Span<const RangeEquivalence> equivalences,
    const std::function<IntervalSet(const IntervalSet&, const IntervalSet&)>&
        reduce,
    bool complement) {
  // All equivalences must concern the same node.
  if (equivalences.empty() ||
      !std::all_of(equivalences.begin(), equivalences.end(),
                   [&](const RangeEquivalence& c) {
                     return c.node == equivalences[0].node;
                   })) {
    return std::nullopt;
  }
  IntervalSet result = equivalences[0].range;
  for (int64_t i = 1; i < equivalences.size(); ++i) {
    result = reduce(result, equivalences[i].range);
  }
  result.Normalize();
  if (complement) {
    result = IntervalSet::Complement(result);
  }
  return RangeEquivalence{equivalences[0].node, result};
}

// Returns a range over which which the comparison `x < limit` holds.
IntervalSet MakeULtRange(const Bits& limit) {
  IntervalSet result(limit.bit_count());
  if (limit.IsZero()) {
    // ULt(x, 0) => empty range
    return result;
  }
  // ULt(x, C) => [0, C-1]
  result.AddInterval(
      Interval(Bits(limit.bit_count()), bits_ops::Decrement(limit)));
  result.Normalize();
  return result;
}

// Returns a range over which which the comparison `x > limit` holds.
IntervalSet MakeUGtRange(const Bits& limit) {
  IntervalSet result(limit.bit_count());
  if (limit.IsAllOnes()) {
    // ULt(x, MAX) => empty range
    return result;
  }
  // ULt(x, C) => [C+1, MAX]
  result.AddInterval(
      Interval(bits_ops::Increment(limit), Bits::AllOnes(limit.bit_count())));
  result.Normalize();
  return result;
}

// Returns a range equivalence for node `node` or std::nullopt if one cannot be
// determined.
std::optional<RangeEquivalence> ComputeRangeEquivalence(
    Node* node,
    const absl::flat_hash_map<Node*, RangeEquivalence>& equivalences) {
  if (!node->GetType()->IsBits() || node->BitCountOrDie() != 1 ||
      (OpIsCompare(node->op()) && !node->operand(0)->GetType()->IsBits())) {
    return std::nullopt;
  }

  // A compare operation with a literal operand trivially has a range
  // equivalence.
  if (OpIsCompare(node->op()) &&
      (node->operand(0)->Is<Literal>() || node->operand(1)->Is<Literal>())) {
    Literal* literal_operand;
    Node* other_operand;
    bool literal_on_rhs;
    if (node->operand(0)->Is<Literal>()) {
      literal_operand = node->operand(0)->As<Literal>();
      other_operand = node->operand(1);
      literal_on_rhs = false;
    } else {
      literal_operand = node->operand(1)->As<Literal>();
      other_operand = node->operand(0);
      literal_on_rhs = true;
    }
    switch (node->op()) {
      case Op::kEq:
        return RangeEquivalence{
            other_operand,
            IntervalSet::Precise(literal_operand->value().bits())};
      case Op::kNe:
        return RangeEquivalence{other_operand,
                                IntervalSet::Complement(IntervalSet::Precise(
                                    literal_operand->value().bits()))};
      // We only need to consider the strict comparisons (kUGt, kULt) because
      // canonicalization transforms non-strict comparions (kULe, kUGe) with
      // literals into the strict form.
      case Op::kULt:
        // ULt(x, C)  =>  [0, C-1]
        // ULt(C, x)  =>  [C+1, MAX]
        return RangeEquivalence{
            other_operand, literal_on_rhs
                               ? MakeULtRange(literal_operand->value().bits())
                               : MakeUGtRange(literal_operand->value().bits())};
      case Op::kUGt:
        // UGt(x, C)  =>  [C+1, MAX]
        // UGt(C, x)  =>  [0, C-1]
        return RangeEquivalence{
            other_operand, literal_on_rhs
                               ? MakeUGtRange(literal_operand->value().bits())
                               : MakeULtRange(literal_operand->value().bits())};
      default:
        return std::nullopt;
    }
  }

  // Range equivalence for logical operations can be computed from the
  // equivalences (if any) of their operand.
  std::vector<RangeEquivalence> operand_equivalences;
  for (Node* operand : node->operands()) {
    if (!equivalences.contains(operand)) {
      return std::nullopt;
    }
    operand_equivalences.push_back(equivalences.at(operand));
  }
  switch (node->op()) {
    case Op::kAnd:
      return ReduceEquivalence(node, operand_equivalences,
                               IntervalSet::Intersect,
                               /*complement=*/false);
    case Op::kOr:
      return ReduceEquivalence(node, operand_equivalences, IntervalSet::Combine,
                               /*complement=*/false);
    case Op::kNot:
      // Not(x) is equivalent to Nand(x).
    case Op::kNand:
      return ReduceEquivalence(node, operand_equivalences,
                               IntervalSet::Intersect,
                               /*complement=*/true);
    case Op::kNor:
      return ReduceEquivalence(node, operand_equivalences, IntervalSet::Combine,
                               /*complement=*/true);
    default:
      // TODO(meheff): 2021/12/14 Handle XOR.
      break;
  }

  return std::nullopt;
}

// If the given range corresponds to values over which ULt(x, C) holds then
// return the constant `C`.
std::optional<Bits> GetULtInterval(const IntervalSet& range) {
  if (range.NumberOfIntervals() != 1) {
    return std::nullopt;
  }
  const Interval& interval = range.Intervals().front();
  if (interval.LowerBound().IsZero() && !interval.UpperBound().IsAllOnes()) {
    return bits_ops::Increment(interval.UpperBound());
  }
  return std::nullopt;
}

// If the given range corresponds to values over which UGt(x, C) holds then
// return the constant `C`.
std::optional<Bits> GetUGtInterval(const IntervalSet& range) {
  if (range.NumberOfIntervals() != 1) {
    return std::nullopt;
  }
  const Interval& interval = range.Intervals().front();
  if (!interval.LowerBound().IsZero() && interval.UpperBound().IsAllOnes()) {
    return bits_ops::Decrement(interval.LowerBound());
  }
  return std::nullopt;
}

// Hashable encapsulation of a comparison operation.
struct Comparison {
  Node* lhs;
  Node* rhs;
  Op op;

  friend bool operator==(const Comparison&, const Comparison&) = default;
  template <typename H>
  friend H AbslHashValue(H h, const Comparison& c) {
    return H::combine(std::move(h), c.lhs, c.rhs, c.op);
  }
};

// Replaces all comparisons which can be trivially derived from other
// comparisons. For example, assuming `a > b` already exists in the graph, then
// the following optimizations are possible:
//
//  b < a    =>    a > b
//  a <= b   =>  !(a > b)
//  b >= a   =>  !(a > b)
//
// This transformation saves a comparison at the (possible) cost of a single
// inverter which is a win. This optimization works across all comparison
// operations.
//
// Returns true if any transformations were performed.
absl::StatusOr<bool> TransformDerivedComparisons(FunctionBase* f) {
  bool changed = false;
  absl::flat_hash_map<Comparison, Node*> comparisons;
  for (Node* node : TopoSort(f)) {
    if (!node->Is<CompareOp>()) {
      continue;
    }
    // Try to replace with an op with commuted operands. Eg: a > b to b < a
    XLS_ASSIGN_OR_RETURN(Op commuted_op, ReverseComparisonOp(node->op()));
    Comparison commuted_comparison{
        .lhs = node->operand(1), .rhs = node->operand(0), .op = commuted_op};
    auto commuted_it = comparisons.find(commuted_comparison);
    if (commuted_it != comparisons.end()) {
      Node* equivalent_node = commuted_it->second;
      VLOG(2) << absl::StreamFormat(
          "Compare op is equivalent to op %s with commuted operands: %s",
          equivalent_node->GetName(), node->ToString());
      XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(equivalent_node));
      changed = true;
      continue;
    }

    // Try to replace with an op with an inverted comparison.
    // Eg: a > b to !(a <= b)
    XLS_ASSIGN_OR_RETURN(Op inverted_op, InvertComparisonOp(node->op()));
    Comparison inverted_comparison{
        .lhs = node->operand(0), .rhs = node->operand(1), .op = inverted_op};
    auto inverted_it = comparisons.find(inverted_comparison);
    if (inverted_it != comparisons.end()) {
      Node* inverted_node = inverted_it->second;
      VLOG(2) << absl::StreamFormat("Compare op is inversion of op %s: %s",
                                    inverted_node->GetName(), node->ToString());
      XLS_RETURN_IF_ERROR(
          node->ReplaceUsesWithNew<UnOp>(inverted_node, Op::kNot).status());
      changed = true;
      continue;
    }

    // Try to replace with an op with an inverted and commuted comparison.
    // Eg: a > b to !(b >= a)
    XLS_ASSIGN_OR_RETURN(Op commuted_and_inverted_op,
                         InvertComparisonOp(commuted_op));
    Comparison commuted_and_inverted_comparison{.lhs = node->operand(1),
                                                .rhs = node->operand(0),
                                                .op = commuted_and_inverted_op};
    auto commuted_and_inverted_it =
        comparisons.find(commuted_and_inverted_comparison);
    if (commuted_and_inverted_it != comparisons.end()) {
      Node* inverted_node = commuted_and_inverted_it->second;
      VLOG(2) << absl::StreamFormat(
          "Compare op is inversion of op %s with commuted operands: %s",
          inverted_node->GetName(), node->ToString());
      XLS_RETURN_IF_ERROR(
          node->ReplaceUsesWithNew<UnOp>(inverted_node, Op::kNot).status());
      changed = true;
      continue;
    }

    comparisons[Comparison{
        .lhs = node->operand(0), .rhs = node->operand(1), .op = node->op()}] =
        node;
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> ComparisonSimplificationPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results) const {
  bool changed = false;

  // Compute range equivalences for all nodes. Single-bit node X has an
  // associated RangeEquivalence{Y, R} if the value of X is equivalent to node Y
  // having a value within range R. Examples:
  //
  //  Eq(x, 42)   => RangeEquivalence{x, [[42, 42]]}
  //  Ne(x, 42)   => RangeEquivalence{x, [[0, 41], [43, MAX]]}
  //  Nor(Eq(x, 42), Eq(x, 37))
  //              => RangeEquivalence{x, [[0, 36], [38, 41], [42, MAX]]}
  absl::flat_hash_map<Node*, RangeEquivalence> equivalences;
  VLOG(3) << absl::StreamFormat("Range equivalences for function `%s`:",
                                f->name());
  for (Node* node : TopoSort(f)) {
    std::optional<RangeEquivalence> equivalence =
        ComputeRangeEquivalence(node, equivalences);
    if (!equivalence.has_value()) {
      VLOG(3) << absl::StreamFormat("  %s : <none>", node->GetName());
      continue;
    }
    VLOG(3) << absl::StreamFormat("  %s : %s in %s", node->GetName(),
                                  equivalence.value().node->GetName(),
                                  equivalence.value().range.ToString());
    equivalences[node] = equivalence.value();
  }

  for (Node* node : TopoSort(f)) {
    if (!equivalences.contains(node)) {
      continue;
    }

    auto is_eq_or_not_eq = [](Node* n) {
      return n->op() == Op::kEq || n->op() == Op::kNe;
    };
    // Returns true if `n` is an inverse of a compare op with multiple
    // users. These should not be replaced in this optimization. This is a guard
    // against undesired transformations which may increase the number of
    // comparison operations.  We don't transform instances like !(a == b) to a
    // != b if a == b has other uses.
    auto is_inverse_of_multiuse_compare_op = [](Node* n) {
      return n->op() == Op::kNot && n->operand(0)->Is<CompareOp>() &&
             !HasSingleUse(n->operand(0));
    };

    const IntervalSet& range = equivalences.at(node).range;

    // First consider cases where the node can be replaced with a constant
    // zero or one.
    if (range.IsMaximal()) {
      // The range is maximal so the condition is always true. Replace
      // `node` with a literal one.
      VLOG(2) << absl::StreamFormat(
          "Compare op has maximal range, replacing with 1: %s",
          node->ToString());
      XLS_RETURN_IF_ERROR(
          node->ReplaceUsesWithNew<Literal>(Value(UBits(1, 1))).status());
      changed = true;
    } else if (range.IsEmpty()) {
      // The range is empty so the condition is always false. Replace
      // `node` with a literal zero.
      VLOG(2) << absl::StreamFormat(
          "Compare op has empty range, replacing with 0: %s", node->ToString());
      XLS_RETURN_IF_ERROR(
          node->ReplaceUsesWithNew<Literal>(Value(UBits(0, 1))).status());
      changed = true;
    } else if (std::optional<Bits> precise_value = range.GetPreciseValue();
               precise_value.has_value()) {
      // The range is a single value C. Replace `node` with Eq(x, C).
      if (is_eq_or_not_eq(node) || is_inverse_of_multiuse_compare_op(node)) {
        // Skip if node is already a ne/eq.
        continue;
      }
      VLOG(2) << absl::StreamFormat(
          "Compare op has precise range, replacing with eq %s: %s",
          Value(precise_value.value()).ToString(), node->ToString());
      XLS_ASSIGN_OR_RETURN(
          Literal * literal,
          f->MakeNode<Literal>(SourceInfo(), Value(precise_value.value())));
      XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<CompareOp>(
                                  equivalences.at(node).node, literal, Op::kEq)
                              .status());
      changed = true;
    } else if (std::optional<Bits> precise_value =
                   IntervalSet::Complement(range).GetPreciseValue();
               precise_value.has_value()) {
      // The range is all values except a single value C. Replace `node`
      // with Ne(x, C).
      if (is_eq_or_not_eq(node) || is_inverse_of_multiuse_compare_op(node)) {
        // Skip if node is already a ne/eq.
        continue;
      }
      VLOG(2) << absl::StreamFormat(
          "Compare op has complementary range, replacing with ne %s: %s",
          Value(precise_value.value()).ToString(), node->ToString());
      XLS_ASSIGN_OR_RETURN(
          Literal * literal,
          f->MakeNode<Literal>(SourceInfo(), Value(precise_value.value())));
      XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<CompareOp>(
                                  equivalences.at(node).node, literal, Op::kNe)
                              .status());
      changed = true;
    } else if (std::optional<Bits> limit = GetULtInterval(range);
               limit.has_value()) {
      // The range is [0, C]. Replace `node` with ULt(x, C+1).
      if (node->Is<CompareOp>() || is_inverse_of_multiuse_compare_op(node)) {
        // Skip if node is already a comparison or an inverse of a multi-user
        // comparison.
        continue;
      }
      VLOG(2) << absl::StreamFormat("Compare op has range < %s, replacing: %s",
                                    Value(limit.value()).ToString(),
                                    node->ToString());
      XLS_ASSIGN_OR_RETURN(
          Literal * literal,
          f->MakeNode<Literal>(SourceInfo(), Value(limit.value())));
      XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<CompareOp>(
                                  equivalences.at(node).node, literal, Op::kULt)
                              .status());
      changed = true;
    } else if (std::optional<Bits> limit = GetUGtInterval(range);
               limit.has_value()) {
      // The range is [C, MAX]. Replace `node` with UGt(x, C-1).
      if (node->Is<CompareOp>()) {
        // Skip if node is already a comparison.
        continue;
      }
      VLOG(2) << absl::StreamFormat("Compare op has range > %s, replacing: %s",
                                    Value(limit.value()).ToString(),
                                    node->ToString());
      XLS_ASSIGN_OR_RETURN(
          Literal * literal,
          f->MakeNode<Literal>(SourceInfo(), Value(limit.value())));
      XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<CompareOp>(
                                  equivalences.at(node).node, literal, Op::kUGt)
                              .status());
      changed = true;
    }
  }

  XLS_ASSIGN_OR_RETURN(bool common_changed, TransformDerivedComparisons(f));
  changed = changed || common_changed;

  return changed;
}

REGISTER_OPT_PASS(ComparisonSimplificationPass);

}  // namespace xls
