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

#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/op.h"

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
absl::optional<RangeEquivalence> ReduceEquivalence(
    Node* node, absl::Span<const RangeEquivalence> equivalences,
    std::function<IntervalSet(const IntervalSet&, const IntervalSet&)> reduce,
    bool complement) {
  // All equivalences must concern the same node.
  if (equivalences.empty() ||
      !std::all_of(equivalences.begin(), equivalences.end(),
                   [&](const RangeEquivalence& c) {
                     return c.node == equivalences[0].node;
                   })) {
    return absl::nullopt;
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

// Returns a range equivalence for node `node` or absl::nullopt if one cannot be
// determined.
absl::optional<RangeEquivalence> ComputeRangeEquivalence(
    Node* node,
    const absl::flat_hash_map<Node*, RangeEquivalence>& equivalences) {
  if (!node->GetType()->IsBits() || node->BitCountOrDie() != 1) {
    return absl::nullopt;
  }

  // A compare operation with a literal operand trivially has a range
  // equivalence.
  if (OpIsCompare(node->op()) &&
      (node->operand(0)->Is<Literal>() || node->operand(1)->Is<Literal>())) {
    Literal* literal_operand;
    Node* other_operand;
    if (node->operand(0)->Is<Literal>()) {
      literal_operand = node->operand(0)->As<Literal>();
      other_operand = node->operand(1);
    } else {
      literal_operand = node->operand(1)->As<Literal>();
      other_operand = node->operand(0);
    }
    switch (node->op()) {
      case Op::kEq:
        return RangeEquivalence{
            other_operand,
            IntervalSet::Precise(literal_operand->value().bits())};
      case Op::kNe: {
        return RangeEquivalence{other_operand,
                                IntervalSet::Complement(IntervalSet::Precise(
                                    literal_operand->value().bits()))};
      }
      default:
        // TODO(meheff): 2021/11/22 Add other comparison operations (ULt, UGt,
        // etc).
        return absl::nullopt;
    }
  }

  // Range equivalence for logical operations can be computed from the
  // equivalences (if any) of their operand.
  std::vector<RangeEquivalence> operand_equivalences;
  for (Node* operand : node->operands()) {
    if (!equivalences.contains(operand)) {
      return absl::nullopt;
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
      break;
  }

  return absl::nullopt;
}

}  // namespace

absl::StatusOr<bool> ComparisonSimplificationPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const PassOptions& options, PassResults* results) const {
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
  XLS_VLOG(3) << absl::StreamFormat("Range equivalences for function `%s`:",
                                    f->name());
  for (Node* node : TopoSort(f)) {
    absl::optional<RangeEquivalence> equivalence =
        ComputeRangeEquivalence(node, equivalences);
    if (!equivalence.has_value()) {
      XLS_VLOG(3) << absl::StreamFormat("  %s : <none>", node->GetName());
      continue;
    }
    XLS_VLOG(3) << absl::StreamFormat("  %s : %s in %s", node->GetName(),
                                      equivalence.value().node->GetName(),
                                      equivalence.value().range.ToString());
    equivalences[node] = equivalence.value();
  }

  for (Node* node : TopoSort(f)) {
    if (node->Is<CompareOp>() || !equivalences.contains(node)) {
      continue;
    }

    const IntervalSet& range = equivalences.at(node).range;
    if (range.IsMaximal()) {
      // The range is maximal so the condition is always true. Replace `node`
      // with a literal one.
      XLS_RETURN_IF_ERROR(
          node->ReplaceUsesWithNew<Literal>(Value(UBits(1, 1))).status());
      changed = true;
    } else if (range.IsEmpty()) {
      // The range is empty so the condition is always false. Replace `node`
      // with a literal zero.
      XLS_RETURN_IF_ERROR(
          node->ReplaceUsesWithNew<Literal>(Value(UBits(0, 1))).status());
      changed = true;
    } else if (absl::optional<Bits> precise_value = range.GetPreciseValue();
               precise_value.has_value()) {
      // The range is a single value C. Replace `node` with Eq(x, C).
      XLS_ASSIGN_OR_RETURN(
          Literal * literal,
          f->MakeNode<Literal>(absl::nullopt, Value(precise_value.value())));
      XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<CompareOp>(
                                  equivalences.at(node).node, literal, Op::kEq)
                              .status());
      changed = true;
    } else if (absl::optional<Bits> precise_value =
                   IntervalSet::Complement(range).GetPreciseValue();
               precise_value.has_value()) {
      // The range is all values except a single value C. Replace `node` with
      // Ne(x, C).
      XLS_ASSIGN_OR_RETURN(
          Literal * literal,
          f->MakeNode<Literal>(absl::nullopt, Value(precise_value.value())));
      XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<CompareOp>(
                                  equivalences.at(node).node, literal, Op::kNe)
                              .status());
      changed = true;
    }
    // TODO(meheff): 2021/11/22 Add conversion to other operations. For example,
    // single-interval ranges starting at min value or ending at max value can
    // be simplified to Ult or UGt, respectively.
  }

  return changed;
}

}  // namespace xls
