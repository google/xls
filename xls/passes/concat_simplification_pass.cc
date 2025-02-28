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

#include "xls/passes/concat_simplification_pass.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <iterator>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"

namespace xls {
namespace {

// Returns true if the given concat has consecutive constant operands.
bool HasConsecutiveConstantOperands(Concat* concat,
                                    const QueryEngine& query_engine) {
  for (int64_t i = 1; i < concat->operand_count(); ++i) {
    if (query_engine.IsFullyKnown(concat->operand(i - 1)) &&
        query_engine.IsFullyKnown(concat->operand(i))) {
      return true;
    }
  }
  return false;
}

// Replaces any consecutive constant operands with a single merged literal
// operand. Returns the newly created concat which never aliases the given
// concat.
absl::StatusOr<Concat*> ReplaceConsecutiveConstantOperands(
    Concat* concat, const QueryEngine& query_engine) {
  std::vector<Node*> new_operands;
  std::vector<Bits> consecutive_constants;

  Node* last_constant = nullptr;
  auto add_consecutive_constants_to_operands = [&]() -> absl::Status {
    if (consecutive_constants.size() > 1) {
      XLS_ASSIGN_OR_RETURN(
          Node * new_literal,
          concat->function_base()->MakeNode<Literal>(
              concat->loc(), Value(bits_ops::Concat(consecutive_constants))));
      new_operands.push_back(new_literal);
    } else if (consecutive_constants.size() == 1) {
      new_operands.push_back(last_constant);
    }
    last_constant = nullptr;
    consecutive_constants.clear();
    return absl::OkStatus();
  };

  for (Node* operand : concat->operands()) {
    if (std::optional<Bits> known_value =
            query_engine.KnownValueAsBits(operand);
        known_value.has_value()) {
      last_constant = operand;
      consecutive_constants.push_back(*std::move(known_value));
    } else {
      XLS_RETURN_IF_ERROR(add_consecutive_constants_to_operands());
      new_operands.push_back(operand);
    }
  }
  XLS_RETURN_IF_ERROR(add_consecutive_constants_to_operands());
  return concat->ReplaceUsesWithNew<Concat>(new_operands);
}

// Replaces the given concat and any of its operand which are concats (and any
// of *their* operands which are concats, etc) with a single concat
// operation. For example:
//
//   Concat(a, b, Concat(c, Concat(d, e))) => Concat(a, b, c, d, e)
//
// Returns the newly created concat which never aliases the given concat.
absl::StatusOr<Concat*> FlattenConcatTree(Concat* concat) {
  std::vector<Node*> new_operands;
  std::deque<Node*> worklist(concat->operands().begin(),
                             concat->operands().end());
  while (!worklist.empty()) {
    Node* node = worklist.front();
    worklist.pop_front();
    if (node->Is<Concat>()) {
      worklist.insert(worklist.begin(), node->operands().begin(),
                      node->operands().end());
    } else {
      new_operands.push_back(node);
    }
  }
  return concat->ReplaceUsesWithNew<Concat>(new_operands);
}

// Attempts to replace the given concat with a simpler or more canonical
// form. Returns true if the concat was replaced.
absl::StatusOr<bool> SimplifyConcat(Concat* concat, int64_t opt_level,
                                    std::deque<Concat*>* worklist) {
  absl::Span<Node* const> operands = concat->operands();

  // Concat with a single operand can be replaced with its operand.
  if (concat->operand_count() == 1) {
    XLS_RETURN_IF_ERROR(concat->ReplaceUsesWith(operands[0]));
    return true;
  }

  // Tree of concats can be flattened to a single concat.
  if (std::any_of(operands.begin(), operands.end(),
                  [](Node* op) { return op->Is<Concat>(); })) {
    XLS_ASSIGN_OR_RETURN(Concat * new_concat, FlattenConcatTree(concat));
    worklist->push_back(new_concat);
    return true;
  }

  // Consecutive literal operands of a concat can be merged into a single
  // literal.
  StatelessQueryEngine query_engine;
  if (HasConsecutiveConstantOperands(concat, query_engine)) {
    XLS_ASSIGN_OR_RETURN(
        Concat * new_concat,
        ReplaceConsecutiveConstantOperands(concat, query_engine));
    worklist->push_back(new_concat);
    return true;
  }

  // Eliminate any zero-bit operands that get concatenated.
  if (std::any_of(operands.begin(), operands.end(),
                  [](Node* op) { return op->BitCountOrDie() == 0; })) {
    std::vector<Node*> new_operands;
    for (Node* operand : operands) {
      if (operand->BitCountOrDie() != 0) {
        new_operands.push_back(operand);
      }
    }
    XLS_ASSIGN_OR_RETURN(Concat * new_concat,
                         concat->ReplaceUsesWithNew<Concat>(new_operands));
    worklist->push_back(new_concat);
    return true;
  }

  // If we concatenate bits and then reverse the concatenation,
  // hoist the reverse above the concatenation.  In the modified IR,
  // concatenation input operands are reversed and then concatenated in reverse
  // order:
  //   reverse(concat(a, b, c)) => concat(reverse(c), reverse(b), reverse(a))
  int64_t num_reverse_users = 0;
  Node* reverse_user = nullptr;
  bool concat_has_nonreversible_user = false;
  for (Node* user : concat->users()) {
    switch (user->op()) {
      case Op::kReverse:
        ++num_reverse_users;
        reverse_user = user;
        break;
      case Op::kAndReduce:
      case Op::kOrReduce:
      case Op::kXorReduce:
        break;
      default:
        concat_has_nonreversible_user = true;
    }
  }
  // If there are multiple reverse users, common-subexpression elimination
  // should combine them later. We can apply the optimization after this.
  if (NarrowingEnabled(opt_level) && num_reverse_users == 1 &&
      !concat_has_nonreversible_user) {
    // Get reversed operands in reverse order.  Simplification should eliminate
    // any reversals of single-bit inputs that we produce here, so we do not
    // check for this case.
    std::vector<Node*> new_operands;
    new_operands.reserve(concat->operands().size());
    for (absl::Span<Node* const>::reverse_iterator riter =
             concat->operands().rbegin();
         riter != concat->operands().rend(); ++riter) {
      XLS_ASSIGN_OR_RETURN(
          Node * hoisted_rev,
          concat->function_base()->MakeNode<UnOp>(
              (*concat->users().begin())->loc(), *riter, Op::kReverse));
      new_operands.push_back(hoisted_rev);
    }

    // Add new concat to function, replace uses of original reverse.
    XLS_ASSIGN_OR_RETURN(
        Concat * new_concat,
        reverse_user->ReplaceUsesWithNew<Concat>(new_operands));
    worklist->push_back(new_concat);
    return true;
  }

  // If consecutive concat inputs are consecutive bit slices, create a new,
  // merged bit slice and a new concat that consumes the merged bit slice.
  for (int64_t idx = 0; idx < concat->operand_count() - 1; ++idx) {
    // Check if consecutive operands are bit slices.
    const Node* higher_op = concat->operands().at(idx);
    const Node* lower_op = concat->operands().at(idx + 1);
    if (!higher_op->Is<BitSlice>() || !lower_op->Is<BitSlice>()) {
      continue;
    }
    const BitSlice* higher_slice = higher_op->As<BitSlice>();
    const BitSlice* lower_slice = lower_op->As<BitSlice>();

    // Note: May want to do some checks for use cases of the slices.
    // If the original slices will not be removed by dead-code elimination,
    // making another merged slice may not be useful. This is complicated
    // by the fact that the number of uses of slice may change during
    // optimization.

    // Check if bit slices have the same input operand.
    if (higher_slice->operand(0) != lower_op->operand(0)) {
      continue;
    }

    // Check if bit slices slice consecutive bits.
    if (lower_slice->start() + lower_slice->width() != higher_slice->start()) {
      continue;
    }

    // Create merged slice node.
    XLS_ASSIGN_OR_RETURN(
        Node * merged_slice,
        concat->function_base()->MakeNode<BitSlice>(
            concat->loc(), higher_slice->operand(0), lower_slice->start(),
            lower_slice->width() + higher_slice->width()));

    // Collect operands for new concat.
    std::vector<Node*> new_operands;
    new_operands.reserve(concat->operands().size() - 1);
    for (int64_t copy_idx = 0; copy_idx < concat->operands().size();
         ++copy_idx) {
      if (copy_idx == idx) {
        new_operands.push_back(merged_slice);
        continue;
      }
      if (copy_idx == idx + 1) {
        continue;
      }
      new_operands.push_back(concat->operand(copy_idx));
    }

    // Add new concat to function, replace uses of original concat.
    // Note: We only merge one pair of slices at a time for simplicity /
    // clarity. If there are mulitple consecutive slices, they will be merged
    // over multiple calls to SimplifyConcat.
    XLS_ASSIGN_OR_RETURN(Concat * new_concat,
                         concat->ReplaceUsesWithNew<Concat>(new_operands));
    worklist->push_back(new_concat);
    return true;
  }

  return false;
}

// Returns the union of the bit ranges of the
// inputs to the concatentation operations that are inputs
// to 'node'.  e.g. for node = {A: u2, B:u3} OR {C: u3, B: u2},
// we get begin_end_bits_inclusive = {{0,1},{2,2},{3, 4}}.
absl::StatusOr<std::map<int64_t, int64_t>> GetBitRangeUnionOfInputConcats(
    Node* node) {
  std::map<int64_t, int64_t> begin_end_bits_inclusive;
  // Record the beginning of all bit-ranges.
  for (int64_t i = 0; i < node->operand_count(); ++i) {
    Concat* concat_i = node->operand(i)->As<Concat>();
    XLS_RET_CHECK(!concat_i->operands().empty());

    int64_t bit_lower_idx = 0;
    for (auto cat_itr = concat_i->operands().rbegin();
         cat_itr != concat_i->operands().rend();
         bit_lower_idx += (*cat_itr)->BitCountOrDie(), ++cat_itr) {
      if ((*cat_itr)->BitCountOrDie() == 0) {
        return absl::InternalError(
            "Zero-bit concat operands should have been optimized away before "
            "calling GetBitRangeUnionOfInputConcats");
      }

      // Record lower index. We don't know the upper index of a range
      // until all bit-ranges are accounted for, so we will calculate the upper
      // index later.
      begin_end_bits_inclusive.insert({bit_lower_idx, 0});
    }
  }

  // Fill in upper indexes of ranges.
  XLS_RET_CHECK(!begin_end_bits_inclusive.empty());
  for (auto current_range_itr = begin_end_bits_inclusive.begin(),
            next_range_itr = std::next(current_range_itr);
       next_range_itr != begin_end_bits_inclusive.end();
       current_range_itr = next_range_itr,
            next_range_itr = std::next(next_range_itr)) {
    current_range_itr->second = next_range_itr->first - 1;
    XLS_RET_CHECK_LE(current_range_itr->first, current_range_itr->second);
  }
  begin_end_bits_inclusive.rbegin()->second = node->BitCountOrDie() - 1;

  return begin_end_bits_inclusive;
}

// If `node` is an n-ary logical operation with a constant and a concat, then
// hoist the operation above the concat. For example, if `x` and `y` are 8-bit:
//
//   0xab & {x, y} => {x & 0xa, x & 0xb}
//
// Returns true if the transformation succeeded.
absl::StatusOr<bool> TryHoistBitWiseWithConstant(
    Node* node, const QueryEngine& query_engine) {
  // TODO(meheff): Handle cases where there are multiple non-literal concat
  // operands. No need to consider multiple literal operands as canonicalization
  // merges multiple literal operands of bitwise operations.
  if (!OpIsBitWise(node->op()) || node->operand_count() != 2) {
    return false;
  }
  // Currently only nary ops are bitwise.
  XLS_RET_CHECK(node->Is<NaryOp>());
  Bits constant_bits;
  if (query_engine.IsFullyKnown(node->operand(0))) {
    constant_bits = *query_engine.KnownValueAsBits(node->operand(0));
  } else if (query_engine.IsFullyKnown(node->operand(1))) {
    constant_bits = *query_engine.KnownValueAsBits(node->operand(1));
  } else {
    return false;
  }
  Concat* concat;
  if (node->operand(0)->Is<Concat>()) {
    concat = node->operand(0)->As<Concat>();
  } else if (node->operand(1)->Is<Concat>()) {
    concat = node->operand(1)->As<Concat>();
  } else {
    return false;
  }
  std::vector<Node*> new_operands;
  int64_t offset = 0;
  for (int64_t i = concat->operand_count() - 1; i >= 0; --i) {
    Node* concat_operand = concat->operand(i);
    int64_t concat_operand_width = concat_operand->BitCountOrDie();
    XLS_ASSIGN_OR_RETURN(Literal * sliced_literal,
                         node->function_base()->MakeNode<Literal>(
                             node->loc(), Value(constant_bits.Slice(
                                              offset, concat_operand_width))));
    XLS_ASSIGN_OR_RETURN(
        Node * new_operand,
        node->function_base()->MakeNode<NaryOp>(
            node->loc(), std::vector<Node*>({concat_operand, sliced_literal}),
            node->op()));
    new_operands.push_back(new_operand);
    offset += concat_operand_width;
  }
  std::reverse(new_operands.begin(), new_operands.end());
  XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<Concat>(new_operands).status());
  return true;
}

// Tries to hoist the given bitwise operation above it's concat
// operations. Example:
//
//   Xor(Concat(a, b), Concat(c, d)) => Concat(Xor(a, c), Xor(b, d))
//
// Hosting the bitwise operations presents more opportunity for optimization and
// simplification.
// Note: The hoisted bitwise operations have operands that are bitslices
// of the original concatenations. This pass, bit slice simplification,
// constant folding, and dead code elimination will often simplify or eliminate
// these bit slices and dependencies on the original concatenations.
//
// Preconditions:
//   * All operands of the bitwise operation are concats.
absl::StatusOr<bool> TryHoistBitWiseOperation(Node* node,
                                              const QueryEngine& query_engine) {
  XLS_RET_CHECK(OpIsBitWise(node->op()));

  {
    XLS_ASSIGN_OR_RETURN(bool changed,
                         TryHoistBitWiseWithConstant(node, query_engine));
    if (changed) {
      return true;
    }
  }

  if (node->operand_count() == 0 ||
      !std::all_of(node->operands().begin(), node->operands().end(),
                   [](Node* op) { return op->Is<Concat>(); })) {
    return false;
  }

  // Collect bit ranges.
  // Note: XLS_ASSIGN_OR_RETURN doesn't seem to handle std::map correctly
  // (probably due to comma).
  auto union_result = GetBitRangeUnionOfInputConcats(node);
  if (!union_result.ok()) {
    return union_result.status();
  }
  std::map<int64_t, int64_t> begin_end_bits_inclusive = union_result.value();

  // Make bitwise operations.
  FunctionBase* func = node->function_base();
  std::vector<Node*> bitwise_ops;
  for (const auto& [start, end] : begin_end_bits_inclusive) {
    std::vector<Node*> slices;
    for (Node* concat : node->operands()) {
      XLS_ASSIGN_OR_RETURN(Node * new_slice,
                           func->MakeNode<BitSlice>(
                               /*loc=*/node->loc(), /*arg=*/concat,
                               /*start=*/start,
                               /*width=*/end - start + 1));
      slices.push_back(new_slice);
    }
    XLS_ASSIGN_OR_RETURN(Node * new_bitwise, node->Clone(slices));
    bitwise_ops.push_back(new_bitwise);
  }

  // Concatenate bitwise operations.
  std::reverse(bitwise_ops.begin(), bitwise_ops.end());
  XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<Concat>(bitwise_ops).status());

  return true;
}

// Transform a reduction of a concat to a reduction of the concat's operands.
// e.g. OrReduce(Concat(a,b)) ==> Or(OrReduce(a), OrReduce(b))
absl::StatusOr<bool> TryBypassReductionOfConcatenation(Node* node) {
  if (!node->Is<BitwiseReductionOp>() || !node->operand(0)->Is<Concat>()) {
    return false;
  }

  // Create reductions of concat operands.
  Concat* concat = node->operand(0)->As<Concat>();

  // Early exit if concat results in bitwith 0.
  if (concat->GetType()->GetFlatBitCount() == 0) {
    return false;
  }

  std::vector<Node*> new_reductions;
  for (Node* cat_operand : concat->operands()) {
    int64_t bit_count = cat_operand->GetType()->GetFlatBitCount();
    if (bit_count > 1) {
      XLS_ASSIGN_OR_RETURN(Node * reduce, node->Clone({cat_operand}));
      new_reductions.push_back(reduce);
    } else if (bit_count == 1) {
      new_reductions.push_back(cat_operand);
    }
  }

  XLS_ASSIGN_OR_RETURN(Op non_reductive_op, OpToNonReductionOp(node->op()));
  XLS_RETURN_IF_ERROR(
      node->ReplaceUsesWithNew<NaryOp>(new_reductions, non_reductive_op)
          .status());
  return true;
}

// Attempts to distribute a reducible operation into the operands (sub-slices)
// of a concat -- this helps the optimizer because the concat could otherwise
// act as something that's hard to "see through"; e.g. imagine you concatenate
// zeroes onto a value, and then check whether that concatenated value is equal
// to zero. By distributing the equality test to the operands, we'll see that
// for the concatenated zero bits there's no equality that needs to be
// performed, it's always true.
//
// So it can enable transforms like like:
//
//    Eq(Concat(bits[7]:0, x), 0) => And(Eq(bits[7], 0), Eq(x, 0)) => Eq(x, 0)
absl::StatusOr<bool> TryDistributeReducibleOperation(Node* node) {
  // For now we only handle eq and ne operations.
  if (node->op() != Op::kEq && node->op() != Op::kNe) {
    return false;
  }

  auto get_concat_and_other = [&](Concat** concat, Node** other) -> bool {
    if (node->operand(0)->Is<Concat>()) {
      *concat = node->operand(0)->As<Concat>();
      *other = node->operand(1);
      return true;
    }
    if (node->operand(1)->Is<Concat>()) {
      *concat = node->operand(1)->As<Concat>();
      *other = node->operand(0);
      return true;
    }
    return false;
  };

  Concat* concat;
  Node* other;
  if (!get_concat_and_other(&concat, &other)) {
    return false;
  }

  // For zero-bit concatenations, we simply report we didn't change anything
  // (there's nothing to change with respect to operands).
  if (concat->GetType()->GetFlatBitCount() == 0) {
    return false;
  }

  // For eq, the reduction is that all the sub-slices are equal (AND).
  // For ne, the reduction is that any one of the sub-slices is not-equal (OR).
  Op reducer = node->op() == Op::kEq ? Op::kAnd : Op::kOr;

  // Walk through the concat operands and grab the corresponding slice out of
  // the "other" node, and distribute the operation to occur on those
  // sub-slices.
  FunctionBase* f = concat->function_base();
  Node* result = nullptr;
  for (int64_t i = 0; i < concat->operands().size(); ++i) {
    SliceData concat_slice = concat->GetOperandSliceData(i);
    XLS_ASSIGN_OR_RETURN(
        Node * other_slice,
        f->MakeNode<BitSlice>(other->loc(), other, concat_slice.start,
                              concat_slice.width));
    XLS_ASSIGN_OR_RETURN(Node * slices_eq,
                         f->MakeNode<CompareOp>(node->loc(), concat->operand(i),
                                                other_slice, node->op()));
    if (result == nullptr) {
      result = slices_eq;
    } else {
      XLS_ASSIGN_OR_RETURN(
          result,
          f->MakeNode<NaryOp>(node->loc(),
                              std::vector<Node*>{result, slices_eq}, reducer));
    }
  }

  XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(result));
  return true;
}

}  // namespace

absl::StatusOr<bool> ConcatSimplificationPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  StatelessQueryEngine query_engine;

  // For optimizations which replace concats with other concats use a worklist
  // of unprocessed concats in the graphs. As new concats are created they are
  // added to the worklist.
  std::deque<Concat*> worklist;
  for (Node* node : f->nodes()) {
    if (node->Is<Concat>()) {
      worklist.push_back(node->As<Concat>());
    }
  }
  bool changed = false;
  while (!worklist.empty()) {
    Concat* concat = worklist.front();
    worklist.pop_front();
    XLS_ASSIGN_OR_RETURN(bool node_changed,
                         SimplifyConcat(concat, options.opt_level, &worklist));
    changed = changed || node_changed;
  }

  // For optimizations which optimize around concats, just iterate through once
  // and find all opportunities.
  if (options.narrowing_enabled()) {
    for (Node* node : context.TopoSort(f)) {
      if (OpIsBitWise(node->op())) {
        XLS_ASSIGN_OR_RETURN(bool bitwise_changed,
                             TryHoistBitWiseOperation(node, query_engine));
        changed = changed || bitwise_changed;
      } else {
        XLS_ASSIGN_OR_RETURN(bool distribute_changed,
                             TryDistributeReducibleOperation(node));
        changed = changed || distribute_changed;

        XLS_ASSIGN_OR_RETURN(bool reduction_changed,
                             TryBypassReductionOfConcatenation(node));
        changed = changed || reduction_changed;
      }
    }
  }

  return changed;
}

REGISTER_OPT_PASS(ConcatSimplificationPass);

}  // namespace xls
