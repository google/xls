// Copyright 2020 Google LLC
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

#include <deque>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"

namespace xls {
namespace {

// Returns true if the given concat as consecutive literal operands.
bool HasConsecutiveLiteralOperands(Concat* concat) {
  for (int64 i = 1; i < concat->operand_count(); ++i) {
    if (concat->operand(i - 1)->Is<Literal>() &&
        concat->operand(i)->Is<Literal>()) {
      return true;
    }
  }
  return false;
}

// Replaces any consecutive literal operands with a single merged literal
// operand. Returns the newly created concat which never aliases the given
// concat.
xabsl::StatusOr<Concat*> ReplaceConsecutiveLiteralOperands(Concat* concat) {
  std::vector<Node*> new_operands;
  std::vector<Literal*> consecutive_literals;

  auto add_consecutive_literals_to_operands = [&]() -> absl::Status {
    if (consecutive_literals.size() > 1) {
      std::vector<Bits> literal_bits(consecutive_literals.size());
      std::transform(consecutive_literals.begin(), consecutive_literals.end(),
                     literal_bits.begin(),
                     [](Literal* l) { return l->value().bits(); });
      XLS_ASSIGN_OR_RETURN(
          Node * new_literal,
          concat->function()->MakeNode<Literal>(
              concat->loc(), Value(bits_ops::Concat(literal_bits))));
      new_operands.push_back(new_literal);
    } else if (consecutive_literals.size() == 1) {
      new_operands.push_back(consecutive_literals.front());
    }
    consecutive_literals.clear();
    return absl::OkStatus();
  };

  for (Node* operand : concat->operands()) {
    if (operand->Is<Literal>()) {
      consecutive_literals.push_back(operand->As<Literal>());
    } else {
      XLS_RETURN_IF_ERROR(add_consecutive_literals_to_operands());
      new_operands.push_back(operand);
    }
  }
  XLS_RETURN_IF_ERROR(add_consecutive_literals_to_operands());
  return concat->ReplaceUsesWithNew<Concat>(new_operands);
}

// Replaces the given concat and any of its operand which are concats (and any
// of *their* operands which are concats, etc) with a single concat
// operation. For example:
//
//   Concat(a, b, Concat(c, Concat(d, e))) => Concat(a, b, c, d, e)
//
// Returns the newly created concat which never aliases the given concat.
xabsl::StatusOr<Concat*> FlattenConcatTree(Concat* concat) {
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
xabsl::StatusOr<bool> SimplifyConcat(Concat* concat,
                                     std::deque<Concat*>* worklist) {
  absl::Span<Node* const> operands = concat->operands();

  // Concat with a single operand can be replaced with its operand.
  if (concat->operand_count() == 1) {
    return concat->ReplaceUsesWith(operands[0]);
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
  if (HasConsecutiveLiteralOperands(concat)) {
    XLS_ASSIGN_OR_RETURN(Concat * new_concat,
                         ReplaceConsecutiveLiteralOperands(concat));
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
  if (concat->users().size() == 1 &&
      concat->users().at(0)->op() == Op::kReverse) {
    Function* func = concat->function();

    // Get reversed operands in reverse order.
    // BDD common subexpression elimination should eliminate any
    // reversals of single-bit inputs that we produce here,
    // so we do not check for this case.
    std::vector<Node*> new_operands;
    new_operands.reserve(concat->operands().size());
    for (absl::Span<Node* const>::reverse_iterator riter =
             concat->operands().rbegin();
         riter != concat->operands().rend(); ++riter) {
      XLS_ASSIGN_OR_RETURN(Node * hoisted_rev,
                           func->MakeNode<UnOp>(concat->users().at(0)->loc(),
                                                *riter, Op::kReverse));
      new_operands.push_back(hoisted_rev);
    }

    // Add new concat to function, replace uses of original reverse.
    XLS_ASSIGN_OR_RETURN(Concat * new_concat,
                         concat->ReplaceUsesWithNew<Concat>(new_operands));
    XLS_ASSIGN_OR_RETURN(
        bool function_changed,
        new_concat->users().at(0)->ReplaceUsesWith(new_concat));
    if (!function_changed) {
      return absl::InternalError(
          "Replacing reverse operation with reversed-input concatenation did "
          "not change function");
    }
    worklist->push_back(new_concat);
    return true;
  }

  return false;
}

// Tries to hoist the given bitwise operation above it's concat
// operations. Example:
//
//   Xor(Concat(a, b), Concat(c, d)) => Concat(Xor(a, c), Xor(b, d))
//
// Hosting the bitwise operations presents more opportunity for optimization and
// simplification.
//
// Preconditions:
//   * All operands of the bitwise operation are concats.
//   * The concats each have the same number of operands, and they are the same
//     size.
xabsl::StatusOr<bool> TryHoistBitWiseOperation(Node* node) {
  XLS_RET_CHECK(OpIsBitWise(node->op()));
  if (node->operand_count() == 0 ||
      !std::all_of(node->operands().begin(), node->operands().end(),
                   [](Node* op) { return op->Is<Concat>(); })) {
    return false;
  }

  Concat* concat_0 = node->operand(0)->As<Concat>();
  for (int i = 1; i < node->operand_count(); ++i) {
    Concat* concat_i = node->operand(i)->As<Concat>();
    if (concat_0->operand_count() != concat_i->operand_count()) {
      return false;
    }
    for (int j = 0; j < concat_0->operand_count(); ++j) {
      if (concat_0->operand(j)->BitCountOrDie() !=
          concat_i->operand(j)->BitCountOrDie()) {
        return false;
      }
    }
  }

  std::vector<Node*> new_concat_operands;
  for (int64 i = 0; i < concat_0->operand_count(); ++i) {
    std::vector<Node*> bitwise_operands;
    for (int64 j = 0; j < node->operand_count(); ++j) {
      bitwise_operands.push_back(node->operand(j)->operand(i));
    }
    XLS_ASSIGN_OR_RETURN(Node * new_bitwise,
                         node->Clone(bitwise_operands, node->function()));
    new_concat_operands.push_back(new_bitwise);
  }

  XLS_RETURN_IF_ERROR(
      node->ReplaceUsesWithNew<Concat>(new_concat_operands).status());
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
xabsl::StatusOr<bool> TryDistributeReducibleOperation(Node* node) {
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

  // For eq, the reduction is that all the sub-slices are equal (AND).
  // For ne, the reduction is that any one of the sub-slices is not-equal (OR).
  Op reducer = node->op() == Op::kEq ? Op::kAnd : Op::kOr;

  // Walk through the concat operands and grab the corresponding slice out of
  // the "other" node, and distribute the operation to occur on those
  // sub-slices.
  Function* f = concat->function();
  Node* result = nullptr;
  for (int64 i = 0; i < concat->operands().size(); ++i) {
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

  XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(result).status());
  return true;
}

}  // namespace

xabsl::StatusOr<bool> ConcatSimplificationPass::RunOnFunction(
    Function* f, const PassOptions& options, PassResults* results) const {
  XLS_VLOG(2) << "Running concat simplifier on function " << f->name();
  XLS_VLOG(3) << "Before:";
  XLS_VLOG_LINES(3, f->DumpIr());

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
    XLS_ASSIGN_OR_RETURN(bool node_changed, SimplifyConcat(concat, &worklist));
    changed = changed || node_changed;
  }

  // For optimizations which optimize around concats, just iterate through once
  // and find all opportunities.
  for (Node* node : TopoSort(f)) {
    bool node_changed = false;
    if (OpIsBitWise(node->op())) {
      XLS_ASSIGN_OR_RETURN(node_changed, TryHoistBitWiseOperation(node));
    } else {
      XLS_ASSIGN_OR_RETURN(node_changed, TryDistributeReducibleOperation(node));
    }
    changed = changed || node_changed;
  }

  XLS_VLOG(3) << "After:";
  XLS_VLOG_LINES(3, f->DumpIr());

  return changed;
}

}  // namespace xls
