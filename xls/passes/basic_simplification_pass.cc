// Copyright 2023 The XLS Authors
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

#include "xls/passes/basic_simplification_pass.h"

#include <algorithm>
#include <functional>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/stateless_query_engine.h"

namespace xls {
namespace {

// Return true iff `node` is an node for which op(x, x) == true.
bool IsBinaryReflexiveRelation(Node* node) {
  switch (node->op()) {
    case Op::kEq:
    case Op::kSGe:
    case Op::kUGe:
    case Op::kSLe:
    case Op::kULe:
      return true;
    default:
      return false;
  }
}

// Return true iff `node` is an node for which op(x, x) == false.
bool IsBinaryIrreflexiveRelation(Node* node) {
  switch (node->op()) {
    case Op::kNe:
    case Op::kSGt:
    case Op::kUGt:
    case Op::kSLt:
    case Op::kULt:
      return true;
    default:
      return false;
  }
}

// MatchPatterns matches simple tree patterns to find opportunities
// for simplification.
//
// Return 'true' if the IR was modified (uses of node was replaced with a
// different expression).
absl::StatusOr<bool> MatchPatterns(Node* n) {
  StatelessQueryEngine query_engine;

  // Pattern: Add/Sub/Or/Shift a value with 0 on the RHS.
  if ((n->op() == Op::kAdd || n->op() == Op::kSub || n->op() == Op::kShll ||
       n->op() == Op::kShrl || n->op() == Op::kShra) &&
      query_engine.IsAllZeros(n->operand(1))) {
    VLOG(2) << "FOUND: Useless operation of value with zero";
    XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(n->operand(0)));
    return true;
  }

  // Returns true if all operands of 'node' are the same.
  auto all_operands_same = [&query_engine](Node* node) {
    return std::all_of(node->operands().begin(), node->operands().end(),
                       [node, &query_engine](Node* op) {
                         return query_engine.NodesKnownUnsignedEquals(
                             op, node->operand(0));
                       });
  };

  // Duplicate operands of AND and OR (and their inverting forms NAND and OR)
  // can be removed.
  //
  //   Op(x, y, y, x)  =>  Op(x, y)
  if (n->Is<NaryOp>() && (n->op() == Op::kAnd || n->op() == Op::kOr ||
                          n->op() == Op::kNand || n->op() == Op::kNor)) {
    std::vector<Node*> unique_operands;
    for (Node* operand : n->operands()) {
      // This is quadratic in the number of operands, but shouldn't cause
      // problems unless we have at least hundreds of thousands of operands
      // which seems unlikely.
      if (std::find(unique_operands.begin(), unique_operands.end(), operand) ==
          unique_operands.end()) {
        unique_operands.push_back(operand);
      }
    }
    if (unique_operands.size() != n->operand_count()) {
      VLOG(2) << "FOUND: remove duplicate operands in and/or/nand/nor";
      XLS_RETURN_IF_ERROR(
          n->ReplaceUsesWithNew<NaryOp>(unique_operands, n->op()).status());
      return true;
    }
  }

  // Single operand forms of non-inverting logical ops (AND, OR) can be
  // replaced with the operand.
  //
  //   Op(x)  =>  x
  if (n->Is<NaryOp>() && (n->op() == Op::kAnd || n->op() == Op::kOr) &&
      n->operand_count() == 1) {
    VLOG(2) << "FOUND: replace single operand or(x) with x";
    XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(n->operand(0)));
    return true;
  }

  // Single operand forms of inverting logical ops (NAND, NOR) can be
  // replaced with the inverted operand.
  //
  //   Op(x)  =>  Not(x)
  if (n->Is<NaryOp>() && (n->op() == Op::kNor || n->op() == Op::kNand) &&
      n->operand_count() == 1) {
    VLOG(2) << "FOUND: replace single operand nand/nor(x) with not(x)";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<UnOp>(n->operand(0), Op::kNot).status());
    return true;
  }

  // All operands the same for XOR:
  //
  //   XOR(x, x, ...)  =>  x  // Odd number of operands.
  //   XOR(x, x, ...)  =>  0  // Even number of operands.
  if (n->op() == Op::kXor && all_operands_same(n)) {
    VLOG(2) << "FOUND: replace xor(x, x, ...) with 0 or 1";
    if (n->operand_count() % 2 == 0) {
      // Even number of operands. Replace with zero.
      XLS_RETURN_IF_ERROR(
          n->ReplaceUsesWithNew<Literal>(Value(UBits(0, n->BitCountOrDie())))
              .status());
    } else {
      // Odd number of operands. Replace with XOR operand.
      XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(n->operand(0)));
    }
    return true;
  }

  // Non-inverting bitwise reduction of a single bit is just the original bit.
  if (n->OpIn({Op::kAndReduce, Op::kOrReduce, Op::kXorReduce}) &&
      n->operand(0)->BitCountOrDie() == 1) {
    VLOG(2) << "FOUND: replace " << OpToString(n->op()) << "(x) with x";
    XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(n->operand(0)));
    return true;
  }

  // Bitwise reduction of an empty operand is always the identity for the
  // operation.
  if (n->Is<BitwiseReductionOp>() && n->operand(0)->BitCountOrDie() == 0) {
    Bits identity;
    switch (n->op()) {
      case Op::kAndReduce:
        identity = UBits(1, 1);
        break;
      case Op::kOrReduce:
      case Op::kXorReduce:
        identity = UBits(0, 1);
        break;
      default:
        return absl::InternalError("Unhandled bitwise reduction op");
    }
    VLOG(2) << "FOUND: replace " << OpToString(n->op())
            << "(empty) with the identity for the operation";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(Value(identity)).status());
    return true;
  }

  // Replaces uses of n with a new node by eliminating operands for which the
  // "predicate" holds. If the predicate holds for all operands, the
  // NaryOpNullaryResult is used as a replacement.
  auto eliminate_operands_where =
      [n](std::function<bool(Node*)> predicate) -> absl::StatusOr<bool> {
    XLS_RET_CHECK(n->Is<NaryOp>());
    std::vector<Node*> new_operands;
    for (Node* operand : n->operands()) {
      if (!predicate(operand)) {
        new_operands.push_back(operand);
      }
    }
    if (new_operands.size() == n->operand_count()) {
      return false;
    }
    if (new_operands.empty()) {
      XLS_RETURN_IF_ERROR(
          n
              ->ReplaceUsesWithNew<Literal>(Value(DoLogicalOp(
                  n->op(), {LogicalOpIdentity(n->op(), n->BitCountOrDie())})))
              .status());
    } else {
      XLS_RETURN_IF_ERROR(
          n->ReplaceUsesWithNew<NaryOp>(new_operands, n->op()).status());
    }
    return true;
  };

  // Or(x, 0, y) => Or(x, y)
  // Xor(x, 0, y) => Xor(x, y)
  // Nor(x, 0, y) => Nor(x, y)
  if ((n->op() == Op::kOr || n->op() == Op::kXor || n->op() == Op::kNor)) {
    VLOG(2) << "FOUND: remove zero valued operands from or, nor, or, xor";
    XLS_ASSIGN_OR_RETURN(bool changed,
                         eliminate_operands_where([&query_engine](Node* node) {
                           return query_engine.IsAllZeros(node);
                         }));
    if (changed) {
      return true;
    }
  }

  // Or(x, -1, y) => -1
  if (n->op() == Op::kOr && AnyOperandWhere(n, [&query_engine](Node* node) {
        return query_engine.IsAllOnes(node);
      })) {
    VLOG(2) << "FOUND: replace or(..., 1, ...) with 1";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(AllOnesOfType(n->GetType())).status());
    return true;
  }

  // Nor(x, -1, y) => 0
  if (n->op() == Op::kNor && AnyOperandWhere(n, [&query_engine](Node* node) {
        return query_engine.IsAllOnes(node);
      })) {
    VLOG(2) << "FOUND: replace nor(..., 1, ...) with 0";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(ZeroOfType(n->GetType())).status());
    return true;
  }

  // And(x, -1, y) => And(x, y)
  // Nand(x, -1, y) => Nand(x, y)
  if (n->op() == Op::kAnd || n->op() == Op::kNand) {
    VLOG(2) << "FOUND: remove all-ones operands from and/nand";
    XLS_ASSIGN_OR_RETURN(bool changed,
                         eliminate_operands_where([&query_engine](Node* node) {
                           return query_engine.IsAllOnes(node);
                         }));
    if (changed) {
      return true;
    }
  }

  // And(x, 0) => 0
  if (n->op() == Op::kAnd && AnyOperandWhere(n, [&query_engine](Node* node) {
        return query_engine.IsAllZeros(node);
      })) {
    VLOG(2) << "FOUND: replace and(..., 0, ...) with 0";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(ZeroOfType(n->GetType())).status());
    return true;
  }

  // Nand(x, 0) => 1
  if (n->op() == Op::kNand && AnyOperandWhere(n, [&query_engine](Node* node) {
        return query_engine.IsAllZeros(node);
      })) {
    VLOG(2) << "FOUND: replace nand(..., 0, ...) with 1";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(AllOnesOfType(n->GetType())).status());
    return true;
  }

  // Returns true if `node` one of its operands also appears inverted in the
  // operands of `node`.
  auto has_inverted_operand = [](Node* node) {
    for (Node* operand : node->operands()) {
      if (operand->op() == Op::kNot &&
          std::find(node->operands().begin(), node->operands().end(),
                    operand->operand(0)) != node->operands().end()) {
        return true;
      }
    }
    return false;
  };

  // And(x, Not(x)) => 0
  // And(Not(x), x) => 0
  //
  // Note that this won't be found through the ternary query engine because
  // conservatively it determines `not(UNKNOWN) = UNKNOWN`.
  if (n->op() == Op::kAnd && has_inverted_operand(n)) {
    VLOG(2) << "FOUND: replace and(x, not(x)) with 0";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(Value(UBits(0, n->BitCountOrDie())))
            .status());
    return true;
  }

  //   Or(X, Not(X), ...)  => 1...
  //
  // Note that this won't be found through the ternary query engine because
  // conservatively it determines `not(UNKNOWN) = UNKNOWN`.
  if (n->op() == Op::kOr && has_inverted_operand(n)) {
    VLOG(2) << "FOUND: replace Or(x, not(x)) with 1...";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(Value(Bits::AllOnes(n->BitCountOrDie())))
            .status());
    return true;
  }

  //   Add(X, Not(X)) => 1...
  //   Add(Not(X), X) => 1...
  //
  // Note that this won't be found through the ternary query engine because
  // conservatively it determines `not(UNKNOWN) = UNKNOWN`.
  if (n->op() == Op::kAdd && n->operand_count() == 2 &&
      has_inverted_operand(n)) {
    VLOG(2) << "FOUND: replace Add(x, not(x)) with 1...";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(Value(Bits::AllOnes(n->BitCountOrDie())))
            .status());
    return true;
  }

  // Xor(x, -1) => Not(x)
  if (n->op() == Op::kXor && n->operand_count() == 2 &&
      query_engine.IsAllOnes(n->operand(1))) {
    VLOG(2) << "FOUND: Found xor with all ones";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<UnOp>(n->operand(0), Op::kNot).status());
    return true;
  }

  // Sub(X, X) => Literal(0)
  if (n->op() == Op::kSub && all_operands_same(n)) {
    VLOG(2) << "FOUND: Sub of value with itself";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(Value(UBits(0, n->BitCountOrDie())))
            .status());
    return true;
  }

  auto is_same_opcode = [&](Node* other) { return n->op() == other->op(); };

  // Flatten nested associative nary ops into their root op:
  //
  //   Op(Op(x, y), z)  =>  Op(x, y, z)
  //
  // This operation should only be performed if the only user of the nested ops
  // is the outer op itself.
  if (OpIsAssociative(n->op()) && n->Is<NaryOp>() &&
      AnyOperandWhere(n, is_same_opcode)) {
    std::vector<Node*> new_operands;
    bool should_transform = false;
    for (Node* operand : n->operands()) {
      if (operand->op() == n->op() && HasSingleUse(operand)) {
        should_transform = true;
        for (Node* suboperand : operand->operands()) {
          new_operands.push_back(suboperand);
        }
      } else {
        new_operands.push_back(operand);
      }
    }
    if (should_transform) {
      VLOG(2) << "FOUND: flatten nested associative nary ops";
      XLS_RETURN_IF_ERROR(
          n->ReplaceUsesWithNew<NaryOp>(new_operands, n->op()).status());
      return true;
    }
  }

  // Fold the constant values presented to the nary op.
  //
  //   Op(C0, x, C1)  =>  Op(C2, x) where C2 == Op(C0, C1)
  if (OpIsCommutative(n->op()) && OpIsAssociative(n->op()) && n->Is<NaryOp>() &&
      AnyTwoOperandsWhere(n, [&query_engine](Node* node) {
        return query_engine.IsFullyKnown(node);
      })) {
    std::vector<Node*> new_operands;
    Bits bits = LogicalOpIdentity(n->op(), n->BitCountOrDie());
    for (Node* operand : n->operands()) {
      if (query_engine.IsFullyKnown(operand)) {
        bits = DoLogicalOp(n->op(),
                           {bits, *query_engine.KnownValueAsBits(operand)});
      } else {
        new_operands.push_back(operand);
      }
    }
    XLS_ASSIGN_OR_RETURN(Node * literal, n->function_base()->MakeNode<Literal>(
                                             n->loc(), Value(bits)));
    new_operands.push_back(literal);
    VLOG(2)
        << "FOUND: fold literal operands of associative commutative operation";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<NaryOp>(new_operands, n->op()).status());
    return true;
  }

  // Pattern: Mul by 0
  if ((n->op() == Op::kSMul || n->op() == Op::kUMul) &&
      query_engine.IsAllZeros(n->operand(1))) {
    VLOG(2) << "FOUND: Mul by 0";
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(Value(UBits(0, n->BitCountOrDie())))
            .status());
    return true;
  }

  // Pattern: Not(Not(x)) => x
  if (n->op() == Op::kNot && n->operand(0)->op() == Op::kNot) {
    VLOG(2) << "FOUND: replace not(not(x)) with x";
    XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(n->operand(0)->operand(0)));
    return true;
  }

  // Pattern: Double negative.
  //   -(-expr)
  if (n->op() == Op::kNeg && n->operand(0)->op() == Op::kNeg) {
    VLOG(2) << "FOUND: Double negative";
    XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(n->operand(0)->operand(0)));
    return true;
  }

  // Patterns (where x is a bits[1] type):
  //   eq(x, 1) => x
  //   eq(x, 0) => not(x)
  //
  // Because eq is commutative, we can rely on the literal being on the right
  // because of canonicalization.
  if (n->op() == Op::kEq && n->operand(0)->GetType()->IsBits() &&
      n->operand(0)->BitCountOrDie() == 1 &&
      query_engine.IsFullyKnown(n->operand(1))) {
    if (query_engine.IsAllZeros(n->operand(1))) {
      XLS_RETURN_IF_ERROR(
          n->ReplaceUsesWithNew<UnOp>(n->operand(0), Op::kNot).status());
      return true;
    }
    VLOG(2) << "FOUND: eq comparison with bits[1]:0 or bits[1]:1";
    XLS_RET_CHECK(query_engine.KnownValueAsBits(n->operand(1))->IsOne());
    XLS_RETURN_IF_ERROR(n->ReplaceUsesWith(n->operand(0)));
    return true;
  }

  // eq(x, x) => 1 (or other reflexive relations such as ule)
  // ne(x, x) => 0 (or other irreflexive relations such as ult)
  if (IsBinaryReflexiveRelation(n) && n->operand(0) == n->operand(1)) {
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(Value(UBits(1, 1))).status());
    return true;
  }
  if (IsBinaryIrreflexiveRelation(n) && n->operand(0) == n->operand(1)) {
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(Value(UBits(0, 1))).status());
    return true;
  }

  // If x and y are zero-width values:
  //    eq(x, y) => 1
  //    ne(x, y) => 0
  if (n->op() == Op::kEq && n->operand(0)->GetType()->GetFlatBitCount() == 0) {
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(Value(UBits(1, 1))).status());
    return true;
  }
  if (n->op() == Op::kNe && n->operand(0)->GetType()->GetFlatBitCount() == 0) {
    XLS_RETURN_IF_ERROR(
        n->ReplaceUsesWithNew<Literal>(Value(UBits(0, 1))).status());
    return true;
  }

  return false;
}

}  // namespace

absl::StatusOr<bool> BasicSimplificationPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext* context) const {
  return TransformNodesToFixedPoint(f, MatchPatterns);
}

REGISTER_OPT_PASS(BasicSimplificationPass);

}  // namespace xls
