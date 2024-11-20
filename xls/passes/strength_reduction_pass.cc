// Copyright 2020 The XLS Authors//
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

#include "xls/passes/strength_reduction_pass.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/ternary.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"
#include "xls/passes/ternary_query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {
namespace {

// Finds and returns the set of adds which may be safely strength-reduced to
// ORs. These are determined ahead of time rather than being transformed inline
// to avoid problems with stale information in QueryEngine.
absl::StatusOr<absl::flat_hash_set<Node*>> FindReducibleAdds(
    FunctionBase* f, const QueryEngine& query_engine) {
  absl::flat_hash_set<Node*> reducible_adds;
  for (Node* node : f->nodes()) {
    // An add can be reduced to an OR if there is at least one zero in every bit
    // position amongst the operands of the add.
    if (node->op() == Op::kAdd) {
      bool reducible = true;
      for (int64_t i = 0; i < node->BitCountOrDie(); ++i) {
        if (!query_engine.IsZero(TreeBitLocation(node->operand(0), i)) &&
            !query_engine.IsZero(TreeBitLocation(node->operand(1), i))) {
          reducible = false;
          break;
        }
      }
      if (reducible) {
        reducible_adds.insert(node);
      }
    }
  }
  return std::move(reducible_adds);
}

absl::StatusOr<bool> MaybeSinkOperationIntoSelect(
    Node* node, const QueryEngine& query_engine, Select* select_val) {
  if (OpIsSideEffecting(node->op())) {
    // Side-effecting operations are not always safe to duplicate so don't
    // bother.
    return false;
  }
  DCHECK(!query_engine.IsFullyKnown(select_val));
  DCHECK(select_val->AllCases(
      [&](Node* c) { return query_engine.IsFullyKnown(c); }));

  auto operands = node->operands();
  int64_t argument_idx =
      std::distance(operands.begin(), absl::c_find(operands, select_val));
  XLS_RET_CHECK_NE(argument_idx, operands.size())
      << select_val->ToString() << " is not an argument of " << node;
  // We need both an unknown select and all other operands to be fully known.
  bool non_select_operands_are_constant = absl::c_all_of(
      operands,
      [&](Node* n) { return n == select_val || query_engine.IsFullyKnown(n); });
  // We don't want to make the select mux wider unless we are pretty sure that
  // benefit is worth it.
  static constexpr std::array<Op, 14> kExpensiveOps{
      Op::kAdd, Op::kSub, Op::kSDiv, Op::kUDiv, Op::kUMod, Op::kSMod, Op::kUMul,
      Op::kSMul, Op::kUMulp, Op::kSMulp, Op::kShll, Op::kShra, Op::kShrl,
      // Encode of a non-constant is quite slow.
      Op::kEncode};
  bool sink_would_improve_ir = node->GetType()->GetFlatBitCount() <=
                                   select_val->GetType()->GetFlatBitCount() ||
                               node->OpIn(kExpensiveOps);
  if (non_select_operands_are_constant && sink_would_improve_ir) {
    std::vector<Node*> new_cases;
    new_cases.reserve(select_val->cases().size());
    std::optional<Node*> new_default;
    std::vector<Node*> ops_vec(operands.cbegin(), operands.cend());
    auto sink_operation = [&](Node* argument) {
      ops_vec[argument_idx] = argument;
      return node->Clone(ops_vec);
    };
    VLOG(2) << "Sinking " << node << " into its select argument " << select_val;
    if (select_val->default_value()) {
      XLS_ASSIGN_OR_RETURN(new_default,
                           sink_operation(*select_val->default_value()));
      VLOG(2) << "    default for new select is " << *new_default;
    }
    for (Node* c : select_val->cases()) {
      XLS_ASSIGN_OR_RETURN(Node * new_case, sink_operation(c));
      new_cases.push_back(new_case);
      VLOG(2) << "    case for new select is " << new_case;
    }
    XLS_ASSIGN_OR_RETURN(Node * new_sel,
                         node->ReplaceUsesWithNew<Select>(
                             select_val->selector(), new_cases, new_default));
    VLOG(2) << "    new select is " << new_sel;
    return true;
  }
  return false;
}

// Attempts to strength-reduce the given node. Returns true if successful.
// 'reducible_adds' is the set of add operations which may be safely replaced
// with an OR.
absl::StatusOr<bool> StrengthReduceNode(
    Node* node, const absl::flat_hash_set<Node*>& reducible_adds,
    const QueryEngine& query_engine, int64_t opt_level) {
  if (!std::all_of(node->operands().begin(), node->operands().end(),
                   [](Node* n) { return n->GetType()->IsBits(); }) ||
      !node->GetType()->IsBits()) {
    return false;
  }

  if (NarrowingEnabled(opt_level) &&
      // Don't replace unused nodes. We don't want to add nodes when they will
      // get DCEd later. This can lead to an infinite loop between strength
      // reduction and DCE.
      !node->users().empty() && !node->Is<Literal>() &&
      query_engine.IsFullyKnown(node)) {
    VLOG(2) << "Replacing node with its (entirely known) value: " << node
            << " as " << query_engine.KnownValue(node)->ToString();
    XLS_RETURN_IF_ERROR(
        node->ReplaceUsesWithNew<Literal>(*query_engine.KnownValue(node))
            .status());
    return true;
  }

  if (reducible_adds.contains(node)) {
    XLS_RET_CHECK_EQ(node->op(), Op::kAdd);
    XLS_RETURN_IF_ERROR(
        node->ReplaceUsesWithNew<NaryOp>(
                std::vector<Node*>{node->operand(0), node->operand(1)}, Op::kOr)
            .status());
    return true;
  }

  // And(x, mask) => Concat(0, Slice(x), 0)
  //
  // Note that we only do this if the mask is a single run of set bits, to avoid
  // putting too many nodes in the graph (e.g. for a 128-bit value where every
  // other bit was set).
  int64_t leading_zeros, selected_bits, trailing_zeros;
  auto is_bitslice_and = [&](int64_t* leading_zeros, int64_t* selected_bits,
                             int64_t* trailing_zeros) -> bool {
    if (node->op() != Op::kAnd || node->operand_count() != 2) {
      return false;
    }
    if (std::optional<Bits> mask =
            query_engine.KnownValueAsBits(node->operand(1));
        mask.has_value() && mask->HasSingleRunOfSetBits(
                                leading_zeros, selected_bits, trailing_zeros)) {
      return true;
    }
    return false;
  };
  if (NarrowingEnabled(opt_level) &&
      is_bitslice_and(&leading_zeros, &selected_bits, &trailing_zeros)) {
    CHECK_GE(leading_zeros, 0);
    CHECK_GE(selected_bits, 0);
    CHECK_GE(trailing_zeros, 0);
    FunctionBase* f = node->function_base();
    XLS_ASSIGN_OR_RETURN(Node * slice,
                         f->MakeNode<BitSlice>(node->loc(), node->operand(0),
                                               /*start=*/trailing_zeros,
                                               /*width=*/selected_bits));
    XLS_ASSIGN_OR_RETURN(
        Node * leading,
        f->MakeNode<Literal>(node->loc(), Value(UBits(0, leading_zeros))));
    XLS_ASSIGN_OR_RETURN(
        Node * trailing,
        f->MakeNode<Literal>(node->loc(), Value(UBits(0, trailing_zeros))));
    XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<Concat>(
                                std::vector<Node*>{leading, slice, trailing})
                            .status());
    return true;
  }

  // We explode single-bit muxes into their constituent gates to expose more
  // optimization opportunities. Since this creates more ops in the general
  // case, we look for certain sub-cases:
  //
  // * At least one of the selected values is a literal.
  // * One of the selected values is also the selector.
  //
  // TODO(meheff): Handle one-hot select here as well.
  auto is_one_bit_mux = [&] {
    return node->Is<Select>() && node->GetType()->IsBits() &&
           node->BitCountOrDie() == 1 && node->operand(0)->BitCountOrDie() == 1;
  };
  if (SplitsEnabled(opt_level) && is_one_bit_mux() &&
      (node->operand(1)->Is<Literal>() || node->operand(2)->Is<Literal>() ||
       (node->operand(0) == node->operand(1) ||
        node->operand(0) == node->operand(2)))) {
    FunctionBase* f = node->function_base();
    Select* select = node->As<Select>();
    Node* s = select->operand(0);
    Node* on_false = select->get_case(0);
    Node* on_true = select->default_value().has_value()
                        ? *select->default_value()
                        : select->get_case(1);
    XLS_ASSIGN_OR_RETURN(
        Node * lhs,
        f->MakeNode<NaryOp>(select->loc(), std::vector<Node*>{s, on_true},
                            Op::kAnd));
    XLS_ASSIGN_OR_RETURN(Node * s_not,
                         f->MakeNode<UnOp>(select->loc(), s, Op::kNot));
    XLS_ASSIGN_OR_RETURN(
        Node * rhs,
        f->MakeNode<NaryOp>(select->loc(), std::vector<Node*>{s_not, on_false},
                            Op::kAnd));
    XLS_RETURN_IF_ERROR(
        select
            ->ReplaceUsesWithNew<NaryOp>(std::vector<Node*>{lhs, rhs}, Op::kOr)
            .status());
    return true;
  }

  // Detects whether an operation is a select that effectively acts like a sign
  // extension (or a invert-then-sign-extension); i.e. if the selector is one
  // yields all ones when the selector is 1 and all zeros when the selector is
  // 0.
  auto is_signext_mux = [&](bool* invert_selector) {
    bool ok = node->op() == Op::kSel && node->GetType()->IsBits() &&
              node->operand(0)->BitCountOrDie() == 1;
    if (!ok) {
      return false;
    }
    if (query_engine.IsAllOnes(node->operand(2)) &&
        query_engine.IsAllZeros(node->operand(1))) {
      *invert_selector = false;
      return true;
    }
    if (query_engine.IsAllOnes(node->operand(1)) &&
        query_engine.IsAllZeros(node->operand(2))) {
      *invert_selector = true;
      return true;
    }
    return false;
  };
  bool invert_selector;
  if (is_signext_mux(&invert_selector)) {
    Node* selector = node->operand(0);
    if (invert_selector) {
      XLS_ASSIGN_OR_RETURN(selector, node->function_base()->MakeNode<UnOp>(
                                         node->loc(), selector, Op::kNot));
    }
    XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<ExtendOp>(
                                selector, node->BitCountOrDie(), Op::kSignExt)
                            .status());
    return true;
  }

  // If we know the MSb of the operand is zero, strength reduce from signext to
  // zeroext.
  if (node->op() == Op::kSignExt && query_engine.IsMsbKnown(node->operand(0)) &&
      query_engine.GetKnownMsb(node->operand(0)) == false) {
    XLS_RETURN_IF_ERROR(
        node->ReplaceUsesWithNew<ExtendOp>(node->operand(0),
                                           node->BitCountOrDie(), Op::kZeroExt)
            .status());
    return true;
  }

  // If we know a Gate op is unconditionally on or off, strength reduce to
  // either a literal zero or the data value as appropriate.
  if (node->Is<Gate>() &&
      query_engine.AllBitsKnown(node->As<Gate>()->condition())) {
    Gate* gate = node->As<Gate>();
    if (query_engine.IsAllOnes(gate->condition())) {
      XLS_RETURN_IF_ERROR(gate->ReplaceUsesWith(gate->data()));
    } else {
      XLS_RETURN_IF_ERROR(
          gate->ReplaceUsesWithNew<Literal>(
                  Value(UBits(0, gate->GetType()->GetFlatBitCount())))
              .status());
    }
    return true;
  }

  // If the gate results in a known zero regardless of the condition value we
  // can remove it.
  if (node->Is<Gate>() && query_engine.IsAllZeros(node->As<Gate>()->data())) {
    Gate* gate = node->As<Gate>();
    XLS_RETURN_IF_ERROR(
        gate->ReplaceUsesWithNew<Literal>(
                Value(UBits(0, gate->GetType()->GetFlatBitCount())))
            .status());
    return true;
  }

  // Single bit add and ne are xor.
  //
  // Truth table for both ne and add (xor):
  //          y
  //        0   1
  //       -------
  //    0 | 0   1
  //  x 1 | 1   0
  if ((node->op() == Op::kAdd || node->op() == Op::kNe) &&
      node->operand(0)->BitCountOrDie() == 1) {
    XLS_RETURN_IF_ERROR(
        node->ReplaceUsesWithNew<NaryOp>(
                std::vector<Node*>{node->operand(0), node->operand(1)},
                Op::kXor)
            .status());
    return true;
  }

  // A test like x >= const, with const being a power of 2 and
  // x having a bitwidth of log2(const), can be converted
  // to a simple bit test, eg.:
  //   x:10 >= 512:10  ->  bit_slice(x, 9, 1) == 1  or
  //   x:10 <  512:10  ->  bit_slice(x, 9, 1) == 0
  //
  // In the more general case, with const being 'any' power of 2,
  // one can still strength reduce this to a comparison of only the
  // leading bits, but please note the comparison operators. Eg.:
  //   x:10 >= 256:10  ->  bit_slice(x, 9, 2) != 0b00  or
  //   x:10 <  256:10  ->  bit_slice(x, 9, 2) == 0b00
  if (NarrowingEnabled(opt_level) &&
      (node->op() == Op::kUGe || node->op() == Op::kULt) &&
      query_engine.IsFullyKnown(node->operand(1))) {
    const Bits op1_bits = *query_engine.KnownValueAsBits(node->operand(1));
    if (op1_bits.IsPowerOfTwo()) {
      int64_t one_position =
          op1_bits.bit_count() - op1_bits.CountLeadingZeros() - 1;
      int64_t width = op1_bits.bit_count() - one_position;
      Op new_op = node->op() == Op::kUGe ? Op::kNe : Op::kEq;
      XLS_ASSIGN_OR_RETURN(Node * slice,
                           node->function_base()->MakeNode<BitSlice>(
                               node->loc(), node->operand(0),
                               /*start=*/one_position,
                               /*width=*/width));
      XLS_ASSIGN_OR_RETURN(Node * zero,
                           node->function_base()->MakeNode<Literal>(
                               node->loc(), Value(UBits(/*value=*/0,
                                                        /*bit_count=*/width))));
      XLS_RETURN_IF_ERROR(
          node->ReplaceUsesWithNew<CompareOp>(slice, zero, new_op).status());
      return true;
    }
  }

  // Eq(x, 0b00) => x_0 == 0 & x_1 == 0 => ~x_0 & ~x_1 => ~(x_0 | x_1)
  //  where bits(x) <= 2
  if (NarrowingEnabled(opt_level) && node->op() == Op::kEq &&
      node->operand(0)->BitCountOrDie() == 2 &&
      query_engine.IsAllZeros(node->operand(1))) {
    FunctionBase* f = node->function_base();
    XLS_ASSIGN_OR_RETURN(
        Node * x_0, f->MakeNode<BitSlice>(node->loc(), node->operand(0), 0, 1));
    XLS_ASSIGN_OR_RETURN(
        Node * x_1, f->MakeNode<BitSlice>(node->loc(), node->operand(0), 1, 1));
    XLS_ASSIGN_OR_RETURN(
        NaryOp * nary_or,
        f->MakeNode<NaryOp>(node->loc(), std::vector<Node*>{x_0, x_1},
                            Op::kOr));
    XLS_RETURN_IF_ERROR(
        node->ReplaceUsesWithNew<UnOp>(nary_or, Op::kNot).status());
    return true;
  }

  // If a string of least-significant bits of an operand of an add is zero the
  // add can be narrowed.
  if (SplitsEnabled(opt_level) && node->op() == Op::kAdd) {
    auto lsb_known_zero_count = [&](Node* n) {
      for (int64_t i = 0; i < n->BitCountOrDie(); ++i) {
        if (!query_engine.IsZero(TreeBitLocation(n, i))) {
          return i;
        }
      }
      return n->BitCountOrDie();
    };
    int64_t op0_known_zero = lsb_known_zero_count(node->operand(0));
    int64_t op1_known_zero = lsb_known_zero_count(node->operand(1));
    if (op0_known_zero > 0 || op1_known_zero > 0) {
      Node* nonzero_operand =
          op0_known_zero > op1_known_zero ? node->operand(1) : node->operand(0);
      int64_t narrow_amt = std::max(op0_known_zero, op1_known_zero);
      auto narrow = [&](Node* n) -> absl::StatusOr<Node*> {
        return node->function_base()->MakeNode<BitSlice>(
            node->loc(), n, /*start=*/narrow_amt,
            /*width=*/n->BitCountOrDie() - narrow_amt);
      };
      XLS_ASSIGN_OR_RETURN(Node * op0_narrowed, narrow(node->operand(0)));
      XLS_ASSIGN_OR_RETURN(Node * op1_narrowed, narrow(node->operand(1)));
      XLS_ASSIGN_OR_RETURN(
          Node * narrowed_add,
          node->function_base()->MakeNode<BinOp>(node->loc(), op0_narrowed,
                                                 op1_narrowed, Op::kAdd));
      XLS_ASSIGN_OR_RETURN(Node * lsb,
                           node->function_base()->MakeNode<BitSlice>(
                               node->loc(), nonzero_operand,
                               /*start=*/0, /*width=*/narrow_amt));
      XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<Concat>(
                                  std::vector<Node*>{narrowed_add, lsb})
                              .status());
      return true;
    }
  }

  // Transform arithmetic operation with exactly one unknown-bit in all of its
  // operands into a select on that one unknown bit.
  constexpr std::array<Op, 6> kExpensiveArithOps = {
      Op::kSMul, Op::kUMul, Op::kSDiv, Op::kUDiv, Op::kSMod, Op::kUMod,
  };
  if (NarrowingEnabled(opt_level) && node->OpIn(kExpensiveArithOps) &&
      query_engine.IsTracked(node->operand(0)) &&
      query_engine.IsTracked(node->operand(1))) {
    Node* left = node->operand(0);
    Node* right = node->operand(1);
    std::optional<LeafTypeTree<TernaryVector>> left_ternary =
        query_engine.GetTernary(left);
    std::optional<LeafTypeTree<TernaryVector>> right_ternary =
        query_engine.GetTernary(right);
    int64_t left_unknown_count =
        left_ternary.has_value()
            ? absl::c_count(left_ternary->Get({}), TernaryValue::kUnknown)
            : left->BitCountOrDie();
    int64_t right_unknown_count =
        right_ternary.has_value()
            ? absl::c_count(right_ternary->Get({}), TernaryValue::kUnknown)
            : right->BitCountOrDie();
    Node* unknown_operand = left_unknown_count == 0 ? right : left;
    auto replace_with_select = [&](Node* variable, const Bits& value,
                                   const Value& true_result,
                                   const Value& false_result) -> absl::Status {
      XLS_ASSIGN_OR_RETURN(
          Node * compare_lit,
          node->function_base()->MakeNodeWithName<Literal>(
              node->loc(), Value(value),
              absl::StrFormat("%s_possible_value", variable->GetName())));
      XLS_ASSIGN_OR_RETURN(Node * eq,
                           node->function_base()->MakeNodeWithName<CompareOp>(
                               node->loc(), variable, compare_lit, Op::kEq,
                               absl::StrFormat("%s_compare", node->GetName())));
      XLS_ASSIGN_OR_RETURN(
          Node * true_node,
          node->function_base()->MakeNodeWithName<Literal>(
              node->loc(), Value(true_result),
              absl::StrFormat("%s_result_value_true", node->GetName())));
      XLS_ASSIGN_OR_RETURN(
          Node * false_node,
          node->function_base()->MakeNodeWithName<Literal>(
              node->loc(), Value(false_result),
              absl::StrFormat("%s_result_value_false", node->GetName())));
      return node
          ->ReplaceUsesWithNew<Select>(
              eq, absl::Span<Node* const>{false_node, true_node}, std::nullopt)
          .status();
    };

    // TODO(allight): It might be good to do this with more unknown bits in some
    // cases (eg 200 bit mul with -> 8 branch select).
    if (left_unknown_count + right_unknown_count == 1) {
      const std::optional<LeafTypeTree<TernaryVector>>& known_ternary =
          left_unknown_count == 0 ? left_ternary : right_ternary;
      const std::optional<LeafTypeTree<TernaryVector>>& unknown_ternary =
          left_unknown_count == 0 ? right_ternary : left_ternary;
      Value known_value =
          Value(ternary_ops::ToKnownBitsValues(known_ternary->Get({})));
      TernaryVector unknown_value =
          unknown_ternary.has_value()
              ? unknown_ternary->Get({})
              : TernaryVector(unknown_operand->BitCountOrDie(),
                              TernaryValue::kUnknown);
      TernaryVector zero_vec(unknown_value);
      TernaryVector one_vec(std::move(unknown_value));
      // Set the single unknown to zero/one respectively.
      absl::c_replace(zero_vec, TernaryValue::kUnknown,
                      TernaryValue::kKnownZero);
      absl::c_replace(one_vec, TernaryValue::kUnknown, TernaryValue::kKnownOne);
      Value zero_value = Value(ternary_ops::ToKnownBitsValues(zero_vec));
      Value one_value = Value(ternary_ops::ToKnownBitsValues(one_vec));
      // Interpret the node, makes sure to pass in the right order to deal with
      // non-commutative ops like mod and div.
      auto get_real_result =
          [&](const Value& materialized_value) -> absl::StatusOr<Value> {
        if (left_unknown_count != 0) {
          // Unknown value is on the left.
          return InterpretNode(node, {materialized_value, known_value});
        }
        // Unknown value is on the right.
        return InterpretNode(node, {known_value, materialized_value});
      };
      XLS_ASSIGN_OR_RETURN(Value zero_result, get_real_result(zero_value));
      XLS_ASSIGN_OR_RETURN(Value one_result, get_real_result(one_value));
      XLS_RETURN_IF_ERROR(replace_with_select(
          unknown_operand, zero_value.bits(), zero_result, one_result));
      return true;
    }
  }

  // Sink operations into selects in some circumstances.
  //
  // If we have an operation where all operands except for one select are
  // known-constant and the select has a constant value in each of its cases we
  // can move the operation into the select in order to allow other
  // narrowing/constant propagation passes to calculate the resulting value.
  //
  // v1 := Select(<variable>, [<const1>, <const2>])
  // v2 := Op(v1, <const3>)
  //
  // Transforms to
  //
  // v1_c1 := Op(<const1>, <const3>)
  // v1_c2 := Op(<const2>, <const3>)
  // v2 := Select(<variable>, [v1_c1, v1_c2])
  if (!query_engine.IsFullyKnown(node)) {
    auto operands = node->operands();
    // Find a non-fully-known select
    auto is_select_with_known_branches_unknown_selector = [&](Node* n) {
      return n->Is<Select>() && !query_engine.IsFullyKnown(n) &&
             n->As<Select>()->AllCases(
                 [&](Node* c) -> bool { return query_engine.IsFullyKnown(c); });
    };
    auto select_it = absl::c_find_if(
        operands, is_select_with_known_branches_unknown_selector);
    // We need both an unknown select and all other operands to be fully known.
    if (select_it != operands.end()) {
      XLS_ASSIGN_OR_RETURN(bool changed,
                           MaybeSinkOperationIntoSelect(
                               node, query_engine, (*select_it)->As<Select>()));
      if (changed) {
        return true;
      }
    }
  }

  return false;
}

}  // namespace

absl::StatusOr<bool> StrengthReductionPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results) const {
  std::vector<std::unique_ptr<QueryEngine>> query_engines;
  query_engines.push_back(std::make_unique<StatelessQueryEngine>());
  query_engines.push_back(std::make_unique<TernaryQueryEngine>());

  UnionQueryEngine query_engine(std::move(query_engines));
  XLS_RETURN_IF_ERROR(query_engine.Populate(f).status());

  XLS_ASSIGN_OR_RETURN(absl::flat_hash_set<Node*> reducible_adds,
                       FindReducibleAdds(f, query_engine));
  // Note: because we introduce new nodes into the graph that were not present
  // for the original QueryEngine analysis, we may get less effective
  // optimizations for these new nodes due to a lack of data.
  //
  // TODO(leary): 2019-09-05: We can eventually implement incremental
  // recomputation of the bit tracking data for newly introduced nodes so the
  // information is always fresh and precise.
  bool modified = false;
  for (Node* node : TopoSort(f)) {
    XLS_ASSIGN_OR_RETURN(bool node_modified,
                         StrengthReduceNode(node, reducible_adds, query_engine,
                                            options.opt_level));
    modified |= node_modified;
  }
  return modified;
}

REGISTER_OPT_PASS(StrengthReductionPass);

}  // namespace xls
