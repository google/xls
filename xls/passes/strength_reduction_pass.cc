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

#include "xls/passes/strength_reduction_pass.h"

#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/ternary_query_engine.h"

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

  if (NarrowingEnabled(opt_level) && !node->Is<Literal>() &&
      node->GetType()->IsBits() && query_engine.AllBitsKnown(node)) {
    XLS_VLOG(2) << "Replacing node with its (entirely known) bits: " << node
                << " as " << ToString(query_engine.GetTernary(node).Get({}));
    XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<Literal>(
                                Value(ternary_ops::ToKnownBitsValues(
                                    query_engine.GetTernary(node).Get({}))))
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
    if (IsLiteralWithRunOfSetBits(node->operand(1), leading_zeros,
                                  selected_bits, trailing_zeros)) {
      return true;
    }
    if (query_engine.AllBitsKnown(node->operand(1)) &&
        ternary_ops::ToKnownBitsValues(
            query_engine.GetTernary(node->operand(1)).Get({}))
            .HasSingleRunOfSetBits(leading_zeros, selected_bits,
                                   trailing_zeros)) {
      return true;
    }
    return false;
  };
  if (NarrowingEnabled(opt_level) &&
      is_bitslice_and(&leading_zeros, &selected_bits, &trailing_zeros)) {
    XLS_CHECK_GE(leading_zeros, 0);
    XLS_CHECK_GE(selected_bits, 0);
    XLS_CHECK_GE(trailing_zeros, 0);
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
    XLS_RET_CHECK(!select->default_value().has_value()) << select->ToString();
    Node* s = select->operand(0);
    Node* on_false = select->get_case(0);
    Node* on_true = select->get_case(1);
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
    if (IsLiteralAllOnes(node->operand(2)) && IsLiteralZero(node->operand(1))) {
      *invert_selector = false;
      return true;
    }
    if (IsLiteralAllOnes(node->operand(1)) && IsLiteralZero(node->operand(2))) {
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
      query_engine.GetKnownMsb(node->operand(0)) == 0) {
    XLS_RETURN_IF_ERROR(
        node->ReplaceUsesWithNew<ExtendOp>(node->operand(0),
                                           node->BitCountOrDie(), Op::kZeroExt)
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
      node->operand(1)->Is<Literal>()) {
    const Bits& op1_literal_bits =
      node->operand(1)->As<Literal>()->value().bits();
    if (op1_literal_bits.IsPowerOfTwo()) {
      int64_t one_position = op1_literal_bits.bit_count() -
                             op1_literal_bits.CountLeadingZeros() - 1;
      int64_t width = op1_literal_bits.bit_count() - one_position;
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
      IsLiteralZero(node->operand(1))) {
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

  return false;
}

}  // namespace

absl::StatusOr<bool> StrengthReductionPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const PassOptions& options, PassResults* results) const {
  TernaryQueryEngine query_engine;
  XLS_RETURN_IF_ERROR(query_engine.Populate(f).status());
  XLS_ASSIGN_OR_RETURN(absl::flat_hash_set<Node*> reducible_adds,
                       FindReducibleAdds(f, query_engine));
  // Note: because we introduce new nodes into the graph that were not present
  // for the original QueryEngine analysis, we must be careful to guard our
  // bit value tests with "IsKnown" sorts of calls.
  //
  // TODO(leary): 2019-09-05: We can eventually implement incremental
  // recomputation of the bit tracking data for newly introduced nodes so the
  // information is always fresh and precise.
  bool modified = false;
  for (Node* node : TopoSort(f)) {
    XLS_ASSIGN_OR_RETURN(
        bool node_modified,
        StrengthReduceNode(node, reducible_adds, query_engine, opt_level_));
    modified |= node_modified;
  }
  return modified;
}

}  // namespace xls
