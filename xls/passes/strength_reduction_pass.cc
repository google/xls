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

#include "xls/passes/strength_reduction_pass.h"

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
xabsl::StatusOr<absl::flat_hash_set<Node*>> FindReducibleAdds(
    Function* f, const QueryEngine& query_engine) {
  absl::flat_hash_set<Node*> reducible_adds;
  for (Node* node : f->nodes()) {
    // An add can be reduced to an OR if there is at least one zero in every bit
    // position amongst the operands of the add.
    if (node->op() == OP_ADD) {
      bool reducible = true;
      for (int64 i = 0; i < node->BitCountOrDie(); ++i) {
        if (!query_engine.IsZero(BitLocation(node->operand(0), i)) &&
            !query_engine.IsZero(BitLocation(node->operand(1), i))) {
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
xabsl::StatusOr<bool> StrengthReduceNode(
    Node* node, const absl::flat_hash_set<Node*>& reducible_adds,
    const QueryEngine& query_engine, bool split_ops) {
  if (!node->Is<Literal>() && node->GetType()->IsBits() &&
      query_engine.AllBitsKnown(node)) {
    XLS_VLOG(1) << "Replacing node with its (entirely known) bits: " << node
                << " as "
                << query_engine.GetKnownBitsValues(node).ToString(
                       FormatPreference::kBinary);
    XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<Literal>(
                                Value(query_engine.GetKnownBitsValues(node)))
                            .status());
    return true;
  }

  if (reducible_adds.contains(node)) {
    XLS_RET_CHECK_EQ(node->op(), OP_ADD);
    XLS_RETURN_IF_ERROR(
        node->ReplaceUsesWithNew<NaryOp>(
                std::vector<Node*>{node->operand(0), node->operand(1)}, OP_OR)
            .status());
    return true;
  }

  // Replace a multiplication by a power of two with a shift left. We rely on
  // canonicalization to place the literal on the RHS.
  // TODO(meheff): 2019/8/6 There are many other possibilities of replacing
  // multiplication by a constant with shifts and adds.
  if ((node->op() == OP_SMUL || node->op() == OP_UMUL) &&
      node->operand(1)->Is<Literal>() &&
      node->BitCountOrDie() >= node->operand(0)->BitCountOrDie()) {
    const Bits& multiplicand = node->operand(1)->As<Literal>()->value().bits();
    // The multiplicand needs to be a power of two. IsPowerOfTwo treats its
    // operand as an unsigned number, so if the operation is a SMul verify that
    // the multiplicand is non-negative (sign bit is zero).
    if (multiplicand.IsPowerOfTwo() &&
        (node->op() == OP_UMUL || !multiplicand.msb())) {
      const int64 result_bit_count = node->BitCountOrDie();
      Node* to_shift = node->operand(0);
      if (result_bit_count > node->operand(0)->BitCountOrDie()) {
        // Sign/Zero extend the lhs to match the result width.
        XLS_ASSIGN_OR_RETURN(
            to_shift,
            node->function()->MakeNode<ExtendOp>(
                node->loc(), node->operand(0),
                /*new_bit_count=*/result_bit_count,
                /*op=*/node->op() == OP_SMUL ? OP_SIGN_EXT : OP_ZERO_EXT));
      }
      const int64 shift_amount =
          multiplicand.bit_count() - multiplicand.CountLeadingZeros() - 1;
      XLS_ASSIGN_OR_RETURN(
          Node * literal,
          node->function()->MakeNode<Literal>(
              node->loc(), Value(UBits(shift_amount, result_bit_count))));
      XLS_RETURN_IF_ERROR(
          node->ReplaceUsesWithNew<BinOp>(to_shift, literal, OP_SHLL)
              .status());
      return true;
    }
  }

  // And(x, mask) => Concat(0, Slice(x), 0)
  //
  // Note that we only do this if the mask is a single run of set bits, to avoid
  // putting too many nodes in the graph (e.g. for a 128-bit value where every
  // other bit was set).
  int64 leading_zeros, selected_bits, trailing_zeros;
  auto is_bitslice_and = [&](int64* leading_zeros, int64* selected_bits,
                             int64* trailing_zeros) -> bool {
    if (node->op() != OP_AND || node->operand_count() != 2) {
      return false;
    }
    if (IsLiteralWithRunOfSetBits(node->operand(1), leading_zeros,
                                  selected_bits, trailing_zeros)) {
      return true;
    }
    if (query_engine.AllBitsKnown(node->operand(1)) &&
        query_engine.GetKnownBitsValues(node->operand(1))
            .HasSingleRunOfSetBits(leading_zeros, selected_bits,
                                   trailing_zeros)) {
      return true;
    }
    return false;
  };
  if (is_bitslice_and(&leading_zeros, &selected_bits, &trailing_zeros)) {
    XLS_CHECK_GE(leading_zeros, 0);
    XLS_CHECK_GE(selected_bits, 0);
    XLS_CHECK_GE(trailing_zeros, 0);
    Function* f = node->function();
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
  if (split_ops && is_one_bit_mux() &&
      (node->operand(1)->Is<Literal>() || node->operand(2)->Is<Literal>() ||
       (node->operand(0) == node->operand(1) ||
        node->operand(0) == node->operand(2)))) {
    Function* f = node->function();
    Select* select = node->As<Select>();
    XLS_RET_CHECK(!select->default_value().has_value()) << select->ToString();
    Node* s = select->operand(0);
    Node* on_false = select->cases()[0];
    Node* on_true = select->cases()[1];
    XLS_ASSIGN_OR_RETURN(
        Node * lhs,
        f->MakeNode<NaryOp>(select->loc(), std::vector<Node*>{s, on_true},
                            OP_AND));
    XLS_ASSIGN_OR_RETURN(Node * s_not,
                         f->MakeNode<UnOp>(select->loc(), s, OP_NOT));
    XLS_ASSIGN_OR_RETURN(
        Node * rhs,
        f->MakeNode<NaryOp>(select->loc(), std::vector<Node*>{s_not, on_false},
                            OP_AND));
    XLS_RETURN_IF_ERROR(
        select
            ->ReplaceUsesWithNew<NaryOp>(std::vector<Node*>{lhs, rhs}, OP_OR)
            .status());
    return true;
  }

  // Detects whether an operation is a select that effectively acts like a sign
  // extension (or a invert-then-sign-extension); i.e. if the selector is one
  // yields all ones when the selector is 1 and all zeros when the selector is
  // 0.
  auto is_signext_mux = [&](bool* invert_selector) {
    bool ok = node->op() == OP_SEL && node->GetType()->IsBits() &&
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
    Function* f = node->function();
    Node* selector = node->operand(0);
    if (invert_selector) {
      XLS_ASSIGN_OR_RETURN(selector,
                           f->MakeNode<UnOp>(node->loc(), selector, OP_NOT));
    }
    XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<ExtendOp>(
                                selector, node->BitCountOrDie(), OP_SIGN_EXT)
                            .status());
    return true;
  }

  // If we know the MSb of the operand is zero, strength reduce from signext to
  // zeroext.
  if (node->op() == OP_SIGN_EXT && query_engine.IsMsbKnown(node->operand(0)) &&
      query_engine.GetKnownMsb(node->operand(0)) == 0) {
    XLS_RETURN_IF_ERROR(
        node->ReplaceUsesWithNew<ExtendOp>(node->operand(0),
                                           node->BitCountOrDie(), OP_ZERO_EXT)
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
  if ((node->op() == OP_ADD || node->op() == OP_NE) &&
      node->operand(0)->BitCountOrDie() == 1) {
    XLS_RETURN_IF_ERROR(
        node->ReplaceUsesWithNew<NaryOp>(
                std::vector<Node*>{node->operand(0), node->operand(1)},
                OP_XOR)
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
  if ((node->op() == OP_UGE || node->op() == OP_ULT) &&
      node->operand(1)->Is<Literal>()) {
    const Bits& op1_literal_bits =
      node->operand(1)->As<Literal>()->value().bits();
    if (op1_literal_bits.IsPowerOfTwo()) {
      int64 one_position = op1_literal_bits.bit_count() -
                           op1_literal_bits.CountLeadingZeros() - 1;
      int64 width = op1_literal_bits.bit_count() - one_position;
      Op new_op = node->op() == OP_UGE ? OP_NE : OP_EQ;
      XLS_ASSIGN_OR_RETURN(Node * slice, node->function()->MakeNode<BitSlice>(
                                             node->loc(), node->operand(0),
                                             /*start=*/one_position,
                                             /*width=*/width));
      XLS_ASSIGN_OR_RETURN(Node * zero,
                           node->function()->MakeNode<Literal>(
                               node->loc(), Value(UBits(/*value=*/0,
                                                        /*bit_count=*/width))));
      XLS_RETURN_IF_ERROR(
          node->ReplaceUsesWithNew<CompareOp>(slice, zero, new_op).status());
      return true;
    }
  }

  // Eq(x, 0b00) => x_0 == 0 & x_1 == 0 => ~x_0 & ~x_1 => ~(x_0 | x_1)
  //  where bits(x) <= 2
  if (node->op() == OP_EQ && node->operand(0)->BitCountOrDie() == 2 &&
      IsLiteralZero(node->operand(1))) {
    Function* f = node->function();
    XLS_ASSIGN_OR_RETURN(
        Node * x_0, f->MakeNode<BitSlice>(node->loc(), node->operand(0), 0, 1));
    XLS_ASSIGN_OR_RETURN(
        Node * x_1, f->MakeNode<BitSlice>(node->loc(), node->operand(0), 1, 1));
    XLS_ASSIGN_OR_RETURN(
        NaryOp * nary_or,
        f->MakeNode<NaryOp>(node->loc(), std::vector<Node*>{x_0, x_1},
                            OP_OR));
    XLS_RETURN_IF_ERROR(
        node->ReplaceUsesWithNew<UnOp>(nary_or, OP_NOT).status());
    return true;
  }

  // If a string of least-significant bits of an operand of an add is zero the
  // add can be narrowed.
  if (split_ops && node->op() == OP_ADD) {
    auto lsb_known_zero_count = [&](Node* n) {
      for (int64 i = 0; i < n->BitCountOrDie(); ++i) {
        if (!query_engine.IsZero(BitLocation(n, i))) {
          return i;
        }
      }
      return n->BitCountOrDie();
    };
    int64 op0_known_zero = lsb_known_zero_count(node->operand(0));
    int64 op1_known_zero = lsb_known_zero_count(node->operand(1));
    if (op0_known_zero > 0 || op1_known_zero > 0) {
      Node* nonzero_operand =
          op0_known_zero > op1_known_zero ? node->operand(1) : node->operand(0);
      int64 narrow_amt = std::max(op0_known_zero, op1_known_zero);
      auto narrow = [&](Node* n) -> xabsl::StatusOr<Node*> {
        return node->function()->MakeNode<BitSlice>(
            node->loc(), n, /*start=*/narrow_amt,
            /*width=*/n->BitCountOrDie() - narrow_amt);
      };
      XLS_ASSIGN_OR_RETURN(Node * op0_narrowed, narrow(node->operand(0)));
      XLS_ASSIGN_OR_RETURN(Node * op1_narrowed, narrow(node->operand(1)));
      XLS_ASSIGN_OR_RETURN(
          Node * narrowed_add,
          node->function()->MakeNode<BinOp>(node->loc(), op0_narrowed,
                                            op1_narrowed, OP_ADD));
      XLS_ASSIGN_OR_RETURN(Node * lsb, node->function()->MakeNode<BitSlice>(
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

xabsl::StatusOr<bool> StrengthReductionPass::RunOnFunction(
    Function* f, const PassOptions& options, PassResults* results) const {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<TernaryQueryEngine> query_engine,
                       TernaryQueryEngine::Run(f));
  XLS_ASSIGN_OR_RETURN(absl::flat_hash_set<Node*> reducible_adds,
                       FindReducibleAdds(f, *query_engine));
  // Note: because we introduce new nodes into the graph that were not present
  // for the original QueryEngine analysis, we must be careful to guard our
  // bit value tests with "IsKnown" sorts of calls.
  //
  // TODO(leary): 2019-09-05: We can eventually implement incremental
  // recomputation of the bit tracking data for newly introduced nodes so the
  // information is always fresh and precise..
  bool modified = false;
  for (Node* node : TopoSort(f)) {
    XLS_ASSIGN_OR_RETURN(
        bool node_modified,
        StrengthReduceNode(node, reducible_adds, *query_engine, split_ops_));
    modified |= node_modified;
  }
  return modified;
}

}  // namespace xls
